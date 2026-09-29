"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.5%   (on for 50% of jets)
  neuron  4:  11.3%   (on for 67% of jets)
  neuron  1:  11.2%   (on for 93% of jets)
  neuron  5:  10.8%   (on for 80% of jets)
  neuron  9:   8.1%   (on for 81% of jets)
  neuron  0:   7.8%   (on for 59% of jets)
  neuron 13:   6.7%   (on for 85% of jets)
  neuron 10:   6.3%   (on for 78% of jets)
  neuron  7:   4.9%   (on for 34% of jets)
  neuron 12:   4.1%   (on for 46% of jets)
  neuron  6:   3.8%   (on for 52% of jets)
  neuron  3:   3.7%   (on for 55% of jets)
  neuron 14:   2.6%   (on for 29% of jets)
  neuron 11:   2.6%   (on for 44% of jets)
  neuron 15:   2.0%   (on for 62% of jets)
  neuron  2:   0.6%   (on for 45% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.6% (the network: 81.1%); same class as the network for 92.1% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_6                    pT of particle 6 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft2_z                pT share of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft5_dr0              ΔR between the hardest and the 5. softest real particle (0 if among the 15 hardest)
  Q.soft7_dr0              ΔR between the hardest and the 7. softest real particle (0 if among the 15 hardest)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_8                   ΔR of particle 8 from the jet axis
  Q.soft5_dr               ΔR from the jet axis of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_phi               pT-weighted mean Δφ
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
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
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
        z_top5=sum(zs[:5]),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_4=pt[4],
        pt_6=pt[6],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        z_6=z[6],
        soft1_z=softp(1, 'z'),
        soft2_z=softp(2, 'z'),
        soft5_z=softp(5, 'z'),
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
        soft7_dr0=softp(7, 'dr0'),
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_8=dr[8] if pt[8] > 0 else 0.0,
        soft5_dr=softp(5, 'dr'),
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    # scale S = 11.25;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.25109 * (-0.1105827
        + 0.1599545 * max(0.0, 0.007872294 - Q.sum_zz_dr2) / 0.002240658   # +16.0%  sum_zz_dr2 < 0.007872
        + 0.1065599 * max(0.0, 0.006936725 - Q.sum_zz_dr2) / 0.001688614   # +10.7%  sum_zz_dr2 < 0.006937
        - 0.1004113 * max(0.0, 0.009606007 - Q.sum_zz_dr2) / 0.003497866   # -10.0%  sum_zz_dr2 < 0.009606
        + 0.08919884 * max(0.0, Q.sum_pt_top40 - 858.8262) / 166.7848   # +8.9%  sum_pt_top40 > 858.8
        - 0.07384654 * max(0.0, 0.00616708 - Q.sum_zz_dr2) / 0.001295555   # -7.4%  sum_zz_dr2 < 0.006167
        - 0.06530182 * max(0.0, Q.sum_pt - 1002.379) / 57.37467   # -6.5%  sum_pt > 1002
        + 0.05405771 * max(0.0, 0.008124776 - Q.sum_z_dr2_top50) / 0.002470567   # +5.4%  sum_z_dr2_top50 < 0.008125
        - 0.04997516 * max(0.0, 0.005852839 - Q.sum_z_dr2_top50) * max(0.0, 1260.541 - Q.sum_pt) / 0.2467156   # -5.0%  sum_z_dr2_top50 < 0.005853 and sum_pt < 1261
        + 0.03898745 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +3.9%  sum_z_dr2_top40 < 0.008841
        + 0.03858277 * max(0.0, 0.008190222 - Q.sum_z_dr2) / 0.002454223   # +3.9%  sum_z_dr2 < 0.00819
        - 0.03373411 * max(0.0, 0.006363916 - Q.sum_z_dr2_top30) / 0.001678624   # -3.4%  sum_z_dr2_top30 < 0.006364
        + 0.03345257 * max(0.0, 0.008376291 - Q.sum_z_dr2_top30) / 0.002981077   # +3.3%  sum_z_dr2_top30 < 0.008376
        + 0.02278416 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # +2.3%  psi_0p3 > 0.9943
        - 0.02020801 * max(0.0, Q.sum_pt_top40 - 1024.942) / 35.52974   # -2.0%  sum_pt_top40 > 1025
        - 0.02020269 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -2.0%  tau1 < 0.07057
        - 0.01931497 * max(0.0, 0.008190222 - Q.sum_z_dr2) * max(0.0, 0.006142802 - Q.sum_z_dr2_top15) / 1.08234e-05   # -1.9%  sum_z_dr2 < 0.00819 and sum_z_dr2_top15 < 0.006143
        - 0.01435111 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -1.4%  lam1 < 0.005914
        + 0.01370624 * max(0.0, 0.2708235 - Q.tau21_b2) / 0.06922804   # +1.4%  tau21_b2 < 0.2708
        - 0.01277701 * max(0.0, 0.006259772 - Q.sum_z_dr2_top40) / 0.001461823   # -1.3%  sum_z_dr2_top40 < 0.00626
        + 0.01191192 * max(0.0, Q.sum_pt - 1115.723) / 20.458   # +1.2%  sum_pt > 1116
        - 0.01142705 * max(0.0, 0.003113918 - Q.sum_z_dr2_top40) / 0.0004182435   # -1.1%  sum_z_dr2_top40 < 0.003114
        - 0.007322457 * max(0.0, Q.log_sum_pt - 6.910131) * max(0.0, 0.0001086251 - Q.e3) / 3.665746e-06   # -0.7%  log_sum_pt > 6.91 and e3 < 0.0001086
        - 0.001931664 * max(0.0, Q.sum_pt - 1167.447) * max(0.0, Q.soft7_dr0 - 0.2576672) / 0.274689   # -0.2%  sum_pt > 1167 and soft7_dr0 > 0.2577
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.06;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.06224 * (0.1396846
        + 0.1310286 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # +13.1%  log_sum_pt > 6.91
        - 0.118016 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # -11.8%  sum_pt_top50 > 934.2
        - 0.1059984 * max(0.0, 7.139296 - Q.log_sum_pt) / 0.2001418   # -10.6%  log_sum_pt < 7.139
        + 0.08775708 * max(0.0, Q.sum_pt - 972.0419) / 81.34075   # +8.8%  sum_pt > 972
        + 0.05276137 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +5.3%  sum_pt_top50 < 1079
        + 0.04402871 * max(0.0, 1225.842 - Q.sum_pt_top40) / 211.117   # +4.4%  sum_pt_top40 < 1226
        - 0.03967023 * max(0.0, Q.log_sum_pt - 6.98945) / 0.02114274   # -4.0%  log_sum_pt > 6.989
        + 0.03617565 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) / 0.00233708   # +3.6%  sum_z_dr2_top30 < 0.007464
        - 0.0317658 * max(0.0, Q.z_top5 - 0.4670285) / 0.1261617   # -3.2%  z_top5 > 0.467
        + 0.03163769 * max(0.0, Q.n_particles - 38.0) / 10.79928   # +3.2%  n_particles > 38
        + 0.02898634 * max(0.0, 0.2454112 - Q.LHA) / 0.03347861   # +2.9%  LHA < 0.2454
        + 0.02870364 * max(0.0, Q.z_top30_slots - 0.9564984) / 0.01947087   # +2.9%  z_top30_slots > 0.9565
        + 0.02633812 * max(0.0, Q.n_particles - 38.0) * max(0.0, Q.tau32 - 0.3293142) / 3.693221   # +2.6%  n_particles > 38 and tau32 > 0.3293
        - 0.02302417 * max(0.0, 0.009606007 - Q.sum_zz_dr2) / 0.003497866   # -2.3%  sum_zz_dr2 < 0.009606
        - 0.02137483 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.002181998 - Q.soft1_z) / 0.01445558   # -2.1%  n_particles > 38 and soft1_z < 0.002182
        - 0.01890016 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # -1.9%  lam2 < 0.001776
        - 0.01391419 * max(0.0, 0.03457336 - Q.M3) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0006031662   # -1.4%  M3 < 0.03457 and psi_0p3 > 0.9299
        - 0.01352174 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.9624339 - Q.tau43) / 1.418537   # -1.4%  n_particles > 38 and tau43 < 0.9624
        - 0.01306705 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # -1.3%  psi_0p3 > 0.9966
        - 0.012994 * max(0.0, Q.z_top30_slots - 0.9564984) * max(0.0, 10.17512 - Q.ptdr0_3) / 0.1427049   # -1.3%  z_top30_slots > 0.9565 and ptdr0_3 < 10.18
        - 0.01254408 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -1.3%  e2 < 0.02211
        - 0.01164323 * max(0.0, Q.z_top5 - 0.6551948) * max(0.0, 28.39396 - Q.pt1_dr01) / 0.8332473   # -1.2%  z_top5 > 0.6552 and pt1_dr01 < 28.39
        + 0.01114121 * max(0.0, 0.2140276 - Q.tau21) / 0.01160104   # +1.1%  tau21 < 0.214
        - 0.01081471 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.036608   # -1.1%  n_dr_0p2_0p4 < 7
        - 0.009246672 * max(0.0, Q.z_dr_0_0p05 - 0.8103116) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.2065903   # -0.9%  z_dr_0_0p05 > 0.8103 and n_dr_0p05_0p1 < 10
        + 0.008402831 * max(0.0, Q.z_top5 - 0.6551948) / 0.03331427   # +0.8%  z_top5 > 0.6552
        + 0.008233673 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.009970338 - Q.zdr_0) / 0.03317992   # +0.8%  n_particles > 38 and zdr_0 < 0.00997
        + 0.007386763 * max(0.0, 0.001153063 - Q.sum_z_dr2_top20) / 0.0001200862   # +0.7%  sum_z_dr2_top20 < 0.001153
        - 0.00676666 * max(0.0, Q.sum_pt - 1052.889) / 33.53105   # -0.7%  sum_pt > 1053
        - 0.006681535 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) / 0.9740454   # -0.7%  n_dr_0p1_0p2 < 7
        - 0.00627289 * max(0.0, Q.zdr_0 - 0.01240028) / 0.002210186   # -0.6%  zdr_0 > 0.0124
        + 0.005902556 * max(0.0, Q.n_dr_0_0p05 - 15.0) / 2.713461   # +0.6%  n_dr_0_0p05 > 15
        + 0.00427942 * max(0.0, 0.002197765 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.9966167) / 5.451807e-07   # +0.4%  sum_z_dr2_top15 < 0.002198 and psi_0p3 > 0.9966
        - 0.00425736 * max(0.0, 0.2773918 - Q.D2_b2) / 0.01604103   # -0.4%  D2_b2 < 0.2774
        + 0.003498454 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.pt1_dr01 - 5.351077) / 0.1255531   # +0.3%  z_top30_slots > 0.9342 and pt1_dr01 > 5.351
        - 0.003264242 * max(0.0, 0.001784491 - Q.sum_z_dr2) / 0.0001206749   # -0.3%  sum_z_dr2 < 0.001784
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 5.582;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.582334 * (-0.1200498
        - 0.1678698 * max(0.0, 0.006941794 - Q.sum_z_dr2) / 0.001690424   # -16.8%  sum_z_dr2 < 0.006942
        + 0.149656 * max(0.0, 0.00634935 - Q.sum_z_dr2_top50) / 0.001425305   # +15.0%  sum_z_dr2_top50 < 0.006349
        + 0.1334454 * max(0.0, 0.01563836 - Q.sum_z_dr2_top15) / 0.009593305   # +13.3%  sum_z_dr2_top15 < 0.01564
        + 0.07709879 * max(0.0, Q.log_sum_pt - 6.930088) / 0.0397436   # +7.7%  log_sum_pt > 6.93
        - 0.06166648 * max(0.0, Q.sum_pt_top50 - 1003.544) / 52.17558   # -6.2%  sum_pt_top50 > 1004
        + 0.05907749 * max(0.0, Q.sum_pt - 1007.788) / 53.71957   # +5.9%  sum_pt > 1008
        - 0.05388127 * max(0.0, Q.sum_pt_top30 - 800.732) / 195.5077   # -5.4%  sum_pt_top30 > 800.7
        - 0.04334456 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.8747961 - Q.psi_0p1) / 0.005466701   # -4.3%  log_sum_pt > 6.903 and psi_0p1 < 0.8748
        - 0.04086008 * max(0.0, 0.003638856 - Q.sum_z_dr2) / 0.0004964601   # -4.1%  sum_z_dr2 < 0.003639
        + 0.03854463 * max(0.0, Q.z_top30_slots - 0.9203881) / 0.04411614   # +3.9%  z_top30_slots > 0.9204
        + 0.03708201 * max(0.0, Q.sum_pt_top50 - 1003.544) * max(0.0, Q.z_dr_0p1_0p2 - 0.003353111) / 6.148304   # +3.7%  sum_pt_top50 > 1004 and z_dr_0p1_0p2 > 0.003353
        - 0.02700357 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -2.7%  sum_pt > 1085
        + 0.02214466 * max(0.0, Q.sum_pt_top40 - 1069.671) / 23.00896   # +2.2%  sum_pt_top40 > 1070
        - 0.02113685 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -2.1%  log_sum_pt > 7.063
        + 0.01900582 * max(0.0, Q.sum_pt_top50 - 1003.544) * max(0.0, 0.1289397 - Q.dr_4) / 4.275705   # +1.9%  sum_pt_top50 > 1004 and dr_4 < 0.1289
        - 0.01368496 * max(0.0, Q.sum_pt_top30 - 1011.524) / 32.30985   # -1.4%  sum_pt_top30 > 1012
        - 0.01124362 * max(0.0, Q.sum_pt_top20 - 993.5664) * max(0.0, 0.2319351 - Q.dr_8) / 4.18434   # -1.1%  sum_pt_top20 > 993.6 and dr_8 < 0.2319
        + 0.01104202 * max(0.0, Q.sum_pt_top20 - 993.5664) / 23.61647   # +1.1%  sum_pt_top20 > 993.6
        + 0.00629525 * max(0.0, Q.sum_pt_top30 - 1111.245) * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.9443349   # +0.6%  sum_pt_top30 > 1111 and z_dr_0p1_0p2 < 0.1203
        - 0.005916689 * max(0.0, Q.sum_pt_top15 - 1003.329) * max(0.0, Q.eta_1 - 0.08734131) / 0.006699938   # -0.6%  sum_pt_top15 > 1003 and eta_1 > 0.08734
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 3.647;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.646788 * (-0.03315866
        + 0.2046771 * max(0.0, 0.007511864 - Q.sum_zz_dr2) / 0.002016671   # +20.5%  sum_zz_dr2 < 0.007512
        - 0.1509563 * max(0.0, 0.347196 - Q.tau21) / 0.05239131   # -15.1%  tau21 < 0.3472
        - 0.1434528 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # -14.3%  LHA < 0.372
        - 0.1150137 * max(0.0, 0.006259772 - Q.sum_z_dr2_top40) / 0.001461823   # -11.5%  sum_z_dr2_top40 < 0.00626
        + 0.09597543 * max(0.0, 0.2018786 - Q.tau21_b2) / 0.03799024   # +9.6%  tau21_b2 < 0.2019
        + 0.06665602 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +6.7%  sum_z_dr2_top40 < 0.008841
        + 0.06463695 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # +6.5%  D2 < 1.788
        + 0.05591113 * max(0.0, 0.0006154841 - Q.lam2) / 0.0001499505   # +5.6%  lam2 < 0.0006155
        + 0.04630737 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 858.8262) / 1128.322   # +4.6%  n_particles < 46 and sum_pt_top40 > 858.8
        - 0.03204261 * max(0.0, 1.788105 - Q.D2) * max(0.0, 0.003582374 - Q.soft5_z) / 0.0005782761   # -3.2%  D2 < 1.788 and soft5_z < 0.003582
        + 0.02437062 * max(0.0, Q.psi_0p2 - 0.9976427) / 0.0003015943   # +2.4%  psi_0p2 > 0.9976
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 11.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.4423 * (0.04402544
        + 0.07832312 * max(0.0, Q.sum_z_dr2_top50 - 0.004573744) / 0.00533432   # +7.8%  sum_z_dr2_top50 > 0.004574
        + 0.07484453 * max(0.0, 0.009614971 - Q.lam1_plus_lam2) / 0.003502545   # +7.5%  lam1_plus_lam2 < 0.009615
        - 0.07194291 * max(0.0, Q.sum_z_dr2_top50 - 0.00242543) / 0.006946092   # -7.2%  sum_z_dr2_top50 > 0.002425
        + 0.07144208 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # +7.1%  sum_z_dr2_top20 < 0.01083
        + 0.06688655 * max(0.0, Q.sum_z_dr2_top15 - 0.002197765) / 0.005632841   # +6.7%  sum_z_dr2_top15 > 0.002198
        - 0.0637402 * max(0.0, 1061.184 - Q.sum_pt_top50) / 53.8984   # -6.4%  sum_pt_top50 < 1061
        - 0.060525 * max(0.0, 0.1011219 - Q.tau2) / 0.06797109   # -6.1%  tau2 < 0.1011
        - 0.05529379 * max(0.0, 0.007538019 - Q.sum_z_dr2_top20) / 0.002740841   # -5.5%  sum_z_dr2_top20 < 0.007538
        + 0.05492068 * max(0.0, 1053.047 - Q.sum_pt_top40) / 57.73988   # +5.5%  sum_pt_top40 < 1053
        + 0.04838674 * max(0.0, Q.LHA - 0.1870291) / 0.08997036   # +4.8%  LHA > 0.187
        - 0.0467294 * max(0.0, 0.008031986 - Q.sum_z_dr2_top40) / 0.002514593   # -4.7%  sum_z_dr2_top40 < 0.008032
        - 0.03901763 * max(0.0, 0.00592208 - Q.sum_zz_dr2) / 0.001192334   # -3.9%  sum_zz_dr2 < 0.005922
        - 0.0385834 * max(0.0, Q.LHA - 0.2454112) / 0.05012361   # -3.9%  LHA > 0.2454
        - 0.03350367 * max(0.0, Q.sum_z_dr2_top15 - 0.007887677) / 0.002755219   # -3.4%  sum_z_dr2_top15 > 0.007888
        + 0.03337008 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21062   # +3.3%  n_dr_0p2_0p4 < 26
        - 0.03133472 * max(0.0, Q.n_particles - 29.0) / 17.61555   # -3.1%  n_particles > 29
        + 0.028423 * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 14.20854   # +2.8%  n_dr_0p1_0p2 < 26
        - 0.02754366 * max(0.0, 0.006716737 - Q.lam1) / 0.001912105   # -2.8%  lam1 < 0.006717
        + 0.01868557 * max(0.0, 0.03263075 - Q.e2) / 0.008034085   # +1.9%  e2 < 0.03263
        + 0.01617092 * max(0.0, Q.sum_z_dr - 0.076787) / 0.01351025   # +1.6%  sum_z_dr > 0.07679
        + 0.01349002 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, Q.sum_pt_top40 - 858.8262) / 0.172906   # +1.3%  psi_0p3 > 0.9974 and sum_pt_top40 > 858.8
        - 0.01182698 * max(0.0, Q.sum_z_dr2_top15 - 0.004169954) / 0.004340368   # -1.2%  sum_z_dr2_top15 > 0.00417
        - 0.007846142 * max(0.0, Q.sj2_dr - 0.2595052) * max(0.0, 0.0329485 - Q.C2_b2) / 0.0001569167   # -0.8%  sj2_dr > 0.2595 and C2_b2 < 0.03295
        - 0.007169216 * max(0.0, 0.985099 - Q.z_top50_slots) / 0.003556075   # -0.7%  z_top50_slots < 0.9851
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 13.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.45117 * (0.02145765
        + 0.1494265 * max(0.0, Q.sum_pt_top50 - 934.2416) / 108.3881   # +14.9%  sum_pt_top50 > 934.2
        - 0.09252788 * max(0.0, Q.log_sum_pt - 6.910131) / 0.05175087   # -9.3%  log_sum_pt > 6.91
        - 0.0919667 * max(0.0, Q.sum_zz_dr2 - 0.007872294) / 0.00365949   # -9.2%  sum_zz_dr2 > 0.007872
        + 0.07442591 * max(0.0, Q.lam1_plus_lam2 - 0.006941794) / 0.004053246   # +7.4%  lam1_plus_lam2 > 0.006942
        + 0.07020338 * max(0.0, Q.sum_pt - 907.9372) / 139.8197   # +7.0%  sum_pt > 907.9
        - 0.06144426 * max(0.0, 902.4062 - Q.sum_pt_top5) / 313.0794   # -6.1%  sum_pt_top5 < 902.4
        + 0.05542988 * max(0.0, Q.sum_pt_top30 - 911.9328) / 96.43087   # +5.5%  sum_pt_top30 > 911.9
        - 0.040855 * max(0.0, Q.z_top30_slots - 0.9203881) / 0.04411614   # -4.1%  z_top30_slots > 0.9204
        + 0.03813719 * max(0.0, 64.0 - Q.n_particles) / 18.18477   # +3.8%  n_particles < 64
        - 0.03718824 * max(0.0, 0.01563836 - Q.sum_z_dr2_top15) / 0.009593305   # -3.7%  sum_z_dr2_top15 < 0.01564
        - 0.03382963 * max(0.0, Q.log_sum_pt - 6.920349) / 0.04509237   # -3.4%  log_sum_pt > 6.92
        + 0.03213979 * max(0.0, 0.01626937 - Q.tau4) / 0.002963016   # +3.2%  tau4 < 0.01627
        - 0.03078771 * max(0.0, Q.sum_pt_top50 - 1048.098) / 31.96699   # -3.1%  sum_pt_top50 > 1048
        - 0.02608908 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -2.6%  sum_pt_top5 > 531.2
        + 0.02194797 * max(0.0, Q.n_pt_above_1 - 26.0) / 16.44623   # +2.2%  n_pt_above_1 > 26
        + 0.01808719 * max(0.0, 0.03187688 - Q.M3) / 0.009170584   # +1.8%  M3 < 0.03188
        - 0.01742751 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.410481 - Q.D2) / 13.17726   # -1.7%  n_particles < 64 and D2 < 2.41
        - 0.01568391 * max(0.0, Q.log_sum_pt - 6.935549) / 0.0371538   # -1.6%  log_sum_pt > 6.936
        + 0.0141629 * max(0.0, Q.log_sum_pt - 6.959294) / 0.02846004   # +1.4%  log_sum_pt > 6.959
        + 0.01350082 * max(0.0, Q.log_sum_pt - 7.017258) / 0.01635989   # +1.4%  log_sum_pt > 7.017
        + 0.01043705 * max(0.0, 579.875 - Q.sum_pt_top5) / 65.87113   # +1.0%  sum_pt_top5 < 579.9
        - 0.009651904 * max(0.0, Q.n_pt_above_1 - 26.0) * max(0.0, 30.0 - Q.n_dr_0_0p05) / 283.4356   # -1.0%  n_pt_above_1 > 26 and n_dr_0_0p05 < 30
        + 0.00915474 * max(0.0, 0.005383629 - Q.sum_z_dr2_top15) / 0.001666876   # +0.9%  sum_z_dr2_top15 < 0.005384
        + 0.008721818 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # +0.9%  sum_pt_top20 > 1018
        - 0.007268901 * max(0.0, Q.sum_pt_top15 - 967.7705) / 19.94058   # -0.7%  sum_pt_top15 > 967.8
        + 0.006383959 * max(0.0, Q.sum_pt_top10 - 822.975) / 44.35907   # +0.6%  sum_pt_top10 > 823
        - 0.005907038 * max(0.0, Q.sum_pt_top30 - 1052.08) / 20.84425   # -0.6%  sum_pt_top30 > 1052
        - 0.005453145 * max(0.0, 0.0004816552 - Q.sum_z_dr2_top15) / 3.66329e-05   # -0.5%  sum_z_dr2_top15 < 0.0004817
        + 0.001760016 * max(0.0, Q.sum_z_dr2 - 0.02928196) / 0.0002029538   # +0.2%  sum_z_dr2 > 0.02928
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 15.5;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.50236 * (0.04053652
        - 0.1746699 * max(0.0, 0.009614971 - Q.lam1_plus_lam2) / 0.003502545   # -17.5%  lam1_plus_lam2 < 0.009615
        + 0.1152587 * max(0.0, 0.02580859 - Q.sum_zz_dr2) / 0.01698103   # +11.5%  sum_zz_dr2 < 0.02581
        - 0.09855301 * max(0.0, Q.e2 - 0.01541561) / 0.01690634   # -9.9%  e2 > 0.01542
        - 0.06677572 * max(0.0, 0.01397874 - Q.sum_z_dr2) / 0.006902497   # -6.7%  sum_z_dr2 < 0.01398
        + 0.05507982 * max(0.0, 0.0003372339 - Q.e3) / 0.0002564224   # +5.5%  e3 < 0.0003372
        + 0.05403819 * max(0.0, Q.psi_0p3 - 0.9853273) / 0.009408723   # +5.4%  psi_0p3 > 0.9853
        + 0.04989984 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +5.0%  lam1 < 0.007671
        + 0.04623683 * max(0.0, 0.003811746 - Q.lam1) / 0.0006796118   # +4.6%  lam1 < 0.003812
        - 0.04613605 * max(0.0, 0.01397874 - Q.sum_z_dr2) * max(0.0, Q.psi_0p3 - 0.9853273) / 7.658135e-05   # -4.6%  sum_z_dr2 < 0.01398 and psi_0p3 > 0.9853
        - 0.04111459 * max(0.0, 0.02265114 - Q.sum_z_dr2_top20) / 0.01537233   # -4.1%  sum_z_dr2_top20 < 0.02265
        + 0.03451075 * max(0.0, Q.LHA - 0.2091025) / 0.07391154   # +3.5%  LHA > 0.2091
        - 0.02518514 * max(0.0, 0.02580859 - Q.sum_zz_dr2) * max(0.0, 6.98945 - Q.log_sum_pt) / 0.001023242   # -2.5%  sum_zz_dr2 < 0.02581 and log_sum_pt < 6.989
        + 0.0248805 * max(0.0, 0.001776308 - Q.lam2) / 0.0009519164   # +2.5%  lam2 < 0.001776
        + 0.0225004 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # +2.3%  log_sum_pt > 6.811
        - 0.0219227 * max(0.0, 7.876005e-05 - Q.e3) / 3.594075e-05   # -2.2%  e3 < 7.876e-05
        + 0.01755601 * max(0.0, 0.007511864 - Q.sum_zz_dr2) / 0.002016671   # +1.8%  sum_zz_dr2 < 0.007512
        - 0.01610827 * max(0.0, Q.lam2 - 0.0009731947) / 0.0007968622   # -1.6%  lam2 > 0.0009732
        + 0.01554392 * max(0.0, 0.007511864 - Q.sum_zz_dr2) * max(0.0, 1007.788 - Q.sum_pt) / 0.02913759   # +1.6%  sum_zz_dr2 < 0.007512 and sum_pt < 1008
        + 0.01466015 * max(0.0, Q.sum_z_dr2_top30 - 0.008376291) / 0.003092781   # +1.5%  sum_z_dr2_top30 > 0.008376
        + 0.01370589 * max(0.0, Q.e2 - 0.04755309) * max(0.0, 0.3289237 - Q.z_dr_0_0p05) / 0.0005973978   # +1.4%  e2 > 0.04755 and z_dr_0_0p05 < 0.3289
        + 0.009908364 * max(0.0, Q.sum_pt_top20 - 695.3094) * max(0.0, 1.36316 - Q.D2_b2) / 96.35963   # +1.0%  sum_pt_top20 > 695.3 and D2_b2 < 1.363
        - 0.009855181 * max(0.0, Q.lam1 - 0.006189818) / 0.003311577   # -1.0%  lam1 > 0.00619
        + 0.009124696 * max(0.0, 0.00287991 - Q.sum_z_dr2_top20) / 0.000559956   # +0.9%  sum_z_dr2_top20 < 0.00288
        - 0.007481535 * max(0.0, 0.0003372339 - Q.e3) * max(0.0, 0.3384815 - Q.N2) / 1.669221e-05   # -0.7%  e3 < 0.0003372 and N2 < 0.3385
        - 0.006951161 * max(0.0, 0.003811746 - Q.lam1) * max(0.0, 1008.935 - Q.sum_pt_top50) / 0.0115984   # -0.7%  lam1 < 0.003812 and sum_pt_top50 < 1009
        + 0.002342618 * max(0.0, Q.lam2 - 0.003687605) * max(0.0, 2.459011 - Q.D2_b2) / 0.0001347626   # +0.2%  lam2 > 0.003688 and D2_b2 < 2.459
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 34.06;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.05818 * (-0.01011796
        + 0.1523455 * max(0.0, 0.009606007 - Q.sum_zz_dr2) / 0.003497866   # +15.2%  sum_zz_dr2 < 0.009606
        - 0.09915088 * max(0.0, 0.009606007 - Q.sum_zz_dr2) * max(0.0, 1167.447 - Q.sum_pt) / 0.4400479   # -9.9%  sum_zz_dr2 < 0.009606 and sum_pt < 1167
        - 0.08696048 * max(0.0, 0.008124776 - Q.sum_z_dr2_top50) / 0.002470567   # -8.7%  sum_z_dr2_top50 < 0.008125
        + 0.05956981 * max(0.0, 0.01397874 - Q.lam1_plus_lam2) / 0.006902497   # +6.0%  lam1_plus_lam2 < 0.01398
        - 0.05842235 * max(0.0, 0.08589404 - Q.sum_z_dr) / 0.02745918   # -5.8%  sum_z_dr < 0.08589
        + 0.05275221 * max(0.0, 0.007820315 - Q.sum_z_dr2_top50) / 0.002264874   # +5.3%  sum_z_dr2_top50 < 0.00782
        - 0.05188063 * max(0.0, 0.007877041 - Q.sum_z_dr2) / 0.002242297   # -5.2%  sum_z_dr2 < 0.007877
        - 0.04518498 * max(0.0, 0.008241985 - Q.lam1) / 0.002945027   # -4.5%  lam1 < 0.008242
        + 0.04516551 * max(0.0, 0.007671243 - Q.lam1) / 0.002527553   # +4.5%  lam1 < 0.007671
        + 0.03840462 * max(0.0, 0.302389 - Q.LHA) / 0.06234723   # +3.8%  LHA < 0.3024
        - 0.03429945 * max(0.0, 0.006936725 - Q.sum_zz_dr2) / 0.001688614   # -3.4%  sum_zz_dr2 < 0.006937
        - 0.02822771 * max(0.0, 0.00634935 - Q.sum_z_dr2_top50) / 0.001425305   # -2.8%  sum_z_dr2_top50 < 0.006349
        - 0.0240536 * max(0.0, 0.007511864 - Q.sum_zz_dr2) / 0.002016671   # -2.4%  sum_zz_dr2 < 0.007512
        - 0.02354512 * max(0.0, 0.04828819 - Q.tau2) / 0.02103241   # -2.4%  tau2 < 0.04829
        - 0.02140237 * max(0.0, 0.2601462 - Q.LHA) / 0.03967371   # -2.1%  LHA < 0.2601
        + 0.01922678 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.9%  e2 < 0.03481
        + 0.01554396 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # +1.6%  n_dr_0p2_0p4 < 15
        - 0.01494407 * max(0.0, Q.max_dr - 0.2404747) / 0.1177359   # -1.5%  max_dr > 0.2405
        + 0.01414708 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.0002814467   # +1.4%  lam2 < 0.002397 and log_sum_pt < 7.139
        + 0.01325755 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.01396296 - Q.sum_zz_dr2) / 0.000313768   # +1.3%  tau21_b2 < 0.2352 and sum_zz_dr2 < 0.01396
        + 0.01135152 * max(0.0, Q.psi_0p3 - 0.9966167) / 0.001363574   # +1.1%  psi_0p3 > 0.9966
        - 0.01063091 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # -1.1%  lam1 < 0.005914
        + 0.01005441 * max(0.0, 0.08589404 - Q.sum_z_dr) * max(0.0, 1156.659 - Q.sum_pt_top50) / 3.29885   # +1.0%  sum_z_dr < 0.08589 and sum_pt_top50 < 1157
        + 0.008474822 * max(0.0, 0.07708632 - Q.tau1) / 0.01617694   # +0.8%  tau1 < 0.07709
        - 0.00820754 * max(0.0, 0.0001086251 - Q.e3) / 5.924919e-05   # -0.8%  e3 < 0.0001086
        + 0.008136537 * max(0.0, 0.05660088 - Q.sum_z_dr) / 0.01080382   # +0.8%  sum_z_dr < 0.0566
        - 0.007124731 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007872294 - Q.sum_zz_dr2) / 5.103177e-05   # -0.7%  tau21_b2 < 0.2352 and sum_zz_dr2 < 0.007872
        + 0.006725156 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1167.447 - Q.sum_pt) / 0.1821423   # +0.7%  lam1 < 0.005914 and sum_pt < 1167
        + 0.006656333 * max(0.0, Q.max_dr - 0.2982) / 0.06837446   # +0.7%  max_dr > 0.2982
        - 0.006501495 * max(0.0, Q.psi_0p3 - 0.9980008) / 0.0006664852   # -0.7%  psi_0p3 > 0.998
        - 0.006051138 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 1225.842 - Q.sum_pt_top40) / 1472.634   # -0.6%  n_dr_0p2_0p4 < 15 and sum_pt_top40 < 1226
        + 0.004905306 * max(0.0, Q.psi_0p3 - 0.9966167) * max(0.0, 2.275391 - Q.soft1_pt) / 0.002319433   # +0.5%  psi_0p3 > 0.9966 and soft1_pt < 2.275
        + 0.003188674 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.sum_pt_top30 - 978.0762) / 2.738836   # +0.3%  tau21_b2 < 0.2352 and sum_pt_top30 > 978.1
        - 0.003118227 * max(0.0, 0.002396991 - Q.lam2) * max(0.0, Q.zdr_0 - 0.002524869) / 9.492656e-06   # -0.3%  lam2 < 0.002397 and zdr_0 > 0.002525
        - 0.0003885174 * max(0.0, Q.sum_pt_top15 - 1082.548) * max(0.0, Q.mean_phi - 2.298159e-05) / 0.0002044402   # -0.0%  sum_pt_top15 > 1083 and mean_phi > 2.298e-05
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 28.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.54176 * (0.05365899
        - 0.1794496 * max(0.0, 0.0258669 - Q.sum_z_dr2) * max(0.0, 7.139296 - Q.log_sum_pt) / 0.003279142   # -17.9%  sum_z_dr2 < 0.02587 and log_sum_pt < 7.139
        + 0.1686048 * max(0.0, 0.0258669 - Q.sum_z_dr2) / 0.01702764   # +16.9%  sum_z_dr2 < 0.02587
        - 0.1028623 * max(0.0, 0.009606007 - Q.sum_zz_dr2) / 0.003497866   # -10.3%  sum_zz_dr2 < 0.009606
        - 0.07244185 * max(0.0, 0.01397874 - Q.sum_z_dr2) / 0.006902497   # -7.2%  sum_z_dr2 < 0.01398
        + 0.06436748 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) * max(0.0, 1260.541 - Q.sum_pt) / 0.4807996   # +6.4%  sum_z_dr2_top30 < 0.007464 and sum_pt < 1261
        + 0.05094315 * max(0.0, 0.1219132 - Q.tau1) / 0.04578087   # +5.1%  tau1 < 0.1219
        + 0.04202986 * max(0.0, 0.01351431 - Q.sum_z_dr2_top50) / 0.006620975   # +4.2%  sum_z_dr2_top50 < 0.01351
        + 0.04149158 * max(0.0, 1245.697 - Q.sum_pt_top50) / 217.4702   # +4.1%  sum_pt_top50 < 1246
        - 0.03921291 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) / 0.00233708   # -3.9%  sum_z_dr2_top30 < 0.007464
        - 0.03377 * max(0.0, 0.008190222 - Q.lam1_plus_lam2) / 0.002454223   # -3.4%  lam1_plus_lam2 < 0.00819
        + 0.02348281 * max(0.0, 1156.659 - Q.sum_pt_top50) / 134.6773   # +2.3%  sum_pt_top50 < 1157
        + 0.02270381 * max(0.0, 0.008840538 - Q.sum_z_dr2_top40) / 0.003108622   # +2.3%  sum_z_dr2_top40 < 0.008841
        - 0.02053298 * max(0.0, Q.sum_z_dr2_top15 - 0.005788041) / 0.003469237   # -2.1%  sum_z_dr2_top15 > 0.005788
        + 0.01826745 * max(0.0, Q.sum_z_dr2_top15 - 0.007887677) / 0.002755219   # +1.8%  sum_z_dr2_top15 > 0.007888
        + 0.01659953 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # +1.7%  sum_pt < 1085
        - 0.01538777 * max(0.0, 0.0258669 - Q.sum_z_dr2) * max(0.0, Q.z_top50_slots - 0.9704436) / 0.000437459   # -1.5%  sum_z_dr2 < 0.02587 and z_top50_slots > 0.9704
        - 0.01525511 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904   # -1.5%  n_dr_0p2_0p4 < 18
        + 0.01477615 * max(0.0, 0.003638856 - Q.sum_z_dr2) / 0.0004964601   # +1.5%  sum_z_dr2 < 0.003639
        - 0.01355919 * max(0.0, 0.007463985 - Q.sum_z_dr2_top30) * max(0.0, 1095.686 - Q.sum_pt_top40) / 0.172292   # -1.4%  sum_z_dr2_top30 < 0.007464 and sum_pt_top40 < 1096
        + 0.01216016 * max(0.0, 1028.184 - Q.sum_pt) / 26.95739   # +1.2%  sum_pt < 1028
        - 0.008739036 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -0.9%  psi_0p3 > 0.9943
        - 0.005027386 * max(0.0, 1007.44 - Q.sum_pt_top40) / 29.24912   # -0.5%  sum_pt_top40 < 1007
        + 0.004538668 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.sj2_zsoft - 0.1181474) / 1.531301   # +0.5%  n_dr_0p2_0p4 < 18 and sj2_zsoft > 0.1181
        + 0.004077136 * max(0.0, 0.01397874 - Q.sum_z_dr2) * max(0.0, 986.0565 - Q.sum_pt) / 0.05983431   # +0.4%  sum_z_dr2 < 0.01398 and sum_pt < 986.1
        - 0.003497497 * max(0.0, 0.1219132 - Q.tau1) * max(0.0, 0.5364935 - Q.planar_flow) / 0.004630037   # -0.3%  tau1 < 0.1219 and planar_flow < 0.5365
        - 0.002307062 * max(0.0, 1.788105 - Q.D2) / 0.2865857   # -0.2%  D2 < 1.788
        + 0.00221766 * max(0.0, 0.002574843 - Q.sum_zz_dr2) / 0.0002582574   # +0.2%  sum_zz_dr2 < 0.002575
        + 0.001697153 * max(0.0, Q.sum_pt_top20 - 1017.778) / 17.66072   # +0.2%  sum_pt_top20 > 1018
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 7.542;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.541737 * (-0.07471854
        + 0.1342369 * max(0.0, 1191.938 - Q.sum_pt_top30) / 206.0686   # +13.4%  sum_pt_top30 < 1192
        + 0.1340885 * max(0.0, 0.006941794 - Q.lam1_plus_lam2) / 0.001690424   # +13.4%  lam1_plus_lam2 < 0.006942
        + 0.1099356 * max(0.0, Q.lam1 - 0.004673423) / 0.004174908   # +11.0%  lam1 > 0.004673
        - 0.0662527 * max(0.0, Q.lam1 - 0.006716737) / 0.003087986   # -6.6%  lam1 > 0.006717
        - 0.06487906 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # -6.5%  sum_z_dr2_top20 < 0.01083
        + 0.06118725 * max(0.0, 0.005712208 - Q.sum_z_dr2_top40) / 0.001219686   # +6.1%  sum_z_dr2_top40 < 0.005712
        + 0.05598251 * max(0.0, 986.0565 - Q.sum_pt) / 11.98437   # +5.6%  sum_pt < 986.1
        - 0.05090376 * max(0.0, 0.02675364 - Q.sum_z_dr2_top15) / 0.01959938   # -5.1%  sum_z_dr2_top15 < 0.02675
        - 0.03569754 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # -3.6%  tau1 < 0.06311
        - 0.03407276 * max(0.0, Q.sum_z_dr - 0.1207452) / 0.004285542   # -3.4%  sum_z_dr > 0.1207
        + 0.03188633 * max(0.0, 0.007259287 - Q.lam1) / 0.00224977   # +3.2%  lam1 < 0.007259
        - 0.0305034 * max(0.0, 0.007259287 - Q.lam1) * max(0.0, Q.sum_pt - 1085.125) / 0.08770494   # -3.1%  lam1 < 0.007259 and sum_pt > 1085
        + 0.02588354 * max(0.0, Q.sum_zz_dr2 - 0.01989454) / 0.001200608   # +2.6%  sum_zz_dr2 > 0.01989
        - 0.02484952 * max(0.0, Q.sum_z_dr2_top15 - 0.02146578) / 0.0006182335   # -2.5%  sum_z_dr2_top15 > 0.02147
        + 0.02173326 * max(0.0, 0.005712208 - Q.sum_z_dr2_top40) * max(0.0, Q.log_sum_pt - 7.017258) / 3.653263e-05   # +2.2%  sum_z_dr2_top40 < 0.005712 and log_sum_pt > 7.017
        - 0.02064202 * max(0.0, Q.sum_z_dr2 - 0.0258669) / 0.0004653552   # -2.1%  sum_z_dr2 > 0.02587
        - 0.01850958 * max(0.0, 972.4111 - Q.sum_pt_top40) * max(0.0, 0.0369869 - Q.tau3) / 0.1284345   # -1.9%  sum_pt_top40 < 972.4 and tau3 < 0.03699
        - 0.01445272 * max(0.0, 6.903423 - Q.log_sum_pt) / 0.01543876   # -1.4%  log_sum_pt < 6.903
        - 0.01422181 * max(0.0, Q.e3 - 0.0005178279) * max(0.0, Q.soft3_pt - 2.144727) / 7.713445e-07   # -1.4%  e3 > 0.0005178 and soft3_pt > 2.145
        + 0.01236751 * max(0.0, Q.sum_z_dr - 0.1207452) * max(0.0, 0.4606099 - Q.sj3_pairmin_over_m) / 0.0004267265   # +1.2%  sum_z_dr > 0.1207 and sj3_pairmin_over_m < 0.4606
        - 0.01219196 * max(0.0, 0.001319197 - Q.sum_z_dr2_top15) / 0.0002099631   # -1.2%  sum_z_dr2_top15 < 0.001319
        - 0.01039649 * max(0.0, Q.e3 - 0.0005178279) / 8.375623e-06   # -1.0%  e3 > 0.0005178
        + 0.007804327 * max(0.0, Q.sum_zz_dr2 - 0.0292152) / 0.0002016952   # +0.8%  sum_zz_dr2 > 0.02922
        + 0.007320953 * max(0.0, 2.485823 - Q.pt_entropy) / 0.09968085   # +0.7%  pt_entropy < 2.486
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 18.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.42022 * (0.1652796
        - 0.2025949 * max(0.0, 0.1207452 - Q.sum_z_dr) / 0.05578614   # -20.3%  sum_z_dr < 0.1207
        - 0.0801644 * max(0.0, 0.0005178279 - Q.e3) / 0.0004236992   # -8.0%  e3 < 0.0005178
        + 0.0785095 * max(0.0, 0.3719813 - Q.LHA) / 0.1170757   # +7.9%  LHA < 0.372
        + 0.06075876 * max(0.0, Q.e2 - 0.006720044) / 0.02422374   # +6.1%  e2 > 0.00672
        - 0.05161353 * max(0.0, Q.sum_z_dr2_top20 - 0.005718442) / 0.003749064   # -5.2%  sum_z_dr2_top20 > 0.005718
        + 0.04196095 * max(0.0, 0.0259543 - Q.tau4) / 0.009964748   # +4.2%  tau4 < 0.02595
        + 0.041488 * max(0.0, 0.1507173 - Q.tau1) / 0.06959675   # +4.1%  tau1 < 0.1507
        + 0.03571651 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4) / 0.05824142   # +3.6%  z_dr_0p2_0p4 < 0.09123
        + 0.03291074 * max(0.0, 0.00818374 - Q.sum_zz_dr2) / 0.002451404   # +3.3%  sum_zz_dr2 < 0.008184
        + 0.03224224 * max(0.0, 0.00592208 - Q.sum_zz_dr2) / 0.001192334   # +3.2%  sum_zz_dr2 < 0.005922
        + 0.03182101 * max(0.0, Q.sum_z_dr2_top20 - 0.008031209) / 0.002906852   # +3.2%  sum_z_dr2_top20 > 0.008031
        + 0.03042902 * max(0.0, 656.3844 - Q.sum_pt_top3) / 209.0067   # +3.0%  sum_pt_top3 < 656.4
        - 0.02559857 * max(0.0, 0.009606007 - Q.sum_zz_dr2) / 0.003497866   # -2.6%  sum_zz_dr2 < 0.009606
        + 0.02415058 * max(0.0, 0.08222447 - Q.M2) / 0.02103373   # +2.4%  M2 < 0.08222
        - 0.02177887 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.482887   # -2.2%  n_dr_0p2_0p4 < 15
        - 0.02118234 * max(0.0, 3.539367 - Q.pt_entropy) / 0.7052473   # -2.1%  pt_entropy < 3.539
        - 0.02078771 * max(0.0, Q.sum_z_dr2 - 0.01397874) / 0.002228371   # -2.1%  sum_z_dr2 > 0.01398
        - 0.02028786 * max(0.0, 0.01976735 - Q.sum_z_dr2_top10) / 0.01371247   # -2.0%  sum_z_dr2_top10 < 0.01977
        - 0.01943438 * max(0.0, 0.03577037 - Q.sum_z_dr) / 0.004182456   # -1.9%  sum_z_dr < 0.03577
        - 0.01809194 * max(0.0, 0.3062621 - Q.tau21_b2) / 0.08794306   # -1.8%  tau21_b2 < 0.3063
        - 0.01794984 * max(0.0, Q.z_dr_0_0p05 - 0.7128619) / 0.07526795   # -1.8%  z_dr_0_0p05 > 0.7129
        - 0.01576741 * max(0.0, 0.006403325 - Q.lam1_plus_lam2) / 0.001406912   # -1.6%  lam1_plus_lam2 < 0.006403
        - 0.01513045 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) / 0.00347358   # -1.5%  sum_z_dr2_top10 < 0.007679
        - 0.01145938 * max(0.0, Q.e2 - 0.006720044) * max(0.0, Q.log_sum_pt - 6.893714) / 0.001240306   # -1.1%  e2 > 0.00672 and log_sum_pt > 6.894
        + 0.01055291 * max(0.0, 0.06413297 - Q.dr_0) / 0.02518956   # +1.1%  dr_0 < 0.06413
        + 0.01050136 * max(0.0, 0.005402331 - Q.sum_z_dr2_top30) / 0.001231293   # +1.1%  sum_z_dr2_top30 < 0.005402
        + 0.009871266 * max(0.0, Q.n_dr_0_0p05 - 6.0) / 7.551284   # +1.0%  n_dr_0_0p05 > 6
        - 0.008902172 * max(0.0, Q.psi_0p3 - 0.9985421) / 0.0004356739   # -0.9%  psi_0p3 > 0.9985
        + 0.006578448 * max(0.0, Q.sum_z_dr2 - 0.01397874) * max(0.0, 114.75 - Q.pt_3) / 0.1232701   # +0.7%  sum_z_dr2 > 0.01398 and pt_3 < 114.8
        - 0.001733303 * max(0.0, Q.sum_z_dr2 - 0.02928196) / 0.0002029538   # -0.2%  sum_z_dr2 > 0.02928
        - 3.160811e-05 * max(0.0, Q.sum_z_dr2 - 0.02928196) * max(0.0, 0.03287546 - Q.sj3_dr13) / 1.371132e-09   # -0.0%  sum_z_dr2 > 0.02928 and sj3_dr13 < 0.03288
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 14.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.97861 * (0.04785978
        + 0.1683896 * max(0.0, 0.00818374 - Q.sum_zz_dr2) / 0.002451404   # +16.8%  sum_zz_dr2 < 0.008184
        - 0.097328 * max(0.0, 0.00818374 - Q.sum_zz_dr2) * max(0.0, 1115.723 - Q.sum_pt) / 0.1997244   # -9.7%  sum_zz_dr2 < 0.008184 and sum_pt < 1116
        + 0.07772259 * max(0.0, 0.01215787 - Q.sum_z_dr2_top30) / 0.005935412   # +7.8%  sum_z_dr2_top30 < 0.01216
        - 0.06673893 * max(0.0, 0.00634935 - Q.sum_z_dr2_top50) / 0.001425305   # -6.7%  sum_z_dr2_top50 < 0.006349
        + 0.06593047 * max(0.0, 0.009606007 - Q.sum_zz_dr2) * max(0.0, 1115.723 - Q.sum_pt) / 0.2875932   # +6.6%  sum_zz_dr2 < 0.009606 and sum_pt < 1116
        - 0.06441179 * max(0.0, 0.004754578 - Q.sum_zz_dr2) / 0.0007999118   # -6.4%  sum_zz_dr2 < 0.004755
        - 0.06155599 * max(0.0, 4.450169 - Q.D2) / 2.013555   # -6.2%  D2 < 4.45
        - 0.04805993 * max(0.0, 0.006363916 - Q.sum_z_dr2_top30) / 0.001678624   # -4.8%  sum_z_dr2_top30 < 0.006364
        - 0.03833461 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 1115.723 - Q.sum_pt) / 0.1207895   # -3.8%  lam1 < 0.005914 and sum_pt < 1116
        - 0.03321107 * max(0.0, Q.n_real_top50 - 22.0) / 19.58685   # -3.3%  n_real_top50 > 22
        - 0.03005884 * max(0.0, 0.07374472 - Q.sum_z_dr) / 0.01915593   # -3.0%  sum_z_dr < 0.07374
        + 0.02981612 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583   # +3.0%  n_dr_0p2_0p4 < 10
        + 0.02937573 * max(0.0, 0.005913555 - Q.lam1) / 0.001463924   # +2.9%  lam1 < 0.005914
        - 0.02923086 * max(0.0, Q.C2_b2 - 0.002289486) / 0.01267025   # -2.9%  C2_b2 > 0.002289
        - 0.02601621 * max(0.0, 0.02210818 - Q.e2) / 0.00342102   # -2.6%  e2 < 0.02211
        + 0.02274827 * max(0.0, 0.02515919 - Q.e2) / 0.004550017   # +2.3%  e2 < 0.02516
        - 0.01649154 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 1156.659 - Q.sum_pt_top50) / 469.6702   # -1.6%  n_dr_0p2_0p4 < 10 and sum_pt_top50 < 1157
        - 0.01444634 * max(0.0, 0.03875945 - Q.e2) / 0.01184119   # -1.4%  e2 < 0.03876
        - 0.01430876 * max(0.0, Q.z_top30_slots - 0.9734886) / 0.01006171   # -1.4%  z_top30_slots > 0.9735
        + 0.01080183 * max(0.0, 0.005809485 - Q.sum_z_dr2_top30) / 0.001402136   # +1.1%  sum_z_dr2_top30 < 0.005809
        + 0.01001319 * max(0.0, Q.psi_0p2 - 0.9935324) / 0.001465572   # +1.0%  psi_0p2 > 0.9935
        + 0.009336017 * max(0.0, Q.z_top30_slots - 0.9734886) * max(0.0, 1.08774 - Q.D2_b2) / 0.003455048   # +0.9%  z_top30_slots > 0.9735 and D2_b2 < 1.088
        - 0.009180485 * max(0.0, Q.sum_pt_top15 - 840.7969) / 74.67343   # -0.9%  sum_pt_top15 > 840.8
        + 0.008763783 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0008024071 - Q.soft2_z) / 0.0008410862   # +0.9%  n_dr_0p2_0p4 < 10 and soft2_z < 0.0008024
        + 0.006233138 * max(0.0, 0.00130722 - Q.sum_z_dr2_top10) / 0.0002794102   # +0.6%  sum_z_dr2_top10 < 0.001307
        - 0.00588043 * max(0.0, 0.0007431905 - Q.sum_z_dr2_top10) / 0.0001243965   # -0.6%  sum_z_dr2_top10 < 0.0007432
        - 0.005615512 * max(0.0, 0.005913555 - Q.lam1) * max(0.0, 3.014827 - Q.D2_b2) / 0.0004725177   # -0.6%  lam1 < 0.005914 and D2_b2 < 3.015
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 6.267;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.267015 * (-0.0563563
        + 0.2195218 * max(0.0, 0.006403325 - Q.sum_z_dr2) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0002246937   # +22.0%  sum_z_dr2 < 0.006403 and log_sum_pt > 6.811
        + 0.2183455 * max(0.0, 0.007511864 - Q.sum_zz_dr2) / 0.002016671   # +21.8%  sum_zz_dr2 < 0.007512
        - 0.1959019 * max(0.0, 0.007511864 - Q.sum_zz_dr2) * max(0.0, Q.log_sum_pt - 6.811175) / 0.0003183593   # -19.6%  sum_zz_dr2 < 0.007512 and log_sum_pt > 6.811
        - 0.07278173 * max(0.0, 0.005809485 - Q.sum_z_dr2_top30) / 0.001402136   # -7.3%  sum_z_dr2_top30 < 0.005809
        + 0.06427427 * max(0.0, Q.psi_0p3 - 0.9943058) * max(0.0, 0.00751625 - Q.sum_z_dr2) / 6.462915e-06   # +6.4%  psi_0p3 > 0.9943 and sum_z_dr2 < 0.007516
        - 0.05978303 * max(0.0, Q.psi_0p3 - 0.9943058) / 0.002758093   # -6.0%  psi_0p3 > 0.9943
        - 0.0486199 * max(0.0, Q.sum_pt - 1085.125) / 25.7173   # -4.9%  sum_pt > 1085
        - 0.0424627 * max(0.0, 0.002752094 - Q.lam1) / 0.0003912817   # -4.2%  lam1 < 0.002752
        - 0.02358689 * max(0.0, Q.psi_0p1 - 0.9371031) * max(0.0, 0.05444336 - Q.absphi_0) / 0.0004737047   # -2.4%  psi_0p1 > 0.9371 and absphi_0 < 0.05444
        + 0.01869836 * max(0.0, Q.LHA - 0.3719813) / 0.007150691   # +1.9%  LHA > 0.372
        - 0.01707392 * max(0.0, Q.z_dr_0_0p05 - 0.878906) / 0.01583525   # -1.7%  z_dr_0_0p05 > 0.8789
        + 0.006817465 * max(0.0, 0.08133662 - Q.psi_0p1) * max(0.0, Q.sum_pt_top50 - 889.8503) / 0.2667141   # +0.7%  psi_0p1 < 0.08134 and sum_pt_top50 > 889.9
        - 0.006805356 * max(0.0, 0.08133662 - Q.psi_0p1) / 0.002484547   # -0.7%  psi_0p1 < 0.08134
        + 0.005327234 * max(0.0, Q.sum_pt - 1085.125) * max(0.0, 0.001868041 - Q.lam1) / 0.006895658   # +0.5%  sum_pt > 1085 and lam1 < 0.001868
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 14.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.18842 * (0.0513531
        + 0.2230666 * max(0.0, Q.log_sum_pt - 6.811175) / 0.1383141   # +22.3%  log_sum_pt > 6.811
        - 0.0822106 * max(0.0, Q.sum_pt - 1017.435) / 47.92778   # -8.2%  sum_pt > 1017
        - 0.07675972 * max(0.0, 1085.125 - Q.sum_pt) / 67.04637   # -7.7%  sum_pt < 1085
        - 0.06408475 * max(0.0, Q.sum_pt_top50 - 889.8503) / 149.655   # -6.4%  sum_pt_top50 > 889.9
        + 0.05671626 * max(0.0, Q.sum_z_dr2_top40 - 0.01292642) * max(0.0, 1245.697 - Q.sum_pt_top50) / 0.6025464   # +5.7%  sum_z_dr2_top40 > 0.01293 and sum_pt_top50 < 1246
        + 0.04631484 * max(0.0, Q.sum_pt_top50 - 976.277) / 72.29277   # +4.6%  sum_pt_top50 > 976.3
        - 0.0435707 * max(0.0, Q.sum_z_dr2_top50 - 0.01351431) / 0.002240755   # -4.4%  sum_z_dr2_top50 > 0.01351
        - 0.04002206 * max(0.0, Q.sum_z_dr2_top40 - 0.008840538) / 0.003161074   # -4.0%  sum_z_dr2_top40 > 0.008841
        + 0.03970996 * max(0.0, Q.sum_z_dr2_top50 - 0.005852839) / 0.004485718   # +4.0%  sum_z_dr2_top50 > 0.005853
        + 0.03501551 * max(0.0, 1078.994 - Q.sum_pt_top50) / 67.77696   # +3.5%  sum_pt_top50 < 1079
        + 0.02935782 * max(0.0, Q.sum_pt_top50 - 1013.916) / 45.94021   # +2.9%  sum_pt_top50 > 1014
        - 0.02915451 * max(0.0, Q.log_sum_pt - 6.893714) / 0.06417424   # -2.9%  log_sum_pt > 6.894
        - 0.02555121 * max(0.0, Q.sum_pt_top40 - 972.4111) / 67.15108   # -2.6%  sum_pt_top40 > 972.4
        + 0.02481618 * max(0.0, Q.sum_z_dr2_top40 - 0.02862386) / 0.0001970429   # +2.5%  sum_z_dr2_top40 > 0.02862
        - 0.02345628 * max(0.0, Q.sum_z_dr2_top40 - 0.01292642) / 0.002259475   # -2.3%  sum_z_dr2_top40 > 0.01293
        - 0.02133607 * max(0.0, Q.lam1_plus_lam2 - 0.01991191) / 0.001207633   # -2.1%  lam1_plus_lam2 > 0.01991
        - 0.01665785 * max(0.0, 25.0 - Q.n_pt_above_5) / 2.617361   # -1.7%  n_pt_above_5 < 25
        - 0.01633674 * max(0.0, Q.sum_z_dr2_top40 - 0.02862386) * max(0.0, Q.pt_6 - 19.46875) / 0.003416206   # -1.6%  sum_z_dr2_top40 > 0.02862 and pt_6 > 19.47
        + 0.01617556 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, Q.tau21 - 0.1295048) / 22.44785   # +1.6%  sum_pt < 1085 and tau21 > 0.1295
        + 0.01530318 * max(0.0, Q.e2 - 0.03680582) / 0.004603798   # +1.5%  e2 > 0.03681
        - 0.01393269 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, 0.01330402 - Q.C2_b2) / 0.0003309112   # -1.4%  sum_z_dr < 0.1207 and C2_b2 < 0.0133
        + 0.01254417 * max(0.0, Q.sum_pt_top10 - 867.9266) * max(0.0, 0.006580753 - Q.C2_b2) / 0.04500262   # +1.3%  sum_pt_top10 > 867.9 and C2_b2 < 0.006581
        - 0.0122355 * max(0.0, Q.sum_z_dr2_top40 - 0.02862386) * max(0.0, Q.z_6 - 0.01906139) / 3.587165e-06   # -1.2%  sum_z_dr2_top40 > 0.02862 and z_6 > 0.01906
        - 0.01212317 * max(0.0, Q.sum_z_dr2_top40 - 0.02862386) * max(0.0, 62.25 - Q.pt_6) / 0.00503292   # -1.2%  sum_z_dr2_top40 > 0.02862 and pt_6 < 62.25
        + 0.01081913 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +1.1%  e2 < 0.03481
        + 0.003943987 * max(0.0, Q.sum_pt_top20 - 1064.139) / 11.11947   # +0.4%  sum_pt_top20 > 1064
        - 0.003381771 * max(0.0, Q.sum_pt_top20 - 1064.139) * max(0.0, Q.M2 - 0.05568888) / 0.2082962   # -0.3%  sum_pt_top20 > 1064 and M2 > 0.05569
        - 0.002797119 * max(0.0, 0.1207452 - Q.sum_z_dr) * max(0.0, 858.8262 - Q.sum_pt_top40) / 0.1447816   # -0.3%  sum_z_dr < 0.1207 and sum_pt_top40 < 858.8
        - 0.002470905 * max(0.0, Q.sum_pt - 1167.447) / 14.21193   # -0.2%  sum_pt > 1167
        - 0.0001351968 * max(0.0, Q.sum_zz_dr2 - 0.0292152) * max(0.0, Q.pt_4 - 90.625) / 1.381275e-05   # -0.0%  sum_zz_dr2 > 0.02922 and pt_4 > 90.62
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 20.46;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.46328 * (-0.02955587
        - 0.07505482 * max(0.0, 0.008190222 - Q.sum_z_dr2) / 0.002454223   # -7.5%  sum_z_dr2 < 0.00819
        + 0.07214479 * max(0.0, 0.01989454 - Q.sum_zz_dr2) / 0.01180402   # +7.2%  sum_zz_dr2 < 0.01989
        - 0.06681078 * max(0.0, 0.01951641 - Q.sum_z_dr2_top50) / 0.01159099   # -6.7%  sum_z_dr2_top50 < 0.01952
        + 0.06388399 * max(0.0, 0.01215787 - Q.sum_z_dr2_top30) * max(0.0, Q.sum_pt - 907.9372) / 0.9148907   # +6.4%  sum_z_dr2_top30 < 0.01216 and sum_pt > 907.9
        - 0.05984521 * max(0.0, 0.006936725 - Q.sum_zz_dr2) / 0.001688614   # -6.0%  sum_zz_dr2 < 0.006937
        - 0.05513764 * max(0.0, 0.05444509 - Q.tau1) / 0.007883192   # -5.5%  tau1 < 0.05445
        + 0.0548052 * max(0.0, 0.009606007 - Q.sum_zz_dr2) / 0.003497866   # +5.5%  sum_zz_dr2 < 0.009606
        - 0.05193133 * max(0.0, 0.01215787 - Q.sum_z_dr2_top30) / 0.005935412   # -5.2%  sum_z_dr2_top30 < 0.01216
        + 0.05174503 * max(0.0, Q.psi_0p3 - 0.9956185) / 0.001937698   # +5.2%  psi_0p3 > 0.9956
        - 0.0440459 * max(0.0, 0.01083435 - Q.sum_z_dr2_top20) / 0.00529825   # -4.4%  sum_z_dr2_top20 < 0.01083
        - 0.03789058 * max(0.0, Q.psi_0p3 - 0.9956185) * max(0.0, 1225.842 - Q.sum_pt_top40) / 0.3755165   # -3.8%  psi_0p3 > 0.9956 and sum_pt_top40 < 1226
        + 0.03560738 * max(0.0, 0.01396296 - Q.sum_zz_dr2) / 0.00689253   # +3.6%  sum_zz_dr2 < 0.01396
        - 0.03547949 * max(0.0, 0.00616708 - Q.sum_zz_dr2) / 0.001295555   # -3.5%  sum_zz_dr2 < 0.006167
        + 0.03326568 * max(0.0, 0.03480688 - Q.e2) / 0.009278192   # +3.3%  e2 < 0.03481
        + 0.03218348 * max(0.0, Q.psi_0p3 - 0.9896594) / 0.00604025   # +3.2%  psi_0p3 > 0.9897
        + 0.03194589 * max(0.0, 0.342495 - Q.tau21_b2) / 0.1089185   # +3.2%  tau21_b2 < 0.3425
        - 0.02713358 * max(0.0, 0.1008497 - Q.tau1) / 0.02962944   # -2.7%  tau1 < 0.1008
        - 0.02017701 * max(0.0, 0.03480688 - Q.e2) * max(0.0, Q.sum_pt_top30 - 886.3438) / 1.374127   # -2.0%  e2 < 0.03481 and sum_pt_top30 > 886.3
        + 0.01940263 * max(0.0, 0.006374178 - Q.sum_z_dr2_top20) / 0.001987751   # +1.9%  sum_z_dr2_top20 < 0.006374
        - 0.01532851 * max(0.0, 0.008376291 - Q.sum_z_dr2_top30) / 0.002981077   # -1.5%  sum_z_dr2_top30 < 0.008376
        - 0.01498395 * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 6.597062   # -1.5%  n_dr_0p1_0p2 < 17
        + 0.01395524 * max(0.0, 0.03787151 - Q.M3) / 0.01370579   # +1.4%  M3 < 0.03787
        - 0.01389121 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.007709916 - Q.sum_z_dr2_top40) / 0.0001062984   # -1.4%  tau21_b2 < 0.3425 and sum_z_dr2_top40 < 0.00771
        - 0.01278236 * max(0.0, 0.5936969 - Q.tau21) / 0.1891413   # -1.3%  tau21 < 0.5937
        - 0.01262758 * max(0.0, Q.LHA - 0.3098384) / 0.0195892   # -1.3%  LHA > 0.3098
        - 0.01108324 * max(0.0, 0.00751625 - Q.lam1_plus_lam2) / 0.002018105   # -1.1%  lam1_plus_lam2 < 0.007516
        - 0.01093745 * max(0.0, 0.2091025 - Q.LHA) / 0.02095782   # -1.1%  LHA < 0.2091
        + 0.00716705 * max(0.0, 0.001155057 - Q.sum_z_dr2_top3) / 0.000329166   # +0.7%  sum_z_dr2_top3 < 0.001155
        - 0.007024311 * max(0.0, 0.006088416 - Q.sum_z_dr2_top30) / 0.001533892   # -0.7%  sum_z_dr2_top30 < 0.006088
        + 0.004024799 * max(0.0, Q.sum_pt_top30 - 1052.08) / 20.84425   # +0.4%  sum_pt_top30 > 1052
        + 0.00386323 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.006259772 - Q.sum_z_dr2_top40) / 3.28785e-05   # +0.4%  tau21_b2 < 0.3425 and sum_z_dr2_top40 < 0.00626
        - 0.003840665 * max(0.0, Q.log_sum_pt - 7.062574) / 0.01087884   # -0.4%  log_sum_pt > 7.063
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 5.138;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.138049 * (-0.03071072
        + 0.165003 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) / 0.004544146   # +16.5%  sum_z_dr2_top5 < 0.00833
        - 0.1237546 * max(0.0, 0.0705748 - Q.tau1) / 0.01344376   # -12.4%  tau1 < 0.07057
        - 0.1051904 * max(0.0, Q.sum_pt_top30 - 988.4375) / 43.28413   # -10.5%  sum_pt_top30 > 988.4
        + 0.08052208 * max(0.0, 0.04082832 - Q.e2) / 0.01334159   # +8.1%  e2 < 0.04083
        + 0.05965346 * max(0.0, Q.sum_pt_top20 - 869.693) / 87.9958   # +6.0%  sum_pt_top20 > 869.7
        + 0.05903196 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) / 0.047292   # +5.9%  z_dr_0p1_0p2 < 0.1203
        + 0.05418867 * max(0.0, Q.sum_pt_top50 - 1156.659) / 13.71464   # +5.4%  sum_pt_top50 > 1157
        - 0.04764385 * max(0.0, 0.008329695 - Q.sum_z_dr2_top5) * max(0.0, Q.psi_0p3 - 0.9299135) / 0.0002825377   # -4.8%  sum_z_dr2_top5 < 0.00833 and psi_0p3 > 0.9299
        + 0.04339417 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr0 - 0.03721986) / 0.009661509   # +4.3%  sd_rg < 0.1778 and soft5_dr0 > 0.03722
        - 0.03738163 * max(0.0, 0.1778185 - Q.sd_rg) * max(0.0, Q.soft5_dr - 0.04139378) / 0.009084795   # -3.7%  sd_rg < 0.1778 and soft5_dr > 0.04139
        + 0.03576983 * max(0.0, 0.06310829 - Q.tau1) / 0.01067358   # +3.6%  tau1 < 0.06311
        - 0.03224134 * max(0.0, 0.9860575 - Q.z_top30_slots) / 0.03906901   # -3.2%  z_top30_slots < 0.9861
        - 0.02820252 * max(0.0, 0.002270363 - Q.sum_z_dr2_top5) / 0.0007634099   # -2.8%  sum_z_dr2_top5 < 0.00227
        + 0.0232222 * max(0.0, Q.sj2_dr - 0.2780918) / 0.009227968   # +2.3%  sj2_dr > 0.2781
        - 0.02134165 * max(0.0, Q.log_sum_pt - 7.139296) / 0.005452502   # -2.1%  log_sum_pt > 7.139
        - 0.01895609 * max(0.0, Q.psi_0p1 - 0.8976117) / 0.02472107   # -1.9%  psi_0p1 > 0.8976
        - 0.01810653 * max(0.0, 0.007678544 - Q.sum_z_dr2_top10) * max(0.0, Q.tau4 - 0.01873799) / 4.309981e-06   # -1.8%  sum_z_dr2_top10 < 0.007679 and tau4 > 0.01874
        - 0.01760766 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 995.6769 - Q.sum_pt) / 0.6029747   # -1.8%  z_dr_0p1_0p2 < 0.1203 and sum_pt < 995.7
        - 0.0165533 * max(0.0, Q.sj3_dr_max - 0.3715619) / 0.009730894   # -1.7%  sj3_dr_max > 0.3716
        + 0.01223507 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 986.0565 - Q.sum_pt) / 0.5137322   # +1.2%  z_dr_0p1_0p2 < 0.1203 and sum_pt < 986.1
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.7043230042016807, 2.797152100840336, 0.2598577731092437, 0.4604825105042017, 0.924765756302521, 1.067070168067227, 0.5417168067226891, 0.5512991596638656, 1.323094537815126, 1.2618430672268908, 1.2610169117647059, 0.6257661764705882, 0.5000474789915966, 1.504973161764706, 0.3873186449579832, 0.3610840336134454]
T = [4.149741214220064, 2.736508739824054, 4.651606036633404, 4.625141389837184, 4.123886006433824]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +42%, n4 -19%, n9 +16%, n3 -8%, n5 -8%, n12 +3% ...
            + 0.4212841 * h[1] / H_AVG[1]
            - 0.1949929 * h[4] / H_AVG[4]
            + 0.15679 * h[9] / H_AVG[9]
            - 0.08322492 * h[3] / H_AVG[3]
            - 0.08035668 * h[5] / H_AVG[5]
            + 0.02824239 * h[12] / H_AVG[12]
            + 0.02039724 * h[6] / H_AVG[6]
            + 0.009963683 * h[8] / H_AVG[8]
            - 0.004748101 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -36%, n9 +26%, n1 -19%, n12 +6%, n11 -6%, n6 +4% ...
            - 0.3590574 * h[4] / H_AVG[4]
            + 0.2593767 * h[9] / H_AVG[9]
            - 0.1916552 * h[1] / H_AVG[1]
            + 0.0628141 * h[12] / H_AVG[12]
            - 0.0571683 * h[11] / H_AVG[11]
            + 0.04330355 * h[6] / H_AVG[6]
            + 0.01186995 * h[2] / H_AVG[2]
            + 0.007554645 * h[8] / H_AVG[8]
            - 0.007200192 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +13%, n14 -11%, n0 +11%, n11 +8%, n7 -7% ...
            - 0.2488834 * h[8] / H_AVG[8]
            + 0.1326208 * h[5] / H_AVG[5]
            - 0.1144902 * h[14] / H_AVG[14]
            + 0.1135613 * h[0] / H_AVG[0]
            + 0.07987535 * h[11] / H_AVG[11]
            - 0.07407377 * h[7] / H_AVG[7]
            + 0.06833946 * h[4] / H_AVG[4]
            - 0.0593404 * h[9] / H_AVG[9]
            - 0.04367186 * h[12] / H_AVG[12]
            + 0.04331001 * h[3] / H_AVG[3]
            - 0.01455481 * h[15] / H_AVG[15]
            + 0.007278626 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -21%, n5 +16%, n6 -11%, n7 +11%, n15 +4% ...
            - 0.2681866 * h[8] / H_AVG[8]
            - 0.2093869 * h[0] / H_AVG[0]
            + 0.1586137 * h[5] / H_AVG[5]
            - 0.1134642 * h[6] / H_AVG[6]
            + 0.1080215 * h[7] / H_AVG[7]
            + 0.04391428 * h[15] / H_AVG[15]
            + 0.04355782 * h[3] / H_AVG[3]
            + 0.03378596 * h[12] / H_AVG[12]
            - 0.02106891 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -33%, n10 +30%, n5 -12%, n8 +7%, n12 -5%, n4 +4% ...
            - 0.3307274 * h[13] / H_AVG[13]
            + 0.3010058 * h[10] / H_AVG[10]
            - 0.1212907 * h[5] / H_AVG[5]
            + 0.06767652 * h[8] / H_AVG[8]
            - 0.04547114 * h[12] / H_AVG[12]
            + 0.04204616 * h[4] / H_AVG[4]
            - 0.03759873 * h[7] / H_AVG[7]
            - 0.03283469 * h[15] / H_AVG[15]
            + 0.02134889 * h[0] / H_AVG[0]
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
