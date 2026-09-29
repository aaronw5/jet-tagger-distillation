"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables or exact equivalents), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.5%   (on for 41% of jets)
  neuron  1:  12.6%   (on for 97% of jets)
  neuron  5:  11.3%   (on for 77% of jets)
  neuron  4:   9.8%   (on for 68% of jets)
  neuron  0:   7.6%   (on for 60% of jets)
  neuron 13:   7.3%   (on for 89% of jets)
  neuron  9:   6.7%   (on for 62% of jets)
  neuron 10:   6.2%   (on for 81% of jets)
  neuron 12:   5.6%   (on for 36% of jets)
  neuron  7:   4.3%   (on for 49% of jets)
  neuron  6:   3.7%   (on for 45% of jets)
  neuron 11:   3.4%   (on for 74% of jets)
  neuron  3:   3.4%   (on for 56% of jets)
  neuron 15:   2.3%   (on for 64% of jets)
  neuron 14:   1.8%   (on for 31% of jets)
  neuron  2:   0.6%   (on for 87% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.5% (the network: 81.1%); same class as the network for 92.4% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.pt_8                   pT of particle 8 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.soft6_dr               ΔR from the jet axis of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        pt_8=pt[8],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        soft6_dr=softp(6, 'dr'),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 16.73;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.72598 * (-0.0552434
        + 0.2246589 * max(0.0, 0.0817 - Q.sum_z_dr) / 0.02440026   # +22.5%  sum_z_dr < 0.0817
        + 0.1325706 * max(0.0, 0.00784 - Q.sum_z_dr2_top15) / 0.003227617   # +13.3%  sum_z_dr2_top15 < 0.00784
        - 0.1241832 * max(0.0, 0.309 - Q.LHA) / 0.06678734   # -12.4%  LHA < 0.309
        - 0.1048966 * max(0.0, 0.0102 - Q.sum_z_dr2_top15) / 0.005085501   # -10.5%  sum_z_dr2_top15 < 0.0102
        - 0.06380646 * max(0.0, Q.sj2_dr - 0.153) / 0.05863877   # -6.4%  sj2_dr > 0.153
        + 0.0514112 * max(0.0, 17.9 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_pt_top50 - 890.0) / 1549.374   # +5.1%  n_dr_0p2_0p4 < 17.9 and sum_pt_top50 > 890
        + 0.04145564 * max(0.0, Q.sj2_dr - 0.175) / 0.04502508   # +4.1%  sj2_dr > 0.175
        - 0.03141642 * max(0.0, 0.00544 - Q.sum_z_dr2_top15) / 0.001695066   # -3.1%  sum_z_dr2_top15 < 0.00544
        - 0.02920074 * max(0.0, Q.log_sum_pt - 6.89) / 0.0671817   # -2.9%  log_sum_pt > 6.89
        - 0.02879821 * max(0.0, 0.0446 - Q.tau1) / 0.00520171   # -2.9%  tau1 < 0.0446
        - 0.02554353 * max(0.0, 18.1 - Q.n_dr_0p2_0p4) * max(0.0, Q.log_sum_pt - 6.92) / 0.4559665   # -2.6%  n_dr_0p2_0p4 < 18.1 and log_sum_pt > 6.92
        - 0.0178967 * max(0.0, 0.0504 - Q.sum_z_dr) / 0.008503972   # -1.8%  sum_z_dr < 0.0504
        + 0.0176608 * max(0.0, 17.9 - Q.n_dr_0p2_0p4) / 9.879406   # +1.8%  n_dr_0p2_0p4 < 17.9
        + 0.01729471 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # +1.7%  psi_0p3 > 0.99
        + 0.01687738 * max(0.0, 0.187 - Q.LHA) / 0.01493602   # +1.7%  LHA < 0.187
        - 0.01233387 * max(0.0, 0.0718 - Q.dr_0) / 0.03074458   # -1.2%  dr_0 < 0.0718
        - 0.01211437 * max(0.0, 2.89e-05 - Q.e3) / 6.49438e-06   # -1.2%  e3 < 2.89e-05
        + 0.007990074 * max(0.0, 0.485 - Q.z_dr_0_0p05) / 0.1715556   # +0.8%  z_dr_0_0p05 < 0.485
        - 0.007301046 * max(0.0, 0.185 - Q.sj2_dr) / 0.03213609   # -0.7%  sj2_dr < 0.185
        + 0.006143992 * max(0.0, 5.12e-05 - Q.e3) / 1.698583e-05   # +0.6%  e3 < 5.12e-05
        + 0.004972397 * max(0.0, 0.214 - Q.sj3_dr_max) / 0.02744825   # +0.5%  sj3_dr_max < 0.214
        - 0.003816354 * max(0.0, 17.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.tau21_b2 - 0.212) / 1.516206   # -0.4%  n_dr_0p2_0p4 < 17.7 and tau21_b2 > 0.212
        - 0.003802166 * max(0.0, 0.166 - Q.sj3_dr_max) / 0.01088955   # -0.4%  sj3_dr_max < 0.166
        + 0.003234639 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 26.2 - Q.n_dr_0p1_0p2) / 0.02652084   # +0.3%  psi_0p3 > 0.996 and n_dr_0p1_0p2 < 26.2
        + 0.003101721 * max(0.0, Q.sum_pt_top40 - 999.0) * max(0.0, 0.142 - Q.sj2_dr) / 1.155441   # +0.3%  sum_pt_top40 > 999 and sj2_dr < 0.142
        + 0.002694715 * max(0.0, Q.sum_pt_top2 - 409.0) / 50.07972   # +0.3%  sum_pt_top2 > 409
        - 0.002457468 * max(0.0, 17.6 - Q.n_dr_0p2_0p4) * max(0.0, Q.sum_z_dr2_top10 - 0.00488) / 0.01268629   # -0.2%  n_dr_0p2_0p4 < 17.6 and sum_z_dr2_top10 > 0.00488
        + 0.002366069 * max(0.0, Q.log_sum_pt - 7.02) / 0.01595759   # +0.2%  log_sum_pt > 7.02
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 13.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.17927 * (0.08498194
        + 0.1533615 * max(0.0, Q.sum_pt - 973.0) / 80.52558   # +15.3%  sum_pt > 973
        - 0.1219032 * max(0.0, Q.sum_pt_top50 - 941.0) / 102.3309   # -12.2%  sum_pt_top50 > 941
        + 0.116436 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +11.6%  log_sum_pt > 6.91
        - 0.1119288 * max(0.0, 7.13 - Q.log_sum_pt) / 0.191328   # -11.2%  log_sum_pt < 7.13
        + 0.07441736 * max(0.0, 1080.0 - Q.sum_pt_top50) / 68.58507   # +7.4%  sum_pt_top50 < 1080
        + 0.06441483 * max(0.0, Q.n_particles - 38.9) / 10.21589   # +6.4%  n_particles > 38.9
        + 0.03721576 * max(0.0, 0.0747 - Q.tau1) / 0.01513817   # +3.7%  tau1 < 0.0747
        - 0.03606888 * max(0.0, Q.log_sum_pt - 6.99) / 0.0210337   # -3.6%  log_sum_pt > 6.99
        + 0.03519423 * max(0.0, 573.0 - Q.sum_pt_top2) / 216.745   # +3.5%  sum_pt_top2 < 573
        - 0.03002585 * max(0.0, Q.sum_pt - 1050.0) / 34.41033   # -3.0%  sum_pt > 1050
        - 0.028618 * max(0.0, 0.0237 - Q.tau4) / 0.008093654   # -2.9%  tau4 < 0.0237
        + 0.02637911 * max(0.0, 0.00682 - Q.sum_z_dr2_top3) / 0.003698483   # +2.6%  sum_z_dr2_top3 < 0.00682
        - 0.02505093 * max(0.0, Q.n_particles - 39.1) * max(0.0, 0.0024 - Q.soft1_z) / 0.01535595   # -2.5%  n_particles > 39.1 and soft1_z < 0.0024
        - 0.0206543 * max(0.0, 0.00168 - Q.lam2) / 0.0008752686   # -2.1%  lam2 < 0.00168
        - 0.01336515 * max(0.0, 0.0334 - Q.M3) * max(0.0, Q.psi_0p3 - 0.932) / 0.0005337665   # -1.3%  M3 < 0.0334 and psi_0p3 > 0.932
        + 0.0115433 * max(0.0, 0.354 - Q.N2) * max(0.0, 0.875 - Q.pt2_over_pt0) / 0.02859629   # +1.2%  N2 < 0.354 and pt2_over_pt0 < 0.875
        - 0.01102304 * max(0.0, Q.z_top5 - 0.645) * max(0.0, 27.4 - Q.pt1_dr01) / 0.875154   # -1.1%  z_top5 > 0.645 and pt1_dr01 < 27.4
        - 0.01018809 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # -1.0%  psi_0p3 > 0.997
        + 0.009897244 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.0115 - Q.zdr_0) / 0.0436249   # +1.0%  n_particles > 38 and zdr_0 < 0.0115
        - 0.008594211 * max(0.0, Q.z_dr_0_0p05 - 0.811) * max(0.0, 10.7 - Q.n_dr_0p05_0p1) / 0.2292823   # -0.9%  z_dr_0_0p05 > 0.811 and n_dr_0p05_0p1 < 10.7
        + 0.007956827 * max(0.0, 0.00194 - Q.sum_z_dr2_top15) / 0.0003758608   # +0.8%  sum_z_dr2_top15 < 0.00194
        + 0.00769115 * max(0.0, 0.0585 - Q.C2) / 0.01115113   # +0.8%  C2 < 0.0585
        + 0.007446415 * max(0.0, Q.n_dr_0_0p05 - 15.2) / 2.645238   # +0.7%  n_dr_0_0p05 > 15.2
        - 0.005650367 * max(0.0, 7.29 - Q.n_dr_0p1_0p2) / 1.07457   # -0.6%  n_dr_0p1_0p2 < 7.29
        + 0.005501442 * max(0.0, Q.z_top30_slots - 0.934) * max(0.0, Q.pt1_dr01 - 6.04) / 0.1212458   # +0.6%  z_top30_slots > 0.934 and pt1_dr01 > 6.04
        - 0.005136829 * max(0.0, Q.zdr_0 - 0.0129) / 0.002064014   # -0.5%  zdr_0 > 0.0129
        + 0.00510875 * max(0.0, 2.4 - Q.soft9_pt) * max(0.0, 0.14 - Q.dr_11) / 0.04518765   # +0.5%  soft9_pt < 2.4 and dr_11 < 0.14
        + 0.004308209 * max(0.0, Q.z_top30_slots - 0.933) * max(0.0, Q.ptdr0_2 - 6.97) / 0.08134534   # +0.4%  z_top30_slots > 0.933 and ptdr0_2 > 6.97
        + 0.003474434 * max(0.0, Q.z_top30_slots - 0.938) * max(0.0, Q.ptdr0_4 - 4.73) / 0.05087833   # +0.3%  z_top30_slots > 0.938 and ptdr0_4 > 4.73
        - 0.001445911 * max(0.0, Q.n_dr_0_0p05 - 29.9) / 0.2175348   # -0.1%  n_dr_0_0p05 > 29.9
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.5097;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5097039 * (0.07259116
        + 0.4577282 * max(0.0, Q.log_sum_pt - 6.92) / 0.04530211   # +45.8%  log_sum_pt > 6.92
        - 0.2871971 * max(0.0, Q.log_sum_pt - 7.05) * max(0.0, Q.sum_z_dr2_top15 - 0.00976) / 1.534439e-05   # -28.7%  log_sum_pt > 7.05 and sum_z_dr2_top15 > 0.00976
        - 0.1657223 * max(0.0, Q.sd_rg - 0.274) / 0.007609846   # -16.6%  sd_rg > 0.274
        - 0.06467239 * max(0.0, Q.sum_pt_top20 - 1120.0) * max(0.0, Q.dr_max_012 - 0.121) / 0.0652748   # -6.5%  sum_pt_top20 > 1120 and dr_max_012 > 0.121
        - 0.02468009 * max(0.0, Q.sum_pt_top15 - 1080.0) * max(0.0, Q.C2 - 0.143) / 0.0003060716   # -2.5%  sum_pt_top15 > 1080 and C2 > 0.143
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 0.8956;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.8956004 * (-0.3148726
        + 0.2971043 * max(0.0, 47.6 - Q.n_particles) / 7.230619   # +29.7%  n_particles < 47.6
        + 0.2083726 * max(0.0, 5.48 - Q.n_dr_0p2_0p4) / 1.314215   # +20.8%  n_dr_0p2_0p4 < 5.48
        + 0.1617759 * max(0.0, 0.000636 - Q.lam2) / 0.0001602727   # +16.2%  lam2 < 0.000636
        + 0.1563919 * max(0.0, 14.0 - Q.n_dr_0p1_0p2) / 4.460659   # +15.6%  n_dr_0p1_0p2 < 14
        - 0.1554289 * max(0.0, 47.7 - Q.n_particles) * max(0.0, Q.e2 - 0.0133) / 0.09802973   # -15.5%  n_particles < 47.7 and e2 > 0.0133
        - 0.02092634 * max(0.0, 10.5 - Q.n_dr_0p1_0p2) * max(0.0, 1010.0 - Q.sum_pt_top40) / 43.89143   # -2.1%  n_dr_0p1_0p2 < 10.5 and sum_pt_top40 < 1010
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 7.292;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.291806 * (0.2345098
        + 0.1386236 * max(0.0, Q.sj2_dr - 0.122) / 0.08218019   # +13.9%  sj2_dr > 0.122
        + 0.1348735 * max(0.0, 0.0099 - Q.sum_z_dr2_top15) / 0.004844686   # +13.5%  sum_z_dr2_top15 < 0.0099
        - 0.1075319 * max(0.0, 0.0076 - Q.sum_z_dr2_top15) / 0.003050979   # -10.8%  sum_z_dr2_top15 < 0.0076
        - 0.08451155 * max(0.0, Q.sj2_dr - 0.182) / 0.04135851   # -8.5%  sj2_dr > 0.182
        + 0.06601304 * max(0.0, Q.sum_z_dr - 0.0967) / 0.008474547   # +6.6%  sum_z_dr > 0.0967
        - 0.06383324 * max(0.0, Q.LHA - 0.334) / 0.01368999   # -6.4%  LHA > 0.334
        - 0.06155227 * max(0.0, 0.0444 - Q.e2) / 0.01614486   # -6.2%  e2 < 0.0444
        - 0.0426301 * max(0.0, 3.57 - Q.pt_entropy) / 0.7331378   # -4.3%  pt_entropy < 3.57
        - 0.04172686 * max(0.0, 0.0758 - Q.sum_z_dr) / 0.02042042   # -4.2%  sum_z_dr < 0.0758
        - 0.04087458 * max(0.0, 1040.0 - Q.sum_pt_top50) / 38.85913   # -4.1%  sum_pt_top50 < 1040
        - 0.0355423 * max(0.0, Q.n_pt_above_1 - 28.1) / 14.72543   # -3.6%  n_pt_above_1 > 28.1
        - 0.03398619 * max(0.0, 0.0475 - Q.z_dr_0p2_0p4) / 0.02536548   # -3.4%  z_dr_0p2_0p4 < 0.0475
        + 0.03211146 * max(0.0, 0.0999 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.997) / 4.609262e-05   # +3.2%  sum_z_dr < 0.0999 and psi_0p3 > 0.997
        + 0.03078586 * max(0.0, 0.0326 - Q.e2) / 0.008017305   # +3.1%  e2 < 0.0326
        - 0.03042786 * max(0.0, 0.00675 - Q.sum_z_dr2_top10) / 0.002804982   # -3.0%  sum_z_dr2_top10 < 0.00675
        - 0.0209173 * max(0.0, 0.0579 - Q.sum_z_dr) * max(0.0, Q.psi_0p3 - 0.997) / 1.146804e-05   # -2.1%  sum_z_dr < 0.0579 and psi_0p3 > 0.997
        - 0.011466 * max(0.0, Q.soft1_pt - 1.59) / 0.09362583   # -1.1%  soft1_pt > 1.59
        + 0.007020551 * max(0.0, Q.soft1_pt - 2.28) / 0.04570759   # +0.7%  soft1_pt > 2.28
        - 0.005308447 * max(0.0, Q.e2 - 0.0559) / 0.001087308   # -0.5%  e2 > 0.0559
        - 0.004964465 * max(0.0, Q.sum_z_dr2_top15 - 0.00558) * max(0.0, 0.41 - Q.sj3_pairmin_over_m) / 0.0003067789   # -0.5%  sum_z_dr2_top15 > 0.00558 and sj3_pairmin_over_m < 0.41
        + 0.002981925 * max(0.0, Q.n_particles - 24.2) * max(0.0, Q.lam2 - 0.00389) / 0.01120805   # +0.3%  n_particles > 24.2 and lam2 > 0.00389
        + 0.002317044 * max(0.0, 0.0077 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 1110.0) / 0.08405691   # +0.2%  sum_z_dr2_top15 < 0.0077 and sum_pt_top40 > 1110
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 4.692;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.692148 * (0.2557464
        + 0.1232804 * max(0.0, 1090.0 - Q.sum_pt) / 70.97545   # +12.3%  sum_pt < 1090
        - 0.09493361 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 6.07e-08 - Q.e4) / 7.653651e-07   # -9.5%  sum_pt < 1000 and e4 < 6.07e-08
        + 0.08302592 * max(0.0, 10.8 - Q.n_dr_0p2_0p4) / 4.352736   # +8.3%  n_dr_0p2_0p4 < 10.8
        - 0.07040588 * max(0.0, 909.0 - Q.sum_pt) / 4.014031   # -7.0%  sum_pt < 909
        - 0.06879632 * max(0.0, 0.0611 - Q.z_dr_0p2_0p4) / 0.03504913   # -6.9%  z_dr_0p2_0p4 < 0.0611
        - 0.06697373 * max(0.0, Q.sum_z_dr2_top10 - 0.00961) / 0.002152401   # -6.7%  sum_z_dr2_top10 > 0.00961
        + 0.06506563 * max(0.0, 46.7 - Q.n_real_top50) / 6.754371   # +6.5%  n_real_top50 < 46.7
        - 0.05406613 * max(0.0, Q.z_top30_slots - 0.951) / 0.02285462   # -5.4%  z_top30_slots > 0.951
        - 0.05395352 * max(0.0, Q.n_particles - 46.1) / 6.159559   # -5.4%  n_particles > 46.1
        + 0.04222338 * max(0.0, 1010.0 - Q.sum_pt_top40) / 30.43292   # +4.2%  sum_pt_top40 < 1010
        - 0.03747158 * max(0.0, Q.n_dr_0p1_0p2 - 7.08) / 6.487904   # -3.7%  n_dr_0p1_0p2 > 7.08
        - 0.03671574 * max(0.0, 6.89 - Q.log_sum_pt) / 0.01257487   # -3.7%  log_sum_pt < 6.89
        + 0.03264217 * max(0.0, 947.0 - Q.sum_pt_top50) / 8.324015   # +3.3%  sum_pt_top50 < 947
        - 0.02723 * max(0.0, 0.984 - Q.z_top50_slots) / 0.00334469   # -2.7%  z_top50_slots < 0.984
        + 0.02549005 * max(0.0, 0.158 - Q.sum_z_dr) * max(0.0, Q.n_dr_0p1_0p2 - 11.1) / 0.1728368   # +2.5%  sum_z_dr < 0.158 and n_dr_0p1_0p2 > 11.1
        - 0.02355922 * max(0.0, 62.4 - Q.n_particles) * max(0.0, 0.0793 - Q.C2) / 0.5070795   # -2.4%  n_particles < 62.4 and C2 < 0.0793
        - 0.02038144 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, Q.absphi_1 - 0.00176) / 0.6130304   # -2.0%  sum_pt < 1000 and absphi_1 > 0.00176
        + 0.01469239 * max(0.0, 914.0 - Q.sum_pt) * max(0.0, 0.14 - Q.dr0_12) / 0.2757554   # +1.5%  sum_pt < 914 and dr0_12 < 0.14
        + 0.0143033 * max(0.0, Q.M2 - 0.0621) / 0.01209246   # +1.4%  M2 > 0.0621
        - 0.01215487 * max(0.0, Q.LHA - 0.433) / 0.00101301   # -1.2%  LHA > 0.433
        - 0.00991633 * max(0.0, Q.C2 - 0.123) / 0.002474941   # -1.0%  C2 > 0.123
        - 0.008554245 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, Q.ptdr0_10 - 3.46) / 13.60603   # -0.9%  sum_pt < 1010 and ptdr0_10 > 3.46
        - 0.008230857 * max(0.0, 0.00071 - Q.sum_z_dr2_top15) / 7.528342e-05   # -0.8%  sum_z_dr2_top15 < 0.00071
        - 0.004003339 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, -0.00325 - Q.mean_eta) / 9.535157e-08   # -0.4%  psi_0p3 > 0.997 and mean_eta < -0.00325
        + 0.001929921 * max(0.0, 931.0 - Q.sum_pt_top50) * max(0.0, 0.0369 - Q.soft6_dr) / 0.009462353   # +0.2%  sum_pt_top50 < 931 and soft6_dr < 0.0369
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 8.201;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.200861 * (0.3060654
        - 0.2008481 * max(0.0, 0.0955 - Q.sum_z_dr) / 0.03497085   # -20.1%  sum_z_dr < 0.0955
        + 0.1018606 * max(0.0, 0.0565 - Q.tau1) / 0.008506562   # +10.2%  tau1 < 0.0565
        - 0.09788117 * max(0.0, 0.00878 - Q.sum_z_dr2_top15) / 0.003954236   # -9.8%  sum_z_dr2_top15 < 0.00878
        - 0.09198244 * max(0.0, 0.0764 - Q.z_dr_0p2_0p4) / 0.04656391   # -9.2%  z_dr_0p2_0p4 < 0.0764
        - 0.07378833 * max(0.0, 0.000324 - Q.e3) * max(0.0, 7.02 - Q.log_sum_pt) / 2.093868e-05   # -7.4%  e3 < 0.000324 and log_sum_pt < 7.02
        + 0.06381945 * max(0.0, 0.0473 - Q.e2) / 0.01855938   # +6.4%  e2 < 0.0473
        + 0.06322814 * max(0.0, 0.00318 - Q.sum_z_dr2_top15) / 0.0007659161   # +6.3%  sum_z_dr2_top15 < 0.00318
        - 0.05491222 * max(0.0, Q.e3 - 0.000116) / 5.170236e-05   # -5.5%  e3 > 0.000116
        - 0.04778168 * max(0.0, 0.191 - Q.sj2_dr) / 0.03530189   # -4.8%  sj2_dr < 0.191
        + 0.04663328 * max(0.0, Q.psi_0p2 - 0.907) * max(0.0, 0.154 - Q.sj2_dr) / 0.001627375   # +4.7%  psi_0p2 > 0.907 and sj2_dr < 0.154
        + 0.0184233 * max(0.0, Q.e2 - 0.0547) / 0.001208696   # +1.8%  e2 > 0.0547
        - 0.01740446 * max(0.0, 0.0014 - Q.sum_z_dr2_top15) / 0.0002302122   # -1.7%  sum_z_dr2_top15 < 0.0014
        - 0.01610321 * max(0.0, 0.333 - Q.LHA) * max(0.0, 0.966 - Q.psi_0p3) / 5.76682e-05   # -1.6%  LHA < 0.333 and psi_0p3 < 0.966
        + 0.01519617 * max(0.0, 0.0988 - Q.sum_z_dr) * max(0.0, 0.966 - Q.psi_0p3) / 1.968748e-05   # +1.5%  sum_z_dr < 0.0988 and psi_0p3 < 0.966
        - 0.01477027 * max(0.0, Q.z_dr_0p2_0p4 - 0.264) / 0.004884233   # -1.5%  z_dr_0p2_0p4 > 0.264
        - 0.01379036 * max(0.0, Q.z_dr_0p1_0p2 - 0.324) / 0.03873043   # -1.4%  z_dr_0p1_0p2 > 0.324
        - 0.01371318 * max(0.0, 0.00298 - Q.sum_z_dr2_top15) * max(0.0, Q.sum_pt_top40 - 852.0) / 0.1449226   # -1.4%  sum_z_dr2_top15 < 0.00298 and sum_pt_top40 > 852
        + 0.01060578 * max(0.0, Q.sum_pt_top50 - 1070.0) / 26.35652   # +1.1%  sum_pt_top50 > 1070
        + 0.01053237 * max(0.0, Q.n_dr_0p1_0p2 - 12.9) * max(0.0, Q.psi_0p3 - 0.99) / 0.01473968   # +1.1%  n_dr_0p1_0p2 > 12.9 and psi_0p3 > 0.99
        - 0.01019449 * max(0.0, Q.n_dr_0p2_0p4 - 15.1) / 1.520065   # -1.0%  n_dr_0p2_0p4 > 15.1
        - 0.006643981 * max(0.0, Q.C2 - 0.111) / 0.003919882   # -0.7%  C2 > 0.111
        + 0.006322447 * max(0.0, 0.0536 - Q.tau1) * max(0.0, 971.0 - Q.sum_pt) / 0.06489301   # +0.6%  tau1 < 0.0536 and sum_pt < 971
        + 0.003564609 * max(0.0, 0.0441 - Q.sj2_zsoft) / 0.002103084   # +0.4%  sj2_zsoft < 0.0441
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 22.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.57722 * (0.02551244
        - 0.1207513 * max(0.0, 0.107 - Q.tau1) / 0.03407787   # -12.1%  tau1 < 0.107
        + 0.1020255 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +10.2%  psi_0p3 > 0.997
        + 0.09701423 * max(0.0, 0.0978 - Q.sum_z_dr) / 0.03681196   # +9.7%  sum_z_dr < 0.0978
        + 0.07774484 * max(0.0, 0.107 - Q.tau1) * max(0.0, 0.287 - Q.z_dr_0p1_0p2) / 0.008126215   # +7.8%  tau1 < 0.107 and z_dr_0p1_0p2 < 0.287
        - 0.07158176 * max(0.0, 0.081 - Q.sum_z_dr) / 0.02390706   # -7.2%  sum_z_dr < 0.081
        + 0.0607826 * max(0.0, 0.302 - Q.LHA) / 0.06209512   # +6.1%  LHA < 0.302
        - 0.05917733 * max(0.0, 0.333 - Q.LHA) * max(0.0, 0.253 - Q.z_dr_0p1_0p2) / 0.01723948   # -5.9%  LHA < 0.333 and z_dr_0p1_0p2 < 0.253
        + 0.04761078 * max(0.0, 0.237 - Q.tau21_b2) * max(0.0, 0.00996 - Q.sum_z_dr2_top15) / 0.000173374   # +4.8%  tau21_b2 < 0.237 and sum_z_dr2_top15 < 0.00996
        - 0.04662892 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 0.00681 - Q.mean_eta2) / 4.637671e-06   # -4.7%  psi_0p3 > 0.997 and mean_eta2 < 0.00681
        - 0.04366079 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 0.00681 - Q.mean_phi2) / 4.627885e-06   # -4.4%  psi_0p3 > 0.997 and mean_phi2 < 0.00681
        - 0.03571505 * max(0.0, 0.236 - Q.tau21_b2) * max(0.0, 1250.0 - Q.sum_pt_top50) / 11.32509   # -3.6%  tau21_b2 < 0.236 and sum_pt_top50 < 1250
        + 0.02954455 * max(0.0, 0.234 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.15) / 0.003206894   # +3.0%  tau21_b2 < 0.234 and sj2_dr > 0.15
        - 0.02423903 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 7.06 - Q.log_sum_pt) / 0.0001334756   # -2.4%  psi_0p3 > 0.997 and log_sum_pt < 7.06
        - 0.02261953 * max(0.0, 0.235 - Q.tau21_b2) * max(0.0, 0.00789 - Q.sum_z_dr2_top15) / 8.080477e-05   # -2.3%  tau21_b2 < 0.235 and sum_z_dr2_top15 < 0.00789
        - 0.01892267 * max(0.0, 14.9 - Q.n_dr_0p2_0p4) / 7.404182   # -1.9%  n_dr_0p2_0p4 < 14.9
        + 0.01853028 * max(0.0, 0.0375 - Q.e2) / 0.01098064   # +1.9%  e2 < 0.0375
        - 0.01639608 * max(0.0, 20.4 - Q.n_dr_0p2_0p4) * max(0.0, 8e-05 - Q.e3) / 0.0005280714   # -1.6%  n_dr_0p2_0p4 < 20.4 and e3 < 8e-05
        + 0.01529225 * max(0.0, 15.4 - Q.n_dr_0p2_0p4) * max(0.0, 0.114 - Q.M2) / 0.3810779   # +1.5%  n_dr_0p2_0p4 < 15.4 and M2 < 0.114
        - 0.01332321 * max(0.0, 0.234 - Q.tau21_b2) / 0.05168405   # -1.3%  tau21_b2 < 0.234
        + 0.01183896 * max(0.0, 14.9 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.996) / 0.0190922   # +1.2%  n_dr_0p2_0p4 < 14.9 and psi_0p3 > 0.996
        + 0.01105749 * max(0.0, 0.236 - Q.tau21_b2) * max(0.0, 1110.0 - Q.sum_pt) / 4.274786   # +1.1%  tau21_b2 < 0.236 and sum_pt < 1110
        - 0.01068777 * max(0.0, 0.235 - Q.tau21_b2) * max(0.0, Q.sj2_dr - 0.193) / 0.001711349   # -1.1%  tau21_b2 < 0.235 and sj2_dr > 0.193
        - 0.008724305 * max(0.0, 0.0475 - Q.e2) * max(0.0, 0.235 - Q.tau21_b2) / 0.000562773   # -0.9%  e2 < 0.0475 and tau21_b2 < 0.235
        - 0.008671366 * max(0.0, Q.psi_0p1 - 0.854) / 0.04419308   # -0.9%  psi_0p1 > 0.854
        - 0.006989621 * max(0.0, 0.0435 - Q.sum_z_dr) / 0.006287101   # -0.7%  sum_z_dr < 0.0435
        - 0.006770595 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.sum_z_dr2_top10 - 0.00774) / 8.685297e-07   # -0.7%  psi_0p3 > 0.997 and sum_z_dr2_top10 > 0.00774
        - 0.005936019 * max(0.0, Q.n_dr_0p2_0p4 - 7.23) / 3.988655   # -0.6%  n_dr_0p2_0p4 > 7.23
        + 0.003718929 * max(0.0, 0.243 - Q.tau21_b2) * max(0.0, Q.n_dr_0p05_0p1 - 2.11) / 0.5954829   # +0.4%  tau21_b2 < 0.243 and n_dr_0p05_0p1 > 2.11
        + 0.003191132 * max(0.0, 14.9 - Q.n_dr_0p2_0p4) * max(0.0, 0.434 - Q.max_dr) / 0.8436405   # +0.3%  n_dr_0p2_0p4 < 14.9 and max_dr < 0.434
        - 0.000853056 * max(0.0, 0.234 - Q.tau21_b2) * max(0.0, 972.0 - Q.sum_pt) / 0.2116443   # -0.1%  tau21_b2 < 0.234 and sum_pt < 972
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 18.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.91409 * (0.1253034
        + 0.200206 * max(0.0, Q.sum_z_dr - 0.0438) / 0.03182112   # +20.0%  sum_z_dr > 0.0438
        - 0.1697023 * max(0.0, Q.LHA - 0.21) / 0.0732823   # -17.0%  LHA > 0.21
        + 0.1364561 * max(0.0, Q.sum_z_dr - 0.0353) / 0.03801095   # +13.6%  sum_z_dr > 0.0353
        - 0.06424815 * max(0.0, Q.sum_z_dr - 0.0747) / 0.01427961   # -6.4%  sum_z_dr > 0.0747
        - 0.05892432 * max(0.0, 7.14 - Q.log_sum_pt) / 0.2008108   # -5.9%  log_sum_pt < 7.14
        - 0.0580163 * max(0.0, 17.6 - Q.n_dr_0p2_0p4) * max(0.0, 7.17 - Q.log_sum_pt) / 2.147408   # -5.8%  n_dr_0p2_0p4 < 17.6 and log_sum_pt < 7.17
        - 0.04636708 * max(0.0, Q.sum_z_dr - 0.0965) / 0.008514476   # -4.6%  sum_z_dr > 0.0965
        - 0.04340282 * max(0.0, Q.z_top50_slots - 0.957) / 0.03584824   # -4.3%  z_top50_slots > 0.957
        - 0.03049303 * max(0.0, Q.e2 - 0.0213) / 0.01273174   # -3.0%  e2 > 0.0213
        - 0.02617472 * max(0.0, Q.psi_0p3 - 0.978) / 0.01547097   # -2.6%  psi_0p3 > 0.978
        + 0.02105778 * max(0.0, Q.sum_z_dr - 0.0683) * max(0.0, 1120.0 - Q.sum_pt) / 2.042506   # +2.1%  sum_z_dr > 0.0683 and sum_pt < 1120
        - 0.01960516 * max(0.0, 0.00717 - Q.sum_z_dr2_top15) / 0.002746768   # -2.0%  sum_z_dr2_top15 < 0.00717
        - 0.01940453 * max(0.0, 16.0 - Q.n_dr_0p2_0p4) * max(0.0, 4.44e-05 - Q.e3) / 0.0001306117   # -1.9%  n_dr_0p2_0p4 < 16 and e3 < 4.44e-05
        + 0.01836468 * max(0.0, Q.sum_z_dr - 0.0757) * max(0.0, 1200.0 - Q.sum_pt_top30) / 4.12531   # +1.8%  sum_z_dr > 0.0757 and sum_pt_top30 < 1200
        + 0.01546355 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # +1.5%  sum_pt < 1010
        + 0.01272138 * max(0.0, Q.sum_z_dr - 0.0787) * max(0.0, Q.e2 - 0.0433) / 0.0002021961   # +1.3%  sum_z_dr > 0.0787 and e2 > 0.0433
        - 0.0114619 * max(0.0, 1260.0 - Q.sum_pt) * max(0.0, 0.786 - Q.planar_flow) / 70.84686   # -1.1%  sum_pt < 1260 and planar_flow < 0.786
        + 0.008176245 * max(0.0, Q.sj2_dr - 0.213) / 0.02786418   # +0.8%  sj2_dr > 0.213
        + 0.007028257 * max(0.0, 1050.0 - Q.sum_pt) * max(0.0, Q.max_dr - 0.189) / 7.224623   # +0.7%  sum_pt < 1050 and max_dr > 0.189
        - 0.005201644 * max(0.0, Q.sum_z_dr - 0.0733) * max(0.0, 965.0 - Q.sum_pt_top50) / 0.3476479   # -0.5%  sum_z_dr > 0.0733 and sum_pt_top50 < 965
        + 0.004981693 * max(0.0, Q.sum_z_dr - 0.0736) * max(0.0, 0.0816 - Q.tau2) / 0.0002739075   # +0.5%  sum_z_dr > 0.0736 and tau2 < 0.0816
        + 0.004765817 * max(0.0, 0.0798 - Q.dr_0) * max(0.0, Q.n_dr_0p4_up - -0.251) / 0.01382532   # +0.5%  dr_0 < 0.0798 and n_dr_0p4_up > -0.251
        - 0.004431901 * max(0.0, 0.0117 - Q.sum_z_dr2_top15) * max(0.0, 0.328 - Q.max_dr) / 0.0001275881   # -0.4%  sum_z_dr2_top15 < 0.0117 and max_dr < 0.328
        + 0.004177046 * max(0.0, 20.5 - Q.n_dr_0p2_0p4) * max(0.0, 0.793 - Q.psi_0p1) / 0.5725001   # +0.4%  n_dr_0p2_0p4 < 20.5 and psi_0p1 < 0.793
        + 0.00391865 * max(0.0, Q.psi_0p3 - 0.971) * max(0.0, Q.n_dr_0p1_0p2 - 16.7) / 0.03963513   # +0.4%  psi_0p3 > 0.971 and n_dr_0p1_0p2 > 16.7
        - 0.003422626 * max(0.0, 0.0454 - Q.sj2_zsoft) / 0.002232271   # -0.3%  sj2_zsoft < 0.0454
        + 0.001826272 * max(0.0, 0.00679 - Q.sum_z_dr2_top5) * max(0.0, Q.sj3_dr23 - 0.264) / 7.460532e-05   # +0.2%  sum_z_dr2_top5 < 0.00679 and sj3_dr23 > 0.264
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 5.772;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.77179 * (-0.1784542
        + 0.2474071 * max(0.0, 0.0966 - Q.tau1) / 0.02674123   # +24.7%  tau1 < 0.0966
        + 0.08843735 * max(0.0, 0.26 - Q.sj2_dr) / 0.08051132   # +8.8%  sj2_dr < 0.26
        - 0.0783435 * max(0.0, Q.sum_pt_top20 - 759.0) / 177.3264   # -7.8%  sum_pt_top20 > 759
        + 0.07321815 * max(0.0, Q.e2 - 0.0261) / 0.009714938   # +7.3%  e2 > 0.0261
        - 0.06726778 * max(0.0, 0.00599 - Q.sum_z_dr2_top15) / 0.001991054   # -6.7%  sum_z_dr2_top15 < 0.00599
        + 0.06725354 * max(0.0, 0.00675 - Q.sum_z_dr2_top15) * max(0.0, 0.0696 - Q.z_dr_0p2_0p4) / 0.0001272699   # +6.7%  sum_z_dr2_top15 < 0.00675 and z_dr_0p2_0p4 < 0.0696
        - 0.06715071 * max(0.0, Q.z_top40_slots - 0.967) / 0.02083762   # -6.7%  z_top40_slots > 0.967
        + 0.0529413 * max(0.0, Q.sum_z_dr - 0.0989) / 0.008041212   # +5.3%  sum_z_dr > 0.0989
        - 0.04760808 * max(0.0, Q.sum_z_dr - 0.122) / 0.004101251   # -4.8%  sum_z_dr > 0.122
        - 0.04445631 * max(0.0, 0.182 - Q.sj2_dr) / 0.03061963   # -4.4%  sj2_dr < 0.182
        + 0.03999434 * max(0.0, 2.18 - Q.D2) / 0.4654011   # +4.0%  D2 < 2.18
        + 0.02839037 * max(0.0, 0.00592 - Q.sum_z_dr2_top15) * max(0.0, Q.psi_0p3 - 0.976) / 3.609322e-05   # +2.8%  sum_z_dr2_top15 < 0.00592 and psi_0p3 > 0.976
        + 0.01489233 * max(0.0, 0.0566 - Q.sum_z_dr) * max(0.0, 1010.0 - Q.sum_pt) / 0.1840587   # +1.5%  sum_z_dr < 0.0566 and sum_pt < 1010
        - 0.01397985 * max(0.0, 988.0 - Q.sum_pt_top50) * max(0.0, Q.sum_pt_top20 - 787.0) / 311.5397   # -1.4%  sum_pt_top50 < 988 and sum_pt_top20 > 787
        - 0.0134347 * max(0.0, Q.sum_z_dr2_top15 - 0.0201) / 0.0007602182   # -1.3%  sum_z_dr2_top15 > 0.0201
        + 0.01329748 * max(0.0, 0.00632 - Q.sum_z_dr2_top15) * max(0.0, Q.n_dr_0p2_0p4 - 7.2) / 0.004104291   # +1.3%  sum_z_dr2_top15 < 0.00632 and n_dr_0p2_0p4 > 7.2
        + 0.01113924 * max(0.0, Q.LHA - 0.375) * max(0.0, 0.00652 - Q.lam2) / 1.336661e-05   # +1.1%  LHA > 0.375 and lam2 < 0.00652
        + 0.01064851 * max(0.0, 895.0 - Q.sum_pt_top50) / 4.070263   # +1.1%  sum_pt_top50 < 895
        + 0.007406702 * max(0.0, Q.z_dr_0p1_0p2 - 0.454) * max(0.0, Q.soft5_z - -0.000103) / 3.685339e-05   # +0.7%  z_dr_0p1_0p2 > 0.454 and soft5_z > -0.000103
        + 0.006516267 * max(0.0, 6.81 - Q.log_sum_pt) * max(0.0, 0.0896 - Q.dr_13) / 0.0001257877   # +0.7%  log_sum_pt < 6.81 and dr_13 < 0.0896
        + 0.006216417 * max(0.0, 983.0 - Q.sum_pt_top50) * max(0.0, 0.00143 - Q.C3) / 0.003616921   # +0.6%  sum_pt_top50 < 983 and C3 < 0.00143
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 11.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.15971 * (0.5492973
        - 0.1493285 * max(0.0, 0.0417 - Q.e2) / 0.0140039   # -14.9%  e2 < 0.0417
        - 0.1366185 * max(0.0, Q.LHA - 0.196) / 0.08331275   # -13.7%  LHA > 0.196
        - 0.1092199 * max(0.0, 0.12 - Q.sum_z_dr) / 0.05515218   # -10.9%  sum_z_dr < 0.12
        - 0.07476745 * max(0.0, Q.sum_pt_top5 - 402.0) / 200.5729   # -7.5%  sum_pt_top5 > 402
        - 0.05447848 * max(0.0, Q.sum_z_dr - 0.0984) / 0.008138745   # -5.4%  sum_z_dr > 0.0984
        + 0.05311754 * max(0.0, Q.LHA - 0.334) / 0.01368999   # +5.3%  LHA > 0.334
        - 0.041369 * max(0.0, 0.0242 - Q.sum_z_dr2_top5) / 0.01869094   # -4.1%  sum_z_dr2_top5 < 0.0242
        + 0.03544504 * max(0.0, 5.77e-05 - Q.e3) / 2.104024e-05   # +3.5%  e3 < 5.77e-05
        + 0.03454586 * max(0.0, Q.e2 - 0.0416) / 0.003212682   # +3.5%  e2 > 0.0416
        - 0.03397065 * max(0.0, Q.pt_entropy - 2.07) / 0.8083214   # -3.4%  pt_entropy > 2.07
        + 0.03299983 * max(0.0, 2.44 - Q.D2) / 0.603719   # +3.3%  D2 < 2.44
        + 0.02910851 * max(0.0, 19.7 - Q.n_dr_0p2_0p4) / 11.43812   # +2.9%  n_dr_0p2_0p4 < 19.7
        - 0.02878719 * max(0.0, Q.e2 - 0.0294) / 0.00787394   # -2.9%  e2 > 0.0294
        + 0.027003 * max(0.0, 0.00427 - Q.sum_z_dr2_top2) / 0.002078247   # +2.7%  sum_z_dr2_top2 < 0.00427
        + 0.01745263 * max(0.0, Q.sum_z_dr - 0.0975) * max(0.0, 1160.0 - Q.sum_pt_top50) / 1.521612   # +1.7%  sum_z_dr > 0.0975 and sum_pt_top50 < 1160
        - 0.01717882 * max(0.0, 0.12 - Q.z_dr_0p1_0p2) / 0.04710335   # -1.7%  z_dr_0p1_0p2 < 0.12
        - 0.01657199 * max(0.0, 1.91 - Q.D2_b2) / 0.6489074   # -1.7%  D2_b2 < 1.91
        + 0.01472873 * max(0.0, Q.psi_0p1 - 0.916) / 0.01778878   # +1.5%  psi_0p1 > 0.916
        - 0.01381318 * max(0.0, 0.203 - Q.tau21_b2) / 0.03844168   # -1.4%  tau21_b2 < 0.203
        + 0.01225407 * max(0.0, 889.0 - Q.sum_pt_top20) / 35.51998   # +1.2%  sum_pt_top20 < 889
        - 0.01065757 * max(0.0, Q.psi_0p3 - 0.998) * max(0.0, Q.eccentricity - 0.526) / 0.0002162463   # -1.1%  psi_0p3 > 0.998 and eccentricity > 0.526
        - 0.009548006 * max(0.0, 0.0295 - Q.M3) / 0.007556951   # -1.0%  M3 < 0.0295
        - 0.009073205 * max(0.0, Q.sum_z_dr - 0.139) / 0.001993196   # -0.9%  sum_z_dr > 0.139
        - 0.008615422 * max(0.0, Q.sj2_dr - 0.111) * max(0.0, Q.M2 - 0.041) / 0.001807249   # -0.9%  sj2_dr > 0.111 and M2 > 0.041
        - 0.004797437 * max(0.0, Q.sum_pt_top5 - 793.0) / 14.2768   # -0.5%  sum_pt_top5 > 793
        + 0.004449521 * max(0.0, 0.109 - Q.sum_z_dr) * max(0.0, 1010.0 - Q.sum_pt_top50) / 0.754641   # +0.4%  sum_z_dr < 0.109 and sum_pt_top50 < 1010
        - 0.00434259 * max(0.0, Q.tau1 - 0.195) * max(0.0, Q.sum_pt - 1170.0) / 0.008908466   # -0.4%  tau1 > 0.195 and sum_pt > 1170
        - 0.004335247 * max(0.0, 6.84 - Q.log_sum_pt) / 0.00664562   # -0.4%  log_sum_pt < 6.84
        + 0.003888794 * max(0.0, Q.n_dr_0p1_0p2 - 21.4) / 1.327151   # +0.4%  n_dr_0p1_0p2 > 21.4
        + 0.003555431 * max(0.0, 0.119 - Q.sum_z_dr) * max(0.0, Q.pt1_dr01 - 6.63) / 0.08496272   # +0.4%  sum_z_dr < 0.119 and pt1_dr01 > 6.63
        - 0.002160653 * max(0.0, Q.tau1 - 0.197) / 0.0007442059   # -0.2%  tau1 > 0.197
        - 0.001817183 * max(0.0, Q.e2 - 0.0278) * max(0.0, Q.soft1_pt - 1.12) / 0.001536307   # -0.2%  e2 > 0.0278 and soft1_pt > 1.12
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 8.562;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.562075 * (-0.03527182
        + 0.1867241 * max(0.0, 0.0876 - Q.sum_z_dr) / 0.02875442   # +18.7%  sum_z_dr < 0.0876
        - 0.1451568 * max(0.0, 0.307 - Q.LHA) / 0.06541283   # -14.5%  LHA < 0.307
        - 0.1295474 * max(0.0, 0.00615 - Q.sum_z_dr2_top15) / 0.002084952   # -13.0%  sum_z_dr2_top15 < 0.00615
        + 0.126551 * max(0.0, 0.00805 - Q.sum_z_dr2_top15) / 0.00338606   # +12.7%  sum_z_dr2_top15 < 0.00805
        - 0.09566148 * max(0.0, 0.0434 - Q.e2) / 0.01533822   # -9.6%  e2 < 0.0434
        + 0.07120442 * max(0.0, 0.0245 - Q.e2) / 0.004293363   # +7.1%  e2 < 0.0245
        + 0.05168537 * max(0.0, 9.68 - Q.n_dr_0p2_0p4) * max(0.0, 34.6 - Q.n_dr_0p1_0p2) / 88.86225   # +5.2%  n_dr_0p2_0p4 < 9.68 and n_dr_0p1_0p2 < 34.6
        + 0.03924329 * max(0.0, 0.00469 - Q.sum_z_dr2_top10) / 0.001592436   # +3.9%  sum_z_dr2_top10 < 0.00469
        + 0.03754733 * max(0.0, 0.0624 - Q.dr_0) / 0.02399127   # +3.8%  dr_0 < 0.0624
        - 0.02452712 * max(0.0, 9.9 - Q.n_dr_0p2_0p4) * max(0.0, 0.00609 - Q.sum_z_dr2_top10) / 0.008677811   # -2.5%  n_dr_0p2_0p4 < 9.9 and sum_z_dr2_top10 < 0.00609
        - 0.02376264 * max(0.0, 3.66e-05 - Q.e3) / 9.463139e-06   # -2.4%  e3 < 3.66e-05
        + 0.02243334 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # +2.2%  psi_0p3 > 0.99
        + 0.01422307 * max(0.0, 20.6 - Q.n_dr_0p1_0p2) * max(0.0, 0.562 - Q.planar_flow) / 1.585664   # +1.4%  n_dr_0p1_0p2 < 20.6 and planar_flow < 0.562
        - 0.01318258 * max(0.0, 0.0926 - Q.sum_z_dr) * max(0.0, Q.tau4 - 0.00981) / 0.0001396909   # -1.3%  sum_z_dr < 0.0926 and tau4 > 0.00981
        + 0.009469726 * max(0.0, Q.psi_0p2 - 0.995) / 0.0009804172   # +0.9%  psi_0p2 > 0.995
        - 0.009080295 * max(0.0, 0.171 - Q.sj3_dr_max) / 0.01203501   # -0.9%  sj3_dr_max < 0.171
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 10.66;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.66478 * (-0.060667
        - 0.203233 * max(0.0, Q.sum_pt - 906.0) / 141.6624   # -20.3%  sum_pt > 906
        + 0.1251704 * max(0.0, 0.0609 - Q.sum_z_dr) / 0.01259353   # +12.5%  sum_z_dr < 0.0609
        - 0.08092083 * max(0.0, 0.0252 - Q.e2) / 0.004566151   # -8.1%  e2 < 0.0252
        + 0.07321646 * max(0.0, 0.00582 - Q.sum_z_dr2_top15) / 0.001895236   # +7.3%  sum_z_dr2_top15 < 0.00582
        - 0.0719273 * max(0.0, 0.244 - Q.LHA) / 0.03292226   # -7.2%  LHA < 0.244
        + 0.06306006 * max(0.0, 4.57e-05 - Q.e3) / 1.386642e-05   # +6.3%  e3 < 4.57e-05
        + 0.05735438 * max(0.0, 0.0507 - Q.sum_z_dr) * max(0.0, Q.log_sum_pt - 6.81) / 0.001332618   # +5.7%  sum_z_dr < 0.0507 and log_sum_pt > 6.81
        + 0.0480356 * max(0.0, 10.1 - Q.n_dr_0p2_0p4) / 3.880977   # +4.8%  n_dr_0p2_0p4 < 10.1
        + 0.04246192 * max(0.0, Q.psi_0p2 - 0.972) * max(0.0, 21.7 - Q.n_dr_0p1_0p2) / 0.1524737   # +4.2%  psi_0p2 > 0.972 and n_dr_0p1_0p2 < 21.7
        + 0.03910031 * max(0.0, Q.psi_0p3 - 0.989) * max(0.0, 0.00655 - Q.sum_z_dr2_top15) / 1.591588e-05   # +3.9%  psi_0p3 > 0.989 and sum_z_dr2_top15 < 0.00655
        - 0.03592132 * max(0.0, 0.00213 - Q.sum_z_dr2_top15) / 0.0004309257   # -3.6%  sum_z_dr2_top15 < 0.00213
        - 0.02530601 * max(0.0, 0.264 - Q.sj3_dr_max) * max(0.0, Q.sum_z_dr2_top10 - 0.00487) / 3.243785e-05   # -2.5%  sj3_dr_max < 0.264 and sum_z_dr2_top10 > 0.00487
        - 0.02397192 * max(0.0, 0.185 - Q.LHA) / 0.0144438   # -2.4%  LHA < 0.185
        + 0.02293554 * max(0.0, 0.204 - Q.sj3_dr_max) / 0.0230757   # +2.3%  sj3_dr_max < 0.204
        - 0.02030259 * max(0.0, 0.154 - Q.sj3_dr_max) / 0.008660906   # -2.0%  sj3_dr_max < 0.154
        - 0.01942104 * max(0.0, 0.0589 - Q.M2) / 0.006881097   # -1.9%  M2 < 0.0589
        + 0.01386642 * max(0.0, Q.LHA - 0.314) * max(0.0, Q.log_sum_pt - 6.85) / 0.001356719   # +1.4%  LHA > 0.314 and log_sum_pt > 6.85
        + 0.0109629 * max(0.0, 1160.0 - Q.sum_pt) * max(0.0, 0.0594 - Q.M2) / 0.9063325   # +1.1%  sum_pt < 1160 and M2 < 0.0594
        - 0.009259792 * max(0.0, Q.psi_0p3 - 0.989) * max(0.0, 0.00762 - Q.C3) / 1.662519e-05   # -0.9%  psi_0p3 > 0.989 and C3 < 0.00762
        + 0.008998176 * max(0.0, Q.pt_dispersion - 0.324) / 0.04049095   # +0.9%  pt_dispersion > 0.324
        + 0.004574081 * max(0.0, Q.e2 - 0.0575) / 0.0009381068   # +0.5%  e2 > 0.0575
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 8.494;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.494229 * (0.2495812
        - 0.1793187 * max(0.0, 6.96 - Q.log_sum_pt) / 0.04364396   # -17.9%  log_sum_pt < 6.96
        + 0.1307632 * max(0.0, 1050.0 - Q.sum_pt_top50) / 45.70915   # +13.1%  sum_pt_top50 < 1050
        + 0.1237678 * max(0.0, 0.000531 - Q.e3) / 0.000436229   # +12.4%  e3 < 0.000531
        - 0.1040185 * max(0.0, Q.sum_z_dr - 0.097) / 0.008414826   # -10.4%  sum_z_dr > 0.097
        - 0.07908933 * max(0.0, Q.sum_z_dr - 0.086) / 0.01078335   # -7.9%  sum_z_dr > 0.086
        + 0.07001437 * max(0.0, Q.LHA - 0.32) / 0.01679995   # +7.0%  LHA > 0.32
        - 0.06881833 * max(0.0, 0.000199 - Q.e3) / 0.0001346909   # -6.9%  e3 < 0.000199
        + 0.06835965 * max(0.0, Q.sum_z_dr - 0.0965) * max(0.0, 1180.0 - Q.sum_pt) / 1.540219   # +6.8%  sum_z_dr > 0.0965 and sum_pt < 1180
        - 0.02620737 * max(0.0, 0.000183 - Q.e3) * max(0.0, 41.1 - Q.pt_9) / 0.001855095   # -2.6%  e3 < 0.000183 and pt_9 < 41.1
        - 0.01841598 * max(0.0, 6.9 - Q.log_sum_pt) / 0.01461958   # -1.8%  log_sum_pt < 6.9
        - 0.01441086 * max(0.0, Q.sum_z_dr - 0.125) / 0.003675951   # -1.4%  sum_z_dr > 0.125
        + 0.01415999 * max(0.0, Q.e2 - 0.0411) / 0.003331805   # +1.4%  e2 > 0.0411
        + 0.01406484 * max(0.0, 957.0 - Q.sum_pt_top50) / 9.634675   # +1.4%  sum_pt_top50 < 957
        - 0.01154994 * max(0.0, Q.sum_z_dr - 0.156) / 0.0006860686   # -1.2%  sum_z_dr > 0.156
        + 0.01105054 * max(0.0, Q.n_dr_0p1_0p2 - 17.4) / 2.072093   # +1.1%  n_dr_0p1_0p2 > 17.4
        + 0.01057501 * max(0.0, Q.sum_z_dr2_top5 - 0.0053) / 0.002916447   # +1.1%  sum_z_dr2_top5 > 0.0053
        - 0.009551242 * max(0.0, Q.sum_pt - 1150.0) / 16.03368   # -1.0%  sum_pt > 1150
        - 0.00940171 * max(0.0, 1070.0 - Q.sum_pt) * max(0.0, Q.sd_zg - 0.152) / 6.884507   # -0.9%  sum_pt < 1070 and sd_zg > 0.152
        - 0.00734915 * max(0.0, Q.sum_z_dr2_top15 - 0.00359) * max(0.0, Q.soft6_pt - 1.66) / 0.002547974   # -0.7%  sum_z_dr2_top15 > 0.00359 and soft6_pt > 1.66
        - 0.005774629 * max(0.0, 712.0 - Q.sum_pt_top15) / 12.17147   # -0.6%  sum_pt_top15 < 712
        - 0.005546906 * max(0.0, Q.sum_z_dr - 0.139) * max(0.0, Q.soft6_pt - 3.21) / 0.0003020301   # -0.6%  sum_z_dr > 0.139 and soft6_pt > 3.21
        + 0.004890359 * max(0.0, 1180.0 - Q.sum_pt) * max(0.0, 6.81 - Q.log_sum_pt) / 2.197875   # +0.5%  sum_pt < 1180 and log_sum_pt < 6.81
        - 0.004290502 * max(0.0, 0.000565 - Q.e3) * max(0.0, 0.992 - Q.psi_0p3) / 2.336186e-06   # -0.4%  e3 < 0.000565 and psi_0p3 < 0.992
        - 0.004034118 * max(0.0, Q.sum_z_dr - 0.141) * max(0.0, Q.soft6_pt - 4.23) / 9.137792e-05   # -0.4%  sum_z_dr > 0.141 and soft6_pt > 4.23
        + 0.002928197 * max(0.0, Q.sum_z_dr - 0.138) * max(0.0, Q.soft6_z - 0.00296) / 4.024721e-07   # +0.3%  sum_z_dr > 0.138 and soft6_z > 0.00296
        + 0.001648868 * max(0.0, Q.sum_z_dr - 0.157) * max(0.0, Q.n_pt_above_50 - 4.84) / 0.0002465821   # +0.2%  sum_z_dr > 0.157 and n_pt_above_50 > 4.84
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 17.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.62742 * (0.1804007
        - 0.255436 * max(0.0, Q.sum_z_dr - 0.121) / 0.00424781   # -25.5%  sum_z_dr > 0.121
        - 0.1263381 * max(0.0, 0.0702 - Q.sum_z_dr) / 0.01713088   # -12.6%  sum_z_dr < 0.0702
        - 0.0998556 * max(0.0, 0.152 - Q.tau1) / 0.07069064   # -10.0%  tau1 < 0.152
        - 0.07720875 * max(0.0, Q.sum_z_dr - 0.0873) / 0.01046916   # -7.7%  sum_z_dr > 0.0873
        + 0.06606144 * max(0.0, 0.26 - Q.LHA) / 0.03960861   # +6.6%  LHA < 0.26
        - 0.06167505 * max(0.0, Q.sum_z_dr - 0.12) * max(0.0, 0.398 - Q.max_dr) / 0.0001295795   # -6.2%  sum_z_dr > 0.12 and max_dr < 0.398
        - 0.04890711 * max(0.0, 0.38 - Q.sj3_dr_max) / 0.1394994   # -4.9%  sj3_dr_max < 0.38
        + 0.04749702 * max(0.0, 0.346 - Q.tau21_b2) / 0.1110411   # +4.7%  tau21_b2 < 0.346
        + 0.03376405 * max(0.0, Q.sd_rg - 0.158) / 0.03629105   # +3.4%  sd_rg > 0.158
        - 0.03270681 * max(0.0, 0.0773 - Q.C2) / 0.02260929   # -3.3%  C2 < 0.0773
        + 0.02628074 * max(0.0, Q.z_dr_0_0p05 - 0.617) / 0.1209561   # +2.6%  z_dr_0_0p05 > 0.617
        - 0.02496146 * max(0.0, 0.342 - Q.tau21_b2) * max(0.0, 1150.0 - Q.sum_pt_top50) / 13.33352   # -2.5%  tau21_b2 < 0.342 and sum_pt_top50 < 1150
        - 0.02424682 * max(0.0, Q.sj3_dr_max - 0.262) / 0.03994476   # -2.4%  sj3_dr_max > 0.262
        - 0.02246722 * max(0.0, Q.sd_rg - 0.201) / 0.02129243   # -2.2%  sd_rg > 0.201
        + 0.01713356 * max(0.0, 0.145 - Q.sd_rg) / 0.0475623   # +1.7%  sd_rg < 0.145
        - 0.007807294 * max(0.0, Q.n_dr_0p2_0p4 - 11.0) / 2.543854   # -0.8%  n_dr_0p2_0p4 > 11
        - 0.006602401 * max(0.0, Q.psi_0p3 - 0.956) * max(0.0, 184.0 - Q.pt_1) / 1.972599   # -0.7%  psi_0p3 > 0.956 and pt_1 < 184
        + 0.006154297 * max(0.0, Q.psi_0p3 - 0.995) * max(0.0, Q.pt_8 - 9.3) / 0.04999281   # +0.6%  psi_0p3 > 0.995 and pt_8 > 9.3
        - 0.004527803 * max(0.0, 0.315 - Q.tau21_b2) * max(0.0, 0.00759 - Q.sum_z_dr2_top15) / 0.0001350482   # -0.5%  tau21_b2 < 0.315 and sum_z_dr2_top15 < 0.00759
        + 0.00363278 * max(0.0, Q.sum_z_dr - 0.0753) * max(0.0, 0.386 - Q.max_dr) / 0.0003518492   # +0.4%  sum_z_dr > 0.0753 and max_dr < 0.386
        - 0.003255451 * max(0.0, Q.sum_z_dr2_top10 - -0.000908) * max(0.0, 36.8 - Q.n_real_top40) / 0.009911089   # -0.3%  sum_z_dr2_top10 > -0.000908 and n_real_top40 < 36.8
        + 0.002397812 * max(0.0, 0.0718 - Q.C2) * max(0.0, Q.pt_3 - 66.9) / 0.2692181   # +0.2%  C2 < 0.0718 and pt_3 > 66.9
        + 0.001082438 * max(0.0, Q.sum_z_dr - 0.0854) * max(0.0, 36.2 - Q.n_real_top40) / 0.000674226   # +0.1%  sum_z_dr > 0.0854 and n_real_top40 < 36.2
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 5.098;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.097894 * (0.09396038
        - 0.1865469 * max(0.0, 0.281 - Q.sd_rg) / 0.1474413   # -18.7%  sd_rg < 0.281
        + 0.134514 * max(0.0, 0.187 - Q.sd_rg) / 0.07195571   # +13.5%  sd_rg < 0.187
        + 0.1191176 * max(0.0, 0.00819 - Q.sum_z_dr2_top5) / 0.004432474   # +11.9%  sum_z_dr2_top5 < 0.00819
        + 0.1102639 * max(0.0, Q.sum_pt_top30 - 838.0) / 160.6038   # +11.0%  sum_pt_top30 > 838
        - 0.07837838 * max(0.0, Q.log_sum_pt - 6.92) / 0.04530211   # -7.8%  log_sum_pt > 6.92
        - 0.07510118 * max(0.0, Q.sj2_dr - 0.137) / 0.07037828   # -7.5%  sj2_dr > 0.137
        - 0.06444268 * max(0.0, 0.0324 - Q.z_dr_0p2_0p4) / 0.01535149   # -6.4%  z_dr_0p2_0p4 < 0.0324
        + 0.05822571 * max(0.0, Q.sj2_dr - 0.216) / 0.0267413   # +5.8%  sj2_dr > 0.216
        - 0.04758511 * max(0.0, Q.n_dr_0p2_0p4 - 7.11) / 4.043063   # -4.8%  n_dr_0p2_0p4 > 7.11
        - 0.0336736 * max(0.0, 0.00107 - Q.sum_z_dr2_top3) / 0.0002959731   # -3.4%  sum_z_dr2_top3 < 0.00107
        + 0.02759479 * max(0.0, 0.0897 - Q.z_dr_0p1_0p2) / 0.03119186   # +2.8%  z_dr_0p1_0p2 < 0.0897
        - 0.02624304 * max(0.0, 2.89e-05 - Q.e3) / 6.49438e-06   # -2.6%  e3 < 2.89e-05
        + 0.02416539 * max(0.0, Q.log_sum_pt - 7.02) / 0.01595759   # +2.4%  log_sum_pt > 7.02
        - 0.008582182 * max(0.0, Q.tau1 - 0.186) / 0.001397797   # -0.9%  tau1 > 0.186
        - 0.005565582 * max(0.0, Q.sj2_dr - 0.212) * max(0.0, 0.0292 - Q.dr_4) / 0.0001167603   # -0.6%  sj2_dr > 0.212 and dr_4 < 0.0292
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6537282037815126, 2.983539705882353, 0.22111911764705883, 0.3984924369747899, 0.7650998949579831, 1.0666, 0.5143669117647058, 0.4556780462184874, 1.2580367647058823, 0.9911646008403361, 1.1818110294117647, 0.7845697478991597, 0.647878781512605, 1.5656498949579831, 0.2533219012605042, 0.38710682773109245]
T = [3.9674216649159657, 2.526993462775735, 4.328753131565127, 4.4002942423844535, 4.089002639180673]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +47%, n4 -17%, n9 +13%, n5 -8%, n3 -8%, n12 +4% ...
            + 0.4700061 * h[1] / H_AVG[1]
            - 0.1687399 * h[4] / H_AVG[4]
            + 0.1288165 * h[9] / H_AVG[9]
            - 0.08401237 * h[5] / H_AVG[5]
            - 0.07533087 * h[3] / H_AVG[3]
            + 0.03827337 * h[12] / H_AVG[12]
            + 0.02025745 * h[6] / H_AVG[6]
            + 0.009909118 * h[8] / H_AVG[8]
            - 0.004654357 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -32%, n1 -22%, n9 +22%, n12 +9%, n11 -8%, n6 +4% ...
            - 0.321694 * h[4] / H_AVG[4]
            - 0.2213752 * h[1] / H_AVG[1]
            + 0.2206298 * h[9] / H_AVG[9]
            + 0.08813174 * h[12] / H_AVG[12]
            - 0.07761889 * h[11] / H_AVG[11]
            + 0.04452634 * h[6] / H_AVG[6]
            + 0.01093786 * h[2] / H_AVG[2]
            + 0.00777874 * h[8] / H_AVG[8]
            - 0.007307418 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +14%, n0 +11%, n11 +11%, n14 -8%, n7 -7% ...
            - 0.2542954 * h[8] / H_AVG[8]
            + 0.1424494 * h[5] / H_AVG[5]
            + 0.113265 * h[0] / H_AVG[0]
            + 0.1076149 * h[11] / H_AVG[11]
            - 0.08046604 * h[14] / H_AVG[14]
            - 0.06579234 * h[7] / H_AVG[7]
            - 0.0608029 * h[12] / H_AVG[12]
            + 0.06075724 * h[4] / H_AVG[4]
            - 0.05008769 * h[9] / H_AVG[9]
            + 0.04027498 * h[3] / H_AVG[3]
            - 0.01676754 * h[15] / H_AVG[15]
            + 0.007426603 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -27%, n0 -20%, n5 +17%, n6 -11%, n7 +9%, n15 +5% ...
            - 0.2680297 * h[8] / H_AVG[8]
            - 0.2042764 * h[0] / H_AVG[0]
            + 0.1666451 * h[5] / H_AVG[5]
            - 0.1132408 * h[6] / H_AVG[6]
            + 0.09384787 * h[7] / H_AVG[7]
            + 0.04948478 * h[15] / H_AVG[15]
            + 0.04601104 * h[12] / H_AVG[12]
            + 0.03962018 * h[3] / H_AVG[3]
            - 0.01884412 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -35%, n10 +28%, n5 -12%, n8 +6%, n12 -6%, n15 -4% ...
            - 0.3469966 * h[13] / H_AVG[13]
            + 0.2845059 * h[10] / H_AVG[10]
            - 0.1222716 * h[5] / H_AVG[5]
            + 0.06489776 * h[8] / H_AVG[8]
            - 0.05941658 * h[12] / H_AVG[12]
            - 0.03550134 * h[15] / H_AVG[15]
            + 0.03508343 * h[4] / H_AVG[4]
            - 0.03134247 * h[7] / H_AVG[7]
            + 0.01998434 * h[0] / H_AVG[0]
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
