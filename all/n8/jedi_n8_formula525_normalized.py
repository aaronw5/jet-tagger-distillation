"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 13:  14.5%   (on for 87% of jets)
  neuron  9:  12.3%   (on for 63% of jets)
  neuron  7:  10.3%   (on for 60% of jets)
  neuron  3:   8.6%   (on for 23% of jets)
  neuron 10:   8.3%   (on for 89% of jets)
  neuron  6:   7.6%   (on for 37% of jets)
  neuron  2:   7.3%   (on for 90% of jets)
  neuron  5:   7.2%   (on for 60% of jets)
  neuron 11:   6.4%   (on for 77% of jets)
  neuron 14:   3.7%   (on for 26% of jets)
  neuron  0:   3.5%   (on for 37% of jets)
  neuron  4:   3.4%   (on for 67% of jets)
  neuron  1:   3.0%   (on for 58% of jets)
  neuron 15:   2.4%   (on for 26% of jets)
  neuron  8:   1.1%   (on for 31% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 90.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.centroid_offset        distance of the pT centroid from the jet axis
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[-0.15625, 0.0, 0.34375, 0.0, 0.015625], [0.390625, 0.0, -0.03125, 0.125, 0.0], [0.4296875, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.5, -0.5625, 0.0625], [-0.03125, -0.09375, 0.0, 0.078125, 0.125], [-0.1875, 0.046875, 0.0, 0.015625, -0.25], [0.109375, 0.125, -0.3125, -0.375, 0.0], [0.0, 0.0, 0.21875, 0.46875, 0.0], [0.0, 0.0625, -0.25, 0.0, 0.1875], [0.171875, 0.25390625, -0.03125, -0.03125, 0.0], [0.0, -0.125, 0.0, 0.0, 0.375], [0.0, 0.0, 0.375, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.5], [0.0, 0.0, 0.0703125, 0.0546875, -0.40625], [0.0, 0.0, -0.75, 0.375, 0.0], [0.0, 0.0625, -0.6875, -0.15625, 0.0]]
B = [-0.4375, 0.03125, -0.125, -0.09375, 1.34375]
INT_BITS = [3, 5, 4, 4, 5, 5, 4, 4, 3, 4, 5, 4, 5, 4, 3, 3]
FRAC_BITS = [3, 3, 4, 3, 2, 3, 3, 3, 3, 2, 4, 4, 3, 3, 4, 3]


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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_4=pair_mass(0, 4),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_3=z[3] * dr[3],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        phi_0=phi[0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 24.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.81974 * (-0.07823447
        + 0.1385797 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +13.9%  girth2 < 0.01324
        - 0.106482 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -10.6%  girth < 0.08724
        - 0.08269525 * max(0.0, 0.004372139 - Q.width) / 0.00156571   # -8.3%  width < 0.004372
        + 0.07788119 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +7.8%  width < 0.008678
        + 0.06477382 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +6.5%  e2 < 0.02455
        - 0.0596001 * max(0.0, 64.61873 - Q.mass) / 27.57279   # -6.0%  mass < 64.62
        - 0.04864626 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -4.9%  sj3_dr_max > 0.2337
        - 0.04554579 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -4.6%  lam1 < 0.005433
        + 0.03956124 * max(0.0, 0.01882765 - Q.girth2) / 0.01314296   # +4.0%  girth2 < 0.01883
        - 0.03362025 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -3.4%  C2_b2 < 0.001563
        + 0.03194836 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +3.2%  sj3_dr_max < 0.3012
        + 0.03132494 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +3.1%  planar_flow < 0.1484
        + 0.02932793 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +2.9%  lam1 < 0.006507
        - 0.02723155 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -2.7%  mass < 29.64
        - 0.02423963 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -2.4%  log_sum_pt > 6.67
        + 0.02177748 * max(0.0, 0.0002758826 - Q.lam1) / 3.586469e-05   # +2.2%  lam1 < 0.0002759
        + 0.01822813 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.8%  sum_pt_top5 > 687.4
        + 0.01276974 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +1.3%  sj3_dr_max > 0.107
        + 0.01231498 * max(0.0, 21.78408 - Q.mass) / 4.431023   # +1.2%  mass < 21.78
        - 0.01170295 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.01452637 - Q.phi_0) / 0.1160543   # -1.2%  mass < 29.64 and phi_0 < 0.01453
        + 0.009890315 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +1.0%  girth2 < 0.01324 and D2 < 1.002
        - 0.009149547 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.04990367 - Q.centroid_offset) / 0.001732318   # -0.9%  planar_flow < 0.1484 and centroid_offset < 0.0499
        - 0.008799578 * max(0.0, 0.01882765 - Q.girth2) * max(0.0, 46.125 - Q.pt_6) / 0.1159788   # -0.9%  girth2 < 0.01883 and pt_6 < 46.12
        - 0.008794996 * max(0.0, 7.300726e-05 - Q.lam2) / 2.767163e-05   # -0.9%  lam2 < 7.301e-05
        + 0.007726866 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.8%  lam2 < 7.301e-05 and D2_b2 < 0.267
        - 0.006959202 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -0.7%  girth2 < 0.01324 and centroid_offset > 0.01838
        + 0.006408222 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +0.6%  n_dr_0_0p05 > 4
        - 0.00576416 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -0.6%  sum_pt > 901.6
        - 0.005664896 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -0.6%  lam1 < 0.006507 and D2 < 0.8757
        + 0.00399458 * max(0.0, Q.log_sum_pt - 6.670067) * max(0.0, 0.07232166 - Q.dr_4) / 0.001949106   # +0.4%  log_sum_pt > 6.67 and dr_4 < 0.07232
        - 0.003495242 * max(0.0, 6.080494 - Q.log_sum_pt) / 0.006097657   # -0.3%  log_sum_pt < 6.08
        - 0.002406119 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 358.375 - Q.sum_pt_top2) / 2.426548   # -0.2%  planar_flow < 0.1484 and sum_pt_top2 < 358.4
        - 0.0008572615 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.eccentricity - 0.9458207) / 0.1898472   # -0.1%  sum_pt > 901.6 and eccentricity > 0.9458
        + 0.0008045841 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +0.1%  centroid_offset > 0.0499
        + 0.0007711965 * max(0.0, 0.1484197 - Q.planar_flow) * max(0.0, 0.02320757 - Q.z_7) / 2.048782e-05   # +0.1%  planar_flow < 0.1484 and z_7 < 0.02321
        + 0.000261999 * max(0.0, Q.sum_pt - 901.5938) * max(0.0, Q.dr0_6 - 0.2347949) / 0.03872508   # +0.0%  sum_pt > 901.6 and dr0_6 > 0.2348
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.74305 * (0.008865382
        - 0.1916503 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -19.2%  girth2 < 0.008678
        + 0.1352079 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +13.5%  mass_over_sum_pt_sq < 0.005833
        + 0.08517846 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +8.5%  lam1 < 0.005954
        - 0.08067664 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -8.1%  sum_pt_top5 > 531.2
        - 0.0801574 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -8.0%  width < 0.006097
        + 0.06346661 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +6.3%  sum_pt_top5 > 367.6
        + 0.05592733 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +5.6%  log_sum_pt > 6.378
        - 0.05426999 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -5.4%  girth < 0.0717
        + 0.05230959 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +5.2%  log_sum_pt > 6.503
        - 0.02938858 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -2.9%  z_7 < 0.06165
        + 0.02786251 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # +2.8%  lam1 < 0.008376
        + 0.02726839 * max(0.0, Q.z_7 - 0.02320757) / 0.02968057   # +2.7%  z_7 > 0.02321
        + 0.02478256 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +2.5%  log_sum_pt > 6.606
        + 0.01016887 * max(0.0, 0.008678045 - Q.girth2) * max(0.0, Q.centroid_offset - 0.02076709) / 7.418362e-06   # +1.0%  girth2 < 0.008678 and centroid_offset > 0.02077
        - 0.01005449 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -1.0%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        - 0.008459005 * max(0.0, 0.0008722282 - Q.lam1) / 0.0001882079   # -0.8%  lam1 < 0.0008722
        + 0.008186285 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 0.008921136 - Q.mean_phi2) / 0.0001024498   # +0.8%  z_7 < 0.06165 and mean_phi2 < 0.008921
        - 0.007956157 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.8%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
        - 0.00723404 * max(0.0, 0.06164517 - Q.z_7) * max(0.0, 1.679198 - Q.D2) / 0.005760154   # -0.7%  z_7 < 0.06165 and D2 < 1.679
        - 0.00658894 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709) / 7.433676e-06   # -0.7%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.00474945 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, 0.1484197 - Q.planar_flow) / 0.0001369594   # -0.5%  lam1 < 0.008376 and planar_flow < 0.1484
        + 0.004367636 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +0.4%  sj3_dr_max > 0.169
        + 0.004126845 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.sj2_dr - 0.1294903) / 0.1877842   # +0.4%  pt_7 > 34.53 and sj2_dr > 0.1295
        - 0.003945003 * max(0.0, 0.008375572 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.00323438   # -0.4%  lam1 < 0.008376 and n_pt_above_50 > 5
        - 0.003853206 * max(0.0, Q.sj3_dr_max - 0.169029) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.5700307   # -0.4%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 5.744
        + 0.003815106 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, Q.centroid_offset - 0.009480685) / 0.0009010889   # +0.4%  log_sum_pt > 6.378 and centroid_offset > 0.009481
        + 0.00374122 * max(0.0, Q.log_sum_pt - 6.605974) * max(0.0, 1.432482 - Q.D2) / 0.01970563   # +0.4%  log_sum_pt > 6.606 and D2 < 1.432
        - 0.003431484 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 2.955458e-05 - Q.e3) / 7.137805e-05   # -0.3%  pt_7 > 34.53 and e3 < 2.955e-05
        - 0.001175931 * max(0.0, Q.pt_7 - 53.4375) / 0.3316343   # -0.1%  pt_7 > 53.44
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 8.637;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.636979 * (0.2792648
        - 0.2159472 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -21.6%  pt_7 < 53.44
        - 0.1077803 * max(0.0, Q.z_7 - 0.03629544) / 0.01879829   # -10.8%  z_7 > 0.0363
        + 0.1030139 * max(0.0, 0.1079857 - Q.mass_over_sum_pt) / 0.05207765   # +10.3%  mass_over_sum_pt < 0.108
        + 0.09890715 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +9.9%  sum_pt < 813.4
        + 0.08519178 * max(0.0, 62.55 - Q.sj3_pair_mass_max) / 30.55764   # +8.5%  sj3_pair_mass_max < 62.55
        - 0.07033573 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # -7.0%  LHA > 0.1329
        - 0.04190204 * max(0.0, Q.sum_pt - 527.1781) * max(0.0, 0.0001947983 - Q.lam2) / 0.02863163   # -4.2%  sum_pt > 527.2 and lam2 < 0.0001948
        + 0.03939874 * max(0.0, Q.sum_pt - 527.1781) / 199.4719   # +3.9%  sum_pt > 527.2
        + 0.0334282 * max(0.0, 6.605974 - Q.log_sum_pt) / 0.1337726   # +3.3%  log_sum_pt < 6.606
        - 0.03202412 * max(0.0, 36.22941 - Q.mass) / 10.11393   # -3.2%  mass < 36.23
        + 0.0279789 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +2.8%  log_sum_pt < 6.464
        + 0.02753614 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +2.8%  girth2 < 0.003563
        + 0.02212806 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # +2.2%  zdr_0 < 0.02118
        - 0.01563993 * max(0.0, 62.55 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.01096064) / 0.2139385   # -1.6%  sj3_pair_mass_max < 62.55 and centroid_offset > 0.01096
        + 0.01333544 * max(0.0, Q.pt_7 - 43.5) / 1.4368   # +1.3%  pt_7 > 43.5
        + 0.00933327 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.129616 - Q.D2_b2) / 6.295733   # +0.9%  sum_pt_top5 > 752.1 and D2_b2 < 1.13
        - 0.008697597 * max(0.0, 0.00595415 - Q.lam1) * max(0.0, Q.max_dr - 0.08050702) / 4.493327e-05   # -0.9%  lam1 < 0.005954 and max_dr > 0.08051
        - 0.006869335 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # -0.7%  girth < 0.007674
        - 0.006802523 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.27415   # -0.7%  sum_pt_top5 > 752.1
        - 0.006335388 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 0.0174408 - Q.abseta_0) / 0.216283   # -0.6%  sum_pt_top5 > 752.1 and abseta_0 < 0.01744
        + 0.005982701 * max(0.0, Q.log_sum_pt - 6.842717) / 0.007875918   # +0.6%  log_sum_pt > 6.843
        + 0.004850335 * max(0.0, 0.03243272 - Q.z_7) / 0.002061024   # +0.5%  z_7 < 0.03243
        + 0.004844109 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.4474937 - Q.z_dr_0p05_0p1) / 0.00315435   # +0.5%  log_sum_pt > 6.843 and z_dr_0p05_0p1 < 0.4475
        - 0.003872704 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 1.345805 - Q.D2_b2) / 0.002918988   # -0.4%  log_sum_pt > 6.843 and D2_b2 < 1.346
        + 0.003761556 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, 0.0174408 - Q.abseta_0) / 8.283244e-05   # +0.4%  log_sum_pt > 6.843 and abseta_0 < 0.01744
        - 0.002756888 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.3%  sum_pt_top5 > 840
        - 0.001345976 * max(0.0, Q.log_sum_pt - 6.842717) * max(0.0, Q.pt_6 - 41.21875) / 0.06090836   # -0.1%  log_sum_pt > 6.843 and pt_6 > 41.22
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 21.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.91155 * (-0.3461368
        - 0.1612503 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # -16.1%  girth2 > 0.008678
        + 0.1387278 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +13.9%  girth > 0.04082
        + 0.07632909 * max(0.0, 0.008329695 - Q.girth2_top5) / 0.004544146   # +7.6%  girth2_top5 < 0.00833
        + 0.05840716 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +5.8%  mass_over_sum_pt > 0.06814
        - 0.05418997 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # -5.4%  girth2_top5 < 0.007164
        - 0.05223428 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -5.2%  tau1 > 0.05357
        + 0.05021414 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +5.0%  sj2_dr > 0.1873
        + 0.04543019 * max(0.0, Q.max_dr - 0.1027585) / 0.04505724   # +4.5%  max_dr > 0.1028
        + 0.04148826 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +4.1%  lam1 > 0.008376
        + 0.03911119 * max(0.0, Q.girth - 0.07608178) / 0.01059033   # +3.9%  girth > 0.07608
        - 0.03366811 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -3.4%  width > 0.01324
        + 0.03170137 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +3.2%  girth > 0.04082 and log_sum_pt > 6.08
        - 0.02828863 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -2.8%  girth2 < 0.004372
        + 0.01961913 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +2.0%  tau1 > 0.1028
        + 0.01809798 * max(0.0, Q.lam1 - 0.01643375) / 0.0006837082   # +1.8%  lam1 > 0.01643
        - 0.01684348 * max(0.0, Q.mass - 64.61873) / 3.274818   # -1.7%  mass > 64.62
        + 0.0166909 * max(0.0, Q.centroid_offset - 0.01096064) / 0.008813008   # +1.7%  centroid_offset > 0.01096
        + 0.0124831 * max(0.0, Q.sd_mass - 62.55) / 3.348271   # +1.2%  sd_mass > 62.55
        - 0.01047251 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.02673292   # -1.0%  max_dr > 0.1028 and z_dr_0p05_0p1 < 0.846
        + 0.01021194 * max(0.0, Q.width - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997) / 0.0004577253   # +1.0%  width > 0.00752 and sj3_pairmin_over_m > 0.07709
        + 0.009798434 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # +1.0%  e2 > 0.06344
        - 0.009179188 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # -0.9%  sj2_dr > 0.2688
        + 0.00779011 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 0.07861328 - Q.abseta_0) / 0.0002969212   # +0.8%  centroid_offset > 0.01096 and abseta_0 < 0.07861
        - 0.007553261 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.8%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        - 0.007536038 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # -0.8%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        - 0.006371871 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -0.6%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.005605677 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.6%  lam2 > 0.001131 and pt_6 < 56.53
        + 0.004227513 * max(0.0, -0.004664942 - Q.mean_eta) / 0.003754527   # +0.4%  mean_eta < -0.004665
        + 0.004046645 * max(0.0, Q.width - 0.007520088) / 0.002470136   # +0.4%  width > 0.00752
        - 0.003897361 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.pt_6 - 31.90625) / 0.1358885   # -0.4%  mass_over_sum_pt > 0.06814 and pt_6 > 31.91
        - 0.003468919 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.02994202   # -0.3%  centroid_offset > 0.01096 and n_pt_above_50 < 8
        - 0.003034302 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2) / 1.823781   # -0.3%  sd_mass > 62.55 and D2_b2 < 0.9207
        + 0.002917649 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, Q.z_6 - 0.05441452) / 6.527736e-06   # +0.3%  lam2 > 0.001131 and z_6 > 0.05441
        + 0.002333245 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.2%  lam2 > 0.001131
        + 0.002197759 * max(0.0, Q.mean_eta - 0.01772426) / 0.001375178   # +0.2%  mean_eta > 0.01772
        + 0.001487465 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.1%  e2 > 0.06344 and sj2_mass1 > 16.86
        - 0.001355614 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, 0.05268713 - Q.dr_3) / 0.000155932   # -0.1%  sj2_dr > 0.1873 and dr_3 < 0.05269
        - 0.000894358 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.z_top5 - 0.7773372) / 2.699283e-05   # -0.1%  e2 > 0.06344 and z_top5 > 0.7773
        + 0.0007500103 * max(0.0, Q.e2 - 0.06344108) * max(0.0, 0.01257746 - Q.zdr_3) / 1.576515e-06   # +0.1%  e2 > 0.06344 and zdr_3 < 0.01258
        + 9.503949e-05 * max(0.0, Q.sj2_dr - 0.1492731) / 0.04449993   # +0.0%  sj2_dr > 0.1493
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 74.87;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 74.86934 * (-0.02566985
        + 0.3221696 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +32.2%  e2_sq < 0.01166
        - 0.2449499 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -24.5%  mass_over_sum_pt_sq < 0.01166
        - 0.05613249 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -5.6%  girth2 < 0.008678
        - 0.05257949 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -5.3%  sj3_dr_max < 0.2337
        + 0.04576043 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +4.6%  C2 < 0.06729
        + 0.02964365 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +3.0%  N2 < 0.2233
        - 0.02306639 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # -2.3%  e2 < 0.05028
        - 0.01820065 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -1.8%  lam2 < 0.0005373
        - 0.01805304 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -1.8%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.01748353 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +1.7%  width > 0.003563
        + 0.01612493 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +1.6%  girth2_top2 < 0.00764
        - 0.01361282 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.4%  mass_over_sum_pt > 0.09041
        + 0.01361238 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # +1.4%  sj3_dr_max < 0.3456
        + 0.01282075 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +1.3%  e3 < 0.0001869
        - 0.01130687 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # -1.1%  centroid_offset < 0.0499
        + 0.01068839 * max(0.0, 0.001653836 - Q.width) / 0.0004289067   # +1.1%  width < 0.001654
        + 0.01059814 * max(0.0, 0.007164202 - Q.girth2_top5) / 0.003638607   # +1.1%  girth2_top5 < 0.007164
        + 0.0105821 * max(0.0, Q.sd_mass - 44.82259) / 8.759065   # +1.1%  sd_mass > 44.82
        + 0.01048872 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +1.0%  sj3_dr_max < 0.2134
        + 0.009600397 * max(0.0, 0.07283629 - Q.tau1) / 0.02748645   # +1.0%  tau1 < 0.07284
        - 0.009529891 * max(0.0, 0.000537286 - Q.lam2) * max(0.0, 0.1410336 - Q.dr01) / 4.252003e-05   # -1.0%  lam2 < 0.0005373 and dr01 < 0.141
        - 0.007496632 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.7%  sum_pt < 739.5
        - 0.006006156 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.6%  N2 < 0.2233 and pt_7 < 53.44
        + 0.004934608 * max(0.0, Q.max_dr - 0.1598486) / 0.0215652   # +0.5%  max_dr > 0.1598
        - 0.003973066 * max(0.0, Q.sd_mass - 74.57663) / 1.60503   # -0.4%  sd_mass > 74.58
        - 0.003573148 * max(0.0, 37.76455 - Q.mass_top5) / 16.71307   # -0.4%  mass_top5 < 37.76
        - 0.0029099 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549) / 0.1760217   # -0.3%  sd_mass > 44.82 and centroid_offset > 0.001309
        + 0.002751155 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # +0.3%  girth > 0.08724
        - 0.002581498 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass) / 0.3959152   # -0.3%  N2 < 0.2233 and mass < 62.55
        - 0.002389307 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg) / 0.3498594   # -0.2%  sd_mass > 44.82 and sd_zg < 0.2758
        - 0.001625545 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.1218872 - Q.abseta_7) / 0.003384012   # -0.2%  N2 < 0.2233 and abseta_7 < 0.1219
        - 0.001418414 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798) / 6.028305e-06   # -0.1%  girth2_top2 < 0.00764 and C2_b2 > 0.0006436
        - 0.001118699 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3) / 5.535701e-05   # -0.1%  width > 0.003563 and sj3_z3 < 0.1058
        + 0.0009257716 * max(0.0, Q.mass - 76.6557) / 1.546047   # +0.1%  mass > 76.66
        - 0.000778335 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.e2_sq - 0.01165737) / 7.11886e-05   # -0.1%  N2 < 0.2233 and e2_sq > 0.01166
        - 0.0005131417 * max(0.0, Q.max_dr - 0.1598486) * max(0.0, 0.0006435798 - Q.C2_b2) / 1.059262e-06   # -0.1%  max_dr > 0.1598 and C2_b2 < 0.0006436
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.46995 * (-0.07443068
        + 0.1249187 * max(0.0, 0.07148865 - Q.z_7) / 0.02140487   # +12.5%  z_7 < 0.07149
        - 0.09214577 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -9.2%  pt_7 < 37.16
        + 0.07605131 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +7.6%  girth2 < 0.001654 and centroid_offset < 0.02355
        + 0.05924561 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +5.9%  z_7 < 0.0494
        + 0.05874797 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +5.9%  width < 0.003563
        + 0.05497749 * max(0.0, 0.005284669 - Q.e2_sq) / 0.00223735   # +5.5%  e2_sq < 0.005285
        - 0.05101245 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -5.1%  zdr_0 < 0.02118
        - 0.04863011 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1) / 4.027805e-05   # -4.9%  LHA < 0.2161 and lam1 < 0.001504
        + 0.04353275 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +4.4%  LHA < 0.2161
        - 0.04112287 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -4.1%  log_sum_pt > 6.701
        - 0.03969275 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -4.0%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.03826793 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -3.8%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.03466622 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 46.125 - Q.pt_6) / 2161.354   # +3.5%  sum_pt_top5 > 430.8 and pt_6 < 46.12
        + 0.02355745 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3) / 0.60086   # +2.4%  z_7 < 0.07149 and mass_top3 < 40.2
        - 0.02135853 * max(0.0, 0.002074109 - Q.e2_sq) / 0.000670556   # -2.1%  e2_sq < 0.002074
        + 0.01979635 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +2.0%  sum_pt_top5 > 687.4
        + 0.01727893 * max(0.0, 0.001653836 - Q.girth2) / 0.0004289067   # +1.7%  girth2 < 0.001654
        - 0.0164461 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -1.6%  girth2_top3 < 0.002152
        + 0.01570126 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +1.6%  z_7 < 0.02807
        + 0.01558626 * max(0.0, 0.006789738 - Q.centroid_offset) / 0.001012351   # +1.6%  centroid_offset < 0.00679
        - 0.01305315 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # -1.3%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.01291987 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.phi_0 - -0.0297699) / 0.0002791968   # +1.3%  zdr_0 < 0.02118 and phi_0 > -0.02977
        + 0.01202854 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2) / 0.0001347035   # +1.2%  log_sum_pt > 6.701 and dr_2 < 0.01342
        + 0.009902727 * max(0.0, 37.15625 - Q.pt_7) * max(0.0, Q.pt_5 - 24.57812) / 73.42646   # +1.0%  pt_7 < 37.16 and pt_5 > 24.58
        - 0.009674147 * max(0.0, 0.006789738 - Q.centroid_offset) * max(0.0, 56.4375 - Q.pt_5) / 0.01266727   # -1.0%  centroid_offset < 0.00679 and pt_5 < 56.44
        + 0.009348602 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655) / 4.310225   # +0.9%  sum_pt_top5 > 430.8 and sj2_dr > 0.1683
        - 0.007878577 * max(0.0, Q.pair_mass_0_4 - 23.26771) / 0.6641733   # -0.8%  pair_mass_0_4 > 23.27
        + 0.006536961 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 9.030369e-05 - Q.mean_eta2) / 9.99897e-07   # +0.7%  log_sum_pt > 6.701 and mean_eta2 < 9.03e-05
        - 0.005058349 * max(0.0, 0.04939969 - Q.z_7) * max(0.0, 0.009661512 - Q.dr_2) / 1.13159e-05   # -0.5%  z_7 < 0.0494 and dr_2 < 0.009662
        + 0.003551036 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, Q.eccentricity - 0.9704496) / 0.009376301   # +0.4%  pair_mass_0_4 > 23.27 and eccentricity > 0.9704
        + 0.003413936 * max(0.0, 24.57812 - Q.pt_5) / 0.2839763   # +0.3%  pt_5 < 24.58
        - 0.003227279 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, Q.mean_phi - 0.002834884) / 4.214377e-05   # -0.3%  log_sum_pt > 6.701 and mean_phi > 0.002835
        - 0.003207343 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -0.3%  log_sum_pt > 6.896
        - 0.003035902 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.02270492 - Q.dr_2) / 4.344748e-05   # -0.3%  log_sum_pt > 6.896 and dr_2 < 0.0227
        + 0.002574652 * max(0.0, Q.pair_mass_0_4 - 23.26771) * max(0.0, 0.01423545 - Q.mean_eta2) / 0.004884555   # +0.3%  pair_mass_0_4 > 23.27 and mean_eta2 < 0.01424
        - 0.001852187 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001534584   # -0.2%  LHA < 0.2161 and n_dr_0p2_0p4 > 0
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 32.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 32.20413 * (-0.05657405
        - 0.1818604 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -18.2%  girth2 < 0.008678
        - 0.095651 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -9.6%  width < 0.01324
        + 0.08342464 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +8.3%  lam1 < 0.012
        + 0.06663265 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +6.7%  tau1 < 0.1136
        + 0.05771445 * max(0.0, 0.003408389 - Q.lam2) / 0.003036545   # +5.8%  lam2 < 0.003408
        + 0.0526326 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +5.3%  sj3_dr_max < 0.179
        - 0.04509903 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -4.5%  max_dr < 0.1452
        + 0.03432077 * max(0.0, 0.02415398 - Q.C2_b2) / 0.02132459   # +3.4%  C2_b2 < 0.02415
        - 0.03408866 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -3.4%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        + 0.03243252 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +3.2%  centroid_offset > 0.008092
        + 0.03119575 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +3.1%  e3 < 0.0005117
        - 0.03041956 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -3.0%  pt_6 < 39.75
        + 0.02857977 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +2.9%  lam1 < 0.005954
        + 0.02844737 * max(0.0, 45.595 - Q.mass) / 14.73532   # +2.8%  mass < 45.59
        + 0.02824749 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +2.8%  pt_6 < 41.22
        - 0.02178346 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -2.2%  centroid_offset > 0.01838
        + 0.02173254 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +2.2%  pt_6 < 41.22 and log_sum_pt < 6.767
        - 0.01969935 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -2.0%  lam2 < 0.001131
        - 0.01644547 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -1.6%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.01635296 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.6%  sj3_dr_max > 0.1879
        + 0.01128168 * max(0.0, 0.00733008 - Q.lam1) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.003289169   # +1.1%  lam1 < 0.00733 and n_dr_0p2_0p4 < 1
        - 0.00929559 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -0.9%  sj2_dr < 0.1873
        - 0.009132303 * max(0.0, Q.eccentricity - 0.927072) / 0.03232992   # -0.9%  eccentricity > 0.9271
        + 0.008684907 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # +0.9%  e2_sq < 0.003013
        + 0.005351405 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.5%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        + 0.005182458 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.5%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.004583285 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.psi_0p1 - 0.4008925) / 0.003485257   # +0.5%  centroid_offset > 0.008092 and psi_0p1 > 0.4009
        - 0.004419131 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -0.4%  sj3_pair_mass_min > 4.502
        - 0.003606251 * max(0.0, 0.01200373 - Q.lam1) * max(0.0, 0.2534037 - Q.planar_flow) / 0.0006566196   # -0.4%  lam1 < 0.012 and planar_flow < 0.2534
        + 0.00299268 * max(0.0, 615.875 - Q.sum_pt) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 43.23531   # +0.3%  sum_pt < 615.9 and n_dr_0p2_0p4 < 2
        + 0.002202448 * max(0.0, 29.90625 - Q.pt_6) / 1.385454   # +0.2%  pt_6 < 29.91
        + 0.002083067 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # +0.2%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.001736935 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # -0.2%  sj3_dr_min > 0.1278
        + 0.001125318 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.M3 - 0.0782171) / 0.01402652   # +0.1%  pt_6 < 41.22 and M3 > 0.07822
        + 0.001073944 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.02702951 - Q.C2) / 2.39871e-05   # +0.1%  centroid_offset > 0.01838 and C2 < 0.02703
        - 0.0002969427 * max(0.0, Q.mean_eta - 0.02644207) / 0.0007397545   # -0.0%  mean_eta > 0.02644
        - 0.0001912295 * max(0.0, Q.mean_eta - 0.02644207) * max(0.0, Q.M3 - 0.06688759) / 9.396674e-06   # -0.0%  mean_eta > 0.02644 and M3 > 0.06689
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 35.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.63917 * (0.2268976
        - 0.08437656 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -8.4%  mass_over_sum_pt > 0.0111
        - 0.08305082 * max(0.0, Q.mass_over_sum_pt - 0.1079857) / 0.005364315   # -8.3%  mass_over_sum_pt > 0.108
        - 0.07890582 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -7.9%  mass_over_sum_pt > 0.09041
        - 0.06316961 * max(0.0, Q.width - 0.005590289) / 0.003084256   # -6.3%  width > 0.00559
        + 0.06243051 * max(0.0, Q.girth2 - 0.01323868) / 0.001442684   # +6.2%  girth2 > 0.01324
        + 0.0586227 * max(0.0, Q.girth2 - 0.001653836) / 0.005220304   # +5.9%  girth2 > 0.001654
        - 0.0552267 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -5.5%  girth < 0.08724
        - 0.05096068 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -5.1%  girth2 > 0.00752
        + 0.05091368 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +5.1%  mass_over_sum_pt > 0.08475
        - 0.04984673 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -5.0%  sj2_dr < 0.1873
        + 0.04656018 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +4.7%  mass_over_sum_pt > 0.07269
        - 0.04336274 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -4.3%  lam1 < 0.008376
        - 0.03515409 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -3.5%  e3 < 8.148e-05
        + 0.0320873 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +3.2%  sj2_dr < 0.1592
        + 0.01894116 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.9%  mass > 36.23
        - 0.01857933 * max(0.0, Q.girth2 - 0.004372139) / 0.003638805   # -1.9%  girth2 > 0.004372
        - 0.01751799 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.004032342 - Q.C2_b2) / 2.736263e-05   # -1.8%  centroid_offset < 0.02077 and C2_b2 < 0.004032
        + 0.0139157 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.4%  tau1 < 0.05357
        + 0.01261021 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +1.3%  sj2_dr < 0.1295
        + 0.01233044 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +1.2%  LHA < 0.3033
        - 0.01196241 * max(0.0, 0.04081947 - Q.girth) / 0.009176315   # -1.2%  girth < 0.04082
        + 0.01000874 * max(0.0, Q.z_7 - 0.03243272) / 0.02180051   # +1.0%  z_7 > 0.03243
        - 0.00933802 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # -0.9%  girth2 < 0.0009641
        + 0.008518716 * max(0.0, 0.1585582 - Q.z_dr_0p1_0p2) / 0.09532154   # +0.9%  z_dr_0p1_0p2 < 0.1586
        - 0.008420597 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.8%  e2 > 0.03556
        - 0.007693039 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.8%  mass > 76.66
        + 0.006347972 * max(0.0, 0.08723651 - Q.girth) * max(0.0, 0.004032342 - Q.C2_b2) / 0.0001217265   # +0.6%  girth < 0.08724 and C2_b2 < 0.004032
        - 0.006089406 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.6%  centroid_offset > 0.03776
        - 0.005198571 * max(0.0, 0.02076709 - Q.centroid_offset) / 0.008334291   # -0.5%  centroid_offset < 0.02077
        + 0.005102017 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.sd_mass - 38.43971) / 1.25962   # +0.5%  planar_flow < 0.195 and sd_mass > 38.44
        + 0.00474453 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, 0.02656143 - Q.tau21_b2) / 4.805781e-05   # +0.5%  centroid_offset < 0.02077 and tau21_b2 < 0.02656
        + 0.004705462 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +0.5%  girth2 > 0.004372 and planar_flow < 0.195
        + 0.004616844 * max(0.0, 0.02076709 - Q.centroid_offset) * max(0.0, Q.sum_pt_top3 - 331.25) / 1.710819   # +0.5%  centroid_offset < 0.02077 and sum_pt_top3 > 331.2
        + 0.004144184 * max(0.0, Q.sd_rg - 0.2787955) / 0.005121863   # +0.4%  sd_rg > 0.2788
        + 0.003348661 * max(0.0, Q.centroid_offset - 0.03117077) / 0.002505019   # +0.3%  centroid_offset > 0.03117
        + 0.003070441 * max(0.0, 0.001056655 - Q.girth2_top2) / 0.0003002514   # +0.3%  girth2_top2 < 0.001057
        - 0.002809168 * max(0.0, 29.04219 - Q.pt_7) / 2.179984   # -0.3%  pt_7 < 29.04
        - 0.00261881 * max(0.0, Q.sd_rg - 0.2037854) / 0.01566501   # -0.3%  sd_rg > 0.2038
        - 0.001029711 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.1690249   # -0.1%  planar_flow < 0.195 and pt_6 < 35.28
        - 0.0007536596 * max(0.0, Q.centroid_offset - 0.03117077) * max(0.0, Q.n_pt_above_50 - 4.0) / 0.00188116   # -0.1%  centroid_offset > 0.03117 and n_pt_above_50 > 4
        + 0.000689581 * max(0.0, Q.girth2 - 0.01323868) * max(0.0, Q.eccentricity - 0.9458207) / 2.125219e-05   # +0.1%  girth2 > 0.01324 and eccentricity > 0.9458
        - 0.0002265051 * max(0.0, Q.centroid_offset - 0.03776099) * max(0.0, Q.pt_4 - 81.375) / 0.0001936307   # -0.0%  centroid_offset > 0.03776 and pt_4 > 81.38
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 15.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.23089 * (-0.04607744
        - 0.1446064 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -14.5%  girth < 0.06109
        + 0.1068559 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +10.7%  girth2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.0906913 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width) / 4.808955e-05   # +9.1%  tau1 < 0.05357 and width < 0.003563
        - 0.07197644 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -7.2%  LHA < 0.1967
        - 0.05423442 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.4%  sj3_dr_max < 0.1426
        - 0.05205381 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width) / 0.02261167   # -5.2%  mass < 29.64 and width < 0.003563
        + 0.04950589 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +5.0%  girth2 < 0.00502
        + 0.04691947 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +4.7%  girth2 < 0.006679 and centroid_offset < 0.02355
        + 0.04502368 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +4.5%  width < 0.003563
        - 0.03445342 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -3.4%  log_sum_pt > 6.701
        - 0.03235284 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # -3.2%  tau1 < 0.05357
        + 0.03174593 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +3.2%  girth < 0.06109 and lam2 < 0.0001948
        + 0.0244799 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2) / 0.0003703114   # +2.4%  sj3_dr_max < 0.1986 and girth2 < 0.006679
        - 0.02426861 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.4%  girth2 < 0.00502 and centroid_offset > 0.00679
        + 0.02133599 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # +2.1%  mass_over_sum_pt < 0.03319
        + 0.01980382 * max(0.0, 0.0001330621 - Q.lam2) * max(0.0, 4.721224 - Q.D2_b2) / 0.0002340602   # +2.0%  lam2 < 0.0001331 and D2_b2 < 4.721
        - 0.01741857 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -1.7%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.01645629 * max(0.0, 0.1986272 - Q.sj3_dr_max) / 0.06578578   # +1.6%  sj3_dr_max < 0.1986
        - 0.0147862 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 4.721224 - Q.D2_b2) / 0.008595177   # -1.5%  girth2 < 0.006679 and D2_b2 < 4.721
        + 0.01474528 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +1.5%  sum_pt_top5 > 687.4
        - 0.01416634 * max(0.0, 49.6681 - Q.mass) / 17.06893   # -1.4%  mass < 49.67
        - 0.01241904 * max(0.0, 21.78408 - Q.mass) * max(0.0, 0.0001947983 - Q.lam2) / 0.0007178522   # -1.2%  mass < 21.78 and lam2 < 0.0001948
        + 0.01125982 * max(0.0, 29.6447 - Q.mass) * max(0.0, 1.762929e-06 - Q.e3) / 9.656157e-06   # +1.1%  mass < 29.64 and e3 < 1.763e-06
        + 0.01053211 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, Q.phi_0 - -0.04013062) / 0.0001189753   # +1.1%  girth2 < 0.006679 and phi_0 > -0.04013
        - 0.01049824 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 43.5 - Q.pt_7) / 0.02255472   # -1.0%  girth2 < 0.00502 and pt_7 < 43.5
        + 0.009754149 * max(0.0, 0.06108601 - Q.girth) * max(0.0, Q.width - 0.0003193707) / 1.048473e-05   # +1.0%  girth < 0.06109 and width > 0.0003194
        + 0.008438378 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 48.71875 - Q.pt_7) / 0.776931   # +0.8%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.004841071 * max(0.0, 0.1967397 - Q.LHA) * max(0.0, Q.pt_7 - 15.55391) / 0.4153091   # +0.5%  LHA < 0.1967 and pt_7 > 15.55
        + 0.002555506 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 3.10694e-07 - Q.e3) / 3.648373e-06   # +0.3%  sum_pt_top5 > 687.4 and e3 < 3.107e-07
        - 0.001821108 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.716559 - Q.D2_b2) / 0.2097233   # -0.2%  mass < 29.64 and D2_b2 < 0.7166
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 32.81;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 32.80682 * (-0.08618516
        + 0.1240212 * max(0.0, 0.00609665 - Q.girth2) / 0.002544039   # +12.4%  girth2 < 0.006097
        - 0.07599348 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -7.6%  sj3_dr_max < 0.1426
        + 0.07018366 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +7.0%  sj3_dr_max < 0.2134
        + 0.0617978 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +6.2%  girth2 < 0.003563
        - 0.06151235 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -6.2%  lam1 < 0.005954
        + 0.05321342 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +5.3%  sum_pt < 988.4
        + 0.04999583 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +5.0%  mass < 53.33 and centroid_offset < 0.02686
        - 0.04958894 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -5.0%  mass_over_sum_pt < 0.07269
        + 0.04737877 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +4.7%  lam1 > 0.005433
        + 0.04396316 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +4.4%  centroid_offset < 0.01838
        + 0.03167075 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +3.2%  max_dr < 0.1118
        - 0.03154574 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -3.2%  mass < 53.33
        - 0.03119559 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -3.1%  lam1 > 0.003377
        - 0.02965258 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -3.0%  e2 > 0.03556
        + 0.02394602 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +2.4%  e2 > 0.04447
        - 0.01870817 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -1.9%  e3 < 2.371e-05
        - 0.01513534 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.5%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.01385568 * max(0.0, Q.girth - 0.0717028) / 0.01201981   # -1.4%  girth > 0.0717
        - 0.01303954 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -1.3%  sj3_pair_mass_max < 24.23
        - 0.01283815 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -1.3%  centroid_offset < 0.01838 and n_for_90pct > 5
        + 0.01255159 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +1.3%  tau1 < 0.0437
        + 0.01200645 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +1.2%  mass < 53.33 and log_sum_pt < 6.843
        - 0.01175454 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.2%  centroid_offset < 0.01838 and pt_1 < 159.2
        - 0.01012377 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -1.0%  C3 < 0.02847
        + 0.00998709 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +1.0%  girth2 < 0.006097 and planar_flow < 0.3221
        - 0.009380671 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # -0.9%  e3 > 8.148e-05
        + 0.008718245 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # +0.9%  e3 > 0.0001869
        - 0.008130667 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -0.8%  girth < 0.05465
        + 0.006948938 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.7%  centroid_offset < 0.01838 and z_2nd < 0.2056
        + 0.005018795 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.5%  lam2 > 0.001131
        - 0.004591043 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.5%  zdr_0 < 0.006292
        - 0.004549106 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.5%  tau1 < 0.0437 and eccentricity > 0.9031
        - 0.004136019 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.002834884 - Q.mean_phi) / 1.344572e-05   # -0.4%  girth2 < 0.006097 and mean_phi < 0.002835
        + 0.003747477 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.07073975 - Q.abseta_1) / 0.0003304963   # +0.4%  centroid_offset < 0.01838 and abseta_1 < 0.07074
        + 0.003212745 * max(0.0, 0.0001721983 - Q.width) / 1.441896e-05   # +0.3%  width < 0.0001722
        - 0.002974321 * max(0.0, 0.05464922 - Q.girth) * max(0.0, Q.z_5 - 0.03672711) / 0.0004195788   # -0.3%  girth < 0.05465 and z_5 > 0.03673
        + 0.002775102 * max(0.0, 2.371297e-05 - Q.e3) * max(0.0, Q.eccentricity - 0.8319502) / 8.425435e-07   # +0.3%  e3 < 2.371e-05 and eccentricity > 0.832
        + 0.002539706 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # +0.3%  sum_pt < 559.7
        - 0.002484701 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.tau4 - 0.001224244) / 1.099241e-05   # -0.2%  centroid_offset < 0.01838 and tau4 > 0.001224
        - 0.002343386 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.2%  sum_pt_top5 > 840
        + 0.002337382 * max(0.0, 0.1426152 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.6875) / 0.3914557   # +0.2%  sj3_dr_max < 0.1426 and pt_6 > 33.69
        + 0.002113562 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.2%  n_dr_0p2_0p4 > 1
        - 0.001640931 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.2%  z_dr_0p2_0p4
        + 0.001494578 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.1%  log_sum_pt > 6.896
        + 0.001291914 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, -0.01275329 - Q.mean_phi) / 1.158446e-05   # +0.1%  tau1 < 0.0437 and mean_phi < -0.01275
        - 0.001270547 * max(0.0, 0.002316125 - Q.centroid_offset) / 9.872932e-05   # -0.1%  centroid_offset < 0.002316
        - 0.000747558 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 0.7166
        + 0.0007149068 * max(0.0, 31.125 - Q.pt_4) / 0.3140792   # +0.1%  pt_4 < 31.12
        - 0.0004455124 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, Q.mean_phi - 0.02612796) / 3.162575e-07   # -0.0%  girth2 < 0.006097 and mean_phi > 0.02613
        - 0.0003949105 * max(0.0, 0.1070199 - Q.sj3_dr_max) * max(0.0, Q.abseta_1 - 0.03625488) / 8.135786e-06   # -0.0%  sj3_dr_max < 0.107 and abseta_1 > 0.03625
        + 0.0003376951 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.mean_phi - 0.02612796) / 2.994783e-06   # +0.0%  tau1 < 0.0437 and mean_phi > 0.02613
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.93;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.929779 * (0.191216
        + 0.1627792 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # +16.3%  girth2 > 0.00752
        + 0.1199647 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +12.0%  mass < 76.66
        - 0.07782654 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -7.8%  LHA > 0.3033
        - 0.07596583 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -7.6%  lam1 < 0.005954
        + 0.06840739 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +6.8%  tau1 > 0.05357
        - 0.06597427 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -6.6%  lam1 > 0.00733
        + 0.05759039 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +5.8%  lam2 > 0.0003061
        - 0.05478264 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -5.5%  lam1 < 0.004184
        - 0.05151769 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -5.2%  lam1 < 0.001504
        + 0.0342986 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +3.4%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        - 0.03160222 * max(0.0, Q.lam2 - 0.0003061234) * max(0.0, Q.planar_flow - 0.04505724) / 0.0002791469   # -3.2%  lam2 > 0.0003061 and planar_flow > 0.04506
        + 0.02990661 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # +3.0%  girth2_top3 < 0.002152
        - 0.01844152 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -1.8%  e3 > 3.892e-05
        - 0.01628632 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 6.572938 - Q.log_sum_pt) / 1.257183   # -1.6%  pt_7 < 45.75 and log_sum_pt < 6.573
        + 0.01618928 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +1.6%  pt_7 < 45.75 and D2 < 1.002
        - 0.01517667 * max(0.0, Q.sj3_dr_max - 0.1986272) / 0.03537498   # -1.5%  sj3_dr_max > 0.1986
        + 0.0146154 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +1.5%  sj3_pair_mass_min > 11.05
        + 0.01397187 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +1.4%  C2_b2 > 0.009033
        - 0.01388513 * max(0.0, 0.002635418 - Q.girth2) / 0.0007941346   # -1.4%  girth2 < 0.002635
        - 0.01332198 * max(0.0, 0.06473447 - Q.z_7) / 0.01631636   # -1.3%  z_7 < 0.06473
        - 0.009109586 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -0.9%  lam1 > 0.00733 and D2_b2 < 0.3809
        + 0.008277218 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # +0.8%  sj3_dr_min > 0.1278
        - 0.007389893 * max(0.0, Q.sj3_dr_min - 0.1278212) * max(0.0, 0.3658817 - Q.z_dr_0_0p05) / 0.002463327   # -0.7%  sj3_dr_min > 0.1278 and z_dr_0_0p05 < 0.3659
        + 0.006642897 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.01258764) / 3.400017e-05   # +0.7%  zdr_0 < 0.02118 and centroid_offset > 0.01259
        - 0.005843948 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # -0.6%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
        - 0.005124697 * max(0.0, Q.sum_pt - 988.4078) / 4.402478   # -0.5%  sum_pt > 988.4
        - 0.002677759 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, Q.D2_b2 - 1.129616) / 0.0002590844   # -0.3%  lam1 > 0.00733 and D2_b2 > 1.13
        - 0.002429694 * max(0.0, Q.girth2 - 0.02530566) / 0.0002851418   # -0.2%  girth2 > 0.02531
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 31.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.57518 * (-0.1153266
        + 0.2166346 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +21.7%  width < 0.008678
        + 0.1336995 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +13.4%  girth2 < 0.01324
        - 0.08224495 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -8.2%  e2_sq < 0.008169
        + 0.07722932 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +7.7%  centroid_offset < 0.03776
        - 0.06447417 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -6.4%  lam1 < 0.008376
        - 0.04895165 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -4.9%  sj3_dr_max < 0.169
        + 0.04392858 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +4.4%  z_7 > 0.01686
        - 0.04002021 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -4.0%  centroid_offset > 0.0499
        - 0.03734172 * max(0.0, 0.3127275 - Q.LHA) / 0.08529692   # -3.7%  LHA < 0.3127
        - 0.03087254 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -3.1%  mass < 69.61
        - 0.02576578 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # -2.6%  lam1 < 0.00733
        + 0.02547842 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +2.5%  sj3_dr_max < 0.2623
        - 0.02110814 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -2.1%  centroid_offset < 0.03776 and sum_pt < 901.6
        + 0.01966353 * max(0.0, 0.2534037 - Q.planar_flow) / 0.109515   # +2.0%  planar_flow < 0.2534
        - 0.0182353 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -1.8%  centroid_offset > 0.0499 and D2 < 2.844
        - 0.01589199 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -1.6%  girth < 0.0717
        + 0.01148164 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +1.1%  max_dr < 0.1452
        - 0.009980051 * max(0.0, Q.pt_7 - 29.04219) / 7.785987   # -1.0%  pt_7 > 29.04
        + 0.009543292 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +1.0%  mass < 15.45
        - 0.009196218 * max(0.0, 0.01437952 - Q.centroid_offset) / 0.004306483   # -0.9%  centroid_offset < 0.01438
        + 0.008148673 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +0.8%  sj3_dr_max < 0.1426
        - 0.006655291 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 0.0345459 - Q.absphi_0) / 0.0003803849   # -0.7%  centroid_offset < 0.03776 and absphi_0 < 0.03455
        - 0.006336056 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 840.0195 - Q.sum_pt) / 14.59087   # -0.6%  planar_flow < 0.2534 and sum_pt < 840
        - 0.006112064 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -0.6%  girth2 < 0.004372
        - 0.005650826 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -0.6%  girth < 0.02054
        + 0.005203553 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.5%  sj2_dr > 0.2688
        - 0.004402022 * max(0.0, 0.0005049491 - Q.lam1) / 8.739619e-05   # -0.4%  lam1 < 0.0005049
        + 0.00423341 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # +0.4%  centroid_offset > 0.0499 and tau32 < 0.5503
        + 0.003081795 * max(0.0, 49.6681 - Q.mass) / 17.06893   # +0.3%  mass < 49.67
        + 0.002669249 * max(0.0, 0.01437952 - Q.centroid_offset) * max(0.0, 0.5327104 - Q.D2_b2) / 0.0007565864   # +0.3%  centroid_offset < 0.01438 and D2_b2 < 0.5327
        - 0.002146712 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, -0.001813533 - Q.mean_phi) / 4.860641e-05   # -0.2%  centroid_offset < 0.03776 and mean_phi < -0.001814
        - 0.001884037 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 37.15625 - Q.pt_7) / 0.5537514   # -0.2%  planar_flow < 0.2534 and pt_7 < 37.16
        - 0.001734699 * max(0.0, 0.2534037 - Q.planar_flow) * max(0.0, 1.050302e-05 - Q.e3) / 1.771164e-07   # -0.2%  planar_flow < 0.2534 and e3 < 1.05e-05
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.3303;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.3302897 * (-2.281463
        + 0.4723772 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +47.2%  girth2 > 0.01883
        - 0.1260663 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -12.6%  e2 > 0.06344
        + 0.1124178 * max(0.0, Q.mass - 91.19) / 0.5839191   # +11.2%  mass > 91.19
        + 0.08971503 * max(0.0, Q.mass - 69.61135) * max(0.0, Q.centroid_offset - 0.009480685) / 0.04081507   # +9.0%  mass > 69.61 and centroid_offset > 0.009481
        - 0.07090526 * max(0.0, Q.zdr_0 - 0.03981924) / 0.0005695785   # -7.1%  zdr_0 > 0.03982
        + 0.06445901 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # +6.4%  centroid_offset > 0.0499
        - 0.06405943 * max(0.0, Q.girth2 - 0.01882765) * max(0.0, 53.4375 - Q.pt_7) / 0.01439677   # -6.4%  girth2 > 0.01883 and pt_7 < 53.44
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.99876 * (0.02109701
        + 0.3198104 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +32.0%  girth < 0.1484
        - 0.1227923 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -12.3%  e2 < 0.08001
        + 0.1095241 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +11.0%  lam1 < 0.01643
        - 0.08443202 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -8.4%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.04586104 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -4.6%  e3 < 5.335e-05 and centroid_offset < 0.03776
        + 0.04580156 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +4.6%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        + 0.044123 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # +4.4%  e3 < 5.335e-05
        - 0.04196096 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -4.2%  girth < 0.1484 and pt_7 < 38.53
        + 0.03530201 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +3.5%  width < 0.01324
        - 0.02444747 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.000537286 - Q.lam2) / 4.021081e-05   # -2.4%  girth < 0.1484 and lam2 < 0.0005373
        - 0.01825795 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55592   # -1.8%  sum_pt_top5 > 658.1
        - 0.01328871 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -1.3%  pt_6 < 31.91 and z_7 < 0.05861
        - 0.01311102 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -1.3%  z_7 < 0.02807
        - 0.01205591 * max(0.0, 0.007520088 - Q.width) / 0.003544991   # -1.2%  width < 0.00752
        - 0.01021145 * max(0.0, Q.mass - 49.6681) / 7.721594   # -1.0%  mass > 49.67
        + 0.009120437 * max(0.0, Q.sum_pt_top5 - 791.125) * max(0.0, 31.90625 - Q.pt_6) / 134.9952   # +0.9%  sum_pt_top5 > 791.1 and pt_6 < 31.91
        + 0.007964806 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 0.07474969 - Q.M3) / 0.0006773733   # +0.8%  girth < 0.1484 and M3 < 0.07475
        + 0.007232913 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.tau2 - 0.008780509) / 0.000269945   # +0.7%  girth < 0.1484 and tau2 > 0.008781
        + 0.007122271 * max(0.0, Q.sum_pt_top5 - 791.125) / 14.55255   # +0.7%  sum_pt_top5 > 791.1
        + 0.006772724 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.z_7 - 0.06164517) / 0.0003465887   # +0.7%  girth < 0.1484 and z_7 > 0.06165
        - 0.00528608 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.5%  sum_pt > 988.4 and pt_6 < 62.25
        - 0.0046033 * max(0.0, 0.02818362 - Q.z_5) / 0.0003804269   # -0.5%  z_5 < 0.02818
        - 0.00408938 * max(0.0, Q.sj3_dr23 - 0.1974628) / 0.02478939   # -0.4%  sj3_dr23 > 0.1975
        - 0.003201627 * max(0.0, 0.02160287 - Q.z_6) / 0.0003105785   # -0.3%  z_6 < 0.0216
        - 0.002115452 * max(0.0, Q.pt_7 - 48.71875) / 0.6759439   # -0.2%  pt_7 > 48.72
        - 0.001511147 * max(0.0, 0.1484084 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.78116) / 0.01215768   # -0.2%  girth < 0.1484 and sj2_mass1 > 31.78
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 38.82;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.82467 * (-0.0630071
        - 0.1249509 * max(0.0, Q.width - 0.006679471) / 0.002702003   # -12.5%  width > 0.006679
        + 0.111353 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +11.1%  girth > 0.0269
        - 0.08439688 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -8.4%  e2 > 0.01655
        - 0.0710922 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -7.1%  girth2 > 0.00752
        - 0.05257948 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -5.3%  girth > 0.08724
        + 0.04990737 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +5.0%  mass_over_sum_pt > 0.08475
        + 0.04940985 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +4.9%  sj2_dr > 0.1592
        + 0.03779131 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +3.8%  mass_over_sum_pt > 0.07992
        + 0.03746098 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +3.7%  width > 0.003563
        + 0.03498989 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +3.5%  girth > 0.04082
        - 0.0306059 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -3.1%  sd_mass < 74.58
        - 0.02450941 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -2.5%  sj2_dr > 0.06154
        + 0.02396428 * max(0.0, Q.e2_sq - 0.002074109) / 0.004484017   # +2.4%  e2_sq > 0.002074
        + 0.01895067 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +1.9%  sd_mass < 49.92
        - 0.0181587 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -1.8%  e3 < 5.335e-05
        + 0.01744361 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +1.7%  e2 > 0.05028
        - 0.01515785 * max(0.0, 0.004032342 - Q.C2_b2) / 0.002883542   # -1.5%  C2_b2 < 0.004032
        - 0.01515256 * max(0.0, Q.e2_sq - 0.0030133) / 0.003940214   # -1.5%  e2_sq > 0.003013
        + 0.01492718 * max(0.0, Q.girth - 0.08065885) / 0.009330634   # +1.5%  girth > 0.08066
        - 0.01463938 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -1.5%  sj2_dr > 0.2002
        - 0.01311431 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -1.3%  z_dr_0p05_0p1 > 0.7509
        + 0.01147516 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +1.1%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        + 0.01076462 * max(0.0, Q.width - 0.01323868) / 0.001442684   # +1.1%  width > 0.01324
        + 0.01012825 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # +1.0%  lam2 < 0.0005373
        + 0.008302595 * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.3724174   # +0.8%  z_dr_0p05_0p1 < 0.5883
        + 0.007987677 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # +0.8%  mass_over_sum_pt > 0.09041
        + 0.007777047 * max(0.0, 0.1115136 - Q.planar_flow) / 0.03343187   # +0.8%  planar_flow < 0.1115
        + 0.006911032 * max(0.0, Q.psi_0p1 - 0.8155839) / 0.1042073   # +0.7%  psi_0p1 > 0.8156
        + 0.006645508 * max(0.0, 5.0 - Q.n_dr_0_0p05) / 1.94322   # +0.7%  n_dr_0_0p05 < 5
        - 0.006603017 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -0.7%  centroid_offset > 0.0499
        + 0.006596877 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +0.7%  e2 > 0.04111
        - 0.006362287 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.6%  centroid_offset > 0.02686
        + 0.005804138 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # +0.6%  e2_sq > 0.01166
        - 0.005729219 * max(0.0, Q.psi_0p1 - 0.7143804) / 0.1805043   # -0.6%  psi_0p1 > 0.7144
        - 0.005411202 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -0.5%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.005304372 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.5%  psi_0p1 > 0.9761
        + 0.005032672 * max(0.0, Q.width - 0.005590289) / 0.003084256   # +0.5%  width > 0.00559
        + 0.00359014 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # +0.4%  LHA > 0.3033
        - 0.003352865 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -0.3%  sj2_dr > 0.1295
        + 0.003290649 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width) / 3.351526e-05   # +0.3%  planar_flow < 0.1115 and width < 0.006097
        - 0.002853609 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.3%  planar_flow < 0.1115 and centroid_offset < 0.01838
        + 0.002505875 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1629004) / 4.043869e-07   # +0.3%  e3 < 5.335e-05 and sj3_dr23 > 0.1629
        - 0.002094151 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 658.125 - Q.sum_pt_top5) / 2.950014   # -0.2%  planar_flow < 0.1115 and sum_pt_top5 < 658.1
        - 0.002055933 * max(0.0, Q.mass - 69.61135) / 2.406702   # -0.2%  mass > 69.61
        - 0.001898819 * max(0.0, Q.sj2_dr - 0.2001708) * max(0.0, 0.1058993 - Q.z_5) / 0.0007616364   # -0.2%  sj2_dr > 0.2002 and z_5 < 0.1059
        + 0.0009665741 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +0.1%  planar_flow < 0.1115 and lam1 < 0.01643
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 41.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.29396 * (-0.01596565
        + 0.1758726 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +17.6%  width < 0.01324
        - 0.1476741 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -14.8%  girth < 0.1019
        + 0.1066244 * max(0.0, 0.01882765 - Q.width) / 0.01314296   # +10.7%  width < 0.01883
        + 0.09134505 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +9.1%  e2 < 0.04111
        - 0.0901376 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -9.0%  e2_sq < 0.008169
        - 0.06082557 * max(0.0, 0.007520088 - Q.girth2) / 0.00354499   # -6.1%  girth2 < 0.00752
        - 0.06043854 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -6.0%  e2_sq < 0.01166
        - 0.05743756 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -5.7%  mass_over_sum_pt < 0.1309
        - 0.05647683 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -5.6%  sj2_dr < 0.2002
        + 0.05636272 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +5.6%  sj2_dr < 0.1592
        + 0.04784863 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +4.8%  mass_over_sum_pt_sq < 0.007183
        - 0.01662763 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -1.7%  lam2 < 0.001131
        + 0.008452169 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +0.8%  lam1 < 0.00484
        + 0.003631934 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +0.4%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        - 0.003615635 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -0.4%  sj2_dr < 0.1493
        - 0.003297691 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 0.5882598 - Q.z_dr_0p05_0p1) / 0.0114178   # -0.3%  N2 < 0.2233 and z_dr_0p05_0p1 < 0.5883
        + 0.003174917 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.sum_pt_top5 - 430.75) / 8.791034   # +0.3%  N2 < 0.2233 and sum_pt_top5 > 430.8
        + 0.002819208 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.1094252   # +0.3%  N2 < 0.2233 and n_dr_0p1_0p2 < 4
        - 0.002496108 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.2%  girth2 < 0.00752 and D2 < 0.746
        - 0.002427334 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2) / 3.18925e-05   # -0.2%  width < 0.006097 and D2 < 0.746
        - 0.00241384 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # -0.2%  girth2_top2 < 0.0005125
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.8495764705882353, 0.696216806722689, 2.1093123949579833, 0.9594892857142857, 1.2807436974789916, 1.7954098739495798, 1.025213025210084, 1.8630697478991596, 0.28188382352941177, 3.157835294117647, 2.080255987394958, 2.133087079831933, 0.08756176470588235, 3.4130636554621847, 0.40827153361344537, 0.3285310924369748]
T = [2.342599391084559, 1.4323582359506304, 3.2625796858587184, 2.502389113379727, 2.945475347951681]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +39%, n9 +23%, n5 -14%, n1 +12%, n0 -6%, n6 +5% ...
            + 0.3868972 * h[2] / H_AVG[2]
            + 0.2316883 * h[9] / H_AVG[9]
            - 0.1437033 * h[5] / H_AVG[5]
            + 0.1160931 * h[1] / H_AVG[1]
            - 0.05666625 * h[0] / H_AVG[0]
            + 0.04786677 * h[6] / H_AVG[6]
            - 0.01708497 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +1% ...
            + 0.5597721 * h[9] / H_AVG[9]
            - 0.1815412 * h[10] / H_AVG[10]
            + 0.08946898 * h[6] / H_AVG[6]
            - 0.0838266 * h[4] / H_AVG[4]
            + 0.05875614 * h[5] / H_AVG[5]
            + 0.01433524 * h[15] / H_AVG[15]
            + 0.01229981 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +25%, n3 -15%, n7 +12%, n6 -10%, n14 -9%, n0 +9% ...
            + 0.2451764 * h[11] / H_AVG[11]
            - 0.1470446 * h[3] / H_AVG[3]
            + 0.1249154 * h[7] / H_AVG[7]
            - 0.09819808 * h[6] / H_AVG[6]
            - 0.09385323 * h[14] / H_AVG[14]
            + 0.08951258 * h[0] / H_AVG[0]
            + 0.07355561 * h[13] / H_AVG[13]
            - 0.069229 * h[15] / H_AVG[15]
            - 0.03024673 * h[9] / H_AVG[9]
            - 0.02159977 * h[8] / H_AVG[8]
            - 0.00666858 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -22%, n6 -15%, n13 +7%, n14 +6%, n4 +4% ...
            + 0.3489921 * h[7] / H_AVG[7]
            - 0.215679 * h[3] / H_AVG[3]
            - 0.1536351 * h[6] / H_AVG[6]
            + 0.07458949 * h[13] / H_AVG[13]
            + 0.06118226 * h[14] / H_AVG[14]
            + 0.03998503 * h[4] / H_AVG[4]
            - 0.03943526 * h[9] / H_AVG[9]
            + 0.03477761 * h[1] / H_AVG[1]
            - 0.02051359 * h[15] / H_AVG[15]
            + 0.0112106 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4707414 * h[13] / H_AVG[13]
            + 0.2648455 * h[10] / H_AVG[10]
            - 0.1523871 * h[5] / H_AVG[5]
            + 0.05435217 * h[4] / H_AVG[4]
            + 0.02035939 * h[3] / H_AVG[3]
            + 0.01794387 * h[8] / H_AVG[8]
            - 0.01486377 * h[12] / H_AVG[12]
            + 0.004506788 * h[0] / H_AVG[0]
        ),
    ]]


def classify(pt, eta, phi):
    s = logits(jet_layer_4(quantities(pt, eta, phi)))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
