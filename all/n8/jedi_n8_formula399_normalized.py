"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.5%   (on for 90% of jets)
  neuron  9:  12.0%   (on for 64% of jets)
  neuron  7:  10.8%   (on for 62% of jets)
  neuron  3:   9.3%   (on for 25% of jets)
  neuron 10:   8.1%   (on for 86% of jets)
  neuron  2:   7.2%   (on for 89% of jets)
  neuron  5:   7.1%   (on for 63% of jets)
  neuron  6:   7.0%   (on for 33% of jets)
  neuron 11:   6.1%   (on for 74% of jets)
  neuron  0:   3.9%   (on for 40% of jets)
  neuron 14:   3.7%   (on for 26% of jets)
  neuron  1:   3.5%   (on for 65% of jets)
  neuron  4:   3.0%   (on for 55% of jets)
  neuron 15:   2.4%   (on for 34% of jets)
  neuron  8:   1.1%   (on for 38% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 89.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
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
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        phi_0=phi[0],
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
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
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 17.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.17644 * (-0.02701375
        + 0.2324265 * max(0.0, 0.013 - Q.girth2) / 0.008032715   # +23.2%  girth2 < 0.013
        - 0.1092548 * max(0.0, 64.7 - Q.mass) / 27.63782   # -10.9%  mass < 64.7
        - 0.1043058 * max(0.0, 0.00444 - Q.width) / 0.001599645   # -10.4%  width < 0.00444
        - 0.07609669 * max(0.0, 0.0751 - Q.girth) / 0.02728748   # -7.6%  girth < 0.0751
        - 0.06628041 * max(0.0, Q.sj3_dr_max - 0.182) / 0.0418552   # -6.6%  sj3_dr_max > 0.182
        + 0.06484258 * max(0.0, 0.0354 - Q.e2) / 0.0136157   # +6.5%  e2 < 0.0354
        + 0.05767353 * max(0.0, Q.sj3_dr_max - 0.11) / 0.08324588   # +5.8%  sj3_dr_max > 0.11
        - 0.04548142 * max(0.0, 0.00151 - Q.C2_b2) / 0.0008400094   # -4.5%  C2_b2 < 0.00151
        - 0.04395217 * max(0.0, 29.6 - Q.mass) / 7.329531   # -4.4%  mass < 29.6
        - 0.03158554 * max(0.0, Q.centroid_offset - 0.00783) / 0.01070073   # -3.2%  centroid_offset > 0.00783
        + 0.03061174 * max(0.0, 0.0243 - Q.e2) / 0.007333344   # +3.1%  e2 < 0.0243
        + 0.02180459 * max(0.0, 0.159 - Q.planar_flow) / 0.05648946   # +2.2%  planar_flow < 0.159
        + 0.01853583 * max(0.0, 0.00414 - Q.width) * max(0.0, 0.0364 - Q.C3) / 3.605657e-05   # +1.9%  width < 0.00414 and C3 < 0.0364
        - 0.01629472 * max(0.0, Q.centroid_offset - 0.0499) / 0.0008065856   # -1.6%  centroid_offset > 0.0499
        - 0.0151735 * max(0.0, 0.245 - Q.sj3_dr_max) * max(0.0, 0.058 - Q.z_7) / 0.001371719   # -1.5%  sj3_dr_max < 0.245 and z_7 < 0.058
        - 0.01440506 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, Q.z_7 - 0.0178) / 0.000540235   # -1.4%  log_sum_pt > 6.68 and z_7 > 0.0178
        + 0.01060347 * max(0.0, 0.0133 - Q.girth2) * max(0.0, 1.02 - Q.D2) / 0.001058894   # +1.1%  girth2 < 0.0133 and D2 < 1.02
        - 0.009184691 * max(0.0, Q.sum_pt - 906.0) / 11.77315   # -0.9%  sum_pt > 906
        + 0.008337364 * max(0.0, 8.01e-05 - Q.lam2) * max(0.0, 0.257 - Q.D2_b2) / 2.60375e-06   # +0.8%  lam2 < 8.01e-05 and D2_b2 < 0.257
        - 0.00640655 * max(0.0, 0.00649 - Q.lam1) * max(0.0, 0.843 - Q.D2) / 6.964666e-05   # -0.6%  lam1 < 0.00649 and D2 < 0.843
        + 0.005578929 * max(0.0, 0.0134 - Q.girth2) * max(0.0, Q.phi_0 - -0.00668) / 0.0001171469   # +0.6%  girth2 < 0.0134 and phi_0 > -0.00668
        + 0.005445028 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0746 - Q.dr_4) / 0.001897083   # +0.5%  log_sum_pt > 6.68 and dr_4 < 0.0746
        - 0.002532997 * max(0.0, Q.centroid_offset - 0.0203) * max(0.0, 0.0609 - Q.z_7) / 2.825186e-05   # -0.3%  centroid_offset > 0.0203 and z_7 < 0.0609
        + 0.001489039 * max(0.0, Q.sum_pt - 915.0) * max(0.0, Q.pt_7 - 35.4) / 54.76743   # +0.1%  sum_pt > 915 and pt_7 > 35.4
        - 0.0009169663 * max(0.0, 0.142 - Q.planar_flow) * max(0.0, 0.023 - Q.dr_2) / 2.382786e-05   # -0.1%  planar_flow < 0.142 and dr_2 < 0.023
        + 0.0007801022 * max(0.0, 0.165 - Q.planar_flow) * max(0.0, 0.0237 - Q.z_7) / 2.627329e-05   # +0.1%  planar_flow < 0.165 and z_7 < 0.0237
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 9.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.090461 * (0.1408069
        - 0.1533511 * max(0.0, Q.sum_pt_top5 - 528.0) / 108.0645   # -15.3%  sum_pt_top5 > 528
        - 0.1520326 * max(0.0, 0.0991 - Q.girth) / 0.04622227   # -15.2%  girth < 0.0991
        + 0.1432702 * max(0.0, Q.log_sum_pt - 6.5) / 0.1264458   # +14.3%  log_sum_pt > 6.5
        + 0.1364864 * max(0.0, Q.sum_pt_top5 - 364.0) / 234.0989   # +13.6%  sum_pt_top5 > 364
        - 0.09644207 * max(0.0, 0.00871 - Q.girth2) / 0.004472974   # -9.6%  girth2 < 0.00871
        + 0.09339983 * max(0.0, 0.00591 - Q.mass_over_sum_pt_sq) / 0.00262863   # +9.3%  mass_over_sum_pt_sq < 0.00591
        - 0.05618258 * max(0.0, 0.0619 - Q.z_7) / 0.01438664   # -5.6%  z_7 < 0.0619
        + 0.03717761 * max(0.0, Q.log_sum_pt - 6.62) / 0.06461981   # +3.7%  log_sum_pt > 6.62
        - 0.01780582 * max(0.0, 0.0607 - Q.z_7) * max(0.0, 1.52 - Q.D2) / 0.00445904   # -1.8%  z_7 < 0.0607 and D2 < 1.52
        - 0.01503371 * max(0.0, 0.00871 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.25) / 0.00294533   # -1.5%  lam1 < 0.00871 and n_pt_above_50 > 5.25
        - 0.01470561 * max(0.0, 0.00085 - Q.lam1) / 0.0001816315   # -1.5%  lam1 < 0.00085
        + 0.01455731 * max(0.0, Q.log_sum_pt - 6.48) * max(0.0, 1.4 - Q.D2) / 0.04255068   # +1.5%  log_sum_pt > 6.48 and D2 < 1.4
        + 0.01396234 * max(0.0, Q.z_7 - 0.0586) / 0.005876117   # +1.4%  z_7 > 0.0586
        - 0.01262365 * max(0.0, 0.00962 - Q.lam1) * max(0.0, 0.15 - Q.planar_flow) / 0.0001925416   # -1.3%  lam1 < 0.00962 and planar_flow < 0.15
        + 0.009375313 * max(0.0, Q.pt_7 - 33.5) * max(0.0, Q.sj2_dr - 0.143) / 0.1786707   # +0.9%  pt_7 > 33.5 and sj2_dr > 0.143
        - 0.009158663 * max(0.0, Q.pt_7 - 33.4) * max(0.0, 53.4 - Q.pt_6) / 18.70932   # -0.9%  pt_7 > 33.4 and pt_6 < 53.4
        - 0.008906017 * max(0.0, Q.sj3_dr_max - 0.104) * max(0.0, Q.sj3_pair_mass_min - 2.6) / 0.9569717   # -0.9%  sj3_dr_max > 0.104 and sj3_pair_mass_min > 2.6
        + 0.006208624 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, Q.centroid_offset - 0.0126) / 0.0006882835   # +0.6%  log_sum_pt > 6.37 and centroid_offset > 0.0126
        + 0.005237199 * max(0.0, 0.00939 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0216) / 7.843254e-06   # +0.5%  girth2 < 0.00939 and centroid_offset > 0.0216
        - 0.004083446 * max(0.0, Q.pt_7 - 53.5) / 0.3284992   # -0.4%  pt_7 > 53.5
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 8.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.689738 * (0.4856303
        - 0.1960186 * max(0.0, 52.4 - Q.pt_7) / 18.14005   # -19.6%  pt_7 < 52.4
        - 0.1672626 * max(0.0, Q.LHA - 0.131) / 0.1191367   # -16.7%  LHA > 0.131
        + 0.1379623 * max(0.0, 0.105 - Q.mass_over_sum_pt) / 0.04953952   # +13.8%  mass_over_sum_pt < 0.105
        + 0.1259054 * max(0.0, 807.0 - Q.sum_pt) / 124.6111   # +12.6%  sum_pt < 807
        - 0.09360092 * max(0.0, Q.z_7 - 0.0366) / 0.01857003   # -9.4%  z_7 > 0.0366
        - 0.03965481 * max(0.0, Q.sum_pt - 538.0) * max(0.0, 0.000194 - Q.lam2) / 0.02734841   # -4.0%  sum_pt > 538 and lam2 < 0.000194
        + 0.03542433 * max(0.0, 6.48 - Q.log_sum_pt) / 0.07581975   # +3.5%  log_sum_pt < 6.48
        - 0.03014127 * max(0.0, 0.236 - Q.max_dr) / 0.1185157   # -3.0%  max_dr < 0.236
        - 0.03005126 * max(0.0, 0.502 - Q.planar_flow) / 0.2798902   # -3.0%  planar_flow < 0.502
        - 0.02658999 * max(0.0, 35.5 - Q.mass) / 9.790681   # -2.7%  mass < 35.5
        + 0.02431778 * max(0.0, 0.000201 - Q.lam2) / 0.0001167487   # +2.4%  lam2 < 0.000201
        + 0.01652147 * max(0.0, 0.00496 - Q.lam1) * max(0.0, Q.pt_6 - 17.3) / 0.0437705   # +1.7%  lam1 < 0.00496 and pt_6 > 17.3
        - 0.01580117 * max(0.0, 59.4 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.0112) / 0.1880931   # -1.6%  sj3_pair_mass_max < 59.4 and centroid_offset > 0.0112
        + 0.01569327 * max(0.0, Q.pt_7 - 43.7) / 1.397238   # +1.6%  pt_7 > 43.7
        - 0.01193524 * max(0.0, 0.00609 - Q.lam1) * max(0.0, Q.max_dr - 0.0769) / 5.034665e-05   # -1.2%  lam1 < 0.00609 and max_dr > 0.0769
        - 0.009033393 * max(0.0, Q.z_7 - 0.0548) * max(0.0, 0.0684 - Q.sj3_dr_min) / 0.0002460747   # -0.9%  z_7 > 0.0548 and sj3_dr_min < 0.0684
        + 0.006538946 * max(0.0, Q.sum_pt_top5 - 745.0) * max(0.0, 1.1 - Q.D2_b2) / 6.591847   # +0.7%  sum_pt_top5 > 745 and D2_b2 < 1.1
        + 0.004064361 * max(0.0, Q.sum_pt - 923.0) / 9.597345   # +0.4%  sum_pt > 923
        - 0.003797071 * max(0.0, 0.00765 - Q.girth) / 0.0002291358   # -0.4%  girth < 0.00765
        + 0.003407683 * max(0.0, Q.m012 - 42.1) / 1.100813   # +0.3%  m012 > 42.1
        - 0.002893145 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, 1.31 - Q.D2_b2) / 0.002557546   # -0.3%  log_sum_pt > 6.85 and D2_b2 < 1.31
        + 0.002835341 * max(0.0, 6.33 - Q.log_sum_pt) * max(0.0, 1.14 - Q.D2_b2) / 0.01699198   # +0.3%  log_sum_pt < 6.33 and D2_b2 < 1.14
        - 0.0005495938 * max(0.0, Q.log_sum_pt - 6.95) * max(0.0, Q.pt_6 - 40.0) / 0.02760593   # -0.1%  log_sum_pt > 6.95 and pt_6 > 40
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 21.7;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.70379 * (-0.2064155
        + 0.2128261 * max(0.0, Q.girth - 0.0409) / 0.02701248   # +21.3%  girth > 0.0409
        - 0.1394044 * max(0.0, Q.girth2 - 0.00879) / 0.002192467   # -13.9%  girth2 > 0.00879
        - 0.09719763 * max(0.0, 0.0118 - Q.mass_over_sum_pt_sq) / 0.007324852   # -9.7%  mass_over_sum_pt_sq < 0.0118
        - 0.09167658 * max(0.0, Q.tau1 - 0.0469) / 0.03023905   # -9.2%  tau1 > 0.0469
        + 0.06139679 * max(0.0, Q.sj3_dr_max - 0.196) / 0.03630908   # +6.1%  sj3_dr_max > 0.196
        + 0.05873636 * max(0.0, Q.sj2_dr - 0.188) / 0.02706586   # +5.9%  sj2_dr > 0.188
        + 0.04850704 * max(0.0, Q.lam1 - 0.00828) / 0.001860047   # +4.9%  lam1 > 0.00828
        - 0.0343189 * max(0.0, Q.girth - 0.103) / 0.005245425   # -3.4%  girth > 0.103
        + 0.03318061 * max(0.0, Q.tau1 - 0.114) / 0.0069917   # +3.3%  tau1 > 0.114
        + 0.03053455 * max(0.0, Q.girth - 0.0307) * max(0.0, Q.log_sum_pt - 6.11) / 0.01050262   # +3.1%  girth > 0.0307 and log_sum_pt > 6.11
        - 0.02770171 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 1.02 - Q.z_dr_0p05_0p1) / 0.01821916   # -2.8%  sj2_dr > 0.199 and z_dr_0p05_0p1 < 1.02
        - 0.02280729 * max(0.0, Q.sj2_dr - 0.269) / 0.008432789   # -2.3%  sj2_dr > 0.269
        + 0.01858592 * max(0.0, Q.sj2_dr - 0.268) * max(0.0, 1.03 - Q.z_dr_0p05_0p1) / 0.006623727   # +1.9%  sj2_dr > 0.268 and z_dr_0p05_0p1 < 1.03
        + 0.01843329 * max(0.0, Q.e2 - 0.046) / 0.004061648   # +1.8%  e2 > 0.046
        + 0.0170271 * max(0.0, Q.girth2 - 0.0187) / 0.0007731226   # +1.7%  girth2 > 0.0187
        - 0.01456289 * max(0.0, Q.mass - 65.1) / 3.179778   # -1.5%  mass > 65.1
        - 0.01423302 * max(0.0, 0.836 - Q.z_dr_0_0p05) / 0.3439984   # -1.4%  z_dr_0_0p05 < 0.836
        + 0.01280252 * max(0.0, Q.centroid_offset - 0.00928) / 0.009783918   # +1.3%  centroid_offset > 0.00928
        - 0.01147981 * max(0.0, Q.sj3_dr_max - 0.297) / 0.0127772   # -1.1%  sj3_dr_max > 0.297
        + 0.009628736 * max(0.0, Q.sd_mass - 63.8) / 3.105202   # +1.0%  sd_mass > 63.8
        + 0.008886107 * max(0.0, 1.04 - Q.n_dr_0_0p05) / 0.2999412   # +0.9%  n_dr_0_0p05 < 1.04
        + 0.006863931 * max(0.0, Q.lam2 - 0.000776) * max(0.0, 64.0 - Q.pt_6) / 0.008920559   # +0.7%  lam2 > 0.000776 and pt_6 < 64
        - 0.005242277 * max(0.0, Q.sj2_dr - 0.206) * max(0.0, Q.sj2_mass1 - -0.0277) / 0.376746   # -0.5%  sj2_dr > 0.206 and sj2_mass1 > -0.0277
        - 0.002891283 * max(0.0, Q.sd_mass - 86.0) / 0.7756714   # -0.3%  sd_mass > 86
        - 0.001075082 * max(0.0, Q.sj2_dr - 0.214) * max(0.0, 0.0549 - Q.dr_3) / 0.0001149426   # -0.1%  sj2_dr > 0.214 and dr_3 < 0.0549
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 28.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.96564 * (0.001505232
        - 0.1625977 * max(0.0, 0.00919 - Q.girth2) / 0.00486042   # -16.3%  girth2 < 0.00919
        + 0.1600477 * max(0.0, 0.0116 - Q.e2_sq) / 0.007154143   # +16.0%  e2_sq < 0.0116
        + 0.1051297 * max(0.0, 0.0651 - Q.C2) / 0.03965037   # +10.5%  C2 < 0.0651
        + 0.09586962 * max(0.0, 0.224 - Q.N2) / 0.05151993   # +9.6%  N2 < 0.224
        - 0.06973945 * max(0.0, 0.237 - Q.sj3_dr_max) / 0.09308974   # -7.0%  sj3_dr_max < 0.237
        - 0.06513836 * max(0.0, 0.000876 - Q.lam2) / 0.0006860996   # -6.5%  lam2 < 0.000876
        - 0.06161068 * max(0.0, 0.224 - Q.N2) * max(0.0, Q.eccentricity - 0.727) / 0.013122   # -6.2%  N2 < 0.224 and eccentricity > 0.727
        + 0.05309745 * max(0.0, 0.351 - Q.LHA) / 0.1165153   # +5.3%  LHA < 0.351
        + 0.04498815 * max(0.0, Q.width - 0.00339) / 0.004163292   # +4.5%  width > 0.00339
        + 0.03542419 * max(0.0, 0.00779 - Q.girth2_top2) / 0.004664019   # +3.5%  girth2_top2 < 0.00779
        - 0.02617474 * max(0.0, 0.000453 - Q.lam2) * max(0.0, 0.173 - Q.dr01) / 4.382473e-05   # -2.6%  lam2 < 0.000453 and dr01 < 0.173
        - 0.02471453 * max(0.0, Q.mass - 45.1) / 9.687039   # -2.5%  mass > 45.1
        - 0.02409041 * max(0.0, 753.0 - Q.sum_pt) / 90.27089   # -2.4%  sum_pt < 753
        - 0.01844314 * max(0.0, Q.mass_over_sum_pt - 0.0903) / 0.008321144   # -1.8%  mass_over_sum_pt > 0.0903
        + 0.01449471 * max(0.0, Q.sd_mass - 42.1) * max(0.0, 26.6 - Q.sj3_pair_mass_min) / 153.2294   # +1.4%  sd_mass > 42.1 and sj3_pair_mass_min < 26.6
        - 0.01198233 * max(0.0, 0.242 - Q.N2) * max(0.0, 52.9 - Q.pt_7) / 1.045409   # -1.2%  N2 < 0.242 and pt_7 < 52.9
        - 0.01145191 * max(0.0, 0.219 - Q.N2) * max(0.0, 63.2 - Q.mass) / 0.3911696   # -1.1%  N2 < 0.219 and mass < 63.2
        - 0.005489457 * max(0.0, Q.sd_mass - 37.0) * max(0.0, 0.276 - Q.sd_zg) / 0.4953446   # -0.5%  sd_mass > 37 and sd_zg < 0.276
        - 0.00413912 * max(0.0, 0.0076 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.000612) / 6.024735e-06   # -0.4%  girth2_top2 < 0.0076 and C2_b2 > 0.000612
        - 0.004120496 * max(0.0, 0.219 - Q.N2) * max(0.0, Q.e2_sq - 0.0116) / 6.859356e-05   # -0.4%  N2 < 0.219 and e2_sq > 0.0116
        - 0.001256086 * max(0.0, Q.max_dr - 0.164) * max(0.0, 0.000596 - Q.C2_b2) / 8.013949e-07   # -0.1%  max_dr > 0.164 and C2_b2 < 0.000596
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 6.887;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.886916 * (-0.06447008
        + 0.1805447 * max(0.0, 0.0702 - Q.z_7) / 0.02038354   # +18.1%  z_7 < 0.0702
        + 0.1429989 * max(0.0, 0.00167 - Q.girth2) * max(0.0, 0.024 - Q.centroid_offset) / 7.294975e-06   # +14.3%  girth2 < 0.00167 and centroid_offset < 0.024
        + 0.1105648 * max(0.0, 0.0039 - Q.width) / 0.001338226   # +11.1%  width < 0.0039
        - 0.09212198 * max(0.0, 35.4 - Q.pt_7) / 4.806336   # -9.2%  pt_7 < 35.4
        - 0.0787181 * max(0.0, 0.0687 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset) / 0.0003738793   # -7.9%  z_7 < 0.0687 and centroid_offset < 0.031
        + 0.06647185 * max(0.0, 0.0473 - Q.z_7) / 0.006549156   # +6.6%  z_7 < 0.0473
        - 0.0584289 * max(0.0, 0.021 - Q.zdr_0) / 0.008902543   # -5.8%  zdr_0 < 0.021
        - 0.05255302 * max(0.0, 0.215 - Q.LHA) * max(0.0, 0.00121 - Q.lam1) / 3.093404e-05   # -5.3%  LHA < 0.215 and lam1 < 0.00121
        + 0.03502331 * max(0.0, 0.0708 - Q.z_7) * max(0.0, 45.4 - Q.mass_top3) / 0.6813633   # +3.5%  z_7 < 0.0708 and mass_top3 < 45.4
        - 0.03454603 * max(0.0, 0.215 - Q.LHA) * max(0.0, 6.78 - Q.log_sum_pt) / 0.003545687   # -3.5%  LHA < 0.215 and log_sum_pt < 6.78
        + 0.03183971 * max(0.0, Q.sum_pt_top5 - 530.0) * max(0.0, 43.2 - Q.pt_6) / 1260.215   # +3.2%  sum_pt_top5 > 530 and pt_6 < 43.2
        - 0.02974228 * max(0.0, Q.log_sum_pt - 6.83) / 0.009226693   # -3.0%  log_sum_pt > 6.83
        + 0.02662789 * max(0.0, 0.0286 - Q.z_7) / 0.001389273   # +2.7%  z_7 < 0.0286
        + 0.01835026 * max(0.0, Q.log_sum_pt - 6.73) * max(0.0, 0.0147 - Q.dr_2) / 0.0001288244   # +1.8%  log_sum_pt > 6.73 and dr_2 < 0.0147
        + 0.01123757 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.48e-05 - Q.mean_eta2) / 9.148015e-07   # +1.1%  log_sum_pt > 6.7 and mean_eta2 < 8.48e-05
        + 0.01015057 * max(0.0, Q.sum_pt_top5 - 438.0) * max(0.0, Q.sj2_dr - 0.176) / 3.679269   # +1.0%  sum_pt_top5 > 438 and sj2_dr > 0.176
        + 0.007950366 * max(0.0, 0.00764 - Q.girth) / 0.0002281396   # +0.8%  girth < 0.00764
        - 0.007020909 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.0244 - Q.dr_2) / 5.232945e-05   # -0.7%  log_sum_pt > 6.89 and dr_2 < 0.0244
        - 0.005108774 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.00334) / 3.900631e-05   # -0.5%  log_sum_pt > 6.7 and mean_phi > 0.00334
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 21.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.3079 * (0.124367
        - 0.212584 * max(0.0, 0.00867 - Q.girth2) / 0.0044409   # -21.3%  girth2 < 0.00867
        + 0.09989501 * max(0.0, 0.0118 - Q.lam1) / 0.007142795   # +10.0%  lam1 < 0.0118
        + 0.07857771 * max(0.0, 0.181 - Q.sj3_dr_max) / 0.05507651   # +7.9%  sj3_dr_max < 0.181
        - 0.07590339 * max(0.0, 0.144 - Q.max_dr) / 0.04701576   # -7.6%  max_dr < 0.144
        - 0.06111813 * max(0.0, 0.29 - Q.sj3_dr_max) / 0.1356561   # -6.1%  sj3_dr_max < 0.29
        + 0.05925094 * max(0.0, 0.111 - Q.tau1) * max(0.0, 0.0157 - Q.mean_phi2) / 0.0008093032   # +5.9%  tau1 < 0.111 and mean_phi2 < 0.0157
        + 0.0539252 * max(0.0, Q.centroid_offset - 0.00762) / 0.01083993   # +5.4%  centroid_offset > 0.00762
        - 0.04694213 * max(0.0, 64.5 - Q.mass) * max(0.0, 0.0534 - Q.z_dr_0p2_0p4) / 1.414764   # -4.7%  mass < 64.5 and z_dr_0p2_0p4 < 0.0534
        - 0.04487037 * max(0.0, 0.00109 - Q.lam2) / 0.0008771499   # -4.5%  lam2 < 0.00109
        - 0.03421512 * max(0.0, 0.0118 - Q.lam1) * max(0.0, 0.254 - Q.planar_flow) / 0.0006395195   # -3.4%  lam1 < 0.0118 and planar_flow < 0.254
        + 0.03246969 * max(0.0, 42.5 - Q.pt_6) * max(0.0, 6.76 - Q.log_sum_pt) / 1.174637   # +3.2%  pt_6 < 42.5 and log_sum_pt < 6.76
        + 0.02732635 * max(0.0, 0.00301 - Q.e2_sq) / 0.001064474   # +2.7%  e2_sq < 0.00301
        - 0.02494642 * max(0.0, Q.centroid_offset - 0.0193) / 0.005211331   # -2.5%  centroid_offset > 0.0193
        - 0.02393309 * max(0.0, Q.sj3_dr_min - 0.0225) / 0.02864966   # -2.4%  sj3_dr_min > 0.0225
        - 0.02244908 * max(0.0, 41.5 - Q.pt_6) * max(0.0, Q.z_7 - 0.0235) / 0.07325311   # -2.2%  pt_6 < 41.5 and z_7 > 0.0235
        + 0.01763846 * max(0.0, 22.5 - Q.mass) / 4.680429   # +1.8%  mass < 22.5
        - 0.01584214 * max(0.0, Q.sj3_pair_mass_min - 4.35) / 3.542107   # -1.6%  sj3_pair_mass_min > 4.35
        + 0.01458274 * max(0.0, Q.sj3_dr_max - 0.191) / 0.03817293   # +1.5%  sj3_dr_max > 0.191
        + 0.01030816 * max(0.0, Q.centroid_offset - 0.0103) * max(0.0, Q.psi_0p1 - 0.335) / 0.003358491   # +1.0%  centroid_offset > 0.0103 and psi_0p1 > 0.335
        - 0.006889077 * max(0.0, 0.0221 - Q.z_6) / 0.0003359079   # -0.7%  z_6 < 0.0221
        + 0.006664935 * max(0.0, 0.0032 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002474142   # +0.7%  lam2 < 0.0032 and n_dr_0_0p05 < 3
        + 0.005790464 * max(0.0, 0.171 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927) / 0.0007010376   # +0.6%  sj2_dr < 0.171 and eccentricity > 0.927
        + 0.005635418 * max(0.0, 19.1 - Q.pt_6) / 0.234072   # +0.6%  pt_6 < 19.1
        + 0.005611422 * max(0.0, Q.centroid_offset - 0.0213) * max(0.0, 0.01 - Q.mean_phi2) / 2.050903e-05   # +0.6%  centroid_offset > 0.0213 and mean_phi2 < 0.01
        + 0.003897736 * max(0.0, Q.sum_pt - 975.0) / 5.126702   # +0.4%  sum_pt > 975
        + 0.003780941 * max(0.0, 589.0 - Q.sum_pt) * max(0.0, 1.97 - Q.n_dr_0p2_0p4) / 31.59369   # +0.4%  sum_pt < 589 and n_dr_0p2_0p4 < 1.97
        - 0.00294264 * max(0.0, Q.sum_pt - 953.0) * max(0.0, 0.00311 - Q.mean_phi2) / 0.01703845   # -0.3%  sum_pt > 953 and mean_phi2 < 0.00311
        + 0.0007834753 * max(0.0, 625.0 - Q.sum_pt) * max(0.0, 24.6 - Q.pt_5) / 1.93444   # +0.1%  sum_pt < 625 and pt_5 < 24.6
        + 0.0006765076 * max(0.0, Q.centroid_offset - 0.017) * max(0.0, Q.pt_4 - 66.3) / 0.007834216   # +0.1%  centroid_offset > 0.017 and pt_4 > 66.3
        - 0.0005493121 * max(0.0, Q.mean_eta - 0.0349) * max(0.0, Q.M3 - 0.0492) / 9.673295e-06   # -0.1%  mean_eta > 0.0349 and M3 > 0.0492
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 33.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.99477 * (0.341229
        - 0.1869737 * max(0.0, Q.mass_over_sum_pt - 0.0108) / 0.05125909   # -18.7%  mass_over_sum_pt > 0.0108
        - 0.0958338 * max(0.0, 0.00664 - Q.width) / 0.002908793   # -9.6%  width < 0.00664
        - 0.07792176 * max(0.0, Q.mass_over_sum_pt - 0.108) / 0.005362211   # -7.8%  mass_over_sum_pt > 0.108
        - 0.06696948 * max(0.0, 0.0884 - Q.girth) / 0.03732151   # -6.7%  girth < 0.0884
        - 0.0658758 * max(0.0, Q.mass_over_sum_pt - 0.0909) / 0.00820305   # -6.6%  mass_over_sum_pt > 0.0909
        + 0.05007867 * max(0.0, Q.mass_over_sum_pt - 0.0803) / 0.01077477   # +5.0%  mass_over_sum_pt > 0.0803
        + 0.04396323 * max(0.0, Q.sj3_dr_max - 0.139) / 0.06497913   # +4.4%  sj3_dr_max > 0.139
        - 0.04052946 * max(0.0, 0.187 - Q.sj2_dr) / 0.0643827   # -4.1%  sj2_dr < 0.187
        + 0.03872998 * max(0.0, 0.156 - Q.sj2_dr) / 0.04685469   # +3.9%  sj2_dr < 0.156
        - 0.03326647 * max(0.0, Q.width - 0.00864) * max(0.0, 40.8 - Q.pt_6) / 0.01018816   # -3.3%  width > 0.00864 and pt_6 < 40.8
        + 0.02688721 * max(0.0, Q.girth2 - 0.0131) * max(0.0, 41.3 - Q.pt_6) / 0.007197043   # +2.7%  girth2 > 0.0131 and pt_6 < 41.3
        + 0.0257573 * max(0.0, Q.girth2 - 0.0134) / 0.001419146   # +2.6%  girth2 > 0.0134
        - 0.02541283 * max(0.0, Q.girth2 - 0.00439) / 0.003629845   # -2.5%  girth2 > 0.00439
        + 0.02372245 * max(0.0, Q.mass - 36.6) / 14.00068   # +2.4%  mass > 36.6
        + 0.01982648 * max(0.0, 0.178 - Q.max_dr) / 0.07094701   # +2.0%  max_dr < 0.178
        + 0.01844851 * max(0.0, Q.mass_over_sum_pt - 0.0911) * max(0.0, 40.8 - Q.pt_6) / 0.03646238   # +1.8%  mass_over_sum_pt > 0.0911 and pt_6 < 40.8
        + 0.01653002 * max(0.0, 0.0893 - Q.girth) * max(0.0, 0.00429 - Q.C2_b2) / 0.0001363918   # +1.7%  girth < 0.0893 and C2_b2 < 0.00429
        - 0.01447403 * max(0.0, 0.109 - Q.max_dr) / 0.02703524   # -1.4%  max_dr < 0.109
        - 0.01383598 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, 0.00426 - Q.C2_b2) / 2.734598e-05   # -1.4%  centroid_offset < 0.02 and C2_b2 < 0.00426
        - 0.01087739 * max(0.0, Q.sj3_dr_max - 0.261) / 0.01906053   # -1.1%  sj3_dr_max > 0.261
        + 0.009572524 * max(0.0, 0.0003 - Q.lam2) * max(0.0, 0.0408 - Q.tau21_b2) / 2.603326e-06   # +1.0%  lam2 < 0.0003 and tau21_b2 < 0.0408
        + 0.00912263 * max(0.0, Q.z_7 - 0.0342) / 0.02040274   # +0.9%  z_7 > 0.0342
        - 0.008949933 * max(0.0, 0.000271 - Q.lam2) / 0.0001709275   # -0.9%  lam2 < 0.000271
        - 0.008452903 * max(0.0, 0.00104 - Q.girth2) / 0.0002280591   # -0.8%  girth2 < 0.00104
        - 0.008111424 * max(0.0, Q.mass - 76.8) / 1.531922   # -0.8%  mass > 76.8
        - 0.008043372 * max(0.0, 0.0407 - Q.tau21_b2) / 0.01129887   # -0.8%  tau21_b2 < 0.0407
        + 0.008021913 * max(0.0, Q.girth2 - 0.00447) * max(0.0, 0.197 - Q.planar_flow) / 0.0002622145   # +0.8%  girth2 > 0.00447 and planar_flow < 0.197
        + 0.005994339 * max(0.0, 0.139 - Q.z_dr_0p1_0p2) / 0.08183782   # +0.6%  z_dr_0p1_0p2 < 0.139
        - 0.005644818 * max(0.0, Q.centroid_offset - 0.0384) / 0.001626223   # -0.6%  centroid_offset > 0.0384
        + 0.005560777 * max(0.0, 0.00109 - Q.girth2_top2) / 0.0003134948   # +0.6%  girth2_top2 < 0.00109
        + 0.004718736 * max(0.0, 0.0216 - Q.centroid_offset) * max(0.0, 0.0279 - Q.tau21_b2) / 5.569873e-05   # +0.5%  centroid_offset < 0.0216 and tau21_b2 < 0.0279
        + 0.004579526 * max(0.0, Q.sd_rg - 0.283) / 0.004717574   # +0.5%  sd_rg > 0.283
        + 0.004394164 * max(0.0, Q.mass_top5 - 53.0) / 2.071825   # +0.4%  mass_top5 > 53
        - 0.003440393 * max(0.0, 29.4 - Q.pt_7) / 2.288755   # -0.3%  pt_7 < 29.4
        - 0.002177009 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.eccentricity - 0.925) / 2.936783e-05   # -0.2%  girth2 > 0.0145 and eccentricity > 0.925
        - 0.001961698 * max(0.0, 0.0586 - Q.D2_b2) / 0.006174766   # -0.2%  D2_b2 < 0.0586
        - 0.001947793 * max(0.0, Q.mass - 76.2) * max(0.0, Q.zdr_6 - 0.00612) / 0.007112219   # -0.2%  mass > 76.2 and zdr_6 > 0.00612
        - 0.001353481 * max(0.0, 0.216 - Q.planar_flow) * max(0.0, 35.6 - Q.pt_6) / 0.2054075   # -0.1%  planar_flow < 0.216 and pt_6 < 35.6
        - 0.001025486 * max(0.0, Q.centroid_offset - 0.0258) * max(0.0, Q.n_pt_above_50 - 3.71) / 0.003486117   # -0.1%  centroid_offset > 0.0258 and n_pt_above_50 > 3.71
        - 0.0005770529 * max(0.0, 0.00107 - Q.girth2_top2) * max(0.0, 0.0257 - Q.tau21_b2) / 2.1676e-07   # -0.1%  girth2_top2 < 0.00107 and tau21_b2 < 0.0257
        - 0.0004355258 * max(0.0, Q.sj3_dr_max - 0.132) * max(0.0, 20.0 - Q.pt_6) / 0.01563421   # -0.0%  sj3_dr_max > 0.132 and pt_6 < 20
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 11.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.5151 * (-0.05488444
        - 0.3375245 * max(0.0, 0.0595 - Q.girth) / 0.01782857   # -33.8%  girth < 0.0595
        + 0.1777459 * max(0.0, 0.00491 - Q.girth2) / 0.00184393   # +17.8%  girth2 < 0.00491
        + 0.1237516 * max(0.0, 0.0617 - Q.tau1) * max(0.0, 0.00304 - Q.width) / 4.750042e-05   # +12.4%  tau1 < 0.0617 and width < 0.00304
        + 0.08413553 * max(0.0, 0.00355 - Q.width) / 0.001178624   # +8.4%  width < 0.00355
        + 0.08161944 * max(0.0, 0.00662 - Q.girth2) * max(0.0, 0.0259 - Q.centroid_offset) / 4.433284e-05   # +8.2%  girth2 < 0.00662 and centroid_offset < 0.0259
        - 0.0783098 * max(0.0, 31.3 - Q.mass) * max(0.0, 0.0291 - Q.centroid_offset) / 0.1497916   # -7.8%  mass < 31.3 and centroid_offset < 0.0291
        + 0.02692814 * max(0.0, 0.000136 - Q.lam2) * max(0.0, 5.27 - Q.D2_b2) / 0.0002768574   # +2.7%  lam2 < 0.000136 and D2_b2 < 5.27
        - 0.02590562 * max(0.0, 0.00535 - Q.girth2) * max(0.0, Q.centroid_offset - 0.00661) / 1.222565e-05   # -2.6%  girth2 < 0.00535 and centroid_offset > 0.00661
        + 0.0242745 * max(0.0, 0.0585 - Q.girth) * max(0.0, Q.width - 0.000656) / 6.655317e-06   # +2.4%  girth < 0.0585 and width > 0.000656
        - 0.01768887 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -1.8%  log_sum_pt > 6.7
        - 0.01550806 * max(0.0, 0.142 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0059) / 0.0002480234   # -1.6%  sj3_dr_max < 0.142 and centroid_offset > 0.0059
        - 0.006608069 * max(0.0, Q.sum_pt_top5 - 679.0) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.96781   # -0.7%  sum_pt_top5 > 679 and n_pt_above_10 < 8
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 24.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.07444 * (-0.2139199
        + 0.1463641 * max(0.0, 0.00365 - Q.girth2) / 0.001223483   # +14.6%  girth2 < 0.00365
        + 0.08940865 * max(0.0, 0.00601 - Q.girth2) / 0.002488397   # +8.9%  girth2 < 0.00601
        - 0.08789336 * max(0.0, 0.00327 - Q.e2_sq) / 0.001182113   # -8.8%  e2_sq < 0.00327
        + 0.0877897 * max(0.0, 986.0 - Q.sum_pt) / 274.4789   # +8.8%  sum_pt < 986
        - 0.08349886 * max(0.0, 0.141 - Q.sj3_dr_max) / 0.03654887   # -8.3%  sj3_dr_max < 0.141
        + 0.06997548 * max(0.0, 0.215 - Q.sj3_dr_max) / 0.0769233   # +7.0%  sj3_dr_max < 0.215
        + 0.05616316 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0267 - Q.centroid_offset) / 0.3086978   # +5.6%  mass < 54.9 and centroid_offset < 0.0267
        + 0.05138366 * max(0.0, 0.0185 - Q.centroid_offset) / 0.006796883   # +5.1%  centroid_offset < 0.0185
        - 0.04765764 * max(0.0, 52.0 - Q.mass) / 18.50534   # -4.8%  mass < 52
        + 0.04630321 * max(0.0, 0.000315 - Q.lam2) / 0.0002060487   # +4.6%  lam2 < 0.000315
        - 0.03036407 * max(0.0, 30.1 - Q.mass) / 7.528298   # -3.0%  mass < 30.1
        + 0.02930834 * max(0.0, 55.0 - Q.mass) * max(0.0, 6.84 - Q.log_sum_pt) / 5.469627   # +2.9%  mass < 55 and log_sum_pt < 6.84
        - 0.02788896 * max(0.0, Q.lam1 - 0.00485) * max(0.0, Q.eccentricity - 0.638) / 0.0007816195   # -2.8%  lam1 > 0.00485 and eccentricity > 0.638
        + 0.02014722 * max(0.0, 0.107 - Q.max_dr) / 0.02607705   # +2.0%  max_dr < 0.107
        - 0.01909325 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1) / 0.2286861   # -1.9%  centroid_offset < 0.019 and pt_1 < 164
        - 0.01820267 * max(0.0, 0.0549 - Q.girth) * max(0.0, Q.z_6 - 0.0338) / 0.0003175501   # -1.8%  girth < 0.0549 and z_6 > 0.0338
        + 0.01644657 * max(0.0, Q.girth2 - 0.0134) / 0.001419146   # +1.6%  girth2 > 0.0134
        + 0.01071778 * max(0.0, 0.15 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 32.9) / 0.444104   # +1.1%  sj3_dr_max < 0.15 and pt_6 > 32.9
        - 0.01048906 * max(0.0, 0.017 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.007515426   # -1.0%  centroid_offset < 0.017 and n_for_90pct > 5
        + 0.01044488 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd) / 0.0002306924   # +1.0%  centroid_offset < 0.0191 and z_2nd < 0.205
        + 0.007488402 * max(0.0, Q.e2 - 0.0456) / 0.004134841   # +0.7%  e2 > 0.0456
        - 0.00676539 * max(0.0, 0.00631 - Q.zdr_0) / 0.001011633   # -0.7%  zdr_0 < 0.00631
        + 0.006690863 * max(0.0, 0.0796 - Q.M3) / 0.01268337   # +0.7%  M3 < 0.0796
        - 0.005894934 * max(0.0, 0.00578 - Q.girth2) * max(0.0, 0.00437 - Q.mean_phi) / 1.493866e-05   # -0.6%  girth2 < 0.00578 and mean_phi < 0.00437
        + 0.004862696 * max(0.0, 0.000176 - Q.width) / 1.498933e-05   # +0.5%  width < 0.000176
        - 0.002115043 * max(0.0, 0.0457 - Q.tau1) * max(0.0, 0.0286 - Q.tau21_b2) / 1.104522e-05   # -0.2%  tau1 < 0.0457 and tau21_b2 < 0.0286
        - 0.002021774 * max(0.0, Q.D2 - 2.16) / 0.2863121   # -0.2%  D2 > 2.16
        + 0.001937864 * max(0.0, 0.046 - Q.tau1) * max(0.0, -0.0119 - Q.mean_phi) / 1.388482e-05   # +0.2%  tau1 < 0.046 and mean_phi < -0.0119
        + 0.001876355 * max(0.0, 0.0528 - Q.girth) * max(0.0, 0.0275 - Q.tau21_b2) / 1.132135e-05   # +0.2%  girth < 0.0528 and tau21_b2 < 0.0275
        - 0.0008061054 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.746 - Q.D2_b2) / 0.0007134755   # -0.1%  log_sum_pt > 6.89 and D2_b2 < 0.746
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.006;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.005715 * (0.2269576
        + 0.1590685 * max(0.0, 79.4 - Q.mass) / 40.3764   # +15.9%  mass < 79.4
        - 0.1306231 * max(0.0, 0.00598 - Q.lam1) / 0.002534925   # -13.1%  lam1 < 0.00598
        + 0.09690497 * max(0.0, Q.girth2 - 0.00749) / 0.002477695   # +9.7%  girth2 > 0.00749
        + 0.0786888 * max(0.0, Q.lam2 - 0.000278) / 0.0004273421   # +7.9%  lam2 > 0.000278
        - 0.07294681 * max(0.0, Q.LHA - 0.305) / 0.01744179   # -7.3%  LHA > 0.305
        - 0.06752614 * max(0.0, 0.00404 - Q.lam1) / 0.001442283   # -6.8%  lam1 < 0.00404
        + 0.0583164 * max(0.0, Q.tau1 - 0.057) / 0.0250643   # +5.8%  tau1 > 0.057
        - 0.05503941 * max(0.0, 0.00144 - Q.lam1) / 0.00037076   # -5.5%  lam1 < 0.00144
        + 0.04209855 * max(0.0, 0.00215 - Q.girth2_top3) / 0.0007781806   # +4.2%  girth2_top3 < 0.00215
        + 0.03451772 * max(0.0, 0.0242 - Q.zdr_0) * max(0.0, 0.248 - Q.z_dr_0p05_0p1) / 0.002102794   # +3.5%  zdr_0 < 0.0242 and z_dr_0p05_0p1 < 0.248
        - 0.02797168 * max(0.0, Q.lam2 - 0.000257) * max(0.0, Q.planar_flow - 0.055) / 0.0002791476   # -2.8%  lam2 > 0.000257 and planar_flow > 0.055
        - 0.02674343 * max(0.0, Q.sj3_dr_max - 0.184) / 0.04099712   # -2.7%  sj3_dr_max > 0.184
        + 0.02395705 * max(0.0, Q.sj3_pair_mass_min - 15.8) / 1.364522   # +2.4%  sj3_pair_mass_min > 15.8
        - 0.02286961 * max(0.0, 43.9 - Q.pt_7) / 10.61046   # -2.3%  pt_7 < 43.9
        - 0.02057923 * max(0.0, Q.e3 - 5.22e-05) / 5.502755e-05   # -2.1%  e3 > 5.22e-05
        - 0.018414 * max(0.0, 48.0 - Q.pt_7) * max(0.0, 6.56 - Q.log_sum_pt) / 1.427027   # -1.8%  pt_7 < 48 and log_sum_pt < 6.56
        + 0.01506129 * max(0.0, 45.6 - Q.pt_7) * max(0.0, 0.985 - Q.D2) / 1.72129   # +1.5%  pt_7 < 45.6 and D2 < 0.985
        - 0.01078921 * max(0.0, Q.sj3_pair_mass_min - 6.68) * max(0.0, Q.sj3_pairmin_over_m - 0.245) / 0.3957389   # -1.1%  sj3_pair_mass_min > 6.68 and sj3_pairmin_over_m > 0.245
        - 0.009741192 * max(0.0, Q.lam1 - 0.00707) * max(0.0, 0.36 - Q.D2_b2) / 0.0002831702   # -1.0%  lam1 > 0.00707 and D2_b2 < 0.36
        + 0.009729811 * max(0.0, Q.C2_b2 - 0.00763) / 0.00184727   # +1.0%  C2_b2 > 0.00763
        + 0.007247743 * max(0.0, 0.0205 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0125) / 3.153765e-05   # +0.7%  zdr_0 < 0.0205 and centroid_offset > 0.0125
        - 0.004188965 * max(0.0, Q.mass_top5 - 51.0) / 2.405466   # -0.4%  mass_top5 > 51
        - 0.003591937 * max(0.0, Q.sum_pt - 988.0) / 4.42251   # -0.4%  sum_pt > 988
        - 0.00338449 * max(0.0, Q.girth2 - 0.0258) / 0.0002611318   # -0.3%  girth2 > 0.0258
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 34.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.18511 * (-0.0421236
        + 0.3290119 * max(0.0, 0.00872 - Q.width) / 0.004480998   # +32.9%  width < 0.00872
        - 0.1449566 * max(0.0, 0.00814 - Q.e2_sq) / 0.004271861   # -14.5%  e2_sq < 0.00814
        - 0.08913286 * max(0.0, 0.00838 - Q.lam1) / 0.004303695   # -8.9%  lam1 < 0.00838
        - 0.05169461 * max(0.0, 0.0737 - Q.girth) / 0.0263366   # -5.2%  girth < 0.0737
        + 0.04799308 * max(0.0, 0.257 - Q.sj3_dr_max) / 0.1086522   # +4.8%  sj3_dr_max < 0.257
        + 0.04242756 * max(0.0, 0.0377 - Q.centroid_offset) / 0.02221119   # +4.2%  centroid_offset < 0.0377
        - 0.03870532 * max(0.0, 0.167 - Q.sj3_dr_max) / 0.04776699   # -3.9%  sj3_dr_max < 0.167
        - 0.03797738 * max(0.0, 70.5 - Q.mass) / 32.45652   # -3.8%  mass < 70.5
        + 0.03720988 * max(0.0, Q.z_7 - 0.0169) / 0.03553139   # +3.7%  z_7 > 0.0169
        + 0.03355923 * max(0.0, 0.0134 - Q.girth2) / 0.008373912   # +3.4%  girth2 < 0.0134
        - 0.01883534 * max(0.0, 0.0377 - Q.centroid_offset) * max(0.0, 892.0 - Q.sum_pt) / 3.336207   # -1.9%  centroid_offset < 0.0377 and sum_pt < 892
        - 0.01853977 * max(0.0, Q.centroid_offset - 0.05) * max(0.0, Q.tau3 - 0.000494) / 7.361023e-06   # -1.9%  centroid_offset > 0.05 and tau3 > 0.000494
        + 0.01754274 * max(0.0, 0.242 - Q.planar_flow) / 0.1026884   # +1.8%  planar_flow < 0.242
        - 0.01463098 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -1.5%  centroid_offset > 0.05
        - 0.01273436 * max(0.0, 0.0137 - Q.centroid_offset) * max(0.0, 17.2 - Q.sj3_pair_mass_min) / 0.05151779   # -1.3%  centroid_offset < 0.0137 and sj3_pair_mass_min < 17.2
        + 0.008992338 * max(0.0, 15.7 - Q.mass) / 2.459232   # +0.9%  mass < 15.7
        - 0.007792443 * max(0.0, Q.pt_7 - 29.6) / 7.399598   # -0.8%  pt_7 > 29.6
        + 0.00745422 * max(0.0, Q.centroid_offset - 0.0494) * max(0.0, 0.619 - Q.tau32) / 0.0001481531   # +0.7%  centroid_offset > 0.0494 and tau32 < 0.619
        - 0.007203056 * max(0.0, 0.0231 - Q.girth) / 0.003252804   # -0.7%  girth < 0.0231
        - 0.005650071 * max(0.0, Q.centroid_offset - 0.0492) * max(0.0, Q.zdr_6 - 0.00714) / 3.382632e-06   # -0.6%  centroid_offset > 0.0492 and zdr_6 > 0.00714
        - 0.005150383 * max(0.0, 0.241 - Q.planar_flow) * max(0.0, 835.0 - Q.sum_pt) / 13.13928   # -0.5%  planar_flow < 0.241 and sum_pt < 835
        + 0.004731442 * max(0.0, 0.0138 - Q.centroid_offset) * max(0.0, 0.571 - Q.D2_b2) / 0.0007593655   # +0.5%  centroid_offset < 0.0138 and D2_b2 < 0.571
        - 0.003584985 * max(0.0, Q.n_dr_0p1_0p2 - 2.92) / 0.3491542   # -0.4%  n_dr_0p1_0p2 > 2.92
        - 0.003397939 * max(0.0, 0.0384 - Q.centroid_offset) * max(0.0, 0.0324 - Q.absphi_0) / 0.0003541431   # -0.3%  centroid_offset < 0.0384 and absphi_0 < 0.0324
        - 0.003048529 * max(0.0, 0.268 - Q.planar_flow) * max(0.0, 36.7 - Q.pt_7) / 0.569477   # -0.3%  planar_flow < 0.268 and pt_7 < 36.7
        - 0.001672767 * max(0.0, 0.233 - Q.planar_flow) * max(0.0, 9.53e-06 - Q.e3) / 1.282146e-07   # -0.2%  planar_flow < 0.233 and e3 < 9.53e-06
        - 0.001628962 * max(0.0, 0.0383 - Q.centroid_offset) * max(0.0, -0.00159 - Q.mean_phi) / 5.204323e-05   # -0.2%  centroid_offset < 0.0383 and mean_phi < -0.00159
        - 0.001429282 * max(0.0, 25.4 - Q.pt_6) / 0.7070937   # -0.1%  pt_6 < 25.4
        + 0.000996242 * max(0.0, 0.264 - Q.sj3_dr_max) * max(0.0, -0.023 - Q.phi_0) / 0.0003709874   # +0.1%  sj3_dr_max < 0.264 and phi_0 < -0.023
        + 0.0009404412 * max(0.0, Q.z_dr_0p05_0p1 - 0.826) / 0.01291128   # +0.1%  z_dr_0p05_0p1 > 0.826
        + 0.0008269984 * max(0.0, 0.00431 - Q.girth2) * max(0.0, Q.sj3_dr13 - 0.162) / 5.926841e-06   # +0.1%  girth2 < 0.00431 and sj3_dr13 > 0.162
        - 0.0005482979 * max(0.0, Q.sum_pt - 971.0) / 5.370666   # -0.1%  sum_pt > 971
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4302;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4302274 * (-1.420179
        + 0.4612572 * max(0.0, Q.girth2 - 0.0188) / 0.0007632519   # +46.1%  girth2 > 0.0188
        - 0.1334001 * max(0.0, Q.e2 - 0.0634) / 0.001765919   # -13.3%  e2 > 0.0634
        - 0.08557604 * max(0.0, Q.girth2_top2 - 0.014) / 0.001132836   # -8.6%  girth2_top2 > 0.014
        - 0.0760801 * max(0.0, Q.girth2 - 0.0188) * max(0.0, 53.4 - Q.pt_7) / 0.01441927   # -7.6%  girth2 > 0.0188 and pt_7 < 53.4
        + 0.07405195 * max(0.0, Q.mass - 91.2) / 0.5835014   # +7.4%  mass > 91.2
        + 0.05601134 * max(0.0, Q.mass - 69.6) * max(0.0, Q.centroid_offset - 0.00948) / 0.04084341   # +5.6%  mass > 69.6 and centroid_offset > 0.00948
        + 0.04199528 * max(0.0, Q.centroid_offset - 0.0499) / 0.0008065856   # +4.2%  centroid_offset > 0.0499
        - 0.0384562 * max(0.0, Q.zdr_0 - 0.0398) / 0.000570514   # -3.8%  zdr_0 > 0.0398
        + 0.03317177 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.lam2 - 0.000537) / 2.837257e-06   # +3.3%  girth2 > 0.0188 and lam2 > 0.000537
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 17.4;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.39809 * (0.04103899
        + 0.3038501 * max(0.0, 0.148 - Q.girth) / 0.0902118   # +30.4%  girth < 0.148
        - 0.1620377 * max(0.0, 0.079 - Q.e2) / 0.05097915   # -16.2%  e2 < 0.079
        + 0.15852 * max(0.0, 0.0161 - Q.lam1) / 0.01090097   # +15.9%  lam1 < 0.0161
        - 0.08800329 * max(0.0, 0.148 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt) / 0.02030622   # -8.8%  girth < 0.148 and log_sum_pt < 6.81
        - 0.05693212 * max(0.0, 5.38e-05 - Q.e3) * max(0.0, 0.0377 - Q.centroid_offset) / 8.394154e-07   # -5.7%  e3 < 5.38e-05 and centroid_offset < 0.0377
        + 0.04139176 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, 38.1 - Q.pt_7) / 666.7941   # +4.1%  sum_pt_top5 > 667 and pt_7 < 38.1
        - 0.03575974 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.2 - Q.pt_7) / 0.6303457   # -3.6%  girth < 0.145 and pt_7 < 38.2
        + 0.034834 * max(0.0, 5.05e-05 - Q.e3) / 3.076371e-05   # +3.5%  e3 < 5.05e-05
        - 0.0201575 * max(0.0, 0.148 - Q.girth) * max(0.0, 0.000536 - Q.lam2) / 3.994329e-05   # -2.0%  girth < 0.148 and lam2 < 0.000536
        - 0.0176994 * max(0.0, Q.sum_pt_top5 - 674.0) / 41.27824   # -1.8%  sum_pt_top5 > 674
        - 0.01411575 * max(0.0, 0.028 - Q.z_7) / 0.001299403   # -1.4%  z_7 < 0.028
        - 0.01115099 * max(0.0, Q.mass - 49.1) / 7.951063   # -1.1%  mass > 49.1
        + 0.0111276 * max(0.0, Q.sum_pt_top5 - 794.0) / 14.13131   # +1.1%  sum_pt_top5 > 794
        - 0.009482035 * max(0.0, 31.9 - Q.pt_6) * max(0.0, 0.0624 - Q.z_7) / 0.0743105   # -0.9%  pt_6 < 31.9 and z_7 < 0.0624
        + 0.008279015 * max(0.0, 0.153 - Q.girth) * max(0.0, Q.z_7 - 0.0616) / 0.0003674466   # +0.8%  girth < 0.153 and z_7 > 0.0616
        + 0.007459007 * max(0.0, 6.94e-05 - Q.e3) * max(0.0, Q.D3 - 0.0598) / 6.179642e-05   # +0.7%  e3 < 6.94e-05 and D3 > 0.0598
        + 0.004927589 * max(0.0, 0.15 - Q.girth) * max(0.0, 0.0746 - Q.M3) / 0.0006858451   # +0.5%  girth < 0.15 and M3 < 0.0746
        + 0.003783779 * max(0.0, 0.142 - Q.girth) * max(0.0, Q.tau2 - 0.00886) / 0.0002342723   # +0.4%  girth < 0.142 and tau2 > 0.00886
        - 0.003707803 * max(0.0, 0.0283 - Q.z_5) / 0.0003862796   # -0.4%  z_5 < 0.0283
        - 0.003058593 * max(0.0, Q.sum_pt - 1000.0) * max(0.0, 70.0 - Q.pt_6) / 114.1924   # -0.3%  sum_pt > 1000 and pt_6 < 70
        - 0.002326086 * max(0.0, Q.pt_7 - 48.4) / 0.7087472   # -0.2%  pt_7 > 48.4
        - 0.00139613 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.sj2_mass1 - 30.4) / 0.01537341   # -0.1%  girth < 0.152 and sj2_mass1 > 30.4
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 40.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.79369 * (-0.01688987
        - 0.09780374 * max(0.0, Q.width - 0.0067) / 0.002695795   # -9.8%  width > 0.0067
        - 0.0861378 * max(0.0, Q.girth2 - 0.00856) / 0.002238139   # -8.6%  girth2 > 0.00856
        + 0.07423989 * max(0.0, Q.girth - 0.0272) / 0.0359255   # +7.4%  girth > 0.0272
        - 0.05721398 * max(0.0, Q.e2 - 0.0167) / 0.01587734   # -5.7%  e2 > 0.0167
        - 0.0507132 * max(0.0, 0.325 - Q.sd_rg) / 0.2085463   # -5.1%  sd_rg < 0.325
        + 0.04631467 * max(0.0, Q.girth - 0.041) / 0.02695216   # +4.6%  girth > 0.041
        - 0.04519118 * max(0.0, Q.width - 0.0131) / 0.001463107   # -4.5%  width > 0.0131
        + 0.04135392 * max(0.0, Q.mass_over_sum_pt - 0.0804) / 0.01074509   # +4.1%  mass_over_sum_pt > 0.0804
        + 0.03952882 * max(0.0, Q.e2_sq - 0.0117) / 0.001427015   # +4.0%  e2_sq > 0.0117
        - 0.03757155 * max(0.0, Q.girth - 0.0868) / 0.007941359   # -3.8%  girth > 0.0868
        + 0.03745634 * max(0.0, 49.6 - Q.sd_mass) / 22.40444   # +3.7%  sd_mass < 49.6
        + 0.03503702 * max(0.0, Q.mass_over_sum_pt - 0.0906) / 0.008261789   # +3.5%  mass_over_sum_pt > 0.0906
        + 0.03394643 * max(0.0, Q.width - 0.00353) / 0.004084956   # +3.4%  width > 0.00353
        - 0.03152544 * max(0.0, 7.93e-05 - Q.e3) / 5.449318e-05   # -3.2%  e3 < 7.93e-05
        + 0.03106743 * max(0.0, Q.sj2_dr - 0.158) / 0.03985393   # +3.1%  sj2_dr > 0.158
        + 0.02166983 * max(0.0, Q.e2 - 0.0499) / 0.003426328   # +2.2%  e2 > 0.0499
        + 0.02165239 * max(0.0, Q.girth - 0.0802) / 0.009446854   # +2.2%  girth > 0.0802
        - 0.0196012 * max(0.0, Q.centroid_offset - 0.0497) / 0.0008167572   # -2.0%  centroid_offset > 0.0497
        - 0.0184319 * max(0.0, 74.8 - Q.sd_mass) / 42.24187   # -1.8%  sd_mass < 74.8
        + 0.01646457 * max(0.0, Q.z_dr_0_0p05 - 0.152) / 0.447767   # +1.6%  z_dr_0_0p05 > 0.152
        - 0.01645341 * max(0.0, Q.sj2_dr - 0.0606) / 0.1009317   # -1.6%  sj2_dr > 0.0606
        - 0.01118146 * max(0.0, Q.sj2_dr - 0.197) / 0.02413403   # -1.1%  sj2_dr > 0.197
        - 0.01041425 * max(0.0, Q.girth - 0.102) / 0.005391318   # -1.0%  girth > 0.102
        - 0.01031303 * max(0.0, 0.00403 - Q.C2_b2) / 0.002881553   # -1.0%  C2_b2 < 0.00403
        - 0.01013651 * max(0.0, 717.0 - Q.sum_pt) / 70.68471   # -1.0%  sum_pt < 717
        + 0.009298976 * max(0.0, 5.06 - Q.n_dr_0_0p05) / 1.975727   # +0.9%  n_dr_0_0p05 < 5.06
        + 0.008979856 * max(0.0, 0.000504 - Q.lam2) / 0.0003626945   # +0.9%  lam2 < 0.000504
        - 0.008499143 * max(0.0, Q.sd_rg - 0.234) / 0.01057047   # -0.8%  sd_rg > 0.234
        + 0.008176201 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0165 - Q.lam1) / 0.0003379305   # +0.8%  planar_flow < 0.115 and lam1 < 0.0165
        - 0.006708114 * max(0.0, Q.centroid_offset - 0.0269) / 0.00324613   # -0.7%  centroid_offset > 0.0269
        - 0.00632251 * max(0.0, Q.psi_0p1 - 0.974) / 0.01146305   # -0.6%  psi_0p1 > 0.974
        - 0.006249033 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.00746 - Q.lam1) / 6.069551e-05   # -0.6%  planar_flow < 0.11 and lam1 < 0.00746
        + 0.005937345 * max(0.0, Q.sj2_dr - 0.159) * max(0.0, 0.0881 - Q.D2_b2) / 0.0006407572   # +0.6%  sj2_dr > 0.159 and D2_b2 < 0.0881
        - 0.004843818 * max(0.0, 0.163 - Q.z_dr_0p05_0p1) / 0.08628701   # -0.5%  z_dr_0p05_0p1 < 0.163
        + 0.004243202 * max(0.0, 0.0677 - Q.z_dr_0p1_0p2) / 0.03667286   # +0.4%  z_dr_0p1_0p2 < 0.0677
        + 0.004131123 * max(0.0, Q.centroid_offset - 0.0499) * max(0.0, 0.000872 - Q.C2_b2) / 1.452791e-07   # +0.4%  centroid_offset > 0.0499 and C2_b2 < 0.000872
        - 0.004123753 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.0886 - Q.D2_b2) / 0.001001328   # -0.4%  sj2_dr > 0.13 and D2_b2 < 0.0886
        + 0.002870038 * max(0.0, 0.112 - Q.planar_flow) * max(0.0, 0.00601 - Q.width) / 3.234239e-05   # +0.3%  planar_flow < 0.112 and width < 0.00601
        - 0.002633013 * max(0.0, Q.mass - 68.9) / 2.515464   # -0.3%  mass > 68.9
        + 0.002541702 * max(0.0, Q.centroid_offset - 0.0267) * max(0.0, 152.0 - Q.pt_1) / 0.1895528   # +0.3%  centroid_offset > 0.0267 and pt_1 < 152
        + 0.002347285 * max(0.0, 5.56e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.162) / 4.392405e-07   # +0.2%  e3 < 5.56e-05 and sj3_dr23 > 0.162
        - 0.002205907 * max(0.0, Q.sj2_dr - 0.198) * max(0.0, 0.0872 - Q.D2_b2) / 0.0003408602   # -0.2%  sj2_dr > 0.198 and D2_b2 < 0.0872
        - 0.001915619 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 0.0179 - Q.centroid_offset) / 0.000195363   # -0.2%  planar_flow < 0.106 and centroid_offset < 0.0179
        + 0.001480758 * max(0.0, 0.627 - Q.z_dr_0p05_0p1) * max(0.0, 714.0 - Q.sum_pt) / 23.78173   # +0.1%  z_dr_0p05_0p1 < 0.627 and sum_pt < 714
        - 0.001307627 * max(0.0, Q.sj2_dr - 0.203) * max(0.0, 0.105 - Q.z_5) / 0.0007140958   # -0.1%  sj2_dr > 0.203 and z_5 < 0.105
        - 0.001021425 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 652.0 - Q.sum_pt_top5) / 2.354108   # -0.1%  planar_flow < 0.1 and sum_pt_top5 < 652
        + 0.000988567 * max(0.0, Q.sj2_dr - 0.197) * max(0.0, 0.0517 - Q.dr_2) / 0.0001395408   # +0.1%  sj2_dr > 0.197 and dr_2 < 0.0517
        - 0.0008496274 * max(0.0, Q.sj2_dr - 0.164) * max(0.0, 0.0332 - Q.dr_2) / 6.09129e-05   # -0.1%  sj2_dr > 0.164 and dr_2 < 0.0332
        - 0.0007522802 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.00684) / 0.0003106102   # -0.1%  planar_flow < 0.109 and mean_eta > -0.00684
        - 0.0001531271 * max(0.0, Q.z_dr_0p05_0p1 - 0.745) * max(0.0, Q.zdr_5 - 0.0154) / 1.958188e-06   # -0.0%  z_dr_0p05_0p1 > 0.745 and zdr_5 > 0.0154
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 28.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.59045 * (4.651903e-05
        + 0.2058315 * max(0.0, 0.0133 - Q.width) / 0.008288473   # +20.6%  width < 0.0133
        - 0.1701065 * max(0.0, 0.00818 - Q.e2_sq) / 0.004303912   # -17.0%  e2_sq < 0.00818
        - 0.1559868 * max(0.0, 0.102 - Q.girth) / 0.04868705   # -15.6%  girth < 0.102
        + 0.09509343 * max(0.0, 0.0409 - Q.e2) / 0.01742797   # +9.5%  e2 < 0.0409
        - 0.09052083 * max(0.0, 0.199 - Q.sj2_dr) / 0.07249388   # -9.1%  sj2_dr < 0.199
        + 0.07908466 * max(0.0, 0.00708 - Q.mass_over_sum_pt_sq) / 0.003452009   # +7.9%  mass_over_sum_pt_sq < 0.00708
        + 0.07435835 * max(0.0, 0.159 - Q.sj2_dr) / 0.04831679   # +7.4%  sj2_dr < 0.159
        - 0.06584676 * max(0.0, 0.00739 - Q.girth2) / 0.003447965   # -6.6%  girth2 < 0.00739
        - 0.04139795 * max(0.0, 0.000513 - Q.girth2_top2) / 0.0001106155   # -4.1%  girth2_top2 < 0.000513
        + 0.005075164 * max(0.0, 0.229 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.116081   # +0.5%  N2 < 0.229 and n_dr_0p1_0p2 < 4
        - 0.005042804 * max(0.0, 0.223 - Q.N2) * max(0.0, 0.548 - Q.z_dr_0p05_0p1) / 0.01037238   # -0.5%  N2 < 0.223 and z_dr_0p05_0p1 < 0.548
        + 0.004039063 * max(0.0, 0.133 - Q.mass_over_sum_pt) * max(0.0, 0.711 - Q.D2) / 0.003689413   # +0.4%  mass_over_sum_pt < 0.133 and D2 < 0.711
        + 0.00366069 * max(0.0, 0.218 - Q.N2) * max(0.0, Q.sum_pt_top5 - 498.0) / 5.782364   # +0.4%  N2 < 0.218 and sum_pt_top5 > 498
        - 0.003427945 * max(0.0, 0.00752 - Q.girth2) * max(0.0, 0.731 - Q.D2) / 8.59706e-05   # -0.3%  girth2 < 0.00752 and D2 < 0.731
        - 0.0005275015 * max(0.0, Q.sj3_pair_mass_max - 81.1) / 0.3077858   # -0.1%  sj3_pair_mass_max > 81.1
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9383411764705882, 0.7896544117647059, 2.0633393907563025, 1.0210233193277312, 1.1384882352941177, 1.7555970588235295, 0.9330218487394958, 1.9468060924369748, 0.28222415966386555, 3.031227731092437, 1.995738025210084, 2.0280650210084032, 0.09242983193277311, 3.366158193277311, 0.4058917016806723, 0.33249579831932774]
T = [2.3294594447544648, 1.3631895319065126, 3.2706752248555673, 2.5348310645351897, 2.8747213038340336]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +4% ...
            + 0.3805995 * h[2] / H_AVG[2]
            + 0.2236537 * h[9] / H_AVG[9]
            - 0.1413094 * h[5] / H_AVG[5]
            + 0.1324165 * h[1] / H_AVG[1]
            - 0.06293984 * h[0] / H_AVG[0]
            + 0.04380813 * h[6] / H_AVG[6]
            - 0.01527297 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5645933 * h[9] / H_AVG[9]
            - 0.1830026 * h[10] / H_AVG[10]
            + 0.08555504 * h[6] / H_AVG[6]
            - 0.07829672 * h[4] / H_AVG[4]
            + 0.06036843 * h[5] / H_AVG[5]
            + 0.01524439 * h[15] / H_AVG[15]
            + 0.01293951 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +23%, n3 -16%, n7 +13%, n0 +10%, n14 -9%, n6 -9% ...
            + 0.2325282 * h[11] / H_AVG[11]
            - 0.1560875 * h[3] / H_AVG[3]
            + 0.1302067 * h[7] / H_AVG[7]
            + 0.09862024 * h[0] / H_AVG[0]
            - 0.09307521 * h[14] / H_AVG[14]
            - 0.08914652 * h[6] / H_AVG[6]
            + 0.07236518 * h[13] / H_AVG[13]
            - 0.06989103 * h[15] / H_AVG[15]
            - 0.02896217 * h[9] / H_AVG[9]
            - 0.02157232 * h[8] / H_AVG[8]
            - 0.007544834 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +36%, n3 -23%, n6 -14%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3600103 * h[7] / H_AVG[7]
            - 0.2265735 * h[3] / H_AVG[3]
            - 0.1380302 * h[6] / H_AVG[6]
            + 0.0726229 * h[13] / H_AVG[13]
            + 0.06004715 * h[14] / H_AVG[14]
            + 0.03894019 * h[1] / H_AVG[1]
            - 0.0373697 * h[9] / H_AVG[9]
            + 0.03508888 * h[4] / H_AVG[4]
            - 0.02049544 * h[15] / H_AVG[15]
            + 0.01082171 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -48%, n10 +26%, n5 -15%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4756989 * h[13] / H_AVG[13]
            + 0.2603389 * h[10] / H_AVG[10]
            - 0.1526754 * h[5] / H_AVG[5]
            + 0.04950429 * h[4] / H_AVG[4]
            + 0.02219831 * h[3] / H_AVG[3]
            + 0.01840771 * h[8] / H_AVG[8]
            - 0.01607631 * h[12] / H_AVG[12]
            + 0.005100175 * h[0] / H_AVG[0]
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
