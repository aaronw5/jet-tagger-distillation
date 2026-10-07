"""JEDI-linear jet tagger, 8 particles, 3 features: one term per observable per neuron (from the 399), re-tuned on the network's predictions (all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  15.9%   (on for 89% of jets)
  neuron  9:  11.7%   (on for 70% of jets)
  neuron  3:   9.8%   (on for 26% of jets)
  neuron  7:   9.6%   (on for 44% of jets)
  neuron 10:   9.5%   (on for 97% of jets)
  neuron  5:   6.6%   (on for 59% of jets)
  neuron  6:   6.4%   (on for 33% of jets)
  neuron  2:   6.3%   (on for 91% of jets)
  neuron 11:   5.8%   (on for 82% of jets)
  neuron  0:   4.1%   (on for 41% of jets)
  neuron 14:   3.6%   (on for 27% of jets)
  neuron  1:   3.5%   (on for 66% of jets)
  neuron 15:   3.2%   (on for 39% of jets)
  neuron  4:   2.8%   (on for 54% of jets)
  neuron  8:   1.0%   (on for 27% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 88.8% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
  Q.zdr_5                  pT share × ΔR of particle 5 (its term in ΣzΔR)
  Q.zdr_6                  pT share × ΔR of particle 6 (its term in ΣzΔR)
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
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
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
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
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
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 15.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.67045 * (-0.03287421
        + 0.2653694 * max(0.0, 0.013 - Q.sum_z_dr2) / 0.008032715   # +26.5%  sum_z_dr2 < 0.013
        - 0.138734 * max(0.0, 0.00444 - Q.lam1_plus_lam2) / 0.001599645   # -13.9%  lam1_plus_lam2 < 0.00444
        - 0.1257563 * max(0.0, 53.20476 - Q.mass) / 19.2771   # -12.6%  mass < 53.2
        + 0.1034179 * max(0.0, 0.03227056 - Q.e2) / 0.01167233   # +10.3%  e2 < 0.03227
        - 0.08304415 * max(0.0, 0.0751 - Q.sum_z_dr) / 0.02728748   # -8.3%  sum_z_dr < 0.0751
        - 0.06622814 * max(0.0, 0.00151 - Q.C2_b2) / 0.0008400094   # -6.6%  C2_b2 < 0.00151
        - 0.04934743 * max(0.0, Q.sj3_dr_max - 0.2122942) / 0.03094364   # -4.9%  sj3_dr_max > 0.2123
        + 0.04653526 * max(0.0, 0.00414 - Q.lam1_plus_lam2) * max(0.0, 0.0364 - Q.C3) / 3.605657e-05   # +4.7%  lam1_plus_lam2 < 0.00414 and C3 < 0.0364
        + 0.02677201 * max(0.0, 0.159 - Q.planar_flow) / 0.05648946   # +2.7%  planar_flow < 0.159
        + 0.02079729 * max(0.0, 0.0133 - Q.sum_z_dr2) * max(0.0, 1.02 - Q.D2) / 0.001058894   # +2.1%  sum_z_dr2 < 0.0133 and D2 < 1.02
        - 0.01205296 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, Q.z_7 - 0.0178) / 0.000540235   # -1.2%  log_sum_pt > 6.68 and z_7 > 0.0178
        - 0.009905756 * max(0.0, Q.sum_pt - 906.0) / 11.77315   # -1.0%  sum_pt > 906
        - 0.008101305 * max(0.0, 0.245 - Q.sj3_dr_max) * max(0.0, 0.058 - Q.z_7) / 0.001371719   # -0.8%  sj3_dr_max < 0.245 and z_7 < 0.058
        - 0.007842794 * max(0.0, Q.centroid_offset - 0.03782766) / 0.001682668   # -0.8%  centroid_offset > 0.03783
        - 0.007663277 * max(0.0, 0.00649 - Q.lam1) * max(0.0, 0.843 - Q.D2) / 6.964666e-05   # -0.8%  lam1 < 0.00649 and D2 < 0.843
        + 0.007276916 * max(0.0, 0.0134 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.00668) / 0.0001171469   # +0.7%  sum_z_dr2 < 0.0134 and phi_0 > -0.00668
        + 0.006504909 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0746 - Q.dr_4) / 0.001897083   # +0.7%  log_sum_pt > 6.68 and dr_4 < 0.0746
        + 0.006416281 * max(0.0, 8.01e-05 - Q.lam2) * max(0.0, 0.257 - Q.D2_b2) / 2.60375e-06   # +0.6%  lam2 < 8.01e-05 and D2_b2 < 0.257
        - 0.004478235 * max(0.0, Q.centroid_offset - 0.0203) * max(0.0, 0.0609 - Q.z_7) / 2.825186e-05   # -0.4%  centroid_offset > 0.0203 and z_7 < 0.0609
        + 0.00133518 * max(0.0, Q.sum_pt - 915.0) * max(0.0, Q.pt_7 - 35.4) / 54.76743   # +0.1%  sum_pt > 915 and pt_7 > 35.4
        + 0.001278634 * max(0.0, 0.165 - Q.planar_flow) * max(0.0, 0.0237 - Q.z_7) / 2.627329e-05   # +0.1%  planar_flow < 0.165 and z_7 < 0.0237
        - 0.001141841 * max(0.0, 0.142 - Q.planar_flow) * max(0.0, 0.023 - Q.dr_2) / 2.382786e-05   # -0.1%  planar_flow < 0.142 and dr_2 < 0.023
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 7.016;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.015709 * (0.3296225
        - 0.215353 * max(0.0, 0.0991 - Q.sum_z_dr) / 0.04622227   # -21.5%  sum_z_dr < 0.0991
        + 0.1831325 * max(0.0, Q.log_sum_pt - 6.538933) / 0.1040764   # +18.3%  log_sum_pt > 6.539
        + 0.1446597 * max(0.0, 0.00591 - Q.mass_over_sum_pt_sq) / 0.00262863   # +14.5%  mass_over_sum_pt_sq < 0.00591
        - 0.1386735 * max(0.0, 0.0801183 - Q.z_7) / 0.02876792   # -13.9%  z_7 < 0.08012
        - 0.1243159 * max(0.0, 0.00871 - Q.sum_z_dr2) / 0.004472974   # -12.4%  sum_z_dr2 < 0.00871
        - 0.05228622 * max(0.0, Q.sum_pt_top5 - 580.375) / 79.19672   # -5.2%  sum_pt_top5 > 580.4
        - 0.02182437 * max(0.0, 0.00085 - Q.lam1) / 0.0001816315   # -2.2%  lam1 < 0.00085
        - 0.02079684 * max(0.0, 0.0607 - Q.z_7) * max(0.0, 1.52 - Q.D2) / 0.00445904   # -2.1%  z_7 < 0.0607 and D2 < 1.52
        + 0.01579642 * max(0.0, Q.pt_7 - 33.5) * max(0.0, Q.sj2_dr - 0.143) / 0.1786707   # +1.6%  pt_7 > 33.5 and sj2_dr > 0.143
        - 0.01519652 * max(0.0, 0.00962 - Q.lam1) * max(0.0, 0.15 - Q.planar_flow) / 0.0001925416   # -1.5%  lam1 < 0.00962 and planar_flow < 0.15
        - 0.01476764 * max(0.0, Q.pt_7 - 33.4) * max(0.0, 53.4 - Q.pt_6) / 18.70932   # -1.5%  pt_7 > 33.4 and pt_6 < 53.4
        + 0.0146827 * max(0.0, Q.log_sum_pt - 6.48) * max(0.0, 1.4 - Q.D2) / 0.04255068   # +1.5%  log_sum_pt > 6.48 and D2 < 1.4
        + 0.01200751 * max(0.0, 0.00939 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0216) / 7.843254e-06   # +1.2%  sum_z_dr2 < 0.00939 and centroid_offset > 0.0216
        - 0.01067955 * max(0.0, Q.sj3_dr_max - 0.104) * max(0.0, Q.sj3_pair_mass_min - 2.6) / 0.9569717   # -1.1%  sj3_dr_max > 0.104 and sj3_pair_mass_min > 2.6
        + 0.008982575 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, Q.centroid_offset - 0.0126) / 0.0006882835   # +0.9%  log_sum_pt > 6.37 and centroid_offset > 0.0126
        - 0.004228887 * max(0.0, Q.pt_7 - 53.5) / 0.3284992   # -0.4%  pt_7 > 53.5
        - 0.002616242 * max(0.0, 0.00871 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 5.25) / 0.00294533   # -0.3%  lam1 < 0.00871 and n_pt_above_50 > 5.25
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 10.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.75196 * (-0.05539475
        + 0.3621444 * Q.pt_7 / 34.64819   # +36.2%  pt_7
        - 0.145395 * max(0.0, Q.LHA - 0.131) / 0.1191367   # -14.5%  LHA > 0.131
        - 0.1041079 * max(0.0, Q.z_7 - 0.0366) / 0.01857003   # -10.4%  z_7 > 0.0366
        + 0.09825394 * max(0.0, 0.105 - Q.mass_over_sum_pt) / 0.04953952   # +9.8%  mass_over_sum_pt < 0.105
        + 0.09268914 * max(0.0, 788.6875 - Q.sum_pt) / 112.3137   # +9.3%  sum_pt < 788.7
        - 0.04266872 * max(0.0, 35.5 - Q.mass) / 9.790681   # -4.3%  mass < 35.5
        + 0.03459944 * max(0.0, 6.48 - Q.log_sum_pt) / 0.07581975   # +3.5%  log_sum_pt < 6.48
        - 0.02973225 * max(0.0, 0.502 - Q.planar_flow) / 0.2798902   # -3.0%  planar_flow < 0.502
        - 0.02390741 * max(0.0, Q.sum_pt - 538.0) * max(0.0, 0.000194 - Q.lam2) / 0.02734841   # -2.4%  sum_pt > 538 and lam2 < 0.000194
        - 0.02200259 * max(0.0, 59.4 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.0112) / 0.1880931   # -2.2%  sj3_pair_mass_max < 59.4 and centroid_offset > 0.0112
        + 0.01272498 * max(0.0, 0.000201 - Q.lam2) / 0.0001167487   # +1.3%  lam2 < 0.000201
        - 0.01247823 * max(0.0, 0.00609 - Q.lam1) * max(0.0, Q.max_dr - 0.0769) / 5.034665e-05   # -1.2%  lam1 < 0.00609 and max_dr > 0.0769
        + 0.004646281 * max(0.0, Q.sum_pt_top5 - 745.0) * max(0.0, 1.1 - Q.D2_b2) / 6.591847   # +0.5%  sum_pt_top5 > 745 and D2_b2 < 1.1
        + 0.00309402 * max(0.0, Q.m012 - 42.1) / 1.100813   # +0.3%  m012 > 42.1
        + 0.002993149 * max(0.0, 0.236 - Q.max_dr) / 0.1185157   # +0.3%  max_dr < 0.236
        + 0.002921267 * max(0.0, 0.00496 - Q.lam1) * max(0.0, Q.pt_6 - 17.3) / 0.0437705   # +0.3%  lam1 < 0.00496 and pt_6 > 17.3
        - 0.002725181 * max(0.0, 0.00765 - Q.sum_z_dr) / 0.0002291358   # -0.3%  sum_z_dr < 0.00765
        - 0.002336435 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, 1.31 - Q.D2_b2) / 0.002557546   # -0.2%  log_sum_pt > 6.85 and D2_b2 < 1.31
        + 0.0004558583 * max(0.0, Q.z_7 - 0.0548) * max(0.0, 0.0684 - Q.sj3_dr_min) / 0.0002460747   # +0.0%  z_7 > 0.0548 and sj3_dr_min < 0.0684
        - 0.0001238724 * max(0.0, Q.log_sum_pt - 6.95) * max(0.0, Q.pt_6 - 40.0) / 0.02760593   # -0.0%  log_sum_pt > 6.95 and pt_6 > 40
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 16.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.19049 * (-0.4678578
        + 0.2465982 * max(0.0, Q.sum_z_dr - 0.02670378) / 0.03627339   # +24.7%  sum_z_dr > 0.0267
        - 0.1622474 * max(0.0, Q.sum_z_dr2 - 0.007498352) / 0.002475592   # -16.2%  sum_z_dr2 > 0.007498
        - 0.1561356 * max(0.0, 0.0118 - Q.mass_over_sum_pt_sq) / 0.007324852   # -15.6%  mass_over_sum_pt_sq < 0.0118
        + 0.09885775 * max(0.0, 0.1023433 - Q.tau1) / 0.04797195   # +9.9%  tau1 < 0.1023
        + 0.07870063 * max(0.0, Q.sj3_dr_max - 0.1786811) / 0.04332973   # +7.9%  sj3_dr_max > 0.1787
        + 0.06252903 * max(0.0, Q.sj2_dr - 0.1488981) / 0.04470703   # +6.3%  sj2_dr > 0.1489
        + 0.04858733 * max(0.0, Q.lam1 - 0.00828) / 0.001860047   # +4.9%  lam1 > 0.00828
        + 0.04042139 * max(0.0, Q.centroid_offset - 0.00928) / 0.009783918   # +4.0%  centroid_offset > 0.00928
        + 0.03585076 * max(0.0, Q.sum_z_dr - 0.0307) * max(0.0, Q.log_sum_pt - 6.11) / 0.01050262   # +3.6%  sum_z_dr > 0.0307 and log_sum_pt > 6.11
        + 0.01539823 * max(0.0, Q.e2 - 0.046) / 0.004061648   # +1.5%  e2 > 0.046
        - 0.01063088 * max(0.0, 0.836 - Q.z_dr_0_0p05) / 0.3439984   # -1.1%  z_dr_0_0p05 < 0.836
        + 0.008523705 * max(0.0, Q.lam2 - 0.000776) * max(0.0, 64.0 - Q.pt_6) / 0.008920559   # +0.9%  lam2 > 0.000776 and pt_6 < 64
        - 0.008001822 * max(0.0, Q.mass - 65.1) / 3.179778   # -0.8%  mass > 65.1
        - 0.007398536 * max(0.0, Q.sj2_dr - 0.206) * max(0.0, Q.sj2_mass1 - -0.0277) / 0.376746   # -0.7%  sj2_dr > 0.206 and sj2_mass1 > -0.0277
        - 0.006963394 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 1.02 - Q.z_dr_0p05_0p1) / 0.01821916   # -0.7%  sj2_dr > 0.199 and z_dr_0p05_0p1 < 1.02
        + 0.006561598 * max(0.0, 1.04 - Q.n_dr_0_0p05) / 0.2999412   # +0.7%  n_dr_0_0p05 < 1.04
        + 0.004501614 * max(0.0, Q.sd_mass - 54.15028) / 5.452615   # +0.5%  sd_mass > 54.15
        - 0.002092092 * max(0.0, Q.sj2_dr - 0.214) * max(0.0, 0.0549 - Q.dr_3) / 0.0001149426   # -0.2%  sj2_dr > 0.214 and dr_3 < 0.0549
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 30.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.23538 * (-0.009372965
        + 0.1581628 * max(0.0, 0.0116 - Q.sum_zz_dr2) / 0.007154143   # +15.8%  sum_zz_dr2 < 0.0116
        - 0.1579348 * max(0.0, 0.00919 - Q.sum_z_dr2) / 0.00486042   # -15.8%  sum_z_dr2 < 0.00919
        + 0.09543547 * max(0.0, 0.224 - Q.N2) / 0.05151993   # +9.5%  N2 < 0.224
        - 0.09357832 * max(0.0, 0.237 - Q.sj3_dr_max) / 0.09308974   # -9.4%  sj3_dr_max < 0.237
        + 0.09247676 * max(0.0, 0.0651 - Q.C2) / 0.03965037   # +9.2%  C2 < 0.0651
        + 0.07585628 * max(0.0, 0.351 - Q.LHA) / 0.1165153   # +7.6%  LHA < 0.351
        - 0.05804585 * max(0.0, 0.224 - Q.N2) * max(0.0, Q.eccentricity - 0.727) / 0.013122   # -5.8%  N2 < 0.224 and eccentricity > 0.727
        - 0.05471165 * max(0.0, 0.000876 - Q.lam2) / 0.0006860996   # -5.5%  lam2 < 0.000876
        + 0.03913446 * max(0.0, Q.lam1_plus_lam2 - 0.00339) / 0.004163292   # +3.9%  lam1_plus_lam2 > 0.00339
        + 0.03242876 * max(0.0, 0.00779 - Q.sum_z_dr2_top2) / 0.004664019   # +3.2%  sum_z_dr2_top2 < 0.00779
        - 0.03128483 * max(0.0, Q.mass - 45.1) / 9.687039   # -3.1%  mass > 45.1
        - 0.02161486 * max(0.0, Q.mass_over_sum_pt - 0.0903) / 0.008321144   # -2.2%  mass_over_sum_pt > 0.0903
        - 0.01724962 * max(0.0, 0.242 - Q.N2) * max(0.0, 52.9 - Q.pt_7) / 1.045409   # -1.7%  N2 < 0.242 and pt_7 < 52.9
        - 0.01651987 * max(0.0, 0.000453 - Q.lam2) * max(0.0, 0.173 - Q.dr01) / 4.382473e-05   # -1.7%  lam2 < 0.000453 and dr01 < 0.173
        + 0.01627066 * max(0.0, Q.sd_mass - 42.1) * max(0.0, 26.6 - Q.sj3_pair_mass_min) / 153.2294   # +1.6%  sd_mass > 42.1 and sj3_pair_mass_min < 26.6
        - 0.01380286 * max(0.0, 753.0 - Q.sum_pt) / 90.27089   # -1.4%  sum_pt < 753
        - 0.01028217 * max(0.0, 0.219 - Q.N2) * max(0.0, 63.2 - Q.mass) / 0.3911696   # -1.0%  N2 < 0.219 and mass < 63.2
        - 0.007639567 * max(0.0, 0.0076 - Q.sum_z_dr2_top2) * max(0.0, Q.C2_b2 - 0.000612) / 6.024735e-06   # -0.8%  sum_z_dr2_top2 < 0.0076 and C2_b2 > 0.000612
        - 0.004431855 * max(0.0, Q.sd_mass - 37.0) * max(0.0, 0.276 - Q.sd_zg) / 0.4953446   # -0.4%  sd_mass > 37 and sd_zg < 0.276
        - 0.002098644 * max(0.0, 0.219 - Q.N2) * max(0.0, Q.sum_zz_dr2 - 0.0116) / 6.859356e-05   # -0.2%  N2 < 0.219 and sum_zz_dr2 > 0.0116
        - 0.001039925 * max(0.0, Q.max_dr - 0.164) * max(0.0, 0.000596 - Q.C2_b2) / 8.013949e-07   # -0.1%  max_dr > 0.164 and C2_b2 < 0.000596
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 6.689;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.689033 * (-0.01417267
        + 0.1728164 * max(0.0, 0.05222815 - Q.z_7) / 0.008796704   # +17.3%  z_7 < 0.05223
        + 0.1552528 * max(0.0, 0.0039 - Q.lam1_plus_lam2) / 0.001338226   # +15.5%  lam1_plus_lam2 < 0.0039
        - 0.1200314 * max(0.0, 35.4 - Q.pt_7) / 4.806336   # -12.0%  pt_7 < 35.4
        + 0.1039311 * max(0.0, 0.00167 - Q.sum_z_dr2) * max(0.0, 0.024 - Q.centroid_offset) / 7.294975e-06   # +10.4%  sum_z_dr2 < 0.00167 and centroid_offset < 0.024
        - 0.08375953 * max(0.0, 0.021 - Q.zdr_0) / 0.008902543   # -8.4%  zdr_0 < 0.021
        + 0.07785547 * max(0.0, 0.0708 - Q.z_7) * max(0.0, 45.4 - Q.mass_top3) / 0.6813633   # +7.8%  z_7 < 0.0708 and mass_top3 < 45.4
        + 0.06161663 * max(0.0, Q.sum_pt_top5 - 530.0) * max(0.0, 43.2 - Q.pt_6) / 1260.215   # +6.2%  sum_pt_top5 > 530 and pt_6 < 43.2
        - 0.05630917 * max(0.0, 0.215 - Q.LHA) * max(0.0, 6.78 - Q.log_sum_pt) / 0.003545687   # -5.6%  LHA < 0.215 and log_sum_pt < 6.78
        - 0.04907727 * max(0.0, Q.log_sum_pt - 6.83) / 0.009226693   # -4.9%  log_sum_pt > 6.83
        - 0.03890475 * max(0.0, 0.215 - Q.LHA) * max(0.0, 0.00121 - Q.lam1) / 3.093404e-05   # -3.9%  LHA < 0.215 and lam1 < 0.00121
        - 0.03258277 * max(0.0, 0.0687 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset) / 0.0003738793   # -3.3%  z_7 < 0.0687 and centroid_offset < 0.031
        + 0.01111576 * max(0.0, 0.00764 - Q.sum_z_dr) / 0.0002281396   # +1.1%  sum_z_dr < 0.00764
        + 0.01085758 * max(0.0, Q.log_sum_pt - 6.73) * max(0.0, 0.0147 - Q.dr_2) / 0.0001288244   # +1.1%  log_sum_pt > 6.73 and dr_2 < 0.0147
        + 0.01067344 * max(0.0, Q.sum_pt_top5 - 438.0) * max(0.0, Q.sj2_dr - 0.176) / 3.679269   # +1.1%  sum_pt_top5 > 438 and sj2_dr > 0.176
        + 0.009564067 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.48e-05 - Q.mean_eta2) / 9.148015e-07   # +1.0%  log_sum_pt > 6.7 and mean_eta2 < 8.48e-05
        - 0.00565176 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.00334) / 3.900631e-05   # -0.6%  log_sum_pt > 6.7 and mean_phi > 0.00334
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 18.87;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.87482 * (0.226679
        - 0.2870161 * max(0.0, 0.00867 - Q.sum_z_dr2) / 0.0044409   # -28.7%  sum_z_dr2 < 0.00867
        + 0.1416128 * max(0.0, 0.0118 - Q.lam1) / 0.007142795   # +14.2%  lam1 < 0.0118
        - 0.07942976 * max(0.0, 0.144 - Q.max_dr) / 0.04701576   # -7.9%  max_dr < 0.144
        - 0.07523673 * max(0.0, 64.5 - Q.mass) * max(0.0, 0.0534 - Q.z_dr_0p2_0p4) / 1.414764   # -7.5%  mass < 64.5 and z_dr_0p2_0p4 < 0.0534
        - 0.06882559 * max(0.0, 0.02343572 - Q.centroid_offset) / 0.0102726   # -6.9%  centroid_offset < 0.02344
        + 0.06363904 * max(0.0, 0.111 - Q.tau1) * max(0.0, 0.0157 - Q.mean_phi2) / 0.0008093032   # +6.4%  tau1 < 0.111 and mean_phi2 < 0.0157
        - 0.04953998 * max(0.0, 0.0118 - Q.lam1) * max(0.0, 0.254 - Q.planar_flow) / 0.0006395195   # -5.0%  lam1 < 0.0118 and planar_flow < 0.254
        + 0.04186406 * max(0.0, 42.5 - Q.pt_6) * max(0.0, 6.76 - Q.log_sum_pt) / 1.174637   # +4.2%  pt_6 < 42.5 and log_sum_pt < 6.76
        + 0.03221537 * max(0.0, 0.1416042 - Q.sj3_dr_max) / 0.03678777   # +3.2%  sj3_dr_max < 0.1416
        - 0.02365216 * max(0.0, 41.5 - Q.pt_6) * max(0.0, Q.z_7 - 0.0235) / 0.07325311   # -2.4%  pt_6 < 41.5 and z_7 > 0.0235
        - 0.02034089 * max(0.0, Q.sj3_dr_min - 0.0225) / 0.02864966   # -2.0%  sj3_dr_min > 0.0225
        - 0.01802281 * max(0.0, 0.00109 - Q.lam2) / 0.0008771499   # -1.8%  lam2 < 0.00109
        + 0.01666381 * max(0.0, 0.00301 - Q.sum_zz_dr2) / 0.001064474   # +1.7%  sum_zz_dr2 < 0.00301
        + 0.01566234 * max(0.0, 22.5 - Q.mass) / 4.680429   # +1.6%  mass < 22.5
        - 0.01061068 * max(0.0, Q.sj3_pair_mass_min - 4.35) / 3.542107   # -1.1%  sj3_pair_mass_min > 4.35
        - 0.01058011 * max(0.0, 0.0221 - Q.z_6) / 0.0003359079   # -1.1%  z_6 < 0.0221
        + 0.009981705 * max(0.0, 0.171 - Q.sj2_dr) * max(0.0, Q.eccentricity - 0.927) / 0.0007010376   # +1.0%  sj2_dr < 0.171 and eccentricity > 0.927
        + 0.009804586 * max(0.0, 19.1 - Q.pt_6) / 0.234072   # +1.0%  pt_6 < 19.1
        + 0.008084722 * max(0.0, Q.centroid_offset - 0.0103) * max(0.0, Q.psi_0p1 - 0.335) / 0.003358491   # +0.8%  centroid_offset > 0.0103 and psi_0p1 > 0.335
        + 0.004410435 * max(0.0, Q.sum_pt - 975.0) / 5.126702   # +0.4%  sum_pt > 975
        + 0.003652922 * max(0.0, 0.0032 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002474142   # +0.4%  lam2 < 0.0032 and n_dr_0_0p05 < 3
        + 0.002965826 * max(0.0, Q.centroid_offset - 0.0213) * max(0.0, 0.01 - Q.mean_phi2) / 2.050903e-05   # +0.3%  centroid_offset > 0.0213 and mean_phi2 < 0.01
        - 0.002190555 * max(0.0, Q.sum_pt - 953.0) * max(0.0, 0.00311 - Q.mean_phi2) / 0.01703845   # -0.2%  sum_pt > 953 and mean_phi2 < 0.00311
        + 0.001314139 * max(0.0, 589.0 - Q.sum_pt) * max(0.0, 1.97 - Q.n_dr_0p2_0p4) / 31.59369   # +0.1%  sum_pt < 589 and n_dr_0p2_0p4 < 1.97
        + 0.001120064 * max(0.0, 625.0 - Q.sum_pt) * max(0.0, 24.6 - Q.pt_5) / 1.93444   # +0.1%  sum_pt < 625 and pt_5 < 24.6
        + 0.001049458 * max(0.0, Q.centroid_offset - 0.017) * max(0.0, Q.pt_4 - 66.3) / 0.007834216   # +0.1%  centroid_offset > 0.017 and pt_4 > 66.3
        - 0.0005133441 * max(0.0, Q.mean_eta - 0.0349) * max(0.0, Q.M3 - 0.0492) / 9.673295e-06   # -0.1%  mean_eta > 0.0349 and M3 > 0.0492
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 19.53;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.52898 * (0.381955
        - 0.2421508 * max(0.0, Q.mass_over_sum_pt - 0.0909) / 0.00820305   # -24.2%  mass_over_sum_pt > 0.0909
        - 0.1249755 * max(0.0, 0.00664 - Q.lam1_plus_lam2) / 0.002908793   # -12.5%  lam1_plus_lam2 < 0.00664
        - 0.1148175 * max(0.0, 0.0884 - Q.sum_z_dr) / 0.03732151   # -11.5%  sum_z_dr < 0.0884
        - 0.06321261 * max(0.0, Q.lam1_plus_lam2 - 0.00864) * max(0.0, 40.8 - Q.pt_6) / 0.01018816   # -6.3%  lam1_plus_lam2 > 0.00864 and pt_6 < 40.8
        - 0.05192235 * max(0.0, 64.54392 - Q.mass) / 27.51298   # -5.2%  mass < 64.54
        - 0.05049682 * max(0.0, 0.02 - Q.centroid_offset) * max(0.0, 0.00426 - Q.C2_b2) / 2.734598e-05   # -5.0%  centroid_offset < 0.02 and C2_b2 < 0.00426
        + 0.04642704 * max(0.0, Q.sum_z_dr2 - 0.0131) * max(0.0, 41.3 - Q.pt_6) / 0.007197043   # +4.6%  sum_z_dr2 > 0.0131 and pt_6 < 41.3
        + 0.04061855 * max(0.0, Q.sj3_dr_max - 0.07254204) / 0.1085142   # +4.1%  sj3_dr_max > 0.07254
        + 0.03082436 * max(0.0, 0.09317241 - Q.sj2_dr) / 0.02197556   # +3.1%  sj2_dr < 0.09317
        + 0.02999003 * max(0.0, Q.mass_over_sum_pt - 0.0911) * max(0.0, 40.8 - Q.pt_6) / 0.03646238   # +3.0%  mass_over_sum_pt > 0.0911 and pt_6 < 40.8
        - 0.0240748 * max(0.0, 0.0407 - Q.tau21_b2) / 0.01129887   # -2.4%  tau21_b2 < 0.0407
        + 0.02198648 * max(0.0, 0.0003 - Q.lam2) * max(0.0, 0.0408 - Q.tau21_b2) / 2.603326e-06   # +2.2%  lam2 < 0.0003 and tau21_b2 < 0.0408
        - 0.02135423 * max(0.0, 0.000271 - Q.lam2) / 0.0001709275   # -2.1%  lam2 < 0.000271
        + 0.0157849 * max(0.0, 0.00109 - Q.sum_z_dr2_top2) / 0.0003134948   # +1.6%  sum_z_dr2_top2 < 0.00109
        + 0.01562709 * max(0.0, Q.sum_z_dr2 - 0.01855051) / 0.0007880277   # +1.6%  sum_z_dr2 > 0.01855
        + 0.01363104 * max(0.0, 0.06139287 - Q.max_dr) / 0.009201578   # +1.4%  max_dr < 0.06139
        + 0.01322335 * max(0.0, 0.0893 - Q.sum_z_dr) * max(0.0, 0.00429 - Q.C2_b2) / 0.0001363918   # +1.3%  sum_z_dr < 0.0893 and C2_b2 < 0.00429
        - 0.01214862 * max(0.0, Q.mass - 76.2) * max(0.0, Q.zdr_6 - 0.00612) / 0.007112219   # -1.2%  mass > 76.2 and zdr_6 > 0.00612
        + 0.01075531 * max(0.0, 0.0216 - Q.centroid_offset) * max(0.0, 0.0279 - Q.tau21_b2) / 5.569873e-05   # +1.1%  centroid_offset < 0.0216 and tau21_b2 < 0.0279
        - 0.01006494 * max(0.0, Q.z_7 - 0.0342) / 0.02040274   # -1.0%  z_7 > 0.0342
        - 0.009376064 * max(0.0, Q.centroid_offset - 0.0384) / 0.001626223   # -0.9%  centroid_offset > 0.0384
        - 0.006564127 * max(0.0, 29.4 - Q.pt_7) / 2.288755   # -0.7%  pt_7 < 29.4
        + 0.005560804 * max(0.0, Q.sum_z_dr2 - 0.00447) * max(0.0, 0.197 - Q.planar_flow) / 0.0002622145   # +0.6%  sum_z_dr2 > 0.00447 and planar_flow < 0.197
        - 0.004429117 * max(0.0, Q.sum_z_dr2 - 0.0145) * max(0.0, Q.eccentricity - 0.925) / 2.936783e-05   # -0.4%  sum_z_dr2 > 0.0145 and eccentricity > 0.925
        + 0.004038394 * max(0.0, Q.sd_rg - 0.283) / 0.004717574   # +0.4%  sd_rg > 0.283
        - 0.003818972 * max(0.0, 0.0586 - Q.D2_b2) / 0.006174766   # -0.4%  D2_b2 < 0.0586
        + 0.003486909 * max(0.0, Q.mass_top5 - 53.0) / 2.071825   # +0.3%  mass_top5 > 53
        - 0.003411408 * max(0.0, Q.centroid_offset - 0.0258) * max(0.0, Q.n_pt_above_50 - 3.71) / 0.003486117   # -0.3%  centroid_offset > 0.0258 and n_pt_above_50 > 3.71
        - 0.003164936 * max(0.0, 0.139 - Q.z_dr_0p1_0p2) / 0.08183782   # -0.3%  z_dr_0p1_0p2 < 0.139
        - 0.001369616 * max(0.0, 0.216 - Q.planar_flow) * max(0.0, 35.6 - Q.pt_6) / 0.2054075   # -0.1%  planar_flow < 0.216 and pt_6 < 35.6
        - 0.0005979795 * max(0.0, 0.00107 - Q.sum_z_dr2_top2) * max(0.0, 0.0257 - Q.tau21_b2) / 2.1676e-07   # -0.1%  sum_z_dr2_top2 < 0.00107 and tau21_b2 < 0.0257
        - 9.534884e-05 * max(0.0, Q.sj3_dr_max - 0.132) * max(0.0, 20.0 - Q.pt_6) / 0.01563421   # -0.0%  sj3_dr_max > 0.132 and pt_6 < 20
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 11.7;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.7025 * (-0.1161904
        - 0.3398266 * max(0.0, 0.0595 - Q.sum_z_dr) / 0.01782857   # -34.0%  sum_z_dr < 0.0595
        + 0.161855 * max(0.0, 0.00491 - Q.sum_z_dr2) / 0.00184393   # +16.2%  sum_z_dr2 < 0.00491
        + 0.1352692 * max(0.0, 0.00662 - Q.sum_z_dr2) * max(0.0, 0.0259 - Q.centroid_offset) / 4.433284e-05   # +13.5%  sum_z_dr2 < 0.00662 and centroid_offset < 0.0259
        + 0.1079648 * max(0.0, 0.0617 - Q.tau1) * max(0.0, 0.00304 - Q.lam1_plus_lam2) / 4.750042e-05   # +10.8%  tau1 < 0.0617 and lam1_plus_lam2 < 0.00304
        - 0.0776237 * max(0.0, 31.3 - Q.mass) * max(0.0, 0.0291 - Q.centroid_offset) / 0.1497916   # -7.8%  mass < 31.3 and centroid_offset < 0.0291
        + 0.07675094 * max(0.0, 0.00355 - Q.lam1_plus_lam2) / 0.001178624   # +7.7%  lam1_plus_lam2 < 0.00355
        + 0.03329464 * max(0.0, 0.000136 - Q.lam2) * max(0.0, 5.27 - Q.D2_b2) / 0.0002768574   # +3.3%  lam2 < 0.000136 and D2_b2 < 5.27
        - 0.02470096 * max(0.0, Q.log_sum_pt - 6.7) / 0.03560998   # -2.5%  log_sum_pt > 6.7
        - 0.01932159 * max(0.0, 0.142 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.0059) / 0.0002480234   # -1.9%  sj3_dr_max < 0.142 and centroid_offset > 0.0059
        + 0.01781302 * max(0.0, 0.0585 - Q.sum_z_dr) * max(0.0, Q.lam1_plus_lam2 - 0.000656) / 6.655317e-06   # +1.8%  sum_z_dr < 0.0585 and lam1_plus_lam2 > 0.000656
        - 0.00557963 * max(0.0, Q.sum_pt_top5 - 679.0) * max(0.0, 8.0 - Q.n_pt_above_10) / 7.96781   # -0.6%  sum_pt_top5 > 679 and n_pt_above_10 < 8
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 21.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.40785 * (-0.1614793
        + 0.2195075 * max(0.0, 0.00365 - Q.sum_z_dr2) / 0.001223483   # +22.0%  sum_z_dr2 < 0.00365
        + 0.1157894 * max(0.0, 986.0 - Q.sum_pt) / 274.4789   # +11.6%  sum_pt < 986
        - 0.1128593 * max(0.0, 0.00327 - Q.sum_zz_dr2) / 0.001182113   # -11.3%  sum_zz_dr2 < 0.00327
        + 0.09474867 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0267 - Q.centroid_offset) / 0.3086978   # +9.5%  mass < 54.9 and centroid_offset < 0.0267
        - 0.08366853 * max(0.0, 41.26805 - Q.mass) / 12.48331   # -8.4%  mass < 41.27
        - 0.05617252 * max(0.0, Q.lam1 - 0.00485) * max(0.0, Q.eccentricity - 0.638) / 0.0007816195   # -5.6%  lam1 > 0.00485 and eccentricity > 0.638
        + 0.05392491 * max(0.0, 0.0185 - Q.centroid_offset) / 0.006796883   # +5.4%  centroid_offset < 0.0185
        + 0.052606 * max(0.0, 0.000315 - Q.lam2) / 0.0002060487   # +5.3%  lam2 < 0.000315
        - 0.04390622 * max(0.0, 0.1059541 - Q.sj3_dr_max) / 0.02361947   # -4.4%  sj3_dr_max < 0.106
        + 0.02849357 * max(0.0, 55.0 - Q.mass) * max(0.0, 6.84 - Q.log_sum_pt) / 5.469627   # +2.8%  mass < 55 and log_sum_pt < 6.84
        - 0.02535139 * max(0.0, 0.0549 - Q.sum_z_dr) * max(0.0, Q.z_6 - 0.0338) / 0.0003175501   # -2.5%  sum_z_dr < 0.0549 and z_6 > 0.0338
        - 0.02220551 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1) / 0.2286861   # -2.2%  centroid_offset < 0.019 and pt_1 < 164
        + 0.01844738 * max(0.0, 0.107 - Q.max_dr) / 0.02607705   # +1.8%  max_dr < 0.107
        + 0.01429115 * max(0.0, 0.15 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 32.9) / 0.444104   # +1.4%  sj3_dr_max < 0.15 and pt_6 > 32.9
        + 0.01162945 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd) / 0.0002306924   # +1.2%  centroid_offset < 0.0191 and z_2nd < 0.205
        + 0.01001697 * max(0.0, 0.0796 - Q.M3) / 0.01268337   # +1.0%  M3 < 0.0796
        - 0.007017568 * max(0.0, 0.00578 - Q.sum_z_dr2) * max(0.0, 0.00437 - Q.mean_phi) / 1.493866e-05   # -0.7%  sum_z_dr2 < 0.00578 and mean_phi < 0.00437
        + 0.005825278 * max(0.0, Q.e2 - 0.0456) / 0.004134841   # +0.6%  e2 > 0.0456
        - 0.005277791 * max(0.0, 0.017 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.007515426   # -0.5%  centroid_offset < 0.017 and n_for_90pct > 5
        - 0.004183587 * max(0.0, 0.00631 - Q.zdr_0) / 0.001011633   # -0.4%  zdr_0 < 0.00631
        + 0.003558439 * max(0.0, 0.000176 - Q.lam1_plus_lam2) / 1.498933e-05   # +0.4%  lam1_plus_lam2 < 0.000176
        - 0.003073468 * max(0.0, 0.0457 - Q.tau1) * max(0.0, 0.0286 - Q.tau21_b2) / 1.104522e-05   # -0.3%  tau1 < 0.0457 and tau21_b2 < 0.0286
        + 0.002340061 * max(0.0, 0.0528 - Q.sum_z_dr) * max(0.0, 0.0275 - Q.tau21_b2) / 1.132135e-05   # +0.2%  sum_z_dr < 0.0528 and tau21_b2 < 0.0275
        - 0.002177578 * max(0.0, Q.D2 - 2.16) / 0.2863121   # -0.2%  D2 > 2.16
        + 0.001970737 * max(0.0, 0.046 - Q.tau1) * max(0.0, -0.0119 - Q.mean_phi) / 1.388482e-05   # +0.2%  tau1 < 0.046 and mean_phi < -0.0119
        - 0.0009570541 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.746 - Q.D2_b2) / 0.0007134755   # -0.1%  log_sum_pt > 6.89 and D2_b2 < 0.746
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 6.938;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.93833 * (0.2504375
        - 0.2291519 * max(0.0, 0.0041726 - Q.lam1) / 0.001507506   # -22.9%  lam1 < 0.004173
        + 0.1785356 * max(0.0, 79.4 - Q.mass) / 40.3764   # +17.9%  mass < 79.4
        + 0.1011021 * max(0.0, Q.tau1 - 0.057) / 0.0250643   # +10.1%  tau1 > 0.057
        - 0.08589156 * max(0.0, Q.LHA - 0.305) / 0.01744179   # -8.6%  LHA > 0.305
        + 0.07836774 * max(0.0, Q.lam2 - 0.000278) / 0.0004273421   # +7.8%  lam2 > 0.000278
        + 0.06039625 * max(0.0, Q.sum_z_dr2 - 0.006688641) / 0.002699226   # +6.0%  sum_z_dr2 > 0.006689
        + 0.04729711 * max(0.0, 0.00215 - Q.sum_z_dr2_top3) / 0.0007781806   # +4.7%  sum_z_dr2_top3 < 0.00215
        + 0.02641695 * max(0.0, 0.0242 - Q.zdr_0) * max(0.0, 0.248 - Q.z_dr_0p05_0p1) / 0.002102794   # +2.6%  zdr_0 < 0.0242 and z_dr_0p05_0p1 < 0.248
        - 0.025247 * max(0.0, 48.0 - Q.pt_7) * max(0.0, 6.56 - Q.log_sum_pt) / 1.427027   # -2.5%  pt_7 < 48 and log_sum_pt < 6.56
        - 0.02510618 * max(0.0, Q.lam2 - 0.000257) * max(0.0, Q.planar_flow - 0.055) / 0.0002791476   # -2.5%  lam2 > 0.000257 and planar_flow > 0.055
        + 0.02250591 * max(0.0, 45.6 - Q.pt_7) * max(0.0, 0.985 - Q.D2) / 1.72129   # +2.3%  pt_7 < 45.6 and D2 < 0.985
        - 0.0220773 * max(0.0, Q.sj3_dr_max - 0.184) / 0.04099712   # -2.2%  sj3_dr_max > 0.184
        + 0.02162264 * max(0.0, Q.sj3_pair_mass_min - 15.8) / 1.364522   # +2.2%  sj3_pair_mass_min > 15.8
        - 0.01925278 * max(0.0, 43.9 - Q.pt_7) / 10.61046   # -1.9%  pt_7 < 43.9
        - 0.01109672 * max(0.0, Q.sj3_pair_mass_min - 6.68) * max(0.0, Q.sj3_pairmin_over_m - 0.245) / 0.3957389   # -1.1%  sj3_pair_mass_min > 6.68 and sj3_pairmin_over_m > 0.245
        - 0.01030039 * max(0.0, Q.e3 - 5.22e-05) / 5.502755e-05   # -1.0%  e3 > 5.22e-05
        + 0.008614114 * max(0.0, 0.0205 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0125) / 3.153765e-05   # +0.9%  zdr_0 < 0.0205 and centroid_offset > 0.0125
        - 0.007963642 * max(0.0, Q.lam1 - 0.00707) * max(0.0, 0.36 - Q.D2_b2) / 0.0002831702   # -0.8%  lam1 > 0.00707 and D2_b2 < 0.36
        - 0.007544706 * max(0.0, Q.sum_pt - 988.0) / 4.42251   # -0.8%  sum_pt > 988
        + 0.006502866 * max(0.0, Q.C2_b2 - 0.00763) / 0.00184727   # +0.7%  C2_b2 > 0.00763
        - 0.0050065 * max(0.0, Q.mass_top5 - 51.0) / 2.405466   # -0.5%  mass_top5 > 51
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 27.39;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.39082 * (-0.03612344
        + 0.405718 * max(0.0, 0.00872 - Q.lam1_plus_lam2) / 0.004480998   # +40.6%  lam1_plus_lam2 < 0.00872
        - 0.2079204 * max(0.0, 0.00814 - Q.sum_zz_dr2) / 0.004271861   # -20.8%  sum_zz_dr2 < 0.00814
        - 0.1192215 * max(0.0, 0.00838 - Q.lam1) / 0.004303695   # -11.9%  lam1 < 0.00838
        - 0.0556201 * max(0.0, 0.06656016 - Q.sum_z_dr) / 0.02181718   # -5.6%  sum_z_dr < 0.06656
        + 0.04426587 * max(0.0, 0.0134 - Q.sum_z_dr2) / 0.008373912   # +4.4%  sum_z_dr2 < 0.0134
        + 0.02868219 * max(0.0, Q.mass - 21.67283) / 23.0405   # +2.9%  mass > 21.67
        + 0.02163199 * max(0.0, 0.242 - Q.planar_flow) / 0.1026884   # +2.2%  planar_flow < 0.242
        + 0.01669477 * max(0.0, Q.z_7 - 0.0169) / 0.03553139   # +1.7%  z_7 > 0.0169
        - 0.0153403 * max(0.0, Q.centroid_offset - 0.05) * max(0.0, Q.tau3 - 0.000494) / 7.361023e-06   # -1.5%  centroid_offset > 0.05 and tau3 > 0.000494
        - 0.01391493 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409   # -1.4%  centroid_offset > 0.05
        + 0.00960054 * max(0.0, 0.0138 - Q.centroid_offset) * max(0.0, 0.571 - Q.D2_b2) / 0.0007593655   # +1.0%  centroid_offset < 0.0138 and D2_b2 < 0.571
        - 0.00944518 * max(0.0, 0.241 - Q.planar_flow) * max(0.0, 835.0 - Q.sum_pt) / 13.13928   # -0.9%  planar_flow < 0.241 and sum_pt < 835
        - 0.008954225 * max(0.0, 0.0137 - Q.centroid_offset) * max(0.0, 17.2 - Q.sj3_pair_mass_min) / 0.05151779   # -0.9%  centroid_offset < 0.0137 and sj3_pair_mass_min < 17.2
        + 0.008197006 * max(0.0, Q.centroid_offset - 0.0494) * max(0.0, 0.619 - Q.tau32) / 0.0001481531   # +0.8%  centroid_offset > 0.0494 and tau32 < 0.619
        - 0.00757536 * max(0.0, 0.1059541 - Q.sj3_dr_max) / 0.02361947   # -0.8%  sj3_dr_max < 0.106
        - 0.004574497 * max(0.0, 0.268 - Q.planar_flow) * max(0.0, 36.7 - Q.pt_7) / 0.569477   # -0.5%  planar_flow < 0.268 and pt_7 < 36.7
        - 0.004294391 * max(0.0, Q.n_dr_0p1_0p2 - 2.92) / 0.3491542   # -0.4%  n_dr_0p1_0p2 > 2.92
        - 0.003372072 * max(0.0, Q.centroid_offset - 0.0492) * max(0.0, Q.zdr_6 - 0.00714) / 3.382632e-06   # -0.3%  centroid_offset > 0.0492 and zdr_6 > 0.00714
        - 0.003192388 * max(0.0, Q.pt_7 - 29.6) / 7.399598   # -0.3%  pt_7 > 29.6
        - 0.002048845 * max(0.0, 0.233 - Q.planar_flow) * max(0.0, 9.53e-06 - Q.e3) / 1.282146e-07   # -0.2%  planar_flow < 0.233 and e3 < 9.53e-06
        + 0.001839341 * max(0.0, 0.264 - Q.sj3_dr_max) * max(0.0, -0.023 - Q.phi_0) / 0.0003709874   # +0.2%  sj3_dr_max < 0.264 and phi_0 < -0.023
        - 0.001615421 * max(0.0, 0.0383 - Q.centroid_offset) * max(0.0, -0.00159 - Q.mean_phi) / 5.204323e-05   # -0.2%  centroid_offset < 0.0383 and mean_phi < -0.00159
        - 0.001485557 * max(0.0, 25.4 - Q.pt_6) / 0.7070937   # -0.1%  pt_6 < 25.4
        + 0.001281211 * max(0.0, 0.00431 - Q.sum_z_dr2) * max(0.0, Q.sj3_dr13 - 0.162) / 5.926841e-06   # +0.1%  sum_z_dr2 < 0.00431 and sj3_dr13 > 0.162
        + 0.001228118 * max(0.0, 0.0377 - Q.centroid_offset) * max(0.0, 892.0 - Q.sum_pt) / 3.336207   # +0.1%  centroid_offset < 0.0377 and sum_pt < 892
        + 0.001031566 * max(0.0, Q.z_dr_0p05_0p1 - 0.826) / 0.01291128   # +0.1%  z_dr_0p05_0p1 > 0.826
        - 0.0007059367 * max(0.0, Q.sum_pt - 971.0) / 5.370666   # -0.1%  sum_pt > 971
        + 0.0005482481 * max(0.0, 0.0384 - Q.centroid_offset) * max(0.0, 0.0324 - Q.absphi_0) / 0.0003541431   # +0.1%  centroid_offset < 0.0384 and absphi_0 < 0.0324
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4059;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4059215 * (-1.787301
        + 0.4096427 * max(0.0, Q.sum_z_dr2 - 0.0188) / 0.0007632519   # +41.0%  sum_z_dr2 > 0.0188
        - 0.1348984 * max(0.0, Q.e2 - 0.0634) / 0.001765919   # -13.5%  e2 > 0.0634
        + 0.09541634 * max(0.0, Q.mass - 91.2) / 0.5835014   # +9.5%  mass > 91.2
        + 0.08470634 * max(0.0, Q.sum_z_dr2 - 0.0188) * max(0.0, Q.lam2 - 0.000537) / 2.837257e-06   # +8.5%  sum_z_dr2 > 0.0188 and lam2 > 0.000537
        - 0.06734251 * max(0.0, Q.sum_z_dr2_top2 - 0.014) / 0.001132836   # -6.7%  sum_z_dr2_top2 > 0.014
        + 0.06687423 * max(0.0, Q.mass - 69.6) * max(0.0, Q.centroid_offset - 0.00948) / 0.04084341   # +6.7%  mass > 69.6 and centroid_offset > 0.00948
        - 0.04931117 * max(0.0, Q.zdr_0 - 0.0398) / 0.000570514   # -4.9%  zdr_0 > 0.0398
        - 0.04593852 * max(0.0, Q.sum_z_dr2 - 0.0188) * max(0.0, 53.4 - Q.pt_7) / 0.01441927   # -4.6%  sum_z_dr2 > 0.0188 and pt_7 < 53.4
        + 0.0458698 * max(0.0, Q.centroid_offset - 0.0499) / 0.0008065856   # +4.6%  centroid_offset > 0.0499
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.93;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.92987 * (0.01646682
        + 0.3138789 * max(0.0, 0.148 - Q.sum_z_dr) / 0.0902118   # +31.4%  sum_z_dr < 0.148
        + 0.1548265 * max(0.0, 0.0161 - Q.lam1) / 0.01090097   # +15.5%  lam1 < 0.0161
        - 0.1458958 * max(0.0, 0.079 - Q.e2) / 0.05097915   # -14.6%  e2 < 0.079
        - 0.08589327 * max(0.0, 0.148 - Q.sum_z_dr) * max(0.0, 6.81 - Q.log_sum_pt) / 0.02030622   # -8.6%  sum_z_dr < 0.148 and log_sum_pt < 6.81
        - 0.08125 * max(0.0, 5.38e-05 - Q.e3) * max(0.0, 0.0377 - Q.centroid_offset) / 8.394154e-07   # -8.1%  e3 < 5.38e-05 and centroid_offset < 0.0377
        + 0.07943688 * max(0.0, 5.05e-05 - Q.e3) / 3.076371e-05   # +7.9%  e3 < 5.05e-05
        - 0.04570159 * max(0.0, 0.145 - Q.sum_z_dr) * max(0.0, 38.2 - Q.pt_7) / 0.6303457   # -4.6%  sum_z_dr < 0.145 and pt_7 < 38.2
        + 0.03700129 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, 38.1 - Q.pt_7) / 666.7941   # +3.7%  sum_pt_top5 > 667 and pt_7 < 38.1
        - 0.01212504 * max(0.0, 31.9 - Q.pt_6) * max(0.0, 0.0624 - Q.z_7) / 0.0743105   # -1.2%  pt_6 < 31.9 and z_7 < 0.0624
        + 0.008959736 * max(0.0, 0.153 - Q.sum_z_dr) * max(0.0, Q.z_7 - 0.0616) / 0.0003674466   # +0.9%  sum_z_dr < 0.153 and z_7 > 0.0616
        - 0.008299748 * max(0.0, Q.mass - 49.1) / 7.951063   # -0.8%  mass > 49.1
        - 0.007543012 * max(0.0, 0.028 - Q.z_7) / 0.001299403   # -0.8%  z_7 < 0.028
        + 0.004197542 * max(0.0, 0.142 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.00886) / 0.0002342723   # +0.4%  sum_z_dr < 0.142 and tau2 > 0.00886
        - 0.003211284 * max(0.0, Q.sum_pt - 1000.0) * max(0.0, 70.0 - Q.pt_6) / 114.1924   # -0.3%  sum_pt > 1000 and pt_6 < 70
        + 0.002578121 * max(0.0, 0.15 - Q.sum_z_dr) * max(0.0, 0.0746 - Q.M3) / 0.0006858451   # +0.3%  sum_z_dr < 0.15 and M3 < 0.0746
        - 0.002153485 * max(0.0, 0.0283 - Q.z_5) / 0.0003862796   # -0.2%  z_5 < 0.0283
        + 0.001808529 * max(0.0, Q.sum_pt_top5 - 902.75) / 4.105455   # +0.2%  sum_pt_top5 > 902.8
        - 0.001472069 * max(0.0, 0.152 - Q.sum_z_dr) * max(0.0, Q.sj2_mass1 - 30.4) / 0.01537341   # -0.1%  sum_z_dr < 0.152 and sj2_mass1 > 30.4
        - 0.001404194 * max(0.0, Q.pt_7 - 48.4) / 0.7087472   # -0.1%  pt_7 > 48.4
        + 0.001341309 * max(0.0, 6.94e-05 - Q.e3) * max(0.0, Q.D3 - 0.0598) / 6.179642e-05   # +0.1%  e3 < 6.94e-05 and D3 > 0.0598
        + 0.001021621 * max(0.0, 0.148 - Q.sum_z_dr) * max(0.0, 0.000536 - Q.lam2) / 3.994329e-05   # +0.1%  sum_z_dr < 0.148 and lam2 < 0.000536
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 27.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.63121 * (0.1075493
        - 0.214174 * max(0.0, 0.1003996 - Q.sum_z_dr) / 0.04732453   # -21.4%  sum_z_dr < 0.1004
        - 0.153162 * max(0.0, Q.lam1_plus_lam2 - 0.008566101) / 0.00223691   # -15.3%  lam1_plus_lam2 > 0.008566
        - 0.1059531 * max(0.0, Q.sum_z_dr2 - 0.00856) / 0.002238139   # -10.6%  sum_z_dr2 > 0.00856
        + 0.09242887 * max(0.0, Q.sum_zz_dr2 - 0.0117) / 0.001427015   # +9.2%  sum_zz_dr2 > 0.0117
        + 0.09197907 * max(0.0, Q.mass_over_sum_pt - 0.08453854) / 0.009615355   # +9.2%  mass_over_sum_pt > 0.08454
        + 0.07261294 * max(0.0, 0.04107712 - Q.e2) / 0.01756014   # +7.3%  e2 < 0.04108
        + 0.03911389 * max(0.0, Q.sj2_dr - 0.159004) / 0.03934505   # +3.9%  sj2_dr > 0.159
        - 0.03309769 * max(0.0, Q.centroid_offset - 0.0497) / 0.0008167572   # -3.3%  centroid_offset > 0.0497
        + 0.02739435 * max(0.0, 0.000504 - Q.lam2) / 0.0003626945   # +2.7%  lam2 < 0.000504
        + 0.02688829 * max(0.0, 44.81215 - Q.sd_mass) / 19.43433   # +2.7%  sd_mass < 44.81
        - 0.02580085 * max(0.0, 0.1441415 - Q.sd_rg) / 0.06067958   # -2.6%  sd_rg < 0.1441
        - 0.02008639 * max(0.0, 7.93e-05 - Q.e3) / 5.449318e-05   # -2.0%  e3 < 7.93e-05
        + 0.01938211 * max(0.0, Q.z_dr_0_0p05 - 0.152) / 0.447767   # +1.9%  z_dr_0_0p05 > 0.152
        - 0.01646906 * max(0.0, 717.0 - Q.sum_pt) / 70.68471   # -1.6%  sum_pt < 717
        - 0.0137921 * max(0.0, 0.00403 - Q.C2_b2) / 0.002881553   # -1.4%  C2_b2 < 0.00403
        - 0.01040329 * max(0.0, 0.163 - Q.z_dr_0p05_0p1) / 0.08628701   # -1.0%  z_dr_0p05_0p1 < 0.163
        - 0.006977121 * max(0.0, Q.sj2_dr - 0.203) * max(0.0, 0.105 - Q.z_5) / 0.0007140958   # -0.7%  sj2_dr > 0.203 and z_5 < 0.105
        + 0.0063983 * max(0.0, 5.06 - Q.n_dr_0_0p05) / 1.975727   # +0.6%  n_dr_0_0p05 < 5.06
        + 0.006258673 * max(0.0, Q.centroid_offset - 0.0499) * max(0.0, 0.000872 - Q.C2_b2) / 1.452791e-07   # +0.6%  centroid_offset > 0.0499 and C2_b2 < 0.000872
        + 0.003747601 * max(0.0, 5.56e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.162) / 4.392405e-07   # +0.4%  e3 < 5.56e-05 and sj3_dr23 > 0.162
        - 0.003512634 * max(0.0, 0.0677 - Q.z_dr_0p1_0p2) / 0.03667286   # -0.4%  z_dr_0p1_0p2 < 0.0677
        - 0.003100957 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.00746 - Q.lam1) / 6.069551e-05   # -0.3%  planar_flow < 0.11 and lam1 < 0.00746
        - 0.00146379 * max(0.0, Q.psi_0p1 - 0.974) / 0.01146305   # -0.1%  psi_0p1 > 0.974
        - 0.001362497 * max(0.0, Q.sj2_dr - 0.164) * max(0.0, 0.0332 - Q.dr_2) / 6.09129e-05   # -0.1%  sj2_dr > 0.164 and dr_2 < 0.0332
        + 0.0009996846 * max(0.0, Q.mass - 68.9) / 2.515464   # +0.1%  mass > 68.9
        - 0.0007297328 * max(0.0, Q.sj2_dr - 0.198) * max(0.0, 0.0872 - Q.D2_b2) / 0.0003408602   # -0.1%  sj2_dr > 0.198 and D2_b2 < 0.0872
        + 0.0006604769 * max(0.0, 0.627 - Q.z_dr_0p05_0p1) * max(0.0, 714.0 - Q.sum_pt) / 23.78173   # +0.1%  z_dr_0p05_0p1 < 0.627 and sum_pt < 714
        - 0.0005882866 * max(0.0, Q.z_dr_0p05_0p1 - 0.745) * max(0.0, Q.zdr_5 - 0.0154) / 1.958188e-06   # -0.1%  z_dr_0p05_0p1 > 0.745 and zdr_5 > 0.0154
        + 0.0005246426 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 652.0 - Q.sum_pt_top5) / 2.354108   # +0.1%  planar_flow < 0.1 and sum_pt_top5 < 652
        - 0.0003898064 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.00684) / 0.0003106102   # -0.0%  planar_flow < 0.109 and mean_eta > -0.00684
        + 0.0002853939 * max(0.0, 0.112 - Q.planar_flow) * max(0.0, 0.00601 - Q.lam1_plus_lam2) / 3.234239e-05   # +0.0%  planar_flow < 0.112 and lam1_plus_lam2 < 0.00601
        - 0.0002088754 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 0.0179 - Q.centroid_offset) / 0.000195363   # -0.0%  planar_flow < 0.106 and centroid_offset < 0.0179
        + 5.361913e-05 * max(0.0, Q.centroid_offset - 0.0267) * max(0.0, 152.0 - Q.pt_1) / 0.1895528   # +0.0%  centroid_offset > 0.0267 and pt_1 < 152
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 25.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.67041 * (-0.02027713
        + 0.2504823 * max(0.0, 0.0133 - Q.lam1_plus_lam2) / 0.008288473   # +25.0%  lam1_plus_lam2 < 0.0133
        - 0.1939712 * max(0.0, 0.00818 - Q.sum_zz_dr2) / 0.004303912   # -19.4%  sum_zz_dr2 < 0.00818
        - 0.1636339 * max(0.0, 0.102 - Q.sum_z_dr) / 0.04868705   # -16.4%  sum_z_dr < 0.102
        + 0.1205045 * max(0.0, 0.0409 - Q.e2) / 0.01742797   # +12.1%  e2 < 0.0409
        - 0.09792853 * max(0.0, 0.00739 - Q.sum_z_dr2) / 0.003447965   # -9.8%  sum_z_dr2 < 0.00739
        - 0.07733132 * max(0.0, 0.000513 - Q.sum_z_dr2_top2) / 0.0001106155   # -7.7%  sum_z_dr2_top2 < 0.000513
        + 0.06507366 * max(0.0, 0.00708 - Q.mass_over_sum_pt_sq) / 0.003452009   # +6.5%  mass_over_sum_pt_sq < 0.00708
        + 0.0104502 * max(0.0, Q.sj2_dr - 0.159004) / 0.03934505   # +1.0%  sj2_dr > 0.159
        - 0.00729323 * max(0.0, 0.00752 - Q.sum_z_dr2) * max(0.0, 0.731 - Q.D2) / 8.59706e-05   # -0.7%  sum_z_dr2 < 0.00752 and D2 < 0.731
        + 0.004314664 * max(0.0, 0.133 - Q.mass_over_sum_pt) * max(0.0, 0.711 - Q.D2) / 0.003689413   # +0.4%  mass_over_sum_pt < 0.133 and D2 < 0.711
        + 0.00389056 * max(0.0, 0.218 - Q.N2) * max(0.0, Q.sum_pt_top5 - 498.0) / 5.782364   # +0.4%  N2 < 0.218 and sum_pt_top5 > 498
        - 0.003162623 * max(0.0, 0.223 - Q.N2) * max(0.0, 0.548 - Q.z_dr_0p05_0p1) / 0.01037238   # -0.3%  N2 < 0.223 and z_dr_0p05_0p1 < 0.548
        + 0.001846074 * max(0.0, 0.229 - Q.N2) * max(0.0, 4.0 - Q.n_dr_0p1_0p2) / 0.116081   # +0.2%  N2 < 0.229 and n_dr_0p1_0p2 < 4
        - 0.0001172414 * max(0.0, Q.sj3_pair_mass_max - 81.1) / 0.3077858   # -0.0%  sj3_pair_mass_max > 81.1
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.043758193277311, 0.8432978991596639, 1.9236297268907563, 1.1357203781512606, 1.1050819327731092, 1.7287785714285715, 0.9060806722689075, 1.8202176470588236, 0.25394579831932773, 3.1349344537815127, 2.480342962184874, 2.01990262605042, 0.10268067226890756, 3.9102720588235296, 0.4134828781512605, 0.4587932773109244]
T = [2.3156593331473214, 1.448466524422269, 3.4537181246717434, 2.589161149225315, 3.2752529378939075]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +36%, n9 +23%, n1 +14%, n5 -14%, n0 -7%, n6 +4% ...
            + 0.3569435 * h[2] / H_AVG[2]
            + 0.232684 * h[9] / H_AVG[9]
            + 0.1422546 * h[1] / H_AVG[1]
            - 0.13998 * h[5] / H_AVG[5]
            - 0.07042798 * h[0] / H_AVG[0]
            + 0.0427967 * h[6] / H_AVG[6]
            - 0.01491317 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +55%, n10 -21%, n6 +8%, n4 -7%, n5 +6%, n15 +2% ...
            + 0.5495325 * h[9] / H_AVG[9]
            - 0.214049 * h[10] / H_AVG[10]
            + 0.0781931 * h[6] / H_AVG[6]
            - 0.07152491 * h[4] / H_AVG[4]
            + 0.05594641 * h[5] / H_AVG[5]
            + 0.01979651 * h[15] / H_AVG[15]
            + 0.01095753 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +22%, n3 -16%, n7 +12%, n0 +10%, n15 -9%, n14 -9% ...
            + 0.2193183 * h[11] / H_AVG[11]
            - 0.16442 * h[3] / H_AVG[3]
            + 0.1152881 * h[7] / H_AVG[7]
            + 0.1038857 * h[0] / H_AVG[0]
            - 0.09132777 * h[15] / H_AVG[15]
            - 0.08979081 * h[14] / H_AVG[14]
            - 0.08198417 * h[6] / H_AVG[6]
            + 0.07960725 * h[13] / H_AVG[13]
            - 0.02836558 * h[9] / H_AVG[9]
            - 0.01838206 * h[8] / H_AVG[8]
            - 0.007630345 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +33%, n3 -25%, n6 -13%, n13 +8%, n14 +6%, n1 +4% ...
            + 0.329538 * h[7] / H_AVG[7]
            - 0.2467373 * h[3] / H_AVG[3]
            - 0.1312318 * h[6] / H_AVG[6]
            + 0.08259162 * h[13] / H_AVG[13]
            + 0.05988661 * h[14] / H_AVG[14]
            + 0.04071289 * h[1] / H_AVG[1]
            - 0.03783724 * h[9] / H_AVG[9]
            + 0.03334459 * h[4] / H_AVG[4]
            - 0.02768713 * h[15] / H_AVG[15]
            + 0.01043279 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -49%, n10 +28%, n5 -13%, n4 +4%, n3 +2%, n12 -2% ...
            - 0.4850154 * h[13] / H_AVG[13]
            + 0.2839868 * h[10] / H_AVG[10]
            - 0.1319576 * h[5] / H_AVG[5]
            + 0.04217544 * h[4] / H_AVG[4]
            + 0.02167238 * h[3] / H_AVG[3]
            - 0.01567523 * h[12] / H_AVG[12]
            + 0.01453776 * h[8] / H_AVG[8]
            + 0.004979378 * h[0] / H_AVG[0]
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
