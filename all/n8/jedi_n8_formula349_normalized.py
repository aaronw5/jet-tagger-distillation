"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.1%   (on for 95% of jets)
  neuron  9:  12.1%   (on for 66% of jets)
  neuron  7:  10.0%   (on for 62% of jets)
  neuron 10:   8.2%   (on for 74% of jets)
  neuron  5:   8.1%   (on for 80% of jets)
  neuron  3:   8.0%   (on for 21% of jets)
  neuron  6:   7.7%   (on for 34% of jets)
  neuron  2:   7.2%   (on for 92% of jets)
  neuron 11:   5.9%   (on for 77% of jets)
  neuron 14:   3.9%   (on for 25% of jets)
  neuron  0:   3.9%   (on for 40% of jets)
  neuron  1:   3.5%   (on for 59% of jets)
  neuron  4:   3.4%   (on for 60% of jets)
  neuron 15:   2.5%   (on for 27% of jets)
  neuron  8:   1.1%   (on for 24% of jets)
  neuron 12:   0.3%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.8% (the network: 65.8%); same class as the network for 87.8% of jets.

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
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top3=mass_of(3),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 25.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.01295 * (-0.1018419
        - 0.1314532 * max(0.0, 0.004372139 - Q.width) / 0.00156571   # -13.1%  width < 0.004372
        + 0.1057582 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +10.6%  width < 0.008678
        + 0.09566012 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +9.6%  girth2 < 0.01324
        - 0.06887299 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -6.9%  girth < 0.08724
        + 0.06817795 * max(0.0, 0.0245477 - Q.e2) / 0.007455821   # +6.8%  e2 < 0.02455
        + 0.05908527 * max(0.0, 0.3012016 - Q.sj3_dr_max) / 0.1451317   # +5.9%  sj3_dr_max < 0.3012
        - 0.05828057 * max(0.0, 0.001563465 - Q.C2_b2) / 0.0008800039   # -5.8%  C2_b2 < 0.001563
        + 0.04582755 * max(0.0, 21.78408 - Q.mass) / 4.431023   # +4.6%  mass < 21.78
        - 0.04454191 * max(0.0, 29.6447 - Q.mass) / 7.34723   # -4.5%  mass < 29.64
        + 0.03881878 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +3.9%  lam1 < 0.006507
        - 0.03716249 * max(0.0, 0.005433361 - Q.lam1) / 0.00219414   # -3.7%  lam1 < 0.005433
        - 0.03668024 * max(0.0, 56.92035 - Q.mass) / 21.78475   # -3.7%  mass < 56.92
        + 0.0349798 * max(0.0, 0.1484197 - Q.planar_flow) / 0.05110362   # +3.5%  planar_flow < 0.1484
        - 0.02793159 * max(0.0, Q.sj3_dr_max - 0.233678) / 0.02511524   # -2.8%  sj3_dr_max > 0.2337
        - 0.02636734 * max(0.0, Q.log_sum_pt - 6.670067) / 0.04529298   # -2.6%  log_sum_pt > 6.67
        + 0.02517561 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +2.5%  sum_pt_top5 > 687.4
        + 0.02322795 * max(0.0, Q.n_dr_0_0p05 - 4.0) / 1.61275   # +2.3%  n_dr_0_0p05 > 4
        + 0.01753322 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, 1.002471 - Q.D2) / 0.001008123   # +1.8%  girth2 < 0.01324 and D2 < 1.002
        + 0.0148426 * max(0.0, Q.sj3_dr_max - 0.1070199) / 0.08518534   # +1.5%  sj3_dr_max > 0.107
        - 0.0114928 * max(0.0, Q.sum_pt - 901.5938) / 12.40362   # -1.1%  sum_pt > 901.6
        - 0.01019346 * max(0.0, 0.01323868 - Q.girth2) * max(0.0, Q.centroid_offset - 0.01837778) / 2.056482e-05   # -1.0%  girth2 < 0.01324 and centroid_offset > 0.01838
        - 0.009995756 * max(0.0, 0.006506576 - Q.lam1) * max(0.0, 0.875672 - Q.D2) / 7.962191e-05   # -1.0%  lam1 < 0.006507 and D2 < 0.8757
        + 0.007940593 * max(0.0, 7.300726e-05 - Q.lam2) * max(0.0, 0.2669656 - Q.D2_b2) / 2.36013e-06   # +0.8%  lam2 < 7.301e-05 and D2_b2 < 0.267
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 17.2;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.1981 * (-0.007156695
        - 0.1512374 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -15.1%  girth2 < 0.008678
        + 0.1338797 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) / 0.002578466   # +13.4%  mass_over_sum_pt_sq < 0.005833
        - 0.09324601 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -9.3%  girth < 0.0717
        - 0.08514141 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -8.5%  width < 0.006097
        + 0.07610354 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +7.6%  lam1 < 0.005954
        + 0.06919845 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # +6.9%  e2_sq < 0.008169
        + 0.06742873 * max(0.0, Q.sum_pt_top5 - 367.5938) / 230.8403   # +6.7%  sum_pt_top5 > 367.6
        - 0.06201367 * max(0.0, 0.06164517 - Q.z_7) / 0.01421946   # -6.2%  z_7 < 0.06165
        + 0.05983855 * max(0.0, Q.log_sum_pt - 6.377723) / 0.209568   # +6.0%  log_sum_pt > 6.378
        + 0.04074226 * max(0.0, Q.log_sum_pt - 6.502799) / 0.1247664   # +4.1%  log_sum_pt > 6.503
        + 0.03445715 * max(0.0, Q.pt_7 - 34.53125) / 4.473662   # +3.4%  pt_7 > 34.53
        - 0.03402721 * max(0.0, Q.sum_pt_top5 - 531.1875) / 106.1442   # -3.4%  sum_pt_top5 > 531.2
        + 0.01999276 * max(0.0, Q.sj3_dr_max - 0.169029) / 0.04795102   # +2.0%  sj3_dr_max > 0.169
        - 0.01814629 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.01713288 - Q.tau2) / 0.03817441   # -1.8%  pt_7 > 34.53 and tau2 < 0.01713
        + 0.01666542 * max(0.0, Q.log_sum_pt - 6.605974) / 0.07073019   # +1.7%  log_sum_pt > 6.606
        - 0.01493106 * max(0.0, Q.log_sum_pt - 6.377723) * max(0.0, 0.7509095 - Q.z_dr_0p05_0p1) / 0.1196218   # -1.5%  log_sum_pt > 6.378 and z_dr_0p05_0p1 < 0.7509
        - 0.01475051 * max(0.0, Q.sj3_dr_min - 0.03628191) / 0.02376249   # -1.5%  sj3_dr_min > 0.03628
        - 0.00819994 * max(0.0, 0.005832932 - Q.mass_over_sum_pt_sq) * max(0.0, Q.centroid_offset - 0.00809236) / 1.769648e-05   # -0.8%  mass_over_sum_pt_sq < 0.005833 and centroid_offset > 0.008092
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 5.902;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.901732 * (0.2902915
        + 0.1901826 * max(0.0, Q.N2 - 0.1150852) / 0.1151384   # +19.0%  N2 > 0.1151
        - 0.1750748 * max(0.0, Q.LHA - 0.1329373) / 0.1175814   # -17.5%  LHA > 0.1329
        - 0.1714583 * max(0.0, 53.4375 - Q.pt_7) / 19.12094   # -17.1%  pt_7 < 53.44
        + 0.1329434 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # +13.3%  lam1 < 0.005954
        + 0.1278487 * max(0.0, 813.4156 - Q.sum_pt) / 129.0758   # +12.8%  sum_pt < 813.4
        - 0.08468109 * max(0.0, 36.22941 - Q.mass) / 10.11393   # -8.5%  mass < 36.23
        + 0.05715283 * max(0.0, 53.4375 - Q.pt_7) * max(0.0, 1.129616 - Q.D2_b2) / 9.433694   # +5.7%  pt_7 < 53.44 and D2_b2 < 1.13
        + 0.03188155 * max(0.0, 6.46415 - Q.log_sum_pt) / 0.07009951   # +3.2%  log_sum_pt < 6.464
        - 0.02877667 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.27415   # -2.9%  sum_pt_top5 > 752.1
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 18.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.08644 * (-0.2491174
        - 0.194743 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # -19.5%  girth2 > 0.008678
        + 0.1097316 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +11.0%  girth > 0.04082
        + 0.07491173 * max(0.0, Q.sj2_dr - 0.1872617) / 0.02732165   # +7.5%  sj2_dr > 0.1873
        - 0.07217827 * max(0.0, 0.004372139 - Q.girth2) / 0.00156571   # -7.2%  girth2 < 0.004372
        + 0.06579122 * max(0.0, Q.width - 0.007520088) / 0.002470136   # +6.6%  width > 0.00752
        + 0.0656169 * max(0.0, Q.girth - 0.04081947) * max(0.0, Q.log_sum_pt - 6.080494) / 0.008781747   # +6.6%  girth > 0.04082 and log_sum_pt > 6.08
        + 0.04275621 * max(0.0, Q.girth - 0.07608178) / 0.01059033   # +4.3%  girth > 0.07608
        - 0.03952813 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # -4.0%  tau1 > 0.05357
        + 0.03784746 * max(0.0, Q.lam1 - 0.008375572) / 0.001840934   # +3.8%  lam1 > 0.008376
        - 0.03319865 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, Q.sj3_pair_mass_min - 5.744224) / 0.1950293   # -3.3%  mass_over_sum_pt > 0.06814 and sj3_pair_mass_min > 5.744
        + 0.02957044 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # +3.0%  e2 > 0.06344
        - 0.02750887 * max(0.0, Q.mass_over_sum_pt - 0.0681391) * max(0.0, 0.2179769 - Q.sj2_dr) / 0.0001196243   # -2.8%  mass_over_sum_pt > 0.06814 and sj2_dr < 0.218
        + 0.02657832 * max(0.0, Q.tau1 - 0.1027642) / 0.008921664   # +2.7%  tau1 > 0.1028
        + 0.02154086 * max(0.0, Q.mass_over_sum_pt - 0.0681391) / 0.01539157   # +2.2%  mass_over_sum_pt > 0.06814
        + 0.02146821 * max(0.0, Q.centroid_offset - 0.01096064) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.02994202   # +2.1%  centroid_offset > 0.01096 and n_pt_above_50 < 8
        + 0.01942745 * max(0.0, Q.width - 0.007520088) * max(0.0, Q.sj3_pairmin_over_m - 0.07708997) / 0.0004577253   # +1.9%  width > 0.00752 and sj3_pairmin_over_m > 0.07709
        - 0.01686341 * max(0.0, Q.lam1 - 0.01200373) / 0.001228921   # -1.7%  lam1 > 0.012
        - 0.01640543 * max(0.0, Q.max_dr - 0.1027585) * max(0.0, 0.8460335 - Q.z_dr_0p05_0p1) / 0.02673292   # -1.6%  max_dr > 0.1028 and z_dr_0p05_0p1 < 0.846
        - 0.01430183 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # -1.4%  lam2 > 0.001131
        - 0.01201308 * max(0.0, Q.width - 0.01323868) / 0.001442684   # -1.2%  width > 0.01324
        - 0.01083682 * max(0.0, Q.mass - 64.61873) / 3.274818   # -1.1%  mass > 64.62
        - 0.01060241 * max(0.0, Q.sd_mass - 62.55) * max(0.0, 0.9206502 - Q.D2_b2) / 1.823781   # -1.1%  sd_mass > 62.55 and D2_b2 < 0.9207
        + 0.00990201 * max(0.0, Q.sd_mass - 62.55) / 3.348271   # +1.0%  sd_mass > 62.55
        - 0.009034869 * max(0.0, Q.sj2_dr - 0.1872617) * max(0.0, Q.sj2_mass1 - 2.250113) / 0.39507   # -0.9%  sj2_dr > 0.1873 and sj2_mass1 > 2.25
        + 0.008450728 * max(0.0, Q.lam2 - 0.001130645) * max(0.0, 56.53125 - Q.pt_6) / 0.005634141   # +0.8%  lam2 > 0.001131 and pt_6 < 56.53
        + 0.008084516 * max(0.0, Q.e2 - 0.06344108) * max(0.0, Q.sj2_mass1 - 16.86126) / 0.02534836   # +0.8%  e2 > 0.06344 and sj2_mass1 > 16.86
        + 0.001107614 * max(0.0, Q.lam1 - 0.01643375) * max(0.0, 6.0 - Q.n_for_90pct) / 5.140257e-06   # +0.1%  lam1 > 0.01643 and n_for_90pct < 6
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 79.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 79.00836 * (-0.02511779
        + 0.2930985 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # +29.3%  e2_sq < 0.01166
        - 0.2427356 * max(0.0, 0.0116609 - Q.mass_over_sum_pt_sq) / 0.007206102   # -24.3%  mass_over_sum_pt_sq < 0.01166
        - 0.05231056 * max(0.0, 0.000537286 - Q.lam2) / 0.0003909825   # -5.2%  lam2 < 0.0005373
        - 0.04723678 * max(0.0, 0.233678 - Q.sj3_dr_max) / 0.09057682   # -4.7%  sj3_dr_max < 0.2337
        + 0.04192086 * max(0.0, 0.06729223 - Q.C2) / 0.04162086   # +4.2%  C2 < 0.06729
        + 0.04014189 * max(0.0, 0.2233283 - Q.N2) / 0.05117962   # +4.0%  N2 < 0.2233
        - 0.03135552 * max(0.0, 0.04990367 - Q.centroid_offset) / 0.03352573   # -3.1%  centroid_offset < 0.0499
        - 0.02700381 * max(0.0, 0.05028464 - Q.e2) / 0.02502152   # -2.7%  e2 < 0.05028
        + 0.02692087 * max(0.0, 0.0001869378 - Q.e3) / 0.0001495243   # +2.7%  e3 < 0.0001869
        + 0.02247816 * max(0.0, 0.007639643 - Q.girth2_top2) / 0.004543329   # +2.2%  girth2_top2 < 0.00764
        + 0.01936286 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +1.9%  width > 0.003563
        + 0.01762195 * max(0.0, 0.001653836 - Q.width) / 0.0004289067   # +1.8%  width < 0.001654
        - 0.01569309 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -1.6%  mass_over_sum_pt > 0.09041
        + 0.01431185 * max(0.0, Q.sd_mass - 44.82259) / 8.759065   # +1.4%  sd_mass > 44.82
        + 0.01319697 * max(0.0, 0.3456459 - Q.sj3_dr_max) / 0.1840851   # +1.3%  sj3_dr_max < 0.3456
        - 0.01292911 * max(0.0, 0.2233283 - Q.N2) * max(0.0, Q.eccentricity - 0.7117266) / 0.01381411   # -1.3%  N2 < 0.2233 and eccentricity > 0.7117
        + 0.01174119 * max(0.0, 0.3467135 - Q.LHA) / 0.1128474   # +1.2%  LHA < 0.3467
        - 0.01112167 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -1.1%  girth2 < 0.008678
        - 0.008558169 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 53.4375 - Q.pt_7) / 0.8992052   # -0.9%  N2 < 0.2233 and pt_7 < 53.44
        + 0.008433946 * max(0.0, Q.max_dr - 0.1598486) / 0.0215652   # +0.8%  max_dr > 0.1598
        - 0.00586963 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.6%  mass > 76.66
        - 0.004324401 * max(0.0, 0.007639643 - Q.girth2_top2) * max(0.0, Q.C2_b2 - 0.0006435798) / 6.028305e-06   # -0.4%  girth2_top2 < 0.00764 and C2_b2 > 0.0006436
        - 0.004238726 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -0.4%  girth > 0.08724
        - 0.00408902 * max(0.0, Q.width - 0.003562611) * max(0.0, 0.1057566 - Q.sj3_z3) / 5.535701e-05   # -0.4%  width > 0.003563 and sj3_z3 < 0.1058
        + 0.004049798 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +0.4%  e2 > 0.04447
        - 0.004014784 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, 0.275762 - Q.sd_zg) / 0.3498594   # -0.4%  sd_mass > 44.82 and sd_zg < 0.2758
        - 0.003926361 * max(0.0, 739.5 - Q.sum_pt) / 82.60981   # -0.4%  sum_pt < 739.5
        - 0.003925799 * max(0.0, 0.2233283 - Q.N2) * max(0.0, 62.55 - Q.mass) / 0.3959152   # -0.4%  N2 < 0.2233 and mass < 62.55
        - 0.003855667 * max(0.0, Q.sd_mass - 44.82259) * max(0.0, Q.centroid_offset - 0.001308549) / 0.1760217   # -0.4%  sd_mass > 44.82 and centroid_offset > 0.001309
        + 0.001867595 * max(0.0, Q.sd_mass - 74.57663) / 1.60503   # +0.2%  sd_mass > 74.58
        - 0.00166489 * max(0.0, 0.177305 - Q.max_dr) / 0.0704235   # -0.2%  max_dr < 0.1773
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.88556 * (0.05477832
        + 0.111646 * max(0.0, 0.005284669 - Q.e2_sq) / 0.00223735   # +11.2%  e2_sq < 0.005285
        - 0.1084948 * max(0.0, 0.0211821 - Q.zdr_0) / 0.009038773   # -10.8%  zdr_0 < 0.02118
        + 0.08243968 * max(0.0, 0.001653836 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 7.0272e-06   # +8.2%  girth2 < 0.001654 and centroid_offset < 0.02355
        + 0.0640824 * max(0.0, 0.001653836 - Q.girth2) / 0.0004289067   # +6.4%  girth2 < 0.001654
        - 0.0570446 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 0.03117077 - Q.centroid_offset) / 0.0004149607   # -5.7%  z_7 < 0.07149 and centroid_offset < 0.03117
        - 0.05165615 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -5.2%  log_sum_pt > 6.701
        - 0.05055332 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.004073493   # -5.1%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.04725391 * max(0.0, 0.002074109 - Q.e2_sq) / 0.000670556   # -4.7%  e2_sq < 0.002074
        + 0.04608047 * max(0.0, 0.04939969 - Q.z_7) / 0.007457505   # +4.6%  z_7 < 0.0494
        + 0.04464465 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 28.39396 - Q.pt1_dr01) / 4201.911   # +4.5%  sum_pt_top5 > 430.8 and pt1_dr01 < 28.39
        + 0.04340155 * max(0.0, 0.2160559 - Q.LHA) / 0.03222865   # +4.3%  LHA < 0.2161
        - 0.04246144 * max(0.0, 0.2160559 - Q.LHA) * max(0.0, 0.001503553 - Q.lam1) / 4.027805e-05   # -4.2%  LHA < 0.2161 and lam1 < 0.001504
        - 0.03720579 * max(0.0, 0.002151568 - Q.girth2_top3) / 0.0007789603   # -3.7%  girth2_top3 < 0.002152
        + 0.0354519 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +3.5%  width < 0.003563
        - 0.02875448 * max(0.0, 37.15625 - Q.pt_7) / 5.803748   # -2.9%  pt_7 < 37.16
        + 0.02787083 * max(0.0, 0.07148865 - Q.z_7) * max(0.0, 40.2 - Q.mass_top3) / 0.60086   # +2.8%  z_7 < 0.07149 and mass_top3 < 40.2
        + 0.02441352 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # +2.4%  z_7 < 0.02807
        + 0.02249335 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, Q.sj3_dr13 - 0.04995258) / 0.000417922   # +2.2%  zdr_0 < 0.02118 and sj3_dr13 > 0.04995
        - 0.01823 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # -1.8%  log_sum_pt > 6.896
        + 0.01791262 * max(0.0, 0.007673833 - Q.girth) / 0.000231517   # +1.8%  girth < 0.007674
        - 0.01347384 * max(0.0, 0.005284669 - Q.e2_sq) * max(0.0, 0.01437952 - Q.centroid_offset) / 1.318015e-05   # -1.3%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.01224017 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, Q.sj2_dr - 0.1682655) / 4.310225   # +1.2%  sum_pt_top5 > 430.8 and sj2_dr > 0.1683
        + 0.0107353 * max(0.0, Q.log_sum_pt - 6.701242) * max(0.0, 0.01341502 - Q.dr_2) / 0.0001347035   # +1.1%  log_sum_pt > 6.701 and dr_2 < 0.01342
        + 0.001459201 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.05744392 - Q.D2_b2) / 1.220721e-05   # +0.1%  log_sum_pt > 6.896 and D2_b2 < 0.05744
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 34.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.37621 * (0.0150494
        - 0.1924843 * max(0.0, 0.008678045 - Q.girth2) / 0.004447347   # -19.2%  girth2 < 0.008678
        + 0.08933652 * max(0.0, 0.1136369 - Q.tau1) / 0.05730443   # +8.9%  tau1 < 0.1136
        - 0.08500425 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # -8.5%  width < 0.01324
        + 0.07346942 * max(0.0, 0.0005116989 - Q.e3) / 0.0004510312   # +7.3%  e3 < 0.0005117
        + 0.06781274 * max(0.0, 0.1789613 - Q.sj3_dr_max) / 0.05394779   # +6.8%  sj3_dr_max < 0.179
        + 0.0626809 * max(0.0, 0.01200373 - Q.lam1) / 0.007316288   # +6.3%  lam1 < 0.012
        - 0.05769248 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # -5.8%  max_dr < 0.1452
        - 0.04995309 * max(0.0, 0.001130645 - Q.lam2) / 0.0009137232   # -5.0%  lam2 < 0.001131
        + 0.04705957 * max(0.0, Q.centroid_offset - 0.00809236) / 0.01052915   # +4.7%  centroid_offset > 0.008092
        - 0.042685 * max(0.0, 60.63098 - Q.mass) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 1.341509   # -4.3%  mass < 60.63 and z_dr_0p2_0p4 < 0.05644
        + 0.04068149 * max(0.0, 0.0030133 - Q.e2_sq) / 0.001065944   # +4.1%  e2_sq < 0.003013
        + 0.03538179 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, 6.766778 - Q.log_sum_pt) / 1.016213   # +3.5%  pt_6 < 41.22 and log_sum_pt < 6.767
        - 0.02915165 * max(0.0, Q.centroid_offset - 0.01837778) / 0.005523694   # -2.9%  centroid_offset > 0.01838
        - 0.02823399 * max(0.0, 41.21875 - Q.pt_6) * max(0.0, Q.z_7 - 0.02320757) / 0.07098489   # -2.8%  pt_6 < 41.22 and z_7 > 0.02321
        + 0.01492519 * max(0.0, Q.sj3_dr_max - 0.1879486) / 0.03937116   # +1.5%  sj3_dr_max > 0.1879
        + 0.01425077 * max(0.0, 0.00733008 - Q.lam1) / 0.003486853   # +1.4%  lam1 < 0.00733
        - 0.01352341 * max(0.0, 39.75 - Q.pt_6) / 4.713628   # -1.4%  pt_6 < 39.75
        + 0.01250884 * max(0.0, 41.21875 - Q.pt_6) / 5.483736   # +1.3%  pt_6 < 41.22
        + 0.01101189 * max(0.0, 45.595 - Q.mass) / 14.73532   # +1.1%  mass < 45.59
        - 0.009635078 * max(0.0, Q.sj3_pair_mass_min - 4.501727) / 3.480268   # -1.0%  sj3_pair_mass_min > 4.502
        + 0.006808664 * max(0.0, 0.003408389 - Q.lam2) * max(0.0, 3.0 - Q.n_dr_0_0p05) / 0.002658741   # +0.7%  lam2 < 0.003408 and n_dr_0_0p05 < 3
        + 0.005532426 * max(0.0, Q.centroid_offset - 0.01837778) * max(0.0, 0.008921136 - Q.mean_phi2) / 2.179362e-05   # +0.6%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        - 0.004173301 * max(0.0, Q.sj3_dr_min - 0.1278212) / 0.008209871   # -0.4%  sj3_dr_min > 0.1278
        + 0.003375885 * max(0.0, Q.centroid_offset - 0.00809236) * max(0.0, Q.sj3_pair_mass_min - 6.811308) / 0.05438016   # +0.3%  centroid_offset > 0.008092 and sj3_pair_mass_min > 6.811
        - 0.002627322 * max(0.0, Q.sj3_pair_mass_min - 4.501727) * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 2.69383   # -0.3%  sj3_pair_mass_min > 4.502 and n_dr_0p2_0p4 > 1
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 42.17;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 42.17466 * (0.2031735
        - 0.08648765 * max(0.0, Q.width - 0.005590289) / 0.003084256   # -8.6%  width > 0.00559
        - 0.07292203 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -7.3%  mass_over_sum_pt > 0.09041
        - 0.06830264 * max(0.0, 0.08723651 - Q.girth) / 0.03638569   # -6.8%  girth < 0.08724
        - 0.06609943 * max(0.0, 8.10414e-05 - Q.girth2_top2) / 7.505045e-06   # -6.6%  girth2_top2 < 8.104e-05
        + 0.06145613 * max(0.0, Q.girth2 - 0.01323868) / 0.001442684   # +6.1%  girth2 > 0.01324
        + 0.06074104 * max(0.0, Q.girth2 - 0.001653836) / 0.005220304   # +6.1%  girth2 > 0.001654
        + 0.06016616 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +6.0%  mass_over_sum_pt > 0.08475
        - 0.05831545 * max(0.0, Q.mass_over_sum_pt - 0.01109984) / 0.05101851   # -5.8%  mass_over_sum_pt > 0.0111
        - 0.05721873 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -5.7%  girth2 > 0.00752
        - 0.05361519 * max(0.0, 0.1872617 - Q.sj2_dr) / 0.06455312   # -5.4%  sj2_dr < 0.1873
        - 0.0461706 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # -4.6%  e3 < 8.148e-05
        + 0.0458269 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +4.6%  sj2_dr < 0.1592
        + 0.03853341 * max(0.0, Q.mass_over_sum_pt - 0.07269073) / 0.01344138   # +3.9%  mass_over_sum_pt > 0.07269
        + 0.03674571 * max(0.0, Q.width - 0.008678045) / 0.002214537   # +3.7%  width > 0.008678
        - 0.03212508 * max(0.0, Q.girth2 - 0.004372139) / 0.003638805   # -3.2%  girth2 > 0.004372
        + 0.02060741 * max(0.0, 0.3033137 - Q.LHA) / 0.07849185   # +2.1%  LHA < 0.3033
        - 0.01827937 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -1.8%  lam1 < 0.008376
        + 0.01729719 * max(0.0, Q.mass - 36.22941) / 14.20528   # +1.7%  mass > 36.23
        + 0.01726075 * max(0.0, 0.05356915 - Q.tau1) / 0.01695492   # +1.7%  tau1 < 0.05357
        - 0.01653843 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # -1.7%  e2 > 0.05028
        - 0.01278112 * max(0.0, 0.0009641429 - Q.girth2) / 0.0002052288   # -1.3%  girth2 < 0.0009641
        + 0.01264667 * max(0.0, Q.girth2 - 0.004372139) * max(0.0, 0.1950135 - Q.planar_flow) / 0.0002638045   # +1.3%  girth2 > 0.004372 and planar_flow < 0.195
        - 0.01165898 * max(0.0, 0.04081947 - Q.girth) / 0.009176315   # -1.2%  girth < 0.04082
        - 0.009389007 * max(0.0, 0.1950135 - Q.planar_flow) * max(0.0, Q.width - 0.007520088) / 0.0001455029   # -0.9%  planar_flow < 0.195 and width > 0.00752
        + 0.009079739 * max(0.0, 0.1294903 - Q.sj2_dr) / 0.03543038   # +0.9%  sj2_dr < 0.1295
        - 0.007786147 * max(0.0, Q.mass - 76.6557) / 1.546047   # -0.8%  mass > 76.66
        - 0.001949025 * max(0.0, Q.centroid_offset - 0.03776099) / 0.001689372   # -0.2%  centroid_offset > 0.03776
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 12.88;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.8828 * (-0.04390082
        - 0.1113681 * max(0.0, 0.1967397 - Q.LHA) / 0.02506564   # -11.1%  LHA < 0.1967
        - 0.09637859 * max(0.0, 0.06108601 - Q.girth) / 0.01868685   # -9.6%  girth < 0.06109
        + 0.09587451 * max(0.0, 0.006679471 - Q.girth2) * max(0.0, 0.02355416 - Q.centroid_offset) / 3.883081e-05   # +9.6%  girth2 < 0.006679 and centroid_offset < 0.02355
        + 0.0809013 * max(0.0, 0.05356915 - Q.tau1) * max(0.0, 0.003562611 - Q.width) / 4.808955e-05   # +8.1%  tau1 < 0.05357 and width < 0.003563
        - 0.08038845 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -8.0%  sj3_dr_max < 0.1426
        - 0.07330115 * max(0.0, 29.6447 - Q.mass) * max(0.0, 0.003562611 - Q.width) / 0.02261167   # -7.3%  mass < 29.64 and width < 0.003563
        + 0.07327101 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, 0.1009734 - Q.z_dr_0p2_0p4) / 0.0001901112   # +7.3%  girth2 < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.06647911 * max(0.0, 0.03319429 - Q.mass_over_sum_pt) / 0.00722828   # +6.6%  mass_over_sum_pt < 0.03319
        + 0.0527207 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.0001947983 - Q.lam2) / 2.869243e-06   # +5.3%  girth < 0.06109 and lam2 < 0.0001948
        + 0.04695749 * max(0.0, 0.1986272 - Q.sj3_dr_max) * max(0.0, 0.006679471 - Q.girth2) / 0.0003703114   # +4.7%  sj3_dr_max < 0.1986 and girth2 < 0.006679
        - 0.04544474 * max(0.0, Q.log_sum_pt - 6.701242) / 0.03523873   # -4.5%  log_sum_pt > 6.701
        + 0.0419931 * max(0.0, 0.005019719 - Q.girth2) / 0.001903424   # +4.2%  girth2 < 0.00502
        + 0.02709899 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619   # +2.7%  sum_pt_top5 > 687.4
        - 0.02639821 * max(0.0, 0.06108601 - Q.girth) * max(0.0, 0.05643852 - Q.z_dr_0p2_0p4) / 0.001011075   # -2.6%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.02148801 * max(0.0, 0.003562611 - Q.width) / 0.001184249   # +2.1%  width < 0.003563
        - 0.02118592 * max(0.0, 0.005019719 - Q.girth2) * max(0.0, Q.centroid_offset - 0.006789738) / 1.064349e-05   # -2.1%  girth2 < 0.00502 and centroid_offset > 0.00679
        - 0.01999982 * max(0.0, 21.78408 - Q.mass) * max(0.0, 48.71875 - Q.pt_7) / 69.12746   # -2.0%  mass < 21.78 and pt_7 < 48.72
        - 0.01875085 * max(0.0, Q.z_dr_0_0p05 - 0.8477313) / 0.0544886   # -1.9%  z_dr_0_0p05 > 0.8477
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 37.19;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.19 * (-0.08637518
        + 0.1065106 * max(0.0, 0.00609665 - Q.girth2) / 0.002544039   # +10.7%  girth2 < 0.006097
        + 0.0749597 * max(0.0, 0.003562611 - Q.girth2) / 0.001184249   # +7.5%  girth2 < 0.003563
        + 0.06719313 * max(0.0, 988.4078 - Q.sum_pt) / 276.7667   # +6.7%  sum_pt < 988.4
        - 0.05836676 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # -5.8%  sj3_dr_max < 0.1426
        - 0.05521275 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -5.5%  lam1 < 0.005954
        + 0.05402735 * max(0.0, 53.33237 - Q.mass) * max(0.0, 0.02685622 - Q.centroid_offset) / 0.2973626   # +5.4%  mass < 53.33 and centroid_offset < 0.02686
        - 0.05096821 * max(0.0, 53.33237 - Q.mass) / 19.36004   # -5.1%  mass < 53.33
        + 0.0478839 * max(0.0, 0.01837778 - Q.centroid_offset) / 0.006717136   # +4.8%  centroid_offset < 0.01838
        + 0.04070951 * max(0.0, Q.lam1 - 0.005433361) / 0.00267714   # +4.1%  lam1 > 0.005433
        + 0.03837836 * max(0.0, 0.213399 - Q.sj3_dr_max) / 0.07579536   # +3.8%  sj3_dr_max < 0.2134
        + 0.03597767 * max(0.0, 0.1117619 - Q.max_dr) / 0.02839465   # +3.6%  max_dr < 0.1118
        - 0.03565823 * max(0.0, 0.07269073 - Q.mass_over_sum_pt) / 0.02485979   # -3.6%  mass_over_sum_pt < 0.07269
        + 0.03240127 * max(0.0, 0.04369778 - Q.tau1) / 0.01230908   # +3.2%  tau1 < 0.0437
        - 0.03013294 * max(0.0, Q.girth - 0.0717028) / 0.01201981   # -3.0%  girth > 0.0717
        - 0.02474759 * max(0.0, 2.371297e-05 - Q.e3) / 1.105328e-05   # -2.5%  e3 < 2.371e-05
        - 0.02388294 * max(0.0, 0.05464922 - Q.girth) / 0.01532978   # -2.4%  girth < 0.05465
        - 0.02266504 * max(0.0, Q.lam1 - 0.003377388) / 0.003672352   # -2.3%  lam1 > 0.003377
        + 0.01982125 * max(0.0, Q.e2 - 0.04447357) / 0.004351923   # +2.0%  e2 > 0.04447
        - 0.0163122 * max(0.0, 0.0284695 - Q.C3) / 0.008714909   # -1.6%  C3 < 0.02847
        - 0.01512571 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -1.5%  e2 > 0.03556
        - 0.01451776 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 159.25 - Q.pt_1) / 0.1931178   # -1.5%  centroid_offset < 0.01838 and pt_1 < 159.2
        + 0.01419415 * max(0.0, Q.e3 - 8.147744e-05) / 4.997827e-05   # +1.4%  e3 > 8.148e-05
        - 0.01365379 * max(0.0, 24.23013 - Q.sj3_pair_mass_max) / 6.087676   # -1.4%  sj3_pair_mass_max < 24.23
        - 0.01317198 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.0) / 0.008712838   # -1.3%  centroid_offset < 0.01838 and n_for_90pct > 5
        + 0.01261791 * max(0.0, 0.00609665 - Q.girth2) * max(0.0, 0.3220738 - Q.planar_flow) / 0.0002201438   # +1.3%  girth2 < 0.006097 and planar_flow < 0.3221
        - 0.01146165 * max(0.0, Q.lam1 - 0.005433361) * max(0.0, Q.eccentricity - 0.7117266) / 0.0005312449   # -1.1%  lam1 > 0.005433 and eccentricity > 0.7117
        - 0.01080367 * max(0.0, Q.e3 - 0.0001869378) / 3.769933e-05   # -1.1%  e3 > 0.0001869
        + 0.009686079 * max(0.0, 53.33237 - Q.mass) * max(0.0, 6.842717 - Q.log_sum_pt) / 5.175436   # +1.0%  mass < 53.33 and log_sum_pt < 6.843
        + 0.009300733 * max(0.0, 0.01837778 - Q.centroid_offset) * max(0.0, 0.2055511 - Q.z_2nd) / 0.0002178654   # +0.9%  centroid_offset < 0.01838 and z_2nd < 0.2056
        - 0.007287104 * max(0.0, 0.006292091 - Q.zdr_0) / 0.001006321   # -0.7%  zdr_0 < 0.006292
        - 0.006680785 * Q.z_dr_0p2_0p4 / 0.02920532   # -0.7%  z_dr_0p2_0p4
        - 0.006455188 * max(0.0, 0.04369778 - Q.tau1) * max(0.0, Q.eccentricity - 0.9031255) / 0.000277001   # -0.6%  tau1 < 0.0437 and eccentricity > 0.9031
        + 0.004715821 * max(0.0, 559.6875 - Q.sum_pt) / 16.18216   # +0.5%  sum_pt < 559.7
        + 0.004629232 * max(0.0, Q.lam2 - 0.001130645) / 0.0003119523   # +0.5%  lam2 > 0.001131
        + 0.004226901 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857   # +0.4%  n_dr_0p2_0p4 > 1
        - 0.00248435 * max(0.0, Q.sum_pt_top5 - 839.9547) / 8.605889   # -0.2%  sum_pt_top5 > 840
        + 0.00224274 * max(0.0, Q.log_sum_pt - 6.896095) / 0.004034898   # +0.2%  log_sum_pt > 6.896
        - 0.0009350807 * max(0.0, Q.log_sum_pt - 6.896095) * max(0.0, 0.716559 - Q.D2_b2) / 0.000627491   # -0.1%  log_sum_pt > 6.896 and D2_b2 < 0.7166
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 10.81;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.80706 * (0.1412037
        + 0.111354 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # +11.1%  girth2 > 0.00752
        + 0.09821674 * max(0.0, Q.tau1 - 0.05356915) / 0.02676347   # +9.8%  tau1 > 0.05357
        - 0.09658578 * Q.e2 / 0.02863215   # -9.7%  e2
        + 0.09646806 * max(0.0, 76.6557 - Q.mass) / 37.88099   # +9.6%  mass < 76.66
        + 0.09154338 * max(0.0, 8.147744e-05 - Q.e3) / 5.634282e-05   # +9.2%  e3 < 8.148e-05
        - 0.07302928 * max(0.0, 0.00595415 - Q.lam1) / 0.002518154   # -7.3%  lam1 < 0.005954
        - 0.07004769 * max(0.0, 0.001503553 - Q.lam1) / 0.0003926183   # -7.0%  lam1 < 0.001504
        - 0.05763854 * max(0.0, Q.LHA - 0.3033137) / 0.01794309   # -5.8%  LHA > 0.3033
        - 0.04804754 * max(0.0, 45.75 - Q.pt_7) / 12.14632   # -4.8%  pt_7 < 45.75
        - 0.04641323 * max(0.0, 0.004183811 - Q.lam1) / 0.001513077   # -4.6%  lam1 < 0.004184
        + 0.03513508 * max(0.0, 0.0211821 - Q.zdr_0) * max(0.0, 0.2919447 - Q.z_dr_0p05_0p1) / 0.002082998   # +3.5%  zdr_0 < 0.02118 and z_dr_0p05_0p1 < 0.2919
        + 0.02911295 * max(0.0, Q.lam2 - 0.0003061234) / 0.0004216599   # +2.9%  lam2 > 0.0003061
        - 0.02871274 * max(0.0, Q.lam1 - 0.00733008) / 0.002073133   # -2.9%  lam1 > 0.00733
        - 0.02555916 * max(0.0, Q.e3 - 3.892127e-05) / 5.800315e-05   # -2.6%  e3 > 3.892e-05
        + 0.0213456 * max(0.0, 45.75 - Q.pt_7) * max(0.0, 1.002471 - Q.D2) / 1.812028   # +2.1%  pt_7 < 45.75 and D2 < 1.002
        + 0.01986077 * max(0.0, Q.C2_b2 - 0.009032972) / 0.001703043   # +2.0%  C2_b2 > 0.009033
        + 0.01861883 * max(0.0, Q.sj3_pair_mass_min - 11.051) / 1.943131   # +1.9%  sj3_pair_mass_min > 11.05
        - 0.01420755 * max(0.0, Q.lam1 - 0.00733008) * max(0.0, 0.380911 - Q.D2_b2) / 0.0002929492   # -1.4%  lam1 > 0.00733 and D2_b2 < 0.3809
        - 0.01012192 * max(0.0, Q.lam2 - 0.003408389) / 0.0001570299   # -1.0%  lam2 > 0.003408
        - 0.007981132 * max(0.0, 8.147744e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.1797097) / 5.905455e-07   # -0.8%  e3 < 8.148e-05 and sj3_dr23 > 0.1797
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 28.51;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.51106 * (-0.1391351
        + 0.2342864 * max(0.0, 0.008678045 - Q.width) / 0.004447348   # +23.4%  width < 0.008678
        - 0.1103096 * max(0.0, 0.169029 - Q.sj3_dr_max) / 0.04876366   # -11.0%  sj3_dr_max < 0.169
        + 0.1007656 * max(0.0, 0.03776099 - Q.centroid_offset) / 0.02226603   # +10.1%  centroid_offset < 0.03776
        + 0.08216264 * max(0.0, 0.01323868 - Q.girth2) / 0.008236125   # +8.2%  girth2 < 0.01324
        - 0.0800612 * max(0.0, 0.008375572 - Q.lam1) / 0.004300145   # -8.0%  lam1 < 0.008376
        + 0.0690143 * max(0.0, 0.2623172 - Q.sj3_dr_max) / 0.1129006   # +6.9%  sj3_dr_max < 0.2623
        - 0.06264593 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -6.3%  e2_sq < 0.008169
        - 0.05754349 * max(0.0, 0.0717028 - Q.girth) / 0.02501835   # -5.8%  girth < 0.0717
        + 0.04301609 * max(0.0, Q.z_7 - 0.01685855) / 0.03557082   # +4.3%  z_7 > 0.01686
        - 0.03102698 * max(0.0, 0.03776099 - Q.centroid_offset) * max(0.0, 901.5938 - Q.sum_pt) / 3.513015   # -3.1%  centroid_offset < 0.03776 and sum_pt < 901.6
        - 0.02393509 * max(0.0, 0.02054282 - Q.girth) / 0.002592388   # -2.4%  girth < 0.02054
        + 0.02311789 * max(0.0, 0.1452311 - Q.max_dr) / 0.04781504   # +2.3%  max_dr < 0.1452
        + 0.01963545 * max(0.0, 0.1426152 - Q.sj3_dr_max) / 0.03718906   # +2.0%  sj3_dr_max < 0.1426
        - 0.01908407 * max(0.0, 0.09538712 - Q.tau1) / 0.0425821   # -1.9%  tau1 < 0.09539
        + 0.01446913 * max(0.0, 15.45403 - Q.mass) / 2.385786   # +1.4%  mass < 15.45
        - 0.01126383 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 2.843757 - Q.D2) / 0.001362698   # -1.1%  centroid_offset > 0.0499 and D2 < 2.844
        + 0.007534327 * max(0.0, Q.sj2_dr - 0.2687922) / 0.008463482   # +0.8%  sj2_dr > 0.2688
        - 0.005735905 * max(0.0, 69.61135 - Q.mass) / 31.69729   # -0.6%  mass < 69.61
        + 0.00439216 * max(0.0, Q.centroid_offset - 0.04990367) * max(0.0, 0.5502779 - Q.tau32) / 0.0001096908   # +0.4%  centroid_offset > 0.0499 and tau32 < 0.5503
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4606;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4606047 * (-3.123307
        + 0.572924 * max(0.0, Q.girth2 - 0.01882765) / 0.0007605369   # +57.3%  girth2 > 0.01883
        - 0.3054028 * max(0.0, Q.e2 - 0.06344108) / 0.001761861   # -30.5%  e2 > 0.06344
        + 0.1216733 * max(0.0, Q.mass - 91.19) / 0.5839191   # +12.2%  mass > 91.19
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 15.08;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.08014 * (0.008017846
        + 0.3109666 * max(0.0, 0.1484084 - Q.girth) / 0.0905999   # +31.1%  girth < 0.1484
        + 0.1701928 * max(0.0, 0.01643375 - Q.lam1) / 0.0112011   # +17.0%  lam1 < 0.01643
        - 0.1351288 * max(0.0, 0.08000524 - Q.e2) / 0.05193423   # -13.5%  e2 < 0.08001
        - 0.0684505 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 6.804164 - Q.log_sum_pt) / 0.01998234   # -6.8%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.06185976 * max(0.0, 0.1484084 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708194   # -6.2%  girth < 0.1484 and pt_7 < 38.53
        + 0.06090282 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 40.04062 - Q.pt_7) / 781.5808   # +6.1%  sum_pt_top5 > 658.1 and pt_7 < 40.04
        - 0.05294181 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, 0.03776099 - Q.centroid_offset) / 8.324263e-07   # -5.3%  e3 < 5.335e-05 and centroid_offset < 0.03776
        + 0.04301961 * max(0.0, 0.006506576 - Q.lam1) / 0.002890078   # +4.3%  lam1 < 0.006507
        - 0.03390386 * max(0.0, 0.007520088 - Q.width) / 0.003544991   # -3.4%  width < 0.00752
        + 0.02744954 * max(0.0, 5.334511e-05 - Q.e3) * max(0.0, Q.D3 - 0.2213841) / 3.794981e-05   # +2.7%  e3 < 5.335e-05 and D3 > 0.2214
        - 0.02362455 * max(0.0, 0.02807091 - Q.z_7) / 0.001309815   # -2.4%  z_7 < 0.02807
        - 0.007138043 * max(0.0, 31.90625 - Q.pt_6) * max(0.0, 0.0586137 - Q.z_7) / 0.06765006   # -0.7%  pt_6 < 31.91 and z_7 < 0.05861
        - 0.004421353 * max(0.0, Q.sum_pt - 988.4078) * max(0.0, 62.25 - Q.pt_6) / 104.494   # -0.4%  sum_pt > 988.4 and pt_6 < 62.25
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 44.86;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 44.86022 * (-0.03143104
        - 0.108625 * max(0.0, Q.width - 0.006679471) / 0.002702003   # -10.9%  width > 0.006679
        + 0.07285997 * max(0.0, Q.girth - 0.02689598) / 0.03613842   # +7.3%  girth > 0.0269
        - 0.0685042 * max(0.0, Q.e2 - 0.01655442) / 0.01596508   # -6.9%  e2 > 0.01655
        - 0.06672669 * max(0.0, Q.girth2 - 0.007520088) / 0.002470136   # -6.7%  girth2 > 0.00752
        - 0.05561226 * max(0.0, Q.girth - 0.08723651) / 0.007853436   # -5.6%  girth > 0.08724
        - 0.05054674 * max(0.0, 74.57663 - Q.sd_mass) / 42.04056   # -5.1%  sd_mass < 74.58
        + 0.04754139 * max(0.0, Q.sj2_dr - 0.1591713) / 0.0392608   # +4.8%  sj2_dr > 0.1592
        + 0.04606229 * max(0.0, Q.mass_over_sum_pt - 0.08475161) / 0.009561895   # +4.6%  mass_over_sum_pt > 0.08475
        + 0.04510506 * max(0.0, Q.mass_over_sum_pt - 0.07992374) / 0.01088755   # +4.5%  mass_over_sum_pt > 0.07992
        + 0.04359065 * max(0.0, Q.girth - 0.04081947) / 0.0270611   # +4.4%  girth > 0.04082
        + 0.03687922 * max(0.0, Q.width - 0.003562611) / 0.004066872   # +3.7%  width > 0.003563
        + 0.0319173 * max(0.0, Q.e2 - 0.05028464) / 0.003369029   # +3.2%  e2 > 0.05028
        + 0.0311164 * max(0.0, 49.91626 - Q.sd_mass) / 22.60916   # +3.1%  sd_mass < 49.92
        - 0.02642187 * max(0.0, Q.mass_over_sum_pt - 0.09041383) / 0.008298543   # -2.6%  mass_over_sum_pt > 0.09041
        + 0.0260068 * max(0.0, Q.e2_sq - 0.002074109) / 0.004484017   # +2.6%  e2_sq > 0.002074
        - 0.02566688 * max(0.0, Q.e2_sq - 0.0030133) / 0.003940214   # -2.6%  e2_sq > 0.003013
        - 0.01996936 * max(0.0, Q.centroid_offset - 0.04990367) / 0.0008064001   # -2.0%  centroid_offset > 0.0499
        + 0.01944794 * max(0.0, Q.width - 0.01323868) / 0.001442684   # +1.9%  width > 0.01324
        + 0.0186688 * max(0.0, Q.e2 - 0.04110972) / 0.005106967   # +1.9%  e2 > 0.04111
        + 0.01658741 * max(0.0, Q.girth - 0.08065885) / 0.009330634   # +1.7%  girth > 0.08066
        - 0.01583627 * max(0.0, Q.girth - 0.1019409) / 0.005400003   # -1.6%  girth > 0.1019
        - 0.01538498 * max(0.0, 5.334511e-05 - Q.e3) / 3.302915e-05   # -1.5%  e3 < 5.335e-05
        - 0.01355791 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00733008 - Q.lam1) / 5.912632e-05   # -1.4%  planar_flow < 0.1115 and lam1 < 0.00733
        - 0.01129457 * max(0.0, Q.sj2_dr - 0.2001708) / 0.02317349   # -1.1%  sj2_dr > 0.2002
        - 0.0101808 * max(0.0, Q.sj2_dr - 0.06154135) / 0.1002688   # -1.0%  sj2_dr > 0.06154
        + 0.01006603 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01643375 - Q.lam1) / 0.0003201213   # +1.0%  planar_flow < 0.1115 and lam1 < 0.01643
        + 0.009304656 * max(0.0, Q.e2_sq - 0.01165737) / 0.001433269   # +0.9%  e2_sq > 0.01166
        - 0.008692073 * max(0.0, Q.sj2_dr - 0.1294903) / 0.05597033   # -0.9%  sj2_dr > 0.1295
        - 0.008304148 * max(0.0, Q.e2 - 0.03556091) / 0.006790965   # -0.8%  e2 > 0.03556
        - 0.007911876 * max(0.0, Q.centroid_offset - 0.02685622) / 0.003254849   # -0.8%  centroid_offset > 0.02686
        + 0.007310059 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.00609665 - Q.width) / 3.351526e-05   # +0.7%  planar_flow < 0.1115 and width < 0.006097
        - 0.00645616 * max(0.0, Q.psi_0p1 - 0.9761279) / 0.01050546   # -0.6%  psi_0p1 > 0.9761
        + 0.006102822 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067297   # +0.6%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.005662245 * max(0.0, Q.z_dr_0p05_0p1 - 0.7509095) / 0.02281319   # -0.6%  z_dr_0p05_0p1 > 0.7509
        - 0.004125596 * max(0.0, 0.1115136 - Q.planar_flow) * max(0.0, 0.01837778 - Q.centroid_offset) / 0.0002206557   # -0.4%  planar_flow < 0.1115 and centroid_offset < 0.01838
        + 0.001953528 * max(0.0, Q.girth2 - 0.008678045) / 0.002214537   # +0.2%  girth2 > 0.008678
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 41.17;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.16944 * (-0.009420999
        + 0.1435965 * max(0.0, 0.01323868 - Q.width) / 0.008236124   # +14.4%  width < 0.01324
        - 0.1320112 * max(0.0, 0.1019409 - Q.girth) / 0.04863667   # -13.2%  girth < 0.1019
        + 0.09854173 * max(0.0, 0.04110972 - Q.e2) / 0.01758453   # +9.9%  e2 < 0.04111
        + 0.08077303 * max(0.0, 0.007182836 - Q.mass_over_sum_pt_sq) / 0.003528849   # +8.1%  mass_over_sum_pt_sq < 0.007183
        - 0.07873928 * max(0.0, 0.008168571 - Q.e2_sq) / 0.004294747   # -7.9%  e2_sq < 0.008169
        + 0.07722406 * max(0.0, 0.01882765 - Q.width) / 0.01314296   # +7.7%  width < 0.01883
        - 0.07400343 * max(0.0, 0.007520088 - Q.girth2) / 0.00354499   # -7.4%  girth2 < 0.00752
        + 0.06660065 * max(0.0, 0.1591713 - Q.sj2_dr) / 0.04840185   # +6.7%  sj2_dr < 0.1592
        - 0.06509161 * max(0.0, 0.2001708 - Q.sj2_dr) / 0.07331403   # -6.5%  sj2_dr < 0.2002
        - 0.0559084 * max(0.0, 0.01165737 - Q.e2_sq) / 0.007203073   # -5.6%  e2_sq < 0.01166
        - 0.05306029 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) / 0.07219749   # -5.3%  mass_over_sum_pt < 0.1309
        - 0.0227203 * max(0.0, 0.00609665 - Q.width) / 0.002544039   # -2.3%  width < 0.006097
        - 0.01436902 * max(0.0, 0.1492731 - Q.sj2_dr) / 0.04374278   # -1.4%  sj2_dr < 0.1493
        - 0.01290501 * max(0.0, 0.0005124533 - Q.girth2_top2) / 0.0001104529   # -1.3%  girth2_top2 < 0.0005125
        + 0.009792844 * max(0.0, 0.1309286 - Q.mass_over_sum_pt) * max(0.0, 0.7459513 - Q.D2) / 0.003993721   # +1.0%  mass_over_sum_pt < 0.1309 and D2 < 0.746
        + 0.007159787 * max(0.0, 0.00483998 - Q.lam1) / 0.001855088   # +0.7%  lam1 < 0.00484
        - 0.003770474 * max(0.0, 0.007520088 - Q.girth2) * max(0.0, 0.7459513 - Q.D2) / 9.127003e-05   # -0.4%  girth2 < 0.00752 and D2 < 0.746
        - 0.00373236 * max(0.0, 0.00609665 - Q.width) * max(0.0, 0.7459513 - Q.D2) / 3.18925e-05   # -0.4%  width < 0.006097 and D2 < 0.746
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.010059243697479, 0.854003781512605, 2.2305216386554623, 0.9484796218487395, 1.3745084033613446, 2.1700981092436975, 1.1166888655462184, 1.9433126050420169, 0.2818590336134454, 3.3152878151260503, 2.197067962184874, 2.1010965336134455, 0.0827016806722689, 3.5477189075630253, 0.46576144957983195, 0.3669298319327731]
T = [2.591643971573004, 1.5271247160582984, 3.435213232011554, 2.640859571953782, 3.1487609309348743]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +37%, n9 +22%, n5 -16%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3698144 * h[2] / H_AVG[2]
            + 0.2198663 * h[9] / H_AVG[9]
            - 0.157002 * h[5] / H_AVG[5]
            + 0.1287195 * h[1] / H_AVG[1]
            - 0.06089639 * h[0] / H_AVG[0]
            + 0.04712756 * h[6] / H_AVG[6]
            - 0.0165738 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +55%, n10 -18%, n6 +9%, n4 -8%, n5 +7%, n15 +2% ...
            + 0.5512139 * h[9] / H_AVG[9]
            - 0.179837 * h[10] / H_AVG[10]
            + 0.09140452 * h[6] / H_AVG[6]
            - 0.0843809 * h[4] / H_AVG[4]
            + 0.06661103 * h[5] / H_AVG[5]
            + 0.01501719 * h[15] / H_AVG[15]
            + 0.01153553 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +23%, n3 -14%, n7 +12%, n14 -10%, n6 -10%, n0 +10% ...
            + 0.2293631 * h[11] / H_AVG[11]
            - 0.1380525 * h[3] / H_AVG[3]
            + 0.1237477 * h[7] / H_AVG[7]
            - 0.1016883 * h[14] / H_AVG[14]
            - 0.1015847 * h[6] / H_AVG[6]
            + 0.1010732 * h[0] / H_AVG[0]
            - 0.07343482 * h[15] / H_AVG[15]
            + 0.07261528 * h[13] / H_AVG[13]
            - 0.03015904 * h[9] / H_AVG[9]
            - 0.02051248 * h[8] / H_AVG[8]
            - 0.007768839 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n3 -20%, n6 -16%, n13 +7%, n14 +7%, n4 +4% ...
            + 0.3449361 * h[7] / H_AVG[7]
            - 0.2020251 * h[3] / H_AVG[3]
            - 0.1585689 * h[6] / H_AVG[6]
            + 0.07346694 * h[13] / H_AVG[13]
            + 0.06613776 * h[14] / H_AVG[14]
            + 0.04066232 * h[4] / H_AVG[4]
            + 0.04042262 * h[1] / H_AVG[1]
            - 0.03923069 * h[9] / H_AVG[9]
            - 0.0217099 * h[15] / H_AVG[15]
            + 0.01283968 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -46%, n10 +26%, n5 -17%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4577232 * h[13] / H_AVG[13]
            + 0.2616586 * h[10] / H_AVG[10]
            - 0.1722978 * h[5] / H_AVG[5]
            + 0.05456545 * h[4] / H_AVG[4]
            + 0.01882645 * h[3] / H_AVG[3]
            + 0.01678393 * h[8] / H_AVG[8]
            - 0.01313242 * h[12] / H_AVG[12]
            + 0.005012186 * h[0] / H_AVG[0]
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
