"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.9%   (on for 90% of jets)
  neuron  9:  11.7%   (on for 72% of jets)
  neuron  7:  10.1%   (on for 68% of jets)
  neuron  6:   8.4%   (on for 35% of jets)
  neuron 10:   7.9%   (on for 73% of jets)
  neuron  3:   7.0%   (on for 25% of jets)
  neuron  5:   6.9%   (on for 57% of jets)
  neuron  2:   6.7%   (on for 88% of jets)
  neuron 11:   6.5%   (on for 78% of jets)
  neuron 14:   4.5%   (on for 29% of jets)
  neuron  0:   3.8%   (on for 44% of jets)
  neuron 15:   3.5%   (on for 33% of jets)
  neuron  4:   3.3%   (on for 75% of jets)
  neuron  1:   3.2%   (on for 64% of jets)
  neuron  8:   1.4%   (on for 51% of jets)
  neuron 12:   0.3%   (on for 99% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 87.2% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr12                   ΔR between particles 1 and 2
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
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
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        max_dr=max(dr[i] for i in real),
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
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
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 17.8;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.80145 * (-0.1123504
        + 0.2874929 * max(0.0, 0.0127 - Q.girth2) / 0.007777796   # +28.7%  girth2 < 0.0127
        - 0.1103006 * max(0.0, 0.0838 - Q.girth) / 0.03367941   # -11.0%  girth < 0.0838
        - 0.09206342 * max(0.0, 0.175 - Q.sd_rg) / 0.07955641   # -9.2%  sd_rg < 0.175
        + 0.07716744 * max(0.0, 0.119 - Q.sd_rg) / 0.04753259   # +7.7%  sd_rg < 0.119
        - 0.07482669 * max(0.0, 0.00555 - Q.girth2) / 0.002205336   # -7.5%  girth2 < 0.00555
        + 0.06702197 * max(0.0, 0.0632 - Q.tau1) / 0.02197215   # +6.7%  tau1 < 0.0632
        - 0.06054022 * max(0.0, 0.00369 - Q.width) / 0.001241594   # -6.1%  width < 0.00369
        - 0.0504492 * max(0.0, 0.0128 - Q.tau2) / 0.004726677   # -5.0%  tau2 < 0.0128
        + 0.04116764 * max(0.0, 0.149 - Q.planar_flow) * max(0.0, 1.48 - Q.D2_b2) / 0.06428451   # +4.1%  planar_flow < 0.149 and D2_b2 < 1.48
        + 0.02581299 * max(0.0, Q.n_dr_0_0p05 - 5.27) / 0.9694272   # +2.6%  n_dr_0_0p05 > 5.27
        + 0.02569345 * max(0.0, 0.02 - Q.girth2) * max(0.0, 0.0907 - Q.M3) / 0.0002209568   # +2.6%  girth2 < 0.02 and M3 < 0.0907
        - 0.02460704 * max(0.0, 0.0208 - Q.girth2) * max(0.0, Q.centroid_offset - 0.00894) / 0.0001016336   # -2.5%  girth2 < 0.0208 and centroid_offset > 0.00894
        - 0.01833722 * max(0.0, Q.centroid_offset - 0.0503) / 0.0007865761   # -1.8%  centroid_offset > 0.0503
        - 0.01632254 * max(0.0, 763.0 - Q.sum_pt) * max(0.0, 0.395 - Q.D2_b2) / 11.2622   # -1.6%  sum_pt < 763 and D2_b2 < 0.395
        + 0.01468404 * max(0.0, 603.0 - Q.sum_pt_top5) / 78.02901   # +1.5%  sum_pt_top5 < 603
        + 0.01170732 * max(0.0, Q.eccentricity - 0.988) / 0.002107252   # +1.2%  eccentricity > 0.988
        - 0.001805316 * max(0.0, 0.015 - Q.girth) * max(0.0, Q.dr0_6 - 0.199) / 4.898967e-07   # -0.2%  girth < 0.015 and dr0_6 > 0.199
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 9.262;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.26185 * (0.04934219
        - 0.3217533 * max(0.0, 0.00902 - Q.width) / 0.004722712   # -32.2%  width < 0.00902
        + 0.2389811 * max(0.0, 0.00723 - Q.e2_sq) / 0.003564262   # +23.9%  e2_sq < 0.00723
        + 0.1153566 * max(0.0, Q.log_sum_pt - 6.37) / 0.2154064   # +11.5%  log_sum_pt > 6.37
        + 0.08463684 * max(0.0, Q.log_sum_pt - 6.57) / 0.08778205   # +8.5%  log_sum_pt > 6.57
        - 0.08320175 * max(0.0, Q.log_sum_pt - 6.34) * max(0.0, 0.0739 - Q.dr_0) / 0.01012618   # -8.3%  log_sum_pt > 6.34 and dr_0 < 0.0739
        - 0.06597958 * max(0.0, 0.0529 - Q.z_7) / 0.009134424   # -6.6%  z_7 < 0.0529
        + 0.05040293 * max(0.0, Q.pt_7 - 35.0) / 4.243858   # +5.0%  pt_7 > 35
        - 0.0241727 * max(0.0, Q.pt_7 - 35.3) * max(0.0, 0.0186 - Q.tau2) / 0.04041225   # -2.4%  pt_7 > 35.3 and tau2 < 0.0186
        + 0.0155152 * max(0.0, Q.sj2_dr - 0.19) * max(0.0, 0.201 - Q.sj3_dr_min) / 0.00277412   # +1.6%  sj2_dr > 0.19 and sj3_dr_min < 0.201
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 8.846;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.846175 * (0.358347
        - 0.2383806 * max(0.0, 56.1 - Q.pt_7) / 21.67273   # -23.8%  pt_7 < 56.1
        + 0.1410158 * max(0.0, 0.0141 - Q.width) / 0.008974464   # +14.1%  width < 0.0141
        - 0.1058832 * max(0.0, Q.LHA - 0.113) / 0.1340002   # -10.6%  LHA > 0.113
        + 0.08995535 * max(0.0, 6.7 - Q.log_sum_pt) / 0.1926782   # +9.0%  log_sum_pt < 6.7
        - 0.0868786 * max(0.0, Q.z_7 - 0.039) / 0.01681714   # -8.7%  z_7 > 0.039
        + 0.05945192 * max(0.0, 0.0236 - Q.zdr_0) / 0.01091125   # +5.9%  zdr_0 < 0.0236
        - 0.05693042 * max(0.0, 0.0395 - Q.e2) / 0.01640445   # -5.7%  e2 < 0.0395
        + 0.05656828 * max(0.0, 0.00702 - Q.lam1) * max(0.0, 0.21 - Q.max_dr) / 0.0004633453   # +5.7%  lam1 < 0.00702 and max_dr < 0.21
        - 0.04658272 * max(0.0, 0.0362 - Q.C2) * max(0.0, 0.0256 - Q.C2_b2) / 0.0003887537   # -4.7%  C2 < 0.0362 and C2_b2 < 0.0256
        + 0.0348718 * max(0.0, Q.N2 - 0.194) / 0.06413349   # +3.5%  N2 > 0.194
        + 0.03187764 * max(0.0, 618.0 - Q.sum_pt) / 30.71843   # +3.2%  sum_pt < 618
        + 0.01314199 * max(0.0, Q.log_sum_pt - 6.83) / 0.009226693   # +1.3%  log_sum_pt > 6.83
        - 0.01102085 * max(0.0, Q.sum_pt - 987.0) / 4.472128   # -1.1%  sum_pt > 987
        - 0.01069763 * max(0.0, 0.00765 - Q.girth) / 0.0002291358   # -1.1%  girth < 0.00765
        - 0.009651918 * max(0.0, 0.00539 - Q.centroid_offset) * max(0.0, 0.0187 - Q.C3) / 4.226859e-06   # -1.0%  centroid_offset < 0.00539 and C3 < 0.0187
        + 0.007091208 * max(0.0, Q.sum_pt_top5 - 703.0) * max(0.0, 0.999 - Q.D2_b2) / 9.184491   # +0.7%  sum_pt_top5 > 703 and D2_b2 < 0.999
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 5.483;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.482819 * (-0.3483609
        + 0.2625161 * max(0.0, Q.sj2_dr - 0.18) / 0.029986   # +26.3%  sj2_dr > 0.18
        + 0.2608835 * max(0.0, Q.girth - 0.0648) / 0.0146856   # +26.1%  girth > 0.0648
        - 0.1760172 * max(0.0, Q.sj2_dr - 0.173) * max(0.0, 7.02 - Q.n_dr_0p05_0p1) / 0.1690141   # -17.6%  sj2_dr > 0.173 and n_dr_0p05_0p1 < 7.02
        + 0.109626 * max(0.0, Q.sj2_dr - 0.181) * max(0.0, 0.063 - Q.tau2) / 0.001038099   # +11.0%  sj2_dr > 0.181 and tau2 < 0.063
        - 0.06897204 * max(0.0, Q.width - 0.0136) / 0.001390299   # -6.9%  width > 0.0136
        - 0.04797986 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.00188 - Q.girth2_top2) / 1.258684e-05   # -4.8%  sj2_dr > 0.13 and girth2_top2 < 0.00188
        - 0.04551435 * max(0.0, Q.width - 0.00674) * max(0.0, Q.log_sum_pt - 6.08) / 0.0006349796   # -4.6%  width > 0.00674 and log_sum_pt > 6.08
        - 0.02849098 * max(0.0, Q.sj2_dr - 0.291) * max(0.0, Q.z_dr_0p05_0p1 - 0.0713) / 0.001259765   # -2.8%  sj2_dr > 0.291 and z_dr_0p05_0p1 > 0.0713
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 19.06;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.05645 * (-0.05877274
        + 0.200575 * max(0.0, 0.000188 - Q.e3) / 0.0001504822   # +20.1%  e3 < 0.000188
        - 0.1829136 * max(0.0, 0.00823 - Q.girth2) / 0.004091179   # -18.3%  girth2 < 0.00823
        - 0.1780774 * max(0.0, 0.00114 - Q.lam2) / 0.0009221532   # -17.8%  lam2 < 0.00114
        + 0.1701319 * max(0.0, 0.206 - Q.sj3_dr_min) / 0.163743   # +17.0%  sj3_dr_min < 0.206
        + 0.1097447 * max(0.0, 0.23 - Q.N2) / 0.05460429   # +11.0%  N2 < 0.23
        - 0.07430716 * max(0.0, 0.221 - Q.N2) * max(0.0, 0.425 - Q.planar_flow) / 0.01836616   # -7.4%  N2 < 0.221 and planar_flow < 0.425
        + 0.05943084 * max(0.0, 0.00263 - Q.girth2) / 0.0007919868   # +5.9%  girth2 < 0.00263
        + 0.0180914 * max(0.0, 0.00564 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.0116) / 1.276881e-05   # +1.8%  girth2_top2 < 0.00564 and centroid_offset > 0.0116
        - 0.006727924 * max(0.0, Q.sj3_dr_max - 0.317) * max(0.0, 1.35 - Q.D2_b2) / 0.004109307   # -0.7%  sj3_dr_max > 0.317 and D2_b2 < 1.35
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.87;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.86806 * (0.1407796
        + 0.1712548 * max(0.0, 0.0352 - Q.e2) / 0.013487   # +17.1%  e2 < 0.0352
        - 0.1425724 * max(0.0, 0.0481 - Q.dr_0) / 0.01504354   # -14.3%  dr_0 < 0.0481
        - 0.1283002 * max(0.0, 0.297 - Q.sj3_dr_max) / 0.1415608   # -12.8%  sj3_dr_max < 0.297
        - 0.1070686 * max(0.0, 721.0 - Q.sum_pt) / 72.72669   # -10.7%  sum_pt < 721
        - 0.09433148 * max(0.0, 431.0 - Q.sum_pt_top5) / 14.10179   # -9.4%  sum_pt_top5 < 431
        + 0.08464135 * max(0.0, 0.00276 - Q.girth2) / 0.0008439329   # +8.5%  girth2 < 0.00276
        + 0.06699664 * max(0.0, 0.00102 - Q.girth2) / 0.0002219888   # +6.7%  girth2 < 0.00102
        - 0.06307155 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt) / 0.004257547   # -6.3%  LHA < 0.22 and log_sum_pt < 6.8
        + 0.03187439 * max(0.0, 0.0515 - Q.z_7) * max(0.0, 0.0683 - Q.absphi_0) / 0.0004292598   # +3.2%  z_7 < 0.0515 and absphi_0 < 0.0683
        + 0.03175434 * max(0.0, 0.0031 - Q.girth2) * max(0.0, 0.0166 - Q.centroid_offset) / 8.848921e-06   # +3.2%  girth2 < 0.0031 and centroid_offset < 0.0166
        + 0.02254967 * max(0.0, 0.0295 - Q.z_7) / 0.001531694   # +2.3%  z_7 < 0.0295
        - 0.02048627 * max(0.0, 0.0415 - Q.e2) * max(0.0, Q.log_sum_pt - 6.9) / 0.0001236922   # -2.0%  e2 < 0.0415 and log_sum_pt > 6.9
        + 0.01994569 * max(0.0, 716.0 - Q.sum_pt) * max(0.0, 0.00797 - Q.zdr_0) / 0.05405758   # +2.0%  sum_pt < 716 and zdr_0 < 0.00797
        - 0.01515273 * max(0.0, Q.n_pt_above_50 - 6.29) / 0.2290413   # -1.5%  n_pt_above_50 > 6.29
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 18.8;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.79799 * (0.1223535
        - 0.3194245 * max(0.0, 0.00872 - Q.girth2) / 0.004480998   # -31.9%  girth2 < 0.00872
        + 0.1256809 * max(0.0, 0.111 - Q.tau1) / 0.05507106   # +12.6%  tau1 < 0.111
        + 0.1046315 * max(0.0, Q.centroid_offset - 0.00758) / 0.01086664   # +10.5%  centroid_offset > 0.00758
        - 0.07910373 * max(0.0, Q.centroid_offset - 0.0149) * max(0.0, 969.0 - Q.sum_pt) / 2.674444   # -7.9%  centroid_offset > 0.0149 and sum_pt < 969
        - 0.07286188 * max(0.0, 0.155 - Q.max_dr) / 0.05435146   # -7.3%  max_dr < 0.155
        + 0.06330531 * max(0.0, 0.0271 - Q.girth) / 0.00439119   # +6.3%  girth < 0.0271
        - 0.05076488 * max(0.0, 0.0151 - Q.width) * max(0.0, 0.369 - Q.planar_flow) / 0.001631244   # -5.1%  width < 0.0151 and planar_flow < 0.369
        + 0.03906448 * max(0.0, 6.36 - Q.log_sum_pt) / 0.04012753   # +3.9%  log_sum_pt < 6.36
        + 0.02471107 * max(0.0, Q.sj3_dr_max - 0.173) / 0.04599193   # +2.5%  sj3_dr_max > 0.173
        + 0.02186407 * max(0.0, Q.centroid_offset - 0.00723) * max(0.0, 0.169 - Q.sj2_dr) / 0.0003368857   # +2.2%  centroid_offset > 0.00723 and sj2_dr < 0.169
        - 0.02115663 * max(0.0, Q.sj3_dr_min - 0.0182) / 0.03059246   # -2.1%  sj3_dr_min > 0.0182
        - 0.02020431 * max(0.0, 6.36 - Q.log_sum_pt) * max(0.0, Q.pt_7 - 21.4) / 0.4777365   # -2.0%  log_sum_pt < 6.36 and pt_7 > 21.4
        + 0.01861247 * max(0.0, Q.centroid_offset - 0.013) * max(0.0, 0.0143 - Q.mean_phi2) / 6.651654e-05   # +1.9%  centroid_offset > 0.013 and mean_phi2 < 0.0143
        - 0.01281834 * max(0.0, Q.tau2 - 0.0189) / 0.005488817   # -1.3%  tau2 > 0.0189
        + 0.01257461 * max(0.0, 6.64 - Q.log_sum_pt) * max(0.0, 0.0591 - Q.z_7) / 0.0003757987   # +1.3%  log_sum_pt < 6.64 and z_7 < 0.0591
        + 0.009231099 * max(0.0, Q.centroid_offset - 0.0139) * max(0.0, 4.55 - Q.n_dr_0p05_0p1) / 0.01790775   # +0.9%  centroid_offset > 0.0139 and n_dr_0p05_0p1 < 4.55
        + 0.002753367 * max(0.0, Q.sum_pt - 1010.0) / 3.497147   # +0.3%  sum_pt > 1010
        + 0.001236891 * max(0.0, 0.0224 - Q.z_6) * max(0.0, 6.83 - Q.log_sum_pt) / 4.428773e-06   # +0.1%  z_6 < 0.0224 and log_sum_pt < 6.83
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 25.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.46571 * (0.2222597
        - 0.1542496 * max(0.0, 0.00545 - Q.width) / 0.002146489   # -15.4%  width < 0.00545
        - 0.1176442 * max(0.0, Q.girth2 - 0.00766) / 0.002435685   # -11.8%  girth2 > 0.00766
        + 0.1144084 * max(0.0, 0.00532 - Q.e2_sq) / 0.002258521   # +11.4%  e2_sq < 0.00532
        - 0.1096239 * max(0.0, 0.184 - Q.sj2_dr) / 0.06245305   # -11.0%  sj2_dr < 0.184
        + 0.1055416 * max(0.0, 0.158 - Q.sj2_dr) / 0.04782368   # +10.6%  sj2_dr < 0.158
        - 0.0914235 * max(0.0, 0.00814 - Q.e2_sq) / 0.004271861   # -9.1%  e2_sq < 0.00814
        + 0.07428714 * max(0.0, Q.girth2 - 0.013) / 0.001477949   # +7.4%  girth2 > 0.013
        - 0.06339551 * max(0.0, 0.0908 - Q.girth) / 0.03928008   # -6.3%  girth < 0.0908
        + 0.0624872 * max(0.0, 0.00112 - Q.lam2) / 0.0009041368   # +6.2%  lam2 < 0.00112
        + 0.0428356 * max(0.0, 0.0644 - Q.tau1) / 0.02263151   # +4.3%  tau1 < 0.0644
        - 0.02706362 * max(0.0, 0.238 - Q.planar_flow) / 0.1003194   # -2.7%  planar_flow < 0.238
        + 0.02574423 * max(0.0, 0.196 - Q.planar_flow) * max(0.0, Q.log_sum_pt - 6.42) / 0.01419037   # +2.6%  planar_flow < 0.196 and log_sum_pt > 6.42
        - 0.01012057 * max(0.0, 0.183 - Q.planar_flow) * max(0.0, 0.0523 - Q.z_7) / 0.0005740035   # -1.0%  planar_flow < 0.183 and z_7 < 0.0523
        - 0.001174945 * max(0.0, 0.057 - Q.tau1) * max(0.0, Q.z_dr_0p05_0p1 - 0.655) / 8.62271e-05   # -0.1%  tau1 < 0.057 and z_dr_0p05_0p1 > 0.655
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 2.102;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.102209 * (-0.1403286
        + 0.4998597 * max(0.0, 0.00708 - Q.girth2) * max(0.0, 0.0236 - Q.centroid_offset) / 4.237134e-05   # +50.0%  girth2 < 0.00708 and centroid_offset < 0.0236
        - 0.2526284 * max(0.0, 0.196 - Q.LHA) * max(0.0, 0.239 - Q.z_dr_0p2_0p4) / 0.005914004   # -25.3%  LHA < 0.196 and z_dr_0p2_0p4 < 0.239
        + 0.1170703 * max(0.0, 0.00901 - Q.girth2) * max(0.0, 0.434 - Q.planar_flow) / 0.0008486423   # +11.7%  girth2 < 0.00901 and planar_flow < 0.434
        - 0.1140687 * max(0.0, Q.log_sum_pt - 6.68) / 0.04192242   # -11.4%  log_sum_pt > 6.68
        - 0.01637285 * max(0.0, 0.104 - Q.sj3_dr_max) / 0.02294609   # -1.6%  sj3_dr_max < 0.104
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 15.4;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.39972 * (-0.2727322
        + 0.2069261 * max(0.0, 0.0067 - Q.width) / 0.002950561   # +20.7%  width < 0.0067
        + 0.1330223 * max(0.0, 6.88 - Q.log_sum_pt) / 0.3419879   # +13.3%  log_sum_pt < 6.88
        + 0.1098349 * max(0.0, 0.00382 - Q.girth2) / 0.001301098   # +11.0%  girth2 < 0.00382
        + 0.108863 * max(0.0, 0.0197 - Q.centroid_offset) * max(0.0, 0.075 - Q.C2) / 0.0004276684   # +10.9%  centroid_offset < 0.0197 and C2 < 0.075
        - 0.1073529 * max(0.0, 0.0565 - Q.tau1) / 0.01843037   # -10.7%  tau1 < 0.0565
        - 0.09926657 * max(0.0, 0.13 - Q.sj2_dr) / 0.03563352   # -9.9%  sj2_dr < 0.13
        + 0.07821941 * max(0.0, 0.0229 - Q.centroid_offset) / 0.009873422   # +7.8%  centroid_offset < 0.0229
        + 0.04559104 * max(0.0, 0.125 - Q.max_dr) / 0.03545907   # +4.6%  max_dr < 0.125
        - 0.03398947 * max(0.0, 0.0218 - Q.centroid_offset) * max(0.0, 0.387 - Q.D2_b2) / 0.001123237   # -3.4%  centroid_offset < 0.0218 and D2_b2 < 0.387
        + 0.01956449 * max(0.0, 0.0255 - Q.centroid_offset) * max(0.0, Q.tau21_b2 - 0.0724) / 0.001956414   # +2.0%  centroid_offset < 0.0255 and tau21_b2 > 0.0724
        + 0.0164655 * max(0.0, Q.e3 - 0.000242) / 3.259179e-05   # +1.6%  e3 > 0.000242
        - 0.01592459 * max(0.0, 0.0181 - Q.centroid_offset) * max(0.0, 1.47 - Q.N3) / 0.0008885303   # -1.6%  centroid_offset < 0.0181 and N3 < 1.47
        - 0.01301688 * max(0.0, 7.03 - Q.log_sum_pt) * max(0.0, Q.planar_flow - 0.116) / 0.09778362   # -1.3%  log_sum_pt < 7.03 and planar_flow > 0.116
        + 0.01196279 * max(0.0, 6.93 - Q.log_sum_pt) * max(0.0, 0.0581 - Q.z_5) / 0.0007675985   # +1.2%  log_sum_pt < 6.93 and z_5 < 0.0581
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.057;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.057446 * (0.6206211
        - 0.1906491 * max(0.0, Q.LHA - 0.184) / 0.07961515   # -19.1%  LHA > 0.184
        - 0.1739371 * max(0.0, 0.0062 - Q.lam1) / 0.002680244   # -17.4%  lam1 < 0.0062
        - 0.1422397 * max(0.0, 0.00345 - Q.lam1) / 0.00116591   # -14.2%  lam1 < 0.00345
        + 0.09617337 * max(0.0, Q.girth2 - 0.0076) / 0.002450319   # +9.6%  girth2 > 0.0076
        + 0.08774901 * max(0.0, Q.tau1 - 0.0475) / 0.0299171   # +8.8%  tau1 > 0.0475
        + 0.06984246 * max(0.0, 4.78 - Q.n_dr_0p05_0p1) / 2.98733   # +7.0%  n_dr_0p05_0p1 < 4.78
        - 0.0666951 * max(0.0, Q.sj2_dr - 0.177) * max(0.0, 0.681 - Q.planar_flow) / 0.01435052   # -6.7%  sj2_dr > 0.177 and planar_flow < 0.681
        - 0.04094126 * max(0.0, Q.LHA - 0.317) * max(0.0, 0.915 - Q.planar_flow) / 0.008116312   # -4.1%  LHA > 0.317 and planar_flow < 0.915
        + 0.04050441 * max(0.0, Q.lam2 - 0.00011) / 0.0004701607   # +4.1%  lam2 > 0.00011
        + 0.02387274 * max(0.0, Q.e3 - 7.84e-05) / 5.044329e-05   # +2.4%  e3 > 7.84e-05
        - 0.02072739 * max(0.0, Q.lam2 - 0.00326) / 0.0001643623   # -2.1%  lam2 > 0.00326
        - 0.01736168 * max(0.0, Q.lam2 - 0.000186) * max(0.0, Q.eccentricity - 0.48) / 8.69001e-05   # -1.7%  lam2 > 0.000186 and eccentricity > 0.48
        - 0.015889 * max(0.0, Q.lam2 - 5.13e-06) * max(0.0, 6.55 - Q.log_sum_pt) / 0.0001423043   # -1.6%  lam2 > 5.13e-06 and log_sum_pt < 6.55
        + 0.01341768 * max(0.0, Q.sj3_dr_min - 0.129) / 0.008093552   # +1.3%  sj3_dr_min > 0.129
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 29.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.29852 * (-0.01385736
        + 0.375904 * max(0.0, 0.00867 - Q.width) / 0.0044409   # +37.6%  width < 0.00867
        - 0.3058743 * max(0.0, 0.00816 - Q.e2_sq) / 0.004287878   # -30.6%  e2_sq < 0.00816
        - 0.08826755 * max(0.0, 0.0726 - Q.girth) / 0.02560504   # -8.8%  girth < 0.0726
        + 0.07964016 * max(0.0, 0.0127 - Q.width) / 0.007777796   # +8.0%  width < 0.0127
        + 0.04841152 * max(0.0, Q.sj2_dr - 0.0847) / 0.08442774   # +4.8%  sj2_dr > 0.0847
        - 0.03631473 * max(0.0, Q.sj2_dr - 0.173) / 0.03283851   # -3.6%  sj2_dr > 0.173
        + 0.03085676 * max(0.0, 0.262 - Q.planar_flow) / 0.1147281   # +3.1%  planar_flow < 0.262
        - 0.01094588 * max(0.0, 0.014 - Q.centroid_offset) * max(0.0, 0.0728 - Q.sj3_dr_min) / 0.000216688   # -1.1%  centroid_offset < 0.014 and sj3_dr_min < 0.0728
        - 0.006896809 * max(0.0, 0.246 - Q.planar_flow) * max(0.0, 851.0 - Q.sum_pt) / 14.85782   # -0.7%  planar_flow < 0.246 and sum_pt < 851
        + 0.006450285 * max(0.0, 0.0146 - Q.centroid_offset) * max(0.0, 0.0608 - Q.tau21_b2) / 7.745238e-05   # +0.6%  centroid_offset < 0.0146 and tau21_b2 < 0.0608
        - 0.00622373 * max(0.0, 0.249 - Q.planar_flow) * max(0.0, 37.2 - Q.pt_7) / 0.5426967   # -0.6%  planar_flow < 0.249 and pt_7 < 37.2
        - 0.004214221 * max(0.0, 40.0 - Q.pt_6) * max(0.0, Q.girth2_top3 - 0.0049) / 0.01055303   # -0.4%  pt_6 < 40 and girth2_top3 > 0.0049
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.3037;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.3036657 * (0.005960503
        + 0.5850596 * max(0.0, Q.girth2 - 0.0251) / 0.0002956115   # +58.5%  girth2 > 0.0251
        - 0.3901428 * max(0.0, Q.girth2 - 0.0239) * max(0.0, 916.0 - Q.sum_pt) / 0.1486486   # -39.0%  girth2 > 0.0239 and sum_pt < 916
        + 0.02479768 * max(0.0, Q.girth2 - 0.021) / 0.0005661807   # +2.5%  girth2 > 0.021
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.99303 * (0.08591757
        + 0.4066504 * max(0.0, 0.148 - Q.girth) / 0.0902118   # +40.7%  girth < 0.148
        + 0.114104 * max(0.0, 0.0148 - Q.lam1) / 0.009743579   # +11.4%  lam1 < 0.0148
        - 0.1063683 * max(0.0, 0.121 - Q.tau1) / 0.06364505   # -10.6%  tau1 < 0.121
        - 0.08915171 * max(0.0, 0.124 - Q.sj3_dr_min) / 0.08807895   # -8.9%  sj3_dr_min < 0.124
        - 0.07046626 * max(0.0, 7.94e-05 - Q.e3) * max(0.0, 0.0419 - Q.centroid_offset) / 1.573503e-06   # -7.0%  e3 < 7.94e-05 and centroid_offset < 0.0419
        - 0.06530972 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt) / 0.0200689   # -6.5%  girth < 0.15 and log_sum_pt < 6.8
        - 0.03810973 * max(0.0, 0.15 - Q.girth) * max(0.0, 38.4 - Q.pt_7) / 0.6731807   # -3.8%  girth < 0.15 and pt_7 < 38.4
        + 0.02598596 * max(0.0, Q.sum_pt_top5 - 687.0) * max(0.0, 40.4 - Q.pt_7) / 666.0336   # +2.6%  sum_pt_top5 > 687 and pt_7 < 40.4
        - 0.02097886 * max(0.0, Q.sj3_dr_max - 0.233) / 0.02528329   # -2.1%  sj3_dr_max > 0.233
        - 0.01837502 * max(0.0, 0.0178 - Q.lam1) * max(0.0, 25.6 - Q.pt_7) / 0.02054259   # -1.8%  lam1 < 0.0178 and pt_7 < 25.6
        + 0.01757119 * max(0.0, Q.z_7 - 0.0583) / 0.00599574   # +1.8%  z_7 > 0.0583
        - 0.0156721 * max(0.0, Q.pt_7 - 32.1) / 5.802101   # -1.6%  pt_7 > 32.1
        - 0.009401871 * max(0.0, 6.47 - Q.log_sum_pt) * max(0.0, 0.0667 - Q.C3) / 0.002096671   # -0.9%  log_sum_pt < 6.47 and C3 < 0.0667
        - 0.001854935 * max(0.0, Q.sum_pt - 994.0) * max(0.0, Q.sj3_pairmin_over_m - 0.225) / 0.4152961   # -0.2%  sum_pt > 994 and sj3_pairmin_over_m > 0.225
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 28.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.95198 * (0.002272729
        + 0.1751434 * max(0.0, Q.girth - 0.04) / 0.02755841   # +17.5%  girth > 0.04
        - 0.1563224 * max(0.0, 0.231 - Q.sj3_dr_max) / 0.08856834   # -15.6%  sj3_dr_max < 0.231
        - 0.1137651 * max(0.0, Q.girth2 - 0.00676) / 0.002677825   # -11.4%  girth2 > 0.00676
        + 0.1118125 * max(0.0, 0.159 - Q.sj3_dr_max) / 0.04404345   # +11.2%  sj3_dr_max < 0.159
        + 0.1040001 * max(0.0, Q.e2_sq - 0.0064) / 0.002447974   # +10.4%  e2_sq > 0.0064
        - 0.08634225 * max(0.0, Q.tau1 - 0.0356) / 0.03665365   # -8.6%  tau1 > 0.0356
        - 0.04751196 * max(0.0, Q.girth2 - 0.00864) * max(0.0, Q.log_sum_pt - 6.25) / 0.0002343382   # -4.8%  girth2 > 0.00864 and log_sum_pt > 6.25
        - 0.04409748 * max(0.0, Q.girth - 0.0883) / 0.007644967   # -4.4%  girth > 0.0883
        - 0.04076476 * max(0.0, Q.girth - 0.103) / 0.005245425   # -4.1%  girth > 0.103
        - 0.02929094 * max(0.0, Q.e2_sq - 0.0174) / 0.000737418   # -2.9%  e2_sq > 0.0174
        + 0.02263174 * max(0.0, Q.width - 0.0129) * max(0.0, Q.sum_pt - 483.0) / 0.1172153   # +2.3%  width > 0.0129 and sum_pt > 483
        + 0.01483786 * max(0.0, Q.e2_sq - 0.00234) * max(0.0, Q.log_sum_pt - 6.22) / 0.0008423242   # +1.5%  e2_sq > 0.00234 and log_sum_pt > 6.22
        - 0.01400794 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.00755 - Q.lam1) / 6.781899e-05   # -1.4%  planar_flow < 0.115 and lam1 < 0.00755
        + 0.01230333 * max(0.0, 0.119 - Q.planar_flow) * max(0.0, 0.0159 - Q.lam1) / 0.0003360432   # +1.2%  planar_flow < 0.119 and lam1 < 0.0159
        - 0.009410052 * max(0.0, Q.centroid_offset - 0.0234) / 0.004030172   # -0.9%  centroid_offset > 0.0234
        + 0.005664531 * max(0.0, 0.104 - Q.planar_flow) * max(0.0, 0.00596 - Q.width) / 2.737886e-05   # +0.6%  planar_flow < 0.104 and width < 0.00596
        - 0.005291593 * max(0.0, Q.centroid_offset - 0.0504) / 0.0007816434   # -0.5%  centroid_offset > 0.0504
        - 0.004067174 * max(0.0, 0.111 - Q.planar_flow) * max(0.0, 0.0181 - Q.centroid_offset) / 0.0002129344   # -0.4%  planar_flow < 0.111 and centroid_offset < 0.0181
        - 0.002734923 * max(0.0, Q.psi_0p1 - 0.819) * max(0.0, Q.centroid_offset - 0.0379) / 4.741404e-05   # -0.3%  psi_0p1 > 0.819 and centroid_offset > 0.0379
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 13.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.43395 * (0.03424161
        - 0.287403 * max(0.0, 0.00756 - Q.girth2) / 0.003574959   # -28.7%  girth2 < 0.00756
        + 0.2519963 * max(0.0, 0.0908 - Q.tau1) / 0.03922716   # +25.2%  tau1 < 0.0908
        - 0.109246 * max(0.0, 0.217 - Q.N2) * max(0.0, Q.LHA - 0.386) / 0.0002164609   # -10.9%  N2 < 0.217 and LHA > 0.386
        - 0.0706058 * max(0.0, 0.247 - Q.N2) * max(0.0, Q.girth2 - 0.00871) / 0.0001500814   # -7.1%  N2 < 0.247 and girth2 > 0.00871
        + 0.07027825 * max(0.0, 0.195 - Q.planar_flow) * max(0.0, Q.sum_pt_top5 - 401.0) / 15.70906   # +7.0%  planar_flow < 0.195 and sum_pt_top5 > 401
        + 0.04028341 * max(0.0, 0.238 - Q.N2) * max(0.0, Q.LHA - 0.289) / 0.001898825   # +4.0%  N2 < 0.238 and LHA > 0.289
        - 0.03560252 * max(0.0, 0.00231 - Q.girth2_top3) / 0.0008586758   # -3.6%  girth2_top3 < 0.00231
        - 0.03160118 * max(0.0, 0.216 - Q.planar_flow) / 0.08753167   # -3.2%  planar_flow < 0.216
        - 0.02608395 * max(0.0, 0.182 - Q.planar_flow) * max(0.0, 0.0723 - Q.z_7) / 0.001516928   # -2.6%  planar_flow < 0.182 and z_7 < 0.0723
        + 0.02323417 * max(0.0, 0.137 - Q.z_dr_0_0p05) / 0.04275706   # +2.3%  z_dr_0_0p05 < 0.137
        - 0.02320827 * max(0.0, 0.219 - Q.N2) * max(0.0, 0.19 - Q.sj2_dr) / 0.0008204703   # -2.3%  N2 < 0.219 and sj2_dr < 0.19
        - 0.01302772 * max(0.0, 0.00214 - Q.girth2_top3) * max(0.0, Q.eccentricity - 0.959) / 4.522317e-06   # -1.3%  girth2_top3 < 0.00214 and eccentricity > 0.959
        - 0.009395092 * max(0.0, 0.00668 - Q.girth2) * max(0.0, Q.dr12 - 0.156) / 1.605765e-06   # -0.9%  girth2 < 0.00668 and dr12 > 0.156
        + 0.008034403 * max(0.0, 0.00638 - Q.e2_sq) * max(0.0, Q.dr12 - 0.157) / 1.439116e-06   # +0.8%  e2_sq < 0.00638 and dr12 > 0.157
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9806960084033614, 0.784622268907563, 2.0757660714285713, 0.8363254201680672, 1.351502100840336, 1.8441941176470589, 1.2205987394957982, 1.96405, 0.36793487394957985, 3.2236415966386556, 2.1311274159663864, 2.327548739495798, 0.06870168067226891, 3.7578556722689074, 0.5365572478991597, 0.5209392857142857]
T = [2.4272447831867123, 1.5061730747767856, 3.681210669971114, 2.6701376953125, 3.126720282956933]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +37%, n9 +23%, n5 -14%, n1 +13%, n0 -6%, n6 +6% ...
            + 0.3674663 * h[2] / H_AVG[2]
            + 0.2282684 * h[9] / H_AVG[9]
            - 0.1424605 * h[5] / H_AVG[5]
            + 0.126272 * h[1] / H_AVG[1]
            - 0.06313074 * h[0] / H_AVG[0]
            + 0.05500186 * h[6] / H_AVG[6]
            - 0.01740016 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +54%, n10 -18%, n6 +10%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5434321 * h[9] / H_AVG[9]
            - 0.1768661 * h[10] / H_AVG[10]
            + 0.1012997 * h[6] / H_AVG[6]
            - 0.08412268 * h[4] / H_AVG[4]
            + 0.05739486 * h[5] / H_AVG[5]
            + 0.02161684 * h[15] / H_AVG[15]
            + 0.01526779 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n7 +12%, n3 -11%, n14 -11%, n6 -10%, n15 -10% ...
            + 0.2371043 * h[11] / H_AVG[11]
            + 0.1167105 * h[7] / H_AVG[7]
            - 0.1135938 * h[3] / H_AVG[3]
            - 0.1093167 * h[14] / H_AVG[14]
            - 0.1036173 * h[6] / H_AVG[6]
            - 0.09729021 * h[15] / H_AVG[15]
            + 0.09157701 * h[0] / H_AVG[0]
            + 0.07177645 * h[13] / H_AVG[13]
            - 0.02736567 * h[9] / H_AVG[9]
            - 0.02498736 * h[8] / H_AVG[8]
            - 0.006660702 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n3 -18%, n6 -17%, n13 +8%, n14 +8%, n4 +4% ...
            + 0.3447944 * h[7] / H_AVG[7]
            - 0.1761831 * h[3] / H_AVG[3]
            - 0.1714236 * h[6] / H_AVG[6]
            + 0.07696522 * h[13] / H_AVG[13]
            + 0.07535528 * h[14] / H_AVG[14]
            + 0.03954332 * h[4] / H_AVG[4]
            - 0.03772794 * h[9] / H_AVG[9]
            + 0.03673136 * h[1] / H_AVG[1]
            - 0.03048411 * h[15] / H_AVG[15]
            + 0.01079178 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -49%, n10 +26%, n5 -15%, n4 +5%, n8 +2%, n3 +2% ...
            - 0.4882525 * h[13] / H_AVG[13]
            + 0.2555946 * h[10] / H_AVG[10]
            - 0.1474544 * h[5] / H_AVG[5]
            + 0.05403034 * h[4] / H_AVG[4]
            + 0.02206395 * h[8] / H_AVG[8]
            + 0.01671731 * h[3] / H_AVG[3]
            - 0.01098622 * h[12] / H_AVG[12]
            + 0.004900782 * h[0] / H_AVG[0]
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
