"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.7%   (on for 67% of jets)
  neuron  1:  11.8%   (on for 95% of jets)
  neuron  5:  11.0%   (on for 76% of jets)
  neuron  4:   9.5%   (on for 65% of jets)
  neuron  0:   7.3%   (on for 52% of jets)
  neuron 13:   6.8%   (on for 89% of jets)
  neuron 10:   6.6%   (on for 76% of jets)
  neuron  9:   6.5%   (on for 55% of jets)
  neuron  7:   5.4%   (on for 49% of jets)
  neuron  6:   4.7%   (on for 71% of jets)
  neuron  3:   4.1%   (on for 90% of jets)
  neuron 12:   3.9%   (on for 43% of jets)
  neuron 11:   3.2%   (on for 88% of jets)
  neuron 15:   2.8%   (on for 77% of jets)
  neuron 14:   2.3%   (on for 54% of jets)
  neuron  2:   0.6%   (on for 75% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.5% (the network: 81.1%); same class as the network for 91.7% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top20=mass_of(20),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
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
    # scale S = 10.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.23862 * (0.1474808
        - 0.3372028 * max(0.0, Q.mass - 78.3) / 21.31167   # -33.7%  mass > 78.3
        - 0.2943871 * max(0.0, Q.mass - 87.4) / 16.56109   # -29.4%  mass > 87.4
        + 0.1399847 * max(0.0, 0.00777 - Q.mass_over_sum_pt_sq) / 0.002174887   # +14.0%  mass_over_sum_pt_sq < 0.00777
        - 0.08022594 * max(0.0, 0.00619 - Q.sum_zz_dr2) / 0.001305887   # -8.0%  sum_zz_dr2 < 0.00619
        - 0.03268995 * max(0.0, 0.0061 - Q.sum_z_dr2_top20) / 0.001828962   # -3.3%  sum_z_dr2_top20 < 0.0061
        - 0.03059279 * max(0.0, 0.0912 - Q.z_dr_0p2_0p4) / 0.05822081   # -3.1%  z_dr_0p2_0p4 < 0.0912
        + 0.03044031 * max(0.0, Q.psi_0p3 - 0.996) / 0.001712455   # +3.0%  psi_0p3 > 0.996
        - 0.0231367 * max(0.0, 0.0693 - Q.tau1) / 0.01294469   # -2.3%  tau1 < 0.0693
        - 0.01826071 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -1.8%  sum_pt < 1010
        - 0.01307894 * max(0.0, 0.992 - Q.z_top50_slots) / 0.005130664   # -1.3%  z_top50_slots < 0.992
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 11.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.62646 * (0.3061981
        + 0.1930666 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # +19.3%  log_sum_pt > 6.9
        - 0.1469836 * max(0.0, 7.14 - Q.log_sum_pt) / 0.2008108   # -14.7%  log_sum_pt < 7.14
        + 0.1184878 * max(0.0, Q.n_particles - 37.8) / 10.93328   # +11.8%  n_particles > 37.8
        - 0.1065393 * max(0.0, Q.sum_pt_top50 - 959.0) / 86.62065   # -10.7%  sum_pt_top50 > 959
        + 0.08946622 * max(0.0, 701.0 - Q.sum_pt_top2) / 334.4616   # +8.9%  sum_pt_top2 < 701
        - 0.05660309 * max(0.0, 33.9 - Q.sj3_mass1) / 18.85655   # -5.7%  sj3_mass1 < 33.9
        - 0.04301788 * max(0.0, 33.4 - Q.sj3_mass1) * max(0.0, 18.4 - Q.sj3_mass2) / 194.6092   # -4.3%  sj3_mass1 < 33.4 and sj3_mass2 < 18.4
        - 0.041248 * max(0.0, Q.log_sum_pt - 6.99) / 0.0210337   # -4.1%  log_sum_pt > 6.99
        - 0.04085039 * max(0.0, Q.n_particles - 38.3) * max(0.0, 2.31 - Q.soft1_pt) / 15.07763   # -4.1%  n_particles > 38.3 and soft1_pt < 2.31
        - 0.0378458 * max(0.0, Q.zdr_0 - 0.00156) / 0.008270915   # -3.8%  zdr_0 > 0.00156
        - 0.02924196 * max(0.0, 0.0323 - Q.M3) / 0.009470209   # -2.9%  M3 < 0.0323
        - 0.02688072 * max(0.0, 703.0 - Q.sum_pt_top2) * max(0.0, 0.965 - Q.tau43) / 41.44929   # -2.7%  sum_pt_top2 < 703 and tau43 < 0.965
        + 0.02515632 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # +2.5%  sum_pt < 1010
        + 0.01757584 * max(0.0, 56.7 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.2) / 58.38423   # +1.8%  mass_top20 < 56.7 and n_real_top40 > 29.2
        + 0.01438787 * max(0.0, Q.z_top30_slots - 0.944) * max(0.0, Q.max_pair_mass - 7.47) / 0.2052515   # +1.4%  z_top30_slots > 0.944 and max_pair_mass > 7.47
        + 0.008385896 * max(0.0, 0.000745 - Q.sum_z_dr2_top15) / 8.193134e-05   # +0.8%  sum_z_dr2_top15 < 0.000745
        + 0.004262701 * max(0.0, Q.sum_pt_top30 - 1180.0) / 7.554898   # +0.4%  sum_pt_top30 > 1180
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 1.041;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.040979 * (1.171973
        - 1.0 * max(0.0, 1270.0 - Q.sum_pt) / 233.4034   # -100.0%  sum_pt < 1270
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 0.7927;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.7927113 * (0.1041993
        + 0.5183437 * max(0.0, 0.0155 - Q.tau4) / 0.002552155   # +51.8%  tau4 < 0.0155
        - 0.3263603 * max(0.0, 0.317 - Q.tau21) * max(0.0, Q.lam1 - 0.00429) / 0.0001796594   # -32.6%  tau21 < 0.317 and lam1 > 0.00429
        + 0.1552961 * max(0.0, 4.64 - Q.n_dr_0p2_0p4) / 0.9693304   # +15.5%  n_dr_0p2_0p4 < 4.64
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 4.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.680085 * (0.2734993
        - 0.3087261 * max(0.0, 80.7 - Q.mass) * max(0.0, 0.0041 - Q.lam2) / 0.03969407   # -30.9%  mass < 80.7 and lam2 < 0.0041
        + 0.1862403 * max(0.0, 104.0 - Q.mass) * max(0.0, 0.00334 - Q.lam2) / 0.07263504   # +18.6%  mass < 104 and lam2 < 0.00334
        - 0.1727108 * max(0.0, Q.n_particles - 22.5) / 23.49713   # -17.3%  n_particles > 22.5
        + 0.08366997 * max(0.0, 69.8 - Q.mass_top40) / 8.331544   # +8.4%  mass_top40 < 69.8
        + 0.06283295 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.22 - Q.soft1_pt) / 35.25942   # +6.3%  n_particles > 22 and soft1_pt < 2.22
        - 0.05993868 * max(0.0, 91.8 - Q.mass_top40) * max(0.0, 9.1 - Q.D2) / 82.99352   # -6.0%  mass_top40 < 91.8 and D2 < 9.1
        - 0.0588409 * max(0.0, 0.00523 - Q.sum_z_dr2_top15) / 0.001591794   # -5.9%  sum_z_dr2_top15 < 0.00523
        + 0.02881129 * max(0.0, Q.sj3_pair_mass_min - 33.4) / 5.665515   # +2.9%  sj3_pair_mass_min > 33.4
        + 0.02204032 * max(0.0, 0.324 - Q.planar_flow) / 0.05400554   # +2.2%  planar_flow < 0.324
        - 0.01618872 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0309 - Q.C2_b2) / 0.0002621611   # -1.6%  sj2_dr > 0.225 and C2_b2 < 0.0309
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 9.488;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.488299 * (0.2455656
        + 0.3310456 * max(0.0, Q.sum_pt_top50 - 917.0) / 124.1525   # +33.1%  sum_pt_top50 > 917
        - 0.1790446 * max(0.0, Q.log_sum_pt - 6.92) / 0.04530211   # -17.9%  log_sum_pt > 6.92
        - 0.1693271 * max(0.0, 159.0 - Q.mass_top40) / 75.07597   # -16.9%  mass_top40 < 159
        - 0.06832837 * max(0.0, Q.mass_top50 - 157.0) / 1.612736   # -6.8%  mass_top50 > 157
        - 0.0629883 * max(0.0, Q.z_top30_slots - 0.895) / 0.06426364   # -6.3%  z_top30_slots > 0.895
        - 0.04488122 * max(0.0, Q.sum_z_dr2_top30 - 0.00764) / 0.003301135   # -4.5%  sum_z_dr2_top30 > 0.00764
        - 0.04087363 * max(0.0, Q.max_dr - 0.267) / 0.09436038   # -4.1%  max_dr > 0.267
        + 0.03950343 * max(0.0, 54.1 - Q.n_particles) / 11.08936   # +4.0%  n_particles < 54.1
        - 0.03928652 * max(0.0, 0.469 - Q.tau21) / 0.1106119   # -3.9%  tau21 < 0.469
        + 0.02027863 * max(0.0, 70.3 - Q.mass) / 7.45774   # +2.0%  mass < 70.3
        + 0.004442641 * max(0.0, Q.max_dr - 0.435) / 0.007748733   # +0.4%  max_dr > 0.435
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 3.546;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.546449 * (-0.1291433
        + 0.4621972 * max(0.0, Q.mass - 83.8) / 18.25344   # +46.2%  mass > 83.8
        - 0.3044427 * max(0.0, Q.mass - 106.0) / 11.11937   # -30.4%  mass > 106
        + 0.1849173 * max(0.0, 93.8 - Q.mass) / 18.2674   # +18.5%  mass < 93.8
        - 0.04844276 * max(0.0, 2.23 - Q.D2) / 0.4908565   # -4.8%  D2 < 2.23
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 16.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.23222 * (0.01078102
        - 0.3826606 * max(0.0, 92.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.965) / 0.5308917   # -38.3%  mass < 92 and psi_0p3 > 0.965
        + 0.177763 * max(0.0, 104.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.966) / 0.7756689   # +17.8%  mass < 104 and psi_0p3 > 0.966
        + 0.1732125 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997) / 0.03127501   # +17.3%  mass < 101 and psi_0p3 > 0.997
        - 0.09984967 * max(0.0, 90.9 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997) / 0.02064691   # -10.0%  mass < 90.9 and psi_0p3 > 0.997
        - 0.06081172 * max(0.0, 82.8 - Q.mass) * max(0.0, Q.psi_0p3 - 0.995) / 0.03143661   # -6.1%  mass < 82.8 and psi_0p3 > 0.995
        - 0.02464943 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # -2.5%  psi_0p3 > 0.998
        + 0.02342843 * max(0.0, 0.249 - Q.tau21_b2) / 0.05859716   # +2.3%  tau21_b2 < 0.249
        - 0.02022483 * max(0.0, 82.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3) / 0.04008473   # -2.0%  mass < 82 and psi_0p3 < 1
        - 0.01638084 * max(0.0, 0.379 - Q.tau21) / 0.06581622   # -1.6%  tau21 < 0.379
        - 0.01322695 * max(0.0, Q.z_top20_slots - 0.918) / 0.02313608   # -1.3%  z_top20_slots > 0.918
        + 0.007792069 * max(0.0, 121.0 - Q.mass) * max(0.0, Q.sd_mass - 75.4) / 38.21227   # +0.8%  mass < 121 and sd_mass > 75.4
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 5.245;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.244974 * (0.3965701
        + 0.3253906 * max(0.0, Q.mass_over_sum_pt - 0.0766) / 0.02046362   # +32.5%  mass_over_sum_pt > 0.0766
        - 0.2067388 * max(0.0, 16.9 - Q.n_dr_0p2_0p4) / 9.036166   # -20.7%  n_dr_0p2_0p4 < 16.9
        - 0.2054434 * max(0.0, Q.sum_pt_top40 - 887.0) / 140.3054   # -20.5%  sum_pt_top40 > 887
        - 0.1744764 * max(0.0, Q.mass - 106.0) / 11.11937   # -17.4%  mass > 106
        + 0.05337784 * max(0.0, Q.log_sum_pt - 6.96) / 0.0282508   # +5.3%  log_sum_pt > 6.96
        + 0.0345729 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.sj3_pair_mass_max - 37.0) / 2.398598   # +3.5%  log_sum_pt > 6.9 and sj3_pair_mass_max > 37
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 3.243;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.242504 * (0.05366224
        + 0.4901653 * max(0.0, 81.7 - Q.mass) / 11.27208   # +49.0%  mass < 81.7
        - 0.1649403 * max(0.0, Q.z_top40_slots - 0.961) / 0.0254676   # -16.5%  z_top40_slots > 0.961
        - 0.1446399 * max(0.0, 64.1 - Q.mass) / 5.84054   # -14.5%  mass < 64.1
        + 0.0525544 * max(0.0, Q.mass - 110.0) / 10.20406   # +5.3%  mass > 110
        - 0.0456011 * max(0.0, 95.5 - Q.mass_top40) * max(0.0, 1050.0 - Q.sum_pt) / 924.1359   # -4.6%  mass_top40 < 95.5 and sum_pt < 1050
        + 0.03758573 * max(0.0, 960.0 - Q.sum_pt_top50) / 10.07206   # +3.8%  sum_pt_top50 < 960
        - 0.03682612 * max(0.0, Q.e3 - 0.000341) / 2.1323e-05   # -3.7%  e3 > 0.000341
        - 0.0230945 * max(0.0, Q.mass - 168.0) / 1.038613   # -2.3%  mass > 168
        + 0.004592619 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, Q.soft5_pt - 2.98) / 3.520469   # +0.5%  sum_pt_top40 < 1010 and soft5_pt > 2.98
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.192;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.191771 * (0.1946669
        - 0.3744824 * max(0.0, 82.7 - Q.sj2_mass1) / 52.70434   # -37.4%  sj2_mass1 < 82.7
        + 0.1185359 * max(0.0, 880.0 - Q.sum_pt_top5) / 291.9463   # +11.9%  sum_pt_top5 < 880
        + 0.1095916 * max(0.0, 3.84 - Q.D2) / 1.539371   # +11.0%  D2 < 3.84
        + 0.1054158 * max(0.0, 0.604 - Q.tau21_b2) / 0.3020425   # +10.5%  tau21_b2 < 0.604
        + 0.08589375 * max(0.0, 80.8 - Q.mass) / 10.85638   # +8.6%  mass < 80.8
        - 0.07223623 * max(0.0, Q.z_dr_0_0p05 - 0.768) / 0.05200264   # -7.2%  z_dr_0_0p05 > 0.768
        + 0.03152309 * max(0.0, Q.mass_top50 - 156.0) / 1.717476   # +3.2%  mass_top50 > 156
        - 0.03063059 * max(0.0, Q.mass - 172.0) / 0.7702383   # -3.1%  mass > 172
        - 0.02773432 * max(0.0, Q.mass - 149.0) / 3.196456   # -2.8%  mass > 149
        + 0.0199031 * max(0.0, 81.5 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 11.9) / 149.8833   # +2.0%  sj2_mass1 < 81.5 and sj2_mass2 > 11.9
        - 0.0106981 * max(0.0, 3.83 - Q.D2) * max(0.0, Q.sj2_dr - 0.196) / 0.04250735   # -1.1%  D2 < 3.83 and sj2_dr > 0.196
        - 0.007294912 * max(0.0, 980.0 - Q.sum_pt) / 10.86197   # -0.7%  sum_pt < 980
        + 0.006060063 * max(0.0, Q.mass - 173.0) * max(0.0, Q.M2 - 0.0406) / 0.01632306   # +0.6%  mass > 173 and M2 > 0.0406
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 1.412;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.412482 * (0.06272648
        + 0.2811783 * max(0.0, 10.3 - Q.n_dr_0p2_0p4) / 4.015766   # +28.1%  n_dr_0p2_0p4 < 10.3
        - 0.2341006 * max(0.0, 78.5 - Q.mass) * max(0.0, 3.87 - Q.D2) / 4.670379   # -23.4%  mass < 78.5 and D2 < 3.87
        + 0.2011882 * max(0.0, 15.7 - Q.n_dr_0p1_0p2) / 5.638385   # +20.1%  n_dr_0p1_0p2 < 15.7
        + 0.1715359 * max(0.0, Q.psi_0p2 - 0.992) / 0.002036062   # +17.2%  psi_0p2 > 0.992
        - 0.111997 * max(0.0, 79.4 - Q.mass) / 10.27231   # -11.2%  mass < 79.4
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.476;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.476436 * (-0.08601796
        + 0.6494852 * max(0.0, 82.6 - Q.mass) / 11.70847   # +64.9%  mass < 82.6
        - 0.190434 * max(0.0, 0.00289 - Q.sum_zz_dr2) / 0.0003231765   # -19.0%  sum_zz_dr2 < 0.00289
        - 0.1405676 * max(0.0, 83.6 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3) / 0.03419095   # -14.1%  mass < 83.6 and psi_0p3 < 0.999
        + 0.01951322 * max(0.0, Q.sj3_pair_mass_max - 126.0) * max(0.0, 52.1 - Q.sj2_mass1) / 16.65319   # +2.0%  sj3_pair_mass_max > 126 and sj2_mass1 < 52.1
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 2.937;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.936782 * (0.7967905
        - 0.2810055 * max(0.0, Q.mass - 143.0) / 4.06528   # -28.1%  mass > 143
        - 0.2183735 * max(0.0, 1100.0 - Q.sum_pt) / 79.17474   # -21.8%  sum_pt < 1100
        + 0.1799926 * max(0.0, Q.mass_top50 - 141.0) / 3.620542   # +18.0%  mass_top50 > 141
        - 0.1563211 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -15.6%  sum_pt < 1010
        + 0.1208365 * max(0.0, 1020.0 - Q.sum_pt_top40) * max(0.0, 0.65 - Q.D3) / 19.18218   # +12.1%  sum_pt_top40 < 1020 and D3 < 0.65
        + 0.02567087 * max(0.0, 6.82 - Q.log_sum_pt) / 0.005346791   # +2.6%  log_sum_pt < 6.82
        + 0.0178 * max(0.0, Q.mass - 174.0) / 0.6753841   # +1.8%  mass > 174
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 2.512;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.511903 * (0.2595642
        - 0.6999769 * max(0.0, 83.6 - Q.mass) / 12.21024   # -70.0%  mass < 83.6
        + 0.1258615 * max(0.0, 0.303 - Q.tau21_b2) / 0.08614489   # +12.6%  tau21_b2 < 0.303
        - 0.09649897 * max(0.0, 0.42 - Q.N2) * max(0.0, Q.max_dr - 0.25) / 0.01170995   # -9.6%  N2 < 0.42 and max_dr > 0.25
        - 0.07766268 * max(0.0, 0.29 - Q.tau21_b2) * max(0.0, 0.00776 - Q.sum_z_dr2_top40) / 7.65024e-05   # -7.8%  tau21_b2 < 0.29 and sum_z_dr2_top40 < 0.00776
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 1.699;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.698755 * (-0.2613678
        + 0.677853 * max(0.0, 94.5 - Q.sd_mass) / 42.49101   # +67.8%  sd_mass < 94.5
        - 0.1362869 * max(0.0, 33.4 - Q.sd_mass) / 8.703689   # -13.6%  sd_mass < 33.4
        - 0.1005123 * max(0.0, 0.0618 - Q.tau1) / 0.0102243   # -10.1%  tau1 < 0.0618
        + 0.0853478 * max(0.0, Q.sj2_dr - 0.202) * max(0.0, 0.0389 - Q.C2_b2) / 0.0005641441   # +8.5%  sj2_dr > 0.202 and C2_b2 < 0.0389
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6457764180672269, 2.8994693802521008, 0.24328067226890757, 0.5015945903361344, 0.771247268907563, 1.078979831932773, 0.6671757352941177, 0.5934712184873949, 1.323399474789916, 1.0009383403361345, 1.2965266806722688, 0.7528401260504202, 0.4682689075630252, 1.4998956932773109, 0.3416767331932773, 0.5029388655462185]
T = [3.9921068917410714, 2.4925971392463238, 4.583568822216386, 4.794503033088236, 4.17692290654543]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -17%, n9 +13%, n3 -9%, n5 -8%, n12 +3% ...
            + 0.4539378 * h[1] / H_AVG[1]
            - 0.1690439 * h[4] / H_AVG[4]
            + 0.1292823 * h[9] / H_AVG[9]
            - 0.09423494 * h[3] / H_AVG[3]
            - 0.08446197 * h[5] / H_AVG[5]
            + 0.02749188 * h[12] / H_AVG[12]
            + 0.02611308 * h[6] / H_AVG[6]
            + 0.0103595 * h[8] / H_AVG[8]
            - 0.005074571 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -22%, n11 -8%, n12 +6%, n6 +6% ...
            - 0.3287536 * h[4] / H_AVG[4]
            + 0.22588 * h[9] / H_AVG[9]
            - 0.218106 * h[1] / H_AVG[1]
            - 0.0755076 * h[11] / H_AVG[11]
            + 0.0645782 * h[12] / H_AVG[12]
            + 0.05855126 * h[6] / H_AVG[6]
            + 0.01220016 * h[2] / H_AVG[2]
            + 0.008295812 * h[8] / H_AVG[8]
            - 0.008127358 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +14%, n0 +11%, n14 -10%, n11 +10%, n7 -8% ...
            - 0.252636 * h[8] / H_AVG[8]
            + 0.1360916 * h[5] / H_AVG[5]
            + 0.1056671 * h[0] / H_AVG[0]
            - 0.1024978 * h[14] / H_AVG[14]
            + 0.09752201 * h[11] / H_AVG[11]
            - 0.08092374 * h[7] / H_AVG[7]
            + 0.05784057 * h[4] / H_AVG[4]
            + 0.04787702 * h[3] / H_AVG[3]
            - 0.0477696 * h[9] / H_AVG[9]
            - 0.04150352 * h[12] / H_AVG[12]
            - 0.02057371 * h[15] / H_AVG[15]
            + 0.009097384 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -19%, n5 +15%, n6 -13%, n7 +11%, n15 +6% ...
            - 0.2587728 * h[8] / H_AVG[8]
            - 0.1852001 * h[0] / H_AVG[0]
            + 0.1547186 * h[5] / H_AVG[5]
            - 0.1348057 * h[6] / H_AVG[6]
            + 0.1121771 * h[7] / H_AVG[7]
            + 0.05900572 * h[15] / H_AVG[15]
            + 0.04577067 * h[3] / H_AVG[3]
            + 0.03052121 * h[12] / H_AVG[12]
            - 0.01902809 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -33%, n10 +31%, n5 -12%, n8 +7%, n15 -5%, n12 -4% ...
            - 0.3254263 * h[13] / H_AVG[13]
            + 0.3055523 * h[10] / H_AVG[10]
            - 0.1210872 * h[5] / H_AVG[5]
            + 0.06683259 * h[8] / H_AVG[8]
            - 0.04515335 * h[15] / H_AVG[15]
            - 0.04204072 * h[12] / H_AVG[12]
            - 0.03996094 * h[7] / H_AVG[7]
            + 0.03462091 * h[4] / H_AVG[4]
            + 0.01932572 * h[0] / H_AVG[0]
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
