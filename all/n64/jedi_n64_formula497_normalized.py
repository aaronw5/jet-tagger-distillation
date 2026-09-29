"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.7%   (on for 53% of jets)
  neuron  1:  11.8%   (on for 95% of jets)
  neuron  5:  11.1%   (on for 79% of jets)
  neuron  4:  10.0%   (on for 68% of jets)
  neuron  0:   8.1%   (on for 64% of jets)
  neuron 13:   7.2%   (on for 89% of jets)
  neuron  9:   6.8%   (on for 67% of jets)
  neuron 10:   6.7%   (on for 80% of jets)
  neuron 12:   4.9%   (on for 53% of jets)
  neuron  7:   4.6%   (on for 37% of jets)
  neuron  6:   4.3%   (on for 65% of jets)
  neuron 11:   3.4%   (on for 69% of jets)
  neuron  3:   3.3%   (on for 50% of jets)
  neuron 15:   2.6%   (on for 68% of jets)
  neuron 14:   2.0%   (on for 36% of jets)
  neuron  2:   0.6%   (on for 38% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.9% (the network: 81.1%); same class as the network for 93.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        e3=ecf('e3'),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_4=z[4] * dr[4],
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 24.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.57051 * (-0.4965301
        + 0.5048233 * Q.z_top50_slots / 0.9923015   # +50.5%  z_top50_slots
        + 0.1049384 * max(0.0, Q.mass - 89.8) / 15.53248   # +10.5%  mass > 89.8
        - 0.09764529 * max(0.0, Q.mass - 79.0) / 20.86256   # -9.8%  mass > 79
        + 0.07148807 * max(0.0, 0.00932 - Q.mass_over_sum_pt_sq) / 0.003283175   # +7.1%  mass_over_sum_pt_sq < 0.00932
        - 0.03710784 * max(0.0, Q.mass_top50 - 82.4) / 17.53382   # -3.7%  mass_top50 > 82.4
        + 0.02636536 * max(0.0, 7.03 - Q.log_sum_pt) / 0.09997075   # +2.6%  log_sum_pt < 7.03
        - 0.01643832 * max(0.0, Q.girth2_top20 - 0.000725) / 0.00716131   # -1.6%  girth2_top20 > 0.000725
        - 0.01613708 * max(0.0, 0.00829 - Q.lam1) / 0.002981176   # -1.6%  lam1 < 0.00829
        + 0.01540527 * max(0.0, Q.mass_over_sum_pt - 0.109) / 0.009656002   # +1.5%  mass_over_sum_pt > 0.109
        - 0.01519834 * max(0.0, 0.00607 - Q.e2_sq) / 0.001253124   # -1.5%  e2_sq < 0.00607
        + 0.01437039 * max(0.0, 14.8 - Q.n_dr_0p2_0p4) / 7.325476   # +1.4%  n_dr_0p2_0p4 < 14.8
        + 0.01248681 * max(0.0, Q.psi_0p3 - 0.988) / 0.007304939   # +1.2%  psi_0p3 > 0.988
        - 0.01148842 * max(0.0, 0.00615 - Q.girth2_top20) / 0.001857081   # -1.1%  girth2_top20 < 0.00615
        - 0.01126931 * max(0.0, 80.9 - Q.mass) / 10.90129   # -1.1%  mass < 80.9
        - 0.0112256 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -1.1%  sum_pt < 1010
        - 0.009738876 * max(0.0, 0.0734 - Q.tau1) / 0.0145908   # -1.0%  tau1 < 0.0734
        - 0.009251565 * max(0.0, Q.psi_0p2 - 0.926) / 0.04465927   # -0.9%  psi_0p2 > 0.926
        + 0.003536688 * max(0.0, 0.0184 - Q.e2) / 0.002239646   # +0.4%  e2 < 0.0184
        - 0.003074847 * max(0.0, 2.36 - Q.D2) / 0.5596338   # -0.3%  D2 < 2.36
        - 0.002969978 * max(0.0, Q.mass_over_sum_pt_sq - 0.0104) / 0.002990733   # -0.3%  mass_over_sum_pt_sq > 0.0104
        - 0.001908194 * max(0.0, Q.girth - 0.0792) / 0.01270605   # -0.2%  girth > 0.0792
        + 0.001306892 * max(0.0, 878.0 - Q.sum_pt_top40) / 4.701464   # +0.1%  sum_pt_top40 < 878
        + 0.001249747 * max(0.0, 2.6 - Q.pt_entropy) / 0.1312261   # +0.1%  pt_entropy < 2.6
        + 0.0005753687 * max(0.0, Q.e2 - 0.0511) / 0.001623089   # +0.1%  e2 > 0.0511
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.63734 * (-0.01640887
        + 0.1215256 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +12.2%  log_sum_pt > 6.91
        + 0.1213202 * max(0.0, 1170.0 - Q.sum_pt) / 140.1699   # +12.1%  sum_pt < 1170
        - 0.1199037 * max(0.0, Q.z_top50_slots - 0.956) / 0.0368059   # -12.0%  z_top50_slots > 0.956
        + 0.09990856 * max(0.0, Q.pt_entropy - 1.88) / 0.9835581   # +10.0%  pt_entropy > 1.88
        + 0.07241825 * max(0.0, Q.n_particles - 23.4) / 22.64751   # +7.2%  n_particles > 23.4
        - 0.05687342 * max(0.0, 119.0 - Q.mass) / 37.54852   # -5.7%  mass < 119
        - 0.03768751 * max(0.0, 895.0 - Q.sum_pt_top15) / 71.82361   # -3.8%  sum_pt_top15 < 895
        - 0.03277203 * max(0.0, 0.117 - Q.dr_0) / 0.06901767   # -3.3%  dr_0 < 0.117
        - 0.03241966 * max(0.0, Q.tau1 - 0.0284) / 0.05993078   # -3.2%  tau1 > 0.0284
        + 0.02603704 * max(0.0, Q.n_particles - 36.5) * max(0.0, 0.111 - Q.dr_0) / 0.6197241   # +2.6%  n_particles > 36.5 and dr_0 < 0.111
        + 0.02594799 * max(0.0, 0.0734 - Q.mass_over_sum_pt) / 0.009126968   # +2.6%  mass_over_sum_pt < 0.0734
        - 0.02592297 * max(0.0, Q.n_dr_0p1_0p2 - 6.88) / 6.625029   # -2.6%  n_dr_0p1_0p2 > 6.88
        - 0.02469015 * max(0.0, Q.log_sum_pt - 6.97) / 0.0255142   # -2.5%  log_sum_pt > 6.97
        + 0.02261027 * max(0.0, 16.7 - Q.n_dr_0p05_0p1) / 6.790161   # +2.3%  n_dr_0p05_0p1 < 16.7
        + 0.0221318 * max(0.0, 14.8 - Q.n_dr_0_0p05) / 5.085835   # +2.2%  n_dr_0_0p05 < 14.8
        + 0.02109896 * max(0.0, Q.sum_pt_top15 - 893.0) / 46.80409   # +2.1%  sum_pt_top15 > 893
        - 0.01707044 * max(0.0, Q.log_sum_pt - 7.07) / 0.01017945   # -1.7%  log_sum_pt > 7.07
        + 0.01496331 * max(0.0, 69.4 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0) / 99.9798   # +1.5%  mass_top20 < 69.4 and n_real_top40 > 29
        - 0.01481102 * max(0.0, 0.00108 - Q.lam2) / 0.0004292963   # -1.5%  lam2 < 0.00108
        + 0.01167208 * max(0.0, Q.pt_entropy - 3.08) / 0.11841   # +1.2%  pt_entropy > 3.08
        + 0.01081172 * max(0.0, 957.0 - Q.sum_pt) / 7.6544   # +1.1%  sum_pt < 957
        - 0.0107661 * max(0.0, 936.0 - Q.sum_pt_top50) / 7.107909   # -1.1%  sum_pt_top50 < 936
        - 0.01024179 * max(0.0, Q.n_dr_0p2_0p4 - 10.4) / 2.739489   # -1.0%  n_dr_0p2_0p4 > 10.4
        + 0.01004161 * max(0.0, 1.57 - Q.D2) / 0.2042369   # +1.0%  D2 < 1.57
        - 0.008358869 * max(0.0, Q.z_dr_0_0p05 - 0.864) / 0.01969821   # -0.8%  z_dr_0_0p05 > 0.864
        + 0.007975017 * max(0.0, 0.000756 - Q.girth2_top10) / 0.0001275799   # +0.8%  girth2_top10 < 0.000756
        - 0.00769575 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # -0.8%  psi_0p3 > 0.998
        - 0.006814871 * max(0.0, Q.n_dr_0p05_0p1 - 15.2) / 2.065234   # -0.7%  n_dr_0p05_0p1 > 15.2
        - 0.003067951 * max(0.0, Q.n_dr_0_0p05 - 24.1) / 0.6787575   # -0.3%  n_dr_0_0p05 > 24.1
        + 0.002441361 * max(0.0, 0.967 - Q.z_top50_slots) / 0.001153914   # +0.2%  z_top50_slots < 0.967
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 7.617;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.617063 * (0.2756968
        - 0.1045624 * max(0.0, 93.8 - Q.mass) / 18.2674   # -10.5%  mass < 93.8
        - 0.09991883 * max(0.0, Q.sum_pt_top50 - 933.0) / 109.5091   # -10.0%  sum_pt_top50 > 933
        - 0.09666119 * max(0.0, Q.mass_over_sum_pt - 0.0612) / 0.03119807   # -9.7%  mass_over_sum_pt > 0.0612
        - 0.08420471 * max(0.0, 1280.0 - Q.sum_pt) / 242.9517   # -8.4%  sum_pt < 1280
        + 0.07577706 * max(0.0, Q.sum_pt - 1020.0) / 46.54828   # +7.6%  sum_pt > 1020
        + 0.0688836 * max(0.0, 0.0211 - Q.girth2_top15) / 0.01437509   # +6.9%  girth2_top15 < 0.0211
        + 0.06289775 * max(0.0, Q.girth2 - 0.0089) / 0.003373916   # +6.3%  girth2 > 0.0089
        - 0.04871304 * max(0.0, Q.mass - 64.5) / 31.1807   # -4.9%  mass > 64.5
        - 0.04406389 * max(0.0, 1030.0 - Q.sum_pt) / 27.96979   # -4.4%  sum_pt < 1030
        + 0.03393762 * max(0.0, 0.0612 - Q.mass_over_sum_pt) / 0.005607484   # +3.4%  mass_over_sum_pt < 0.0612
        + 0.03032211 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.0209 - Q.girth2_top15) / 0.0009313122   # +3.0%  log_sum_pt > 6.9 and girth2_top15 < 0.0209
        + 0.02600538 * max(0.0, Q.mass_top15 - 31.5) / 31.79529   # +2.6%  mass_top15 > 31.5
        - 0.02563093 * max(0.0, Q.girth2_top30 - 0.00765) / 0.003297844   # -2.6%  girth2_top30 > 0.00765
        - 0.02255848 * max(0.0, Q.psi_0p3 - 0.961) / 0.03046621   # -2.3%  psi_0p3 > 0.961
        - 0.01945187 * max(0.0, Q.sum_pt_top50 - 995.0) * max(0.0, 0.0218 - Q.girth2_top15) / 0.987774   # -1.9%  sum_pt_top50 > 995 and girth2_top15 < 0.0218
        + 0.01754334 * max(0.0, 1030.0 - Q.sum_pt_top50) / 32.59238   # +1.8%  sum_pt_top50 < 1030
        + 0.01562376 * max(0.0, Q.mass_top30 - 88.3) / 10.43922   # +1.6%  mass_top30 > 88.3
        - 0.01539186 * max(0.0, Q.log_sum_pt - 6.96) / 0.0282508   # -1.5%  log_sum_pt > 6.96
        - 0.01484255 * max(0.0, Q.sum_pt - 1110.0) / 21.33144   # -1.5%  sum_pt > 1110
        - 0.01208948 * max(0.0, 50.2 - Q.n_particles) / 8.687389   # -1.2%  n_particles < 50.2
        + 0.01042722 * max(0.0, 963.0 - Q.sum_pt) / 8.360504   # +1.0%  sum_pt < 963
        - 0.009151633 * max(0.0, Q.psi_0p3 - 0.993) / 0.003630655   # -0.9%  psi_0p3 > 0.993
        - 0.007931806 * max(0.0, Q.sd_mass - 85.3) / 6.517483   # -0.8%  sd_mass > 85.3
        + 0.007231999 * max(0.0, Q.sum_pt_top40 - 1060.0) / 25.03936   # +0.7%  sum_pt_top40 > 1060
        - 0.007018249 * max(0.0, 0.00145 - Q.lam2) / 0.0006969811   # -0.7%  lam2 < 0.00145
        - 0.006954449 * max(0.0, Q.sum_pt_top40 - 987.0) / 56.59453   # -0.7%  sum_pt_top40 > 987
        - 0.005719628 * max(0.0, 0.997 - Q.z_top40_slots) / 0.01910823   # -0.6%  z_top40_slots < 0.997
        - 0.003532587 * max(0.0, Q.z_top5_slots - 0.498) / 0.1055213   # -0.4%  z_top5_slots > 0.498
        + 0.003200396 * max(0.0, Q.sd_rg - 0.264) / 0.009028747   # +0.3%  sd_rg > 0.264
        - 0.003179285 * max(0.0, 0.433 - Q.z_dr_0p1_0p2) / 0.2862507   # -0.3%  z_dr_0p1_0p2 < 0.433
        + 0.003093109 * max(0.0, Q.LHA - 0.29) / 0.02683418   # +0.3%  LHA > 0.29
        - 0.002883973 * max(0.0, Q.sj2_dr - 0.174) / 0.04557553   # -0.3%  sj2_dr > 0.174
        + 0.002069965 * max(0.0, Q.mass_top20 - 109.0) / 3.043834   # +0.2%  mass_top20 > 109
        + 0.001780983 * max(0.0, Q.sum_pt_top15 - 1040.0) / 8.752168   # +0.2%  sum_pt_top15 > 1040
        - 0.001729607 * max(0.0, Q.LHA - 0.427) / 0.00134434   # -0.2%  LHA > 0.427
        - 0.001410325 * max(0.0, Q.psi_0p2 - 0.816) / 0.1375485   # -0.1%  psi_0p2 > 0.816
        - 0.001344083 * max(0.0, Q.n_dr_0p1_0p2 - 21.6) / 1.297587   # -0.1%  n_dr_0p1_0p2 > 21.6
        - 0.0009039708 * max(0.0, Q.e3 - 0.00037) / 1.860974e-05   # -0.1%  e3 > 0.00037
        - 0.0006305019 * max(0.0, Q.z_dr_0p1_0p2 - 0.324) / 0.03873043   # -0.1%  z_dr_0p1_0p2 > 0.324
        - 0.0004573162 * max(0.0, Q.e2 - 0.0602) / 0.0007182281   # -0.0%  e2 > 0.0602
        - 0.000269032 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922642   # -0.0%  log_sum_pt > 6.9
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 7.944;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.943949 * (0.1510584
        - 0.1435176 * max(0.0, 0.173 - Q.tau1) / 0.08907003   # -14.4%  tau1 < 0.173
        + 0.1159056 * max(0.0, 0.00974 - Q.girth2) / 0.003596674   # +11.6%  girth2 < 0.00974
        + 0.08005861 * max(0.0, Q.mass - 95.4) / 13.76583   # +8.0%  mass > 95.4
        - 0.06697128 * max(0.0, Q.n_particles - 24.4) / 21.71496   # -6.7%  n_particles > 24.4
        - 0.06474579 * max(0.0, Q.mass - 79.6) / 20.49152   # -6.5%  mass > 79.6
        - 0.05804807 * max(0.0, 0.00851 - Q.girth2_top10) / 0.00411724   # -5.8%  girth2_top10 < 0.00851
        + 0.04864091 * max(0.0, Q.mass_top30 - 46.8) / 35.77786   # +4.9%  mass_top30 > 46.8
        - 0.04502586 * max(0.0, Q.mass_over_sum_pt - 0.1) / 0.01172731   # -4.5%  mass_over_sum_pt > 0.1
        + 0.04448567 * max(0.0, 25.2 - Q.n_dr_0p1_0p2) / 13.48824   # +4.4%  n_dr_0p1_0p2 < 25.2
        - 0.04367247 * max(0.0, Q.mass_top10 - 20.6) / 29.40101   # -4.4%  mass_top10 > 20.6
        - 0.0408238 * max(0.0, 91.7 - Q.mass) / 16.80322   # -4.1%  mass < 91.7
        - 0.03769269 * max(0.0, Q.mass_top50 - 95.1) / 12.63413   # -3.8%  mass_top50 > 95.1
        + 0.02842528 * max(0.0, Q.lam1 - 0.00748) / 0.002808569   # +2.8%  lam1 > 0.00748
        - 0.02688264 * max(0.0, Q.mass - 66.2) / 29.90957   # -2.7%  mass > 66.2
        + 0.02561744 * max(0.0, 12.6 - Q.n_dr_0p2_0p4) / 5.637219   # +2.6%  n_dr_0p2_0p4 < 12.6
        - 0.02490257 * max(0.0, Q.psi_0p3 - 0.966) / 0.02596125   # -2.5%  psi_0p3 > 0.966
        + 0.02456829 * max(0.0, 0.0287 - Q.e2) / 0.006061156   # +2.5%  e2 < 0.0287
        - 0.01142889 * max(0.0, 1.45 - Q.D2) / 0.1644755   # -1.1%  D2 < 1.45
        + 0.0113368 * max(0.0, 5.39 - Q.n_dr_0p2_0p4) * max(0.0, Q.psi_0p3 - 0.993) / 0.00804098   # +1.1%  n_dr_0p2_0p4 < 5.39 and psi_0p3 > 0.993
        - 0.0103683 * max(0.0, 0.99 - Q.z_top50_slots) / 0.004627259   # -1.0%  z_top50_slots < 0.99
        + 0.01021264 * max(0.0, 0.00121 - Q.lam2) / 0.0005200558   # +1.0%  lam2 < 0.00121
        + 0.009077743 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +0.9%  psi_0p3 > 0.997
        - 0.006337703 * max(0.0, Q.n_dr_0p05_0p1 - 9.06) / 4.576945   # -0.6%  n_dr_0p05_0p1 > 9.06
        + 0.006220425 * max(0.0, 4.05 - Q.n_dr_0p2_0p4) * max(0.0, Q.eccentricity - 0.75) / 0.1110444   # +0.6%  n_dr_0p2_0p4 < 4.05 and eccentricity > 0.75
        + 0.005501261 * max(0.0, Q.LHA - 0.398) / 0.003800151   # +0.6%  LHA > 0.398
        + 0.005481664 * max(0.0, 0.533 - Q.D2_b2) / 0.07046287   # +0.5%  D2_b2 < 0.533
        + 0.002282024 * max(0.0, Q.mass - 160.0) / 1.812828   # +0.2%  mass > 160
        + 0.001767968 * max(0.0, 24.1 - Q.n_pt_above_1) / 0.5464842   # +0.2%  n_pt_above_1 < 24.1
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 5.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.00958 * (0.02754722
        + 0.2134891 * max(0.0, 112.0 - Q.mass) / 32.02067   # +21.3%  mass < 112
        - 0.1401681 * max(0.0, 80.0 - Q.mass) / 10.51172   # -14.0%  mass < 80
        - 0.1253466 * max(0.0, 86.8 - Q.mass) / 13.89234   # -12.5%  mass < 86.8
        + 0.08769739 * max(0.0, 21.2 - Q.n_dr_0p2_0p4) / 12.77114   # +8.8%  n_dr_0p2_0p4 < 21.2
        + 0.0753729 * max(0.0, Q.girth2_top50 - 0.00885) / 0.003283361   # +7.5%  girth2_top50 > 0.00885
        + 0.05055505 * max(0.0, 49.3 - Q.n_particles) / 8.169663   # +5.1%  n_particles < 49.3
        - 0.04504989 * max(0.0, 0.00751 - Q.lam1) / 0.002416285   # -4.5%  lam1 < 0.00751
        - 0.03655134 * max(0.0, 0.0072 - Q.girth2_top30) / 0.002169513   # -3.7%  girth2_top30 < 0.0072
        - 0.03533693 * max(0.0, Q.girth2_top15 - 0.00761) / 0.002827846   # -3.5%  girth2_top15 > 0.00761
        + 0.03096206 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +3.1%  psi_0p3 > 0.997
        - 0.0216135 * max(0.0, 6.98 - Q.log_sum_pt) / 0.05852677   # -2.2%  log_sum_pt < 6.98
        - 0.01787547 * max(0.0, Q.n_particles - 48.7) / 4.947436   # -1.8%  n_particles > 48.7
        + 0.01684847 * max(0.0, Q.girth2_top15 - 0.0161) / 0.001267324   # +1.7%  girth2_top15 > 0.0161
        - 0.01321485 * max(0.0, 0.988 - Q.z_top50_slots) / 0.004163576   # -1.3%  z_top50_slots < 0.988
        - 0.01212102 * max(0.0, Q.psi_0p3 - 0.996) * max(0.0, 14.2 - Q.n_dr_0_0p05) / 0.007958217   # -1.2%  psi_0p3 > 0.996 and n_dr_0_0p05 < 14.2
        + 0.01134019 * max(0.0, Q.n_dr_0_0p05 - 14.1) / 3.054279   # +1.1%  n_dr_0_0p05 > 14.1
        - 0.01080854 * max(0.0, Q.girth - 0.126) / 0.003538971   # -1.1%  girth > 0.126
        + 0.009351393 * max(0.0, Q.sj3_pair_mass_min - 27.5) / 7.308353   # +0.9%  sj3_pair_mass_min > 27.5
        - 0.008331467 * max(0.0, 110.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3) / 0.1300223   # -0.8%  mass < 110 and psi_0p3 < 1
        - 0.007965992 * max(0.0, 2.77 - Q.pt_entropy) / 0.1918571   # -0.8%  pt_entropy < 2.77
        - 0.0077473 * max(0.0, Q.sj2_dr - 0.245) / 0.01724921   # -0.8%  sj2_dr > 0.245
        - 0.007187312 * max(0.0, Q.mass_top15 - 102.0) / 2.156013   # -0.7%  mass_top15 > 102
        - 0.005640879 * max(0.0, 20.4 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.0) / 30.98512   # -0.6%  n_dr_0p2_0p4 < 20.4 and n_dr_0p1_0p2 > 12
        + 0.004819905 * max(0.0, Q.mass - 154.0) / 2.530996   # +0.5%  mass > 154
        - 0.004604404 * max(0.0, Q.lam1 - 0.0191) / 0.0006552878   # -0.5%  lam1 > 0.0191
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.10409 * (0.2929507
        - 0.1908227 * max(0.0, 1160.0 - Q.sum_pt) / 131.1626   # -19.1%  sum_pt < 1160
        - 0.1765014 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # -17.7%  log_sum_pt > 6.91
        + 0.1360655 * max(0.0, 0.0258 - Q.mass_over_sum_pt_sq) / 0.01697306   # +13.6%  mass_over_sum_pt_sq < 0.0258
        + 0.08058101 * max(0.0, Q.sum_pt_top50 - 968.0) / 79.04833   # +8.1%  sum_pt_top50 > 968
        - 0.05557978 * max(0.0, 94.9 - Q.mass_top50) / 19.77405   # -5.6%  mass_top50 < 94.9
        + 0.0449742 * max(0.0, 63.0 - Q.n_particles) / 17.41086   # +4.5%  n_particles < 63
        + 0.03955343 * max(0.0, Q.mass_top50 - 97.3) / 12.07406   # +4.0%  mass_top50 > 97.3
        - 0.03830723 * max(0.0, Q.sum_pt_top15 - 740.0) / 146.0603   # -3.8%  sum_pt_top15 > 740
        - 0.0327449 * max(0.0, 835.0 - Q.sum_pt_top5) / 250.6495   # -3.3%  sum_pt_top5 < 835
        - 0.03157968 * max(0.0, Q.mass - 91.5) / 14.91046   # -3.2%  mass > 91.5
        - 0.02542697 * max(0.0, Q.z_top50_slots - 0.989) / 0.007692108   # -2.5%  z_top50_slots > 0.989
        + 0.02374122 * max(0.0, Q.log_sum_pt - 7.05) / 0.01217682   # +2.4%  log_sum_pt > 7.05
        + 0.01866707 * max(0.0, 65.8 - Q.mass) / 6.266238   # +1.9%  mass < 65.8
        - 0.01506785 * max(0.0, 61.5 - Q.n_particles) * max(0.0, 2.57 - Q.D2) / 13.35499   # -1.5%  n_particles < 61.5 and D2 < 2.57
        - 0.01364711 * max(0.0, 968.0 - Q.sum_pt) / 9.012525   # -1.4%  sum_pt < 968
        + 0.01177745 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 3.83 - Q.D2) / 0.002315183   # +1.2%  psi_0p3 > 0.997 and D2 < 3.83
        + 0.01066697 * max(0.0, 6.86 - Q.log_sum_pt) / 0.008420316   # +1.1%  log_sum_pt < 6.86
        + 0.01053289 * max(0.0, 0.0362 - Q.e2) / 0.01013574   # +1.1%  e2 < 0.0362
        - 0.009323462 * max(0.0, Q.mass_top40 - 112.0) / 7.417724   # -0.9%  mass_top40 > 112
        + 0.009199211 * max(0.0, Q.sd_mass - 65.3) / 13.81124   # +0.9%  sd_mass > 65.3
        - 0.008476691 * max(0.0, Q.mass - 150.0) / 3.058902   # -0.8%  mass > 150
        + 0.005877301 * max(0.0, Q.n_dr_0p1_0p2 - 7.0) / 6.540173   # +0.6%  n_dr_0p1_0p2 > 7
        - 0.005240477 * max(0.0, Q.width - 0.0247) / 0.0005876831   # -0.5%  width > 0.0247
        - 0.003268254 * max(0.0, 0.000928 - Q.girth2_top15) / 0.0001192156   # -0.3%  girth2_top15 < 0.000928
        + 0.001457316 * max(0.0, Q.z_top5_slots - 0.772) / 0.009145868   # +0.1%  z_top5_slots > 0.772
        + 0.0009199073 * max(0.0, Q.mass_top5 - 55.6) / 1.699237   # +0.1%  mass_top5 > 55.6
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 24.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.92289 * (0.5416708
        - 0.260972 * Q.sum_pt_top50 / 1035.697   # -26.1%  sum_pt_top50
        - 0.1253063 * Q.mass / 89.74122   # -12.5%  mass
        - 0.09106375 * max(0.0, Q.e2_sq - 0.00656) / 0.004218535   # -9.1%  e2_sq > 0.00656
        - 0.08809275 * max(0.0, 0.172 - Q.tau1) / 0.08817372   # -8.8%  tau1 < 0.172
        - 0.06108411 * max(0.0, 120.0 - Q.mass) / 38.34741   # -6.1%  mass < 120
        + 0.05790455 * max(0.0, Q.mass_over_sum_pt - 0.0796) / 0.01876656   # +5.8%  mass_over_sum_pt > 0.0796
        - 0.04179156 * max(0.0, 102.0 - Q.mass) / 24.33566   # -4.2%  mass < 102
        + 0.03240941 * max(0.0, 84.4 - Q.mass) / 12.62088   # +3.2%  mass < 84.4
        + 0.02495256 * max(0.0, Q.lam1 - 0.00674) / 0.003078662   # +2.5%  lam1 > 0.00674
        + 0.0243485 * max(0.0, 0.0627 - Q.girth) / 0.01339592   # +2.4%  girth < 0.0627
        - 0.02243439 * max(0.0, Q.e2 - 0.0287) / 0.008246751   # -2.2%  e2 > 0.0287
        - 0.01388274 * max(0.0, 0.27 - Q.LHA) / 0.04424526   # -1.4%  LHA < 0.27
        - 0.01332697 * max(0.0, 1060.0 - Q.sum_pt) / 47.7222   # -1.3%  sum_pt < 1060
        + 0.01300198 * max(0.0, Q.mass_top30 - 80.1) / 13.55844   # +1.3%  mass_top30 > 80.1
        - 0.01239813 * max(0.0, Q.pt_entropy - 2.46) / 0.4805555   # -1.2%  pt_entropy > 2.46
        - 0.01232152 * max(0.0, Q.mass_over_sum_pt - 0.102) / 0.01124864   # -1.2%  mass_over_sum_pt > 0.102
        + 0.00970998 * max(0.0, Q.e3 - 1.71e-05) / 8.832143e-05   # +1.0%  e3 > 1.71e-05
        + 0.009680489 * max(0.0, 0.0306 - Q.e2) / 0.006972998   # +1.0%  e2 < 0.0306
        + 0.009571835 * max(0.0, Q.log_sum_pt - 6.97) / 0.0255142   # +1.0%  log_sum_pt > 6.97
        + 0.009270799 * max(0.0, Q.girth2_top20 - 0.00768) / 0.003000715   # +0.9%  girth2_top20 > 0.00768
        + 0.009239893 * max(0.0, 0.382 - Q.pt_dispersion) / 0.08313531   # +0.9%  pt_dispersion < 0.382
        + 0.006962909 * max(0.0, 89.1 - Q.mass_top15) / 33.11752   # +0.7%  mass_top15 < 89.1
        + 0.006308205 * max(0.0, 75.5 - Q.mass_top40) / 10.20901   # +0.6%  mass_top40 < 75.5
        + 0.0052419 * max(0.0, 0.263 - Q.z_dr_0p1_0p2) / 0.141236   # +0.5%  z_dr_0p1_0p2 < 0.263
        - 0.004729454 * max(0.0, Q.log_sum_pt - 7.05) / 0.01217682   # -0.5%  log_sum_pt > 7.05
        + 0.004331185 * max(0.0, Q.LHA - 0.4) / 0.003586234   # +0.4%  LHA > 0.4
        + 0.004179804 * max(0.0, 0.203 - Q.LHA) / 0.01918468   # +0.4%  LHA < 0.203
        + 0.00407527 * max(0.0, Q.sum_pt - 1160.0) / 14.95839   # +0.4%  sum_pt > 1160
        - 0.003230963 * max(0.0, Q.pt_entropy - 2.93) / 0.1830112   # -0.3%  pt_entropy > 2.93
        - 0.00322334 * max(0.0, Q.mass_over_sum_pt - 0.0295) * max(0.0, 0.694 - Q.z_dr_0p05_0p1) / 0.02315128   # -0.3%  mass_over_sum_pt > 0.0295 and z_dr_0p05_0p1 < 0.694
        - 0.00308671 * max(0.0, 0.00419 - Q.girth2_top50) / 0.0006631872   # -0.3%  girth2_top50 < 0.00419
        - 0.002681542 * max(0.0, 0.000618 - Q.lam2) / 0.0001512031   # -0.3%  lam2 < 0.000618
        + 0.002087755 * max(0.0, 122.0 - Q.mass) * max(0.0, 1000.0 - Q.sum_pt) / 609.2844   # +0.2%  mass < 122 and sum_pt < 1000
        + 0.001835597 * max(0.0, Q.eccentricity - 0.929) / 0.008797767   # +0.2%  eccentricity > 0.929
        + 0.00177884 * max(0.0, Q.lam2 - 0.00381) / 0.0003144244   # +0.2%  lam2 > 0.00381
        - 0.00151 * max(0.0, 0.0152 - Q.e2) / 0.001383587   # -0.2%  e2 < 0.0152
        + 0.0009282772 * max(0.0, Q.e2 - 0.0489) * max(0.0, 1.6 - Q.D2_b2) / 0.0008352112   # +0.1%  e2 > 0.0489 and D2_b2 < 1.6
        - 0.000650487 * max(0.0, 6.8 - Q.log_sum_pt) / 0.004369815   # -0.1%  log_sum_pt < 6.8
        + 0.0003935686 * max(0.0, Q.mass_top10 - 98.2) / 0.7910377   # +0.0%  mass_top10 > 98.2
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 32.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 32.98558 * (-0.143396
        + 0.1203464 * max(0.0, 0.0251 - Q.girth2_top50) / 0.01647176   # +12.0%  girth2_top50 < 0.0251
        + 0.08516575 * max(0.0, 103.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.977) / 0.4826876   # +8.5%  mass < 103 and psi_0p3 > 0.977
        + 0.06917544 * max(0.0, 0.00944 - Q.girth2_top50) / 0.003441617   # +6.9%  girth2_top50 < 0.00944
        - 0.06609291 * max(0.0, 96.5 - Q.mass_top50) / 20.96262   # -6.6%  mass_top50 < 96.5
        - 0.05873193 * max(0.0, 93.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.979) / 0.3079979   # -5.9%  mass < 93 and psi_0p3 > 0.979
        + 0.05821952 * max(0.0, 0.268 - Q.sd_rg) / 0.1361989   # +5.8%  sd_rg < 0.268
        + 0.05791468 * max(0.0, 102.0 - Q.mass) / 24.33566   # +5.8%  mass < 102
        - 0.04580268 * max(0.0, 90.4 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997) / 0.02019823   # -4.6%  mass < 90.4 and psi_0p3 > 0.997
        + 0.03712783 * max(0.0, 99.7 - Q.mass) * max(0.0, Q.psi_0p3 - 0.997) / 0.02987032   # +3.7%  mass < 99.7 and psi_0p3 > 0.997
        - 0.03679558 * max(0.0, 95.3 - Q.mass) * max(0.0, Q.psi_0p3 - 0.967) / 0.5645226   # -3.7%  mass < 95.3 and psi_0p3 > 0.967
        - 0.03427449 * max(0.0, 0.214 - Q.sd_rg) / 0.09191576   # -3.4%  sd_rg < 0.214
        - 0.03317534 * max(0.0, 0.00836 - Q.girth2) / 0.002574842   # -3.3%  girth2 < 0.00836
        - 0.02827094 * max(0.0, 0.00825 - Q.lam1) / 0.002951055   # -2.8%  lam1 < 0.00825
        - 0.02485113 * max(0.0, Q.mass - 100.0) / 12.57253   # -2.5%  mass > 100
        - 0.02410804 * max(0.0, 0.00816 - Q.girth2_top30) / 0.00281992   # -2.4%  girth2_top30 < 0.00816
        + 0.02328161 * max(0.0, Q.mass_top50 - 83.6) / 16.9527   # +2.3%  mass_top50 > 83.6
        - 0.02101291 * max(0.0, 0.108 - Q.tau1) / 0.03483031   # -2.1%  tau1 < 0.108
        + 0.01954885 * max(0.0, Q.lam1 - 0.00761) / 0.002767512   # +2.0%  lam1 > 0.00761
        + 0.0182896 * max(0.0, 80.7 - Q.mass_top30) / 14.82292   # +1.8%  mass_top30 < 80.7
        - 0.01760207 * max(0.0, Q.girth - 0.0894) / 0.009993364   # -1.8%  girth > 0.0894
        + 0.01487191 * max(0.0, Q.LHA - 0.32) / 0.01679995   # +1.5%  LHA > 0.32
        + 0.0127703 * max(0.0, 82.0 - Q.mass) / 11.4156   # +1.3%  mass < 82
        - 0.01128229 * max(0.0, Q.psi_0p2 - 0.806) / 0.1465169   # -1.1%  psi_0p2 > 0.806
        + 0.00972854 * max(0.0, 0.0859 - Q.tau1) / 0.02043959   # +1.0%  tau1 < 0.0859
        - 0.008835179 * max(0.0, Q.psi_0p3 - 0.964) / 0.02775557   # -0.9%  psi_0p3 > 0.964
        + 0.008647647 * max(0.0, Q.sd_rg - 0.158) / 0.03629105   # +0.9%  sd_rg > 0.158
        - 0.007411519 * max(0.0, Q.girth2_top15 - 0.00865) / 0.002581555   # -0.7%  girth2_top15 > 0.00865
        + 0.00647738 * max(0.0, Q.lam2 - 0.000777) / 0.0008720822   # +0.6%  lam2 > 0.000777
        + 0.006142388 * max(0.0, 6.54 - Q.n_dr_0p2_0p4) / 1.80902   # +0.6%  n_dr_0p2_0p4 < 6.54
        + 0.00611769 * max(0.0, Q.e2 - 0.0342) / 0.005605432   # +0.6%  e2 > 0.0342
        - 0.004573533 * max(0.0, Q.LHA - 0.395) / 0.004133168   # -0.5%  LHA > 0.395
        - 0.00447556 * max(0.0, 1090.0 - Q.sum_pt) / 70.97545   # -0.4%  sum_pt < 1090
        + 0.003776254 * max(0.0, Q.mass - 125.0) / 7.117824   # +0.4%  mass > 125
        + 0.003684822 * max(0.0, 49.5 - Q.mass) / 2.749909   # +0.4%  mass < 49.5
        - 0.002917467 * max(0.0, Q.sd_mass - 92.8) / 5.091764   # -0.3%  sd_mass > 92.8
        + 0.002610988 * max(0.0, 47.5 - Q.n_particles) / 7.177081   # +0.3%  n_particles < 47.5
        - 0.002568897 * max(0.0, 2.9 - Q.pt_entropy) / 0.2506999   # -0.3%  pt_entropy < 2.9
        - 0.00231791 * max(0.0, 91.7 - Q.mass) * max(0.0, 0.577 - Q.z_dr_0_0p05) / 0.6216065   # -0.2%  mass < 91.7 and z_dr_0_0p05 < 0.577
        + 0.001002057 * max(0.0, Q.log_sum_pt - 6.96) / 0.0282508   # +0.1%  log_sum_pt > 6.96
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 10.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.55886 * (0.2377151
        - 0.1894515 * max(0.0, 102.0 - Q.mass) / 24.33566   # -18.9%  mass < 102
        - 0.1881312 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # -18.8%  log_sum_pt < 7.06
        + 0.1500086 * max(0.0, 1170.0 - Q.sum_pt) / 140.1699   # +15.0%  sum_pt < 1170
        + 0.08097119 * max(0.0, 83.1 - Q.mass) / 11.95753   # +8.1%  mass < 83.1
        - 0.06624846 * max(0.0, 18.7 - Q.n_dr_0p2_0p4) / 10.56659   # -6.6%  n_dr_0p2_0p4 < 18.7
        - 0.06424287 * max(0.0, Q.z_top50_slots - 0.97) / 0.02371788   # -6.4%  z_top50_slots > 0.97
        + 0.05095052 * max(0.0, Q.mass_over_sum_pt - 0.0747) * max(0.0, 1100.0 - Q.sum_pt) / 2.178054   # +5.1%  mass_over_sum_pt > 0.0747 and sum_pt < 1100
        + 0.04741604 * max(0.0, 1030.0 - Q.sum_pt) / 27.96979   # +4.7%  sum_pt < 1030
        - 0.04180132 * max(0.0, Q.mass - 124.0) / 7.30752   # -4.2%  mass > 124
        + 0.04158002 * max(0.0, Q.mass_top50 - 112.0) / 8.728379   # +4.2%  mass_top50 > 112
        + 0.03392576 * max(0.0, 0.0443 - Q.e2) / 0.01606355   # +3.4%  e2 < 0.0443
        - 0.0262401 * max(0.0, Q.girth2_top40 - 0.00536) * max(0.0, 6.98 - Q.log_sum_pt) / 0.0003714014   # -2.6%  girth2_top40 > 0.00536 and log_sum_pt < 6.98
        + 0.01335188 * max(0.0, Q.girth2_top20 - 0.0074) / 0.003084914   # +1.3%  girth2_top20 > 0.0074
        + 0.005680573 * max(0.0, 20.9 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.228) / 0.4410321   # +0.6%  n_dr_0p2_0p4 < 20.9 and z_dr_0p1_0p2 > 0.228
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 16.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.06897 * (0.6167165
        - 0.2611821 * max(0.0, 158.0 - Q.mass) / 70.30029   # -26.1%  mass < 158
        - 0.2557808 * max(0.0, 143.0 - Q.mass) / 57.32406   # -25.6%  mass < 143
        - 0.2275771 * max(0.0, Q.mass - 65.1) / 30.73051   # -22.8%  mass > 65.1
        + 0.07823384 * max(0.0, 88.3 - Q.mass) / 14.72058   # +7.8%  mass < 88.3
        + 0.04297743 * max(0.0, 69.5 - Q.mass) / 7.239026   # +4.3%  mass < 69.5
        + 0.02727668 * max(0.0, 0.00887 - Q.girth2_top40) / 0.003130773   # +2.7%  girth2_top40 < 0.00887
        - 0.01989062 * max(0.0, Q.mass_over_sum_pt_sq - 0.0255) / 0.0004947705   # -2.0%  mass_over_sum_pt_sq > 0.0255
        + 0.01692666 * max(0.0, Q.mass_over_sum_pt - 0.159) / 0.001511078   # +1.7%  mass_over_sum_pt > 0.159
        + 0.01582871 * max(0.0, Q.n_particles - 40.4) / 9.282885   # +1.6%  n_particles > 40.4
        + 0.01360524 * max(0.0, Q.girth - 0.104) / 0.007075154   # +1.4%  girth > 0.104
        - 0.01017796 * max(0.0, 0.0419 - Q.girth) / 0.005820264   # -1.0%  girth < 0.0419
        - 0.009497472 * max(0.0, 96.8 - Q.mass_top40) * max(0.0, 1020.0 - Q.sum_pt) / 561.0831   # -0.9%  mass_top40 < 96.8 and sum_pt < 1020
        + 0.008533545 * max(0.0, 994.0 - Q.sum_pt) / 13.71253   # +0.9%  sum_pt < 994
        + 0.006360187 * max(0.0, Q.mass_top30 - 125.0) / 3.384161   # +0.6%  mass_top30 > 125
        + 0.00245648 * max(0.0, 2.36 - Q.pt_entropy) / 0.07216291   # +0.2%  pt_entropy < 2.36
        + 0.002429575 * max(0.0, 101.0 - Q.mass_top40) * max(0.0, 918.0 - Q.sum_pt_top40) / 205.4778   # +0.2%  mass_top40 < 101 and sum_pt_top40 < 918
        + 0.001265562 * max(0.0, 110.0 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.63) / 22.97884   # +0.1%  mass_top40 < 110 and n_dr_0p2_0p4 > 9.63
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.77277 * (0.03905836
        - 0.1484323 * max(0.0, 0.0891 - Q.girth) / 0.02991481   # -14.8%  girth < 0.0891
        + 0.0923264 * max(0.0, 90.3 - Q.mass) / 15.89646   # +9.2%  mass < 90.3
        + 0.0820479 * max(0.0, Q.mass - 78.2) / 21.37697   # +8.2%  mass > 78.2
        + 0.08036466 * max(0.0, Q.e2 - 0.0131) / 0.01869619   # +8.0%  e2 > 0.0131
        + 0.06232569 * max(0.0, 0.276 - Q.LHA) / 0.04721656   # +6.2%  LHA < 0.276
        - 0.06216364 * max(0.0, Q.LHA - 0.276) / 0.03327278   # -6.2%  LHA > 0.276
        + 0.04422263 * max(0.0, Q.n_particles - 33.1) / 14.2952   # +4.4%  n_particles > 33.1
        + 0.04211613 * max(0.0, Q.girth - 0.0881) / 0.01028383   # +4.2%  girth > 0.0881
        - 0.03994256 * max(0.0, Q.z_top5_slots - 0.339) / 0.2323079   # -4.0%  z_top5_slots > 0.339
        - 0.03499609 * max(0.0, Q.mass - 98.1) / 13.05528   # -3.5%  mass > 98.1
        + 0.03268467 * max(0.0, 0.000137 - Q.e3) / 8.23964e-05   # +3.3%  e3 < 0.000137
        - 0.03248829 * max(0.0, Q.mass - 147.0) / 3.477839   # -3.2%  mass > 147
        + 0.02777026 * max(0.0, Q.mass_top50 - 142.0) / 3.47664   # +2.8%  mass_top50 > 142
        + 0.02730531 * max(0.0, Q.mass_top10 - 18.7) / 30.79198   # +2.7%  mass_top10 > 18.7
        - 0.02168343 * max(0.0, 0.0389 - Q.girth) / 0.004989474   # -2.2%  girth < 0.0389
        - 0.01992541 * max(0.0, Q.n_dr_0p05_0p1 - 4.41) / 7.586428   # -2.0%  n_dr_0p05_0p1 > 4.41
        - 0.01607553 * max(0.0, Q.girth2_top50 - 0.00838) / 0.003407173   # -1.6%  girth2_top50 > 0.00838
        - 0.01585728 * max(0.0, 0.0204 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.998) / 1.018504e-05   # -1.6%  girth2_top10 < 0.0204 and psi_0p3 > 0.998
        - 0.01548172 * max(0.0, Q.n_dr_0p1_0p2 - 6.11) / 7.169523   # -1.5%  n_dr_0p1_0p2 > 6.11
        - 0.01512828 * max(0.0, Q.mass_top20 - 74.7) / 10.69314   # -1.5%  mass_top20 > 74.7
        - 0.01341758 * max(0.0, 59.9 - Q.mass) / 4.846326   # -1.3%  mass < 59.9
        - 0.01083047 * max(0.0, Q.n_dr_0_0p05 - 12.0) / 3.970124   # -1.1%  n_dr_0_0p05 > 12
        + 0.01050068 * max(0.0, 11.9 - Q.n_dr_0_0p05) / 3.416829   # +1.1%  n_dr_0_0p05 < 11.9
        + 0.01015128 * max(0.0, Q.mass_top5 - 25.7) / 10.94617   # +1.0%  mass_top5 > 25.7
        - 0.008916546 * max(0.0, Q.mass - 159.0) / 1.925761   # -0.9%  mass > 159
        - 0.007069665 * max(0.0, Q.n_dr_0p2_0p4 - 9.37) / 3.099065   # -0.7%  n_dr_0p2_0p4 > 9.37
        + 0.006500539 * max(0.0, 0.00789 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.998) / 1.714838e-06   # +0.7%  girth2_top30 < 0.00789 and psi_0p3 > 0.998
        - 0.00409789 * max(0.0, Q.mass_top10 - 75.1) / 3.072953   # -0.4%  mass_top10 > 75.1
        - 0.003314649 * max(0.0, 2.46 - Q.pt_entropy) / 0.09344758   # -0.3%  pt_entropy < 2.46
        - 0.003195701 * max(0.0, 0.0208 - Q.girth) / 0.001192155   # -0.3%  girth < 0.0208
        - 0.002625485 * max(0.0, Q.mass_over_sum_pt_sq - 0.0273) / 0.0003286921   # -0.3%  mass_over_sum_pt_sq > 0.0273
        - 0.002305195 * max(0.0, 973.0 - Q.sum_pt) / 9.729745   # -0.2%  sum_pt < 973
        - 0.001984516 * max(0.0, 2.48 - Q.D2) * max(0.0, Q.sj2_dr - 0.205) / 0.01236996   # -0.2%  D2 < 2.48 and sj2_dr > 0.205
        - 0.001751678 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, Q.dr_3 - 0.00491) / 1.060538   # -0.2%  sum_pt < 1000 and dr_3 > 0.00491
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 12.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.04307 * (0.1644099
        - 0.1168281 * max(0.0, Q.LHA - 0.176) / 0.0983894   # -11.7%  LHA > 0.176
        + 0.1110942 * max(0.0, 100.0 - Q.mass) / 22.83131   # +11.1%  mass < 100
        - 0.08426722 * max(0.0, Q.mass_over_sum_pt - 0.097) / 0.01246727   # -8.4%  mass_over_sum_pt > 0.097
        - 0.07554885 * max(0.0, 0.0768 - Q.girth) / 0.02106111   # -7.6%  girth < 0.0768
        + 0.06642695 * max(0.0, Q.e2_sq - 0.00885) / 0.003375461   # +6.6%  e2_sq > 0.00885
        - 0.0588058 * max(0.0, 79.8 - Q.mass) / 10.43008   # -5.9%  mass < 79.8
        - 0.05795201 * max(0.0, 0.0552 - Q.e2) / 0.02547153   # -5.8%  e2 < 0.0552
        + 0.0480689 * max(0.0, 94.2 - Q.mass) / 18.55439   # +4.8%  mass < 94.2
        - 0.04117481 * max(0.0, 82.7 - Q.mass_top50) / 12.27403   # -4.1%  mass_top50 < 82.7
        + 0.03630018 * max(0.0, 0.00821 - Q.e2_sq) / 0.002469861   # +3.6%  e2_sq < 0.00821
        - 0.0350932 * max(0.0, 0.00816 - Q.girth2_top10) / 0.003842089   # -3.5%  girth2_top10 < 0.00816
        + 0.03186511 * max(0.0, 0.0255 - Q.e2) / 0.004685637   # +3.2%  e2 < 0.0255
        + 0.03165577 * max(0.0, 79.9 - Q.mass_top50) * max(0.0, 0.00411 - Q.zdr_4) / 0.03562921   # +3.2%  mass_top50 < 79.9 and zdr_4 < 0.00411
        - 0.03091326 * max(0.0, 90.1 - Q.mass) * max(0.0, 0.00447 - Q.zdr_4) / 0.05221466   # -3.1%  mass < 90.1 and zdr_4 < 0.00447
        - 0.02842695 * max(0.0, 0.0066 - Q.girth2_top30) / 0.001811363   # -2.8%  girth2_top30 < 0.0066
        + 0.02819977 * max(0.0, 15.1 - Q.n_dr_0p2_0p4) / 7.563736   # +2.8%  n_dr_0p2_0p4 < 15.1
        + 0.01767656 * max(0.0, 0.00394 - Q.girth2_top10) / 0.001244912   # +1.8%  girth2_top10 < 0.00394
        + 0.01657242 * max(0.0, 9.12 - Q.n_dr_0p2_0p4) * max(0.0, 20.8 - Q.n_dr_0p1_0p2) / 36.35388   # +1.7%  n_dr_0p2_0p4 < 9.12 and n_dr_0p1_0p2 < 20.8
        - 0.01514013 * max(0.0, Q.n_dr_0p1_0p2 - 7.42) / 6.265761   # -1.5%  n_dr_0p1_0p2 > 7.42
        - 0.01281665 * max(0.0, 9.41 - Q.n_dr_0p2_0p4) * max(0.0, 0.0059 - Q.girth2) / 0.005197028   # -1.3%  n_dr_0p2_0p4 < 9.41 and girth2 < 0.0059
        + 0.01241823 * max(0.0, Q.girth2_top15 - 0.00856) / 0.002600932   # +1.2%  girth2_top15 > 0.00856
        + 0.01049628 * max(0.0, 0.0328 - Q.girth) / 0.003472729   # +1.0%  girth < 0.0328
        - 0.00671294 * max(0.0, 9.72 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top40 - 25.2) / 33.96823   # -0.7%  n_dr_0p2_0p4 < 9.72 and n_real_top40 > 25.2
        + 0.006139292 * max(0.0, 1.59 - Q.D2) / 0.2112454   # +0.6%  D2 < 1.59
        + 0.005959535 * max(0.0, 5.79 - Q.n_dr_0p2_0p4) / 1.449921   # +0.6%  n_dr_0p2_0p4 < 5.79
        + 0.003451753 * max(0.0, Q.psi_0p2 - 0.995) / 0.0009804172   # +0.3%  psi_0p2 > 0.995
        + 0.003304482 * max(0.0, 0.00909 - Q.mass_over_sum_pt_sq) * max(0.0, Q.z_dr_0p1_0p2 - 0.0755) / 8.325545e-05   # +0.3%  mass_over_sum_pt_sq < 0.00909 and z_dr_0p1_0p2 > 0.0755
        - 0.002906356 * max(0.0, Q.n_dr_0p05_0p1 - 14.1) / 2.413892   # -0.3%  n_dr_0p05_0p1 > 14.1
        - 0.002351799 * max(0.0, Q.n_dr_0_0p05 - 18.0) / 1.792587   # -0.2%  n_dr_0_0p05 > 18
        - 0.001432526 * max(0.0, Q.sum_pt_top5 - 762.0) / 19.38427   # -0.1%  sum_pt_top5 > 762
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 4.826;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.825907 * (-0.1251578
        + 0.2688877 * max(0.0, 99.7 - Q.mass) / 22.60674   # +26.9%  mass < 99.7
        + 0.1831222 * max(0.0, 81.7 - Q.mass) / 11.27208   # +18.3%  mass < 81.7
        - 0.1058251 * max(0.0, 85.5 - Q.mass) / 13.19643   # -10.6%  mass < 85.5
        - 0.05237312 * max(0.0, Q.sd_mass - 65.6) / 13.66205   # -5.2%  sd_mass > 65.6
        - 0.04860275 * max(0.0, Q.pt_entropy - 2.17) / 0.7194859   # -4.9%  pt_entropy > 2.17
        - 0.04328116 * max(0.0, 60.2 - Q.mass) / 4.914609   # -4.3%  mass < 60.2
        + 0.04256822 * max(0.0, 21.1 - Q.n_dr_0p2_0p4) / 12.68088   # +4.3%  n_dr_0p2_0p4 < 21.1
        - 0.03803082 * max(0.0, 0.00732 - Q.girth2_top10) / 0.003208623   # -3.8%  girth2_top10 < 0.00732
        + 0.03778815 * max(0.0, Q.sd_mass - 83.1) / 7.041009   # +3.8%  sd_mass > 83.1
        - 0.03310766 * max(0.0, 0.0719 - Q.mass_over_sum_pt) / 0.008636459   # -3.3%  mass_over_sum_pt < 0.0719
        + 0.03173795 * max(0.0, Q.mass - 115.0) / 9.11693   # +3.2%  mass > 115
        - 0.02945 * max(0.0, 79.7 - Q.mass) * max(0.0, 0.0842 - Q.z_dr_0p1_0p2) / 0.6899174   # -2.9%  mass < 79.7 and z_dr_0p1_0p2 < 0.0842
        + 0.02521099 * max(0.0, Q.LHA - 0.368) / 0.007749422   # +2.5%  LHA > 0.368
        - 0.02158753 * max(0.0, Q.lam2 - 0.000289) / 0.001148615   # -2.2%  lam2 > 0.000289
        - 0.01476788 * max(0.0, 90.0 - Q.mass) * max(0.0, 1.0 - Q.psi_0p3) / 0.05939035   # -1.5%  mass < 90 and psi_0p3 < 1
        - 0.01295998 * max(0.0, Q.girth2_top20 - 0.018) / 0.00110893   # -1.3%  girth2_top20 > 0.018
        - 0.005490388 * max(0.0, 990.0 - Q.sum_pt_top50) / 15.96151   # -0.5%  sum_pt_top50 < 990
        + 0.003952674 * max(0.0, 0.000882 - Q.girth2_top10) * max(0.0, 1060.0 - Q.sum_pt_top40) / 0.007451265   # +0.4%  girth2_top10 < 0.000882 and sum_pt_top40 < 1060
        - 0.001255767 * max(0.0, Q.mass - 183.0) / 0.4359866   # -0.1%  mass > 183
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 10.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.2356 * (0.2501076
        - 0.09689655 * max(0.0, 0.0877 - Q.girth) / 0.02883122   # -9.7%  girth < 0.0877
        - 0.06747589 * max(0.0, 1110.0 - Q.sum_pt) / 87.5356   # -6.7%  sum_pt < 1110
        + 0.06659132 * max(0.0, Q.mass - 73.5) / 24.60656   # +6.7%  mass > 73.5
        - 0.06373289 * max(0.0, Q.LHA - 0.179) / 0.09607424   # -6.4%  LHA > 0.179
        - 0.05424787 * max(0.0, 72.2 - Q.mass_top10) / 28.62161   # -5.4%  mass_top10 < 72.2
        + 0.04617005 * max(0.0, Q.log_sum_pt - 6.86) / 0.09302715   # +4.6%  log_sum_pt > 6.86
        + 0.04087269 * max(0.0, 65.7 - Q.mass_top10) / 23.63595   # +4.1%  mass_top10 < 65.7
        - 0.038822 * max(0.0, Q.mass - 154.0) / 2.530996   # -3.9%  mass > 154
        + 0.03315649 * max(0.0, 1010.0 - Q.sum_pt_top50) / 22.47526   # +3.3%  sum_pt_top50 < 1010
        - 0.03205901 * max(0.0, Q.mass - 100.0) / 12.57253   # -3.2%  mass > 100
        - 0.03183477 * max(0.0, Q.girth2_top15 - 0.000862) / 0.006622924   # -3.2%  girth2_top15 > 0.000862
        + 0.03057125 * max(0.0, Q.z_top40_slots - 0.933) / 0.04912322   # +3.1%  z_top40_slots > 0.933
        - 0.03014194 * max(0.0, 6.91 - Q.log_sum_pt) / 0.01723579   # -3.0%  log_sum_pt < 6.91
        + 0.03006229 * max(0.0, 1060.0 - Q.sum_pt_top40) / 63.05439   # +3.0%  sum_pt_top40 < 1060
        + 0.02924968 * max(0.0, Q.girth - 0.0859) / 0.01080823   # +2.9%  girth > 0.0859
        - 0.02854416 * max(0.0, Q.sum_pt_top40 - 986.0) / 57.28754   # -2.9%  sum_pt_top40 > 986
        + 0.02776706 * max(0.0, Q.n_particles - 21.0) / 24.93091   # +2.8%  n_particles > 21
        - 0.02775216 * max(0.0, Q.mass - 145.0) / 3.767373   # -2.8%  mass > 145
        + 0.02629253 * max(0.0, 0.0244 - Q.girth2_top50) * max(0.0, Q.psi_0p3 - 0.963) / 0.000502089   # +2.6%  girth2_top50 < 0.0244 and psi_0p3 > 0.963
        + 0.02588424 * max(0.0, Q.mass_top50 - 147.0) / 2.791788   # +2.6%  mass_top50 > 147
        + 0.02330188 * max(0.0, 0.0893 - Q.tau1) / 0.02229053   # +2.3%  tau1 < 0.0893
        - 0.01932049 * max(0.0, 77.8 - Q.mass_top15) / 24.11667   # -1.9%  mass_top15 < 77.8
        - 0.01719749 * max(0.0, Q.mass - 48.1) * max(0.0, 1020.0 - Q.sum_pt) / 1047.777   # -1.7%  mass > 48.1 and sum_pt < 1020
        + 0.01624812 * max(0.0, 0.0081 - Q.girth2_top40) / 0.002562546   # +1.6%  girth2_top40 < 0.0081
        + 0.0152446 * max(0.0, Q.pt_entropy - 2.11) / 0.7724633   # +1.5%  pt_entropy > 2.11
        - 0.01064537 * max(0.0, 1040.0 - Q.sum_pt) * max(0.0, 4.15 - Q.D2) / 55.5927   # -1.1%  sum_pt < 1040 and D2 < 4.15
        - 0.008865824 * max(0.0, Q.mass - 133.0) * max(0.0, Q.D2 - 1.01) / 6.721999   # -0.9%  mass > 133 and D2 > 1.01
        - 0.008555563 * max(0.0, 6.86 - Q.log_sum_pt) / 0.008420316   # -0.9%  log_sum_pt < 6.86
        - 0.006610101 * max(0.0, Q.n_dr_0p05_0p1 - 6.5) / 6.095344   # -0.7%  n_dr_0p05_0p1 > 6.5
        + 0.006496101 * max(0.0, Q.mass_over_sum_pt - 0.169) / 0.0006583313   # +0.6%  mass_over_sum_pt > 0.169
        + 0.005663358 * max(0.0, Q.mass - 156.0) * max(0.0, Q.D2 - 1.33) / 1.958373   # +0.6%  mass > 156 and D2 > 1.33
        + 0.005092672 * max(0.0, 0.000844 - Q.girth2_top15) / 0.0001016112   # +0.5%  girth2_top15 < 0.000844
        - 0.004083714 * max(0.0, Q.sum_pt_top15 - 964.0) / 20.89962   # -0.4%  sum_pt_top15 > 964
        + 0.00397847 * max(0.0, 951.0 - Q.sum_pt) / 7.021037   # +0.4%  sum_pt < 951
        + 0.003388964 * max(0.0, Q.mass - 131.0) * max(0.0, 1010.0 - Q.sum_pt) / 111.5372   # +0.3%  mass > 131 and sum_pt < 1010
        - 0.003008239 * max(0.0, Q.z_dr_0p1_0p2 - 0.286) / 0.04541463   # -0.3%  z_dr_0p1_0p2 > 0.286
        - 0.003003004 * max(0.0, Q.n_dr_0p2_0p4 - 12.5) / 2.11983   # -0.3%  n_dr_0p2_0p4 > 12.5
        + 0.002912728 * max(0.0, Q.mass_top5 - 50.1) / 2.57013   # +0.3%  mass_top5 > 50.1
        + 0.002794718 * max(0.0, Q.sum_pt_top15 - 1000.0) / 13.42986   # +0.3%  sum_pt_top15 > 1000
        - 0.002792979 * max(0.0, Q.z_top5_slots - 0.752) / 0.01191159   # -0.3%  z_top5_slots > 0.752
        - 0.001612142 * max(0.0, 0.971 - Q.z_top50_slots) / 0.001513874   # -0.2%  z_top50_slots < 0.971
        - 0.001058619 * max(0.0, Q.n_dr_0_0p05 - 26.7) / 0.4199844   # -0.1%  n_dr_0_0p05 > 26.7
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 25.6;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.6025 * (0.101162
        - 0.08104748 * max(0.0, 7.06 - Q.log_sum_pt) / 0.1265255   # -8.1%  log_sum_pt < 7.06
        + 0.06835518 * max(0.0, 1150.0 - Q.sum_pt_top50) / 128.6811   # +6.8%  sum_pt_top50 < 1150
        - 0.06814402 * max(0.0, 91.9 - Q.mass) / 16.93842   # -6.8%  mass < 91.9
        - 0.05943583 * max(0.0, 21.1 - Q.n_dr_0p2_0p4) / 12.68088   # -5.9%  n_dr_0p2_0p4 < 21.1
        + 0.04859215 * max(0.0, 0.00955 - Q.e2_sq) / 0.003455779   # +4.9%  e2_sq < 0.00955
        - 0.04139466 * max(0.0, 0.0896 - Q.mass_over_sum_pt) / 0.01728886   # -4.1%  mass_over_sum_pt < 0.0896
        - 0.03848529 * max(0.0, 17.1 - Q.n_dr_0p05_0p1) / 7.088632   # -3.8%  n_dr_0p05_0p1 < 17.1
        + 0.03804229 * max(0.0, 0.0793 - Q.mass_over_sum_pt) / 0.01143166   # +3.8%  mass_over_sum_pt < 0.0793
        - 0.03747538 * max(0.0, Q.n_particles - 46.2) / 6.111232   # -3.7%  n_particles > 46.2
        - 0.03529229 * max(0.0, Q.mass_over_sum_pt - 0.0807) / 0.01821716   # -3.5%  mass_over_sum_pt > 0.0807
        - 0.03433322 * max(0.0, 0.00837 - Q.lam1) / 0.003041579   # -3.4%  lam1 < 0.00837
        + 0.0334583 * max(0.0, 46.0 - Q.n_particles) / 6.392659   # +3.3%  n_particles < 46
        - 0.02647218 * max(0.0, 0.00873 - Q.girth2_top40) / 0.003025688   # -2.6%  girth2_top40 < 0.00873
        + 0.02512292 * max(0.0, Q.mass_top30 - 50.9) / 32.65023   # +2.5%  mass_top30 > 50.9
        + 0.02490137 * max(0.0, Q.n_dr_0p1_0p2 - 10.1) / 4.687775   # +2.5%  n_dr_0p1_0p2 > 10.1
        + 0.0247239 * max(0.0, Q.girth - 0.0339) / 0.03907369   # +2.5%  girth > 0.0339
        + 0.02263021 * max(0.0, 0.00728 - Q.lam1) / 0.002263242   # +2.3%  lam1 < 0.00728
        + 0.02150987 * max(0.0, Q.n_dr_0_0p05 - 12.2) / 3.878215   # +2.2%  n_dr_0_0p05 > 12.2
        - 0.01963403 * max(0.0, 12.0 - Q.n_dr_0_0p05) / 3.466761   # -2.0%  n_dr_0_0p05 < 12
        - 0.01849683 * max(0.0, 0.091 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.914) / 0.00135692   # -1.8%  mass_over_sum_pt < 0.091 and psi_0p2 > 0.914
        - 0.01819807 * max(0.0, Q.mass - 84.2) / 18.05877   # -1.8%  mass > 84.2
        + 0.01614465 * max(0.0, 81.3 - Q.mass_top40) / 12.64048   # +1.6%  mass_top40 < 81.3
        + 0.01506358 * max(0.0, Q.lam1 - 0.00806) / 0.002641543   # +1.5%  lam1 > 0.00806
        - 0.01399135 * max(0.0, 10.4 - Q.n_dr_0p1_0p2) / 2.372275   # -1.4%  n_dr_0p1_0p2 < 10.4
        + 0.01254508 * max(0.0, Q.psi_0p3 - 0.992) / 0.004328643   # +1.3%  psi_0p3 > 0.992
        - 0.01193771 * max(0.0, 122.0 - Q.mass) / 39.95231   # -1.2%  mass < 122
        - 0.01193393 * max(0.0, Q.e2 - 0.0152) / 0.01706918   # -1.2%  e2 > 0.0152
        + 0.00990045 * max(0.0, 1050.0 - Q.sum_pt_top40) / 55.46527   # +1.0%  sum_pt_top40 < 1050
        + 0.008611189 * max(0.0, Q.n_dr_0p05_0p1 - 16.9) / 1.597594   # +0.9%  n_dr_0p05_0p1 > 16.9
        - 0.008151393 * max(0.0, 734.0 - Q.sum_pt_top5) / 165.6318   # -0.8%  sum_pt_top5 < 734
        + 0.007926163 * max(0.0, 6.86 - Q.log_sum_pt) / 0.008420316   # +0.8%  log_sum_pt < 6.86
        - 0.007806003 * max(0.0, Q.sum_pt_top50 - 1150.0) / 14.37793   # -0.8%  sum_pt_top50 > 1150
        - 0.007449713 * max(0.0, 953.0 - Q.sum_pt) / 7.22467   # -0.7%  sum_pt < 953
        + 0.007447902 * max(0.0, Q.psi_0p3 - 0.965) / 0.02685703   # +0.7%  psi_0p3 > 0.965
        + 0.007065156 * max(0.0, 82.5 - Q.mass_top30) / 15.86716   # +0.7%  mass_top30 < 82.5
        + 0.007042674 * max(0.0, 1020.0 - Q.sum_pt_top50) / 27.07359   # +0.7%  sum_pt_top50 < 1020
        + 0.00692762 * max(0.0, Q.sum_pt - 1170.0) / 13.9657   # +0.7%  sum_pt > 1170
        + 0.006468976 * max(0.0, 66.4 - Q.mass_top5) / 40.19951   # +0.6%  mass_top5 < 66.4
        + 0.006089716 * max(0.0, 86.1 - Q.mass_top10) / 40.60207   # +0.6%  mass_top10 < 86.1
        - 0.005829804 * max(0.0, Q.LHA - 0.325) / 0.0155964   # -0.6%  LHA > 0.325
        - 0.005675329 * max(0.0, 0.000109 - Q.e3) / 5.955026e-05   # -0.6%  e3 < 0.000109
        + 0.0045687 * max(0.0, 0.0056 - Q.girth2_top20) / 0.001570069   # +0.5%  girth2_top20 < 0.0056
        - 0.004287447 * max(0.0, 971.0 - Q.sum_pt_top40) / 17.23224   # -0.4%  sum_pt_top40 < 971
        + 0.004180202 * max(0.0, 63.7 - Q.sd_mass) / 21.49069   # +0.4%  sd_mass < 63.7
        - 0.003783621 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.185 - Q.sd_rg) / 0.0002356938   # -0.4%  psi_0p3 > 0.993 and sd_rg < 0.185
        + 0.003229525 * max(0.0, Q.n_dr_0p2_0p4 - 21.4) / 0.6079701   # +0.3%  n_dr_0p2_0p4 > 21.4
        + 0.003023545 * max(0.0, Q.mass_top20 - 85.4) / 7.167622   # +0.3%  mass_top20 > 85.4
        + 0.002741913 * max(0.0, Q.lam2 - 0.00294) / 0.0004081385   # +0.3%  lam2 > 0.00294
        - 0.002635147 * max(0.0, Q.sum_pt_top15 - 884.0) / 51.11088   # -0.3%  sum_pt_top15 > 884
        - 0.001357735 * max(0.0, 0.972 - Q.z_top50_slots) / 0.00161681   # -0.1%  z_top50_slots < 0.972
        + 0.0004383067 * max(0.0, Q.sum_pt_top15 - 1050.0) / 7.958686   # +0.0%  sum_pt_top15 > 1050
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 11.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.5575 * (-0.1081549
        + 0.1870864 * max(0.0, 0.0827 - Q.girth) / 0.02511325   # +18.7%  girth < 0.0827
        - 0.09185243 * max(0.0, 0.296 - Q.LHA) / 0.05832881   # -9.2%  LHA < 0.296
        + 0.0811956 * max(0.0, Q.LHA - 0.178) / 0.09684396   # +8.1%  LHA > 0.178
        + 0.06023196 * max(0.0, 7.01 - Q.log_sum_pt) / 0.08287272   # +6.0%  log_sum_pt < 7.01
        - 0.04348058 * max(0.0, 1060.0 - Q.sum_pt_top50) / 53.00916   # -4.3%  sum_pt_top50 < 1060
        - 0.04030455 * max(0.0, Q.girth - 0.0857) / 0.01085827   # -4.0%  girth > 0.0857
        + 0.03767153 * max(0.0, 0.205 - Q.sd_rg) / 0.08503685   # +3.8%  sd_rg < 0.205
        - 0.03476545 * max(0.0, Q.sd_mass - 52.3) / 20.81874   # -3.5%  sd_mass > 52.3
        + 0.03402287 * max(0.0, 0.441 - Q.z_dr_0p1_0p2) / 0.2934472   # +3.4%  z_dr_0p1_0p2 < 0.441
        - 0.03145302 * max(0.0, Q.n_particles - 34.6) / 13.17095   # -3.1%  n_particles > 34.6
        + 0.03136149 * max(0.0, 6.97 - Q.log_sum_pt) / 0.05090736   # +3.1%  log_sum_pt < 6.97
        - 0.02974128 * max(0.0, 0.00832 - Q.width) / 0.002546184   # -3.0%  width < 0.00832
        - 0.02561685 * max(0.0, 0.166 - Q.sd_rg) / 0.05851121   # -2.6%  sd_rg < 0.166
        + 0.0253508 * max(0.0, 0.00229 - Q.lam2) / 0.001375548   # +2.5%  lam2 < 0.00229
        - 0.02528847 * max(0.0, Q.z_top50_slots - 0.973) / 0.02102672   # -2.5%  z_top50_slots > 0.973
        + 0.02180035 * max(0.0, Q.sd_rg - 0.217) / 0.01749705   # +2.2%  sd_rg > 0.217
        - 0.02064961 * max(0.0, 54.7 - Q.n_pt_above_1) / 14.73197   # -2.1%  n_pt_above_1 < 54.7
        + 0.0198326 * max(0.0, 0.00895 - Q.girth2_top10) / 0.004468133   # +2.0%  girth2_top10 < 0.00895
        - 0.01908874 * max(0.0, 6.91 - Q.log_sum_pt) / 0.01723579   # -1.9%  log_sum_pt < 6.91
        - 0.01719414 * max(0.0, Q.e2 - 0.0299) / 0.007613842   # -1.7%  e2 > 0.0299
        + 0.01715414 * max(0.0, Q.e3 - 4.09e-06) / 9.863629e-05   # +1.7%  e3 > 4.09e-06
        + 0.01691338 * max(0.0, Q.mass_top30 - 76.7) / 15.27159   # +1.7%  mass_top30 > 76.7
        - 0.01661916 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # -1.7%  psi_0p3 > 0.99
        - 0.01590739 * max(0.0, 86.8 - Q.mass_top50) / 14.47635   # -1.6%  mass_top50 < 86.8
        + 0.01004962 * max(0.0, Q.sd_mass - 86.7) / 6.211149   # +1.0%  sd_mass > 86.7
        - 0.00823363 * max(0.0, Q.sd_rg - 0.279) / 0.006945998   # -0.8%  sd_rg > 0.279
        + 0.006598315 * max(0.0, 35.9 - Q.n_particles) / 2.317934   # +0.7%  n_particles < 35.9
        - 0.004952003 * max(0.0, 0.00282 - Q.girth2_top10) / 0.0007905079   # -0.5%  girth2_top10 < 0.00282
        + 0.004752539 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0393 - Q.C2_b2) / 0.0004009304   # +0.5%  sj2_dr > 0.225 and C2_b2 < 0.0393
        - 0.003955861 * max(0.0, Q.z_dr_0_0p05 - 0.907) / 0.009524971   # -0.4%  z_dr_0_0p05 > 0.907
        - 0.003513095 * max(0.0, Q.psi_0p2 - 0.996) / 0.0006905202   # -0.4%  psi_0p2 > 0.996
        + 0.003178241 * max(0.0, Q.n_dr_0p1_0p2 - 16.7) / 2.239788   # +0.3%  n_dr_0p1_0p2 > 16.7
        - 0.002907149 * max(0.0, Q.e2 - 0.0423) / 0.003054488   # -0.3%  e2 > 0.0423
        + 0.002150542 * max(0.0, 893.0 - Q.sum_pt_top40) / 5.780206   # +0.2%  sum_pt_top40 < 893
        + 0.002090217 * max(0.0, 0.897 - Q.z_dr_0_0p05) * max(0.0, 1000.0 - Q.sum_pt) / 7.043056   # +0.2%  z_dr_0_0p05 < 0.897 and sum_pt < 1000
        + 0.001818709 * max(0.0, 0.0013 - Q.e2_sq) / 5.822639e-05   # +0.2%  e2_sq < 0.0013
        + 0.0007129538 * max(0.0, 0.102 - Q.LHA) / 0.001267687   # +0.1%  LHA < 0.102
        + 0.0005043452 * max(0.0, 0.00303 - Q.girth2_top20) * max(0.0, Q.sum_pt - 1120.0) / 0.02224797   # +0.1%  girth2_top20 < 0.00303 and sum_pt > 1120
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.690141544117647, 2.7694844537815126, 0.2128518907563025, 0.38288660714285716, 0.7710797268907563, 1.0395954831932772, 0.5777781512605042, 0.48586974789915965, 1.1746807773109245, 1.0047840336134455, 1.25500756302521, 0.7706998949579832, 0.5542981092436975, 1.5282901260504203, 0.2734795693277311, 0.44466018907563026]
T = [3.812262578781512, 2.477915849855567, 4.280810620732668, 4.435643654805673, 4.097610976070115]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -18%, n9 +14%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4540421 * h[1] / H_AVG[1]
            - 0.1769801 * h[4] / H_AVG[4]
            + 0.1359014 * h[9] / H_AVG[9]
            - 0.08521805 * h[5] / H_AVG[5]
            - 0.07532665 * h[3] / H_AVG[3]
            + 0.03407783 * h[12] / H_AVG[12]
            + 0.02368091 * h[6] / H_AVG[6]
            + 0.009629131 * h[8] / H_AVG[8]
            - 0.005143794 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -33%, n9 +23%, n1 -21%, n11 -8%, n12 +8%, n6 +5% ...
            - 0.3306296 * h[4] / H_AVG[4]
            + 0.2280913 * h[9] / H_AVG[9]
            - 0.2095625 * h[1] / H_AVG[1]
            - 0.07775687 * h[11] / H_AVG[11]
            + 0.07689526 * h[12] / H_AVG[12]
            + 0.05100616 * h[6] / H_AVG[6]
            + 0.01073745 * h[2] / H_AVG[2]
            - 0.007913704 * h[10] / H_AVG[10]
            + 0.007407187 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -24%, n5 +14%, n0 +12%, n11 +11%, n14 -9%, n7 -7% ...
            - 0.2401054 * h[8] / H_AVG[8]
            + 0.1403977 * h[5] / H_AVG[5]
            + 0.1209131 * h[0] / H_AVG[0]
            + 0.1068964 * h[11] / H_AVG[11]
            - 0.08784187 * h[14] / H_AVG[14]
            - 0.07093717 * h[7] / H_AVG[7]
            + 0.06191787 * h[4] / H_AVG[4]
            - 0.05260303 * h[12] / H_AVG[12]
            - 0.0513446 * h[9] / H_AVG[9]
            + 0.03913111 * h[3] / H_AVG[3]
            - 0.01947617 * h[15] / H_AVG[15]
            + 0.008435583 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n0 -21%, n5 +16%, n6 -13%, n7 +10%, n15 +6% ...
            - 0.2482759 * h[8] / H_AVG[8]
            - 0.2139362 * h[0] / H_AVG[0]
            + 0.1611315 * h[5] / H_AVG[5]
            - 0.1261875 * h[6] / H_AVG[6]
            + 0.09926845 * h[7] / H_AVG[7]
            + 0.05638897 * h[15] / H_AVG[15]
            + 0.03905141 * h[12] / H_AVG[12]
            + 0.03776518 * h[3] / H_AVG[3]
            - 0.01799501 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -12%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.338005 * h[13] / H_AVG[13]
            + 0.3014923 * h[10] / H_AVG[10]
            - 0.1189255 * h[5] / H_AVG[5]
            + 0.06047041 * h[8] / H_AVG[8]
            - 0.05072756 * h[12] / H_AVG[12]
            - 0.04069385 * h[15] / H_AVG[15]
            + 0.03528335 * h[4] / H_AVG[4]
            - 0.03334891 * h[7] / H_AVG[7]
            + 0.02105317 * h[0] / H_AVG[0]
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
