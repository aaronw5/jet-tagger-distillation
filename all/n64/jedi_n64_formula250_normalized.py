"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  12.9%   (on for 53% of jets)
  neuron  1:  12.7%   (on for 96% of jets)
  neuron  5:  11.9%   (on for 80% of jets)
  neuron  4:   9.6%   (on for 65% of jets)
  neuron 13:   7.7%   (on for 87% of jets)
  neuron  0:   7.4%   (on for 62% of jets)
  neuron  9:   7.1%   (on for 60% of jets)
  neuron 10:   6.5%   (on for 77% of jets)
  neuron 12:   4.5%   (on for 46% of jets)
  neuron  6:   3.9%   (on for 55% of jets)
  neuron  7:   3.8%   (on for 50% of jets)
  neuron 11:   3.5%   (on for 74% of jets)
  neuron  3:   3.4%   (on for 64% of jets)
  neuron 15:   2.3%   (on for 54% of jets)
  neuron 14:   2.1%   (on for 37% of jets)
  neuron  2:   0.6%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.7% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.z_2nd                  2nd-largest pT share
  Q.pt2_over_pt0           pT2 / pT0
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
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
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top20=mass_of(20),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
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
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 8.437;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.4368 * (0.09458563
        - 0.3940619 * max(0.0, Q.mass - 78.3) / 21.31167   # -39.4%  mass > 78.3
        + 0.1250752 * max(0.0, 0.00766 - Q.mass_over_sum_pt_sq) / 0.002106256   # +12.5%  mass_over_sum_pt_sq < 0.00766
        + 0.09962899 * max(0.0, Q.mass_top50 - 82.0) / 17.73312   # +10.0%  mass_top50 > 82
        - 0.05928119 * max(0.0, Q.mass - 93.8) / 14.20862   # -5.9%  mass > 93.8
        + 0.04928983 * max(0.0, 1160.0 - Q.sum_pt_top50) / 137.6982   # +4.9%  sum_pt_top50 < 1160
        - 0.04792143 * max(0.0, 0.00614 - Q.sum_zz_dr2) / 0.001283503   # -4.8%  sum_zz_dr2 < 0.00614
        + 0.04523473 * max(0.0, 15.3 - Q.n_dr_0p2_0p4) / 7.725433   # +4.5%  n_dr_0p2_0p4 < 15.3
        - 0.04228515 * max(0.0, 99.1 - Q.mass) / 22.15847   # -4.2%  mass < 99.1
        - 0.03296226 * max(0.0, 0.0913 - Q.z_dr_0p2_0p4) / 0.05830105   # -3.3%  z_dr_0p2_0p4 < 0.0913
        - 0.02874059 * max(0.0, 0.0699 - Q.tau1) / 0.01317818   # -2.9%  tau1 < 0.0699
        - 0.01919853 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -1.9%  sum_pt < 1010
        - 0.01818374 * max(0.0, 0.00652 - Q.sum_z_dr2_top20) / 0.002075948   # -1.8%  sum_z_dr2_top20 < 0.00652
        + 0.01404384 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 963.0) / 2.093376   # +1.4%  log_sum_pt < 7 and sum_pt_top50 > 963
        - 0.01276671 * max(0.0, 0.991 - Q.z_top50_slots) / 0.004873763   # -1.3%  z_top50_slots < 0.991
        + 0.01132597 * max(0.0, Q.psi_0p3 - 0.996) / 0.001712455   # +1.1%  psi_0p3 > 0.996
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 16.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.6467 * (0.06367627
        - 0.1286322 * max(0.0, 2.27 - Q.soft1_pt) / 1.647156   # -12.9%  soft1_pt < 2.27
        + 0.09342863 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # +9.3%  log_sum_pt > 6.91
        + 0.08852724 * max(0.0, Q.pt_entropy - 2.0) / 0.872004   # +8.9%  pt_entropy > 2
        + 0.07952778 * max(0.0, 0.196 - Q.tau1) / 0.1103229   # +8.0%  tau1 < 0.196
        + 0.07829328 * max(0.0, Q.log_sum_pt - 6.89) / 0.0671817   # +7.8%  log_sum_pt > 6.89
        + 0.07064293 * max(0.0, 1.5 - Q.soft1_pt) / 0.9333111   # +7.1%  soft1_pt < 1.5
        - 0.06285359 * max(0.0, Q.sum_pt_top50 - 953.0) / 91.78116   # -6.3%  sum_pt_top50 > 953
        - 0.05034207 * max(0.0, Q.z_top50_slots - 0.96) / 0.03299329   # -5.0%  z_top50_slots > 0.96
        - 0.04654283 * max(0.0, Q.n_for_90pct - 10.4) / 10.85133   # -4.7%  n_for_90pct > 10.4
        - 0.03555835 * max(0.0, Q.log_sum_pt - 6.97) / 0.0255142   # -3.6%  log_sum_pt > 6.97
        + 0.03336743 * max(0.0, Q.n_particles - 38.4) / 10.53999   # +3.3%  n_particles > 38.4
        + 0.03192736 * max(0.0, 1070.0 - Q.sum_pt_top40) / 70.95933   # +3.2%  sum_pt_top40 < 1070
        - 0.03192017 * max(0.0, 122.0 - Q.mass) / 39.95231   # -3.2%  mass < 122
        + 0.02535667 * max(0.0, 0.258 - Q.LHA) / 0.03872523   # +2.5%  LHA < 0.258
        - 0.02372025 * max(0.0, 29.5 - Q.sj3_mass1) * max(0.0, 21.0 - Q.sj3_mass2) / 193.5608   # -2.4%  sj3_mass1 < 29.5 and sj3_mass2 < 21
        + 0.01103984 * max(0.0, 0.00509 - Q.lam1) / 0.001107091   # +1.1%  lam1 < 0.00509
        - 0.01037868 * max(0.0, 49.0 - Q.mass_top20) / 6.375307   # -1.0%  mass_top20 < 49
        + 0.009880568 * max(0.0, Q.z_top30_slots - 0.939) * max(0.0, 0.075 - Q.C2) / 0.0008223945   # +1.0%  z_top30_slots > 0.939 and C2 < 0.075
        + 0.00936881 * max(0.0, Q.n_particles - 39.4) * max(0.0, 0.111 - Q.dr_0) / 0.5113436   # +0.9%  n_particles > 39.4 and dr_0 < 0.111
        - 0.009120203 * max(0.0, 0.032 - Q.M3) / 0.009257398   # -0.9%  M3 < 0.032
        - 0.008513043 * max(0.0, 6.57 - Q.n_dr_0p2_0p4) / 1.823862   # -0.9%  n_dr_0p2_0p4 < 6.57
        - 0.00713271 * max(0.0, 31.6 - Q.pt_9) / 6.708255   # -0.7%  pt_9 < 31.6
        - 0.007071982 * max(0.0, Q.z_top20_slots - 0.887) * max(0.0, 0.0414 - Q.dr_2) / 0.0006727154   # -0.7%  z_top20_slots > 0.887 and dr_2 < 0.0414
        + 0.006973292 * max(0.0, 50.6 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.8) / 41.16395   # +0.7%  mass_top20 < 50.6 and n_real_top40 > 29.8
        + 0.00667416 * max(0.0, Q.z_top30_slots - 0.942) * max(0.0, Q.max_pair_mass - 9.45) / 0.1966421   # +0.7%  z_top30_slots > 0.942 and max_pair_mass > 9.45
        - 0.006569757 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # -0.7%  psi_0p3 > 0.998
        - 0.006359597 * max(0.0, Q.log_sum_pt - 7.07) / 0.01017945   # -0.6%  log_sum_pt > 7.07
        - 0.006161316 * max(0.0, Q.z_dr_0_0p05 - 0.875) / 0.01681403   # -0.6%  z_dr_0_0p05 > 0.875
        + 0.005051579 * max(0.0, 0.000986 - Q.sum_z_dr2_top15) / 0.0001318059   # +0.5%  sum_z_dr2_top15 < 0.000986
        - 0.004426947 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.4%  n_dr_0p1_0p2 < 8
        + 0.003157633 * max(0.0, Q.z_top30_slots - 0.914) * max(0.0, Q.ptdr0_3 - 6.21) / 0.09031647   # +0.3%  z_top30_slots > 0.914 and ptdr0_3 > 6.21
        + 0.001479063 * max(0.0, Q.sum_pt_top30 - 1190.0) / 7.03472   # +0.1%  sum_pt_top30 > 1190
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.2339;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.2339364 * (0.04035284
        + 1.0 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 0.0272 - Q.sum_z_dr2_top15) / 0.001135614   # +100.0%  log_sum_pt > 6.91 and sum_z_dr2_top15 < 0.0272
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.086;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.085912 * (-0.05976545
        + 0.1815954 * max(0.0, 46.1 - Q.n_particles) / 6.444332   # +18.2%  n_particles < 46.1
        - 0.1654543 * max(0.0, 69.2 - Q.mass_top50) / 7.455136   # -16.5%  mass_top50 < 69.2
        + 0.1638386 * max(0.0, 0.000625 - Q.lam2) / 0.0001547081   # +16.4%  lam2 < 0.000625
        + 0.1295294 * max(0.0, 7.15 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.98) / 0.03801554   # +13.0%  n_dr_0p2_0p4 < 7.15 and z_top50_slots > 0.98
        + 0.1235621 * max(0.0, 0.00669 - Q.sum_z_dr2_top40) / 0.001689894   # +12.4%  sum_z_dr2_top40 < 0.00669
        - 0.1179771 * max(0.0, 0.375 - Q.tau21) / 0.06405638   # -11.8%  tau21 < 0.375
        + 0.06827162 * max(0.0, 11.2 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.934) / 0.1252313   # +6.8%  n_dr_0p1_0p2 < 11.2 and psi_0p2 > 0.934
        - 0.04977139 * max(0.0, 0.406 - Q.tau21) * max(0.0, Q.lam1 - 0.00572) / 0.0002525577   # -5.0%  tau21 < 0.406 and lam1 > 0.00572
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 7.62;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.620463 * (0.1456604
        - 0.1456527 * max(0.0, 86.2 - Q.mass) / 13.56896   # -14.6%  mass < 86.2
        - 0.1447274 * max(0.0, 166.0 - Q.mass_top40) / 81.69555   # -14.5%  mass_top40 < 166
        + 0.1374725 * max(0.0, 119.0 - Q.mass) / 37.54852   # +13.7%  mass < 119
        - 0.07672557 * max(0.0, Q.n_particles - 23.1) / 22.9288   # -7.7%  n_particles > 23.1
        + 0.06843291 * max(0.0, Q.sum_zz_dr2 - 0.0103) / 0.003014396   # +6.8%  sum_zz_dr2 > 0.0103
        - 0.06642201 * max(0.0, 78.4 - Q.mass) / 9.90541   # -6.6%  mass < 78.4
        + 0.06292818 * max(0.0, 24.2 - Q.n_dr_0p2_0p4) / 15.51915   # +6.3%  n_dr_0p2_0p4 < 24.2
        + 0.06112197 * max(0.0, 67.5 - Q.mass_top40) / 7.635699   # +6.1%  mass_top40 < 67.5
        + 0.02910111 * max(0.0, Q.n_particles - 20.4) * max(0.0, 2.44 - Q.soft1_pt) / 42.89437   # +2.9%  n_particles > 20.4 and soft1_pt < 2.44
        + 0.02791196 * max(0.0, 101.0 - Q.mass) * max(0.0, 3.79 - Q.D2_b2) / 24.33662   # +2.8%  mass < 101 and D2_b2 < 3.79
        + 0.02764498 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +2.8%  psi_0p3 > 0.997
        - 0.0268417 * max(0.0, Q.sum_z_dr2_top15 - 0.00752) / 0.002852806   # -2.7%  sum_z_dr2_top15 > 0.00752
        - 0.02483399 * max(0.0, Q.lam1 - 0.00794) / 0.002672973   # -2.5%  lam1 > 0.00794
        - 0.02009518 * max(0.0, 88.5 - Q.mass_top40) * max(0.0, 3.21 - Q.D2_b2) / 8.233042   # -2.0%  mass_top40 < 88.5 and D2_b2 < 3.21
        - 0.01994229 * max(0.0, 0.00531 - Q.sum_z_dr2_top15) / 0.001630573   # -2.0%  sum_z_dr2_top15 < 0.00531
        + 0.01320347 * max(0.0, Q.sum_z_dr2_top15 - 0.016) / 0.00128174   # +1.3%  sum_z_dr2_top15 > 0.016
        + 0.009361076 * max(0.0, Q.n_particles - 27.6) * max(0.0, Q.n_dr_0_0p05 - 10.7) / 108.7435   # +0.9%  n_particles > 27.6 and n_dr_0_0p05 > 10.7
        - 0.00891862 * max(0.0, Q.sj2_dr - 0.232) * max(0.0, 20.6 - Q.D2_b2) / 0.35033   # -0.9%  sj2_dr > 0.232 and D2_b2 < 20.6
        - 0.008584292 * max(0.0, Q.sum_z_dr - 0.121) / 0.00424781   # -0.9%  sum_z_dr > 0.121
        - 0.007167831 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 13.2 - Q.n_dr_0_0p05) / 0.004749756   # -0.7%  psi_0p3 > 0.997 and n_dr_0_0p05 < 13.2
        - 0.005539927 * max(0.0, 22.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.1) / 37.69358   # -0.6%  n_dr_0p2_0p4 < 22.7 and n_dr_0p1_0p2 > 12.1
        - 0.004320911 * max(0.0, Q.C2 - 0.0725) / 0.01291268   # -0.4%  C2 > 0.0725
        - 0.003049466 * max(0.0, Q.e2 - 0.0568) / 0.001001653   # -0.3%  e2 > 0.0568
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 9.468;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.4683 * (-0.03960584
        + 0.1826644 * max(0.0, Q.sum_pt_top50 - 943.0) / 100.5535   # +18.3%  sum_pt_top50 > 943
        + 0.1579409 * max(0.0, Q.sum_pt - 908.0) / 139.7599   # +15.8%  sum_pt > 908
        - 0.1259339 * max(0.0, Q.log_sum_pt - 6.91) / 0.05184263   # -12.6%  log_sum_pt > 6.91
        - 0.1104291 * max(0.0, Q.sum_pt - 990.0) / 66.59719   # -11.0%  sum_pt > 990
        + 0.08732911 * max(0.0, Q.mass - 73.6) / 24.53585   # +8.7%  mass > 73.6
        - 0.04755538 * max(0.0, Q.mass_over_sum_pt - 0.0781) / 0.0195769   # -4.8%  mass_over_sum_pt > 0.0781
        - 0.04421736 * max(0.0, Q.mass_top50 - 158.0) / 1.51142   # -4.4%  mass_top50 > 158
        - 0.03397564 * max(0.0, Q.mass - 92.4) / 14.62235   # -3.4%  mass > 92.4
        + 0.03221572 * max(0.0, Q.sd_mass - 72.1) / 10.66532   # +3.2%  sd_mass > 72.1
        + 0.02759302 * max(0.0, 0.459 - Q.max_dr) / 0.1126117   # +2.8%  max_dr < 0.459
        + 0.02296775 * max(0.0, 47.5 - Q.n_particles) / 7.177081   # +2.3%  n_particles < 47.5
        - 0.02286455 * max(0.0, Q.sd_mass - 86.4) / 6.275027   # -2.3%  sd_mass > 86.4
        - 0.0185044 * max(0.0, 56.1 - Q.n_particles) * max(0.0, 2.7 - Q.D2) / 11.15957   # -1.9%  n_particles < 56.1 and D2 < 2.7
        + 0.01751016 * max(0.0, 12.4 - Q.n_dr_0p2_0p4) / 5.489783   # +1.8%  n_dr_0p2_0p4 < 12.4
        - 0.01547338 * max(0.0, Q.mass - 171.0) / 0.827721   # -1.5%  mass > 171
        - 0.01524877 * max(0.0, 22.6 - Q.n_dr_0p1_0p2) / 11.19224   # -1.5%  n_dr_0p1_0p2 < 22.6
        - 0.01449592 * max(0.0, Q.sum_z_dr2_top50 - 0.0191) / 0.001270849   # -1.4%  sum_z_dr2_top50 > 0.0191
        - 0.01223972 * max(0.0, 0.561 - Q.tau21) / 0.1667473   # -1.2%  tau21 < 0.561
        + 0.01084085 * max(0.0, Q.log_sum_pt - 6.99) / 0.0210337   # +1.1%  log_sum_pt > 6.99
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 6.567;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.566892 * (0.1343101
        - 0.3891102 * max(0.0, 102.0 - Q.mass) / 24.33566   # -38.9%  mass < 102
        + 0.2789848 * max(0.0, 93.9 - Q.mass) / 18.33897   # +27.9%  mass < 93.9
        + 0.09368601 * max(0.0, Q.sum_z_dr2_top20 - 0.00733) / 0.003107201   # +9.4%  sum_z_dr2_top20 > 0.00733
        - 0.0668462 * max(0.0, Q.mass_over_sum_pt - 0.121) / 0.007196258   # -6.7%  mass_over_sum_pt > 0.121
        + 0.04049276 * max(0.0, 73.3 - Q.mass_top50) / 8.661614   # +4.0%  mass_top50 < 73.3
        - 0.03800944 * max(0.0, Q.e2 - 0.0294) / 0.00787394   # -3.8%  e2 > 0.0294
        - 0.03697341 * max(0.0, Q.sum_z_dr2_top20 - 0.0035) * max(0.0, Q.C2_b2 - 0.00379) / 0.0001042062   # -3.7%  sum_z_dr2_top20 > 0.0035 and C2_b2 > 0.00379
        + 0.01746573 * max(0.0, 0.0533 - Q.tau1) / 0.00754576   # +1.7%  tau1 < 0.0533
        + 0.01467879 * max(0.0, Q.e2 - 0.053) / 0.001394993   # +1.5%  e2 > 0.053
        - 0.007859551 * max(0.0, Q.sum_z_dr2_top30 - 0.0275) / 0.0002314476   # -0.8%  sum_z_dr2_top30 > 0.0275
        + 0.006142242 * max(0.0, 102.0 - Q.mass) * max(0.0, 1010.0 - Q.sum_pt) / 478.475   # +0.6%  mass < 102 and sum_pt < 1010
        - 0.004878534 * max(0.0, Q.sj3_pair_mass_min - 77.9) / 0.2692169   # -0.5%  sj3_pair_mass_min > 77.9
        + 0.004872333 * max(0.0, 0.931 - Q.z_top40_slots) / 0.002688747   # +0.5%  z_top40_slots < 0.931
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 27.41;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.41115 * (0.0006639633
        - 0.1897604 * max(0.0, 91.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978) / 0.3006677   # -19.0%  mass < 91 and psi_0p3 > 0.978
        + 0.1845564 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964) / 0.5308399   # +18.5%  mass < 91.2 and psi_0p3 > 0.964
        - 0.1772278 * max(0.0, 93.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964) / 0.574234   # -17.7%  mass < 93.2 and psi_0p3 > 0.964
        + 0.1207372 * max(0.0, 102.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978) / 0.4448315   # +12.1%  mass < 102 and psi_0p3 > 0.978
        + 0.06043613 * max(0.0, 82.3 - Q.mass) * max(0.0, Q.psi_0p3 - 0.978) / 0.2148669   # +6.0%  mass < 82.3 and psi_0p3 > 0.978
        - 0.04202056 * max(0.0, 0.107 - Q.tau1) / 0.03407787   # -4.2%  tau1 < 0.107
        + 0.03987841 * max(0.0, 101.0 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998) / 0.01743402   # +4.0%  mass < 101 and psi_0p3 > 0.998
        - 0.02744527 * max(0.0, 91.5 - Q.mass) * max(0.0, Q.psi_0p3 - 0.998) / 0.01160967   # -2.7%  mass < 91.5 and psi_0p3 > 0.998
        - 0.02634935 * max(0.0, 87.2 - Q.sd_mass) / 36.47809   # -2.6%  sd_mass < 87.2
        + 0.02408972 * max(0.0, 69.8 - Q.sd_mass) / 24.63907   # +2.4%  sd_mass < 69.8
        + 0.02299541 * max(0.0, 0.0957 - Q.tau1) / 0.0261548   # +2.3%  tau1 < 0.0957
        + 0.02227998 * max(0.0, 121.0 - Q.mass) / 39.14871   # +2.2%  mass < 121
        - 0.01372349 * max(0.0, Q.sd_mass - 97.2) / 4.404878   # -1.4%  sd_mass > 97.2
        + 0.01216155 * max(0.0, 0.234 - Q.tau21_b2) / 0.05168405   # +1.2%  tau21_b2 < 0.234
        - 0.01158013 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # -1.2%  psi_0p3 > 0.998
        - 0.007853909 * max(0.0, 0.231 - Q.tau21_b2) * max(0.0, 1270.0 - Q.sum_pt) / 11.70026   # -0.8%  tau21_b2 < 0.231 and sum_pt < 1270
        + 0.005907913 * max(0.0, 8.15 - Q.n_dr_0p2_0p4) / 2.672322   # +0.6%  n_dr_0p2_0p4 < 8.15
        + 0.005366342 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.z_top50_slots - 0.979) / 2.128765e-05   # +0.5%  psi_0p3 > 0.997 and z_top50_slots > 0.979
        - 0.002677896 * max(0.0, 0.23 - Q.tau21_b2) * max(0.0, 0.00808 - Q.mass_over_sum_pt_sq) / 5.560925e-05   # -0.3%  tau21_b2 < 0.23 and mass_over_sum_pt_sq < 0.00808
        - 0.002210678 * max(0.0, 93.2 - Q.mass) * max(0.0, 0.503 - Q.z_dr_0_0p05) / 0.6133324   # -0.2%  mass < 93.2 and z_dr_0_0p05 < 0.503
        - 0.0007414089 * max(0.0, 0.00796 - Q.lam1) * max(0.0, Q.zdr_0 - 0.0152) / 6.729428e-07   # -0.1%  lam1 < 0.00796 and zdr_0 > 0.0152
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 8.131;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.130906 * (0.1598838
        + 0.1408154 * max(0.0, Q.mass - 71.1) / 26.32084   # +14.1%  mass > 71.1
        - 0.09724466 * max(0.0, 0.00987 - Q.lam1_plus_lam2) / 0.0036948   # -9.7%  lam1_plus_lam2 < 0.00987
        + 0.09709237 * max(0.0, Q.sum_z_dr2_top20 - 0.00788) / 0.002945705   # +9.7%  sum_z_dr2_top20 > 0.00788
        - 0.09650519 * max(0.0, Q.z_top50_slots - 0.972) / 0.02191829   # -9.7%  z_top50_slots > 0.972
        - 0.09093242 * max(0.0, Q.sum_z_dr2_top20 - 0.0059) / 0.003660213   # -9.1%  sum_z_dr2_top20 > 0.0059
        - 0.08225845 * max(0.0, 17.9 - Q.n_dr_0p2_0p4) / 9.879406   # -8.2%  n_dr_0p2_0p4 < 17.9
        + 0.07410836 * max(0.0, Q.mass_over_sum_pt - 0.0785) * max(0.0, 1100.0 - Q.sum_pt) / 1.975633   # +7.4%  mass_over_sum_pt > 0.0785 and sum_pt < 1100
        + 0.06707873 * max(0.0, 1030.0 - Q.sum_pt) / 27.96979   # +6.7%  sum_pt < 1030
        - 0.05937883 * max(0.0, Q.mass - 103.0) / 11.83342   # -5.9%  mass > 103
        + 0.048434 * max(0.0, 0.00754 - Q.sum_z_dr2_top30) / 0.002386741   # +4.8%  sum_z_dr2_top30 < 0.00754
        - 0.04068951 * max(0.0, 1030.0 - Q.sum_pt_top40) / 41.61542   # -4.1%  sum_pt_top40 < 1030
        - 0.04009079 * max(0.0, Q.sum_z_dr2_top40 - 0.00555) * max(0.0, 6.99 - Q.log_sum_pt) / 0.0004024376   # -4.0%  sum_z_dr2_top40 > 0.00555 and log_sum_pt < 6.99
        - 0.01733753 * max(0.0, Q.mass - 127.0) / 6.744969   # -1.7%  mass > 127
        + 0.01699252 * max(0.0, 8.6 - Q.n_dr_0p2_0p4) / 2.939673   # +1.7%  n_dr_0p2_0p4 < 8.6
        + 0.01137194 * max(0.0, Q.sj2_dr - 0.23) / 0.02185914   # +1.1%  sj2_dr > 0.23
        + 0.007605528 * max(0.0, 0.0848 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.4) / 0.1534487   # +0.8%  tau2 < 0.0848 and sj3_mass1 > 13.4
        + 0.007589243 * max(0.0, Q.sum_pt_top30 - 987.0) / 44.07673   # +0.8%  sum_pt_top30 > 987
        + 0.004474521 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.313) / 0.175758   # +0.4%  n_dr_0p2_0p4 < 18 and z_dr_0p1_0p2 > 0.313
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 11.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.08569 * (-0.1019332
        + 0.3933255 * max(0.0, 178.0 - Q.mass) / 88.80418   # +39.3%  mass < 178
        - 0.3402353 * max(0.0, 147.0 - Q.mass) / 60.73662   # -34.0%  mass < 147
        + 0.1226114 * max(0.0, 85.5 - Q.mass) / 13.19643   # +12.3%  mass < 85.5
        + 0.03594398 * max(0.0, 0.00617 - Q.sum_z_dr2_top40) / 0.001418021   # +3.6%  sum_z_dr2_top40 < 0.00617
        - 0.03503316 * max(0.0, 62.9 - Q.mass) / 5.548098   # -3.5%  mass < 62.9
        - 0.02434062 * max(0.0, Q.z_top40_slots - 0.954) / 0.03108671   # -2.4%  z_top40_slots > 0.954
        + 0.01743657 * max(0.0, Q.sum_z_dr - 0.0976) / 0.008295986   # +1.7%  sum_z_dr > 0.0976
        - 0.01203712 * max(0.0, 97.9 - Q.mass_top40) * max(0.0, 1020.0 - Q.sum_pt) / 577.6614   # -1.2%  mass_top40 < 97.9 and sum_pt < 1020
        + 0.009299207 * max(0.0, 989.0 - Q.sum_pt) / 12.58708   # +0.9%  sum_pt < 989
        + 0.005376905 * max(0.0, 100.0 - Q.mass_top40) * max(0.0, 961.0 - Q.sum_pt_top40) / 322.1985   # +0.5%  mass_top40 < 100 and sum_pt_top40 < 961
        + 0.004360231 * max(0.0, 116.0 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 8.57) / 36.61832   # +0.4%  mass_top40 < 116 and n_dr_0p2_0p4 > 8.57
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.02;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.020321 * (0.03852714
        + 0.1803857 * max(0.0, 87.9 - Q.mass) / 14.4965   # +18.0%  mass < 87.9
        - 0.09916317 * max(0.0, 65.0 - Q.sj2_mass1) / 36.316   # -9.9%  sj2_mass1 < 65
        - 0.09158507 * max(0.0, 99.6 - Q.mass) / 22.53195   # -9.2%  mass < 99.6
        + 0.07059077 * max(0.0, Q.e2 - 0.0128) / 0.01893514   # +7.1%  e2 > 0.0128
        - 0.05991401 * max(0.0, 0.047 - Q.tau1) / 0.005810515   # -6.0%  tau1 < 0.047
        + 0.05728634 * max(0.0, 3.04 - Q.D2) / 0.9713633   # +5.7%  D2 < 3.04
        + 0.04893625 * max(0.0, 941.0 - Q.sum_pt_top15) / 99.61533   # +4.9%  sum_pt_top15 < 941
        - 0.04251678 * max(0.0, 0.00533 - Q.sum_z_dr2_top20) / 0.001444908   # -4.3%  sum_z_dr2_top20 < 0.00533
        - 0.0408194 * max(0.0, Q.mass - 145.0) / 3.767373   # -4.1%  mass > 145
        + 0.0375392 * max(0.0, 0.0485 - Q.tau2) / 0.02120256   # +3.8%  tau2 < 0.0485
        + 0.0368999 * max(0.0, Q.mass - 78.8) / 20.98929   # +3.7%  mass > 78.8
        + 0.03345943 * max(0.0, 0.0859 - Q.z_dr_0p2_0p4) / 0.05399504   # +3.3%  z_dr_0p2_0p4 < 0.0859
        - 0.03125866 * max(0.0, 11.1 - Q.n_dr_0p2_0p4) / 4.558263   # -3.1%  n_dr_0p2_0p4 < 11.1
        - 0.02618412 * max(0.0, 0.0234 - Q.sum_z_dr2_top10) * max(0.0, Q.psi_0p3 - 0.998) / 1.213902e-05   # -2.6%  sum_z_dr2_top10 < 0.0234 and psi_0p3 > 0.998
        + 0.02592081 * max(0.0, 0.0664 - Q.dr_0) / 0.02679036   # +2.6%  dr_0 < 0.0664
        + 0.02199867 * max(0.0, Q.mass_top50 - 138.0) / 4.065355   # +2.2%  mass_top50 > 138
        - 0.02142019 * max(0.0, Q.pt_dispersion - 0.27) / 0.06763653   # -2.1%  pt_dispersion > 0.27
        + 0.02109715 * max(0.0, Q.mass_top5 - 17.5) / 15.10767   # +2.1%  mass_top5 > 17.5
        - 0.01545749 * max(0.0, 984.0 - Q.sum_pt) / 11.58636   # -1.5%  sum_pt < 984
        - 0.009981729 * max(0.0, 2.99 - Q.D2) * max(0.0, Q.sj2_dr - 0.192) / 0.02493977   # -1.0%  D2 < 2.99 and sj2_dr > 0.192
        - 0.008358891 * max(0.0, Q.mass - 164.0) / 1.393784   # -0.8%  mass > 164
        + 0.006637867 * max(0.0, 63.3 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.52) / 165.3349   # +0.7%  sj2_mass1 < 63.3 and sj2_mass2 > 7.52
        - 0.005460646 * max(0.0, Q.mass_over_sum_pt - 0.16) / 0.001408236   # -0.5%  mass_over_sum_pt > 0.16
        - 0.005282918 * max(0.0, Q.sum_pt_top10 - 941.0) / 12.46197   # -0.5%  sum_pt_top10 > 941
        - 0.001623159 * max(0.0, Q.mass_top50 - 133.0) * max(0.0, Q.soft5_z - 0.00163) / 0.001810606   # -0.2%  mass_top50 > 133 and soft5_z > 0.00163
        - 0.0002216999 * max(0.0, Q.sum_pt_top10 - 943.0) * max(0.0, -0.078 - Q.eta_0) / 0.002027485   # -0.0%  sum_pt_top10 > 943 and eta_0 < -0.078
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 5.449;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.448507 * (-0.1055335
        + 0.3119509 * max(0.0, 94.7 - Q.mass_top50) / 19.62663   # +31.2%  mass_top50 < 94.7
        - 0.1645056 * max(0.0, 80.9 - Q.mass_top50) / 11.38894   # -16.5%  mass_top50 < 80.9
        - 0.1361952 * max(0.0, 0.00666 - Q.sum_z_dr2_top30) / 0.001845922   # -13.6%  sum_z_dr2_top30 < 0.00666
        + 0.09070484 * max(0.0, 0.0266 - Q.e2) / 0.005137276   # +9.1%  e2 < 0.0266
        + 0.07800721 * max(0.0, 7.5 - Q.n_dr_0p2_0p4) / 2.309907   # +7.8%  n_dr_0p2_0p4 < 7.5
        + 0.06816985 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # +6.8%  psi_0p3 > 0.99
        + 0.06422198 * max(0.0, 16.7 - Q.n_dr_0p1_0p2) / 6.37366   # +6.4%  n_dr_0p1_0p2 < 16.7
        - 0.031491 * max(0.0, 9.67 - Q.n_dr_0p2_0p4) * max(0.0, 0.0062 - Q.sum_z_dr2) / 0.006020314   # -3.1%  n_dr_0p2_0p4 < 9.67 and sum_z_dr2 < 0.0062
        - 0.02138946 * max(0.0, Q.z_top5_slots - 0.552) / 0.07470553   # -2.1%  z_top5_slots > 0.552
        - 0.01790221 * max(0.0, 8.65 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 36.1) / 19.12556   # -1.8%  n_dr_0p2_0p4 < 8.65 and n_particles > 36.1
        - 0.01546176 * max(0.0, 119.0 - Q.mass) * max(0.0, 872.0 - Q.sum_pt_top10) / 3178.999   # -1.5%  mass < 119 and sum_pt_top10 < 872
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.474;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.473608 * (-0.06104443
        + 0.5825264 * max(0.0, 83.4 - Q.mass) / 12.10876   # +58.3%  mass < 83.4
        - 0.1855505 * max(0.0, 59.7 - Q.mass) / 4.801038   # -18.6%  mass < 59.7
        - 0.133326 * max(0.0, 0.00586 - Q.sum_z_dr2_top15) / 0.001917421   # -13.3%  sum_z_dr2_top15 < 0.00586
        - 0.06682335 * max(0.0, 90.0 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3) / 0.04763538   # -6.7%  mass < 90 and psi_0p3 < 0.999
        - 0.01854552 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2) / 1.349245e-05   # -1.9%  log_sum_pt < 6.85 and lam2 < 0.0036
        + 0.00765295 * max(0.0, Q.sd_mass - 124.0) / 1.314611   # +0.8%  sd_mass > 124
        + 0.005575289 * max(0.0, Q.sj3_pair_mass_max - 123.0) * max(0.0, 65.0 - Q.sj3_pair_mass_min) / 31.55854   # +0.6%  sj3_pair_mass_max > 123 and sj3_pair_mass_min < 65
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.99966 * (0.6567411
        - 0.1662709 * max(0.0, 6.92 - Q.log_sum_pt) / 0.02069527   # -16.6%  log_sum_pt < 6.92
        + 0.1612063 * max(0.0, Q.mass - 63.1) / 32.2376   # +16.1%  mass > 63.1
        - 0.1451476 * max(0.0, Q.mass - 173.0) * max(0.0, Q.soft5_z - 0.000296) / 0.001323384   # -14.5%  mass > 173 and soft5_z > 0.000296
        - 0.1227835 * max(0.0, 1090.0 - Q.sum_pt) * max(0.0, 1.0 - Q.tau21_b2) / 47.21906   # -12.3%  sum_pt < 1090 and tau21_b2 < 1
        + 0.09605201 * max(0.0, 1020.0 - Q.sum_pt_top40) / 35.61475   # +9.6%  sum_pt_top40 < 1020
        - 0.06720925 * max(0.0, Q.mass - 142.0) * max(0.0, Q.D2 - 0.0302) / 8.960217   # -6.7%  mass > 142 and D2 > 0.0302
        - 0.06080112 * max(0.0, Q.mass - 101.0) / 12.32315   # -6.1%  mass > 101
        - 0.05665756 * max(0.0, Q.mass - 161.0) / 1.70294   # -5.7%  mass > 161
        + 0.04723171 * max(0.0, Q.mass - 169.0) * max(0.0, Q.D2 - 0.281) / 1.746967   # +4.7%  mass > 169 and D2 > 0.281
        - 0.03383248 * max(0.0, Q.mass_top50 - 134.0) * max(0.0, Q.soft6_z - 0.00094) / 0.003690397   # -3.4%  mass_top50 > 134 and soft6_z > 0.00094
        + 0.02052585 * max(0.0, Q.mass_top5 - 40.1) / 5.088478   # +2.1%  mass_top5 > 40.1
        - 0.0173605 * max(0.0, Q.mass_top50 - 90.5) * max(0.0, 0.00189 - Q.soft7_z) / 0.006861082   # -1.7%  mass_top50 > 90.5 and soft7_z < 0.00189
        + 0.004921274 * max(0.0, Q.log_sum_pt - 7.14) * max(0.0, Q.sd_zg - 0.446) / 1.294925e-05   # +0.5%  log_sum_pt > 7.14 and sd_zg > 0.446
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 9.202;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.201886 * (-0.04173057
        - 0.2032963 * max(0.0, 92.0 - Q.mass) / 17.00645   # -20.3%  mass < 92
        - 0.1757947 * max(0.0, 82.8 - Q.mass) / 11.80761   # -17.6%  mass < 82.8
        - 0.1322838 * max(0.0, 0.00262 - Q.sum_z_dr2_top30) / 0.0003612049   # -13.2%  sum_z_dr2_top30 < 0.00262
        + 0.1158647 * max(0.0, 0.144 - Q.mass_over_sum_pt) / 0.06057806   # +11.6%  mass_over_sum_pt < 0.144
        + 0.07770306 * max(0.0, 78.4 - Q.mass_top50) / 10.34754   # +7.8%  mass_top50 < 78.4
        + 0.05997243 * max(0.0, Q.psi_0p3 - 0.993) / 0.003630655   # +6.0%  psi_0p3 > 0.993
        - 0.05865296 * max(0.0, 0.0905 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.914) / 0.001332637   # -5.9%  mass_over_sum_pt < 0.0905 and psi_0p2 > 0.914
        + 0.04395313 * max(0.0, 0.0302 - Q.e2) / 0.006774736   # +4.4%  e2 < 0.0302
        + 0.02794612 * max(0.0, 0.0067 - Q.sum_z_dr2_top5) / 0.003296885   # +2.8%  sum_z_dr2_top5 < 0.0067
        - 0.01914633 * max(0.0, 0.00154 - Q.sum_z_dr2_top3) / 0.0004907586   # -1.9%  sum_z_dr2_top3 < 0.00154
        + 0.01893705 * max(0.0, 6.94 - Q.log_sum_pt) / 0.03062506   # +1.9%  log_sum_pt < 6.94
        - 0.01605231 * max(0.0, 0.484 - Q.N2) * max(0.0, Q.max_dr - 0.24) / 0.01951275   # -1.6%  N2 < 0.484 and max_dr > 0.24
        - 0.01437147 * max(0.0, 0.00654 - Q.sum_z_dr2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.01758573   # -1.4%  sum_z_dr2_top5 < 0.00654 and n_pt_above_10 > 13
        - 0.01114101 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.188 - Q.sd_rg) / 0.0002440912   # -1.1%  psi_0p3 > 0.993 and sd_rg < 0.188
        + 0.01084581 * max(0.0, Q.psi_0p2 - 0.978) * max(0.0, Q.pt2_over_pt0 - 0.121) / 0.00324032   # +1.1%  psi_0p2 > 0.978 and pt2_over_pt0 > 0.121
        - 0.00578457 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.124 - Q.z_2nd) / 5.668685e-05   # -0.6%  psi_0p3 > 0.993 and z_2nd < 0.124
        - 0.005590337 * max(0.0, 0.432 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -12.9) / 4.946312   # -0.6%  tau21_b2 < 0.432 and orientation_deg > -12.9
        - 0.002664008 * max(0.0, 0.322 - Q.tau21_b2) * max(0.0, 30.6 - Q.n_real_top40) / 0.1114268   # -0.3%  tau21_b2 < 0.322 and n_real_top40 < 30.6
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 4.768;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.767834 * (0.06753591
        + 0.164283 * max(0.0, 73.6 - Q.sd_mass) / 26.82445   # +16.4%  sd_mass < 73.6
        + 0.1608226 * max(0.0, 7.03 - Q.log_sum_pt) / 0.09997075   # +16.1%  log_sum_pt < 7.03
        - 0.1024248 * max(0.0, Q.psi_0p2 - 0.857) / 0.1015269   # -10.2%  psi_0p2 > 0.857
        - 0.09766549 * max(0.0, 44.2 - Q.sd_mass) / 12.89897   # -9.8%  sd_mass < 44.2
        - 0.09187851 * max(0.0, 1020.0 - Q.sum_pt_top40) / 35.61475   # -9.2%  sum_pt_top40 < 1020
        - 0.08323415 * max(0.0, Q.e2 - 0.0294) / 0.00787394   # -8.3%  e2 > 0.0294
        - 0.06568798 * max(0.0, Q.psi_0p3 - 0.989) / 0.0065384   # -6.6%  psi_0p3 > 0.989
        + 0.06262502 * max(0.0, 0.00181 - Q.lam2) / 0.0009789695   # +6.3%  lam2 < 0.00181
        + 0.04294913 * max(0.0, 13.6 - Q.n_dr_0p1_0p2) / 4.204811   # +4.3%  n_dr_0p1_0p2 < 13.6
        - 0.04021961 * max(0.0, 0.0678 - Q.tau1) / 0.01237164   # -4.0%  tau1 < 0.0678
        - 0.03271664 * max(0.0, Q.n_dr_0p2_0p4 - 8.79) / 3.318883   # -3.3%  n_dr_0p2_0p4 > 8.79
        + 0.02484633 * max(0.0, Q.sd_rg - 0.203) / 0.02078301   # +2.5%  sd_rg > 0.203
        + 0.01824926 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0493 - Q.C2_b2) / 0.0005919008   # +1.8%  sj2_dr > 0.225 and C2_b2 < 0.0493
        - 0.01239745 * max(0.0, 0.294 - Q.max_dr) / 0.01268433   # -1.2%  max_dr < 0.294
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6100095588235294, 2.8940838760504204, 0.23891008403361344, 0.3863388655462185, 0.7169547268907563, 1.072654306722689, 0.5167144957983193, 0.38580084033613443, 1.1532330882352941, 1.0163778361344538, 1.1739074579831932, 0.7575243697478992, 0.4992758403361345, 1.5604694327731092, 0.28714774159663864, 0.38011712184873947]
T = [3.8373015034138653, 2.416381446953782, 4.11665467601103, 4.136022295168067, 3.9647664415375528]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +47%, n4 -16%, n9 +14%, n5 -9%, n3 -8%, n12 +3% ...
            + 0.4713735 * h[1] / H_AVG[1]
            - 0.1634835 * h[4] / H_AVG[4]
            + 0.1365725 * h[9] / H_AVG[9]
            - 0.08735422 * h[5] / H_AVG[5]
            - 0.07550987 * h[3] / H_AVG[3]
            + 0.03049481 * h[12] / H_AVG[12]
            + 0.02103995 * h[6] / H_AVG[6]
            + 0.009391635 * h[8] / H_AVG[8]
            - 0.004780001 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -32%, n9 +24%, n1 -22%, n11 -8%, n12 +7%, n6 +5% ...
            - 0.3152501 * h[4] / H_AVG[4]
            + 0.2365986 * h[9] / H_AVG[9]
            - 0.2245675 * h[1] / H_AVG[1]
            - 0.07837384 * h[11] / H_AVG[11]
            + 0.07102607 * h[12] / H_AVG[12]
            + 0.04677709 * h[6] / H_AVG[6]
            + 0.01235888 * h[2] / H_AVG[2]
            - 0.007590815 * h[10] / H_AVG[10]
            + 0.007457129 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +15%, n0 +11%, n11 +11%, n14 -10%, n4 +6% ...
            - 0.2451211 * h[8] / H_AVG[8]
            + 0.1506389 * h[5] / H_AVG[5]
            + 0.1111357 * h[0] / H_AVG[0]
            + 0.1092586 * h[11] / H_AVG[11]
            - 0.09590995 * h[14] / H_AVG[14]
            + 0.05986735 * h[4] / H_AVG[4]
            - 0.05857317 * h[7] / H_AVG[7]
            - 0.05400809 * h[9] / H_AVG[9]
            - 0.04927079 * h[12] / H_AVG[12]
            + 0.0410584 * h[3] / H_AVG[3]
            - 0.01731308 * h[15] / H_AVG[15]
            + 0.007844879 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -20%, n5 +18%, n6 -12%, n7 +8%, n15 +5% ...
            - 0.2613999 * h[8] / H_AVG[8]
            - 0.2027946 * h[0] / H_AVG[0]
            + 0.1782993 * h[5] / H_AVG[5]
            - 0.1210262 * h[6] / H_AVG[6]
            + 0.0845334 * h[7] / H_AVG[7]
            + 0.05169602 * h[15] / H_AVG[15]
            + 0.04086614 * h[3] / H_AVG[3]
            + 0.03772313 * h[12] / H_AVG[12]
            - 0.02166122 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -36%, n10 +29%, n5 -13%, n8 +6%, n12 -5%, n15 -4% ...
            - 0.3566857 * h[13] / H_AVG[13]
            + 0.2914586 * h[10] / H_AVG[10]
            - 0.1268187 * h[5] / H_AVG[5]
            + 0.06135547 * h[8] / H_AVG[8]
            - 0.04722307 * h[12] / H_AVG[12]
            - 0.03595267 * h[15] / H_AVG[15]
            + 0.03390591 * h[4] / H_AVG[4]
            - 0.02736769 * h[7] / H_AVG[7]
            + 0.0192322 * h[0] / H_AVG[0]
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
