"""JEDI-linear jet tagger, 64 particles, 3 features: one term per observable per neuron (from the 250), re-tuned on the network's predictions (all observables), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  13.1%   (on for 47% of jets)
  neuron  4:  12.6%   (on for 69% of jets)
  neuron  1:  11.9%   (on for 98% of jets)
  neuron  5:   8.8%   (on for 75% of jets)
  neuron  9:   8.6%   (on for 82% of jets)
  neuron 13:   8.5%   (on for 84% of jets)
  neuron  0:   7.5%   (on for 59% of jets)
  neuron 10:   5.7%   (on for 87% of jets)
  neuron 12:   4.3%   (on for 46% of jets)
  neuron  7:   4.1%   (on for 51% of jets)
  neuron  3:   3.6%   (on for 57% of jets)
  neuron 11:   3.5%   (on for 75% of jets)
  neuron  6:   3.1%   (on for 54% of jets)
  neuron 15:   2.0%   (on for 54% of jets)
  neuron 14:   2.0%   (on for 31% of jets)
  neuron  2:   0.5%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 93.0% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
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
    # scale S = 7.752;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.75198 * (0.05339455
        - 0.5069092 * max(0.0, Q.mass - 82.83567) / 18.73092   # -50.7%  mass > 82.84
        + 0.1493606 * max(0.0, 0.00766 - Q.mass_over_sum_pt_sq) / 0.002106256   # +14.9%  mass_over_sum_pt_sq < 0.00766
        - 0.08697381 * max(0.0, 0.00614 - Q.sum_zz_dr2) / 0.001283503   # -8.7%  sum_zz_dr2 < 0.00614
        + 0.0527876 * max(0.0, Q.mass_top50 - 82.0) / 17.73312   # +5.3%  mass_top50 > 82
        + 0.04631684 * max(0.0, 15.3 - Q.n_dr_0p2_0p4) / 7.725433   # +4.6%  n_dr_0p2_0p4 < 15.3
        - 0.02588309 * max(0.0, 0.0913 - Q.z_dr_0p2_0p4) / 0.05830105   # -2.6%  z_dr_0p2_0p4 < 0.0913
        - 0.02468952 * max(0.0, 0.0699 - Q.tau1) / 0.01317818   # -2.5%  tau1 < 0.0699
        - 0.02297054 * max(0.0, 0.00652 - Q.sum_z_dr2_top20) / 0.002075948   # -2.3%  sum_z_dr2_top20 < 0.00652
        + 0.02259434 * max(0.0, 1160.0 - Q.sum_pt_top50) / 137.6982   # +2.3%  sum_pt_top50 < 1160
        + 0.02229592 * max(0.0, 7.0 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 963.0) / 2.093376   # +2.2%  log_sum_pt < 7 and sum_pt_top50 > 963
        + 0.01451854 * max(0.0, Q.psi_0p3 - 0.996) / 0.001712455   # +1.5%  psi_0p3 > 0.996
        - 0.0140868 * max(0.0, 0.991 - Q.z_top50_slots) / 0.004873763   # -1.4%  z_top50_slots < 0.991
        - 0.0106132 * max(0.0, 1010.0 - Q.sum_pt) / 18.51133   # -1.1%  sum_pt < 1010
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 11.62;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.61759 * (0.0242314
        + 0.1972995 * max(0.0, Q.log_sum_pt - 6.879733) / 0.07575656   # +19.7%  log_sum_pt > 6.88
        + 0.1341982 * max(0.0, Q.pt_entropy - 2.0) / 0.872004   # +13.4%  pt_entropy > 2
        - 0.09591148 * max(0.0, Q.sum_pt_top50 - 953.0) / 91.78116   # -9.6%  sum_pt_top50 > 953
        - 0.08152363 * max(0.0, Q.z_top50_slots - 0.96) / 0.03299329   # -8.2%  z_top50_slots > 0.96
        + 0.06869138 * max(0.0, 0.196 - Q.tau1) / 0.1103229   # +6.9%  tau1 < 0.196
        - 0.06615921 * max(0.0, Q.n_for_90pct - 10.4) / 10.85133   # -6.6%  n_for_90pct > 10.4
        + 0.0472487 * max(0.0, 1070.0 - Q.sum_pt_top40) / 70.95933   # +4.7%  sum_pt_top40 < 1070
        + 0.0405456 * max(0.0, Q.n_particles - 38.4) / 10.53999   # +4.1%  n_particles > 38.4
        + 0.03834914 * max(0.0, 0.258 - Q.LHA) / 0.03872523   # +3.8%  LHA < 0.258
        - 0.03533443 * max(0.0, 29.5 - Q.sj3_mass1) * max(0.0, 21.0 - Q.sj3_mass2) / 193.5608   # -3.5%  sj3_mass1 < 29.5 and sj3_mass2 < 21
        - 0.02578691 * max(0.0, 122.0 - Q.mass) / 39.95231   # -2.6%  mass < 122
        + 0.01748596 * max(0.0, Q.z_top30_slots - 0.939) * max(0.0, 0.075 - Q.C2) / 0.0008223945   # +1.7%  z_top30_slots > 0.939 and C2 < 0.075
        + 0.01666159 * max(0.0, 0.00509 - Q.lam1) / 0.001107091   # +1.7%  lam1 < 0.00509
        + 0.01529543 * max(0.0, Q.n_particles - 39.4) * max(0.0, 0.111 - Q.dr_0) / 0.5113436   # +1.5%  n_particles > 39.4 and dr_0 < 0.111
        - 0.01477973 * max(0.0, 0.032 - Q.M3) / 0.009257398   # -1.5%  M3 < 0.032
        - 0.0147242 * max(0.0, 6.57 - Q.n_dr_0p2_0p4) / 1.823862   # -1.5%  n_dr_0p2_0p4 < 6.57
        - 0.01065638 * max(0.0, 49.0 - Q.mass_top20) / 6.375307   # -1.1%  mass_top20 < 49
        - 0.01049869 * max(0.0, Q.z_top20_slots - 0.887) * max(0.0, 0.0414 - Q.dr_2) / 0.0006727154   # -1.0%  z_top20_slots > 0.887 and dr_2 < 0.0414
        + 0.01014563 * max(0.0, Q.z_top30_slots - 0.942) * max(0.0, Q.max_pair_mass - 9.45) / 0.1966421   # +1.0%  z_top30_slots > 0.942 and max_pair_mass > 9.45
        - 0.009790002 * max(0.0, Q.z_dr_0_0p05 - 0.875) / 0.01681403   # -1.0%  z_dr_0_0p05 > 0.875
        + 0.007778495 * max(0.0, 0.000986 - Q.sum_z_dr2_top15) / 0.0001318059   # +0.8%  sum_z_dr2_top15 < 0.000986
        - 0.00766153 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # -0.8%  psi_0p3 > 0.998
        + 0.00717122 * max(0.0, 50.6 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.8) / 41.16395   # +0.7%  mass_top20 < 50.6 and n_real_top40 > 29.8
        - 0.006772627 * max(0.0, 31.6 - Q.pt_9) / 6.708255   # -0.7%  pt_9 < 31.6
        - 0.006588254 * max(0.0, 8.0 - Q.n_dr_0p1_0p2) / 1.320682   # -0.7%  n_dr_0p1_0p2 < 8
        + 0.004374781 * max(0.0, Q.z_top30_slots - 0.914) * max(0.0, Q.ptdr0_3 - 6.21) / 0.09031647   # +0.4%  z_top30_slots > 0.914 and ptdr0_3 > 6.21
        + 0.004352687 * max(0.0, Q.soft1_pt - 0.8061523) / 0.2012472   # +0.4%  soft1_pt > 0.8062
        - 0.004214606 * max(0.0, Q.sum_pt_top30 - 1190.0) / 7.03472   # -0.4%  sum_pt_top30 > 1190
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.1622;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.1622245 * (0.2399684
        + 1.0 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, 0.0272 - Q.sum_z_dr2_top15) / 0.001135614   # +100.0%  log_sum_pt > 6.91 and sum_z_dr2_top15 < 0.0272
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.136;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.136023 * (-0.1074551
        + 0.2132139 * max(0.0, 0.000625 - Q.lam2) / 0.0001547081   # +21.3%  lam2 < 0.000625
        + 0.1999103 * max(0.0, 46.1 - Q.n_particles) / 6.444332   # +20.0%  n_particles < 46.1
        - 0.1629428 * max(0.0, 0.375 - Q.tau21) / 0.06405638   # -16.3%  tau21 < 0.375
        + 0.1506229 * max(0.0, 7.15 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.98) / 0.03801554   # +15.1%  n_dr_0p2_0p4 < 7.15 and z_top50_slots > 0.98
        - 0.1232992 * max(0.0, 69.2 - Q.mass_top50) / 7.455136   # -12.3%  mass_top50 < 69.2
        + 0.07824884 * max(0.0, 11.2 - Q.n_dr_0p1_0p2) * max(0.0, Q.psi_0p2 - 0.934) / 0.1252313   # +7.8%  n_dr_0p1_0p2 < 11.2 and psi_0p2 > 0.934
        + 0.0622309 * max(0.0, 0.00669 - Q.sum_z_dr2_top40) / 0.001689894   # +6.2%  sum_z_dr2_top40 < 0.00669
        - 0.009531214 * max(0.0, 0.406 - Q.tau21) * max(0.0, Q.lam1 - 0.00572) / 0.0002525577   # -1.0%  tau21 < 0.406 and lam1 > 0.00572
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 5.809;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.809382 * (0.1270836
        - 0.2801158 * max(0.0, 78.23178 - Q.mass) / 9.84675   # -28.0%  mass < 78.23
        - 0.1355773 * max(0.0, Q.n_particles - 23.1) / 22.9288   # -13.6%  n_particles > 23.1
        + 0.0919081 * max(0.0, 24.2 - Q.n_dr_0p2_0p4) / 15.51915   # +9.2%  n_dr_0p2_0p4 < 24.2
        + 0.08458704 * max(0.0, Q.sum_zz_dr2 - 0.0103) / 0.003014396   # +8.5%  sum_zz_dr2 > 0.0103
        + 0.08325548 * max(0.0, 101.0 - Q.mass) * max(0.0, 3.79 - Q.D2_b2) / 24.33662   # +8.3%  mass < 101 and D2_b2 < 3.79
        + 0.06758768 * max(0.0, Q.n_particles - 20.4) * max(0.0, 2.44 - Q.soft1_pt) / 42.89437   # +6.8%  n_particles > 20.4 and soft1_pt < 2.44
        - 0.05679807 * max(0.0, 88.5 - Q.mass_top40) * max(0.0, 3.21 - Q.D2_b2) / 8.233042   # -5.7%  mass_top40 < 88.5 and D2_b2 < 3.21
        + 0.05000693 * max(0.0, 57.37264 - Q.mass_top40) / 4.890706   # +5.0%  mass_top40 < 57.37
        + 0.04838613 * max(0.0, Q.psi_0p3 - 0.997) / 0.001157514   # +4.8%  psi_0p3 > 0.997
        - 0.03165562 * max(0.0, Q.lam1 - 0.00794) / 0.002672973   # -3.2%  lam1 > 0.00794
        - 0.01336318 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, 13.2 - Q.n_dr_0_0p05) / 0.004749756   # -1.3%  psi_0p3 > 0.997 and n_dr_0_0p05 < 13.2
        + 0.01296331 * max(0.0, Q.n_particles - 27.6) * max(0.0, Q.n_dr_0_0p05 - 10.7) / 108.7435   # +1.3%  n_particles > 27.6 and n_dr_0_0p05 > 10.7
        - 0.01066576 * max(0.0, Q.sj2_dr - 0.232) * max(0.0, 20.6 - Q.D2_b2) / 0.35033   # -1.1%  sj2_dr > 0.232 and D2_b2 < 20.6
        - 0.01041936 * max(0.0, 22.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 12.1) / 37.69358   # -1.0%  n_dr_0p2_0p4 < 22.7 and n_dr_0p1_0p2 > 12.1
        - 0.008001951 * max(0.0, 0.003248562 - Q.sum_z_dr2_top15) / 0.0007895494   # -0.8%  sum_z_dr2_top15 < 0.003249
        - 0.006153113 * max(0.0, Q.sum_z_dr - 0.121) / 0.00424781   # -0.6%  sum_z_dr > 0.121
        - 0.005671845 * max(0.0, Q.C2 - 0.0725) / 0.01291268   # -0.6%  C2 > 0.0725
        - 0.002883332 * max(0.0, Q.e2 - 0.0568) / 0.001001653   # -0.3%  e2 > 0.0568
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 6.871;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.871152 * (0.06913589
        + 0.3227113 * max(0.0, Q.sum_pt_top50 - 943.0) / 100.5535   # +32.3%  sum_pt_top50 > 943
        - 0.2398499 * max(0.0, Q.log_sum_pt - 6.903354) / 0.05667477   # -24.0%  log_sum_pt > 6.903
        - 0.1548053 * max(0.0, Q.mass_top50 - 158.0) / 1.51142   # -15.5%  mass_top50 > 158
        - 0.053725 * max(0.0, Q.sum_pt - 1034.628) / 39.8323   # -5.4%  sum_pt > 1035
        + 0.04901138 * max(0.0, 0.459 - Q.max_dr) / 0.1126117   # +4.9%  max_dr < 0.459
        - 0.03544069 * max(0.0, 97.07106 - Q.sd_mass) / 44.66664   # -3.5%  sd_mass < 97.07
        + 0.02837578 * max(0.0, 47.5 - Q.n_particles) / 7.177081   # +2.8%  n_particles < 47.5
        - 0.0233548 * max(0.0, 22.6 - Q.n_dr_0p1_0p2) / 11.19224   # -2.3%  n_dr_0p1_0p2 < 22.6
        - 0.02039109 * max(0.0, 56.1 - Q.n_particles) * max(0.0, 2.7 - Q.D2) / 11.15957   # -2.0%  n_particles < 56.1 and D2 < 2.7
        + 0.02033629 * max(0.0, 12.4 - Q.n_dr_0p2_0p4) / 5.489783   # +2.0%  n_dr_0p2_0p4 < 12.4
        - 0.01819981 * max(0.0, 0.561 - Q.tau21) / 0.1667473   # -1.8%  tau21 < 0.561
        - 0.01606312 * max(0.0, Q.mass - 172.4703) / 0.7455617   # -1.6%  mass > 172.5
        - 0.0154853 * max(0.0, Q.sum_z_dr2_top50 - 0.0191) / 0.001270849   # -1.5%  sum_z_dr2_top50 > 0.0191
        + 0.002250239 * max(0.0, Q.mass_over_sum_pt - 0.0781) / 0.0195769   # +0.2%  mass_over_sum_pt > 0.0781
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 4.385;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.385159 * (0.2279436
        - 0.3187105 * max(0.0, 162.9488 - Q.mass) / 74.70601   # -31.9%  mass < 162.9
        + 0.1860878 * max(0.0, Q.sum_z_dr2_top20 - 0.00733) / 0.003107201   # +18.6%  sum_z_dr2_top20 > 0.00733
        - 0.1440301 * max(0.0, Q.mass_over_sum_pt - 0.121) / 0.007196258   # -14.4%  mass_over_sum_pt > 0.121
        + 0.08830283 * max(0.0, 0.04728002 - Q.e2) / 0.01854242   # +8.8%  e2 < 0.04728
        - 0.07905291 * max(0.0, Q.sum_z_dr2_top20 - 0.0035) * max(0.0, Q.C2_b2 - 0.00379) / 0.0001042062   # -7.9%  sum_z_dr2_top20 > 0.0035 and C2_b2 > 0.00379
        + 0.06126499 * max(0.0, 73.3 - Q.mass_top50) / 8.661614   # +6.1%  mass_top50 < 73.3
        - 0.05017627 * max(0.0, Q.sj3_pair_mass_min - 77.9) / 0.2692169   # -5.0%  sj3_pair_mass_min > 77.9
        - 0.03584643 * max(0.0, Q.sum_z_dr2_top30 - 0.0275) / 0.0002314476   # -3.6%  sum_z_dr2_top30 > 0.0275
        + 0.01831924 * max(0.0, 0.0533 - Q.tau1) / 0.00754576   # +1.8%  tau1 < 0.0533
        + 0.01246076 * max(0.0, 102.0 - Q.mass) * max(0.0, 1010.0 - Q.sum_pt) / 478.475   # +1.2%  mass < 102 and sum_pt < 1010
        + 0.005748242 * max(0.0, 0.931 - Q.z_top40_slots) / 0.002688747   # +0.6%  z_top40_slots < 0.931
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 5.499;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.499117 * (0.02148591
        - 0.2376812 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.psi_0p3 - 0.964) / 0.5308399   # -23.8%  mass < 91.2 and psi_0p3 > 0.964
        + 0.2205372 * max(0.0, 121.0 - Q.mass) / 39.14871   # +22.1%  mass < 121
        - 0.163196 * max(0.0, Q.sd_mass - 97.2) / 4.404878   # -16.3%  sd_mass > 97.2
        + 0.1139278 * max(0.0, 0.234 - Q.tau21_b2) / 0.05168405   # +11.4%  tau21_b2 < 0.234
        - 0.07821194 * max(0.0, 0.1199231 - Q.tau1) / 0.04418675   # -7.8%  tau1 < 0.1199
        - 0.06308331 * max(0.0, 0.231 - Q.tau21_b2) * max(0.0, 1270.0 - Q.sum_pt) / 11.70026   # -6.3%  tau21_b2 < 0.231 and sum_pt < 1270
        + 0.03757171 * max(0.0, 8.15 - Q.n_dr_0p2_0p4) / 2.672322   # +3.8%  n_dr_0p2_0p4 < 8.15
        - 0.03199615 * max(0.0, 0.23 - Q.tau21_b2) * max(0.0, 0.00808 - Q.mass_over_sum_pt_sq) / 5.560925e-05   # -3.2%  tau21_b2 < 0.23 and mass_over_sum_pt_sq < 0.00808
        + 0.03145894 * max(0.0, Q.psi_0p3 - 0.997) * max(0.0, Q.z_top50_slots - 0.979) / 2.128765e-05   # +3.1%  psi_0p3 > 0.997 and z_top50_slots > 0.979
        - 0.01254658 * max(0.0, 93.2 - Q.mass) * max(0.0, 0.503 - Q.z_dr_0_0p05) / 0.6133324   # -1.3%  mass < 93.2 and z_dr_0_0p05 < 0.503
        - 0.008137494 * max(0.0, Q.psi_0p3 - 0.998) / 0.0006668586   # -0.8%  psi_0p3 > 0.998
        - 0.001651785 * max(0.0, 0.00796 - Q.lam1) * max(0.0, Q.zdr_0 - 0.0152) / 6.729428e-07   # -0.2%  lam1 < 0.00796 and zdr_0 > 0.0152
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 6.914;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.914061 * (0.3982718
        - 0.2131889 * max(0.0, 103.0 - Q.mass) / 25.0922   # -21.3%  mass < 103
        - 0.1956748 * max(0.0, 0.00987 - Q.lam1_plus_lam2) / 0.0036948   # -19.6%  lam1_plus_lam2 < 0.00987
        + 0.0953058 * max(0.0, Q.mass_over_sum_pt - 0.0785) * max(0.0, 1100.0 - Q.sum_pt) / 1.975633   # +9.5%  mass_over_sum_pt > 0.0785 and sum_pt < 1100
        + 0.09227014 * max(0.0, 0.00754 - Q.sum_z_dr2_top30) / 0.002386741   # +9.2%  sum_z_dr2_top30 < 0.00754
        - 0.09068774 * max(0.0, Q.z_top50_slots - 0.972) / 0.02191829   # -9.1%  z_top50_slots > 0.972
        + 0.0855586 * max(0.0, 1030.0 - Q.sum_pt) / 27.96979   # +8.6%  sum_pt < 1030
        - 0.07476664 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063   # -7.5%  n_dr_0p2_0p4 < 21
        - 0.06286628 * max(0.0, 1030.0 - Q.sum_pt_top40) / 41.61542   # -6.3%  sum_pt_top40 < 1030
        - 0.03625888 * max(0.0, Q.sum_z_dr2_top40 - 0.00555) * max(0.0, 6.99 - Q.log_sum_pt) / 0.0004024376   # -3.6%  sum_z_dr2_top40 > 0.00555 and log_sum_pt < 6.99
        + 0.02097707 * max(0.0, Q.sj2_dr - 0.23) / 0.02185914   # +2.1%  sj2_dr > 0.23
        + 0.009504272 * max(0.0, Q.sum_pt_top30 - 987.0) / 44.07673   # +1.0%  sum_pt_top30 > 987
        + 0.009364656 * max(0.0, 0.0848 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.4) / 0.1534487   # +0.9%  tau2 < 0.0848 and sj3_mass1 > 13.4
        - 0.007627351 * max(0.0, Q.sum_z_dr2_top20 - 0.01062209) / 0.0023458   # -0.8%  sum_z_dr2_top20 > 0.01062
        + 0.005948882 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.313) / 0.175758   # +0.6%  n_dr_0p2_0p4 < 18 and z_dr_0p1_0p2 > 0.313
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 1.649;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.648829 * (-0.1097399
        + 0.3835386 * max(0.0, 91.02273 - Q.mass) / 16.3551   # +38.4%  mass < 91.02
        + 0.2483606 * max(0.0, 0.00617 - Q.sum_z_dr2_top40) / 0.001418021   # +24.8%  sum_z_dr2_top40 < 0.00617
        + 0.112382 * max(0.0, 989.0 - Q.sum_pt) / 12.58708   # +11.2%  sum_pt < 989
        + 0.08076246 * max(0.0, Q.sum_z_dr - 0.0976) / 0.008295986   # +8.1%  sum_z_dr > 0.0976
        + 0.05889736 * max(0.0, 100.0 - Q.mass_top40) * max(0.0, 961.0 - Q.sum_pt_top40) / 322.1985   # +5.9%  mass_top40 < 100 and sum_pt_top40 < 961
        - 0.04379871 * max(0.0, 97.9 - Q.mass_top40) * max(0.0, 1020.0 - Q.sum_pt) / 577.6614   # -4.4%  mass_top40 < 97.9 and sum_pt < 1020
        + 0.03772421 * max(0.0, Q.z_top40_slots - 0.954) / 0.03108671   # +3.8%  z_top40_slots > 0.954
        + 0.03453611 * max(0.0, 116.0 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 8.57) / 36.61832   # +3.5%  mass_top40 < 116 and n_dr_0p2_0p4 > 8.57
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 6.633;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.632562 * (0.009703854
        + 0.1612361 * max(0.0, 87.9 - Q.mass) / 14.4965   # +16.1%  mass < 87.9
        - 0.1477768 * max(0.0, 65.0 - Q.sj2_mass1) / 36.316   # -14.8%  sj2_mass1 < 65
        + 0.1053163 * max(0.0, Q.e2 - 0.0128) / 0.01893514   # +10.5%  e2 > 0.0128
        - 0.08315521 * max(0.0, 0.00533 - Q.sum_z_dr2_top20) / 0.001444908   # -8.3%  sum_z_dr2_top20 < 0.00533
        + 0.0670547 * max(0.0, 941.0 - Q.sum_pt_top15) / 99.61533   # +6.7%  sum_pt_top15 < 941
        + 0.06061539 * max(0.0, 0.0485 - Q.tau2) / 0.02120256   # +6.1%  tau2 < 0.0485
        + 0.05689153 * max(0.0, 3.04 - Q.D2) / 0.9713633   # +5.7%  D2 < 3.04
        + 0.05312999 * max(0.0, 0.0859 - Q.z_dr_0p2_0p4) / 0.05399504   # +5.3%  z_dr_0p2_0p4 < 0.0859
        - 0.04579333 * max(0.0, 0.0234 - Q.sum_z_dr2_top10) * max(0.0, Q.psi_0p3 - 0.998) / 1.213902e-05   # -4.6%  sum_z_dr2_top10 < 0.0234 and psi_0p3 > 0.998
        - 0.04279161 * max(0.0, 11.1 - Q.n_dr_0p2_0p4) / 4.558263   # -4.3%  n_dr_0p2_0p4 < 11.1
        + 0.03870881 * max(0.0, 0.0664 - Q.dr_0) / 0.02679036   # +3.9%  dr_0 < 0.0664
        - 0.03370867 * max(0.0, 0.047 - Q.tau1) / 0.005810515   # -3.4%  tau1 < 0.047
        - 0.02277462 * max(0.0, Q.pt_dispersion - 0.27) / 0.06763653   # -2.3%  pt_dispersion > 0.27
        + 0.01745292 * max(0.0, Q.mass_top5 - 17.5) / 15.10767   # +1.7%  mass_top5 > 17.5
        - 0.01434554 * max(0.0, 2.99 - Q.D2) * max(0.0, Q.sj2_dr - 0.192) / 0.02493977   # -1.4%  D2 < 2.99 and sj2_dr > 0.192
        - 0.01172735 * max(0.0, 984.0 - Q.sum_pt) / 11.58636   # -1.2%  sum_pt < 984
        - 0.01141219 * max(0.0, Q.mass_over_sum_pt - 0.16) / 0.001408236   # -1.1%  mass_over_sum_pt > 0.16
        + 0.01125749 * max(0.0, 63.3 - Q.sj2_mass1) * max(0.0, Q.sj2_mass2 - 7.52) / 165.3349   # +1.1%  sj2_mass1 < 63.3 and sj2_mass2 > 7.52
        - 0.007240986 * max(0.0, Q.sum_pt_top10 - 941.0) / 12.46197   # -0.7%  sum_pt_top10 > 941
        - 0.004402305 * max(0.0, Q.mass_top50 - 138.0) / 4.065355   # -0.4%  mass_top50 > 138
        - 0.00287094 * max(0.0, Q.mass_top50 - 133.0) * max(0.0, Q.soft5_z - 0.00163) / 0.001810606   # -0.3%  mass_top50 > 133 and soft5_z > 0.00163
        - 0.0003371901 * max(0.0, Q.sum_pt_top10 - 943.0) * max(0.0, -0.078 - Q.eta_0) / 0.002027485   # -0.0%  sum_pt_top10 > 943 and eta_0 < -0.078
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 3.983;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.983099 * (-0.1043296
        - 0.2406859 * max(0.0, 0.00666 - Q.sum_z_dr2_top30) / 0.001845922   # -24.1%  sum_z_dr2_top30 < 0.00666
        + 0.2280058 * max(0.0, 115.5062 - Q.mass_top50) / 35.7055   # +22.8%  mass_top50 < 115.5
        + 0.1805399 * max(0.0, 0.0266 - Q.e2) / 0.005137276   # +18.1%  e2 < 0.0266
        + 0.1169676 * max(0.0, 7.5 - Q.n_dr_0p2_0p4) / 2.309907   # +11.7%  n_dr_0p2_0p4 < 7.5
        + 0.09746059 * max(0.0, 16.7 - Q.n_dr_0p1_0p2) / 6.37366   # +9.7%  n_dr_0p1_0p2 < 16.7
        - 0.05280965 * max(0.0, 9.67 - Q.n_dr_0p2_0p4) * max(0.0, 0.0062 - Q.sum_z_dr2) / 0.006020314   # -5.3%  n_dr_0p2_0p4 < 9.67 and sum_z_dr2 < 0.0062
        - 0.02739432 * max(0.0, 119.0 - Q.mass) * max(0.0, 872.0 - Q.sum_pt_top10) / 3178.999   # -2.7%  mass < 119 and sum_pt_top10 < 872
        - 0.02577644 * max(0.0, Q.z_top5_slots - 0.552) / 0.07470553   # -2.6%  z_top5_slots > 0.552
        + 0.02021791 * max(0.0, Q.psi_0p3 - 0.99) / 0.005785419   # +2.0%  psi_0p3 > 0.99
        - 0.01014192 * max(0.0, 8.65 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 36.1) / 19.12556   # -1.0%  n_dr_0p2_0p4 < 8.65 and n_particles > 36.1
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.004;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.003822 * (-0.4477016
        + 0.8165811 * max(0.0, 89.70783 - Q.mass) / 15.53596   # +81.7%  mass < 89.71
        - 0.07103064 * max(0.0, 0.00586 - Q.sum_z_dr2_top15) / 0.001917421   # -7.1%  sum_z_dr2_top15 < 0.00586
        - 0.05060395 * max(0.0, 90.0 - Q.mass) * max(0.0, 0.999 - Q.psi_0p3) / 0.04763538   # -5.1%  mass < 90 and psi_0p3 < 0.999
        + 0.04180233 * max(0.0, Q.sd_mass - 124.0) / 1.314611   # +4.2%  sd_mass > 124
        + 0.01697864 * max(0.0, Q.sj3_pair_mass_max - 123.0) * max(0.0, 65.0 - Q.sj3_pair_mass_min) / 31.55854   # +1.7%  sj3_pair_mass_max > 123 and sj3_pair_mass_min < 65
        - 0.003003343 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0036 - Q.lam2) / 1.349245e-05   # -0.3%  log_sum_pt < 6.85 and lam2 < 0.0036
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 2.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.469529 * (1.055991
        - 0.2403004 * max(0.0, 6.92 - Q.log_sum_pt) / 0.02069527   # -24.0%  log_sum_pt < 6.92
        - 0.1737308 * max(0.0, Q.mass - 173.0) * max(0.0, Q.soft5_z - 0.000296) / 0.001323384   # -17.4%  mass > 173 and soft5_z > 0.000296
        - 0.1326773 * max(0.0, Q.mass - 162.9488) / 1.498386   # -13.3%  mass > 162.9
        - 0.127405 * max(0.0, Q.mass_top50 - 134.0) * max(0.0, Q.soft6_z - 0.00094) / 0.003690397   # -12.7%  mass_top50 > 134 and soft6_z > 0.00094
        - 0.1251953 * max(0.0, 1090.0 - Q.sum_pt) * max(0.0, 1.0 - Q.tau21_b2) / 47.21906   # -12.5%  sum_pt < 1090 and tau21_b2 < 1
        + 0.1017208 * max(0.0, 1020.0 - Q.sum_pt_top40) / 35.61475   # +10.2%  sum_pt_top40 < 1020
        - 0.06164765 * max(0.0, Q.mass_top50 - 90.5) * max(0.0, 0.00189 - Q.soft7_z) / 0.006861082   # -6.2%  mass_top50 > 90.5 and soft7_z < 0.00189
        + 0.0216693 * max(0.0, Q.mass_top5 - 40.1) / 5.088478   # +2.2%  mass_top5 > 40.1
        + 0.01537706 * max(0.0, Q.mass - 169.0) * max(0.0, Q.D2 - 0.281) / 1.746967   # +1.5%  mass > 169 and D2 > 0.281
        - 0.000276399 * max(0.0, Q.log_sum_pt - 7.14) * max(0.0, Q.sd_zg - 0.446) / 1.294925e-05   # -0.0%  log_sum_pt > 7.14 and sd_zg > 0.446
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 10.42;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.41524 * (-0.05871768
        - 0.3193618 * max(0.0, 87.3546 - Q.mass) / 14.19492   # -31.9%  mass < 87.35
        - 0.1698162 * max(0.0, 0.00262 - Q.sum_z_dr2_top30) / 0.0003612049   # -17.0%  sum_z_dr2_top30 < 0.00262
        + 0.09184182 * max(0.0, 78.4 - Q.mass_top50) / 10.34754   # +9.2%  mass_top50 < 78.4
        + 0.08525777 * max(0.0, 0.144 - Q.mass_over_sum_pt) / 0.06057806   # +8.5%  mass_over_sum_pt < 0.144
        + 0.07580245 * max(0.0, Q.psi_0p3 - 0.993) / 0.003630655   # +7.6%  psi_0p3 > 0.993
        - 0.06598777 * max(0.0, 0.0905 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.914) / 0.001332637   # -6.6%  mass_over_sum_pt < 0.0905 and psi_0p2 > 0.914
        + 0.0550881 * max(0.0, 0.0302 - Q.e2) / 0.006774736   # +5.5%  e2 < 0.0302
        + 0.02930662 * max(0.0, 0.0067 - Q.sum_z_dr2_top5) / 0.003296885   # +2.9%  sum_z_dr2_top5 < 0.0067
        - 0.02547021 * max(0.0, 0.00154 - Q.sum_z_dr2_top3) / 0.0004907586   # -2.5%  sum_z_dr2_top3 < 0.00154
        - 0.01726122 * max(0.0, 0.484 - Q.N2) * max(0.0, Q.max_dr - 0.24) / 0.01951275   # -1.7%  N2 < 0.484 and max_dr > 0.24
        - 0.01570072 * max(0.0, 0.00654 - Q.sum_z_dr2_top5) * max(0.0, Q.n_pt_above_10 - 13.0) / 0.01758573   # -1.6%  sum_z_dr2_top5 < 0.00654 and n_pt_above_10 > 13
        + 0.01338785 * max(0.0, Q.psi_0p2 - 0.978) * max(0.0, Q.pt2_over_pt0 - 0.121) / 0.00324032   # +1.3%  psi_0p2 > 0.978 and pt2_over_pt0 > 0.121
        + 0.01270976 * max(0.0, 6.94 - Q.log_sum_pt) / 0.03062506   # +1.3%  log_sum_pt < 6.94
        - 0.008070124 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.188 - Q.sd_rg) / 0.0002440912   # -0.8%  psi_0p3 > 0.993 and sd_rg < 0.188
        - 0.006561155 * max(0.0, Q.psi_0p3 - 0.993) * max(0.0, 0.124 - Q.z_2nd) / 5.668685e-05   # -0.7%  psi_0p3 > 0.993 and z_2nd < 0.124
        - 0.006180371 * max(0.0, 0.432 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -12.9) / 4.946312   # -0.6%  tau21_b2 < 0.432 and orientation_deg > -12.9
        - 0.00219607 * max(0.0, 0.322 - Q.tau21_b2) * max(0.0, 30.6 - Q.n_real_top40) / 0.1114268   # -0.2%  tau21_b2 < 0.322 and n_real_top40 < 30.6
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 3.679;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.678863 * (0.1035544
        + 0.1891602 * max(0.0, 7.03 - Q.log_sum_pt) / 0.09997075   # +18.9%  log_sum_pt < 7.03
        - 0.1638262 * max(0.0, Q.psi_0p2 - 0.857) / 0.1015269   # -16.4%  psi_0p2 > 0.857
        - 0.1274222 * max(0.0, 1020.0 - Q.sum_pt_top40) / 35.61475   # -12.7%  sum_pt_top40 < 1020
        - 0.1266388 * max(0.0, Q.e2 - 0.0294) / 0.00787394   # -12.7%  e2 > 0.0294
        + 0.1254282 * max(0.0, 87.75185 - Q.sd_mass) / 36.91786   # +12.5%  sd_mass < 87.75
        - 0.07125159 * max(0.0, 0.0678 - Q.tau1) / 0.01237164   # -7.1%  tau1 < 0.0678
        - 0.05363437 * max(0.0, Q.n_dr_0p2_0p4 - 8.79) / 3.318883   # -5.4%  n_dr_0p2_0p4 > 8.79
        + 0.05016659 * max(0.0, 13.6 - Q.n_dr_0p1_0p2) / 4.204811   # +5.0%  n_dr_0p1_0p2 < 13.6
        + 0.03269604 * max(0.0, Q.sd_rg - 0.203) / 0.02078301   # +3.3%  sd_rg > 0.203
        + 0.02480188 * max(0.0, 0.00181 - Q.lam2) / 0.0009789695   # +2.5%  lam2 < 0.00181
        + 0.0220983 * max(0.0, Q.sj2_dr - 0.225) * max(0.0, 0.0493 - Q.C2_b2) / 0.0005919008   # +2.2%  sj2_dr > 0.225 and C2_b2 < 0.0493
        - 0.009533039 * max(0.0, 0.294 - Q.max_dr) / 0.01268433   # -1.0%  max_dr < 0.294
        - 0.003342611 * max(0.0, Q.psi_0p3 - 0.989) / 0.0065384   # -0.3%  psi_0p3 > 0.989
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.6411559873949579, 2.8327531512605044, 0.189115756302521, 0.4345018907563025, 0.9879290966386555, 0.8344523109243698, 0.4256715336134454, 0.43969411764705885, 1.2286597689075631, 1.2898114495798318, 1.0909368697478992, 0.803555987394958, 0.5021951680672269, 1.8159126575630251, 0.27542841386554623, 0.3483189075630252]
T = [4.126265149028361, 2.832852222295168, 4.276768802521008, 4.0318644334296225, 4.077869720818015]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +43%, n4 -21%, n9 +16%, n3 -8%, n5 -6%, n12 +3% ...
            + 0.4290734 * h[1] / H_AVG[1]
            - 0.2094965 * h[4] / H_AVG[4]
            + 0.161177 * h[9] / H_AVG[9]
            - 0.07897612 * h[3] / H_AVG[3]
            - 0.0631967 * h[5] / H_AVG[5]
            + 0.02852507 * h[12] / H_AVG[12]
            + 0.01611898 * h[6] / H_AVG[6]
            + 0.009305175 * h[8] / H_AVG[8]
            - 0.00413107 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -37%, n9 +26%, n1 -19%, n11 -7%, n12 +6%, n6 +3% ...
            - 0.3705363 * h[4] / H_AVG[4]
            + 0.256109 * h[9] / H_AVG[9]
            - 0.1874934 * h[1] / H_AVG[1]
            - 0.07091404 * h[11] / H_AVG[11]
            + 0.06093844 * h[12] / H_AVG[12]
            + 0.03286993 * h[6] / H_AVG[6]
            + 0.008344759 * h[2] / H_AVG[2]
            + 0.006776848 * h[8] / H_AVG[8]
            - 0.006017218 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +11%, n0 +11%, n11 +11%, n14 -9%, n4 +8% ...
            - 0.2513761 * h[8] / H_AVG[8]
            + 0.1127996 * h[5] / H_AVG[5]
            + 0.112437 * h[0] / H_AVG[0]
            + 0.1115588 * h[11] / H_AVG[11]
            - 0.08855145 * h[14] / H_AVG[14]
            + 0.07940589 * h[4] / H_AVG[4]
            - 0.06597183 * h[9] / H_AVG[9]
            - 0.06425618 * h[7] / H_AVG[7]
            - 0.04770349 * h[12] / H_AVG[12]
            + 0.04444818 * h[3] / H_AVG[3]
            - 0.01527083 * h[15] / H_AVG[15]
            + 0.006220694 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -29%, n0 -22%, n5 +14%, n6 -10%, n7 +10%, n15 +5% ...
            - 0.2856913 * h[8] / H_AVG[8]
            - 0.2186555 * h[0] / H_AVG[0]
            + 0.142288 * h[5] / H_AVG[5]
            - 0.1022776 * h[6] / H_AVG[6]
            + 0.0988309 * h[7] / H_AVG[7]
            + 0.04859523 * h[15] / H_AVG[15]
            + 0.04714806 * h[3] / H_AVG[3]
            + 0.03892393 * h[12] / H_AVG[12]
            - 0.01758948 * h[2] / H_AVG[2]
        ),
        0.78125 + T[4] * (   # class t: n13 -40%, n10 +26%, n5 -10%, n8 +6%, n12 -5%, n4 +5% ...
            - 0.4035614 * h[13] / H_AVG[13]
            + 0.2633461 * h[10] / H_AVG[10]
            - 0.09592006 * h[5] / H_AVG[5]
            + 0.06355535 * h[8] / H_AVG[8]
            - 0.04618176 * h[12] / H_AVG[12]
            + 0.04542487 * h[4] / H_AVG[4]
            - 0.03203133 * h[15] / H_AVG[15]
            - 0.03032563 * h[7] / H_AVG[7]
            + 0.01965352 * h[0] / H_AVG[0]
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
