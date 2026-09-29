"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  14.1%   (on for 88% of jets)
  neuron  9:  12.0%   (on for 65% of jets)
  neuron  7:  10.3%   (on for 60% of jets)
  neuron  3:   8.5%   (on for 24% of jets)
  neuron 10:   8.1%   (on for 83% of jets)
  neuron  6:   7.4%   (on for 36% of jets)
  neuron  5:   7.3%   (on for 68% of jets)
  neuron  2:   7.3%   (on for 90% of jets)
  neuron 11:   6.5%   (on for 78% of jets)
  neuron  0:   3.8%   (on for 41% of jets)
  neuron 14:   3.7%   (on for 27% of jets)
  neuron  1:   3.5%   (on for 68% of jets)
  neuron  4:   3.2%   (on for 60% of jets)
  neuron 15:   2.9%   (on for 28% of jets)
  neuron  8:   1.2%   (on for 42% of jets)
  neuron 12:   0.4%   (on for 7% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 90.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_6               |Δη| of particle 6
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_3                  ΔR between particle 3 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_4=pt[4],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_6=abs(eta[6]),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr1_3=math.sqrt(dist2(1, 3)) if pt[3] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        phi_0=phi[0],
        phi_1=phi[1],
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
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
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 18.86;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.86128 * (-0.02560802
        + 0.1269963 * max(0.0, 0.0132 - Q.girth2) / 0.008203127   # +12.7%  girth2 < 0.0132
        - 0.09390095 * max(0.0, 0.00449 - Q.width) / 0.001624855   # -9.4%  width < 0.00449
        + 0.08742959 * max(0.0, 0.00866 - Q.width) / 0.004432888   # +8.7%  width < 0.00866
        - 0.08573762 * max(0.0, 0.0847 - Q.mass_over_sum_pt) / 0.03300248   # -8.6%  mass_over_sum_pt < 0.0847
        - 0.07590115 * max(0.0, 0.0753 - Q.girth) / 0.02742515   # -7.6%  girth < 0.0753
        - 0.05754049 * max(0.0, Q.sj3_dr_max - 0.183) / 0.04142318   # -5.8%  sj3_dr_max > 0.183
        + 0.05671934 * max(0.0, 0.0355 - Q.e2) / 0.0136803   # +5.7%  e2 < 0.0355
        + 0.04501857 * max(0.0, Q.sj3_dr_max - 0.11) / 0.08324588   # +4.5%  sj3_dr_max > 0.11
        - 0.03960562 * max(0.0, 59.3 - Q.mass) / 23.49097   # -4.0%  mass < 59.3
        + 0.03947194 * max(0.0, 0.0248 - Q.e2) / 0.007581378   # +3.9%  e2 < 0.0248
        - 0.03666919 * max(0.0, 0.00158 - Q.C2_b2) / 0.0008924231   # -3.7%  C2_b2 < 0.00158
        + 0.02999294 * max(0.0, 0.15 - Q.planar_flow) / 0.05189956   # +3.0%  planar_flow < 0.15
        - 0.02698234 * max(0.0, Q.centroid_offset - 0.00711) / 0.01118509   # -2.7%  centroid_offset > 0.00711
        - 0.02681309 * max(0.0, Q.centroid_offset - 0.0499) / 0.0008065856   # -2.7%  centroid_offset > 0.0499
        + 0.02507178 * max(0.0, Q.log_sum_pt - 6.36) / 0.2230594   # +2.5%  log_sum_pt > 6.36
        + 0.02323695 * max(0.0, 0.00432 - Q.width) * max(0.0, 0.0369 - Q.C3) / 3.878571e-05   # +2.3%  width < 0.00432 and C3 < 0.0369
        + 0.0224001 * max(0.0, 0.000264 - Q.lam1) / 3.353132e-05   # +2.2%  lam1 < 0.000264
        - 0.0181774 * max(0.0, 0.274 - Q.sj3_dr_max) * max(0.0, 0.0577 - Q.z_7) / 0.001640426   # -1.8%  sj3_dr_max < 0.274 and z_7 < 0.0577
        - 0.01493142 * max(0.0, 7.82e-05 - Q.lam2) / 3.081243e-05   # -1.5%  lam2 < 7.82e-05
        - 0.01371016 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, Q.z_7 - 0.018) / 0.0005903908   # -1.4%  log_sum_pt > 6.67 and z_7 > 0.018
        + 0.008792021 * max(0.0, 0.0132 - Q.girth2) * max(0.0, 1.04 - Q.D2) / 0.001090979   # +0.9%  girth2 < 0.0132 and D2 < 1.04
        - 0.008551499 * max(0.0, Q.sum_pt - 906.0) / 11.77315   # -0.9%  sum_pt > 906
        + 0.007611304 * max(0.0, 7.64e-05 - Q.lam2) * max(0.0, 0.266 - Q.D2_b2) / 2.527446e-06   # +0.8%  lam2 < 7.64e-05 and D2_b2 < 0.266
        + 0.006288133 * max(0.0, Q.log_sum_pt - 6.68) * max(0.0, 0.0755 - Q.dr_4) / 0.001928492   # +0.6%  log_sum_pt > 6.68 and dr_4 < 0.0755
        + 0.005394111 * max(0.0, Q.centroid_offset - 0.00817) * max(0.0, 6.81 - Q.n_pt_above_50) / 0.02338847   # +0.5%  centroid_offset > 0.00817 and n_pt_above_50 < 6.81
        - 0.004051339 * max(0.0, 0.00631 - Q.lam1) * max(0.0, 0.837 - Q.D2) / 5.923522e-05   # -0.4%  lam1 < 0.00631 and D2 < 0.837
        - 0.002785266 * max(0.0, Q.centroid_offset - 0.0198) * max(0.0, 0.0628 - Q.z_7) / 3.38927e-05   # -0.3%  centroid_offset > 0.0198 and z_7 < 0.0628
        + 0.002743775 * max(0.0, 0.0141 - Q.girth2) * max(0.0, Q.phi_0 - 0.000879) / 8.727e-05   # +0.3%  girth2 < 0.0141 and phi_0 > 0.000879
        - 0.002585438 * max(0.0, 6.11 - Q.log_sum_pt) / 0.007643367   # -0.3%  log_sum_pt < 6.11
        - 0.002444815 * max(0.0, 63.7 - Q.mass) * max(0.0, 0.163 - Q.D2_b2) / 0.2519801   # -0.2%  mass < 63.7 and D2_b2 < 0.163
        - 0.001574437 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, Q.mean_eta - 0.000559) / 0.0006599088   # -0.2%  log_sum_pt > 6.38 and mean_eta > 0.000559
        - 0.0008709288 * max(0.0, 0.149 - Q.planar_flow) * max(0.0, 0.0225 - Q.dr_2) / 2.466492e-05   # -0.1%  planar_flow < 0.149 and dr_2 < 0.0225
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 8.664;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.664175 * (0.1200345
        + 0.1815334 * max(0.0, 0.00605 - Q.mass_over_sum_pt_sq) / 0.002721171   # +18.2%  mass_over_sum_pt_sq < 0.00605
        + 0.1697882 * max(0.0, Q.log_sum_pt - 6.39) / 0.2004188   # +17.0%  log_sum_pt > 6.39
        - 0.1522042 * max(0.0, 0.00669 - Q.width) / 0.002943581   # -15.2%  width < 0.00669
        - 0.1505656 * max(0.0, 0.0962 - Q.girth) / 0.04377606   # -15.1%  girth < 0.0962
        - 0.07933092 * max(0.0, Q.sum_pt_top5 - 524.0) / 110.5043   # -7.9%  sum_pt_top5 > 524
        + 0.06396464 * max(0.0, Q.log_sum_pt - 6.59) / 0.07805645   # +6.4%  log_sum_pt > 6.59
        - 0.04689005 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, 0.74 - Q.z_dr_0p05_0p1) / 0.1167424   # -4.7%  log_sum_pt > 6.38 and z_dr_0p05_0p1 < 0.74
        - 0.02441991 * max(0.0, 0.0475 - Q.z_7) / 0.006632551   # -2.4%  z_7 < 0.0475
        + 0.01664248 * max(0.0, Q.sj3_dr_max - 0.171) / 0.04696851   # +1.7%  sj3_dr_max > 0.171
        - 0.01618541 * max(0.0, 0.0626 - Q.z_7) * max(0.0, 1.62 - Q.D2) / 0.005654564   # -1.6%  z_7 < 0.0626 and D2 < 1.62
        - 0.01152706 * max(0.0, Q.sj3_dr_max - 0.139) * max(0.0, Q.sj3_pair_mass_min - 1.38) / 0.8838274   # -1.2%  sj3_dr_max > 0.139 and sj3_pair_mass_min > 1.38
        + 0.01143425 * max(0.0, Q.z_7 - 0.0481) / 0.01095889   # +1.1%  z_7 > 0.0481
        - 0.01095296 * max(0.0, 0.00874 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 4.82) / 0.00400415   # -1.1%  lam1 < 0.00874 and n_pt_above_50 > 4.82
        - 0.01079902 * max(0.0, 0.0609 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.09) / 0.002047366   # -1.1%  z_7 < 0.0609 and z_dr_0p05_0p1 > 0.09
        + 0.01055833 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 1.49 - Q.D2) / 0.04315059   # +1.1%  log_sum_pt > 6.5 and D2 < 1.49
        + 0.009948433 * max(0.0, Q.pt_7 - 33.3) * max(0.0, Q.sj2_dr - 0.136) / 0.2023356   # +1.0%  pt_7 > 33.3 and sj2_dr > 0.136
        + 0.008973955 * max(0.0, Q.log_sum_pt - 6.32) * max(0.0, Q.centroid_offset - 0.0126) / 0.0008967926   # +0.9%  log_sum_pt > 6.32 and centroid_offset > 0.0126
        - 0.008709856 * max(0.0, 0.0103 - Q.lam1) * max(0.0, 0.145 - Q.planar_flow) / 0.0002107925   # -0.9%  lam1 < 0.0103 and planar_flow < 0.145
        - 0.008037914 * max(0.0, Q.pt_7 - 32.5) * max(0.0, 53.1 - Q.pt_6) / 21.56096   # -0.8%  pt_7 > 32.5 and pt_6 < 53.1
        + 0.003624462 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.n_dr_0p1_0p2 - 0.551) / 3.774396   # +0.4%  pt_7 > 35 and n_dr_0p1_0p2 > 0.551
        - 0.002340939 * max(0.0, Q.pt_7 - 55.0) / 0.2613699   # -0.2%  pt_7 > 55
        - 0.001568064 * max(0.0, 0.00606 - Q.mass_over_sum_pt_sq) * max(0.0, 0.0392 - Q.D2_b2) / 5.781267e-07   # -0.2%  mass_over_sum_pt_sq < 0.00606 and D2_b2 < 0.0392
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 10.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.48421 * (0.3920181
        - 0.1755186 * max(0.0, 53.3 - Q.pt_7) / 18.99044   # -17.6%  pt_7 < 53.3
        - 0.1311931 * max(0.0, Q.LHA - 0.112) / 0.1348487   # -13.1%  LHA > 0.112
        + 0.1254432 * max(0.0, 798.0 - Q.sum_pt) / 118.4841   # +12.5%  sum_pt < 798
        - 0.1032966 * max(0.0, 718.0 - Q.sum_pt_top5) / 153.3971   # -10.3%  sum_pt_top5 < 718
        + 0.07085424 * max(0.0, 0.0118 - Q.lam1) / 0.007142795   # +7.1%  lam1 < 0.0118
        + 0.05346691 * max(0.0, 6.59 - Q.log_sum_pt) / 0.1251246   # +5.3%  log_sum_pt < 6.59
        + 0.0524878 * max(0.0, 0.0534 - Q.z_7) / 0.009390667   # +5.2%  z_7 < 0.0534
        - 0.03671333 * max(0.0, Q.z_7 - 0.0532) / 0.008259877   # -3.7%  z_7 > 0.0532
        - 0.03311141 * max(0.0, Q.sum_pt_top5 - 608.0) / 66.24942   # -3.3%  sum_pt_top5 > 608
        + 0.02688528 * max(0.0, Q.pt_6 - 27.2) / 14.02343   # +2.7%  pt_6 > 27.2
        - 0.02323338 * max(0.0, 66.3 - Q.sj3_pair_mass_max) * max(0.0, 0.065 - Q.z_7) / 0.595559   # -2.3%  sj3_pair_mass_max < 66.3 and z_7 < 0.065
        + 0.02137481 * max(0.0, 0.00431 - Q.girth2) / 0.001534917   # +2.1%  girth2 < 0.00431
        - 0.02034759 * max(0.0, Q.sj3_pair_mass_max - 28.4) / 12.77415   # -2.0%  sj3_pair_mass_max > 28.4
        + 0.01452902 * max(0.0, Q.log_sum_pt - 6.85) / 0.007185156   # +1.5%  log_sum_pt > 6.85
        + 0.01375501 * max(0.0, 839.0 - Q.sum_pt) * max(0.0, 0.0223 - Q.dr_5) / 0.1885103   # +1.4%  sum_pt < 839 and dr_5 < 0.0223
        - 0.01056022 * max(0.0, 793.0 - Q.sum_pt) * max(0.0, 0.0223 - Q.dr_5) / 0.1337144   # -1.1%  sum_pt < 793 and dr_5 < 0.0223
        - 0.009627523 * max(0.0, 0.0726 - Q.z_7) * max(0.0, 0.64 - Q.planar_flow) / 0.008482097   # -1.0%  z_7 < 0.0726 and planar_flow < 0.64
        - 0.008475309 * max(0.0, 67.5 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.0124) / 0.2278382   # -0.8%  sj3_pair_mass_max < 67.5 and centroid_offset > 0.0124
        - 0.008339067 * max(0.0, Q.pt_6 - 31.4) * max(0.0, 0.781 - Q.planar_flow) / 5.568695   # -0.8%  pt_6 > 31.4 and planar_flow < 0.781
        + 0.008009568 * max(0.0, Q.pt_7 - 43.7) / 1.397238   # +0.8%  pt_7 > 43.7
        + 0.007192006 * max(0.0, 588.0 - Q.sum_pt) * max(0.0, 4.15 - Q.D2_b2) / 75.4025   # +0.7%  sum_pt < 588 and D2_b2 < 4.15
        - 0.006760678 * max(0.0, Q.sum_pt_top5 - 835.0) / 9.098891   # -0.7%  sum_pt_top5 > 835
        - 0.006395033 * max(0.0, 0.00596 - Q.lam1) * max(0.0, Q.max_dr - 0.0963) / 3.585394e-05   # -0.6%  lam1 < 0.00596 and max_dr > 0.0963
        - 0.006368576 * max(0.0, Q.z_7 - 0.0396) * max(0.0, 0.0633 - Q.sj3_dr_min) / 0.0005058294   # -0.6%  z_7 > 0.0396 and sj3_dr_min < 0.0633
        - 0.004594176 * max(0.0, 0.0219 - Q.dr_5) / 0.002632039   # -0.5%  dr_5 < 0.0219
        - 0.004334016 * max(0.0, 0.0078 - Q.girth) / 0.0002442942   # -0.4%  girth < 0.0078
        + 0.003860381 * max(0.0, Q.m012 - 42.8) / 1.032475   # +0.4%  m012 > 42.8
        - 0.003801616 * max(0.0, 0.00464 - Q.centroid_offset) / 0.0004613071   # -0.4%  centroid_offset < 0.00464
        - 0.003583805 * max(0.0, Q.log_sum_pt - 6.88) * max(0.0, 2.99 - Q.D2_b2) / 0.006050461   # -0.4%  log_sum_pt > 6.88 and D2_b2 < 2.99
        + 0.003145758 * max(0.0, Q.sum_pt_top5 - 842.0) * max(0.0, 2.32 - Q.D2_b2) / 5.715908   # +0.3%  sum_pt_top5 > 842 and D2_b2 < 2.32
        + 0.002006764 * max(0.0, 0.00794 - Q.girth) * max(0.0, 72.3 - Q.pt_4) / 0.004429332   # +0.2%  girth < 0.00794 and pt_4 < 72.3
        - 0.0007351017 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, Q.pt_6 - 43.4) / 0.05172456   # -0.1%  log_sum_pt > 6.85 and pt_6 > 43.4
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 13.86;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.86019 * (-0.3124055
        - 0.1881662 * max(0.0, Q.girth2 - 0.0087) / 0.002210186   # -18.8%  girth2 > 0.0087
        + 0.1381612 * max(0.0, Q.girth - 0.0382) / 0.02866678   # +13.8%  girth > 0.0382
        + 0.09169978 * max(0.0, Q.lam1 - 0.0081) / 0.00189698   # +9.2%  lam1 > 0.0081
        - 0.07553176 * max(0.0, Q.tau1 - 0.053) / 0.02705128   # -7.6%  tau1 > 0.053
        + 0.06670545 * max(0.0, Q.girth - 0.077) / 0.01031864   # +6.7%  girth > 0.077
        - 0.06599512 * max(0.0, Q.width - 0.0133) / 0.001433707   # -6.6%  width > 0.0133
        + 0.06021266 * max(0.0, Q.mass_over_sum_pt - 0.0707) / 0.01426597   # +6.0%  mass_over_sum_pt > 0.0707
        + 0.04823358 * max(0.0, Q.sj2_dr - 0.188) / 0.02706586   # +4.8%  sj2_dr > 0.188
        + 0.04439389 * max(0.0, Q.girth - 0.0415) * max(0.0, Q.log_sum_pt - 6.12) / 0.007700975   # +4.4%  girth > 0.0415 and log_sum_pt > 6.12
        + 0.03824695 * max(0.0, Q.max_dr - 0.103) / 0.04492459   # +3.8%  max_dr > 0.103
        - 0.02326915 * max(0.0, 0.00529 - Q.girth2) / 0.002054235   # -2.3%  girth2 < 0.00529
        + 0.02081347 * max(0.0, Q.e2 - 0.0628) / 0.001825814   # +2.1%  e2 > 0.0628
        + 0.02078949 * max(0.0, Q.lam1 - 0.016) / 0.0007276421   # +2.1%  lam1 > 0.016
        - 0.01567661 * max(0.0, Q.mass - 65.4) / 3.121852   # -1.6%  mass > 65.4
        + 0.01531693 * max(0.0, Q.width - 0.00678) * max(0.0, Q.sj3_pairmin_over_m - 0.0341) / 0.000575327   # +1.5%  width > 0.00678 and sj3_pairmin_over_m > 0.0341
        - 0.01448041 * max(0.0, Q.mass_over_sum_pt - 0.0659) * max(0.0, 0.219 - Q.sj2_dr) / 0.0001486676   # -1.4%  mass_over_sum_pt > 0.0659 and sj2_dr < 0.219
        + 0.01308547 * max(0.0, Q.centroid_offset - 0.0107) * max(0.0, 0.0822 - Q.abseta_0) / 0.0003273775   # +1.3%  centroid_offset > 0.0107 and abseta_0 < 0.0822
        + 0.01195583 * max(0.0, Q.lam2 - 0.000916) * max(0.0, 56.4 - Q.pt_6) / 0.00600399   # +1.2%  lam2 > 0.000916 and pt_6 < 56.4
        + 0.01146084 * max(0.0, 0.069 - Q.z_dr_0_0p05) / 0.02020984   # +1.1%  z_dr_0_0p05 < 0.069
        - 0.01112425 * max(0.0, Q.sj2_dr - 0.187) * max(0.0, Q.sj2_mass1 - 5.18) / 0.3359134   # -1.1%  sj2_dr > 0.187 and sj2_mass1 > 5.18
        + 0.007028026 * max(0.0, -0.00445 - Q.mean_eta) / 0.003819992   # +0.7%  mean_eta < -0.00445
        + 0.005379277 * max(0.0, Q.lam2 - 0.00162) * max(0.0, Q.z_6 - 0.0476) / 7.309589e-06   # +0.5%  lam2 > 0.00162 and z_6 > 0.0476
        - 0.004361071 * max(0.0, Q.max_dr - 0.112) * max(0.0, 0.0104 - Q.zdr_1) / 8.268848e-05   # -0.4%  max_dr > 0.112 and zdr_1 < 0.0104
        + 0.004025353 * max(0.0, Q.mean_eta - 0.018) / 0.001347637   # +0.4%  mean_eta > 0.018
        - 0.003887253 * max(0.0, Q.sj2_dr - 0.186) * max(0.0, 0.0524 - Q.dr_3) / 0.0001566223   # -0.4%  sj2_dr > 0.186 and dr_3 < 0.0524
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 30.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.5822 * (-0.1513953
        - 0.1378216 * max(0.0, 0.00892 - Q.girth2) / 0.004641945   # -13.8%  girth2 < 0.00892
        + 0.1341859 * max(0.0, 0.0174 - Q.e2_sq) / 0.01224985   # +13.4%  e2_sq < 0.0174
        + 0.1197002 * max(0.0, 0.0119 - Q.e2_sq) / 0.007410311   # +12.0%  e2_sq < 0.0119
        + 0.07881587 * max(0.0, 0.221 - Q.N2) / 0.05000752   # +7.9%  N2 < 0.221
        + 0.06921956 * max(0.0, 0.00913 - Q.C2_b2) / 0.007375911   # +6.9%  C2_b2 < 0.00913
        - 0.04719449 * max(0.0, 0.00115 - Q.lam2) / 0.0009311686   # -4.7%  lam2 < 0.00115
        + 0.04574076 * max(0.0, Q.sj3_dr_max - 0.162) / 0.05161819   # +4.6%  sj3_dr_max > 0.162
        + 0.04541482 * max(0.0, Q.girth2 - 0.00353) / 0.004084956   # +4.5%  girth2 > 0.00353
        - 0.04435718 * max(0.0, 0.000532 - Q.lam2) / 0.0003864787   # -4.4%  lam2 < 0.000532
        - 0.04429384 * max(0.0, 0.221 - Q.N2) * max(0.0, Q.eccentricity - 0.696) / 0.01428906   # -4.4%  N2 < 0.221 and eccentricity > 0.696
        - 0.03498214 * max(0.0, Q.sj3_dr_max - 0.232) / 0.02553295   # -3.5%  sj3_dr_max > 0.232
        + 0.03044167 * max(0.0, 0.00251 - Q.girth2) / 0.0007447785   # +3.0%  girth2 < 0.00251
        - 0.02807976 * max(0.0, Q.girth - 0.0543) / 0.0195613   # -2.8%  girth > 0.0543
        + 0.02747824 * max(0.0, 0.00812 - Q.girth2_top5) / 0.004376797   # +2.7%  girth2_top5 < 0.00812
        - 0.02037675 * max(0.0, 1.38e-05 - Q.e3) / 5.281067e-06   # -2.0%  e3 < 1.38e-05
        + 0.01833586 * max(0.0, Q.tau1 - 0.0711) * max(0.0, 0.208 - Q.sj3_dr_min) / 0.00192038   # +1.8%  tau1 > 0.0711 and sj3_dr_min < 0.208
        + 0.01606438 * max(0.0, 0.0112 - Q.e2_sq) * max(0.0, 1.05 - Q.D2) / 0.0008298719   # +1.6%  e2_sq < 0.0112 and D2 < 1.05
        - 0.0139086 * max(0.0, 762.0 - Q.sum_pt) / 95.58552   # -1.4%  sum_pt < 762
        - 0.01088015 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, 0.621 - Q.z_dr_0p05_0p1) / 0.002868439   # -1.1%  centroid_offset < 0.018 and z_dr_0p05_0p1 < 0.621
        - 0.01015267 * max(0.0, 1.1 - Q.D2) * max(0.0, 51.6 - Q.pt_7) / 3.385943   # -1.0%  D2 < 1.1 and pt_7 < 51.6
        - 0.01013349 * max(0.0, 68.1 - Q.mass) * max(0.0, 0.985 - Q.D2) / 1.878208   # -1.0%  mass < 68.1 and D2 < 0.985
        - 0.005684149 * max(0.0, 0.222 - Q.N2) * max(0.0, Q.e2_sq - 0.0128) / 6.078104e-05   # -0.6%  N2 < 0.222 and e2_sq > 0.0128
        + 0.004000881 * max(0.0, Q.max_dr - 0.239) / 0.005854341   # +0.4%  max_dr > 0.239
        - 0.002737155 * max(0.0, 0.212 - Q.N2) * max(0.0, 53.1 - Q.mass) / 0.1428468   # -0.3%  N2 < 0.212 and mass < 53.1
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.667 * (0.08887221
        + 0.09628529 * max(0.0, Q.log_sum_pt - 6.26) / 0.3047701   # +9.6%  log_sum_pt > 6.26
        - 0.09266338 * max(0.0, Q.pt_7 - 22.9) / 12.62376   # -9.3%  pt_7 > 22.9
        + 0.09002183 * max(0.0, 0.0968 - Q.tau1) / 0.04364832   # +9.0%  tau1 < 0.0968
        - 0.07967088 * max(0.0, 0.0896 - Q.mass_over_sum_pt) / 0.03679002   # -8.0%  mass_over_sum_pt < 0.0896
        + 0.0704881 * max(0.0, 0.00172 - Q.girth2) * max(0.0, 0.0234 - Q.centroid_offset) / 7.299968e-06   # +7.0%  girth2 < 0.00172 and centroid_offset < 0.0234
        + 0.05703879 * max(0.0, 0.186 - Q.sj3_dr_max) / 0.05794599   # +5.7%  sj3_dr_max < 0.186
        - 0.03686903 * max(0.0, 0.0738 - Q.z_7) * max(0.0, 0.0304 - Q.centroid_offset) / 0.0004312302   # -3.7%  z_7 < 0.0738 and centroid_offset < 0.0304
        - 0.0350764 * max(0.0, 37.6 - Q.pt_7) / 6.074027   # -3.5%  pt_7 < 37.6
        - 0.03502076 * max(0.0, 0.0239 - Q.zdr_0) / 0.01115124   # -3.5%  zdr_0 < 0.0239
        - 0.0310021 * max(0.0, 0.109 - Q.sj3_dr_max) / 0.02467906   # -3.1%  sj3_dr_max < 0.109
        + 0.02968169 * max(0.0, 0.00258 - Q.girth2) / 0.0007722308   # +3.0%  girth2 < 0.00258
        + 0.02932742 * max(0.0, 0.0497 - Q.z_7) * max(0.0, 72.9 - Q.mass_top5) / 0.3847916   # +2.9%  z_7 < 0.0497 and mass_top5 < 72.9
        + 0.02637621 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, 0.0113 - Q.C3) / 0.0005191054   # +2.6%  log_sum_pt > 6.27 and C3 < 0.0113
        + 0.02230836 * max(0.0, 0.0475 - Q.z_6) / 0.004382382   # +2.2%  z_6 < 0.0475
        - 0.02169953 * max(0.0, Q.log_sum_pt - 6.77) / 0.01837055   # -2.2%  log_sum_pt > 6.77
        + 0.0215529 * max(0.0, Q.log_sum_pt - 6.26) * max(0.0, 7.76 - Q.ptdr0_5) / 1.653991   # +2.2%  log_sum_pt > 6.26 and ptdr0_5 < 7.76
        + 0.02078673 * max(0.0, 0.0283 - Q.z_7) / 0.001343831   # +2.1%  z_7 < 0.0283
        + 0.02066949 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 8.69e-05 - Q.mean_eta2) / 9.50351e-07   # +2.1%  log_sum_pt > 6.7 and mean_eta2 < 8.69e-05
        + 0.02051097 * max(0.0, Q.sum_pt_top5 - 714.0) / 29.84865   # +2.1%  sum_pt_top5 > 714
        - 0.01973768 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00424 - Q.mean_eta2) / 0.0001203096   # -2.0%  log_sum_pt > 6.7 and mean_eta2 < 0.00424
        - 0.01935789 * max(0.0, 0.209 - Q.LHA) * max(0.0, 6.82 - Q.log_sum_pt) / 0.003910807   # -1.9%  LHA < 0.209 and log_sum_pt < 6.82
        - 0.01575581 * max(0.0, 0.00987 - Q.C3) / 0.0008845642   # -1.6%  C3 < 0.00987
        + 0.01393007 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 0.00175 - Q.zdr_3) / 2.354866e-05   # +1.4%  log_sum_pt > 6.7 and zdr_3 < 0.00175
        - 0.01344764 * max(0.0, 0.00272 - Q.girth2_top3) * max(0.0, 0.0266 - Q.phi_1) / 2.897898e-05   # -1.3%  girth2_top3 < 0.00272 and phi_1 < 0.0266
        - 0.01289011 * max(0.0, Q.sum_pt_top5 - 720.0) * max(0.0, 8.92e-05 - Q.mean_eta2) / 0.0007947911   # -1.3%  sum_pt_top5 > 720 and mean_eta2 < 8.92e-05
        - 0.01190037 * max(0.0, 0.0237 - Q.zdr_0) * max(0.0, 0.0218 - Q.abseta_0) / 0.0001186367   # -1.2%  zdr_0 < 0.0237 and abseta_0 < 0.0218
        - 0.01147458 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 4.61 - Q.D2_b2) / 0.09202962   # -1.1%  log_sum_pt > 6.7 and D2_b2 < 4.61
        + 0.009083655 * max(0.0, 0.104 - Q.tau1) * max(0.0, 0.464 - Q.planar_flow) / 0.0095936   # +0.9%  tau1 < 0.104 and planar_flow < 0.464
        - 0.007094657 * max(0.0, 0.0463 - Q.z_6) * max(0.0, 0.00185 - Q.zdr_3) / 2.761997e-06   # -0.7%  z_6 < 0.0463 and zdr_3 < 0.00185
        - 0.006921181 * max(0.0, Q.max_pair_mass - 20.4) * max(0.0, 0.248 - Q.dr_max_012) / 0.1130601   # -0.7%  max_pair_mass > 20.4 and dr_max_012 < 0.248
        - 0.006786072 * max(0.0, 36.1 - Q.pt_7) * max(0.0, 0.0221 - Q.C3) / 0.02827619   # -0.7%  pt_7 < 36.1 and C3 < 0.0221
        - 0.005607923 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.00263 - Q.zdr_3) / 6.357038e-06   # -0.6%  log_sum_pt > 6.89 and zdr_3 < 0.00263
        - 0.004085665 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, Q.mean_phi - 0.000708) / 6.456562e-05   # -0.4%  log_sum_pt > 6.7 and mean_phi > 0.000708
        + 0.003951335 * max(0.0, 0.022 - Q.z_6) * max(0.0, 0.167 - Q.abseta_6) / 4.052778e-05   # +0.4%  z_6 < 0.022 and abseta_6 < 0.167
        - 0.0009255107 * max(0.0, Q.log_sum_pt - 6.91) * max(0.0, Q.sj3_pair_mass_max - 21.9) / 0.04273777   # -0.1%  log_sum_pt > 6.91 and sj3_pair_mass_max > 21.9
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 20.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.62687 * (0.1527134
        - 0.2311992 * max(0.0, 0.00869 - Q.girth2) / 0.004456932   # -23.1%  girth2 < 0.00869
        + 0.1156105 * max(0.0, 0.0894 - Q.mass_over_sum_pt) / 0.03663108   # +11.6%  mass_over_sum_pt < 0.0894
        + 0.1051973 * max(0.0, 0.18 - Q.sj3_dr_max) / 0.05451989   # +10.5%  sj3_dr_max < 0.18
        - 0.07047939 * max(0.0, 0.145 - Q.max_dr) / 0.04766457   # -7.0%  max_dr < 0.145
        + 0.06422376 * max(0.0, 0.118 - Q.tau1) / 0.06104771   # +6.4%  tau1 < 0.118
        - 0.06203292 * max(0.0, 0.293 - Q.sj3_dr_max) / 0.1381798   # -6.2%  sj3_dr_max < 0.293
        - 0.05364767 * max(0.0, 0.00114 - Q.lam2) / 0.0009221532   # -5.4%  lam2 < 0.00114
        + 0.04886629 * max(0.0, 41.1 - Q.pt_6) * max(0.0, 995.0 - Q.sum_pt) / 1191.441   # +4.9%  pt_6 < 41.1 and sum_pt < 995
        - 0.0376616 * max(0.0, 66.5 - Q.mass) * max(0.0, 0.0612 - Q.z_dr_0p2_0p4) / 1.714881   # -3.8%  mass < 66.5 and z_dr_0p2_0p4 < 0.0612
        - 0.03473941 * max(0.0, 40.2 - Q.pt_6) / 4.94183   # -3.5%  pt_6 < 40.2
        + 0.02173831 * max(0.0, Q.centroid_offset - 0.00779) / 0.01072712   # +2.2%  centroid_offset > 0.00779
        - 0.02171436 * max(0.0, 0.0132 - Q.lam1) * max(0.0, 0.265 - Q.planar_flow) / 0.0008233444   # -2.2%  lam1 < 0.0132 and planar_flow < 0.265
        - 0.02109991 * max(0.0, 41.1 - Q.pt_6) * max(0.0, Q.z_7 - 0.0225) / 0.07205716   # -2.1%  pt_6 < 41.1 and z_7 > 0.0225
        - 0.01998519 * max(0.0, Q.sj3_dr_min - 0.0235) / 0.02823507   # -2.0%  sj3_dr_min > 0.0235
        + 0.01945109 * max(0.0, Q.sj3_dr_max - 0.189) / 0.03895293   # +1.9%  sj3_dr_max > 0.189
        - 0.01587499 * max(0.0, Q.centroid_offset - 0.0187) / 0.005412421   # -1.6%  centroid_offset > 0.0187
        + 0.009484651 * max(0.0, 0.00333 - Q.lam2) * max(0.0, 2.87 - Q.n_dr_0_0p05) / 0.002457773   # +0.9%  lam2 < 0.00333 and n_dr_0_0p05 < 2.87
        - 0.009039112 * max(0.0, Q.sj3_pair_mass_min - 3.99) / 3.699378   # -0.9%  sj3_pair_mass_min > 3.99
        + 0.007640709 * max(0.0, Q.centroid_offset - 0.00885) * max(0.0, Q.psi_0p1 - 0.406) / 0.003249566   # +0.8%  centroid_offset > 0.00885 and psi_0p1 > 0.406
        + 0.007167167 * max(0.0, Q.centroid_offset - 0.0183) * max(0.0, 0.00887 - Q.mean_phi2) / 2.174062e-05   # +0.7%  centroid_offset > 0.0183 and mean_phi2 < 0.00887
        + 0.005852003 * max(0.0, 761.0 - Q.sum_pt) * max(0.0, 0.0766 - Q.z_7) / 1.014357   # +0.6%  sum_pt < 761 and z_7 < 0.0766
        + 0.005455419 * max(0.0, 601.0 - Q.sum_pt) * max(0.0, 2.19 - Q.n_dr_0p2_0p4) / 41.21914   # +0.5%  sum_pt < 601 and n_dr_0p2_0p4 < 2.19
        + 0.003029893 * max(0.0, 19.1 - Q.pt_6) / 0.234072   # +0.3%  pt_6 < 19.1
        + 0.002602121 * max(0.0, 484.0 - Q.sum_pt) / 5.539074   # +0.3%  sum_pt < 484
        + 0.001700533 * max(0.0, Q.centroid_offset - 0.0185) * max(0.0, Q.sum_pt_top5 - 655.0) / 0.02719122   # +0.2%  centroid_offset > 0.0185 and sum_pt_top5 > 655
        - 0.001631741 * max(0.0, Q.mean_eta - 0.0325) / 0.0004892111   # -0.2%  mean_eta > 0.0325
        + 0.001589176 * max(0.0, 40.8 - Q.pt_6) * max(0.0, Q.M3 - 0.078) / 0.01332509   # +0.2%  pt_6 < 40.8 and M3 > 0.078
        - 0.001060556 * max(0.0, 0.146 - Q.max_dr) * max(0.0, Q.mean_phi - 0.00126) / 0.0001429801   # -0.1%  max_dr < 0.146 and mean_phi > 0.00126
        + 0.0001624502 * max(0.0, 0.00761 - Q.lam1) * max(0.0, Q.mass_top5 - 66.8) / 3.161169e-05   # +0.0%  lam1 < 0.00761 and mass_top5 > 66.8
        + 6.254354e-05 * max(0.0, 617.0 - Q.sum_pt) * max(0.0, 0.0507 - Q.z_3) / 0.0001228645   # +0.0%  sum_pt < 617 and z_3 < 0.0507
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 35.71;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.70936 * (0.2559553
        - 0.1015482 * max(0.0, Q.mass_over_sum_pt - 0.0113) / 0.05085866   # -10.2%  mass_over_sum_pt > 0.0113
        - 0.08293858 * max(0.0, 0.00675 - Q.width) / 0.002985568   # -8.3%  width < 0.00675
        - 0.07962413 * max(0.0, 0.0868 - Q.girth) / 0.0360371   # -8.0%  girth < 0.0868
        - 0.06610384 * max(0.0, Q.mass_over_sum_pt - 0.0902) / 0.008341081   # -6.6%  mass_over_sum_pt > 0.0902
        - 0.05899367 * max(0.0, Q.width - 0.00559) / 0.003084373   # -5.9%  width > 0.00559
        + 0.04998743 * max(0.0, Q.mass_over_sum_pt - 0.0842) / 0.009701191   # +5.0%  mass_over_sum_pt > 0.0842
        + 0.04300327 * max(0.0, Q.mass_over_sum_pt - 0.0732) / 0.0132381   # +4.3%  mass_over_sum_pt > 0.0732
        + 0.04098098 * max(0.0, 0.158 - Q.sj2_dr) / 0.04782368   # +4.1%  sj2_dr < 0.158
        - 0.04007917 * max(0.0, Q.girth2 - 0.00462) / 0.003516466   # -4.0%  girth2 > 0.00462
        - 0.03912432 * max(0.0, 0.187 - Q.sj2_dr) / 0.0643827   # -3.9%  sj2_dr < 0.187
        + 0.03299176 * max(0.0, Q.sj3_dr_max - 0.145) / 0.06136014   # +3.3%  sj3_dr_max > 0.145
        - 0.03198464 * max(0.0, Q.mass_over_sum_pt - 0.108) / 0.005362211   # -3.2%  mass_over_sum_pt > 0.108
        - 0.0246358 * max(0.0, Q.width - 0.00861) * max(0.0, 38.5 - Q.pt_6) / 0.007270486   # -2.5%  width > 0.00861 and pt_6 < 38.5
        + 0.02411139 * max(0.0, 0.303 - Q.LHA) / 0.07827295   # +2.4%  LHA < 0.303
        - 0.02281629 * max(0.0, Q.e2 - 0.0503) / 0.003366757   # -2.3%  e2 > 0.0503
        + 0.01893391 * max(0.0, 0.174 - Q.max_dr) / 0.06795156   # +1.9%  max_dr < 0.174
        + 0.01881653 * max(0.0, Q.mass - 35.9) / 14.38814   # +1.9%  mass > 35.9
        + 0.01583849 * max(0.0, 0.0549 - Q.tau1) / 0.01761938   # +1.6%  tau1 < 0.0549
        - 0.01519237 * max(0.0, 0.0204 - Q.centroid_offset) * max(0.0, 0.0042 - Q.C2_b2) / 2.782102e-05   # -1.5%  centroid_offset < 0.0204 and C2_b2 < 0.0042
        + 0.01494843 * max(0.0, Q.girth2 - 0.0132) * max(0.0, Q.eccentricity - 0.943) / 2.300857e-05   # +1.5%  girth2 > 0.0132 and eccentricity > 0.943
        - 0.01415436 * max(0.0, 0.000261 - Q.lam2) / 0.0001630461   # -1.4%  lam2 < 0.000261
        + 0.01286853 * max(0.0, 0.0892 - Q.girth) * max(0.0, 0.00397 - Q.C2_b2) / 0.000124533   # +1.3%  girth < 0.0892 and C2_b2 < 0.00397
        - 0.01161124 * max(0.0, 0.111 - Q.max_dr) / 0.02801553   # -1.2%  max_dr < 0.111
        + 0.01098916 * max(0.0, Q.mass_over_sum_pt - 0.0904) * max(0.0, 37.4 - Q.pt_6) / 0.02192268   # +1.1%  mass_over_sum_pt > 0.0904 and pt_6 < 37.4
        + 0.01080359 * max(0.0, 0.000311 - Q.lam2) * max(0.0, 0.04 - Q.tau21_b2) / 2.642393e-06   # +1.1%  lam2 < 0.000311 and tau21_b2 < 0.04
        - 0.01031119 * max(0.0, Q.mass - 75.9) / 1.622053   # -1.0%  mass > 75.9
        - 0.01003169 * max(0.0, Q.sj3_dr_max - 0.265) / 0.01827679   # -1.0%  sj3_dr_max > 0.265
        - 0.009493711 * max(0.0, Q.sd_rg - 0.204) / 0.01562278   # -0.9%  sd_rg > 0.204
        - 0.008769244 * max(0.0, 0.00107 - Q.girth2) / 0.0002372304   # -0.9%  girth2 < 0.00107
        - 0.00830652 * max(0.0, 0.0413 - Q.tau21_b2) / 0.01154165   # -0.8%  tau21_b2 < 0.0413
        + 0.00782795 * max(0.0, Q.girth2 - 0.0135) / 0.001404679   # +0.8%  girth2 > 0.0135
        + 0.007784601 * max(0.0, Q.girth2 - 0.0135) * max(0.0, 39.2 - Q.pt_6) / 0.005119395   # +0.8%  girth2 > 0.0135 and pt_6 < 39.2
        + 0.006916471 * max(0.0, Q.z_7 - 0.0333) / 0.02110964   # +0.7%  z_7 > 0.0333
        + 0.005814791 * max(0.0, Q.sd_rg - 0.279) / 0.005101781   # +0.6%  sd_rg > 0.279
        + 0.005505965 * max(0.0, Q.girth2 - 0.00467) * max(0.0, 0.181 - Q.planar_flow) / 0.000223172   # +0.6%  girth2 > 0.00467 and planar_flow < 0.181
        + 0.004433805 * max(0.0, 0.197 - Q.planar_flow) * max(0.0, Q.sd_mass - 39.0) / 1.24668   # +0.4%  planar_flow < 0.197 and sd_mass > 39
        + 0.003991132 * max(0.0, Q.mass_top5 - 53.4) / 2.010166   # +0.4%  mass_top5 > 53.4
        + 0.0036762 * max(0.0, 0.144 - Q.z_dr_0p1_0p2) / 0.08524335   # +0.4%  z_dr_0p1_0p2 < 0.144
        - 0.003555992 * max(0.0, Q.centroid_offset - 0.0381) / 0.00165557   # -0.4%  centroid_offset > 0.0381
        + 0.003396319 * max(0.0, 0.0203 - Q.centroid_offset) * max(0.0, 0.0272 - Q.tau21_b2) / 4.774819e-05   # +0.3%  centroid_offset < 0.0203 and tau21_b2 < 0.0272
        - 0.002911861 * max(0.0, 0.0568 - Q.D2_b2) / 0.005808979   # -0.3%  D2_b2 < 0.0568
        - 0.002785974 * max(0.0, 29.0 - Q.pt_7) / 2.167437   # -0.3%  pt_7 < 29
        - 0.002458926 * max(0.0, Q.mass_over_sum_pt - 0.0155) * max(0.0, 35.0 - Q.pt_6) / 0.09371045   # -0.2%  mass_over_sum_pt > 0.0155 and pt_6 < 35
        - 0.002150958 * max(0.0, Q.girth2 - 0.0133) * max(0.0, 0.0732 - Q.dr_6) / 3.000365e-06   # -0.2%  girth2 > 0.0133 and dr_6 < 0.0732
        - 0.002096883 * max(0.0, Q.mass - 76.8) * max(0.0, Q.zdr_6 - 0.00582) / 0.007199843   # -0.2%  mass > 76.8 and zdr_6 > 0.00582
        + 0.001778878 * max(0.0, 0.00672 - Q.width) * max(0.0, 0.00613 - Q.mean_phi) / 2.326836e-05   # +0.2%  width < 0.00672 and mean_phi < 0.00613
        - 0.001015453 * max(0.0, Q.centroid_offset - 0.0297) * max(0.0, Q.n_pt_above_50 - 3.89) / 0.002295012   # -0.1%  centroid_offset > 0.0297 and n_pt_above_50 > 3.89
        + 0.0008440968 * max(0.0, Q.sd_rg - 0.204) * max(0.0, 0.0702 - Q.dr_6) / 5.057409e-05   # +0.1%  sd_rg > 0.204 and dr_6 < 0.0702
        - 0.0005939233 * max(0.0, 0.00102 - Q.girth2_top2) * max(0.0, 0.0266 - Q.tau21_b2) / 2.039291e-07   # -0.1%  girth2_top2 < 0.00102 and tau21_b2 < 0.0266
        - 0.0003132868 * max(0.0, Q.sj3_dr_max - 0.132) * max(0.0, 19.3 - Q.pt_6) / 0.01356033   # -0.0%  sj3_dr_max > 0.132 and pt_6 < 19.3
        - 0.0001540896 * max(0.0, Q.centroid_offset - 0.0378) * max(0.0, Q.pt_4 - 81.4) / 0.000192393   # -0.0%  centroid_offset > 0.0378 and pt_4 > 81.4
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 10.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.78939 * (-0.05079064
        - 0.3509103 * max(0.0, 0.0594 - Q.girth) / 0.01777515   # -35.1%  girth < 0.0594
        + 0.1594159 * max(0.0, 0.00495 - Q.girth2) / 0.00186551   # +15.9%  girth2 < 0.00495
        + 0.1169255 * max(0.0, 0.0605 - Q.tau1) * max(0.0, 0.00306 - Q.width) / 4.672424e-05   # +11.7%  tau1 < 0.0605 and width < 0.00306
        + 0.1069457 * max(0.0, 0.00351 - Q.width) / 0.001160844   # +10.7%  width < 0.00351
        + 0.08068489 * max(0.0, 0.00643 - Q.girth2) * max(0.0, 0.0258 - Q.centroid_offset) / 4.225925e-05   # +8.1%  girth2 < 0.00643 and centroid_offset < 0.0258
        - 0.06867257 * max(0.0, 31.2 - Q.mass) * max(0.0, 0.0297 - Q.centroid_offset) / 0.1534027   # -6.9%  mass < 31.2 and centroid_offset < 0.0297
        + 0.02922644 * max(0.0, 0.00014 - Q.lam2) * max(0.0, 5.69 - Q.D2_b2) / 0.0003169201   # +2.9%  lam2 < 0.00014 and D2_b2 < 5.69
        + 0.0269329 * max(0.0, 0.0584 - Q.girth) * max(0.0, Q.width - 0.000544) / 7.175049e-06   # +2.7%  girth < 0.0584 and width > 0.000544
        - 0.02560129 * max(0.0, 0.00539 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0057) / 1.347426e-05   # -2.6%  girth2 < 0.00539 and centroid_offset > 0.0057
        - 0.01492198 * max(0.0, 0.143 - Q.sj3_dr_max) * max(0.0, Q.centroid_offset - 0.00799) / 0.0002138101   # -1.5%  sj3_dr_max < 0.143 and centroid_offset > 0.00799
        - 0.01154632 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 3.53e-05 - Q.e3) / 1.046872e-06   # -1.2%  log_sum_pt > 6.7 and e3 < 3.53e-05
        - 0.005188258 * max(0.0, Q.sum_pt_top5 - 670.0) * max(0.0, 8.0 - Q.n_pt_above_10) / 8.21999   # -0.5%  sum_pt_top5 > 670 and n_pt_above_10 < 8
        - 0.003028048 * max(0.0, Q.log_sum_pt - 6.72) * max(0.0, Q.n_dr_0p05_0p1 - 0.999) / 0.01765989   # -0.3%  log_sum_pt > 6.72 and n_dr_0p05_0p1 > 0.999
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 24.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.2363 * (-0.1510132
        + 0.1548645 * max(0.0, 0.00598 - Q.girth2) / 0.002469305   # +15.5%  girth2 < 0.00598
        - 0.07649514 * max(0.0, 0.0724 - Q.mass_over_sum_pt) / 0.02468654   # -7.6%  mass_over_sum_pt < 0.0724
        - 0.07210761 * max(0.0, 0.143 - Q.sj3_dr_max) / 0.03734234   # -7.2%  sj3_dr_max < 0.143
        + 0.07201554 * max(0.0, 0.214 - Q.sj3_dr_max) / 0.07621791   # +7.2%  sj3_dr_max < 0.214
        + 0.06136021 * max(0.0, 0.00364 - Q.girth2) / 0.001218971   # +6.1%  girth2 < 0.00364
        + 0.06079198 * max(0.0, 54.9 - Q.mass) * max(0.0, 0.0266 - Q.centroid_offset) / 0.3069526   # +6.1%  mass < 54.9 and centroid_offset < 0.0266
        - 0.05784191 * max(0.0, 51.6 - Q.mass) / 18.25356   # -5.8%  mass < 51.6
        + 0.05680016 * max(0.0, 991.0 - Q.sum_pt) / 279.2344   # +5.7%  sum_pt < 991
        + 0.05235594 * max(0.0, 0.0179 - Q.centroid_offset) / 0.006408658   # +5.2%  centroid_offset < 0.0179
        - 0.04070923 * max(0.0, 30.4 - Q.mass) / 7.648381   # -4.1%  mass < 30.4
        + 0.03144137 * max(0.0, 0.111 - Q.max_dr) / 0.02801553   # +3.1%  max_dr < 0.111
        + 0.03112496 * max(0.0, 52.7 - Q.mass) * max(0.0, 6.85 - Q.log_sum_pt) / 5.166807   # +3.1%  mass < 52.7 and log_sum_pt < 6.85
        - 0.02780291 * max(0.0, 2.39e-05 - Q.e3) / 1.117478e-05   # -2.8%  e3 < 2.39e-05
        + 0.02500228 * max(0.0, 0.000325 - Q.lam2) / 0.0002141211   # +2.5%  lam2 < 0.000325
        + 0.02049829 * max(0.0, 0.0418 - Q.tau1) / 0.0114735   # +2.0%  tau1 < 0.0418
        + 0.01913947 * max(0.0, Q.lam1 - 0.00845) / 0.001826259   # +1.9%  lam1 > 0.00845
        - 0.01736236 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 164.0 - Q.pt_1) / 0.2032848   # -1.7%  centroid_offset < 0.0179 and pt_1 < 164
        - 0.01583676 * max(0.0, 0.0575 - Q.girth) * max(0.0, Q.z_6 - 0.0334) / 0.0003521326   # -1.6%  girth < 0.0575 and z_6 > 0.0334
        - 0.01476463 * max(0.0, Q.lam1 - 0.00596) * max(0.0, Q.eccentricity - 0.708) / 0.000494254   # -1.5%  lam1 > 0.00596 and eccentricity > 0.708
        - 0.01172098 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.n_for_90pct - 5.02) / 0.008713901   # -1.2%  centroid_offset < 0.0185 and n_for_90pct > 5.02
        + 0.01027994 * max(0.0, 0.0191 - Q.centroid_offset) * max(0.0, 0.205 - Q.z_2nd) / 0.0002306924   # +1.0%  centroid_offset < 0.0191 and z_2nd < 0.205
        + 0.008523913 * max(0.0, 0.146 - Q.sj3_dr_max) * max(0.0, Q.pt_6 - 33.3) / 0.4156702   # +0.9%  sj3_dr_max < 0.146 and pt_6 > 33.3
        - 0.006603365 * max(0.0, 0.0235 - Q.absphi_1) / 0.007513668   # -0.7%  absphi_1 < 0.0235
        - 0.006598733 * max(0.0, 0.0064 - Q.zdr_0) / 0.001038499   # -0.7%  zdr_0 < 0.0064
        + 0.005995464 * max(0.0, Q.lam2 - 0.00153) / 0.0002757265   # +0.6%  lam2 > 0.00153
        + 0.004941226 * max(0.0, 0.000168 - Q.width) / 1.379689e-05   # +0.5%  width < 0.000168
        - 0.004862397 * max(0.0, Q.D2 - 2.2) / 0.2792571   # -0.5%  D2 > 2.2
        - 0.004705773 * max(0.0, Q.mass - 45.5) / 9.50421   # -0.5%  mass > 45.5
        - 0.004114457 * max(0.0, 0.00594 - Q.girth2) * max(0.0, 0.00379 - Q.mean_phi) / 1.455755e-05   # -0.4%  girth2 < 0.00594 and mean_phi < 0.00379
        + 0.003378337 * max(0.0, 0.00677 - Q.girth2) * max(0.0, 0.296 - Q.planar_flow) / 0.000245881   # +0.3%  girth2 < 0.00677 and planar_flow < 0.296
        + 0.002395454 * max(0.0, 0.0185 - Q.centroid_offset) * max(0.0, Q.mean_eta - 0.000187) / 1.161139e-05   # +0.2%  centroid_offset < 0.0185 and mean_eta > 0.000187
        + 0.002284109 * max(0.0, 53.3 - Q.mass) * max(0.0, Q.D2 - 2.81) / 5.64881   # +0.2%  mass < 53.3 and D2 > 2.81
        + 0.002202091 * max(0.0, 0.0552 - Q.girth) * max(0.0, 0.0289 - Q.tau21_b2) / 1.542501e-05   # +0.2%  girth < 0.0552 and tau21_b2 < 0.0289
        - 0.002101641 * max(0.0, 0.0447 - Q.tau1) * max(0.0, 0.0309 - Q.tau21_b2) / 1.198494e-05   # -0.2%  tau1 < 0.0447 and tau21_b2 < 0.0309
        - 0.002091103 * max(0.0, Q.sd_mass - 44.8) * max(0.0, Q.n_dr_0p05_0p1 - 6.78) / 0.7138113   # -0.2%  sd_mass > 44.8 and n_dr_0p05_0p1 > 6.78
        + 0.002010715 * max(0.0, 0.0643 - Q.mass_over_sum_pt) * max(0.0, -0.00767 - Q.phi_0) / 5.245672e-05   # +0.2%  mass_over_sum_pt < 0.0643 and phi_0 < -0.00767
        - 0.001601035 * max(0.0, 0.0736 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top2 - 1.83) / 0.01949907   # -0.2%  mass_over_sum_pt < 0.0736 and mass_top2 > 1.83
        + 0.001251058 * max(0.0, 36.2 - Q.pt_4) / 0.6768084   # +0.1%  pt_4 < 36.2
        - 0.001248709 * max(0.0, 0.00228 - Q.centroid_offset) / 9.517009e-05   # -0.1%  centroid_offset < 0.00228
        - 0.0009910089 * max(0.0, Q.log_sum_pt - 6.87) * max(0.0, 0.754 - Q.D2_b2) / 0.0009273509   # -0.1%  log_sum_pt > 6.87 and D2_b2 < 0.754
        - 0.00068225 * max(0.0, 0.00585 - Q.girth2) * max(0.0, Q.mean_phi - 0.0255) / 3.149565e-07   # -0.1%  girth2 < 0.00585 and mean_phi > 0.0255
        + 0.0006436171 * max(0.0, 0.0446 - Q.tau1) * max(0.0, Q.mean_phi - 0.0255) / 3.311868e-06   # +0.1%  tau1 < 0.0446 and mean_phi > 0.0255
        - 0.0004573447 * max(0.0, 54.4 - Q.mass) * max(0.0, Q.dr1_7 - 0.209) / 0.02348378   # -0.0%  mass < 54.4 and dr1_7 > 0.209
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 6.585;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.584868 * (0.2247577
        - 0.14985 * max(0.0, 0.00435 - Q.lam1) / 0.001596671   # -15.0%  lam1 < 0.00435
        + 0.1455427 * max(0.0, 76.0 - Q.mass) / 37.29104   # +14.6%  mass < 76
        + 0.08204588 * max(0.0, Q.lam2 - 0.000304) / 0.0004220792   # +8.2%  lam2 > 0.000304
        - 0.06222365 * max(0.0, Q.LHA - 0.304) / 0.01773743   # -6.2%  LHA > 0.304
        - 0.0597259 * max(0.0, 0.00165 - Q.lam1) / 0.0004438907   # -6.0%  lam1 < 0.00165
        + 0.05513879 * max(0.0, Q.girth2 - 0.00732) / 0.002521401   # +5.5%  girth2 > 0.00732
        - 0.04161798 * max(0.0, 45.0 - Q.pt_7) / 11.51466   # -4.2%  pt_7 < 45
        - 0.0396153 * max(0.0, 0.0208 - Q.zdr_0) / 0.008753743   # -4.0%  zdr_0 < 0.0208
        + 0.03765295 * max(0.0, 0.0213 - Q.zdr_0) * max(0.0, 0.26 - Q.z_dr_0p05_0p1) / 0.001850296   # +3.8%  zdr_0 < 0.0213 and z_dr_0p05_0p1 < 0.26
        + 0.03135866 * max(0.0, 0.00217 - Q.girth2_top3) / 0.00078814   # +3.1%  girth2_top3 < 0.00217
        + 0.02997524 * max(0.0, Q.tau1 - 0.0557) / 0.02570091   # +3.0%  tau1 > 0.0557
        - 0.02988187 * max(0.0, Q.lam2 - 0.00023) * max(0.0, Q.planar_flow - 0.0514) / 0.0002831197   # -3.0%  lam2 > 0.00023 and planar_flow > 0.0514
        + 0.02948792 * max(0.0, Q.mass_top5 - 22.8) / 12.36777   # +2.9%  mass_top5 > 22.8
        + 0.02772529 * max(0.0, Q.sj3_pair_mass_min - 15.2) / 1.426308   # +2.8%  sj3_pair_mass_min > 15.2
        + 0.02286118 * max(0.0, Q.tau1 - 0.0451) * max(0.0, 0.973 - Q.D2) / 0.01045402   # +2.3%  tau1 > 0.0451 and D2 < 0.973
        + 0.02054193 * max(0.0, 0.0229 - Q.centroid_offset) / 0.009873422   # +2.1%  centroid_offset < 0.0229
        + 0.02045829 * max(0.0, 45.6 - Q.pt_7) * max(0.0, 1.0 - Q.D2) / 1.779593   # +2.0%  pt_7 < 45.6 and D2 < 1
        + 0.01737651 * max(0.0, Q.sj3_dr_min - 0.127) / 0.008291453   # +1.7%  sj3_dr_min > 0.127
        - 0.01216068 * max(0.0, Q.mass_top5 - 45.4) / 3.590872   # -1.2%  mass_top5 > 45.4
        - 0.01084258 * max(0.0, 44.7 - Q.pt_7) * max(0.0, 6.54 - Q.log_sum_pt) / 1.011288   # -1.1%  pt_7 < 44.7 and log_sum_pt < 6.54
        + 0.009967434 * max(0.0, 0.0209 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0107) / 3.9068e-05   # +1.0%  zdr_0 < 0.0209 and centroid_offset > 0.0107
        - 0.009894115 * max(0.0, Q.sj3_pair_mass_min - 9.6) * max(0.0, Q.sj3_pairmin_over_m - 0.292) / 0.2505825   # -1.0%  sj3_pair_mass_min > 9.6 and sj3_pairmin_over_m > 0.292
        - 0.00955208 * max(0.0, 0.0255 - Q.M2) / 0.001839158   # -1.0%  M2 < 0.0255
        - 0.009158458 * max(0.0, Q.sj3_dr_min - 0.134) * max(0.0, 0.43 - Q.z_dr_0_0p05) / 0.002716542   # -0.9%  sj3_dr_min > 0.134 and z_dr_0_0p05 < 0.43
        - 0.008802613 * max(0.0, Q.lam1 - 0.00686) * max(0.0, 0.383 - Q.D2_b2) / 0.0003184838   # -0.9%  lam1 > 0.00686 and D2_b2 < 0.383
        - 0.007704929 * max(0.0, 7.95e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.177) / 5.961921e-07   # -0.8%  e3 < 7.95e-05 and sj3_dr23 > 0.177
        - 0.005470195 * max(0.0, 0.00619 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.215) / 0.0001026225   # -0.5%  lam1 < 0.00619 and z_dr_0p05_0p1 > 0.215
        - 0.004690816 * max(0.0, Q.sum_pt - 986.0) / 4.522461   # -0.5%  sum_pt > 986
        + 0.003755877 * max(0.0, 0.022 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.0771) / 0.0001413255   # +0.4%  zdr_0 < 0.022 and dr_7 > 0.0771
        - 0.003654051 * max(0.0, Q.girth2 - 0.0253) / 0.0002854264   # -0.4%  girth2 > 0.0253
        - 0.001266099 * max(0.0, Q.lam2 - 0.000412) * max(0.0, 31.9 - Q.pt_6) / 0.0003459375   # -0.1%  lam2 > 0.000412 and pt_6 < 31.9
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 30.53;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.52846 * (-0.02427243
        + 0.2797394 * max(0.0, 0.00865 - Q.width) / 0.004424878   # +28.0%  width < 0.00865
        - 0.1341159 * max(0.0, 0.00806 - Q.e2_sq) / 0.004207966   # -13.4%  e2_sq < 0.00806
        + 0.0819548 * max(0.0, 0.0132 - Q.girth2) / 0.008203127   # +8.2%  girth2 < 0.0132
        - 0.06791939 * max(0.0, 0.0084 - Q.lam1) / 0.004319738   # -6.8%  lam1 < 0.0084
        - 0.06176115 * max(0.0, 0.0722 - Q.girth) / 0.02534237   # -6.2%  girth < 0.0722
        + 0.05347408 * max(0.0, Q.z_7 - 0.0167) / 0.03572169   # +5.3%  z_7 > 0.0167
        - 0.04586667 * max(0.0, 68.2 - Q.mass) / 30.50629   # -4.6%  mass < 68.2
        - 0.03960178 * max(0.0, 0.169 - Q.sj3_dr_max) / 0.04874925   # -4.0%  sj3_dr_max < 0.169
        + 0.03296252 * max(0.0, 0.263 - Q.sj3_dr_max) / 0.1134493   # +3.3%  sj3_dr_max < 0.263
        + 0.02263802 * max(0.0, 0.0237 - Q.centroid_offset) / 0.01047127   # +2.3%  centroid_offset < 0.0237
        + 0.02119882 * max(0.0, 0.248 - Q.planar_flow) / 0.1062672   # +2.1%  planar_flow < 0.248
        - 0.01983575 * max(0.0, 0.0392 - Q.centroid_offset) * max(0.0, 895.0 - Q.sum_pt) / 3.626077   # -2.0%  centroid_offset < 0.0392 and sum_pt < 895
        - 0.01692718 * max(0.0, 0.014 - Q.centroid_offset) * max(0.0, 16.6 - Q.sj3_pair_mass_min) / 0.05116443   # -1.7%  centroid_offset < 0.014 and sj3_pair_mass_min < 16.6
        - 0.01331857 * max(0.0, Q.pt_7 - 28.6) / 8.099509   # -1.3%  pt_7 > 28.6
        + 0.01076709 * max(0.0, Q.centroid_offset - 0.0467) / 0.0009841398   # +1.1%  centroid_offset > 0.0467
        + 0.009052823 * max(0.0, 0.143 - Q.max_dr) / 0.04637059   # +0.9%  max_dr < 0.143
        - 0.009034172 * max(0.0, 0.0263 - Q.girth) / 0.004153605   # -0.9%  girth < 0.0263
        - 0.008339309 * max(0.0, Q.centroid_offset - 0.0231) / 0.00410623   # -0.8%  centroid_offset > 0.0231
        - 0.007928062 * max(0.0, Q.centroid_offset - 0.0488) * max(0.0, Q.tau4 - 0.00201) / 2.80779e-06   # -0.8%  centroid_offset > 0.0488 and tau4 > 0.00201
        + 0.007786941 * max(0.0, 15.8 - Q.mass) / 2.489249   # +0.8%  mass < 15.8
        + 0.007534104 * max(0.0, 0.0167 - Q.e2) / 0.00394519   # +0.8%  e2 < 0.0167
        - 0.007377528 * max(0.0, 0.245 - Q.planar_flow) * max(0.0, 847.0 - Q.sum_pt) / 14.43747   # -0.7%  planar_flow < 0.245 and sum_pt < 847
        - 0.004932913 * max(0.0, Q.centroid_offset - 0.0481) * max(0.0, 3.22 - Q.D2) / 0.001856896   # -0.5%  centroid_offset > 0.0481 and D2 < 3.22
        + 0.004670196 * max(0.0, 0.0137 - Q.centroid_offset) * max(0.0, 0.592 - Q.D2_b2) / 0.000783373   # +0.5%  centroid_offset < 0.0137 and D2_b2 < 0.592
        + 0.00418992 * max(0.0, Q.sj2_dr - 0.266) / 0.008882763   # +0.4%  sj2_dr > 0.266
        - 0.003876599 * max(0.0, Q.n_dr_0p1_0p2 - 3.01) / 0.3305771   # -0.4%  n_dr_0p1_0p2 > 3.01
        + 0.003644564 * max(0.0, Q.eccentricity - 0.988) / 0.002107252   # +0.4%  eccentricity > 0.988
        - 0.003581266 * max(0.0, 501.0 - Q.sum_pt_top5) / 32.73369   # -0.4%  sum_pt_top5 < 501
        - 0.003056883 * max(0.0, 0.255 - Q.planar_flow) * max(0.0, 37.0 - Q.pt_7) / 0.5489524   # -0.3%  planar_flow < 0.255 and pt_7 < 37
        - 0.002184886 * max(0.0, 0.247 - Q.planar_flow) * max(0.0, 9.97e-06 - Q.e3) / 1.544009e-07   # -0.2%  planar_flow < 0.247 and e3 < 9.97e-06
        - 0.0021155 * max(0.0, Q.eccentricity - 0.988) * max(0.0, 2.9 - Q.sj2_mass2) / 0.003509942   # -0.2%  eccentricity > 0.988 and sj2_mass2 < 2.9
        - 0.001666599 * max(0.0, 0.252 - Q.planar_flow) * max(0.0, Q.sj3_pair_mass_min - 1.69) / 0.3583007   # -0.2%  planar_flow < 0.252 and sj3_pair_mass_min > 1.69
        - 0.001587462 * max(0.0, 0.042 - Q.centroid_offset) * max(0.0, 0.0004 - Q.mean_phi) / 8.716323e-05   # -0.2%  centroid_offset < 0.042 and mean_phi < 0.0004
        + 0.001394957 * max(0.0, 0.267 - Q.sj3_dr_max) * max(0.0, -0.0184 - Q.phi_0) / 0.0004674631   # +0.1%  sj3_dr_max < 0.267 and phi_0 < -0.0184
        - 0.0009988871 * max(0.0, Q.max_pair_mass - 32.4) / 1.030219   # -0.1%  max_pair_mass > 32.4
        - 0.0009246172 * max(0.0, 24.2 - Q.pt_6) / 0.5832053   # -0.1%  pt_6 < 24.2
        - 0.0008527899 * max(0.0, 0.00865 - Q.e2_sq) * max(0.0, Q.mass_top2 - 9.42) / 0.005785414   # -0.1%  e2_sq < 0.00865 and mass_top2 > 9.42
        - 0.0007450974 * max(0.0, Q.centroid_offset - 0.0504) * max(0.0, Q.zdr_7 - 0.00649) / 3.004845e-06   # -0.1%  centroid_offset > 0.0504 and zdr_7 > 0.00649
        + 0.0004428042 * max(0.0, Q.sj2_dr - 0.177) * max(0.0, 0.0469 - Q.dr_7) / 7.637361e-05   # +0.0%  sj2_dr > 0.177 and dr_7 < 0.0469
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5669;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5668755 * (-0.7814767
        + 0.4739395 * max(0.0, Q.girth2 - 0.0188) / 0.0007632519   # +47.4%  girth2 > 0.0188
        - 0.2373794 * max(0.0, Q.mass_over_sum_pt - 0.131) / 0.002534172   # -23.7%  mass_over_sum_pt > 0.131
        + 0.08483567 * max(0.0, Q.mass - 88.2) / 0.7210084   # +8.5%  mass > 88.2
        - 0.07223936 * max(0.0, Q.girth2 - 0.0188) * max(0.0, 53.4 - Q.pt_7) / 0.01441927   # -7.2%  girth2 > 0.0188 and pt_7 < 53.4
        - 0.0591522 * max(0.0, Q.girth2_top2 - 0.014) / 0.001132836   # -5.9%  girth2_top2 > 0.014
        + 0.02672712 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.lam2 - 0.000537) / 2.837257e-06   # +2.7%  girth2 > 0.0188 and lam2 > 0.000537
        - 0.02505982 * max(0.0, Q.zdr_0 - 0.0398) / 0.000570514   # -2.5%  zdr_0 > 0.0398
        + 0.02066697 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.abseta_6 - 0.0284) / 6.332756e-05   # +2.1%  girth2 > 0.0188 and abseta_6 > 0.0284
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 15.55;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.55257 * (0.02584781
        + 0.3669345 * max(0.0, 0.149 - Q.girth) / 0.09116256   # +36.7%  girth < 0.149
        - 0.1589761 * max(0.0, 0.0799 - Q.e2) / 0.05183412   # -15.9%  e2 < 0.0799
        + 0.1279065 * max(0.0, 0.016 - Q.lam1) / 0.01081128   # +12.8%  lam1 < 0.016
        - 0.08499052 * max(0.0, 0.149 - Q.girth) * max(0.0, 6.81 - Q.log_sum_pt) / 0.0205571   # -8.5%  girth < 0.149 and log_sum_pt < 6.81
        - 0.04859415 * max(0.0, 5.3e-05 - Q.e3) * max(0.0, 0.0378 - Q.centroid_offset) / 8.268755e-07   # -4.9%  e3 < 5.3e-05 and centroid_offset < 0.0378
        - 0.04080902 * max(0.0, 0.145 - Q.girth) * max(0.0, 38.3 - Q.pt_7) / 0.6359572   # -4.1%  girth < 0.145 and pt_7 < 38.3
        + 0.03608469 * max(0.0, Q.sum_pt_top5 - 662.0) * max(0.0, 40.4 - Q.pt_7) / 778.377   # +3.6%  sum_pt_top5 > 662 and pt_7 < 40.4
        - 0.02445498 * max(0.0, 0.142 - Q.girth) * max(0.0, 0.000592 - Q.lam2) / 4.211936e-05   # -2.4%  girth < 0.142 and lam2 < 0.000592
        + 0.02203895 * max(0.0, 5.1e-05 - Q.e3) / 3.116023e-05   # +2.2%  e3 < 5.1e-05
        - 0.01752102 * max(0.0, 31.8 - Q.pt_6) * max(0.0, 0.0579 - Q.z_7) / 0.06566192   # -1.8%  pt_6 < 31.8 and z_7 < 0.0579
        + 0.01148792 * max(0.0, Q.sum_pt_top5 - 789.0) * max(0.0, 35.9 - Q.pt_6) / 180.8367   # +1.1%  sum_pt_top5 > 789 and pt_6 < 35.9
        - 0.010181 * max(0.0, 0.0273 - Q.z_7) / 0.001199551   # -1.0%  z_7 < 0.0273
        + 0.009182822 * max(0.0, 6.01e-05 - Q.e3) * max(0.0, Q.D3 - 0.207) / 4.533858e-05   # +0.9%  e3 < 6.01e-05 and D3 > 0.207
        - 0.008344697 * max(0.0, Q.mass - 50.0) / 7.589563   # -0.8%  mass > 50
        + 0.008148676 * max(0.0, 0.152 - Q.girth) * max(0.0, Q.z_7 - 0.0616) / 0.0003631315   # +0.8%  girth < 0.152 and z_7 > 0.0616
        + 0.004727185 * max(0.0, 0.149 - Q.girth) * max(0.0, 0.0747 - Q.M3) / 0.0006807399   # +0.5%  girth < 0.149 and M3 < 0.0747
        - 0.004345647 * max(0.0, 0.0276 - Q.z_5) / 0.0003520105   # -0.4%  z_5 < 0.0276
        + 0.004287494 * max(0.0, 0.144 - Q.girth) * max(0.0, Q.tau2 - 0.00941) / 0.0002315332   # +0.4%  girth < 0.144 and tau2 > 0.00941
        - 0.004257584 * max(0.0, Q.sj3_dr23 - 0.2) / 0.02416657   # -0.4%  sj3_dr23 > 0.2
        - 0.003545113 * max(0.0, Q.sum_pt - 995.0) * max(0.0, 54.8 - Q.pt_6) / 72.45156   # -0.4%  sum_pt > 995 and pt_6 < 54.8
        - 0.002012774 * max(0.0, Q.pt_7 - 48.6) / 0.6879959   # -0.2%  pt_7 > 48.6
        - 0.001168625 * max(0.0, 0.146 - Q.girth) * max(0.0, Q.sj2_mass1 - 31.7) / 0.0112889   # -0.1%  girth < 0.146 and sj2_mass1 > 31.7
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 34.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.17753 * (-0.005061805
        - 0.1035597 * max(0.0, Q.width - 0.00668) / 0.002701842   # -10.4%  width > 0.00668
        - 0.07717718 * max(0.0, Q.girth2 - 0.00754) / 0.002465164   # -7.7%  girth2 > 0.00754
        - 0.06125326 * max(0.0, Q.girth - 0.0873) / 0.007840768   # -6.1%  girth > 0.0873
        - 0.05784556 * max(0.0, 0.325 - Q.sd_rg) / 0.2085463   # -5.8%  sd_rg < 0.325
        + 0.05729952 * max(0.0, Q.girth - 0.027) / 0.03606549   # +5.7%  girth > 0.027
        + 0.04554512 * max(0.0, Q.mass_over_sum_pt - 0.0848) / 0.009549813   # +4.6%  mass_over_sum_pt > 0.0848
        - 0.04534057 * max(0.0, Q.e2 - 0.0167) / 0.01587734   # -4.5%  e2 > 0.0167
        + 0.04507144 * max(0.0, Q.mass_over_sum_pt - 0.0798) / 0.01092504   # +4.5%  mass_over_sum_pt > 0.0798
        + 0.04270777 * max(0.0, Q.girth - 0.0412) / 0.02683173   # +4.3%  girth > 0.0412
        + 0.03898902 * max(0.0, Q.width - 0.00348) / 0.004112803   # +3.9%  width > 0.00348
        - 0.033409 * max(0.0, Q.e2_sq - 0.0116) / 0.001441714   # -3.3%  e2_sq > 0.0116
        + 0.03269564 * max(0.0, Q.sj2_dr - 0.159) / 0.03934705   # +3.3%  sj2_dr > 0.159
        + 0.03170109 * max(0.0, Q.e2 - 0.0501) / 0.003396442   # +3.2%  e2 > 0.0501
        + 0.02928969 * max(0.0, 49.2 - Q.sd_mass) / 22.1471   # +2.9%  sd_mass < 49.2
        - 0.0286032 * max(0.0, Q.sj2_dr - 0.0614) / 0.1003682   # -2.9%  sj2_dr > 0.0614
        - 0.02749754 * max(0.0, 7.91e-05 - Q.e3) / 5.432358e-05   # -2.7%  e3 < 7.91e-05
        - 0.02429266 * max(0.0, Q.girth - 0.102) / 0.005391318   # -2.4%  girth > 0.102
        + 0.01834169 * max(0.0, Q.z_dr_0_0p05 - 0.152) / 0.447767   # +1.8%  z_dr_0_0p05 > 0.152
        - 0.01757374 * max(0.0, Q.centroid_offset - 0.0498) / 0.0008116582   # -1.8%  centroid_offset > 0.0498
        + 0.0158713 * max(0.0, Q.girth - 0.0807) / 0.009320306   # +1.6%  girth > 0.0807
        + 0.01385674 * max(0.0, Q.LHA - 0.304) / 0.01773743   # +1.4%  LHA > 0.304
        + 0.01252225 * max(0.0, 0.000504 - Q.lam2) / 0.0003626945   # +1.3%  lam2 < 0.000504
        - 0.01241832 * max(0.0, 0.00406 - Q.C2_b2) / 0.002907038   # -1.2%  C2_b2 < 0.00406
        + 0.01200344 * max(0.0, Q.sj2_dr - 0.159) * max(0.0, 0.0866 - Q.D2_b2) / 0.0006225312   # +1.2%  sj2_dr > 0.159 and D2_b2 < 0.0866
        - 0.01147537 * max(0.0, 724.0 - Q.sum_pt) / 74.28024   # -1.1%  sum_pt < 724
        + 0.01121471 * max(0.0, 5.06 - Q.n_dr_0_0p05) / 1.975727   # +1.1%  n_dr_0_0p05 < 5.06
        + 0.01047907 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0164 - Q.lam1) / 0.0003347186   # +1.0%  planar_flow < 0.115 and lam1 < 0.0164
        - 0.008465995 * max(0.0, Q.sj2_dr - 0.199) / 0.02352413   # -0.8%  sj2_dr > 0.199
        - 0.007528308 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.087 - Q.D2_b2) / 0.0009709395   # -0.8%  sj2_dr > 0.13 and D2_b2 < 0.087
        - 0.005756249 * max(0.0, 0.163 - Q.z_dr_0p05_0p1) / 0.08628701   # -0.6%  z_dr_0p05_0p1 < 0.163
        - 0.00528871 * max(0.0, Q.sd_rg - 0.234) / 0.01057047   # -0.5%  sd_rg > 0.234
        - 0.004928937 * max(0.0, Q.sj2_dr - 0.2) * max(0.0, 0.0864 - Q.D2_b2) / 0.0003264707   # -0.5%  sj2_dr > 0.2 and D2_b2 < 0.0864
        - 0.004926872 * max(0.0, Q.z_dr_0p05_0p1 - 0.75) * max(0.0, 0.024 - Q.C2_b2) / 0.0005245742   # -0.5%  z_dr_0p05_0p1 > 0.75 and C2_b2 < 0.024
        - 0.004810294 * max(0.0, Q.width - 0.0119) / 0.001647334   # -0.5%  width > 0.0119
        - 0.004601716 * max(0.0, Q.psi_0p1 - 0.973) / 0.01191479   # -0.5%  psi_0p1 > 0.973
        - 0.004334927 * max(0.0, 0.105 - Q.planar_flow) * max(0.0, 0.0074 - Q.lam1) / 5.467051e-05   # -0.4%  planar_flow < 0.105 and lam1 < 0.0074
        + 0.004194866 * max(0.0, Q.z_dr_0p05_0p1 - 0.752) * max(0.0, 1.01 - Q.n_dr_0p2_0p4) / 0.02074821   # +0.4%  z_dr_0p05_0p1 > 0.752 and n_dr_0p2_0p4 < 1.01
        + 0.003576165 * max(0.0, Q.centroid_offset - 0.0497) * max(0.0, 0.000829 - Q.C2_b2) / 1.346085e-07   # +0.4%  centroid_offset > 0.0497 and C2_b2 < 0.000829
        - 0.003453794 * max(0.0, Q.centroid_offset - 0.0381) / 0.00165557   # -0.3%  centroid_offset > 0.0381
        + 0.002821259 * max(0.0, Q.sd_rg - 0.279) / 0.005101781   # +0.3%  sd_rg > 0.279
        + 0.001923138 * max(0.0, 5.6e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.164) / 4.295955e-07   # +0.2%  e3 < 5.6e-05 and sj3_dr23 > 0.164
        - 0.001879266 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, 0.0183 - Q.centroid_offset) / 0.0002119759   # -0.2%  planar_flow < 0.109 and centroid_offset < 0.0183
        + 0.001713242 * max(0.0, 0.601 - Q.z_dr_0p05_0p1) * max(0.0, 720.0 - Q.sum_pt) / 23.42176   # +0.2%  z_dr_0p05_0p1 < 0.601 and sum_pt < 720
        - 0.001711546 * max(0.0, Q.mass - 71.0) / 2.207411   # -0.2%  mass > 71
        - 0.001358483 * max(0.0, 0.095 - Q.planar_flow) * max(0.0, 667.0 - Q.sum_pt_top5) / 2.405678   # -0.1%  planar_flow < 0.095 and sum_pt_top5 < 667
        - 0.001177875 * max(0.0, 0.579 - Q.z_dr_0p05_0p1) * max(0.0, 0.184 - Q.D2_b2) / 0.008439595   # -0.1%  z_dr_0p05_0p1 < 0.579 and D2_b2 < 0.184
        - 0.001087875 * max(0.0, Q.e2_sq - 0.00194) * max(0.0, 0.184 - Q.sj2_dr) / 1.831571e-05   # -0.1%  e2_sq > 0.00194 and sj2_dr < 0.184
        - 0.001067261 * max(0.0, Q.sj2_dr - 0.199) * max(0.0, 0.106 - Q.z_5) / 0.0007760926   # -0.1%  sj2_dr > 0.199 and z_5 < 0.106
        - 0.001052455 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, Q.mean_eta - -0.00694) / 0.0002997526   # -0.1%  planar_flow < 0.106 and mean_eta > -0.00694
        + 0.001044467 * max(0.0, Q.sj2_dr - 0.198) * max(0.0, 0.0508 - Q.dr_2) / 0.0001317244   # +0.1%  sj2_dr > 0.198 and dr_2 < 0.0508
        - 0.0009656653 * max(0.0, Q.sj2_dr - 0.16) * max(0.0, 0.033 - Q.dr_2) / 6.238952e-05   # -0.1%  sj2_dr > 0.16 and dr_2 < 0.033
        - 0.0007105365 * max(0.0, Q.sd_rg - 0.278) * max(0.0, 0.263 - Q.D2_b2) / 0.0004081409   # -0.1%  sd_rg > 0.278 and D2_b2 < 0.263
        + 0.0006454679 * max(0.0, Q.psi_0p1 - 0.821) * max(0.0, 0.0838 - Q.D2_b2) / 0.00058828   # +0.1%  psi_0p1 > 0.821 and D2_b2 < 0.0838
        + 0.0004807403 * max(0.0, 0.589 - Q.z_dr_0p05_0p1) * max(0.0, 0.315 - Q.D3) / 0.004551389   # +0.0%  z_dr_0p05_0p1 < 0.589 and D3 < 0.315
        - 0.0002630034 * max(0.0, Q.girth - 0.027) * max(0.0, Q.D2_b2 - 4.77) / 0.0001139266   # -0.0%  girth > 0.027 and D2_b2 > 4.77
        - 0.0001963437 * max(0.0, Q.z_dr_0p05_0p1 - 0.754) * max(0.0, Q.zdr_5 - 0.0156) / 1.690313e-06   # -0.0%  z_dr_0p05_0p1 > 0.754 and zdr_5 > 0.0156
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 29.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.11082 * (-0.06011511
        + 0.2040645 * max(0.0, 0.0187 - Q.width) * max(0.0, 0.00343 - Q.lam2) / 4.304699e-05   # +20.4%  width < 0.0187 and lam2 < 0.00343
        - 0.1288478 * max(0.0, 0.0992 - Q.girth) / 0.04630696   # -12.9%  girth < 0.0992
        - 0.08535765 * max(0.0, 0.00754 - Q.girth2) / 0.00355993   # -8.5%  girth2 < 0.00754
        - 0.07872398 * max(0.0, 0.00834 - Q.e2_sq) / 0.004432726   # -7.9%  e2_sq < 0.00834
        + 0.06882559 * max(0.0, 0.0622 - Q.tau1) / 0.02142855   # +6.9%  tau1 < 0.0622
        + 0.05217568 * max(0.0, 0.0416 - Q.e2) / 0.01795362   # +5.2%  e2 < 0.0416
        - 0.04978366 * max(0.0, 0.2 - Q.sj2_dr) / 0.07319409   # -5.0%  sj2_dr < 0.2
        + 0.04548564 * max(0.0, 0.163 - Q.sj2_dr) / 0.05034692   # +4.5%  sj2_dr < 0.163
        - 0.04089064 * max(0.0, 0.00498 - Q.lam1) / 0.001932403   # -4.1%  lam1 < 0.00498
        + 0.02780775 * max(0.0, 0.00752 - Q.girth2_top2) / 0.004447837   # +2.8%  girth2_top2 < 0.00752
        - 0.02332019 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, 0.286 - Q.dr01) / 0.001539387   # -2.3%  centroid_offset < 0.0179 and dr01 < 0.286
        - 0.02224768 * max(0.0, 0.0199 - Q.centroid_offset) * max(0.0, 0.0123 - Q.zdr_0) / 4.151591e-05   # -2.2%  centroid_offset < 0.0199 and zdr_0 < 0.0123
        + 0.01811838 * max(0.0, Q.mass - 45.8) / 9.368397   # +1.8%  mass > 45.8
        + 0.01754722 * max(0.0, 0.00686 - Q.girth2_top3) / 0.003728569   # +1.8%  girth2_top3 < 0.00686
        - 0.01697345 * max(0.0, Q.z_dr_0p05_0p1 - 0.735) / 0.02533903   # -1.7%  z_dr_0p05_0p1 > 0.735
        + 0.01565076 * max(0.0, 0.000524 - Q.girth2_top2) / 0.0001139016   # +1.6%  girth2_top2 < 0.000524
        - 0.01485852 * max(0.0, 0.000329 - Q.lam2) / 0.0002173586   # -1.5%  lam2 < 0.000329
        + 0.01397268 * max(0.0, Q.z_dr_0p05_0p1 - 0.741) * max(0.0, 2.03 - Q.n_dr_0p2_0p4) / 0.04611749   # +1.4%  z_dr_0p05_0p1 > 0.741 and n_dr_0p2_0p4 < 2.03
        + 0.01213498 * max(0.0, 0.226 - Q.N2) * max(0.0, Q.sum_pt_top5 - 426.0) / 9.22348   # +1.2%  N2 < 0.226 and sum_pt_top5 > 426
        - 0.01172157 * max(0.0, 0.228 - Q.N2) / 0.05356741   # -1.2%  N2 < 0.228
        + 0.009304974 * max(0.0, 0.0106 - Q.e2_sq) * max(0.0, Q.lam2 - -1.14e-06) / 6.590642e-07   # +0.9%  e2_sq < 0.0106 and lam2 > -1.14e-06
        + 0.007883496 * max(0.0, 0.217 - Q.N2) * max(0.0, 717.0 - Q.sum_pt_top5) / 6.830209   # +0.8%  N2 < 0.217 and sum_pt_top5 < 717
        - 0.006537408 * max(0.0, 0.00111 - Q.lam2) * max(0.0, Q.mean_phi - -0.0179) / 1.66938e-05   # -0.7%  lam2 < 0.00111 and mean_phi > -0.0179
        - 0.00490626 * max(0.0, Q.sj3_pair_mass_max - 38.7) / 7.556891   # -0.5%  sj3_pair_mass_max > 38.7
        - 0.004395166 * max(0.0, 0.00758 - Q.girth2) * max(0.0, 0.771 - Q.D2) / 0.0001040219   # -0.4%  girth2 < 0.00758 and D2 < 0.771
        + 0.003679585 * max(0.0, 0.23 - Q.N2) * max(0.0, 4.02 - Q.n_dr_0p1_0p2) / 0.1182293   # +0.4%  N2 < 0.23 and n_dr_0p1_0p2 < 4.02
        - 0.002991063 * max(0.0, 0.229 - Q.N2) * max(0.0, 0.56 - Q.z_dr_0p05_0p1) / 0.01141183   # -0.3%  N2 < 0.229 and z_dr_0p05_0p1 < 0.56
        - 0.002472281 * max(0.0, Q.mass - 77.3) / 1.48392   # -0.2%  mass > 77.3
        + 0.002023873 * max(0.0, 0.00611 - Q.width) * max(0.0, 0.713 - Q.D2) / 2.753112e-05   # +0.2%  width < 0.00611 and D2 < 0.713
        + 0.001685974 * max(0.0, 0.0182 - Q.width) * max(0.0, Q.dr1_3 - 0.0627) / 0.0002711607   # +0.2%  width < 0.0182 and dr1_3 > 0.0627
        + 0.001337839 * max(0.0, 0.065 - Q.e2) * max(0.0, Q.dr12 - 0.0982) / 0.0003894559   # +0.1%  e2 < 0.065 and dr12 > 0.0982
        - 0.001271291 * max(0.0, 6.09 - Q.log_sum_pt) / 0.006561758   # -0.1%  log_sum_pt < 6.09
        + 0.0007672014 * max(0.0, 0.0187 - Q.width) * max(0.0, Q.ptdr0_2 - 14.7) / 0.005193921   # +0.1%  width < 0.0187 and ptdr0_2 > 14.7
        - 0.00076487 * max(0.0, Q.mass - 46.7) * max(0.0, 23.5 - Q.pt_7) / 5.226758   # -0.1%  mass > 46.7 and pt_7 < 23.5
        - 0.0006431405 * max(0.0, 0.1 - Q.girth) * max(0.0, Q.dr1_3 - 0.155) / 0.00014859   # -0.1%  girth < 0.1 and dr1_3 > 0.155
        - 0.0005982011 * max(0.0, 0.228 - Q.N2) * max(0.0, 5.9 - Q.n_for_90pct) / 0.005047572   # -0.1%  N2 < 0.228 and n_for_90pct < 5.9
        - 0.0002290636 * max(0.0, 0.0267 - Q.tau1) * max(0.0, Q.zdr_3 - 0.00824) / 3.969184e-08   # -0.0%  tau1 < 0.0267 and zdr_3 > 0.00824
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.9266201680672269, 0.7960985294117647, 2.1208365546218486, 0.938627731092437, 1.2070096638655463, 1.8243716386554623, 0.9945617647058823, 1.8634323529411765, 0.288127731092437, 3.0506949579831932, 2.0064631302521008, 2.1524827731092437, 0.09292542016806722, 3.3146457983193276, 0.41297594537815124, 0.3934004201680672]
T = [2.379964469537815, 1.3909887145483193, 3.318946976759454, 2.4896771336659667, 2.8795969800420167]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +38%, n9 +22%, n5 -14%, n1 +13%, n0 -6%, n6 +5% ...
            + 0.3829036 * h[2] / H_AVG[2]
            + 0.2203135 * h[9] / H_AVG[9]
            - 0.1437289 * h[5] / H_AVG[5]
            + 0.1306641 * h[1] / H_AVG[1]
            - 0.06083469 * h[0] / H_AVG[0]
            + 0.04570665 * h[6] / H_AVG[6]
            - 0.01584858 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +56%, n10 -18%, n6 +9%, n4 -8%, n5 +6%, n15 +2% ...
            + 0.5568633 * h[9] / H_AVG[9]
            - 0.1803091 * h[10] / H_AVG[10]
            + 0.08937543 * h[6] / H_AVG[6]
            - 0.08135016 * h[4] / H_AVG[4]
            + 0.06147959 * h[5] / H_AVG[5]
            + 0.01767629 * h[15] / H_AVG[15]
            + 0.01294617 * h[8] / H_AVG[8]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n3 -14%, n7 +12%, n0 +10%, n6 -9%, n14 -9% ...
            + 0.243204 * h[11] / H_AVG[11]
            - 0.1414044 * h[3] / H_AVG[3]
            + 0.1228178 * h[7] / H_AVG[7]
            + 0.09597191 * h[0] / H_AVG[0]
            - 0.09364433 * h[6] / H_AVG[6]
            - 0.09332236 * h[14] / H_AVG[14]
            - 0.08149054 * h[15] / H_AVG[15]
            + 0.07022138 * h[13] / H_AVG[13]
            - 0.02872424 * h[9] / H_AVG[9]
            - 0.02170325 * h[8] / H_AVG[8]
            - 0.007495775 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +35%, n3 -21%, n6 -15%, n13 +7%, n14 +6%, n1 +4% ...
            + 0.3508422 * h[7] / H_AVG[7]
            - 0.2120669 * h[3] / H_AVG[3]
            - 0.1498028 * h[6] / H_AVG[6]
            + 0.07280851 * h[13] / H_AVG[13]
            + 0.06220324 * h[14] / H_AVG[14]
            + 0.03996997 * h[1] / H_AVG[1]
            - 0.0382918 * h[9] / H_AVG[9]
            + 0.03787545 * h[4] / H_AVG[4]
            - 0.02468947 * h[15] / H_AVG[15]
            + 0.0114496 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -16%, n4 +5%, n3 +2%, n8 +2% ...
            - 0.4676262 * h[13] / H_AVG[13]
            + 0.2612948 * h[10] / H_AVG[10]
            - 0.1583878 * h[5] / H_AVG[5]
            + 0.0523949 * h[4] / H_AVG[4]
            + 0.02037238 * h[3] / H_AVG[3]
            + 0.01876094 * h[8] / H_AVG[8]
            - 0.01613514 * h[12] / H_AVG[12]
            + 0.00502794 * h[0] / H_AVG[0]
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
