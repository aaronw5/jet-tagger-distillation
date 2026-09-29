"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables, tuned for agreement), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  6:  11.7%   (on for 72% of jets)
  neuron 13:  10.9%   (on for 88% of jets)
  neuron  7:  10.8%   (on for 69% of jets)
  neuron  5:   9.1%   (on for 69% of jets)
  neuron 10:   8.6%   (on for 89% of jets)
  neuron  3:   8.2%   (on for 32% of jets)
  neuron 11:   7.8%   (on for 74% of jets)
  neuron  9:   7.0%   (on for 62% of jets)
  neuron  2:   5.3%   (on for 59% of jets)
  neuron  0:   5.1%   (on for 78% of jets)
  neuron 14:   4.2%   (on for 38% of jets)
  neuron  1:   4.1%   (on for 78% of jets)
  neuron  4:   3.8%   (on for 96% of jets)
  neuron 15:   2.0%   (on for 50% of jets)
  neuron  8:   1.2%   (on for 40% of jets)
  neuron 12:   0.3%   (on for 8% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.3% (the network: 65.8%); same class as the network for 87.1% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
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
  Q.pair_mass_0_5          mass of particles 0 and 5 [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_2               |Δφ| of particle 2
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_3                  ΔR between particle 3 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.phi_0                  Δφ of particle 0
  Q.phi_2                  Δφ of particle 2
  Q.phi_4                  Δφ of particle 4
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
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
        pair_mass_0_5=pair_mass(0, 5),
        pair_mass_0_6=pair_mass(0, 6),
        sj3_mass1=subjets(3)["mass"][0],
        mass_top5=mass_of(5),
        sj2_mass2=subjets(2)["mass"][1],
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_5=z[5] * dr[5],
        zdr_7=z[7] * dr[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_7=abs(eta[7]),
        absphi_2=abs(phi[2]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_3=math.sqrt(dist2(1, 3)) if pt[3] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        eta_0=eta[0],
        phi_0=phi[0],
        phi_2=phi[2],
        phi_4=phi[4],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
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
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 17.96;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.9621 * (-0.2037624
        + 0.1994528 * max(0.0, 0.013 - Q.sum_z_dr2) / 0.008032715   # +19.9%  sum_z_dr2 < 0.013
        + 0.1459265 * max(0.0, Q.sj3_dr_max - 0.105) / 0.08650648   # +14.6%  sj3_dr_max > 0.105
        + 0.1453956 * max(0.0, 0.307 - Q.sj3_dr_max) / 0.1500925   # +14.5%  sj3_dr_max < 0.307
        - 0.06760624 * max(0.0, 0.0788 - Q.sum_z_dr) / 0.02991009   # -6.8%  sum_z_dr < 0.0788
        - 0.05480318 * max(0.0, 62.5 - Q.mass) / 25.90474   # -5.5%  mass < 62.5
        - 0.0547273 * max(0.0, 0.00477 - Q.lam1) / 0.001817037   # -5.5%  lam1 < 0.00477
        - 0.04063183 * max(0.0, Q.sj3_dr_max - 0.179) / 0.04318537   # -4.1%  sj3_dr_max > 0.179
        - 0.03516996 * max(0.0, Q.sj3_dr_max - 0.231) / 0.02578474   # -3.5%  sj3_dr_max > 0.231
        + 0.03366255 * max(0.0, 0.0228 - Q.e2) / 0.006608197   # +3.4%  e2 < 0.0228
        + 0.03340533 * max(0.0, Q.log_sum_pt - 6.36) / 0.2230594   # +3.3%  log_sum_pt > 6.36
        + 0.02913095 * max(0.0, 0.139 - Q.planar_flow) * max(0.0, 0.0561 - Q.centroid_offset) / 0.001855507   # +2.9%  planar_flow < 0.139 and centroid_offset < 0.0561
        - 0.01778214 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, Q.z_7 - 0.0163) / 0.0006491963   # -1.8%  log_sum_pt > 6.67 and z_7 > 0.0163
        + 0.01476222 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.0674 - Q.dr_4) / 0.001767736   # +1.5%  log_sum_pt > 6.67 and dr_4 < 0.0674
        + 0.01440988 * max(0.0, Q.sum_pt_top5 - 680.0) / 39.39599   # +1.4%  sum_pt_top5 > 680
        - 0.01195962 * max(0.0, 0.00027 - Q.lam1) / 3.470433e-05   # -1.2%  lam1 < 0.00027
        - 0.01104695 * max(0.0, Q.sum_pt - 900.0) / 12.63862   # -1.1%  sum_pt > 900
        + 0.01101162 * max(0.0, 0.0122 - Q.sum_z_dr2) * max(0.0, Q.phi_0 - -0.0104) / 0.0001198738   # +1.1%  sum_z_dr2 < 0.0122 and phi_0 > -0.0104
        - 0.008826709 * max(0.0, Q.centroid_offset - 0.0212) * max(0.0, 0.0684 - Q.z_7) / 4.391861e-05   # -0.9%  centroid_offset > 0.0212 and z_7 < 0.0684
        - 0.008006832 * max(0.0, 0.0151 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0192) / 2.393003e-05   # -0.8%  sum_z_dr2 < 0.0151 and centroid_offset > 0.0192
        + 0.007311197 * max(0.0, 0.163 - Q.planar_flow) * max(0.0, 417.0 - Q.sum_pt_top2) / 4.881948   # +0.7%  planar_flow < 0.163 and sum_pt_top2 < 417
        - 0.006401289 * max(0.0, Q.m012 - 38.4) / 1.533074   # -0.6%  m012 > 38.4
        + 0.005806593 * max(0.0, 0.145 - Q.planar_flow) * max(0.0, 0.0231 - Q.z_7) / 1.938635e-05   # +0.6%  planar_flow < 0.145 and z_7 < 0.0231
        + 0.00566832 * max(0.0, 6.27e-05 - Q.lam2) * max(0.0, 0.231 - Q.D2_b2) / 1.515103e-06   # +0.6%  lam2 < 6.27e-05 and D2_b2 < 0.231
        - 0.004839786 * max(0.0, 0.132 - Q.planar_flow) * max(0.0, Q.dr0_6 - 0.175) / 0.0007184521   # -0.5%  planar_flow < 0.132 and dr0_6 > 0.175
        + 0.004393474 * max(0.0, 0.0879 - Q.sum_z_dr) * max(0.0, Q.dr1_3 - 0.165) / 6.983717e-05   # +0.4%  sum_z_dr < 0.0879 and dr1_3 > 0.165
        - 0.004320726 * max(0.0, 0.00437 - Q.lam1_plus_lam2) * max(0.0, Q.eccentricity - 0.962) / 5.879492e-06   # -0.4%  lam1_plus_lam2 < 0.00437 and eccentricity > 0.962
        + 0.004133561 * max(0.0, 0.0201 - Q.sum_z_dr2) * max(0.0, -0.0955 - Q.phi_2) / 9.280928e-06   # +0.4%  sum_z_dr2 < 0.0201 and phi_2 < -0.0955
        - 0.003026659 * max(0.0, 6.08 - Q.log_sum_pt) / 0.006074317   # -0.3%  log_sum_pt < 6.08
        - 0.002900106 * max(0.0, 0.00483 - Q.lam1_plus_lam2) * max(0.0, 0.764 - Q.D2) / 1.160178e-05   # -0.3%  lam1_plus_lam2 < 0.00483 and D2 < 0.764
        - 0.002627132 * max(0.0, Q.centroid_offset - 0.0209) * max(0.0, 0.048 - Q.dr_2) / 1.08981e-05   # -0.3%  centroid_offset > 0.0209 and dr_2 < 0.048
        - 0.002087439 * max(0.0, Q.sum_pt - 899.0) * max(0.0, Q.eccentricity - 0.95) / 0.1760318   # -0.2%  sum_pt > 899 and eccentricity > 0.95
        + 0.001867113 * max(0.0, 0.00443 - Q.lam1_plus_lam2) * max(0.0, Q.pair_mass_0_6 - 20.4) / 9.290099e-05   # +0.2%  lam1_plus_lam2 < 0.00443 and pair_mass_0_6 > 20.4
        - 0.001577374 * max(0.0, 0.146 - Q.planar_flow) * max(0.0, 0.0219 - Q.dr_2) / 2.162821e-05   # -0.2%  planar_flow < 0.146 and dr_2 < 0.0219
        + 0.00157609 * max(0.0, Q.log_sum_pt - 6.66) * max(0.0, Q.phi_4 - 0.11) / 7.651319e-05   # +0.2%  log_sum_pt > 6.66 and phi_4 > 0.11
        + 0.001378331 * max(0.0, Q.sum_pt - 906.0) * max(0.0, Q.dr0_6 - 0.239) / 0.03433801   # +0.1%  sum_pt > 906 and dr0_6 > 0.239
        + 0.001349537 * max(0.0, Q.n_dr_0_0p05 - 4.05) * max(0.0, Q.dr_4 - 0.194) / 0.001049373   # +0.1%  n_dr_0_0p05 > 4.05 and dr_4 > 0.194
        - 0.001016974 * max(0.0, 30.5 - Q.mass) * max(0.0, 0.186 - Q.D2_b2) / 0.004359661   # -0.1%  mass < 30.5 and D2_b2 < 0.186
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 11.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.78366 * (0.1722724
        + 0.1071798 * max(0.0, 0.00385 - Q.sum_zz_dr2) / 0.001458395   # +10.7%  sum_zz_dr2 < 0.00385
        - 0.1060394 * max(0.0, 0.00992 - Q.sum_z_dr2) / 0.005456474   # -10.6%  sum_z_dr2 < 0.00992
        + 0.1055013 * max(0.0, 50.6 - Q.mass) / 17.63393   # +10.6%  mass < 50.6
        - 0.09101727 * max(0.0, 0.0623 - Q.mass_over_sum_pt) / 0.01922073   # -9.1%  mass_over_sum_pt < 0.0623
        + 0.08068639 * max(0.0, 0.0774 - Q.sum_z_dr) / 0.02889912   # +8.1%  sum_z_dr < 0.0774
        - 0.07829024 * max(0.0, 0.0625 - Q.z_7) / 0.01478438   # -7.8%  z_7 < 0.0625
        + 0.07414931 * max(0.0, Q.log_sum_pt - 6.6) / 0.0734244   # +7.4%  log_sum_pt > 6.6
        - 0.07380228 * max(0.0, Q.log_sum_pt - 6.42) * max(0.0, 1.67e-05 - Q.e3) / 1.728948e-06   # -7.4%  log_sum_pt > 6.42 and e3 < 1.67e-05
        - 0.04849323 * max(0.0, Q.log_sum_pt - 6.35) * max(0.0, 0.765 - Q.z_dr_0p05_0p1) / 0.1335112   # -4.8%  log_sum_pt > 6.35 and z_dr_0p05_0p1 < 0.765
        - 0.04079972 * max(0.0, 714.0 - Q.sum_pt) / 69.17555   # -4.1%  sum_pt < 714
        - 0.03338305 * max(0.0, 0.0178 - Q.centroid_offset) / 0.00634475   # -3.3%  centroid_offset < 0.0178
        + 0.02415626 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, Q.centroid_offset - 0.00857) / 0.000873157   # +2.4%  log_sum_pt > 6.4 and centroid_offset > 0.00857
        - 0.01709479 * max(0.0, 0.06 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.0548) / 0.002068165   # -1.7%  z_7 < 0.06 and z_dr_0p05_0p1 > 0.0548
        + 0.01574286 * max(0.0, 0.00947 - Q.sum_z_dr2) * max(0.0, Q.centroid_offset - 0.0199) / 9.61184e-06   # +1.6%  sum_z_dr2 < 0.00947 and centroid_offset > 0.0199
        + 0.01435281 * max(0.0, Q.pt_7 - 33.6) * max(0.0, Q.sj2_dr - 0.128) / 0.216277   # +1.4%  pt_7 > 33.6 and sj2_dr > 0.128
        + 0.01414854 * max(0.0, Q.log_sum_pt - 6.56) * max(0.0, 1.42 - Q.D2) / 0.02702132   # +1.4%  log_sum_pt > 6.56 and D2 < 1.42
        + 0.01329008 * max(0.0, Q.pt_7 - 33.3) / 5.117838   # +1.3%  pt_7 > 33.3
        - 0.01317481 * max(0.0, Q.pt_7 - 33.8) * max(0.0, 53.1 - Q.pt_6) / 16.44571   # -1.3%  pt_7 > 33.8 and pt_6 < 53.1
        + 0.01147173 * max(0.0, 578.0 - Q.sum_pt) / 20.08604   # +1.1%  sum_pt < 578
        + 0.008462448 * max(0.0, Q.z_7 - 0.0447) * max(0.0, 0.00244 - Q.sum_z_dr2_top2) / 8.379716e-06   # +0.8%  z_7 > 0.0447 and sum_z_dr2_top2 < 0.00244
        - 0.007808507 * max(0.0, Q.sj3_dr_max - 0.169) * max(0.0, Q.sj3_pair_mass_min - 4.47) / 0.6134187   # -0.8%  sj3_dr_max > 0.169 and sj3_pair_mass_min > 4.47
        - 0.007491853 * max(0.0, Q.pt_7 - 33.5) * max(0.0, Q.z_dr_0p05_0p1 - 0.316) / 0.7882273   # -0.7%  pt_7 > 33.5 and z_dr_0p05_0p1 > 0.316
        + 0.007325095 * max(0.0, 1.53 - Q.n_dr_0_0p05) * max(0.0, Q.n_pt_above_50 - 5.38) / 0.247325   # +0.7%  n_dr_0_0p05 < 1.53 and n_pt_above_50 > 5.38
        + 0.004928604 * max(0.0, 0.00494 - Q.tau21_b2) / 0.000177064   # +0.5%  tau21_b2 < 0.00494
        + 0.001209586 * max(0.0, Q.sum_pt_top5 - 604.0) * max(0.0, -0.0456 - Q.phi_0) / 0.05701342   # +0.1%  sum_pt_top5 > 604 and phi_0 < -0.0456
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 16.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.44812 * (0.1623285
        - 0.1187061 * max(0.0, Q.LHA - 0.132) / 0.1183329   # -11.9%  LHA > 0.132
        + 0.08564116 * max(0.0, 62.4 - Q.sj3_pair_mass_max) / 30.42411   # +8.6%  sj3_pair_mass_max < 62.4
        - 0.08261809 * max(0.0, 54.1 - Q.pt_7) / 19.75164   # -8.3%  pt_7 < 54.1
        + 0.06659459 * max(0.0, 0.00599 - Q.lam1) / 0.002541429   # +6.7%  lam1 < 0.00599
        + 0.06465862 * max(0.0, 6.57 - Q.log_sum_pt) / 0.1148502   # +6.5%  log_sum_pt < 6.57
        + 0.05659141 * max(0.0, 820.0 - Q.sum_pt) / 133.7389   # +5.7%  sum_pt < 820
        - 0.04055055 * max(0.0, Q.sum_pt - 544.0) * max(0.0, 0.000197 - Q.lam2) / 0.02722369   # -4.1%  sum_pt > 544 and lam2 < 0.000197
        - 0.04020997 * max(0.0, 54.3 - Q.sj3_pair_mass_max) * max(0.0, 0.0697 - Q.z_7) / 0.5087528   # -4.0%  sj3_pair_mass_max < 54.3 and z_7 < 0.0697
        - 0.03363188 * max(0.0, 0.507 - Q.planar_flow) / 0.2836827   # -3.4%  planar_flow < 0.507
        - 0.03241486 * max(0.0, 52.7 - Q.pt_7) * max(0.0, 1.2 - Q.D2_b2) / 9.836969   # -3.2%  pt_7 < 52.7 and D2_b2 < 1.2
        + 0.02743151 * max(0.0, Q.pt_7 - 32.4) / 5.625895   # +2.7%  pt_7 > 32.4
        - 0.02247785 * max(0.0, Q.sum_pt - 540.0) * max(0.0, 0.0228 - Q.absphi_2) / 1.768988   # -2.2%  sum_pt > 540 and absphi_2 < 0.0228
        + 0.02082268 * max(0.0, 37.8 - Q.mass) * max(0.0, 72.6 - Q.pt_5) / 267.5734   # +2.1%  mass < 37.8 and pt_5 < 72.6
        - 0.02058282 * max(0.0, Q.sum_pt_top5 - 761.0) / 19.56929   # -2.1%  sum_pt_top5 > 761
        + 0.01935302 * max(0.0, Q.log_sum_pt - 6.83) / 0.009226693   # +1.9%  log_sum_pt > 6.83
        + 0.01924709 * max(0.0, Q.sum_pt - 869.0) / 17.98742   # +1.9%  sum_pt > 869
        - 0.01874859 * max(0.0, 0.00766 - Q.sum_z_dr) / 0.0002301337   # -1.9%  sum_z_dr < 0.00766
        + 0.01836213 * max(0.0, 0.00669 - Q.lam1) * max(0.0, Q.pt_6 - 25.5) / 0.0468252   # +1.8%  lam1 < 0.00669 and pt_6 > 25.5
        - 0.01785042 * max(0.0, Q.z_7 - 0.0484) * max(0.0, 0.973 - Q.D2_b2) / 0.005097325   # -1.8%  z_7 > 0.0484 and D2_b2 < 0.973
        + 0.01474513 * max(0.0, 4.17 - Q.m012) / 1.183072   # +1.5%  m012 < 4.17
        - 0.01404727 * max(0.0, 61.4 - Q.sj3_pair_mass_max) * max(0.0, Q.centroid_offset - 0.00874) / 0.2391835   # -1.4%  sj3_pair_mass_max < 61.4 and centroid_offset > 0.00874
        - 0.01360111 * max(0.0, 0.00635 - Q.lam1) * max(0.0, Q.max_dr - 0.0787) / 5.496628e-05   # -1.4%  lam1 < 0.00635 and max_dr > 0.0787
        + 0.0132631 * max(0.0, 6.46 - Q.log_sum_pt) * max(0.0, 1.12 - Q.D2_b2) / 0.03529984   # +1.3%  log_sum_pt < 6.46 and D2_b2 < 1.12
        - 0.01217365 * max(0.0, 0.03 - Q.z_7) / 0.001614788   # -1.2%  z_7 < 0.03
        - 0.01059829 * max(0.0, 0.0311 - Q.z_7) * max(0.0, 0.806 - Q.D2_b2) / 0.0003047588   # -1.1%  z_7 < 0.0311 and D2_b2 < 0.806
        + 0.01031833 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, 0.47 - Q.z_dr_0p05_0p1) / 0.003030663   # +1.0%  log_sum_pt > 6.85 and z_dr_0p05_0p1 < 0.47
        + 0.009964434 * max(0.0, Q.pt_7 - 42.7) * max(0.0, 0.00511 - Q.C2_b2) / 0.007157041   # +1.0%  pt_7 > 42.7 and C2_b2 < 0.00511
        + 0.009687989 * max(0.0, Q.sum_pt_top5 - 755.0) * max(0.0, 1.26 - Q.D2_b2) / 7.050851   # +1.0%  sum_pt_top5 > 755 and D2_b2 < 1.26
        - 0.009340778 * max(0.0, 0.00441 - Q.centroid_offset) / 0.0004141193   # -0.9%  centroid_offset < 0.00441
        - 0.00893953 * max(0.0, 59.5 - Q.pt_7) * max(0.0, Q.abseta_0 - 0.0188) / 0.4609357   # -0.9%  pt_7 < 59.5 and abseta_0 > 0.0188
        - 0.007974223 * max(0.0, 0.00446 - Q.centroid_offset) * max(0.0, 0.0255 - Q.abseta_7) / 4.930865e-06   # -0.8%  centroid_offset < 0.00446 and abseta_7 < 0.0255
        + 0.007282858 * max(0.0, Q.log_sum_pt - 6.84) * max(0.0, 0.00018 - Q.lam2) / 1.24134e-06   # +0.7%  log_sum_pt > 6.84 and lam2 < 0.00018
        + 0.00722382 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.0158 - Q.absphi_2) / 0.0004228409   # +0.7%  log_sum_pt < 6.7 and absphi_2 < 0.0158
        + 0.007078303 * max(0.0, 6.61 - Q.log_sum_pt) * max(0.0, 0.0375 - Q.max_dr) / 0.000114142   # +0.7%  log_sum_pt < 6.61 and max_dr < 0.0375
        + 0.006797425 * max(0.0, Q.m012 - 40.3) / 1.295537   # +0.7%  m012 > 40.3
        - 0.005693092 * max(0.0, 0.262 - Q.max_dr) * max(0.0, Q.phi_0 - 0.00486) / 0.001434007   # -0.6%  max_dr < 0.262 and phi_0 > 0.00486
        + 0.00513587 * max(0.0, Q.sum_pt_top5 - 849.0) * max(0.0, 0.0498 - Q.dr_7) / 0.1446497   # +0.5%  sum_pt_top5 > 849 and dr_7 < 0.0498
        - 0.004850114 * max(0.0, Q.log_sum_pt - 6.83) * max(0.0, 1.34 - Q.D2_b2) / 0.003423832   # -0.5%  log_sum_pt > 6.83 and D2_b2 < 1.34
        + 0.004461516 * max(0.0, Q.log_sum_pt - 6.86) * max(0.0, 0.0183 - Q.abseta_0) / 7.124618e-05   # +0.4%  log_sum_pt > 6.86 and abseta_0 < 0.0183
        - 0.003281701 * max(0.0, 0.00837 - Q.sum_z_dr) * max(0.0, 74.6 - Q.pt_4) / 0.005681876   # -0.3%  sum_z_dr < 0.00837 and pt_4 < 74.6
        + 0.002490454 * max(0.0, Q.log_sum_pt - 6.85) * max(0.0, Q.pt_6 - 39.5) / 0.062349   # +0.2%  log_sum_pt > 6.85 and pt_6 > 39.5
        - 0.002187304 * max(0.0, 7.96 - Q.n_pt_above_5) / 0.005692571   # -0.2%  n_pt_above_5 < 7.96
        + 0.001684182 * max(0.0, Q.sum_pt_top5 - 774.0) * max(0.0, Q.eta_0 - 0.00474) / 0.0625319   # +0.2%  sum_pt_top5 > 774 and eta_0 > 0.00474
        - 0.0006862147 * max(0.0, Q.sum_pt_top5 - 853.0) * max(0.0, 0.00117 - Q.absphi_2) / 0.0006449682   # -0.1%  sum_pt_top5 > 853 and absphi_2 < 0.00117
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 13.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.13323 * (-0.2543168
        + 0.2115443 * max(0.0, Q.lam1_plus_lam2 - 0.00549) / 0.003125151   # +21.2%  lam1_plus_lam2 > 0.00549
        - 0.1493148 * max(0.0, Q.lam1_plus_lam2 - 0.0138) / 0.001361795   # -14.9%  lam1_plus_lam2 > 0.0138
        + 0.1044928 * max(0.0, Q.sj2_dr - 0.185) / 0.02812146   # +10.4%  sj2_dr > 0.185
        - 0.07044141 * max(0.0, Q.sj2_dr - 0.187) * max(0.0, 0.992 - Q.z_dr_0p05_0p1) / 0.02042214   # -7.0%  sj2_dr > 0.187 and z_dr_0p05_0p1 < 0.992
        - 0.06615883 * max(0.0, Q.sum_z_dr2 - 0.00895) / 0.00216139   # -6.6%  sum_z_dr2 > 0.00895
        + 0.06312327 * max(0.0, Q.sum_z_dr - 0.0874) / 0.007820869   # +6.3%  sum_z_dr > 0.0874
        + 0.05749181 * max(0.0, Q.sj3_dr_max - 0.186) / 0.04016239   # +5.7%  sj3_dr_max > 0.186
        + 0.03190608 * max(0.0, Q.sum_z_dr - 0.0433) * max(0.0, Q.log_sum_pt - 6.11) / 0.007536505   # +3.2%  sum_z_dr > 0.0433 and log_sum_pt > 6.11
        + 0.03056447 * max(0.0, Q.max_dr - 0.101) / 0.04603326   # +3.1%  max_dr > 0.101
        + 0.02793987 * max(0.0, Q.centroid_offset - 0.0111) / 0.008736681   # +2.8%  centroid_offset > 0.0111
        + 0.02716793 * max(0.0, Q.centroid_offset - 0.00992) * max(0.0, 0.0872 - Q.abseta_0) / 0.0003844856   # +2.7%  centroid_offset > 0.00992 and abseta_0 < 0.0872
        - 0.02332688 * max(0.0, 0.0537 - Q.tau1) / 0.01701984   # -2.3%  tau1 < 0.0537
        + 0.01626404 * max(0.0, 1.0 - Q.n_dr_0_0p05) / 0.2859429   # +1.6%  n_dr_0_0p05 < 1
        - 0.01574897 * max(0.0, Q.mass_over_sum_pt - 0.0454) * max(0.0, Q.sj3_pair_mass_min - 8.19) / 0.226049   # -1.6%  mass_over_sum_pt > 0.0454 and sj3_pair_mass_min > 8.19
        - 0.0156733 * max(0.0, Q.mass_over_sum_pt - 0.0668) * max(0.0, 0.22 - Q.sj2_dr) / 0.0001419593   # -1.6%  mass_over_sum_pt > 0.0668 and sj2_dr < 0.22
        + 0.01391048 * max(0.0, Q.sd_mass - 64.0) * max(0.0, 1.93 - Q.D2_b2) / 4.444998   # +1.4%  sd_mass > 64 and D2_b2 < 1.93
        + 0.01297334 * max(0.0, Q.centroid_offset - 0.0121) * max(0.0, 7.57 - Q.n_pt_above_50) / 0.02469302   # +1.3%  centroid_offset > 0.0121 and n_pt_above_50 < 7.57
        + 0.01250731 * max(0.0, -0.00371 - Q.mean_eta) / 0.004055835   # +1.3%  mean_eta < -0.00371
        + 0.0117604 * max(0.0, Q.sj2_dr - 0.275) * max(0.0, 0.958 - Q.z_dr_0p05_0p1) / 0.005289452   # +1.2%  sj2_dr > 0.275 and z_dr_0p05_0p1 < 0.958
        + 0.008899283 * max(0.0, Q.centroid_offset - 0.0109) * max(0.0, Q.pair_mass_0_5 - 16.4) / 0.0182905   # +0.9%  centroid_offset > 0.0109 and pair_mass_0_5 > 16.4
        - 0.00873323 * max(0.0, Q.sj2_dr - 0.179) * max(0.0, 0.05 - Q.dr_3) / 0.0001533362   # -0.9%  sj2_dr > 0.179 and dr_3 < 0.05
        + 0.007988684 * max(0.0, Q.mean_eta - 0.0176) / 0.001387794   # +0.8%  mean_eta > 0.0176
        - 0.00531913 * max(0.0, Q.abseta_7 - 0.0774) / 0.01221282   # -0.5%  abseta_7 > 0.0774
        + 0.004527651 * max(0.0, Q.sum_z_dr - 0.0868) * max(0.0, Q.z_dr_0p05_0p1 - 0.683) / 5.717564e-05   # +0.5%  sum_z_dr > 0.0868 and z_dr_0p05_0p1 > 0.683
        - 0.001112066 * max(0.0, Q.sj2_dr - 0.215) * max(0.0, 25.3 - Q.pt_6) / 0.01390954   # -0.1%  sj2_dr > 0.215 and pt_6 < 25.3
        + 0.001109693 * max(0.0, Q.sd_mass - 113.0) / 0.1071606   # +0.1%  sd_mass > 113
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 17.63;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.62998 * (0.2217813
        - 0.2619481 * max(0.0, 0.00896 - Q.sum_z_dr2) / 0.00467423   # -26.2%  sum_z_dr2 < 0.00896
        + 0.2181362 * max(0.0, 0.000149 - Q.e3) / 0.0001154876   # +21.8%  e3 < 0.000149
        + 0.1870146 * max(0.0, 0.374 - Q.sj3_dr_max) / 0.210004   # +18.7%  sj3_dr_max < 0.374
        - 0.1370604 * max(0.0, 0.179 - Q.max_dr) / 0.07170243   # -13.7%  max_dr < 0.179
        - 0.1283533 * max(0.0, 0.000586 - Q.lam2) / 0.0004326703   # -12.8%  lam2 < 0.000586
        + 0.0537029 * max(0.0, 0.11 - Q.max_dr) / 0.0275227   # +5.4%  max_dr < 0.11
        + 0.007922442 * max(0.0, 0.347 - Q.sj3_dr_max) * max(0.0, Q.pair_mass_0_6 - 17.4) / 0.05060597   # +0.8%  sj3_dr_max < 0.347 and pair_mass_0_6 > 17.4
        - 0.005861977 * max(0.0, Q.mass - 77.8) / 1.437365   # -0.6%  mass > 77.8
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.03007 * (-0.08663948
        + 0.1655742 * max(0.0, 0.0744 - Q.z_7) / 0.02379256   # +16.6%  z_7 < 0.0744
        + 0.1212434 * max(0.0, 0.00154 - Q.sum_z_dr2) * max(0.0, 0.0251 - Q.centroid_offset) / 7.029363e-06   # +12.1%  sum_z_dr2 < 0.00154 and centroid_offset < 0.0251
        + 0.1205649 * max(0.0, 0.00337 - Q.lam1_plus_lam2) / 0.00109934   # +12.1%  lam1_plus_lam2 < 0.00337
        + 0.08488663 * max(0.0, 0.0486 - Q.z_7) * max(0.0, 71.3 - Q.mass_top5) / 0.3503781   # +8.5%  z_7 < 0.0486 and mass_top5 < 71.3
        - 0.08349361 * max(0.0, 0.0214 - Q.zdr_0) / 0.009202711   # -8.3%  zdr_0 < 0.0214
        + 0.07025421 * max(0.0, Q.sum_pt_top5 - 489.0) * max(0.0, 49.6 - Q.pt_6) / 2135.317   # +7.0%  sum_pt_top5 > 489 and pt_6 < 49.6
        + 0.04369248 * max(0.0, 0.00708 - Q.sum_zz_dr2) * max(0.0, 0.011 - Q.centroid_offset) / 1.22072e-05   # +4.4%  sum_zz_dr2 < 0.00708 and centroid_offset < 0.011
        - 0.0414373 * max(0.0, 0.228 - Q.LHA) * max(0.0, 6.78 - Q.log_sum_pt) / 0.004384167   # -4.1%  LHA < 0.228 and log_sum_pt < 6.78
        + 0.03671372 * max(0.0, 0.0688 - Q.z_7) * max(0.0, 0.00212 - Q.mean_phi2) / 2.287212e-05   # +3.7%  z_7 < 0.0688 and mean_phi2 < 0.00212
        + 0.02714274 * max(0.0, Q.log_sum_pt - 6.69) * max(0.0, 0.0131 - Q.dr_2) / 0.0001388998   # +2.7%  log_sum_pt > 6.69 and dr_2 < 0.0131
        + 0.02571984 * max(0.0, 0.026 - Q.z_7) / 0.001027776   # +2.6%  z_7 < 0.026
        + 0.02524879 * max(0.0, 0.00546 - Q.sum_zz_dr2) * max(0.0, 0.329 - Q.planar_flow) / 0.0002110393   # +2.5%  sum_zz_dr2 < 0.00546 and planar_flow < 0.329
        - 0.02503928 * max(0.0, 0.049 - Q.z_7) * max(0.0, 510.0 - Q.sum_pt_top2) / 0.2414862   # -2.5%  z_7 < 0.049 and sum_pt_top2 < 510
        - 0.02430145 * max(0.0, Q.log_sum_pt - 6.87) / 0.005577693   # -2.4%  log_sum_pt > 6.87
        + 0.02099162 * max(0.0, 0.00685 - Q.centroid_offset) * max(0.0, 59.7 - Q.pt_5) / 0.01525705   # +2.1%  centroid_offset < 0.00685 and pt_5 < 59.7
        + 0.01717277 * max(0.0, 0.00768 - Q.sum_z_dr) / 0.0002321348   # +1.7%  sum_z_dr < 0.00768
        + 0.01360041 * max(0.0, 0.249 - Q.LHA) * max(0.0, 0.065 - Q.z_3) / 0.0001465232   # +1.4%  LHA < 0.249 and z_3 < 0.065
        + 0.009936411 * max(0.0, 23.5 - Q.pt_5) / 0.235054   # +1.0%  pt_5 < 23.5
        - 0.009885001 * max(0.0, Q.log_sum_pt - 6.71) * max(0.0, Q.mean_phi - 0.00234) / 4.183428e-05   # -1.0%  log_sum_pt > 6.71 and mean_phi > 0.00234
        - 0.008426238 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.023 - Q.dr_2) / 4.225788e-05   # -0.8%  log_sum_pt > 6.9 and dr_2 < 0.023
        - 0.007677402 * max(0.0, Q.log_sum_pt - 6.71) * max(0.0, -0.00253 - Q.mean_eta) / 4.326116e-05   # -0.8%  log_sum_pt > 6.71 and mean_eta < -0.00253
        - 0.007336175 * max(0.0, 0.00435 - Q.sum_zz_dr2) * max(0.0, Q.n_pt_above_50 - 6.24) / 0.0004747248   # -0.7%  sum_zz_dr2 < 0.00435 and n_pt_above_50 > 6.24
        + 0.005885276 * max(0.0, 0.00356 - Q.lam1_plus_lam2) * max(0.0, Q.sj3_dr13 - 0.182) / 2.34245e-06   # +0.6%  lam1_plus_lam2 < 0.00356 and sj3_dr13 > 0.182
        + 0.00220326 * max(0.0, 22.7 - Q.pt_5) * max(0.0, 8.02 - Q.n_pt_above_5) / 0.07892448   # +0.2%  pt_5 < 22.7 and n_pt_above_5 < 8.02
        + 0.001572916 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.0532 - Q.D2_b2) / 1.12689e-05   # +0.2%  log_sum_pt > 6.89 and D2_b2 < 0.0532
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 15.83;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.82997 * (-0.001775114
        - 0.2634395 * max(0.0, 0.00919 - Q.sum_z_dr2) / 0.00486042   # -26.3%  sum_z_dr2 < 0.00919
        + 0.2246575 * max(0.0, 0.00724 - Q.lam1) / 0.003419541   # +22.5%  lam1 < 0.00724
        + 0.09802067 * max(0.0, 0.128 - Q.tau1) * max(0.0, 0.0142 - Q.mean_phi2) / 0.0009074061   # +9.8%  tau1 < 0.128 and mean_phi2 < 0.0142
        + 0.04885853 * max(0.0, 0.18 - Q.sj2_dr) / 0.05995575   # +4.9%  sj2_dr < 0.18
        + 0.04097159 * max(0.0, 0.00292 - Q.sum_zz_dr2) / 0.001024612   # +4.1%  sum_zz_dr2 < 0.00292
        + 0.04067978 * max(0.0, 40.5 - Q.pt_6) * max(0.0, 6.79 - Q.log_sum_pt) / 0.9922339   # +4.1%  pt_6 < 40.5 and log_sum_pt < 6.79
        - 0.03492846 * max(0.0, Q.tau2 - 0.0098) / 0.007733098   # -3.5%  tau2 > 0.0098
        + 0.0297818 * max(0.0, 617.0 - Q.sum_pt) / 30.41581   # +3.0%  sum_pt < 617
        - 0.026324 * max(0.0, 39.9 - Q.pt_6) * max(0.0, Q.z_7 - 0.02) / 0.06656681   # -2.6%  pt_6 < 39.9 and z_7 > 0.02
        + 0.02439924 * max(0.0, Q.sj2_dr - 0.226) / 0.01643571   # +2.4%  sj2_dr > 0.226
        - 0.02317337 * max(0.0, Q.centroid_offset - 0.00869) * max(0.0, Q.psi_0p1 - 0.352) / 0.003683071   # -2.3%  centroid_offset > 0.00869 and psi_0p1 > 0.352
        - 0.02270002 * max(0.0, Q.max_dr - 0.19) / 0.01387415   # -2.3%  max_dr > 0.19
        + 0.02036316 * max(0.0, 6.63 - Q.log_sum_pt) * max(0.0, 0.0758 - Q.z_7) / 0.001407635   # +2.0%  log_sum_pt < 6.63 and z_7 < 0.0758
        + 0.017829 * max(0.0, 39.1 - Q.pt_6) / 4.396145   # +1.8%  pt_6 < 39.1
        - 0.01304259 * max(0.0, Q.centroid_offset - 0.0174) * max(0.0, 0.00444 - Q.mean_phi2) / 7.971576e-06   # -1.3%  centroid_offset > 0.0174 and mean_phi2 < 0.00444
        + 0.01257152 * max(0.0, 0.183 - Q.sj3_dr_max) * max(0.0, 0.129 - Q.tau21_b2) / 0.0007865884   # +1.3%  sj3_dr_max < 0.183 and tau21_b2 < 0.129
        - 0.0101945 * max(0.0, 0.147 - Q.max_dr) * max(0.0, Q.mean_phi - 0.00438) / 0.0001002352   # -1.0%  max_dr < 0.147 and mean_phi > 0.00438
        + 0.008636013 * max(0.0, 6.26 - Q.log_sum_pt) / 0.02183831   # +0.9%  log_sum_pt < 6.26
        - 0.00762586 * max(0.0, Q.centroid_offset - 0.00767) * max(0.0, 0.000229 - Q.mean_phi2) / 1.183501e-07   # -0.8%  centroid_offset > 0.00767 and mean_phi2 < 0.000229
        + 0.006505694 * max(0.0, 41.7 - Q.pt_6) * max(0.0, Q.M3 - 0.0775) / 0.01616718   # +0.7%  pt_6 < 41.7 and M3 > 0.0775
        - 0.005431078 * max(0.0, Q.sum_pt_top5 - 846.0) / 8.034936   # -0.5%  sum_pt_top5 > 846
        + 0.003354104 * max(0.0, 15.3 - Q.pt_6) / 0.1032984   # +0.3%  pt_6 < 15.3
        - 0.003183216 * max(0.0, 63.4 - Q.mass) * max(0.0, -0.0103 - Q.phi_0) / 0.1250378   # -0.3%  mass < 63.4 and phi_0 < -0.0103
        + 0.002830869 * max(0.0, 0.0122 - Q.lam1_plus_lam2) * max(0.0, -0.0267 - Q.mean_eta) / 1.413646e-06   # +0.3%  lam1_plus_lam2 < 0.0122 and mean_eta < -0.0267
        + 0.002775305 * max(0.0, Q.sum_pt - 989.0) * max(0.0, 0.00241 - Q.mean_phi2) / 0.008481274   # +0.3%  sum_pt > 989 and mean_phi2 < 0.00241
        - 0.00243712 * max(0.0, Q.centroid_offset - 0.0179) * max(0.0, Q.pt_4 - 63.9) / 0.009098949   # -0.2%  centroid_offset > 0.0179 and pt_4 > 63.9
        + 0.001700505 * max(0.0, 0.0127 - Q.lam1_plus_lam2) * max(0.0, Q.mean_phi - 0.0262) / 1.520845e-06   # +0.2%  lam1_plus_lam2 < 0.0127 and mean_phi > 0.0262
        + 0.001289578 * max(0.0, 0.0109 - Q.lam1_plus_lam2) * max(0.0, -0.0262 - Q.mean_phi) / 1.097526e-06   # +0.1%  lam1_plus_lam2 < 0.0109 and mean_phi < -0.0262
        + 0.001039214 * max(0.0, 633.0 - Q.sum_pt) * max(0.0, 24.5 - Q.pt_5) / 1.960755   # +0.1%  sum_pt < 633 and pt_5 < 24.5
        - 0.001032892 * max(0.0, Q.sum_pt - 993.0) * max(0.0, Q.n_pt_above_50 - 6.19) / 1.98913   # -0.1%  sum_pt > 993 and n_pt_above_50 > 6.19
        + 0.0002232725 * max(0.0, 615.0 - Q.sum_pt) * max(0.0, 0.051 - Q.z_3) / 0.0001231497   # +0.0%  sum_pt < 615 and z_3 < 0.051
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 33.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.36623 * (0.2017009
        - 0.1272277 * max(0.0, Q.sum_z_dr2 - 0.00673) / 0.002686778   # -12.7%  sum_z_dr2 > 0.00673
        - 0.1068584 * max(0.0, 0.0894 - Q.sum_z_dr) / 0.03813328   # -10.7%  sum_z_dr < 0.0894
        - 0.08083218 * max(0.0, Q.sum_z_dr2 - 0.00438) / 0.003634859   # -8.1%  sum_z_dr2 > 0.00438
        - 0.07502561 * max(0.0, Q.mass_over_sum_pt - 0.0906) / 0.008261789   # -7.5%  mass_over_sum_pt > 0.0906
        + 0.06933725 * max(0.0, Q.mass_over_sum_pt - 0.0795) / 0.01101678   # +6.9%  mass_over_sum_pt > 0.0795
        + 0.06327362 * max(0.0, 0.157 - Q.sj2_dr) / 0.04733638   # +6.3%  sj2_dr < 0.157
        + 0.06299103 * max(0.0, Q.lam1_plus_lam2 - 0.00863) / 0.002224099   # +6.3%  lam1_plus_lam2 > 0.00863
        - 0.05463026 * max(0.0, 0.186 - Q.sj2_dr) / 0.06373448   # -5.5%  sj2_dr < 0.186
        + 0.04275687 * max(0.0, Q.sum_z_dr2 - 0.0132) / 0.001448361   # +4.3%  sum_z_dr2 > 0.0132
        - 0.03298563 * max(0.0, 0.0204 - Q.centroid_offset) * max(0.0, 0.00469 - Q.C2_b2) / 3.1536e-05   # -3.3%  centroid_offset < 0.0204 and C2_b2 < 0.00469
        + 0.02859038 * max(0.0, Q.sum_z_dr2 - 0.00441) * max(0.0, 0.208 - Q.planar_flow) / 0.0002864725   # +2.9%  sum_z_dr2 > 0.00441 and planar_flow < 0.208
        - 0.02668413 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.lam1_plus_lam2 - 0.00761) / 0.0001601347   # -2.7%  planar_flow < 0.21 and lam1_plus_lam2 > 0.00761
        + 0.02592125 * max(0.0, Q.e2 - 0.0352) / 0.006919155   # +2.6%  e2 > 0.0352
        + 0.02534279 * max(0.0, Q.mass - 37.7) / 13.40085   # +2.5%  mass > 37.7
        + 0.024065 * max(0.0, Q.z_7 - 0.0274) / 0.02598571   # +2.4%  z_7 > 0.0274
        + 0.02301489 * max(0.0, Q.sj3_dr_max - 0.137) / 0.06620002   # +2.3%  sj3_dr_max > 0.137
        - 0.02210676 * max(0.0, 0.00106 - Q.sum_z_dr2) / 0.0002341649   # -2.2%  sum_z_dr2 < 0.00106
        + 0.02056669 * max(0.0, 0.000991 - Q.sum_zz_dr2) / 0.0002701705   # +2.1%  sum_zz_dr2 < 0.000991
        + 0.01468197 * max(0.0, 0.0508 - Q.tau1) / 0.01560134   # +1.5%  tau1 < 0.0508
        + 0.009291588 * max(0.0, 0.176 - Q.planar_flow) / 0.06540618   # +0.9%  planar_flow < 0.176
        - 0.008659124 * max(0.0, Q.sj3_dr_max - 0.254) / 0.02049095   # -0.9%  sj3_dr_max > 0.254
        + 0.008069859 * max(0.0, 0.0197 - Q.centroid_offset) * max(0.0, 0.0317 - Q.tau21_b2) / 5.668649e-05   # +0.8%  centroid_offset < 0.0197 and tau21_b2 < 0.0317
        - 0.007656214 * max(0.0, Q.centroid_offset - 0.0347) / 0.002027453   # -0.8%  centroid_offset > 0.0347
        + 0.007603772 * max(0.0, 0.0876 - Q.sum_z_dr) * max(0.0, 0.00261 - Q.mean_phi) / 0.0001865509   # +0.8%  sum_z_dr < 0.0876 and mean_phi < 0.00261
        - 0.004946744 * max(0.0, 29.3 - Q.pt_7) / 2.257924   # -0.5%  pt_7 < 29.3
        + 0.004793934 * max(0.0, Q.mass_top5 - 52.4) / 2.167419   # +0.5%  mass_top5 > 52.4
        - 0.004744326 * max(0.0, Q.mass - 77.2) / 1.493399   # -0.5%  mass > 77.2
        - 0.003058806 * max(0.0, Q.mass_over_sum_pt - 0.0793) * max(0.0, 34.7 - Q.pt_6) / 0.01762709   # -0.3%  mass_over_sum_pt > 0.0793 and pt_6 < 34.7
        + 0.002907196 * max(0.0, 0.00372 - Q.sum_z_dr2) * max(0.0, -0.00688 - Q.mean_eta) / 1.515659e-06   # +0.3%  sum_z_dr2 < 0.00372 and mean_eta < -0.00688
        + 0.002708003 * max(0.0, 0.00108 - Q.sum_z_dr2_top2) * max(0.0, Q.centroid_offset - 0.00962) / 5.410531e-07   # +0.3%  sum_z_dr2_top2 < 0.00108 and centroid_offset > 0.00962
        - 0.002323175 * max(0.0, Q.centroid_offset - 0.0308) * max(0.0, Q.n_pt_above_50 - 4.03) / 0.001899892   # -0.2%  centroid_offset > 0.0308 and n_pt_above_50 > 4.03
        - 0.002202309 * max(0.0, 0.169 - Q.planar_flow) * max(0.0, 35.5 - Q.pt_6) / 0.1405024   # -0.2%  planar_flow < 0.169 and pt_6 < 35.5
        + 0.001286451 * max(0.0, 0.0209 - Q.centroid_offset) * max(0.0, Q.zdr_5 - 0.0079) / 3.700348e-06   # +0.1%  centroid_offset < 0.0209 and zdr_5 > 0.0079
        - 0.001175974 * max(0.0, 0.000993 - Q.sum_z_dr2_top2) * max(0.0, 0.0274 - Q.tau21_b2) / 2.012196e-07   # -0.1%  sum_z_dr2_top2 < 0.000993 and tau21_b2 < 0.0274
        + 0.0009130123 * max(0.0, 0.167 - Q.planar_flow) * max(0.0, Q.pt_4 - 71.8) / 0.1721118   # +0.1%  planar_flow < 0.167 and pt_4 > 71.8
        - 0.000445587 * max(0.0, Q.centroid_offset - 0.0377) * max(0.0, Q.pt_4 - 81.3) / 0.0001964011   # -0.0%  centroid_offset > 0.0377 and pt_4 > 81.3
        - 0.0003215345 * max(0.0, 0.000995 - Q.sum_zz_dr2) * max(0.0, Q.phi_0 - 0.0412) / 2.103607e-08   # -0.0%  sum_zz_dr2 < 0.000995 and phi_0 > 0.0412
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 8.744;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.743771 * (-0.05546806
        - 0.2929914 * max(0.0, 0.0572 - Q.sum_z_dr) * max(0.0, 0.00478 - Q.sum_z_dr2) / 6.849866e-05   # -29.3%  sum_z_dr < 0.0572 and sum_z_dr2 < 0.00478
        + 0.2857673 * max(0.0, 0.00355 - Q.lam1_plus_lam2) / 0.001178624   # +28.6%  lam1_plus_lam2 < 0.00355
        + 0.1299723 * max(0.0, 0.00696 - Q.sum_z_dr2) * max(0.0, 0.0288 - Q.centroid_offset) / 5.598268e-05   # +13.0%  sum_z_dr2 < 0.00696 and centroid_offset < 0.0288
        + 0.1194589 * max(0.0, 48.6 - Q.mass) * max(0.0, 0.000168 - Q.lam2) / 0.001967084   # +11.9%  mass < 48.6 and lam2 < 0.000168
        - 0.0927137 * max(0.0, 21.5 - Q.mass) * max(0.0, 0.000241 - Q.lam2) / 0.0008987443   # -9.3%  mass < 21.5 and lam2 < 0.000241
        - 0.04345236 * max(0.0, 0.00393 - Q.lam1_plus_lam2) * max(0.0, Q.centroid_offset - 0.00848) / 5.662258e-06   # -4.3%  lam1_plus_lam2 < 0.00393 and centroid_offset > 0.00848
        - 0.01882909 * max(0.0, 0.00673 - Q.sum_z_dr2) * max(0.0, 6.57 - Q.log_sum_pt) / 0.0001959967   # -1.9%  sum_z_dr2 < 0.00673 and log_sum_pt < 6.57
        - 0.01681483 * max(0.0, Q.log_sum_pt - 6.73) * max(0.0, 0.0302 - Q.centroid_offset) / 0.000650553   # -1.7%  log_sum_pt > 6.73 and centroid_offset < 0.0302
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 15.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.73562 * (-0.2319578
        + 0.2905254 * max(0.0, 0.0059 - Q.sum_z_dr2) / 0.002418835   # +29.1%  sum_z_dr2 < 0.0059
        + 0.145554 * max(0.0, 53.3 - Q.mass) * max(0.0, 0.0258 - Q.centroid_offset) / 0.2796561   # +14.6%  mass < 53.3 and centroid_offset < 0.0258
        + 0.1176451 * max(0.0, 0.018 - Q.centroid_offset) / 0.006472794   # +11.8%  centroid_offset < 0.018
        - 0.09139928 * max(0.0, 27.8 - Q.mass) / 6.627762   # -9.1%  mass < 27.8
        + 0.07172638 * max(0.0, 0.226 - Q.sj3_dr_max) / 0.08486159   # +7.2%  sj3_dr_max < 0.226
        + 0.03866543 * max(0.0, 0.00814 - Q.sum_z_dr2) * max(0.0, 0.349 - Q.planar_flow) / 0.000490665   # +3.9%  sum_z_dr2 < 0.00814 and planar_flow < 0.349
        - 0.0266383 * max(0.0, Q.lam1 - 0.00723) * max(0.0, Q.eccentricity - 0.625) / 0.0005656818   # -2.7%  lam1 > 0.00723 and eccentricity > 0.625
        - 0.02593639 * max(0.0, Q.m012 - 26.5) / 3.924281   # -2.6%  m012 > 26.5
        + 0.0252224 * max(0.0, 0.0193 - Q.dr_2) / 0.002699933   # +2.5%  dr_2 < 0.0193
        + 0.02279215 * max(0.0, 0.0281 - Q.centroid_offset) * max(0.0, 186.0 - Q.pt_1) / 0.7130191   # +2.3%  centroid_offset < 0.0281 and pt_1 < 186
        - 0.01991679 * max(0.0, 0.00579 - Q.sum_z_dr2) * max(0.0, 0.000234 - Q.mean_phi) / 8.470355e-06   # -2.0%  sum_z_dr2 < 0.00579 and mean_phi < 0.000234
        - 0.01776193 * Q.z_dr_0p2_0p4 / 0.02920532   # -1.8%  z_dr_0p2_0p4
        + 0.01638879 * max(0.0, 0.00017 - Q.lam1_plus_lam2) / 1.409223e-05   # +1.6%  lam1_plus_lam2 < 0.00017
        - 0.01474025 * max(0.0, 0.0433 - Q.tau1) * max(0.0, Q.eccentricity - 0.897) / 0.0003020143   # -1.5%  tau1 < 0.0433 and eccentricity > 0.897
        + 0.01460795 * max(0.0, 0.0192 - Q.centroid_offset) * max(0.0, Q.mean_eta - -0.00207) / 2.275892e-05   # +1.5%  centroid_offset < 0.0192 and mean_eta > -0.00207
        + 0.01407534 * max(0.0, 38.9 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt) / 2.384115   # +1.4%  mass < 38.9 and log_sum_pt < 6.8
        + 0.00922913 * max(0.0, 56.4 - Q.mass) * max(0.0, Q.D2 - 2.26) / 8.542711   # +0.9%  mass < 56.4 and D2 > 2.26
        - 0.007882496 * max(0.0, 0.0512 - Q.tau1) * max(0.0, 0.0288 - Q.tau21_b2) / 1.664912e-05   # -0.8%  tau1 < 0.0512 and tau21_b2 < 0.0288
        - 0.007290992 * max(0.0, 26.4 - Q.sj3_pair_mass_max) * max(0.0, Q.absphi_2 - 0.00759) / 0.0379895   # -0.7%  sj3_pair_mass_max < 26.4 and absphi_2 > 0.00759
        + 0.005914619 * max(0.0, 34.4 - Q.pt_4) / 0.5199453   # +0.6%  pt_4 < 34.4
        - 0.00540238 * max(0.0, Q.max_dr - 0.278) / 0.002415051   # -0.5%  max_dr > 0.278
        + 0.003539217 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.00145 - Q.zdr_5) / 2.994182e-06   # +0.4%  log_sum_pt > 6.89 and zdr_5 < 0.00145
        - 0.002687702 * max(0.0, 0.00493 - Q.sum_z_dr2) * max(0.0, Q.mean_phi - 0.0256) / 2.063056e-07   # -0.3%  sum_z_dr2 < 0.00493 and mean_phi > 0.0256
        - 0.002672354 * max(0.0, Q.log_sum_pt - 6.89) * max(0.0, 0.731 - Q.D2_b2) / 0.0006939134   # -0.3%  log_sum_pt > 6.89 and D2_b2 < 0.731
        + 0.001785116 * max(0.0, 30.7 - Q.pt_4) * max(0.0, Q.M3 - 0.0506) / 0.0007296082   # +0.2%  pt_4 < 30.7 and M3 > 0.0506
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 8.881;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.880524 * (0.4222724
        - 0.144176 * max(0.0, 0.00552 - Q.lam1) / 0.002246243   # -14.4%  lam1 < 0.00552
        + 0.1077387 * max(0.0, Q.lam2 - 0.000307) / 0.0004214872   # +10.8%  lam2 > 0.000307
        - 0.1016119 * max(0.0, 0.0667 - Q.z_7) / 0.01772823   # -10.2%  z_7 < 0.0667
        - 0.08148813 * max(0.0, 0.00289 - Q.sum_z_dr2) / 0.0008967253   # -8.1%  sum_z_dr2 < 0.00289
        + 0.07603347 * max(0.0, 45.2 - Q.pt_7) / 11.68196   # +7.6%  pt_7 < 45.2
        + 0.07324901 * max(0.0, 0.0217 - Q.zdr_0) * max(0.0, 0.321 - Q.z_dr_0p05_0p1) / 0.002391506   # +7.3%  zdr_0 < 0.0217 and z_dr_0p05_0p1 < 0.321
        + 0.05951879 * max(0.0, 0.00222 - Q.sum_z_dr2_top3) / 0.0008131663   # +6.0%  sum_z_dr2_top3 < 0.00222
        + 0.04860953 * max(0.0, Q.lam1 - 0.00752) / 0.002026658   # +4.9%  lam1 > 0.00752
        - 0.04700801 * max(0.0, 0.0228 - Q.centroid_offset) / 0.009799432   # -4.7%  centroid_offset < 0.0228
        + 0.04293929 * max(0.0, 50.9 - Q.pt_7) * max(0.0, 1.07 - Q.D2) / 3.050587   # +4.3%  pt_7 < 50.9 and D2 < 1.07
        + 0.03590876 * max(0.0, Q.tau1 - 0.0524) * max(0.0, 1.12 - Q.D2) / 0.01217132   # +3.6%  tau1 > 0.0524 and D2 < 1.12
        - 0.03259482 * max(0.0, Q.lam2 - -2.81e-05) * max(0.0, Q.planar_flow - 0.0763) / 0.0003043734   # -3.3%  lam2 > -2.81e-05 and planar_flow > 0.0763
        + 0.0285207 * max(0.0, 0.0213 - Q.zdr_0) * max(0.0, Q.centroid_offset - 0.0104) / 4.207289e-05   # +2.9%  zdr_0 < 0.0213 and centroid_offset > 0.0104
        + 0.02705873 * max(0.0, Q.sj3_pair_mass_min - 14.5) / 1.501848   # +2.7%  sj3_pair_mass_min > 14.5
        - 0.02017011 * max(0.0, Q.lam1 - 0.00681) * max(0.0, 0.484 - Q.D2_b2) / 0.0004390224   # -2.0%  lam1 > 0.00681 and D2_b2 < 0.484
        - 0.01706213 * max(0.0, 0.00665 - Q.mean_phi) / 0.009712863   # -1.7%  mean_phi < 0.00665
        - 0.009517815 * max(0.0, 0.00588 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.263) / 6.985387e-05   # -1.0%  lam1 < 0.00588 and z_dr_0p05_0p1 > 0.263
        - 0.009120073 * max(0.0, Q.mass_top5 - 45.3) / 3.615671   # -0.9%  mass_top5 > 45.3
        + 0.008851892 * max(0.0, 0.0203 - Q.zdr_0) * max(0.0, Q.dr_7 - 0.0795) / 0.0001134335   # +0.9%  zdr_0 < 0.0203 and dr_7 > 0.0795
        - 0.006803767 * max(0.0, Q.sum_z_dr2 - 0.0252) / 0.0002904856   # -0.7%  sum_z_dr2 > 0.0252
        - 0.006375204 * max(0.0, 6.05 - Q.log_sum_pt) / 0.004797894   # -0.6%  log_sum_pt < 6.05
        + 0.005774468 * max(0.0, Q.LHA - 0.284) * max(0.0, Q.sj3_mass1 - 6.1) / 0.009842669   # +0.6%  LHA > 0.284 and sj3_mass1 > 6.1
        + 0.0035597 * max(0.0, Q.sum_pt - 986.0) / 4.522461   # +0.4%  sum_pt > 986
        + 0.003257786 * max(0.0, Q.mean_eta - 0.0243) / 0.0008584821   # +0.3%  mean_eta > 0.0243
        - 0.00305117 * max(0.0, Q.lam2 - 0.00107) * max(0.0, 33.2 - Q.pt_6) / 0.0003651751   # -0.3%  lam2 > 0.00107 and pt_6 < 33.2
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 24.02;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.02367 * (-0.169
        + 0.2952144 * max(0.0, 0.0128 - Q.sum_z_dr2) / 0.007862676   # +29.5%  sum_z_dr2 < 0.0128
        + 0.1473728 * max(0.0, 0.0407 - Q.centroid_offset) / 0.02493265   # +14.7%  centroid_offset < 0.0407
        - 0.08999557 * max(0.0, 0.0718 - Q.sum_z_dr) / 0.02508148   # -9.0%  sum_z_dr < 0.0718
        + 0.06139331 * max(0.0, Q.z_7 - 0.0176) / 0.03486744   # +6.1%  z_7 > 0.0176
        + 0.06101395 * max(0.0, 0.26 - Q.sj3_dr_max) / 0.1110439   # +6.1%  sj3_dr_max < 0.26
        - 0.05093686 * max(0.0, 0.173 - Q.sj3_dr_max) / 0.05077553   # -5.1%  sj3_dr_max < 0.173
        + 0.04437144 * max(0.0, 0.281 - Q.planar_flow) / 0.126449   # +4.4%  planar_flow < 0.281
        - 0.04212296 * max(0.0, 0.0147 - Q.centroid_offset) * max(0.0, 15.4 - Q.sj3_pair_mass_min) / 0.05059741   # -4.2%  centroid_offset < 0.0147 and sj3_pair_mass_min < 15.4
        - 0.02211341 * max(0.0, 0.0264 - Q.sum_z_dr) / 0.004183033   # -2.2%  sum_z_dr < 0.0264
        - 0.02202546 * max(0.0, 0.0206 - Q.sum_z_dr) / 0.002606564   # -2.2%  sum_z_dr < 0.0206
        - 0.01853703 * max(0.0, 0.0363 - Q.centroid_offset) * max(0.0, 912.0 - Q.sum_pt) / 3.45215   # -1.9%  centroid_offset < 0.0363 and sum_pt < 912
        + 0.01849365 * max(0.0, 16.0 - Q.mass) * max(0.0, 0.00455 - Q.zdr_7) / 0.009255947   # +1.8%  mass < 16 and zdr_7 < 0.00455
        + 0.01554965 * max(0.0, 0.0151 - Q.sum_z_dr2) * max(0.0, Q.pt_6 - 21.1) / 0.1935543   # +1.6%  sum_z_dr2 < 0.0151 and pt_6 > 21.1
        - 0.01499822 * max(0.0, Q.sj2_dr - 0.164) * max(0.0, 0.00122 - Q.lam2) / 2.537411e-05   # -1.5%  sj2_dr > 0.164 and lam2 < 0.00122
        - 0.01430298 * max(0.0, 0.0135 - Q.centroid_offset) * max(0.0, 0.00286 - Q.zdr_7) / 5.327288e-06   # -1.4%  centroid_offset < 0.0135 and zdr_7 < 0.00286
        + 0.01375727 * max(0.0, 0.0144 - Q.centroid_offset) * max(0.0, 0.55 - Q.D2_b2) / 0.00079067   # +1.4%  centroid_offset < 0.0144 and D2_b2 < 0.55
        - 0.01130764 * max(0.0, 535.0 - Q.sum_pt_top5) / 45.42659   # -1.1%  sum_pt_top5 < 535
        - 0.0101553 * max(0.0, Q.centroid_offset - 0.0479) / 0.0009137359   # -1.0%  centroid_offset > 0.0479
        - 0.008982965 * max(0.0, 0.265 - Q.planar_flow) * max(0.0, 1.08e-05 - Q.e3) / 2.016858e-07   # -0.9%  planar_flow < 0.265 and e3 < 1.08e-05
        - 0.006970178 * max(0.0, 0.0285 - Q.sum_z_dr) * max(0.0, Q.sj3_pair_mass_min - 1.58) / 0.00581421   # -0.7%  sum_z_dr < 0.0285 and sj3_pair_mass_min > 1.58
        - 0.005072132 * max(0.0, 48.1 - Q.mass) * max(0.0, 1.33 - Q.D2) / 1.183022   # -0.5%  mass < 48.1 and D2 < 1.33
        + 0.004981853 * max(0.0, 0.304 - Q.sj3_dr_max) * max(0.0, -0.0201 - Q.phi_0) / 0.0006366085   # +0.5%  sj3_dr_max < 0.304 and phi_0 < -0.0201
        - 0.003902243 * max(0.0, Q.sj2_dr - 0.18) * max(0.0, Q.n_pt_above_50 - 4.21) / 0.02428658   # -0.4%  sj2_dr > 0.18 and n_pt_above_50 > 4.21
        + 0.003700821 * max(0.0, 0.00462 - Q.sum_z_dr2) * max(0.0, Q.sj3_dr13 - 0.165) / 7.056135e-06   # +0.4%  sum_z_dr2 < 0.00462 and sj3_dr13 > 0.165
        + 0.003479609 * max(0.0, Q.z_dr_0p05_0p1 - 0.846) / 0.01077229   # +0.3%  z_dr_0p05_0p1 > 0.846
        + 0.002305282 * max(0.0, 0.0111 - Q.sum_z_dr2) * max(0.0, 0.44 - Q.D2) / 6.957455e-05   # +0.2%  sum_z_dr2 < 0.0111 and D2 < 0.44
        - 0.002105505 * max(0.0, Q.m012 - 43.0) / 1.013667   # -0.2%  m012 > 43
        - 0.00184046 * max(0.0, 14.7 - Q.mass) * max(0.0, -0.00424 - Q.phi_0) / 0.005425104   # -0.2%  mass < 14.7 and phi_0 < -0.00424
        + 0.001512194 * max(0.0, Q.sj2_dr - 0.161) * max(0.0, 0.0501 - Q.dr_7) / 0.0001168118   # +0.2%  sj2_dr > 0.161 and dr_7 < 0.0501
        - 0.001484823 * max(0.0, Q.sum_pt - 1010.0) / 3.497147   # -0.1%  sum_pt > 1010
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4817;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4817318 * (-2.050934
        + 0.3362753 * max(0.0, Q.sum_z_dr2 - 0.0189) / 0.0007534629   # +33.6%  sum_z_dr2 > 0.0189
        + 0.1862126 * max(0.0, Q.mass - 60.7) * max(0.0, Q.centroid_offset - 0.00297) / 0.08881635   # +18.6%  mass > 60.7 and centroid_offset > 0.00297
        + 0.1263379 * max(0.0, Q.mass - 82.7) / 1.045721   # +12.6%  mass > 82.7
        - 0.1081041 * max(0.0, Q.sum_z_dr2 - 0.0139) * max(0.0, Q.lam2 - -2.24e-05) / 5.056035e-06   # -10.8%  sum_z_dr2 > 0.0139 and lam2 > -2.24e-05
        + 0.1020055 * max(0.0, Q.centroid_offset - 0.0454) / 0.001065929   # +10.2%  centroid_offset > 0.0454
        - 0.09675716 * max(0.0, Q.zdr_0 - 0.0398) / 0.000570514   # -9.7%  zdr_0 > 0.0398
        - 0.04430746 * max(0.0, Q.sum_z_dr2_top2 - 0.00676) * max(0.0, -0.00459 - Q.mean_phi) / 1.958193e-05   # -4.4%  sum_z_dr2_top2 > 0.00676 and mean_phi < -0.00459
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 11.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.9088 * (0.03442831
        + 0.3355822 * max(0.0, 0.148 - Q.sum_z_dr) / 0.0902118   # +33.6%  sum_z_dr < 0.148
        + 0.1674466 * max(0.0, 0.0146 - Q.lam1_plus_lam2) / 0.009406076   # +16.7%  lam1_plus_lam2 < 0.0146
        - 0.1105827 * max(0.0, 0.153 - Q.sum_z_dr) * max(0.0, 39.7 - Q.pt_7) / 0.7746514   # -11.1%  sum_z_dr < 0.153 and pt_7 < 39.7
        - 0.08382698 * max(0.0, 0.147 - Q.sum_z_dr) * max(0.0, 6.87 - Q.log_sum_pt) / 0.02464887   # -8.4%  sum_z_dr < 0.147 and log_sum_pt < 6.87
        + 0.06491803 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 43.0 - Q.pt_7) / 1007.948   # +6.5%  sum_pt_top5 > 640 and pt_7 < 43
        + 0.05920579 * max(0.0, Q.log_sum_pt - 6.49) / 0.132532   # +5.9%  log_sum_pt > 6.49
        - 0.04291373 * max(0.0, 31.4 - Q.pt_6) * max(0.0, 0.0719 - Q.z_7) / 0.08632621   # -4.3%  pt_6 < 31.4 and z_7 < 0.0719
        - 0.02868296 * max(0.0, Q.mass - 50.5) / 7.3935   # -2.9%  mass > 50.5
        - 0.02695564 * max(0.0, 0.000432 - Q.sum_z_dr2_top2) / 8.723082e-05   # -2.7%  sum_z_dr2_top2 < 0.000432
        + 0.01956858 * max(0.0, 0.129 - Q.sum_z_dr) * max(0.0, 0.0793 - Q.M3) / 0.0006583005   # +2.0%  sum_z_dr < 0.129 and M3 < 0.0793
        + 0.01954495 * max(0.0, 0.146 - Q.sum_z_dr) * max(0.0, Q.tau2 - 0.00911) / 0.000248672   # +2.0%  sum_z_dr < 0.146 and tau2 > 0.00911
        - 0.01638607 * max(0.0, Q.sj3_dr23 - 0.189) / 0.02699011   # -1.6%  sj3_dr23 > 0.189
        + 0.01344341 * max(0.0, Q.sum_pt - 988.0) / 4.42251   # +1.3%  sum_pt > 988
        + 0.01094242 * max(0.0, Q.sum_pt_top5 - 774.0) * max(0.0, 29.8 - Q.pt_6) / 130.4416   # +1.1%  sum_pt_top5 > 774 and pt_6 < 29.8
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 20.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.75195 * (-0.003141874
        - 0.1386887 * max(0.0, Q.lam1_plus_lam2 - 0.00672) / 0.002689776   # -13.9%  lam1_plus_lam2 > 0.00672
        + 0.07829065 * max(0.0, Q.sum_z_dr - 0.0266) / 0.03634639   # +7.8%  sum_z_dr > 0.0266
        + 0.0710719 * max(0.0, Q.mass_over_sum_pt - 0.0798) / 0.01092504   # +7.1%  mass_over_sum_pt > 0.0798
        - 0.06470615 * max(0.0, Q.mass_over_sum_pt - 0.0889) / 0.008607555   # -6.5%  mass_over_sum_pt > 0.0889
        - 0.05520591 * max(0.0, Q.sum_z_dr - 0.087) / 0.007900897   # -5.5%  sum_z_dr > 0.087
        + 0.0519704 * max(0.0, Q.lam1_plus_lam2 - 0.00364) / 0.004024205   # +5.2%  lam1_plus_lam2 > 0.00364
        + 0.05068186 * max(0.0, Q.lam1_plus_lam2 - 0.0125) / 0.001553541   # +5.1%  lam1_plus_lam2 > 0.0125
        + 0.0494873 * max(0.0, Q.sj2_dr - 0.159) / 0.03934705   # +4.9%  sj2_dr > 0.159
        - 0.04374716 * max(0.0, 75.0 - Q.sd_mass) / 42.42237   # -4.4%  sd_mass < 75
        - 0.0408706 * max(0.0, 8.4e-05 - Q.e3) / 5.849273e-05   # -4.1%  e3 < 8.4e-05
        + 0.03245775 * max(0.0, 0.107 - Q.planar_flow) * max(0.0, 0.0165 - Q.lam1) / 0.0003020455   # +3.2%  planar_flow < 0.107 and lam1 < 0.0165
        + 0.03168645 * max(0.0, Q.LHA - 0.305) / 0.01744179   # +3.2%  LHA > 0.305
        - 0.02858122 * max(0.0, 0.109 - Q.planar_flow) * max(0.0, 0.00734 - Q.lam1) / 5.703039e-05   # -2.9%  planar_flow < 0.109 and lam1 < 0.00734
        + 0.02670476 * max(0.0, 50.8 - Q.sd_mass) / 23.18727   # +2.7%  sd_mass < 50.8
        + 0.02620226 * max(0.0, Q.e2 - 0.0414) / 0.005034702   # +2.6%  e2 > 0.0414
        + 0.02074744 * max(0.0, Q.sj2_dr - 0.158) * max(0.0, 0.0893 - Q.D2_b2) / 0.0006664858   # +2.1%  sj2_dr > 0.158 and D2_b2 < 0.0893
        - 0.01613093 * max(0.0, Q.sj2_dr - 0.201) / 0.02292796   # -1.6%  sj2_dr > 0.201
        + 0.01542902 * max(0.0, 4.97 - Q.n_dr_0_0p05) / 1.928808   # +1.5%  n_dr_0_0p05 < 4.97
        - 0.01469953 * max(0.0, Q.psi_0p1 - 0.975) / 0.01101241   # -1.5%  psi_0p1 > 0.975
        - 0.01427186 * max(0.0, 717.0 - Q.sum_pt) / 70.68471   # -1.4%  sum_pt < 717
        - 0.01342641 * max(0.0, Q.sj2_dr - 0.13) * max(0.0, 0.09 - Q.D2_b2) / 0.001028133   # -1.3%  sj2_dr > 0.13 and D2_b2 < 0.09
        - 0.01266365 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, 0.0183 - Q.centroid_offset) / 0.0002285177   # -1.3%  planar_flow < 0.115 and centroid_offset < 0.0183
        - 0.01153051 * max(0.0, Q.e2 - 0.051) / 0.003264399   # -1.2%  e2 > 0.051
        - 0.01099851 * max(0.0, Q.sum_zz_dr2 - 0.00219) * max(0.0, 0.187 - Q.sj2_dr) / 1.902003e-05   # -1.1%  sum_zz_dr2 > 0.00219 and sj2_dr < 0.187
        - 0.009945108 * max(0.0, Q.sj2_dr - 0.2) * max(0.0, 0.087 - Q.D2_b2) / 0.0003302086   # -1.0%  sj2_dr > 0.2 and D2_b2 < 0.087
        + 0.00867399 * max(0.0, 0.112 - Q.planar_flow) * max(0.0, 0.00597 - Q.lam1_plus_lam2) / 3.169052e-05   # +0.9%  planar_flow < 0.112 and lam1_plus_lam2 < 0.00597
        - 0.007816629 * max(0.0, Q.sd_rg - 0.237) / 0.01013814   # -0.8%  sd_rg > 0.237
        - 0.007791464 * max(0.0, 0.161 - Q.z_dr_0p05_0p1) / 0.08509897   # -0.8%  z_dr_0p05_0p1 < 0.161
        + 0.006756154 * max(0.0, Q.centroid_offset - 0.0257) * max(0.0, 158.0 - Q.pt_1) / 0.2197544   # +0.7%  centroid_offset > 0.0257 and pt_1 < 158
        + 0.005244478 * max(0.0, 5.72e-05 - Q.e3) * max(0.0, Q.sj3_dr23 - 0.165) / 4.370808e-07   # +0.5%  e3 < 5.72e-05 and sj3_dr23 > 0.165
        - 0.004579677 * max(0.0, Q.centroid_offset - 0.0464) / 0.001002502   # -0.5%  centroid_offset > 0.0464
        - 0.003680653 * max(0.0, Q.mass - 67.3) / 2.77748   # -0.4%  mass > 67.3
        - 0.003342557 * max(0.0, Q.sj2_dr - 0.159) * max(0.0, 0.033 - Q.dr_2) / 6.305869e-05   # -0.3%  sj2_dr > 0.159 and dr_2 < 0.033
        + 0.002940286 * max(0.0, Q.sj2_dr - 0.202) * max(0.0, 0.0526 - Q.dr_2) / 0.0001349926   # +0.3%  sj2_dr > 0.202 and dr_2 < 0.0526
        + 0.002877857 * max(0.0, Q.psi_0p1 - 0.82) * max(0.0, 0.085 - Q.D2_b2) / 0.0006087781   # +0.3%  psi_0p1 > 0.82 and D2_b2 < 0.085
        + 0.002688775 * max(0.0, Q.z_dr_0p05_0p1 - 0.739) / 0.02468908   # +0.3%  z_dr_0p05_0p1 > 0.739
        - 0.002528728 * max(0.0, Q.sd_rg - 0.281) * max(0.0, 0.271 - Q.D2_b2) / 0.0004005804   # -0.3%  sd_rg > 0.281 and D2_b2 < 0.271
        - 0.002362866 * max(0.0, 0.605 - Q.z_dr_0p05_0p1) * max(0.0, 0.156 - Q.D2_b2) / 0.00678203   # -0.2%  z_dr_0p05_0p1 < 0.605 and D2_b2 < 0.156
        - 0.002235693 * max(0.0, Q.centroid_offset - 0.0514) * max(0.0, 0.000846 - Q.C2_b2) / 1.250539e-07   # -0.2%  centroid_offset > 0.0514 and C2_b2 < 0.000846
        - 0.002224314 * max(0.0, 0.118 - Q.planar_flow) * max(0.0, Q.z_dr_0p05_0p1 - 0.682) / 0.00231954   # -0.2%  planar_flow < 0.118 and z_dr_0p05_0p1 > 0.682
        + 0.001425219 * max(0.0, 73.8 - Q.sd_mass) * max(0.0, Q.sj2_mass2 - 3.15) / 8.98969   # +0.1%  sd_mass < 73.8 and sj2_mass2 > 3.15
        - 0.001068248 * max(0.0, Q.z_dr_0p05_0p1 - 0.753) * max(0.0, Q.zdr_5 - 0.0155) / 1.74553e-06   # -0.1%  z_dr_0p05_0p1 > 0.753 and zdr_5 > 0.0155
        - 0.0007736913 * max(0.0, 0.136 - Q.planar_flow) * max(0.0, 0.00163 - Q.zdr_7) / 5.017375e-06   # -0.1%  planar_flow < 0.136 and zdr_7 < 0.00163
        - 0.000601301 * max(0.0, Q.sj2_dr - 0.147) * max(0.0, 0.0231 - Q.dr_3) / 2.158852e-05   # -0.1%  sj2_dr > 0.147 and dr_3 < 0.0231
        + 0.0001913718 * max(0.0, Q.psi_0p1 - 0.778) * max(0.0, Q.phi_4 - 0.101) / 0.000116804   # +0.0%  psi_0p1 > 0.778 and phi_4 > 0.101
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 9.573;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.573028 * (0.003990378
        - 0.2424318 * max(0.0, 0.00816 - Q.sum_z_dr2) / 0.004036184   # -24.2%  sum_z_dr2 < 0.00816
        + 0.1662017 * max(0.0, 0.02 - Q.lam1_plus_lam2) / 0.01420583   # +16.6%  lam1_plus_lam2 < 0.02
        + 0.1653795 * max(0.0, 0.161 - Q.sj2_dr) / 0.04932034   # +16.5%  sj2_dr < 0.161
        - 0.1340221 * max(0.0, 0.195 - Q.sj2_dr) / 0.0697281   # -13.4%  sj2_dr < 0.195
        + 0.1009228 * max(0.0, 0.0413 - Q.e2) / 0.01772727   # +10.1%  e2 < 0.0413
        - 0.04791898 * max(0.0, 0.00189 - Q.sum_z_dr2_top3) / 0.0006516047   # -4.8%  sum_z_dr2_top3 < 0.00189
        - 0.03554715 * max(0.0, 0.00776 - Q.sum_z_dr2) * max(0.0, 0.765 - Q.D2) / 0.0001123082   # -3.6%  sum_z_dr2 < 0.00776 and D2 < 0.765
        - 0.0289202 * max(0.0, 0.000542 - Q.sum_z_dr2_top2) / 0.0001193335   # -2.9%  sum_z_dr2_top2 < 0.000542
        + 0.02721794 * max(0.0, 0.25 - Q.N2) / 0.06546687   # +2.7%  N2 < 0.25
        + 0.02594881 * max(0.0, 0.141 - Q.mass_over_sum_pt) * max(0.0, 0.774 - Q.D2) / 0.005330658   # +2.6%  mass_over_sum_pt < 0.141 and D2 < 0.774
        - 0.01357497 * max(0.0, 0.266 - Q.N2) * max(0.0, 0.489 - Q.z_dr_0p05_0p1) / 0.01403386   # -1.4%  N2 < 0.266 and z_dr_0p05_0p1 < 0.489
        + 0.01191415 * max(0.0, 0.207 - Q.N2) * max(0.0, Q.sum_pt_top5 - 412.0) / 8.205359   # +1.2%  N2 < 0.207 and sum_pt_top5 > 412
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [2.405421008403361, 1.7859640756302522, 2.946674894957983, 1.7585268907563025, 2.7586915966386556, 4.404541386554622, 3.0543039915966386, 3.7997409663865547, 0.5717138655462185, 3.432285294117647, 4.12818518907563, 5.017447268907563, 0.13216533613445378, 4.933523739495798, 0.9063273109243698, 0.5322733193277311]
T = [4.175687774914653, 2.3033792492778358, 7.0719035139837185, 5.2233546431854, 5.3190569130777305]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +30%, n5 -20%, n1 +17%, n9 +14%, n0 -9%, n6 +8% ...
            + 0.3032194 * h[2] / H_AVG[2]
            - 0.1977762 * h[5] / H_AVG[5]
            + 0.1670724 * h[1] / H_AVG[1]
            + 0.1412759 * h[9] / H_AVG[9]
            - 0.09000841 * h[0] / H_AVG[0]
            + 0.08000227 * h[6] / H_AVG[6]
            - 0.02064549 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +38%, n10 -22%, n6 +17%, n4 -11%, n5 +9%, n8 +2% ...
            + 0.3783479 * h[9] / H_AVG[9]
            - 0.2240287 * h[10] / H_AVG[10]
            + 0.1657513 * h[6] / H_AVG[6]
            - 0.1122817 * h[4] / H_AVG[4]
            + 0.08963477 * h[5] / H_AVG[5]
            + 0.01551291 * h[8] / H_AVG[8]
            + 0.01444273 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +27%, n6 -13%, n3 -12%, n7 +12%, n0 +12%, n14 -10% ...
            + 0.2660589 * h[11] / H_AVG[11]
            - 0.1349665 * h[6] / H_AVG[6]
            - 0.1243319 * h[3] / H_AVG[3]
            + 0.1175346 * h[7] / H_AVG[7]
            + 0.1169223 * h[0] / H_AVG[0]
            - 0.09611917 * h[14] / H_AVG[14]
            - 0.05174532 * h[15] / H_AVG[15]
            + 0.04905163 * h[13] / H_AVG[13]
            - 0.02021075 * h[8] / H_AVG[8]
            - 0.01516691 * h[9] / H_AVG[9]
            - 0.007891988 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +34%, n6 -22%, n3 -19%, n14 +7%, n13 +5%, n1 +4% ...
            + 0.3409932 * h[7] / H_AVG[7]
            - 0.2192775 * h[6] / H_AVG[6]
            - 0.1893747 * h[3] / H_AVG[3]
            + 0.0650679 * h[14] / H_AVG[14]
            + 0.05165303 * h[13] / H_AVG[13]
            + 0.04273987 * h[1] / H_AVG[1]
            + 0.04126137 * h[4] / H_AVG[4]
            - 0.02053449 * h[9] / H_AVG[9]
            - 0.01592228 * h[15] / H_AVG[15]
            + 0.01317562 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -38%, n10 +29%, n5 -21%, n4 +6%, n3 +2%, n8 +2% ...
            - 0.3768044 * h[13] / H_AVG[13]
            + 0.2910421 * h[10] / H_AVG[10]
            - 0.207017 * h[5] / H_AVG[5]
            + 0.06483037 * h[4] / H_AVG[4]
            + 0.02066305 * h[3] / H_AVG[3]
            + 0.02015326 * h[8] / H_AVG[8]
            - 0.01242376 * h[12] / H_AVG[12]
            + 0.007066046 * h[0] / H_AVG[0]
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
