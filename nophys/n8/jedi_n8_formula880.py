"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.9% (the network: 65.8%); same class as the network for 88.6% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
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
  Q.pair_mass_0_2          mass of particles 0 and 2 [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.sj3_mass3              mass of subjet 3 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.n_for_50pct            number of hardest particles that carry 50% of the jet pT
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.z_1st                  largest pT share
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_2               |Δη| of particle 2
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_6               |Δη| of particle 6
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_5               |Δφ| of particle 5
  Q.absphi_6               |Δφ| of particle 6
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
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
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.centroid_offset        distance of the pT centroid from the jet axis
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
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
        pair_mass_0_2=pair_mass(0, 2),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass2=subjets(3)["mass"][1],
        sj3_mass3=subjets(3)["mass"][2],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        m01=pair_mass(0, 1),
        n_for_50pct=ncum(0.5),
        n_for_90pct=ncum(0.9),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_3=z[3] * dr[3],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        z_1st=zs[0],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_2=abs(eta[2]),
        abseta_4=abs(eta[4]),
        abseta_6=abs(eta[6]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_5=abs(phi[5]),
        absphi_6=abs(phi[6]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
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
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -1.747012
    if Q.planar_flow < 0.1484197:
        z += -9.377076 * Q.planar_flow + 1.391743
    if Q.lam1_plus_lam2 < 0.004372139:
        z += 190.1512 * Q.lam1_plus_lam2 + 1.332327
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -502.4946 * Q.lam1_plus_lam2 + 4.360671
    if Q.sum_z_dr2 < 0.01323868:
        z += -385.7182 * Q.sum_z_dr2 + 5.349401
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -43.47897 * Q.sum_z_dr2 + 0.8186069
    if Q.mass < 21.78408:
        z += 0.2859868 * Q.mass - 8.360302
    if 21.78408 <= Q.mass < 56.92035:
        z += 0.05226809 * Q.mass - 3.268956
    if 56.92035 <= Q.mass < 64.61873:
        z += 0.03816879 * Q.mass - 2.466419
    if Q.sum_z_dr2_top3 < 0.006756161:
        z += 39.64328 * Q.sum_z_dr2_top3 - 0.4923341
    if 0.006756161 <= Q.sum_z_dr2_top3 < 0.007929074:
        z += 191.4019 * Q.sum_z_dr2_top3 - 1.51764
    if Q.sum_pt >= 901.5938:
        z += -0.01300685 * Q.sum_pt + 11.72689
    if Q.C2_b2 < 0.001563465:
        z += 493.3141 * Q.C2_b2 - 0.7712795
    if Q.sj3_dr_max < 0.04889979:
        z += 16.14913 * Q.sj3_dr_max + 0.0055471
    if 0.04889979 <= Q.sj3_dr_max < 0.1070199:
        z += -3.151923 * Q.sj3_dr_max + 0.9493643
    if 0.1070199 <= Q.sj3_dr_max < 0.1789613:
        z += 14.88402 * Q.sj3_dr_max - 0.980841
    if 0.1789613 <= Q.sj3_dr_max < 0.233678:
        z += 0.5424337 * Q.sj3_dr_max + 1.585748
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -7.920117 * Q.sj3_dr_max + 3.56326
    if Q.sj3_dr_max >= 0.3012016:
        z += -4.768194 * Q.sj3_dr_max + 2.613896
    if 0.006789738 <= Q.centroid_offset < 0.02076709:
        z += -43.24432 * Q.centroid_offset + 0.2936176
    if 0.02076709 <= Q.centroid_offset < 0.04990367:
        z += -52.23292 * Q.centroid_offset + 0.4802847
    if Q.centroid_offset >= 0.04990367:
        z += -406.0206 * Q.centroid_offset + 18.13559
    if Q.sum_z_dr < 0.07608178:
        z += 81.63953 * Q.sum_z_dr - 6.575142
    if 0.07608178 <= Q.sum_z_dr < 0.08723651:
        z += 32.61945 * Q.sum_z_dr - 2.845607
    if Q.e2 < 0.0245477:
        z += -58.9847 * Q.e2 + 1.890363
    if 0.0245477 <= Q.e2 < 0.03556091:
        z += -40.17211 * Q.e2 + 1.428557
    if Q.sum_z_dr2_top5 < 0.008329695:
        z += 53.01488 * Q.sum_z_dr2_top5 - 0.4415978
    if Q.lam1 < 0.0002758826:
        z += 5164.608 * Q.lam1 - 2.824018
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 271.2939 * Q.lam1 - 1.474038
    if Q.log_sum_pt < 6.080494:
        z += 2.52986 * Q.log_sum_pt - 15.3828
    if 6.377723 <= Q.log_sum_pt < 6.670067:
        z += 1.772283 * Q.log_sum_pt - 11.30313
    if Q.log_sum_pt >= 6.670067:
        z += -2.133653 * Q.log_sum_pt + 14.74972
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += -336.4628 * Q.mass_over_sum_pt_sq + 1.31375
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.2269809 * Q.n_dr_0_0p05 - 0.9079235
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.005376956 * Q.sum_pt_top5 - 3.696321
    if Q.lam2 < 7.300726e-05:
        z += 10378.64 * Q.lam2 - 0.7577161
    if Q.zdr_1 < 0.01587232:
        z += -14.42797 * Q.zdr_1 + 0.2290053
    if Q.mass_over_sum_pt < 0.08475161:
        z += 15.15032 * Q.mass_over_sum_pt - 0.9746527
    if 0.08475161 <= Q.mass_over_sum_pt < 0.1309286:
        z += -6.699461 * Q.mass_over_sum_pt + 0.8771513
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -893.652 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 1.002471:
        z += 84.03558 * (0.01323868 - Q.sum_z_dr2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.04253713 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.sum_z_dr2 < 0.01323868 and Q.mean_phi < -0.003096655:
        z += -2740.752 * (0.01323868 - Q.sum_z_dr2) * (-0.003096655 - Q.mean_phi)
    if Q.sum_z_dr2 < 0.01882765 and Q.mass_top2 > 28.78966:
        z += 3.275313 * (0.01882765 - Q.sum_z_dr2) * (Q.mass_top2 - 28.78966)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -2219.744 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.01837778)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 90.36867 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 53889.68 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -51.30435 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.mass < 56.92035 and Q.dr_max_012 > 0.06112084:
        z += -0.1670612 * (56.92035 - Q.mass) * (Q.dr_max_012 - 0.06112084)
    if Q.log_sum_pt > 6.377723 and Q.mean_eta > 6.288824e-05:
        z += -49.95764 * (Q.log_sum_pt - 6.377723) * (Q.mean_eta - 6.288824e-05)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_4 < 0.06336451:
        z += -0.08750305 * (Q.sum_pt_top5 - 687.4375) * (0.06336451 - Q.dr_4)
    if Q.lam2 < 7.300726e-05 and Q.dr_2 < 0.08525808:
        z += 71657.34 * (7.300726e-05 - Q.lam2) * (0.08525808 - Q.dr_2)
    if Q.sum_pt > 901.5938 and Q.pt_7 > 33.21875:
        z += 0.0003122841 * (Q.sum_pt - 901.5938) * (Q.pt_7 - 33.21875)
    if Q.log_sum_pt > 6.670067 and Q.z_7 > 0.01685855:
        z += -308.0768 * (Q.log_sum_pt - 6.670067) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.3012016 and Q.z_7 < 0.05557716:
        z += -119.7702 * (0.3012016 - Q.sj3_dr_max) * (0.05557716 - Q.z_7)
    if Q.sj3_dr_max < 0.3012016 and Q.n_dr_0p05_0p1 < 2.0:
        z += -1.391967 * (0.3012016 - Q.sj3_dr_max) * (2.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt > 6.670067 and Q.m01 < 28.78966:
        z += 0.1010528 * (Q.log_sum_pt - 6.670067) * (28.78966 - Q.m01)
    if Q.sum_z_dr2_top3 < 0.007929074 and Q.D2_b2 < 0.5327104:
        z += -79.52872 * (0.007929074 - Q.sum_z_dr2_top3) * (0.5327104 - Q.D2_b2)
    if Q.centroid_offset > 0.02076709 and Q.z_7 < 0.06473447:
        z += -1559.429 * (Q.centroid_offset - 0.02076709) * (0.06473447 - Q.z_7)
    if Q.planar_flow < 0.1484197 and Q.pt_dispersion < 0.4160096:
        z += 73.29101 * (0.1484197 - Q.planar_flow) * (0.4160096 - Q.pt_dispersion)
    if Q.centroid_offset > 0.006789738 and Q.n_pt_above_50 < 7.0:
        z += 3.567498 * (Q.centroid_offset - 0.006789738) * (7.0 - Q.n_pt_above_50)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.C3 < 0.03532852:
        z += 7830.962 * (0.004372139 - Q.lam1_plus_lam2) * (0.03532852 - Q.C3)
    if Q.centroid_offset > 0.006789738 and Q.pt_1 < 223.375:
        z += 0.100179 * (Q.centroid_offset - 0.006789738) * (223.375 - Q.pt_1)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.07718453
    if Q.lam1 < 0.0008722282:
        z += 706.0202 * Q.lam1 + 1.377883
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -376.0898 * Q.lam1 + 2.32173
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -34.04397 * Q.lam1 + 0.2851377
    if Q.pt_7 < 25.57812:
        z += 0.05088982 * Q.pt_7 - 1.301666
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.06754494 * Q.pt_7 - 2.332411
    if Q.pt_7 >= 53.4375:
        z += -0.02834681 * Q.pt_7 + 2.791804
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 4.620955 * Q.log_sum_pt - 29.47117
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 13.32706 * Q.log_sum_pt - 86.08524
    if Q.log_sum_pt >= 6.605974:
        z += 17.361 * Q.log_sum_pt - 112.7333
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -130.4892 * Q.mass_over_sum_pt_sq + 0.7611343
    if Q.z_7 < 0.02320757:
        z += 34.93314 * Q.z_7 - 2.15346
    if 0.02320757 <= Q.z_7 < 0.04624032:
        z += 63.48072 * Q.z_7 - 2.81598
    if 0.04624032 <= Q.z_7 < 0.06164517:
        z += 34.62626 * Q.z_7 - 1.48174
    if Q.z_7 >= 0.06164517:
        z += -0.3068867 * Q.z_7 + 0.6717199
    if Q.sum_z_dr2 < 0.005019719:
        z += 484.9381 * Q.sum_z_dr2 - 3.822318
    if 0.005019719 <= Q.sum_z_dr2 < 0.008678045:
        z += 379.4263 * Q.sum_z_dr2 - 3.292678
    if Q.mass < 49.6681:
        z += -0.03496129 * Q.mass + 1.736461
    if Q.sj3_dr_min >= 0.03628191:
        z += -1.993698 * Q.sj3_dr_min + 0.07233517
    if Q.sj3_dr_max >= 0.169029:
        z += 2.981648 * Q.sj3_dr_max - 0.5039851
    if Q.sum_zz_dr2 < 0.008168571:
        z += -92.56419 * Q.sum_zz_dr2 + 0.7561171
    if Q.e3 < 1.627072e-05:
        z += -33843.45 * Q.e3 + 0.9282025
    if 1.627072e-05 <= Q.e3 < 0.0005116989:
        z += -762.0582 * Q.e3 + 0.3899444
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.002110797 * Q.sum_pt_top5 - 0.7759157
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.0125258 * Q.sum_pt_top5 + 6.998863
    if Q.lam1_plus_lam2 < 0.00609665:
        z += 655.3992 * Q.lam1_plus_lam2 - 3.99574
    if Q.sum_z_dr < 0.0717028:
        z += -11.22937 * Q.sum_z_dr + 0.1967308
    if 0.0717028 <= Q.sum_z_dr < 0.1019409:
        z += 20.12184 * Q.sum_z_dr - 2.051239
    if Q.centroid_offset < 0.01837778:
        z += 18.42318 * Q.centroid_offset - 0.3385772
    if Q.tau1 < 0.07283629:
        z += -7.231209 * Q.tau1 + 0.5266944
    if Q.pt_6 >= 62.25:
        z += -0.03036423 * Q.pt_6 + 1.890173
    if Q.zdr_0 < 0.0211821:
        z += 10.39745 * Q.zdr_0 - 0.2202398
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -1341.066 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.627072e-05:
        z += -121298.8 * (Q.log_sum_pt - 6.377723) * (1.627072e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += 1.939296 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.tau2 < 0.06297984:
        z += 667.3378 * (0.06164517 - Q.z_7) * (0.06297984 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -24.46345 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.2112213 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -3.200872 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -27.96348 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.006471971 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.006047993 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 64.99033 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 3.807161 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -731.7369 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.z_7 < 0.06164517 and Q.centroid_offset > 0.02076709:
        z += -831.3567 * (0.06164517 - Q.z_7) * (Q.centroid_offset - 0.02076709)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.05144351 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -49.15528 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.log_sum_pt > 6.605974 and Q.z_4 < 0.09925997:
        z += 21.89487 * (Q.log_sum_pt - 6.605974) * (0.09925997 - Q.z_4)
    if Q.sum_z_dr2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 11442.33 * (0.008678045 - Q.sum_z_dr2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -2140.724 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 1294.336 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    if Q.z_7 < 0.06164517 and Q.sum_z_dr2_top2 < 0.01403324:
        z += 851.9958 * (0.06164517 - Q.z_7) * (0.01403324 - Q.sum_z_dr2_top2)
    if Q.log_sum_pt > 6.502799 and Q.zdr_6 < 0.008654951:
        z += -592.0883 * (Q.log_sum_pt - 6.502799) * (0.008654951 - Q.zdr_6)
    if Q.sum_z_dr < 0.1019409 and Q.pair_mass_0_6 > 4.037975:
        z += -0.3994233 * (0.1019409 - Q.sum_z_dr) * (Q.pair_mass_0_6 - 4.037975)
    if Q.sum_pt_top5 > 531.1875 and Q.zdr_6 < 0.01392641:
        z += 0.2969278 * (Q.sum_pt_top5 - 531.1875) * (0.01392641 - Q.zdr_6)
    if Q.z_7 < 0.06164517 and Q.abseta_0 < 0.1057739:
        z += 83.89929 * (0.06164517 - Q.z_7) * (0.1057739 - Q.abseta_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 3.933101
    if Q.sj3_pair_mass_max < 63.68899:
        z += -0.00936357 * Q.sj3_pair_mass_max + 0.5963563
    if Q.log_sum_pt < 6.46415:
        z += -6.890456 * Q.log_sum_pt + 45.39125
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -5.995487 * Q.log_sum_pt + 39.60603
    if 6.842717 <= Q.log_sum_pt < 6.896095:
        z += 22.84483 * Q.log_sum_pt - 156.3207
    if Q.log_sum_pt >= 6.896095:
        z += 27.3814 * Q.log_sum_pt - 187.6053
    if Q.lam1 < 0.00595415:
        z += -223.4222 * Q.lam1 + 1.962175
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -104.4512 * Q.lam1 + 1.253803
    if Q.z_7 < 0.02320757:
        z += 11.23991 * Q.z_7 + 0.609235
    if 0.02320757 <= Q.z_7 < 0.03243272:
        z += -50.43548 * Q.z_7 + 2.040571
    if 0.03243272 <= Q.z_7 < 0.04939969:
        z += -86.60826 * Q.z_7 + 3.213752
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -76.88698 * Q.z_7 + 2.733524
    if Q.z_7 >= 0.07148865:
        z += -51.95412 * Q.z_7 + 0.9511079
    if Q.sum_z_dr < 0.007673833:
        z += 315.2576 * Q.sum_z_dr - 3.677
    if 0.007673833 <= Q.sum_z_dr < 0.1484084:
        z += 8.937152 * Q.sum_z_dr - 1.326349
    if Q.LHA >= 0.111565:
        z += -7.996646 * Q.LHA + 0.8921462
    if Q.sum_pt_top5 < 605.1875:
        z += 0.008050831 * Q.sum_pt_top5 - 5.771502
    if 605.1875 <= Q.sum_pt_top5 < 716.8828:
        z += 0.004468464 * Q.sum_pt_top5 - 3.603499
    if 716.8828 <= Q.sum_pt_top5 < 839.9547:
        z += -0.003582367 * Q.sum_pt_top5 + 2.168004
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.01803212 * Q.sum_pt_top5 + 14.30514
    if Q.sj3_pair_mass_min < 15.95929:
        z += 0.03494905 * Q.sj3_pair_mass_min - 0.5577618
    if Q.pt_6 < 27.57812:
        z += 0.01971608 * Q.pt_6 - 0.8126723
    if 27.57812 <= Q.pt_6 < 41.21875:
        z += 0.06224745 * Q.pt_6 - 1.985608
    if Q.pt_6 >= 41.21875:
        z += 0.04253136 * Q.pt_6 - 1.172935
    if Q.z_6 >= 0.02886576:
        z += -19.05016 * Q.z_6 + 0.5498974
    if Q.sum_pt < 559.6875:
        z += -0.00830552 * Q.sum_pt + 7.628218
    if 559.6875 <= Q.sum_pt < 788.4484:
        z += -0.01223723 * Q.sum_pt + 9.828745
    if 788.4484 <= Q.sum_pt < 840.0195:
        z += -0.003496577 * Q.sum_pt + 2.937193
    if Q.sum_z_dr2 < 0.0003193707:
        z += 11298.36 * Q.sum_z_dr2 - 2.53391
    if 0.0003193707 <= Q.sum_z_dr2 < 0.005019719:
        z += -228.5906 * Q.sum_z_dr2 + 1.147461
    if Q.pt_7 < 20.125:
        z += 0.04076155 * Q.pt_7 - 2.748047
    if 20.125 <= Q.pt_7 < 43.5:
        z += 0.1001592 * Q.pt_7 - 3.943424
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.1575027 * Q.pt_7 - 6.437867
    if Q.pt_7 >= 53.4375:
        z += 0.05939762 * Q.pt_7 - 1.195377
    if 15.45403 <= Q.mass < 36.22941:
        z += 0.02353659 * Q.mass - 0.3637352
    if Q.mass >= 36.22941:
        z += -0.01382246 * Q.mass + 0.989761
    if Q.zdr_0 < 0.0211821:
        z += -15.75295 * Q.zdr_0 + 0.3336805
    if Q.lam2 < 0.0001947983:
        z += 1124.617 * Q.lam2 - 0.2190734
    if Q.dr_5 < 0.02164863:
        z += 18.12165 * Q.dr_5 - 0.3923089
    if Q.z_top5 < 0.8233866:
        z += -2.055702 * Q.z_top5 + 1.692637
    if Q.z_top5 >= 0.8770155:
        z += 10.67251 * Q.z_top5 - 9.359952
    if Q.pt_5 < 33.02656:
        z += 0.04093784 * Q.pt_5 - 1.352036
    if Q.z_5 < 0.03672711:
        z += -21.55663 * Q.z_5 + 0.7917129
    if Q.sj3_pair_mass_max < 63.68899 and Q.z_7 < 0.06810151:
        z += -0.5425646 * (63.68899 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 63.68899 and Q.centroid_offset > 0.01096064:
        z += -0.2454592 * (63.68899 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.0931108:
        z += -1923.095 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.0931108)
    if Q.z_7 > 0.04939969 and Q.sj3_dr_min < 0.0623951:
        z += -150.6789 * (Q.z_7 - 0.04939969) * (0.0623951 - Q.sj3_dr_min)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += -0.2179838 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.sum_z_dr < 0.007673833 and Q.pt_4 < 71.6875:
        z += 5.634482 * (0.007673833 - Q.sum_z_dr) * (71.6875 - Q.pt_4)
    if Q.pt_6 > 27.57812 and Q.planar_flow < 0.7974684:
        z += -0.009509346 * (Q.pt_6 - 27.57812) * (0.7974684 - Q.planar_flow)
    if Q.log_sum_pt < 6.605974 and Q.max_dr < 0.03623337:
        z += -255.9796 * (6.605974 - Q.log_sum_pt) * (0.03623337 - Q.max_dr)
    if Q.sum_pt < 788.4484 and Q.max_dr < 0.03623337:
        z += 0.4185135 * (788.4484 - Q.sum_pt) * (0.03623337 - Q.max_dr)
    if Q.sum_z_dr2 < 0.0003193707 and Q.mass_top2 < 36.76827:
        z += 298.405 * (0.0003193707 - Q.sum_z_dr2) * (36.76827 - Q.mass_top2)
    if Q.z_7 < 0.07148865 and Q.planar_flow < 0.6947818:
        z += -8.750761 * (0.07148865 - Q.z_7) * (0.6947818 - Q.planar_flow)
    if Q.log_sum_pt > 6.896095 and Q.dr_5 < 0.04416271:
        z += 129.367 * (Q.log_sum_pt - 6.896095) * (0.04416271 - Q.dr_5)
    if Q.log_sum_pt > 6.842717 and Q.dr_5 < 0.04416271:
        z += -132.4859 * (Q.log_sum_pt - 6.842717) * (0.04416271 - Q.dr_5)
    if Q.sum_pt < 840.0195 and Q.dr_5 < 0.02164863:
        z += 1.327501 * (840.0195 - Q.sum_pt) * (0.02164863 - Q.dr_5)
    if Q.sum_pt < 788.4484 and Q.dr_5 < 0.02164863:
        z += -1.417772 * (788.4484 - Q.sum_pt) * (0.02164863 - Q.dr_5)
    if Q.z_6 > 0.02886576 and Q.dr_5 < 0.02164863:
        z += -487.37 * (Q.z_6 - 0.02886576) * (0.02164863 - Q.dr_5)
    if Q.pt_7 > 34.53125 and Q.abseta_2 < 0.1286621:
        z += 0.1150962 * (Q.pt_7 - 34.53125) * (0.1286621 - Q.abseta_2)
    if Q.log_sum_pt < 6.605974 and Q.lam2 < 0.0003061234:
        z += 3303.068 * (6.605974 - Q.log_sum_pt) * (0.0003061234 - Q.lam2)
    if Q.z_7 < 0.07148865 and Q.dr_7 < 0.1242755:
        z += 49.28551 * (0.07148865 - Q.z_7) * (0.1242755 - Q.dr_7)
    if Q.sum_pt_top5 > 839.9547 and Q.D2_b2 < 2.326647:
        z += 0.008751202 * (Q.sum_pt_top5 - 839.9547) * (2.326647 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 3.032555:
        z += -7.950116 * (Q.log_sum_pt - 6.896095) * (3.032555 - Q.D2_b2)
    if Q.pt_7 < 43.5 and Q.D3 < 3.916009:
        z += -0.008414996 * (43.5 - Q.pt_7) * (3.916009 - Q.D3)
    if Q.sj3_pair_mass_min < 15.95929 and Q.D2 > 0.6203774:
        z += 0.00569583 * (15.95929 - Q.sj3_pair_mass_min) * (Q.D2 - 0.6203774)
    if Q.pt_7 < 43.5 and Q.D2_b2 < 3.032555:
        z += -0.005193821 * (43.5 - Q.pt_7) * (3.032555 - Q.D2_b2)
    if Q.sum_pt < 559.6875 and Q.D2_b2 < 4.721224:
        z += 0.001736267 * (559.6875 - Q.sum_pt) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.zdr_5 < 0.002460108:
        z += 2751.51 * (Q.log_sum_pt - 6.896095) * (0.002460108 - Q.zdr_5)
    if Q.lam1 < 0.01200373 and Q.centroid_offset > 0.01627885:
        z += -2698.383 * (0.01200373 - Q.lam1) * (Q.centroid_offset - 0.01627885)
    if Q.pt_6 > 27.57812 and Q.centroid_offset > 0.02076709:
        z += -0.4446456 * (Q.pt_6 - 27.57812) * (Q.centroid_offset - 0.02076709)
    return max(0.0, z)


def neuron_3(Q):
    z = -1.882194
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 31.67716 * Q.mass_over_sum_pt - 2.158453
    if Q.centroid_offset >= 0.01096064:
        z += -29.5797 * Q.centroid_offset + 0.3242124
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -22.74256 * Q.tau1 + 1.2183
    if Q.tau1 >= 0.1027642:
        z += 8.370499 * Q.tau1 - 1.97901
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 779.3499 * Q.lam1 - 6.527501
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 628.7725 * Q.lam1 - 4.720011
    if Q.lam1 >= 0.01643375:
        z += 669.6477 * Q.lam1 - 5.391744
    if 0.04081947 <= Q.sum_z_dr < 0.07608178:
        z += 66.2985 * Q.sum_z_dr - 2.70627
    if 0.07608178 <= Q.sum_z_dr < 0.08723651:
        z += 43.1747 * Q.sum_z_dr - 0.9469698
    if Q.sum_z_dr >= 0.08723651:
        z += 63.5197 * Q.sum_z_dr - 2.721797
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 806.7641 * Q.lam1_plus_lam2 - 6.066937
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += 398.6409 * Q.lam1_plus_lam2 - 0.6639266
    if Q.e2 >= 0.06344108:
        z += -44.07707 * Q.e2 + 2.796297
    if Q.sum_z_dr2 < 0.004372139:
        z += 401.1832 * Q.sum_z_dr2 - 1.754029
    if 0.008678045 <= Q.sum_z_dr2 < 0.01882765:
        z += -1349.919 * Q.sum_z_dr2 + 11.71466
    if Q.sum_z_dr2 >= 0.01882765:
        z += -1213.187 * Q.sum_z_dr2 + 9.140305
    if 0.1492731 <= Q.sj2_dr < 0.1872617:
        z += -7.207498 * Q.sj2_dr + 1.075886
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += 12.0684 * Q.sj2_dr - 2.533752
    if Q.sj2_dr >= 0.2687922:
        z += -1.219891 * Q.sj2_dr + 1.038036
    if Q.mean_eta < -0.004664942:
        z += -12.23259 * Q.mean_eta - 0.05706431
    if Q.mean_eta >= 0.01772426:
        z += 18.31131 * Q.mean_eta - 0.3245544
    if Q.sum_z_dr2_top5 < 0.008329695:
        z += -49.46053 * Q.sum_z_dr2_top5 + 0.4119912
    if Q.max_dr >= 0.1027585:
        z += 6.560723 * Q.max_dr - 0.6741702
    if Q.lam2 >= 0.001130645:
        z += 363.7934 * Q.lam2 - 0.411321
    if Q.mass >= 64.61873:
        z += -0.05319894 * Q.mass + 3.437648
    if Q.sd_mass >= 62.73432:
        z += 0.01599352 * Q.sd_mass - 1.003343
    if Q.z_dr_0_0p05 < 0.05226226:
        z += -6.66803 * Q.z_dr_0_0p05 + 0.3484863
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += -0.4009751 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += -18.33258 * (Q.sum_z_dr - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 0.7706698 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.09389288 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -401.6595 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.centroid_offset > 0.01096064 and Q.n_pt_above_50 < 8.0:
        z += 3.581872 * (Q.centroid_offset - 0.01096064) * (8.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 192.5539 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.tau1 > 0.05356915 and Q.mass_top3 < 50.3522:
        z += -0.08632775 * (Q.tau1 - 0.05356915) * (50.3522 - Q.mass_top3)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 9.743613 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.73432 and Q.D2_b2 < 0.9206502:
        z += 0.03454459 * (Q.sd_mass - 62.73432) * (0.9206502 - Q.D2_b2)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -102.5255 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.mean_eta > 0.01772426 and Q.mean_phi < 0.02612796:
        z += -297.764 * (Q.mean_eta - 0.01772426) * (0.02612796 - Q.mean_phi)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05096142:
        z += 2685.61 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05096142)
    if Q.max_dr > 0.1027585 and Q.eta_1 < -0.05963135:
        z += -35.73621 * (Q.max_dr - 0.1027585) * (-0.05963135 - Q.eta_1)
    if Q.max_dr > 0.1027585 and Q.eta_1 < 0.05792236:
        z += 13.71751 * (Q.max_dr - 0.1027585) * (0.05792236 - Q.eta_1)
    return max(0.0, z)


def neuron_4(Q):
    z = -3.181996
    if Q.N2 < 0.2233283:
        z += -34.78183 * Q.N2 + 7.767765
    if Q.lam2 < 0.000537286:
        z += 3423.547 * Q.lam2 - 2.680503
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += 1417.487 * Q.lam2 - 1.602675
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -91.08377 * Q.mass_over_sum_pt + 8.235232
    if Q.sum_z_dr2 < 0.002635418:
        z += 324.7544 * Q.sum_z_dr2 - 11.63743
    if 0.002635418 <= Q.sum_z_dr2 < 0.003562611:
        z += 1043.454 * Q.sum_z_dr2 - 13.5315
    if 0.003562611 <= Q.sum_z_dr2 < 0.008678045:
        z += 1225.724 * Q.sum_z_dr2 - 14.18086
    if 0.008678045 <= Q.sum_z_dr2 < 0.01882765:
        z += 623.3083 * Q.sum_z_dr2 - 8.953065
    if Q.sum_z_dr2 >= 0.01882765:
        z += 182.2706 * Q.sum_z_dr2 - 0.6493592
    if 0.009668065 <= Q.e2 < 0.04447357:
        z += 88.41229 * Q.e2 - 0.8547758
    if Q.e2 >= 0.04447357:
        z += 172.1049 * Q.e2 - 4.576883
    if Q.sum_pt < 739.5:
        z += 0.002938565 * Q.sum_pt - 2.173069
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -49.97789 * Q.sum_z_dr2_top2 + 0.3818132
    if 0.169029 <= Q.sj3_dr_max < 0.233678:
        z += 14.02046 * Q.sj3_dr_max - 2.369865
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -17.50137 * Q.sj3_dr_max + 4.996092
    if Q.sj3_dr_max >= 0.3456459:
        z += -30.50412 * Q.sj3_dr_max + 9.490442
    if Q.sum_zz_dr2 < 0.01165737:
        z += -4424.508 * Q.sum_zz_dr2 + 54.03384
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -446.0752 * Q.sum_zz_dr2 + 7.655759
    if Q.sum_z_dr < 0.05464922:
        z += -38.02581 * Q.sum_z_dr + 4.736257
    if 0.05464922 <= Q.sum_z_dr < 0.1019409:
        z += -56.4785 * Q.sum_z_dr + 5.744682
    if 0.1019409 <= Q.sum_z_dr < 0.1245537:
        z += -33.09952 * Q.sum_z_dr + 3.361408
    if Q.sum_z_dr >= 0.1245537:
        z += 4.926291 * Q.sum_z_dr - 1.37485
    if 0.07283629 <= Q.tau1 < 0.1136369:
        z += -31.36805 * Q.tau1 + 2.284732
    if Q.tau1 >= 0.1136369:
        z += -67.87851 * Q.tau1 + 6.433667
    if Q.mass < 69.61135:
        z += 0.04530641 * Q.mass - 2.959876
    if 69.61135 <= Q.mass < 76.6557:
        z += -0.02753467 * Q.mass + 2.11069
    if Q.mass >= 88.15578:
        z += -0.09183433 * Q.mass + 8.095727
    if Q.C2 < 0.01867771:
        z += 40.93111 * Q.C2 - 2.09536
    if 0.01867771 <= Q.C2 < 0.05119235:
        z += 15.1768 * Q.C2 - 1.614328
    if Q.C2 >= 0.05119235:
        z += -25.75431 * Q.C2 + 0.4810317
    if Q.sum_z_dr2_top5 < 0.008329695:
        z += -176.1624 * Q.sum_z_dr2_top5 + 1.467379
    if 0.1117619 <= Q.max_dr < 0.2215867:
        z += 17.44402 * Q.max_dr - 1.949576
    if Q.max_dr >= 0.2215867:
        z += 11.84911 * Q.max_dr - 0.7098191
    if Q.D2 < 0.875672:
        z += -5.317658 * Q.D2 + 5.144551
    if 0.875672 <= Q.D2 < 1.002471:
        z += -3.848835 * Q.D2 + 3.858345
    if Q.tau21_b2 < 0.004811143:
        z += 203.9746 * Q.tau21_b2 - 0.9813512
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 408.2635 * Q.lam1_plus_lam2 - 5.404868
    if Q.C2_b2 < 0.009032972:
        z += -255.9926 * Q.C2_b2 + 2.312374
    if 0.006823012 <= Q.mean_eta < 0.02644207:
        z += 18.93931 * Q.mean_eta - 0.1292232
    if Q.mean_eta >= 0.02644207:
        z += -37.6496 * Q.mean_eta + 1.367105
    if Q.tau21 < 0.2838437:
        z += 5.670557 * Q.tau21 - 1.609552
    if Q.sd_mass >= 38.43971:
        z += 0.05120478 * Q.sd_mass - 1.968297
    if 0.1667615 <= Q.sd_rg < 0.2037854:
        z += 3.846761 * Q.sd_rg - 0.6414917
    if 0.2037854 <= Q.sd_rg < 0.324646:
        z += -19.29969 * Q.sd_rg + 4.075417
    if Q.sd_rg >= 0.324646:
        z += 19.0127 * Q.sd_rg - 8.362549
    if Q.e3 < 1.340118e-05:
        z += 15543.78 * Q.e3 - 0.2083051
    if Q.lam1 < 0.002464291:
        z += 179.0215 * Q.lam1 - 0.4411612
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 3193.749 * Q.mass_over_sum_pt_sq - 37.242
    if Q.sum_pt_top5 < 430.75:
        z += 0.002335164 * Q.sum_pt_top5 - 1.005872
    if Q.mass_top5 < 37.76455:
        z += 0.01106923 * Q.mass_top5 - 0.4180244
    if Q.zdr_0 < 0.03981924:
        z += -34.58304 * Q.zdr_0 + 1.37707
    if Q.N2 < 0.2233283 and Q.mass < 60.63098:
        z += 0.1708073 * (0.2233283 - Q.N2) * (60.63098 - Q.mass)
    if Q.N2 < 0.2233283 and Q.sum_zz_dr2 > 0.01165737:
        z += -1900.262 * (0.2233283 - Q.N2) * (Q.sum_zz_dr2 - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_6 < 62.25:
        z += -0.1209771 * (0.2233283 - Q.N2) * (62.25 - Q.pt_6)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 282.9222 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -50.89442 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.sum_pt < 739.5 and Q.n_dr_0p05_0p1 > 0.0:
        z += -0.0001967964 * (739.5 - Q.sum_pt) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -31.53737 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.tau1 > 0.07283629 and Q.sj3_dr_min < 0.2089872:
        z += 224.8797 * (Q.tau1 - 0.07283629) * (0.2089872 - Q.sj3_dr_min)
    if Q.C2 < 0.05119235 and Q.mean_eta > 0.02644207:
        z += -1536.566 * (0.05119235 - Q.C2) * (Q.mean_eta - 0.02644207)
    if Q.mass < 76.6557 and Q.D2 < 0.875672:
        z += -0.1430614 * (76.6557 - Q.mass) * (0.875672 - Q.D2)
    if Q.sum_zz_dr2 < 0.01165737 and Q.D2 < 0.875672:
        z += 117.258 * (0.01165737 - Q.sum_zz_dr2) * (0.875672 - Q.D2)
    if Q.mass < 76.6557 and Q.mean_eta2 > 0.006395693:
        z += 3.554821 * (76.6557 - Q.mass) * (Q.mean_eta2 - 0.006395693)
    if Q.N2 < 0.2233283 and Q.pt_7 < 25.57812:
        z += -0.5763057 * (0.2233283 - Q.N2) * (25.57812 - Q.pt_7)
    if Q.D2 < 1.002471 and Q.pt_7 < 53.4375:
        z += -0.08362657 * (1.002471 - Q.D2) * (53.4375 - Q.pt_7)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.mean_eta < -0.01790907:
        z += 14921.44 * (0.007639643 - Q.sum_z_dr2_top2) * (-0.01790907 - Q.mean_eta)
    if Q.lam2 < 0.000537286 and Q.phi_0 > -0.0297699:
        z += 6240.568 * (0.000537286 - Q.lam2) * (Q.phi_0 - -0.0297699)
    if Q.sj3_dr_max > 0.169029 and Q.dr01 < 0.02117036:
        z += -117.7019 * (Q.sj3_dr_max - 0.169029) * (0.02117036 - Q.dr01)
    if Q.lam2 < 0.000537286 and Q.mean_phi < -0.01753483:
        z += -79377.69 * (0.000537286 - Q.lam2) * (-0.01753483 - Q.mean_phi)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.mean_phi < -0.01753483:
        z += 8457.958 * (0.007639643 - Q.sum_z_dr2_top2) * (-0.01753483 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.dr_4 < 0.1551147:
        z += -22.88547 * (0.2233283 - Q.N2) * (0.1551147 - Q.dr_4)
    if Q.sj3_dr_max > 0.233678 and Q.pt_0 < 130.375:
        z += 0.1591632 * (Q.sj3_dr_max - 0.233678) * (130.375 - Q.pt_0)
    if Q.sd_mass > 38.43971 and Q.pt_0 < 153.375:
        z += -0.0007376579 * (Q.sd_mass - 38.43971) * (153.375 - Q.pt_0)
    if Q.sj3_dr_max > 0.3456459 and Q.pt_0 < 118.1875:
        z += -0.6756495 * (Q.sj3_dr_max - 0.3456459) * (118.1875 - Q.pt_0)
    if Q.e2 > 0.009668065 and Q.mean_phi > -0.009352575:
        z += -253.0827 * (Q.e2 - 0.009668065) * (Q.mean_phi - -0.009352575)
    if Q.sd_rg > 0.2037854 and Q.pt_0 < 176.25:
        z += 0.3047262 * (Q.sd_rg - 0.2037854) * (176.25 - Q.pt_0)
    if Q.sd_rg > 0.1667615 and Q.z_1st < 0.3331409:
        z += -74.62729 * (Q.sd_rg - 0.1667615) * (0.3331409 - Q.z_1st)
    if Q.D2 < 1.002471 and Q.absphi_5 < 0.1132812:
        z += -6.197477 * (1.002471 - Q.D2) * (0.1132812 - Q.absphi_5)
    if Q.N2 < 0.2233283 and Q.abseta_0 > 0.02487183:
        z += 58.50451 * (0.2233283 - Q.N2) * (Q.abseta_0 - 0.02487183)
    if Q.centroid_offset < 0.01837778 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -134.8193 * (0.01837778 - Q.centroid_offset) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.zdr_0 < 0.03981924 and Q.sj3_dr23 > 0.1414609:
        z += -84.38011 * (0.03981924 - Q.zdr_0) * (Q.sj3_dr23 - 0.1414609)
    if Q.lam2 < 0.000537286 and Q.absphi_6 < 0.1591797:
        z += -4614.522 * (0.000537286 - Q.lam2) * (0.1591797 - Q.absphi_6)
    if Q.tau21_b2 < 0.004811143 and Q.pt_7 < 43.5:
        z += 10.9734 * (0.004811143 - Q.tau21_b2) * (43.5 - Q.pt_7)
    if Q.lam2 < 0.001130645 and Q.absphi_7 > 0.02111816:
        z += 1780.846 * (0.001130645 - Q.lam2) * (Q.absphi_7 - 0.02111816)
    if Q.sum_z_dr2 < 0.002635418 and Q.dr1_7 > 0.1343804:
        z += -5495.653 * (0.002635418 - Q.sum_z_dr2) * (Q.dr1_7 - 0.1343804)
    if Q.tau21 < 0.2838437 and Q.dr1_7 > 0.2411619:
        z += -25.10205 * (0.2838437 - Q.tau21) * (Q.dr1_7 - 0.2411619)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.7931222
    if Q.LHA < 0.2160559:
        z += -4.685163 * Q.LHA + 1.012257
    if Q.z_7 < 0.02807091:
        z += -184.8363 * Q.z_7 + 6.76193
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -48.26614 * Q.z_7 + 2.92828
    if 0.04939969 <= Q.z_7 < 0.0586137:
        z += -32.3809 * Q.z_7 + 2.143554
    if 0.0586137 <= Q.z_7 < 0.07148865:
        z += -19.07496 * Q.z_7 + 1.363643
    if Q.sum_zz_dr2 < 0.005284669:
        z += -187.7388 * Q.sum_zz_dr2 + 0.9921375
    if Q.pt_7 < 23.21641:
        z += 0.09149841 * Q.pt_7 - 3.643141
    if 23.21641 <= Q.pt_7 < 37.15625:
        z += 0.03857633 * Q.pt_7 - 2.41448
    if 37.15625 <= Q.pt_7 < 43.5:
        z += -0.01455313 * Q.pt_7 - 0.4403888
    if 43.5 <= Q.pt_7 < 53.4375:
        z += -0.05292208 * Q.pt_7 + 1.22866
    if Q.pt_7 >= 53.4375:
        z += -0.000671532 * Q.pt_7 - 1.563478
    if Q.zdr_0 < 0.02383244:
        z += 18.81326 * Q.zdr_0 - 0.448366
    if Q.sj3_dr_max < 0.1070199:
        z += 6.535177 * Q.sj3_dr_max - 0.7828362
    if 0.1070199 <= Q.sj3_dr_max < 0.1879486:
        z += -4.538851 * Q.sj3_dr_max + 0.4023056
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 3.980156 * Q.sj3_dr_max - 1.198829
    if Q.sum_z_dr2 < 9.121392e-05:
        z += -5249.38 * Q.sum_z_dr2 + 1.985567
    if 9.121392e-05 <= Q.sum_z_dr2 < 0.002635418:
        z += -592.2288 * Q.sum_z_dr2 + 1.56077
    if 6.267538 <= Q.log_sum_pt < 6.701242:
        z += 3.654328 * Q.log_sum_pt - 22.90364
    if 6.701242 <= Q.log_sum_pt < 6.766778:
        z += -1.851216 * Q.log_sum_pt + 13.99034
    if Q.log_sum_pt >= 6.766778:
        z += -4.698965 * Q.log_sum_pt + 33.26043
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += 200.0321 * Q.sum_z_dr2_top3 - 0.5831999
    if Q.mass_over_sum_pt < 0.09041383:
        z += 17.31448 * Q.mass_over_sum_pt - 1.565469
    if Q.tau1 < 0.09538712:
        z += -18.52034 * Q.tau1 + 1.766602
    if Q.sum_pt_top5 >= 716.8828:
        z += 0.007196619 * Q.sum_pt_top5 - 5.159133
    if Q.z_6 < 0.04737278:
        z += -26.19398 * Q.z_6 + 1.240882
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 58.42053 * Q.mass_over_sum_pt_sq - 0.6812363
    if Q.max_pair_mass >= 21.83614:
        z += -0.01404622 * Q.max_pair_mass + 0.3067152
    if Q.C3 < 0.009518026:
        z += 57.83474 * Q.C3 - 0.5504725
    if Q.z_7 < 0.04939969 and Q.mass_top5 < 68.43422:
        z += 0.4086209 * (0.04939969 - Q.z_7) * (68.43422 - Q.mass_top5)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -35.41285 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.sum_zz_dr2 < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += 16910.78 * (0.005284669 - Q.sum_zz_dr2) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -648.3663 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.C3 < 0.03532852:
        z += 151.8834 * (0.07148865 - Q.z_7) * (0.03532852 - Q.C3)
    if Q.sum_z_dr2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 58664.01 * (0.001653836 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.C2_b2 > 0.001109927:
        z += -400.472 * (0.07148865 - Q.z_7) * (Q.C2_b2 - 0.001109927)
    if Q.mass < 60.63098 and Q.D2 < 3.885568:
        z += 0.003629277 * (60.63098 - Q.mass) * (3.885568 - Q.D2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 < 0.0002302115:
        z += 14034.58 * (Q.log_sum_pt - 6.701242) * (0.0002302115 - Q.mean_phi2)
    if Q.sum_z_dr2 < 0.001653836 and Q.absphi_0 < 0.01452637:
        z += -15049.31 * (0.001653836 - Q.sum_z_dr2) * (0.01452637 - Q.absphi_0)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi2 < 0.0002302115:
        z += -22764.77 * (Q.log_sum_pt - 6.896095) * (0.0002302115 - Q.mean_phi2)
    if Q.tau1 < 0.09538712 and Q.planar_flow < 0.4926918:
        z += 7.763209 * (0.09538712 - Q.tau1) * (0.4926918 - Q.planar_flow)
    if Q.sum_z_dr2_top3 < 0.002915531 and Q.tau3 < 0.006646257:
        z += -17326.05 * (0.002915531 - Q.sum_z_dr2_top3) * (0.006646257 - Q.tau3)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 85484.25 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 0.00424745:
        z += -1280.707 * (Q.log_sum_pt - 6.701242) * (0.00424745 - Q.mean_eta2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 > 0.0007823696:
        z += 439.3115 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi2 - 0.0007823696)
    if Q.zdr_0 < 0.02383244 and Q.abseta_0 < 0.02090454:
        z += -675.6279 * (0.02383244 - Q.zdr_0) * (0.02090454 - Q.abseta_0)
    if Q.z_5 < 0.02818362 and Q.abseta_7 < 0.1626038:
        z += 414.3661 * (0.02818362 - Q.z_5) * (0.1626038 - Q.abseta_7)
    if Q.sum_pt_top5 > 716.8828 and Q.mean_eta2 < 9.030369e-05:
        z += -65.68893 * (Q.sum_pt_top5 - 716.8828) * (9.030369e-05 - Q.mean_eta2)
    if Q.z_6 < 0.02160287 and Q.abseta_6 < 0.1583252:
        z += 754.2559 * (0.02160287 - Q.z_6) * (0.1583252 - Q.abseta_6)
    if Q.pt_7 > 23.21641 and Q.absphi_1 < 0.07141113:
        z += -0.2274178 * (Q.pt_7 - 23.21641) * (0.07141113 - Q.absphi_1)
    if Q.max_pair_mass > 21.83614 and Q.dr_max_012 < 0.2366102:
        z += -0.450048 * (Q.max_pair_mass - 21.83614) * (0.2366102 - Q.dr_max_012)
    if Q.log_sum_pt > 6.701242 and Q.D2_b2 < 4.721224:
        z += -0.6318052 * (Q.log_sum_pt - 6.701242) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.896095 and Q.dr_max_012 < 0.03086731:
        z += -245.0883 * (Q.log_sum_pt - 6.896095) * (0.03086731 - Q.dr_max_012)
    if Q.log_sum_pt > 6.701242 and Q.dr_max_012 < 0.06112084:
        z += 78.02168 * (Q.log_sum_pt - 6.701242) * (0.06112084 - Q.dr_max_012)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi > 0.0006973656:
        z += -263.3971 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi - 0.0006973656)
    if Q.log_sum_pt > 6.267538 and Q.C3 < 0.0113471:
        z += 169.2387 * (Q.log_sum_pt - 6.267538) * (0.0113471 - Q.C3)
    if Q.log_sum_pt > 6.766778 and Q.C3 < 0.05481757:
        z += -56.27705 * (Q.log_sum_pt - 6.766778) * (0.05481757 - Q.C3)
    if Q.log_sum_pt > 6.701242 and Q.zdr_3 < 0.001736329:
        z += 2634.486 * (Q.log_sum_pt - 6.701242) * (0.001736329 - Q.zdr_3)
    if Q.log_sum_pt > 6.896095 and Q.zdr_3 < 0.002727284:
        z += -3547.656 * (Q.log_sum_pt - 6.896095) * (0.002727284 - Q.zdr_3)
    if Q.log_sum_pt > 6.267538 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -0.3722537 * (Q.log_sum_pt - 6.267538) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.z_6 < 0.04737278 and Q.zdr_3 < 0.001736329:
        z += -9505.853 * (0.04737278 - Q.z_6) * (0.001736329 - Q.zdr_3)
    if Q.pt_7 < 37.15625 and Q.zdr_3 < 0.01257746:
        z += 1.674579 * (37.15625 - Q.pt_7) * (0.01257746 - Q.zdr_3)
    if Q.log_sum_pt > 6.267538 and Q.zdr_0 < 0.03981924:
        z += -27.44144 * (Q.log_sum_pt - 6.267538) * (0.03981924 - Q.zdr_0)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.9727805
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 21.87157 * Q.centroid_offset - 0.1769926
    if Q.centroid_offset >= 0.01837778:
        z += 7.659673 * Q.centroid_offset + 0.0841905
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 411.904 * Q.lam1_plus_lam2 - 5.453063
    if Q.mass < 45.7571:
        z += 0.008804372 * Q.mass - 0.1534812
    if 45.7571 <= Q.mass < 60.63098:
        z += -0.01676639 * Q.mass + 1.016562
    if Q.pt_6 < 19.46875:
        z += 0.03877533 * Q.pt_6 - 3.526014
    if 19.46875 <= Q.pt_6 < 39.75:
        z += 0.1308056 * Q.pt_6 - 5.317729
    if 39.75 <= Q.pt_6 < 41.21875:
        z += 0.08048 * Q.pt_6 - 3.317285
    if Q.lam2 < 0.001130645:
        z += 260.2691 * Q.lam2 + 0.814224
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -486.6639 * Q.lam2 + 1.65874
    if Q.tau1 < 0.1136369:
        z += -13.39559 * Q.tau1 + 1.522234
    if Q.lam1 < 0.00595415:
        z += -639.0984 * Q.lam1 + 6.118598
    if 0.00595415 <= Q.lam1 < 0.00733008:
        z += -747.1303 * Q.lam1 + 6.761836
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -275.0124 * Q.lam1 + 3.301174
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.02763958 * Q.sj3_pair_mass_min + 0.1244258
    if Q.sum_pt < 488.9312:
        z += -0.007940449 * Q.sum_pt + 3.904608
    if 488.9312 <= Q.sum_pt < 615.875:
        z += -0.001013725 * Q.sum_pt + 0.5179159
    if 615.875 <= Q.sum_pt < 763.825:
        z += 0.0007192416 * Q.sum_pt - 0.5493747
    if Q.sum_zz_dr2 < 0.0030133:
        z += -241.2696 * Q.sum_zz_dr2 + 0.7270178
    if Q.eccentricity >= 0.927072:
        z += 6.485684 * Q.eccentricity - 6.012696
    if Q.sj3_dr_max < 0.1789613:
        z += -15.43339 * Q.sj3_dr_max + 2.15906
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 4.932261 * Q.sj3_dr_max - 1.485605
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 7.48568 * Q.sj3_dr_max - 1.965516
    if Q.sj3_dr_max >= 0.3012016:
        z += 2.55342 * Q.sj3_dr_max - 0.4799115
    if Q.max_dr < 0.1452311:
        z += 17.03249 * Q.max_dr - 2.473648
    if Q.mean_eta >= 0.02644207:
        z += -21.1665 * Q.mean_eta + 0.5596861
    if Q.sum_z_dr2 < 0.008678045:
        z += 857.9358 * Q.sum_z_dr2 - 7.445205
    if Q.C2_b2 < 0.02415398:
        z += -28.77768 * Q.C2_b2 + 0.6950957
    if Q.sj2_dr < 0.1872617:
        z += -0.5694339 * Q.sj2_dr + 0.1066332
    if Q.sj3_dr_min >= 0.02400746:
        z += -4.854053 * Q.sj3_dr_min + 0.1165334
    if Q.z_6 < 0.03932388:
        z += -64.88014 * Q.z_6 + 2.551339
    if Q.mass_over_sum_pt < 0.09041383:
        z += -45.43151 * Q.mass_over_sum_pt + 4.107636
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += 16.174 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 3088.823 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -482.5383 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -2.815523 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_eta < -0.02665591:
        z += 8153.999 * (0.01323868 - Q.lam1_plus_lam2) * (-0.02665591 - Q.mean_eta)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_phi < -0.02594505:
        z += 6093.811 * (0.01323868 - Q.lam1_plus_lam2) * (-0.02594505 - Q.mean_phi)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.mean_phi > 0.02612796:
        z += 5826.556 * (0.01323868 - Q.lam1_plus_lam2) * (Q.mean_phi - 0.02612796)
    if Q.pt_6 < 29.90625 and Q.n_pt_above_10 < 8.0:
        z += 0.01458114 * (29.90625 - Q.pt_6) * (8.0 - Q.n_pt_above_10)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.5464226 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.00165835 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 36.84614 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += 0.1042189 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 9219.7 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.009041639 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.centroid_offset > 0.01837778 and Q.sum_pt_top5 > 658.125:
        z += 0.3416112 * (Q.centroid_offset - 0.01837778) * (Q.sum_pt_top5 - 658.125)
    if Q.pt_6 < 41.21875 and Q.sum_pt < 988.4078:
        z += 0.0006088163 * (41.21875 - Q.pt_6) * (988.4078 - Q.sum_pt)
    if Q.sum_pt < 763.825 and Q.z_7 < 0.0753896:
        z += 0.1009984 * (763.825 - Q.sum_pt) * (0.0753896 - Q.z_7)
    return max(0.0, z)


def neuron_7(Q):
    z = 10.85797
    if Q.sum_z_dr2_top2 < 8.10414e-05:
        z += 6889.959 * Q.sum_z_dr2_top2 - 0.1308634
    if 8.10414e-05 <= Q.sum_z_dr2_top2 < 0.001056655:
        z += -438.1944 * Q.sum_z_dr2_top2 + 0.4630204
    if Q.sum_z_dr2 < 0.0009641429:
        z += 1456.501 * Q.sum_z_dr2 - 2.452483
    if 0.0009641429 <= Q.sum_z_dr2 < 0.001653836:
        z += 403.3947 * Q.sum_z_dr2 - 1.437139
    if 0.001653836 <= Q.sum_z_dr2 < 0.003562611:
        z += 186.2812 * Q.sum_z_dr2 - 1.078068
    if 0.003562611 <= Q.sum_z_dr2 < 0.004372139:
        z += -217.1135 * Q.sum_z_dr2 + 0.3590702
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += -765.1339 * Q.sum_z_dr2 + 2.755092
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -2422.371 * Q.sum_z_dr2 + 15.21766
    if Q.sum_z_dr2 >= 0.01323868:
        z += -3193.596 * Q.sum_z_dr2 + 25.42767
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -56.35722 * Q.mass_over_sum_pt + 0.6255562
    if 0.07269073 <= Q.mass_over_sum_pt < 0.07992374:
        z += 90.3161 * Q.mass_over_sum_pt - 10.03624
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 204.6498 * Q.mass_over_sum_pt - 19.17421
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 460.7329 * Q.mass_over_sum_pt - 40.87766
    if 0.09041383 <= Q.mass_over_sum_pt < 0.1079857:
        z += 436.7621 * Q.mass_over_sum_pt - 38.71037
    if Q.mass_over_sum_pt >= 0.1079857:
        z += 467.5919 * Q.mass_over_sum_pt - 42.03955
    if Q.tau1 < 0.05356915:
        z += -42.44412 * Q.tau1 + 2.273696
    if Q.sum_z_dr < 0.04081947:
        z += 104.484 * Q.sum_z_dr - 8.208781
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 84.96448 * Q.sum_z_dr - 7.412005
    if 36.22941 <= Q.mass < 69.61135:
        z += 0.0320989 * Q.mass - 1.162924
    if 69.61135 <= Q.mass < 76.6557:
        z += 0.05110554 * Q.mass - 2.486002
    if Q.mass >= 76.6557:
        z += -0.1077253 * Q.mass + 9.689291
    if Q.lam1_plus_lam2 < 0.005590289:
        z += 912.0907 * Q.lam1_plus_lam2 - 6.092284
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.006679471:
        z += -43.67883 * Q.lam1_plus_lam2 - 0.7492561
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -955.7695 * Q.lam1_plus_lam2 + 5.343028
    if Q.lam1_plus_lam2 >= 0.008678045:
        z += -1288.841 * Q.lam1_plus_lam2 + 8.233436
    if Q.centroid_offset < 0.02076709:
        z += 31.84728 * Q.centroid_offset - 0.6613755
    if Q.centroid_offset >= 0.03776099:
        z += -113.4226 * Q.centroid_offset + 4.282949
    if Q.z_7 >= 0.03243272:
        z += 10.37493 * Q.z_7 - 0.3364872
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += 32.2648 * Q.e2 - 1.147366
    if Q.e2 >= 0.05028464:
        z += 189.0135 * Q.e2 - 9.029416
    if Q.sj2_dr < 0.1294903:
        z += -6.69826 * Q.sj2_dr + 0.4905047
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -8.071917 * Q.sj2_dr + 0.6683799
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 21.94476 * Q.sj2_dr - 4.109414
    if Q.LHA < 0.3033137:
        z += -17.72846 * Q.LHA + 5.377287
    if Q.pt_7 < 29.04219:
        z += 0.03575956 * Q.pt_7 - 1.038536
    if 0.2037854 <= Q.sd_rg < 0.2787955:
        z += -18.97628 * Q.sd_rg + 3.867088
    if Q.sd_rg >= 0.2787955:
        z += 18.26389 * Q.sd_rg - 6.515305
    if Q.lam1 < 0.008375572:
        z += 153.7964 * Q.lam1 - 1.288133
    if Q.mass_top5 >= 53.60766:
        z += 0.02711758 * Q.mass_top5 - 1.45371
    if Q.D2_b2 < 0.05744392:
        z += 6.405673 * Q.D2_b2 - 0.3679669
    if 0.04889979 <= Q.sj3_dr_max < 0.1426152:
        z += -8.640085 * Q.sj3_dr_max + 0.4224983
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += 11.42691 * Q.sj3_dr_max - 2.439359
    if 0.213399 <= Q.sj3_dr_max < 0.2623172:
        z += 8.043074 * Q.sj3_dr_max - 1.717253
    if Q.sj3_dr_max >= 0.2623172:
        z += -8.721339 * Q.sj3_dr_max + 2.680342
    if Q.lam2 < 0.0003061234:
        z += 1739.091 * Q.lam2 - 0.5323764
    if Q.D2 < 2.357246:
        z += -0.1718656 * Q.D2 + 0.4051296
    if Q.max_dr < 0.1117619:
        z += 7.751815 * Q.max_dr - 0.5515458
    if 0.1117619 <= Q.max_dr < 0.177305:
        z += -4.803119 * Q.max_dr + 0.8516169
    if Q.tau21_b2 < 0.04019753:
        z += 24.53525 * Q.tau21_b2 - 0.9862564
    if Q.planar_flow < 0.1950135 and Q.lam1_plus_lam2 > 0.007520088:
        z += -1750.549 * (0.1950135 - Q.planar_flow) * (Q.lam1_plus_lam2 - 0.007520088)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1180153 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -12.59916 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.tau1 < 0.05356915 and Q.n_dr_0p05_0p1 > 5.0:
        z += -21.91801 * (0.05356915 - Q.tau1) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.1102669 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.sum_z_dr < 0.08723651 and Q.mean_phi < 0.004406178:
        z += -109.4336 * (0.08723651 - Q.sum_z_dr) * (0.004406178 - Q.mean_phi)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 1703.36 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += -11505.18 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.centroid_offset > 0.03776099 and Q.pt_4 > 81.375:
        z += -23.69233 * (Q.centroid_offset - 0.03776099) * (Q.pt_4 - 81.375)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -16538.11 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.sum_z_dr < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 3693.062 * (0.08723651 - Q.sum_z_dr) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 1481.55 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    if Q.sum_z_dr2_top2 < 0.001056655 and Q.tau21_b2 < 0.02656143:
        z += -55238.77 * (0.001056655 - Q.sum_z_dr2_top2) * (0.02656143 - Q.tau21_b2)
    if Q.pt_7 < 29.04219 and Q.tau21 < 0.2838437:
        z += -0.1815932 * (29.04219 - Q.pt_7) * (0.2838437 - Q.tau21)
    if Q.lam2 < 0.0003061234 and Q.tau21_b2 < 0.04019753:
        z += 104254.5 * (0.0003061234 - Q.lam2) * (0.04019753 - Q.tau21_b2)
    if Q.mass_over_sum_pt > 0.09041383 and Q.pt_6 < 36.8125:
        z += -5.281644 * (Q.mass_over_sum_pt - 0.09041383) * (36.8125 - Q.pt_6)
    if Q.sum_z_dr2 > 0.01323868 and Q.pt_6 < 38.25:
        z += 13.29795 * (Q.sum_z_dr2 - 0.01323868) * (38.25 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.01109984 and Q.pt_6 < 35.28125:
        z += -0.6959673 * (Q.mass_over_sum_pt - 0.01109984) * (35.28125 - Q.pt_6)
    if Q.lam1_plus_lam2 > 0.008678045 and Q.pt_6 < 38.25:
        z += 48.42539 * (Q.lam1_plus_lam2 - 0.008678045) * (38.25 - Q.pt_6)
    if Q.mass > 76.6557 and Q.zdr_6 > 0.006366792:
        z += -38.26371 * (Q.mass - 76.6557) * (Q.zdr_6 - 0.006366792)
    if Q.lam1_plus_lam2 < 0.006679471 and Q.mean_phi < 0.006452173:
        z += 3421.439 * (0.006679471 - Q.lam1_plus_lam2) * (0.006452173 - Q.mean_phi)
    if Q.sd_rg > 0.2037854 and Q.dr_6 < 0.06970457:
        z += 423.5021 * (Q.sd_rg - 0.2037854) * (0.06970457 - Q.dr_6)
    if Q.sum_z_dr2 > 0.01323868 and Q.dr_6 < 0.06970457:
        z += 28206.46 * (Q.sum_z_dr2 - 0.01323868) * (0.06970457 - Q.dr_6)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.2444994
    if Q.sum_z_dr2 < 0.005019719:
        z += -234.1617 * Q.sum_z_dr2 + 1.175426
    if Q.LHA < 0.1967397:
        z += 9.282332 * Q.LHA - 1.826203
    if Q.log_sum_pt >= 6.701242:
        z += -8.137055 * Q.log_sum_pt + 54.52838
    if Q.mass < 21.78408:
        z += 0.008585736 * Q.mass - 0.1246012
    if 21.78408 <= Q.mass < 29.6447:
        z += -0.007942261 * Q.mass + 0.2354459
    if Q.sum_z_dr < 0.06108601:
        z += 70.75442 * Q.sum_z_dr - 4.322105
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.003096091 * Q.sum_pt_top5 - 2.128369
    if Q.lam2 < 0.0001330621:
        z += 3222.632 * Q.lam2 - 0.42881
    if Q.lam1_plus_lam2 < 0.002635418:
        z += -1135.446 * Q.lam1_plus_lam2 + 3.86241
    if 0.002635418 <= Q.lam1_plus_lam2 < 0.003562611:
        z += -938.3528 * Q.lam1_plus_lam2 + 3.342986
    if Q.z_6 < 0.03448406:
        z += 25.31521 * Q.z_6 - 0.8729712
    if Q.mass_over_sum_pt < 0.03319429:
        z += 37.50333 * Q.mass_over_sum_pt - 1.244896
    if Q.sj3_dr_max < 0.1070199:
        z += 8.84759 * Q.sj3_dr_max - 1.48815
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 13.09023 * Q.sj3_dr_max - 1.942197
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += 1.344926 * Q.sj3_dr_max - 0.267139
    if Q.e2 < 0.01289969:
        z += -78.40777 * Q.e2 + 1.011436
    if Q.sum_z_dr2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -21.32211 * (0.006679471 - Q.sum_z_dr2) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 13081.6 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -7.863931 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 956.4471 * (0.006679471 - Q.sum_z_dr2) * (Q.phi_0 - -0.04013062)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 4479.024 * (0.005019719 - Q.sum_z_dr2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -160.7337 * (0.06108601 - Q.sum_z_dr) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.1427525 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_z_dr < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 101910.2 * (0.06108601 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 986.6539 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.701242 and Q.e3 < 2.955458e-05:
        z += 70175.8 * (Q.log_sum_pt - 6.701242) * (2.955458e-05 - Q.e3)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -523.0547 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 16630.58 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.mass < 49.6681 and Q.lam2 < 0.0001330621:
        z += 67.59834 * (49.6681 - Q.mass) * (0.0001330621 - Q.lam2)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += 0.001103773 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.06618646 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.mass < 21.78408 and Q.D2_b2 < 0.716559:
        z += 0.1716124 * (21.78408 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.001368413 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -25292.11 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.09455471 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.sum_z_dr < 0.06108601 and Q.lam1_plus_lam2 > 0.0003193707:
        z += 9497.973 * (0.06108601 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.0003193707)
    if Q.log_sum_pt > 6.701242 and Q.lam1_plus_lam2 < 0.008678045:
        z += -97.40591 * (Q.log_sum_pt - 6.701242) * (0.008678045 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2 < 0.006679471:
        z += 3107.94 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.sum_z_dr2)
    if Q.tau1 < 0.05356915 and Q.lam1_plus_lam2 < 0.003562611:
        z += 21252.51 * (0.05356915 - Q.tau1) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.log_sum_pt > 6.701242 and Q.centroid_offset < 0.02355416:
        z += -108.0625 * (Q.log_sum_pt - 6.701242) * (0.02355416 - Q.centroid_offset)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2 < 0.0005611231:
        z += 3536.797 * (Q.log_sum_pt - 6.701242) * (0.0005611231 - Q.sum_z_dr2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top5 < 0.005691733:
        z += -1195.271 * (0.1986272 - Q.sj3_dr_max) * (0.005691733 - Q.sum_z_dr2_top5)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0001947983:
        z += 15677.83 * (0.1986272 - Q.sj3_dr_max) * (0.0001947983 - Q.lam2)
    if Q.lam1_plus_lam2 < 0.003562611 and Q.centroid_offset > 0.006789738:
        z += -1426.224 * (0.003562611 - Q.lam1_plus_lam2) * (Q.centroid_offset - 0.006789738)
    if Q.mass < 29.6447 and Q.centroid_offset < 0.02685622:
        z += -3.430136 * (29.6447 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.03319429 and Q.centroid_offset < 0.02685622:
        z += 3963.368 * (0.03319429 - Q.mass_over_sum_pt) * (0.02685622 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1986272 and Q.lam1_plus_lam2 < 0.003562611:
        z += -2331.706 * (0.1986272 - Q.sj3_dr_max) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.sum_z_dr < 0.06108601 and Q.sum_z_dr2 < 0.003562611:
        z += -30554.72 * (0.06108601 - Q.sum_z_dr) * (0.003562611 - Q.sum_z_dr2)
    if Q.sj3_dr_max < 0.1426152 and Q.centroid_offset > 0.01627885:
        z += -1095.616 * (0.1426152 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.e2 < 0.01289969 and Q.psi_0p2 > 0.79448:
        z += -577.5894 * (0.01289969 - Q.e2) * (Q.psi_0p2 - 0.79448)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.088005
    if Q.sum_z_dr < 0.05464922:
        z += 50.16238 * Q.sum_z_dr - 2.741335
    if Q.sum_z_dr >= 0.0717028:
        z += -19.32658 * Q.sum_z_dr + 1.38577
    if Q.tau1 < 0.04369778:
        z += -51.93411 * Q.tau1 + 2.269405
    if Q.mass < 29.6447:
        z += 0.1088553 * Q.mass - 4.279026
    if 29.6447 <= Q.mass < 45.7571:
        z += 0.04441305 * Q.mass - 2.368654
    if 45.7571 <= Q.mass < 53.33237:
        z += 0.05295112 * Q.mass - 2.759331
    if Q.mass >= 53.33237:
        z += 0.008538065 * Q.mass - 0.3906771
    if Q.e3 < 2.371297e-05:
        z += 46720.87 * Q.e3 - 1.107891
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -4868.671 * Q.e3 + 0.3966869
    if Q.e3 >= 0.0001869378:
        z += 139.5923 * Q.e3 - 0.5395471
    if Q.sum_z_dr2 < 0.003562611:
        z += -3579.391 * Q.sum_z_dr2 + 16.77666
    if 0.003562611 <= Q.sum_z_dr2 < 0.005019719:
        z += -1766.396 * Q.sum_z_dr2 + 10.31767
    if 0.005019719 <= Q.sum_z_dr2 < 0.00609665:
        z += -1347.212 * Q.sum_z_dr2 + 8.213478
    if Q.sj3_dr_max < 0.1070199:
        z += 12.81178 * Q.sj3_dr_max - 0.8039745
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 23.14628 * Q.sj3_dr_max - 1.909972
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -19.65193 * Q.sj3_dr_max + 4.193702
    if Q.lam2 < 0.0003061234:
        z += -1909.154 * Q.lam2 + 0.5844365
    if Q.lam2 >= 0.001130645:
        z += 341.4131 * Q.lam2 - 0.3860169
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.6495294 * Q.n_dr_0p2_0p4 - 0.6495294
    if Q.lam1 < 0.003377388:
        z += 522.8401 * Q.lam1 - 3.113068
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 347.9272 * Q.lam1 - 2.52232
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 830.0761 * Q.lam1 - 5.142008
    if Q.lam1 >= 0.00595415:
        z += 307.236 * Q.lam1 - 2.02894
    if Q.centroid_offset < 0.002316125:
        z += 138.5332 * Q.centroid_offset + 2.157584
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -154.3082 * Q.centroid_offset + 2.835842
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -34.04639 * Q.e2 + 1.210721
    if Q.e2 >= 0.04447357:
        z += 15.25393 * Q.e2 - 0.9818402
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.01390128 * Q.sj3_pair_mass_max - 0.3368297
    if Q.lam1_plus_lam2 < 0.0001721983:
        z += -6657.077 * Q.lam1_plus_lam2 + 1.146338
    if Q.C3 < 0.0284695:
        z += 15.17171 * Q.C3 - 0.431931
    if Q.zdr_0 < 0.006292091:
        z += 61.99852 * Q.zdr_0 - 0.3901004
    if Q.mass_over_sum_pt < 0.07269073:
        z += 105.833 * Q.mass_over_sum_pt - 7.693078
    if Q.sum_pt < 559.6875:
        z += -0.003174627 * Q.sum_pt + 3.862733
    if 559.6875 <= Q.sum_pt < 988.4078:
        z += -0.004865489 * Q.sum_pt + 4.809088
    z += -3.334157 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 9.643979 * Q.log_sum_pt - 66.5058
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.005417429 * Q.sum_pt_top5 + 4.550395
    if Q.pt_4 < 31.125:
        z += -0.04012042 * Q.pt_4 + 1.248748
    if Q.n_for_90pct < 7.0:
        z += 0.2606602 * Q.n_for_90pct - 1.824621
    if Q.max_dr < 0.1117619:
        z += -12.63454 * Q.max_dr + 1.41206
    if Q.M3 < 0.0782171:
        z += -6.516365 * Q.M3 + 0.5096912
    if Q.absphi_1 < 0.02227783:
        z += 13.80679 * Q.absphi_1 - 0.3075853
    if Q.sum_z_dr2_top5 >= 0.01148293:
        z += -29.55078 * Q.sum_z_dr2_top5 + 0.3393294
    if Q.N2 >= 0.2233283:
        z += 2.00677 * Q.N2 - 0.4481685
    if Q.z_5 < 0.06503035:
        z += -9.495387 * Q.z_5 + 0.6174883
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -414.6188 * Q.mass_over_sum_pt_sq + 2.978139
    if Q.zdr_2 < 0.002170915:
        z += -190.2237 * Q.zdr_2 + 0.4129597
    if Q.D2 >= 2.055451:
        z += -0.1814303 * Q.D2 + 0.3729212
    if Q.sum_zz_dr2 < 0.0030133:
        z += 715.6833 * Q.sum_zz_dr2 - 2.156569
    if Q.sd_mass >= 44.82259:
        z += -0.0131237 * Q.sd_mass + 0.5882382
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 5.416973 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.07651296 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 734.5146 * (0.00609665 - Q.sum_z_dr2) * (0.3220738 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -7929.914 * (0.00609665 - Q.sum_z_dr2) * (0.002834884 - Q.mean_phi)
    if Q.mass < 53.33237 and Q.D2 > 2.843757:
        z += 0.00247338 * (53.33237 - Q.mass) * (Q.D2 - 2.843757)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -462.9616 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -350.9191 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.e3 < 2.371297e-05 and Q.eccentricity > 0.8319502:
        z += 91521.09 * (2.371297e-05 - Q.e3) * (Q.eccentricity - 0.8319502)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += -235.3634 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -18.22851 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -29.21184 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.3297204 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -1.698183 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 928.325 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -2.382599 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.sum_z_dr < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += 2223.56 * (0.05464922 - Q.sum_z_dr) * (0.02656143 - Q.tau21_b2)
    if Q.tau1 < 0.04369778 and Q.tau21_b2 < 0.02656143:
        z += -2611.968 * (0.04369778 - Q.tau1) * (0.02656143 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.07269073 and Q.mass_top2 > 1.933087:
        z += -0.9755488 * (0.07269073 - Q.mass_over_sum_pt) * (Q.mass_top2 - 1.933087)
    if Q.mass_over_sum_pt < 0.07269073 and Q.phi_0 < -0.004917145:
        z += 375.9381 * (0.07269073 - Q.mass_over_sum_pt) * (-0.004917145 - Q.phi_0)
    if Q.e3 > 0.0001869378 and Q.pt_6 < 33.6875:
        z += -191.9183 * (Q.e3 - 0.0001869378) * (33.6875 - Q.pt_6)
    if Q.sum_z_dr < 0.05464922 and Q.z_6 > 0.03448406:
        z += -669.4676 * (0.05464922 - Q.sum_z_dr) * (Q.z_6 - 0.03448406)
    if Q.centroid_offset < 0.01837778 and Q.pair_mass_0_2 > 17.9037:
        z += -1.676469 * (0.01837778 - Q.centroid_offset) * (Q.pair_mass_0_2 - 17.9037)
    if Q.sum_pt < 988.4078 and Q.dr_2 < 0.01778111:
        z += -0.1561195 * (988.4078 - Q.sum_pt) * (0.01778111 - Q.dr_2)
    if Q.n_for_90pct < 7.0 and Q.dr_2 > 0.00656258:
        z += 1.40513 * (7.0 - Q.n_for_90pct) * (Q.dr_2 - 0.00656258)
    if Q.e2 > 0.04447357 and Q.sj3_mass3 < 0.2723288:
        z += -30.44133 * (Q.e2 - 0.04447357) * (0.2723288 - Q.sj3_mass3)
    if Q.sd_mass > 44.82259 and Q.n_dr_0p05_0p1 > 7.0:
        z += -0.03148055 * (Q.sd_mass - 44.82259) * (Q.n_dr_0p05_0p1 - 7.0)
    if Q.mass < 53.33237 and Q.dr1_7 > 0.2025074:
        z += -0.4057193 * (53.33237 - Q.mass) * (Q.dr1_7 - 0.2025074)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.404275
    z += -6.78373 * Q.e2
    if 11.051 <= Q.sj3_pair_mass_min < 15.95929:
        z += 0.03949173 * Q.sj3_pair_mass_min - 0.4364232
    if Q.sj3_pair_mass_min >= 15.95929:
        z += 0.1140811 * Q.sj3_pair_mass_min - 1.626816
    if Q.lam1 < 0.001503553:
        z += 1076.143 * Q.lam1 - 3.837469
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 585.8529 * Q.lam1 - 3.100292
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 366.706 * Q.lam1 - 2.183423
    if Q.lam1 >= 0.00733008:
        z += -243.7994 * Q.lam1 + 1.787069
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 965.2499 * Q.lam2 - 0.2954855
    if Q.lam2 >= 0.003408389:
        z += 477.8082 * Q.lam2 + 1.365905
    if Q.sj3_dr_min >= 0.1278212:
        z += 12.51483 * Q.sj3_dr_min - 1.59966
    if Q.e3 < 3.892127e-05:
        z += -3604.21 * Q.e3 + 0.2936618
    if 3.892127e-05 <= Q.e3 < 8.147744e-05:
        z += -5443.012 * Q.e3 + 0.3652303
    if Q.e3 >= 8.147744e-05:
        z += -1838.803 * Q.e3 + 0.07156853
    if 22.18342 <= Q.mass_top5 < 45.32077:
        z += 0.005815002 * Q.mass_top5 - 0.1289966
    if Q.mass_top5 >= 45.32077:
        z += -0.01988244 * Q.mass_top5 + 1.035631
    if Q.M2 < 0.02563286:
        z += 29.34657 * Q.M2 - 0.7522366
    z += 4.428727 * Q.mean_phi
    if Q.sum_pt >= 988.4078:
        z += 0.00339564 * Q.sum_pt - 3.356277
    if Q.z_7 < 0.06473447:
        z += 16.79915 * Q.z_7 - 1.087484
    if Q.LHA >= 0.3033137:
        z += -19.70331 * Q.LHA + 5.976285
    if Q.tau1 >= 0.05356915:
        z += 7.828172 * Q.tau1 - 0.4193486
    if Q.sum_z_dr2 < 0.002635418:
        z += 428.8141 * Q.sum_z_dr2 - 1.130104
    if 0.007520088 <= Q.sum_z_dr2 < 0.02530566:
        z += 502.438 * Q.sum_z_dr2 - 3.778378
    if Q.sum_z_dr2 >= 0.02530566:
        z += 425.4211 * Q.sum_z_dr2 - 1.829414
    if Q.mass < 76.6557:
        z += -0.01618389 * Q.mass + 1.240587
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -273.3032 * Q.sum_z_dr2_top3 + 0.5880304
    if Q.sj2_dr >= 0.3003793:
        z += 5.983833 * Q.sj2_dr - 1.79742
    if Q.sj3_dr_max >= 0.1986272:
        z += -5.023029 * Q.sj3_dr_max + 0.9977102
    if Q.C2_b2 >= 0.009032972:
        z += 36.98113 * Q.C2_b2 - 0.3340495
    if Q.n_pt_above_50 >= 6.0:
        z += -0.1298751 * Q.n_pt_above_50 + 0.7792507
    if Q.zdr_0 < 0.004918231:
        z += 76.51497 * Q.zdr_0 - 0.3763183
    if Q.sj3_pair_mass_min > 11.051 and Q.n_dr_0p2_0p4 > 0.0:
        z += 0.008066898 * (Q.sj3_pair_mass_min - 11.051) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.05534367 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.164371 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -456.7743 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -17.05335 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 121.1628 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.lam1 > 0.00733008 and Q.sj3_mass3 < 0.2723288:
        z += -101.0991 * (Q.lam1 - 0.00733008) * (0.2723288 - Q.sj3_mass3)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += -52221.55 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -306.2279 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.sj3_pair_mass_min > 11.051 and Q.sj3_pairmin_over_m > 0.28737:
        z += -0.2512711 * (Q.sj3_pair_mass_min - 11.051) * (Q.sj3_pairmin_over_m - 0.28737)
    if Q.sj3_dr_min > 0.1278212 and Q.sj3_mass2 < 0.5973755:
        z += -12.33319 * (Q.sj3_dr_min - 0.1278212) * (0.5973755 - Q.sj3_mass2)
    if Q.lam2 > 0.0003061234 and Q.pt_6 < 31.90625:
        z += -31.06527 * (Q.lam2 - 0.0003061234) * (31.90625 - Q.pt_6)
    if Q.lam1 < 0.00595415 and Q.n_pt_above_50 > 6.0:
        z += 48.76362 * (0.00595415 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.tau1 > 0.05356915 and Q.D2 < 1.002471:
        z += 8.934608 * (Q.tau1 - 0.05356915) * (1.002471 - Q.D2)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.163186
    if Q.planar_flow < 0.2534037:
        z += -5.895655 * Q.planar_flow + 1.493981
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += -10.19363 * Q.sj2_dr + 1.813237
    if Q.sj2_dr >= 0.2687922:
        z += -5.314428 * Q.sj2_dr + 0.5017441
    if Q.mass < 15.45403:
        z += -0.01020455 * Q.mass - 1.23934
    if 15.45403 <= Q.mass < 49.6681:
        z += 0.02266841 * Q.mass - 1.74736
    if 49.6681 <= Q.mass < 69.61135:
        z += 0.03116157 * Q.mass - 2.169199
    if Q.sum_z_dr2 < 0.004372139:
        z += -181.0818 * Q.sum_z_dr2 + 6.103832
    if 0.004372139 <= Q.sum_z_dr2 < 0.01323868:
        z += -599.1198 * Q.sum_z_dr2 + 7.931552
    if Q.tau1 < 0.09538712:
        z += 30.76077 * Q.tau1 - 2.934182
    if Q.centroid_offset < 0.01437952:
        z += -96.95087 * Q.centroid_offset + 3.064782
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -71.45293 * Q.centroid_offset + 2.698134
    if Q.centroid_offset >= 0.04990367:
        z += -133.886 * Q.centroid_offset + 6.681403
    if Q.sj3_dr_max < 0.1426152:
        z += 4.751748 * Q.sj3_dr_max - 1.179006
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 31.78576 * Q.sj3_dr_max - 5.034466
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -3.625859 * Q.sj3_dr_max + 0.9511254
    if Q.lam1_plus_lam2 < 0.006679471:
        z += -1546.685 * Q.lam1_plus_lam2 + 12.89222
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -1281.505 * Q.lam1_plus_lam2 + 11.12096
    if Q.lam1 < 0.00733008:
        z += 622.9304 * Q.lam1 - 5.03225
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 445.8385 * Q.lam1 - 3.734152
    if Q.sum_zz_dr2 < 0.008168571:
        z += 507.6498 * Q.sum_zz_dr2 - 4.146773
    if Q.sum_z_dr < 0.02689598:
        z += 124.6662 * Q.sum_z_dr - 5.77636
    if 0.02689598 <= Q.sum_z_dr < 0.0717028:
        z += 54.08416 * Q.sum_z_dr - 3.877986
    if Q.z_7 >= 0.01685855:
        z += 41.94184 * Q.z_7 - 0.7070784
    if Q.max_pair_mass >= 33.3761:
        z += 0.02385376 * Q.max_pair_mass - 0.7961455
    if Q.pt_7 >= 29.04219:
        z += -0.0385957 * Q.pt_7 + 1.120904
    if Q.eccentricity >= 0.9884745:
        z += 64.59837 * Q.eccentricity - 63.85385
    if Q.C2 < 0.03578649:
        z += 23.37143 * Q.C2 - 0.8363814
    if Q.max_dr < 0.1452311:
        z += -7.883147 * Q.max_dr + 1.144878
    if Q.e3 < 1.050302e-05:
        z += -64434.13 * Q.e3 + 0.6767531
    if Q.sum_pt_top5 < 506.875:
        z += 0.004962912 * Q.sum_pt_top5 - 2.515576
    if Q.sum_pt >= 988.4078:
        z += 0.002048741 * Q.sum_pt - 2.024992
    if Q.mass_over_sum_pt < 0.07637363:
        z += 24.47432 * Q.mass_over_sum_pt - 1.869193
    if Q.e2 < 0.01655442:
        z += -84.8742 * Q.e2 + 1.405043
    if Q.pt_6 < 24.42188:
        z += 0.04854043 * Q.pt_6 - 1.185448
    if Q.zdr_0 < 0.003676313:
        z += -125.2391 * Q.zdr_0 + 0.4604181
    if Q.n_dr_0p1_0p2 >= 3.0:
        z += -0.1036412 * Q.n_dr_0p1_0p2 + 0.3109237
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.004250093 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.tau1 < 0.09538712 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += 9.948393 * (0.09538712 - Q.tau1) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -535700.7 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.mass < 49.6681 and Q.D2 < 1.332146:
        z += -0.02494926 * (49.6681 - Q.mass) * (1.332146 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.pt_6 > 19.46875:
        z += -0.5604442 * (0.01323868 - Q.sum_z_dr2) * (Q.pt_6 - 19.46875)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.163393 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset < 0.03776099 and Q.mean_phi < -0.001813533:
        z += 850.9391 * (0.03776099 - Q.centroid_offset) * (-0.001813533 - Q.mean_phi)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += 59.93345 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 0.415246:
        z += 479.6428 * (0.01323868 - Q.sum_z_dr2) * (0.415246 - Q.D2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -273.0782 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.mass < 69.61135 and Q.D2 < 0.415246:
        z += -0.1766857 * (69.61135 - Q.mass) * (0.415246 - Q.D2)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -609.1407 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 150.6152 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.pt_6 < 41.21875 and Q.D2_b2 < 4.721224:
        z += -0.005887252 * (41.21875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += -849.0319 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    if Q.sum_zz_dr2 < 0.008168571 and Q.mass_top2 > 6.779915:
        z += -4.001846 * (0.008168571 - Q.sum_zz_dr2) * (Q.mass_top2 - 6.779915)
    if Q.eccentricity > 0.9884745 and Q.sj2_mass2 < 2.771069:
        z += -18.45834 * (Q.eccentricity - 0.9884745) * (2.771069 - Q.sj2_mass2)
    if Q.eccentricity > 0.9884745 and Q.eta_0 < 0.02130127:
        z += -379.5187 * (Q.eccentricity - 0.9884745) * (0.02130127 - Q.eta_0)
    if Q.centroid_offset < 0.01437952 and Q.zdr_7 < 0.00279494:
        z += -12530.85 * (0.01437952 - Q.centroid_offset) * (0.00279494 - Q.zdr_7)
    if Q.mass < 15.45403 and Q.zdr_7 < 0.003780225:
        z += 14.13016 * (15.45403 - Q.mass) * (0.003780225 - Q.zdr_7)
    if Q.centroid_offset < 0.01437952 and Q.sj3_pair_mass_min < 15.95929:
        z += -8.894162 * (0.01437952 - Q.centroid_offset) * (15.95929 - Q.sj3_pair_mass_min)
    if Q.sum_z_dr < 0.02689598 and Q.sj3_pair_mass_min > 1.590484:
        z += -7.272052 * (0.02689598 - Q.sum_z_dr) * (Q.sj3_pair_mass_min - 1.590484)
    if Q.planar_flow < 0.2534037 and Q.sj3_pair_mass_min > 1.282345:
        z += -0.18193 * (0.2534037 - Q.planar_flow) * (Q.sj3_pair_mass_min - 1.282345)
    if Q.centroid_offset > 0.04990367 and Q.tau4 > 0.002207727:
        z += -9383.705 * (Q.centroid_offset - 0.04990367) * (Q.tau4 - 0.002207727)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.419368
    if Q.sum_z_dr2 >= 0.01882765:
        z += 452.5002 * Q.sum_z_dr2 - 8.519517
    if Q.mass >= 88.15578:
        z += 0.0579364 * Q.mass - 5.107429
    if Q.mass_over_sum_pt >= 0.1309286:
        z += -72.86234 * Q.mass_over_sum_pt + 9.539766
    if Q.zdr_0 >= 0.03981924:
        z += -24.80181 * Q.zdr_0 + 0.9875895
    if Q.sum_z_dr2_top2 >= 0.01403324:
        z += -34.6494 * Q.sum_z_dr2_top2 + 0.4862435
    if Q.mean_phi >= 0.02612796:
        z += 16.7633 * Q.mean_phi - 0.4379908
    if Q.sum_pt < 437.2453:
        z += 0.01303045 * Q.sum_pt - 5.697501
    if Q.sum_z_dr2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += 6645.483 * (Q.sum_z_dr2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.mass > 88.15578 and Q.n_dr_0p2_0p4 > 1.0:
        z += 0.005026645 * (Q.mass - 88.15578) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -3.693877 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.4710226
    if Q.sum_z_dr < 0.1484084:
        z += -74.44243 * Q.sum_z_dr + 11.04788
    if Q.lam1 < 0.006506576:
        z += -508.8993 * Q.lam1 + 4.704112
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -140.3139 * Q.lam1 + 2.305883
    if 658.125 <= Q.sum_pt_top5 < 791.125:
        z += -0.002187306 * Q.sum_pt_top5 + 1.439521
    if Q.sum_pt_top5 >= 791.125:
        z += 0.001149752 * Q.sum_pt_top5 - 1.200509
    if Q.e3 < 5.334511e-05:
        z += -21238.97 * Q.e3 + 1.132995
    if Q.pt_6 < 31.90625:
        z += 0.07972275 * Q.pt_6 - 2.543654
    if Q.e2 < 0.08000524:
        z += 57.50505 * Q.e2 - 4.600705
    if Q.mass >= 49.6681:
        z += -0.02333561 * Q.mass + 1.159035
    if Q.z_5 < 0.02818362:
        z += 93.22263 * Q.z_5 - 2.627351
    if Q.z_7 < 0.02807091:
        z += 106.213 * Q.z_7 - 2.981496
    if Q.sj3_dr23 >= 0.1974628:
        z += -2.318058 * Q.sj3_dr23 + 0.4577303
    if Q.log_sum_pt >= 6.502799:
        z += 1.775956 * Q.log_sum_pt - 11.54869
    if Q.pt_7 >= 48.71875:
        z += -0.03391868 * Q.pt_7 + 1.652475
    if Q.lam1_plus_lam2 < 0.007520088:
        z += 446.9453 * Q.lam1_plus_lam2 - 3.14359
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -38.03005 * Q.lam1_plus_lam2 + 0.5034675
    if Q.lam2 < 0.0001947983:
        z += 1181.115 * Q.lam2 - 0.2300791
    if Q.z_6 < 0.02160287:
        z += 144.2053 * Q.z_6 - 3.11525
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -64.33668 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -1.013156 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -914367.9 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0006006751 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -0.729886 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.sum_z_dr < 0.1484084 and Q.z_7 > 0.06164517:
        z += 343.3006 * (0.1484084 - Q.sum_z_dr) * (Q.z_7 - 0.06164517)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.000537286:
        z += -7661.93 * (0.1484084 - Q.sum_z_dr) * (0.000537286 - Q.lam2)
    if Q.sum_z_dr < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -1.137324 * (0.1484084 - Q.sum_z_dr) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_pt > 988.4078 and Q.M3 < 0.08151794:
        z += -0.1017998 * (Q.sum_pt - 988.4078) * (0.08151794 - Q.M3)
    if Q.sum_z_dr < 0.1484084 and Q.M3 < 0.07474969:
        z += 89.25812 * (0.1484084 - Q.sum_z_dr) * (0.07474969 - Q.M3)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 > 0.008780509:
        z += 209.3671 * (0.1484084 - Q.sum_z_dr) * (Q.tau2 - 0.008780509)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -336.8182 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.08564202 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.0006168307 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0002927578 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.6522384
    if Q.planar_flow < 0.1115136:
        z += -12.04041 * Q.planar_flow + 1.34267
    if Q.z_dr_0p05_0p1 < 0.163898:
        z += 1.72467 * Q.z_dr_0p05_0p1 - 0.2210174
    if 0.163898 <= Q.z_dr_0p05_0p1 < 0.5882598:
        z += -0.1452832 * Q.z_dr_0p05_0p1 + 0.08546428
    if 0.7143804 <= Q.psi_0p1 < 0.9761279:
        z += 0.9059017 * Q.psi_0p1 - 0.6471584
    if Q.psi_0p1 >= 0.9761279:
        z += -33.45124 * Q.psi_0p1 + 32.88981
    if 0.0009641429 <= Q.sum_z_dr2 < 0.007520088:
        z += -103.9138 * Q.sum_z_dr2 + 0.1001878
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -1325.302 * Q.sum_z_dr2 + 9.285136
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1242.859 * Q.sum_z_dr2 + 8.569692
    if 0.002074109 <= Q.sum_zz_dr2 < 0.0030133:
        z += 308.8806 * Q.sum_zz_dr2 - 0.6406521
    if 0.0030133 <= Q.sum_zz_dr2 < 0.01165737:
        z += -15.7843 * Q.sum_zz_dr2 + 0.3376608
    if Q.sum_zz_dr2 >= 0.01165737:
        z += 642.5327 * Q.sum_zz_dr2 - 7.336587
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.005590289:
        z += 425.0844 * Q.lam1_plus_lam2 - 1.51441
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.006679471:
        z += 393.9585 * Q.lam1_plus_lam2 - 1.340408
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -968.7508 * Q.lam1_plus_lam2 + 7.76177
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -1029.835 * Q.lam1_plus_lam2 + 8.570441
    if 0.01655442 <= Q.e2 < 0.04110972:
        z += -92.15296 * Q.e2 + 1.525539
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += 1.15947 * Q.e2 - 2.310509
    if Q.e2 >= 0.05028464:
        z += 32.11383 * Q.e2 - 3.867038
    if 0.02685622 <= Q.centroid_offset < 0.03776099:
        z += -58.98315 * Q.centroid_offset + 1.584065
    if 0.03776099 <= Q.centroid_offset < 0.04990367:
        z += -192.566 * Q.centroid_offset + 6.628287
    if Q.centroid_offset >= 0.04990367:
        z += -627.4928 * Q.centroid_offset + 28.33273
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 156.6503 * Q.mass_over_sum_pt - 12.52008
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 341.2592 * Q.mass_over_sum_pt - 28.16598
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 280.4791 * Q.mass_over_sum_pt - 22.67062
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1206887 * Q.n_dr_0_0p05 + 0.6034433
    if Q.e3 < 5.334511e-05:
        z += 14811.89 * Q.e3 - 1.017714
    if 5.334511e-05 <= Q.e3 < 8.147744e-05:
        z += 8089.365 * Q.e3 - 0.6591007
    if Q.sd_mass < 49.91626:
        z += -0.04029819 * Q.sd_mass + 1.817401
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.00787231 * Q.sd_mass - 0.5870904
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 64.14404 * Q.sum_z_dr - 1.725217
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 107.1968 * Q.sum_z_dr - 3.482609
    if Q.sum_z_dr >= 0.08723651:
        z += -17.18038 * Q.sum_z_dr + 7.367624
    if Q.mass >= 69.61135:
        z += -0.04244272 * Q.mass + 2.954495
    if Q.mass_top5 >= 49.18618:
        z += 0.02750719 * Q.mass_top5 - 1.352974
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -6.102641 * Q.sj2_dr + 0.3755647
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -8.130017 * Q.sj2_dr + 0.6380903
    if 0.1591713 <= Q.sj2_dr < 0.1778793:
        z += 40.24136 * Q.sj2_dr - 7.061245
    if 0.1778793 <= Q.sj2_dr < 0.2001708:
        z += 22.80303 * Q.sj2_dr - 3.959327
    if Q.sj2_dr >= 0.2001708:
        z += -15.29722 * Q.sj2_dr + 3.667231
    if Q.lam2 < 0.000537286:
        z += -1311.132 * Q.lam2 + 0.7044527
    if Q.C2_b2 < 0.004032342:
        z += 185.661 * Q.C2_b2 - 0.7486486
    if Q.LHA >= 0.3033137:
        z += 17.74424 * Q.LHA - 5.382071
    if Q.sum_pt < 715.4688:
        z += 0.004494741 * Q.sum_pt - 3.215847
    if Q.sd_rg < 0.2330919:
        z += 8.277851 * Q.sd_rg - 2.687372
    if 0.2330919 <= Q.sd_rg < 0.2787955:
        z += -6.596613 * Q.sd_rg + 0.7797454
    if 0.2787955 <= Q.sd_rg < 0.324646:
        z += 20.7575 * Q.sd_rg - 6.846458
    if Q.sd_rg >= 0.324646:
        z += 12.47965 * Q.sd_rg - 4.159087
    if Q.z_dr_0_0p05 >= 0.1515405:
        z += 0.7497566 * Q.z_dr_0_0p05 - 0.1136185
    if Q.z_dr_0p1_0p2 < 0.07870506:
        z += -8.233332 * Q.z_dr_0p1_0p2 + 0.6480049
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -5172.409 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 4274.981 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.01534061 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -423.4088 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.planar_flow < 0.1115136 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -12.28567 * (0.1115136 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 8.240739 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 150511.7 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.1591713 and Q.D2_b2 < 0.08499387:
        z += 471.9775 * (Q.sj2_dr - 0.1591713) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.2001708 and Q.D2_b2 < 0.08499387:
        z += -311.4373 * (Q.sj2_dr - 0.2001708) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1294903 and Q.D2_b2 < 0.08499387:
        z += -225.7545 * (Q.sj2_dr - 0.1294903) * (0.08499387 - Q.D2_b2)
    if Q.psi_0p1 > 0.8155839 and Q.D2_b2 < 0.08499387:
        z += 37.9939 * (Q.psi_0p1 - 0.8155839) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1591713 and Q.dr_3 < 0.02358801:
        z += -501.9183 * (Q.sj2_dr - 0.1591713) * (0.02358801 - Q.dr_3)
    if Q.sj2_dr > 0.2001708 and Q.dr_3 < 0.05268713:
        z += 236.729 * (Q.sj2_dr - 0.2001708) * (0.05268713 - Q.dr_3)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.sum_pt < 715.4688:
        z += 0.003040551 * (0.5882598 - Q.z_dr_0p05_0p1) * (715.4688 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.C2_b2 < 0.0008333816:
        z += 688102.9 * (Q.centroid_offset - 0.04990367) * (0.0008333816 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.C2_b2 < 0.02415398:
        z += -382.4125 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.02415398 - Q.C2_b2)
    if Q.sd_rg > 0.2787955 and Q.D2_b2 < 0.2669656:
        z += -41.83912 * (Q.sd_rg - 0.2787955) * (0.2669656 - Q.D2_b2)
    if Q.sj2_dr > 0.1591713 and Q.dr_2 < 0.03293672:
        z += -582.8827 * (Q.sj2_dr - 0.1591713) * (0.03293672 - Q.dr_2)
    if Q.sj2_dr > 0.2001708 and Q.dr_2 < 0.05056028:
        z += 575.1707 * (Q.sj2_dr - 0.2001708) * (0.05056028 - Q.dr_2)
    if Q.centroid_offset > 0.02685622 and Q.pt_1 < 150.625:
        z += 0.6068287 * (Q.centroid_offset - 0.02685622) * (150.625 - Q.pt_1)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.622217
    if Q.N2 < 0.2233283:
        z += 8.889771 * Q.N2 - 1.985337
    if Q.sum_z_dr2 < 0.007520088:
        z += 528.1703 * Q.sum_z_dr2 - 3.971888
    if Q.mass_over_sum_pt < 0.1309286:
        z += 21.824 * Q.mass_over_sum_pt - 2.857387
    if Q.lam1_plus_lam2 < 0.00609665:
        z += -8.89238 * Q.lam1_plus_lam2 + 6.024565
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -715.9416 * Q.lam1_plus_lam2 + 10.3352
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += -153.3515 * Q.lam1_plus_lam2 + 2.88725
    if Q.e2 < 0.03846696:
        z += -79.28393 * Q.e2 + 2.166238
    if 0.03846696 <= Q.e2 < 0.04110972:
        z += -121.6175 * Q.e2 + 3.794684
    if 0.04110972 <= Q.e2 < 0.06344108:
        z += 53.95906 * Q.e2 - 3.423221
    if Q.sum_zz_dr2 < 0.008168571:
        z += 1034.933 * Q.sum_zz_dr2 - 10.12085
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 477.7917 * Q.sum_zz_dr2 - 5.569796
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -380.7098 * Q.mass_over_sum_pt_sq + 2.734576
    if Q.lam1 < 0.002464291:
        z += -1128.539 * Q.lam1 + 5.186446
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -718.2868 * Q.lam1 + 4.175466
    if 0.00483998 <= Q.lam1 < 0.005433361:
        z += -633.4391 * Q.lam1 + 3.764804
    if 0.005433361 <= Q.lam1 < 0.008375572:
        z += -109.8158 * Q.lam1 + 0.9197706
    if Q.lam2 < 0.001130645:
        z += -751.2123 * Q.lam2 + 0.8493543
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 254.8972 * Q.sum_z_dr2_top3 - 0.4095512
    if 0.002151568 <= Q.sum_z_dr2_top3 < 0.006756161:
        z += -30.16063 * Q.sum_z_dr2_top3 + 0.2037701
    if Q.sum_z_dr < 0.08723651:
        z += 92.46609 * Q.sum_z_dr - 8.801075
    if 0.08723651 <= Q.sum_z_dr < 0.1019409:
        z += 49.96157 * Q.sum_z_dr - 5.093129
    if Q.sj2_dr < 0.1492731:
        z += -3.454563 * Q.sj2_dr - 0.1221994
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -30.56071 * Q.sj2_dr + 3.92402
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 22.93611 * Q.sj2_dr - 4.591139
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += 2168.78 * Q.sum_z_dr2_top2 - 0.4365182
    if 0.0005124533 <= Q.sum_z_dr2_top2 < 0.004007842:
        z += -55.348 * Q.sum_z_dr2_top2 + 0.7032439
    if 0.004007842 <= Q.sum_z_dr2_top2 < 0.007639643:
        z += -132.5562 * Q.sum_z_dr2_top2 + 1.012682
    if 45.7571 <= Q.mass < 76.6557:
        z += 0.03397817 * Q.mass - 1.554742
    if Q.mass >= 76.6557:
        z += -0.03102317 * Q.mass + 3.427981
    if Q.tau1 < 0.06345984:
        z += -48.45047 * Q.tau1 + 3.074659
    if Q.dr_0 < 0.008355823:
        z += -119.0986 * Q.dr_0 + 0.995167
    if 0.6747704 <= Q.z_dr_0p05_0p1 < 0.7509095:
        z += 2.036989 * Q.z_dr_0p05_0p1 - 1.3745
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -12.83439 * Q.z_dr_0p05_0p1 + 9.792557
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -4.397872 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 < 716.8828:
        z += 0.02672934 * (0.2233283 - Q.N2) * (716.8828 - Q.sum_pt_top5)
    if Q.sum_z_dr2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -4612.384 * (0.007520088 - Q.sum_z_dr2) * (0.7459513 - Q.D2)
    if Q.lam1_plus_lam2 < 0.00609665 and Q.D2 < 0.7459513:
        z += 3371.404 * (0.00609665 - Q.lam1_plus_lam2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.z_dr_0p1_0p2 > 0.4684459:
        z += 58.53239 * (0.1309286 - Q.mass_over_sum_pt) * (Q.z_dr_0p1_0p2 - 0.4684459)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.03870338 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.n_for_90pct < 6.0:
        z += -2.921058 * (0.2233283 - Q.N2) * (6.0 - Q.n_for_90pct)
    if Q.centroid_offset < 0.01837778 and Q.dr01 < 0.2512159:
        z += -187.1499 * (0.01837778 - Q.centroid_offset) * (0.2512159 - Q.dr01)
    if Q.lam2 < 0.001130645 and Q.n_for_50pct > 1.0:
        z += -111.0565 * (0.001130645 - Q.lam2) * (Q.n_for_50pct - 1.0)
    if Q.lam2 < 0.0003061234 and Q.D2_b2 < 0.08499387:
        z += 23978.57 * (0.0003061234 - Q.lam2) * (0.08499387 - Q.D2_b2)
    if Q.mass_over_sum_pt_sq < 0.007182836 and Q.D2 < 0.7459513:
        z += 2499.038 * (0.007182836 - Q.mass_over_sum_pt_sq) * (0.7459513 - Q.D2)
    if Q.lam1_plus_lam2 < 0.00609665 and Q.D2_b2 < 0.08499387:
        z += -6711.249 * (0.00609665 - Q.lam1_plus_lam2) * (0.08499387 - Q.D2_b2)
    if Q.lam1_plus_lam2 < 0.01323868 and Q.tau21_b2 < 0.01272888:
        z += 34384.66 * (0.01323868 - Q.lam1_plus_lam2) * (0.01272888 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.tau21_b2 < 0.01272888:
        z += -4257.797 * (0.1309286 - Q.mass_over_sum_pt) * (0.01272888 - Q.tau21_b2)
    if Q.lam1_plus_lam2 < 0.01882765 and Q.lam2 < 0.003408389:
        z += 89568.67 * (0.01882765 - Q.lam1_plus_lam2) * (0.003408389 - Q.lam2)
    if Q.sum_zz_dr2 < 0.01165737 and Q.lam2 > 6.531723e-06:
        z += 132428.2 * (0.01165737 - Q.sum_zz_dr2) * (Q.lam2 - 6.531723e-06)
    if Q.tau1 < 0.0262518 and Q.zdr_3 > 0.008191788:
        z += -2289545.0 * (0.0262518 - Q.tau1) * (Q.zdr_3 - 0.008191788)
    if Q.sj3_pair_mass_max > 72.58129 and Q.dr_3 < 0.01039257:
        z += -410.6295 * (Q.sj3_pair_mass_max - 72.58129) * (0.01039257 - Q.dr_3)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 2.0:
        z += 6.298344 * (Q.z_dr_0p05_0p1 - 0.7509095) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.mass > 45.7571 and Q.pt_7 < 23.21641:
        z += -0.001963804 * (Q.mass - 45.7571) * (23.21641 - Q.pt_7)
    if Q.z_dr_0p05_0p1 > 0.6747704 and Q.pair_mass_0_6 > 24.79428:
        z += -0.5821875 * (Q.z_dr_0p05_0p1 - 0.6747704) * (Q.pair_mass_0_6 - 24.79428)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(5)]


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
