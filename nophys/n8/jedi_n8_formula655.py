"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 89.9% of jets.

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
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmax_over_m     largest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
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
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_3               |Δφ| of particle 3
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_3                  Δη of particle 3
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
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
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_pairmax_over_m=max(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_7=pair_mass(0, 7),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
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
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_3=abs(phi[3]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_3=eta[3],
        phi_0=phi[0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
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
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -1.874171
    if Q.planar_flow < 0.1484197:
        z += -10.6124 * Q.planar_flow + 1.575089
    if Q.lam1_plus_lam2 < 0.004372139:
        z += 812.2171 * Q.lam1_plus_lam2 - 1.61164
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -450.4247 * Q.lam1_plus_lam2 + 3.908806
    if Q.sum_z_dr2 < 0.01323868:
        z += -339.4072 * Q.sum_z_dr2 + 4.620391
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -22.73938 * Q.sum_z_dr2 + 0.4281291
    if Q.mass < 21.78408:
        z += 0.2792833 * Q.mass - 7.775014
    if 21.78408 <= Q.mass < 29.6447:
        z += 0.09912972 * Q.mass - 3.850535
    if 29.6447 <= Q.mass < 56.92035:
        z += 0.029842 * Q.mass - 1.796522
    if 56.92035 <= Q.mass < 64.61873:
        z += 0.01271759 * Q.mass - 0.8217946
    if Q.sum_z_dr2_top3 < 0.007929074:
        z += 46.24528 * Q.sum_z_dr2_top3 - 0.3666822
    if Q.lam1 < 0.0002758826:
        z += -22098.91 * Q.lam1 + 4.504991
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 324.2726 * Q.lam1 - 1.681175
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += -75.20901 * Q.lam1 + 0.4893531
    if Q.sum_pt >= 901.5938:
        z += -0.006718713 * Q.sum_pt + 6.05755
    if Q.C2_b2 < 0.001563465:
        z += 523.2858 * Q.C2_b2 - 0.8181392
    if Q.sj3_dr_max < 0.1070199:
        z += -6.1321 * Q.sj3_dr_max + 1.846998
    if 0.1070199 <= Q.sj3_dr_max < 0.233678:
        z += -0.08972263 * Q.sj3_dr_max + 1.200343
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -31.49587 * Q.sj3_dr_max + 8.539269
    if Q.sj3_dr_max >= 0.3012016:
        z += -25.36377 * Q.sj3_dr_max + 6.692271
    if Q.centroid_offset >= 0.04990367:
        z += 25.12933 * Q.centroid_offset - 1.254046
    if Q.sum_z_dr < 0.08723651:
        z += 37.57753 * Q.sum_z_dr - 3.278132
    if Q.e2 < 0.0245477:
        z += -130.5956 * Q.e2 + 3.205821
    if Q.log_sum_pt < 6.080494:
        z += 13.31018 * Q.log_sum_pt - 80.93248
    if Q.log_sum_pt >= 6.670067:
        z += -10.67599 * Q.log_sum_pt + 71.20954
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += -161.3164 * Q.mass_over_sum_pt_sq + 0.6298748
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.05358801 * Q.n_dr_0_0p05 - 0.2143521
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.01030262 * Q.sum_pt_top5 - 7.082406
    if Q.lam2 < 7.300726e-05:
        z += 10147.02 * Q.lam2 - 0.7408059
    if Q.lam1_plus_lam2 < 0.004372139 and Q.D2 < 0.7459513:
        z += -304.9053 * (0.004372139 - Q.lam1_plus_lam2) * (0.7459513 - Q.D2)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -1715.355 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 1.002471:
        z += 278.1886 * (0.01323868 - Q.sum_z_dr2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.01679745 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.sum_pt > 901.5938 and Q.dr12 < 0.119512:
        z += -0.01779231 * (Q.sum_pt - 901.5938) * (0.119512 - Q.dr12)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -9231.534 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.01837778)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.eccentricity > 0.9598562:
        z += 2509.521 * (0.004372139 - Q.lam1_plus_lam2) * (Q.eccentricity - 0.9598562)
    if Q.mass < 29.6447 and Q.phi_0 < 0.01452637:
        z += -1.449816 * (29.6447 - Q.mass) * (0.01452637 - Q.phi_0)
    if Q.sum_pt > 901.5938 and Q.absphi_3 < 0.03363037:
        z += 0.0816066 * (Q.sum_pt - 901.5938) * (0.03363037 - Q.absphi_3)
    if Q.sum_pt > 901.5938 and Q.eccentricity > 0.9458207:
        z += -0.1663102 * (Q.sum_pt - 901.5938) * (Q.eccentricity - 0.9458207)
    if Q.planar_flow < 0.1484197 and Q.z_7 < 0.02320757:
        z += 687.1106 * (0.1484197 - Q.planar_flow) * (0.02320757 - Q.z_7)
    if Q.sum_pt > 901.5938 and Q.dr0_6 > 0.2347949:
        z += 0.1243412 * (Q.sum_pt - 901.5938) * (Q.dr0_6 - 0.2347949)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 76.45641 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.sum_z_dr2 < 0.01882765 and Q.pt_6 < 46.125:
        z += -1.704895 * (0.01882765 - Q.sum_z_dr2) * (46.125 - Q.pt_6)
    if Q.mass < 64.61873 and Q.D2_b2 < 0.1830092:
        z += -0.1473432 * (64.61873 - Q.mass) * (0.1830092 - Q.D2_b2)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 89736.09 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -64.58833 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.planar_flow < 0.1484197 and Q.dr0_6 > 0.1755206:
        z += -29.89908 * (0.1484197 - Q.planar_flow) * (Q.dr0_6 - 0.1755206)
    if Q.mass < 56.92035 and Q.dr_max_012 > 0.06112084:
        z += -0.194468 * (56.92035 - Q.mass) * (Q.dr_max_012 - 0.06112084)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.1823321
    if Q.lam1 < 0.0008722282:
        z += 189.0691 * Q.lam1 + 2.264175
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -444.0807 * Q.lam1 + 2.816426
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -71.15776 * Q.lam1 + 0.595987
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 5.048174 * Q.log_sum_pt - 32.19586
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 10.83103 * Q.log_sum_pt - 69.80062
    if Q.log_sum_pt >= 6.605974:
        z += 14.29522 * Q.log_sum_pt - 92.68499
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -537.425 * Q.mass_over_sum_pt_sq + 3.134764
    if Q.z_7 < 0.02320757:
        z += 45.82112 * Q.z_7 - 2.824651
    if 0.02320757 <= Q.z_7 < 0.06164517:
        z += 60.29582 * Q.z_7 - 3.160573
    if Q.z_7 >= 0.06164517:
        z += 14.4747 * Q.z_7 - 0.3359227
    if Q.sum_z_dr2 < 0.008678045:
        z += 583.1673 * Q.sum_z_dr2 - 5.060752
    if Q.sj3_dr_min >= 0.03628191:
        z += -2.169594 * Q.sj3_dr_min + 0.07871702
    if Q.sj3_dr_max >= 0.169029:
        z += 2.612153 * Q.sj3_dr_max - 0.4415297
    if Q.sum_zz_dr2 < 0.008168571:
        z += -195.9185 * Q.sum_zz_dr2 + 1.600374
    if Q.pt_7 >= 53.4375:
        z += -0.06357171 * Q.pt_7 + 3.397113
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.00325869 * Q.sum_pt_top5 - 1.197874
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.004904293 * Q.sum_pt_top5 + 3.1382
    if Q.lam1_plus_lam2 < 0.00609665:
        z += 493.2437 * Q.lam1_plus_lam2 - 3.007135
    if Q.sum_z_dr < 0.0717028:
        z += 23.58817 * Q.sum_z_dr - 1.691338
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += -15352.74 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -558.572 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.627072e-05:
        z += 44860.26 * (Q.log_sum_pt - 6.377723) * (1.627072e-05 - Q.e3)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -11.59833 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.359195 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -3.050965 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -29.62393 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.008098937 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.006776197 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 107.8444 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 2.169016 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -818.6823 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.07607413 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.D2_b2 < 0.03885671:
        z += -29914.02 * (0.005832932 - Q.mass_over_sum_pt_sq) * (0.03885671 - Q.D2_b2)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -23.3692 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.sum_z_dr2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 21428.98 * (0.008678045 - Q.sum_z_dr2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -4615.517 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.n_dr_0_0p05 < 1.0 and Q.zdr_7 > 0.0007928864:
        z += 12.55166 * (1.0 - Q.n_dr_0_0p05) * (Q.zdr_7 - 0.0007928864)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 1552.408 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.858162
    if Q.sj3_pair_mass_max < 63.68899:
        z += -0.02354106 * Q.sj3_pair_mass_max + 1.499306
    if Q.log_sum_pt < 6.46415:
        z += -4.726662 * Q.log_sum_pt + 30.82
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -1.876596 * Q.log_sum_pt + 12.39675
    if 6.842717 <= Q.log_sum_pt < 6.896095:
        z += 15.35175 * Q.log_sum_pt - 105.0477
    if Q.log_sum_pt >= 6.896095:
        z += 9.359931 * Q.log_sum_pt - 63.72752
    if Q.lam1 < 0.00595415:
        z += -152.81 * Q.lam1 + 0.9098533
    if Q.z_7 < 0.03243272:
        z += 28.89961 * Q.z_7 - 0.1912503
    if 0.03243272 <= Q.z_7 < 0.04939969:
        z += -19.10191 * Q.z_7 + 1.36557
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -36.98985 * Q.z_7 + 2.249228
    if Q.z_7 >= 0.07148865:
        z += -17.88794 * Q.z_7 + 0.8836587
    if Q.sum_z_dr < 0.007673833:
        z += 376.2809 * Q.sum_z_dr - 2.887517
    if Q.pt_7 < 34.53125:
        z += 0.03372864 * Q.pt_7 - 1.802374
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.08232459 * Q.pt_7 - 3.480453
    if Q.pt_7 >= 53.4375:
        z += 0.04859595 * Q.pt_7 - 1.678079
    if Q.LHA >= 0.111565:
        z += -6.597797 * Q.LHA + 0.7360836
    if Q.sum_pt_top5 < 716.8828:
        z += 0.00557612 * Q.sum_pt_top5 - 3.997425
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.003210333 * Q.sum_pt_top5 + 2.696535
    if Q.pt_6 >= 27.57812:
        z += 0.04313034 * Q.pt_6 - 1.189454
    if Q.z_6 >= 0.02886576:
        z += -14.63073 * Q.z_6 + 0.4223272
    if Q.sum_pt < 788.4484:
        z += -0.009934853 * Q.sum_pt + 7.992876
    if 788.4484 <= Q.sum_pt < 840.0195:
        z += -0.003097806 * Q.sum_pt + 2.602217
    if Q.sum_z_dr2 < 0.0003193707:
        z += 17871.84 * Q.sum_z_dr2 - 5.707741
    if Q.m012 >= 42.18339:
        z += 0.02528703 * Q.m012 - 1.066692
    if 15.45403 <= Q.mass < 36.22941:
        z += 0.006959032 * Q.mass - 0.1075451
    if Q.mass >= 36.22941:
        z += -0.01128333 * Q.mass + 0.5533648
    if Q.zdr_0 < 0.0211821:
        z += -20.55602 * Q.zdr_0 + 0.4354197
    if Q.sj3_pair_mass_max < 63.68899 and Q.z_7 < 0.06810151:
        z += -0.5056852 * (63.68899 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 63.68899 and Q.centroid_offset > 0.01096064:
        z += -0.4833543 * (63.68899 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.0931108:
        z += -2020.488 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.0931108)
    if Q.z_7 > 0.04939969 and Q.sj3_dr_min < 0.0623951:
        z += -148.0364 * (Q.z_7 - 0.04939969) * (0.0623951 - Q.sj3_dr_min)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += -0.2322119 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.sum_z_dr < 0.007673833 and Q.pt_4 < 71.6875:
        z += 6.249713 * (0.007673833 - Q.sum_z_dr) * (71.6875 - Q.pt_4)
    if Q.pt_6 > 27.57812 and Q.planar_flow < 0.7974684:
        z += -0.01269083 * (Q.pt_6 - 27.57812) * (0.7974684 - Q.planar_flow)
    if Q.sum_pt < 788.4484 and Q.max_dr < 0.03623337:
        z += 0.04938503 * (788.4484 - Q.sum_pt) * (0.03623337 - Q.max_dr)
    if Q.sum_z_dr2 < 0.0003193707 and Q.mass_top2 < 36.76827:
        z += 476.4327 * (0.0003193707 - Q.sum_z_dr2) * (36.76827 - Q.mass_top2)
    if Q.z_7 < 0.07148865 and Q.planar_flow < 0.6947818:
        z += -14.19778 * (0.07148865 - Q.z_7) * (0.6947818 - Q.planar_flow)
    if Q.sum_pt < 840.0195 and Q.dr_5 < 0.02164863:
        z += 0.7723672 * (840.0195 - Q.sum_pt) * (0.02164863 - Q.dr_5)
    if Q.sum_pt < 788.4484 and Q.dr_5 < 0.02164863:
        z += -0.8541207 * (788.4484 - Q.sum_pt) * (0.02164863 - Q.dr_5)
    if Q.z_6 > 0.02886576 and Q.dr_5 < 0.02164863:
        z += -448.3863 * (Q.z_6 - 0.02886576) * (0.02164863 - Q.dr_5)
    if Q.log_sum_pt < 6.605974 and Q.lam2 < 0.0003061234:
        z += 1798.494 * (6.605974 - Q.log_sum_pt) * (0.0003061234 - Q.lam2)
    return max(0.0, z)


def neuron_3(Q):
    z = -3.84875
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 64.43229 * Q.mass_over_sum_pt - 4.390358
    if Q.centroid_offset >= 0.01096064:
        z += 15.8477 * Q.centroid_offset - 0.1737009
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -40.95871 * Q.tau1 + 2.194123
    if Q.tau1 >= 0.1027642:
        z += -10.6513 * Q.tau1 - 0.9203943
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 557.6477 * Q.lam1 - 4.670619
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 507.2742 * Q.lam1 - 4.065948
    if Q.lam1 >= 0.01643375:
        z += 912.2924 * Q.lam1 - 10.72192
    if 0.04081947 <= Q.sum_z_dr < 0.07608178:
        z += 43.77155 * Q.sum_z_dr - 1.786732
    if Q.sum_z_dr >= 0.07608178:
        z += 112.2043 * Q.sum_z_dr - 6.993216
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 618.5131 * Q.lam1_plus_lam2 - 4.651273
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -29.81592 * Q.lam1_plus_lam2 + 3.931744
    if Q.e2 >= 0.06344108:
        z += 110.7194 * Q.e2 - 7.024157
    if Q.sum_z_dr2 < 0.004372139:
        z += 196.3784 * Q.sum_z_dr2 - 0.8585938
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1490.812 * Q.sum_z_dr2 + 12.93734
    if 0.1492731 <= Q.sj2_dr < 0.1872617:
        z += -7.94712 * Q.sj2_dr + 1.186291
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += 32.7513 * Q.sj2_dr - 6.434965
    if Q.sj2_dr >= 0.2687922:
        z += 20.65139 * Q.sj2_dr - 3.182603
    if Q.mean_eta < -0.004664942:
        z += -21.60212 * Q.mean_eta - 0.1007726
    if Q.mean_eta >= 0.01772426:
        z += 33.75097 * Q.mean_eta - 0.5982108
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += 0.9839401 * Q.sum_z_dr2_top5 + 0.07354018
    if 0.007164202 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -69.14611 * Q.sum_z_dr2_top5 + 0.575966
    if Q.max_dr >= 0.1027585:
        z += 8.789384 * Q.max_dr - 0.9031841
    if Q.mass >= 64.61873:
        z += -0.1086169 * Q.mass + 7.018689
    if Q.sd_mass >= 62.73432:
        z += 0.02561613 * Q.sd_mass - 1.607011
    if Q.z_dr_0_0p05 < 0.05226226:
        z += -10.60462 * Q.z_dr_0_0p05 + 0.5542214
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += -0.7755443 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 87.26238 * (Q.sum_z_dr - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 1.259894 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.3882688 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -1443.659 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.lam1_plus_lam2 > 0.007520088 and Q.sj3_pairmin_over_m > 0.07708997:
        z += 314.7526 * (Q.lam1_plus_lam2 - 0.007520088) * (Q.sj3_pairmin_over_m - 0.07708997)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 497.7627 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 19.72442 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.e2 > 0.06344108 and Q.z_top5 > 0.7773372:
        z += -546.35 * (Q.e2 - 0.06344108) * (Q.z_top5 - 0.7773372)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -315.9025 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.tau1 > 0.05356915 and Q.pt_5 > 59.125:
        z += 0.440445 * (Q.tau1 - 0.05356915) * (Q.pt_5 - 59.125)
    if Q.sum_z_dr > 0.07608178 and Q.eta_0 > 0.07952881:
        z += 88.20127 * (Q.sum_z_dr - 0.07608178) * (Q.eta_0 - 0.07952881)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05096142:
        z += 9402.264 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05096142)
    if Q.max_dr > 0.1027585 and Q.eta_1 < -0.05963135:
        z += -42.62861 * (Q.max_dr - 0.1027585) * (-0.05963135 - Q.eta_1)
    if Q.max_dr > 0.1027585 and Q.zdr_1 < 0.01084112:
        z += -762.3022 * (Q.max_dr - 0.1027585) * (0.01084112 - Q.zdr_1)
    if Q.max_dr > 0.1027585 and Q.abseta_0 > 0.1057739:
        z += -67.24217 * (Q.max_dr - 0.1027585) * (Q.abseta_0 - 0.1057739)
    return max(0.0, z)


def neuron_4(Q):
    z = -3.096122
    if Q.N2 < 0.2233283:
        z += -54.64143 * Q.N2 + 12.20297
    if Q.lam2 < 0.000537286:
        z += 5022.328 * Q.lam2 - 2.698427
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -92.32117 * Q.mass_over_sum_pt + 8.34711
    if Q.sum_z_dr2 < 0.002635418:
        z += 428.4106 * Q.sum_z_dr2 - 10.34559
    if 0.002635418 <= Q.sum_z_dr2 < 0.003562611:
        z += 1094.421 * Q.sum_z_dr2 - 12.1008
    if 0.003562611 <= Q.sum_z_dr2 < 0.008678045:
        z += 1270.285 * Q.sum_z_dr2 - 12.72734
    if 0.008678045 <= Q.sum_z_dr2 < 0.01882765:
        z += 432.3646 * Q.sum_z_dr2 - 5.455825
    if Q.sum_z_dr2 >= 0.01882765:
        z += 175.8649 * Q.sum_z_dr2 - 0.6265382
    if 0.009668065 <= Q.e2 < 0.04447357:
        z += 53.20971 * Q.e2 - 0.514435
    if Q.e2 >= 0.04447357:
        z += 22.90589 * Q.e2 + 0.8332844
    if Q.sum_pt < 739.5:
        z += 0.01019426 * Q.sum_pt - 7.538652
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -71.92804 * Q.sum_z_dr2_top2 + 0.5495046
    if 0.169029 <= Q.sj3_dr_max < 0.233678:
        z += 29.07239 * Q.sj3_dr_max - 4.914078
    if Q.sj3_dr_max >= 0.233678:
        z += -13.9991 * Q.sj3_dr_max + 5.15078
    if Q.sum_zz_dr2 < 0.01165737:
        z += -1040.938 * Q.sum_zz_dr2 + 14.16898
    if 0.01165737 <= Q.sum_zz_dr2 < 0.01716248:
        z += -369.542 * Q.sum_zz_dr2 + 6.342258
    if 0.05464922 <= Q.sum_z_dr < 0.1019409:
        z += -47.734 * Q.sum_z_dr + 2.608626
    if Q.sum_z_dr >= 0.1019409:
        z += -32.51441 * Q.sum_z_dr + 1.057127
    if 0.07283629 <= Q.tau1 < 0.1136369:
        z += 8.215302 * Q.tau1 - 0.5983721
    if Q.tau1 >= 0.1136369:
        z += 46.38904 * Q.tau1 - 4.936317
    if Q.mass < 69.61135:
        z += -0.04198769 * Q.mass + 3.383041
    if 69.61135 <= Q.mass < 76.6557:
        z += -0.06533196 * Q.mass + 5.008067
    if Q.mass >= 88.15578:
        z += -0.05318825 * Q.mass + 4.688851
    if Q.C2 < 0.05119235:
        z += 21.51164 * Q.C2 - 1.101231
    if Q.sum_z_dr2_top5 < 0.008329695:
        z += -61.95441 * Q.sum_z_dr2_top5 + 0.5160614
    if 0.1117619 <= Q.max_dr < 0.2215867:
        z += 2.335267 * Q.max_dr - 0.2609938
    if Q.max_dr >= 0.2215867:
        z += 21.59863 * Q.max_dr - 4.5295
    if Q.D2 < 1.002471:
        z += -2.615985 * Q.D2 + 2.622448
    if Q.tau21_b2 < 0.004811143:
        z += 149.2902 * Q.tau21_b2 - 0.7182566
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 268.7117 * Q.lam1_plus_lam2 - 3.557386
    if Q.C2_b2 < 0.009032972:
        z += -264.6314 * Q.C2_b2 + 2.390408
    if Q.mean_eta >= 0.02644207:
        z += -16.67347 * Q.mean_eta + 0.4408811
    if Q.sd_mass >= 38.43971:
        z += -0.01364275 * Q.sd_mass + 0.5244233
    if Q.N2 < 0.2233283 and Q.mass < 60.63098:
        z += -0.611209 * (0.2233283 - Q.N2) * (60.63098 - Q.mass)
    if Q.N2 < 0.2233283 and Q.sum_zz_dr2 > 0.01165737:
        z += -1640.867 * (0.2233283 - Q.N2) * (Q.sum_zz_dr2 - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_6 < 62.25:
        z += -0.04888712 * (0.2233283 - Q.N2) * (62.25 - Q.pt_6)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 276.4641 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -86.85861 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1626038:
        z += -28.66649 * (0.2233283 - Q.N2) * (0.1626038 - Q.abseta_7)
    if Q.tau1 > 0.07283629 and Q.sj3_dr_min < 0.2089872:
        z += 203.026 * (Q.tau1 - 0.07283629) * (0.2089872 - Q.sj3_dr_min)
    if Q.C2 < 0.05119235 and Q.mean_eta > 0.02644207:
        z += -715.1789 * (0.05119235 - Q.C2) * (Q.mean_eta - 0.02644207)
    if Q.mass < 76.6557 and Q.D2 < 0.875672:
        z += -0.1959676 * (76.6557 - Q.mass) * (0.875672 - Q.D2)
    if Q.sum_zz_dr2 < 0.01165737 and Q.D2 < 0.875672:
        z += 497.0181 * (0.01165737 - Q.sum_zz_dr2) * (0.875672 - Q.D2)
    if Q.mass < 76.6557 and Q.mean_eta2 > 0.006395693:
        z += 5.187951 * (76.6557 - Q.mass) * (Q.mean_eta2 - 0.006395693)
    if Q.N2 < 0.2233283 and Q.pt_7 < 25.57812:
        z += -0.5002723 * (0.2233283 - Q.N2) * (25.57812 - Q.pt_7)
    if Q.D2 < 1.002471 and Q.pt_7 < 53.4375:
        z += -0.09493871 * (1.002471 - Q.D2) * (53.4375 - Q.pt_7)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.mean_eta < -0.01790907:
        z += 3148.807 * (0.007639643 - Q.sum_z_dr2_top2) * (-0.01790907 - Q.mean_eta)
    if Q.lam2 < 0.000537286 and Q.mean_phi < -0.01753483:
        z += -76595.25 * (0.000537286 - Q.lam2) * (-0.01753483 - Q.mean_phi)
    if Q.sum_pt < 739.5 and Q.sj3_mass2 < 1.186019:
        z += 0.001265794 * (739.5 - Q.sum_pt) * (1.186019 - Q.sj3_mass2)
    if Q.tau21_b2 < 0.004811143 and Q.sj3_mass1 > 0.8392664:
        z += -191.3518 * (0.004811143 - Q.tau21_b2) * (Q.sj3_mass1 - 0.8392664)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.04474208
    if Q.z_7 < 0.02807091:
        z += -196.029 * Q.z_7 + 7.876839
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -100.4372 * Q.z_7 + 5.19349
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -10.4994 * Q.z_7 + 0.7505882
    if 6.267538 <= Q.log_sum_pt < 6.701242:
        z += 4.956934 * Q.log_sum_pt - 31.06778
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -10.39595 * Q.log_sum_pt + 71.8156
    if Q.log_sum_pt >= 6.896095:
        z += -16.05337 * Q.log_sum_pt + 110.8297
    if Q.sum_zz_dr2 < 0.005284669:
        z += -191.3703 * Q.sum_zz_dr2 + 1.011329
    if Q.z_6 < 0.02886576:
        z += -117.9949 * Q.z_6 + 3.406013
    if Q.sum_z_dr2 < 9.121392e-05:
        z += -12436.69 * Q.sum_z_dr2 + 2.99931
    if 9.121392e-05 <= Q.sum_z_dr2 < 0.001653836:
        z += -850.9657 * Q.sum_z_dr2 + 1.942531
    if 0.001653836 <= Q.sum_z_dr2 < 0.002635418:
        z += -545.2147 * Q.sum_z_dr2 + 1.436868
    if Q.sum_z_dr < 0.007673833:
        z += 181.4254 * Q.sum_z_dr - 1.392228
    if Q.pt_7 < 23.21641:
        z += 0.1335653 * Q.pt_7 - 4.962785
    if 23.21641 <= Q.pt_7 < 37.15625:
        z += 0.07934516 * Q.pt_7 - 3.703989
    if Q.pt_7 >= 37.15625:
        z += -0.05422014 * Q.pt_7 + 1.258797
    if Q.zdr_0 < 0.02383244:
        z += 27.86018 * Q.zdr_0 - 0.6639759
    if Q.sj3_dr_max < 0.1070199:
        z += 6.76897 * Q.sj3_dr_max - 1.821453
    if 0.1070199 <= Q.sj3_dr_max < 0.1879486:
        z += 4.679391 * Q.sj3_dr_max - 1.597827
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 6.342803 * Q.sj3_dr_max - 1.910462
    if Q.mass < 60.63098:
        z += -0.008766005 * Q.mass + 0.5314914
    if Q.sum_z_dr2_top3 < 0.002915531:
        z += 319.9088 * Q.sum_z_dr2_top3 - 0.9327042
    if Q.z_5 < 0.02818362:
        z += -92.87887 * Q.z_5 + 2.617663
    if Q.mass_over_sum_pt < 0.09041383:
        z += 25.26794 * Q.mass_over_sum_pt - 2.284571
    if Q.tau1 < 0.09538712:
        z += -32.45207 * Q.tau1 + 3.095509
    if Q.z_dr_0p1_0p2 < 0.07870506:
        z += -4.516486 * Q.z_dr_0p1_0p2 + 0.3554703
    if Q.sum_pt_top5 >= 716.8828:
        z += 0.003680227 * Q.sum_pt_top5 - 2.638292
    if Q.z_7 < 0.04939969 and Q.mass_top5 < 68.43422:
        z += -0.007252004 * (0.04939969 - Q.z_7) * (68.43422 - Q.mass_top5)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -88.45407 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.sum_zz_dr2 < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += 4804.217 * (0.005284669 - Q.sum_zz_dr2) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -734.813 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.04939969 and Q.sum_pt_top2 < 501.625:
        z += -0.154071 * (0.04939969 - Q.z_7) * (501.625 - Q.sum_pt_top2)
    if Q.log_sum_pt > 6.896095 and Q.sj3_pair_mass_max > 17.83163:
        z += -0.109175 * (Q.log_sum_pt - 6.896095) * (Q.sj3_pair_mass_max - 17.83163)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 42.18339:
        z += 0.4047586 * (0.07148865 - Q.z_7) * (42.18339 - Q.mass_top3)
    if Q.z_7 < 0.07148865 and Q.C3 < 0.03532852:
        z += -206.7258 * (0.07148865 - Q.z_7) * (0.03532852 - Q.C3)
    if Q.sum_z_dr2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 88021.47 * (0.001653836 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.C2_b2 > 0.001109927:
        z += -938.9206 * (0.07148865 - Q.z_7) * (Q.C2_b2 - 0.001109927)
    if Q.sum_zz_dr2 < 0.005284669 and Q.pt_6 > 27.57812:
        z += -2.744027 * (0.005284669 - Q.sum_zz_dr2) * (Q.pt_6 - 27.57812)
    if Q.mass < 60.63098 and Q.D2 < 3.885568:
        z += 0.003260656 * (60.63098 - Q.mass) * (3.885568 - Q.D2)
    if Q.z_6 < 0.02886576 and Q.phi_0 > 0.008850098:
        z += -3290.219 * (0.02886576 - Q.z_6) * (Q.phi_0 - 0.008850098)
    if Q.log_sum_pt > 6.701242 and Q.abseta_4 < 0.01402779:
        z += 188.1581 * (Q.log_sum_pt - 6.701242) * (0.01402779 - Q.abseta_4)
    if Q.log_sum_pt > 6.267538 and Q.abseta_4 < 0.03601074:
        z += -21.4222 * (Q.log_sum_pt - 6.267538) * (0.03601074 - Q.abseta_4)
    if Q.z_7 < 0.02807091 and Q.pt_5 > 24.57812:
        z += 2.280694 * (0.02807091 - Q.z_7) * (Q.pt_5 - 24.57812)
    if Q.log_sum_pt > 6.896095 and Q.eccentricity > 0.9781608:
        z += -701.4154 * (Q.log_sum_pt - 6.896095) * (Q.eccentricity - 0.9781608)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 < 0.0002302115:
        z += 55894.08 * (Q.log_sum_pt - 6.701242) * (0.0002302115 - Q.mean_phi2)
    if Q.sum_z_dr2 < 0.001653836 and Q.absphi_0 < 0.01452637:
        z += -14636.36 * (0.001653836 - Q.sum_z_dr2) * (0.01452637 - Q.absphi_0)
    if Q.zdr_0 < 0.02383244 and Q.z_5 > 0.02818362:
        z += 215.5856 * (0.02383244 - Q.zdr_0) * (Q.z_5 - 0.02818362)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi2 < 0.0002302115:
        z += -109231.0 * (Q.log_sum_pt - 6.896095) * (0.0002302115 - Q.mean_phi2)
    if Q.tau1 < 0.09538712 and Q.planar_flow < 0.4926918:
        z += 8.555479 * (0.09538712 - Q.tau1) * (0.4926918 - Q.planar_flow)
    if Q.sum_z_dr2_top3 < 0.002915531 and Q.tau3 < 0.006646257:
        z += -18717.68 * (0.002915531 - Q.sum_z_dr2_top3) * (0.006646257 - Q.tau3)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 68924.6 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 0.00424745:
        z += -418.2422 * (Q.log_sum_pt - 6.701242) * (0.00424745 - Q.mean_eta2)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.09820732
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 59.91779 * Q.centroid_offset - 0.4848763
    if Q.centroid_offset >= 0.01837778:
        z += -16.29999 * Q.centroid_offset + 0.9158372
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 464.5751 * Q.lam1_plus_lam2 - 6.150359
    if Q.mass < 45.7571:
        z += -0.05236822 * Q.mass + 2.49013
    if 45.7571 <= Q.mass < 60.63098:
        z += -0.006313885 * Q.mass + 0.382817
    if Q.pt_6 < 29.90625:
        z += -0.01576177 * Q.pt_6 + 0.1729598
    if 29.90625 <= Q.pt_6 < 39.75:
        z += 0.04315639 * Q.pt_6 - 1.589061
    if 39.75 <= Q.pt_6 < 41.21875:
        z += -0.08606307 * Q.pt_6 + 3.547412
    if Q.lam2 < 0.001130645:
        z += 628.5942 * Q.lam2 + 0.05910073
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -337.9736 * Q.lam2 + 1.151946
    if Q.tau1 < 0.1136369:
        z += -36.03442 * Q.tau1 + 4.094839
    if Q.sj3_dr_min >= 0.1278212:
        z += -7.004465 * Q.sj3_dr_min + 0.8953188
    if Q.lam1 < 0.00733008:
        z += -423.9186 * Q.lam1 + 4.140606
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -221.0799 * Q.lam1 + 2.653782
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.04379737 * Q.sj3_pair_mass_min + 0.1971638
    if Q.sum_pt < 615.875:
        z += 0.00208068 * Q.sum_pt - 1.281439
    if Q.sum_zz_dr2 < 0.0030133:
        z += -493.4409 * Q.sum_zz_dr2 + 1.486886
    if Q.eccentricity >= 0.927072:
        z += -2.894557 * Q.eccentricity + 2.683463
    if Q.sj3_dr_max < 0.1789613:
        z += -26.10442 * Q.sj3_dr_max + 4.301674
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 3.026901 * Q.sj3_dr_max - 0.9117072
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 14.32203 * Q.sj3_dr_max - 3.03461
    if Q.sj3_dr_max >= 0.3012016:
        z += 11.29513 * Q.sj3_dr_max - 2.122903
    if Q.max_dr < 0.1452311:
        z += 27.42978 * Q.max_dr - 3.983657
    if Q.mean_eta >= 0.02644207:
        z += -71.43581 * Q.mean_eta + 1.888911
    if Q.e3 < 0.0005116989:
        z += -1370.529 * Q.e3 + 0.7012983
    if Q.sum_z_dr2 < 0.008678045:
        z += 883.6243 * Q.sum_z_dr2 - 7.668131
    if Q.C2_b2 < 0.02415398:
        z += -60.83549 * Q.C2_b2 + 1.46942
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += 1.017592 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.6567881 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += 36.83475 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.C2 < 0.02702951:
        z += 1104.218 * (Q.centroid_offset - 0.01837778) * (0.02702951 - Q.C2)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 10674.15 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -251.0534 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -6.786157 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.sum_pt < 615.875 and Q.mean_phi > 0.01251955:
        z += -0.07715355 * (615.875 - Q.sum_pt) * (Q.mean_phi - 0.01251955)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.845165 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.003253661 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 77.52845 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += 0.102069 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 9992.427 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.00433128:
        z += -4388.842 * (Q.centroid_offset - 0.01837778) * (0.00433128 - Q.mean_phi2)
    if Q.sum_pt < 615.875 and Q.z_3 < 0.05101919:
        z += 10.31201 * (615.875 - Q.sum_pt) * (0.05101919 - Q.z_3)
    if Q.pt_6 < 41.21875 and Q.M3 > 0.0782171:
        z += 2.664867 * (41.21875 - Q.pt_6) * (Q.M3 - 0.0782171)
    if Q.lam1 < 0.00733008 and Q.n_dr_0p2_0p4 < 1.0:
        z += 121.0366 * (0.00733008 - Q.lam1) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.mass < 45.7571 and Q.eta_3 > -0.00592804:
        z += -0.1455616 * (45.7571 - Q.mass) * (Q.eta_3 - -0.00592804)
    if Q.centroid_offset > 0.01837778 and Q.eta_0 < 0.02980347:
        z += -94.27305 * (Q.centroid_offset - 0.01837778) * (0.02980347 - Q.eta_0)
    if Q.mean_eta > 0.02644207 and Q.M3 > 0.06688759:
        z += -1312.969 * (Q.mean_eta - 0.02644207) * (Q.M3 - 0.06688759)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.01019824 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.31565
    if Q.sum_z_dr2_top2 < 0.001056655:
        z += -310.8044 * Q.sum_z_dr2_top2 + 0.3284131
    if Q.sum_z_dr2 < 0.0009641429:
        z += 1648.502 * Q.sum_z_dr2 - 1.881514
    if 0.0009641429 <= Q.sum_z_dr2 < 0.001653836:
        z += 112.4207 * Q.sum_z_dr2 - 0.4005114
    if 0.001653836 <= Q.sum_z_dr2 < 0.003562611:
        z += 511.3796 * Q.sum_z_dr2 - 1.060324
    if 0.003562611 <= Q.sum_z_dr2 < 0.004372139:
        z += 398.9588 * Q.sum_z_dr2 - 0.6598126
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += 110.5998 * Q.sum_z_dr2 + 0.6009333
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -912.8895 * Q.sum_z_dr2 + 8.297663
    if Q.sum_z_dr2 >= 0.01323868:
        z += -99.0282 * Q.sum_z_dr2 - 2.476782
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -57.2821 * Q.mass_over_sum_pt + 0.6358223
    if 0.07269073 <= Q.mass_over_sum_pt < 0.08475161:
        z += 76.00261 * Q.mass_over_sum_pt - 9.052741
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 268.117 * Q.mass_over_sum_pt - 25.33475
    if 0.09041383 <= Q.mass_over_sum_pt < 0.1079857:
        z += -19.44449 * Q.mass_over_sum_pt + 0.6647894
    if Q.mass_over_sum_pt >= 0.1079857:
        z += -144.6339 * Q.mass_over_sum_pt + 14.18345
    if Q.tau1 < 0.05356915:
        z += -32.37513 * Q.tau1 + 1.734308
    if Q.sum_z_dr < 0.04081947:
        z += 122.2895 * Q.sum_z_dr - 8.718778
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 80.29346 * Q.sum_z_dr - 7.004521
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.0630198 * Q.mass - 2.28317
    if Q.mass >= 76.6557:
        z += -0.1072492 * Q.mass + 10.76892
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -3.31892 * Q.z_dr_0p1_0p2 + 0.526242
    if Q.centroid_offset < 0.02076709:
        z += 24.21902 * Q.centroid_offset - 0.5029586
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += 42.04938 * Q.centroid_offset - 1.310712
    if Q.centroid_offset >= 0.03776099:
        z += -65.98578 * Q.centroid_offset + 2.768803
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -1139.602 * Q.lam1_plus_lam2 + 6.370704
    if Q.lam1_plus_lam2 >= 0.008678045:
        z += -304.1452 * Q.lam1_plus_lam2 - 0.8794273
    if Q.z_7 >= 0.03243272:
        z += 14.45937 * Q.z_7 - 0.4689567
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += -21.31951 * Q.e2 + 0.7581413
    if Q.e2 >= 0.05028464:
        z += -140.4994 * Q.e2 + 6.751059
    if Q.sj2_dr < 0.1294903:
        z += -12.8847 * Q.sj2_dr + 0.7175255
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -0.6033821 * Q.sj2_dr - 0.872786
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 34.48959 * Q.sj2_dr - 6.458581
    if Q.sum_zz_dr2 < 0.001101266:
        z += 342.589 * Q.sum_zz_dr2 - 0.3772817
    if Q.LHA < 0.3033137:
        z += -9.97061 * Q.LHA + 3.024223
    if Q.pt_7 < 29.04219:
        z += 0.04879002 * Q.pt_7 - 1.416969
    if Q.sd_rg >= 0.2787955:
        z += 31.50659 * Q.sd_rg - 8.783897
    if Q.lam1 < 0.008375572:
        z += 230.5287 * Q.lam1 - 1.93081
    if Q.e3 < 8.147744e-05:
        z += 20952.65 * Q.e3 - 1.707168
    if Q.planar_flow < 0.1950135 and Q.lam1_plus_lam2 > 0.007520088:
        z += -217.1256 * (0.1950135 - Q.planar_flow) * (Q.lam1_plus_lam2 - 0.007520088)
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 35.28125:
        z += -0.2014438 * (0.1950135 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1131601 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.z_dr_0p1_0p2 < 0.1585582 and Q.N3 < 2.080881:
        z += -1.390714 * (0.1585582 - Q.z_dr_0p1_0p2) * (2.080881 - Q.N3)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -15.01365 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.07497185 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 710.4113 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += -5919.083 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.centroid_offset > 0.03776099 and Q.pt_4 > 81.375:
        z += -38.32036 * (Q.centroid_offset - 0.03776099) * (Q.pt_4 - 81.375)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -22631.38 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.sum_z_dr < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 1836.969 * (0.08723651 - Q.sum_z_dr) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 3109.259 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.4595087
    if Q.sum_z_dr2 < 0.001653836:
        z += -194.0794 * Q.sum_z_dr2 + 1.411125
    if 0.001653836 <= Q.sum_z_dr2 < 0.005019719:
        z += -346.5569 * Q.sum_z_dr2 + 1.663298
    if 0.005019719 <= Q.sum_z_dr2 < 0.006679471:
        z += 45.98282 * Q.sum_z_dr2 - 0.3071409
    if Q.tau1 < 0.05356915:
        z += 12.07604 * Q.tau1 - 0.6469031
    if Q.LHA < 0.1967397:
        z += 37.98194 * Q.LHA - 7.472557
    if Q.log_sum_pt >= 6.701242:
        z += -11.01734 * Q.log_sum_pt + 73.82989
    if Q.sum_z_dr < 0.06108601:
        z += 71.84133 * Q.sum_z_dr - 4.388501
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.005153167 * Q.sum_pt_top5 - 3.54248
    if Q.lam2 < 0.0001330621:
        z += 2844.916 * Q.lam2 - 0.3785504
    if Q.mass < 49.6681:
        z += 0.01513307 * Q.mass - 0.751631
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -535.8749 * Q.lam1_plus_lam2 + 1.909114
    if Q.z_6 < 0.03448406:
        z += -18.49339 * Q.z_6 + 0.6377269
    if Q.mass_over_sum_pt < 0.03319429:
        z += -26.98368 * Q.mass_over_sum_pt + 0.895704
    if Q.sj3_dr_max < 0.1426152:
        z += 15.2713 * Q.sj3_dr_max - 2.117859
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -1.072267 * Q.sj3_dr_max + 0.2129815
    if Q.sum_z_dr2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -12.77938 * (0.006679471 - Q.sum_z_dr2) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 16809.66 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -6.776062 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.LHA < 0.1967397 and Q.mean_phi > -0.0008556753:
        z += 550.0564 * (0.1967397 - Q.LHA) * (Q.mean_phi - -0.0008556753)
    if Q.sum_z_dr2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 353.8733 * (0.006679471 - Q.sum_z_dr2) * (Q.phi_0 - -0.04013062)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 6238.318 * (0.005019719 - Q.sum_z_dr2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -183.8555 * (0.06108601 - Q.sum_z_dr) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242 and Q.zdr_0 > 0.007798268:
        z += -91.40972 * (Q.log_sum_pt - 6.701242) * (Q.zdr_0 - 0.007798268)
    if Q.sum_pt_top5 > 687.4375 and Q.e3 < 3.10694e-07:
        z += 8109.469 * (Q.sum_pt_top5 - 687.4375) * (3.10694e-07 - Q.e3)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.2568748 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_z_dr < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 140407.0 * (0.06108601 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 1684.945 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.tau1 < 0.05356915 and Q.D2_b2 < 4.721224:
        z += -3.598493 * (0.05356915 - Q.tau1) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.701242 and Q.e3 < 2.955458e-05:
        z += -46306.01 * (Q.log_sum_pt - 6.701242) * (2.955458e-05 - Q.e3)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -410.9239 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 17451.73 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.mass < 49.6681 and Q.lam2 < 0.0001330621:
        z += 98.41059 * (49.6681 - Q.mass) * (0.0001330621 - Q.lam2)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.1219599 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.mass < 21.78408 and Q.D2_b2 < 0.716559:
        z += 0.2804892 * (21.78408 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.0008376401 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -26930.95 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.1632371 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.sum_z_dr < 0.06108601 and Q.lam1_plus_lam2 > 0.0003193707:
        z += 4155.522 * (0.06108601 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.0003193707)
    if Q.mass < 29.6447 and Q.lam1_plus_lam2 < 0.003562611:
        z += -25.70359 * (29.6447 - Q.mass) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2 < 0.006679471:
        z += 1252.217 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.sum_z_dr2)
    if Q.tau1 < 0.05356915 and Q.lam1_plus_lam2 < 0.003562611:
        z += 15483.85 * (0.05356915 - Q.tau1) * (0.003562611 - Q.lam1_plus_lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.921289
    if Q.sum_z_dr < 0.05464922:
        z += 31.14545 * Q.sum_z_dr - 1.702075
    if Q.sum_z_dr >= 0.0717028:
        z += -17.89475 * Q.sum_z_dr + 1.283104
    if Q.tau1 < 0.04369778:
        z += -19.47037 * Q.tau1 + 0.8508121
    if Q.mass < 53.33237:
        z += 0.04353112 * Q.mass - 2.321618
    if Q.e3 < 2.371297e-05:
        z += 60537.82 * Q.e3 - 1.435532
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -4424.75 * Q.e3 + 0.3605173
    if Q.e3 >= 0.0001869378:
        z += -207.2295 * Q.e3 - 0.4278969
    if Q.sum_z_dr2 < 0.003562611:
        z += -2825.783 * Q.sum_z_dr2 + 13.72722
    if 0.003562611 <= Q.sum_z_dr2 < 0.00609665:
        z += -1444.354 * Q.sum_z_dr2 + 8.805718
    if Q.sj3_dr_max < 0.1070199:
        z += 31.06659 * Q.sj3_dr_max - 2.574832
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 26.83857 * Q.sj3_dr_max - 2.12235
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -24.09078 * Q.sj3_dr_max + 5.140948
    if Q.lam2 >= 0.001130645:
        z += 416.5605 * Q.lam2 - 0.4709819
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.6504163 * Q.n_dr_0p2_0p4 - 0.6504163
    if Q.lam1 < 0.003377388:
        z += 621.9623 * Q.lam1 - 3.703257
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 298.4887 * Q.lam1 - 2.610761
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 750.4229 * Q.lam1 - 5.066282
    if Q.lam1 >= 0.00595415:
        z += 128.4606 * Q.lam1 - 1.363025
    if Q.centroid_offset < 0.002316125:
        z += 254.3546 * Q.centroid_offset + 2.411712
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -186.8318 * Q.centroid_offset + 3.433555
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -79.06794 * Q.e2 + 2.811728
    if Q.e2 >= 0.04447357:
        z += 55.93993 * Q.e2 - 3.192554
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.08267852 * Q.sj3_pair_mass_max - 2.003311
    if Q.lam1_plus_lam2 < 0.0001721983:
        z += -6994.041 * Q.lam1_plus_lam2 + 1.204362
    if Q.C3 < 0.0284695:
        z += 48.23023 * Q.C3 - 1.37309
    if Q.zdr_0 < 0.006292091:
        z += 137.1282 * Q.zdr_0 - 0.8628229
    if Q.mass_over_sum_pt < 0.07269073:
        z += 50.01535 * Q.mass_over_sum_pt - 3.635652
    if Q.sum_pt < 559.6875:
        z += -0.008634436 * Q.sum_pt + 6.660252
    if 559.6875 <= Q.sum_pt < 988.4078:
        z += -0.004263074 * Q.sum_pt + 4.213655
    z += -0.9509515 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 14.09279 * Q.log_sum_pt - 97.18521
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.009718702 * Q.sum_pt_top5 + 8.163269
    if Q.pt_4 < 31.125:
        z += -0.07010899 * Q.pt_4 + 2.182142
    if Q.max_dr < 0.1117619:
        z += -34.23768 * Q.max_dr + 3.826467
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 5.762122 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.09287655 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 1373.201 * (0.00609665 - Q.sum_z_dr2) * (0.3220738 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -6931.86 * (0.00609665 - Q.sum_z_dr2) * (0.002834884 - Q.mean_phi)
    if Q.sum_z_dr < 0.05464922 and Q.z_5 > 0.03672711:
        z += -199.8666 * (0.05464922 - Q.sum_z_dr) * (Q.z_5 - 0.03672711)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -610.973 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -491.0214 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.e3 < 2.371297e-05 and Q.eccentricity > 0.8319502:
        z += 78100.87 * (2.371297e-05 - Q.e3) * (Q.eccentricity - 0.8319502)
    if Q.sj3_dr_max < 0.1070199 and Q.abseta_1 > 0.03625488:
        z += -1508.778 * (0.1070199 - Q.sj3_dr_max) * (Q.abseta_1 - 0.03625488)
    if Q.centroid_offset < 0.01837778 and Q.abseta_1 < 0.07073975:
        z += 577.8145 * (0.01837778 - Q.centroid_offset) * (0.07073975 - Q.abseta_1)
    if Q.tau1 < 0.04369778 and Q.mean_phi < -0.01275329:
        z += 2520.248 * (0.04369778 - Q.tau1) * (-0.01275329 - Q.mean_phi)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += 4863.967 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.centroid_offset < 0.01837778 and Q.tau4 > 0.001224244:
        z += -8742.896 * (0.01837778 - Q.centroid_offset) * (Q.tau4 - 0.001224244)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi > 0.02612796:
        z += -50929.31 * (0.00609665 - Q.sum_z_dr2) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -50.08579 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -42.42123 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.1383081 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -1.928994 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 1005.68 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -2.752444 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.sum_z_dr < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += 1931.66 * (0.05464922 - Q.sum_z_dr) * (0.02656143 - Q.tau21_b2)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.652466
    if Q.sj3_pair_mass_min >= 11.051:
        z += 0.0592415 * Q.sj3_pair_mass_min - 0.654678
    if Q.lam1 < 0.001503553:
        z += 1276.976 * Q.lam1 - 3.334443
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 496.7536 * Q.lam1 - 2.161336
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 46.89116 * Q.lam1 - 0.279197
    if Q.lam1 >= 0.00733008:
        z += -349.1788 * Q.lam1 + 2.559509
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 796.0377 * Q.lam2 - 0.2436857
    if Q.lam2 >= 0.003408389:
        z += 630.4244 * Q.lam2 + 0.320789
    if Q.pt_7 < 45.75:
        z += 0.01969313 * Q.pt_7 - 0.9009608
    if Q.sj3_dr_min >= 0.1278212:
        z += 6.610852 * Q.sj3_dr_min - 0.8450068
    if Q.e3 < 3.892127e-05:
        z += -5331.121 * Q.e3 + 0.4343661
    if 3.892127e-05 <= Q.e3 < 8.147744e-05:
        z += -6374.157 * Q.e3 + 0.4749624
    if Q.e3 >= 8.147744e-05:
        z += -1043.036 * Q.e3 + 0.04059627
    if Q.zdr_0 < 0.0211821:
        z += 15.82849 * Q.zdr_0 - 0.3352808
    if Q.M2 < 0.02563286:
        z += 29.78019 * Q.M2 - 0.7633514
    if Q.sum_pt >= 988.4078:
        z += -0.01146538 * Q.sum_pt + 11.33247
    if Q.z_7 < 0.06473447:
        z += 8.339206 * Q.z_7 - 0.5398341
    if Q.LHA >= 0.3033137:
        z += -26.40597 * Q.LHA + 8.009292
    if Q.tau1 >= 0.05356915:
        z += 22.51753 * Q.tau1 - 1.206245
    if Q.sum_z_dr2 < 0.002635418:
        z += 273.4847 * Q.sum_z_dr2 - 0.7207465
    if Q.sum_z_dr2 >= 0.007520088:
        z += 463.6855 * Q.sum_z_dr2 - 3.486956
    if Q.mass < 76.6557:
        z += -0.01399708 * Q.mass + 1.072956
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -291.7416 * Q.sum_z_dr2_top3 + 0.6277019
    if Q.C2_b2 >= 0.009032972:
        z += 35.19322 * Q.C2_b2 - 0.3178994
    if Q.lam2 > 0.0003061234 and Q.sj3_pairmax_over_m > 0.6745796:
        z += 716.2866 * (Q.lam2 - 0.0003061234) * (Q.sj3_pairmax_over_m - 0.6745796)
    if Q.sj3_pair_mass_min > 11.051 and Q.n_dr_0p2_0p4 > 0.0:
        z += 0.003081681 * (Q.sj3_pair_mass_min - 11.051) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.07806696 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.dr_7 > 0.1078355:
        z += 191.5926 * (0.0211821 - Q.zdr_0) * (Q.dr_7 - 0.1078355)
    if Q.zdr_0 < 0.0211821 and Q.centroid_offset > 0.01258764:
        z += 1975.285 * (0.0211821 - Q.zdr_0) * (Q.centroid_offset - 0.01258764)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.06454549 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -528.4319 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -20.5213 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 122.5889 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += -95671.72 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -168.5984 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.lam1 > 0.00733008 and Q.D2_b2 > 1.129616:
        z += -47.56468 * (Q.lam1 - 0.00733008) * (Q.D2_b2 - 1.129616)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.708627
    if Q.planar_flow < 0.2534037:
        z += -4.863579 * Q.planar_flow + 1.232449
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += 4.30599 * Q.sj2_dr - 0.7659465
    if Q.sj2_dr >= 0.2687922:
        z += 18.67504 * Q.sj2_dr - 4.628235
    if Q.sum_z_dr2 < 0.004372139:
        z += -259.6636 * Q.sum_z_dr2 + 4.638364
    if 0.004372139 <= Q.sum_z_dr2 < 0.01323868:
        z += -395.0899 * Q.sum_z_dr2 + 5.230467
    if Q.tau1 < 0.09538712:
        z += 20.27272 * Q.tau1 - 1.933756
    if Q.LHA < 0.3127275:
        z += -0.7480587 * Q.LHA + 0.2339385
    if Q.mass < 15.45403:
        z += -0.09468635 * Q.mass - 0.1562287
    if 15.45403 <= Q.mass < 69.61135:
        z += 0.02990389 * Q.mass - 2.08165
    if Q.centroid_offset < 0.01437952:
        z += -31.80195 * Q.centroid_offset + 2.195998
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -74.36234 * Q.centroid_offset + 2.807996
    if Q.centroid_offset >= 0.04990367:
        z += -199.0388 * Q.centroid_offset + 9.932767
    if Q.lam1 < 0.0005049491:
        z += 1403.141 * Q.lam1 - 5.268711
    if 0.0005049491 <= Q.lam1 < 0.00733008:
        z += 596.4268 * Q.lam1 - 4.861361
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 468.2057 * Q.lam1 - 3.92149
    if Q.sj3_dr_max < 0.1426152:
        z += 11.5946 * Q.sj3_dr_max - 1.35424
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 28.58059 * Q.sj3_dr_max - 3.7767
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -11.301 * Q.sj3_dr_max + 2.964446
    if Q.lam1_plus_lam2 < 0.008678045:
        z += -1354.334 * Q.lam1_plus_lam2 + 11.75297
    if Q.sum_z_dr < 0.02054282:
        z += 152.8198 * Q.sum_z_dr - 5.469243
    if 0.02054282 <= Q.sum_z_dr < 0.0717028:
        z += 45.54135 * Q.sum_z_dr - 3.265442
    if Q.sum_zz_dr2 < 0.008168571:
        z += 416.0058 * Q.sum_zz_dr2 - 3.398173
    if Q.z_7 >= 0.01685855:
        z += 25.31677 * Q.z_7 - 0.426804
    if Q.max_pair_mass >= 33.3761:
        z += -0.04550501 * Q.max_pair_mass + 1.51878
    if Q.pt_7 >= 29.04219:
        z += -0.02520017 * Q.pt_7 + 0.7318681
    if Q.eccentricity >= 0.9884745:
        z += 55.0228 * Q.eccentricity - 54.38864
    if Q.C2 < 0.03578649:
        z += 9.40778 * Q.C2 - 0.3366714
    if Q.max_dr < 0.1452311:
        z += -6.922196 * Q.max_dr + 1.005318
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01356538 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1613649 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -263551.4 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.1694464 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += 12.29744 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -508.7828 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.mass < 49.6681 and Q.dr1_7 > 0.1792439:
        z += 0.354986 * (49.6681 - Q.mass) * (Q.dr1_7 - 0.1792439)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -635.3674 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 62.77185 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += -1482.173 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    if Q.eccentricity > 0.9884745 and Q.sj2_mass2 < 2.771069:
        z += -11.5951 * (Q.eccentricity - 0.9884745) * (2.771069 - Q.sj2_mass2)
    if Q.eccentricity > 0.9884745 and Q.eta_0 < 0.02130127:
        z += -474.3944 * (Q.eccentricity - 0.9884745) * (0.02130127 - Q.eta_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.712185
    if Q.sum_z_dr2 >= 0.01882765:
        z += 378.6626 * Q.sum_z_dr2 - 7.129328
    if Q.mass >= 88.15578:
        z += 0.06103814 * Q.mass - 5.380865
    if Q.mass_over_sum_pt >= 0.1309286:
        z += -49.73314 * Q.mass_over_sum_pt + 6.511492
    if Q.zdr_0 >= 0.03981924:
        z += -25.42266 * Q.zdr_0 + 1.012311
    if Q.sum_z_dr2_top2 >= 0.01403324:
        z += -30.70123 * Q.sum_z_dr2_top2 + 0.4308378
    if Q.sum_z_dr2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += 4318.23 * (Q.sum_z_dr2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -2.20671 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.2941829
    if Q.sum_z_dr < 0.007673833:
        z += 57.03125 * Q.sum_z_dr + 8.1104
    if 0.007673833 <= Q.sum_z_dr < 0.1484084:
        z += -60.73879 * Q.sum_z_dr + 9.014148
    if Q.lam1 < 0.006506576:
        z += -287.7079 * Q.lam1 + 3.185342
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -132.2983 * Q.lam1 + 2.174157
    if Q.e3 < 5.334511e-05:
        z += -16531.87 * Q.e3 + 0.8818944
    if Q.e2 < 0.08000524:
        z += 37.91047 * Q.e2 - 3.033036
    if Q.mass >= 49.6681:
        z += -0.01034574 * Q.mass + 0.513853
    if Q.z_5 < 0.02818362:
        z += 212.2533 * Q.z_5 - 5.982067
    if Q.z_7 < 0.02807091:
        z += 143.7366 * Q.z_7 - 4.034818
    if Q.sum_pt_top5 >= 791.125:
        z += 0.004106362 * Q.sum_pt_top5 - 3.248646
    if Q.sj3_dr23 >= 0.1974628:
        z += -2.586075 * Q.sj3_dr23 + 0.5106537
    if Q.log_sum_pt >= 6.502799:
        z += -0.5500508 * Q.log_sum_pt + 3.57687
    if Q.pt_7 >= 48.71875:
        z += -0.04094033 * Q.pt_7 + 1.994561
    if Q.lam1_plus_lam2 < 0.007520088:
        z += 138.3813 * Q.lam1_plus_lam2 - 0.4954739
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -95.33222 * Q.lam1_plus_lam2 + 1.262072
    if Q.z_6 < 0.02160287:
        z += 149.1059 * Q.z_6 - 3.221116
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -68.13593 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.9586129 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.z_7 > 0.02320757:
        z += -0.152491 * (Q.sum_pt_top5 - 658.125) * (Q.z_7 - 0.02320757)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -866191.1 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0007050781 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -3.317419 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.sum_z_dr < 0.1484084 and Q.z_7 > 0.06164517:
        z += 376.2133 * (0.1484084 - Q.sum_z_dr) * (Q.z_7 - 0.06164517)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.000537286:
        z += -9978.791 * (0.1484084 - Q.sum_z_dr) * (0.000537286 - Q.lam2)
    if Q.sum_z_dr < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -1.784297 * (0.1484084 - Q.sum_z_dr) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_z_dr < 0.1484084 and Q.M3 < 0.07474969:
        z += 154.833 * (0.1484084 - Q.sum_z_dr) * (0.07474969 - Q.M3)
    if Q.e3 < 5.334511e-05 and Q.D3 > 0.2213841:
        z += 1725.118 * (5.334511e-05 - Q.e3) * (Q.D3 - 0.2213841)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 > 0.008780509:
        z += 344.8033 * (0.1484084 - Q.sum_z_dr) * (Q.tau2 - 0.008780509)
    if Q.mass > 49.6681 and Q.z_dr_0p1_0p2 < 0.4684459:
        z += -0.02255144 * (Q.mass - 49.6681) * (0.4684459 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -197.7295 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.03918186 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.001329877 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.000733421 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.8039097
    if Q.planar_flow < 0.1115136:
        z += -3.280138 * Q.planar_flow + 0.3657799
    if Q.z_dr_0p05_0p1 < 0.5882598:
        z += -0.4033104 * Q.z_dr_0p05_0p1 + 0.2372513
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -13.53662 * Q.z_dr_0p05_0p1 + 10.16478
    if 0.7143804 <= Q.psi_0p1 < 0.8155839:
        z += -1.438508 * Q.psi_0p1 + 1.027642
    if 0.8155839 <= Q.psi_0p1 < 0.9761279:
        z += 0.7981046 * Q.psi_0p1 - 0.7965033
    if Q.psi_0p1 >= 0.9761279:
        z += -15.80547 * Q.psi_0p1 + 15.41071
    if 0.007520088 <= Q.sum_z_dr2 < 0.008678045:
        z += -781.1536 * Q.sum_z_dr2 + 5.874344
    if Q.sum_z_dr2 >= 0.008678045:
        z += -564.8674 * Q.sum_z_dr2 + 3.997402
    if 0.002074109 <= Q.sum_zz_dr2 < 0.0030133:
        z += 281.6466 * Q.sum_zz_dr2 - 0.5841658
    if 0.0030133 <= Q.sum_zz_dr2 < 0.01165737:
        z += 23.55524 * Q.sum_zz_dr2 + 0.193541
    if Q.sum_zz_dr2 >= 0.01165737:
        z += -266.267 * Q.sum_zz_dr2 + 3.572107
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.005590289:
        z += 317.4199 * Q.lam1_plus_lam2 - 1.130844
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.006679471:
        z += 468.0139 * Q.lam1_plus_lam2 - 1.972708
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -987.0707 * Q.lam1_plus_lam2 + 7.746488
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -974.0257 * Q.lam1_plus_lam2 + 7.57379
    if 0.01655442 <= Q.e2 < 0.03556091:
        z += -119.3339 * Q.e2 + 1.975504
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -127.5068 * Q.e2 + 2.266138
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += -67.19403 * Q.e2 - 0.213302
    if Q.e2 >= 0.05028464:
        z += 144.1033 * Q.e2 - 10.83831
    if 0.02685622 <= Q.centroid_offset < 0.04990367:
        z += -36.96931 * Q.centroid_offset + 0.9928561
    if Q.centroid_offset >= 0.04990367:
        z += -394.9118 * Q.centroid_offset + 18.8555
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 121.9277 * Q.mass_over_sum_pt - 9.744915
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 298.7933 * Q.mass_over_sum_pt - 24.73456
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 224.7565 * Q.mass_over_sum_pt - 18.04061
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1259586 * Q.n_dr_0_0p05 + 0.629793
    if Q.e3 < 5.334511e-05:
        z += 15384.62 * Q.e3 - 0.8206944
    if Q.sd_mass < 49.91626:
        z += -0.005773176 * Q.sd_mass - 0.4427915
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.02964136 * Q.sd_mass - 2.210553
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 55.41953 * Q.sum_z_dr - 1.490562
    if 0.04081947 <= Q.sum_z_dr < 0.08065885:
        z += 98.97441 * Q.sum_z_dr - 3.26845
    if 0.08065885 <= Q.sum_z_dr < 0.08723651:
        z += 113.622 * Q.sum_z_dr - 4.449911
    if Q.sum_z_dr >= 0.08723651:
        z += -125.907 * Q.sum_z_dr + 16.44577
    if Q.mass >= 69.61135:
        z += -0.0282684 * Q.mass + 1.967802
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -4.722065 * Q.sj2_dr + 0.2906023
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -18.07711 * Q.sj2_dr + 2.019951
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 20.92273 * Q.sj2_dr - 4.187704
    if Q.sj2_dr >= 0.2001708:
        z += 8.220213 * Q.sj2_dr - 1.645031
    if Q.lam2 < 0.000537286:
        z += -746.4981 * Q.lam2 + 0.401083
    if Q.C2_b2 < 0.004032342:
        z += 171.9601 * Q.C2_b2 - 0.693402
    if Q.LHA >= 0.3033137:
        z += 33.91576 * Q.LHA - 10.28712
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -4934.999 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 1081.289 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 3134.772 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.02305214 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -480.9234 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.sum_zz_dr2 > 0.002074109 and Q.sj2_dr < 0.1872617:
        z += -4803.605 * (Q.sum_zz_dr2 - 0.002074109) * (0.1872617 - Q.sj2_dr)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 13.61807 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.sd_mass < 74.57663 and Q.sj2_mass2 > 3.697444:
        z += 0.001536003 * (74.57663 - Q.sd_mass) * (Q.sj2_mass2 - 3.697444)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 211358.6 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.2001708 and Q.z_5 < 0.1058993:
        z += -95.91051 * (Q.sj2_dr - 0.2001708) * (0.1058993 - Q.z_5)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.3247903
    if Q.N2 < 0.2233283:
        z += 4.095983 * Q.N2 - 0.9147488
    if Q.sum_z_dr2 < 0.007520088:
        z += 853.4423 * Q.sum_z_dr2 - 6.417961
    if Q.mass_over_sum_pt < 0.1309286:
        z += 8.92072 * Q.mass_over_sum_pt - 1.167978
    if Q.lam1_plus_lam2 < 0.00609665:
        z += -147.4595 * Q.lam1_plus_lam2 + 7.136641
    if 0.00609665 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -746.4086 * Q.lam1_plus_lam2 + 10.78822
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += -162.2414 * Q.lam1_plus_lam2 + 3.054626
    if Q.e2 < 0.04110972:
        z += -147.9569 * Q.e2 + 5.781475
    if 0.04110972 <= Q.e2 < 0.06344108:
        z += 13.4784 * Q.e2 - 0.8550844
    if Q.sum_zz_dr2 < 0.008168571:
        z += 864.2217 * Q.sum_zz_dr2 - 7.928201
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 249.0093 * Q.sum_zz_dr2 - 2.902794
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -655.3154 * Q.mass_over_sum_pt_sq + 4.707023
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -167.3549 * Q.sum_z_dr2_top3 + 1.492459
    if 0.002151568 <= Q.sum_z_dr2_top3 < 0.006756161:
        z += -245.9248 * Q.sum_z_dr2_top3 + 1.661507
    if Q.lam1 < 0.002464291:
        z += 133.2625 * Q.lam1 - 0.7984595
    if 0.002464291 <= Q.lam1 < 0.005433361:
        z += -31.50479 * Q.lam1 - 0.3924248
    if 0.005433361 <= Q.lam1 < 0.008375572:
        z += 191.5572 * Q.lam1 - 1.604401
    if Q.sum_z_dr < 0.1019409:
        z += 39.53508 * Q.sum_z_dr - 4.030243
    if Q.sj2_dr < 0.1492731:
        z += -13.49486 * Q.sj2_dr + 1.043931
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -25.20583 * Q.sj2_dr + 2.792063
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 29.756 * Q.sj2_dr - 5.956283
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += -3997.986 * Q.sum_z_dr2_top2 + 2.048781
    if Q.sj3_pair_mass_max >= 72.58129:
        z += -0.04377646 * Q.sj3_pair_mass_max + 3.177352
    if Q.lam2 < 0.0003061234:
        z += 2616.752 * Q.lam2 - 0.8010489
    if Q.centroid_offset < 0.01837778:
        z += 54.15305 * Q.centroid_offset - 0.9952129
    if Q.LHA < 0.3033137:
        z += -1.031688 * Q.LHA + 0.3129252
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -11.3982 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.max_dr < 0.121681:
        z += -46.06171 * (0.2233283 - Q.N2) * (0.121681 - Q.max_dr)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 < 716.8828:
        z += 0.0176231 * (0.2233283 - Q.N2) * (716.8828 - Q.sum_pt_top5)
    if Q.sum_z_dr2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -1488.525 * (0.007520088 - Q.sum_z_dr2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 71.52746 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 < 4.0:
        z += 0.5395884 * (0.2233283 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.0390071 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.n_for_90pct < 6.0:
        z += -6.740655 * (0.2233283 - Q.N2) * (6.0 - Q.n_for_90pct)
    if Q.sj3_pair_mass_max > 72.58129 and Q.N3 < 1.666763:
        z += 0.04444965 * (Q.sj3_pair_mass_max - 72.58129) * (1.666763 - Q.N3)
    if Q.lam2 < 0.001130645 and Q.mean_phi > -0.01753483:
        z += -11827.28 * (0.001130645 - Q.lam2) * (Q.mean_phi - -0.01753483)
    if Q.e2 < 0.06344108 and Q.dr12 > 0.119512:
        z += 164.6886 * (0.06344108 - Q.e2) * (Q.dr12 - 0.119512)
    if Q.lam1_plus_lam2 < 0.01882765 and Q.ptdr0_2 > 14.78048:
        z += 6.090909 * (0.01882765 - Q.lam1_plus_lam2) * (Q.ptdr0_2 - 14.78048)
    if Q.centroid_offset < 0.01837778 and Q.dr01 < 0.2512159:
        z += -284.3173 * (0.01837778 - Q.centroid_offset) * (0.2512159 - Q.dr01)
    if Q.centroid_offset < 0.01837778 and Q.zdr_0 < 0.01216943:
        z += -15778.85 * (0.01837778 - Q.centroid_offset) * (0.01216943 - Q.zdr_0)
    if Q.mass_over_sum_pt_sq < 0.007182836 and Q.D2 < 0.7459513:
        z += -1195.414 * (0.007182836 - Q.mass_over_sum_pt_sq) * (0.7459513 - Q.D2)
    if Q.lam1_plus_lam2 < 0.00609665 and Q.D2_b2 < 0.08499387:
        z += -2496.694 * (0.00609665 - Q.lam1_plus_lam2) * (0.08499387 - Q.D2_b2)
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
