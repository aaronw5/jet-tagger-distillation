"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's neuron values (from 100 if-statements per neuron, pruned; all observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.9% (the network: 65.8%); same class as the network for 88.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
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
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_2               |Δφ| of particle 2
  Q.absphi_7               |Δφ| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_6                  ΔR between particle 6 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_2                  Δη of particle 2
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
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
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
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
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        z_2nd=zs[1],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_0=abs(eta[0]),
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        absphi_2=abs(phi[2]),
        absphi_7=abs(phi[7]),
        sj2_dr=subjets(2)["dr"][0],
        dr1_6=math.sqrt(dist2(1, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_2=eta[2],
        phi_0=phi[0],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
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
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -1.744741
    if Q.planar_flow < 0.1484197:
        z += -9.331172 * Q.planar_flow + 1.38493
    if Q.width < 0.004372139:
        z += 189.0308 * Q.width + 1.340923
    if 0.004372139 <= Q.width < 0.008678045:
        z += -503.3534 * Q.width + 4.368123
    if Q.girth2 < 0.01323868:
        z += -379.7621 * Q.girth2 + 5.258095
    if 0.01323868 <= Q.girth2 < 0.01882765:
        z += -41.25042 * Q.girth2 + 0.7766486
    if Q.mass < 21.78408:
        z += 0.5727882 * Q.mass - 14.52017
    if 21.78408 <= Q.mass < 56.92035:
        z += 0.0502959 * Q.mass - 3.138161
    if 56.92035 <= Q.mass < 64.61873:
        z += 0.03576088 * Q.mass - 2.310823
    if Q.girth2_top3 < 0.006756161:
        z += 69.84865 * Q.girth2_top3 - 0.7196915
    if 0.006756161 <= Q.girth2_top3 < 0.007929074:
        z += 211.2542 * Q.girth2_top3 - 1.675051
    if Q.sum_pt >= 901.5938:
        z += -0.01300942 * Q.sum_pt + 11.72921
    if Q.C2_b2 < 0.001563465:
        z += 583.3875 * Q.C2_b2 - 0.9121061
    if Q.sj3_dr_max < 0.04889979:
        z += 16.29694 * Q.sj3_dr_max - 0.03897357
    if 0.04889979 <= Q.sj3_dr_max < 0.1070199:
        z += -3.004113 * Q.sj3_dr_max + 0.9048436
    if 0.1070199 <= Q.sj3_dr_max < 0.1789613:
        z += 15.34993 * Q.sj3_dr_max - 1.059405
    if 0.1789613 <= Q.sj3_dr_max < 0.233678:
        z += 1.079973 * Q.sj3_dr_max + 1.494366
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -7.617602 * Q.sj3_dr_max + 3.526798
    if Q.sj3_dr_max >= 0.3012016:
        z += -4.613489 * Q.sj3_dr_max + 2.621954
    if 0.006789738 <= Q.centroid_offset < 0.02076709:
        z += -43.55109 * Q.centroid_offset + 0.2957005
    if 0.02076709 <= Q.centroid_offset < 0.04990367:
        z += -53.57576 * Q.centroid_offset + 0.5038838
    if Q.centroid_offset >= 0.04990367:
        z += -400.9505 * Q.centroid_offset + 17.83916
    if Q.girth < 0.07608178:
        z += 79.5653 * Q.girth - 6.415867
    if 0.07608178 <= Q.girth < 0.08723651:
        z += 32.48821 * Q.girth - 2.834158
    if Q.e2 < 0.0245477:
        z += -55.36995 * Q.e2 + 1.781749
    if 0.0245477 <= Q.e2 < 0.03556091:
        z += -38.36706 * Q.e2 + 1.364368
    if Q.girth2_top5 < 0.008329695:
        z += 53.19402 * Q.girth2_top5 - 0.44309
    if Q.lam1 < 0.0002758826:
        z += 5136.053 * Q.lam1 - 2.668865
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 242.7384 * Q.lam1 - 1.318885
    if Q.log_sum_pt < 6.080494:
        z += 2.605509 * Q.log_sum_pt - 15.84278
    if 6.377723 <= Q.log_sum_pt < 6.670067:
        z += 1.859149 * Q.log_sum_pt - 11.85714
    if Q.log_sum_pt >= 6.670067:
        z += -2.071349 * Q.log_sum_pt + 14.35955
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += -336.9326 * Q.mass_over_sum_pt_sq + 1.315585
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.2221309 * Q.n_dr_0_0p05 - 0.8885236
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.005365134 * Q.sum_pt_top5 - 3.688194
    if Q.lam2 < 7.300726e-05:
        z += 9757.155 * Q.lam2 - 0.7123432
    if Q.zdr_1 < 0.01587232:
        z += -13.38891 * Q.zdr_1 + 0.212513
    if Q.mass_over_sum_pt < 0.08475161:
        z += 14.33782 * Q.mass_over_sum_pt - 0.8735965
    if 0.08475161 <= Q.mass_over_sum_pt < 0.1309286:
        z += -7.39668 * Q.mass_over_sum_pt + 0.9684372
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -968.7307 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.D2 < 1.002471:
        z += 98.73473 * (0.01323868 - Q.girth2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.0419644 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.girth2 < 0.01323868 and Q.mean_phi < -0.003096655:
        z += -2726.037 * (0.01323868 - Q.girth2) * (-0.003096655 - Q.mean_phi)
    if Q.girth2 < 0.01882765 and Q.mass_top2 > 28.78966:
        z += 3.290435 * (0.01882765 - Q.girth2) * (Q.mass_top2 - 28.78966)
    if Q.girth2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -2080.407 * (0.01323868 - Q.girth2) * (Q.centroid_offset - 0.01837778)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 90.39271 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 51422.09 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -43.44248 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.mass < 56.92035 and Q.dr_max_012 > 0.06112084:
        z += -0.1851523 * (56.92035 - Q.mass) * (Q.dr_max_012 - 0.06112084)
    if Q.log_sum_pt > 6.377723 and Q.mean_eta > 6.288824e-05:
        z += -49.49885 * (Q.log_sum_pt - 6.377723) * (Q.mean_eta - 6.288824e-05)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_4 < 0.06336451:
        z += -0.08717468 * (Q.sum_pt_top5 - 687.4375) * (0.06336451 - Q.dr_4)
    if Q.lam2 < 7.300726e-05 and Q.dr_2 < 0.08525808:
        z += 57387.48 * (7.300726e-05 - Q.lam2) * (0.08525808 - Q.dr_2)
    if Q.sum_pt > 901.5938 and Q.pt_7 > 33.21875:
        z += 0.0003115628 * (Q.sum_pt - 901.5938) * (Q.pt_7 - 33.21875)
    if Q.log_sum_pt > 6.670067 and Q.z_7 > 0.01685855:
        z += -308.0568 * (Q.log_sum_pt - 6.670067) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.3012016 and Q.z_7 < 0.05557716:
        z += -120.1022 * (0.3012016 - Q.sj3_dr_max) * (0.05557716 - Q.z_7)
    if Q.sj3_dr_max < 0.3012016 and Q.n_dr_0p05_0p1 < 2.0:
        z += -1.419817 * (0.3012016 - Q.sj3_dr_max) * (2.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt > 6.670067 and Q.m01 < 28.78966:
        z += 0.09986198 * (Q.log_sum_pt - 6.670067) * (28.78966 - Q.m01)
    if Q.centroid_offset > 0.02076709 and Q.z_7 < 0.06473447:
        z += -1541.603 * (Q.centroid_offset - 0.02076709) * (0.06473447 - Q.z_7)
    if Q.planar_flow < 0.1484197 and Q.pt_dispersion < 0.4160096:
        z += 73.3605 * (0.1484197 - Q.planar_flow) * (0.4160096 - Q.pt_dispersion)
    if Q.centroid_offset > 0.006789738 and Q.n_pt_above_50 < 7.0:
        z += 3.641317 * (Q.centroid_offset - 0.006789738) * (7.0 - Q.n_pt_above_50)
    if Q.width < 0.004372139 and Q.C3 < 0.03532852:
        z += 7754.2 * (0.004372139 - Q.width) * (0.03532852 - Q.C3)
    if Q.centroid_offset > 0.006789738 and Q.pt_1 < 223.375:
        z += 0.1013279 * (Q.centroid_offset - 0.006789738) * (223.375 - Q.pt_1)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.05978096
    if Q.lam1 < 0.0008722282:
        z += 711.9613 * Q.lam1 + 1.375574
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -377.3039 * Q.lam1 + 2.325662
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -32.68258 * Q.lam1 + 0.2737353
    if Q.pt_7 < 25.57812:
        z += 0.05322682 * Q.pt_7 - 1.361442
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.06816578 * Q.pt_7 - 2.35385
    if Q.pt_7 >= 53.4375:
        z += -0.04709513 * Q.pt_7 + 3.805406
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 4.436117 * Q.log_sum_pt - 28.29233
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 12.83175 * Q.log_sum_pt - 82.88747
    if Q.log_sum_pt >= 6.605974:
        z += 16.442 * Q.log_sum_pt - 106.7366
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -131.869 * Q.mass_over_sum_pt_sq + 0.7691831
    if Q.z_7 < 0.02320757:
        z += 36.11369 * Q.z_7 - 2.226234
    if 0.02320757 <= Q.z_7 < 0.04624032:
        z += 65.12553 * Q.z_7 - 2.899529
    if 0.04624032 <= Q.z_7 < 0.06164517:
        z += 36.12955 * Q.z_7 - 1.558745
    if Q.z_7 >= 0.06164517:
        z += 0.0158596 * Q.z_7 + 0.6674894
    if Q.girth2 < 0.005019719:
        z += 491.0678 * Q.girth2 - 3.85262
    if 0.005019719 <= Q.girth2 < 0.008678045:
        z += 379.2986 * Q.girth2 - 3.29157
    if Q.mass < 49.6681:
        z += -0.03531203 * Q.mass + 1.753881
    if Q.sj3_dr_min >= 0.03628191:
        z += -2.00522 * Q.sj3_dr_min + 0.07275322
    if Q.sj3_dr_max >= 0.169029:
        z += 2.991179 * Q.sj3_dr_max - 0.505596
    if Q.e2_sq < 0.008168571:
        z += -92.67581 * Q.e2_sq + 0.7570289
    if Q.e3 < 1.627072e-05:
        z += -34886.64 * Q.e3 + 0.9326764
    if 1.627072e-05 <= Q.e3 < 0.0005116989:
        z += -736.8284 * Q.e3 + 0.3770343
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.002297974 * Q.sum_pt_top5 - 0.844721
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.01151456 * Q.sum_pt_top5 + 6.492327
    if Q.width < 0.00609665:
        z += 654.9177 * Q.width - 3.992804
    if Q.girth < 0.0717028:
        z += -11.18569 * Q.girth + 0.1932499
    if 0.0717028 <= Q.girth < 0.1019409:
        z += 20.13336 * Q.girth - 2.052413
    if Q.centroid_offset < 0.01837778:
        z += 18.51919 * Q.centroid_offset - 0.3403415
    if Q.tau1 < 0.07283629:
        z += -7.319406 * Q.tau1 + 0.5331183
    if Q.zdr_0 < 0.0211821:
        z += 10.43484 * Q.zdr_0 - 0.2210319
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -1397.391 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.log_sum_pt > 6.377723 and Q.e3 < 1.627072e-05:
        z += -125697.9 * (Q.log_sum_pt - 6.377723) * (1.627072e-05 - Q.e3)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += 1.993386 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.tau2 < 0.06297984:
        z += 661.2532 * (0.06164517 - Q.z_7) * (0.06297984 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -23.6902 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.2082539 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -3.227843 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -28.17985 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.006472215 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.006079252 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 64.52834 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 3.689908 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -724.2019 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.z_7 < 0.06164517 and Q.centroid_offset > 0.02076709:
        z += -822.3152 * (0.06164517 - Q.z_7) * (Q.centroid_offset - 0.02076709)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.05220105 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -48.42826 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.log_sum_pt > 6.605974 and Q.z_4 < 0.09925997:
        z += 22.10863 * (Q.log_sum_pt - 6.605974) * (0.09925997 - Q.z_4)
    if Q.girth2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 11379.42 * (0.008678045 - Q.girth2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -2191.013 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 1295.831 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    if Q.z_7 < 0.06164517 and Q.girth2_top2 < 0.01403324:
        z += 870.9285 * (0.06164517 - Q.z_7) * (0.01403324 - Q.girth2_top2)
    if Q.log_sum_pt > 6.502799 and Q.zdr_6 < 0.008654951:
        z += -579.3842 * (Q.log_sum_pt - 6.502799) * (0.008654951 - Q.zdr_6)
    if Q.girth < 0.1019409 and Q.pair_mass_0_6 > 4.037975:
        z += -0.3865097 * (0.1019409 - Q.girth) * (Q.pair_mass_0_6 - 4.037975)
    if Q.sum_pt_top5 > 531.1875 and Q.zdr_6 < 0.01392641:
        z += 0.2956778 * (Q.sum_pt_top5 - 531.1875) * (0.01392641 - Q.zdr_6)
    if Q.z_7 < 0.06164517 and Q.abseta_0 < 0.1057739:
        z += 84.64143 * (0.06164517 - Q.z_7) * (0.1057739 - Q.abseta_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.8786
    if Q.sj3_pair_mass_max < 62.55:
        z += -0.006114776 * Q.sj3_pair_mass_max + 0.3824792
    if Q.log_sum_pt < 6.46415:
        z += -5.539527 * Q.log_sum_pt + 36.44464
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -4.486596 * Q.log_sum_pt + 29.63834
    if Q.log_sum_pt >= 6.842717:
        z += 10.00973 * Q.log_sum_pt - 68.49375
    if Q.lam1 < 0.001503553:
        z += -381.9402 * Q.lam1 + 1.299491
    if 0.001503553 <= Q.lam1 < 0.00595415:
        z += -162.9497 * Q.lam1 + 0.9702269
    if Q.girth < 0.007673833:
        z += 325.0483 * Q.girth - 2.494366
    if Q.pt_7 < 34.53125:
        z += 0.07666316 * Q.pt_7 - 4.096688
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.1075458 * Q.pt_7 - 5.163103
    if Q.pt_7 >= 53.4375:
        z += 0.03088261 * Q.pt_7 - 1.066415
    if Q.LHA >= 0.1329373:
        z += -8.464454 * Q.LHA + 1.125242
    if Q.z_7 < 0.03629544:
        z += -14.68091 * Q.z_7 + 1.049519
    if 0.03629544 <= Q.z_7 < 0.04939969:
        z += -36.47724 * Q.z_7 + 1.840626
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -48.09197 * Q.z_7 + 2.41439
    if Q.z_7 >= 0.07148865:
        z += -33.41105 * Q.z_7 + 1.364871
    if 752.1 <= Q.sum_pt_top5 < 839.9547:
        z += -0.00632576 * Q.sum_pt_top5 + 4.757604
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.01085375 * Q.sum_pt_top5 + 8.560913
    if Q.planar_flow < 0.4926918:
        z += 0.4030258 * Q.planar_flow - 0.1985675
    if Q.mass < 36.22941:
        z += 0.07739159 * Q.mass - 2.803852
    if Q.max_dr < 0.2507612:
        z += -2.611167 * Q.max_dr + 0.6547794
    if Q.girth2 < 0.003562611:
        z += -219.838 * Q.girth2 + 0.7831974
    if Q.sum_pt < 527.1781:
        z += -0.006191706 * Q.sum_pt + 5.03643
    if 527.1781 <= Q.sum_pt < 813.4156:
        z += -0.002876775 * Q.sum_pt + 3.288871
    if 813.4156 <= Q.sum_pt < 868.5094:
        z += 0.00331493 * Q.sum_pt - 1.747559
    if Q.sum_pt >= 868.5094:
        z += 0.008368987 * Q.sum_pt - 6.137054
    if Q.zdr_0 < 0.0211821:
        z += -14.42848 * Q.zdr_0 + 0.3056256
    if Q.mass_over_sum_pt < 0.1079857:
        z += -11.82156 * Q.mass_over_sum_pt + 1.276559
    if Q.e2 < 0.009668065:
        z += -74.67566 * Q.e2 + 0.7219691
    if Q.lam2 < 0.0001947983:
        z += -1818.571 * Q.lam2 + 0.3542546
    if Q.absphi_2 < 0.01573181:
        z += -37.50888 * Q.absphi_2 + 0.5900826
    if Q.sj3_pair_mass_max < 62.55 and Q.z_7 < 0.06810151:
        z += -0.4644062 * (62.55 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 62.55 and Q.centroid_offset > 0.01096064:
        z += -0.5375006 * (62.55 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.08050702:
        z += -1314.985 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.08050702)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += -0.2267077 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.lam1 < 0.00595415 and Q.pt_6 > 19.46875:
        z += 3.659349 * (0.00595415 - Q.lam1) * (Q.pt_6 - 19.46875)
    if Q.girth < 0.007673833 and Q.pt_4 < 75.625:
        z += 5.532511 * (0.007673833 - Q.girth) * (75.625 - Q.pt_4)
    if Q.mass < 36.22941 and Q.pt_5 < 73.75:
        z += 0.0005526588 * (36.22941 - Q.mass) * (73.75 - Q.pt_5)
    if Q.log_sum_pt < 6.46415 and Q.D2_b2 < 1.129616:
        z += 1.813132 * (6.46415 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.pt_7 < 53.4375 and Q.D2_b2 < 1.129616:
        z += -0.01448024 * (53.4375 - Q.pt_7) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.z_dr_0p05_0p1 < 0.4474937:
        z += 3.940323 * (Q.log_sum_pt - 6.842717) * (0.4474937 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top5 > 752.1 and Q.D2_b2 < 1.129616:
        z += 0.01125901 * (Q.sum_pt_top5 - 752.1) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.D2_b2 < 1.345805:
        z += -12.73145 * (Q.log_sum_pt - 6.842717) * (1.345805 - Q.D2_b2)
    if Q.sum_pt_top5 > 752.1 and Q.abseta_0 < 0.0174408:
        z += -0.4582177 * (Q.sum_pt_top5 - 752.1) * (0.0174408 - Q.abseta_0)
    if Q.log_sum_pt > 6.842717 and Q.abseta_0 < 0.0174408:
        z += 529.0355 * (Q.log_sum_pt - 6.842717) * (0.0174408 - Q.abseta_0)
    if Q.sum_pt > 527.1781 and Q.lam2 < 0.0001947983:
        z += -15.4838 * (Q.sum_pt - 527.1781) * (0.0001947983 - Q.lam2)
    if Q.log_sum_pt > 6.842717 and Q.lam2 < 0.0001947983:
        z += 41081.79 * (Q.log_sum_pt - 6.842717) * (0.0001947983 - Q.lam2)
    if Q.z_7 > 0.03629544 and Q.C2_b2 < 0.004032342:
        z += -4468.268 * (Q.z_7 - 0.03629544) * (0.004032342 - Q.C2_b2)
    if Q.pt_7 > 43.5 and Q.C2_b2 < 0.004032342:
        z += 6.312705 * (Q.pt_7 - 43.5) * (0.004032342 - Q.C2_b2)
    if Q.sum_pt > 527.1781 and Q.absphi_2 < 0.01922607:
        z += -0.1239574 * (Q.sum_pt - 527.1781) * (0.01922607 - Q.absphi_2)
    if Q.log_sum_pt < 6.605974 and Q.absphi_2 < 0.01573181:
        z += -90.2898 * (6.605974 - Q.log_sum_pt) * (0.01573181 - Q.absphi_2)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.2173254
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 6.857695 * Q.mass_over_sum_pt - 0.4672771
    if Q.centroid_offset >= 0.01096064:
        z += -28.81315 * Q.centroid_offset + 0.3158106
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -16.92002 * Q.tau1 + 0.9063912
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += -26.50875 * Q.tau1 + 1.89177
    if Q.tau1 >= 0.1136369:
        z += 8.733483 * Q.tau1 - 2.113048
    if 0.04081947 <= Q.girth < 0.07608178:
        z += 52.5201 * Q.girth - 2.143843
    if 0.07608178 <= Q.girth < 0.08723651:
        z += 25.73458 * Q.girth - 0.1059531
    if 0.08723651 <= Q.girth < 0.1019409:
        z += 86.89738 * Q.girth - 5.441583
    if Q.girth >= 0.1019409:
        z += -24.32464 * Q.girth + 5.896494
    if 0.005590289 <= Q.width < 0.007520088:
        z += 216.4891 * Q.width - 1.210237
    if 0.007520088 <= Q.width < 0.01323868:
        z += 820.443 * Q.width - 5.752023
    if Q.width >= 0.01323868:
        z += 702.0368 * Q.width - 4.184482
    if 0.04447357 <= Q.e2 < 0.06344108:
        z += 33.57561 * Q.e2 - 1.493227
    if Q.e2 >= 0.06344108:
        z += -7.707851 * Q.e2 + 1.12584
    if Q.girth2 < 0.004372139:
        z += 1134.486 * Q.girth2 - 4.96013
    if 0.008678045 <= Q.girth2 < 0.01882765:
        z += -1467.458 * Q.girth2 + 12.73467
    if Q.girth2 >= 0.01882765:
        z += -1356.497 * Q.girth2 + 10.64554
    if 0.1492731 <= Q.sj2_dr < 0.1872617:
        z += -9.773478 * Q.sj2_dr + 1.458917
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += 13.95608 * Q.sj2_dr - 2.984721
    if Q.sj2_dr >= 0.2687922:
        z += -7.044771 * Q.sj2_dr + 2.660144
    if Q.mean_eta < -0.004664942:
        z += -13.1634 * Q.mean_eta - 0.06140648
    if Q.mean_eta >= 0.01772426:
        z += 10.36514 * Q.mean_eta - 0.1837144
    if Q.girth2_top5 < 0.008329695:
        z += -32.04894 * Q.girth2_top5 + 0.2669579
    if Q.max_dr >= 0.1027585:
        z += 6.306811 * Q.max_dr - 0.6480786
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 671.7057 * Q.lam1 - 5.625919
    if Q.lam1 >= 0.01200373:
        z += 622.992 * Q.lam1 - 5.041174
    if Q.lam2 >= 0.001130645:
        z += 468.8245 * Q.lam2 - 0.5300739
    if Q.mass >= 64.61873:
        z += -0.05101338 * Q.mass + 3.29642
    if Q.sd_mass >= 62.55:
        z += 0.01124333 * Q.sd_mass - 0.7032703
    if Q.sj3_pair_mass_min >= 11.051:
        z += -0.06078038 * Q.sj3_pair_mass_min + 0.6716841
    if 0.1986272 <= Q.sj3_dr_max < 0.3012016:
        z += 7.427936 * Q.sj3_dr_max - 1.47539
    if Q.sj3_dr_max >= 0.3012016:
        z += -0.1034164 * Q.sj3_dr_max + 0.7930651
    if Q.n_dr_0_0p05 < 1.0:
        z += -0.2085228 * Q.n_dr_0_0p05 + 0.2085228
    if Q.z_dr_0_0p05 < 0.9008535:
        z += 0.4129739 * Q.z_dr_0_0p05 - 0.3720289
    if Q.mass_over_sum_pt_sq < 0.00817466:
        z += -0.2553101 * Q.mass_over_sum_pt_sq - 1.024338
    if 0.00817466 <= Q.mass_over_sum_pt_sq < 0.0116609:
        z += 294.4215 * Q.mass_over_sum_pt_sq - 3.433221
    if Q.LHA >= 0.3467135:
        z += 30.84236 * Q.LHA - 10.69346
    if Q.planar_flow < 0.03320012:
        z += -11.80591 * Q.planar_flow + 0.3919576
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += 0.6668679 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += -0.4276835 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.girth > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += -13.46281 * (Q.girth - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.06320055 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -287.579 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.centroid_offset > 0.01096064 and Q.n_pt_above_50 < 8.0:
        z += 3.642586 * (Q.centroid_offset - 0.01096064) * (8.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 203.8117 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 5.746798 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.55 and Q.D2_b2 < 0.9206502:
        z += 0.03632722 * (Q.sd_mass - 62.55) * (0.9206502 - Q.D2_b2)
    if Q.max_dr > 0.1027585 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -3.550225 * (Q.max_dr - 0.1027585) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.sj2_dr > 0.2687922 and Q.z_dr_0p05_0p1 < 0.9641201:
        z += 23.2623 * (Q.sj2_dr - 0.2687922) * (0.9641201 - Q.z_dr_0p05_0p1)
    if Q.sj2_dr > 0.1872617 and Q.z_dr_0p05_0p1 < 0.9641201:
        z += -12.34056 * (Q.sj2_dr - 0.1872617) * (0.9641201 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0_0p05 < 1.0 and Q.n_dr_0p05_0p1 < 4.0:
        z += 0.1723668 * (1.0 - Q.n_dr_0_0p05) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.max_dr > 0.1027585 and Q.eta_2 > -0.04544525:
        z += -2.114504 * (Q.max_dr - 0.1027585) * (Q.eta_2 - -0.04544525)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.3679507
    if Q.N2 < 0.2233283:
        z += -33.60714 * Q.N2 + 7.505425
    if Q.lam2 < 0.000537286:
        z += 2025.983 * Q.lam2 - 1.749112
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += 1113.29 * Q.lam2 - 1.258735
    if Q.mass_over_sum_pt < 0.07269073:
        z += 35.348 * Q.mass_over_sum_pt - 2.569472
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -145.5343 * Q.mass_over_sum_pt + 13.15831
    if Q.width < 0.001653836:
        z += -1001.058 * Q.width + 1.655586
    if Q.width >= 0.003562611:
        z += 421.4049 * Q.width - 1.501302
    if Q.e2 < 0.04447357:
        z += 122.6099 * Q.e2 - 6.165393
    if 0.04447357 <= Q.e2 < 0.05028464:
        z += 183.7938 * Q.e2 - 8.886459
    if Q.e2 >= 0.05028464:
        z += 61.18389 * Q.e2 - 2.721066
    if Q.sum_pt < 739.5:
        z += 0.008180348 * Q.sum_pt - 6.049367
    if Q.max_dr < 0.1117619:
        z += -9.700613 * Q.max_dr + 0.288754
    if 0.1117619 <= Q.max_dr < 0.1598486:
        z += 12.13559 * Q.max_dr - 2.151701
    if 0.1598486 <= Q.max_dr < 0.177305:
        z += 25.12744 * Q.max_dr - 4.22843
    if Q.max_dr >= 0.177305:
        z += 12.99185 * Q.max_dr - 2.076729
    if Q.C2 < 0.06729223:
        z += -50.27468 * Q.C2 + 3.383095
    if 44.82259 <= Q.sd_mass < 74.57663:
        z += 0.07404595 * Q.sd_mass - 3.318931
    if Q.sd_mass >= 74.57663:
        z += 0.174548 * Q.sd_mass - 10.81403
    if Q.e3 < 0.0001869378:
        z += -2386.761 * Q.e3 + 0.4461759
    if Q.sj3_dr_max < 0.213399:
        z += 12.23092 * Q.sj3_dr_max - 1.233035
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 13.91884 * Q.sj3_dr_max - 1.593236
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -14.81934 * Q.sj3_dr_max + 5.122244
    if Q.girth2 < 0.008678045:
        z += 588.9812 * Q.girth2 - 5.111205
    if Q.girth2_top5 < 0.001501708:
        z += -236.2662 * Q.girth2_top5 + 1.692659
    if 0.001501708 <= Q.girth2_top5 < 0.007164202:
        z += -270.1779 * Q.girth2_top5 + 1.743584
    if Q.girth2_top5 >= 0.007164202:
        z += -33.91171 * Q.girth2_top5 + 0.05092551
    if Q.centroid_offset < 0.04990367:
        z += 61.5962 * Q.centroid_offset - 3.073876
    if Q.girth2_top2 < 0.007639643:
        z += -197.1138 * Q.girth2_top2 + 1.505879
    if 45.7571 <= Q.mass < 76.6557:
        z += -0.04111379 * Q.mass + 1.881248
    if Q.mass >= 76.6557:
        z += -0.274647 * Q.mass + 19.7829
    if Q.e2_sq < 0.01165737:
        z += -3130.486 * Q.e2_sq + 36.49324
    if Q.tau1 < 0.07283629:
        z += -40.10818 * Q.tau1 + 2.921331
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 2619.898 * Q.mass_over_sum_pt_sq - 30.55038
    if Q.LHA < 0.2809341:
        z += -11.86672 * Q.LHA + 4.114353
    if 0.2809341 <= Q.LHA < 0.3467135:
        z += -20.5456 * Q.LHA + 6.552546
    if Q.LHA >= 0.3467135:
        z += -8.678879 * Q.LHA + 2.438193
    if Q.lam1 < 0.001503553:
        z += 1021.878 * Q.lam1 - 1.536447
    if Q.mass_top5 >= 14.54404:
        z += 0.03582009 * Q.mass_top5 - 0.5209688
    if Q.mean_eta >= 0.02644207:
        z += -37.70103 * Q.mean_eta + 0.9968932
    if Q.M2 < 0.03874536:
        z += 39.98357 * Q.M2 - 1.549178
    if Q.sj3_pair_mass_max >= 80.4:
        z += -0.05526775 * Q.sj3_pair_mass_max + 4.443527
    if Q.N2 < 0.2233283 and Q.mass < 62.55:
        z += -0.3284658 * (0.2233283 - Q.N2) * (62.55 - Q.mass)
    if Q.N2 < 0.2233283 and Q.e2_sq > 0.01165737:
        z += -666.9069 * (0.2233283 - Q.N2) * (Q.e2_sq - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.4028409 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 365.6549 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -49.52686 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1218872:
        z += -42.91831 * (0.2233283 - Q.N2) * (0.1218872 - Q.abseta_7)
    if Q.e3 < 0.0001869378 and Q.mean_eta < -0.01284493:
        z += 238175.0 * (0.0001869378 - Q.e3) * (-0.01284493 - Q.mean_eta)
    if Q.sd_mass > 44.82259 and Q.centroid_offset > 0.001308549:
        z += -1.923679 * (Q.sd_mass - 44.82259) * (Q.centroid_offset - 0.001308549)
    if Q.sd_mass > 44.82259 and Q.sd_zg < 0.275762:
        z += -0.2984794 * (Q.sd_mass - 44.82259) * (0.275762 - Q.sd_zg)
    if Q.width > 0.003562611 and Q.sj3_z3 < 0.1057566:
        z += -661.7228 * (Q.width - 0.003562611) * (0.1057566 - Q.sj3_z3)
    if Q.lam2 < 0.000537286 and Q.dr01 < 0.1410336:
        z += -4353.771 * (0.000537286 - Q.lam2) * (0.1410336 - Q.dr01)
    if Q.sj3_dr_max < 0.3456459 and Q.pair_mass_0_6 > 15.55293:
        z += 0.3410466 * (0.3456459 - Q.sj3_dr_max) * (Q.pair_mass_0_6 - 15.55293)
    if Q.max_dr > 0.1598486 and Q.C2_b2 < 0.0006435798:
        z += -26893.08 * (Q.max_dr - 0.1598486) * (0.0006435798 - Q.C2_b2)
    if Q.girth2_top2 < 0.007639643 and Q.C2_b2 > 0.0006435798:
        z += -14140.01 * (0.007639643 - Q.girth2_top2) * (Q.C2_b2 - 0.0006435798)
    if Q.lam2 < 0.000537286 and Q.mean_phi < -0.01275329:
        z += -103508.0 * (0.000537286 - Q.lam2) * (-0.01275329 - Q.mean_phi)
    if Q.e3 < 0.0001869378 and Q.mean_phi < -0.009352575:
        z += 115574.0 * (0.0001869378 - Q.e3) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += -0.9248613 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.sd_mass > 44.82259 and Q.zdr_5 < 0.005440034:
        z += -6.949303 * (Q.sd_mass - 44.82259) * (0.005440034 - Q.zdr_5)
    if Q.lam2 < 0.000537286 and Q.dr1_6 < 0.07796252:
        z += -7855.08 * (0.000537286 - Q.lam2) * (0.07796252 - Q.dr1_6)
    if Q.mass_top5 > 14.54404 and Q.pt_6 > 42.78125:
        z += 0.0005998812 * (Q.mass_top5 - 14.54404) * (Q.pt_6 - 42.78125)
    if Q.mass > 76.6557 and Q.M3 < 0.107953:
        z += 0.7169371 * (Q.mass - 76.6557) * (0.107953 - Q.M3)
    if Q.sd_mass > 44.82259 and Q.sj3_pair_mass_min < 27.42324:
        z += 0.002880421 * (Q.sd_mass - 44.82259) * (27.42324 - Q.sj3_pair_mass_min)
    if Q.lam2 < 0.000537286 and Q.mean_eta > 0.01271871:
        z += 74103.46 * (0.000537286 - Q.lam2) * (Q.mean_eta - 0.01271871)
    if Q.lam2 < 0.000537286 and Q.absphi_7 < 0.1647949:
        z += -7225.392 * (0.000537286 - Q.lam2) * (0.1647949 - Q.absphi_7)
    if Q.sum_pt < 739.5 and Q.n_for_50pct > 1.0:
        z += 0.0007283893 * (739.5 - Q.sum_pt) * (Q.n_for_50pct - 1.0)
    if Q.sj3_dr_max < 0.3456459 and Q.dr_5 > 0.1351015:
        z += 69.5219 * (0.3456459 - Q.sj3_dr_max) * (Q.dr_5 - 0.1351015)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.3454461
    if Q.LHA < 0.2160559:
        z += -8.471102 * Q.LHA + 1.830231
    if Q.z_7 < 0.02807091:
        z += -155.3281 * Q.z_7 + 5.992215
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -48.1439 * Q.z_7 + 2.983458
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -27.39668 * Q.z_7 + 1.958552
    if 6.701242 <= Q.log_sum_pt < 6.842717:
        z += -8.884302 * Q.log_sum_pt + 59.53586
    if 6.842717 <= Q.log_sum_pt < 6.896095:
        z += -13.31602 * Q.log_sum_pt + 89.86085
    if Q.log_sum_pt >= 6.896095:
        z += -17.16419 * Q.log_sum_pt + 116.3982
    if Q.e2_sq < 0.002074109:
        z += 99.90819 * Q.e2_sq + 0.3549691
    if 0.002074109 <= Q.e2_sq < 0.005284669:
        z += -175.1064 * Q.e2_sq + 0.9253795
    if Q.girth2 < 0.001653836:
        z += -536.4476 * Q.girth2 + 0.8871965
    if Q.girth < 0.007673833:
        z += -135.1914 * Q.girth + 1.037436
    if Q.pt_7 < 37.15625:
        z += 0.06827342 * Q.pt_7 - 2.536784
    if Q.zdr_0 < 0.0211821:
        z += 51.88649 * Q.zdr_0 - 1.099065
    if 430.75 <= Q.sum_pt_top5 < 687.4375:
        z += -0.001507591 * Q.sum_pt_top5 + 0.6493946
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.005227673 * Q.sum_pt_top5 - 3.980678
    if Q.pt_5 < 24.57812:
        z += -0.1034182 * Q.pt_5 + 2.541826
    if Q.width < 0.003562611:
        z += -381.4496 * Q.width + 1.358956
    if Q.girth2_top3 < 0.002151568:
        z += 229.6823 * Q.girth2_top3 - 0.494177
    if Q.z_6 < 0.08051087:
        z += 5.293256 * Q.z_6 - 0.4261646
    if Q.z_7 < 0.04939969 and Q.mass_top5 < 62.55:
        z += 0.4121615 * (0.04939969 - Q.z_7) * (62.55 - Q.mass_top5)
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -45.1109 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.e2_sq < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += 13671.07 * (0.005284669 - Q.e2_sq) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 40.2:
        z += 0.2364885 * (0.07148865 - Q.z_7) * (40.2 - Q.mass_top3)
    if Q.z_7 < 0.07148865 and Q.C3 < 0.03532852:
        z += 319.4759 * (0.07148865 - Q.z_7) * (0.03532852 - Q.C3)
    if Q.girth2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 76389.95 * (0.001653836 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.LHA < 0.2160559 and Q.lam1 < 0.001503553:
        z += -7880.644 * (0.2160559 - Q.LHA) * (0.001503553 - Q.lam1)
    if Q.log_sum_pt > 6.701242 and Q.sj3_dr_max > 0.169029:
        z += 17.47403 * (Q.log_sum_pt - 6.701242) * (Q.sj3_dr_max - 0.169029)
    if Q.log_sum_pt > 6.701242 and Q.dr_2 < 0.01341502:
        z += 666.6938 * (Q.log_sum_pt - 6.701242) * (0.01341502 - Q.dr_2)
    if Q.log_sum_pt > 6.896095 and Q.dr_2 < 0.02270492:
        z += -613.5496 * (Q.log_sum_pt - 6.896095) * (0.02270492 - Q.dr_2)
    if Q.sum_pt_top5 > 430.75 and Q.centroid_offset > 0.009480685:
        z += 0.1214483 * (Q.sum_pt_top5 - 430.75) * (Q.centroid_offset - 0.009480685)
    if Q.z_7 < 0.04939969 and Q.dr_2 < 0.009661512:
        z += -2691.754 * (0.04939969 - Q.z_7) * (0.009661512 - Q.dr_2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi > 0.002834884:
        z += -329.7404 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi - 0.002834884)
    if Q.sum_pt_top5 > 430.75 and Q.pt1_dr01 < 28.39396:
        z += 4.382046e-05 * (Q.sum_pt_top5 - 430.75) * (28.39396 - Q.pt1_dr01)
    if Q.sum_pt_top5 > 430.75 and Q.sj2_dr > 0.1682655:
        z += 0.01170836 * (Q.sum_pt_top5 - 430.75) * (Q.sj2_dr - 0.1682655)
    if Q.pt_7 < 37.15625 and Q.pt_5 > 24.57812:
        z += 0.0006849285 * (37.15625 - Q.pt_7) * (Q.pt_5 - 24.57812)
    if Q.sum_pt_top5 > 430.75 and Q.pt_6 < 46.125:
        z += 0.0001191637 * (Q.sum_pt_top5 - 430.75) * (46.125 - Q.pt_6)
    if Q.z_7 < 0.07148865 and Q.mean_phi2 < 0.002127561:
        z += 3575.919 * (0.07148865 - Q.z_7) * (0.002127561 - Q.mean_phi2)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 27244.18 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.width < 0.003562611 and Q.sj3_dr13 > 0.181053:
        z += -5858.792 * (0.003562611 - Q.width) * (Q.sj3_dr13 - 0.181053)
    if Q.zdr_0 < 0.0211821 and Q.sj3_dr13 > 0.04995258:
        z += 98.84139 * (0.0211821 - Q.zdr_0) * (Q.sj3_dr13 - 0.04995258)
    if Q.centroid_offset < 0.006789738 and Q.pt_5 < 56.4375:
        z += -2.78285 * (0.006789738 - Q.centroid_offset) * (56.4375 - Q.pt_5)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.6740229
    if Q.centroid_offset >= 0.00809236:
        z += 20.23796 * Q.centroid_offset - 0.1637728
    if Q.width < 0.01323868:
        z += 339.5227 * Q.width - 4.494831
    if Q.mass < 21.78408:
        z += -0.07059876 * Q.mass + 1.71829
    if 21.78408 <= Q.mass < 45.595:
        z += 0.004524192 * Q.mass + 0.08180575
    if 45.595 <= Q.mass < 60.63098:
        z += -0.0191598 * Q.mass + 1.161677
    if Q.pt_6 < 19.46875:
        z += -0.2105963 * Q.pt_6 + 4.290765
    if 19.46875 <= Q.pt_6 < 29.90625:
        z += -0.02057114 * Q.pt_6 + 0.5912116
    if 29.90625 <= Q.pt_6 < 39.75:
        z += 0.01127655 * Q.pt_6 - 0.3612334
    if 39.75 <= Q.pt_6 < 41.21875:
        z += -0.05924053 * Q.pt_6 + 2.44182
    if Q.lam2 < 0.001130645:
        z += 320.0647 * Q.lam2 + 0.5802247
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -413.6128 * Q.lam2 + 1.409753
    if 0.02400746 <= Q.sj3_dr_min < 0.1278212:
        z += -6.423041 * Q.sj3_dr_min + 0.1542009
    if Q.sj3_dr_min >= 0.1278212:
        z += -3.218179 * Q.sj3_dr_min - 0.2554483
    if Q.lam1 < 0.00595415:
        z += -599.3804 * Q.lam1 + 5.636324
    if 0.00595415 <= Q.lam1 < 0.00733008:
        z += -707.5398 * Q.lam1 + 6.280321
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -234.0779 * Q.lam1 + 2.809807
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.0177146 * Q.sj3_pair_mass_min + 0.07974627
    if Q.sum_pt < 615.875:
        z += -0.001611411 * Q.sum_pt + 0.992428
    if Q.sum_pt >= 988.4078:
        z += 0.02812481 * Q.sum_pt - 27.79878
    if Q.e2_sq < 0.0030133:
        z += -274.4302 * Q.e2_sq + 0.8269406
    if Q.sj3_dr13 >= 0.181053:
        z += -1.799086 * Q.sj3_dr13 + 0.3257299
    if Q.eccentricity >= 0.927072:
        z += 8.439139 * Q.eccentricity - 7.82369
    if Q.sj3_dr_max < 0.1789613:
        z += -9.514649 * Q.sj3_dr_max + 1.244884
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 3.745656 * Q.sj3_dr_max - 1.128198
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 9.973305 * Q.sj3_dr_max - 2.298675
    if Q.sj3_dr_max >= 0.3012016:
        z += 6.227649 * Q.sj3_dr_max - 1.170478
    if Q.max_dr < 0.1452311:
        z += 18.54918 * Q.max_dr - 2.693918
    if Q.mean_eta >= 0.02644207:
        z += -22.26534 * Q.mean_eta + 0.5887417
    if Q.girth2 < 0.008678045:
        z += 664.3256 * Q.girth2 - 5.765047
    if Q.C2_b2 < 0.02415398:
        z += -31.48935 * Q.C2_b2 + 0.7605932
    if Q.sj2_dr < 0.1872617:
        z += -2.13514 * Q.sj2_dr + 0.3998299
    if Q.sj3_dr23 >= 0.2207152:
        z += -2.64246 * Q.sj3_dr23 + 0.5832312
    if Q.log_sum_pt < 6.267538:
        z += -2.010613 * Q.log_sum_pt + 12.60159
    if Q.z_6 < 0.02160287:
        z += 130.6614 * Q.z_6 - 2.822663
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.01737766 * Q.sum_pt_top5 + 14.59645
    if Q.tau2 >= 0.008780509:
        z += -4.877803 * Q.tau2 + 0.0428296
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.2762678 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += 23.39149 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 2445.663 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -740.6754 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -2.581294 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.width < 0.01323868 and Q.mean_eta < -0.02665591:
        z += 8199.833 * (0.01323868 - Q.width) * (-0.02665591 - Q.mean_eta)
    if Q.width < 0.01323868 and Q.mean_phi < -0.02594505:
        z += 8150.935 * (0.01323868 - Q.width) * (-0.02594505 - Q.mean_phi)
    if Q.width < 0.01323868 and Q.mean_phi > 0.02612796:
        z += 7825.562 * (0.01323868 - Q.width) * (Q.mean_phi - 0.02612796)
    if Q.pt_6 < 29.90625 and Q.n_pt_above_10 < 8.0:
        z += 0.01839228 * (29.90625 - Q.pt_6) * (8.0 - Q.n_pt_above_10)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.489381 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.001624003 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.sj3_dr_max < 0.1789613 and Q.tau21_b2 < 0.1230713:
        z += 56.44306 * (0.1789613 - Q.sj3_dr_max) * (0.1230713 - Q.tau21_b2)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 44.59723 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += 0.09138891 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 9503.015 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.009220341 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sj2_dr < 0.1872617 and Q.eccentricity > 0.927072:
        z += 59.83218 * (0.1872617 - Q.sj2_dr) * (Q.eccentricity - 0.927072)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 < 0.002776626:
        z += -2.697941 * (Q.sum_pt - 988.4078) * (0.002776626 - Q.mean_phi2)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 > 0.008921136:
        z += -3.133432 * (Q.sum_pt - 988.4078) * (Q.mean_phi2 - 0.008921136)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 < 0.002127561:
        z += -2.079254 * (Q.sum_pt - 988.4078) * (0.002127561 - Q.mean_phi2)
    if Q.tau1 < 0.1136369 and Q.mean_phi2 < 0.01426135:
        z += 1631.926 * (0.1136369 - Q.tau1) * (0.01426135 - Q.mean_phi2)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0753896:
        z += 64.38758 * (6.638339 - Q.log_sum_pt) * (0.0753896 - Q.z_7)
    if Q.sum_pt > 988.4078 and Q.n_pt_above_50 > 6.0:
        z += -0.002052971 * (Q.sum_pt - 988.4078) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt < 615.875 and Q.e4 < 3.0374e-08:
        z += -59030.13 * (615.875 - Q.sum_pt) * (3.0374e-08 - Q.e4)
    if Q.centroid_offset > 0.01837778 and Q.sj3_dr12 > 0.1692253:
        z += -93.86285 * (Q.centroid_offset - 0.01837778) * (Q.sj3_dr12 - 0.1692253)
    return max(0.0, z)


def neuron_7(Q):
    z = 10.75864
    if Q.girth2_top2 < 8.10414e-05:
        z += 6656.919 * Q.girth2_top2 - 0.1330625
    if 8.10414e-05 <= Q.girth2_top2 < 0.001056655:
        z += -416.5825 * Q.girth2_top2 + 0.440184
    if Q.girth2 < 0.0009641429:
        z += 1452.284 * Q.girth2 - 2.405585
    if 0.0009641429 <= Q.girth2 < 0.001653836:
        z += 386.9109 * Q.girth2 - 1.378413
    if 0.001653836 <= Q.girth2 < 0.003562611:
        z += 93.88608 * Q.girth2 - 0.8937981
    if 0.003562611 <= Q.girth2 < 0.004372139:
        z += -293.0249 * Q.girth2 + 0.4846152
    if 0.004372139 <= Q.girth2 < 0.007520088:
        z += -832.6106 * Q.girth2 + 2.843759
    if 0.007520088 <= Q.girth2 < 0.01323868:
        z += -2488.617 * Q.girth2 + 15.29707
    if Q.girth2 >= 0.01323868:
        z += -2590.422 * Q.girth2 + 16.64484
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -50.43529 * Q.mass_over_sum_pt + 0.5598237
    if 0.07269073 <= Q.mass_over_sum_pt < 0.07992374:
        z += 103.6702 * Q.mass_over_sum_pt - 10.64222
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 221.4926 * Q.mass_over_sum_pt - 20.05902
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 477.4881 * Q.mass_over_sum_pt - 41.75506
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 458.5396 * Q.mass_over_sum_pt - 40.04185
    if Q.tau1 < 0.05356915:
        z += -39.93806 * Q.tau1 + 2.139448
    if Q.girth < 0.04081947:
        z += 101.1787 * Q.girth - 8.022326
    if 0.04081947 <= Q.girth < 0.08723651:
        z += 83.85421 * Q.girth - 7.315149
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.0324626 * Q.mass - 1.176101
    if Q.mass >= 76.6557:
        z += -0.1109069 * Q.mass + 9.813986
    if Q.centroid_offset < 0.02076709:
        z += 33.91804 * Q.centroid_offset - 0.704379
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += 23.13807 * Q.centroid_offset - 0.7212314
    if Q.centroid_offset >= 0.03776099:
        z += -111.5987 * Q.centroid_offset + 4.366563
    if Q.width < 0.005590289:
        z += 909.4562 * Q.width - 6.074686
    if 0.005590289 <= Q.width < 0.006679471:
        z += -87.31738 * Q.width - 0.5024346
    if 0.006679471 <= Q.width < 0.008678045:
        z += -996.7736 * Q.width + 5.572252
    if Q.width >= 0.008678045:
        z += -1360.416 * Q.width + 8.727956
    if Q.z_7 >= 0.03243272:
        z += 10.37731 * Q.z_7 - 0.3365644
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += 36.12814 * Q.e2 - 1.28475
    if Q.e2 >= 0.05028464:
        z += 209.126 * Q.e2 - 9.983888
    if Q.sj2_dr < 0.1294903:
        z += -5.658527 * Q.sj2_dr + 0.3438903
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -8.058678 * Q.sj2_dr + 0.6546865
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 22.35722 * Q.sj2_dr - 4.186651
    if Q.e2_sq < 0.001101266:
        z += -240.9794 * Q.e2_sq + 0.2653825
    if Q.LHA < 0.3033137:
        z += -17.58148 * Q.LHA + 5.332705
    if Q.pt_7 < 29.04219:
        z += 0.03463694 * Q.pt_7 - 1.005933
    if 0.2037854 <= Q.sd_rg < 0.2787955:
        z += -11.2367 * Q.sd_rg + 2.289875
    if Q.sd_rg >= 0.2787955:
        z += 25.59534 * Q.sd_rg - 7.978734
    if Q.lam1 < 0.008375572:
        z += 160.1135 * Q.lam1 - 1.341042
    if Q.mass_top5 >= 53.60766:
        z += 0.03485658 * Q.mass_top5 - 1.86858
    if Q.D2_b2 < 0.05744392:
        z += 6.522459 * Q.D2_b2 - 0.3746756
    if 0.04889979 <= Q.sj3_dr_max < 0.1426152:
        z += -8.644014 * Q.sj3_dr_max + 0.4226905
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += 11.39454 * Q.sj3_dr_max - 2.435112
    if 0.213399 <= Q.sj3_dr_max < 0.2623172:
        z += 7.902933 * Q.sj3_dr_max - 1.690006
    if Q.sj3_dr_max >= 0.2623172:
        z += -8.688309 * Q.sj3_dr_max + 2.662163
    if Q.lam2 < 0.0003061234:
        z += 1752.572 * Q.lam2 - 0.5365032
    if Q.D2 < 2.357246:
        z += -0.1748898 * Q.D2 + 0.4122583
    if Q.max_dr < 0.1117619:
        z += 7.582569 * Q.max_dr - 0.5138164
    if 0.1117619 <= Q.max_dr < 0.177305:
        z += -5.090169 * Q.max_dr + 0.9025123
    if Q.tau21_b2 < 0.04019753:
        z += 24.91 * Q.tau21_b2 - 1.001321
    if Q.planar_flow < 0.1950135 and Q.width > 0.007520088:
        z += -1810.622 * (0.1950135 - Q.planar_flow) * (Q.width - 0.007520088)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1168587 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -13.33402 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.tau1 < 0.05356915 and Q.n_dr_0p05_0p1 > 5.0:
        z += -20.90028 * (0.05356915 - Q.tau1) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.110071 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.girth < 0.08723651 and Q.mean_phi < 0.004406178:
        z += 220.6782 * (0.08723651 - Q.girth) * (0.004406178 - Q.mean_phi)
    if Q.girth2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 1720.51 * (Q.girth2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.girth2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += -1544.945 * (Q.girth2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.centroid_offset > 0.03776099 and Q.pt_4 > 81.375:
        z += -24.33568 * (Q.centroid_offset - 0.03776099) * (Q.pt_4 - 81.375)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -16561.36 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.girth < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 3649.65 * (0.08723651 - Q.girth) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 1517.196 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    if Q.girth2_top2 < 0.001056655 and Q.tau21_b2 < 0.02656143:
        z += -55172.35 * (0.001056655 - Q.girth2_top2) * (0.02656143 - Q.tau21_b2)
    if Q.pt_7 < 29.04219 and Q.tau21 < 0.2838437:
        z += -0.1900599 * (29.04219 - Q.pt_7) * (0.2838437 - Q.tau21)
    if Q.lam2 < 0.0003061234 and Q.tau21_b2 < 0.04019753:
        z += 105573.0 * (0.0003061234 - Q.lam2) * (0.04019753 - Q.tau21_b2)
    if Q.girth2 > 0.01323868 and Q.pt_6 < 38.25:
        z += 53.1095 * (Q.girth2 - 0.01323868) * (38.25 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.01109984 and Q.pt_6 < 35.28125:
        z += -0.7219878 * (Q.mass_over_sum_pt - 0.01109984) * (35.28125 - Q.pt_6)
    if Q.width > 0.008678045 and Q.pt_6 < 38.25:
        z += 20.52653 * (Q.width - 0.008678045) * (38.25 - Q.pt_6)
    if Q.mass > 76.6557 and Q.zdr_6 > 0.006366792:
        z += -37.78693 * (Q.mass - 76.6557) * (Q.zdr_6 - 0.006366792)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.24033
    if Q.girth2 < 0.005019719:
        z += -240.3292 * Q.girth2 + 1.206385
    if Q.LHA < 0.1967397:
        z += 9.118854 * Q.LHA - 1.794041
    if Q.log_sum_pt >= 6.701242:
        z += -8.068594 * Q.log_sum_pt + 54.0696
    if Q.mass < 21.78408:
        z += 0.002495252 * Q.mass - 0.03266746
    if 21.78408 <= Q.mass < 29.6447:
        z += -0.002759236 * Q.mass + 0.08179671
    if Q.girth < 0.06108601:
        z += 70.68946 * Q.girth - 4.318137
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.00311795 * Q.sum_pt_top5 - 2.143396
    if Q.lam2 < 0.0001330621:
        z += 3260.86 * Q.lam2 - 0.4338967
    if Q.width < 0.002635418:
        z += -1128.048 * Q.width + 3.831354
    if 0.002635418 <= Q.width < 0.003562611:
        z += -925.8879 * Q.width + 3.298578
    if Q.z_6 < 0.03448406:
        z += 24.71018 * Q.z_6 - 0.8521071
    if Q.mass_over_sum_pt < 0.03319429:
        z += 34.36206 * Q.mass_over_sum_pt - 1.140624
    if Q.sj3_dr_max < 0.1070199:
        z += 9.644069 * Q.sj3_dr_max - 1.612966
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 13.82907 * Q.sj3_dr_max - 2.060844
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += 1.581961 * Q.sj3_dr_max - 0.3142205
    if Q.e2 < 0.01289969:
        z += -74.32626 * Q.e2 + 0.9587858
    if Q.girth2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -20.39526 * (0.006679471 - Q.girth2) * (4.721224 - Q.D2_b2)
    if Q.girth2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 13117.44 * (0.006679471 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.girth2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -7.727257 * (0.005019719 - Q.girth2) * (43.5 - Q.pt_7)
    if Q.girth2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 955.78 * (0.006679471 - Q.girth2) * (Q.phi_0 - -0.04013062)
    if Q.girth2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 4544.174 * (0.005019719 - Q.girth2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -155.6851 * (0.06108601 - Q.girth) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.06778087 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.girth < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 101558.4 * (0.06108601 - Q.girth) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 1023.481 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.log_sum_pt > 6.701242 and Q.e3 < 2.955458e-05:
        z += 69603.57 * (Q.log_sum_pt - 6.701242) * (2.955458e-05 - Q.e3)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -467.0551 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 14425.94 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.mass < 49.6681 and Q.lam2 < 0.0001330621:
        z += 55.45723 * (49.6681 - Q.mass) * (0.0001330621 - Q.lam2)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += 0.00106201 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.06511338 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.001003868 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.girth2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -26133.39 * (0.005019719 - Q.girth2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.09094755 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.girth < 0.06108601 and Q.width > 0.0003193707:
        z += 8807.998 * (0.06108601 - Q.girth) * (Q.width - 0.0003193707)
    if Q.log_sum_pt > 6.701242 and Q.width < 0.008678045:
        z += -96.75926 * (Q.log_sum_pt - 6.701242) * (0.008678045 - Q.width)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2 < 0.006679471:
        z += 3101.376 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.girth2)
    if Q.tau1 < 0.05356915 and Q.width < 0.003562611:
        z += 21372.35 * (0.05356915 - Q.tau1) * (0.003562611 - Q.width)
    if Q.log_sum_pt > 6.701242 and Q.centroid_offset < 0.02355416:
        z += -110.6554 * (Q.log_sum_pt - 6.701242) * (0.02355416 - Q.centroid_offset)
    if Q.log_sum_pt > 6.701242 and Q.girth2 < 0.0005611231:
        z += 3742.513 * (Q.log_sum_pt - 6.701242) * (0.0005611231 - Q.girth2)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2_top5 < 0.005691733:
        z += -1203.731 * (0.1986272 - Q.sj3_dr_max) * (0.005691733 - Q.girth2_top5)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0001947983:
        z += 13679.33 * (0.1986272 - Q.sj3_dr_max) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.centroid_offset < 0.02685622:
        z += -3.562815 * (29.6447 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.03319429 and Q.centroid_offset < 0.02685622:
        z += 4089.403 * (0.03319429 - Q.mass_over_sum_pt) * (0.02685622 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1986272 and Q.width < 0.003562611:
        z += -2316.647 * (0.1986272 - Q.sj3_dr_max) * (0.003562611 - Q.width)
    if Q.girth < 0.06108601 and Q.girth2 < 0.003562611:
        z += -30552.9 * (0.06108601 - Q.girth) * (0.003562611 - Q.girth2)
    if Q.sj3_dr_max < 0.1426152 and Q.centroid_offset > 0.01627885:
        z += -1108.731 * (0.1426152 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.e2 < 0.01289969 and Q.psi_0p2 > 0.79448:
        z += -598.2097 * (0.01289969 - Q.e2) * (Q.psi_0p2 - 0.79448)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.325511
    if Q.girth < 0.05464922:
        z += 47.49498 * Q.girth - 2.595564
    if Q.girth >= 0.0717028:
        z += -14.51188 * Q.girth + 1.040543
    if Q.tau1 < 0.04369778:
        z += -57.71657 * Q.tau1 + 2.522086
    if Q.mass < 29.6447:
        z += 0.1167634 * Q.mass - 4.719787
    if 29.6447 <= Q.mass < 45.595:
        z += 0.0531235 * Q.mass - 2.833202
    if 45.595 <= Q.mass < 53.33237:
        z += 0.07422796 * Q.mass - 3.79546
    if Q.mass >= 53.33237:
        z += 0.02110447 * Q.mass - 0.9622582
    if Q.e3 < 2.371297e-05:
        z += 45974.77 * Q.e3 - 1.090199
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -3590.64 * Q.e3 + 0.2925562
    if Q.e3 >= 0.0001869378:
        z += 287.5518 * Q.e3 - 0.4324246
    if Q.girth2 < 0.003562611:
        z += -3561.373 * Q.girth2 + 16.63189
    if 0.003562611 <= Q.girth2 < 0.005019719:
        z += -1734.3 * Q.girth2 + 10.12274
    if 0.005019719 <= Q.girth2 < 0.00609665:
        z += -1315.814 * Q.girth2 + 8.022058
    if Q.girth2 >= 0.01882765:
        z += 67.28339 * Q.girth2 - 1.266788
    if Q.sj3_dr_max < 0.1070199:
        z += 14.30408 * Q.sj3_dr_max - 1.00559
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 23.94054 * Q.sj3_dr_max - 2.036883
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -19.45927 * Q.sj3_dr_max + 4.152589
    if Q.lam2 < 0.0003061234:
        z += -1959.966 * Q.lam2 + 0.5999914
    if Q.lam2 >= 0.001130645:
        z += 260.1872 * Q.lam2 - 0.2941793
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.7258258 * Q.n_dr_0p2_0p4 - 0.7258258
    if Q.lam1 < 0.003377388:
        z += 556.5455 * Q.lam1 - 3.313755
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 337.5063 * Q.lam1 - 2.573975
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 777.2021 * Q.lam1 - 4.963001
    if Q.lam1 >= 0.00595415:
        z += 220.6566 * Q.lam1 - 1.649245
    if Q.centroid_offset < 0.002316125:
        z += 148.6223 * Q.centroid_offset + 1.9134
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -140.5601 * Q.centroid_offset + 2.583183
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -36.9332 * Q.e2 + 1.313378
    if Q.e2 >= 0.04447357:
        z += 5.919403 * Q.e2 - 0.5924299
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.0163561 * Q.sj3_pair_mass_max - 0.3963105
    if Q.width < 0.0001721983:
        z += -6662.121 * Q.width + 1.147206
    if Q.C3 < 0.0284695:
        z += 16.69652 * Q.C3 - 0.4753414
    if Q.zdr_0 < 0.006292091:
        z += 56.0873 * Q.zdr_0 - 0.3529064
    if Q.mass_over_sum_pt < 0.07269073:
        z += 95.42335 * Q.mass_over_sum_pt - 6.936393
    if Q.sum_pt < 988.4078:
        z += -0.006008313 * Q.sum_pt + 5.938664
    z += -3.893589 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 6.316431 * Q.log_sum_pt - 43.55871
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.005151343 * Q.sum_pt_top5 + 4.326895
    if Q.n_for_90pct < 7.0:
        z += 0.141621 * Q.n_for_90pct - 0.9913473
    if Q.max_dr < 0.1117619:
        z += -11.7814 * Q.max_dr + 1.316711
    if Q.M3 < 0.0782171:
        z += -8.206219 * Q.M3 + 0.6418666
    if Q.absphi_1 < 0.02227783:
        z += 12.50772 * Q.absphi_1 - 0.2786449
    if Q.girth2_top5 >= 0.01148293:
        z += -33.61931 * Q.girth2_top5 + 0.3860481
    if Q.N2 >= 0.2233283:
        z += 2.221723 * Q.N2 - 0.4961735
    if Q.z_5 < 0.06503035:
        z += -9.409015 * Q.z_5 + 0.6118715
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -410.6743 * Q.mass_over_sum_pt_sq + 2.949806
    if Q.zdr_2 < 0.002170915:
        z += -177.3048 * Q.zdr_2 + 0.3849136
    if Q.D2 >= 2.055451:
        z += -0.2652617 * Q.D2 + 0.5452325
    if Q.sd_mass >= 45.595:
        z += -0.01279099 * Q.sd_mass + 0.5832053
    if Q.e2_sq < 0.0030133:
        z += 693.1064 * Q.e2_sq - 2.088538
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 5.472482 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.06132759 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.girth2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 753.8943 * (0.00609665 - Q.girth2) * (0.3220738 - Q.planar_flow)
    if Q.girth2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -7944.411 * (0.00609665 - Q.girth2) * (0.002834884 - Q.mean_phi)
    if Q.mass < 53.33237 and Q.D2 > 2.843757:
        z += 0.003923639 * (53.33237 - Q.mass) * (Q.D2 - 2.843757)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -436.9911 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -365.3494 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.e3 < 2.371297e-05 and Q.eccentricity > 0.8319502:
        z += 92735.88 * (2.371297e-05 - Q.e3) * (Q.eccentricity - 0.8319502)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += -106.0078 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -15.74848 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -24.09764 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.3527407 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -1.690969 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 922.5851 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -2.44836 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.girth < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += 2270.062 * (0.05464922 - Q.girth) * (0.02656143 - Q.tau21_b2)
    if Q.tau1 < 0.04369778 and Q.tau21_b2 < 0.02656143:
        z += -2833.666 * (0.04369778 - Q.tau1) * (0.02656143 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.07269073 and Q.phi_0 < -0.004917145:
        z += 379.2628 * (0.07269073 - Q.mass_over_sum_pt) * (-0.004917145 - Q.phi_0)
    if Q.e3 > 0.0001869378 and Q.pt_6 < 33.6875:
        z += -182.6107 * (Q.e3 - 0.0001869378) * (33.6875 - Q.pt_6)
    if Q.girth < 0.05464922 and Q.z_6 > 0.03448406:
        z += -670.0645 * (0.05464922 - Q.girth) * (Q.z_6 - 0.03448406)
    if Q.centroid_offset < 0.01837778 and Q.pair_mass_0_2 > 17.9037:
        z += -0.9955814 * (0.01837778 - Q.centroid_offset) * (Q.pair_mass_0_2 - 17.9037)
    if Q.sum_pt < 988.4078 and Q.dr_2 < 0.01778111:
        z += -0.1610592 * (988.4078 - Q.sum_pt) * (0.01778111 - Q.dr_2)
    if Q.log_sum_pt > 6.896095 and Q.zdr_5 < 0.001255404:
        z += 3846.948 * (Q.log_sum_pt - 6.896095) * (0.001255404 - Q.zdr_5)
    if Q.sd_mass > 45.595 and Q.n_dr_0p05_0p1 > 7.0:
        z += -0.03502538 * (Q.sd_mass - 45.595) * (Q.n_dr_0p05_0p1 - 7.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.401504
    z += -6.557615 * Q.e2
    if 11.051 <= Q.sj3_pair_mass_min < 15.95929:
        z += 0.03991168 * Q.sj3_pair_mass_min - 0.4410641
    if Q.sj3_pair_mass_min >= 15.95929:
        z += 0.1143458 * Q.sj3_pair_mass_min - 1.62898
    if Q.lam1 < 0.001503553:
        z += 1081.055 * Q.lam1 - 3.848349
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 586.6918 * Q.lam1 - 3.105048
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 367.4102 * Q.lam1 - 2.187615
    if Q.lam1 >= 0.00733008:
        z += -241.921 * Q.lam1 + 1.7733
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 965.8324 * Q.lam2 - 0.2956639
    if Q.lam2 >= 0.003408389:
        z += 479.8601 * Q.lam2 + 1.360719
    if Q.sj3_dr_min >= 0.1278212:
        z += 12.54663 * Q.sj3_dr_min - 1.603724
    if Q.e3 < 3.892127e-05:
        z += -3575.376 * Q.e3 + 0.2913125
    if 3.892127e-05 <= Q.e3 < 8.147744e-05:
        z += -5411.911 * Q.e3 + 0.3627928
    if Q.e3 >= 8.147744e-05:
        z += -1836.535 * Q.e3 + 0.07148029
    if 22.18342 <= Q.mass_top5 < 45.32077:
        z += 0.005866229 * Q.mass_top5 - 0.130133
    if Q.mass_top5 >= 45.32077:
        z += -0.01991546 * Q.mass_top5 + 1.038313
    if Q.M2 < 0.02563286:
        z += 29.22986 * Q.M2 - 0.7492451
    if Q.sum_pt >= 988.4078:
        z += 0.003403856 * Q.sum_pt - 3.364398
    if Q.z_7 < 0.06473447:
        z += 16.78576 * Q.z_7 - 1.086617
    if Q.LHA >= 0.3033137:
        z += -19.64895 * Q.LHA + 5.959796
    if Q.tau1 >= 0.05356915:
        z += 7.619995 * Q.tau1 - 0.4081967
    if Q.girth2 < 0.002635418:
        z += 423.4749 * Q.girth2 - 1.116033
    if 0.007520088 <= Q.girth2 < 0.02530566:
        z += 500.988 * Q.girth2 - 3.767474
    if Q.girth2 >= 0.02530566:
        z += 424.1182 * Q.girth2 - 1.822234
    if Q.mass < 76.6557:
        z += -0.01619824 * Q.mass + 1.241688
    if Q.girth2_top3 < 0.002151568:
        z += -274.6198 * Q.girth2_top3 + 0.590863
    if Q.sj2_dr >= 0.3003793:
        z += 5.965174 * Q.sj2_dr - 1.791815
    if Q.sj3_dr_max >= 0.1986272:
        z += -5.010618 * Q.sj3_dr_max + 0.9952451
    if Q.C2_b2 >= 0.009032972:
        z += 36.71022 * Q.C2_b2 - 0.3316024
    if Q.n_pt_above_50 >= 6.0:
        z += -0.1298601 * Q.n_pt_above_50 + 0.7791607
    if Q.zdr_0 < 0.004918231:
        z += 77.04594 * Q.zdr_0 - 0.3789297
    if Q.sj3_pair_mass_min > 11.051 and Q.n_dr_0p2_0p4 > 0.0:
        z += 0.008108574 * (Q.sj3_pair_mass_min - 11.051) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.05551922 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.1639731 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -454.6254 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -17.26421 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 120.8444 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.lam1 > 0.00733008 and Q.sj3_mass3 < 0.2723288:
        z += -100.8982 * (Q.lam1 - 0.00733008) * (0.2723288 - Q.sj3_mass3)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += -51776.76 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -307.3182 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.sj3_pair_mass_min > 11.051 and Q.sj3_pairmin_over_m > 0.28737:
        z += -0.2519657 * (Q.sj3_pair_mass_min - 11.051) * (Q.sj3_pairmin_over_m - 0.28737)
    if Q.sj3_dr_min > 0.1278212 and Q.sj3_mass2 < 0.5973755:
        z += -12.26706 * (Q.sj3_dr_min - 0.1278212) * (0.5973755 - Q.sj3_mass2)
    if Q.lam2 > 0.0003061234 and Q.pt_6 < 31.90625:
        z += -31.18634 * (Q.lam2 - 0.0003061234) * (31.90625 - Q.pt_6)
    if Q.lam1 < 0.00595415 and Q.n_pt_above_50 > 6.0:
        z += 49.18577 * (0.00595415 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.tau1 > 0.05356915 and Q.D2 < 1.002471:
        z += 9.021277 * (Q.tau1 - 0.05356915) * (1.002471 - Q.D2)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.233243
    if Q.planar_flow < 0.2534037:
        z += -5.474663 * Q.planar_flow + 1.3873
    if 0.1778793 <= Q.sj2_dr < 0.2687922:
        z += -7.147271 * Q.sj2_dr + 1.271352
    if Q.sj2_dr >= 0.2687922:
        z += -1.919298 * Q.sj2_dr - 0.1338866
    if Q.mass < 15.45403:
        z += -0.01567609 * Q.mass - 1.063215
    if 15.45403 <= Q.mass < 49.6681:
        z += 0.01516368 * Q.mass - 1.539814
    if 49.6681 <= Q.mass < 69.61135:
        z += 0.03944506 * Q.mass - 2.745824
    if Q.girth2 < 0.004372139:
        z += -18.78174 * Q.girth2 + 4.915841
    if 0.004372139 <= Q.girth2 < 0.01323868:
        z += -545.1649 * Q.girth2 + 7.217261
    if Q.tau1 < 0.09538712:
        z += 16.67316 * Q.tau1 - 1.590404
    if Q.centroid_offset < 0.01437952:
        z += -81.71014 * Q.centroid_offset + 2.74277
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -67.05383 * Q.centroid_offset + 2.532019
    if Q.centroid_offset >= 0.04990367:
        z += -194.9641 * Q.centroid_offset + 9.729421
    if Q.sj3_dr_max < 0.1426152:
        z += -1.413503 * Q.sj3_dr_max - 0.01109746
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 25.81953 * Q.sj3_dr_max - 3.894942
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -5.030746 * Q.sj3_dr_max + 1.319651
    if Q.width < 0.006679471:
        z += -1330.118 * Q.width + 11.31813
    if 0.006679471 <= Q.width < 0.008678045:
        z += -1217.689 * Q.width + 10.56716
    if Q.lam1 < 0.00733008:
        z += 245.2904 * Q.lam1 - 2.012759
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 205.416 * Q.lam1 - 1.720477
    if Q.e2_sq < 0.008168571:
        z += 3635.092 * Q.e2_sq - 29.6935
    if Q.girth < 0.02689598:
        z += 113.4076 * Q.girth - 5.447244
    if 0.02689598 <= Q.girth < 0.0717028:
        z += 53.49712 * Q.girth - 3.835893
    if Q.z_7 >= 0.01685855:
        z += 37.85634 * Q.z_7 - 0.6382029
    if Q.max_pair_mass >= 33.3761:
        z += 0.0244641 * Q.max_pair_mass - 0.8165163
    if Q.pt_7 >= 29.04219:
        z += -0.03251141 * Q.pt_7 + 0.9442024
    if Q.eccentricity >= 0.9884745:
        z += 54.10109 * Q.eccentricity - 53.47755
    if Q.C2 < 0.03578649:
        z += 32.98427 * Q.C2 - 1.180391
    if Q.max_dr < 0.1027585:
        z += -1.477367 * Q.max_dr + 0.6271913
    if 0.1027585 <= Q.max_dr < 0.1452311:
        z += -11.19261 * Q.max_dr + 1.625516
    if Q.e3 < 1.050302e-05:
        z += -81246.64 * Q.e3 + 0.8533352
    if Q.sum_pt_top5 < 506.875:
        z += 0.004997317 * Q.sum_pt_top5 - 2.533015
    if Q.mass_over_sum_pt < 0.07637363:
        z += 23.51832 * Q.mass_over_sum_pt - 1.79618
    if Q.e2 < 0.01655442:
        z += -32.61728 * Q.e2 + 0.5399602
    if Q.pt_6 < 24.42188:
        z += 0.06512712 * Q.pt_6 - 1.590526
    if Q.zdr_0 < 0.003676313:
        z += -102.1946 * Q.zdr_0 + 0.3756995
    if Q.lam2 < 0.001130645:
        z += -851.3832 * Q.lam2 + 0.962612
    if Q.mass_over_sum_pt_sq < 0.00817466:
        z += -2820.483 * Q.mass_over_sum_pt_sq + 23.05649
    if Q.n_dr_0p1_0p2 >= 3.0:
        z += -0.1036065 * Q.n_dr_0p1_0p2 + 0.3108196
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.005771817 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.tau1 < 0.09538712 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += 9.748886 * (0.09538712 - Q.tau1) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -510129.3 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -6523.831 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.mass < 49.6681 and Q.D2 < 1.332146:
        z += -0.02322234 * (49.6681 - Q.mass) * (1.332146 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.pt_6 > 19.46875:
        z += -0.4893341 * (0.01323868 - Q.girth2) * (Q.pt_6 - 19.46875)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.1559749 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += 82.13165 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.D2 < 0.415246:
        z += 183.5 * (0.01323868 - Q.girth2) * (0.415246 - Q.D2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -298.2953 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -673.1672 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 153.6258 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.pt_6 < 41.21875 and Q.D2_b2 < 4.721224:
        z += -0.006081003 * (41.21875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += -587.9672 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    if Q.eccentricity > 0.9884745 and Q.sj2_mass2 < 2.771069:
        z += -16.92425 * (Q.eccentricity - 0.9884745) * (2.771069 - Q.sj2_mass2)
    if Q.centroid_offset < 0.01437952 and Q.zdr_7 < 0.00279494:
        z += -11356.0 * (0.01437952 - Q.centroid_offset) * (0.00279494 - Q.zdr_7)
    if Q.mass < 15.45403 and Q.zdr_7 < 0.003780225:
        z += 16.78741 * (15.45403 - Q.mass) * (0.003780225 - Q.zdr_7)
    if Q.centroid_offset < 0.01437952 and Q.sj3_pair_mass_min < 15.95929:
        z += -8.462698 * (0.01437952 - Q.centroid_offset) * (15.95929 - Q.sj3_pair_mass_min)
    if Q.girth < 0.02689598 and Q.sj3_pair_mass_min > 1.590484:
        z += -7.709562 * (0.02689598 - Q.girth) * (Q.sj3_pair_mass_min - 1.590484)
    if Q.planar_flow < 0.2534037 and Q.sj3_pair_mass_min > 1.282345:
        z += -0.2287435 * (0.2534037 - Q.planar_flow) * (Q.sj3_pair_mass_min - 1.282345)
    if Q.centroid_offset > 0.04990367 and Q.zdr_6 > 0.007390416:
        z += -91085.32 * (Q.centroid_offset - 0.04990367) * (Q.zdr_6 - 0.007390416)
    if Q.sum_pt_top5 < 506.875 and Q.z_dr_0p1_0p2 > 0.04510668:
        z += 0.003690014 * (506.875 - Q.sum_pt_top5) * (Q.z_dr_0p1_0p2 - 0.04510668)
    if Q.centroid_offset > 0.04990367 and Q.tau3 > 0.001996306:
        z += 1211.603 * (Q.centroid_offset - 0.04990367) * (Q.tau3 - 0.001996306)
    if Q.girth2 < 0.004372139 and Q.sj3_dr13 > 0.1659434:
        z += 4799.298 * (0.004372139 - Q.girth2) * (Q.sj3_dr13 - 0.1659434)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.389304
    if Q.girth2 >= 0.01882765:
        z += 242.3774 * Q.girth2 - 4.563397
    if 69.61135 <= Q.mass < 91.19:
        z += -0.01508632 * Q.mass + 1.050179
    if Q.mass >= 91.19:
        z += 0.06963491 * Q.mass - 6.67555
    if Q.zdr_0 >= 0.03981924:
        z += -28.89404 * Q.zdr_0 + 1.150539
    if Q.girth2_top2 >= 0.01403324:
        z += -29.86329 * Q.girth2_top2 + 0.4190788
    if Q.e2 >= 0.06344108:
        z += -39.30024 * Q.e2 + 2.493249
    if Q.centroid_offset >= 0.04990367:
        z += 23.34937 * Q.centroid_offset - 1.165219
    if Q.girth2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += 8421.641 * (Q.girth2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.girth2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -3.926117 * (Q.girth2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.mass > 69.61135 and Q.centroid_offset > 0.009480685:
        z += 0.7499266 * (Q.mass - 69.61135) * (Q.centroid_offset - 0.009480685)
    if Q.mass > 69.61135 and Q.log_sum_pt < 6.423044:
        z += 0.2333504 * (Q.mass - 69.61135) * (6.423044 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.4729582
    if Q.girth < 0.1484084:
        z += -74.45644 * Q.girth + 11.04996
    if Q.lam1 < 0.006506576:
        z += -505.9917 * Q.lam1 + 4.665576
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -138.3378 * Q.lam1 + 2.273409
    if 658.125 <= Q.sum_pt_top5 < 791.125:
        z += -0.002185457 * Q.sum_pt_top5 + 1.438304
    if Q.sum_pt_top5 >= 791.125:
        z += 0.001150736 * Q.sum_pt_top5 - 1.201041
    if Q.e3 < 5.334511e-05:
        z += -21192.55 * Q.e3 + 1.130519
    if Q.pt_6 < 31.90625:
        z += 0.0794929 * Q.pt_6 - 2.53632
    if Q.e2 < 0.08000524:
        z += 57.36297 * Q.e2 - 4.589338
    if Q.mass >= 49.6681:
        z += -0.02334113 * Q.mass + 1.15931
    if Q.z_5 < 0.02818362:
        z += 93.21896 * Q.z_5 - 2.627248
    if Q.z_7 < 0.02807091:
        z += 106.132 * Q.z_7 - 2.979222
    if Q.sj3_dr23 >= 0.1974628:
        z += -2.325249 * Q.sj3_dr23 + 0.4591502
    if Q.log_sum_pt >= 6.502799:
        z += 1.775087 * Q.log_sum_pt - 11.54303
    if Q.pt_7 >= 48.71875:
        z += -0.03391402 * Q.pt_7 + 1.652249
    if Q.width < 0.007520088:
        z += 444.4774 * Q.width - 3.113555
    if 0.007520088 <= Q.width < 0.01323868:
        z += -40.03681 * Q.width + 0.5300343
    if Q.lam2 < 0.0001947983:
        z += 1173.087 * Q.lam2 - 0.2285154
    if Q.z_6 < 0.02160287:
        z += 144.0481 * Q.z_6 - 3.111853
    if Q.girth < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -64.33632 * (0.1484084 - Q.girth) * (6.804164 - Q.log_sum_pt)
    if Q.girth < 0.1484084 and Q.pt_7 < 38.53125:
        z += -1.013244 * (0.1484084 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -913243.0 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0006005809 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -0.7369886 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.girth < 0.1484084 and Q.z_7 > 0.06164517:
        z += 343.3239 * (0.1484084 - Q.girth) * (Q.z_7 - 0.06164517)
    if Q.girth < 0.1484084 and Q.lam2 < 0.000537286:
        z += -7753.954 * (0.1484084 - Q.girth) * (0.000537286 - Q.lam2)
    if Q.girth < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -1.135401 * (0.1484084 - Q.girth) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_pt > 988.4078 and Q.M3 < 0.08151794:
        z += -0.1017935 * (Q.sum_pt - 988.4078) * (0.08151794 - Q.M3)
    if Q.girth < 0.1484084 and Q.M3 < 0.07474969:
        z += 89.2293 * (0.1484084 - Q.girth) * (0.07474969 - Q.M3)
    if Q.girth < 0.1484084 and Q.tau2 > 0.008780509:
        z += 209.6605 * (0.1484084 - Q.girth) * (Q.tau2 - 0.008780509)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -336.698 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.08558717 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.0006170236 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0002927736 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.6445626
    if Q.planar_flow < 0.1115136:
        z += -11.59179 * Q.planar_flow + 1.292642
    if Q.z_dr_0p05_0p1 < 0.163898:
        z += 1.78061 * Q.z_dr_0p05_0p1 - 0.2415901
    if 0.163898 <= Q.z_dr_0p05_0p1 < 0.5882598:
        z += -0.1184094 * Q.z_dr_0p05_0p1 + 0.06965549
    if 0.7143804 <= Q.psi_0p1 < 0.9761279:
        z += 0.9551651 * Q.psi_0p1 - 0.6823512
    if Q.psi_0p1 >= 0.9761279:
        z += -30.00373 * Q.psi_0p1 + 29.53749
    if 0.0009641429 <= Q.girth2 < 0.007520088:
        z += -98.52373 * Q.girth2 + 0.09499095
    if 0.007520088 <= Q.girth2 < 0.008678045:
        z += -1321.402 * Q.girth2 + 9.291142
    if Q.girth2 >= 0.008678045:
        z += -1234.114 * Q.girth2 + 8.533652
    if 0.002074109 <= Q.e2_sq < 0.0030133:
        z += 315.9721 * Q.e2_sq - 0.6553607
    if 0.0030133 <= Q.e2_sq < 0.01165737:
        z += -5.74527 * Q.e2_sq + 0.3140704
    if Q.e2_sq >= 0.01165737:
        z += 620.2048 * Q.e2_sq - 6.982864
    if 0.003562611 <= Q.width < 0.005590289:
        z += 427.1329 * Q.width - 1.521709
    if 0.005590289 <= Q.width < 0.006679471:
        z += 387.1862 * Q.width - 1.298395
    if 0.006679471 <= Q.width < 0.01323868:
        z += -978.2779 * Q.width + 7.822183
    if Q.width >= 0.01323868:
        z += -982.4941 * Q.width + 7.878001
    if 0.01655442 <= Q.e2 < 0.04110972:
        z += -94.32066 * Q.e2 + 1.561424
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += 4.108337 * Q.e2 - 2.484964
    if Q.e2 >= 0.05028464:
        z += 35.50843 * Q.e2 - 4.063907
    if 0.02685622 <= Q.centroid_offset < 0.03776099:
        z += -60.41233 * Q.centroid_offset + 1.622447
    if 0.03776099 <= Q.centroid_offset < 0.04990367:
        z += -197.4749 * Q.centroid_offset + 6.798065
    if Q.centroid_offset >= 0.04990367:
        z += -619.7695 * Q.centroid_offset + 27.87212
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 157.3656 * Q.mass_over_sum_pt - 12.57725
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 340.9681 * Q.mass_over_sum_pt - 28.13785
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 276.313 * Q.mass_over_sum_pt - 22.29214
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1057437 * Q.n_dr_0_0p05 + 0.5287187
    if Q.e3 < 5.334511e-05:
        z += 15096.66 * Q.e3 - 1.052019
    if 5.334511e-05 <= Q.e3 < 8.147744e-05:
        z += 8768.758 * Q.e3 - 0.7144559
    if Q.sd_mass < 49.91626:
        z += -0.04062359 * Q.sd_mass + 1.833125
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.007893351 * Q.sd_mass - 0.5886595
    if 0.02689598 <= Q.girth < 0.04081947:
        z += 64.47173 * Q.girth - 1.73403
    if 0.04081947 <= Q.girth < 0.08723651:
        z += 107.1422 * Q.girth - 3.475815
    if Q.girth >= 0.08723651:
        z += -16.59356 * Q.girth + 7.318458
    if Q.mass >= 69.61135:
        z += -0.04279044 * Q.mass + 2.978701
    if Q.mass_top5 >= 49.18618:
        z += 0.02772794 * Q.mass_top5 - 1.363832
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -5.995292 * Q.sj2_dr + 0.3689584
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -7.879936 * Q.sj2_dr + 0.6130014
    if 0.1591713 <= Q.sj2_dr < 0.1778793:
        z += 40.80598 * Q.sj2_dr - 7.136399
    if 0.1778793 <= Q.sj2_dr < 0.2001708:
        z += 23.2814 * Q.sj2_dr - 4.019139
    if Q.sj2_dr >= 0.2001708:
        z += -15.6334 * Q.sj2_dr + 3.770467
    if Q.lam2 < 0.000537286:
        z += -1305.461 * Q.lam2 + 0.701406
    if Q.C2_b2 < 0.004032342:
        z += 184.8651 * Q.C2_b2 - 0.7454394
    if Q.LHA >= 0.3033137:
        z += 18.10053 * Q.LHA - 5.490139
    if Q.sum_pt < 715.4688:
        z += 0.00449649 * Q.sum_pt - 3.217098
    if Q.sd_rg < 0.2330919:
        z += 8.400363 * Q.sd_rg - 2.727145
    if 0.2330919 <= Q.sd_rg < 0.2787955:
        z += -6.913018 * Q.sd_rg + 0.8422804
    if 0.2787955 <= Q.sd_rg < 0.324646:
        z += 19.57959 * Q.sd_rg - 6.54374
    if Q.sd_rg >= 0.324646:
        z += 11.17923 * Q.sd_rg - 3.816596
    if Q.z_dr_0_0p05 >= 0.1515405:
        z += 0.7640852 * Q.z_dr_0_0p05 - 0.1157898
    if Q.z_dr_0p1_0p2 < 0.07870506:
        z += -8.276841 * Q.z_dr_0p1_0p2 + 0.6514293
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -5172.187 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.width < 0.00609665:
        z += 4458.944 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.width)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.01538619 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -416.2855 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.planar_flow < 0.1115136 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -4.874055 * (0.1115136 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 8.459139 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 152094.0 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.1591713 and Q.D2_b2 < 0.08499387:
        z += 338.8912 * (Q.sj2_dr - 0.1591713) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.2001708 and Q.D2_b2 < 0.08499387:
        z += -230.7604 * (Q.sj2_dr - 0.2001708) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1294903 and Q.D2_b2 < 0.08499387:
        z += -143.0821 * (Q.sj2_dr - 0.1294903) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1591713 and Q.dr_3 < 0.02358801:
        z += -503.7793 * (Q.sj2_dr - 0.1591713) * (0.02358801 - Q.dr_3)
    if Q.sj2_dr > 0.2001708 and Q.dr_3 < 0.05268713:
        z += 236.8591 * (Q.sj2_dr - 0.2001708) * (0.05268713 - Q.dr_3)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.sum_pt < 715.4688:
        z += 0.00294175 * (0.5882598 - Q.z_dr_0p05_0p1) * (715.4688 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.C2_b2 < 0.0008333816:
        z += 670696.7 * (Q.centroid_offset - 0.04990367) * (0.0008333816 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.C2_b2 < 0.02415398:
        z += -385.9906 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.02415398 - Q.C2_b2)
    if Q.sd_rg > 0.2787955 and Q.D2_b2 < 0.2669656:
        z += -44.43142 * (Q.sd_rg - 0.2787955) * (0.2669656 - Q.D2_b2)
    if Q.sj2_dr > 0.1591713 and Q.dr_2 < 0.03293672:
        z += -589.0591 * (Q.sj2_dr - 0.1591713) * (0.03293672 - Q.dr_2)
    if Q.sj2_dr > 0.2001708 and Q.dr_2 < 0.05056028:
        z += 580.1418 * (Q.sj2_dr - 0.2001708) * (0.05056028 - Q.dr_2)
    if Q.centroid_offset > 0.02685622 and Q.pt_1 < 150.625:
        z += 0.6189116 * (Q.centroid_offset - 0.02685622) * (150.625 - Q.pt_1)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.7092252
    if Q.N2 < 0.2233283:
        z += 4.383552 * Q.N2 - 0.9789709
    if Q.girth2 < 0.007520088:
        z += 949.0579 * Q.girth2 - 7.136999
    if Q.mass_over_sum_pt < 0.1309286:
        z += 79.22995 * Q.mass_over_sum_pt - 10.37347
    if Q.width < 0.00609665:
        z += -1032.462 * Q.width + 18.59398
    if 0.00609665 <= Q.width < 0.01323868:
        z += -1414.893 * Q.width + 20.92553
    if 0.01323868 <= Q.width < 0.01882765:
        z += -392.5982 * Q.width + 7.391703
    if Q.e2 < 0.04110972:
        z += -128.2402 * Q.e2 + 5.271917
    if Q.e2_sq < 0.008168571:
        z += 1062.104 * Q.e2_sq - 10.98932
    if 0.008168571 <= Q.e2_sq < 0.01165737:
        z += 663.1076 * Q.e2_sq - 7.730094
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -920.4695 * Q.mass_over_sum_pt_sq + 6.611582
    if Q.lam1 < 0.002464291:
        z += -771.6766 * Q.lam1 + 3.376599
    if 0.002464291 <= Q.lam1 < 0.00483998:
        z += -620.8569 * Q.lam1 + 3.004935
    if Q.lam2 < 0.001130645:
        z += -853.1076 * Q.lam2 + 0.9645616
    if Q.girth2_top3 < 0.002151568:
        z += 127.4551 * Q.girth2_top3 - 0.2742283
    if Q.girth < 0.1019409:
        z += 84.11788 * Q.girth - 8.575055
    if Q.sj2_dr < 0.1492731:
        z += -15.08523 * Q.sj2_dr + 1.546973
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -38.63191 * Q.sj2_dr + 5.061859
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 26.5182 * Q.sj2_dr - 5.308168
    if Q.sj3_pair_mass_max >= 80.4:
        z += -0.04197802 * Q.sj3_pair_mass_max + 3.375033
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -5.211706 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.girth2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -2583.506 * (0.007520088 - Q.girth2) * (0.7459513 - Q.D2)
    if Q.width < 0.00609665 and Q.D2 < 0.7459513:
        z += 2337.677 * (0.00609665 - Q.width) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 41.67503 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.z_dr_0p1_0p2 > 0.4684459:
        z += 73.89314 * (0.1309286 - Q.mass_over_sum_pt) * (Q.z_dr_0p1_0p2 - 0.4684459)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.02188537 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
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
