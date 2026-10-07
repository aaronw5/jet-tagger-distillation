"""JEDI-linear jet tagger, 8 particles, 3 features: one term per observable per neuron (from the 782), re-tuned on the network's predictions (all observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 89.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
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
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.pair_mass_0_7          mass of particles 0 and 7 [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
  Q.zdr_2                  pT share × ΔR of particle 2 (its term in ΣzΔR)
  Q.zdr_5                  pT share × ΔR of particle 5 (its term in ΣzΔR)
  Q.zdr_6                  pT share × ΔR of particle 6 (its term in ΣzΔR)
  Q.zdr_7                  pT share × ΔR of particle 7 (its term in ΣzΔR)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_1               |Δη| of particle 1
  Q.abseta_4               |Δη| of particle 4
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.absphi_1               |Δφ| of particle 1
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr1_6                  ΔR between particle 6 and the 2nd-hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr02                   ΔR between particles 0 and 2
  Q.dr12                   ΔR between particles 1 and 2
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
        pair_mass_0_4=pair_mass(0, 4),
        pair_mass_0_6=pair_mass(0, 6),
        pair_mass_0_7=pair_mass(0, 7),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
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
        sd_zg=softdrop("zg"),
        abseta_0=abs(eta[0]),
        abseta_1=abs(eta[1]),
        abseta_4=abs(eta[4]),
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        absphi_1=abs(phi[1]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr1_6=math.sqrt(dist2(1, 6)) if pt[6] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr02=math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
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
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -7.58129
    if Q.planar_flow < 0.1484197:
        z += -10.21213 * Q.planar_flow + 1.515681
    if Q.lam1_plus_lam2 < 0.002631161:
        z += 1328.478 * Q.lam1_plus_lam2 - 3.49544
    if Q.sum_z_dr2 < 0.01279734:
        z += -383.8806 * Q.sum_z_dr2 + 4.91265
    if Q.mass < 41.26805:
        z += 0.03096775 * Q.mass - 1.277978
    if Q.sum_z_dr2_top3 < 0.002919842:
        z += -170.8396 * Q.sum_z_dr2_top3 + 0.4988246
    if Q.lam1 < 0.00543239:
        z += 216.3184 * Q.lam1 - 1.175126
    if Q.sum_pt >= 901.5938:
        z += -0.01574453 * Q.sum_pt + 14.19517
    if Q.C2_b2 < 0.001563465:
        z += 672.9724 * Q.C2_b2 - 1.052169
    if Q.sj3_dr_max >= 0.2321253:
        z += -14.16047 * Q.sj3_dr_max + 3.287003
    if Q.centroid_offset >= 0.03782766:
        z += -188.3442 * Q.centroid_offset + 7.124619
    if Q.sum_z_dr < 0.08061324:
        z += 36.96094 * Q.sum_z_dr - 2.979541
    if Q.e2 < 0.03227056:
        z += -163.7386 * Q.e2 + 5.283936
    z += 1.129251 * Q.log_sum_pt
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += 326.4479 * Q.mass_over_sum_pt_sq - 1.274646
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.002349652 * Q.sum_pt_top5 - 1.615239
    if Q.lam2 < 7.300726e-05:
        z += 22605.65 * Q.lam2 - 1.650376
    if Q.mass_over_sum_pt < 0.07988007:
        z += 42.62242 * Q.mass_over_sum_pt - 3.404682
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += 17.22889 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -1420.42 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 1.002471:
        z += 351.7556 * (0.01323868 - Q.sum_z_dr2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.04817318 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.sum_z_dr2 < 0.01323868 and Q.phi_0 > -0.008995056:
        z += 846.3049 * (0.01323868 - Q.sum_z_dr2) * (Q.phi_0 - -0.008995056)
    if Q.planar_flow < 0.1484197 and Q.z_7 < 0.02320757:
        z += 499.5254 * (0.1484197 - Q.planar_flow) * (0.02320757 - Q.z_7)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 119.2525 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.mass < 64.61873 and Q.D2_b2 < 0.1830092:
        z += -0.2262945 * (64.61873 - Q.mass) * (0.1830092 - Q.D2_b2)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 130085.7 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -104.285 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.planar_flow < 0.1484197 and Q.dr0_6 > 0.1755206:
        z += -31.00965 * (0.1484197 - Q.planar_flow) * (Q.dr0_6 - 0.1755206)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_4 < 0.06336451:
        z += -0.07466417 * (Q.sum_pt_top5 - 687.4375) * (0.06336451 - Q.dr_4)
    if Q.lam2 < 7.300726e-05 and Q.dr_2 < 0.08525808:
        z += 82051.62 * (7.300726e-05 - Q.lam2) * (0.08525808 - Q.dr_2)
    if Q.sum_pt > 901.5938 and Q.pt_7 > 33.21875:
        z += 0.0002391899 * (Q.sum_pt - 901.5938) * (Q.pt_7 - 33.21875)
    if Q.log_sum_pt > 6.670067 and Q.z_7 > 0.01685855:
        z += -407.6438 * (Q.log_sum_pt - 6.670067) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.3012016 and Q.z_7 < 0.05557716:
        z += -44.22107 * (0.3012016 - Q.sj3_dr_max) * (0.05557716 - Q.z_7)
    if Q.log_sum_pt > 6.670067 and Q.m01 < 28.78966:
        z += 0.07302918 * (Q.log_sum_pt - 6.670067) * (28.78966 - Q.m01)
    if Q.planar_flow < 0.1484197 and Q.dr_2 < 0.02270492:
        z += -476.5812 * (0.1484197 - Q.planar_flow) * (0.02270492 - Q.dr_2)
    if Q.sum_z_dr2_top3 < 0.007929074 and Q.D2_b2 < 0.5327104:
        z += -101.0988 * (0.007929074 - Q.sum_z_dr2_top3) * (0.5327104 - Q.D2_b2)
    if Q.centroid_offset > 0.02076709 and Q.z_7 < 0.06473447:
        z += -2217.889 * (Q.centroid_offset - 0.02076709) * (0.06473447 - Q.z_7)
    if Q.planar_flow < 0.1484197 and Q.pt_dispersion < 0.4160096:
        z += 88.78204 * (0.1484197 - Q.planar_flow) * (0.4160096 - Q.pt_dispersion)
    if Q.centroid_offset > 0.006789738 and Q.n_pt_above_50 < 7.0:
        z += -4.613132 * (Q.centroid_offset - 0.006789738) * (7.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.02076709 and Q.dr_2 < 0.05056028:
        z += -2860.569 * (Q.centroid_offset - 0.02076709) * (0.05056028 - Q.dr_2)
    if Q.lam1_plus_lam2 < 0.004372139 and Q.C3 < 0.03532852:
        z += 4816.303 * (0.004372139 - Q.lam1_plus_lam2) * (0.03532852 - Q.C3)
    if Q.centroid_offset > 0.006789738 and Q.pt_1 < 223.375:
        z += -0.1653237 * (Q.centroid_offset - 0.006789738) * (223.375 - Q.pt_1)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.9099477
    if Q.lam1 < 0.008291375:
        z += -180.3408 * Q.lam1 + 1.495273
    if Q.log_sum_pt >= 6.503015:
        z += 15.50402 * Q.log_sum_pt - 100.8229
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -846.0127 * Q.mass_over_sum_pt_sq + 4.934734
    if Q.z_7 < 0.07141996:
        z += 25.29267 * Q.z_7 - 1.806401
    if Q.sum_z_dr2 < 0.008678045:
        z += 528.42 * Q.sum_z_dr2 - 4.585652
    if Q.mass < 49.6681:
        z += -0.004635407 * Q.mass + 0.2302318
    if Q.sj3_dr_max >= 0.169029:
        z += 2.892759 * Q.sj3_dr_max - 0.4889602
    if Q.sum_zz_dr2 < 0.008168571:
        z += -23.50815 * Q.sum_zz_dr2 + 0.192028
    if Q.pt_7 < 25.53125:
        z += 0.06695624 * Q.pt_7 - 1.709476
    if Q.sum_pt_top5 >= 555.7359:
        z += -0.008469363 * Q.sum_pt_top5 + 4.706729
    if Q.lam1_plus_lam2 < 0.00609665:
        z += 299.3494 * Q.lam1_plus_lam2 - 1.825029
    if Q.sum_z_dr < 0.1019409:
        z += 20.10278 * Q.sum_z_dr - 2.049296
    if Q.tau1 < 0.07283629:
        z += 19.44213 * Q.tau1 - 1.416093
    if Q.e3 < 0.0005116989:
        z += -1950.272 * Q.e3 + 0.9979521
    if Q.zdr_0 < 0.0211821:
        z += 12.40615 * Q.zdr_0 - 0.2627884
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += -12813.68 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += -0.8569544 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.tau2 < 0.06297984:
        z += -65.00107 * (0.06164517 - Q.z_7) * (0.06297984 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -26.0588 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.5069531 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -2.275063 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -22.65223 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.005507326 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.004623493 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 85.28325 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 2.957858 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -354.3083 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.09326502 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.D2_b2 < 0.03885671:
        z += -28006.68 * (0.005832932 - Q.mass_over_sum_pt_sq) * (0.03885671 - Q.D2_b2)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += 0.4825808 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.sum_z_dr2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 20684.17 * (0.008678045 - Q.sum_z_dr2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -4857.204 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 1061.047 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    if Q.log_sum_pt > 6.502799 and Q.zdr_6 < 0.008654951:
        z += -616.6774 * (Q.log_sum_pt - 6.502799) * (0.008654951 - Q.zdr_6)
    if Q.sum_z_dr < 0.1019409 and Q.pair_mass_0_6 > 4.037975:
        z += -0.2880232 * (0.1019409 - Q.sum_z_dr) * (Q.pair_mass_0_6 - 4.037975)
    if Q.sum_pt_top5 > 531.1875 and Q.zdr_6 < 0.01392641:
        z += 0.4993469 * (Q.sum_pt_top5 - 531.1875) * (0.01392641 - Q.zdr_6)
    if Q.z_7 < 0.06164517 and Q.abseta_0 < 0.1057739:
        z += 16.02969 * (0.06164517 - Q.z_7) * (0.1057739 - Q.abseta_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.8168668
    if Q.sj3_pair_mass_max < 62.55:
        z += -0.02552162 * Q.sj3_pair_mass_max + 1.596378
    if Q.log_sum_pt < 6.572982:
        z += -5.285773 * Q.log_sum_pt + 34.74329
    if Q.sum_z_dr < 0.007673833:
        z += 298.3163 * Q.sum_z_dr - 2.289229
    if Q.pt_7 >= 20.09375:
        z += 0.07423168 * Q.pt_7 - 1.491593
    if Q.LHA >= 0.1329373:
        z += -8.707918 * Q.LHA + 1.157607
    if Q.z_7 >= 0.03971184:
        z += -29.54341 * Q.z_7 + 1.173223
    if Q.planar_flow < 0.4926918:
        z += 0.8096178 * Q.planar_flow - 0.398892
    if Q.mass < 36.22941:
        z += 0.040267 * Q.mass - 1.45885
    if Q.max_dr < 0.2507612:
        z += 1.311709 * Q.max_dr - 0.3289257
    if Q.sum_z_dr2 < 0.003562611:
        z += 7.787301 * Q.sum_z_dr2 - 0.02774313
    if Q.m012 >= 40.2:
        z += 0.02456018 * Q.m012 - 0.9873191
    if Q.sum_pt < 715.5:
        z += -0.00300343 * Q.sum_pt + 2.148954
    if Q.sum_pt_top5 >= 752.1:
        z += -0.002816642 * Q.sum_pt_top5 + 2.118397
    if Q.zdr_0 < 0.0211821:
        z += -21.52935 * Q.zdr_0 + 0.4560369
    if Q.mass_over_sum_pt < 0.1079857:
        z += -13.98268 * Q.mass_over_sum_pt + 1.509929
    if Q.lam2 < 0.0001947983:
        z += -1854.494 * Q.lam2 + 0.3612523
    if Q.lam1 < 0.001503553:
        z += -109.545 * Q.lam1 + 0.1647067
    if Q.sj3_pair_mass_max < 62.55 and Q.z_7 < 0.06810151:
        z += -0.1383403 * (62.55 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 62.55 and Q.centroid_offset > 0.01096064:
        z += -0.6779323 * (62.55 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.08050702:
        z += -1837.876 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.08050702)
    if Q.z_7 > 0.04939969 and Q.sj3_dr_min < 0.0623951:
        z += -119.1405 * (Q.z_7 - 0.04939969) * (0.0623951 - Q.sj3_dr_min)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += -0.1956673 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.z_7 < 0.03243272 and Q.mass_top3 > 16.89912:
        z += -1.370003 * (0.03243272 - Q.z_7) * (Q.mass_top3 - 16.89912)
    if Q.lam1 < 0.00595415 and Q.pt_6 > 19.46875:
        z += 3.521377 * (0.00595415 - Q.lam1) * (Q.pt_6 - 19.46875)
    if Q.sum_z_dr < 0.007673833 and Q.pt_4 < 75.625:
        z += 3.84496 * (0.007673833 - Q.sum_z_dr) * (75.625 - Q.pt_4)
    if Q.mass < 36.22941 and Q.pt_5 < 73.75:
        z += 0.0002881623 * (36.22941 - Q.mass) * (73.75 - Q.pt_5)
    if Q.log_sum_pt > 6.842717 and Q.D2_b2 < 1.345805:
        z += -12.01611 * (Q.log_sum_pt - 6.842717) * (1.345805 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.z_dr_0p05_0p1 < 0.4474937:
        z += 6.178779 * (Q.log_sum_pt - 6.842717) * (0.4474937 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top5 > 752.1 and Q.D2_b2 < 1.129616:
        z += 0.01269362 * (Q.sum_pt_top5 - 752.1) * (1.129616 - Q.D2_b2)
    if Q.sum_pt_top5 > 752.1 and Q.abseta_0 < 0.0174408:
        z += -0.4569733 * (Q.sum_pt_top5 - 752.1) * (0.0174408 - Q.abseta_0)
    if Q.log_sum_pt > 6.842717 and Q.abseta_0 < 0.0174408:
        z += 554.8763 * (Q.log_sum_pt - 6.842717) * (0.0174408 - Q.abseta_0)
    if Q.sum_pt > 527.1781 and Q.lam2 < 0.0001947983:
        z += -9.933988 * (Q.sum_pt - 527.1781) * (0.0001947983 - Q.lam2)
    if Q.log_sum_pt > 6.842717 and Q.dr02 > 0.2623104:
        z += -26.20595 * (Q.log_sum_pt - 6.842717) * (Q.dr02 - 0.2623104)
    if Q.log_sum_pt > 6.842717 and Q.lam2 < 0.0001947983:
        z += 36383.01 * (Q.log_sum_pt - 6.842717) * (0.0001947983 - Q.lam2)
    return max(0.0, z)


def neuron_3(Q):
    z = 3.086719
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 77.42875 * Q.mass_over_sum_pt - 5.275925
    if Q.centroid_offset >= 0.01096064:
        z += 66.21967 * Q.centroid_offset - 0.7258098
    if Q.tau1 >= 0.03461338:
        z += -28.04396 * Q.tau1 + 0.9706961
    if Q.lam1 >= 0.01167645:
        z += 485.5971 * Q.lam1 - 5.670052
    if Q.sum_z_dr < 0.1019409:
        z += 112.2873 * Q.sum_z_dr - 11.44667
    if Q.lam1_plus_lam2 >= 0.01279734:
        z += 148.8533 * Q.lam1_plus_lam2 - 1.904926
    if Q.e2 >= 0.04447357:
        z += 253.2803 * Q.e2 - 11.26428
    if Q.sum_z_dr2 >= 0.008566102:
        z += -1460.295 * Q.sum_z_dr2 + 12.50903
    if Q.sj2_dr >= 0.1488981:
        z += 23.78496 * Q.sj2_dr - 3.541537
    if Q.mean_eta < -0.004664942:
        z += -19.36666 * Q.mean_eta - 0.09034433
    if Q.sum_z_dr2_top5 < 0.01115288:
        z += 16.3845 * Q.sum_z_dr2_top5 - 0.1827344
    if Q.mass >= 64.61873:
        z += -0.08247673 * Q.mass + 5.329542
    if Q.sd_mass >= 54.15028:
        z += 0.00388428 * Q.sd_mass - 0.2103348
    if Q.sj3_pair_mass_min >= 11.051:
        z += 0.06316762 * Q.sj3_pair_mass_min - 0.6980655
    if Q.sj3_dr_max >= 0.1874999:
        z += 35.69669 * Q.sj3_dr_max - 6.693126
    if Q.n_dr_0_0p05 < 1.0:
        z += -0.7513551 * Q.n_dr_0_0p05 + 0.7513551
    if Q.z_dr_0_0p05 < 0.9008535:
        z += 1.342033 * Q.z_dr_0_0p05 - 1.208975
    if Q.mass_over_sum_pt_sq < 0.01683059:
        z += 197.7689 * Q.mass_over_sum_pt_sq - 3.328567
    if Q.LHA >= 0.3467135:
        z += 6.025398 * Q.LHA - 2.089087
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += -1.936599 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += 0.2042157 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 57.03863 * (Q.sum_z_dr - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 4.677983 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.6508048 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -2143.133 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.sj2_dr > 0.1872617 and Q.pt_6 < 27.57812:
        z += -0.8956634 * (Q.sj2_dr - 0.1872617) * (27.57812 - Q.pt_6)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 490.2005 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 21.02516 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.55 and Q.D2_b2 < 0.9206502:
        z += 0.0480459 * (Q.sd_mass - 62.55) * (0.9206502 - Q.D2_b2)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -208.3545 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.sum_z_dr > 0.07608178 and Q.eta_0 > 0.07952881:
        z += 224.6341 * (Q.sum_z_dr - 0.07608178) * (Q.eta_0 - 0.07952881)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05441452:
        z += 6584.898 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05441452)
    if Q.max_dr > 0.1027585 and Q.z_6 < 0.06081235:
        z += 150.804 * (Q.max_dr - 0.1027585) * (0.06081235 - Q.z_6)
    if Q.max_dr > 0.1027585 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -16.24521 * (Q.max_dr - 0.1027585) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.max_dr > 0.1027585 and Q.eta_1 < -0.0892334:
        z += -55.17797 * (Q.max_dr - 0.1027585) * (-0.0892334 - Q.eta_1)
    if Q.max_dr > 0.1027585 and Q.abseta_0 > 0.1057739:
        z += -88.19246 * (Q.max_dr - 0.1027585) * (Q.abseta_0 - 0.1057739)
    if Q.sj2_dr > 0.2687922 and Q.z_dr_0p05_0p1 < 0.9641201:
        z += -6.03385 * (Q.sj2_dr - 0.2687922) * (0.9641201 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0_0p05 < 1.0 and Q.n_dr_0p05_0p1 < 4.0:
        z += 0.2061518 * (1.0 - Q.n_dr_0_0p05) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.sum_z_dr2 > 0.01882765 and Q.z_dr_0p05_0p1 > 0.7509095:
        z += -5023.111 * (Q.sum_z_dr2 - 0.01882765) * (Q.z_dr_0p05_0p1 - 0.7509095)
    if Q.sum_z_dr > 0.08723651 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -305.9441 * (Q.sum_z_dr - 0.08723651) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.centroid_offset > 0.01096064 and Q.abseta_7 < 0.1626038:
        z += -84.98287 * (Q.centroid_offset - 0.01096064) * (0.1626038 - Q.abseta_7)
    return max(0.0, z)


def neuron_4(Q):
    z = -2.877921
    if Q.N2 < 0.2233283:
        z += -48.63252 * Q.N2 + 10.86102
    if Q.lam2 < 0.001100839:
        z += 2176.195 * Q.lam2 - 2.395641
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -106.0846 * Q.mass_over_sum_pt + 9.591511
    if Q.lam1_plus_lam2 >= 0.006688642:
        z += 368.7775 * Q.lam1_plus_lam2 - 2.466621
    if Q.e2 < 0.06278829:
        z += 64.12829 * Q.e2 - 4.026506
    if Q.sum_pt < 739.5:
        z += 0.002321025 * Q.sum_pt - 1.716398
    if Q.max_dr >= 0.06139287:
        z += 13.82047 * Q.max_dr - 0.8484783
    if Q.C2 < 0.06729223:
        z += -107.2415 * Q.C2 + 7.216522
    if Q.sd_mass >= 86.29383:
        z += -0.1117363 * Q.sd_mass + 9.64215
    if Q.e3 < 0.0001869378:
        z += -3422.008 * Q.e3 + 0.6397027
    if Q.sj3_dr_max < 0.2122942:
        z += 20.3636 * Q.sj3_dr_max - 4.323073
    if Q.sum_z_dr2 < 0.008678045:
        z += 922.2606 * Q.sum_z_dr2 - 8.003419
    if Q.sum_z_dr2_top5 < 0.01662178:
        z += -216.564 * Q.sum_z_dr2_top5 + 3.59968
    if Q.centroid_offset < 0.04990367:
        z += 50.18908 * Q.centroid_offset - 2.504619
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -217.587 * Q.sum_z_dr2_top2 + 1.662287
    if Q.mass >= 45.7571:
        z += -0.0217449 * Q.mass + 0.9949833
    if Q.sum_zz_dr2 < 0.01165737:
        z += -3177.223 * Q.sum_zz_dr2 + 37.03808
    if Q.tau1 < 0.07283629:
        z += -33.63163 * Q.tau1 + 2.449603
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 2569.706 * Q.mass_over_sum_pt_sq - 29.9651
    if Q.LHA < 0.3467135:
        z += -11.32811 * Q.LHA + 3.927609
    if Q.lam1 < 0.001503553:
        z += -175.1956 * Q.lam1 + 0.2634159
    if Q.mass_top5 >= 14.54404:
        z += 0.02321453 * Q.mass_top5 - 0.337633
    if Q.M2 < 0.03874536:
        z += 43.07614 * Q.M2 - 1.669001
    if Q.N2 < 0.2233283 and Q.mass < 62.55:
        z += -0.9254985 * (0.2233283 - Q.N2) * (62.55 - Q.mass)
    if Q.N2 < 0.2233283 and Q.sum_zz_dr2 > 0.01165737:
        z += -1422.963 * (0.2233283 - Q.N2) * (Q.sum_zz_dr2 - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.3543214 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 212.1348 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -81.64442 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1218872:
        z += -38.43246 * (0.2233283 - Q.N2) * (0.1218872 - Q.abseta_7)
    if Q.sd_mass > 44.82259 and Q.centroid_offset > 0.001308549:
        z += -1.215295 * (Q.sd_mass - 44.82259) * (Q.centroid_offset - 0.001308549)
    if Q.sd_mass > 44.82259 and Q.sd_zg < 0.275762:
        z += -0.3081537 * (Q.sd_mass - 44.82259) * (0.275762 - Q.sd_zg)
    if Q.lam1_plus_lam2 > 0.003562611 and Q.sj3_z3 < 0.1057566:
        z += -1014.021 * (Q.lam1_plus_lam2 - 0.003562611) * (0.1057566 - Q.sj3_z3)
    if Q.lam2 < 0.000537286 and Q.dr01 < 0.1410336:
        z += -17286.33 * (0.000537286 - Q.lam2) * (0.1410336 - Q.dr01)
    if Q.max_dr > 0.1598486 and Q.C2_b2 < 0.0006435798:
        z += -41811.49 * (Q.max_dr - 0.1598486) * (0.0006435798 - Q.C2_b2)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.C2_b2 > 0.0006435798:
        z += -9565.76 * (0.007639643 - Q.sum_z_dr2_top2) * (Q.C2_b2 - 0.0006435798)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += -0.5757764 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.lam2 < 0.000537286 and Q.dr1_6 < 0.07796252:
        z += -12778.9 * (0.000537286 - Q.lam2) * (0.07796252 - Q.dr1_6)
    if Q.mass > 76.6557 and Q.M3 < 0.107953:
        z += 0.9330773 * (Q.mass - 76.6557) * (0.107953 - Q.M3)
    if Q.sd_mass > 44.82259 and Q.sj3_pair_mass_min < 27.42324:
        z += 0.002612316 * (Q.sd_mass - 44.82259) * (27.42324 - Q.sj3_pair_mass_min)
    if Q.sj3_dr_max < 0.233678 and Q.ptdr0_5 > 7.737156:
        z += 2.34605 * (0.233678 - Q.sj3_dr_max) * (Q.ptdr0_5 - 7.737156)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.006675322
    if Q.LHA < 0.2160559:
        z += -15.5782 * Q.LHA + 3.365762
    if Q.z_7 < 0.0553074:
        z += -116.0292 * Q.z_7 + 6.417275
    if Q.log_sum_pt >= 6.80271:
        z += -27.04131 * Q.log_sum_pt + 183.9542
    if Q.sum_zz_dr2 < 0.01122239:
        z += -40.71033 * Q.sum_zz_dr2 + 0.4568672
    if Q.sum_z_dr2 < 0.001653836:
        z += -1192.388 * Q.sum_z_dr2 + 1.972014
    if Q.sum_z_dr < 0.007673833:
        z += -59.7636 * Q.sum_z_dr + 0.4586159
    if Q.pt_7 < 37.15625:
        z += 0.1199728 * Q.pt_7 - 4.457739
    if Q.zdr_0 < 0.0211821:
        z += 39.7143 * Q.zdr_0 - 0.8412324
    if Q.pair_mass_0_4 >= 23.26771:
        z += -0.09856932 * Q.pair_mass_0_4 + 2.293482
    if Q.sum_pt_top5 >= 752.2188:
        z += 0.004674412 * Q.sum_pt_top5 - 3.516181
    if Q.pt_5 < 24.57812:
        z += -0.2174269 * Q.pt_5 + 5.343947
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -123.9105 * Q.lam1_plus_lam2 + 0.4414449
    if Q.centroid_offset < 0.006789738:
        z += -81.31406 * Q.centroid_offset + 0.5521012
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 315.1024 * Q.sum_z_dr2_top3 - 0.6779643
    if Q.z_6 < 0.08051087:
        z += 20.67055 * Q.z_6 - 1.664204
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -88.6115 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -327.481 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.04939969 and Q.sum_pt_top2 < 501.625:
        z += -0.2532117 * (0.04939969 - Q.z_7) * (501.625 - Q.sum_pt_top2)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 40.2:
        z += 0.5344998 * (0.07148865 - Q.z_7) * (40.2 - Q.mass_top3)
    if Q.sum_z_dr2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 119318.6 * (0.001653836 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.LHA < 0.2160559 and Q.lam1 < 0.001503553:
        z += -16781.86 * (0.2160559 - Q.LHA) * (0.001503553 - Q.lam1)
    if Q.log_sum_pt > 6.701242 and Q.dr_2 < 0.01341502:
        z += 761.3914 * (Q.log_sum_pt - 6.701242) * (0.01341502 - Q.dr_2)
    if Q.LHA < 0.2160559 and Q.n_dr_0p2_0p4 > 0.0:
        z += -11.85206 * (0.2160559 - Q.LHA) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.sum_pt_top5 > 430.75 and Q.centroid_offset > 0.009480685:
        z += 0.04530506 * (Q.sum_pt_top5 - 430.75) * (Q.centroid_offset - 0.009480685)
    if Q.z_7 < 0.04939969 and Q.dr_2 < 0.009661512:
        z += -3282.908 * (0.04939969 - Q.z_7) * (0.009661512 - Q.dr_2)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi > -0.02594505:
        z += -287.2567 * (Q.log_sum_pt - 6.896095) * (Q.mean_phi - -0.02594505)
    if Q.sum_pt_top5 > 430.75 and Q.sj2_dr > 0.1682655:
        z += 0.02376036 * (Q.sum_pt_top5 - 430.75) * (Q.sj2_dr - 0.1682655)
    if Q.pt_7 < 37.15625 and Q.pt_5 > 24.57812:
        z += 0.001134193 * (37.15625 - Q.pt_7) * (Q.pt_5 - 24.57812)
    if Q.sum_pt_top5 > 430.75 and Q.pt_6 < 46.125:
        z += 0.0002140506 * (Q.sum_pt_top5 - 430.75) * (46.125 - Q.pt_6)
    if Q.pair_mass_0_4 > 23.26771 and Q.eccentricity > 0.9704496:
        z += 3.516625 * (Q.pair_mass_0_4 - 23.26771) * (Q.eccentricity - 0.9704496)
    if Q.zdr_0 < 0.0211821 and Q.phi_0 > -0.0297699:
        z += 305.4245 * (0.0211821 - Q.zdr_0) * (Q.phi_0 - -0.0297699)
    if Q.z_7 < 0.07148865 and Q.mean_phi2 < 0.002127561:
        z += 6387.539 * (0.07148865 - Q.z_7) * (0.002127561 - Q.mean_phi2)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 59657.5 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.pair_mass_0_4 > 23.26771 and Q.mean_eta2 < 0.01423545:
        z += 4.978361 * (Q.pair_mass_0_4 - 23.26771) * (0.01423545 - Q.mean_eta2)
    if Q.centroid_offset < 0.006789738 and Q.pt_5 < 56.4375:
        z += -5.890854 * (0.006789738 - Q.centroid_offset) * (56.4375 - Q.pt_5)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.1471063
    if Q.centroid_offset < 0.02677766:
        z += 89.4667 * Q.centroid_offset - 2.395709
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 235.3789 * Q.lam1_plus_lam2 - 3.116105
    if Q.mass < 29.56813:
        z += -0.0634777 * Q.mass + 1.876917
    if Q.pt_6 < 19.45312:
        z += -0.5456014 * Q.pt_6 + 10.61365
    if Q.lam2 < 0.0005177258:
        z += 601.3467 * Q.lam2 - 0.3113327
    if Q.tau1 < 0.1136369:
        z += -10.75632 * Q.tau1 + 1.222314
    if Q.sj3_dr_min < 0.1278212:
        z += -15.90069 * Q.sj3_dr_min + 2.032445
    if Q.lam1 < 0.01167645:
        z += -475.8754 * Q.lam1 + 5.556536
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.06219562 * Q.sj3_pair_mass_min + 0.2799877
    if Q.sum_pt >= 988.4078:
        z += 0.02412193 * Q.sum_pt - 23.8423
    if Q.sum_zz_dr2 < 0.0030133:
        z += -464.1611 * Q.sum_zz_dr2 + 1.398657
    if Q.sj3_dr13 >= 0.181053:
        z += 1.266092 * Q.sj3_dr13 - 0.2292297
    if Q.sj3_dr_max >= 0.2321253:
        z += 15.64291 * Q.sj3_dr_max - 3.631115
    if Q.max_dr < 0.1452311:
        z += 35.24688 * Q.max_dr - 5.118943
    if Q.mean_eta >= 0.02644207:
        z += -21.94989 * Q.mean_eta + 0.5804006
    if Q.sum_z_dr2 < 0.008678045:
        z += 1123.095 * Q.sum_z_dr2 - 9.746269
    if Q.C2_b2 < 0.02415398:
        z += -68.93467 * Q.C2_b2 + 1.665047
    if Q.sj2_dr < 0.1872617:
        z += -14.24414 * Q.sj2_dr + 2.667383
    if Q.sj3_dr23 >= 0.2207152:
        z += 3.12602 * Q.sj3_dr23 - 0.6899603
    if Q.log_sum_pt >= 6.270279:
        z += 1.394072 * Q.log_sum_pt - 8.74122
    if Q.z_6 < 0.02160287:
        z += 636.2402 * Q.z_6 - 13.74462
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.01126374 * Q.sum_pt_top5 + 9.461034
    if Q.tau2 >= 0.008780509:
        z += -12.5311 * Q.tau2 + 0.1100294
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += 1.183349 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.594106 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += 41.88632 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 4320.803 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -1563.994 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -6.526233 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.pt_6 < 29.90625 and Q.n_pt_above_10 < 8.0:
        z += 0.04301843 * (29.90625 - Q.pt_6) * (8.0 - Q.n_pt_above_10)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.8629167 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.004716304 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 1.319403 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += 0.09106025 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 6206.659 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.mean_eta > 0.02644207 and Q.M3 > 0.06688759:
        z += -1409.966 * (Q.mean_eta - 0.02644207) * (Q.M3 - 0.06688759)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.01850702 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sj2_dr < 0.1872617 and Q.eccentricity > 0.927072:
        z += 246.5827 * (0.1872617 - Q.sj2_dr) * (Q.eccentricity - 0.927072)
    if Q.sj2_mass1 > 40.2 and Q.tau21_b2 < 0.4490772:
        z += -2.629432 * (Q.sj2_mass1 - 40.2) * (0.4490772 - Q.tau21_b2)
    if Q.sum_pt < 615.875 and Q.pt_5 < 24.57812:
        z += 0.0118477 * (615.875 - Q.sum_pt) * (24.57812 - Q.pt_5)
    if Q.centroid_offset > 0.01837778 and Q.pt_4 > 65.1875:
        z += 2.264823 * (Q.centroid_offset - 0.01837778) * (Q.pt_4 - 65.1875)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 < 0.002776626:
        z += -4.135308 * (Q.sum_pt - 988.4078) * (0.002776626 - Q.mean_phi2)
    if Q.tau1 < 0.1136369 and Q.mean_phi2 < 0.01426135:
        z += 1384.13 * (0.1136369 - Q.tau1) * (0.01426135 - Q.mean_phi2)
    if Q.sj2_dr < 0.1872617 and Q.dr1_6 > 0.07796252:
        z += -65.73093 * (0.1872617 - Q.sj2_dr) * (Q.dr1_6 - 0.07796252)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0753896:
        z += 93.38631 * (6.638339 - Q.log_sum_pt) * (0.0753896 - Q.z_7)
    if Q.centroid_offset > 0.01837778 and Q.sj3_dr12 > 0.1692253:
        z += -30.07765 * (Q.centroid_offset - 0.01837778) * (Q.sj3_dr12 - 0.1692253)
    return max(0.0, z)


def neuron_7(Q):
    z = 4.535524
    if Q.planar_flow < 0.1950135:
        z += 0.6384606 * Q.planar_flow - 0.1245085
    if Q.sum_z_dr2_top2 < 0.001056655:
        z += 236.5604 * Q.sum_z_dr2_top2 - 0.2499628
    if Q.sum_z_dr2 >= 0.01855051:
        z += 1125.93 * Q.sum_z_dr2 - 20.88658
    if Q.mass_over_sum_pt >= 0.1060006:
        z += -610.076 * Q.mass_over_sum_pt + 64.6684
    if Q.tau1 < 0.05356915:
        z += -22.35077 * Q.tau1 + 1.197312
    if Q.sum_z_dr < 0.08699327:
        z += 66.95837 * Q.sum_z_dr - 5.824927
    if Q.mass >= 87.93725:
        z += -0.3425506 * Q.mass + 30.12295
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -0.6925289 * Q.z_dr_0p1_0p2 + 0.1098061
    if Q.centroid_offset >= 0.0500602:
        z += -170.6118 * Q.centroid_offset + 8.540859
    if Q.lam1_plus_lam2 >= 0.01279734:
        z += -723.2496 * Q.lam1_plus_lam2 + 9.255672
    if Q.z_7 >= 0.03243272:
        z += 10.9914 * Q.z_7 - 0.3564812
    if Q.e2 >= 0.05028464:
        z += -45.62234 * Q.e2 + 2.294103
    if Q.sj2_dr < 0.09317241:
        z += -13.05629 * Q.sj2_dr + 1.216486
    if Q.sum_zz_dr2 < 0.001101266:
        z += -556.6048 * Q.sum_zz_dr2 + 0.6129701
    if Q.LHA < 0.3033137:
        z += -7.346126 * Q.LHA + 2.228181
    if Q.pt_7 < 29.04219:
        z += 0.05162052 * Q.pt_7 - 1.499173
    if Q.sd_rg >= 0.2787955:
        z += -8.772783 * Q.sd_rg + 2.445813
    if Q.lam1 < 0.008375572:
        z += 254.6555 * Q.lam1 - 2.132885
    if Q.mass_top5 >= 53.60766:
        z += 0.03439206 * Q.mass_top5 - 1.843678
    if Q.D2_b2 < 0.05744392:
        z += 13.25259 * Q.D2_b2 - 0.7612805
    if Q.sj3_dr_max >= 0.1416042:
        z += 13.15219 * Q.sj3_dr_max - 1.862405
    if Q.lam2 < 0.0003061234:
        z += 1859.58 * Q.lam2 - 0.569261
    if Q.D2 < 2.357246:
        z += 0.1193843 * Q.D2 - 0.2814183
    if Q.max_dr < 0.06139287:
        z += 23.41598 * Q.max_dr - 1.437574
    if Q.tau21_b2 < 0.04019753:
        z += 21.11696 * Q.tau21_b2 - 0.8488497
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 35.28125:
        z += -0.172731 * (0.1950135 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.2447299 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -21.37782 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.04865775 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 255.3346 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += -8040.88 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -26445.65 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.sum_z_dr < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 3745.491 * (0.08723651 - Q.sum_z_dr) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 3764.575 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    if Q.sum_z_dr2_top2 < 0.001056655 and Q.tau21_b2 < 0.02656143:
        z += -75763.0 * (0.001056655 - Q.sum_z_dr2_top2) * (0.02656143 - Q.tau21_b2)
    if Q.lam2 < 0.0003061234 and Q.tau21_b2 < 0.04019753:
        z += 129576.6 * (0.0003061234 - Q.lam2) * (0.04019753 - Q.tau21_b2)
    if Q.mass_over_sum_pt > 0.09041383 and Q.pt_6 < 36.8125:
        z += 6.058158 * (Q.mass_over_sum_pt - 0.09041383) * (36.8125 - Q.pt_6)
    if Q.sj3_dr_max > 0.1426152 and Q.pt_6 < 19.46875:
        z += -1.293218 * (Q.sj3_dr_max - 0.1426152) * (19.46875 - Q.pt_6)
    if Q.sum_z_dr2 > 0.01323868 and Q.pt_6 < 38.25:
        z += 93.18139 * (Q.sum_z_dr2 - 0.01323868) * (38.25 - Q.pt_6)
    if Q.lam1_plus_lam2 > 0.008678045 and Q.pt_6 < 38.25:
        z += -117.6433 * (Q.lam1_plus_lam2 - 0.008678045) * (38.25 - Q.pt_6)
    if Q.mass > 76.6557 and Q.zdr_6 > 0.006366792:
        z += -30.96017 * (Q.mass - 76.6557) * (Q.zdr_6 - 0.006366792)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.8951656
    if Q.sum_z_dr2 < 0.005019719:
        z += -473.3955 * Q.sum_z_dr2 + 2.376312
    if Q.tau1 < 0.05356915:
        z += 47.46025 * Q.tau1 - 2.542406
    if Q.LHA < 0.1967397:
        z += 29.27656 * Q.LHA - 5.759863
    if Q.log_sum_pt >= 6.701242:
        z += -12.95608 * Q.log_sum_pt + 86.8218
    if Q.sum_z_dr < 0.06108601:
        z += 79.40762 * Q.sum_z_dr - 4.850695
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.002607795 * Q.sum_pt_top5 - 1.792696
    if Q.mass < 49.6681:
        z += 0.02784729 * Q.mass - 1.383122
    if Q.lam1_plus_lam2 < 0.003559066:
        z += -1506.644 * Q.lam1_plus_lam2 + 5.362243
    if Q.mass_over_sum_pt < 0.03319429:
        z += 21.14724 * Q.mass_over_sum_pt - 0.7019677
    if Q.sj3_dr_max < 0.1591651:
        z += 10.40974 * Q.sj3_dr_max - 1.656867
    if Q.e2 < 0.01289969:
        z += -93.84471 * Q.e2 + 1.210568
    if Q.sum_pt >= 988.4078:
        z += -0.008526207 * Q.sum_pt + 8.42737
    if Q.sum_z_dr2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -23.73006 * (0.006679471 - Q.sum_z_dr2) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 21187.28 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -11.76039 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 1804.477 * (0.006679471 - Q.sum_z_dr2) * (Q.phi_0 - -0.04013062)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 7463.931 * (0.005019719 - Q.sum_z_dr2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -442.2874 * (0.06108601 - Q.sum_z_dr) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.1219635 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_z_dr < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 102001.2 * (0.06108601 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 1028.916 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -343.1215 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 15714.78 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += 0.003012337 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.2645167 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.00897359 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.3279492 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.sum_z_dr < 0.06108601 and Q.lam1_plus_lam2 > 0.0003193707:
        z += 11555.65 * (0.06108601 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.0003193707)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2 < 0.006679471:
        z += 3698.073 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.sum_z_dr2)
    if Q.tau1 < 0.05356915 and Q.lam1_plus_lam2 < 0.003562611:
        z += 39113.76 * (0.05356915 - Q.tau1) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.log_sum_pt > 6.701242 and Q.sum_z_dr2 < 0.0005611231:
        z += 3539.784 * (Q.log_sum_pt - 6.701242) * (0.0005611231 - Q.sum_z_dr2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2_top5 < 0.005691733:
        z += -2075.013 * (0.1986272 - Q.sj3_dr_max) * (0.005691733 - Q.sum_z_dr2_top5)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0001947983:
        z += 44265.93 * (0.1986272 - Q.sj3_dr_max) * (0.0001947983 - Q.lam2)
    if Q.LHA < 0.1967397 and Q.centroid_offset > 0.01837778:
        z += 10590.82 * (0.1967397 - Q.LHA) * (Q.centroid_offset - 0.01837778)
    if Q.log_sum_pt > 6.701242 and Q.n_dr_0p05_0p1 > 2.0:
        z += -1.282855 * (Q.log_sum_pt - 6.701242) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.sum_z_dr < 0.06108601 and Q.centroid_offset > 0.006789738:
        z += -3143.742 * (0.06108601 - Q.sum_z_dr) * (Q.centroid_offset - 0.006789738)
    if Q.mass < 29.6447 and Q.centroid_offset < 0.02685622:
        z += -6.948775 * (29.6447 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.03319429 and Q.centroid_offset < 0.02685622:
        z += 5750.547 * (0.03319429 - Q.mass_over_sum_pt) * (0.02685622 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1986272 and Q.lam1_plus_lam2 < 0.003562611:
        z += -5648.361 * (0.1986272 - Q.sj3_dr_max) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.e2 < 0.007078158 and Q.sum_z_dr2_top5 > 5.672047e-05:
        z += -318380.3 * (0.007078158 - Q.e2) * (Q.sum_z_dr2_top5 - 5.672047e-05)
    if Q.sum_z_dr < 0.06108601 and Q.sum_z_dr2 < 0.003562611:
        z += -32277.11 * (0.06108601 - Q.sum_z_dr) * (0.003562611 - Q.sum_z_dr2)
    if Q.sj3_dr_max < 0.1426152 and Q.centroid_offset > 0.01627885:
        z += -2099.801 * (0.1426152 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.e2 < 0.01289969 and Q.psi_0p2 > 0.79448:
        z += -687.506 * (0.01289969 - Q.e2) * (Q.psi_0p2 - 0.79448)
    if Q.sum_pt > 988.4078 and Q.dr12 < 0.05062541:
        z += 0.1105229 * (Q.sum_pt - 988.4078) * (0.05062541 - Q.dr12)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.201702
    if Q.sum_z_dr >= 0.0760135:
        z += -57.81582 * Q.sum_z_dr + 4.394783
    if Q.tau1 < 0.04369778:
        z += -80.29137 * Q.tau1 + 3.508555
    if Q.mass < 36.17854:
        z += 0.1553712 * Q.mass - 5.621104
    if Q.e3 >= 0.0001869378:
        z += -3296.076 * Q.e3 + 0.6161613
    if Q.sum_z_dr2 < 0.005019719:
        z += -3330.18 * Q.sum_z_dr2 + 16.71657
    if Q.sj3_dr_max < 0.1059541:
        z += 23.99663 * Q.sj3_dr_max - 2.542542
    if Q.lam2 >= 0.003274457:
        z += 651.7903 * Q.lam2 - 2.134259
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.2857293 * Q.n_dr_0p2_0p4 - 0.2857293
    if Q.lam1 < 0.01167645:
        z += 479.0164 * Q.lam1 - 5.593213
    if Q.centroid_offset < 0.01837778:
        z += -217.8141 * Q.centroid_offset + 4.002939
    if Q.e2 >= 0.04965722:
        z += 132.7135 * Q.e2 - 6.590182
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.007194954 * Q.sj3_pair_mass_max - 0.1743347
    if Q.lam1_plus_lam2 < 0.0001721983:
        z += -7851.195 * Q.lam1_plus_lam2 + 1.351963
    if Q.C3 < 0.0284695:
        z += 43.0578 * Q.C3 - 1.225834
    if Q.zdr_0 < 0.006292091:
        z += 115.7968 * Q.zdr_0 - 0.7286042
    if Q.mass_over_sum_pt < 0.07269073:
        z += 88.72729 * Q.mass_over_sum_pt - 6.449652
    if Q.sum_pt < 988.0299:
        z += -0.006643829 * Q.sum_pt + 6.564302
    z += 3.38996 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 5.749217 * Q.log_sum_pt - 39.64715
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.00798467 * Q.sum_pt_top5 + 6.706761
    if Q.pt_4 < 31.125:
        z += -0.04333109 * Q.pt_4 + 1.34868
    if Q.n_for_90pct < 7.0:
        z += 0.1180452 * Q.n_for_90pct - 0.8263167
    if Q.max_dr < 0.1117619:
        z += -22.69947 * Q.max_dr + 2.536935
    if Q.M3 < 0.0782171:
        z += -3.764874 * Q.M3 + 0.2944775
    if Q.absphi_1 < 0.02227783:
        z += 14.68433 * Q.absphi_1 - 0.327135
    if Q.N2 >= 0.2233283:
        z += 2.585536 * Q.N2 - 0.5774233
    if Q.z_5 < 0.06503035:
        z += -6.689373 * Q.z_5 + 0.4350122
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -254.3557 * Q.mass_over_sum_pt_sq + 1.826995
    if Q.D2 >= 2.055451:
        z += -0.6539716 * Q.D2 + 1.344207
    if Q.sum_zz_dr2 < 0.0030133:
        z += 1171.55 * Q.sum_zz_dr2 - 3.530232
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 5.017199 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.1135589 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 1386.268 * (0.00609665 - Q.sum_z_dr2) * (0.3220738 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -11623.06 * (0.00609665 - Q.sum_z_dr2) * (0.002834884 - Q.mean_phi)
    if Q.mass < 53.33237 and Q.D2 > 2.843757:
        z += 0.01583158 * (53.33237 - Q.mass) * (Q.D2 - 2.843757)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -567.5294 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -488.7564 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.sj3_dr_max < 0.1070199 and Q.abseta_1 > 0.03625488:
        z += -1483.216 * (0.1070199 - Q.sj3_dr_max) * (Q.abseta_1 - 0.03625488)
    if Q.tau1 < 0.04369778 and Q.mean_phi < -0.01275329:
        z += 2138.25 * (0.04369778 - Q.tau1) * (-0.01275329 - Q.mean_phi)
    if Q.centroid_offset < 0.01837778 and Q.tau4 > 0.001224244:
        z += -9725.962 * (0.01837778 - Q.centroid_offset) * (Q.tau4 - 0.001224244)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -32.4318 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -28.34162 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.427333 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -1.526677 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 711.0383 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -2.800213 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.sum_z_dr < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += 2264.076 * (0.05464922 - Q.sum_z_dr) * (0.02656143 - Q.tau21_b2)
    if Q.tau1 < 0.04369778 and Q.tau21_b2 < 0.02656143:
        z += -3919.171 * (0.04369778 - Q.tau1) * (0.02656143 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.07269073 and Q.phi_0 < -0.004917145:
        z += 529.6116 * (0.07269073 - Q.mass_over_sum_pt) * (-0.004917145 - Q.phi_0)
    if Q.centroid_offset < 0.01837778 and Q.mean_eta > 6.288824e-05:
        z += 6260.842 * (0.01837778 - Q.centroid_offset) * (Q.mean_eta - 6.288824e-05)
    if Q.sum_z_dr < 0.05464922 and Q.z_6 > 0.03448406:
        z += -1041.136 * (0.05464922 - Q.sum_z_dr) * (Q.z_6 - 0.03448406)
    if Q.sum_pt < 988.4078 and Q.dr_2 < 0.01778111:
        z += -0.09884834 * (988.4078 - Q.sum_pt) * (0.01778111 - Q.dr_2)
    if Q.e3 > 0.0001869378 and Q.pt_2 > 56.5:
        z += 12.04181 * (Q.e3 - 0.0001869378) * (Q.pt_2 - 56.5)
    if Q.tau1 < 0.04369778 and Q.zdr_2 > 0.00759156:
        z += -25571.21 * (0.04369778 - Q.tau1) * (Q.zdr_2 - 0.00759156)
    if Q.log_sum_pt > 6.896095 and Q.zdr_5 < 0.001255404:
        z += 8746.344 * (Q.log_sum_pt - 6.896095) * (0.001255404 - Q.zdr_5)
    if Q.sd_mass > 45.595 and Q.n_dr_0p05_0p1 > 7.0:
        z += -0.0374889 * (Q.sd_mass - 45.595) * (Q.n_dr_0p05_0p1 - 7.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.232732
    z += 7.094009 * Q.e2
    if Q.lam1 < 0.003377149:
        z += 999.8213 * Q.lam1 - 3.376545
    if Q.lam2 >= 0.0003061234:
        z += 1213.534 * Q.lam2 - 0.3714912
    if Q.pt_7 < 45.75:
        z += 0.005378432 * Q.pt_7 - 0.2460633
    if Q.sj3_dr_min >= 0.1278212:
        z += 6.779858 * Q.sj3_dr_min - 0.8666093
    if Q.mass_top5 < 37.72285:
        z += 0.008142848 * Q.mass_top5 - 0.3071715
    if Q.zdr_0 < 0.009238653:
        z += 40.21386 * Q.zdr_0 - 0.3715219
    if Q.M2 < 0.02563286:
        z += 31.20999 * Q.M2 - 0.8000013
    if Q.sum_pt >= 988.4078:
        z += -0.0136637 * Q.sum_pt + 13.5053
    if Q.LHA >= 0.3033137:
        z += -31.8866 * Q.LHA + 9.671645
    if Q.tau1 >= 0.05356915:
        z += 13.90683 * Q.tau1 - 0.744977
    if Q.sum_z_dr2 >= 0.006688641:
        z += 328.5454 * Q.sum_z_dr2 - 2.197522
    if Q.e3 >= 3.892127e-05:
        z += -4365.355 * Q.e3 + 0.1699052
    if Q.mass < 76.6557:
        z += -0.02552153 * Q.mass + 1.956371
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -386.4408 * Q.sum_z_dr2_top3 + 0.8314536
    if Q.sj2_dr >= 0.3003793:
        z += 1.043502 * Q.sj2_dr - 0.3134465
    if Q.sj3_dr_max >= 0.1986272:
        z += -4.680443 * Q.sj3_dr_max + 0.9296633
    if Q.C2_b2 >= 0.009032972:
        z += 34.66108 * Q.C2_b2 - 0.3130926
    if Q.sj3_pair_mass_min >= 15.95929:
        z += 0.1079386 * Q.sj3_pair_mass_min - 1.722623
    if Q.zdr_7 < 0.00325401:
        z += 109.4468 * Q.zdr_7 - 0.3561412
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.06537829 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.centroid_offset > 0.01258764:
        z += 1101.519 * (0.0211821 - Q.zdr_0) * (Q.centroid_offset - 0.01258764)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.1048824 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -549.4663 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -20.98478 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 143.1024 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -315.2364 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.sj3_pair_mass_min > 11.051 and Q.sj3_pairmin_over_m > 0.28737:
        z += -0.1789294 * (Q.sj3_pair_mass_min - 11.051) * (Q.sj3_pairmin_over_m - 0.28737)
    if Q.tau1 > 0.05356915 and Q.D2 < 1.002471:
        z += -1.542697 * (Q.tau1 - 0.05356915) * (1.002471 - Q.D2)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.766899
    if Q.planar_flow < 0.2534037:
        z += -11.14023 * Q.planar_flow + 2.822976
    if Q.sj2_dr >= 0.1778793:
        z += -0.9293118 * Q.sj2_dr + 0.1653053
    if Q.mass < 11.07122:
        z += -0.1727394 * Q.mass + 1.912436
    if Q.sum_z_dr2 < 0.01323868:
        z += -407.7725 * Q.sum_z_dr2 + 5.398368
    if Q.tau1 < 0.09538712:
        z += 28.61696 * Q.tau1 - 2.729689
    if Q.LHA < 0.3127275:
        z += 4.994255 * Q.LHA - 1.561841
    if Q.centroid_offset >= 0.04990367:
        z += -582.2188 * Q.centroid_offset + 29.05485
    if Q.sj3_dr_max < 0.1059541:
        z += 6.399875 * Q.sj3_dr_max - 0.6780929
    if Q.lam1_plus_lam2 < 0.008566101:
        z += -1924.391 * Q.lam1_plus_lam2 + 16.48453
    if Q.lam1 < 0.008291375:
        z += 312.7289 * Q.lam1 - 2.592953
    if Q.sum_z_dr < 0.06101366:
        z += 33.57753 * Q.sum_z_dr - 2.048688
    if Q.sum_zz_dr2 < 0.008168571:
        z += 3734.991 * Q.sum_zz_dr2 - 30.50954
    if Q.z_7 >= 0.01685855:
        z += 26.06644 * Q.z_7 - 0.4394423
    if Q.z_dr_0p05_0p1 >= 0.8460335:
        z += -1.067331 * Q.z_dr_0p05_0p1 + 0.9029975
    if Q.pt_6 < 19.45312:
        z += 0.1133416 * Q.pt_6 - 2.204847
    if Q.pt_7 >= 29.04219:
        z += -0.02579146 * Q.pt_7 + 0.7490403
    if Q.C2 < 0.03578649:
        z += 39.22194 * Q.C2 - 1.403615
    if Q.max_dr < 0.1762762:
        z += -8.862929 * Q.max_dr + 1.562324
    if Q.e3 < 1.050302e-05:
        z += -102698.6 * Q.e3 + 1.078645
    if Q.sum_pt_top5 < 506.875:
        z += 0.005226925 * Q.sum_pt_top5 - 2.649398
    if Q.sum_pt >= 988.4078:
        z += -0.001218974 * Q.sum_pt + 1.204843
    if Q.mass_over_sum_pt < 0.07637363:
        z += 72.04217 * Q.mass_over_sum_pt - 5.502122
    if Q.lam2 < 0.001130645:
        z += -776.2155 * Q.lam2 + 0.8776239
    if Q.mass_over_sum_pt_sq < 0.00817466:
        z += -2724.934 * Q.mass_over_sum_pt_sq + 22.27541
    if Q.n_dr_0p1_0p2 >= 3.0:
        z += -0.2470647 * Q.n_dr_0p1_0p2 + 0.7411942
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.02658038 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1974994 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -701886.7 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -18662.59 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.09995504 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset < 0.03776099 and Q.mean_phi < -0.001813533:
        z += -1069.251 * (0.03776099 - Q.centroid_offset) * (-0.001813533 - Q.mean_phi)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -3.845445 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -130.2395 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.sj2_dr > 0.1778793 and Q.n_pt_above_50 > 4.0:
        z += 1.561764 * (Q.sj2_dr - 0.1778793) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -317.8035 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 282.3502 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.pt_6 < 41.21875 and Q.D2_b2 < 4.721224:
        z += -0.002940298 * (41.21875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += 1894.065 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    if Q.sj3_dr_max < 0.2623172 and Q.phi_0 < -0.0216713:
        z += 100.2527 * (0.2623172 - Q.sj3_dr_max) * (-0.0216713 - Q.phi_0)
    if Q.centroid_offset < 0.01437952 and Q.sj3_pair_mass_min < 15.95929:
        z += -6.960497 * (0.01437952 - Q.centroid_offset) * (15.95929 - Q.sj3_pair_mass_min)
    if Q.sum_z_dr < 0.02689598 and Q.sj3_pair_mass_min > 1.590484:
        z += -4.021513 * (0.02689598 - Q.sum_z_dr) * (Q.sj3_pair_mass_min - 1.590484)
    if Q.planar_flow < 0.2534037 and Q.sj3_pair_mass_min > 1.282345:
        z += -0.1487548 * (0.2534037 - Q.planar_flow) * (Q.sj3_pair_mass_min - 1.282345)
    if Q.centroid_offset > 0.04990367 and Q.zdr_6 > 0.007390416:
        z += -133007.1 * (Q.centroid_offset - 0.04990367) * (Q.zdr_6 - 0.007390416)
    if Q.sum_pt_top5 < 506.875 and Q.z_dr_0p1_0p2 > 0.04510668:
        z += 0.01159418 * (506.875 - Q.sum_pt_top5) * (Q.z_dr_0p1_0p2 - 0.04510668)
    if Q.centroid_offset > 0.04990367 and Q.tau3 > 0.001996306:
        z += -114514.0 * (Q.centroid_offset - 0.04990367) * (Q.tau3 - 0.001996306)
    if Q.sum_z_dr2 < 0.004372139 and Q.sj3_dr13 > 0.1659434:
        z += 8574.984 * (0.004372139 - Q.sum_z_dr2) * (Q.sj3_dr13 - 0.1659434)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.8202875
    if Q.sum_z_dr2 >= 0.01882765:
        z += 283.276 * Q.sum_z_dr2 - 5.333423
    if Q.mass >= 91.19:
        z += 0.06229581 * Q.mass - 5.680755
    if Q.zdr_0 >= 0.03981924:
        z += -32.9397 * Q.zdr_0 + 1.311634
    if Q.sum_z_dr2_top2 >= 0.01403324:
        z += -25.44365 * Q.sum_z_dr2_top2 + 0.3570569
    if Q.e2 >= 0.06344108:
        z += -40.4626 * Q.e2 + 2.566991
    if Q.centroid_offset >= 0.04990367:
        z += 26.35102 * Q.centroid_offset - 1.315013
    if Q.sum_z_dr2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += 5785.009 * (Q.sum_z_dr2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -1.300649 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.mass > 69.61135 and Q.centroid_offset > 0.009480685:
        z += 0.5777417 * (Q.mass - 69.61135) * (Q.centroid_offset - 0.009480685)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.6904204
    if Q.sum_z_dr < 0.1484084:
        z += -60.62585 * Q.sum_z_dr + 8.997386
    if Q.lam1 < 0.01167645:
        z += -387.469 * Q.lam1 + 4.524264
    if Q.sum_pt_top5 < 839.6875:
        z += 0.000170543 * Q.sum_pt_top5 - 0.1432029
    if Q.e3 < 5.334511e-05:
        z += -20705.38 * Q.e3 + 1.104531
    if Q.pt_6 < 31.90625:
        z += -0.03754183 * Q.pt_6 + 1.197819
    if Q.e2 < 0.08000524:
        z += 41.65023 * Q.e2 - 3.332237
    if Q.mass >= 49.6681:
        z += -0.005694692 * Q.mass + 0.2828445
    if Q.z_5 < 0.02818362:
        z += 222.4598 * Q.z_5 - 6.269723
    if Q.z_7 < 0.02807091:
        z += 68.16563 * Q.z_7 - 1.913472
    if Q.sj3_dr23 >= 0.1974628:
        z += -2.423417 * Q.sj3_dr23 + 0.4785348
    if Q.log_sum_pt >= 6.502799:
        z += -0.7111812 * Q.log_sum_pt + 4.624669
    if Q.pt_7 >= 48.71875:
        z += -0.04484754 * Q.pt_7 + 2.184916
    if Q.lam1_plus_lam2 < 0.006102347:
        z += 134.0377 * Q.lam1_plus_lam2 - 0.8179447
    if Q.z_6 < 0.02160287:
        z += 82.75485 * Q.z_6 - 1.787743
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -73.39266 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.9995866 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -1408151.0 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0005866901 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -4.376142 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.sum_z_dr < 0.1484084 and Q.z_7 > 0.06164517:
        z += 353.3238 * (0.1484084 - Q.sum_z_dr) * (Q.z_7 - 0.06164517)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.000537286:
        z += -11717.43 * (0.1484084 - Q.sum_z_dr) * (0.000537286 - Q.lam2)
    if Q.sum_z_dr < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -1.900037 * (0.1484084 - Q.sum_z_dr) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_z_dr < 0.1484084 and Q.M3 < 0.07474969:
        z += 149.6704 * (0.1484084 - Q.sum_z_dr) * (0.07474969 - Q.M3)
    if Q.e3 < 5.334511e-05 and Q.D3 > 0.2213841:
        z += 2082.581 * (5.334511e-05 - Q.e3) * (Q.D3 - 0.2213841)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 > 0.008780509:
        z += 184.4881 * (0.1484084 - Q.sum_z_dr) * (Q.tau2 - 0.008780509)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -234.362 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.04174263 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.001354124 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0004280637 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = 1.738232
    if Q.planar_flow < 0.1483346:
        z += -9.747521 * Q.planar_flow + 1.445894
    if Q.psi_0p1 >= 0.9461839:
        z += -3.873938 * Q.psi_0p1 + 3.665458
    if Q.sum_z_dr2 >= 0.007520088:
        z += -2433.564 * Q.sum_z_dr2 + 18.30061
    if Q.sum_zz_dr2 >= 0.01122239:
        z += 929.0636 * Q.sum_zz_dr2 - 10.42631
    if Q.lam1_plus_lam2 >= 0.01279734:
        z += -2223.531 * Q.lam1_plus_lam2 + 28.45528
    if Q.e2 < 0.04107712:
        z += -94.63305 * Q.e2 + 3.887253
    if Q.centroid_offset >= 0.04990367:
        z += -1112.97 * Q.centroid_offset + 55.54131
    if Q.mass_over_sum_pt >= 0.08475161:
        z += 289.9424 * Q.mass_over_sum_pt - 24.57308
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.159466 * Q.n_dr_0_0p05 + 0.7973299
    if Q.e3 < 7.988762e-05:
        z += 17356.69 * Q.e3 - 1.386584
    if Q.sd_mass < 44.81215:
        z += -0.03860237 * Q.sd_mass + 1.729855
    if Q.sum_z_dr < 0.08723651:
        z += 125.947 * Q.sum_z_dr - 10.98717
    if Q.mass >= 69.61135:
        z += -0.02410959 * Q.mass + 1.678301
    if Q.mass_top5 >= 49.18618:
        z += 0.005033683 * Q.mass_top5 - 0.2475876
    if Q.sj2_dr >= 0.1591713:
        z += 24.70843 * Q.sj2_dr - 3.932872
    if Q.lam2 < 0.000537286:
        z += -1003.991 * Q.lam2 + 0.5394303
    if Q.C2_b2 < 0.004032342:
        z += 108.5074 * Q.C2_b2 - 0.4375389
    if Q.LHA >= 0.3033137:
        z += 9.55206 * Q.LHA - 2.897271
    if Q.z_dr_0p05_0p1 < 0.163898:
        z += 2.637256 * Q.z_dr_0p05_0p1 - 0.4322411
    if Q.sum_pt < 715.4688:
        z += 0.00437971 * Q.sum_pt - 3.133545
    if Q.z_dr_0_0p05 >= 0.1515405:
        z += 1.686896 * Q.z_dr_0_0p05 - 0.2556329
    if Q.z_dr_0p1_0p2 < 0.07870506:
        z += 0.8505449 * Q.z_dr_0p1_0p2 - 0.06694219
    if Q.sd_rg < 0.1441415:
        z += 10.93556 * Q.sd_rg - 1.576269
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -4326.342 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 914.585 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.01713047 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -406.3457 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 1.367021 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 260876.9 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.2001708 and Q.z_5 < 0.1058993:
        z += -230.8212 * (Q.sj2_dr - 0.2001708) * (0.1058993 - Q.z_5)
    if Q.sj2_dr > 0.2001708 and Q.D2_b2 < 0.08499387:
        z += -35.26265 * (Q.sj2_dr - 0.2001708) * (0.08499387 - Q.D2_b2)
    if Q.planar_flow < 0.1115136 and Q.mean_eta > -0.006779839:
        z += -70.23845 * (0.1115136 - Q.planar_flow) * (Q.mean_eta - -0.006779839)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.sum_pt < 715.4688:
        z += -0.0006985975 * (0.5882598 - Q.z_dr_0p05_0p1) * (715.4688 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.C2_b2 < 0.0008333816:
        z += 1261426.0 * (Q.centroid_offset - 0.04990367) * (0.0008333816 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.C2_b2 < 0.02415398:
        z += -99.90701 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.02415398 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.zdr_5 > 0.0155217:
        z += -2473.843 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.zdr_5 - 0.0155217)
    if Q.sj2_dr > 0.1591713 and Q.dr_2 < 0.03293672:
        z += -489.071 * (Q.sj2_dr - 0.1591713) * (0.03293672 - Q.dr_2)
    if Q.centroid_offset > 0.02685622 and Q.pt_1 < 150.625:
        z += -0.24454 * (Q.centroid_offset - 0.02685622) * (150.625 - Q.pt_1)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.2467468
    if Q.N2 < 0.2233283:
        z += 2.239701 * Q.N2 - 0.5001885
    if Q.sum_z_dr2 < 0.007520088:
        z += 720.5105 * Q.sum_z_dr2 - 5.418303
    if Q.mass_over_sum_pt < 0.1309286:
        z += 36.07216 * Q.mass_over_sum_pt - 4.722879
    if Q.lam1_plus_lam2 < 0.01323868:
        z += -995.072 * Q.lam1_plus_lam2 + 13.17343
    if Q.e2 < 0.04110972:
        z += -230.8703 * Q.e2 + 9.491014
    if Q.sum_zz_dr2 < 0.008168571:
        z += 1270.728 * Q.sum_zz_dr2 - 10.38003
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -592.231 * Q.mass_over_sum_pt_sq + 4.253898
    if Q.lam1 < 0.00483998:
        z += -169.8143 * Q.lam1 + 0.8218979
    if Q.lam2 < 0.001130645:
        z += 154.8605 * Q.lam2 - 0.1750922
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 203.2912 * Q.sum_z_dr2_top3 - 0.4373948
    if Q.sum_z_dr < 0.1019409:
        z += 93.7718 * Q.sum_z_dr - 9.559185
    if Q.sj2_dr < 0.09317241:
        z += -16.29248 * Q.sj2_dr + 1.51801
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += 22821.72 * Q.sum_z_dr2_top2 - 11.69507
    if Q.sj3_pair_mass_max >= 80.4:
        z += -0.04689526 * Q.sj3_pair_mass_max + 3.770379
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -10.62676 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.max_dr < 0.121681:
        z += -190.5512 * (0.2233283 - Q.N2) * (0.121681 - Q.max_dr)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.02445333 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.sum_z_dr2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -1839.117 * (0.007520088 - Q.sum_z_dr2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 34.37521 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.z_dr_0p1_0p2 > 0.4684459:
        z += 121.8442 * (0.1309286 - Q.mass_over_sum_pt) * (Q.z_dr_0p1_0p2 - 0.4684459)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 < 4.0:
        z += 1.380195 * (0.2233283 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.2233283 and Q.pt_entropy < 1.54073:
        z += -15.31542 * (0.2233283 - Q.N2) * (1.54073 - Q.pt_entropy)
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
