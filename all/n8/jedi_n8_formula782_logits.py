"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; all observables), as if-statements, with each class score (logit) written out as a formula.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.7% (the network: 65.8%); same class as the network for 90.3% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
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
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -2.269336
    if Q.planar_flow < 0.1484197:
        z += -15.48922 * Q.planar_flow + 2.298905
    if Q.width < 0.004372139:
        z += 698.3644 * Q.width - 0.8918249
    if 0.004372139 <= Q.width < 0.008678045:
        z += -501.9901 * Q.width + 4.356292
    if Q.girth2 < 0.01323868:
        z += -256.6293 * Q.girth2 + 3.203868
    if 0.01323868 <= Q.girth2 < 0.01882765:
        z += 34.63325 * Q.girth2 - 0.6520628
    if Q.mass < 29.6447:
        z += 0.1451976 * Q.mass - 5.644388
    if 29.6447 <= Q.mass < 56.92035:
        z += 0.03876052 * Q.mass - 2.489093
    if 56.92035 <= Q.mass < 64.61873:
        z += 0.036739 * Q.mass - 2.374028
    if Q.girth2_top3 < 0.006756161:
        z += -59.18719 * Q.girth2_top3 + 0.05556218
    if 0.006756161 <= Q.girth2_top3 < 0.007929074:
        z += 293.5564 * Q.girth2_top3 - 2.32763
    if Q.lam1 < 0.0002758826:
        z += -2403.526 * Q.lam1 + 0.07274109
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 157.7858 * Q.lam1 - 0.6338803
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += -208.1847 * Q.lam1 + 1.35457
    if Q.sum_pt >= 901.5938:
        z += -0.01615072 * Q.sum_pt + 14.56139
    if Q.C2_b2 < 0.001563465:
        z += 705.0084 * Q.C2_b2 - 1.102256
    if Q.sj3_dr_max < 0.04889979:
        z += 52.00124 * Q.sj3_dr_max + 0.1570275
    if 0.04889979 <= Q.sj3_dr_max < 0.1070199:
        z += -10.70098 * Q.sj3_dr_max + 3.223153
    if 0.1070199 <= Q.sj3_dr_max < 0.1789613:
        z += 9.801799 * Q.sj3_dr_max + 1.028946
    if 0.1789613 <= Q.sj3_dr_max < 0.233678:
        z += -8.781373 * Q.sj3_dr_max + 4.354616
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -23.55468 * Q.sj3_dr_max + 7.806813
    if Q.sj3_dr_max >= 0.3012016:
        z += -12.8537 * Q.sj3_dr_max + 4.583661
    if 0.006789738 <= Q.centroid_offset < 0.04990367:
        z += -79.15749 * Q.centroid_offset + 0.5374586
    if Q.centroid_offset >= 0.04990367:
        z += -494.564 * Q.centroid_offset + 21.26777
    if Q.girth < 0.07608178:
        z += 53.22454 * Q.girth - 4.186574
    if 0.07608178 <= Q.girth < 0.08723651:
        z += 12.29579 * Q.girth - 1.072642
    if Q.e2 < 0.0245477:
        z += -153.4974 * Q.e2 + 4.718383
    if 0.0245477 <= Q.e2 < 0.03556091:
        z += -86.29402 * Q.e2 + 3.068694
    if Q.log_sum_pt < 6.080494:
        z += 6.069335 * Q.log_sum_pt - 36.90456
    if 6.377723 <= Q.log_sum_pt < 6.670067:
        z += 1.223961 * Q.log_sum_pt - 7.806085
    if Q.log_sum_pt >= 6.670067:
        z += 0.971813 * Q.log_sum_pt - 6.12424
    if Q.mass_over_sum_pt_sq < 0.003904593:
        z += -225.2411 * Q.mass_over_sum_pt_sq + 0.8794751
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.001143026 * Q.sum_pt_top5 - 0.785759
    if Q.lam2 < 7.300726e-05:
        z += 13821.91 * Q.lam2 - 1.0091
    if Q.mass_over_sum_pt < 0.08475161:
        z += 46.56798 * Q.mass_over_sum_pt - 3.570384
    if 0.08475161 <= Q.mass_over_sum_pt < 0.1309286:
        z += -8.149673 * Q.mass_over_sum_pt + 1.067026
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += -141.4717 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -1589.287 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.D2 < 1.002471:
        z += 270.1994 * (0.01323868 - Q.girth2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.03776972 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.girth2 < 0.01323868 and Q.phi_0 > -0.008995056:
        z += 791.6353 * (0.01323868 - Q.girth2) * (Q.phi_0 - -0.008995056)
    if Q.planar_flow < 0.1484197 and Q.z_7 < 0.02320757:
        z += 535.8036 * (0.1484197 - Q.planar_flow) * (0.02320757 - Q.z_7)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 113.1808 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.mass < 64.61873 and Q.D2_b2 < 0.1830092:
        z += -0.1144242 * (64.61873 - Q.mass) * (0.1830092 - Q.D2_b2)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 98643.09 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    if Q.sj3_dr_max < 0.3012016 and Q.D2_b2 < 0.05744392:
        z += -77.02873 * (0.3012016 - Q.sj3_dr_max) * (0.05744392 - Q.D2_b2)
    if Q.planar_flow < 0.1484197 and Q.dr0_6 > 0.1755206:
        z += -18.55101 * (0.1484197 - Q.planar_flow) * (Q.dr0_6 - 0.1755206)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_4 < 0.06336451:
        z += -0.08391744 * (Q.sum_pt_top5 - 687.4375) * (0.06336451 - Q.dr_4)
    if Q.lam2 < 7.300726e-05 and Q.dr_2 < 0.08525808:
        z += 107535.7 * (7.300726e-05 - Q.lam2) * (0.08525808 - Q.dr_2)
    if Q.sum_pt > 901.5938 and Q.pt_7 > 33.21875:
        z += 0.0004998458 * (Q.sum_pt - 901.5938) * (Q.pt_7 - 33.21875)
    if Q.log_sum_pt > 6.670067 and Q.z_7 > 0.01685855:
        z += -527.3621 * (Q.log_sum_pt - 6.670067) * (Q.z_7 - 0.01685855)
    if Q.sj3_dr_max < 0.3012016 and Q.z_7 < 0.05557716:
        z += -169.4092 * (0.3012016 - Q.sj3_dr_max) * (0.05557716 - Q.z_7)
    if Q.log_sum_pt > 6.670067 and Q.m01 < 28.78966:
        z += 0.08522835 * (Q.log_sum_pt - 6.670067) * (28.78966 - Q.m01)
    if Q.planar_flow < 0.1484197 and Q.dr_2 < 0.02270492:
        z += -627.5257 * (0.1484197 - Q.planar_flow) * (0.02270492 - Q.dr_2)
    if Q.girth2_top3 < 0.007929074 and Q.D2_b2 < 0.5327104:
        z += -129.2294 * (0.007929074 - Q.girth2_top3) * (0.5327104 - Q.D2_b2)
    if Q.centroid_offset > 0.02076709 and Q.z_7 < 0.06473447:
        z += -1315.439 * (Q.centroid_offset - 0.02076709) * (0.06473447 - Q.z_7)
    if Q.planar_flow < 0.1484197 and Q.pt_dispersion < 0.4160096:
        z += 66.73296 * (0.1484197 - Q.planar_flow) * (0.4160096 - Q.pt_dispersion)
    if Q.centroid_offset > 0.006789738 and Q.n_pt_above_50 < 7.0:
        z += 4.504132 * (Q.centroid_offset - 0.006789738) * (7.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.02076709 and Q.dr_2 < 0.05056028:
        z += -1865.497 * (Q.centroid_offset - 0.02076709) * (0.05056028 - Q.dr_2)
    if Q.width < 0.004372139 and Q.C3 < 0.03532852:
        z += 8963.268 * (0.004372139 - Q.width) * (0.03532852 - Q.C3)
    if Q.centroid_offset > 0.006789738 and Q.pt_1 < 223.375:
        z += 0.1580019 * (Q.centroid_offset - 0.006789738) * (223.375 - Q.pt_1)
    return max(0.0, z)


def neuron_1(Q):
    z = -0.1500556
    if Q.lam1 < 0.0008722282:
        z += 793.089 * Q.lam1 + 1.388221
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -359.6445 * Q.lam1 + 2.393668
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -104.1909 * Q.lam1 + 0.8726587
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 3.887184 * Q.log_sum_pt - 24.79138
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 13.39246 * Q.log_sum_pt - 86.60232
    if Q.log_sum_pt >= 6.605974:
        z += 19.57982 * Q.log_sum_pt - 127.4759
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -683.8452 * Q.mass_over_sum_pt_sq + 3.988822
    if Q.z_7 < 0.02320757:
        z += 34.52295 * Q.z_7 - 2.128173
    if 0.02320757 <= Q.z_7 < 0.06164517:
        z += 50.31695 * Q.z_7 - 2.494714
    if Q.z_7 >= 0.06164517:
        z += 15.794 * Q.z_7 - 0.3665404
    if Q.girth2 < 0.008678045:
        z += 465.6551 * Q.girth2 - 4.040976
    if Q.mass < 49.6681:
        z += -0.01806086 * Q.mass + 0.8970484
    if Q.sj3_dr_max >= 0.169029:
        z += 2.602426 * Q.sj3_dr_max - 0.4398856
    if Q.e2_sq < 0.008168571:
        z += -102.4513 * Q.e2_sq + 0.8368808
    if Q.pt_7 < 25.57812:
        z += 0.0674416 * Q.pt_7 - 1.72503
    if Q.pt_7 >= 53.4375:
        z += -0.113761 * Q.pt_7 + 6.079102
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.003377352 * Q.sum_pt_top5 - 1.241493
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.01061212 * Q.sum_pt_top5 + 6.189541
    if Q.width < 0.00609665:
        z += 507.9495 * Q.width - 3.09679
    if Q.girth < 0.0717028:
        z += 20.15249 * Q.girth - 2.135073
    if 0.0717028 <= Q.girth < 0.1019409:
        z += 22.82163 * Q.girth - 2.326458
    if Q.tau1 < 0.07283629:
        z += 12.89297 * Q.tau1 - 0.9390759
    if Q.e3 < 0.0005116989:
        z += -1521.101 * Q.e3 + 0.7783456
    if Q.zdr_0 < 0.0211821:
        z += 15.95468 * Q.zdr_0 - 0.3379537
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += -16623.31 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += 1.01486 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.tau2 < 0.06297984:
        z += 180.3467 * (0.06164517 - Q.z_7) * (0.06297984 - Q.tau2)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -20.47006 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.3523999 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -3.167588 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.z_7 < 0.06164517 and Q.z_dr_0p05_0p1 > 0.04875823:
        z += -33.41669 * (0.06164517 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.04875823)
    if Q.pt_7 > 34.53125 and Q.n_dr_0p1_0p2 > 1.0:
        z += 0.007277961 * (Q.pt_7 - 34.53125) * (Q.n_dr_0p1_0p2 - 1.0)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.003642603 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 55.95848 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 2.664448 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -480.006 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.06999584 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.D2_b2 < 0.03885671:
        z += -26685.97 * (0.005832932 - Q.mass_over_sum_pt_sq) * (0.03885671 - Q.D2_b2)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -26.67187 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.girth2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 23611.4 * (0.008678045 - Q.girth2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -5749.598 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 2067.335 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    if Q.log_sum_pt > 6.502799 and Q.zdr_6 < 0.008654951:
        z += -498.3831 * (Q.log_sum_pt - 6.502799) * (0.008654951 - Q.zdr_6)
    if Q.girth < 0.1019409 and Q.pair_mass_0_6 > 4.037975:
        z += -0.3634692 * (0.1019409 - Q.girth) * (Q.pair_mass_0_6 - 4.037975)
    if Q.sum_pt_top5 > 531.1875 and Q.zdr_6 < 0.01392641:
        z += 0.3407817 * (Q.sum_pt_top5 - 531.1875) * (0.01392641 - Q.zdr_6)
    if Q.z_7 < 0.06164517 and Q.abseta_0 < 0.1057739:
        z += 90.81855 * (0.06164517 - Q.z_7) * (0.1057739 - Q.abseta_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.437633
    if Q.sj3_pair_mass_max < 62.55:
        z += -0.02046268 * Q.sj3_pair_mass_max + 1.279941
    if Q.log_sum_pt < 6.46415:
        z += -4.002287 * Q.log_sum_pt + 26.13068
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -1.828323 * Q.log_sum_pt + 12.07786
    if Q.log_sum_pt >= 6.842717:
        z += -6.174875 * Q.log_sum_pt + 42.25292
    if Q.girth < 0.007673833:
        z += 200.7826 * Q.girth - 1.540772
    if Q.pt_7 < 34.53125:
        z += 0.06340656 * Q.pt_7 - 3.388288
    if 34.53125 <= Q.pt_7 < 43.5:
        z += 0.0872474 * Q.pt_7 - 4.211542
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.1536877 * Q.pt_7 - 7.101695
    if Q.pt_7 >= 53.4375:
        z += 0.09028113 * Q.pt_7 - 3.713407
    if Q.LHA >= 0.1329373:
        z += -8.633589 * Q.LHA + 1.147726
    if 0.03629544 <= Q.z_7 < 0.04939969:
        z += -29.3835 * Q.z_7 + 1.066487
    if Q.z_7 >= 0.04939969:
        z += -39.42003 * Q.z_7 + 1.562289
    if Q.planar_flow < 0.4926918:
        z += 0.7558091 * Q.planar_flow - 0.3723809
    if Q.mass < 36.22941:
        z += 0.03617439 * Q.mass - 1.310577
    if Q.max_dr < 0.2507612:
        z += 2.422453 * Q.max_dr - 0.6074572
    if Q.girth2 < 0.003562611:
        z += -80.46552 * Q.girth2 + 0.2866673
    if Q.m012 >= 40.2:
        z += 0.02284849 * Q.m012 - 0.9185095
    if Q.sum_pt < 527.1781:
        z += -0.007063142 * Q.sum_pt + 5.74527
    if 527.1781 <= Q.sum_pt < 813.4156:
        z += -0.005366317 * Q.sum_pt + 4.850741
    if 813.4156 <= Q.sum_pt < 868.5094:
        z += 0.001696825 * Q.sum_pt - 0.8945293
    if Q.sum_pt >= 868.5094:
        z += 0.009113927 * Q.sum_pt - 7.336352
    if Q.sum_pt_top5 >= 752.1:
        z += -0.005520247 * Q.sum_pt_top5 + 4.151778
    if Q.zdr_0 < 0.0211821:
        z += -21.75995 * Q.zdr_0 + 0.4609215
    if Q.mass_over_sum_pt < 0.1079857:
        z += -18.67865 * Q.mass_over_sum_pt + 2.017026
    if Q.lam2 < 0.0001947983:
        z += -1985.14 * Q.lam2 + 0.3867018
    if Q.lam1 < 0.001503553:
        z += -198.1116 * Q.lam1 + 0.2978714
    if Q.sj3_pair_mass_max < 62.55 and Q.z_7 < 0.06810151:
        z += -0.245409 * (62.55 - Q.sj3_pair_mass_max) * (0.06810151 - Q.z_7)
    if Q.sj3_pair_mass_max < 62.55 and Q.centroid_offset > 0.01096064:
        z += -0.6714346 * (62.55 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.08050702:
        z += -2031.748 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.08050702)
    if Q.z_7 > 0.04939969 and Q.sj3_dr_min < 0.0623951:
        z += -283.7425 * (Q.z_7 - 0.04939969) * (0.0623951 - Q.sj3_dr_min)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += -0.1636037 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.z_7 < 0.03243272 and Q.mass_top3 > 16.89912:
        z += -2.173151 * (0.03243272 - Q.z_7) * (Q.mass_top3 - 16.89912)
    if Q.lam1 < 0.00595415 and Q.pt_6 > 19.46875:
        z += 2.219864 * (0.00595415 - Q.lam1) * (Q.pt_6 - 19.46875)
    if Q.girth < 0.007673833 and Q.pt_4 < 75.625:
        z += 4.41883 * (0.007673833 - Q.girth) * (75.625 - Q.pt_4)
    if Q.mass < 36.22941 and Q.pt_5 < 73.75:
        z += 0.0002222242 * (36.22941 - Q.mass) * (73.75 - Q.pt_5)
    if Q.log_sum_pt < 6.46415 and Q.D2_b2 < 1.129616:
        z += 0.9879352 * (6.46415 - Q.log_sum_pt) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.z_dr_0p05_0p1 < 0.4474937:
        z += 7.143003 * (Q.log_sum_pt - 6.842717) * (0.4474937 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top5 > 752.1 and Q.D2_b2 < 1.129616:
        z += 0.01115289 * (Q.sum_pt_top5 - 752.1) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.D2_b2 < 1.345805:
        z += -10.83418 * (Q.log_sum_pt - 6.842717) * (1.345805 - Q.D2_b2)
    if Q.sum_pt_top5 > 752.1 and Q.abseta_0 < 0.0174408:
        z += -0.3398585 * (Q.sum_pt_top5 - 752.1) * (0.0174408 - Q.abseta_0)
    if Q.log_sum_pt > 6.842717 and Q.abseta_0 < 0.0174408:
        z += 383.8185 * (Q.log_sum_pt - 6.842717) * (0.0174408 - Q.abseta_0)
    if Q.sum_pt > 527.1781 and Q.lam2 < 0.0001947983:
        z += -16.29214 * (Q.sum_pt - 527.1781) * (0.0001947983 - Q.lam2)
    if Q.log_sum_pt > 6.842717 and Q.dr02 > 0.2623104:
        z += -549.2448 * (Q.log_sum_pt - 6.842717) * (Q.dr02 - 0.2623104)
    if Q.log_sum_pt > 6.842717 and Q.dr02 > 0.15647:
        z += 122.3369 * (Q.log_sum_pt - 6.842717) * (Q.dr02 - 0.15647)
    if Q.log_sum_pt > 6.842717 and Q.lam2 < 0.0001947983:
        z += 27617.11 * (Q.log_sum_pt - 6.842717) * (0.0001947983 - Q.lam2)
    return max(0.0, z)


def neuron_3(Q):
    z = -2.890099
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 59.38677 * Q.mass_over_sum_pt - 4.046561
    if Q.centroid_offset >= 0.01096064:
        z += 31.74809 * Q.centroid_offset - 0.3479794
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -56.39337 * Q.tau1 + 3.020945
    if 0.1027642 <= Q.tau1 < 0.1136369:
        z += -136.4117 * Q.tau1 + 11.24397
    if Q.tau1 >= 0.1136369:
        z += -8.681168 * Q.tau1 - 3.270934
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 381.3318 * Q.lam1 - 3.193872
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += 631.1092 * Q.lam1 - 6.192131
    if Q.lam1 >= 0.01643375:
        z += 755.1569 * Q.lam1 - 8.230701
    if 0.04081947 <= Q.girth < 0.07608178:
        z += 125.295 * Q.girth - 5.114476
    if 0.07608178 <= Q.girth < 0.08723651:
        z += 201.1992 * Q.girth - 10.8894
    if 0.08723651 <= Q.girth < 0.1019409:
        z += 221.2916 * Q.girth - 12.64219
    if Q.girth >= 0.1019409:
        z += -75.24731 * Q.girth + 17.58726
    if 0.005590289 <= Q.width < 0.007520088:
        z += 24.16376 * Q.width - 0.1350824
    if 0.007520088 <= Q.width < 0.01323868:
        z += 103.0607 * Q.width - 0.7283943
    if Q.width >= 0.01323868:
        z += 381.0378 * Q.width - 4.408443
    if 0.04447357 <= Q.e2 < 0.06344108:
        z += 149.2331 * Q.e2 - 6.63693
    if Q.e2 >= 0.06344108:
        z += 173.3047 * Q.e2 - 8.164057
    if Q.girth2 < 0.004372139:
        z += 432.9008 * Q.girth2 - 1.892703
    if 0.008678045 <= Q.girth2 < 0.01882765:
        z += -1815.727 * Q.girth2 + 15.75696
    if Q.girth2 >= 0.01882765:
        z += -1409.028 * Q.girth2 + 8.099779
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += 41.99947 * Q.sj2_dr - 7.864894
    if Q.sj2_dr >= 0.2687922:
        z += -17.91956 * Q.sj2_dr + 8.240873
    if Q.mean_eta < -0.004664942:
        z += -16.63338 * Q.mean_eta - 0.07759373
    if Q.girth2_top5 < 0.007164202:
        z += -41.06764 * Q.girth2_top5 + 0.4703627
    if 0.007164202 <= Q.girth2_top5 < 0.008329695:
        z += -151.1342 * Q.girth2_top5 + 1.258902
    if Q.mass >= 64.61873:
        z += -0.089206 * Q.mass + 5.764379
    if 38.43971 <= Q.sd_mass < 62.55:
        z += -0.005989295 * Q.sd_mass + 0.2302268
    if 62.55 <= Q.sd_mass < 86.4:
        z += 0.06358293 * Q.sd_mass - 4.121516
    if Q.sd_mass >= 86.4:
        z += -0.01767888 * Q.sd_mass + 2.899504
    if Q.sj3_pair_mass_min >= 11.051:
        z += 0.02842561 * Q.sj3_pair_mass_min - 0.3141314
    if 0.1986272 <= Q.sj3_dr_max < 0.3012016:
        z += 37.66752 * Q.sj3_dr_max - 7.481794
    if Q.sj3_dr_max >= 0.3012016:
        z += 19.45099 * Q.sj3_dr_max - 1.994946
    if Q.n_dr_0_0p05 < 1.0:
        z += -0.6363606 * Q.n_dr_0_0p05 + 0.6363606
    if Q.z_dr_0_0p05 < 0.9008535:
        z += 2.124904 * Q.z_dr_0_0p05 - 1.914227
    if Q.mass_over_sum_pt_sq < 0.00817466:
        z += 68.3725 * Q.mass_over_sum_pt_sq - 2.695337
    if 0.00817466 <= Q.mass_over_sum_pt_sq < 0.0116609:
        z += 612.8129 * Q.mass_over_sum_pt_sq - 7.145952
    if Q.LHA >= 0.3467135:
        z += 39.6911 * Q.LHA - 13.76144
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += -1.063356 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += -0.4597182 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.girth > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 91.83663 * (Q.girth - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 1.796479 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.2915604 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -517.9197 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.sj2_dr > 0.1872617 and Q.pt_6 < 27.57812:
        z += -0.5168159 * (Q.sj2_dr - 0.1872617) * (27.57812 - Q.pt_6)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 383.4973 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 21.82449 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.55 and Q.D2_b2 < 0.9206502:
        z += -0.02324077 * (Q.sd_mass - 62.55) * (0.9206502 - Q.D2_b2)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -172.9897 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.girth > 0.07608178 and Q.eta_0 > 0.07952881:
        z += 144.9962 * (Q.girth - 0.07608178) * (Q.eta_0 - 0.07952881)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05441452:
        z += 13239.58 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05441452)
    if Q.max_dr > 0.1027585 and Q.z_6 < 0.06081235:
        z += 117.0552 * (Q.max_dr - 0.1027585) * (0.06081235 - Q.z_6)
    if Q.max_dr > 0.1027585 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -13.27766 * (Q.max_dr - 0.1027585) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.max_dr > 0.1027585 and Q.eta_1 < -0.0892334:
        z += -43.8559 * (Q.max_dr - 0.1027585) * (-0.0892334 - Q.eta_1)
    if Q.max_dr > 0.1027585 and Q.abseta_0 > 0.1057739:
        z += -70.00298 * (Q.max_dr - 0.1027585) * (Q.abseta_0 - 0.1057739)
    if Q.sj2_dr > 0.2687922 and Q.z_dr_0p05_0p1 < 0.9641201:
        z += 66.1847 * (Q.sj2_dr - 0.2687922) * (0.9641201 - Q.z_dr_0p05_0p1)
    if Q.sj2_dr > 0.1872617 and Q.z_dr_0p05_0p1 < 0.9641201:
        z += -25.96528 * (Q.sj2_dr - 0.1872617) * (0.9641201 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0_0p05 < 1.0 and Q.n_dr_0p05_0p1 < 4.0:
        z += 0.2503092 * (1.0 - Q.n_dr_0_0p05) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.girth2 > 0.01882765 and Q.z_dr_0p05_0p1 > 0.7509095:
        z += 16141.75 * (Q.girth2 - 0.01882765) * (Q.z_dr_0p05_0p1 - 0.7509095)
    if Q.girth > 0.08723651 and Q.z_dr_0p05_0p1 > 0.6747704:
        z += -350.2965 * (Q.girth - 0.08723651) * (Q.z_dr_0p05_0p1 - 0.6747704)
    if Q.centroid_offset > 0.01096064 and Q.abseta_7 < 0.1626038:
        z += -100.4443 * (Q.centroid_offset - 0.01096064) * (0.1626038 - Q.abseta_7)
    return max(0.0, z)


def neuron_4(Q):
    z = -1.36422
    if Q.N2 < 0.2233283:
        z += -48.36013 * Q.N2 + 10.80018
    if Q.lam2 < 0.000537286:
        z += 4048.623 * Q.lam2 - 3.024358
    if 0.000537286 <= Q.lam2 < 0.001130645:
        z += 1430.989 * Q.lam2 - 1.61794
    if Q.mass_over_sum_pt < 0.07269073:
        z += 20.23486 * Q.mass_over_sum_pt - 1.470887
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -95.8291 * Q.mass_over_sum_pt + 8.664276
    if Q.width < 0.001653836:
        z += -2151.704 * Q.width + 3.558567
    if Q.width >= 0.003562611:
        z += 464.7411 * Q.width - 1.655692
    if Q.e2 < 0.04447357:
        z += 69.41232 * Q.e2 - 3.490373
    if 0.04447357 <= Q.e2 < 0.05028464:
        z += 82.76979 * Q.e2 - 4.084428
    if Q.e2 >= 0.05028464:
        z += 13.35747 * Q.e2 - 0.5940545
    if Q.sum_pt < 739.5:
        z += 0.008282799 * Q.sum_pt - 6.12513
    if Q.max_dr < 0.1117619:
        z += 8.810592 * Q.max_dr - 1.751787
    if 0.1117619 <= Q.max_dr < 0.1598486:
        z += 11.70372 * Q.max_dr - 2.075128
    if 0.1598486 <= Q.max_dr < 0.177305:
        z += 26.71638 * Q.max_dr - 4.474881
    if Q.max_dr >= 0.177305:
        z += 15.01265 * Q.max_dr - 2.399752
    if Q.C2 < 0.06729223:
        z += -94.4345 * Q.C2 + 6.354708
    if 44.82259 <= Q.sd_mass < 74.57663:
        z += 0.0157155 * Q.sd_mass - 0.7044095
    if Q.sd_mass >= 74.57663:
        z += -0.06604402 * Q.sd_mass + 5.39294
    if Q.e3 < 1.340118e-05:
        z += 3213.871 * Q.e3 + 0.4901311
    if 1.340118e-05 <= Q.e3 < 0.0001869378:
        z += -3072.554 * Q.e3 + 0.5743767
    if Q.sj3_dr_max < 0.213399:
        z += 17.87146 * Q.sj3_dr_max - 3.252767
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 26.46111 * Q.sj3_dr_max - 5.085789
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -9.802721 * Q.sj3_dr_max + 3.38827
    if Q.girth2 < 0.002635418:
        z += 601.8658 * Q.girth2 - 6.759471
    if 0.002635418 <= Q.girth2 < 0.008678045:
        z += 856.1347 * Q.girth2 - 7.429575
    if Q.girth2_top5 < 0.001501708:
        z += -221.939 * Q.girth2_top5 + 1.590016
    if 0.001501708 <= Q.girth2_top5 < 0.007164202:
        z += -319.502 * Q.girth2_top5 + 1.736527
    if Q.girth2_top5 >= 0.007164202:
        z += -97.56298 * Q.girth2_top5 + 0.1465112
    if Q.centroid_offset < 0.04990367:
        z += 39.94441 * Q.centroid_offset - 1.993373
    if Q.girth2_top2 < 0.007639643:
        z += -197.5056 * Q.girth2_top2 + 1.508872
    if 45.7571 <= Q.mass < 76.6557:
        z += -0.09183872 * Q.mass + 4.202273
    if Q.mass >= 76.6557:
        z += -0.1076689 * Q.mass + 5.415748
    if Q.e2_sq < 0.01165737:
        z += -3217.27 * Q.e2_sq + 37.50492
    if Q.tau1 < 0.07283629:
        z += -35.46632 * Q.tau1 + 2.583235
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 2529.083 * Q.mass_over_sum_pt_sq - 29.49139
    if Q.LHA < 0.3467135:
        z += -8.402896 * Q.LHA + 2.913397
    if Q.lam1 < 0.001503553:
        z += 1189.982 * Q.lam1 - 1.789201
    if Q.mass_top5 >= 14.54404:
        z += 0.03104866 * Q.mass_top5 - 0.4515729
    if Q.M2 < 0.03874536:
        z += 57.40931 * Q.M2 - 2.224344
    if Q.N2 < 0.2233283 and Q.mass < 62.55:
        z += -0.7048962 * (0.2233283 - Q.N2) * (62.55 - Q.mass)
    if Q.N2 < 0.2233283 and Q.e2_sq > 0.01165737:
        z += -1374.393 * (0.2233283 - Q.N2) * (Q.e2_sq - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.4083367 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.mean_phi < -0.009352575:
        z += 188.9144 * (0.2233283 - Q.N2) * (-0.009352575 - Q.mean_phi)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -69.92348 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1218872:
        z += -37.27454 * (0.2233283 - Q.N2) * (0.1218872 - Q.abseta_7)
    if Q.sd_mass > 44.82259 and Q.centroid_offset > 0.001308549:
        z += -1.160978 * (Q.sd_mass - 44.82259) * (Q.centroid_offset - 0.001308549)
    if Q.sd_mass > 44.82259 and Q.sd_zg < 0.275762:
        z += -0.3597226 * (Q.sd_mass - 44.82259) * (0.275762 - Q.sd_zg)
    if Q.width > 0.003562611 and Q.sj3_z3 < 0.1057566:
        z += -1114.583 * (Q.width - 0.003562611) * (0.1057566 - Q.sj3_z3)
    if Q.lam2 < 0.000537286 and Q.dr01 < 0.1410336:
        z += -16541.84 * (0.000537286 - Q.lam2) * (0.1410336 - Q.dr01)
    if Q.max_dr > 0.1598486 and Q.C2_b2 < 0.0006435798:
        z += -45769.68 * (Q.max_dr - 0.1598486) * (0.0006435798 - Q.C2_b2)
    if Q.girth2_top2 < 0.007639643 and Q.C2_b2 > 0.0006435798:
        z += -18925.8 * (0.007639643 - Q.girth2_top2) * (Q.C2_b2 - 0.0006435798)
    if Q.N2 < 0.2233283 and Q.pt_6 < 24.42188:
        z += -0.5772732 * (0.2233283 - Q.N2) * (24.42188 - Q.pt_6)
    if Q.lam2 < 0.000537286 and Q.dr1_6 < 0.07796252:
        z += -11176.61 * (0.000537286 - Q.lam2) * (0.07796252 - Q.dr1_6)
    if Q.mass > 76.6557 and Q.M3 < 0.107953:
        z += 1.49788 * (Q.mass - 76.6557) * (0.107953 - Q.M3)
    if Q.sd_mass > 44.82259 and Q.sj3_pair_mass_min < 27.42324:
        z += 0.003970411 * (Q.sd_mass - 44.82259) * (27.42324 - Q.sj3_pair_mass_min)
    if Q.sj3_dr_max < 0.233678 and Q.ptdr0_5 > 7.737156:
        z += 2.646643 * (0.233678 - Q.sj3_dr_max) * (Q.ptdr0_5 - 7.737156)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.398469
    if Q.LHA < 0.2160559:
        z += -13.40902 * Q.LHA + 2.897097
    if Q.z_7 < 0.02807091:
        z += -223.4224 * Q.z_7 + 10.24893
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -122.4352 * Q.z_7 + 7.414131
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -61.83505 * Q.z_7 + 4.420504
    if 6.701242 <= Q.log_sum_pt < 6.842717:
        z += -6.944766 * Q.log_sum_pt + 46.53856
    if 6.842717 <= Q.log_sum_pt < 6.896095:
        z += -29.05273 * Q.log_sum_pt + 197.8171
    if Q.log_sum_pt >= 6.896095:
        z += -40.76201 * Q.log_sum_pt + 278.5654
    if Q.e2_sq < 0.002074109:
        z += 200.0263 * Q.e2_sq + 0.0865044
    if 0.002074109 <= Q.e2_sq < 0.005284669:
        z += -156.1662 * Q.e2_sq + 0.8252864
    if Q.girth2 < 0.001653836:
        z += -620.8823 * Q.girth2 + 1.026838
    if Q.girth < 0.007673833:
        z += -214.4341 * Q.girth + 1.645531
    if Q.pt_7 < 37.15625:
        z += 0.1318823 * Q.pt_7 - 4.900252
    if Q.zdr_0 < 0.0211821:
        z += 61.76186 * Q.zdr_0 - 1.308246
    if Q.pair_mass_0_4 >= 23.26771:
        z += -0.1154497 * Q.pair_mass_0_4 + 2.686249
    if 430.75 <= Q.sum_pt_top5 < 687.4375:
        z += -0.001738826 * Q.sum_pt_top5 + 0.7489995
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.003737998 * Q.sum_pt_top5 - 3.015975
    if Q.pt_5 < 24.57812:
        z += -0.1290162 * Q.pt_5 + 3.170976
    if Q.width < 0.003562611:
        z += -465.042 * Q.width + 1.656764
    if Q.centroid_offset < 0.006789738:
        z += -84.71758 * Q.centroid_offset + 0.5752102
    if Q.girth2_top3 < 0.002151568:
        z += 62.11241 * Q.girth2_top3 - 0.1336391
    if Q.z_6 < 0.08051087:
        z += 10.81142 * Q.z_6 - 0.8704369
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -73.19399 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -871.1133 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.04939969 and Q.sum_pt_top2 < 501.625:
        z += -0.1686926 * (0.04939969 - Q.z_7) * (501.625 - Q.sum_pt_top2)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 40.2:
        z += 0.30385 * (0.07148865 - Q.z_7) * (40.2 - Q.mass_top3)
    if Q.girth2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 118115.4 * (0.001653836 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.LHA < 0.2160559 and Q.lam1 < 0.001503553:
        z += -15041.9 * (0.2160559 - Q.LHA) * (0.001503553 - Q.lam1)
    if Q.log_sum_pt > 6.701242 and Q.dr_2 < 0.01341502:
        z += 1181.364 * (Q.log_sum_pt - 6.701242) * (0.01341502 - Q.dr_2)
    if Q.log_sum_pt > 6.896095 and Q.dr_2 < 0.02270492:
        z += -993.7858 * (Q.log_sum_pt - 6.896095) * (0.02270492 - Q.dr_2)
    if Q.LHA < 0.2160559 and Q.n_dr_0p2_0p4 > 0.0:
        z += -8.982728 * (0.2160559 - Q.LHA) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.sum_pt_top5 > 430.75 and Q.centroid_offset > 0.009480685:
        z += 0.06892429 * (Q.sum_pt_top5 - 430.75) * (Q.centroid_offset - 0.009480685)
    if Q.z_7 < 0.04939969 and Q.dr_2 < 0.009661512:
        z += -5237.94 * (0.04939969 - Q.z_7) * (0.009661512 - Q.dr_2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi > 0.002834884:
        z += -961.6557 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi - 0.002834884)
    if Q.sum_pt_top5 > 430.75 and Q.sj2_dr > 0.1682655:
        z += 0.02167568 * (Q.sum_pt_top5 - 430.75) * (Q.sj2_dr - 0.1682655)
    if Q.pt_7 < 37.15625 and Q.pt_5 > 24.57812:
        z += 0.001382729 * (37.15625 - Q.pt_7) * (Q.pt_5 - 24.57812)
    if Q.sum_pt_top5 > 430.75 and Q.pt_6 < 46.125:
        z += 0.0001727959 * (Q.sum_pt_top5 - 430.75) * (46.125 - Q.pt_6)
    if Q.pair_mass_0_4 > 23.26771 and Q.eccentricity > 0.9704496:
        z += 3.231817 * (Q.pair_mass_0_4 - 23.26771) * (Q.eccentricity - 0.9704496)
    if Q.zdr_0 < 0.0211821 and Q.phi_0 > -0.0297699:
        z += 490.7976 * (0.0211821 - Q.zdr_0) * (Q.phi_0 - -0.0297699)
    if Q.z_7 < 0.07148865 and Q.mean_phi2 < 0.002127561:
        z += 3616.421 * (0.07148865 - Q.z_7) * (0.002127561 - Q.mean_phi2)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 75613.08 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.pair_mass_0_4 > 23.26771 and Q.mean_eta2 < 0.01423545:
        z += 6.005498 * (Q.pair_mass_0_4 - 23.26771) * (0.01423545 - Q.mean_eta2)
    if Q.centroid_offset < 0.006789738 and Q.pt_5 < 56.4375:
        z += -5.177201 * (0.006789738 - Q.centroid_offset) * (56.4375 - Q.pt_5)
    if Q.log_sum_pt > 6.896095 and Q.mean_phi > -0.02594505:
        z += 529.2628 * (Q.log_sum_pt - 6.896095) * (Q.mean_phi - -0.02594505)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.6379248
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 116.0834 * Q.centroid_offset - 0.9393887
    if Q.centroid_offset >= 0.01837778:
        z += 14.16416 * Q.centroid_offset + 0.9336606
    if Q.width < 0.01323868:
        z += 159.7529 * Q.width - 2.114916
    if Q.mass < 21.78408:
        z += -0.09141926 * Q.mass + 2.620619
    if 21.78408 <= Q.mass < 45.595:
        z += -0.02213296 * Q.mass + 1.111281
    if 45.595 <= Q.mass < 60.63098:
        z += -0.006792307 * Q.mass + 0.4118242
    if Q.pt_6 < 19.46875:
        z += -0.4363286 * Q.pt_6 + 8.099425
    if 19.46875 <= Q.pt_6 < 39.75:
        z += 0.02894904 * Q.pt_6 - 0.9589487
    if 39.75 <= Q.pt_6 < 41.21875:
        z += -0.1305706 * Q.pt_6 + 5.381957
    if Q.lam2 < 0.001130645:
        z += 1079.712 * Q.lam2 - 0.5731244
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -284.3365 * Q.lam2 + 0.9691293
    if Q.tau1 < 0.1136369:
        z += -11.03106 * Q.tau1 + 1.253536
    if 0.02400746 <= Q.sj3_dr_min < 0.1278212:
        z += -18.97678 * Q.sj3_dr_min + 0.4555842
    if Q.sj3_dr_min >= 0.1278212:
        z += -2.497503 * Q.sj3_dr_min - 1.650816
    if Q.lam1 < 0.00733008:
        z += -590.0246 * Q.lam1 + 6.580615
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -482.6398 * Q.lam1 + 5.793476
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.07691363 * Q.sj3_pair_mass_min + 0.3462442
    if Q.sum_pt < 615.875:
        z += -0.003550469 * Q.sum_pt + 2.186645
    if Q.sum_pt >= 988.4078:
        z += 0.02323689 * Q.sum_pt - 22.96752
    if Q.e2_sq < 0.0030133:
        z += -486.4457 * Q.e2_sq + 1.465807
    if Q.sj3_dr13 >= 0.181053:
        z += -3.165067 * Q.sj3_dr13 + 0.573045
    if Q.sj3_dr_max < 0.1789613:
        z += -16.89112 * Q.sj3_dr_max + 1.704947
    if 0.1789613 <= Q.sj3_dr_max < 0.1879486:
        z += 10.78131 * Q.sj3_dr_max - 3.247347
    if 0.1879486 <= Q.sj3_dr_max < 0.3012016:
        z += 22.17879 * Q.sj3_dr_max - 5.389487
    if Q.sj3_dr_max >= 0.3012016:
        z += 11.39748 * Q.sj3_dr_max - 2.14214
    if Q.max_dr < 0.1452311:
        z += 35.03171 * Q.max_dr - 5.087694
    if Q.mean_eta >= 0.02644207:
        z += -35.24897 * Q.mean_eta + 0.9320557
    if Q.girth2 < 0.008678045:
        z += 1162.898 * Q.girth2 - 10.09168
    if Q.C2_b2 < 0.02415398:
        z += -66.10892 * Q.C2_b2 + 1.596794
    if Q.sj2_dr < 0.1872617:
        z += -4.89649 * Q.sj2_dr + 0.9169252
    if Q.sj3_dr23 >= 0.2207152:
        z += -1.694968 * Q.sj3_dr23 + 0.3741053
    if Q.log_sum_pt < 6.267538:
        z += -1.15669 * Q.log_sum_pt + 6.771514
    if 6.267538 <= Q.log_sum_pt < 6.638339:
        z += 1.289341 * Q.log_sum_pt - 8.559084
    if Q.z_6 < 0.02160287:
        z += 436.7268 * Q.z_6 - 9.434553
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.003377501 * Q.sum_pt_top5 + 2.836948
    if Q.tau2 >= 0.008780509:
        z += -10.28444 * Q.tau2 + 0.09030258
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += 1.635062 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.5741405 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += 43.09187 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 8216.023 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -1150.361 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -6.780534 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.pt_6 < 29.90625 and Q.n_pt_above_10 < 8.0:
        z += 0.02634107 * (29.90625 - Q.pt_6) * (8.0 - Q.n_pt_above_10)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.8826322 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.001852162 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 64.16369 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.pt_6 < 41.21875 and Q.dr1_7 < 0.1343804:
        z += 0.1704714 * (41.21875 - Q.pt_6) * (0.1343804 - Q.dr1_7)
    if Q.lam1 < 0.01200373 and Q.mean_eta > 0.02644207:
        z += 6733.299 * (0.01200373 - Q.lam1) * (Q.mean_eta - 0.02644207)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.00433128:
        z += -4167.001 * (Q.centroid_offset - 0.01837778) * (0.00433128 - Q.mean_phi2)
    if Q.mean_eta > 0.02644207 and Q.M3 > 0.06688759:
        z += -1212.622 * (Q.mean_eta - 0.02644207) * (Q.M3 - 0.06688759)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.007979125 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    if Q.sj2_dr < 0.1872617 and Q.eccentricity > 0.927072:
        z += 168.9303 * (0.1872617 - Q.sj2_dr) * (Q.eccentricity - 0.927072)
    if Q.sj2_mass1 > 40.2 and Q.tau21_b2 < 0.4490772:
        z += -1.225344 * (Q.sj2_mass1 - 40.2) * (0.4490772 - Q.tau21_b2)
    if Q.sum_pt < 615.875 and Q.pt_5 < 24.57812:
        z += 0.008472129 * (615.875 - Q.sum_pt) * (24.57812 - Q.pt_5)
    if Q.centroid_offset > 0.01837778 and Q.pt_4 > 65.1875:
        z += 1.952657 * (Q.centroid_offset - 0.01837778) * (Q.pt_4 - 65.1875)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 < 0.002776626:
        z += -13.71961 * (Q.sum_pt - 988.4078) * (0.002776626 - Q.mean_phi2)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 > 0.008921136:
        z += -5.221002 * (Q.sum_pt - 988.4078) * (Q.mean_phi2 - 0.008921136)
    if Q.sum_pt > 988.4078 and Q.mean_phi2 < 0.002127561:
        z += 10.82127 * (Q.sum_pt - 988.4078) * (0.002127561 - Q.mean_phi2)
    if Q.tau1 < 0.1136369 and Q.mean_phi2 < 0.01426135:
        z += 1047.338 * (0.1136369 - Q.tau1) * (0.01426135 - Q.mean_phi2)
    if Q.sj2_dr < 0.1872617 and Q.dr1_6 > 0.07796252:
        z += -72.95877 * (0.1872617 - Q.sj2_dr) * (Q.dr1_6 - 0.07796252)
    if Q.log_sum_pt < 6.638339 and Q.z_7 < 0.0753896:
        z += 76.58414 * (6.638339 - Q.log_sum_pt) * (0.0753896 - Q.z_7)
    if Q.centroid_offset > 0.01837778 and Q.sj3_dr12 > 0.1692253:
        z += -165.1611 * (Q.centroid_offset - 0.01837778) * (Q.sj3_dr12 - 0.1692253)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.614472
    if Q.planar_flow < 0.1950135:
        z += 1.517517 * Q.planar_flow - 0.2959364
    if Q.girth2_top2 < 0.001056655:
        z += -507.8775 * Q.girth2_top2 + 0.5366514
    if Q.girth2 < 0.0009641429:
        z += 1295.389 * Q.girth2 - 1.24894
    if 0.001653836 <= Q.girth2 < 0.004372139:
        z += -168.1142 * Q.girth2 + 0.2780334
    if 0.004372139 <= Q.girth2 < 0.007520088:
        z += -351.0778 * Q.girth2 + 1.077976
    if 0.007520088 <= Q.girth2 < 0.01323868:
        z += -465.6422 * Q.girth2 + 1.93951
    if Q.girth2 >= 0.01323868:
        z += 989.8922 * Q.girth2 - 17.32984
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -58.9776 * Q.mass_over_sum_pt + 0.654642
    if 0.07269073 <= Q.mass_over_sum_pt < 0.07992374:
        z += -13.2781 * Q.mass_over_sum_pt - 2.667288
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 49.93319 * Q.mass_over_sum_pt - 7.719371
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 145.5757 * Q.mass_over_sum_pt - 15.82523
    if 0.09041383 <= Q.mass_over_sum_pt < 0.1079857:
        z += -86.11058 * Q.mass_over_sum_pt + 5.122415
    if Q.mass_over_sum_pt >= 0.1079857:
        z += -541.3433 * Q.mass_over_sum_pt + 54.28102
    if Q.tau1 < 0.05356915:
        z += -15.63849 * Q.tau1 + 0.8377406
    if Q.girth < 0.04081947:
        z += 82.05795 * Q.girth - 6.677154
    if 0.04081947 <= Q.girth < 0.08723651:
        z += 71.68901 * Q.girth - 6.253899
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.04414009 * Q.mass - 1.59917
    if Q.mass >= 76.6557:
        z += -0.13792 * Q.mass + 12.35677
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -2.360422 * Q.z_dr_0p1_0p2 + 0.3742642
    if Q.centroid_offset < 0.02076709:
        z += 25.13883 * Q.centroid_offset - 0.5220603
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += 40.94333 * Q.centroid_offset - 1.276235
    if Q.centroid_offset >= 0.03776099:
        z += -74.59528 * Q.centroid_offset + 3.086617
    if Q.width < 0.005590289:
        z += 733.7546 * Q.width - 4.901093
    if 0.005590289 <= Q.width < 0.006679471:
        z += 400.2233 * Q.width - 3.036556
    if 0.006679471 <= Q.width < 0.008678045:
        z += -333.5314 * Q.width + 1.864537
    if Q.width >= 0.008678045:
        z += -589.9925 * Q.width + 4.090118
    if Q.z_7 >= 0.03243272:
        z += 15.86343 * Q.z_7 - 0.5144941
    if 0.03556091 <= Q.e2 < 0.05028464:
        z += -10.24248 * Q.e2 + 0.3642319
    if Q.e2 >= 0.05028464:
        z += 71.86047 * Q.e2 - 3.764285
    if Q.sj2_dr < 0.1294903:
        z += -8.115737 * Q.sj2_dr + 0.678978
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -4.34907 * Q.sj2_dr + 0.191231
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 17.83583 * Q.sj2_dr - 3.339969
    if Q.e2_sq < 0.001101266:
        z += -432.2437 * Q.e2_sq + 0.4760155
    if Q.LHA < 0.3033137:
        z += -5.529236 * Q.LHA + 1.677093
    if Q.pt_7 < 29.04219:
        z += 0.0457802 * Q.pt_7 - 1.329557
    if 0.2037854 <= Q.sd_rg < 0.2787955:
        z += -4.568401 * Q.sd_rg + 0.9309734
    if Q.sd_rg >= 0.2787955:
        z += 35.51837 * Q.sd_rg - 10.24504
    if Q.lam1 < 0.008375572:
        z += 214.5924 * Q.lam1 - 1.797334
    if Q.mass_top5 >= 53.60766:
        z += 0.06505791 * Q.mass_top5 - 3.487602
    if Q.D2_b2 < 0.05744392:
        z += 12.01494 * Q.D2_b2 - 0.690185
    if 0.04889979 <= Q.sj3_dr_max < 0.1426152:
        z += -5.340162 * Q.sj3_dr_max + 0.2611328
    if 0.1426152 <= Q.sj3_dr_max < 0.2623172:
        z += 20.71511 * Q.sj3_dr_max - 3.454744
    if Q.sj3_dr_max >= 0.2623172:
        z += 2.430037 * Q.sj3_dr_max + 1.341745
    if Q.lam2 < 0.0003061234:
        z += 1475.672 * Q.lam2 - 0.4517377
    if Q.D2 < 2.357246:
        z += 0.1774587 * Q.D2 - 0.418314
    if Q.max_dr < 0.1117619:
        z += 8.199044 * Q.max_dr - 0.343899
    if 0.1117619 <= Q.max_dr < 0.177305:
        z += -8.733813 * Q.max_dr + 1.548549
    if Q.tau21_b2 < 0.04019753:
        z += 21.40172 * Q.tau21_b2 - 0.8602961
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 35.28125:
        z += -0.1760653 * (0.1950135 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1042829 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -14.87514 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.05992967 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.girth2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 923.0889 * (Q.girth2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.girth2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += -5360.926 * (Q.girth2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -15925.93 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.girth < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 4806.095 * (0.08723651 - Q.girth) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 2762.124 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    if Q.girth2_top2 < 0.001056655 and Q.tau21_b2 < 0.02656143:
        z += -91233.14 * (0.001056655 - Q.girth2_top2) * (0.02656143 - Q.tau21_b2)
    if Q.lam2 < 0.0003061234 and Q.tau21_b2 < 0.04019753:
        z += 124308.6 * (0.0003061234 - Q.lam2) * (0.04019753 - Q.tau21_b2)
    if Q.mass_over_sum_pt > 0.09041383 and Q.pt_6 < 36.8125:
        z += 15.70038 * (Q.mass_over_sum_pt - 0.09041383) * (36.8125 - Q.pt_6)
    if Q.sj3_dr_max > 0.1426152 and Q.pt_6 < 19.46875:
        z += -0.973034 * (Q.sj3_dr_max - 0.1426152) * (19.46875 - Q.pt_6)
    if Q.girth2 > 0.01323868 and Q.pt_6 < 38.25:
        z += 97.99787 * (Q.girth2 - 0.01323868) * (38.25 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.01109984 and Q.pt_6 < 35.28125:
        z += -0.3991365 * (Q.mass_over_sum_pt - 0.01109984) * (35.28125 - Q.pt_6)
    if Q.width > 0.008678045 and Q.pt_6 < 38.25:
        z += -89.72004 * (Q.width - 0.008678045) * (38.25 - Q.pt_6)
    if Q.mass > 76.6557 and Q.zdr_6 > 0.006366792:
        z += -8.099427 * (Q.mass - 76.6557) * (Q.zdr_6 - 0.006366792)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.5618408
    if Q.girth2 < 0.005019719:
        z += -467.3774 * Q.girth2 + 2.20201
    if 0.005019719 <= Q.girth2 < 0.006679471:
        z += 86.81568 * Q.girth2 - 0.5798829
    if Q.tau1 < 0.05356915:
        z += 39.31009 * Q.tau1 - 2.105808
    if Q.LHA < 0.07658656:
        z += 1.37491 * Q.LHA - 4.113151
    if 0.07658656 <= Q.LHA < 0.1967397:
        z += 33.35618 * Q.LHA - 6.562486
    if Q.log_sum_pt >= 6.701242:
        z += -12.8531 * Q.log_sum_pt + 86.13173
    if Q.girth < 0.06108601:
        z += 77.03488 * Q.girth - 4.705754
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.002087584 * Q.sum_pt_top5 - 1.435084
    if Q.mass < 49.6681:
        z += 0.02420544 * Q.mass - 1.202238
    if Q.width < 0.002635418:
        z += -1759.497 * Q.width + 5.755984
    if 0.002635418 <= Q.width < 0.003562611:
        z += -1206.84 * Q.width + 4.299502
    if Q.mass_over_sum_pt < 0.03319429:
        z += 24.89135 * Q.mass_over_sum_pt - 0.8262505
    if Q.sj3_dr_max < 0.1070199:
        z += 10.84659 * Q.sj3_dr_max - 1.815063
    if 0.1070199 <= Q.sj3_dr_max < 0.1426152:
        z += 18.38059 * Q.sj3_dr_max - 2.621352
    if Q.e2 < 0.01289969:
        z += -85.56252 * Q.e2 + 1.10373
    if Q.sum_pt >= 988.4078:
        z += -0.005863292 * Q.sum_pt + 5.795324
    if Q.girth2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -22.0689 * (0.006679471 - Q.girth2) * (4.721224 - Q.D2_b2)
    if Q.girth2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 21201.13 * (0.006679471 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.girth2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -11.99991 * (0.005019719 - Q.girth2) * (43.5 - Q.pt_7)
    if Q.girth2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 1787.235 * (0.006679471 - Q.girth2) * (Q.phi_0 - -0.04013062)
    if Q.girth2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 7206.612 * (0.005019719 - Q.girth2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -381.127 * (0.06108601 - Q.girth) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.09859104 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.girth < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 110798.6 * (0.06108601 - Q.girth) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 1135.896 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -267.8643 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 20685.39 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.mass < 21.78408 and Q.pt_7 < 48.71875:
        z += 0.003152217 * (21.78408 - Q.mass) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.2567562 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.sum_pt_top5 > 687.4375 and Q.n_pt_above_10 < 8.0:
        z += -0.009103148 * (Q.sum_pt_top5 - 687.4375) * (8.0 - Q.n_pt_above_10)
    if Q.girth2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -18522.45 * (0.005019719 - Q.girth2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.3750503 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.girth < 0.06108601 and Q.width > 0.0003193707:
        z += 9279.752 * (0.06108601 - Q.girth) * (Q.width - 0.0003193707)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2 < 0.006679471:
        z += 3277.692 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.girth2)
    if Q.tau1 < 0.05356915 and Q.width < 0.003562611:
        z += 35914.52 * (0.05356915 - Q.tau1) * (0.003562611 - Q.width)
    if Q.log_sum_pt > 6.701242 and Q.girth2 < 0.0005611231:
        z += 7541.419 * (Q.log_sum_pt - 6.701242) * (0.0005611231 - Q.girth2)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2_top5 < 0.005691733:
        z += -2120.232 * (0.1986272 - Q.sj3_dr_max) * (0.005691733 - Q.girth2_top5)
    if Q.sj3_dr_max < 0.1986272 and Q.lam2 < 0.0001947983:
        z += 15312.15 * (0.1986272 - Q.sj3_dr_max) * (0.0001947983 - Q.lam2)
    if Q.LHA < 0.1967397 and Q.centroid_offset > 0.01837778:
        z += 11180.31 * (0.1967397 - Q.LHA) * (Q.centroid_offset - 0.01837778)
    if Q.log_sum_pt > 6.701242 and Q.n_dr_0p05_0p1 > 2.0:
        z += -1.083764 * (Q.log_sum_pt - 6.701242) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.girth < 0.06108601 and Q.centroid_offset > 0.006789738:
        z += -1491.392 * (0.06108601 - Q.girth) * (Q.centroid_offset - 0.006789738)
    if Q.mass < 29.6447 and Q.centroid_offset < 0.02685622:
        z += -7.484416 * (29.6447 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.03319429 and Q.centroid_offset < 0.02685622:
        z += 5852.024 * (0.03319429 - Q.mass_over_sum_pt) * (0.02685622 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.1986272 and Q.width < 0.003562611:
        z += -4424.902 * (0.1986272 - Q.sj3_dr_max) * (0.003562611 - Q.width)
    if Q.e2 < 0.007078158 and Q.girth2_top5 > 5.672047e-05:
        z += -134883.4 * (0.007078158 - Q.e2) * (Q.girth2_top5 - 5.672047e-05)
    if Q.girth < 0.06108601 and Q.girth2 < 0.003562611:
        z += -31414.88 * (0.06108601 - Q.girth) * (0.003562611 - Q.girth2)
    if Q.sj3_dr_max < 0.1426152 and Q.centroid_offset > 0.01627885:
        z += -2158.191 * (0.1426152 - Q.sj3_dr_max) * (Q.centroid_offset - 0.01627885)
    if Q.e2 < 0.01289969 and Q.psi_0p2 > 0.79448:
        z += -702.4659 * (0.01289969 - Q.e2) * (Q.psi_0p2 - 0.79448)
    if Q.sum_pt > 988.4078 and Q.dr12 < 0.05062541:
        z += 0.07023078 * (Q.sum_pt - 988.4078) * (0.05062541 - Q.dr12)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.613712
    if Q.girth < 0.05464922:
        z += 5.448432 * Q.girth - 0.2977526
    if Q.girth >= 0.0717028:
        z += -44.80066 * Q.girth + 3.212333
    if Q.tau1 < 0.04369778:
        z += -77.41923 * Q.tau1 + 3.383048
    if Q.mass < 29.6447:
        z += 0.1685673 * Q.mass - 6.788631
    if 29.6447 <= Q.mass < 45.595:
        z += 0.07563022 * Q.mass - 4.033539
    if 45.595 <= Q.mass < 53.33237:
        z += 0.0484711 * Q.mass - 2.795219
    if Q.mass >= 53.33237:
        z += -0.02715912 * Q.mass + 1.23832
    if Q.e3 < 2.371297e-05:
        z += 56306.62 * Q.e3 - 1.335197
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -4819.255 * Q.e3 + 0.3926606
    if Q.e3 >= 0.0001869378:
        z += -3764.67 * Q.e3 + 0.1955187
    if Q.girth2 < 0.003562611:
        z += -3993.961 * Q.girth2 + 18.22021
    if 0.003562611 <= Q.girth2 < 0.005019719:
        z += -1760.989 * Q.girth2 + 10.26501
    if 0.005019719 <= Q.girth2 < 0.00609665:
        z += -1323.515 * Q.girth2 + 8.06901
    if Q.girth2 >= 0.01882765:
        z += 144.5764 * Q.girth2 - 2.722035
    if Q.sj3_dr_max < 0.1426152:
        z += 26.14183 * Q.sj3_dr_max - 2.118101
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -22.74702 * Q.sj3_dr_max + 4.85419
    if Q.lam2 < 0.0003061234:
        z += -3416.239 * Q.lam2 + 1.04579
    if Q.lam2 >= 0.001130645:
        z += 615.1565 * Q.lam2 - 0.6955234
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.5566155 * Q.n_dr_0p2_0p4 - 0.5566155
    if Q.lam1 < 0.003377388:
        z += 734.6149 * Q.lam1 - 4.374007
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 531.4921 * Q.lam1 - 3.687983
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 927.7637 * Q.lam1 - 5.841069
    if Q.lam1 >= 0.00595415:
        z += 193.1488 * Q.lam1 - 1.467062
    if Q.centroid_offset < 0.002316125:
        z += 108.9831 * Q.centroid_offset + 3.360715
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -224.954 * Q.centroid_offset + 4.134155
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -64.61744 * Q.e2 + 2.297855
    if Q.e2 >= 0.04447357:
        z += 105.8371 * Q.e2 - 5.282867
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.0290146 * Q.sj3_pair_mass_max - 0.7030275
    if Q.width < 0.0001721983:
        z += -8540.381 * Q.width + 1.470639
    if Q.C3 < 0.0284695:
        z += 26.79748 * Q.C3 - 0.7629107
    if Q.zdr_0 < 0.006292091:
        z += 133.3712 * Q.zdr_0 - 0.8391835
    if Q.mass_over_sum_pt < 0.07269073:
        z += 80.33466 * Q.mass_over_sum_pt - 5.839586
    if Q.sum_pt < 559.6875:
        z += -0.008761744 * Q.sum_pt + 7.592031
    if 559.6875 <= Q.sum_pt < 988.4078:
        z += -0.006270272 * Q.sum_pt + 6.197586
    z += -2.20447 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 5.656781 * Q.log_sum_pt - 39.0097
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.0079053 * Q.sum_pt_top5 + 6.640094
    if Q.pt_4 < 31.125:
        z += -0.05962416 * Q.pt_4 + 1.855802
    if Q.n_for_90pct < 7.0:
        z += 0.2220417 * Q.n_for_90pct - 1.554292
    if Q.max_dr < 0.1117619:
        z += -23.88725 * Q.max_dr + 2.669684
    if Q.M3 < 0.0782171:
        z += -7.400136 * Q.M3 + 0.5788172
    if Q.absphi_1 < 0.02227783:
        z += 17.15288 * Q.absphi_1 - 0.382129
    if Q.N2 >= 0.2233283:
        z += 2.500529 * Q.N2 - 0.5584388
    if Q.z_5 < 0.06503035:
        z += -12.07492 * Q.z_5 + 0.7852364
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -235.2245 * Q.mass_over_sum_pt_sq + 1.689579
    if Q.D2 >= 2.055451:
        z += -0.4935104 * Q.D2 + 1.014386
    if Q.e2_sq < 0.0030133:
        z += 841.6344 * Q.e2_sq - 2.536097
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 4.694232 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.1363605 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.girth2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 781.7286 * (0.00609665 - Q.girth2) * (0.3220738 - Q.planar_flow)
    if Q.girth2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -12592.37 * (0.00609665 - Q.girth2) * (0.002834884 - Q.mean_phi)
    if Q.mass < 53.33237 and Q.D2 > 2.843757:
        z += 0.01085954 * (53.33237 - Q.mass) * (Q.D2 - 2.843757)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -799.9734 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -267.7732 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.sj3_dr_max < 0.1070199 and Q.abseta_1 > 0.03625488:
        z += -1080.664 * (0.1070199 - Q.sj3_dr_max) * (Q.abseta_1 - 0.03625488)
    if Q.tau1 < 0.04369778 and Q.mean_phi < -0.01275329:
        z += 2545.503 * (0.04369778 - Q.tau1) * (-0.01275329 - Q.mean_phi)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += 4093.496 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.centroid_offset < 0.01837778 and Q.tau4 > 0.001224244:
        z += -7070.696 * (0.01837778 - Q.centroid_offset) * (Q.tau4 - 0.001224244)
    if Q.girth2 < 0.00609665 and Q.mean_phi > 0.02612796:
        z += -45843.1 * (0.00609665 - Q.girth2) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -29.96581 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -44.79714 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.3883631 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -1.92324 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 935.4972 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.sj3_dr_min < 0.2089872:
        z += -4.405389 * (Q.n_dr_0p2_0p4 - 1.0) * (0.2089872 - Q.sj3_dr_min)
    if Q.girth < 0.05464922 and Q.tau21_b2 < 0.02656143:
        z += 4484.49 * (0.05464922 - Q.girth) * (0.02656143 - Q.tau21_b2)
    if Q.tau1 < 0.04369778 and Q.tau21_b2 < 0.02656143:
        z += -5411.655 * (0.04369778 - Q.tau1) * (0.02656143 - Q.tau21_b2)
    if Q.mass_over_sum_pt < 0.07269073 and Q.phi_0 < -0.004917145:
        z += 570.0371 * (0.07269073 - Q.mass_over_sum_pt) * (-0.004917145 - Q.phi_0)
    if Q.centroid_offset < 0.01837778 and Q.mean_eta > 6.288824e-05:
        z += 4869.75 * (0.01837778 - Q.centroid_offset) * (Q.mean_eta - 6.288824e-05)
    if Q.girth < 0.05464922 and Q.z_6 > 0.03448406:
        z += -851.4941 * (0.05464922 - Q.girth) * (Q.z_6 - 0.03448406)
    if Q.sum_pt < 988.4078 and Q.dr_2 < 0.01778111:
        z += -0.1515805 * (988.4078 - Q.sum_pt) * (0.01778111 - Q.dr_2)
    if Q.e3 > 0.0001869378 and Q.pt_2 > 56.5:
        z += 27.2556 * (Q.e3 - 0.0001869378) * (Q.pt_2 - 56.5)
    if Q.tau1 < 0.04369778 and Q.zdr_2 > 0.00759156:
        z += -18745.7 * (0.04369778 - Q.tau1) * (Q.zdr_2 - 0.00759156)
    if Q.log_sum_pt > 6.896095 and Q.zdr_5 < 0.001255404:
        z += 8671.945 * (Q.log_sum_pt - 6.896095) * (0.001255404 - Q.zdr_5)
    if Q.sd_mass > 45.595 and Q.n_dr_0p05_0p1 > 7.0:
        z += -0.06982752 * (Q.sd_mass - 45.595) * (Q.n_dr_0p05_0p1 - 7.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.877862
    z += -5.505892 * Q.e2
    if Q.lam1 < 0.001503553:
        z += 1329.819 * Q.lam1 - 4.256684
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 614.9619 * Q.lam1 - 3.181858
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 343.9872 * Q.lam1 - 2.048152
    if Q.lam1 >= 0.00733008:
        z += -157.3286 * Q.lam1 + 1.153231
    if Q.lam2 >= 0.0003061234:
        z += 1174.042 * Q.lam2 - 0.3594017
    if Q.pt_7 < 45.75:
        z += 0.01588354 * Q.pt_7 - 0.7266721
    if Q.sj3_dr_min >= 0.1278212:
        z += 8.492401 * Q.sj3_dr_min - 1.085509
    if 22.18342 <= Q.mass_top5 < 45.32077:
        z += 0.01116601 * Q.mass_top5 - 0.2477001
    if Q.mass_top5 >= 45.32077:
        z += -0.01087941 * Q.mass_top5 + 0.7514148
    if Q.zdr_0 < 0.004918231:
        z += 106.1569 * Q.zdr_0 - 0.7460563
    if 0.004918231 <= Q.zdr_0 < 0.0211821:
        z += 13.76992 * Q.zdr_0 - 0.2916758
    if Q.M2 < 0.02563286:
        z += 27.69737 * Q.M2 - 0.709963
    if Q.sum_pt >= 988.4078:
        z += -0.005811264 * Q.sum_pt + 5.743899
    if Q.LHA >= 0.3033137:
        z += -28.88851 * Q.LHA + 8.762281
    if Q.tau1 >= 0.05356915:
        z += 14.01775 * Q.tau1 - 0.750919
    if Q.girth2 < 0.002635418:
        z += 139.9156 * Q.girth2 - 0.3687362
    if 0.007520088 <= Q.girth2 < 0.02530566:
        z += 438.4937 * Q.girth2 - 3.297511
    if Q.girth2 >= 0.02530566:
        z += 328.2404 * Q.girth2 - 0.5074793
    if Q.e3 >= 3.892127e-05:
        z += -2465.483 * Q.e3 + 0.09595973
    if Q.mass < 76.6557:
        z += -0.02573304 * Q.mass + 1.972584
    if Q.girth2_top3 < 0.002151568:
        z += -428.231 * Q.girth2_top3 + 0.921368
    if Q.sj2_dr >= 0.3003793:
        z += 4.914724 * Q.sj2_dr - 1.476281
    if Q.sj3_dr_max >= 0.1986272:
        z += -5.683162 * Q.sj3_dr_max + 1.128831
    if Q.C2_b2 >= 0.009032972:
        z += 43.6594 * Q.C2_b2 - 0.3943742
    if Q.sj3_pair_mass_min >= 15.95929:
        z += 0.112156 * Q.sj3_pair_mass_min - 1.789929
    if Q.zdr_7 < 0.00325401:
        z += 87.63116 * Q.zdr_7 - 0.2851527
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.06440124 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.centroid_offset > 0.01258764:
        z += 1716.548 * (0.0211821 - Q.zdr_0) * (Q.centroid_offset - 0.01258764)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.08885276 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -728.452 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -24.41942 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 146.7586 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -250.5117 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.lam1 > 0.00733008 and Q.D2_b2 > 1.129616:
        z += -64.61012 * (Q.lam1 - 0.00733008) * (Q.D2_b2 - 1.129616)
    if Q.sj3_pair_mass_min > 11.051 and Q.sj3_pairmin_over_m > 0.28737:
        z += -0.1998451 * (Q.sj3_pair_mass_min - 11.051) * (Q.sj3_pairmin_over_m - 0.28737)
    if Q.tau1 > 0.05356915 and Q.D2 < 1.002471:
        z += 4.296139 * (Q.tau1 - 0.05356915) * (1.002471 - Q.D2)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.932914
    if Q.planar_flow < 0.2534037:
        z += -7.084084 * Q.planar_flow + 1.795133
    if Q.sj2_dr >= 0.1778793:
        z += 7.56904 * Q.sj2_dr - 1.346375
    if Q.mass < 15.45403:
        z += -0.1542963 * Q.mass + 0.8283801
    if 15.45403 <= Q.mass < 49.6681:
        z += 0.01939805 * Q.mass - 1.855898
    if 49.6681 <= Q.mass < 69.61135:
        z += 0.04474865 * Q.mass - 3.115014
    if Q.girth2 < 0.01323868:
        z += -349.218 * Q.girth2 + 4.623184
    if Q.tau1 < 0.09538712:
        z += 13.01384 * Q.tau1 - 1.241352
    if Q.LHA < 0.3127275:
        z += 4.647601 * Q.LHA - 1.453432
    if Q.centroid_offset < 0.03776099:
        z += -81.97433 * Q.centroid_offset + 3.095432
    if Q.centroid_offset >= 0.04990367:
        z += -669.5516 * Q.centroid_offset + 33.41308
    if Q.sj3_dr_max < 0.1426152:
        z += 12.5645 * Q.sj3_dr_max - 0.9137885
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 17.45616 * Q.sj3_dr_max - 1.611413
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -14.35535 * Q.sj3_dr_max + 3.765656
    if Q.width < 0.006679471:
        z += -1832.711 * Q.width + 14.90832
    if 0.006679471 <= Q.width < 0.008678045:
        z += -1334.338 * Q.width + 11.57945
    if Q.lam1 < 0.00733008:
        z += 471.4335 * Q.lam1 - 3.738049
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 270.1159 * Q.lam1 - 2.262375
    if Q.girth < 0.02054282:
        z += 116.3084 * Q.girth - 4.775512
    if 0.02054282 <= Q.girth < 0.02689598:
        z += 73.52392 * Q.girth - 3.896599
    if 0.02689598 <= Q.girth < 0.0717028:
        z += 42.83055 * Q.girth - 3.07107
    if Q.e2_sq < 0.008168571:
        z += 3527.515 * Q.e2_sq - 28.81475
    if Q.z_7 >= 0.01685855:
        z += 49.18912 * Q.z_7 - 0.829257
    if Q.z_dr_0p05_0p1 >= 0.8460335:
        z += 2.469528 * Q.z_dr_0p05_0p1 - 2.089304
    if Q.pt_6 < 24.42188:
        z += 0.05555319 * Q.pt_6 - 1.040896
    if 24.42188 <= Q.pt_6 < 41.21875:
        z += -0.01880213 * Q.pt_6 + 0.7750002
    if Q.pt_7 >= 29.04219:
        z += -0.05250817 * Q.pt_7 + 1.524952
    if Q.C2 < 0.03578649:
        z += 18.01789 * Q.C2 - 0.6447969
    if Q.max_dr < 0.1027585:
        z += -1.497084 * Q.max_dr + 0.4446536
    if 0.1027585 <= Q.max_dr < 0.1452311:
        z += -6.847133 * Q.max_dr + 0.9944167
    if Q.e3 < 1.050302e-05:
        z += -92353.96 * Q.e3 + 0.9699956
    if Q.sum_pt_top5 < 506.875:
        z += 0.004508946 * Q.sum_pt_top5 - 2.285472
    if Q.sum_pt >= 988.4078:
        z += -0.002836768 * Q.sum_pt + 2.803883
    if Q.mass_over_sum_pt < 0.07637363:
        z += 51.10193 * Q.mass_over_sum_pt - 3.90284
    if Q.lam2 < 0.001130645:
        z += -410.9954 * Q.lam2 + 0.4646898
    if Q.mass_over_sum_pt_sq < 0.00817466:
        z += -2927.786 * Q.mass_over_sum_pt_sq + 23.93366
    if Q.n_dr_0p1_0p2 >= 3.0:
        z += -0.3524166 * Q.n_dr_0p1_0p2 + 1.05725
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01427528 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1374526 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -535252.5 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.sj2_dr > 0.1778793 and Q.lam2 < 0.001130645:
        z += -6853.246 * (Q.sj2_dr - 0.1778793) * (0.001130645 - Q.lam2)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.2139412 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset < 0.03776099 and Q.mean_phi < -0.001813533:
        z += -1160.443 * (0.03776099 - Q.centroid_offset) * (-0.001813533 - Q.mean_phi)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += 1.102595 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -298.4192 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.sj2_dr > 0.1778793 and Q.n_pt_above_50 > 4.0:
        z += 0.7661647 * (Q.sj2_dr - 0.1778793) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.01437952 and Q.abseta_4 < 0.03601074:
        z += -766.3616 * (0.01437952 - Q.centroid_offset) * (0.03601074 - Q.abseta_4)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 186.2592 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.pt_6 < 41.21875 and Q.D2_b2 < 4.721224:
        z += -0.006033824 * (41.21875 - Q.pt_6) * (4.721224 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += 1927.742 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    if Q.sj3_dr_max < 0.2623172 and Q.phi_0 < -0.0216713:
        z += 89.3539 * (0.2623172 - Q.sj3_dr_max) * (-0.0216713 - Q.phi_0)
    if Q.centroid_offset < 0.01437952 and Q.sj3_pair_mass_min < 15.95929:
        z += -8.115129 * (0.01437952 - Q.centroid_offset) * (15.95929 - Q.sj3_pair_mass_min)
    if Q.girth < 0.02689598 and Q.sj3_pair_mass_min > 1.590484:
        z += -6.579625 * (0.02689598 - Q.girth) * (Q.sj3_pair_mass_min - 1.590484)
    if Q.planar_flow < 0.2534037 and Q.sj3_pair_mass_min > 1.282345:
        z += -0.1460161 * (0.2534037 - Q.planar_flow) * (Q.sj3_pair_mass_min - 1.282345)
    if Q.centroid_offset > 0.04990367 and Q.zdr_6 > 0.007390416:
        z += -133744.0 * (Q.centroid_offset - 0.04990367) * (Q.zdr_6 - 0.007390416)
    if Q.sum_pt_top5 < 506.875 and Q.z_dr_0p1_0p2 > 0.04510668:
        z += 0.00549201 * (506.875 - Q.sum_pt_top5) * (Q.z_dr_0p1_0p2 - 0.04510668)
    if Q.centroid_offset > 0.04990367 and Q.tau3 > 0.001996306:
        z += -105255.2 * (Q.centroid_offset - 0.04990367) * (Q.tau3 - 0.001996306)
    if Q.girth2 < 0.004372139 and Q.sj3_dr13 > 0.1659434:
        z += 4685.336 * (0.004372139 - Q.girth2) * (Q.sj3_dr13 - 0.1659434)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.6079916
    if Q.girth2 >= 0.01882765:
        z += 260.7203 * Q.girth2 - 4.908751
    if Q.mass >= 91.19:
        z += 0.05454314 * Q.mass - 4.973789
    if Q.zdr_0 >= 0.03981924:
        z += -29.03299 * Q.zdr_0 + 1.156072
    if Q.girth2_top2 >= 0.01403324:
        z += -32.49094 * Q.girth2_top2 + 0.4559533
    if Q.e2 >= 0.06344108:
        z += -32.60921 * Q.e2 + 2.068763
    if Q.centroid_offset >= 0.04990367:
        z += 22.33777 * Q.centroid_offset - 1.114736
    if Q.girth2 > 0.01882765 and Q.lam2 > 0.000537286:
        z += 5036.278 * (Q.girth2 - 0.01882765) * (Q.lam2 - 0.000537286)
    if Q.girth2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -2.272871 * (Q.girth2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.mass > 69.61135 and Q.centroid_offset > 0.009480685:
        z += 0.590303 * (Q.mass - 69.61135) * (Q.centroid_offset - 0.009480685)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.8444358
    if Q.girth < 0.007673833:
        z += 82.26117 * Q.girth + 7.69067
    if 0.007673833 <= Q.girth < 0.1484084:
        z += -59.13208 * Q.girth + 8.775698
    if Q.lam1 < 0.006506576:
        z += -319.0382 * Q.lam1 + 4.047754
    if 0.006506576 <= Q.lam1 < 0.01643375:
        z += -198.6374 * Q.lam1 + 3.264357
    if 658.125 <= Q.sum_pt_top5 < 791.125:
        z += -0.006592536 * Q.sum_pt_top5 + 4.338713
    if Q.sum_pt_top5 >= 791.125:
        z += 0.003395726 * Q.sum_pt_top5 - 3.563251
    if Q.e3 < 5.334511e-05:
        z += -18235.44 * Q.e3 + 0.9727714
    if Q.pt_6 < 31.90625:
        z += 0.01189002 * Q.pt_6 - 0.3793661
    if Q.e2 < 0.08000524:
        z += 52.47295 * Q.e2 - 4.198111
    if Q.mass >= 49.6681:
        z += -0.02639315 * Q.mass + 1.310898
    if Q.z_5 < 0.02818362:
        z += 189.3025 * Q.z_5 - 5.335231
    if Q.z_7 < 0.02807091:
        z += 154.5652 * Q.z_7 - 4.338786
    if Q.sj3_dr23 >= 0.1974628:
        z += -1.342353 * Q.sj3_dr23 + 0.2650647
    if Q.log_sum_pt >= 6.502799:
        z += 0.9615516 * Q.log_sum_pt - 6.252777
    if Q.pt_7 >= 48.71875:
        z += -0.05140126 * Q.pt_7 + 2.504205
    if Q.width < 0.007520088:
        z += 79.11963 * Q.width - 0.3948124
    if 0.007520088 <= Q.width < 0.01323868:
        z += -35.00415 * Q.width + 0.4634085
    if Q.z_6 < 0.02160287:
        z += 118.2688 * Q.z_6 - 2.554946
    if Q.girth < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -74.34982 * (0.1484084 - Q.girth) * (6.804164 - Q.log_sum_pt)
    if Q.girth < 0.1484084 and Q.pt_7 < 38.53125:
        z += -0.9281986 * (0.1484084 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -1176805.0 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0007890644 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -2.305436 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.girth < 0.1484084 and Q.z_7 > 0.06164517:
        z += 393.681 * (0.1484084 - Q.girth) * (Q.z_7 - 0.06164517)
    if Q.girth < 0.1484084 and Q.lam2 < 0.000537286:
        z += -8636.869 * (0.1484084 - Q.girth) * (0.000537286 - Q.lam2)
    if Q.girth < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -1.596882 * (0.1484084 - Q.girth) * (Q.sj2_mass1 - 31.78116)
    if Q.girth < 0.1484084 and Q.M3 < 0.07474969:
        z += 123.312 * (0.1484084 - Q.girth) * (0.07474969 - Q.M3)
    if Q.e3 < 5.334511e-05 and Q.D3 > 0.2213841:
        z += 2961.202 * (5.334511e-05 - Q.e3) * (Q.D3 - 0.2213841)
    if Q.girth < 0.1484084 and Q.tau2 > 0.008780509:
        z += 267.2991 * (0.1484084 - Q.girth) * (Q.tau2 - 0.008780509)
    if Q.log_sum_pt > 6.502799 and Q.zdr_7 > 0.0007928864:
        z += -349.4281 * (Q.log_sum_pt - 6.502799) * (Q.zdr_7 - 0.0007928864)
    if Q.log_sum_pt > 6.502799 and Q.pair_mass_0_7 > 10.2219:
        z += 0.09625755 * (Q.log_sum_pt - 6.502799) * (Q.pair_mass_0_7 - 10.2219)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.0009289747 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0005823282 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.1090685
    if Q.planar_flow < 0.08366273:
        z += -2.041946 * Q.planar_flow + 0.3734874
    if 0.08366273 <= Q.planar_flow < 0.1115136:
        z += -7.276356 * Q.planar_flow + 0.8114124
    if 0.7143804 <= Q.psi_0p1 < 0.9761279:
        z += -0.8256235 * Q.psi_0p1 + 0.5898092
    if Q.psi_0p1 >= 0.9761279:
        z += -21.45759 * Q.psi_0p1 + 20.72924
    if 0.0009641429 <= Q.girth2 < 0.007520088:
        z += -199.3247 * Q.girth2 + 0.1921775
    if 0.007520088 <= Q.girth2 < 0.008678045:
        z += -1229.261 * Q.girth2 + 7.937387
    if Q.girth2 >= 0.008678045:
        z += -2264.432 * Q.girth2 + 16.92065
    if 0.002074109 <= Q.e2_sq < 0.0030133:
        z += 308.1275 * Q.e2_sq - 0.63909
    if 0.0030133 <= Q.e2_sq < 0.01165737:
        z += 39.03647 * Q.e2_sq + 0.171762
    if Q.e2_sq >= 0.01165737:
        z += 1114.372 * Q.e2_sq - 12.36382
    if 0.003562611 <= Q.width < 0.005590289:
        z += 488.8137 * Q.width - 1.741453
    if 0.005590289 <= Q.width < 0.006679471:
        z += 594.1047 * Q.width - 2.33006
    if 0.006679471 <= Q.width < 0.01323868:
        z += -567.1791 * Q.width + 5.426702
    if Q.width >= 0.01323868:
        z += -2037.083 * Q.width + 24.88628
    if 0.01655442 <= Q.e2 < 0.04110972:
        z += -161.3663 * Q.e2 + 2.671326
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += -114.8512 * Q.e2 + 0.7591033
    if Q.e2 >= 0.05028464:
        z += 127.1946 * Q.e2 - 11.41209
    if 0.02685622 <= Q.centroid_offset < 0.03776099:
        z += -70.57018 * Q.centroid_offset + 1.895248
    if 0.03776099 <= Q.centroid_offset < 0.04990367:
        z += -117.0319 * Q.centroid_offset + 3.649691
    if Q.centroid_offset >= 0.04990367:
        z += -982.4942 * Q.centroid_offset + 46.83943
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 82.67535 * Q.mass_over_sum_pt - 6.607723
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 212.44 * Q.mass_over_sum_pt - 17.60549
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 346.3331 * Q.mass_over_sum_pt - 29.71128
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1864514 * Q.n_dr_0_0p05 + 0.9322568
    if Q.e3 < 5.334511e-05:
        z += 26811.3 * Q.e3 - 1.906308
    if 5.334511e-05 <= Q.e3 < 8.147744e-05:
        z += 16922.01 * Q.e3 - 1.378762
    if Q.sd_mass < 49.91626:
        z += -0.05018175 * Q.sd_mass + 1.961271
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.02204403 * Q.sd_mass - 1.643969
    if 0.02689598 <= Q.girth < 0.04081947:
        z += 86.42634 * Q.girth - 2.324521
    if 0.04081947 <= Q.girth < 0.08065885:
        z += 152.6357 * Q.girth - 5.02715
    if 0.08065885 <= Q.girth < 0.08723651:
        z += 209.0558 * Q.girth - 9.577932
    if 0.08723651 <= Q.girth < 0.1019409:
        z += 32.69567 * Q.girth + 5.807107
    if Q.girth >= 0.1019409:
        z += -67.84852 * Q.girth + 16.05668
    if Q.mass >= 69.61135:
        z += -0.02929639 * Q.mass + 2.039361
    if Q.mass_top5 >= 49.18618:
        z += -0.01260475 * Q.mass_top5 + 0.6199793
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -5.499785 * Q.sj2_dr + 0.3384642
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -10.16056 * Q.sj2_dr + 0.941989
    if 0.1591713 <= Q.sj2_dr < 0.1778793:
        z += 30.13636 * Q.sj2_dr - 5.472124
    if 0.1778793 <= Q.sj2_dr < 0.2001708:
        z += 21.76984 * Q.sj2_dr - 3.983894
    if Q.sj2_dr >= 0.2001708:
        z += 6.395785 * Q.sj2_dr - 0.9064561
    if Q.lam2 < 0.000537286:
        z += -894.291 * Q.lam2 + 0.48049
    if Q.C2_b2 < 0.004032342:
        z += 153.0448 * Q.C2_b2 - 0.6171289
    if Q.LHA >= 0.3033137:
        z += 8.566813 * Q.LHA - 2.598432
    if Q.z_dr_0p05_0p1 < 0.163898:
        z += 2.427491 * Q.z_dr_0p05_0p1 - 0.397861
    if Q.sum_pt < 715.4688:
        z += 0.005893283 * Q.sum_pt - 4.21646
    if Q.z_dr_0_0p05 >= 0.1515405:
        z += 1.476969 * Q.z_dr_0_0p05 - 0.2238206
    if Q.z_dr_0p1_0p2 < 0.07870506:
        z += -4.726592 * Q.z_dr_0p1_0p2 + 0.3720067
    if Q.sd_rg < 0.2330919:
        z += 10.07115 * Q.sd_rg - 3.269558
    if 0.2330919 <= Q.sd_rg < 0.324646:
        z += -22.43345 * Q.sd_rg + 4.307001
    if Q.sd_rg >= 0.324646:
        z += -32.5046 * Q.sd_rg + 7.576559
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -3417.14 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 616.1302 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.width < 0.00609665:
        z += 3070.628 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.width)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.01646807 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -370.9294 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 1.078728 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 229644.3 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.2001708 and Q.z_5 < 0.1058993:
        z += -74.14345 * (Q.sj2_dr - 0.2001708) * (0.1058993 - Q.z_5)
    if Q.sj2_dr > 0.1591713 and Q.D2_b2 < 0.08499387:
        z += 297.5947 * (Q.sj2_dr - 0.1591713) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.2001708 and Q.D2_b2 < 0.08499387:
        z += -232.7587 * (Q.sj2_dr - 0.2001708) * (0.08499387 - Q.D2_b2)
    if Q.sj2_dr > 0.1294903 and Q.D2_b2 < 0.08499387:
        z += -119.269 * (Q.sj2_dr - 0.1294903) * (0.08499387 - Q.D2_b2)
    if Q.planar_flow < 0.1115136 and Q.mean_eta > -0.006779839:
        z += -97.30527 * (0.1115136 - Q.planar_flow) * (Q.mean_eta - -0.006779839)
    if Q.z_dr_0p05_0p1 < 0.5882598 and Q.sum_pt < 715.4688:
        z += 0.002987334 * (0.5882598 - Q.z_dr_0p05_0p1) * (715.4688 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.C2_b2 < 0.0008333816:
        z += 1129652.0 * (Q.centroid_offset - 0.04990367) * (0.0008333816 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.C2_b2 < 0.02415398:
        z += -41.48815 * (Q.z_dr_0p05_0p1 - 0.7509095) * (0.02415398 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.zdr_5 > 0.0155217:
        z += -3130.635 * (Q.z_dr_0p05_0p1 - 0.7509095) * (Q.zdr_5 - 0.0155217)
    if Q.sj2_dr > 0.1591713 and Q.dr_2 < 0.03293672:
        z += -527.7355 * (Q.sj2_dr - 0.1591713) * (0.03293672 - Q.dr_2)
    if Q.sj2_dr > 0.2001708 and Q.dr_2 < 0.05056028:
        z += 265.2737 * (Q.sj2_dr - 0.2001708) * (0.05056028 - Q.dr_2)
    if Q.centroid_offset > 0.02685622 and Q.pt_1 < 150.625:
        z += 0.5656453 * (Q.centroid_offset - 0.02685622) * (150.625 - Q.pt_1)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.5372828
    if Q.N2 < 0.2233283:
        z += 7.421145 * Q.N2 - 1.657351
    if Q.girth2 < 0.007520088:
        z += 697.7913 * Q.girth2 - 5.247452
    if Q.mass_over_sum_pt < 0.1309286:
        z += 32.7811 * Q.mass_over_sum_pt - 4.291985
    if Q.width < 0.00609665:
        z += -958.2174 * Q.width + 14.58428
    if 0.00609665 <= Q.width < 0.01323868:
        z += -1017.38 * Q.width + 14.94497
    if 0.01323868 <= Q.width < 0.01882765:
        z += -264.1283 * Q.width + 4.972916
    if Q.e2 < 0.04110972:
        z += -193.9847 * Q.e2 + 7.974655
    if Q.e2_sq < 0.008168571:
        z += 1169.422 * Q.e2_sq - 10.64635
    if 0.008168571 <= Q.e2_sq < 0.01165737:
        z += 313.532 * Q.e2_sq - 3.65496
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -664.5536 * Q.mass_over_sum_pt_sq + 4.773379
    if Q.lam1 < 0.00483998:
        z += -107.4469 * Q.lam1 + 0.520041
    if Q.lam2 < 0.001130645:
        z += 229.7644 * Q.lam2 - 0.2597819
    if Q.girth2_top3 < 0.002151568:
        z += 350.4885 * Q.girth2_top3 - 0.7540997
    if Q.girth < 0.1019409:
        z += 90.52842 * Q.girth - 9.228551
    if Q.sj2_dr < 0.1492731:
        z += -11.62661 * Q.sj2_dr + 0.6404711
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -14.60471 * Q.sj2_dr + 1.085022
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 30.23522 * Q.sj2_dr - 6.052208
    if Q.girth2_top2 < 0.0005124533:
        z += 13575.97 * Q.girth2_top2 - 6.95705
    if Q.sj3_pair_mass_max >= 80.4:
        z += -0.04679999 * Q.sj3_pair_mass_max + 3.762719
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -12.4387 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.N2 < 0.2233283 and Q.max_dr < 0.121681:
        z += -47.87085 * (0.2233283 - Q.N2) * (0.121681 - Q.max_dr)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 < 716.8828:
        z += 0.01839205 * (0.2233283 - Q.N2) * (716.8828 - Q.sum_pt_top5)
    if Q.girth2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -1558.609 * (0.007520088 - Q.girth2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 49.72276 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.z_dr_0p1_0p2 > 0.4684459:
        z += 69.60583 * (0.1309286 - Q.mass_over_sum_pt) * (Q.z_dr_0p1_0p2 - 0.4684459)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 < 4.0:
        z += 1.639021 * (0.2233283 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.03175499 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.N2 < 0.2233283 and Q.pt_entropy < 1.54073:
        z += -16.2082 * (0.2233283 - Q.N2) * (1.54073 - Q.pt_entropy)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.4375 - 0.15625 * h0 + 0.390625 * h1 + 0.4296875 * h2 - 0.03125 * h4 - 0.1875 * h5 + 0.109375 * h6 + 0.171875 * h9


def logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.03125 - 0.09375 * h4 + 0.046875 * h5 + 0.125 * h6 + 0.0625 * h8 + 0.2539062 * h9 - 0.125 * h10 + 0.0625 * h15


def logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.125 + 0.34375 * h0 - 0.03125 * h1 - 0.5 * h3 - 0.3125 * h6 + 0.21875 * h7 - 0.25 * h8 - 0.03125 * h9 + 0.375 * h11 + 0.0703125 * h13 - 0.75 * h14 - 0.6875 * h15


def logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.09375 + 0.125 * h1 - 0.5625 * h3 + 0.078125 * h4 + 0.015625 * h5 - 0.375 * h6 + 0.46875 * h7 - 0.03125 * h9 + 0.0546875 * h13 + 0.375 * h14 - 0.15625 * h15


def logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 1.34375 + 0.015625 * h0 + 0.0625 * h3 + 0.125 * h4 - 0.25 * h5 + 0.1875 * h8 + 0.375 * h10 - 0.5 * h12 - 0.40625 * h13


def logits(h):
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15 = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15), logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15)]


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
