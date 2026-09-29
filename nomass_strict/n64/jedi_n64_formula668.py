"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned; no mass observables or exact equivalents), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.7% (the network: 81.1%); same class as the network for 92.8% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
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
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.ptdr0_10               pT10 · ΔR(0, 10) [GeV]
  Q.pt_11                  pT of particle 11 [GeV]
  Q.ptdr0_12               pT12 · ΔR(0, 12) [GeV]
  Q.ptdr0_14               pT14 · ΔR(0, 14) [GeV]
  Q.ptdr0_2                pT2 · ΔR(0, 2) [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.ptdr0_4                pT4 · ΔR(0, 4) [GeV]
  Q.ptdr0_5                pT5 · ΔR(0, 5) [GeV]
  Q.ptdr0_6                pT6 · ΔR(0, 6) [GeV]
  Q.pt_8                   pT of particle 8 [GeV]
  Q.ptdr0_8                pT8 · ΔR(0, 8) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_pt               pT [GeV] of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_0                    pT of particle 0 / total pT
  Q.z_14                   pT of particle 14 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft9_z                pT share of the 9. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top3_slots           pT share of the 3 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the girth)
  Q.zdr_4                  pT share × ΔR of particle 4 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.pt2_over_pt0           pT2 / pT0
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_12              |Δη| of particle 12
  Q.abseta_9               |Δη| of particle 9
  Q.absphi_1               |Δφ| of particle 1
  Q.absphi_13              |Δφ| of particle 13
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft2_dr0              ΔR between the hardest and the 2. softest real particle (0 if among the 15 hardest)
  Q.dr0_12                 ΔR between particle 12 and the hardest particle
  Q.dr1_11                 ΔR between particle 11 and the 2nd-hardest particle
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_11                  ΔR of particle 11 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.soft2_dr               ΔR from the jet axis of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_dr               ΔR from the jet axis of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.phi_10                 Δφ of particle 10
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
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
        z_top5=sum(zs[:5]),
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
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        ptdr0_10=pt[10] * math.sqrt(dist2(0, 10)) if pt[10] > 0 else 0.0,
        pt_11=pt[11],
        ptdr0_12=pt[12] * math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        ptdr0_14=pt[14] * math.sqrt(dist2(0, 14)) if pt[14] > 0 else 0.0,
        ptdr0_2=pt[2] * math.sqrt(dist2(0, 2)) if pt[2] > 0 else 0.0,
        pt_3=pt[3],
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        ptdr0_4=pt[4] * math.sqrt(dist2(0, 4)) if pt[4] > 0 else 0.0,
        ptdr0_5=pt[5] * math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        ptdr0_6=pt[6] * math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        pt_8=pt[8],
        ptdr0_8=pt[8] * math.sqrt(dist2(0, 8)) if pt[8] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        soft9_pt=softp(9, 'pt'),
        z_0=z[0],
        z_14=z[14],
        z_3=z[3],
        z_7=z[7],
        soft1_z=softp(1, 'z'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft9_z=softp(9, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top3_slots=sum(pt[:3]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_3=z[3] * dr[3],
        zdr_4=z[4] * dr[4],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_zg=softdrop("zg"),
        abseta_12=abs(eta[12]),
        abseta_9=abs(eta[9]),
        absphi_1=abs(phi[1]),
        absphi_13=abs(phi[13]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        soft2_dr0=softp(2, 'dr0'),
        dr0_12=math.sqrt(dist2(0, 12)) if pt[12] > 0 else 0.0,
        dr1_11=math.sqrt(dist2(1, 11)) if pt[11] > 0 else 0.0,
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_11=dr[11] if pt[11] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        soft2_dr=softp(2, 'dr'),
        soft6_dr=softp(6, 'dr'),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        phi_10=phi[10],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -0.4404398
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.03311414 * Q.n_dr_0p2_0p4 + 0.5960546
    if Q.girth2_top15 < 0.005383629:
        z += -31.11725 * Q.girth2_top15 + 0.3732105
    if 0.005383629 <= Q.girth2_top15 < 0.007887677:
        z += -326.247 * Q.girth2_top15 + 1.96208
    if 0.007887677 <= Q.girth2_top15 < 0.009962397:
        z += 294.6188 * Q.girth2_top15 - 2.935109
    if Q.tau1 < 0.04466492:
        z += 96.2639 * Q.tau1 - 4.299619
    if Q.girth < 0.05048381:
        z += -122.3657 * Q.girth + 10.24488
    if 0.05048381 <= Q.girth < 0.08068193:
        z += -157.8974 * Q.girth + 12.03865
    if 0.08068193 <= Q.girth < 0.09749958:
        z += 41.67136 * Q.girth - 4.06294
    if Q.sj3_dr_max < 0.1210264:
        z += 1.2868 * Q.sj3_dr_max - 0.2307425
    if 0.1210264 <= Q.sj3_dr_max < 0.1623049:
        z += 6.229637 * Q.sj3_dr_max - 0.8289565
    if 0.1623049 <= Q.sj3_dr_max < 0.2125209:
        z += -3.627207 * Q.sj3_dr_max + 0.7708573
    if 6.893714 <= Q.log_sum_pt < 7.017258:
        z += -7.446366 * Q.log_sum_pt + 51.33311
    if Q.log_sum_pt >= 7.017258:
        z += -3.895039 * Q.log_sum_pt + 26.41254
    if Q.sum_pt_top2 >= 405.0:
        z += 0.0009441872 * Q.sum_pt_top2 - 0.3823958
    if Q.sj2_dr < 0.06289464:
        z += 6.291028 * Q.sj2_dr - 1.148143
    if 0.06289464 <= Q.sj2_dr < 0.1411617:
        z += 1.698079 * Q.sj2_dr - 0.8592712
    if 0.1411617 <= Q.sj2_dr < 0.1512157:
        z += 15.98743 * Q.sj2_dr - 2.87638
    if 0.1512157 <= Q.sj2_dr < 0.1745007:
        z += -15.251 * Q.sj2_dr + 1.847359
    if 0.1745007 <= Q.sj2_dr < 0.1825048:
        z += 3.456312 * Q.sj2_dr - 1.417079
    if Q.sj2_dr >= 0.1825048:
        z += -2.834717 * Q.sj2_dr - 0.2689359
    if Q.z_dr_0_0p05 < 0.4947602:
        z += -0.7187094 * Q.z_dr_0_0p05 + 0.3555888
    if Q.dr_0 < 0.05775119:
        z += 4.177954 * Q.dr_0 - 0.4225564
    if 0.05775119 <= Q.dr_0 < 0.06413297:
        z += 28.40503 * Q.dr_0 - 1.821699
    if Q.LHA < 0.1870291:
        z += 12.15234 * Q.LHA - 5.718418
    if 0.1870291 <= Q.LHA < 0.3098384:
        z += 32.93923 * Q.LHA - 9.606171
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -37.02503 * Q.LHA + 12.07144
    if 0.3203321 <= Q.LHA < 0.3332345:
        z += -16.36399 * Q.LHA + 5.453048
    if Q.e3 < 2.883342e-05:
        z += 25128.53 * Q.e3 - 0.5889928
    if 2.883342e-05 <= Q.e3 < 5.13841e-05:
        z += -6010.847 * Q.e3 + 0.308862
    if Q.psi_0p3 >= 0.9896594:
        z += 46.32267 * Q.psi_0p3 - 45.84367
    if Q.sum_pt_top30 >= 1052.08:
        z += -0.001048096 * Q.sum_pt_top30 + 1.102681
    if Q.n_dr_0p2_0p4 < 18.0 and Q.girth2_top10 > 0.004752876:
        z += -3.74807 * (18.0 - Q.n_dr_0p2_0p4) * (Q.girth2_top10 - 0.004752876)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.tau21_b2 > 0.2018786:
        z += -0.04054947 * (18.0 - Q.n_dr_0p2_0p4) * (Q.tau21_b2 - 0.2018786)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt > 6.920349:
        z += -0.903147 * (18.0 - Q.n_dr_0p2_0p4) * (Q.log_sum_pt - 6.920349)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.sum_pt_top50 > 889.8503:
        z += 0.0005291339 * (18.0 - Q.n_dr_0p2_0p4) * (Q.sum_pt_top50 - 889.8503)
    if Q.psi_0p3 > 0.9956185 and Q.n_dr_0p1_0p2 < 26.0:
        z += 2.188731 * (Q.psi_0p3 - 0.9956185) * (26.0 - Q.n_dr_0p1_0p2)
    if Q.sum_pt_top40 > 1001.523 and Q.sj2_dr < 0.1512157:
        z += 0.04276993 * (Q.sum_pt_top40 - 1001.523) * (0.1512157 - Q.sj2_dr)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.4971845
    if Q.n_particles >= 38.0:
        z += 0.07214988 * Q.n_particles - 2.741695
    if Q.log_sum_pt < 6.910131:
        z += 6.790383 * Q.log_sum_pt - 48.47855
    if 6.910131 <= Q.log_sum_pt < 6.98945:
        z += 39.01924 * Q.log_sum_pt - 271.1842
    if 6.98945 <= Q.log_sum_pt < 7.062574:
        z += 24.42839 * Q.log_sum_pt - 169.2022
    if 7.062574 <= Q.log_sum_pt < 7.139296:
        z += 19.83708 * Q.log_sum_pt - 136.7757
    if Q.log_sum_pt >= 7.139296:
        z += 13.0467 * Q.log_sum_pt - 88.29715
    if Q.sum_pt_top50 < 934.2416:
        z += -0.009909895 * Q.sum_pt_top50 + 10.69272
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.02083582 * Q.sum_pt_top50 + 20.90017
    if 959.0957 <= Q.sum_pt_top50 < 1078.994:
        z += -0.02815045 * Q.sum_pt_top50 + 27.9156
    if Q.sum_pt_top50 >= 1078.994:
        z += -0.01824055 * Q.sum_pt_top50 + 17.22288
    if Q.girth2_top15 < 0.002197765:
        z += -192.4234 * Q.girth2_top15 + 0.4229014
    if Q.psi_0p3 >= 0.9966167:
        z += -126.3608 * Q.psi_0p3 + 125.9333
    if Q.tau21 < 0.2140276:
        z += -3.806082 * Q.tau21 + 0.8146064
    if Q.C2 < 0.05602756:
        z += -8.811555 * Q.C2 + 0.4936899
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.03179167 * Q.n_dr_0p2_0p4 - 0.2225417
    if Q.pt_11 < 29.04688:
        z += 0.01698926 * Q.pt_11 - 0.493485
    if 972.0419 <= Q.sum_pt < 1052.889:
        z += 0.02527572 * Q.sum_pt - 24.56906
    if Q.sum_pt >= 1052.889:
        z += 0.01010555 * Q.sum_pt - 8.596551
    if Q.girth2_top5 < 0.0006570502:
        z += -595.1931 * Q.girth2_top5 + 0.3910717
    if Q.sum_pt_top2 < 605.875:
        z += -0.00180348 * Q.sum_pt_top2 + 1.092683
    if Q.z_top30_slots >= 0.9341838:
        z += 10.11581 * Q.z_top30_slots - 9.450023
    if 15.0 <= Q.n_dr_0_0p05 < 30.0:
        z += 0.03163575 * Q.n_dr_0_0p05 - 0.4745363
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.05051325 * Q.n_dr_0_0p05 + 1.989934
    if Q.n_dr_0p1_0p2 < 7.0:
        z += 0.07324111 * Q.n_dr_0p1_0p2 - 0.5126878
    if Q.tau1 < 0.07708632:
        z += -28.26192 * Q.tau1 + 2.178607
    if Q.tau1 >= 0.1219132:
        z += -19.93213 * Q.tau1 + 2.42999
    if Q.lam2 < 0.001776308:
        z += 263.1464 * Q.lam2 - 0.4674292
    if Q.sum_pt_top40 < 1225.842:
        z += -0.003502262 * Q.sum_pt_top40 + 4.293221
    if Q.zdr_0 >= 0.01113024:
        z += -31.61772 * Q.zdr_0 + 0.3519127
    if Q.tau4 < 0.02178815:
        z += 47.23372 * Q.tau4 - 1.029135
    if Q.N2 < 0.3572263:
        z += -1.1014 * Q.N2 + 0.3934491
    if Q.D2_b2 < 0.8495689:
        z += 0.4676556 * Q.D2_b2 - 0.3973056
    if Q.e2 >= 0.05557149:
        z += 22.78098 * Q.e2 - 1.265973
    if Q.LHA >= 0.3332345:
        z += 12.05269 * Q.LHA - 4.016373
    if Q.z_top50_slots < 1.0:
        z += -6.754965 * Q.z_top50_slots + 6.754965
    if Q.girth2_top3 < 0.006756161:
        z += -83.6206 * Q.girth2_top3 + 0.5649542
    if Q.z_top15_slots >= 0.9885666:
        z += 60.87476 * Q.z_top15_slots - 60.17876
    if Q.sum_pt_top30 >= 1038.262:
        z += 0.002486429 * Q.sum_pt_top30 - 2.581564
    if Q.soft1_pt < 1.521582:
        z += -0.2427919 * Q.soft1_pt + 0.3694278
    if Q.n_particles > 38.0 and Q.zdr_0 < 0.009970338:
        z += 2.224471 * (Q.n_particles - 38.0) * (0.009970338 - Q.zdr_0)
    if Q.n_particles > 38.0 and Q.soft1_z < 0.002181998:
        z += -24.85268 * (Q.n_particles - 38.0) * (0.002181998 - Q.soft1_z)
    if Q.z_top5 > 0.6551948 and Q.pt1_dr01 < 28.39396:
        z += -0.1663284 * (Q.z_top5 - 0.6551948) * (28.39396 - Q.pt1_dr01)
    if Q.girth2_top15 < 0.002197765 and Q.psi_0p3 > 0.9966167:
        z += 51250.71 * (0.002197765 - Q.girth2_top15) * (Q.psi_0p3 - 0.9966167)
    if Q.n_particles > 38.0 and Q.absphi_1 < 0.1178619:
        z += 0.09272996 * (Q.n_particles - 38.0) * (0.1178619 - Q.absphi_1)
    if Q.z_top30_slots > 0.9341838 and Q.pt1_dr01 > 5.351077:
        z += 0.3759178 * (Q.z_top30_slots - 0.9341838) * (Q.pt1_dr01 - 5.351077)
    if Q.n_particles > 38.0 and Q.tau32 > 0.3293142:
        z += 0.01991292 * (Q.n_particles - 38.0) * (Q.tau32 - 0.3293142)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_2 > 7.740999:
        z += 0.3860166 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_2 - 7.740999)
    if Q.M3 < 0.03457336 and Q.psi_0p3 > 0.9299135:
        z += -311.3179 * (0.03457336 - Q.M3) * (Q.psi_0p3 - 0.9299135)
    if Q.z_dr_0_0p05 > 0.8103116 and Q.n_dr_0p05_0p1 < 10.0:
        z += -0.5527883 * (Q.z_dr_0_0p05 - 0.8103116) * (10.0 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_4 > 2.381691:
        z += 0.4475536 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_4 - 2.381691)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 < 17.94219:
        z += -0.5478213 * (Q.z_top30_slots - 0.9341838) * (17.94219 - Q.ptdr0_3)
    if Q.z_top30_slots > 0.9341838 and Q.dr12 > 0.003946546:
        z += 29.04288 * (Q.z_top30_slots - 0.9341838) * (Q.dr12 - 0.003946546)
    if Q.z_top5 > 0.6551948 and Q.ptdr0_5 > 0.5368514:
        z += 0.4693474 * (Q.z_top5 - 0.6551948) * (Q.ptdr0_5 - 0.5368514)
    if Q.N2 < 0.3572263 and Q.pt2_over_pt0 < 0.7904923:
        z += 3.447278 * (0.3572263 - Q.N2) * (0.7904923 - Q.pt2_over_pt0)
    if Q.soft9_pt < 2.5 and Q.dr_11 < 0.1224381:
        z += 1.273761 * (2.5 - Q.soft9_pt) * (0.1224381 - Q.dr_11)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_6 < 11.6057:
        z += -0.2610714 * (Q.z_top30_slots - 0.9341838) * (11.6057 - Q.ptdr0_6)
    if Q.pt1_dr01 < 12.6865 and Q.soft2_dr < 0.3845054:
        z += -0.1463273 * (12.6865 - Q.pt1_dr01) * (0.3845054 - Q.soft2_dr)
    if Q.pt1_dr01 < 12.6865 and Q.soft2_dr0 < 0.4086381:
        z += 0.1136669 * (12.6865 - Q.pt1_dr01) * (0.4086381 - Q.soft2_dr0)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.6396009
    if 6.903423 <= Q.log_sum_pt < 6.930088:
        z += 5.585062 * Q.log_sum_pt - 38.55604
    if 6.930088 <= Q.log_sum_pt < 7.062574:
        z += 16.41487 * Q.log_sum_pt - 113.6076
    if Q.log_sum_pt >= 7.062574:
        z += 10.96798 * Q.log_sum_pt - 75.13853
    if Q.girth < 0.03577037:
        z += 57.77464 * Q.girth - 2.06662
    if 0.1596365 <= Q.sd_rg < 0.1690338:
        z += 26.2978 * Q.sd_rg - 4.198088
    if 0.1690338 <= Q.sd_rg < 0.2639816:
        z += 18.18627 * Q.sd_rg - 2.826966
    if Q.sd_rg >= 0.2639816:
        z += -42.94203 * Q.sd_rg + 13.30978
    if 1064.139 <= Q.sum_pt_top20 < 1129.275:
        z += 0.0033622 * Q.sum_pt_top20 - 3.577848
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.006526796 * Q.sum_pt_top20 - 7.151547
    if Q.sum_pt_top50 >= 997.0189:
        z += -0.005458712 * Q.sum_pt_top50 + 5.442439
    if Q.LHA < 0.1870291:
        z += -10.21788 * Q.LHA + 1.91104
    if Q.sum_pt_top15 >= 1082.548:
        z += -0.003051353 * Q.sum_pt_top15 + 3.303235
    if Q.psi_0p3 >= 0.9966167:
        z += 67.11485 * Q.psi_0p3 - 66.88778
    if Q.sum_pt_top30 >= 996.8867:
        z += -0.00284198 * Q.sum_pt_top30 + 2.833132
    if 1018.698 <= Q.sum_pt_top40 < 1069.671:
        z += -0.004540255 * Q.sum_pt_top40 + 4.625147
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.003628904 * Q.sum_pt_top40 - 4.113166
    if Q.sum_pt >= 1052.889:
        z += -0.01054043 * Q.sum_pt + 11.09791
    if Q.z_top50_slots >= 0.9586536:
        z += -16.4502 * Q.z_top50_slots + 15.77005
    if Q.psi_0p2 >= 0.9734513:
        z += -21.67486 * Q.psi_0p2 + 21.09942
    if Q.z_top40_slots >= 0.9300465:
        z += 5.103651 * Q.z_top40_slots - 4.746632
    if Q.z_top15_slots >= 0.6225177:
        z += 1.603802 * Q.z_top15_slots - 0.9983951
    if Q.girth2_top10 < 0.01414829:
        z += -39.83928 * Q.girth2_top10 + 0.5636576
    if Q.girth2_top15 < 0.006615185:
        z += -55.75201 * Q.girth2_top15 + 0.3688098
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 229.7699 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 6.903423 and Q.tau21 < 0.8065577:
        z += 3.364948 * (Q.log_sum_pt - 6.903423) * (0.8065577 - Q.tau21)
    if Q.log_sum_pt > 7.062574 and Q.girth2_top15 > 0.009962397:
        z += -7157.536 * (Q.log_sum_pt - 7.062574) * (Q.girth2_top15 - 0.009962397)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p1 < 0.8747961:
        z += -10.69982 * (Q.log_sum_pt - 6.903423) * (0.8747961 - Q.psi_0p1)
    if Q.sum_pt_top50 > 997.0189 and Q.zdr_4 < 0.006794973:
        z += 0.217631 * (Q.sum_pt_top50 - 997.0189) * (0.006794973 - Q.zdr_4)
    if Q.log_sum_pt > 6.903423 and Q.z_dr_0p1_0p2 > 0.003353111:
        z += 3.471969 * (Q.log_sum_pt - 6.903423) * (Q.z_dr_0p1_0p2 - 0.003353111)
    if Q.sum_pt_top20 > 1129.275 and Q.dr_max_012 > 0.1206357:
        z += -0.5269529 * (Q.sum_pt_top20 - 1129.275) * (Q.dr_max_012 - 0.1206357)
    if Q.log_sum_pt > 6.903423 and Q.dr_max_012 < 0.1206357:
        z += -20.5932 * (Q.log_sum_pt - 6.903423) * (0.1206357 - Q.dr_max_012)
    if Q.log_sum_pt > 6.903423 and Q.zdr_3 > 0.00453462:
        z += -424.5455 * (Q.log_sum_pt - 6.903423) * (Q.zdr_3 - 0.00453462)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -0.7841412 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    if Q.z_top20_slots > 0.8281581 and Q.z_top50_slots > 0.9995789:
        z += -2951.214 * (Q.z_top20_slots - 0.8281581) * (Q.z_top50_slots - 0.9995789)
    if Q.log_sum_pt > 6.930088 and Q.C2 > 0.06655881:
        z += 185.2813 * (Q.log_sum_pt - 6.930088) * (Q.C2 - 0.06655881)
    if Q.sd_rg > 0.1596365 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += -10.76036 * (Q.sd_rg - 0.1596365) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.sum_pt > 1052.889 and Q.z_dr_0p05_0p1 < 0.5106729:
        z += -0.001394513 * (Q.sum_pt - 1052.889) * (0.5106729 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top50 > 997.0189 and Q.C2 > 0.06655881:
        z += -0.2205183 * (Q.sum_pt_top50 - 997.0189) * (Q.C2 - 0.06655881)
    if Q.log_sum_pt > 6.930088 and Q.C2 > 0.1088881:
        z += -643.3644 * (Q.log_sum_pt - 6.930088) * (Q.C2 - 0.1088881)
    if Q.sum_pt_top15 > 1082.548 and Q.C2 > 0.1423914:
        z += -18.42842 * (Q.sum_pt_top15 - 1082.548) * (Q.C2 - 0.1423914)
    if Q.sum_pt_top50 > 997.0189 and Q.C2 > 0.1088881:
        z += 0.6268058 * (Q.sum_pt_top50 - 997.0189) * (Q.C2 - 0.1088881)
    if Q.log_sum_pt > 6.903423 and Q.dr_max_012 < 0.2002199:
        z += 25.50345 * (Q.log_sum_pt - 6.903423) * (0.2002199 - Q.dr_max_012)
    if Q.sd_rg > 0.2639816 and Q.z_dr_0p05_0p1 < 0.5912328:
        z += 120.1857 * (Q.sd_rg - 0.2639816) * (0.5912328 - Q.z_dr_0p05_0p1)
    if Q.sd_rg > 0.1690338 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += -16.07715 * (Q.sd_rg - 0.1690338) * (0.8509215 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.408896
    if Q.lam2 < 0.0006154841:
        z += -943.0285 * Q.lam2 + 0.580419
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.2738755 * Q.n_dr_0p2_0p4 + 1.572894
    if 5.0 <= Q.n_dr_0p2_0p4 < 8.0:
        z += -0.06783873 * Q.n_dr_0p2_0p4 + 0.5427098
    if Q.n_particles < 46.0:
        z += -0.05280203 * Q.n_particles + 2.428893
    if Q.tau21 < 0.347196:
        z += 2.028941 * Q.tau21 - 0.7044403
    if Q.psi_0p2 >= 0.9985434:
        z += 95.1594 * Q.psi_0p2 - 95.02079
    if Q.n_dr_0p1_0p2 < 17.0:
        z += -0.03602159 * Q.n_dr_0p1_0p2 + 0.612367
    if Q.n_dr_0p2_0p4 < 5.0 and Q.e2 < 0.03029714:
        z += -7.765626 * (5.0 - Q.n_dr_0p2_0p4) * (0.03029714 - Q.e2)
    if Q.n_particles < 46.0 and Q.e2 > 0.01036127:
        z += -1.432932 * (46.0 - Q.n_particles) * (Q.e2 - 0.01036127)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.n_dr_0p05_0p1 > 2.0:
        z += -0.002675933 * (8.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.n_dr_0p1_0p2 < 10.0 and Q.sum_pt_top40 < 1013.042:
        z += -0.0004739462 * (10.0 - Q.n_dr_0p1_0p2) * (1013.042 - Q.sum_pt_top40)
    if Q.psi_0p3 > 0.9980008 and Q.eccentricity > 0.8680812:
        z += 1823.463 * (Q.psi_0p3 - 0.9980008) * (Q.eccentricity - 0.8680812)
    if Q.n_particles < 46.0 and Q.psi_0p3 > 0.9777125:
        z += -0.8271448 * (46.0 - Q.n_particles) * (Q.psi_0p3 - 0.9777125)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.857
    if Q.e2 < 0.01879315:
        z += 30.30027 * Q.e2 - 0.8442498
    if 0.01879315 <= Q.e2 < 0.03263075:
        z += -0.6064739 * Q.e2 - 0.2634148
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 25.85052 * Q.e2 - 1.126727
    if Q.e2 >= 0.05557149:
        z += -38.60912 * Q.e2 + 2.145566
    if Q.psi_0p1 >= 0.8976117:
        z += -3.647719 * Q.psi_0p1 + 3.274235
    if Q.n_particles >= 26.0:
        z += -0.01025955 * Q.n_particles + 0.2667482
    if Q.n_dr_0_0p05 >= 14.0:
        z += 0.01712691 * Q.n_dr_0_0p05 - 0.2397767
    if Q.girth2_top15 < 0.004169954:
        z += 13.10305 * Q.girth2_top15 + 0.2303109
    if 0.004169954 <= Q.girth2_top15 < 0.005788041:
        z += 106.7824 * Q.girth2_top15 - 0.1603277
    if 0.005788041 <= Q.girth2_top15 < 0.007887677:
        z += -25.1151 * Q.girth2_top15 + 0.6031005
    if 0.007887677 <= Q.girth2_top15 < 0.009962397:
        z += -195.2074 * Q.girth2_top15 + 1.944734
    if Q.pt_entropy < 3.355186:
        z += 0.3395498 * Q.pt_entropy - 1.139253
    if Q.girth < 0.05660088:
        z += 38.59454 * Q.girth - 2.818615
    if 0.05660088 <= Q.girth < 0.07374472:
        z += 24.54017 * Q.girth - 2.023125
    if 0.07374472 <= Q.girth < 0.09749958:
        z += 8.984136 * Q.girth - 0.8759495
    if Q.girth >= 0.09749958:
        z += 62.44687 * Q.girth - 6.088544
    if Q.girth2_top10 < 0.006876086:
        z += 76.4026 * Q.girth2_top10 - 0.5253508
    if Q.sum_pt_top30 < 1011.524:
        z += -0.001585434 * Q.sum_pt_top30 + 1.603705
    if 1.091797 <= Q.soft1_pt < 1.521582:
        z += 0.5132735 * Q.soft1_pt - 0.5603904
    if 1.521582 <= Q.soft1_pt < 2.275391:
        z += -0.7661237 * Q.soft1_pt + 1.386317
    if Q.soft1_pt >= 2.275391:
        z += 0.3905427 * Q.soft1_pt - 1.245551
    if 0.1219342 <= Q.sj2_dr < 0.1411617:
        z += 5.804647 * Q.sj2_dr - 0.7077852
    if 0.1411617 <= Q.sj2_dr < 0.1825048:
        z += 11.27406 * Q.sj2_dr - 1.479856
    if Q.sj2_dr >= 0.1825048:
        z += -2.387912 * Q.sj2_dr + 1.013519
    if Q.z_dr_0p2_0p4 < 0.05180474:
        z += 9.108783 * Q.z_dr_0p2_0p4 - 0.4718781
    if Q.LHA < 0.2454112:
        z += -8.487327 * Q.LHA + 2.082885
    if Q.LHA >= 0.3332345:
        z += -36.49057 * Q.LHA + 12.15992
    if Q.psi_0p2 >= 0.8706159:
        z += 1.586621 * Q.psi_0p2 - 1.381338
    if Q.soft9_pt >= 1.458984:
        z += 0.05716659 * Q.soft9_pt - 0.08340516
    if Q.sum_pt_top50 < 1038.855:
        z += 0.009520341 * Q.sum_pt_top50 - 9.890254
    if Q.n_pt_above_1 >= 32.0:
        z += -0.01101252 * Q.n_pt_above_1 + 0.3524006
    if Q.n_particles > 26.0 and Q.soft1_pt > 0.4909668:
        z += -0.006249136 * (Q.n_particles - 26.0) * (Q.soft1_pt - 0.4909668)
    if Q.n_particles > 26.0 and Q.lam2 > 0.003687605:
        z += 1.974164 * (Q.n_particles - 26.0) * (Q.lam2 - 0.003687605)
    if Q.girth2_top15 < 0.007887677 and Q.sum_pt_top40 > 1095.686:
        z += 0.21649 * (0.007887677 - Q.girth2_top15) * (Q.sum_pt_top40 - 1095.686)
    if Q.girth < 0.09749958 and Q.psi_0p3 > 0.9973959:
        z += 6529.695 * (0.09749958 - Q.girth) * (Q.psi_0p3 - 0.9973959)
    if Q.girth < 0.05660088 and Q.psi_0p3 > 0.9973959:
        z += -16442.18 * (0.05660088 - Q.girth) * (Q.psi_0p3 - 0.9973959)
    if Q.girth2_top15 < 0.004169954 and Q.psi_0p3 > 0.9973959:
        z += -30217.05 * (0.004169954 - Q.girth2_top15) * (Q.psi_0p3 - 0.9973959)
    if Q.z_dr_0p2_0p4 < 0.05180474 and Q.soft9_z > 0.0007402181:
        z += -1532.006 * (0.05180474 - Q.z_dr_0p2_0p4) * (Q.soft9_z - 0.0007402181)
    if Q.girth2_top15 > 0.004855289 and Q.sj3_pairmin_over_m < 0.4090302:
        z += -119.8433 * (Q.girth2_top15 - 0.004855289) * (0.4090302 - Q.sj3_pairmin_over_m)
    return max(0.0, z)


def neuron_5(Q):
    z = -4.468653
    z += -0.05327229 * Q.n_particles + 3.409426
    if Q.sum_pt < 907.9372:
        z += 0.07826509 * Q.sum_pt - 71.51354
    if 907.9372 <= Q.sum_pt < 1002.379:
        z += 0.009608506 * Q.sum_pt - 9.177676
    if 1002.379 <= Q.sum_pt < 1085.125:
        z += -0.005482828 * Q.sum_pt + 5.949553
    if Q.sum_pt_top50 < 934.2416:
        z += -0.02271752 * Q.sum_pt_top50 + 21.39848
    if 934.2416 <= Q.sum_pt_top50 < 959.0957:
        z += -0.007034345 * Q.sum_pt_top50 + 6.74661
    if Q.z_top30_slots >= 0.9460751:
        z += -8.822384 * Q.z_top30_slots + 8.346638
    z += 0.7683562 * Q.z_top3_slots
    if 967.7705 <= Q.sum_pt_top15 < 1003.329:
        z += -0.003508167 * Q.sum_pt_top15 + 3.395101
    if Q.sum_pt_top15 >= 1003.329:
        z += 0.002405587 * Q.sum_pt_top15 - 2.538341
    if Q.girth < 0.02783745:
        z += -27.74192 * Q.girth + 5.489991
    if 0.02783745 <= Q.girth < 0.08068193:
        z += 17.58112 * Q.girth + 4.228313
    if 0.08068193 <= Q.girth < 0.1564779:
        z += 2.42207 * Q.girth + 5.451375
    if Q.girth >= 0.1564779:
        z += 45.32304 * Q.girth - 1.261678
    if 0.1632346 <= Q.LHA < 0.302389:
        z += -6.905437 * Q.LHA + 1.127207
    if 0.302389 <= Q.LHA < 0.4331369:
        z += -1.258818 * Q.LHA - 0.5802692
    if Q.LHA >= 0.4331369:
        z += -174.3245 * Q.LHA + 74.38085
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += 1.004309 * Q.z_dr_0_0p05 - 0.7159335
    if Q.tau1 < 0.1072713:
        z += 2.209399 * Q.tau1 - 0.2370049
    if Q.girth2_top15 < 0.0007894752:
        z += 788.5568 * Q.girth2_top15 - 0.6225461
    if Q.n_pt_above_1 < 58.0:
        z += 0.01851737 * Q.n_pt_above_1 - 1.074008
    if Q.C2 >= 0.1235569:
        z += -18.84553 * Q.C2 + 2.328496
    if Q.tau4 < 0.01626937:
        z += -31.9524 * Q.tau4 + 0.5198453
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.07892789 * Q.n_dr_0p2_0p4 + 0.8682068
    if Q.z_top50_slots < 0.985099:
        z += 38.38499 * Q.z_top50_slots - 37.81302
    if Q.z_dr_0p2_0p4 < 0.05180474:
        z += 7.242975 * Q.z_dr_0p2_0p4 - 0.3752204
    if Q.sum_pt_top40 < 1013.042:
        z += -0.007921687 * Q.sum_pt_top40 + 8.025006
    if 0.005337976 <= Q.girth2_top10 < 0.008956554:
        z += 37.84008 * Q.girth2_top10 - 0.2019895
    if Q.girth2_top10 >= 0.008956554:
        z += -138.305 * Q.girth2_top10 + 1.375663
    if Q.M2 >= 0.05233747:
        z += 5.883905 * Q.M2 - 0.3079487
    if Q.z_top20_slots >= 0.9358352:
        z += -4.075117 * Q.z_top20_slots + 3.813637
    if Q.n_dr_0p1_0p2 >= 8.0:
        z += -0.03591038 * Q.n_dr_0p1_0p2 + 0.2872831
    if Q.girth2_top5 < 0.0004005745:
        z += 1010.903 * Q.girth2_top5 - 0.4049421
    if Q.log_sum_pt < 6.893714:
        z += 17.19324 * Q.log_sum_pt - 118.505
    if 6.893714 <= Q.log_sum_pt < 6.903423:
        z += -2.088479 * Q.log_sum_pt + 14.41766
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.003272639 * Q.sum_pt_top10 + 3.088364
    if Q.n_particles < 64.0 and Q.C2 < 0.07996447:
        z += -0.2148625 * (64.0 - Q.n_particles) * (0.07996447 - Q.C2)
    if Q.girth < 0.1564779 and Q.n_dr_0p1_0p2 > 11.0:
        z += 0.7393562 * (0.1564779 - Q.girth) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.sum_pt < 1002.379 and Q.dr_11 < 0.199671:
        z += 0.03103275 * (1002.379 - Q.sum_pt) * (0.199671 - Q.dr_11)
    if Q.sum_pt < 1002.379 and Q.ptdr0_10 > 3.719859:
        z += -0.003262159 * (1002.379 - Q.sum_pt) * (Q.ptdr0_10 - 3.719859)
    if Q.sum_pt_top50 < 934.2416 and Q.soft6_dr < 0.03626613:
        z += 0.6876636 * (934.2416 - Q.sum_pt_top50) * (0.03626613 - Q.soft6_dr)
    if Q.sum_pt < 1002.379 and Q.sj3_dr23 > 0.1834565:
        z += 0.0009657272 * (1002.379 - Q.sum_pt) * (Q.sj3_dr23 - 0.1834565)
    if Q.sum_pt < 1002.379 and Q.absphi_1 > 0.02227783:
        z += -0.1584804 * (1002.379 - Q.sum_pt) * (Q.absphi_1 - 0.02227783)
    if Q.sum_pt < 907.9372 and Q.dr0_12 < 0.1569963:
        z += 0.2381256 * (907.9372 - Q.sum_pt) * (0.1569963 - Q.dr0_12)
    if Q.sum_pt < 1002.379 and Q.ptdr0_12 > 2.024972:
        z += -0.001928157 * (1002.379 - Q.sum_pt) * (Q.ptdr0_12 - 2.024972)
    if Q.sum_pt < 1002.379 and Q.e4 < 5.8505e-08:
        z += -386705.8 * (1002.379 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 1085.125 and Q.e4 < 5.8505e-08:
        z += 36369.68 * (1085.125 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.mean_eta < -0.003213499:
        z += -250010.6 * (Q.psi_0p3 - 0.9973959) * (-0.003213499 - Q.mean_eta)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.4194375
    if Q.tau1 < 0.05444509:
        z += -45.59307 * Q.tau1 + 2.482319
    if Q.psi_0p2 >= 0.9087063:
        z += -4.948503 * Q.psi_0p2 + 4.496735
    if 0.06655881 <= Q.C2 < 0.1088881:
        z += 4.998832 * Q.C2 - 0.3327163
    if Q.C2 >= 0.1088881:
        z += -11.51654 * Q.C2 + 1.465611
    if Q.girth2_top15 < 0.001319197:
        z += 291.8526 * Q.girth2_top15 - 0.5045269
    if 0.001319197 <= Q.girth2_top15 < 0.003270031:
        z += -463.8441 * Q.girth2_top15 + 0.4923859
    if 0.003270031 <= Q.girth2_top15 < 0.00727763:
        z += 224.063 * Q.girth2_top15 - 1.757092
    if 0.00727763 <= Q.girth2_top15 < 0.009962397:
        z += 47.0969 * Q.girth2_top15 - 0.469198
    if Q.girth < 0.05048381:
        z += 25.05719 * Q.girth - 6.983306
    if 0.05048381 <= Q.girth < 0.09749958:
        z += 121.6256 * Q.girth - 11.85845
    if Q.e3 < 5.13841e-05:
        z += -3034.157 * Q.e3 + 1.023221
    if 5.13841e-05 <= Q.e3 < 0.0001086251:
        z += 4493.033 * Q.e3 + 0.6364426
    if 0.0001086251 <= Q.e3 < 0.0003372339:
        z += -8483.062 * Q.e3 + 2.045972
    if Q.e3 >= 0.0003372339:
        z += -5448.906 * Q.e3 + 1.022751
    if Q.sj2_zsoft < 0.04575675:
        z += -14.74796 * Q.sj2_zsoft + 1.117869
    if 0.04575675 <= Q.sj2_zsoft < 0.3676068:
        z += -1.376575 * Q.sj2_zsoft + 0.5060383
    if Q.e2 < 0.04358622:
        z += -38.76652 * Q.e2 + 1.790728
    if 0.04358622 <= Q.e2 < 0.04755309:
        z += -25.4714 * Q.e2 + 1.211244
    if Q.e2 >= 0.05557149:
        z += 104.8491 * Q.e2 - 5.826618
    if Q.z_dr_0_0p05 >= 0.3289237:
        z += 1.104443 * Q.z_dr_0_0p05 - 0.3632775
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += -0.08558419 * Q.n_dr_0p2_0p4 + 1.283763
    if Q.girth2_top5 < 0.007164202:
        z += -59.81356 * Q.girth2_top5 + 0.4285164
    if Q.lam2 < 0.006427167:
        z += -195.701 * Q.lam2 + 1.257803
    if Q.LHA < 0.2284021:
        z += -9.045515 * Q.LHA + 4.17523
    if 0.2284021 <= Q.LHA < 0.302389:
        z += -17.3858 * Q.LHA + 6.080168
    if 0.302389 <= Q.LHA < 0.3332345:
        z += -26.6779 * Q.LHA + 8.889997
    if Q.sum_pt_top50 >= 1078.994:
        z += 0.00274758 * Q.sum_pt_top50 - 2.964623
    if Q.sum_pt >= 907.9372:
        z += 0.001310302 * Q.sum_pt - 1.189672
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += -1.651911 * Q.z_dr_0p1_0p2 + 0.5518172
    if Q.z_dr_0p05_0p1 < 0.7108211:
        z += 0.3373736 * Q.z_dr_0p05_0p1 - 0.2398123
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += 7.596691 * Q.z_dr_0p2_0p4 - 0.5203088
    if Q.z_dr_0p2_0p4 >= 0.2708738:
        z += -20.64331 * Q.z_dr_0p2_0p4 + 5.591731
    if Q.sj2_dr < 0.1825048:
        z += 7.954796 * Q.sj2_dr - 1.451789
    if Q.n_dr_0p1_0p2 > 15.0 and Q.lam2 < 0.006427167:
        z += -9.279568 * (Q.n_dr_0p1_0p2 - 15.0) * (0.006427167 - Q.lam2)
    if Q.psi_0p2 > 0.9087063 and Q.sj2_dr < 0.1512157:
        z += 202.1091 * (Q.psi_0p2 - 0.9087063) * (0.1512157 - Q.sj2_dr)
    if Q.e3 < 0.0003372339 and Q.log_sum_pt < 7.017258:
        z += -27321.69 * (0.0003372339 - Q.e3) * (7.017258 - Q.log_sum_pt)
    if Q.tau1 < 0.05444509 and Q.sum_pt < 972.0419:
        z += 0.777894 * (0.05444509 - Q.tau1) * (972.0419 - Q.sum_pt)
    if Q.n_dr_0p1_0p2 > 26.0 and Q.sum_pt > 986.0565:
        z += -0.0002374834 * (Q.n_dr_0p1_0p2 - 26.0) * (Q.sum_pt - 986.0565)
    if Q.n_dr_0p1_0p2 > 15.0 and Q.psi_0p3 > 0.9853273:
        z += 7.282552 * (Q.n_dr_0p1_0p2 - 15.0) * (Q.psi_0p3 - 0.9853273)
    if Q.girth < 0.08589404 and Q.M2 < 0.1134943:
        z += -80.68134 * (0.08589404 - Q.girth) * (0.1134943 - Q.M2)
    if Q.sj2_zsoft < 0.3676068 and Q.zdr_1 < 0.008824206:
        z += -122.8127 * (0.3676068 - Q.sj2_zsoft) * (0.008824206 - Q.zdr_1)
    if Q.girth2_top15 < 0.003270031 and Q.sum_pt_top40 > 858.8262:
        z += -0.9462346 * (0.003270031 - Q.girth2_top15) * (Q.sum_pt_top40 - 858.8262)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.eta_0 < 0.07952881:
        z += 0.2058384 * (Q.n_dr_0p2_0p4 - 15.0) * (0.07952881 - Q.eta_0)
    if Q.girth < 0.09749958 and Q.psi_0p3 < 0.9638082:
        z += 8327.804 * (0.09749958 - Q.girth) * (0.9638082 - Q.psi_0p3)
    if Q.e2 < 0.04358622 and Q.psi_0p3 < 0.9638082:
        z += -1525.029 * (0.04358622 - Q.e2) * (0.9638082 - Q.psi_0p3)
    if Q.LHA < 0.3332345 and Q.psi_0p3 < 0.9638082:
        z += -2781.931 * (0.3332345 - Q.LHA) * (0.9638082 - Q.psi_0p3)
    if Q.e3 > 5.13841e-05 and Q.orientation_deg > 26.6454:
        z += -29.83442 * (Q.e3 - 5.13841e-05) * (Q.orientation_deg - 26.6454)
    if Q.n_dr_0p1_0p2 > 26.0 and Q.ptdr0_10 > 3.023504:
        z += -0.007806791 * (Q.n_dr_0p1_0p2 - 26.0) * (Q.ptdr0_10 - 3.023504)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.3106651
    if Q.tau21_b2 < 0.2352054:
        z += 5.843542 * Q.tau21_b2 - 1.374433
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.03634508 * Q.n_dr_0p2_0p4 - 0.008880839
    if 15.0 <= Q.n_dr_0p2_0p4 < 21.0:
        z += -0.08938256 * Q.n_dr_0p2_0p4 + 1.877034
    if Q.psi_0p3 >= 0.9973959:
        z += 2302.221 * Q.psi_0p3 - 2296.225
    if Q.e3 < 3.376709e-05:
        z += 4878.556 * Q.e3 + 0.1816801
    if 3.376709e-05 <= Q.e3 < 6.567534e-05:
        z += -10856.59 * Q.e3 + 0.7130102
    if Q.psi_0p1 >= 0.8509811:
        z += -4.447011 * Q.psi_0p1 + 3.784322
    if Q.e2 < 0.03263075:
        z += -35.92111 * Q.e2 + 1.644714
    if 0.03263075 <= Q.e2 < 0.03680582:
        z += -47.55704 * Q.e2 + 2.024403
    if 0.03680582 <= Q.e2 < 0.05557149:
        z += -14.60261 * Q.e2 + 0.8114886
    if Q.tau1 < 0.1072713:
        z += 59.48753 * Q.tau1 - 6.198932
    if 0.1072713 <= Q.tau1 < 0.1219132:
        z += -12.45526 * Q.tau1 + 1.518461
    if Q.LHA < 0.302389:
        z += -18.53925 * Q.LHA + 5.227002
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 12.28906 * Q.LHA - 4.095138
    if Q.girth < 0.04362872:
        z += 42.39458 * Q.girth - 1.726261
    if 0.04362872 <= Q.girth < 0.08068193:
        z += 21.52373 * Q.girth - 0.8156918
    if 0.08068193 <= Q.girth < 0.09749958:
        z += -54.757 * Q.girth + 5.338784
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1937688:
        z += -145.1108 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1937688)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.007887677:
        z += -6173.048 * (0.2352054 - Q.tau21_b2) * (0.007887677 - Q.girth2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.girth2_top15 < 0.009962397:
        z += 6181.221 * (0.2352054 - Q.tau21_b2) * (0.009962397 - Q.girth2_top15)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt_top50 < 1245.697:
        z += -0.08101825 * (0.2352054 - Q.tau21_b2) * (1245.697 - Q.sum_pt_top50)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.e3 < 7.876005e-05:
        z += -1187.607 * (21.0 - Q.n_dr_0p2_0p4) * (7.876005e-05 - Q.e3)
    if Q.psi_0p3 > 0.9973959 and Q.mean_eta2 < 0.006802603:
        z += -266082.2 * (Q.psi_0p3 - 0.9973959) * (0.006802603 - Q.mean_eta2)
    if Q.psi_0p3 > 0.9973959 and Q.log_sum_pt < 7.062574:
        z += -4881.388 * (Q.psi_0p3 - 0.9973959) * (7.062574 - Q.log_sum_pt)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 972.0419:
        z += -0.08916968 * (0.2352054 - Q.tau21_b2) * (972.0419 - Q.sum_pt)
    if Q.psi_0p3 > 0.9973959 and Q.mean_phi2 < 0.006808102:
        z += -249629.5 * (Q.psi_0p3 - 0.9973959) * (0.006808102 - Q.mean_phi2)
    if Q.psi_0p3 > 0.9973959 and Q.girth2_top10 > 0.007678544:
        z += -206800.9 * (Q.psi_0p3 - 0.9973959) * (Q.girth2_top10 - 0.007678544)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1115.723:
        z += 0.06658677 * (0.2352054 - Q.tau21_b2) * (1115.723 - Q.sum_pt)
    if Q.tau1 < 0.1072713 and Q.z_dr_0p1_0p2 < 0.2864926:
        z += 152.5416 * (0.1072713 - Q.tau1) * (0.2864926 - Q.z_dr_0p1_0p2)
    if Q.LHA < 0.3332345 and Q.z_dr_0p1_0p2 < 0.250441:
        z += -52.72062 * (0.3332345 - Q.LHA) * (0.250441 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.M2 < 0.1134943:
        z += 0.9494622 * (15.0 - Q.n_dr_0p2_0p4) * (0.1134943 - Q.M2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr < 0.4357228:
        z += 0.08049732 * (15.0 - Q.n_dr_0p2_0p4) * (0.4357228 - Q.max_dr)
    if Q.psi_0p3 > 0.9973959 and Q.lam2 < 0.003687605:
        z += 30628.74 * (Q.psi_0p3 - 0.9973959) * (0.003687605 - Q.lam2)
    if Q.tau21_b2 < 0.2352054 and Q.n_dr_0p05_0p1 > 2.0:
        z += 0.1412402 * (0.2352054 - Q.tau21_b2) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.tau21_b2 < 0.2352054 and Q.sj2_dr > 0.1512157:
        z += 215.1128 * (0.2352054 - Q.tau21_b2) * (Q.sj2_dr - 0.1512157)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.psi_0p3 > 0.9956185:
        z += 9.977366 * (15.0 - Q.n_dr_0p2_0p4) * (Q.psi_0p3 - 0.9956185)
    if Q.e2 < 0.04755309 and Q.tau21_b2 < 0.2352054:
        z += -377.5575 * (0.04755309 - Q.e2) * (0.2352054 - Q.tau21_b2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.tau21_b2 < 0.4823776:
        z += 0.03023907 * (21.0 - Q.n_dr_0p2_0p4) * (0.4823776 - Q.tau21_b2)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.188077
    if 0.03577037 <= Q.girth < 0.04362872:
        z += 79.62509 * Q.girth - 2.848219
    if 0.04362872 <= Q.girth < 0.07031778:
        z += 183.0096 * Q.girth - 7.358751
    if 0.07031778 <= Q.girth < 0.07374472:
        z += 159.6359 * Q.girth - 5.71517
    if 0.07374472 <= Q.girth < 0.09749958:
        z += 100.7376 * Q.girth - 1.371731
    if Q.girth >= 0.09749958:
        z += -72.26878 * Q.girth + 15.49632
    if Q.sum_pt_top40 < 1007.44:
        z += -0.004853895 * Q.sum_pt_top40 + 4.890009
    if Q.n_dr_0p2_0p4 < 8.0:
        z += 0.07165094 * Q.n_dr_0p2_0p4 - 0.8810788
    if 8.0 <= Q.n_dr_0p2_0p4 < 18.0:
        z += 0.03078713 * Q.n_dr_0p2_0p4 - 0.5541683
    if Q.sum_pt < 1002.379:
        z += -0.03165917 * Q.sum_pt + 34.79093
    if 1002.379 <= Q.sum_pt < 1042.609:
        z += -0.01606139 * Q.sum_pt + 19.15604
    if 1042.609 <= Q.sum_pt < 1260.541:
        z += -0.01105985 * Q.sum_pt + 13.94139
    if 0.2091025 <= Q.LHA < 0.3098384:
        z += -36.91096 * Q.LHA + 7.718175
    if 0.3098384 <= Q.LHA < 0.3203321:
        z += -38.81694 * Q.LHA + 8.308719
    if 0.3203321 <= Q.LHA < 0.3332345:
        z += -58.02267 * Q.LHA + 14.46093
    if 0.3332345 <= Q.LHA < 0.3719813:
        z += -6.169686 * Q.LHA - 2.818273
    if Q.LHA >= 0.3719813:
        z += 5.875139 * Q.LHA - 7.298723
    if Q.psi_0p3 >= 0.9777125:
        z += -26.17182 * Q.psi_0p3 + 25.58852
    if Q.z_top50_slots >= 0.9586536:
        z += -21.4219 * Q.z_top50_slots + 20.53618
    if Q.e3 < 3.793233e-05:
        z += -15982.38 * Q.e3 + 0.606249
    if Q.sum_pt_top30 < 966.0633:
        z += 0.00958237 * Q.sum_pt_top30 - 10.13683
    if 966.0633 <= Q.sum_pt_top30 < 1027.303:
        z += 0.00779997 * Q.sum_pt_top30 - 8.414916
    if 1027.303 <= Q.sum_pt_top30 < 1191.938:
        z += 0.00244165 * Q.sum_pt_top30 - 2.910295
    if Q.log_sum_pt < 7.139296:
        z += 20.58809 * Q.log_sum_pt - 146.9845
    if Q.girth2_top15 < 0.00727763:
        z += 168.4316 * Q.girth2_top15 - 1.225783
    if 0.02210818 <= Q.e2 < 0.04358622:
        z += -52.71909 * Q.e2 + 1.165523
    if Q.e2 >= 0.04358622:
        z += -29.02238 * Q.e2 + 0.1326733
    if Q.D2 < 2.178951:
        z += 0.1459682 * Q.D2 - 0.3180576
    if Q.sj2_dr >= 0.2232169:
        z += 6.339495 * Q.sj2_dr - 1.415082
    if Q.dr_0 < 0.08082334:
        z += -4.407711 * Q.dr_0 + 0.3562459
    if Q.sj2_zsoft < 0.04575675:
        z += 28.66311 * Q.sj2_zsoft - 1.311531
    if Q.sum_pt_top50 < 1156.659:
        z += -0.006233954 * Q.sum_pt_top50 + 7.210562
    if Q.n_dr_0p2_0p4 < 18.0 and Q.log_sum_pt < 7.139296:
        z += -0.3837477 * (18.0 - Q.n_dr_0p2_0p4) * (7.139296 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.psi_0p1 < 0.8252004:
        z += 0.1481959 * (18.0 - Q.n_dr_0p2_0p4) * (0.8252004 - Q.psi_0p1)
    if Q.girth > 0.07031778 and Q.e2 > 0.04358622:
        z += 965.9595 * (Q.girth - 0.07031778) * (Q.e2 - 0.04358622)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.e3 < 3.793233e-05:
        z += -3516.743 * (18.0 - Q.n_dr_0p2_0p4) * (3.793233e-05 - Q.e3)
    if Q.girth > 0.07031778 and Q.tau2 < 0.0795038:
        z += 345.4553 * (Q.girth - 0.07031778) * (0.0795038 - Q.tau2)
    if Q.psi_0p3 > 0.9777125 and Q.n_dr_0p1_0p2 > 17.0:
        z += 2.095235 * (Q.psi_0p3 - 0.9777125) * (Q.n_dr_0p1_0p2 - 17.0)
    if Q.girth > 0.07031778 and Q.sum_pt_top30 < 1191.938:
        z += 0.1190554 * (Q.girth - 0.07031778) * (1191.938 - Q.sum_pt_top30)
    if Q.girth > 0.07031778 and Q.sum_pt_top50 < 976.277:
        z += -0.2980688 * (Q.girth - 0.07031778) * (976.277 - Q.sum_pt_top50)
    if Q.girth > 0.07031778 and Q.sum_pt < 1115.723:
        z += 0.1923356 * (Q.girth - 0.07031778) * (1115.723 - Q.sum_pt)
    if Q.sum_pt < 1260.541 and Q.D2 < 4.450169:
        z += -0.0004308569 * (1260.541 - Q.sum_pt) * (4.450169 - Q.D2)
    if Q.sum_pt < 1042.609 and Q.max_dr > 0.1939977:
        z += 0.02147705 * (1042.609 - Q.sum_pt) * (Q.max_dr - 0.1939977)
    if Q.sum_pt < 1002.379 and Q.z_0 > 0.08872428:
        z += -0.01292921 * (1002.379 - Q.sum_pt) * (Q.z_0 - 0.08872428)
    if Q.girth2_top15 < 0.02675364 and Q.tau4 > 0.01517184:
        z += 1493.029 * (0.02675364 - Q.girth2_top15) * (Q.tau4 - 0.01517184)
    if Q.girth2_top5 < 0.007164202 and Q.sj3_dr23 > 0.2563461:
        z += 420.2094 * (0.007164202 - Q.girth2_top5) * (Q.sj3_dr23 - 0.2563461)
    if Q.LHA > 0.3203321 and Q.z_14 < 0.02037449:
        z += -780.6838 * (Q.LHA - 0.3203321) * (0.02037449 - Q.z_14)
    if Q.sum_pt < 1260.541 and Q.planar_flow < 0.7225388:
        z += -0.00232315 * (1260.541 - Q.sum_pt) * (0.7225388 - Q.planar_flow)
    if Q.dr_0 < 0.08082334 and Q.n_dr_0p4_up > 0.0:
        z += 6.05537 * (0.08082334 - Q.dr_0) * (Q.n_dr_0p4_up - 0.0)
    if Q.girth2_top15 < 0.009962397 and Q.max_dr < 0.3315857:
        z += -659.9368 * (0.009962397 - Q.girth2_top15) * (0.3315857 - Q.max_dr)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.226355
    if Q.e3 < 3.376709e-05:
        z += 14551.49 * Q.e3 - 0.4913615
    if Q.girth < 0.05660088:
        z += -5.537042 * Q.girth + 0.3134014
    if 0.09749958 <= Q.girth < 0.1207452:
        z += 38.25303 * Q.girth - 3.729654
    if 0.1207452 <= Q.girth < 0.1402186:
        z += -47.17653 * Q.girth + 6.585553
    if 0.1402186 <= Q.girth < 0.1564779:
        z += -159.9566 * Q.girth + 22.39942
    if Q.girth >= 0.1564779:
        z += -104.3541 * Q.girth + 13.69885
    if Q.sj2_dr < 0.1825048:
        z += 2.823429 * Q.sj2_dr + 0.04350457
    if 0.1825048 <= Q.sj2_dr < 0.2595052:
        z += -7.257034 * Q.sj2_dr + 1.883238
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += 1.892177 * Q.z_dr_0_0p05 - 1.533253
    if Q.LHA < 0.2454112:
        z += 4.655195 * Q.LHA - 1.142437
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 12.59172 * Q.LHA - 4.683885
    if Q.LHA >= 0.404204:
        z += 61.69952 * Q.LHA - 24.53345
    if Q.sum_pt_top20 >= 750.7313:
        z += -0.002689465 * Q.sum_pt_top20 + 2.019065
    if Q.girth2_top15 < 0.0007894752:
        z += 694.8694 * Q.girth2_top15 - 1.551979
    if 0.0007894752 <= Q.girth2_top15 < 0.006142802:
        z += 187.4342 * Q.girth2_top15 - 1.151371
    if Q.girth2_top15 >= 0.02146578:
        z += -103.3899 * Q.girth2_top15 + 2.219345
    if Q.D2 < 2.178951:
        z += -0.4918343 * Q.D2 + 1.071683
    if Q.psi_0p1 >= 0.8509811:
        z += 2.476232 * Q.psi_0p1 - 2.107226
    if Q.z_top40_slots >= 0.9674996:
        z += -23.54377 * Q.z_top40_slots + 22.77859
    if Q.tau1 < 0.09591084:
        z += -51.5728 * Q.tau1 + 5.216266
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += -23.75572 * Q.tau1 + 2.548306
    if Q.e2 >= 0.02515919:
        z += 50.28722 * Q.e2 - 1.265186
    if Q.log_sum_pt < 6.811175:
        z += 6.045125 * Q.log_sum_pt - 41.1744
    if Q.sum_pt_top50 < 889.8503:
        z += -0.02320438 * Q.sum_pt_top50 + 20.64843
    if Q.girth < 0.05660088 and Q.sum_pt < 1007.788:
        z += 0.4761645 * (0.05660088 - Q.girth) * (1007.788 - Q.sum_pt)
    if Q.sum_pt_top50 < 988.4554 and Q.sum_pt_top20 > 789.7344:
        z += -0.0002719217 * (988.4554 - Q.sum_pt_top50) * (Q.sum_pt_top20 - 789.7344)
    if Q.girth < 0.05660088 and Q.psi_0p3 > 0.9638082:
        z += 425.3465 * (0.05660088 - Q.girth) * (Q.psi_0p3 - 0.9638082)
    if Q.girth2_top15 < 0.006142802 and Q.psi_0p3 > 0.9777125:
        z += 3253.257 * (0.006142802 - Q.girth2_top15) * (Q.psi_0p3 - 0.9777125)
    if Q.girth2_top15 < 0.006142802 and Q.z_dr_0p2_0p4 < 0.0684915:
        z += 4333.532 * (0.006142802 - Q.girth2_top15) * (0.0684915 - Q.z_dr_0p2_0p4)
    if Q.girth2_top15 < 0.006142802 and Q.n_dr_0p2_0p4 > 7.0:
        z += 21.35389 * (0.006142802 - Q.girth2_top15) * (Q.n_dr_0p2_0p4 - 7.0)
    if Q.LHA > 0.3719813 and Q.lam2 < 0.006427167:
        z += 4516.936 * (Q.LHA - 0.3719813) * (0.006427167 - Q.lam2)
    if Q.sum_pt_top50 < 988.4554 and Q.C3 < 0.001416411:
        z += 9.786628 * (988.4554 - Q.sum_pt_top50) * (0.001416411 - Q.C3)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.09061548:
        z += 299.9708 * (6.811175 - Q.log_sum_pt) * (0.09061548 - Q.dr_13)
    if Q.z_dr_0p1_0p2 > 0.4479367 and Q.soft5_z > 0.0004140594:
        z += 1091.219 * (Q.z_dr_0p1_0p2 - 0.4479367) * (Q.soft5_z - 0.0004140594)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.533938
    if Q.girth < 0.09749958:
        z += 37.21665 * Q.girth - 4.493731
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -34.49372 * Q.girth + 2.498
    if 0.1207452 <= Q.girth < 0.1402186:
        z += -71.71037 * Q.girth + 6.991732
    if Q.girth >= 0.1402186:
        z += -150.4619 * Q.girth + 18.03416
    if 0.04466492 <= Q.tau1 < 0.1751567:
        z += -21.50032 * Q.tau1 + 0.9603099
    if 0.1751567 <= Q.tau1 < 0.1953848:
        z += -9.758612 * Q.tau1 - 1.096328
    if Q.tau1 >= 0.1953848:
        z += -39.56258 * Q.tau1 + 4.726914
    if 0.008484542 <= Q.e2 < 0.03029714:
        z += 119.8555 * Q.e2 - 1.016919
    if Q.e2 >= 0.03029714:
        z += 81.91409 * Q.e2 + 0.1325984
    if 402.625 <= Q.sum_pt_top5 < 791.125:
        z += -0.004031805 * Q.sum_pt_top5 + 1.623306
    if Q.sum_pt_top5 >= 791.125:
        z += -0.008097086 * Q.sum_pt_top5 + 4.839451
    if 0.9985421 <= Q.psi_0p3 < 0.9989733:
        z += -487.5199 * Q.psi_0p3 + 486.8092
    if Q.psi_0p3 >= 0.9989733:
        z += 228.7998 * Q.psi_0p3 - 228.7751
    if Q.sum_pt_top50 < 889.8503:
        z += 0.003987105 * Q.sum_pt_top50 - 3.547927
    if 0.1870291 <= Q.LHA < 0.3332345:
        z += -13.19663 * Q.LHA + 2.468154
    if 0.3332345 <= Q.LHA < 0.404204:
        z += 31.32782 * Q.LHA - 12.36893
    if Q.LHA >= 0.404204:
        z += 45.66484 * Q.LHA - 18.16401
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += 0.03277582 * Q.n_dr_0p1_0p2 - 0.6882923
    if Q.girth2_top10 < 0.01414829:
        z += 17.46911 * Q.girth2_top10 - 0.2471579
    if Q.z_dr_0_0p05 >= 0.7128619:
        z += -0.9537079 * Q.z_dr_0_0p05 + 0.679862
    if Q.tau21_b2 < 0.2018786:
        z += 3.987399 * Q.tau21_b2 - 0.8049705
    if Q.log_sum_pt < 6.856375:
        z += 4.484777 * Q.log_sum_pt - 30.74931
    if Q.D2 < 2.410481:
        z += -0.4686046 * Q.D2 + 1.129563
    if Q.psi_0p1 >= 0.9184255:
        z += 8.107989 * Q.psi_0p1 - 7.446584
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += 3.113287 * Q.z_dr_0p1_0p2 - 0.3746644
    if Q.girth2_top5 < 0.02441963:
        z += 22.9856 * Q.girth2_top5 - 0.5612999
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.03760171 * Q.n_dr_0p2_0p4 + 0.6768308
    if Q.sum_pt_top20 < 889.8383:
        z += -0.00284525 * Q.sum_pt_top20 + 2.531813
    if Q.e3 < 5.13841e-05:
        z += -17585.18 * Q.e3 + 1.103663
    if 5.13841e-05 <= Q.e3 < 0.0001086251:
        z += -3495.121 * Q.e3 + 0.3796577
    if Q.girth2_top2 < 0.004007842:
        z += -160.0783 * Q.girth2_top2 + 0.6415686
    if Q.pt_entropy >= 2.07371:
        z += -0.4288511 * Q.pt_entropy + 0.8893128
    if Q.D2_b2 < 2.026142:
        z += 0.2420113 * Q.D2_b2 - 0.4903492
    if Q.M3 < 0.02944575:
        z += 12.36897 * Q.M3 - 0.3642135
    if Q.girth < 0.1207452 and Q.psi_0p2 > 0.948102:
        z += 117.1802 * (0.1207452 - Q.girth) * (Q.psi_0p2 - 0.948102)
    if Q.e2 > 0.03029714 and Q.soft1_pt > 1.091797:
        z += -14.2566 * (Q.e2 - 0.03029714) * (Q.soft1_pt - 1.091797)
    if Q.girth < 0.1207452 and Q.pt1_dr01 > 5.351077:
        z += 0.3666422 * (0.1207452 - Q.girth) * (Q.pt1_dr01 - 5.351077)
    if Q.n_dr_0p2_0p4 < 11.0 and Q.n_real_top40 > 22.0:
        z += -0.002345649 * (11.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top40 - 22.0)
    if Q.tau1 > 0.1953848 and Q.sum_pt > 1167.447:
        z += -5.109184 * (Q.tau1 - 0.1953848) * (Q.sum_pt - 1167.447)
    if Q.sj2_dr > 0.09395198 and Q.M2 > 0.04260132:
        z += -50.98749 * (Q.sj2_dr - 0.09395198) * (Q.M2 - 0.04260132)
    if Q.tau1 > 0.1953848 and Q.pt_3 < 114.75:
        z += -0.9345191 * (Q.tau1 - 0.1953848) * (114.75 - Q.pt_3)
    if Q.girth < 0.1207452 and Q.sum_pt_top50 < 997.0189:
        z += 0.04266545 * (0.1207452 - Q.girth) * (997.0189 - Q.sum_pt_top50)
    if Q.tau1 > 0.1953848 and Q.z_3 < 0.1082864:
        z += 1117.18 * (Q.tau1 - 0.1953848) * (0.1082864 - Q.z_3)
    if Q.girth > 0.09749958 and Q.sum_pt_top50 < 1156.659:
        z += 0.1520774 * (Q.girth - 0.09749958) * (1156.659 - Q.sum_pt_top50)
    if Q.sum_pt_top40 < 1024.942 and Q.n_dr_0_0p05 > 1.0:
        z += 0.0001231641 * (1024.942 - Q.sum_pt_top40) * (Q.n_dr_0_0p05 - 1.0)
    if Q.psi_0p3 > 0.9989733 and Q.eccentricity > 0.5245966:
        z += -1091.866 * (Q.psi_0p3 - 0.9989733) * (Q.eccentricity - 0.5245966)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.4669936
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.02169369 * Q.n_dr_0p2_0p4 + 0.101952
    if 10.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.02299699 * Q.n_dr_0p2_0p4 - 0.3449549
    if Q.girth < 0.08589404:
        z += -48.84694 * Q.girth + 3.990256
    if 0.08589404 <= Q.girth < 0.09749958:
        z += 17.69883 * Q.girth - 1.725628
    if Q.D2 < 1.409617:
        z += -1.318423 * Q.D2 + 1.858472
    if 0.8706159 <= Q.psi_0p2 < 0.995185:
        z += 5.665252 * Q.psi_0p2 - 4.932259
    if Q.psi_0p2 >= 0.995185:
        z += 90.83177 * Q.psi_0p2 - 89.6887
    if Q.girth2_top15 < 0.006142802:
        z += 220.1129 * Q.girth2_top15 - 0.723455
    if 0.006142802 <= Q.girth2_top15 < 0.007887677:
        z += -360.2863 * Q.girth2_top15 + 2.841822
    if Q.LHA < 0.3098384:
        z += 20.34918 * Q.LHA - 5.923981
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += -16.28372 * Q.LHA + 5.426299
    if Q.e2 < 0.01879315:
        z += -108.879 * Q.e2 + 1.770613
    if 0.01879315 <= Q.e2 < 0.02515919:
        z += -40.68539 * Q.e2 + 0.4890395
    if 0.02515919 <= Q.e2 < 0.04082832:
        z += 37.34466 * Q.e2 - 1.474134
    if 0.04082832 <= Q.e2 < 0.04358622:
        z += -18.34221 * Q.e2 + 0.7994678
    if Q.e3 < 3.376709e-05:
        z += 21735.45 * Q.e3 - 0.7339431
    if Q.z_dr_0p2_0p4 < 0.0684915:
        z += 5.161935 * Q.z_dr_0p2_0p4 - 0.1888039
    if 0.0684915 <= Q.z_dr_0p2_0p4 < 0.09122568:
        z += -7.246569 * Q.z_dr_0p2_0p4 + 0.6610732
    if Q.sj3_dr_max < 0.1506299:
        z += 6.175656 * Q.sj3_dr_max - 1.071958
    if 0.1506299 <= Q.sj3_dr_max < 0.1623049:
        z += 12.13875 * Q.sj3_dr_max - 1.970178
    if Q.psi_0p3 >= 0.9896594:
        z += 39.89824 * Q.psi_0p3 - 39.48566
    if Q.z_dr_0_0p05 >= 0.8103116:
        z += 2.33299 * Q.z_dr_0_0p05 - 1.890449
    if Q.n_dr_0p1_0p2 >= 14.0:
        z += -0.02598538 * Q.n_dr_0p1_0p2 + 0.3637953
    if Q.zdr_0 < 0.01564747:
        z += -22.06259 * Q.zdr_0 + 0.3452238
    if Q.z_0 < 0.2891675:
        z += 1.466502 * Q.z_0 - 0.4240647
    if Q.tau2 < 0.01505687:
        z += 31.09897 * Q.tau2 - 0.468253
    if Q.z_dr_0p05_0p1 >= 0.4026646:
        z += 0.9214316 * Q.z_dr_0p05_0p1 - 0.3710279
    if Q.girth2_top10 < 0.004752876:
        z += -199.5977 * Q.girth2_top10 + 0.5987599
    if 0.004752876 <= Q.girth2_top10 < 0.007678544:
        z += 119.5978 * Q.girth2_top10 - 0.9183368
    if Q.dr_0 < 0.05775119:
        z += -14.4918 * Q.dr_0 + 0.836919
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2_top10 < 0.005337976:
        z += -18.0869 * (10.0 - Q.n_dr_0p2_0p4) * (0.005337976 - Q.girth2_top10)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 33.0:
        z += 0.002821843 * (10.0 - Q.n_dr_0p2_0p4) * (33.0 - Q.n_dr_0p1_0p2)
    if Q.D2 < 1.409617 and Q.n_real_top50 > 22.0:
        z += -0.04934327 * (1.409617 - Q.D2) * (Q.n_real_top50 - 22.0)
    if Q.girth < 0.09749958 and Q.tau4 > 0.008810529:
        z += -631.1851 * (0.09749958 - Q.girth) * (Q.tau4 - 0.008810529)
    if Q.D2 < 1.409617 and Q.z_7 < 0.04963857:
        z += -28.95919 * (1.409617 - Q.D2) * (0.04963857 - Q.z_7)
    if Q.n_dr_0p1_0p2 < 19.0 and Q.planar_flow < 0.6025827:
        z += 0.08289466 * (19.0 - Q.n_dr_0p1_0p2) * (0.6025827 - Q.planar_flow)
    if Q.dr_0 < 0.09338587 and Q.eta_0 > -0.02893372:
        z += -68.60172 * (0.09338587 - Q.dr_0) * (Q.eta_0 - -0.02893372)
    if Q.z_dr_0p05_0p1 > 0.4026646 and Q.C2_b2 < 0.02704832:
        z += -49.83568 * (Q.z_dr_0p05_0p1 - 0.4026646) * (0.02704832 - Q.C2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.6859851
    if Q.girth < 0.05048381:
        z += -113.6141 * Q.girth + 6.952157
    if 0.05048381 <= Q.girth < 0.06171014:
        z += -108.3599 * Q.girth + 6.686906
    if Q.sj3_dr_max < 0.1210264:
        z += 5.831077 * Q.sj3_dr_max - 0.4981204
    if 0.1210264 <= Q.sj3_dr_max < 0.1506299:
        z += 16.25737 * Q.sj3_dr_max - 1.759978
    if 0.1506299 <= Q.sj3_dr_max < 0.1999777:
        z += -12.33421 * Q.sj3_dr_max + 2.546768
    if 0.1999777 <= Q.sj3_dr_max < 0.2628766:
        z += -1.275087 * Q.sj3_dr_max + 0.3351905
    if Q.LHA < 0.1870291:
        z += 41.88091 * Q.LHA - 9.055386
    if 0.1870291 <= Q.LHA < 0.2284021:
        z += 24.80207 * Q.LHA - 5.861145
    if 0.2284021 <= Q.LHA < 0.2454112:
        z += 11.54089 * Q.LHA - 2.832265
    if 0.3719813 <= Q.LHA < 0.404204:
        z += 13.29875 * Q.LHA - 4.946888
    if Q.LHA >= 0.404204:
        z += -9.92146 * Q.LHA + 4.438816
    if Q.log_sum_pt >= 6.811175:
        z += -4.765206 * Q.log_sum_pt + 32.45666
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.1345102 * Q.n_dr_0p2_0p4 + 1.345102
    if Q.sj3_dr12 < 0.06069672:
        z += -0.05674028 * Q.sj3_dr12 - 0.05783636
    if 0.06069672 <= Q.sj3_dr12 < 0.1048824:
        z += 8.906023 * Q.sj3_dr12 - 0.6018467
    if 0.1048824 <= Q.sj3_dr12 < 0.1840219:
        z += -4.198141 * Q.sj3_dr12 + 0.7725496
    if Q.mean_eta2 < 0.005850286:
        z += -60.87696 * Q.mean_eta2 + 0.3561476
    if Q.eccentricity >= 0.8903081:
        z += -4.060846 * Q.eccentricity + 3.615404
    if Q.e2 < 0.02515919:
        z += 177.7058 * Q.e2 - 4.470935
    if Q.e2 >= 0.06524004:
        z += 73.60403 * Q.e2 - 4.801929
    if Q.girth2_top10 < 0.01414829:
        z += 23.71498 * Q.girth2_top10 - 0.3355264
    if Q.e3 < 4.646151e-05:
        z += -44695.36 * Q.e3 + 2.076614
    if Q.M2 < 0.05910514:
        z += 23.73122 * Q.M2 - 1.402637
    if Q.girth2_top15 < 0.002197765:
        z += 400.7448 * Q.girth2_top15 + 0.4742054
    if 0.002197765 <= Q.girth2_top15 < 0.005788041:
        z += -377.3938 * Q.girth2_top15 + 2.184371
    if Q.pt_dispersion >= 0.3227599:
        z += 2.168701 * Q.pt_dispersion - 0.6999697
    if Q.sum_pt >= 907.9372:
        z += -0.01114357 * Q.sum_pt + 10.11767
    if Q.girth < 0.05048381 and Q.log_sum_pt > 6.811175:
        z += 463.8711 * (0.05048381 - Q.girth) * (Q.log_sum_pt - 6.811175)
    if Q.sj3_dr_max < 0.2628766 and Q.girth2_top10 > 0.004752876:
        z += -7838.93 * (0.2628766 - Q.sj3_dr_max) * (Q.girth2_top10 - 0.004752876)
    if Q.psi_0p3 > 0.9896594 and Q.girth2_top15 < 0.006615185:
        z += 25684.47 * (Q.psi_0p3 - 0.9896594) * (0.006615185 - Q.girth2_top15)
    if Q.LHA > 0.3719813 and Q.log_sum_pt > 6.856375:
        z += 182.8606 * (Q.LHA - 0.3719813) * (Q.log_sum_pt - 6.856375)
    if Q.psi_0p2 > 0.9734513 and Q.n_dr_0p1_0p2 < 21.0:
        z += 2.883096 * (Q.psi_0p2 - 0.9734513) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.log_sum_pt > 6.811175 and Q.mean_eta2 < 0.01204531:
        z += -14.37407 * (Q.log_sum_pt - 6.811175) * (0.01204531 - Q.mean_eta2)
    if Q.sj3_dr_max < 0.2628766 and Q.eccentricity > 0.5245966:
        z += 8.387763 * (0.2628766 - Q.sj3_dr_max) * (Q.eccentricity - 0.5245966)
    if Q.LHA > 0.404204 and Q.C2_b2 < 0.0329485:
        z += 546.2303 * (Q.LHA - 0.404204) * (0.0329485 - Q.C2_b2)
    if Q.log_sum_pt > 6.811175 and Q.ptdr0_12 > 2.533834:
        z += -0.2709196 * (Q.log_sum_pt - 6.811175) * (Q.ptdr0_12 - 2.533834)
    if Q.sum_pt < 1167.447 and Q.M2 < 0.05910514:
        z += 0.1109159 * (1167.447 - Q.sum_pt) * (0.05910514 - Q.M2)
    if Q.psi_0p3 > 0.9896594 and Q.C3 < 0.008152108:
        z += -5155.113 * (Q.psi_0p3 - 0.9896594) * (0.008152108 - Q.C3)
    if Q.sj3_dr_max < 0.1999777 and Q.ptdr0_14 < 2.084411:
        z += -1.780296 * (0.1999777 - Q.sj3_dr_max) * (2.084411 - Q.ptdr0_14)
    if Q.sj3_dr_max < 0.1999777 and Q.ptdr0_8 < 3.519576:
        z += -0.8161504 * (0.1999777 - Q.sj3_dr_max) * (3.519576 - Q.ptdr0_8)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.871447
    if Q.sum_pt < 1085.125:
        z += -0.1761815 * Q.sum_pt + 191.4163
    if 1085.125 <= Q.sum_pt < 1115.723:
        z += -0.007758667 * Q.sum_pt + 8.656521
    if Q.sum_pt >= 1167.447:
        z += -0.01971585 * Q.sum_pt + 23.0172
    if Q.n_pt_above_5 < 23.0:
        z += 0.02827203 * Q.n_pt_above_5 - 0.6502568
    if Q.n_dr_0p1_0p2 < 21.0:
        z += 0.01527672 * Q.n_dr_0p1_0p2 - 0.3971946
    if 21.0 <= Q.n_dr_0p1_0p2 < 26.0:
        z += 0.06391232 * Q.n_dr_0p1_0p2 - 1.418542
    if Q.n_dr_0p1_0p2 >= 26.0:
        z += 0.04863561 * Q.n_dr_0p1_0p2 - 1.021348
    if Q.log_sum_pt < 6.959294:
        z += 209.6713 * Q.log_sum_pt - 1464.764
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 185.6848 * Q.log_sum_pt - 1297.835
    if Q.log_sum_pt >= 7.062574:
        z += 14.95147 * Q.log_sum_pt - 105.5959
    if 0.08589404 <= Q.girth < 0.09749958:
        z += -56.1302 * Q.girth + 4.82125
    if 0.09749958 <= Q.girth < 0.1207452:
        z += -148.9149 * Q.girth + 13.86772
    if 0.1207452 <= Q.girth < 0.1402186:
        z += -171.9027 * Q.girth + 16.64339
    if 0.1402186 <= Q.girth < 0.1564779:
        z += -190.2828 * Q.girth + 19.22061
    if Q.girth >= 0.1564779:
        z += -330.0222 * Q.girth + 41.08673
    if Q.LHA >= 0.3203321:
        z += 32.73614 * Q.LHA - 10.48644
    if Q.e2 >= 0.04082832:
        z += 46.92142 * Q.e2 - 1.915723
    if Q.sum_pt_top50 < 959.0957:
        z += -0.03212591 * Q.sum_pt_top50 + 32.56083
    if 959.0957 <= Q.sum_pt_top50 < 1048.098:
        z += -0.01965118 * Q.sum_pt_top50 + 20.59637
    if Q.e3 < 0.0001841806:
        z += 1189.626 * Q.e3 + 0.9164383
    if 0.0001841806 <= Q.e3 < 0.0005178279:
        z += -3403.428 * Q.e3 + 1.76239
    if Q.psi_0p2 >= 0.9087063:
        z += -2.3161 * Q.psi_0p2 + 2.104655
    if 0.005383629 <= Q.girth2_top15 < 0.00727763:
        z += 53.21153 * Q.girth2_top15 - 0.2864712
    if 0.00727763 <= Q.girth2_top15 < 0.01563836:
        z += -64.30892 * Q.girth2_top15 + 0.5687992
    if Q.girth2_top15 >= 0.01563836:
        z += -98.45924 * Q.girth2_top15 + 1.102854
    if Q.sum_pt_top15 < 708.7945:
        z += 0.003399587 * Q.sum_pt_top15 - 2.409609
    if Q.sum_pt_top30 >= 1111.245:
        z += 0.002966292 * Q.sum_pt_top30 - 3.296277
    if Q.girth2_top5 >= 0.00422683:
        z += 32.74266 * Q.girth2_top5 - 0.1383977
    if Q.girth2_top15 > 0.009962397 and Q.sum_pt_top40 < 1225.842:
        z += 0.2698777 * (Q.girth2_top15 - 0.009962397) * (1225.842 - Q.sum_pt_top40)
    if Q.sum_pt < 1085.125 and Q.log_sum_pt < 6.811175:
        z += 0.097644 * (1085.125 - Q.sum_pt) * (6.811175 - Q.log_sum_pt)
    if Q.girth > 0.09749958 and Q.sum_pt < 1167.447:
        z += 0.3055972 * (Q.girth - 0.09749958) * (1167.447 - Q.sum_pt)
    if Q.log_sum_pt > 7.139296 and Q.psi_0p3 > 0.9777125:
        z += 174.2995 * (Q.log_sum_pt - 7.139296) * (Q.psi_0p3 - 0.9777125)
    if Q.e3 < 0.0005178279 and Q.psi_0p3 < 0.9966167:
        z += -16510.42 * (0.0005178279 - Q.e3) * (0.9966167 - Q.psi_0p3)
    if Q.sum_pt_top50 < 1048.098 and Q.girth2_top2 < 0.0003125151:
        z += 17.45789 * (1048.098 - Q.sum_pt_top50) * (0.0003125151 - Q.girth2_top2)
    if Q.girth > 0.1402186 and Q.soft6_pt > 4.250195:
        z += -313.8781 * (Q.girth - 0.1402186) * (Q.soft6_pt - 4.250195)
    if Q.girth > 0.1564779 and Q.n_pt_above_50 > 5.0:
        z += 61.22757 * (Q.girth - 0.1564779) * (Q.n_pt_above_50 - 5.0)
    if Q.e3 < 0.0001841806 and Q.pt_9 < 41.4375:
        z += -92.36777 * (0.0001841806 - Q.e3) * (41.4375 - Q.pt_9)
    if Q.sum_pt < 1085.125 and Q.sd_zg > 0.1634067:
        z += -0.008231391 * (1085.125 - Q.sum_pt) * (Q.sd_zg - 0.1634067)
    if Q.log_sum_pt > 7.139296 and Q.absphi_13 > 0.01228943:
        z += -52.86176 * (Q.log_sum_pt - 7.139296) * (Q.absphi_13 - 0.01228943)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.0297699:
        z += -258.4552 * (Q.log_sum_pt - 7.139296) * (-0.0297699 - Q.phi_0)
    if Q.log_sum_pt > 7.139296 and Q.phi_0 < -0.004917145:
        z += 152.4825 * (Q.log_sum_pt - 7.139296) * (-0.004917145 - Q.phi_0)
    if Q.sum_pt > 1167.447 and Q.abseta_12 < 0.06868286:
        z += 0.04179371 * (Q.sum_pt - 1167.447) * (0.06868286 - Q.abseta_12)
    if Q.sum_pt > 1167.447 and Q.eta_1 < 0.002292633:
        z += 0.02981463 * (Q.sum_pt - 1167.447) * (0.002292633 - Q.eta_1)
    if Q.log_sum_pt > 7.139296 and Q.abseta_9 > 0.1726074:
        z += -386.1945 * (Q.log_sum_pt - 7.139296) * (Q.abseta_9 - 0.1726074)
    if Q.log_sum_pt > 7.062574 and Q.eta_1 < 0.0001021922:
        z += -68.09692 * (Q.log_sum_pt - 7.062574) * (0.0001021922 - Q.eta_1)
    if Q.girth > 0.1402186 and Q.soft6_pt > 3.328125:
        z += -145.0668 * (Q.girth - 0.1402186) * (Q.soft6_pt - 3.328125)
    if Q.girth > 0.1402186 and Q.soft6_z > 0.002707742:
        z += 62114.02 * (Q.girth - 0.1402186) * (Q.soft6_z - 0.002707742)
    if Q.girth > 0.1402186 and Q.soft6_pt > 2.162109:
        z += -27.15858 * (Q.girth - 0.1402186) * (Q.soft6_pt - 2.162109)
    if Q.girth2_top15 > 0.005383629 and Q.soft6_pt > 1.740234:
        z += -30.09867 * (Q.girth2_top15 - 0.005383629) * (Q.soft6_pt - 1.740234)
    if Q.log_sum_pt > 7.062574 and Q.phi_10 > 0.1347656:
        z += -142.4827 * (Q.log_sum_pt - 7.062574) * (Q.phi_10 - 0.1347656)
    if Q.sum_pt < 1115.723 and Q.e4 < 2.0948e-08:
        z += -156591.4 * (1115.723 - Q.sum_pt) * (2.0948e-08 - Q.e4)
    if Q.log_sum_pt < 6.959294 and Q.C3 < 0.01311308:
        z += 138.2551 * (6.959294 - Q.log_sum_pt) * (0.01311308 - Q.C3)
    return max(0.0, z)


def neuron_14(Q):
    z = 2.457923
    if Q.tau21_b2 < 0.342495:
        z += -5.311497 * Q.tau21_b2 + 1.819161
    if Q.n_dr_0p1_0p2 < 13.0:
        z += 0.01459026 * Q.n_dr_0p1_0p2 - 0.4689703
    if 13.0 <= Q.n_dr_0p1_0p2 < 17.0:
        z += 0.06982423 * Q.n_dr_0p1_0p2 - 1.187012
    if Q.girth < 0.07031778:
        z += 162.0781 * Q.girth - 11.39697
    if 0.076787 <= Q.girth < 0.08068193:
        z += -137.6696 * Q.girth + 10.57123
    if 0.08068193 <= Q.girth < 0.08589404:
        z += -21.97599 * Q.girth + 1.236852
    if 0.08589404 <= Q.girth < 0.1207452:
        z += -184.1911 * Q.girth + 15.17016
    if Q.girth >= 0.1207452:
        z += -1102.304 * Q.girth + 126.0279
    if Q.girth2_top10 >= 0.0001295334:
        z += 42.09776 * Q.girth2_top10 - 0.005453066
    if Q.e3 < 3.376709e-05:
        z += 26869.93 * Q.e3 - 1.401949
    if 3.376709e-05 <= Q.e3 < 7.876005e-05:
        z += 10993.48 * Q.e3 - 0.8658471
    if Q.psi_0p3 >= 0.9924477:
        z += 46.82859 * Q.psi_0p3 - 46.47493
    if Q.e2 < 0.01879315:
        z += -146.7649 * Q.e2 + 3.742684
    if 0.01879315 <= Q.e2 < 0.03480688:
        z += -61.47913 * Q.e2 + 2.139897
    if Q.tau1 < 0.1507173:
        z += 36.80579 * Q.tau1 - 5.908025
    if 0.1507173 <= Q.tau1 < 0.1751567:
        z += 14.76123 * Q.tau1 - 2.585528
    if Q.z_dr_0_0p05 >= 0.6283153:
        z += 3.879379 * Q.z_dr_0_0p05 - 2.437473
    if Q.dr_6 < 0.05347848:
        z += -9.807283 * Q.dr_6 + 0.5244786
    if Q.z_dr_0p05_0p1 >= 0.7108211:
        z += -1.962716 * Q.z_dr_0p05_0p1 + 1.39514
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -4.017871 * Q.z_dr_0p1_0p2 + 0.4835253
    if Q.LHA < 0.2601462:
        z += -37.86536 * Q.LHA + 9.090602
    if 0.2601462 <= Q.LHA < 0.2941033:
        z += -11.57834 * Q.LHA + 2.252134
    if 0.2941033 <= Q.LHA < 0.302389:
        z += 13.13202 * Q.LHA - 5.015266
    if 0.302389 <= Q.LHA < 0.3332345:
        z += 33.85537 * Q.LHA - 11.28178
    if Q.sd_rg < 0.15159:
        z += -5.97815 * Q.sd_rg + 1.803695
    if 0.15159 <= Q.sd_rg < 0.1596365:
        z += 0.262423 * Q.sd_rg + 0.8576864
    if 0.1596365 <= Q.sd_rg < 0.2042612:
        z += 19.44697 * Q.sd_rg - 2.204867
    if 0.2042612 <= Q.sd_rg < 0.3017146:
        z += -4.653965 * Q.sd_rg + 2.718018
    if Q.sd_rg >= 0.3017146:
        z += 1.324186 * Q.sd_rg + 0.9143231
    if Q.M3 < 0.03787151:
        z += -8.034405 * Q.M3 + 0.304275
    if Q.tau21 < 0.5100475:
        z += -0.7672879 * Q.tau21 + 0.3913532
    if Q.C2 < 0.07996447:
        z += 19.50271 * Q.C2 - 1.559524
    if Q.lam2 < 0.001776308:
        z += -226.3247 * Q.lam2 + 0.4020224
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.01217701 * Q.n_dr_0p2_0p4 + 0.5127741
    if 11.0 <= Q.n_dr_0p2_0p4 < 21.0:
        z += -0.06467212 * Q.n_dr_0p2_0p4 + 1.358115
    if Q.zdr_0 < 0.009970338:
        z += 33.57541 * Q.zdr_0 - 0.3347581
    if Q.dr_0 < 0.06413297:
        z += -4.589741 * Q.dr_0 + 0.2943538
    if Q.sj3_dr_max < 0.1717401:
        z += 3.293669 * Q.sj3_dr_max - 1.223802
    if 0.1717401 <= Q.sj3_dr_max < 0.1999777:
        z += 7.985209 * Q.sj3_dr_max - 2.029527
    if 0.1999777 <= Q.sj3_dr_max < 0.2628766:
        z += 6.115046 * Q.sj3_dr_max - 1.655536
    if 0.2628766 <= Q.sj3_dr_max < 0.3715619:
        z += -4.897442 * Q.sj3_dr_max + 1.23939
    if Q.sj3_dr_max >= 0.3715619:
        z += -8.191111 * Q.sj3_dr_max + 2.463191
    if Q.tau21_b2 < 0.342495 and Q.girth2_top15 < 0.00727763:
        z += -581.6867 * (0.342495 - Q.tau21_b2) * (0.00727763 - Q.girth2_top15)
    if Q.tau21_b2 < 0.342495 and Q.girth2_top10 < 0.01976735:
        z += 138.9976 * (0.342495 - Q.tau21_b2) * (0.01976735 - Q.girth2_top10)
    if Q.tau21_b2 < 0.342495 and Q.sum_pt_top50 < 1156.659:
        z += -0.03757447 * (0.342495 - Q.tau21_b2) * (1156.659 - Q.sum_pt_top50)
    if Q.psi_0p3 > 0.9924477 and Q.dr_6 < 0.05347848:
        z += -2048.304 * (Q.psi_0p3 - 0.9924477) * (0.05347848 - Q.dr_6)
    if Q.girth > 0.076787 and Q.max_dr < 0.4021783:
        z += 168.1595 * (Q.girth - 0.076787) * (0.4021783 - Q.max_dr)
    if Q.girth > 0.1207452 and Q.max_dr < 0.4021783:
        z += -15183.97 * (Q.girth - 0.1207452) * (0.4021783 - Q.max_dr)
    if Q.sd_rg > 0.15159 and Q.n_pt_above_50 > 4.0:
        z += -0.7416885 * (Q.sd_rg - 0.15159) * (Q.n_pt_above_50 - 4.0)
    if Q.psi_0p3 > 0.9638082 and Q.pt_1 < 182.125:
        z += -0.0964152 * (Q.psi_0p3 - 0.9638082) * (182.125 - Q.pt_1)
    if Q.C2 < 0.07996447 and Q.pt_3 > 55.15625:
        z += 0.1746195 * (0.07996447 - Q.C2) * (Q.pt_3 - 55.15625)
    if Q.girth > 0.08589404 and Q.n_real_top40 < 36.0:
        z += 30.81634 * (Q.girth - 0.08589404) * (36.0 - Q.n_real_top40)
    if Q.girth2_top10 > 0.0001295334 and Q.n_real_top40 < 36.0:
        z += -6.213269 * (Q.girth2_top10 - 0.0001295334) * (36.0 - Q.n_real_top40)
    if Q.psi_0p3 > 0.9924477 and Q.pt_8 > 16.73281:
        z += 2.381321 * (Q.psi_0p3 - 0.9924477) * (Q.pt_8 - 16.73281)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.1722814
    if Q.z_dr_0_0p05 < 0.8459004:
        z += -0.8213364 * Q.z_dr_0_0p05 + 0.6947688
    if Q.e3 < 2.883342e-05:
        z += 26054.82 * Q.e3 - 0.7512497
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -3.973041 * Q.z_dr_0p1_0p2 + 0.4781303
    if Q.girth2_top5 < 0.008329695:
        z += -123.929 * Q.girth2_top5 + 1.032291
    if 6.915514 <= Q.log_sum_pt < 7.017258:
        z += -9.246072 * Q.log_sum_pt + 63.94134
    if Q.log_sum_pt >= 7.017258:
        z += 0.08114147 * Q.log_sum_pt - 1.510124
    if Q.sum_pt >= 907.9372:
        z += 0.002658269 * Q.sum_pt - 2.413541
    if Q.psi_0p3 >= 0.9896594:
        z += 36.0949 * Q.psi_0p3 - 35.72165
    if Q.n_dr_0p2_0p4 >= 8.0:
        z += -0.05129641 * Q.n_dr_0p2_0p4 + 0.4103712
    if Q.girth2_top3 < 0.001155057:
        z += 539.6191 * Q.girth2_top3 - 0.6232906
    if Q.girth2_top10 < 0.007678544:
        z += -97.18046 * Q.girth2_top10 + 0.7462044
    if 0.1512157 <= Q.sj2_dr < 0.2232169:
        z += -6.969311 * Q.sj2_dr + 1.053869
    if 0.2232169 <= Q.sj2_dr < 0.2780918:
        z += 10.50806 * Q.sj2_dr - 2.847375
    if Q.sj2_dr >= 0.2780918:
        z += 2.945755 * Q.sj2_dr - 0.7443607
    if Q.lam2 < 0.001776308:
        z += -179.3221 * Q.lam2 + 0.3185313
    if Q.sum_pt_top30 >= 886.3438:
        z += 0.003732387 * Q.sum_pt_top30 - 3.308177
    if Q.tau1 < 0.07708632:
        z += 25.69082 * Q.tau1 - 1.980411
    if Q.tau1 >= 0.1953848:
        z += -51.49244 * Q.tau1 + 10.06084
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.003929269 * Q.sum_pt_top40 + 3.935254
    if Q.sd_rg < 0.1690338:
        z += -1.591341 * Q.sd_rg - 0.2505547
    if 0.1690338 <= Q.sd_rg < 0.1881908:
        z += -3.470761 * Q.sd_rg + 0.06713088
    if 0.1881908 <= Q.sd_rg < 0.3017146:
        z += 5.162218 * Q.sd_rg - 1.557516
    if Q.soft5_z < 0.0004140594:
        z += -2124.574 * Q.soft5_z + 0.8796999
    if Q.girth < 0.08068193:
        z += -18.92579 * Q.girth + 1.526969
    if Q.soft5_pt < 0.5297852:
        z += 1.188307 * Q.soft5_pt - 0.6295474
    if Q.z_dr_0p2_0p4 < 0.02647293:
        z += 27.05464 * Q.z_dr_0p2_0p4 - 0.7162154
    if Q.girth2_top15 < 0.004169954:
        z += 50.21809 * Q.girth2_top15 - 1.069001
    if 0.004169954 <= Q.girth2_top15 < 0.009962397:
        z += 148.3991 * Q.girth2_top15 - 1.478411
    if Q.e2 < 0.02793599:
        z += -19.88943 * Q.e2 + 1.211692
    if 0.02793599 <= Q.e2 < 0.04358622:
        z += -41.92022 * Q.e2 + 1.827144
    if Q.n_dr_0p05_0p1 >= 6.0:
        z += -0.01779188 * Q.n_dr_0p05_0p1 + 0.1067513
    if Q.LHA >= 0.302389:
        z += -3.857233 * Q.LHA + 1.166385
    if Q.girth2_top5 < 0.008329695 and Q.sum_pt_top2 < 689.25:
        z += -0.1246962 * (0.008329695 - Q.girth2_top5) * (689.25 - Q.sum_pt_top2)
    if Q.psi_0p3 > 0.9896594 and Q.tau21_b2 < 0.4299592:
        z += -109.856 * (Q.psi_0p3 - 0.9896594) * (0.4299592 - Q.tau21_b2)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.soft5_pt < 1.480811:
        z += -1.845325 * (0.1203437 - Q.z_dr_0p1_0p2) * (1.480811 - Q.soft5_pt)
    if Q.sj2_dr > 0.2232169 and Q.dr_4 < 0.03006824:
        z += -216.6674 * (Q.sj2_dr - 0.2232169) * (0.03006824 - Q.dr_4)
    if Q.psi_0p1 > 0.707925 and Q.dr_max_012 < 0.1828389:
        z += 5.978588 * (Q.psi_0p1 - 0.707925) * (0.1828389 - Q.dr_max_012)
    if Q.girth < 0.08068193 and Q.dr1_11 > 0.1911348:
        z += -126.9197 * (0.08068193 - Q.girth) * (Q.dr1_11 - 0.1911348)
    if Q.psi_0p1 > 0.707925 and Q.dr1_11 > 0.2195171:
        z += 31.46552 * (Q.psi_0p1 - 0.707925) * (Q.dr1_11 - 0.2195171)
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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:64] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:64] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:64] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
