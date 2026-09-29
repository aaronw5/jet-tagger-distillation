"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 90.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
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
  Q.pair_mass_0_4          mass of particles 0 and 4 [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.sj3_z3                 pT share of subjet 3 of 3 (by pT)
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
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
  Q.abseta_7               |Δη| of particle 7
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
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
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_4=pair_mass(0, 4),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        sj3_z3=subjets(3)["z"][2],
        zdr_0=z[0] * dr[0],
        zdr_3=z[3] * dr[3],
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
        abseta_7=abs(eta[7]),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
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
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -1.941759
    if Q.planar_flow < 0.1484197:
        z += -15.21373 * Q.planar_flow + 2.258018
    if Q.lam1_plus_lam2 < 0.004372139:
        z += 876.2513 * Q.lam1_plus_lam2 - 1.959578
    if 0.004372139 <= Q.lam1_plus_lam2 < 0.008678045:
        z += -434.639 * Q.lam1_plus_lam2 + 3.771817
    if Q.sum_z_dr2 < 0.01323868:
        z += -492.322 * Q.sum_z_dr2 + 6.935239
    if 0.01323868 <= Q.sum_z_dr2 < 0.01882765:
        z += -74.70921 * Q.sum_z_dr2 + 1.406599
    if Q.mass < 21.78408:
        z += 0.07665974 * Q.mass - 4.691116
    if 21.78408 <= Q.mass < 29.6447:
        z += 0.1456404 * Q.mass - 6.193795
    if 29.6447 <= Q.mass < 64.61873:
        z += 0.05364924 * Q.mass - 3.466746
    if Q.lam1 < 0.0002758826:
        z += -14807.51 * Q.lam1 + 2.997267
    if 0.0002758826 <= Q.lam1 < 0.005433361:
        z += 263.3405 * Q.lam1 - 1.160518
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += -251.8658 * Q.lam1 + 1.638784
    if Q.sum_pt >= 901.5938:
        z += -0.01153413 * Q.sum_pt + 10.3991
    if Q.C2_b2 < 0.001563465:
        z += 948.2296 * Q.C2_b2 - 1.482524
    if Q.sj3_dr_max < 0.1070199:
        z += -5.463657 * Q.sj3_dr_max + 1.645662
    if 0.1070199 <= Q.sj3_dr_max < 0.233678:
        z += -1.743046 * Q.sj3_dr_max + 1.247483
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -49.81695 * Q.sj3_dr_max + 12.4813
    if Q.sj3_dr_max >= 0.3012016:
        z += -44.35329 * Q.sj3_dr_max + 10.83563
    if Q.centroid_offset >= 0.04990367:
        z += 24.76385 * Q.centroid_offset - 1.235807
    if Q.sum_z_dr < 0.08723651:
        z += 72.63448 * Q.sum_z_dr - 6.336379
    if Q.e2 < 0.0245477:
        z += -215.6261 * Q.e2 + 5.293124
    if Q.log_sum_pt < 6.080494:
        z += 14.22694 * Q.log_sum_pt - 86.50682
    if Q.log_sum_pt >= 6.670067:
        z += -13.28288 * Q.log_sum_pt + 88.5977
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.09862064 * Q.n_dr_0_0p05 - 0.3944826
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.01217938 * Q.sum_pt_top5 - 8.37256
    if Q.lam2 < 7.300726e-05:
        z += 7888.566 * Q.lam2 - 0.5759226
    if Q.planar_flow < 0.1484197 and Q.centroid_offset < 0.04990367:
        z += -131.0899 * (0.1484197 - Q.planar_flow) * (0.04990367 - Q.centroid_offset)
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -1765.861 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.sum_z_dr2 < 0.01323868 and Q.D2 < 1.002471:
        z += 243.4972 * (0.01323868 - Q.sum_z_dr2) * (1.002471 - Q.D2)
    if Q.planar_flow < 0.1484197 and Q.sum_pt_top2 < 358.375:
        z += -0.02461078 * (0.1484197 - Q.planar_flow) * (358.375 - Q.sum_pt_top2)
    if Q.sum_z_dr2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -8399.079 * (0.01323868 - Q.sum_z_dr2) * (Q.centroid_offset - 0.01837778)
    if Q.mass < 29.6447 and Q.phi_0 < 0.01452637:
        z += -2.50283 * (29.6447 - Q.mass) * (0.01452637 - Q.phi_0)
    if Q.sum_pt > 901.5938 and Q.eccentricity > 0.9458207:
        z += -0.1120744 * (Q.sum_pt - 901.5938) * (Q.eccentricity - 0.9458207)
    if Q.planar_flow < 0.1484197 and Q.z_7 < 0.02320757:
        z += 934.2573 * (0.1484197 - Q.planar_flow) * (0.02320757 - Q.z_7)
    if Q.sum_pt > 901.5938 and Q.dr0_6 > 0.2347949:
        z += 0.1679208 * (Q.sum_pt - 901.5938) * (Q.dr0_6 - 0.2347949)
    if Q.log_sum_pt > 6.670067 and Q.dr_4 < 0.07232166:
        z += 50.86662 * (Q.log_sum_pt - 6.670067) * (0.07232166 - Q.dr_4)
    if Q.sum_z_dr2 < 0.01882765 and Q.pt_6 < 46.125:
        z += -1.883131 * (0.01882765 - Q.sum_z_dr2) * (46.125 - Q.pt_6)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 81257.74 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.1484335
    if Q.lam1 < 0.0008722282:
        z += 77.68481 * Q.lam1 + 3.624372
    if 0.0008722282 <= Q.lam1 < 0.00595415:
        z += -674.8317 * Q.lam1 + 4.280738
    if 0.00595415 <= Q.lam1 < 0.008375572:
        z += -108.4855 * Q.lam1 + 0.908628
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 4.46821 * Q.log_sum_pt - 28.497
    if 6.502799 <= Q.log_sum_pt < 6.605974:
        z += 11.4879 * Q.log_sum_pt - 74.14466
    if Q.log_sum_pt >= 6.605974:
        z += 17.35436 * Q.log_sum_pt - 112.8983
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -877.9612 * Q.mass_over_sum_pt_sq + 5.121088
    if Q.z_7 < 0.02320757:
        z += 34.60429 * Q.z_7 - 2.133187
    if 0.02320757 <= Q.z_7 < 0.06164517:
        z += 49.98661 * Q.z_7 - 2.490173
    if Q.z_7 >= 0.06164517:
        z += 15.38232 * Q.z_7 - 0.3569862
    if Q.sum_z_dr2 < 0.008678045:
        z += 721.5112 * Q.sum_z_dr2 - 6.261306
    if Q.sj3_dr_max >= 0.169029:
        z += 1.525046 * Q.sj3_dr_max - 0.2577771
    if Q.pt_7 >= 53.4375:
        z += -0.05936863 * Q.pt_7 + 3.172511
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.00460329 * Q.sum_pt_top5 - 1.69214
    if Q.sum_pt_top5 >= 531.1875:
        z += -0.008122532 * Q.sum_pt_top5 + 5.067657
    if Q.lam1_plus_lam2 < 0.00609665:
        z += 527.5388 * Q.lam1_plus_lam2 - 3.216219
    if Q.sum_z_dr < 0.0717028:
        z += 36.31914 * Q.sum_z_dr - 2.604184
    if Q.lam1 < 0.008375572 and Q.centroid_offset > 0.02076709:
        z += -14840.43 * (0.008375572 - Q.lam1) * (Q.centroid_offset - 0.02076709)
    if Q.pt_7 > 34.53125 and Q.e3 < 2.955458e-05:
        z += -804.9185 * (Q.pt_7 - 34.53125) * (2.955458e-05 - Q.e3)
    if Q.z_7 < 0.06164517 and Q.D2 < 1.679198:
        z += -21.0272 * (0.06164517 - Q.z_7) * (1.679198 - Q.D2)
    if Q.pt_7 > 34.53125 and Q.sj2_dr > 0.1294903:
        z += 0.3679541 * (Q.pt_7 - 34.53125) * (Q.sj2_dr - 0.1294903)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -1.407292 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    if Q.log_sum_pt > 6.377723 and Q.centroid_offset > 0.009480685:
        z += 70.88812 * (Q.log_sum_pt - 6.377723) * (Q.centroid_offset - 0.009480685)
    if Q.log_sum_pt > 6.605974 and Q.D2 < 1.432482:
        z += 3.178758 * (Q.log_sum_pt - 6.605974) * (1.432482 - Q.D2)
    if Q.lam1 < 0.008375572 and Q.planar_flow < 0.1484197:
        z += -580.6121 * (0.008375572 - Q.lam1) * (0.1484197 - Q.planar_flow)
    if Q.sj3_dr_max > 0.169029 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.1131771 * (Q.sj3_dr_max - 0.169029) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.lam1 < 0.008375572 and Q.n_pt_above_50 > 5.0:
        z += -20.42165 * (0.008375572 - Q.lam1) * (Q.n_pt_above_50 - 5.0)
    if Q.sum_z_dr2 < 0.008678045 and Q.centroid_offset > 0.02076709:
        z += 22950.87 * (0.008678045 - Q.sum_z_dr2) * (Q.centroid_offset - 0.02076709)
    if Q.mass_over_sum_pt_sq < 0.005832932 and Q.centroid_offset > 0.00809236:
        z += -7527.503 * (0.005832932 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00809236)
    if Q.z_7 < 0.06164517 and Q.mean_phi2 < 0.008921136:
        z += 1337.858 * (0.06164517 - Q.z_7) * (0.008921136 - Q.mean_phi2)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.412004
    if Q.sj3_pair_mass_max < 62.55:
        z += -0.02407907 * Q.sj3_pair_mass_max + 1.506146
    if Q.log_sum_pt < 6.46415:
        z += -5.605565 * Q.log_sum_pt + 36.54131
    if 6.46415 <= Q.log_sum_pt < 6.605974:
        z += -2.158278 * Q.log_sum_pt + 14.25753
    if Q.log_sum_pt >= 6.842717:
        z += 6.560817 * Q.log_sum_pt - 44.89381
    if Q.z_7 < 0.03243272:
        z += -20.32594 * Q.z_7 + 0.6592254
    if Q.z_7 >= 0.03629544:
        z += -49.52026 * Q.z_7 + 1.797359
    if Q.sum_z_dr < 0.007673833:
        z += 256.2675 * Q.sum_z_dr - 1.966554
    if Q.LHA >= 0.1329373:
        z += -5.166534 * Q.LHA + 0.686825
    if 752.1 <= Q.sum_pt_top5 < 839.9547:
        z += -0.00276172 * Q.sum_pt_top5 + 2.07709
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.005528568 * Q.sum_pt_top5 + 4.401117
    if Q.pt_7 < 43.5:
        z += 0.09754388 * Q.pt_7 - 5.212501
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.1777067 * Q.pt_7 - 8.699582
    if Q.pt_7 >= 53.4375:
        z += 0.08016279 * Q.pt_7 - 3.487081
    if Q.mass < 36.22941:
        z += 0.02734759 * Q.mass - 0.9907872
    if Q.sum_z_dr2 < 0.003562611:
        z += -200.8268 * Q.sum_z_dr2 + 0.7154678
    if Q.sum_pt < 527.1781:
        z += -0.006618272 * Q.sum_pt + 5.383406
    if 527.1781 <= Q.sum_pt < 813.4156:
        z += -0.004912337 * Q.sum_pt + 4.484074
    if Q.sum_pt >= 813.4156:
        z += 0.001705935 * Q.sum_pt - 0.8993315
    if Q.zdr_0 < 0.0211821:
        z += -21.14442 * Q.zdr_0 + 0.4478832
    if Q.mass_over_sum_pt < 0.1079857:
        z += -17.08467 * Q.mass_over_sum_pt + 1.844899
    if Q.sj3_pair_mass_max < 62.55 and Q.centroid_offset > 0.01096064:
        z += -0.6314046 * (62.55 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.01096064)
    if Q.lam1 < 0.00595415 and Q.max_dr > 0.08050702:
        z += -1671.834 * (0.00595415 - Q.lam1) * (Q.max_dr - 0.08050702)
    if Q.log_sum_pt > 6.842717 and Q.pt_6 > 41.21875:
        z += -0.1908633 * (Q.log_sum_pt - 6.842717) * (Q.pt_6 - 41.21875)
    if Q.log_sum_pt > 6.842717 and Q.z_dr_0p05_0p1 < 0.4474937:
        z += 13.26374 * (Q.log_sum_pt - 6.842717) * (0.4474937 - Q.z_dr_0p05_0p1)
    if Q.sum_pt_top5 > 752.1 and Q.D2_b2 < 1.129616:
        z += 0.01280411 * (Q.sum_pt_top5 - 752.1) * (1.129616 - Q.D2_b2)
    if Q.log_sum_pt > 6.842717 and Q.D2_b2 < 1.345805:
        z += -11.45892 * (Q.log_sum_pt - 6.842717) * (1.345805 - Q.D2_b2)
    if Q.sum_pt_top5 > 752.1 and Q.abseta_0 < 0.0174408:
        z += -0.2529955 * (Q.sum_pt_top5 - 752.1) * (0.0174408 - Q.abseta_0)
    if Q.log_sum_pt > 6.842717 and Q.abseta_0 < 0.0174408:
        z += 392.2192 * (Q.log_sum_pt - 6.842717) * (0.0174408 - Q.abseta_0)
    if Q.sum_pt > 527.1781 and Q.lam2 < 0.0001947983:
        z += -12.64011 * (Q.sum_pt - 527.1781) * (0.0001947983 - Q.lam2)
    return max(0.0, z)


def neuron_3(Q):
    z = -7.584393
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 83.14883 * Q.mass_over_sum_pt - 5.665686
    if Q.centroid_offset >= 0.01096064:
        z += 41.49814 * Q.centroid_offset - 0.4548461
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -42.76478 * Q.tau1 + 2.290873
    if Q.tau1 >= 0.1027642:
        z += 5.419674 * Q.tau1 - 2.660765
    if 0.008375572 <= Q.lam1 < 0.01643375:
        z += 493.8104 * Q.lam1 - 4.135945
    if Q.lam1 >= 0.01643375:
        z += 1073.816 * Q.lam1 - 13.66762
    if 0.04081947 <= Q.sum_z_dr < 0.07608178:
        z += 112.3288 * Q.sum_z_dr - 4.585203
    if Q.sum_z_dr >= 0.07608178:
        z += 193.2504 * Q.sum_z_dr - 10.74186
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += 35.89611 * Q.lam1_plus_lam2 - 0.2699419
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -475.4568 * Q.lam1_plus_lam2 + 6.499693
    if Q.e2 >= 0.06344108:
        z += 121.8591 * Q.e2 - 7.730874
    if Q.sum_z_dr2 < 0.004372139:
        z += 395.8891 * Q.sum_z_dr2 - 1.730882
    if Q.sum_z_dr2 >= 0.008678045:
        z += -1595.478 * Q.sum_z_dr2 + 13.84563
    if 0.1492731 <= Q.sj2_dr < 0.1872617:
        z += 0.04679698 * Q.sj2_dr - 0.006985531
    if 0.1872617 <= Q.sj2_dr < 0.2687922:
        z += 40.31778 * Q.sj2_dr - 7.548199
    if Q.sj2_dr >= 0.2687922:
        z += 16.5533 * Q.sj2_dr - 1.160493
    if Q.mean_eta < -0.004664942:
        z += -24.67191 * Q.mean_eta - 0.115093
    if Q.mean_eta >= 0.01772426:
        z += 35.01825 * Q.mean_eta - 0.6206725
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += -41.72348 * Q.sum_z_dr2_top5 + 0.7278792
    if 0.007164202 <= Q.sum_z_dr2_top5 < 0.008329695:
        z += -368.0534 * Q.sum_z_dr2_top5 + 3.065773
    if Q.max_dr >= 0.1027585:
        z += 22.09292 * Q.max_dr - 2.270235
    if Q.lam2 >= 0.001130645:
        z += 163.8873 * Q.lam2 - 0.1852983
    if Q.mass >= 64.61873:
        z += -0.1126984 * Q.mass + 7.282427
    if Q.sd_mass >= 62.55:
        z += 0.08169114 * Q.sd_mass - 5.109781
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj3_pair_mass_min > 5.744224:
        z += -0.8466741 * (Q.mass_over_sum_pt - 0.0681391) * (Q.sj3_pair_mass_min - 5.744224)
    if Q.mass_over_sum_pt > 0.0681391 and Q.pt_6 > 31.90625:
        z += -0.6284361 * (Q.mass_over_sum_pt - 0.0681391) * (Q.pt_6 - 31.90625)
    if Q.sum_z_dr > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 79.09886 * (Q.sum_z_dr - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.e2 > 0.06344108 and Q.sj2_mass1 > 16.86126:
        z += 1.285789 * (Q.e2 - 0.06344108) * (Q.sj2_mass1 - 16.86126)
    if Q.sj2_dr > 0.1872617 and Q.sj2_mass1 > 2.250113:
        z += -0.4189224 * (Q.sj2_dr - 0.1872617) * (Q.sj2_mass1 - 2.250113)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -1167.134 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.lam1_plus_lam2 > 0.007520088 and Q.sj3_pairmin_over_m > 0.07708997:
        z += 488.851 * (Q.lam1_plus_lam2 - 0.007520088) * (Q.sj3_pairmin_over_m - 0.07708997)
    if Q.centroid_offset > 0.01096064 and Q.n_pt_above_50 < 8.0:
        z += -2.538552 * (Q.centroid_offset - 0.01096064) * (8.0 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.01096064 and Q.abseta_0 < 0.07861328:
        z += 574.8776 * (Q.centroid_offset - 0.01096064) * (0.07861328 - Q.abseta_0)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 21.80085 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    if Q.sd_mass > 62.55 and Q.D2_b2 < 0.9206502:
        z += -0.03645519 * (Q.sd_mass - 62.55) * (0.9206502 - Q.D2_b2)
    if Q.e2 > 0.06344108 and Q.z_top5 > 0.7773372:
        z += -725.9991 * (Q.e2 - 0.06344108) * (Q.z_top5 - 0.7773372)
    if Q.sj2_dr > 0.1872617 and Q.dr_3 < 0.05268713:
        z += -190.4909 * (Q.sj2_dr - 0.1872617) * (0.05268713 - Q.dr_3)
    if Q.lam2 > 0.001130645 and Q.z_6 > 0.05441452:
        z += 9793.629 * (Q.lam2 - 0.001130645) * (Q.z_6 - 0.05441452)
    if Q.max_dr > 0.1027585 and Q.z_dr_0p05_0p1 < 0.8460335:
        z += -8.583763 * (Q.max_dr - 0.1027585) * (0.8460335 - Q.z_dr_0p05_0p1)
    if Q.e2 > 0.06344108 and Q.zdr_3 < 0.01257746:
        z += 10424.19 * (Q.e2 - 0.06344108) * (0.01257746 - Q.zdr_3)
    return max(0.0, z)


def neuron_4(Q):
    z = -1.921885
    if Q.N2 < 0.2233283:
        z += -43.36493 * Q.N2 + 9.684614
    if Q.lam2 < 0.000537286:
        z += 3485.246 * Q.lam2 - 1.872574
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -122.8147 * Q.mass_over_sum_pt + 11.10415
    if Q.lam1_plus_lam2 < 0.001653836:
        z += -1865.75 * Q.lam1_plus_lam2 + 3.085644
    if Q.lam1_plus_lam2 >= 0.003562611:
        z += 321.8642 * Q.lam1_plus_lam2 - 1.146677
    if Q.sum_pt < 739.5:
        z += 0.006794204 * Q.sum_pt - 5.024314
    if Q.C2 < 0.06729223:
        z += -82.31577 * Q.C2 + 5.539212
    if Q.sum_z_dr >= 0.08723651:
        z += 26.22764 * Q.sum_z_dr - 2.288008
    if 44.82259 <= Q.sd_mass < 74.57663:
        z += 0.09045196 * Q.sd_mass - 4.054291
    if Q.sd_mass >= 74.57663:
        z += -0.09487842 * Q.sd_mass + 9.767023
    if Q.e3 < 0.0001869378:
        z += -6419.565 * Q.e3 + 1.20006
    if Q.sj3_dr_max < 0.213399:
        z += 27.56449 * Q.sj3_dr_max - 6.031429
    if 0.213399 <= Q.sj3_dr_max < 0.233678:
        z += 37.92507 * Q.sj3_dr_max - 8.242365
    if 0.233678 <= Q.sj3_dr_max < 0.3456459:
        z += -5.536296 * Q.sj3_dr_max + 1.913598
    if Q.max_dr >= 0.1598486:
        z += 17.13181 * Q.max_dr - 2.738496
    if Q.sum_z_dr2 < 0.008678045:
        z += 944.9684 * Q.sum_z_dr2 - 8.200478
    if Q.sum_z_dr2_top5 < 0.007164202:
        z += -218.0714 * Q.sum_z_dr2_top5 + 1.562307
    if Q.centroid_offset < 0.04990367:
        z += 25.2504 * Q.centroid_offset - 1.260087
    if Q.sum_z_dr2_top2 < 0.007639643:
        z += -265.722 * Q.sum_z_dr2_top2 + 2.030022
    if Q.mass_top5 < 37.76455:
        z += 0.01600659 * Q.mass_top5 - 0.6044816
    if Q.mass >= 76.6557:
        z += 0.04483171 * Q.mass - 3.436606
    if Q.sum_zz_dr2 < 0.01165737:
        z += -3348.658 * Q.sum_zz_dr2 + 39.03656
    if Q.e2 < 0.05028464:
        z += 69.0192 * Q.e2 - 3.470606
    if Q.tau1 < 0.07283629:
        z += -26.15017 * Q.tau1 + 1.904681
    if Q.mass_over_sum_pt_sq < 0.0116609:
        z += 2544.96 * Q.mass_over_sum_pt_sq - 29.67653
    if Q.N2 < 0.2233283 and Q.mass < 62.55:
        z += -0.4881729 * (0.2233283 - Q.N2) * (62.55 - Q.mass)
    if Q.N2 < 0.2233283 and Q.sum_zz_dr2 > 0.01165737:
        z += -818.5781 * (0.2233283 - Q.N2) * (Q.sum_zz_dr2 - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_7 < 53.4375:
        z += -0.5000827 * (0.2233283 - Q.N2) * (53.4375 - Q.pt_7)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -97.84335 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.N2 < 0.2233283 and Q.abseta_7 < 0.1218872:
        z += -35.96426 * (0.2233283 - Q.N2) * (0.1218872 - Q.abseta_7)
    if Q.sd_mass > 44.82259 and Q.centroid_offset > 0.001308549:
        z += -1.237701 * (Q.sd_mass - 44.82259) * (Q.centroid_offset - 0.001308549)
    if Q.sd_mass > 44.82259 and Q.sd_zg < 0.275762:
        z += -0.5113078 * (Q.sd_mass - 44.82259) * (0.275762 - Q.sd_zg)
    if Q.lam1_plus_lam2 > 0.003562611 and Q.sj3_z3 < 0.1057566:
        z += -1513.02 * (Q.lam1_plus_lam2 - 0.003562611) * (0.1057566 - Q.sj3_z3)
    if Q.lam2 < 0.000537286 and Q.dr01 < 0.1410336:
        z += -16780.25 * (0.000537286 - Q.lam2) * (0.1410336 - Q.dr01)
    if Q.max_dr > 0.1598486 and Q.C2_b2 < 0.0006435798:
        z += -36269.2 * (Q.max_dr - 0.1598486) * (0.0006435798 - Q.C2_b2)
    if Q.sum_z_dr2_top2 < 0.007639643 and Q.C2_b2 > 0.0006435798:
        z += -17616.18 * (0.007639643 - Q.sum_z_dr2_top2) * (Q.C2_b2 - 0.0006435798)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.7792854
    if Q.LHA < 0.2160559:
        z += -14.14225 * Q.LHA + 3.055516
    if Q.z_7 < 0.02807091:
        z += -269.7876 * Q.z_7 + 12.0002
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -144.2803 * Q.z_7 + 8.477091
    if 0.04939969 <= Q.z_7 < 0.07148865:
        z += -61.10252 * Q.z_7 + 4.368137
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -12.21821 * Q.log_sum_pt + 81.87721
    if Q.log_sum_pt >= 6.896095:
        z += -20.54078 * Q.log_sum_pt + 139.2704
    if Q.sum_zz_dr2 < 0.002074109:
        z += 76.21478 * Q.sum_zz_dr2 + 0.667915
    if 0.002074109 <= Q.sum_zz_dr2 < 0.005284669:
        z += -257.2738 * Q.sum_zz_dr2 + 1.359607
    if Q.sum_z_dr2 < 0.001653836:
        z += -421.7921 * Q.sum_z_dr2 + 0.6975752
    if Q.pt_7 < 37.15625:
        z += 0.1662308 * Q.pt_7 - 6.176511
    if Q.zdr_0 < 0.0211821:
        z += 59.08962 * Q.zdr_0 - 1.251642
    if Q.pair_mass_0_4 >= 23.26771:
        z += -0.1241969 * Q.pair_mass_0_4 + 2.889778
    if Q.pt_5 < 24.57812:
        z += -0.1258687 * Q.pt_5 + 3.093617
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -519.3907 * Q.lam1_plus_lam2 + 1.850387
    if Q.centroid_offset < 0.006789738:
        z += -161.1964 * Q.centroid_offset + 1.094481
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += 221.0508 * Q.sum_z_dr2_top3 - 0.4756059
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.005579758 * Q.sum_pt_top5 - 3.835735
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -102.0208 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.sum_zz_dr2 < 0.005284669 and Q.centroid_offset < 0.01437952:
        z += -10369.07 * (0.005284669 - Q.sum_zz_dr2) * (0.01437952 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -965.5451 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 40.2:
        z += 0.410487 * (0.07148865 - Q.z_7) * (40.2 - Q.mass_top3)
    if Q.sum_z_dr2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 113310.2 * (0.001653836 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.LHA < 0.2160559 and Q.lam1 < 0.001503553:
        z += -12641.0 * (0.2160559 - Q.LHA) * (0.001503553 - Q.lam1)
    if Q.log_sum_pt > 6.701242 and Q.dr_2 < 0.01341502:
        z += 934.9284 * (Q.log_sum_pt - 6.701242) * (0.01341502 - Q.dr_2)
    if Q.log_sum_pt > 6.896095 and Q.dr_2 < 0.02270492:
        z += -731.5899 * (Q.log_sum_pt - 6.896095) * (0.02270492 - Q.dr_2)
    if Q.LHA < 0.2160559 and Q.n_dr_0p2_0p4 > 0.0:
        z += -12.63685 * (0.2160559 - Q.LHA) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.z_7 < 0.04939969 and Q.dr_2 < 0.009661512:
        z += -4680.196 * (0.04939969 - Q.z_7) * (0.009661512 - Q.dr_2)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi > 0.002834884:
        z += -801.7661 * (Q.log_sum_pt - 6.701242) * (Q.mean_phi - 0.002834884)
    if Q.sum_pt_top5 > 430.75 and Q.sj2_dr > 0.1682655:
        z += 0.02270865 * (Q.sum_pt_top5 - 430.75) * (Q.sj2_dr - 0.1682655)
    if Q.pt_7 < 37.15625 and Q.pt_5 > 24.57812:
        z += 0.001412039 * (37.15625 - Q.pt_7) * (Q.pt_5 - 24.57812)
    if Q.sum_pt_top5 > 430.75 and Q.pt_6 < 46.125:
        z += 0.0001679287 * (Q.sum_pt_top5 - 430.75) * (46.125 - Q.pt_6)
    if Q.pair_mass_0_4 > 23.26771 and Q.eccentricity > 0.9704496:
        z += 3.965226 * (Q.pair_mass_0_4 - 23.26771) * (Q.eccentricity - 0.9704496)
    if Q.zdr_0 < 0.0211821 and Q.phi_0 > -0.0297699:
        z += 484.4983 * (0.0211821 - Q.zdr_0) * (Q.phi_0 - -0.0297699)
    if Q.log_sum_pt > 6.701242 and Q.mean_eta2 < 9.030369e-05:
        z += 68448.7 * (Q.log_sum_pt - 6.701242) * (9.030369e-05 - Q.mean_eta2)
    if Q.pair_mass_0_4 > 23.26771 and Q.mean_eta2 < 0.01423545:
        z += 5.518715 * (Q.pair_mass_0_4 - 23.26771) * (0.01423545 - Q.mean_eta2)
    if Q.centroid_offset < 0.006789738 and Q.pt_5 < 56.4375:
        z += -7.996024 * (0.006789738 - Q.centroid_offset) * (56.4375 - Q.pt_5)
    return max(0.0, z)


def neuron_6(Q):
    z = -1.821918
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 99.19714 * Q.centroid_offset - 0.802739
    if Q.centroid_offset >= 0.01837778:
        z += -27.80437 * Q.centroid_offset + 1.531267
    if Q.lam1_plus_lam2 < 0.01323868:
        z += 374.0057 * Q.lam1_plus_lam2 - 4.95134
    if Q.pt_6 < 29.90625:
        z += -0.009252276 * Q.pt_6 + 0.1074779
    if 29.90625 <= Q.pt_6 < 39.75:
        z += 0.04194245 * Q.pt_6 - 1.423564
    if 39.75 <= Q.pt_6 < 41.21875:
        z += -0.165888 * Q.pt_6 + 6.837696
    if Q.lam2 < 0.001130645:
        z += 82.21088 * Q.lam2 + 1.301237
    if 0.001130645 <= Q.lam2 < 0.003408389:
        z += -612.0916 * Q.lam2 + 2.086246
    if Q.tau1 < 0.1136369:
        z += -37.44643 * Q.tau1 + 4.255297
    if Q.sj3_dr_min >= 0.1278212:
        z += -6.813321 * Q.sj3_dr_min + 0.8708867
    if Q.lam1 < 0.00595415:
        z += -732.7112 * Q.lam1 + 6.58414
    if 0.00595415 <= Q.lam1 < 0.01200373:
        z += -367.2106 * Q.lam1 + 4.407895
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.04089176 * Q.sj3_pair_mass_min + 0.1840835
    if Q.sum_zz_dr2 < 0.0030133:
        z += -262.3871 * Q.sum_zz_dr2 + 0.7906511
    if Q.eccentricity >= 0.927072:
        z += -9.096772 * Q.eccentricity + 8.433363
    if Q.sj3_dr_max < 0.1789613:
        z += -31.41903 * Q.sj3_dr_max + 5.622792
    if Q.sj3_dr_max >= 0.1879486:
        z += 13.37611 * Q.sj3_dr_max - 2.51402
    if Q.max_dr < 0.1452311:
        z += 30.37485 * Q.max_dr - 4.411374
    if Q.mean_eta >= 0.02644207:
        z += -12.92697 * Q.mean_eta + 0.3418158
    if Q.e3 < 0.0005116989:
        z += -2227.411 * Q.e3 + 1.139764
    if Q.mass < 45.595:
        z += -0.0621719 * Q.mass + 2.834728
    if Q.sum_z_dr2 < 0.008678045:
        z += 1316.887 * Q.sum_z_dr2 - 11.42801
    if Q.C2_b2 < 0.02415398:
        z += -51.8308 * Q.C2_b2 + 1.25192
    if Q.sj2_dr < 0.1872617:
        z += 4.637366 * Q.sj2_dr - 0.8684011
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += 1.2336 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 0.6887114 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.00809236 and Q.psi_0p1 > 0.4008925:
        z += 42.35003 * (Q.centroid_offset - 0.00809236) * (Q.psi_0p1 - 0.4008925)
    if Q.centroid_offset > 0.01837778 and Q.C2 < 0.02702951:
        z += 1441.834 * (Q.centroid_offset - 0.01837778) * (0.02702951 - Q.C2)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 7907.696 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.lam1 < 0.01200373 and Q.planar_flow < 0.2534037:
        z += -176.8698 * (0.01200373 - Q.lam1) * (0.2534037 - Q.planar_flow)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -7.460911 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.8183293 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 615.875 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.002229119 * (615.875 - Q.sum_pt) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 62.77277 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.pt_6 < 41.21875 and Q.M3 > 0.0782171:
        z += 2.583668 * (41.21875 - Q.pt_6) * (Q.M3 - 0.0782171)
    if Q.lam1 < 0.00733008 and Q.n_dr_0p2_0p4 < 1.0:
        z += 110.4585 * (0.00733008 - Q.lam1) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.mean_eta > 0.02644207 and Q.M3 > 0.06688759:
        z += -655.3786 * (Q.mean_eta - 0.02644207) * (Q.M3 - 0.06688759)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.086442
    if Q.sum_z_dr2_top2 < 0.001056655:
        z += -364.4544 * Q.sum_z_dr2_top2 + 0.3851027
    if Q.sum_z_dr2 < 0.0009641429:
        z += 1621.601 * Q.sum_z_dr2 - 1.563455
    if 0.001653836 <= Q.sum_z_dr2 < 0.004372139:
        z += 400.2188 * Q.sum_z_dr2 - 0.6618965
    if 0.004372139 <= Q.sum_z_dr2 < 0.007520088:
        z += 218.2492 * Q.sum_z_dr2 + 0.1337001
    if 0.007520088 <= Q.sum_z_dr2 < 0.01323868:
        z += -517.0123 * Q.sum_z_dr2 + 5.662931
    if Q.sum_z_dr2 >= 0.01323868:
        z += 1025.233 * Q.sum_z_dr2 - 14.75435
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -58.94155 * Q.mass_over_sum_pt + 0.6542419
    if 0.07269073 <= Q.mass_over_sum_pt < 0.08475161:
        z += 64.51047 * Q.mass_over_sum_pt - 8.319576
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 254.2763 * Q.mass_over_sum_pt - 24.40254
    if 0.09041383 <= Q.mass_over_sum_pt < 0.1079857:
        z += -84.59494 * Q.mass_over_sum_pt + 6.23611
    if Q.mass_over_sum_pt >= 0.1079857:
        z += -636.3637 * Q.mass_over_sum_pt + 65.81923
    if Q.tau1 < 0.05356915:
        z += -29.25074 * Q.tau1 + 1.566938
    if Q.sum_z_dr < 0.04081947:
        z += 100.5535 * Q.sum_z_dr - 6.615407
    if 0.04081947 <= Q.sum_z_dr < 0.08723651:
        z += 54.09362 * Q.sum_z_dr - 4.718939
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.04752086 * Q.mass - 1.721653
    if Q.mass >= 76.6557:
        z += -0.1298176 * Q.mass + 11.87235
    if Q.z_dr_0p1_0p2 < 0.1585582:
        z += -3.185009 * Q.z_dr_0p1_0p2 + 0.5050092
    if Q.centroid_offset < 0.02076709:
        z += 22.23017 * Q.centroid_offset - 0.4616561
    if 0.03117077 <= Q.centroid_offset < 0.03776099:
        z += 47.64176 * Q.centroid_offset - 1.48503
    if Q.centroid_offset >= 0.03776099:
        z += -80.82099 * Q.centroid_offset + 3.365851
    if Q.z_7 >= 0.03243272:
        z += 16.36214 * Q.z_7 - 0.5306688
    if Q.e2 >= 0.03556091:
        z += -44.19152 * Q.e2 + 1.571491
    if Q.sj2_dr < 0.1294903:
        z += -8.791081 * Q.sj2_dr + 0.2497526
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += 3.893444 * Q.sj2_dr - 1.39277
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 27.51991 * Q.sj2_dr - 5.153426
    if Q.LHA < 0.3033137:
        z += -5.598626 * Q.LHA + 1.69814
    if Q.lam1_plus_lam2 >= 0.005590289:
        z += -729.9369 * Q.lam1_plus_lam2 + 4.080558
    if Q.pt_7 < 29.04219:
        z += 0.0459253 * Q.pt_7 - 1.333771
    if 0.2037854 <= Q.sd_rg < 0.2787955:
        z += -5.958006 * Q.sd_rg + 1.214155
    if Q.sd_rg >= 0.2787955:
        z += 22.87823 * Q.sd_rg - 6.82526
    if Q.lam1 < 0.008375572:
        z += 359.386 * Q.lam1 - 3.010063
    if Q.e3 < 8.147744e-05:
        z += 22236.42 * Q.e3 - 1.811766
    if Q.planar_flow < 0.1950135 and Q.pt_6 < 35.28125:
        z += -0.2171163 * (0.1950135 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.1950135 and Q.sd_mass > 38.43971:
        z += 0.1443543 * (0.1950135 - Q.planar_flow) * (Q.sd_mass - 38.43971)
    if Q.centroid_offset > 0.03117077 and Q.n_pt_above_50 > 4.0:
        z += -14.27832 * (Q.centroid_offset - 0.03117077) * (Q.n_pt_above_50 - 4.0)
    if Q.centroid_offset < 0.02076709 and Q.sum_pt_top3 > 331.25:
        z += 0.09617643 * (0.02076709 - Q.centroid_offset) * (Q.sum_pt_top3 - 331.25)
    if Q.sum_z_dr2 > 0.004372139 and Q.planar_flow < 0.1950135:
        z += 635.6932 * (Q.sum_z_dr2 - 0.004372139) * (0.1950135 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.01323868 and Q.eccentricity > 0.9458207:
        z += 1156.403 * (Q.sum_z_dr2 - 0.01323868) * (Q.eccentricity - 0.9458207)
    if Q.centroid_offset > 0.03776099 and Q.pt_4 > 81.375:
        z += -41.68995 * (Q.centroid_offset - 0.03776099) * (Q.pt_4 - 81.375)
    if Q.centroid_offset < 0.02076709 and Q.C2_b2 < 0.004032342:
        z += -22816.76 * (0.02076709 - Q.centroid_offset) * (0.004032342 - Q.C2_b2)
    if Q.sum_z_dr < 0.08723651 and Q.C2_b2 < 0.004032342:
        z += 1858.564 * (0.08723651 - Q.sum_z_dr) * (0.004032342 - Q.C2_b2)
    if Q.centroid_offset < 0.02076709 and Q.tau21_b2 < 0.02656143:
        z += 3518.494 * (0.02076709 - Q.centroid_offset) * (0.02656143 - Q.tau21_b2)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.7018005
    if Q.tau1 < 0.05356915:
        z += 29.0631 * Q.tau1 - 1.556886
    if Q.sum_z_dr2 < 0.005019719:
        z += -396.1382 * Q.sum_z_dr2 + 1.988502
    if Q.LHA < 0.1967397:
        z += 43.73578 * Q.LHA - 8.604566
    if Q.log_sum_pt >= 6.701242:
        z += -14.89147 * Q.log_sum_pt + 99.79135
    if Q.sum_z_dr < 0.06108601:
        z += 117.8629 * Q.sum_z_dr - 7.199772
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.006045945 * Q.sum_pt_top5 - 4.15621
    if Q.mass < 49.6681:
        z += 0.01264087 * Q.mass - 0.6278478
    if Q.lam1_plus_lam2 < 0.003562611:
        z += -579.0594 * Q.lam1_plus_lam2 + 2.062964
    if Q.mass_over_sum_pt < 0.03319429:
        z += -44.95761 * Q.mass_over_sum_pt + 1.492336
    if Q.sj3_dr_max < 0.1426152:
        z += 18.40187 * Q.sj3_dr_max - 2.410979
    if 0.1426152 <= Q.sj3_dr_max < 0.1986272:
        z += -3.810003 * Q.sj3_dr_max + 0.7567703
    if Q.sum_z_dr2 < 0.006679471 and Q.D2_b2 < 4.721224:
        z += -26.20156 * (0.006679471 - Q.sum_z_dr2) * (4.721224 - Q.D2_b2)
    if Q.sum_z_dr2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 18403.57 * (0.006679471 - Q.sum_z_dr2) * (0.02355416 - Q.centroid_offset)
    if Q.sum_z_dr2 < 0.005019719 and Q.pt_7 < 43.5:
        z += -7.089315 * (0.005019719 - Q.sum_z_dr2) * (43.5 - Q.pt_7)
    if Q.sum_z_dr2 < 0.006679471 and Q.phi_0 > -0.04013062:
        z += 1348.293 * (0.006679471 - Q.sum_z_dr2) * (Q.phi_0 - -0.04013062)
    if Q.sum_z_dr2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 8560.839 * (0.005019719 - Q.sum_z_dr2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.sum_z_dr < 0.06108601 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -262.3944 * (0.06108601 - Q.sum_z_dr) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.sum_pt_top5 > 687.4375 and Q.e3 < 3.10694e-07:
        z += 10668.49 * (Q.sum_pt_top5 - 687.4375) * (3.10694e-07 - Q.e3)
    if Q.mass < 29.6447 and Q.D2_b2 < 0.716559:
        z += -0.1322557 * (29.6447 - Q.mass) * (0.716559 - Q.D2_b2)
    if Q.sum_z_dr < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 168517.9 * (0.06108601 - Q.sum_z_dr) * (0.0001947983 - Q.lam2)
    if Q.lam2 < 0.0001330621 and Q.D2_b2 < 4.721224:
        z += 1288.685 * (0.0001330621 - Q.lam2) * (4.721224 - Q.D2_b2)
    if Q.mass < 21.78408 and Q.lam2 < 0.0001947983:
        z += -263.4986 * (21.78408 - Q.mass) * (0.0001947983 - Q.lam2)
    if Q.mass < 29.6447 and Q.e3 < 1.762929e-06:
        z += 17760.4 * (29.6447 - Q.mass) * (1.762929e-06 - Q.e3)
    if Q.log_sum_pt > 6.701242 and Q.pt_7 < 48.71875:
        z += 0.1654253 * (Q.log_sum_pt - 6.701242) * (48.71875 - Q.pt_7)
    if Q.sum_z_dr2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -34728.52 * (0.005019719 - Q.sum_z_dr2) * (Q.centroid_offset - 0.006789738)
    if Q.LHA < 0.1967397 and Q.pt_7 > 15.55391:
        z += 0.1775397 * (0.1967397 - Q.LHA) * (Q.pt_7 - 15.55391)
    if Q.sum_z_dr < 0.06108601 and Q.lam1_plus_lam2 > 0.0003193707:
        z += 14169.6 * (0.06108601 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.0003193707)
    if Q.mass < 29.6447 and Q.lam1_plus_lam2 < 0.003562611:
        z += -35.0627 * (29.6447 - Q.mass) * (0.003562611 - Q.lam1_plus_lam2)
    if Q.sj3_dr_max < 0.1986272 and Q.sum_z_dr2 < 0.006679471:
        z += 1006.857 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.sum_z_dr2)
    if Q.tau1 < 0.05356915 and Q.lam1_plus_lam2 < 0.003562611:
        z += 28723.7 * (0.05356915 - Q.tau1) * (0.003562611 - Q.lam1_plus_lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.827461
    if Q.sum_z_dr < 0.05464922:
        z += 17.40021 * Q.sum_z_dr - 0.9509078
    if Q.sum_z_dr >= 0.0717028:
        z += -37.81763 * Q.sum_z_dr + 2.71163
    if Q.tau1 < 0.04369778:
        z += -33.45319 * Q.tau1 + 1.46183
    if Q.mass < 53.33237:
        z += 0.05345627 * Q.mass - 2.85095
    if Q.e3 < 2.371297e-05:
        z += 55527.0 * Q.e3 - 1.31671
    if 8.147744e-05 <= Q.e3 < 0.0001869378:
        z += -6157.677 * Q.e3 + 0.5017117
    if Q.e3 >= 0.0001869378:
        z += 1429.142 * Q.e3 - 0.9165516
    if Q.sum_z_dr2 < 0.003562611:
        z += -3311.285 * Q.sum_z_dr2 + 15.84957
    if 0.003562611 <= Q.sum_z_dr2 < 0.00609665:
        z += -1599.323 * Q.sum_z_dr2 + 9.750514
    if Q.sj3_dr_max < 0.1426152:
        z += 36.66077 * Q.sj3_dr_max - 3.078119
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -30.37789 * Q.sj3_dr_max + 6.48261
    if Q.lam2 >= 0.001130645:
        z += 527.8073 * Q.lam2 - 0.5967625
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.4535366 * Q.n_dr_0p2_0p4 - 0.4535366
    if Q.lam1 < 0.003377388:
        z += 801.3905 * Q.lam1 - 4.771599
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 522.7058 * Q.lam1 - 3.830373
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 1103.306 * Q.lam1 - 6.984981
    if Q.lam1 >= 0.00595415:
        z += 301.9152 * Q.lam1 - 2.213382
    if Q.centroid_offset < 0.002316125:
        z += 207.4726 * Q.centroid_offset + 2.968198
    if 0.002316125 <= Q.centroid_offset < 0.01837778:
        z += -214.7182 * Q.centroid_offset + 3.946044
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -143.2502 * Q.e2 + 5.094107
    if Q.e2 >= 0.04447357:
        z += 37.26608 * Q.e2 - 2.934095
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.0702708 * Q.sj3_pair_mass_max - 1.702671
    if Q.lam1_plus_lam2 < 0.0001721983:
        z += -7309.814 * Q.lam1_plus_lam2 + 1.258738
    if Q.C3 < 0.0284695:
        z += 38.1104 * Q.C3 - 1.084984
    if Q.zdr_0 < 0.006292091:
        z += 149.6714 * Q.zdr_0 - 0.9417462
    if Q.mass_over_sum_pt < 0.07269073:
        z += 65.44125 * Q.mass_over_sum_pt - 4.756973
    if Q.sum_pt < 559.6875:
        z += -0.01145657 * Q.sum_pt + 9.116339
    if 559.6875 <= Q.sum_pt < 988.4078:
        z += -0.006307707 * Q.sum_pt + 6.234587
    z += -1.843285 * Q.z_dr_0p2_0p4
    if Q.log_sum_pt >= 6.896095:
        z += 12.15207 * Q.log_sum_pt - 83.80183
    if Q.sum_pt_top5 >= 839.9547:
        z += -0.008933309 * Q.sum_pt_top5 + 7.503575
    if Q.pt_4 < 31.125:
        z += -0.07467485 * Q.pt_4 + 2.324255
    if Q.max_dr < 0.1117619:
        z += -36.59199 * Q.max_dr + 4.089588
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 5.515841 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.07610829 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00609665 and Q.planar_flow < 0.3220738:
        z += 1488.321 * (0.00609665 - Q.sum_z_dr2) * (0.3220738 - Q.planar_flow)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi < 0.002834884:
        z += -10091.66 * (0.00609665 - Q.sum_z_dr2) * (0.002834884 - Q.mean_phi)
    if Q.sum_z_dr < 0.05464922 and Q.z_5 > 0.03672711:
        z += -232.5618 * (0.05464922 - Q.sum_z_dr) * (Q.z_5 - 0.03672711)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -934.6768 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.tau1 < 0.04369778 and Q.eccentricity > 0.9031255:
        z += -538.7768 * (0.04369778 - Q.tau1) * (Q.eccentricity - 0.9031255)
    if Q.e3 < 2.371297e-05 and Q.eccentricity > 0.8319502:
        z += 108056.5 * (2.371297e-05 - Q.e3) * (Q.eccentricity - 0.8319502)
    if Q.sj3_dr_max < 0.1070199 and Q.abseta_1 > 0.03625488:
        z += -1592.441 * (0.1070199 - Q.sj3_dr_max) * (Q.abseta_1 - 0.03625488)
    if Q.centroid_offset < 0.01837778 and Q.abseta_1 < 0.07073975:
        z += 371.9946 * (0.01837778 - Q.centroid_offset) * (0.07073975 - Q.abseta_1)
    if Q.tau1 < 0.04369778 and Q.mean_phi < -0.01275329:
        z += 3658.659 * (0.04369778 - Q.tau1) * (-0.01275329 - Q.mean_phi)
    if Q.tau1 < 0.04369778 and Q.mean_phi > 0.02612796:
        z += 3699.334 * (0.04369778 - Q.tau1) * (Q.mean_phi - 0.02612796)
    if Q.centroid_offset < 0.01837778 and Q.tau4 > 0.001224244:
        z += -7415.586 * (0.01837778 - Q.centroid_offset) * (Q.tau4 - 0.001224244)
    if Q.sum_z_dr2 < 0.00609665 and Q.mean_phi > 0.02612796:
        z += -46215.02 * (0.00609665 - Q.sum_z_dr2) * (Q.mean_phi - 0.02612796)
    if Q.log_sum_pt > 6.896095 and Q.D2_b2 < 0.716559:
        z += -39.08424 * (Q.log_sum_pt - 6.896095) * (0.716559 - Q.D2_b2)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -48.34005 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.sj3_dr_max < 0.1426152 and Q.pt_6 > 33.6875:
        z += 0.1958896 * (0.1426152 - Q.sj3_dr_max) * (Q.pt_6 - 33.6875)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -1.996859 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 1046.392 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.516301
    if Q.sj3_pair_mass_min >= 11.051:
        z += 0.05964441 * Q.sj3_pair_mass_min - 0.6591306
    if Q.lam1 < 0.001503553:
        z += 1566.838 * Q.lam1 - 4.190014
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 526.3262 * Q.lam1 - 2.625549
    if 0.004183811 <= Q.lam1 < 0.00595415:
        z += 239.2198 * Q.lam1 - 1.42435
    if Q.lam1 >= 0.00733008:
        z += -252.353 * Q.lam1 + 1.849768
    if Q.lam2 >= 0.0003061234:
        z += 1083.051 * Q.lam2 - 0.3315471
    if Q.sj3_dr_min >= 0.1278212:
        z += 7.994828 * Q.sj3_dr_min - 1.021908
    if Q.sum_pt >= 988.4078:
        z += -0.009230645 * Q.sum_pt + 9.123642
    if Q.z_7 < 0.06473447:
        z += 6.474507 * Q.z_7 - 0.4191238
    if Q.LHA >= 0.3033137:
        z += -34.3947 * Q.LHA + 10.43239
    if Q.tau1 >= 0.05356915:
        z += 20.2685 * Q.tau1 - 1.085766
    if Q.sum_z_dr2 < 0.002635418:
        z += 138.6491 * Q.sum_z_dr2 - 0.3653982
    if 0.007520088 <= Q.sum_z_dr2 < 0.02530566:
        z += 522.5635 * Q.sum_z_dr2 - 3.929724
    if Q.sum_z_dr2 >= 0.02530566:
        z += 454.9939 * Q.sum_z_dr2 - 2.219828
    if Q.e3 >= 3.892127e-05:
        z += -2521.194 * Q.e3 + 0.09812807
    if Q.mass < 76.6557:
        z += -0.0251127 * Q.mass + 1.925032
    if Q.sum_z_dr2_top3 < 0.002151568:
        z += -304.4479 * Q.sum_z_dr2_top3 + 0.6550403
    if Q.sj3_dr_max >= 0.1986272:
        z += -3.402056 * Q.sj3_dr_max + 0.6757409
    if Q.C2_b2 >= 0.009032972:
        z += 65.0564 * Q.C2_b2 - 0.5876526
    if Q.pt_7 < 45.75 and Q.D2 < 1.002471:
        z += 0.07084734 * (45.75 - Q.pt_7) * (1.002471 - Q.D2)
    if Q.zdr_0 < 0.0211821 and Q.centroid_offset > 0.01258764:
        z += 1549.307 * (0.0211821 - Q.zdr_0) * (Q.centroid_offset - 0.01258764)
    if Q.pt_7 < 45.75 and Q.log_sum_pt < 6.572938:
        z += -0.1027272 * (45.75 - Q.pt_7) * (6.572938 - Q.log_sum_pt)
    if Q.lam2 > 0.0003061234 and Q.planar_flow > 0.04505724:
        z += -897.7303 * (Q.lam2 - 0.0003061234) * (Q.planar_flow - 0.04505724)
    if Q.sj3_dr_min > 0.1278212 and Q.z_dr_0_0p05 < 0.3658817:
        z += -23.78906 * (Q.sj3_dr_min - 0.1278212) * (0.3658817 - Q.z_dr_0_0p05)
    if Q.zdr_0 < 0.0211821 and Q.z_dr_0p05_0p1 < 0.2919447:
        z += 130.5715 * (0.0211821 - Q.zdr_0) * (0.2919447 - Q.z_dr_0p05_0p1)
    if Q.e3 < 8.147744e-05 and Q.sj3_dr23 > 0.1797097:
        z += -78471.88 * (8.147744e-05 - Q.e3) * (Q.sj3_dr23 - 0.1797097)
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -246.5854 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    if Q.lam1 > 0.00733008 and Q.D2_b2 > 1.129616:
        z += -81.95798 * (Q.lam1 - 0.00733008) * (Q.D2_b2 - 1.129616)
    return max(0.0, z)


def neuron_11(Q):
    z = -3.641458
    if Q.planar_flow < 0.2534037:
        z += -5.669355 * Q.planar_flow + 1.436636
    if Q.mass < 15.45403:
        z += -0.10125 * Q.mass + 0.09423847
    if 15.45403 <= Q.mass < 49.6681:
        z += 0.02505271 * Q.mass - 1.857647
    if 49.6681 <= Q.mass < 69.61135:
        z += 0.03075361 * Q.mass - 2.1408
    if Q.sum_z_dr2 < 0.004372139:
        z += -389.3095 * Q.sum_z_dr2 + 6.246831
    if 0.004372139 <= Q.sum_z_dr2 < 0.01323868:
        z += -512.5695 * Q.sum_z_dr2 + 6.785741
    if Q.LHA < 0.3127275:
        z += 13.82314 * Q.LHA - 4.322877
    if Q.centroid_offset < 0.01437952:
        z += -42.09119 * Q.centroid_offset + 3.165942
    if 0.01437952 <= Q.centroid_offset < 0.03776099:
        z += -109.518 * Q.centroid_offset + 4.135507
    if Q.centroid_offset >= 0.04990367:
        z += -1567.02 * Q.centroid_offset + 78.20007
    if Q.lam1 < 0.0005049491:
        z += 2297.141 * Q.lam1 - 6.478519
    if 0.0005049491 <= Q.lam1 < 0.00733008:
        z += 706.7441 * Q.lam1 - 5.67545
    if 0.00733008 <= Q.lam1 < 0.008375572:
        z += 473.4221 * Q.lam1 - 3.965181
    if Q.sj3_dr_max < 0.1426152:
        z += 17.65271 * Q.sj3_dr_max - 2.501832
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 24.5713 * Q.sj3_dr_max - 3.488528
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -7.12561 * Q.sj3_dr_max + 1.86917
    if Q.lam1_plus_lam2 < 0.008678045:
        z += -1538.058 * Q.lam1_plus_lam2 + 13.34733
    if Q.sum_z_dr < 0.02054282:
        z += 88.88381 * Q.sum_z_dr - 2.852038
    if 0.02054282 <= Q.sum_z_dr < 0.0717028:
        z += 20.05698 * Q.sum_z_dr - 1.438141
    if Q.sum_zz_dr2 < 0.008168571:
        z += 604.6688 * Q.sum_zz_dr2 - 4.93928
    if Q.z_7 >= 0.01685855:
        z += 38.99413 * Q.z_7 - 0.6573842
    if Q.sj2_dr >= 0.2687922:
        z += 19.41319 * Q.sj2_dr - 5.218112
    if Q.pt_7 >= 29.04219:
        z += -0.04047296 * Q.pt_7 + 1.175423
    if Q.max_dr < 0.1452311:
        z += -7.582025 * Q.max_dr + 1.101146
    if Q.planar_flow < 0.2534037 and Q.sum_pt < 840.0195:
        z += -0.01371146 * (0.2534037 - Q.planar_flow) * (840.0195 - Q.sum_pt)
    if Q.planar_flow < 0.2534037 and Q.pt_7 < 37.15625:
        z += -0.1074287 * (0.2534037 - Q.planar_flow) * (37.15625 - Q.pt_7)
    if Q.planar_flow < 0.2534037 and Q.e3 < 1.050302e-05:
        z += -309251.0 * (0.2534037 - Q.planar_flow) * (1.050302e-05 - Q.e3)
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.1897212 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset < 0.03776099 and Q.mean_phi < -0.001813533:
        z += -1394.524 * (0.03776099 - Q.centroid_offset) * (-0.001813533 - Q.mean_phi)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -422.5316 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset < 0.03776099 and Q.absphi_0 < 0.0345459:
        z += -552.4459 * (0.03776099 - Q.centroid_offset) * (0.0345459 - Q.absphi_0)
    if Q.centroid_offset < 0.01437952 and Q.D2_b2 < 0.5327104:
        z += 111.3978 * (0.01437952 - Q.centroid_offset) * (0.5327104 - Q.D2_b2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += 1218.613 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.7535439
    if Q.sum_z_dr2 >= 0.01882765:
        z += 205.1463 * Q.sum_z_dr2 - 3.862423
    if Q.mass >= 91.19:
        z += 0.06358833 * Q.mass - 5.79862
    if Q.zdr_0 >= 0.03981924:
        z += -41.11686 * Q.zdr_0 + 1.637242
    if Q.e2 >= 0.06344108:
        z += -23.6332 * Q.e2 + 1.499315
    if Q.centroid_offset >= 0.04990367:
        z += 26.40147 * Q.centroid_offset - 1.31753
    if Q.sum_z_dr2 > 0.01882765 and Q.pt_7 < 53.4375:
        z += -1.469647 * (Q.sum_z_dr2 - 0.01882765) * (53.4375 - Q.pt_7)
    if Q.mass > 69.61135 and Q.centroid_offset > 0.009480685:
        z += 0.7260051 * (Q.mass - 69.61135) * (Q.centroid_offset - 0.009480685)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.3375261
    if Q.sum_z_dr < 0.1484084:
        z += -56.47434 * Q.sum_z_dr + 8.381267
    if Q.lam1 < 0.01643375:
        z += -156.4355 * Q.lam1 + 2.570822
    if 658.125 <= Q.sum_pt_top5 < 791.125:
        z += -0.006274273 * Q.sum_pt_top5 + 4.129256
    if Q.sum_pt_top5 >= 791.125:
        z += 0.001555798 * Q.sum_pt_top5 - 2.065309
    if Q.e3 < 5.334511e-05:
        z += -21372.44 * Q.e3 + 1.140115
    if Q.e2 < 0.08000524:
        z += 37.82717 * Q.e2 - 3.026372
    if Q.mass >= 49.6681:
        z += -0.02115762 * Q.mass + 1.050859
    if Q.z_5 < 0.02818362:
        z += 193.5907 * Q.z_5 - 5.456086
    if Q.z_7 < 0.02807091:
        z += 160.1448 * Q.z_7 - 4.495412
    if Q.sj3_dr23 >= 0.1974628:
        z += -2.639234 * Q.sj3_dr23 + 0.5211506
    if Q.pt_7 >= 48.71875:
        z += -0.05007015 * Q.pt_7 + 2.439355
    if Q.lam1_plus_lam2 < 0.007520088:
        z += -14.16546 * Q.lam1_plus_lam2 + 0.4986749
    if 0.007520088 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -68.57453 * Q.lam1_plus_lam2 + 0.9078359
    if Q.z_6 < 0.02160287:
        z += 164.9247 * Q.z_6 - 3.562847
    if Q.sum_z_dr < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -67.60007 * (0.1484084 - Q.sum_z_dr) * (6.804164 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.1484084 and Q.pt_7 < 38.53125:
        z += -1.000751 * (0.1484084 - Q.sum_z_dr) * (38.53125 - Q.pt_7)
    if Q.e3 < 5.334511e-05 and Q.centroid_offset < 0.03776099:
        z += -881423.1 * (5.334511e-05 - Q.e3) * (0.03776099 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.0009375463 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.pt_6 < 31.90625 and Q.z_7 < 0.0586137:
        z += -3.142686 * (31.90625 - Q.pt_6) * (0.0586137 - Q.z_7)
    if Q.sum_z_dr < 0.1484084 and Q.z_7 > 0.06164517:
        z += 312.6334 * (0.1484084 - Q.sum_z_dr) * (Q.z_7 - 0.06164517)
    if Q.sum_z_dr < 0.1484084 and Q.lam2 < 0.000537286:
        z += -9726.966 * (0.1484084 - Q.sum_z_dr) * (0.000537286 - Q.lam2)
    if Q.sum_z_dr < 0.1484084 and Q.sj2_mass1 > 31.78116:
        z += -1.988578 * (0.1484084 - Q.sum_z_dr) * (Q.sj2_mass1 - 31.78116)
    if Q.sum_z_dr < 0.1484084 and Q.M3 < 0.07474969:
        z += 188.1194 * (0.1484084 - Q.sum_z_dr) * (0.07474969 - Q.M3)
    if Q.sum_z_dr < 0.1484084 and Q.tau2 > 0.008780509:
        z += 428.6711 * (0.1484084 - Q.sum_z_dr) * (Q.tau2 - 0.008780509)
    if Q.sum_pt_top5 > 791.125 and Q.pt_6 < 31.90625:
        z += 0.001080896 * (Q.sum_pt_top5 - 791.125) * (31.90625 - Q.pt_6)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.000809336 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -2.44623
    if Q.planar_flow < 0.1115136:
        z += -9.031541 * Q.planar_flow + 1.007139
    if Q.z_dr_0p05_0p1 < 0.5882598:
        z += -0.865549 * Q.z_dr_0p05_0p1 + 0.5091677
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -22.31862 * Q.z_dr_0p05_0p1 + 16.75926
    if 0.7143804 <= Q.psi_0p1 < 0.8155839:
        z += -1.232298 * Q.psi_0p1 + 0.8803292
    if 0.8155839 <= Q.psi_0p1 < 0.9761279:
        z += 1.342556 * Q.psi_0p1 - 1.21968
    if Q.psi_0p1 >= 0.9761279:
        z += -18.26062 * Q.psi_0p1 + 17.91553
    if Q.sum_z_dr2 >= 0.007520088:
        z += -1117.4 * Q.sum_z_dr2 + 8.40295
    if 0.002074109 <= Q.sum_zz_dr2 < 0.0030133:
        z += 207.4937 * Q.sum_zz_dr2 - 0.4303646
    if 0.0030133 <= Q.sum_zz_dr2 < 0.01165737:
        z += 58.18883 * Q.sum_zz_dr2 + 0.01953585
    if Q.sum_zz_dr2 >= 0.01165737:
        z += 215.4124 * Q.sum_zz_dr2 - 1.813279
    if 0.003562611 <= Q.lam1_plus_lam2 < 0.005590289:
        z += 357.6237 * Q.lam1_plus_lam2 - 1.274074
    if 0.005590289 <= Q.lam1_plus_lam2 < 0.006679471:
        z += 420.9751 * Q.lam1_plus_lam2 - 1.628227
    if 0.006679471 <= Q.lam1_plus_lam2 < 0.01323868:
        z += -1374.425 * Q.lam1_plus_lam2 + 10.3641
    if Q.lam1_plus_lam2 >= 0.01323868:
        z += -1084.734 * Q.lam1_plus_lam2 + 6.528971
    if 0.01655442 <= Q.e2 < 0.04110972:
        z += -205.2405 * Q.e2 + 3.397637
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += -155.0891 * Q.e2 + 1.335927
    if Q.e2 >= 0.05028464:
        z += 45.93097 * Q.e2 - 8.772294
    if 0.02685622 <= Q.centroid_offset < 0.04990367:
        z += -75.89097 * Q.centroid_offset + 2.038145
    if Q.centroid_offset >= 0.04990367:
        z += -393.7976 * Q.centroid_offset + 17.90285
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 134.7627 * Q.mass_over_sum_pt - 10.77074
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 337.4042 * Q.mass_over_sum_pt - 27.94493
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 374.7745 * Q.mass_over_sum_pt - 31.32372
    if Q.n_dr_0_0p05 < 5.0:
        z += -0.1327743 * Q.n_dr_0_0p05 + 0.6638713
    if Q.e3 < 5.334511e-05:
        z += 21344.95 * Q.e3 - 1.138649
    if Q.sd_mass < 49.91626:
        z += -0.004277566 * Q.sd_mass - 0.4834978
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.0282647 * Q.sd_mass - 2.107886
    if 0.02689598 <= Q.sum_z_dr < 0.04081947:
        z += 119.6301 * Q.sum_z_dr - 3.217569
    if 0.04081947 <= Q.sum_z_dr < 0.08065885:
        z += 169.8302 * Q.sum_z_dr - 5.266712
    if 0.08065885 <= Q.sum_z_dr < 0.08723651:
        z += 231.9421 * Q.sum_z_dr - 10.27658
    if Q.sum_z_dr >= 0.08723651:
        z += -27.99266 * Q.sum_z_dr + 12.39922
    if Q.mass >= 69.61135:
        z += -0.03316609 * Q.mass + 2.308736
    if 0.06154135 <= Q.sj2_dr < 0.1294903:
        z += -9.490188 * Q.sj2_dr + 0.584039
    if 0.1294903 <= Q.sj2_dr < 0.1591713:
        z += -11.81595 * Q.sj2_dr + 0.885203
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 37.04502 * Q.sj2_dr - 6.892062
    if Q.sj2_dr >= 0.2001708:
        z += 12.51832 * Q.sj2_dr - 1.982533
    if Q.lam2 < 0.000537286:
        z += -1005.737 * Q.lam2 + 0.5403687
    if Q.C2_b2 < 0.004032342:
        z += 204.0888 * Q.C2_b2 - 0.8229558
    if Q.LHA >= 0.3033137:
        z += 7.768226 * Q.LHA - 2.35621
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -3553.208 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.01643375:
        z += 117.2272 * (0.1115136 - Q.planar_flow) * (0.01643375 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.lam1_plus_lam2 < 0.00609665:
        z += 3811.947 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.lam1_plus_lam2)
    if Q.planar_flow < 0.1115136 and Q.sum_pt_top5 < 658.125:
        z += -0.02756079 * (0.1115136 - Q.planar_flow) * (658.125 - Q.sum_pt_top5)
    if Q.planar_flow < 0.1115136 and Q.centroid_offset < 0.01837778:
        z += -502.0964 * (0.1115136 - Q.planar_flow) * (0.01837778 - Q.centroid_offset)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 21.55082 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.e3 < 5.334511e-05 and Q.sj3_dr23 > 0.1629004:
        z += 240585.9 * (5.334511e-05 - Q.e3) * (Q.sj3_dr23 - 0.1629004)
    if Q.sj2_dr > 0.2001708 and Q.z_5 < 0.1058993:
        z += -96.79292 * (Q.sj2_dr - 0.2001708) * (0.1058993 - Q.z_5)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.659285
    if Q.sum_z_dr2 < 0.007520088:
        z += 708.529 * Q.sum_z_dr2 - 5.328201
    if Q.mass_over_sum_pt < 0.1309286:
        z += 32.85189 * Q.mass_over_sum_pt - 4.301254
    if Q.lam1_plus_lam2 < 0.01323868:
        z += -1216.787 * Q.lam1_plus_lam2 + 17.98098
    if 0.01323868 <= Q.lam1_plus_lam2 < 0.01882765:
        z += -335.0041 * Q.lam1_plus_lam2 + 6.307341
    if Q.e2 < 0.04110972:
        z += -214.5067 * Q.e2 + 8.818308
    if Q.sum_zz_dr2 < 0.008168571:
        z += 1213.156 * Q.sum_zz_dr2 - 11.11856
    if 0.008168571 <= Q.sum_zz_dr2 < 0.01165737:
        z += 346.4836 * Q.sum_zz_dr2 - 4.039089
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -559.9161 * Q.mass_over_sum_pt_sq + 4.021786
    if Q.lam1 < 0.00483998:
        z += -188.1439 * Q.lam1 + 0.9106129
    if Q.lam2 < 0.001130645:
        z += 751.4535 * Q.lam2 - 0.8496269
    if Q.sum_z_dr < 0.1019409:
        z += 125.3796 * Q.sum_z_dr - 12.78132
    if Q.sj2_dr < 0.1492731:
        z += -12.86209 * Q.sj2_dr + 0.7768483
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -16.27531 * Q.sj2_dr + 1.286351
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 31.81045 * Q.sj2_dr - 6.367523
    if Q.sum_z_dr2_top2 < 0.0005124533:
        z += 902.4391 * Q.sum_z_dr2_top2 - 0.4624579
    if Q.N2 < 0.2233283 and Q.z_dr_0p05_0p1 < 0.5882598:
        z += -11.92653 * (0.2233283 - Q.N2) * (0.5882598 - Q.z_dr_0p05_0p1)
    if Q.sum_z_dr2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -1129.332 * (0.007520088 - Q.sum_z_dr2) * (0.7459513 - Q.D2)
    if Q.lam1_plus_lam2 < 0.00609665 and Q.D2 < 0.7459513:
        z += -3142.879 * (0.00609665 - Q.lam1_plus_lam2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 37.55319 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.N2 < 0.2233283 and Q.n_dr_0p1_0p2 < 4.0:
        z += 1.063889 * (0.2233283 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.01491348 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
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
