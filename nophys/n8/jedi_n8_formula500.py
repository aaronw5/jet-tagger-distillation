"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the sum_z_dr)
  Q.zdr_3                  pT share × ΔR of particle 3 (its part of the sum_z_dr)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the sum_z_dr)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the sum_z_dr)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the sum_z_dr)
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
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -0.483
    if Q.C2_b2 < 0.00158:
        z += 775.0 * Q.C2_b2 - 1.2245
    if 0.00711 <= Q.centroid_offset < 0.0499:
        z += -45.5 * Q.centroid_offset + 0.323505
    if Q.centroid_offset >= 0.0499:
        z += -672.5 * Q.centroid_offset + 31.6108
    if Q.e2 < 0.0248:
        z += -176.4 * Q.e2 + 5.21146
    if 0.0248 <= Q.e2 < 0.0355:
        z += -78.2 * Q.e2 + 2.7761
    if Q.sum_z_dr < 0.0753:
        z += 52.2 * Q.sum_z_dr - 3.93066
    if Q.sum_z_dr2 < 0.0132:
        z += -292.0 * Q.sum_z_dr2 + 3.8544
    if Q.lam1 < 0.000264:
        z += -12600.0 * Q.lam1 + 3.3264
    if Q.lam2 < 7.82e-05:
        z += 9140.0 * Q.lam2 - 0.714748
    if Q.log_sum_pt < 6.11:
        z += 6.38 * Q.log_sum_pt - 38.9818
    if Q.log_sum_pt >= 6.36:
        z += 2.12 * Q.log_sum_pt - 13.4832
    if Q.mass < 59.3:
        z += 0.0318 * Q.mass - 1.88574
    if Q.mass_over_sum_pt < 0.0847:
        z += 49.0 * Q.mass_over_sum_pt - 4.1503
    if Q.planar_flow < 0.15:
        z += -10.9 * Q.planar_flow + 1.635
    if 0.11 <= Q.sj3_dr_max < 0.183:
        z += 10.2 * Q.sj3_dr_max - 1.122
    if Q.sj3_dr_max >= 0.183:
        z += -16.0 * Q.sj3_dr_max + 3.6726
    if Q.sum_pt >= 906.0:
        z += -0.0137 * Q.sum_pt + 12.4122
    if Q.lam1_plus_lam2 < 0.00449:
        z += 718.0 * Q.lam1_plus_lam2 - 1.67258
    if 0.00449 <= Q.lam1_plus_lam2 < 0.00866:
        z += -372.0 * Q.lam1_plus_lam2 + 3.22152
    if Q.centroid_offset > 0.00817 and Q.n_pt_above_50 < 6.81:
        z += 4.35 * (Q.centroid_offset - 0.00817) * (6.81 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.0198 and Q.z_7 < 0.0628:
        z += -1550.0 * (Q.centroid_offset - 0.0198) * (0.0628 - Q.z_7)
    if Q.sum_z_dr2 < 0.0132 and Q.D2 < 1.04:
        z += 152.0 * (0.0132 - Q.sum_z_dr2) * (1.04 - Q.D2)
    if Q.sum_z_dr2 < 0.0141 and Q.phi_0 > 0.000879:
        z += 593.0 * (0.0141 - Q.sum_z_dr2) * (Q.phi_0 - 0.000879)
    if Q.lam1 < 0.00631 and Q.D2 < 0.837:
        z += -1290.0 * (0.00631 - Q.lam1) * (0.837 - Q.D2)
    if Q.lam2 < 7.64e-05 and Q.D2_b2 < 0.266:
        z += 56800.0 * (7.64e-05 - Q.lam2) * (0.266 - Q.D2_b2)
    if Q.log_sum_pt > 6.68 and Q.dr_4 < 0.0755:
        z += 61.5 * (Q.log_sum_pt - 6.68) * (0.0755 - Q.dr_4)
    if Q.log_sum_pt > 6.38 and Q.mean_eta > 0.000559:
        z += -45.0 * (Q.log_sum_pt - 6.38) * (Q.mean_eta - 0.000559)
    if Q.log_sum_pt > 6.67 and Q.z_7 > 0.018:
        z += -438.0 * (Q.log_sum_pt - 6.67) * (Q.z_7 - 0.018)
    if Q.mass < 63.7 and Q.D2_b2 < 0.163:
        z += -0.183 * (63.7 - Q.mass) * (0.163 - Q.D2_b2)
    if Q.planar_flow < 0.149 and Q.dr_2 < 0.0225:
        z += -666.0 * (0.149 - Q.planar_flow) * (0.0225 - Q.dr_2)
    if Q.sj3_dr_max < 0.274 and Q.z_7 < 0.0577:
        z += -209.0 * (0.274 - Q.sj3_dr_max) * (0.0577 - Q.z_7)
    if Q.lam1_plus_lam2 < 0.00432 and Q.C3 < 0.0369:
        z += 11300.0 * (0.00432 - Q.lam1_plus_lam2) * (0.0369 - Q.C3)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.04
    if Q.sum_z_dr < 0.0962:
        z += 29.8 * Q.sum_z_dr - 2.86676
    if 6.39 <= Q.log_sum_pt < 6.59:
        z += 7.34 * Q.log_sum_pt - 46.9026
    if Q.log_sum_pt >= 6.59:
        z += 14.44 * Q.log_sum_pt - 93.6916
    if Q.mass_over_sum_pt_sq < 0.00605:
        z += -578.0 * Q.mass_over_sum_pt_sq + 3.4969
    if Q.pt_7 >= 55.0:
        z += -0.0776 * Q.pt_7 + 4.268
    if Q.sj3_dr_max >= 0.171:
        z += 3.07 * Q.sj3_dr_max - 0.52497
    if Q.sum_pt_top5 >= 524.0:
        z += -0.00622 * Q.sum_pt_top5 + 3.25928
    if Q.lam1_plus_lam2 < 0.00669:
        z += 448.0 * Q.lam1_plus_lam2 - 2.99712
    if Q.z_7 < 0.0475:
        z += 31.9 * Q.z_7 - 1.51525
    if Q.z_7 >= 0.0481:
        z += 9.04 * Q.z_7 - 0.434824
    if Q.lam1 < 0.00874 and Q.n_pt_above_50 > 4.82:
        z += -23.7 * (0.00874 - Q.lam1) * (Q.n_pt_above_50 - 4.82)
    if Q.lam1 < 0.0103 and Q.planar_flow < 0.145:
        z += -358.0 * (0.0103 - Q.lam1) * (0.145 - Q.planar_flow)
    if Q.log_sum_pt > 6.5 and Q.D2 < 1.49:
        z += 2.12 * (Q.log_sum_pt - 6.5) * (1.49 - Q.D2)
    if Q.log_sum_pt > 6.32 and Q.centroid_offset > 0.0126:
        z += 86.7 * (Q.log_sum_pt - 6.32) * (Q.centroid_offset - 0.0126)
    if Q.log_sum_pt > 6.38 and Q.z_dr_0p05_0p1 < 0.74:
        z += -3.48 * (Q.log_sum_pt - 6.38) * (0.74 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt_sq < 0.00606 and Q.D2_b2 < 0.0392:
        z += -23500.0 * (0.00606 - Q.mass_over_sum_pt_sq) * (0.0392 - Q.D2_b2)
    if Q.pt_7 > 35.0 and Q.n_dr_0p1_0p2 > 0.551:
        z += 0.00832 * (Q.pt_7 - 35.0) * (Q.n_dr_0p1_0p2 - 0.551)
    if Q.pt_7 > 32.5 and Q.pt_6 < 53.1:
        z += -0.00323 * (Q.pt_7 - 32.5) * (53.1 - Q.pt_6)
    if Q.pt_7 > 33.3 and Q.sj2_dr > 0.136:
        z += 0.426 * (Q.pt_7 - 33.3) * (Q.sj2_dr - 0.136)
    if Q.sj3_dr_max > 0.139 and Q.sj3_pair_mass_min > 1.38:
        z += -0.113 * (Q.sj3_dr_max - 0.139) * (Q.sj3_pair_mass_min - 1.38)
    if Q.z_7 < 0.0626 and Q.D2 < 1.62:
        z += -24.8 * (0.0626 - Q.z_7) * (1.62 - Q.D2)
    if Q.z_7 < 0.0609 and Q.z_dr_0p05_0p1 > 0.09:
        z += -45.7 * (0.0609 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.09)
    return max(0.0, z)


def neuron_2(Q):
    z = 4.11
    if Q.LHA >= 0.112:
        z += -10.2 * Q.LHA + 1.1424
    if Q.centroid_offset < 0.00464:
        z += 86.4 * Q.centroid_offset - 0.400896
    if Q.dr_5 < 0.0219:
        z += 18.3 * Q.dr_5 - 0.40077
    if Q.sum_z_dr < 0.0078:
        z += 186.0 * Q.sum_z_dr - 1.4508
    if Q.sum_z_dr2 < 0.00431:
        z += -146.0 * Q.sum_z_dr2 + 0.62926
    if Q.lam1 < 0.0118:
        z += -104.0 * Q.lam1 + 1.2272
    if Q.log_sum_pt < 6.59:
        z += -4.48 * Q.log_sum_pt + 29.5232
    if Q.log_sum_pt >= 6.85:
        z += 21.2 * Q.log_sum_pt - 145.22
    if Q.m012 >= 42.8:
        z += 0.0392 * Q.m012 - 1.67776
    if Q.pt_6 >= 27.2:
        z += 0.0201 * Q.pt_6 - 0.54672
    if Q.pt_7 < 43.7:
        z += 0.0969 * Q.pt_7 - 5.16477
    if 43.7 <= Q.pt_7 < 53.3:
        z += 0.157 * Q.pt_7 - 7.79114
    if Q.pt_7 >= 53.3:
        z += 0.0601 * Q.pt_7 - 2.62637
    if Q.sj3_pair_mass_max >= 28.4:
        z += -0.0167 * Q.sj3_pair_mass_max + 0.47428
    if Q.sum_pt < 798.0:
        z += -0.0111 * Q.sum_pt + 8.8578
    if Q.sum_pt_top5 < 608.0:
        z += 0.00706 * Q.sum_pt_top5 - 5.06908
    if 608.0 <= Q.sum_pt_top5 < 718.0:
        z += 0.00182 * Q.sum_pt_top5 - 1.88316
    if 718.0 <= Q.sum_pt_top5 < 835.0:
        z += -0.00524 * Q.sum_pt_top5 + 3.18592
    if Q.sum_pt_top5 >= 835.0:
        z += -0.01303 * Q.sum_pt_top5 + 9.69057
    if Q.z_7 < 0.0532:
        z += -58.6 * Q.z_7 + 3.12924
    if 0.0532 <= Q.z_7 < 0.0534:
        z += -105.2 * Q.z_7 + 5.60836
    if Q.z_7 >= 0.0534:
        z += -46.6 * Q.z_7 + 2.47912
    if Q.sum_z_dr < 0.00794 and Q.pt_4 < 72.3:
        z += 4.75 * (0.00794 - Q.sum_z_dr) * (72.3 - Q.pt_4)
    if Q.lam1 < 0.00596 and Q.max_dr > 0.0963:
        z += -1870.0 * (0.00596 - Q.lam1) * (Q.max_dr - 0.0963)
    if Q.log_sum_pt > 6.88 and Q.D2_b2 < 2.99:
        z += -6.21 * (Q.log_sum_pt - 6.88) * (2.99 - Q.D2_b2)
    if Q.log_sum_pt > 6.85 and Q.pt_6 > 43.4:
        z += -0.149 * (Q.log_sum_pt - 6.85) * (Q.pt_6 - 43.4)
    if Q.pt_6 > 31.4 and Q.planar_flow < 0.781:
        z += -0.0157 * (Q.pt_6 - 31.4) * (0.781 - Q.planar_flow)
    if Q.sj3_pair_mass_max < 67.5 and Q.centroid_offset > 0.0124:
        z += -0.39 * (67.5 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.0124)
    if Q.sj3_pair_mass_max < 66.3 and Q.z_7 < 0.065:
        z += -0.409 * (66.3 - Q.sj3_pair_mass_max) * (0.065 - Q.z_7)
    if Q.sum_pt < 588.0 and Q.D2_b2 < 4.15:
        z += 0.001 * (588.0 - Q.sum_pt) * (4.15 - Q.D2_b2)
    if Q.sum_pt < 793.0 and Q.dr_5 < 0.0223:
        z += -0.828 * (793.0 - Q.sum_pt) * (0.0223 - Q.dr_5)
    if Q.sum_pt < 839.0 and Q.dr_5 < 0.0223:
        z += 0.765 * (839.0 - Q.sum_pt) * (0.0223 - Q.dr_5)
    if Q.sum_pt_top5 > 842.0 and Q.D2_b2 < 2.32:
        z += 0.00577 * (Q.sum_pt_top5 - 842.0) * (2.32 - Q.D2_b2)
    if Q.z_7 < 0.0726 and Q.planar_flow < 0.64:
        z += -11.9 * (0.0726 - Q.z_7) * (0.64 - Q.planar_flow)
    if Q.z_7 > 0.0396 and Q.sj3_dr_min < 0.0633:
        z += -132.0 * (Q.z_7 - 0.0396) * (0.0633 - Q.sj3_dr_min)
    return max(0.0, z)


def neuron_3(Q):
    z = -4.33
    if Q.e2 >= 0.0628:
        z += 158.0 * Q.e2 - 9.9224
    if 0.0382 <= Q.sum_z_dr < 0.077:
        z += 66.8 * Q.sum_z_dr - 2.55176
    if Q.sum_z_dr >= 0.077:
        z += 156.4 * Q.sum_z_dr - 9.45096
    if Q.sum_z_dr2 < 0.00529:
        z += 157.0 * Q.sum_z_dr2 - 0.83053
    if Q.sum_z_dr2 >= 0.0087:
        z += -1180.0 * Q.sum_z_dr2 + 10.266
    if 0.0081 <= Q.lam1 < 0.016:
        z += 670.0 * Q.lam1 - 5.427
    if Q.lam1 >= 0.016:
        z += 1066.0 * Q.lam1 - 11.763
    if Q.mass >= 65.4:
        z += -0.0696 * Q.mass + 4.55184
    if Q.mass_over_sum_pt >= 0.0707:
        z += 58.5 * Q.mass_over_sum_pt - 4.13595
    if Q.max_dr >= 0.103:
        z += 11.8 * Q.max_dr - 1.2154
    if Q.mean_eta < -0.00445:
        z += -25.5 * Q.mean_eta - 0.113475
    if Q.mean_eta >= 0.018:
        z += 41.4 * Q.mean_eta - 0.7452
    if Q.sj2_dr >= 0.188:
        z += 24.7 * Q.sj2_dr - 4.6436
    if Q.tau1 >= 0.053:
        z += -38.7 * Q.tau1 + 2.0511
    if Q.lam1_plus_lam2 >= 0.0133:
        z += -638.0 * Q.lam1_plus_lam2 + 8.4854
    if Q.z_dr_0_0p05 < 0.069:
        z += -7.86 * Q.z_dr_0_0p05 + 0.54234
    if Q.centroid_offset > 0.0107 and Q.abseta_0 < 0.0822:
        z += 554.0 * (Q.centroid_offset - 0.0107) * (0.0822 - Q.abseta_0)
    if Q.sum_z_dr > 0.0415 and Q.log_sum_pt > 6.12:
        z += 79.9 * (Q.sum_z_dr - 0.0415) * (Q.log_sum_pt - 6.12)
    if Q.lam2 > 0.000916 and Q.pt_6 < 56.4:
        z += 27.6 * (Q.lam2 - 0.000916) * (56.4 - Q.pt_6)
    if Q.lam2 > 0.00162 and Q.z_6 > 0.0476:
        z += 10200.0 * (Q.lam2 - 0.00162) * (Q.z_6 - 0.0476)
    if Q.mass_over_sum_pt > 0.0659 and Q.sj2_dr < 0.219:
        z += -1350.0 * (Q.mass_over_sum_pt - 0.0659) * (0.219 - Q.sj2_dr)
    if Q.max_dr > 0.112 and Q.zdr_1 < 0.0104:
        z += -731.0 * (Q.max_dr - 0.112) * (0.0104 - Q.zdr_1)
    if Q.sj2_dr > 0.186 and Q.dr_3 < 0.0524:
        z += -344.0 * (Q.sj2_dr - 0.186) * (0.0524 - Q.dr_3)
    if Q.sj2_dr > 0.187 and Q.sj2_mass1 > 5.18:
        z += -0.459 * (Q.sj2_dr - 0.187) * (Q.sj2_mass1 - 5.18)
    if Q.lam1_plus_lam2 > 0.00678 and Q.sj3_pairmin_over_m > 0.0341:
        z += 369.0 * (Q.lam1_plus_lam2 - 0.00678) * (Q.sj3_pairmin_over_m - 0.0341)
    return max(0.0, z)


def neuron_4(Q):
    z = -4.63
    if Q.C2_b2 < 0.00913:
        z += -287.0 * Q.C2_b2 + 2.62031
    if Q.N2 < 0.221:
        z += -48.2 * Q.N2 + 10.6522
    if Q.sum_zz_dr2 < 0.0119:
        z += -829.0 * Q.sum_zz_dr2 + 11.7076
    if 0.0119 <= Q.sum_zz_dr2 < 0.0174:
        z += -335.0 * Q.sum_zz_dr2 + 5.829
    if Q.e3 < 1.38e-05:
        z += 118000.0 * Q.e3 - 1.6284
    if Q.sum_z_dr >= 0.0543:
        z += -43.9 * Q.sum_z_dr + 2.38377
    if Q.sum_z_dr2 < 0.00251:
        z += -342.0 * Q.sum_z_dr2 - 4.96186
    if 0.00251 <= Q.sum_z_dr2 < 0.00353:
        z += 908.0 * Q.sum_z_dr2 - 8.09936
    if 0.00353 <= Q.sum_z_dr2 < 0.00892:
        z += 1248.0 * Q.sum_z_dr2 - 9.29956
    if Q.sum_z_dr2 >= 0.00892:
        z += 340.0 * Q.sum_z_dr2 - 1.2002
    if Q.sum_z_dr2_top5 < 0.00812:
        z += -192.0 * Q.sum_z_dr2_top5 + 1.55904
    if Q.lam2 < 0.000532:
        z += 5060.0 * Q.lam2 - 3.64982
    if 0.000532 <= Q.lam2 < 0.00115:
        z += 1550.0 * Q.lam2 - 1.7825
    if Q.max_dr >= 0.239:
        z += 20.9 * Q.max_dr - 4.9951
    if 0.162 <= Q.sj3_dr_max < 0.232:
        z += 27.1 * Q.sj3_dr_max - 4.3902
    if Q.sj3_dr_max >= 0.232:
        z += -14.8 * Q.sj3_dr_max + 5.3306
    if Q.sum_pt < 762.0:
        z += 0.00445 * Q.sum_pt - 3.3909
    if Q.D2 < 1.1 and Q.pt_7 < 51.6:
        z += -0.0917 * (1.1 - Q.D2) * (51.6 - Q.pt_7)
    if Q.N2 < 0.222 and Q.sum_zz_dr2 > 0.0128:
        z += -2860.0 * (0.222 - Q.N2) * (Q.sum_zz_dr2 - 0.0128)
    if Q.N2 < 0.221 and Q.eccentricity > 0.696:
        z += -94.8 * (0.221 - Q.N2) * (Q.eccentricity - 0.696)
    if Q.N2 < 0.212 and Q.mass < 53.1:
        z += -0.586 * (0.212 - Q.N2) * (53.1 - Q.mass)
    if Q.centroid_offset < 0.018 and Q.z_dr_0p05_0p1 < 0.621:
        z += -116.0 * (0.018 - Q.centroid_offset) * (0.621 - Q.z_dr_0p05_0p1)
    if Q.sum_zz_dr2 < 0.0112 and Q.D2 < 1.05:
        z += 592.0 * (0.0112 - Q.sum_zz_dr2) * (1.05 - Q.D2)
    if Q.mass < 68.1 and Q.D2 < 0.985:
        z += -0.165 * (68.1 - Q.mass) * (0.985 - Q.D2)
    if Q.tau1 > 0.0711 and Q.sj3_dr_min < 0.208:
        z += 292.0 * (Q.tau1 - 0.0711) * (0.208 - Q.sj3_dr_min)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.948
    if Q.C3 < 0.00987:
        z += 190.0 * Q.C3 - 1.8753
    if Q.sum_z_dr2 < 0.00258:
        z += -410.0 * Q.sum_z_dr2 + 1.0578
    if 6.26 <= Q.log_sum_pt < 6.77:
        z += 3.37 * Q.log_sum_pt - 21.0962
    if Q.log_sum_pt >= 6.77:
        z += -9.23 * Q.log_sum_pt + 64.2058
    if Q.mass_over_sum_pt < 0.0896:
        z += 23.1 * Q.mass_over_sum_pt - 2.06976
    if Q.pt_7 < 22.9:
        z += 0.0616 * Q.pt_7 - 2.31616
    if 22.9 <= Q.pt_7 < 37.6:
        z += -0.0167 * Q.pt_7 - 0.52309
    if Q.pt_7 >= 37.6:
        z += -0.0783 * Q.pt_7 + 1.79307
    if Q.sj3_dr_max < 0.109:
        z += 2.9 * Q.sj3_dr_max + 0.4924
    if 0.109 <= Q.sj3_dr_max < 0.186:
        z += -10.5 * Q.sj3_dr_max + 1.953
    if Q.sum_pt_top5 >= 714.0:
        z += 0.00733 * Q.sum_pt_top5 - 5.23362
    if Q.tau1 < 0.0968:
        z += -22.0 * Q.tau1 + 2.1296
    if Q.z_6 < 0.0475:
        z += -54.3 * Q.z_6 + 2.57925
    if Q.z_7 < 0.0283:
        z += -165.0 * Q.z_7 + 4.6695
    if Q.zdr_0 < 0.0239:
        z += 33.5 * Q.zdr_0 - 0.80065
    if Q.LHA < 0.209 and Q.log_sum_pt < 6.82:
        z += -52.8 * (0.209 - Q.LHA) * (6.82 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00172 and Q.centroid_offset < 0.0234:
        z += 103000.0 * (0.00172 - Q.sum_z_dr2) * (0.0234 - Q.centroid_offset)
    if Q.sum_z_dr2_top3 < 0.00272 and Q.phi_1 < 0.0266:
        z += -4950.0 * (0.00272 - Q.sum_z_dr2_top3) * (0.0266 - Q.phi_1)
    if Q.log_sum_pt > 6.27 and Q.C3 < 0.0113:
        z += 542.0 * (Q.log_sum_pt - 6.27) * (0.0113 - Q.C3)
    if Q.log_sum_pt > 6.7 and Q.D2_b2 < 4.61:
        z += -1.33 * (Q.log_sum_pt - 6.7) * (4.61 - Q.D2_b2)
    if Q.log_sum_pt > 6.7 and Q.mean_eta2 < 8.69e-05:
        z += 232000.0 * (Q.log_sum_pt - 6.7) * (8.69e-05 - Q.mean_eta2)
    if Q.log_sum_pt > 6.7 and Q.mean_eta2 < 0.00424:
        z += -1750.0 * (Q.log_sum_pt - 6.7) * (0.00424 - Q.mean_eta2)
    if Q.log_sum_pt > 6.7 and Q.mean_phi > 0.000708:
        z += -675.0 * (Q.log_sum_pt - 6.7) * (Q.mean_phi - 0.000708)
    if Q.log_sum_pt > 6.26 and Q.ptdr0_5 < 7.76:
        z += 0.139 * (Q.log_sum_pt - 6.26) * (7.76 - Q.ptdr0_5)
    if Q.log_sum_pt > 6.91 and Q.sj3_pair_mass_max > 21.9:
        z += -0.231 * (Q.log_sum_pt - 6.91) * (Q.sj3_pair_mass_max - 21.9)
    if Q.log_sum_pt > 6.7 and Q.zdr_3 < 0.00175:
        z += 6310.0 * (Q.log_sum_pt - 6.7) * (0.00175 - Q.zdr_3)
    if Q.log_sum_pt > 6.89 and Q.zdr_3 < 0.00263:
        z += -9410.0 * (Q.log_sum_pt - 6.89) * (0.00263 - Q.zdr_3)
    if Q.max_pair_mass > 20.4 and Q.dr_max_012 < 0.248:
        z += -0.653 * (Q.max_pair_mass - 20.4) * (0.248 - Q.dr_max_012)
    if Q.pt_7 < 36.1 and Q.C3 < 0.0221:
        z += -2.56 * (36.1 - Q.pt_7) * (0.0221 - Q.C3)
    if Q.sum_pt_top5 > 720.0 and Q.mean_eta2 < 8.92e-05:
        z += -173.0 * (Q.sum_pt_top5 - 720.0) * (8.92e-05 - Q.mean_eta2)
    if Q.tau1 < 0.104 and Q.planar_flow < 0.464:
        z += 10.1 * (0.104 - Q.tau1) * (0.464 - Q.planar_flow)
    if Q.z_6 < 0.022 and Q.abseta_6 < 0.167:
        z += 1040.0 * (0.022 - Q.z_6) * (0.167 - Q.abseta_6)
    if Q.z_6 < 0.0463 and Q.zdr_3 < 0.00185:
        z += -27400.0 * (0.0463 - Q.z_6) * (0.00185 - Q.zdr_3)
    if Q.z_7 < 0.0738 and Q.centroid_offset < 0.0304:
        z += -912.0 * (0.0738 - Q.z_7) * (0.0304 - Q.centroid_offset)
    if Q.z_7 < 0.0497 and Q.mass_top5 < 72.9:
        z += 0.813 * (0.0497 - Q.z_7) * (72.9 - Q.mass_top5)
    if Q.zdr_0 < 0.0237 and Q.abseta_0 < 0.0218:
        z += -1070.0 * (0.0237 - Q.zdr_0) * (0.0218 - Q.abseta_0)
    return max(0.0, z)


def neuron_6(Q):
    z = 3.15
    if 0.00779 <= Q.centroid_offset < 0.0187:
        z += 41.8 * Q.centroid_offset - 0.325622
    if Q.centroid_offset >= 0.0187:
        z += -18.7 * Q.centroid_offset + 0.805728
    if Q.sum_z_dr2 < 0.00869:
        z += 1070.0 * Q.sum_z_dr2 - 9.2983
    if Q.lam2 < 0.00114:
        z += 1200.0 * Q.lam2 - 1.368
    if Q.mass_over_sum_pt < 0.0894:
        z += -65.1 * Q.mass_over_sum_pt + 5.81994
    if Q.max_dr < 0.145:
        z += 30.5 * Q.max_dr - 4.4225
    if Q.mean_eta >= 0.0325:
        z += -68.8 * Q.mean_eta + 2.236
    if Q.pt_6 < 19.1:
        z += -0.122 * Q.pt_6 - 0.7293
    if 19.1 <= Q.pt_6 < 40.2:
        z += 0.145 * Q.pt_6 - 5.829
    if Q.sj3_dr_max < 0.18:
        z += -30.54 * Q.sj3_dr_max + 4.45082
    if 0.18 <= Q.sj3_dr_max < 0.189:
        z += 9.26 * Q.sj3_dr_max - 2.71318
    if 0.189 <= Q.sj3_dr_max < 0.293:
        z += 19.56 * Q.sj3_dr_max - 4.65988
    if Q.sj3_dr_max >= 0.293:
        z += 10.3 * Q.sj3_dr_max - 1.9467
    if Q.sj3_dr_min >= 0.0235:
        z += -14.6 * Q.sj3_dr_min + 0.3431
    if Q.sj3_pair_mass_min >= 3.99:
        z += -0.0504 * Q.sj3_pair_mass_min + 0.201096
    if Q.sum_pt < 484.0:
        z += -0.00969 * Q.sum_pt + 4.68996
    if Q.tau1 < 0.118:
        z += -21.7 * Q.tau1 + 2.5606
    if Q.centroid_offset > 0.0183 and Q.mean_phi2 < 0.00887:
        z += 6800.0 * (Q.centroid_offset - 0.0183) * (0.00887 - Q.mean_phi2)
    if Q.centroid_offset > 0.00885 and Q.psi_0p1 > 0.406:
        z += 48.5 * (Q.centroid_offset - 0.00885) * (Q.psi_0p1 - 0.406)
    if Q.centroid_offset > 0.0185 and Q.sum_pt_top5 > 655.0:
        z += 1.29 * (Q.centroid_offset - 0.0185) * (Q.sum_pt_top5 - 655.0)
    if Q.lam1 < 0.00761 and Q.mass_top5 > 66.8:
        z += 106.0 * (0.00761 - Q.lam1) * (Q.mass_top5 - 66.8)
    if Q.lam1 < 0.0132 and Q.planar_flow < 0.265:
        z += -544.0 * (0.0132 - Q.lam1) * (0.265 - Q.planar_flow)
    if Q.lam2 < 0.00333 and Q.n_dr_0_0p05 < 2.87:
        z += 79.6 * (0.00333 - Q.lam2) * (2.87 - Q.n_dr_0_0p05)
    if Q.mass < 66.5 and Q.z_dr_0p2_0p4 < 0.0612:
        z += -0.453 * (66.5 - Q.mass) * (0.0612 - Q.z_dr_0p2_0p4)
    if Q.max_dr < 0.146 and Q.mean_phi > 0.00126:
        z += -153.0 * (0.146 - Q.max_dr) * (Q.mean_phi - 0.00126)
    if Q.pt_6 < 40.8 and Q.M3 > 0.078:
        z += 2.46 * (40.8 - Q.pt_6) * (Q.M3 - 0.078)
    if Q.pt_6 < 41.1 and Q.sum_pt < 995.0:
        z += 0.000846 * (41.1 - Q.pt_6) * (995.0 - Q.sum_pt)
    if Q.pt_6 < 41.1 and Q.z_7 > 0.0225:
        z += -6.04 * (41.1 - Q.pt_6) * (Q.z_7 - 0.0225)
    if Q.sum_pt < 601.0 and Q.n_dr_0p2_0p4 < 2.19:
        z += 0.00273 * (601.0 - Q.sum_pt) * (2.19 - Q.n_dr_0p2_0p4)
    if Q.sum_pt < 617.0 and Q.z_3 < 0.0507:
        z += 10.5 * (617.0 - Q.sum_pt) * (0.0507 - Q.z_3)
    if Q.sum_pt < 761.0 and Q.z_7 < 0.0766:
        z += 0.119 * (761.0 - Q.sum_pt) * (0.0766 - Q.z_7)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.14
    if Q.D2_b2 < 0.0568:
        z += 17.9 * Q.D2_b2 - 1.01672
    if Q.LHA < 0.303:
        z += -11.0 * Q.LHA + 3.333
    if Q.centroid_offset >= 0.0381:
        z += -76.7 * Q.centroid_offset + 2.92227
    if Q.e2 >= 0.0503:
        z += -242.0 * Q.e2 + 12.1726
    if Q.sum_z_dr < 0.0868:
        z += 78.9 * Q.sum_z_dr - 6.84852
    if Q.sum_z_dr2 < 0.00107:
        z += 1320.0 * Q.sum_z_dr2 - 1.4124
    if 0.00462 <= Q.sum_z_dr2 < 0.0135:
        z += -407.0 * Q.sum_z_dr2 + 1.88034
    if Q.sum_z_dr2 >= 0.0135:
        z += -208.0 * Q.sum_z_dr2 - 0.80616
    if Q.lam2 < 0.000261:
        z += 3100.0 * Q.lam2 - 0.8091
    if 35.9 <= Q.mass < 75.9:
        z += 0.0467 * Q.mass - 1.67653
    if Q.mass >= 75.9:
        z += -0.1803 * Q.mass + 15.55277
    if 0.0113 <= Q.mass_over_sum_pt < 0.0732:
        z += -71.3 * Q.mass_over_sum_pt + 0.80569
    if 0.0732 <= Q.mass_over_sum_pt < 0.0842:
        z += 44.7 * Q.mass_over_sum_pt - 7.68551
    if 0.0842 <= Q.mass_over_sum_pt < 0.0902:
        z += 228.7 * Q.mass_over_sum_pt - 23.17831
    if 0.0902 <= Q.mass_over_sum_pt < 0.108:
        z += -54.3 * Q.mass_over_sum_pt + 2.34829
    if Q.mass_over_sum_pt >= 0.108:
        z += -267.3 * Q.mass_over_sum_pt + 25.35229
    if Q.mass_top5 >= 53.4:
        z += 0.0709 * Q.mass_top5 - 3.78606
    if Q.max_dr < 0.111:
        z += 4.85 * Q.max_dr + 0.0885
    if 0.111 <= Q.max_dr < 0.174:
        z += -9.95 * Q.max_dr + 1.7313
    if Q.pt_7 < 29.0:
        z += 0.0459 * Q.pt_7 - 1.3311
    if 0.204 <= Q.sd_rg < 0.279:
        z += -21.7 * Q.sd_rg + 4.4268
    if Q.sd_rg >= 0.279:
        z += 19.0 * Q.sd_rg - 6.9285
    if Q.sj2_dr < 0.158:
        z += -8.9 * Q.sj2_dr + 0.7769
    if 0.158 <= Q.sj2_dr < 0.187:
        z += 21.7 * Q.sj2_dr - 4.0579
    if 0.145 <= Q.sj3_dr_max < 0.265:
        z += 19.2 * Q.sj3_dr_max - 2.784
    if Q.sj3_dr_max >= 0.265:
        z += -0.4 * Q.sj3_dr_max + 2.41
    if Q.tau1 < 0.0549:
        z += -32.1 * Q.tau1 + 1.76229
    if Q.tau21_b2 < 0.0413:
        z += 25.7 * Q.tau21_b2 - 1.06141
    if Q.lam1_plus_lam2 < 0.00559:
        z += 992.0 * Q.lam1_plus_lam2 - 6.696
    if 0.00559 <= Q.lam1_plus_lam2 < 0.00675:
        z += 309.0 * Q.lam1_plus_lam2 - 2.87803
    if Q.lam1_plus_lam2 >= 0.00675:
        z += -683.0 * Q.lam1_plus_lam2 + 3.81797
    if Q.z_7 >= 0.0333:
        z += 11.7 * Q.z_7 - 0.38961
    if Q.z_dr_0p1_0p2 < 0.144:
        z += -1.54 * Q.z_dr_0p1_0p2 + 0.22176
    if Q.centroid_offset < 0.0204 and Q.C2_b2 < 0.0042:
        z += -19500.0 * (0.0204 - Q.centroid_offset) * (0.0042 - Q.C2_b2)
    if Q.centroid_offset > 0.0297 and Q.n_pt_above_50 > 3.89:
        z += -15.8 * (Q.centroid_offset - 0.0297) * (Q.n_pt_above_50 - 3.89)
    if Q.centroid_offset > 0.0378 and Q.pt_4 > 81.4:
        z += -28.6 * (Q.centroid_offset - 0.0378) * (Q.pt_4 - 81.4)
    if Q.centroid_offset < 0.0203 and Q.tau21_b2 < 0.0272:
        z += 2540.0 * (0.0203 - Q.centroid_offset) * (0.0272 - Q.tau21_b2)
    if Q.sum_z_dr < 0.0892 and Q.C2_b2 < 0.00397:
        z += 3690.0 * (0.0892 - Q.sum_z_dr) * (0.00397 - Q.C2_b2)
    if Q.sum_z_dr2 > 0.0133 and Q.dr_6 < 0.0732:
        z += -25600.0 * (Q.sum_z_dr2 - 0.0133) * (0.0732 - Q.dr_6)
    if Q.sum_z_dr2 > 0.0132 and Q.eccentricity > 0.943:
        z += 23200.0 * (Q.sum_z_dr2 - 0.0132) * (Q.eccentricity - 0.943)
    if Q.sum_z_dr2 > 0.00467 and Q.planar_flow < 0.181:
        z += 881.0 * (Q.sum_z_dr2 - 0.00467) * (0.181 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.0135 and Q.pt_6 < 39.2:
        z += 54.3 * (Q.sum_z_dr2 - 0.0135) * (39.2 - Q.pt_6)
    if Q.sum_z_dr2_top2 < 0.00102 and Q.tau21_b2 < 0.0266:
        z += -104000.0 * (0.00102 - Q.sum_z_dr2_top2) * (0.0266 - Q.tau21_b2)
    if Q.lam2 < 0.000311 and Q.tau21_b2 < 0.04:
        z += 146000.0 * (0.000311 - Q.lam2) * (0.04 - Q.tau21_b2)
    if Q.mass > 76.8 and Q.zdr_6 > 0.00582:
        z += -10.4 * (Q.mass - 76.8) * (Q.zdr_6 - 0.00582)
    if Q.mass_over_sum_pt > 0.0155 and Q.pt_6 < 35.0:
        z += -0.937 * (Q.mass_over_sum_pt - 0.0155) * (35.0 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.0904 and Q.pt_6 < 37.4:
        z += 17.9 * (Q.mass_over_sum_pt - 0.0904) * (37.4 - Q.pt_6)
    if Q.planar_flow < 0.197 and Q.sd_mass > 39.0:
        z += 0.127 * (0.197 - Q.planar_flow) * (Q.sd_mass - 39.0)
    if Q.sd_rg > 0.204 and Q.dr_6 < 0.0702:
        z += 596.0 * (Q.sd_rg - 0.204) * (0.0702 - Q.dr_6)
    if Q.sj3_dr_max > 0.132 and Q.pt_6 < 19.3:
        z += -0.825 * (Q.sj3_dr_max - 0.132) * (19.3 - Q.pt_6)
    if Q.lam1_plus_lam2 < 0.00672 and Q.mean_phi < 0.00613:
        z += 2730.0 * (0.00672 - Q.lam1_plus_lam2) * (0.00613 - Q.mean_phi)
    if Q.lam1_plus_lam2 > 0.00861 and Q.pt_6 < 38.5:
        z += -121.0 * (Q.lam1_plus_lam2 - 0.00861) * (38.5 - Q.pt_6)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.548
    if Q.sum_z_dr < 0.0594:
        z += 213.0 * Q.sum_z_dr - 12.6522
    if Q.sum_z_dr2 < 0.00495:
        z += -922.0 * Q.sum_z_dr2 + 4.5639
    if Q.lam1_plus_lam2 < 0.00351:
        z += -994.0 * Q.lam1_plus_lam2 + 3.48894
    if Q.sum_z_dr < 0.0584 and Q.lam1_plus_lam2 > 0.000544:
        z += 40500.0 * (0.0584 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.000544)
    if Q.sum_z_dr2 < 0.00539 and Q.centroid_offset > 0.0057:
        z += -20500.0 * (0.00539 - Q.sum_z_dr2) * (Q.centroid_offset - 0.0057)
    if Q.sum_z_dr2 < 0.00643 and Q.centroid_offset < 0.0258:
        z += 20600.0 * (0.00643 - Q.sum_z_dr2) * (0.0258 - Q.centroid_offset)
    if Q.lam2 < 0.00014 and Q.D2_b2 < 5.69:
        z += 995.0 * (0.00014 - Q.lam2) * (5.69 - Q.D2_b2)
    if Q.log_sum_pt > 6.7 and Q.e3 < 3.53e-05:
        z += -119000.0 * (Q.log_sum_pt - 6.7) * (3.53e-05 - Q.e3)
    if Q.log_sum_pt > 6.72 and Q.n_dr_0p05_0p1 > 0.999:
        z += -1.85 * (Q.log_sum_pt - 6.72) * (Q.n_dr_0p05_0p1 - 0.999)
    if Q.mass < 31.2 and Q.centroid_offset < 0.0297:
        z += -4.83 * (31.2 - Q.mass) * (0.0297 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.143 and Q.centroid_offset > 0.00799:
        z += -753.0 * (0.143 - Q.sj3_dr_max) * (Q.centroid_offset - 0.00799)
    if Q.sum_pt_top5 > 670.0 and Q.n_pt_above_10 < 8.0:
        z += -0.00681 * (Q.sum_pt_top5 - 670.0) * (8.0 - Q.n_pt_above_10)
    if Q.tau1 < 0.0605 and Q.lam1_plus_lam2 < 0.00306:
        z += 27000.0 * (0.0605 - Q.tau1) * (0.00306 - Q.lam1_plus_lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.66
    if Q.D2 >= 2.2:
        z += -0.422 * Q.D2 + 0.9284
    if Q.absphi_1 < 0.0235:
        z += 21.3 * Q.absphi_1 - 0.50055
    if Q.centroid_offset < 0.00228:
        z += 120.0 * Q.centroid_offset + 2.81916
    if 0.00228 <= Q.centroid_offset < 0.0179:
        z += -198.0 * Q.centroid_offset + 3.5442
    if Q.e3 < 2.39e-05:
        z += 60300.0 * Q.e3 - 1.44117
    if Q.sum_z_dr2 < 0.00364:
        z += -2740.0 * Q.sum_z_dr2 + 13.5304
    if 0.00364 <= Q.sum_z_dr2 < 0.00598:
        z += -1520.0 * Q.sum_z_dr2 + 9.0896
    if Q.lam1 >= 0.00845:
        z += 254.0 * Q.lam1 - 2.1463
    if Q.lam2 < 0.000325:
        z += -2830.0 * Q.lam2 + 0.91975
    if Q.lam2 >= 0.00153:
        z += 527.0 * Q.lam2 - 0.80631
    if Q.mass < 30.4:
        z += 0.2058 * Q.mass - 7.88448
    if 30.4 <= Q.mass < 45.5:
        z += 0.0768 * Q.mass - 3.96288
    if 45.5 <= Q.mass < 51.6:
        z += 0.0648 * Q.mass - 3.41688
    if Q.mass >= 51.6:
        z += -0.012 * Q.mass + 0.546
    if Q.mass_over_sum_pt < 0.0724:
        z += 75.1 * Q.mass_over_sum_pt - 5.43724
    if Q.max_dr < 0.111:
        z += -27.2 * Q.max_dr + 3.0192
    if Q.pt_4 < 36.2:
        z += -0.0448 * Q.pt_4 + 1.62176
    if Q.sj3_dr_max < 0.143:
        z += 23.9 * Q.sj3_dr_max - 1.7918
    if 0.143 <= Q.sj3_dr_max < 0.214:
        z += -22.9 * Q.sj3_dr_max + 4.9006
    if Q.sum_pt < 991.0:
        z += -0.00493 * Q.sum_pt + 4.88563
    if Q.tau1 < 0.0418:
        z += -43.3 * Q.tau1 + 1.80994
    if Q.lam1_plus_lam2 < 0.000168:
        z += -8680.0 * Q.lam1_plus_lam2 + 1.45824
    if Q.zdr_0 < 0.0064:
        z += 154.0 * Q.zdr_0 - 0.9856
    if Q.centroid_offset < 0.0185 and Q.mean_eta > 0.000187:
        z += 5000.0 * (0.0185 - Q.centroid_offset) * (Q.mean_eta - 0.000187)
    if Q.centroid_offset < 0.0185 and Q.n_for_90pct > 5.02:
        z += -32.6 * (0.0185 - Q.centroid_offset) * (Q.n_for_90pct - 5.02)
    if Q.centroid_offset < 0.0179 and Q.pt_1 < 164.0:
        z += -2.07 * (0.0179 - Q.centroid_offset) * (164.0 - Q.pt_1)
    if Q.centroid_offset < 0.0191 and Q.z_2nd < 0.205:
        z += 1080.0 * (0.0191 - Q.centroid_offset) * (0.205 - Q.z_2nd)
    if Q.sum_z_dr < 0.0552 and Q.tau21_b2 < 0.0289:
        z += 3460.0 * (0.0552 - Q.sum_z_dr) * (0.0289 - Q.tau21_b2)
    if Q.sum_z_dr < 0.0575 and Q.z_6 > 0.0334:
        z += -1090.0 * (0.0575 - Q.sum_z_dr) * (Q.z_6 - 0.0334)
    if Q.sum_z_dr2 < 0.00585 and Q.mean_phi > 0.0255:
        z += -52500.0 * (0.00585 - Q.sum_z_dr2) * (Q.mean_phi - 0.0255)
    if Q.sum_z_dr2 < 0.00594 and Q.mean_phi < 0.00379:
        z += -6850.0 * (0.00594 - Q.sum_z_dr2) * (0.00379 - Q.mean_phi)
    if Q.sum_z_dr2 < 0.00677 and Q.planar_flow < 0.296:
        z += 333.0 * (0.00677 - Q.sum_z_dr2) * (0.296 - Q.planar_flow)
    if Q.lam1 > 0.00596 and Q.eccentricity > 0.708:
        z += -724.0 * (Q.lam1 - 0.00596) * (Q.eccentricity - 0.708)
    if Q.log_sum_pt > 6.87 and Q.D2_b2 < 0.754:
        z += -25.9 * (Q.log_sum_pt - 6.87) * (0.754 - Q.D2_b2)
    if Q.mass < 53.3 and Q.D2 > 2.81:
        z += 0.0098 * (53.3 - Q.mass) * (Q.D2 - 2.81)
    if Q.mass < 54.9 and Q.centroid_offset < 0.0266:
        z += 4.8 * (54.9 - Q.mass) * (0.0266 - Q.centroid_offset)
    if Q.mass < 54.4 and Q.dr1_7 > 0.209:
        z += -0.472 * (54.4 - Q.mass) * (Q.dr1_7 - 0.209)
    if Q.mass < 52.7 and Q.log_sum_pt < 6.85:
        z += 0.146 * (52.7 - Q.mass) * (6.85 - Q.log_sum_pt)
    if Q.mass_over_sum_pt < 0.0736 and Q.mass_top2 > 1.83:
        z += -1.99 * (0.0736 - Q.mass_over_sum_pt) * (Q.mass_top2 - 1.83)
    if Q.mass_over_sum_pt < 0.0643 and Q.phi_0 < -0.00767:
        z += 929.0 * (0.0643 - Q.mass_over_sum_pt) * (-0.00767 - Q.phi_0)
    if Q.sd_mass > 44.8 and Q.n_dr_0p05_0p1 > 6.78:
        z += -0.071 * (Q.sd_mass - 44.8) * (Q.n_dr_0p05_0p1 - 6.78)
    if Q.sj3_dr_max < 0.146 and Q.pt_6 > 33.3:
        z += 0.497 * (0.146 - Q.sj3_dr_max) * (Q.pt_6 - 33.3)
    if Q.tau1 < 0.0446 and Q.mean_phi > 0.0255:
        z += 4710.0 * (0.0446 - Q.tau1) * (Q.mean_phi - 0.0255)
    if Q.tau1 < 0.0447 and Q.tau21_b2 < 0.0309:
        z += -4250.0 * (0.0447 - Q.tau1) * (0.0309 - Q.tau21_b2)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.48
    if Q.LHA >= 0.304:
        z += -23.1 * Q.LHA + 7.0224
    if Q.M2 < 0.0255:
        z += 34.2 * Q.M2 - 0.8721
    if Q.centroid_offset < 0.0229:
        z += -13.7 * Q.centroid_offset + 0.31373
    if 0.00732 <= Q.sum_z_dr2 < 0.0253:
        z += 144.0 * Q.sum_z_dr2 - 1.05408
    if Q.sum_z_dr2 >= 0.0253:
        z += 59.7 * Q.sum_z_dr2 + 1.07871
    if Q.sum_z_dr2_top3 < 0.00217:
        z += -262.0 * Q.sum_z_dr2_top3 + 0.56854
    if Q.lam1 < 0.00165:
        z += 1504.0 * Q.lam1 - 4.1502
    if 0.00165 <= Q.lam1 < 0.00435:
        z += 618.0 * Q.lam1 - 2.6883
    if Q.lam2 >= 0.000304:
        z += 1280.0 * Q.lam2 - 0.38912
    if Q.mass < 76.0:
        z += -0.0257 * Q.mass + 1.9532
    if 22.8 <= Q.mass_top5 < 45.4:
        z += 0.0157 * Q.mass_top5 - 0.35796
    if Q.mass_top5 >= 45.4:
        z += -0.0066 * Q.mass_top5 + 0.65446
    if Q.pt_7 < 45.0:
        z += 0.0238 * Q.pt_7 - 1.071
    if Q.sj3_dr_min >= 0.127:
        z += 13.8 * Q.sj3_dr_min - 1.7526
    if Q.sj3_pair_mass_min >= 15.2:
        z += 0.128 * Q.sj3_pair_mass_min - 1.9456
    if Q.sum_pt >= 986.0:
        z += -0.00683 * Q.sum_pt + 6.73438
    if Q.tau1 >= 0.0557:
        z += 7.68 * Q.tau1 - 0.427776
    if Q.zdr_0 < 0.0208:
        z += 29.8 * Q.zdr_0 - 0.61984
    if Q.e3 < 7.95e-05 and Q.sj3_dr23 > 0.177:
        z += -85100.0 * (7.95e-05 - Q.e3) * (Q.sj3_dr23 - 0.177)
    if Q.lam1 > 0.00686 and Q.D2_b2 < 0.383:
        z += -182.0 * (Q.lam1 - 0.00686) * (0.383 - Q.D2_b2)
    if Q.lam1 < 0.00619 and Q.z_dr_0p05_0p1 > 0.215:
        z += -351.0 * (0.00619 - Q.lam1) * (Q.z_dr_0p05_0p1 - 0.215)
    if Q.lam2 > 0.00023 and Q.planar_flow > 0.0514:
        z += -695.0 * (Q.lam2 - 0.00023) * (Q.planar_flow - 0.0514)
    if Q.lam2 > 0.000412 and Q.pt_6 < 31.9:
        z += -24.1 * (Q.lam2 - 0.000412) * (31.9 - Q.pt_6)
    if Q.pt_7 < 45.6 and Q.D2 < 1.0:
        z += 0.0757 * (45.6 - Q.pt_7) * (1.0 - Q.D2)
    if Q.pt_7 < 44.7 and Q.log_sum_pt < 6.54:
        z += -0.0706 * (44.7 - Q.pt_7) * (6.54 - Q.log_sum_pt)
    if Q.sj3_dr_min > 0.134 and Q.z_dr_0_0p05 < 0.43:
        z += -22.2 * (Q.sj3_dr_min - 0.134) * (0.43 - Q.z_dr_0_0p05)
    if Q.sj3_pair_mass_min > 9.6 and Q.sj3_pairmin_over_m > 0.292:
        z += -0.26 * (Q.sj3_pair_mass_min - 9.6) * (Q.sj3_pairmin_over_m - 0.292)
    if Q.tau1 > 0.0451 and Q.D2 < 0.973:
        z += 14.4 * (Q.tau1 - 0.0451) * (0.973 - Q.D2)
    if Q.zdr_0 < 0.0209 and Q.centroid_offset > 0.0107:
        z += 1680.0 * (0.0209 - Q.zdr_0) * (Q.centroid_offset - 0.0107)
    if Q.zdr_0 < 0.022 and Q.dr_7 > 0.0771:
        z += 175.0 * (0.022 - Q.zdr_0) * (Q.dr_7 - 0.0771)
    if Q.zdr_0 < 0.0213 and Q.z_dr_0p05_0p1 < 0.26:
        z += 134.0 * (0.0213 - Q.zdr_0) * (0.26 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.741
    if Q.centroid_offset < 0.0231:
        z += -66.0 * Q.centroid_offset + 1.5642
    if 0.0231 <= Q.centroid_offset < 0.0237:
        z += -128.0 * Q.centroid_offset + 2.9964
    if 0.0237 <= Q.centroid_offset < 0.0467:
        z += -62.0 * Q.centroid_offset + 1.4322
    if Q.centroid_offset >= 0.0467:
        z += 272.0 * Q.centroid_offset - 14.1656
    if Q.e2 < 0.0167:
        z += -58.3 * Q.e2 + 0.97361
    if Q.sum_zz_dr2 < 0.00806:
        z += 973.0 * Q.sum_zz_dr2 - 7.84238
    if Q.eccentricity >= 0.988:
        z += 52.8 * Q.eccentricity - 52.1664
    if Q.sum_z_dr < 0.0263:
        z += 140.8 * Q.sum_z_dr - 7.118
    if 0.0263 <= Q.sum_z_dr < 0.0722:
        z += 74.4 * Q.sum_z_dr - 5.37168
    if Q.sum_z_dr2 < 0.0132:
        z += -305.0 * Q.sum_z_dr2 + 4.026
    if Q.lam1 < 0.0084:
        z += 480.0 * Q.lam1 - 4.032
    if Q.mass < 15.8:
        z += -0.0496 * Q.mass - 1.62148
    if 15.8 <= Q.mass < 68.2:
        z += 0.0459 * Q.mass - 3.13038
    if Q.max_dr < 0.143:
        z += -5.96 * Q.max_dr + 0.85228
    if Q.max_pair_mass >= 32.4:
        z += -0.0296 * Q.max_pair_mass + 0.95904
    if Q.n_dr_0p1_0p2 >= 3.01:
        z += -0.358 * Q.n_dr_0p1_0p2 + 1.07758
    if Q.planar_flow < 0.248:
        z += -6.09 * Q.planar_flow + 1.51032
    if Q.pt_6 < 24.2:
        z += 0.0484 * Q.pt_6 - 1.17128
    if Q.pt_7 >= 28.6:
        z += -0.0502 * Q.pt_7 + 1.43572
    if Q.sj2_dr >= 0.266:
        z += 14.4 * Q.sj2_dr - 3.8304
    if Q.sj3_dr_max < 0.169:
        z += 15.93 * Q.sj3_dr_max - 1.85839
    if 0.169 <= Q.sj3_dr_max < 0.263:
        z += -8.87 * Q.sj3_dr_max + 2.33281
    if Q.sum_pt_top5 < 501.0:
        z += 0.00334 * Q.sum_pt_top5 - 1.67334
    if Q.lam1_plus_lam2 < 0.00865:
        z += -1930.0 * Q.lam1_plus_lam2 + 16.6945
    if Q.z_7 >= 0.0167:
        z += 45.7 * Q.z_7 - 0.76319
    if Q.centroid_offset > 0.0481 and Q.D2 < 3.22:
        z += -81.1 * (Q.centroid_offset - 0.0481) * (3.22 - Q.D2)
    if Q.centroid_offset < 0.0137 and Q.D2_b2 < 0.592:
        z += 182.0 * (0.0137 - Q.centroid_offset) * (0.592 - Q.D2_b2)
    if Q.centroid_offset < 0.042 and Q.mean_phi < 0.0004:
        z += -556.0 * (0.042 - Q.centroid_offset) * (0.0004 - Q.mean_phi)
    if Q.centroid_offset < 0.014 and Q.sj3_pair_mass_min < 16.6:
        z += -10.1 * (0.014 - Q.centroid_offset) * (16.6 - Q.sj3_pair_mass_min)
    if Q.centroid_offset < 0.0392 and Q.sum_pt < 895.0:
        z += -0.167 * (0.0392 - Q.centroid_offset) * (895.0 - Q.sum_pt)
    if Q.centroid_offset > 0.0488 and Q.tau4 > 0.00201:
        z += -86200.0 * (Q.centroid_offset - 0.0488) * (Q.tau4 - 0.00201)
    if Q.centroid_offset > 0.0504 and Q.zdr_7 > 0.00649:
        z += -7570.0 * (Q.centroid_offset - 0.0504) * (Q.zdr_7 - 0.00649)
    if Q.sum_zz_dr2 < 0.00865 and Q.mass_top2 > 9.42:
        z += -4.5 * (0.00865 - Q.sum_zz_dr2) * (Q.mass_top2 - 9.42)
    if Q.eccentricity > 0.988 and Q.sj2_mass2 < 2.9:
        z += -18.4 * (Q.eccentricity - 0.988) * (2.9 - Q.sj2_mass2)
    if Q.planar_flow < 0.247 and Q.e3 < 9.97e-06:
        z += -432000.0 * (0.247 - Q.planar_flow) * (9.97e-06 - Q.e3)
    if Q.planar_flow < 0.255 and Q.pt_7 < 37.0:
        z += -0.17 * (0.255 - Q.planar_flow) * (37.0 - Q.pt_7)
    if Q.planar_flow < 0.252 and Q.sj3_pair_mass_min > 1.69:
        z += -0.142 * (0.252 - Q.planar_flow) * (Q.sj3_pair_mass_min - 1.69)
    if Q.planar_flow < 0.245 and Q.sum_pt < 847.0:
        z += -0.0156 * (0.245 - Q.planar_flow) * (847.0 - Q.sum_pt)
    if Q.sj2_dr > 0.177 and Q.dr_7 < 0.0469:
        z += 177.0 * (Q.sj2_dr - 0.177) * (0.0469 - Q.dr_7)
    if Q.sj3_dr_max < 0.267 and Q.phi_0 < -0.0184:
        z += 91.1 * (0.267 - Q.sj3_dr_max) * (-0.0184 - Q.phi_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.443
    if Q.sum_z_dr2 >= 0.0188:
        z += 352.0 * Q.sum_z_dr2 - 6.6176
    if Q.sum_z_dr2_top2 >= 0.014:
        z += -29.6 * Q.sum_z_dr2_top2 + 0.4144
    if Q.mass >= 88.2:
        z += 0.0667 * Q.mass - 5.88294
    if Q.mass_over_sum_pt >= 0.131:
        z += -53.1 * Q.mass_over_sum_pt + 6.9561
    if Q.zdr_0 >= 0.0398:
        z += -24.9 * Q.zdr_0 + 0.99102
    if Q.sum_z_dr2 > 0.0188 and Q.abseta_6 > 0.0284:
        z += 185.0 * (Q.sum_z_dr2 - 0.0188) * (Q.abseta_6 - 0.0284)
    if Q.sum_z_dr2 > 0.0188 and Q.lam2 > 0.000537:
        z += 5340.0 * (Q.sum_z_dr2 - 0.0188) * (Q.lam2 - 0.000537)
    if Q.sum_z_dr2 > 0.0188 and Q.pt_7 < 53.4:
        z += -2.84 * (Q.sum_z_dr2 - 0.0188) * (53.4 - Q.pt_7)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.402
    if Q.e2 < 0.0799:
        z += 47.7 * Q.e2 - 3.81123
    if Q.e3 < 5.1e-05:
        z += -11000.0 * Q.e3 + 0.561
    if Q.sum_z_dr < 0.149:
        z += -62.6 * Q.sum_z_dr + 9.3274
    if Q.lam1 < 0.016:
        z += -184.0 * Q.lam1 + 2.944
    if Q.mass >= 50.0:
        z += -0.0171 * Q.mass + 0.855
    if Q.pt_7 >= 48.6:
        z += -0.0455 * Q.pt_7 + 2.2113
    if Q.sj3_dr23 >= 0.2:
        z += -2.74 * Q.sj3_dr23 + 0.548
    if Q.z_5 < 0.0276:
        z += 192.0 * Q.z_5 - 5.2992
    if Q.z_7 < 0.0273:
        z += 132.0 * Q.z_7 - 3.6036
    if Q.e3 < 6.01e-05 and Q.D3 > 0.207:
        z += 3150.0 * (6.01e-05 - Q.e3) * (Q.D3 - 0.207)
    if Q.e3 < 5.3e-05 and Q.centroid_offset < 0.0378:
        z += -914000.0 * (5.3e-05 - Q.e3) * (0.0378 - Q.centroid_offset)
    if Q.sum_z_dr < 0.149 and Q.M3 < 0.0747:
        z += 108.0 * (0.149 - Q.sum_z_dr) * (0.0747 - Q.M3)
    if Q.sum_z_dr < 0.142 and Q.lam2 < 0.000592:
        z += -9030.0 * (0.142 - Q.sum_z_dr) * (0.000592 - Q.lam2)
    if Q.sum_z_dr < 0.149 and Q.log_sum_pt < 6.81:
        z += -64.3 * (0.149 - Q.sum_z_dr) * (6.81 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.145 and Q.pt_7 < 38.3:
        z += -0.998 * (0.145 - Q.sum_z_dr) * (38.3 - Q.pt_7)
    if Q.sum_z_dr < 0.146 and Q.sj2_mass1 > 31.7:
        z += -1.61 * (0.146 - Q.sum_z_dr) * (Q.sj2_mass1 - 31.7)
    if Q.sum_z_dr < 0.144 and Q.tau2 > 0.00941:
        z += 288.0 * (0.144 - Q.sum_z_dr) * (Q.tau2 - 0.00941)
    if Q.sum_z_dr < 0.152 and Q.z_7 > 0.0616:
        z += 349.0 * (0.152 - Q.sum_z_dr) * (Q.z_7 - 0.0616)
    if Q.pt_6 < 31.8 and Q.z_7 < 0.0579:
        z += -4.15 * (31.8 - Q.pt_6) * (0.0579 - Q.z_7)
    if Q.sum_pt > 995.0 and Q.pt_6 < 54.8:
        z += -0.000761 * (Q.sum_pt - 995.0) * (54.8 - Q.pt_6)
    if Q.sum_pt_top5 > 789.0 and Q.pt_6 < 35.9:
        z += 0.000988 * (Q.sum_pt_top5 - 789.0) * (35.9 - Q.pt_6)
    if Q.sum_pt_top5 > 662.0 and Q.pt_7 < 40.4:
        z += 0.000721 * (Q.sum_pt_top5 - 662.0) * (40.4 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.173
    if Q.C2_b2 < 0.00406:
        z += 146.0 * Q.C2_b2 - 0.59276
    if Q.LHA >= 0.304:
        z += 26.7 * Q.LHA - 8.1168
    if 0.0381 <= Q.centroid_offset < 0.0498:
        z += -71.3 * Q.centroid_offset + 2.71653
    if Q.centroid_offset >= 0.0498:
        z += -811.3 * Q.centroid_offset + 39.56853
    if 0.0167 <= Q.e2 < 0.0501:
        z += -97.6 * Q.e2 + 1.62992
    if Q.e2 >= 0.0501:
        z += 221.4 * Q.e2 - 14.35198
    if Q.sum_zz_dr2 >= 0.0116:
        z += -792.0 * Q.sum_zz_dr2 + 9.1872
    if Q.e3 < 7.91e-05:
        z += 17300.0 * Q.e3 - 1.36843
    if 0.027 <= Q.sum_z_dr < 0.0412:
        z += 54.3 * Q.sum_z_dr - 1.4661
    if 0.0412 <= Q.sum_z_dr < 0.0807:
        z += 108.7 * Q.sum_z_dr - 3.70738
    if 0.0807 <= Q.sum_z_dr < 0.0873:
        z += 166.9 * Q.sum_z_dr - 8.40412
    if 0.0873 <= Q.sum_z_dr < 0.102:
        z += -100.1 * Q.sum_z_dr + 14.90498
    if Q.sum_z_dr >= 0.102:
        z += -254.1 * Q.sum_z_dr + 30.61298
    if Q.sum_z_dr2 >= 0.00754:
        z += -1070.0 * Q.sum_z_dr2 + 8.0678
    if Q.lam2 < 0.000504:
        z += -1180.0 * Q.lam2 + 0.59472
    if Q.mass >= 71.0:
        z += -0.0265 * Q.mass + 1.8815
    if 0.0798 <= Q.mass_over_sum_pt < 0.0848:
        z += 141.0 * Q.mass_over_sum_pt - 11.2518
    if Q.mass_over_sum_pt >= 0.0848:
        z += 304.0 * Q.mass_over_sum_pt - 25.0742
    if Q.n_dr_0_0p05 < 5.06:
        z += -0.194 * Q.n_dr_0_0p05 + 0.98164
    if Q.psi_0p1 >= 0.973:
        z += -13.2 * Q.psi_0p1 + 12.8436
    if Q.sd_mass < 49.2:
        z += -0.0452 * Q.sd_mass + 2.22384
    if Q.sd_rg < 0.234:
        z += 9.48 * Q.sd_rg - 3.081
    if 0.234 <= Q.sd_rg < 0.279:
        z += -7.62 * Q.sd_rg + 0.9204
    if 0.279 <= Q.sd_rg < 0.325:
        z += 11.28 * Q.sd_rg - 4.3527
    if Q.sd_rg >= 0.325:
        z += 1.8 * Q.sd_rg - 1.2717
    if 0.0614 <= Q.sj2_dr < 0.159:
        z += -9.74 * Q.sj2_dr + 0.598036
    if 0.159 <= Q.sj2_dr < 0.199:
        z += 18.66 * Q.sj2_dr - 3.917564
    if Q.sj2_dr >= 0.199:
        z += 6.36 * Q.sj2_dr - 1.469864
    if Q.sum_pt < 724.0:
        z += 0.00528 * Q.sum_pt - 3.82272
    if 0.00348 <= Q.lam1_plus_lam2 < 0.00668:
        z += 324.0 * Q.lam1_plus_lam2 - 1.12752
    if 0.00668 <= Q.lam1_plus_lam2 < 0.0119:
        z += -986.0 * Q.lam1_plus_lam2 + 7.62328
    if Q.lam1_plus_lam2 >= 0.0119:
        z += -1085.8 * Q.lam1_plus_lam2 + 8.8109
    if Q.z_dr_0_0p05 >= 0.152:
        z += 1.4 * Q.z_dr_0_0p05 - 0.2128
    if Q.z_dr_0p05_0p1 < 0.163:
        z += 2.28 * Q.z_dr_0p05_0p1 - 0.37164
    if Q.centroid_offset > 0.0497 and Q.C2_b2 < 0.000829:
        z += 908000.0 * (Q.centroid_offset - 0.0497) * (0.000829 - Q.C2_b2)
    if Q.sum_zz_dr2 > 0.00194 and Q.sj2_dr < 0.184:
        z += -2030.0 * (Q.sum_zz_dr2 - 0.00194) * (0.184 - Q.sj2_dr)
    if Q.e3 < 5.6e-05 and Q.sj3_dr23 > 0.164:
        z += 153000.0 * (5.6e-05 - Q.e3) * (Q.sj3_dr23 - 0.164)
    if Q.sum_z_dr > 0.027 and Q.D2_b2 > 4.77:
        z += -78.9 * (Q.sum_z_dr - 0.027) * (Q.D2_b2 - 4.77)
    if Q.planar_flow < 0.109 and Q.centroid_offset < 0.0183:
        z += -303.0 * (0.109 - Q.planar_flow) * (0.0183 - Q.centroid_offset)
    if Q.planar_flow < 0.105 and Q.lam1 < 0.0074:
        z += -2710.0 * (0.105 - Q.planar_flow) * (0.0074 - Q.lam1)
    if Q.planar_flow < 0.115 and Q.lam1 < 0.0164:
        z += 1070.0 * (0.115 - Q.planar_flow) * (0.0164 - Q.lam1)
    if Q.planar_flow < 0.106 and Q.mean_eta > -0.00694:
        z += -120.0 * (0.106 - Q.planar_flow) * (Q.mean_eta - -0.00694)
    if Q.planar_flow < 0.095 and Q.sum_pt_top5 < 667.0:
        z += -0.0193 * (0.095 - Q.planar_flow) * (667.0 - Q.sum_pt_top5)
    if Q.psi_0p1 > 0.821 and Q.D2_b2 < 0.0838:
        z += 37.5 * (Q.psi_0p1 - 0.821) * (0.0838 - Q.D2_b2)
    if Q.sd_rg > 0.278 and Q.D2_b2 < 0.263:
        z += -59.5 * (Q.sd_rg - 0.278) * (0.263 - Q.D2_b2)
    if Q.sj2_dr > 0.13 and Q.D2_b2 < 0.087:
        z += -265.0 * (Q.sj2_dr - 0.13) * (0.087 - Q.D2_b2)
    if Q.sj2_dr > 0.159 and Q.D2_b2 < 0.0866:
        z += 659.0 * (Q.sj2_dr - 0.159) * (0.0866 - Q.D2_b2)
    if Q.sj2_dr > 0.2 and Q.D2_b2 < 0.0864:
        z += -516.0 * (Q.sj2_dr - 0.2) * (0.0864 - Q.D2_b2)
    if Q.sj2_dr > 0.16 and Q.dr_2 < 0.033:
        z += -529.0 * (Q.sj2_dr - 0.16) * (0.033 - Q.dr_2)
    if Q.sj2_dr > 0.198 and Q.dr_2 < 0.0508:
        z += 271.0 * (Q.sj2_dr - 0.198) * (0.0508 - Q.dr_2)
    if Q.sj2_dr > 0.199 and Q.z_5 < 0.106:
        z += -47.0 * (Q.sj2_dr - 0.199) * (0.106 - Q.z_5)
    if Q.z_dr_0p05_0p1 > 0.75 and Q.C2_b2 < 0.024:
        z += -321.0 * (Q.z_dr_0p05_0p1 - 0.75) * (0.024 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 < 0.579 and Q.D2_b2 < 0.184:
        z += -4.77 * (0.579 - Q.z_dr_0p05_0p1) * (0.184 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 < 0.589 and Q.D3 < 0.315:
        z += 3.61 * (0.589 - Q.z_dr_0p05_0p1) * (0.315 - Q.D3)
    if Q.z_dr_0p05_0p1 > 0.752 and Q.n_dr_0p2_0p4 < 1.01:
        z += 6.91 * (Q.z_dr_0p05_0p1 - 0.752) * (1.01 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 < 0.601 and Q.sum_pt < 720.0:
        z += 0.0025 * (0.601 - Q.z_dr_0p05_0p1) * (720.0 - Q.sum_pt)
    if Q.z_dr_0p05_0p1 > 0.754 and Q.zdr_5 > 0.0156:
        z += -3970.0 * (Q.z_dr_0p05_0p1 - 0.754) * (Q.zdr_5 - 0.0156)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.75
    if Q.N2 < 0.228:
        z += 6.37 * Q.N2 - 1.45236
    if Q.e2 < 0.0416:
        z += -84.6 * Q.e2 + 3.51936
    if Q.sum_zz_dr2 < 0.00834:
        z += 517.0 * Q.sum_zz_dr2 - 4.31178
    if Q.sum_z_dr < 0.0992:
        z += 81.0 * Q.sum_z_dr - 8.0352
    if Q.sum_z_dr2 < 0.00754:
        z += 698.0 * Q.sum_z_dr2 - 5.26292
    if Q.sum_z_dr2_top2 < 0.000524:
        z += -4182.0 * Q.sum_z_dr2_top2 + 3.46464
    if 0.000524 <= Q.sum_z_dr2_top2 < 0.00752:
        z += -182.0 * Q.sum_z_dr2_top2 + 1.36864
    if Q.sum_z_dr2_top3 < 0.00686:
        z += -137.0 * Q.sum_z_dr2_top3 + 0.93982
    if Q.lam1 < 0.00498:
        z += 616.0 * Q.lam1 - 3.06768
    if Q.lam2 < 0.000329:
        z += 1990.0 * Q.lam2 - 0.65471
    if Q.log_sum_pt < 6.09:
        z += 5.64 * Q.log_sum_pt - 34.3476
    if 45.8 <= Q.mass < 77.3:
        z += 0.0563 * Q.mass - 2.57854
    if Q.mass >= 77.3:
        z += 0.0078 * Q.mass + 1.17051
    if Q.sj2_dr < 0.163:
        z += -6.5 * Q.sj2_dr + 0.3269
    if 0.163 <= Q.sj2_dr < 0.2:
        z += 19.8 * Q.sj2_dr - 3.96
    if Q.sj3_pair_mass_max >= 38.7:
        z += -0.0189 * Q.sj3_pair_mass_max + 0.73143
    if Q.tau1 < 0.0622:
        z += -93.5 * Q.tau1 + 5.8157
    if Q.z_dr_0p05_0p1 >= 0.735:
        z += -19.5 * Q.z_dr_0p05_0p1 + 14.3325
    if Q.N2 < 0.23 and Q.n_dr_0p1_0p2 < 4.02:
        z += 0.906 * (0.23 - Q.N2) * (4.02 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.228 and Q.n_for_90pct < 5.9:
        z += -3.45 * (0.228 - Q.N2) * (5.9 - Q.n_for_90pct)
    if Q.N2 < 0.217 and Q.sum_pt_top5 < 717.0:
        z += 0.0336 * (0.217 - Q.N2) * (717.0 - Q.sum_pt_top5)
    if Q.N2 < 0.226 and Q.sum_pt_top5 > 426.0:
        z += 0.0383 * (0.226 - Q.N2) * (Q.sum_pt_top5 - 426.0)
    if Q.N2 < 0.229 and Q.z_dr_0p05_0p1 < 0.56:
        z += -7.63 * (0.229 - Q.N2) * (0.56 - Q.z_dr_0p05_0p1)
    if Q.centroid_offset < 0.0179 and Q.dr01 < 0.286:
        z += -441.0 * (0.0179 - Q.centroid_offset) * (0.286 - Q.dr01)
    if Q.centroid_offset < 0.0199 and Q.zdr_0 < 0.0123:
        z += -15600.0 * (0.0199 - Q.centroid_offset) * (0.0123 - Q.zdr_0)
    if Q.e2 < 0.065 and Q.dr12 > 0.0982:
        z += 100.0 * (0.065 - Q.e2) * (Q.dr12 - 0.0982)
    if Q.sum_zz_dr2 < 0.0106 and Q.lam2 > -1.14e-06:
        z += 411000.0 * (0.0106 - Q.sum_zz_dr2) * (Q.lam2 - -1.14e-06)
    if Q.sum_z_dr < 0.1 and Q.dr1_3 > 0.155:
        z += -126.0 * (0.1 - Q.sum_z_dr) * (Q.dr1_3 - 0.155)
    if Q.sum_z_dr2 < 0.00758 and Q.D2 < 0.771:
        z += -1230.0 * (0.00758 - Q.sum_z_dr2) * (0.771 - Q.D2)
    if Q.lam2 < 0.00111 and Q.mean_phi > -0.0179:
        z += -11400.0 * (0.00111 - Q.lam2) * (Q.mean_phi - -0.0179)
    if Q.mass > 46.7 and Q.pt_7 < 23.5:
        z += -0.00426 * (Q.mass - 46.7) * (23.5 - Q.pt_7)
    if Q.tau1 < 0.0267 and Q.zdr_3 > 0.00824:
        z += -168000.0 * (0.0267 - Q.tau1) * (Q.zdr_3 - 0.00824)
    if Q.lam1_plus_lam2 < 0.00611 and Q.D2 < 0.713:
        z += 2140.0 * (0.00611 - Q.lam1_plus_lam2) * (0.713 - Q.D2)
    if Q.lam1_plus_lam2 < 0.0182 and Q.dr1_3 > 0.0627:
        z += 181.0 * (0.0182 - Q.lam1_plus_lam2) * (Q.dr1_3 - 0.0627)
    if Q.lam1_plus_lam2 < 0.0187 and Q.lam2 < 0.00343:
        z += 138000.0 * (0.0187 - Q.lam1_plus_lam2) * (0.00343 - Q.lam2)
    if Q.lam1_plus_lam2 < 0.0187 and Q.ptdr0_2 > 14.7:
        z += 4.3 * (0.0187 - Q.lam1_plus_lam2) * (Q.ptdr0_2 - 14.7)
    if Q.z_dr_0p05_0p1 > 0.741 and Q.n_dr_0p2_0p4 < 2.03:
        z += 8.82 * (Q.z_dr_0p05_0p1 - 0.741) * (2.03 - Q.n_dr_0p2_0p4)
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
