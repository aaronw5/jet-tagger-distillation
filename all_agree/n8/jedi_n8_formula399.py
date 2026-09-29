"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables, tuned for agreement), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
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
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
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
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
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
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -3.66
    if Q.e2 < 0.0228:
        z += -91.5 * Q.e2 + 2.0862
    if Q.girth < 0.0788:
        z += 40.6 * Q.girth - 3.19928
    if Q.girth2 < 0.013:
        z += -446.0 * Q.girth2 + 5.798
    if Q.lam1 < 0.00027:
        z += 6731.0 * Q.lam1 - 4.25187
    if 0.00027 <= Q.lam1 < 0.00477:
        z += 541.0 * Q.lam1 - 2.58057
    if Q.log_sum_pt < 6.08:
        z += 8.95 * Q.log_sum_pt - 54.416
    if Q.log_sum_pt >= 6.36:
        z += 2.69 * Q.log_sum_pt - 17.1084
    if Q.m012 >= 38.4:
        z += -0.075 * Q.m012 + 2.88
    if Q.mass < 62.5:
        z += 0.038 * Q.mass - 2.375
    if Q.sj3_dr_max < 0.105:
        z += -17.4 * Q.sj3_dr_max + 5.3418
    if 0.105 <= Q.sj3_dr_max < 0.179:
        z += 12.9 * Q.sj3_dr_max + 2.1603
    if 0.179 <= Q.sj3_dr_max < 0.231:
        z += -4.0 * Q.sj3_dr_max + 5.1854
    if 0.231 <= Q.sj3_dr_max < 0.307:
        z += -28.5 * Q.sj3_dr_max + 10.8449
    if Q.sj3_dr_max >= 0.307:
        z += -11.1 * Q.sj3_dr_max + 5.5031
    if Q.sum_pt >= 900.0:
        z += -0.0157 * Q.sum_pt + 14.13
    if Q.sum_pt_top5 >= 680.0:
        z += 0.00657 * Q.sum_pt_top5 - 4.4676
    if Q.centroid_offset > 0.0209 and Q.dr_2 < 0.048:
        z += -4330.0 * (Q.centroid_offset - 0.0209) * (0.048 - Q.dr_2)
    if Q.centroid_offset > 0.0212 and Q.z_7 < 0.0684:
        z += -3610.0 * (Q.centroid_offset - 0.0212) * (0.0684 - Q.z_7)
    if Q.girth < 0.0879 and Q.dr1_3 > 0.165:
        z += 1130.0 * (0.0879 - Q.girth) * (Q.dr1_3 - 0.165)
    if Q.girth2 < 0.0151 and Q.centroid_offset > 0.0192:
        z += -6010.0 * (0.0151 - Q.girth2) * (Q.centroid_offset - 0.0192)
    if Q.girth2 < 0.0122 and Q.phi_0 > -0.0104:
        z += 1650.0 * (0.0122 - Q.girth2) * (Q.phi_0 - -0.0104)
    if Q.girth2 < 0.0201 and Q.phi_2 < -0.0955:
        z += 8000.0 * (0.0201 - Q.girth2) * (-0.0955 - Q.phi_2)
    if Q.lam2 < 6.27e-05 and Q.D2_b2 < 0.231:
        z += 67200.0 * (6.27e-05 - Q.lam2) * (0.231 - Q.D2_b2)
    if Q.log_sum_pt > 6.67 and Q.dr_4 < 0.0674:
        z += 150.0 * (Q.log_sum_pt - 6.67) * (0.0674 - Q.dr_4)
    if Q.log_sum_pt > 6.66 and Q.phi_4 > 0.11:
        z += 370.0 * (Q.log_sum_pt - 6.66) * (Q.phi_4 - 0.11)
    if Q.log_sum_pt > 6.67 and Q.z_7 > 0.0163:
        z += -492.0 * (Q.log_sum_pt - 6.67) * (Q.z_7 - 0.0163)
    if Q.mass < 30.5 and Q.D2_b2 < 0.186:
        z += -4.19 * (30.5 - Q.mass) * (0.186 - Q.D2_b2)
    if Q.n_dr_0_0p05 > 4.05 and Q.dr_4 > 0.194:
        z += 23.1 * (Q.n_dr_0_0p05 - 4.05) * (Q.dr_4 - 0.194)
    if Q.planar_flow < 0.139 and Q.centroid_offset < 0.0561:
        z += 282.0 * (0.139 - Q.planar_flow) * (0.0561 - Q.centroid_offset)
    if Q.planar_flow < 0.132 and Q.dr0_6 > 0.175:
        z += -121.0 * (0.132 - Q.planar_flow) * (Q.dr0_6 - 0.175)
    if Q.planar_flow < 0.146 and Q.dr_2 < 0.0219:
        z += -1310.0 * (0.146 - Q.planar_flow) * (0.0219 - Q.dr_2)
    if Q.planar_flow < 0.163 and Q.sum_pt_top2 < 417.0:
        z += 0.0269 * (0.163 - Q.planar_flow) * (417.0 - Q.sum_pt_top2)
    if Q.planar_flow < 0.145 and Q.z_7 < 0.0231:
        z += 5380.0 * (0.145 - Q.planar_flow) * (0.0231 - Q.z_7)
    if Q.sum_pt > 906.0 and Q.dr0_6 > 0.239:
        z += 0.721 * (Q.sum_pt - 906.0) * (Q.dr0_6 - 0.239)
    if Q.sum_pt > 899.0 and Q.eccentricity > 0.95:
        z += -0.213 * (Q.sum_pt - 899.0) * (Q.eccentricity - 0.95)
    if Q.width < 0.00483 and Q.D2 < 0.764:
        z += -4490.0 * (0.00483 - Q.width) * (0.764 - Q.D2)
    if Q.width < 0.00437 and Q.eccentricity > 0.962:
        z += -13200.0 * (0.00437 - Q.width) * (Q.eccentricity - 0.962)
    if Q.width < 0.00443 and Q.pair_mass_0_6 > 20.4:
        z += 361.0 * (0.00443 - Q.width) * (Q.pair_mass_0_6 - 20.4)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.03
    if Q.centroid_offset < 0.0178:
        z += 62.0 * Q.centroid_offset - 1.1036
    if Q.e2_sq < 0.00385:
        z += -866.0 * Q.e2_sq + 3.3341
    if Q.girth < 0.0774:
        z += -32.9 * Q.girth + 2.54646
    if Q.girth2 < 0.00992:
        z += 229.0 * Q.girth2 - 2.27168
    if Q.log_sum_pt >= 6.6:
        z += 11.9 * Q.log_sum_pt - 78.54
    if Q.mass < 50.6:
        z += -0.0705 * Q.mass + 3.5673
    if Q.mass_over_sum_pt < 0.0623:
        z += 55.8 * Q.mass_over_sum_pt - 3.47634
    if Q.pt_7 >= 33.3:
        z += 0.0306 * Q.pt_7 - 1.01898
    if Q.sum_pt < 578.0:
        z += 0.00022 * Q.sum_pt - 1.07236
    if 578.0 <= Q.sum_pt < 714.0:
        z += 0.00695 * Q.sum_pt - 4.9623
    if Q.tau21_b2 < 0.00494:
        z += -328.0 * Q.tau21_b2 + 1.62032
    if Q.z_7 < 0.0625:
        z += 62.4 * Q.z_7 - 3.9
    if Q.girth2 < 0.00947 and Q.centroid_offset > 0.0199:
        z += 19300.0 * (0.00947 - Q.girth2) * (Q.centroid_offset - 0.0199)
    if Q.log_sum_pt > 6.56 and Q.D2 < 1.42:
        z += 6.17 * (Q.log_sum_pt - 6.56) * (1.42 - Q.D2)
    if Q.log_sum_pt > 6.4 and Q.centroid_offset > 0.00857:
        z += 326.0 * (Q.log_sum_pt - 6.4) * (Q.centroid_offset - 0.00857)
    if Q.log_sum_pt > 6.42 and Q.e3 < 1.67e-05:
        z += -503000.0 * (Q.log_sum_pt - 6.42) * (1.67e-05 - Q.e3)
    if Q.log_sum_pt > 6.35 and Q.z_dr_0p05_0p1 < 0.765:
        z += -4.28 * (Q.log_sum_pt - 6.35) * (0.765 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0_0p05 < 1.53 and Q.n_pt_above_50 > 5.38:
        z += 0.349 * (1.53 - Q.n_dr_0_0p05) * (Q.n_pt_above_50 - 5.38)
    if Q.pt_7 > 33.8 and Q.pt_6 < 53.1:
        z += -0.00944 * (Q.pt_7 - 33.8) * (53.1 - Q.pt_6)
    if Q.pt_7 > 33.6 and Q.sj2_dr > 0.128:
        z += 0.782 * (Q.pt_7 - 33.6) * (Q.sj2_dr - 0.128)
    if Q.pt_7 > 33.5 and Q.z_dr_0p05_0p1 > 0.316:
        z += -0.112 * (Q.pt_7 - 33.5) * (Q.z_dr_0p05_0p1 - 0.316)
    if Q.sj3_dr_max > 0.169 and Q.sj3_pair_mass_min > 4.47:
        z += -0.15 * (Q.sj3_dr_max - 0.169) * (Q.sj3_pair_mass_min - 4.47)
    if Q.sum_pt_top5 > 604.0 and Q.phi_0 < -0.0456:
        z += 0.25 * (Q.sum_pt_top5 - 604.0) * (-0.0456 - Q.phi_0)
    if Q.z_7 > 0.0447 and Q.girth2_top2 < 0.00244:
        z += 11900.0 * (Q.z_7 - 0.0447) * (0.00244 - Q.girth2_top2)
    if Q.z_7 < 0.06 and Q.z_dr_0p05_0p1 > 0.0548:
        z += -97.4 * (0.06 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.0548)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.67
    if Q.LHA >= 0.132:
        z += -16.5 * Q.LHA + 2.178
    if Q.centroid_offset < 0.00441:
        z += 371.0 * Q.centroid_offset - 1.63611
    if Q.girth < 0.00766:
        z += 1340.0 * Q.girth - 10.2644
    if Q.lam1 < 0.00599:
        z += -431.0 * Q.lam1 + 2.58169
    if Q.log_sum_pt < 6.57:
        z += -9.26 * Q.log_sum_pt + 60.8382
    if Q.log_sum_pt >= 6.83:
        z += 34.5 * Q.log_sum_pt - 235.635
    if Q.m012 < 4.17:
        z += -0.205 * Q.m012 + 0.85485
    if Q.m012 >= 40.3:
        z += 0.0863 * Q.m012 - 3.47789
    if Q.n_pt_above_5 < 7.96:
        z += 6.32 * Q.n_pt_above_5 - 50.3072
    if Q.planar_flow < 0.507:
        z += 1.95 * Q.planar_flow - 0.98865
    if Q.pt_7 < 32.4:
        z += 0.0688 * Q.pt_7 - 3.72208
    if 32.4 <= Q.pt_7 < 54.1:
        z += 0.149 * Q.pt_7 - 6.32056
    if Q.pt_7 >= 54.1:
        z += 0.0802 * Q.pt_7 - 2.59848
    if Q.sj3_pair_mass_max < 62.4:
        z += -0.0463 * Q.sj3_pair_mass_max + 2.88912
    if Q.sum_pt < 820.0:
        z += -0.00696 * Q.sum_pt + 5.7072
    if Q.sum_pt >= 869.0:
        z += 0.0176 * Q.sum_pt - 15.2944
    if Q.sum_pt_top5 >= 761.0:
        z += -0.0173 * Q.sum_pt_top5 + 13.1653
    if Q.z_7 < 0.03:
        z += 124.0 * Q.z_7 - 3.72
    if Q.centroid_offset < 0.00446 and Q.abseta_7 < 0.0255:
        z += -26600.0 * (0.00446 - Q.centroid_offset) * (0.0255 - Q.abseta_7)
    if Q.girth < 0.00837 and Q.pt_4 < 74.6:
        z += -9.5 * (0.00837 - Q.girth) * (74.6 - Q.pt_4)
    if Q.lam1 < 0.00635 and Q.max_dr > 0.0787:
        z += -4070.0 * (0.00635 - Q.lam1) * (Q.max_dr - 0.0787)
    if Q.lam1 < 0.00669 and Q.pt_6 > 25.5:
        z += 6.45 * (0.00669 - Q.lam1) * (Q.pt_6 - 25.5)
    if Q.log_sum_pt > 6.83 and Q.D2_b2 < 1.34:
        z += -23.3 * (Q.log_sum_pt - 6.83) * (1.34 - Q.D2_b2)
    if Q.log_sum_pt < 6.46 and Q.D2_b2 < 1.12:
        z += 6.18 * (6.46 - Q.log_sum_pt) * (1.12 - Q.D2_b2)
    if Q.log_sum_pt > 6.86 and Q.abseta_0 < 0.0183:
        z += 1030.0 * (Q.log_sum_pt - 6.86) * (0.0183 - Q.abseta_0)
    if Q.log_sum_pt < 6.7 and Q.absphi_2 < 0.0158:
        z += 281.0 * (6.7 - Q.log_sum_pt) * (0.0158 - Q.absphi_2)
    if Q.log_sum_pt > 6.84 and Q.lam2 < 0.00018:
        z += 96500.0 * (Q.log_sum_pt - 6.84) * (0.00018 - Q.lam2)
    if Q.log_sum_pt < 6.61 and Q.max_dr < 0.0375:
        z += 1020.0 * (6.61 - Q.log_sum_pt) * (0.0375 - Q.max_dr)
    if Q.log_sum_pt > 6.85 and Q.pt_6 > 39.5:
        z += 0.657 * (Q.log_sum_pt - 6.85) * (Q.pt_6 - 39.5)
    if Q.log_sum_pt > 6.85 and Q.z_dr_0p05_0p1 < 0.47:
        z += 56.0 * (Q.log_sum_pt - 6.85) * (0.47 - Q.z_dr_0p05_0p1)
    if Q.mass < 37.8 and Q.pt_5 < 72.6:
        z += 0.00128 * (37.8 - Q.mass) * (72.6 - Q.pt_5)
    if Q.max_dr < 0.262 and Q.phi_0 > 0.00486:
        z += -65.3 * (0.262 - Q.max_dr) * (Q.phi_0 - 0.00486)
    if Q.pt_7 > 42.7 and Q.C2_b2 < 0.00511:
        z += 22.9 * (Q.pt_7 - 42.7) * (0.00511 - Q.C2_b2)
    if Q.pt_7 < 52.7 and Q.D2_b2 < 1.2:
        z += -0.0542 * (52.7 - Q.pt_7) * (1.2 - Q.D2_b2)
    if Q.pt_7 < 59.5 and Q.abseta_0 > 0.0188:
        z += -0.319 * (59.5 - Q.pt_7) * (Q.abseta_0 - 0.0188)
    if Q.sj3_pair_mass_max < 61.4 and Q.centroid_offset > 0.00874:
        z += -0.966 * (61.4 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.00874)
    if Q.sj3_pair_mass_max < 54.3 and Q.z_7 < 0.0697:
        z += -1.3 * (54.3 - Q.sj3_pair_mass_max) * (0.0697 - Q.z_7)
    if Q.sum_pt > 540.0 and Q.absphi_2 < 0.0228:
        z += -0.209 * (Q.sum_pt - 540.0) * (0.0228 - Q.absphi_2)
    if Q.sum_pt > 544.0 and Q.lam2 < 0.000197:
        z += -24.5 * (Q.sum_pt - 544.0) * (0.000197 - Q.lam2)
    if Q.sum_pt_top5 > 755.0 and Q.D2_b2 < 1.26:
        z += 0.0226 * (Q.sum_pt_top5 - 755.0) * (1.26 - Q.D2_b2)
    if Q.sum_pt_top5 > 853.0 and Q.absphi_2 < 0.00117:
        z += -17.5 * (Q.sum_pt_top5 - 853.0) * (0.00117 - Q.absphi_2)
    if Q.sum_pt_top5 > 849.0 and Q.dr_7 < 0.0498:
        z += 0.584 * (Q.sum_pt_top5 - 849.0) * (0.0498 - Q.dr_7)
    if Q.sum_pt_top5 > 774.0 and Q.eta_0 > 0.00474:
        z += 0.443 * (Q.sum_pt_top5 - 774.0) * (Q.eta_0 - 0.00474)
    if Q.z_7 > 0.0484 and Q.D2_b2 < 0.973:
        z += -57.6 * (Q.z_7 - 0.0484) * (0.973 - Q.D2_b2)
    if Q.z_7 < 0.0311 and Q.D2_b2 < 0.806:
        z += -572.0 * (0.0311 - Q.z_7) * (0.806 - Q.D2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = -3.34
    if Q.abseta_7 >= 0.0774:
        z += -5.72 * Q.abseta_7 + 0.442728
    if Q.centroid_offset >= 0.0111:
        z += 42.0 * Q.centroid_offset - 0.4662
    if Q.girth >= 0.0874:
        z += 106.0 * Q.girth - 9.2644
    if Q.girth2 >= 0.00895:
        z += -402.0 * Q.girth2 + 3.5979
    if Q.max_dr >= 0.101:
        z += 8.72 * Q.max_dr - 0.88072
    if Q.mean_eta < -0.00371:
        z += -40.5 * Q.mean_eta - 0.150255
    if Q.mean_eta >= 0.0176:
        z += 75.6 * Q.mean_eta - 1.33056
    if Q.n_dr_0_0p05 < 1.0:
        z += -0.747 * Q.n_dr_0_0p05 + 0.747
    if Q.sd_mass >= 113.0:
        z += 0.136 * Q.sd_mass - 15.368
    if Q.sj2_dr >= 0.185:
        z += 48.8 * Q.sj2_dr - 9.028
    if Q.sj3_dr_max >= 0.186:
        z += 18.8 * Q.sj3_dr_max - 3.4968
    if Q.tau1 < 0.0537:
        z += 18.0 * Q.tau1 - 0.9666
    if 0.00549 <= Q.width < 0.0138:
        z += 889.0 * Q.width - 4.88061
    if Q.width >= 0.0138:
        z += -551.0 * Q.width + 14.99139
    if Q.centroid_offset > 0.00992 and Q.abseta_0 < 0.0872:
        z += 928.0 * (Q.centroid_offset - 0.00992) * (0.0872 - Q.abseta_0)
    if Q.centroid_offset > 0.0121 and Q.n_pt_above_50 < 7.57:
        z += 6.9 * (Q.centroid_offset - 0.0121) * (7.57 - Q.n_pt_above_50)
    if Q.centroid_offset > 0.0109 and Q.pair_mass_0_5 > 16.4:
        z += 6.39 * (Q.centroid_offset - 0.0109) * (Q.pair_mass_0_5 - 16.4)
    if Q.girth > 0.0433 and Q.log_sum_pt > 6.11:
        z += 55.6 * (Q.girth - 0.0433) * (Q.log_sum_pt - 6.11)
    if Q.girth > 0.0868 and Q.z_dr_0p05_0p1 > 0.683:
        z += 1040.0 * (Q.girth - 0.0868) * (Q.z_dr_0p05_0p1 - 0.683)
    if Q.mass_over_sum_pt > 0.0668 and Q.sj2_dr < 0.22:
        z += -1450.0 * (Q.mass_over_sum_pt - 0.0668) * (0.22 - Q.sj2_dr)
    if Q.mass_over_sum_pt > 0.0454 and Q.sj3_pair_mass_min > 8.19:
        z += -0.915 * (Q.mass_over_sum_pt - 0.0454) * (Q.sj3_pair_mass_min - 8.19)
    if Q.sd_mass > 64.0 and Q.D2_b2 < 1.93:
        z += 0.0411 * (Q.sd_mass - 64.0) * (1.93 - Q.D2_b2)
    if Q.sj2_dr > 0.179 and Q.dr_3 < 0.05:
        z += -748.0 * (Q.sj2_dr - 0.179) * (0.05 - Q.dr_3)
    if Q.sj2_dr > 0.215 and Q.pt_6 < 25.3:
        z += -1.05 * (Q.sj2_dr - 0.215) * (25.3 - Q.pt_6)
    if Q.sj2_dr > 0.187 and Q.z_dr_0p05_0p1 < 0.992:
        z += -45.3 * (Q.sj2_dr - 0.187) * (0.992 - Q.z_dr_0p05_0p1)
    if Q.sj2_dr > 0.275 and Q.z_dr_0p05_0p1 < 0.958:
        z += 29.2 * (Q.sj2_dr - 0.275) * (0.958 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_4(Q):
    z = 3.91
    if Q.e3 < 0.000149:
        z += -33300.0 * Q.e3 + 4.9617
    if Q.girth2 < 0.00896:
        z += 988.0 * Q.girth2 - 8.85248
    if Q.lam2 < 0.000586:
        z += 5230.0 * Q.lam2 - 3.06478
    if Q.mass >= 77.8:
        z += -0.0719 * Q.mass + 5.59382
    if Q.max_dr < 0.11:
        z += -0.7 * Q.max_dr - 2.2483
    if 0.11 <= Q.max_dr < 0.179:
        z += 33.7 * Q.max_dr - 6.0323
    if Q.sj3_dr_max < 0.374:
        z += -15.7 * Q.sj3_dr_max + 5.8718
    if Q.sj3_dr_max < 0.347 and Q.pair_mass_0_6 > 17.4:
        z += 2.76 * (0.347 - Q.sj3_dr_max) * (Q.pair_mass_0_6 - 17.4)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.869
    if Q.girth < 0.00768:
        z += -742.0 * Q.girth + 5.69856
    if Q.log_sum_pt >= 6.87:
        z += -43.7 * Q.log_sum_pt + 300.219
    if Q.pt_5 < 23.5:
        z += -0.424 * Q.pt_5 + 9.964
    if Q.width < 0.00337:
        z += -1100.0 * Q.width + 3.707
    if Q.z_7 < 0.026:
        z += -320.8 * Q.z_7 + 11.71912
    if 0.026 <= Q.z_7 < 0.0744:
        z += -69.8 * Q.z_7 + 5.19312
    if Q.zdr_0 < 0.0214:
        z += 91.0 * Q.zdr_0 - 1.9474
    if Q.LHA < 0.228 and Q.log_sum_pt < 6.78:
        z += -94.8 * (0.228 - Q.LHA) * (6.78 - Q.log_sum_pt)
    if Q.LHA < 0.249 and Q.z_3 < 0.065:
        z += 931.0 * (0.249 - Q.LHA) * (0.065 - Q.z_3)
    if Q.centroid_offset < 0.00685 and Q.pt_5 < 59.7:
        z += 13.8 * (0.00685 - Q.centroid_offset) * (59.7 - Q.pt_5)
    if Q.e2_sq < 0.00708 and Q.centroid_offset < 0.011:
        z += 35900.0 * (0.00708 - Q.e2_sq) * (0.011 - Q.centroid_offset)
    if Q.e2_sq < 0.00435 and Q.n_pt_above_50 > 6.24:
        z += -155.0 * (0.00435 - Q.e2_sq) * (Q.n_pt_above_50 - 6.24)
    if Q.e2_sq < 0.00546 and Q.planar_flow < 0.329:
        z += 1200.0 * (0.00546 - Q.e2_sq) * (0.329 - Q.planar_flow)
    if Q.girth2 < 0.00154 and Q.centroid_offset < 0.0251:
        z += 173000.0 * (0.00154 - Q.girth2) * (0.0251 - Q.centroid_offset)
    if Q.log_sum_pt > 6.89 and Q.D2_b2 < 0.0532:
        z += 1400.0 * (Q.log_sum_pt - 6.89) * (0.0532 - Q.D2_b2)
    if Q.log_sum_pt > 6.69 and Q.dr_2 < 0.0131:
        z += 1960.0 * (Q.log_sum_pt - 6.69) * (0.0131 - Q.dr_2)
    if Q.log_sum_pt > 6.9 and Q.dr_2 < 0.023:
        z += -2000.0 * (Q.log_sum_pt - 6.9) * (0.023 - Q.dr_2)
    if Q.log_sum_pt > 6.71 and Q.mean_eta < -0.00253:
        z += -1780.0 * (Q.log_sum_pt - 6.71) * (-0.00253 - Q.mean_eta)
    if Q.log_sum_pt > 6.71 and Q.mean_phi > 0.00234:
        z += -2370.0 * (Q.log_sum_pt - 6.71) * (Q.mean_phi - 0.00234)
    if Q.pt_5 < 22.7 and Q.n_pt_above_5 < 8.02:
        z += 0.28 * (22.7 - Q.pt_5) * (8.02 - Q.n_pt_above_5)
    if Q.sum_pt_top5 > 489.0 and Q.pt_6 < 49.6:
        z += 0.00033 * (Q.sum_pt_top5 - 489.0) * (49.6 - Q.pt_6)
    if Q.width < 0.00356 and Q.sj3_dr13 > 0.182:
        z += 25200.0 * (0.00356 - Q.width) * (Q.sj3_dr13 - 0.182)
    if Q.z_7 < 0.0486 and Q.mass_top5 < 71.3:
        z += 2.43 * (0.0486 - Q.z_7) * (71.3 - Q.mass_top5)
    if Q.z_7 < 0.0688 and Q.mean_phi2 < 0.00212:
        z += 16100.0 * (0.0688 - Q.z_7) * (0.00212 - Q.mean_phi2)
    if Q.z_7 < 0.049 and Q.sum_pt_top2 < 510.0:
        z += -1.04 * (0.049 - Q.z_7) * (510.0 - Q.sum_pt_top2)
    return max(0.0, z)


def neuron_6(Q):
    z = -0.0281
    if Q.e2_sq < 0.00292:
        z += -633.0 * Q.e2_sq + 1.84836
    if Q.girth2 < 0.00919:
        z += 858.0 * Q.girth2 - 7.88502
    if Q.lam1 < 0.00724:
        z += -1040.0 * Q.lam1 + 7.5296
    if Q.log_sum_pt < 6.26:
        z += -6.26 * Q.log_sum_pt + 39.1876
    if Q.max_dr >= 0.19:
        z += -25.9 * Q.max_dr + 4.921
    if Q.pt_6 < 15.3:
        z += -0.5782 * Q.pt_6 + 10.37442
    if 15.3 <= Q.pt_6 < 39.1:
        z += -0.0642 * Q.pt_6 + 2.51022
    if Q.sj2_dr < 0.18:
        z += -12.9 * Q.sj2_dr + 2.322
    if Q.sj2_dr >= 0.226:
        z += 23.5 * Q.sj2_dr - 5.311
    if Q.sum_pt < 617.0:
        z += -0.0155 * Q.sum_pt + 9.5635
    if Q.sum_pt_top5 >= 846.0:
        z += -0.0107 * Q.sum_pt_top5 + 9.0522
    if Q.tau2 >= 0.0098:
        z += -71.5 * Q.tau2 + 0.7007
    if Q.centroid_offset > 0.00767 and Q.mean_phi2 < 0.000229:
        z += -1020000.0 * (Q.centroid_offset - 0.00767) * (0.000229 - Q.mean_phi2)
    if Q.centroid_offset > 0.0174 and Q.mean_phi2 < 0.00444:
        z += -25900.0 * (Q.centroid_offset - 0.0174) * (0.00444 - Q.mean_phi2)
    if Q.centroid_offset > 0.00869 and Q.psi_0p1 > 0.352:
        z += -99.6 * (Q.centroid_offset - 0.00869) * (Q.psi_0p1 - 0.352)
    if Q.centroid_offset > 0.0179 and Q.pt_4 > 63.9:
        z += -4.24 * (Q.centroid_offset - 0.0179) * (Q.pt_4 - 63.9)
    if Q.log_sum_pt < 6.63 and Q.z_7 < 0.0758:
        z += 229.0 * (6.63 - Q.log_sum_pt) * (0.0758 - Q.z_7)
    if Q.mass < 63.4 and Q.phi_0 < -0.0103:
        z += -0.403 * (63.4 - Q.mass) * (-0.0103 - Q.phi_0)
    if Q.max_dr < 0.147 and Q.mean_phi > 0.00438:
        z += -1610.0 * (0.147 - Q.max_dr) * (Q.mean_phi - 0.00438)
    if Q.pt_6 < 41.7 and Q.M3 > 0.0775:
        z += 6.37 * (41.7 - Q.pt_6) * (Q.M3 - 0.0775)
    if Q.pt_6 < 40.5 and Q.log_sum_pt < 6.79:
        z += 0.649 * (40.5 - Q.pt_6) * (6.79 - Q.log_sum_pt)
    if Q.pt_6 < 39.9 and Q.z_7 > 0.02:
        z += -6.26 * (39.9 - Q.pt_6) * (Q.z_7 - 0.02)
    if Q.sj3_dr_max < 0.183 and Q.tau21_b2 < 0.129:
        z += 253.0 * (0.183 - Q.sj3_dr_max) * (0.129 - Q.tau21_b2)
    if Q.sum_pt > 989.0 and Q.mean_phi2 < 0.00241:
        z += 5.18 * (Q.sum_pt - 989.0) * (0.00241 - Q.mean_phi2)
    if Q.sum_pt > 993.0 and Q.n_pt_above_50 > 6.19:
        z += -0.00822 * (Q.sum_pt - 993.0) * (Q.n_pt_above_50 - 6.19)
    if Q.sum_pt < 633.0 and Q.pt_5 < 24.5:
        z += 0.00839 * (633.0 - Q.sum_pt) * (24.5 - Q.pt_5)
    if Q.sum_pt < 615.0 and Q.z_3 < 0.051:
        z += 28.7 * (615.0 - Q.sum_pt) * (0.051 - Q.z_3)
    if Q.tau1 < 0.128 and Q.mean_phi2 < 0.0142:
        z += 1710.0 * (0.128 - Q.tau1) * (0.0142 - Q.mean_phi2)
    if Q.width < 0.0122 and Q.mean_eta < -0.0267:
        z += 31700.0 * (0.0122 - Q.width) * (-0.0267 - Q.mean_eta)
    if Q.width < 0.0109 and Q.mean_phi < -0.0262:
        z += 18600.0 * (0.0109 - Q.width) * (-0.0262 - Q.mean_phi)
    if Q.width < 0.0127 and Q.mean_phi > 0.0262:
        z += 17700.0 * (0.0127 - Q.width) * (Q.mean_phi - 0.0262)
    return max(0.0, z)


def neuron_7(Q):
    z = 6.73
    if Q.centroid_offset >= 0.0347:
        z += -126.0 * Q.centroid_offset + 4.3722
    if Q.e2 >= 0.0352:
        z += 125.0 * Q.e2 - 4.4
    if Q.e2_sq < 0.000991:
        z += -2540.0 * Q.e2_sq + 2.51714
    if Q.girth < 0.0894:
        z += 93.5 * Q.girth - 8.3589
    if Q.girth2 < 0.00106:
        z += 3150.0 * Q.girth2 - 3.339
    if 0.00438 <= Q.girth2 < 0.00673:
        z += -742.0 * Q.girth2 + 3.24996
    if 0.00673 <= Q.girth2 < 0.0132:
        z += -2322.0 * Q.girth2 + 13.88336
    if Q.girth2 >= 0.0132:
        z += -1337.0 * Q.girth2 + 0.88136
    if 37.7 <= Q.mass < 77.2:
        z += 0.0631 * Q.mass - 2.37887
    if Q.mass >= 77.2:
        z += -0.0429 * Q.mass + 5.80433
    if 0.0795 <= Q.mass_over_sum_pt < 0.0906:
        z += 210.0 * Q.mass_over_sum_pt - 16.695
    if Q.mass_over_sum_pt >= 0.0906:
        z += -93.0 * Q.mass_over_sum_pt + 10.7568
    if Q.mass_top5 >= 52.4:
        z += 0.0738 * Q.mass_top5 - 3.86712
    if Q.planar_flow < 0.176:
        z += -4.74 * Q.planar_flow + 0.83424
    if Q.pt_7 < 29.3:
        z += 0.0731 * Q.pt_7 - 2.14183
    if Q.sj2_dr < 0.157:
        z += -16.0 * Q.sj2_dr + 1.6826
    if 0.157 <= Q.sj2_dr < 0.186:
        z += 28.6 * Q.sj2_dr - 5.3196
    if 0.137 <= Q.sj3_dr_max < 0.254:
        z += 11.6 * Q.sj3_dr_max - 1.5892
    if Q.sj3_dr_max >= 0.254:
        z += -2.5 * Q.sj3_dr_max + 1.9922
    if Q.tau1 < 0.0508:
        z += -31.4 * Q.tau1 + 1.59512
    if Q.width >= 0.00863:
        z += 945.0 * Q.width - 8.15535
    if Q.z_7 >= 0.0274:
        z += 30.9 * Q.z_7 - 0.84666
    if Q.centroid_offset < 0.0204 and Q.C2_b2 < 0.00469:
        z += -34900.0 * (0.0204 - Q.centroid_offset) * (0.00469 - Q.C2_b2)
    if Q.centroid_offset > 0.0308 and Q.n_pt_above_50 > 4.03:
        z += -40.8 * (Q.centroid_offset - 0.0308) * (Q.n_pt_above_50 - 4.03)
    if Q.centroid_offset > 0.0377 and Q.pt_4 > 81.3:
        z += -75.7 * (Q.centroid_offset - 0.0377) * (Q.pt_4 - 81.3)
    if Q.centroid_offset < 0.0197 and Q.tau21_b2 < 0.0317:
        z += 4750.0 * (0.0197 - Q.centroid_offset) * (0.0317 - Q.tau21_b2)
    if Q.centroid_offset < 0.0209 and Q.zdr_5 > 0.0079:
        z += 11600.0 * (0.0209 - Q.centroid_offset) * (Q.zdr_5 - 0.0079)
    if Q.e2_sq < 0.000995 and Q.phi_0 > 0.0412:
        z += -510000.0 * (0.000995 - Q.e2_sq) * (Q.phi_0 - 0.0412)
    if Q.girth < 0.0876 and Q.mean_phi < 0.00261:
        z += 1360.0 * (0.0876 - Q.girth) * (0.00261 - Q.mean_phi)
    if Q.girth2 < 0.00372 and Q.mean_eta < -0.00688:
        z += 64000.0 * (0.00372 - Q.girth2) * (-0.00688 - Q.mean_eta)
    if Q.girth2 > 0.00441 and Q.planar_flow < 0.208:
        z += 3330.0 * (Q.girth2 - 0.00441) * (0.208 - Q.planar_flow)
    if Q.girth2_top2 < 0.00108 and Q.centroid_offset > 0.00962:
        z += 167000.0 * (0.00108 - Q.girth2_top2) * (Q.centroid_offset - 0.00962)
    if Q.girth2_top2 < 0.000993 and Q.tau21_b2 < 0.0274:
        z += -195000.0 * (0.000993 - Q.girth2_top2) * (0.0274 - Q.tau21_b2)
    if Q.mass_over_sum_pt > 0.0793 and Q.pt_6 < 34.7:
        z += -5.79 * (Q.mass_over_sum_pt - 0.0793) * (34.7 - Q.pt_6)
    if Q.planar_flow < 0.167 and Q.pt_4 > 71.8:
        z += 0.177 * (0.167 - Q.planar_flow) * (Q.pt_4 - 71.8)
    if Q.planar_flow < 0.169 and Q.pt_6 < 35.5:
        z += -0.523 * (0.169 - Q.planar_flow) * (35.5 - Q.pt_6)
    if Q.planar_flow < 0.21 and Q.width > 0.00761:
        z += -5560.0 * (0.21 - Q.planar_flow) * (Q.width - 0.00761)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.485
    if Q.width < 0.00355:
        z += -2120.0 * Q.width + 7.526
    if Q.girth < 0.0572 and Q.girth2 < 0.00478:
        z += -37400.0 * (0.0572 - Q.girth) * (0.00478 - Q.girth2)
    if Q.girth2 < 0.00696 and Q.centroid_offset < 0.0288:
        z += 20300.0 * (0.00696 - Q.girth2) * (0.0288 - Q.centroid_offset)
    if Q.girth2 < 0.00673 and Q.log_sum_pt < 6.57:
        z += -840.0 * (0.00673 - Q.girth2) * (6.57 - Q.log_sum_pt)
    if Q.log_sum_pt > 6.73 and Q.centroid_offset < 0.0302:
        z += -226.0 * (Q.log_sum_pt - 6.73) * (0.0302 - Q.centroid_offset)
    if Q.mass < 21.5 and Q.lam2 < 0.000241:
        z += -902.0 * (21.5 - Q.mass) * (0.000241 - Q.lam2)
    if Q.mass < 48.6 and Q.lam2 < 0.000168:
        z += 531.0 * (48.6 - Q.mass) * (0.000168 - Q.lam2)
    if Q.width < 0.00393 and Q.centroid_offset > 0.00848:
        z += -67100.0 * (0.00393 - Q.width) * (Q.centroid_offset - 0.00848)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.65
    if Q.centroid_offset < 0.018:
        z += -286.0 * Q.centroid_offset + 5.148
    if Q.dr_2 < 0.0193:
        z += -147.0 * Q.dr_2 + 2.8371
    if Q.girth2 < 0.0059:
        z += -1890.0 * Q.girth2 + 11.151
    if Q.m012 >= 26.5:
        z += -0.104 * Q.m012 + 2.756
    if Q.mass < 27.8:
        z += 0.217 * Q.mass - 6.0326
    if Q.max_dr >= 0.278:
        z += -35.2 * Q.max_dr + 9.7856
    if Q.pt_4 < 34.4:
        z += -0.179 * Q.pt_4 + 6.1576
    if Q.sj3_dr_max < 0.226:
        z += -13.3 * Q.sj3_dr_max + 3.0058
    if Q.width < 0.00017:
        z += -18300.0 * Q.width + 3.111
    z += -9.57 * Q.z_dr_0p2_0p4
    if Q.centroid_offset < 0.0192 and Q.mean_eta > -0.00207:
        z += 10100.0 * (0.0192 - Q.centroid_offset) * (Q.mean_eta - -0.00207)
    if Q.centroid_offset < 0.0281 and Q.pt_1 < 186.0:
        z += 0.503 * (0.0281 - Q.centroid_offset) * (186.0 - Q.pt_1)
    if Q.girth2 < 0.00493 and Q.mean_phi > 0.0256:
        z += -205000.0 * (0.00493 - Q.girth2) * (Q.mean_phi - 0.0256)
    if Q.girth2 < 0.00579 and Q.mean_phi < 0.000234:
        z += -37000.0 * (0.00579 - Q.girth2) * (0.000234 - Q.mean_phi)
    if Q.girth2 < 0.00814 and Q.planar_flow < 0.349:
        z += 1240.0 * (0.00814 - Q.girth2) * (0.349 - Q.planar_flow)
    if Q.lam1 > 0.00723 and Q.eccentricity > 0.625:
        z += -741.0 * (Q.lam1 - 0.00723) * (Q.eccentricity - 0.625)
    if Q.log_sum_pt > 6.89 and Q.D2_b2 < 0.731:
        z += -60.6 * (Q.log_sum_pt - 6.89) * (0.731 - Q.D2_b2)
    if Q.log_sum_pt > 6.89 and Q.zdr_5 < 0.00145:
        z += 18600.0 * (Q.log_sum_pt - 6.89) * (0.00145 - Q.zdr_5)
    if Q.mass < 56.4 and Q.D2 > 2.26:
        z += 0.017 * (56.4 - Q.mass) * (Q.D2 - 2.26)
    if Q.mass < 53.3 and Q.centroid_offset < 0.0258:
        z += 8.19 * (53.3 - Q.mass) * (0.0258 - Q.centroid_offset)
    if Q.mass < 38.9 and Q.log_sum_pt < 6.8:
        z += 0.0929 * (38.9 - Q.mass) * (6.8 - Q.log_sum_pt)
    if Q.pt_4 < 30.7 and Q.M3 > 0.0506:
        z += 38.5 * (30.7 - Q.pt_4) * (Q.M3 - 0.0506)
    if Q.sj3_pair_mass_max < 26.4 and Q.absphi_2 > 0.00759:
        z += -3.02 * (26.4 - Q.sj3_pair_mass_max) * (Q.absphi_2 - 0.00759)
    if Q.tau1 < 0.0433 and Q.eccentricity > 0.897:
        z += -768.0 * (0.0433 - Q.tau1) * (Q.eccentricity - 0.897)
    if Q.tau1 < 0.0512 and Q.tau21_b2 < 0.0288:
        z += -7450.0 * (0.0512 - Q.tau1) * (0.0288 - Q.tau21_b2)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.75
    if Q.centroid_offset < 0.0228:
        z += 42.6 * Q.centroid_offset - 0.97128
    if Q.girth2 < 0.00289:
        z += 807.0 * Q.girth2 - 2.33223
    if Q.girth2 >= 0.0252:
        z += -208.0 * Q.girth2 + 5.2416
    if Q.girth2_top3 < 0.00222:
        z += -650.0 * Q.girth2_top3 + 1.443
    if Q.lam1 < 0.00552:
        z += 570.0 * Q.lam1 - 3.1464
    if Q.lam1 >= 0.00752:
        z += 213.0 * Q.lam1 - 1.60176
    if Q.lam2 >= 0.000307:
        z += 2270.0 * Q.lam2 - 0.69689
    if Q.log_sum_pt < 6.05:
        z += 11.8 * Q.log_sum_pt - 71.39
    if Q.mass_top5 >= 45.3:
        z += -0.0224 * Q.mass_top5 + 1.01472
    if Q.mean_eta >= 0.0243:
        z += 33.7 * Q.mean_eta - 0.81891
    if Q.mean_phi < 0.00665:
        z += 15.6 * Q.mean_phi - 0.10374
    if Q.pt_7 < 45.2:
        z += -0.0578 * Q.pt_7 + 2.61256
    if Q.sj3_pair_mass_min >= 14.5:
        z += 0.16 * Q.sj3_pair_mass_min - 2.32
    if Q.sum_pt >= 986.0:
        z += 0.00699 * Q.sum_pt - 6.89214
    if Q.z_7 < 0.0667:
        z += 50.9 * Q.z_7 - 3.39503
    if Q.LHA > 0.284 and Q.sj3_mass1 > 6.1:
        z += 5.21 * (Q.LHA - 0.284) * (Q.sj3_mass1 - 6.1)
    if Q.lam1 > 0.00681 and Q.D2_b2 < 0.484:
        z += -408.0 * (Q.lam1 - 0.00681) * (0.484 - Q.D2_b2)
    if Q.lam1 < 0.00588 and Q.z_dr_0p05_0p1 > 0.263:
        z += -1210.0 * (0.00588 - Q.lam1) * (Q.z_dr_0p05_0p1 - 0.263)
    if Q.lam2 > -2.81e-05 and Q.planar_flow > 0.0763:
        z += -951.0 * (Q.lam2 - -2.81e-05) * (Q.planar_flow - 0.0763)
    if Q.lam2 > 0.00107 and Q.pt_6 < 33.2:
        z += -74.2 * (Q.lam2 - 0.00107) * (33.2 - Q.pt_6)
    if Q.pt_7 < 50.9 and Q.D2 < 1.07:
        z += 0.125 * (50.9 - Q.pt_7) * (1.07 - Q.D2)
    if Q.tau1 > 0.0524 and Q.D2 < 1.12:
        z += 26.2 * (Q.tau1 - 0.0524) * (1.12 - Q.D2)
    if Q.zdr_0 < 0.0213 and Q.centroid_offset > 0.0104:
        z += 6020.0 * (0.0213 - Q.zdr_0) * (Q.centroid_offset - 0.0104)
    if Q.zdr_0 < 0.0203 and Q.dr_7 > 0.0795:
        z += 693.0 * (0.0203 - Q.zdr_0) * (Q.dr_7 - 0.0795)
    if Q.zdr_0 < 0.0217 and Q.z_dr_0p05_0p1 < 0.321:
        z += 272.0 * (0.0217 - Q.zdr_0) * (0.321 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_11(Q):
    z = -4.06
    if Q.centroid_offset < 0.0407:
        z += -142.0 * Q.centroid_offset + 5.7794
    if Q.centroid_offset >= 0.0479:
        z += -267.0 * Q.centroid_offset + 12.7893
    if Q.girth < 0.0206:
        z += 416.2 * Q.girth - 13.72376
    if 0.0206 <= Q.girth < 0.0264:
        z += 213.2 * Q.girth - 9.54196
    if 0.0264 <= Q.girth < 0.0718:
        z += 86.2 * Q.girth - 6.18916
    if Q.girth2 < 0.0128:
        z += -902.0 * Q.girth2 + 11.5456
    if Q.m012 >= 43.0:
        z += -0.0499 * Q.m012 + 2.1457
    if Q.planar_flow < 0.281:
        z += -8.43 * Q.planar_flow + 2.36883
    if Q.sj3_dr_max < 0.173:
        z += 10.9 * Q.sj3_dr_max - 0.7373
    if 0.173 <= Q.sj3_dr_max < 0.26:
        z += -13.2 * Q.sj3_dr_max + 3.432
    if Q.sum_pt >= 1010.0:
        z += -0.0102 * Q.sum_pt + 10.302
    if Q.sum_pt_top5 < 535.0:
        z += 0.00598 * Q.sum_pt_top5 - 3.1993
    if Q.z_7 >= 0.0176:
        z += 42.3 * Q.z_7 - 0.74448
    if Q.z_dr_0p05_0p1 >= 0.846:
        z += 7.76 * Q.z_dr_0p05_0p1 - 6.56496
    if Q.centroid_offset < 0.0144 and Q.D2_b2 < 0.55:
        z += 418.0 * (0.0144 - Q.centroid_offset) * (0.55 - Q.D2_b2)
    if Q.centroid_offset < 0.0147 and Q.sj3_pair_mass_min < 15.4:
        z += -20.0 * (0.0147 - Q.centroid_offset) * (15.4 - Q.sj3_pair_mass_min)
    if Q.centroid_offset < 0.0363 and Q.sum_pt < 912.0:
        z += -0.129 * (0.0363 - Q.centroid_offset) * (912.0 - Q.sum_pt)
    if Q.centroid_offset < 0.0135 and Q.zdr_7 < 0.00286:
        z += -64500.0 * (0.0135 - Q.centroid_offset) * (0.00286 - Q.zdr_7)
    if Q.girth < 0.0285 and Q.sj3_pair_mass_min > 1.58:
        z += -28.8 * (0.0285 - Q.girth) * (Q.sj3_pair_mass_min - 1.58)
    if Q.girth2 < 0.0111 and Q.D2 < 0.44:
        z += 796.0 * (0.0111 - Q.girth2) * (0.44 - Q.D2)
    if Q.girth2 < 0.0151 and Q.pt_6 > 21.1:
        z += 1.93 * (0.0151 - Q.girth2) * (Q.pt_6 - 21.1)
    if Q.girth2 < 0.00462 and Q.sj3_dr13 > 0.165:
        z += 12600.0 * (0.00462 - Q.girth2) * (Q.sj3_dr13 - 0.165)
    if Q.mass < 48.1 and Q.D2 < 1.33:
        z += -0.103 * (48.1 - Q.mass) * (1.33 - Q.D2)
    if Q.mass < 14.7 and Q.phi_0 < -0.00424:
        z += -8.15 * (14.7 - Q.mass) * (-0.00424 - Q.phi_0)
    if Q.mass < 16.0 and Q.zdr_7 < 0.00455:
        z += 48.0 * (16.0 - Q.mass) * (0.00455 - Q.zdr_7)
    if Q.planar_flow < 0.265 and Q.e3 < 1.08e-05:
        z += -1070000.0 * (0.265 - Q.planar_flow) * (1.08e-05 - Q.e3)
    if Q.sj2_dr > 0.161 and Q.dr_7 < 0.0501:
        z += 311.0 * (Q.sj2_dr - 0.161) * (0.0501 - Q.dr_7)
    if Q.sj2_dr > 0.164 and Q.lam2 < 0.00122:
        z += -14200.0 * (Q.sj2_dr - 0.164) * (0.00122 - Q.lam2)
    if Q.sj2_dr > 0.18 and Q.n_pt_above_50 > 4.21:
        z += -3.86 * (Q.sj2_dr - 0.18) * (Q.n_pt_above_50 - 4.21)
    if Q.sj3_dr_max < 0.304 and Q.phi_0 < -0.0201:
        z += 188.0 * (0.304 - Q.sj3_dr_max) * (-0.0201 - Q.phi_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.988
    if Q.centroid_offset >= 0.0454:
        z += 46.1 * Q.centroid_offset - 2.09294
    if Q.girth2 >= 0.0189:
        z += 215.0 * Q.girth2 - 4.0635
    if Q.mass >= 82.7:
        z += 0.0582 * Q.mass - 4.81314
    if Q.zdr_0 >= 0.0398:
        z += -81.7 * Q.zdr_0 + 3.25166
    if Q.girth2 > 0.0139 and Q.lam2 > -2.24e-05:
        z += -10300.0 * (Q.girth2 - 0.0139) * (Q.lam2 - -2.24e-05)
    if Q.girth2_top2 > 0.00676 and Q.mean_phi < -0.00459:
        z += -1090.0 * (Q.girth2_top2 - 0.00676) * (-0.00459 - Q.mean_phi)
    if Q.mass > 60.7 and Q.centroid_offset > 0.00297:
        z += 1.01 * (Q.mass - 60.7) * (Q.centroid_offset - 0.00297)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.41
    if Q.girth < 0.148:
        z += -44.3 * Q.girth + 6.5564
    if Q.girth2_top2 < 0.000432:
        z += 3680.0 * Q.girth2_top2 - 1.58976
    if Q.log_sum_pt >= 6.49:
        z += 5.32 * Q.log_sum_pt - 34.5268
    if Q.mass >= 50.5:
        z += -0.0462 * Q.mass + 2.3331
    if Q.sj3_dr23 >= 0.189:
        z += -7.23 * Q.sj3_dr23 + 1.36647
    if Q.sum_pt >= 988.0:
        z += 0.0362 * Q.sum_pt - 35.7656
    if Q.width < 0.0146:
        z += -212.0 * Q.width + 3.0952
    if Q.girth < 0.129 and Q.M3 < 0.0793:
        z += 354.0 * (0.129 - Q.girth) * (0.0793 - Q.M3)
    if Q.girth < 0.147 and Q.log_sum_pt < 6.87:
        z += -40.5 * (0.147 - Q.girth) * (6.87 - Q.log_sum_pt)
    if Q.girth < 0.153 and Q.pt_7 < 39.7:
        z += -1.7 * (0.153 - Q.girth) * (39.7 - Q.pt_7)
    if Q.girth < 0.146 and Q.tau2 > 0.00911:
        z += 936.0 * (0.146 - Q.girth) * (Q.tau2 - 0.00911)
    if Q.pt_6 < 31.4 and Q.z_7 < 0.0719:
        z += -5.92 * (31.4 - Q.pt_6) * (0.0719 - Q.z_7)
    if Q.sum_pt_top5 > 774.0 and Q.pt_6 < 29.8:
        z += 0.000999 * (Q.sum_pt_top5 - 774.0) * (29.8 - Q.pt_6)
    if Q.sum_pt_top5 > 640.0 and Q.pt_7 < 43.0:
        z += 0.000767 * (Q.sum_pt_top5 - 640.0) * (43.0 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.0652
    if Q.LHA >= 0.305:
        z += 37.7 * Q.LHA - 11.4985
    if Q.centroid_offset >= 0.0464:
        z += -94.8 * Q.centroid_offset + 4.39872
    if 0.0414 <= Q.e2 < 0.051:
        z += 108.0 * Q.e2 - 4.4712
    if Q.e2 >= 0.051:
        z += 34.7 * Q.e2 - 0.7329
    if Q.e3 < 8.4e-05:
        z += 14500.0 * Q.e3 - 1.218
    if 0.0266 <= Q.girth < 0.087:
        z += 44.7 * Q.girth - 1.18902
    if Q.girth >= 0.087:
        z += -100.3 * Q.girth + 11.42598
    if Q.mass >= 67.3:
        z += -0.0275 * Q.mass + 1.85075
    if 0.0798 <= Q.mass_over_sum_pt < 0.0889:
        z += 135.0 * Q.mass_over_sum_pt - 10.773
    if Q.mass_over_sum_pt >= 0.0889:
        z += -21.0 * Q.mass_over_sum_pt + 3.0954
    if Q.n_dr_0_0p05 < 4.97:
        z += -0.166 * Q.n_dr_0_0p05 + 0.82502
    if Q.psi_0p1 >= 0.975:
        z += -27.7 * Q.psi_0p1 + 27.0075
    if Q.sd_mass < 50.8:
        z += -0.0025 * Q.sd_mass - 0.39088
    if 50.8 <= Q.sd_mass < 75.0:
        z += 0.0214 * Q.sd_mass - 1.605
    if Q.sd_rg >= 0.237:
        z += -16.0 * Q.sd_rg + 3.792
    if 0.159 <= Q.sj2_dr < 0.201:
        z += 26.1 * Q.sj2_dr - 4.1499
    if Q.sj2_dr >= 0.201:
        z += 11.5 * Q.sj2_dr - 1.2153
    if Q.sum_pt < 717.0:
        z += 0.00419 * Q.sum_pt - 3.00423
    if 0.00364 <= Q.width < 0.00672:
        z += 268.0 * Q.width - 0.97552
    if 0.00672 <= Q.width < 0.0125:
        z += -802.0 * Q.width + 6.21488
    if Q.width >= 0.0125:
        z += -125.0 * Q.width - 2.24762
    if Q.z_dr_0p05_0p1 < 0.161:
        z += 1.9 * Q.z_dr_0p05_0p1 - 0.3059
    if Q.z_dr_0p05_0p1 >= 0.739:
        z += 2.26 * Q.z_dr_0p05_0p1 - 1.67014
    if Q.centroid_offset > 0.0514 and Q.C2_b2 < 0.000846:
        z += -371000.0 * (Q.centroid_offset - 0.0514) * (0.000846 - Q.C2_b2)
    if Q.centroid_offset > 0.0257 and Q.pt_1 < 158.0:
        z += 0.638 * (Q.centroid_offset - 0.0257) * (158.0 - Q.pt_1)
    if Q.e2_sq > 0.00219 and Q.sj2_dr < 0.187:
        z += -12000.0 * (Q.e2_sq - 0.00219) * (0.187 - Q.sj2_dr)
    if Q.e3 < 5.72e-05 and Q.sj3_dr23 > 0.165:
        z += 249000.0 * (5.72e-05 - Q.e3) * (Q.sj3_dr23 - 0.165)
    if Q.planar_flow < 0.115 and Q.centroid_offset < 0.0183:
        z += -1150.0 * (0.115 - Q.planar_flow) * (0.0183 - Q.centroid_offset)
    if Q.planar_flow < 0.107 and Q.lam1 < 0.0165:
        z += 2230.0 * (0.107 - Q.planar_flow) * (0.0165 - Q.lam1)
    if Q.planar_flow < 0.109 and Q.lam1 < 0.00734:
        z += -10400.0 * (0.109 - Q.planar_flow) * (0.00734 - Q.lam1)
    if Q.planar_flow < 0.112 and Q.width < 0.00597:
        z += 5680.0 * (0.112 - Q.planar_flow) * (0.00597 - Q.width)
    if Q.planar_flow < 0.118 and Q.z_dr_0p05_0p1 > 0.682:
        z += -19.9 * (0.118 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.682)
    if Q.planar_flow < 0.136 and Q.zdr_7 < 0.00163:
        z += -3200.0 * (0.136 - Q.planar_flow) * (0.00163 - Q.zdr_7)
    if Q.psi_0p1 > 0.82 and Q.D2_b2 < 0.085:
        z += 98.1 * (Q.psi_0p1 - 0.82) * (0.085 - Q.D2_b2)
    if Q.psi_0p1 > 0.778 and Q.phi_4 > 0.101:
        z += 34.0 * (Q.psi_0p1 - 0.778) * (Q.phi_4 - 0.101)
    if Q.sd_mass < 73.8 and Q.sj2_mass2 > 3.15:
        z += 0.00329 * (73.8 - Q.sd_mass) * (Q.sj2_mass2 - 3.15)
    if Q.sd_rg > 0.281 and Q.D2_b2 < 0.271:
        z += -131.0 * (Q.sd_rg - 0.281) * (0.271 - Q.D2_b2)
    if Q.sj2_dr > 0.13 and Q.D2_b2 < 0.09:
        z += -271.0 * (Q.sj2_dr - 0.13) * (0.09 - Q.D2_b2)
    if Q.sj2_dr > 0.158 and Q.D2_b2 < 0.0893:
        z += 646.0 * (Q.sj2_dr - 0.158) * (0.0893 - Q.D2_b2)
    if Q.sj2_dr > 0.2 and Q.D2_b2 < 0.087:
        z += -625.0 * (Q.sj2_dr - 0.2) * (0.087 - Q.D2_b2)
    if Q.sj2_dr > 0.159 and Q.dr_2 < 0.033:
        z += -1100.0 * (Q.sj2_dr - 0.159) * (0.033 - Q.dr_2)
    if Q.sj2_dr > 0.202 and Q.dr_2 < 0.0526:
        z += 452.0 * (Q.sj2_dr - 0.202) * (0.0526 - Q.dr_2)
    if Q.sj2_dr > 0.147 and Q.dr_3 < 0.0231:
        z += -578.0 * (Q.sj2_dr - 0.147) * (0.0231 - Q.dr_3)
    if Q.z_dr_0p05_0p1 < 0.605 and Q.D2_b2 < 0.156:
        z += -7.23 * (0.605 - Q.z_dr_0p05_0p1) * (0.156 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 > 0.753 and Q.zdr_5 > 0.0155:
        z += -12700.0 * (Q.z_dr_0p05_0p1 - 0.753) * (Q.zdr_5 - 0.0155)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.0382
    if Q.N2 < 0.25:
        z += -3.98 * Q.N2 + 0.995
    if Q.e2 < 0.0413:
        z += -54.5 * Q.e2 + 2.25085
    if Q.girth2 < 0.00816:
        z += 575.0 * Q.girth2 - 4.692
    if Q.girth2_top2 < 0.000542:
        z += 2320.0 * Q.girth2_top2 - 1.25744
    if Q.girth2_top3 < 0.00189:
        z += 704.0 * Q.girth2_top3 - 1.33056
    if Q.sj2_dr < 0.161:
        z += -13.7 * Q.sj2_dr + 1.5801
    if 0.161 <= Q.sj2_dr < 0.195:
        z += 18.4 * Q.sj2_dr - 3.588
    if Q.width < 0.02:
        z += -112.0 * Q.width + 2.24
    if Q.N2 < 0.207 and Q.sum_pt_top5 > 412.0:
        z += 0.0139 * (0.207 - Q.N2) * (Q.sum_pt_top5 - 412.0)
    if Q.N2 < 0.266 and Q.z_dr_0p05_0p1 < 0.489:
        z += -9.26 * (0.266 - Q.N2) * (0.489 - Q.z_dr_0p05_0p1)
    if Q.girth2 < 0.00776 and Q.D2 < 0.765:
        z += -3030.0 * (0.00776 - Q.girth2) * (0.765 - Q.D2)
    if Q.mass_over_sum_pt < 0.141 and Q.D2 < 0.774:
        z += 46.6 * (0.141 - Q.mass_over_sum_pt) * (0.774 - Q.D2)
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
