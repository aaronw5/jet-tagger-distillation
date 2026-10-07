"""JEDI-linear jet tagger, 8 particles, 3 features: one term per observable per neuron (from the 399), re-tuned on the network's predictions (all observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 88.8% of jets.

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
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
  Q.zdr_5                  pT share × ΔR of particle 5 (its term in ΣzΔR)
  Q.zdr_6                  pT share × ΔR of particle 6 (its term in ΣzΔR)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_z_dr2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        sj2_mass1=subjets(2)["mass"][0],
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_5=z[5] * dr[5],
        zdr_6=z[6] * dr[6],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        phi_0=phi[0],
        sum_pt_top5=sum(pt[:5]),
        sum_z_dr2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
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
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -0.5151538
    if Q.C2_b2 < 0.00151:
        z += 1235.492 * Q.C2_b2 - 1.865593
    if Q.centroid_offset >= 0.03782766:
        z += -73.03883 * Q.centroid_offset + 2.762888
    if Q.e2 < 0.03227056:
        z += -138.8416 * Q.e2 + 4.480497
    if Q.sum_z_dr < 0.0751:
        z += 47.68999 * Q.sum_z_dr - 3.581518
    if Q.sum_z_dr2 < 0.013:
        z += -517.6904 * Q.sum_z_dr2 + 6.729976
    if Q.mass < 53.20476:
        z += 0.1022279 * Q.mass - 5.439013
    if Q.planar_flow < 0.159:
        z += -7.426687 * Q.planar_flow + 1.180843
    if Q.sj3_dr_max >= 0.2122942:
        z += -24.99049 * Q.sj3_dr_max + 5.305334
    if Q.sum_pt >= 906.0:
        z += -0.01318488 * Q.sum_pt + 11.94551
    if Q.lam1_plus_lam2 < 0.00444:
        z += 1359.067 * Q.lam1_plus_lam2 - 6.034256
    if Q.centroid_offset > 0.0203 and Q.z_7 < 0.0609:
        z += -2483.941 * (Q.centroid_offset - 0.0203) * (0.0609 - Q.z_7)
    if Q.sum_z_dr2 < 0.0133 and Q.D2 < 1.02:
        z += 307.7768 * (0.0133 - Q.sum_z_dr2) * (1.02 - Q.D2)
    if Q.sum_z_dr2 < 0.0134 and Q.phi_0 > -0.00668:
        z += 973.4155 * (0.0134 - Q.sum_z_dr2) * (Q.phi_0 - -0.00668)
    if Q.lam1 < 0.00649 and Q.D2 < 0.843:
        z += -1724.232 * (0.00649 - Q.lam1) * (0.843 - Q.D2)
    if Q.lam2 < 8.01e-05 and Q.D2_b2 < 0.257:
        z += 38615.86 * (8.01e-05 - Q.lam2) * (0.257 - Q.D2_b2)
    if Q.log_sum_pt > 6.68 and Q.dr_4 < 0.0746:
        z += 53.73243 * (Q.log_sum_pt - 6.68) * (0.0746 - Q.dr_4)
    if Q.log_sum_pt > 6.68 and Q.z_7 > 0.0178:
        z += -349.6169 * (Q.log_sum_pt - 6.68) * (Q.z_7 - 0.0178)
    if Q.planar_flow < 0.142 and Q.dr_2 < 0.023:
        z += -750.9349 * (0.142 - Q.planar_flow) * (0.023 - Q.dr_2)
    if Q.planar_flow < 0.165 and Q.z_7 < 0.0237:
        z += 762.629 * (0.165 - Q.planar_flow) * (0.0237 - Q.z_7)
    if Q.sj3_dr_max < 0.245 and Q.z_7 < 0.058:
        z += -92.54891 * (0.245 - Q.sj3_dr_max) * (0.058 - Q.z_7)
    if Q.sum_pt > 915.0 and Q.pt_7 > 35.4:
        z += 0.0003820313 * (Q.sum_pt - 915.0) * (Q.pt_7 - 35.4)
    if Q.lam1_plus_lam2 < 0.00414 and Q.C3 < 0.0364:
        z += 20224.57 * (0.00414 - Q.lam1_plus_lam2) * (0.0364 - Q.C3)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.312535
    if Q.sum_z_dr < 0.0991:
        z += 32.68671 * Q.sum_z_dr - 3.239253
    if Q.sum_z_dr2 < 0.00871:
        z += 194.9853 * Q.sum_z_dr2 - 1.698322
    if Q.lam1 < 0.00085:
        z += 842.9896 * Q.lam1 - 0.7165411
    if Q.log_sum_pt >= 6.538933:
        z += 12.34482 * Q.log_sum_pt - 80.72197
    if Q.mass_over_sum_pt_sq < 0.00591:
        z += -386.0909 * Q.mass_over_sum_pt_sq + 2.281797
    if Q.pt_7 >= 53.5:
        z += -0.09031572 * Q.pt_7 + 4.831891
    if Q.sum_pt_top5 >= 580.375:
        z += -0.00463182 * Q.sum_pt_top5 + 2.688192
    if Q.z_7 < 0.0801183:
        z += 33.81866 * Q.z_7 - 2.709494
    if Q.sum_z_dr2 < 0.00939 and Q.centroid_offset > 0.0216:
        z += 10740.6 * (0.00939 - Q.sum_z_dr2) * (Q.centroid_offset - 0.0216)
    if Q.lam1 < 0.00871 and Q.n_pt_above_50 > 5.25:
        z += -6.231827 * (0.00871 - Q.lam1) * (Q.n_pt_above_50 - 5.25)
    if Q.lam1 < 0.00962 and Q.planar_flow < 0.15:
        z += -553.7211 * (0.00962 - Q.lam1) * (0.15 - Q.planar_flow)
    if Q.log_sum_pt > 6.48 and Q.D2 < 1.4:
        z += 2.420868 * (Q.log_sum_pt - 6.48) * (1.4 - Q.D2)
    if Q.log_sum_pt > 6.37 and Q.centroid_offset > 0.0126:
        z += 91.55984 * (Q.log_sum_pt - 6.37) * (Q.centroid_offset - 0.0126)
    if Q.pt_7 > 33.4 and Q.pt_6 < 53.4:
        z += -0.005537638 * (Q.pt_7 - 33.4) * (53.4 - Q.pt_6)
    if Q.pt_7 > 33.5 and Q.sj2_dr > 0.143:
        z += 0.6202644 * (Q.pt_7 - 33.5) * (Q.sj2_dr - 0.143)
    if Q.sj3_dr_max > 0.104 and Q.sj3_pair_mass_min > 2.6:
        z += -0.07829343 * (Q.sj3_dr_max - 0.104) * (Q.sj3_pair_mass_min - 2.6)
    if Q.z_7 < 0.0607 and Q.D2 < 1.52:
        z += -32.72108 * (0.0607 - Q.z_7) * (1.52 - Q.D2)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.5956023
    if Q.LHA >= 0.131:
        z += -13.12174 * Q.LHA + 1.718948
    if Q.sum_z_dr < 0.00765:
        z += 127.8763 * Q.sum_z_dr - 0.978254
    if Q.lam2 < 0.000201:
        z += -1171.906 * Q.lam2 + 0.2355532
    if Q.log_sum_pt < 6.48:
        z += -4.906531 * Q.log_sum_pt + 31.79432
    if Q.m012 >= 42.1:
        z += 0.0302202 * Q.m012 - 1.27227
    if Q.mass < 35.5:
        z += 0.04685808 * Q.mass - 1.663462
    if Q.mass_over_sum_pt < 0.105:
        z += -21.32485 * Q.mass_over_sum_pt + 2.239109
    if Q.max_dr < 0.236:
        z += -0.271544 * Q.max_dr + 0.06408439
    if Q.planar_flow < 0.502:
        z += 1.142162 * Q.planar_flow - 0.5733655
    z += 0.11238 * Q.pt_7
    if Q.sum_pt < 788.6875:
        z += -0.008873274 * Q.sum_pt + 6.99824
    if Q.z_7 >= 0.0366:
        z += -60.27797 * Q.z_7 + 2.206174
    if Q.lam1 < 0.00609 and Q.max_dr > 0.0769:
        z += -2664.835 * (0.00609 - Q.lam1) * (Q.max_dr - 0.0769)
    if Q.lam1 < 0.00496 and Q.pt_6 > 17.3:
        z += 0.7175919 * (0.00496 - Q.lam1) * (Q.pt_6 - 17.3)
    if Q.log_sum_pt > 6.85 and Q.D2_b2 < 1.31:
        z += -9.822411 * (Q.log_sum_pt - 6.85) * (1.31 - Q.D2_b2)
    if Q.log_sum_pt > 6.95 and Q.pt_6 > 40.0:
        z += -0.04824585 * (Q.log_sum_pt - 6.95) * (Q.pt_6 - 40.0)
    if Q.sj3_pair_mass_max < 59.4 and Q.centroid_offset > 0.0112:
        z += -1.257734 * (59.4 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.0112)
    if Q.sum_pt > 538.0 and Q.lam2 < 0.000194:
        z += -9.399145 * (Q.sum_pt - 538.0) * (0.000194 - Q.lam2)
    if Q.sum_pt_top5 > 745.0 and Q.D2_b2 < 1.1:
        z += 0.00757855 * (Q.sum_pt_top5 - 745.0) * (1.1 - Q.D2_b2)
    if Q.z_7 > 0.0548 and Q.sj3_dr_min < 0.0684:
        z += 19.91823 * (Q.z_7 - 0.0548) * (0.0684 - Q.sj3_dr_min)
    return max(0.0, z)


def neuron_3(Q):
    z = -7.574845
    if Q.centroid_offset >= 0.00928:
        z += 66.88956 * Q.centroid_offset - 0.6207351
    if Q.e2 >= 0.046:
        z += 61.38023 * Q.e2 - 2.823491
    if Q.sum_z_dr >= 0.02670378:
        z += 110.0682 * Q.sum_z_dr - 2.939237
    if Q.sum_z_dr2 >= 0.007498352:
        z += -1061.106 * Q.sum_z_dr2 + 7.956544
    if Q.lam1 >= 0.00828:
        z += 422.9207 * Q.lam1 - 3.501784
    if Q.mass >= 65.1:
        z += -0.0407429 * Q.mass + 2.652363
    if Q.mass_over_sum_pt_sq < 0.0118:
        z += 345.1144 * Q.mass_over_sum_pt_sq - 4.07235
    if Q.n_dr_0_0p05 < 1.04:
        z += -0.3541876 * Q.n_dr_0_0p05 + 0.3683551
    if Q.sd_mass >= 54.15028:
        z += 0.01336667 * Q.sd_mass - 0.7238091
    if Q.sj2_dr >= 0.1488981:
        z += 22.64466 * Q.sj2_dr - 3.371748
    if Q.sj3_dr_max >= 0.1786811:
        z += 29.40709 * Q.sj3_dr_max - 5.254493
    if Q.tau1 < 0.1023433:
        z += -33.3644 * Q.tau1 + 3.414622
    if Q.z_dr_0_0p05 < 0.836:
        z += 0.5003487 * Q.z_dr_0_0p05 - 0.4182916
    if Q.sum_z_dr > 0.0307 and Q.log_sum_pt > 6.11:
        z += 55.26632 * (Q.sum_z_dr - 0.0307) * (Q.log_sum_pt - 6.11)
    if Q.lam2 > 0.000776 and Q.pt_6 < 64.0:
        z += 15.47021 * (Q.lam2 - 0.000776) * (64.0 - Q.pt_6)
    if Q.sj2_dr > 0.214 and Q.dr_3 < 0.0549:
        z += -294.686 * (Q.sj2_dr - 0.214) * (0.0549 - Q.dr_3)
    if Q.sj2_dr > 0.206 and Q.sj2_mass1 > -0.0277:
        z += -0.3179487 * (Q.sj2_dr - 0.206) * (Q.sj2_mass1 - -0.0277)
    if Q.sj2_dr > 0.199 and Q.z_dr_0p05_0p1 < 1.02:
        z += -6.188033 * (Q.sj2_dr - 0.199) * (1.02 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.2833952
    if Q.C2 < 0.0651:
        z += -70.51814 * Q.C2 + 4.590731
    if Q.LHA < 0.351:
        z += -19.68449 * Q.LHA + 6.909256
    if Q.N2 < 0.224:
        z += -56.00799 * Q.N2 + 12.54579
    if Q.sum_zz_dr2 < 0.0116:
        z += -668.4394 * Q.sum_zz_dr2 + 7.753897
    if Q.sum_z_dr2 < 0.00919:
        z += 982.4705 * Q.sum_z_dr2 - 9.028904
    if Q.sum_z_dr2_top2 < 0.00779:
        z += -210.2255 * Q.sum_z_dr2_top2 + 1.637657
    if Q.lam2 < 0.000876:
        z += 2411.06 * Q.lam2 - 2.112089
    if Q.mass >= 45.1:
        z += -0.09764683 * Q.mass + 4.403872
    if Q.mass_over_sum_pt >= 0.0903:
        z += -78.5389 * Q.mass_over_sum_pt + 7.092063
    if Q.sj3_dr_max < 0.237:
        z += 30.39407 * Q.sj3_dr_max - 7.203395
    if Q.sum_pt < 753.0:
        z += 0.004623138 * Q.sum_pt - 3.481223
    if Q.lam1_plus_lam2 >= 0.00339:
        z += 284.2091 * Q.lam1_plus_lam2 - 0.9634688
    if Q.N2 < 0.219 and Q.sum_zz_dr2 > 0.0116:
        z += -925.0621 * (0.219 - Q.N2) * (Q.sum_zz_dr2 - 0.0116)
    if Q.N2 < 0.224 and Q.eccentricity > 0.727:
        z += -133.7478 * (0.224 - Q.N2) * (Q.eccentricity - 0.727)
    if Q.N2 < 0.219 and Q.mass < 63.2:
        z += -0.7947584 * (0.219 - Q.N2) * (63.2 - Q.mass)
    if Q.N2 < 0.242 and Q.pt_7 < 52.9:
        z += -0.4988942 * (0.242 - Q.N2) * (52.9 - Q.pt_7)
    if Q.sum_z_dr2_top2 < 0.0076 and Q.C2_b2 > 0.000612:
        z += -38339.48 * (0.0076 - Q.sum_z_dr2_top2) * (Q.C2_b2 - 0.000612)
    if Q.lam2 < 0.000453 and Q.dr01 < 0.173:
        z += -11397.32 * (0.000453 - Q.lam2) * (0.173 - Q.dr01)
    if Q.max_dr > 0.164 and Q.C2_b2 < 0.000596:
        z += -39234.75 * (Q.max_dr - 0.164) * (0.000596 - Q.C2_b2)
    if Q.sd_mass > 37.0 and Q.sd_zg < 0.276:
        z += -0.2705164 * (Q.sd_mass - 37.0) * (0.276 - Q.sd_zg)
    if Q.sd_mass > 42.1 and Q.sj3_pair_mass_min < 26.6:
        z += 0.003210545 * (Q.sd_mass - 42.1) * (26.6 - Q.sj3_pair_mass_min)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.09480149
    if Q.sum_z_dr < 0.00764:
        z += -325.9132 * Q.sum_z_dr + 2.489977
    if Q.log_sum_pt >= 6.83:
        z += -35.57932 * Q.log_sum_pt + 243.0067
    if Q.pt_7 < 35.4:
        z += 0.1670491 * Q.pt_7 - 5.91354
    if Q.lam1_plus_lam2 < 0.0039:
        z += -776.0207 * Q.lam1_plus_lam2 + 3.026481
    if Q.z_7 < 0.05222815:
        z += -131.41 * Q.z_7 + 6.863301
    if Q.zdr_0 < 0.021:
        z += 62.93374 * Q.zdr_0 - 1.321609
    if Q.LHA < 0.215 and Q.lam1 < 0.00121:
        z += -8412.584 * (0.215 - Q.LHA) * (0.00121 - Q.lam1)
    if Q.LHA < 0.215 and Q.log_sum_pt < 6.78:
        z += -106.2288 * (0.215 - Q.LHA) * (6.78 - Q.log_sum_pt)
    if Q.sum_z_dr2 < 0.00167 and Q.centroid_offset < 0.024:
        z += 95298.26 * (0.00167 - Q.sum_z_dr2) * (0.024 - Q.centroid_offset)
    if Q.log_sum_pt > 6.73 and Q.dr_2 < 0.0147:
        z += 563.7656 * (Q.log_sum_pt - 6.73) * (0.0147 - Q.dr_2)
    if Q.log_sum_pt > 6.7 and Q.mean_eta2 < 8.48e-05:
        z += 69932.5 * (Q.log_sum_pt - 6.7) * (8.48e-05 - Q.mean_eta2)
    if Q.log_sum_pt > 6.7 and Q.mean_phi > 0.00334:
        z += -969.1973 * (Q.log_sum_pt - 6.7) * (Q.mean_phi - 0.00334)
    if Q.sum_pt_top5 > 530.0 and Q.pt_6 < 43.2:
        z += 0.0003270519 * (Q.sum_pt_top5 - 530.0) * (43.2 - Q.pt_6)
    if Q.sum_pt_top5 > 438.0 and Q.sj2_dr > 0.176:
        z += 0.01940468 * (Q.sum_pt_top5 - 438.0) * (Q.sj2_dr - 0.176)
    if Q.z_7 < 0.0687 and Q.centroid_offset < 0.031:
        z += -582.9348 * (0.0687 - Q.z_7) * (0.031 - Q.centroid_offset)
    if Q.z_7 < 0.0708 and Q.mass_top3 < 45.4:
        z += 0.7643175 * (0.0708 - Q.z_7) * (45.4 - Q.mass_top3)
    return max(0.0, z)


def neuron_6(Q):
    z = 4.278525
    if Q.centroid_offset < 0.02343572:
        z += 126.4598 * Q.centroid_offset - 2.963677
    if Q.sum_zz_dr2 < 0.00301:
        z += -295.476 * Q.sum_zz_dr2 + 0.8893827
    if Q.sum_z_dr2 < 0.00867:
        z += 1219.883 * Q.sum_z_dr2 - 10.57639
    if Q.lam1 < 0.0118:
        z += -374.2115 * Q.lam1 + 4.415696
    if Q.lam2 < 0.00109:
        z += 387.8213 * Q.lam2 - 0.4227252
    if Q.mass < 22.5:
        z += -0.06316171 * Q.mass + 1.421138
    if Q.max_dr < 0.144:
        z += 31.88767 * Q.max_dr - 4.591824
    if Q.pt_6 < 19.1:
        z += -0.7906107 * Q.pt_6 + 15.10066
    if Q.sj3_dr_max < 0.1416042:
        z += -16.52885 * Q.sj3_dr_max + 2.340554
    if Q.sj3_dr_min >= 0.0225:
        z += -13.40088 * Q.sj3_dr_min + 0.3015199
    if Q.sj3_pair_mass_min >= 4.35:
        z += -0.05654115 * Q.sj3_pair_mass_min + 0.245954
    if Q.sum_pt >= 975.0:
        z += 0.01623777 * Q.sum_pt - 15.83182
    if Q.z_6 < 0.0221:
        z += 594.5014 * Q.z_6 - 13.13848
    if Q.centroid_offset > 0.0213 and Q.mean_phi2 < 0.01:
        z += 2729.503 * (Q.centroid_offset - 0.0213) * (0.01 - Q.mean_phi2)
    if Q.centroid_offset > 0.0103 and Q.psi_0p1 > 0.335:
        z += 45.43639 * (Q.centroid_offset - 0.0103) * (Q.psi_0p1 - 0.335)
    if Q.centroid_offset > 0.017 and Q.pt_4 > 66.3:
        z += 2.528438 * (Q.centroid_offset - 0.017) * (Q.pt_4 - 66.3)
    if Q.lam1 < 0.0118 and Q.planar_flow < 0.254:
        z += -1462.126 * (0.0118 - Q.lam1) * (0.254 - Q.planar_flow)
    if Q.lam2 < 0.0032 and Q.n_dr_0_0p05 < 3.0:
        z += 27.86754 * (0.0032 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.mass < 64.5 and Q.z_dr_0p2_0p4 < 0.0534:
        z += -1.003757 * (64.5 - Q.mass) * (0.0534 - Q.z_dr_0p2_0p4)
    if Q.mean_eta > 0.0349 and Q.M3 > 0.0492:
        z += -1001.652 * (Q.mean_eta - 0.0349) * (Q.M3 - 0.0492)
    if Q.pt_6 < 42.5 and Q.log_sum_pt < 6.76:
        z += 0.6726989 * (42.5 - Q.pt_6) * (6.76 - Q.log_sum_pt)
    if Q.pt_6 < 41.5 and Q.z_7 > 0.0235:
        z += -6.094355 * (41.5 - Q.pt_6) * (Q.z_7 - 0.0235)
    if Q.sj2_dr < 0.171 and Q.eccentricity > 0.927:
        z += 268.7487 * (0.171 - Q.sj2_dr) * (Q.eccentricity - 0.927)
    if Q.sum_pt > 953.0 and Q.mean_phi2 < 0.00311:
        z += -2.42665 * (Q.sum_pt - 953.0) * (0.00311 - Q.mean_phi2)
    if Q.sum_pt < 589.0 and Q.n_dr_0p2_0p4 < 1.97:
        z += 0.0007850977 * (589.0 - Q.sum_pt) * (1.97 - Q.n_dr_0p2_0p4)
    if Q.sum_pt < 625.0 and Q.pt_5 < 24.6:
        z += 0.01092875 * (625.0 - Q.sum_pt) * (24.6 - Q.pt_5)
    if Q.tau1 < 0.111 and Q.mean_phi2 < 0.0157:
        z += 1484.21 * (0.111 - Q.tau1) * (0.0157 - Q.mean_phi2)
    return max(0.0, z)


def neuron_7(Q):
    z = 7.459191
    if Q.D2_b2 < 0.0586:
        z += 12.07829 * Q.D2_b2 - 0.7077879
    if Q.centroid_offset >= 0.0384:
        z += -112.5953 * Q.centroid_offset + 4.323658
    if Q.sum_z_dr < 0.0884:
        z += 60.07981 * Q.sum_z_dr - 5.311055
    if Q.sum_z_dr2 >= 0.01855051:
        z += 387.2719 * Q.sum_z_dr2 - 7.184092
    if Q.sum_z_dr2_top2 < 0.00109:
        z += -983.3116 * Q.sum_z_dr2_top2 + 1.07181
    if Q.lam2 < 0.000271:
        z += 2439.785 * Q.lam2 - 0.6611816
    if Q.mass < 64.54392:
        z += 0.03685498 * Q.mass - 2.378765
    if Q.mass_over_sum_pt >= 0.0909:
        z += -576.4877 * Q.mass_over_sum_pt + 52.40273
    if Q.mass_top5 >= 53.0:
        z += 0.03286753 * Q.mass_top5 - 1.741979
    if Q.max_dr < 0.06139287:
        z += -28.92986 * Q.max_dr + 1.776087
    if Q.pt_7 < 29.4:
        z += 0.05600893 * Q.pt_7 - 1.646663
    if Q.sd_rg >= 0.283:
        z += 16.71743 * Q.sd_rg - 4.731033
    if Q.sj2_dr < 0.09317241:
        z += -27.39263 * Q.sj2_dr + 2.552237
    if Q.sj3_dr_max >= 0.07254204:
        z += 7.310002 * Q.sj3_dr_max - 0.5302825
    if Q.tau21_b2 < 0.0407:
        z += 41.61093 * Q.tau21_b2 - 1.693565
    if Q.lam1_plus_lam2 < 0.00664:
        z += 839.0575 * Q.lam1_plus_lam2 - 5.571342
    if Q.z_7 >= 0.0342:
        z += -9.6339 * Q.z_7 + 0.3294794
    if Q.z_dr_0p1_0p2 < 0.139:
        z += 0.7552495 * Q.z_dr_0p1_0p2 - 0.1049797
    if Q.centroid_offset < 0.02 and Q.C2_b2 < 0.00426:
        z += -36062.02 * (0.02 - Q.centroid_offset) * (0.00426 - Q.C2_b2)
    if Q.centroid_offset > 0.0258 and Q.n_pt_above_50 > 3.71:
        z += -19.11046 * (Q.centroid_offset - 0.0258) * (Q.n_pt_above_50 - 3.71)
    if Q.centroid_offset < 0.0216 and Q.tau21_b2 < 0.0279:
        z += 3771.006 * (0.0216 - Q.centroid_offset) * (0.0279 - Q.tau21_b2)
    if Q.sum_z_dr < 0.0893 and Q.C2_b2 < 0.00429:
        z += 1893.358 * (0.0893 - Q.sum_z_dr) * (0.00429 - Q.C2_b2)
    if Q.sum_z_dr2 > 0.0145 and Q.eccentricity > 0.925:
        z += -2945.268 * (Q.sum_z_dr2 - 0.0145) * (Q.eccentricity - 0.925)
    if Q.sum_z_dr2 > 0.00447 and Q.planar_flow < 0.197:
        z += 414.1526 * (Q.sum_z_dr2 - 0.00447) * (0.197 - Q.planar_flow)
    if Q.sum_z_dr2 > 0.0131 and Q.pt_6 < 41.3:
        z += 125.9785 * (Q.sum_z_dr2 - 0.0131) * (41.3 - Q.pt_6)
    if Q.sum_z_dr2_top2 < 0.00107 and Q.tau21_b2 < 0.0257:
        z += -53874.93 * (0.00107 - Q.sum_z_dr2_top2) * (0.0257 - Q.tau21_b2)
    if Q.lam2 < 0.0003 and Q.tau21_b2 < 0.0408:
        z += 164932.7 * (0.0003 - Q.lam2) * (0.0408 - Q.tau21_b2)
    if Q.mass > 76.2 and Q.zdr_6 > 0.00612:
        z += -33.35809 * (Q.mass - 76.2) * (Q.zdr_6 - 0.00612)
    if Q.mass_over_sum_pt > 0.0911 and Q.pt_6 < 40.8:
        z += 16.06244 * (Q.mass_over_sum_pt - 0.0911) * (40.8 - Q.pt_6)
    if Q.planar_flow < 0.216 and Q.pt_6 < 35.6:
        z += -0.1302154 * (0.216 - Q.planar_flow) * (35.6 - Q.pt_6)
    if Q.sj3_dr_max > 0.132 and Q.pt_6 < 20.0:
        z += -0.119102 * (Q.sj3_dr_max - 0.132) * (20.0 - Q.pt_6)
    if Q.lam1_plus_lam2 > 0.00864 and Q.pt_6 < 40.8:
        z += -121.1679 * (Q.lam1_plus_lam2 - 0.00864) * (40.8 - Q.pt_6)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.359718
    if Q.sum_z_dr < 0.0595:
        z += 223.0589 * Q.sum_z_dr - 13.272
    if Q.sum_z_dr2 < 0.00491:
        z += -1027.213 * Q.sum_z_dr2 + 5.043613
    if Q.log_sum_pt >= 6.7:
        z += -8.117471 * Q.log_sum_pt + 54.38705
    if Q.lam1_plus_lam2 < 0.00355:
        z += -762.0562 * Q.lam1_plus_lam2 + 2.7053
    if Q.sum_z_dr < 0.0585 and Q.lam1_plus_lam2 > 0.000656:
        z += 31321.85 * (0.0585 - Q.sum_z_dr) * (Q.lam1_plus_lam2 - 0.000656)
    if Q.sum_z_dr2 < 0.00662 and Q.centroid_offset < 0.0259:
        z += 35706.89 * (0.00662 - Q.sum_z_dr2) * (0.0259 - Q.centroid_offset)
    if Q.lam2 < 0.000136 and Q.D2_b2 < 5.27:
        z += 1407.333 * (0.000136 - Q.lam2) * (5.27 - Q.D2_b2)
    if Q.mass < 31.3 and Q.centroid_offset < 0.0291:
        z += -6.064369 * (31.3 - Q.mass) * (0.0291 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.142 and Q.centroid_offset > 0.0059:
        z += -911.6516 * (0.142 - Q.sj3_dr_max) * (Q.centroid_offset - 0.0059)
    if Q.sum_pt_top5 > 679.0 and Q.n_pt_above_10 < 8.0:
        z += -0.008194927 * (Q.sum_pt_top5 - 679.0) * (8.0 - Q.n_pt_above_10)
    if Q.tau1 < 0.0617 and Q.lam1_plus_lam2 < 0.00304:
        z += 26598.87 * (0.0617 - Q.tau1) * (0.00304 - Q.lam1_plus_lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -3.456924
    if Q.D2 >= 2.16:
        z += -0.1628197 * Q.D2 + 0.3516906
    if Q.M3 < 0.0796:
        z += -16.90733 * Q.M3 + 1.345823
    if Q.centroid_offset < 0.0185:
        z += -169.845 * Q.centroid_offset + 3.142132
    if Q.e2 >= 0.0456:
        z += 30.15997 * Q.e2 - 1.375295
    if Q.sum_zz_dr2 < 0.00327:
        z += 2043.86 * Q.sum_zz_dr2 - 6.683422
    if Q.sum_z_dr2 < 0.00365:
        z += -3840.823 * Q.sum_z_dr2 + 14.019
    if Q.lam2 < 0.000315:
        z += -5465.607 * Q.lam2 + 1.721666
    if Q.mass < 41.26805:
        z += 0.1434847 * Q.mass - 5.921332
    if Q.max_dr < 0.107:
        z += -15.14431 * Q.max_dr + 1.620441
    if Q.sj3_dr_max < 0.1059541:
        z += 39.79504 * Q.sj3_dr_max - 4.216447
    if Q.sum_pt < 986.0:
        z += -0.009030937 * Q.sum_pt + 8.904504
    if Q.lam1_plus_lam2 < 0.000176:
        z += -5082.183 * Q.lam1_plus_lam2 + 0.8944642
    if Q.zdr_0 < 0.00631:
        z += 88.53168 * Q.zdr_0 - 0.5586349
    if Q.centroid_offset < 0.017 and Q.n_for_90pct > 5.0:
        z += -15.0339 * (0.017 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.centroid_offset < 0.019 and Q.pt_1 < 164.0:
        z += -2.07871 * (0.019 - Q.centroid_offset) * (164.0 - Q.pt_1)
    if Q.centroid_offset < 0.0191 and Q.z_2nd < 0.205:
        z += 1079.192 * (0.0191 - Q.centroid_offset) * (0.205 - Q.z_2nd)
    if Q.sum_z_dr < 0.0528 and Q.tau21_b2 < 0.0275:
        z += 4424.884 * (0.0528 - Q.sum_z_dr) * (0.0275 - Q.tau21_b2)
    if Q.sum_z_dr < 0.0549 and Q.z_6 > 0.0338:
        z += -1709.08 * (0.0549 - Q.sum_z_dr) * (Q.z_6 - 0.0338)
    if Q.sum_z_dr2 < 0.00578 and Q.mean_phi < 0.00437:
        z += -10056.53 * (0.00578 - Q.sum_z_dr2) * (0.00437 - Q.mean_phi)
    if Q.lam1 > 0.00485 and Q.eccentricity > 0.638:
        z += -1538.514 * (Q.lam1 - 0.00485) * (Q.eccentricity - 0.638)
    if Q.log_sum_pt > 6.89 and Q.D2_b2 < 0.746:
        z += -28.71643 * (Q.log_sum_pt - 6.89) * (0.746 - Q.D2_b2)
    if Q.mass < 54.9 and Q.centroid_offset < 0.0267:
        z += 6.570714 * (54.9 - Q.mass) * (0.0267 - Q.centroid_offset)
    if Q.mass < 55.0 and Q.log_sum_pt < 6.84:
        z += 0.1115224 * (55.0 - Q.mass) * (6.84 - Q.log_sum_pt)
    if Q.sj3_dr_max < 0.15 and Q.pt_6 > 32.9:
        z += 0.6888987 * (0.15 - Q.sj3_dr_max) * (Q.pt_6 - 32.9)
    if Q.tau1 < 0.046 and Q.mean_phi < -0.0119:
        z += 3038.516 * (0.046 - Q.tau1) * (-0.0119 - Q.mean_phi)
    if Q.tau1 < 0.0457 and Q.tau21_b2 < 0.0286:
        z += -5956.994 * (0.0457 - Q.tau1) * (0.0286 - Q.tau21_b2)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.737618
    if Q.C2_b2 >= 0.00763:
        z += 24.4247 * Q.C2_b2 - 0.1863605
    if Q.LHA >= 0.305:
        z += -34.16759 * Q.LHA + 10.42112
    if Q.e3 >= 5.22e-05:
        z += -1298.758 * Q.e3 + 0.06779518
    if Q.sum_z_dr2 >= 0.006688641:
        z += 155.2479 * Q.sum_z_dr2 - 1.038398
    if Q.sum_z_dr2_top3 < 0.00215:
        z += -421.7054 * Q.sum_z_dr2_top3 + 0.9066666
    if Q.lam1 < 0.0041726:
        z += 1054.677 * Q.lam1 - 4.400743
    if Q.lam2 >= 0.000278:
        z += 1272.379 * Q.lam2 - 0.3537215
    if Q.mass < 79.4:
        z += -0.03067978 * Q.mass + 2.435975
    if Q.mass_top5 >= 51.0:
        z += -0.01444076 * Q.mass_top5 + 0.7364785
    if Q.pt_7 < 43.9:
        z += 0.01258967 * Q.pt_7 - 0.5526863
    if Q.sj3_dr_max >= 0.184:
        z += -3.736351 * Q.sj3_dr_max + 0.6874886
    if Q.sj3_pair_mass_min >= 15.8:
        z += 0.1099469 * Q.sj3_pair_mass_min - 1.737161
    if Q.sum_pt >= 988.0:
        z += -0.01183664 * Q.sum_pt + 11.6946
    if Q.tau1 >= 0.057:
        z += 27.98722 * Q.tau1 - 1.595271
    if Q.lam1 > 0.00707 and Q.D2_b2 < 0.36:
        z += -195.1278 * (Q.lam1 - 0.00707) * (0.36 - Q.D2_b2)
    if Q.lam2 > 0.000257 and Q.planar_flow > 0.055:
        z += -624.0246 * (Q.lam2 - 0.000257) * (Q.planar_flow - 0.055)
    if Q.pt_7 < 45.6 and Q.D2 < 0.985:
        z += 0.09071884 * (45.6 - Q.pt_7) * (0.985 - Q.D2)
    if Q.pt_7 < 48.0 and Q.log_sum_pt < 6.56:
        z += -0.1227532 * (48.0 - Q.pt_7) * (6.56 - Q.log_sum_pt)
    if Q.sj3_pair_mass_min > 6.68 and Q.sj3_pairmin_over_m > 0.245:
        z += -0.1945542 * (Q.sj3_pair_mass_min - 6.68) * (Q.sj3_pairmin_over_m - 0.245)
    if Q.zdr_0 < 0.0205 and Q.centroid_offset > 0.0125:
        z += 1895.118 * (0.0205 - Q.zdr_0) * (Q.centroid_offset - 0.0125)
    if Q.zdr_0 < 0.0242 and Q.z_dr_0p05_0p1 < 0.248:
        z += 87.16476 * (0.0242 - Q.zdr_0) * (0.248 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.9894507
    if Q.centroid_offset >= 0.05:
        z += -475.5109 * Q.centroid_offset + 23.77554
    if Q.sum_zz_dr2 < 0.00814:
        z += 1333.169 * Q.sum_zz_dr2 - 10.852
    if Q.sum_z_dr < 0.06656016:
        z += 69.82938 * Q.sum_z_dr - 4.647855
    if Q.sum_z_dr2 < 0.0134:
        z += -144.7924 * Q.sum_z_dr2 + 1.940218
    if Q.lam1 < 0.00838:
        z += 758.7842 * Q.lam1 - 6.358612
    if Q.mass >= 21.67283:
        z += 0.03409773 * Q.mass - 0.7389945
    if Q.n_dr_0p1_0p2 >= 2.92:
        z += -0.3368911 * Q.n_dr_0p1_0p2 + 0.983722
    if Q.planar_flow < 0.242:
        z += -5.770055 * Q.planar_flow + 1.396353
    if Q.pt_6 < 25.4:
        z += 0.05754632 * Q.pt_6 - 1.461676
    if Q.pt_7 >= 29.6:
        z += -0.01181715 * Q.pt_7 + 0.3497876
    if Q.sj3_dr_max < 0.1059541:
        z += 8.784928 * Q.sj3_dr_max - 0.9307992
    if Q.sum_pt >= 971.0:
        z += -0.003600333 * Q.sum_pt + 3.495923
    if Q.lam1_plus_lam2 < 0.00872:
        z += -2480.016 * Q.lam1_plus_lam2 + 21.62574
    if Q.z_7 >= 0.0169:
        z += 12.86985 * Q.z_7 - 0.2175004
    if Q.z_dr_0p05_0p1 >= 0.826:
        z += 2.188431 * Q.z_dr_0p05_0p1 - 1.807644
    if Q.centroid_offset < 0.0138 and Q.D2_b2 < 0.571:
        z += 346.2979 * (0.0138 - Q.centroid_offset) * (0.571 - Q.D2_b2)
    if Q.centroid_offset < 0.0384 and Q.absphi_0 < 0.0324:
        z += 42.40367 * (0.0384 - Q.centroid_offset) * (0.0324 - Q.absphi_0)
    if Q.centroid_offset < 0.0383 and Q.mean_phi < -0.00159:
        z += -850.2109 * (0.0383 - Q.centroid_offset) * (-0.00159 - Q.mean_phi)
    if Q.centroid_offset < 0.0137 and Q.sj3_pair_mass_min < 17.2:
        z += -4.760756 * (0.0137 - Q.centroid_offset) * (17.2 - Q.sj3_pair_mass_min)
    if Q.centroid_offset < 0.0377 and Q.sum_pt < 892.0:
        z += 0.01008306 * (0.0377 - Q.centroid_offset) * (892.0 - Q.sum_pt)
    if Q.centroid_offset > 0.05 and Q.tau3 > 0.000494:
        z += -57082.21 * (Q.centroid_offset - 0.05) * (Q.tau3 - 0.000494)
    if Q.centroid_offset > 0.0494 and Q.tau32 < 0.619:
        z += 1515.478 * (Q.centroid_offset - 0.0494) * (0.619 - Q.tau32)
    if Q.centroid_offset > 0.0492 and Q.zdr_6 > 0.00714:
        z += -27305.31 * (Q.centroid_offset - 0.0492) * (Q.zdr_6 - 0.00714)
    if Q.sum_z_dr2 < 0.00431 and Q.sj3_dr13 > 0.162:
        z += 5921.102 * (0.00431 - Q.sum_z_dr2) * (Q.sj3_dr13 - 0.162)
    if Q.planar_flow < 0.233 and Q.e3 < 9.53e-06:
        z += -437700.0 * (0.233 - Q.planar_flow) * (9.53e-06 - Q.e3)
    if Q.planar_flow < 0.268 and Q.pt_7 < 36.7:
        z += -0.2200251 * (0.268 - Q.planar_flow) * (36.7 - Q.pt_7)
    if Q.planar_flow < 0.241 and Q.sum_pt < 835.0:
        z += -0.01968991 * (0.241 - Q.planar_flow) * (835.0 - Q.sum_pt)
    if Q.sj3_dr_max < 0.264 and Q.phi_0 < -0.023:
        z += 135.8026 * (0.264 - Q.sj3_dr_max) * (-0.023 - Q.phi_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.7255036
    if Q.centroid_offset >= 0.0499:
        z += 23.08439 * Q.centroid_offset - 1.151911
    if Q.e2 >= 0.0634:
        z += -31.00831 * Q.e2 + 1.965927
    if Q.sum_z_dr2 >= 0.0188:
        z += 217.8609 * Q.sum_z_dr2 - 4.095785
    if Q.sum_z_dr2_top2 >= 0.014:
        z += -24.1304 * Q.sum_z_dr2_top2 + 0.3378255
    if Q.mass >= 91.2:
        z += 0.0663778 * Q.mass - 6.053656
    if Q.zdr_0 >= 0.0398:
        z += -35.08496 * Q.zdr_0 + 1.396381
    if Q.sum_z_dr2 > 0.0188 and Q.lam2 > 0.000537:
        z += 12118.79 * (Q.sum_z_dr2 - 0.0188) * (Q.lam2 - 0.000537)
    if Q.sum_z_dr2 > 0.0188 and Q.pt_7 < 53.4:
        z += -1.29323 * (Q.sum_z_dr2 - 0.0188) * (53.4 - Q.pt_7)
    if Q.mass > 69.6 and Q.centroid_offset > 0.00948:
        z += 0.6646283 * (Q.mass - 69.6) * (Q.centroid_offset - 0.00948)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.2787811
    if Q.e2 < 0.079:
        z += 48.45113 * Q.e2 - 3.827639
    if Q.e3 < 5.05e-05:
        z += -43715.66 * Q.e3 + 2.207641
    if Q.sum_z_dr < 0.148:
        z += -58.90503 * Q.sum_z_dr + 8.717944
    if Q.lam1 < 0.0161:
        z += -240.455 * Q.lam1 + 3.871326
    if Q.mass >= 49.1:
        z += -0.01767231 * Q.mass + 0.8677104
    if Q.pt_7 >= 48.4:
        z += -0.03354203 * Q.pt_7 + 1.623434
    if Q.sum_pt_top5 >= 902.75:
        z += 0.007457922 * Q.sum_pt_top5 - 6.732639
    if Q.z_5 < 0.0283:
        z += 94.383 * Q.z_5 - 2.671039
    if Q.z_7 < 0.028:
        z += 98.27762 * Q.z_7 - 2.751773
    if Q.e3 < 6.94e-05 and Q.D3 > 0.0598:
        z += 367.4676 * (6.94e-05 - Q.e3) * (Q.D3 - 0.0598)
    if Q.e3 < 5.38e-05 and Q.centroid_offset < 0.0377:
        z += -1638702.0 * (5.38e-05 - Q.e3) * (0.0377 - Q.centroid_offset)
    if Q.sum_z_dr < 0.15 and Q.M3 < 0.0746:
        z += 63.64009 * (0.15 - Q.sum_z_dr) * (0.0746 - Q.M3)
    if Q.sum_z_dr < 0.148 and Q.lam2 < 0.000536:
        z += 433.0117 * (0.148 - Q.sum_z_dr) * (0.000536 - Q.lam2)
    if Q.sum_z_dr < 0.148 and Q.log_sum_pt < 6.81:
        z += -71.61163 * (0.148 - Q.sum_z_dr) * (6.81 - Q.log_sum_pt)
    if Q.sum_z_dr < 0.145 and Q.pt_7 < 38.2:
        z += -1.227456 * (0.145 - Q.sum_z_dr) * (38.2 - Q.pt_7)
    if Q.sum_z_dr < 0.152 and Q.sj2_mass1 > 30.4:
        z += -1.621106 * (0.152 - Q.sum_z_dr) * (Q.sj2_mass1 - 30.4)
    if Q.sum_z_dr < 0.142 and Q.tau2 > 0.00886:
        z += 303.3386 * (0.142 - Q.sum_z_dr) * (Q.tau2 - 0.00886)
    if Q.sum_z_dr < 0.153 and Q.z_7 > 0.0616:
        z += 412.8142 * (0.153 - Q.sum_z_dr) * (Q.z_7 - 0.0616)
    if Q.pt_6 < 31.9 and Q.z_7 < 0.0624:
        z += -2.762401 * (31.9 - Q.pt_6) * (0.0624 - Q.z_7)
    if Q.sum_pt > 1000.0 and Q.pt_6 < 70.0:
        z += -0.0004760964 * (Q.sum_pt - 1000.0) * (70.0 - Q.pt_6)
    if Q.sum_pt_top5 > 667.0 and Q.pt_7 < 38.1:
        z += 0.0009394609 * (Q.sum_pt_top5 - 667.0) * (38.1 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = 2.971718
    if Q.C2_b2 < 0.00403:
        z += 132.2524 * Q.C2_b2 - 0.5329773
    if Q.centroid_offset >= 0.0497:
        z += -1119.707 * Q.centroid_offset + 55.64946
    if Q.e2 < 0.04107712:
        z += -114.2578 * Q.e2 + 4.693382
    if Q.sum_zz_dr2 >= 0.0117:
        z += 1789.695 * Q.sum_zz_dr2 - 20.93944
    if Q.e3 < 7.93e-05:
        z += 10184.96 * Q.e3 - 0.8076676
    if Q.sum_z_dr < 0.1003996:
        z += 125.049 * Q.sum_z_dr - 12.55487
    if Q.sum_z_dr2 >= 0.00856:
        z += -1308.056 * Q.sum_z_dr2 + 11.19696
    if Q.lam2 < 0.000504:
        z += -2086.988 * Q.lam2 + 1.051842
    if Q.mass >= 68.9:
        z += 0.01098107 * Q.mass - 0.7565958
    if Q.mass_over_sum_pt >= 0.08453854:
        z += 264.3161 * Q.mass_over_sum_pt - 22.34489
    if Q.n_dr_0_0p05 < 5.06:
        z += -0.08948237 * Q.n_dr_0_0p05 + 0.4527808
    if Q.psi_0p1 >= 0.974:
        z += -3.528407 * Q.psi_0p1 + 3.436668
    if Q.sd_mass < 44.81215:
        z += -0.03822904 * Q.sd_mass + 1.713125
    if Q.sd_rg < 0.1441415:
        z += 11.74874 * Q.sd_rg - 1.693481
    if Q.sj2_dr >= 0.159004:
        z += 27.46886 * Q.sj2_dr - 4.367658
    if Q.sum_pt < 717.0:
        z += 0.006437885 * Q.sum_pt - 4.615963
    if Q.lam1_plus_lam2 >= 0.008566101:
        z += -1891.918 * Q.lam1_plus_lam2 + 16.20636
    if Q.z_dr_0_0p05 >= 0.152:
        z += 1.196048 * Q.z_dr_0_0p05 - 0.1817994
    if Q.z_dr_0p05_0p1 < 0.163:
        z += 3.331389 * Q.z_dr_0p05_0p1 - 0.5430164
    if Q.z_dr_0p1_0p2 < 0.0677:
        z += 2.646598 * Q.z_dr_0p1_0p2 - 0.1791747
    if Q.centroid_offset > 0.0499 and Q.C2_b2 < 0.000872:
        z += 1190362.0 * (Q.centroid_offset - 0.0499) * (0.000872 - Q.C2_b2)
    if Q.centroid_offset > 0.0267 and Q.pt_1 < 152.0:
        z += 0.007816087 * (Q.centroid_offset - 0.0267) * (152.0 - Q.pt_1)
    if Q.e3 < 5.56e-05 and Q.sj3_dr23 > 0.162:
        z += 235749.5 * (5.56e-05 - Q.e3) * (Q.sj3_dr23 - 0.162)
    if Q.planar_flow < 0.106 and Q.centroid_offset < 0.0179:
        z += -29.54235 * (0.106 - Q.planar_flow) * (0.0179 - Q.centroid_offset)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.00746:
        z += -1411.689 * (0.11 - Q.planar_flow) * (0.00746 - Q.lam1)
    if Q.planar_flow < 0.109 and Q.mean_eta > -0.00684:
        z += -34.67633 * (0.109 - Q.planar_flow) * (Q.mean_eta - -0.00684)
    if Q.planar_flow < 0.1 and Q.sum_pt_top5 < 652.0:
        z += 0.006157964 * (0.1 - Q.planar_flow) * (652.0 - Q.sum_pt_top5)
    if Q.planar_flow < 0.112 and Q.lam1_plus_lam2 < 0.00601:
        z += 243.8218 * (0.112 - Q.planar_flow) * (0.00601 - Q.lam1_plus_lam2)
    if Q.sj2_dr > 0.198 and Q.D2_b2 < 0.0872:
        z += -59.15445 * (Q.sj2_dr - 0.198) * (0.0872 - Q.D2_b2)
    if Q.sj2_dr > 0.164 and Q.dr_2 < 0.0332:
        z += -618.0536 * (Q.sj2_dr - 0.164) * (0.0332 - Q.dr_2)
    if Q.sj2_dr > 0.203 and Q.z_5 < 0.105:
        z += -269.9726 * (Q.sj2_dr - 0.203) * (0.105 - Q.z_5)
    if Q.z_dr_0p05_0p1 < 0.627 and Q.sum_pt < 714.0:
        z += 0.0007673862 * (0.627 - Q.z_dr_0p05_0p1) * (714.0 - Q.sum_pt)
    if Q.z_dr_0p05_0p1 > 0.745 and Q.zdr_5 > 0.0154:
        z += -8301.076 * (Q.z_dr_0p05_0p1 - 0.745) * (Q.zdr_5 - 0.0154)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.5205224
    if Q.e2 < 0.0409:
        z += -177.4962 * Q.e2 + 7.259595
    if Q.sum_zz_dr2 < 0.00818:
        z += 1156.929 * Q.sum_zz_dr2 - 9.463679
    if Q.sum_z_dr < 0.102:
        z += 86.27654 * Q.sum_z_dr - 8.800207
    if Q.sum_z_dr2 < 0.00739:
        z += 729.0869 * Q.sum_z_dr2 - 5.387952
    if Q.sum_z_dr2_top2 < 0.000513:
        z += 17946.19 * Q.sum_z_dr2_top2 - 9.206394
    if Q.mass_over_sum_pt_sq < 0.00708:
        z += -483.9117 * Q.mass_over_sum_pt_sq + 3.426095
    if Q.sj2_dr >= 0.159004:
        z += 6.818164 * Q.sj2_dr - 1.084115
    if Q.sj3_pair_mass_max >= 81.1:
        z += -0.009778342 * Q.sj3_pair_mass_max + 0.7930236
    if Q.lam1_plus_lam2 < 0.0133:
        z += -775.7744 * Q.lam1_plus_lam2 + 10.3178
    if Q.N2 < 0.229 and Q.n_dr_0p1_0p2 < 4.0:
        z += 0.408245 * (0.229 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.218 and Q.sum_pt_top5 > 498.0:
        z += 0.01727188 * (0.218 - Q.N2) * (Q.sum_pt_top5 - 498.0)
    if Q.N2 < 0.223 and Q.z_dr_0p05_0p1 < 0.548:
        z += -7.827121 * (0.223 - Q.N2) * (0.548 - Q.z_dr_0p05_0p1)
    if Q.sum_z_dr2 < 0.00752 and Q.D2 < 0.731:
        z += -2177.724 * (0.00752 - Q.sum_z_dr2) * (0.731 - Q.D2)
    if Q.mass_over_sum_pt < 0.133 and Q.D2 < 0.711:
        z += 30.02082 * (0.133 - Q.mass_over_sum_pt) * (0.711 - Q.D2)
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
