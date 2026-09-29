"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 89.8% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
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
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
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
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
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
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
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
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
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
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -0.464
    if Q.C2_b2 < 0.00151:
        z += 930.0 * Q.C2_b2 - 1.4043
    if 0.00783 <= Q.centroid_offset < 0.0499:
        z += -50.7 * Q.centroid_offset + 0.396981
    if Q.centroid_offset >= 0.0499:
        z += -397.7 * Q.centroid_offset + 17.71228
    if Q.e2 < 0.0243:
        z += -153.5 * Q.e2 + 4.63803
    if 0.0243 <= Q.e2 < 0.0354:
        z += -81.8 * Q.e2 + 2.89572
    if Q.girth < 0.0751:
        z += 47.9 * Q.girth - 3.59729
    if Q.girth2 < 0.013:
        z += -497.0 * Q.girth2 + 6.461
    if Q.mass < 29.6:
        z += 0.1709 * Q.mass - 7.44193
    if 29.6 <= Q.mass < 64.7:
        z += 0.0679 * Q.mass - 4.39313
    if Q.planar_flow < 0.159:
        z += -6.63 * Q.planar_flow + 1.05417
    if 0.11 <= Q.sj3_dr_max < 0.182:
        z += 11.9 * Q.sj3_dr_max - 1.309
    if Q.sj3_dr_max >= 0.182:
        z += -15.3 * Q.sj3_dr_max + 3.6414
    if Q.sum_pt >= 906.0:
        z += -0.0134 * Q.sum_pt + 12.1404
    if Q.width < 0.00444:
        z += 1120.0 * Q.width - 4.9728
    if Q.centroid_offset > 0.0203 and Q.z_7 < 0.0609:
        z += -1540.0 * (Q.centroid_offset - 0.0203) * (0.0609 - Q.z_7)
    if Q.girth2 < 0.0133 and Q.D2 < 1.02:
        z += 172.0 * (0.0133 - Q.girth2) * (1.02 - Q.D2)
    if Q.girth2 < 0.0134 and Q.phi_0 > -0.00668:
        z += 818.0 * (0.0134 - Q.girth2) * (Q.phi_0 - -0.00668)
    if Q.lam1 < 0.00649 and Q.D2 < 0.843:
        z += -1580.0 * (0.00649 - Q.lam1) * (0.843 - Q.D2)
    if Q.lam2 < 8.01e-05 and Q.D2_b2 < 0.257:
        z += 55000.0 * (8.01e-05 - Q.lam2) * (0.257 - Q.D2_b2)
    if Q.log_sum_pt > 6.68 and Q.dr_4 < 0.0746:
        z += 49.3 * (Q.log_sum_pt - 6.68) * (0.0746 - Q.dr_4)
    if Q.log_sum_pt > 6.68 and Q.z_7 > 0.0178:
        z += -458.0 * (Q.log_sum_pt - 6.68) * (Q.z_7 - 0.0178)
    if Q.planar_flow < 0.142 and Q.dr_2 < 0.023:
        z += -661.0 * (0.142 - Q.planar_flow) * (0.023 - Q.dr_2)
    if Q.planar_flow < 0.165 and Q.z_7 < 0.0237:
        z += 510.0 * (0.165 - Q.planar_flow) * (0.0237 - Q.z_7)
    if Q.sj3_dr_max < 0.245 and Q.z_7 < 0.058:
        z += -190.0 * (0.245 - Q.sj3_dr_max) * (0.058 - Q.z_7)
    if Q.sum_pt > 915.0 and Q.pt_7 > 35.4:
        z += 0.000467 * (Q.sum_pt - 915.0) * (Q.pt_7 - 35.4)
    if Q.width < 0.00414 and Q.C3 < 0.0364:
        z += 8830.0 * (0.00414 - Q.width) * (0.0364 - Q.C3)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.28
    if Q.girth < 0.0991:
        z += 29.9 * Q.girth - 2.96309
    if Q.girth2 < 0.00871:
        z += 196.0 * Q.girth2 - 1.70716
    if Q.lam1 < 0.00085:
        z += 736.0 * Q.lam1 - 0.6256
    if 6.5 <= Q.log_sum_pt < 6.62:
        z += 10.3 * Q.log_sum_pt - 66.95
    if Q.log_sum_pt >= 6.62:
        z += 15.53 * Q.log_sum_pt - 101.5726
    if Q.mass_over_sum_pt_sq < 0.00591:
        z += -323.0 * Q.mass_over_sum_pt_sq + 1.90893
    if Q.pt_7 >= 53.5:
        z += -0.113 * Q.pt_7 + 6.0455
    if 364.0 <= Q.sum_pt_top5 < 528.0:
        z += 0.0053 * Q.sum_pt_top5 - 1.9292
    if Q.sum_pt_top5 >= 528.0:
        z += -0.0076 * Q.sum_pt_top5 + 4.882
    if Q.z_7 < 0.0586:
        z += 35.5 * Q.z_7 - 2.19745
    if 0.0586 <= Q.z_7 < 0.0619:
        z += 57.1 * Q.z_7 - 3.46321
    if Q.z_7 >= 0.0619:
        z += 21.6 * Q.z_7 - 1.26576
    if Q.girth2 < 0.00939 and Q.centroid_offset > 0.0216:
        z += 6070.0 * (0.00939 - Q.girth2) * (Q.centroid_offset - 0.0216)
    if Q.lam1 < 0.00871 and Q.n_pt_above_50 > 5.25:
        z += -46.4 * (0.00871 - Q.lam1) * (Q.n_pt_above_50 - 5.25)
    if Q.lam1 < 0.00962 and Q.planar_flow < 0.15:
        z += -596.0 * (0.00962 - Q.lam1) * (0.15 - Q.planar_flow)
    if Q.log_sum_pt > 6.48 and Q.D2 < 1.4:
        z += 3.11 * (Q.log_sum_pt - 6.48) * (1.4 - Q.D2)
    if Q.log_sum_pt > 6.37 and Q.centroid_offset > 0.0126:
        z += 82.0 * (Q.log_sum_pt - 6.37) * (Q.centroid_offset - 0.0126)
    if Q.pt_7 > 33.4 and Q.pt_6 < 53.4:
        z += -0.00445 * (Q.pt_7 - 33.4) * (53.4 - Q.pt_6)
    if Q.pt_7 > 33.5 and Q.sj2_dr > 0.143:
        z += 0.477 * (Q.pt_7 - 33.5) * (Q.sj2_dr - 0.143)
    if Q.sj3_dr_max > 0.104 and Q.sj3_pair_mass_min > 2.6:
        z += -0.0846 * (Q.sj3_dr_max - 0.104) * (Q.sj3_pair_mass_min - 2.6)
    if Q.z_7 < 0.0607 and Q.D2 < 1.52:
        z += -36.3 * (0.0607 - Q.z_7) * (1.52 - Q.D2)
    return max(0.0, z)


def neuron_2(Q):
    z = 4.22
    if Q.LHA >= 0.131:
        z += -12.2 * Q.LHA + 1.5982
    if Q.girth < 0.00765:
        z += 144.0 * Q.girth - 1.1016
    if Q.lam2 < 0.000201:
        z += -1810.0 * Q.lam2 + 0.36381
    if Q.log_sum_pt < 6.48:
        z += -4.06 * Q.log_sum_pt + 26.3088
    if Q.m012 >= 42.1:
        z += 0.0269 * Q.m012 - 1.13249
    if Q.mass < 35.5:
        z += 0.0236 * Q.mass - 0.8378
    if Q.mass_over_sum_pt < 0.105:
        z += -24.2 * Q.mass_over_sum_pt + 2.541
    if Q.max_dr < 0.236:
        z += 2.21 * Q.max_dr - 0.52156
    if Q.planar_flow < 0.502:
        z += 0.933 * Q.planar_flow - 0.468366
    if Q.pt_7 < 43.7:
        z += 0.0939 * Q.pt_7 - 4.92036
    if 43.7 <= Q.pt_7 < 52.4:
        z += 0.1915 * Q.pt_7 - 9.18548
    if Q.pt_7 >= 52.4:
        z += 0.0976 * Q.pt_7 - 4.26512
    if Q.sum_pt < 807.0:
        z += -0.00878 * Q.sum_pt + 7.08546
    if Q.sum_pt >= 923.0:
        z += 0.00368 * Q.sum_pt - 3.39664
    if Q.z_7 >= 0.0366:
        z += -43.8 * Q.z_7 + 1.60308
    if Q.lam1 < 0.00609 and Q.max_dr > 0.0769:
        z += -2060.0 * (0.00609 - Q.lam1) * (Q.max_dr - 0.0769)
    if Q.lam1 < 0.00496 and Q.pt_6 > 17.3:
        z += 3.28 * (0.00496 - Q.lam1) * (Q.pt_6 - 17.3)
    if Q.log_sum_pt > 6.85 and Q.D2_b2 < 1.31:
        z += -9.83 * (Q.log_sum_pt - 6.85) * (1.31 - Q.D2_b2)
    if Q.log_sum_pt < 6.33 and Q.D2_b2 < 1.14:
        z += 1.45 * (6.33 - Q.log_sum_pt) * (1.14 - Q.D2_b2)
    if Q.log_sum_pt > 6.95 and Q.pt_6 > 40.0:
        z += -0.173 * (Q.log_sum_pt - 6.95) * (Q.pt_6 - 40.0)
    if Q.sj3_pair_mass_max < 59.4 and Q.centroid_offset > 0.0112:
        z += -0.73 * (59.4 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.0112)
    if Q.sum_pt > 538.0 and Q.lam2 < 0.000194:
        z += -12.6 * (Q.sum_pt - 538.0) * (0.000194 - Q.lam2)
    if Q.sum_pt_top5 > 745.0 and Q.D2_b2 < 1.1:
        z += 0.00862 * (Q.sum_pt_top5 - 745.0) * (1.1 - Q.D2_b2)
    if Q.z_7 > 0.0548 and Q.sj3_dr_min < 0.0684:
        z += -319.0 * (Q.z_7 - 0.0548) * (0.0684 - Q.sj3_dr_min)
    return max(0.0, z)


def neuron_3(Q):
    z = -4.48
    if Q.centroid_offset >= 0.00928:
        z += 28.4 * Q.centroid_offset - 0.263552
    if Q.e2 >= 0.046:
        z += 98.5 * Q.e2 - 4.531
    if 0.0409 <= Q.girth < 0.103:
        z += 171.0 * Q.girth - 6.9939
    if Q.girth >= 0.103:
        z += 29.0 * Q.girth + 7.6321
    if 0.00879 <= Q.girth2 < 0.0187:
        z += -1380.0 * Q.girth2 + 12.1302
    if Q.girth2 >= 0.0187:
        z += -902.0 * Q.girth2 + 3.1916
    if Q.lam1 >= 0.00828:
        z += 566.0 * Q.lam1 - 4.68648
    if Q.mass >= 65.1:
        z += -0.0994 * Q.mass + 6.47094
    if Q.mass_over_sum_pt_sq < 0.0118:
        z += 288.0 * Q.mass_over_sum_pt_sq - 3.3984
    if Q.n_dr_0_0p05 < 1.04:
        z += -0.643 * Q.n_dr_0_0p05 + 0.66872
    if 63.8 <= Q.sd_mass < 86.0:
        z += 0.0673 * Q.sd_mass - 4.29374
    if Q.sd_mass >= 86.0:
        z += -0.0136 * Q.sd_mass + 2.66366
    if 0.188 <= Q.sj2_dr < 0.269:
        z += 47.1 * Q.sj2_dr - 8.8548
    if Q.sj2_dr >= 0.269:
        z += -11.6 * Q.sj2_dr + 6.9355
    if 0.196 <= Q.sj3_dr_max < 0.297:
        z += 36.7 * Q.sj3_dr_max - 7.1932
    if Q.sj3_dr_max >= 0.297:
        z += 17.2 * Q.sj3_dr_max - 1.4017
    if 0.0469 <= Q.tau1 < 0.114:
        z += -65.8 * Q.tau1 + 3.08602
    if Q.tau1 >= 0.114:
        z += 37.2 * Q.tau1 - 8.65598
    if Q.z_dr_0_0p05 < 0.836:
        z += 0.898 * Q.z_dr_0_0p05 - 0.750728
    if Q.girth > 0.0307 and Q.log_sum_pt > 6.11:
        z += 63.1 * (Q.girth - 0.0307) * (Q.log_sum_pt - 6.11)
    if Q.lam2 > 0.000776 and Q.pt_6 < 64.0:
        z += 16.7 * (Q.lam2 - 0.000776) * (64.0 - Q.pt_6)
    if Q.sj2_dr > 0.214 and Q.dr_3 < 0.0549:
        z += -203.0 * (Q.sj2_dr - 0.214) * (0.0549 - Q.dr_3)
    if Q.sj2_dr > 0.206 and Q.sj2_mass1 > -0.0277:
        z += -0.302 * (Q.sj2_dr - 0.206) * (Q.sj2_mass1 - -0.0277)
    if Q.sj2_dr > 0.199 and Q.z_dr_0p05_0p1 < 1.02:
        z += -33.0 * (Q.sj2_dr - 0.199) * (1.02 - Q.z_dr_0p05_0p1)
    if Q.sj2_dr > 0.268 and Q.z_dr_0p05_0p1 < 1.03:
        z += 60.9 * (Q.sj2_dr - 0.268) * (1.03 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.0436
    if Q.C2 < 0.0651:
        z += -76.8 * Q.C2 + 4.99968
    if Q.LHA < 0.351:
        z += -13.2 * Q.LHA + 4.6332
    if Q.N2 < 0.224:
        z += -53.9 * Q.N2 + 12.0736
    if Q.e2_sq < 0.0116:
        z += -648.0 * Q.e2_sq + 7.5168
    if Q.girth2 < 0.00919:
        z += 969.0 * Q.girth2 - 8.90511
    if Q.girth2_top2 < 0.00779:
        z += -220.0 * Q.girth2_top2 + 1.7138
    if Q.lam2 < 0.000876:
        z += 2750.0 * Q.lam2 - 2.409
    if Q.mass >= 45.1:
        z += -0.0739 * Q.mass + 3.33289
    if Q.mass_over_sum_pt >= 0.0903:
        z += -64.2 * Q.mass_over_sum_pt + 5.79726
    if Q.sj3_dr_max < 0.237:
        z += 21.7 * Q.sj3_dr_max - 5.1429
    if Q.sum_pt < 753.0:
        z += 0.00773 * Q.sum_pt - 5.82069
    if Q.width >= 0.00339:
        z += 313.0 * Q.width - 1.06107
    if Q.N2 < 0.219 and Q.e2_sq > 0.0116:
        z += -1740.0 * (0.219 - Q.N2) * (Q.e2_sq - 0.0116)
    if Q.N2 < 0.224 and Q.eccentricity > 0.727:
        z += -136.0 * (0.224 - Q.N2) * (Q.eccentricity - 0.727)
    if Q.N2 < 0.219 and Q.mass < 63.2:
        z += -0.848 * (0.219 - Q.N2) * (63.2 - Q.mass)
    if Q.N2 < 0.242 and Q.pt_7 < 52.9:
        z += -0.332 * (0.242 - Q.N2) * (52.9 - Q.pt_7)
    if Q.girth2_top2 < 0.0076 and Q.C2_b2 > 0.000612:
        z += -19900.0 * (0.0076 - Q.girth2_top2) * (Q.C2_b2 - 0.000612)
    if Q.lam2 < 0.000453 and Q.dr01 < 0.173:
        z += -17300.0 * (0.000453 - Q.lam2) * (0.173 - Q.dr01)
    if Q.max_dr > 0.164 and Q.C2_b2 < 0.000596:
        z += -45400.0 * (Q.max_dr - 0.164) * (0.000596 - Q.C2_b2)
    if Q.sd_mass > 37.0 and Q.sd_zg < 0.276:
        z += -0.321 * (Q.sd_mass - 37.0) * (0.276 - Q.sd_zg)
    if Q.sd_mass > 42.1 and Q.sj3_pair_mass_min < 26.6:
        z += 0.00274 * (Q.sd_mass - 42.1) * (26.6 - Q.sj3_pair_mass_min)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.444
    if Q.girth < 0.00764:
        z += -240.0 * Q.girth + 1.8336
    if Q.log_sum_pt >= 6.83:
        z += -22.2 * Q.log_sum_pt + 151.626
    if Q.pt_7 < 35.4:
        z += 0.132 * Q.pt_7 - 4.6728
    if Q.width < 0.0039:
        z += -569.0 * Q.width + 2.2191
    if Q.z_7 < 0.0286:
        z += -262.9 * Q.z_7 + 11.36367
    if 0.0286 <= Q.z_7 < 0.0473:
        z += -130.9 * Q.z_7 + 7.58847
    if 0.0473 <= Q.z_7 < 0.0702:
        z += -61.0 * Q.z_7 + 4.2822
    if Q.zdr_0 < 0.021:
        z += 45.2 * Q.zdr_0 - 0.9492
    if Q.LHA < 0.215 and Q.lam1 < 0.00121:
        z += -11700.0 * (0.215 - Q.LHA) * (0.00121 - Q.lam1)
    if Q.LHA < 0.215 and Q.log_sum_pt < 6.78:
        z += -67.1 * (0.215 - Q.LHA) * (6.78 - Q.log_sum_pt)
    if Q.girth2 < 0.00167 and Q.centroid_offset < 0.024:
        z += 135000.0 * (0.00167 - Q.girth2) * (0.024 - Q.centroid_offset)
    if Q.log_sum_pt > 6.73 and Q.dr_2 < 0.0147:
        z += 981.0 * (Q.log_sum_pt - 6.73) * (0.0147 - Q.dr_2)
    if Q.log_sum_pt > 6.89 and Q.dr_2 < 0.0244:
        z += -924.0 * (Q.log_sum_pt - 6.89) * (0.0244 - Q.dr_2)
    if Q.log_sum_pt > 6.7 and Q.mean_eta2 < 8.48e-05:
        z += 84600.0 * (Q.log_sum_pt - 6.7) * (8.48e-05 - Q.mean_eta2)
    if Q.log_sum_pt > 6.7 and Q.mean_phi > 0.00334:
        z += -902.0 * (Q.log_sum_pt - 6.7) * (Q.mean_phi - 0.00334)
    if Q.sum_pt_top5 > 530.0 and Q.pt_6 < 43.2:
        z += 0.000174 * (Q.sum_pt_top5 - 530.0) * (43.2 - Q.pt_6)
    if Q.sum_pt_top5 > 438.0 and Q.sj2_dr > 0.176:
        z += 0.019 * (Q.sum_pt_top5 - 438.0) * (Q.sj2_dr - 0.176)
    if Q.z_7 < 0.0687 and Q.centroid_offset < 0.031:
        z += -1450.0 * (0.0687 - Q.z_7) * (0.031 - Q.centroid_offset)
    if Q.z_7 < 0.0708 and Q.mass_top3 < 45.4:
        z += 0.354 * (0.0708 - Q.z_7) * (45.4 - Q.mass_top3)
    return max(0.0, z)


def neuron_6(Q):
    z = 2.65
    if 0.00762 <= Q.centroid_offset < 0.0193:
        z += 106.0 * Q.centroid_offset - 0.80772
    if Q.centroid_offset >= 0.0193:
        z += 4.0 * Q.centroid_offset + 1.16088
    if Q.e2_sq < 0.00301:
        z += -547.0 * Q.e2_sq + 1.64647
    if Q.girth2 < 0.00867:
        z += 1020.0 * Q.girth2 - 8.8434
    if Q.lam1 < 0.0118:
        z += -298.0 * Q.lam1 + 3.5164
    if Q.lam2 < 0.00109:
        z += 1090.0 * Q.lam2 - 1.1881
    if Q.mass < 22.5:
        z += -0.0803 * Q.mass + 1.80675
    if Q.max_dr < 0.144:
        z += 34.4 * Q.max_dr - 4.9536
    if Q.pt_6 < 19.1:
        z += -0.513 * Q.pt_6 + 9.7983
    if Q.sj3_dr_max < 0.181:
        z += -20.8 * Q.sj3_dr_max + 2.7184
    if 0.181 <= Q.sj3_dr_max < 0.191:
        z += 9.6 * Q.sj3_dr_max - 2.784
    if 0.191 <= Q.sj3_dr_max < 0.29:
        z += 17.74 * Q.sj3_dr_max - 4.33874
    if Q.sj3_dr_max >= 0.29:
        z += 8.14 * Q.sj3_dr_max - 1.55474
    if Q.sj3_dr_min >= 0.0225:
        z += -17.8 * Q.sj3_dr_min + 0.4005
    if Q.sj3_pair_mass_min >= 4.35:
        z += -0.0953 * Q.sj3_pair_mass_min + 0.414555
    if Q.sum_pt >= 975.0:
        z += 0.0162 * Q.sum_pt - 15.795
    if Q.z_6 < 0.0221:
        z += 437.0 * Q.z_6 - 9.6577
    if Q.centroid_offset > 0.0213 and Q.mean_phi2 < 0.01:
        z += 5830.0 * (Q.centroid_offset - 0.0213) * (0.01 - Q.mean_phi2)
    if Q.centroid_offset > 0.0103 and Q.psi_0p1 > 0.335:
        z += 65.4 * (Q.centroid_offset - 0.0103) * (Q.psi_0p1 - 0.335)
    if Q.centroid_offset > 0.017 and Q.pt_4 > 66.3:
        z += 1.84 * (Q.centroid_offset - 0.017) * (Q.pt_4 - 66.3)
    if Q.lam1 < 0.0118 and Q.planar_flow < 0.254:
        z += -1140.0 * (0.0118 - Q.lam1) * (0.254 - Q.planar_flow)
    if Q.lam2 < 0.0032 and Q.n_dr_0_0p05 < 3.0:
        z += 57.4 * (0.0032 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.mass < 64.5 and Q.z_dr_0p2_0p4 < 0.0534:
        z += -0.707 * (64.5 - Q.mass) * (0.0534 - Q.z_dr_0p2_0p4)
    if Q.mean_eta > 0.0349 and Q.M3 > 0.0492:
        z += -1210.0 * (Q.mean_eta - 0.0349) * (Q.M3 - 0.0492)
    if Q.pt_6 < 42.5 and Q.log_sum_pt < 6.76:
        z += 0.589 * (42.5 - Q.pt_6) * (6.76 - Q.log_sum_pt)
    if Q.pt_6 < 41.5 and Q.z_7 > 0.0235:
        z += -6.53 * (41.5 - Q.pt_6) * (Q.z_7 - 0.0235)
    if Q.sj2_dr < 0.171 and Q.eccentricity > 0.927:
        z += 176.0 * (0.171 - Q.sj2_dr) * (Q.eccentricity - 0.927)
    if Q.sum_pt > 953.0 and Q.mean_phi2 < 0.00311:
        z += -3.68 * (Q.sum_pt - 953.0) * (0.00311 - Q.mean_phi2)
    if Q.sum_pt < 589.0 and Q.n_dr_0p2_0p4 < 1.97:
        z += 0.00255 * (589.0 - Q.sum_pt) * (1.97 - Q.n_dr_0p2_0p4)
    if Q.sum_pt < 625.0 and Q.pt_5 < 24.6:
        z += 0.00863 * (625.0 - Q.sum_pt) * (24.6 - Q.pt_5)
    if Q.tau1 < 0.111 and Q.mean_phi2 < 0.0157:
        z += 1560.0 * (0.111 - Q.tau1) * (0.0157 - Q.mean_phi2)
    return max(0.0, z)


def neuron_7(Q):
    z = 11.6
    if Q.D2_b2 < 0.0586:
        z += 10.8 * Q.D2_b2 - 0.63288
    if Q.centroid_offset >= 0.0384:
        z += -118.0 * Q.centroid_offset + 4.5312
    if Q.girth < 0.0884:
        z += 61.0 * Q.girth - 5.3924
    if Q.girth2 < 0.00104:
        z += 1260.0 * Q.girth2 - 1.3104
    if 0.00439 <= Q.girth2 < 0.0134:
        z += -238.0 * Q.girth2 + 1.04482
    if Q.girth2 >= 0.0134:
        z += 379.0 * Q.girth2 - 7.22298
    if Q.girth2_top2 < 0.00109:
        z += -603.0 * Q.girth2_top2 + 0.65727
    if Q.lam2 < 0.000271:
        z += 1780.0 * Q.lam2 - 0.48238
    if 36.6 <= Q.mass < 76.8:
        z += 0.0576 * Q.mass - 2.10816
    if Q.mass >= 76.8:
        z += -0.1224 * Q.mass + 11.71584
    if 0.0108 <= Q.mass_over_sum_pt < 0.0803:
        z += -124.0 * Q.mass_over_sum_pt + 1.3392
    if 0.0803 <= Q.mass_over_sum_pt < 0.0909:
        z += 34.0 * Q.mass_over_sum_pt - 11.3482
    if 0.0909 <= Q.mass_over_sum_pt < 0.108:
        z += -239.0 * Q.mass_over_sum_pt + 13.4675
    if Q.mass_over_sum_pt >= 0.108:
        z += -733.0 * Q.mass_over_sum_pt + 66.8195
    if Q.mass_top5 >= 53.0:
        z += 0.0721 * Q.mass_top5 - 3.8213
    if Q.max_dr < 0.109:
        z += 8.7 * Q.max_dr - 0.2928
    if 0.109 <= Q.max_dr < 0.178:
        z += -9.5 * Q.max_dr + 1.691
    if Q.pt_7 < 29.4:
        z += 0.0511 * Q.pt_7 - 1.50234
    if Q.sd_rg >= 0.283:
        z += 33.0 * Q.sd_rg - 9.339
    if Q.sj2_dr < 0.156:
        z += -6.7 * Q.sj2_dr + 0.3818
    if 0.156 <= Q.sj2_dr < 0.187:
        z += 21.4 * Q.sj2_dr - 4.0018
    if 0.139 <= Q.sj3_dr_max < 0.261:
        z += 23.0 * Q.sj3_dr_max - 3.197
    if Q.sj3_dr_max >= 0.261:
        z += 3.6 * Q.sj3_dr_max + 1.8664
    if Q.tau21_b2 < 0.0407:
        z += 24.2 * Q.tau21_b2 - 0.98494
    if Q.width < 0.00664:
        z += 1120.0 * Q.width - 7.4368
    if Q.z_7 >= 0.0342:
        z += 15.2 * Q.z_7 - 0.51984
    if Q.z_dr_0p1_0p2 < 0.139:
        z += -2.49 * Q.z_dr_0p1_0p2 + 0.34611
    if Q.centroid_offset < 0.02 and Q.C2_b2 < 0.00426:
        z += -17200.0 * (0.02 - Q.centroid_offset) * (0.00426 - Q.C2_b2)
    if Q.centroid_offset > 0.0258 and Q.n_pt_above_50 > 3.71:
        z += -10.0 * (Q.centroid_offset - 0.0258) * (Q.n_pt_above_50 - 3.71)
    if Q.centroid_offset < 0.0216 and Q.tau21_b2 < 0.0279:
        z += 2880.0 * (0.0216 - Q.centroid_offset) * (0.0279 - Q.tau21_b2)
    if Q.girth < 0.0893 and Q.C2_b2 < 0.00429:
        z += 4120.0 * (0.0893 - Q.girth) * (0.00429 - Q.C2_b2)
    if Q.girth2 > 0.0145 and Q.eccentricity > 0.925:
        z += -2520.0 * (Q.girth2 - 0.0145) * (Q.eccentricity - 0.925)
    if Q.girth2 > 0.00447 and Q.planar_flow < 0.197:
        z += 1040.0 * (Q.girth2 - 0.00447) * (0.197 - Q.planar_flow)
    if Q.girth2 > 0.0131 and Q.pt_6 < 41.3:
        z += 127.0 * (Q.girth2 - 0.0131) * (41.3 - Q.pt_6)
    if Q.girth2_top2 < 0.00107 and Q.tau21_b2 < 0.0257:
        z += -90500.0 * (0.00107 - Q.girth2_top2) * (0.0257 - Q.tau21_b2)
    if Q.lam2 < 0.0003 and Q.tau21_b2 < 0.0408:
        z += 125000.0 * (0.0003 - Q.lam2) * (0.0408 - Q.tau21_b2)
    if Q.mass > 76.2 and Q.zdr_6 > 0.00612:
        z += -9.31 * (Q.mass - 76.2) * (Q.zdr_6 - 0.00612)
    if Q.mass_over_sum_pt > 0.0911 and Q.pt_6 < 40.8:
        z += 17.2 * (Q.mass_over_sum_pt - 0.0911) * (40.8 - Q.pt_6)
    if Q.planar_flow < 0.216 and Q.pt_6 < 35.6:
        z += -0.224 * (0.216 - Q.planar_flow) * (35.6 - Q.pt_6)
    if Q.sj3_dr_max > 0.132 and Q.pt_6 < 20.0:
        z += -0.947 * (Q.sj3_dr_max - 0.132) * (20.0 - Q.pt_6)
    if Q.width > 0.00864 and Q.pt_6 < 40.8:
        z += -111.0 * (Q.width - 0.00864) * (40.8 - Q.pt_6)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.632
    if Q.girth < 0.0595:
        z += 218.0 * Q.girth - 12.971
    if Q.girth2 < 0.00491:
        z += -1110.0 * Q.girth2 + 5.4501
    if Q.log_sum_pt >= 6.7:
        z += -5.72 * Q.log_sum_pt + 38.324
    if Q.width < 0.00355:
        z += -822.0 * Q.width + 2.9181
    if Q.girth < 0.0585 and Q.width > 0.000656:
        z += 42000.0 * (0.0585 - Q.girth) * (Q.width - 0.000656)
    if Q.girth2 < 0.00535 and Q.centroid_offset > 0.00661:
        z += -24400.0 * (0.00535 - Q.girth2) * (Q.centroid_offset - 0.00661)
    if Q.girth2 < 0.00662 and Q.centroid_offset < 0.0259:
        z += 21200.0 * (0.00662 - Q.girth2) * (0.0259 - Q.centroid_offset)
    if Q.lam2 < 0.000136 and Q.D2_b2 < 5.27:
        z += 1120.0 * (0.000136 - Q.lam2) * (5.27 - Q.D2_b2)
    if Q.mass < 31.3 and Q.centroid_offset < 0.0291:
        z += -6.02 * (31.3 - Q.mass) * (0.0291 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.142 and Q.centroid_offset > 0.0059:
        z += -720.0 * (0.142 - Q.sj3_dr_max) * (Q.centroid_offset - 0.0059)
    if Q.sum_pt_top5 > 679.0 and Q.n_pt_above_10 < 8.0:
        z += -0.00955 * (Q.sum_pt_top5 - 679.0) * (8.0 - Q.n_pt_above_10)
    if Q.tau1 < 0.0617 and Q.width < 0.00304:
        z += 30000.0 * (0.0617 - Q.tau1) * (0.00304 - Q.width)
    return max(0.0, z)


def neuron_9(Q):
    z = -5.15
    if Q.D2 >= 2.16:
        z += -0.17 * Q.D2 + 0.3672
    if Q.M3 < 0.0796:
        z += -12.7 * Q.M3 + 1.01092
    if Q.centroid_offset < 0.0185:
        z += -182.0 * Q.centroid_offset + 3.367
    if Q.e2 >= 0.0456:
        z += 43.6 * Q.e2 - 1.98816
    if Q.e2_sq < 0.00327:
        z += 1790.0 * Q.e2_sq - 5.8533
    if Q.girth2 < 0.00365:
        z += -3745.0 * Q.girth2 + 15.71065
    if 0.00365 <= Q.girth2 < 0.00601:
        z += -865.0 * Q.girth2 + 5.19865
    if Q.girth2 >= 0.0134:
        z += 279.0 * Q.girth2 - 3.7386
    if Q.lam2 < 0.000315:
        z += -5410.0 * Q.lam2 + 1.70415
    if Q.mass < 30.1:
        z += 0.1591 * Q.mass - 6.14671
    if 30.1 <= Q.mass < 52.0:
        z += 0.062 * Q.mass - 3.224
    if Q.max_dr < 0.107:
        z += -18.6 * Q.max_dr + 1.9902
    if Q.sj3_dr_max < 0.141:
        z += 33.1 * Q.sj3_dr_max - 3.0465
    if 0.141 <= Q.sj3_dr_max < 0.215:
        z += -21.9 * Q.sj3_dr_max + 4.7085
    if Q.sum_pt < 986.0:
        z += -0.0077 * Q.sum_pt + 7.5922
    if Q.width < 0.000176:
        z += -7810.0 * Q.width + 1.37456
    if Q.zdr_0 < 0.00631:
        z += 161.0 * Q.zdr_0 - 1.01591
    if Q.centroid_offset < 0.017 and Q.n_for_90pct > 5.0:
        z += -33.6 * (0.017 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.centroid_offset < 0.019 and Q.pt_1 < 164.0:
        z += -2.01 * (0.019 - Q.centroid_offset) * (164.0 - Q.pt_1)
    if Q.centroid_offset < 0.0191 and Q.z_2nd < 0.205:
        z += 1090.0 * (0.0191 - Q.centroid_offset) * (0.205 - Q.z_2nd)
    if Q.girth < 0.0528 and Q.tau21_b2 < 0.0275:
        z += 3990.0 * (0.0528 - Q.girth) * (0.0275 - Q.tau21_b2)
    if Q.girth < 0.0549 and Q.z_6 > 0.0338:
        z += -1380.0 * (0.0549 - Q.girth) * (Q.z_6 - 0.0338)
    if Q.girth2 < 0.00578 and Q.mean_phi < 0.00437:
        z += -9500.0 * (0.00578 - Q.girth2) * (0.00437 - Q.mean_phi)
    if Q.lam1 > 0.00485 and Q.eccentricity > 0.638:
        z += -859.0 * (Q.lam1 - 0.00485) * (Q.eccentricity - 0.638)
    if Q.log_sum_pt > 6.89 and Q.D2_b2 < 0.746:
        z += -27.2 * (Q.log_sum_pt - 6.89) * (0.746 - Q.D2_b2)
    if Q.mass < 54.9 and Q.centroid_offset < 0.0267:
        z += 4.38 * (54.9 - Q.mass) * (0.0267 - Q.centroid_offset)
    if Q.mass < 55.0 and Q.log_sum_pt < 6.84:
        z += 0.129 * (55.0 - Q.mass) * (6.84 - Q.log_sum_pt)
    if Q.sj3_dr_max < 0.15 and Q.pt_6 > 32.9:
        z += 0.581 * (0.15 - Q.sj3_dr_max) * (Q.pt_6 - 32.9)
    if Q.tau1 < 0.046 and Q.mean_phi < -0.0119:
        z += 3360.0 * (0.046 - Q.tau1) * (-0.0119 - Q.mean_phi)
    if Q.tau1 < 0.0457 and Q.tau21_b2 < 0.0286:
        z += -4610.0 * (0.0457 - Q.tau1) * (0.0286 - Q.tau21_b2)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.59
    if Q.C2_b2 >= 0.00763:
        z += 36.9 * Q.C2_b2 - 0.281547
    if Q.LHA >= 0.305:
        z += -29.3 * Q.LHA + 8.9365
    if Q.e3 >= 5.22e-05:
        z += -2620.0 * Q.e3 + 0.136764
    if 0.00749 <= Q.girth2 < 0.0258:
        z += 274.0 * Q.girth2 - 2.05226
    if Q.girth2 >= 0.0258:
        z += 183.2 * Q.girth2 + 0.29038
    if Q.girth2_top3 < 0.00215:
        z += -379.0 * Q.girth2_top3 + 0.81485
    if Q.lam1 < 0.00144:
        z += 1729.0 * Q.lam1 - 4.9815
    if 0.00144 <= Q.lam1 < 0.00404:
        z += 689.0 * Q.lam1 - 3.4839
    if 0.00404 <= Q.lam1 < 0.00598:
        z += 361.0 * Q.lam1 - 2.15878
    if Q.lam2 >= 0.000278:
        z += 1290.0 * Q.lam2 - 0.35862
    if Q.mass < 79.4:
        z += -0.0276 * Q.mass + 2.19144
    if Q.mass_top5 >= 51.0:
        z += -0.0122 * Q.mass_top5 + 0.6222
    if Q.pt_7 < 43.9:
        z += 0.0151 * Q.pt_7 - 0.66289
    if Q.sj3_dr_max >= 0.184:
        z += -4.57 * Q.sj3_dr_max + 0.84088
    if Q.sj3_pair_mass_min >= 15.8:
        z += 0.123 * Q.sj3_pair_mass_min - 1.9434
    if Q.sum_pt >= 988.0:
        z += -0.00569 * Q.sum_pt + 5.62172
    if Q.tau1 >= 0.057:
        z += 16.3 * Q.tau1 - 0.9291
    if Q.lam1 > 0.00707 and Q.D2_b2 < 0.36:
        z += -241.0 * (Q.lam1 - 0.00707) * (0.36 - Q.D2_b2)
    if Q.lam2 > 0.000257 and Q.planar_flow > 0.055:
        z += -702.0 * (Q.lam2 - 0.000257) * (Q.planar_flow - 0.055)
    if Q.pt_7 < 45.6 and Q.D2 < 0.985:
        z += 0.0613 * (45.6 - Q.pt_7) * (0.985 - Q.D2)
    if Q.pt_7 < 48.0 and Q.log_sum_pt < 6.56:
        z += -0.0904 * (48.0 - Q.pt_7) * (6.56 - Q.log_sum_pt)
    if Q.sj3_pair_mass_min > 6.68 and Q.sj3_pairmin_over_m > 0.245:
        z += -0.191 * (Q.sj3_pair_mass_min - 6.68) * (Q.sj3_pairmin_over_m - 0.245)
    if Q.zdr_0 < 0.0205 and Q.centroid_offset > 0.0125:
        z += 1610.0 * (0.0205 - Q.zdr_0) * (Q.centroid_offset - 0.0125)
    if Q.zdr_0 < 0.0242 and Q.z_dr_0p05_0p1 < 0.248:
        z += 115.0 * (0.0242 - Q.zdr_0) * (0.248 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.44
    if Q.centroid_offset < 0.0377:
        z += -65.3 * Q.centroid_offset + 2.46181
    if Q.centroid_offset >= 0.05:
        z += -624.0 * Q.centroid_offset + 31.2
    if Q.e2_sq < 0.00814:
        z += 1160.0 * Q.e2_sq - 9.4424
    if Q.girth < 0.0231:
        z += 142.8 * Q.girth - 6.69394
    if 0.0231 <= Q.girth < 0.0737:
        z += 67.1 * Q.girth - 4.94527
    if Q.girth2 < 0.0134:
        z += -137.0 * Q.girth2 + 1.8358
    if Q.lam1 < 0.00838:
        z += 708.0 * Q.lam1 - 5.93304
    if Q.mass < 15.7:
        z += -0.085 * Q.mass - 0.8575
    if 15.7 <= Q.mass < 70.5:
        z += 0.04 * Q.mass - 2.82
    if Q.n_dr_0p1_0p2 >= 2.92:
        z += -0.351 * Q.n_dr_0p1_0p2 + 1.02492
    if Q.planar_flow < 0.242:
        z += -5.84 * Q.planar_flow + 1.41328
    if Q.pt_6 < 25.4:
        z += 0.0691 * Q.pt_6 - 1.75514
    if Q.pt_7 >= 29.6:
        z += -0.036 * Q.pt_7 + 1.0656
    if Q.sj3_dr_max < 0.167:
        z += 12.6 * Q.sj3_dr_max - 0.7452
    if 0.167 <= Q.sj3_dr_max < 0.257:
        z += -15.1 * Q.sj3_dr_max + 3.8807
    if Q.sum_pt >= 971.0:
        z += -0.00349 * Q.sum_pt + 3.38879
    if Q.width < 0.00872:
        z += -2510.0 * Q.width + 21.8872
    if Q.z_7 >= 0.0169:
        z += 35.8 * Q.z_7 - 0.60502
    if Q.z_dr_0p05_0p1 >= 0.826:
        z += 2.49 * Q.z_dr_0p05_0p1 - 2.05674
    if Q.centroid_offset < 0.0138 and Q.D2_b2 < 0.571:
        z += 213.0 * (0.0138 - Q.centroid_offset) * (0.571 - Q.D2_b2)
    if Q.centroid_offset < 0.0384 and Q.absphi_0 < 0.0324:
        z += -328.0 * (0.0384 - Q.centroid_offset) * (0.0324 - Q.absphi_0)
    if Q.centroid_offset < 0.0383 and Q.mean_phi < -0.00159:
        z += -1070.0 * (0.0383 - Q.centroid_offset) * (-0.00159 - Q.mean_phi)
    if Q.centroid_offset < 0.0137 and Q.sj3_pair_mass_min < 17.2:
        z += -8.45 * (0.0137 - Q.centroid_offset) * (17.2 - Q.sj3_pair_mass_min)
    if Q.centroid_offset < 0.0377 and Q.sum_pt < 892.0:
        z += -0.193 * (0.0377 - Q.centroid_offset) * (892.0 - Q.sum_pt)
    if Q.centroid_offset > 0.05 and Q.tau3 > 0.000494:
        z += -86100.0 * (Q.centroid_offset - 0.05) * (Q.tau3 - 0.000494)
    if Q.centroid_offset > 0.0494 and Q.tau32 < 0.619:
        z += 1720.0 * (Q.centroid_offset - 0.0494) * (0.619 - Q.tau32)
    if Q.centroid_offset > 0.0492 and Q.zdr_6 > 0.00714:
        z += -57100.0 * (Q.centroid_offset - 0.0492) * (Q.zdr_6 - 0.00714)
    if Q.girth2 < 0.00431 and Q.sj3_dr13 > 0.162:
        z += 4770.0 * (0.00431 - Q.girth2) * (Q.sj3_dr13 - 0.162)
    if Q.planar_flow < 0.233 and Q.e3 < 9.53e-06:
        z += -446000.0 * (0.233 - Q.planar_flow) * (9.53e-06 - Q.e3)
    if Q.planar_flow < 0.268 and Q.pt_7 < 36.7:
        z += -0.183 * (0.268 - Q.planar_flow) * (36.7 - Q.pt_7)
    if Q.planar_flow < 0.241 and Q.sum_pt < 835.0:
        z += -0.0134 * (0.241 - Q.planar_flow) * (835.0 - Q.sum_pt)
    if Q.sj3_dr_max < 0.264 and Q.phi_0 < -0.023:
        z += 91.8 * (0.264 - Q.sj3_dr_max) * (-0.023 - Q.phi_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.611
    if Q.centroid_offset >= 0.0499:
        z += 22.4 * Q.centroid_offset - 1.11776
    if Q.e2 >= 0.0634:
        z += -32.5 * Q.e2 + 2.0605
    if Q.girth2 >= 0.0188:
        z += 260.0 * Q.girth2 - 4.888
    if Q.girth2_top2 >= 0.014:
        z += -32.5 * Q.girth2_top2 + 0.455
    if Q.mass >= 91.2:
        z += 0.0546 * Q.mass - 4.97952
    if Q.zdr_0 >= 0.0398:
        z += -29.0 * Q.zdr_0 + 1.1542
    if Q.girth2 > 0.0188 and Q.lam2 > 0.000537:
        z += 5030.0 * (Q.girth2 - 0.0188) * (Q.lam2 - 0.000537)
    if Q.girth2 > 0.0188 and Q.pt_7 < 53.4:
        z += -2.27 * (Q.girth2 - 0.0188) * (53.4 - Q.pt_7)
    if Q.mass > 69.6 and Q.centroid_offset > 0.00948:
        z += 0.59 * (Q.mass - 69.6) * (Q.centroid_offset - 0.00948)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.714
    if Q.e2 < 0.079:
        z += 55.3 * Q.e2 - 4.3687
    if Q.e3 < 5.05e-05:
        z += -19700.0 * Q.e3 + 0.99485
    if Q.girth < 0.148:
        z += -58.6 * Q.girth + 8.6728
    if Q.lam1 < 0.0161:
        z += -253.0 * Q.lam1 + 4.0733
    if Q.mass >= 49.1:
        z += -0.0244 * Q.mass + 1.19804
    if Q.pt_7 >= 48.4:
        z += -0.0571 * Q.pt_7 + 2.76364
    if 674.0 <= Q.sum_pt_top5 < 794.0:
        z += -0.00746 * Q.sum_pt_top5 + 5.02804
    if Q.sum_pt_top5 >= 794.0:
        z += 0.00624 * Q.sum_pt_top5 - 5.84976
    if Q.z_5 < 0.0283:
        z += 167.0 * Q.z_5 - 4.7261
    if Q.z_7 < 0.028:
        z += 189.0 * Q.z_7 - 5.292
    if Q.e3 < 6.94e-05 and Q.D3 > 0.0598:
        z += 2100.0 * (6.94e-05 - Q.e3) * (Q.D3 - 0.0598)
    if Q.e3 < 5.38e-05 and Q.centroid_offset < 0.0377:
        z += -1180000.0 * (5.38e-05 - Q.e3) * (0.0377 - Q.centroid_offset)
    if Q.girth < 0.15 and Q.M3 < 0.0746:
        z += 125.0 * (0.15 - Q.girth) * (0.0746 - Q.M3)
    if Q.girth < 0.148 and Q.lam2 < 0.000536:
        z += -8780.0 * (0.148 - Q.girth) * (0.000536 - Q.lam2)
    if Q.girth < 0.148 and Q.log_sum_pt < 6.81:
        z += -75.4 * (0.148 - Q.girth) * (6.81 - Q.log_sum_pt)
    if Q.girth < 0.145 and Q.pt_7 < 38.2:
        z += -0.987 * (0.145 - Q.girth) * (38.2 - Q.pt_7)
    if Q.girth < 0.152 and Q.sj2_mass1 > 30.4:
        z += -1.58 * (0.152 - Q.girth) * (Q.sj2_mass1 - 30.4)
    if Q.girth < 0.142 and Q.tau2 > 0.00886:
        z += 281.0 * (0.142 - Q.girth) * (Q.tau2 - 0.00886)
    if Q.girth < 0.153 and Q.z_7 > 0.0616:
        z += 392.0 * (0.153 - Q.girth) * (Q.z_7 - 0.0616)
    if Q.pt_6 < 31.9 and Q.z_7 < 0.0624:
        z += -2.22 * (31.9 - Q.pt_6) * (0.0624 - Q.z_7)
    if Q.sum_pt > 1000.0 and Q.pt_6 < 70.0:
        z += -0.000466 * (Q.sum_pt - 1000.0) * (70.0 - Q.pt_6)
    if Q.sum_pt_top5 > 667.0 and Q.pt_7 < 38.1:
        z += 0.00108 * (Q.sum_pt_top5 - 667.0) * (38.1 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.689
    if Q.C2_b2 < 0.00403:
        z += 146.0 * Q.C2_b2 - 0.58838
    if 0.0269 <= Q.centroid_offset < 0.0497:
        z += -84.3 * Q.centroid_offset + 2.26767
    if Q.centroid_offset >= 0.0497:
        z += -1063.3 * Q.centroid_offset + 50.92397
    if 0.0167 <= Q.e2 < 0.0499:
        z += -147.0 * Q.e2 + 2.4549
    if Q.e2 >= 0.0499:
        z += 111.0 * Q.e2 - 10.4193
    if Q.e2_sq >= 0.0117:
        z += 1130.0 * Q.e2_sq - 13.221
    if Q.e3 < 7.93e-05:
        z += 23600.0 * Q.e3 - 1.87148
    if 0.0272 <= Q.girth < 0.041:
        z += 84.3 * Q.girth - 2.29296
    if 0.041 <= Q.girth < 0.0802:
        z += 154.4 * Q.girth - 5.16706
    if 0.0802 <= Q.girth < 0.0868:
        z += 247.9 * Q.girth - 12.66576
    if 0.0868 <= Q.girth < 0.102:
        z += 54.9 * Q.girth + 4.08664
    if Q.girth >= 0.102:
        z += -23.9 * Q.girth + 12.12424
    if Q.girth2 >= 0.00856:
        z += -1570.0 * Q.girth2 + 13.4392
    if Q.lam2 < 0.000504:
        z += -1010.0 * Q.lam2 + 0.50904
    if Q.mass >= 68.9:
        z += -0.0427 * Q.mass + 2.94203
    if 0.0804 <= Q.mass_over_sum_pt < 0.0906:
        z += 157.0 * Q.mass_over_sum_pt - 12.6228
    if Q.mass_over_sum_pt >= 0.0906:
        z += 330.0 * Q.mass_over_sum_pt - 28.2966
    if Q.n_dr_0_0p05 < 5.06:
        z += -0.192 * Q.n_dr_0_0p05 + 0.97152
    if Q.psi_0p1 >= 0.974:
        z += -22.5 * Q.psi_0p1 + 21.915
    if Q.sd_mass < 49.6:
        z += -0.0504 * Q.sd_mass + 2.05128
    if 49.6 <= Q.sd_mass < 74.8:
        z += 0.0178 * Q.sd_mass - 1.33144
    if Q.sd_rg < 0.234:
        z += 9.92 * Q.sd_rg - 3.224
    if 0.234 <= Q.sd_rg < 0.325:
        z += -22.88 * Q.sd_rg + 4.4512
    if Q.sd_rg >= 0.325:
        z += -32.8 * Q.sd_rg + 7.6752
    if 0.0606 <= Q.sj2_dr < 0.158:
        z += -6.65 * Q.sj2_dr + 0.40299
    if 0.158 <= Q.sj2_dr < 0.197:
        z += 25.15 * Q.sj2_dr - 4.62141
    if Q.sj2_dr >= 0.197:
        z += 6.25 * Q.sj2_dr - 0.89811
    if Q.sum_pt < 717.0:
        z += 0.00585 * Q.sum_pt - 4.19445
    if 0.00353 <= Q.width < 0.0067:
        z += 339.0 * Q.width - 1.19667
    if 0.0067 <= Q.width < 0.0131:
        z += -1141.0 * Q.width + 8.71933
    if Q.width >= 0.0131:
        z += -2401.0 * Q.width + 25.22533
    if Q.z_dr_0_0p05 >= 0.152:
        z += 1.5 * Q.z_dr_0_0p05 - 0.228
    if Q.z_dr_0p05_0p1 < 0.163:
        z += 2.29 * Q.z_dr_0p05_0p1 - 0.37327
    if Q.z_dr_0p1_0p2 < 0.0677:
        z += -4.72 * Q.z_dr_0p1_0p2 + 0.319544
    if Q.centroid_offset > 0.0499 and Q.C2_b2 < 0.000872:
        z += 1160000.0 * (Q.centroid_offset - 0.0499) * (0.000872 - Q.C2_b2)
    if Q.centroid_offset > 0.0267 and Q.pt_1 < 152.0:
        z += 0.547 * (Q.centroid_offset - 0.0267) * (152.0 - Q.pt_1)
    if Q.e3 < 5.56e-05 and Q.sj3_dr23 > 0.162:
        z += 218000.0 * (5.56e-05 - Q.e3) * (Q.sj3_dr23 - 0.162)
    if Q.planar_flow < 0.106 and Q.centroid_offset < 0.0179:
        z += -400.0 * (0.106 - Q.planar_flow) * (0.0179 - Q.centroid_offset)
    if Q.planar_flow < 0.11 and Q.lam1 < 0.00746:
        z += -4200.0 * (0.11 - Q.planar_flow) * (0.00746 - Q.lam1)
    if Q.planar_flow < 0.115 and Q.lam1 < 0.0165:
        z += 987.0 * (0.115 - Q.planar_flow) * (0.0165 - Q.lam1)
    if Q.planar_flow < 0.109 and Q.mean_eta > -0.00684:
        z += -98.8 * (0.109 - Q.planar_flow) * (Q.mean_eta - -0.00684)
    if Q.planar_flow < 0.1 and Q.sum_pt_top5 < 652.0:
        z += -0.0177 * (0.1 - Q.planar_flow) * (652.0 - Q.sum_pt_top5)
    if Q.planar_flow < 0.112 and Q.width < 0.00601:
        z += 3620.0 * (0.112 - Q.planar_flow) * (0.00601 - Q.width)
    if Q.sj2_dr > 0.13 and Q.D2_b2 < 0.0886:
        z += -168.0 * (Q.sj2_dr - 0.13) * (0.0886 - Q.D2_b2)
    if Q.sj2_dr > 0.159 and Q.D2_b2 < 0.0881:
        z += 378.0 * (Q.sj2_dr - 0.159) * (0.0881 - Q.D2_b2)
    if Q.sj2_dr > 0.198 and Q.D2_b2 < 0.0872:
        z += -264.0 * (Q.sj2_dr - 0.198) * (0.0872 - Q.D2_b2)
    if Q.sj2_dr > 0.164 and Q.dr_2 < 0.0332:
        z += -569.0 * (Q.sj2_dr - 0.164) * (0.0332 - Q.dr_2)
    if Q.sj2_dr > 0.197 and Q.dr_2 < 0.0517:
        z += 289.0 * (Q.sj2_dr - 0.197) * (0.0517 - Q.dr_2)
    if Q.sj2_dr > 0.203 and Q.z_5 < 0.105:
        z += -74.7 * (Q.sj2_dr - 0.203) * (0.105 - Q.z_5)
    if Q.z_dr_0p05_0p1 < 0.627 and Q.sum_pt < 714.0:
        z += 0.00254 * (0.627 - Q.z_dr_0p05_0p1) * (714.0 - Q.sum_pt)
    if Q.z_dr_0p05_0p1 > 0.745 and Q.zdr_5 > 0.0154:
        z += -3190.0 * (Q.z_dr_0p05_0p1 - 0.745) * (Q.zdr_5 - 0.0154)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.00133
    if Q.e2 < 0.0409:
        z += -156.0 * Q.e2 + 6.3804
    if Q.e2_sq < 0.00818:
        z += 1130.0 * Q.e2_sq - 9.2434
    if Q.girth < 0.102:
        z += 91.6 * Q.girth - 9.3432
    if Q.girth2 < 0.00739:
        z += 546.0 * Q.girth2 - 4.03494
    if Q.girth2_top2 < 0.000513:
        z += 10700.0 * Q.girth2_top2 - 5.4891
    if Q.mass_over_sum_pt_sq < 0.00708:
        z += -655.0 * Q.mass_over_sum_pt_sq + 4.6374
    if Q.sj2_dr < 0.159:
        z += -8.3 * Q.sj2_dr - 0.1083
    if 0.159 <= Q.sj2_dr < 0.199:
        z += 35.7 * Q.sj2_dr - 7.1043
    if Q.sj3_pair_mass_max >= 81.1:
        z += -0.049 * Q.sj3_pair_mass_max + 3.9739
    if Q.width < 0.0133:
        z += -710.0 * Q.width + 9.443
    if Q.N2 < 0.229 and Q.n_dr_0p1_0p2 < 4.0:
        z += 1.25 * (0.229 - Q.N2) * (4.0 - Q.n_dr_0p1_0p2)
    if Q.N2 < 0.218 and Q.sum_pt_top5 > 498.0:
        z += 0.0181 * (0.218 - Q.N2) * (Q.sum_pt_top5 - 498.0)
    if Q.N2 < 0.223 and Q.z_dr_0p05_0p1 < 0.548:
        z += -13.9 * (0.223 - Q.N2) * (0.548 - Q.z_dr_0p05_0p1)
    if Q.girth2 < 0.00752 and Q.D2 < 0.731:
        z += -1140.0 * (0.00752 - Q.girth2) * (0.731 - Q.D2)
    if Q.mass_over_sum_pt < 0.133 and Q.D2 < 0.711:
        z += 31.3 * (0.133 - Q.mass_over_sum_pt) * (0.711 - Q.D2)
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
