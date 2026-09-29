"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network (step 4; no mass observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 89.9% of jets.

Quantities:
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
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.mratio_min_012         smallest pair mass / mass of the 3 hardest (dimensionless)
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_over_pt0           pT1 / pT0
  Q.pt1_dr01               pT1 · ΔR01
  Q.z_3rd                  3rd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.abseta_0               |Δη| of particle 0
  Q.abseta_6               |Δη| of particle 6
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr0_5                  ΔR between particle 5 and the hardest particle
  Q.dr0_6                  ΔR between particle 6 and the hardest particle
  Q.dr0_7                  ΔR between particle 7 and the hardest particle
  Q.dr1_7                  ΔR between particle 7 and the 2nd-hardest particle
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.dr12                   ΔR between particles 1 and 2
  Q.eta_5                  Δη of particle 5
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
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
        e4=ecf('e4'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        mratio_min_012=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)) / max(math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2), 1e-9),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        z_3rd=zs[2],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        abseta_0=abs(eta[0]),
        abseta_6=abs(eta[6]),
        sj2_dr=subjets(2)["dr"][0],
        dr0_5=math.sqrt(dist2(0, 5)) if pt[5] > 0 else 0.0,
        dr0_6=math.sqrt(dist2(0, 6)) if pt[6] > 0 else 0.0,
        dr0_7=math.sqrt(dist2(0, 7)) if pt[7] > 0 else 0.0,
        dr1_7=math.sqrt(dist2(1, 7)) if pt[7] > 0 else 0.0,
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        dr12=math.sqrt(dist2(1, 2)) if pt[2] > 0 else 0.0,
        eta_5=eta[5],
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
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
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -1.08
    if Q.centroid_offset >= 0.05:
        z += -324.0 * Q.centroid_offset + 16.2
    if Q.e2 < 0.0316:
        z += -99.1 * Q.e2 + 3.13156
    if Q.girth < 0.0153:
        z += 283.3 * Q.girth - 10.3124
    if 0.0153 <= Q.girth < 0.083:
        z += 88.3 * Q.girth - 7.3289
    if Q.girth2 < 0.00559:
        z += -37.0 * Q.girth2 + 5.7369
    if 0.00559 <= Q.girth2 < 0.0128:
        z += -767.0 * Q.girth2 + 9.8176
    if Q.n_dr_0_0p05 >= 5.06:
        z += 0.35 * Q.n_dr_0_0p05 - 1.771
    if Q.planar_flow < 0.145:
        z += -6.04 * Q.planar_flow + 0.8758
    if Q.sd_rg < 0.113:
        z += -5.94 * Q.sd_rg + 0.06932
    if 0.113 <= Q.sd_rg < 0.178:
        z += 9.26 * Q.sd_rg - 1.64828
    if Q.sum_pt < 783.0:
        z += 0.00234 * Q.sum_pt - 1.83222
    if Q.tau2 < 0.0135:
        z += 155.0 * Q.tau2 - 2.0925
    if Q.width < 0.00374:
        z += 844.0 * Q.width - 3.15656
    if Q.D2_b2 < 0.262 and Q.pt_5 > 16.1:
        z += -0.0428 * (0.262 - Q.D2_b2) * (Q.pt_5 - 16.1)
    if Q.e3 < 8e-05 and Q.dr0_7 > 0.241:
        z += -536000.0 * (8e-05 - Q.e3) * (Q.dr0_7 - 0.241)
    if Q.girth2 < 0.00505 and Q.M3 < 0.094:
        z += -2540.0 * (0.00505 - Q.girth2) * (0.094 - Q.M3)
    if Q.girth2 < 0.0189 and Q.M3 < 0.0889:
        z += 545.0 * (0.0189 - Q.girth2) * (0.0889 - Q.M3)
    if Q.girth2 < 0.0183 and Q.centroid_offset > 0.0105:
        z += -4400.0 * (0.0183 - Q.girth2) * (Q.centroid_offset - 0.0105)
    if Q.girth2 < 0.00558 and Q.dr0_5 > 0.172:
        z += 11700.0 * (0.00558 - Q.girth2) * (Q.dr0_5 - 0.172)
    if Q.planar_flow < 0.171 and Q.D2_b2 < 1.21:
        z += 5.93 * (0.171 - Q.planar_flow) * (1.21 - Q.D2_b2)
    if Q.sum_pt < 747.0 and Q.D2_b2 < 1.07:
        z += -0.00526 * (747.0 - Q.sum_pt) * (1.07 - Q.D2_b2)
    if Q.tau1 < 0.0631 and Q.dr0_6 > 0.177:
        z += -669.0 * (0.0631 - Q.tau1) * (Q.dr0_6 - 0.177)
    if Q.tau1 < 0.0629 and Q.dr_5 > 0.163:
        z += -967.0 * (0.0629 - Q.tau1) * (Q.dr_5 - 0.163)
    if Q.width < 0.00438 and Q.dr0_6 > 0.177:
        z += 18700.0 * (0.00438 - Q.width) * (Q.dr0_6 - 0.177)
    if Q.width < 0.0036 and Q.dr0_7 > 0.179:
        z += 20400.0 * (0.0036 - Q.width) * (Q.dr0_7 - 0.179)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.27
    if Q.e2_sq < 0.00594:
        z += -576.0 * Q.e2_sq + 3.42144
    if Q.girth < 0.102:
        z += 21.6 * Q.girth - 2.2032
    if Q.girth2 < 0.00625:
        z += 309.0 * Q.girth2 - 1.93125
    if 6.38 <= Q.log_sum_pt < 6.57:
        z += 6.56 * Q.log_sum_pt - 41.8528
    if Q.log_sum_pt >= 6.57:
        z += 13.93 * Q.log_sum_pt - 90.2737
    if 37.8 <= Q.pt_7 < 53.7:
        z += 0.131 * Q.pt_7 - 4.9518
    if Q.pt_7 >= 53.7:
        z += 0.019 * Q.pt_7 + 1.0626
    if Q.sj3_dr_max < 0.159:
        z += -1.93 * Q.sj3_dr_max - 0.49233
    if 0.159 <= Q.sj3_dr_max < 0.344:
        z += 4.32 * Q.sj3_dr_max - 1.48608
    if Q.width < 0.00868:
        z += 488.0 * Q.width - 4.23584
    if Q.z_7 < 0.052:
        z += 121.1 * Q.z_7 - 7.02392
    if 0.052 <= Q.z_7 < 0.0616:
        z += 75.7 * Q.z_7 - 4.66312
    if Q.zdr_0 < 0.0204:
        z += 41.2 * Q.zdr_0 - 0.84048
    if Q.lam1 < 0.00857 and Q.centroid_offset > 0.0202:
        z += 6520.0 * (0.00857 - Q.lam1) * (Q.centroid_offset - 0.0202)
    if Q.lam1 < 0.00807 and Q.dr_0 < 0.147:
        z += 3760.0 * (0.00807 - Q.lam1) * (0.147 - Q.dr_0)
    if Q.lam2 < 0.00342 and Q.tau21_b2 < 0.291:
        z += 423.0 * (0.00342 - Q.lam2) * (0.291 - Q.tau21_b2)
    if Q.log_sum_pt > 6.65 and Q.dr01 < 0.139:
        z += 21.7 * (Q.log_sum_pt - 6.65) * (0.139 - Q.dr01)
    if Q.log_sum_pt > 6.37 and Q.dr_0 < 0.0793:
        z += -82.0 * (Q.log_sum_pt - 6.37) * (0.0793 - Q.dr_0)
    if Q.log_sum_pt > 6.58 and Q.girth2_top3 < 0.00784:
        z += -704.0 * (Q.log_sum_pt - 6.58) * (0.00784 - Q.girth2_top3)
    if Q.pt_7 > 34.7 and Q.e3 < 3.08e-05:
        z += -2480.0 * (Q.pt_7 - 34.7) * (3.08e-05 - Q.e3)
    if Q.z_7 < 0.0522 and Q.D2_b2 < 0.121:
        z += -395.0 * (0.0522 - Q.z_7) * (0.121 - Q.D2_b2)
    if Q.z_7 < 0.0615 and Q.e3 < 0.000187:
        z += 252000.0 * (0.0615 - Q.z_7) * (0.000187 - Q.e3)
    if Q.z_7 < 0.061 and Q.zdr_0 < 0.0409:
        z += 1070.0 * (0.061 - Q.z_7) * (0.0409 - Q.zdr_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.525
    if Q.LHA >= 0.118:
        z += -9.8 * Q.LHA + 1.1564
    if Q.e2_sq < 0.00122:
        z += 779.0 * Q.e2_sq - 2.29953
    if 0.00122 <= Q.e2_sq < 0.00727:
        z += 223.0 * Q.e2_sq - 1.62121
    if Q.log_sum_pt >= 6.84:
        z += 15.3 * Q.log_sum_pt - 104.652
    if Q.pt_7 < 34.4:
        z += 0.1424 * Q.pt_7 - 5.79452
    if 34.4 <= Q.pt_7 < 34.5:
        z += 0.2397 * Q.pt_7 - 9.14164
    if 34.5 <= Q.pt_7 < 53.3:
        z += 0.1442 * Q.pt_7 - 5.84689
    if Q.pt_7 >= 53.3:
        z += 0.0973 * Q.pt_7 - 3.34712
    if Q.sum_pt < 610.0:
        z += -0.02104 * Q.sum_pt + 15.05857
    if 610.0 <= Q.sum_pt < 766.0:
        z += -0.01192 * Q.sum_pt + 9.49537
    if 766.0 <= Q.sum_pt < 831.0:
        z += -0.00561 * Q.sum_pt + 4.66191
    if Q.sum_pt >= 981.0:
        z += -0.0129 * Q.sum_pt + 12.6549
    if Q.width < 0.00173:
        z += -565.0 * Q.width + 2.38334
    if 0.00173 <= Q.width < 0.0128:
        z += -127.0 * Q.width + 1.6256
    if Q.z_7 < 0.0225:
        z += -2.9 * Q.z_7 + 3.59013
    if 0.0225 <= Q.z_7 < 0.0679:
        z += -77.3 * Q.z_7 + 5.26413
    if 0.0679 <= Q.z_7 < 0.0681:
        z += -152.2 * Q.z_7 + 10.34984
    if Q.z_7 >= 0.0681:
        z += -74.9 * Q.z_7 + 5.08571
    if Q.zdr_0 < 0.023:
        z += -21.3 * Q.zdr_0 + 0.4899
    if Q.C2 < 0.0404 and Q.C2_b2 < 0.024:
        z += -532.0 * (0.0404 - Q.C2) * (0.024 - Q.C2_b2)
    if Q.centroid_offset < 0.00582 and Q.C3 < 0.0163:
        z += -18200.0 * (0.00582 - Q.centroid_offset) * (0.0163 - Q.C3)
    if Q.lam1 < 0.00721 and Q.max_dr < 0.256:
        z += 1460.0 * (0.00721 - Q.lam1) * (0.256 - Q.max_dr)
    if Q.lam1 < 0.00714 and Q.pt_6 < 60.7:
        z += -2.75 * (0.00714 - Q.lam1) * (60.7 - Q.pt_6)
    if Q.log_sum_pt < 6.61 and Q.D2_b2 < 1.15:
        z += 5.18 * (6.61 - Q.log_sum_pt) * (1.15 - Q.D2_b2)
    if Q.log_sum_pt < 6.7 and Q.D2_b2 < 1.15:
        z += -4.0 * (6.7 - Q.log_sum_pt) * (1.15 - Q.D2_b2)
    if Q.sum_pt > 976.0 and Q.abseta_0 < 0.0147:
        z += 0.756 * (Q.sum_pt - 976.0) * (0.0147 - Q.abseta_0)
    if Q.sum_pt > 954.0 and Q.n_pt_above_50 > 4.97:
        z += -0.00349 * (Q.sum_pt - 954.0) * (Q.n_pt_above_50 - 4.97)
    if Q.sum_pt_top5 > 782.0 and Q.abseta_0 < 0.0148:
        z += -0.632 * (Q.sum_pt_top5 - 782.0) * (0.0148 - Q.abseta_0)
    return max(0.0, z)


def neuron_3(Q):
    z = -2.8
    if Q.e2_sq >= 0.00748:
        z += 439.0 * Q.e2_sq - 3.28372
    if Q.girth >= 0.066:
        z += 159.0 * Q.girth - 10.494
    if Q.girth2 >= 0.00875:
        z += -853.0 * Q.girth2 + 7.46375
    if Q.max_dr >= 0.105:
        z += 8.1 * Q.max_dr - 0.8505
    if 0.178 <= Q.sj2_dr < 0.268:
        z += 49.5 * Q.sj2_dr - 8.811
    if Q.sj2_dr >= 0.268:
        z += 5.0 * Q.sj2_dr + 3.115
    if Q.tau1 >= 0.0801:
        z += -58.9 * Q.tau1 + 4.71789
    if 0.00584 <= Q.width < 0.02:
        z += 277.0 * Q.width - 1.61768
    if Q.width >= 0.02:
        z += 475.0 * Q.width - 5.57768
    if Q.sj2_dr > 0.189 and Q.eccentricity > 0.941:
        z += 194.0 * (Q.sj2_dr - 0.189) * (Q.eccentricity - 0.941)
    if Q.sj2_dr > 0.179 and Q.girth2_top2 < 0.00187:
        z += -30100.0 * (Q.sj2_dr - 0.179) * (0.00187 - Q.girth2_top2)
    if Q.sj2_dr > 0.269 and Q.girth2_top2 < 0.00767:
        z += 4570.0 * (Q.sj2_dr - 0.269) * (0.00767 - Q.girth2_top2)
    if Q.sj2_dr > 0.17 and Q.n_dr_0p05_0p1 < 7.1:
        z += -4.37 * (Q.sj2_dr - 0.17) * (7.1 - Q.n_dr_0p05_0p1)
    if Q.sj2_dr > 0.157 and Q.n_dr_0p1_0p2 > 0.955:
        z += 1.77 * (Q.sj2_dr - 0.157) * (Q.n_dr_0p1_0p2 - 0.955)
    if Q.sj2_dr > 0.165 and Q.sum_pt < 991.0:
        z += -0.0293 * (Q.sj2_dr - 0.165) * (991.0 - Q.sum_pt)
    if Q.sj2_dr > 0.18 and Q.tau2 < 0.0651:
        z += 271.0 * (Q.sj2_dr - 0.18) * (0.0651 - Q.tau2)
    if Q.sj2_dr > 0.302 and Q.z_dr_0p05_0p1 > 0.116:
        z += -57.7 * (Q.sj2_dr - 0.302) * (Q.z_dr_0p05_0p1 - 0.116)
    if Q.width > 0.0061 and Q.log_sum_pt > 6.15:
        z += -413.0 * (Q.width - 0.0061) * (Q.log_sum_pt - 6.15)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.764
    if Q.C2 >= 0.0724:
        z += -45.8 * Q.C2 + 3.31592
    if Q.N2 < 0.238:
        z += -17.0 * Q.N2 + 4.046
    if Q.e2 >= 0.0212:
        z += 116.0 * Q.e2 - 2.4592
    if Q.e2_sq < 0.00334:
        z += -360.0 * Q.e2_sq + 3.996
    if 0.00334 <= Q.e2_sq < 0.0111:
        z += -839.0 * Q.e2_sq + 5.59586
    if Q.e2_sq >= 0.0111:
        z += -479.0 * Q.e2_sq + 1.59986
    if Q.e3 < 0.000205:
        z += -20100.0 * Q.e3 + 4.1205
    if Q.girth2 < 0.0026:
        z += 20.0 * Q.girth2 - 6.8563
    if 0.0026 <= Q.girth2 < 0.00873:
        z += 1110.0 * Q.girth2 - 9.6903
    if Q.lam2 < 0.00117:
        z += 2030.0 * Q.lam2 - 2.3751
    if Q.max_dr >= 0.0975:
        z += 12.5 * Q.max_dr - 1.21875
    if Q.sj2_dr >= 0.219:
        z += -18.0 * Q.sj2_dr + 3.942
    if Q.sj3_dr_min < 0.21:
        z += -17.3 * Q.sj3_dr_min + 3.633
    if Q.tau2 < 0.0134:
        z += 108.0 * Q.tau2 - 1.4472
    if Q.D2 < 0.767 and Q.pt_6 > 13.8:
        z += 0.0979 * (0.767 - Q.D2) * (Q.pt_6 - 13.8)
    if Q.N2 < 0.228 and Q.planar_flow < 0.452:
        z += -38.0 * (0.228 - Q.N2) * (0.452 - Q.planar_flow)
    if Q.girth2_top2 < 0.00575 and Q.centroid_offset > 0.0132:
        z += 21200.0 * (0.00575 - Q.girth2_top2) * (Q.centroid_offset - 0.0132)
    if Q.lam2 < 0.000974 and Q.dr_max_012 < 0.217:
        z += -3580.0 * (0.000974 - Q.lam2) * (0.217 - Q.dr_max_012)
    if Q.sj3_dr_max < 0.312 and Q.log_sum_pt < 6.74:
        z += -55.8 * (0.312 - Q.sj3_dr_max) * (6.74 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.931
    if Q.LHA < 0.211:
        z += -40.7 * Q.LHA + 8.5877
    if Q.girth < 0.00762:
        z += -285.0 * Q.girth + 2.1717
    if Q.log_sum_pt >= 6.9:
        z += -22.0 * Q.log_sum_pt + 151.8
    if Q.max_dr < 0.164:
        z += -12.6 * Q.max_dr + 2.0664
    if Q.pt_7 >= 36.1:
        z += -0.0987 * Q.pt_7 + 3.56307
    if Q.sj3_dr_max < 0.288:
        z += 8.23 * Q.sj3_dr_max - 2.37024
    if Q.sum_pt < 712.0:
        z += 0.0116 * Q.sum_pt - 8.2592
    if Q.sum_pt_top5 < 427.0:
        z += 0.0419 * Q.sum_pt_top5 - 17.8913
    if Q.z_6 < 0.054:
        z += -37.0 * Q.z_6 + 1.998
    if Q.z_7 < 0.0321:
        z += -104.0 * Q.z_7 + 3.3384
    if Q.LHA < 0.217 and Q.log_sum_pt < 6.81:
        z += -186.0 * (0.217 - Q.LHA) * (6.81 - Q.log_sum_pt)
    if Q.LHA < 0.212 and Q.z_6 < 0.098:
        z += -206.0 * (0.212 - Q.LHA) * (0.098 - Q.z_6)
    if Q.LHA < 0.174 and Q.zdr_0 < 0.00587:
        z += -2900.0 * (0.174 - Q.LHA) * (0.00587 - Q.zdr_0)
    if Q.centroid_offset < 0.0261 and Q.pt_5 > 34.6:
        z += -0.737 * (0.0261 - Q.centroid_offset) * (Q.pt_5 - 34.6)
    if Q.centroid_offset < 0.0284 and Q.z_5 < 0.0962:
        z += -693.0 * (0.0284 - Q.centroid_offset) * (0.0962 - Q.z_5)
    if Q.e2 < 0.0421 and Q.planar_flow < 0.326:
        z += 47.9 * (0.0421 - Q.e2) * (0.326 - Q.planar_flow)
    if Q.e2 < 0.037 and Q.zdr_0 > 0.00385:
        z += 3060.0 * (0.037 - Q.e2) * (Q.zdr_0 - 0.00385)
    if Q.girth2 < 0.00244 and Q.centroid_offset < 0.0186:
        z += 70200.0 * (0.00244 - Q.girth2) * (0.0186 - Q.centroid_offset)
    if Q.girth2 < 0.00282 and Q.n_dr_0p2_0p4 < 1.96:
        z += 413.0 * (0.00282 - Q.girth2) * (1.96 - Q.n_dr_0p2_0p4)
    if Q.pt_5 < 22.8 and Q.abseta_6 < 0.181:
        z += 1.02 * (22.8 - Q.pt_5) * (0.181 - Q.abseta_6)
    if Q.sj3_dr_max < 0.302 and Q.pt1_dr01 > 13.7:
        z += -0.692 * (0.302 - Q.sj3_dr_max) * (Q.pt1_dr01 - 13.7)
    if Q.sum_pt < 710.0 and Q.dr_0 < 0.035:
        z += 0.904 * (710.0 - Q.sum_pt) * (0.035 - Q.dr_0)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.45
    if Q.C3 < 0.0387:
        z += -48.1 * Q.C3 + 1.86147
    if Q.D3 < 1.67:
        z += 0.999 * Q.D3 - 1.66833
    if 0.00732 <= Q.centroid_offset < 0.0165:
        z += 77.8 * Q.centroid_offset - 0.569496
    if Q.centroid_offset >= 0.0165:
        z += 228.8 * Q.centroid_offset - 3.060996
    if Q.e2_sq >= 0.0116:
        z += -128.0 * Q.e2_sq + 1.4848
    if Q.girth < 0.0268:
        z += -200.0 * Q.girth + 5.36
    if Q.girth2 < 0.00867:
        z += 1490.0 * Q.girth2 - 12.9183
    if Q.girth2_top2 < 0.00998:
        z += -97.1 * Q.girth2_top2 + 0.969058
    if Q.lam1 < 0.0121:
        z += -248.0 * Q.lam1 + 3.0008
    if Q.log_sum_pt < 6.33:
        z += -11.01 * Q.log_sum_pt + 70.1445
    if 6.33 <= Q.log_sum_pt < 6.65:
        z += -1.41 * Q.log_sum_pt + 9.3765
    if Q.max_dr < 0.16:
        z += 21.9 * Q.max_dr - 3.504
    if Q.n_dr_0_0p05 < 2.95:
        z += -0.335 * Q.n_dr_0_0p05 + 0.98825
    if Q.pt_6 < 29.8:
        z += -0.169 * Q.pt_6 + 5.0362
    if 0.072 <= Q.sj3_dr_max < 0.18:
        z += -9.8 * Q.sj3_dr_max + 0.7056
    if Q.sj3_dr_max >= 0.18:
        z += 12.2 * Q.sj3_dr_max - 3.2544
    if Q.sj3_dr_min >= 0.0185:
        z += -15.5 * Q.sj3_dr_min + 0.28675
    if Q.sum_pt >= 991.0:
        z += 0.0124 * Q.sum_pt - 12.2884
    if Q.tau1 < 0.104:
        z += -47.7 * Q.tau1 + 4.9608
    if Q.z_6 < 0.0216:
        z += 558.8 * Q.z_6 - 13.30248
    if 0.0216 <= Q.z_6 < 0.0346:
        z += 94.8 * Q.z_6 - 3.28008
    if Q.LHA > 0.267 and Q.eccentricity > 0.892:
        z += 180.0 * (Q.LHA - 0.267) * (Q.eccentricity - 0.892)
    if Q.centroid_offset > 0.00856 and Q.mean_phi2 < 0.0148:
        z += 4520.0 * (Q.centroid_offset - 0.00856) * (0.0148 - Q.mean_phi2)
    if Q.centroid_offset > 0.00848 and Q.n_dr_0p05_0p1 < 6.16:
        z += 5.25 * (Q.centroid_offset - 0.00848) * (6.16 - Q.n_dr_0p05_0p1)
    if Q.centroid_offset > 0.00932 and Q.sj2_dr < 0.174:
        z += 722.0 * (Q.centroid_offset - 0.00932) * (0.174 - Q.sj2_dr)
    if Q.centroid_offset > 0.00811 and Q.sum_pt < 714.0:
        z += 0.42 * (Q.centroid_offset - 0.00811) * (714.0 - Q.sum_pt)
    if Q.centroid_offset > 0.0162 and Q.sum_pt < 993.0:
        z += -0.773 * (Q.centroid_offset - 0.0162) * (993.0 - Q.sum_pt)
    if Q.eccentricity > 0.905 and Q.mean_eta > 0.0178:
        z += -397.0 * (Q.eccentricity - 0.905) * (Q.mean_eta - 0.0178)
    if Q.lam1 < 0.0117 and Q.pt_6 < 19.6:
        z += 28.9 * (0.0117 - Q.lam1) * (19.6 - Q.pt_6)
    if Q.log_sum_pt < 6.33 and Q.pt_7 > 25.5:
        z += -0.533 * (6.33 - Q.log_sum_pt) * (Q.pt_7 - 25.5)
    if Q.log_sum_pt < 6.64 and Q.z_7 < 0.0586:
        z += 410.0 * (6.64 - Q.log_sum_pt) * (0.0586 - Q.z_7)
    if Q.sj3_dr_max > 0.0735 and Q.eccentricity > 0.9:
        z += -209.0 * (Q.sj3_dr_max - 0.0735) * (Q.eccentricity - 0.9)
    if Q.sj3_dr_max > 0.178 and Q.eccentricity > 0.901:
        z += 301.0 * (Q.sj3_dr_max - 0.178) * (Q.eccentricity - 0.901)
    if Q.sum_pt > 985.0 and Q.dr01 > 0.141:
        z += 2.02 * (Q.sum_pt - 985.0) * (Q.dr01 - 0.141)
    if Q.sum_pt > 985.0 and Q.dr01 > 0.166:
        z += -2.2 * (Q.sum_pt - 985.0) * (Q.dr01 - 0.166)
    if Q.width < 0.0135 and Q.planar_flow < 0.335:
        z += -450.0 * (0.0135 - Q.width) * (0.335 - Q.planar_flow)
    if Q.z_6 < 0.0216 and Q.log_sum_pt < 6.84:
        z += 4770.0 * (0.0216 - Q.z_6) * (6.84 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_7(Q):
    z = 7.77
    if Q.LHA < 0.213:
        z += 4.7 * Q.LHA + 0.1815
    if 0.213 <= Q.LHA < 0.294:
        z += -14.6 * Q.LHA + 4.2924
    if Q.centroid_offset < 0.0215:
        z += -58.4 * Q.centroid_offset + 1.2556
    if Q.e2_sq < 0.00535:
        z += 627.0 * Q.e2_sq - 6.97523
    if 0.00535 <= Q.e2_sq < 0.00714:
        z += 1022.0 * Q.e2_sq - 9.08848
    if 0.00714 <= Q.e2_sq < 0.0082:
        z += 1690.0 * Q.e2_sq - 13.858
    if Q.girth < 0.0876:
        z += 66.4 * Q.girth - 5.81664
    if 0.00442 <= Q.girth2 < 0.00745:
        z += -474.0 * Q.girth2 + 2.09508
    if Q.girth2 >= 0.00745:
        z += -1432.0 * Q.girth2 + 9.23218
    if Q.lam2 < 0.000461:
        z += 1030.0 * Q.lam2 - 0.47483
    if Q.max_dr < 0.204:
        z += -7.66 * Q.max_dr + 1.56264
    if Q.sj2_dr < 0.159:
        z += -7.1 * Q.sj2_dr + 0.651
    if 0.159 <= Q.sj2_dr < 0.186:
        z += 17.7 * Q.sj2_dr - 3.2922
    if Q.sj3_dr_max < 0.142:
        z += -11.8 * Q.sj3_dr_max + 1.6756
    if Q.tau1 < 0.0509:
        z += -64.8 * Q.tau1 + 5.08878
    if 0.0509 <= Q.tau1 < 0.095:
        z += -40.6 * Q.tau1 + 3.857
    if Q.width < 0.00562:
        z += 792.0 * Q.width - 4.45104
    if Q.centroid_offset < 0.0208 and Q.D2_b2 < 4.67:
        z += -19.8 * (0.0208 - Q.centroid_offset) * (4.67 - Q.D2_b2)
    if Q.e2_sq < 0.0119 and Q.pt_7 < 48.7:
        z += -1.42 * (0.0119 - Q.e2_sq) * (48.7 - Q.pt_7)
    if Q.e2_sq < 0.00738 and Q.sj3_dr13 > 0.201:
        z += 2510.0 * (0.00738 - Q.e2_sq) * (Q.sj3_dr13 - 0.201)
    if Q.girth2 > 0.0136 and Q.eccentricity > 0.934:
        z += 16000.0 * (Q.girth2 - 0.0136) * (Q.eccentricity - 0.934)
    if Q.girth2 > 0.00448 and Q.planar_flow < 0.205:
        z += 1810.0 * (Q.girth2 - 0.00448) * (0.205 - Q.planar_flow)
    if Q.max_dr < 0.2 and Q.n_dr_0p1_0p2 > 4.12:
        z += -7.33 * (0.2 - Q.max_dr) * (Q.n_dr_0p1_0p2 - 4.12)
    if Q.planar_flow < 0.196 and Q.log_sum_pt > 6.42:
        z += 34.0 * (0.196 - Q.planar_flow) * (Q.log_sum_pt - 6.42)
    if Q.planar_flow < 0.207 and Q.pt_6 < 33.6:
        z += -0.331 * (0.207 - Q.planar_flow) * (33.6 - Q.pt_6)
    if Q.planar_flow < 0.198 and Q.z_7 < 0.0524:
        z += -155.0 * (0.198 - Q.planar_flow) * (0.0524 - Q.z_7)
    if Q.sj3_dr_max < 0.214 and Q.eccentricity > 0.945:
        z += -166.0 * (0.214 - Q.sj3_dr_max) * (Q.eccentricity - 0.945)
    if Q.width < 0.00568 and Q.mean_eta < -0.00679:
        z += 18900.0 * (0.00568 - Q.width) * (-0.00679 - Q.mean_eta)
    if Q.width < 0.00569 and Q.mean_eta > 0.00949:
        z += 18900.0 * (0.00569 - Q.width) * (Q.mean_eta - 0.00949)
    if Q.width < 0.00569 and Q.mean_phi < -0.00458:
        z += 17100.0 * (0.00569 - Q.width) * (-0.00458 - Q.mean_phi)
    if Q.width < 0.00569 and Q.mean_phi > 0.00912:
        z += 20100.0 * (0.00569 - Q.width) * (Q.mean_phi - 0.00912)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.731
    if Q.girth2 < 0.00539:
        z += -706.0 * Q.girth2 + 3.80534
    if Q.sj3_dr_max < 0.129:
        z += 21.1 * Q.sj3_dr_max - 2.7219
    if Q.LHA < 0.188 and Q.centroid_offset > 0.0171:
        z += -6600.0 * (0.188 - Q.LHA) * (Q.centroid_offset - 0.0171)
    if Q.LHA < 0.201 and Q.z_dr_0p2_0p4 < 0.198:
        z += -150.0 * (0.201 - Q.LHA) * (0.198 - Q.z_dr_0p2_0p4)
    if Q.girth2 < 0.00528 and Q.centroid_offset > 0.00735:
        z += -25900.0 * (0.00528 - Q.girth2) * (Q.centroid_offset - 0.00735)
    if Q.girth2 < 0.00632 and Q.centroid_offset < 0.0222:
        z += 13700.0 * (0.00632 - Q.girth2) * (0.0222 - Q.centroid_offset)
    if Q.girth2 < 0.00427 and Q.log_sum_pt < 6.52:
        z += -1110.0 * (0.00427 - Q.girth2) * (6.52 - Q.log_sum_pt)
    if Q.girth2 < 0.00543 and Q.pt_7 < 50.2:
        z += -5.96 * (0.00543 - Q.girth2) * (50.2 - Q.pt_7)
    if Q.log_sum_pt > 6.68 and Q.pt_7 > 19.2:
        z += -0.183 * (Q.log_sum_pt - 6.68) * (Q.pt_7 - 19.2)
    if Q.sj3_dr_max < 0.194 and Q.girth2_top3 > 0.00121:
        z += 1630.0 * (0.194 - Q.sj3_dr_max) * (Q.girth2_top3 - 0.00121)
    if Q.sj3_dr_max < 0.223 and Q.lam2 < 0.000253:
        z += 41100.0 * (0.223 - Q.sj3_dr_max) * (0.000253 - Q.lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -4.83
    if Q.centroid_offset < 0.0242:
        z += -93.6 * Q.centroid_offset + 2.26512
    if Q.dr_0 < 0.0662:
        z += 20.1 * Q.dr_0 - 1.33062
    if Q.e2_sq < 0.00293:
        z += 1070.0 * Q.e2_sq - 3.1351
    if Q.e3 >= 0.000257:
        z += 4050.0 * Q.e3 - 1.04085
    if Q.girth < 0.0541:
        z += 82.3 * Q.girth - 4.45243
    if Q.girth2 < 0.0036:
        z += -1910.0 * Q.girth2 + 6.876
    if Q.lam2 < 0.000178:
        z += -6390.0 * Q.lam2 + 1.13742
    if Q.lam2 >= 0.0009:
        z += 416.0 * Q.lam2 - 0.3744
    if Q.log_sum_pt < 6.9:
        z += -6.33 * Q.log_sum_pt + 43.677
    if Q.max_dr < 0.114:
        z += -15.1 * Q.max_dr + 1.7214
    if Q.sj2_dr < 0.126:
        z += 23.9 * Q.sj2_dr - 3.0114
    if Q.sj3_dr_max < 0.142:
        z += 12.3 * Q.sj3_dr_max - 0.789
    if 0.142 <= Q.sj3_dr_max < 0.198:
        z += -17.1 * Q.sj3_dr_max + 3.3858
    if Q.width < 0.00652:
        z += -1520.0 * Q.width + 9.9104
    if Q.LHA < 0.216 and Q.mean_eta < 0.00318:
        z += -618.0 * (0.216 - Q.LHA) * (0.00318 - Q.mean_eta)
    if Q.centroid_offset < 0.0163 and Q.M2 < 0.0457:
        z += -2140.0 * (0.0163 - Q.centroid_offset) * (0.0457 - Q.M2)
    if Q.centroid_offset < 0.0224 and Q.dr_7 < 0.0834:
        z += 749.0 * (0.0224 - Q.centroid_offset) * (0.0834 - Q.dr_7)
    if Q.centroid_offset < 0.0183 and Q.mratio_min_012 < 0.0768:
        z += -900.0 * (0.0183 - Q.centroid_offset) * (0.0768 - Q.mratio_min_012)
    if Q.centroid_offset < 0.0183 and Q.n_dr_0p05_0p1 < 5.08:
        z += 13.3 * (0.0183 - Q.centroid_offset) * (5.08 - Q.n_dr_0p05_0p1)
    if Q.centroid_offset < 0.0188 and Q.pt1_over_pt0 > 0.151:
        z += -97.1 * (0.0188 - Q.centroid_offset) * (Q.pt1_over_pt0 - 0.151)
    if Q.centroid_offset < 0.0193 and Q.pt_1 > 93.5:
        z += 0.532 * (0.0193 - Q.centroid_offset) * (Q.pt_1 - 93.5)
    if Q.centroid_offset < 0.0223 and Q.tau21_b2 > 0.00316:
        z += 87.5 * (0.0223 - Q.centroid_offset) * (Q.tau21_b2 - 0.00316)
    if Q.centroid_offset < 0.0186 and Q.tau3 < 0.0139:
        z += 9250.0 * (0.0186 - Q.centroid_offset) * (0.0139 - Q.tau3)
    if Q.girth < 0.0705 and Q.mean_phi < 0.00149:
        z += -1690.0 * (0.0705 - Q.girth) * (0.00149 - Q.mean_phi)
    if Q.girth < 0.0716 and Q.mean_phi > 0.00423:
        z += -2320.0 * (0.0716 - Q.girth) * (Q.mean_phi - 0.00423)
    if Q.girth2 < 0.0042 and Q.D3 < 0.802:
        z += 1290.0 * (0.0042 - Q.girth2) * (0.802 - Q.D3)
    if Q.lam1 < 0.005 and Q.dr1_7 < 0.18:
        z += -1310.0 * (0.005 - Q.lam1) * (0.18 - Q.dr1_7)
    if Q.log_sum_pt < 6.95 and Q.D2_b2 < 0.789:
        z += -1.77 * (6.95 - Q.log_sum_pt) * (0.789 - Q.D2_b2)
    if Q.log_sum_pt < 6.84 and Q.planar_flow > 0.0775:
        z += -2.2 * (6.84 - Q.log_sum_pt) * (Q.planar_flow - 0.0775)
    if Q.log_sum_pt < 6.9 and Q.z_5 < 0.0572:
        z += 180.0 * (6.9 - Q.log_sum_pt) * (0.0572 - Q.z_5)
    if Q.sj2_dr < 0.13 and Q.mean_eta < -0.00798:
        z += 707.0 * (0.13 - Q.sj2_dr) * (-0.00798 - Q.mean_eta)
    if Q.width < 0.00729 and Q.D3 < 0.826:
        z += -809.0 * (0.00729 - Q.width) * (0.826 - Q.D3)
    if Q.width < 0.00558 and Q.mean_eta > 0.0093:
        z += -31900.0 * (0.00558 - Q.width) * (Q.mean_eta - 0.0093)
    if Q.width < 0.00641 and Q.mean_eta < -0.00924:
        z += -31600.0 * (0.00641 - Q.width) * (-0.00924 - Q.mean_eta)
    return max(0.0, z)


def neuron_10(Q):
    z = 16.9
    if Q.LHA >= 0.193:
        z += -19.6 * Q.LHA + 3.7828
    if Q.centroid_offset >= 0.023:
        z += 15.8 * Q.centroid_offset - 0.3634
    if Q.e3 >= 8.06e-05:
        z += 2840.0 * Q.e3 - 0.228904
    if Q.girth < 0.122:
        z += 24.9 * Q.girth - 3.0378
    if 0.00285 <= Q.girth2 < 0.0247:
        z += -335.0 * Q.girth2 + 0.95475
    if Q.girth2 >= 0.0247:
        z += -463.0 * Q.girth2 + 4.11635
    if Q.girth2_top3 < 0.00296:
        z += -296.0 * Q.girth2_top3 + 0.87616
    if Q.lam1 < 0.00482:
        z += 961.0 * Q.lam1 - 15.5682
    if 0.00482 <= Q.lam1 < 0.0162:
        z += 507.0 * Q.lam1 - 13.37992
    if Q.lam1 >= 0.0162:
        z += 556.0 * Q.lam1 - 14.17372
    if Q.lam2 >= 0.00024:
        z += 768.0 * Q.lam2 - 0.18432
    if Q.log_sum_pt >= 6.67:
        z += -2.85 * Q.log_sum_pt + 19.0095
    if Q.n_dr_0p05_0p1 < 4.74:
        z += -0.0749 * Q.n_dr_0p05_0p1 + 0.355026
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.919 * Q.n_dr_0p2_0p4 - 0.919
    if Q.sj2_dr >= 0.299:
        z += 15.6 * Q.sj2_dr - 4.6644
    if Q.sum_pt >= 989.0:
        z += 0.00831 * Q.sum_pt - 8.21859
    if Q.tau1 >= 0.0492:
        z += 16.9 * Q.tau1 - 0.83148
    if Q.z_7 < 0.0679:
        z += 16.5 * Q.z_7 - 1.12035
    if Q.LHA > 0.311 and Q.planar_flow < 0.838:
        z += -27.2 * (Q.LHA - 0.311) * (0.838 - Q.planar_flow)
    if Q.lam2 > 0.000588 and Q.log_sum_pt < 6.54:
        z += -543.0 * (Q.lam2 - 0.000588) * (6.54 - Q.log_sum_pt)
    if Q.n_dr_0p2_0p4 > 1.02 and Q.D2_b2 < 4.62:
        z += -0.216 * (Q.n_dr_0p2_0p4 - 1.02) * (4.62 - Q.D2_b2)
    if Q.pt_6 < 48.7 and Q.D2_b2 < 0.127:
        z += 0.296 * (48.7 - Q.pt_6) * (0.127 - Q.D2_b2)
    if Q.pt_6 < 49.1 and Q.log_sum_pt < 6.48:
        z += -0.0899 * (49.1 - Q.pt_6) * (6.48 - Q.log_sum_pt)
    if Q.sj2_dr > 0.17 and Q.planar_flow < 0.676:
        z += -23.3 * (Q.sj2_dr - 0.17) * (0.676 - Q.planar_flow)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.05
    if Q.LHA >= 0.252:
        z += 7.82 * Q.LHA - 1.97064
    if Q.centroid_offset < 0.0391:
        z += -82.2 * Q.centroid_offset + 3.21402
    if Q.e2_sq < 0.00642:
        z += 478.0 * Q.e2_sq - 3.06876
    if Q.girth < 0.0345:
        z += 108.0 * Q.girth - 6.56935
    if 0.0345 <= Q.girth < 0.071:
        z += 77.9 * Q.girth - 5.5309
    if Q.lam1 < 0.00843:
        z += 937.0 * Q.lam1 - 7.89891
    if Q.log_sum_pt >= 6.73:
        z += -3.6 * Q.log_sum_pt + 24.228
    if Q.mean_phi2 < 0.00162:
        z += 300.0 * Q.mean_phi2 - 0.486
    if Q.planar_flow < 0.264:
        z += -6.61 * Q.planar_flow + 1.74504
    if 0.0899 <= Q.sj2_dr < 0.175:
        z += 8.43 * Q.sj2_dr - 0.757857
    if 0.175 <= Q.sj2_dr < 0.269:
        z += -16.27 * Q.sj2_dr + 3.564643
    if Q.sj2_dr >= 0.269:
        z += -0.17 * Q.sj2_dr - 0.766257
    if Q.width < 0.00861:
        z += -1868.0 * Q.width + 17.997
    if 0.00861 <= Q.width < 0.0133:
        z += -408.0 * Q.width + 5.4264
    if Q.centroid_offset > 0.0483 and Q.D2 < 3.12:
        z += -137.0 * (Q.centroid_offset - 0.0483) * (3.12 - Q.D2)
    if Q.centroid_offset < 0.014 and Q.D2 < 1.38:
        z += -106.0 * (0.014 - Q.centroid_offset) * (1.38 - Q.D2)
    if Q.centroid_offset > 0.0502 and Q.M3 > 0.0444:
        z += -22800.0 * (Q.centroid_offset - 0.0502) * (Q.M3 - 0.0444)
    if Q.centroid_offset < 0.037 and Q.pt_2 < 120.0:
        z += -0.847 * (0.037 - Q.centroid_offset) * (120.0 - Q.pt_2)
    if Q.centroid_offset < 0.039 and Q.pt_7 < 53.6:
        z += -1.41 * (0.039 - Q.centroid_offset) * (53.6 - Q.pt_7)
    if Q.centroid_offset < 0.0142 and Q.sj3_dr_min < 0.09:
        z += -1730.0 * (0.0142 - Q.centroid_offset) * (0.09 - Q.sj3_dr_min)
    if Q.centroid_offset < 0.0141 and Q.tau21_b2 < 0.0608:
        z += 2260.0 * (0.0141 - Q.centroid_offset) * (0.0608 - Q.tau21_b2)
    if Q.centroid_offset < 0.0144 and Q.tau21_b2 < 0.263:
        z += 398.0 * (0.0144 - Q.centroid_offset) * (0.263 - Q.tau21_b2)
    if Q.centroid_offset < 0.0384 and Q.z_3rd < 0.156:
        z += 584.0 * (0.0384 - Q.centroid_offset) * (0.156 - Q.z_3rd)
    if Q.e2_sq < 0.00346 and Q.girth2_top3 > 0.000792:
        z += -69800.0 * (0.00346 - Q.e2_sq) * (Q.girth2_top3 - 0.000792)
    if Q.e2_sq < 0.0061 and Q.pt_2 < 121.0:
        z += 2.58 * (0.0061 - Q.e2_sq) * (121.0 - Q.pt_2)
    if Q.e2_sq < 0.0061 and Q.z_3rd < 0.147:
        z += -2140.0 * (0.0061 - Q.e2_sq) * (0.147 - Q.z_3rd)
    if Q.log_sum_pt > 6.57 and Q.z_7 < 0.0594:
        z += 128.0 * (Q.log_sum_pt - 6.57) * (0.0594 - Q.z_7)
    if Q.mean_phi2 < 0.00143 and Q.M2 > 0.0154:
        z += 3510.0 * (0.00143 - Q.mean_phi2) * (Q.M2 - 0.0154)
    if Q.planar_flow < 0.239 and Q.e3 < 9.55e-06:
        z += -347000.0 * (0.239 - Q.planar_flow) * (9.55e-06 - Q.e3)
    if Q.planar_flow < 0.266 and Q.pt_7 < 36.9:
        z += -0.247 * (0.266 - Q.planar_flow) * (36.9 - Q.pt_7)
    if Q.planar_flow < 0.249 and Q.sum_pt < 862.0:
        z += -0.0173 * (0.249 - Q.planar_flow) * (862.0 - Q.sum_pt)
    if Q.width < 0.00556 and Q.sj3_dr23 > 0.146:
        z += 1680.0 * (0.00556 - Q.width) * (Q.sj3_dr23 - 0.146)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.201
    if Q.e2 >= 0.0634:
        z += -83.3 * Q.e2 + 5.28122
    if 0.0185 <= Q.girth2 < 0.0253:
        z += 349.0 * Q.girth2 - 6.4565
    if Q.girth2 >= 0.0253:
        z += 508.0 * Q.girth2 - 10.4792
    if Q.width >= 0.0163:
        z += 28.2 * Q.width - 0.45966
    if Q.girth2 > 0.0187 and Q.planar_flow < 0.798:
        z += -123.0 * (Q.girth2 - 0.0187) * (0.798 - Q.planar_flow)
    if Q.girth2 > 0.019 and Q.pt_2 < 96.3:
        z += -2.02 * (Q.girth2 - 0.019) * (96.3 - Q.pt_2)
    if Q.girth2 > 0.0189 and Q.pt_7 < 52.7:
        z += -3.46 * (Q.girth2 - 0.0189) * (52.7 - Q.pt_7)
    if Q.width > 0.0122 and Q.log_sum_pt > 6.34:
        z += 267.0 * (Q.width - 0.0122) * (Q.log_sum_pt - 6.34)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.745
    if Q.C2_b2 < 0.00934:
        z += 211.0 * Q.C2_b2 - 1.97074
    if Q.M2 < 0.0299:
        z += -30.4 * Q.M2 + 0.90896
    if Q.girth < 0.149:
        z += -72.6 * Q.girth + 10.8174
    if Q.log_sum_pt < 6.48:
        z += 3.15 * Q.log_sum_pt - 20.412
    if Q.pt_5 < 25.3:
        z += 0.194 * Q.pt_5 - 4.9082
    if Q.pt_6 < 27.7:
        z += 0.1287 * Q.pt_6 - 4.08255
    if 27.7 <= Q.pt_6 < 50.5:
        z += 0.0227 * Q.pt_6 - 1.14635
    if Q.pt_7 >= 31.8:
        z += -0.0607 * Q.pt_7 + 1.93026
    if Q.sj3_dr_max >= 0.232:
        z += -7.49 * Q.sj3_dr_max + 1.73768
    if Q.sum_pt_top5 >= 492.0:
        z += -0.0104 * Q.sum_pt_top5 + 5.1168
    if Q.tau1 < 0.116:
        z += 34.4 * Q.tau1 - 3.9904
    if Q.width < 0.0137:
        z += -268.0 * Q.width + 3.6716
    if Q.z_7 >= 0.0465:
        z += 51.4 * Q.z_7 - 2.3901
    if Q.e2 < 0.0822 and Q.pt_4 < 59.8:
        z += 0.276 * (0.0822 - Q.e2) * (59.8 - Q.pt_4)
    if Q.e3 < 9.3e-05 and Q.centroid_offset < 0.0387:
        z += -558000.0 * (9.3e-05 - Q.e3) * (0.0387 - Q.centroid_offset)
    if Q.girth < 0.149 and Q.lam2 < 0.000283:
        z += -11400.0 * (0.149 - Q.girth) * (0.000283 - Q.lam2)
    if Q.girth < 0.145 and Q.log_sum_pt < 6.8:
        z += -58.8 * (0.145 - Q.girth) * (6.8 - Q.log_sum_pt)
    if Q.girth < 0.15 and Q.pt_7 < 38.3:
        z += -0.691 * (0.15 - Q.girth) * (38.3 - Q.pt_7)
    if Q.lam1 < 0.0175 and Q.pt_7 < 25.5:
        z += -10.3 * (0.0175 - Q.lam1) * (25.5 - Q.pt_7)
    if Q.sj3_dr_max > 0.225 and Q.tau32 < 0.418:
        z += -18.2 * (Q.sj3_dr_max - 0.225) * (0.418 - Q.tau32)
    if Q.sum_pt_top5 > 494.0 and Q.e4 < 3.04e-08:
        z += 373000.0 * (Q.sum_pt_top5 - 494.0) * (3.04e-08 - Q.e4)
    if Q.sum_pt_top5 > 687.0 and Q.pt_7 < 38.7:
        z += 0.00071 * (Q.sum_pt_top5 - 687.0) * (38.7 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.883
    if Q.C2 < 0.0356:
        z += 13.9 * Q.C2 - 0.49484
    if Q.LHA >= 0.304:
        z += 18.9 * Q.LHA - 5.7456
    if 0.0234 <= Q.centroid_offset < 0.0376:
        z += -46.1 * Q.centroid_offset + 1.07874
    if 0.0376 <= Q.centroid_offset < 0.05:
        z += -293.1 * Q.centroid_offset + 10.36594
    if Q.centroid_offset >= 0.05:
        z += -737.1 * Q.centroid_offset + 32.56594
    if 0.00715 <= Q.e2_sq < 0.00815:
        z += 902.0 * Q.e2_sq - 6.4493
    if 0.00815 <= Q.e2_sq < 0.0117:
        z += 1436.0 * Q.e2_sq - 10.8014
    if Q.e2_sq >= 0.0117:
        z += -734.0 * Q.e2_sq + 14.5876
    if Q.e3 < 1.95e-05:
        z += -24000.0 * Q.e3 + 0.468
    if 0.0263 <= Q.girth < 0.0411:
        z += 40.8 * Q.girth - 1.07304
    if 0.0411 <= Q.girth < 0.0873:
        z += 157.8 * Q.girth - 5.88174
    if 0.0873 <= Q.girth < 0.102:
        z += -48.2 * Q.girth + 12.10206
    if Q.girth >= 0.102:
        z += -407.2 * Q.girth + 48.72006
    if 0.00671 <= Q.girth2 < 0.00748:
        z += -755.0 * Q.girth2 + 5.06605
    if Q.girth2 >= 0.00748:
        z += -1636.0 * Q.girth2 + 11.65593
    if Q.n_dr_0_0p05 >= 5.02:
        z += 0.388 * Q.n_dr_0_0p05 - 1.94776
    if Q.n_dr_0p05_0p1 < 4.96:
        z += 0.272 * Q.n_dr_0p05_0p1 - 1.34912
    if Q.n_dr_0p1_0p2 >= 1.09:
        z += 0.284 * Q.n_dr_0p1_0p2 - 0.30956
    if Q.psi_0p1 >= 0.975:
        z += -17.4 * Q.psi_0p1 + 16.965
    if Q.sj2_dr < 0.159:
        z += -9.91 * Q.sj2_dr + 1.57569
    if Q.sj3_dr_max < 0.233:
        z += 27.0 * Q.sj3_dr_max - 6.291
    if 0.0345 <= Q.tau1 < 0.0969:
        z += -53.2 * Q.tau1 + 1.8354
    if Q.tau1 >= 0.0969:
        z += -34.0 * Q.tau1 - 0.02508
    if 0.00556 <= Q.width < 0.0129:
        z += 292.0 * Q.width - 1.62352
    if Q.width >= 0.0129:
        z += 35.0 * Q.width + 1.69178
    if Q.z_dr_0p05_0p1 < 0.572:
        z += -1.17 * Q.z_dr_0p05_0p1 + 0.66924
    if Q.z_dr_0p1_0p2 < 0.154:
        z += -2.11 * Q.z_dr_0p1_0p2 + 0.32494
    if Q.centroid_offset > 0.0491 and Q.eta_5 > 0.107:
        z += -8520.0 * (Q.centroid_offset - 0.0491) * (Q.eta_5 - 0.107)
    if Q.e2_sq > 0.00217 and Q.log_sum_pt > 6.26:
        z += 399.0 * (Q.e2_sq - 0.00217) * (Q.log_sum_pt - 6.26)
    if Q.e2_sq > 0.00232 and Q.sj2_dr < 0.188:
        z += -6300.0 * (Q.e2_sq - 0.00232) * (0.188 - Q.sj2_dr)
    if Q.e2_sq > 0.00817 and Q.sum_pt > 493.0:
        z += 3.48 * (Q.e2_sq - 0.00817) * (Q.sum_pt - 493.0)
    if Q.girth2 > 0.00868 and Q.log_sum_pt > 6.27:
        z += -9300.0 * (Q.girth2 - 0.00868) * (Q.log_sum_pt - 6.27)
    if Q.planar_flow < 0.109 and Q.centroid_offset < 0.0185:
        z += -397.0 * (0.109 - Q.planar_flow) * (0.0185 - Q.centroid_offset)
    if Q.planar_flow < 0.112 and Q.lam1 < 0.00725:
        z += -3050.0 * (0.112 - Q.planar_flow) * (0.00725 - Q.lam1)
    if Q.planar_flow < 0.111 and Q.z_dr_0p05_0p1 > 0.676:
        z += -34.3 * (0.111 - Q.planar_flow) * (Q.z_dr_0p05_0p1 - 0.676)
    if Q.psi_0p1 > 0.809 and Q.centroid_offset > 0.0375:
        z += 952.0 * (Q.psi_0p1 - 0.809) * (Q.centroid_offset - 0.0375)
    if Q.psi_0p1 > 0.635 and Q.eccentricity > 0.971:
        z += 159.0 * (Q.psi_0p1 - 0.635) * (Q.eccentricity - 0.971)
    if Q.width > 0.0133 and Q.sum_pt > 488.0:
        z += -0.104 * (Q.width - 0.0133) * (Q.sum_pt - 488.0)
    if Q.z_dr_0p05_0p1 > 0.751 and Q.dr0_5 > 0.29:
        z += -8030.0 * (Q.z_dr_0p05_0p1 - 0.751) * (Q.dr0_5 - 0.29)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.06
    if Q.N2 < 0.223:
        z += -15.3 * Q.N2 + 3.4119
    if Q.e2_sq < 0.00642:
        z += -408.0 * Q.e2_sq + 2.61936
    if Q.girth2 < 0.00749:
        z += 1129.0 * Q.girth2 - 6.8004
    if 0.00749 <= Q.girth2 < 0.0136:
        z += -271.0 * Q.girth2 + 3.6856
    if Q.girth2_top3 < 0.00216:
        z += 425.0 * Q.girth2_top3 - 0.918
    if Q.max_dr >= 0.12:
        z += -12.6 * Q.max_dr + 1.512
    if 0.185 <= Q.sj3_dr_max < 0.268:
        z += 26.8 * Q.sj3_dr_max - 4.958
    if Q.sj3_dr_max >= 0.268:
        z += 5.9 * Q.sj3_dr_max + 0.6432
    if Q.tau1 < 0.0822:
        z += -69.3 * Q.tau1 + 5.35728
    if 0.0822 <= Q.tau1 < 0.102:
        z += -19.9 * Q.tau1 + 1.2966
    if 0.102 <= Q.tau1 < 0.114:
        z += 61.1 * Q.tau1 - 6.9654
    if Q.width < 0.00268:
        z += 878.0 * Q.width - 2.35304
    if Q.z_dr_0p05_0p1 >= 0.752:
        z += -11.4 * Q.z_dr_0p05_0p1 + 8.5728
    if Q.N2 < 0.22 and Q.LHA > 0.409:
        z += -3200.0 * (0.22 - Q.N2) * (Q.LHA - 0.409)
    if Q.N2 < 0.223 and Q.LHA > 0.292:
        z += 125.0 * (0.223 - Q.N2) * (Q.LHA - 0.292)
    if Q.N2 < 0.233 and Q.LHA > 0.327:
        z += 408.0 * (0.233 - Q.N2) * (Q.LHA - 0.327)
    if Q.N2 < 0.22 and Q.e3 < 8.24e-05:
        z += -143000.0 * (0.22 - Q.N2) * (8.24e-05 - Q.e3)
    if Q.N2 < 0.226 and Q.girth2 > 0.00886:
        z += -6880.0 * (0.226 - Q.N2) * (Q.girth2 - 0.00886)
    if Q.N2 < 0.219 and Q.pt_7 < 25.7:
        z += -1.02 * (0.219 - Q.N2) * (25.7 - Q.pt_7)
    if Q.N2 < 0.221 and Q.sj2_dr < 0.187:
        z += -315.0 * (0.221 - Q.N2) * (0.187 - Q.sj2_dr)
    if Q.N2 < 0.232 and Q.sum_pt_top5 > 408.0:
        z += 0.0224 * (0.232 - Q.N2) * (Q.sum_pt_top5 - 408.0)
    if Q.N2 < 0.227 and Q.z_dr_0p05_0p1 < 0.647:
        z += -15.4 * (0.227 - Q.N2) * (0.647 - Q.z_dr_0p05_0p1)
    if Q.girth2 < 0.0153 and Q.centroid_offset > 0.0501:
        z += -14300.0 * (0.0153 - Q.girth2) * (Q.centroid_offset - 0.0501)
    if Q.girth2 < 0.00726 and Q.dr12 > 0.171:
        z += -5030.0 * (0.00726 - Q.girth2) * (Q.dr12 - 0.171)
    if Q.girth2 < 0.00642 and Q.pt_entropy > 1.81:
        z += -772.0 * (0.00642 - Q.girth2) * (Q.pt_entropy - 1.81)
    if Q.girth2_top3 < 0.00207 and Q.eccentricity > 0.957:
        z += -22800.0 * (0.00207 - Q.girth2_top3) * (Q.eccentricity - 0.957)
    if Q.planar_flow < 0.207 and Q.mean_phi > -0.0197:
        z += -102.0 * (0.207 - Q.planar_flow) * (Q.mean_phi - -0.0197)
    if Q.planar_flow < 0.198 and Q.sum_pt_top5 > 414.0:
        z += 0.0362 * (0.198 - Q.planar_flow) * (Q.sum_pt_top5 - 414.0)
    if Q.planar_flow < 0.199 and Q.z_7 < 0.0715:
        z += -167.0 * (0.199 - Q.planar_flow) * (0.0715 - Q.z_7)
    if Q.z_dr_0p05_0p1 > 0.743 and Q.z_dr_0p2_0p4 < 0.0548:
        z += 213.0 * (Q.z_dr_0p05_0p1 - 0.743) * (0.0548 - Q.z_dr_0p2_0p4)
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
