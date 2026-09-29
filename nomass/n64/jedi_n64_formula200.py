"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; no mass observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 91.8% of jets.

Quantities:
  Q.z_top5                 pT share of the 5 largest
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.n_particles            number of real particles (pT > 0)
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft1_z                pT share of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.pt1_dr01               pT1 · ΔR01
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.soft7_dr0              ΔR between the hardest and the 7. softest real particle (0 if among the 15 hardest)
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_5           number of particles with pT > 5 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi               pT-weighted mean Δφ
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.tau4                   N-subjettiness τ4 (β=1)
  Q.tau43                  N-subjettiness τ4/τ3
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
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        e3=ecf('e3'),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        n_particles=len(real),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        soft3_pt=softp(3, 'pt'),
        soft1_z=softp(1, 'z'),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sj2_dr=subjets(2)["dr"][0],
        soft7_dr0=softp(7, 'dr0'),
        eta_1=eta[1],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_5=sum(1 for x in pt if x > 5),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    z = -1.47
    if Q.e2_sq < 0.00618:
        z += -806.0 * Q.e2_sq + 7.33933
    if 0.00618 <= Q.e2_sq < 0.00693:
        z += -1637.0 * Q.e2_sq + 12.47491
    if 0.00693 <= Q.e2_sq < 0.00788:
        z += -1190.0 * Q.e2_sq + 9.3772
    if Q.girth2_top40 < 0.00313:
        z += 267.0 * Q.girth2_top40 - 0.83571
    if Q.psi_0p3 >= 0.994:
        z += 91.6 * Q.psi_0p3 - 91.0504
    if 1000.0 <= Q.sum_pt < 1120.0:
        z += -0.0131 * Q.sum_pt + 13.1
    if Q.sum_pt >= 1120.0:
        z += -0.0071 * Q.sum_pt + 6.38
    if 862.0 <= Q.sum_pt_top40 < 1020.0:
        z += 0.00603 * Q.sum_pt_top40 - 5.19786
    if Q.sum_pt_top40 >= 1020.0:
        z += -0.00087 * Q.sum_pt_top40 + 1.84014
    if Q.tau1 < 0.0712:
        z += 14.9 * Q.tau1 - 1.06088
    if Q.tau21_b2 < 0.299:
        z += -2.17 * Q.tau21_b2 + 0.64883
    if Q.girth2 < 0.00812 and Q.girth2_top15 < 0.00585:
        z += -29000.0 * (0.00812 - Q.girth2) * (0.00585 - Q.girth2_top15)
    if Q.girth2_top50 < 0.00588 and Q.sum_pt < 1260.0:
        z += -2.11 * (0.00588 - Q.girth2_top50) * (1260.0 - Q.sum_pt)
    if Q.sum_pt > 1170.0 and Q.soft7_dr0 > 0.256:
        z += -0.0791 * (Q.sum_pt - 1170.0) * (Q.soft7_dr0 - 0.256)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.98
    if Q.LHA < 0.247:
        z += -8.32 * Q.LHA + 2.05504
    if Q.girth2_top20 < 0.00119:
        z += -824.0 * Q.girth2_top20 + 0.98056
    if Q.girth2_top30 < 0.00706:
        z += -126.0 * Q.girth2_top30 + 0.88956
    if Q.lam2 < 0.00178:
        z += 414.0 * Q.lam2 - 0.73692
    if Q.log_sum_pt < 6.91:
        z += 6.73 * Q.log_sum_pt - 48.1195
    if 6.91 <= Q.log_sum_pt < 6.99:
        z += 44.03 * Q.log_sum_pt - 305.8625
    if 6.99 <= Q.log_sum_pt < 7.15:
        z += 10.33 * Q.log_sum_pt - 70.2995
    if Q.log_sum_pt >= 7.15:
        z += 3.6 * Q.log_sum_pt - 22.18
    if Q.n_dr_0_0p05 >= 14.1:
        z += 0.0348 * Q.n_dr_0_0p05 - 0.49068
    if Q.n_dr_0p1_0p2 < 6.99:
        z += 0.119 * Q.n_dr_0p1_0p2 - 0.83181
    if Q.n_dr_0p2_0p4 < 7.42:
        z += 0.116 * Q.n_dr_0p2_0p4 - 0.86072
    if Q.n_particles >= 39.4:
        z += 0.0428 * Q.n_particles - 1.68632
    if Q.sum_pt >= 973.0:
        z += 0.0186 * Q.sum_pt - 18.0978
    if Q.sum_pt_top50 < 932.0:
        z += -0.0134 * Q.sum_pt_top50 + 14.472
    if 932.0 <= Q.sum_pt_top50 < 1080.0:
        z += -0.0312 * Q.sum_pt_top50 + 31.0616
    if Q.sum_pt_top50 >= 1080.0:
        z += -0.0178 * Q.sum_pt_top50 + 16.5896
    if Q.tau21 < 0.207:
        z += -10.1 * Q.tau21 + 2.0907
    if Q.z_top5 >= 0.487:
        z += -4.96 * Q.z_top5 + 2.41552
    if Q.zdr_0 >= 0.0126:
        z += -42.0 * Q.zdr_0 + 0.5292
    if Q.M3 < 0.0364 and Q.psi_0p3 > 0.943:
        z += -509.0 * (0.0364 - Q.M3) * (Q.psi_0p3 - 0.943)
    if Q.n_particles > 37.3 and Q.soft1_z < 0.00219:
        z += -25.5 * (Q.n_particles - 37.3) * (0.00219 - Q.soft1_z)
    if Q.n_particles > 38.1 and Q.tau32 > 0.316:
        z += 0.113 * (Q.n_particles - 38.1) * (Q.tau32 - 0.316)
    if Q.n_particles > 37.5 and Q.tau43 < 0.959:
        z += -0.148 * (Q.n_particles - 37.5) * (0.959 - Q.tau43)
    if Q.n_particles > 35.4 and Q.zdr_0 < 0.01:
        z += 3.76 * (Q.n_particles - 35.4) * (0.01 - Q.zdr_0)
    if Q.z_dr_0_0p05 > 0.827 and Q.n_dr_0p05_0p1 < 10.7:
        z += -0.886 * (Q.z_dr_0_0p05 - 0.827) * (10.7 - Q.n_dr_0p05_0p1)
    if Q.z_top30_slots > 0.958 and Q.pt1_dr01 > -4.86:
        z += 1.13 * (Q.z_top30_slots - 0.958) * (Q.pt1_dr01 - -4.86)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.753
    if Q.girth2 < 0.00386:
        z += 328.0 * Q.girth2 - 1.26608
    if Q.girth2_top15 < 0.0192:
        z += -61.8 * Q.girth2_top15 + 1.18656
    if Q.log_sum_pt >= 6.92:
        z += 5.85 * Q.log_sum_pt - 40.482
    if Q.sum_pt_top15 > 1010.0 and Q.eta_1 > 0.0872:
        z += -4.78 * (Q.sum_pt_top15 - 1010.0) * (Q.eta_1 - 0.0872)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.0471
    if Q.LHA < 0.378:
        z += 4.99 * Q.LHA - 1.88622
    if Q.e2_sq < 0.00921:
        z += -185.0 * Q.e2_sq + 1.70385
    if Q.lam2 < 0.000631:
        z += -1340.0 * Q.lam2 + 0.84554
    if Q.psi_0p2 >= 0.998:
        z += 383.0 * Q.psi_0p2 - 382.234
    if Q.tau21 < 0.348:
        z += 8.46 * Q.tau21 - 2.94408
    if Q.tau21_b2 < 0.203:
        z += -8.44 * Q.tau21_b2 + 1.71332
    if Q.n_particles < 47.9 and Q.sum_pt_top40 > 822.0:
        z += 0.000134 * (47.9 - Q.n_particles) * (Q.sum_pt_top40 - 822.0)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.25
    if Q.e2_sq < 0.00615:
        z += 405.0 * Q.e2_sq - 2.49075
    if Q.girth2_top20 < 0.00742:
        z += 163.0 * Q.girth2_top20 - 0.70384
    if 0.00742 <= Q.girth2_top20 < 0.0106:
        z += -159.0 * Q.girth2_top20 + 1.6854
    if Q.girth2_top50 >= 0.00443:
        z += 47.5 * Q.girth2_top50 - 0.210425
    if Q.n_dr_0p1_0p2 < 25.8:
        z += -0.0227 * Q.n_dr_0p1_0p2 + 0.58566
    if Q.n_dr_0p2_0p4 < 23.4:
        z += -0.0226 * Q.n_dr_0p2_0p4 + 0.52884
    if Q.n_particles >= 28.9:
        z += -0.0218 * Q.n_particles + 0.63002
    if Q.sum_pt_top40 < 1050.0:
        z += -0.0118 * Q.sum_pt_top40 + 12.39
    if Q.sum_pt_top50 < 1060.0:
        z += 0.0144 * Q.sum_pt_top50 - 15.264
    if Q.tau2 < 0.103:
        z += 8.29 * Q.tau2 - 0.85387
    if Q.z_top50_slots < 0.984:
        z += 21.4 * Q.z_top50_slots - 21.0576
    if Q.psi_0p3 > 0.997 and Q.sum_pt_top40 > 847.0:
        z += 0.802 * (Q.psi_0p3 - 0.997) * (Q.sum_pt_top40 - 847.0)
    if Q.sj2_dr > 0.261 and Q.C2_b2 < 0.0333:
        z += -574.0 * (Q.sj2_dr - 0.261) * (0.0333 - Q.C2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.483
    if Q.M3 < 0.0312:
        z += -27.1 * Q.M3 + 0.84552
    if Q.e2_sq >= 0.00946:
        z += -57.4 * Q.e2_sq + 0.543004
    if Q.girth2_top15 < 0.000452:
        z += 1930.0 * Q.girth2_top15 - 0.87236
    if 6.92 <= Q.log_sum_pt < 7.01:
        z += -33.0 * Q.log_sum_pt + 228.36
    if Q.log_sum_pt >= 7.01:
        z += -20.6 * Q.log_sum_pt + 141.436
    if 934.0 <= Q.sum_pt_top50 < 1050.0:
        z += 0.0256 * Q.sum_pt_top50 - 23.9104
    if Q.sum_pt_top50 >= 1050.0:
        z += 0.01714 * Q.sum_pt_top50 - 15.0274
    if Q.tau4 < 0.0156:
        z += -166.0 * Q.tau4 + 2.5896
    if Q.n_particles < 69.5 and Q.D2 < 2.43:
        z += -0.0139 * (69.5 - Q.n_particles) * (2.43 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.13
    if Q.LHA >= 0.207:
        z += 5.65 * Q.LHA - 1.16955
    if Q.e2 >= 0.0143:
        z += -68.0 * Q.e2 + 0.9724
    if Q.lam1 < 0.00382:
        z += -1542.0 * Q.lam1 + 7.46628
    if 0.00382 <= Q.lam1 < 0.00774:
        z += -402.0 * Q.lam1 + 3.11148
    if Q.lam2 < 0.00104:
        z += -559.0 * Q.lam2 + 1.15713
    if 0.00104 <= Q.lam2 < 0.00207:
        z += -750.0 * Q.lam2 + 1.35577
    if Q.lam2 >= 0.00207:
        z += -191.0 * Q.lam2 + 0.19864
    if Q.log_sum_pt >= 6.81:
        z += 3.22 * Q.log_sum_pt - 21.9282
    if Q.psi_0p3 >= 0.985:
        z += 89.3 * Q.psi_0p3 - 87.9605
    if Q.width < 0.00988:
        z += 804.0 * Q.width - 7.94352
    if Q.e2 > 0.0493 and Q.z_dr_0_0p05 < 0.255:
        z += 399.0 * (Q.e2 - 0.0493) * (0.255 - Q.z_dr_0_0p05)
    if Q.e2_sq < 0.0252 and Q.log_sum_pt < 6.98:
        z += -383.0 * (0.0252 - Q.e2_sq) * (6.98 - Q.log_sum_pt)
    if Q.e2_sq < 0.00961 and Q.sum_pt < 1010.0:
        z += 3.04 * (0.00961 - Q.e2_sq) * (1010.0 - Q.sum_pt)
    if Q.girth2 < 0.0136 and Q.psi_0p3 > 0.985:
        z += -9620.0 * (0.0136 - Q.girth2) * (Q.psi_0p3 - 0.985)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.709
    if Q.e2 < 0.0343:
        z += -49.7 * Q.e2 + 1.70471
    if Q.e2_sq < 0.00967:
        z += -948.0 * Q.e2_sq + 9.16716
    if Q.girth2 < 0.0078:
        z += 1310.0 * Q.girth2 - 10.218
    if Q.girth2_top50 < 0.00652:
        z += 1210.0 * Q.girth2_top50 - 7.8892
    if Q.n_dr_0p2_0p4 < 15.1:
        z += -0.0438 * Q.n_dr_0p2_0p4 + 0.66138
    if Q.psi_0p3 >= 0.996:
        z += 220.0 * Q.psi_0p3 - 219.12
    if Q.tau2 < 0.0474:
        z += 46.9 * Q.tau2 - 2.22306
    if Q.width < 0.0138:
        z += -245.0 * Q.width + 3.381
    if Q.e2_sq < 0.00966 and Q.sum_pt < 1170.0:
        z += -6.58 * (0.00966 - Q.e2_sq) * (1170.0 - Q.sum_pt)
    if Q.lam2 < 0.0024 and Q.log_sum_pt < 7.14:
        z += 1220.0 * (0.0024 - Q.lam2) * (7.14 - Q.log_sum_pt)
    if Q.sum_pt_top15 > 1080.0 and Q.mean_phi > 1.99e-05:
        z += -67.2 * (Q.sum_pt_top15 - 1080.0) * (Q.mean_phi - 1.99e-05)
    if Q.tau21_b2 < 0.23 and Q.e2_sq < 0.00822:
        z += -4930.0 * (0.23 - Q.tau21_b2) * (0.00822 - Q.e2_sq)
    if Q.tau21_b2 < 0.24 and Q.e2_sq < 0.0146:
        z += 1580.0 * (0.24 - Q.tau21_b2) * (0.0146 - Q.e2_sq)
    if Q.tau21_b2 < 0.224 and Q.sum_pt_top30 > 983.0:
        z += 0.0405 * (0.224 - Q.tau21_b2) * (Q.sum_pt_top30 - 983.0)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.6
    if Q.e2_sq < 0.00979:
        z += 699.0 * Q.e2_sq - 6.84321
    if Q.girth2 < 0.00359:
        z += -1520.0 * Q.girth2 + 11.7036
    if 0.00359 <= Q.girth2 < 0.0259:
        z += -280.0 * Q.girth2 + 7.252
    if Q.girth2_top30 < 0.00729:
        z += 381.0 * Q.girth2_top30 - 2.77749
    if Q.lam2 < 0.00115:
        z += 421.0 * Q.lam2 - 0.48415
    if Q.n_dr_0p2_0p4 < 18.4:
        z += 0.0426 * Q.n_dr_0p2_0p4 - 0.78384
    if Q.psi_0p3 >= 0.994:
        z += -87.1 * Q.psi_0p3 + 86.5774
    if Q.sum_pt < 1070.0:
        z += -0.0163 * Q.sum_pt + 17.441
    if Q.sum_pt_top50 < 1240.0:
        z += -0.00716 * Q.sum_pt_top50 + 8.8784
    if Q.tau1 < 0.122:
        z += -27.3 * Q.tau1 + 3.3306
    if Q.width < 0.00821:
        z += 419.0 * Q.width - 3.43999
    if Q.girth2 < 0.0254 and Q.log_sum_pt < 7.15:
        z += -1650.0 * (0.0254 - Q.girth2) * (7.15 - Q.log_sum_pt)
    if Q.girth2 < 0.0153 and Q.sum_pt < 988.0:
        z += 2.03 * (0.0153 - Q.girth2) * (988.0 - Q.sum_pt)
    if Q.girth2 < 0.0254 and Q.z_top50_slots > 0.973:
        z += -1290.0 * (0.0254 - Q.girth2) * (Q.z_top50_slots - 0.973)
    if Q.girth2_top30 < 0.00758 and Q.sum_pt < 1250.0:
        z += 3.68 * (0.00758 - Q.girth2_top30) * (1250.0 - Q.sum_pt)
    if Q.girth2_top30 < 0.00748 and Q.sum_pt_top40 < 1060.0:
        z += -2.41 * (0.00748 - Q.girth2_top30) * (1060.0 - Q.sum_pt_top40)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.154
    if Q.e3 >= 0.000531:
        z += -9950.0 * Q.e3 + 5.28345
    if Q.girth2_top15 >= 0.0217:
        z += -321.0 * Q.girth2_top15 + 6.9657
    if Q.girth2_top20 < 0.0118:
        z += 153.0 * Q.girth2_top20 - 1.8054
    if Q.girth2_top40 < 0.00565:
        z += -433.0 * Q.girth2_top40 + 2.44645
    if Q.pt_entropy < 2.51:
        z += -0.678 * Q.pt_entropy + 1.70178
    if Q.sum_pt < 985.0:
        z += -0.0262 * Q.sum_pt + 25.807
    if Q.sum_pt_top30 < 1190.0:
        z += -0.00507 * Q.sum_pt_top30 + 6.0333
    if Q.tau1 < 0.0602:
        z += 32.1 * Q.tau1 - 1.93242
    if Q.width < 0.00686:
        z += -620.0 * Q.width + 4.2532
    if Q.e3 > 0.000515 and Q.soft3_pt > 2.1:
        z += -123000.0 * (Q.e3 - 0.000515) * (Q.soft3_pt - 2.1)
    if Q.girth > 0.114 and Q.sj3_pairmin_over_m < 0.381:
        z += 159.0 * (Q.girth - 0.114) * (0.381 - Q.sj3_pairmin_over_m)
    if Q.lam1 < 0.012 and Q.sum_pt > 1040.0:
        z += -0.25 * (0.012 - Q.lam1) * (Q.sum_pt - 1040.0)
    if Q.sum_pt_top40 < 973.0 and Q.tau3 < 0.0368:
        z += -1.07 * (973.0 - Q.sum_pt_top40) * (0.0368 - Q.tau3)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.98
    if Q.LHA < 0.368:
        z += -12.9 * Q.LHA + 4.7472
    if Q.M2 < 0.082:
        z += -17.8 * Q.M2 + 1.4596
    if Q.e2 >= 0.0076:
        z += 31.1 * Q.e2 - 0.23636
    if Q.e2_sq < 0.00608:
        z += -425.0 * Q.e2_sq + 2.584
    if Q.e3 < 0.00051:
        z += 3260.0 * Q.e3 - 1.6626
    if Q.girth < 0.0357:
        z += 137.7 * Q.girth - 9.59886
    if 0.0357 <= Q.girth < 0.121:
        z += 54.9 * Q.girth - 6.6429
    if 0.014 <= Q.girth2 < 0.0289:
        z += -124.0 * Q.girth2 + 1.736
    if Q.girth2 >= 0.0289:
        z += -286.0 * Q.girth2 + 6.4178
    if Q.n_dr_0_0p05 >= 5.86:
        z += 0.032 * Q.n_dr_0_0p05 - 0.18752
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.0587 * Q.n_dr_0p2_0p4 - 0.8805
    if Q.psi_0p3 >= 0.999:
        z += -544.0 * Q.psi_0p3 + 543.456
    if Q.sum_pt_top3 < 696.0:
        z += -0.00348 * Q.sum_pt_top3 + 2.42208
    if Q.tau21_b2 < 0.307:
        z += 3.82 * Q.tau21_b2 - 1.17274
    if Q.tau4 < 0.0259:
        z += -76.5 * Q.tau4 + 1.98135
    if Q.z_dr_0_0p05 >= 0.709:
        z += -5.25 * Q.z_dr_0_0p05 + 3.72225
    if Q.z_dr_0p2_0p4 < 0.0893:
        z += -14.2 * Q.z_dr_0p2_0p4 + 1.26806
    if Q.e2 > 0.00978 and Q.log_sum_pt > 6.89:
        z += -175.0 * (Q.e2 - 0.00978) * (Q.log_sum_pt - 6.89)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.413
    if Q.C2_b2 >= 0.00676:
        z += -29.0 * Q.C2_b2 + 0.19604
    if Q.e2_sq < 0.00476:
        z += 120.0 * Q.e2_sq + 3.2438
    if 0.00476 <= Q.e2_sq < 0.00826:
        z += -1090.0 * Q.e2_sq + 9.0034
    if Q.girth2_top50 < 0.00632:
        z += 789.0 * Q.girth2_top50 - 4.98648
    if Q.n_dr_0p2_0p4 < 10.4:
        z += -0.0801 * Q.n_dr_0p2_0p4 + 0.83304
    if Q.psi_0p2 >= 0.993:
        z += 142.0 * Q.psi_0p2 - 141.006
    if Q.e2_sq < 0.0081 and Q.sum_pt < 1120.0:
        z += -8.92 * (0.0081 - Q.e2_sq) * (1120.0 - Q.sum_pt)
    if Q.e2_sq < 0.00944 and Q.sum_pt < 1110.0:
        z += 3.27 * (0.00944 - Q.e2_sq) * (1110.0 - Q.sum_pt)
    if Q.lam1 < 0.00606 and Q.D2_b2 < 3.37:
        z += -186.0 * (0.00606 - Q.lam1) * (3.37 - Q.D2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.0788
    if Q.LHA >= 0.382:
        z += 11.4 * Q.LHA - 4.3548
    if Q.e2_sq < 0.00747:
        z += -562.0 * Q.e2_sq + 4.19814
    if Q.girth2_top30 < 0.00552:
        z += 306.0 * Q.girth2_top30 - 1.68912
    if Q.lam1 < 0.00273:
        z += 785.0 * Q.lam1 - 2.14305
    if Q.psi_0p3 >= 0.994:
        z += -122.0 * Q.psi_0p3 + 121.268
    if Q.sum_pt >= 1080.0:
        z += -0.0102 * Q.sum_pt + 11.016
    if Q.z_dr_0_0p05 >= 0.893:
        z += -9.74 * Q.z_dr_0_0p05 + 8.69782
    if Q.e2_sq < 0.00757 and Q.log_sum_pt > 6.81:
        z += -4890.0 * (0.00757 - Q.e2_sq) * (Q.log_sum_pt - 6.81)
    if Q.girth2 < 0.0064 and Q.log_sum_pt > 6.82:
        z += 7850.0 * (0.0064 - Q.girth2) * (Q.log_sum_pt - 6.82)
    if Q.psi_0p1 < 0.0886 and Q.sum_pt_top50 > 965.0:
        z += 0.125 * (0.0886 - Q.psi_0p1) * (Q.sum_pt_top50 - 965.0)
    if Q.psi_0p3 > 0.994 and Q.girth2 < 0.00759:
        z += 59000.0 * (Q.psi_0p3 - 0.994) * (0.00759 - Q.girth2)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.882
    if Q.e2 >= 0.038:
        z += 48.3 * Q.e2 - 1.8354
    if 0.00894 <= Q.girth2_top40 < 0.028:
        z += -218.0 * Q.girth2_top40 + 1.94892
    if Q.girth2_top40 >= 0.028:
        z += -471.0 * Q.girth2_top40 + 9.03292
    if 0.00656 <= Q.girth2_top50 < 0.0134:
        z += 133.0 * Q.girth2_top50 - 0.87248
    if Q.girth2_top50 >= 0.0134:
        z += -208.0 * Q.girth2_top50 + 3.69692
    if Q.log_sum_pt >= 6.81:
        z += 16.9 * Q.log_sum_pt - 115.089
    if Q.n_pt_above_5 < 25.2:
        z += 0.0829 * Q.n_pt_above_5 - 2.08908
    if Q.sum_pt < 1020.0:
        z += 0.0196 * Q.sum_pt - 21.364
    if 1020.0 <= Q.sum_pt < 1090.0:
        z += 0.0006 * Q.sum_pt - 1.984
    if Q.sum_pt >= 1090.0:
        z += -0.019 * Q.sum_pt + 19.38
    if Q.sum_pt_top50 < 1070.0:
        z += -0.0111 * Q.sum_pt_top50 + 11.877
    if Q.width >= 0.02:
        z += -244.0 * Q.width + 4.88
    if Q.girth < 0.122 and Q.C2_b2 < 0.0128:
        z += -613.0 * (0.122 - Q.girth) * (0.0128 - Q.C2_b2)
    if Q.girth < 0.124 and Q.sum_pt_top40 < 854.0:
        z += -0.338 * (0.124 - Q.girth) * (854.0 - Q.sum_pt_top40)
    if Q.girth2_top40 > 0.0126 and Q.sum_pt_top50 < 1220.0:
        z += 1.23 * (Q.girth2_top40 - 0.0126) * (1220.0 - Q.sum_pt_top50)
    if Q.sum_pt < 1090.0 and Q.tau21 > 0.099:
        z += 0.0112 * (1090.0 - Q.sum_pt) * (Q.tau21 - 0.099)
    if Q.sum_pt_top10 > 873.0 and Q.C2_b2 < 0.0069:
        z += 4.09 * (Q.sum_pt_top10 - 873.0) * (0.0069 - Q.C2_b2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.876
    if Q.LHA >= 0.326:
        z += -9.53 * Q.LHA + 3.10678
    if Q.M3 < 0.0353:
        z += -23.9 * Q.M3 + 0.84367
    if Q.e2_sq < 0.00736:
        z += 1290.0 * Q.e2_sq - 9.4944
    if Q.n_dr_0p1_0p2 < 17.5:
        z += 0.0341 * Q.n_dr_0p1_0p2 - 0.59675
    if Q.psi_0p3 >= 0.995:
        z += 623.0 * Q.psi_0p3 - 619.885
    if Q.tau1 < 0.0564:
        z += 208.0 * Q.tau1 - 11.7312
    if Q.tau21_b2 < 0.323:
        z += -3.99 * Q.tau21_b2 + 1.28877
    if Q.girth2_top30 < 0.014 and Q.sum_pt > 904.0:
        z += 0.693 * (0.014 - Q.girth2_top30) * (Q.sum_pt - 904.0)
    if Q.psi_0p3 > 0.996 and Q.sum_pt_top40 < 1210.0:
        z += -2.1 * (Q.psi_0p3 - 0.996) * (1210.0 - Q.sum_pt_top40)
    if Q.tau21_b2 < 0.342 and Q.girth2_top40 < 0.00885:
        z += -949.0 * (0.342 - Q.tau21_b2) * (0.00885 - Q.girth2_top40)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.546
    if Q.girth2_top5 < 0.00958:
        z += -157.0 * Q.girth2_top5 + 1.50406
    if Q.sj2_dr >= 0.272:
        z += 12.6 * Q.sj2_dr - 3.4272
    if Q.sj3_dr_max >= 0.369:
        z += -8.79 * Q.sj3_dr_max + 3.24351
    if Q.sum_pt_top20 >= 834.0:
        z += 0.004 * Q.sum_pt_top20 - 3.336
    if Q.sum_pt_top30 >= 988.0:
        z += -0.0114 * Q.sum_pt_top30 + 11.2632
    if Q.sum_pt_top50 >= 1120.0:
        z += 0.00781 * Q.sum_pt_top50 - 8.7472
    if Q.tau1 < 0.0697:
        z += 30.5 * Q.tau1 - 2.12585
    if Q.z_dr_0p1_0p2 < 0.127:
        z += -4.95 * Q.z_dr_0p1_0p2 + 0.62865
    if Q.girth2_top10 < 0.00819 and Q.tau4 > 0.019:
        z += -18100.0 * (0.00819 - Q.girth2_top10) * (Q.tau4 - 0.019)
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
