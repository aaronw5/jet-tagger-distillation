"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements, with each class score (logit) written out as a formula.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 87.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
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
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr01                   ΔR between particles 0 and 1
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
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
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        max_dr=max(dr[i] for i in real),
        n_for_90pct=ncum(0.9),
        pt_1=pt[1],
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr01=math.sqrt(dist2(0, 1)),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau3=tau_n(3),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -2.417431
    if Q.planar_flow < 0.1484197:
        z += -13.83471 * Q.planar_flow + 2.053344
    if Q.width < 0.004372139:
        z += 1659.928 * Q.width - 4.662843
    if 0.004372139 <= Q.width < 0.008678045:
        z += -602.5658 * Q.width + 5.229093
    if Q.mass < 21.78408:
        z += -0.03661288 * Q.mass - 1.624534
    if 21.78408 <= Q.mass < 56.92035:
        z += 0.0689348 * Q.mass - 3.923793
    if Q.girth2 < 0.01323868:
        z += -235.418 * Q.girth2 + 3.116622
    if Q.lam1 < 0.005433361:
        z += -60.66656 * Q.lam1 + 0.7222299
    if 0.005433361 <= Q.lam1 < 0.006506576:
        z += -365.8228 * Q.lam1 + 2.380254
    if Q.sum_pt >= 901.5938:
        z += -0.02260682 * Q.sum_pt + 20.38216
    if Q.C2_b2 < 0.001563465:
        z += 1364.818 * Q.C2_b2 - 2.133845
    if Q.sj3_dr_max < 0.1070199:
        z += -10.34001 * Q.sj3_dr_max + 3.114428
    if 0.1070199 <= Q.sj3_dr_max < 0.233678:
        z += -3.223255 * Q.sj3_dr_max + 2.352793
    if 0.233678 <= Q.sj3_dr_max < 0.3012016:
        z += -32.50908 * Q.sj3_dr_max + 9.196247
    if Q.sj3_dr_max >= 0.3012016:
        z += -22.16907 * Q.sj3_dr_max + 6.081819
    if Q.girth < 0.08723651:
        z += 43.05435 * Q.girth - 3.755912
    if Q.e2 < 0.0245477:
        z += -227.4702 * Q.e2 + 5.58387
    if Q.n_dr_0_0p05 >= 4.0:
        z += 0.40509 * Q.n_dr_0_0p05 - 1.62036
    if Q.log_sum_pt >= 6.670067:
        z += -13.21815 * Q.log_sum_pt + 88.16592
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.01865327 * Q.sum_pt_top5 - 12.82296
    if Q.lam1 < 0.006506576 and Q.D2 < 0.875672:
        z += -2988.999 * (0.006506576 - Q.lam1) * (0.875672 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.D2 < 1.002471:
        z += 461.6906 * (0.01323868 - Q.girth2) * (1.002471 - Q.D2)
    if Q.girth2 < 0.01323868 and Q.centroid_offset > 0.01837778:
        z += -12219.26 * (0.01323868 - Q.girth2) * (Q.centroid_offset - 0.01837778)
    if Q.lam2 < 7.300726e-05 and Q.D2_b2 < 0.2669656:
        z += 73765.7 * (7.300726e-05 - Q.lam2) * (0.2669656 - Q.D2_b2)
    return max(0.0, z)


def neuron_1(Q):
    z = -0.09412154
    if Q.pt_7 >= 34.53125:
        z += 0.1487708 * Q.pt_7 - 5.13724
    if 6.377723 <= Q.log_sum_pt < 6.502799:
        z += 5.815856 * Q.log_sum_pt - 37.09192
    if Q.log_sum_pt >= 6.502799:
        z += 11.70176 * Q.log_sum_pt - 75.36675
    if Q.mass_over_sum_pt_sq < 0.005832932:
        z += -633.314 * Q.mass_over_sum_pt_sq + 3.694077
    if Q.z_7 < 0.06164517:
        z += 77.3932 * Q.z_7 - 4.770917
    if Q.girth2 < 0.008678045:
        z += 540.0432 * Q.girth2 - 4.686519
    if Q.e2_sq < 0.008168571:
        z += -219.1891 * Q.e2_sq + 1.790462
    if 367.5938 <= Q.sum_pt_top5 < 531.1875:
        z += 0.004937477 * Q.sum_pt_top5 - 1.814986
    if Q.sum_pt_top5 >= 531.1875:
        z += 0.0005230694 * Q.sum_pt_top5 + 0.5298924
    if Q.width < 0.00609665:
        z += 417.6075 * Q.width - 2.546007
    if Q.pt_7 > 34.53125 and Q.tau2 < 0.01713288:
        z += -9.214913 * (Q.pt_7 - 34.53125) * (0.01713288 - Q.tau2)
    if Q.log_sum_pt > 6.377723 and Q.z_dr_0p05_0p1 < 0.7509095:
        z += -3.109254 * (Q.log_sum_pt - 6.377723) * (0.7509095 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.7863654
    if Q.log_sum_pt < 6.46415:
        z += -3.857629 * Q.log_sum_pt + 24.93629
    if 6.842717 <= Q.log_sum_pt < 6.896095:
        z += 15.2788 * Q.log_sum_pt - 104.5485
    if Q.log_sum_pt >= 6.896095:
        z += -9.135899 * Q.log_sum_pt + 63.81759
    if Q.lam1 < 0.00595415:
        z += -525.5309 * Q.lam1 + 3.12909
    if Q.z_7 < 0.03243272:
        z += 132.0194 * Q.z_7 - 4.281749
    if Q.girth < 0.007673833:
        z += 558.6993 * Q.girth - 4.287365
    if Q.sum_pt < 788.4484:
        z += -0.003140693 * Q.sum_pt + 2.476275
    if Q.girth2 < 0.0003193707:
        z += 13594.71 * Q.girth2 - 4.341752
    if Q.girth2 < 0.0003193707 and Q.mass_top2 < 36.76827:
        z += 468.1396 * (0.0003193707 - Q.girth2) * (36.76827 - Q.mass_top2)
    return max(0.0, z)


def neuron_3(Q):
    z = -4.866957
    if Q.mass_over_sum_pt >= 0.0681391:
        z += 33.64804 * Q.mass_over_sum_pt - 2.292747
    if 0.05356915 <= Q.tau1 < 0.1027642:
        z += -50.95515 * Q.tau1 + 2.729624
    if Q.tau1 >= 0.1027642:
        z += 9.801266 * Q.tau1 - 3.513962
    if 0.008375572 <= Q.lam1 < 0.01200373:
        z += 315.049 * Q.lam1 - 2.638715
    if 0.01200373 <= Q.lam1 < 0.01643375:
        z += -41.80328 * Q.lam1 + 1.644841
    if Q.lam1 >= 0.01643375:
        z += 113.4903 * Q.lam1 - 0.9072144
    if 0.04081947 <= Q.girth < 0.07608178:
        z += 101.1847 * Q.girth - 4.130308
    if Q.girth >= 0.07608178:
        z += 176.4524 * Q.girth - 9.856802
    if 0.007520088 <= Q.width < 0.01323868:
        z += 559.5157 * Q.width - 4.207608
    if Q.width >= 0.01323868:
        z += 332.8429 * Q.width - 1.206759
    if Q.e2 >= 0.06344108:
        z += 278.6867 * Q.e2 - 17.68018
    if Q.girth2 < 0.004372139:
        z += 545.2256 * Q.girth2 - 2.383803
    if Q.girth2 >= 0.008678045:
        z += -1560.378 * Q.girth2 + 13.54103
    if Q.sj2_dr >= 0.1872617:
        z += 32.90918 * Q.sj2_dr - 6.16263
    if Q.lam2 >= 0.001130645:
        z += -978.3712 * Q.lam2 + 1.10619
    if Q.mass >= 64.61873:
        z += -0.09491761 * Q.mass + 6.133456
    if Q.girth > 0.04081947 and Q.log_sum_pt > 6.080494:
        z += 130.0719 * (Q.girth - 0.04081947) * (Q.log_sum_pt - 6.080494)
    if Q.mass_over_sum_pt > 0.0681391 and Q.sj2_dr < 0.2179769:
        z += -3751.477 * (Q.mass_over_sum_pt - 0.0681391) * (0.2179769 - Q.sj2_dr)
    if Q.centroid_offset > 0.01096064 and Q.n_pt_above_50 < 8.0:
        z += 13.73175 * (Q.centroid_offset - 0.01096064) * (8.0 - Q.n_pt_above_50)
    if Q.lam2 > 0.001130645 and Q.pt_6 < 56.53125:
        z += 34.65721 * (Q.lam2 - 0.001130645) * (56.53125 - Q.pt_6)
    return max(0.0, z)


def neuron_4(Q):
    z = -3.164716
    if Q.N2 < 0.2233283:
        z += -64.31693 * Q.N2 + 14.36379
    if Q.lam2 < 0.000537286:
        z += 9547.75 * Q.lam2 - 5.129872
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -148.3439 * Q.mass_over_sum_pt + 13.41234
    if Q.girth2 < 0.002635418:
        z += -672.431 * Q.girth2 - 6.266078
    if 0.002635418 <= Q.girth2 < 0.003562611:
        z += 690.8075 * Q.girth2 - 9.858781
    if 0.003562611 <= Q.girth2 < 0.008678045:
        z += 1133.778 * Q.girth2 - 11.43691
    if 0.008678045 <= Q.girth2 < 0.01882765:
        z += 823.6676 * Q.girth2 - 8.745759
    if Q.girth2 >= 0.01882765:
        z += 442.9708 * Q.girth2 - 1.578133
    if Q.sj3_dr_max >= 0.233678:
        z += -21.48242 * Q.sj3_dr_max + 5.019969
    if Q.e2_sq < 0.01165737:
        z += -959.476 * Q.e2_sq + 12.27215
    if 0.01165737 <= Q.e2_sq < 0.01716248:
        z += -197.4851 * Q.e2_sq + 3.389335
    if Q.mass < 69.61135:
        z += 0.01684045 * Q.mass - 0.7868035
    if 69.61135 <= Q.mass < 76.6557:
        z += -0.05472227 * Q.mass + 4.194774
    if Q.max_dr >= 0.1117619:
        z += 23.96368 * Q.max_dr - 2.678225
    if Q.width < 0.01323868:
        z += 204.6598 * Q.width - 2.709425
    if Q.e2 >= 0.009668065:
        z += 24.52623 * Q.e2 - 0.2371212
    if Q.girth >= 0.05464922:
        z += -83.57821 * Q.girth + 4.567484
    if Q.C2_b2 < 0.009032972:
        z += -635.306 * Q.C2_b2 + 5.738702
    if Q.tau1 >= 0.1136369:
        z += 40.95097 * Q.tau1 - 4.653541
    if Q.N2 < 0.2233283 and Q.e2_sq > 0.01165737:
        z += -2741.871 * (0.2233283 - Q.N2) * (Q.e2_sq - 0.01165737)
    if Q.N2 < 0.2233283 and Q.pt_6 < 62.25:
        z += -0.2348624 * (0.2233283 - Q.N2) * (62.25 - Q.pt_6)
    if Q.N2 < 0.2233283 and Q.eccentricity > 0.7117266:
        z += -86.98067 * (0.2233283 - Q.N2) * (Q.eccentricity - 0.7117266)
    if Q.tau1 > 0.07283629 and Q.sj3_dr_min < 0.2089872:
        z += 286.7206 * (Q.tau1 - 0.07283629) * (0.2089872 - Q.sj3_dr_min)
    if Q.mass < 76.6557 and Q.D2 < 0.875672:
        z += -0.3276204 * (76.6557 - Q.mass) * (0.875672 - Q.D2)
    if Q.e2_sq < 0.01165737 and Q.D2 < 0.875672:
        z += 634.2974 * (0.01165737 - Q.e2_sq) * (0.875672 - Q.D2)
    if Q.sj3_dr_max > 0.169029 and Q.dr01 < 0.02117036:
        z += -585.9924 * (Q.sj3_dr_max - 0.169029) * (0.02117036 - Q.dr01)
    if Q.D2 < 1.002471 and Q.M3 > 0.04581318:
        z += 44.53423 * (1.002471 - Q.D2) * (Q.M3 - 0.04581318)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.685334
    if Q.z_7 < 0.02807091:
        z += -251.6268 * Q.z_7 + 8.812063
    if 0.02807091 <= Q.z_7 < 0.04939969:
        z += -81.98639 * Q.z_7 + 4.050102
    if 6.267538 <= Q.log_sum_pt < 6.701242:
        z += 3.944505 * Q.log_sum_pt - 24.72234
    if 6.701242 <= Q.log_sum_pt < 6.896095:
        z += -13.97733 * Q.log_sum_pt + 95.37621
    if Q.log_sum_pt >= 6.896095:
        z += -65.66487 * Q.log_sum_pt + 451.8185
    if Q.e2_sq < 0.005284669:
        z += -760.8062 * Q.e2_sq + 4.020608
    if Q.pt_7 < 23.21641:
        z += 0.09400035 * Q.pt_7 - 3.492701
    if 23.21641 <= Q.pt_7 < 37.15625:
        z += 0.04609782 * Q.pt_7 - 2.380576
    if Q.pt_7 >= 37.15625:
        z += -0.04790254 * Q.pt_7 + 1.112125
    if Q.zdr_0 < 0.02383244:
        z += 128.5201 * Q.zdr_0 - 3.062947
    if Q.sj3_dr_max < 0.3012016:
        z += 6.093912 * Q.sj3_dr_max - 1.835496
    if Q.mass < 60.63098:
        z += -0.03822879 * Q.mass + 2.317849
    if Q.girth2_top3 < 0.002915531:
        z += 562.8433 * Q.girth2_top3 - 1.640987
    if Q.mass_over_sum_pt < 0.09041383:
        z += 29.95585 * Q.mass_over_sum_pt - 2.708423
    if Q.tau1 < 0.09538712:
        z += -11.60847 * Q.tau1 + 1.107298
    if Q.LHA < 0.2160559 and Q.log_sum_pt < 6.804164:
        z += -153.4978 * (0.2160559 - Q.LHA) * (6.804164 - Q.log_sum_pt)
    if Q.z_7 < 0.07148865 and Q.centroid_offset < 0.03117077:
        z += -1600.013 * (0.07148865 - Q.z_7) * (0.03117077 - Q.centroid_offset)
    if Q.z_7 < 0.07148865 and Q.mass_top3 < 42.18339:
        z += 0.8812857 * (0.07148865 - Q.z_7) * (42.18339 - Q.mass_top3)
    if Q.girth2 < 0.001653836 and Q.centroid_offset < 0.02355416:
        z += 160212.7 * (0.001653836 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.log_sum_pt > 6.701242 and Q.mean_phi2 < 0.0002302115:
        z += 53078.77 * (Q.log_sum_pt - 6.701242) * (0.0002302115 - Q.mean_phi2)
    if Q.girth2_top3 < 0.002915531 and Q.tau3 < 0.006646257:
        z += 195913.6 * (0.002915531 - Q.girth2_top3) * (0.006646257 - Q.tau3)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.7932636
    if 0.00809236 <= Q.centroid_offset < 0.01837778:
        z += 151.4742 * Q.centroid_offset - 1.225784
    if Q.centroid_offset >= 0.01837778:
        z += -37.86377 * Q.centroid_offset + 2.253828
    if Q.width < 0.01323868:
        z += 404.8059 * Q.width - 5.359094
    if Q.tau1 < 0.1136369:
        z += -62.78069 * Q.tau1 + 7.134203
    if Q.sj3_dr_min >= 0.1278212:
        z += -26.39771 * Q.sj3_dr_min + 3.374186
    if Q.lam1 < 0.00733008:
        z += -518.7249 * Q.lam1 + 5.136187
    if 0.00733008 <= Q.lam1 < 0.01200373:
        z += -285.4071 * Q.lam1 + 3.425948
    if Q.sj3_pair_mass_min >= 4.501727:
        z += -0.105165 * Q.sj3_pair_mass_min + 0.4734242
    if Q.e2_sq < 0.0030133:
        z += -1180.545 * Q.e2_sq + 3.557336
    if Q.lam2 < 0.001130645:
        z += 2047.89 * Q.lam2 - 2.315435
    if Q.sj3_dr_max < 0.1789613:
        z += -43.57116 * Q.sj3_dr_max + 7.797554
    if Q.sj3_dr_max >= 0.1879486:
        z += 13.69197 * Q.sj3_dr_max - 2.573386
    if Q.max_dr < 0.1452311:
        z += 43.15722 * Q.max_dr - 6.267771
    if Q.e3 < 0.0005116989:
        z += -6230.968 * Q.e3 + 3.188379
    if Q.mass < 45.7571:
        z += -0.03968707 * Q.mass + 1.815965
    if Q.girth2 < 0.008678045:
        z += 1738.251 * Q.girth2 - 15.08462
    if Q.centroid_offset > 0.00809236 and Q.sj3_pair_mass_min > 6.811308:
        z += 3.010573 * (Q.centroid_offset - 0.00809236) * (Q.sj3_pair_mass_min - 6.811308)
    if Q.pt_6 < 41.21875 and Q.log_sum_pt < 6.766778:
        z += 1.173602 * (41.21875 - Q.pt_6) * (6.766778 - Q.log_sum_pt)
    if Q.centroid_offset > 0.01837778 and Q.mean_phi2 < 0.008921136:
        z += 8349.59 * (Q.centroid_offset - 0.01837778) * (0.008921136 - Q.mean_phi2)
    if Q.pt_6 < 41.21875 and Q.z_7 > 0.02320757:
        z += -13.53752 * (41.21875 - Q.pt_6) * (Q.z_7 - 0.02320757)
    if Q.mass < 60.63098 and Q.z_dr_0p2_0p4 < 0.05643852:
        z += -0.814989 * (60.63098 - Q.mass) * (0.05643852 - Q.z_dr_0p2_0p4)
    if Q.lam2 < 0.003408389 and Q.n_dr_0_0p05 < 3.0:
        z += 108.5842 * (0.003408389 - Q.lam2) * (3.0 - Q.n_dr_0_0p05)
    if Q.sj3_pair_mass_min > 4.501727 and Q.n_dr_0p2_0p4 > 1.0:
        z += -0.04309208 * (Q.sj3_pair_mass_min - 4.501727) * (Q.n_dr_0p2_0p4 - 1.0)
    return max(0.0, z)


def neuron_7(Q):
    z = 8.388207
    if Q.girth2 < 0.0009641429:
        z += 3857.88 * Q.girth2 - 3.719548
    if 0.001653836 <= Q.girth2 < 0.004372139:
        z += 470.1983 * Q.girth2 - 0.7776311
    if 0.004372139 <= Q.girth2 < 0.007520088:
        z += 164.4604 * Q.girth2 + 0.5590976
    if 0.007520088 <= Q.girth2 < 0.01323868:
        z += -740.5789 * Q.girth2 + 7.365073
    if Q.girth2 >= 0.01323868:
        z += 1084.942 * Q.girth2 - 16.80241
    if 0.01109984 <= Q.mass_over_sum_pt < 0.07269073:
        z += -38.45169 * Q.mass_over_sum_pt + 0.4268077
    if 0.07269073 <= Q.mass_over_sum_pt < 0.08475161:
        z += 75.60696 * Q.mass_over_sum_pt - 7.864199
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 276.1647 * Q.mass_over_sum_pt - 24.86179
    if Q.mass_over_sum_pt >= 0.09041383:
        z += -59.36496 * Q.mass_over_sum_pt + 5.47473
    if Q.tau1 < 0.05356915:
        z += -34.21965 * Q.tau1 + 1.833118
    if 36.22941 <= Q.mass < 76.6557:
        z += 0.08619455 * Q.mass - 3.122778
    if Q.mass >= 76.6557:
        z += -0.1746868 * Q.mass + 16.87527
    if 0.005590289 <= Q.width < 0.008678045:
        z += -1202.871 * Q.width + 6.724396
    if Q.width >= 0.008678045:
        z += -701.4083 * Q.width + 2.37268
    if Q.centroid_offset < 0.02076709:
        z += 88.799 * Q.centroid_offset - 1.844097
    if Q.girth2_top2 < 8.10414e-05:
        z += 507325.9 * Q.girth2_top2 - 41.1144
    if Q.sj2_dr < 0.1591713:
        z += -14.66429 * Q.sj2_dr + 1.646133
    if 0.1591713 <= Q.sj2_dr < 0.1872617:
        z += 24.49241 * Q.sj2_dr - 4.58649
    if Q.girth < 0.08723651:
        z += 60.46098 * Q.girth - 5.274405
    if Q.LHA < 0.3033137:
        z += -10.04638 * Q.LHA + 3.047205
    if Q.lam1 < 0.008375572:
        z += 347.5352 * Q.lam1 - 2.910806
    if Q.e3 < 8.147744e-05:
        z += 31629.58 * Q.e3 - 2.577097
    if Q.e2 >= 0.05028464:
        z += -304.9666 * Q.e2 + 15.33514
    return max(0.0, z)


def neuron_8(Q):
    z = -0.7267793
    if Q.girth2 < 0.005019719:
        z += -230.018 * Q.girth2 + 1.154626
    if Q.LHA < 0.1967397:
        z += 64.20506 * Q.LHA - 12.63169
    if Q.log_sum_pt >= 6.701242:
        z += -12.38385 * Q.log_sum_pt + 82.9872
    if Q.girth < 0.06108601:
        z += 75.46003 * Q.girth - 4.609552
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.01007342 * Q.sum_pt_top5 - 6.924844
    if Q.mass_over_sum_pt < 0.03319429:
        z += -89.18823 * Q.mass_over_sum_pt + 2.96054
    if Q.sj3_dr_max < 0.1426152:
        z += 31.67499 * Q.sj3_dr_max - 4.517334
    if Q.girth2 < 0.006679471 and Q.centroid_offset < 0.02355416:
        z += 33692.16 * (0.006679471 - Q.girth2) * (0.02355416 - Q.centroid_offset)
    if Q.girth2 < 0.005019719 and Q.z_dr_0p2_0p4 < 0.1009734:
        z += 4974.301 * (0.005019719 - Q.girth2) * (0.1009734 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.06108601 and Q.lam2 < 0.0001947983:
        z += 276112.5 * (0.06108601 - Q.girth) * (0.0001947983 - Q.lam2)
    if Q.log_sum_pt > 6.701242 and Q.e3 < 2.955458e-05:
        z += -519602.5 * (Q.log_sum_pt - 6.701242) * (2.955458e-05 - Q.e3)
    if Q.girth2 < 0.005019719 and Q.centroid_offset > 0.006789738:
        z += -24783.14 * (0.005019719 - Q.girth2) * (Q.centroid_offset - 0.006789738)
    if Q.log_sum_pt > 6.701242 and Q.width < 0.008678045:
        z += 1019.179 * (Q.log_sum_pt - 6.701242) * (0.008678045 - Q.width)
    if Q.mass < 29.6447 and Q.width < 0.003562611:
        z += -44.36972 * (29.6447 - Q.mass) * (0.003562611 - Q.width)
    if Q.sj3_dr_max < 0.1986272 and Q.girth2 < 0.006679471:
        z += 2399.552 * (0.1986272 - Q.sj3_dr_max) * (0.006679471 - Q.girth2)
    if Q.tau1 < 0.05356915 and Q.width < 0.003562611:
        z += 22400.78 * (0.05356915 - Q.tau1) * (0.003562611 - Q.width)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.832185
    if Q.girth < 0.05464922:
        z += 64.86963 * Q.girth - 3.545075
    if Q.girth >= 0.0717028:
        z += -77.76084 * Q.girth + 5.57567
    if Q.tau1 < 0.04369778:
        z += -89.9129 * Q.tau1 + 3.928994
    if Q.mass < 53.33237:
        z += 0.09226678 * Q.mass - 4.920807
    if Q.girth2 < 0.003562611:
        z += -3768.767 * Q.girth2 + 17.90775
    if 0.003562611 <= Q.girth2 < 0.00609665:
        z += -1768.36 * Q.girth2 + 10.78108
    if Q.sj3_dr_max < 0.1426152:
        z += 38.16582 * Q.sj3_dr_max - 3.832108
    if 0.1426152 <= Q.sj3_dr_max < 0.213399:
        z += -22.75828 * Q.sj3_dr_max + 4.856593
    if Q.lam2 >= 0.001130645:
        z += 579.7119 * Q.lam2 - 0.6554482
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.9639006 * Q.n_dr_0p2_0p4 - 0.9639006
    if Q.lam1 < 0.003377388:
        z += 687.1715 * Q.lam1 - 4.091522
    if 0.003377388 <= Q.lam1 < 0.005433361:
        z += 431.9753 * Q.lam1 - 3.229625
    if 0.005433361 <= Q.lam1 < 0.00595415:
        z += 961.365 * Q.lam1 - 6.105991
    if Q.lam1 >= 0.00595415:
        z += 274.1935 * Q.lam1 - 2.014468
    if Q.centroid_offset < 0.01837778:
        z += -200.828 * Q.centroid_offset + 3.690772
    if 0.03556091 <= Q.e2 < 0.04447357:
        z += -53.04834 * Q.e2 + 1.886447
    if Q.e2 >= 0.04447357:
        z += 117.6224 * Q.e2 - 5.703888
    if Q.sj3_pair_mass_max < 24.23013:
        z += 0.1499527 * Q.sj3_pair_mass_max - 3.633374
    if Q.C3 < 0.0284695:
        z += 72.7653 * Q.C3 - 2.071591
    if Q.e3 < 2.371297e-05:
        z += 84531.47 * Q.e3 - 2.004493
    if Q.mass_over_sum_pt < 0.07269073:
        z += 66.49413 * Q.mass_over_sum_pt - 4.833507
    if Q.sum_pt < 988.4078:
        z += -0.008796196 * Q.sum_pt + 8.694229
    z += -8.805234 * Q.z_dr_0p2_0p4
    if Q.max_dr < 0.1117619:
        z += -51.85093 * Q.max_dr + 5.794956
    if Q.mass < 53.33237 and Q.centroid_offset < 0.02685622:
        z += 6.692211 * (53.33237 - Q.mass) * (0.02685622 - Q.centroid_offset)
    if Q.mass < 53.33237 and Q.log_sum_pt < 6.842717:
        z += 0.1183909 * (53.33237 - Q.mass) * (6.842717 - Q.log_sum_pt)
    if Q.lam1 > 0.005433361 and Q.eccentricity > 0.7117266:
        z += -748.4049 * (Q.lam1 - 0.005433361) * (Q.eccentricity - 0.7117266)
    if Q.centroid_offset < 0.01837778 and Q.n_for_90pct > 5.0:
        z += -62.37537 * (0.01837778 - Q.centroid_offset) * (Q.n_for_90pct - 5.0)
    if Q.centroid_offset < 0.01837778 and Q.pt_1 < 159.25:
        z += -3.009064 * (0.01837778 - Q.centroid_offset) * (159.25 - Q.pt_1)
    if Q.centroid_offset < 0.01837778 and Q.z_2nd < 0.2055511:
        z += 1848.072 * (0.01837778 - Q.centroid_offset) * (0.2055511 - Q.z_2nd)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.88596
    z += -48.00774 * Q.e2
    if Q.sj3_pair_mass_min >= 11.051:
        z += 0.1771744 * Q.sj3_pair_mass_min - 1.957955
    if Q.lam1 < 0.001503553:
        z += 2506.196 * Q.lam1 - 5.378889
    if 0.001503553 <= Q.lam1 < 0.004183811:
        z += 600.9459 * Q.lam1 - 2.514244
    if Q.lam1 >= 0.00733008:
        z += -301.0504 * Q.lam1 + 2.206724
    if 0.0003061234 <= Q.lam2 < 0.003408389:
        z += 530.9472 * Q.lam2 - 0.1625353
    if Q.lam2 >= 0.003408389:
        z += -95.30048 * Q.lam2 + 1.97196
    if Q.LHA >= 0.3033137:
        z += -29.69881 * Q.LHA + 9.008056
    if Q.tau1 >= 0.05356915:
        z += 50.7083 * Q.tau1 - 2.716401
    if Q.girth2 >= 0.007520088:
        z += 522.6854 * Q.girth2 - 3.930641
    if Q.e3 >= 3.892127e-05:
        z += -4828.053 * Q.e3 + 0.1879139
    if Q.mass < 76.6557:
        z += -0.02897324 * Q.mass + 2.220964
    if Q.C2_b2 >= 0.009032972:
        z += 109.8528 * Q.C2_b2 - 0.9922975
    if Q.lam1 > 0.00733008 and Q.D2_b2 < 0.380911:
        z += -395.1002 * (Q.lam1 - 0.00733008) * (0.380911 - Q.D2_b2)
    return max(0.0, z)


def neuron_11(Q):
    z = -3.687039
    if Q.girth2 < 0.01323868:
        z += -314.0508 * Q.girth2 + 4.157617
    if Q.tau1 < 0.09538712:
        z += 20.73316 * Q.tau1 - 1.977676
    if Q.mass < 15.45403:
        z += -0.1493811 * Q.mass + 2.30854
    if Q.centroid_offset < 0.03776099:
        z += -92.23491 * Q.centroid_offset + 3.482882
    if Q.sj3_dr_max < 0.1426152:
        z += 33.92889 * Q.sj3_dr_max - 4.155775
    if 0.1426152 <= Q.sj3_dr_max < 0.169029:
        z += 48.8287 * Q.sj3_dr_max - 6.280714
    if 0.169029 <= Q.sj3_dr_max < 0.2623172:
        z += -21.14689 * Q.sj3_dr_max + 5.547193
    if Q.width < 0.008678045:
        z += -1513.354 * Q.width + 13.13296
    if Q.girth < 0.02054282:
        z += 291.9157 * Q.girth - 9.604258
    if 0.02054282 <= Q.girth < 0.0717028:
        z += 70.51385 * Q.girth - 5.056041
    if Q.e2_sq < 0.008168571:
        z += 383.7354 * Q.e2_sq - 3.134569
    if Q.z_7 >= 0.01685855:
        z += 26.13782 * Q.z_7 - 0.4406457
    if Q.sj2_dr >= 0.2687922:
        z += 35.06553 * Q.sj2_dr - 9.425339
    if Q.lam1 < 0.008375572:
        z += 486.1113 * Q.lam1 - 4.07146
    if Q.max_dr < 0.1452311:
        z += -16.0053 * Q.max_dr + 2.324467
    if Q.centroid_offset < 0.03776099 and Q.sum_pt < 901.5938:
        z += -0.2027867 * (0.03776099 - Q.centroid_offset) * (901.5938 - Q.sum_pt)
    if Q.centroid_offset > 0.04990367 and Q.D2 < 2.843757:
        z += -224.5919 * (Q.centroid_offset - 0.04990367) * (2.843757 - Q.D2)
    if Q.centroid_offset > 0.04990367 and Q.tau32 < 0.5502779:
        z += 904.1195 * (Q.centroid_offset - 0.04990367) * (0.5502779 - Q.tau32)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.395752
    if Q.girth2 >= 0.01882765:
        z += 287.8117 * Q.girth2 - 5.418819
    if Q.mass >= 88.15578:
        z += 0.1086228 * Q.mass - 9.57573
    if Q.mass_over_sum_pt >= 0.1309286:
        z += -51.66452 * Q.mass_over_sum_pt + 6.764365
    return max(0.0, z)


def neuron_13(Q):
    z = -0.4485388
    if Q.girth < 0.1484084:
        z += -44.61491 * Q.girth + 6.621229
    if Q.lam1 < 0.01643375:
        z += -156.5979 * Q.lam1 + 2.573491
    if Q.e2 < 0.08000524:
        z += 18.71031 * Q.e2 - 1.496923
    if Q.z_7 < 0.02807091:
        z += 254.8358 * Q.z_7 - 7.153474
    if Q.girth < 0.1484084 and Q.log_sum_pt < 6.804164:
        z += -45.16549 * (0.1484084 - Q.girth) * (6.804164 - Q.log_sum_pt)
    if Q.girth < 0.1484084 and Q.pt_7 < 38.53125:
        z += -1.463587 * (0.1484084 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 40.04062:
        z += 0.001170437 * (Q.sum_pt_top5 - 658.125) * (40.04062 - Q.pt_7)
    if Q.girth < 0.1484084 and Q.tau2 > 0.008780509:
        z += 945.0926 * (0.1484084 - Q.girth) * (Q.tau2 - 0.008780509)
    if Q.sum_pt > 988.4078 and Q.pt_6 < 62.25:
        z += -0.0006137886 * (Q.sum_pt - 988.4078) * (62.25 - Q.pt_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -3.181758
    if Q.girth2 >= 0.007520088:
        z += -1372.959 * Q.girth2 + 10.32477
    if 0.002074109 <= Q.e2_sq < 0.0030133:
        z += 144.4483 * Q.e2_sq - 0.2996015
    if Q.e2_sq >= 0.0030133:
        z += -53.28058 * Q.e2_sq + 0.2962149
    if 0.003562611 <= Q.width < 0.005590289:
        z += 343.9398 * Q.width - 1.225324
    if 0.005590289 <= Q.width < 0.006679471:
        z += 635.0673 * Q.width - 2.85281
    if 0.006679471 <= Q.width < 0.01323868:
        z += -1371.943 * Q.width + 10.55296
    if Q.width >= 0.01323868:
        z += -1046.802 * Q.width + 6.248514
    if 0.01655442 <= Q.e2 < 0.03556091:
        z += -212.952 * Q.e2 + 3.525296
    if 0.03556091 <= Q.e2 < 0.04110972:
        z += -290.8748 * Q.e2 + 6.296303
    if 0.04110972 <= Q.e2 < 0.05028464:
        z += -134.3173 * Q.e2 - 0.1397306
    if Q.e2 >= 0.05028464:
        z += 310.9543 * Q.e2 - 22.53005
    if 0.02685622 <= Q.centroid_offset < 0.04990367:
        z += -111.5262 * Q.centroid_offset + 2.995172
    if Q.centroid_offset >= 0.04990367:
        z += -1093.821 * Q.centroid_offset + 52.01527
    if 0.07992374 <= Q.mass_over_sum_pt < 0.08475161:
        z += 210.088 * Q.mass_over_sum_pt - 16.79102
    if 0.08475161 <= Q.mass_over_sum_pt < 0.09041383:
        z += 444.2515 * Q.mass_over_sum_pt - 36.63675
    if Q.mass_over_sum_pt >= 0.09041383:
        z += 288.2448 * Q.mass_over_sum_pt - 22.53159
    if Q.z_dr_0p05_0p1 >= 0.7509095:
        z += -24.17888 * Q.z_dr_0p05_0p1 + 18.15615
    if Q.sd_mass < 49.91626:
        z += -0.004128227 * Q.sd_mass - 0.4724486
    if 49.91626 <= Q.sd_mass < 74.57663:
        z += 0.02751436 * Q.sd_mass - 2.051928
    if 0.02689598 <= Q.girth < 0.04081947:
        z += 121.3632 * Q.girth - 3.264183
    if 0.04081947 <= Q.girth < 0.08065885:
        z += 218.0052 * Q.girth - 7.209056
    if 0.08065885 <= Q.girth < 0.08723651:
        z += 307.9337 * Q.girth - 14.46259
    if 0.08723651 <= Q.girth < 0.1019409:
        z += -6.867447 * Q.girth + 12.99957
    if Q.girth >= 0.1019409:
        z += -108.5488 * Q.girth + 23.36506
    if 0.06154135 <= Q.sj2_dr < 0.1591713:
        z += -13.86663 * Q.sj2_dr + 0.8533712
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 54.36639 * Q.sj2_dr - 10.00737
    if Q.sj2_dr >= 0.2001708:
        z += 24.79498 * Q.sj2_dr - 4.088035
    if Q.planar_flow < 0.1115136 and Q.lam1 < 0.00733008:
        z += -7276.977 * (0.1115136 - Q.planar_flow) * (0.00733008 - Q.lam1)
    if Q.planar_flow < 0.1115136 and Q.width < 0.00609665:
        z += 11051.99 * (0.1115136 - Q.planar_flow) * (0.00609665 - Q.width)
    if Q.z_dr_0p05_0p1 > 0.7509095 and Q.n_dr_0p2_0p4 < 1.0:
        z += 24.07586 * (Q.z_dr_0p05_0p1 - 0.7509095) * (1.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.025458
    if Q.N2 < 0.2233283:
        z += 20.70682 * Q.N2 - 4.624418
    if Q.girth2 < 0.007520088:
        z += 1053.515 * Q.girth2 - 7.922526
    if Q.mass_over_sum_pt < 0.1309286:
        z += 12.41601 * Q.mass_over_sum_pt - 1.625612
    if Q.width < 0.00609665:
        z += 232.3104 * Q.width + 5.91143
    if 0.00609665 <= Q.width < 0.01323868:
        z += -868.0472 * Q.width + 12.61992
    if 0.01323868 <= Q.width < 0.01882765:
        z += -201.8491 * Q.width + 3.800344
    if Q.e2 < 0.04110972:
        z += -174.9853 * Q.e2 + 7.193597
    if Q.e2_sq < 0.008168571:
        z += 1050.392 * Q.e2_sq - 9.589806
    if 0.008168571 <= Q.e2_sq < 0.01165737:
        z += 289.3842 * Q.e2_sq - 3.373461
    if Q.mass_over_sum_pt_sq < 0.007182836:
        z += -746.0186 * Q.mass_over_sum_pt_sq + 5.358529
    if Q.girth < 0.1019409:
        z += 28.63268 * Q.girth - 2.918842
    if Q.sj2_dr < 0.1492731:
        z += -7.753113 * Q.sj2_dr + 0.991416
    if 0.1492731 <= Q.sj2_dr < 0.1591713:
        z += -46.29313 * Q.sj2_dr + 6.744404
    if 0.1591713 <= Q.sj2_dr < 0.2001708:
        z += 15.22296 * Q.sj2_dr - 3.047192
    if Q.girth2_top2 < 0.0005124533:
        z += -1686.831 * Q.girth2_top2 + 0.8644224
    if Q.girth2_top3 < 0.006756161:
        z += -184.3761 * Q.girth2_top3 + 1.245675
    if Q.lam1 < 0.005433361:
        z += -295.5799 * Q.lam1 + 1.151973
    if 0.005433361 <= Q.lam1 < 0.008375572:
        z += 154.3123 * Q.lam1 - 1.292454
    if Q.centroid_offset < 0.01837778:
        z += 202.3983 * Q.centroid_offset - 3.719631
    if Q.girth2 < 0.007520088 and Q.D2 < 0.7459513:
        z += -2847.625 * (0.007520088 - Q.girth2) * (0.7459513 - Q.D2)
    if Q.mass_over_sum_pt < 0.1309286 and Q.D2 < 0.7459513:
        z += 146.7409 * (0.1309286 - Q.mass_over_sum_pt) * (0.7459513 - Q.D2)
    if Q.N2 < 0.2233283 and Q.sum_pt_top5 > 430.75:
        z += 0.07160176 * (0.2233283 - Q.N2) * (Q.sum_pt_top5 - 430.75)
    if Q.centroid_offset < 0.01837778 and Q.dr01 < 0.2512159:
        z += -449.5484 * (0.01837778 - Q.centroid_offset) * (0.2512159 - Q.dr01)
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
