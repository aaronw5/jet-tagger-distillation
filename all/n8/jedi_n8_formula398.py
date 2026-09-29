"""JEDI-linear jet tagger, 8 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; all observables), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 87.2% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_max             largest distance among the 3 subjet axes
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_6          mass of particles 0 and 6 [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_6                  pT share × ΔR of particle 6 (its part of the girth)
  Q.zdr_7                  pT share × ΔR of particle 7 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.abseta_7               |Δη| of particle 7
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr12               distance between subjet axes 1 and 2 (of 3)
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.sj3_dr23               distance between subjet axes 2 and 3 (of 3)
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        sj3_dr_max=max(subjets(3)["dr"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_6=pair_mass(0, 6),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        pt_6=pt[6],
        pt_7=pt[7],
        z_7=z[7],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_2=z[2] * dr[2],
        zdr_6=z[6] * dr[6],
        zdr_7=z[7] * dr[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        abseta_7=abs(eta[7]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr12=subjets(3)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        sj3_dr23=subjets(3)["dr"][2],
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -11.1
    if Q.LHA < 0.148:
        z += 23.9 * Q.LHA - 3.8001
    if 0.148 <= Q.LHA < 0.159:
        z += 91.3 * Q.LHA - 13.7753
    if Q.LHA >= 0.159:
        z += 67.4 * Q.LHA - 9.9752
    if Q.N2 < 0.339:
        z += 5.76 * Q.N2 - 1.95264
    if Q.e2_sq >= 0.00697:
        z += 473.0 * Q.e2_sq - 3.29681
    if Q.girth < 0.054:
        z += -211.0 * Q.girth + 11.394
    if Q.girth >= 0.0541:
        z += -155.0 * Q.girth + 8.3855
    if Q.girth2 < 0.0129:
        z += -1221.0 * Q.girth2 + 18.9429
    if 0.0129 <= Q.girth2 < 0.0186:
        z += -560.0 * Q.girth2 + 10.416
    if Q.girth2_top5 < 0.0165:
        z += 265.0 * Q.girth2_top5 - 4.3725
    z += -578.0 * Q.lam2
    if Q.mass < 37.3:
        z += 0.0715 * Q.mass - 4.44015
    if 37.3 <= Q.mass < 62.1:
        z += 0.1202 * Q.mass - 6.25666
    if Q.mass >= 62.1:
        z += 0.0487 * Q.mass - 1.81651
    if Q.mass_over_sum_pt < 0.083:
        z += 124.0 * Q.mass_over_sum_pt - 10.292
    if Q.mass_over_sum_pt_sq >= 0.0164:
        z += -401.0 * Q.mass_over_sum_pt_sq + 6.5764
    if Q.mass_top5 < 42.2:
        z += -0.0223 * Q.mass_top5 + 0.94106
    if Q.planar_flow < 0.128:
        z += -7.87 * Q.planar_flow + 1.00736
    if 0.0156 <= Q.sj3_dr_max < 0.105:
        z += -22.4 * Q.sj3_dr_max + 0.34944
    if 0.105 <= Q.sj3_dr_max < 0.179:
        z += 25.3 * Q.sj3_dr_max - 4.65906
    if Q.sj3_dr_max >= 0.179:
        z += -4.2 * Q.sj3_dr_max + 0.62144
    if Q.sj3_pair_mass_min >= 13.2:
        z += -0.0589 * Q.sj3_pair_mass_min + 0.77748
    if Q.sum_pt >= 913.0:
        z += -0.00826 * Q.sum_pt + 7.54138
    if Q.log_sum_pt > 6.64 and Q.z_7 > 0.0118:
        z += -188.0 * (Q.log_sum_pt - 6.64) * (Q.z_7 - 0.0118)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.35
    if Q.centroid_offset < 0.0146:
        z += 14.4 * Q.centroid_offset - 0.21024
    if Q.e2_sq < 0.00462:
        z += 240.0 * Q.e2_sq - 1.1088
    if Q.girth < 0.0565:
        z += -47.7 * Q.girth + 2.69505
    if Q.girth2 < 0.00838:
        z += 341.0 * Q.girth2 - 2.85758
    if Q.lam1 < 0.000754:
        z += 505.0 * Q.lam1 - 0.38077
    if Q.log_sum_pt >= 6.45:
        z += 10.5 * Q.log_sum_pt - 67.725
    if 41.4 <= Q.pt_7 < 55.9:
        z += 0.0585 * Q.pt_7 - 2.4219
    if Q.pt_7 >= 55.9:
        z += -0.0625 * Q.pt_7 + 4.342
    if Q.sj3_dr_min >= 0.022:
        z += -3.94 * Q.sj3_dr_min + 0.08668
    if 393.0 <= Q.sum_pt_top5 < 494.0:
        z += 0.004 * Q.sum_pt_top5 - 1.572
    if Q.sum_pt_top5 >= 494.0:
        z += -0.00482 * Q.sum_pt_top5 + 2.78508
    if Q.girth < 0.0951 and Q.pair_mass_0_6 > 5.5:
        z += -0.562 * (0.0951 - Q.girth) * (Q.pair_mass_0_6 - 5.5)
    if Q.girth2 < 0.00863 and Q.centroid_offset > 0.0155:
        z += 11900.0 * (0.00863 - Q.girth2) * (Q.centroid_offset - 0.0155)
    if Q.lam1 < 0.00776 and Q.n_pt_above_50 > 5.39:
        z += -78.1 * (0.00776 - Q.lam1) * (Q.n_pt_above_50 - 5.39)
    if Q.lam1 < 0.0112 and Q.planar_flow < 0.284:
        z += -298.0 * (0.0112 - Q.lam1) * (0.284 - Q.planar_flow)
    if Q.log_sum_pt > 6.6 and Q.D2 < 1.65:
        z += 4.31 * (Q.log_sum_pt - 6.6) * (1.65 - Q.D2)
    if Q.log_sum_pt > 6.33 and Q.centroid_offset > 0.00512:
        z += 29.5 * (Q.log_sum_pt - 6.33) * (Q.centroid_offset - 0.00512)
    if Q.log_sum_pt > 6.31 and Q.z_dr_0p05_0p1 < 0.824:
        z += -3.14 * (Q.log_sum_pt - 6.31) * (0.824 - Q.z_dr_0p05_0p1)
    if Q.log_sum_pt > 6.44 and Q.zdr_6 < 0.00846:
        z += -315.0 * (Q.log_sum_pt - 6.44) * (0.00846 - Q.zdr_6)
    if Q.mass_over_sum_pt_sq < 0.00703 and Q.centroid_offset > 0.00444:
        z += 5670.0 * (0.00703 - Q.mass_over_sum_pt_sq) * (Q.centroid_offset - 0.00444)
    if Q.pt_7 > 30.8 and Q.n_dr_0p1_0p2 > 0.533:
        z += 0.00795 * (Q.pt_7 - 30.8) * (Q.n_dr_0p1_0p2 - 0.533)
    if Q.pt_7 > 30.0 and Q.pt_6 < 52.2:
        z += -0.00389 * (Q.pt_7 - 30.0) * (52.2 - Q.pt_6)
    if Q.pt_7 > 27.5 and Q.sj2_dr > 0.139:
        z += 0.325 * (Q.pt_7 - 27.5) * (Q.sj2_dr - 0.139)
    if Q.z_7 < 0.0661 and Q.D2 < 1.67:
        z += -20.7 * (0.0661 - Q.z_7) * (1.67 - Q.D2)
    if Q.z_7 < 0.0537 and Q.z_dr_0p05_0p1 > 0.00935:
        z += -54.0 * (0.0537 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.00935)
    return max(0.0, z)


def neuron_2(Q):
    z = 3.29
    if Q.LHA >= 0.13:
        z += -9.96 * Q.LHA + 1.2948
    if Q.girth2 < 7.84e-05:
        z += 21300.0 * Q.girth2 - 1.66992
    if 6.2 <= Q.log_sum_pt < 6.83:
        z += -8.45 * Q.log_sum_pt + 52.39
    if Q.log_sum_pt >= 6.83:
        z += 4.95 * Q.log_sum_pt - 39.132
    if Q.max_dr < 0.256:
        z += -4.05 * Q.max_dr + 1.0368
    if Q.pt_7 >= 20.7:
        z += 0.123 * Q.pt_7 - 2.5461
    if Q.sum_pt < 495.0:
        z += -0.0233 * Q.sum_pt + 11.5335
    if Q.sum_pt >= 737.0:
        z += 0.0086 * Q.sum_pt - 6.3382
    if Q.width < 0.00763:
        z += -119.0 * Q.width + 0.90797
    if Q.z_7 < 0.023:
        z += 128.2 * Q.z_7 - 1.63898
    if 0.023 <= Q.z_7 < 0.0518:
        z += -43.8 * Q.z_7 + 2.31702
    if 0.0518 <= Q.z_7 < 0.0529:
        z += -93.5 * Q.z_7 + 4.89148
    if Q.z_7 >= 0.0529:
        z += -49.7 * Q.z_7 + 2.57446
    if Q.lam1 < 0.00565 and Q.pt_6 > 16.7:
        z += 5.37 * (0.00565 - Q.lam1) * (Q.pt_6 - 16.7)
    if Q.log_sum_pt > 6.91 and Q.pt_6 > 30.2:
        z += -0.271 * (Q.log_sum_pt - 6.91) * (Q.pt_6 - 30.2)
    if Q.sj3_pair_mass_max < 53.7 and Q.centroid_offset > 0.0106:
        z += -0.89 * (53.7 - Q.sj3_pair_mass_max) * (Q.centroid_offset - 0.0106)
    if Q.sum_pt > 559.0 and Q.lam2 < 0.00018:
        z += -13.8 * (Q.sum_pt - 559.0) * (0.00018 - Q.lam2)
    if Q.z_7 > 0.0316 and Q.C2_b2 < 0.00419:
        z += -3260.0 * (Q.z_7 - 0.0316) * (0.00419 - Q.C2_b2)
    return max(0.0, z)


def neuron_3(Q):
    z = 1.07
    if Q.centroid_offset < 0.0287:
        z += 0.9 * Q.centroid_offset - 0.66871
    if Q.centroid_offset >= 0.0287:
        z += -22.4 * Q.centroid_offset
    if Q.e2 < 0.048:
        z += -96.9 * Q.e2 + 4.6512
    if Q.girth >= 0.0567:
        z += 47.2 * Q.girth - 2.67624
    if Q.girth2 < 0.00831:
        z += 783.0 * Q.girth2 - 6.50673
    if Q.girth2 >= 0.01:
        z += -1030.0 * Q.girth2 + 10.3
    if Q.lam1 < 0.00789:
        z += -497.0 * Q.lam1 + 3.92133
    if Q.lam2 < 6.28e-05:
        z += 13100.0 * Q.lam2 - 0.82268
    if Q.mass < 43.0:
        z += -0.0545 * Q.mass + 2.3435
    if Q.mass >= 49.8:
        z += -0.0565 * Q.mass + 2.8137
    if Q.mass_over_sum_pt_sq < 0.0124:
        z += 337.0 * Q.mass_over_sum_pt_sq - 4.1788
    if Q.planar_flow < 0.0489:
        z += -21.5 * Q.planar_flow + 1.05135
    if 0.208 <= Q.sj3_dr_max < 0.235:
        z += 38.6 * Q.sj3_dr_max - 8.0288
    if Q.sj3_dr_max >= 0.235:
        z += 1.8 * Q.sj3_dr_max + 0.6192
    if Q.width >= 0.00866:
        z += 1020.0 * Q.width - 8.8332
    if Q.mass_over_sum_pt > 0.0741 and Q.pt_6 > 30.4:
        z += -0.757 * (Q.mass_over_sum_pt - 0.0741) * (Q.pt_6 - 30.4)
    if Q.sd_mass > 49.9 and Q.D2_b2 < 0.962:
        z += 0.0472 * (Q.sd_mass - 49.9) * (0.962 - Q.D2_b2)
    return max(0.0, z)


def neuron_4(Q):
    z = 11.4
    if Q.N2 < 0.225:
        z += -8.04 * Q.N2 + 1.809
    if Q.abseta_7 >= 0.0225:
        z += 3.71 * Q.abseta_7 - 0.083475
    if Q.centroid_offset < 0.0499:
        z += 49.6 * Q.centroid_offset - 2.47504
    if Q.e3 >= 2.03e-05:
        z += -3040.0 * Q.e3 + 0.061712
    if Q.girth >= 0.0421:
        z += -51.9 * Q.girth + 2.18499
    if Q.lam1 >= 0.00741:
        z += 528.0 * Q.lam1 - 3.91248
    if Q.lam2 < 0.000285:
        z += 5800.0 * Q.lam2 - 3.4656
    if 0.000285 <= Q.lam2 < 0.00094:
        z += 2650.0 * Q.lam2 - 2.56785
    if 0.00094 <= Q.lam2 < 0.000969:
        z += 3476.0 * Q.lam2 - 3.34429
    if Q.lam2 >= 0.000969:
        z += 826.0 * Q.lam2 - 0.77644
    if Q.log_sum_pt < 6.61:
        z += 5.35 * Q.log_sum_pt - 35.3635
    if Q.mass < 18.8:
        z += -0.0255 * Q.mass + 0.4794
    if Q.mass_over_sum_pt >= 0.0874:
        z += -117.0 * Q.mass_over_sum_pt + 10.2258
    if Q.max_dr >= 0.127:
        z += 21.3 * Q.max_dr - 2.7051
    if Q.mean_eta >= 0.0245:
        z += -43.6 * Q.mean_eta + 1.0682
    if Q.n_dr_0p1_0p2 >= 1.31:
        z += 0.177 * Q.n_dr_0p1_0p2 - 0.23187
    if Q.pt_7 < 23.6:
        z += 0.0784 * Q.pt_7 - 1.85024
    if Q.sd_mass < 27.5:
        z += -0.041 * Q.sd_mass - 4.0637
    if 27.5 <= Q.sd_mass < 77.9:
        z += 0.103 * Q.sd_mass - 8.0237
    if Q.sj2_dr >= 0.236:
        z += -12.4 * Q.sj2_dr + 2.9264
    if Q.sj3_dr12 < 0.084:
        z += -21.4 * Q.sj3_dr12 + 1.92386
    if 0.084 <= Q.sj3_dr12 < 0.0899:
        z += -44.7 * Q.sj3_dr12 + 3.88106
    if Q.sj3_dr12 >= 0.0899:
        z += -23.3 * Q.sj3_dr12 + 1.9572
    if Q.sj3_dr13 < 0.108:
        z += -24.4 * Q.sj3_dr13 + 2.6352
    if Q.sj3_dr13 >= 0.112:
        z += -23.9 * Q.sj3_dr13 + 2.6768
    if Q.sj3_dr23 < 0.102:
        z += -26.9 * Q.sj3_dr23 + 2.7438
    if Q.sj3_dr23 >= 0.103:
        z += -22.3 * Q.sj3_dr23 + 2.2969
    if Q.sj3_dr_max < 0.214:
        z += 46.2 * Q.sj3_dr_max - 5.0076
    if Q.sj3_dr_max >= 0.214:
        z += 22.8 * Q.sj3_dr_max
    if Q.sj3_pair_mass_min >= 10.5:
        z += -0.104 * Q.sj3_pair_mass_min + 1.092
    if Q.sum_pt >= 708.0:
        z += 0.00496 * Q.sum_pt - 3.51168
    if Q.sum_pt_top2 >= 262.0:
        z += -0.00212 * Q.sum_pt_top2 + 0.55544
    if Q.sum_pt_top5 < 403.0:
        z += -0.0085 * Q.sum_pt_top5 + 3.4255
    if Q.width < 0.00817:
        z += 793.0 * Q.width - 6.47881
    if Q.N2 < 0.226 and Q.mean_phi < -0.0107:
        z += 242.0 * (0.226 - Q.N2) * (-0.0107 - Q.mean_phi)
    if Q.N2 < 0.25 and Q.pt_7 < 56.1:
        z += -0.277 * (0.25 - Q.N2) * (56.1 - Q.pt_7)
    if Q.sd_mass > 42.8 and Q.centroid_offset > 0.00257:
        z += -1.26 * (Q.sd_mass - 42.8) * (Q.centroid_offset - 0.00257)
    if Q.sd_mass > 33.8 and Q.sd_zg < 0.355:
        z += -0.222 * (Q.sd_mass - 33.8) * (0.355 - Q.sd_zg)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.408
    if Q.log_sum_pt < 6.83:
        z += 1.52 * Q.log_sum_pt - 10.3816
    if Q.pt_7 >= 37.4:
        z += -0.0431 * Q.pt_7 + 1.61194
    if Q.sum_pt >= 793.0:
        z += -0.00303 * Q.sum_pt + 2.40279
    if Q.width < 0.00437:
        z += -256.0 * Q.width + 1.11872
    if Q.z_7 < 0.0295:
        z += -101.0 * Q.z_7 + 2.9795
    if Q.LHA < 0.196 and Q.log_sum_pt < 6.81:
        z += -48.4 * (0.196 - Q.LHA) * (6.81 - Q.log_sum_pt)
    if Q.girth2 < 0.00199 and Q.centroid_offset < 0.0201:
        z += 97900.0 * (0.00199 - Q.girth2) * (0.0201 - Q.centroid_offset)
    if Q.log_sum_pt > 6.67 and Q.dr_2 < 0.023:
        z += 307.0 * (Q.log_sum_pt - 6.67) * (0.023 - Q.dr_2)
    if Q.log_sum_pt > 6.87 and Q.dr_2 < 0.0421:
        z += -364.0 * (Q.log_sum_pt - 6.87) * (0.0421 - Q.dr_2)
    if Q.sum_pt_top5 > 387.0 and Q.centroid_offset > 0.00458:
        z += 0.129 * (Q.sum_pt_top5 - 387.0) * (Q.centroid_offset - 0.00458)
    if Q.sum_pt_top5 > 529.0 and Q.pt_6 < 42.0:
        z += 0.000165 * (Q.sum_pt_top5 - 529.0) * (42.0 - Q.pt_6)
    if Q.z_7 < 0.0535 and Q.mass_top5 < 71.0:
        z += 0.712 * (0.0535 - Q.z_7) * (71.0 - Q.mass_top5)
    return max(0.0, z)


def neuron_6(Q):
    z = -6.4
    if Q.C2_b2 < 0.00779:
        z += -182.0 * Q.C2_b2 + 1.02828
    if Q.C2_b2 >= 0.00779:
        z += -50.0 * Q.C2_b2
    if Q.centroid_offset < 0.0292:
        z += 23.9 * Q.centroid_offset - 0.69788
    if Q.e2_sq < 0.00247:
        z += -620.0 * Q.e2_sq + 1.5314
    if Q.lam1 < 0.00779:
        z += -1000.0 * Q.lam1 + 7.79
    if Q.lam2 < 0.0038:
        z += -459.0 * Q.lam2 + 1.7442
    if Q.log_sum_pt < 6.88:
        z += -8.78 * Q.log_sum_pt + 60.4064
    if Q.mass_over_sum_pt >= 0.0519:
        z += -19.1 * Q.mass_over_sum_pt + 0.99129
    if Q.max_dr < 0.175:
        z += 11.9 * Q.max_dr - 2.0825
    if Q.pt_6 < 31.5:
        z += -0.0782 * Q.pt_6 + 2.4633
    z += 0.163 * Q.pt_7
    if Q.sj2_dr < 0.144:
        z += -10.5 * Q.sj2_dr + 1.512
    if Q.sj3_dr_max >= 0.196:
        z += 5.91 * Q.sj3_dr_max - 1.15836
    if Q.sum_pt < 520.0:
        z += -0.00936 * Q.sum_pt + 4.8672
    if Q.width < 0.00797:
        z += 1732.0 * Q.width - 16.021
    if 0.00797 <= Q.width < 0.0123:
        z += 512.0 * Q.width - 6.2976
    if Q.z_7 < 0.0508:
        z += -139.0 * Q.z_7 + 7.2002
    if 0.0508 <= Q.z_7 < 0.0518:
        z += -258.0 * Q.z_7 + 13.2454
    if Q.z_7 >= 0.0518:
        z += -119.0 * Q.z_7 + 6.0452
    if Q.lam1 < 0.0123 and Q.mean_eta > 0.0238:
        z += 11000.0 * (0.0123 - Q.lam1) * (Q.mean_eta - 0.0238)
    if Q.lam1 < 0.0176 and Q.planar_flow < 0.3:
        z += -171.0 * (0.0176 - Q.lam1) * (0.3 - Q.planar_flow)
    if Q.tau1 < 0.0957 and Q.mean_phi2 < 0.0151:
        z += 1820.0 * (0.0957 - Q.tau1) * (0.0151 - Q.mean_phi2)
    if Q.width < 0.0185 and Q.mean_eta < -0.0217:
        z += 6640.0 * (0.0185 - Q.width) * (-0.0217 - Q.mean_eta)
    if Q.width < 0.0122 and Q.mean_phi > 0.0251:
        z += 13300.0 * (0.0122 - Q.width) * (Q.mean_phi - 0.0251)
    if Q.width < 0.0138 and Q.mean_phi < -0.0196:
        z += 8240.0 * (0.0138 - Q.width) * (-0.0196 - Q.mean_phi)
    return max(0.0, z)


def neuron_7(Q):
    z = 10.6
    if Q.girth < 0.0213:
        z += 107.0 * Q.girth - 9.4909
    if 0.0213 <= Q.girth < 0.0887:
        z += 15.5 * Q.girth - 7.54195
    if Q.girth >= 0.0887:
        z += -91.5 * Q.girth + 1.94895
    if Q.girth2 < 0.00402:
        z += 586.0 * Q.girth2 - 2.35572
    if 0.00757 <= Q.girth2 < 0.0219:
        z += -1200.0 * Q.girth2 + 9.084
    if Q.girth2 >= 0.0219:
        z += 290.0 * Q.girth2 - 23.547
    if Q.lam2 >= 0.000262:
        z += -226.0 * Q.lam2 + 0.059212
    if Q.mass >= 79.0:
        z += -0.213 * Q.mass + 16.827
    if Q.mass_over_sum_pt < 0.0689:
        z += -83.5 * Q.mass_over_sum_pt + 6.6633
    if 0.0689 <= Q.mass_over_sum_pt < 0.0798:
        z += 102.5 * Q.mass_over_sum_pt - 6.1521
    if Q.mass_over_sum_pt >= 0.0798:
        z += 186.0 * Q.mass_over_sum_pt - 12.8154
    if Q.mass_top5 >= 61.4:
        z += 0.0783 * Q.mass_top5 - 4.80762
    if Q.pt_7 < 45.1:
        z += 0.0246 * Q.pt_7 - 1.10946
    if Q.sj2_dr < 0.148:
        z += -5.3 * Q.sj2_dr - 0.2402
    if 0.148 <= Q.sj2_dr < 0.195:
        z += 21.8 * Q.sj2_dr - 4.251
    if Q.sum_pt >= 752.0:
        z += 0.00178 * Q.sum_pt - 1.33856
    if Q.tau1 >= 0.0953:
        z += 94.8 * Q.tau1 - 9.03444
    if Q.tau21_b2 < 0.0361:
        z += 44.3 * Q.tau21_b2 - 1.59923
    if Q.width < 0.00496:
        z += 756.0 * Q.width - 4.79304
    if 0.00496 <= Q.width < 0.00634:
        z += -103.0 * Q.width - 0.5324
    if Q.width >= 0.00634:
        z += -859.0 * Q.width + 4.26064
    if Q.centroid_offset < 0.0264 and Q.C2_b2 < 0.00481:
        z += -11200.0 * (0.0264 - Q.centroid_offset) * (0.00481 - Q.C2_b2)
    if Q.centroid_offset > 0.0426 and Q.n_pt_above_50 > 3.06:
        z += -36.8 * (Q.centroid_offset - 0.0426) * (Q.n_pt_above_50 - 3.06)
    if Q.lam2 < 0.000433 and Q.tau21_b2 < 0.0396:
        z += 125000.0 * (0.000433 - Q.lam2) * (0.0396 - Q.tau21_b2)
    if Q.mass_over_sum_pt > 0.0176 and Q.pt_6 < 30.9:
        z += -1.55 * (Q.mass_over_sum_pt - 0.0176) * (30.9 - Q.pt_6)
    if Q.planar_flow < 0.233 and Q.sd_mass > 46.4:
        z += 0.216 * (0.233 - Q.planar_flow) * (Q.sd_mass - 46.4)
    if Q.tau1 < 0.0638 and Q.n_dr_0p05_0p1 > 4.09:
        z += -19.6 * (0.0638 - Q.tau1) * (Q.n_dr_0p05_0p1 - 4.09)
    if Q.width > 0.00978 and Q.pt_6 < 51.3:
        z += 16.4 * (Q.width - 0.00978) * (51.3 - Q.pt_6)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.21
    if Q.C2 < 0.067:
        z += -11.4 * Q.C2 + 0.7638
    if Q.C2_b2 < 0.00134:
        z += 311.0 * Q.C2_b2 - 0.41674
    if Q.N2 >= 0.212:
        z += 1.36 * Q.N2 - 0.28832
    if Q.centroid_offset >= 0.0274:
        z += 17.5 * Q.centroid_offset - 0.4795
    if Q.dr_3 < 0.0898:
        z += -2.36 * Q.dr_3 + 0.211928
    if Q.dr_4 < 0.0954:
        z += -2.38 * Q.dr_4 + 0.227052
    if Q.e3 < 2.91e-05:
        z += 11600.0 * Q.e3 - 0.33756
    if Q.girth < 0.0606:
        z += 164.0 * Q.girth - 9.9384
    if Q.girth2 < 0.00431:
        z += -2320.0 * Q.girth2 + 10.6222
    if 0.00431 <= Q.girth2 < 0.0052:
        z += -700.0 * Q.girth2 + 3.64
    if Q.girth2 >= 0.0197:
        z += -38.1 * Q.girth2 + 0.75057
    if Q.girth2_top5 < 0.00033:
        z += 1966.0 * Q.girth2_top5 - 1.6103
    if 0.00033 <= Q.girth2_top5 < 0.0074:
        z += 136.0 * Q.girth2_top5 - 1.0064
    if Q.girth2_top5 >= 0.00926:
        z += -18.5 * Q.girth2_top5 + 0.17131
    if Q.lam1 < 0.00424:
        z += 1264.0 * Q.lam1 - 6.81928
    if 0.00424 <= Q.lam1 < 0.00732:
        z += 474.0 * Q.lam1 - 3.46968
    if Q.lam2 < 0.000166:
        z += -3386.0 * Q.lam2 - 0.02074
    if 0.000166 <= Q.lam2 < 0.0011:
        z += 624.0 * Q.lam2 - 0.6864
    if Q.log_sum_pt < 6.84:
        z += 2.39 * Q.log_sum_pt - 16.3476
    if Q.mass < 28.4:
        z += 0.0538 * Q.mass - 1.52792
    if Q.mass >= 41.1:
        z += -0.00663 * Q.mass + 0.272493
    if Q.mass_over_sum_pt < 0.0326:
        z += 66.3 * Q.mass_over_sum_pt - 2.16138
    if 0.0621 <= Q.mass_over_sum_pt < 0.0851:
        z += 59.8 * Q.mass_over_sum_pt - 3.71358
    if 0.0851 <= Q.mass_over_sum_pt < 0.151:
        z += 13.9 * Q.mass_over_sum_pt + 0.19251
    if Q.mass_over_sum_pt >= 0.151:
        z += 32.6 * Q.mass_over_sum_pt - 2.63119
    if Q.mass_over_sum_pt_sq < 0.00387:
        z += 557.0 * Q.mass_over_sum_pt_sq - 2.15559
    if Q.max_pair_mass >= 24.2:
        z += 0.0181 * Q.max_pair_mass - 0.43802
    if Q.mean_eta < -0.000531:
        z += 5.11 * Q.mean_eta + 0.00271341
    if Q.mean_eta2 < 0.00853:
        z += -17.1 * Q.mean_eta2 + 0.145863
    if Q.mean_phi < -0.0133:
        z += -1.2 * Q.mean_phi - 0.34509
    if -0.0133 <= Q.mean_phi < 0.00101:
        z += 23.0 * Q.mean_phi - 0.02323
    if Q.pt_7 < 20.5:
        z += -0.031 * Q.pt_7 + 1.5004
    if 20.5 <= Q.pt_7 < 48.4:
        z += -0.0664 * Q.pt_7 + 2.2261
    if Q.pt_7 >= 48.4:
        z += -0.0354 * Q.pt_7 + 0.7257
    if 0.145 <= Q.sj2_dr < 0.218:
        z += -3.24 * Q.sj2_dr + 0.4698
    if Q.sj2_dr >= 0.218:
        z += -0.07 * Q.sj2_dr - 0.22126
    if Q.sj3_dr_max < 0.2:
        z += -11.0 * Q.sj3_dr_max + 2.2
    if 718.0 <= Q.sum_pt < 847.0:
        z += -0.00194 * Q.sum_pt + 1.39292
    if 847.0 <= Q.sum_pt < 963.0:
        z += -0.00772 * Q.sum_pt + 6.28858
    if Q.sum_pt >= 963.0:
        z += -0.00262 * Q.sum_pt + 1.37728
    if Q.width < 0.00748:
        z += -1260.0 * Q.width + 9.4248
    if Q.z_7 < 0.0516:
        z += 39.9 * Q.z_7 - 2.10273
    if 0.0516 <= Q.z_7 < 0.0527:
        z += 72.7 * Q.z_7 - 3.79521
    if Q.z_7 >= 0.0527:
        z += 32.8 * Q.z_7 - 1.69248
    if Q.zdr_0 < 0.02:
        z += -7.85 * Q.zdr_0 + 0.157
    if Q.zdr_1 < 0.0138:
        z += -14.8 * Q.zdr_1 + 0.20424
    if Q.zdr_2 < 0.0113:
        z += -19.3 * Q.zdr_2 + 0.21809
    if Q.girth < 0.0602 and Q.width > 0.000434:
        z += 29700.0 * (0.0602 - Q.girth) * (Q.width - 0.000434)
    if Q.girth2 < 0.00526 and Q.centroid_offset > 0.00577:
        z += -29500.0 * (0.00526 - Q.girth2) * (Q.centroid_offset - 0.00577)
    if Q.girth2 < 0.0054 and Q.pt_7 < 44.1:
        z += -7.85 * (0.0054 - Q.girth2) * (44.1 - Q.pt_7)
    if Q.log_sum_pt > 6.76 and Q.girth2 < 0.000634:
        z += 6030.0 * (Q.log_sum_pt - 6.76) * (0.000634 - Q.girth2)
    if Q.mass < 30.5 and Q.centroid_offset < 0.0268:
        z += -2.2 * (30.5 - Q.mass) * (0.0268 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.0329 and Q.centroid_offset < 0.0269:
        z += 2840.0 * (0.0329 - Q.mass_over_sum_pt) * (0.0269 - Q.centroid_offset)
    if Q.sj3_dr_max < 0.135 and Q.centroid_offset > 0.0142:
        z += -731.0 * (0.135 - Q.sj3_dr_max) * (Q.centroid_offset - 0.0142)
    if Q.sj3_dr_max < 0.196 and Q.girth2 < 0.00673:
        z += -4410.0 * (0.196 - Q.sj3_dr_max) * (0.00673 - Q.girth2)
    if Q.sj3_dr_max < 0.199 and Q.width < 0.00348:
        z += 5790.0 * (0.199 - Q.sj3_dr_max) * (0.00348 - Q.width)
    if Q.tau1 < 0.0549 and Q.width < 0.00353:
        z += 21900.0 * (0.0549 - Q.tau1) * (0.00353 - Q.width)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.983
    if Q.girth2 < 0.00766:
        z += -1629.0 * Q.girth2 + 10.7606
    if 0.00766 <= Q.girth2 < 0.0158:
        z += 211.0 * Q.girth2 - 3.3338
    if Q.lam1 < 0.00718:
        z += 1060.0 * Q.lam1 - 7.6108
    if Q.lam2 >= 0.000378:
        z += 346.0 * Q.lam2 - 0.130788
    if Q.mass < 28.4:
        z += 0.182 * Q.mass - 5.1688
    if Q.mass_over_sum_pt >= 0.129:
        z += 34.6 * Q.mass_over_sum_pt - 4.4634
    if 0.153 <= Q.sj3_dr_max < 0.235:
        z += -25.0 * Q.sj3_dr_max + 3.825
    if Q.sj3_dr_max >= 0.235:
        z += 2.8 * Q.sj3_dr_max - 2.708
    if Q.mass < 58.9 and Q.centroid_offset < 0.0258:
        z += 6.72 * (58.9 - Q.mass) * (0.0258 - Q.centroid_offset)
    if Q.mass < 46.2 and Q.log_sum_pt < 6.83:
        z += 0.0859 * (46.2 - Q.mass) * (6.83 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_10(Q):
    z = -1.26
    if Q.C2_b2 >= 0.012:
        z += 30.5 * Q.C2_b2 - 0.366
    if 0.165 <= Q.LHA < 0.311:
        z += -44.1 * Q.LHA + 7.2765
    if Q.LHA >= 0.311:
        z += -69.8 * Q.LHA + 15.2692
    if Q.girth < 0.0207:
        z += -1.0 * Q.girth + 2.8773
    if Q.girth >= 0.0207:
        z += 138.0 * Q.girth
    if Q.girth2 < 0.00263:
        z += 488.0 * Q.girth2 - 1.28344
    if Q.lam1 < 0.00627:
        z += 650.0 * Q.lam1 - 4.0755
    if Q.lam2 < 0.000935:
        z += 799.0 * Q.lam2 - 1.0387
    if 0.000935 <= Q.lam2 < 0.0013:
        z += 1577.0 * Q.lam2 - 1.76613
    if 0.0013 <= Q.lam2 < 0.00398:
        z += 778.0 * Q.lam2 - 0.72743
    if Q.lam2 >= 0.00398:
        z += 161.0 * Q.lam2 + 1.72823
    if Q.sj3_pair_mass_min >= 16.3:
        z += 0.0622 * Q.sj3_pair_mass_min - 1.01386
    if Q.width < 0.00763:
        z += -656.0 * Q.width + 5.00528
    if Q.z_7 < 0.0572:
        z += 28.7 * Q.z_7 - 1.64164
    if Q.lam1 > 0.00759 and Q.D2_b2 < 0.386:
        z += -342.0 * (Q.lam1 - 0.00759) * (0.386 - Q.D2_b2)
    if Q.pt_7 < 53.4 and Q.D2 < 1.02:
        z += 0.0488 * (53.4 - Q.pt_7) * (1.02 - Q.D2)
    if Q.pt_7 < 42.7 and Q.log_sum_pt < 6.56:
        z += -0.188 * (42.7 - Q.pt_7) * (6.56 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_11(Q):
    z = -3.07
    if Q.LHA >= 0.12:
        z += 19.4 * Q.LHA - 2.328
    if Q.e2_sq < 0.00857:
        z += 1530.0 * Q.e2_sq - 13.1121
    if Q.girth >= 0.08:
        z += -57.1 * Q.girth + 4.568
    if Q.girth2 < 0.00479:
        z += 109.0 * Q.girth2 + 4.10141
    if 0.00479 <= Q.girth2 < 0.0126:
        z += -592.0 * Q.girth2 + 7.4592
    if Q.mass < 81.9:
        z += 0.0718 * Q.mass - 5.88042
    if Q.planar_flow < 0.273:
        z += -5.5 * Q.planar_flow + 1.5015
    if Q.sj3_dr_max >= 0.176:
        z += -5.33 * Q.sj3_dr_max + 0.93808
    if Q.tau1 < 0.0489:
        z += -63.6 * Q.tau1 + 3.11004
    if Q.width < 0.00907:
        z += -1740.0 * Q.width + 15.7818
    if Q.z_7 >= 0.0102:
        z += 17.7 * Q.z_7 - 0.18054
    if Q.centroid_offset < 0.0152 and Q.D2_b2 < 0.309:
        z += 458.0 * (0.0152 - Q.centroid_offset) * (0.309 - Q.D2_b2)
    if Q.centroid_offset < 0.0154 and Q.sj3_pair_mass_min < 13.9:
        z += -9.76 * (0.0154 - Q.centroid_offset) * (13.9 - Q.sj3_pair_mass_min)
    if Q.planar_flow < 0.26 and Q.e3 < 1.03e-05:
        z += -498000.0 * (0.26 - Q.planar_flow) * (1.03e-05 - Q.e3)
    if Q.sj2_dr > 0.168 and Q.lam2 < 0.00143:
        z += -8170.0 * (Q.sj2_dr - 0.168) * (0.00143 - Q.lam2)
    return max(0.0, z)


def neuron_12(Q):
    z = -3.18
    if Q.LHA >= 0.0973:
        z += 0.0665 * Q.LHA - 0.00647045
    if Q.N2 < 0.343:
        z += -2.4 * Q.N2 + 0.8232
    if 0.00369 <= Q.centroid_offset < 0.0425:
        z += 30.9 * Q.centroid_offset - 0.114021
    if Q.centroid_offset >= 0.0425:
        z += 41.6 * Q.centroid_offset - 0.568771
    if Q.dr_2 >= 0.134:
        z += 5.06 * Q.dr_2 - 0.67804
    if Q.dr_3 >= 0.133:
        z += 4.98 * Q.dr_3 - 0.66234
    if Q.dr_4 >= 0.147:
        z += 5.55 * Q.dr_4 - 0.81585
    if Q.dr_5 >= 0.152:
        z += 5.79 * Q.dr_5 - 0.88008
    if Q.e3 < 5.21e-05:
        z += 6850.0 * Q.e3 - 0.356885
    if Q.e3 >= 0.00026:
        z += 1330.0 * Q.e3 - 0.3458
    if Q.e4 < 2.84e-09:
        z += -1.62e+08 * Q.e4 + 0.46008
    if Q.girth < 0.13:
        z += 13.6 * Q.girth - 1.768
    if Q.girth >= 0.138:
        z += 18.3 * Q.girth - 2.5254
    if Q.girth2 < 0.00777:
        z += -313.0 * Q.girth2 + 2.43201
    if Q.girth2_top2 >= 0.0111:
        z += 18.0 * Q.girth2_top2 - 0.1998
    if Q.girth2_top5 < 0.00397:
        z += 101.0 * Q.girth2_top5 - 0.40097
    if Q.lam1 >= 0.00601:
        z += -124.0 * Q.lam1 + 0.74524
    if Q.log_sum_pt < 6.06:
        z += 4.26 * Q.log_sum_pt - 25.8156
    if 49.4 <= Q.mass < 102.0:
        z += 0.0555 * Q.mass - 2.7417
    if Q.mass >= 102.0:
        z += 0.1007 * Q.mass - 7.3521
    if Q.mass_over_sum_pt >= 0.15:
        z += 22.6 * Q.mass_over_sum_pt - 3.39
    if Q.mean_eta < -0.0296:
        z += -10.9 * Q.mean_eta - 0.32264
    if Q.mean_phi >= 0.0149:
        z += 11.1 * Q.mean_phi - 0.16539
    if Q.sd_mass >= 23.0:
        z += -0.00942 * Q.sd_mass + 0.21666
    if Q.sd_rg >= 0.23:
        z += 4.5 * Q.sd_rg - 1.035
    if Q.sj2_dr < 0.116:
        z += -8.13 * Q.sj2_dr + 0.62291
    if 0.116 <= Q.sj2_dr < 0.217:
        z += 3.17 * Q.sj2_dr - 0.68789
    if Q.sj3_dr_max >= 0.298:
        z += 2.72 * Q.sj3_dr_max - 0.81056
    if Q.sj3_pair_mass_min >= 11.6:
        z += -0.02 * Q.sj3_pair_mass_min + 0.232
    if Q.sum_pt >= 758.0:
        z += 0.00263 * Q.sum_pt - 1.99354
    if Q.width >= 0.0196:
        z += -4.67 * Q.width + 0.091532
    if Q.z_7 < 0.0668:
        z += 18.7 * Q.z_7 - 1.24916
    if Q.zdr_0 >= 0.0431:
        z += -28.9 * Q.zdr_0 + 1.24559
    if Q.zdr_6 >= 0.00434:
        z += 53.3 * Q.zdr_6 - 0.231322
    if Q.zdr_7 >= 0.00379:
        z += 57.0 * Q.zdr_7 - 0.21603
    if Q.mass > 76.9 and Q.log_sum_pt < 6.56:
        z += 0.0852 * (Q.mass - 76.9) * (6.56 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_13(Q):
    z = 12.7
    if Q.LHA < 0.239:
        z += -14.0 * Q.LHA
    if Q.LHA >= 0.239:
        z += -25.9 * Q.LHA + 2.8441
    if Q.centroid_offset < 0.0215:
        z += 35.4 * Q.centroid_offset - 0.7611
    if Q.e2 < 0.0694:
        z += 63.6 * Q.e2 - 4.41384
    if Q.e2_sq < 0.00482:
        z += -369.0 * Q.e2_sq + 1.77858
    if Q.lam1 < 0.00518:
        z += -652.0 * Q.lam1 + 4.26408
    if 0.00518 <= Q.lam1 < 0.00654:
        z += -1006.0 * Q.lam1 + 6.0978
    if Q.lam1 >= 0.00654:
        z += -354.0 * Q.lam1 + 1.83372
    if Q.lam2 < 0.000319:
        z += 1890.0 * Q.lam2 - 0.60291
    if Q.lam2 >= 0.00111:
        z += -314.0 * Q.lam2 + 0.34854
    if Q.mass >= 74.9:
        z += -0.0421 * Q.mass + 3.15329
    if Q.pt_6 < 23.9:
        z += 0.194 * Q.pt_6 - 4.6366
    if Q.pt_7 < 28.8:
        z += 0.161 * Q.pt_7 - 4.6368
    if Q.tau1 >= 0.153:
        z += 76.5 * Q.tau1 - 11.7045
    if Q.width < 0.00729:
        z += 657.0 * Q.width - 4.78953
    if Q.girth < 0.133 and Q.log_sum_pt < 6.83:
        z += -69.3 * (0.133 - Q.girth) * (6.83 - Q.log_sum_pt)
    if Q.girth < 0.144 and Q.pt_7 < 39.0:
        z += -1.08 * (0.144 - Q.girth) * (39.0 - Q.pt_7)
    if Q.girth < 0.152 and Q.z_7 > 0.0608:
        z += 203.0 * (0.152 - Q.girth) * (Q.z_7 - 0.0608)
    if Q.sum_pt > 1060.0 and Q.pt_6 < 56.9:
        z += -0.000297 * (Q.sum_pt - 1060.0) * (56.9 - Q.pt_6)
    if Q.sum_pt_top5 > 576.0 and Q.pt_7 < 37.7:
        z += 0.000623 * (Q.sum_pt_top5 - 576.0) * (37.7 - Q.pt_7)
    return max(0.0, z)


def neuron_14(Q):
    z = 9.01
    if Q.D2 >= 0.404:
        z += 0.191 * Q.D2 - 0.077164
    if Q.D2_b2 < 0.145:
        z += 23.4 * Q.D2_b2 - 3.393
    if Q.LHA < 0.239:
        z += 9.55 * Q.LHA - 2.28245
    if Q.centroid_offset < 0.021:
        z += 19.2 * Q.centroid_offset - 0.4032
    if Q.centroid_offset >= 0.0259:
        z += -80.7 * Q.centroid_offset + 2.09013
    if Q.e2_sq < 0.0169:
        z += 540.0 * Q.e2_sq - 9.126
    if 0.0534 <= Q.girth < 0.0905:
        z += 67.0 * Q.girth - 3.5778
    if Q.girth >= 0.0905:
        z += -60.0 * Q.girth + 7.9157
    if Q.girth2 >= 0.00937:
        z += 793.0 * Q.girth2 - 7.43041
    if Q.lam1 >= 0.00402:
        z += 326.0 * Q.lam1 - 1.31052
    if Q.mass < 60.6:
        z += 0.0572 * Q.mass - 3.46632
    if Q.mass_over_sum_pt < 0.0829:
        z += -189.0 * Q.mass_over_sum_pt + 15.6681
    if Q.mass_over_sum_pt >= 0.0916:
        z += -204.0 * Q.mass_over_sum_pt + 18.6864
    if Q.mean_phi < 0.00703:
        z += -6.88 * Q.mean_phi + 0.0483664
    if Q.planar_flow < 0.121:
        z += -11.4 * Q.planar_flow + 1.3794
    if Q.pt_7 >= 28.7:
        z += -0.0328 * Q.pt_7 + 0.94136
    if Q.sd_mass < 60.4:
        z += -0.0029 * Q.sd_mass - 0.59666
    if 60.4 <= Q.sd_mass < 75.3:
        z += 0.0518 * Q.sd_mass - 3.90054
    if 0.137 <= Q.sj2_dr < 0.205:
        z += 12.0 * Q.sj2_dr - 1.644
    if Q.sj2_dr >= 0.205:
        z += -2.1 * Q.sj2_dr + 1.2465
    if Q.sj3_dr12 >= 0.0649:
        z += 5.8 * Q.sj3_dr12 - 0.37642
    if Q.sj3_dr13 >= 0.081:
        z += 6.36 * Q.sj3_dr13 - 0.51516
    if Q.sj3_dr23 >= 0.0687:
        z += 6.07 * Q.sj3_dr23 - 0.417009
    if Q.sj3_dr_max >= 0.26:
        z += -12.5 * Q.sj3_dr_max + 3.25
    if Q.sum_pt >= 743.0:
        z += -0.00275 * Q.sum_pt + 2.04325
    if Q.tau1 >= 0.0969:
        z += 54.1 * Q.tau1 - 5.24229
    if Q.width < 0.00335:
        z += 2123.0 * Q.width - 14.3037
    if 0.00335 <= Q.width < 0.0045:
        z += 753.0 * Q.width - 9.7142
    if 0.0045 <= Q.width < 0.00723:
        z += 370.0 * Q.width - 7.9907
    if 0.00723 <= Q.width < 0.0167:
        z += -1370.0 * Q.width + 4.5895
    if Q.width >= 0.0167:
        z += -540.0 * Q.width - 9.2715
    if Q.z_7 >= 0.0363:
        z += 22.6 * Q.z_7 - 0.82038
    if Q.centroid_offset > 0.0509 and Q.C2_b2 < 0.00111:
        z += -155000.0 * (Q.centroid_offset - 0.0509) * (0.00111 - Q.C2_b2)
    if Q.planar_flow < 0.107 and Q.lam1 < 0.00746:
        z += -2430.0 * (0.107 - Q.planar_flow) * (0.00746 - Q.lam1)
    if Q.planar_flow < 0.117 and Q.sum_pt_top5 < 673.0:
        z += -0.0229 * (0.117 - Q.planar_flow) * (673.0 - Q.sum_pt_top5)
    if Q.sj2_dr > 0.121 and Q.D2_b2 < 0.142:
        z += 352.0 * (Q.sj2_dr - 0.121) * (0.142 - Q.D2_b2)
    if Q.sj2_dr > 0.185 and Q.D2_b2 < 0.136:
        z += -374.0 * (Q.sj2_dr - 0.185) * (0.136 - Q.D2_b2)
    if Q.z_dr_0p05_0p1 > 0.74 and Q.C2_b2 < 0.0243:
        z += -443.0 * (Q.z_dr_0p05_0p1 - 0.74) * (0.0243 - Q.C2_b2)
    if Q.z_dr_0p05_0p1 > 0.735 and Q.n_dr_0p2_0p4 < 1.03:
        z += 10.0 * (Q.z_dr_0p05_0p1 - 0.735) * (1.03 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 < 0.684 and Q.sum_pt < 632.0:
        z += 0.00319 * (0.684 - Q.z_dr_0p05_0p1) * (632.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_15(Q):
    z = -11.6
    if Q.D2 >= 1.29:
        z += 0.114 * Q.D2 - 0.14706
    if Q.LHA >= 0.32:
        z += 42.9 * Q.LHA - 13.728
    if Q.centroid_offset < 0.0293:
        z += 32.4 * Q.centroid_offset - 0.94932
    if Q.e2 < 0.0439:
        z += -51.4 * Q.e2 + 2.25646
    if Q.girth < 0.0228:
        z += 91.3 * Q.girth - 3.98544
    if 0.0228 <= Q.girth < 0.0882:
        z += 28.5 * Q.girth - 2.5536
    if 0.0882 <= Q.girth < 0.0896:
        z += -69.3 * Q.girth + 6.07236
    if Q.girth >= 0.0896:
        z += -97.8 * Q.girth + 8.62596
    if Q.girth2 < 0.00723:
        z += 1250.0 * Q.girth2 - 9.0375
    if Q.girth2 >= 0.018:
        z += -652.0 * Q.girth2 + 11.736
    if Q.lam1 < 0.0047:
        z += -772.0 * Q.lam1 + 3.6284
    if Q.lam1 >= 0.00706:
        z += 224.0 * Q.lam1 - 1.58144
    if Q.lam2 < 0.000915:
        z += -575.0 * Q.lam2 + 0.526125
    if Q.log_sum_pt >= 6.82:
        z += 13.5 * Q.log_sum_pt - 92.07
    if Q.mass >= 79.4:
        z += -0.0904 * Q.mass + 7.17776
    if Q.mass_over_sum_pt >= 0.0808:
        z += 117.0 * Q.mass_over_sum_pt - 9.4536
    if Q.mass_top5 >= 70.6:
        z += 0.0544 * Q.mass_top5 - 3.84064
    if Q.mean_phi < 0.0076:
        z += -4.28 * Q.mean_phi + 0.032528
    if Q.sj2_dr < 0.163:
        z += -6.0 * Q.sj2_dr + 0.2443
    if 0.163 <= Q.sj2_dr < 0.192:
        z += 25.3 * Q.sj2_dr - 4.8576
    if 0.168 <= Q.sj3_dr_max < 0.25:
        z += 6.81 * Q.sj3_dr_max - 1.14408
    if Q.sj3_dr_max >= 0.25:
        z += -0.27 * Q.sj3_dr_max + 0.62592
    if 709.0 <= Q.sum_pt_top5 < 799.0:
        z += -0.00271 * Q.sum_pt_top5 + 1.92139
    if Q.sum_pt_top5 >= 799.0:
        z += -0.0098 * Q.sum_pt_top5 + 7.5863
    if Q.tau1 >= 0.0407:
        z += 30.5 * Q.tau1 - 1.24135
    if Q.width < 0.00478:
        z += -294.0 * Q.width + 15.14292
    if 0.00478 <= Q.width < 0.0175:
        z += -1080.0 * Q.width + 18.9
    if Q.z_dr_0p05_0p1 < 0.637:
        z += -0.794 * Q.z_dr_0p05_0p1 + 0.505778
    if Q.N2 < 0.266 and Q.sum_pt_top5 > 389.0:
        z += 0.0235 * (0.266 - Q.N2) * (Q.sum_pt_top5 - 389.0)
    if Q.N2 < 0.277 and Q.z_dr_0p05_0p1 < 0.62:
        z += -7.5 * (0.277 - Q.N2) * (0.62 - Q.z_dr_0p05_0p1)
    if Q.girth2 < 0.00804 and Q.D2 < 0.856:
        z += -832.0 * (0.00804 - Q.girth2) * (0.856 - Q.D2)
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
