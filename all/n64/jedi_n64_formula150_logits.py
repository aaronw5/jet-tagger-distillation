"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the true labels (step 4; all observables), as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.6% (the network: 81.1%); same class as the network for 91.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.eta_1                  Δη of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        z_top5=sum(zs[:5]),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        z_top2_slots=sum(pt[:2]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        eta_1=eta[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam2=lam2,
        tau1=tau_n(1),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    z = 1.58
    if Q.e2_sq < 0.00614:
        z += 662.0 * Q.e2_sq - 4.06468
    if Q.girth2_top20 < 0.00639:
        z += 204.0 * Q.girth2_top20 - 1.30356
    if 78.9 <= Q.mass < 88.9:
        z += -0.161 * Q.mass + 12.7029
    if Q.mass >= 88.9:
        z += -0.264 * Q.mass + 21.8596
    if Q.mass_over_sum_pt_sq < 0.00498:
        z += -860.0 * Q.mass_over_sum_pt_sq + 5.9595
    if 0.00498 <= Q.mass_over_sum_pt_sq < 0.00768:
        z += -621.0 * Q.mass_over_sum_pt_sq + 4.76928
    if Q.sum_pt < 1010.0:
        z += 0.0101 * Q.sum_pt - 10.201
    if Q.sum_pt_top40 < 852.0:
        z += -0.0176 * Q.sum_pt_top40 + 14.9952
    if Q.tau1 < 0.0695:
        z += 26.2 * Q.tau1 - 1.8209
    if Q.z_top50_slots < 0.994:
        z += 21.4 * Q.z_top50_slots - 21.2716
    return max(0.0, z)


def neuron_1(Q):
    z = 3.08
    if Q.M3 < 0.0312:
        z += 46.1 * Q.M3 - 1.43832
    if Q.girth2_top3 < 0.000549:
        z += -1590.0 * Q.girth2_top3 + 0.87291
    if Q.log_sum_pt < 6.9:
        z += 10.9 * Q.log_sum_pt - 78.044
    if 6.9 <= Q.log_sum_pt < 6.99:
        z += 31.2 * Q.log_sum_pt - 218.114
    if 6.99 <= Q.log_sum_pt < 7.16:
        z += 12.8 * Q.log_sum_pt - 89.498
    if Q.log_sum_pt >= 7.16:
        z += 1.9 * Q.log_sum_pt - 11.454
    if Q.n_dr_0p2_0p4 < 6.23:
        z += 0.11 * Q.n_dr_0p2_0p4 - 0.6853
    if Q.n_particles >= 38.5:
        z += 0.118 * Q.n_particles - 4.543
    if Q.sj3_mass1 < 31.5:
        z += 0.0252 * Q.sj3_mass1 - 0.7938
    if Q.sum_pt < 1010.0:
        z += -0.019 * Q.sum_pt + 19.19
    if Q.sum_pt_top2 < 699.0:
        z += -0.00447 * Q.sum_pt_top2 + 3.12453
    if Q.mass_top20 < 51.1 and Q.n_real_top40 > 27.7:
        z += 0.00307 * (51.1 - Q.mass_top20) * (Q.n_real_top40 - 27.7)
    if Q.n_particles > 35.1 and Q.dr_0 < 0.127:
        z += 0.414 * (Q.n_particles - 35.1) * (0.127 - Q.dr_0)
    if Q.n_particles > 37.8 and Q.soft1_pt < 2.32:
        z += -0.0425 * (Q.n_particles - 37.8) * (2.32 - Q.soft1_pt)
    if Q.sj3_mass1 < 32.3 and Q.sj3_mass2 < 18.5:
        z += -0.0029 * (32.3 - Q.sj3_mass1) * (18.5 - Q.sj3_mass2)
    if Q.sum_pt_top2 < 706.0 and Q.tau43 < 0.959:
        z += -0.0075 * (706.0 - Q.sum_pt_top2) * (0.959 - Q.tau43)
    if Q.z_top30_slots > 0.947 and Q.max_pair_mass > 2.55:
        z += 0.538 * (Q.z_top30_slots - 0.947) * (Q.max_pair_mass - 2.55)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.0765
    if Q.log_sum_pt > 6.93 and Q.girth2_top15 < 0.0271:
        z += 200.0 * (Q.log_sum_pt - 6.93) * (0.0271 - Q.girth2_top15)
    if Q.sum_pt_top20 > 1130.0 and Q.eta_1 > 0.0866:
        z += -6.94 * (Q.sum_pt_top20 - 1130.0) * (Q.eta_1 - 0.0866)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.0448
    if Q.lam1 < 0.00144:
        z += 621.0 * Q.lam1 - 0.89424
    if Q.n_particles < 51.5:
        z += 0.0391 * Q.n_particles - 2.01365
    if Q.sj2_mass1 < 26.6:
        z += -0.0385 * Q.sj2_mass1 + 1.0241
    if Q.tau21 < 0.384:
        z += 5.86 * Q.tau21 - 2.25024
    if Q.tau4 < 0.0166:
        z += -146.0 * Q.tau4 + 2.4236
    if Q.D2 < 2.43 and Q.psi_0p3 > 0.999:
        z += 615.0 * (2.43 - Q.D2) * (Q.psi_0p3 - 0.999)
    if Q.n_particles < 53.2 and Q.sum_pt_top30 > 762.0:
        z += 0.000126 * (53.2 - Q.n_particles) * (Q.sum_pt_top30 - 762.0)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.01
    if Q.girth2_top15 < 0.00506:
        z += 171.0 * Q.girth2_top15 - 0.86526
    if Q.mass < 73.7:
        z += 0.0415 * Q.mass - 3.02446
    if 73.7 <= Q.mass < 78.2:
        z += 0.1645 * Q.mass - 12.08956
    if 78.2 <= Q.mass < 87.4:
        z += 0.0185 * Q.mass - 0.67236
    if 87.4 <= Q.mass < 104.0:
        z += -0.0569 * Q.mass + 5.9176
    if Q.mass_top30 < 59.7:
        z += -0.0476 * Q.mass_top30 + 2.84172
    if Q.n_particles >= 22.4:
        z += -0.0248 * Q.n_particles + 0.55552
    if Q.sj3_pair_mass_min >= 33.1:
        z += 0.0282 * Q.sj3_pair_mass_min - 0.93342
    return max(0.0, z)


def neuron_5(Q):
    z = 0.0908
    if Q.girth2_top30 >= 0.00755:
        z += -180.0 * Q.girth2_top30 + 1.359
    if Q.log_sum_pt >= 6.91:
        z += -26.0 * Q.log_sum_pt + 179.66
    if Q.mass >= 70.6:
        z += 0.018 * Q.mass - 1.2708
    if Q.mass_top50 >= 154.0:
        z += -0.133 * Q.mass_top50 + 20.482
    if Q.n_particles < 57.5:
        z += -0.036 * Q.n_particles + 2.07
    if 71.9 <= Q.sd_mass < 88.0:
        z += 0.0472 * Q.sd_mass - 3.39368
    if Q.sd_mass >= 88.0:
        z += -0.0135 * Q.sd_mass + 1.94792
    if Q.sum_pt_top40 >= 1020.0:
        z += -0.0117 * Q.sum_pt_top40 + 11.934
    if Q.sum_pt_top50 >= 934.0:
        z += 0.0287 * Q.sum_pt_top50 - 26.8058
    if Q.tau21 < 0.524:
        z += 2.56 * Q.tau21 - 1.34144
    if Q.z_top30_slots >= 0.898:
        z += -9.04 * Q.z_top30_slots + 8.11792
    return max(0.0, z)


def neuron_6(Q):
    z = 1.15
    if Q.LHA >= 0.37:
        z += -36.6 * Q.LHA + 13.542
    if Q.e2 >= 0.0274:
        z += -34.4 * Q.e2 + 0.94256
    if Q.girth2_top20 >= 0.00754:
        z += 286.0 * Q.girth2_top20 - 2.15644
    if Q.mass < 92.6:
        z += 0.01 * Q.mass - 2.1272
    if 92.6 <= Q.mass < 101.0:
        z += 0.143 * Q.mass - 14.443
    if Q.mass_top15 < 92.1:
        z += -0.0107 * Q.mass_top15 + 0.98547
    if Q.mass_top50 < 72.6:
        z += -0.0375 * Q.mass_top50 + 2.7225
    if Q.sj3_dr_min >= 0.119:
        z += -12.2 * Q.sj3_dr_min + 1.4518
    if Q.e2 > 0.0572 and Q.zdr_0 > 0.00437:
        z += 3630.0 * (Q.e2 - 0.0572) * (Q.zdr_0 - 0.00437)
    if Q.girth2_top20 > 0.00729 and Q.C2_b2 > -0.000102:
        z += -6610.0 * (Q.girth2_top20 - 0.00729) * (Q.C2_b2 - -0.000102)
    if Q.sj3_dr_min > 0.112 and Q.sj3_mass2 < 7.8:
        z += 2.21 * (Q.sj3_dr_min - 0.112) * (7.8 - Q.sj3_mass2)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.241
    if Q.sd_mass < 69.8:
        z += -0.0091 * Q.sd_mass + 0.21701
    if 69.8 <= Q.sd_mass < 85.7:
        z += 0.0263 * Q.sd_mass - 2.25391
    if Q.tau1 < 0.106:
        z += 17.4 * Q.tau1 - 1.8444
    if Q.tau21 < 0.394:
        z += 4.83 * Q.tau21 - 1.90302
    if Q.tau21_b2 < 0.248:
        z += -8.91 * Q.tau21_b2 + 2.20968
    if Q.mass < 83.1 and Q.psi_0p3 > 0.998:
        z += -136.0 * (83.1 - Q.mass) * (Q.psi_0p3 - 0.998)
    if Q.mass < 91.4 and Q.psi_0p3 > 0.975:
        z += -14.7 * (91.4 - Q.mass) * (Q.psi_0p3 - 0.975)
    if Q.mass < 93.8 and Q.psi_0p3 > 0.967:
        z += -6.9 * (93.8 - Q.mass) * (Q.psi_0p3 - 0.967)
    if Q.mass < 98.2 and Q.psi_0p3 > 0.997:
        z += 26.7 * (98.2 - Q.mass) * (Q.psi_0p3 - 0.997)
    if Q.mass < 102.0 and Q.psi_0p3 > 0.977:
        z += 12.4 * (102.0 - Q.mass) * (Q.psi_0p3 - 0.977)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.659
    if 73.3 <= Q.mass < 104.0:
        z += 0.063 * Q.mass - 4.6179
    if Q.mass >= 104.0:
        z += -0.0084 * Q.mass + 2.8077
    if Q.n_dr_0p2_0p4 < 14.9:
        z += 0.0851 * Q.n_dr_0p2_0p4 - 1.26799
    if Q.n_particles >= 50.1:
        z += 0.07 * Q.n_particles - 3.507
    if Q.psi_0p3 >= 0.994:
        z += -118.0 * Q.psi_0p3 + 117.292
    if Q.sum_pt < 1030.0:
        z += -0.00662 * Q.sum_pt + 6.8186
    if Q.mass_over_sum_pt > 0.0758 and Q.sum_pt < 1120.0:
        z += 0.129 * (Q.mass_over_sum_pt - 0.0758) * (1120.0 - Q.sum_pt)
    if Q.n_particles > 50.0 and Q.soft1_pt < 1.05:
        z += -0.0676 * (Q.n_particles - 50.0) * (1.05 - Q.soft1_pt)
    return max(0.0, z)


def neuron_9(Q):
    z = 1.28
    if Q.girth2_top40 < 0.00598:
        z += -343.0 * Q.girth2_top40 + 2.05114
    if Q.mass < 64.7:
        z += -0.0033 * Q.mass + 0.36347
    if 64.7 <= Q.mass < 85.1:
        z += -0.0842 * Q.mass + 5.5977
    if 85.1 <= Q.mass < 136.0:
        z += 0.0308 * Q.mass - 4.1888
    if Q.sj3_dr13 >= 0.161:
        z += -3.22 * Q.sj3_dr13 + 0.51842
    if Q.sum_pt < 984.0:
        z += -0.0162 * Q.sum_pt + 15.9408
    if Q.z_top5 >= 0.805:
        z += 6.83 * Q.z_top5 - 5.49815
    if Q.mass_top40 < 84.5 and Q.sum_pt < 1030.0:
        z += -0.000222 * (84.5 - Q.mass_top40) * (1030.0 - Q.sum_pt)
    if Q.sum_pt_top40 < 960.0 and Q.soft3_pt > 2.92:
        z += 0.00693 * (960.0 - Q.sum_pt_top40) * (Q.soft3_pt - 2.92)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.62
    if Q.e2 >= 0.0101:
        z += 41.0 * Q.e2 - 0.4141
    if Q.e3 < 0.000117:
        z += -8130.0 * Q.e3 + 0.95121
    if Q.mass < 89.3:
        z += -0.0573 * Q.mass + 4.2698
    if 89.3 <= Q.mass < 102.0:
        z += 0.0667 * Q.mass - 6.8034
    if Q.mass >= 148.0:
        z += -0.157 * Q.mass + 23.236
    if Q.mass_top15 < 70.6:
        z += 0.0235 * Q.mass_top15 - 1.6591
    if Q.mass_top5 < 62.6:
        z += 0.00941 * Q.mass_top5 - 0.589066
    if 138.0 <= Q.mass_top50 < 173.0:
        z += 0.105 * Q.mass_top50 - 14.49
    if Q.mass_top50 >= 173.0:
        z += -0.011 * Q.mass_top50 + 5.578
    if Q.n_dr_0p2_0p4 < 11.9:
        z += 0.0511 * Q.n_dr_0p2_0p4 - 0.60809
    if Q.sj2_mass1 < 67.4:
        z += 0.0103 * Q.sj2_mass1 - 0.69422
    if Q.sum_pt_top50 < 967.0:
        z += 0.00409 * Q.sum_pt_top50 - 3.95503
    if Q.z_dr_0_0p05 >= 0.769:
        z += -10.8 * Q.z_dr_0_0p05 + 8.3052
    if Q.z_dr_0p2_0p4 < 0.0949:
        z += -8.57 * Q.z_dr_0p2_0p4 + 0.813293
    if Q.z_top2_slots < 0.298:
        z += -3.02 * Q.z_top2_slots + 0.96338
    if 0.298 <= Q.z_top2_slots < 0.319:
        z += -6.18 * Q.z_top2_slots + 1.90506
    if Q.z_top2_slots >= 0.319:
        z += -3.16 * Q.z_top2_slots + 0.94168
    if Q.girth2_top10 < 0.0201 and Q.psi_0p3 > 0.998:
        z += -16500.0 * (0.0201 - Q.girth2_top10) * (Q.psi_0p3 - 0.998)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.605
    if Q.e2 < 0.0242:
        z += -86.3 * Q.e2 + 2.08846
    if Q.lam1 < 0.00632:
        z += 304.0 * Q.lam1 - 1.92128
    if Q.mass < 81.2:
        z += 0.0108 * Q.mass + 0.23179
    if 81.2 <= Q.mass < 93.7:
        z += -0.0887 * Q.mass + 8.31119
    if Q.psi_0p2 >= 0.994:
        z += 150.0 * Q.psi_0p2 - 149.1
    if Q.psi_0p3 >= 0.99:
        z += 80.3 * Q.psi_0p3 - 79.497
    if Q.mass < 115.0 and Q.planar_flow < 0.351:
        z += 0.115 * (115.0 - Q.mass) * (0.351 - Q.planar_flow)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.402
    if Q.mass < 60.9:
        z += -0.008 * Q.mass + 2.9482
    if 60.9 <= Q.mass < 82.3:
        z += -0.115 * Q.mass + 9.4645
    if Q.mass >= 162.0:
        z += -0.0599 * Q.mass + 9.7038
    if Q.mass < 92.5 and Q.psi_0p3 < 0.999:
        z += -3.83 * (92.5 - Q.mass) * (0.999 - Q.psi_0p3)
    if Q.sd_mass > 128.0 and Q.lam2 > 0.000638:
        z += -18.5 * (Q.sd_mass - 128.0) * (Q.lam2 - 0.000638)
    if Q.sd_mass > 128.0 and Q.sd_zg < 0.422:
        z += 0.379 * (Q.sd_mass - 128.0) * (0.422 - Q.sd_zg)
    if Q.sj3_pair_mass_max > 122.0 and Q.sj3_pair_mass_min < 80.6:
        z += 0.00116 * (Q.sj3_pair_mass_max - 122.0) * (80.6 - Q.sj3_pair_mass_min)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.58
    if 142.0 <= Q.mass < 160.0:
        z += -0.12 * Q.mass + 17.04
    if 160.0 <= Q.mass < 172.0:
        z += -0.278 * Q.mass + 42.32
    if Q.mass >= 172.0:
        z += -0.086 * Q.mass + 9.296
    if 99.2 <= Q.mass_top50 < 137.0:
        z += -0.0205 * Q.mass_top50 + 2.0336
    if Q.mass_top50 >= 137.0:
        z += 0.1095 * Q.mass_top50 - 15.7764
    if Q.sum_pt < 1010.0:
        z += 0.0304 * Q.sum_pt - 31.52
    if 1010.0 <= Q.sum_pt < 1090.0:
        z += 0.0102 * Q.sum_pt - 11.118
    if Q.sum_pt_top40 < 1020.0:
        z += -0.00816 * Q.sum_pt_top40 + 8.3232
    if Q.log_sum_pt > 7.14 and Q.sd_zg > 0.446:
        z += 1620.0 * (Q.log_sum_pt - 7.14) * (Q.sd_zg - 0.446)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.714
    if Q.girth2_top5 < 0.006:
        z += -76.3 * Q.girth2_top5 + 0.4578
    if Q.log_sum_pt < 6.94:
        z += -20.0 * Q.log_sum_pt + 138.8
    if Q.mass < 82.7:
        z += 0.241 * Q.mass - 20.8784
    if 82.7 <= Q.mass < 90.8:
        z += 0.117 * Q.mass - 10.6236
    if Q.mass_over_sum_pt < 0.0775:
        z += 32.1 * Q.mass_over_sum_pt - 0.91635
    if 0.0775 <= Q.mass_over_sum_pt < 0.118:
        z += -38.8 * Q.mass_over_sum_pt + 4.5784
    if Q.psi_0p3 >= 0.993:
        z += 174.0 * Q.psi_0p3 - 172.782
    if Q.sum_pt_top50 < 973.0:
        z += 0.0264 * Q.sum_pt_top50 - 25.6872
    if Q.tau1 < 0.0657:
        z += -61.5 * Q.tau1 + 4.04055
    if Q.N2 < 0.455 and Q.max_dr > 0.244:
        z += -8.51 * (0.455 - Q.N2) * (Q.max_dr - 0.244)
    if Q.psi_0p3 > 0.993 and Q.sd_rg < 0.235:
        z += -676.0 * (Q.psi_0p3 - 0.993) * (0.235 - Q.sd_rg)
    if Q.tau21_b2 < 0.331 and Q.girth2_top40 < 0.00769:
        z += -1470.0 * (0.331 - Q.tau21_b2) * (0.00769 - Q.girth2_top40)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.268
    if Q.girth2_top10 < 0.0102:
        z += -151.0 * Q.girth2_top10 + 1.5402
    if Q.psi_0p3 >= 0.989:
        z += -57.5 * Q.psi_0p3 + 56.8675
    if Q.tau1 < 0.0674:
        z += 22.4 * Q.tau1 - 1.50976
    if Q.z_dr_0p1_0p2 < 0.142:
        z += -6.5 * Q.z_dr_0p1_0p2 + 0.923
    if Q.sj2_dr > 0.225 and Q.C2_b2 < 0.0397:
        z += 392.0 * (Q.sj2_dr - 0.225) * (0.0397 - Q.C2_b2)
    if Q.z_dr_0p1_0p2 < 0.139 and Q.n_dr_0p2_0p4 > 5.49:
        z += -0.433 * (0.139 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.49)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -1.078125 + 0.625 * h1 - 0.75 * h3 - 0.875 * h4 - 0.3125 * h5 + 0.15625 * h6 + 0.03125 * h8 + 0.515625 * h9 - 0.015625 * h10 + 0.234375 * h12


def logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 1.359375 - 0.1875 * h1 + 0.125 * h2 - 1.0625 * h4 + 0.21875 * h6 + 0.015625 * h8 + 0.5625 * h9 - 0.015625 * h10 - 0.25 * h11 + 0.34375 * h12


def logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.09375 + 0.75 * h0 + 0.4375 * h3 + 0.34375 * h4 + 0.578125 * h5 + 0.0625 * h6 - 0.625 * h7 - 0.875 * h8 - 0.21875 * h9 + 0.59375 * h11 - 0.40625 * h12 - 1.375 * h14 - 0.1875 * h15


def logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.984375 - 1.375 * h0 - 0.375 * h2 + 0.4375 * h3 + 0.6875 * h5 - 0.96875 * h6 + 0.90625 * h7 - 0.9375 * h8 + 0.3125 * h12 + 0.5625 * h15


def logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.78125 + 0.125 * h0 + 0.1875 * h4 - 0.46875 * h5 - 0.28125 * h7 + 0.2109375 * h8 + 0.984375 * h10 - 0.375 * h12 - 0.90625 * h13 - 0.375 * h15


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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:64] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:64] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:64] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
