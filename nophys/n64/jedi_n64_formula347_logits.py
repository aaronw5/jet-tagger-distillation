"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; no W/Z/H/t mass values offered as thresholds), as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.8% (the network: 81.1%); same class as the network for 92.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M2                     generalized ECF ratio M2 (smallest angle)
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.N3                     generalized ECF ratio N3 (small = three-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_11                  pT of particle 11 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_2                  pT share × ΔR of particle 2 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
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
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M2=ecf('g31') / max(ecf('e2'), 1e-30),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        N3=ecf('g42') / max(ecf('g31') ** 2, 1e-30),
        e3=ecf('e3'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_11=pt[11],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft5_pt=softp(5, 'pt'),
        z_11=z[11],
        z_top10_slots=sum(pt[:10]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        zdr_2=z[2] * dr[2],
        zdr_5=z[5] * dr[5],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_mass=softdrop("mass"),
        sj2_dr=subjets(2)["dr"][0],
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
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
        tau4=tau_n(4),
        tau43=tau_n(4) / max(tau_n(3), 1e-12),
    )


def neuron_0(Q):
    z = 1.290131
    if Q.mass < 74.25181:
        z += 0.02652359 * Q.mass - 2.680201
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.002129702 * Q.mass - 0.8689108
    if 78.26182 <= Q.mass < 87.36377:
        z += -0.1538042 * Q.mass + 11.33476
    if 87.36377 <= Q.mass < 91.03469:
        z += -0.3264583 * Q.mass + 26.41847
    if 91.03469 <= Q.mass < 92.85979:
        z += -0.2806937 * Q.mass + 22.25231
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.3453276 * Q.mass + 28.2542
    if Q.mass >= 101.0497:
        z += -0.3718512 * Q.mass + 30.9344
    if Q.sum_pt < 1012.673:
        z += 0.01011209 * Q.sum_pt - 10.24024
    if Q.psi_0p3 >= 0.9956185:
        z += 168.3064 * Q.psi_0p3 - 167.5689
    if Q.girth2_top20 < 0.006374178:
        z += 155.4406 * Q.girth2_top20 - 0.9082283
    if 0.006374178 <= Q.girth2_top20 < 0.007538019:
        z += -70.95295 * Q.girth2_top20 + 0.5348446
    if Q.mass_top30 < 80.24626:
        z += -0.01468743 * Q.mass_top30 + 1.178611
    if Q.girth2_top30 < 0.006363916:
        z += 135.7624 * Q.girth2_top30 - 0.6452271
    if 0.006363916 <= Q.girth2_top30 < 0.007856958:
        z += -146.5153 * Q.girth2_top30 + 1.151164
    if Q.lam1 < 0.005913555:
        z += 138.0553 * Q.lam1 - 0.8163977
    if Q.mass_over_sum_pt_sq < 0.006938798:
        z += -973.1052 * Q.mass_over_sum_pt_sq + 7.178813
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -456.5516 * Q.mass_over_sum_pt_sq + 3.594552
    if Q.tau1 < 0.0705748:
        z += 17.6065 * Q.tau1 - 1.242576
    if Q.e2_sq < 0.00616708:
        z += 581.9955 * Q.e2_sq - 3.589212
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.02931703 * Q.mass_top50 - 2.104821
    if Q.mass_top50 >= 82.04491:
        z += 0.01129647 * Q.mass_top50 - 0.6263258
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 5.359746 * Q.z_dr_0p2_0p4 - 0.4889464
    if Q.z_top50_slots < 0.9906378:
        z += 28.94799 * Q.z_top50_slots - 28.67697
    if Q.sum_pt_top50 < 1156.659:
        z += -0.002732019 * Q.sum_pt_top50 + 3.160015
    if Q.sum_pt < 1012.673 and Q.M3 < 0.03457336:
        z += -0.273688 * (1012.673 - Q.sum_pt) * (0.03457336 - Q.M3)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.699144
    if Q.n_particles >= 38.0:
        z += 0.1110718 * Q.n_particles - 4.220727
    if Q.log_sum_pt < 6.893714:
        z += 7.734848 * Q.log_sum_pt - 55.22137
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 31.79219 * Q.log_sum_pt - 221.0658
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 53.1633 * Q.log_sum_pt - 368.743
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 49.45047 * Q.log_sum_pt - 342.9043
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 25.20746 * Q.log_sum_pt - 173.459
    if Q.log_sum_pt >= 7.139296:
        z += 17.47261 * Q.log_sum_pt - 118.2376
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.01530065 * Q.sum_pt_top50 + 14.67479
    if Q.sum_pt_top2 < 689.25:
        z += -0.003618716 * Q.sum_pt_top2 + 2.4942
    if Q.mass_top20 < 47.88842:
        z += 0.04804845 * Q.mass_top20 - 2.300964
    if Q.sj3_mass1 < 32.50209:
        z += 0.03191178 * Q.sj3_mass1 - 1.0372
    if Q.sum_pt_top40 < 1069.671:
        z += -0.0032811 * Q.sum_pt_top40 + 3.509699
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.08479318 * Q.n_dr_0p2_0p4 - 0.5935523
    if Q.M3 < 0.03187688:
        z += 29.0832 * Q.M3 - 0.9270819
    if Q.mass < 89.74183:
        z += -0.001368456 * Q.mass - 0.2413894
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.03220738 * Q.mass - 3.254547
    if Q.girth2_top15 < 0.0007894752:
        z += -1444.61 * Q.girth2_top15 + 1.140484
    if Q.sum_pt < 1017.435:
        z += -0.01240808 * Q.sum_pt + 12.62441
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.005714293 * Q.sum_pt_top30 - 6.811082
    if Q.mass_top30 < 80.24626:
        z += -0.01638019 * Q.mass_top30 + 1.314449
    if Q.tau1 < 0.1751567:
        z += -9.509696 * Q.tau1 + 1.665687
    if Q.mass_top10 >= 31.33272:
        z += 0.008045675 * Q.mass_top10 - 0.2520929
    if Q.zdr_0 >= 0.001901263:
        z += -43.4562 * Q.zdr_0 + 0.08262167
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.5826408 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.004765498 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.03159573 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 269.9196 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.002724569 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.n_particles > 38.0 and Q.zdr_2 < 0.01395978:
        z += 2.302253 * (Q.n_particles - 38.0) * (0.01395978 - Q.zdr_2)
    if Q.sum_pt_top2 < 689.25 and Q.tau43 < 0.9624339:
        z += -0.007497506 * (689.25 - Q.sum_pt_top2) * (0.9624339 - Q.tau43)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.1748484
    if Q.sum_pt < 1017.435:
        z += 0.003162768 * Q.sum_pt - 3.986798
    if 1017.435 <= Q.sum_pt < 1052.889:
        z += 0.0113531 * Q.sum_pt - 12.31993
    if 1052.889 <= Q.sum_pt < 1115.723:
        z += 0.004510836 * Q.sum_pt - 5.115778
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.001172513 * Q.sum_pt - 1.391136
    if Q.sum_pt >= 1260.541:
        z += -0.001990254 * Q.sum_pt + 2.595662
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.005464577 * Q.sum_pt_top50 - 6.320655
    if Q.mass < 91.03469:
        z += 0.004062472 * Q.mass + 0.2374752
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.04229866 * Q.mass - 3.243344
    if 92.85979 <= Q.mass < 121.3913:
        z += -0.017703 * Q.mass + 2.328397
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.008010576 * Q.mass + 1.151822
    if Q.sum_pt_top30 < 996.8867:
        z += 0.009834525 * Q.sum_pt_top30 - 9.803908
    if Q.sum_pt_top20 < 926.0902:
        z += -0.004269679 * Q.sum_pt_top20 + 3.954108
    if Q.sum_pt_top40 < 1053.047:
        z += -0.005339832 * Q.sum_pt_top40 + 5.623096
    if Q.log_sum_pt > 6.903423 and Q.mass_top40 > 77.93668:
        z += -0.0282944 * (Q.log_sum_pt - 6.903423) * (Q.mass_top40 - 77.93668)
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 654.9003 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -50.88179 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.girth2_top15 < 0.02146578:
        z += -0.4347747 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 7.062574 and Q.dr_3 > 0.1042479:
        z += -192.8288 * (Q.log_sum_pt - 7.062574) * (Q.dr_3 - 0.1042479)
    if Q.sum_pt_top40 < 1041.263 and Q.sj3_pair_mass_min < 26.08605:
        z += -0.0005093501 * (1041.263 - Q.sum_pt_top40) * (26.08605 - Q.sj3_pair_mass_min)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.1813042
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.2345929 * Q.n_dr_0p2_0p4 + 1.172964
    if Q.n_particles < 46.0:
        z += 0.04334046 * Q.n_particles - 1.993661
    if Q.tau21 < 0.3861957:
        z += 4.145237 * Q.tau21 - 1.600873
    if Q.mass_over_sum_pt < 0.08665515:
        z += -10.96933 * Q.mass_over_sum_pt + 0.9505489
    if Q.mass_top50 < 79.21004:
        z += 0.01614239 * Q.mass_top50 - 1.278639
    if Q.mass < 87.36377:
        z += -0.02264908 * Q.mass + 1.978709
    if Q.girth2 < 0.009614971:
        z += 133.4898 * Q.girth2 - 1.283501
    if Q.n_dr_0p1_0p2 < 10.0:
        z += -0.08496657 * Q.n_dr_0p1_0p2 + 0.8496657
    if Q.girth2_top40 < 0.006259772:
        z += 229.8354 * Q.girth2_top40 - 1.438718
    if Q.z_dr_0p1_0p2 < 0.06472584:
        z += 8.498907 * Q.z_dr_0p1_0p2 - 0.5500989
    if Q.tau4 < 0.01626937:
        z += -184.1081 * Q.tau4 + 2.995322
    if Q.girth2_top50 < 0.008124776:
        z += -170.9969 * Q.girth2_top50 + 1.389312
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 8.458651e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 279.3914 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -26.2678 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -1164.553 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.320231
    if Q.planar_flow < 0.3036026:
        z += -1.815251 * Q.planar_flow + 0.5511148
    if Q.mass_top40 < 67.72643:
        z += -0.02747003 * Q.mass_top40 + 1.425068
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.004172085 * Q.mass_top40 - 0.7179393
    if 83.32554 <= Q.mass_top40 < 94.64253:
        z += 0.03272052 * Q.mass_top40 - 3.096753
    if Q.mass < 74.25181:
        z += -0.02353644 * Q.mass + 2.010123
    if 74.25181 <= Q.mass < 79.65241:
        z += 0.05325362 * Q.mass - 3.691678
    if 79.65241 <= Q.mass < 87.36377:
        z += 0.009803616 * Q.mass - 0.23078
    if 87.36377 <= Q.mass < 121.3913:
        z += -0.01838809 * Q.mass + 2.232154
    if Q.mass_top30 < 121.737:
        z += 0.01492449 * Q.mass_top30 - 1.816862
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.02031191 * Q.sj3_pair_mass_min - 0.6604145
    if Q.n_particles >= 22.0:
        z += -0.0295923 * Q.n_particles + 0.6510307
    if Q.girth2_top15 < 0.004855289:
        z += 248.5671 * Q.girth2_top15 - 1.206865
    if Q.mass_top15 < 57.87349:
        z += -0.01542349 * Q.mass_top15 + 0.892611
    if Q.mass_top40 < 83.32554 and Q.D2 < 6.916121:
        z += -0.00464297 * (83.32554 - Q.mass_top40) * (6.916121 - Q.D2)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.007693702 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.mass < 79.65241 and Q.lam2 < 0.003687605:
        z += -34.44664 * (79.65241 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.0329485:
        z += -283.6493 * (Q.sj2_dr - 0.2232169) * (0.0329485 - Q.C2_b2)
    if Q.mass < 101.0497 and Q.lam2 < 0.003687605:
        z += 14.8246 * (101.0497 - Q.mass) * (0.003687605 - Q.lam2)
    return max(0.0, z)


def neuron_5(Q):
    z = 1.587376
    z += -0.03551522 * Q.n_particles + 2.272974
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -10.32355 * Q.mass_over_sum_pt + 0.9339457
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 216.4008 * Q.mass_over_sum_pt - 37.80874
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.008465681 * Q.sum_pt - 7.686307
    if Q.sum_pt >= 986.0565:
        z += 0.002319324 * Q.sum_pt - 1.625651
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -17.02726 * Q.log_sum_pt + 117.6606
    if 6.920349 <= Q.log_sum_pt < 6.98945:
        z += -31.59757 * Q.log_sum_pt + 218.4922
    if Q.log_sum_pt >= 6.98945:
        z += -26.04328 * Q.log_sum_pt + 179.6708
    if Q.sum_pt_top50 >= 934.2416:
        z += 0.02182161 * Q.sum_pt_top50 - 20.38665
    if Q.sum_pt_top40 >= 1024.942:
        z += -0.01354899 * Q.sum_pt_top40 + 13.88693
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.004782319 * Q.sum_pt_top30 - 4.4628
    if 0.2404747 <= Q.max_dr < 0.4357228:
        z += -4.054803 * Q.max_dr + 0.9750777
    if Q.max_dr >= 0.4357228:
        z += 1.341394 * Q.max_dr - 1.376169
    if Q.z_top30_slots >= 0.9048492:
        z += -13.21984 * Q.z_top30_slots + 11.96196
    if Q.mass_top40 < 111.2487:
        z += 0.01584143 * Q.mass_top40 - 2.376443
    if 111.2487 <= Q.mass_top40 < 150.0144:
        z += -0.01022114 * Q.mass_top40 + 0.522983
    if Q.mass_top40 >= 150.0144:
        z += -0.02606257 * Q.mass_top40 + 2.899426
    if Q.girth2_top30 >= 0.006363916:
        z += -92.67976 * Q.girth2_top30 + 0.5898062
    if Q.mass_top50 >= 157.5448:
        z += -0.2978184 * Q.mass_top50 + 46.91974
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -485.1822 * Q.mass_over_sum_pt_sq + 14.16733
    if 69.65633 <= Q.sd_mass < 83.30647:
        z += 0.03047514 * Q.sd_mass - 2.122786
    if Q.sd_mass >= 83.30647:
        z += -0.006488951 * Q.sd_mass + 0.9565617
    if Q.tau21 < 0.5100475:
        z += 2.165932 * Q.tau21 - 1.104728
    if Q.z_11 < 0.01375115:
        z += -156.1481 * Q.z_11 + 2.147215
    if Q.pt_11 < 14.14062:
        z += 0.1408581 * Q.pt_11 - 1.991821
    if Q.mass < 64.48544:
        z += -0.01843801 * Q.mass + 1.188983
    if Q.mass >= 101.0497:
        z += 0.02141708 * Q.mass - 2.164189
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01557447 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.5705026
    if Q.mass_top50 < 71.79516:
        z += -0.05212258 * Q.mass_top50 + 3.742149
    if Q.mass < 82.85409:
        z += -0.004676255 * Q.mass - 1.123224
    if 82.85409 <= Q.mass < 89.74183:
        z += 0.02544517 * Q.mass - 3.618907
    if 89.74183 <= Q.mass < 92.85979:
        z += 0.06324326 * Q.mass - 7.010977
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1139347 * Q.mass - 11.71817
    if 101.0497 <= Q.mass < 121.3913:
        z += 0.02948052 * Q.mass - 3.184103
    if 121.3913 <= Q.mass < 172.4888:
        z += -0.007722004 * Q.mass + 1.331959
    if Q.e3 < 0.0003372339:
        z += -2796.072 * Q.e3 + 0.9429301
    if Q.D2 < 2.178951:
        z += 0.4640216 * Q.D2 - 1.01108
    if Q.lam1 < 0.007259287:
        z += -178.0424 * Q.lam1 + 0.7080048
    if 0.007259287 <= Q.lam1 < 0.01649354:
        z += 63.29214 * Q.lam1 - 1.043912
    if Q.mass_over_sum_pt < 0.09795415:
        z += 46.95896 * Q.mass_over_sum_pt - 4.599825
    if Q.e2_sq < 0.02580859:
        z += -200.8755 * Q.e2_sq + 5.184313
    if Q.girth2_top50 < 0.02550569:
        z += 78.40662 * Q.girth2_top50 - 1.999815
    if Q.girth2_top30 < 0.008376291:
        z += -30.01615 * Q.girth2_top30 - 1.029039
    if 0.008376291 <= Q.girth2_top30 < 0.02412652:
        z += 81.29808 * Q.girth2_top30 - 1.96144
    if Q.mass < 121.3913 and Q.sum_pt_top50 < 1003.544:
        z += 7.163951e-05 * (121.3913 - Q.mass) * (1003.544 - Q.sum_pt_top50)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.2129564
    if Q.tau21_b2 < 0.2352054:
        z += -9.321959 * Q.tau21_b2 + 2.192575
    if Q.girth2 < 0.007877041:
        z += 168.6935 * Q.girth2 - 1.328806
    if Q.mass_over_sum_pt < 0.1182259:
        z += -13.3668 * Q.mass_over_sum_pt + 1.580303
    if Q.mass < 78.26182:
        z += -0.106915 * Q.mass + 8.119945
    if 78.26182 <= Q.mass < 91.03469:
        z += 0.05069512 * Q.mass - 4.214908
    if 91.03469 <= Q.mass < 92.85979:
        z += 0.1608488 * Q.mass - 14.24272
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.0341726 * Q.mass + 3.866933
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.02034264 * Q.mass + 2.469419
    if Q.psi_0p3 >= 0.9980008:
        z += -1118.301 * Q.psi_0p3 + 1116.065
    if Q.tau1 < 0.08786745:
        z += 7.408558 * Q.tau1 - 1.60623
    if 0.08786745 <= Q.tau1 < 0.1072713:
        z += 49.2305 * Q.tau1 - 5.281017
    if Q.LHA < 0.2601462:
        z += -0.5943727 * Q.LHA + 0.6190922
    if 0.2601462 <= Q.LHA < 0.2845608:
        z += -8.779975 * Q.LHA + 2.748545
    if 0.2845608 <= Q.LHA < 0.3098384:
        z += -18.43814 * Q.LHA + 5.496879
    if 0.3098384 <= Q.LHA < 0.3332345:
        z += 9.230698 * Q.LHA - 3.075987
    if Q.tau21 < 0.3861957:
        z += 4.812168 * Q.tau21 - 1.858439
    if Q.mass_top20 < 70.42121:
        z += 0.007589117 * Q.mass_top20 - 0.7554286
    if 70.42121 <= Q.mass_top20 < 73.35236:
        z += 0.07539474 * Q.mass_top20 - 5.530383
    if Q.z_top20_slots >= 0.9102775:
        z += -10.26318 * Q.z_top20_slots + 9.342341
    if Q.girth2_top50 < 0.003418057:
        z += 1310.111 * Q.girth2_top50 - 4.478034
    if Q.lam1 < 0.006189818:
        z += -80.05234 * Q.lam1 - 0.298409
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 386.8683 * Q.lam1 - 3.188563
    if Q.mass_over_sum_pt_sq < 0.009595015:
        z += -444.8661 * Q.mass_over_sum_pt_sq + 4.268497
    if Q.width < 0.008190222:
        z += 386.6692 * Q.width - 3.166907
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9299135:
        z += -1.052441 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9299135)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9638082:
        z += 3.606845 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9995915:
        z += -8.315009 * (82.85409 - Q.mass) * (0.9995915 - Q.psi_0p3)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9638082:
        z += -6.606819 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.01339084 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.mass < 91.03469 and Q.psi_0p3 < 0.9973959:
        z += 3.74815 * (91.03469 - Q.mass) * (0.9973959 - Q.psi_0p3)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9973959:
        z += -99.57103 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass_over_sum_pt < 0.1182259 and Q.psi_0p3 > 0.9973959:
        z += 9467.25 * (0.1182259 - Q.mass_over_sum_pt) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 91.03469 and Q.psi_0p3 > 0.9973959:
        z += -106.2355 * (91.03469 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9973959:
        z += 110.2737 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9973959)
    if Q.mass < 121.3913 and Q.sd_mass > 76.29481:
        z += 0.003507411 * (121.3913 - Q.mass) * (Q.sd_mass - 76.29481)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.123515
    if Q.mass_over_sum_pt >= 0.07696632:
        z += 62.43621 * Q.mass_over_sum_pt - 4.805485
    if Q.sum_pt_top40 < 858.8262:
        z += -0.003108768 * Q.sum_pt_top40 + 3.113503
    if 858.8262 <= Q.sum_pt_top40 < 1001.523:
        z += -0.00731051 * Q.sum_pt_top40 + 6.722069
    if Q.sum_pt_top40 >= 1001.523:
        z += -0.004201743 * Q.sum_pt_top40 + 3.608567
    if 0.005196966 <= Q.girth2_top40 < 0.006026828:
        z += 43.62621 * Q.girth2_top40 - 0.2267239
    if 0.006026828 <= Q.girth2_top40 < 0.008031986:
        z += -120.1396 * Q.girth2_top40 + 0.7602647
    if Q.girth2_top40 >= 0.008031986:
        z += 38.2714 * Q.girth2_top40 - 0.5120905
    if Q.n_dr_0p2_0p4 < 8.0:
        z += 0.09117773 * Q.n_dr_0p2_0p4 - 1.367666
    if 8.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.1361073 * Q.n_dr_0p2_0p4 - 1.727103
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.04492959 * Q.n_dr_0p2_0p4 - 0.3594367
    if 64.48544 <= Q.mass < 80.78464:
        z += 0.02949538 * Q.mass - 1.902023
    if 80.78464 <= Q.mass < 89.74183:
        z += 0.0843008 * Q.mass - 6.329459
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.1070811 * Q.mass - 8.373805
    if 101.0497 <= Q.mass < 121.3913:
        z += -0.01167287 * Q.mass + 3.626249
    if 121.3913 <= Q.mass < 143.7876:
        z += -0.05036187 * Q.mass + 8.322757
    if Q.mass >= 143.7876:
        z += -0.07804109 * Q.mass + 12.30269
    if Q.log_sum_pt < 6.935549:
        z += -9.75284 * Q.log_sum_pt + 67.6413
    if Q.log_sum_pt >= 6.959294:
        z += 3.973755 * Q.log_sum_pt - 27.65452
    if Q.girth2_top15 < 0.001319197:
        z += -114.3316 * Q.girth2_top15 + 1.21774
    if 0.001319197 <= Q.girth2_top15 < 0.005383629:
        z += -69.45597 * Q.girth2_top15 + 1.15854
    if 0.005383629 <= Q.girth2_top15 < 0.007887677:
        z += -195.6233 * Q.girth2_top15 + 1.837778
    if Q.girth2_top15 >= 0.007887677:
        z += 44.87561 * Q.girth2_top15 - 0.05919976
    if 80.35535 <= Q.mass_top50 < 97.93004:
        z += -0.06021091 * Q.mass_top50 + 4.838268
    if Q.mass_top50 >= 97.93004:
        z += -0.02307745 * Q.mass_top50 + 1.201788
    if 0.007872294 <= Q.e2_sq < 0.009606007:
        z += 341.8495 * Q.e2_sq - 2.69114
    if Q.e2_sq >= 0.009606007:
        z += -178.6727 * Q.e2_sq + 2.309001
    if Q.sum_pt_top50 < 1003.544:
        z += 0.00987398 * Q.sum_pt_top50 - 9.908975
    if Q.tau1 >= 0.06310829:
        z += -13.75792 * Q.tau1 + 0.8682387
    if Q.z_dr_0p2_0p4 < 0.1291856:
        z += -7.677561 * Q.z_dr_0p2_0p4 + 0.9918302
    if Q.mass_top40 >= 111.2487:
        z += 0.03520039 * Q.mass_top40 - 3.915997
    if Q.LHA >= 0.3332345:
        z += 16.80617 * Q.LHA - 5.600397
    if Q.e3 < 0.0003372339:
        z += -4192.388 * Q.e3 + 1.413815
    if Q.girth2_top20 < 0.02737453:
        z += 47.41452 * Q.girth2_top20 - 1.29795
    if Q.mass_over_sum_pt_sq >= 0.001785266:
        z += 99.76002 * Q.mass_over_sum_pt_sq - 0.1780982
    if Q.log_sum_pt > 6.959294 and Q.sj3_pair_mass_max > 28.35435:
        z += 0.06151402 * (Q.log_sum_pt - 6.959294) * (Q.sj3_pair_mass_max - 28.35435)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.8206395
    if Q.mass_top40 < 80.89043:
        z += -0.005023215 * Q.mass_top40 + 0.2944152
    if 80.89043 <= Q.mass_top40 < 89.6788:
        z += 0.01273443 * Q.mass_top40 - 1.142008
    if Q.girth2_top40 < 0.006259772:
        z += -411.9231 * Q.girth2_top40 + 2.578545
    if Q.mass_top30 < 121.737:
        z += 0.004281801 * Q.mass_top30 - 0.894147
    if 121.737 <= Q.mass_top30 < 152.6883:
        z += 0.01204777 * Q.mass_top30 - 1.839553
    if Q.mass < 64.48544:
        z += -0.02707681 * Q.mass + 3.837862
    if 64.48544 <= Q.mass < 79.65241:
        z += -0.1125679 * Q.mass + 9.350792
    if 79.65241 <= Q.mass < 87.36377:
        z += -0.04986 * Q.mass + 4.355958
    if 121.3913 <= Q.mass < 143.7876:
        z += 0.01829782 * Q.mass - 2.221195
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.004464298 * Q.mass + 1.051715
    if Q.mass >= 172.4888:
        z += -0.08979616 * Q.mass + 15.77051
    if Q.z_top40_slots >= 0.9574183:
        z += -19.80164 * Q.z_top40_slots + 18.95845
    if Q.sum_pt_top50 < 959.0957:
        z += -0.01182259 * Q.sum_pt_top50 + 11.339
    if Q.girth2_top20 < 0.01083435:
        z += 92.11723 * Q.girth2_top20 - 0.9980306
    if Q.tau1 < 0.05444509:
        z += 50.80126 * Q.tau1 - 2.765879
    if Q.e3 >= 0.0003372339:
        z += -4883.59 * Q.e3 + 1.646912
    if Q.LHA < 0.2091025:
        z += -19.71374 * Q.LHA + 4.122193
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0001695353 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.sum_pt_top40 < 956.2133 and Q.soft5_pt > 2.894531:
        z += 0.02130183 * (956.2133 - Q.sum_pt_top40) * (Q.soft5_pt - 2.894531)
    if Q.sum_pt_top50 < 959.0957 and Q.soft5_pt > 2.894531:
        z += -0.02446064 * (959.0957 - Q.sum_pt_top50) * (Q.soft5_pt - 2.894531)
    return max(0.0, z)


def neuron_10(Q):
    z = -3.927037
    if Q.sj2_mass1 < 80.3008:
        z += 0.04428965 * Q.sj2_mass1 - 3.556494
    if Q.mass_top40 < 67.72643:
        z += -0.004041502 * Q.mass_top40 + 2.674937
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.03341941 * Q.mass_top40 + 0.1378434
    if 83.32554 <= Q.mass_top40 < 87.27603:
        z += -0.05024303 * Q.mass_top40 + 7.109061
    if 87.27603 <= Q.mass_top40 < 111.2487:
        z += -0.01120176 * Q.mass_top40 + 3.701694
    if 111.2487 <= Q.mass_top40 < 132.4278:
        z += -0.001498595 * Q.mass_top40 + 2.62223
    if Q.mass_top40 >= 132.4278:
        z += 0.03746091 * Q.mass_top40 - 2.537094
    if Q.D2 < 3.814159:
        z += -0.4760902 * Q.D2 + 1.815884
    if Q.mass_top50 >= 157.5448:
        z += 0.07892971 * Q.mass_top50 - 12.43497
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -8.426499 * Q.z_dr_0_0p05 + 6.467115
    if Q.log_sum_pt >= 6.903423:
        z += -6.278313 * Q.log_sum_pt + 43.34185
    if Q.mass < 64.48544:
        z += -0.03023939 * Q.mass + 2.245329
    if 64.48544 <= Q.mass < 74.25181:
        z += -0.09311703 * Q.mass + 6.300022
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.06287764 * Q.mass + 4.054692
    if 82.85409 <= Q.mass < 143.7876:
        z += 0.01891669 * Q.mass - 2.722302
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.05531313 * Q.mass + 7.951027
    if Q.mass >= 172.4888:
        z += -0.2985889 * Q.mass + 49.91337
    z += 0.003245351 * Q.sum_pt
    if Q.mass_top15 < 75.26407:
        z += 0.01138367 * Q.mass_top15 - 0.8567813
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -360.7204 * Q.mass_over_sum_pt + 61.63995
    if Q.sum_pt_top5 < 839.9547:
        z += -0.003063681 * Q.sum_pt_top5 + 2.573353
    if Q.girth2_top30 < 0.01807679:
        z += -48.49059 * Q.girth2_top30 + 0.876554
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += 837.5802 * Q.mass_over_sum_pt_sq - 24.45735
    if Q.tau21_b2 < 0.6133424:
        z += -1.801406 * Q.tau21_b2 + 1.104879
    if Q.D2 < 3.814159 and Q.sj2_dr > 0.1937688:
        z += -2.506262 * (3.814159 - Q.D2) * (Q.sj2_dr - 0.1937688)
    if Q.mass > 172.4888 and Q.M2 > 0.04260132:
        z += 2.654827 * (Q.mass - 172.4888) * (Q.M2 - 0.04260132)
    if Q.sj2_mass1 < 80.3008 and Q.sj2_mass2 > 11.91979:
        z += 0.0008305667 * (80.3008 - Q.sj2_mass1) * (Q.sj2_mass2 - 11.91979)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.3048289
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.09017519 * Q.n_dr_0p2_0p4 + 0.9017519
    if Q.psi_0p2 >= 0.9935324:
        z += 103.6906 * Q.psi_0p2 - 103.0199
    if Q.mass < 64.48544:
        z += -0.02975972 * Q.mass + 3.125562
    if 64.48544 <= Q.mass < 79.65241:
        z += -0.01243256 * Q.mass + 2.008212
    if 79.65241 <= Q.mass < 92.85979:
        z += -0.07707272 * Q.mass + 7.156957
    if Q.girth < 0.05048381:
        z += 75.35613 * Q.girth - 4.755143
    if 0.05048381 <= Q.girth < 0.07374472:
        z += 40.87882 * Q.girth - 3.014597
    if Q.girth2_top10 < 0.003213724:
        z += -137.7768 * Q.girth2_top10 + 0.4427768
    if Q.lam1 < 0.006716737:
        z += 252.4746 * Q.lam1 - 1.695806
    if Q.girth2_top30 < 0.006929741:
        z += 118.473 * Q.girth2_top30 - 0.8209871
    if Q.e2_sq < 0.00818374:
        z += -324.5386 * Q.e2_sq + 2.655939
    if Q.LHA < 0.1632346:
        z += -16.70687 * Q.LHA + 3.124247
    if 0.1632346 <= Q.LHA < 0.2845608:
        z += -3.27305 * Q.LHA + 0.9313818
    if Q.n_dr_0p1_0p2 < 15.0:
        z += -0.0663157 * Q.n_dr_0p1_0p2 + 0.9947355
    if Q.mass_top50 < 85.8667:
        z += 0.04018418 * Q.mass_top50 - 3.450483
    if Q.z_top10_slots >= 0.7271951:
        z += -2.178899 * Q.z_top10_slots + 1.584485
    if Q.e2 < 0.02793599:
        z += -59.92109 * Q.e2 + 1.673955
    if Q.tau1 < 0.1072713:
        z += -8.829669 * Q.tau1 + 0.9471697
    if Q.mass < 79.65241 and Q.D2 < 3.814159:
        z += -0.0657739 * (79.65241 - Q.mass) * (3.814159 - Q.D2)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.3564799
    if Q.mass < 53.87362:
        z += -0.1040182 * Q.mass + 9.768016
    if 53.87362 <= Q.mass < 80.78464:
        z += -0.145664 * Q.mass + 12.01163
    if 80.78464 <= Q.mass < 82.85409:
        z += -0.07191464 * Q.mass + 6.053811
    if 82.85409 <= Q.mass < 87.36377:
        z += -0.0211521 * Q.mass + 1.847927
    if Q.sd_mass >= 133.2575:
        z += -0.05023374 * Q.sd_mass + 6.694024
    if Q.e2_sq < 0.00363788:
        z += 590.416 * Q.e2_sq - 2.147863
    if Q.sj3_pair_mass_max >= 122.7494:
        z += -0.009460028 * Q.sj3_pair_mass_max + 1.161213
    if Q.mass_top40 < 77.93668:
        z += 0.03205357 * Q.mass_top40 - 2.498149
    if Q.sum_pt < 972.0419:
        z += 0.005710475 * Q.sum_pt - 5.550821
    if Q.mass_over_sum_pt < 0.06895248:
        z += 16.22844 * Q.mass_over_sum_pt - 1.118991
    if Q.mass < 87.36377 and Q.psi_0p3 < 0.9989733:
        z += -5.702853 * (87.36377 - Q.mass) * (0.9989733 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 122.7494 and Q.sj2_mass1 < 65.20727:
        z += 0.001487776 * (Q.sj3_pair_mass_max - 122.7494) * (65.20727 - Q.sj2_mass1)
    if Q.sd_mass > 133.2575 and Q.sj3_mass1 < 32.50209:
        z += 0.003350507 * (Q.sd_mass - 133.2575) * (32.50209 - Q.sj3_mass1)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.762228
    if Q.sum_pt < 1012.673:
        z += 0.04108572 * Q.sum_pt - 42.07788
    if 1012.673 <= Q.sum_pt < 1085.125:
        z += 0.006507519 * Q.sum_pt - 7.061471
    if 64.48544 <= Q.mass < 78.26182:
        z += 0.0232466 * Q.mass - 1.499067
    if 78.26182 <= Q.mass < 143.7876:
        z += 0.03534306 * Q.mass - 2.445758
    if 143.7876 <= Q.mass < 172.4888:
        z += -0.1273602 * Q.mass + 20.94896
    if Q.mass >= 172.4888:
        z += -0.007722022 * Q.mass + 0.3127068
    if Q.log_sum_pt < 6.811175:
        z += -15.14465 * Q.log_sum_pt + 103.5966
    if 6.811175 <= Q.log_sum_pt < 6.879399:
        z += -6.504316 * Q.log_sum_pt + 44.74578
    if Q.sum_pt_top40 < 1053.047:
        z += -0.003602301 * Q.sum_pt_top40 + 3.793393
    if 97.93004 <= Q.mass_top50 < 138.8977:
        z += -0.03938065 * Q.mass_top50 + 3.856549
    if 138.8977 <= Q.mass_top50 < 157.5448:
        z += 0.06344659 * Q.mass_top50 - 10.42592
    if Q.mass_top50 >= 157.5448:
        z += -0.05692143 * Q.mass_top50 + 8.537441
    if 0.06030419 <= Q.mass_over_sum_pt < 0.07435617:
        z += -6.271736 * Q.mass_over_sum_pt + 0.378212
    if Q.mass_over_sum_pt >= 0.07435617:
        z += -19.8636 * Q.mass_over_sum_pt + 1.388851
    if Q.sum_pt_top50 < 1008.935:
        z += -0.01045673 * Q.sum_pt_top50 + 10.55017
    if Q.mass_top40 >= 150.0144:
        z += 0.1006837 * Q.mass_top40 - 15.10401
    if Q.sum_pt_top40 < 1053.047 and Q.D3 < 0.5260785:
        z += 0.007708099 * (1053.047 - Q.sum_pt_top40) * (0.5260785 - Q.D3)
    if Q.mass_top50 > 97.93004 and Q.soft3_pt < 2.873047:
        z += 0.005318985 * (Q.mass_top50 - 97.93004) * (2.873047 - Q.soft3_pt)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.09159952
    if Q.tau21_b2 < 0.342495:
        z += -1.872814 * Q.tau21_b2 + 0.6414293
    if Q.mass < 79.65241:
        z += 0.2759194 * Q.mass - 23.2878
    if 79.65241 <= Q.mass < 82.85409:
        z += 0.2378356 * Q.mass - 20.25433
    if 82.85409 <= Q.mass < 91.03469:
        z += 0.06707092 * Q.mass - 6.10578
    if Q.mass_over_sum_pt < 0.08873143:
        z += -9.107435 * Q.mass_over_sum_pt + 1.84003
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += 32.66425 * Q.mass_over_sum_pt - 1.866431
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -72.78529 * Q.mass_over_sum_pt + 7.673324
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -26.82068 * Q.mass_over_sum_pt + 3.170901
    if Q.girth2_top20 < 0.01083435:
        z += 80.13932 * Q.girth2_top20 - 0.8682577
    if Q.mass_over_sum_pt_sq < 0.01986381:
        z += -125.0817 * Q.mass_over_sum_pt_sq + 2.484599
    if Q.psi_0p1 >= 0.9538343:
        z += -18.53314 * Q.psi_0p1 + 17.67755
    if Q.girth2_top40 < 0.01292642:
        z += 198.9006 * Q.girth2_top40 - 2.894161
    if 0.01292642 <= Q.girth2_top40 < 0.01897915:
        z += 53.37889 * Q.girth2_top40 - 1.013086
    if Q.mass_top40 < 67.72643:
        z += -0.07385614 * Q.mass_top40 + 5.002013
    if Q.e2 < 0.01256572:
        z += 191.9648 * Q.e2 - 1.618457
    if 0.01256572 <= Q.e2 < 0.03680582:
        z += -32.74407 * Q.e2 + 1.205172
    if Q.tau2 < 0.04142826:
        z += 14.54324 * Q.tau2 - 0.6025011
    if Q.tau1 < 0.0705748:
        z += -27.49739 * Q.tau1 + 1.940623
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.007709916:
        z += -2503.318 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.girth2_top40)
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2738063:
        z += -20.53072 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2738063)
    if Q.tau21_b2 < 0.342495 and Q.C2_b2 < 0.01691783:
        z += 132.7743 * (0.342495 - Q.tau21_b2) * (0.01691783 - Q.C2_b2)
    if Q.psi_0p3 > 0.9924477 and Q.N3 < 1.059188:
        z += 104.6997 * (Q.psi_0p3 - 0.9924477) * (1.059188 - Q.N3)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.956537
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -5.010468 * Q.z_dr_0p1_0p2 + 0.6029781
    if Q.girth2_top5 < 0.002270363:
        z += 53.92532 * Q.girth2_top5 + 0.8465375
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -159.9133 * Q.girth2_top5 + 1.332029
    if Q.sum_pt < 986.0565:
        z += -0.00647135 * Q.sum_pt + 6.381117
    if Q.log_sum_pt < 6.903423:
        z += 4.112289 * Q.log_sum_pt - 27.71355
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -5.932479 * Q.log_sum_pt + 41.62974
    if Q.girth2_top10 < 0.007678544:
        z += -103.211 * Q.girth2_top10 + 0.7925103
    if Q.tau1 < 0.0705748:
        z += 28.87719 * Q.tau1 - 2.038002
    if Q.sd_mass < 40.97891:
        z += -0.003135469 * Q.sd_mass + 0.960549
    if 40.97891 <= Q.sd_mass < 79.18312:
        z += -0.0217793 * Q.sd_mass + 1.724553
    if Q.sum_pt_top30 < 1018.832:
        z += 0.003186645 * Q.sum_pt_top30 - 3.246655
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_particles < 58.0:
        z += -0.09615909 * (0.1203437 - Q.z_dr_0p1_0p2) * (58.0 - Q.n_particles)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 283.84 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.girth2_top5 < 0.008329695 and Q.n_real_top40 > 22.0:
        z += -4.092722 * (0.008329695 - Q.girth2_top5) * (Q.n_real_top40 - 22.0)
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
