"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the true labels (from 60 if-statements per neuron, pruned; all observables), as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.8% (the network: 81.1%); same class as the network for 92.3% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.dr_max_012             largest distance among the 3 hardest
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_11                  pT of particle 11 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_11                 pT share × ΔR of particle 11 (its part of the girth)
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_dr_min             smallest distance among the 3 subjet axes
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_13                  ΔR of particle 13 from the jet axis
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
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
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
        z_top5=sum(zs[:5]),
        C2=e3 / max(e2 ** 2, 1e-12),
        C2_b2=ecf('e3b2') / max(ecf('e2b2') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        dr_max_012=max(math.sqrt(dist2(0, 1)), math.sqrt(dist2(0, 2)), math.sqrt(dist2(1, 2))) if pt[2] > 0 else math.sqrt(dist2(0, 1)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_11=pt[11],
        pt_2=pt[2],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        z_11=z[11],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        zdr_11=z[11] * dr[11],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_dr_min=min(subjets(3)["dr"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_13=dr[13] if pt[13] > 0 else 0.0,
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
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
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
    z = 1.348244
    if Q.mass < 74.25181:
        z += 0.02437571 * Q.mass - 2.463158
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.007188682 * Q.mass - 1.18699
    if 78.26182 <= Q.mass < 89.74183:
        z += -0.1383323 * Q.mass + 10.20175
    if 89.74183 <= Q.mass < 91.19:
        z += -0.2325952 * Q.mass + 18.66107
    if 91.19 <= Q.mass < 92.85979:
        z += -0.2787028 * Q.mass + 22.86562
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.3641191 * Q.mass + 30.79737
    if Q.mass >= 101.0497:
        z += -0.3884949 * Q.mass + 33.26053
    if Q.girth2_top20 < 0.005312783:
        z += 119.7051 * Q.girth2_top20 - 0.9035521
    if 0.005312783 <= Q.girth2_top20 < 0.006374178:
        z += 252.1068 * Q.girth2_top20 - 1.606973
    if Q.sum_pt < 1012.673:
        z += 0.008300073 * Q.sum_pt - 8.405259
    if Q.mass_top30 < 80.4:
        z += -0.009118849 * Q.mass_top30 + 0.7331555
    if Q.girth2_top30 < 0.006363916:
        z += 154.7314 * Q.girth2_top30 - 0.8007542
    if 0.006363916 <= Q.girth2_top30 < 0.007856958:
        z += -123.2004 * Q.girth2_top30 + 0.9679807
    if Q.lam1 < 0.005913555:
        z += 151.5123 * Q.lam1 - 0.8959761
    if Q.mass_over_sum_pt_sq < 0.004754444:
        z += -1183.366 * Q.mass_over_sum_pt_sq + 8.169863
    if 0.004754444 <= Q.mass_over_sum_pt_sq < 0.006938798:
        z += -975.1202 * Q.mass_over_sum_pt_sq + 7.17977
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -442.6132 * Q.mass_over_sum_pt_sq + 3.484811
    if Q.tau1 < 0.0705748:
        z += 27.28144 * Q.tau1 - 1.925382
    if Q.e2_sq < 0.00616708:
        z += 584.1328 * Q.e2_sq - 3.602394
    if 71.79516 <= Q.mass_top50 < 82.04491:
        z += 0.0333566 * Q.mass_top50 - 2.394842
    if Q.mass_top50 >= 82.04491:
        z += -0.02173572 * Q.mass_top50 + 2.125202
    if Q.z_top50_slots < 0.9906378:
        z += 30.09608 * Q.z_top50_slots - 29.81431
    if Q.sum_pt_top40 < 858.8262:
        z += -0.009459652 * Q.sum_pt_top40 + 7.412517
    if 858.8262 <= Q.sum_pt_top40 < 1069.671:
        z += 0.003375368 * Q.sum_pt_top40 - 3.610534
    if Q.sum_pt_top20 < 846.1934:
        z += -0.003710437 * Q.sum_pt_top20 + 3.139747
    if Q.sum_pt_top50 < 1156.659:
        z += -0.004181412 * Q.sum_pt_top50 + 4.836469
    if Q.sum_pt < 1012.673 and Q.M3 < 0.03457336:
        z += -0.2622419 * (1012.673 - Q.sum_pt) * (0.03457336 - Q.M3)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.48714
    if Q.n_particles >= 38.0:
        z += 0.09328952 * Q.n_particles - 3.545002
    if Q.log_sum_pt < 6.893714:
        z += 5.901248 * Q.log_sum_pt - 42.13076
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 28.74357 * Q.log_sum_pt - 199.5992
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 49.79447 * Q.log_sum_pt - 345.0637
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 47.05241 * Q.log_sum_pt - 325.9808
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 26.31154 * Q.log_sum_pt - 181.0136
    if Q.log_sum_pt >= 7.139296:
        z += 20.41029 * Q.log_sum_pt - 138.8828
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.01273634 * Q.sum_pt_top50 + 12.21537
    if Q.sum_pt_top2 < 689.25:
        z += -0.00253462 * Q.sum_pt_top2 + 1.746987
    if Q.mass_top20 < 47.88842:
        z += 0.01626576 * Q.mass_top20 - 0.7789413
    if Q.sj3_mass1 < 32.50209:
        z += 0.02671432 * Q.sj3_mass1 - 0.8682711
    if Q.sum_pt_top40 < 1069.671:
        z += -0.005140426 * Q.sum_pt_top40 + 5.498565
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.1279552 * Q.n_dr_0p2_0p4 - 0.8956861
    if Q.girth2_top3 < 0.0005522528:
        z += -1862.292 * Q.girth2_top3 + 1.028456
    if Q.M3 < 0.03187688:
        z += 44.64202 * Q.M3 - 1.423048
    if Q.pt_9 < 31.35938:
        z += 0.02125498 * Q.pt_9 - 0.6665429
    if Q.lam1 < 0.004673423:
        z += -171.814 * Q.lam1 + 0.8029595
    if Q.mass < 120.6:
        z += 0.01121205 * Q.mass - 1.352173
    if Q.girth2_top30 < 0.02412652:
        z += -54.39146 * Q.girth2_top30 + 1.312277
    if Q.sum_pt < 1017.435:
        z += -0.009289864 * Q.sum_pt + 9.451829
    if Q.pt_entropy >= 2.07371:
        z += 0.529206 * Q.pt_entropy - 1.09742
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.5237017 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.3875843 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.003762992 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.03179651 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 160.675 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.002737062 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.sum_pt_top2 < 689.25 and Q.tau43 < 0.9624339:
        z += -0.007804931 * (689.25 - Q.sum_pt_top2) * (0.9624339 - Q.tau43)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.3078427
    if Q.sum_pt < 972.0419:
        z += 0.01516493 * Q.sum_pt - 16.07533
    if 972.0419 <= Q.sum_pt < 1017.435:
        z += 0.004625244 * Q.sum_pt - 5.830309
    if 1017.435 <= Q.sum_pt < 1115.723:
        z += 0.01412922 * Q.sum_pt - 15.49999
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.008646763 * Q.sum_pt - 9.383082
    if Q.sum_pt >= 1260.541:
        z += 0.004021519 * Q.sum_pt - 3.552773
    if Q.sum_pt_top50 < 988.4554:
        z += -0.01051345 * Q.sum_pt_top50 + 11.01913
    if 988.4554 <= Q.sum_pt_top50 < 1048.098:
        z += -0.01405501 * Q.sum_pt_top50 + 14.5198
    if Q.sum_pt_top50 >= 1048.098:
        z += -0.003541556 * Q.sum_pt_top50 + 3.50067
    if Q.z_top20_slots >= 0.7516206:
        z += 2.314562 * Q.z_top20_slots - 1.739672
    if Q.mass < 91.19:
        z += 0.04050616 * Q.mass - 3.814951
    if 91.19 <= Q.mass < 92.85979:
        z += 0.07258039 * Q.mass - 6.7398
    if Q.girth2_top30 < 0.004763596:
        z += -175.842 * Q.girth2_top30 + 0.8376403
    if Q.lam1 < 0.01174405:
        z += -74.57761 * Q.lam1 + 0.8758433
    if Q.log_sum_pt >= 6.959294:
        z += -7.395631 * Q.log_sum_pt + 51.46837
    if Q.mass_top50 < 92.16545:
        z += -0.01188559 * Q.mass_top50 + 1.09544
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.004063689 * Q.sum_pt_top40 - 4.346811
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 647.1664 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.sum_pt_top50 > 988.4554 and Q.girth2_top15 < 0.02146578:
        z += -0.4691034 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.girth2_top15)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -12.724 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.06251741
    if Q.n_particles < 46.0:
        z += 0.0393809 * Q.n_particles - 1.811522
    if Q.tau21 < 0.3861957:
        z += 6.400079 * Q.tau21 - 2.471683
    if Q.mass_over_sum_pt < 0.08665515:
        z += -24.27948 * Q.mass_over_sum_pt + 2.103942
    if Q.mass_top50 < 79.21004:
        z += 0.01364892 * Q.mass_top50 - 1.081132
    if Q.girth2 < 0.009614971:
        z += 147.3656 * Q.girth2 - 1.416916
    if Q.girth2_top40 < 0.008840538:
        z += -99.26788 * Q.girth2_top40 + 0.8775815
    if Q.mass < 92.85979:
        z += -0.01474629 * Q.mass + 1.369337
    if Q.sj2_mass1 < 27.56535:
        z += -0.0439486 * Q.sj2_mass1 + 1.211459
    if Q.mass_top40 < 80.4:
        z += 0.02451443 * Q.mass_top40 - 1.970961
    if Q.lam1 < 0.001260456:
        z += 731.6249 * Q.lam1 - 0.9221807
    if Q.tau4 < 0.01626937:
        z += -150.2475 * Q.tau4 + 2.444432
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 0.0001463702 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 480.666 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.mass_top50 < 79.21004 and Q.mass_top3 > 8.921413:
        z += -0.00500493 * (79.21004 - Q.mass_top50) * (Q.mass_top3 - 8.921413)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.367567
    if Q.mass_top40 < 67.72643:
        z += -0.0150008 * Q.mass_top40 - 0.1224861
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.02984468 * Q.mass_top40 - 3.159711
    if 83.32554 <= Q.mass_top40 < 163.2541:
        z += 0.008418597 * Q.mass_top40 - 1.374371
    if Q.mass_top30 < 60.43821:
        z += -0.03053927 * Q.mass_top30 + 1.845739
    if Q.mass < 74.25181:
        z += 0.04508401 * Q.mass - 2.593275
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.1335131 * Q.mass - 9.159294
    if 78.26182 <= Q.mass < 80.78464:
        z += 0.01888553 * Q.mass - 0.1883321
    if 80.78464 <= Q.mass < 86.4:
        z += -0.01393118 * Q.mass + 2.462754
    if 86.4 <= Q.mass < 92.85979:
        z += -0.04232384 * Q.mass + 4.91588
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.07230487 * Q.mass + 7.699912
    if 101.0497 <= Q.mass < 120.6:
        z += -0.0201289 * Q.mass + 2.427545
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.02532736 * Q.sj3_pair_mass_min - 0.8234853
    if Q.n_particles >= 22.0:
        z += -0.02472351 * Q.n_particles + 0.5439172
    if Q.girth2_top15 < 0.004855289:
        z += 282.8053 * Q.girth2_top15 - 1.373101
    if 0.00727763 <= Q.girth2_top15 < 0.01563836:
        z += -38.14918 * Q.girth2_top15 + 0.2776356
    if Q.girth2_top15 >= 0.01563836:
        z += 17.0848 * Q.girth2_top15 - 0.5861331
    if Q.mass_top15 < 57.87349:
        z += -0.01708885 * Q.mass_top15 + 0.9889915
    if Q.girth >= 0.1207452:
        z += -13.37402 * Q.girth + 1.614848
    if Q.lam1 >= 0.008241985:
        z += -149.4077 * Q.lam1 + 1.231416
    if Q.e2_sq >= 0.009606007:
        z += 167.0418 * Q.e2_sq - 1.604605
    if Q.lam2 >= 0.001776308:
        z += -143.5591 * Q.lam2 + 0.2550052
    return max(0.0, z)


def neuron_5(Q):
    z = 0.6380697
    z += -0.0381413 * Q.n_particles + 2.441043
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -10.20386 * Q.mass_over_sum_pt + 0.923118
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 157.0822 * Q.mass_over_sum_pt - 27.66275
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.006261253 * Q.sum_pt - 5.684825
    if Q.sum_pt >= 986.0565:
        z += 0.001508747 * Q.sum_pt - 0.9985849
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -15.12077 * Q.log_sum_pt + 104.4865
    if 6.920349 <= Q.log_sum_pt < 6.98945:
        z += -30.88943 * Q.log_sum_pt + 213.6111
    if Q.log_sum_pt >= 6.98945:
        z += -21.52714 * Q.log_sum_pt + 148.1739
    if Q.sum_pt_top50 >= 934.2416:
        z += 0.02016727 * Q.sum_pt_top50 - 18.8411
    if Q.sum_pt_top20 >= 1129.275:
        z += -0.005608218 * Q.sum_pt_top20 + 6.333221
    if Q.sum_pt_top40 >= 1024.942:
        z += -0.01244614 * Q.sum_pt_top40 + 12.75657
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.005879184 * Q.sum_pt_top30 - 5.486381
    if 0.2404747 <= Q.max_dr < 0.4357228:
        z += -2.518499 * Q.max_dr + 0.6056354
    if Q.max_dr >= 0.4357228:
        z += 1.324715 * Q.max_dr - 1.068941
    if Q.z_top30_slots >= 0.9048492:
        z += -14.75085 * Q.z_top30_slots + 13.3473
    if Q.mass_top40 < 120.6:
        z += 0.005410795 * Q.mass_top40 - 0.8116974
    if 120.6 <= Q.mass_top40 < 150.0144:
        z += -0.02028433 * Q.mass_top40 + 2.287135
    if Q.mass_top40 >= 150.0144:
        z += -0.02569512 * Q.mass_top40 + 3.098832
    if 74.25181 <= Q.mass < 172.8:
        z += 0.01644698 * Q.mass - 1.221218
    if Q.mass >= 172.8:
        z += 0.09448421 * Q.mass - 14.70605
    if Q.girth2_top30 >= 0.006363916:
        z += -115.5639 * Q.girth2_top30 + 0.7354392
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -511.5796 * Q.mass_over_sum_pt_sq + 14.93813
    if Q.tau21 < 0.5494307:
        z += 2.370548 * Q.tau21 - 1.302452
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.05009839 * Q.sd_mass - 3.48967
    if Q.sd_mass >= 86.4:
        z += -0.02453418 * Q.sd_mass + 2.958585
    if Q.mass_top50 >= 157.5448:
        z += -0.1307292 * Q.mass_top50 + 20.5957
    if Q.z_11 < 0.01375115:
        z += -84.69973 * Q.z_11 + 1.164718
    if Q.mass_top10 >= 71.781:
        z += 0.02046421 * Q.mass_top10 - 1.468941
    if Q.pt_11 < 14.14062:
        z += 0.06378087 * Q.pt_11 - 0.9019013
    if Q.D2 < 1.976207:
        z += -0.402659 * Q.D2 + 0.7957377
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01647099 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 29743.12 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += -55.14108 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.528864
    if Q.mass_top50 < 71.79516:
        z += -0.04046403 * Q.mass_top50 + 3.699157
    if 71.79516 <= Q.mass_top50 < 172.8:
        z += -0.007861364 * Q.mass_top50 + 1.358444
    if Q.mass < 86.4:
        z += 0.02000309 * Q.mass - 3.824134
    if 86.4 <= Q.mass < 92.85979:
        z += 0.05564607 * Q.mass - 6.903687
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.1231083 * Q.mass - 13.16821
    if 101.0497 <= Q.mass < 120.6:
        z += 0.0372454 * Q.mass - 4.491795
    if Q.log_sum_pt < 6.811175:
        z += 6.208754 * Q.log_sum_pt - 42.28891
    if Q.tau1 < 0.06310829:
        z += -17.12618 * Q.tau1 + 1.080804
    if Q.sj3_dr_min >= 0.1204829:
        z += -13.78118 * Q.sj3_dr_min + 1.660397
    if Q.girth2_top30 < 0.008376291:
        z += 31.71488 * Q.girth2_top30 - 0.5733031
    if 0.008376291 <= Q.girth2_top30 < 0.01807679:
        z += 262.8203 * Q.girth2_top30 - 2.509109
    if 0.01807679 <= Q.girth2_top30 < 0.02809026:
        z += 231.1054 * Q.girth2_top30 - 1.935806
    if Q.girth2_top30 >= 0.02809026:
        z += 348.0586 * Q.girth2_top30 - 5.221052
    if Q.girth2_top20 >= 0.008031209:
        z += 164.6479 * Q.girth2_top20 - 1.322322
    if Q.lam1 < 0.003811746:
        z += -313.3979 * Q.lam1 + 1.194593
    if Q.girth2_top50 < 0.004573744:
        z += 337.6895 * Q.girth2_top50 - 1.544505
    if Q.mass_top15 < 91.19:
        z += -0.01124433 * Q.mass_top15 + 1.025371
    if Q.LHA >= 0.3719813:
        z += -36.05405 * Q.LHA + 13.41143
    if Q.e2 >= 0.02793599:
        z += -40.531 * Q.e2 + 1.132274
    if Q.mass_over_sum_pt >= 0.09795415:
        z += -26.50031 * Q.mass_over_sum_pt + 2.595815
    if Q.mass < 120.6 and Q.sum_pt < 1007.788:
        z += 0.0001010919 * (120.6 - Q.mass) * (1007.788 - Q.sum_pt)
    if Q.mass_over_sum_pt > 0.1182259 and Q.C2_b2 > 0.02704832:
        z += 2203.189 * (Q.mass_over_sum_pt - 0.1182259) * (Q.C2_b2 - 0.02704832)
    if Q.e2 > 0.05557149 and Q.zdr_0 > 0.001369707:
        z += 3022.732 * (Q.e2 - 0.05557149) * (Q.zdr_0 - 0.001369707)
    if Q.sj3_dr_min > 0.1204829 and Q.sj3_mass2 < 7.863702:
        z += 2.275593 * (Q.sj3_dr_min - 0.1204829) * (7.863702 - Q.sj3_mass2)
    if Q.girth2_top30 > 0.008376291 and Q.D2_b2 > 1.67722:
        z += -43.46545 * (Q.girth2_top30 - 0.008376291) * (Q.D2_b2 - 1.67722)
    if Q.mass_over_sum_pt > 0.1182259 and Q.D2_b2 < 7.36624:
        z += -1.351723 * (Q.mass_over_sum_pt - 0.1182259) * (7.36624 - Q.D2_b2)
    if Q.girth2_top30 > 0.008376291 and Q.orientation_deg > 26.6454:
        z += -1.050915 * (Q.girth2_top30 - 0.008376291) * (Q.orientation_deg - 26.6454)
    if Q.girth2_top20 > 0.008031209 and Q.C2_b2 > 0.0008187529:
        z += -7945.985 * (Q.girth2_top20 - 0.008031209) * (Q.C2_b2 - 0.0008187529)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.02822393
    if Q.tau21_b2 < 0.2352054:
        z += -15.81855 * Q.tau21_b2 + 3.72061
    if Q.mass_over_sum_pt < 0.1182259:
        z += -8.27917 * Q.mass_over_sum_pt + 0.9788127
    if Q.girth2 < 0.006403325:
        z += -35.25454 * Q.girth2 + 0.9707515
    if 0.006403325 <= Q.girth2 < 0.008190222:
        z += -91.14493 * Q.girth2 + 1.328636
    if 0.008190222 <= Q.girth2 < 0.009614971:
        z += -408.5902 * Q.girth2 + 3.928583
    if Q.mass < 78.26182:
        z += 0.08108167 * Q.mass - 7.183451
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.2639736 * Q.mass - 21.49691
    if 82.85409 <= Q.mass < 101.0497:
        z += 0.02270984 * Q.mass - 1.507218
    if 101.0497 <= Q.mass < 120.6:
        z += -0.04028611 * Q.mass + 4.858505
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.07985738 * Q.n_dr_0p2_0p4 + 0.4791443
    if 0.9973959 <= Q.psi_0p3 < 0.9980008:
        z += 548.1077 * Q.psi_0p3 - 546.6803
    if Q.psi_0p3 >= 0.9980008:
        z += -1085.926 * Q.psi_0p3 + 1084.087
    if Q.tau21 < 0.3861957:
        z += 6.339537 * Q.tau21 - 2.448302
    if Q.e2_sq < 0.00363788:
        z += 817.5024 * Q.e2_sq - 2.973976
    if Q.tau1 < 0.09591084:
        z += 15.54544 * Q.tau1 - 1.975415
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += 42.6427 * Q.tau1 - 4.574336
    if Q.lam1 < 0.006189818:
        z += 6.624817 * Q.lam1 - 0.7662926
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 353.4245 * Q.lam1 - 2.91292
    if Q.sd_mass < 69.65633:
        z += -0.00902506 * Q.sd_mass + 0.06724086
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.03352977 * Q.sd_mass - 2.896972
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -2245.457 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9638082:
        z += 7.807953 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9777125:
        z += 10.95938 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9985421:
        z += 4.81802 * (82.85409 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9777125:
        z += -19.21488 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9980008:
        z += -109.9719 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9980008:
        z += -66.66523 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9980008:
        z += 123.7156 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.02163195 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.mass < 92.85979 and Q.psi_0p3 > 0.9638082:
        z += -8.922787 * (92.85979 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.girth2 < 0.006403325 and Q.psi_0p3 > 0.9980008:
        z += 152371.2 * (0.006403325 - Q.girth2) * (Q.psi_0p3 - 0.9980008)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.120228
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += -14.22432 * Q.mass_over_sum_pt + 1.094793
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -40.7828 * Q.mass_over_sum_pt + 4.234694
    if Q.girth2_top40 < 0.005196966:
        z += -132.7549 * Q.girth2_top40 + 1.173625
    if 0.005196966 <= Q.girth2_top40 < 0.008840538:
        z += -114.9723 * Q.girth2_top40 + 1.081209
    if Q.girth2_top40 >= 0.008840538:
        z += 17.78255 * Q.girth2_top40 - 0.09241532
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.0724109 * Q.n_dr_0p2_0p4 - 1.086163
    if 64.48544 <= Q.mass < 74.25181:
        z += 0.01750729 * Q.mass - 1.128965
    if 74.25181 <= Q.mass < 87.36377:
        z += 0.06525753 * Q.mass - 4.674507
    if 87.36377 <= Q.mass < 101.0497:
        z += 0.1222373 * Q.mass - 9.652471
    if 101.0497 <= Q.mass < 125.1:
        z += 0.04562924 * Q.mass - 1.911253
    if Q.mass >= 125.1:
        z += 0.008404231 * Q.mass + 2.745596
    if Q.n_particles >= 51.0:
        z += 0.05891433 * Q.n_particles - 3.004631
    if Q.sum_pt < 1017.435:
        z += -0.02705056 * Q.sum_pt + 27.7078
    if 1017.435 <= Q.sum_pt < 1028.184:
        z += -0.0172691 * Q.sum_pt + 17.75581
    if 0.006043209 <= Q.girth2_top20 < 0.008031209:
        z += -195.7614 * Q.girth2_top20 + 1.183027
    if Q.girth2_top20 >= 0.008031209:
        z += 43.44763 * Q.girth2_top20 - 0.7381108
    if Q.log_sum_pt < 6.930088:
        z += 12.99823 * Q.log_sum_pt - 90.07891
    if Q.girth2_top30 < 0.007463985:
        z += -132.4598 * Q.girth2_top30 + 0.988678
    if Q.width < 0.009614971:
        z += 648.1557 * Q.width - 6.231998
    if Q.girth2 < 0.007877041:
        z += -175.6781 * Q.girth2 + 1.383824
    if Q.n_for_90pct < 7.0:
        z += 0.03764843 * Q.n_for_90pct - 1.468289
    if 7.0 <= Q.n_for_90pct < 39.0:
        z += -0.02078587 * Q.n_for_90pct - 1.059249
    if Q.n_for_90pct >= 39.0:
        z += -0.0584343 * Q.n_for_90pct + 0.4090401
    if Q.psi_0p3 >= 0.9943058:
        z += -107.4047 * Q.psi_0p3 + 106.7932
    if Q.lam1 < 0.007671243:
        z += -189.9806 * Q.lam1 + 1.457387
    if Q.tau2 < 0.0795038:
        z += -12.06935 * Q.tau2 + 0.9595593
    if 5.13841e-05 <= Q.e3 < 0.0003372339:
        z += -1112.627 * Q.e3 + 0.05717136
    if Q.e3 >= 0.0003372339:
        z += 3812.867 * Q.e3 - 1.603872
    if 82.04491 <= Q.mass_top50 < 117.0487:
        z += -0.04839883 * Q.mass_top50 + 3.970878
    if Q.mass_top50 >= 117.0487:
        z += 0.003918115 * Q.mass_top50 - 2.152755
    if Q.mass_top20 >= 119.2969:
        z += -0.02550469 * Q.mass_top20 + 3.04263
    if Q.sum_pt_top40 < 1024.942:
        z += 0.006395085 * Q.sum_pt_top40 - 6.554594
    if Q.z_top15_slots < 0.8316924:
        z += -4.235687 * Q.z_top15_slots + 3.522789
    if Q.psi_0p1 < 0.3628388:
        z += -2.532682 * Q.psi_0p1 + 0.9189554
    if Q.mass_over_sum_pt > 0.07696632 and Q.sum_pt < 1115.723:
        z += 0.4111008 * (Q.mass_over_sum_pt - 0.07696632) * (1115.723 - Q.sum_pt)
    if Q.girth2_top40 > 0.005196966 and Q.log_sum_pt < 7.017258:
        z += -974.6301 * (Q.girth2_top40 - 0.005196966) * (7.017258 - Q.log_sum_pt)
    if Q.n_particles > 51.0 and Q.soft1_pt < 1.091797:
        z += -0.06636952 * (Q.n_particles - 51.0) * (1.091797 - Q.soft1_pt)
    if Q.sum_pt < 1017.435 and Q.dr_max_012 > 0.1828389:
        z += -0.3927259 * (1017.435 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.sum_pt < 1028.184 and Q.dr_max_012 > 0.1828389:
        z += 0.3571579 * (1028.184 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.6800824
    if Q.mass_top40 < 80.89043:
        z += -0.007617386 * Q.mass_top40 - 1.369897
    if 80.89043 <= Q.mass_top40 < 163.2541:
        z += 0.02411342 * Q.mass_top40 - 3.936616
    if Q.sum_pt_top40 < 956.2133:
        z += -0.006757492 * Q.sum_pt_top40 + 6.461604
    if Q.girth2_top40 < 0.006259772:
        z += -315.7675 * Q.girth2_top40 + 1.976633
    if Q.z_dr_0p1_0p2 >= 0.6882177:
        z += 7.232401 * Q.z_dr_0p1_0p2 - 4.977467
    if Q.mass < 64.48544:
        z += 0.01631876 * Q.mass - 0.08565806
    if 64.48544 <= Q.mass < 82.85409:
        z += -0.07157734 * Q.mass + 5.582361
    if 82.85409 <= Q.mass < 92.85979:
        z += -0.0167193 * Q.mass + 1.037148
    if 92.85979 <= Q.mass < 143.7876:
        z += 0.04946811 * Q.mass - 5.109002
    if 143.7876 <= Q.mass < 160.8:
        z += -0.04734784 * Q.mass + 8.811934
    if 160.8 <= Q.mass < 162.8363:
        z += -0.2058251 * Q.mass + 34.29508
    if 162.8363 <= Q.mass < 172.8:
        z += -0.07821117 * Q.mass + 13.51489
    if 0.09749958 <= Q.girth < 0.1402186:
        z += -12.42092 * Q.girth + 1.211035
    if Q.girth >= 0.1402186:
        z += -96.52523 * Q.girth + 13.00402
    if Q.z_top5 >= 0.7963975:
        z += 6.542096 * Q.z_top5 - 5.210109
    if Q.girth2_top30 >= 0.0008564881:
        z += 75.21263 * Q.girth2_top30 - 0.06441872
    if Q.mass_top50 < 92.16545:
        z += -0.008712653 * Q.mass_top50 + 3.140486
    if 92.16545 <= Q.mass_top50 < 138.8977:
        z += -0.05001859 * Q.mass_top50 + 6.947466
    if Q.mass_top30 < 73.33139:
        z += -0.01394976 * Q.mass_top30 + 1.022955
    if Q.LHA >= 0.404204:
        z += 67.9024 * Q.LHA - 27.44642
    if Q.sj3_dr13 >= 0.1654269:
        z += -3.076444 * Q.sj3_dr13 + 0.5089267
    if Q.e3 >= 0.0003372339:
        z += -3833.846 * Q.e3 + 1.292903
    if Q.sum_pt < 986.0565:
        z += -0.02278801 * Q.sum_pt + 22.47027
    if Q.log_sum_pt < 6.903423:
        z += 13.1781 * Q.log_sum_pt - 90.97399
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += -0.003253904 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0002556704 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top40 < 906.6023:
        z += 0.0001718745 * (80.89043 - Q.mass_top40) * (906.6023 - Q.sum_pt_top40)
    if Q.sum_pt_top40 < 956.2133 and Q.soft3_pt > 2.873047:
        z += 0.01028863 * (956.2133 - Q.sum_pt_top40) * (Q.soft3_pt - 2.873047)
    if Q.z_dr_0p1_0p2 > 0.6882177 and Q.pt_2 < 154.25:
        z += -0.09432579 * (Q.z_dr_0p1_0p2 - 0.6882177) * (154.25 - Q.pt_2)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.1516575
    if Q.girth < 0.1207452:
        z += 9.156365 * Q.girth - 1.105587
    if Q.mass < 64.48544:
        z += -0.04440869 * Q.mass + 3.502886
    if 64.48544 <= Q.mass < 78.26182:
        z += -0.08627637 * Q.mass + 6.202742
    if 78.26182 <= Q.mass < 89.74183:
        z += -0.02373968 * Q.mass + 1.308507
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.06333231 * Q.mass - 6.505494
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.02066901 * Q.mass - 2.194379
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.09461688 * Q.mass + 14.3823
    if Q.mass >= 162.8363:
        z += -0.1357945 * Q.mass + 21.08752
    if Q.sj2_mass1 < 65.20727:
        z += 0.01079364 * Q.sj2_mass1 - 0.7038235
    if Q.D2 < 2.975532:
        z += -0.1971708 * Q.D2 + 0.586688
    if Q.z_top2_slots < 0.5760704:
        z += -3.47822 * Q.z_top2_slots + 2.0037
    if Q.mass_top5 < 59.40777:
        z += 0.008408502 * Q.mass_top5 - 0.4995303
    if Q.sum_pt_top50 < 959.0957:
        z += 0.003694212 * Q.sum_pt_top50 - 3.543103
    if Q.girth2 < 0.002575211:
        z += 552.6863 * Q.girth2 - 1.423284
    if Q.dr_0 < 0.06413297:
        z += -7.608833 * Q.dr_0 + 0.4879771
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -8.225958 * Q.z_dr_0_0p05 + 6.313204
    if 136.785 <= Q.mass_top50 < 172.8:
        z += 0.08793941 * Q.mass_top50 - 12.02879
    if Q.mass_top50 >= 172.8:
        z += -0.02328748 * Q.mass_top50 + 7.191215
    if Q.e2 >= 0.01256572:
        z += 44.35472 * Q.e2 - 0.5573491
    if Q.e3 < 0.0001086251:
        z += -5259.916 * Q.e3 + 0.5713588
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.03858257 * Q.n_dr_0p2_0p4 - 0.4244083
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -7.189171 * Q.z_dr_0p2_0p4 + 0.655837
    if Q.mass_top15 < 72.18744:
        z += 0.02327332 * Q.mass_top15 - 1.680042
    if Q.girth2_top20 < 0.01655983:
        z += -89.15488 * Q.girth2_top20 + 1.47639
    if Q.mass_top30 >= 78.53034:
        z += -0.0108149 * Q.mass_top30 + 0.8492981
    if Q.mass_top50 > 160.8 and Q.soft4_z > 0.001721109:
        z += 75.61714 * (Q.mass_top50 - 160.8) * (Q.soft4_z - 0.001721109)
    if Q.girth2_top10 < 0.01976735 and Q.psi_0p3 > 0.9985421:
        z += -22285.17 * (0.01976735 - Q.girth2_top10) * (Q.psi_0p3 - 0.9985421)
    if Q.mass > 162.8363 and Q.soft5_z > 0.001434897:
        z += -49.35608 * (Q.mass - 162.8363) * (Q.soft5_z - 0.001434897)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.6431159
    if 0.9313699 <= Q.psi_0p2 < 0.9935324:
        z += -7.908947 * Q.psi_0p2 + 7.366155
    if Q.psi_0p2 >= 0.9935324:
        z += 114.3229 * Q.psi_0p2 - 114.0752
    if Q.mass < 80.4:
        z += -0.01822249 * Q.mass + 2.679877
    if 80.4 <= Q.mass < 92.85979:
        z += -0.09046547 * Q.mass + 8.488212
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.01069702 * Q.mass + 1.080931
    if Q.girth2_top10 < 0.00406126:
        z += -40.5329 * Q.girth2_top10 - 0.1008898
    if 0.00406126 <= Q.girth2_top10 < 0.00625621:
        z += 120.9615 * Q.girth2_top10 - 0.7567603
    if Q.e2 < 0.02515919:
        z += -136.6024 * Q.e2 + 3.726107
    if 0.02515919 <= Q.e2 < 0.03029714:
        z += -56.30672 * Q.e2 + 1.705933
    if Q.girth2_top30 < 0.006929741:
        z += 106.264 * Q.girth2_top30 - 0.7363821
    if Q.mass_top40 < 67.72643:
        z += 0.001648858 * Q.mass_top40 - 0.4657825
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.02270075 * Q.mass_top40 - 1.891552
    if Q.girth < 0.076787:
        z += 18.47866 * Q.girth - 1.222162
    if 0.076787 <= Q.girth < 0.08589404:
        z += -21.60515 * Q.girth + 1.855754
    if Q.e2_sq < 0.00818374:
        z += -222.1106 * Q.e2_sq + 1.817695
    if Q.lam1 < 0.006716737:
        z += 206.561 * Q.lam1 - 1.387416
    if Q.e3 < 3.793233e-05:
        z += 22559.03 * Q.e3 - 0.8557168
    if Q.mass_top50 < 80.35535:
        z += 0.02686474 * Q.mass_top50 - 2.158726
    if Q.psi_0p3 >= 0.9896594:
        z += 89.62396 * Q.psi_0p3 - 88.6972
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -33.04289 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.007455982 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.0497 and Q.planar_flow < 0.3563114:
        z += 0.129551 * (101.0497 - Q.mass) * (0.3563114 - Q.planar_flow)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.5687879
    if Q.mass_over_sum_pt < 0.05077291:
        z += 68.97321 * Q.mass_over_sum_pt - 3.846368
    if 0.05077291 <= Q.mass_over_sum_pt < 0.06895248:
        z += 18.94421 * Q.mass_over_sum_pt - 1.30625
    if Q.girth2_top10 < 0.0007431905:
        z += -464.2041 * Q.girth2_top10 + 0.3449921
    if Q.mass < 62.55:
        z += -0.02709751 * Q.mass + 3.673238
    if 62.55 <= Q.mass < 74.25181:
        z += -0.1112354 * Q.mass + 8.93606
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.07865754 * Q.mass + 6.517099
    if Q.mass >= 160.8:
        z += -0.06440303 * Q.mass + 10.35601
    if Q.mass_top40 < 74.78616:
        z += 0.02207065 * Q.mass_top40 - 1.650579
    if Q.sd_mass > 125.1 and Q.sd_zg < 0.4199841:
        z += 0.4328124 * (Q.sd_mass - 125.1) * (0.4199841 - Q.sd_zg)
    if Q.mass < 86.4 and Q.psi_0p3 < 0.9985421:
        z += -5.354943 * (86.4 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.001064185 * (Q.sj3_pair_mass_max - 120.6) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.sj3_pair_mass_max > 120.6 and Q.z_dr_0p1_0p2 > 0.2864926:
        z += 0.04677431 * (Q.sj3_pair_mass_max - 120.6) * (Q.z_dr_0p1_0p2 - 0.2864926)
    if Q.mass < 86.4 and Q.lam2 < 0.003687605:
        z += 13.10186 * (86.4 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.sd_mass > 125.1 and Q.lam2 > 0.0001679609:
        z += -18.48827 * (Q.sd_mass - 125.1) * (Q.lam2 - 0.0001679609)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.342864
    if Q.sum_pt < 1007.788:
        z += 0.02877398 * Q.sum_pt - 29.65003
    if 1007.788 <= Q.sum_pt < 1085.125:
        z += 0.008430008 * Q.sum_pt - 9.147611
    if 74.25181 <= Q.mass < 136.785:
        z += 0.0103348 * Q.mass - 0.7673775
    if 136.785 <= Q.mass < 143.7876:
        z += -0.03061943 * Q.mass + 4.834547
    if 143.7876 <= Q.mass < 160.8:
        z += -0.1059659 * Q.mass + 15.66843
    if 160.8 <= Q.mass < 162.8363:
        z += -0.3226656 * Q.mass + 50.51375
    if 162.8363 <= Q.mass < 172.8:
        z += -0.2833858 * Q.mass + 44.11757
    if Q.mass >= 172.8:
        z += -0.1461962 * Q.mass + 20.41121
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 229.6855 * Q.mass_over_sum_pt - 39.24868
    if Q.sum_pt_top40 < 1007.44:
        z += -0.01203785 * Q.sum_pt_top40 + 12.44835
    if 1007.44 <= Q.sum_pt_top40 < 1053.047:
        z += -0.007036842 * Q.sum_pt_top40 + 7.410128
    if 92.16545 <= Q.mass_top50 < 136.785:
        z += -0.02938187 * Q.mass_top50 + 2.707994
    if 136.785 <= Q.mass_top50 < 168.9698:
        z += 0.09612004 * Q.mass_top50 - 14.45879
    if Q.mass_top50 >= 168.9698:
        z += 0.1800092 * Q.mass_top50 - 28.63352
    if Q.girth2_top50 < 0.007820315:
        z += -137.7844 * Q.girth2_top50 + 2.250925
    if 0.007820315 <= Q.girth2_top50 < 0.02550569:
        z += -66.34904 * Q.girth2_top50 + 1.692278
    if Q.tau1 < 0.08286256:
        z += 6.774786 * Q.tau1 - 0.5613761
    if Q.mass_over_sum_pt_sq < 0.02580396:
        z += 20.76268 * Q.mass_over_sum_pt_sq - 0.5357594
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -584.77 * Q.mass_over_sum_pt_sq + 17.07529
    if Q.log_sum_pt < 6.910131:
        z += 9.559906 * Q.log_sum_pt - 66.0602
    if Q.mass_top40 < 160.8:
        z += 0.01118256 * Q.mass_top40 - 1.798156
    if Q.sum_pt_top50 < 997.0189:
        z += -0.01108942 * Q.sum_pt_top50 + 11.05636
    if Q.sum_pt_top30 < 966.0633:
        z += 0.005400036 * Q.sum_pt_top30 - 5.216777
    if Q.sum_pt < 1085.125 and Q.tau21_b2 < 0.8310045:
        z += -0.004984864 * (1085.125 - Q.sum_pt) * (0.8310045 - Q.tau21_b2)
    if Q.log_sum_pt < 6.811175 and Q.dr_13 < 0.1312677:
        z += -164.6668 * (6.811175 - Q.log_sum_pt) * (0.1312677 - Q.dr_13)
    if Q.log_sum_pt > 7.139296 and Q.sd_zg > 0.4462823:
        z += 1592.06 * (Q.log_sum_pt - 7.139296) * (Q.sd_zg - 0.4462823)
    if Q.log_sum_pt < 6.811175 and Q.zdr_11 < 0.005041702:
        z += 4232.593 * (6.811175 - Q.log_sum_pt) * (0.005041702 - Q.zdr_11)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.6994699
    if Q.mass < 80.4:
        z += 0.279211 * Q.mass - 24.1816
    if 80.4 <= Q.mass < 82.85409:
        z += 0.2548435 * Q.mass - 22.22245
    if 82.85409 <= Q.mass < 91.19:
        z += 0.1328738 * Q.mass - 12.11676
    if Q.psi_0p3 >= 0.9924477:
        z += 163.3773 * Q.psi_0p3 - 162.1435
    if Q.e2 < 0.03029714:
        z += -40.43768 * Q.e2 + 1.225146
    if Q.mass_over_sum_pt < 0.07852883:
        z += -4.17609 * Q.mass_over_sum_pt + 3.259084
    if 0.07852883 <= Q.mass_over_sum_pt < 0.08873143:
        z += -79.62167 * Q.mass_over_sum_pt + 9.183737
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += -39.60423 * Q.mass_over_sum_pt + 5.632932
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -102.8232 * Q.mass_over_sum_pt + 11.35219
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -49.52193 * Q.mass_over_sum_pt + 6.131115
    if 0.1182259 <= Q.mass_over_sum_pt < 0.140939:
        z += -12.16646 * Q.mass_over_sum_pt + 1.714729
    if Q.girth2_top20 < 0.006374178:
        z += -22.60397 * Q.girth2_top20 - 0.2436404
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 86.9298 * Q.girth2_top20 - 0.9418282
    if Q.girth < 0.076787:
        z += 10.77636 * Q.girth - 0.8274841
    if Q.girth2_top5 < 0.007164202:
        z += -129.0502 * Q.girth2_top5 + 0.9245417
    if Q.sd_rg < 0.1778185:
        z += -1.54243 * Q.sd_rg - 0.1623073
    if 0.1778185 <= Q.sd_rg < 0.3017146:
        z += 3.523758 * Q.sd_rg - 1.063169
    if Q.girth2_top30 < 0.01215787:
        z += 75.38352 * Q.girth2_top30 - 0.9165033
    if Q.tau1 < 0.06310829:
        z += -49.67054 * Q.tau1 + 3.134623
    if Q.log_sum_pt < 6.941997:
        z += -19.46807 * Q.log_sum_pt + 135.0502
    if 6.941997 <= Q.log_sum_pt < 6.98945:
        z += 2.046417 * Q.log_sum_pt - 14.30333
    if Q.sum_pt_top50 < 976.277:
        z += 0.03000849 * Q.sum_pt_top50 - 29.2966
    if Q.sum_pt_top40 < 1024.942:
        z += -0.004099871 * Q.sum_pt_top40 + 4.202132
    if Q.tau21_b2 < 0.342495 and Q.girth2_top40 < 0.007709916:
        z += -1660.737 * (0.342495 - Q.tau21_b2) * (0.007709916 - Q.girth2_top40)
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2404747:
        z += -10.17149 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2404747)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg < 0.1881908:
        z += -916.5762 * (Q.psi_0p3 - 0.9924477) * (0.1881908 - Q.sd_rg)
    if Q.girth2_top5 < 0.007164202 and Q.n_pt_above_10 > 13.0:
        z += -7.44323 * (0.007164202 - Q.girth2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.girth < 0.076787 and Q.z_dr_0p2_0p4 < 0.05180474:
        z += -267.7451 * (0.076787 - Q.girth) * (0.05180474 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.4554918
    if Q.z_dr_0_0p05 < 0.8459004:
        z += -0.6593533 * Q.z_dr_0_0p05 + 0.5577472
    if Q.z_dr_0p1_0p2 < 0.06472584:
        z += -1.560017 * Q.z_dr_0p1_0p2 + 0.6987166
    if 0.06472584 <= Q.z_dr_0p1_0p2 < 0.1203437:
        z += -10.74733 * Q.z_dr_0p1_0p2 + 1.293373
    if Q.girth2_top5 < 0.002270363:
        z += 196.2148 * Q.girth2_top5 + 0.1110615
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -91.84845 * Q.girth2_top5 + 0.7650696
    if Q.sum_pt < 986.0565:
        z += -0.002518139 * Q.sum_pt + 2.377512
    if 986.0565 <= Q.sum_pt < 1002.379:
        z += 0.006464618 * Q.sum_pt - 6.479994
    if Q.log_sum_pt < 6.903423:
        z += -1.442605 * Q.log_sum_pt + 10.69835
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -6.495682 * Q.log_sum_pt + 45.58187
    if Q.psi_0p3 >= 0.9896594:
        z += -72.05958 * Q.psi_0p3 + 71.31444
    if Q.girth2_top10 < 0.007678544:
        z += -100.6571 * Q.girth2_top10 + 0.7729001
    if Q.tau1 < 0.0705748:
        z += 16.10477 * Q.tau1 - 1.136591
    if Q.sd_mass < 45.595:
        z += 0.003534922 * Q.sd_mass + 0.4850322
    if 45.595 <= Q.sd_mass < 79.18312:
        z += -0.01923916 * Q.sd_mass + 1.523416
    if Q.sum_pt_top40 < 935.8189:
        z += 0.01180287 * Q.sum_pt_top40 - 11.49551
    if 935.8189 <= Q.sum_pt_top40 < 1018.698:
        z += 0.005431578 * Q.sum_pt_top40 - 5.533136
    if Q.sum_pt_top30 < 933.1875:
        z += -0.005659686 * Q.sum_pt_top30 + 5.281548
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 1260.541:
        z += -0.004182898 * (0.8459004 - Q.z_dr_0_0p05) * (1260.541 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.4257766 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 390.9258 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -3.407665 * (0.008329695 - Q.girth2_top5) * (Q.sj3_mass1 - 5.112677)
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
