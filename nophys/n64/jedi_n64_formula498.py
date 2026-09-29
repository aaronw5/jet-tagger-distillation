"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network's neuron values (step 4; no W/Z/H/t mass values offered as thresholds), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 93.1% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
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
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft2_pt               pT [GeV] of the 2. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_pt               pT [GeV] of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_pt               pT [GeV] of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.z_11                   pT of particle 11 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_9                    pT of particle 9 / total pT
  Q.soft8_z                pT share of the 8. softest real particle (0 if it is among the 15 hardest)
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the sum_z_dr)
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.absphi_0               |Δφ| of particle 0
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.sj3_dr13               distance between subjet axes 1 and 3 (of 3)
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_12                  ΔR of particle 12 from the jet axis
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p1                pT share within ΔR < 0.1 of the jet axis
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.tau3                   N-subjettiness τ3 (β=1)
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
        C3=ecf('e4') * ecf('e2') / max(ecf('e3') ** 2, 1e-30),
        D2=e3 / max(e2 ** 3, 1e-12),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
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
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        pt_3=pt[3],
        pt_9=pt[9],
        soft2_pt=softp(2, 'pt'),
        soft4_pt=softp(4, 'pt'),
        soft5_pt=softp(5, 'pt'),
        soft6_pt=softp(6, 'pt'),
        z_11=z[11],
        z_5=z[5],
        z_6=z[6],
        z_9=z[9],
        soft8_z=softp(8, 'z'),
        z_top10_slots=sum(pt[:10]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        absphi_0=abs(phi[0]),
        sj2_dr=subjets(2)["dr"][0],
        sj3_dr13=subjets(3)["dr"][1],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_12=dr[12] if pt[12] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p1=sum(z[i] for i in real if dr[i] < 0.1),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        tau3=tau_n(3),
    )


def neuron_0(Q):
    z = -2.48
    if Q.e2 < 0.0192:
        z += -36.5 * Q.e2 + 0.7008
    if Q.sum_zz_dr2 < 0.00609:
        z += 230.0 * Q.sum_zz_dr2 - 1.4007
    if Q.sum_z_dr2_top15 >= 0.00637:
        z += -96.6 * Q.sum_z_dr2_top15 + 0.615342
    if Q.sum_z_dr2_top30 < 0.00619:
        z += 168.0 * Q.sum_z_dr2_top30 - 1.03992
    if Q.log_sum_pt < 7.03:
        z += -6.75 * Q.log_sum_pt + 47.4525
    if Q.mass < 64.2:
        z += 0.0263 * Q.mass - 0.10481
    if 64.2 <= Q.mass < 64.7:
        z += 0.0793 * Q.mass - 3.50741
    if 64.7 <= Q.mass < 79.1:
        z += 0.0337 * Q.mass - 0.55709
    if 79.1 <= Q.mass < 81.2:
        z += -0.0973 * Q.mass + 9.80501
    if 81.2 <= Q.mass < 99.7:
        z += -0.1471 * Q.mass + 13.84877
    if Q.mass >= 99.7:
        z += -0.078 * Q.mass + 6.9595
    if Q.mass_over_sum_pt >= 0.105:
        z += -32.4 * Q.mass_over_sum_pt + 3.402
    if Q.mass_over_sum_pt_sq < 0.00926:
        z += -447.0 * Q.mass_over_sum_pt_sq + 4.13922
    if Q.mass_over_sum_pt_sq >= 0.0102:
        z += 204.0 * Q.mass_over_sum_pt_sq - 2.0808
    if Q.mass_top50 >= 89.2:
        z += 0.0948 * Q.mass_top50 - 8.45616
    if Q.n_dr_0p2_0p4 < 15.8:
        z += -0.0511 * Q.n_dr_0p2_0p4 + 0.80738
    if Q.psi_0p2 >= 0.923:
        z += -5.88 * Q.psi_0p2 + 5.42724
    if Q.psi_0p3 >= 0.988:
        z += 42.7 * Q.psi_0p3 - 42.1876
    if Q.pt_entropy < 2.54:
        z += -0.263 * Q.pt_entropy + 0.66802
    if Q.sum_pt < 1010.0:
        z += 0.015 * Q.sum_pt - 15.15
    if Q.sum_pt_top40 < 879.0:
        z += -0.00683 * Q.sum_pt_top40 + 6.00357
    if Q.tau1 < 0.0703:
        z += 14.5 * Q.tau1 - 1.01935
    if Q.tau1 >= 0.0973:
        z += -6.74 * Q.tau1 + 0.655802
    if Q.z_top50_slots < 0.991:
        z += 13.2 * Q.z_top50_slots - 13.0812
    return max(0.0, z)


def neuron_1(Q):
    z = -15.0
    if Q.M3 < 0.0401:
        z += 24.5 * Q.M3 - 0.98245
    if Q.sum_z_dr2_top10 < 0.000858:
        z += -909.0 * Q.sum_z_dr2_top10 + 0.779922
    if Q.sum_z_dr2_top30 < 0.00551:
        z += -291.0 * Q.sum_z_dr2_top30 + 1.60341
    if 6.91 <= Q.log_sum_pt < 7.0:
        z += 35.1 * Q.log_sum_pt - 242.541
    if Q.log_sum_pt >= 7.0:
        z += 14.2 * Q.log_sum_pt - 96.241
    if Q.mass_over_sum_pt_sq < 0.0111:
        z += 121.0 * Q.mass_over_sum_pt_sq - 1.3431
    if Q.n_dr_0p2_0p4 < 7.25:
        z += 0.0859 * Q.n_dr_0p2_0p4 - 0.622775
    if Q.psi_0p3 >= 0.997:
        z += -102.0 * Q.psi_0p3 + 101.694
    if Q.pt_entropy >= 2.05:
        z += 1.34 * Q.pt_entropy - 2.747
    if Q.sj2_mass1 < 34.4:
        z += 0.0186 * Q.sj2_mass1 - 0.63984
    if Q.sj3_mass1 >= 5.79:
        z += 0.0331 * Q.sj3_mass1 - 0.191649
    if Q.sj3_mass2 >= 3.03:
        z += 0.0334 * Q.sj3_mass2 - 0.101202
    if Q.sum_pt < 959.0:
        z += -0.0221 * Q.sum_pt + 21.1939
    z += 0.0473 * Q.sum_pt_top40
    if Q.sum_pt_top50 < 933.0:
        z += -0.0295 * Q.sum_pt_top50
    if Q.sum_pt_top50 >= 933.0:
        z += -0.053 * Q.sum_pt_top50 + 21.9255
    if Q.tau3 >= 0.0163:
        z += -29.4 * Q.tau3 + 0.47922
    if Q.z_dr_0_0p05 >= 0.872:
        z += -4.98 * Q.z_dr_0_0p05 + 4.34256
    if Q.z_top40_slots < 0.966:
        z += -44.6 * Q.z_top40_slots + 43.0836
    if Q.z_top40_slots >= 0.969:
        z += -40.4 * Q.z_top40_slots + 39.1476
    if Q.mass_top20 < 72.1 and Q.n_real_top40 > 32.4:
        z += 0.00233 * (72.1 - Q.mass_top20) * (Q.n_real_top40 - 32.4)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.132:
        z += 0.336 * (Q.n_particles - 38.0) * (0.132 - Q.dr_0)
    if Q.n_particles > 39.2 and Q.dr_1 < 0.176:
        z += 0.206 * (Q.n_particles - 39.2) * (0.176 - Q.dr_1)
    if Q.z_top30_slots > 0.961 and Q.C2 < 0.0852:
        z += 427.0 * (Q.z_top30_slots - 0.961) * (0.0852 - Q.C2)
    if Q.z_top30_slots > 0.942 and Q.max_pair_mass > 6.05:
        z += 0.372 * (Q.z_top30_slots - 0.942) * (Q.max_pair_mass - 6.05)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.79
    if Q.C2 >= 0.117:
        z += -4.53 * Q.C2 + 0.53001
    if 0.309 <= Q.LHA < 0.416:
        z += 6.52 * Q.LHA - 2.01468
    if Q.LHA >= 0.416:
        z += -8.38 * Q.LHA + 4.18372
    if Q.N2 < 0.338:
        z += -0.521 * Q.N2 + 0.176098
    if Q.e2 >= 0.0414:
        z += 18.6 * Q.e2 - 0.77004
    if Q.sum_zz_dr2 >= 0.00818:
        z += 146.0 * Q.sum_zz_dr2 - 1.19428
    if Q.e3 < 0.000195:
        z += 1040.0 * Q.e3 - 0.33384
    if 0.000195 <= Q.e3 < 0.000321:
        z += 389.0 * Q.e3 - 0.206895
    if Q.e3 >= 0.000321:
        z += -651.0 * Q.e3 + 0.126945
    if Q.sum_z_dr >= 0.0821:
        z += -24.6 * Q.sum_z_dr + 2.01966
    if Q.sum_z_dr2_top15 >= 0.00603:
        z += -27.4 * Q.sum_z_dr2_top15 + 0.165222
    if Q.sum_z_dr2_top40 < 0.0251:
        z += -43.1 * Q.sum_z_dr2_top40 + 1.08181
    if Q.lam2 >= 7.15e-05:
        z += 28.8 * Q.lam2 - 0.0020592
    if Q.log_sum_pt < 6.89:
        z += -5.15 * Q.log_sum_pt + 34.9565
    if 6.89 <= Q.log_sum_pt < 7.06:
        z += 3.1 * Q.log_sum_pt - 21.886
    if Q.mass < 64.2:
        z += 0.0047 * Q.mass - 1.03326
    if 64.2 <= Q.mass < 64.5:
        z += 0.0376 * Q.mass - 3.14544
    if 64.5 <= Q.mass < 93.9:
        z += 0.0089 * Q.mass - 1.29429
    if 93.9 <= Q.mass < 126.0:
        z += -0.0407 * Q.mass + 3.36315
    if Q.mass >= 126.0:
        z += -0.0287 * Q.mass + 1.85115
    if Q.mass_top20 >= 116.0:
        z += -0.00116 * Q.mass_top20 + 0.13456
    if Q.mass_top50 >= 81.3:
        z += 0.0102 * Q.mass_top50 - 0.82926
    if Q.n_particles < 53.3:
        z += 0.0157 * Q.n_particles - 0.83681
    if Q.psi_0p1 < 0.355:
        z += -1.19 * Q.psi_0p1 + 0.42245
    if Q.psi_0p3 >= 0.988:
        z += -18.0 * Q.psi_0p3 + 17.784
    if Q.sd_mass >= 84.5:
        z += -0.00668 * Q.sd_mass + 0.56446
    if Q.sd_rg >= 0.275:
        z += 2.19 * Q.sd_rg - 0.60225
    if Q.sj3_dr13 >= 0.309:
        z += 1.1 * Q.sj3_dr13 - 0.3399
    if Q.sum_pt < 933.0:
        z += 0.0092 * Q.sum_pt - 10.38965
    if 933.0 <= Q.sum_pt < 1020.0:
        z += 0.01735 * Q.sum_pt - 17.9936
    if 1020.0 <= Q.sum_pt < 1050.0:
        z += 0.01785 * Q.sum_pt - 18.5036
    if 1050.0 <= Q.sum_pt < 1120.0:
        z += 0.01251 * Q.sum_pt - 12.8966
    if 1120.0 <= Q.sum_pt < 1160.0:
        z += 0.00782 * Q.sum_pt - 7.6438
    if 1160.0 <= Q.sum_pt < 1280.0:
        z += 0.01105 * Q.sum_pt - 11.3906
    if Q.sum_pt >= 1280.0:
        z += 0.00817 * Q.sum_pt - 7.7042
    if Q.sum_pt_top15 < 1020.0:
        z += 0.000983 * Q.sum_pt_top15 - 1.00266
    if Q.sum_pt_top15 >= 1030.0:
        z += 0.00175 * Q.sum_pt_top15 - 1.8025
    if Q.sum_pt_top3 < 781.0:
        z += -0.000328 * Q.sum_pt_top3 + 0.256168
    if Q.sum_pt_top30 >= 1030.0:
        z += 0.00172 * Q.sum_pt_top30 - 1.7716
    if Q.sum_pt_top50 >= 930.0:
        z += -0.01 * Q.sum_pt_top50 + 9.3
    if Q.z_dr_0p1_0p2 >= 0.418:
        z += -1.02 * Q.z_dr_0p1_0p2 + 0.42636
    if Q.log_sum_pt > 6.89 and Q.mass_top40 > 115.0:
        z += 0.018 * (Q.log_sum_pt - 6.89) * (Q.mass_top40 - 115.0)
    return max(0.0, z)


def neuron_3(Q):
    z = -1.14
    if Q.e2 < 0.0295:
        z += -26.2 * Q.e2 + 0.7729
    if Q.sum_zz_dr2 < 0.0067:
        z += -79.8 * Q.sum_zz_dr2 + 0.53466
    if Q.sum_z_dr2 < 0.0101:
        z += -84.9 * Q.sum_z_dr2 + 0.85749
    if Q.sum_z_dr2_top20 < 0.00803:
        z += 97.1 * Q.sum_z_dr2_top20 - 0.779713
    if Q.sum_z_dr2_top50 >= 0.00631:
        z += -133.0 * Q.sum_z_dr2_top50 + 0.83923
    if Q.lam2 < 0.00113:
        z += -309.0 * Q.lam2 + 0.34917
    if Q.mass < 73.4:
        z += 0.0173 * Q.mass - 1.26982
    if 80.2 <= Q.mass < 92.8:
        z += -0.0106 * Q.mass + 0.85012
    if 92.8 <= Q.mass < 144.0:
        z += 0.0405 * Q.mass - 3.89196
    if Q.mass >= 144.0:
        z += 0.067 * Q.mass - 7.70796
    if Q.mass_over_sum_pt >= 0.09:
        z += 28.6 * Q.mass_over_sum_pt - 2.574
    if Q.mass_top5 < 66.4:
        z += -0.00284 * Q.mass_top5 + 0.188576
    if Q.mass_top50 >= 91.6:
        z += -0.0626 * Q.mass_top50 + 5.73416
    if Q.max_dr < 0.288:
        z += -1.31 * Q.max_dr + 0.37728
    if Q.max_dr >= 0.288:
        z += 0.312 * Q.max_dr - 0.089856
    if Q.n_dr_0_0p05 < 14.7:
        z += 0.0211 * Q.n_dr_0_0p05 - 0.31017
    if 11.8 <= Q.n_dr_0p1_0p2 < 23.5:
        z += -0.0292 * Q.n_dr_0p1_0p2 + 0.34456
    if Q.n_dr_0p1_0p2 >= 23.5:
        z += -0.02713 * Q.n_dr_0p1_0p2 + 0.295915
    if Q.n_particles < 61.2:
        z += -0.0147 * Q.n_particles + 0.89964
    if Q.psi_0p3 >= 0.998:
        z += 89.2 * Q.psi_0p3 - 89.0216
    if Q.sj2_mass1 >= 7.0:
        z += -0.00485 * Q.sj2_mass1 + 0.03395
    if Q.sj3_mass1 < 26.4:
        z += -0.00584 * Q.sj3_mass1 + 0.154176
    if Q.tau1 >= 0.0163:
        z += 9.06 * Q.tau1 - 0.147678
    if Q.tau21 < 0.466:
        z += 1.08 * Q.tau21 - 0.50328
    if Q.z_top50_slots < 0.984:
        z += 12.7 * Q.z_top50_slots - 12.4968
    if Q.n_dr_0p1_0p2 < 10.9 and Q.psi_0p2 > 0.92:
        z += 0.394 * (10.9 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.92)
    if Q.n_dr_0p2_0p4 < 4.44 and Q.psi_0p1 < 1.02:
        z += 0.327 * (4.44 - Q.n_dr_0p2_0p4) * (1.02 - Q.psi_0p1)
    if Q.n_dr_0p2_0p4 < 3.31 and Q.z_dr_0p05_0p1 > 0.521:
        z += 0.213 * (3.31 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p05_0p1 - 0.521)
    if Q.n_dr_0p2_0p4 < 7.56 and Q.z_top50_slots > 0.979:
        z += 3.26 * (7.56 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.979)
    if Q.n_particles < 47.7 and Q.sum_pt_top30 > 780.0:
        z += 4.27e-05 * (47.7 - Q.n_particles) * (Q.sum_pt_top30 - 780.0)
    if Q.tau21 < 0.56 and Q.lam1 > 0.00516:
        z += 107.0 * (0.56 - Q.tau21) * (Q.lam1 - 0.00516)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.0362
    if Q.LHA >= 0.341:
        z += -6.05 * Q.LHA + 2.06305
    if Q.sum_z_dr2_top10 < 0.00697:
        z += 46.2 * Q.sum_z_dr2_top10 - 0.322014
    if Q.sum_z_dr2_top30 < 0.00635:
        z += 127.0 * Q.sum_z_dr2_top30 - 0.80645
    if Q.sum_z_dr2_top50 >= 0.00801:
        z += 116.0 * Q.sum_z_dr2_top50 - 0.92916
    if Q.lam1 >= 0.0203:
        z += -38.2 * Q.lam1 + 0.77546
    if Q.log_sum_pt >= 6.93:
        z += 0.979 * Q.log_sum_pt - 6.78447
    if Q.mass < 67.7:
        z += -0.0058 * Q.mass + 1.61996
    if 67.7 <= Q.mass < 86.4:
        z += 0.0218 * Q.mass - 0.24856
    if 86.4 <= Q.mass < 103.0:
        z += -0.0398 * Q.mass + 5.07368
    if 103.0 <= Q.mass < 146.0:
        z += 0.0276 * Q.mass - 1.86852
    if Q.mass >= 146.0:
        z += 0.0421 * Q.mass - 3.98552
    if Q.mass_top10 >= 85.9:
        z += -0.0117 * Q.mass_top10 + 1.00503
    if Q.mass_top40 < 80.1:
        z += -0.0267 * Q.mass_top40 + 2.13867
    if Q.mass_top50 >= 97.5:
        z += -0.0445 * Q.mass_top50 + 4.33875
    if Q.n_dr_0p2_0p4 < 14.9:
        z += -0.0421 * Q.n_dr_0p2_0p4 + 0.62729
    if Q.n_particles >= 22.4:
        z += -0.0198 * Q.n_particles + 0.44352
    if Q.n_pt_above_1 < 47.9:
        z += -0.0158 * Q.n_pt_above_1 + 0.75682
    if Q.psi_0p2 >= 0.942:
        z += -4.88 * Q.psi_0p2 + 4.59696
    if Q.psi_0p3 >= 0.998:
        z += 96.7 * Q.psi_0p3 - 96.5066
    if Q.pt_entropy < 3.06:
        z += 0.285 * Q.pt_entropy - 0.8721
    if 0.147 <= Q.sj2_dr < 0.179:
        z += 7.15 * Q.sj2_dr - 1.05105
    if Q.sj2_dr >= 0.179:
        z += -2.01 * Q.sj2_dr + 0.58859
    if Q.sj3_pair_mass_min >= 16.1:
        z += 0.00601 * Q.sj3_pair_mass_min - 0.096761
    if Q.sum_pt < 1050.0:
        z += 0.00327 * Q.sum_pt - 3.4335
    if Q.tau1 < 0.0691:
        z += 17.0 * Q.tau1 - 1.1747
    if Q.z_top50_slots < 0.989:
        z += 21.1 * Q.z_top50_slots - 20.8679
    if Q.mass < 79.8 and Q.lam2 < 0.00376:
        z += -22.6 * (79.8 - Q.mass) * (0.00376 - Q.lam2)
    if Q.n_dr_0_0p05 > 3.1 and Q.lam2 < 0.00123:
        z += 21.4 * (Q.n_dr_0_0p05 - 3.1) * (0.00123 - Q.lam2)
    if Q.n_dr_0p2_0p4 < 14.0 and Q.max_dr > 0.214:
        z += -0.0469 * (14.0 - Q.n_dr_0p2_0p4) * (Q.max_dr - 0.214)
    return max(0.0, z)


def neuron_5(Q):
    z = 3.53
    if Q.sum_z_dr2_top10 < 0.000636:
        z += 326.2 * Q.sum_z_dr2_top10 - 0.01263
    if 0.000636 <= Q.sum_z_dr2_top10 < 0.00489:
        z += -45.8 * Q.sum_z_dr2_top10 + 0.223962
    if Q.sum_z_dr2_top50 >= 0.0157:
        z += -73.5 * Q.sum_z_dr2_top50 + 1.15395
    if Q.log_sum_pt < 6.91:
        z += -15.3 * Q.log_sum_pt + 108.018
    if 6.91 <= Q.log_sum_pt < 7.06:
        z += -46.9 * Q.log_sum_pt + 326.374
    if Q.log_sum_pt >= 7.06:
        z += -3.7 * Q.log_sum_pt + 21.382
    if Q.mass < 66.0:
        z += -0.043 * Q.mass + 2.838
    if 91.1 <= Q.mass < 173.0:
        z += -0.0254 * Q.mass + 2.31394
    if Q.mass >= 173.0:
        z += 0.0129 * Q.mass - 4.31196
    if Q.mass_over_sum_pt < 0.0806:
        z += 72.6 * Q.mass_over_sum_pt - 5.85156
    if Q.mass_over_sum_pt_sq < 0.00646:
        z += -503.0 * Q.mass_over_sum_pt_sq + 3.24938
    if 39.0 <= Q.mass_top50 < 150.0:
        z += 0.0162 * Q.mass_top50 - 0.6318
    if Q.mass_top50 >= 150.0:
        z += -0.0104 * Q.mass_top50 + 3.3582
    if Q.max_dr < 0.437:
        z += -0.806 * Q.max_dr + 0.352222
    if Q.n_dr_0p1_0p2 < 21.6:
        z += 0.0105 * Q.n_dr_0p1_0p2 - 0.2268
    if Q.n_particles < 60.2:
        z += -0.0325 * Q.n_particles + 1.9565
    if Q.n_pt_above_1 >= 30.5:
        z += 0.0213 * Q.n_pt_above_1 - 0.64965
    if Q.sd_mass >= 66.2:
        z += 0.00705 * Q.sd_mass - 0.46671
    if Q.sum_pt < 981.0:
        z += 0.0348 * Q.sum_pt - 40.368
    if 981.0 <= Q.sum_pt < 1160.0:
        z += 0.0239 * Q.sum_pt - 29.6751
    if Q.sum_pt >= 1160.0:
        z += -0.0109 * Q.sum_pt + 10.6929
    if Q.sum_pt_top20 >= 842.0:
        z += -0.00227 * Q.sum_pt_top20 + 1.91134
    if Q.sum_pt_top3 < 769.0:
        z += 0.000738 * Q.sum_pt_top3 - 0.567522
    if Q.sum_pt_top50 >= 958.0:
        z += 0.0146 * Q.sum_pt_top50 - 13.9868
    if Q.z_11 < 0.0159:
        z += -15.2 * Q.z_11 + 0.24168
    if Q.z_dr_0_0p05 >= 0.886:
        z += 3.19 * Q.z_dr_0_0p05 - 2.82634
    if Q.zdr_0 >= 0.00541:
        z += 6.39 * Q.zdr_0 - 0.0345699
    if Q.n_particles < 66.6 and Q.D2 < 2.47:
        z += -0.0108 * (66.6 - Q.n_particles) * (2.47 - Q.D2)
    if Q.psi_0p3 > 0.996 and Q.D2 < 3.6:
        z += 35.2 * (Q.psi_0p3 - 0.996) * (3.6 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.434
    if 0.189 <= Q.LHA < 0.392:
        z += 8.78 * Q.LHA - 1.65942
    if Q.LHA >= 0.392:
        z += 20.08 * Q.LHA - 6.08902
    if Q.e2 < 0.0179:
        z += 16.8 * Q.e2 - 0.30072
    if Q.e2 >= 0.0267:
        z += -50.2 * Q.e2 + 1.34034
    if Q.sum_zz_dr2 < 0.00662:
        z += 198.0 * Q.sum_zz_dr2 - 1.31076
    if Q.e3 >= 2.64e-05:
        z += 1950.0 * Q.e3 - 0.05148
    if Q.sum_z_dr < 0.0673:
        z += -18.1 * Q.sum_z_dr + 1.21813
    if Q.sum_z_dr2_top20 < 0.00585:
        z += -136.0 * Q.sum_z_dr2_top20 + 0.7956
    if Q.lam1 < 0.00432:
        z += -174.0 * Q.lam1 + 1.35198
    if 0.00432 <= Q.lam1 < 0.00777:
        z += -58.0 * Q.lam1 + 0.85086
    if Q.lam1 >= 0.00777:
        z += 116.0 * Q.lam1 - 0.50112
    if Q.log_sum_pt < 7.06:
        z += 5.64 * Q.log_sum_pt - 39.8184
    if Q.mass < 65.3:
        z += -0.0112 * Q.mass - 0.85034
    if 65.3 <= Q.mass < 82.6:
        z += -0.0253 * Q.mass + 0.07039
    if 82.6 <= Q.mass < 104.0:
        z += 0.0548 * Q.mass - 6.54587
    if 104.0 <= Q.mass < 118.0:
        z += 0.0074 * Q.mass - 1.61627
    if 118.0 <= Q.mass < 138.0:
        z += -0.0141 * Q.mass + 0.92073
    if Q.mass >= 138.0:
        z += -0.0346 * Q.mass + 3.74973
    if 0.0808 <= Q.mass_over_sum_pt < 0.1:
        z += 63.6 * Q.mass_over_sum_pt - 5.13888
    if Q.mass_over_sum_pt >= 0.1:
        z += 25.4 * Q.mass_over_sum_pt - 1.31888
    if Q.mass_over_sum_pt_sq >= 0.00623:
        z += -269.0 * Q.mass_over_sum_pt_sq + 1.67587
    if Q.mass_top15 >= 105.0:
        z += 0.0157 * Q.mass_top15 - 1.6485
    if Q.mass_top30 >= 83.5:
        z += 0.0236 * Q.mass_top30 - 1.9706
    if Q.n_for_90pct >= 36.8:
        z += 0.0326 * Q.n_for_90pct - 1.19968
    if Q.pt_entropy >= 3.0:
        z += -0.334 * Q.pt_entropy + 1.002
    if Q.sj2_mass1 < 73.6:
        z += -0.0136 * Q.sj2_mass1 + 1.00096
    if Q.sj2_mass2 >= 3.52:
        z += -0.0137 * Q.sj2_mass2 + 0.048224
    if Q.sum_pt >= 1180.0:
        z += 0.00399 * Q.sum_pt - 4.7082
    if Q.sum_pt_top40 >= 953.0:
        z += -0.00403 * Q.sum_pt_top40 + 3.84059
    if Q.sum_pt_top50 < 1020.0:
        z += -0.00442 * Q.sum_pt_top50 + 4.5084
    if Q.tau1 < 0.0412:
        z += 1.1 * Q.tau1 + 0.35486
    if 0.0412 <= Q.tau1 < 0.0626:
        z += -18.7 * Q.tau1 + 1.17062
    if Q.tau21_b2 >= 0.107:
        z += 0.508 * Q.tau21_b2 - 0.054356
    if Q.z_dr_0p1_0p2 < 0.275:
        z += -0.73 * Q.z_dr_0p1_0p2 + 0.20075
    if Q.e2 > 0.019 and Q.psi_0p3 > 0.99:
        z += 1370.0 * (Q.e2 - 0.019) * (Q.psi_0p3 - 0.99)
    if Q.e2 > 0.0462 and Q.zdr_0 > -0.00249:
        z += 426.0 * (Q.e2 - 0.0462) * (Q.zdr_0 - -0.00249)
    if Q.mass < 119.0 and Q.sum_pt_top50 < 990.0:
        z += 7.3e-05 * (119.0 - Q.mass) * (990.0 - Q.sum_pt_top50)
    if Q.z_dr_0p1_0p2 < 0.263 and Q.sj3_dr13 > 0.263:
        z += 5.81 * (0.263 - Q.z_dr_0p1_0p2) * (Q.sj3_dr13 - 0.263)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.133
    if Q.LHA < 0.318:
        z += 5.2 * Q.LHA - 1.7368
    if 0.318 <= Q.LHA < 0.334:
        z += 25.3 * Q.LHA - 8.1286
    if Q.LHA >= 0.334:
        z += 20.1 * Q.LHA - 6.3918
    if Q.e2 >= 0.0332:
        z += 28.6 * Q.e2 - 0.94952
    if Q.sum_zz_dr2 < 0.00639:
        z += -218.0 * Q.sum_zz_dr2 + 1.39302
    if Q.sum_z_dr >= 0.0882:
        z += -47.0 * Q.sum_z_dr + 4.1454
    if Q.sum_z_dr2_top15 >= 0.00849:
        z += -50.9 * Q.sum_z_dr2_top15 + 0.432141
    if Q.sum_z_dr2_top30 < 0.00841:
        z += 149.0 * Q.sum_z_dr2_top30 - 1.25309
    if Q.sum_z_dr2_top50 < 0.00924:
        z += -428.0 * Q.sum_z_dr2_top50 + 3.95472
    if Q.mass < 49.7:
        z += -0.0552 * Q.mass + 3.79311
    if 49.7 <= Q.mass < 82.4:
        z += -0.0321 * Q.mass + 2.64504
    if Q.mass_over_sum_pt < 0.0813:
        z += 9.12 * Q.mass_over_sum_pt - 0.741456
    if Q.mass_top30 < 87.5:
        z += -0.0147 * Q.mass_top30 + 1.28625
    if Q.mass_top50 < 84.5:
        z += 0.0377 * Q.mass_top50 - 4.00515
    if 84.5 <= Q.mass_top50 < 95.5:
        z += 0.0745 * Q.mass_top50 - 7.11475
    if Q.n_dr_0p2_0p4 < 6.65:
        z += -0.113 * Q.n_dr_0p2_0p4 + 0.75145
    if Q.planar_flow < 0.467:
        z += -0.805 * Q.planar_flow + 0.375935
    if Q.sd_mass >= 69.7:
        z += -0.00808 * Q.sd_mass + 0.563176
    if 0.156 <= Q.sd_rg < 0.189:
        z += 6.3 * Q.sd_rg - 0.9828
    if 0.189 <= Q.sd_rg < 0.279:
        z += -4.6 * Q.sd_rg + 1.0773
    if Q.sd_rg >= 0.279:
        z += 8.6 * Q.sd_rg - 2.6055
    if Q.sum_pt_top30 >= 1090.0:
        z += 0.0027 * Q.sum_pt_top30 - 2.943
    if Q.sum_pt_top50 >= 1180.0:
        z += -0.00299 * Q.sum_pt_top50 + 3.5282
    if Q.tau1 < 0.0911:
        z += -19.8 * Q.tau1 + 1.80378
    if Q.lam1_plus_lam2 < 0.00818:
        z += 480.0 * Q.lam1_plus_lam2 - 3.9264
    if Q.z_dr_0p1_0p2 >= 0.385:
        z += -0.198 * Q.z_dr_0p1_0p2 + 0.07623
    if Q.zdr_0 >= 0.0112:
        z += -20.7 * Q.zdr_0 + 0.23184
    if Q.mass < 93.7 and Q.n_dr_0_0p05 < 11.5:
        z += -0.00369 * (93.7 - Q.mass) * (11.5 - Q.n_dr_0_0p05)
    if Q.mass < 90.4 and Q.psi_0p3 > 0.997:
        z += -78.1 * (90.4 - Q.mass) * (Q.psi_0p3 - 0.997)
    if Q.mass < 94.4 and Q.psi_0p3 > 0.967:
        z += -4.33 * (94.4 - Q.mass) * (Q.psi_0p3 - 0.967)
    if Q.mass < 100.0 and Q.psi_0p3 > 0.997:
        z += 44.4 * (100.0 - Q.mass) * (Q.psi_0p3 - 0.997)
    if Q.mass < 103.0 and Q.psi_0p3 > 0.963:
        z += 3.0 * (103.0 - Q.mass) * (Q.psi_0p3 - 0.963)
    if Q.mass < 91.9 and Q.sd_mass > 48.4:
        z += -0.00272 * (91.9 - Q.mass) * (Q.sd_mass - 48.4)
    if Q.mass < 113.0 and Q.sd_mass > 69.1:
        z += 0.00186 * (113.0 - Q.mass) * (Q.sd_mass - 69.1)
    if Q.psi_0p3 > 0.994 and Q.n_dr_0p1_0p2 > 14.6:
        z += -5.0 * (Q.psi_0p3 - 0.994) * (Q.n_dr_0p1_0p2 - 14.6)
    return max(0.0, z)


def neuron_8(Q):
    z = 6.68
    if Q.e2 >= 0.0447:
        z += 25.0 * Q.e2 - 1.1175
    if 0.0107 <= Q.sum_zz_dr2 < 0.0251:
        z += -263.0 * Q.sum_zz_dr2 + 2.8141
    if Q.sum_zz_dr2 >= 0.0251:
        z += 38.0 * Q.sum_zz_dr2 - 4.741
    if Q.sum_z_dr >= 0.0102:
        z += -9.84 * Q.sum_z_dr + 0.100368
    if Q.sum_z_dr2 < 0.0251:
        z += 309.0 * Q.sum_z_dr2 - 7.7559
    if Q.sum_z_dr2_top30 < 0.00796:
        z += -240.0 * Q.sum_z_dr2_top30 + 1.9104
    if Q.log_sum_pt < 6.94:
        z += 13.7 * Q.log_sum_pt - 96.722
    if 6.94 <= Q.log_sum_pt < 7.06:
        z += 24.7 * Q.log_sum_pt - 173.062
    if Q.log_sum_pt >= 7.06:
        z += 11.0 * Q.log_sum_pt - 76.34
    if 73.3 <= Q.mass < 100.0:
        z += 0.0395 * Q.mass - 2.89535
    if 100.0 <= Q.mass < 126.0:
        z += -0.0079 * Q.mass + 1.84465
    if Q.mass >= 126.0:
        z += -0.0485 * Q.mass + 6.96025
    if Q.mass_over_sum_pt >= 0.0871:
        z += 41.2 * Q.mass_over_sum_pt - 3.58852
    if Q.n_dr_0p1_0p2 >= 18.8:
        z += 0.0329 * Q.n_dr_0p1_0p2 - 0.61852
    if Q.n_dr_0p2_0p4 < 18.4:
        z += 0.0399 * Q.n_dr_0p2_0p4 - 0.73416
    if Q.sum_pt >= 1160.0:
        z += 0.0105 * Q.sum_pt - 12.18
    if Q.sum_pt_top50 < 1050.0:
        z += -0.0263 * Q.sum_pt_top50 + 27.615
    if Q.sum_pt_top50 >= 1050.0:
        z += -0.0192 * Q.sum_pt_top50 + 20.16
    if Q.z_top50_slots < 0.969:
        z += 27.0 * Q.z_top50_slots - 26.163
    if Q.log_sum_pt > 7.03 and Q.sj3_pair_mass_max > 38.9:
        z += 0.0452 * (Q.log_sum_pt - 7.03) * (Q.sj3_pair_mass_max - 38.9)
    return max(0.0, z)


def neuron_9(Q):
    z = 5.1
    if Q.LHA < 0.205:
        z += -16.2 * Q.LHA + 3.321
    if Q.sum_z_dr < 0.0364:
        z += 106.4 * Q.sum_z_dr - 4.28255
    if 0.0364 <= Q.sum_z_dr < 0.0475:
        z += 36.9 * Q.sum_z_dr - 1.75275
    if Q.sum_z_dr2_top40 < 0.00908:
        z += -423.0 * Q.sum_z_dr2_top40 + 3.84084
    if Q.mass < 60.3:
        z += -0.0219 * Q.mass + 3.22293
    if 60.3 <= Q.mass < 85.2:
        z += -0.0764 * Q.mass + 6.50928
    if 99.4 <= Q.mass < 147.0:
        z += 0.0122 * Q.mass - 1.21268
    if 147.0 <= Q.mass < 155.0:
        z += -0.0452 * Q.mass + 7.22512
    if Q.mass >= 155.0:
        z += -0.0895 * Q.mass + 14.09162
    if 0.0598 <= Q.mass_over_sum_pt < 0.0971:
        z += -158.0 * Q.mass_over_sum_pt + 9.4484
    if 0.0971 <= Q.mass_over_sum_pt < 0.172:
        z += 12.0 * Q.mass_over_sum_pt - 7.0586
    if Q.mass_over_sum_pt >= 0.172:
        z += -42.3 * Q.mass_over_sum_pt + 2.281
    if Q.mass_over_sum_pt_sq < 0.00936:
        z += 1220.0 * Q.mass_over_sum_pt_sq - 11.4192
    if Q.mass_top5 >= 38.8:
        z += 0.00743 * Q.mass_top5 - 0.288284
    if Q.n_dr_0p1_0p2 >= 15.1:
        z += 0.0261 * Q.n_dr_0p1_0p2 - 0.39411
    if Q.psi_0p3 >= 0.992:
        z += 27.7 * Q.psi_0p3 - 27.4784
    if Q.pt_entropy < 2.34:
        z += -0.546 * Q.pt_entropy + 1.27764
    if Q.sum_pt_top15 >= 949.0:
        z += -0.00127 * Q.sum_pt_top15 + 1.20523
    if Q.sum_pt_top40 < 957.0:
        z += -0.011 * Q.sum_pt_top40 + 10.527
    if Q.lam1_plus_lam2 < 0.00355:
        z += -1370.0 * Q.lam1_plus_lam2 + 4.8635
    if Q.lam1 > 0.0142 and Q.soft5_pt < 3.27:
        z += 27.2 * (Q.lam1 - 0.0142) * (3.27 - Q.soft5_pt)
    if Q.mass_top30 < 117.0 and Q.n_dr_0p2_0p4 > 8.2:
        z += 0.000844 * (117.0 - Q.mass_top30) * (Q.n_dr_0p2_0p4 - 8.2)
    if Q.mass_top40 < 85.7 and Q.sum_pt < 1030.0:
        z += -0.000213 * (85.7 - Q.mass_top40) * (1030.0 - Q.sum_pt)
    if Q.mass_top40 < 88.7 and Q.sum_pt_top50 < 907.0:
        z += 0.000209 * (88.7 - Q.mass_top40) * (907.0 - Q.sum_pt_top50)
    if Q.sum_pt_top40 < 922.0 and Q.n_pt_above_10 < 20.0:
        z += -0.000625 * (922.0 - Q.sum_pt_top40) * (20.0 - Q.n_pt_above_10)
    if Q.sum_pt_top40 < 1000.0 and Q.soft4_pt > 1.64:
        z += -0.00232 * (1000.0 - Q.sum_pt_top40) * (Q.soft4_pt - 1.64)
    return max(0.0, z)


def neuron_10(Q):
    z = 4.33
    if Q.D2 < 3.7:
        z += -0.23 * Q.D2 + 0.851
    if Q.LHA >= 0.109:
        z += -25.6 * Q.LHA + 2.7904
    if Q.e2 < 0.0631:
        z += 24.5 * Q.e2 - 1.54595
    if Q.sum_zz_dr2 < 0.0262:
        z += -124.0 * Q.sum_zz_dr2 + 3.2488
    if Q.e3 >= 0.000163:
        z += 1360.0 * Q.e3 - 0.22168
    if Q.sum_z_dr < 0.0667:
        z += 96.3 * Q.sum_z_dr - 6.4521
    if 0.0667 <= Q.sum_z_dr < 0.067:
        z += 166.9 * Q.sum_z_dr - 11.16112
    if Q.sum_z_dr >= 0.067:
        z += 70.6 * Q.sum_z_dr - 4.70902
    if Q.mass < 64.6:
        z += -0.048 * Q.mass + 3.5472
    if 64.6 <= Q.mass < 73.9:
        z += -0.1028 * Q.mass + 7.08728
    if 73.9 <= Q.mass < 81.3:
        z += -0.0548 * Q.mass + 3.54008
    if 81.3 <= Q.mass < 89.6:
        z += -0.0038 * Q.mass - 0.60622
    if 89.6 <= Q.mass < 147.0:
        z += 0.0256 * Q.mass - 3.24046
    if 147.0 <= Q.mass < 159.0:
        z += -0.0844 * Q.mass + 12.92954
    if Q.mass >= 159.0:
        z += -0.1557 * Q.mass + 24.26624
    if Q.mass_over_sum_pt_sq >= 0.0268:
        z += -253.0 * Q.mass_over_sum_pt_sq + 6.7804
    if Q.mass_top10 < 71.5:
        z += 0.0112 * Q.mass_top10 - 0.8008
    if Q.mass_top20 >= 74.4:
        z += -0.0187 * Q.mass_top20 + 1.39128
    if Q.mass_top5 >= 24.7:
        z += 0.0118 * Q.mass_top5 - 0.29146
    if Q.mass_top50 >= 144.0:
        z += 0.0858 * Q.mass_top50 - 12.3552
    if Q.max_dr < 0.445:
        z += 1.27 * Q.max_dr - 0.56515
    if Q.n_dr_0p2_0p4 < 12.2:
        z += 0.0509 * Q.n_dr_0p2_0p4 - 0.62098
    if Q.n_real_top40 >= 33.6:
        z += 0.0617 * Q.n_real_top40 - 2.07312
    if Q.sj2_mass1 < 80.3:
        z += 0.0185 * Q.sj2_mass1 - 1.48555
    if Q.sum_pt < 978.0:
        z += 0.00466 * Q.sum_pt - 4.55748
    if Q.sum_pt_top5 < 871.0:
        z += -0.00231 * Q.sum_pt_top5 + 2.01201
    if Q.tau1 < 0.0519:
        z += 44.1 * Q.tau1 - 2.28879
    if Q.tau21 < 0.633:
        z += -1.55 * Q.tau21 + 0.98115
    if Q.z_dr_0_0p05 >= 0.93:
        z += -8.25 * Q.z_dr_0_0p05 + 7.6725
    if Q.z_dr_0p2_0p4 < 0.0977:
        z += -5.76 * Q.z_dr_0p2_0p4 + 0.562752
    if Q.z_top2_slots >= 0.326:
        z += -1.17 * Q.z_top2_slots + 0.38142
    if Q.D2 < 3.87 and Q.n_real_top40 > 32.2:
        z += -0.0263 * (3.87 - Q.D2) * (Q.n_real_top40 - 32.2)
    if Q.mass > 67.9 and Q.soft2_pt < 2.92:
        z += 0.00291 * (Q.mass - 67.9) * (2.92 - Q.soft2_pt)
    if Q.mass_top40 < 121.0 and Q.pt1_dr01 > 11.9:
        z += 0.000409 * (121.0 - Q.mass_top40) * (Q.pt1_dr01 - 11.9)
    if Q.psi_0p3 > 0.998 and Q.n_real_top30 > 22.2:
        z += -14.9 * (Q.psi_0p3 - 0.998) * (Q.n_real_top30 - 22.2)
    if Q.sj2_mass1 < 87.2 and Q.sj2_mass2 > 6.95:
        z += 0.00021 * (87.2 - Q.sj2_mass1) * (Q.sj2_mass2 - 6.95)
    return max(0.0, z)


def neuron_11(Q):
    z = 1.69
    if Q.LHA >= 0.179:
        z += -10.4 * Q.LHA + 1.8616
    if Q.e2 < 0.0267:
        z += -32.8 * Q.e2 + 0.3383
    if 0.0267 <= Q.e2 < 0.0421:
        z += 34.9 * Q.e2 - 1.46929
    if Q.sum_z_dr < 0.0348:
        z += 8.3 * Q.sum_z_dr - 1.90292
    if 0.0348 <= Q.sum_z_dr < 0.0764:
        z += 38.8 * Q.sum_z_dr - 2.96432
    if Q.sum_z_dr2_top10 < 0.00342:
        z += -70.0 * Q.sum_z_dr2_top10 - 0.31401
    if 0.00342 <= Q.sum_z_dr2_top10 < 0.00771:
        z += 129.0 * Q.sum_z_dr2_top10 - 0.99459
    if Q.sum_z_dr2_top15 >= 0.00805:
        z += 42.7 * Q.sum_z_dr2_top15 - 0.343735
    if Q.mass < 80.9:
        z += -0.0001 * Q.mass + 0.56009
    if 80.9 <= Q.mass < 99.8:
        z += -0.044 * Q.mass + 4.1116
    if 99.8 <= Q.mass < 101.0:
        z += 0.233 * Q.mass - 23.533
    if Q.mass >= 131.0:
        z += 0.0215 * Q.mass - 2.8165
    if Q.mass_over_sum_pt >= 0.0798:
        z += -56.4 * Q.mass_over_sum_pt + 4.50072
    if Q.mass_over_sum_pt_sq >= 0.00635:
        z += 214.0 * Q.mass_over_sum_pt_sq - 1.3589
    if Q.mass_top10 < 40.4:
        z += -0.0016 * Q.mass_top10 + 0.82293
    if 40.4 <= Q.mass_top10 < 71.1:
        z += -0.0247 * Q.mass_top10 + 1.75617
    if Q.mass_top40 >= 84.1:
        z += -0.0213 * Q.mass_top40 + 1.79133
    if Q.n_dr_0p1_0p2 >= 8.59:
        z += -0.0214 * Q.n_dr_0p1_0p2 + 0.183826
    if Q.n_dr_0p2_0p4 < 5.46:
        z += -0.1 * Q.n_dr_0p2_0p4 + 0.546
    if Q.n_dr_0p2_0p4 >= 11.3:
        z += -0.0477 * Q.n_dr_0p2_0p4 + 0.53901
    if Q.n_for_90pct >= 19.1:
        z += -0.0257 * Q.n_for_90pct + 0.49087
    if 0.988 <= Q.psi_0p3 < 0.997:
        z += -18.7 * Q.psi_0p3 + 18.4756
    if Q.psi_0p3 >= 0.997:
        z += 38.4 * Q.psi_0p3 - 38.4531
    if Q.pt1_dr01 >= 9.57:
        z += 0.0103 * Q.pt1_dr01 - 0.098571
    if Q.z_dr_0_0p05 < 0.138:
        z += 1.9 * Q.z_dr_0_0p05 - 0.2622
    if Q.mass < 80.2 and Q.D2 < 4.32:
        z += -0.0039 * (80.2 - Q.mass) * (4.32 - Q.D2)
    if Q.mass < 102.0 and Q.mass_top10 > 40.4:
        z += 0.00116 * (102.0 - Q.mass) * (Q.mass_top10 - 40.4)
    if Q.mass_over_sum_pt_sq < 0.00902 and Q.z_dr_0p1_0p2 > 0.0762:
        z += 603.0 * (0.00902 - Q.mass_over_sum_pt_sq) * (Q.z_dr_0p1_0p2 - 0.0762)
    if Q.n_dr_0p2_0p4 < 8.29 and Q.sum_z_dr2 < 0.00605:
        z += -30.2 * (8.29 - Q.n_dr_0p2_0p4) * (0.00605 - Q.sum_z_dr2)
    if Q.n_dr_0p2_0p4 < 9.27 and Q.n_dr_0p1_0p2 < 21.1:
        z += 0.00605 * (9.27 - Q.n_dr_0p2_0p4) * (21.1 - Q.n_dr_0p1_0p2)
    if Q.tau21_b2 < 0.198 and Q.z_5 > 0.0251:
        z += 64.6 * (0.198 - Q.tau21_b2) * (Q.z_5 - 0.0251)
    if Q.z_dr_0_0p05 < 0.166 and Q.max_dr > 0.235:
        z += 7.05 * (0.166 - Q.z_dr_0_0p05) * (Q.max_dr - 0.235)
    if Q.z_top30_slots > 0.976 and Q.C2_b2 < 0.053:
        z += 226.0 * (Q.z_top30_slots - 0.976) * (0.053 - Q.C2_b2)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.0159
    if Q.LHA >= 0.336:
        z += 14.3 * Q.LHA - 4.8048
    if Q.lam2 >= 0.00046:
        z += -85.5 * Q.lam2 + 0.03933
    if Q.mass < 61.7:
        z += -0.0415 * Q.mass + 4.4072
    if 61.7 <= Q.mass < 80.7:
        z += -0.0831 * Q.mass + 6.97392
    if 80.7 <= Q.mass < 88.2:
        z += -0.0357 * Q.mass + 3.14874
    if 106.0 <= Q.mass < 187.0:
        z += 0.028 * Q.mass - 2.968
    if Q.mass >= 187.0:
        z += 0.0133 * Q.mass - 0.2191
    if Q.mass_over_sum_pt >= 0.0773:
        z += -24.6 * Q.mass_over_sum_pt + 1.90158
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.0193 * Q.n_dr_0p2_0p4 + 0.2895
    if Q.pt_entropy >= 1.99:
        z += -0.216 * Q.pt_entropy + 0.42984
    if Q.sum_pt >= 933.0:
        z += -0.000873 * Q.sum_pt + 0.814509
    if Q.z_top50_slots < 0.999:
        z += 7.29 * Q.z_top50_slots - 7.28271
    if Q.mass < 90.1 and Q.psi_0p3 < 1.0:
        z += -1.12 * (90.1 - Q.mass) * (1.0 - Q.psi_0p3)
    if Q.mass < 79.3 and Q.z_dr_0p1_0p2 < 0.0962:
        z += -0.215 * (79.3 - Q.mass) * (0.0962 - Q.z_dr_0p1_0p2)
    return max(0.0, z)


def neuron_13(Q):
    z = 3.3
    if Q.sum_z_dr >= 0.0703:
        z += 9.16 * Q.sum_z_dr - 0.643948
    if Q.sum_z_dr2_top40 < 0.00841:
        z += -55.7 * Q.sum_z_dr2_top40 + 0.468437
    if Q.sum_z_dr2_top50 >= 0.0079:
        z += -58.6 * Q.sum_z_dr2_top50 + 0.46294
    if Q.lam1 >= 0.018:
        z += 24.2 * Q.lam1 - 0.4356
    if Q.log_sum_pt < 6.86:
        z += 110.9 * Q.log_sum_pt - 777.274
    if 6.86 <= Q.log_sum_pt < 7.06:
        z += 82.5 * Q.log_sum_pt - 582.45
    if 76.3 <= Q.mass < 102.0:
        z += 0.0302 * Q.mass - 2.30426
    if 102.0 <= Q.mass < 146.0:
        z += 0.0025 * Q.mass + 0.52114
    if 146.0 <= Q.mass < 157.0:
        z += -0.1795 * Q.mass + 27.09314
    if Q.mass >= 157.0:
        z += -0.2378 * Q.mass + 36.24624
    if Q.mass_over_sum_pt_sq >= 0.0281:
        z += 121.0 * Q.mass_over_sum_pt_sq - 3.4001
    if Q.mass_top5 >= 33.9:
        z += 0.00588 * Q.mass_top5 - 0.199332
    if Q.mass_top50 >= 142.0:
        z += 0.113 * Q.mass_top50 - 16.046
    if Q.n_dr_0p1_0p2 < 24.4:
        z += 0.00784 * Q.n_dr_0p1_0p2 - 0.191296
    if Q.n_for_90pct < 13.7:
        z += 0.0906 * Q.n_for_90pct - 1.24122
    if Q.sj2_zsoft < 0.117:
        z += 1.09 * Q.sj2_zsoft - 0.14388
    if 0.117 <= Q.sj2_zsoft < 0.132:
        z += 0.695 * Q.sj2_zsoft - 0.097665
    if Q.sj2_zsoft >= 0.132:
        z += -0.395 * Q.sj2_zsoft + 0.046215
    if Q.sum_pt < 951.0:
        z += -0.0824 * Q.sum_pt + 91.4461
    if 951.0 <= Q.sum_pt < 1010.0:
        z += -0.0443 * Q.sum_pt + 55.213
    if 1010.0 <= Q.sum_pt < 1160.0:
        z += -0.0698 * Q.sum_pt + 80.968
    if Q.sum_pt_top10 >= 557.0:
        z += -0.00128 * Q.sum_pt_top10 + 0.71296
    if Q.sum_pt_top15 < 952.0:
        z += 0.00243 * Q.sum_pt_top15 - 2.31336
    if Q.sum_pt_top20 >= 1060.0:
        z += 0.00386 * Q.sum_pt_top20 - 4.0916
    if Q.sum_pt_top30 >= 943.0:
        z += -0.00376 * Q.sum_pt_top30 + 3.54568
    if Q.sum_pt_top50 < 1010.0:
        z += -0.0135 * Q.sum_pt_top50 + 13.635
    if Q.log_sum_pt > 7.21 and Q.C3 < 0.00379:
        z += -1170.0 * (Q.log_sum_pt - 7.21) * (0.00379 - Q.C3)
    if Q.mass > 173.0 and Q.pt_9 < 39.5:
        z += 0.00879 * (Q.mass - 173.0) * (39.5 - Q.pt_9)
    if Q.mass > 174.0 and Q.z_9 < 0.0293:
        z += -9.67 * (Q.mass - 174.0) * (0.0293 - Q.z_9)
    if Q.n_for_90pct < 13.8 and Q.D2 < 3.95:
        z += 0.0298 * (13.8 - Q.n_for_90pct) * (3.95 - Q.D2)
    if Q.psi_0p3 > 0.981 and Q.max_dr > 0.224:
        z += 24.4 * (Q.psi_0p3 - 0.981) * (Q.max_dr - 0.224)
    if Q.sum_pt < 1030.0 and Q.dr_12 < 0.122:
        z += 0.0365 * (1030.0 - Q.sum_pt) * (0.122 - Q.dr_12)
    if Q.sum_pt < 1010.0 and Q.z_9 < 0.0174:
        z += 1.06 * (1010.0 - Q.sum_pt) * (0.0174 - Q.z_9)
    if Q.sum_pt_top40 < 1040.0 and Q.D3 < 0.529:
        z += 0.00973 * (1040.0 - Q.sum_pt_top40) * (0.529 - Q.D3)
    return max(0.0, z)


def neuron_14(Q):
    z = -1.82
    if 0.187 <= Q.LHA < 0.271:
        z += -20.5 * Q.LHA + 3.8335
    if 0.271 <= Q.LHA < 0.322:
        z += -36.6 * Q.LHA + 8.1966
    if Q.LHA >= 0.322:
        z += -50.3 * Q.LHA + 12.608
    if Q.e2 < 0.0302:
        z += -39.6 * Q.e2 + 1.20384
    if 0.0302 <= Q.e2 < 0.0304:
        z += -62.5 * Q.e2 + 1.89542
    if Q.e2 >= 0.0304:
        z += -22.9 * Q.e2 + 0.69158
    if Q.sum_z_dr >= 0.0351:
        z += 95.6 * Q.sum_z_dr - 3.35556
    if Q.sum_z_dr2_top20 < 0.00627:
        z += -3.0 * Q.sum_z_dr2_top20 - 0.426596
    if 0.00627 <= Q.sum_z_dr2_top20 < 0.0109:
        z += 96.2 * Q.sum_z_dr2_top20 - 1.04858
    if Q.sum_z_dr2_top40 < 0.00873:
        z += 232.0 * Q.sum_z_dr2_top40 - 2.02536
    if Q.sum_z_dr2_top5 < 0.00722:
        z += -34.0 * Q.sum_z_dr2_top5 + 0.24548
    if 6.97 <= Q.log_sum_pt < 7.06:
        z += 12.2 * Q.log_sum_pt - 85.034
    if Q.log_sum_pt >= 7.06:
        z += 0.1 * Q.log_sum_pt + 0.392
    if Q.mass < 57.3:
        z += 0.0492 * Q.mass - 5.42133
    if 57.3 <= Q.mass < 79.1:
        z += 0.0738 * Q.mass - 6.83091
    if 79.1 <= Q.mass < 90.8:
        z += 0.0849 * Q.mass - 7.70892
    if Q.mass >= 96.1:
        z += -0.0389 * Q.mass + 3.73829
    if Q.mass_over_sum_pt < 0.091:
        z += 64.2 * Q.mass_over_sum_pt - 5.8422
    if Q.mass_over_sum_pt >= 0.0971:
        z += -59.3 * Q.mass_over_sum_pt + 5.75803
    if Q.mass_over_sum_pt_sq < 0.00949:
        z += -712.0 * Q.mass_over_sum_pt_sq + 6.75688
    if Q.mass_top10 >= 84.9:
        z += 0.0126 * Q.mass_top10 - 1.06974
    if 79.4 <= Q.mass_top30 < 145.0:
        z += 0.0166 * Q.mass_top30 - 1.31804
    if Q.mass_top30 >= 145.0:
        z += 0.01313 * Q.mass_top30 - 0.81489
    if Q.mass_top40 < 81.5:
        z += -0.0232 * Q.mass_top40 + 1.8908
    if Q.mass_top50 >= 78.6:
        z += 0.0161 * Q.mass_top50 - 1.26546
    if Q.sj2_dr >= 0.143:
        z += 1.01 * Q.sj2_dr - 0.14443
    if Q.sj2_mass1 < 65.4:
        z += -0.0199 * Q.sj2_mass1 + 1.30146
    if Q.sj2_mass2 >= 6.11:
        z += -0.022 * Q.sj2_mass2 + 0.13442
    if Q.soft6_pt < 2.03:
        z += 0.191 * Q.soft6_pt - 0.38773
    if Q.sum_pt < 1070.0:
        z += 0.0134 * Q.sum_pt - 14.338
    if Q.sum_pt >= 1160.0:
        z += 0.00893 * Q.sum_pt - 10.3588
    if Q.sum_pt_top20 < 945.0:
        z += 0.00266 * Q.sum_pt_top20 - 2.5137
    if Q.sum_pt_top20 >= 1060.0:
        z += 0.00118 * Q.sum_pt_top20 - 1.2508
    if Q.sum_pt_top40 < 972.0:
        z += -0.01013 * Q.sum_pt_top40 + 11.0406
    if 972.0 <= Q.sum_pt_top40 < 1050.0:
        z += -0.01696 * Q.sum_pt_top40 + 17.67936
    if 1050.0 <= Q.sum_pt_top40 < 1140.0:
        z += -0.01132 * Q.sum_pt_top40 + 11.75736
    if Q.sum_pt_top40 >= 1140.0:
        z += -0.0103 * Q.sum_pt_top40 + 10.59456
    if Q.sum_pt_top50 < 1020.0:
        z += -0.00546 * Q.sum_pt_top50 + 5.5692
    if Q.tau1 >= 0.0813:
        z += 29.1 * Q.tau1 - 2.36583
    if Q.tau21 < 0.455:
        z += 1.83 * Q.tau21 - 0.83265
    if Q.tau21_b2 >= 0.275:
        z += 1.29 * Q.tau21_b2 - 0.35475
    if Q.z_top10_slots >= 0.914:
        z += -4.68 * Q.z_top10_slots + 4.27752
    if Q.z_top40_slots < 0.968:
        z += 6.48 * Q.z_top40_slots - 6.2856
    if 0.968 <= Q.z_top40_slots < 0.97:
        z += 18.38 * Q.z_top40_slots - 17.8048
    if Q.z_top40_slots >= 0.97:
        z += 11.9 * Q.z_top40_slots - 11.5192
    if Q.z_top50_slots < 0.969:
        z += 9.12 * Q.z_top50_slots - 8.83728
    if Q.sum_z_dr2_top20 < 0.0122 and Q.mass_top3 > 26.0:
        z += -1.9 * (0.0122 - Q.sum_z_dr2_top20) * (Q.mass_top3 - 26.0)
    if Q.mass < 94.8 and Q.soft8_z < 0.00256:
        z += 6.97 * (94.8 - Q.mass) * (0.00256 - Q.soft8_z)
    if Q.psi_0p3 > 0.993 and Q.N3 < 1.09:
        z += 34.5 * (Q.psi_0p3 - 0.993) * (1.09 - Q.N3)
    if Q.psi_0p3 > 0.996 and Q.pt_3 > 62.1:
        z += 0.733 * (Q.psi_0p3 - 0.996) * (Q.pt_3 - 62.1)
    if Q.psi_0p3 > 0.992 and Q.soft6_pt < 4.18:
        z += 11.2 * (Q.psi_0p3 - 0.992) * (4.18 - Q.soft6_pt)
    if Q.tau21_b2 < 0.355 and Q.sum_z_dr2_top40 < 0.0089:
        z += -250.0 * (0.355 - Q.tau21_b2) * (0.0089 - Q.sum_z_dr2_top40)
    if Q.tau21_b2 < 0.371 and Q.z_6 < 0.0574:
        z += 28.1 * (0.371 - Q.tau21_b2) * (0.0574 - Q.z_6)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.0651
    if Q.LHA < 0.203:
        z += 5.8 * Q.LHA - 2.8774
    if 0.203 <= Q.LHA < 0.303:
        z += 17.0 * Q.LHA - 5.151
    if Q.e2 < 0.0388:
        z += -21.7 * Q.e2 + 1.20001
    if 0.0388 <= Q.e2 < 0.0553:
        z += -60.3 * Q.e2 + 2.69769
    if Q.e2 >= 0.0553:
        z += -38.6 * Q.e2 + 1.49768
    if Q.e3 >= 9.44e-06:
        z += 2090.0 * Q.e3 - 0.0197296
    if Q.sum_z_dr < 0.0399:
        z += -40.0 * Q.sum_z_dr + 3.18
    if 0.0399 <= Q.sum_z_dr < 0.0795:
        z += -55.6 * Q.sum_z_dr + 3.80244
    if Q.sum_z_dr >= 0.0795:
        z += -15.6 * Q.sum_z_dr + 0.62244
    if Q.sum_z_dr2_top5 < 0.0025:
        z += 86.9 * Q.sum_z_dr2_top5 + 0.126702
    if 0.0025 <= Q.sum_z_dr2_top5 < 0.00842:
        z += -58.1 * Q.sum_z_dr2_top5 + 0.489202
    if Q.sum_z_dr2_top50 < 0.00824:
        z += 79.4 * Q.sum_z_dr2_top50 - 0.654256
    if Q.lam1 < 0.00457:
        z += -121.0 * Q.lam1 + 0.55297
    if Q.lam2 < 0.0022:
        z += -241.0 * Q.lam2 + 0.5302
    if Q.log_sum_pt < 7.01:
        z += -8.17 * Q.log_sum_pt + 57.2717
    if Q.mass < 88.5:
        z += 0.0171 * Q.mass - 1.51335
    if Q.mass_top10 >= 24.0:
        z += -0.00508 * Q.mass_top10 + 0.12192
    if Q.mass_top30 >= 79.5:
        z += 0.0128 * Q.mass_top30 - 1.0176
    if Q.max_dr < 0.299:
        z += 1.87 * Q.max_dr - 0.55913
    if Q.n_dr_0p1_0p2 < 11.6:
        z += -0.0222 * Q.n_dr_0p1_0p2 + 0.3774
    if 11.6 <= Q.n_dr_0p1_0p2 < 17.0:
        z += -0.0113 * Q.n_dr_0p1_0p2 + 0.25096
    if Q.n_dr_0p1_0p2 >= 17.0:
        z += 0.0109 * Q.n_dr_0p1_0p2 - 0.12644
    if Q.n_particles < 39.6:
        z += -0.0142 * Q.n_particles + 0.56232
    if Q.psi_0p1 < 0.715:
        z += 0.452 * Q.psi_0p1 - 0.32318
    if Q.psi_0p1 >= 0.91:
        z += -9.08 * Q.psi_0p1 + 8.2628
    if Q.psi_0p3 >= 0.988:
        z += -27.0 * Q.psi_0p3 + 26.676
    if 58.2 <= Q.sd_mass < 86.2:
        z += -0.023 * Q.sd_mass + 1.3386
    if Q.sd_mass >= 86.2:
        z += 0.0032 * Q.sd_mass - 0.91984
    if 0.164 <= Q.sd_rg < 0.215:
        z += -3.61 * Q.sd_rg + 0.59204
    if 0.215 <= Q.sd_rg < 0.287:
        z += 11.59 * Q.sd_rg - 2.67596
    if Q.sd_rg >= 0.287:
        z += -0.91 * Q.sd_rg + 0.91154
    if Q.soft2_pt < 0.84:
        z += 0.137 * Q.soft2_pt - 0.11508
    if Q.sum_pt < 1000.0:
        z += 0.0108 * Q.sum_pt - 10.8
    if Q.sum_pt_top40 < 1040.0:
        z += 0.00343 * Q.sum_pt_top40 - 3.5672
    if Q.tau1 < 0.0722:
        z += 23.2 * Q.tau1 - 1.67504
    if Q.z_dr_0p1_0p2 < 0.076:
        z += -8.83 * Q.z_dr_0p1_0p2 + 0.95366
    if 0.076 <= Q.z_dr_0p1_0p2 < 0.218:
        z += -1.99 * Q.z_dr_0p1_0p2 + 0.43382
    if Q.z_top20_slots < 0.9:
        z += 1.03 * Q.z_top20_slots - 0.94245
    if 0.9 <= Q.z_top20_slots < 0.915:
        z += 3.44 * Q.z_top20_slots - 3.11145
    if Q.z_top20_slots >= 0.915:
        z += 2.41 * Q.z_top20_slots - 2.169
    if Q.sum_z_dr2_top20 < 0.00265 and Q.max_dr < 0.428:
        z += 810.0 * (0.00265 - Q.sum_z_dr2_top20) * (0.428 - Q.max_dr)
    if Q.sum_z_dr2_top5 < 0.00822 and Q.psi_0p2 < 0.95:
        z += -414.0 * (0.00822 - Q.sum_z_dr2_top5) * (0.95 - Q.psi_0p2)
    if Q.sum_z_dr2_top5 < 0.0062 and Q.z_6 < 0.0461:
        z += 842.0 * (0.0062 - Q.sum_z_dr2_top5) * (0.0461 - Q.z_6)
    if Q.n_dr_0p1_0p2 < 16.4 and Q.absphi_0 < 0.0316:
        z += -0.459 * (16.4 - Q.n_dr_0p1_0p2) * (0.0316 - Q.absphi_0)
    if Q.sj2_dr > 0.231 and Q.C2_b2 < 0.0423:
        z += 111.0 * (Q.sj2_dr - 0.231) * (0.0423 - Q.C2_b2)
    if Q.z_dr_0_0p05 < 1.05 and Q.sum_pt < 968.0:
        z += 0.00465 * (1.05 - Q.z_dr_0_0p05) * (968.0 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.12 and Q.n_dr_0p2_0p4 > 3.57:
        z += -0.274 * (0.12 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 3.57)
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
