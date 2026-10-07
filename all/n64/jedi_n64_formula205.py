"""JEDI-linear jet tagger, 64 particles, 3 features: one term per observable per neuron (from the 250), re-tuned on the network's predictions (all observables), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 93.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.sj2_mass1              mass of the harder of 2 subjets [GeV]
  Q.sj2_mass2              mass of the softer of 2 subjets [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_for_90pct            number of hardest particles that carry 90% of the jet pT
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_entropy             pT entropy −Σ zᵢ ln zᵢ
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.soft6_z                pT share of the 6. softest real particle (0 if it is among the 15 hardest)
  Q.soft7_z                pT share of the 7. softest real particle (0 if it is among the 15 hardest)
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.zdr_0                  pT share × ΔR of particle 0 (its term in ΣzΔR)
  Q.z_2nd                  2nd-largest pT share
  Q.pt2_over_pt0           pT2 / pT0
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.sd_zg                  momentum sharing of the soft-drop splitting
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.sum_z_dr2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.sum_z_dr2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.sum_z_dr2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.sum_z_dr2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.sum_z_dr2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.sum_z_dr2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.sum_z_dr2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.sum_z_dr2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.sum_z_dr                  pT-weighted mean ΔR
  Q.sum_z_dr2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.sum_zz_dr2                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam1_plus_lam2                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau1                   N-subjettiness τ1 (β=1)
  Q.tau2                   N-subjettiness τ2 (β=1)
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau21_b2               N-subjettiness τ2/τ1 with β = 2 (same axes)
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top20=mass_of(20),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        sj2_mass1=subjets(2)["mass"][0],
        sj2_mass2=subjets(2)["mass"][1],
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_for_90pct=ncum(0.9),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_entropy=-sum(z[i] * math.log(z[i]) for i in real),
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft5_z=softp(5, 'z'),
        soft6_z=softp(6, 'z'),
        soft7_z=softp(7, 'z'),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        zdr_0=z[0] * dr[0],
        z_2nd=zs[1],
        pt2_over_pt0=pt[2] / max(pt[0], 1e-9),
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        sd_zg=softdrop("zg"),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        eta_0=eta[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        sum_z_dr2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        sum_z_dr2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        sum_z_dr2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        sum_z_dr2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        sum_z_dr2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        sum_z_dr2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        sum_z_dr2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        sum_z_dr2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        sum_z_dr=sum(z[i] * dr[i] for i in P),
        sum_z_dr2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        sum_zz_dr2=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        lam1_plus_lam2=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 0.4139135
    if Q.sum_zz_dr2 < 0.00614:
        z += 525.2961 * Q.sum_zz_dr2 - 3.225318
    if Q.sum_z_dr2_top20 < 0.00652:
        z += 85.77631 * Q.sum_z_dr2_top20 - 0.5592615
    if Q.mass >= 82.83567:
        z += -0.2097895 * Q.mass + 17.37806
    if Q.mass_over_sum_pt_sq < 0.00766:
        z += -549.7149 * Q.mass_over_sum_pt_sq + 4.210816
    if Q.mass_top50 >= 82.0:
        z += 0.02307594 * Q.mass_top50 - 1.892227
    if Q.n_dr_0p2_0p4 < 15.3:
        z += -0.046476 * Q.n_dr_0p2_0p4 + 0.7110828
    if Q.psi_0p3 >= 0.996:
        z += 65.72284 * Q.psi_0p3 - 65.45995
    if Q.sum_pt < 1010.0:
        z += 0.004444485 * Q.sum_pt - 4.488929
    if Q.sum_pt_top50 < 1160.0:
        z += -0.001271991 * Q.sum_pt_top50 + 1.47551
    if Q.tau1 < 0.0699:
        z += 14.52344 * Q.tau1 - 1.015189
    if Q.z_dr_0p2_0p4 < 0.0913:
        z += 3.441537 * Q.z_dr_0p2_0p4 - 0.3142123
    if Q.z_top50_slots < 0.991:
        z += 22.4058 * Q.z_top50_slots - 22.20415
    if Q.log_sum_pt < 7.0 and Q.sum_pt_top50 > 963.0:
        z += 0.08256399 * (7.0 - Q.log_sum_pt) * (Q.sum_pt_top50 - 963.0)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.2815105
    if Q.LHA < 0.258:
        z += -11.50476 * Q.LHA + 2.968229
    if Q.M3 < 0.032:
        z += 18.54785 * Q.M3 - 0.5935313
    if Q.sum_z_dr2_top15 < 0.000986:
        z += -685.6096 * Q.sum_z_dr2_top15 + 0.6760111
    if Q.lam1 < 0.00509:
        z += -174.8435 * Q.lam1 + 0.8899532
    if Q.log_sum_pt >= 6.879733:
        z += 30.25671 * Q.log_sum_pt - 208.1581
    if Q.mass < 122.0:
        z += 0.007498484 * Q.mass - 0.9148151
    if Q.mass_top20 < 49.0:
        z += 0.0194189 * Q.mass_top20 - 0.951526
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.05795461 * Q.n_dr_0p1_0p2 - 0.4636369
    if Q.n_dr_0p2_0p4 < 6.57:
        z += 0.09378979 * Q.n_dr_0p2_0p4 - 0.6161989
    if Q.n_for_90pct >= 10.4:
        z += -0.07083102 * Q.n_for_90pct + 0.7366426
    if Q.n_particles >= 38.4:
        z += 0.04469093 * Q.n_particles - 1.716132
    if Q.psi_0p3 >= 0.998:
        z += -133.4743 * Q.psi_0p3 + 133.2074
    if Q.pt_9 < 31.6:
        z += 0.01172907 * Q.pt_9 - 0.3706386
    if Q.pt_entropy >= 2.0:
        z += 1.787904 * Q.pt_entropy - 3.575809
    if Q.soft1_pt >= 0.8061523:
        z += 0.2512718 * Q.soft1_pt - 0.2025633
    if Q.sum_pt_top30 >= 1190.0:
        z += -0.006960272 * Q.sum_pt_top30 + 8.282724
    if Q.sum_pt_top40 < 1070.0:
        z += -0.007735642 * Q.sum_pt_top40 + 8.277137
    if Q.sum_pt_top50 >= 953.0:
        z += -0.0121404 * Q.sum_pt_top50 + 11.5698
    if Q.tau1 < 0.196:
        z += -7.233565 * Q.tau1 + 1.417779
    if Q.z_dr_0_0p05 >= 0.875:
        z += -6.764362 * Q.z_dr_0_0p05 + 5.918817
    if Q.z_top50_slots >= 0.96:
        z += -28.70608 * Q.z_top50_slots + 27.55784
    if Q.mass_top20 < 50.6 and Q.n_real_top40 > 29.8:
        z += 0.002023914 * (50.6 - Q.mass_top20) * (Q.n_real_top40 - 29.8)
    if Q.n_particles > 39.4 and Q.dr_0 < 0.111:
        z += 0.3475081 * (Q.n_particles - 39.4) * (0.111 - Q.dr_0)
    if Q.sj3_mass1 < 29.5 and Q.sj3_mass2 < 21.0:
        z += -0.002120786 * (29.5 - Q.sj3_mass1) * (21.0 - Q.sj3_mass2)
    if Q.z_top20_slots > 0.887 and Q.dr_2 < 0.0414:
        z += -181.3091 * (Q.z_top20_slots - 0.887) * (0.0414 - Q.dr_2)
    if Q.z_top30_slots > 0.939 and Q.C2 < 0.075:
        z += 247.0161 * (Q.z_top30_slots - 0.939) * (0.075 - Q.C2)
    if Q.z_top30_slots > 0.942 and Q.max_pair_mass > 9.45:
        z += 0.5994024 * (Q.z_top30_slots - 0.942) * (Q.max_pair_mass - 9.45)
    if Q.z_top30_slots > 0.914 and Q.ptdr0_3 > 6.21:
        z += 0.5627369 * (Q.z_top30_slots - 0.914) * (Q.ptdr0_3 - 6.21)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.03892876
    if Q.log_sum_pt > 6.91 and Q.sum_z_dr2_top15 < 0.0272:
        z += 142.8519 * (Q.log_sum_pt - 6.91) * (0.0272 - Q.sum_z_dr2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.1220715
    if Q.sum_z_dr2_top40 < 0.00669:
        z += -41.83441 * Q.sum_z_dr2_top40 + 0.2798722
    if Q.lam2 < 0.000625:
        z += -1565.631 * Q.lam2 + 0.9785194
    if Q.mass_top50 < 69.2:
        z += 0.01878848 * Q.mass_top50 - 1.300163
    if Q.n_particles < 46.1:
        z += -0.03524068 * Q.n_particles + 1.624595
    if Q.tau21 < 0.375:
        z += 2.889746 * Q.tau21 - 1.083655
    if Q.n_dr_0p1_0p2 < 11.2 and Q.psi_0p2 > 0.934:
        z += 0.7098261 * (11.2 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.934)
    if Q.n_dr_0p2_0p4 < 7.15 and Q.z_top50_slots > 0.98:
        z += 4.501082 * (7.15 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.98)
    if Q.tau21 < 0.406 and Q.lam1 > 0.00572:
        z += -42.87209 * (0.406 - Q.tau21) * (Q.lam1 - 0.00572)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.7382771
    if Q.C2 >= 0.0725:
        z += -2.551748 * Q.C2 + 0.1850017
    if Q.e2 >= 0.0568:
        z += -16.72274 * Q.e2 + 0.9498515
    if Q.sum_zz_dr2 >= 0.0103:
        z += 163.0172 * Q.sum_zz_dr2 - 1.679078
    if Q.sum_z_dr >= 0.121:
        z += -8.415109 * Q.sum_z_dr + 1.018228
    if Q.sum_z_dr2_top15 < 0.003248562:
        z += 58.87712 * Q.sum_z_dr2_top15 - 0.1912659
    if Q.lam1 >= 0.00794:
        z += -68.79964 * Q.lam1 + 0.5462692
    if Q.mass < 78.23178:
        z += 0.1652626 * Q.mass - 12.92879
    if Q.mass_top40 < 57.37264:
        z += -0.05940029 * Q.mass_top40 + 3.407952
    if Q.n_dr_0p2_0p4 < 24.2:
        z += -0.03440454 * Q.n_dr_0p2_0p4 + 0.8325898
    if Q.n_particles >= 23.1:
        z += -0.0343507 * Q.n_particles + 0.7935011
    if Q.psi_0p3 >= 0.997:
        z += 242.8424 * Q.psi_0p3 - 242.1139
    if Q.mass < 101.0 and Q.D2_b2 < 3.79:
        z += 0.01987387 * (101.0 - Q.mass) * (3.79 - Q.D2_b2)
    if Q.mass_top40 < 88.5 and Q.D2_b2 < 3.21:
        z += -0.04007774 * (88.5 - Q.mass_top40) * (3.21 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 < 22.7 and Q.n_dr_0p1_0p2 > 12.1:
        z += -0.001605845 * (22.7 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 12.1)
    if Q.n_particles > 27.6 and Q.n_dr_0_0p05 > 10.7:
        z += 0.0006925365 * (Q.n_particles - 27.6) * (Q.n_dr_0_0p05 - 10.7)
    if Q.n_particles > 20.4 and Q.soft1_pt < 2.44:
        z += 0.009153711 * (Q.n_particles - 20.4) * (2.44 - Q.soft1_pt)
    if Q.psi_0p3 > 0.997 and Q.n_dr_0_0p05 < 13.2:
        z += -16.34438 * (Q.psi_0p3 - 0.997) * (13.2 - Q.n_dr_0_0p05)
    if Q.sj2_dr > 0.232 and Q.D2_b2 < 20.6:
        z += -0.176866 * (Q.sj2_dr - 0.232) * (20.6 - Q.D2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.4750433
    if Q.sum_z_dr2_top50 >= 0.0191:
        z += -83.725 * Q.sum_z_dr2_top50 + 1.599147
    if Q.log_sum_pt >= 6.903354:
        z += -29.07899 * Q.log_sum_pt + 200.7426
    if Q.mass >= 172.4703:
        z += -0.1480389 * Q.mass + 25.53232
    if Q.mass_over_sum_pt >= 0.0781:
        z += 0.789795 * Q.mass_over_sum_pt - 0.06168299
    if Q.mass_top50 >= 158.0:
        z += -0.7037694 * Q.mass_top50 + 111.1956
    if Q.max_dr < 0.459:
        z += -2.990496 * Q.max_dr + 1.372638
    if Q.n_dr_0p1_0p2 < 22.6:
        z += 0.014338 * Q.n_dr_0p1_0p2 - 0.3240388
    if Q.n_dr_0p2_0p4 < 12.4:
        z += -0.02545342 * Q.n_dr_0p2_0p4 + 0.3156224
    if Q.n_particles < 47.5:
        z += -0.02716624 * Q.n_particles + 1.290397
    if Q.sd_mass < 97.07106:
        z += 0.005451908 * Q.sd_mass - 0.5292225
    if Q.sum_pt >= 1034.628:
        z += -0.009267671 * Q.sum_pt + 9.588589
    if Q.sum_pt_top50 >= 943.0:
        z += 0.02205192 * Q.sum_pt_top50 - 20.79496
    if Q.tau21 < 0.561:
        z += 0.7499594 * Q.tau21 - 0.4207272
    if Q.n_particles < 56.1 and Q.D2 < 2.7:
        z += -0.01255517 * (56.1 - Q.n_particles) * (2.7 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.9995688
    if Q.e2 < 0.04728002:
        z += -20.88303 * Q.e2 + 0.9873503
    if Q.sum_z_dr2_top20 >= 0.00733:
        z += 262.6236 * Q.sum_z_dr2_top20 - 1.925031
    if Q.sum_z_dr2_top30 >= 0.0275:
        z += -679.17 * Q.sum_z_dr2_top30 + 18.67717
    if Q.mass < 162.9488:
        z += 0.01870795 * Q.mass - 3.048438
    if Q.mass_over_sum_pt >= 0.121:
        z += -87.7671 * Q.mass_over_sum_pt + 10.61982
    if Q.mass_top50 < 73.3:
        z += -0.03101693 * Q.mass_top50 + 2.273541
    if Q.sj3_pair_mass_min >= 77.9:
        z += -0.8172998 * Q.sj3_pair_mass_min + 63.66765
    if Q.tau1 < 0.0533:
        z += -10.64608 * Q.tau1 + 0.5674361
    if Q.z_top40_slots < 0.931:
        z += -9.374983 * Q.z_top40_slots + 8.728109
    if Q.sum_z_dr2_top20 > 0.0035 and Q.C2_b2 > 0.00379:
        z += -3326.67 * (Q.sum_z_dr2_top20 - 0.0035) * (Q.C2_b2 - 0.00379)
    if Q.mass < 102.0 and Q.sum_pt < 1010.0:
        z += 0.0001142012 * (102.0 - Q.mass) * (1010.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.1181536
    if Q.mass < 121.0:
        z += -0.03097828 * Q.mass + 3.748372
    if Q.n_dr_0p2_0p4 < 8.15:
        z += -0.07731524 * Q.n_dr_0p2_0p4 + 0.6301192
    if Q.psi_0p3 >= 0.998:
        z += -67.10423 * Q.psi_0p3 + 66.97002
    if Q.sd_mass >= 97.2:
        z += -0.2037364 * Q.sd_mass + 19.80317
    if Q.tau1 < 0.1199231:
        z += 9.73361 * Q.tau1 - 1.167285
    if Q.tau21_b2 < 0.234:
        z += -12.12177 * Q.tau21_b2 + 2.836494
    if Q.lam1 < 0.00796 and Q.zdr_0 > 0.0152:
        z += -13497.97 * (0.00796 - Q.lam1) * (Q.zdr_0 - 0.0152)
    if Q.mass < 91.2 and Q.psi_0p3 > 0.964:
        z += -2.462205 * (91.2 - Q.mass) * (Q.psi_0p3 - 0.964)
    if Q.mass < 93.2 and Q.z_dr_0_0p05 < 0.503:
        z += -0.1124922 * (93.2 - Q.mass) * (0.503 - Q.z_dr_0_0p05)
    if Q.psi_0p3 > 0.997 and Q.z_top50_slots > 0.979:
        z += 8126.61 * (Q.psi_0p3 - 0.997) * (Q.z_top50_slots - 0.979)
    if Q.tau21_b2 < 0.23 and Q.mass_over_sum_pt_sq < 0.00808:
        z += -3164.053 * (0.23 - Q.tau21_b2) * (0.00808 - Q.mass_over_sum_pt_sq)
    if Q.tau21_b2 < 0.231 and Q.sum_pt < 1270.0:
        z += -0.02964914 * (0.231 - Q.tau21_b2) * (1270.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.753676
    if Q.sum_z_dr2_top20 >= 0.01062209:
        z += -22.48102 * Q.sum_z_dr2_top20 + 0.2387953
    if Q.sum_z_dr2_top30 < 0.00754:
        z += -267.2939 * Q.sum_z_dr2_top30 + 2.015396
    if Q.mass < 103.0:
        z += 0.05874338 * Q.mass - 6.050568
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.0410576 * Q.n_dr_0p2_0p4 - 0.8622096
    if Q.sj2_dr >= 0.23:
        z += 6.635064 * Q.sj2_dr - 1.526065
    if Q.sum_pt < 1030.0:
        z += -0.02114987 * Q.sum_pt + 21.78436
    if Q.sum_pt_top30 >= 987.0:
        z += 0.00149088 * Q.sum_pt_top30 - 1.471499
    if Q.sum_pt_top40 < 1030.0:
        z += 0.01044472 * Q.sum_pt_top40 - 10.75806
    if Q.lam1_plus_lam2 < 0.00987:
        z += 366.1653 * Q.lam1_plus_lam2 - 3.614052
    if Q.z_top50_slots >= 0.972:
        z += -28.60719 * Q.z_top50_slots + 27.80619
    if Q.sum_z_dr2_top40 > 0.00555 and Q.log_sum_pt < 6.99:
        z += -622.944 * (Q.sum_z_dr2_top40 - 0.00555) * (6.99 - Q.log_sum_pt)
    if Q.mass_over_sum_pt > 0.0785 and Q.sum_pt < 1100.0:
        z += 0.3335387 * (Q.mass_over_sum_pt - 0.0785) * (1100.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.z_dr_0p1_0p2 > 0.313:
        z += 0.2340203 * (18.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.313)
    if Q.tau2 < 0.0848 and Q.sj3_mass1 > 13.4:
        z += 0.4219508 * (0.0848 - Q.tau2) * (Q.sj3_mass1 - 13.4)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.1809424
    if Q.sum_z_dr >= 0.0976:
        z += 16.05155 * Q.sum_z_dr - 1.566632
    if Q.sum_z_dr2_top40 < 0.00617:
        z += -288.7856 * Q.sum_z_dr2_top40 + 1.781807
    if Q.mass < 91.02273:
        z += -0.0386662 * Q.mass + 3.519503
    if Q.sum_pt < 989.0:
        z += -0.01472134 * Q.sum_pt + 14.5594
    if Q.z_top40_slots >= 0.954:
        z += 2.00088 * Q.z_top40_slots - 1.908839
    if Q.mass_top40 < 116.0 and Q.n_dr_0p2_0p4 > 8.57:
        z += 0.001555072 * (116.0 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 8.57)
    if Q.mass_top40 < 97.9 and Q.sum_pt < 1020.0:
        z += -0.0001250154 * (97.9 - Q.mass_top40) * (1020.0 - Q.sum_pt)
    if Q.mass_top40 < 100.0 and Q.sum_pt_top40 < 961.0:
        z += 0.0003014032 * (100.0 - Q.mass_top40) * (961.0 - Q.sum_pt_top40)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.06436142
    if Q.D2 < 3.04:
        z += -0.3884609 * Q.D2 + 1.180921
    if Q.dr_0 < 0.0664:
        z += -9.583243 * Q.dr_0 + 0.6363274
    if Q.e2 >= 0.0128:
        z += 36.89 * Q.e2 - 0.472192
    if Q.sum_z_dr2_top20 < 0.00533:
        z += 381.7075 * Q.sum_z_dr2_top20 - 2.034501
    if Q.mass < 87.9:
        z += -0.0737701 * Q.mass + 6.484392
    if Q.mass_over_sum_pt >= 0.16:
        z += -53.74954 * Q.mass_over_sum_pt + 8.599926
    if Q.mass_top5 >= 17.5:
        z += 0.007662173 * Q.mass_top5 - 0.134088
    if Q.mass_top50 >= 138.0:
        z += -0.007182291 * Q.mass_top50 + 0.9911562
    if Q.n_dr_0p2_0p4 < 11.1:
        z += 0.06226451 * Q.n_dr_0p2_0p4 - 0.6911361
    if Q.pt_dispersion >= 0.27:
        z += -2.233321 * Q.pt_dispersion + 0.6029967
    if Q.sj2_mass1 < 65.0:
        z += 0.02698917 * Q.sj2_mass1 - 1.754296
    if Q.sum_pt < 984.0:
        z += 0.006713269 * Q.sum_pt - 6.605857
    if Q.sum_pt_top10 >= 941.0:
        z += -0.003853828 * Q.sum_pt_top10 + 3.626452
    if Q.sum_pt_top15 < 941.0:
        z += -0.004464619 * Q.sum_pt_top15 + 4.201206
    if Q.tau1 < 0.047:
        z += 38.47763 * Q.tau1 - 1.808448
    if Q.tau2 < 0.0485:
        z += -18.96164 * Q.tau2 + 0.9196394
    if Q.z_dr_0p2_0p4 < 0.0859:
        z += -6.526303 * Q.z_dr_0p2_0p4 + 0.5606095
    if Q.D2 < 2.99 and Q.sj2_dr > 0.192:
        z += -3.815099 * (2.99 - Q.D2) * (Q.sj2_dr - 0.192)
    if Q.sum_z_dr2_top10 < 0.0234 and Q.psi_0p3 > 0.998:
        z += -25020.73 * (0.0234 - Q.sum_z_dr2_top10) * (Q.psi_0p3 - 0.998)
    if Q.mass_top50 > 133.0 and Q.soft5_z > 0.00163:
        z += -10.51675 * (Q.mass_top50 - 133.0) * (Q.soft5_z - 0.00163)
    if Q.sj2_mass1 < 63.3 and Q.sj2_mass2 > 7.52:
        z += 0.0004516046 * (63.3 - Q.sj2_mass1) * (Q.sj2_mass2 - 7.52)
    if Q.sum_pt_top10 > 943.0 and Q.eta_0 < -0.078:
        z += -1.103058 * (Q.sum_pt_top10 - 943.0) * (-0.078 - Q.eta_0)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.4155552
    if Q.e2 < 0.0266:
        z += -139.9785 * Q.e2 + 3.723428
    if Q.sum_z_dr2_top30 < 0.00666:
        z += 519.3479 * Q.sum_z_dr2_top30 - 3.458857
    if Q.mass_top50 < 115.5062:
        z += -0.02543501 * Q.mass_top50 + 2.937901
    if Q.n_dr_0p1_0p2 < 16.7:
        z += -0.06090616 * Q.n_dr_0p1_0p2 + 1.017133
    if Q.n_dr_0p2_0p4 < 7.5:
        z += -0.2016936 * Q.n_dr_0p2_0p4 + 1.512702
    if Q.psi_0p3 >= 0.99:
        z += 13.91947 * Q.psi_0p3 - 13.78027
    if Q.z_top5_slots >= 0.552:
        z += -1.374331 * Q.z_top5_slots + 0.7586306
    if Q.mass < 119.0 and Q.sum_pt_top10 < 872.0:
        z += -3.432347e-05 * (119.0 - Q.mass) * (872.0 - Q.sum_pt_top10)
    if Q.n_dr_0p2_0p4 < 9.67 and Q.sum_z_dr2 < 0.0062:
        z += -34.93938 * (9.67 - Q.n_dr_0p2_0p4) * (0.0062 - Q.sum_z_dr2)
    if Q.n_dr_0p2_0p4 < 8.65 and Q.n_particles > 36.1:
        z += -0.002112162 * (8.65 - Q.n_dr_0p2_0p4) * (Q.n_particles - 36.1)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.4494128
    if Q.sum_z_dr2_top15 < 0.00586:
        z += 37.18649 * Q.sum_z_dr2_top15 - 0.2179128
    if Q.mass < 89.70783:
        z += -0.0527616 * Q.mass + 4.733129
    if Q.sd_mass >= 124.0:
        z += 0.03191979 * Q.sd_mass - 3.958054
    if Q.log_sum_pt < 6.85 and Q.lam2 < 0.0036:
        z += -223.4451 * (6.85 - Q.log_sum_pt) * (0.0036 - Q.lam2)
    if Q.mass < 90.0 and Q.psi_0p3 < 0.999:
        z += -1.066379 * (90.0 - Q.mass) * (0.999 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 123.0 and Q.sj3_pair_mass_min < 65.0:
        z += 0.0005400613 * (Q.sj3_pair_mass_max - 123.0) * (65.0 - Q.sj3_pair_mass_min)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.6078
    if Q.log_sum_pt < 6.92:
        z += 28.67461 * Q.log_sum_pt - 198.4283
    if Q.mass >= 162.9488:
        z += -0.218669 * Q.mass + 35.63186
    if Q.mass_top5 >= 40.1:
        z += 0.0105165 * Q.mass_top5 - 0.4217117
    if Q.sum_pt_top40 < 1020.0:
        z += -0.007053324 * Q.sum_pt_top40 + 7.194391
    if Q.log_sum_pt > 7.14 and Q.sd_zg > 0.446:
        z += -52.71157 * (Q.log_sum_pt - 7.14) * (Q.sd_zg - 0.446)
    if Q.mass > 169.0 and Q.D2 > 0.281:
        z += 0.02173715 * (Q.mass - 169.0) * (Q.D2 - 0.281)
    if Q.mass > 173.0 and Q.soft5_z > 0.000296:
        z += -324.1941 * (Q.mass - 173.0) * (Q.soft5_z - 0.000296)
    if Q.mass_top50 > 134.0 and Q.soft6_z > 0.00094:
        z += -85.25647 * (Q.mass_top50 - 134.0) * (Q.soft6_z - 0.00094)
    if Q.mass_top50 > 90.5 and Q.soft7_z < 0.00189:
        z += -22.18902 * (Q.mass_top50 - 90.5) * (0.00189 - Q.soft7_z)
    if Q.sum_pt < 1090.0 and Q.tau21_b2 < 1.0:
        z += -0.00654764 * (1090.0 - Q.sum_pt) * (1.0 - Q.tau21_b2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.6115586
    if Q.e2 < 0.0302:
        z += -84.69048 * Q.e2 + 2.557653
    if Q.sum_z_dr2_top3 < 0.00154:
        z += 540.5474 * Q.sum_z_dr2_top3 - 0.8324429
    if Q.sum_z_dr2_top30 < 0.00262:
        z += 4896.6 * Q.sum_z_dr2_top30 - 12.82909
    if Q.sum_z_dr2_top5 < 0.0067:
        z += -92.58298 * Q.sum_z_dr2_top5 + 0.6203059
    if Q.log_sum_pt < 6.94:
        z += -4.322447 * Q.log_sum_pt + 29.99778
    if Q.mass < 87.3546:
        z += 0.2343253 * Q.mass - 20.46939
    if Q.mass_over_sum_pt < 0.144:
        z += -14.65844 * Q.mass_over_sum_pt + 2.110816
    if Q.mass_top50 < 78.4:
        z += -0.09244273 * Q.mass_top50 + 7.24751
    if Q.psi_0p3 >= 0.993:
        z += 217.454 * Q.psi_0p3 - 215.9319
    if Q.N2 < 0.484 and Q.max_dr > 0.24:
        z += -9.21345 * (0.484 - Q.N2) * (Q.max_dr - 0.24)
    if Q.sum_z_dr2_top5 < 0.00654 and Q.n_pt_above_10 > 13.0:
        z += -9.298833 * (0.00654 - Q.sum_z_dr2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.mass_over_sum_pt < 0.0905 and Q.psi_0p2 > 0.914:
        z += -515.7283 * (0.0905 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.914)
    if Q.psi_0p2 > 0.978 and Q.pt2_over_pt0 > 0.121:
        z += 43.03205 * (Q.psi_0p2 - 0.978) * (Q.pt2_over_pt0 - 0.121)
    if Q.psi_0p3 > 0.993 and Q.sd_rg < 0.188:
        z += -344.3478 * (Q.psi_0p3 - 0.993) * (0.188 - Q.sd_rg)
    if Q.psi_0p3 > 0.993 and Q.z_2nd < 0.124:
        z += -1205.5 * (Q.psi_0p3 - 0.993) * (0.124 - Q.z_2nd)
    if Q.tau21_b2 < 0.322 and Q.n_real_top40 < 30.6:
        z += -0.2052701 * (0.322 - Q.tau21_b2) * (30.6 - Q.n_real_top40)
    if Q.tau21_b2 < 0.432 and Q.orientation_deg > -12.9:
        z += -0.01301374 * (0.432 - Q.tau21_b2) * (Q.orientation_deg - -12.9)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.3809624
    if Q.e2 >= 0.0294:
        z += -59.16821 * Q.e2 + 1.739545
    if Q.lam2 < 0.00181:
        z += -93.20281 * Q.lam2 + 0.1686971
    if Q.log_sum_pt < 7.03:
        z += -6.960981 * Q.log_sum_pt + 48.9357
    if Q.max_dr < 0.294:
        z += 2.764888 * Q.max_dr - 0.8128771
    if Q.n_dr_0p1_0p2 < 13.6:
        z += -0.04389163 * Q.n_dr_0p1_0p2 + 0.5969261
    if Q.n_dr_0p2_0p4 >= 8.79:
        z += -0.05945178 * Q.n_dr_0p2_0p4 + 0.5225811
    if Q.psi_0p2 >= 0.857:
        z += -5.936298 * Q.psi_0p2 + 5.087407
    if Q.psi_0p3 >= 0.989:
        z += -1.880737 * Q.psi_0p3 + 1.860048
    if Q.sd_mass < 87.75185:
        z += -0.01249892 * Q.sd_mass + 1.096803
    if Q.sd_rg >= 0.203:
        z += 5.787622 * Q.sd_rg - 1.174887
    if Q.sum_pt_top40 < 1020.0:
        z += 0.01316221 * Q.sum_pt_top40 - 13.42545
    if Q.tau1 < 0.0678:
        z += 21.18756 * Q.tau1 - 1.436517
    if Q.sj2_dr > 0.225 and Q.C2_b2 < 0.0493:
        z += 137.3484 * (Q.sj2_dr - 0.225) * (0.0493 - Q.C2_b2)
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
