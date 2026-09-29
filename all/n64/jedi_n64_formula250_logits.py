"""JEDI-linear jet tagger, 64 particles, 3 features: smaller version of the formula tuned on the network (step 4; all observables), as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.7% of jets.

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
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
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
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
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
  Q.psi_0p2                pT share within ΔR < 0.2 of the jet axis
  Q.psi_0p3                pT share within ΔR < 0.3 of the jet axis
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
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
        psi_0p2=sum(z[i] for i in real if dr[i] < 0.2),
        psi_0p3=sum(z[i] for i in real if dr[i] < 0.3),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau1=tau_n(1),
        tau2=tau_n(2),
        tau21=tau(2) / max(tau(1), 1e-12),
        tau21_b2=tau_n(2, 2) / max(tau_n(1, 2), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 0.798
    if Q.e2_sq < 0.00614:
        z += 315.0 * Q.e2_sq - 1.9341
    if Q.girth2_top20 < 0.00652:
        z += 73.9 * Q.girth2_top20 - 0.481828
    if Q.mass < 78.3:
        z += 0.0161 * Q.mass - 1.59551
    if 78.3 <= Q.mass < 93.8:
        z += -0.1399 * Q.mass + 10.61929
    if 93.8 <= Q.mass < 99.1:
        z += -0.1751 * Q.mass + 13.92105
    if Q.mass >= 99.1:
        z += -0.1912 * Q.mass + 15.51656
    if Q.mass_over_sum_pt_sq < 0.00766:
        z += -501.0 * Q.mass_over_sum_pt_sq + 3.83766
    if Q.mass_top50 >= 82.0:
        z += 0.0474 * Q.mass_top50 - 3.8868
    if Q.n_dr_0p2_0p4 < 15.3:
        z += -0.0494 * Q.n_dr_0p2_0p4 + 0.75582
    if Q.psi_0p3 >= 0.996:
        z += 55.8 * Q.psi_0p3 - 55.5768
    if Q.sum_pt < 1010.0:
        z += 0.00875 * Q.sum_pt - 8.8375
    if Q.sum_pt_top50 < 1160.0:
        z += -0.00302 * Q.sum_pt_top50 + 3.5032
    if Q.tau1 < 0.0699:
        z += 18.4 * Q.tau1 - 1.28616
    if Q.z_dr_0p2_0p4 < 0.0913:
        z += 4.77 * Q.z_dr_0p2_0p4 - 0.435501
    if Q.z_top50_slots < 0.991:
        z += 22.1 * Q.z_top50_slots - 21.9011
    if Q.log_sum_pt < 7.0 and Q.sum_pt_top50 > 963.0:
        z += 0.0566 * (7.0 - Q.log_sum_pt) * (Q.sum_pt_top50 - 963.0)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.06
    if Q.LHA < 0.258:
        z += -10.9 * Q.LHA + 2.8122
    if Q.M3 < 0.032:
        z += 16.4 * Q.M3 - 0.5248
    if Q.girth2_top15 < 0.000986:
        z += -638.0 * Q.girth2_top15 + 0.629068
    if Q.lam1 < 0.00509:
        z += -166.0 * Q.lam1 + 0.84494
    if 6.89 <= Q.log_sum_pt < 6.91:
        z += 19.4 * Q.log_sum_pt - 133.666
    if 6.91 <= Q.log_sum_pt < 6.97:
        z += 49.4 * Q.log_sum_pt - 340.966
    if 6.97 <= Q.log_sum_pt < 7.07:
        z += 26.2 * Q.log_sum_pt - 179.262
    if Q.log_sum_pt >= 7.07:
        z += 15.8 * Q.log_sum_pt - 105.734
    if Q.mass < 122.0:
        z += 0.0133 * Q.mass - 1.6226
    if Q.mass_top20 < 49.0:
        z += 0.0271 * Q.mass_top20 - 1.3279
    if Q.n_dr_0p1_0p2 < 8.0:
        z += 0.0558 * Q.n_dr_0p1_0p2 - 0.4464
    if Q.n_dr_0p2_0p4 < 6.57:
        z += 0.0777 * Q.n_dr_0p2_0p4 - 0.510489
    if Q.n_for_90pct >= 10.4:
        z += -0.0714 * Q.n_for_90pct + 0.74256
    if Q.n_particles >= 38.4:
        z += 0.0527 * Q.n_particles - 2.02368
    if Q.psi_0p3 >= 0.998:
        z += -164.0 * Q.psi_0p3 + 163.672
    if Q.pt_9 < 31.6:
        z += 0.0177 * Q.pt_9 - 0.55932
    if Q.pt_entropy >= 2.0:
        z += 1.69 * Q.pt_entropy - 3.38
    if Q.soft1_pt < 1.5:
        z += 0.04 * Q.soft1_pt - 1.061
    if 1.5 <= Q.soft1_pt < 2.27:
        z += 1.3 * Q.soft1_pt - 2.951
    if Q.sum_pt_top30 >= 1190.0:
        z += 0.0035 * Q.sum_pt_top30 - 4.165
    if Q.sum_pt_top40 < 1070.0:
        z += -0.00749 * Q.sum_pt_top40 + 8.0143
    if Q.sum_pt_top50 >= 953.0:
        z += -0.0114 * Q.sum_pt_top50 + 10.8642
    if Q.tau1 < 0.196:
        z += -12.0 * Q.tau1 + 2.352
    if Q.z_dr_0_0p05 >= 0.875:
        z += -6.1 * Q.z_dr_0_0p05 + 5.3375
    if Q.z_top50_slots >= 0.96:
        z += -25.4 * Q.z_top50_slots + 24.384
    if Q.mass_top20 < 50.6 and Q.n_real_top40 > 29.8:
        z += 0.00282 * (50.6 - Q.mass_top20) * (Q.n_real_top40 - 29.8)
    if Q.n_particles > 39.4 and Q.dr_0 < 0.111:
        z += 0.305 * (Q.n_particles - 39.4) * (0.111 - Q.dr_0)
    if Q.sj3_mass1 < 29.5 and Q.sj3_mass2 < 21.0:
        z += -0.00204 * (29.5 - Q.sj3_mass1) * (21.0 - Q.sj3_mass2)
    if Q.z_top20_slots > 0.887 and Q.dr_2 < 0.0414:
        z += -175.0 * (Q.z_top20_slots - 0.887) * (0.0414 - Q.dr_2)
    if Q.z_top30_slots > 0.939 and Q.C2 < 0.075:
        z += 200.0 * (Q.z_top30_slots - 0.939) * (0.075 - Q.C2)
    if Q.z_top30_slots > 0.942 and Q.max_pair_mass > 9.45:
        z += 0.565 * (Q.z_top30_slots - 0.942) * (Q.max_pair_mass - 9.45)
    if Q.z_top30_slots > 0.914 and Q.ptdr0_3 > 6.21:
        z += 0.582 * (Q.z_top30_slots - 0.914) * (Q.ptdr0_3 - 6.21)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.00944
    if Q.log_sum_pt > 6.91 and Q.girth2_top15 < 0.0272:
        z += 206.0 * (Q.log_sum_pt - 6.91) * (0.0272 - Q.girth2_top15)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.0649
    if Q.girth2_top40 < 0.00669:
        z += -79.4 * Q.girth2_top40 + 0.531186
    if Q.lam2 < 0.000625:
        z += -1150.0 * Q.lam2 + 0.71875
    if Q.mass_top50 < 69.2:
        z += 0.0241 * Q.mass_top50 - 1.66772
    if Q.n_particles < 46.1:
        z += -0.0306 * Q.n_particles + 1.41066
    if Q.tau21 < 0.375:
        z += 2.0 * Q.tau21 - 0.75
    if Q.n_dr_0p1_0p2 < 11.2 and Q.psi_0p2 > 0.934:
        z += 0.592 * (11.2 - Q.n_dr_0p1_0p2) * (Q.psi_0p2 - 0.934)
    if Q.n_dr_0p2_0p4 < 7.15 and Q.z_top50_slots > 0.98:
        z += 3.7 * (7.15 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.98)
    if Q.tau21 < 0.406 and Q.lam1 > 0.00572:
        z += -214.0 * (0.406 - Q.tau21) * (Q.lam1 - 0.00572)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.11
    if Q.C2 >= 0.0725:
        z += -2.55 * Q.C2 + 0.184875
    if Q.e2 >= 0.0568:
        z += -23.2 * Q.e2 + 1.31776
    if Q.e2_sq >= 0.0103:
        z += 173.0 * Q.e2_sq - 1.7819
    if Q.girth >= 0.121:
        z += -15.4 * Q.girth + 1.8634
    if Q.girth2_top15 < 0.00531:
        z += 93.2 * Q.girth2_top15 - 0.494892
    if 0.00752 <= Q.girth2_top15 < 0.016:
        z += -71.7 * Q.girth2_top15 + 0.539184
    if Q.girth2_top15 >= 0.016:
        z += 6.8 * Q.girth2_top15 - 0.716816
    if Q.lam1 >= 0.00794:
        z += -70.8 * Q.lam1 + 0.562152
    if Q.mass < 78.4:
        z += 0.105 * Q.mass - 7.7373
    if 78.4 <= Q.mass < 86.2:
        z += 0.0539 * Q.mass - 3.73106
    if 86.2 <= Q.mass < 119.0:
        z += -0.0279 * Q.mass + 3.3201
    if Q.mass_top40 < 67.5:
        z += -0.0475 * Q.mass_top40 + 1.8765
    if 67.5 <= Q.mass_top40 < 166.0:
        z += 0.0135 * Q.mass_top40 - 2.241
    if Q.n_dr_0p2_0p4 < 24.2:
        z += -0.0309 * Q.n_dr_0p2_0p4 + 0.74778
    if Q.n_particles >= 23.1:
        z += -0.0255 * Q.n_particles + 0.58905
    if Q.psi_0p3 >= 0.997:
        z += 182.0 * Q.psi_0p3 - 181.454
    if Q.mass < 101.0 and Q.D2_b2 < 3.79:
        z += 0.00874 * (101.0 - Q.mass) * (3.79 - Q.D2_b2)
    if Q.mass_top40 < 88.5 and Q.D2_b2 < 3.21:
        z += -0.0186 * (88.5 - Q.mass_top40) * (3.21 - Q.D2_b2)
    if Q.n_dr_0p2_0p4 < 22.7 and Q.n_dr_0p1_0p2 > 12.1:
        z += -0.00112 * (22.7 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 12.1)
    if Q.n_particles > 27.6 and Q.n_dr_0_0p05 > 10.7:
        z += 0.000656 * (Q.n_particles - 27.6) * (Q.n_dr_0_0p05 - 10.7)
    if Q.n_particles > 20.4 and Q.soft1_pt < 2.44:
        z += 0.00517 * (Q.n_particles - 20.4) * (2.44 - Q.soft1_pt)
    if Q.psi_0p3 > 0.997 and Q.n_dr_0_0p05 < 13.2:
        z += -11.5 * (Q.psi_0p3 - 0.997) * (13.2 - Q.n_dr_0_0p05)
    if Q.sj2_dr > 0.232 and Q.D2_b2 < 20.6:
        z += -0.194 * (Q.sj2_dr - 0.232) * (20.6 - Q.D2_b2)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.375
    if Q.girth2_top50 >= 0.0191:
        z += -108.0 * Q.girth2_top50 + 2.0628
    if 6.91 <= Q.log_sum_pt < 6.99:
        z += -23.0 * Q.log_sum_pt + 158.93
    if Q.log_sum_pt >= 6.99:
        z += -18.12 * Q.log_sum_pt + 124.8188
    if 73.6 <= Q.mass < 92.4:
        z += 0.0337 * Q.mass - 2.48032
    if 92.4 <= Q.mass < 171.0:
        z += 0.0117 * Q.mass - 0.44752
    if Q.mass >= 171.0:
        z += -0.1653 * Q.mass + 29.81948
    if Q.mass_over_sum_pt >= 0.0781:
        z += -23.0 * Q.mass_over_sum_pt + 1.7963
    if Q.mass_top50 >= 158.0:
        z += -0.277 * Q.mass_top50 + 43.766
    if Q.max_dr < 0.459:
        z += -2.32 * Q.max_dr + 1.06488
    if Q.n_dr_0p1_0p2 < 22.6:
        z += 0.0129 * Q.n_dr_0p1_0p2 - 0.29154
    if Q.n_dr_0p2_0p4 < 12.4:
        z += -0.0302 * Q.n_dr_0p2_0p4 + 0.37448
    if Q.n_particles < 47.5:
        z += -0.0303 * Q.n_particles + 1.43925
    if 72.1 <= Q.sd_mass < 86.4:
        z += 0.0286 * Q.sd_mass - 2.06206
    if Q.sd_mass >= 86.4:
        z += -0.0059 * Q.sd_mass + 0.91874
    if 908.0 <= Q.sum_pt < 990.0:
        z += 0.0107 * Q.sum_pt - 9.7156
    if Q.sum_pt >= 990.0:
        z += -0.005 * Q.sum_pt + 5.8274
    if Q.sum_pt_top50 >= 943.0:
        z += 0.0172 * Q.sum_pt_top50 - 16.2196
    if Q.tau21 < 0.561:
        z += 0.695 * Q.tau21 - 0.389895
    if Q.n_particles < 56.1 and Q.D2 < 2.7:
        z += -0.0157 * (56.1 - Q.n_particles) * (2.7 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.882
    if 0.0294 <= Q.e2 < 0.053:
        z += -31.7 * Q.e2 + 0.93198
    if Q.e2 >= 0.053:
        z += 37.4 * Q.e2 - 2.73032
    if Q.girth2_top20 >= 0.00733:
        z += 198.0 * Q.girth2_top20 - 1.45134
    if Q.girth2_top30 >= 0.0275:
        z += -223.0 * Q.girth2_top30 + 6.1325
    if Q.mass < 93.9:
        z += 0.0051 * Q.mass - 1.32939
    if 93.9 <= Q.mass < 102.0:
        z += 0.105 * Q.mass - 10.71
    if Q.mass_over_sum_pt >= 0.121:
        z += -61.0 * Q.mass_over_sum_pt + 7.381
    if Q.mass_top50 < 73.3:
        z += -0.0307 * Q.mass_top50 + 2.25031
    if Q.sj3_pair_mass_min >= 77.9:
        z += -0.119 * Q.sj3_pair_mass_min + 9.2701
    if Q.tau1 < 0.0533:
        z += -15.2 * Q.tau1 + 0.81016
    if Q.z_top40_slots < 0.931:
        z += -11.9 * Q.z_top40_slots + 11.0789
    if Q.girth2_top20 > 0.0035 and Q.C2_b2 > 0.00379:
        z += -2330.0 * (Q.girth2_top20 - 0.0035) * (Q.C2_b2 - 0.00379)
    if Q.mass < 102.0 and Q.sum_pt < 1010.0:
        z += 8.43e-05 * (102.0 - Q.mass) * (1010.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_7(Q):
    z = 0.0182
    if Q.mass < 121.0:
        z += -0.0156 * Q.mass + 1.8876
    if Q.n_dr_0p2_0p4 < 8.15:
        z += -0.0606 * Q.n_dr_0p2_0p4 + 0.49389
    if Q.psi_0p3 >= 0.998:
        z += -476.0 * Q.psi_0p3 + 475.048
    if Q.sd_mass < 69.8:
        z += -0.007 * Q.sd_mass + 0.14408
    if 69.8 <= Q.sd_mass < 87.2:
        z += 0.0198 * Q.sd_mass - 1.72656
    if Q.sd_mass >= 97.2:
        z += -0.0854 * Q.sd_mass + 8.30088
    if Q.tau1 < 0.0957:
        z += 9.7 * Q.tau1 - 1.31023
    if 0.0957 <= Q.tau1 < 0.107:
        z += 33.8 * Q.tau1 - 3.6166
    if Q.tau21_b2 < 0.234:
        z += -6.45 * Q.tau21_b2 + 1.5093
    if Q.lam1 < 0.00796 and Q.zdr_0 > 0.0152:
        z += -30200.0 * (0.00796 - Q.lam1) * (Q.zdr_0 - 0.0152)
    if Q.mass < 82.3 and Q.psi_0p3 > 0.978:
        z += 7.71 * (82.3 - Q.mass) * (Q.psi_0p3 - 0.978)
    if Q.mass < 91.0 and Q.psi_0p3 > 0.978:
        z += -17.3 * (91.0 - Q.mass) * (Q.psi_0p3 - 0.978)
    if Q.mass < 91.2 and Q.psi_0p3 > 0.964:
        z += 9.53 * (91.2 - Q.mass) * (Q.psi_0p3 - 0.964)
    if Q.mass < 91.5 and Q.psi_0p3 > 0.998:
        z += -64.8 * (91.5 - Q.mass) * (Q.psi_0p3 - 0.998)
    if Q.mass < 93.2 and Q.psi_0p3 > 0.964:
        z += -8.46 * (93.2 - Q.mass) * (Q.psi_0p3 - 0.964)
    if Q.mass < 101.0 and Q.psi_0p3 > 0.998:
        z += 62.7 * (101.0 - Q.mass) * (Q.psi_0p3 - 0.998)
    if Q.mass < 102.0 and Q.psi_0p3 > 0.978:
        z += 7.44 * (102.0 - Q.mass) * (Q.psi_0p3 - 0.978)
    if Q.mass < 93.2 and Q.z_dr_0_0p05 < 0.503:
        z += -0.0988 * (93.2 - Q.mass) * (0.503 - Q.z_dr_0_0p05)
    if Q.psi_0p3 > 0.997 and Q.z_top50_slots > 0.979:
        z += 6910.0 * (Q.psi_0p3 - 0.997) * (Q.z_top50_slots - 0.979)
    if Q.tau21_b2 < 0.23 and Q.mass_over_sum_pt_sq < 0.00808:
        z += -1320.0 * (0.23 - Q.tau21_b2) * (0.00808 - Q.mass_over_sum_pt_sq)
    if Q.tau21_b2 < 0.231 and Q.sum_pt < 1270.0:
        z += -0.0184 * (0.231 - Q.tau21_b2) * (1270.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_8(Q):
    z = 1.3
    if 0.0059 <= Q.girth2_top20 < 0.00788:
        z += -202.0 * Q.girth2_top20 + 1.1918
    if Q.girth2_top20 >= 0.00788:
        z += 66.0 * Q.girth2_top20 - 0.92004
    if Q.girth2_top30 < 0.00754:
        z += -165.0 * Q.girth2_top30 + 1.2441
    if 71.1 <= Q.mass < 103.0:
        z += 0.0435 * Q.mass - 3.09285
    if 103.0 <= Q.mass < 127.0:
        z += 0.0027 * Q.mass + 1.10955
    if Q.mass >= 127.0:
        z += -0.0182 * Q.mass + 3.76385
    if Q.n_dr_0p2_0p4 < 8.6:
        z += 0.0207 * Q.n_dr_0p2_0p4 - 0.80763
    if 8.6 <= Q.n_dr_0p2_0p4 < 17.9:
        z += 0.0677 * Q.n_dr_0p2_0p4 - 1.21183
    if Q.sj2_dr >= 0.23:
        z += 4.23 * Q.sj2_dr - 0.9729
    if Q.sum_pt < 1030.0:
        z += -0.0195 * Q.sum_pt + 20.085
    if Q.sum_pt_top30 >= 987.0:
        z += 0.0014 * Q.sum_pt_top30 - 1.3818
    if Q.sum_pt_top40 < 1030.0:
        z += 0.00795 * Q.sum_pt_top40 - 8.1885
    if Q.width < 0.00987:
        z += 214.0 * Q.width - 2.11218
    if Q.z_top50_slots >= 0.972:
        z += -35.8 * Q.z_top50_slots + 34.7976
    if Q.girth2_top40 > 0.00555 and Q.log_sum_pt < 6.99:
        z += -810.0 * (Q.girth2_top40 - 0.00555) * (6.99 - Q.log_sum_pt)
    if Q.mass_over_sum_pt > 0.0785 and Q.sum_pt < 1100.0:
        z += 0.305 * (Q.mass_over_sum_pt - 0.0785) * (1100.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 18.0 and Q.z_dr_0p1_0p2 > 0.313:
        z += 0.207 * (18.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.313)
    if Q.tau2 < 0.0848 and Q.sj3_mass1 > 13.4:
        z += 0.403 * (0.0848 - Q.tau2) * (Q.sj3_mass1 - 13.4)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.13
    if Q.girth >= 0.0976:
        z += 23.3 * Q.girth - 2.27408
    if Q.girth2_top40 < 0.00617:
        z += -281.0 * Q.girth2_top40 + 1.73377
    if Q.mass < 62.9:
        z += -0.02 * Q.mass + 4.0146
    if 62.9 <= Q.mass < 85.5:
        z += -0.09 * Q.mass + 8.4176
    if 85.5 <= Q.mass < 147.0:
        z += 0.013 * Q.mass - 0.3889
    if 147.0 <= Q.mass < 178.0:
        z += -0.0491 * Q.mass + 8.7398
    if Q.sum_pt < 989.0:
        z += -0.00819 * Q.sum_pt + 8.09991
    if Q.z_top40_slots >= 0.954:
        z += -8.68 * Q.z_top40_slots + 8.28072
    if Q.mass_top40 < 116.0 and Q.n_dr_0p2_0p4 > 8.57:
        z += 0.00132 * (116.0 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 8.57)
    if Q.mass_top40 < 97.9 and Q.sum_pt < 1020.0:
        z += -0.000231 * (97.9 - Q.mass_top40) * (1020.0 - Q.sum_pt)
    if Q.mass_top40 < 100.0 and Q.sum_pt_top40 < 961.0:
        z += 0.000185 * (100.0 - Q.mass_top40) * (961.0 - Q.sum_pt_top40)
    return max(0.0, z)


def neuron_10(Q):
    z = 0.309
    if Q.D2 < 3.04:
        z += -0.473 * Q.D2 + 1.43792
    if Q.dr_0 < 0.0664:
        z += -7.76 * Q.dr_0 + 0.515264
    if Q.e2 >= 0.0128:
        z += 29.9 * Q.e2 - 0.38272
    if Q.girth2_top20 < 0.00533:
        z += 236.0 * Q.girth2_top20 - 1.25788
    if Q.mass < 78.8:
        z += -0.0672 * Q.mass + 5.52546
    if 78.8 <= Q.mass < 87.9:
        z += -0.0531 * Q.mass + 4.41438
    if 87.9 <= Q.mass < 99.6:
        z += 0.0467 * Q.mass - 4.35804
    if 99.6 <= Q.mass < 145.0:
        z += 0.0141 * Q.mass - 1.11108
    if 145.0 <= Q.mass < 164.0:
        z += -0.0728 * Q.mass + 11.48942
    if Q.mass >= 164.0:
        z += -0.1209 * Q.mass + 19.37782
    if Q.mass_over_sum_pt >= 0.16:
        z += -31.1 * Q.mass_over_sum_pt + 4.976
    if Q.mass_top5 >= 17.5:
        z += 0.0112 * Q.mass_top5 - 0.196
    if Q.mass_top50 >= 138.0:
        z += 0.0434 * Q.mass_top50 - 5.9892
    if Q.n_dr_0p2_0p4 < 11.1:
        z += 0.055 * Q.n_dr_0p2_0p4 - 0.6105
    if Q.pt_dispersion >= 0.27:
        z += -2.54 * Q.pt_dispersion + 0.6858
    if Q.sj2_mass1 < 65.0:
        z += 0.0219 * Q.sj2_mass1 - 1.4235
    if Q.sum_pt < 984.0:
        z += 0.0107 * Q.sum_pt - 10.5288
    if Q.sum_pt_top10 >= 941.0:
        z += -0.0034 * Q.sum_pt_top10 + 3.1994
    if Q.sum_pt_top15 < 941.0:
        z += -0.00394 * Q.sum_pt_top15 + 3.70754
    if Q.tau1 < 0.047:
        z += 82.7 * Q.tau1 - 3.8869
    if Q.tau2 < 0.0485:
        z += -14.2 * Q.tau2 + 0.6887
    if Q.z_dr_0p2_0p4 < 0.0859:
        z += -4.97 * Q.z_dr_0p2_0p4 + 0.426923
    if Q.D2 < 2.99 and Q.sj2_dr > 0.192:
        z += -3.21 * (2.99 - Q.D2) * (Q.sj2_dr - 0.192)
    if Q.girth2_top10 < 0.0234 and Q.psi_0p3 > 0.998:
        z += -17300.0 * (0.0234 - Q.girth2_top10) * (Q.psi_0p3 - 0.998)
    if Q.mass_top50 > 133.0 and Q.soft5_z > 0.00163:
        z += -7.19 * (Q.mass_top50 - 133.0) * (Q.soft5_z - 0.00163)
    if Q.sj2_mass1 < 63.3 and Q.sj2_mass2 > 7.52:
        z += 0.000322 * (63.3 - Q.sj2_mass1) * (Q.sj2_mass2 - 7.52)
    if Q.sum_pt_top10 > 943.0 and Q.eta_0 < -0.078:
        z += -0.877 * (Q.sum_pt_top10 - 943.0) * (-0.078 - Q.eta_0)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.575
    if Q.e2 < 0.0266:
        z += -96.2 * Q.e2 + 2.55892
    if Q.girth2_top30 < 0.00666:
        z += 402.0 * Q.girth2_top30 - 2.67732
    if Q.mass_top50 < 80.9:
        z += -0.0079 * Q.mass_top50 + 1.83419
    if 80.9 <= Q.mass_top50 < 94.7:
        z += -0.0866 * Q.mass_top50 + 8.20102
    if Q.n_dr_0p1_0p2 < 16.7:
        z += -0.0549 * Q.n_dr_0p1_0p2 + 0.91683
    if Q.n_dr_0p2_0p4 < 7.5:
        z += -0.184 * Q.n_dr_0p2_0p4 + 1.38
    if Q.psi_0p3 >= 0.99:
        z += 64.2 * Q.psi_0p3 - 63.558
    if Q.z_top5_slots >= 0.552:
        z += -1.56 * Q.z_top5_slots + 0.86112
    if Q.mass < 119.0 and Q.sum_pt_top10 < 872.0:
        z += -2.65e-05 * (119.0 - Q.mass) * (872.0 - Q.sum_pt_top10)
    if Q.n_dr_0p2_0p4 < 9.67 and Q.girth2 < 0.0062:
        z += -28.5 * (9.67 - Q.n_dr_0p2_0p4) * (0.0062 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 8.65 and Q.n_particles > 36.1:
        z += -0.0051 * (8.65 - Q.n_dr_0p2_0p4) * (Q.n_particles - 36.1)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.151
    if Q.girth2_top15 < 0.00586:
        z += 172.0 * Q.girth2_top15 - 1.00792
    if Q.mass < 59.7:
        z += -0.0234 * Q.mass + 4.21728
    if 59.7 <= Q.mass < 83.4:
        z += -0.119 * Q.mass + 9.9246
    if Q.sd_mass >= 124.0:
        z += 0.0144 * Q.sd_mass - 1.7856
    if Q.log_sum_pt < 6.85 and Q.lam2 < 0.0036:
        z += -3400.0 * (6.85 - Q.log_sum_pt) * (0.0036 - Q.lam2)
    if Q.mass < 90.0 and Q.psi_0p3 < 0.999:
        z += -3.47 * (90.0 - Q.mass) * (0.999 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 123.0 and Q.sj3_pair_mass_min < 65.0:
        z += 0.000437 * (Q.sj3_pair_mass_max - 123.0) * (65.0 - Q.sj3_pair_mass_min)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.97
    if Q.log_sum_pt < 6.92:
        z += 24.1 * Q.log_sum_pt - 166.772
    if 63.1 <= Q.mass < 101.0:
        z += 0.015 * Q.mass - 0.9465
    if 101.0 <= Q.mass < 161.0:
        z += 0.0002 * Q.mass + 0.5483
    if Q.mass >= 161.0:
        z += -0.0996 * Q.mass + 16.6161
    if Q.mass_top5 >= 40.1:
        z += 0.0121 * Q.mass_top5 - 0.48521
    if Q.sum_pt_top40 < 1020.0:
        z += -0.00809 * Q.sum_pt_top40 + 8.2518
    if Q.log_sum_pt > 7.14 and Q.sd_zg > 0.446:
        z += 1140.0 * (Q.log_sum_pt - 7.14) * (Q.sd_zg - 0.446)
    if Q.mass > 142.0 and Q.D2 > 0.0302:
        z += -0.0225 * (Q.mass - 142.0) * (Q.D2 - 0.0302)
    if Q.mass > 169.0 and Q.D2 > 0.281:
        z += 0.0811 * (Q.mass - 169.0) * (Q.D2 - 0.281)
    if Q.mass > 173.0 and Q.soft5_z > 0.000296:
        z += -329.0 * (Q.mass - 173.0) * (Q.soft5_z - 0.000296)
    if Q.mass_top50 > 134.0 and Q.soft6_z > 0.00094:
        z += -27.5 * (Q.mass_top50 - 134.0) * (Q.soft6_z - 0.00094)
    if Q.mass_top50 > 90.5 and Q.soft7_z < 0.00189:
        z += -7.59 * (Q.mass_top50 - 90.5) * (0.00189 - Q.soft7_z)
    if Q.sum_pt < 1090.0 and Q.tau21_b2 < 1.0:
        z += -0.0078 * (1090.0 - Q.sum_pt) * (1.0 - Q.tau21_b2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.384
    if Q.e2 < 0.0302:
        z += -59.7 * Q.e2 + 1.80294
    if Q.girth2_top3 < 0.00154:
        z += 359.0 * Q.girth2_top3 - 0.55286
    if Q.girth2_top30 < 0.00262:
        z += 3370.0 * Q.girth2_top30 - 8.8294
    if Q.girth2_top5 < 0.0067:
        z += -78.0 * Q.girth2_top5 + 0.5226
    if Q.log_sum_pt < 6.94:
        z += -5.69 * Q.log_sum_pt + 39.4886
    if Q.mass < 82.8:
        z += 0.247 * Q.mass - 21.4636
    if 82.8 <= Q.mass < 92.0:
        z += 0.11 * Q.mass - 10.12
    if Q.mass_over_sum_pt < 0.144:
        z += -17.6 * Q.mass_over_sum_pt + 2.5344
    if Q.mass_top50 < 78.4:
        z += -0.0691 * Q.mass_top50 + 5.41744
    if Q.psi_0p3 >= 0.993:
        z += 152.0 * Q.psi_0p3 - 150.936
    if Q.N2 < 0.484 and Q.max_dr > 0.24:
        z += -7.57 * (0.484 - Q.N2) * (Q.max_dr - 0.24)
    if Q.girth2_top5 < 0.00654 and Q.n_pt_above_10 > 13.0:
        z += -7.52 * (0.00654 - Q.girth2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.mass_over_sum_pt < 0.0905 and Q.psi_0p2 > 0.914:
        z += -405.0 * (0.0905 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.914)
    if Q.psi_0p2 > 0.978 and Q.pt2_over_pt0 > 0.121:
        z += 30.8 * (Q.psi_0p2 - 0.978) * (Q.pt2_over_pt0 - 0.121)
    if Q.psi_0p3 > 0.993 and Q.sd_rg < 0.188:
        z += -420.0 * (Q.psi_0p3 - 0.993) * (0.188 - Q.sd_rg)
    if Q.psi_0p3 > 0.993 and Q.z_2nd < 0.124:
        z += -939.0 * (Q.psi_0p3 - 0.993) * (0.124 - Q.z_2nd)
    if Q.tau21_b2 < 0.322 and Q.n_real_top40 < 30.6:
        z += -0.22 * (0.322 - Q.tau21_b2) * (30.6 - Q.n_real_top40)
    if Q.tau21_b2 < 0.432 and Q.orientation_deg > -12.9:
        z += -0.0104 * (0.432 - Q.tau21_b2) * (Q.orientation_deg - -12.9)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.322
    if Q.e2 >= 0.0294:
        z += -50.4 * Q.e2 + 1.48176
    if Q.lam2 < 0.00181:
        z += -305.0 * Q.lam2 + 0.55205
    if Q.log_sum_pt < 7.03:
        z += -7.67 * Q.log_sum_pt + 53.9201
    if Q.max_dr < 0.294:
        z += 4.66 * Q.max_dr - 1.37004
    if Q.n_dr_0p1_0p2 < 13.6:
        z += -0.0487 * Q.n_dr_0p1_0p2 + 0.66232
    if Q.n_dr_0p2_0p4 >= 8.79:
        z += -0.047 * Q.n_dr_0p2_0p4 + 0.41313
    if Q.psi_0p2 >= 0.857:
        z += -4.81 * Q.psi_0p2 + 4.12217
    if Q.psi_0p3 >= 0.989:
        z += -47.9 * Q.psi_0p3 + 47.3731
    if Q.sd_mass < 44.2:
        z += 0.0069 * Q.sd_mass + 0.5535
    if 44.2 <= Q.sd_mass < 73.6:
        z += -0.0292 * Q.sd_mass + 2.14912
    if Q.sd_rg >= 0.203:
        z += 5.7 * Q.sd_rg - 1.1571
    if Q.sum_pt_top40 < 1020.0:
        z += 0.0123 * Q.sum_pt_top40 - 12.546
    if Q.tau1 < 0.0678:
        z += 15.5 * Q.tau1 - 1.0509
    if Q.sj2_dr > 0.225 and Q.C2_b2 < 0.0493:
        z += 147.0 * (Q.sj2_dr - 0.225) * (0.0493 - Q.C2_b2)
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
