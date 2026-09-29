"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.3% (the network: 81.1%); same class as the network for 93.9% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.C2_b2                  energy correlation ratio e3/e2² with β = 2
  Q.C3                     energy correlation ratio e4·e2/e3² (small = three-prong)
  Q.D2                     energy correlation ratio e3/e2³
  Q.D2_b2                  energy correlation ratio e3/e2³ with β = 2
  Q.D3                     energy correlation ratio e4·e2³/e3³ (small = three-prong)
  Q.LHA                    Les Houches angularity
  Q.M3                     generalized ECF ratio M3
  Q.N2                     generalized ECF ratio N2 (two smallest angles; small = two-prong)
  Q.e3                     energy correlation e3 (β=1, 24 hardest)
  Q.e4                     energy correlation e4 = Σ zᵢzⱼzₖzₗ × product of the 6 ΔR (β=1, 12 hardest)
  Q.sj3_pair_mass_max      largest mass of two of the 3 subjets [GeV]
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.pair_mass_0_13         mass of particles 0 and 13 [GeV]
  Q.sj3_mass1              mass of subjet 1 of 3 [GeV]
  Q.sj3_mass2              mass of subjet 2 of 3 [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
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
  Q.ptdr0_3                pT3 · ΔR(0, 3) [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.soft1_pt               pT [GeV] of the 1. softest real particle (0 if it is among the 15 hardest)
  Q.soft3_pt               pT [GeV] of the 3. softest real particle (0 if it is among the 15 hardest)
  Q.soft4_pt               pT [GeV] of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.z_10                   pT of particle 10 / total pT
  Q.z_11                   pT of particle 11 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_3                    pT of particle 3 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.soft4_z                pT share of the 4. softest real particle (0 if it is among the 15 hardest)
  Q.soft5_z                pT share of the 5. softest real particle (0 if it is among the 15 hardest)
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.sj2_zsoft              pT share of the softer of the 2 N-subjettiness subjets
  Q.zdr_0                  pT share × ΔR of particle 0 (its part of the girth)
  Q.zdr_1                  pT share × ΔR of particle 1 (its part of the girth)
  Q.zdr_5                  pT share × ΔR of particle 5 (its part of the girth)
  Q.z_2nd                  2nd-largest pT share
  Q.sj3_pair_mass_min      smallest mass of two of the 3 subjets [GeV]
  Q.sj3_pairmin_over_m     smallest subjet-pair mass / jet mass (dimensionless)
  Q.sd_rg                  angle of the soft-drop splitting
  Q.sd_mass                soft-drop groomed mass, C/A on the 20 hardest, β=0, z_cut=0.1 [GeV]
  Q.abseta_4               |Δη| of particle 4
  Q.absphi_0               |Δφ| of particle 0
  Q.orientation_deg        direction of the major axis [degrees]
  Q.sj2_dr                 distance between the 2 subjet axes
  Q.dr1_12                 ΔR between particle 12 and the 2nd-hardest particle
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
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
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_1           number of particles with pT > 1 GeV
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
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
        D2_b2=ecf('e3b2') / max(ecf('e2b2') ** 3, 1e-30),
        D3=ecf('e4') * ecf('e2') ** 3 / max(ecf('e3') ** 3, 1e-30),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        M3=ecf('g41') / max(ecf('g31'), 1e-30),
        N2=ecf('g32') / max(ecf('e2') ** 2, 1e-30),
        e3=ecf('e3'),
        e4=ecf('e4'),
        sj3_pair_mass_max=max(subjets(3)["mpair"]),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        pair_mass_0_13=pair_mass(0, 13),
        sj3_mass1=subjets(3)["mass"][0],
        sj3_mass2=subjets(3)["mass"][1],
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
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
        ptdr0_3=pt[3] * math.sqrt(dist2(0, 3)) if pt[3] > 0 else 0.0,
        pt_6=pt[6],
        pt_9=pt[9],
        soft1_pt=softp(1, 'pt'),
        soft3_pt=softp(3, 'pt'),
        soft4_pt=softp(4, 'pt'),
        z_10=z[10],
        z_11=z[11],
        z_2=z[2],
        z_3=z[3],
        z_6=z[6],
        soft4_z=softp(4, 'z'),
        soft5_z=softp(5, 'z'),
        z_top15_slots=sum(pt[:15]) / tot,
        z_top2_slots=sum(pt[:2]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        sj2_zsoft=subjets(2)["z"][1],
        zdr_0=z[0] * dr[0],
        zdr_1=z[1] * dr[1],
        zdr_5=z[5] * dr[5],
        z_2nd=zs[1],
        sj3_pair_mass_min=min(subjets(3)["mpair"]),
        sj3_pairmin_over_m=min(subjets(3)["mpair"]) / max(mass_of(n), 1e-9),
        sd_rg=softdrop("rg"),
        sd_mass=softdrop("mass"),
        abseta_4=abs(eta[4]),
        absphi_0=abs(phi[0]),
        orientation_deg=math.degrees(0.5 * math.atan2(2 * tb, ta - tc)),
        sj2_dr=subjets(2)["dr"][0],
        dr1_12=math.sqrt(dist2(1, 12)) if pt[12] > 0 else 0.0,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        phi_0=phi[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
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
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_1=sum(1 for x in pt if x > 1),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
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
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 0.3323458
            + 0.1148417 * max(0.0, Q.n_particles - 38.0)
            + 15.17785 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.009297987 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 12.53513 * max(0.0, Q.log_sum_pt - 6.98945)
            - 98.94735 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 0.002148106 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 0.03049833 * max(0.0, 47.88842 - Q.mass_top20)
            - 0.008059208 * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.4506887 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.005807989 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.2905178 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0)
            + 0.003746863 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03858602 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 169.6487 * max(0.0, 0.003270031 - Q.girth2_top15)
            - 0.08100759 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 169.4878 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            + 1273.615 * max(0.0, 0.0005522528 - Q.girth2_top3)
            - 0.00148257 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 26.16099 * max(0.0, 0.03187688 - Q.M3)
            - 0.01779958 * max(0.0, 30.26161 - Q.sj2_mass1)
            - 0.04332009 * max(0.0, 31.35938 - Q.pt_9)
            + 0.1266055 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1)
            + 0.7369499 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874)
            + 27.58894 * max(0.0, Q.log_sum_pt - 6.893714)
            + 134.0114 * max(0.0, 0.004673423 - Q.lam1)
            - 0.00956025 * max(0.0, 120.6 - Q.mass)
            + 49.43327 * max(0.0, 0.02412652 - Q.girth2_top30)
            + 0.04339613 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12)
            + 2.023623 * max(0.0, 0.1416054 - Q.D3)
            - 4.403608 * max(0.0, Q.z_dr_0_0p05 - 0.878906)
            + 0.01634918 * max(0.0, 42.41192 - Q.mass_top30)
            - 7.741519 * max(0.0, 7.139296 - Q.log_sum_pt)
            + 0.01548025 * max(0.0, 12.0 - Q.n_dr_0_0p05)
            - 17.21745 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.001609659 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13)
            + 0.5230061 * max(0.0, 1.521582 - Q.soft1_pt)
            + 0.01023091 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.003192628 * max(0.0, Q.sum_pt_top30 - 1191.938)
            + 4.214125 * max(0.0, Q.z_top20_slots - 0.8965411)
            - 0.009909199 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0)
            + 0.2333272 * max(0.0, Q.pt_entropy - 2.07371)
        ))
        - 0.75 * grid(3, max(0.0, -0.09849501
            + 806.4848 * max(0.0, 0.0006154841 - Q.lam2)
            + 0.184767 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            + 0.02504442 * max(0.0, 46.0 - Q.n_particles)
            - 0.001116056 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30)
            - 0.001252897 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349)
            + 280.4078 * max(0.0, Q.psi_0p2 - 0.9985434)
            - 0.0130078 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0)
            + 4.924219e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 4.009614 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03447112 - Q.z_10)
            - 1.922232 * max(0.0, 0.3861957 - Q.tau21)
            - 0.1042589 * max(0.0, 2.410481 - Q.D2)
            + 11.90596 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 97.15128 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 18.20203 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            + 0.5548369 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1)
            - 326.0059 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.005771511 * max(0.0, 79.21004 - Q.mass_top50)
            - 0.008459062 * max(0.0, 86.4 - Q.mass)
            - 133.8843 * max(0.0, 0.009614971 - Q.girth2)
            + 144.7654 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 1.334769 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741)
            - 0.01126831 * max(0.0, 53.87362 - Q.mass)
            + 0.009280592 * max(0.0, 92.85979 - Q.mass)
            + 0.01680284 * max(0.0, 27.56535 - Q.sj2_mass1)
            - 0.01025869 * max(0.0, 80.4 - Q.mass_top40)
            + 21.90357 * max(0.0, 0.01626937 - Q.tau4)
            - 4059.072 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.08575439 - Q.abseta_4)
        ))
        - 0.875 * grid(4, max(0.0, 0.9994559
            - 0.002910387 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03907728 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0212475 * max(0.0, 120.6 - Q.mass)
            - 0.08147182 * max(0.0, 86.4 - Q.mass)
            - 0.004867626 * max(0.0, 120.6 - Q.mass_top30)
            + 176.9157 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.005725281 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02142293 * max(0.0, Q.n_particles - 22.0)
            - 177.857 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 12.45749 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05)
            + 0.0004443633 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0)
            + 0.006175368 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 60.28447 * max(0.0, Q.e2_sq - 0.01396296)
            - 78.6805 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01343086 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.06442411 * max(0.0, 101.0497 - Q.mass)
            + 0.03242072 * max(0.0, 26.0 - Q.n_dr_0p2_0p4)
            - 0.01126814 * max(0.0, 91.69753 - Q.mass_top30)
            - 0.008703325 * max(0.0, 80.78464 - Q.mass)
            - 0.01034394 * max(0.0, 74.25181 - Q.mass)
            - 0.0146105 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.001032432 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0)
            - 0.04550326 * max(0.0, 92.85979 - Q.mass)
            + 0.002005798 * max(0.0, 143.7876 - Q.mass)
            + 0.03131972 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.03630314 * max(0.0, 78.26182 - Q.mass)
            - 20.22809 * max(0.0, Q.girth - 0.1207452)
            - 123.4678 * max(0.0, Q.lam1 - 0.008241985)
            + 0.5648683 * max(0.0, 0.9909875 - Q.psi_0p1)
            + 76.19543 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 134.6129 * max(0.0, Q.e2_sq - 0.009606007)
            - 60.85614 * max(0.0, Q.lam2 - 0.001776308)
            - 0.00529117 * max(0.0, 91.19 - Q.mass_top15)
        ))
        - 0.3125 * grid(5, max(0.0, 0.6833986
            + 0.03663246 * max(0.0, 64.0 - Q.n_particles)
            - 29.0871 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 5.20108 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.005175716 * max(0.0, Q.sum_pt - 907.9372)
            - 11.67306 * max(0.0, Q.log_sum_pt - 6.920349)
            + 195.4541 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 59.68106 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01729337 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.01866498 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01145388 * max(0.0, Q.n_pt_above_1 - 28.0)
            + 0.02827257 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 25726.1 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 17.63508 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.004013931 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 6.094535 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.001172525 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.224028 * max(0.0, Q.max_dr - 0.2404747)
            - 7.40116 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.01465448 * max(0.0, Q.mass_top40 - 120.6)
            + 0.02871264 * max(0.0, Q.mass - 172.8)
            + 28.39648 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005553738 * max(0.0, 150.0144 - Q.mass_top40)
            - 510.6845 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 1.454749 * max(0.0, 0.5494307 - Q.tau21)
            + 0.03687624 * max(0.0, Q.sd_mass - 69.65633)
            - 0.2087503 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.01065712 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002118901 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 161.3584 * max(0.0, 0.01375115 - Q.z_11)
            - 0.01378824 * max(0.0, Q.mass - 91.19)
            + 0.01994 * max(0.0, Q.mass - 74.25181)
            + 10.15916 * max(0.0, 0.03875945 - Q.e2)
            - 0.02020562 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.06683936 * max(0.0, 8.0 - Q.n_dr_0p1_0p2)
            - 0.04854561 * max(0.0, Q.sd_mass - 86.4)
            + 0.007455275 * max(0.0, Q.mass_top10 - 71.781)
            - 2.502079 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4)
            - 0.1477321 * max(0.0, 14.14062 - Q.pt_11)
            + 1.869872 * max(0.0, Q.max_dr - 0.4357228)
            + 0.3503087 * max(0.0, 1.976207 - Q.D2)
        ))
        + 0.15625 * grid(6, max(0.0, 1.170115
            + 0.01447067 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03719278 * max(0.0, 120.6 - Q.mass)
            + 0.0339753 * max(0.0, 86.4 - Q.mass)
            + 9.2013e-05 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 114.2108 * max(0.0, 0.002197765 - Q.girth2_top15)
            - 0.006836022 * max(0.0, Q.sj3_pair_mass_min - 29.00832)
            + 1.980592 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 78.82779 * max(0.0, Q.e2 - 0.05557149)
            - 0.07532423 * max(0.0, 101.0497 - Q.mass)
            + 0.07202664 * max(0.0, 92.85979 - Q.mass)
            - 0.02471428 * max(0.0, Q.sj3_mass1 - 21.11128)
            - 4.312245 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 21.73339 * max(0.0, 0.06310829 - Q.tau1)
            + 12.94304 * max(0.0, 0.9300465 - Q.z_top40_slots)
            - 0.1218576 * max(0.0, Q.sj3_pair_mass_min - 76.60223)
            - 6.175682 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
            + 157.402 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 1161.606 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 1133.33 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            - 0.003083477 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 33.3761)
            - 255.1533 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 116.7522 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 274.7502 * max(0.0, 0.003811746 - Q.lam1)
            - 44.33966 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 6.021715 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 318.9114 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 146.3964 * max(0.0, 0.003687605 - Q.lam2)
            + 0.009229968 * max(0.0, 91.19 - Q.mass_top15)
            + 0.005383053 * max(0.0, 172.8 - Q.mass_top50)
            - 2864.26 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 0.01535736 * max(0.0, Q.n_dr_0_0p05 - 9.0)
            - 34.5741 * max(0.0, Q.e2 - 0.02793599)
            + 188.0375 * max(0.0, 0.00375223 - Q.girth2_top30)
            - 46.89314 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 31.79539 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        + 0.03125 * grid(8, max(0.0, 2.066925
            - 18.60172 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.004676679 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 50.27689 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03026977 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0343019 * max(0.0, Q.mass - 125.1)
            + 0.04023553 * max(0.0, Q.mass - 87.36377)
            + 0.2598764 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.01794018 * max(0.0, 1017.435 - Q.sum_pt)
            - 594.6978 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06530352 * max(0.0, Q.mass - 101.0497)
            + 238.3678 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 14.98268 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 145.493 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.02477471 * max(0.0, Q.mass - 64.48544)
            + 151.623 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 432.9856 * max(0.0, 0.009614971 - Q.width)
            + 25.12502 * max(0.0, 0.007877041 - Q.girth2)
            + 5.86249 * max(0.0, Q.sj2_dr - 0.2232169)
            + 1.347847 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477)
            + 0.0408415 * max(0.0, 9.0 - Q.n_dr_0p2_0p4)
            - 0.0006934448 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct)
            - 0.06565953 * max(0.0, Q.n_for_90pct - 7.0)
            - 0.1541735 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296)
            + 49.8174 * max(0.0, Q.e2 - 0.04755309)
            + 100.3115 * max(0.0, 0.007671243 - Q.lam1)
            + 7.418645 * max(0.0, Q.C2 - 0.06655881)
            - 2876.53 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.05103926 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.03785952 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.02799519 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.0423991 * max(0.0, Q.mass - 74.25181)
            - 0.01009848 * max(0.0, Q.mass_top20 - 119.2969)
            - 694.5983 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157)
            + 0.02069189 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            + 0.4102108 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202)
            + 0.03298981 * max(0.0, Q.n_dr_0p2_0p4 - 3.0)
            + 0.01687216 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.009837468 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 1.308164 * max(0.0, 0.8316924 - Q.z_top15_slots)
            - 1.925304 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.2181557 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.1935704 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 193.494 * max(0.0, 0.008840538 - Q.girth2_top40)
            - 19.65455 * max(0.0, Q.z_top50_slots - 0.9704436)
            + 1395.118 * max(0.0, Q.e3 - 0.0003372339)
            - 12.77399 * max(0.0, 0.02146578 - Q.girth2_top15)
            - 23.13066 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.515625 * grid(9, max(0.0, -0.9585951
            + 0.02969574 * max(0.0, 80.89043 - Q.mass_top40)
            + 0.003029648 * max(0.0, 956.2133 - Q.sum_pt_top40)
            + 196.2683 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 0.002660812 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002300043 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            - 281.419 * max(0.0, 0.00217213 - Q.girth2_top40)
            + 0.07978109 * max(0.0, 172.8 - Q.mass)
            - 0.1562173 * max(0.0, 160.8 - Q.mass)
            + 0.06615134 * max(0.0, 92.85979 - Q.mass)
            + 21.56199 * max(0.0, Q.girth - 0.09749958)
            - 0.0967386 * max(0.0, 143.7876 - Q.mass)
            + 0.0002574458 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40)
            - 0.006547837 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047)
            + 0.05483302 * max(0.0, 82.85409 - Q.mass)
            - 0.06673127 * max(0.0, 64.48544 - Q.mass)
            - 0.02427501 * max(0.0, 163.2541 - Q.mass_top40)
            + 54.99904 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 0.0009842942 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            + 58.80979 * max(0.0, Q.lam1 - 0.01174405)
            - 23.01644 * max(0.0, Q.girth2_top30 - 0.0008564881)
            + 45.42692 * max(0.0, 0.004673423 - Q.lam1)
            - 90.31644 * max(0.0, Q.girth - 0.1402186)
            + 0.1295996 * max(0.0, 162.8363 - Q.mass)
            + 0.05061472 * max(0.0, 138.8977 - Q.mass_top50)
            + 0.01076456 * max(0.0, 73.33139 - Q.mass_top30)
            + 46.68351 * max(0.0, Q.LHA - 0.404204)
            - 0.04123687 * max(0.0, 92.16545 - Q.mass_top50)
            - 7.95405 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.02268123 * max(0.0, 986.0565 - Q.sum_pt)
            - 14.45285 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
        - 0.015625 * grid(10, max(0.0, -0.0267736
            - 5.715259 * max(0.0, 0.1207452 - Q.girth)
            - 0.0516948 * max(0.0, Q.mass - 162.8363)
            - 248.0863 * max(0.0, 0.005402331 - Q.girth2_top30)
            + 0.02087975 * max(0.0, Q.mass_top50 - 160.8)
            + 0.4566084 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, 631.275 - Q.sum_pt_top5)
            + 0.01892133 * max(0.0, 86.4 - Q.mass)
            - 0.007794864 * max(0.0, 65.20727 - Q.sj2_mass1)
            + 0.3495594 * max(0.0, 2.975532 - Q.D2)
            + 1.800257 * max(0.0, 0.5760704 - Q.z_top2_slots)
            - 0.005213073 * max(0.0, 59.40777 - Q.mass_top5)
            - 17.68862 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109)
            - 0.009389672 * max(0.0, 959.0957 - Q.sum_pt_top50)
            - 576.1586 * max(0.0, 0.002575211 - Q.girth2)
            - 0.0383639 * max(0.0, 101.0497 - Q.mass)
            - 3.062242 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855)
            + 7.297847 * max(0.0, 0.06413297 - Q.dr_0)
            - 3.460823 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 0.03371252 * max(0.0, Q.mass_top50 - 172.8)
            - 9.858262 * max(0.0, Q.e2 - 0.03263075)
            + 184716.1 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 22809.65 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 0.002926603 * max(0.0, Q.sum_pt_top10 - 943.6922)
            - 111.5653 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.4357228 - Q.max_dr)
            + 0.006737216 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1)
            + 39.3616 * max(0.0, Q.e2 - 0.01256572)
            - 52.75721 * max(0.0, Q.mass_over_sum_pt - 0.1606361)
            + 2352.6 * max(0.0, 0.0001086251 - Q.e3)
            + 22.66657 * max(0.0, 0.008824206 - Q.zdr_1)
            - 0.05728831 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 530.6533 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, 0.09696199 - Q.z_3)
            - 0.1022449 * max(0.0, Q.mass - 143.7876)
            + 0.0574559 * max(0.0, Q.mass_top50 - 136.785)
            - 11.72716 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897)
            + 0.06208356 * max(0.0, Q.mass - 78.26182)
            + 0.002535827 * max(0.0, 935.1043 - Q.sum_pt_top15)
            + 3.521078 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 0.002957831 * max(0.0, 72.18744 - Q.mass_top15)
            + 73.34568 * max(0.0, 0.01655983 - Q.girth2_top20)
            + 77929.37 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.soft5_z - 0.001434897)
            - 0.03458123 * max(0.0, Q.mass - 64.48544)
            + 0.0674523 * max(0.0, 89.74183 - Q.mass)
            - 30.45374 * max(0.0, 0.04358622 - Q.e2)
            + 0.01109902 * max(0.0, Q.mass_top5 - 22.18342)
            - 0.01374664 * max(0.0, Q.mass_top30 - 78.53034)
        ))
        + 0.234375 * grid(12, max(0.0, -0.1367654
            - 0.01007103 * max(0.0, 86.4 - Q.mass)
            + 0.03305676 * max(0.0, Q.sd_mass - 125.1)
            - 0.1930504 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.007896893 * max(0.0, 89.6788 - Q.mass_top40)
            + 0.0007088699 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
            - 0.02023311 * max(0.0, Q.sj3_pair_mass_max - 120.6)
            - 3.712162 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            + 5.06142 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.0004162666 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.02589869 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 0.08176304 * max(0.0, 82.85409 - Q.mass)
            - 0.01703448 * max(0.0, 74.78616 - Q.mass_top40)
            - 92.36683 * max(0.0, 0.002752094 - Q.lam1)
            - 113.8474 * max(0.0, 0.006142802 - Q.girth2_top15)
            - 0.02567037 * max(0.0, 53.87362 - Q.mass)
            + 0.05065715 * max(0.0, 74.25181 - Q.mass)
            - 0.06247111 * max(0.0, 62.55 - Q.mass)
            + 11.77132 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 19.30728 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 4.322854 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40)
            - 1.469643 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots)
            - 5619.382 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 0.3323458
            + 0.1148417 * max(0.0, Q.n_particles - 38.0)
            + 15.17785 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.009297987 * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 12.53513 * max(0.0, Q.log_sum_pt - 6.98945)
            - 98.94735 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 0.002148106 * max(0.0, 689.25 - Q.sum_pt_top2)
            - 0.03049833 * max(0.0, 47.88842 - Q.mass_top20)
            - 0.008059208 * max(0.0, 32.50209 - Q.sj3_mass1)
            + 0.4506887 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.max_pair_mass - 13.04793)
            + 0.005807989 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.2905178 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1119555 - Q.dr_0)
            + 0.003746863 * max(0.0, 47.88842 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 29.0)
            - 0.03858602 * max(0.0, Q.n_particles - 38.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 169.6487 * max(0.0, 0.003270031 - Q.girth2_top15)
            - 0.08100759 * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 169.4878 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, 0.07279889 - Q.C2)
            + 1273.615 * max(0.0, 0.0005522528 - Q.girth2_top3)
            - 0.00148257 * max(0.0, 32.50209 - Q.sj3_mass1) * max(0.0, 18.68222 - Q.sj3_mass2)
            - 26.16099 * max(0.0, 0.03187688 - Q.M3)
            - 0.01779958 * max(0.0, 30.26161 - Q.sj2_mass1)
            - 0.04332009 * max(0.0, 31.35938 - Q.pt_9)
            + 0.1266055 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.1611545 - Q.dr_1)
            + 0.7369499 * max(0.0, Q.z_top30_slots - 0.9341838) * max(0.0, Q.ptdr0_3 - 7.407874)
            + 27.58894 * max(0.0, Q.log_sum_pt - 6.893714)
            + 134.0114 * max(0.0, 0.004673423 - Q.lam1)
            - 0.00956025 * max(0.0, 120.6 - Q.mass)
            + 49.43327 * max(0.0, 0.02412652 - Q.girth2_top30)
            + 0.04339613 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 0.3241858 - Q.dr1_12)
            + 2.023623 * max(0.0, 0.1416054 - Q.D3)
            - 4.403608 * max(0.0, Q.z_dr_0_0p05 - 0.878906)
            + 0.01634918 * max(0.0, 42.41192 - Q.mass_top30)
            - 7.741519 * max(0.0, 7.139296 - Q.log_sum_pt)
            + 0.01548025 * max(0.0, 12.0 - Q.n_dr_0_0p05)
            - 17.21745 * max(0.0, Q.log_sum_pt - 6.959294)
            + 0.001609659 * max(0.0, 31.35938 - Q.pt_9) * max(0.0, 11.29505 - Q.pair_mass_0_13)
            + 0.5230061 * max(0.0, 1.521582 - Q.soft1_pt)
            + 0.01023091 * max(0.0, 1017.435 - Q.sum_pt)
            + 0.003192628 * max(0.0, Q.sum_pt_top30 - 1191.938)
            + 4.214125 * max(0.0, Q.z_top20_slots - 0.8965411)
            - 0.009909199 * max(0.0, Q.sum_pt_top5 - 430.75) * max(0.0, 0.02980347 - Q.eta_0)
            + 0.2333272 * max(0.0, Q.pt_entropy - 2.07371)
        ))
        + 0.125 * grid(2, max(0.0, 0.3308957
            - 4.268359 * max(0.0, Q.log_sum_pt - 7.062574)
            + 0.01248307 * max(0.0, Q.sum_pt - 1017.435)
            + 687.5768 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 23.5794 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 0.0008429288 * max(0.0, Q.sum_pt_top50 - 988.4554)
            - 0.4424092 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 0.003557991 * max(0.0, Q.sum_pt_top50 - 1156.659)
            + 0.02283334 * max(0.0, 91.19 - Q.mass)
            + 10.12298 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 0.00468143 * max(0.0, 1041.263 - Q.sum_pt_top40)
            - 0.004172341 * max(0.0, 996.8867 - Q.sum_pt_top30)
            - 0.006163507 * max(0.0, Q.sum_pt - 1115.723)
            + 133.7992 * max(0.0, 0.004763596 - Q.girth2_top30)
            - 0.002673083 * max(0.0, 1260.541 - Q.sum_pt)
            - 89.90898 * max(0.0, 0.006363916 - Q.girth2_top30)
            - 0.05842994 * max(0.0, 92.85979 - Q.mass)
            + 62.50371 * max(0.0, 0.01174405 - Q.lam1)
            - 0.01553186 * max(0.0, 972.0419 - Q.sum_pt)
            + 271938.5 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4)
            - 12.56052 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.001175126 * max(0.0, Q.sum_pt_top40 - 984.7009)
            + 0.006898626 * max(0.0, 1048.098 - Q.sum_pt_top50)
            - 0.006325273 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.397021 - Q.max_dr)
            + 0.01054376 * max(0.0, 92.16545 - Q.mass_top50)
            - 0.3783806 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131)
            + 0.004361336 * max(0.0, Q.sum_pt_top40 - 1069.671)
            - 0.002689542 * max(0.0, 1078.994 - Q.sum_pt_top50)
        ))
        - 1.0625 * grid(4, max(0.0, 0.9994559
            - 0.002910387 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03907728 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0212475 * max(0.0, 120.6 - Q.mass)
            - 0.08147182 * max(0.0, 86.4 - Q.mass)
            - 0.004867626 * max(0.0, 120.6 - Q.mass_top30)
            + 176.9157 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.005725281 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02142293 * max(0.0, Q.n_particles - 22.0)
            - 177.857 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 12.45749 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05)
            + 0.0004443633 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0)
            + 0.006175368 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 60.28447 * max(0.0, Q.e2_sq - 0.01396296)
            - 78.6805 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01343086 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.06442411 * max(0.0, 101.0497 - Q.mass)
            + 0.03242072 * max(0.0, 26.0 - Q.n_dr_0p2_0p4)
            - 0.01126814 * max(0.0, 91.69753 - Q.mass_top30)
            - 0.008703325 * max(0.0, 80.78464 - Q.mass)
            - 0.01034394 * max(0.0, 74.25181 - Q.mass)
            - 0.0146105 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.001032432 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0)
            - 0.04550326 * max(0.0, 92.85979 - Q.mass)
            + 0.002005798 * max(0.0, 143.7876 - Q.mass)
            + 0.03131972 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.03630314 * max(0.0, 78.26182 - Q.mass)
            - 20.22809 * max(0.0, Q.girth - 0.1207452)
            - 123.4678 * max(0.0, Q.lam1 - 0.008241985)
            + 0.5648683 * max(0.0, 0.9909875 - Q.psi_0p1)
            + 76.19543 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 134.6129 * max(0.0, Q.e2_sq - 0.009606007)
            - 60.85614 * max(0.0, Q.lam2 - 0.001776308)
            - 0.00529117 * max(0.0, 91.19 - Q.mass_top15)
        ))
        + 0.21875 * grid(6, max(0.0, 1.170115
            + 0.01447067 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03719278 * max(0.0, 120.6 - Q.mass)
            + 0.0339753 * max(0.0, 86.4 - Q.mass)
            + 9.2013e-05 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 114.2108 * max(0.0, 0.002197765 - Q.girth2_top15)
            - 0.006836022 * max(0.0, Q.sj3_pair_mass_min - 29.00832)
            + 1.980592 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 78.82779 * max(0.0, Q.e2 - 0.05557149)
            - 0.07532423 * max(0.0, 101.0497 - Q.mass)
            + 0.07202664 * max(0.0, 92.85979 - Q.mass)
            - 0.02471428 * max(0.0, Q.sj3_mass1 - 21.11128)
            - 4.312245 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 21.73339 * max(0.0, 0.06310829 - Q.tau1)
            + 12.94304 * max(0.0, 0.9300465 - Q.z_top40_slots)
            - 0.1218576 * max(0.0, Q.sj3_pair_mass_min - 76.60223)
            - 6.175682 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
            + 157.402 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 1161.606 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 1133.33 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            - 0.003083477 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 33.3761)
            - 255.1533 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 116.7522 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 274.7502 * max(0.0, 0.003811746 - Q.lam1)
            - 44.33966 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 6.021715 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 318.9114 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 146.3964 * max(0.0, 0.003687605 - Q.lam2)
            + 0.009229968 * max(0.0, 91.19 - Q.mass_top15)
            + 0.005383053 * max(0.0, 172.8 - Q.mass_top50)
            - 2864.26 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 0.01535736 * max(0.0, Q.n_dr_0_0p05 - 9.0)
            - 34.5741 * max(0.0, Q.e2 - 0.02793599)
            + 188.0375 * max(0.0, 0.00375223 - Q.girth2_top30)
            - 46.89314 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 31.79539 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        + 0.015625 * grid(8, max(0.0, 2.066925
            - 18.60172 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.004676679 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 50.27689 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03026977 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0343019 * max(0.0, Q.mass - 125.1)
            + 0.04023553 * max(0.0, Q.mass - 87.36377)
            + 0.2598764 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.01794018 * max(0.0, 1017.435 - Q.sum_pt)
            - 594.6978 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06530352 * max(0.0, Q.mass - 101.0497)
            + 238.3678 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 14.98268 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 145.493 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.02477471 * max(0.0, Q.mass - 64.48544)
            + 151.623 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 432.9856 * max(0.0, 0.009614971 - Q.width)
            + 25.12502 * max(0.0, 0.007877041 - Q.girth2)
            + 5.86249 * max(0.0, Q.sj2_dr - 0.2232169)
            + 1.347847 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477)
            + 0.0408415 * max(0.0, 9.0 - Q.n_dr_0p2_0p4)
            - 0.0006934448 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct)
            - 0.06565953 * max(0.0, Q.n_for_90pct - 7.0)
            - 0.1541735 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296)
            + 49.8174 * max(0.0, Q.e2 - 0.04755309)
            + 100.3115 * max(0.0, 0.007671243 - Q.lam1)
            + 7.418645 * max(0.0, Q.C2 - 0.06655881)
            - 2876.53 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.05103926 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.03785952 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.02799519 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.0423991 * max(0.0, Q.mass - 74.25181)
            - 0.01009848 * max(0.0, Q.mass_top20 - 119.2969)
            - 694.5983 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157)
            + 0.02069189 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            + 0.4102108 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202)
            + 0.03298981 * max(0.0, Q.n_dr_0p2_0p4 - 3.0)
            + 0.01687216 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.009837468 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 1.308164 * max(0.0, 0.8316924 - Q.z_top15_slots)
            - 1.925304 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.2181557 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.1935704 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 193.494 * max(0.0, 0.008840538 - Q.girth2_top40)
            - 19.65455 * max(0.0, Q.z_top50_slots - 0.9704436)
            + 1395.118 * max(0.0, Q.e3 - 0.0003372339)
            - 12.77399 * max(0.0, 0.02146578 - Q.girth2_top15)
            - 23.13066 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.5625 * grid(9, max(0.0, -0.9585951
            + 0.02969574 * max(0.0, 80.89043 - Q.mass_top40)
            + 0.003029648 * max(0.0, 956.2133 - Q.sum_pt_top40)
            + 196.2683 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 0.002660812 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002300043 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            - 281.419 * max(0.0, 0.00217213 - Q.girth2_top40)
            + 0.07978109 * max(0.0, 172.8 - Q.mass)
            - 0.1562173 * max(0.0, 160.8 - Q.mass)
            + 0.06615134 * max(0.0, 92.85979 - Q.mass)
            + 21.56199 * max(0.0, Q.girth - 0.09749958)
            - 0.0967386 * max(0.0, 143.7876 - Q.mass)
            + 0.0002574458 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40)
            - 0.006547837 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047)
            + 0.05483302 * max(0.0, 82.85409 - Q.mass)
            - 0.06673127 * max(0.0, 64.48544 - Q.mass)
            - 0.02427501 * max(0.0, 163.2541 - Q.mass_top40)
            + 54.99904 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 0.0009842942 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            + 58.80979 * max(0.0, Q.lam1 - 0.01174405)
            - 23.01644 * max(0.0, Q.girth2_top30 - 0.0008564881)
            + 45.42692 * max(0.0, 0.004673423 - Q.lam1)
            - 90.31644 * max(0.0, Q.girth - 0.1402186)
            + 0.1295996 * max(0.0, 162.8363 - Q.mass)
            + 0.05061472 * max(0.0, 138.8977 - Q.mass_top50)
            + 0.01076456 * max(0.0, 73.33139 - Q.mass_top30)
            + 46.68351 * max(0.0, Q.LHA - 0.404204)
            - 0.04123687 * max(0.0, 92.16545 - Q.mass_top50)
            - 7.95405 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.02268123 * max(0.0, 986.0565 - Q.sum_pt)
            - 14.45285 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
        - 0.015625 * grid(10, max(0.0, -0.0267736
            - 5.715259 * max(0.0, 0.1207452 - Q.girth)
            - 0.0516948 * max(0.0, Q.mass - 162.8363)
            - 248.0863 * max(0.0, 0.005402331 - Q.girth2_top30)
            + 0.02087975 * max(0.0, Q.mass_top50 - 160.8)
            + 0.4566084 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, 631.275 - Q.sum_pt_top5)
            + 0.01892133 * max(0.0, 86.4 - Q.mass)
            - 0.007794864 * max(0.0, 65.20727 - Q.sj2_mass1)
            + 0.3495594 * max(0.0, 2.975532 - Q.D2)
            + 1.800257 * max(0.0, 0.5760704 - Q.z_top2_slots)
            - 0.005213073 * max(0.0, 59.40777 - Q.mass_top5)
            - 17.68862 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109)
            - 0.009389672 * max(0.0, 959.0957 - Q.sum_pt_top50)
            - 576.1586 * max(0.0, 0.002575211 - Q.girth2)
            - 0.0383639 * max(0.0, 101.0497 - Q.mass)
            - 3.062242 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855)
            + 7.297847 * max(0.0, 0.06413297 - Q.dr_0)
            - 3.460823 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 0.03371252 * max(0.0, Q.mass_top50 - 172.8)
            - 9.858262 * max(0.0, Q.e2 - 0.03263075)
            + 184716.1 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 22809.65 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 0.002926603 * max(0.0, Q.sum_pt_top10 - 943.6922)
            - 111.5653 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.4357228 - Q.max_dr)
            + 0.006737216 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1)
            + 39.3616 * max(0.0, Q.e2 - 0.01256572)
            - 52.75721 * max(0.0, Q.mass_over_sum_pt - 0.1606361)
            + 2352.6 * max(0.0, 0.0001086251 - Q.e3)
            + 22.66657 * max(0.0, 0.008824206 - Q.zdr_1)
            - 0.05728831 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 530.6533 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, 0.09696199 - Q.z_3)
            - 0.1022449 * max(0.0, Q.mass - 143.7876)
            + 0.0574559 * max(0.0, Q.mass_top50 - 136.785)
            - 11.72716 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897)
            + 0.06208356 * max(0.0, Q.mass - 78.26182)
            + 0.002535827 * max(0.0, 935.1043 - Q.sum_pt_top15)
            + 3.521078 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 0.002957831 * max(0.0, 72.18744 - Q.mass_top15)
            + 73.34568 * max(0.0, 0.01655983 - Q.girth2_top20)
            + 77929.37 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.soft5_z - 0.001434897)
            - 0.03458123 * max(0.0, Q.mass - 64.48544)
            + 0.0674523 * max(0.0, 89.74183 - Q.mass)
            - 30.45374 * max(0.0, 0.04358622 - Q.e2)
            + 0.01109902 * max(0.0, Q.mass_top5 - 22.18342)
            - 0.01374664 * max(0.0, Q.mass_top30 - 78.53034)
        ))
        - 0.25 * grid(11, max(0.0, -0.486065
            + 0.07880341 * max(0.0, 10.0 - Q.n_dr_0p2_0p4)
            - 0.002318625 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0)
            - 33.52185 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2)
            + 0.003232239 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 41.96531 * max(0.0, Q.psi_0p2 - 0.9935324)
            + 0.02467683 * max(0.0, 101.0497 - Q.mass)
            - 0.07091456 * max(0.0, 80.4 - Q.mass)
            + 0.07792138 * max(0.0, 92.85979 - Q.mass)
            - 3.654132e-05 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10)
            - 123.7977 * max(0.0, 0.00625621 - Q.girth2_top10)
            + 57.8536 * max(0.0, 0.02515919 - Q.e2)
            - 158.5417 * max(0.0, 0.006929741 - Q.girth2_top30)
            - 23.17973 * max(0.0, 0.03875945 - Q.e2)
            + 232.3638 * max(0.0, 0.00406126 - Q.girth2_top10)
            + 0.01903503 * max(0.0, 60.43821 - Q.mass_top30)
            - 0.01586427 * max(0.0, 83.32554 - Q.mass_top40)
            + 6.878053 * max(0.0, 0.07374472 - Q.girth)
            + 19.71852 * max(0.0, 0.08589404 - Q.girth)
            + 210.2211 * max(0.0, 0.00818374 - Q.e2_sq)
            - 6.533153 * max(0.0, 0.06030419 - Q.mass_over_sum_pt)
            - 38.30052 * max(0.0, 0.076787 - Q.girth)
            - 236.4662 * max(0.0, 0.006716737 - Q.lam1)
            + 11906.21 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, 0.006580753 - Q.C2_b2)
            - 17119.6 * max(0.0, 3.793233e-05 - Q.e3)
            + 0.005379502 * max(0.0, 67.72643 - Q.mass_top40)
            - 7.062757 * max(0.0, Q.psi_0p2 - 0.9313699)
            + 74.45728 * max(0.0, 0.03029714 - Q.e2)
            - 0.02496561 * max(0.0, 80.35535 - Q.mass_top50)
            + 61.7223 * max(0.0, Q.psi_0p3 - 0.9896594)
            + 0.3503997 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0402832 - Q.phi_0)
            - 6.09833 * max(0.0, Q.C2_b2 - 0.01330402)
        ))
        + 0.34375 * grid(12, max(0.0, -0.1367654
            - 0.01007103 * max(0.0, 86.4 - Q.mass)
            + 0.03305676 * max(0.0, Q.sd_mass - 125.1)
            - 0.1930504 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.007896893 * max(0.0, 89.6788 - Q.mass_top40)
            + 0.0007088699 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
            - 0.02023311 * max(0.0, Q.sj3_pair_mass_max - 120.6)
            - 3.712162 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            + 5.06142 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.0004162666 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.02589869 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 0.08176304 * max(0.0, 82.85409 - Q.mass)
            - 0.01703448 * max(0.0, 74.78616 - Q.mass_top40)
            - 92.36683 * max(0.0, 0.002752094 - Q.lam1)
            - 113.8474 * max(0.0, 0.006142802 - Q.girth2_top15)
            - 0.02567037 * max(0.0, 53.87362 - Q.mass)
            + 0.05065715 * max(0.0, 74.25181 - Q.mass)
            - 0.06247111 * max(0.0, 62.55 - Q.mass)
            + 11.77132 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 19.30728 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 4.322854 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40)
            - 1.469643 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots)
            - 5619.382 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.228678
            - 0.1158331 * max(0.0, Q.mass - 78.26182)
            - 0.1296368 * max(0.0, Q.mass - 92.85979)
            + 89.55922 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.009377984 * max(0.0, 1012.673 - Q.sum_pt)
            + 49.82551 * max(0.0, Q.psi_0p3 - 0.9956185)
            + 0.06008246 * max(0.0, Q.mass - 91.19)
            - 0.02973846 * max(0.0, Q.mass - 74.25181)
            + 0.01198491 * max(0.0, 80.4 - Q.mass_top30)
            - 63.6503 * max(0.0, 0.005913555 - Q.lam1)
            + 298.1063 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 19.58872 * max(0.0, 0.0705748 - Q.tau1)
            - 395.3828 * max(0.0, 0.00616708 - Q.e2_sq)
            + 0.04855343 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 3.5055 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 20.24792 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.03329021 * max(0.0, 101.0497 - Q.mass)
            - 167.5541 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 421.5461 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            + 2.028084 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.003657897 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.001987747 * max(0.0, 846.1934 - Q.sum_pt_top20)
            + 0.01113031 * max(0.0, 80.89043 - Q.mass_top40)
            - 128.8417 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.08750739 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3)
            + 0.01335146 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.01918552 * max(0.0, Q.mass - 89.74183)
            + 0.003006878 * max(0.0, 1156.659 - Q.sum_pt_top50)
            + 0.04044547 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
        ))
        + 0.4375 * grid(3, max(0.0, -0.09849501
            + 806.4848 * max(0.0, 0.0006154841 - Q.lam2)
            + 0.184767 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            + 0.02504442 * max(0.0, 46.0 - Q.n_particles)
            - 0.001116056 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30)
            - 0.001252897 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349)
            + 280.4078 * max(0.0, Q.psi_0p2 - 0.9985434)
            - 0.0130078 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0)
            + 4.924219e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 4.009614 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03447112 - Q.z_10)
            - 1.922232 * max(0.0, 0.3861957 - Q.tau21)
            - 0.1042589 * max(0.0, 2.410481 - Q.D2)
            + 11.90596 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 97.15128 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 18.20203 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            + 0.5548369 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1)
            - 326.0059 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.005771511 * max(0.0, 79.21004 - Q.mass_top50)
            - 0.008459062 * max(0.0, 86.4 - Q.mass)
            - 133.8843 * max(0.0, 0.009614971 - Q.girth2)
            + 144.7654 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 1.334769 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741)
            - 0.01126831 * max(0.0, 53.87362 - Q.mass)
            + 0.009280592 * max(0.0, 92.85979 - Q.mass)
            + 0.01680284 * max(0.0, 27.56535 - Q.sj2_mass1)
            - 0.01025869 * max(0.0, 80.4 - Q.mass_top40)
            + 21.90357 * max(0.0, 0.01626937 - Q.tau4)
            - 4059.072 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.08575439 - Q.abseta_4)
        ))
        + 0.34375 * grid(4, max(0.0, 0.9994559
            - 0.002910387 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03907728 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0212475 * max(0.0, 120.6 - Q.mass)
            - 0.08147182 * max(0.0, 86.4 - Q.mass)
            - 0.004867626 * max(0.0, 120.6 - Q.mass_top30)
            + 176.9157 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.005725281 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02142293 * max(0.0, Q.n_particles - 22.0)
            - 177.857 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 12.45749 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05)
            + 0.0004443633 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0)
            + 0.006175368 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 60.28447 * max(0.0, Q.e2_sq - 0.01396296)
            - 78.6805 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01343086 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.06442411 * max(0.0, 101.0497 - Q.mass)
            + 0.03242072 * max(0.0, 26.0 - Q.n_dr_0p2_0p4)
            - 0.01126814 * max(0.0, 91.69753 - Q.mass_top30)
            - 0.008703325 * max(0.0, 80.78464 - Q.mass)
            - 0.01034394 * max(0.0, 74.25181 - Q.mass)
            - 0.0146105 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.001032432 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0)
            - 0.04550326 * max(0.0, 92.85979 - Q.mass)
            + 0.002005798 * max(0.0, 143.7876 - Q.mass)
            + 0.03131972 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.03630314 * max(0.0, 78.26182 - Q.mass)
            - 20.22809 * max(0.0, Q.girth - 0.1207452)
            - 123.4678 * max(0.0, Q.lam1 - 0.008241985)
            + 0.5648683 * max(0.0, 0.9909875 - Q.psi_0p1)
            + 76.19543 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 134.6129 * max(0.0, Q.e2_sq - 0.009606007)
            - 60.85614 * max(0.0, Q.lam2 - 0.001776308)
            - 0.00529117 * max(0.0, 91.19 - Q.mass_top15)
        ))
        + 0.578125 * grid(5, max(0.0, 0.6833986
            + 0.03663246 * max(0.0, 64.0 - Q.n_particles)
            - 29.0871 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 5.20108 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.005175716 * max(0.0, Q.sum_pt - 907.9372)
            - 11.67306 * max(0.0, Q.log_sum_pt - 6.920349)
            + 195.4541 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 59.68106 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01729337 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.01866498 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01145388 * max(0.0, Q.n_pt_above_1 - 28.0)
            + 0.02827257 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 25726.1 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 17.63508 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.004013931 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 6.094535 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.001172525 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.224028 * max(0.0, Q.max_dr - 0.2404747)
            - 7.40116 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.01465448 * max(0.0, Q.mass_top40 - 120.6)
            + 0.02871264 * max(0.0, Q.mass - 172.8)
            + 28.39648 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005553738 * max(0.0, 150.0144 - Q.mass_top40)
            - 510.6845 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 1.454749 * max(0.0, 0.5494307 - Q.tau21)
            + 0.03687624 * max(0.0, Q.sd_mass - 69.65633)
            - 0.2087503 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.01065712 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002118901 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 161.3584 * max(0.0, 0.01375115 - Q.z_11)
            - 0.01378824 * max(0.0, Q.mass - 91.19)
            + 0.01994 * max(0.0, Q.mass - 74.25181)
            + 10.15916 * max(0.0, 0.03875945 - Q.e2)
            - 0.02020562 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.06683936 * max(0.0, 8.0 - Q.n_dr_0p1_0p2)
            - 0.04854561 * max(0.0, Q.sd_mass - 86.4)
            + 0.007455275 * max(0.0, Q.mass_top10 - 71.781)
            - 2.502079 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4)
            - 0.1477321 * max(0.0, 14.14062 - Q.pt_11)
            + 1.869872 * max(0.0, Q.max_dr - 0.4357228)
            + 0.3503087 * max(0.0, 1.976207 - Q.D2)
        ))
        + 0.0625 * grid(6, max(0.0, 1.170115
            + 0.01447067 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03719278 * max(0.0, 120.6 - Q.mass)
            + 0.0339753 * max(0.0, 86.4 - Q.mass)
            + 9.2013e-05 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 114.2108 * max(0.0, 0.002197765 - Q.girth2_top15)
            - 0.006836022 * max(0.0, Q.sj3_pair_mass_min - 29.00832)
            + 1.980592 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 78.82779 * max(0.0, Q.e2 - 0.05557149)
            - 0.07532423 * max(0.0, 101.0497 - Q.mass)
            + 0.07202664 * max(0.0, 92.85979 - Q.mass)
            - 0.02471428 * max(0.0, Q.sj3_mass1 - 21.11128)
            - 4.312245 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 21.73339 * max(0.0, 0.06310829 - Q.tau1)
            + 12.94304 * max(0.0, 0.9300465 - Q.z_top40_slots)
            - 0.1218576 * max(0.0, Q.sj3_pair_mass_min - 76.60223)
            - 6.175682 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
            + 157.402 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 1161.606 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 1133.33 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            - 0.003083477 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 33.3761)
            - 255.1533 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 116.7522 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 274.7502 * max(0.0, 0.003811746 - Q.lam1)
            - 44.33966 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 6.021715 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 318.9114 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 146.3964 * max(0.0, 0.003687605 - Q.lam2)
            + 0.009229968 * max(0.0, 91.19 - Q.mass_top15)
            + 0.005383053 * max(0.0, 172.8 - Q.mass_top50)
            - 2864.26 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 0.01535736 * max(0.0, Q.n_dr_0_0p05 - 9.0)
            - 34.5741 * max(0.0, Q.e2 - 0.02793599)
            + 188.0375 * max(0.0, 0.00375223 - Q.girth2_top30)
            - 46.89314 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 31.79539 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        - 0.625 * grid(7, max(0.0, -0.2381245
            + 8.160562 * max(0.0, 0.2352054 - Q.tau21_b2)
            + 328.1319 * max(0.0, 0.007877041 - Q.girth2)
            - 1169.139 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            + 18.48574 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 30.18798 * max(0.0, 0.006403325 - Q.girth2)
            + 0.03818142 * max(0.0, 91.19 - Q.mass)
            - 0.162007 * max(0.0, 82.85409 - Q.mass)
            - 0.09537123 * max(0.0, 101.0497 - Q.mass)
            + 0.07266804 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 804.8958 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 10.29801 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            + 8.300748 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 2.242106 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 16.57704 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 4.495176 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 0.009451522 * max(0.0, 92.85979 - Q.mass)
            - 91.94118 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 4.132055 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 77.00884 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 0.01354448 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 0.04096687 * max(0.0, 120.6 - Q.mass)
            - 52.52341 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.1063277 - Q.z_2)
            - 8.543108 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 1.301918 * max(0.0, 0.3861957 - Q.tau21)
            - 0.02170227 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088)
            + 196.5933 * max(0.0, 0.000404306 - Q.lam2)
            - 1585.736 * max(0.0, 0.00363788 - Q.e2_sq)
            + 0.06802817 * max(0.0, 78.26182 - Q.mass)
            - 492.1524 * max(0.0, 0.008190222 - Q.girth2)
            + 171.937 * max(0.0, 0.009614971 - Q.girth2)
            - 27.88261 * max(0.0, 0.1072713 - Q.tau1)
            - 0.1300488 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05)
            + 18.86846 * max(0.0, 0.09591084 - Q.tau1)
            - 180.3144 * max(0.0, 0.008241985 - Q.lam1)
            + 158.1114 * max(0.0, 0.006189818 - Q.lam1)
            + 292.4853 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.03434674 * max(0.0, 69.65633 - Q.sd_mass)
            - 0.02866072 * max(0.0, 86.4 - Q.sd_mass)
            - 0.01869258 * max(0.0, Q.mass_top20 - 73.35236)
            + 0.03496524 * max(0.0, Q.mass_top20 - 85.79457)
            - 0.09463646 * max(0.0, Q.sd_mass - 98.05743)
        ))
        - 0.875 * grid(8, max(0.0, 2.066925
            - 18.60172 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.004676679 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 50.27689 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03026977 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0343019 * max(0.0, Q.mass - 125.1)
            + 0.04023553 * max(0.0, Q.mass - 87.36377)
            + 0.2598764 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.01794018 * max(0.0, 1017.435 - Q.sum_pt)
            - 594.6978 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06530352 * max(0.0, Q.mass - 101.0497)
            + 238.3678 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 14.98268 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 145.493 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.02477471 * max(0.0, Q.mass - 64.48544)
            + 151.623 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 432.9856 * max(0.0, 0.009614971 - Q.width)
            + 25.12502 * max(0.0, 0.007877041 - Q.girth2)
            + 5.86249 * max(0.0, Q.sj2_dr - 0.2232169)
            + 1.347847 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477)
            + 0.0408415 * max(0.0, 9.0 - Q.n_dr_0p2_0p4)
            - 0.0006934448 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct)
            - 0.06565953 * max(0.0, Q.n_for_90pct - 7.0)
            - 0.1541735 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296)
            + 49.8174 * max(0.0, Q.e2 - 0.04755309)
            + 100.3115 * max(0.0, 0.007671243 - Q.lam1)
            + 7.418645 * max(0.0, Q.C2 - 0.06655881)
            - 2876.53 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.05103926 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.03785952 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.02799519 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.0423991 * max(0.0, Q.mass - 74.25181)
            - 0.01009848 * max(0.0, Q.mass_top20 - 119.2969)
            - 694.5983 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157)
            + 0.02069189 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            + 0.4102108 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202)
            + 0.03298981 * max(0.0, Q.n_dr_0p2_0p4 - 3.0)
            + 0.01687216 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.009837468 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 1.308164 * max(0.0, 0.8316924 - Q.z_top15_slots)
            - 1.925304 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.2181557 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.1935704 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 193.494 * max(0.0, 0.008840538 - Q.girth2_top40)
            - 19.65455 * max(0.0, Q.z_top50_slots - 0.9704436)
            + 1395.118 * max(0.0, Q.e3 - 0.0003372339)
            - 12.77399 * max(0.0, 0.02146578 - Q.girth2_top15)
            - 23.13066 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        - 0.21875 * grid(9, max(0.0, -0.9585951
            + 0.02969574 * max(0.0, 80.89043 - Q.mass_top40)
            + 0.003029648 * max(0.0, 956.2133 - Q.sum_pt_top40)
            + 196.2683 * max(0.0, 0.006259772 - Q.girth2_top40)
            + 0.002660812 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft4_pt - 1.789258)
            - 0.0002300043 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 1034.834 - Q.sum_pt)
            - 281.419 * max(0.0, 0.00217213 - Q.girth2_top40)
            + 0.07978109 * max(0.0, 172.8 - Q.mass)
            - 0.1562173 * max(0.0, 160.8 - Q.mass)
            + 0.06615134 * max(0.0, 92.85979 - Q.mass)
            + 21.56199 * max(0.0, Q.girth - 0.09749958)
            - 0.0967386 * max(0.0, 143.7876 - Q.mass)
            + 0.0002574458 * max(0.0, 80.89043 - Q.mass_top40) * max(0.0, 906.6023 - Q.sum_pt_top40)
            - 0.006547837 * max(0.0, 956.2133 - Q.sum_pt_top40) * max(0.0, Q.soft3_pt - 2.873047)
            + 0.05483302 * max(0.0, 82.85409 - Q.mass)
            - 0.06673127 * max(0.0, 64.48544 - Q.mass)
            - 0.02427501 * max(0.0, 163.2541 - Q.mass_top40)
            + 54.99904 * max(0.0, Q.psi_0p3 - 0.9943058)
            + 0.0009842942 * max(0.0, 125.1 - Q.mass_top40) * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            + 58.80979 * max(0.0, Q.lam1 - 0.01174405)
            - 23.01644 * max(0.0, Q.girth2_top30 - 0.0008564881)
            + 45.42692 * max(0.0, 0.004673423 - Q.lam1)
            - 90.31644 * max(0.0, Q.girth - 0.1402186)
            + 0.1295996 * max(0.0, 162.8363 - Q.mass)
            + 0.05061472 * max(0.0, 138.8977 - Q.mass_top50)
            + 0.01076456 * max(0.0, 73.33139 - Q.mass_top30)
            + 46.68351 * max(0.0, Q.LHA - 0.404204)
            - 0.04123687 * max(0.0, 92.16545 - Q.mass_top50)
            - 7.95405 * max(0.0, Q.z_top40_slots - 0.9574183)
            + 0.02268123 * max(0.0, 986.0565 - Q.sum_pt)
            - 14.45285 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
        + 0.59375 * grid(11, max(0.0, -0.486065
            + 0.07880341 * max(0.0, 10.0 - Q.n_dr_0p2_0p4)
            - 0.002318625 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_particles - 34.0)
            - 33.52185 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208 - Q.girth2)
            + 0.003232239 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 41.96531 * max(0.0, Q.psi_0p2 - 0.9935324)
            + 0.02467683 * max(0.0, 101.0497 - Q.mass)
            - 0.07091456 * max(0.0, 80.4 - Q.mass)
            + 0.07792138 * max(0.0, 92.85979 - Q.mass)
            - 3.654132e-05 * max(0.0, 101.0497 - Q.mass) * max(0.0, 891.875 - Q.sum_pt_top10)
            - 123.7977 * max(0.0, 0.00625621 - Q.girth2_top10)
            + 57.8536 * max(0.0, 0.02515919 - Q.e2)
            - 158.5417 * max(0.0, 0.006929741 - Q.girth2_top30)
            - 23.17973 * max(0.0, 0.03875945 - Q.e2)
            + 232.3638 * max(0.0, 0.00406126 - Q.girth2_top10)
            + 0.01903503 * max(0.0, 60.43821 - Q.mass_top30)
            - 0.01586427 * max(0.0, 83.32554 - Q.mass_top40)
            + 6.878053 * max(0.0, 0.07374472 - Q.girth)
            + 19.71852 * max(0.0, 0.08589404 - Q.girth)
            + 210.2211 * max(0.0, 0.00818374 - Q.e2_sq)
            - 6.533153 * max(0.0, 0.06030419 - Q.mass_over_sum_pt)
            - 38.30052 * max(0.0, 0.076787 - Q.girth)
            - 236.4662 * max(0.0, 0.006716737 - Q.lam1)
            + 11906.21 * max(0.0, Q.z_top40_slots - 0.996191) * max(0.0, 0.006580753 - Q.C2_b2)
            - 17119.6 * max(0.0, 3.793233e-05 - Q.e3)
            + 0.005379502 * max(0.0, 67.72643 - Q.mass_top40)
            - 7.062757 * max(0.0, Q.psi_0p2 - 0.9313699)
            + 74.45728 * max(0.0, 0.03029714 - Q.e2)
            - 0.02496561 * max(0.0, 80.35535 - Q.mass_top50)
            + 61.7223 * max(0.0, Q.psi_0p3 - 0.9896594)
            + 0.3503997 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.0402832 - Q.phi_0)
            - 6.09833 * max(0.0, Q.C2_b2 - 0.01330402)
        ))
        - 0.40625 * grid(12, max(0.0, -0.1367654
            - 0.01007103 * max(0.0, 86.4 - Q.mass)
            + 0.03305676 * max(0.0, Q.sd_mass - 125.1)
            - 0.1930504 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.007896893 * max(0.0, 89.6788 - Q.mass_top40)
            + 0.0007088699 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
            - 0.02023311 * max(0.0, Q.sj3_pair_mass_max - 120.6)
            - 3.712162 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            + 5.06142 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.0004162666 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.02589869 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 0.08176304 * max(0.0, 82.85409 - Q.mass)
            - 0.01703448 * max(0.0, 74.78616 - Q.mass_top40)
            - 92.36683 * max(0.0, 0.002752094 - Q.lam1)
            - 113.8474 * max(0.0, 0.006142802 - Q.girth2_top15)
            - 0.02567037 * max(0.0, 53.87362 - Q.mass)
            + 0.05065715 * max(0.0, 74.25181 - Q.mass)
            - 0.06247111 * max(0.0, 62.55 - Q.mass)
            + 11.77132 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 19.30728 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 4.322854 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40)
            - 1.469643 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots)
            - 5619.382 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2)
        ))
        - 1.375 * grid(14, max(0.0, -0.2037838
            + 1.50336 * max(0.0, 0.342495 - Q.tau21_b2)
            - 112.178 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 0.02048524 - Q.lam1)
            - 0.1275332 * max(0.0, 91.19 - Q.mass)
            - 0.05047521 * max(0.0, 82.85409 - Q.mass)
            + 137.9205 * max(0.0, Q.psi_0p3 - 0.9924477)
            - 9.18554 * max(0.0, 0.4226723 - Q.N2) * max(0.0, Q.max_dr - 0.2404747)
            + 0.01999488 * max(0.0, 18.0 - Q.n_dr_0p2_0p4)
            + 40.52825 * max(0.0, 0.03029714 - Q.e2)
            - 88.62893 * max(0.0, 0.09046749 - Q.mass_over_sum_pt)
            + 20.83798 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            - 1657.383 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, Q.sd_rg - 0.2280025)
            - 703.889 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1881908 - Q.sd_rg)
            - 90.0185 * max(0.0, 0.01083435 - Q.girth2_top20)
            + 56.55063 * max(0.0, 0.08873143 - Q.mass_over_sum_pt)
            - 0.07610898 * max(0.0, 80.4 - Q.mass)
            - 0.2926539 * max(0.0, 0.342495 - Q.tau21_b2) * max(0.0, 32.0 - Q.n_real_top40)
            - 214.4554 * max(0.0, 0.09046749 - Q.mass_over_sum_pt) * max(0.0, Q.psi_0p2 - 0.9087063)
            - 13.71272 * max(0.0, 0.076787 - Q.girth)
            + 113.3541 * max(0.0, 0.007164202 - Q.girth2_top5)
            + 35.02081 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            + 101.5093 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.009710323 * max(0.0, 69.65633 - Q.sd_mass)
            - 2.802161 * max(0.0, 0.3017146 - Q.sd_rg)
            + 15.24583 * max(0.0, 0.140939 - Q.mass_over_sum_pt)
            - 121.6839 * max(0.0, 0.01215787 - Q.girth2_top30)
            + 0.01973522 * max(0.0, 82.66587 - Q.mass_top30)
            + 9.652055 * max(0.0, 0.02210818 - Q.e2)
            - 0.005433328 * max(0.0, 906.6023 - Q.sum_pt_top40)
            - 959.0022 * max(0.0, Q.psi_0p3 - 0.9924477) * max(0.0, 0.1231747 - Q.z_2nd)
            - 258.5159 * max(0.0, 0.001592178 - Q.girth2_top3)
            + 11.57 * max(0.0, 6.941997 - Q.log_sum_pt)
            - 0.006240268 * max(0.0, 976.277 - Q.sum_pt_top50)
            - 4.91618 * max(0.0, 6.98945 - Q.log_sum_pt)
            - 9.007927 * max(0.0, 0.007164202 - Q.girth2_top5) * max(0.0, Q.n_pt_above_10 - 13.0)
            - 323.3158 * max(0.0, 0.076787 - Q.girth) * max(0.0, 0.05180474 - Q.z_dr_0p2_0p4)
            + 1.326096 * max(0.0, 0.1778185 - Q.sd_rg)
            + 0.002901682 * max(0.0, 1024.942 - Q.sum_pt_top40)
        ))
        - 0.1875 * grid(15, max(0.0, -0.66192
            + 6.674051 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 134.9697 * max(0.0, 0.008329695 - Q.girth2_top5)
            - 5.300563 * max(0.0, Q.psi_0p1 - 0.8976117)
            + 0.07699942 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles)
            + 0.01109338 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.90068 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 0.006240211 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt)
            - 49.79105 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.001355688 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt)
            - 0.2898822 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0)
            - 0.004203414 * max(0.0, 1002.379 - Q.sum_pt)
            - 261.8278 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 3.286926 * max(0.0, Q.sj2_dr - 0.2232169)
            + 168.3261 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 44.8179 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 35.67514 * max(0.0, Q.lam1 - 0.01649354)
            - 1.70142 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            - 0.8869562 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0)
            + 123.2734 * max(0.0, 0.001776308 - Q.lam2)
            - 10.89404 * max(0.0, 0.0705748 - Q.tau1)
            - 150.3673 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.0179492 * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            - 91.33302 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583)
            + 0.01806152 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02176641 * max(0.0, 45.595 - Q.sd_mass)
            + 0.01832243 * max(0.0, 35.30013 - Q.sj3_pair_mass_max)
            - 0.007504977 * max(0.0, 71.79516 - Q.mass_top50)
            + 80.78979 * max(0.0, 0.004673423 - Q.lam1)
            - 0.005476328 * max(0.0, 935.8189 - Q.sum_pt_top40)
            - 0.009753303 * max(0.0, 1018.698 - Q.sum_pt_top40)
            + 0.006422017 * max(0.0, 933.1875 - Q.sum_pt_top30)
            - 7.906435 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.228678
            - 0.1158331 * max(0.0, Q.mass - 78.26182)
            - 0.1296368 * max(0.0, Q.mass - 92.85979)
            + 89.55922 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.009377984 * max(0.0, 1012.673 - Q.sum_pt)
            + 49.82551 * max(0.0, Q.psi_0p3 - 0.9956185)
            + 0.06008246 * max(0.0, Q.mass - 91.19)
            - 0.02973846 * max(0.0, Q.mass - 74.25181)
            + 0.01198491 * max(0.0, 80.4 - Q.mass_top30)
            - 63.6503 * max(0.0, 0.005913555 - Q.lam1)
            + 298.1063 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 19.58872 * max(0.0, 0.0705748 - Q.tau1)
            - 395.3828 * max(0.0, 0.00616708 - Q.e2_sq)
            + 0.04855343 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 3.5055 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 20.24792 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.03329021 * max(0.0, 101.0497 - Q.mass)
            - 167.5541 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 421.5461 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            + 2.028084 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.003657897 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.001987747 * max(0.0, 846.1934 - Q.sum_pt_top20)
            + 0.01113031 * max(0.0, 80.89043 - Q.mass_top40)
            - 128.8417 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.08750739 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3)
            + 0.01335146 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.01918552 * max(0.0, Q.mass - 89.74183)
            + 0.003006878 * max(0.0, 1156.659 - Q.sum_pt_top50)
            + 0.04044547 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
        ))
        - 0.375 * grid(2, max(0.0, 0.3308957
            - 4.268359 * max(0.0, Q.log_sum_pt - 7.062574)
            + 0.01248307 * max(0.0, Q.sum_pt - 1017.435)
            + 687.5768 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, 0.02146578 - Q.girth2_top15)
            - 23.5794 * max(0.0, Q.log_sum_pt - 6.903423) * max(0.0, Q.psi_0p3 - 0.9299135)
            - 0.0008429288 * max(0.0, Q.sum_pt_top50 - 988.4554)
            - 0.4424092 * max(0.0, Q.sum_pt_top50 - 988.4554) * max(0.0, 0.02146578 - Q.girth2_top15)
            + 0.003557991 * max(0.0, Q.sum_pt_top50 - 1156.659)
            + 0.02283334 * max(0.0, 91.19 - Q.mass)
            + 10.12298 * max(0.0, 0.09795415 - Q.mass_over_sum_pt)
            + 0.00468143 * max(0.0, 1041.263 - Q.sum_pt_top40)
            - 0.004172341 * max(0.0, 996.8867 - Q.sum_pt_top30)
            - 0.006163507 * max(0.0, Q.sum_pt - 1115.723)
            + 133.7992 * max(0.0, 0.004763596 - Q.girth2_top30)
            - 0.002673083 * max(0.0, 1260.541 - Q.sum_pt)
            - 89.90898 * max(0.0, 0.006363916 - Q.girth2_top30)
            - 0.05842994 * max(0.0, 92.85979 - Q.mass)
            + 62.50371 * max(0.0, 0.01174405 - Q.lam1)
            - 0.01553186 * max(0.0, 972.0419 - Q.sum_pt)
            + 271938.5 * max(0.0, 972.0419 - Q.sum_pt) * max(0.0, 5.8505e-08 - Q.e4)
            - 12.56052 * max(0.0, Q.log_sum_pt - 6.959294)
            - 0.001175126 * max(0.0, Q.sum_pt_top40 - 984.7009)
            + 0.006898626 * max(0.0, 1048.098 - Q.sum_pt_top50)
            - 0.006325273 * max(0.0, 1260.541 - Q.sum_pt) * max(0.0, 0.397021 - Q.max_dr)
            + 0.01054376 * max(0.0, 92.16545 - Q.mass_top50)
            - 0.3783806 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.08734131)
            + 0.004361336 * max(0.0, Q.sum_pt_top40 - 1069.671)
            - 0.002689542 * max(0.0, 1078.994 - Q.sum_pt_top50)
        ))
        + 0.4375 * grid(3, max(0.0, -0.09849501
            + 806.4848 * max(0.0, 0.0006154841 - Q.lam2)
            + 0.184767 * max(0.0, 5.0 - Q.n_dr_0p2_0p4)
            + 0.02504442 * max(0.0, 46.0 - Q.n_particles)
            - 0.001116056 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30)
            - 0.001252897 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top15 - 57.87349)
            + 280.4078 * max(0.0, Q.psi_0p2 - 0.9985434)
            - 0.0130078 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 9.0)
            + 4.924219e-05 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top30 - 800.732)
            - 4.009614 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.03447112 - Q.z_10)
            - 1.922232 * max(0.0, 0.3861957 - Q.tau21)
            - 0.1042589 * max(0.0, 2.410481 - Q.D2)
            + 11.90596 * max(0.0, 0.08665515 - Q.mass_over_sum_pt)
            + 97.15128 * max(0.0, 2.410481 - Q.D2) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 18.20203 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.007099471 - Q.zdr_5)
            + 0.5548369 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.9538343 - Q.psi_0p1)
            - 326.0059 * max(0.0, 0.3861957 - Q.tau21) * max(0.0, Q.lam1 - 0.007671243)
            - 0.005771511 * max(0.0, 79.21004 - Q.mass_top50)
            - 0.008459062 * max(0.0, 86.4 - Q.mass)
            - 133.8843 * max(0.0, 0.009614971 - Q.girth2)
            + 144.7654 * max(0.0, 0.008840538 - Q.girth2_top40)
            + 1.334769 * max(0.0, 8.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741)
            - 0.01126831 * max(0.0, 53.87362 - Q.mass)
            + 0.009280592 * max(0.0, 92.85979 - Q.mass)
            + 0.01680284 * max(0.0, 27.56535 - Q.sj2_mass1)
            - 0.01025869 * max(0.0, 80.4 - Q.mass_top40)
            + 21.90357 * max(0.0, 0.01626937 - Q.tau4)
            - 4059.072 * max(0.0, Q.psi_0p2 - 0.9985434) * max(0.0, 0.08575439 - Q.abseta_4)
        ))
        + 0.6875 * grid(5, max(0.0, 0.6833986
            + 0.03663246 * max(0.0, 64.0 - Q.n_particles)
            - 29.0871 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 5.20108 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.005175716 * max(0.0, Q.sum_pt - 907.9372)
            - 11.67306 * max(0.0, Q.log_sum_pt - 6.920349)
            + 195.4541 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 59.68106 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01729337 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.01866498 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01145388 * max(0.0, Q.n_pt_above_1 - 28.0)
            + 0.02827257 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 25726.1 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 17.63508 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.004013931 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 6.094535 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.001172525 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.224028 * max(0.0, Q.max_dr - 0.2404747)
            - 7.40116 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.01465448 * max(0.0, Q.mass_top40 - 120.6)
            + 0.02871264 * max(0.0, Q.mass - 172.8)
            + 28.39648 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005553738 * max(0.0, 150.0144 - Q.mass_top40)
            - 510.6845 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 1.454749 * max(0.0, 0.5494307 - Q.tau21)
            + 0.03687624 * max(0.0, Q.sd_mass - 69.65633)
            - 0.2087503 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.01065712 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002118901 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 161.3584 * max(0.0, 0.01375115 - Q.z_11)
            - 0.01378824 * max(0.0, Q.mass - 91.19)
            + 0.01994 * max(0.0, Q.mass - 74.25181)
            + 10.15916 * max(0.0, 0.03875945 - Q.e2)
            - 0.02020562 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.06683936 * max(0.0, 8.0 - Q.n_dr_0p1_0p2)
            - 0.04854561 * max(0.0, Q.sd_mass - 86.4)
            + 0.007455275 * max(0.0, Q.mass_top10 - 71.781)
            - 2.502079 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4)
            - 0.1477321 * max(0.0, 14.14062 - Q.pt_11)
            + 1.869872 * max(0.0, Q.max_dr - 0.4357228)
            + 0.3503087 * max(0.0, 1.976207 - Q.D2)
        ))
        - 0.96875 * grid(6, max(0.0, 1.170115
            + 0.01447067 * max(0.0, 71.79516 - Q.mass_top50)
            - 0.03719278 * max(0.0, 120.6 - Q.mass)
            + 0.0339753 * max(0.0, 86.4 - Q.mass)
            + 9.2013e-05 * max(0.0, 120.6 - Q.mass) * max(0.0, 1007.788 - Q.sum_pt)
            - 114.2108 * max(0.0, 0.002197765 - Q.girth2_top15)
            - 0.006836022 * max(0.0, Q.sj3_pair_mass_min - 29.00832)
            + 1.980592 * max(0.0, Q.sj3_pair_mass_min - 29.00832) * max(0.0, Q.psi_0p3 - 0.9896594)
            + 78.82779 * max(0.0, Q.e2 - 0.05557149)
            - 0.07532423 * max(0.0, 101.0497 - Q.mass)
            + 0.07202664 * max(0.0, 92.85979 - Q.mass)
            - 0.02471428 * max(0.0, Q.sj3_mass1 - 21.11128)
            - 4.312245 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 21.73339 * max(0.0, 0.06310829 - Q.tau1)
            + 12.94304 * max(0.0, 0.9300465 - Q.z_top40_slots)
            - 0.1218576 * max(0.0, Q.sj3_pair_mass_min - 76.60223)
            - 6.175682 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
            + 157.402 * max(0.0, Q.girth2_top30 - 0.008376291)
            + 1161.606 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, Q.C2_b2 - 0.02704832)
            + 1133.33 * max(0.0, Q.e2 - 0.05557149) * max(0.0, Q.zdr_0 - 0.001369707)
            - 0.003083477 * max(0.0, Q.n_dr_0p2_0p4 - 15.0) * max(0.0, Q.max_pair_mass - 33.3761)
            - 255.1533 * max(0.0, Q.girth2_top30 - 0.02809026)
            + 116.7522 * max(0.0, Q.girth2_top20 - 0.008031209)
            + 274.7502 * max(0.0, 0.003811746 - Q.lam1)
            - 44.33966 * max(0.0, Q.girth2_top30 - 0.008376291) * max(0.0, Q.D2_b2 - 1.67722)
            - 6.021715 * max(0.0, Q.mass_over_sum_pt - 0.1182259) * max(0.0, 7.36624 - Q.D2_b2)
            - 318.9114 * max(0.0, 0.004573744 - Q.girth2_top50)
            + 146.3964 * max(0.0, 0.003687605 - Q.lam2)
            + 0.009229968 * max(0.0, 91.19 - Q.mass_top15)
            + 0.005383053 * max(0.0, 172.8 - Q.mass_top50)
            - 2864.26 * max(0.0, Q.girth2_top20 - 0.008031209) * max(0.0, Q.C2_b2 - 0.0008187529)
            - 0.01535736 * max(0.0, Q.n_dr_0_0p05 - 9.0)
            - 34.5741 * max(0.0, Q.e2 - 0.02793599)
            + 188.0375 * max(0.0, 0.00375223 - Q.girth2_top30)
            - 46.89314 * max(0.0, 0.01807679 - Q.girth2_top30)
            - 31.79539 * max(0.0, Q.mass_over_sum_pt - 0.09795415)
        ))
        + 0.90625 * grid(7, max(0.0, -0.2381245
            + 8.160562 * max(0.0, 0.2352054 - Q.tau21_b2)
            + 328.1319 * max(0.0, 0.007877041 - Q.girth2)
            - 1169.139 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            + 18.48574 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 30.18798 * max(0.0, 0.006403325 - Q.girth2)
            + 0.03818142 * max(0.0, 91.19 - Q.mass)
            - 0.162007 * max(0.0, 82.85409 - Q.mass)
            - 0.09537123 * max(0.0, 101.0497 - Q.mass)
            + 0.07266804 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 804.8958 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 10.29801 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            + 8.300748 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 2.242106 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 16.57704 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 4.495176 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 0.009451522 * max(0.0, 92.85979 - Q.mass)
            - 91.94118 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 4.132055 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 77.00884 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 0.01354448 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 0.04096687 * max(0.0, 120.6 - Q.mass)
            - 52.52341 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.1063277 - Q.z_2)
            - 8.543108 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 1.301918 * max(0.0, 0.3861957 - Q.tau21)
            - 0.02170227 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088)
            + 196.5933 * max(0.0, 0.000404306 - Q.lam2)
            - 1585.736 * max(0.0, 0.00363788 - Q.e2_sq)
            + 0.06802817 * max(0.0, 78.26182 - Q.mass)
            - 492.1524 * max(0.0, 0.008190222 - Q.girth2)
            + 171.937 * max(0.0, 0.009614971 - Q.girth2)
            - 27.88261 * max(0.0, 0.1072713 - Q.tau1)
            - 0.1300488 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05)
            + 18.86846 * max(0.0, 0.09591084 - Q.tau1)
            - 180.3144 * max(0.0, 0.008241985 - Q.lam1)
            + 158.1114 * max(0.0, 0.006189818 - Q.lam1)
            + 292.4853 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.03434674 * max(0.0, 69.65633 - Q.sd_mass)
            - 0.02866072 * max(0.0, 86.4 - Q.sd_mass)
            - 0.01869258 * max(0.0, Q.mass_top20 - 73.35236)
            + 0.03496524 * max(0.0, Q.mass_top20 - 85.79457)
            - 0.09463646 * max(0.0, Q.sd_mass - 98.05743)
        ))
        - 0.9375 * grid(8, max(0.0, 2.066925
            - 18.60172 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.004676679 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 50.27689 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03026977 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0343019 * max(0.0, Q.mass - 125.1)
            + 0.04023553 * max(0.0, Q.mass - 87.36377)
            + 0.2598764 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.01794018 * max(0.0, 1017.435 - Q.sum_pt)
            - 594.6978 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06530352 * max(0.0, Q.mass - 101.0497)
            + 238.3678 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 14.98268 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 145.493 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.02477471 * max(0.0, Q.mass - 64.48544)
            + 151.623 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 432.9856 * max(0.0, 0.009614971 - Q.width)
            + 25.12502 * max(0.0, 0.007877041 - Q.girth2)
            + 5.86249 * max(0.0, Q.sj2_dr - 0.2232169)
            + 1.347847 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477)
            + 0.0408415 * max(0.0, 9.0 - Q.n_dr_0p2_0p4)
            - 0.0006934448 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct)
            - 0.06565953 * max(0.0, Q.n_for_90pct - 7.0)
            - 0.1541735 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296)
            + 49.8174 * max(0.0, Q.e2 - 0.04755309)
            + 100.3115 * max(0.0, 0.007671243 - Q.lam1)
            + 7.418645 * max(0.0, Q.C2 - 0.06655881)
            - 2876.53 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.05103926 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.03785952 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.02799519 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.0423991 * max(0.0, Q.mass - 74.25181)
            - 0.01009848 * max(0.0, Q.mass_top20 - 119.2969)
            - 694.5983 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157)
            + 0.02069189 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            + 0.4102108 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202)
            + 0.03298981 * max(0.0, Q.n_dr_0p2_0p4 - 3.0)
            + 0.01687216 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.009837468 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 1.308164 * max(0.0, 0.8316924 - Q.z_top15_slots)
            - 1.925304 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.2181557 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.1935704 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 193.494 * max(0.0, 0.008840538 - Q.girth2_top40)
            - 19.65455 * max(0.0, Q.z_top50_slots - 0.9704436)
            + 1395.118 * max(0.0, Q.e3 - 0.0003372339)
            - 12.77399 * max(0.0, 0.02146578 - Q.girth2_top15)
            - 23.13066 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.3125 * grid(12, max(0.0, -0.1367654
            - 0.01007103 * max(0.0, 86.4 - Q.mass)
            + 0.03305676 * max(0.0, Q.sd_mass - 125.1)
            - 0.1930504 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.007896893 * max(0.0, 89.6788 - Q.mass_top40)
            + 0.0007088699 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
            - 0.02023311 * max(0.0, Q.sj3_pair_mass_max - 120.6)
            - 3.712162 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            + 5.06142 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.0004162666 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.02589869 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 0.08176304 * max(0.0, 82.85409 - Q.mass)
            - 0.01703448 * max(0.0, 74.78616 - Q.mass_top40)
            - 92.36683 * max(0.0, 0.002752094 - Q.lam1)
            - 113.8474 * max(0.0, 0.006142802 - Q.girth2_top15)
            - 0.02567037 * max(0.0, 53.87362 - Q.mass)
            + 0.05065715 * max(0.0, 74.25181 - Q.mass)
            - 0.06247111 * max(0.0, 62.55 - Q.mass)
            + 11.77132 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 19.30728 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 4.322854 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40)
            - 1.469643 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots)
            - 5619.382 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2)
        ))
        + 0.5625 * grid(15, max(0.0, -0.66192
            + 6.674051 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 134.9697 * max(0.0, 0.008329695 - Q.girth2_top5)
            - 5.300563 * max(0.0, Q.psi_0p1 - 0.8976117)
            + 0.07699942 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles)
            + 0.01109338 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.90068 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 0.006240211 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt)
            - 49.79105 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.001355688 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt)
            - 0.2898822 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0)
            - 0.004203414 * max(0.0, 1002.379 - Q.sum_pt)
            - 261.8278 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 3.286926 * max(0.0, Q.sj2_dr - 0.2232169)
            + 168.3261 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 44.8179 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 35.67514 * max(0.0, Q.lam1 - 0.01649354)
            - 1.70142 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            - 0.8869562 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0)
            + 123.2734 * max(0.0, 0.001776308 - Q.lam2)
            - 10.89404 * max(0.0, 0.0705748 - Q.tau1)
            - 150.3673 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.0179492 * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            - 91.33302 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583)
            + 0.01806152 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02176641 * max(0.0, 45.595 - Q.sd_mass)
            + 0.01832243 * max(0.0, 35.30013 - Q.sj3_pair_mass_max)
            - 0.007504977 * max(0.0, 71.79516 - Q.mass_top50)
            + 80.78979 * max(0.0, 0.004673423 - Q.lam1)
            - 0.005476328 * max(0.0, 935.8189 - Q.sum_pt_top40)
            - 0.009753303 * max(0.0, 1018.698 - Q.sum_pt_top40)
            + 0.006422017 * max(0.0, 933.1875 - Q.sum_pt_top30)
            - 7.906435 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.228678
            - 0.1158331 * max(0.0, Q.mass - 78.26182)
            - 0.1296368 * max(0.0, Q.mass - 92.85979)
            + 89.55922 * max(0.0, 0.005312783 - Q.girth2_top20)
            - 0.009377984 * max(0.0, 1012.673 - Q.sum_pt)
            + 49.82551 * max(0.0, Q.psi_0p3 - 0.9956185)
            + 0.06008246 * max(0.0, Q.mass - 91.19)
            - 0.02973846 * max(0.0, Q.mass - 74.25181)
            + 0.01198491 * max(0.0, 80.4 - Q.mass_top30)
            - 63.6503 * max(0.0, 0.005913555 - Q.lam1)
            + 298.1063 * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            - 19.58872 * max(0.0, 0.0705748 - Q.tau1)
            - 395.3828 * max(0.0, 0.00616708 - Q.e2_sq)
            + 0.04855343 * max(0.0, 6.98945 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top50 - 959.0957)
            - 3.5055 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 20.24792 * max(0.0, 0.9906378 - Q.z_top50_slots)
            - 0.03329021 * max(0.0, 101.0497 - Q.mass)
            - 167.5541 * max(0.0, 0.006363916 - Q.girth2_top30)
            + 421.5461 * max(0.0, 0.006938798 - Q.mass_over_sum_pt_sq)
            + 2.028084 * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.003657897 * max(0.0, 1069.671 - Q.sum_pt_top40)
            + 0.001987747 * max(0.0, 846.1934 - Q.sum_pt_top20)
            + 0.01113031 * max(0.0, 80.89043 - Q.mass_top40)
            - 128.8417 * max(0.0, 0.006374178 - Q.girth2_top20)
            + 0.08750739 * max(0.0, 1012.673 - Q.sum_pt) * max(0.0, 0.0223982 - Q.C3)
            + 0.01335146 * max(0.0, Q.mass_top50 - 71.79516)
            - 0.01918552 * max(0.0, Q.mass - 89.74183)
            + 0.003006878 * max(0.0, 1156.659 - Q.sum_pt_top50)
            + 0.04044547 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
        ))
        + 0.1875 * grid(4, max(0.0, 0.9994559
            - 0.002910387 * max(0.0, 83.32554 - Q.mass_top40)
            + 0.03907728 * max(0.0, 60.43821 - Q.mass_top30)
            + 0.0212475 * max(0.0, 120.6 - Q.mass)
            - 0.08147182 * max(0.0, 86.4 - Q.mass)
            - 0.004867626 * max(0.0, 120.6 - Q.mass_top30)
            + 176.9157 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.005725281 * max(0.0, Q.sj3_pair_mass_min - 32.51366)
            - 0.02142293 * max(0.0, Q.n_particles - 22.0)
            - 177.857 * max(0.0, 0.004855289 - Q.girth2_top15)
            - 12.45749 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 13.0 - Q.n_dr_0_0p05)
            + 0.0004443633 * max(0.0, Q.n_particles - 22.0) * max(0.0, Q.n_dr_0_0p05 - 10.0)
            + 0.006175368 * max(0.0, Q.n_particles - 22.0) * max(0.0, 2.275391 - Q.soft1_pt)
            + 60.28447 * max(0.0, Q.e2_sq - 0.01396296)
            - 78.6805 * max(0.0, Q.girth2_top15 - 0.00727763)
            + 0.01343086 * max(0.0, 57.87349 - Q.mass_top15)
            + 0.06442411 * max(0.0, 101.0497 - Q.mass)
            + 0.03242072 * max(0.0, 26.0 - Q.n_dr_0p2_0p4)
            - 0.01126814 * max(0.0, 91.69753 - Q.mass_top30)
            - 0.008703325 * max(0.0, 80.78464 - Q.mass)
            - 0.01034394 * max(0.0, 74.25181 - Q.mass)
            - 0.0146105 * max(0.0, 163.2541 - Q.mass_top40)
            - 0.001032432 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.0)
            - 0.04550326 * max(0.0, 92.85979 - Q.mass)
            + 0.002005798 * max(0.0, 143.7876 - Q.mass)
            + 0.03131972 * max(0.0, 67.72643 - Q.mass_top40)
            - 0.03630314 * max(0.0, 78.26182 - Q.mass)
            - 20.22809 * max(0.0, Q.girth - 0.1207452)
            - 123.4678 * max(0.0, Q.lam1 - 0.008241985)
            + 0.5648683 * max(0.0, 0.9909875 - Q.psi_0p1)
            + 76.19543 * max(0.0, Q.girth2_top15 - 0.01563836)
            + 134.6129 * max(0.0, Q.e2_sq - 0.009606007)
            - 60.85614 * max(0.0, Q.lam2 - 0.001776308)
            - 0.00529117 * max(0.0, 91.19 - Q.mass_top15)
        ))
        - 0.46875 * grid(5, max(0.0, 0.6833986
            + 0.03663246 * max(0.0, 64.0 - Q.n_particles)
            - 29.0871 * max(0.0, Q.mass_over_sum_pt - 0.09046749)
            + 5.20108 * max(0.0, Q.log_sum_pt - 6.935549)
            + 0.005175716 * max(0.0, Q.sum_pt - 907.9372)
            - 11.67306 * max(0.0, Q.log_sum_pt - 6.920349)
            + 195.4541 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 59.68106 * max(0.0, Q.girth2_top50 - 0.01951641)
            - 0.01729337 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 2.178951 - Q.D2)
            + 0.01866498 * max(0.0, Q.sum_pt_top50 - 934.2416)
            + 0.01145388 * max(0.0, Q.n_pt_above_1 - 28.0)
            + 0.02827257 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 25726.1 * max(0.0, Q.sum_pt - 907.9372) * max(0.0, 5.8505e-08 - Q.e4)
            - 17.63508 * max(0.0, Q.log_sum_pt - 6.910131)
            - 0.004013931 * max(0.0, Q.sum_pt_top40 - 1024.942)
            + 6.094535 * max(0.0, Q.log_sum_pt - 6.98945)
            + 0.001172525 * max(0.0, Q.sum_pt_top30 - 933.1875)
            - 2.224028 * max(0.0, Q.max_dr - 0.2404747)
            - 7.40116 * max(0.0, Q.z_top30_slots - 0.9048492)
            - 0.01465448 * max(0.0, Q.mass_top40 - 120.6)
            + 0.02871264 * max(0.0, Q.mass - 172.8)
            + 28.39648 * max(0.0, Q.psi_0p3 - 0.9973959) * max(0.0, 3.345339 - Q.D2)
            - 0.005553738 * max(0.0, 150.0144 - Q.mass_top40)
            - 510.6845 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 1.454749 * max(0.0, 0.5494307 - Q.tau21)
            + 0.03687624 * max(0.0, Q.sd_mass - 69.65633)
            - 0.2087503 * max(0.0, Q.mass_top50 - 157.5448)
            - 0.01065712 * max(0.0, Q.sum_pt - 986.0565)
            - 0.002118901 * max(0.0, Q.sum_pt_top10 - 943.6922)
            + 161.3584 * max(0.0, 0.01375115 - Q.z_11)
            - 0.01378824 * max(0.0, Q.mass - 91.19)
            + 0.01994 * max(0.0, Q.mass - 74.25181)
            + 10.15916 * max(0.0, 0.03875945 - Q.e2)
            - 0.02020562 * max(0.0, 21.0 - Q.n_dr_0p1_0p2)
            + 0.06683936 * max(0.0, 8.0 - Q.n_dr_0p1_0p2)
            - 0.04854561 * max(0.0, Q.sd_mass - 86.4)
            + 0.007455275 * max(0.0, Q.mass_top10 - 71.781)
            - 2.502079 * max(0.0, 0.1937447 - Q.z_dr_0p2_0p4)
            - 0.1477321 * max(0.0, 14.14062 - Q.pt_11)
            + 1.869872 * max(0.0, Q.max_dr - 0.4357228)
            + 0.3503087 * max(0.0, 1.976207 - Q.D2)
        ))
        - 0.28125 * grid(7, max(0.0, -0.2381245
            + 8.160562 * max(0.0, 0.2352054 - Q.tau21_b2)
            + 328.1319 * max(0.0, 0.007877041 - Q.girth2)
            - 1169.139 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.007873266 - Q.mass_over_sum_pt_sq)
            + 18.48574 * max(0.0, 0.1182259 - Q.mass_over_sum_pt)
            - 30.18798 * max(0.0, 0.006403325 - Q.girth2)
            + 0.03818142 * max(0.0, 91.19 - Q.mass)
            - 0.162007 * max(0.0, 82.85409 - Q.mass)
            - 0.09537123 * max(0.0, 101.0497 - Q.mass)
            + 0.07266804 * max(0.0, 6.0 - Q.n_dr_0p2_0p4)
            - 804.8958 * max(0.0, Q.psi_0p3 - 0.9980008)
            + 10.29801 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            + 8.300748 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 2.242106 * max(0.0, 82.85409 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            - 16.57704 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 4.495176 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9777125)
            + 0.009451522 * max(0.0, 92.85979 - Q.mass)
            - 91.94118 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 4.132055 * max(0.0, 82.85409 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            + 77.00884 * max(0.0, 101.0497 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9980008)
            - 0.01354448 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 1260.541 - Q.sum_pt)
            + 0.04096687 * max(0.0, 120.6 - Q.mass)
            - 52.52341 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, 0.1063277 - Q.z_2)
            - 8.543108 * max(0.0, 92.85979 - Q.mass) * max(0.0, Q.psi_0p3 - 0.9638082)
            - 1.301918 * max(0.0, 0.3861957 - Q.tau21)
            - 0.02170227 * max(0.0, 0.2352054 - Q.tau21_b2) * max(0.0, Q.orientation_deg - -9.840088)
            + 196.5933 * max(0.0, 0.000404306 - Q.lam2)
            - 1585.736 * max(0.0, 0.00363788 - Q.e2_sq)
            + 0.06802817 * max(0.0, 78.26182 - Q.mass)
            - 492.1524 * max(0.0, 0.008190222 - Q.girth2)
            + 171.937 * max(0.0, 0.009614971 - Q.girth2)
            - 27.88261 * max(0.0, 0.1072713 - Q.tau1)
            - 0.1300488 * max(0.0, 92.85979 - Q.mass) * max(0.0, 0.4947602 - Q.z_dr_0_0p05)
            + 18.86846 * max(0.0, 0.09591084 - Q.tau1)
            - 180.3144 * max(0.0, 0.008241985 - Q.lam1)
            + 158.1114 * max(0.0, 0.006189818 - Q.lam1)
            + 292.4853 * max(0.0, Q.psi_0p3 - 0.9973959)
            + 0.03434674 * max(0.0, 69.65633 - Q.sd_mass)
            - 0.02866072 * max(0.0, 86.4 - Q.sd_mass)
            - 0.01869258 * max(0.0, Q.mass_top20 - 73.35236)
            + 0.03496524 * max(0.0, Q.mass_top20 - 85.79457)
            - 0.09463646 * max(0.0, Q.sd_mass - 98.05743)
        ))
        + 0.2109375 * grid(8, max(0.0, 2.066925
            - 18.60172 * max(0.0, Q.mass_over_sum_pt - 0.07696632)
            + 0.004676679 * max(0.0, 1001.523 - Q.sum_pt_top40)
            + 50.27689 * max(0.0, Q.girth2_top40 - 0.005196966)
            - 0.03026977 * max(0.0, 15.0 - Q.n_dr_0p2_0p4)
            - 0.0343019 * max(0.0, Q.mass - 125.1)
            + 0.04023553 * max(0.0, Q.mass - 87.36377)
            + 0.2598764 * max(0.0, Q.mass_over_sum_pt - 0.07696632) * max(0.0, 1115.723 - Q.sum_pt)
            + 0.01794018 * max(0.0, 1017.435 - Q.sum_pt)
            - 594.6978 * max(0.0, Q.girth2_top40 - 0.005196966) * max(0.0, 7.017258 - Q.log_sum_pt)
            - 0.06530352 * max(0.0, Q.mass - 101.0497)
            + 238.3678 * max(0.0, Q.girth2_top20 - 0.008031209)
            - 14.98268 * max(0.0, 6.930088 - Q.log_sum_pt)
            - 145.493 * max(0.0, Q.girth2_top20 - 0.006043209)
            + 0.02477471 * max(0.0, Q.mass - 64.48544)
            + 151.623 * max(0.0, 0.007463985 - Q.girth2_top30)
            - 432.9856 * max(0.0, 0.009614971 - Q.width)
            + 25.12502 * max(0.0, 0.007877041 - Q.girth2)
            + 5.86249 * max(0.0, Q.sj2_dr - 0.2232169)
            + 1.347847 * max(0.0, Q.z_dr_0p1_0p2 - 0.3340477)
            + 0.0408415 * max(0.0, 9.0 - Q.n_dr_0p2_0p4)
            - 0.0006934448 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, 13.0 - Q.n_for_90pct)
            - 0.06565953 * max(0.0, Q.n_for_90pct - 7.0)
            - 0.1541735 * max(0.0, Q.mass - 64.48544) * max(0.0, Q.zdr_0 - 0.0008718296)
            + 49.8174 * max(0.0, Q.e2 - 0.04755309)
            + 100.3115 * max(0.0, 0.007671243 - Q.lam1)
            + 7.418645 * max(0.0, Q.C2 - 0.06655881)
            - 2876.53 * max(0.0, Q.e3 - 5.13841e-05)
            - 0.05103926 * max(0.0, 39.0 - Q.n_for_90pct)
            - 0.03785952 * max(0.0, Q.mass_top50 - 82.04491)
            + 0.02799519 * max(0.0, Q.mass_top50 - 117.0487)
            + 0.0423991 * max(0.0, Q.mass - 74.25181)
            - 0.01009848 * max(0.0, Q.mass_top20 - 119.2969)
            - 694.5983 * max(0.0, 0.007463985 - Q.girth2_top30) * max(0.0, Q.sj2_dr - 0.1512157)
            + 0.02069189 * max(0.0, Q.n_dr_0p1_0p2 - 19.0)
            + 0.4102108 * max(0.0, 0.0795038 - Q.tau2) * max(0.0, Q.sj3_mass1 - 13.38202)
            + 0.03298981 * max(0.0, Q.n_dr_0p2_0p4 - 3.0)
            + 0.01687216 * max(0.0, 1028.184 - Q.sum_pt)
            - 0.009837468 * max(0.0, 1024.942 - Q.sum_pt_top40)
            + 1.308164 * max(0.0, 0.8316924 - Q.z_top15_slots)
            - 1.925304 * max(0.0, 0.3628388 - Q.psi_0p1)
            - 0.2181557 * max(0.0, 1017.435 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 0.1935704 * max(0.0, 1028.184 - Q.sum_pt) * max(0.0, Q.dr_max_012 - 0.1828389)
            + 193.494 * max(0.0, 0.008840538 - Q.girth2_top40)
            - 19.65455 * max(0.0, Q.z_top50_slots - 0.9704436)
            + 1395.118 * max(0.0, Q.e3 - 0.0003372339)
            - 12.77399 * max(0.0, 0.02146578 - Q.girth2_top15)
            - 23.13066 * max(0.0, Q.mass_over_sum_pt - 0.1182259)
        ))
        + 0.984375 * grid(10, max(0.0, -0.0267736
            - 5.715259 * max(0.0, 0.1207452 - Q.girth)
            - 0.0516948 * max(0.0, Q.mass - 162.8363)
            - 248.0863 * max(0.0, 0.005402331 - Q.girth2_top30)
            + 0.02087975 * max(0.0, Q.mass_top50 - 160.8)
            + 0.4566084 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, 631.275 - Q.sum_pt_top5)
            + 0.01892133 * max(0.0, 86.4 - Q.mass)
            - 0.007794864 * max(0.0, 65.20727 - Q.sj2_mass1)
            + 0.3495594 * max(0.0, 2.975532 - Q.D2)
            + 1.800257 * max(0.0, 0.5760704 - Q.z_top2_slots)
            - 0.005213073 * max(0.0, 59.40777 - Q.mass_top5)
            - 17.68862 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.soft4_z - 0.001721109)
            - 0.009389672 * max(0.0, 959.0957 - Q.sum_pt_top50)
            - 576.1586 * max(0.0, 0.002575211 - Q.girth2)
            - 0.0383639 * max(0.0, 101.0497 - Q.mass)
            - 3.062242 * max(0.0, 2.975532 - Q.D2) * max(0.0, Q.sj2_dr - 0.2070855)
            + 7.297847 * max(0.0, 0.06413297 - Q.dr_0)
            - 3.460823 * max(0.0, Q.z_dr_0_0p05 - 0.7674734)
            - 0.03371252 * max(0.0, Q.mass_top50 - 172.8)
            - 9.858262 * max(0.0, Q.e2 - 0.03263075)
            + 184716.1 * max(0.0, 0.005402331 - Q.girth2_top30) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 22809.65 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, Q.psi_0p3 - 0.9985421)
            - 0.002926603 * max(0.0, Q.sum_pt_top10 - 943.6922)
            - 111.5653 * max(0.0, 0.01976735 - Q.girth2_top10) * max(0.0, 0.4357228 - Q.max_dr)
            + 0.006737216 * max(0.0, 59.40777 - Q.mass_top5) * max(0.0, 0.8509215 - Q.z_dr_0p05_0p1)
            + 39.3616 * max(0.0, Q.e2 - 0.01256572)
            - 52.75721 * max(0.0, Q.mass_over_sum_pt - 0.1606361)
            + 2352.6 * max(0.0, 0.0001086251 - Q.e3)
            + 22.66657 * max(0.0, 0.008824206 - Q.zdr_1)
            - 0.05728831 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            + 530.6533 * max(0.0, Q.mass_over_sum_pt - 0.1606361) * max(0.0, 0.09696199 - Q.z_3)
            - 0.1022449 * max(0.0, Q.mass - 143.7876)
            + 0.0574559 * max(0.0, Q.mass_top50 - 136.785)
            - 11.72716 * max(0.0, Q.mass - 162.8363) * max(0.0, Q.soft5_z - 0.001434897)
            + 0.06208356 * max(0.0, Q.mass - 78.26182)
            + 0.002535827 * max(0.0, 935.1043 - Q.sum_pt_top15)
            + 3.521078 * max(0.0, 0.09122568 - Q.z_dr_0p2_0p4)
            - 0.002957831 * max(0.0, 72.18744 - Q.mass_top15)
            + 73.34568 * max(0.0, 0.01655983 - Q.girth2_top20)
            + 77929.37 * max(0.0, Q.psi_0p3 - 0.9985421) * max(0.0, Q.soft5_z - 0.001434897)
            - 0.03458123 * max(0.0, Q.mass - 64.48544)
            + 0.0674523 * max(0.0, 89.74183 - Q.mass)
            - 30.45374 * max(0.0, 0.04358622 - Q.e2)
            + 0.01109902 * max(0.0, Q.mass_top5 - 22.18342)
            - 0.01374664 * max(0.0, Q.mass_top30 - 78.53034)
        ))
        - 0.375 * grid(12, max(0.0, -0.1367654
            - 0.01007103 * max(0.0, 86.4 - Q.mass)
            + 0.03305676 * max(0.0, Q.sd_mass - 125.1)
            - 0.1930504 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.06472584 - Q.z_dr_0p1_0p2)
            - 0.007896893 * max(0.0, 89.6788 - Q.mass_top40)
            + 0.0007088699 * max(0.0, 89.6788 - Q.mass_top40) * max(0.0, Q.n_dr_0p05_0p1 - 1.0)
            - 0.02023311 * max(0.0, Q.sj3_pair_mass_max - 120.6)
            - 3.712162 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.9985421 - Q.psi_0p3)
            + 5.06142 * max(0.0, 0.05077291 - Q.mass_over_sum_pt)
            + 0.0004162666 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, 76.60223 - Q.sj3_pair_mass_min)
            + 0.02589869 * max(0.0, Q.sj3_pair_mass_max - 120.6) * max(0.0, Q.z_dr_0p1_0p2 - 0.2864926)
            + 0.08176304 * max(0.0, 82.85409 - Q.mass)
            - 0.01703448 * max(0.0, 74.78616 - Q.mass_top40)
            - 92.36683 * max(0.0, 0.002752094 - Q.lam1)
            - 113.8474 * max(0.0, 0.006142802 - Q.girth2_top15)
            - 0.02567037 * max(0.0, 53.87362 - Q.mass)
            + 0.05065715 * max(0.0, 74.25181 - Q.mass)
            - 0.06247111 * max(0.0, 62.55 - Q.mass)
            + 11.77132 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.003687605 - Q.lam2)
            - 19.30728 * max(0.0, 0.06895248 - Q.mass_over_sum_pt)
            + 4.322854 * max(0.0, 0.0007431905 - Q.girth2_top10) * max(0.0, 1069.671 - Q.sum_pt_top40)
            - 1.469643 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.985099 - Q.z_top50_slots)
            - 5619.382 * max(0.0, 6.856375 - Q.log_sum_pt) * max(0.0, 0.002396991 - Q.lam2)
        ))
        - 0.90625 * grid(13, max(0.0, 2.212836
            - 0.004405036 * max(0.0, 1085.125 - Q.sum_pt)
            - 0.03664991 * max(0.0, Q.mass - 136.785)
            - 0.06930863 * max(0.0, 16.0 - Q.n_for_90pct)
            + 0.0382748 * max(0.0, Q.mass - 172.8)
            + 0.0195719 * max(0.0, Q.mass - 74.25181)
            - 3.308651e-05 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 858.8262 - Q.sum_pt_top40)
            - 0.01401395 * max(0.0, 1007.788 - Q.sum_pt)
            + 262.2861 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            + 17.21668 * max(0.0, 6.811175 - Q.log_sum_pt)
            + 0.002455553 * max(0.0, 1053.047 - Q.sum_pt_top40)
            - 0.005116102 * max(0.0, 1085.125 - Q.sum_pt) * max(0.0, 0.8310045 - Q.tau21_b2)
            + 0.01299926 * max(0.0, 16.0 - Q.n_for_90pct) * max(0.0, 3.814159 - Q.D2)
            - 0.08447323 * max(0.0, Q.mass - 160.8)
            + 0.01242689 * max(0.0, 1053.047 - Q.sum_pt_top40) * max(0.0, 0.2179035 - Q.sj2_zsoft)
            + 0.08631755 * max(0.0, Q.mass_top50 - 136.785)
            + 46.91348 * max(0.0, 0.02550569 - Q.girth2_top50)
            + 4.552817e-05 * max(0.0, 1007.788 - Q.sum_pt) * max(0.0, Q.sum_pt_top3 - 512.4375)
            - 1783.757 * max(0.0, Q.log_sum_pt - 7.139296) * max(0.0, 0.00317537 - Q.C3)
            - 0.03045358 * max(0.0, Q.mass_top50 - 92.16545)
            - 0.1176332 * max(0.0, Q.mass - 143.7876)
            - 587.2654 * max(0.0, Q.mass_over_sum_pt_sq - 0.02920002)
            - 3.231438 * max(0.0, Q.tau1 - 0.1751567)
            + 0.006377498 * Q.n_particles
            - 12.35443 * max(0.0, 6.910131 - Q.log_sum_pt)
            + 86.8368 * max(0.0, 0.007820315 - Q.girth2_top50)
            - 0.01245685 * max(0.0, 160.8 - Q.mass_top40)
            + 0.00534158 * max(0.0, 997.0189 - Q.sum_pt_top50)
            - 0.002939385 * max(0.0, 966.0633 - Q.sum_pt_top30)
            + 0.01444172 * max(0.0, Q.mass_top5 - 33.71058)
            + 0.01724623 * max(0.0, 18.32812 - Q.pt_11)
            - 0.002906893 * max(0.0, Q.mass - 172.8) * max(0.0, 56.53125 - Q.pt_6)
            + 2.487708 * max(0.0, Q.mass - 172.8) * max(0.0, 0.04568661 - Q.z_6)
            + 0.005103648 * max(0.0, 76.9886 - Q.mass_top10)
            - 35.67838 * max(0.0, 0.02580396 - Q.mass_over_sum_pt_sq)
            + 0.06545624 * max(0.0, Q.mass - 162.8363)
            + 0.0008927598 * max(0.0, Q.sum_pt_top20 - 956.5062)
            + 0.006270314 * max(0.0, 1007.44 - Q.sum_pt_top40)
        ))
        - 0.375 * grid(15, max(0.0, -0.66192
            + 6.674051 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2)
            + 134.9697 * max(0.0, 0.008329695 - Q.girth2_top5)
            - 5.300563 * max(0.0, Q.psi_0p1 - 0.8976117)
            + 0.07699942 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, 58.0 - Q.n_particles)
            + 0.01109338 * max(0.0, 986.0565 - Q.sum_pt)
            + 6.90068 * max(0.0, 7.017258 - Q.log_sum_pt)
            + 0.006240211 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 949.9169 - Q.sum_pt)
            - 49.79105 * max(0.0, Q.psi_0p3 - 0.9896594)
            - 0.001355688 * max(0.0, 0.8459004 - Q.z_dr_0_0p05) * max(0.0, 1260.541 - Q.sum_pt)
            - 0.2898822 * max(0.0, 0.1203437 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 5.0)
            - 0.004203414 * max(0.0, 1002.379 - Q.sum_pt)
            - 261.8278 * max(0.0, 0.002270363 - Q.girth2_top5)
            + 3.286926 * max(0.0, Q.sj2_dr - 0.2232169)
            + 168.3261 * max(0.0, Q.sj2_dr - 0.2232169) * max(0.0, 0.04008677 - Q.C2_b2)
            + 44.8179 * max(0.0, 0.007678544 - Q.girth2_top10)
            + 35.67514 * max(0.0, Q.lam1 - 0.01649354)
            - 1.70142 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_mass1 - 5.112677)
            - 0.8869562 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.02970886 - Q.absphi_0)
            + 123.2734 * max(0.0, 0.001776308 - Q.lam2)
            - 10.89404 * max(0.0, 0.0705748 - Q.tau1)
            - 150.3673 * max(0.0, Q.mass_over_sum_pt - 0.1708801)
            - 0.0179492 * max(0.0, Q.n_dr_0p2_0p4 - 9.0)
            - 91.33302 * max(0.0, 0.008329695 - Q.girth2_top5) * max(0.0, Q.sj3_pairmin_over_m - 0.09540583)
            + 0.01806152 * max(0.0, 79.18312 - Q.sd_mass)
            - 0.02176641 * max(0.0, 45.595 - Q.sd_mass)
            + 0.01832243 * max(0.0, 35.30013 - Q.sj3_pair_mass_max)
            - 0.007504977 * max(0.0, 71.79516 - Q.mass_top50)
            + 80.78979 * max(0.0, 0.004673423 - Q.lam1)
            - 0.005476328 * max(0.0, 935.8189 - Q.sum_pt_top40)
            - 0.009753303 * max(0.0, 1018.698 - Q.sum_pt_top40)
            + 0.006422017 * max(0.0, 933.1875 - Q.sum_pt_top30)
            - 7.906435 * max(0.0, 6.903423 - Q.log_sum_pt)
        ))
    )


def logits(Q):
    return [logit_g(Q), logit_q(Q), logit_W(Q), logit_Z(Q), logit_t(Q)]


def classify(pt, eta, phi):
    s = logits(quantities(pt, eta, phi))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
