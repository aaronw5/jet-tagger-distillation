"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned; all observables), as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

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


def neuron_0(Q):
    z = 1.228678
    if Q.mass < 74.25181:
        z += 0.03329021 * Q.mass - 3.363967
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.003551757 * Q.mass - 1.155832
    if 78.26182 <= Q.mass < 89.74183:
        z += -0.1122813 * Q.mass + 7.909476
    if 89.74183 <= Q.mass < 91.19:
        z += -0.1314669 * Q.mass + 9.63122
    if 91.19 <= Q.mass < 92.85979:
        z += -0.07138439 * Q.mass + 4.1523
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.2010212 * Q.mass + 16.19034
    if Q.mass >= 101.0497:
        z += -0.2343114 * Q.mass + 19.55431
    if Q.girth2_top20 < 0.005312783:
        z += 39.28246 * Q.girth2_top20 - 0.345451
    if 0.005312783 <= Q.girth2_top20 < 0.006374178:
        z += 128.8417 * Q.girth2_top20 - 0.8212598
    if Q.sum_pt < 1012.673:
        z += 0.009377984 * Q.sum_pt - 9.496831
    if Q.psi_0p3 >= 0.9956185:
        z += 49.82551 * Q.psi_0p3 - 49.6072
    if Q.mass_top30 < 80.4:
        z += -0.01198491 * Q.mass_top30 + 0.9635864
    if Q.lam1 < 0.005913555:
        z += 63.6503 * Q.lam1 - 0.3763995
    if Q.mass_over_sum_pt_sq < 0.006938798:
        z += -719.6524 * Q.mass_over_sum_pt_sq + 5.272094
    if 0.006938798 <= Q.mass_over_sum_pt_sq < 0.007873266:
        z += -298.1063 * Q.mass_over_sum_pt_sq + 2.34707
    if Q.tau1 < 0.0705748:
        z += 19.58872 * Q.tau1 - 1.38247
    if Q.e2_sq < 0.00616708:
        z += 395.3828 * Q.e2_sq - 2.438357
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += 3.5055 * Q.z_dr_0p2_0p4 - 0.3197916
    if Q.z_top50_slots < 0.9906378:
        z += 20.24792 * Q.z_top50_slots - 20.05835
    if Q.girth2_top30 < 0.006363916:
        z += 167.5541 * Q.girth2_top30 - 1.0663
    if Q.log_sum_pt < 7.017258:
        z += -2.028084 * Q.log_sum_pt + 14.23159
    if Q.sum_pt_top40 < 1069.671:
        z += 0.003657897 * Q.sum_pt_top40 - 3.912748
    if Q.sum_pt_top20 < 846.1934:
        z += -0.001987747 * Q.sum_pt_top20 + 1.682018
    if Q.mass_top40 < 80.89043:
        z += -0.01113031 * Q.mass_top40 + 0.9003356
    if Q.mass_top50 >= 71.79516:
        z += 0.01335146 * Q.mass_top50 - 0.9585704
    if Q.sum_pt_top50 < 1156.659:
        z += -0.003006878 * Q.sum_pt_top50 + 3.477934
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.04044547 * Q.n_dr_0p2_0p4 + 0.606682
    if Q.log_sum_pt < 6.98945 and Q.sum_pt_top50 > 959.0957:
        z += 0.04855343 * (6.98945 - Q.log_sum_pt) * (Q.sum_pt_top50 - 959.0957)
    if Q.sum_pt < 1012.673 and Q.C3 < 0.0223982:
        z += 0.08750739 * (1012.673 - Q.sum_pt) * (0.0223982 - Q.C3)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.3323458
    if Q.n_particles >= 38.0:
        z += 0.1148417 * Q.n_particles - 4.363985
    if Q.log_sum_pt < 6.893714:
        z += 7.741519 * Q.log_sum_pt - 55.269
    if 6.893714 <= Q.log_sum_pt < 6.910131:
        z += 35.33046 * Q.log_sum_pt - 245.4593
    if 6.910131 <= Q.log_sum_pt < 6.959294:
        z += 50.50831 * Q.log_sum_pt - 350.3402
    if 6.959294 <= Q.log_sum_pt < 6.98945:
        z += 33.29085 * Q.log_sum_pt - 230.5189
    if 6.98945 <= Q.log_sum_pt < 7.139296:
        z += 20.75572 * Q.log_sum_pt - 142.9052
    if Q.log_sum_pt >= 7.139296:
        z += 13.01421 * Q.log_sum_pt - 87.6362
    if Q.sum_pt_top50 >= 959.0957:
        z += -0.009297987 * Q.sum_pt_top50 + 8.91766
    if Q.psi_0p3 >= 0.9980008:
        z += -98.94735 * Q.psi_0p3 + 98.74954
    if Q.sum_pt_top2 < 689.25:
        z += -0.002148106 * Q.sum_pt_top2 + 1.480582
    if Q.mass_top20 < 47.88842:
        z += 0.03049833 * Q.mass_top20 - 1.460517
    if Q.sj3_mass1 < 32.50209:
        z += 0.008059208 * Q.sj3_mass1 - 0.2619411
    if Q.sum_pt_top40 < 1069.671:
        z += -0.005807989 * Q.sum_pt_top40 + 6.212639
    if Q.girth2_top15 < 0.003270031:
        z += -169.6487 * Q.girth2_top15 + 0.5547566
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.08100759 * Q.n_dr_0p2_0p4 - 0.5670531
    if Q.girth2_top3 < 0.0005522528:
        z += -1273.615 * Q.girth2_top3 + 0.7033576
    if Q.M3 < 0.03187688:
        z += 26.16099 * Q.M3 - 0.8339308
    if Q.sj2_mass1 < 30.26161:
        z += 0.01779958 * Q.sj2_mass1 - 0.5386438
    if Q.pt_9 < 31.35938:
        z += 0.04332009 * Q.pt_9 - 1.358491
    if Q.lam1 < 0.004673423:
        z += -134.0114 * Q.lam1 + 0.6262918
    if Q.mass < 120.6:
        z += 0.00956025 * Q.mass - 1.152966
    if Q.girth2_top30 < 0.02412652:
        z += -49.43327 * Q.girth2_top30 + 1.192653
    if Q.D3 < 0.1416054:
        z += -2.023623 * Q.D3 + 0.2865559
    if Q.z_dr_0_0p05 >= 0.878906:
        z += -4.403608 * Q.z_dr_0_0p05 + 3.870357
    if Q.mass_top30 < 42.41192:
        z += -0.01634918 * Q.mass_top30 + 0.6934002
    if Q.n_dr_0_0p05 < 12.0:
        z += -0.01548025 * Q.n_dr_0_0p05 + 0.185763
    if Q.soft1_pt < 1.521582:
        z += -0.5230061 * Q.soft1_pt + 0.7957967
    if Q.sum_pt < 1017.435:
        z += -0.01023091 * Q.sum_pt + 10.40928
    if Q.sum_pt_top30 >= 1191.938:
        z += 0.003192628 * Q.sum_pt_top30 - 3.805414
    if Q.z_top20_slots >= 0.8965411:
        z += 4.214125 * Q.z_top20_slots - 3.778136
    if Q.pt_entropy >= 2.07371:
        z += 0.2333272 * Q.pt_entropy - 0.4838528
    if Q.z_top30_slots > 0.9341838 and Q.max_pair_mass > 13.04793:
        z += 0.4506887 * (Q.z_top30_slots - 0.9341838) * (Q.max_pair_mass - 13.04793)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.1119555:
        z += 0.2905178 * (Q.n_particles - 38.0) * (0.1119555 - Q.dr_0)
    if Q.mass_top20 < 47.88842 and Q.n_real_top40 > 29.0:
        z += 0.003746863 * (47.88842 - Q.mass_top20) * (Q.n_real_top40 - 29.0)
    if Q.n_particles > 38.0 and Q.soft1_pt < 2.275391:
        z += -0.03858602 * (Q.n_particles - 38.0) * (2.275391 - Q.soft1_pt)
    if Q.z_top30_slots > 0.9341838 and Q.C2 < 0.07279889:
        z += 169.4878 * (Q.z_top30_slots - 0.9341838) * (0.07279889 - Q.C2)
    if Q.sj3_mass1 < 32.50209 and Q.sj3_mass2 < 18.68222:
        z += -0.00148257 * (32.50209 - Q.sj3_mass1) * (18.68222 - Q.sj3_mass2)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.1611545:
        z += 0.1266055 * (Q.n_particles - 38.0) * (0.1611545 - Q.dr_1)
    if Q.z_top30_slots > 0.9341838 and Q.ptdr0_3 > 7.407874:
        z += 0.7369499 * (Q.z_top30_slots - 0.9341838) * (Q.ptdr0_3 - 7.407874)
    if Q.pt_9 < 31.35938 and Q.dr1_12 < 0.3241858:
        z += 0.04339613 * (31.35938 - Q.pt_9) * (0.3241858 - Q.dr1_12)
    if Q.pt_9 < 31.35938 and Q.pair_mass_0_13 < 11.29505:
        z += 0.001609659 * (31.35938 - Q.pt_9) * (11.29505 - Q.pair_mass_0_13)
    if Q.sum_pt_top5 > 430.75 and Q.eta_0 < 0.02980347:
        z += -0.009909199 * (Q.sum_pt_top5 - 430.75) * (0.02980347 - Q.eta_0)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.3308957
    if 6.959294 <= Q.log_sum_pt < 7.062574:
        z += -12.56052 * Q.log_sum_pt + 87.41232
    if Q.log_sum_pt >= 7.062574:
        z += -16.82888 * Q.log_sum_pt + 117.5579
    if Q.sum_pt < 972.0419:
        z += 0.01820494 * Q.sum_pt - 18.46715
    if 972.0419 <= Q.sum_pt < 1017.435:
        z += 0.002673083 * Q.sum_pt - 3.369531
    if 1017.435 <= Q.sum_pt < 1115.723:
        z += 0.01515615 * Q.sum_pt - 16.07024
    if 1115.723 <= Q.sum_pt < 1260.541:
        z += 0.008992648 * Q.sum_pt - 9.193475
    if Q.sum_pt >= 1260.541:
        z += 0.006319564 * Q.sum_pt - 5.823945
    if Q.sum_pt_top50 < 988.4554:
        z += -0.004209084 * Q.sum_pt_top50 + 4.328438
    if 988.4554 <= Q.sum_pt_top50 < 1048.098:
        z += -0.005052013 * Q.sum_pt_top50 + 5.161635
    if 1048.098 <= Q.sum_pt_top50 < 1078.994:
        z += 0.001846613 * Q.sum_pt_top50 - 2.068802
    if 1078.994 <= Q.sum_pt_top50 < 1156.659:
        z += -0.0008429288 * Q.sum_pt_top50 + 0.8331975
    if Q.sum_pt_top50 >= 1156.659:
        z += 0.002715063 * Q.sum_pt_top50 - 3.282187
    if Q.mass < 91.19:
        z += 0.0355966 * Q.mass - 3.34362
    if 91.19 <= Q.mass < 92.85979:
        z += 0.05842994 * Q.mass - 5.425792
    if Q.mass_over_sum_pt < 0.09795415:
        z += -10.12298 * Q.mass_over_sum_pt + 0.9915877
    if Q.sum_pt_top40 < 984.7009:
        z += -0.00468143 * Q.sum_pt_top40 + 4.8746
    if 984.7009 <= Q.sum_pt_top40 < 1041.263:
        z += -0.005856556 * Q.sum_pt_top40 + 6.031747
    if 1041.263 <= Q.sum_pt_top40 < 1069.671:
        z += -0.001175126 * Q.sum_pt_top40 + 1.157148
    if Q.sum_pt_top40 >= 1069.671:
        z += 0.00318621 * Q.sum_pt_top40 - 3.508047
    if Q.sum_pt_top30 < 996.8867:
        z += 0.004172341 * Q.sum_pt_top30 - 4.159352
    if Q.girth2_top30 < 0.004763596:
        z += -43.89024 * Q.girth2_top30 + 0.06519224
    if 0.004763596 <= Q.girth2_top30 < 0.006363916:
        z += 89.90898 * Q.girth2_top30 - 0.5721732
    if Q.lam1 < 0.01174405:
        z += -62.50371 * Q.lam1 + 0.7340468
    if Q.mass_top50 < 92.16545:
        z += -0.01054376 * Q.mass_top50 + 0.97177
    if Q.log_sum_pt > 6.903423 and Q.girth2_top15 < 0.02146578:
        z += 687.5768 * (Q.log_sum_pt - 6.903423) * (0.02146578 - Q.girth2_top15)
    if Q.log_sum_pt > 6.903423 and Q.psi_0p3 > 0.9299135:
        z += -23.5794 * (Q.log_sum_pt - 6.903423) * (Q.psi_0p3 - 0.9299135)
    if Q.sum_pt_top50 > 988.4554 and Q.girth2_top15 < 0.02146578:
        z += -0.4424092 * (Q.sum_pt_top50 - 988.4554) * (0.02146578 - Q.girth2_top15)
    if Q.sum_pt < 972.0419 and Q.e4 < 5.8505e-08:
        z += 271938.5 * (972.0419 - Q.sum_pt) * (5.8505e-08 - Q.e4)
    if Q.sum_pt < 1260.541 and Q.max_dr < 0.397021:
        z += -0.006325273 * (1260.541 - Q.sum_pt) * (0.397021 - Q.max_dr)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.08734131:
        z += -0.3783806 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.08734131)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.09849501
    if Q.lam2 < 0.0006154841:
        z += -806.4848 * Q.lam2 + 0.4963785
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.184767 * Q.n_dr_0p2_0p4 + 0.923835
    if Q.n_particles < 46.0:
        z += -0.02504442 * Q.n_particles + 1.152043
    if Q.psi_0p2 >= 0.9985434:
        z += 280.4078 * Q.psi_0p2 - 279.9993
    if Q.tau21 < 0.3861957:
        z += 1.922232 * Q.tau21 - 0.7423577
    if Q.D2 < 2.410481:
        z += 0.1042589 * Q.D2 - 0.2513141
    if Q.mass_over_sum_pt < 0.08665515:
        z += -11.90596 * Q.mass_over_sum_pt + 1.031713
    if Q.mass_top50 < 79.21004:
        z += 0.005771511 * Q.mass_top50 - 0.4571616
    if Q.mass < 53.87362:
        z += 0.01044678 * Q.mass - 0.4761336
    if 53.87362 <= Q.mass < 86.4:
        z += -0.0008215299 * Q.mass + 0.1309309
    if 86.4 <= Q.mass < 92.85979:
        z += -0.009280592 * Q.mass + 0.8617939
    if Q.girth2 < 0.009614971:
        z += 133.8843 * Q.girth2 - 1.287294
    if Q.girth2_top40 < 0.008840538:
        z += -144.7654 * Q.girth2_top40 + 1.279804
    if Q.sj2_mass1 < 27.56535:
        z += -0.01680284 * Q.sj2_mass1 + 0.4631762
    if Q.mass_top40 < 80.4:
        z += 0.01025869 * Q.mass_top40 - 0.8247983
    if Q.tau4 < 0.01626937:
        z += -21.90357 * Q.tau4 + 0.3563572
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += -0.001116056 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_particles < 46.0 and Q.mass_top15 > 57.87349:
        z += -0.001252897 * (46.0 - Q.n_particles) * (Q.mass_top15 - 57.87349)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_dr_0p1_0p2 > 9.0:
        z += -0.0130078 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 9.0)
    if Q.n_particles < 46.0 and Q.sum_pt_top30 > 800.732:
        z += 4.924219e-05 * (46.0 - Q.n_particles) * (Q.sum_pt_top30 - 800.732)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_10 < 0.03447112:
        z += -4.009614 * (5.0 - Q.n_dr_0p2_0p4) * (0.03447112 - Q.z_10)
    if Q.D2 < 2.410481 and Q.psi_0p3 > 0.9985421:
        z += 97.15128 * (2.410481 - Q.D2) * (Q.psi_0p3 - 0.9985421)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.zdr_5 < 0.007099471:
        z += -18.20203 * (5.0 - Q.n_dr_0p2_0p4) * (0.007099471 - Q.zdr_5)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.psi_0p1 < 0.9538343:
        z += 0.5548369 * (5.0 - Q.n_dr_0p2_0p4) * (0.9538343 - Q.psi_0p1)
    if Q.tau21 < 0.3861957 and Q.lam1 > 0.007671243:
        z += -326.0059 * (0.3861957 - Q.tau21) * (Q.lam1 - 0.007671243)
    if Q.n_dr_0p2_0p4 < 8.0 and Q.z_top50_slots > 0.978741:
        z += 1.334769 * (8.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741)
    if Q.psi_0p2 > 0.9985434 and Q.abseta_4 < 0.08575439:
        z += -4059.072 * (Q.psi_0p2 - 0.9985434) * (0.08575439 - Q.abseta_4)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.9994559
    if Q.mass_top40 < 67.72643:
        z += -0.01379883 * Q.mass_top40 - 0.506561
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.01752089 * Q.mass_top40 - 2.627734
    if 83.32554 <= Q.mass_top40 < 163.2541:
        z += 0.0146105 * Q.mass_top40 - 2.385224
    if Q.mass_top30 < 60.43821:
        z += -0.02294151 * Q.mass_top30 + 0.7414642
    if 60.43821 <= Q.mass_top30 < 91.69753:
        z += 0.01613577 * Q.mass_top30 - 1.620297
    if 91.69753 <= Q.mass_top30 < 120.6:
        z += 0.004867626 * Q.mass_top30 - 0.5870357
    if Q.mass < 74.25181:
        z += 0.09464808 * Q.mass - 6.215995
    if 74.25181 <= Q.mass < 78.26182:
        z += 0.08430413 * Q.mass - 5.447938
    if 78.26182 <= Q.mass < 80.78464:
        z += 0.04800099 * Q.mass - 2.606788
    if 80.78464 <= Q.mass < 86.4:
        z += 0.03929767 * Q.mass - 1.903693
    if 86.4 <= Q.mass < 92.85979:
        z += -0.04217415 * Q.mass + 5.135472
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.08767741 * Q.mass + 9.360895
    if 101.0497 <= Q.mass < 120.6:
        z += -0.0232533 * Q.mass + 2.850857
    if 120.6 <= Q.mass < 143.7876:
        z += -0.002005798 * Q.mass + 0.2884089
    if Q.psi_0p3 >= 0.9973959:
        z += 176.9157 * Q.psi_0p3 - 176.455
    if Q.sj3_pair_mass_min >= 32.51366:
        z += 0.005725281 * Q.sj3_pair_mass_min - 0.1861498
    if Q.n_particles >= 22.0:
        z += -0.02142293 * Q.n_particles + 0.4713044
    if Q.girth2_top15 < 0.004855289:
        z += 177.857 * Q.girth2_top15 - 0.8635472
    if 0.00727763 <= Q.girth2_top15 < 0.01563836:
        z += -78.6805 * Q.girth2_top15 + 0.5726076
    if Q.girth2_top15 >= 0.01563836:
        z += -2.485077 * Q.girth2_top15 - 0.6189635
    if 0.009606007 <= Q.e2_sq < 0.01396296:
        z += 134.6129 * Q.e2_sq - 1.293093
    if Q.e2_sq >= 0.01396296:
        z += 194.8974 * Q.e2_sq - 2.134842
    if Q.mass_top15 < 57.87349:
        z += -0.008139694 * Q.mass_top15 + 0.2947892
    if 57.87349 <= Q.mass_top15 < 91.19:
        z += 0.00529117 * Q.mass_top15 - 0.4825018
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.03242072 * Q.n_dr_0p2_0p4 + 0.8429388
    if Q.girth >= 0.1207452:
        z += -20.22809 * Q.girth + 2.442445
    if Q.lam1 >= 0.008241985:
        z += -123.4678 * Q.lam1 + 1.017619
    if Q.psi_0p1 < 0.9909875:
        z += -0.5648683 * Q.psi_0p1 + 0.5597774
    if Q.lam2 >= 0.001776308:
        z += -60.85614 * Q.lam2 + 0.1080993
    if Q.psi_0p3 > 0.9973959 and Q.n_dr_0_0p05 < 13.0:
        z += -12.45749 * (Q.psi_0p3 - 0.9973959) * (13.0 - Q.n_dr_0_0p05)
    if Q.n_particles > 22.0 and Q.n_dr_0_0p05 > 10.0:
        z += 0.0004443633 * (Q.n_particles - 22.0) * (Q.n_dr_0_0p05 - 10.0)
    if Q.n_particles > 22.0 and Q.soft1_pt < 2.275391:
        z += 0.006175368 * (Q.n_particles - 22.0) * (2.275391 - Q.soft1_pt)
    if Q.n_dr_0p2_0p4 < 26.0 and Q.n_dr_0p1_0p2 > 11.0:
        z += -0.001032432 * (26.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 11.0)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.6833986
    z += -0.03663246 * Q.n_particles + 2.344477
    if 0.09046749 <= Q.mass_over_sum_pt < 0.1708801:
        z += -29.0871 * Q.mass_over_sum_pt + 2.631437
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 166.367 * Q.mass_over_sum_pt - 30.76778
    if 6.910131 <= Q.log_sum_pt < 6.920349:
        z += -17.63508 * Q.log_sum_pt + 121.8607
    if 6.920349 <= Q.log_sum_pt < 6.935549:
        z += -29.30814 * Q.log_sum_pt + 202.6424
    if 6.935549 <= Q.log_sum_pt < 6.98945:
        z += -24.10706 * Q.log_sum_pt + 166.57
    if Q.log_sum_pt >= 6.98945:
        z += -18.01253 * Q.log_sum_pt + 123.9726
    if 907.9372 <= Q.sum_pt < 986.0565:
        z += 0.005175716 * Q.sum_pt - 4.699225
    if Q.sum_pt >= 986.0565:
        z += -0.0054814 * Q.sum_pt + 5.809294
    if Q.girth2_top50 >= 0.01951641:
        z += -59.68106 * Q.girth2_top50 + 1.16476
    if Q.sum_pt_top50 >= 934.2416:
        z += 0.01866498 * Q.sum_pt_top50 - 17.4376
    if Q.n_pt_above_1 >= 28.0:
        z += 0.01145388 * Q.n_pt_above_1 - 0.3207086
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.02827257 * Q.n_dr_0p2_0p4 + 0.3109982
    if Q.sum_pt_top40 >= 1024.942:
        z += -0.004013931 * Q.sum_pt_top40 + 4.114048
    if Q.sum_pt_top30 >= 933.1875:
        z += 0.001172525 * Q.sum_pt_top30 - 1.094185
    if 0.2404747 <= Q.max_dr < 0.4357228:
        z += -2.224028 * Q.max_dr + 0.5348224
    if Q.max_dr >= 0.4357228:
        z += -0.3541552 * Q.max_dr - 0.2799237
    if Q.z_top30_slots >= 0.9048492:
        z += -7.40116 * Q.z_top30_slots + 6.696934
    if Q.mass_top40 < 120.6:
        z += 0.005553738 * Q.mass_top40 - 0.833141
    if 120.6 <= Q.mass_top40 < 150.0144:
        z += -0.009100745 * Q.mass_top40 + 0.9341896
    if Q.mass_top40 >= 150.0144:
        z += -0.01465448 * Q.mass_top40 + 1.767331
    if 74.25181 <= Q.mass < 91.19:
        z += 0.01994 * Q.mass - 1.480581
    if 91.19 <= Q.mass < 172.8:
        z += 0.006151759 * Q.mass - 0.2232314
    if Q.mass >= 172.8:
        z += 0.0348644 * Q.mass - 5.184776
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -510.6845 * Q.mass_over_sum_pt_sq + 14.91199
    if Q.tau21 < 0.5494307:
        z += 1.454749 * Q.tau21 - 0.799284
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.03687624 * Q.sd_mass - 2.568664
    if Q.sd_mass >= 86.4:
        z += -0.01166937 * Q.sd_mass + 1.625677
    if Q.mass_top50 >= 157.5448:
        z += -0.2087503 * Q.mass_top50 + 32.88753
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.002118901 * Q.sum_pt_top10 + 1.99959
    if Q.z_11 < 0.01375115:
        z += -161.3584 * Q.z_11 + 2.218863
    if Q.e2 < 0.03875945:
        z += -10.15916 * Q.e2 + 0.3937635
    if Q.n_dr_0p1_0p2 < 8.0:
        z += -0.04663374 * Q.n_dr_0p1_0p2 + 0.1103968
    if 8.0 <= Q.n_dr_0p1_0p2 < 21.0:
        z += 0.02020562 * Q.n_dr_0p1_0p2 - 0.424318
    if Q.mass_top10 >= 71.781:
        z += 0.007455275 * Q.mass_top10 - 0.5351471
    if Q.z_dr_0p2_0p4 < 0.1937447:
        z += 2.502079 * Q.z_dr_0p2_0p4 - 0.4847646
    if Q.pt_11 < 14.14062:
        z += 0.1477321 * Q.pt_11 - 2.089024
    if Q.D2 < 1.976207:
        z += -0.3503087 * Q.D2 + 0.6922825
    if Q.n_particles < 64.0 and Q.D2 < 2.178951:
        z += -0.01729337 * (64.0 - Q.n_particles) * (2.178951 - Q.D2)
    if Q.sum_pt > 907.9372 and Q.e4 < 5.8505e-08:
        z += 25726.1 * (Q.sum_pt - 907.9372) * (5.8505e-08 - Q.e4)
    if Q.psi_0p3 > 0.9973959 and Q.D2 < 3.345339:
        z += 28.39648 * (Q.psi_0p3 - 0.9973959) * (3.345339 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.170115
    if Q.mass_top50 < 71.79516:
        z += -0.01985372 * Q.mass_top50 + 1.969115
    if 71.79516 <= Q.mass_top50 < 172.8:
        z += -0.005383053 * Q.mass_top50 + 0.9301916
    if Q.mass < 86.4:
        z += 0.006515071 * Q.mass - 2.473096
    if 86.4 <= Q.mass < 92.85979:
        z += 0.04049037 * Q.mass - 5.408562
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.112517 * Q.mass - 12.09694
    if 101.0497 <= Q.mass < 120.6:
        z += 0.03719278 * Q.mass - 4.48545
    if Q.girth2_top15 < 0.002197765:
        z += 114.2108 * Q.girth2_top15 - 0.2510085
    if 29.00832 <= Q.sj3_pair_mass_min < 76.60223:
        z += -0.006836022 * Q.sj3_pair_mass_min + 0.1983015
    if Q.sj3_pair_mass_min >= 76.60223:
        z += -0.1286936 * Q.sj3_pair_mass_min + 9.532867
    if 0.02793599 <= Q.e2 < 0.05557149:
        z += -34.5741 * Q.e2 + 0.965862
    if Q.e2 >= 0.05557149:
        z += 44.25368 * Q.e2 - 3.414716
    if Q.sj3_mass1 >= 21.11128:
        z += -0.02471428 * Q.sj3_mass1 + 0.5217499
    if Q.log_sum_pt < 6.811175:
        z += 4.312245 * Q.log_sum_pt - 29.37146
    if Q.tau1 < 0.06310829:
        z += -21.73339 * Q.tau1 + 1.371557
    if Q.z_top40_slots < 0.9300465:
        z += -12.94304 * Q.z_top40_slots + 12.03763
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -31.79539 * Q.mass_over_sum_pt + 3.11449
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -37.97107 * Q.mass_over_sum_pt + 3.844616
    if Q.girth2_top30 < 0.00375223:
        z += -141.1444 * Q.girth2_top30 - 0.1421173
    if 0.00375223 <= Q.girth2_top30 < 0.008376291:
        z += 46.89314 * Q.girth2_top30 - 0.8476773
    if 0.008376291 <= Q.girth2_top30 < 0.01807679:
        z += 204.2951 * Q.girth2_top30 - 2.166122
    if 0.01807679 <= Q.girth2_top30 < 0.02809026:
        z += 157.402 * Q.girth2_top30 - 1.318445
    if Q.girth2_top30 >= 0.02809026:
        z += -97.75136 * Q.girth2_top30 + 5.848878
    if Q.girth2_top20 >= 0.008031209:
        z += 116.7522 * Q.girth2_top20 - 0.9376615
    if Q.lam1 < 0.003811746:
        z += -274.7502 * Q.lam1 + 1.047278
    if Q.girth2_top50 < 0.004573744:
        z += 318.9114 * Q.girth2_top50 - 1.458619
    if Q.lam2 < 0.003687605:
        z += -146.3964 * Q.lam2 + 0.539852
    if Q.mass_top15 < 91.19:
        z += -0.009229968 * Q.mass_top15 + 0.8416808
    if Q.n_dr_0_0p05 >= 9.0:
        z += -0.01535736 * Q.n_dr_0_0p05 + 0.1382163
    if Q.mass < 120.6 and Q.sum_pt < 1007.788:
        z += 9.2013e-05 * (120.6 - Q.mass) * (1007.788 - Q.sum_pt)
    if Q.sj3_pair_mass_min > 29.00832 and Q.psi_0p3 > 0.9896594:
        z += 1.980592 * (Q.sj3_pair_mass_min - 29.00832) * (Q.psi_0p3 - 0.9896594)
    if Q.mass_over_sum_pt > 0.1182259 and Q.C2_b2 > 0.02704832:
        z += 1161.606 * (Q.mass_over_sum_pt - 0.1182259) * (Q.C2_b2 - 0.02704832)
    if Q.e2 > 0.05557149 and Q.zdr_0 > 0.001369707:
        z += 1133.33 * (Q.e2 - 0.05557149) * (Q.zdr_0 - 0.001369707)
    if Q.n_dr_0p2_0p4 > 15.0 and Q.max_pair_mass > 33.3761:
        z += -0.003083477 * (Q.n_dr_0p2_0p4 - 15.0) * (Q.max_pair_mass - 33.3761)
    if Q.girth2_top30 > 0.008376291 and Q.D2_b2 > 1.67722:
        z += -44.33966 * (Q.girth2_top30 - 0.008376291) * (Q.D2_b2 - 1.67722)
    if Q.mass_over_sum_pt > 0.1182259 and Q.D2_b2 < 7.36624:
        z += -6.021715 * (Q.mass_over_sum_pt - 0.1182259) * (7.36624 - Q.D2_b2)
    if Q.girth2_top20 > 0.008031209 and Q.C2_b2 > 0.0008187529:
        z += -2864.26 * (Q.girth2_top20 - 0.008031209) * (Q.C2_b2 - 0.0008187529)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.2381245
    if Q.tau21_b2 < 0.2352054:
        z += -8.160562 * Q.tau21_b2 + 1.919409
    if Q.girth2 < 0.006403325:
        z += 22.27138 * Q.girth2 + 0.01373787
    if 0.006403325 <= Q.girth2 < 0.007877041:
        z += -7.916595 * Q.girth2 + 0.2070413
    if 0.007877041 <= Q.girth2 < 0.008190222:
        z += 320.2153 * Q.girth2 - 2.377667
    if 0.008190222 <= Q.girth2 < 0.009614971:
        z += -171.937 * Q.girth2 + 1.65317
    if Q.mass_over_sum_pt < 0.1182259:
        z += -18.48574 * Q.mass_over_sum_pt + 2.185494
    if Q.mass < 78.26182:
        z += 0.1007502 * Q.mass - 8.436133
    if 78.26182 <= Q.mass < 82.85409:
        z += 0.1687784 * Q.mass - 13.76014
    if 82.85409 <= Q.mass < 91.19:
        z += 0.006771417 * Q.mass - 0.3372005
    if 91.19 <= Q.mass < 92.85979:
        z += 0.04495284 * Q.mass - 3.818965
    if 92.85979 <= Q.mass < 101.0497:
        z += 0.05440436 * Q.mass - 4.696631
    if 101.0497 <= Q.mass < 120.6:
        z += -0.04096687 * Q.mass + 4.940604
    if Q.n_dr_0p2_0p4 < 6.0:
        z += -0.07266804 * Q.n_dr_0p2_0p4 + 0.4360082
    if 0.9973959 <= Q.psi_0p3 < 0.9980008:
        z += 292.4853 * Q.psi_0p3 - 291.7236
    if Q.psi_0p3 >= 0.9980008:
        z += -512.4105 * Q.psi_0p3 + 511.563
    if Q.tau21 < 0.3861957:
        z += 1.301918 * Q.tau21 - 0.502795
    if Q.lam2 < 0.000404306:
        z += -196.5933 * Q.lam2 + 0.07948385
    if Q.e2_sq < 0.00363788:
        z += 1585.736 * Q.e2_sq - 5.768718
    if Q.tau1 < 0.09591084:
        z += 9.014149 * Q.tau1 - 1.181313
    if 0.09591084 <= Q.tau1 < 0.1072713:
        z += 27.88261 * Q.tau1 - 2.991003
    if Q.lam1 < 0.006189818:
        z += 22.20303 * Q.lam1 - 0.5074681
    if 0.006189818 <= Q.lam1 < 0.008241985:
        z += 180.3144 * Q.lam1 - 1.486149
    if Q.sd_mass < 69.65633:
        z += -0.005686026 * Q.sd_mass - 0.08381807
    if 69.65633 <= Q.sd_mass < 86.4:
        z += 0.02866072 * Q.sd_mass - 2.476286
    if Q.sd_mass >= 98.05743:
        z += -0.09463646 * Q.sd_mass + 9.279809
    if 73.35236 <= Q.mass_top20 < 85.79457:
        z += -0.01869258 * Q.mass_top20 + 1.371145
    if Q.mass_top20 >= 85.79457:
        z += 0.01627266 * Q.mass_top20 - 1.628683
    if Q.tau21_b2 < 0.2352054 and Q.mass_over_sum_pt_sq < 0.007873266:
        z += -1169.139 * (0.2352054 - Q.tau21_b2) * (0.007873266 - Q.mass_over_sum_pt_sq)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9638082:
        z += 10.29801 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9777125:
        z += 8.300748 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 < 0.9985421:
        z += 2.242106 * (82.85409 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9777125:
        z += -16.57704 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9777125:
        z += 4.495176 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9777125)
    if Q.mass < 91.19 and Q.psi_0p3 > 0.9980008:
        z += -91.94118 * (91.19 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 82.85409 and Q.psi_0p3 > 0.9980008:
        z += 4.132055 * (82.85409 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.mass < 101.0497 and Q.psi_0p3 > 0.9980008:
        z += 77.00884 * (101.0497 - Q.mass) * (Q.psi_0p3 - 0.9980008)
    if Q.tau21_b2 < 0.2352054 and Q.sum_pt < 1260.541:
        z += -0.01354448 * (0.2352054 - Q.tau21_b2) * (1260.541 - Q.sum_pt)
    if Q.tau21_b2 < 0.2352054 and Q.z_2 < 0.1063277:
        z += -52.52341 * (0.2352054 - Q.tau21_b2) * (0.1063277 - Q.z_2)
    if Q.mass < 92.85979 and Q.psi_0p3 > 0.9638082:
        z += -8.543108 * (92.85979 - Q.mass) * (Q.psi_0p3 - 0.9638082)
    if Q.tau21_b2 < 0.2352054 and Q.orientation_deg > -9.840088:
        z += -0.02170227 * (0.2352054 - Q.tau21_b2) * (Q.orientation_deg - -9.840088)
    if Q.mass < 92.85979 and Q.z_dr_0_0p05 < 0.4947602:
        z += -0.1300488 * (92.85979 - Q.mass) * (0.4947602 - Q.z_dr_0_0p05)
    return max(0.0, z)


def neuron_8(Q):
    z = 2.066925
    if 0.07696632 <= Q.mass_over_sum_pt < 0.1182259:
        z += -18.60172 * Q.mass_over_sum_pt + 1.431706
    if Q.mass_over_sum_pt >= 0.1182259:
        z += -41.73238 * Q.mass_over_sum_pt + 4.16635
    if Q.sum_pt_top40 < 1001.523:
        z += 0.005160789 * Q.sum_pt_top40 - 5.399036
    if 1001.523 <= Q.sum_pt_top40 < 1024.942:
        z += 0.009837468 * Q.sum_pt_top40 - 10.08284
    if Q.girth2_top40 < 0.005196966:
        z += -193.494 * Q.girth2_top40 + 1.710591
    if 0.005196966 <= Q.girth2_top40 < 0.008840538:
        z += -143.2171 * Q.girth2_top40 + 1.449304
    if Q.girth2_top40 >= 0.008840538:
        z += 50.27689 * Q.girth2_top40 - 0.2612873
    if Q.n_dr_0p2_0p4 < 3.0:
        z += -0.01057173 * Q.n_dr_0p2_0p4 - 0.08647298
    if 3.0 <= Q.n_dr_0p2_0p4 < 9.0:
        z += 0.02241808 * Q.n_dr_0p2_0p4 - 0.1854424
    if 9.0 <= Q.n_dr_0p2_0p4 < 15.0:
        z += 0.06325958 * Q.n_dr_0p2_0p4 - 0.5530159
    if Q.n_dr_0p2_0p4 >= 15.0:
        z += 0.03298981 * Q.n_dr_0p2_0p4 - 0.09896943
    if 64.48544 <= Q.mass < 74.25181:
        z += 0.02477471 * Q.mass - 1.597608
    if 74.25181 <= Q.mass < 87.36377:
        z += 0.06717381 * Q.mass - 4.745818
    if 87.36377 <= Q.mass < 101.0497:
        z += 0.1074093 * Q.mass - 8.260946
    if 101.0497 <= Q.mass < 125.1:
        z += 0.04210583 * Q.mass - 1.662044
    if Q.mass >= 125.1:
        z += 0.007803926 * Q.mass + 2.629123
    if Q.sum_pt < 1017.435:
        z += -0.03481234 * Q.sum_pt + 35.60065
    if 1017.435 <= Q.sum_pt < 1028.184:
        z += -0.01687216 * Q.sum_pt + 17.34768
    if 0.006043209 <= Q.girth2_top20 < 0.008031209:
        z += -145.493 * Q.girth2_top20 + 0.8792448
    if Q.girth2_top20 >= 0.008031209:
        z += 92.87479 * Q.girth2_top20 - 1.035137
    if Q.log_sum_pt < 6.930088:
        z += 14.98268 * Q.log_sum_pt - 103.8313
    if Q.girth2_top30 < 0.007463985:
        z += -151.623 * Q.girth2_top30 + 1.131712
    if Q.width < 0.009614971:
        z += 432.9856 * Q.width - 4.163144
    if Q.girth2 < 0.007877041:
        z += -25.12502 * Q.girth2 + 0.1979108
    if Q.sj2_dr >= 0.2232169:
        z += 5.86249 * Q.sj2_dr - 1.308607
    if Q.z_dr_0p1_0p2 >= 0.3340477:
        z += 1.347847 * Q.z_dr_0p1_0p2 - 0.4502453
    if Q.n_for_90pct < 7.0:
        z += 0.05103926 * Q.n_for_90pct - 1.990531
    if 7.0 <= Q.n_for_90pct < 39.0:
        z += -0.01462027 * Q.n_for_90pct - 1.530915
    if Q.n_for_90pct >= 39.0:
        z += -0.06565953 * Q.n_for_90pct + 0.4596167
    if Q.e2 >= 0.04755309:
        z += 49.8174 * Q.e2 - 2.368971
    if Q.lam1 < 0.007671243:
        z += -100.3115 * Q.lam1 + 0.7695141
    if Q.C2 >= 0.06655881:
        z += 7.418645 * Q.C2 - 0.4937761
    if 5.13841e-05 <= Q.e3 < 0.0003372339:
        z += -2876.53 * Q.e3 + 0.1478079
    if Q.e3 >= 0.0003372339:
        z += -1481.412 * Q.e3 - 0.3226733
    if 82.04491 <= Q.mass_top50 < 117.0487:
        z += -0.03785952 * Q.mass_top50 + 3.106181
    if Q.mass_top50 >= 117.0487:
        z += -0.009864328 * Q.mass_top50 - 0.170621
    if Q.mass_top20 >= 119.2969:
        z += -0.01009848 * Q.mass_top20 + 1.204717
    if Q.n_dr_0p1_0p2 >= 19.0:
        z += 0.02069189 * Q.n_dr_0p1_0p2 - 0.3931459
    if Q.z_top15_slots < 0.8316924:
        z += -1.308164 * Q.z_top15_slots + 1.08799
    if Q.psi_0p1 < 0.3628388:
        z += 1.925304 * Q.psi_0p1 - 0.6985751
    if Q.z_top50_slots >= 0.9704436:
        z += -19.65455 * Q.z_top50_slots + 19.07363
    if Q.girth2_top15 < 0.02146578:
        z += 12.77399 * Q.girth2_top15 - 0.2742036
    if Q.mass_over_sum_pt > 0.07696632 and Q.sum_pt < 1115.723:
        z += 0.2598764 * (Q.mass_over_sum_pt - 0.07696632) * (1115.723 - Q.sum_pt)
    if Q.girth2_top40 > 0.005196966 and Q.log_sum_pt < 7.017258:
        z += -594.6978 * (Q.girth2_top40 - 0.005196966) * (7.017258 - Q.log_sum_pt)
    if Q.sum_pt < 1017.435 and Q.n_for_90pct < 13.0:
        z += -0.0006934448 * (1017.435 - Q.sum_pt) * (13.0 - Q.n_for_90pct)
    if Q.mass > 64.48544 and Q.zdr_0 > 0.0008718296:
        z += -0.1541735 * (Q.mass - 64.48544) * (Q.zdr_0 - 0.0008718296)
    if Q.girth2_top30 < 0.007463985 and Q.sj2_dr > 0.1512157:
        z += -694.5983 * (0.007463985 - Q.girth2_top30) * (Q.sj2_dr - 0.1512157)
    if Q.tau2 < 0.0795038 and Q.sj3_mass1 > 13.38202:
        z += 0.4102108 * (0.0795038 - Q.tau2) * (Q.sj3_mass1 - 13.38202)
    if Q.sum_pt < 1017.435 and Q.dr_max_012 > 0.1828389:
        z += -0.2181557 * (1017.435 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    if Q.sum_pt < 1028.184 and Q.dr_max_012 > 0.1828389:
        z += 0.1935704 * (1028.184 - Q.sum_pt) * (Q.dr_max_012 - 0.1828389)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.9585951
    if Q.mass_top40 < 80.89043:
        z += -0.005420735 * Q.mass_top40 - 1.560894
    if 80.89043 <= Q.mass_top40 < 163.2541:
        z += 0.02427501 * Q.mass_top40 - 3.962995
    if Q.sum_pt_top40 < 956.2133:
        z += -0.003029648 * Q.sum_pt_top40 + 2.89699
    if Q.girth2_top40 < 0.00217213:
        z += 85.15073 * Q.girth2_top40 + 0.6173162
    if 0.00217213 <= Q.girth2_top40 < 0.006259772:
        z += -196.2683 * Q.girth2_top40 + 1.228595
    if Q.mass < 64.48544:
        z += -0.01067787 * Q.mass + 2.242886
    if 64.48544 <= Q.mass < 82.85409:
        z += -0.07740914 * Q.mass + 6.546082
    if 82.85409 <= Q.mass < 92.85979:
        z += -0.02257612 * Q.mass + 2.002942
    if 92.85979 <= Q.mass < 143.7876:
        z += 0.04357522 * Q.mass - 4.139858
    if 143.7876 <= Q.mass < 160.8:
        z += -0.05316338 * Q.mass + 9.769955
    if 160.8 <= Q.mass < 162.8363:
        z += -0.2093807 * Q.mass + 34.8897
    if 162.8363 <= Q.mass < 172.8:
        z += -0.07978109 * Q.mass + 13.78617
    if 0.09749958 <= Q.girth < 0.1402186:
        z += 21.56199 * Q.girth - 2.102285
    if Q.girth >= 0.1402186:
        z += -68.75445 * Q.girth + 10.56176
    if Q.psi_0p3 >= 0.9943058:
        z += 54.99904 * Q.psi_0p3 - 54.68587
    if Q.lam1 < 0.004673423:
        z += -45.42692 * Q.lam1 + 0.2122992
    if Q.lam1 >= 0.01174405:
        z += 58.80979 * Q.lam1 - 0.6906651
    if Q.girth2_top30 >= 0.0008564881:
        z += -23.01644 * Q.girth2_top30 + 0.01971331
    if Q.mass_top50 < 92.16545:
        z += -0.009377848 * Q.mass_top50 + 3.229652
    if 92.16545 <= Q.mass_top50 < 138.8977:
        z += -0.05061472 * Q.mass_top50 + 7.030267
    if Q.mass_top30 < 73.33139:
        z += -0.01076456 * Q.mass_top30 + 0.78938
    if Q.LHA >= 0.404204:
        z += 46.68351 * Q.LHA - 18.86966
    if Q.z_top40_slots >= 0.9574183:
        z += -7.95405 * Q.z_top40_slots + 7.615353
    if Q.sum_pt < 986.0565:
        z += -0.02268123 * Q.sum_pt + 22.36497
    if Q.log_sum_pt < 6.903423:
        z += 14.45285 * Q.log_sum_pt - 99.77412
    if Q.sum_pt_top40 < 956.2133 and Q.soft4_pt > 1.789258:
        z += 0.002660812 * (956.2133 - Q.sum_pt_top40) * (Q.soft4_pt - 1.789258)
    if Q.mass_top40 < 80.89043 and Q.sum_pt < 1034.834:
        z += -0.0002300043 * (80.89043 - Q.mass_top40) * (1034.834 - Q.sum_pt)
    if Q.mass_top40 < 80.89043 and Q.sum_pt_top40 < 906.6023:
        z += 0.0002574458 * (80.89043 - Q.mass_top40) * (906.6023 - Q.sum_pt_top40)
    if Q.sum_pt_top40 < 956.2133 and Q.soft3_pt > 2.873047:
        z += -0.006547837 * (956.2133 - Q.sum_pt_top40) * (Q.soft3_pt - 2.873047)
    if Q.mass_top40 < 125.1 and Q.n_dr_0p2_0p4 > 9.0:
        z += 0.0009842942 * (125.1 - Q.mass_top40) * (Q.n_dr_0p2_0p4 - 9.0)
    return max(0.0, z)


def neuron_10(Q):
    z = -0.0267736
    if Q.girth < 0.1207452:
        z += 5.715259 * Q.girth - 0.69009
    if Q.mass < 64.48544:
        z += -0.04800972 * Q.mass + 3.811434
    if 64.48544 <= Q.mass < 78.26182:
        z += -0.08259095 * Q.mass + 6.04142
    if 78.26182 <= Q.mass < 86.4:
        z += -0.02050738 * Q.mass + 1.182647
    if 86.4 <= Q.mass < 89.74183:
        z += -0.001586057 * Q.mass - 0.4521552
    if 89.74183 <= Q.mass < 101.0497:
        z += 0.06586624 * Q.mass - 6.505448
    if 101.0497 <= Q.mass < 143.7876:
        z += 0.02750234 * Q.mass - 2.628787
    if 143.7876 <= Q.mass < 162.8363:
        z += -0.07474255 * Q.mass + 12.07276
    if Q.mass >= 162.8363:
        z += -0.1264374 * Q.mass + 20.49055
    if Q.girth2_top30 < 0.005402331:
        z += 248.0863 * Q.girth2_top30 - 1.340244
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.0574559 * Q.mass_top50 - 7.859105
    if 160.8 <= Q.mass_top50 < 172.8:
        z += 0.07833565 * Q.mass_top50 - 11.21657
    if Q.mass_top50 >= 172.8:
        z += 0.04462313 * Q.mass_top50 - 5.391046
    if Q.sj2_mass1 < 65.20727:
        z += 0.007794864 * Q.sj2_mass1 - 0.5082818
    if Q.D2 < 2.975532:
        z += -0.3495594 * Q.D2 + 1.040125
    if Q.z_top2_slots < 0.5760704:
        z += -1.800257 * Q.z_top2_slots + 1.037075
    if Q.mass_top5 < 22.18342:
        z += 0.005213073 * Q.mass_top5 - 0.3096971
    if 22.18342 <= Q.mass_top5 < 59.40777:
        z += 0.01631209 * Q.mass_top5 - 0.5559112
    if Q.mass_top5 >= 59.40777:
        z += 0.01109902 * Q.mass_top5 - 0.2462141
    if Q.sum_pt_top50 < 959.0957:
        z += 0.009389672 * Q.sum_pt_top50 - 9.005595
    if Q.girth2 < 0.002575211:
        z += 576.1586 * Q.girth2 - 1.48373
    if Q.dr_0 < 0.06413297:
        z += -7.297847 * Q.dr_0 + 0.4680326
    if Q.z_dr_0_0p05 >= 0.7674734:
        z += -3.460823 * Q.z_dr_0_0p05 + 2.65609
    if Q.e2 < 0.01256572:
        z += 30.45374 * Q.e2 - 1.327363
    if 0.01256572 <= Q.e2 < 0.03263075:
        z += 69.81533 * Q.e2 - 1.82197
    if 0.03263075 <= Q.e2 < 0.04358622:
        z += 59.95707 * Q.e2 - 1.500288
    if Q.e2 >= 0.04358622:
        z += 29.50333 * Q.e2 - 0.1729244
    if Q.sum_pt_top10 >= 943.6922:
        z += -0.002926603 * Q.sum_pt_top10 + 2.761813
    if Q.mass_over_sum_pt >= 0.1606361:
        z += -52.75721 * Q.mass_over_sum_pt + 8.474712
    if Q.e3 < 0.0001086251:
        z += -2352.6 * Q.e3 + 0.2555513
    if Q.zdr_1 < 0.008824206:
        z += -22.66657 * Q.zdr_1 + 0.2000145
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.05728831 * Q.n_dr_0p2_0p4 - 0.6301714
    if Q.sum_pt_top15 < 935.1043:
        z += -0.002535827 * Q.sum_pt_top15 + 2.371263
    if Q.z_dr_0p2_0p4 < 0.09122568:
        z += -3.521078 * Q.z_dr_0p2_0p4 + 0.3212127
    if Q.mass_top15 < 72.18744:
        z += 0.002957831 * Q.mass_top15 - 0.2135183
    if Q.girth2_top20 < 0.01655983:
        z += -73.34568 * Q.girth2_top20 + 1.214592
    if Q.mass_top30 >= 78.53034:
        z += -0.01374664 * Q.mass_top30 + 1.079528
    if Q.girth2_top30 < 0.005402331 and Q.sum_pt_top5 < 631.275:
        z += 0.4566084 * (0.005402331 - Q.girth2_top30) * (631.275 - Q.sum_pt_top5)
    if Q.mass_top50 > 160.8 and Q.soft4_z > 0.001721109:
        z += -17.68862 * (Q.mass_top50 - 160.8) * (Q.soft4_z - 0.001721109)
    if Q.D2 < 2.975532 and Q.sj2_dr > 0.2070855:
        z += -3.062242 * (2.975532 - Q.D2) * (Q.sj2_dr - 0.2070855)
    if Q.girth2_top30 < 0.005402331 and Q.psi_0p3 > 0.9985421:
        z += 184716.1 * (0.005402331 - Q.girth2_top30) * (Q.psi_0p3 - 0.9985421)
    if Q.girth2_top10 < 0.01976735 and Q.psi_0p3 > 0.9985421:
        z += -22809.65 * (0.01976735 - Q.girth2_top10) * (Q.psi_0p3 - 0.9985421)
    if Q.girth2_top10 < 0.01976735 and Q.max_dr < 0.4357228:
        z += -111.5653 * (0.01976735 - Q.girth2_top10) * (0.4357228 - Q.max_dr)
    if Q.mass_top5 < 59.40777 and Q.z_dr_0p05_0p1 < 0.8509215:
        z += 0.006737216 * (59.40777 - Q.mass_top5) * (0.8509215 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.1606361 and Q.z_3 < 0.09696199:
        z += 530.6533 * (Q.mass_over_sum_pt - 0.1606361) * (0.09696199 - Q.z_3)
    if Q.mass > 162.8363 and Q.soft5_z > 0.001434897:
        z += -11.72716 * (Q.mass - 162.8363) * (Q.soft5_z - 0.001434897)
    if Q.psi_0p3 > 0.9985421 and Q.soft5_z > 0.001434897:
        z += 77929.37 * (Q.psi_0p3 - 0.9985421) * (Q.soft5_z - 0.001434897)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.486065
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.07880341 * Q.n_dr_0p2_0p4 + 0.7880341
    if 0.9313699 <= Q.psi_0p2 < 0.9935324:
        z += -7.062757 * Q.psi_0p2 + 6.578039
    if Q.psi_0p2 >= 0.9935324:
        z += 34.90255 * Q.psi_0p2 - 35.11586
    if Q.mass < 80.4:
        z += -0.03168365 * Q.mass + 4.027819
    if 80.4 <= Q.mass < 92.85979:
        z += -0.1025982 * Q.mass + 9.72935
    if 92.85979 <= Q.mass < 101.0497:
        z += -0.02467683 * Q.mass + 2.493586
    if Q.girth2_top10 < 0.00406126:
        z += -108.5661 * Q.girth2_top10 + 0.1691852
    if 0.00406126 <= Q.girth2_top10 < 0.00625621:
        z += 123.7977 * Q.girth2_top10 - 0.7745044
    if Q.e2 < 0.02515919:
        z += -109.1312 * Q.e2 + 2.812959
    if 0.02515919 <= Q.e2 < 0.03029714:
        z += -51.27756 * Q.e2 + 1.35741
    if 0.03029714 <= Q.e2 < 0.03875945:
        z += 23.17973 * Q.e2 - 0.8984334
    if Q.girth2_top30 < 0.006929741:
        z += 158.5417 * Q.girth2_top30 - 1.098653
    if Q.mass_top30 < 60.43821:
        z += -0.01903503 * Q.mass_top30 + 1.150443
    if Q.mass_top40 < 67.72643:
        z += 0.01048477 * Q.mass_top40 - 0.9575645
    if 67.72643 <= Q.mass_top40 < 83.32554:
        z += 0.01586427 * Q.mass_top40 - 1.321899
    if Q.girth < 0.07374472:
        z += 11.70395 * Q.girth - 0.7400588
    if 0.07374472 <= Q.girth < 0.076787:
        z += 18.582 * Q.girth - 1.247279
    if 0.076787 <= Q.girth < 0.08589404:
        z += -19.71852 * Q.girth + 1.693703
    if Q.e2_sq < 0.00818374:
        z += -210.2211 * Q.e2_sq + 1.720395
    if Q.mass_over_sum_pt < 0.06030419:
        z += 6.533153 * Q.mass_over_sum_pt - 0.3939765
    if Q.lam1 < 0.006716737:
        z += 236.4662 * Q.lam1 - 1.588281
    if Q.e3 < 3.793233e-05:
        z += 17119.6 * Q.e3 - 0.6493865
    if Q.mass_top50 < 80.35535:
        z += 0.02496561 * Q.mass_top50 - 2.006121
    if Q.psi_0p3 >= 0.9896594:
        z += 61.7223 * Q.psi_0p3 - 61.08405
    if Q.C2_b2 >= 0.01330402:
        z += -6.09833 * Q.C2_b2 + 0.08113232
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_particles > 34.0:
        z += -0.002318625 * (10.0 - Q.n_dr_0p2_0p4) * (Q.n_particles - 34.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208:
        z += -33.52185 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.003232239 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.0497 and Q.sum_pt_top10 < 891.875:
        z += -3.654132e-05 * (101.0497 - Q.mass) * (891.875 - Q.sum_pt_top10)
    if Q.z_top40_slots > 0.996191 and Q.C2_b2 < 0.006580753:
        z += 11906.21 * (Q.z_top40_slots - 0.996191) * (0.006580753 - Q.C2_b2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.phi_0 < 0.0402832:
        z += 0.3503997 * (10.0 - Q.n_dr_0p2_0p4) * (0.0402832 - Q.phi_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.1367654
    if Q.mass < 53.87362:
        z += -0.03420769 * Q.mass + 4.375127
    if 53.87362 <= Q.mass < 62.55:
        z += -0.05987806 * Q.mass + 5.758083
    if 62.55 <= Q.mass < 74.25181:
        z += -0.1223492 * Q.mass + 9.665651
    if 74.25181 <= Q.mass < 82.85409:
        z += -0.07169201 * Q.mass + 5.904265
    if 82.85409 <= Q.mass < 86.4:
        z += 0.01007103 * Q.mass - 0.870137
    if Q.sd_mass >= 125.1:
        z += 0.03305676 * Q.sd_mass - 4.1354
    if Q.mass_top40 < 74.78616:
        z += 0.02493137 * Q.mass_top40 - 1.982127
    if 74.78616 <= Q.mass_top40 < 89.6788:
        z += 0.007896893 * Q.mass_top40 - 0.7081838
    if Q.sj3_pair_mass_max >= 120.6:
        z += -0.02023311 * Q.sj3_pair_mass_max + 2.440113
    if Q.mass_over_sum_pt < 0.05077291:
        z += 14.24586 * Q.mass_over_sum_pt - 1.074301
    if 0.05077291 <= Q.mass_over_sum_pt < 0.06895248:
        z += 19.30728 * Q.mass_over_sum_pt - 1.331284
    if Q.lam1 < 0.002752094:
        z += 92.36683 * Q.lam1 - 0.2542022
    if Q.girth2_top15 < 0.006142802:
        z += 113.8474 * Q.girth2_top15 - 0.6993418
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.06472584:
        z += -0.1930504 * (86.4 - Q.mass) * (0.06472584 - Q.z_dr_0p1_0p2)
    if Q.mass_top40 < 89.6788 and Q.n_dr_0p05_0p1 > 1.0:
        z += 0.0007088699 * (89.6788 - Q.mass_top40) * (Q.n_dr_0p05_0p1 - 1.0)
    if Q.mass < 86.4 and Q.psi_0p3 < 0.9985421:
        z += -3.712162 * (86.4 - Q.mass) * (0.9985421 - Q.psi_0p3)
    if Q.sj3_pair_mass_max > 120.6 and Q.sj3_pair_mass_min < 76.60223:
        z += 0.0004162666 * (Q.sj3_pair_mass_max - 120.6) * (76.60223 - Q.sj3_pair_mass_min)
    if Q.sj3_pair_mass_max > 120.6 and Q.z_dr_0p1_0p2 > 0.2864926:
        z += 0.02589869 * (Q.sj3_pair_mass_max - 120.6) * (Q.z_dr_0p1_0p2 - 0.2864926)
    if Q.mass < 86.4 and Q.lam2 < 0.003687605:
        z += 11.77132 * (86.4 - Q.mass) * (0.003687605 - Q.lam2)
    if Q.girth2_top10 < 0.0007431905 and Q.sum_pt_top40 < 1069.671:
        z += 4.322854 * (0.0007431905 - Q.girth2_top10) * (1069.671 - Q.sum_pt_top40)
    if Q.mass < 86.4 and Q.z_top50_slots < 0.985099:
        z += -1.469643 * (86.4 - Q.mass) * (0.985099 - Q.z_top50_slots)
    if Q.log_sum_pt < 6.856375 and Q.lam2 < 0.002396991:
        z += -5619.382 * (6.856375 - Q.log_sum_pt) * (0.002396991 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.212836
    if Q.sum_pt < 1007.788:
        z += 0.01841898 * Q.sum_pt - 18.90311
    if 1007.788 <= Q.sum_pt < 1085.125:
        z += 0.004405036 * Q.sum_pt - 4.780014
    if 74.25181 <= Q.mass < 136.785:
        z += 0.0195719 * Q.mass - 1.453249
    if 136.785 <= Q.mass < 143.7876:
        z += -0.01707801 * Q.mass + 3.559909
    if 143.7876 <= Q.mass < 160.8:
        z += -0.1347112 * Q.mass + 20.47411
    if 160.8 <= Q.mass < 162.8363:
        z += -0.2191845 * Q.mass + 34.0574
    if 162.8363 <= Q.mass < 172.8:
        z += -0.1537282 * Q.mass + 23.39875
    if Q.mass >= 172.8:
        z += -0.1154534 * Q.mass + 16.78486
    if Q.n_for_90pct < 16.0:
        z += 0.06930863 * Q.n_for_90pct - 1.108938
    if Q.mass_over_sum_pt >= 0.1708801:
        z += 262.2861 * Q.mass_over_sum_pt - 44.81948
    if Q.log_sum_pt < 6.811175:
        z += -4.862247 * Q.log_sum_pt + 31.89507
    if 6.811175 <= Q.log_sum_pt < 6.910131:
        z += 12.35443 * Q.log_sum_pt - 85.37076
    if Q.sum_pt_top40 < 1007.44:
        z += -0.008725866 * Q.sum_pt_top40 + 8.902779
    if 1007.44 <= Q.sum_pt_top40 < 1053.047:
        z += -0.002455553 * Q.sum_pt_top40 + 2.585813
    if 92.16545 <= Q.mass_top50 < 136.785:
        z += -0.03045358 * Q.mass_top50 + 2.806768
    if Q.mass_top50 >= 136.785:
        z += 0.05586398 * Q.mass_top50 - 9.000179
    if Q.girth2_top50 < 0.007820315:
        z += -133.7503 * Q.girth2_top50 + 1.875652
    if 0.007820315 <= Q.girth2_top50 < 0.02550569:
        z += -46.91348 * Q.girth2_top50 + 1.196561
    if Q.mass_over_sum_pt_sq < 0.02580396:
        z += 35.67838 * Q.mass_over_sum_pt_sq - 0.9206436
    if Q.mass_over_sum_pt_sq >= 0.02920002:
        z += -587.2654 * Q.mass_over_sum_pt_sq + 17.14816
    if Q.tau1 >= 0.1751567:
        z += -3.231438 * Q.tau1 + 0.5660079
    z += 0.006377498 * Q.n_particles
    if Q.mass_top40 < 160.8:
        z += 0.01245685 * Q.mass_top40 - 2.003061
    if Q.sum_pt_top50 < 997.0189:
        z += -0.00534158 * Q.sum_pt_top50 + 5.325656
    if Q.sum_pt_top30 < 966.0633:
        z += 0.002939385 * Q.sum_pt_top30 - 2.839632
    if Q.mass_top5 >= 33.71058:
        z += 0.01444172 * Q.mass_top5 - 0.4868389
    if Q.pt_11 < 18.32812:
        z += -0.01724623 * Q.pt_11 + 0.3160911
    if Q.mass_top10 < 76.9886:
        z += -0.005103648 * Q.mass_top10 + 0.3929227
    if Q.sum_pt_top20 >= 956.5062:
        z += 0.0008927598 * Q.sum_pt_top20 - 0.8539303
    if Q.sum_pt < 1085.125 and Q.sum_pt_top40 < 858.8262:
        z += -3.308651e-05 * (1085.125 - Q.sum_pt) * (858.8262 - Q.sum_pt_top40)
    if Q.sum_pt < 1085.125 and Q.tau21_b2 < 0.8310045:
        z += -0.005116102 * (1085.125 - Q.sum_pt) * (0.8310045 - Q.tau21_b2)
    if Q.n_for_90pct < 16.0 and Q.D2 < 3.814159:
        z += 0.01299926 * (16.0 - Q.n_for_90pct) * (3.814159 - Q.D2)
    if Q.sum_pt_top40 < 1053.047 and Q.sj2_zsoft < 0.2179035:
        z += 0.01242689 * (1053.047 - Q.sum_pt_top40) * (0.2179035 - Q.sj2_zsoft)
    if Q.sum_pt < 1007.788 and Q.sum_pt_top3 > 512.4375:
        z += 4.552817e-05 * (1007.788 - Q.sum_pt) * (Q.sum_pt_top3 - 512.4375)
    if Q.log_sum_pt > 7.139296 and Q.C3 < 0.00317537:
        z += -1783.757 * (Q.log_sum_pt - 7.139296) * (0.00317537 - Q.C3)
    if Q.mass > 172.8 and Q.pt_6 < 56.53125:
        z += -0.002906893 * (Q.mass - 172.8) * (56.53125 - Q.pt_6)
    if Q.mass > 172.8 and Q.z_6 < 0.04568661:
        z += 2.487708 * (Q.mass - 172.8) * (0.04568661 - Q.z_6)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.2037838
    if Q.tau21_b2 < 0.342495:
        z += -1.50336 * Q.tau21_b2 + 0.5148934
    if Q.mass < 80.4:
        z += 0.2541174 * Q.mass - 21.931
    if 80.4 <= Q.mass < 82.85409:
        z += 0.1780085 * Q.mass - 15.81183
    if 82.85409 <= Q.mass < 91.19:
        z += 0.1275332 * Q.mass - 11.62976
    if Q.psi_0p3 >= 0.9924477:
        z += 137.9205 * Q.psi_0p3 - 136.8789
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.01999488 * Q.n_dr_0p2_0p4 + 0.3599078
    if Q.e2 < 0.02210818:
        z += -50.1803 * Q.e2 + 1.44128
    if 0.02210818 <= Q.e2 < 0.03029714:
        z += -40.52825 * Q.e2 + 1.22789
    if Q.mass_over_sum_pt < 0.08873143:
        z += -39.02631 * Q.mass_over_sum_pt + 5.330047
    if 0.08873143 <= Q.mass_over_sum_pt < 0.09046749:
        z += 17.52432 * Q.mass_over_sum_pt + 0.3122291
    if 0.09046749 <= Q.mass_over_sum_pt < 0.09795415:
        z += -71.10461 * Q.mass_over_sum_pt + 8.330266
    if 0.09795415 <= Q.mass_over_sum_pt < 0.1182259:
        z += -50.26664 * Q.mass_over_sum_pt + 6.2891
    if 0.1182259 <= Q.mass_over_sum_pt < 0.140939:
        z += -15.24583 * Q.mass_over_sum_pt + 2.148732
    if Q.girth2_top20 < 0.006374178:
        z += -11.49083 * Q.girth2_top20 - 0.3282537
    if 0.006374178 <= Q.girth2_top20 < 0.01083435:
        z += 90.0185 * Q.girth2_top20 - 0.9752923
    if Q.girth < 0.076787:
        z += 13.71272 * Q.girth - 1.052959
    if Q.girth2_top5 < 0.007164202:
        z += -113.3541 * Q.girth2_top5 + 0.8120916
    if Q.sd_mass < 69.65633:
        z += -0.009710323 * Q.sd_mass + 0.6763854
    if Q.sd_rg < 0.1778185:
        z += 1.476064 * Q.sd_rg - 0.6096482
    if 0.1778185 <= Q.sd_rg < 0.3017146:
        z += 2.802161 * Q.sd_rg - 0.8454527
    if Q.girth2_top30 < 0.01215787:
        z += 121.6839 * Q.girth2_top30 - 1.479417
    if Q.mass_top30 < 82.66587:
        z += -0.01973522 * Q.mass_top30 + 1.631429
    if Q.sum_pt_top40 < 906.6023:
        z += 0.002531647 * Q.sum_pt_top40 - 1.951812
    if 906.6023 <= Q.sum_pt_top40 < 1024.942:
        z += -0.002901682 * Q.sum_pt_top40 + 2.974057
    if Q.girth2_top3 < 0.001592178:
        z += 258.5159 * Q.girth2_top3 - 0.4116032
    if Q.log_sum_pt < 6.941997:
        z += -6.653822 * Q.log_sum_pt + 45.95752
    if 6.941997 <= Q.log_sum_pt < 6.98945:
        z += 4.91618 * Q.log_sum_pt - 34.36139
    if Q.sum_pt_top50 < 976.277:
        z += 0.006240268 * Q.sum_pt_top50 - 6.09223
    if Q.tau21_b2 < 0.342495 and Q.lam1 < 0.02048524:
        z += -112.178 * (0.342495 - Q.tau21_b2) * (0.02048524 - Q.lam1)
    if Q.N2 < 0.4226723 and Q.max_dr > 0.2404747:
        z += -9.18554 * (0.4226723 - Q.N2) * (Q.max_dr - 0.2404747)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg > 0.2280025:
        z += -1657.383 * (Q.psi_0p3 - 0.9924477) * (Q.sd_rg - 0.2280025)
    if Q.psi_0p3 > 0.9924477 and Q.sd_rg < 0.1881908:
        z += -703.889 * (Q.psi_0p3 - 0.9924477) * (0.1881908 - Q.sd_rg)
    if Q.tau21_b2 < 0.342495 and Q.n_real_top40 < 32.0:
        z += -0.2926539 * (0.342495 - Q.tau21_b2) * (32.0 - Q.n_real_top40)
    if Q.mass_over_sum_pt < 0.09046749 and Q.psi_0p2 > 0.9087063:
        z += -214.4554 * (0.09046749 - Q.mass_over_sum_pt) * (Q.psi_0p2 - 0.9087063)
    if Q.psi_0p3 > 0.9924477 and Q.z_2nd < 0.1231747:
        z += -959.0022 * (Q.psi_0p3 - 0.9924477) * (0.1231747 - Q.z_2nd)
    if Q.girth2_top5 < 0.007164202 and Q.n_pt_above_10 > 13.0:
        z += -9.007927 * (0.007164202 - Q.girth2_top5) * (Q.n_pt_above_10 - 13.0)
    if Q.girth < 0.076787 and Q.z_dr_0p2_0p4 < 0.05180474:
        z += -323.3158 * (0.076787 - Q.girth) * (0.05180474 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.66192
    if Q.z_dr_0p1_0p2 < 0.1203437:
        z += -6.674051 * Q.z_dr_0p1_0p2 + 0.8031798
    if Q.girth2_top5 < 0.002270363:
        z += 126.8581 * Q.girth2_top5 + 0.5298122
    if 0.002270363 <= Q.girth2_top5 < 0.008329695:
        z += -134.9697 * Q.girth2_top5 + 1.124256
    if Q.psi_0p1 >= 0.8976117:
        z += -5.300563 * Q.psi_0p1 + 4.757848
    if Q.sum_pt < 986.0565:
        z += -0.006889963 * Q.sum_pt + 6.725285
    if 986.0565 <= Q.sum_pt < 1002.379:
        z += 0.004203414 * Q.sum_pt - 4.213412
    if Q.log_sum_pt < 6.903423:
        z += 1.005756 * Q.log_sum_pt - 6.157621
    if 6.903423 <= Q.log_sum_pt < 7.017258:
        z += -6.90068 * Q.log_sum_pt + 48.42385
    if Q.psi_0p3 >= 0.9896594:
        z += -49.79105 * Q.psi_0p3 + 49.27618
    if Q.sj2_dr >= 0.2232169:
        z += 3.286926 * Q.sj2_dr - 0.7336975
    if Q.girth2_top10 < 0.007678544:
        z += -44.8179 * Q.girth2_top10 + 0.3441362
    if Q.lam1 < 0.004673423:
        z += -80.78979 * Q.lam1 + 0.3775649
    if Q.lam1 >= 0.01649354:
        z += 35.67514 * Q.lam1 - 0.5884094
    if Q.lam2 < 0.001776308:
        z += -123.2734 * Q.lam2 + 0.2189716
    if Q.tau1 < 0.0705748:
        z += 10.89404 * Q.tau1 - 0.7688445
    if Q.mass_over_sum_pt >= 0.1708801:
        z += -150.3673 * Q.mass_over_sum_pt + 25.69479
    if Q.n_dr_0p2_0p4 >= 9.0:
        z += -0.0179492 * Q.n_dr_0p2_0p4 + 0.1615428
    if Q.sd_mass < 45.595:
        z += 0.003704883 * Q.sd_mass + 0.4377284
    if 45.595 <= Q.sd_mass < 79.18312:
        z += -0.01806152 * Q.sd_mass + 1.430168
    if Q.sj3_pair_mass_max < 35.30013:
        z += -0.01832243 * Q.sj3_pair_mass_max + 0.6467841
    if Q.mass_top50 < 71.79516:
        z += 0.007504977 * Q.mass_top50 - 0.538821
    if Q.sum_pt_top40 < 935.8189:
        z += 0.01522963 * Q.sum_pt_top40 - 15.06052
    if 935.8189 <= Q.sum_pt_top40 < 1018.698:
        z += 0.009753303 * Q.sum_pt_top40 - 9.935668
    if Q.sum_pt_top30 < 933.1875:
        z += -0.006422017 * Q.sum_pt_top30 + 5.992946
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_particles < 58.0:
        z += 0.07699942 * (0.1203437 - Q.z_dr_0p1_0p2) * (58.0 - Q.n_particles)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 949.9169:
        z += 0.006240211 * (0.8459004 - Q.z_dr_0_0p05) * (949.9169 - Q.sum_pt)
    if Q.z_dr_0_0p05 < 0.8459004 and Q.sum_pt < 1260.541:
        z += -0.001355688 * (0.8459004 - Q.z_dr_0_0p05) * (1260.541 - Q.sum_pt)
    if Q.z_dr_0p1_0p2 < 0.1203437 and Q.n_dr_0p2_0p4 > 5.0:
        z += -0.2898822 * (0.1203437 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 5.0)
    if Q.sj2_dr > 0.2232169 and Q.C2_b2 < 0.04008677:
        z += 168.3261 * (Q.sj2_dr - 0.2232169) * (0.04008677 - Q.C2_b2)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_mass1 > 5.112677:
        z += -1.70142 * (0.008329695 - Q.girth2_top5) * (Q.sj3_mass1 - 5.112677)
    if Q.n_dr_0p1_0p2 < 13.0 and Q.absphi_0 < 0.02970886:
        z += -0.8869562 * (13.0 - Q.n_dr_0p1_0p2) * (0.02970886 - Q.absphi_0)
    if Q.girth2_top5 < 0.008329695 and Q.sj3_pairmin_over_m > 0.09540583:
        z += -91.33302 * (0.008329695 - Q.girth2_top5) * (Q.sj3_pairmin_over_m - 0.09540583)
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
