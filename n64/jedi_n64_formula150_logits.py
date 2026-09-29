"""JEDI-linear jet tagger, 64 particles, 3 features: the formula simplified by hand with the training data (main result), as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 91.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau32                  N-subjettiness τ3/τ2
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

    return SimpleNamespace(
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_9=pt[9],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def neuron_0(Q):
    z = 1.15
    if Q.girth < 0.0555:
        z += 36.5 * Q.girth - 2.02575
    if Q.girth2_top20 < 0.00622:
        z += 167.0 * Q.girth2_top20 - 1.03874
    if Q.lam1 < 0.00578:
        z += 305.0 * Q.lam1 - 1.7629000000000001
    if 80.4 <= Q.mass < 91.2:
        z += -0.199 * Q.mass + 15.999600000000003
    if Q.mass >= 91.2:
        z += 0.0040000000000000036 * Q.mass - 2.5139999999999976
    if Q.mass_over_sum_pt_sq < 0.00822:
        z += -465.0 * Q.mass_over_sum_pt_sq + 3.8223
    if Q.mass_top50 < 81.8:
        z += 0.0121 * Q.mass_top50 - 0.9897799999999999
    if Q.n_dr_0p2_0p4 < 11.0:
        z += -0.0298 * Q.n_dr_0p2_0p4 + 0.3278
    if Q.z_dr_0_0p05 >= 0.883:
        z += -4.39 * Q.z_dr_0_0p05 + 3.8763699999999996
    if Q.mass > 80.4 and Q.n_dr_0p2_0p4 < 6.91:
        z += -0.007 * (Q.mass - 80.4) * (6.91 - Q.n_dr_0p2_0p4)
    if Q.mass_top50 < 72.0 and Q.z_dr_0p05_0p1 < 0.259:
        z += 0.0617 * (72.0 - Q.mass_top50) * (0.259 - Q.z_dr_0p05_0p1)
    if Q.sum_pt < 1020.0 and Q.z_dr_0p1_0p2 > 0.0771:
        z += -0.0185 * (1020.0 - Q.sum_pt) * (Q.z_dr_0p1_0p2 - 0.0771)
    if Q.sum_pt < 1010.0 and Q.z_dr_0p2_0p4 < 0.212:
        z += -0.0367 * (1010.0 - Q.sum_pt) * (0.212 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.15
    if Q.girth2_top3 < 0.00073:
        z += -1170.0 * Q.girth2_top3 + 0.8541
    if 6.91 <= Q.log_sum_pt < 6.99:
        z += 42.3 * Q.log_sum_pt - 292.293
    if Q.log_sum_pt >= 6.99:
        z += 19.9 * Q.log_sum_pt - 135.717
    if Q.max_dr < 0.435:
        z += 3.12 * Q.max_dr - 1.3572
    if Q.pt_9 < 33.6:
        z += 0.0342 * Q.pt_9 - 1.1491200000000001
    if Q.sum_pt_top2 < 662.0:
        z += -0.00354 * Q.sum_pt_top2 + 2.34348
    if Q.sum_pt_top30 < 1070.0:
        z += -0.00313 * Q.sum_pt_top30 + 3.3491
    if Q.sum_pt_top50 >= 930.0:
        z += -0.00905 * Q.sum_pt_top50 + 8.416500000000001
    if Q.tau32 >= 0.276:
        z += 2.15 * Q.tau32 - 0.5934
    if Q.z_top20_slots < 0.931:
        z += 5.47 * Q.z_top20_slots - 5.09257
    if Q.z_top50_slots >= 0.96:
        z += -29.5 * Q.z_top50_slots + 28.32
    if Q.girth2_top3 < 0.000994 and Q.n_dr_0p05_0p1 < 9.47:
        z += -80.0 * (0.000994 - Q.girth2_top3) * (9.47 - Q.n_dr_0p05_0p1)
    if Q.mass_top20 < 48.5 and Q.n_real_top40 > 27.0:
        z += 0.00323 * (48.5 - Q.mass_top20) * (Q.n_real_top40 - 27.0)
    if Q.n_dr_0p1_0p2 < 6.42 and Q.dr_6 < 0.0751:
        z += -1.39 * (6.42 - Q.n_dr_0p1_0p2) * (0.0751 - Q.dr_6)
    if Q.n_particles > 38.6 and Q.dr_0 < 0.136:
        z += 0.55 * (Q.n_particles - 38.6) * (0.136 - Q.dr_0)
    if Q.n_particles > 37.9 and Q.z_top50_slots < 0.985:
        z += 1.48 * (Q.n_particles - 37.9) * (0.985 - Q.z_top50_slots)
    if Q.sum_pt_top2 < 748.0 and Q.n_dr_0p2_0p4 < 7.95:
        z += -0.000242 * (748.0 - Q.sum_pt_top2) * (7.95 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.38
    if Q.sum_pt_top50 > 1090.0 and Q.C2 > 0.0936:
        z += 0.637 * (Q.sum_pt_top50 - 1090.0) * (Q.C2 - 0.0936)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.217
    if Q.girth2_top5 < 0.000115:
        z += 6960.0 * Q.girth2_top5 - 0.8004
    if Q.lam2 < 0.000569:
        z += -2040.0 * Q.lam2 + 1.16076
    if Q.mass_over_sum_pt_sq < 0.00813:
        z += -105.0 * Q.mass_over_sum_pt_sq + 0.85365
    if Q.n_dr_0p2_0p4 < 5.2 and Q.dr_0 < 0.0493:
        z += -6.07 * (5.2 - Q.n_dr_0p2_0p4) * (0.0493 - Q.dr_0)
    if Q.n_dr_0p2_0p4 < 4.84 and Q.z_top50_slots > 0.979:
        z += 13.3 * (4.84 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.979)
    if Q.n_particles < 46.0 and Q.sum_pt_top40 > 822.0:
        z += 0.000131 * (46.0 - Q.n_particles) * (Q.sum_pt_top40 - 822.0)
    if Q.tau21 < 0.419 and Q.lam1 < 0.0208:
        z += -186.0 * (0.419 - Q.tau21) * (0.0208 - Q.lam1)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.494
    if Q.C2 >= 0.106:
        z += 18.1 * Q.C2 - 1.9186
    if Q.D2 < 6.52:
        z += -0.258 * Q.D2 + 1.6821599999999999
    if Q.girth < 0.063:
        z += 22.7 * Q.girth - 1.4301
    if Q.mass < 80.4:
        z += 0.09280000000000002 * Q.mass - 6.139800000000001
    if 80.4 <= Q.mass < 109.0:
        z += -0.0462 * Q.mass + 5.0358
    if Q.n_particles >= 22.7:
        z += -0.0178 * Q.n_particles + 0.40406
    if Q.sum_pt < 1070.0:
        z += 0.00597 * Q.sum_pt - 6.387899999999999
    if Q.tau21 < 0.451:
        z += 1.65 * Q.tau21 - 0.74415
    if Q.tau32 < 0.578:
        z += -2.44 * Q.tau32 + 1.4103199999999998
    if Q.z_dr_0p2_0p4 < 0.0697:
        z += 6.43 * Q.z_dr_0p2_0p4 - 0.448171
    if Q.mass < 74.4 and Q.max_dr < 0.35:
        z += -0.652 * (74.4 - Q.mass) * (0.35 - Q.max_dr)
    if Q.mass < 124.0 and Q.max_dr < 0.401:
        z += 0.11 * (124.0 - Q.mass) * (0.401 - Q.max_dr)
    if Q.mass_top40 < 84.9 and Q.D2 < 6.27:
        z += -0.0163 * (84.9 - Q.mass_top40) * (6.27 - Q.D2)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.865
    if 6.85 <= Q.log_sum_pt < 6.91:
        z += 30.2 * Q.log_sum_pt - 206.86999999999998
    if Q.log_sum_pt >= 6.91:
        z += -13.599999999999998 * Q.log_sum_pt + 95.78799999999998
    if Q.mass >= 63.2:
        z += 0.0278 * Q.mass - 1.75696
    if Q.mass_over_sum_pt_sq >= 0.0293:
        z += -448.0 * Q.mass_over_sum_pt_sq + 13.1264
    if 102.0 <= Q.mass_top50 < 157.0:
        z += -0.0309 * Q.mass_top50 + 3.1518
    if Q.mass_top50 >= 157.0:
        z += -0.2399 * Q.mass_top50 + 35.9648
    if Q.max_dr >= 0.436:
        z += 19.7 * Q.max_dr - 8.5892
    if Q.z_top30_slots >= 0.943:
        z += -5.92 * Q.z_top30_slots + 5.58256
    if Q.girth2_top30 < 0.0226 and Q.tau32 < 0.858:
        z += 77.3 * (0.0226 - Q.girth2_top30) * (0.858 - Q.tau32)
    if Q.mass_top50 > 156.0 and Q.z_dr_0p05_0p1 > 0.587:
        z += 1.32 * (Q.mass_top50 - 156.0) * (Q.z_dr_0p05_0p1 - 0.587)
    if Q.n_particles < 67.6 and Q.e2 < 0.0387:
        z += 1.47 * (67.6 - Q.n_particles) * (0.0387 - Q.e2)
    return max(0.0, z)


def neuron_6(Q):
    z = 1.22
    if Q.e2 >= 0.0521:
        z += -49.4 * Q.e2 + 2.57374
    if Q.lam1 >= 0.00717:
        z += 186.0 * Q.lam1 - 1.33362
    if Q.mass < 91.2:
        z += 0.010600000000000012 * Q.mass - 1.7046600000000014
    if 91.2 <= Q.mass < 101.0:
        z += 0.0753 * Q.mass - 7.605300000000001
    if Q.mass_over_sum_pt >= 0.0534:
        z += -25.3 * Q.mass_over_sum_pt + 1.35102
    if Q.mass_top10 < 92.5:
        z += -0.00816 * Q.mass_top10 + 0.7548
    if Q.mass_top50 < 73.1:
        z += -0.0538 * Q.mass_top50 + 3.9327799999999997
    if Q.n_dr_0p2_0p4 < 17.7:
        z += 0.059 * Q.n_dr_0p2_0p4 - 1.0443
    if Q.mass_over_sum_pt > 0.0538 and Q.n_dr_0p2_0p4 < 19.4:
        z += 1.76 * (Q.mass_over_sum_pt - 0.0538) * (19.4 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.726
    if Q.girth2_top20 < 0.00812:
        z += 323.0 * Q.girth2_top20 - 2.62276
    if Q.girth2_top40 < 0.0129:
        z += -269.0 * Q.girth2_top40 + 3.4701
    if Q.mass < 91.2:
        z += 0.1381 * Q.mass - 11.7235
    if 91.2 <= Q.mass < 101.0:
        z += -0.0889 * Q.mass + 8.978900000000001
    if Q.mass_top20 < 66.0:
        z += -0.0401 * Q.mass_top20 + 2.6466
    if Q.z_top50_slots < 0.998:
        z += 27.6 * Q.z_top50_slots - 27.544800000000002
    if Q.D2 < 1.99 and Q.girth2_top50 < 0.00792:
        z += -564.0 * (1.99 - Q.D2) * (0.00792 - Q.girth2_top50)
    if Q.D2 < 2.0 and Q.girth2_top50 < 0.00623:
        z += 715.0 * (2.0 - Q.D2) * (0.00623 - Q.girth2_top50)
    if Q.D2 < 2.06 and Q.n_dr_0p2_0p4 < 8.39:
        z += 0.185 * (2.06 - Q.D2) * (8.39 - Q.n_dr_0p2_0p4)
    if Q.mass < 100.0 and Q.D2 < 1.61:
        z += 0.307 * (100.0 - Q.mass) * (1.61 - Q.D2)
    if Q.mass < 91.2 and Q.max_dr < 0.391:
        z += -1.31 * (91.2 - Q.mass) * (0.391 - Q.max_dr)
    if Q.mass < 102.0 and Q.max_dr < 0.393:
        z += 0.819 * (102.0 - Q.mass) * (0.393 - Q.max_dr)
    if Q.mass_top50 < 97.2 and Q.D2 < 1.61:
        z += -0.439 * (97.2 - Q.mass_top50) * (1.61 - Q.D2)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.721
    if Q.C2 >= 0.0661:
        z += 7.66 * Q.C2 - 0.506326
    if Q.girth2_top30 < 0.00815:
        z += -74.3 * Q.girth2_top30 + 0.6055449999999999
    if Q.lam2 < 0.00143:
        z += 343.0 * Q.lam2 - 0.49049000000000004
    if 91.2 <= Q.mass < 104.0:
        z += 0.105 * Q.mass - 9.576
    if Q.mass >= 104.0:
        z += -0.0040000000000000036 * Q.mass + 1.7599999999999998
    if Q.n_dr_0p2_0p4 < 20.2:
        z += 0.0677 * Q.n_dr_0p2_0p4 - 1.36754
    if Q.n_particles >= 49.3:
        z += 0.0763 * Q.n_particles - 3.76159
    if Q.sum_pt < 1010.0:
        z += -0.0142 * Q.sum_pt + 14.342
    if Q.girth2_top40 > 0.00145 and Q.sum_pt_top30 < 932.0:
        z += 0.687 * (Q.girth2_top40 - 0.00145) * (932.0 - Q.sum_pt_top30)
    if Q.mass_over_sum_pt > 0.0762 and Q.sum_pt < 1130.0:
        z += 0.477 * (Q.mass_over_sum_pt - 0.0762) * (1130.0 - Q.sum_pt)
    if Q.n_particles > 50.6 and Q.z_top50_slots > 0.972:
        z += -3.12 * (Q.n_particles - 50.6) * (Q.z_top50_slots - 0.972)
    if Q.sum_pt < 1010.0 and Q.n_real_top50 < 43.2:
        z += -0.00105 * (1010.0 - Q.sum_pt) * (43.2 - Q.n_real_top50)
    if Q.sum_pt_top40 < 958.0 and Q.log_sum_pt < 6.82:
        z += 0.0688 * (958.0 - Q.sum_pt_top40) * (6.82 - Q.log_sum_pt)
    if Q.sum_pt_top40 < 1040.0 and Q.max_dr < 0.415:
        z += -0.0644 * (1040.0 - Q.sum_pt_top40) * (0.415 - Q.max_dr)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.0128
    if Q.girth2_top15 < 0.0191:
        z += 104.0 * Q.girth2_top15 - 1.9864
    if 61.8 <= Q.mass < 80.4:
        z += -0.0736 * Q.mass + 4.54848
    if 80.4 <= Q.mass < 142.0:
        z += 0.0374 * Q.mass - 4.375920000000001
    if Q.mass >= 142.0:
        z += -0.0109 * Q.mass + 2.4826799999999993
    if Q.mass_top50 < 128.0:
        z += -0.0454 * Q.mass_top50 + 5.8112
    if Q.width >= 0.026:
        z += -134.0 * Q.width + 3.484
    if Q.girth2_top15 < 0.0145 and Q.n_particles > 35.1:
        z += 2.2 * (0.0145 - Q.girth2_top15) * (Q.n_particles - 35.1)
    if Q.girth2_top40 < 0.00576 and Q.sum_pt < 997.0:
        z += 3.67 * (0.00576 - Q.girth2_top40) * (997.0 - Q.sum_pt)
    if Q.log_sum_pt < 6.85 and Q.dr_7 < 0.0848:
        z += 160.0 * (6.85 - Q.log_sum_pt) * (0.0848 - Q.dr_7)
    if Q.mass_top40 < 113.0 and Q.max_dr > 0.386:
        z += 0.0589 * (113.0 - Q.mass_top40) * (Q.max_dr - 0.386)
    return max(0.0, z)


def neuron_10(Q):
    z = 4.71
    if Q.e2 < 0.0611:
        z += 33.4 * Q.e2 - 2.04074
    if Q.girth2_top5 < 0.0254:
        z += 99.2 * Q.girth2_top5 - 2.51968
    if Q.lam1 >= 0.00248:
        z += -53.6 * Q.lam1 + 0.132928
    if Q.mass < 80.4:
        z += -0.064 * Q.mass + 3.714240000000001
    if 80.4 <= Q.mass < 123.0:
        z += 0.0336 * Q.mass - 4.1328
    if 144.0 <= Q.mass < 162.0:
        z += -0.0928 * Q.mass + 13.363199999999999
    if Q.mass >= 162.0:
        z += -0.14839999999999998 * Q.mass + 22.370399999999997
    if Q.mass_top50 >= 137.0:
        z += 0.0798 * Q.mass_top50 - 10.932599999999999
    if Q.max_dr < 0.436:
        z += 2.55 * Q.max_dr - 1.1118
    if Q.n_dr_0p2_0p4 < 11.3:
        z += 0.107 * Q.n_dr_0p2_0p4 - 1.2091
    if Q.sum_pt_top50 < 954.0:
        z += 0.0083 * Q.sum_pt_top50 - 7.9182
    if Q.z_dr_0_0p05 >= 0.769:
        z += -12.3 * Q.z_dr_0_0p05 + 9.4587
    if Q.z_dr_0p2_0p4 < 0.0879:
        z += -12.6 * Q.z_dr_0p2_0p4 + 1.10754
    if Q.dr_0 < 0.0612 and Q.n_dr_0p2_0p4 > -0.94:
        z += 0.679 * (0.0612 - Q.dr_0) * (Q.n_dr_0p2_0p4 - -0.94)
    if Q.girth2_top5 < 0.0261 and Q.sum_pt_top3 < 668.0:
        z += 0.128 * (0.0261 - Q.girth2_top5) * (668.0 - Q.sum_pt_top3)
    if Q.mass < 122.0 and Q.tau21 < 0.47:
        z += 0.0788 * (122.0 - Q.mass) * (0.47 - Q.tau21)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.196
    if Q.z_dr_0p2_0p4 < 0.00632:
        z += -178.0 * Q.z_dr_0p2_0p4 + 1.12496
    if Q.mass < 80.4 and Q.D2 < 3.34:
        z += -0.0284 * (80.4 - Q.mass) * (3.34 - Q.D2)
    if Q.mass < 103.0 and Q.planar_flow < 0.362:
        z += 0.103 * (103.0 - Q.mass) * (0.362 - Q.planar_flow)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.00648:
        z += -38.9 * (10.0 - Q.n_dr_0p2_0p4) * (0.00648 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.6 and Q.n_dr_0p1_0p2 < 26.9:
        z += 0.00712 * (10.6 - Q.n_dr_0p2_0p4) * (26.9 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.085
    if Q.C2 >= 0.0528:
        z += -19.0 * Q.C2 + 1.0032
    if Q.mass < 80.4:
        z += -0.0807 * Q.mass + 6.48828
    if Q.mass_top30 >= 136.0:
        z += 0.0686 * Q.mass_top30 - 9.3296
    if Q.mass_top50 < 63.0:
        z += 0.066 * Q.mass_top50 - 4.158
    if Q.LHA > 0.405 and Q.lam2 < 0.00471:
        z += 12300.0 * (Q.LHA - 0.405) * (0.00471 - Q.lam2)
    if Q.mass_top20 > 131.0 and Q.C2 > 0.065:
        z += -1.26 * (Q.mass_top20 - 131.0) * (Q.C2 - 0.065)
    return max(0.0, z)


def neuron_13(Q):
    z = 3.03
    if Q.log_sum_pt < 6.82:
        z += -6.599999999999998 * Q.log_sum_pt + 42.66599999999998
    if 6.82 <= Q.log_sum_pt < 6.99:
        z += 13.8 * Q.log_sum_pt - 96.462
    if 144.0 <= Q.mass < 172.8:
        z += -0.0922 * Q.mass + 13.276800000000001
    if Q.mass >= 172.8:
        z += 0.01079999999999999 * Q.mass - 4.521599999999999
    if Q.mass_top10 >= 66.7:
        z += 0.0156 * Q.mass_top10 - 1.04052
    if Q.n_dr_0p2_0p4 < 4.7:
        z += 0.154 * Q.n_dr_0p2_0p4 - 0.7238
    if Q.sum_pt < 1020.0:
        z += 0.0229 * Q.sum_pt - 23.358
    if Q.sum_pt_top40 < 1020.0:
        z += -0.0131 * Q.sum_pt_top40 + 13.362
    return max(0.0, z)


def neuron_14(Q):
    z = 0.349
    if Q.mass < 91.2:
        z += 0.10240000000000002 * Q.mass - 7.842400000000002
    if 91.2 <= Q.mass < 131.0:
        z += -0.0376 * Q.mass + 4.9256
    if Q.mass_over_sum_pt < 0.089:
        z += 26.7 * Q.mass_over_sum_pt - 2.3762999999999996
    if Q.max_dr >= 0.237:
        z += -2.12 * Q.max_dr + 0.50244
    if Q.z_top50_slots < 0.961:
        z += -88.4 * Q.z_top50_slots + 84.9524
    return max(0.0, z)


def neuron_15(Q):
    z = -0.204
    if Q.log_sum_pt < 7.04:
        z += -2.02 * Q.log_sum_pt + 14.2208
    if Q.z_dr_0p1_0p2 < 0.231 and Q.planar_flow < 1.09:
        z += 3.35 * (0.231 - Q.z_dr_0p1_0p2) * (1.09 - Q.planar_flow)
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
