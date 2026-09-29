"""JEDI-linear jet tagger, 64 particles, 3 features: the formula with the fewest quantities (31) at the network's accuracy, as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 92.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
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
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def neuron_0(Q):
    z = 1.3
    if Q.girth < 0.0562:
        z += 27.2 * Q.girth - 1.52864
    if Q.girth2_top20 < 0.00611:
        z += 162.0 * Q.girth2_top20 - 0.98982
    if Q.lam1 < 0.00583:
        z += 197.0 * Q.lam1 - 1.14851
    if 80.4 <= Q.mass < 91.2:
        z += -0.159 * Q.mass + 12.783600000000002
    if Q.mass >= 91.2:
        z += -0.0050000000000000044 * Q.mass - 1.2611999999999988
    if 0.0782 <= Q.mass_over_sum_pt < 0.0857:
        z += -66.9 * Q.mass_over_sum_pt + 5.231580000000001
    if Q.mass_over_sum_pt >= 0.0857:
        z += 13.0 * Q.mass_over_sum_pt - 1.6158499999999991
    if Q.mass_over_sum_pt_sq < 0.00784:
        z += -340.0 * Q.mass_over_sum_pt_sq + 2.6656
    if Q.mass_top50 < 80.8:
        z += 0.0156 * Q.mass_top50 - 1.2604799999999998
    if Q.n_dr_0p2_0p4 < 10.8:
        z += -0.0396 * Q.n_dr_0p2_0p4 + 0.42768000000000006
    if Q.sum_pt < 1010.0:
        z += 0.00481 * Q.sum_pt - 4.8581
    if Q.z_dr_0_0p05 >= 0.871:
        z += -2.97 * Q.z_dr_0_0p05 + 2.5868700000000002
    if Q.log_sum_pt < 6.99 and Q.sum_pt_top40 > 936.0:
        z += 0.0511 * (6.99 - Q.log_sum_pt) * (Q.sum_pt_top40 - 936.0)
    if Q.mass > 80.4 and Q.n_dr_0p2_0p4 < 7.0:
        z += -0.00923 * (Q.mass - 80.4) * (7.0 - Q.n_dr_0p2_0p4)
    if Q.mass_top50 < 70.3 and Q.z_dr_0p05_0p1 < 0.231:
        z += 0.0451 * (70.3 - Q.mass_top50) * (0.231 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0p2_0p4 < 9.75 and Q.z_top50_slots < 0.98:
        z += -2.13 * (9.75 - Q.n_dr_0p2_0p4) * (0.98 - Q.z_top50_slots)
    if Q.z_top30_slots > 0.913 and Q.C2 > 0.0639:
        z += -64.4 * (Q.z_top30_slots - 0.913) * (Q.C2 - 0.0639)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.703
    if Q.D2 < 1.15:
        z += -1.34 * Q.D2 + 1.541
    if Q.LHA < 0.398:
        z += -5.81 * Q.LHA + 2.31238
    if Q.girth < 0.0156:
        z += 92.7 * Q.girth - 1.44612
    if Q.girth2_top20 < 0.00133:
        z += -1050.0 * Q.girth2_top20 + 1.3965
    if Q.lam2 < 0.000885:
        z += 591.0 * Q.lam2 - 0.523035
    if 6.8 <= Q.log_sum_pt < 6.89:
        z += -8.65 * Q.log_sum_pt + 58.82
    if 6.89 <= Q.log_sum_pt < 6.91:
        z += 8.950000000000001 * Q.log_sum_pt - 62.44400000000001
    if 6.91 <= Q.log_sum_pt < 6.98:
        z += 41.75 * Q.log_sum_pt - 289.092
    if Q.log_sum_pt >= 6.98:
        z += 21.75 * Q.log_sum_pt - 149.49199999999996
    if Q.mass_top30 < 79.3:
        z += -0.0328 * Q.mass_top30 + 2.6010400000000002
    if Q.mass_top50 < 114.0:
        z += 0.0189 * Q.mass_top50 - 2.1546
    if Q.max_dr < 0.429:
        z += 2.73 * Q.max_dr - 1.17117
    if Q.n_dr_0p2_0p4 < 7.51:
        z += 0.0998 * Q.n_dr_0p2_0p4 - 0.749498
    if Q.n_particles >= 26.7:
        z += 0.0364 * Q.n_particles - 0.9718800000000001
    if Q.sum_pt_top3 < 761.0:
        z += -0.00448 * Q.sum_pt_top3 + 3.40928
    if Q.sum_pt_top40 >= 1120.0:
        z += 0.0124 * Q.sum_pt_top40 - 13.888
    if 962.0 <= Q.sum_pt_top50 < 1150.0:
        z += -0.0104 * Q.sum_pt_top50 + 10.0048
    if Q.sum_pt_top50 >= 1150.0:
        z += -0.0223 * Q.sum_pt_top50 + 23.689799999999998
    if Q.tau32 >= 0.326:
        z += 2.04 * Q.tau32 - 0.6650400000000001
    if Q.z_dr_0_0p05 >= 0.877:
        z += -6.58 * Q.z_dr_0_0p05 + 5.77066
    if Q.z_top30_slots < 0.997:
        z += 12.0 * Q.z_top30_slots - 11.964
    if Q.z_top50_slots >= 0.958:
        z += -34.4 * Q.z_top50_slots + 32.9552
    if Q.n_particles > 34.8 and Q.dr_0 < 0.131:
        z += 0.358 * (Q.n_particles - 34.8) * (0.131 - Q.dr_0)
    if Q.n_particles > 38.8 and Q.z_top50_slots < 0.985:
        z += 2.08 * (Q.n_particles - 38.8) * (0.985 - Q.z_top50_slots)
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
    z = -0.431
    if Q.C2 >= 0.107:
        z += 17.4 * Q.C2 - 1.8618
    if Q.D2 < 6.81:
        z += -0.224 * Q.D2 + 1.52544
    if Q.e2 >= 0.0356:
        z += 18.6 * Q.e2 - 0.6621600000000001
    if Q.girth < 0.0616:
        z += 21.8 * Q.girth - 1.34288
    if Q.girth2_top20 < 0.00777:
        z += 39.3 * Q.girth2_top20 + 0.85512
    if 0.00777 <= Q.girth2_top20 < 0.0231:
        z += -75.7 * Q.girth2_top20 + 1.74867
    if Q.mass < 80.4:
        z += 0.0855 * Q.mass - 5.845900000000001
    if 80.4 <= Q.mass < 103.0:
        z += -0.0455 * Q.mass + 4.6865
    if Q.mass_over_sum_pt >= 0.0909:
        z += -71.7 * Q.mass_over_sum_pt + 6.51753
    if 0.00841 <= Q.mass_over_sum_pt_sq < 0.0265:
        z += 313.0 * Q.mass_over_sum_pt_sq - 2.63233
    if Q.mass_over_sum_pt_sq >= 0.0265:
        z += 194.0 * Q.mass_over_sum_pt_sq + 0.5211699999999997
    if Q.n_dr_0p2_0p4 < 18.0:
        z += -0.0257 * Q.n_dr_0p2_0p4 + 0.4626
    if Q.n_particles >= 23.7:
        z += -0.013 * Q.n_particles + 0.3081
    if Q.sum_pt < 1080.0:
        z += 0.00554 * Q.sum_pt - 5.9832
    if Q.tau21 < 0.454:
        z += 1.5 * Q.tau21 - 0.681
    if Q.tau32 < 0.573:
        z += -2.27 * Q.tau32 + 1.3007099999999998
    if Q.z_dr_0p2_0p4 < 0.0698:
        z += 10.1 * Q.z_dr_0p2_0p4 - 0.7049799999999999
    if Q.mass < 74.3 and Q.max_dr < 0.36:
        z += -0.564 * (74.3 - Q.mass) * (0.36 - Q.max_dr)
    if Q.mass < 118.0 and Q.max_dr < 0.411:
        z += 0.104 * (118.0 - Q.mass) * (0.411 - Q.max_dr)
    if Q.mass_top40 < 84.6 and Q.D2 < 6.72:
        z += -0.015 * (84.6 - Q.mass_top40) * (6.72 - Q.D2)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.546
    if 6.85 <= Q.log_sum_pt < 6.91:
        z += 31.6 * Q.log_sum_pt - 216.46
    if Q.log_sum_pt >= 6.91:
        z += -12.5 * Q.log_sum_pt + 88.27099999999999
    if 0.0786 <= Q.mass_over_sum_pt < 0.0906:
        z += 39.6 * Q.mass_over_sum_pt - 3.11256
    if Q.mass_over_sum_pt >= 0.0906:
        z += -0.10000000000000142 * Q.mass_over_sum_pt + 0.4842599999999999
    if Q.mass_over_sum_pt_sq < 0.00636:
        z += 103.0 * Q.mass_over_sum_pt_sq - 0.65508
    if Q.mass_over_sum_pt_sq >= 0.0293:
        z += -450.0 * Q.mass_over_sum_pt_sq + 13.185
    if Q.mass_top50 >= 155.0:
        z += -0.213 * Q.mass_top50 + 33.015
    if Q.max_dr >= 0.437:
        z += 19.7 * Q.max_dr - 8.6089
    if 239.0 <= Q.sum_pt_top3 < 791.0:
        z += 0.000694 * Q.sum_pt_top3 - 0.16586599999999999
    if Q.sum_pt_top3 >= 791.0:
        z += -0.0041459999999999995 * Q.sum_pt_top3 + 3.6625739999999998
    if Q.tau32 < 0.881:
        z += -1.14 * Q.tau32 + 1.00434
    if Q.z_top30_slots >= 0.936:
        z += -8.88 * Q.z_top30_slots + 8.31168
    if Q.mass_top50 > 155.0 and Q.z_dr_0p05_0p1 > 0.581:
        z += 1.23 * (Q.mass_top50 - 155.0) * (Q.z_dr_0p05_0p1 - 0.581)
    if Q.n_particles < 65.0 and Q.e2 < 0.0378:
        z += 1.72 * (65.0 - Q.n_particles) * (0.0378 - Q.e2)
    return max(0.0, z)


def neuron_6(Q):
    z = 2.55
    if Q.e2 < 0.0477:
        z += -23.0 * Q.e2 + 1.0971
    if Q.e2 >= 0.0526:
        z += -51.9 * Q.e2 + 2.72994
    if Q.lam1 >= 0.00793:
        z += 228.0 * Q.lam1 - 1.8080399999999999
    if Q.log_sum_pt < 6.82:
        z += -6.95 * Q.log_sum_pt + 47.399
    if Q.mass < 91.2:
        z += 0.004299999999999998 * Q.mass - 1.3584799999999992
    if 91.2 <= Q.mass < 110.0:
        z += 0.0514 * Q.mass - 5.654
    if Q.mass_over_sum_pt >= 0.0491:
        z += -40.3 * Q.mass_over_sum_pt + 1.9787299999999999
    if Q.mass_over_sum_pt_sq < 0.0101:
        z += 166.0 * Q.mass_over_sum_pt_sq - 1.6765999999999999
    if Q.mass_top50 < 70.5:
        z += -0.0531 * Q.mass_top50 + 3.74355
    if Q.n_dr_0p2_0p4 < 18.5:
        z += 0.0589 * Q.n_dr_0p2_0p4 - 1.08965
    if Q.mass_over_sum_pt > 0.0497 and Q.n_dr_0p2_0p4 < 18.6:
        z += 1.77 * (Q.mass_over_sum_pt - 0.0497) * (18.6 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.571
    if Q.e2 < 0.0286:
        z += -66.5 * Q.e2 + 1.9019
    if Q.girth < 0.0884:
        z += 19.4 * Q.girth - 1.71496
    if Q.girth2_top20 < 0.0063:
        z += 36.0 * Q.girth2_top20 - 1.1673000000000004
    if 0.0063 <= Q.girth2_top20 < 0.00795:
        z += 570.0 * Q.girth2_top20 - 4.5315
    if Q.mass < 91.2:
        z += 0.1255 * Q.mass - 9.588
    if 91.2 <= Q.mass < 120.0:
        z += -0.0645 * Q.mass + 7.74
    if Q.mass_over_sum_pt < 0.0799:
        z += 26.0 * Q.mass_over_sum_pt - 3.4255999999999993
    if 0.0799 <= Q.mass_over_sum_pt < 0.0906:
        z += 126.0 * Q.mass_over_sum_pt - 11.4156
    if Q.mass_over_sum_pt_sq < 0.0064:
        z += -274.0 * Q.mass_over_sum_pt_sq + 4.1964
    if 0.0064 <= Q.mass_over_sum_pt_sq < 0.0095:
        z += -788.0 * Q.mass_over_sum_pt_sq + 7.486
    if Q.n_dr_0p2_0p4 < 13.0:
        z += -0.0775 * Q.n_dr_0p2_0p4 + 1.0075
    if Q.z_dr_0p2_0p4 < 0.0059:
        z += -89.0 * Q.z_dr_0p2_0p4 - 0.3337399999999999
    if 0.0059 <= Q.z_dr_0p2_0p4 < 0.0901:
        z += 10.2 * Q.z_dr_0p2_0p4 - 0.91902
    if Q.z_top50_slots < 0.991:
        z += 36.2 * Q.z_top50_slots - 35.8742
    if Q.D2 < 1.8 and Q.n_dr_0p2_0p4 < 9.09:
        z += 0.0834 * (1.8 - Q.D2) * (9.09 - Q.n_dr_0p2_0p4)
    if Q.mass < 100.0 and Q.D2 < 1.64:
        z += 0.273 * (100.0 - Q.mass) * (1.64 - Q.D2)
    if Q.mass < 91.2 and Q.max_dr < 0.392:
        z += -1.33 * (91.2 - Q.mass) * (0.392 - Q.max_dr)
    if Q.mass < 101.0 and Q.max_dr < 0.397:
        z += 0.814 * (101.0 - Q.mass) * (0.397 - Q.max_dr)
    if Q.mass < 91.2 and Q.z_dr_0p05_0p1 > 0.422:
        z += -0.149 * (91.2 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.422)
    if Q.mass_top30 < 75.6 and Q.D2 < 1.61:
        z += 0.183 * (75.6 - Q.mass_top30) * (1.61 - Q.D2)
    if Q.mass_top50 < 97.4 and Q.D2 < 1.65:
        z += -0.388 * (97.4 - Q.mass_top50) * (1.65 - Q.D2)
    return max(0.0, z)


def neuron_8(Q):
    z = 5.9
    if Q.C2 >= 0.0695:
        z += 9.62 * Q.C2 - 0.66859
    if Q.D2 < 1.8:
        z += -0.411 * Q.D2 + 0.7398
    if Q.e2 >= 0.0203:
        z += -43.5 * Q.e2 + 0.8830499999999999
    if Q.girth2_top20 >= 0.00792:
        z += 243.0 * Q.girth2_top20 - 1.92456
    if Q.lam2 < 0.00148:
        z += 487.0 * Q.lam2 - 0.72076
    if Q.log_sum_pt >= 6.94:
        z += 5.7 * Q.log_sum_pt - 39.55800000000001
    if Q.mass < 64.5:
        z += -0.118 * Q.mass + 7.611
    if 65.2 <= Q.mass < 86.5:
        z += -0.0718 * Q.mass + 4.681360000000001
    if 86.5 <= Q.mass < 105.0:
        z += 0.017299999999999996 * Q.mass - 3.025789999999999
    if Q.mass >= 105.0:
        z += -0.05600000000000001 * Q.mass + 4.6707100000000015
    if Q.mass_over_sum_pt < 0.161:
        z += 146.0 * Q.mass_over_sum_pt - 23.506
    if Q.mass_over_sum_pt_sq < 0.0255:
        z += -441.0 * Q.mass_over_sum_pt_sq + 11.2455
    if Q.n_dr_0p2_0p4 < 20.6:
        z += 0.0592 * Q.n_dr_0p2_0p4 - 1.2195200000000002
    if Q.n_particles < 46.1:
        z += 0.0196 * Q.n_particles - 0.90356
    if Q.n_particles >= 50.7:
        z += 0.074 * Q.n_particles - 3.7518000000000002
    if Q.sum_pt_top40 < 1000.0:
        z += -0.00403 * Q.sum_pt_top40 + 3.719199999999999
    if 1000.0 <= Q.sum_pt_top40 < 1040.0:
        z += 0.00777 * Q.sum_pt_top40 - 8.0808
    if Q.mass_over_sum_pt > 0.0757 and Q.sum_pt < 1090.0:
        z += 0.635 * (Q.mass_over_sum_pt - 0.0757) * (1090.0 - Q.sum_pt)
    if Q.n_particles > 51.3 and Q.z_top50_slots > 0.97:
        z += -2.79 * (Q.n_particles - 51.3) * (Q.z_top50_slots - 0.97)
    if Q.sum_pt_top40 < 883.0 and Q.log_sum_pt < 6.89:
        z += 0.0676 * (883.0 - Q.sum_pt_top40) * (6.89 - Q.log_sum_pt)
    if Q.sum_pt_top40 < 1010.0 and Q.max_dr < 0.39:
        z += -0.0926 * (1010.0 - Q.sum_pt_top40) * (0.39 - Q.max_dr)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.389
    if Q.girth < 0.0417:
        z += 29.4 * Q.girth - 1.22598
    if 84.5 <= Q.mass < 151.0:
        z += 0.0366 * Q.mass - 3.0927000000000002
    if Q.mass >= 151.0:
        z += -0.0010000000000000009 * Q.mass + 2.5848999999999998
    if Q.mass_over_sum_pt >= 0.0588:
        z += -59.5 * Q.mass_over_sum_pt + 3.4985999999999997
    if 0.00745 <= Q.mass_over_sum_pt_sq < 0.0257:
        z += 216.0 * Q.mass_over_sum_pt_sq - 1.6092
    if Q.mass_over_sum_pt_sq >= 0.0257:
        z += -99.0 * Q.mass_over_sum_pt_sq + 6.4863
    if Q.mass_top40 < 80.4:
        z += -0.0602 * Q.mass_top40 + 4.84008
    if Q.sum_pt_top40 < 1010.0:
        z += -0.00843 * Q.sum_pt_top40 + 8.5143
    if Q.z_top50_slots < 0.979:
        z += 42.5 * Q.z_top50_slots - 41.6075
    if Q.mass_top40 < 115.0 and Q.max_dr > 0.375:
        z += 0.058 * (115.0 - Q.mass_top40) * (Q.max_dr - 0.375)
    if Q.sum_pt_top40 < 931.0 and Q.z_top30_slots > 0.977:
        z += 0.769 * (931.0 - Q.sum_pt_top40) * (Q.z_top30_slots - 0.977)
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
    z = 0.0318
    if Q.e2 < 0.0234:
        z += -73.9 * Q.e2 + 1.7292600000000002
    if Q.lam2 < 0.000827:
        z += -748.0 * Q.lam2 + 0.618596
    if Q.mass_over_sum_pt < 0.0797:
        z += 41.7 * Q.mass_over_sum_pt - 3.32349
    if Q.n_dr_0p2_0p4 < 14.3:
        z += -0.0626 * Q.n_dr_0p2_0p4 + 0.8951800000000001
    if Q.z_dr_0p2_0p4 < 0.0049:
        z += -183.0 * Q.z_dr_0p2_0p4 + 0.8966999999999999
    if Q.mass < 80.4 and Q.D2 < 3.24:
        z += -0.0276 * (80.4 - Q.mass) * (3.24 - Q.D2)
    if Q.mass < 98.1 and Q.D2 < 1.27:
        z += 0.091 * (98.1 - Q.mass) * (1.27 - Q.D2)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.0817
    if Q.C2 >= 0.0556:
        z += -21.6 * Q.C2 + 1.20096
    if Q.log_sum_pt < 6.85:
        z += 5.35 * Q.log_sum_pt - 36.647499999999994
    if Q.mass < 80.4:
        z += -0.0839 * Q.mass + 6.74556
    if Q.mass_top30 >= 128.0:
        z += 0.0361 * Q.mass_top30 - 4.6208
    if Q.mass_top50 < 62.5:
        z += 0.0701 * Q.mass_top50 - 4.38125
    if Q.LHA > 0.406 and Q.lam2 < 0.0046:
        z += 15700.0 * (Q.LHA - 0.406) * (0.0046 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 3.07
    if Q.e2 >= 0.0369:
        z += 16.0 * Q.e2 - 0.5904
    if Q.log_sum_pt < 6.8:
        z += -4.800000000000001 * Q.log_sum_pt + 29.840000000000003
    if 6.8 <= Q.log_sum_pt < 7.0:
        z += 14.0 * Q.log_sum_pt - 98.0
    if 143.0 <= Q.mass < 172.8:
        z += -0.145 * Q.mass + 20.735
    if Q.mass >= 172.8:
        z += -0.040999999999999995 * Q.mass + 2.7638
    if Q.mass_top50 >= 137.0:
        z += 0.0597 * Q.mass_top50 - 8.1789
    if Q.n_dr_0p2_0p4 < 4.41:
        z += 0.156 * Q.n_dr_0p2_0p4 - 0.68796
    if Q.sum_pt < 1020.0:
        z += 0.0226 * Q.sum_pt - 23.052
    if Q.sum_pt_top40 < 1040.0:
        z += -0.0145 * Q.sum_pt_top40 + 15.08
    if Q.sum_pt_top40 < 1040.0 and Q.D2 < 5.03:
        z += -0.00146 * (1040.0 - Q.sum_pt_top40) * (5.03 - Q.D2)
    if Q.sum_pt_top50 < 961.0 and Q.D2 < 5.16:
        z += 0.00193 * (961.0 - Q.sum_pt_top50) * (5.16 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.0493
    if Q.log_sum_pt >= 6.94:
        z += 2.54 * Q.log_sum_pt - 17.6276
    if Q.mass < 91.2:
        z += 0.0872 * Q.mass - 5.859000000000001
    if 91.2 <= Q.mass < 135.0:
        z += -0.0478 * Q.mass + 6.453
    if Q.mass_over_sum_pt < 0.0888:
        z += 52.0 * Q.mass_over_sum_pt - 4.6176
    if Q.max_dr >= 0.225:
        z += -2.24 * Q.max_dr + 0.5040000000000001
    if Q.z_top50_slots < 0.96:
        z += -89.4 * Q.z_top50_slots + 85.824
    if Q.girth2_top20 < 0.0129 and Q.tau21 < 0.6:
        z += -118.0 * (0.0129 - Q.girth2_top20) * (0.6 - Q.tau21)
    return max(0.0, z)


def neuron_15(Q):
    z = 2.19
    if Q.LHA >= 0.163:
        z += 34.5 * Q.LHA - 5.6235
    if Q.girth >= 0.0293:
        z += -120.0 * Q.girth + 3.516
    if Q.lam1 < 0.0176:
        z += 136.0 * Q.lam1 - 2.3936
    if Q.log_sum_pt < 7.05:
        z += -2.61 * Q.log_sum_pt + 18.400499999999997
    if Q.mass_over_sum_pt >= 0.0926:
        z += 28.9 * Q.mass_over_sum_pt - 2.6761399999999997
    if Q.girth2_top5 < 0.0105 and Q.max_dr < 0.328:
        z += -583.0 * (0.0105 - Q.girth2_top5) * (0.328 - Q.max_dr)
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
