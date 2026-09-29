"""JEDI-linear jet tagger, 64 particles, 3 features: the simpler version of the simplified formula, as if-statements, with each class score (logit) written out as a formula.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.5% (the network: 81.1%); same class as the network for 91.3% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
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
        mass_top10=mass_of(10),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
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
    z = 1.32
    if Q.girth < 0.057:
        z += 45.8 * Q.girth - 2.6106
    if Q.girth2_top20 < 0.0062:
        z += 195.0 * Q.girth2_top20 - 1.2089999999999999
    if Q.lam1 < 0.0058:
        z += 316.0 * Q.lam1 - 1.8327999999999998
    if 80.4 <= Q.mass < 91.2:
        z += -0.2 * Q.mass + 16.080000000000002
    if Q.mass >= 91.2:
        z += 0.003999999999999976 * Q.mass - 2.524799999999999
    if Q.mass_over_sum_pt_sq < 0.0079:
        z += -532.0 * Q.mass_over_sum_pt_sq + 4.202800000000001
    if Q.sum_pt < 1000.0 and Q.z_dr_0p2_0p4 < 0.21:
        z += -0.052 * (1000.0 - Q.sum_pt) * (0.21 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.97
    if Q.girth2_top20 < 0.0021:
        z += -500.0 * Q.girth2_top20 + 1.05
    if 6.8 <= Q.log_sum_pt < 6.9:
        z += -10.0 * Q.log_sum_pt + 68.0
    if 6.9 <= Q.log_sum_pt < 7.0:
        z += 27.299999999999997 * Q.log_sum_pt - 189.37
    if Q.log_sum_pt >= 7.0:
        z += 8.399999999999999 * Q.log_sum_pt - 57.07000000000002
    if Q.mass_top30 < 73.0:
        z += -0.0144 * Q.mass_top30 + 1.0512
    if Q.n_particles < 43.0:
        z += 0.0759 * Q.n_particles - 3.2636999999999996
    if Q.sum_pt_top3 < 760.0:
        z += -0.00512 * Q.sum_pt_top3 + 3.8912000000000004
    if Q.tau32 >= 0.28:
        z += 2.12 * Q.tau32 - 0.5936000000000001
    if Q.z_top20_slots < 0.96:
        z += 6.69 * Q.z_top20_slots - 6.4224000000000006
    if Q.z_top50_slots >= 0.96:
        z += -49.5 * Q.z_top50_slots + 47.519999999999996
    if Q.n_particles > 38.0 and Q.dr_0 < 0.11:
        z += 0.733 * (Q.n_particles - 38.0) * (0.11 - Q.dr_0)
    if Q.n_particles > 40.0 and Q.z_top50_slots < 0.98:
        z += 1.62 * (Q.n_particles - 40.0) * (0.98 - Q.z_top50_slots)
    if Q.sum_pt_top2 < 820.0 and Q.n_dr_0p2_0p4 < 7.7:
        z += -0.0003 * (820.0 - Q.sum_pt_top2) * (7.7 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.381
    if Q.sum_pt_top50 > 1100.0 and Q.C2 > 0.094:
        z += 0.663 * (Q.sum_pt_top50 - 1100.0) * (Q.C2 - 0.094)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.287
    if Q.girth2_top5 < 0.00011:
        z += 7010.0 * Q.girth2_top5 - 0.7711
    if Q.lam2 < 0.00052:
        z += -2090.0 * Q.lam2 + 1.0868
    if Q.mass_over_sum_pt_sq < 0.008:
        z += -126.0 * Q.mass_over_sum_pt_sq + 1.008
    if Q.n_particles < 45.0 and Q.sum_pt_top40 > 830.0:
        z += 0.000125 * (45.0 - Q.n_particles) * (Q.sum_pt_top40 - 830.0)
    return max(0.0, z)


def neuron_4(Q):
    z = -0.532
    if Q.C2 >= 0.1:
        z += 16.9 * Q.C2 - 1.69
    if Q.D2 < 7.7:
        z += -0.209 * Q.D2 + 1.6093
    if Q.mass < 80.4:
        z += 0.09530000000000001 * Q.mass - 6.368600000000001
    if 80.4 <= Q.mass < 110.0:
        z += -0.0437 * Q.mass + 4.807
    if Q.sum_pt < 1100.0:
        z += 0.00427 * Q.sum_pt - 4.697
    if Q.tau32 < 0.61:
        z += -2.89 * Q.tau32 + 1.7629000000000001
    if Q.mass < 75.0 and Q.max_dr < 0.33:
        z += -0.627 * (75.0 - Q.mass) * (0.33 - Q.max_dr)
    if Q.mass < 120.0 and Q.max_dr < 0.4:
        z += 0.118 * (120.0 - Q.mass) * (0.4 - Q.max_dr)
    if Q.mass_top40 < 85.0 and Q.D2 < 8.2:
        z += -0.0121 * (85.0 - Q.mass_top40) * (8.2 - Q.D2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.871
    if Q.log_sum_pt >= 6.9:
        z += -7.36 * Q.log_sum_pt + 50.784000000000006
    if Q.mass_over_sum_pt_sq >= 0.029:
        z += -1240.0 * Q.mass_over_sum_pt_sq + 35.96
    if Q.mass_top30 >= 49.0:
        z += 0.0156 * Q.mass_top30 - 0.7644
    if Q.mass_top50 >= 150.0:
        z += -0.144 * Q.mass_top50 + 21.599999999999998
    if Q.max_dr >= 0.44:
        z += 19.8 * Q.max_dr - 8.712
    if Q.z_top30_slots >= 0.93:
        z += -9.11 * Q.z_top30_slots + 8.4723
    if Q.girth2_top30 < 0.02 and Q.tau32 < 0.86:
        z += 96.1 * (0.02 - Q.girth2_top30) * (0.86 - Q.tau32)
    if Q.mass_top50 > 160.0 and Q.z_dr_0p05_0p1 > 0.55:
        z += 0.75 * (Q.mass_top50 - 160.0) * (Q.z_dr_0p05_0p1 - 0.55)
    if Q.n_particles < 66.0 and Q.e2 < 0.038:
        z += 1.57 * (66.0 - Q.n_particles) * (0.038 - Q.e2)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.985
    if Q.mass < 91.2:
        z += -0.034 * Q.mass + 2.0183999999999997
    if 91.2 <= Q.mass < 100.0:
        z += 0.123 * Q.mass - 12.3
    return max(0.0, z)


def neuron_7(Q):
    z = -0.612
    if Q.e2 < 0.026:
        z += -45.3 * Q.e2 + 1.1778
    if Q.mass < 91.2:
        z += 0.1462 * Q.mass - 12.072
    if 91.2 <= Q.mass < 120.0:
        z += -0.0438 * Q.mass + 5.256
    if Q.n_dr_0p2_0p4 < 9.4:
        z += -0.123 * Q.n_dr_0p2_0p4 + 1.1562000000000001
    if Q.z_top50_slots < 0.99:
        z += 35.5 * Q.z_top50_slots - 35.145
    if Q.mass < 100.0 and Q.D2 < 1.6:
        z += 0.459 * (100.0 - Q.mass) * (1.6 - Q.D2)
    if Q.mass < 91.2 and Q.max_dr < 0.39:
        z += -1.6 * (91.2 - Q.mass) * (0.39 - Q.max_dr)
    if Q.mass < 100.0 and Q.max_dr < 0.4:
        z += 1.02 * (100.0 - Q.mass) * (0.4 - Q.max_dr)
    if Q.mass_top50 < 97.0 and Q.D2 < 1.6:
        z += -0.589 * (97.0 - Q.mass_top50) * (1.6 - Q.D2)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.0348
    if Q.girth < 0.096:
        z += -16.9 * Q.girth + 1.6223999999999998
    if Q.lam2 < 0.0015:
        z += 508.0 * Q.lam2 - 0.762
    if 71.0 <= Q.mass < 120.0:
        z += 0.0494 * Q.mass - 3.5074
    if Q.mass >= 120.0:
        z += 0.0022000000000000006 * Q.mass + 2.1565999999999996
    if Q.n_dr_0p2_0p4 < 19.0:
        z += 0.0584 * Q.n_dr_0p2_0p4 - 1.1096
    if Q.n_particles >= 58.0:
        z += 0.11 * Q.n_particles - 6.38
    if Q.sum_pt < 1000.0:
        z += -0.0165 * Q.sum_pt + 16.5
    if Q.mass_over_sum_pt > 0.085 and Q.sum_pt < 1100.0:
        z += 0.758 * (Q.mass_over_sum_pt - 0.085) * (1100.0 - Q.sum_pt)
    if Q.sum_pt < 1000.0 and Q.n_real_top50 < 44.0:
        z += -0.00117 * (1000.0 - Q.sum_pt) * (44.0 - Q.n_real_top50)
    if Q.sum_pt_top40 < 930.0 and Q.log_sum_pt < 6.9:
        z += 0.0562 * (930.0 - Q.sum_pt_top40) * (6.9 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.155
    if Q.girth2_top40 < 0.0056:
        z += -584.0 * Q.girth2_top40 + 3.2704
    if Q.mass >= 91.2:
        z += 0.00853 * Q.mass - 0.777936
    if Q.mass_over_sum_pt_sq >= 0.026:
        z += -159.0 * Q.mass_over_sum_pt_sq + 4.1339999999999995
    if Q.sum_pt_top40 < 920.0:
        z += -0.00423 * Q.sum_pt_top40 + 3.8916000000000004
    if Q.girth2_top40 < 0.0094 and Q.sum_pt < 1000.0:
        z += 2.97 * (0.0094 - Q.girth2_top40) * (1000.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.08
    if Q.girth2_top5 < 0.021:
        z += 104.0 * Q.girth2_top5 - 2.184
    if Q.mass < 80.4:
        z += -0.05890000000000001 * Q.mass + 3.432720000000001
    if 80.4 <= Q.mass < 120.0:
        z += 0.0329 * Q.mass - 3.948
    if Q.mass >= 150.0:
        z += -0.159 * Q.mass + 23.85
    if Q.mass_top50 >= 140.0:
        z += 0.0992 * Q.mass_top50 - 13.888
    if Q.n_dr_0p2_0p4 < 11.0:
        z += 0.144 * Q.n_dr_0p2_0p4 - 1.5839999999999999
    if Q.sum_pt_top50 < 960.0:
        z += 0.00809 * Q.sum_pt_top50 - 7.7664
    if Q.z_dr_0_0p05 >= 0.77:
        z += -13.2 * Q.z_dr_0_0p05 + 10.164
    if Q.z_dr_0p2_0p4 < 0.088:
        z += -15.2 * Q.z_dr_0p2_0p4 + 1.3376
    if Q.girth2_top5 < 0.021 and Q.sum_pt_top3 < 720.0:
        z += 0.113 * (0.021 - Q.girth2_top5) * (720.0 - Q.sum_pt_top3)
    if Q.mass < 120.0 and Q.tau21 < 0.45:
        z += 0.0916 * (120.0 - Q.mass) * (0.45 - Q.tau21)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.241
    if Q.z_dr_0p2_0p4 < 0.0048:
        z += -206.0 * Q.z_dr_0p2_0p4 + 0.9887999999999999
    if Q.mass < 80.4 and Q.D2 < 3.1:
        z += -0.0477 * (80.4 - Q.mass) * (3.1 - Q.D2)
    if Q.mass < 100.0 and Q.D2 < 1.6:
        z += 0.0641 * (100.0 - Q.mass) * (1.6 - Q.D2)
    if Q.mass < 120.0 and Q.planar_flow < 0.39:
        z += 0.0701 * (120.0 - Q.mass) * (0.39 - Q.planar_flow)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.0851
    if Q.C2 >= 0.055:
        z += -21.1 * Q.C2 + 1.1605
    if Q.mass < 80.4:
        z += -0.0817 * Q.mass + 6.56868
    if Q.mass_top30 >= 130.0:
        z += 0.0392 * Q.mass_top30 - 5.096
    if Q.mass_top50 < 63.0:
        z += 0.0673 * Q.mass_top50 - 4.2399
    if Q.LHA > 0.41 and Q.lam2 < 0.0044:
        z += 17700.0 * (Q.LHA - 0.41) * (0.0044 - Q.lam2)
    return max(0.0, z)


def neuron_13(Q):
    z = 3.1
    if Q.log_sum_pt < 6.8:
        z += -7.700000000000001 * Q.log_sum_pt + 49.33999999999999
    if 6.8 <= Q.log_sum_pt < 7.0:
        z += 15.1 * Q.log_sum_pt - 105.7
    if 140.0 <= Q.mass < 172.8:
        z += -0.0775 * Q.mass + 10.85
    if Q.mass >= 172.8:
        z += 0.006699999999999998 * Q.mass - 3.6997600000000013
    if Q.mass_top10 >= 67.0:
        z += 0.0178 * Q.mass_top10 - 1.1925999999999999
    if Q.sum_pt < 1000.0:
        z += 0.0241 * Q.sum_pt - 24.1
    if Q.sum_pt_top40 < 1000.0:
        z += -0.015 * Q.sum_pt_top40 + 15.0
    return max(0.0, z)


def neuron_14(Q):
    z = 0.0615
    if Q.mass < 91.2:
        z += 0.1048 * Q.mass - 8.2696
    if 91.2 <= Q.mass < 130.0:
        z += -0.0332 * Q.mass + 4.316
    if Q.z_top50_slots < 0.96:
        z += -99.3 * Q.z_top50_slots + 95.32799999999999
    return max(0.0, z)


def neuron_15(Q):
    z = 0.265
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
