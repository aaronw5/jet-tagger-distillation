"""JEDI-linear jet tagger, 8 particles, 3 features: the formula simplified by hand with the training data (main result), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 87.6% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.phi_1                  Δφ of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.centroid_offset        distance of the pT centroid from the jet axis
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[-0.15625, 0.0, 0.34375, 0.0, 0.015625], [0.390625, 0.0, -0.03125, 0.125, 0.0], [0.4296875, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.5, -0.5625, 0.0625], [-0.03125, -0.09375, 0.0, 0.078125, 0.125], [-0.1875, 0.046875, 0.0, 0.015625, -0.25], [0.109375, 0.125, -0.3125, -0.375, 0.0], [0.0, 0.0, 0.21875, 0.46875, 0.0], [0.0, 0.0625, -0.25, 0.0, 0.1875], [0.171875, 0.25390625, -0.03125, -0.03125, 0.0], [0.0, -0.125, 0.0, 0.0, 0.375], [0.0, 0.0, 0.375, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.5], [0.0, 0.0, 0.0703125, 0.0546875, -0.40625], [0.0, 0.0, -0.75, 0.375, 0.0], [0.0, 0.0625, -0.6875, -0.15625, 0.0]]
B = [-0.4375, 0.03125, -0.125, -0.09375, 1.34375]
INT_BITS = [3, 5, 4, 4, 5, 5, 4, 4, 3, 4, 5, 4, 5, 4, 3, 3]
FRAC_BITS = [3, 3, 4, 3, 2, 3, 3, 3, 3, 2, 4, 4, 3, 3, 4, 3]


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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        max_dr=max(dr[i] for i in real),
        pt_0=pt[0],
        pt_7=pt[7],
        z_4=z[4],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        phi_1=phi[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -0.48
    if Q.centroid_offset < 0.0338:
        z += -71.0 * Q.centroid_offset + 2.3998
    if Q.eccentricity >= 0.997:
        z += 499.0 * Q.eccentricity - 497.503
    if Q.girth < 0.0775:
        z += 56.8 * Q.girth - 4.402
    if Q.girth2 < 0.0121:
        z += -301.0 * Q.girth2 + 3.6421
    if Q.mass < 22.5:
        z += 0.5695 * Q.mass - 14.106950000000001
    if 22.5 <= Q.mass < 71.3:
        z += 0.0265 * Q.mass - 1.8894499999999999
    if Q.sum_pt >= 895.0:
        z += -0.0276 * Q.sum_pt + 24.701999999999998
    if Q.width < 0.00449:
        z += 799.0 * Q.width - 3.58751
    if Q.z_dr_0_0p05 >= 0.852:
        z += 15.4 * Q.z_dr_0_0p05 - 13.1208
    if Q.girth2 < 0.0197 and Q.eccentricity > 0.961:
        z += 3910.0 * (0.0197 - Q.girth2) * (Q.eccentricity - 0.961)
    if Q.lam1 < 0.00645 and Q.D2 < 0.886:
        z += -2060.0 * (0.00645 - Q.lam1) * (0.886 - Q.D2)
    if Q.mass < 30.0 and Q.D2 < 0.91:
        z += -2.9 * (30.0 - Q.mass) * (0.91 - Q.D2)
    if Q.mass < 64.8 and Q.centroid_offset > 0.0106:
        z += 2.55 * (64.8 - Q.mass) * (Q.centroid_offset - 0.0106)
    if Q.mass < 29.3 and Q.phi_1 > -0.0771:
        z += -4.03 * (29.3 - Q.mass) * (Q.phi_1 - -0.0771)
    if Q.mass < 64.9 and Q.pt_7 < 40.1:
        z += -0.0022 * (64.9 - Q.mass) * (40.1 - Q.pt_7)
    if Q.planar_flow < 0.148 and Q.sum_pt_top2 < 388.0:
        z += -0.0518 * (0.148 - Q.planar_flow) * (388.0 - Q.sum_pt_top2)
    if Q.sum_pt > 871.0 and Q.pt_7 < 27.8:
        z += 0.00176 * (Q.sum_pt - 871.0) * (27.8 - Q.pt_7)
    if Q.sum_pt_top5 > 655.0 and Q.z_7 > 0.0266:
        z += -0.514 * (Q.sum_pt_top5 - 655.0) * (Q.z_7 - 0.0266)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.2
    if Q.C2 < 0.0491:
        z += 23.7 * Q.C2 - 1.16367
    if Q.e2_sq < 0.00775:
        z += -470.0 * Q.e2_sq + 3.6425
    if Q.log_sum_pt >= 6.41:
        z += 4.58 * Q.log_sum_pt - 29.3578
    if Q.max_dr < 0.0455:
        z += 63.3 * Q.max_dr - 2.88015
    if Q.pt_7 >= 30.4:
        z += 0.115 * Q.pt_7 - 3.496
    if Q.width < 0.009:
        z += 657.0 * Q.width - 5.912999999999999
    if Q.z_7 < 0.0562:
        z += 33.7 * Q.z_7 - 1.8939400000000002
    if Q.e2 < 0.0384 and Q.eccentricity > 0.979:
        z += 6000.0 * (0.0384 - Q.e2) * (Q.eccentricity - 0.979)
    if Q.e2 > 0.031 and Q.tau32 < 0.629:
        z += -82.7 * (Q.e2 - 0.031) * (0.629 - Q.tau32)
    if Q.e2_sq < 0.00778 and Q.planar_flow < 0.0847:
        z += -8390.0 * (0.00778 - Q.e2_sq) * (0.0847 - Q.planar_flow)
    if Q.lam1 < 0.0083 and Q.centroid_offset > 0.021:
        z += 14300.0 * (0.0083 - Q.lam1) * (Q.centroid_offset - 0.021)
    if Q.log_sum_pt > 6.4 and Q.centroid_offset < 0.0235:
        z += -235.0 * (Q.log_sum_pt - 6.4) * (0.0235 - Q.centroid_offset)
    if Q.log_sum_pt > 6.6 and Q.girth2_top3 < 0.0063:
        z += -1090.0 * (Q.log_sum_pt - 6.6) * (0.0063 - Q.girth2_top3)
    if Q.log_sum_pt > 6.59 and Q.lam2 < 0.0012:
        z += 10300.0 * (Q.log_sum_pt - 6.59) * (0.0012 - Q.lam2)
    if Q.log_sum_pt > 6.33 and Q.max_dr < 0.191:
        z += 19.0 * (Q.log_sum_pt - 6.33) * (0.191 - Q.max_dr)
    if Q.log_sum_pt > 6.59 and Q.n_pt_above_50 > 7.01:
        z += -7.05 * (Q.log_sum_pt - 6.59) * (Q.n_pt_above_50 - 7.01)
    if Q.pt_7 > 32.9 and Q.centroid_offset > 0.0144:
        z += 1.53 * (Q.pt_7 - 32.9) * (Q.centroid_offset - 0.0144)
    if Q.pt_7 > 29.2 and Q.mass < 97.0:
        z += -0.00155 * (Q.pt_7 - 29.2) * (97.0 - Q.mass)
    return max(0.0, z)


def neuron_2(Q):
    z = 3.38
    if Q.LHA >= 0.116:
        z += -7.91 * Q.LHA + 0.91756
    if Q.girth2 < 4.87e-05:
        z += 54100.0 * Q.girth2 - 2.63467
    if Q.lam1 < 0.0053:
        z += -267.0 * Q.lam1 + 1.4151
    if Q.log_sum_pt < 6.46:
        z += -4.86 * Q.log_sum_pt + 31.3956
    if 6.83 <= Q.log_sum_pt < 6.89:
        z += 11.4 * Q.log_sum_pt - 77.86200000000001
    if Q.log_sum_pt >= 6.89:
        z += -2.299999999999999 * Q.log_sum_pt + 16.530999999999977
    if Q.planar_flow < 0.654:
        z += 0.791 * Q.planar_flow - 0.517314
    if Q.pt_7 < 30.2:
        z += 0.0503 * Q.pt_7 - 2.72626
    if 30.2 <= Q.pt_7 < 54.2:
        z += 0.2233 * Q.pt_7 - 7.95086
    if Q.pt_7 >= 54.2:
        z += 0.173 * Q.pt_7 - 5.2246
    if Q.sum_pt < 788.0:
        z += -0.00563 * Q.sum_pt + 4.436439999999999
    if Q.z_7 < 0.0185:
        z += 303.0 * Q.z_7 - 5.6055
    if Q.z_7 >= 0.045:
        z += -45.7 * Q.z_7 + 2.0565
    if Q.lam1 < 0.00374 and Q.centroid_offset < 0.00583:
        z += -32100.0 * (0.00374 - Q.lam1) * (0.00583 - Q.centroid_offset)
    if Q.pt_7 > 29.6 and Q.C2 < 0.0514:
        z += -1.91 * (Q.pt_7 - 29.6) * (0.0514 - Q.C2)
    if Q.pt_7 > 29.2 and Q.max_dr > 0.0755:
        z += -0.564 * (Q.pt_7 - 29.2) * (Q.max_dr - 0.0755)
    return max(0.0, z)


def neuron_3(Q):
    z = 2.9
    if Q.centroid_offset >= 0.0103:
        z += 49.3 * Q.centroid_offset - 0.50779
    if Q.e2 < 0.0278:
        z += -179.0 * Q.e2 + 7.750699999999999
    if 0.0278 <= Q.e2 < 0.0433:
        z += -305.0 * Q.e2 + 11.253499999999999
    if Q.e2 >= 0.0433:
        z += -126.0 * Q.e2 + 3.5027999999999997
    if Q.girth2 < 0.00905:
        z += 1274.0 * Q.girth2 - 13.09525
    if 0.00905 <= Q.girth2 < 0.0126:
        z += 441.0 * Q.girth2 - 5.5566
    if Q.mass >= 71.9:
        z += -0.0685 * Q.mass + 4.92515
    if Q.LHA > 0.316 and Q.eccentricity > 0.953:
        z += 1090.0 * (Q.LHA - 0.316) * (Q.eccentricity - 0.953)
    if Q.LHA > 0.308 and Q.max_dr < 0.156:
        z += -1610.0 * (Q.LHA - 0.308) * (0.156 - Q.max_dr)
    if Q.e2 < 0.0435 and Q.z_dr_0p1_0p2 > 0.000595:
        z += -328.0 * (0.0435 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.000595)
    if Q.lam1 > 0.0152 and Q.eccentricity > 0.951:
        z += -12100.0 * (Q.lam1 - 0.0152) * (Q.eccentricity - 0.951)
    if Q.mass > 62.0 and Q.max_dr < 0.176:
        z += 2.06 * (Q.mass - 62.0) * (0.176 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.0682 and Q.dr_7 < 0.0419:
        z += -7530.0 * (Q.mass_over_sum_pt - 0.0682) * (0.0419 - Q.dr_7)
    if Q.mass_over_sum_pt > 0.0909 and Q.max_dr < 0.148:
        z += 9260.0 * (Q.mass_over_sum_pt - 0.0909) * (0.148 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.0616 and Q.n_dr_0_0p05 < 5.99:
        z += 10.8 * (Q.mass_over_sum_pt - 0.0616) * (5.99 - Q.n_dr_0_0p05)
    if Q.mass_over_sum_pt > 0.0749 and Q.tau32 < 0.529:
        z += -133.0 * (Q.mass_over_sum_pt - 0.0749) * (0.529 - Q.tau32)
    return max(0.0, z)


def neuron_4(Q):
    z = 2.33
    if Q.C2 >= 0.00878:
        z += -57.4 * Q.C2 + 0.503972
    if Q.lam2 < 0.000389:
        z += 6470.0 * Q.lam2 - 2.51683
    if Q.mass_over_sum_pt < 0.0885:
        z += -41.6 * Q.mass_over_sum_pt + 4.6176
    if 0.0885 <= Q.mass_over_sum_pt < 0.111:
        z += -215.6 * Q.mass_over_sum_pt + 20.0166
    if Q.mass_over_sum_pt >= 0.111:
        z += -174.0 * Q.mass_over_sum_pt + 15.399
    if Q.sum_pt_top5 < 486.0:
        z += -0.0167 * Q.sum_pt_top5 + 8.1162
    if Q.tau21 < 0.279:
        z += -15.5 * Q.tau21 + 4.3245000000000005
    if Q.width >= 0.00219:
        z += 393.0 * Q.width - 0.86067
    if Q.C2 > 0.0157 and Q.pt_7 > 39.7:
        z += 11.0 * (Q.C2 - 0.0157) * (Q.pt_7 - 39.7)
    if Q.max_dr > 0.109 and Q.pt_7 > 39.2:
        z += -2.35 * (Q.max_dr - 0.109) * (Q.pt_7 - 39.2)
    if Q.tau21 < 0.285 and Q.mass < 66.2:
        z += -0.609 * (0.285 - Q.tau21) * (66.2 - Q.mass)
    if Q.tau21 < 0.243 and Q.planar_flow > 0.0325:
        z += 54.6 * (0.243 - Q.tau21) * (Q.planar_flow - 0.0325)
    if Q.tau21 < 0.246 and Q.pt_7 > 34.2:
        z += 0.67 * (0.246 - Q.tau21) * (Q.pt_7 - 34.2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.465
    if Q.dr_0 < 0.0238:
        z += 169.0 * Q.dr_0 - 4.022200000000001
    if Q.e2 < 0.0374:
        z += -106.0 * Q.e2 + 3.9644000000000004
    if Q.girth2 < 6.4e-05:
        z += -31500.0 * Q.girth2 + 2.016
    if Q.log_sum_pt >= 6.89:
        z += -67.9 * Q.log_sum_pt + 467.831
    if Q.z_7 < 0.0333:
        z += -292.0 * Q.z_7 + 13.4686
    if 0.0333 <= Q.z_7 < 0.0683:
        z += -107.0 * Q.z_7 + 7.3081
    if Q.LHA < 0.221 and Q.log_sum_pt < 6.79:
        z += -90.9 * (0.221 - Q.LHA) * (6.79 - Q.log_sum_pt)
    if Q.e2 < 0.0338 and Q.lam2 < 7.09e-05:
        z += 685000.0 * (0.0338 - Q.e2) * (7.09e-05 - Q.lam2)
    if Q.log_sum_pt > 6.6 and Q.dr_0 < 0.0225:
        z += 757.0 * (Q.log_sum_pt - 6.6) * (0.0225 - Q.dr_0)
    if Q.log_sum_pt > 6.59 and Q.lam1 < 0.0124:
        z += -982.0 * (Q.log_sum_pt - 6.59) * (0.0124 - Q.lam1)
    if Q.sum_pt_top2 < 546.0 and Q.girth2_top3 < 0.00383:
        z += -1.79 * (546.0 - Q.sum_pt_top2) * (0.00383 - Q.girth2_top3)
    if Q.width < 0.00267 and Q.centroid_offset < 0.0234:
        z += 75200.0 * (0.00267 - Q.width) * (0.0234 - Q.centroid_offset)
    if Q.z_7 < 0.0707 and Q.centroid_offset < 0.0301:
        z += -3370.0 * (0.0707 - Q.z_7) * (0.0301 - Q.centroid_offset)
    if Q.z_7 < 0.0684 and Q.sum_pt < 803.0:
        z += -0.278 * (0.0684 - Q.z_7) * (803.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_6(Q):
    z = 6.56
    if Q.C2 >= 0.0101:
        z += -20.8 * Q.C2 + 0.21008
    if Q.centroid_offset >= 0.0212:
        z += -59.4 * Q.centroid_offset + 1.25928
    if Q.e2 < 0.0508:
        z += -85.7 * Q.e2 + 4.35356
    if Q.e2_sq < 0.00328:
        z += -626.0 * Q.e2_sq + 2.05328
    if Q.girth >= 0.0868:
        z += 126.0 * Q.girth - 10.9368
    if Q.girth2 < 0.00862:
        z += 1680.0 * Q.girth2 - 14.481599999999998
    if Q.lam1 < 0.00802:
        z += -919.0 * Q.lam1 + 7.370379999999999
    if Q.lam2 < 0.000518:
        z += 2400.0 * Q.lam2 - 1.2432
    if Q.lam2 >= 0.0029:
        z += -1090.0 * Q.lam2 + 3.1609999999999996
    if Q.mass_over_sum_pt >= 0.00941:
        z += -53.2 * Q.mass_over_sum_pt + 0.5006120000000001
    if Q.max_dr >= 0.0279:
        z += 14.7 * Q.max_dr - 0.41013
    if Q.width < 0.0133:
        z += 540.0 * Q.width - 7.1819999999999995
    if Q.C2 > 0.0103 and Q.pt_7 > 32.6:
        z += 2.56 * (Q.C2 - 0.0103) * (Q.pt_7 - 32.6)
    if Q.centroid_offset > 0.00781 and Q.lam2 < 0.00351:
        z += 19800.0 * (Q.centroid_offset - 0.00781) * (0.00351 - Q.lam2)
    if Q.e2 < 0.0502 and Q.z_dr_0p1_0p2 > 0.159:
        z += 766.0 * (0.0502 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.159)
    if Q.log_sum_pt < 6.57 and Q.pt_7 < 45.4:
        z += 0.343 * (6.57 - Q.log_sum_pt) * (45.4 - Q.pt_7)
    if Q.log_sum_pt < 6.31 and Q.z_7 < 0.0712:
        z += 1090.0 * (6.31 - Q.log_sum_pt) * (0.0712 - Q.z_7)
    if Q.log_sum_pt < 6.71 and Q.z_7 < 0.0492:
        z += 833.0 * (6.71 - Q.log_sum_pt) * (0.0492 - Q.z_7)
    if Q.mass < 46.6 and Q.z_dr_0p05_0p1 < 0.8:
        z += 0.0576 * (46.6 - Q.mass) * (0.8 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.0171 and Q.tau32 < 0.53:
        z += -68.7 * (Q.mass_over_sum_pt - 0.0171) * (0.53 - Q.tau32)
    return max(0.0, z)


def neuron_7(Q):
    z = 4.74
    if Q.centroid_offset < 0.0201:
        z += 37.9 * Q.centroid_offset - 0.76179
    if Q.e2 < 0.0253:
        z += -138.1 * Q.e2 + 4.464969999999999
    if 0.0253 <= Q.e2 < 0.0372:
        z += -81.6 * Q.e2 + 3.0355199999999996
    if Q.e2_sq < 0.00115:
        z += -1320.0 * Q.e2_sq + 1.518
    if Q.girth < 0.089:
        z += 40.9 * Q.girth - 3.6401
    if Q.girth2 < 0.000708:
        z += 2290.0 * Q.girth2 - 1.6213199999999999
    if Q.girth2 >= 0.00445:
        z += -765.0 * Q.girth2 + 3.4042499999999998
    if Q.mass >= 80.4:
        z += -0.206 * Q.mass + 16.5624
    if 0.0729 <= Q.mass_over_sum_pt < 0.0906:
        z += 183.0 * Q.mass_over_sum_pt - 13.340700000000002
    if Q.mass_over_sum_pt >= 0.0906:
        z += -2.0 * Q.mass_over_sum_pt + 3.4202999999999975
    if Q.planar_flow < 0.184:
        z += 7.43 * Q.planar_flow - 1.36712
    if Q.width < 0.00564:
        z += 1070.0 * Q.width - 6.0348
    if Q.centroid_offset < 0.0228 and Q.C2 > 0.0245:
        z += 1100.0 * (0.0228 - Q.centroid_offset) * (Q.C2 - 0.0245)
    if Q.centroid_offset > 0.0311 and Q.pt_0 > 375.0:
        z += -8.88 * (Q.centroid_offset - 0.0311) * (Q.pt_0 - 375.0)
    if Q.e2 < 0.0513 and Q.D2 < 1.1:
        z += 120.0 * (0.0513 - Q.e2) * (1.1 - Q.D2)
    if Q.girth2 > 0.0044 and Q.eccentricity > 0.945:
        z += 10600.0 * (Q.girth2 - 0.0044) * (Q.eccentricity - 0.945)
    if Q.lam1 < 0.00797 and Q.D2 < 1.06:
        z += -754.0 * (0.00797 - Q.lam1) * (1.06 - Q.D2)
    if Q.mass > 80.4 and Q.eccentricity > 0.928:
        z += -2.23 * (Q.mass - 80.4) * (Q.eccentricity - 0.928)
    if Q.planar_flow < 0.179 and Q.sum_pt > 607.0:
        z += 0.0407 * (0.179 - Q.planar_flow) * (Q.sum_pt - 607.0)
    if Q.planar_flow < 0.205 and Q.width > 0.00768:
        z += -3780.0 * (0.205 - Q.planar_flow) * (Q.width - 0.00768)
    if Q.pt_7 < 44.5 and Q.planar_flow < 0.596:
        z += -0.0953 * (44.5 - Q.pt_7) * (0.596 - Q.planar_flow)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.104
    if Q.centroid_offset < 0.00366:
        z += -477.0 * Q.centroid_offset + 1.74582
    if Q.log_sum_pt >= 6.71:
        z += -11.6 * Q.log_sum_pt + 77.836
    if Q.pt_7 >= 37.1:
        z += -0.0426 * Q.pt_7 + 1.58046
    if Q.LHA < 0.198 and Q.lam2 < 0.000321:
        z += -169000.0 * (0.198 - Q.LHA) * (0.000321 - Q.lam2)
    if Q.e2 < 0.0218 and Q.sum_pt > 830.0:
        z += 0.561 * (0.0218 - Q.e2) * (Q.sum_pt - 830.0)
    if Q.girth < 0.0603 and Q.lam1 > 0.00119:
        z += -32700.0 * (0.0603 - Q.girth) * (Q.lam1 - 0.00119)
    if Q.girth2 < 0.00621 and Q.planar_flow < 0.453:
        z += 1240.0 * (0.00621 - Q.girth2) * (0.453 - Q.planar_flow)
    if Q.mass < 20.2 and Q.centroid_offset > 0.0163:
        z += -15.2 * (20.2 - Q.mass) * (Q.centroid_offset - 0.0163)
    if Q.width < 0.00544 and Q.centroid_offset > 0.00598:
        z += -76700.0 * (0.00544 - Q.width) * (Q.centroid_offset - 0.00598)
    if Q.width < 0.00506 and Q.z_dr_0p2_0p4 < 0.0996:
        z += 12900.0 * (0.00506 - Q.width) * (0.0996 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.22
    if Q.centroid_offset < 0.0174:
        z += -170.0 * Q.centroid_offset + 2.9579999999999997
    if Q.lam2 >= 0.00136:
        z += 1500.0 * Q.lam2 - 2.04
    if Q.log_sum_pt >= 6.36:
        z += -4.6 * Q.log_sum_pt + 29.256
    if Q.mass < 31.3:
        z += 0.136 * Q.mass - 4.2568
    if Q.max_dr < 0.257:
        z += -14.7 * Q.max_dr + 3.7779
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 1.23 * Q.n_dr_0p2_0p4 - 1.23
    if Q.width < 0.00662:
        z += -1550.0 * Q.width + 10.261
    if Q.centroid_offset < 0.019 and Q.z_4 > 0.031:
        z += -1420.0 * (0.019 - Q.centroid_offset) * (Q.z_4 - 0.031)
    if Q.girth2 > 0.0184 and Q.pt_7 < 29.8:
        z += 59.4 * (Q.girth2 - 0.0184) * (29.8 - Q.pt_7)
    if Q.mass < 51.1 and Q.centroid_offset < 0.0241:
        z += 5.63 * (51.1 - Q.mass) * (0.0241 - Q.centroid_offset)
    if Q.mass < 49.9 and Q.log_sum_pt < 6.79:
        z += 0.208 * (49.9 - Q.mass) * (6.79 - Q.log_sum_pt)
    if Q.width < 0.00676 and Q.centroid_offset > 0.00285:
        z += -43400.0 * (0.00676 - Q.width) * (Q.centroid_offset - 0.00285)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.73
    if Q.C2 >= 0.0596:
        z += 27.9 * Q.C2 - 1.6628399999999999
    if Q.e2 < 0.0458:
        z += 70.6 * Q.e2 - 3.29702
    if 0.0458 <= Q.e2 < 0.0467:
        z += 131.89999999999998 * Q.e2 - 6.104559999999999
    if Q.e2 >= 0.0467:
        z += 61.3 * Q.e2 - 2.80754
    if Q.girth2 < 0.00182:
        z += 1260.0 * Q.girth2 - 2.2932
    if Q.lam2 >= 0.000227:
        z += 514.0 * Q.lam2 - 0.11667799999999999
    if Q.log_sum_pt < 6.24:
        z += 3.17 * Q.log_sum_pt - 19.7808
    if Q.log_sum_pt >= 6.69:
        z += -6.91 * Q.log_sum_pt + 46.227900000000005
    if Q.mass >= 9.7:
        z += 0.0236 * Q.mass - 0.22891999999999998
    if Q.mass_over_sum_pt < 0.0966:
        z += -31.9 * Q.mass_over_sum_pt + 3.08154
    if Q.n_dr_0p05_0p1 < 2.95:
        z += -0.324 * Q.n_dr_0p05_0p1 + 0.9558000000000001
    if Q.n_dr_0p2_0p4 >= 1.56:
        z += 0.519 * Q.n_dr_0p2_0p4 - 0.80964
    if Q.tau32 < 0.271:
        z += -12.1 * Q.tau32 + 3.2791
    if Q.LHA > 0.301 and Q.tau21 < 0.699:
        z += -42.4 * (Q.LHA - 0.301) * (0.699 - Q.tau21)
    if Q.eccentricity > 0.901 and Q.z_dr_0p2_0p4 < 0.0539:
        z += 259.0 * (Q.eccentricity - 0.901) * (0.0539 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.004 and Q.log_sum_pt > 6.69:
        z += 3650.0 * (0.004 - Q.lam1) * (Q.log_sum_pt - 6.69)
    if Q.lam1 < 0.00395 and Q.sum_pt > 989.0:
        z += -9.18 * (0.00395 - Q.lam1) * (Q.sum_pt - 989.0)
    if Q.lam2 > 8.42e-05 and Q.tau21 < 0.518:
        z += 2350.0 * (Q.lam2 - 8.42e-05) * (0.518 - Q.tau21)
    if Q.tau32 < 0.29 and Q.n_dr_0p2_0p4 < 2.0:
        z += -4.11 * (0.29 - Q.tau32) * (2.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.15
    if Q.C2 < 0.0355:
        z += 19.1 * Q.C2 - 0.67805
    if Q.LHA < 0.157:
        z += 22.8 * Q.LHA - 3.5796
    if Q.centroid_offset < 0.0144:
        z += -101.0 * Q.centroid_offset + 5.0298
    if 0.0144 <= Q.centroid_offset < 0.0498:
        z += -174.1 * Q.centroid_offset + 6.08244
    if Q.centroid_offset >= 0.0498:
        z += -73.1 * Q.centroid_offset + 1.0526399999999998
    if Q.girth < 0.0766:
        z += 84.9 * Q.girth - 7.496670000000001
    if 0.0766 <= Q.girth < 0.0883:
        z += 3.4000000000000057 * Q.girth - 1.2537700000000003
    if Q.girth >= 0.0883:
        z += -81.5 * Q.girth + 6.242900000000001
    if Q.planar_flow < 0.271:
        z += -8.98 * Q.planar_flow + 2.43358
    if Q.sum_pt_top5 < 688.0:
        z += -0.00545 * Q.sum_pt_top5 + 3.7496
    if Q.width < 0.00365:
        z += -556.0 * Q.width + 7.432899999999999
    if 0.00365 <= Q.width < 0.0087:
        z += -1070.0 * Q.width + 9.309
    if Q.centroid_offset < 0.0437 and Q.log_sum_pt < 6.84:
        z += -101.0 * (0.0437 - Q.centroid_offset) * (6.84 - Q.log_sum_pt)
    if Q.girth > 0.0771 and Q.n_pt_above_50 < 7.47:
        z += 12.6 * (Q.girth - 0.0771) * (7.47 - Q.n_pt_above_50)
    if Q.planar_flow < 0.254 and Q.mass < 66.7:
        z += -0.128 * (0.254 - Q.planar_flow) * (66.7 - Q.mass)
    if Q.planar_flow < 0.281 and Q.max_dr > 0.114:
        z += -57.9 * (0.281 - Q.planar_flow) * (Q.max_dr - 0.114)
    if Q.planar_flow < 0.217 and Q.width > 0.0053:
        z += -1310.0 * (0.217 - Q.planar_flow) * (Q.width - 0.0053)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.91
    if Q.e2 >= 0.0622:
        z += -124.0 * Q.e2 + 7.7128
    if Q.girth2 >= 0.0188:
        z += 220.0 * Q.girth2 - 4.136
    if Q.mass >= 91.2:
        z += 0.0778 * Q.mass - 7.0953599999999994
    if Q.girth2 > 0.0186 and Q.lam2 > 0.000489:
        z += 15300.0 * (Q.girth2 - 0.0186) * (Q.lam2 - 0.000489)
    if Q.girth2 > 0.0197 and Q.pt_7 > 16.1:
        z += 6.92 * (Q.girth2 - 0.0197) * (Q.pt_7 - 16.1)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.68
    if Q.C2 >= 0.0664:
        z += -52.4 * Q.C2 + 3.47936
    if Q.e2 < 0.0509:
        z += 88.6 * Q.e2 - 4.50974
    if Q.girth < 0.15:
        z += -67.0 * Q.girth + 10.049999999999999
    if Q.pt_0 >= 179.0:
        z += 0.0022 * Q.pt_0 - 0.39380000000000004
    if Q.pt_7 < 24.8:
        z += 0.246 * Q.pt_7 - 6.1008000000000004
    if Q.sum_pt >= 998.0:
        z += -0.0241 * Q.sum_pt + 24.0518
    if Q.width < 0.0159:
        z += -217.0 * Q.width + 3.4503000000000004
    if Q.girth < 0.156 and Q.log_sum_pt < 6.91:
        z += -41.1 * (0.156 - Q.girth) * (6.91 - Q.log_sum_pt)
    if Q.girth < 0.148 and Q.pt_7 < 39.5:
        z += -1.09 * (0.148 - Q.girth) * (39.5 - Q.pt_7)
    if Q.sum_pt > 1070.0 and Q.n_pt_above_50 > 6.01:
        z += 0.019 * (Q.sum_pt - 1070.0) * (Q.n_pt_above_50 - 6.01)
    if Q.sum_pt < 764.0 and Q.z_4 < 0.0379:
        z += 6.47 * (764.0 - Q.sum_pt) * (0.0379 - Q.z_4)
    if Q.sum_pt_top5 > 653.0 and Q.pt_7 < 42.6:
        z += 0.000594 * (Q.sum_pt_top5 - 653.0) * (42.6 - Q.pt_7)
    if Q.sum_pt_top5 > 677.0 and Q.z_7 > 0.0289:
        z += -0.566 * (Q.sum_pt_top5 - 677.0) * (Q.z_7 - 0.0289)
    if Q.tau21 < 0.518 and Q.max_dr > -0.0382:
        z += -14.9 * (0.518 - Q.tau21) * (Q.max_dr - -0.0382)
    return max(0.0, z)


def neuron_14(Q):
    z = -1.51
    if Q.e2 < 0.0385:
        z += -106.0 * Q.e2 + 4.0809999999999995
    if Q.girth < 0.0869:
        z += 80.1 * Q.girth - 6.96069
    if Q.girth2 < 0.0131:
        z += -632.0 * Q.girth2 + 8.2792
    if Q.lam1 >= 0.00732:
        z += -101.0 * Q.lam1 + 0.73932
    if Q.mass >= 5.61:
        z += 0.0187 * Q.mass - 0.10490700000000001
    if Q.max_dr < 0.177:
        z += 20.1 * Q.max_dr - 3.5577
    if Q.width < 0.00741:
        z += 1040.0 * Q.width - 7.7064
    if Q.z_dr_0p05_0p1 < 0.572:
        z += -1.85 * Q.z_dr_0p05_0p1 + 1.0582
    if Q.D2 < 1.04 and Q.centroid_offset < 0.0307:
        z += 38.5 * (1.04 - Q.D2) * (0.0307 - Q.centroid_offset)
    if Q.e2 < 0.0382 and Q.D2 < 0.961:
        z += 192.0 * (0.0382 - Q.e2) * (0.961 - Q.D2)
    if Q.girth2 < 0.0135 and Q.eccentricity > 0.971:
        z += 11600.0 * (0.0135 - Q.girth2) * (Q.eccentricity - 0.971)
    if Q.girth2 < 0.00453 and Q.n_dr_0p2_0p4 < 0.93:
        z += -486.0 * (0.00453 - Q.girth2) * (0.93 - Q.n_dr_0p2_0p4)
    if Q.lam1 > 0.0055 and Q.max_dr < 0.168:
        z += 6850.0 * (Q.lam1 - 0.0055) * (0.168 - Q.max_dr)
    if Q.lam1 > 0.00727 and Q.max_dr < 0.134:
        z += 43000.0 * (Q.lam1 - 0.00727) * (0.134 - Q.max_dr)
    if Q.mass < 75.3 and Q.D2 < 0.865:
        z += -0.0486 * (75.3 - Q.mass) * (0.865 - Q.D2)
    if Q.planar_flow < 0.11 and Q.centroid_offset > 0.00946:
        z += 1270.0 * (0.11 - Q.planar_flow) * (Q.centroid_offset - 0.00946)
    if Q.planar_flow < 0.116 and Q.centroid_offset > 0.0186:
        z += -1310.0 * (0.116 - Q.planar_flow) * (Q.centroid_offset - 0.0186)
    if Q.planar_flow < 0.116 and Q.max_dr < 0.164:
        z += -187.0 * (0.116 - Q.planar_flow) * (0.164 - Q.max_dr)
    if Q.planar_flow < 0.106 and Q.sum_pt < 740.0:
        z += -0.0507 * (0.106 - Q.planar_flow) * (740.0 - Q.sum_pt)
    if Q.width < 0.00857 and Q.D2 < 1.04:
        z += -986.0 * (0.00857 - Q.width) * (1.04 - Q.D2)
    if Q.width < 0.00793 and Q.log_sum_pt < 6.82:
        z += 381.0 * (0.00793 - Q.width) * (6.82 - Q.log_sum_pt)
    if Q.width < 0.00758 and Q.planar_flow < 0.109:
        z += -6890.0 * (0.00758 - Q.width) * (0.109 - Q.planar_flow)
    if Q.z_dr_0p05_0p1 < 0.597 and Q.C2 < 0.0662:
        z += -77.0 * (0.597 - Q.z_dr_0p05_0p1) * (0.0662 - Q.C2)
    return max(0.0, z)


def neuron_15(Q):
    z = 0.951
    if Q.D2 < 0.711:
        z += 1.98 * Q.D2 - 1.4077799999999998
    if Q.LHA >= 0.342:
        z += -54.8 * Q.LHA + 18.741600000000002
    if Q.e2 < 0.0244:
        z += -303.0 * Q.e2 + 9.1196
    if 0.0244 <= Q.e2 < 0.041:
        z += -104.0 * Q.e2 + 4.264
    if Q.girth2_top3 < 0.00198:
        z += 665.0 * Q.girth2_top3 - 1.3167
    if Q.lam1 < 0.00679:
        z += -820.0 * Q.lam1 + 4.509200000000001
    if 0.00679 <= Q.lam1 < 0.00813:
        z += 790.0 * Q.lam1 - 6.4227
    if Q.log_sum_pt >= 6.9:
        z += 37.6 * Q.log_sum_pt - 259.44
    if Q.mass < 37.4:
        z += 0.0463 * Q.mass - 1.73162
    if Q.n_dr_0_0p05 < 2.12:
        z += -0.454 * Q.n_dr_0_0p05 + 0.9624800000000001
    if Q.width < 0.00695:
        z += 2070.0 * Q.width - 14.3865
    if Q.z_dr_0p05_0p1 >= 0.757:
        z += -6.11 * Q.z_dr_0p05_0p1 + 4.62527
    if Q.z_dr_0p1_0p2 < 0.342:
        z += -3.61 * Q.z_dr_0p1_0p2 + 1.23462
    if Q.LHA > 0.186 and Q.sum_pt_top3 > 338.0:
        z += 0.0311 * (Q.LHA - 0.186) * (Q.sum_pt_top3 - 338.0)
    if Q.mass > 80.4 and Q.z_dr_0p2_0p4 < 0.236:
        z += -0.572 * (Q.mass - 80.4) * (0.236 - Q.z_dr_0p2_0p4)
    if Q.tau21 < 0.26 and Q.z_dr_0p05_0p1 < 0.513:
        z += -9.02 * (0.26 - Q.tau21) * (0.513 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.24 and Q.z_dr_0p2_0p4 < 0.21:
        z += 62.2 * (0.24 - Q.tau21) * (0.21 - Q.z_dr_0p2_0p4)
    if Q.width < 0.00788 and Q.e2 > 0.0243:
        z += -35800.0 * (0.00788 - Q.width) * (Q.e2 - 0.0243)
    if Q.width < 0.00614 and Q.log_sum_pt > 6.9:
        z += -7700.0 * (0.00614 - Q.width) * (Q.log_sum_pt - 6.9)
    if Q.width < 0.00868 and Q.planar_flow < 0.0755:
        z += -3420.0 * (0.00868 - Q.width) * (0.0755 - Q.planar_flow)
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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
