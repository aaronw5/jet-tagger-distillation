"""JEDI-linear jet tagger, 8 particles, 3 features: the formula with the fewest quantities (24) at the main result's accuracy (from the 931-term tuned formula), as if-statements, with each class score (logit) written out as a formula.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 87.5% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
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
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        pt_7=pt[7],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    z = -0.915
    if Q.D2 >= 3.7:
        z += -0.856 * Q.D2 + 3.1672000000000002
    if Q.e2 < 0.0252:
        z += -97.4 * Q.e2 + 2.45448
    if Q.eccentricity >= 0.997:
        z += 367.0 * Q.eccentricity - 365.899
    if Q.girth < 0.0768:
        z += 54.5 * Q.girth - 4.1856
    if Q.girth2 < 0.0128:
        z += -517.0 * Q.girth2 + 6.6176
    if Q.lam1 < 0.000706:
        z += -1830.0 * Q.lam1 - 2.2209000000000003
    if 0.000706 <= Q.lam1 < 0.00142:
        z += 4920.0 * Q.lam1 - 6.986400000000001
    if Q.mass < 29.3:
        z += 0.26539999999999997 * Q.mass - 9.496419999999999
    if 29.3 <= Q.mass < 59.8:
        z += 0.0564 * Q.mass - 3.3727199999999997
    if 816.0 <= Q.sum_pt < 890.0:
        z += -0.0159 * Q.sum_pt + 12.974400000000001
    if Q.sum_pt >= 890.0:
        z += -0.032 * Q.sum_pt + 27.303400000000003
    if Q.sum_pt_top5 >= 699.0:
        z += 0.0131 * Q.sum_pt_top5 - 9.1569
    if Q.width < 0.00459:
        z += 855.0 * Q.width - 3.92445
    if Q.lam1 < 0.00646 and Q.D2 < 0.866:
        z += -2050.0 * (0.00646 - Q.lam1) * (0.866 - Q.D2)
    if Q.mass < 56.8 and Q.C2 > 0.0184:
        z += 1.65 * (56.8 - Q.mass) * (Q.C2 - 0.0184)
    if Q.mass < 67.8 and Q.centroid_offset > 0.0133:
        z += 1.12 * (67.8 - Q.mass) * (Q.centroid_offset - 0.0133)
    if Q.mass < 64.6 and Q.pt_7 < 38.6:
        z += -0.00156 * (64.6 - Q.mass) * (38.6 - Q.pt_7)
    if Q.planar_flow < 0.158 and Q.centroid_offset < 0.0558:
        z += 272.0 * (0.158 - Q.planar_flow) * (0.0558 - Q.centroid_offset)
    if Q.sum_pt > 901.0 and Q.pt_7 > 25.9:
        z += -0.000688 * (Q.sum_pt - 901.0) * (Q.pt_7 - 25.9)
    if Q.sum_pt > 905.0 and Q.pt_7 < 26.0:
        z += 0.00226 * (Q.sum_pt - 905.0) * (26.0 - Q.pt_7)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.09
    if Q.LHA >= 0.282:
        z += 13.9 * Q.LHA - 3.9197999999999995
    if Q.e2 < 0.00749:
        z += 161.0 * Q.e2 - 1.2058900000000001
    if Q.e2 >= 0.0316:
        z += -41.2 * Q.e2 + 1.3019200000000002
    if Q.log_sum_pt >= 6.4:
        z += 3.28 * Q.log_sum_pt - 20.992
    if Q.max_dr < 0.252:
        z += 4.55 * Q.max_dr - 1.1466
    if Q.pt_7 >= 33.5:
        z += 0.0932 * Q.pt_7 - 3.1222000000000003
    if Q.z_7 < 0.0515:
        z += 63.3 * Q.z_7 - 3.2599499999999995
    if Q.C2 < 0.0389 and Q.tau21 < 0.263:
        z += -191.0 * (0.0389 - Q.C2) * (0.263 - Q.tau21)
    if Q.e2 < 0.0346 and Q.eccentricity > 0.979:
        z += 3750.0 * (0.0346 - Q.e2) * (Q.eccentricity - 0.979)
    if Q.lam1 < 0.00989 and Q.centroid_offset > 0.0201:
        z += 14200.0 * (0.00989 - Q.lam1) * (Q.centroid_offset - 0.0201)
    if Q.lam1 < 0.00911 and Q.z_7 > 0.0272:
        z += -5070.0 * (0.00911 - Q.lam1) * (Q.z_7 - 0.0272)
    if Q.log_sum_pt > 6.55 and Q.D2 < 1.2:
        z += 7.17 * (Q.log_sum_pt - 6.55) * (1.2 - Q.D2)
    if Q.log_sum_pt > 6.47 and Q.centroid_offset < 0.0272:
        z += -173.0 * (Q.log_sum_pt - 6.47) * (0.0272 - Q.centroid_offset)
    if Q.log_sum_pt > 6.61 and Q.lam2 < 0.00151:
        z += 4690.0 * (Q.log_sum_pt - 6.61) * (0.00151 - Q.lam2)
    if Q.log_sum_pt > 6.36 and Q.max_dr < 0.195:
        z += 12.4 * (Q.log_sum_pt - 6.36) * (0.195 - Q.max_dr)
    if Q.pt_7 > 35.4 and Q.centroid_offset > 0.0188:
        z += 1.56 * (Q.pt_7 - 35.4) * (Q.centroid_offset - 0.0188)
    if Q.tau21 < 0.223 and Q.lam2 < 0.000218:
        z += 29600.0 * (0.223 - Q.tau21) * (0.000218 - Q.lam2)
    if Q.width < 0.00946 and Q.planar_flow < 0.0841:
        z += -5320.0 * (0.00946 - Q.width) * (0.0841 - Q.planar_flow)
    return max(0.0, z)


def neuron_2(Q):
    z = 5.16
    z += -9.84 * Q.LHA
    if Q.girth2 < 4.88e-05:
        z += 34000.0 * Q.girth2 - 1.6592
    if Q.lam1 < 0.00591:
        z += -377.0 * Q.lam1 + 2.22807
    if Q.log_sum_pt < 6.43:
        z += -3.62 * Q.log_sum_pt + 23.2766
    if Q.log_sum_pt >= 6.93:
        z += -9.11 * Q.log_sum_pt + 63.132299999999994
    if Q.mass < 36.7:
        z += 0.125 * Q.mass - 4.5875
    if Q.planar_flow < 0.748:
        z += 1.21 * Q.planar_flow - 0.90508
    if Q.pt_7 < 31.0:
        z += 0.0366 * Q.pt_7 - 1.9764
    if 31.0 <= Q.pt_7 < 54.0:
        z += 0.1344 * Q.pt_7 - 5.0082
    if Q.pt_7 >= 54.0:
        z += 0.0978 * Q.pt_7 - 3.0318
    if Q.sum_pt < 794.0:
        z += -0.00873 * Q.sum_pt + 6.93162
    if Q.sum_pt_top5 < 714.0:
        z += 0.00532 * Q.sum_pt_top5 - 3.79848
    if Q.lam1 < 0.00343 and Q.centroid_offset < 0.0071:
        z += -30100.0 * (0.00343 - Q.lam1) * (0.0071 - Q.centroid_offset)
    if Q.lam1 < 0.00581 and Q.max_dr > 0.0819:
        z += -2440.0 * (0.00581 - Q.lam1) * (Q.max_dr - 0.0819)
    if Q.mass < 38.0 and Q.lam2 < 0.00111:
        z += 84.6 * (38.0 - Q.mass) * (0.00111 - Q.lam2)
    if Q.mass < 73.7 and Q.max_pair_mass > 12.4:
        z += -0.00156 * (73.7 - Q.mass) * (Q.max_pair_mass - 12.4)
    if Q.mass < 72.6 and Q.z_7 < 0.0637:
        z += -0.455 * (72.6 - Q.mass) * (0.0637 - Q.z_7)
    if Q.pt_7 > 31.2 and Q.C2 < 0.05:
        z += -2.27 * (Q.pt_7 - 31.2) * (0.05 - Q.C2)
    if Q.pt_7 > 28.1 and Q.centroid_offset > 0.0132:
        z += -1.07 * (Q.pt_7 - 28.1) * (Q.centroid_offset - 0.0132)
    if Q.pt_7 > 29.5 and Q.max_dr > 0.111:
        z += -0.513 * (Q.pt_7 - 29.5) * (Q.max_dr - 0.111)
    if Q.z_7 < 0.0301 and Q.width < 0.00468:
        z += -11500.0 * (0.0301 - Q.z_7) * (0.00468 - Q.width)
    return max(0.0, z)


def neuron_3(Q):
    z = 7.47
    if Q.C2 >= 0.0894:
        z += -194.0 * Q.C2 + 17.3436
    if Q.LHA >= 0.326:
        z += 105.0 * Q.LHA - 34.230000000000004
    if 0.0106 <= Q.centroid_offset < 0.0495:
        z += 46.4 * Q.centroid_offset - 0.49184
    if Q.centroid_offset >= 0.0495:
        z += -130.6 * Q.centroid_offset + 8.26966
    if Q.e2 < 0.0263:
        z += -110.0 * Q.e2 + 4.367
    if 0.0263 <= Q.e2 < 0.0397:
        z += -250.0 * Q.e2 + 8.049
    if Q.e2 >= 0.0397:
        z += -140.0 * Q.e2 + 3.682
    if Q.girth < 0.121:
        z += 144.0 * Q.girth - 17.424
    if Q.girth2 < 0.00863:
        z += 856.0 * Q.girth2 - 7.3872800000000005
    if Q.mass >= 75.6:
        z += -0.295 * Q.mass + 22.301999999999996
    if Q.max_dr >= 0.122:
        z += 18.1 * Q.max_dr - 2.2082
    if Q.width < 0.00539:
        z += -1170.0 * Q.width + 6.3063
    if Q.width >= 0.0185:
        z += 1730.0 * Q.width - 32.004999999999995
    if Q.LHA > 0.313 and Q.eccentricity > 0.957:
        z += 2440.0 * (Q.LHA - 0.313) * (Q.eccentricity - 0.957)
    if Q.LHA > 0.312 and Q.max_dr < 0.149:
        z += -4360.0 * (Q.LHA - 0.312) * (0.149 - Q.max_dr)
    if Q.LHA > 0.322 and Q.planar_flow > 0.00227:
        z += -116.0 * (Q.LHA - 0.322) * (Q.planar_flow - 0.00227)
    if Q.LHA > 0.424 and Q.pt_7 > 38.5:
        z += -33.4 * (Q.LHA - 0.424) * (Q.pt_7 - 38.5)
    if Q.e2 > 0.0319 and Q.n_pt_above_50 > 1.68:
        z += -15.0 * (Q.e2 - 0.0319) * (Q.n_pt_above_50 - 1.68)
    if Q.lam1 > 0.0147 and Q.eccentricity > 0.957:
        z += 13800.0 * (Q.lam1 - 0.0147) * (Q.eccentricity - 0.957)
    if Q.lam1 > 0.0088 and Q.pt_7 < 36.3:
        z += 123.0 * (Q.lam1 - 0.0088) * (36.3 - Q.pt_7)
    if Q.lam1 > 0.00553 and Q.z_7 < 0.0813:
        z += -13600.0 * (Q.lam1 - 0.00553) * (0.0813 - Q.z_7)
    if Q.mass > 35.4 and Q.eccentricity > 0.705:
        z += 0.228 * (Q.mass - 35.4) * (Q.eccentricity - 0.705)
    if Q.mass > 75.1 and Q.max_dr < 0.185:
        z += 4.98 * (Q.mass - 75.1) * (0.185 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.0901 and Q.max_dr < 0.147:
        z += 9190.0 * (Q.mass_over_sum_pt - 0.0901) * (0.147 - Q.max_dr)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.85
    if Q.C2 < 0.0602:
        z += -38.1 * Q.C2 + 2.2936199999999998
    if Q.C2 >= 0.0604:
        z += -141.0 * Q.C2 + 8.5164
    if Q.centroid_offset < 0.0429:
        z += 34.7 * Q.centroid_offset - 1.4886300000000001
    if Q.e2 >= 0.0328:
        z += 105.0 * Q.e2 - 3.4440000000000004
    if Q.girth >= 0.0651:
        z += -77.1 * Q.girth + 5.01921
    if Q.mass < 47.9:
        z += -0.0876 * Q.mass + 4.19604
    if 0.0906 <= Q.mass_over_sum_pt < 0.109:
        z += -284.0 * Q.mass_over_sum_pt + 25.7304
    if Q.mass_over_sum_pt >= 0.109:
        z += -174.0 * Q.mass_over_sum_pt + 13.7404
    if Q.sum_pt < 762.0:
        z += -0.0071 * Q.sum_pt + 5.410200000000001
    if Q.sum_pt_top5 < 439.0:
        z += -0.0187 * Q.sum_pt_top5 + 8.2093
    if Q.tau21 < 0.238:
        z += -28.1 * Q.tau21 + 6.6878
    if Q.width >= 0.000872:
        z += 447.0 * Q.width - 0.389784
    if Q.C2 > 0.00455 and Q.pt_7 > 38.6:
        z += 4.37 * (Q.C2 - 0.00455) * (Q.pt_7 - 38.6)
    if Q.C2 > 0.067 and Q.pt_7 < 36.4:
        z += -12.3 * (Q.C2 - 0.067) * (36.4 - Q.pt_7)
    if Q.max_dr > 0.11 and Q.eccentricity > 0.982:
        z += -1520.0 * (Q.max_dr - 0.11) * (Q.eccentricity - 0.982)
    if Q.tau21 < 0.278 and Q.mass < 63.5:
        z += -0.599 * (0.278 - Q.tau21) * (63.5 - Q.mass)
    if Q.tau21 < 0.294 and Q.planar_flow > 0.00357:
        z += 44.4 * (0.294 - Q.tau21) * (Q.planar_flow - 0.00357)
    if Q.tau21 < 0.267 and Q.pt_7 < 26.3:
        z += -1.24 * (0.267 - Q.tau21) * (26.3 - Q.pt_7)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.638
    if Q.e2 < 0.0405:
        z += -81.9 * Q.e2 + 3.3169500000000003
    if Q.girth2 < 5.65e-05:
        z += -62400.0 * Q.girth2 + 3.5256
    if Q.pt_7 < 54.4:
        z += -0.122 * Q.pt_7 + 6.6368
    if Q.sum_pt >= 935.0:
        z += -0.0465 * Q.sum_pt + 43.4775
    if Q.sum_pt_top5 < 721.0:
        z += 0.00508 * Q.sum_pt_top5 - 3.6626800000000004
    if Q.z_7 < 0.0243:
        z += -169.0 * Q.z_7 + 4.1067
    if Q.LHA < 0.16 and Q.centroid_offset > 0.00367:
        z += -4360.0 * (0.16 - Q.LHA) * (Q.centroid_offset - 0.00367)
    if Q.LHA < 0.217 and Q.log_sum_pt < 6.83:
        z += -121.0 * (0.217 - Q.LHA) * (6.83 - Q.log_sum_pt)
    if Q.log_sum_pt > 6.27 and Q.centroid_offset > 0.00241:
        z += 125.0 * (Q.log_sum_pt - 6.27) * (Q.centroid_offset - 0.00241)
    if Q.log_sum_pt > 6.58 and Q.centroid_offset > 0.0151:
        z += -580.0 * (Q.log_sum_pt - 6.58) * (Q.centroid_offset - 0.0151)
    if Q.sum_pt > 864.0 and Q.centroid_offset < 0.0114:
        z += 1.57 * (Q.sum_pt - 864.0) * (0.0114 - Q.centroid_offset)
    if Q.width < 0.00234 and Q.centroid_offset < 0.0257:
        z += 78100.0 * (0.00234 - Q.width) * (0.0257 - Q.centroid_offset)
    if Q.z_7 < 0.072 and Q.centroid_offset < 0.0298:
        z += -2690.0 * (0.072 - Q.z_7) * (0.0298 - Q.centroid_offset)
    if Q.z_7 < 0.076 and Q.sum_pt < 803.0:
        z += -0.258 * (0.076 - Q.z_7) * (803.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.53
    if Q.C2 >= 0.0419:
        z += -24.3 * Q.C2 + 1.01817
    if Q.LHA >= 0.313:
        z += 32.6 * Q.LHA - 10.203800000000001
    if Q.centroid_offset >= 0.0525:
        z += 88.1 * Q.centroid_offset - 4.625249999999999
    if Q.e2 < 0.0508:
        z += -101.0 * Q.e2 + 5.1308
    if Q.girth2 < 0.00371:
        z += 1098.0 * Q.girth2 - 12.373479999999999
    if 0.00371 <= Q.girth2 < 0.00868:
        z += 1670.0 * Q.girth2 - 14.4956
    if Q.lam1 < 0.00819:
        z += -1120.0 * Q.lam1 + 9.172799999999999
    if Q.lam2 >= 0.000765:
        z += -935.0 * Q.lam2 + 0.715275
    if Q.log_sum_pt < 6.33:
        z += -0.9100000000000001 * Q.log_sum_pt + 5.054400000000001
    if 6.33 <= Q.log_sum_pt < 6.72:
        z += 1.81 * Q.log_sum_pt - 12.1632
    if Q.mass_over_sum_pt < 0.132:
        z += -76.8 * Q.mass_over_sum_pt + 10.1376
    if Q.max_dr < 0.0452:
        z += 26.1 * Q.max_dr - 1.1797199999999999
    if Q.max_dr >= 0.0866:
        z += 14.0 * Q.max_dr - 1.2124
    if Q.planar_flow < 0.0747:
        z += 9.51 * Q.planar_flow - 0.7103970000000001
    if Q.sum_pt >= 997.0:
        z += 0.0125 * Q.sum_pt - 12.4625
    if Q.width < 0.0135:
        z += 843.0 * Q.width - 11.3805
    if Q.C2 > 0.00343 and Q.pt_7 > 31.0:
        z += 1.59 * (Q.C2 - 0.00343) * (Q.pt_7 - 31.0)
    if Q.centroid_offset > 0.00712 and Q.C2 < 0.0973:
        z += 515.0 * (Q.centroid_offset - 0.00712) * (0.0973 - Q.C2)
    if Q.e2 < 0.0502 and Q.z_dr_0p1_0p2 > 0.159:
        z += 1150.0 * (0.0502 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.159)
    if Q.log_sum_pt < 6.62 and Q.pt_7 < 46.1:
        z += 0.376 * (6.62 - Q.log_sum_pt) * (46.1 - Q.pt_7)
    if Q.log_sum_pt < 6.32 and Q.z_7 < 0.0711:
        z += 1040.0 * (6.32 - Q.log_sum_pt) * (0.0711 - Q.z_7)
    if Q.log_sum_pt < 6.72 and Q.z_7 < 0.0495:
        z += 647.0 * (6.72 - Q.log_sum_pt) * (0.0495 - Q.z_7)
    if Q.mass_over_sum_pt > 0.00716 and Q.pt_7 < 44.9:
        z += -0.538 * (Q.mass_over_sum_pt - 0.00716) * (44.9 - Q.pt_7)
    return max(0.0, z)


def neuron_7(Q):
    z = 5.45
    if Q.LHA < 0.236:
        z += 21.6 * Q.LHA - 5.0976
    if Q.centroid_offset < 0.0204:
        z += 65.7 * Q.centroid_offset - 1.3402800000000001
    if Q.centroid_offset >= 0.0284:
        z += -58.4 * Q.centroid_offset + 1.65856
    if Q.e2 < 0.0384:
        z += -57.0 * Q.e2 + 0.38699999999999957
    if 0.0384 <= Q.e2 < 0.0501:
        z += 154.0 * Q.e2 - 7.7154
    if Q.eccentricity >= 0.955:
        z += -16.3 * Q.eccentricity + 15.5665
    if Q.girth2 < 0.000555:
        z += 2650.0 * Q.girth2 - 1.4707500000000002
    if 0.00741 <= Q.girth2 < 0.00874:
        z += -1710.0 * Q.girth2 + 12.6711
    if Q.girth2 >= 0.00874:
        z += -290.0 * Q.girth2 + 0.2602999999999991
    if Q.lam1 < 0.00845:
        z += 434.0 * Q.lam1 - 3.6672999999999996
    if Q.mass < 27.6:
        z += -0.124 * Q.mass + 3.4224
    if Q.mass >= 80.4:
        z += -0.132 * Q.mass + 10.612800000000002
    if 0.0849 <= Q.mass_over_sum_pt < 0.0905:
        z += 198.0 * Q.mass_over_sum_pt - 16.810200000000002
    if Q.mass_over_sum_pt >= 0.0905:
        z += -114.0 * Q.mass_over_sum_pt + 11.425799999999999
    if Q.width < 0.00524:
        z += 586.0 * Q.width - 3.07064
    if Q.centroid_offset < 0.0249 and Q.C2 > 0.0213:
        z += 1700.0 * (0.0249 - Q.centroid_offset) * (Q.C2 - 0.0213)
    if Q.centroid_offset < 0.0458 and Q.C2 > 0.065:
        z += -1660.0 * (0.0458 - Q.centroid_offset) * (Q.C2 - 0.065)
    if Q.centroid_offset < 0.0397 and Q.sum_pt > 559.0:
        z += 0.207 * (0.0397 - Q.centroid_offset) * (Q.sum_pt - 559.0)
    if Q.e2 < 0.0492 and Q.D2 < 1.14:
        z += 132.0 * (0.0492 - Q.e2) * (1.14 - Q.D2)
    if Q.e2 < 0.0219 and Q.tau21 < 0.469:
        z += 319.0 * (0.0219 - Q.e2) * (0.469 - Q.tau21)
    if Q.girth2 > 0.00457 and Q.eccentricity > 0.942:
        z += 7960.0 * (Q.girth2 - 0.00457) * (Q.eccentricity - 0.942)
    if Q.girth2 > 0.00779 and Q.log_sum_pt > 6.13:
        z += -1510.0 * (Q.girth2 - 0.00779) * (Q.log_sum_pt - 6.13)
    if Q.lam1 < 0.00849 and Q.D2 < 1.17:
        z += -1010.0 * (0.00849 - Q.lam1) * (1.17 - Q.D2)
    if Q.lam1 < 0.00831 and Q.n_pt_above_50 < 3.92:
        z += -82.5 * (0.00831 - Q.lam1) * (3.92 - Q.n_pt_above_50)
    if Q.mass > 80.4 and Q.eccentricity > 0.934:
        z += -2.21 * (Q.mass - 80.4) * (Q.eccentricity - 0.934)
    if Q.max_dr < 0.197 and Q.D2 < 1.21:
        z += 10.8 * (0.197 - Q.max_dr) * (1.21 - Q.D2)
    if Q.planar_flow < 0.174 and Q.sum_pt > 627.0:
        z += 0.0263 * (0.174 - Q.planar_flow) * (Q.sum_pt - 627.0)
    if Q.pt_7 < 47.2 and Q.planar_flow < 0.801:
        z += -0.0804 * (47.2 - Q.pt_7) * (0.801 - Q.planar_flow)
    return max(0.0, z)


def neuron_8(Q):
    z = -1.26
    if Q.C2 < 0.0262:
        z += 97.9 * Q.C2 - 2.5649800000000003
    if Q.LHA < 0.234:
        z += 19.4 * Q.LHA - 4.5396
    if Q.centroid_offset < 0.00353:
        z += -533.0 * Q.centroid_offset + 1.88149
    if Q.mass < 24.1:
        z += 0.114 * Q.mass - 2.7474000000000003
    if Q.max_dr < 0.188:
        z += -12.2 * Q.max_dr + 2.2936
    if Q.width < 0.0055:
        z += -1260.0 * Q.width + 6.93
    if Q.girth < 0.0441 and Q.log_sum_pt > 6.66:
        z += 336.0 * (0.0441 - Q.girth) * (Q.log_sum_pt - 6.66)
    if Q.girth2 < 0.00659 and Q.planar_flow < 0.401:
        z += 559.0 * (0.00659 - Q.girth2) * (0.401 - Q.planar_flow)
    if Q.log_sum_pt > 6.67 and Q.girth2 < 0.0159:
        z += -1190.0 * (Q.log_sum_pt - 6.67) * (0.0159 - Q.girth2)
    if Q.log_sum_pt > 6.6 and Q.pt_7 < 55.8:
        z += 0.172 * (Q.log_sum_pt - 6.6) * (55.8 - Q.pt_7)
    if Q.mass < 31.3 and Q.centroid_offset > 0.00794:
        z += 1.84 * (31.3 - Q.mass) * (Q.centroid_offset - 0.00794)
    if Q.mass < 21.8 and Q.max_pair_mass > 13.0:
        z += 66.8 * (21.8 - Q.mass) * (Q.max_pair_mass - 13.0)
    if Q.max_dr < 0.184 and Q.lam2 < 0.000195:
        z += 65700.0 * (0.184 - Q.max_dr) * (0.000195 - Q.lam2)
    if Q.width < 0.00581 and Q.centroid_offset > 0.00627:
        z += -58200.0 * (0.00581 - Q.width) * (Q.centroid_offset - 0.00627)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.21
    if Q.girth < 0.0383:
        z += 183.0 * Q.girth - 7.008900000000001
    if Q.girth2 >= 0.0241:
        z += 243.0 * Q.girth2 - 5.8563
    if Q.lam2 >= 0.00101:
        z += 629.0 * Q.lam2 - 0.63529
    if Q.log_sum_pt >= 6.37:
        z += -6.26 * Q.log_sum_pt + 39.8762
    if Q.mass >= 40.0:
        z += 0.0415 * Q.mass - 1.6600000000000001
    if Q.max_dr < 0.194:
        z += -15.0 * Q.max_dr + 2.91
    if Q.sum_pt >= 863.0:
        z += -0.00945 * Q.sum_pt + 8.15535
    if Q.sum_pt_top5 >= 446.0:
        z += 0.00702 * Q.sum_pt_top5 - 3.13092
    if Q.width < 0.00617:
        z += -2400.0 * Q.width + 14.808
    if Q.e2 < 0.0186 and Q.centroid_offset < 0.0237:
        z += 29100.0 * (0.0186 - Q.e2) * (0.0237 - Q.centroid_offset)
    if Q.e2 < 0.0199 and Q.eccentricity > 0.907:
        z += -1530.0 * (0.0199 - Q.e2) * (Q.eccentricity - 0.907)
    if Q.e2 < 0.0182 and Q.pt_7 < 58.1:
        z += -2.18 * (0.0182 - Q.e2) * (58.1 - Q.pt_7)
    if Q.e2 < 0.0327 and Q.tau21 < 0.43:
        z += -272.0 * (0.0327 - Q.e2) * (0.43 - Q.tau21)
    if Q.girth2 < 0.00397 and Q.centroid_offset > 0.012:
        z += -54500.0 * (0.00397 - Q.girth2) * (Q.centroid_offset - 0.012)
    if Q.mass < 38.7 and Q.centroid_offset < 0.0262:
        z += -11.1 * (38.7 - Q.mass) * (0.0262 - Q.centroid_offset)
    if Q.mass < 50.9 and Q.centroid_offset < 0.0277:
        z += 5.67 * (50.9 - Q.mass) * (0.0277 - Q.centroid_offset)
    if Q.mass < 57.9 and Q.lam1 < 0.000614:
        z += -81.5 * (57.9 - Q.mass) * (0.000614 - Q.lam1)
    if Q.mass < 54.3 and Q.log_sum_pt < 6.79:
        z += 0.15 * (54.3 - Q.mass) * (6.79 - Q.log_sum_pt)
    if Q.mass < 60.5 and Q.planar_flow < 0.317:
        z += 0.131 * (60.5 - Q.mass) * (0.317 - Q.planar_flow)
    if Q.width < 0.00673 and Q.C2 > 0.0308:
        z += -11300.0 * (0.00673 - Q.width) * (Q.C2 - 0.0308)
    if Q.width < 0.00619 and Q.centroid_offset > 0.00326:
        z += -56500.0 * (0.00619 - Q.width) * (Q.centroid_offset - 0.00326)
    return max(0.0, z)


def neuron_10(Q):
    z = 5.34
    if Q.C2 >= 0.0492:
        z += 42.4 * Q.C2 - 2.08608
    if Q.LHA >= 0.24:
        z += -13.8 * Q.LHA + 3.312
    if Q.girth < 0.0634:
        z += -61.8 * Q.girth + 3.9181199999999996
    if Q.girth2 < 0.00157:
        z += 1540.0 * Q.girth2 - 2.4178
    if Q.girth2 >= 0.00809:
        z += 310.0 * Q.girth2 - 2.5079
    if Q.lam1 < 0.00516:
        z += 518.0 * Q.lam1 - 2.6728799999999997
    if Q.lam2 < 0.00362:
        z += 906.0 * Q.lam2 - 3.2797199999999997
    if Q.log_sum_pt < 6.28:
        z += 7.51 * Q.log_sum_pt - 47.1628
    if Q.mass >= 39.2:
        z += 0.0313 * Q.mass - 1.22696
    if Q.max_dr >= 0.113:
        z += -10.6 * Q.max_dr + 1.1978
    if Q.sum_pt >= 792.0:
        z += -0.005 * Q.sum_pt + 3.96
    if Q.LHA > 0.299 and Q.tau21 < 0.663:
        z += -52.2 * (Q.LHA - 0.299) * (0.663 - Q.tau21)
    if Q.lam2 > 0.000223 and Q.tau21 < 0.562:
        z += 1630.0 * (Q.lam2 - 0.000223) * (0.562 - Q.tau21)
    return max(0.0, z)


def neuron_11(Q):
    z = 1.49
    if Q.centroid_offset < 0.0143:
        z += -133.0 * Q.centroid_offset + 6.6633
    if 0.0143 <= Q.centroid_offset < 0.0501:
        z += -195.3 * Q.centroid_offset + 7.554189999999999
    if Q.centroid_offset >= 0.0501:
        z += -62.3 * Q.centroid_offset + 0.89089
    if Q.girth < 0.0204:
        z += 163.89999999999998 * Q.girth - 12.64684
    if 0.0204 <= Q.girth < 0.0709:
        z += 89.8 * Q.girth - 11.1352
    if 0.0709 <= Q.girth < 0.124:
        z += 35.699999999999996 * Q.girth - 7.299509999999999
    if Q.girth >= 0.124:
        z += -54.1 * Q.girth + 3.8356900000000005
    if Q.mass >= 91.2:
        z += 0.357 * Q.mass - 32.5584
    if Q.mass_over_sum_pt >= 0.0933:
        z += -34.4 * Q.mass_over_sum_pt + 3.2095199999999995
    if Q.planar_flow < 0.289:
        z += -8.49 * Q.planar_flow + 2.45361
    if Q.sum_pt_top5 < 674.0:
        z += -0.00821 * Q.sum_pt_top5 + 5.53354
    if Q.width < 0.0034:
        z += -667.0 * Q.width + 8.014700000000001
    if 0.0034 <= Q.width < 0.00909:
        z += -1010.0 * Q.width + 9.180900000000001
    if Q.z_7 < 0.0424:
        z += 31.3 * Q.z_7 - 1.32712
    if Q.LHA < 0.139 and Q.z_7 < 0.0257:
        z += 1330.0 * (0.139 - Q.LHA) * (0.0257 - Q.z_7)
    if Q.centroid_offset < 0.0554 and Q.log_sum_pt < 6.82:
        z += -146.0 * (0.0554 - Q.centroid_offset) * (6.82 - Q.log_sum_pt)
    if Q.centroid_offset > 0.0186 and Q.tau21 < 0.103:
        z += 2070.0 * (Q.centroid_offset - 0.0186) * (0.103 - Q.tau21)
    if Q.girth > 0.079 and Q.pt_7 < 41.8:
        z += -11.3 * (Q.girth - 0.079) * (41.8 - Q.pt_7)
    if Q.planar_flow < 0.188 and Q.girth2 > 0.0161:
        z += 3050.0 * (0.188 - Q.planar_flow) * (Q.girth2 - 0.0161)
    if Q.planar_flow < 0.276 and Q.mass < 80.4:
        z += -0.11 * (0.276 - Q.planar_flow) * (80.4 - Q.mass)
    if Q.planar_flow < 0.361 and Q.max_dr > 0.121:
        z += -23.1 * (0.361 - Q.planar_flow) * (Q.max_dr - 0.121)
    if Q.planar_flow < 0.195 and Q.width > 0.00586:
        z += -2230.0 * (0.195 - Q.planar_flow) * (Q.width - 0.00586)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.34
    if Q.e2 >= 0.0629:
        z += -120.0 * Q.e2 + 7.548
    if Q.girth2 >= 0.0193:
        z += 238.0 * Q.girth2 - 4.5934
    if Q.mass >= 91.2:
        z += 0.0994 * Q.mass - 9.065280000000001
    if Q.girth2 > 0.0145 and Q.lam2 > 5.56e-06:
        z += 8280.0 * (Q.girth2 - 0.0145) * (Q.lam2 - 5.56e-06)
    if Q.girth2 > 0.0188 and Q.pt_7 > 15.4:
        z += 6.79 * (Q.girth2 - 0.0188) * (Q.pt_7 - 15.4)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.02
    if Q.C2 >= 0.0657:
        z += -51.6 * Q.C2 + 3.39012
    if Q.centroid_offset < 0.0387:
        z += -30.5 * Q.centroid_offset + 1.18035
    if Q.e2 < 0.0489:
        z += 39.9 * Q.e2 - 1.95111
    if Q.girth < 0.143:
        z += -67.5 * Q.girth + 9.6525
    if Q.lam1 < 0.00647:
        z += -719.0 * Q.lam1 + 6.2302599999999995
    if 0.00647 <= Q.lam1 < 0.0157:
        z += -171.0 * Q.lam1 + 2.6847
    if Q.width < 0.00741:
        z += 447.0 * Q.width - 3.31227
    if Q.z_7 < 0.0286:
        z += 226.0 * Q.z_7 - 6.4636000000000005
    if Q.girth < 0.133 and Q.log_sum_pt < 6.88:
        z += -43.9 * (0.133 - Q.girth) * (6.88 - Q.log_sum_pt)
    if Q.girth < 0.147 and Q.pt_7 < 38.2:
        z += -0.833 * (0.147 - Q.girth) * (38.2 - Q.pt_7)
    if Q.lam1 < 0.0168 and Q.centroid_offset < 0.0384:
        z += -2490.0 * (0.0168 - Q.lam1) * (0.0384 - Q.centroid_offset)
    if Q.lam2 < 0.000307 and Q.centroid_offset < 0.0423:
        z += -71500.0 * (0.000307 - Q.lam2) * (0.0423 - Q.centroid_offset)
    if Q.sum_pt > 977.0 and Q.D2 < 4.88:
        z += -0.00354 * (Q.sum_pt - 977.0) * (4.88 - Q.D2)
    if Q.sum_pt_top5 > 878.0 and Q.n_pt_above_50 > 6.08:
        z += 0.0733 * (Q.sum_pt_top5 - 878.0) * (Q.n_pt_above_50 - 6.08)
    if Q.sum_pt_top5 > 653.0 and Q.pt_7 < 44.9:
        z += 0.000554 * (Q.sum_pt_top5 - 653.0) * (44.9 - Q.pt_7)
    if Q.sum_pt_top5 > 667.0 and Q.z_7 > 0.023:
        z += -0.659 * (Q.sum_pt_top5 - 667.0) * (Q.z_7 - 0.023)
    if Q.tau21 < 0.498 and Q.max_dr > -0.0292:
        z += -17.5 * (0.498 - Q.tau21) * (Q.max_dr - -0.0292)
    return max(0.0, z)


def neuron_14(Q):
    z = -2.44
    if Q.C2 < 0.0356:
        z += 67.1 * Q.C2 - 2.38876
    if 0.0359 <= Q.C2 < 0.0669:
        z += 60.5 * Q.C2 - 2.1719500000000003
    if Q.C2 >= 0.0669:
        z += 2.299999999999997 * Q.C2 + 1.7216299999999998
    if 0.0296 <= Q.centroid_offset < 0.0496:
        z += -56.1 * Q.centroid_offset + 1.66056
    if Q.centroid_offset >= 0.0496:
        z += -304.1 * Q.centroid_offset + 13.961359999999999
    if Q.e2 < 0.0437:
        z += -180.0 * Q.e2 + 7.8660000000000005
    if Q.girth < 0.0332:
        z += 208.6 * Q.girth - 12.100140000000001
    if 0.0332 <= Q.girth < 0.0879:
        z += 94.6 * Q.girth - 8.31534
    if Q.girth2 < 0.00457:
        z += -218.0 * Q.girth2 + 4.98607
    if 0.00457 <= Q.girth2 < 0.0137:
        z += -437.0 * Q.girth2 + 5.9869
    if 0.00419 <= Q.lam1 < 0.00616:
        z += 937.0 * Q.lam1 - 3.9260300000000004
    if Q.lam1 >= 0.00616:
        z += 5.0 * Q.lam1 + 1.8150899999999992
    if Q.sum_pt_top5 < 457.0:
        z += 0.00642 * Q.sum_pt_top5 - 2.93394
    if Q.tau21 < 0.136:
        z += -9.22 * Q.tau21 + 1.2539200000000001
    if Q.width < 0.00735:
        z += 881.0 * Q.width - 6.47535
    if Q.D2 < 1.09 and Q.centroid_offset < 0.0317:
        z += 48.0 * (1.09 - Q.D2) * (0.0317 - Q.centroid_offset)
    if Q.e2 < 0.0411 and Q.D2 < 0.984:
        z += 190.0 * (0.0411 - Q.e2) * (0.984 - Q.D2)
    if Q.girth2 < 0.0135 and Q.eccentricity > 0.968:
        z += 8900.0 * (0.0135 - Q.girth2) * (Q.eccentricity - 0.968)
    if Q.lam1 > 0.00255 and Q.D2 > 0.419:
        z += 474.0 * (Q.lam1 - 0.00255) * (Q.D2 - 0.419)
    if Q.lam1 > 0.00268 and Q.D2 > 1.65:
        z += -319.0 * (Q.lam1 - 0.00268) * (Q.D2 - 1.65)
    if Q.lam1 > 0.00417 and Q.D2 > 0.395:
        z += -829.0 * (Q.lam1 - 0.00417) * (Q.D2 - 0.395)
    if Q.lam1 > 0.00709 and Q.D2 > 0.378:
        z += 371.0 * (Q.lam1 - 0.00709) * (Q.D2 - 0.378)
    if Q.lam1 > 0.00655 and Q.max_dr < 0.165:
        z += 7610.0 * (Q.lam1 - 0.00655) * (0.165 - Q.max_dr)
    if Q.mass < 75.6 and Q.D2 < 0.734:
        z += -0.0722 * (75.6 - Q.mass) * (0.734 - Q.D2)
    if Q.planar_flow < 0.104 and Q.centroid_offset > 0.00913:
        z += 1360.0 * (0.104 - Q.planar_flow) * (Q.centroid_offset - 0.00913)
    if Q.planar_flow < 0.11 and Q.centroid_offset > 0.0184:
        z += -1680.0 * (0.11 - Q.planar_flow) * (Q.centroid_offset - 0.0184)
    if Q.planar_flow < 0.1 and Q.max_dr < 0.17:
        z += -131.0 * (0.1 - Q.planar_flow) * (0.17 - Q.max_dr)
    if Q.planar_flow < 0.11 and Q.sum_pt < 741.0:
        z += -0.042 * (0.11 - Q.planar_flow) * (741.0 - Q.sum_pt)
    if Q.width < 0.00766 and Q.D2 < 1.0:
        z += -1710.0 * (0.00766 - Q.width) * (1.0 - Q.D2)
    if Q.width < 0.00787 and Q.planar_flow < 0.117:
        z += -5010.0 * (0.00787 - Q.width) * (0.117 - Q.planar_flow)
    return max(0.0, z)


def neuron_15(Q):
    z = -5.86
    if Q.LHA < 0.305:
        z += 9.22 * Q.LHA - 2.8121
    if Q.LHA >= 0.343:
        z += -73.4 * Q.LHA + 25.176200000000005
    if Q.e2 < 0.0244:
        z += -237.0 * Q.e2 + 14.859900000000001
    if 0.0244 <= Q.e2 < 0.0497:
        z += -90.0 * Q.e2 + 11.273100000000001
    if 0.0497 <= Q.e2 < 0.0627:
        z += -172.4 * Q.e2 + 15.368380000000002
    if Q.e2 >= 0.0627:
        z += 64.6 * Q.e2 + 0.5084800000000005
    if Q.girth >= 0.0329:
        z += 50.0 * Q.girth - 1.645
    if Q.lam1 < 0.00855:
        z += 941.0 * Q.lam1 - 8.04555
    if Q.lam2 < 0.000335:
        z += 3620.0 * Q.lam2 - 4.54195
    if 0.000335 <= Q.lam2 < 0.00323:
        z += 1150.0 * Q.lam2 - 3.7144999999999997
    if Q.log_sum_pt >= 6.9:
        z += 44.3 * Q.log_sum_pt - 305.67
    if Q.mass < 80.4:
        z += 0.0311 * Q.mass - 2.50044
    if Q.tau21 < 0.25:
        z += -6.19 * Q.tau21 + 1.5475
    if Q.width < 0.0136:
        z += -546.0 * Q.width + 7.425599999999999
    if Q.width < 0.00762 and Q.e2 > 0.0244:
        z += -59700.0 * (0.00762 - Q.width) * (Q.e2 - 0.0244)
    if Q.width < 0.00626 and Q.log_sum_pt > 6.9:
        z += -9720.0 * (0.00626 - Q.width) * (Q.log_sum_pt - 6.9)
    if Q.width < 0.00818 and Q.planar_flow < 0.0618:
        z += -3120.0 * (0.00818 - Q.width) * (0.0618 - Q.planar_flow)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logit_g(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.4375 - 0.15625 * h0 + 0.390625 * h1 + 0.4296875 * h2 - 0.03125 * h4 - 0.1875 * h5 + 0.109375 * h6 + 0.171875 * h9


def logit_q(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 0.03125 - 0.09375 * h4 + 0.046875 * h5 + 0.125 * h6 + 0.0625 * h8 + 0.25390625 * h9 - 0.125 * h10 + 0.0625 * h15


def logit_W(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.125 + 0.34375 * h0 - 0.03125 * h1 - 0.5 * h3 - 0.3125 * h6 + 0.21875 * h7 - 0.25 * h8 - 0.03125 * h9 + 0.375 * h11 + 0.0703125 * h13 - 0.75 * h14 - 0.6875 * h15


def logit_Z(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return -0.09375 + 0.125 * h1 - 0.5625 * h3 + 0.078125 * h4 + 0.015625 * h5 - 0.375 * h6 + 0.46875 * h7 - 0.03125 * h9 + 0.0546875 * h13 + 0.375 * h14 - 0.15625 * h15


def logit_t(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15):
    return 1.34375 + 0.015625 * h0 + 0.0625 * h3 + 0.125 * h4 - 0.25 * h5 + 0.1875 * h8 + 0.375 * h10 - 0.5 * h12 - 0.40625 * h13


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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:8] + [0.0] * 0
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:8] + [0.0] * 0
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:8] + [0.0] * 0
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
