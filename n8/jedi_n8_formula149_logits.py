"""JEDI-linear jet tagger, 8 particles, 3 features: the simpler version of the simplified formula, as if-statements, with each class score (logit) written out as a formula.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 86.5% of jets.

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
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
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
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
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
    z = -0.139
    if Q.centroid_offset < 0.033:
        z += -59.6 * Q.centroid_offset + 1.9668
    if Q.girth2 < 0.013:
        z += -262.0 * Q.girth2 + 3.4059999999999997
    if Q.mass < 29.0:
        z += 0.4268 * Q.mass - 14.7082
    if 29.0 <= Q.mass < 74.0:
        z += 0.0518 * Q.mass - 3.8331999999999997
    if Q.sum_pt >= 880.0:
        z += -0.0332 * Q.sum_pt + 29.216
    if Q.width < 0.005:
        z += 743.0 * Q.width - 3.715
    if Q.girth2 < 0.018 and Q.eccentricity > 0.97:
        z += 4970.0 * (0.018 - Q.girth2) * (Q.eccentricity - 0.97)
    if Q.lam1 < 0.0062 and Q.D2 < 0.95:
        z += -1980.0 * (0.0062 - Q.lam1) * (0.95 - Q.D2)
    if Q.mass < 65.0 and Q.centroid_offset > 0.008:
        z += 2.47 * (65.0 - Q.mass) * (Q.centroid_offset - 0.008)
    if Q.mass < 66.0 and Q.pt_7 < 39.0:
        z += -0.00141 * (66.0 - Q.mass) * (39.0 - Q.pt_7)
    if Q.sum_pt > 860.0 and Q.pt_7 < 33.0:
        z += 0.00154 * (Q.sum_pt - 860.0) * (33.0 - Q.pt_7)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.55
    if Q.pt_7 >= 32.0:
        z += 0.123 * Q.pt_7 - 3.936
    if Q.width < 0.01:
        z += 259.0 * Q.width - 2.59
    if Q.lam1 < 0.012 and Q.centroid_offset > 0.018:
        z += 10900.0 * (0.012 - Q.lam1) * (Q.centroid_offset - 0.018)
    if Q.log_sum_pt > 6.5 and Q.centroid_offset < 0.028:
        z += -238.0 * (Q.log_sum_pt - 6.5) * (0.028 - Q.centroid_offset)
    if Q.log_sum_pt > 6.5 and Q.lam2 < 0.0015:
        z += 6760.0 * (Q.log_sum_pt - 6.5) * (0.0015 - Q.lam2)
    if Q.pt_7 > 34.0 and Q.mass < 98.0:
        z += -0.00168 * (Q.pt_7 - 34.0) * (98.0 - Q.mass)
    return max(0.0, z)


def neuron_2(Q):
    z = 1.33
    if Q.LHA >= 0.13:
        z += -7.67 * Q.LHA + 0.9971
    if Q.lam1 < 0.005:
        z += -317.0 * Q.lam1 + 1.585
    if Q.log_sum_pt < 6.4:
        z += -4.3 * Q.log_sum_pt + 27.52
    if Q.pt_7 >= 28.0:
        z += 0.224 * Q.pt_7 - 6.272
    if Q.sum_pt < 780.0:
        z += -0.00698 * Q.sum_pt + 5.4444
    if Q.z_7 < 0.018:
        z += 429.0 * Q.z_7 - 7.7219999999999995
    if Q.z_7 >= 0.047:
        z += -42.8 * Q.z_7 + 2.0116
    if Q.pt_7 > 28.0 and Q.C2 < 0.054:
        z += -2.23 * (Q.pt_7 - 28.0) * (0.054 - Q.C2)
    if Q.pt_7 > 28.0 and Q.max_dr > 0.07:
        z += -0.584 * (Q.pt_7 - 28.0) * (Q.max_dr - 0.07)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.508
    if Q.LHA >= 0.27:
        z += 19.8 * Q.LHA - 5.346000000000001
    if Q.centroid_offset >= 0.013:
        z += 22.5 * Q.centroid_offset - 0.2925
    if Q.e2 < 0.047:
        z += -188.0 * Q.e2 + 8.836
    if Q.girth2 < 0.01:
        z += 1040.0 * Q.girth2 - 10.4
    if Q.lam1 >= 0.016:
        z += -189.0 * Q.lam1 + 3.024
    if Q.mass_over_sum_pt > 0.071 and Q.tau32 < 0.55:
        z += -109.0 * (Q.mass_over_sum_pt - 0.071) * (0.55 - Q.tau32)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.821
    if Q.C2 >= 0.063:
        z += -105.0 * Q.C2 + 6.615
    if Q.lam2 < 0.00042:
        z += 9010.0 * Q.lam2 - 3.7842000000000002
    if Q.mass < 55.0:
        z += -0.144 * Q.mass + 7.919999999999999
    if Q.mass_over_sum_pt >= 0.086:
        z += -184.0 * Q.mass_over_sum_pt + 15.823999999999998
    if Q.tau21 < 0.27:
        z += -29.4 * Q.tau21 + 7.938
    if Q.width >= 0.0016:
        z += 427.0 * Q.width - 0.6832
    if Q.C2 > -0.00087 and Q.pt_7 > 38.0:
        z += 4.15 * (Q.C2 - -0.00087) * (Q.pt_7 - 38.0)
    if Q.tau21 < 0.26 and Q.mass < 57.0:
        z += -0.848 * (0.26 - Q.tau21) * (57.0 - Q.mass)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.31
    if Q.e2 < 0.04:
        z += -59.6 * Q.e2 + 2.384
    if Q.log_sum_pt >= 6.9:
        z += -71.3 * Q.log_sum_pt + 491.97
    if Q.z_7 < 0.035:
        z += -277.6 * Q.z_7 + 12.555600000000002
    if 0.035 <= Q.z_7 < 0.066:
        z += -91.6 * Q.z_7 + 6.0456
    if Q.LHA < 0.22 and Q.log_sum_pt < 6.8:
        z += -132.0 * (0.22 - Q.LHA) * (6.8 - Q.log_sum_pt)
    if Q.e2 < 0.025 and Q.lam2 < 9.2e-05:
        z += 874000.0 * (0.025 - Q.e2) * (9.2e-05 - Q.lam2)
    if Q.width < 0.0026 and Q.centroid_offset < 0.021:
        z += 57000.0 * (0.0026 - Q.width) * (0.021 - Q.centroid_offset)
    if Q.z_7 < 0.072 and Q.centroid_offset < 0.032:
        z += -3020.0 * (0.072 - Q.z_7) * (0.032 - Q.centroid_offset)
    return max(0.0, z)


def neuron_6(Q):
    z = 9.45
    if Q.girth >= 0.09:
        z += 124.0 * Q.girth - 11.16
    if Q.lam2 < 0.00066:
        z += 2250.0 * Q.lam2 - 1.485
    if Q.lam2 >= 0.0027:
        z += -1280.0 * Q.lam2 + 3.4560000000000004
    if Q.mass_over_sum_pt >= 0.0089:
        z += -70.5 * Q.mass_over_sum_pt + 0.62745
    z += 12.9 * Q.max_dr
    if Q.width < 0.012:
        z += 991.0 * Q.width - 11.892
    if Q.centroid_offset > 0.0059 and Q.lam2 < 0.0026:
        z += 17000.0 * (Q.centroid_offset - 0.0059) * (0.0026 - Q.lam2)
    if Q.e2 < 0.049 and Q.z_dr_0p1_0p2 > 0.18:
        z += 744.0 * (0.049 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.18)
    if Q.log_sum_pt < 6.5 and Q.pt_7 < 44.0:
        z += 0.267 * (6.5 - Q.log_sum_pt) * (44.0 - Q.pt_7)
    if Q.log_sum_pt < 6.3 and Q.z_7 < 0.071:
        z += 1250.0 * (6.3 - Q.log_sum_pt) * (0.071 - Q.z_7)
    if Q.log_sum_pt < 6.7 and Q.z_7 < 0.049:
        z += 838.0 * (6.7 - Q.log_sum_pt) * (0.049 - Q.z_7)
    if Q.mass < 42.0 and Q.z_dr_0p05_0p1 < 0.94:
        z += 0.0882 * (42.0 - Q.mass) * (0.94 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.015 and Q.tau32 < 0.55:
        z += -67.3 * (Q.mass_over_sum_pt - 0.015) * (0.55 - Q.tau32)
    return max(0.0, z)


def neuron_7(Q):
    z = 3.96
    if Q.e2 < 0.038:
        z += -132.0 * Q.e2 + 5.016
    if Q.e2_sq < 0.0011:
        z += -1430.0 * Q.e2_sq + 1.5730000000000002
    if Q.girth < 0.087:
        z += 44.1 * Q.girth - 3.8367
    if Q.girth2 >= 0.0046:
        z += -442.0 * Q.girth2 + 2.0332
    if Q.mass >= 80.4:
        z += -0.329 * Q.mass + 26.451600000000003
    if 0.073 <= Q.mass_over_sum_pt < 0.09:
        z += 150.0 * Q.mass_over_sum_pt - 10.95
    if Q.mass_over_sum_pt >= 0.09:
        z += -52.0 * Q.mass_over_sum_pt + 7.23
    if Q.planar_flow < 0.19:
        z += 7.33 * Q.planar_flow - 1.3927
    if Q.width < 0.00069:
        z += 3750.0 * Q.width - 7.5957
    if 0.00069 <= Q.width < 0.0056:
        z += 1020.0 * Q.width - 5.712
    if Q.centroid_offset > 0.031 and Q.pt_0 > 370.0:
        z += -7.69 * (Q.centroid_offset - 0.031) * (Q.pt_0 - 370.0)
    if Q.girth2 > 0.0047 and Q.eccentricity > 0.94:
        z += 11800.0 * (Q.girth2 - 0.0047) * (Q.eccentricity - 0.94)
    if Q.planar_flow < 0.18 and Q.sum_pt > 590.0:
        z += 0.0363 * (0.18 - Q.planar_flow) * (Q.sum_pt - 590.0)
    if Q.planar_flow < 0.22 and Q.width > 0.0078:
        z += -4770.0 * (0.22 - Q.planar_flow) * (Q.width - 0.0078)
    if Q.pt_7 < 45.0 and Q.planar_flow < 0.6:
        z += -0.0886 * (45.0 - Q.pt_7) * (0.6 - Q.planar_flow)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.139
    if Q.girth < 0.063:
        z += 107.0 * Q.girth - 6.741
    if Q.width < 0.0049:
        z += -2310.0 * Q.width + 11.318999999999999
    if Q.LHA < 0.17 and Q.lam2 < 0.00039:
        z += -117000.0 * (0.17 - Q.LHA) * (0.00039 - Q.lam2)
    if Q.girth2 < 0.0051 and Q.planar_flow < 0.54:
        z += 845.0 * (0.0051 - Q.girth2) * (0.54 - Q.planar_flow)
    if Q.mass < 20.0 and Q.centroid_offset > 0.015:
        z += -9.9 * (20.0 - Q.mass) * (Q.centroid_offset - 0.015)
    if Q.width < 0.0051 and Q.centroid_offset > 0.0014:
        z += -71600.0 * (0.0051 - Q.width) * (Q.centroid_offset - 0.0014)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.18
    if Q.centroid_offset < 0.019:
        z += -129.0 * Q.centroid_offset + 2.451
    if Q.lam2 >= 0.0015:
        z += 1460.0 * Q.lam2 - 2.19
    if Q.log_sum_pt >= 6.3:
        z += -5.02 * Q.log_sum_pt + 31.625999999999998
    if Q.mass < 25.0:
        z += 0.101 * Q.mass - 2.5250000000000004
    if Q.n_dr_0p2_0p4 >= 1.5:
        z += 1.2 * Q.n_dr_0p2_0p4 - 1.7999999999999998
    if Q.width < 0.00029:
        z += -9700.0 * Q.width + 17.406499999999998
    if 0.00029 <= Q.width < 0.0065:
        z += -2350.0 * Q.width + 15.274999999999999
    if Q.girth2 > 0.019 and Q.pt_7 < 30.0:
        z += 57.9 * (Q.girth2 - 0.019) * (30.0 - Q.pt_7)
    if Q.mass < 56.0 and Q.log_sum_pt < 6.8:
        z += 0.172 * (56.0 - Q.mass) * (6.8 - Q.log_sum_pt)
    if Q.width < 0.0091 and Q.C2 > 0.026:
        z += -7340.0 * (0.0091 - Q.width) * (Q.C2 - 0.026)
    if Q.width < 0.0061 and Q.centroid_offset > 0.0016:
        z += -69600.0 * (0.0061 - Q.width) * (Q.centroid_offset - 0.0016)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.67
    if Q.C2 >= 0.059:
        z += 31.5 * Q.C2 - 1.8584999999999998
    if Q.e2 < 0.037:
        z += 41.9 * Q.e2 - 1.5502999999999998
    if Q.e2 >= 0.037:
        z += 79.2 * Q.e2 - 2.9304
    if Q.girth2 < 0.0018:
        z += 865.0 * Q.girth2 - 1.557
    if Q.lam2 >= 0.00016:
        z += 532.0 * Q.lam2 - 0.08512
    if Q.log_sum_pt < 6.3:
        z += 4.67 * Q.log_sum_pt - 29.421
    if Q.n_dr_0p2_0p4 >= 1.5:
        z += 0.599 * Q.n_dr_0p2_0p4 - 0.8985
    if Q.z_dr_0p05_0p1 < 0.34:
        z += -2.28 * Q.z_dr_0p05_0p1 + 0.7752
    if Q.LHA > 0.29 and Q.tau21 < 0.69:
        z += -40.2 * (Q.LHA - 0.29) * (0.69 - Q.tau21)
    if Q.eccentricity > 0.9 and Q.z_dr_0p2_0p4 < 0.041:
        z += 337.0 * (Q.eccentricity - 0.9) * (0.041 - Q.z_dr_0p2_0p4)
    if Q.lam2 > 0.00022 and Q.tau21 < 0.52:
        z += 2380.0 * (Q.lam2 - 0.00022) * (0.52 - Q.tau21)
    return max(0.0, z)


def neuron_11(Q):
    z = -2.21
    if Q.LHA < 0.15:
        z += 22.1 * Q.LHA - 3.315
    if Q.centroid_offset < 0.05:
        z += -136.0 * Q.centroid_offset + 6.800000000000001
    if Q.girth < 0.074:
        z += 127.0 * Q.girth - 11.176
    if 0.074 <= Q.girth < 0.088:
        z += 75.4 * Q.girth - 7.3576
    if Q.girth >= 0.088:
        z += -51.6 * Q.girth + 3.8184
    if Q.width < 0.0038:
        z += -682.0 * Q.width + 9.3416
    if 0.0038 <= Q.width < 0.0088:
        z += -1350.0 * Q.width + 11.88
    if Q.planar_flow < 0.39 and Q.width > 0.0078:
        z += -959.0 * (0.39 - Q.planar_flow) * (Q.width - 0.0078)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.12
    if Q.e2 >= 0.062:
        z += -107.0 * Q.e2 + 6.634
    if Q.mass >= 91.2:
        z += 0.0826 * Q.mass - 7.533120000000001
    if Q.width >= 0.018:
        z += 310.0 * Q.width - 5.579999999999999
    if Q.girth2 > 0.019 and Q.lam2 > 0.0016:
        z += 15500.0 * (Q.girth2 - 0.019) * (Q.lam2 - 0.0016)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.17
    if Q.C2 >= 0.065:
        z += -66.1 * Q.C2 + 4.2965
    if Q.e2 < 0.047:
        z += 82.4 * Q.e2 - 3.8728000000000002
    if Q.girth < 0.14:
        z += -83.1 * Q.girth + 11.634
    if Q.sum_pt >= 980.0:
        z += -0.017 * Q.sum_pt + 16.66
    if Q.girth < 0.15 and Q.log_sum_pt < 6.7:
        z += -48.0 * (0.15 - Q.girth) * (6.7 - Q.log_sum_pt)
    if Q.girth < 0.15 and Q.pt_7 < 35.0:
        z += -1.72 * (0.15 - Q.girth) * (35.0 - Q.pt_7)
    if Q.sum_pt_top5 > 660.0 and Q.pt_7 < 48.0:
        z += 0.00048 * (Q.sum_pt_top5 - 660.0) * (48.0 - Q.pt_7)
    if Q.tau21 < 0.52 and Q.max_dr > 0.031:
        z += -24.9 * (0.52 - Q.tau21) * (Q.max_dr - 0.031)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.0515
    if Q.e2 < 0.038:
        z += -116.0 * Q.e2 + 4.4079999999999995
    if Q.girth < 0.091:
        z += 88.8 * Q.girth - 8.0808
    if Q.girth2 < 0.014:
        z += -741.0 * Q.girth2 + 10.374
    if Q.lam1 >= 0.0075:
        z += -61.6 * Q.lam1 + 0.46199999999999997
    if Q.mass < 80.4:
        z += 0.0511 * Q.mass - 4.10844
    if Q.max_dr < 0.18:
        z += 20.6 * Q.max_dr - 3.708
    if Q.width < 0.0075:
        z += 1170.0 * Q.width - 8.775
    if Q.z_dr_0p05_0p1 < 0.64:
        z += -1.68 * Q.z_dr_0p05_0p1 + 1.0752
    if Q.girth2 < 0.014 and Q.eccentricity > 0.97:
        z += 8210.0 * (0.014 - Q.girth2) * (Q.eccentricity - 0.97)
    if Q.lam1 > 0.0064 and Q.max_dr < 0.16:
        z += 16900.0 * (Q.lam1 - 0.0064) * (0.16 - Q.max_dr)
    if Q.planar_flow < 0.12 and Q.centroid_offset > 0.011:
        z += 1070.0 * (0.12 - Q.planar_flow) * (Q.centroid_offset - 0.011)
    if Q.planar_flow < 0.12 and Q.centroid_offset > 0.018:
        z += -1220.0 * (0.12 - Q.planar_flow) * (Q.centroid_offset - 0.018)
    if Q.planar_flow < 0.12 and Q.max_dr < 0.17:
        z += -194.0 * (0.12 - Q.planar_flow) * (0.17 - Q.max_dr)
    if Q.width < 0.0087 and Q.D2 < 1.0:
        z += -454.0 * (0.0087 - Q.width) * (1.0 - Q.D2)
    if Q.width < 0.0083 and Q.log_sum_pt < 6.9:
        z += 572.0 * (0.0083 - Q.width) * (6.9 - Q.log_sum_pt)
    if Q.width < 0.0078 and Q.planar_flow < 0.097:
        z += -4740.0 * (0.0078 - Q.width) * (0.097 - Q.planar_flow)
    if Q.z_dr_0p05_0p1 < 0.63 and Q.C2 < 0.067:
        z += -73.2 * (0.63 - Q.z_dr_0p05_0p1) * (0.067 - Q.C2)
    return max(0.0, z)


def neuron_15(Q):
    z = 1.79
    if Q.LHA >= 0.34:
        z += -40.2 * Q.LHA + 13.668000000000003
    if Q.e2 < 0.041:
        z += -176.0 * Q.e2 + 7.216
    if Q.lam1 < 0.0065:
        z += -970.0 * Q.lam1 + 4.369
    if 0.0065 <= Q.lam1 < 0.0081:
        z += 1210.0 * Q.lam1 - 9.801
    if Q.log_sum_pt >= 6.9:
        z += 37.7 * Q.log_sum_pt - 260.13000000000005
    if Q.width < 0.0069:
        z += 2050.0 * Q.width - 14.145
    if Q.tau21 < 0.31 and Q.z_dr_0p05_0p1 < 0.68:
        z += -6.46 * (0.31 - Q.tau21) * (0.68 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.29 and Q.z_dr_0p2_0p4 < 0.2:
        z += 32.8 * (0.29 - Q.tau21) * (0.2 - Q.z_dr_0p2_0p4)
    if Q.width < 0.0065 and Q.log_sum_pt > 6.9:
        z += -7460.0 * (0.0065 - Q.width) * (Q.log_sum_pt - 6.9)
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
