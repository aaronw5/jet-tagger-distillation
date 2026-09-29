"""JEDI-linear jet tagger, 8 particles, 3 features: the simplest formula at the network's accuracy (from the 931-term tuned formula), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.8% (the network: 65.8%); same class as the network for 87.8% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.z_top5                 pT share of the 5 largest
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.m01                    mass of particles 0 and 1 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_7                  Δη of particle 7
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.phi_2                  Δφ of particle 2
  Q.phi_7                  Δφ of particle 7
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_eta               pT-weighted mean Δη
  Q.mean_eta2              pT-weighted mean Δη²
  Q.mean_phi               pT-weighted mean Δφ
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.centroid_offset        distance of the pT centroid from the jet axis
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
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
        mass_over_sum_pt_sq=(mass_of(n) / tot) ** 2,
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top2=mass_of(2),
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        m01=pair_mass(0, 1),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_7=z[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_7=eta[7],
        phi_0=phi[0],
        phi_1=phi[1],
        phi_2=phi[2],
        phi_7=phi[7],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_eta=sum(z[i] * eta[i] for i in P),
        mean_eta2=sum(z[i] * eta[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = -0.511
    if Q.D2 >= 3.9:
        z += -0.707 * Q.D2 + 2.7573
    if Q.centroid_offset < 0.033:
        z += -30.5 * Q.centroid_offset + 1.0065
    if Q.girth < 0.078:
        z += 75.6 * Q.girth - 5.8968
    if Q.girth2 < 0.013:
        z += -299.0 * Q.girth2 + 3.887
    if Q.girth2_top5 < 0.0085:
        z += 127.0 * Q.girth2_top5 - 1.0795000000000001
    if Q.lam1 < 0.0006:
        z += -7520.0 * Q.lam1 + 2.8289999999999993
    if 0.0006 <= Q.lam1 < 0.0015:
        z += 1870.0 * Q.lam1 - 2.805
    if Q.mass < 22.0:
        z += 0.4349 * Q.mass - 12.507100000000001
    if 22.0 <= Q.mass < 30.0:
        z += 0.1829 * Q.mass - 6.9631
    if 30.0 <= Q.mass < 59.0:
        z += 0.0509 * Q.mass - 3.0031
    if Q.n_dr_0_0p05 >= 3.8:
        z += 0.172 * Q.n_dr_0_0p05 - 0.6536
    if Q.planar_flow < 0.013:
        z += -81.1 * Q.planar_flow + 1.0542999999999998
    if 810.0 <= Q.sum_pt < 900.0:
        z += -0.0126 * Q.sum_pt + 10.206
    if Q.sum_pt >= 900.0:
        z += -0.0378 * Q.sum_pt + 32.885999999999996
    if Q.sum_pt_top5 >= 700.0:
        z += 0.0107 * Q.sum_pt_top5 - 7.489999999999999
    if Q.tau32 >= 0.44:
        z += -1.89 * Q.tau32 + 0.8316
    if Q.width < 0.0044:
        z += 485.0 * Q.width - 0.6490000000000005
    if 0.0044 <= Q.width < 0.0089:
        z += -330.0 * Q.width + 2.937
    if Q.z_dr_0_0p05 >= 0.85:
        z += 11.5 * Q.z_dr_0_0p05 - 9.775
    if Q.D2 > 3.9 and Q.phi_2 < -0.01:
        z += -126.0 * (Q.D2 - 3.9) * (-0.01 - Q.phi_2)
    if Q.girth < 0.087 and Q.m01 > 29.0:
        z += 7.56 * (0.087 - Q.girth) * (Q.m01 - 29.0)
    if Q.girth2 < 0.019 and Q.eccentricity > 0.96:
        z += 2570.0 * (0.019 - Q.girth2) * (Q.eccentricity - 0.96)
    if Q.girth2 < 0.019 and Q.mass_top2 > 29.0:
        z += -8.4 * (0.019 - Q.girth2) * (Q.mass_top2 - 29.0)
    if Q.girth2_top3 < 0.0039 and Q.m01 > 29.0:
        z += 632.0 * (0.0039 - Q.girth2_top3) * (Q.m01 - 29.0)
    if Q.lam1 < 0.0065 and Q.D2 < 0.88:
        z += -1970.0 * (0.0065 - Q.lam1) * (0.88 - Q.D2)
    if Q.lam1 < 0.0053 and Q.mass_top3 > 15.0:
        z += -29.1 * (0.0053 - Q.lam1) * (Q.mass_top3 - 15.0)
    if Q.mass < 56.0 and Q.C2 > 0.024:
        z += 1.57 * (56.0 - Q.mass) * (Q.C2 - 0.024)
    if Q.mass < 29.0 and Q.D2 < 0.87:
        z += -1.28 * (29.0 - Q.mass) * (0.87 - Q.D2)
    if Q.mass < 63.0 and Q.centroid_offset > 0.012:
        z += 1.83 * (63.0 - Q.mass) * (Q.centroid_offset - 0.012)
    if Q.mass < 30.0 and Q.dr_7 > 0.13:
        z += -4.44 * (30.0 - Q.mass) * (Q.dr_7 - 0.13)
    if Q.mass < 30.0 and Q.phi_1 > -0.056:
        z += -1.13 * (30.0 - Q.mass) * (Q.phi_1 - -0.056)
    if Q.mass < 64.0 and Q.pt_7 < 40.0:
        z += -0.00174 * (64.0 - Q.mass) * (40.0 - Q.pt_7)
    if Q.mass_over_sum_pt_sq < 0.0081 and Q.n_pt_above_50 < 7.9:
        z += 37.6 * (0.0081 - Q.mass_over_sum_pt_sq) * (7.9 - Q.n_pt_above_50)
    if Q.planar_flow < 0.14 and Q.centroid_offset < 0.051:
        z += 135.0 * (0.14 - Q.planar_flow) * (0.051 - Q.centroid_offset)
    if Q.planar_flow < 0.15 and Q.dr_2 < 0.028:
        z += -380.0 * (0.15 - Q.planar_flow) * (0.028 - Q.dr_2)
    if Q.planar_flow < 0.15 and Q.sum_pt_top2 < 370.0:
        z += -0.0376 * (0.15 - Q.planar_flow) * (370.0 - Q.sum_pt_top2)
    if Q.planar_flow < 0.18 and Q.z_top5 < 0.82:
        z += 43.5 * (0.18 - Q.planar_flow) * (0.82 - Q.z_top5)
    if Q.sum_pt > 900.0 and Q.pt_7 < 26.0:
        z += 0.00239 * (Q.sum_pt - 900.0) * (26.0 - Q.pt_7)
    if Q.sum_pt > 900.0 and Q.pt_7 > 26.0:
        z += -0.000904 * (Q.sum_pt - 900.0) * (Q.pt_7 - 26.0)
    if Q.z_dr_0p05_0p1 > 0.84 and Q.dr_7 < 0.0073:
        z += -121000.0 * (Q.z_dr_0p05_0p1 - 0.84) * (0.0073 - Q.dr_7)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.28
    if Q.LHA >= 0.28:
        z += 13.3 * Q.LHA - 3.7240000000000006
    if Q.e2 >= 0.034:
        z += -20.8 * Q.e2 + 0.7072
    if Q.e2_sq < 0.0081:
        z += -381.0 * Q.e2_sq + 3.0860999999999996
    if 6.4 <= Q.log_sum_pt < 6.6:
        z += 3.43 * Q.log_sum_pt - 21.952
    if Q.log_sum_pt >= 6.6:
        z += 13.53 * Q.log_sum_pt - 88.612
    if Q.mass_top5 < 6.1:
        z += 0.181 * Q.mass_top5 - 1.1040999999999999
    if Q.max_dr < 0.048:
        z += 28.58 * Q.max_dr - 2.802
    if 0.048 <= Q.max_dr < 0.25:
        z += 7.08 * Q.max_dr - 1.77
    if 34.0 <= Q.pt_7 < 54.0:
        z += 0.136 * Q.pt_7 - 4.6240000000000006
    if Q.pt_7 >= 54.0:
        z += 0.016000000000000014 * Q.pt_7 + 1.855999999999999
    if Q.width < 0.0087:
        z += 409.0 * Q.width - 3.5582999999999996
    if Q.z_7 < 0.044:
        z += 125.5 * Q.z_7 - 6.379
    if 0.044 <= Q.z_7 < 0.054:
        z += 85.7 * Q.z_7 - 4.6278
    if Q.C2 < 0.044 and Q.tau21 < 0.22:
        z += -350.0 * (0.044 - Q.C2) * (0.22 - Q.tau21)
    if Q.LHA < 0.33 and Q.planar_flow < 0.082:
        z += -127.0 * (0.33 - Q.LHA) * (0.082 - Q.planar_flow)
    if Q.e2 < 0.037 and Q.eccentricity > 0.98:
        z += 5240.0 * (0.037 - Q.e2) * (Q.eccentricity - 0.98)
    if Q.e2 > 0.025 and Q.pt_1 < 81.0:
        z += -0.427 * (Q.e2 - 0.025) * (81.0 - Q.pt_1)
    if Q.e2 > 0.029 and Q.tau32 < 0.64:
        z += -66.2 * (Q.e2 - 0.029) * (0.64 - Q.tau32)
    if Q.e2_sq < 0.0081 and Q.planar_flow < 0.086:
        z += -4230.0 * (0.0081 - Q.e2_sq) * (0.086 - Q.planar_flow)
    if Q.lam1 < 0.0083 and Q.centroid_offset > 0.02:
        z += 14100.0 * (0.0083 - Q.lam1) * (Q.centroid_offset - 0.02)
    if Q.lam1 < 0.0085 and Q.z_7 > 0.037:
        z += -3520.0 * (0.0085 - Q.lam1) * (Q.z_7 - 0.037)
    if Q.log_sum_pt > 6.6 and Q.D2 < 1.2:
        z += 7.52 * (Q.log_sum_pt - 6.6) * (1.2 - Q.D2)
    if Q.log_sum_pt > 6.4 and Q.centroid_offset < 0.027:
        z += -132.0 * (Q.log_sum_pt - 6.4) * (0.027 - Q.centroid_offset)
    if Q.log_sum_pt > 6.6 and Q.girth2_top3 < 0.0068:
        z += -749.0 * (Q.log_sum_pt - 6.6) * (0.0068 - Q.girth2_top3)
    if Q.log_sum_pt > 6.4 and Q.max_dr < 0.19:
        z += 22.1 * (Q.log_sum_pt - 6.4) * (0.19 - Q.max_dr)
    if Q.log_sum_pt > 6.6 and Q.n_pt_above_50 > 7.0:
        z += -3.36 * (Q.log_sum_pt - 6.6) * (Q.n_pt_above_50 - 7.0)
    if Q.mass < 46.0 and Q.tau21 < 0.2:
        z += 0.515 * (46.0 - Q.mass) * (0.2 - Q.tau21)
    if Q.n_pt_above_50 > 6.0 and Q.girth2_top2 < 0.00075:
        z += 483.0 * (Q.n_pt_above_50 - 6.0) * (0.00075 - Q.girth2_top2)
    if Q.n_pt_above_50 > 6.1 and Q.tau32 < 0.16:
        z += -19.3 * (Q.n_pt_above_50 - 6.1) * (0.16 - Q.tau32)
    if Q.pt_7 > 35.0 and Q.centroid_offset > 0.016:
        z += 1.49 * (Q.pt_7 - 35.0) * (Q.centroid_offset - 0.016)
    if Q.pt_7 > 34.0 and Q.mass < 80.4:
        z += -0.00154 * (Q.pt_7 - 34.0) * (80.4 - Q.mass)
    if Q.pt_7 > 54.0 and Q.mass_top3 < 52.0:
        z += 0.00243 * (Q.pt_7 - 54.0) * (52.0 - Q.mass_top3)
    if Q.pt_7 > 35.0 and Q.max_dr < 0.08:
        z += 1.23 * (Q.pt_7 - 35.0) * (0.08 - Q.max_dr)
    if Q.tau21 < 0.21 and Q.lam2 < 0.00021:
        z += 41000.0 * (0.21 - Q.tau21) * (0.00021 - Q.lam2)
    if Q.z_7 < 0.054 and Q.girth2_top2 < 0.014:
        z += 4820.0 * (0.054 - Q.z_7) * (0.014 - Q.girth2_top2)
    if Q.z_7 < 0.042 and Q.z_dr_0p05_0p1 > 0.2:
        z += -89.8 * (0.042 - Q.z_7) * (Q.z_dr_0p05_0p1 - 0.2)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.32
    if Q.LHA >= 0.1:
        z += -7.11 * Q.LHA + 0.7110000000000001
    if Q.girth2_top2 < 3.7e-05:
        z += -10700.0 * Q.girth2_top2 + 0.3959
    if Q.lam1 < 0.0057:
        z += -286.0 * Q.lam1 + 1.6302
    if Q.log_sum_pt < 6.3:
        z += -1.81 * Q.log_sum_pt + 11.765
    if 6.3 <= Q.log_sum_pt < 6.5:
        z += 0.5899999999999999 * Q.log_sum_pt - 3.3549999999999986
    if 6.5 <= Q.log_sum_pt < 6.8:
        z += 2.4 * Q.log_sum_pt - 15.12
    if 6.8 <= Q.log_sum_pt < 6.9:
        z += 6.76 * Q.log_sum_pt - 44.768
    if Q.log_sum_pt >= 6.9:
        z += -0.9700000000000006 * Q.log_sum_pt + 8.569000000000003
    if Q.mass < 8.5:
        z += -0.0877 * Q.mass - 0.18349999999999977
    if 8.5 <= Q.mass < 36.0:
        z += 0.0573 * Q.mass - 1.4159999999999997
    if 36.0 <= Q.mass < 69.0:
        z += -0.0196 * Q.mass + 1.3524
    if Q.mass_over_sum_pt < 0.0098:
        z += 186.0 * Q.mass_over_sum_pt - 1.8228
    if Q.mass_over_sum_pt_sq < 0.012:
        z += -77.2 * Q.mass_over_sum_pt_sq + 0.9264
    if Q.max_dr < 0.16:
        z += 5.05 * Q.max_dr - 0.8079999999999999
    if Q.planar_flow < 0.68:
        z += 1.3 * Q.planar_flow - 0.8840000000000001
    if Q.pt_7 < 31.0:
        z += 0.0481 * Q.pt_7 - 2.5974
    if 31.0 <= Q.pt_7 < 54.0:
        z += 0.1371 * Q.pt_7 - 5.3564
    if Q.pt_7 >= 54.0:
        z += 0.089 * Q.pt_7 - 2.759
    if Q.sum_pt < 790.0:
        z += -0.00556 * Q.sum_pt + 4.3924
    if Q.z_7 < 0.017:
        z += 246.0 * Q.z_7 - 4.182
    if Q.lam1 < 0.004 and Q.centroid_offset < 0.0062:
        z += -35100.0 * (0.004 - Q.lam1) * (0.0062 - Q.centroid_offset)
    if Q.lam1 < 0.0059 and Q.max_dr > 0.078:
        z += -3030.0 * (0.0059 - Q.lam1) * (Q.max_dr - 0.078)
    if Q.log_sum_pt < 6.5 and Q.eccentricity > 0.79:
        z += 8.11 * (6.5 - Q.log_sum_pt) * (Q.eccentricity - 0.79)
    if Q.log_sum_pt > 6.9 and Q.n_pt_above_50 > 5.0:
        z += -2.16 * (Q.log_sum_pt - 6.9) * (Q.n_pt_above_50 - 5.0)
    if Q.mass < 68.0 and Q.centroid_offset > 0.01:
        z += -0.406 * (68.0 - Q.mass) * (Q.centroid_offset - 0.01)
    if Q.mass < 37.0 and Q.lam2 < 0.0011:
        z += 48.8 * (37.0 - Q.mass) * (0.0011 - Q.lam2)
    if Q.mass < 71.0 and Q.max_pair_mass > 13.0:
        z += -0.00168 * (71.0 - Q.mass) * (Q.max_pair_mass - 13.0)
    if Q.mass < 69.0 and Q.z_7 < 0.068:
        z += -0.454 * (69.0 - Q.mass) * (0.068 - Q.z_7)
    if Q.pt_6 < 53.0 and Q.lam2 > -0.00063:
        z += 5.77 * (53.0 - Q.pt_6) * (Q.lam2 - -0.00063)
    if Q.pt_7 > 30.0 and Q.C2 < 0.051:
        z += -1.87 * (Q.pt_7 - 30.0) * (0.051 - Q.C2)
    if Q.pt_7 > 30.0 and Q.centroid_offset > 0.013:
        z += -1.28 * (Q.pt_7 - 30.0) * (Q.centroid_offset - 0.013)
    if Q.pt_7 > 30.0 and Q.max_dr > 0.092:
        z += -0.446 * (Q.pt_7 - 30.0) * (Q.max_dr - 0.092)
    if Q.pt_7 < 14.0 and Q.n_pt_above_10 < 7.9:
        z += 0.125 * (14.0 - Q.pt_7) * (7.9 - Q.n_pt_above_10)
    if Q.sum_pt < 790.0 and Q.dr_7 < 0.064:
        z += 0.0414 * (790.0 - Q.sum_pt) * (0.064 - Q.dr_7)
    if Q.sum_pt_top5 > 740.0 and Q.mean_eta2 < 0.0015:
        z += -3.84 * (Q.sum_pt_top5 - 740.0) * (0.0015 - Q.mean_eta2)
    if Q.z_7 > 0.044 and Q.dr_7 < 0.066:
        z += -483.0 * (Q.z_7 - 0.044) * (0.066 - Q.dr_7)
    return max(0.0, z)


def neuron_3(Q):
    z = 1.38
    if Q.C2 >= 0.094:
        z += -247.0 * Q.C2 + 23.218
    if Q.LHA >= 0.32:
        z += 87.6 * Q.LHA - 28.032
    if Q.centroid_offset >= 0.013:
        z += 82.8 * Q.centroid_offset - 1.0764
    if Q.e2 < 0.028:
        z += -130.0 * Q.e2 + 5.59
    if 0.028 <= Q.e2 < 0.043:
        z += -219.1 * Q.e2 + 8.0848
    if 0.043 <= Q.e2 < 0.051:
        z += -89.1 * Q.e2 + 2.4947999999999997
    if Q.e2 >= 0.051:
        z += 27.900000000000006 * Q.e2 - 3.4722
    if Q.girth2 < 0.0088:
        z += 1132.0 * Q.girth2 - 11.524000000000001
    if 0.0088 <= Q.girth2 < 0.013:
        z += 372.0 * Q.girth2 - 4.835999999999999
    if Q.lam1 >= 0.016:
        z += 556.0 * Q.lam1 - 8.896
    if Q.mass >= 70.0:
        z += -0.0845 * Q.mass + 5.915
    if Q.mass_top5 >= 53.0:
        z += -0.0939 * Q.mass_top5 + 4.9767
    if Q.max_dr >= 0.15:
        z += 24.5 * Q.max_dr - 3.675
    if Q.LHA > 0.31 and Q.eccentricity > 0.96:
        z += 2050.0 * (Q.LHA - 0.31) * (Q.eccentricity - 0.96)
    if Q.LHA > 0.31 and Q.max_dr < 0.15:
        z += -3470.0 * (Q.LHA - 0.31) * (0.15 - Q.max_dr)
    if Q.LHA > 0.32 and Q.planar_flow > 0.012:
        z += -77.9 * (Q.LHA - 0.32) * (Q.planar_flow - 0.012)
    if Q.LHA > 0.31 and Q.pt_5 < 30.0:
        z += 16.0 * (Q.LHA - 0.31) * (30.0 - Q.pt_5)
    if Q.LHA > 0.42 and Q.pt_7 > 38.0:
        z += -30.1 * (Q.LHA - 0.42) * (Q.pt_7 - 38.0)
    if Q.centroid_offset > 0.012 and Q.mean_phi < -0.0092:
        z += -2390.0 * (Q.centroid_offset - 0.012) * (-0.0092 - Q.mean_phi)
    if Q.centroid_offset > 0.015 and Q.phi_7 > -0.03:
        z += 246.0 * (Q.centroid_offset - 0.015) * (Q.phi_7 - -0.03)
    if Q.e2 > 0.027 and Q.n_pt_above_50 > 3.0:
        z += -13.0 * (Q.e2 - 0.027) * (Q.n_pt_above_50 - 3.0)
    if Q.e2 < 0.038 and Q.phi_0 > 0.08:
        z += -50100.0 * (0.038 - Q.e2) * (Q.phi_0 - 0.08)
    if Q.lam1 > 0.018 and Q.eccentricity > 0.96:
        z += 12900.0 * (Q.lam1 - 0.018) * (Q.eccentricity - 0.96)
    if Q.lam1 > 0.012 and Q.pt_6 < 38.0:
        z += 78.8 * (Q.lam1 - 0.012) * (38.0 - Q.pt_6)
    if Q.lam1 > 0.0086 and Q.pt_7 < 38.0:
        z += 41.6 * (Q.lam1 - 0.0086) * (38.0 - Q.pt_7)
    if Q.lam1 > 0.0028 and Q.z_7 < 0.077:
        z += -4000.0 * (Q.lam1 - 0.0028) * (0.077 - Q.z_7)
    if Q.mass > 38.0 and Q.dr_6 < 0.046:
        z += -1.28 * (Q.mass - 38.0) * (0.046 - Q.dr_6)
    if Q.mass > 37.0 and Q.lam2 < 0.0013:
        z += 55.1 * (Q.mass - 37.0) * (0.0013 - Q.lam2)
    if Q.mass > 70.0 and Q.max_dr < 0.18:
        z += 3.89 * (Q.mass - 70.0) * (0.18 - Q.max_dr)
    if Q.mass > 70.0 and Q.mean_phi > 0.00098:
        z += -5.08 * (Q.mass - 70.0) * (Q.mean_phi - 0.00098)
    if Q.mass > 36.0 and Q.pt_7 < 20.0:
        z += -0.00436 * (Q.mass - 36.0) * (20.0 - Q.pt_7)
    if Q.mass_over_sum_pt > 0.069 and Q.dr_7 < 0.042:
        z += -10900.0 * (Q.mass_over_sum_pt - 0.069) * (0.042 - Q.dr_7)
    if Q.mass_over_sum_pt > 0.11 and Q.dr_7 < 0.049:
        z += 21900.0 * (Q.mass_over_sum_pt - 0.11) * (0.049 - Q.dr_7)
    if Q.mass_over_sum_pt > 0.091 and Q.max_dr < 0.15:
        z += 8590.0 * (Q.mass_over_sum_pt - 0.091) * (0.15 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.065 and Q.n_dr_0_0p05 < 4.9:
        z += 10.8 * (Q.mass_over_sum_pt - 0.065) * (4.9 - Q.n_dr_0_0p05)
    if Q.mass_over_sum_pt > 0.07 and Q.tau32 < 0.52:
        z += -178.0 * (Q.mass_over_sum_pt - 0.07) * (0.52 - Q.tau32)
    if Q.mass_top5 > 54.0 and Q.eta_7 < 0.035:
        z += -0.763 * (Q.mass_top5 - 54.0) * (0.035 - Q.eta_7)
    if Q.max_dr > 0.16 and Q.dr_1 < 0.047:
        z += -640.0 * (Q.max_dr - 0.16) * (0.047 - Q.dr_1)
    if Q.max_dr > 0.15 and Q.dr_3 < 0.046:
        z += -507.0 * (Q.max_dr - 0.15) * (0.046 - Q.dr_3)
    if Q.max_dr > 0.14 and Q.phi_0 < -0.041:
        z += 704.0 * (Q.max_dr - 0.14) * (-0.041 - Q.phi_0)
    if Q.mean_eta < -0.015 and Q.pt_4 < 69.0:
        z += -3.14 * (-0.015 - Q.mean_eta) * (69.0 - Q.pt_4)
    if Q.mean_eta < -0.016 and Q.z_4 < 0.13:
        z += 1710.0 * (-0.016 - Q.mean_eta) * (0.13 - Q.z_4)
    if Q.width > 0.019 and Q.pt_dispersion < 0.49:
        z += 6930.0 * (Q.width - 0.019) * (0.49 - Q.pt_dispersion)
    return max(0.0, z)


def neuron_4(Q):
    z = -6.77
    if 0.013 <= Q.C2 < 0.067:
        z += -83.0 * Q.C2 + 1.079
    if Q.C2 >= 0.067:
        z += -185.0 * Q.C2 + 7.913
    if Q.centroid_offset < 0.014:
        z += -96.8 * Q.centroid_offset + 1.3552
    if 0.02 <= Q.e2 < 0.064:
        z += 87.9 * Q.e2 - 1.7580000000000002
    if Q.e2 >= 0.064:
        z += -77.1 * Q.e2 + 8.802
    if Q.e2_sq < 0.003:
        z += 584.0 * Q.e2_sq + 5.612
    if 0.003 <= Q.e2_sq < 0.017:
        z += -526.0 * Q.e2_sq + 8.942
    if Q.girth < 0.13:
        z += -49.2 * Q.girth + 6.396000000000001
    if 0.0034 <= Q.girth2 < 0.0081:
        z += 839.0 * Q.girth2 - 2.8526
    if Q.girth2 >= 0.0081:
        z += 355.0 * Q.girth2 + 1.0678
    if Q.lam2 < 0.00033:
        z += 4450.0 * Q.lam2 - 1.4685
    if Q.mass < 44.0:
        z += -0.0815 * Q.mass + 3.5860000000000003
    if 0.089 <= Q.mass_over_sum_pt < 0.11:
        z += -137.0 * Q.mass_over_sum_pt + 12.193
    if Q.mass_over_sum_pt >= 0.11:
        z += -35.0 * Q.mass_over_sum_pt + 0.972999999999999
    if 0.094 <= Q.max_dr < 0.2:
        z += 18.4 * Q.max_dr - 1.7295999999999998
    if Q.max_dr >= 0.2:
        z += -6.200000000000003 * Q.max_dr + 3.1904000000000012
    if Q.sum_pt < 760.0:
        z += -0.00591 * Q.sum_pt + 4.4916
    if Q.sum_pt_top5 < 430.0:
        z += -0.0221 * Q.sum_pt_top5 + 9.503
    if Q.tau21 < 0.24:
        z += -41.1 * Q.tau21 + 9.864
    if Q.C2 > 0.015 and Q.pt_7 > 39.0:
        z += 9.44 * (Q.C2 - 0.015) * (Q.pt_7 - 39.0)
    if Q.C2 > 0.065 and Q.pt_7 < 36.0:
        z += -9.53 * (Q.C2 - 0.065) * (36.0 - Q.pt_7)
    if Q.centroid_offset < 0.013 and Q.z_dr_0p05_0p1 < 0.65:
        z += -231.0 * (0.013 - Q.centroid_offset) * (0.65 - Q.z_dr_0p05_0p1)
    if Q.girth2_top2 < 0.0087 and Q.centroid_offset > 0.016:
        z += 17100.0 * (0.0087 - Q.girth2_top2) * (Q.centroid_offset - 0.016)
    if Q.mass < 70.0 and Q.mean_eta2 > 0.0044:
        z += -6.87 * (70.0 - Q.mass) * (Q.mean_eta2 - 0.0044)
    if Q.max_dr > 0.11 and Q.eccentricity > 0.98:
        z += -952.0 * (Q.max_dr - 0.11) * (Q.eccentricity - 0.98)
    if Q.max_dr > 0.098 and Q.pt_7 > 38.0:
        z += -1.64 * (Q.max_dr - 0.098) * (Q.pt_7 - 38.0)
    if Q.tau21 < 0.25 and Q.e2_sq > 0.011:
        z += -2370.0 * (0.25 - Q.tau21) * (Q.e2_sq - 0.011)
    if Q.tau21 < 0.26 and Q.lam2 < 0.0013:
        z += -16900.0 * (0.26 - Q.tau21) * (0.0013 - Q.lam2)
    if Q.tau21 < 0.23 and Q.mass < 65.0:
        z += -0.618 * (0.23 - Q.tau21) * (65.0 - Q.mass)
    if Q.tau21 < 0.22 and Q.mean_phi < -0.028:
        z += -416.0 * (0.22 - Q.tau21) * (-0.028 - Q.mean_phi)
    if Q.tau21 < 0.24 and Q.pt_7 > 32.0:
        z += 0.476 * (0.24 - Q.tau21) * (Q.pt_7 - 32.0)
    if Q.tau21 < 0.29 and Q.pt_7 < 24.0:
        z += -1.15 * (0.29 - Q.tau21) * (24.0 - Q.pt_7)
    if Q.tau21 < 0.23 and Q.sum_pt_top2 < 410.0:
        z += 0.0435 * (0.23 - Q.tau21) * (410.0 - Q.sum_pt_top2)
    if Q.width > -0.00013 and Q.C2 < 0.063:
        z += 2140.0 * (Q.width - -0.00013) * (0.063 - Q.C2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.107
    if Q.dr_0 < 0.023:
        z += 216.0 * Q.dr_0 - 4.968
    if Q.e2 < 0.034:
        z += -65.7 * Q.e2 + 2.2338000000000005
    if Q.girth2 < 5.2e-05:
        z += -55900.0 * Q.girth2 + 2.9067999999999996
    if 6.6 <= Q.log_sum_pt < 6.8:
        z += 7.85 * Q.log_sum_pt - 51.809999999999995
    if Q.log_sum_pt >= 6.8:
        z += -24.85 * Q.log_sum_pt + 170.55
    if Q.pt_7 < 53.0:
        z += -0.0737 * Q.pt_7 + 3.9061
    if Q.sum_pt >= 870.0:
        z += -0.00368 * Q.sum_pt + 3.2016
    if Q.sum_pt_top2 < 540.0:
        z += 0.00655 * Q.sum_pt_top2 - 3.537
    if Q.z_7 < 0.024:
        z += -248.0 * Q.z_7 + 5.952
    if Q.LHA < 0.15 and Q.centroid_offset > 0.0034:
        z += -3370.0 * (0.15 - Q.LHA) * (Q.centroid_offset - 0.0034)
    if Q.LHA < 0.21 and Q.centroid_offset > 0.0028:
        z += -920.0 * (0.21 - Q.LHA) * (Q.centroid_offset - 0.0028)
    if Q.LHA < 0.22 and Q.log_sum_pt < 6.8:
        z += -62.5 * (0.22 - Q.LHA) * (6.8 - Q.log_sum_pt)
    if Q.LHA < 0.21 and Q.mass_top3 > 3.5:
        z += -1.74 * (0.21 - Q.LHA) * (Q.mass_top3 - 3.5)
    if Q.LHA < 0.22 and Q.n_dr_0p2_0p4 > -1.6e-05:
        z += -14.7 * (0.22 - Q.LHA) * (Q.n_dr_0p2_0p4 - -1.6e-05)
    if Q.LHA < 0.21 and Q.planar_flow < 0.083:
        z += 397.0 * (0.21 - Q.LHA) * (0.083 - Q.planar_flow)
    if Q.log_sum_pt > 6.3 and Q.centroid_offset > 0.00063:
        z += 137.0 * (Q.log_sum_pt - 6.3) * (Q.centroid_offset - 0.00063)
    if Q.log_sum_pt > 6.6 and Q.centroid_offset > 0.017:
        z += -799.0 * (Q.log_sum_pt - 6.6) * (Q.centroid_offset - 0.017)
    if Q.log_sum_pt > 6.6 and Q.dr_0 < 0.021:
        z += 792.0 * (Q.log_sum_pt - 6.6) * (0.021 - Q.dr_0)
    if Q.log_sum_pt > 6.9 and Q.dr_0 < 0.043:
        z += -1630.0 * (Q.log_sum_pt - 6.9) * (0.043 - Q.dr_0)
    if Q.log_sum_pt > 6.9 and Q.dr_4 > 0.039:
        z += 201.0 * (Q.log_sum_pt - 6.9) * (Q.dr_4 - 0.039)
    if Q.log_sum_pt > 6.6 and Q.mean_phi2 < 0.00014:
        z += 29200.0 * (Q.log_sum_pt - 6.6) * (0.00014 - Q.mean_phi2)
    if Q.log_sum_pt > 6.6 and Q.n_dr_0p2_0p4 < 0.99:
        z += -10.6 * (Q.log_sum_pt - 6.6) * (0.99 - Q.n_dr_0p2_0p4)
    if Q.log_sum_pt > 6.9 and Q.pt_5 > 48.0:
        z += 0.534 * (Q.log_sum_pt - 6.9) * (Q.pt_5 - 48.0)
    if Q.mass < 57.0 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.0124 * (57.0 - Q.mass) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.mass_over_sum_pt < 0.081 and Q.mass_top3 > 29.0:
        z += 6.77 * (0.081 - Q.mass_over_sum_pt) * (Q.mass_top3 - 29.0)
    if Q.mean_phi2 < 0.015 and Q.max_pair_mass < 41.0:
        z += 1.83 * (0.015 - Q.mean_phi2) * (41.0 - Q.max_pair_mass)
    if Q.sum_pt > 870.0 and Q.centroid_offset < 0.013:
        z += 1.93 * (Q.sum_pt - 870.0) * (0.013 - Q.centroid_offset)
    if Q.sum_pt_top2 < 550.0 and Q.dr_0 < 0.022:
        z += 0.401 * (550.0 - Q.sum_pt_top2) * (0.022 - Q.dr_0)
    if Q.sum_pt_top2 < 570.0 and Q.girth2_top3 < 0.0036:
        z += -1.43 * (570.0 - Q.sum_pt_top2) * (0.0036 - Q.girth2_top3)
    if Q.sum_pt_top5 > 730.0 and Q.D2 < 1.6:
        z += -0.012 * (Q.sum_pt_top5 - 730.0) * (1.6 - Q.D2)
    if Q.width < 0.0026 and Q.centroid_offset < 0.025:
        z += 69700.0 * (0.0026 - Q.width) * (0.025 - Q.centroid_offset)
    if Q.z_7 < 0.071 and Q.centroid_offset < 0.031:
        z += -2300.0 * (0.071 - Q.z_7) * (0.031 - Q.centroid_offset)
    if Q.z_7 < 0.07 and Q.n_dr_0p2_0p4 < 1.0:
        z += 25.8 * (0.07 - Q.z_7) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_7 < 0.073 and Q.sum_pt < 790.0:
        z += -0.289 * (0.073 - Q.z_7) * (790.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_6(Q):
    z = 10.5
    if Q.C2 < 0.033:
        z += -27.2 * Q.C2 + 0.9248000000000001
    if 0.033 <= Q.C2 < 0.034:
        z += -51.099999999999994 * Q.C2 + 1.7135
    if Q.C2 >= 0.034:
        z += -23.9 * Q.C2 + 0.7887
    if 0.019 <= Q.centroid_offset < 0.05:
        z += -38.1 * Q.centroid_offset + 0.7239
    if Q.centroid_offset >= 0.05:
        z += 67.9 * Q.centroid_offset - 4.5761
    if Q.e2 < 0.051:
        z += -57.2 * Q.e2 + 2.9172
    if Q.girth >= 0.083:
        z += 100.0 * Q.girth - 8.3
    if Q.girth2 < 0.0036:
        z += 717.0 * Q.girth2 - 9.0312
    if 0.0036 <= Q.girth2 < 0.0086:
        z += 1290.0 * Q.girth2 - 11.094
    if Q.girth2_top2 < 0.01:
        z += -106.0 * Q.girth2_top2 + 1.06
    if Q.girth2_top5 >= 0.011:
        z += 178.0 * Q.girth2_top5 - 1.958
    if Q.lam1 < 0.0078:
        z += -485.0 * Q.lam1 + 3.783
    if Q.lam2 >= 0.0034:
        z += -1060.0 * Q.lam2 + 3.6039999999999996
    if Q.log_sum_pt < 6.8:
        z += 1.05 * Q.log_sum_pt - 7.14
    if Q.mass < 50.0:
        z += 0.028 * Q.mass - 1.4000000000000001
    if Q.mass_over_sum_pt >= 0.0081:
        z += -80.7 * Q.mass_over_sum_pt + 0.65367
    if Q.max_dr < 0.11:
        z += 13.8 * Q.max_dr - 1.518
    if Q.max_dr >= 0.12:
        z += 14.1 * Q.max_dr - 1.692
    if Q.planar_flow < 0.06:
        z += 10.5 * Q.planar_flow - 0.63
    if Q.pt_6 < 25.0:
        z += -0.0785 * Q.pt_6 + 1.9625
    if Q.sum_pt >= 990.0:
        z += 0.0257 * Q.sum_pt - 25.443
    if Q.sum_pt_top5 >= 840.0:
        z += -0.00831 * Q.sum_pt_top5 + 6.9803999999999995
    if Q.width < 0.013:
        z += 657.0 * Q.width - 8.541
    if Q.C2 > 0.01 and Q.pt_7 > 32.0:
        z += 1.88 * (Q.C2 - 0.01) * (Q.pt_7 - 32.0)
    if Q.D2 < 1.7 and Q.min_pair_mass < 2.4:
        z += -0.302 * (1.7 - Q.D2) * (2.4 - Q.min_pair_mass)
    if Q.D2 < 1.8 and Q.pt_4 < 86.0:
        z += -0.0111 * (1.8 - Q.D2) * (86.0 - Q.pt_4)
    if Q.centroid_offset > 0.0084 and Q.C2 < 0.096:
        z += 477.0 * (Q.centroid_offset - 0.0084) * (0.096 - Q.C2)
    if Q.centroid_offset > 0.0076 and Q.lam2 < 0.0035:
        z += 14800.0 * (Q.centroid_offset - 0.0076) * (0.0035 - Q.lam2)
    if Q.centroid_offset > 0.0078 and Q.mass_top3 < 3.5:
        z += -10.1 * (Q.centroid_offset - 0.0078) * (3.5 - Q.mass_top3)
    if Q.centroid_offset > 0.0079 and Q.mean_eta2 < 0.0012:
        z += -24600.0 * (Q.centroid_offset - 0.0079) * (0.0012 - Q.mean_eta2)
    if Q.centroid_offset > 0.019 and Q.pt_5 < 25.0:
        z += 31.8 * (Q.centroid_offset - 0.019) * (25.0 - Q.pt_5)
    if Q.centroid_offset > 0.019 and Q.pt_5 > 59.0:
        z += 4.75 * (Q.centroid_offset - 0.019) * (Q.pt_5 - 59.0)
    if Q.e2 < 0.05 and Q.z_dr_0p1_0p2 > 0.16:
        z += 1160.0 * (0.05 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.16)
    if Q.girth2 < 0.0035 and Q.dr_7 > 0.13:
        z += -7360.0 * (0.0035 - Q.girth2) * (Q.dr_7 - 0.13)
    if Q.girth2_top2 < 0.0096 and Q.mean_phi < -0.0091:
        z += 4430.0 * (0.0096 - Q.girth2_top2) * (-0.0091 - Q.mean_phi)
    if Q.girth2_top5 > 0.0082 and Q.mean_eta > 0.014:
        z += -4370.0 * (Q.girth2_top5 - 0.0082) * (Q.mean_eta - 0.014)
    if Q.girth2_top5 > 0.011 and Q.pt_7 > 17.0:
        z += -6.12 * (Q.girth2_top5 - 0.011) * (Q.pt_7 - 17.0)
    if Q.lam2 < 0.00054 and Q.z_dr_0p2_0p4 < 0.2:
        z += -8070.0 * (0.00054 - Q.lam2) * (0.2 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt < 6.7 and Q.mean_phi > 0.0098:
        z += -76.1 * (6.7 - Q.log_sum_pt) * (Q.mean_phi - 0.0098)
    if Q.log_sum_pt < 6.3 and Q.pt_6 > 27.0:
        z += 0.248 * (6.3 - Q.log_sum_pt) * (Q.pt_6 - 27.0)
    if Q.log_sum_pt < 6.7 and Q.pt_6 < 37.0:
        z += 0.234 * (6.7 - Q.log_sum_pt) * (37.0 - Q.pt_6)
    if Q.log_sum_pt < 6.7 and Q.pt_7 < 46.0:
        z += 0.317 * (6.7 - Q.log_sum_pt) * (46.0 - Q.pt_7)
    if Q.log_sum_pt < 6.7 and Q.pt_dispersion > 0.4:
        z += -20.2 * (6.7 - Q.log_sum_pt) * (Q.pt_dispersion - 0.4)
    if Q.log_sum_pt < 6.3 and Q.z_7 < 0.071:
        z += 1010.0 * (6.3 - Q.log_sum_pt) * (0.071 - Q.z_7)
    if Q.log_sum_pt < 6.7 and Q.z_7 < 0.049:
        z += 542.0 * (6.7 - Q.log_sum_pt) * (0.049 - Q.z_7)
    if Q.mass < 49.0 and Q.dr_7 > 0.15:
        z += 0.972 * (49.0 - Q.mass) * (Q.dr_7 - 0.15)
    if Q.mass < 50.0 and Q.z_dr_0p05_0p1 < 0.76:
        z += 0.056 * (50.0 - Q.mass) * (0.76 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.0086 and Q.pt_7 < 42.0:
        z += -0.709 * (Q.mass_over_sum_pt - 0.0086) * (42.0 - Q.pt_7)
    if Q.mass_over_sum_pt > 0.015 and Q.tau32 < 0.52:
        z += -54.5 * (Q.mass_over_sum_pt - 0.015) * (0.52 - Q.tau32)
    if Q.sum_pt > 970.0 and Q.pt_6 > 36.0:
        z += -0.000204 * (Q.sum_pt - 970.0) * (Q.pt_6 - 36.0)
    if Q.width < 0.014 and Q.mean_phi < -0.026:
        z += 5630.0 * (0.014 - Q.width) * (-0.026 - Q.mean_phi)
    if Q.width < 0.014 and Q.mean_phi > 0.026:
        z += 7910.0 * (0.014 - Q.width) * (Q.mean_phi - 0.026)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.24
    if Q.centroid_offset < 0.021:
        z += 70.8 * Q.centroid_offset - 1.4868000000000001
    if Q.e2_sq < 0.0012:
        z += -1250.0 * Q.e2_sq + 1.4999999999999998
    if Q.girth < 0.088:
        z += 52.9 * Q.girth - 4.6552
    if 0.0015 <= Q.girth2 < 0.0044:
        z += -612.0 * Q.girth2 + 0.918
    if 0.0044 <= Q.girth2 < 0.0075:
        z += -1069.0 * Q.girth2 + 2.9288000000000003
    if 0.0075 <= Q.girth2 < 0.0087:
        z += -2479.0 * Q.girth2 + 13.5038
    if 0.0087 <= Q.girth2 < 0.015:
        z += -1009.0 * Q.girth2 + 0.7148000000000003
    if Q.girth2 >= 0.015:
        z += -455.0 * Q.girth2 - 7.5952
    if Q.girth2_top2 < 0.0011:
        z += 1090.0 * Q.girth2_top2 - 1.199
    if Q.lam1 < 0.0084:
        z += 610.0 * Q.lam1 - 5.124
    if Q.mass < 30.0:
        z += -0.0933 * Q.mass + 2.799
    if Q.mass >= 80.4:
        z += -0.189 * Q.mass + 15.1956
    if 0.072 <= Q.mass_over_sum_pt < 0.085:
        z += 97.8 * Q.mass_over_sum_pt - 7.041599999999999
    if 0.085 <= Q.mass_over_sum_pt < 0.091:
        z += 288.8 * Q.mass_over_sum_pt - 23.2766
    if Q.mass_over_sum_pt >= 0.091:
        z += 8.800000000000011 * Q.mass_over_sum_pt + 2.203400000000002
    if Q.max_dr < 0.16:
        z += 40.2 * Q.max_dr - 6.432
    if Q.width < 0.00052:
        z += 3081.0 * Q.width - 5.491499999999999
    if 0.00052 <= Q.width < 0.0055:
        z += 781.0 * Q.width - 4.2955
    if Q.z_dr_0p05_0p1 >= 0.66:
        z += -2.07 * Q.z_dr_0p05_0p1 + 1.3661999999999999
    if Q.centroid_offset < 0.022 and Q.C2 > 0.025:
        z += 2380.0 * (0.022 - Q.centroid_offset) * (Q.C2 - 0.025)
    if Q.centroid_offset < 0.037 and Q.C2 > 0.068:
        z += -1970.0 * (0.037 - Q.centroid_offset) * (Q.C2 - 0.068)
    if Q.centroid_offset > 0.031 and Q.pt_0 > 380.0:
        z += -10.9 * (Q.centroid_offset - 0.031) * (Q.pt_0 - 380.0)
    if Q.centroid_offset > 0.031 and Q.pt_2 > 64.0:
        z += -1.32 * (Q.centroid_offset - 0.031) * (Q.pt_2 - 64.0)
    if Q.centroid_offset < 0.039 and Q.sum_pt > 560.0:
        z += 0.227 * (0.039 - Q.centroid_offset) * (Q.sum_pt - 560.0)
    if Q.e2 < 0.025 and Q.D2 < 1.1:
        z += -509.0 * (0.025 - Q.e2) * (1.1 - Q.D2)
    if Q.e2 < 0.038 and Q.D2 < 1.1:
        z += 325.0 * (0.038 - Q.e2) * (1.1 - Q.D2)
    if Q.e2 < 0.025 and Q.phi_0 > 0.055:
        z += -38000.0 * (0.025 - Q.e2) * (Q.phi_0 - 0.055)
    if Q.e2 < 0.025 and Q.tau21 < 0.46:
        z += 338.0 * (0.025 - Q.e2) * (0.46 - Q.tau21)
    if Q.girth2 > 0.004 and Q.eccentricity > 0.95:
        z += 9130.0 * (Q.girth2 - 0.004) * (Q.eccentricity - 0.95)
    if Q.girth2 > 0.0075 and Q.log_sum_pt > 6.2:
        z += -1440.0 * (Q.girth2 - 0.0075) * (Q.log_sum_pt - 6.2)
    if Q.girth2_top2 < 0.001 and Q.log_sum_pt > 6.3:
        z += -2170.0 * (0.001 - Q.girth2_top2) * (Q.log_sum_pt - 6.3)
    if Q.girth2_top2 < 0.0011 and Q.n_dr_0p2_0p4 < 0.97:
        z += 1880.0 * (0.0011 - Q.girth2_top2) * (0.97 - Q.n_dr_0p2_0p4)
    if Q.girth2_top3 < 0.0059 and Q.max_pair_mass > 46.0:
        z += 255.0 * (0.0059 - Q.girth2_top3) * (Q.max_pair_mass - 46.0)
    if Q.girth2_top3 < 0.005 and Q.n_dr_0p2_0p4 > 0.88:
        z += 147.0 * (0.005 - Q.girth2_top3) * (Q.n_dr_0p2_0p4 - 0.88)
    if Q.lam1 < 0.0081 and Q.D2 < 1.1:
        z += -1250.0 * (0.0081 - Q.lam1) * (1.1 - Q.D2)
    if Q.lam1 < 0.0087 and Q.n_pt_above_50 < 4.1:
        z += -78.8 * (0.0087 - Q.lam1) * (4.1 - Q.n_pt_above_50)
    if Q.mass > 80.4 and Q.eccentricity > 0.92:
        z += -1.05 * (Q.mass - 80.4) * (Q.eccentricity - 0.92)
    if Q.mass > 80.4 and Q.m012 < 5.4:
        z += -0.0557 * (Q.mass - 80.4) * (5.4 - Q.m012)
    if Q.mass_over_sum_pt < 0.13 and Q.z_dr_0p05_0p1 > 0.27:
        z += -34.7 * (0.13 - Q.mass_over_sum_pt) * (Q.z_dr_0p05_0p1 - 0.27)
    if Q.max_dr < 0.16 and Q.D2 < 1.2:
        z += 34.6 * (0.16 - Q.max_dr) * (1.2 - Q.D2)
    if Q.max_dr < 0.16 and Q.z_dr_0p05_0p1 < 0.67:
        z += 56.9 * (0.16 - Q.max_dr) * (0.67 - Q.z_dr_0p05_0p1)
    if Q.max_dr < 0.2 and Q.z_dr_0p05_0p1 > 0.056:
        z += 37.6 * (0.2 - Q.max_dr) * (Q.z_dr_0p05_0p1 - 0.056)
    if Q.planar_flow < 0.2 and Q.D2 < 1.6:
        z += -4.77 * (0.2 - Q.planar_flow) * (1.6 - Q.D2)
    if Q.planar_flow < 0.19 and Q.n_dr_0p1_0p2 < 2.9:
        z += -1.17 * (0.19 - Q.planar_flow) * (2.9 - Q.n_dr_0p1_0p2)
    if Q.planar_flow < 0.2 and Q.sum_pt > 620.0:
        z += 0.0233 * (0.2 - Q.planar_flow) * (Q.sum_pt - 620.0)
    if Q.pt_7 < 47.0 and Q.planar_flow < 0.74:
        z += -0.0966 * (47.0 - Q.pt_7) * (0.74 - Q.planar_flow)
    if Q.width < 0.006 and Q.mean_phi < -0.0044:
        z += 7290.0 * (0.006 - Q.width) * (-0.0044 - Q.mean_phi)
    if Q.width < 0.0053 and Q.n_dr_0p1_0p2 < 3.1:
        z += 128.0 * (0.0053 - Q.width) * (3.1 - Q.n_dr_0p1_0p2)
    if Q.width < 0.0056 and Q.n_dr_0p2_0p4 < 0.96:
        z += -445.0 * (0.0056 - Q.width) * (0.96 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.441
    if Q.C2 < 0.027:
        z += 92.2 * Q.C2 - 2.4894
    if Q.centroid_offset < 0.0034:
        z += -433.0 * Q.centroid_offset + 1.4722
    if Q.dr_0 < 0.016:
        z += 152.0 * Q.dr_0 - 2.432
    if Q.log_sum_pt >= 6.7:
        z += -16.9 * Q.log_sum_pt + 113.22999999999999
    if Q.pt_7 >= 34.0:
        z += -0.0347 * Q.pt_7 + 1.1798
    if Q.width < 0.0049:
        z += -2510.0 * Q.width + 12.299
    if Q.LHA < 0.19 and Q.girth2 > 0.0075:
        z += 178000.0 * (0.19 - Q.LHA) * (Q.girth2 - 0.0075)
    if Q.LHA < 0.2 and Q.mean_phi > -0.00075:
        z += -912.0 * (0.2 - Q.LHA) * (Q.mean_phi - -0.00075)
    if Q.LHA < 0.2 and Q.width < 0.00067:
        z += 51300.0 * (0.2 - Q.LHA) * (0.00067 - Q.width)
    if Q.girth < 0.057 and Q.log_sum_pt > 6.7:
        z += 219.0 * (0.057 - Q.girth) * (Q.log_sum_pt - 6.7)
    if Q.girth < 0.063 and Q.width < 0.0055:
        z += -31300.0 * (0.063 - Q.girth) * (0.0055 - Q.width)
    if Q.girth2 < 0.0067 and Q.centroid_offset < 0.025:
        z += 24600.0 * (0.0067 - Q.girth2) * (0.025 - Q.centroid_offset)
    if Q.girth2 < 0.0065 and Q.planar_flow < 0.38:
        z += 523.0 * (0.0065 - Q.girth2) * (0.38 - Q.planar_flow)
    if Q.girth2_top5 < 0.00022 and Q.n_dr_0p2_0p4 > 0.029:
        z += 23500.0 * (0.00022 - Q.girth2_top5) * (Q.n_dr_0p2_0p4 - 0.029)
    if Q.log_sum_pt > 6.7 and Q.pt_7 < 50.0:
        z += 0.155 * (Q.log_sum_pt - 6.7) * (50.0 - Q.pt_7)
    if Q.mass < 22.0 and Q.centroid_offset > 0.031:
        z += -28.0 * (22.0 - Q.mass) * (Q.centroid_offset - 0.031)
    if Q.mass < 29.0 and Q.centroid_offset < 0.024:
        z += -7.46 * (29.0 - Q.mass) * (0.024 - Q.centroid_offset)
    if Q.mass < 22.0 and Q.max_pair_mass > 13.0:
        z += 61.5 * (22.0 - Q.mass) * (Q.max_pair_mass - 13.0)
    if Q.max_dr < 0.18 and Q.lam2 < 0.00021:
        z += 85800.0 * (0.18 - Q.max_dr) * (0.00021 - Q.lam2)
    if Q.width < 0.0051 and Q.centroid_offset > 0.0071:
        z += -60500.0 * (0.0051 - Q.width) * (Q.centroid_offset - 0.0071)
    if Q.width < 0.0048 and Q.mass_over_sum_pt_sq > 0.00045:
        z += -402000.0 * (0.0048 - Q.width) * (Q.mass_over_sum_pt_sq - 0.00045)
    if Q.z_dr_0_0p05 > 0.87 and Q.lam2 < 0.00054:
        z += -18700.0 * (Q.z_dr_0_0p05 - 0.87) * (0.00054 - Q.lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.82
    if Q.C2 >= 0.051:
        z += 29.4 * Q.C2 - 1.4993999999999998
    if Q.centroid_offset < 0.018:
        z += -151.0 * Q.centroid_offset + 2.718
    if Q.girth < 0.055:
        z += 92.3 * Q.girth - 5.0765
    if Q.girth2 >= 0.018:
        z += 326.0 * Q.girth2 - 5.867999999999999
    if Q.lam2 >= 0.0017:
        z += 376.0 * Q.lam2 - 0.6392
    if Q.log_sum_pt >= 6.4:
        z += -2.06 * Q.log_sum_pt + 13.184000000000001
    if Q.mass_over_sum_pt < 0.076:
        z += 127.0 * Q.mass_over_sum_pt - 9.652
    if Q.mass_over_sum_pt_sq < 0.0033:
        z += -1400.0 * Q.mass_over_sum_pt_sq + 4.62
    if Q.max_dr < 0.23:
        z += -12.1 * Q.max_dr + 2.783
    if Q.mean_phi < 0.00042:
        z += 34.1 * Q.mean_phi - 0.014322000000000001
    if Q.n_dr_0p2_0p4 >= 0.9:
        z += 1.09 * Q.n_dr_0p2_0p4 - 0.9810000000000001
    if Q.sum_pt >= 850.0:
        z += -0.00911 * Q.sum_pt + 7.7435
    if Q.width < 0.0061:
        z += -3030.0 * Q.width + 18.483
    if Q.C2 > 0.048 and Q.dr_5 < 0.19:
        z += -230.0 * (Q.C2 - 0.048) * (0.19 - Q.dr_5)
    if Q.C2 > 0.051 and Q.pt_2 < 85.0:
        z += -1.03 * (Q.C2 - 0.051) * (85.0 - Q.pt_2)
    if Q.centroid_offset < 0.018 and Q.pt_4 > 49.0:
        z += 3.08 * (0.018 - Q.centroid_offset) * (Q.pt_4 - 49.0)
    if Q.centroid_offset < 0.018 and Q.z_4 > 0.043:
        z += -2590.0 * (0.018 - Q.centroid_offset) * (Q.z_4 - 0.043)
    if Q.centroid_offset < 0.018 and Q.z_5 > 0.033:
        z += -1200.0 * (0.018 - Q.centroid_offset) * (Q.z_5 - 0.033)
    if Q.e2 < 0.017 and Q.centroid_offset < 0.024:
        z += 11400.0 * (0.017 - Q.e2) * (0.024 - Q.centroid_offset)
    if Q.e2 < 0.032 and Q.dr01 < 0.056:
        z += 823.0 * (0.032 - Q.e2) * (0.056 - Q.dr01)
    if Q.e2 < 0.019 and Q.pt_7 < 53.0:
        z += -2.29 * (0.019 - Q.e2) * (53.0 - Q.pt_7)
    if Q.e2 < 0.032 and Q.tau21 < 0.45:
        z += -265.0 * (0.032 - Q.e2) * (0.45 - Q.tau21)
    if Q.girth2 < 0.0046 and Q.centroid_offset > 0.011:
        z += -60200.0 * (0.0046 - Q.girth2) * (Q.centroid_offset - 0.011)
    if Q.girth2 > 0.019 and Q.mean_eta < 0.025:
        z += -2270.0 * (Q.girth2 - 0.019) * (0.025 - Q.mean_eta)
    if Q.girth2 > 0.018 and Q.planar_flow < 0.36:
        z += -581.0 * (Q.girth2 - 0.018) * (0.36 - Q.planar_flow)
    if Q.girth2_top2 < 0.00075 and Q.pt_7 > 33.0:
        z += 111.0 * (0.00075 - Q.girth2_top2) * (Q.pt_7 - 33.0)
    if Q.girth2_top2 < 0.00073 and Q.z_7 > 0.023:
        z += -69100.0 * (0.00073 - Q.girth2_top2) * (Q.z_7 - 0.023)
    if Q.lam2 > 0.001 and Q.mass_top2 > 16.0:
        z += 10.8 * (Q.lam2 - 0.001) * (Q.mass_top2 - 16.0)
    if Q.log_sum_pt > 6.4 and Q.mean_phi > 0.026:
        z += -2120.0 * (Q.log_sum_pt - 6.4) * (Q.mean_phi - 0.026)
    if Q.mass < 43.0 and Q.centroid_offset < 0.027:
        z += -6.37 * (43.0 - Q.mass) * (0.027 - Q.centroid_offset)
    if Q.mass < 55.0 and Q.centroid_offset < 0.026:
        z += 5.17 * (55.0 - Q.mass) * (0.026 - Q.centroid_offset)
    if Q.mass < 52.0 and Q.log_sum_pt < 6.8:
        z += 0.19 * (52.0 - Q.mass) * (6.8 - Q.log_sum_pt)
    if Q.mass < 33.0 and Q.planar_flow < 0.32:
        z += -0.335 * (33.0 - Q.mass) * (0.32 - Q.planar_flow)
    if Q.mass < 51.0 and Q.planar_flow < 0.32:
        z += 0.219 * (51.0 - Q.mass) * (0.32 - Q.planar_flow)
    if Q.mass_over_sum_pt < 0.072 and Q.girth2_top2 < 0.00078:
        z += -35700.0 * (0.072 - Q.mass_over_sum_pt) * (0.00078 - Q.girth2_top2)
    if Q.n_dr_0p2_0p4 > 0.87 and Q.dr_6 > 0.22:
        z += 9.75 * (Q.n_dr_0p2_0p4 - 0.87) * (Q.dr_6 - 0.22)
    if Q.width < 0.006 and Q.C2 > 0.031:
        z += -9190.0 * (0.006 - Q.width) * (Q.C2 - 0.031)
    if Q.width < 0.0059 and Q.centroid_offset > 0.0029:
        z += -48100.0 * (0.0059 - Q.width) * (Q.centroid_offset - 0.0029)
    return max(0.0, z)


def neuron_10(Q):
    z = 2.32
    if Q.C2 >= 0.055:
        z += 24.2 * Q.C2 - 1.331
    if Q.LHA >= 0.33:
        z += -17.0 * Q.LHA + 5.61
    if Q.centroid_offset >= 0.038:
        z += 17.5 * Q.centroid_offset - 0.665
    if Q.e2 < 0.036:
        z += 56.7 * Q.e2 - 2.0979
    if 0.036 <= Q.e2 < 0.037:
        z += 102.6 * Q.e2 - 3.7503
    if Q.e2 >= 0.037:
        z += 45.9 * Q.e2 - 1.6523999999999999
    if Q.girth2 < 0.0017:
        z += 1460.0 * Q.girth2 - 2.4819999999999998
    if Q.girth2 >= 0.0085:
        z += 494.0 * Q.girth2 - 4.199000000000001
    if Q.girth2_top2 < 0.0038:
        z += -349.0 * Q.girth2_top2 + 1.3262
    if Q.girth2_top3 < 0.0016:
        z += -463.0 * Q.girth2_top3 + 0.7408
    if Q.lam1 < 0.0044:
        z += 271.0 * Q.lam1 - 1.1924000000000001
    if Q.lam1 >= 0.0085:
        z += -347.0 * Q.lam1 + 2.9495
    if Q.lam2 < 0.0034:
        z += 835.0 * Q.lam2 - 2.839
    if Q.log_sum_pt < 6.3:
        z += 5.54 * Q.log_sum_pt - 34.902
    if Q.log_sum_pt >= 6.7:
        z += -8.24 * Q.log_sum_pt + 55.208000000000006
    if Q.mass >= 17.0:
        z += 0.0237 * Q.mass - 0.4029
    if Q.mass_over_sum_pt < 0.1:
        z += -41.3 * Q.mass_over_sum_pt + 4.13
    if Q.n_dr_0p05_0p1 < 3.0:
        z += -0.183 * Q.n_dr_0p05_0p1 + 0.5489999999999999
    if Q.n_dr_0p2_0p4 >= 1.9:
        z += 0.756 * Q.n_dr_0p2_0p4 - 1.4364
    if Q.pt_7 >= 46.0:
        z += 0.0331 * Q.pt_7 - 1.5226
    if Q.tau32 < 0.27:
        z += -9.95 * Q.tau32 + 2.6865
    if Q.z_7 >= 0.062:
        z += -18.0 * Q.z_7 + 1.116
    if Q.LHA > 0.3 and Q.tau21 < 0.57:
        z += -40.7 * (Q.LHA - 0.3) * (0.57 - Q.tau21)
    if Q.eccentricity > 0.89 and Q.mass_top2 < 37.0:
        z += -0.145 * (Q.eccentricity - 0.89) * (37.0 - Q.mass_top2)
    if Q.eccentricity > 0.89 and Q.z_dr_0p2_0p4 < 0.044:
        z += 149.0 * (Q.eccentricity - 0.89) * (0.044 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.0044 and Q.log_sum_pt > 6.8:
        z += 2070.0 * (0.0044 - Q.lam1) * (Q.log_sum_pt - 6.8)
    if Q.lam1 > 0.0083 and Q.min_pair_mass < 2.8:
        z += -16.3 * (Q.lam1 - 0.0083) * (2.8 - Q.min_pair_mass)
    if Q.lam2 > 0.00048 and Q.n_pt_above_50 < 8.1:
        z += -49.5 * (Q.lam2 - 0.00048) * (8.1 - Q.n_pt_above_50)
    if Q.lam2 > 0.00023 and Q.tau21 < 0.51:
        z += 1370.0 * (Q.lam2 - 0.00023) * (0.51 - Q.tau21)
    if Q.mass > 8.0 and Q.n_dr_0p2_0p4 < 1.9:
        z += -0.00332 * (Q.mass - 8.0) * (1.9 - Q.n_dr_0p2_0p4)
    if Q.tau32 < 0.27 and Q.n_dr_0p2_0p4 < 2.0:
        z += -3.53 * (0.27 - Q.tau32) * (2.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.351
    if Q.C2 < 0.035:
        z += 27.1 * Q.C2 - 0.9485000000000001
    if Q.centroid_offset < 0.014:
        z += -98.3 * Q.centroid_offset + 3.932
    if 0.014 <= Q.centroid_offset < 0.04:
        z += -182.89999999999998 * Q.centroid_offset + 5.1164
    if Q.centroid_offset >= 0.04:
        z += -84.6 * Q.centroid_offset + 1.1844
    if Q.e2_sq < 0.0062:
        z += 393.0 * Q.e2_sq - 2.4366
    if Q.girth < 0.021:
        z += 139.4 * Q.girth - 7.1366
    if 0.021 <= Q.girth < 0.076:
        z += 61.9 * Q.girth - 5.509099999999999
    if 0.076 <= Q.girth < 0.089:
        z += -41.1 * Q.girth + 2.3189
    if Q.girth >= 0.089:
        z += -103.0 * Q.girth + 7.827999999999999
    if Q.girth2_top3 < 0.002:
        z += 399.0 * Q.girth2_top3 - 0.798
    if Q.mass_top5 < 5.5:
        z += -0.103 * Q.mass_top5 + 0.5665
    if Q.max_dr < 0.22:
        z += -6.36 * Q.max_dr + 1.3992
    if Q.n_dr_0p1_0p2 < 2.8:
        z += 0.16 * Q.n_dr_0p1_0p2 - 0.44799999999999995
    if Q.planar_flow < 0.27:
        z += -9.88 * Q.planar_flow + 2.6676
    if Q.sum_pt_top5 < 690.0:
        z += -0.00903 * Q.sum_pt_top5 + 6.2307
    if Q.width < 0.0036:
        z += -895.0 * Q.width + 9.690000000000001
    if 0.0036 <= Q.width < 0.0085:
        z += -1320.0 * Q.width + 11.22
    if Q.LHA < 0.16 and Q.z_7 < 0.028:
        z += 1400.0 * (0.16 - Q.LHA) * (0.028 - Q.z_7)
    if Q.centroid_offset < 0.05 and Q.log_sum_pt < 6.8:
        z += -133.0 * (0.05 - Q.centroid_offset) * (6.8 - Q.log_sum_pt)
    if Q.centroid_offset > 0.015 and Q.tau21 < 0.11:
        z += 1650.0 * (Q.centroid_offset - 0.015) * (0.11 - Q.tau21)
    if Q.e2_sq < 0.0062 and Q.mean_phi > 0.0014:
        z += 15100.0 * (0.0062 - Q.e2_sq) * (Q.mean_phi - 0.0014)
    if Q.e2_sq < 0.0063 and Q.mean_phi < -1.8e-05:
        z += 14900.0 * (0.0063 - Q.e2_sq) * (-1.8e-05 - Q.mean_phi)
    if Q.girth > 0.077 and Q.n_pt_above_50 < 7.0:
        z += 27.4 * (Q.girth - 0.077) * (7.0 - Q.n_pt_above_50)
    if Q.girth > 0.089 and Q.n_pt_above_50 < 7.1:
        z += -37.2 * (Q.girth - 0.089) * (7.1 - Q.n_pt_above_50)
    if Q.girth > 0.075 and Q.pt_7 < 40.0:
        z += -4.29 * (Q.girth - 0.075) * (40.0 - Q.pt_7)
    if Q.girth2 < 0.013 and Q.mass_top5 > 49.0:
        z += 7.0 * (0.013 - Q.girth2) * (Q.mass_top5 - 49.0)
    if Q.girth2 < 0.013 and Q.mean_phi < -0.0014:
        z += -6780.0 * (0.013 - Q.girth2) * (-0.0014 - Q.mean_phi)
    if Q.girth2 < 0.014 and Q.mean_phi > 0.0032:
        z += -6840.0 * (0.014 - Q.girth2) * (Q.mean_phi - 0.0032)
    if Q.log_sum_pt < 6.7 and Q.dr_0 < 0.12:
        z += -27.8 * (6.7 - Q.log_sum_pt) * (0.12 - Q.dr_0)
    if Q.m01 > 46.0 and Q.n_pt_above_50 > 6.0:
        z += 0.931 * (Q.m01 - 46.0) * (Q.n_pt_above_50 - 6.0)
    if Q.planar_flow < 0.23 and Q.girth2 > 0.015:
        z += 3710.0 * (0.23 - Q.planar_flow) * (Q.girth2 - 0.015)
    if Q.planar_flow < 0.27 and Q.mass < 71.0:
        z += -0.147 * (0.27 - Q.planar_flow) * (71.0 - Q.mass)
    if Q.planar_flow < 0.26 and Q.max_dr > 0.11:
        z += -33.1 * (0.26 - Q.planar_flow) * (Q.max_dr - 0.11)
    if Q.planar_flow < 0.21 and Q.width > 0.0056:
        z += -1310.0 * (0.21 - Q.planar_flow) * (Q.width - 0.0056)
    if Q.pt_7 < 30.0 and Q.n_dr_0p2_0p4 < 1.9:
        z += -0.0413 * (30.0 - Q.pt_7) * (1.9 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.38
    if Q.e2 >= 0.063:
        z += -119.0 * Q.e2 + 7.497
    if Q.girth2 >= 0.019:
        z += 232.0 * Q.girth2 - 4.4079999999999995
    if Q.mass >= 91.2:
        z += 0.0994 * Q.mass - 9.065280000000001
    if Q.girth2 > 0.015 and Q.lam2 > 5.6e-06:
        z += 8450.0 * (Q.girth2 - 0.015) * (Q.lam2 - 5.6e-06)
    if Q.girth2 > 0.019 and Q.pt_7 > 15.0:
        z += 6.87 * (Q.girth2 - 0.019) * (Q.pt_7 - 15.0)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.817
    if Q.C2 >= 0.066:
        z += -52.5 * Q.C2 + 3.4650000000000003
    if Q.centroid_offset < 0.038:
        z += -31.6 * Q.centroid_offset + 1.2008
    if Q.e2 < 0.049:
        z += 48.1 * Q.e2 - 2.3569
    if Q.girth < 0.15:
        z += -64.5 * Q.girth + 9.674999999999999
    if Q.lam1 < 0.0062:
        z += -596.0 * Q.lam1 + 5.3408
    if 0.0062 <= Q.lam1 < 0.015:
        z += -187.0 * Q.lam1 + 2.8049999999999997
    if Q.pt_7 < 25.0:
        z += 0.139 * Q.pt_7 - 3.4750000000000005
    if Q.sum_pt >= 1000.0:
        z += -0.0233 * Q.sum_pt + 23.3
    if Q.tau21 < 0.5:
        z += 1.34 * Q.tau21 - 0.67
    if Q.width < 0.0076:
        z += 321.0 * Q.width - 2.4396
    if Q.z_7 < 0.027:
        z += 160.0 * Q.z_7 - 4.32
    if Q.e2 < 0.053 and Q.pt_dispersion > 0.4:
        z += 65.0 * (0.053 - Q.e2) * (Q.pt_dispersion - 0.4)
    if Q.girth < 0.14 and Q.log_sum_pt < 6.8:
        z += -42.8 * (0.14 - Q.girth) * (6.8 - Q.log_sum_pt)
    if Q.girth < 0.15 and Q.pt_7 < 39.0:
        z += -0.794 * (0.15 - Q.girth) * (39.0 - Q.pt_7)
    if Q.lam1 < 0.017 and Q.centroid_offset < 0.037:
        z += -2220.0 * (0.017 - Q.lam1) * (0.037 - Q.centroid_offset)
    if Q.lam1 < 0.007 and Q.z_dr_0p05_0p1 > 0.18:
        z += 277.0 * (0.007 - Q.lam1) * (Q.z_dr_0p05_0p1 - 0.18)
    if Q.lam2 < 0.00031 and Q.centroid_offset < 0.041:
        z += -77100.0 * (0.00031 - Q.lam2) * (0.041 - Q.centroid_offset)
    if Q.sum_pt > 980.0 and Q.D2 < 4.2:
        z += -0.0113 * (Q.sum_pt - 980.0) * (4.2 - Q.D2)
    if Q.sum_pt > 990.0 and Q.n_pt_above_50 > 6.0:
        z += 0.0277 * (Q.sum_pt - 990.0) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt < 760.0 and Q.z_4 < 0.037:
        z += 7.44 * (760.0 - Q.sum_pt) * (0.037 - Q.z_4)
    if Q.sum_pt_top5 > 900.0 and Q.D2 < 4.3:
        z += 0.0103 * (Q.sum_pt_top5 - 900.0) * (4.3 - Q.D2)
    if Q.sum_pt_top5 > 830.0 and Q.n_pt_above_50 > 0.88:
        z += 0.00378 * (Q.sum_pt_top5 - 830.0) * (Q.n_pt_above_50 - 0.88)
    if Q.sum_pt_top5 > 900.0 and Q.n_pt_above_50 > 6.1:
        z += 0.0372 * (Q.sum_pt_top5 - 900.0) * (Q.n_pt_above_50 - 6.1)
    if Q.sum_pt_top5 < 520.0 and Q.pt_5 < 30.0:
        z += 0.00149 * (520.0 - Q.sum_pt_top5) * (30.0 - Q.pt_5)
    if Q.sum_pt_top5 > 640.0 and Q.pt_7 < 45.0:
        z += 0.000499 * (Q.sum_pt_top5 - 640.0) * (45.0 - Q.pt_7)
    if Q.sum_pt_top5 > 650.0 and Q.tau32 < 0.37:
        z += -0.0396 * (Q.sum_pt_top5 - 650.0) * (0.37 - Q.tau32)
    if Q.sum_pt_top5 > 660.0 and Q.z_7 > 0.023:
        z += -0.559 * (Q.sum_pt_top5 - 660.0) * (Q.z_7 - 0.023)
    if Q.tau21 < 0.53 and Q.max_dr > 0.013:
        z += -13.9 * (0.53 - Q.tau21) * (Q.max_dr - 0.013)
    if Q.width < 0.0074 and Q.m012 > 16.0:
        z += -12.0 * (0.0074 - Q.width) * (Q.m012 - 16.0)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.0316
    if Q.C2 < 0.067:
        z += 24.2 * Q.C2 - 1.6214
    if Q.D2 < 0.8:
        z += 2.01 * Q.D2 - 1.6079999999999999
    if 0.031 <= Q.centroid_offset < 0.05:
        z += -81.4 * Q.centroid_offset + 2.5234
    if Q.centroid_offset >= 0.05:
        z += -377.4 * Q.centroid_offset + 17.3234
    if Q.e2 < 0.043:
        z += -119.0 * Q.e2 + 5.117
    if Q.e2_sq < 0.0058:
        z += -700.0 * Q.e2_sq + 4.06
    if Q.girth < 0.034:
        z += 182.5 * Q.girth - 10.927299999999999
    if 0.034 <= Q.girth < 0.087:
        z += 89.1 * Q.girth - 7.751699999999999
    if Q.girth2 < 0.013:
        z += -487.0 * Q.girth2 + 6.3309999999999995
    if 0.0025 <= Q.lam1 < 0.0042:
        z += -542.0 * Q.lam1 + 1.355
    if 0.0042 <= Q.lam1 < 0.0061:
        z += 538.0 * Q.lam1 - 3.1809999999999996
    if Q.lam1 >= 0.0061:
        z += 60.0 * Q.lam1 - 0.2651999999999992
    if Q.mass_over_sum_pt_sq < 0.0081:
        z += 443.0 * Q.mass_over_sum_pt_sq - 3.5883
    if Q.max_dr < 0.078:
        z += 6.5 * Q.max_dr - 1.782
    if 0.078 <= Q.max_dr < 0.18:
        z += 12.5 * Q.max_dr - 2.25
    if Q.n_dr_0p05_0p1 < 4.8:
        z += 0.122 * Q.n_dr_0p05_0p1 - 0.5856
    if Q.sum_pt_top3 < 310.0:
        z += 0.00771 * Q.sum_pt_top3 - 2.3901
    if Q.tau21 < 0.14:
        z += -9.9 * Q.tau21 + 1.3860000000000001
    if Q.width < 0.0061:
        z += 1210.0 * Q.width - 7.381
    if Q.z_dr_0p05_0p1 < 0.6:
        z += -1.38 * Q.z_dr_0p05_0p1 + 0.828
    if Q.D2 < 1.2 and Q.centroid_offset < 0.03:
        z += 41.5 * (1.2 - Q.D2) * (0.03 - Q.centroid_offset)
    if Q.e2 < 0.041 and Q.D2 < 0.99:
        z += 304.0 * (0.041 - Q.e2) * (0.99 - Q.D2)
    if Q.girth2 < 0.013 and Q.eccentricity > 0.97:
        z += 11300.0 * (0.013 - Q.girth2) * (Q.eccentricity - 0.97)
    if Q.girth2 < 0.013 and Q.mass_top3 > 24.0:
        z += 5.76 * (0.013 - Q.girth2) * (Q.mass_top3 - 24.0)
    if Q.girth2 < 0.013 and Q.n_dr_0p1_0p2 < 3.0:
        z += -45.7 * (0.013 - Q.girth2) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.girth2 < 0.0043 and Q.n_dr_0p2_0p4 < 1.1:
        z += -384.0 * (0.0043 - Q.girth2) * (1.1 - Q.n_dr_0p2_0p4)
    if Q.lam1 > 0.0025 and Q.D2 > 0.4:
        z += 760.0 * (Q.lam1 - 0.0025) * (Q.D2 - 0.4)
    if Q.lam1 > 0.0025 and Q.D2 > 1.6:
        z += -503.0 * (Q.lam1 - 0.0025) * (Q.D2 - 1.6)
    if Q.lam1 > 0.0042 and Q.D2 > 0.41:
        z += -857.0 * (Q.lam1 - 0.0042) * (Q.D2 - 0.41)
    if Q.lam1 > 0.0054 and Q.max_dr < 0.16:
        z += 9530.0 * (Q.lam1 - 0.0054) * (0.16 - Q.max_dr)
    if Q.planar_flow < 0.11 and Q.centroid_offset > 0.0094:
        z += 1340.0 * (0.11 - Q.planar_flow) * (Q.centroid_offset - 0.0094)
    if Q.planar_flow < 0.11 and Q.centroid_offset > 0.018:
        z += -1670.0 * (0.11 - Q.planar_flow) * (Q.centroid_offset - 0.018)
    if Q.planar_flow < 0.11 and Q.max_dr < 0.16:
        z += -202.0 * (0.11 - Q.planar_flow) * (0.16 - Q.max_dr)
    if Q.planar_flow < 0.11 and Q.sum_pt < 750.0:
        z += -0.0475 * (0.11 - Q.planar_flow) * (750.0 - Q.sum_pt)
    if Q.width < 0.0076 and Q.D2 < 1.0:
        z += -1950.0 * (0.0076 - Q.width) * (1.0 - Q.D2)
    if Q.width < 0.0074 and Q.mass_top3 > 24.0:
        z += -18.5 * (0.0074 - Q.width) * (Q.mass_top3 - 24.0)
    if Q.width < 0.006 and Q.mean_eta < -0.027:
        z += 35400.0 * (0.006 - Q.width) * (-0.027 - Q.mean_eta)
    if Q.width < 0.006 and Q.mean_phi < -0.026:
        z += 35800.0 * (0.006 - Q.width) * (-0.026 - Q.mean_phi)
    if Q.width < 0.0077 and Q.n_dr_0p1_0p2 < 3.0:
        z += 87.0 * (0.0077 - Q.width) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.width < 0.0075 and Q.planar_flow < 0.11:
        z += -6520.0 * (0.0075 - Q.width) * (0.11 - Q.planar_flow)
    if Q.z_dr_0p05_0p1 < 0.58 and Q.C2 < 0.068:
        z += -44.3 * (0.58 - Q.z_dr_0p05_0p1) * (0.068 - Q.C2)
    if Q.z_dr_0p05_0p1 < 0.6 and Q.n_dr_0p1_0p2 < 3.0:
        z += 0.317 * (0.6 - Q.z_dr_0p05_0p1) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.z_dr_0p05_0p1 > 0.74 and Q.n_dr_0p2_0p4 < 1.0:
        z += 2.71 * (Q.z_dr_0p05_0p1 - 0.74) * (1.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.86
    if Q.LHA >= 0.34:
        z += -35.4 * Q.LHA + 12.036
    if Q.e2 < 0.024:
        z += -308.2 * Q.e2 + 8.488199999999999
    if 0.024 <= Q.e2 < 0.041:
        z += -64.2 * Q.e2 + 2.6322
    if Q.girth >= 0.032:
        z += 33.7 * Q.girth - 1.0784
    if Q.girth2_top2 < 0.0038:
        z += 346.1 * Q.girth2_top2 - 0.99154
    if 0.0038 <= Q.girth2_top2 < 0.0074:
        z += -89.9 * Q.girth2_top2 + 0.6652600000000001
    if Q.lam1 < 0.0067:
        z += -279.0 * Q.lam1 + 0.8228999999999997
    if 0.0067 <= Q.lam1 < 0.0083:
        z += 654.0 * Q.lam1 - 5.4282
    if Q.lam2 < 0.00031:
        z += 3280.0 * Q.lam2 - 1.0168
    if Q.log_sum_pt >= 6.9:
        z += 48.6 * Q.log_sum_pt - 335.34000000000003
    if Q.mass_over_sum_pt < 0.068:
        z += 27.1 * Q.mass_over_sum_pt - 1.8428000000000002
    if Q.width < 0.0067:
        z += 1075.0 * Q.width - 5.218000000000001
    if 0.0067 <= Q.width < 0.013:
        z += -315.0 * Q.width + 4.095
    if Q.z_dr_0p05_0p1 >= 0.75:
        z += -5.8 * Q.z_dr_0p05_0p1 + 4.35
    if Q.z_dr_0p1_0p2 < 0.32:
        z += -3.68 * Q.z_dr_0p1_0p2 + 1.1776
    if Q.LHA > 0.31 and Q.pt_dispersion < 0.45:
        z += -163.0 * (Q.LHA - 0.31) * (0.45 - Q.pt_dispersion)
    if Q.LHA > 0.18 and Q.sum_pt_top3 > 350.0:
        z += 0.0494 * (Q.LHA - 0.18) * (Q.sum_pt_top3 - 350.0)
    if Q.lam1 < 0.0082 and Q.D2 < 0.76:
        z += -852.0 * (0.0082 - Q.lam1) * (0.76 - Q.D2)
    if Q.lam1 < 0.0083 and Q.m01 > 17.0:
        z += -24.5 * (0.0083 - Q.lam1) * (Q.m01 - 17.0)
    if Q.lam1 < 0.0065 and Q.pt1_dr01 > 1.3:
        z += 33.0 * (0.0065 - Q.lam1) * (Q.pt1_dr01 - 1.3)
    if Q.log_sum_pt > 6.9 and Q.mean_phi > 0.017:
        z += 17900.0 * (Q.log_sum_pt - 6.9) * (Q.mean_phi - 0.017)
    if Q.mass > 80.4 and Q.z_dr_0p2_0p4 < 0.22:
        z += -0.665 * (Q.mass - 80.4) * (0.22 - Q.z_dr_0p2_0p4)
    if Q.n_dr_0_0p05 < 1.8 and Q.pt_dispersion < 0.43:
        z += 5.81 * (1.8 - Q.n_dr_0_0p05) * (0.43 - Q.pt_dispersion)
    if Q.tau21 < 0.27 and Q.girth2_top5 > 0.0084:
        z += -1160.0 * (0.27 - Q.tau21) * (Q.girth2_top5 - 0.0084)
    if Q.tau21 < 0.23 and Q.z_dr_0p05_0p1 < 0.63:
        z += -7.98 * (0.23 - Q.tau21) * (0.63 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.23 and Q.z_dr_0p2_0p4 < 0.22:
        z += 69.7 * (0.23 - Q.tau21) * (0.22 - Q.z_dr_0p2_0p4)
    if Q.width < 0.0076 and Q.e2 > 0.024:
        z += -66400.0 * (0.0076 - Q.width) * (Q.e2 - 0.024)
    if Q.width < 0.0061 and Q.log_sum_pt > 6.9:
        z += -10600.0 * (0.0061 - Q.width) * (Q.log_sum_pt - 6.9)
    if Q.width < 0.0086 and Q.planar_flow < 0.064:
        z += -2630.0 * (0.0086 - Q.width) * (0.064 - Q.planar_flow)
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
