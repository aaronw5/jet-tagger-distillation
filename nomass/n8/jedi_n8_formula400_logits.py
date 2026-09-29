"""JEDI-linear jet tagger, 8 particles, 3 features: simplified from the formula tuned on the network (624), keeping validation agreement with the network within 0.5 point, as if-statements, with each class score (logit) written out as a formula.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 90.0% of jets.

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
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
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
        mass_top3=mass_of(3),
        mass_top5=mass_of(5),
        max_dr=max(dr[i] for i in real),
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_top5_slots=sum(pt[:5]) / tot,
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
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
    z = -0.847
    if Q.centroid_offset < 0.0327:
        z += -32.2 * Q.centroid_offset + 1.05294
    if Q.e2 < 0.0206:
        z += -134.0 * Q.e2 + 2.7604
    if Q.girth < 0.0764:
        z += 122.0 * Q.girth - 9.3208
    if Q.girth2 < 0.0133:
        z += -355.0 * Q.girth2 + 4.7215
    if Q.girth2_top3 < 0.00406:
        z += -218.0 * Q.girth2_top3 + 0.23546
    if 0.00406 <= Q.girth2_top3 < 0.00807:
        z += 162.0 * Q.girth2_top3 - 1.3073400000000002
    if Q.lam1 < 0.000456:
        z += -4130.0 * Q.lam1 + 1.88328
    if Q.mass < 29.1:
        z += 0.461 * Q.mass - 15.6111
    if 29.1 <= Q.mass < 59.6:
        z += 0.072 * Q.mass - 4.2912
    if 821.0 <= Q.sum_pt < 889.0:
        z += -0.0129 * Q.sum_pt + 10.5909
    if Q.sum_pt >= 889.0:
        z += -0.0236 * Q.sum_pt + 20.1032
    if Q.sum_pt_top5 >= 698.0:
        z += 0.00831 * Q.sum_pt_top5 - 5.80038
    if Q.width < 0.00435:
        z += 199.0 * Q.width + 1.1850000000000005
    if 0.00435 <= Q.width < 0.00876:
        z += -465.0 * Q.width + 4.0734
    if Q.z_dr_0_0p05 >= 0.851:
        z += 5.86 * Q.z_dr_0_0p05 - 4.98686
    if Q.girth2 < 0.019 and Q.eccentricity > 0.955:
        z += 1590.0 * (0.019 - Q.girth2) * (Q.eccentricity - 0.955)
    if Q.lam1 < 0.00657 and Q.D2 < 0.872:
        z += -1490.0 * (0.00657 - Q.lam1) * (0.872 - Q.D2)
    if Q.log_sum_pt > 6.63 and Q.dr_4 < 0.0771:
        z += 42.6 * (Q.log_sum_pt - 6.63) * (0.0771 - Q.dr_4)
    if Q.mass < 56.0 and Q.C2 > 0.0235:
        z += 1.0 * (56.0 - Q.mass) * (Q.C2 - 0.0235)
    if Q.mass < 29.7 and Q.D2 < 0.874:
        z += -3.17 * (29.7 - Q.mass) * (0.874 - Q.D2)
    if Q.mass < 63.4 and Q.centroid_offset > 0.0108:
        z += 1.69 * (63.4 - Q.mass) * (Q.centroid_offset - 0.0108)
    if Q.mass < 28.9 and Q.dr_0 < 0.109:
        z += 3.42 * (28.9 - Q.mass) * (0.109 - Q.dr_0)
    if Q.mass < 64.0 and Q.pt_7 < 40.1:
        z += -0.00208 * (64.0 - Q.mass) * (40.1 - Q.pt_7)
    if Q.mass_over_sum_pt_sq < 0.00809 and Q.n_pt_above_50 < 7.9:
        z += 18.0 * (0.00809 - Q.mass_over_sum_pt_sq) * (7.9 - Q.n_pt_above_50)
    if Q.sum_pt > 876.0 and Q.pt_7 < 28.8:
        z += 0.000974 * (Q.sum_pt - 876.0) * (28.8 - Q.pt_7)
    if Q.sum_pt > 904.0 and Q.pt_7 > 29.4:
        z += -0.00248 * (Q.sum_pt - 904.0) * (Q.pt_7 - 29.4)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.823
    if Q.LHA >= 0.259:
        z += 8.82 * Q.LHA - 2.28438
    if Q.e2 >= 0.0311:
        z += -16.5 * Q.e2 + 0.51315
    if Q.e2_sq < 0.00817:
        z += -404.0 * Q.e2_sq + 3.3006800000000003
    if 6.38 <= Q.log_sum_pt < 6.63:
        z += 4.7 * Q.log_sum_pt - 29.986
    if Q.log_sum_pt >= 6.63:
        z += 11.719999999999999 * Q.log_sum_pt - 76.5286
    if Q.mass < 55.5:
        z += 0.025 * Q.mass - 1.3875000000000002
    if Q.max_dr < 0.249:
        z += 7.11 * Q.max_dr - 1.7703900000000001
    if 34.5 <= Q.pt_7 < 52.6:
        z += 0.155 * Q.pt_7 - 5.3475
    if Q.pt_7 >= 52.6:
        z += 0.05 * Q.pt_7 + 0.17549999999999955
    if Q.width < 0.00866:
        z += 295.0 * Q.width - 2.5547
    if Q.z_7 < 0.043:
        z += 121.4 * Q.z_7 - 5.9352
    if 0.043 <= Q.z_7 < 0.0555:
        z += 57.2 * Q.z_7 - 3.1746000000000003
    if Q.e2 < 0.0346 and Q.eccentricity > 0.979:
        z += 6960.0 * (0.0346 - Q.e2) * (Q.eccentricity - 0.979)
    if Q.e2_sq < 0.0085 and Q.planar_flow < 0.0833:
        z += -5800.0 * (0.0085 - Q.e2_sq) * (0.0833 - Q.planar_flow)
    if Q.lam1 < 0.00818 and Q.centroid_offset > 0.0213:
        z += 14200.0 * (0.00818 - Q.lam1) * (Q.centroid_offset - 0.0213)
    if Q.lam1 < 0.00818 and Q.lam2 < 0.00054:
        z += -428000.0 * (0.00818 - Q.lam1) * (0.00054 - Q.lam2)
    if Q.log_sum_pt > 6.37 and Q.centroid_offset < 0.0246:
        z += -93.8 * (Q.log_sum_pt - 6.37) * (0.0246 - Q.centroid_offset)
    if Q.log_sum_pt > 6.64 and Q.girth2_top3 < 0.00799:
        z += -679.0 * (Q.log_sum_pt - 6.64) * (0.00799 - Q.girth2_top3)
    if Q.log_sum_pt > 6.57 and Q.lam2 < 0.00118:
        z += 4120.0 * (Q.log_sum_pt - 6.57) * (0.00118 - Q.lam2)
    if Q.log_sum_pt > 6.38 and Q.max_dr < 0.2:
        z += 22.4 * (Q.log_sum_pt - 6.38) * (0.2 - Q.max_dr)
    if Q.pt_7 > 34.8 and Q.mass < 91.2:
        z += -0.00174 * (Q.pt_7 - 34.8) * (91.2 - Q.mass)
    if Q.pt_7 > 35.0 and Q.max_dr < 0.0787:
        z += 0.917 * (Q.pt_7 - 35.0) * (0.0787 - Q.max_dr)
    if Q.z_7 < 0.0559 and Q.girth2_top2 < 0.0139:
        z += 3150.0 * (0.0559 - Q.z_7) * (0.0139 - Q.girth2_top2)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.21
    if Q.LHA >= 0.119:
        z += -11.6 * Q.LHA + 1.3803999999999998
    if Q.lam1 < 0.00333:
        z += -396.0 * Q.lam1 + 1.65618
    if 0.00333 <= Q.lam1 < 0.00583:
        z += -135.0 * Q.lam1 + 0.78705
    if Q.log_sum_pt < 6.49:
        z += -3.81 * Q.log_sum_pt + 24.7269
    if Q.log_sum_pt >= 6.9:
        z += -8.8 * Q.log_sum_pt + 60.720000000000006
    if Q.mass < 36.4:
        z += 0.06369999999999999 * Q.mass - 1.5497799999999997
    if 36.4 <= Q.mass < 69.4:
        z += -0.0233 * Q.mass + 1.6170200000000001
    if Q.pt_7 < 30.4:
        z += -0.002599999999999998 * Q.pt_7 - 0.49916000000000027
    if 30.4 <= Q.pt_7 < 43.4:
        z += 0.0612 * Q.pt_7 - 2.4386799999999997
    if 43.4 <= Q.pt_7 < 53.4:
        z += 0.125 * Q.pt_7 - 5.207599999999999
    if Q.pt_7 >= 53.4:
        z += 0.0638 * Q.pt_7 - 1.9395199999999997
    if Q.sum_pt < 781.0:
        z += -0.00497 * Q.sum_pt + 3.8815699999999995
    if Q.lam1 < 0.00595 and Q.max_dr > 0.0801:
        z += -2270.0 * (0.00595 - Q.lam1) * (Q.max_dr - 0.0801)
    if Q.mass < 36.8 and Q.lam2 < 0.00114:
        z += 37.1 * (36.8 - Q.mass) * (0.00114 - Q.lam2)
    if Q.pt_7 > 30.6 and Q.C2 < 0.0507:
        z += -1.1 * (Q.pt_7 - 30.6) * (0.0507 - Q.C2)
    if Q.pt_7 > 30.5 and Q.max_dr > 0.0929:
        z += -0.536 * (Q.pt_7 - 30.5) * (Q.max_dr - 0.0929)
    return max(0.0, z)


def neuron_3(Q):
    z = 2.94
    if Q.C2 >= 0.0926:
        z += -165.0 * Q.C2 + 15.279
    if Q.centroid_offset >= 0.0146:
        z += 66.1 * Q.centroid_offset - 0.9650599999999999
    if Q.e2 < 0.0269:
        z += -199.0 * Q.e2 + 8.8555
    if 0.0269 <= Q.e2 < 0.0445:
        z += -253.2 * Q.e2 + 10.313479999999998
    if Q.e2 >= 0.0445:
        z += -54.2 * Q.e2 + 1.45798
    if Q.girth2 < 0.00389:
        z += 824.0 * Q.girth2 - 13.999310000000001
    if 0.00389 <= Q.girth2 < 0.00872:
        z += 1805.0 * Q.girth2 - 17.8154
    if 0.00872 <= Q.girth2 < 0.0126:
        z += 535.0 * Q.girth2 - 6.741
    if Q.lam1 >= 0.00353:
        z += -541.0 * Q.lam1 + 1.9097300000000001
    if Q.mass >= 69.5:
        z += -0.0793 * Q.mass + 5.511349999999999
    if Q.max_dr >= 0.14:
        z += 6.89 * Q.max_dr - 0.9646
    if Q.width >= 0.0182:
        z += 387.0 * Q.width - 7.0434
    if Q.z_dr_0p1_0p2 < 0.458:
        z += -2.43 * Q.z_dr_0p1_0p2 + 1.11294
    if Q.LHA > 0.312 and Q.max_dr < 0.147:
        z += -1760.0 * (Q.LHA - 0.312) * (0.147 - Q.max_dr)
    if Q.centroid_offset > 0.0105 and Q.pt_7 > 25.2:
        z += 1.16 * (Q.centroid_offset - 0.0105) * (Q.pt_7 - 25.2)
    if Q.girth2 < 0.00803 and Q.pt1_dr01 > 4.22:
        z += 21.9 * (0.00803 - Q.girth2) * (Q.pt1_dr01 - 4.22)
    if Q.mass > 36.0 and Q.eccentricity > 0.688:
        z += 0.099 * (Q.mass - 36.0) * (Q.eccentricity - 0.688)
    if Q.mass_over_sum_pt > 0.0689 and Q.dr_7 < 0.0428:
        z += -4920.0 * (Q.mass_over_sum_pt - 0.0689) * (0.0428 - Q.dr_7)
    if Q.mass_over_sum_pt > 0.109 and Q.dr_7 < 0.0487:
        z += 12100.0 * (Q.mass_over_sum_pt - 0.109) * (0.0487 - Q.dr_7)
    if Q.mass_over_sum_pt > 0.0904 and Q.max_dr < 0.149:
        z += 10100.0 * (Q.mass_over_sum_pt - 0.0904) * (0.149 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.0674 and Q.n_dr_0_0p05 < 4.63:
        z += 23.7 * (Q.mass_over_sum_pt - 0.0674) * (4.63 - Q.n_dr_0_0p05)
    if Q.max_dr > 0.141 and Q.dr_1 < 0.0468:
        z += -673.0 * (Q.max_dr - 0.141) * (0.0468 - Q.dr_1)
    return max(0.0, z)


def neuron_4(Q):
    z = 0.765
    if 0.0136 <= Q.C2 < 0.0674:
        z += -54.1 * Q.C2 + 0.73576
    if Q.C2 >= 0.0674:
        z += -164.1 * Q.C2 + 8.14976
    if Q.e2 >= 0.0214:
        z += 114.0 * Q.e2 - 2.4396
    if Q.e2_sq < 0.00295:
        z += 1680.0 * Q.e2_sq - 4.9559999999999995
    if Q.girth < 0.0766:
        z += -67.6 * Q.girth + 5.17816
    if 0.00354 <= Q.girth2 < 0.00739:
        z += 743.0 * Q.girth2 - 2.63022
    if Q.girth2 >= 0.00739:
        z += 60.0 * Q.girth2 + 2.41715
    if Q.mass < 61.3:
        z += -0.0624 * Q.mass + 3.8251199999999996
    if 0.09 <= Q.mass_over_sum_pt < 0.108:
        z += -254.0 * Q.mass_over_sum_pt + 22.86
    if Q.mass_over_sum_pt >= 0.108:
        z += 162.0 * Q.mass_over_sum_pt - 22.067999999999998
    if Q.sum_pt < 760.0:
        z += -0.00389 * Q.sum_pt + 2.9564
    if Q.tau21 < 0.236:
        z += -20.9 * Q.tau21 + 4.9323999999999995
    if Q.centroid_offset < 0.016 and Q.z_dr_0p05_0p1 < 0.541:
        z += -358.0 * (0.016 - Q.centroid_offset) * (0.541 - Q.z_dr_0p05_0p1)
    if Q.girth2 > 0.0187 and Q.pt_6 > 62.0:
        z += -254.0 * (Q.girth2 - 0.0187) * (Q.pt_6 - 62.0)
    if Q.girth2_top2 < 0.0107 and Q.centroid_offset > 0.0188:
        z += 10600.0 * (0.0107 - Q.girth2_top2) * (Q.centroid_offset - 0.0188)
    if Q.lam2 < 0.000297 and Q.mass_top3 < 43.8:
        z += -107.0 * (0.000297 - Q.lam2) * (43.8 - Q.mass_top3)
    if Q.mass_over_sum_pt > 0.108 and Q.D2 < 3.95:
        z += -96.4 * (Q.mass_over_sum_pt - 0.108) * (3.95 - Q.D2)
    if Q.max_dr > 0.105 and Q.eccentricity > 0.985:
        z += -927.0 * (Q.max_dr - 0.105) * (Q.eccentricity - 0.985)
    if Q.tau21 < 0.229 and Q.mass < 60.9:
        z += -0.618 * (0.229 - Q.tau21) * (60.9 - Q.mass)
    if Q.tau21 < 0.221 and Q.planar_flow > 0.0357:
        z += 45.7 * (0.221 - Q.tau21) * (Q.planar_flow - 0.0357)
    return max(0.0, z)


def neuron_5(Q):
    z = -0.779
    if Q.LHA < 0.224:
        z += -20.0 * Q.LHA + 4.48
    if Q.dr_0 < 0.0199:
        z += 113.0 * Q.dr_0 - 2.2487
    if Q.e2 < 0.0346:
        z += -39.8 * Q.e2 + 1.3770799999999999
    if 6.55 <= Q.log_sum_pt < 6.83:
        z += 8.09 * Q.log_sum_pt - 52.9895
    if Q.log_sum_pt >= 6.83:
        z += -18.81 * Q.log_sum_pt + 130.7375
    if Q.pt_7 < 54.7:
        z += -0.079 * Q.pt_7 + 4.3213
    if Q.sum_pt_top2 < 539.0:
        z += 0.00275 * Q.sum_pt_top2 - 1.4822499999999998
    if Q.width < 0.0025:
        z += -736.0 * Q.width + 1.84
    if Q.z_6 < 0.0513:
        z += -61.0 * Q.z_6 + 3.1292999999999997
    if Q.z_7 < 0.0233:
        z += -217.0 * Q.z_7 + 5.056100000000001
    if Q.LHA < 0.216 and Q.centroid_offset > 0.00235:
        z += -1910.0 * (0.216 - Q.LHA) * (Q.centroid_offset - 0.00235)
    if Q.LHA < 0.228 and Q.log_sum_pt < 6.81:
        z += -108.0 * (0.228 - Q.LHA) * (6.81 - Q.log_sum_pt)
    if Q.e2 < 0.0334 and Q.lam2 < 7.69e-05:
        z += 363000.0 * (0.0334 - Q.e2) * (7.69e-05 - Q.lam2)
    if Q.log_sum_pt > 6.9 and Q.centroid_offset < 0.0182:
        z += -2200.0 * (Q.log_sum_pt - 6.9) * (0.0182 - Q.centroid_offset)
    if Q.log_sum_pt > 6.58 and Q.dr_0 < 0.0208:
        z += 371.0 * (Q.log_sum_pt - 6.58) * (0.0208 - Q.dr_0)
    if Q.log_sum_pt > 6.55 and Q.lam1 < 0.0123:
        z += -985.0 * (Q.log_sum_pt - 6.55) * (0.0123 - Q.lam1)
    if Q.log_sum_pt > 6.58 and Q.mean_eta2 < 9.11e-05:
        z += 43800.0 * (Q.log_sum_pt - 6.58) * (9.11e-05 - Q.mean_eta2)
    if Q.log_sum_pt > 6.57 and Q.mean_phi2 < 0.000148:
        z += 28800.0 * (Q.log_sum_pt - 6.57) * (0.000148 - Q.mean_phi2)
    if Q.log_sum_pt > 6.88 and Q.planar_flow < 0.0438:
        z += -1180.0 * (Q.log_sum_pt - 6.88) * (0.0438 - Q.planar_flow)
    if Q.mean_phi2 < 0.0137 and Q.pt_5 < 24.6:
        z += 22.9 * (0.0137 - Q.mean_phi2) * (24.6 - Q.pt_5)
    if Q.sum_pt > 861.0 and Q.centroid_offset < 0.0117:
        z += 1.39 * (Q.sum_pt - 861.0) * (0.0117 - Q.centroid_offset)
    if Q.sum_pt_top5 > 682.0 and Q.D2 < 1.55:
        z += -0.00733 * (Q.sum_pt_top5 - 682.0) * (1.55 - Q.D2)
    if Q.z_6 < 0.05 and Q.e2_sq > 0.00225:
        z += -9060.0 * (0.05 - Q.z_6) * (Q.e2_sq - 0.00225)
    if Q.z_6 < 0.0283 and Q.n_dr_0p2_0p4 < 1.93:
        z += 73.5 * (0.0283 - Q.z_6) * (1.93 - Q.n_dr_0p2_0p4)
    if Q.z_7 < 0.0537 and Q.centroid_offset < 0.0119:
        z += -5810.0 * (0.0537 - Q.z_7) * (0.0119 - Q.centroid_offset)
    if Q.z_7 < 0.0707 and Q.lam2 < 0.00124:
        z += 37700.0 * (0.0707 - Q.z_7) * (0.00124 - Q.lam2)
    if Q.z_7 < 0.0323 and Q.pt_5 < 30.9:
        z += -11.4 * (0.0323 - Q.z_7) * (30.9 - Q.pt_5)
    if Q.z_7 < 0.0751 and Q.sum_pt < 818.0:
        z += -0.215 * (0.0751 - Q.z_7) * (818.0 - Q.sum_pt)
    return max(0.0, z)


def neuron_6(Q):
    z = 8.53
    if Q.D2 < 1.66:
        z += -1.16 * Q.D2 + 1.9255999999999998
    if 0.0186 <= Q.centroid_offset < 0.0499:
        z += -205.0 * Q.centroid_offset + 3.8129999999999997
    if Q.centroid_offset >= 0.0499:
        z += -93.0 * Q.centroid_offset - 1.7758000000000003
    if Q.e2 < 0.0501:
        z += -132.0 * Q.e2 + 6.6132
    if Q.girth >= 0.0885:
        z += 77.0 * Q.girth - 6.8145
    if Q.girth2 < 0.00869:
        z += 1030.0 * Q.girth2 - 8.9507
    if Q.girth2_top2 < 0.00947:
        z += -147.0 * Q.girth2_top2 + 1.3920899999999998
    if Q.lam2 >= 0.00354:
        z += -2560.0 * Q.lam2 + 9.0624
    if Q.log_sum_pt < 6.56:
        z += -1.0300000000000002 * Q.log_sum_pt + 6.153600000000001
    if 6.56 <= Q.log_sum_pt < 6.69:
        z += 4.64 * Q.log_sum_pt - 31.0416
    if Q.mass_over_sum_pt >= 0.00657:
        z += -79.3 * Q.mass_over_sum_pt + 0.521001
    if Q.max_dr < 0.172:
        z += 12.8 * Q.max_dr - 2.2016
    if Q.max_dr >= 0.172:
        z += 12.2 * Q.max_dr - 2.0984
    if Q.pt1_dr01 < 5.76:
        z += 0.109 * Q.pt1_dr01 - 0.62784
    if Q.pt_4 < 31.1:
        z += -0.261 * Q.pt_4 + 8.1171
    if Q.pt_6 < 24.0:
        z += -0.226 * Q.pt_6 + 5.424
    if Q.sum_pt >= 987.0:
        z += 0.048 * Q.sum_pt - 47.376
    if Q.sum_pt_top5 >= 837.0:
        z += -0.0277 * Q.sum_pt_top5 + 23.1849
    if Q.width < 0.0132:
        z += 664.0 * Q.width - 8.7648
    if Q.z_4 < 0.0376:
        z += 185.0 * Q.z_4 - 6.956
    if Q.z_dr_0_0p05 >= 0.755:
        z += -3.02 * Q.z_dr_0_0p05 + 2.2801
    if Q.C2 > 0.00342 and Q.pt_7 > 28.9:
        z += 1.12 * (Q.C2 - 0.00342) * (Q.pt_7 - 28.9)
    if Q.D2 < 1.66 and Q.pt_4 < 89.2:
        z += -0.0156 * (1.66 - Q.D2) * (89.2 - Q.pt_4)
    if Q.LHA > 0.313 and Q.eccentricity > 0.878:
        z += 418.0 * (Q.LHA - 0.313) * (Q.eccentricity - 0.878)
    if Q.centroid_offset > 0.00824 and Q.lam2 < 0.00351:
        z += 45800.0 * (Q.centroid_offset - 0.00824) * (0.00351 - Q.lam2)
    if Q.centroid_offset > 0.0188 and Q.mean_phi2 < 0.00883:
        z += 9700.0 * (Q.centroid_offset - 0.0188) * (0.00883 - Q.mean_phi2)
    if Q.centroid_offset > 0.00324 and Q.planar_flow > 0.0184:
        z += 85.1 * (Q.centroid_offset - 0.00324) * (Q.planar_flow - 0.0184)
    if Q.e2 < 0.0505 and Q.z_dr_0p1_0p2 > 0.156:
        z += 410.0 * (0.0505 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.156)
    if Q.girth2_top5 > 0.0111 and Q.mean_eta > 0.0139:
        z += -9630.0 * (Q.girth2_top5 - 0.0111) * (Q.mean_eta - 0.0139)
    if Q.girth2_top5 > 0.00208 and Q.z_dr_0p05_0p1 > 0.275:
        z += 187.0 * (Q.girth2_top5 - 0.00208) * (Q.z_dr_0p05_0p1 - 0.275)
    if Q.lam2 < 0.000717 and Q.z_dr_0p2_0p4 < 0.199:
        z += -6250.0 * (0.000717 - Q.lam2) * (0.199 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt < 6.68 and Q.mean_eta2 > 0.00426:
        z += 417.0 * (6.68 - Q.log_sum_pt) * (Q.mean_eta2 - 0.00426)
    if Q.log_sum_pt < 6.71 and Q.pt_6 < 36.7:
        z += 0.27 * (6.71 - Q.log_sum_pt) * (36.7 - Q.pt_6)
    if Q.log_sum_pt < 6.7 and Q.pt_7 < 46.5:
        z += 0.242 * (6.7 - Q.log_sum_pt) * (46.5 - Q.pt_7)
    if Q.log_sum_pt < 6.69 and Q.z_7 < 0.0493:
        z += 296.0 * (6.69 - Q.log_sum_pt) * (0.0493 - Q.z_7)
    if Q.mass < 49.8 and Q.z_dr_0p05_0p1 < 0.75:
        z += 0.062 * (49.8 - Q.mass) * (0.75 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt > 0.00635 and Q.tau32 < 0.524:
        z += -24.6 * (Q.mass_over_sum_pt - 0.00635) * (0.524 - Q.tau32)
    if Q.sum_pt > 989.0 and Q.pt_6 > 29.9:
        z += -0.000337 * (Q.sum_pt - 989.0) * (Q.pt_6 - 29.9)
    return max(0.0, z)


def neuron_7(Q):
    z = 11.3
    if Q.C2 < 0.0673:
        z += 28.2 * Q.C2 - 1.8978599999999999
    if Q.centroid_offset < 0.021:
        z += 53.5 * Q.centroid_offset - 1.1235000000000002
    if Q.centroid_offset >= 0.0312:
        z += 76.8 * Q.centroid_offset - 2.3961599999999996
    if Q.e2 < 0.0174:
        z += -132.0 * Q.e2 + 3.234
    if 0.0174 <= Q.e2 < 0.0245:
        z += -212.4 * Q.e2 + 4.63296
    if Q.e2 >= 0.0245:
        z += -80.4 * Q.e2 + 1.39896
    if Q.girth < 0.0883:
        z += 63.8 * Q.girth - 5.63354
    if 0.00153 <= Q.girth2 < 0.00751:
        z += -680.0 * Q.girth2 + 1.0404
    if Q.girth2 >= 0.00751:
        z += -1710.0 * Q.girth2 + 8.7757
    if Q.lam1 < 0.00846:
        z += 707.0 * Q.lam1 - 5.98122
    if Q.mass < 29.5:
        z += -0.136 * Q.mass + 4.0120000000000005
    if Q.mass >= 80.4:
        z += -0.316 * Q.mass + 25.4064
    if 0.0726 <= Q.mass_over_sum_pt < 0.0842:
        z += 87.7 * Q.mass_over_sum_pt - 6.36702
    if 0.0842 <= Q.mass_over_sum_pt < 0.108:
        z += 196.7 * Q.mass_over_sum_pt - 15.54482
    if Q.mass_over_sum_pt >= 0.108:
        z += -379.3 * Q.mass_over_sum_pt + 46.66318
    if Q.mass_top5 >= 53.5:
        z += 0.0311 * Q.mass_top5 - 1.66385
    if Q.max_dr < 0.158:
        z += 32.8 * Q.max_dr - 5.1823999999999995
    if Q.width < 0.000546:
        z += 4880.0 * Q.width - 7.998940000000001
    if 0.000546 <= Q.width < 0.00544:
        z += 1090.0 * Q.width - 5.929600000000001
    if Q.centroid_offset < 0.0227 and Q.C2 > 0.0275:
        z += 1290.0 * (0.0227 - Q.centroid_offset) * (Q.C2 - 0.0275)
    if Q.centroid_offset > 0.031 and Q.pt_2 > 57.7:
        z += -1.44 * (Q.centroid_offset - 0.031) * (Q.pt_2 - 57.7)
    if Q.centroid_offset < 0.0397 and Q.sum_pt > 567.0:
        z += 0.229 * (0.0397 - Q.centroid_offset) * (Q.sum_pt - 567.0)
    if Q.e2 < 0.0388 and Q.D2 < 1.13:
        z += 287.0 * (0.0388 - Q.e2) * (1.13 - Q.D2)
    if Q.girth2 > 0.00396 and Q.eccentricity > 0.951:
        z += 6120.0 * (Q.girth2 - 0.00396) * (Q.eccentricity - 0.951)
    if Q.girth2 > 0.00793 and Q.log_sum_pt > 6.18:
        z += -1430.0 * (Q.girth2 - 0.00793) * (Q.log_sum_pt - 6.18)
    if Q.girth2_top2 < 0.00101 and Q.log_sum_pt > 6.4:
        z += -3890.0 * (0.00101 - Q.girth2_top2) * (Q.log_sum_pt - 6.4)
    if Q.lam1 < 0.0081 and Q.D2 < 1.14:
        z += -1160.0 * (0.0081 - Q.lam1) * (1.14 - Q.D2)
    if Q.mass_over_sum_pt < 0.131 and Q.z_dr_0p05_0p1 > 0.277:
        z += -36.3 * (0.131 - Q.mass_over_sum_pt) * (Q.z_dr_0p05_0p1 - 0.277)
    if Q.max_dr < 0.157 and Q.z_dr_0p05_0p1 < 0.671:
        z += 46.1 * (0.157 - Q.max_dr) * (0.671 - Q.z_dr_0p05_0p1)
    if Q.max_dr < 0.196 and Q.z_dr_0p05_0p1 > 0.0498:
        z += 40.1 * (0.196 - Q.max_dr) * (Q.z_dr_0p05_0p1 - 0.0498)
    if Q.planar_flow < 0.191 and Q.sum_pt > 621.0:
        z += 0.0123 * (0.191 - Q.planar_flow) * (Q.sum_pt - 621.0)
    if Q.pt_7 < 49.2 and Q.planar_flow < 0.686:
        z += -0.0907 * (49.2 - Q.pt_7) * (0.686 - Q.planar_flow)
    if Q.width < 0.00581 and Q.n_dr_0p1_0p2 < 2.77:
        z += 87.7 * (0.00581 - Q.width) * (2.77 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.26
    if Q.C2 < 0.0264:
        z += 54.3 * Q.C2 - 1.43352
    if Q.width < 0.00509:
        z += -2730.0 * Q.width + 13.8957
    if Q.C2 < 0.0264 and Q.centroid_offset > 0.0157:
        z += -6340.0 * (0.0264 - Q.C2) * (Q.centroid_offset - 0.0157)
    if Q.C2 < 0.026 and Q.width < 0.00453:
        z += -16000.0 * (0.026 - Q.C2) * (0.00453 - Q.width)
    if Q.LHA < 0.196 and Q.girth2 > 0.00678:
        z += 497000.0 * (0.196 - Q.LHA) * (Q.girth2 - 0.00678)
    if Q.LHA < 0.196 and Q.lam2 < 0.000294:
        z += -125000.0 * (0.196 - Q.LHA) * (0.000294 - Q.lam2)
    if Q.LHA < 0.202 and Q.width < 0.000479:
        z += 53000.0 * (0.202 - Q.LHA) * (0.000479 - Q.width)
    if Q.girth < 0.0632 and Q.centroid_offset > 0.00688:
        z += -3320.0 * (0.0632 - Q.girth) * (Q.centroid_offset - 0.00688)
    if Q.girth < 0.0613 and Q.lam1 > 0.000235:
        z += -33700.0 * (0.0613 - Q.girth) * (Q.lam1 - 0.000235)
    if Q.girth < 0.0636 and Q.lam2 < 0.000187:
        z += 372000.0 * (0.0636 - Q.girth) * (0.000187 - Q.lam2)
    if Q.girth < 0.0634 and Q.log_sum_pt > 6.67:
        z += 498.0 * (0.0634 - Q.girth) * (Q.log_sum_pt - 6.67)
    if Q.girth < 0.0604 and Q.width < 0.00498:
        z += -61400.0 * (0.0604 - Q.girth) * (0.00498 - Q.width)
    if Q.girth2 < 0.00673 and Q.centroid_offset < 0.0236:
        z += 29300.0 * (0.00673 - Q.girth2) * (0.0236 - Q.centroid_offset)
    if Q.girth2 < 0.0068 and Q.centroid_offset > 0.0184:
        z += -39800.0 * (0.0068 - Q.girth2) * (Q.centroid_offset - 0.0184)
    if Q.log_sum_pt > 6.66 and Q.pt_7 < 50.2:
        z += 0.108 * (Q.log_sum_pt - 6.66) * (50.2 - Q.pt_7)
    if Q.log_sum_pt > 6.69 and Q.width < 0.00721:
        z += -4010.0 * (Q.log_sum_pt - 6.69) * (0.00721 - Q.width)
    if Q.mass < 21.6 and Q.centroid_offset > 0.0322:
        z += -101.0 * (21.6 - Q.mass) * (Q.centroid_offset - 0.0322)
    if Q.mass < 24.7 and Q.lam2 < 0.000203:
        z += -694.0 * (24.7 - Q.mass) * (0.000203 - Q.lam2)
    if Q.max_dr < 0.183 and Q.lam2 < 0.000197:
        z += 84200.0 * (0.183 - Q.max_dr) * (0.000197 - Q.lam2)
    if Q.sum_pt_top5 > 655.0 and Q.girth2 < 0.0017:
        z += -3.96 * (Q.sum_pt_top5 - 655.0) * (0.0017 - Q.girth2)
    if Q.width < 0.00515 and Q.mass_over_sum_pt_sq > 1.24e-05:
        z += -371000.0 * (0.00515 - Q.width) * (Q.mass_over_sum_pt_sq - 1.24e-05)
    return max(0.0, z)


def neuron_9(Q):
    z = -2.15
    if Q.centroid_offset < 0.0194:
        z += -181.0 * Q.centroid_offset + 3.5114
    if Q.e2 < 0.0202:
        z += -227.0 * Q.e2 + 4.5854
    if Q.girth < 0.0535:
        z += 90.8 * Q.girth - 4.8578
    if Q.girth2 < 0.00415:
        z += -1000.0 * Q.girth2 + 4.15
    if Q.girth2 >= 0.0165:
        z += 138.0 * Q.girth2 - 2.277
    if Q.lam2 >= 0.00114:
        z += 1060.0 * Q.lam2 - 1.2084
    if Q.log_sum_pt >= 6.34:
        z += -2.25 * Q.log_sum_pt + 14.265
    if Q.mass < 27.9:
        z += 0.116 * Q.mass - 3.2364
    if Q.mass_over_sum_pt < 0.0767:
        z += 64.6 * Q.mass_over_sum_pt - 4.95482
    if Q.max_dr < 0.211:
        z += -14.0 * Q.max_dr + 2.9539999999999997
    if Q.pt_5 < 35.8:
        z += -0.073 * Q.pt_5 + 2.6133999999999995
    if Q.sum_pt >= 853.0:
        z += -0.00509 * Q.sum_pt + 4.34177
    if Q.width < 0.000176:
        z += -13240.0 * Q.width + 15.3072
    if 0.000176 <= Q.width < 0.00624:
        z += -2140.0 * Q.width + 13.3536
    if Q.centroid_offset < 0.0179 and Q.pt_4 > 47.2:
        z += 2.33 * (0.0179 - Q.centroid_offset) * (Q.pt_4 - 47.2)
    if Q.centroid_offset < 0.0176 and Q.z_4 > 0.0442:
        z += -1900.0 * (0.0176 - Q.centroid_offset) * (Q.z_4 - 0.0442)
    if Q.e2 < 0.0322 and Q.dr01 < 0.0556:
        z += 794.0 * (0.0322 - Q.e2) * (0.0556 - Q.dr01)
    if Q.e2 < 0.0214 and Q.eccentricity > 0.904:
        z += -1110.0 * (0.0214 - Q.e2) * (Q.eccentricity - 0.904)
    if Q.e2 < 0.0194 and Q.pt_7 < 53.0:
        z += -3.77 * (0.0194 - Q.e2) * (53.0 - Q.pt_7)
    if Q.e2 < 0.0348 and Q.tau21 < 0.432:
        z += -191.0 * (0.0348 - Q.e2) * (0.432 - Q.tau21)
    if Q.girth2 < 0.00445 and Q.centroid_offset > 0.0105:
        z += -59200.0 * (0.00445 - Q.girth2) * (Q.centroid_offset - 0.0105)
    if Q.girth2 > 0.019 and Q.pt_7 < 29.1:
        z += 55.1 * (Q.girth2 - 0.019) * (29.1 - Q.pt_7)
    if Q.girth2_top2 < 0.000666 and Q.pt_7 > 32.4:
        z += 131.0 * (0.000666 - Q.girth2_top2) * (Q.pt_7 - 32.4)
    if Q.girth2_top2 < 0.000725 and Q.z_7 > 0.0232:
        z += -82700.0 * (0.000725 - Q.girth2_top2) * (Q.z_7 - 0.0232)
    if Q.lam1 < 0.00588 and Q.mean_phi2 < 0.00203:
        z += -122000.0 * (0.00588 - Q.lam1) * (0.00203 - Q.mean_phi2)
    if Q.log_sum_pt > 6.48 and Q.dr_5 < 0.0257:
        z += 100.0 * (Q.log_sum_pt - 6.48) * (0.0257 - Q.dr_5)
    if Q.mass < 39.0 and Q.centroid_offset < 0.0299:
        z += -3.62 * (39.0 - Q.mass) * (0.0299 - Q.centroid_offset)
    if Q.mass < 51.9 and Q.log_sum_pt < 6.83:
        z += 0.227 * (51.9 - Q.mass) * (6.83 - Q.log_sum_pt)
    if Q.mass < 61.8 and Q.planar_flow < 0.323:
        z += 0.144 * (61.8 - Q.mass) * (0.323 - Q.planar_flow)
    if Q.max_dr < 0.238 and Q.z_top5 < 0.903:
        z += -40.5 * (0.238 - Q.max_dr) * (0.903 - Q.z_top5)
    if Q.sum_pt > 986.0 and Q.dr_3 < 0.0548:
        z += 0.511 * (Q.sum_pt - 986.0) * (0.0548 - Q.dr_3)
    if Q.width < 0.00621 and Q.C2 > 0.0292:
        z += -9180.0 * (0.00621 - Q.width) * (Q.C2 - 0.0292)
    if Q.width < 0.00609 and Q.centroid_offset > 0.00339:
        z += -44100.0 * (0.00609 - Q.width) * (Q.centroid_offset - 0.00339)
    return max(0.0, z)


def neuron_10(Q):
    z = -1.45
    if 0.303 <= Q.LHA < 0.347:
        z += -10.3 * Q.LHA + 3.1209000000000002
    if Q.LHA >= 0.347:
        z += -37.400000000000006 * Q.LHA + 12.524600000000001
    if Q.centroid_offset < 0.0373:
        z += -23.3 * Q.centroid_offset + 0.88773
    if 0.0373 <= Q.centroid_offset < 0.0381:
        z += -6.100000000000001 * Q.centroid_offset + 0.24617
    if Q.centroid_offset >= 0.0381:
        z += 17.2 * Q.centroid_offset - 0.64156
    z += 62.2 * Q.e2
    if Q.girth2 < 0.00165:
        z += 1030.0 * Q.girth2 - 1.6995
    if Q.girth2 >= 0.00871:
        z += 201.0 * Q.girth2 - 1.7507100000000002
    if Q.girth2_top2 < 0.004:
        z += -237.0 * Q.girth2_top2 + 0.9480000000000001
    if Q.girth2_top5 < 0.00227:
        z += -464.0 * Q.girth2_top5 + 1.05328
    if Q.lam1 < 0.00422:
        z += 653.0 * Q.lam1 - 2.7556599999999998
    if Q.lam2 < 0.000205:
        z += 512.0 * Q.lam2 - 1.80224
    if 0.000205 <= Q.lam2 < 0.00352:
        z += 1267.0 * Q.lam2 - 1.9570150000000002
    if Q.lam2 >= 0.00352:
        z += 755.0 * Q.lam2 - 0.154775
    if Q.log_sum_pt >= 6.7:
        z += -11.7 * Q.log_sum_pt + 78.39
    if 15.3 <= Q.mass < 53.2:
        z += 0.0576 * Q.mass - 0.8812800000000001
    if Q.mass >= 53.2:
        z += 0.006100000000000001 * Q.mass + 1.8585199999999997
    if Q.mass_over_sum_pt < 0.108:
        z += -20.1 * Q.mass_over_sum_pt + 2.1708000000000003
    if Q.mass_over_sum_pt_sq < 0.00396:
        z += -376.0 * Q.mass_over_sum_pt_sq + 1.48896
    if Q.n_dr_0p05_0p1 < 3.02:
        z += -0.179 * Q.n_dr_0p05_0p1 + 0.54058
    if Q.tau21 < 0.385:
        z += 3.11 * Q.tau21 - 1.19735
    if Q.eccentricity > 0.898 and Q.z_dr_0p2_0p4 < 0.0572:
        z += 155.0 * (Q.eccentricity - 0.898) * (0.0572 - Q.z_dr_0p2_0p4)
    if Q.lam1 < 0.00426 and Q.sum_pt > 994.0:
        z += -7.66 * (0.00426 - Q.lam1) * (Q.sum_pt - 994.0)
    if Q.lam2 > 0.000227 and Q.D2 < 2.0:
        z += -192.0 * (Q.lam2 - 0.000227) * (2.0 - Q.D2)
    if Q.lam2 > 0.000209 and Q.n_pt_above_50 < 8.04:
        z += -61.2 * (Q.lam2 - 0.000209) * (8.04 - Q.n_pt_above_50)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.39
    if Q.C2 < 0.0328:
        z += 18.8 * Q.C2 - 0.6166400000000001
    if Q.LHA < 0.187:
        z += 24.3 * Q.LHA - 4.5441
    if Q.centroid_offset < 0.0144:
        z += -124.0 * Q.centroid_offset + 6.1876
    if 0.0144 <= Q.centroid_offset < 0.0499:
        z += -221.1 * Q.centroid_offset + 7.585839999999999
    if Q.centroid_offset >= 0.0499:
        z += -97.1 * Q.centroid_offset + 1.39824
    if Q.e2_sq < 0.00575:
        z += 227.0 * Q.e2_sq - 1.30525
    if Q.girth < 0.0754:
        z += 103.0 * Q.girth - 9.053700000000001
    if 0.0754 <= Q.girth < 0.0879:
        z += -86.0 * Q.girth + 5.196899999999998
    if Q.girth >= 0.0879:
        z += -189.0 * Q.girth + 14.250599999999999
    if Q.log_sum_pt < 6.69:
        z += -2.54 * Q.log_sum_pt + 16.9926
    if Q.mass < 22.3:
        z += -0.0474 * Q.mass + 1.05702
    if Q.planar_flow < 0.268:
        z += -10.2 * Q.planar_flow + 2.7336
    if Q.width < 0.00368:
        z += -904.0 * Q.width + 10.16552
    if 0.00368 <= Q.width < 0.0086:
        z += -1390.0 * Q.width + 11.954
    if Q.LHA < 0.17 and Q.z_7 < 0.0283:
        z += 831.0 * (0.17 - Q.LHA) * (0.0283 - Q.z_7)
    if Q.centroid_offset < 0.0494 and Q.log_sum_pt < 6.8:
        z += -166.0 * (0.0494 - Q.centroid_offset) * (6.8 - Q.log_sum_pt)
    if Q.centroid_offset < 0.048 and Q.mean_phi2 < 0.00154:
        z += -9290.0 * (0.048 - Q.centroid_offset) * (0.00154 - Q.mean_phi2)
    if Q.centroid_offset < 0.0507 and Q.pt_7 < 48.5:
        z += -1.05 * (0.0507 - Q.centroid_offset) * (48.5 - Q.pt_7)
    if Q.girth > 0.0761 and Q.n_pt_above_50 < 7.11:
        z += 58.0 * (Q.girth - 0.0761) * (7.11 - Q.n_pt_above_50)
    if Q.girth > 0.0994 and Q.n_pt_above_50 < 6.38:
        z += -76.7 * (Q.girth - 0.0994) * (6.38 - Q.n_pt_above_50)
    if Q.girth > 0.0763 and Q.pt_7 < 40.3:
        z += -8.16 * (Q.girth - 0.0763) * (40.3 - Q.pt_7)
    if Q.girth > 0.076 and Q.z_dr_0p1_0p2 < 0.197:
        z += -978.0 * (Q.girth - 0.076) * (0.197 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt < 6.71 and Q.dr_0 < 0.119:
        z += -19.7 * (6.71 - Q.log_sum_pt) * (0.119 - Q.dr_0)
    if Q.log_sum_pt < 6.72 and Q.z_dr_0p2_0p4 < 0.055:
        z += 48.6 * (6.72 - Q.log_sum_pt) * (0.055 - Q.z_dr_0p2_0p4)
    if Q.planar_flow < 0.26 and Q.mass < 70.6:
        z += -0.137 * (0.26 - Q.planar_flow) * (70.6 - Q.mass)
    if Q.planar_flow < 0.235 and Q.width > 0.00609:
        z += -3660.0 * (0.235 - Q.planar_flow) * (Q.width - 0.00609)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.931
    if Q.e2 >= 0.0634:
        z += -82.9 * Q.e2 + 5.25586
    if Q.girth2 >= 0.0188:
        z += 259.0 * Q.girth2 - 4.8692
    if Q.girth2 > 0.0188 and Q.pt_7 > 15.6:
        z += 4.81 * (Q.girth2 - 0.0188) * (Q.pt_7 - 15.6)
    return max(0.0, z)


def neuron_13(Q):
    z = 0.629
    if Q.C2 >= 0.0669:
        z += -57.9 * Q.C2 + 3.87351
    if Q.centroid_offset < 0.0378:
        z += -60.9 * Q.centroid_offset + 2.30202
    if Q.e2 < 0.0501:
        z += 121.0 * Q.e2 - 6.0621
    if Q.girth < 0.149:
        z += -59.3 * Q.girth + 8.8357
    if Q.lam1 < 0.016:
        z += -256.0 * Q.lam1 + 4.096
    if Q.lam2 < 0.0003:
        z += -3140.0 * Q.lam2 + 0.942
    if Q.mass >= 53.5:
        z += -0.0504 * Q.mass + 2.6964
    if Q.pt_6 < 31.3:
        z += 0.117 * Q.pt_6 - 3.6621
    if Q.sum_pt >= 988.0:
        z += 0.19 * Q.sum_pt - 187.72
    if Q.sum_pt_top5 >= 903.0:
        z += 0.259 * Q.sum_pt_top5 - 233.877
    if Q.width < 0.00801:
        z += -34.0 * Q.width + 1.1183100000000001
    if 0.00801 <= Q.width < 0.0132:
        z += -163.0 * Q.width + 2.1516
    if Q.z_7 < 0.0289:
        z += 269.0 * Q.z_7 - 7.7741
    if Q.z_top5_slots >= 0.931:
        z += -602.0 * Q.z_top5_slots + 560.462
    if Q.e2 < 0.0493 and Q.pt_dispersion > 0.397:
        z += 131.0 * (0.0493 - Q.e2) * (Q.pt_dispersion - 0.397)
    if Q.girth < 0.147 and Q.log_sum_pt < 6.8:
        z += -48.2 * (0.147 - Q.girth) * (6.8 - Q.log_sum_pt)
    if Q.girth > 0.102 and Q.pt_4 > 81.4:
        z += -25.0 * (Q.girth - 0.102) * (Q.pt_4 - 81.4)
    if Q.girth < 0.152 and Q.pt_7 < 38.1:
        z += -0.966 * (0.152 - Q.girth) * (38.1 - Q.pt_7)
    if Q.lam1 < 0.0168 and Q.centroid_offset < 0.0374:
        z += -3430.0 * (0.0168 - Q.lam1) * (0.0374 - Q.centroid_offset)
    if Q.lam1 < 0.0172 and Q.pt_6 < 56.9:
        z += -1.69 * (0.0172 - Q.lam1) * (56.9 - Q.pt_6)
    if Q.lam2 < 0.00031 and Q.centroid_offset < 0.0495:
        z += -148000.0 * (0.00031 - Q.lam2) * (0.0495 - Q.centroid_offset)
    if Q.sum_pt > 989.0 and Q.D2 < 3.94:
        z += 0.0294 * (Q.sum_pt - 989.0) * (3.94 - Q.D2)
    if Q.sum_pt > 988.0 and Q.n_pt_above_50 > 6.0:
        z += -0.126 * (Q.sum_pt - 988.0) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt_top5 > 902.0 and Q.D2 < 3.86:
        z += 0.0915 * (Q.sum_pt_top5 - 902.0) * (3.86 - Q.D2)
    if Q.sum_pt_top5 > 840.0 and Q.n_pt_above_50 > 2.02:
        z += 0.0109 * (Q.sum_pt_top5 - 840.0) * (Q.n_pt_above_50 - 2.02)
    if Q.sum_pt_top5 > 902.0 and Q.n_pt_above_50 > 6.0:
        z += -0.256 * (Q.sum_pt_top5 - 902.0) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt_top5 > 657.0 and Q.pt_7 < 42.1:
        z += 0.000632 * (Q.sum_pt_top5 - 657.0) * (42.1 - Q.pt_7)
    if Q.sum_pt_top5 > 658.0 and Q.tau32 < 0.362:
        z += -0.141 * (Q.sum_pt_top5 - 658.0) * (0.362 - Q.tau32)
    if Q.sum_pt_top5 > 660.0 and Q.z_7 > 0.0221:
        z += -0.805 * (Q.sum_pt_top5 - 660.0) * (Q.z_7 - 0.0221)
    if Q.tau21 < 0.515 and Q.max_dr > 0.015:
        z += -16.5 * (0.515 - Q.tau21) * (Q.max_dr - 0.015)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.305
    if Q.C2 < 0.0669:
        z += 47.2 * Q.C2 - 3.15768
    if Q.LHA < 0.302:
        z += 13.6 * Q.LHA - 4.1072
    if 0.0123 <= Q.centroid_offset < 0.0497:
        z += -37.9 * Q.centroid_offset + 0.46617
    if Q.centroid_offset >= 0.0497:
        z += -268.9 * Q.centroid_offset + 11.94687
    if Q.e2 < 0.0443:
        z += -214.0 * Q.e2 + 9.4802
    if Q.e2_sq < 0.00575:
        z += -385.0 * Q.e2_sq + 2.21375
    if Q.girth < 0.0337:
        z += 150.8 * Q.girth - 10.913459999999999
    if 0.0337 <= Q.girth < 0.0872:
        z += 109.0 * Q.girth - 9.5048
    if Q.girth2 < 0.0133:
        z += -463.0 * Q.girth2 + 6.1579
    if 0.00428 <= Q.lam1 < 0.006:
        z += 649.0 * Q.lam1 - 2.77772
    if Q.lam1 >= 0.006:
        z += 25.0 * Q.lam1 + 0.9662800000000002
    if Q.mass < 69.5:
        z += 0.021500000000000002 * Q.mass - 1.08358
    if 69.5 <= Q.mass < 86.4:
        z += -0.0243 * Q.mass + 2.09952
    if Q.max_dr < 0.0804:
        z += 73.2 * Q.max_dr - 7.02516
    if 0.0804 <= Q.max_dr < 0.177:
        z += 11.8 * Q.max_dr - 2.0886
    if Q.n_dr_0p05_0p1 < 5.09:
        z += 0.151 * Q.n_dr_0p05_0p1 - 0.76859
    if Q.tau21 < 0.134:
        z += -10.5 * Q.tau21 + 1.407
    if Q.width < 0.00609:
        z += 1712.0 * Q.width - 11.33964
    if 0.00609 <= Q.width < 0.00747:
        z += 662.0 * Q.width - 4.94514
    if Q.z_dr_0_0p05 >= 0.602:
        z += -1.33 * Q.z_dr_0_0p05 + 0.80066
    if Q.z_dr_0p05_0p1 < 0.582:
        z += -1.74 * Q.z_dr_0p05_0p1 + 1.01268
    if Q.girth2 < 0.00436 and Q.D2 < 0.884:
        z += -11800.0 * (0.00436 - Q.girth2) * (0.884 - Q.D2)
    if Q.girth2 < 0.0125 and Q.centroid_offset > 0.031:
        z += -8450.0 * (0.0125 - Q.girth2) * (Q.centroid_offset - 0.031)
    if Q.girth2 < 0.0134 and Q.eccentricity > 0.971:
        z += 3530.0 * (0.0134 - Q.girth2) * (Q.eccentricity - 0.971)
    if Q.girth2 < 0.0131 and Q.n_dr_0p1_0p2 < 2.98:
        z += -43.6 * (0.0131 - Q.girth2) * (2.98 - Q.n_dr_0p1_0p2)
    if Q.lam1 > 0.0023 and Q.D2 > 0.452:
        z += 241.0 * (Q.lam1 - 0.0023) * (Q.D2 - 0.452)
    if Q.lam1 > 0.0023 and Q.D2 > 1.63:
        z += -283.0 * (Q.lam1 - 0.0023) * (Q.D2 - 1.63)
    if Q.lam1 > 0.00418 and Q.D2 > 0.451:
        z += -708.0 * (Q.lam1 - 0.00418) * (Q.D2 - 0.451)
    if Q.lam1 > 0.00594 and Q.D2 > 1.68:
        z += -3670.0 * (Q.lam1 - 0.00594) * (Q.D2 - 1.68)
    if Q.lam1 > 0.00543 and Q.max_dr < 0.161:
        z += 8400.0 * (Q.lam1 - 0.00543) * (0.161 - Q.max_dr)
    if Q.planar_flow < 0.115 and Q.centroid_offset > 0.0185:
        z += -656.0 * (0.115 - Q.planar_flow) * (Q.centroid_offset - 0.0185)
    if Q.planar_flow < 0.119 and Q.centroid_offset > 0.00973:
        z += 441.0 * (0.119 - Q.planar_flow) * (Q.centroid_offset - 0.00973)
    if Q.planar_flow < 0.111 and Q.max_dr < 0.159:
        z += -150.0 * (0.111 - Q.planar_flow) * (0.159 - Q.max_dr)
    if Q.width < 0.0089 and Q.D2 < 1.03:
        z += -493.0 * (0.0089 - Q.width) * (1.03 - Q.D2)
    if Q.width < 0.00746 and Q.n_dr_0p1_0p2 < 3.0:
        z += 166.0 * (0.00746 - Q.width) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.z_dr_0p05_0p1 < 0.601 and Q.C2 < 0.067:
        z += -19.5 * (0.601 - Q.z_dr_0p05_0p1) * (0.067 - Q.C2)
    if Q.z_dr_0p05_0p1 < 0.588 and Q.n_dr_0p1_0p2 < 3.0:
        z += 0.275 * (0.588 - Q.z_dr_0p05_0p1) * (3.0 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_15(Q):
    z = -2.42
    if Q.D2 < 0.744:
        z += 2.29 * Q.D2 - 1.70376
    if Q.LHA >= 0.341:
        z += -47.8 * Q.LHA + 16.2998
    if Q.e2 < 0.0245:
        z += -432.6 * Q.e2 + 12.11304
    if 0.0245 <= Q.e2 < 0.0424:
        z += -84.6 * Q.e2 + 3.5870399999999996
    if Q.e2_sq < 0.00854:
        z += 565.0 * Q.e2_sq - 4.825100000000001
    if Q.girth >= 0.0325:
        z += 46.3 * Q.girth - 1.50475
    if Q.girth2_top3 < 0.00218:
        z += 571.0 * Q.girth2_top3 - 1.24478
    if Q.log_sum_pt >= 6.9:
        z += 32.3 * Q.log_sum_pt - 222.87
    if Q.mass < 36.2:
        z += 0.438 * Q.mass - 15.8556
    if Q.mass >= 80.4:
        z += -0.475 * Q.mass + 38.19
    if Q.mass_over_sum_pt < 0.0676:
        z += 78.1 * Q.mass_over_sum_pt - 5.279559999999999
    if Q.planar_flow < 0.0798:
        z += 8.91 * Q.planar_flow - 0.7110179999999999
    if Q.width < 0.0068:
        z += 523.0 * Q.width - 1.9430999999999998
    if 0.0068 <= Q.width < 0.0141:
        z += -221.0 * Q.width + 3.1161
    if Q.z_dr_0p05_0p1 >= 0.754:
        z += -6.57 * Q.z_dr_0p05_0p1 + 4.95378
    if Q.z_dr_0p1_0p2 < 0.323:
        z += -4.04 * Q.z_dr_0p1_0p2 + 1.30492
    if Q.LHA > 0.179 and Q.sum_pt_top3 > 350.0:
        z += 0.0454 * (Q.LHA - 0.179) * (Q.sum_pt_top3 - 350.0)
    if Q.LHA > 0.179 and Q.z_top5 > 0.865:
        z += -213.0 * (Q.LHA - 0.179) * (Q.z_top5 - 0.865)
    if Q.girth2_top2 < 0.00812 and Q.mean_phi < 0.00167:
        z += 6900.0 * (0.00812 - Q.girth2_top2) * (0.00167 - Q.mean_phi)
    if Q.log_sum_pt > 6.9 and Q.mean_phi > 0.0175:
        z += 49200.0 * (Q.log_sum_pt - 6.9) * (Q.mean_phi - 0.0175)
    if Q.mass > 80.4 and Q.z_dr_0p2_0p4 < 0.211:
        z += 1.94 * (Q.mass - 80.4) * (0.211 - Q.z_dr_0p2_0p4)
    if Q.tau21 < 0.243 and Q.girth2_top5 > 0.00856:
        z += -1140.0 * (0.243 - Q.tau21) * (Q.girth2_top5 - 0.00856)
    if Q.tau21 < 0.249 and Q.z_dr_0p05_0p1 < 0.605:
        z += -12.3 * (0.249 - Q.tau21) * (0.605 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.245 and Q.z_dr_0p2_0p4 < 0.221:
        z += 52.1 * (0.245 - Q.tau21) * (0.221 - Q.z_dr_0p2_0p4)
    if Q.width < 0.00761 and Q.e2 > 0.0242:
        z += -74100.0 * (0.00761 - Q.width) * (Q.e2 - 0.0242)
    if Q.width < 0.0059 and Q.log_sum_pt > 6.9:
        z += -8130.0 * (0.0059 - Q.width) * (Q.log_sum_pt - 6.9)
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
