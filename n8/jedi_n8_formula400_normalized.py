"""JEDI-linear jet tagger, 8 particles, 3 features: simplified from the formula tuned on the network (624), keeping validation agreement with the network within 0.5 point, as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron 13:  15.3%   (on for 85% of jets)
  neuron  9:  11.9%   (on for 71% of jets)
  neuron  6:   8.9%   (on for 33% of jets)
  neuron  5:   8.4%   (on for 60% of jets)
  neuron  4:   7.7%   (on for 83% of jets)
  neuron 10:   7.0%   (on for 71% of jets)
  neuron  7:   7.0%   (on for 51% of jets)
  neuron  3:   6.3%   (on for 25% of jets)
  neuron  2:   6.2%   (on for 83% of jets)
  neuron 11:   6.1%   (on for 76% of jets)
  neuron  1:   4.1%   (on for 67% of jets)
  neuron  0:   3.4%   (on for 44% of jets)
  neuron 14:   2.8%   (on for 28% of jets)
  neuron  8:   2.3%   (on for 47% of jets)
  neuron 15:   2.3%   (on for 24% of jets)
  neuron 12:   0.2%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

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
    # scale S = 22.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.481164620202165 * (-0.03767598406529453
        - 0.15298489631068016 * max(0.0, 0.0764 - Q.girth) / 0.02819080851118819   # -15.3%  girth < 0.0764
        + 0.13088324943819382 * max(0.0, 0.0133 - Q.girth2) / 0.00828847289196343   # +13.1%  girth2 < 0.0133
        - 0.12341560066829249 * max(0.0, 29.1 - Q.mass) / 7.132458702635002   # -12.3%  mass < 29.1
        + 0.10113523251705557 * max(0.0, 28.9 - Q.mass) * max(0.0, 0.109 - Q.dr_0) / 0.6648063775199852   # +10.1%  mass < 28.9 and dr_0 < 0.109
        + 0.09334928235823242 * max(0.0, 0.00876 - Q.width) / 0.00451311953521109   # +9.3%  width < 0.00876
        - 0.07594069763747612 * max(0.0, 59.6 - Q.mass) / 23.711601735570813   # -7.6%  mass < 59.6
        - 0.045919615649554 * max(0.0, 0.00435 - Q.width) / 0.0015547084920392094   # -4.6%  width < 0.00435
        - 0.0336772923066167 * max(0.0, 0.00807 - Q.girth2_top3) / 0.004673486125356284   # -3.4%  girth2_top3 < 0.00807
        + 0.03334662834150636 * max(0.0, 0.0206 - Q.e2) / 0.005594560009508236   # +3.3%  e2 < 0.0206
        + 0.031061241026399317 * max(0.0, 0.00406 - Q.girth2_top3) / 0.0018376128232164743   # +3.1%  girth2_top3 < 0.00406
        + 0.025496329639245235 * max(0.0, 0.0327 - Q.centroid_offset) / 0.01780084421834819   # +2.5%  centroid_offset < 0.0327
        - 0.020968207604314816 * max(0.0, 64.0 - Q.mass) * max(0.0, 40.1 - Q.pt_7) / 226.62967641498858   # -2.1%  mass < 64 and pt_7 < 40.1
        - 0.016926218915821748 * max(0.0, Q.sum_pt - 821.0) / 29.49776076311384   # -1.7%  sum_pt > 821
        + 0.014146744821470486 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.eccentricity - 0.955) / 0.00020002220073677391   # +1.4%  girth2 < 0.019 and eccentricity > 0.955
        + 0.013866401227074202 * max(0.0, 0.000456 - Q.lam1) / 7.548010863816669e-05   # +1.4%  lam1 < 0.000456
        + 0.01382451590451402 * max(0.0, Q.z_dr_0_0p05 - 0.851) / 0.05303604400067964   # +1.4%  z_dr_0_0p05 > 0.851
        + 0.01312087589527002 * max(0.0, 63.4 - Q.mass) * max(0.0, Q.centroid_offset - 0.0108) / 0.17453998281822947   # +1.3%  mass < 63.4 and centroid_offset > 0.0108
        + 0.012607967212983105 * max(0.0, Q.sum_pt_top5 - 698.0) / 34.10851822396928   # +1.3%  sum_pt_top5 > 698
        - 0.009294692434025136 * max(0.0, Q.sum_pt - 904.0) * max(0.0, Q.pt_7 - 29.4) / 84.25625431591395   # -0.9%  sum_pt > 904 and pt_7 > 29.4
        + 0.00911319930102953 * max(0.0, 0.00809 - Q.mass_over_sum_pt_sq) * max(0.0, 7.9 - Q.n_pt_above_50) / 0.011381962983508676   # +0.9%  mass_over_sum_pt_sq < 0.00809 and n_pt_above_50 < 7.9
        - 0.006836474061022344 * max(0.0, Q.sum_pt - 889.0) / 14.363728858652836   # -0.7%  sum_pt > 889
        - 0.005451390641530837 * max(0.0, 0.00657 - Q.lam1) * max(0.0, 0.872 - Q.D2) / 8.225074524918404e-05   # -0.5%  lam1 < 0.00657 and D2 < 0.872
        + 0.005260874851117599 * max(0.0, Q.sum_pt - 876.0) * max(0.0, 28.8 - Q.pt_7) / 121.42771414194691   # +0.5%  sum_pt > 876 and pt_7 < 28.8
        + 0.005230293577985831 * max(0.0, Q.log_sum_pt - 6.63) * max(0.0, 0.0771 - Q.dr_4) / 0.002760166453959757   # +0.5%  log_sum_pt > 6.63 and dr_4 < 0.0771
        + 0.003184007907018091 * max(0.0, 56.0 - Q.mass) * max(0.0, Q.C2 - 0.0235) / 0.07158020590969905   # +0.3%  mass < 56 and C2 > 0.0235
        - 0.0029580697515702694 * max(0.0, 29.7 - Q.mass) * max(0.0, 0.874 - Q.D2) / 0.02097818707983967   # -0.3%  mass < 29.7 and D2 < 0.874
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 11.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.78559824468047 * (0.06983099057966514
        + 0.14725946118705735 * max(0.0, 0.00817 - Q.e2_sq) / 0.004295893186333601   # +14.7%  e2_sq < 0.00817
        - 0.11095761046915129 * max(0.0, 0.00866 - Q.width) / 0.00443288752264125   # -11.1%  width < 0.00866
        + 0.08289231762733146 * max(0.0, Q.log_sum_pt - 6.38) / 0.20785862832471777   # +8.3%  log_sum_pt > 6.38
        - 0.07846757490741509 * max(0.0, 0.249 - Q.max_dr) / 0.13006853911296265   # -7.8%  max_dr < 0.249
        - 0.06980652361835493 * max(0.0, 0.00818 - Q.lam1) * max(0.0, 0.00054 - Q.lam2) / 1.922223463139555e-06   # -7.0%  lam1 < 0.00818 and lam2 < 0.00054
        + 0.05904152908917558 * max(0.0, Q.pt_7 - 34.5) / 4.489288655462185   # +5.9%  pt_7 > 34.5
        - 0.051019987242530276 * max(0.0, 0.0555 - Q.z_7) / 0.010512256505055677   # -5.1%  z_7 < 0.0555
        - 0.04412905347944948 * max(0.0, 55.5 - Q.mass) / 20.803491809072415   # -4.4%  mass < 55.5
        + 0.04077954009961546 * max(0.0, Q.log_sum_pt - 6.38) * max(0.0, 0.2 - Q.max_dr) / 0.021455860545397536   # +4.1%  log_sum_pt > 6.38 and max_dr < 0.2
        + 0.03600470925133804 * max(0.0, Q.log_sum_pt - 6.63) / 0.060446871531737945   # +3.6%  log_sum_pt > 6.63
        + 0.034507092710420906 * max(0.0, Q.log_sum_pt - 6.57) * max(0.0, 0.00118 - Q.lam2) / 9.871037166916575e-05   # +3.5%  log_sum_pt > 6.57 and lam2 < 0.00118
        - 0.03424458017227461 * max(0.0, Q.pt_7 - 34.8) * max(0.0, 91.2 - Q.mass) / 231.94992182079267   # -3.4%  pt_7 > 34.8 and mass < 91.2
        + 0.03419187690423007 * max(0.0, 0.0559 - Q.z_7) * max(0.0, 0.0139 - Q.girth2_top2) / 0.00012792753156343637   # +3.4%  z_7 < 0.0559 and girth2_top2 < 0.0139
        - 0.02899356539852779 * max(0.0, 0.0085 - Q.e2_sq) * max(0.0, 0.0833 - Q.planar_flow) / 5.891491611515819e-05   # -2.9%  e2_sq < 0.0085 and planar_flow < 0.0833
        - 0.026751423471452235 * max(0.0, 0.043 - Q.z_7) / 0.004910927250901111   # -2.7%  z_7 < 0.043
        + 0.02661928168836029 * max(0.0, Q.LHA - 0.259) / 0.03556963260101973   # +2.7%  LHA > 0.259
        - 0.025317655608115854 * max(0.0, Q.log_sum_pt - 6.37) * max(0.0, 0.0246 - Q.centroid_offset) / 0.0031810630862946147   # -2.5%  log_sum_pt > 6.37 and centroid_offset < 0.0246
        - 0.021235491055544342 * max(0.0, Q.log_sum_pt - 6.64) * max(0.0, 0.00799 - Q.girth2_top3) / 0.0003685905244611947   # -2.1%  log_sum_pt > 6.64 and girth2_top3 < 0.00799
        + 0.01798308161290184 * max(0.0, 0.0346 - Q.e2) * max(0.0, Q.eccentricity - 0.979) / 3.0451346995827807e-05   # +1.8%  e2 < 0.0346 and eccentricity > 0.979
        - 0.011919869817942495 * max(0.0, Q.e2 - 0.0311) / 0.008514108897161381   # -1.2%  e2 > 0.0311
        + 0.008068294178863823 * max(0.0, 0.00818 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0213) / 6.696455895210079e-06   # +0.8%  lam1 < 0.00818 and centroid_offset > 0.0213
        + 0.006454054645469909 * max(0.0, Q.pt_7 - 35.0) * max(0.0, 0.0787 - Q.max_dr) / 0.08294972202914067   # +0.6%  pt_7 > 35 and max_dr < 0.0787
        - 0.0033554257644771105 * max(0.0, Q.pt_7 - 52.6) / 0.3766257142854958   # -0.3%  pt_7 > 52.6
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 7.742;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.742074356748286 * (0.2854532129459181
        - 0.1932227852663134 * max(0.0, Q.LHA - 0.119) / 0.12896079059912116   # -19.3%  LHA > 0.119
        - 0.15086689783421406 * max(0.0, 53.4 - Q.pt_7) / 19.085338905301178   # -15.1%  pt_7 < 53.4
        - 0.11451013142808072 * max(0.0, 36.4 - Q.mass) / 10.190183357669193   # -11.5%  mass < 36.4
        + 0.09485365453267412 * max(0.0, 69.4 - Q.mass) / 31.51777023181447   # +9.5%  mass < 69.4
        + 0.08412723672018124 * max(0.0, 43.4 - Q.pt_7) / 10.20876680431674   # +8.4%  pt_7 < 43.4
        + 0.06891223280846012 * max(0.0, 781.0 - Q.sum_pt) / 107.34881901260505   # +6.9%  sum_pt < 781
        + 0.05656407813634569 * max(0.0, Q.pt_7 - 30.4) / 6.8640015494123565   # +5.7%  pt_7 > 30.4
        + 0.053757539717087355 * max(0.0, 36.8 - Q.mass) * max(0.0, 0.00114 - Q.lam2) / 0.011218190558639879   # +5.4%  mass < 36.8 and lam2 < 0.00114
        + 0.04252096899964249 * max(0.0, 0.00583 - Q.lam1) / 0.002438522249749785   # +4.3%  lam1 < 0.00583
        + 0.039172515878425414 * max(0.0, 6.49 - Q.log_sum_pt) / 0.07960013928390353   # +3.9%  log_sum_pt < 6.49
        + 0.03749836060516657 * max(0.0, 0.00333 - Q.lam1) / 0.0011123183757140239   # +3.7%  lam1 < 0.00333
        - 0.02920296167942279 * max(0.0, Q.pt_7 - 30.6) * max(0.0, 0.0507 - Q.C2) / 0.20553772796305636   # -2.9%  pt_7 > 30.6 and C2 < 0.0507
        - 0.0171891650753276 * max(0.0, Q.pt_7 - 30.5) * max(0.0, Q.max_dr - 0.0929) / 0.24828319802911758   # -1.7%  pt_7 > 30.5 and max_dr > 0.0929
        - 0.01322641185710146 * max(0.0, 0.00595 - Q.lam1) * max(0.0, Q.max_dr - 0.0801) / 4.5110072277822325e-05   # -1.3%  lam1 < 0.00595 and max_dr > 0.0801
        - 0.0043750594615568645 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # -0.4%  log_sum_pt > 6.9
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 22.18;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.178705423030483 * (0.1325595855990354
        - 0.2565915301185898 * max(0.0, 0.00872 - Q.girth2) / 0.0044809983941298094   # -25.7%  girth2 < 0.00872
        - 0.18557252606048444 * max(0.0, 0.0126 - Q.girth2) / 0.007693006336641367   # -18.6%  girth2 < 0.0126
        + 0.1813758957550415 * max(0.0, 0.0445 - Q.e2) / 0.020214485240147493   # +18.1%  e2 < 0.0445
        - 0.08753290746309594 * max(0.0, Q.lam1 - 0.00353) / 0.003588477947218838   # -8.8%  lam1 > 0.00353
        + 0.06733412196347476 * max(0.0, Q.mass_over_sum_pt - 0.0674) * max(0.0, 4.63 - Q.n_dr_0_0p05) / 0.06301196860532968   # +6.7%  mass_over_sum_pt > 0.0674 and n_dr_0_0p05 < 4.63
        + 0.05898568816143776 * max(0.0, 0.00389 - Q.girth2) / 0.0013335639163172932   # +5.9%  girth2 < 0.00389
        + 0.0375411802365982 * max(0.0, 0.458 - Q.z_dr_0p1_0p2) / 0.3426398262141585   # +3.8%  z_dr_0p1_0p2 < 0.458
        - 0.025390272694607353 * max(0.0, Q.e2 - 0.0269) / 0.010389730234393189   # -2.5%  e2 > 0.0269
        + 0.020901325337340963 * max(0.0, Q.centroid_offset - 0.0146) / 0.007013076211918434   # +2.1%  centroid_offset > 0.0146
        + 0.015652360027609072 * max(0.0, Q.mass - 36.0) * max(0.0, Q.eccentricity - 0.688) / 3.5065563861369577   # +1.6%  mass > 36 and eccentricity > 0.688
        + 0.014373016439501576 * max(0.0, Q.width - 0.0182) / 0.0008237077458710074   # +1.4%  width > 0.0182
        + 0.008746709785015707 * max(0.0, Q.max_dr - 0.14) / 0.028155399091814463   # +0.9%  max_dr > 0.14
        - 0.008664939570946469 * max(0.0, Q.mass - 69.5) / 2.423419196120831   # -0.9%  mass > 69.5
        - 0.007048605107584543 * max(0.0, Q.C2 - 0.0926) / 0.0009474480989356707   # -0.7%  C2 > 0.0926
        - 0.005825942116947503 * max(0.0, Q.max_dr - 0.141) * max(0.0, 0.0468 - Q.dr_1) / 0.0001919938395592946   # -0.6%  max_dr > 0.141 and dr_1 < 0.0468
        + 0.0047287069562963364 * max(0.0, Q.centroid_offset - 0.0105) * max(0.0, Q.pt_7 - 25.2) / 0.09041086087545822   # +0.5%  centroid_offset > 0.0105 and pt_7 > 25.2
        + 0.003998904177039711 * max(0.0, 0.00803 - Q.girth2) * max(0.0, Q.pt1_dr01 - 4.22) / 0.004049795331392232   # +0.4%  girth2 < 0.00803 and pt1_dr01 > 4.22
        - 0.0037904562021416795 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.147 - Q.max_dr) / 4.776557473079499e-05   # -0.4%  LHA > 0.312 and max_dr < 0.147
        - 0.002283045710690794 * max(0.0, Q.mass_over_sum_pt - 0.0689) * max(0.0, 0.0428 - Q.dr_7) / 1.0291666318033415e-05   # -0.2%  mass_over_sum_pt > 0.0689 and dr_7 < 0.0428
        + 0.001875660770880012 * max(0.0, Q.mass_over_sum_pt - 0.0904) * max(0.0, 0.149 - Q.max_dr) / 4.118784921869511e-06   # +0.2%  mass_over_sum_pt > 0.0904 and max_dr < 0.149
        + 0.0017862053446757724 * max(0.0, Q.mass_over_sum_pt - 0.109) * max(0.0, 0.0487 - Q.dr_7) / 3.274026625174098e-06   # +0.2%  mass_over_sum_pt > 0.109 and dr_7 < 0.0487
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 22.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.157021084161038 * (0.034526301938073295
        + 0.13679625430084014 * max(0.0, Q.girth2 - 0.00354) / 0.004079404429041682   # +13.7%  girth2 > 0.00354
        + 0.10067598064782979 * max(0.0, Q.mass_over_sum_pt - 0.108) / 0.005362211119909984   # +10.1%  mass_over_sum_pt > 0.108
        - 0.09607895805030675 * max(0.0, Q.mass_over_sum_pt - 0.09) / 0.008381194878995553   # -9.6%  mass_over_sum_pt > 0.09
        + 0.08643814637093122 * max(0.0, 0.0766 - Q.girth) / 0.028331535970658597   # +8.6%  girth < 0.0766
        - 0.07869238781936665 * max(0.0, 0.00295 - Q.e2_sq) / 0.0010378505333789787   # -7.9%  e2_sq < 0.00295
        - 0.0771622075097518 * max(0.0, Q.girth2 - 0.00739) / 0.0025031986218067054   # -7.7%  girth2 > 0.00739
        - 0.07068453176525724 * max(0.0, Q.mass_over_sum_pt - 0.108) * max(0.0, 3.95 - Q.D2) / 0.016246459135340823   # -7.1%  mass_over_sum_pt > 0.108 and D2 < 3.95
        + 0.07036222011888969 * max(0.0, 61.3 - Q.mass) / 24.984249915106073   # +7.0%  mass < 61.3
        + 0.0678553901488766 * max(0.0, Q.e2 - 0.0214) / 0.013188362370198525   # +6.8%  e2 > 0.0214
        + 0.0521617758976661 * max(0.0, 0.236 - Q.tau21) / 0.05529902240918042   # +5.2%  tau21 < 0.236
        - 0.04087719371638238 * max(0.0, Q.C2 - 0.0136) / 0.016741531294551193   # -4.1%  C2 > 0.0136
        - 0.03249509925076076 * max(0.0, 0.016 - Q.centroid_offset) * max(0.0, 0.541 - Q.z_dr_0p05_0p1) / 0.002011158098410647   # -3.2%  centroid_offset < 0.016 and z_dr_0p05_0p1 < 0.541
        - 0.029871262418440115 * max(0.0, 0.000297 - Q.lam2) * max(0.0, 43.8 - Q.mass_top3) / 0.006185590572111073   # -3.0%  lam2 < 0.000297 and mass_top3 < 43.8
        + 0.016571590065424243 * max(0.0, 760.0 - Q.sum_pt) / 94.38999241071429   # +1.7%  sum_pt < 760
        - 0.01412063881626732 * max(0.0, Q.C2 - 0.0674) / 0.002844284472489616   # -1.4%  C2 > 0.0674
        - 0.009439962198126971 * max(0.0, 0.229 - Q.tau21) * max(0.0, 60.9 - Q.mass) / 0.33844893439738266   # -0.9%  tau21 < 0.229 and mass < 60.9
        + 0.008964940551982246 * max(0.0, 0.0107 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.0188) / 1.873928083287933e-05   # +0.9%  girth2_top2 < 0.0107 and centroid_offset > 0.0188
        - 0.0065220414821395905 * max(0.0, Q.max_dr - 0.105) * max(0.0, Q.eccentricity - 0.985) / 0.00015588890035764813   # -0.7%  max_dr > 0.105 and eccentricity > 0.985
        + 0.0034952957666330147 * max(0.0, 0.221 - Q.tau21) * max(0.0, Q.planar_flow - 0.0357) / 0.0016946464331874514   # +0.3%  tau21 < 0.221 and planar_flow > 0.0357
        - 0.0007341231041274523 * max(0.0, Q.girth2 - 0.0187) * max(0.0, Q.pt_6 - 62.0) / 6.403929565559728e-05   # -0.1%  girth2 > 0.0187 and pt_6 > 62
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.4;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.396878071468077 * (-0.07492633794925349
        + 0.15444142479306333 * max(0.0, 54.7 - Q.pt_7) / 20.325426110851726   # +15.4%  pt_7 < 54.7
        - 0.09044128839651717 * max(0.0, Q.log_sum_pt - 6.55) * max(0.0, 0.0123 - Q.lam1) / 0.0009546264447564159   # -9.0%  log_sum_pt > 6.55 and lam1 < 0.0123
        + 0.08331831668082666 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.00124 - Q.lam2) / 2.2977463651207414e-05   # +8.3%  z_7 < 0.0707 and lam2 < 0.00124
        + 0.07634134341239479 * max(0.0, Q.log_sum_pt - 6.55) / 0.09811021499020289   # +7.6%  log_sum_pt > 6.55
        + 0.0682022837592173 * max(0.0, 0.224 - Q.LHA) / 0.03545454142201249   # +6.8%  LHA < 0.224
        - 0.05267879913402985 * max(0.0, 0.228 - Q.LHA) * max(0.0, 6.81 - Q.log_sum_pt) / 0.00507125047729506   # -5.3%  LHA < 0.228 and log_sum_pt < 6.81
        + 0.052447009116363526 * max(0.0, 0.0025 - Q.width) / 0.0007408765747228347   # +5.2%  width < 0.0025
        + 0.0501656028621074 * max(0.0, 0.0346 - Q.e2) / 0.013104664732136199   # +5.0%  e2 < 0.0346
        - 0.04958877558648543 * max(0.0, 539.0 - Q.sum_pt_top2) / 187.47943763130252   # -5.0%  sum_pt_top2 < 539
        - 0.033837968628158244 * max(0.0, 0.0199 - Q.dr_0) / 0.003113356053213479   # -3.4%  dr_0 < 0.0199
        + 0.032976684124479155 * max(0.0, 0.0513 - Q.z_6) / 0.005620566623664374   # +3.3%  z_6 < 0.0513
        - 0.031025530864489836 * max(0.0, 0.0751 - Q.z_7) * max(0.0, 818.0 - Q.sum_pt) / 1.5003193558170715   # -3.1%  z_7 < 0.0751 and sum_pt < 818
        - 0.02554804420478703 * max(0.0, 0.0537 - Q.z_7) * max(0.0, 0.0119 - Q.centroid_offset) / 4.571771094004257e-05   # -2.6%  z_7 < 0.0537 and centroid_offset < 0.0119
        - 0.024532461397637255 * max(0.0, 0.216 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00235) / 0.00013353979578232034   # -2.5%  LHA < 0.216 and centroid_offset > 0.00235
        - 0.023872363154830694 * max(0.0, Q.log_sum_pt - 6.83) / 0.009226693271322741   # -2.4%  log_sum_pt > 6.83
        + 0.021613455127536735 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, 0.0208 - Q.dr_0) / 0.0006056939559680483   # +2.2%  log_sum_pt > 6.58 and dr_0 < 0.0208
        + 0.018910177813339885 * max(0.0, 0.0334 - Q.e2) * max(0.0, 7.69e-05 - Q.lam2) / 5.41616564834919e-07   # +1.9%  e2 < 0.0334 and lam2 < 7.69e-05
        + 0.017363289970721923 * max(0.0, Q.sum_pt - 861.0) * max(0.0, 0.0117 - Q.centroid_offset) / 0.1298733875864319   # +1.7%  sum_pt > 861 and centroid_offset < 0.0117
        + 0.01513316009252698 * max(0.0, 0.0233 - Q.z_7) / 0.0007250581581475096   # +1.5%  z_7 < 0.0233
        - 0.013426887422721159 * max(0.0, 0.0323 - Q.z_7) * max(0.0, 30.9 - Q.pt_5) / 0.012245413281873695   # -1.3%  z_7 < 0.0323 and pt_5 < 30.9
        + 0.011991648841209207 * max(0.0, Q.log_sum_pt - 6.57) * max(0.0, 0.000148 - Q.mean_phi2) / 4.329017738816443e-06   # +1.2%  log_sum_pt > 6.57 and mean_phi2 < 0.000148
        - 0.010354298026814195 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.0182 - Q.centroid_offset) / 4.8932897318377125e-05   # -1.0%  log_sum_pt > 6.9 and centroid_offset < 0.0182
        + 0.008601630464468499 * max(0.0, 0.0283 - Q.z_6) * max(0.0, 1.93 - Q.n_dr_0p2_0p4) / 0.001216736097345637   # +0.9%  z_6 < 0.0283 and n_dr_0p2_0p4 < 1.93
        + 0.008396719085533426 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, 9.11e-05 - Q.mean_eta2) / 1.9931430258598165e-06   # +0.8%  log_sum_pt > 6.58 and mean_eta2 < 9.11e-05
        + 0.008019104985132041 * max(0.0, 0.0137 - Q.mean_phi2) * max(0.0, 24.6 - Q.pt_5) / 0.0036407710381100287   # +0.8%  mean_phi2 < 0.0137 and pt_5 < 24.6
        - 0.0070886005135678614 * max(0.0, 0.05 - Q.z_6) * max(0.0, Q.e2_sq - 0.00225) / 8.134582255729697e-06   # -0.7%  z_6 < 0.05 and e2_sq > 0.00225
        - 0.0068238895239212715 * max(0.0, Q.sum_pt_top5 - 682.0) * max(0.0, 1.55 - Q.D2) / 9.679010552984694   # -0.7%  sum_pt_top5 > 682 and D2 < 1.55
        - 0.0028592420171191166 * max(0.0, Q.log_sum_pt - 6.88) * max(0.0, 0.0438 - Q.planar_flow) / 2.5192534431191435e-05   # -0.3%  log_sum_pt > 6.88 and planar_flow < 0.0438
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 30.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.679342837947885 * (0.27803724626881754
        - 0.17754214307385732 * max(0.0, 0.0132 - Q.width) / 0.008203126921004313   # -17.8%  width < 0.0132
        - 0.14963293037534758 * max(0.0, 0.00869 - Q.girth2) / 0.0044569320105165774   # -15.0%  girth2 < 0.00869
        - 0.1417565545656085 * max(0.0, Q.mass_over_sum_pt - 0.00657) / 0.054842344729439725   # -14.2%  mass_over_sum_pt > 0.00657
        + 0.10698032260161291 * max(0.0, 0.0501 - Q.e2) / 0.02486428783340263   # +10.7%  e2 < 0.0501
        + 0.04048500620073307 * max(0.0, Q.centroid_offset - 0.00824) * max(0.0, 0.00351 - Q.lam2) / 2.7119069542112136e-05   # +4.0%  centroid_offset > 0.00824 and lam2 < 0.00351
        - 0.03639502658590603 * max(0.0, Q.centroid_offset - 0.0186) / 0.00544670974744019   # -3.6%  centroid_offset > 0.0186
        + 0.029028218651050892 * max(0.0, 0.00947 - Q.girth2_top2) / 0.006058276680071452   # +2.9%  girth2_top2 < 0.00947
        - 0.028093885732205334 * max(0.0, 6.69 - Q.log_sum_pt) / 0.18575473104061638   # -2.8%  log_sum_pt < 6.69
        - 0.02773255155288232 * max(0.0, 0.172 - Q.max_dr) / 0.06647003569233911   # -2.8%  max_dr < 0.172
        + 0.023232117692980156 * max(0.0, 49.8 - Q.mass) * max(0.0, 0.75 - Q.z_dr_0p05_0p1) / 11.49590489604021   # +2.3%  mass < 49.8 and z_dr_0p05_0p1 < 0.75
        - 0.02106601095656599 * max(0.0, 0.000717 - Q.lam2) * max(0.0, 0.199 - Q.z_dr_0p2_0p4) / 0.0001034066195783127   # -2.1%  lam2 < 0.000717 and z_dr_0p2_0p4 < 0.199
        + 0.02079150813764605 * max(0.0, 1.66 - Q.D2) / 0.5498877640282988   # +2.1%  D2 < 1.66
        + 0.020318662029480888 * max(0.0, 6.56 - Q.log_sum_pt) / 0.10994059936699084   # +2.0%  log_sum_pt < 6.56
        + 0.019091403653131386 * max(0.0, Q.girth - 0.0885) / 0.007606645687429461   # +1.9%  girth > 0.0885
        + 0.016687288126377528 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.5 - Q.pt_7) / 2.1155166672097243   # +1.7%  log_sum_pt < 6.7 and pt_7 < 46.5
        - 0.012578497028720692 * max(0.0, Q.lam2 - 0.00354) / 0.00015074219637899644   # -1.3%  lam2 > 0.00354
        - 0.01253334406809283 * max(0.0, 5.76 - Q.pt1_dr01) / 3.527658344687893   # -1.3%  pt1_dr01 < 5.76
        + 0.011536356565535586 * max(0.0, Q.LHA - 0.313) * max(0.0, Q.eccentricity - 0.878) / 0.0008467173162078404   # +1.2%  LHA > 0.313 and eccentricity > 0.878
        - 0.009710701772765049 * max(0.0, Q.z_dr_0_0p05 - 0.755) / 0.0986483274449428   # -1.0%  z_dr_0_0p05 > 0.755
        + 0.00968105885086535 * max(0.0, Q.centroid_offset - 0.00324) * max(0.0, Q.planar_flow - 0.0184) / 0.003490111909753794   # +1.0%  centroid_offset > 0.00324 and planar_flow > 0.0184
        - 0.008367107924661743 * max(0.0, 1.66 - Q.D2) * max(0.0, 89.2 - Q.pt_4) / 16.454959780949245   # -0.8%  D2 < 1.66 and pt_4 < 89.2
        - 0.008033101772326074 * max(0.0, Q.sum_pt_top5 - 837.0) / 8.897122141708246   # -0.8%  sum_pt_top5 > 837
        + 0.007224451704921069 * max(0.0, Q.max_dr - 0.172) / 0.018167330382907427   # +0.7%  max_dr > 0.172
        + 0.007032580764367239 * max(0.0, 6.68 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.00426) / 0.0005173979767519907   # +0.7%  log_sum_pt < 6.68 and mean_eta2 > 0.00426
        + 0.006996960967257599 * max(0.0, Q.sum_pt - 987.0) / 4.472128423713236   # +0.7%  sum_pt > 987
        + 0.006552842375305083 * max(0.0, Q.centroid_offset - 0.0188) * max(0.0, 0.00883 - Q.mean_phi2) / 2.072545338092963e-05   # +0.7%  centroid_offset > 0.0188 and mean_phi2 < 0.00883
        + 0.005945271754923184 * max(0.0, Q.C2 - 0.00342) * max(0.0, Q.pt_7 - 28.9) / 0.16285449145897896   # +0.6%  C2 > 0.00342 and pt_7 > 28.9
        - 0.004862904999945699 * max(0.0, Q.mass_over_sum_pt - 0.00635) * max(0.0, 0.524 - Q.tau32) / 0.006064663808199391   # -0.5%  mass_over_sum_pt > 0.00635 and tau32 < 0.524
        + 0.004842178632303612 * max(0.0, Q.girth2_top5 - 0.00208) * max(0.0, Q.z_dr_0p05_0p1 - 0.275) / 0.0007944110071819685   # +0.5%  girth2_top5 > 0.00208 and z_dr_0p05_0p1 > 0.275
        + 0.004157594328207804 * max(0.0, 24.0 - Q.pt_6) / 0.5643905388327206   # +0.4%  pt_6 < 24
        + 0.003663550200958033 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 36.7 - Q.pt_6) / 0.4162789356267569   # +0.4%  log_sum_pt < 6.71 and pt_6 < 36.7
        - 0.003376834342530658 * max(0.0, Q.girth2_top5 - 0.0111) * max(0.0, Q.mean_eta - 0.0139) / 1.0757950000151029e-05   # -0.3%  girth2_top5 > 0.0111 and mean_eta > 0.0139
        + 0.0029920654621568583 * max(0.0, 0.0505 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.156) / 0.00022388927343193474   # +0.3%  e2 < 0.0505 and z_dr_0p1_0p2 > 0.156
        + 0.0029445738571078723 * max(0.0, Q.centroid_offset - 0.0499) / 0.0008065856328024195   # +0.3%  centroid_offset > 0.0499
        - 0.0028851443642562338 * max(0.0, 0.0376 - Q.z_4) / 0.00047845585452967665   # -0.3%  z_4 < 0.0376
        + 0.0026613844716929067 * max(0.0, 31.1 - Q.pt_4) / 0.3128334353665038   # +0.3%  pt_4 < 31.1
        + 0.0018004230946314544 * max(0.0, 6.69 - Q.log_sum_pt) * max(0.0, 0.0493 - Q.z_7) / 0.0001866074235593158   # +0.2%  log_sum_pt < 6.69 and z_7 < 0.0493
        - 0.000787440761081085 * max(0.0, Q.sum_pt - 989.0) * max(0.0, Q.pt_6 - 29.9) / 71.6859497738315   # -0.1%  sum_pt > 989 and pt_6 > 29.9
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 33.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 33.677287401802346 * (0.3355377131530862
        - 0.10704907196048508 * max(0.0, Q.girth2 - 0.00153) / 0.0053016505331021715   # -10.7%  girth2 > 0.00153
        - 0.09171265987719822 * max(0.0, Q.mass_over_sum_pt - 0.108) / 0.005362211119909984   # -9.2%  mass_over_sum_pt > 0.108
        - 0.09169781182477366 * max(0.0, 0.00846 - Q.lam1) / 0.004367939975868871   # -9.2%  lam1 < 0.00846
        - 0.07562499728249958 * max(0.0, Q.girth2 - 0.00751) / 0.0024726648235371454   # -7.6%  girth2 > 0.00751
        - 0.07055072026846143 * max(0.0, 0.0883 - Q.girth) / 0.03724070349349745   # -7.1%  girth < 0.0883
        - 0.06928449623051522 * max(0.0, 0.00544 - Q.width) / 0.0021406549468294977   # -6.9%  width < 0.00544
        - 0.05495452496533324 * max(0.0, 0.158 - Q.max_dr) / 0.056424369856312495   # -5.5%  max_dr < 0.158
        + 0.0396949993338224 * max(0.0, 0.157 - Q.max_dr) * max(0.0, 0.671 - Q.z_dr_0p05_0p1) / 0.028998262494132097   # +4.0%  max_dr < 0.157 and z_dr_0p05_0p1 < 0.671
        - 0.03690699693457617 * max(0.0, Q.e2 - 0.0174) / 0.015459297797302983   # -3.7%  e2 > 0.0174
        + 0.035098292926587796 * max(0.0, Q.mass_over_sum_pt - 0.0726) / 0.013477939546195479   # +3.5%  mass_over_sum_pt > 0.0726
        - 0.03485749627622265 * max(0.0, 0.0673 - Q.C2) / 0.04162786951069523   # -3.5%  C2 < 0.0673
        + 0.03314561683057332 * max(0.0, 0.0397 - Q.centroid_offset) * max(0.0, Q.sum_pt - 567.0) / 4.8744736424158726   # +3.3%  centroid_offset < 0.0397 and sum_pt > 567
        + 0.03139890253996892 * max(0.0, Q.mass_over_sum_pt - 0.0842) / 0.009701191421465277   # +3.1%  mass_over_sum_pt > 0.0842
        + 0.029439365257807923 * max(0.0, 29.5 - Q.mass) / 7.289985034660532   # +2.9%  mass < 29.5
        + 0.029130814217624895 * max(0.0, 0.0245 - Q.e2) / 0.007432172747389875   # +2.9%  e2 < 0.0245
        + 0.020045725410939283 * max(0.0, 0.196 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.0498) / 0.016835053761641263   # +2.0%  max_dr < 0.196 and z_dr_0p05_0p1 > 0.0498
        - 0.01676469894175584 * max(0.0, 49.2 - Q.pt_7) * max(0.0, 0.686 - Q.planar_flow) / 6.224802474820321   # -1.7%  pt_7 < 49.2 and planar_flow < 0.686
        + 0.01579845565509968 * max(0.0, 0.00581 - Q.width) * max(0.0, 2.77 - Q.n_dr_0p1_0p2) / 0.006066694773106287   # +1.6%  width < 0.00581 and n_dr_0p1_0p2 < 2.77
        - 0.015297582007909562 * max(0.0, Q.girth2 - 0.00793) * max(0.0, Q.log_sum_pt - 6.18) / 0.00036026647960350414   # -1.5%  girth2 > 0.00793 and log_sum_pt > 6.18
        - 0.014018041964644143 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.14 - Q.D2) / 0.00040697381728779894   # -1.4%  lam1 < 0.0081 and D2 < 1.14
        - 0.013500318232506702 * max(0.0, 0.021 - Q.centroid_offset) / 0.00849820742302655   # -1.4%  centroid_offset < 0.021
        + 0.012864819253444975 * max(0.0, Q.girth2 - 0.00396) * max(0.0, Q.eccentricity - 0.951) / 7.079284564877561e-05   # +1.3%  girth2 > 0.00396 and eccentricity > 0.951
        - 0.011408526336749972 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -1.1%  mass > 80.4
        - 0.010234059004009562 * max(0.0, 0.000546 - Q.width) / 9.093808611214593e-05   # -1.0%  width < 0.000546
        - 0.009999115545586388 * max(0.0, 0.00101 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.4) / 8.656634652764586e-05   # -1.0%  girth2_top2 < 0.00101 and log_sum_pt > 6.4
        - 0.008206819004487987 * max(0.0, 0.131 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.277) / 0.0076138678310940875   # -0.8%  mass_over_sum_pt < 0.131 and z_dr_0p05_0p1 > 0.277
        + 0.006814169205059887 * max(0.0, 0.0388 - Q.e2) * max(0.0, 1.13 - Q.D2) / 0.0007995914101857589   # +0.7%  e2 < 0.0388 and D2 < 1.13
        + 0.005702581607294389 * max(0.0, Q.centroid_offset - 0.0312) / 0.0025006182255349623   # +0.6%  centroid_offset > 0.0312
        + 0.0034885018779717203 * max(0.0, 0.191 - Q.planar_flow) * max(0.0, Q.sum_pt - 621.0) / 9.551486207006572   # +0.3%  planar_flow < 0.191 and sum_pt > 621
        + 0.0018423402857648105 * max(0.0, Q.mass_top5 - 53.5) / 1.9950168262257295   # +0.2%  mass_top5 > 53.5
        - 0.0017441841038337322 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_2 - 57.7) / 0.04079124260171088   # -0.2%  centroid_offset > 0.031 and pt_2 > 57.7
        + 0.001723294836490978 * max(0.0, 0.0227 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0275) / 4.498906626864236e-05   # +0.2%  centroid_offset < 0.0227 and C2 > 0.0275
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 19.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.785270215320903 * (-0.01314108916231362
        + 0.26796621119487557 * max(0.0, 0.00509 - Q.width) / 0.001942045383540792   # +26.8%  width < 0.00509
        - 0.24127754432033838 * max(0.0, 0.0604 - Q.girth) * max(0.0, 0.00498 - Q.width) / 7.77482314538593e-05   # -24.1%  girth < 0.0604 and width < 0.00498
        + 0.058306956863100425 * max(0.0, 0.00673 - Q.girth2) * max(0.0, 0.0236 - Q.centroid_offset) / 3.937265859964169e-05   # +5.8%  girth2 < 0.00673 and centroid_offset < 0.0236
        + 0.054814915986366994 * max(0.0, 0.0636 - Q.girth) * max(0.0, 0.000187 - Q.lam2) / 2.9153976468289905e-06   # +5.5%  girth < 0.0636 and lam2 < 0.000187
        + 0.0455377345376069 * max(0.0, 0.183 - Q.max_dr) * max(0.0, 0.000197 - Q.lam2) / 1.0700432100001232e-05   # +4.6%  max_dr < 0.183 and lam2 < 0.000197
        + 0.042679723385170634 * max(0.0, 0.0634 - Q.girth) * max(0.0, Q.log_sum_pt - 6.67) / 0.0016956422889372522   # +4.3%  girth < 0.0634 and log_sum_pt > 6.67
        - 0.04194352384312264 * max(0.0, 0.196 - Q.LHA) * max(0.0, 0.000294 - Q.lam2) / 6.638911624151492e-06   # -4.2%  LHA < 0.196 and lam2 < 0.000294
        - 0.04118539413407263 * max(0.0, Q.log_sum_pt - 6.69) * max(0.0, 0.00721 - Q.width) / 0.0002032080179219749   # -4.1%  log_sum_pt > 6.69 and width < 0.00721
        - 0.03215005859058448 * max(0.0, 24.7 - Q.mass) * max(0.0, 0.000203 - Q.lam2) / 0.0009165671421514597   # -3.2%  mass < 24.7 and lam2 < 0.000203
        - 0.023236499325650566 * max(0.0, 0.0264 - Q.C2) / 0.008466674364937725   # -2.3%  C2 < 0.0264
        + 0.02183907589258231 * max(0.0, 0.202 - Q.LHA) * max(0.0, 0.000479 - Q.width) / 8.152679580898897e-06   # +2.2%  LHA < 0.202 and width < 0.000479
        - 0.021495190432098743 * max(0.0, 0.00515 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 1.24e-05) / 1.1463292480562105e-06   # -2.1%  width < 0.00515 and mass_over_sum_pt_sq > 1.24e-05
        - 0.02025255515650995 * max(0.0, 21.6 - Q.mass) * max(0.0, Q.centroid_offset - 0.0322) / 0.003967349270517229   # -2.0%  mass < 21.6 and centroid_offset > 0.0322
        - 0.01807535055704626 * max(0.0, 0.026 - Q.C2) * max(0.0, 0.00453 - Q.width) / 2.2351605937988214e-05   # -1.8%  C2 < 0.026 and width < 0.00453
        - 0.018074849821944756 * max(0.0, 0.0613 - Q.girth) * max(0.0, Q.lam1 - 0.000235) / 1.0611744445950205e-05   # -1.8%  girth < 0.0613 and lam1 > 0.000235
        - 0.015075911145992323 * max(0.0, 0.0632 - Q.girth) * max(0.0, Q.centroid_offset - 0.00688) / 8.984366739928505e-05   # -1.5%  girth < 0.0632 and centroid_offset > 0.00688
        - 0.012143412587898049 * max(0.0, 0.0068 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0184) / 6.036700989640507e-06   # -1.2%  girth2 < 0.0068 and centroid_offset > 0.0184
        - 0.008975160649052237 * max(0.0, 0.0264 - Q.C2) * max(0.0, Q.centroid_offset - 0.0157) / 2.8008829442809693e-05   # -0.9%  C2 < 0.0264 and centroid_offset > 0.0157
        - 0.008474303719955355 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, 0.0017 - Q.girth2) / 0.04233999721869086   # -0.8%  sum_pt_top5 > 655 and girth2 < 0.0017
        + 0.006071938493413571 * max(0.0, Q.log_sum_pt - 6.66) * max(0.0, 50.2 - Q.pt_7) / 1.1123605909536667   # +0.6%  log_sum_pt > 6.66 and pt_7 < 50.2
        + 0.00042368936261722205 * max(0.0, 0.196 - Q.LHA) * max(0.0, Q.girth2 - 0.00678) / 1.6866817961245112e-08   # +0.0%  LHA < 0.196 and girth2 > 0.00678
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 23.14;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.13788956149086 * (-0.09292117996700593
        + 0.24395802195041225 * max(0.0, 0.00624 - Q.width) / 0.0026376980231441124   # +24.4%  width < 0.00624
        - 0.076394348639745 * max(0.0, 0.0767 - Q.mass_over_sum_pt) / 0.027362291051833576   # -7.6%  mass_over_sum_pt < 0.0767
        + 0.0629635080481837 * max(0.0, 0.00415 - Q.girth2) / 0.001456842695622915   # +6.3%  girth2 < 0.00415
        + 0.05877656568906802 * max(0.0, 0.211 - Q.max_dr) / 0.09714040612266919   # +5.9%  max_dr < 0.211
        - 0.057941625249274015 * max(0.0, 0.0535 - Q.girth) / 0.014764833987125466   # -5.8%  girth < 0.0535
        + 0.05784262270597604 * max(0.0, 0.0194 - Q.centroid_offset) / 0.007394233238220205   # +5.8%  centroid_offset < 0.0194
        + 0.05314171496971443 * max(0.0, 0.0202 - Q.e2) / 0.005416683401222371   # +5.3%  e2 < 0.0202
        + 0.046079225424500506 * max(0.0, 51.9 - Q.mass) * max(0.0, 6.83 - Q.log_sum_pt) / 4.696810700225262   # +4.6%  mass < 51.9 and log_sum_pt < 6.83
        - 0.039788382760233075 * max(0.0, 0.00609 - Q.width) * max(0.0, Q.centroid_offset - 0.00339) / 2.08757189600136e-05   # -4.0%  width < 0.00609 and centroid_offset > 0.00339
        - 0.03382489128249115 * max(0.0, 39.0 - Q.mass) * max(0.0, 0.0299 - Q.centroid_offset) / 0.21619795550378873   # -3.4%  mass < 39 and centroid_offset < 0.0299
        - 0.033420425481326524 * max(0.0, 27.9 - Q.mass) / 6.666190636939381   # -3.3%  mass < 27.9
        - 0.023208957767491965 * max(0.0, Q.log_sum_pt - 6.34) / 0.23866946740512646   # -2.3%  log_sum_pt > 6.34
        - 0.02069270027297065 * max(0.0, 0.00588 - Q.lam1) * max(0.0, 0.00203 - Q.mean_phi2) / 3.92447060364776e-06   # -2.1%  lam1 < 0.00588 and mean_phi2 < 0.00203
        - 0.01857042797845766 * max(0.0, 0.0176 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.0442) / 0.0002261476377237754   # -1.9%  centroid_offset < 0.0176 and z_4 > 0.0442
        + 0.018539898087467258 * max(0.0, 0.0322 - Q.e2) * max(0.0, 0.0556 - Q.dr01) / 0.0005402696655278501   # +1.9%  e2 < 0.0322 and dr01 < 0.0556
        - 0.018213401628509755 * max(0.0, 0.238 - Q.max_dr) * max(0.0, 0.903 - Q.z_top5) / 0.010405424084433   # -1.8%  max_dr < 0.238 and z_top5 < 0.903
        - 0.017404743918368337 * max(0.0, 0.0194 - Q.e2) * max(0.0, 53.0 - Q.pt_7) / 0.10681937470271519   # -1.7%  e2 < 0.0194 and pt_7 < 53
        - 0.015125348635856115 * max(0.0, 0.00445 - Q.girth2) * max(0.0, Q.centroid_offset - 0.0105) / 5.9116325391129254e-06   # -1.5%  girth2 < 0.00445 and centroid_offset > 0.0105
        + 0.014427769436874292 * max(0.0, 61.8 - Q.mass) * max(0.0, 0.323 - Q.planar_flow) / 2.318250943396184   # +1.4%  mass < 61.8 and planar_flow < 0.323
        + 0.014248861715236443 * max(0.0, Q.lam2 - 0.00114) / 0.00031102697051329814   # +1.4%  lam2 > 0.00114
        - 0.012610499164227107 * max(0.0, 0.000725 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.0232) / 3.52817819803095e-06   # -1.3%  girth2_top2 < 0.000725 and z_7 > 0.0232
        + 0.010464956312185894 * max(0.0, 0.0179 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.2) / 0.10392146069407032   # +1.0%  centroid_offset < 0.0179 and pt_4 > 47.2
        + 0.007190870791866011 * max(0.0, 0.000176 - Q.width) / 1.4989331011995144e-05   # +0.7%  width < 0.000176
        - 0.007051264434215647 * max(0.0, 0.0348 - Q.e2) * max(0.0, 0.432 - Q.tau21) / 0.0008541956950143977   # -0.7%  e2 < 0.0348 and tau21 < 0.432
        - 0.0065054413872957404 * max(0.0, 0.0214 - Q.e2) * max(0.0, Q.eccentricity - 0.904) / 0.00013560557150270336   # -0.7%  e2 < 0.0214 and eccentricity > 0.904
        + 0.006032965648021987 * max(0.0, Q.girth2 - 0.0165) / 0.0010115224122623251   # +0.6%  girth2 > 0.0165
        + 0.004903538128054896 * max(0.0, 0.000666 - Q.girth2_top2) * max(0.0, Q.pt_7 - 32.4) / 0.0008660879669274337   # +0.5%  girth2_top2 < 0.000666 and pt_7 > 32.4
        + 0.004859256752401167 * max(0.0, 35.8 - Q.pt_5) / 1.5401773436710537   # +0.5%  pt_5 < 35.8
        - 0.004700712196089328 * max(0.0, Q.sum_pt - 853.0) / 21.368282839581145   # -0.5%  sum_pt > 853
        + 0.003726120125919134 * max(0.0, Q.log_sum_pt - 6.48) * max(0.0, 0.0257 - Q.dr_5) / 0.0008621455596636533   # +0.4%  log_sum_pt > 6.48 and dr_5 < 0.0257
        + 0.0035637828623812126 * max(0.0, Q.sum_pt - 986.0) * max(0.0, 0.0548 - Q.dr_3) / 0.16136675986479507   # +0.4%  sum_pt > 986 and dr_3 < 0.0548
        - 0.0023899069930635223 * max(0.0, 0.00621 - Q.width) * max(0.0, Q.C2 - 0.0292) / 6.0236823603200954e-06   # -0.2%  width < 0.00621 and C2 > 0.0292
        + 0.001437243562121117 * max(0.0, Q.girth2 - 0.019) * max(0.0, 29.1 - Q.pt_7) / 0.0006035350782817083   # +0.1%  girth2 > 0.019 and pt_7 < 29.1
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 12.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.691805811969472 * (-0.11424694180496557
        + 0.14032045442269253 * Q.e2 / 0.028632153681352513   # +14.0%  e2
        - 0.12678411742229342 * max(0.0, 0.00352 - Q.lam2) / 0.003142811324932976   # -12.7%  lam2 < 0.00352
        + 0.12417340455798533 * max(0.0, Q.mass - 15.3) / 27.360846139949214   # +12.4%  mass > 15.3
        + 0.08249468930527715 * max(0.0, 0.108 - Q.mass_over_sum_pt) / 0.052089879461757825   # +8.2%  mass_over_sum_pt < 0.108
        - 0.07877680765347707 * max(0.0, 0.00422 - Q.lam1) / 0.0015311178334223604   # -7.9%  lam1 < 0.00422
        + 0.044825433064138415 * max(0.0, 0.00396 - Q.mass_over_sum_pt_sq) / 0.001513073648636916   # +4.5%  mass_over_sum_pt_sq < 0.00396
        + 0.041436948473887646 * max(0.0, 0.0381 - Q.centroid_offset) / 0.022571231908633763   # +4.1%  centroid_offset < 0.0381
        + 0.03551963248792866 * max(0.0, 0.004 - Q.girth2_top2) / 0.0019021446331194626   # +3.6%  girth2_top2 < 0.004
        + 0.034971367474552824 * max(0.0, Q.girth2 - 0.00871) / 0.0022082079849057197   # +3.5%  girth2 > 0.00871
        - 0.03469991649507273 * max(0.0, 0.00165 - Q.girth2) / 0.0004275772833466208   # -3.5%  girth2 < 0.00165
        - 0.03283550038364662 * max(0.0, 0.385 - Q.tau21) / 0.13400057704440257   # -3.3%  tau21 < 0.385
        - 0.03282722761404458 * max(0.0, Q.log_sum_pt - 6.7) / 0.0356099827540834   # -3.3%  log_sum_pt > 6.7
        + 0.029287311087389425 * max(0.0, Q.eccentricity - 0.898) * max(0.0, 0.0572 - Q.z_dr_0p2_0p4) / 0.0023981217101670133   # +2.9%  eccentricity > 0.898 and z_dr_0p2_0p4 < 0.0572
        + 0.027903597529959776 * max(0.0, 0.00227 - Q.girth2_top5) / 0.0007632479338482769   # +2.8%  girth2_top5 < 0.00227
        + 0.026391422880480426 * max(0.0, Q.lam2 - 0.000205) / 0.00044364876066308026   # +2.6%  lam2 > 0.000205
        - 0.025948302300922273 * max(0.0, Q.mass - 53.2) / 6.3947730864774   # -2.6%  mass > 53.2
        + 0.02374166191859896 * max(0.0, 3.02 - Q.n_dr_0p05_0p1) / 1.6833774453870871   # +2.4%  n_dr_0p05_0p1 < 3.02
        - 0.018911342567831932 * max(0.0, Q.LHA - 0.347) / 0.00885679289721605   # -1.9%  LHA > 0.347
        - 0.01463863990298227 * max(0.0, Q.LHA - 0.303) / 0.018037939320388212   # -1.5%  LHA > 0.303
        - 0.008132314893986343 * max(0.0, 0.00426 - Q.lam1) * max(0.0, Q.sum_pt - 994.0) / 0.013474381388545922   # -0.8%  lam1 < 0.00426 and sum_pt > 994
        - 0.007853894895434558 * max(0.0, Q.lam2 - 0.000209) * max(0.0, 8.04 - Q.n_pt_above_50) / 0.0016287599490273478   # -0.8%  lam2 > 0.000209 and n_pt_above_50 < 8.04
        - 0.005172790579760685 * max(0.0, Q.lam2 - 0.000227) * max(0.0, 2.0 - Q.D2) / 0.000341937778876602   # -0.5%  lam2 > 0.000227 and D2 < 2
        + 0.002353222087656494 * max(0.0, Q.centroid_offset - 0.0373) / 0.001736432428428699   # +0.2%  centroid_offset > 0.0373
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 27.84;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.83821741984969 * (-0.049931357997401005
        + 0.2189422920628442 * max(0.0, 0.0086 - Q.width) / 0.00438486556032064   # +21.9%  width < 0.0086
        + 0.1493184217370177 * max(0.0, 0.0499 - Q.centroid_offset) / 0.03352224749277346   # +14.9%  centroid_offset < 0.0499
        - 0.13659540718380017 * max(0.0, 0.0879 - Q.girth) / 0.036918181007141046   # -13.7%  girth < 0.0879
        - 0.07331292556096296 * max(0.0, Q.girth - 0.0754) / 0.010798418843657898   # -7.3%  girth > 0.0754
        + 0.05758417938938012 * max(0.0, Q.girth - 0.0761) * max(0.0, 7.11 - Q.n_pt_above_50) / 0.027638636306641224   # +5.8%  girth > 0.0761 and n_pt_above_50 < 7.11
        - 0.04385800801623284 * max(0.0, 0.0494 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt) / 0.007354992546731327   # -4.4%  centroid_offset < 0.0494 and log_sum_pt < 6.8
        + 0.043382078883834004 * max(0.0, 0.268 - Q.planar_flow) / 0.11839997491110209   # +4.3%  planar_flow < 0.268
        - 0.03297278958656016 * max(0.0, Q.girth - 0.0994) * max(0.0, 6.38 - Q.n_pt_above_50) / 0.011967453526070633   # -3.3%  girth > 0.0994 and n_pt_above_50 < 6.38
        - 0.03179691841489055 * max(0.0, 0.235 - Q.planar_flow) * max(0.0, Q.width - 0.00609) / 0.00024184959784561354   # -3.2%  planar_flow < 0.235 and width > 0.00609
        - 0.02477227465872519 * max(0.0, Q.centroid_offset - 0.0144) / 0.007102121193963178   # -2.5%  centroid_offset > 0.0144
        - 0.021596568119544243 * max(0.0, 0.00368 - Q.width) / 0.0012370575284659817   # -2.2%  width < 0.00368
        - 0.020590187889655914 * max(0.0, 0.00575 - Q.e2_sq) / 0.0025250842607391953   # -2.1%  e2_sq < 0.00575
        - 0.020169956984785097 * max(0.0, Q.girth - 0.0763) * max(0.0, 40.3 - Q.pt_7) / 0.06881074116316954   # -2.0%  girth > 0.0763 and pt_7 < 40.3
        - 0.019177561846058202 * max(0.0, 0.0507 - Q.centroid_offset) * max(0.0, 48.5 - Q.pt_7) / 0.5084467964316021   # -1.9%  centroid_offset < 0.0507 and pt_7 < 48.5
        - 0.019035679585426648 * max(0.0, 0.187 - Q.LHA) / 0.02180738218245684   # -1.9%  LHA < 0.187
        + 0.01694853552320999 * max(0.0, 6.69 - Q.log_sum_pt) / 0.18575473104061638   # +1.7%  log_sum_pt < 6.69
        + 0.014422501637892715 * max(0.0, 6.72 - Q.log_sum_pt) * max(0.0, 0.055 - Q.z_dr_0p2_0p4) / 0.00826124971880238   # +1.4%  log_sum_pt < 6.72 and z_dr_0p2_0p4 < 0.055
        - 0.011644734802326858 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, 70.6 - Q.mass) / 2.3661945928734704   # -1.2%  planar_flow < 0.26 and mass < 70.6
        - 0.009677070889056257 * max(0.0, Q.girth - 0.076) * max(0.0, 0.197 - Q.z_dr_0p1_0p2) / 0.0002754523552114992   # -1.0%  girth > 0.076 and z_dr_0p1_0p2 < 0.197
        - 0.008677218448892337 * max(0.0, 0.0328 - Q.C2) / 0.012848845413818923   # -0.9%  C2 < 0.0328
        - 0.007942123986132836 * max(0.0, 0.048 - Q.centroid_offset) * max(0.0, 0.00154 - Q.mean_phi2) / 2.3799200678295928e-05   # -0.8%  centroid_offset < 0.048 and mean_phi2 < 0.00154
        + 0.007850141176048715 * max(0.0, 22.3 - Q.mass) / 4.610420608340901   # +0.8%  mass < 22.3
        - 0.007521987606899657 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.119 - Q.dr_0) / 0.010629376976156728   # -0.8%  log_sum_pt < 6.71 and dr_0 < 0.119
        + 0.002210436009822742 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.0283 - Q.z_7) / 7.404885467401986e-05   # +0.2%  LHA < 0.17 and z_7 < 0.0283
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4139;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4138560552136745 * (-2.249574431185556
        + 0.477659388337898 * max(0.0, Q.girth2 - 0.0188) / 0.0007632518540281819   # +47.8%  girth2 > 0.0188
        - 0.35373332069313806 * max(0.0, Q.e2 - 0.0634) / 0.0017659188986694303   # -35.4%  e2 > 0.0634
        + 0.16860729096896396 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.pt_7 - 15.6) / 0.014507099442968739   # +16.9%  girth2 > 0.0188 and pt_7 > 15.6
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 25.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.56407260849857 * (0.024604843274889387
        + 0.21146630510697406 * max(0.0, 0.149 - Q.girth) / 0.09116256286687346   # +21.1%  girth < 0.149
        - 0.11768777510206024 * max(0.0, 0.0501 - Q.e2) / 0.02486428783340263   # -11.8%  e2 < 0.0501
        + 0.1082647581368672 * max(0.0, 0.016 - Q.lam1) / 0.0108112817888766   # +10.8%  lam1 < 0.016
        + 0.05312680388376751 * max(0.0, 0.0378 - Q.centroid_offset) / 0.022301107913663326   # +5.3%  centroid_offset < 0.0378
        + 0.052304251697328996 * max(0.0, 0.0132 - Q.width) / 0.008203126921004313   # +5.2%  width < 0.0132
        - 0.04340682718114533 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.0495 - Q.centroid_offset) / 7.497670822725328e-06   # -4.3%  lam2 < 0.00031 and centroid_offset < 0.0495
        + 0.041469050920147334 * max(0.0, Q.sum_pt_top5 - 903.0) / 4.093119029838498   # +4.1%  sum_pt_top5 > 903
        - 0.0379290278098249 * max(0.0, 0.0168 - Q.lam1) * max(0.0, 0.0374 - Q.centroid_offset) / 0.0002826881693586371   # -3.8%  lam1 < 0.0168 and centroid_offset < 0.0374
        - 0.03646562857727612 * max(0.0, 0.147 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt) / 0.019340455947386453   # -3.6%  girth < 0.147 and log_sum_pt < 6.8
        + 0.032869449420554725 * max(0.0, Q.sum_pt - 988.0) / 4.42251048204438   # +3.3%  sum_pt > 988
        - 0.02523329243902209 * max(0.0, 0.152 - Q.girth) * max(0.0, 38.1 - Q.pt_7) / 0.6677698965451747   # -2.5%  girth < 0.152 and pt_7 < 38.1
        + 0.02382866858528404 * max(0.0, 0.0003 - Q.lam2) / 0.00019399930378281852   # +2.4%  lam2 < 0.0003
        - 0.022218859121402555 * max(0.0, Q.z_top5_slots - 0.931) / 0.000943529116042418   # -2.2%  z_top5_slots > 0.931
        + 0.021597849635110074 * max(0.0, Q.sum_pt_top5 - 657.0) * max(0.0, 42.1 - Q.pt_7) / 873.6218295243485   # +2.2%  sum_pt_top5 > 657 and pt_7 < 42.1
        - 0.01977630323485299 * max(0.0, 0.00801 - Q.width) / 0.003919091874600519   # -2.0%  width < 0.00801
        + 0.018940661883414436 * max(0.0, Q.sum_pt_top5 - 902.0) * max(0.0, 3.86 - Q.D2) / 5.291808258367518   # +1.9%  sum_pt_top5 > 902 and D2 < 3.86
        - 0.018865365691375903 * max(0.0, 0.515 - Q.tau21) * max(0.0, Q.max_dr - 0.015) / 0.029228822928497662   # -1.9%  tau21 < 0.515 and max_dr > 0.015
        - 0.01510756774813702 * max(0.0, 0.0289 - Q.z_7) / 0.0014357284715657472   # -1.5%  z_7 < 0.0289
        - 0.013755844448928134 * max(0.0, 0.0172 - Q.lam1) * max(0.0, 56.9 - Q.pt_6) / 0.20808012206130816   # -1.4%  lam1 < 0.0172 and pt_6 < 56.9
        - 0.01240099414313646 * max(0.0, Q.mass - 53.5) / 6.290077672474324   # -1.2%  mass > 53.5
        - 0.011527365982285476 * max(0.0, Q.sum_pt - 988.0) * max(0.0, Q.n_pt_above_50 - 6.0) / 2.338781118697479   # -1.2%  sum_pt > 988 and n_pt_above_50 > 6
        - 0.00977252630850141 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, Q.z_7 - 0.0221) / 0.3103423256136553   # -1.0%  sum_pt_top5 > 660 and z_7 > 0.0221
        - 0.00959832125617704 * max(0.0, Q.sum_pt_top5 - 902.0) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.9584850840336134   # -1.0%  sum_pt_top5 > 902 and n_pt_above_50 > 6
        + 0.009058929353174344 * max(0.0, Q.sum_pt_top5 - 840.0) * max(0.0, Q.n_pt_above_50 - 2.02) / 21.246158508239258   # +0.9%  sum_pt_top5 > 840 and n_pt_above_50 > 2.02
        + 0.008915085783582466 * max(0.0, Q.sum_pt - 989.0) * max(0.0, 3.94 - Q.D2) / 7.751901370152909   # +0.9%  sum_pt > 989 and D2 < 3.94
        + 0.008190211894593472 * max(0.0, 0.0493 - Q.e2) * max(0.0, Q.pt_dispersion - 0.397) / 0.001598283752308215   # +0.8%  e2 < 0.0493 and pt_dispersion > 0.397
        - 0.007699675808133031 * max(0.0, 31.3 - Q.pt_6) / 1.6823510377864335   # -0.8%  pt_6 < 31.3
        - 0.006554089119861157 * max(0.0, Q.C2 - 0.0669) / 0.00289376874167014   # -0.7%  C2 > 0.0669
        - 0.0009919122117789435 * max(0.0, Q.girth - 0.102) * max(0.0, Q.pt_4 - 81.4) / 0.001014292632126933   # -0.1%  girth > 0.102 and pt_4 > 81.4
        - 0.000976597515302509 * max(0.0, Q.sum_pt_top5 - 658.0) * max(0.0, 0.362 - Q.tau32) / 0.17706248078349388   # -0.1%  sum_pt_top5 > 658 and tau32 < 0.362
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 37.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.27107398133745 * (0.008183289812166964
        + 0.11514846202353228 * max(0.0, 0.0443 - Q.e2) / 0.020054704892132232   # +11.5%  e2 < 0.0443
        - 0.1063252292954857 * max(0.0, 0.0872 - Q.girth) / 0.03635647235921757   # -10.6%  girth < 0.0872
        + 0.10296357306206498 * max(0.0, 0.0133 - Q.girth2) / 0.00828847289196343   # +10.3%  girth2 < 0.0133
        - 0.07154957470485293 * max(0.0, 0.00609 - Q.width) / 0.002539742373483623   # -7.2%  width < 0.00609
        - 0.062299636203997616 * max(0.0, 0.00747 - Q.width) / 0.003507514123821155   # -6.2%  width < 0.00747
        + 0.05567944879347734 * max(0.0, Q.lam1 - 0.00428) / 0.003197585293099822   # +5.6%  lam1 > 0.00428
        - 0.052261067790836654 * max(0.0, 0.0669 - Q.C2) / 0.04126750262660942   # -5.2%  C2 < 0.0669
        + 0.04171176629058374 * max(0.0, 0.00746 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.009365315224750629   # +4.2%  width < 0.00746 and n_dr_0p1_0p2 < 3
        - 0.04125787885562865 * max(0.0, Q.lam1 - 0.006) / 0.0024643036140083226   # -4.1%  lam1 > 0.006
        - 0.03883445150017523 * max(0.0, 69.5 - Q.mass) / 31.602657530298956   # -3.9%  mass < 69.5
        + 0.030574750686318975 * max(0.0, 86.4 - Q.mass) / 46.89521789262315   # +3.1%  mass < 86.4
        - 0.02830802346368166 * max(0.0, 0.302 - Q.LHA) / 0.07757870858678792   # -2.8%  LHA < 0.302
        - 0.027673911552121472 * max(0.0, Q.lam1 - 0.00418) * max(0.0, Q.D2 - 0.451) / 0.0014568310802430913   # -2.8%  lam1 > 0.00418 and D2 > 0.451
        + 0.02608342976302141 * max(0.0, 0.00575 - Q.e2_sq) / 0.0025250842607391953   # +2.6%  e2_sq < 0.00575
        - 0.025249745817999877 * max(0.0, 0.0804 - Q.max_dr) / 0.015327119615515299   # -2.5%  max_dr < 0.0804
        - 0.022850204580463846 * max(0.0, 0.0131 - Q.girth2) * max(0.0, 2.98 - Q.n_dr_0p1_0p2) / 0.01953329507814596   # -2.3%  girth2 < 0.0131 and n_dr_0p1_0p2 < 2.98
        - 0.022223431277452056 * max(0.0, 0.177 - Q.max_dr) / 0.07019416536110883   # -2.2%  max_dr < 0.177
        + 0.017168470697606058 * max(0.0, 0.582 - Q.z_dr_0p05_0p1) / 0.3677513456993676   # +1.7%  z_dr_0p05_0p1 < 0.582
        - 0.013106376020569258 * max(0.0, 5.09 - Q.n_dr_0p05_0p1) / 3.235024571456053   # -1.3%  n_dr_0p05_0p1 < 5.09
        + 0.013019841989667682 * max(0.0, Q.lam1 - 0.0023) * max(0.0, Q.D2 - 0.452) / 0.0020135414689719   # +1.3%  lam1 > 0.0023 and D2 > 0.452
        - 0.008480226832710946 * max(0.0, 0.601 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2) / 0.016208572392845778   # -0.8%  z_dr_0p05_0p1 < 0.601 and C2 < 0.067
        - 0.008241496089364172 * max(0.0, Q.centroid_offset - 0.0123) / 0.008104733785319137   # -0.8%  centroid_offset > 0.0123
        - 0.007323743543559736 * max(0.0, 0.0337 - Q.girth) / 0.006530234149099461   # -0.7%  girth < 0.0337
        - 0.007252244511255144 * max(0.0, Q.lam1 - 0.00594) * max(0.0, Q.D2 - 1.68) / 7.36509377955692e-05   # -0.7%  lam1 > 0.00594 and D2 > 1.68
        - 0.006399462252856086 * max(0.0, Q.z_dr_0_0p05 - 0.602) / 0.17933445944885382   # -0.6%  z_dr_0_0p05 > 0.602
        + 0.005906128973860539 * max(0.0, 0.588 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.8004646179202792   # +0.6%  z_dr_0p05_0p1 < 0.588 and n_dr_0p1_0p2 < 3
        + 0.005630709688070079 * max(0.0, 0.0134 - Q.girth2) * max(0.0, Q.eccentricity - 0.971) / 5.945116072280267e-05   # +0.6%  girth2 < 0.0134 and eccentricity > 0.971
        - 0.00543753758311583 * max(0.0, 0.0089 - Q.width) * max(0.0, 1.03 - Q.D2) / 0.0004110808631574299   # -0.5%  width < 0.0089 and D2 < 1.03
        - 0.005062126836999014 * max(0.0, Q.centroid_offset - 0.0497) / 0.000816757159500883   # -0.5%  centroid_offset > 0.0497
        - 0.0048455862141832656 * max(0.0, 0.00436 - Q.girth2) * max(0.0, 0.884 - Q.D2) / 1.530510188743842e-05   # -0.5%  girth2 < 0.00436 and D2 < 0.884
        + 0.004591152245652035 * max(0.0, 0.134 - Q.tau21) / 0.0162968738102172   # +0.5%  tau21 < 0.134
        - 0.004148307419043012 * max(0.0, 0.111 - Q.planar_flow) * max(0.0, 0.159 - Q.max_dr) / 0.0010307458180832208   # -0.4%  planar_flow < 0.111 and max_dr < 0.159
        + 0.003668710814211708 * max(0.0, 0.119 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00973) / 0.00031006075322588967   # +0.4%  planar_flow < 0.119 and centroid_offset > 0.00973
        + 0.0030028647792153342 * max(0.0, Q.lam1 - 0.00543) * max(0.0, 0.161 - Q.max_dr) / 1.3323808969296103e-05   # +0.3%  lam1 > 0.00543 and max_dr < 0.161
        - 0.002772051127425026 * max(0.0, 0.115 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0185) / 0.00015749591867425004   # -0.3%  planar_flow < 0.115 and centroid_offset > 0.0185
        - 0.0018286178865126214 * max(0.0, Q.lam1 - 0.0023) * max(0.0, Q.D2 - 1.63) / 0.00024082880753289345   # -0.2%  lam1 > 0.0023 and D2 > 1.63
        - 0.00111975883242791 * max(0.0, 0.0125 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031) / 4.939007607654045e-06   # -0.1%  girth2 < 0.0125 and centroid_offset > 0.031
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 23.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.771655365655064 * (-0.10180191336175856
        - 0.18611045912539864 * max(0.0, 36.2 - Q.mass) / 10.1008075211708   # -18.6%  mass < 36.2
        - 0.10921212976674158 * max(0.0, 0.00854 - Q.e2_sq) / 0.004594961257635716   # -10.9%  e2_sq < 0.00854
        + 0.10880168319403045 * max(0.0, 0.0245 - Q.e2) / 0.007432172747389875   # +10.9%  e2 < 0.0245
        - 0.09454281048095006 * max(0.0, 0.0068 - Q.width) / 0.0030207514893193367   # -9.5%  width < 0.0068
        + 0.08343367704002365 * max(0.0, 0.0141 - Q.width) / 0.008974464327985565   # +8.3%  width < 0.0141
        - 0.07216396569259523 * max(0.0, 0.0676 - Q.mass_over_sum_pt) / 0.02196487736572765   # -7.2%  mass_over_sum_pt < 0.0676
        + 0.06606902823551895 * max(0.0, 0.0424 - Q.e2) / 0.018564659214639366   # +6.6%  e2 < 0.0424
        + 0.06295501033538002 * max(0.0, Q.girth - 0.0325) / 0.032322782056887836   # +6.3%  girth > 0.0325
        + 0.03806764873131693 * max(0.0, 0.323 - Q.z_dr_0p1_0p2) / 0.22399282827269365   # +3.8%  z_dr_0p1_0p2 < 0.323
        + 0.02592610132454936 * max(0.0, 0.245 - Q.tau21) * max(0.0, 0.221 - Q.z_dr_0p2_0p4) / 0.011829296461847228   # +2.6%  tau21 < 0.245 and z_dr_0p2_0p4 < 0.221
        - 0.0242948238280191 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -2.4%  mass > 80.4
        - 0.019628357681850676 * max(0.0, Q.LHA - 0.341) / 0.009761476029430177   # -2.0%  LHA > 0.341
        - 0.01905115621936239 * max(0.0, 0.00218 - Q.girth2_top3) / 0.0007931305078177562   # -1.9%  girth2_top3 < 0.00218
        - 0.013014370123601595 * max(0.0, 0.00761 - Q.width) * max(0.0, Q.e2 - 0.0242) / 4.175075862069296e-06   # -1.3%  width < 0.00761 and e2 > 0.0242
        + 0.011823692757042327 * max(0.0, Q.LHA - 0.179) * max(0.0, Q.sum_pt_top3 - 350.0) / 6.190941616074936   # +1.2%  LHA > 0.179 and sum_pt_top3 > 350
        - 0.009216501973593411 * max(0.0, 0.744 - Q.D2) / 0.09567314785726738   # -0.9%  D2 < 0.744
        - 0.007464730955247004 * max(0.0, 0.0798 - Q.planar_flow) / 0.019915713991635116   # -0.7%  planar_flow < 0.0798
        - 0.007374679698090503 * max(0.0, 0.249 - Q.tau21) * max(0.0, 0.605 - Q.z_dr_0p05_0p1) / 0.014252710911796794   # -0.7%  tau21 < 0.249 and z_dr_0p05_0p1 < 0.605
        + 0.007110644641122375 * max(0.0, 0.00812 - Q.girth2_top2) * max(0.0, 0.00167 - Q.mean_phi) / 2.449736142556567e-05   # +0.7%  girth2_top2 < 0.00812 and mean_phi < 0.00167
        + 0.006842550898586174 * max(0.0, Q.mass - 80.4) * max(0.0, 0.211 - Q.z_dr_0p2_0p4) / 0.08384472256863089   # +0.7%  mass > 80.4 and z_dr_0p2_0p4 < 0.211
        - 0.0062361475889839806 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.log_sum_pt - 6.9) / 1.8234139150638124e-05   # -0.6%  width < 0.0059 and log_sum_pt > 6.9
        - 0.006174508188306811 * max(0.0, Q.z_dr_0p05_0p1 - 0.754) / 0.02234068199464912   # -0.6%  z_dr_0p05_0p1 > 0.754
        - 0.005896823166513504 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00856) / 0.00012296249830400863   # -0.6%  tau21 < 0.243 and girth2_top5 > 0.00856
        + 0.0052300004086286035 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # +0.5%  log_sum_pt > 6.9
        - 0.0026125236742205386 * max(0.0, Q.LHA - 0.179) * max(0.0, Q.z_top5 - 0.865) / 0.0002915681334187115   # -0.3%  LHA > 0.179 and z_top5 > 0.865
        + 0.0007459742703262336 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.mean_phi - 0.0175) / 3.604277086553096e-07   # +0.1%  log_sum_pt > 6.9 and mean_phi > 0.0175
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.8899016806722689, 1.0306703781512605, 1.9705911764705883, 0.7707991596638656, 3.2135432773109245, 2.293000630252101, 1.3266686974789916, 1.3858252100840336, 0.6383090336134454, 3.335466806722689, 1.9087110294117646, 2.2182794117647058, 0.06416302521008403, 3.9483974789915965, 0.34236481092436977, 0.3427701680672269]
T = [2.6371397419905462, 1.7213898716517855, 3.306960136554623, 2.4985068244485293, 3.5085902967436975]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +32%, n9 +22%, n5 -16%, n1 +15%, n6 +6%, n0 -5% ...
            + 0.32108211129554215 * h[2] / H_AVG[2]
            + 0.21738831214637894 * h[9] / H_AVG[9]
            - 0.1630317921066051 * h[5] / H_AVG[5]
            + 0.15266753219586474 * h[1] / H_AVG[1]
            + 0.05502339768966437 * h[6] / H_AVG[6]
            - 0.05272649582842641 * h[0] / H_AVG[0]
            - 0.03808035873751828 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +49%, n4 -18%, n10 -14%, n6 +10%, n5 +6%, n8 +2% ...
            + 0.49198376430656066 * h[9] / H_AVG[9]
            - 0.1750153682261482 * h[4] / H_AVG[4]
            - 0.13860246455820555 * h[10] / H_AVG[10]
            + 0.0963370296966752 * h[6] / H_AVG[6]
            + 0.06244047691527833 * h[5] / H_AVG[5]
            + 0.023175641531200105 * h[8] / H_AVG[8]
            + 0.01244525476593213 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +25%, n6 -13%, n3 -12%, n0 +9%, n7 +9%, n13 +8% ...
            + 0.2515466606980143 * h[11] / H_AVG[11]
            - 0.1253670896662582 * h[6] / H_AVG[6]
            - 0.1165419490763695 * h[3] / H_AVG[3]
            + 0.0925029906921709 * h[0] / H_AVG[0]
            + 0.09167006924423356 * h[7] / H_AVG[7]
            + 0.08395072401170173 * h[13] / H_AVG[13]
            - 0.07764641773420303 * h[14] / H_AVG[14]
            - 0.0712601545876923 * h[15] / H_AVG[15]
            - 0.04825496885777943 * h[8] / H_AVG[8]
            - 0.03151938136716707 * h[9] / H_AVG[9]
            - 0.009739594064409699 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +26%, n6 -20%, n3 -17%, n4 +10%, n13 +9%, n1 +5% ...
            + 0.2599975156642895 * h[7] / H_AVG[7]
            - 0.199119232609833 * h[6] / H_AVG[6]
            - 0.1735334572906852 * h[3] / H_AVG[3]
            + 0.10048324306471708 * h[4] / H_AVG[4]
            + 0.08642281262530176 * h[13] / H_AVG[13]
            + 0.05156431673839585 * h[1] / H_AVG[1]
            + 0.05138541261538328 * h[14] / H_AVG[14]
            - 0.04171825215370001 * h[9] / H_AVG[9]
            - 0.02143593855194912 * h[15] / H_AVG[15]
            + 0.014339818685745262 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -46%, n10 +20%, n5 -16%, n4 +11%, n8 +3%, n3 +1% ...
            - 0.45717406142547706 * h[13] / H_AVG[13]
            + 0.2040040516254379 * h[10] / H_AVG[10]
            - 0.16338475258711607 * h[5] / H_AVG[5]
            + 0.11448840579553402 * h[4] / H_AVG[4]
            + 0.03411140477518793 * h[8] / H_AVG[8]
            + 0.013730570800387407 * h[3] / H_AVG[3]
            - 0.009143704420210215 * h[12] / H_AVG[12]
            + 0.003963048570649325 * h[0] / H_AVG[0]
        ),
    ]]


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
