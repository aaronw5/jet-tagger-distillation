"""JEDI-linear jet tagger, 8 particles, 3 features: the formula simplified by hand with the training data (main result), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  17.1%   (on for 94% of jets)
  neuron  9:  12.7%   (on for 70% of jets)
  neuron 10:   9.6%   (on for 98% of jets)
  neuron  6:   8.9%   (on for 45% of jets)
  neuron  4:   7.7%   (on for 91% of jets)
  neuron  5:   7.1%   (on for 88% of jets)
  neuron  7:   6.7%   (on for 57% of jets)
  neuron  2:   6.2%   (on for 94% of jets)
  neuron 11:   5.8%   (on for 82% of jets)
  neuron  3:   3.9%   (on for 21% of jets)
  neuron 15:   3.5%   (on for 35% of jets)
  neuron  0:   3.4%   (on for 48% of jets)
  neuron 14:   3.4%   (on for 29% of jets)
  neuron  8:   2.2%   (on for 64% of jets)
  neuron  1:   1.7%   (on for 46% of jets)
  neuron 12:   0.2%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

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
    # scale S = 15.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.74708530888524 * (-0.03048183143639678
        - 0.16139324199206764 * max(0.0, 22.5 - Q.mass) / 4.680429373713902   # -16.1%  mass < 22.5
        - 0.14231528049108846 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771) / 0.5560920254717372   # -14.2%  mass < 29.3 and phi_1 > -0.0771
        + 0.13897215042557537 * max(0.0, 0.0121 - Q.girth2) / 0.00727045285153079   # +13.9%  girth2 < 0.0121
        - 0.10449739735001899 * max(0.0, 0.0775 - Q.girth) / 0.028970588567380766   # -10.4%  girth < 0.0775
        + 0.08456265185001038 * max(0.0, 0.0338 - Q.centroid_offset) / 0.018755144966586982   # +8.5%  centroid_offset < 0.0338
        - 0.08244442469945673 * max(0.0, 0.00449 - Q.width) / 0.0016248553053620906   # -8.2%  width < 0.00449
        - 0.0557792862273086 * max(0.0, 71.3 - Q.mass) / 33.14570485623231   # -5.6%  mass < 71.3
        + 0.05143413315294425 * max(0.0, Q.z_dr_0_0p05 - 0.852) / 0.0525933560096088   # +5.1%  z_dr_0_0p05 > 0.852
        + 0.04249190670898627 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961) / 0.0001711313759293094   # +4.2%  girth2 < 0.0197 and eccentricity > 0.961
        - 0.032436666989784646 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7) / 232.17407373820026   # -3.2%  mass < 64.9 and pt_7 < 40.1
        + 0.030076868971849073 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106) / 0.18573451824465523   # +3.0%  mass < 64.8 and centroid_offset > 0.0106
        - 0.023486825068869597 * max(0.0, Q.sum_pt - 895.0) / 13.400327463563551   # -2.3%  sum_pt > 895
        + 0.013113378904393663 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7) / 117.32812289501311   # +1.3%  sum_pt > 871 and pt_7 < 27.8
        - 0.010744325319444222 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2) / 3.26625110794795   # -1.1%  planar_flow < 0.148 and sum_pt_top2 < 388
        - 0.010376050687834988 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2) / 7.931677444206549e-05   # -1.0%  lam1 < 0.00645 and D2 < 0.886
        - 0.007343938123027911 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266) / 0.2249914788453211   # -0.7%  sum_pt_top5 > 655 and z_7 > 0.0266
        - 0.00523868920147037 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2) / 0.028446236469755106   # -0.5%  mass < 30 and D2 < 0.91
        + 0.0032927838358689926 * max(0.0, Q.eccentricity - 0.997) / 0.00010391131857143768   # +0.3%  eccentricity > 0.997
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 11.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.9845052712344 * (0.100129289682093
        - 0.25801645463954453 * max(0.0, 0.009 - Q.width) / 0.004706544232409183   # -25.8%  width < 0.009
        + 0.15542744303896278 * max(0.0, 0.00775 - Q.e2_sq) / 0.003963236192329647   # +15.5%  e2_sq < 0.00775
        + 0.0769255229281821 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2) / 8.950624611895739e-05   # +7.7%  log_sum_pt > 6.59 and lam2 < 0.0012
        + 0.0710343359181735 * max(0.0, Q.log_sum_pt - 6.41) / 0.18587584568776974   # +7.1%  log_sum_pt > 6.41
        + 0.0658650616206135 * max(0.0, Q.pt_7 - 30.4) / 6.8640015494123565   # +6.6%  pt_7 > 30.4
        - 0.05744508858631793 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass) / 444.1619141737125   # -5.7%  pt_7 > 29.2 and mass < 97
        - 0.05295676353452302 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset) / 0.0027006834541574674   # -5.3%  log_sum_pt > 6.4 and centroid_offset < 0.0235
        - 0.050818832026603804 * max(0.0, 0.0491 - Q.C2) / 0.025697829548557337   # -5.1%  C2 < 0.0491
        + 0.03693461566431115 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr) / 0.02329700505894493   # +3.7%  log_sum_pt > 6.33 and max_dr < 0.191
        - 0.03371014131381108 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow) / 4.8152487040455955e-05   # -3.4%  e2_sq < 0.00778 and planar_flow < 0.0847
        - 0.03215777690910079 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3) / 0.0003535734375030259   # -3.2%  log_sum_pt > 6.6 and girth2_top3 < 0.0063
        - 0.03065813374259036 * max(0.0, 0.0562 - Q.z_7) / 0.010902746749088529   # -3.1%  z_7 < 0.0562
        - 0.025755426084541594 * max(0.0, 0.0455 - Q.max_dr) / 0.004876240753129174   # -2.6%  max_dr < 0.0455
        + 0.0205783230989308 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979) / 4.1103503608716804e-05   # +2.1%  e2 < 0.0384 and eccentricity > 0.979
        - 0.015179689901474733 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32) / 0.002199771144376391   # -1.5%  e2 > 0.031 and tau32 < 0.629
        + 0.008495997790449303 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021) / 7.120302818464021e-06   # +0.8%  lam1 < 0.0083 and centroid_offset > 0.021
        - 0.004293382066512168 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01) / 0.0072984482280196205   # -0.4%  log_sum_pt > 6.59 and n_pt_above_50 > 7.01
        + 0.003747011135356878 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144) / 0.029350375622914048   # +0.4%  pt_7 > 32.9 and centroid_offset > 0.0144
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 6.701;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.700742982996225 * (0.5044216751152928
        + 0.18061754675662078 * max(0.0, Q.pt_7 - 30.2) / 6.995790514655604   # +18.1%  pt_7 > 30.2
        - 0.15519532964332655 * max(0.0, Q.LHA - 0.116) / 0.1314695342732372   # -15.5%  LHA > 0.116
        - 0.14898486920681575 * max(0.0, 54.2 - Q.pt_7) / 19.847103715908172   # -14.9%  pt_7 < 54.2
        + 0.09398953129686156 * max(0.0, 788.0 - Q.sum_pt) / 111.86495422958245   # +9.4%  sum_pt < 788
        - 0.08735501443760492 * max(0.0, Q.z_7 - 0.045) / 0.012808391685389825   # -8.7%  z_7 > 0.045
        + 0.08428633715703313 * max(0.0, 0.0053 - Q.lam1) / 0.0021152849523125234   # +8.4%  lam1 < 0.0053
        - 0.0651974101356716 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2) / 0.22872831857388926   # -6.5%  pt_7 > 29.6 and C2 < 0.0514
        + 0.049795233874856674 * max(0.0, 6.46 - Q.log_sum_pt) / 0.06865536295753123   # +5.0%  log_sum_pt < 6.46
        - 0.047317582824663224 * max(0.0, 0.654 - Q.planar_flow) / 0.4008381303220035   # -4.7%  planar_flow < 0.654
        - 0.03087775625254124 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755) / 0.36685090166646017   # -3.1%  pt_7 > 29.2 and max_dr > 0.0755
        + 0.0156974090127012 * max(0.0, Q.log_sum_pt - 6.83) / 0.009226693271322741   # +1.6%  log_sum_pt > 6.83
        - 0.01567144343764393 * max(0.0, 0.0185 - Q.z_7) / 0.0003465686952086297   # -1.6%  z_7 < 0.0185
        - 0.009162986327893263 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset) / 1.9127357115240224e-06   # -0.9%  lam1 < 0.00374 and centroid_offset < 0.00583
        - 0.008887451082549045 * max(0.0, Q.log_sum_pt - 6.89) / 0.0043468996699352344   # -0.9%  log_sum_pt > 6.89
        - 0.006964098553217011 * max(0.0, 4.87e-05 - Q.girth2) / 8.625625603209437e-07   # -0.7%  girth2 < 4.87e-05
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 14.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.9075063618068 * (0.19453287019416157
        - 0.2652510074175492 * max(0.0, 0.00905 - Q.girth2) / 0.0047469760870981715   # -26.5%  girth2 < 0.00905
        + 0.23131065158609027 * max(0.0, 0.0433 - Q.e2) / 0.01926405033560512   # +23.1%  e2 < 0.0433
        - 0.2275776855041056 * max(0.0, 0.0126 - Q.girth2) / 0.007693006336638823   # -22.8%  girth2 < 0.0126
        - 0.08425273535697961 * max(0.0, Q.e2 - 0.0278) / 0.009968239589950778   # -8.4%  e2 > 0.0278
        + 0.07007429744980068 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05) / 0.0967252810214857   # +7.0%  mass_over_sum_pt > 0.0616 and n_dr_0_0p05 < 5.99
        + 0.03036996019946953 * max(0.0, Q.centroid_offset - 0.0103) / 0.009183374744044855   # +3.0%  centroid_offset > 0.0103
        - 0.02214954315771483 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32) / 0.0024826650792086325   # -2.2%  mass_over_sum_pt > 0.0749 and tau32 < 0.529
        + 0.015746604543017306 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953) / 0.00021536019027695996   # +1.6%  LHA > 0.316 and eccentricity > 0.953
        - 0.011309608763423521 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951) / 1.3933724346304303e-05   # -1.1%  lam1 > 0.0152 and eccentricity > 0.951
        - 0.011028524995636538 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595) / 0.0005012433126030397   # -1.1%  e2 < 0.0435 and z_dr_0p1_0p2 > 0.000595
        - 0.00958866995517093 * max(0.0, Q.mass - 71.9) / 2.0867614358828663   # -1.0%  mass > 71.9
        - 0.009120351340042175 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr) / 8.444825815129932e-05   # -0.9%  LHA > 0.308 and max_dr < 0.156
        - 0.005030884332182068 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7) / 9.959885815075475e-06   # -0.5%  mass_over_sum_pt > 0.0682 and dr_7 < 0.0419
        + 0.004987406440678885 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr) / 0.03609213264239645   # +0.5%  mass > 62 and max_dr < 0.176
        + 0.0022020689581387734 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr) / 3.545070950603783e-06   # +0.2%  mass_over_sum_pt > 0.0909 and max_dr < 0.148
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 11.43;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.42889240188076 * (0.20386927429788132
        + 0.19895039876844267 * max(0.0, 0.111 - Q.mass_over_sum_pt) / 0.054658238000860615   # +19.9%  mass_over_sum_pt < 0.111
        + 0.16772484188598238 * max(0.0, Q.width - 0.00219) / 0.004877631478466554   # +16.8%  width > 0.00219
        - 0.1508333594466353 * max(0.0, 0.000389 - Q.lam2) / 0.00026643867631063366   # -15.1%  lam2 < 0.000389
        - 0.1323391758197355 * max(0.0, Q.mass_over_sum_pt - 0.0885) / 0.008692472419524925   # -13.2%  mass_over_sum_pt > 0.0885
        + 0.10272615735434261 * max(0.0, 0.279 - Q.tau21) / 0.07574491608138409   # +10.3%  tau21 < 0.279
        - 0.10172651328773372 * max(0.0, Q.C2 - 0.00878) / 0.020254727783693416   # -10.2%  C2 > 0.00878
        - 0.04611869201242366 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass) / 0.8654935447052015   # -4.6%  tau21 < 0.285 and mass < 66.2
        + 0.040777551903722106 * max(0.0, 486.0 - Q.sum_pt_top5) / 27.906721743697478   # +4.1%  sum_pt_top5 < 486
        + 0.018199363654531422 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2) / 0.31044562535871495   # +1.8%  tau21 < 0.246 and pt_7 > 34.2
        + 0.018103436118454188 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7) / 0.018809293045648615   # +1.8%  C2 > 0.0157 and pt_7 > 39.7
        - 0.011614377957663527 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2) / 0.05648488340123917   # -1.2%  max_dr > 0.109 and pt_7 > 39.2
        + 0.01088613179033288 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325) / 0.002278689174073407   # +1.1%  tau21 < 0.243 and planar_flow > 0.0325
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 9.942;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.942407371905428 * (0.04676935701849887
        + 0.20361704363321484 * max(0.0, 0.0683 - Q.z_7) / 0.018920033604340792   # +20.4%  z_7 < 0.0683
        + 0.15927245657344208 * max(0.0, 0.0374 - Q.e2) / 0.014939166475257337   # +15.9%  e2 < 0.0374
        - 0.13005102951510084 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset) / 0.00038368555328629946   # -13.0%  z_7 < 0.0707 and centroid_offset < 0.0301
        + 0.09301795691305442 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset) / 1.2298170485796993e-05   # +9.3%  width < 0.00267 and centroid_offset < 0.0234
        - 0.0768747712743767 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1) / 0.0007783302368960353   # -7.7%  log_sum_pt > 6.59 and lam1 < 0.0124
        - 0.07285063803022432 * max(0.0, 0.0238 - Q.dr_0) / 0.004285862252069327   # -7.3%  dr_0 < 0.0238
        + 0.04747539279632977 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0) / 0.0006235398881404689   # +4.7%  log_sum_pt > 6.6 and dr_0 < 0.0225
        - 0.044667433849120725 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3) / 0.24810157742212063   # -4.5%  sum_pt_top2 < 546 and girth2_top3 < 0.00383
        + 0.04163224604079111 * max(0.0, 0.0333 - Q.z_7) / 0.0022374310807834713   # +4.2%  z_7 < 0.0333
        - 0.03765707393917616 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt) / 0.004118833548264608   # -3.8%  LHA < 0.221 and log_sum_pt < 6.79
        + 0.0337093436560428 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2) / 4.892730319239904e-07   # +3.4%  e2 < 0.0338 and lam2 < 7.09e-05
        - 0.02968642065729772 * max(0.0, Q.log_sum_pt - 6.89) / 0.0043468996699352344   # -3.0%  log_sum_pt > 6.89
        - 0.023758405399680445 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt) / 0.8496969244262611   # -2.4%  z_7 < 0.0684 and sum_pt < 803
        + 0.005729787722147945 * max(0.0, 6.4e-05 - Q.girth2) / 1.8085042440678394e-06   # +0.6%  girth2 < 6.4e-05
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 28.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.092534342111662 * (0.23351399770886236
        - 0.2631819784504917 * max(0.0, 0.00862 - Q.girth2) / 0.004400862361812696   # -26.3%  girth2 < 0.00862
        - 0.15932259108960278 * max(0.0, 0.0133 - Q.width) / 0.008288472891960887   # -15.9%  width < 0.0133
        + 0.1314251928774708 * max(0.0, 0.00802 - Q.lam1) / 0.00401748285563547   # +13.1%  lam1 < 0.00802
        - 0.09921969391589756 * max(0.0, Q.mass_over_sum_pt - 0.00941) / 0.05239347102906316   # -9.9%  mass_over_sum_pt > 0.00941
        + 0.07767290618864654 * max(0.0, 0.0508 - Q.e2) / 0.02546124602749324   # +7.8%  e2 < 0.0508
        + 0.050857851551879614 * max(0.0, Q.max_dr - 0.0279) / 0.09719224090389081   # +5.1%  max_dr > 0.0279
        + 0.03561840550924232 * max(0.0, Q.girth - 0.0868) / 0.00794135936491785   # +3.6%  girth > 0.0868
        - 0.03200036757580522 * max(0.0, 0.000518 - Q.lam2) / 0.0003745714271181269   # -3.2%  lam2 < 0.000518
        + 0.02644411030378785 * max(0.0, 0.00328 - Q.e2_sq) / 0.0011867125828366597   # +2.6%  e2_sq < 0.00328
        + 0.02281966876794754 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1) / 11.129554314916305   # +2.3%  mass < 46.6 and z_dr_0p05_0p1 < 0.8
        + 0.019701827562800433 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2) / 2.795324582880471e-05   # +2.0%  centroid_offset > 0.00781 and lam2 < 0.00351
        + 0.01473362742090931 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7) / 1.2067199250897167   # +1.5%  log_sum_pt < 6.57 and pt_7 < 45.4
        - 0.014223292513475786 * max(0.0, Q.C2 - 0.0101) / 0.01921001602849607   # -1.4%  C2 > 0.0101
        - 0.01366681138555802 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32) / 0.005588578867481097   # -1.4%  mass_over_sum_pt > 0.0171 and tau32 < 0.53
        - 0.00977746380129061 * max(0.0, Q.centroid_offset - 0.0212) / 0.0046241369969109445   # -1.0%  centroid_offset > 0.0212
        - 0.007113224480357742 * max(0.0, Q.lam2 - 0.0029) / 0.00018332890183265986   # -0.7%  lam2 > 0.0029
        + 0.0067787536169291104 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6) / 0.07438764405480261   # +0.7%  C2 > 0.0103 and pt_7 > 32.6
        + 0.006550051646371875 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7) / 0.00022089741994994828   # +0.7%  log_sum_pt < 6.71 and z_7 < 0.0492
        + 0.005739375950486558 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159) / 0.00021048774933594437   # +0.6%  e2 < 0.0502 and z_dr_0p1_0p2 > 0.159
        + 0.00315280539104861 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7) / 8.125715020369529e-05   # +0.3%  log_sum_pt < 6.31 and z_7 < 0.0712
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 16.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.975760482979307 * (0.2792216587146447
        - 0.16222713794108157 * max(0.0, Q.girth2 - 0.00445) / 0.0035999072385974455   # -16.2%  girth2 > 0.00445
        + 0.14399444696650063 * max(0.0, Q.mass_over_sum_pt - 0.0729) / 0.013357460341980223   # +14.4%  mass_over_sum_pt > 0.0729
        - 0.142393976556704 * max(0.0, 0.00564 - Q.width) / 0.0022591084488276427   # -14.2%  width < 0.00564
        - 0.0910909016743002 * max(0.0, 0.089 - Q.girth) / 0.03780775865529435   # -9.1%  girth < 0.089
        - 0.09003608336870672 * max(0.0, Q.mass_over_sum_pt - 0.0906) / 0.008261789114014715   # -9.0%  mass_over_sum_pt > 0.0906
        + 0.07115951602017392 * max(0.0, 0.0372 - Q.e2) / 0.014803761029941197   # +7.1%  e2 < 0.0372
        + 0.047369107523168606 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945) / 7.586100222696259e-05   # +4.7%  girth2 > 0.0044 and eccentricity > 0.945
        - 0.03407844070541011 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768) / 0.00015304429816096678   # -3.4%  planar_flow < 0.205 and width > 0.00768
        - 0.030509461030906224 * max(0.0, 0.184 - Q.planar_flow) / 0.06970677024568708   # -3.1%  planar_flow < 0.184
        + 0.026068851198484257 * max(0.0, 0.0253 - Q.e2) / 0.007832541132953926   # +2.6%  e2 < 0.0253
        + 0.02526856784595673 * max(0.0, 0.00115 - Q.e2_sq) / 0.00032496451174308624   # +2.5%  e2_sq < 0.00115
        + 0.022731252909844533 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0) / 9.481088571880768   # +2.3%  planar_flow < 0.179 and sum_pt > 607
        - 0.02070425964286381 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow) / 3.688043574760449   # -2.1%  pt_7 < 44.5 and planar_flow < 0.596
        + 0.01846104830242108 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2) / 0.0026115861187217663   # +1.8%  e2 < 0.0513 and D2 < 1.1
        - 0.01788450806689296 * max(0.0, 0.000708 - Q.girth2) / 0.00013257778397357474   # -1.8%  girth2 < 0.000708
        - 0.01757232511082988 * max(0.0, 0.0201 - Q.centroid_offset) / 0.007870806918482609   # -1.8%  centroid_offset < 0.0201
        - 0.014754264040663902 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -1.5%  mass > 80.4
        - 0.014047092436589567 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2) / 0.0003162600486549266   # -1.4%  lam1 < 0.00797 and D2 < 1.06
        - 0.0049832592164316455 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928) / 0.0379348048801537   # -0.5%  mass > 80.4 and eccentricity > 0.928
        + 0.0033675251900049928 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245) / 5.196936458720377e-05   # +0.3%  centroid_offset < 0.0228 and C2 > 0.0245
        - 0.0012979742520645072 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0) / 0.0024813175693830275   # -0.1%  centroid_offset > 0.0311 and pt_0 > 375
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 6.439;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.439391780736883 * (0.01615059364940502
        + 0.3798903459704536 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4) / 0.00018963277297856046   # +38.0%  width < 0.00506 and z_dr_0p2_0p4 < 0.0996
        - 0.19686992643043483 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2) / 7.501317077694774e-06   # -19.7%  LHA < 0.198 and lam2 < 0.000321
        - 0.15922684270614199 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598) / 1.336797942887373e-05   # -15.9%  width < 0.00544 and centroid_offset > 0.00598
        + 0.07845328801831668 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow) / 0.000407412466158816   # +7.8%  girth2 < 0.00621 and planar_flow < 0.453
        - 0.05888416739672968 * max(0.0, Q.log_sum_pt - 6.71) / 0.03268777789224447   # -5.9%  log_sum_pt > 6.71
        - 0.029507069947115005 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163) / 0.01250049892704478   # -3.0%  mass < 20.2 and centroid_offset > 0.0163
        + 0.029169320756196434 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0) / 0.33481761912144214   # +2.9%  e2 < 0.0218 and sum_pt > 830
        - 0.02551576749372733 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119) / 5.0246490360336735e-06   # -2.6%  girth < 0.0603 and lam1 > 0.00119
        - 0.02195159527541628 * max(0.0, Q.pt_7 - 37.1) / 3.318190661750662   # -2.2%  pt_7 > 37.1
        + 0.020531676005468163 * max(0.0, 0.00366 - Q.centroid_offset) / 0.00027717296795464223   # +2.1%  centroid_offset < 0.00366
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 14.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.092208329027422 * (-0.1575338618452864
        + 0.318413007357865 * max(0.0, 0.00662 - Q.width) / 0.0028949306028123708   # +31.8%  width < 0.00662
        + 0.14322971414785984 * max(0.0, 0.257 - Q.max_dr) / 0.13730768508018282   # +14.3%  max_dr < 0.257
        + 0.09440816389592124 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset) / 0.2363089722171025   # +9.4%  mass < 51.1 and centroid_offset < 0.0241
        - 0.08188342432536268 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285) / 2.6587978670211116e-05   # -8.2%  width < 0.00676 and centroid_offset > 0.00285
        - 0.07732532235562262 * max(0.0, 31.3 - Q.mass) / 8.012386409887025   # -7.7%  mass < 31.3
        + 0.07348424671876118 * max(0.0, 0.0174 - Q.centroid_offset) / 0.006091501845073131   # +7.3%  centroid_offset < 0.0174
        - 0.07281137468664553 * max(0.0, Q.log_sum_pt - 6.36) / 0.22305936104501795   # -7.3%  log_sum_pt > 6.36
        + 0.0559421875778268 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt) / 3.790139239078198   # +5.6%  mass < 49.9 and log_sum_pt < 6.79
        - 0.034911401716712816 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031) / 0.00034646390566921305   # -3.5%  centroid_offset < 0.019 and z_4 > 0.031
        + 0.03090998982669157 * max(0.0, Q.lam2 - 0.00136) / 0.00029039334405723725   # +3.1%  lam2 > 0.00136
        + 0.013344212928223641 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857142857143   # +1.3%  n_dr_0p2_0p4 > 1
        + 0.00333695446250705 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7) / 0.0007916676341772235   # +0.3%  girth2 > 0.0184 and pt_7 < 29.8
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.46;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.459666025181411 * (0.2319138677469047
        - 0.20826942692514142 * max(0.0, 0.0467 - Q.e2) / 0.0220059542226272   # -20.8%  e2 < 0.0467
        + 0.18173884493492978 * max(0.0, 0.0966 - Q.mass_over_sum_pt) / 0.04249878015726673   # +18.2%  mass_over_sum_pt < 0.0966
        + 0.0995911172969425 * max(0.0, Q.mass - 9.7) / 31.479511614824524   # +10.0%  mass > 9.7
        - 0.08230404820623699 * max(0.0, 0.00182 - Q.girth2) / 0.0004872704064594914   # -8.2%  girth2 < 0.00182
        + 0.07518752105174371 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4) / 0.0021655358930783965   # +7.5%  eccentricity > 0.901 and z_dr_0p2_0p4 < 0.0539
        + 0.0710910505907584 * max(0.0, 2.95 - Q.n_dr_0p05_0p1) / 1.6367762184763373   # +7.1%  n_dr_0p05_0p1 < 2.95
        + 0.04800825735301537 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69) / 9.81165935080697e-05   # +4.8%  lam1 < 0.004 and log_sum_pt > 6.69
        - 0.046771983061159636 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21) / 0.008228853136124835   # -4.7%  LHA > 0.301 and tau21 < 0.699
        - 0.035835940754271836 * max(0.0, Q.log_sum_pt - 6.69) / 0.038686562912453766   # -3.6%  log_sum_pt > 6.69
        + 0.033675461895835646 * max(0.0, Q.e2 - 0.0458) / 0.0040980048774332195   # +3.4%  e2 > 0.0458
        + 0.0302114798118277 * max(0.0, Q.lam2 - 0.000227) / 0.000438458267553979   # +3.0%  lam2 > 0.000227
        + 0.018435117931430017 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21) / 5.851907357587282e-05   # +1.8%  lam2 > 8.42e-05 and tau21 < 0.518
        + 0.01818424916777848 * max(0.0, 0.271 - Q.tau32) / 0.011210613695066983   # +1.8%  tau32 < 0.271
        - 0.01599384324301137 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0) / 0.012996593578645882   # -1.6%  lam1 < 0.00395 and sum_pt > 989
        + 0.01378448542808517 * max(0.0, Q.C2 - 0.0596) / 0.003685579126254308   # +1.4%  C2 > 0.0596
        - 0.008146914466151811 * max(0.0, 6.24 - Q.log_sum_pt) / 0.01917137572656518   # -0.8%  log_sum_pt < 6.24
        + 0.007006531881659888 * max(0.0, Q.n_dr_0p2_0p4 - 1.56) / 0.10070594957990106   # +0.7%  n_dr_0p2_0p4 > 1.56
        - 0.005763726000020303 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.010461185163213187   # -0.6%  tau32 < 0.29 and n_dr_0p2_0p4 < 2
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 17.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.674560020875607 * (-0.06506526887468332
        + 0.2703036628343068 * max(0.0, 0.0087 - Q.width) / 0.004464951694044372   # +27.0%  width < 0.0087
        + 0.1910180125525075 * max(0.0, 0.0498 - Q.centroid_offset) / 0.033427320078491723   # +19.1%  centroid_offset < 0.0498
        - 0.17888624796677116 * max(0.0, 0.0883 - Q.girth) / 0.03724070349349745   # -17.9%  girth < 0.0883
        + 0.06109410114733056 * max(0.0, 0.271 - Q.planar_flow) / 0.12024625363585069   # +6.1%  planar_flow < 0.271
        - 0.048121006827578 * max(0.0, Q.girth - 0.0766) / 0.01043579906060113   # -4.8%  girth > 0.0766
        + 0.04055775821431442 * max(0.0, 688.0 - Q.sum_pt_top5) / 131.53037282037815   # +4.1%  sum_pt_top5 < 688
        - 0.03955694742581224 * max(0.0, 0.0437 - Q.centroid_offset) * max(0.0, 6.84 - Q.log_sum_pt) / 0.006922293480397418   # -4.0%  centroid_offset < 0.0437 and log_sum_pt < 6.84
        - 0.03558054435566619 * max(0.0, 0.00365 - Q.width) / 0.0012234833984234414   # -3.6%  width < 0.00365
        - 0.029373577541203923 * max(0.0, Q.centroid_offset - 0.0144) / 0.007102121193978831   # -2.9%  centroid_offset > 0.0144
        + 0.02175409198202475 * max(0.0, Q.girth - 0.0771) * max(0.0, 7.47 - Q.n_pt_above_50) / 0.030515397177455973   # +2.2%  girth > 0.0771 and n_pt_above_50 < 7.47
        - 0.01872957099420022 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053) / 0.00025269994404751345   # -1.9%  planar_flow < 0.217 and width > 0.0053
        - 0.017974439993037737 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114) / 0.00548687942139158   # -1.8%  planar_flow < 0.281 and max_dr > 0.114
        - 0.017017751772194903 * max(0.0, 0.157 - Q.LHA) / 0.013192161189386887   # -1.7%  LHA < 0.157
        - 0.016030007883482005 * max(0.0, 0.0355 - Q.C2) / 0.01483368253778596   # -1.6%  C2 < 0.0355
        - 0.014002278509569291 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass) / 1.9334696245812428   # -1.4%  planar_flow < 0.254 and mass < 66.7
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5787;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5787296560751424 * (-1.5724094841993814
        - 0.40429583580883305 * max(0.0, Q.e2 - 0.0622) / 0.0018869192742762757   # -40.4%  e2 > 0.0622
        + 0.29014481308073464 * max(0.0, Q.girth2 - 0.0188) / 0.0007632518540281819   # +29.0%  girth2 > 0.0188
        + 0.14975121107414016 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1) / 0.012523911399100173   # +15.0%  girth2 > 0.0197 and pt_7 > 16.1
        + 0.07844147017383628 * max(0.0, Q.mass - 91.2) / 0.5835013503307559   # +7.8%  mass > 91.2
        + 0.07736666986245591 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489) / 2.9264304732796208e-06   # +7.7%  girth2 > 0.0186 and lam2 > 0.000489
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 14.77;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.773036248725724 * (0.11372069842074012
        + 0.4177687128477889 * max(0.0, 0.15 - Q.girth) / 0.09211510952961005   # +41.8%  girth < 0.15
        + 0.15480071670572298 * max(0.0, 0.0159 - Q.width) / 0.01053860183973441   # +15.5%  width < 0.0159
        - 0.15321423534838904 * max(0.0, 0.0509 - Q.e2) / 0.02554672068422738   # -15.3%  e2 < 0.0509
        - 0.0862915607985988 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt) / 0.03101674831233138   # -8.6%  girth < 0.156 and log_sum_pt < 6.91
        - 0.0536000067418882 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7) / 0.7264539839723464   # -5.4%  girth < 0.148 and pt_7 < 39.5
        - 0.041641883615606196 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382) / 0.041287050746212625   # -4.2%  tau21 < 0.518 and max_dr > -0.0382
        + 0.03685545669734586 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7) / 916.6111073286729   # +3.7%  sum_pt_top5 > 653 and pt_7 < 42.6
        - 0.019650147461338467 * max(0.0, 24.8 - Q.pt_7) / 1.1800501656063371   # -2.0%  pt_7 < 24.8
        + 0.012106710596559028 * max(0.0, Q.pt_0 - 179.0) / 81.29676113445379   # +1.2%  pt_0 > 179
        - 0.010441746215157454 * max(0.0, Q.C2 - 0.0664) / 0.002943822430086177   # -1.0%  C2 > 0.0664
        - 0.0064678750073278325 * max(0.0, Q.sum_pt - 998.0) / 3.9647365948332456   # -0.6%  sum_pt > 998
        - 0.0051440062193165706 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289) / 0.1342625271053609   # -0.5%  sum_pt_top5 > 677 and z_7 > 0.0289
        + 0.001667819597073706 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01) / 1.2967768086265785   # +0.2%  sum_pt > 1070 and n_pt_above_50 > 6.01
        + 0.0003491221478870104 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4) / 0.0007971552003041404   # +0.0%  sum_pt < 764 and z_4 < 0.0379
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 21.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.306340032807594 * (-0.07087092375672666
        + 0.24079667304737443 * max(0.0, 0.0131 - Q.girth2) / 0.008117873092905308   # +24.1%  girth2 < 0.0131
        - 0.1690261925217487 * max(0.0, 0.00741 - Q.width) / 0.0034628168579992095   # -16.9%  width < 0.00741
        - 0.13577920755976158 * max(0.0, 0.0869 - Q.girth) / 0.03611682853499925   # -13.6%  girth < 0.0869
        + 0.07809081974583945 * max(0.0, 0.0385 - Q.e2) / 0.015696505273071138   # +7.8%  e2 < 0.0385
        - 0.06621985388319972 * max(0.0, 0.177 - Q.max_dr) / 0.07019416536110883   # -6.6%  max_dr < 0.177
        - 0.05714777429596832 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2) / 0.0158131157307538   # -5.7%  z_dr_0p05_0p1 < 0.597 and C2 < 0.0662
        - 0.03380097247315214 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4) / 0.0014818415904373436   # -3.4%  girth2 < 0.00453 and n_dr_0p2_0p4 < 0.93
        + 0.03278837655420493 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971) / 6.0224163792036755e-05   # +3.3%  girth2 < 0.0135 and eccentricity > 0.971
        + 0.03128717659111041 * max(0.0, 0.572 - Q.z_dr_0p05_0p1) / 0.36033255303605205   # +3.1%  z_dr_0p05_0p1 < 0.572
        + 0.030586170361501363 * max(0.0, Q.mass - 5.61) / 34.8491628889588   # +3.1%  mass > 5.61
        - 0.01998633121727268 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow) / 6.1804872122428e-05   # -2.0%  width < 0.00758 and planar_flow < 0.109
        - 0.01727660299057313 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2) / 0.00037332776666224303   # -1.7%  width < 0.00857 and D2 < 1.04
        + 0.016567405151232957 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946) / 0.00027794548630980903   # +1.7%  planar_flow < 0.11 and centroid_offset > 0.00946
        + 0.014526296935663482 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt) / 0.0008123417898393101   # +1.5%  width < 0.00793 and log_sum_pt < 6.82
        - 0.010742281080537985 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr) / 0.0012239502322456584   # -1.1%  planar_flow < 0.116 and max_dr < 0.164
        - 0.009839389789648673 * max(0.0, Q.lam1 - 0.00732) / 0.0020756572730058396   # -1.0%  lam1 > 0.00732
        - 0.009754459699004252 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186) / 0.00015865025586511648   # -1.0%  planar_flow < 0.116 and centroid_offset > 0.0186
        + 0.006196769762118946 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset) / 0.0034293632118630258   # +0.6%  D2 < 1.04 and centroid_offset < 0.0307
        + 0.005106754986248424 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr) / 1.5884125284853224e-05   # +0.5%  lam1 > 0.0055 and max_dr < 0.168
        - 0.004618358815203394 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2) / 2.0246980104987227   # -0.5%  mass < 75.3 and D2 < 0.865
        - 0.004257919965054074 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt) / 1.7893627338840692   # -0.4%  planar_flow < 0.106 and sum_pt < 740
        + 0.0037963447332826196 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2) / 0.00042128235296394657   # +0.4%  e2 < 0.0382 and D2 < 0.961
        + 0.0018078678402982816 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr) / 8.957917892970392e-07   # +0.2%  lam1 > 0.00727 and max_dr < 0.134
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 22.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.61002267119432 * (0.0420609927654604
        - 0.2863128559001961 * max(0.0, 0.00695 - Q.width) / 0.003127314088385423   # -28.6%  width < 0.00695
        + 0.2200700273183981 * max(0.0, 0.00679 - Q.lam1) / 0.003090551743428158   # +22.0%  lam1 < 0.00679
        - 0.1434077695228293 * max(0.0, 0.00813 - Q.lam1) / 0.004104370784983013   # -14.3%  lam1 < 0.00813
        + 0.08050688014102962 * max(0.0, 0.041 - Q.e2) / 0.017502522934382727   # +8.1%  e2 < 0.041
        + 0.06497808983800582 * max(0.0, 0.0244 - Q.e2) / 0.0073826938912975515   # +6.5%  e2 < 0.0244
        + 0.03835583156557037 * max(0.0, 0.342 - Q.z_dr_0p1_0p2) / 0.24022887015791047   # +3.8%  z_dr_0p1_0p2 < 0.342
        + 0.029631214637645245 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4) / 0.010771100236851832   # +3.0%  tau21 < 0.24 and z_dr_0p2_0p4 < 0.21
        - 0.023284939721941424 * max(0.0, Q.LHA - 0.342) / 0.009607171806760743   # -2.3%  LHA > 0.342
        - 0.021793505203651826 * max(0.0, 37.4 - Q.mass) / 10.64258416283713   # -2.2%  mass < 37.4
        - 0.020434998509137475 * max(0.0, 0.00198 - Q.girth2_top3) / 0.0006947906459810832   # -2.0%  girth2_top3 < 0.00198
        + 0.013716593134705253 * max(0.0, 2.12 - Q.n_dr_0_0p05) / 0.683111193275846   # +1.4%  n_dr_0_0p05 < 2.12
        + 0.0085981399839176 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0) / 6.250936976414125   # +0.9%  LHA > 0.186 and sum_pt_top3 > 338
        - 0.007757777688559737 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243) / 4.8995399278329925e-06   # -0.8%  width < 0.00788 and e2 > 0.0243
        - 0.007515994537102279 * max(0.0, 0.711 - Q.D2) / 0.08582667014164402   # -0.8%  D2 < 0.711
        - 0.007198915951225651 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow) / 4.7592880954746897e-05   # -0.7%  width < 0.00868 and planar_flow < 0.0755
        - 0.0065081809532171 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9) / 1.911040505200957e-05   # -0.7%  width < 0.00614 and log_sum_pt > 6.9
        + 0.006400965301097275 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # +0.6%  log_sum_pt > 6.9
        - 0.005914812006721108 * max(0.0, Q.z_dr_0p05_0p1 - 0.757) / 0.02188773053483087   # -0.6%  z_dr_0p05_0p1 > 0.757
        - 0.005040058678605802 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1) / 0.012633685253594996   # -0.5%  tau21 < 0.26 and z_dr_0p05_0p1 < 0.513
        - 0.002572449406442925 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4) / 0.10168381014016592   # -0.3%  mass > 80.4 and z_dr_0p2_0p4 < 0.236
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0999546218487395, 0.5141785714285714, 2.4120813025210084, 0.5757481092436975, 3.9047726890756302, 2.3764600840336136, 1.610221218487395, 1.6310056722689075, 0.7225256302521008, 4.345371428571428, 3.2235485294117647, 2.5950478991596637, 0.07584348739495798, 5.369000630252101, 0.5028894957983193, 0.642822268907563]
T = [2.899749171152836, 2.270341432510504, 4.028210673253677, 2.8171246733849786, 4.69876515887605]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +36%, n9 +26%, n5 -15%, n1 +7%, n6 +6%, n0 -6% ...
            + 0.3574244265633997 * h[2] / H_AVG[2]
            + 0.25756045444054365 * h[9] / H_AVG[9]
            - 0.15366372725926275 * h[5] / H_AVG[5]
            + 0.06926495797029043 * h[1] / H_AVG[1]
            + 0.060735579312896464 * h[6] / H_AVG[6]
            - 0.059269922851822754 * h[0] / H_AVG[0]
            - 0.04208093160178438 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +49%, n10 -18%, n4 -16%, n6 +9%, n5 +5%, n8 +2% ...
            + 0.48596962046615405 * h[9] / H_AVG[9]
            - 0.17748148380083192 * h[10] / H_AVG[10]
            - 0.16124113948625068 * h[4] / H_AVG[4]
            + 0.08865523459542164 * h[6] / H_AVG[6]
            + 0.04906599723015902 * h[5] / H_AVG[5]
            + 0.019890335102954772 * h[8] / H_AVG[8]
            + 0.017696189318227935 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n6 -12%, n15 -11%, n0 +9%, n13 +9%, n14 -9% ...
            + 0.24158194323010526 * h[11] / H_AVG[11]
            - 0.12491753078318259 * h[6] / H_AVG[6]
            - 0.10971131991887215 * h[15] / H_AVG[15]
            + 0.0938653491415077 * h[0] / H_AVG[0]
            + 0.09371601622555634 * h[13] / H_AVG[13]
            - 0.09363142904938808 * h[14] / H_AVG[14]
            + 0.08857096109142727 * h[7] / H_AVG[7]
            - 0.07146449825309814 * h[3] / H_AVG[3]
            - 0.0448415989666015 * h[8] / H_AVG[8]
            - 0.03371046555347468 * h[9] / H_AVG[9]
            - 0.0039888877867860625 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +27%, n6 -21%, n3 -11%, n4 +11%, n13 +10%, n14 +7% ...
            + 0.271388027693289 * h[7] / H_AVG[7]
            - 0.21434371103186722 * h[6] / H_AVG[6]
            - 0.11496058889734571 * h[3] / H_AVG[3]
            + 0.10828784725650127 * h[4] / H_AVG[4]
            + 0.10422585295598916 * h[13] / H_AVG[13]
            + 0.06694185837993921 * h[14] / H_AVG[14]
            - 0.04820264378987964 * h[9] / H_AVG[9]
            - 0.035653721848284275 * h[15] / H_AVG[15]
            + 0.022814865822513842 * h[1] / H_AVG[1]
            + 0.01318088232439078 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -46%, n10 +26%, n5 -13%, n4 +10%, n8 +3%, n12 -1% ...
            - 0.46419781203997246 * h[13] / H_AVG[13]
            + 0.2572656129123434 * h[10] / H_AVG[10]
            - 0.12644067130831377 * h[5] / H_AVG[5]
            + 0.10387762946876175 * h[4] / H_AVG[4]
            + 0.02883173580538644 * h[8] / H_AVG[8]
            - 0.008070576505796241 * h[12] / H_AVG[12]
            + 0.007658236922046678 * h[3] / H_AVG[3]
            + 0.0036577250373793216 * h[0] / H_AVG[0]
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
