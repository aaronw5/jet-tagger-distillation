"""JEDI-linear jet tagger, 8 particles, 3 features: the tuned formula (start), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 88.1% of jets.

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
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_5                    pT of particle 5 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_0                  Δη of particle 0
  Q.phi_1                  Δφ of particle 1
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
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_4=z[4],
        z_5=z[5],
        z_6=z[6],
        z_7=z[7],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_0=eta[0],
        phi_1=phi[1],
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
    z = -0.6855505108833313
    if Q.planar_flow < 0.012569162668:
        z += -112.71226096153259 * Q.planar_flow + 1.8664792577496332
    if 0.012569162668 <= Q.planar_flow < 0.148419710734:
        z += -3.3108479976654053 * Q.planar_flow + 0.49139510209774256
    if Q.width < 0.004372139331:
        z += -106.3123779296875 * Q.width + 3.7169331622007755
    if 0.004372139331 <= Q.width < 0.008678044951:
        z += -755.2698364257812 * Q.width + 6.554265590637347
    if Q.girth2 < 0.013238675334:
        z += -472.3391799926758 * Q.girth2 + 6.594518053687365
    if 0.013238675334 <= Q.girth2 < 0.018827652745:
        z += -61.079689025878906 * Q.girth2 + 1.1499871747518353
    if Q.mass < 21.784077072144:
        z += 0.43330323975533247 * Q.mass - 11.722995584310008
    if 21.784077072144 <= Q.mass < 29.644699859619:
        z += 0.15169438999146223 * Q.mass - 5.588406696856039
    if 29.644699859619 <= Q.mass < 56.920347213745:
        z += 0.03437567036598921 * Q.mass - 2.1105284656440975
    if 56.920347213745 <= Q.mass < 64.618731689453:
        z += 0.019985150545835495 * Q.mass - 1.291415080894669
    if Q.girth2_top3 < 0.003952581551:
        z += -111.67242431640625 * Q.girth2_top3 - 0.07950934378510688
    if 0.003952581551 <= Q.girth2_top3 < 0.007929074034:
        z += 130.9957733154297 * Q.girth2_top3 - 1.0386751847591236
    if Q.lam1 < 0.00543336053:
        z += 194.51498413085938 * Q.lam1 - 1.5212619188718624
    if 0.00543336053 <= Q.lam1 < 0.006506575659:
        z += 3.3397064208984375 * Q.lam1 - 0.4825377106507718
    if 0.006506575659 <= Q.lam1 < 0.008375572068:
        z += 246.55352783203125 * Q.lam1 - 2.0650268409768215
    if Q.sum_pt >= 901.59375:
        z += -0.03148293495178223 * Q.sum_pt + 28.384817384183407
    if Q.girth2_top5 < 0.00832969537:
        z += 121.0089111328125 * Q.girth2_top5 - 1.0079673667917297
    if Q.z_dr_0_0p05 >= 0.847731333971:
        z += 16.126718521118164 * Q.z_dr_0_0p05 - 13.671124604482333
    if Q.girth < 0.076081777364:
        z += 72.77807235717773 * Q.girth - 5.837737193776768
    if 0.076081777364 <= Q.girth < 0.087236513197:
        z += 26.952865600585938 * Q.girth - 2.3512740156624825
    if Q.centroid_offset < 0.031170772021:
        z += -43.41165542602539 * Q.centroid_offset + 1.3531748143388451
    if Q.e2 < 0.020459658932:
        z += 56.11396026611328 * Q.e2 - 1.1480724883684776
    if Q.mass_over_sum_pt_sq < 0.008174660116:
        z += 332.32537841796875 * Q.mass_over_sum_pt_sq - 2.7166470164879764
    if 6.377722943814 <= Q.log_sum_pt < 6.638338705138:
        z += 0.685285747051239 * Q.log_sum_pt - 4.370562632037404
    if Q.log_sum_pt >= 6.638338705138:
        z += -4.110272705554962 * Q.log_sum_pt + 27.463978656650035
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.009818922728300095 * Q.sum_pt_top5 - 6.749895693035796
    if Q.mass_over_sum_pt < 0.13092863437:
        z += 23.11655616760254 * Q.mass_over_sum_pt - 3.0266191303616012
    if Q.planar_flow < 0.148419710734 and Q.centroid_offset < 0.049903668404:
        z += 107.9202880859375 * (0.148419710734 - Q.planar_flow) * (0.049903668404 - Q.centroid_offset)
    if Q.lam1 < 0.006506575659 and Q.D2 < 0.87567204833:
        z += -1941.3631591796875 * (0.006506575659 - Q.lam1) * (0.87567204833 - Q.D2)
    if Q.sum_pt > 901.59375 and Q.pt_7 < 25.578125:
        z += 0.0021638909820467234 * (Q.sum_pt - 901.59375) * (25.578125 - Q.pt_7)
    if Q.planar_flow < 0.148419710734 and Q.sum_pt_top2 < 380.5875:
        z += -0.05131261795759201 * (0.148419710734 - Q.planar_flow) * (380.5875 - Q.sum_pt_top2)
    if Q.sum_pt > 901.59375 and Q.dr_2 < 0.038438041256:
        z += -0.15407106280326843 * (Q.sum_pt - 901.59375) * (0.038438041256 - Q.dr_2)
    if Q.mass < 56.920347213745 and Q.C2 > 0.023843882605:
        z += 0.9480746388435364 * (56.920347213745 - Q.mass) * (Q.C2 - 0.023843882605)
    if Q.mass < 29.644699859619 and Q.D2 < 0.87567204833:
        z += -2.9894216060638428 * (29.644699859619 - Q.mass) * (0.87567204833 - Q.D2)
    if Q.planar_flow < 0.148419710734 and Q.max_dr < 0.111761856824:
        z += -148.85391235351562 * (0.148419710734 - Q.planar_flow) * (0.111761856824 - Q.max_dr)
    if Q.girth2 < 0.013238675334 and Q.centroid_offset > 0.014379521101:
        z += -1929.25634765625 * (0.013238675334 - Q.girth2) * (Q.centroid_offset - 0.014379521101)
    if Q.mass < 64.618731689453 and Q.centroid_offset > 0.012587644117:
        z += 3.318119525909424 * (64.618731689453 - Q.mass) * (Q.centroid_offset - 0.012587644117)
    if Q.girth2 < 0.018827652745 and Q.eccentricity > 0.959856212153:
        z += 2494.130859375 * (0.018827652745 - Q.girth2) * (Q.eccentricity - 0.959856212153)
    if Q.girth2_top3 < 0.007929074034 and Q.pt_6 < 35.28125:
        z += -3.411953926086426 * (0.007929074034 - Q.girth2_top3) * (35.28125 - Q.pt_6)
    if Q.sum_pt_top5 > 687.4375 and Q.z_7 > 0.023207568189:
        z += -0.7383662462234497 * (Q.sum_pt_top5 - 687.4375) * (Q.z_7 - 0.023207568189)
    if Q.mass < 64.618731689453 and Q.pt_7 < 40.040625:
        z += -0.002179805189371109 * (64.618731689453 - Q.mass) * (40.040625 - Q.pt_7)
    if Q.log_sum_pt > 6.638338705138 and Q.dr_7 < 0.093979107928:
        z += 14.785979270935059 * (Q.log_sum_pt - 6.638338705138) * (0.093979107928 - Q.dr_7)
    if Q.mass < 29.644699859619 and Q.phi_1 > -0.058901977539:
        z += -3.114445209503174 * (29.644699859619 - Q.mass) * (Q.phi_1 - -0.058901977539)
    if Q.width < 0.004372139331 and Q.n_dr_0p1_0p2 > 1.0:
        z += -570.86083984375 * (0.004372139331 - Q.width) * (Q.n_dr_0p1_0p2 - 1.0)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.8196374177932739
    if Q.lam1 < 0.008375572068:
        z += 140.32012939453125 * Q.lam1 - 1.1752613563349816
    if Q.pt_7 >= 34.53125:
        z += 0.1119973435997963 * Q.pt_7 - 3.867408271180466
    if Q.e2 < 0.032346998155:
        z += -4.5604376792907715 * Q.e2 + 0.16217332420781977
    if 0.032346998155 <= Q.e2 < 0.035560912266:
        z += -37.640817165374756 * Q.e2 + 1.2322242984108784
    if Q.e2 >= 0.035560912266:
        z += -33.080379486083984 * Q.e2 + 1.0700509742030586
    if 6.377722943814 <= Q.log_sum_pt < 6.572937922293:
        z += 3.375440835952759 * Q.log_sum_pt - 21.527626464942617
    if Q.log_sum_pt >= 6.572937922293:
        z += 9.939862966537476 * Q.log_sum_pt - 64.67516562500231
    if Q.z_7 < 0.055577157257:
        z += 82.4200668334961 * Q.z_7 - 4.580673015537663
    if Q.girth < 0.076081777364:
        z += -13.023504257202148 * Q.girth + 0.99085135139556
    if Q.mass < 53.332374954224:
        z += 0.01573619805276394 * Q.mass - 0.839248814903936
    if Q.width < 0.008678044951:
        z += 472.0110778808594 * Q.width - 4.096133351220059
    if Q.planar_flow < 0.083662731125:
        z += -3.6813559532165527 * Q.planar_flow + 0.30799229328937455
    if Q.tau21 < 0.197783735394:
        z += 2.7851734161376953 * Q.tau21 - 0.550862001963781
    if Q.e2_sq < 0.008168570676:
        z += -468.915283203125 * Q.e2_sq + 3.8303676319012827
    if Q.C2 < 0.042322802544:
        z += 30.05048179626465 * Q.C2 - 1.271820607415375
    if Q.LHA < 0.346713497427:
        z += 4.485969066619873 * Q.LHA - 1.555346024437111
    if Q.mass_over_sum_pt >= 0.054892207095:
        z += 22.743154525756836 * Q.mass_over_sum_pt - 1.2484219482214307
    if Q.max_dr < 0.046566883102:
        z += 57.71333694458008 * Q.max_dr - 2.6875302149245983
    if Q.lam1 < 0.008375572068 and Q.centroid_offset > 0.02076709205:
        z += 15641.892578125 * (0.008375572068 - Q.lam1) * (Q.centroid_offset - 0.02076709205)
    if Q.pt_7 > 34.53125 and Q.mass < 91.19:
        z += -0.0016289105406031013 * (Q.pt_7 - 34.53125) * (91.19 - Q.mass)
    if Q.pt_7 > 34.53125 and Q.centroid_offset > 0.014379521101:
        z += 1.3202705383300781 * (Q.pt_7 - 34.53125) * (Q.centroid_offset - 0.014379521101)
    if Q.log_sum_pt > 6.377722943814 and Q.centroid_offset < 0.023554160423:
        z += -225.88729858398438 * (Q.log_sum_pt - 6.377722943814) * (0.023554160423 - Q.centroid_offset)
    if Q.pt_7 > 53.4375 and Q.mass_top3 < 45.595:
        z += 0.004256657790392637 * (Q.pt_7 - 53.4375) * (45.595 - Q.mass_top3)
    if Q.log_sum_pt > 6.377722943814 and Q.max_dr < 0.197968879342:
        z += 16.93523406982422 * (Q.log_sum_pt - 6.377722943814) * (0.197968879342 - Q.max_dr)
    if Q.pt_7 > 53.4375 and Q.tau21 < 0.553068161011:
        z += -0.1409144252538681 * (Q.pt_7 - 53.4375) * (0.553068161011 - Q.tau21)
    if Q.width < 0.008678044951 and Q.planar_flow < 0.083662731125:
        z += -2104.9384765625 * (0.008678044951 - Q.width) * (0.083662731125 - Q.planar_flow)
    if Q.e2 < 0.035560912266 and Q.eccentricity > 0.978160776925:
        z += 6639.4853515625 * (0.035560912266 - Q.e2) * (Q.eccentricity - 0.978160776925)
    if Q.z_7 < 0.055577157257 and Q.C2 < 0.042322802544:
        z += 817.3705444335938 * (0.055577157257 - Q.z_7) * (0.042322802544 - Q.C2)
    if Q.pt_7 > 34.53125 and Q.max_dr < 0.080507021025:
        z += 0.3280085027217865 * (Q.pt_7 - 34.53125) * (0.080507021025 - Q.max_dr)
    if Q.pt_7 > 53.4375 and Q.mass_top3 < 32.617988451746:
        z += -0.00410256115719676 * (Q.pt_7 - 53.4375) * (32.617988451746 - Q.mass_top3)
    if Q.tau21 < 0.197783735394 and Q.pt_6 < 50.25:
        z += -0.20382951200008392 * (0.197783735394 - Q.tau21) * (50.25 - Q.pt_6)
    if Q.log_sum_pt > 6.572937922293 and Q.girth2_top3 < 0.006756161242:
        z += -1270.469970703125 * (Q.log_sum_pt - 6.572937922293) * (0.006756161242 - Q.girth2_top3)
    if Q.z_7 < 0.055577157257 and Q.girth2_top3 < 0.007929074034:
        z += 7158.86376953125 * (0.055577157257 - Q.z_7) * (0.007929074034 - Q.girth2_top3)
    if Q.tau21 < 0.197783735394 and Q.lam2 < 0.000194798295:
        z += 25721.294921875 * (0.197783735394 - Q.tau21) * (0.000194798295 - Q.lam2)
    if Q.log_sum_pt > 6.572937922293 and Q.lam2 < 0.001130644719:
        z += 6611.62841796875 * (Q.log_sum_pt - 6.572937922293) * (0.001130644719 - Q.lam2)
    if Q.planar_flow < 0.083662731125 and Q.tau32 < 0.269169217348:
        z += -365.6162414550781 * (0.083662731125 - Q.planar_flow) * (0.269169217348 - Q.tau32)
    if Q.LHA < 0.346713497427 and Q.planar_flow < 0.083662731125:
        z += -77.50286865234375 * (0.346713497427 - Q.LHA) * (0.083662731125 - Q.planar_flow)
    if Q.lam1 < 0.008375572068 and Q.n_pt_above_50 > 6.0:
        z += -74.69569396972656 * (0.008375572068 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.e2_sq < 0.008168570676 and Q.planar_flow < 0.083662731125:
        z += -6446.86376953125 * (0.008168570676 - Q.e2_sq) * (0.083662731125 - Q.planar_flow)
    if Q.e2_sq < 0.008168570676 and Q.n_pt_above_50 > 6.0:
        z += 74.844970703125 * (0.008168570676 - Q.e2_sq) * (Q.n_pt_above_50 - 6.0)
    if Q.log_sum_pt > 6.572937922293 and Q.planar_flow < 0.033200121667:
        z += 101.3279800415039 * (Q.log_sum_pt - 6.572937922293) * (0.033200121667 - Q.planar_flow)
    if Q.log_sum_pt > 6.572937922293 and Q.n_pt_above_50 > 7.0:
        z += -6.983942031860352 * (Q.log_sum_pt - 6.572937922293) * (Q.n_pt_above_50 - 7.0)
    if Q.e2 > 0.032346998155 and Q.tau32 < 0.641386964917:
        z += -76.17869567871094 * (Q.e2 - 0.032346998155) * (0.641386964917 - Q.tau32)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.805358648300171
    if Q.mass < 36.229410171509:
        z += 0.06266826204955578 * Q.mass - 1.5817882134662173
    if 36.229410171509 <= Q.mass < 69.611351776123:
        z += -0.020629296079277992 * Q.mass + 1.4360331862684153
    if Q.log_sum_pt < 6.267538488641:
        z += -3.9003331661224365 * Q.log_sum_pt + 25.212339117840322
    if 6.267538488641 <= Q.log_sum_pt < 6.464150123592:
        z += -3.1148114800453186 * Q.log_sum_pt + 20.289051716689812
    if 6.464150123592 <= Q.log_sum_pt < 6.804164030582:
        z += 0.7855216860771179 * Q.log_sum_pt - 4.9232874011505094
    if 6.804164030582 <= Q.log_sum_pt < 6.842716632804:
        z += 4.438126981258392 * Q.log_sum_pt - 29.776212968536285
    if 6.842716632804 <= Q.log_sum_pt < 6.896095378249:
        z += 12.976591527462006 * Q.log_sum_pt - 88.20250633745101
    if Q.log_sum_pt >= 6.896095378249:
        z += -6.020820200443268 * Q.log_sum_pt + 42.80545687804989
    if Q.z_7 < 0.016858545121:
        z += 253.0012664794922 * Q.z_7 - 3.492971543085936
    if 0.016858545121 <= Q.z_7 < 0.028070914944:
        z += -68.87586975097656 * Q.z_7 + 1.9334086814736855
    if Q.z_7 >= 0.046240320761:
        z += -32.77480697631836 * Q.z_7 + 1.5155175874648215
    if Q.lam1 < 0.003377388461:
        z += -408.52796936035156 * Q.lam1 + 2.097990406905875
    if 0.003377388461 <= Q.lam1 < 0.005954149834:
        z += -278.73468017578125 * Q.lam1 + 1.659628049698671
    if Q.width < 9.1213921e-05:
        z += 9678.7880859375 * Q.width - 0.8828402118464443
    if Q.pt_7 < 30.484375:
        z += 0.027273189276456833 * Q.pt_7 - 1.9131548339501023
    if 30.484375 <= Q.pt_7 < 43.5:
        z += 0.1837640218436718 * Q.pt_7 - 6.683680057991296
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.22962503135204315 * Q.pt_7 - 8.67863397160545
    if Q.pt_7 >= 53.4375:
        z += 0.15649083256721497 * Q.pt_7 - 4.770525224041194
    if Q.LHA >= 0.111565049159:
        z += -5.835907936096191 * Q.LHA + 0.6510833557779698
    if Q.planar_flow < 0.694781820497:
        z += 0.8839413523674011 * Q.planar_flow - 0.6141463820104031
    if Q.sum_pt_top5 < 687.4375:
        z += 0.006118496414273977 * Q.sum_pt_top5 - 4.206083878787467
    if Q.sum_pt < 788.4484375:
        z += -0.011290072463452816 * Q.sum_pt + 8.901639993071148
    if Q.girth2 < 4.8108519e-05:
        z += 75530.390625 * Q.girth2 - 3.6336552324602343
    if Q.max_dr < 0.15984864831:
        z += 1.8806202411651611 * Q.max_dr - 0.3006146035346772
    if Q.mass < 69.611351776123 and Q.z_7 < 0.068101508468:
        z += -0.30071333050727844 * (69.611351776123 - Q.mass) * (0.068101508468 - Q.z_7)
    if Q.pt_7 > 30.484375 and Q.C2 < 0.051192347892:
        z += -1.9363497495651245 * (Q.pt_7 - 30.484375) * (0.051192347892 - Q.C2)
    if Q.z_7 < 0.028070914944 and Q.width < 0.00752008842:
        z += -17836.787109375 * (0.028070914944 - Q.z_7) * (0.00752008842 - Q.width)
    if Q.lam1 < 0.005954149834 and Q.max_dr > 0.080507021025:
        z += -2027.5107421875 * (0.005954149834 - Q.lam1) * (Q.max_dr - 0.080507021025)
    if Q.pt_6 < 56.53125 and Q.m012 > 32.617988451746:
        z += -0.0007020625635050237 * (56.53125 - Q.pt_6) * (Q.m012 - 32.617988451746)
    if Q.mass < 36.229410171509 and Q.lam2 < 0.001130644719:
        z += 48.38774108886719 * (36.229410171509 - Q.mass) * (0.001130644719 - Q.lam2)
    if Q.pt_7 > 30.484375 and Q.max_dr > 0.093110798299:
        z += -0.5230648517608643 * (Q.pt_7 - 30.484375) * (Q.max_dr - 0.093110798299)
    if Q.lam1 < 0.003377388461 and Q.centroid_offset < 0.006789738266:
        z += -32517.5546875 * (0.003377388461 - Q.lam1) * (0.006789738266 - Q.centroid_offset)
    if Q.mass < 8.379955863953 and Q.centroid_offset < 0.010960638421:
        z += 45.29866409301758 * (8.379955863953 - Q.mass) * (0.010960638421 - Q.centroid_offset)
    if Q.sum_pt_top5 > 839.9546875 and Q.dr_7 < 0.04881348081:
        z += 0.1693996787071228 * (Q.sum_pt_top5 - 839.9546875) * (0.04881348081 - Q.dr_7)
    if Q.mass < 69.611351776123 and Q.max_pair_mass > 13.047927274731:
        z += -0.0019603243563324213 * (69.611351776123 - Q.mass) * (Q.max_pair_mass - 13.047927274731)
    if Q.width < 0.000172198326 and Q.dr_7 < 0.222994708167:
        z += -24038.572265625 * (0.000172198326 - Q.width) * (0.222994708167 - Q.dr_7)
    return max(0.0, z)


def neuron_3(Q):
    z = 3.0607335567474365
    if 0.06813910019 <= Q.mass_over_sum_pt < 0.090413827016:
        z += -57.21665573120117 * Q.mass_over_sum_pt + 3.8986914374050543
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += -15.975017547607422 * Q.mass_over_sum_pt + 0.16987709681714858
    if Q.mass_over_sum_pt >= 0.107985668755:
        z += -56.59498977661133 * Q.mass_over_sum_pt + 4.556251962775663
    if Q.centroid_offset >= 0.014379521101:
        z += 69.26995086669922 * Q.centroid_offset - 0.9960687201529346
    if Q.width < 0.006679471358:
        z += -83.43787384033203 * Q.width + 0.5573208884889153
    if Q.width >= 0.018827653081:
        z += 115.33635711669922 * Q.width - 2.1715129194195386
    if 36.229410171509 <= Q.mass < 69.611351776123:
        z += -0.026995524764060974 * Q.mass + 0.9780319394722938
    if Q.mass >= 69.611351776123:
        z += -0.09272447973489761 * Q.mass + 5.553513345824152
    if Q.e2 < 0.028531698044:
        z += -156.0390167236328 * Q.e2 + 6.9396118755252765
    if 0.028531698044 <= Q.e2 < 0.04447356835:
        z += -214.3046112060547 * Q.e2 + 8.60202822365189
    if Q.e2 >= 0.04447356835:
        z += -58.265594482421875 * Q.e2 + 1.6624163481266134
    if 0.312727471086 <= Q.LHA < 0.325582223496:
        z += 14.386687278747559 * Q.LHA - 4.4991123299878515
    if 0.325582223496 <= Q.LHA < 0.423592510895:
        z += 22.494805335998535 * Q.LHA - 7.138971435435693
    if Q.LHA >= 0.423592510895:
        z += -2.6731081008911133 * Q.LHA + 3.5219682112844026
    if Q.girth2 < 0.004372139461:
        z += 1063.1531372070312 * Q.girth2 - 14.168730398050428
    if 0.004372139461 <= Q.girth2 < 0.008678044751:
        z += 1714.9747924804688 * Q.girth2 - 17.018585578605762
    if 0.008678044751 <= Q.girth2 < 0.013238675334:
        z += 468.34698486328125 * Q.girth2 - 6.200293676262793
    if Q.lam1 < 0.007330079875:
        z += -346.7208251953125 * Q.lam1 + 2.541491343007553
    if Q.lam1 >= 0.012003726523:
        z += -329.45892333984375 * Q.lam1 + 3.9547348163335063
    if Q.mass_top5 >= 53.607658247923:
        z += -0.08511016517877579 * Q.mass_top5 + 4.562556648328089
    if Q.mean_eta < -0.012844925793:
        z += -17.782264709472656 * Q.mean_eta - 0.22841187062465898
    if Q.max_dr >= 0.145231109113:
        z += 15.918156623840332 * Q.max_dr - 2.311811541514779
    if Q.mass_over_sum_pt > 0.06813910019 and Q.tau32 < 0.518696343899:
        z += -119.858154296875 * (Q.mass_over_sum_pt - 0.06813910019) * (0.518696343899 - Q.tau32)
    if Q.e2 > 0.028531698044 and Q.n_pt_above_50 > 3.0:
        z += -3.755054235458374 * (Q.e2 - 0.028531698044) * (Q.n_pt_above_50 - 3.0)
    if Q.mass > 36.229410171509 and Q.lam2 < 0.001130644719:
        z += 27.896549224853516 * (Q.mass - 36.229410171509) * (0.001130644719 - Q.lam2)
    if Q.mass_over_sum_pt > 0.06813910019 and Q.n_dr_0_0p05 < 5.0:
        z += 13.978013038635254 * (Q.mass_over_sum_pt - 0.06813910019) * (5.0 - Q.n_dr_0_0p05)
    if Q.LHA > 0.312727471086 and Q.mass_top3 < 50.352200171245:
        z += -0.5618948936462402 * (Q.LHA - 0.312727471086) * (50.352200171245 - Q.mass_top3)
    if Q.e2 < 0.04447356835 and Q.z_dr_0p05_0p1 < 0.964120104909:
        z += -30.43331527709961 * (0.04447356835 - Q.e2) * (0.964120104909 - Q.z_dr_0p05_0p1)
    if Q.mass > 36.229410171509 and Q.eccentricity > 0.620723099573:
        z += 0.1261162906885147 * (Q.mass - 36.229410171509) * (Q.eccentricity - 0.620723099573)
    if Q.e2 < 0.04447356835 and Q.z_dr_0p1_0p2 > 0.0:
        z += -240.20448303222656 * (0.04447356835 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.0)
    if Q.centroid_offset > 0.014379521101 and Q.phi_7 > -0.041534423828:
        z += 62.41315460205078 * (Q.centroid_offset - 0.014379521101) * (Q.phi_7 - -0.041534423828)
    if Q.centroid_offset > 0.014379521101 and Q.eta_0 < -0.039672851562:
        z += -409.456298828125 * (Q.centroid_offset - 0.014379521101) * (-0.039672851562 - Q.eta_0)
    if Q.width > 0.018827653081 and Q.pt_dispersion < 0.492494773865:
        z += 995.661376953125 * (Q.width - 0.018827653081) * (0.492494773865 - Q.pt_dispersion)
    if Q.mass > 36.229410171509 and Q.dr_6 < 0.046481671275:
        z += -1.6770453453063965 * (Q.mass - 36.229410171509) * (0.046481671275 - Q.dr_6)
    if Q.mass_over_sum_pt > 0.06813910019 and Q.dr_7 < 0.042151962757:
        z += -9052.3330078125 * (Q.mass_over_sum_pt - 0.06813910019) * (0.042151962757 - Q.dr_7)
    if Q.LHA > 0.312727471086 and Q.max_dr < 0.145231109113:
        z += -2554.23193359375 * (Q.LHA - 0.312727471086) * (0.145231109113 - Q.max_dr)
    if Q.LHA > 0.312727471086 and Q.planar_flow > 0.00804883781:
        z += -17.711071014404297 * (Q.LHA - 0.312727471086) * (Q.planar_flow - 0.00804883781)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.max_dr < 0.145231109113:
        z += 11114.1328125 * (Q.mass_over_sum_pt - 0.090413827016) * (0.145231109113 - Q.max_dr)
    if Q.mass > 69.611351776123 and Q.max_dr < 0.177304983139:
        z += 3.3799712657928467 * (Q.mass - 69.611351776123) * (0.177304983139 - Q.max_dr)
    if Q.max_dr > 0.145231109113 and Q.dr_3 < 0.04586879935:
        z += -535.7786865234375 * (Q.max_dr - 0.145231109113) * (0.04586879935 - Q.dr_3)
    if Q.lam1 > 0.016433749775 and Q.eccentricity > 0.959856212153:
        z += -9652.0625 * (Q.lam1 - 0.016433749775) * (Q.eccentricity - 0.959856212153)
    if Q.mean_eta < -0.012844925793 and Q.pt_4 < 68.125:
        z += -2.0833659172058105 * (-0.012844925793 - Q.mean_eta) * (68.125 - Q.pt_4)
    if Q.mean_eta < -0.012844925793 and Q.z_4 < 0.120257140434:
        z += 1025.875244140625 * (-0.012844925793 - Q.mean_eta) * (0.120257140434 - Q.z_4)
    if Q.LHA > 0.312727471086 and Q.pt_5 < 29.875:
        z += 4.88065242767334 * (Q.LHA - 0.312727471086) * (29.875 - Q.pt_5)
    if Q.LHA > 0.312727471086 and Q.eccentricity > 0.959856212153:
        z += 1277.3314208984375 * (Q.LHA - 0.312727471086) * (Q.eccentricity - 0.959856212153)
    if Q.e2 > 0.028531698044 and Q.eccentricity > 0.959856212153:
        z += -1373.02490234375 * (Q.e2 - 0.028531698044) * (Q.eccentricity - 0.959856212153)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.pt_6 < 56.53125:
        z += -1.589561104774475 * (Q.mass_over_sum_pt - 0.090413827016) * (56.53125 - Q.pt_6)
    if Q.lam1 > 0.012003726523 and Q.pt_6 < 38.25:
        z += 28.534944534301758 * (Q.lam1 - 0.012003726523) * (38.25 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.pt_7 < 48.71875:
        z += 0.8980383276939392 * (Q.mass_over_sum_pt - 0.090413827016) * (48.71875 - Q.pt_7)
    if Q.mass > 36.229410171509 and Q.pt_7 < 20.125:
        z += -0.0058251237496733665 * (Q.mass - 36.229410171509) * (20.125 - Q.pt_7)
    return max(0.0, z)


def neuron_4(Q):
    z = -7.008694648742676
    if Q.tau21 < 0.23799610585:
        z += -27.538393020629883 * Q.tau21 + 6.554030300276731
    if Q.lam2 < 0.000306123359:
        z += 5182.751953125 * Q.lam2 - 1.8568160268205294
    if 0.000306123359 <= Q.lam2 < 0.001130644719:
        z += 327.771484375 * Q.lam2 - 0.37059309784738476
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += -230.8987579345703 * Q.mass_over_sum_pt + 20.876440358105498
    if Q.mass_over_sum_pt >= 0.107985668755:
        z += 139.88804626464844 * Q.mass_over_sum_pt - 19.163220658876384
    if 0.000319370692 <= Q.width < 0.001653836415:
        z += -494.15948486328125 * Q.width + 0.15782005663914966
    if Q.width >= 0.001653836415:
        z += 506.640380859375 * Q.width - 1.4973392054200896
    if Q.sum_pt < 763.825:
        z += -0.008025175891816616 * Q.sum_pt + 6.129829975566827
    if Q.mass < 56.920347213745:
        z += -0.0014021024107933044 * Q.mass - 0.8651049440262168
    if 56.920347213745 <= Q.mass < 69.611351776123:
        z += 0.07445534318685532 * Q.mass - 5.182937086192148
    if 0.014943876117 <= Q.C2 < 0.067292226106:
        z += -49.01862335205078 * Q.C2 + 0.7325282347989301
    if Q.C2 >= 0.067292226106:
        z += -196.74713897705078 * Q.C2 + 10.673508910540184
    if Q.girth2_top3 < 0.005011406868:
        z += -403.519287109375 * Q.girth2_top3 + 2.022199326790386
    if Q.girth < 0.047915700823:
        z += -33.46483612060547 * Q.girth + 1.6034910756456555
    if Q.girth >= 0.101940929517:
        z += 7.6168928146362305 * Q.girth - 0.7764731335553757
    if Q.e2_sq < 0.017162483186:
        z += -507.98319244384766 * Q.e2_sq + 9.416673935415995
    if 0.017162483186 <= Q.e2_sq < 0.023780909279:
        z += -105.52674102783203 * Q.e2_sq + 2.5095218548914007
    if Q.girth2 >= 0.013238675334:
        z += -189.97628784179688 * Q.girth2 + 2.5150343958960804
    if Q.sum_pt_top5 < 430.75:
        z += -0.012057801708579063 * Q.sum_pt_top5 + 5.193898085970432
    if 0.007078157854 <= Q.e2 < 0.063441075385:
        z += 74.27765655517578 * Q.e2 - 0.525748978122732
    if Q.e2 >= 0.063441075385:
        z += -45.53630065917969 * Q.e2 + 7.075377313688358
    if Q.centroid_offset < 0.014379521101:
        z += -104.00109100341797 * Q.centroid_offset + 1.4954858826106698
    if 0.102758520097 <= Q.max_dr < 0.197968879342:
        z += 14.388410568237305 * Q.max_dr - 1.4785317765401003
    if Q.max_dr >= 0.197968879342:
        z += -8.279176712036133 * Q.max_dr + 3.0089450747226056
    if Q.mass_over_sum_pt_sq < 0.001101860861:
        z += -966.3388671875 * Q.mass_over_sum_pt_sq + 1.0647709762169835
    if Q.tau21 < 0.23799610585 and Q.mass < 62.55:
        z += -0.6512523889541626 * (0.23799610585 - Q.tau21) * (62.55 - Q.mass)
    if Q.tau21 < 0.23799610585 and Q.e2_sq > 0.011657374702:
        z += -1813.19873046875 * (0.23799610585 - Q.tau21) * (Q.e2_sq - 0.011657374702)
    if Q.tau21 < 0.23799610585 and Q.lam2 < 0.001130644719:
        z += -11762.017578125 * (0.23799610585 - Q.tau21) * (0.001130644719 - Q.lam2)
    if Q.width > 0.000319370692 and Q.C2 < 0.067292226106:
        z += 3998.565673828125 * (Q.width - 0.000319370692) * (0.067292226106 - Q.C2)
    if Q.girth2_top2 < 0.009530300104 and Q.centroid_offset > 0.016278845848:
        z += 10970.2421875 * (0.009530300104 - Q.girth2_top2) * (Q.centroid_offset - 0.016278845848)
    if Q.tau21 < 0.23799610585 and Q.planar_flow > 0.045057236346:
        z += 26.038362503051758 * (0.23799610585 - Q.tau21) * (Q.planar_flow - 0.045057236346)
    if Q.sum_pt < 763.825 and Q.n_dr_0p2_0p4 < 1.0:
        z += -0.002834032056853175 * (763.825 - Q.sum_pt) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.width > 0.000319370692 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += -731.7764282226562 * (Q.width - 0.000319370692) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.tau21 < 0.23799610585 and Q.dr_7 < 0.175465903809:
        z += 22.814733505249023 * (0.23799610585 - Q.tau21) * (0.175465903809 - Q.dr_7)
    if Q.lam2 < 0.001130644719 and Q.n_dr_0p1_0p2 > 2.0:
        z += -252.45872497558594 * (0.001130644719 - Q.lam2) * (Q.n_dr_0p1_0p2 - 2.0)
    if Q.mass < 69.611351776123 and Q.dr_3 < 0.104247858869:
        z += 0.1646825671195984 * (69.611351776123 - Q.mass) * (0.104247858869 - Q.dr_3)
    if Q.tau21 < 0.23799610585 and Q.mean_phi < -0.025945045147:
        z += -452.0412902832031 * (0.23799610585 - Q.tau21) * (-0.025945045147 - Q.mean_phi)
    if Q.tau21 < 0.23799610585 and Q.pt_7 > 33.21875:
        z += 0.8376650214195251 * (0.23799610585 - Q.tau21) * (Q.pt_7 - 33.21875)
    if Q.mass_over_sum_pt > 0.107985668755 and Q.D2 < 3.885568320751:
        z += -63.07530975341797 * (Q.mass_over_sum_pt - 0.107985668755) * (3.885568320751 - Q.D2)
    if Q.max_dr > 0.102758520097 and Q.pt_7 > 37.15625:
        z += -2.6649463176727295 * (Q.max_dr - 0.102758520097) * (Q.pt_7 - 37.15625)
    if Q.C2 > 0.014943876117 and Q.pt_7 > 38.53125:
        z += 12.515557289123535 * (Q.C2 - 0.014943876117) * (Q.pt_7 - 38.53125)
    if Q.lam2 < 0.000306123359 and Q.mass_top3 < 50.352200171245:
        z += -70.32608795166016 * (0.000306123359 - Q.lam2) * (50.352200171245 - Q.mass_top3)
    if Q.centroid_offset < 0.014379521101 and Q.z_dr_0p05_0p1 < 0.674770402908:
        z += -139.7335968017578 * (0.014379521101 - Q.centroid_offset) * (0.674770402908 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.23799610585 and Q.pt_2 > 73.6875:
        z += 0.036674901843070984 * (0.23799610585 - Q.tau21) * (Q.pt_2 - 73.6875)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.38814184069633484
    if Q.LHA < 0.216055863061:
        z += -15.331228256225586 * Q.LHA + 3.312401752684009
    if Q.z_7 < 0.03243272066:
        z += -179.99537086486816 * Q.z_7 + 8.960601811604153
    if 0.03243272066 <= Q.z_7 < 0.049399692737:
        z += -89.77286720275879 * Q.z_7 + 6.034440553085134
    if 0.049399692737 <= Q.z_7 < 0.071488645583:
        z += -72.42029571533203 * Q.z_7 + 5.177228853409425
    if Q.log_sum_pt >= 6.896095378249:
        z += -27.82596778869629 * Q.log_sum_pt + 191.89052786293402
    if Q.e2_sq < 0.00528466865:
        z += -191.58621215820312 * Q.e2_sq + 1.012469649164705
    if Q.z_6 < 0.028865759995:
        z += -110.65009307861328 * Q.z_6 + 3.193999030231662
    if Q.dr_0 < 0.021588001063:
        z += 269.8489685058594 * Q.dr_0 - 6.285803671437314
    if 0.021588001063 <= Q.dr_0 < 0.026454043164:
        z += 94.59512329101562 * Q.dr_0 - 2.5024234746444294
    if Q.e2 < 0.035560912266:
        z += -51.872314453125 * Q.e2 + 1.8446268233019418
    if Q.width < 0.002635417778:
        z += -414.52789306640625 * Q.width + 1.09245417886409
    if Q.girth2 < 4.8108519e-05:
        z += -29633.046875 * Q.girth2 + 1.4256019986138282
    if Q.mass_over_sum_pt < 0.084751611895:
        z += 8.096582412719727 * Q.mass_over_sum_pt - 0.686198410318705
    if Q.girth2_top5 < 0.002270363079:
        z += 575.0505981445312 * Q.girth2_top5 - 1.3055736465842098
    if Q.sum_pt_top5 >= 752.1:
        z += 0.01413116604089737 * Q.sum_pt_top5 - 10.628049979358911
    if Q.sum_pt_top2 < 548.196875:
        z += 0.0019051458220928907 * Q.sum_pt_top2 - 1.0443949860906285
    if Q.pt_5 < 24.578125:
        z += 0.07115589082241058 * Q.pt_5 - 1.7488783791195601
    if Q.sum_pt >= 868.509375:
        z += -0.029430223628878593 * Q.sum_pt + 25.56042513002758
    if Q.mean_phi2 < 0.01426135283:
        z += -41.02766418457031 * Q.mean_phi2 + 0.5851099947269115
    if Q.z_7 < 0.049399692737 and Q.mass_top5 < 62.55:
        z += 1.0780237913131714 * (0.049399692737 - Q.z_7) * (62.55 - Q.mass_top5)
    if Q.LHA < 0.216055863061 and Q.log_sum_pt < 6.804164030582:
        z += -107.15074920654297 * (0.216055863061 - Q.LHA) * (6.804164030582 - Q.log_sum_pt)
    if Q.e2_sq < 0.00528466865 and Q.centroid_offset < 0.014379521101:
        z += -5278.8583984375 * (0.00528466865 - Q.e2_sq) * (0.014379521101 - Q.centroid_offset)
    if Q.z_7 < 0.071488645583 and Q.centroid_offset < 0.031170772021:
        z += -3192.267578125 * (0.071488645583 - Q.z_7) * (0.031170772021 - Q.centroid_offset)
    if Q.z_7 < 0.071488645583 and Q.sum_pt < 788.4484375:
        z += -0.24188272655010223 * (0.071488645583 - Q.z_7) * (788.4484375 - Q.sum_pt)
    if Q.z_7 < 0.071488645583 and Q.lam1 < 0.001503553356:
        z += -26296.830078125 * (0.071488645583 - Q.z_7) * (0.001503553356 - Q.lam1)
    if Q.e2_sq < 0.00528466865 and Q.n_pt_above_50 > 6.0:
        z += -57.107547760009766 * (0.00528466865 - Q.e2_sq) * (Q.n_pt_above_50 - 6.0)
    if Q.LHA < 0.216055863061 and Q.n_dr_0p2_0p4 > 0.0:
        z += -7.856075286865234 * (0.216055863061 - Q.LHA) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.z_7 < 0.071488645583 and Q.n_dr_0p2_0p4 < 1.0:
        z += 6.9307756423950195 * (0.071488645583 - Q.z_7) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_7 < 0.071488645583 and Q.lam2 < 0.001130644719:
        z += 11840.1982421875 * (0.071488645583 - Q.z_7) * (0.001130644719 - Q.lam2)
    if Q.width < 0.002635417778 and Q.centroid_offset < 0.023554160423:
        z += 82593.8984375 * (0.002635417778 - Q.width) * (0.023554160423 - Q.centroid_offset)
    if Q.log_sum_pt > 6.572937922293 and Q.centroid_offset > 0.018377780003:
        z += -192.214111328125 * (Q.log_sum_pt - 6.572937922293) * (Q.centroid_offset - 0.018377780003)
    if Q.z_7 < 0.03243272066 and Q.pt_5 > 33.0265625:
        z += 2.6221656799316406 * (0.03243272066 - Q.z_7) * (Q.pt_5 - 33.0265625)
    if Q.sum_pt > 868.509375 and Q.centroid_offset < 0.012587644117:
        z += 1.7417271137237549 * (Q.sum_pt - 868.509375) * (0.012587644117 - Q.centroid_offset)
    if Q.mass_over_sum_pt < 0.084751611895 and Q.mass_top3 > 28.345095968085:
        z += -3.767728567123413 * (0.084751611895 - Q.mass_over_sum_pt) * (Q.mass_top3 - 28.345095968085)
    if Q.log_sum_pt > 6.896095378249 and Q.centroid_offset < 0.018377780003:
        z += -967.9581909179688 * (Q.log_sum_pt - 6.896095378249) * (0.018377780003 - Q.centroid_offset)
    if Q.log_sum_pt > 6.572937922293 and Q.mean_phi2 < 0.000145482056:
        z += 16792.763671875 * (Q.log_sum_pt - 6.572937922293) * (0.000145482056 - Q.mean_phi2)
    if Q.sum_pt_top2 < 548.196875 and Q.dr_0 < 0.021588001063:
        z += 0.4094429016113281 * (548.196875 - Q.sum_pt_top2) * (0.021588001063 - Q.dr_0)
    if Q.log_sum_pt > 6.572937922293 and Q.dr_0 < 0.021588001063:
        z += 534.6256713867188 * (Q.log_sum_pt - 6.572937922293) * (0.021588001063 - Q.dr_0)
    if Q.log_sum_pt > 6.896095378249 and Q.dr_0 < 0.04118638065:
        z += -1043.9903564453125 * (Q.log_sum_pt - 6.896095378249) * (0.04118638065 - Q.dr_0)
    if Q.z_7 < 0.071488645583 and Q.mean_phi2 < 0.004331280361:
        z += -2585.630126953125 * (0.071488645583 - Q.z_7) * (0.004331280361 - Q.mean_phi2)
    if Q.log_sum_pt > 6.572937922293 and Q.lam1 < 0.012003726523:
        z += -1121.2227783203125 * (Q.log_sum_pt - 6.572937922293) * (0.012003726523 - Q.lam1)
    if Q.log_sum_pt > 6.572937922293 and Q.girth2_top2 < 0.006299534492:
        z += 673.7136840820312 * (Q.log_sum_pt - 6.572937922293) * (0.006299534492 - Q.girth2_top2)
    if Q.sum_pt_top2 < 548.196875 and Q.girth2_top3 < 0.003952581551:
        z += -2.1523637771606445 * (548.196875 - Q.sum_pt_top2) * (0.003952581551 - Q.girth2_top3)
    if Q.log_sum_pt > 6.572937922293 and Q.mean_eta2 < 9.0303693e-05:
        z += 31034.154296875 * (Q.log_sum_pt - 6.572937922293) * (9.0303693e-05 - Q.mean_eta2)
    if Q.mass_over_sum_pt < 0.084751611895 and Q.max_pair_mass < 18.097979966098:
        z += 0.775526762008667 * (0.084751611895 - Q.mass_over_sum_pt) * (18.097979966098 - Q.max_pair_mass)
    if Q.LHA < 0.216055863061 and Q.mass_top3 > 3.559569591142:
        z += -1.6385955810546875 * (0.216055863061 - Q.LHA) * (Q.mass_top3 - 3.559569591142)
    if Q.mean_phi2 < 0.01426135283 and Q.max_pair_mass < 40.046952646555:
        z += 1.789231300354004 * (0.01426135283 - Q.mean_phi2) * (40.046952646555 - Q.max_pair_mass)
    if Q.LHA < 0.216055863061 and Q.planar_flow < 0.083662731125:
        z += 449.9190368652344 * (0.216055863061 - Q.LHA) * (0.083662731125 - Q.planar_flow)
    if Q.e2 < 0.035560912266 and Q.lam2 < 7.3007261e-05:
        z += 417995.875 * (0.035560912266 - Q.e2) * (7.3007261e-05 - Q.lam2)
    return max(0.0, z)


def neuron_6(Q):
    z = 8.519858360290527
    if 0.008092360237 <= Q.centroid_offset < 0.018377780003:
        z += 21.563961029052734 * Q.centroid_offset - 0.17450334078372393
    if 0.018377780003 <= Q.centroid_offset < 0.049903668404:
        z += -83.35322952270508 * Q.centroid_offset + 1.7536417057093112
    if Q.centroid_offset >= 0.049903668404:
        z += -96.82500648498535 * Q.centroid_offset + 2.4259327960475923
    if Q.width < 0.013238675006:
        z += 657.2191162109375 * Q.width - 8.700710287247148
    if Q.mass_over_sum_pt < 0.008374148675:
        z += -5.2152018547058105 * Q.mass_over_sum_pt + 0.8040318761306707
    if 0.008374148675 <= Q.mass_over_sum_pt < 0.154170806525:
        z += -84.72536420822144 * Q.mass_over_sum_pt + 1.4698617968523986
    if Q.mass_over_sum_pt >= 0.154170806525:
        z += -79.51016235351562 * Q.mass_over_sum_pt + 0.6658299207217278
    if Q.log_sum_pt < 6.327378592257:
        z += -2.196419358253479 * Q.log_sum_pt + 13.766939090470483
    if 6.327378592257 <= Q.log_sum_pt < 6.572937922293:
        z += -0.6759984493255615 * Q.log_sum_pt + 4.146660380100048
    if 6.572937922293 <= Q.log_sum_pt < 6.701242202626:
        z += 2.3119685649871826 * Q.log_sum_pt - 15.49306131883678
    if Q.e2 < 0.050284641981:
        z += -71.60969543457031 * Q.e2 + 3.6008678972958186
    z += 16.249183654785156 * Q.max_dr
    if Q.C2 >= 0.010539266048:
        z += -21.693979263305664 * Q.C2 + 0.22863861909577343
    if Q.lam2 < 0.000537286005:
        z += 2033.771484375 * Q.lam2 - 1.0927169559227636
    if Q.lam2 >= 0.003408388935:
        z += -1051.9156494140625 * Q.lam2 + 3.58533766001623
    if 0.002270363079 <= Q.girth2_top5 < 0.011482925368:
        z += -32.54750061035156 * Q.girth2_top5 + 0.07389464369947216
    if Q.girth2_top5 >= 0.011482925368:
        z += 151.57972717285156 * Q.girth2_top5 - 2.0404245711517857
    if Q.girth2 < 0.003562611155:
        z += 736.35595703125 * Q.girth2 - 9.574132053465945
    if 0.003562611155 <= Q.girth2 < 0.008678044751:
        z += 1358.7864990234375 * Q.girth2 - 11.791610045580008
    if Q.lam1 < 0.007330079875:
        z += -549.3066864013672 * Q.lam1 + 4.33645603357235
    if 0.007330079875 <= Q.lam1 < 0.008375572068:
        z += -296.5054626464844 * Q.lam1 + 2.483402870951312
    if Q.girth >= 0.087236513197:
        z += 89.02523040771484 * Q.girth - 7.766250687328582
    if Q.D2 < 1.679198372364:
        z += -0.3606502413749695 * Q.D2 + 0.6056032983095325
    if Q.mass < 49.668099212646:
        z += 0.012822299264371395 * Q.mass - 0.6368592319970362
    if Q.z_7 >= 0.06164517166:
        z += -4.518622398376465 * Q.z_7 + 0.27855125341463804
    if Q.centroid_offset > 0.008092360237 and Q.lam2 < 0.003408388935:
        z += 14174.123046875 * (Q.centroid_offset - 0.008092360237) * (0.003408388935 - Q.lam2)
    if Q.log_sum_pt < 6.327378592257 and Q.z_7 < 0.071488645583:
        z += 871.4190673828125 * (6.327378592257 - Q.log_sum_pt) * (0.071488645583 - Q.z_7)
    if Q.centroid_offset > 0.008092360237 and Q.planar_flow > 0.00804883781:
        z += 33.431434631347656 * (Q.centroid_offset - 0.008092360237) * (Q.planar_flow - 0.00804883781)
    if Q.centroid_offset > 0.008092360237 and Q.C2 < 0.094821243733:
        z += 210.83309936523438 * (Q.centroid_offset - 0.008092360237) * (0.094821243733 - Q.C2)
    if Q.log_sum_pt < 6.327378592257 and Q.pt_6 > 27.578125:
        z += 0.06460689008235931 * (6.327378592257 - Q.log_sum_pt) * (Q.pt_6 - 27.578125)
    if Q.log_sum_pt < 6.701242202626 and Q.z_7 < 0.049399692737:
        z += 690.1288452148438 * (6.701242202626 - Q.log_sum_pt) * (0.049399692737 - Q.z_7)
    if Q.centroid_offset > 0.018377780003 and Q.mean_phi2 < 0.008921136335:
        z += 3782.0 * (Q.centroid_offset - 0.018377780003) * (0.008921136335 - Q.mean_phi2)
    if Q.LHA > 0.312727471086 and Q.eccentricity > 0.872657364787:
        z += 179.82835388183594 * (Q.LHA - 0.312727471086) * (Q.eccentricity - 0.872657364787)
    if Q.e2 < 0.050284641981 and Q.n_dr_0p05_0p1 < 4.0:
        z += -2.408892869949341 * (0.050284641981 - Q.e2) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.width < 0.013238675006 and Q.mean_phi > 0.026127964072:
        z += 6150.400390625 * (0.013238675006 - Q.width) * (Q.mean_phi - 0.026127964072)
    if Q.log_sum_pt < 6.701242202626 and Q.mean_phi > 0.009050007537:
        z += -46.330692291259766 * (6.701242202626 - Q.log_sum_pt) * (Q.mean_phi - 0.009050007537)
    if Q.mass_over_sum_pt > 0.008374148675 and Q.tau32 < 0.518696343899:
        z += -58.42142105102539 * (Q.mass_over_sum_pt - 0.008374148675) * (0.518696343899 - Q.tau32)
    if Q.centroid_offset > 0.018377780003 and Q.pt_2 > 56.5:
        z += 0.5225324630737305 * (Q.centroid_offset - 0.018377780003) * (Q.pt_2 - 56.5)
    if Q.lam1 < 0.007330079875 and Q.D2 < 1.232133567333:
        z += 133.6427459716797 * (0.007330079875 - Q.lam1) * (1.232133567333 - Q.D2)
    if Q.e2 < 0.050284641981 and Q.z_dr_0p1_0p2 > 0.15855820179:
        z += 791.982421875 * (0.050284641981 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.15855820179)
    if Q.log_sum_pt < 6.701242202626 and Q.pt_6 < 36.8125:
        z += 0.11659538745880127 * (6.701242202626 - Q.log_sum_pt) * (36.8125 - Q.pt_6)
    if Q.centroid_offset > 0.018377780003 and Q.pt_5 < 24.578125:
        z += 60.996524810791016 * (Q.centroid_offset - 0.018377780003) * (24.578125 - Q.pt_5)
    if Q.width < 0.013238675006 and Q.mean_eta > 0.02644207105:
        z += 9071.90234375 * (0.013238675006 - Q.width) * (Q.mean_eta - 0.02644207105)
    if Q.lam2 < 0.000537286005 and Q.mean_eta > 0.02644207105:
        z += -131384.3125 * (0.000537286005 - Q.lam2) * (Q.mean_eta - 0.02644207105)
    if Q.D2 < 1.679198372364 and Q.pt_4 < 90.625:
        z += -0.017002524808049202 * (1.679198372364 - Q.D2) * (90.625 - Q.pt_4)
    if Q.log_sum_pt < 6.701242202626 and Q.pt_7 < 45.75:
        z += 0.23635320365428925 * (6.701242202626 - Q.log_sum_pt) * (45.75 - Q.pt_7)
    if Q.C2 > 0.010539266048 and Q.pt_7 > 31.859375:
        z += 2.7531425952911377 * (Q.C2 - 0.010539266048) * (Q.pt_7 - 31.859375)
    if Q.mass < 49.668099212646 and Q.z_dr_0p05_0p1 < 0.750909513235:
        z += 0.062347136437892914 * (49.668099212646 - Q.mass) * (0.750909513235 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.877108573913574
    if Q.planar_flow < 0.195013533663:
        z += 6.695348262786865 * Q.planar_flow - 1.305683523830495
    if Q.girth2_top2 < 0.001056655216:
        z += 740.2387084960938 * Q.girth2_top2 - 0.782177092417501
    if 0.0016538364 <= Q.girth2 < 0.004372139461:
        z += -540.9749755859375 * Q.girth2 + 0.8946841061131348
    if 0.004372139461 <= Q.girth2 < 0.007520088344:
        z += -1031.8551330566406 * Q.girth2 + 3.04088061321269
    if 0.007520088344 <= Q.girth2 < 0.008678044751:
        z += -2447.5968322753906 * Q.girth2 + 13.687383263622365
    if Q.girth2 >= 0.008678044751:
        z += -970.1920471191406 * Q.girth2 + 0.8663984226948891
    if 0.072690732432 <= Q.mass_over_sum_pt < 0.084751611895:
        z += 108.78565979003906 * Q.mass_over_sum_pt - 7.907709288236311
    if 0.084751611895 <= Q.mass_over_sum_pt < 0.090413827016:
        z += 309.5626983642578 * Q.mass_over_sum_pt - 24.92388693890594
    if Q.mass_over_sum_pt >= 0.090413827016:
        z += -9.125442504882812 * Q.mass_over_sum_pt + 3.889927501687179
    if Q.e2 < 0.024547699839:
        z += -107.90422058105469 * Q.e2 + 2.346524925123763
    if 0.024547699839 <= Q.e2 < 0.038466955721:
        z += -53.19683837890625 * Q.e2 + 1.0035845278479725
    if 0.038466955721 <= Q.e2 < 0.050284641981:
        z += 88.23519897460938 * Q.e2 - 4.436875390560531
    if Q.girth < 0.04081947431:
        z += 99.89858055114746 * Q.girth - 7.802191300423226
    if 0.04081947431 <= Q.girth < 0.087236513197:
        z += 80.2374267578125 * Q.girth - 6.999633338251231
    if Q.pt_7 < 48.71875:
        z += -0.011530628427863121 * Q.pt_7 + 0.5617578037199564
    if Q.width < 0.000561123155:
        z += 2502.7073974609375 * Q.width - 6.510666184818597
    if 0.000561123155 <= Q.width < 0.005590288644:
        z += 1015.34521484375 * Q.width - 5.676072824280755
    if Q.centroid_offset < 0.02076709205:
        z += 43.834442138671875 * Q.centroid_offset - 0.9103138948541978
    if Q.centroid_offset >= 0.031170772021:
        z += 4.447295188903809 * Q.centroid_offset - 0.13862562444341076
    if Q.e2_sq < 0.001101266364:
        z += -1563.093017578125 * Q.e2_sq + 1.7213817640620497
    if Q.mass < 29.644699859619:
        z += -0.044130589812994 * Q.mass + 1.308238089634167
    if Q.mass >= 80.4:
        z += -0.18447263538837433 * Q.mass + 14.831599885225296
    if Q.lam1 < 0.008375572068:
        z += 447.8858642578125 * Q.lam1 - 3.7513003343297737
    if Q.LHA < 0.293190627853:
        z += -13.329102516174316 * Q.LHA + 3.90796793543415
    if Q.girth2_top3 < 0.005884990035:
        z += -78.26609802246094 * Q.girth2_top3 + 0.4605952069405158
    if Q.max_dr < 0.197968879342:
        z += 3.966810464859009 * Q.max_dr - 0.785305022290256
    if Q.planar_flow < 0.195013533663 and Q.width > 0.00752008842:
        z += -2188.5205078125 * (0.195013533663 - Q.planar_flow) * (Q.width - 0.00752008842)
    if Q.girth2_top2 < 0.001056655216 and Q.centroid_offset > 0.006789738266:
        z += 46611.49609375 * (0.001056655216 - Q.girth2_top2) * (Q.centroid_offset - 0.006789738266)
    if Q.planar_flow < 0.195013533663 and Q.pt_6 < 35.28125:
        z += -0.20107802748680115 * (0.195013533663 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.195013533663 and Q.mass_top3 > 23.663861485439:
        z += 0.1201358214020729 * (0.195013533663 - Q.planar_flow) * (Q.mass_top3 - 23.663861485439)
    if Q.planar_flow < 0.195013533663 and Q.sum_pt > 615.875:
        z += 0.035463083535432816 * (0.195013533663 - Q.planar_flow) * (Q.sum_pt - 615.875)
    if Q.e2 < 0.024547699839 and Q.tau21 < 0.446608647704:
        z += 160.6828155517578 * (0.024547699839 - Q.e2) * (0.446608647704 - Q.tau21)
    if Q.girth2 > 0.004372139461 and Q.eccentricity > 0.945820652852:
        z += 8770.5771484375 * (Q.girth2 - 0.004372139461) * (Q.eccentricity - 0.945820652852)
    if Q.lam1 < 0.008375572068 and Q.D2 < 1.122624260187:
        z += -733.45849609375 * (0.008375572068 - Q.lam1) * (1.122624260187 - Q.D2)
    if Q.e2 < 0.050284641981 and Q.D2 < 1.122624260187:
        z += 156.13519287109375 * (0.050284641981 - Q.e2) * (1.122624260187 - Q.D2)
    if Q.centroid_offset < 0.02076709205 and Q.C2 > 0.023843882605:
        z += 1701.0601806640625 * (0.02076709205 - Q.centroid_offset) * (Q.C2 - 0.023843882605)
    if Q.mass > 80.4 and Q.eccentricity > 0.927072033478:
        z += -2.79331636428833 * (Q.mass - 80.4) * (Q.eccentricity - 0.927072033478)
    if Q.pt_7 < 48.71875 and Q.planar_flow < 0.694781820497:
        z += -0.08261273056268692 * (48.71875 - Q.pt_7) * (0.694781820497 - Q.planar_flow)
    if Q.centroid_offset > 0.031170772021 and Q.pt_0 > 376.5:
        z += -9.273751258850098 * (Q.centroid_offset - 0.031170772021) * (Q.pt_0 - 376.5)
    if Q.e2_sq < 0.001101266364 and Q.phi_1 < -0.009460449219:
        z += -6215.2568359375 * (0.001101266364 - Q.e2_sq) * (-0.009460449219 - Q.phi_1)
    if Q.centroid_offset > 0.031170772021 and Q.pt_2 > 56.5:
        z += -1.0859417915344238 * (Q.centroid_offset - 0.031170772021) * (Q.pt_2 - 56.5)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.4330105185508728
    if Q.e2 < 0.016554418951:
        z += -155.87185668945312 * Q.e2 + 2.887251045178409
    if 0.016554418951 <= Q.e2 < 0.024547699839:
        z += -38.39262390136719 * Q.e2 + 0.9424506075623789
    if Q.width < 0.005019718802:
        z += -1307.453857421875 * Q.width + 6.563050710848013
    if Q.LHA < 0.196739721581:
        z += 16.744583129882812 * Q.LHA - 3.294324622963054
    if Q.log_sum_pt >= 6.701242202626:
        z += -8.307278633117676 * Q.log_sum_pt + 55.6690861652214
    if Q.mass < 8.379955863953:
        z += -0.07054489850997925 * Q.mass - 1.1436221635309542
    if 8.379955863953 <= Q.mass < 21.784077072144:
        z += 0.12942178547382355 * Q.mass - 2.819334149576259
    if Q.girth < 0.061086014472:
        z += 42.318607330322266 * Q.girth - 2.5850750598149514
    if Q.sum_pt_top5 >= 658.125:
        z += -0.0015973843401297927 * Q.sum_pt_top5 + 1.0512785688479198
    if Q.centroid_offset < 0.003343241496:
        z += -372.7151794433594 * Q.centroid_offset + 1.2460768541041252
    if Q.max_dr < 0.177304983139:
        z += -6.718590259552002 * Q.max_dr + 1.1912395326877174
    if Q.z_dr_0_0p05 >= 0.847731333971:
        z += -3.6077804565429688 * Q.z_dr_0_0p05 + 3.0584285390996744
    if Q.pt_7 >= 34.53125:
        z += -0.06071065738797188 * Q.pt_7 + 2.096414887928404
    if Q.girth2 < 0.000964142894:
        z += -1081.369873046875 * Q.girth2 + 1.0425950788838267
    if Q.girth2_top5 < 0.000222950415:
        z += 9547.5693359375 * Q.girth2_top5 - 2.12863454568854
    if Q.girth2 < 0.006679471442 and Q.tau21 < 0.501026660204:
        z += -502.7020568847656 * (0.006679471442 - Q.girth2) * (0.501026660204 - Q.tau21)
    if Q.girth2 < 0.006679471442 and Q.centroid_offset < 0.023554160423:
        z += 26070.666015625 * (0.006679471442 - Q.girth2) * (0.023554160423 - Q.centroid_offset)
    if Q.girth2 < 0.006679471442 and Q.planar_flow < 0.40079469091:
        z += 674.8002319335938 * (0.006679471442 - Q.girth2) * (0.40079469091 - Q.planar_flow)
    if Q.width < 0.005019718802 and Q.sum_pt_top2 < 248.125:
        z += 4.552778244018555 * (0.005019718802 - Q.width) * (248.125 - Q.sum_pt_top2)
    if Q.girth2 < 0.006679471442 and Q.n_dr_0p05_0p1 > 0.0:
        z += -50.38880920410156 * (0.006679471442 - Q.girth2) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.LHA < 0.196739721581 and Q.mean_phi > -0.000855675264:
        z += -783.2173461914062 * (0.196739721581 - Q.LHA) * (Q.mean_phi - -0.000855675264)
    if Q.width < 0.005019718802 and Q.z_dr_0p2_0p4 < 0.1009733513:
        z += 6300.97216796875 * (0.005019718802 - Q.width) * (0.1009733513 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.061086014472 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += -175.2258758544922 * (0.061086014472 - Q.girth) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242202626 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += -45.99507522583008 * (Q.log_sum_pt - 6.701242202626) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.width < 0.005019718802 and Q.centroid_offset > 0.006789738266:
        z += -39510.41015625 * (0.005019718802 - Q.width) * (Q.centroid_offset - 0.006789738266)
    if Q.girth < 0.061086014472 and Q.lam1 > 0.00027588256:
        z += -15653.9765625 * (0.061086014472 - Q.girth) * (Q.lam1 - 0.00027588256)
    if Q.width < 0.005019718802 and Q.mass_over_sum_pt_sq > 0.00012320649:
        z += -100211.4609375 * (0.005019718802 - Q.width) * (Q.mass_over_sum_pt_sq - 0.00012320649)
    if Q.max_dr < 0.177304983139 and Q.lam2 < 0.000194798295:
        z += 44376.78515625 * (0.177304983139 - Q.max_dr) * (0.000194798295 - Q.lam2)
    if Q.mass < 15.454033088684 and Q.lam2 < 0.000306123359:
        z += -300.3320007324219 * (15.454033088684 - Q.mass) * (0.000306123359 - Q.lam2)
    if Q.log_sum_pt > 6.701242202626 and Q.pt_7 < 48.71875:
        z += 0.2026367336511612 * (Q.log_sum_pt - 6.701242202626) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242202626 and Q.girth2 < 0.018827652745:
        z += -202.66424560546875 * (Q.log_sum_pt - 6.701242202626) * (0.018827652745 - Q.girth2)
    if Q.log_sum_pt > 6.701242202626 and Q.mass_over_sum_pt_sq < 0.00023679558:
        z += 23924.650390625 * (Q.log_sum_pt - 6.701242202626) * (0.00023679558 - Q.mass_over_sum_pt_sq)
    if Q.log_sum_pt > 6.701242202626 and Q.centroid_offset < 0.023554160423:
        z += 342.672119140625 * (Q.log_sum_pt - 6.701242202626) * (0.023554160423 - Q.centroid_offset)
    if Q.mass < 21.784077072144 and Q.centroid_offset < 0.026856224803:
        z += -2.2714972496032715 * (21.784077072144 - Q.mass) * (0.026856224803 - Q.centroid_offset)
    if Q.LHA < 0.196739721581 and Q.width < 0.000561123155:
        z += 31131.765625 * (0.196739721581 - Q.LHA) * (0.000561123155 - Q.width)
    if Q.LHA < 0.196739721581 and Q.z_6 > 0.02160287394:
        z += -131.79656982421875 * (0.196739721581 - Q.LHA) * (Q.z_6 - 0.02160287394)
    if Q.mass < 29.644699859619 and Q.centroid_offset < 0.023554160423:
        z += -8.376321792602539 * (29.644699859619 - Q.mass) * (0.023554160423 - Q.centroid_offset)
    if Q.e2 < 0.024547699839 and Q.sum_pt > 788.4484375:
        z += 0.25839924812316895 * (0.024547699839 - Q.e2) * (Q.sum_pt - 788.4484375)
    if Q.girth < 0.061086014472 and Q.width < 0.005019718802:
        z += -18206.416015625 * (0.061086014472 - Q.girth) * (0.005019718802 - Q.width)
    if Q.girth < 0.061086014472 and Q.lam2 < 0.000194798295:
        z += 181989.203125 * (0.061086014472 - Q.girth) * (0.000194798295 - Q.lam2)
    if Q.girth < 0.061086014472 and Q.centroid_offset > 0.006789738266:
        z += -2524.905517578125 * (0.061086014472 - Q.girth) * (Q.centroid_offset - 0.006789738266)
    if Q.e2 < 0.024547699839 and Q.centroid_offset < 0.031170772021:
        z += 4740.8525390625 * (0.024547699839 - Q.e2) * (0.031170772021 - Q.centroid_offset)
    if Q.girth2 < 0.006679471442 and Q.centroid_offset > 0.018377780003:
        z += -20906.5234375 * (0.006679471442 - Q.girth2) * (Q.centroid_offset - 0.018377780003)
    if Q.LHA < 0.196739721581 and Q.lam2 < 0.000306123359:
        z += -86292.015625 * (0.196739721581 - Q.LHA) * (0.000306123359 - Q.lam2)
    if Q.mass < 15.454033088684 and Q.z_7 < 0.058613700176:
        z += -2.2242581844329834 * (15.454033088684 - Q.mass) * (0.058613700176 - Q.z_7)
    if Q.LHA < 0.196739721581 and Q.z_7 > 0.016858545121:
        z += 259.0406494140625 * (0.196739721581 - Q.LHA) * (Q.z_7 - 0.016858545121)
    if Q.mass < 21.784077072144 and Q.centroid_offset > 0.016278845848:
        z += -13.785367965698242 * (21.784077072144 - Q.mass) * (Q.centroid_offset - 0.016278845848)
    if Q.z_dr_0_0p05 > 0.847731333971 and Q.lam2 < 0.000537286005:
        z += -18788.77734375 * (Q.z_dr_0_0p05 - 0.847731333971) * (0.000537286005 - Q.lam2)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.6382436752319336
    if Q.girth < 0.054649224505:
        z += 86.57256317138672 * Q.girth - 4.731123440726408
    if Q.e2 < 0.020459658932:
        z += -75.24676132202148 * Q.e2 + 1.0498762399663053
    if 0.020459658932 <= Q.e2 < 0.032346998155:
        z += 41.190616607666016 * Q.e2 - 1.332392799411485
    if Q.mass < 29.644699859619:
        z += 0.13031964749097824 * Q.mass - 5.368745781881035
    if 29.644699859619 <= Q.mass < 41.377904891968:
        z += 0.03844698518514633 * Q.mass - 2.645208282520515
    if 41.377904891968 <= Q.mass < 53.332374954224:
        z += 0.08819735050201416 * Q.mass - 4.7037741669425355
    if Q.lam2 >= 0.001130644719:
        z += 849.4024658203125 * Q.lam2 - 0.9603724122853143
    if Q.width < 0.000172198326:
        z += -8057.7003173828125 * Q.width + 11.07278105166721
    if 0.000172198326 <= Q.width < 0.006096650059:
        z += -1634.7940673828125 * Q.width + 9.966767347362273
    if Q.girth2 < 0.000964142894:
        z += -5628.297821044922 * Q.girth2 + 11.671018300474625
    if 0.000964142894 <= Q.girth2 < 0.004372139461:
        z += -1382.6923522949219 * Q.girth2 + 7.577647957051774
    if 0.004372139461 <= Q.girth2 < 0.007520088344:
        z += -486.7690734863281 * Q.girth2 + 3.6605464357442155
    if Q.girth2 >= 0.018827652745:
        z += 231.5805206298828 * Q.girth2 - 4.360117624925742
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 1.044736385345459 * Q.n_dr_0p2_0p4 - 1.044736385345459
    if Q.max_dr < 0.15984864831:
        z += -8.442212104797363 * Q.max_dr + 1.8706821959311815
    if 0.15984864831 <= Q.max_dr < 0.221586732566:
        z += -11.816805839538574 * Q.max_dr + 2.410106443024959
    if Q.max_dr >= 0.221586732566:
        z += -3.374593734741211 * Q.max_dr + 0.5394242470937772
    if Q.C2 >= 0.051192347892:
        z += 23.139305114746094 * Q.C2 - 1.184555357413217
    if Q.centroid_offset < 0.018377780003:
        z += -194.10372924804688 * Q.centroid_offset + 3.567195633882482
    if Q.log_sum_pt >= 6.377722943814:
        z += -5.507915019989014 * Q.log_sum_pt + 35.12795599556168
    if Q.mass_over_sum_pt < 0.076373631775:
        z += 38.707889556884766 * Q.mass_over_sum_pt - 2.9562621038048853
    if Q.lam1 < 0.001503553356:
        z += 822.0446166992188 * Q.lam1 - 1.235987942219844
    if Q.pt_5 < 35.5:
        z += -0.06326994299888611 * Q.pt_5 + 2.246082976460457
    if Q.mass < 53.332374954224 and Q.centroid_offset < 0.026856224803:
        z += 4.124216079711914 * (53.332374954224 - Q.mass) * (0.026856224803 - Q.centroid_offset)
    if Q.mass < 53.332374954224 and Q.lam1 < 0.000872228216:
        z += -63.83016586303711 * (53.332374954224 - Q.mass) * (0.000872228216 - Q.lam1)
    if Q.girth2 > 0.018827652745 and Q.planar_flow < 0.40079469091:
        z += -492.7013244628906 * (Q.girth2 - 0.018827652745) * (0.40079469091 - Q.planar_flow)
    if Q.e2 < 0.032346998155 and Q.dr01 < 0.055953954317:
        z += 402.8077392578125 * (0.032346998155 - Q.e2) * (0.055953954317 - Q.dr01)
    if Q.mass < 53.332374954224 and Q.log_sum_pt < 6.842716632804:
        z += 0.22528095543384552 * (53.332374954224 - Q.mass) * (6.842716632804 - Q.log_sum_pt)
    if Q.width < 0.006096650059 and Q.C2 > 0.030867108516:
        z += -10697.7900390625 * (0.006096650059 - Q.width) * (Q.C2 - 0.030867108516)
    if Q.width < 0.006096650059 and Q.centroid_offset > 0.003343241496:
        z += -52030.19921875 * (0.006096650059 - Q.width) * (Q.centroid_offset - 0.003343241496)
    if Q.centroid_offset < 0.018377780003 and Q.z_5 > 0.036727111752:
        z += -991.2238159179688 * (0.018377780003 - Q.centroid_offset) * (Q.z_5 - 0.036727111752)
    if Q.mass < 29.644699859619 and Q.mean_phi2 < 0.002127561159:
        z += -15.360997200012207 * (29.644699859619 - Q.mass) * (0.002127561159 - Q.mean_phi2)
    if Q.mass < 53.332374954224 and Q.planar_flow < 0.322073846732:
        z += 0.3322337567806244 * (53.332374954224 - Q.mass) * (0.322073846732 - Q.planar_flow)
    if Q.centroid_offset < 0.018377780003 and Q.pt_4 > 47.34375:
        z += 2.5802652835845947 * (0.018377780003 - Q.centroid_offset) * (Q.pt_4 - 47.34375)
    if Q.centroid_offset < 0.018377780003 and Q.z_4 > 0.047491459878:
        z += -2388.785888671875 * (0.018377780003 - Q.centroid_offset) * (Q.z_4 - 0.047491459878)
    if Q.e2 < 0.020459658932 and Q.eccentricity > 0.903125533696:
        z += -666.1741333007812 * (0.020459658932 - Q.e2) * (Q.eccentricity - 0.903125533696)
    if Q.C2 > 0.051192347892 and Q.eccentricity < 0.620723099573:
        z += 202.7896728515625 * (Q.C2 - 0.051192347892) * (0.620723099573 - Q.eccentricity)
    if Q.mass < 41.377904891968 and Q.planar_flow < 0.322073846732:
        z += -0.40454405546188354 * (41.377904891968 - Q.mass) * (0.322073846732 - Q.planar_flow)
    if Q.e2 < 0.032346998155 and Q.tau21 < 0.446608647704:
        z += -197.85508728027344 * (0.032346998155 - Q.e2) * (0.446608647704 - Q.tau21)
    if Q.girth2 > 0.018827652745 and Q.pt_7 < 29.0421875:
        z += 63.05718231201172 * (Q.girth2 - 0.018827652745) * (29.0421875 - Q.pt_7)
    if Q.lam2 > 0.001130644719 and Q.mass_top2 > 16.308019673264:
        z += 12.805971145629883 * (Q.lam2 - 0.001130644719) * (Q.mass_top2 - 16.308019673264)
    if Q.e2 < 0.020459658932 and Q.pt_7 < 53.4375:
        z += 0.593216598033905 * (0.020459658932 - Q.e2) * (53.4375 - Q.pt_7)
    if Q.girth2 > 0.018827652745 and Q.mean_eta < 0.02644207105:
        z += -1862.0118408203125 * (Q.girth2 - 0.018827652745) * (0.02644207105 - Q.mean_eta)
    if Q.pt_5 < 35.5 and Q.min_pair_mass > 0.173071536962:
        z += -0.02553240768611431 * (35.5 - Q.pt_5) * (Q.min_pair_mass - 0.173071536962)
    if Q.C2 > 0.051192347892 and Q.n_dr_0p05_0p1 > 1.0:
        z += -9.165243148803711 * (Q.C2 - 0.051192347892) * (Q.n_dr_0p05_0p1 - 1.0)
    return max(0.0, z)


def neuron_10(Q):
    z = -1.7323570251464844
    z += 67.38975524902344 * Q.e2
    if Q.lam2 >= 0.000194798295:
        z += 1562.142578125 * Q.lam2 - 0.3043027107656543
    if Q.LHA >= 0.303313749495:
        z += -11.272607803344727 * Q.LHA + 3.4191369394190847
    if Q.log_sum_pt < 6.267538488641:
        z += 2.553096294403076 * Q.log_sum_pt - 16.00162929037799
    if Q.log_sum_pt >= 6.701242202626:
        z += -17.604631423950195 * Q.log_sum_pt + 117.9728990598509
    if Q.C2 >= 0.051192347892:
        z += 28.143735885620117 * Q.C2 - 1.4407439184372297
    if Q.n_dr_0p2_0p4 >= 2.0:
        z += 0.43542203307151794 * Q.n_dr_0p2_0p4 - 0.8708440661430359
    if Q.lam1 < 0.004183811014:
        z += 630.4684295654297 * Q.lam1 - 3.0216922843012326
    if 0.004183811014 <= Q.lam1 < 0.006506575659:
        z += 165.29075622558594 * Q.lam1 - 1.0754768111151003
    if Q.mass_over_sum_pt < 0.090413827016:
        z += -37.67039108276367 * Q.mass_over_sum_pt + 3.405924222982063
    if Q.n_dr_0p05_0p1 < 3.0:
        z += -0.24841953814029694 * Q.n_dr_0p05_0p1 + 0.7452586144208908
    if Q.tau32 < 0.269169217348:
        z += -10.949302673339844 * Q.tau32 + 2.9472152310892494
    if Q.tau21 < 0.391541349888:
        z += -1.4110872745513916 * Q.tau21 + 0.5524990162876308
    if Q.girth2_top2 < 0.002412890926:
        z += -537.9927978515625 * Q.girth2_top2 + 1.2981179401893876
    if Q.mass >= 15.454033088684:
        z += 0.03868935629725456 * Q.mass - 0.5979065923976566
    if Q.max_dr >= 0.121680960059:
        z += -4.816457748413086 * Q.max_dr + 0.5860712029105138
    if Q.girth2 < 0.0016538364:
        z += 1402.7121276855469 * Q.girth2 - 1.2475144124350457
    if 0.0016538364 <= Q.girth2 < 0.006679471442:
        z += -213.37442016601562 * Q.girth2 + 1.4252283459522104
    if Q.sum_pt >= 813.415625:
        z += 0.01101782824844122 * Q.sum_pt - 8.962073650848469
    if Q.lam2 > 0.000194798295 and Q.planar_flow > 0.012569162668:
        z += -587.1337280273438 * (Q.lam2 - 0.000194798295) * (Q.planar_flow - 0.012569162668)
    if Q.centroid_offset > 0.00231612516 and Q.pt_7 < 34.53125:
        z += 0.658306360244751 * (Q.centroid_offset - 0.00231612516) * (34.53125 - Q.pt_7)
    if Q.lam2 > 0.000194798295 and Q.n_pt_above_50 < 8.0:
        z += -48.846946716308594 * (Q.lam2 - 0.000194798295) * (8.0 - Q.n_pt_above_50)
    if Q.lam2 > 0.000194798295 and Q.tau21 < 0.501026660204:
        z += 1977.1407470703125 * (Q.lam2 - 0.000194798295) * (0.501026660204 - Q.tau21)
    if Q.LHA > 0.303313749495 and Q.tau21 < 0.553068161011:
        z += -39.403076171875 * (Q.LHA - 0.303313749495) * (0.553068161011 - Q.tau21)
    if Q.eccentricity > 0.903125533696 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += 225.6640167236328 * (Q.eccentricity - 0.903125533696) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.tau32 < 0.269169217348 and Q.n_dr_0p2_0p4 < 2.0:
        z += -3.8826003074645996 * (0.269169217348 - Q.tau32) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.tau21 < 0.391541349888 and Q.pt_4 > 39.8125:
        z += -0.07969776540994644 * (0.391541349888 - Q.tau21) * (Q.pt_4 - 39.8125)
    if Q.C2 > 0.051192347892 and Q.dr_7 < 0.222994708167:
        z += -27.198741912841797 * (Q.C2 - 0.051192347892) * (0.222994708167 - Q.dr_7)
    if Q.lam2 > 0.000194798295 and Q.D2 < 2.055451202393:
        z += -265.931884765625 * (Q.lam2 - 0.000194798295) * (2.055451202393 - Q.D2)
    if Q.lam1 < 0.004183811014 and Q.dr_7 < 0.175465903809:
        z += 1220.8304443359375 * (0.004183811014 - Q.lam1) * (0.175465903809 - Q.dr_7)
    if Q.tau21 < 0.391541349888 and Q.z_4 > 0.075444822386:
        z += 79.0702133178711 * (0.391541349888 - Q.tau21) * (Q.z_4 - 0.075444822386)
    if Q.lam1 < 0.004183811014 and Q.sum_pt > 988.4078125:
        z += -9.046772956848145 * (0.004183811014 - Q.lam1) * (Q.sum_pt - 988.4078125)
    if Q.lam1 < 0.004183811014 and Q.log_sum_pt > 6.701242202626:
        z += 3908.609375 * (0.004183811014 - Q.lam1) * (Q.log_sum_pt - 6.701242202626)
    if Q.mass > 15.454033088684 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.0027797119691967964 * (Q.mass - 15.454033088684) * (2.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.618403434753418
    if Q.planar_flow < 0.253403707141:
        z += -8.610761642456055 * Q.planar_flow + 2.1819989215058904
    if Q.LHA < 0.154689112391:
        z += 32.07748794555664 * Q.LHA - 4.962038138031159
    if Q.girth < 0.076081777364:
        z += 67.29173278808594 * Q.girth - 5.870296135416857
    if 0.076081777364 <= Q.girth < 0.087236513197:
        z += 12.826702117919922 * Q.girth - 1.7264997978458547
    if Q.girth >= 0.087236513197:
        z += -54.465030670166016 * Q.girth + 4.143796337571002
    if Q.centroid_offset < 0.014379521101:
        z += -82.97048091888428 * Q.centroid_offset + 4.336524919861032
    if 0.014379521101 <= Q.centroid_offset < 0.049903668404:
        z += -129.92790508270264 * Q.centroid_offset + 5.011750191473265
    if Q.centroid_offset >= 0.049903668404:
        z += -43.02998638153076 * Q.centroid_offset + 0.6752252716122333
    if Q.width < 0.003562611091:
        z += -635.2930908203125 * Q.width + 7.812918511341373
    if 0.003562611091 <= Q.width < 0.008678044951:
        z += -1084.876953125 * Q.width + 9.41461096552267
    if Q.e2_sq < 0.006390124748:
        z += 461.1095275878906 * Q.e2_sq - 2.9465474037779686
    if Q.n_dr_0_0p05 < 7.0:
        z += 0.06590835750102997 * Q.n_dr_0_0p05 - 0.4613585025072098
    if Q.girth2 < 0.006679471442:
        z += -940.830322265625 * Q.girth2 + 7.763188008804992
    if 0.006679471442 <= Q.girth2 < 0.013238675334:
        z += -225.475341796875 * Q.girth2 + 2.9849948458715083
    if Q.e2 < 0.04447356835:
        z += 36.674808502197266 * Q.e2 - 1.6310596026456312
    if Q.lam1 < 0.004839980301:
        z += 503.5763397216797 * Q.lam1 - 3.61709485818639
    if 0.004839980301 <= Q.lam1 < 0.008375572068:
        z += 333.6910400390625 * Q.lam1 - 2.7948533542930414
    if Q.mass_top5 < 9.257203159811:
        z += -0.038133859634399414 * Q.mass_top5 + 0.35301288590335145
    if Q.mass < 15.454033088684:
        z += -0.047983553260564804 * Q.mass + 0.7415394198013995
    if Q.pt_7 < 29.0421875:
        z += 0.05266731604933739 * Q.pt_7 - 1.5295740678266156
    if Q.sum_pt_top5 < 687.4375:
        z += -0.006858064327389002 * Q.sum_pt_top5 + 4.714490596059477
    if Q.C2 < 0.035786485299:
        z += 24.042226791381836 * Q.C2 - 0.86038679562501
    if Q.max_dr < 0.111761856824:
        z += 3.566955327987671 * Q.max_dr - 0.06295230527661266
    if 0.111761856824 <= Q.max_dr < 0.221586732566:
        z += -3.056659460067749 * Q.max_dr + 0.6773151823233663
    if Q.n_dr_0p1_0p2 < 3.0:
        z += 0.26168859004974365 * Q.n_dr_0p1_0p2 - 0.785065770149231
    if Q.planar_flow < 0.253403707141 and Q.width > 0.006096650059:
        z += -1129.4271240234375 * (0.253403707141 - Q.planar_flow) * (Q.width - 0.006096650059)
    if Q.planar_flow < 0.253403707141 and Q.mass < 69.611351776123:
        z += -0.1234804317355156 * (0.253403707141 - Q.planar_flow) * (69.611351776123 - Q.mass)
    if Q.planar_flow < 0.253403707141 and Q.max_dr > 0.102758520097:
        z += -47.76771926879883 * (0.253403707141 - Q.planar_flow) * (Q.max_dr - 0.102758520097)
    if Q.centroid_offset < 0.049903668404 and Q.pt_7 < 48.71875:
        z += 0.3606238067150116 * (0.049903668404 - Q.centroid_offset) * (48.71875 - Q.pt_7)
    if Q.centroid_offset < 0.049903668404 and Q.n_dr_0p05_0p1 > 2.0:
        z += 3.431257724761963 * (0.049903668404 - Q.centroid_offset) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.centroid_offset < 0.049903668404 and Q.log_sum_pt < 6.804164030582:
        z += -115.19680786132812 * (0.049903668404 - Q.centroid_offset) * (6.804164030582 - Q.log_sum_pt)
    if Q.e2_sq < 0.006390124748 and Q.D2 < 1.332146394253:
        z += 132.60784912109375 * (0.006390124748 - Q.e2_sq) * (1.332146394253 - Q.D2)
    if Q.LHA < 0.154689112391 and Q.z_7 < 0.028070914944:
        z += 1026.357421875 * (0.154689112391 - Q.LHA) * (0.028070914944 - Q.z_7)
    if Q.girth > 0.076081777364 and Q.n_pt_above_50 < 7.0:
        z += 10.635859489440918 * (Q.girth - 0.076081777364) * (7.0 - Q.n_pt_above_50)
    if Q.pt_7 < 29.0421875 and Q.mass_top2 < 22.844978847276:
        z += 0.0019490555860102177 * (29.0421875 - Q.pt_7) * (22.844978847276 - Q.mass_top2)
    if Q.e2 < 0.04447356835 and Q.pt_5 > 43.0625:
        z += -0.23641669750213623 * (0.04447356835 - Q.e2) * (Q.pt_5 - 43.0625)
    if Q.pt_7 < 29.0421875 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.024639900773763657 * (29.0421875 - Q.pt_7) * (2.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_12(Q):
    z = -1.3594197034835815
    if Q.girth2 >= 0.018827652745:
        z += 195.4222412109375 * Q.girth2 - 3.6793420961691594
    if Q.mass >= 91.19:
        z += 0.08246996253728867 * Q.mass - 7.520435883775353
    if Q.e2 >= 0.063441075385:
        z += -115.92047119140625 * Q.e2 + 7.354119351518725
    if Q.mean_phi >= 0.026127964072:
        z += 24.440589904785156 * Q.mean_phi - 0.6385828549307124
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.12673042714595795 * Q.n_dr_0p2_0p4 - 0.12673042714595795
    if Q.girth2 > 0.018827652745 and Q.lam2 > 0.000537286005:
        z += 15108.0244140625 * (Q.girth2 - 0.018827652745) * (Q.lam2 - 0.000537286005)
    if Q.girth2 > 0.018827652745 and Q.pt_7 > 15.55390625:
        z += 6.701659202575684 * (Q.girth2 - 0.018827652745) * (Q.pt_7 - 15.55390625)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.551501750946045
    if Q.girth < 0.148408418149:
        z += -46.41705322265625 * Q.girth + 6.888681443912357
    if Q.lam1 < 0.006506575659:
        z += -497.8289337158203 * Q.lam1 + 4.998379238659054
    if 0.006506575659 <= Q.lam1 < 0.016433749775:
        z += -177.2123260498047 * Q.lam1 + 2.912263023348204
    if Q.sum_pt_top5 < 531.1875:
        z += -0.002590498421341181 * Q.sum_pt_top5 + 1.3760403801861685
    if 658.125 <= Q.sum_pt_top5 < 839.9546875:
        z += 0.0010897907195612788 * Q.sum_pt_top5 - 0.7172185173112666
    if 839.9546875 <= Q.sum_pt_top5 < 902.40625:
        z += 0.01766071270685643 * Q.sum_pt_top5 - 14.636042116736643
    if Q.sum_pt_top5 >= 902.40625:
        z += 0.011628642561845481 * Q.sum_pt_top5 - 9.192664317440357
    if Q.lam2 < 0.000306123359:
        z += -2995.64990234375 * Q.lam2 + 0.9170384104934908
    if Q.e2 < 0.050284641981:
        z += 78.48339080810547 * Q.e2 - 3.94650920824049
    if Q.z_7 < 0.028070914944:
        z += 93.78596496582031 * Q.z_7 - 2.6326578454965057
    if Q.tau21 < 0.501026660204:
        z += 0.5941035747528076 * Q.tau21 - 0.29766172987365663
    if Q.C2 >= 0.067292226106:
        z += -56.29425811767578 * Q.C2 + 3.7881659457241645
    if Q.z_6 < 0.067272114405:
        z += -19.926326751708984 * Q.z_6 + 1.3404861329123787
    if Q.centroid_offset < 0.037760993714:
        z += -43.4168815612793 * Q.centroid_offset + 1.63946459171695
    if Q.LHA >= 0.09323897448:
        z += -4.315232276916504 * Q.LHA + 0.4023478321426902
    if Q.width < 0.00752008842:
        z += 231.8017578125 * Q.width - 0.9066604603992556
    if 0.00752008842 <= Q.width < 0.013238675006:
        z += -146.27902221679688 * Q.width + 1.9365404353236273
    if Q.sum_pt < 763.825:
        z += 0.0020022292155772448 * Q.sum_pt - 1.529352730588289
    if Q.sum_pt >= 988.4078125:
        z += -0.042725589126348495 * Q.sum_pt + 42.2303060861479
    if Q.pt_7 < 25.578125:
        z += 0.19033272564411163 * Q.pt_7 - 4.868354248115793
    if Q.girth < 0.148408418149 and Q.log_sum_pt < 6.804164030582:
        z += -37.56578826904297 * (0.148408418149 - Q.girth) * (6.804164030582 - Q.log_sum_pt)
    if Q.girth < 0.148408418149 and Q.pt_7 < 38.53125:
        z += -1.0888617038726807 * (0.148408418149 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.z_7 > 0.023207568189:
        z += -0.31147146224975586 * (Q.sum_pt_top5 - 658.125) * (Q.z_7 - 0.023207568189)
    if Q.lam2 < 0.000306123359 and Q.centroid_offset < 0.049903668404:
        z += -91305.0703125 * (0.000306123359 - Q.lam2) * (0.049903668404 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 43.5:
        z += 0.0005559679702855647 * (Q.sum_pt_top5 - 658.125) * (43.5 - Q.pt_7)
    if Q.lam1 < 0.016433749775 and Q.pt_6 < 56.53125:
        z += -1.667509913444519 * (0.016433749775 - Q.lam1) * (56.53125 - Q.pt_6)
    if Q.e2 < 0.050284641981 and Q.pt_dispersion > 0.396830244362:
        z += 100.75765991210938 * (0.050284641981 - Q.e2) * (Q.pt_dispersion - 0.396830244362)
    if Q.lam1 < 0.016433749775 and Q.centroid_offset < 0.037760993714:
        z += -1508.5775146484375 * (0.016433749775 - Q.lam1) * (0.037760993714 - Q.centroid_offset)
    if Q.sum_pt_top5 > 902.40625 and Q.D2 < 3.885568320751:
        z += 0.007977412082254887 * (Q.sum_pt_top5 - 902.40625) * (3.885568320751 - Q.D2)
    if Q.sum_pt_top5 > 839.9546875 and Q.z_dr_0p1_0p2 < 0.15855820179:
        z += 0.03161969035863876 * (Q.sum_pt_top5 - 839.9546875) * (0.15855820179 - Q.z_dr_0p1_0p2)
    if Q.tau21 < 0.501026660204 and Q.max_dr > 0.015595615841:
        z += -16.137348175048828 * (0.501026660204 - Q.tau21) * (Q.max_dr - 0.015595615841)
    if Q.sum_pt < 763.825 and Q.z_4 < 0.037477688199:
        z += 6.592033863067627 * (763.825 - Q.sum_pt) * (0.037477688199 - Q.z_4)
    if Q.sum_pt > 988.4078125 and Q.n_pt_above_50 > 6.0:
        z += 0.01756887696683407 * (Q.sum_pt - 988.4078125) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt > 988.4078125 and Q.D2 < 3.885568320751:
        z += -0.003981756512075663 * (Q.sum_pt - 988.4078125) * (3.885568320751 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.9585873484611511
    if Q.planar_flow < 0.111513564951:
        z += -6.083240985870361 * Q.planar_flow + 0.6783638887904397
    if Q.z_dr_0p05_0p1 < 0.588259786367:
        z += -2.098081588745117 * Q.z_dr_0p05_0p1 + 1.2342170271757387
    if Q.z_dr_0p05_0p1 >= 0.750909513235:
        z += -5.301498889923096 * Q.z_dr_0p05_0p1 + 3.9809459508480445
    if 0.002464291268 <= Q.lam1 < 0.004183811014:
        z += 297.7964782714844 * Q.lam1 - 0.7338572610455707
    if 0.004183811014 <= Q.lam1 < 0.00543336053:
        z += 87.86642456054688 * Q.lam1 + 0.14445040983986113
    if 0.00543336053 <= Q.lam1 < 0.005954149834:
        z += 29.443862915039062 * Q.lam1 + 0.4618812503460552
    if 0.005954149834 <= Q.lam1 < 0.007330079875:
        z += 234.65985107421875 * Q.lam1 - 0.7600054914860706
    if 0.007330079875 <= Q.lam1 < 0.008375572068:
        z += 61.81787109375 * Q.lam1 + 0.5069400275239166
    if Q.lam1 >= 0.008375572068:
        z += -162.5731964111328 * Q.lam1 + 2.3863435848265153
    if Q.mass < 76.655700683594:
        z += 0.014777259901165962 * Q.mass - 1.132761211907454
    if Q.girth2 < 0.004372139461:
        z += -577.1469421386719 * Q.girth2 + 8.338273741072324
    if 0.004372139461 <= Q.girth2 < 0.013238675334:
        z += -655.8262329101562 * Q.girth2 + 8.682270573017824
    if Q.width < 0.00752008842:
        z += 750.3577880859375 * Q.width - 4.422670897207139
    if 0.00752008842 <= Q.width < 0.008678044951:
        z += -1053.6544189453125 * Q.width + 9.143660410427207
    if Q.e2 < 0.035560912266:
        z += -103.33319091796875 * Q.e2 + 4.0792649207347385
    if 0.035560912266 <= Q.e2 < 0.038466955721:
        z += -139.24168395996094 * Q.e2 + 5.356203691405294
    if Q.mass_over_sum_pt_sq < 0.011660904657:
        z += 393.81146240234375 * Q.mass_over_sum_pt_sq - 4.59219791590747
    if Q.girth < 0.087236513197:
        z += 89.24246215820312 * Q.girth - 7.78520122779686
    if Q.D2 < 0.74595130682:
        z += 0.11732196807861328 * Q.D2 - 0.6965533484843605
    if 0.74595130682 <= Q.D2 < 1.122624260187:
        z += 1.6168850660324097 * Q.D2 - 1.8151544010620426
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.13957715034484863 * Q.n_dr_0p05_0p1 - 0.6978857517242432
    if Q.e2_sq < 0.017162483186:
        z += -111.2732925415039 * Q.e2_sq + 1.90972601229442
    if Q.max_dr < 0.177304983139:
        z += 21.646635055541992 * Q.max_dr - 3.8380562635389595
    if Q.planar_flow < 0.111513564951 and Q.centroid_offset > 0.018377780003:
        z += -1331.9940185546875 * (0.111513564951 - Q.planar_flow) * (Q.centroid_offset - 0.018377780003)
    if Q.planar_flow < 0.111513564951 and Q.max_dr < 0.15984864831:
        z += -209.09561157226562 * (0.111513564951 - Q.planar_flow) * (0.15984864831 - Q.max_dr)
    if Q.planar_flow < 0.111513564951 and Q.sum_pt < 739.5:
        z += -0.048383958637714386 * (0.111513564951 - Q.planar_flow) * (739.5 - Q.sum_pt)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.C2 < 0.067292226106:
        z += -82.44739532470703 * (0.588259786367 - Q.z_dr_0p05_0p1) * (0.067292226106 - Q.C2)
    if Q.lam1 > 0.007330079875 and Q.max_dr < 0.13261153996:
        z += 43514.51953125 * (Q.lam1 - 0.007330079875) * (0.13261153996 - Q.max_dr)
    if Q.lam1 > 0.00543336053 and Q.max_dr < 0.15984864831:
        z += 8852.8603515625 * (Q.lam1 - 0.00543336053) * (0.15984864831 - Q.max_dr)
    if Q.planar_flow < 0.111513564951 and Q.centroid_offset > 0.009480684835:
        z += 1364.820068359375 * (0.111513564951 - Q.planar_flow) * (Q.centroid_offset - 0.009480684835)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.eccentricity > 0.970449631164:
        z += 39.185340881347656 * (0.588259786367 - Q.z_dr_0p05_0p1) * (Q.eccentricity - 0.970449631164)
    if Q.z_dr_0p05_0p1 > 0.750909513235 and Q.n_dr_0p2_0p4 < 1.0:
        z += 5.597701072692871 * (Q.z_dr_0p05_0p1 - 0.750909513235) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.n_dr_0p1_0p2 < 3.0:
        z += 0.30681881308555603 * (0.588259786367 - Q.z_dr_0p05_0p1) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 76.655700683594 and Q.D2 < 0.74595130682:
        z += -0.06831799447536469 * (76.655700683594 - Q.mass) * (0.74595130682 - Q.D2)
    if Q.width < 0.00752008842 and Q.planar_flow < 0.111513564951:
        z += -6231.85302734375 * (0.00752008842 - Q.width) * (0.111513564951 - Q.planar_flow)
    if Q.girth2 < 0.013238675334 and Q.eccentricity > 0.970449631164:
        z += 5885.08056640625 * (0.013238675334 - Q.girth2) * (Q.eccentricity - 0.970449631164)
    if Q.width < 0.00752008842 and Q.D2 < 1.002470755577:
        z += 529.3541259765625 * (0.00752008842 - Q.width) * (1.002470755577 - Q.D2)
    if Q.e2 < 0.038466955721 and Q.D2 < 1.002470755577:
        z += 199.25184631347656 * (0.038466955721 - Q.e2) * (1.002470755577 - Q.D2)
    if Q.D2 < 1.122624260187 and Q.centroid_offset < 0.031170772021:
        z += 39.43013000488281 * (1.122624260187 - Q.D2) * (0.031170772021 - Q.centroid_offset)
    if Q.z_dr_0p05_0p1 > 0.750909513235 and Q.eccentricity > 0.984196588116:
        z += 160.78482055664062 * (Q.z_dr_0p05_0p1 - 0.750909513235) * (Q.eccentricity - 0.984196588116)
    if Q.width < 0.00752008842 and Q.log_sum_pt < 6.804164030582:
        z += 333.97674560546875 * (0.00752008842 - Q.width) * (6.804164030582 - Q.log_sum_pt)
    if Q.width < 0.008678044951 and Q.D2 < 1.002470755577:
        z += -1466.0433349609375 * (0.008678044951 - Q.width) * (1.002470755577 - Q.D2)
    if Q.girth2 < 0.013238675334 and Q.D2 < 1.002470755577:
        z += 155.06509399414062 * (0.013238675334 - Q.girth2) * (1.002470755577 - Q.D2)
    if Q.width < 0.00752008842 and Q.n_dr_0p1_0p2 < 3.0:
        z += 134.75198364257812 * (0.00752008842 - Q.width) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.girth2 < 0.004372139461 and Q.n_dr_0p2_0p4 < 1.0:
        z += -263.0376892089844 * (0.004372139461 - Q.girth2) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.girth2 < 0.013238675334 and Q.n_dr_0p1_0p2 < 3.0:
        z += -49.104454040527344 * (0.013238675334 - Q.girth2) * (3.0 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_15(Q):
    z = -1.9570142030715942
    if Q.tau21 < 0.23799610585:
        z += -5.739184856414795 * Q.tau21 + 1.3659036465800125
    if Q.width < 0.006096650059:
        z += 1182.2533111572266 * Q.width - 5.010576582509551
    if 0.006096650059 <= Q.width < 0.006679471358:
        z += 1316.9964599609375 * Q.width - 5.832058408613541
    if 0.006679471358 <= Q.width < 0.00752008842:
        z += -2.1453857421875 * Q.width + 2.979111766899739
    if 0.00752008842 <= Q.width < 0.013238675006:
        z += -518.1312255859375 * Q.width + 6.859370905992698
    if Q.girth2 < 0.018827652745:
        z += -133.21267700195312 * Q.girth2 + 2.508082023824621
    if Q.lam1 < 0.006506575659:
        z += 2.4473419189453125 * Q.lam1 - 2.0681099213683156
    if 0.006506575659 <= Q.lam1 < 0.008375572068:
        z += 711.6178741455078 * Q.lam1 - 6.682381644433742
    if 0.008375572068 <= Q.lam1 < 0.012003726523:
        z += 199.0474395751953 * Q.lam1 - 2.389311029764012
    if Q.lam2 < 0.000306123359:
        z += 2882.7284545898438 * Q.lam2 - 1.3818683199326085
    if 0.000306123359 <= Q.lam2 < 0.001130644719:
        z += 605.6820678710938 * Q.lam2 - 0.6848112314314517
    if Q.e2_sq < 0.008168570676:
        z += 676.1878814697266 * Q.e2_sq - 6.3114808523462695
    if 0.008168570676 <= Q.e2_sq < 0.011657374702:
        z += 225.86317443847656 * Q.e2_sq - 2.6329716558125096
    if Q.mass_over_sum_pt_sq < 0.007182835724:
        z += -555.727783203125 * Q.mass_over_sum_pt_sq + 3.9917013740107334
    if Q.e2 < 0.024547699839:
        z += -372.6427836418152 * Q.e2 + 11.047188274525078
    if 0.024547699839 <= Q.e2 < 0.041109715588:
        z += -121.42931318283081 * Q.e2 + 4.880475406184436
    if 0.041109715588 <= Q.e2 < 0.063441075385:
        z += 4.9907002449035645 * Q.e2 - 0.316615390460865
    if Q.z_dr_0p05_0p1 >= 0.750909513235:
        z += -8.978464126586914 * Q.z_dr_0p05_0p1 + 6.742014126893289
    if Q.z_dr_0p1_0p2 < 0.328461505473:
        z += -3.9758412837982178 * Q.z_dr_0p1_0p2 + 1.3059108135980677
    if Q.n_dr_0_0p05 < 2.0:
        z += -0.30176523327827454 * Q.n_dr_0_0p05 + 0.6035304665565491
    if Q.log_sum_pt >= 6.896095378249:
        z += 38.43477249145508 * Q.log_sum_pt - 265.0498569423752
    if Q.mass_over_sum_pt < 0.06813910019:
        z += 23.045787811279297 * Q.mass_over_sum_pt - 1.5703192446302408
    if Q.mass < 36.229410171509:
        z += 0.02879525162279606 * Q.mass - 1.0432349820340887
    if Q.D2 < 0.74595130682:
        z += 1.5350931882858276 * Q.D2 - 1.1451047698922934
    if Q.n_dr_0p05_0p1 >= 5.0:
        z += 0.24260537326335907 * Q.n_dr_0p05_0p1 - 1.2130268663167953
    if Q.LHA >= 0.346713497427:
        z += -61.869537353515625 * Q.LHA + 21.451003680027817
    if Q.girth >= 0.033604209498:
        z += 31.87446403503418 * Q.girth - 1.0711161670697549
    if Q.girth2_top3 < 0.002151567843:
        z += 596.3739624023438 * Q.girth2_top3 - 1.2831390399073739
    if Q.girth2_top2 < 0.007639643088:
        z += -40.99702835083008 * Q.girth2_top2 + 0.31320266426895904
    if Q.tau21 < 0.23799610585 and Q.z_dr_0p05_0p1 < 0.588259786367:
        z += -11.337957382202148 * (0.23799610585 - Q.tau21) * (0.588259786367 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.23799610585 and Q.max_dr < 0.121680960059:
        z += -148.56634521484375 * (0.23799610585 - Q.tau21) * (0.121680960059 - Q.max_dr)
    if Q.tau21 < 0.23799610585 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += 48.39396667480469 * (0.23799610585 - Q.tau21) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.width < 0.00752008842 and Q.e2 > 0.024547699839:
        z += -45346.53515625 * (0.00752008842 - Q.width) * (Q.e2 - 0.024547699839)
    if Q.LHA > 0.176724128067 and Q.sum_pt_top3 > 353.0625:
        z += 0.045643966645002365 * (Q.LHA - 0.176724128067) * (Q.sum_pt_top3 - 353.0625)
    if Q.z_dr_0p05_0p1 > 0.750909513235 and Q.n_dr_0p2_0p4 < 2.0:
        z += 1.0284262895584106 * (Q.z_dr_0p05_0p1 - 0.750909513235) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.lam1 < 0.008375572068 and Q.m01 > 16.308019673264:
        z += -27.529054641723633 * (0.008375572068 - Q.lam1) * (Q.m01 - 16.308019673264)
    if Q.lam1 < 0.006506575659 and Q.pt1_dr01 > 1.21960336377:
        z += 34.10919952392578 * (0.006506575659 - Q.lam1) * (Q.pt1_dr01 - 1.21960336377)
    if Q.mass > 80.4 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += -0.7409502863883972 * (Q.mass - 80.4) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.LHA > 0.176724128067 and Q.z_top5 > 0.865048766136:
        z += -76.41674041748047 * (Q.LHA - 0.176724128067) * (Q.z_top5 - 0.865048766136)
    if Q.width < 0.006096650059 and Q.log_sum_pt > 6.896095378249:
        z += -7694.9873046875 * (0.006096650059 - Q.width) * (Q.log_sum_pt - 6.896095378249)
    if Q.girth2 < 0.018827652745 and Q.mass_top2 > 22.844978847276:
        z += 2.2104477882385254 * (0.018827652745 - Q.girth2) * (Q.mass_top2 - 22.844978847276)
    if Q.width < 0.00752008842 and Q.planar_flow < 0.061262048692:
        z += -5289.552734375 * (0.00752008842 - Q.width) * (0.061262048692 - Q.planar_flow)
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
