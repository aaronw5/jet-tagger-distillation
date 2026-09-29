"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned), as if-statements, with each class score (logit) written out as a formula.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      each neuron is rounded to the network's fixed-point grid (round to a multiple of 2^-f, then
                  wrap modulo 2^i); each class score is then its own written-out formula (logit_g ... logit_t).
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.6% (the network: 65.8%); same class as the network for 90.1% of jets.

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
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.phi_1                  Δφ of particle 1
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
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_top5_slots=sum(pt[:5]) / tot,
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        phi_1=phi[1],
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
    z = -0.9060827493667603
    if Q.width < 0.004372139331:
        z += -298.49951171875 * Q.width + 5.120466002363744
    if 0.004372139331 <= Q.width < 0.008678044951:
        z += -886.081787109375 * Q.width + 7.689457578797569
    if Q.girth2 < 0.013238675334:
        z += -527.0708923339844 * Q.girth2 + 7.3013791473220735
    if 0.013238675334 <= Q.girth2 < 0.018827652745:
        z += -57.910186767578125 * Q.girth2 + 1.090312886858055
    if Q.mass < 21.784077072144:
        z += 0.5061936900019646 * Q.mass - 16.860982066388775
    if 21.784077072144 <= Q.mass < 29.644699859619:
        z += 0.46516888961195946 * Q.mass - 15.96729465282358
    if 29.644699859619 <= Q.mass < 56.920347213745:
        z += 0.07018765434622765 * Q.mass - 4.258194483189403
    if 56.920347213745 <= Q.mass < 64.618731689453:
        z += 0.03417455032467842 * Q.mass - 2.208316098038104
    if Q.girth2_top3 < 0.003952581551:
        z += -271.24620056152344 * Q.girth2_top3 + 0.6174392456764325
    if 0.003952581551 <= Q.girth2_top3 < 0.007929074034:
        z += 114.34284973144531 * Q.girth2_top3 - 0.9066329207791669
    if 813.415625 <= Q.sum_pt < 901.59375:
        z += -0.013911877758800983 * Q.sum_pt + 11.316138742098701
    if Q.sum_pt >= 901.59375:
        z += -0.02463722787797451 * Q.sum_pt + 20.986047376107308
    if Q.girth2_top5 < 0.000657050184:
        z += 1739.278793334961 * Q.girth2_top5 - 1.7689990058465606
    if 0.000657050184 <= Q.girth2_top5 < 0.00832969537:
        z += 81.61534118652344 * Q.girth2_top5 - 0.6798309296023545
    if Q.z_dr_0_0p05 >= 0.847731333971:
        z += 4.9030327796936035 * Q.z_dr_0_0p05 - 4.156454518833199
    if Q.girth < 0.076081777364:
        z += 98.7485237121582 * Q.girth - 7.907171179164931
    if 0.076081777364 <= Q.girth < 0.087236513197:
        z += 35.3399658203125 * Q.girth - 3.0829353946652205
    if Q.centroid_offset < 0.031170772021:
        z += -29.808443069458008 * Q.centroid_offset + 0.929152183219033
    if Q.e2 < 0.020459658932:
        z += -134.85116577148438 * Q.e2 + 2.759008858267163
    if Q.mass_over_sum_pt_sq < 0.008174660116:
        z += 129.36219787597656 * Q.mass_over_sum_pt_sq - 1.0574919994948455
    if Q.lam1 < 0.000504949057:
        z += -5600.730262756348 * Q.lam1 + 0.1938859001891824
    if 0.000504949057 <= Q.lam1 < 0.001503553356:
        z += 1014.4718856811523 * Q.lam1 - 3.1464541865287075
    if 0.001503553356 <= Q.lam1 < 0.00543336053:
        z += 188.1227035522461 * Q.lam1 - 1.903994100510935
    if 0.00543336053 <= Q.lam1 < 0.008375572068:
        z += 299.7254333496094 * Q.lam1 - 2.510371967632184
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.0085982047021389 * Q.sum_pt_top5 - 5.910728344926611
    if Q.mass_over_sum_pt < 0.13092863437:
        z += 16.672531127929688 * Q.mass_over_sum_pt - 2.1829117320711497
    if Q.LHA < 0.293190627853:
        z += 9.253565788269043 * Q.LHA - 2.7130587633416416
    if Q.lam1 < 0.006506575659 and Q.D2 < 0.87567204833:
        z += -1528.458251953125 * (0.006506575659 - Q.lam1) * (0.87567204833 - Q.D2)
    if Q.sum_pt > 901.59375 and Q.pt_7 < 25.578125:
        z += 0.0012613540748134255 * (Q.sum_pt - 901.59375) * (25.578125 - Q.pt_7)
    if Q.mass < 56.920347213745 and Q.C2 > 0.023843882605:
        z += 1.1673489809036255 * (56.920347213745 - Q.mass) * (Q.C2 - 0.023843882605)
    if Q.mass < 29.644699859619 and Q.D2 < 0.87567204833:
        z += -2.851515769958496 * (29.644699859619 - Q.mass) * (0.87567204833 - Q.D2)
    if Q.mass < 64.618731689453 and Q.centroid_offset > 0.012587644117:
        z += 1.9938530921936035 * (64.618731689453 - Q.mass) * (Q.centroid_offset - 0.012587644117)
    if Q.girth2 < 0.018827652745 and Q.eccentricity > 0.959856212153:
        z += 1545.8367919921875 * (0.018827652745 - Q.girth2) * (Q.eccentricity - 0.959856212153)
    if Q.sum_pt > 901.59375 and Q.pt_7 > 29.0421875:
        z += -0.0017508724704384804 * (Q.sum_pt - 901.59375) * (Q.pt_7 - 29.0421875)
    if Q.log_sum_pt > 6.638338705138 and Q.dr_4 < 0.072321663733:
        z += 113.71412658691406 * (Q.log_sum_pt - 6.638338705138) * (0.072321663733 - Q.dr_4)
    if Q.sum_pt_top5 > 687.4375 and Q.z_7 > 0.023207568189:
        z += -0.31951525807380676 * (Q.sum_pt_top5 - 687.4375) * (Q.z_7 - 0.023207568189)
    if Q.mass < 64.618731689453 and Q.pt_7 < 40.040625:
        z += -0.002186864148825407 * (64.618731689453 - Q.mass) * (40.040625 - Q.pt_7)
    if Q.mass < 29.644699859619 and Q.phi_1 > -0.058901977539:
        z += 0.73537278175354 * (29.644699859619 - Q.mass) * (Q.phi_1 - -0.058901977539)
    if Q.sum_pt_top5 > 687.4375 and Q.dr_4 < 0.063364507347:
        z += -0.07891817390918732 * (Q.sum_pt_top5 - 687.4375) * (0.063364507347 - Q.dr_4)
    if Q.lam1 < 0.00543336053 and Q.n_dr_0p2_0p4 < 2.0:
        z += -86.45292663574219 * (0.00543336053 - Q.lam1) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.mass_over_sum_pt_sq < 0.008174660116 and Q.n_pt_above_50 < 8.0:
        z += 19.45945167541504 * (0.008174660116 - Q.mass_over_sum_pt_sq) * (8.0 - Q.n_pt_above_50)
    if Q.log_sum_pt > 6.638338705138 and Q.dr_3 < 0.078464230803:
        z += 45.501792907714844 * (Q.log_sum_pt - 6.638338705138) * (0.078464230803 - Q.dr_3)
    if Q.mass < 29.644699859619 and Q.dr_0 < 0.111955475493:
        z += 2.9790232181549072 * (29.644699859619 - Q.mass) * (0.111955475493 - Q.dr_0)
    if Q.girth2_top5 < 0.000657050184 and Q.dr_7 < 0.222994708167:
        z += 4316.3857421875 * (0.000657050184 - Q.girth2_top5) * (0.222994708167 - Q.dr_7)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.9744626879692078
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.15350846946239471 * Q.pt_7 - 5.3008393361233175
    if Q.pt_7 >= 53.4375:
        z += 0.03332465887069702 * Q.pt_7 + 1.121483042370528
    if 6.377722943814 <= Q.log_sum_pt < 6.572937922293:
        z += 4.62814474105835 * Q.log_sum_pt - 29.51702490233994
    if 6.572937922293 <= Q.log_sum_pt < 6.638338705138:
        z += 8.051950454711914 * Q.log_sum_pt - 52.0214873161769
    if Q.log_sum_pt >= 6.638338705138:
        z += 14.833037853240967 * Q.log_sum_pt - 97.03664225675587
    if Q.z_7 < 0.043044721986:
        z += 132.4020233154297 * Q.z_7 - 6.629238190456036
    if 0.043044721986 <= Q.z_7 < 0.055577157257:
        z += 74.20983123779297 * Q.z_7 - 4.124371460718251
    if Q.mass < 49.668099212646:
        z += 0.026059100404381752 * Q.mass - 1.4822136864789968
    if 49.668099212646 <= Q.mass < 53.332374954224:
        z += 0.05128099396824837 * Q.mass - 2.7349371983399213
    if Q.width < 0.008678044951:
        z += 226.5238800048828 * Q.width - 1.9657844131573032
    if Q.e2_sq < 0.008168570676:
        z += -338.37109375 * Q.e2_sq + 2.764008194012297
    if Q.e2 >= 0.032346998155:
        z += -15.969844818115234 * Q.e2 + 0.5165765408672098
    if Q.sum_pt_top5 >= 579.875:
        z += -0.003741353750228882 * Q.sum_pt_top5 + 2.169517505913973
    if Q.LHA >= 0.266912960293:
        z += 8.112305641174316 * Q.LHA - 2.16527951348744
    if Q.max_dr < 0.250761204958:
        z += 7.172453880310059 * Q.max_dr - 1.798573177532233
    if Q.lam1 < 0.008375572068 and Q.centroid_offset > 0.02076709205:
        z += 13736.7509765625 * (0.008375572068 - Q.lam1) * (Q.centroid_offset - 0.02076709205)
    if Q.pt_7 > 34.53125 and Q.mass < 91.19:
        z += -0.0016555784968659282 * (Q.pt_7 - 34.53125) * (91.19 - Q.mass)
    if Q.lam1 < 0.008375572068 and Q.lam2 < 0.000537286005:
        z += -433672.34375 * (0.008375572068 - Q.lam1) * (0.000537286005 - Q.lam2)
    if Q.z_7 < 0.055577157257 and Q.lam2 < 0.003408388935:
        z += 6992.92529296875 * (0.055577157257 - Q.z_7) * (0.003408388935 - Q.lam2)
    if Q.log_sum_pt > 6.377722943814 and Q.centroid_offset < 0.023554160423:
        z += -90.9103012084961 * (Q.log_sum_pt - 6.377722943814) * (0.023554160423 - Q.centroid_offset)
    if Q.log_sum_pt > 6.377722943814 and Q.max_dr < 0.197968879342:
        z += 20.740331649780273 * (Q.log_sum_pt - 6.377722943814) * (0.197968879342 - Q.max_dr)
    if Q.e2 < 0.035560912266 and Q.eccentricity > 0.978160776925:
        z += 7155.45751953125 * (0.035560912266 - Q.e2) * (Q.eccentricity - 0.978160776925)
    if Q.pt_7 > 34.53125 and Q.max_dr < 0.080507021025:
        z += 0.9408159255981445 * (Q.pt_7 - 34.53125) * (0.080507021025 - Q.max_dr)
    if Q.log_sum_pt > 6.572937922293 and Q.lam2 < 0.001130644719:
        z += 4123.1162109375 * (Q.log_sum_pt - 6.572937922293) * (0.001130644719 - Q.lam2)
    if Q.LHA < 0.346713497427 and Q.planar_flow < 0.083662731125:
        z += -87.7962417602539 * (0.346713497427 - Q.LHA) * (0.083662731125 - Q.planar_flow)
    if Q.lam1 < 0.008375572068 and Q.n_pt_above_50 > 6.0:
        z += -116.31182098388672 * (0.008375572068 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.e2_sq < 0.008168570676 and Q.planar_flow < 0.083662731125:
        z += -5280.4130859375 * (0.008168570676 - Q.e2_sq) * (0.083662731125 - Q.planar_flow)
    if Q.e2_sq < 0.008168570676 and Q.n_pt_above_50 > 6.0:
        z += 92.10450744628906 * (0.008168570676 - Q.e2_sq) * (Q.n_pt_above_50 - 6.0)
    if Q.z_7 < 0.055577157257 and Q.girth2_top2 < 0.0140332421:
        z += 2791.332763671875 * (0.055577157257 - Q.z_7) * (0.0140332421 - Q.girth2_top2)
    if Q.log_sum_pt > 6.638338705138 and Q.girth2_top3 < 0.007929074034:
        z += -581.4722900390625 * (Q.log_sum_pt - 6.638338705138) * (0.007929074034 - Q.girth2_top3)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.3244855403900146
    if Q.mass < 36.229410171509:
        z += 0.06460991874337196 * Q.mass - 1.5635111539882927
    if 36.229410171509 <= Q.mass < 69.611351776123:
        z += -0.02328408882021904 * Q.mass + 1.6208368976507603
    if Q.log_sum_pt < 6.464150123592:
        z += -3.3861618041992188 * Q.log_sum_pt + 21.78851310520686
    if 6.464150123592 <= Q.log_sum_pt < 6.572937922293:
        z += -0.6603982448577881 * Q.log_sum_pt + 4.1687682562073824
    if 6.572937922293 <= Q.log_sum_pt < 6.638338705138:
        z += 2.6297607421875 * Q.log_sum_pt - 17.457242520115713
    if Q.log_sum_pt >= 6.896095378249:
        z += -8.561300277709961 * Q.log_sum_pt + 59.03954327691754
    if Q.lam1 < 0.003377388461:
        z += -391.93360137939453 * Q.lam1 + 1.6426590011225919
    if 0.003377388461 <= Q.lam1 < 0.005954149834:
        z += -123.77823638916016 * Q.lam1 + 0.7369941656493307
    if Q.pt_7 < 30.484375:
        z += -0.0009451285004615784 * Q.pt_7 - 0.5802820776589215
    if 30.484375 <= Q.pt_7 < 43.5:
        z += 0.06066584959626198 * Q.pt_7 - 2.4584542380762286
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.12414130941033363 * Q.pt_7 - 5.219636739988346
    if Q.pt_7 >= 53.4375:
        z += 0.061610978096723557 * Q.pt_7 - 1.8781721604173072
    if Q.LHA >= 0.111565049159:
        z += -11.568245887756348 * Q.LHA + 1.2906119211509364
    if Q.sum_pt < 788.4484375:
        z += -0.005896972492337227 * Q.sum_pt + 4.6494587475637665
    if Q.width < 0.000172198326:
        z += -9165.322265625 * Q.width + 1.5782531513911524
    if Q.pt_7 > 30.484375 and Q.C2 < 0.051192347892:
        z += -1.0417691469192505 * (Q.pt_7 - 30.484375) * (0.051192347892 - Q.C2)
    if Q.lam1 < 0.005954149834 and Q.max_dr > 0.080507021025:
        z += -2249.44384765625 * (0.005954149834 - Q.lam1) * (Q.max_dr - 0.080507021025)
    if Q.mass < 36.229410171509 and Q.lam2 < 0.001130644719:
        z += 39.26345443725586 * (36.229410171509 - Q.mass) * (0.001130644719 - Q.lam2)
    if Q.pt_7 > 30.484375 and Q.max_dr > 0.093110798299:
        z += -0.5310762524604797 * (Q.pt_7 - 30.484375) * (Q.max_dr - 0.093110798299)
    if Q.width < 0.000172198326 and Q.dr_7 < 0.222994708167:
        z += -48855.02734375 * (0.000172198326 - Q.width) * (0.222994708167 - Q.dr_7)
    return max(0.0, z)


def neuron_3(Q):
    z = 3.645557165145874
    if 0.06813910019 <= Q.mass_over_sum_pt < 0.090413827016:
        z += -37.136661529541016 * Q.mass_over_sum_pt + 2.5304587006835138
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += -5.539743423461914 * Q.mass_over_sum_pt - 0.3263395871982402
    if Q.mass_over_sum_pt >= 0.107985668755:
        z += -38.72684669494629 * Q.mass_over_sum_pt + 3.257391953614248
    if Q.centroid_offset >= 0.014379521101:
        z += 62.023929595947266 * Q.centroid_offset - 0.891874404391862
    if Q.width < 0.006679471358:
        z += 105.51240539550781 * Q.width - 0.7047670897529791
    if Q.width >= 0.018827653081:
        z += 589.7814331054688 * Q.width - 11.104200216124775
    if 36.229410171509 <= Q.mass < 69.611351776123:
        z += -0.029400018975138664 * Q.mass + 1.0651453465004463
    if Q.mass >= 69.611351776123:
        z += -0.10060749016702175 * Q.mass + 6.021993672726764
    if Q.e2 < 0.028531698044:
        z += -196.51567459106445 * Q.e2 + 8.441129038786686
    if 0.028531698044 <= Q.e2 < 0.038466955721:
        z += -293.56196212768555 * Q.e2 + 11.21002441107276
    if 0.038466955721 <= Q.e2 < 0.04447356835:
        z += -243.8460464477539 * Q.e2 + 9.29760448398386
    if 0.04447356835 <= Q.e2 < 0.050284641981:
        z += -97.0462875366211 * Q.e2 + 2.7688953722860736
    if Q.e2 >= 0.050284641981:
        z += -31.008331298828125 * Q.e2 - 0.5517996142882917
    if Q.girth2 < 0.004372139461:
        z += 1115.7690124511719 * Q.girth2 - 16.79197876018249
    if 0.004372139461 <= Q.girth2 < 0.008678044751:
        z += 2273.1499938964844 * Q.girth2 - 21.85220982057045
    if 0.008678044751 <= Q.girth2 < 0.013238675334:
        z += 466.1005554199219 * Q.girth2 - 6.1705539262014195
    if Q.lam1 < 0.004183811014:
        z += -440.7293701171875 * Q.lam1 + 3.2305814862174227
    if 0.004183811014 <= Q.lam1 < 0.007330079875:
        z += -870.3404235839844 * Q.lam1 + 5.02799294344795
    if 0.007330079875 <= Q.lam1 < 0.008375572068:
        z += -429.6110534667969 * Q.lam1 + 1.7974114572305275
    if 0.008375572068 <= Q.lam1 < 0.012003726523:
        z += -532.8588409423828 * Q.lam1 + 2.662170742093845
    if 0.012003726523 <= Q.lam1 < 0.016433749775:
        z += -823.9448394775391 * Q.lam1 + 6.15628746318424
    if Q.lam1 >= 0.016433749775:
        z += -604.4842071533203 * Q.lam1 + 2.549726346104752
    if 0.325582223496 <= Q.LHA < 0.423592510895:
        z += 28.593263626098633 * Q.LHA - 9.309458348392493
    if Q.LHA >= 0.423592510895:
        z += -27.75153923034668 * Q.LHA + 14.557778169452945
    if Q.z_dr_0p1_0p2 < 0.468445876241:
        z += -2.6197173595428467 * Q.z_dr_0p1_0p2 + 1.2271957939948077
    if Q.max_dr >= 0.145231109113:
        z += 7.913630485534668 * Q.max_dr - 1.1493053325246485
    if Q.C2 >= 0.094821243733:
        z += -172.31881713867188 * Q.C2 + 16.339484559688263
    if Q.mass_over_sum_pt_sq >= 0.006387803907:
        z += -88.9245376586914 * Q.mass_over_sum_pt_sq + 0.5680325090843577
    if Q.mass_over_sum_pt > 0.06813910019 and Q.n_dr_0_0p05 < 5.0:
        z += 25.180967330932617 * (Q.mass_over_sum_pt - 0.06813910019) * (5.0 - Q.n_dr_0_0p05)
    if Q.LHA > 0.312727471086 and Q.mass_top3 < 50.352200171245:
        z += -0.29575347900390625 * (Q.LHA - 0.312727471086) * (50.352200171245 - Q.mass_top3)
    if Q.mass > 36.229410171509 and Q.eccentricity > 0.620723099573:
        z += 0.17889399826526642 * (Q.mass - 36.229410171509) * (Q.eccentricity - 0.620723099573)
    if Q.width > 0.018827653081 and Q.z_3 > 0.090493038582:
        z += -1859.630615234375 * (Q.width - 0.018827653081) * (Q.z_3 - 0.090493038582)
    if Q.mass_over_sum_pt > 0.06813910019 and Q.dr_7 < 0.042151962757:
        z += -5048.0380859375 * (Q.mass_over_sum_pt - 0.06813910019) * (0.042151962757 - Q.dr_7)
    if Q.LHA > 0.312727471086 and Q.max_dr < 0.145231109113:
        z += -2200.3857421875 * (Q.LHA - 0.312727471086) * (0.145231109113 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.max_dr < 0.145231109113:
        z += 11014.212890625 * (Q.mass_over_sum_pt - 0.090413827016) * (0.145231109113 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.107985668755 and Q.dr_7 < 0.04881348081:
        z += 12012.7578125 * (Q.mass_over_sum_pt - 0.107985668755) * (0.04881348081 - Q.dr_7)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.pt_6 < 56.53125:
        z += 1.2739681005477905 * (Q.mass_over_sum_pt - 0.090413827016) * (56.53125 - Q.pt_6)
    if Q.max_dr > 0.145231109113 and Q.dr_1 < 0.047732555332:
        z += -692.3062133789062 * (Q.max_dr - 0.145231109113) * (0.047732555332 - Q.dr_1)
    if Q.mass_over_sum_pt > 0.107985668755 and Q.z_7 < 0.068101508468:
        z += 1002.8162231445312 * (Q.mass_over_sum_pt - 0.107985668755) * (0.068101508468 - Q.z_7)
    if Q.centroid_offset > 0.014379521101 and Q.pt_7 > 25.578125:
        z += 1.321939468383789 * (Q.centroid_offset - 0.014379521101) * (Q.pt_7 - 25.578125)
    if Q.lam1 > 0.004183811014 and Q.z_7 < 0.075389597551:
        z += -3123.0693359375 * (Q.lam1 - 0.004183811014) * (0.075389597551 - Q.z_7)
    if Q.girth2 < 0.008678044751 and Q.pt1_dr01 > 5.351076855015:
        z += 18.41227912902832 * (0.008678044751 - Q.girth2) * (Q.pt1_dr01 - 5.351076855015)
    return max(0.0, z)


def neuron_4(Q):
    z = -12.38253402709961
    if Q.tau21 < 0.23799610585:
        z += -34.1045036315918 * Q.tau21 + 8.11673905626603
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += -188.69395446777344 * Q.mass_over_sum_pt + 17.060542558214248
    if Q.mass_over_sum_pt >= 0.107985668755:
        z += 198.1792449951172 * Q.mass_over_sum_pt - 24.716218609172504
    if 0.000319370692 <= Q.width < 0.001653836415:
        z += -363.8856506347656 * Q.width + 0.11621441205209534
    if 0.001653836415 <= Q.width < 0.002635417778:
        z += 5.435089111328125 * Q.width - 0.49458167615473236
    if Q.width >= 0.002635417778:
        z += 409.34967041015625 * Q.width - 1.5590653445030904
    if 0.007078157854 <= Q.e2 < 0.020459658932:
        z += 50.273075103759766 * Q.e2 - 0.35584076139040904
    if 0.020459658932 <= Q.e2 < 0.041109715588:
        z += 229.9799690246582 * Q.e2 - 4.032582518741095
    if 0.041109715588 <= Q.e2 < 0.063441075385:
        z += 142.0878562927246 * Q.e2 - 0.4193627619028719
    if Q.e2 >= 0.063441075385:
        z += 36.6415901184082 * Q.e2 + 6.270261759528712
    if Q.sum_pt < 763.825:
        z += -0.004746763966977596 * Q.sum_pt + 3.6256969870766627
    if Q.girth2_top2 < 0.003111083776:
        z += 46.50109100341797 * Q.girth2_top2 - 1.4393172361288527
    if 0.003111083776 <= Q.girth2_top2 < 0.007639643088:
        z += 235.28119659423828 * Q.girth2_top2 - 2.0266279598640207
    if 0.007639643088 <= Q.girth2_top2 < 0.009530300104:
        z += 121.20844268798828 * Q.girth2_top2 - 1.1551528339550128
    if Q.mass < 41.377904891968:
        z += -0.07366244681179523 * Q.mass + 4.222717240699753
    if 41.377904891968 <= Q.mass < 56.920347213745:
        z += -0.04700022004544735 * Q.mass + 3.1194901573537273
    if 56.920347213745 <= Q.mass < 80.4:
        z += -0.018919415771961212 * Q.mass + 1.5211210280656815
    if 0.014943876117 <= Q.C2 < 0.067292226106:
        z += -61.78887939453125 * Q.C2 + 0.923365359080129
    if Q.C2 >= 0.067292226106:
        z += -179.90470123291016 * Q.C2 + 8.871641948924335
    if Q.lam2 < 0.001130644719:
        z += -808.099609375 * Q.lam2 + 0.9136735557658067
    if Q.girth2_top3 < 0.005011406868:
        z += -227.0253143310547 * Q.girth2_top3 + 1.1377162194485064
    if Q.girth < 0.047915700823:
        z += -35.805253982543945 * Q.girth + 5.2990805808415615
    if 0.047915700823 <= Q.girth < 0.076081777364:
        z += -85.8037281036377 * Q.girth + 7.694792508434396
    if 0.076081777364 <= Q.girth < 0.101940929517:
        z += -24.069425582885742 * Q.girth + 2.997937048328723
    if 0.101940929517 <= Q.girth < 0.124553743005:
        z += 20.606035232543945 * Q.girth - 1.5563209537964897
    if Q.girth >= 0.124553743005:
        z += 44.67546081542969 * Q.girth - 4.554258002125213
    if Q.e2_sq < 0.003013300392:
        z += -491.57159423828125 * Q.e2_sq + 32.763530196749414
    if 0.003013300392 <= Q.e2_sq < 0.007182789718:
        z += -2305.1647338867188 * Q.e2_sq + 38.22843111538056
    if 0.007182789718 <= Q.e2_sq < 0.017162483186:
        z += -2102.2796020507812 * Q.e2_sq + 36.77114987649431
    if 0.017162483186 <= Q.e2_sq < 0.023780909279:
        z += -104.37701416015625 * Q.e2_sq + 2.482180304555574
    if 0.003562611155 <= Q.girth2 < 0.007520088344:
        z += 739.008544921875 * Q.girth2 - 2.6328000857789906
    if 0.007520088344 <= Q.girth2 < 0.013238675334:
        z += -162.93792724609375 * Q.girth2 + 4.149917066483272
    if Q.girth2 >= 0.013238675334:
        z += -587.5093688964844 * Q.girth2 + 9.770680538581118
    if Q.centroid_offset < 0.014379521101:
        z += -70.69116973876953 * Q.centroid_offset + 1.0165051669130092
    if 0.102758520097 <= Q.max_dr < 0.197968879342:
        z += 15.14682388305664 * Q.max_dr - 1.5564652063927953
    if Q.max_dr >= 0.197968879342:
        z += -8.604887008666992 * Q.max_dr + 3.145634381296907
    if Q.mass_over_sum_pt_sq < 0.001101860861:
        z += 2284.985595703125 * Q.mass_over_sum_pt_sq - 26.63999086921063
    if 0.001101860861 <= Q.mass_over_sum_pt_sq < 0.017142307326:
        z += 1503.83935546875 * Q.mass_over_sum_pt_sq - 25.77927640037907
    if Q.tau21 < 0.23799610585 and Q.mass < 62.55:
        z += -0.4459865987300873 * (0.23799610585 - Q.tau21) * (62.55 - Q.mass)
    if Q.tau21 < 0.23799610585 and Q.e2_sq > 0.011657374702:
        z += -785.2164916992188 * (0.23799610585 - Q.tau21) * (Q.e2_sq - 0.011657374702)
    if Q.tau21 < 0.23799610585 and Q.lam2 < 0.001130644719:
        z += -11392.1845703125 * (0.23799610585 - Q.tau21) * (0.001130644719 - Q.lam2)
    if Q.girth2_top2 < 0.009530300104 and Q.centroid_offset > 0.016278845848:
        z += 13073.2978515625 * (0.009530300104 - Q.girth2_top2) * (Q.centroid_offset - 0.016278845848)
    if Q.tau21 < 0.23799610585 and Q.planar_flow > 0.045057236346:
        z += 23.324840545654297 * (0.23799610585 - Q.tau21) * (Q.planar_flow - 0.045057236346)
    if Q.sum_pt < 763.825 and Q.n_dr_0p2_0p4 < 1.0:
        z += -0.0021504589822143316 * (763.825 - Q.sum_pt) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.tau21 < 0.23799610585 and Q.dr_7 < 0.175465903809:
        z += -26.981035232543945 * (0.23799610585 - Q.tau21) * (0.175465903809 - Q.dr_7)
    if Q.mass < 69.611351776123 and Q.dr_3 < 0.104247858869:
        z += 0.1130632758140564 * (69.611351776123 - Q.mass) * (0.104247858869 - Q.dr_3)
    if Q.tau21 < 0.23799610585 and Q.pt_7 > 33.21875:
        z += 0.24040380120277405 * (0.23799610585 - Q.tau21) * (Q.pt_7 - 33.21875)
    if Q.mass_over_sum_pt > 0.107985668755 and Q.D2 < 3.885568320751:
        z += -71.87755584716797 * (Q.mass_over_sum_pt - 0.107985668755) * (3.885568320751 - Q.D2)
    if Q.max_dr > 0.102758520097 and Q.pt_7 > 37.15625:
        z += -1.3731632232666016 * (Q.max_dr - 0.102758520097) * (Q.pt_7 - 37.15625)
    if Q.C2 > 0.014943876117 and Q.pt_7 > 38.53125:
        z += 2.9048683643341064 * (Q.C2 - 0.014943876117) * (Q.pt_7 - 38.53125)
    if Q.lam2 < 0.000306123359 and Q.mass_top3 < 50.352200171245:
        z += -109.2301025390625 * (0.000306123359 - Q.lam2) * (50.352200171245 - Q.mass_top3)
    if Q.centroid_offset < 0.014379521101 and Q.z_dr_0p05_0p1 < 0.674770402908:
        z += -387.19110107421875 * (0.014379521101 - Q.centroid_offset) * (0.674770402908 - Q.z_dr_0p05_0p1)
    if Q.girth2 > 0.018827652745 and Q.pt_6 > 62.25:
        z += -157.218505859375 * (Q.girth2 - 0.018827652745) * (Q.pt_6 - 62.25)
    if Q.max_dr > 0.197968879342 and Q.z_7 > 0.028070914944:
        z += 436.62017822265625 * (Q.max_dr - 0.197968879342) * (Q.z_7 - 0.028070914944)
    if Q.max_dr > 0.102758520097 and Q.eccentricity > 0.984196588116:
        z += -679.1522827148438 * (Q.max_dr - 0.102758520097) * (Q.eccentricity - 0.984196588116)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.09480305761098862
    if Q.LHA < 0.154689112391:
        z += -9.299341201782227 * Q.LHA + 3.2048700888833452
    if 0.154689112391 <= Q.LHA < 0.216055863061:
        z += -28.78371810913086 * Q.LHA + 6.218891058172803
    if 6.572937922293 <= Q.log_sum_pt < 6.842716632804:
        z += 6.3885040283203125 * Q.log_sum_pt - 41.99124039446817
    if 6.842716632804 <= Q.log_sum_pt < 6.896095378249:
        z += -28.57244110107422 * Q.log_sum_pt + 197.23660034098776
    if Q.log_sum_pt >= 6.896095378249:
        z += 4.226310729980469 * Q.log_sum_pt - 28.946720573484384
    if Q.z_6 < 0.028865759995:
        z += 34.02033615112305 * Q.z_6 + 0.34754473587334545
    if 0.028865759995 <= Q.z_6 < 0.05096141791:
        z += -60.17325210571289 * Q.z_6 + 3.0665142475630223
    if Q.z_7 < 0.023207568189:
        z += -265.07571029663086 * Q.z_7 + 7.828225899147549
    if 0.023207568189 <= Q.z_7 < 0.071488645583:
        z += -34.72298812866211 * Q.z_7 + 2.4822993919126417
    if Q.e2 < 0.035560912266:
        z += -64.57422637939453 * Q.e2 + 2.296318398922472
    if Q.width < 0.002635417778:
        z += -472.68682861328125 * Q.width + 1.2457272715538805
    if Q.mass_over_sum_pt < 0.084751611895:
        z += 23.273845672607422 * Q.mass_over_sum_pt - 2.2825636926781483
    if 0.084751611895 <= Q.mass_over_sum_pt < 0.107985668755:
        z += 13.345398902893066 * Q.mass_over_sum_pt - 1.441111825331151
    if Q.sum_pt_top5 >= 752.1:
        z += 0.010081915184855461 * Q.sum_pt_top5 - 7.582608410529793
    if Q.sum_pt_top2 < 548.196875:
        z += 0.0041093300096690655 * Q.sum_pt_top2 - 2.252721869644301
    if Q.sum_pt >= 868.509375:
        z += -0.01696869730949402 * Q.sum_pt + 14.737472694832832
    if Q.dr_0 < 0.021588001063:
        z += 158.9989471435547 * Q.dr_0 - 3.4324694399509394
    if Q.pt_7 < 53.4375:
        z += -0.0339660607278347 * Q.pt_7 + 1.8150613701436669
    if Q.girth2_top2 < 0.004007841607:
        z += 245.62486267089844 * Q.girth2_top2 - 0.9844255443260878
    if Q.z_7 < 0.049399692737 and Q.mass_top5 < 62.55:
        z += 0.5339535474777222 * (0.049399692737 - Q.z_7) * (62.55 - Q.mass_top5)
    if Q.LHA < 0.216055863061 and Q.log_sum_pt < 6.804164030582:
        z += -86.01834106445312 * (0.216055863061 - Q.LHA) * (6.804164030582 - Q.log_sum_pt)
    if Q.e2_sq < 0.00528466865 and Q.centroid_offset < 0.014379521101:
        z += -20419.18359375 * (0.00528466865 - Q.e2_sq) * (0.014379521101 - Q.centroid_offset)
    if Q.z_7 < 0.071488645583 and Q.sum_pt < 788.4484375:
        z += -0.26329854130744934 * (0.071488645583 - Q.z_7) * (788.4484375 - Q.sum_pt)
    if Q.log_sum_pt > 6.896095378249 and Q.planar_flow < 0.045057236346:
        z += -1569.57763671875 * (Q.log_sum_pt - 6.896095378249) * (0.045057236346 - Q.planar_flow)
    if Q.z_7 < 0.071488645583 and Q.lam2 < 0.001130644719:
        z += 31967.833984375 * (0.071488645583 - Q.z_7) * (0.001130644719 - Q.lam2)
    if Q.width < 0.002635417778 and Q.centroid_offset < 0.023554160423:
        z += 17761.201171875 * (0.002635417778 - Q.width) * (0.023554160423 - Q.centroid_offset)
    if Q.sum_pt > 868.509375 and Q.centroid_offset < 0.012587644117:
        z += 1.8995431661605835 * (Q.sum_pt - 868.509375) * (0.012587644117 - Q.centroid_offset)
    if Q.log_sum_pt > 6.896095378249 and Q.centroid_offset < 0.018377780003:
        z += -2715.055419921875 * (Q.log_sum_pt - 6.896095378249) * (0.018377780003 - Q.centroid_offset)
    if Q.log_sum_pt > 6.572937922293 and Q.mean_phi2 < 0.000145482056:
        z += 32840.95703125 * (Q.log_sum_pt - 6.572937922293) * (0.000145482056 - Q.mean_phi2)
    if Q.sum_pt_top2 < 548.196875 and Q.dr_0 < 0.021588001063:
        z += 0.31046414375305176 * (548.196875 - Q.sum_pt_top2) * (0.021588001063 - Q.dr_0)
    if Q.log_sum_pt > 6.572937922293 and Q.dr_0 < 0.021588001063:
        z += 582.4913330078125 * (Q.log_sum_pt - 6.572937922293) * (0.021588001063 - Q.dr_0)
    if Q.log_sum_pt > 6.896095378249 and Q.dr_0 < 0.04118638065:
        z += -573.1027221679688 * (Q.log_sum_pt - 6.896095378249) * (0.04118638065 - Q.dr_0)
    if Q.log_sum_pt > 6.572937922293 and Q.lam1 < 0.012003726523:
        z += -781.5698852539062 * (Q.log_sum_pt - 6.572937922293) * (0.012003726523 - Q.lam1)
    if Q.log_sum_pt > 6.572937922293 and Q.girth2_top2 < 0.006299534492:
        z += 595.0860595703125 * (Q.log_sum_pt - 6.572937922293) * (0.006299534492 - Q.girth2_top2)
    if Q.log_sum_pt > 6.572937922293 and Q.mean_eta2 < 9.0303693e-05:
        z += 47180.02734375 * (Q.log_sum_pt - 6.572937922293) * (9.0303693e-05 - Q.mean_eta2)
    if Q.mass_over_sum_pt < 0.084751611895 and Q.max_pair_mass < 18.097979966098:
        z += -0.5074002742767334 * (0.084751611895 - Q.mass_over_sum_pt) * (18.097979966098 - Q.max_pair_mass)
    if Q.z_7 < 0.03243272066 and Q.pt_5 < 29.875:
        z += -9.329241752624512 * (0.03243272066 - Q.z_7) * (29.875 - Q.pt_5)
    if Q.e2 < 0.035560912266 and Q.lam2 < 7.3007261e-05:
        z += 350012.03125 * (0.035560912266 - Q.e2) * (7.3007261e-05 - Q.lam2)
    if Q.LHA < 0.216055863061 and Q.centroid_offset > 0.00231612516:
        z += -2481.211669921875 * (0.216055863061 - Q.LHA) * (Q.centroid_offset - 0.00231612516)
    if Q.log_sum_pt > 6.572937922293 and Q.n_dr_0p2_0p4 < 1.0:
        z += -2.9955241680145264 * (Q.log_sum_pt - 6.572937922293) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_7 < 0.049399692737 and Q.pt_5 < 73.75:
        z += -0.6582204103469849 * (0.049399692737 - Q.z_7) * (73.75 - Q.pt_5)
    if Q.mean_phi2 < 0.01426135283 and Q.pt_5 < 24.578125:
        z += 23.2356014251709 * (0.01426135283 - Q.mean_phi2) * (24.578125 - Q.pt_5)
    if Q.z_6 < 0.028865759995 and Q.n_dr_0p2_0p4 < 2.0:
        z += 95.18636322021484 * (0.028865759995 - Q.z_6) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.z_6 < 0.05096141791 and Q.e2_sq > 0.003902458471:
        z += -9242.20703125 * (0.05096141791 - Q.z_6) * (Q.e2_sq - 0.003902458471)
    if Q.z_7 < 0.049399692737 and Q.centroid_offset < 0.010960638421:
        z += -6466.27490234375 * (0.049399692737 - Q.z_7) * (0.010960638421 - Q.centroid_offset)
    if Q.mass < 60.630975723267 and Q.n_dr_0p2_0p4 < 2.0:
        z += 0.020554961636662483 * (60.630975723267 - Q.mass) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.sum_pt_top5 > 752.1 and Q.D2 < 1.679198372364:
        z += -0.007601612247526646 * (Q.sum_pt_top5 - 752.1) * (1.679198372364 - Q.D2)
    return max(0.0, z)


def neuron_6(Q):
    z = 6.268299102783203
    if Q.width < 0.013238675006:
        z += 711.9116821289062 * Q.width - 9.424767392679367
    if Q.mass_over_sum_pt < 0.008374148675:
        z += -17.66200065612793 * Q.mass_over_sum_pt + 2.722964886000322
    if 0.008374148675 <= Q.mass_over_sum_pt < 0.154170806525:
        z += -79.23361015319824 * Q.mass_over_sum_pt + 3.238574698087831
    if Q.mass_over_sum_pt >= 0.154170806525:
        z += -61.57160949707031 * Q.mass_over_sum_pt + 0.5156098120875088
    if Q.e2 < 0.050284641981:
        z += -132.80885314941406 * Q.e2 + 6.67824563252549
    if 0.000537286005 <= Q.lam2 < 0.003408388935:
        z += 304.7294921875 * Q.lam2 - 0.16372689146310057
    if Q.lam2 >= 0.003408388935:
        z += -2514.62548828125 * Q.lam2 + 9.445731427803729
    if 0.018377780003 <= Q.centroid_offset < 0.049903668404:
        z += -211.5895233154297 * Q.centroid_offset + 3.888545710430606
    if Q.centroid_offset >= 0.049903668404:
        z += -98.04016876220703 * Q.centroid_offset - 1.7779836266816447
    if Q.log_sum_pt < 6.572937922293:
        z += -0.9739456176757812 * Q.log_sum_pt + 5.71439575811295
    if 6.572937922293 <= Q.log_sum_pt < 6.701242202626:
        z += 5.356706142425537 * Q.log_sum_pt - 35.89658526868793
    if Q.LHA >= 0.312727471086:
        z += -5.162337779998779 * Q.LHA + 1.6144048388307337
    z += 12.712364196777344 * Q.max_dr
    if Q.girth2_top5 < 0.002270363079:
        z += 90.85687255859375 * Q.girth2_top5 - 2.218691345397041
    if 0.002270363079 <= Q.girth2_top5 < 0.011482925368:
        z += -13.84783935546875 * Q.girth2_top5 - 1.980973633270022
    if 0.011482925368 <= Q.girth2_top5 < 0.024419631481:
        z += 176.7898712158203 * Q.girth2_top5 - 4.170052236086519
    if Q.girth2_top5 >= 0.024419631481:
        z += 85.93299865722656 * Q.girth2_top5 - 1.9513608906894782
    if Q.girth2 < 0.008678044751:
        z += 1234.09814453125 * Q.girth2 - 10.709558925368253
    if Q.girth >= 0.087236513197:
        z += 63.83720397949219 * Q.girth - 5.568935087416551
    if Q.D2 < 1.679198372364:
        z += -1.0852128267288208 * Q.D2 + 1.8222876123115714
    if Q.lam1 < 0.008375572068:
        z += -224.8246612548828 * Q.lam1 + 1.8830351530039582
    if Q.girth2_top2 < 0.009530300104:
        z += -142.34466552734375 * Q.girth2_top2 + 1.3565873806790893
    if Q.pt1_dr01 < 5.351076855015:
        z += 0.11435382813215256 * Q.pt1_dr01 - 0.6119161230003247
    if Q.z_dr_0_0p05 >= 0.768138587475:
        z += -2.9528326988220215 * Q.z_dr_0_0p05 + 2.2681847383231397
    if Q.pt_4 < 31.125:
        z += -0.2530548572540283 * Q.pt_4 + 7.8763324320316315
    if Q.z_4 < 0.037477688199:
        z += 181.21458435058594 * Q.z_4 - 6.791503689402645
    if Q.sum_pt >= 988.4078125:
        z += 0.04807520657777786 * Q.sum_pt - 47.51790976902703
    if Q.sum_pt_top5 >= 839.9546875:
        z += -0.027865860611200333 * Q.sum_pt_top5 + 23.406060241599334
    if Q.pt_6 < 24.421875:
        z += -0.23047934472560883 * Q.pt_6 + 5.628737746970728
    if Q.centroid_offset > 0.008092360237 and Q.lam2 < 0.003408388935:
        z += 48096.25 * (Q.centroid_offset - 0.008092360237) * (0.003408388935 - Q.lam2)
    if Q.centroid_offset > 0.008092360237 and Q.planar_flow > 0.00804883781:
        z += 88.9769058227539 * (Q.centroid_offset - 0.008092360237) * (Q.planar_flow - 0.00804883781)
    if Q.log_sum_pt < 6.701242202626 and Q.z_7 < 0.049399692737:
        z += 306.8141174316406 * (6.701242202626 - Q.log_sum_pt) * (0.049399692737 - Q.z_7)
    if Q.centroid_offset > 0.018377780003 and Q.mean_phi2 < 0.008921136335:
        z += 9297.5810546875 * (Q.centroid_offset - 0.018377780003) * (0.008921136335 - Q.mean_phi2)
    if Q.LHA > 0.312727471086 and Q.eccentricity > 0.872657364787:
        z += 425.4704895019531 * (Q.LHA - 0.312727471086) * (Q.eccentricity - 0.872657364787)
    if Q.mass_over_sum_pt > 0.008374148675 and Q.tau32 < 0.518696343899:
        z += -27.437686920166016 * (Q.mass_over_sum_pt - 0.008374148675) * (0.518696343899 - Q.tau32)
    if Q.e2 < 0.050284641981 and Q.z_dr_0p1_0p2 > 0.15855820179:
        z += 419.9459228515625 * (0.050284641981 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.15855820179)
    if Q.log_sum_pt < 6.701242202626 and Q.pt_6 < 36.8125:
        z += 0.2581874430179596 * (6.701242202626 - Q.log_sum_pt) * (36.8125 - Q.pt_6)
    if Q.log_sum_pt < 6.701242202626 and Q.mean_eta2 > 0.004247450386:
        z += 385.63885498046875 * (6.701242202626 - Q.log_sum_pt) * (Q.mean_eta2 - 0.004247450386)
    if Q.D2 < 1.679198372364 and Q.pt_4 < 90.625:
        z += -0.013430390506982803 * (1.679198372364 - Q.D2) * (90.625 - Q.pt_4)
    if Q.girth2_top5 > 0.011482925368 and Q.mean_eta > 0.0127187056:
        z += -9543.306640625 * (Q.girth2_top5 - 0.011482925368) * (Q.mean_eta - 0.0127187056)
    if Q.log_sum_pt < 6.701242202626 and Q.pt_7 < 45.75:
        z += 0.26105985045433044 * (6.701242202626 - Q.log_sum_pt) * (45.75 - Q.pt_7)
    if Q.C2 > 0.010539266048 and Q.pt_7 > 31.859375:
        z += 1.087256908416748 * (Q.C2 - 0.010539266048) * (Q.pt_7 - 31.859375)
    if Q.mass < 49.668099212646 and Q.z_dr_0p05_0p1 < 0.750909513235:
        z += 0.06049972027540207 * (49.668099212646 - Q.mass) * (0.750909513235 - Q.z_dr_0p05_0p1)
    if Q.girth2_top5 > 0.002270363079 and Q.z_dr_0p05_0p1 > 0.291944718361:
        z += 183.786865234375 * (Q.girth2_top5 - 0.002270363079) * (Q.z_dr_0p05_0p1 - 0.291944718361)
    if Q.girth2_top5 > 0.011482925368 and Q.pt_7 > 15.55390625:
        z += -4.019756317138672 * (Q.girth2_top5 - 0.011482925368) * (Q.pt_7 - 15.55390625)
    if Q.lam2 < 0.000537286005 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += -7379.115234375 * (0.000537286005 - Q.lam2) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.mass_over_sum_pt > 0.008374148675 and Q.pt_7 < 41.65625:
        z += -0.4093879461288452 * (Q.mass_over_sum_pt - 0.008374148675) * (41.65625 - Q.pt_7)
    if Q.sum_pt > 988.4078125 and Q.pt_6 > 29.90625:
        z += -0.0003305621212348342 * (Q.sum_pt - 988.4078125) * (Q.pt_6 - 29.90625)
    return max(0.0, z)


def neuron_7(Q):
    z = 10.08840274810791
    if 0.0016538364 <= Q.girth2 < 0.004372139461:
        z += -528.0969848632812 * Q.girth2 + 0.8733860162971435
    if 0.004372139461 <= Q.girth2 < 0.007520088344:
        z += -861.7186889648438 * Q.girth2 + 2.3320266338456506
    if 0.007520088344 <= Q.girth2 < 0.008678044751:
        z += -1995.9165649414062 * Q.girth2 + 10.861294860766556
    if 0.008678044751 <= Q.girth2 < 0.013238675334:
        z += -1572.8219909667969 * Q.girth2 + 7.189661213909616
    if Q.girth2 >= 0.013238675334:
        z += -2019.3940124511719 * Q.girth2 + 13.10168321958933
    if Q.mass_over_sum_pt < 0.072690732432:
        z += -6.581986904144287 * Q.mass_over_sum_pt + 0.8617705568008356
    if 0.072690732432 <= Q.mass_over_sum_pt < 0.084751611895:
        z += 99.7753586769104 * Q.mass_over_sum_pt - 6.8694227930093685
    if 0.084751611895 <= Q.mass_over_sum_pt < 0.090413827016:
        z += 277.22939920425415 * Q.mass_over_sum_pt - 21.908938764982405
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += 124.92045450210571 * Q.mass_over_sum_pt - 8.13810418569285
    if 0.107985668755 <= Q.mass_over_sum_pt < 0.13092863437:
        z += -378.66346883773804 * Q.mass_over_sum_pt + 46.24174255042683
    if Q.mass_over_sum_pt >= 0.13092863437:
        z += -372.08148193359375 * Q.mass_over_sum_pt + 45.37997199362599
    if Q.e2 < 0.016554418951:
        z += -159.56078720092773 * Q.e2 + 4.698795401723687
    if 0.016554418951 <= Q.e2 < 0.024547699839:
        z += -212.67237854003906 * Q.e2 + 5.578026935905639
    if 0.024547699839 <= Q.e2 < 0.038466955721:
        z += -109.28881072998047 * Q.e2 + 3.0401981450194175
    if Q.e2 >= 0.038466955721:
        z += -53.11159133911133 * Q.e2 + 0.879231534181952
    if Q.girth < 0.04081947431:
        z += 115.78513717651367 * Q.girth - 8.252100457908181
    if 0.04081947431 <= Q.girth < 0.087236513197:
        z += 75.95943450927734 * Q.girth - 6.62643621100523
    if Q.width < 0.000561123155:
        z += 5215.779052734375 * Q.width - 8.090214484908357
    if 0.000561123155 <= Q.width < 0.005590288644:
        z += 1026.715087890625 * Q.width - 5.739633696458423
    if Q.centroid_offset < 0.02076709205:
        z += 56.28274917602539 * Q.centroid_offset - 1.168829032965581
    if Q.centroid_offset >= 0.031170772021:
        z += 69.29428100585938 * Q.centroid_offset - 2.159956235592753
    if Q.mass < 29.644699859619:
        z += -0.12477409094572067 * Q.mass + 3.6988904763426937
    if Q.mass >= 80.4:
        z += -0.24539683759212494 * Q.mass + 19.729905742406846
    if Q.lam1 < 0.002464291268:
        z += 278.51739501953125 * Q.lam1 - 4.982547374355002
    if 0.002464291268 <= Q.lam1 < 0.008375572068:
        z += 726.77978515625 * Q.lam1 - 6.087196468141728
    if Q.LHA < 0.293190627853:
        z += -7.532948017120361 * Q.LHA + 2.20858975872353
    if Q.max_dr < 0.15984864831:
        z += 29.202367782592773 * Q.max_dr - 4.667959017498947
    if Q.mass_top5 >= 53.607658247923:
        z += 0.03102882020175457 * Q.mass_top5 - 1.6633823892119082
    if Q.C2 < 0.067292226106:
        z += 32.77997589111328 * Q.C2 - 2.205837549414024
    if Q.planar_flow < 0.195013533663 and Q.width > 0.00752008842:
        z += 832.0961303710938 * (0.195013533663 - Q.planar_flow) * (Q.width - 0.00752008842)
    if Q.planar_flow < 0.195013533663 and Q.sum_pt > 615.875:
        z += 0.013925769366323948 * (0.195013533663 - Q.planar_flow) * (Q.sum_pt - 615.875)
    if Q.girth2 > 0.004372139461 and Q.eccentricity > 0.945820652852:
        z += 5104.90283203125 * (Q.girth2 - 0.004372139461) * (Q.eccentricity - 0.945820652852)
    if Q.lam1 < 0.008375572068 and Q.D2 < 1.122624260187:
        z += -1357.239013671875 * (0.008375572068 - Q.lam1) * (1.122624260187 - Q.D2)
    if Q.e2 < 0.050284641981 and Q.D2 < 1.122624260187:
        z += 110.86986541748047 * (0.050284641981 - Q.e2) * (1.122624260187 - Q.D2)
    if Q.centroid_offset < 0.02076709205 and Q.C2 > 0.023843882605:
        z += 1459.6451416015625 * (0.02076709205 - Q.centroid_offset) * (Q.C2 - 0.023843882605)
    if Q.mass > 80.4 and Q.eccentricity > 0.927072033478:
        z += -1.137751817703247 * (Q.mass - 80.4) * (Q.eccentricity - 0.927072033478)
    if Q.pt_7 < 48.71875 and Q.planar_flow < 0.694781820497:
        z += -0.09202834218740463 * (48.71875 - Q.pt_7) * (0.694781820497 - Q.planar_flow)
    if Q.centroid_offset > 0.031170772021 and Q.pt_2 > 56.5:
        z += -1.5251333713531494 * (Q.centroid_offset - 0.031170772021) * (Q.pt_2 - 56.5)
    if Q.max_dr < 0.197968879342 and Q.z_dr_0p05_0p1 > 0.048758227378:
        z += 37.849884033203125 * (0.197968879342 - Q.max_dr) * (Q.z_dr_0p05_0p1 - 0.048758227378)
    if Q.max_dr < 0.15984864831 and Q.z_dr_0p05_0p1 < 0.674770402908:
        z += 43.398284912109375 * (0.15984864831 - Q.max_dr) * (0.674770402908 - Q.z_dr_0p05_0p1)
    if Q.mass_over_sum_pt < 0.13092863437 and Q.z_dr_0p05_0p1 > 0.291944718361:
        z += -34.78810119628906 * (0.13092863437 - Q.mass_over_sum_pt) * (Q.z_dr_0p05_0p1 - 0.291944718361)
    if Q.centroid_offset < 0.037760993714 and Q.sum_pt > 559.6875:
        z += 0.2356172800064087 * (0.037760993714 - Q.centroid_offset) * (Q.sum_pt - 559.6875)
    if Q.girth2_top2 < 0.001056655216 and Q.log_sum_pt > 6.327378592257:
        z += -3386.675048828125 * (0.001056655216 - Q.girth2_top2) * (Q.log_sum_pt - 6.327378592257)
    if Q.girth2 > 0.007520088344 and Q.log_sum_pt > 6.19222188581:
        z += -1157.81103515625 * (Q.girth2 - 0.007520088344) * (Q.log_sum_pt - 6.19222188581)
    if Q.width < 0.005590288644 and Q.n_dr_0p1_0p2 < 3.0:
        z += 75.88017272949219 * (0.005590288644 - Q.width) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.LHA < 0.293190627853 and Q.m012 < 16.899120053094:
        z += -0.25709688663482666 * (0.293190627853 - Q.LHA) * (16.899120053094 - Q.m012)
    if Q.e2 < 0.038466955721 and Q.D2 < 1.122624260187:
        z += 176.35147094726562 * (0.038466955721 - Q.e2) * (1.122624260187 - Q.D2)
    if Q.girth2_top2 < 0.001056655216 and Q.n_dr_0p2_0p4 < 1.0:
        z += 791.5525512695312 * (0.001056655216 - Q.girth2_top2) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.width < 0.005590288644 and Q.n_dr_0p2_0p4 < 1.0:
        z += -158.79415893554688 * (0.005590288644 - Q.width) * (1.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.6172708868980408
    if Q.girth2 < 0.000964142894:
        z += -1555.366943359375 * Q.girth2 - 0.27096842623245365
    if 0.000964142894 <= Q.girth2 < 0.006679471442:
        z += 309.792236328125 * Q.girth2 - 2.0692483955070258
    if Q.width < 0.000319370692:
        z += -5393.9571533203125 * Q.width + 10.546935648345906
    if 0.000319370692 <= Q.width < 0.005019718802:
        z += -1877.3638916015625 * Q.width + 9.423838824868252
    if Q.LHA < 0.196739721581:
        z += 17.91229248046875 * Q.LHA - 3.524059435484862
    if Q.mass < 21.784077072144:
        z += 0.11205773800611496 * Q.mass - 2.4410744012553285
    if Q.girth < 0.061086014472:
        z += 31.9608211517334 * Q.girth - 1.9523591834117902
    if Q.sum_pt_top5 >= 658.125:
        z += 0.005760885309427977 * Q.sum_pt_top5 - 3.791382644267287
    if Q.max_dr < 0.177304983139:
        z += -5.607046127319336 * Q.max_dr + 0.9941572190639502
    if Q.girth2_top5 < 0.000222950415:
        z += 6694.51025390625 * Q.girth2_top5 - 1.4925438393301538
    if Q.C2 < 0.027029510401:
        z += 50.047645568847656 * Q.C2 - 1.3527633564487294
    if Q.lam2 < 0.000194798295:
        z += -3532.73193359375 * Q.lam2 + 0.6881701573561158
    if Q.girth2_top3 < 0.000823693417:
        z += 1364.0670166015625 * Q.girth2_top3 - 1.1235730219215367
    if Q.width < 0.005019718802 and Q.pt_7 < 48.71875:
        z += -7.481902122497559 * (0.005019718802 - Q.width) * (48.71875 - Q.pt_7)
    if Q.girth2 < 0.006679471442 and Q.centroid_offset < 0.023554160423:
        z += 38550.43359375 * (0.006679471442 - Q.girth2) * (0.023554160423 - Q.centroid_offset)
    if Q.girth2 < 0.006679471442 and Q.planar_flow < 0.40079469091:
        z += 619.1161499023438 * (0.006679471442 - Q.girth2) * (0.40079469091 - Q.planar_flow)
    if Q.width < 0.005019718802 and Q.z_dr_0p2_0p4 < 0.1009733513:
        z += 6739.0244140625 * (0.005019718802 - Q.width) * (0.1009733513 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.061086014472 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += -225.6107177734375 * (0.061086014472 - Q.girth) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.701242202626 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += -43.22272872924805 * (Q.log_sum_pt - 6.701242202626) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.width < 0.005019718802 and Q.centroid_offset > 0.006789738266:
        z += 39202.27734375 * (0.005019718802 - Q.width) * (Q.centroid_offset - 0.006789738266)
    if Q.girth < 0.061086014472 and Q.lam1 > 0.00027588256:
        z += -21716.611328125 * (0.061086014472 - Q.girth) * (Q.lam1 - 0.00027588256)
    if Q.width < 0.005019718802 and Q.mass_over_sum_pt_sq > 0.00012320649:
        z += -294633.96875 * (0.005019718802 - Q.width) * (Q.mass_over_sum_pt_sq - 0.00012320649)
    if Q.max_dr < 0.177304983139 and Q.lam2 < 0.000194798295:
        z += 36010.078125 * (0.177304983139 - Q.max_dr) * (0.000194798295 - Q.lam2)
    if Q.log_sum_pt > 6.701242202626 and Q.pt_7 < 48.71875:
        z += 0.18536098301410675 * (Q.log_sum_pt - 6.701242202626) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242202626 and Q.girth2 < 0.018827652745:
        z += -331.8063049316406 * (Q.log_sum_pt - 6.701242202626) * (0.018827652745 - Q.girth2)
    if Q.log_sum_pt > 6.701242202626 and Q.mass_over_sum_pt_sq < 0.00023679558:
        z += 28792.296875 * (Q.log_sum_pt - 6.701242202626) * (0.00023679558 - Q.mass_over_sum_pt_sq)
    if Q.mass < 21.784077072144 and Q.centroid_offset < 0.026856224803:
        z += 2.9585981369018555 * (21.784077072144 - Q.mass) * (0.026856224803 - Q.centroid_offset)
    if Q.LHA < 0.196739721581 and Q.width < 0.000561123155:
        z += 46513.1796875 * (0.196739721581 - Q.LHA) * (0.000561123155 - Q.width)
    if Q.girth2 < 0.006679471442 and Q.log_sum_pt > 6.670067010936:
        z += -893.5982666015625 * (0.006679471442 - Q.girth2) * (Q.log_sum_pt - 6.670067010936)
    if Q.mass < 29.644699859619 and Q.centroid_offset < 0.023554160423:
        z += -7.15792179107666 * (29.644699859619 - Q.mass) * (0.023554160423 - Q.centroid_offset)
    if Q.girth < 0.061086014472 and Q.width < 0.005019718802:
        z += -26877.736328125 * (0.061086014472 - Q.girth) * (0.005019718802 - Q.width)
    if Q.girth < 0.061086014472 and Q.lam2 < 0.000194798295:
        z += 196083.171875 * (0.061086014472 - Q.girth) * (0.000194798295 - Q.lam2)
    if Q.girth < 0.061086014472 and Q.centroid_offset > 0.006789738266:
        z += -7483.2783203125 * (0.061086014472 - Q.girth) * (Q.centroid_offset - 0.006789738266)
    if Q.e2 < 0.024547699839 and Q.centroid_offset < 0.031170772021:
        z += 1904.3818359375 * (0.024547699839 - Q.e2) * (0.031170772021 - Q.centroid_offset)
    if Q.girth2 < 0.006679471442 and Q.centroid_offset > 0.018377780003:
        z += -49875.0859375 * (0.006679471442 - Q.girth2) * (Q.centroid_offset - 0.018377780003)
    if Q.LHA < 0.196739721581 and Q.lam2 < 0.000306123359:
        z += -66155.21875 * (0.196739721581 - Q.LHA) * (0.000306123359 - Q.lam2)
    if Q.mass < 15.454033088684 and Q.z_7 < 0.058613700176:
        z += 2.5783169269561768 * (15.454033088684 - Q.mass) * (0.058613700176 - Q.z_7)
    if Q.mass < 21.784077072144 and Q.centroid_offset > 0.016278845848:
        z += 12.129545211791992 * (21.784077072144 - Q.mass) * (Q.centroid_offset - 0.016278845848)
    if Q.z_dr_0_0p05 > 0.847731333971 and Q.lam2 < 0.000537286005:
        z += -6533.7822265625 * (Q.z_dr_0_0p05 - 0.847731333971) * (0.000537286005 - Q.lam2)
    if Q.mass < 21.784077072144 and Q.centroid_offset > 0.031170772021:
        z += -110.11437225341797 * (21.784077072144 - Q.mass) * (Q.centroid_offset - 0.031170772021)
    if Q.C2 < 0.027029510401 and Q.width < 0.004372139331:
        z += -17301.56640625 * (0.027029510401 - Q.C2) * (0.004372139331 - Q.width)
    if Q.C2 < 0.027029510401 and Q.width < 0.000964142901:
        z += 76634.1953125 * (0.027029510401 - Q.C2) * (0.000964142901 - Q.width)
    if Q.LHA < 0.196739721581 and Q.girth2 > 0.006679471442:
        z += 480092.75 * (0.196739721581 - Q.LHA) * (Q.girth2 - 0.006679471442)
    if Q.C2 < 0.027029510401 and Q.width < 0.006096650059:
        z += -7113.23046875 * (0.027029510401 - Q.C2) * (0.006096650059 - Q.width)
    if Q.sum_pt_top5 > 658.125 and Q.girth2 < 0.0016538364:
        z += -7.197484970092773 * (Q.sum_pt_top5 - 658.125) * (0.0016538364 - Q.girth2)
    if Q.C2 < 0.027029510401 and Q.centroid_offset > 0.016278845848:
        z += -6009.7431640625 * (0.027029510401 - Q.C2) * (Q.centroid_offset - 0.016278845848)
    if Q.mass < 21.784077072144 and Q.lam2 < 0.000194798295:
        z += -463.6203308105469 * (21.784077072144 - Q.mass) * (0.000194798295 - Q.lam2)
    if Q.mass < 15.454033088684 and Q.width < 0.000964142901:
        z += -141.08474731445312 * (15.454033088684 - Q.mass) * (0.000964142901 - Q.width)
    if Q.e2 < 0.024547699839 and Q.eccentricity > 0.872657364787:
        z += -429.1506652832031 * (0.024547699839 - Q.e2) * (Q.eccentricity - 0.872657364787)
    if Q.log_sum_pt > 6.701242202626 and Q.width < 0.006679471358:
        z += -827.8920288085938 * (Q.log_sum_pt - 6.701242202626) * (0.006679471358 - Q.width)
    if Q.girth < 0.061086014472 and Q.log_sum_pt > 6.670067010936:
        z += 336.7549133300781 * (0.061086014472 - Q.girth) * (Q.log_sum_pt - 6.670067010936)
    if Q.e2 < 0.016554418951 and Q.mass_top5 < 9.257203159811:
        z += 9.776162147521973 * (0.016554418951 - Q.e2) * (9.257203159811 - Q.mass_top5)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.8856210708618164
    if Q.girth < 0.033604209498:
        z += 128.84294509887695 * Q.girth - 6.116760108206564
    if 0.033604209498 <= Q.girth < 0.054649224505:
        z += 84.917724609375 * Q.girth - 4.640687796631498
    if Q.e2 < 0.020459658932:
        z += -173.48016738891602 * Q.e2 + 3.187844360989061
    if 0.020459658932 <= Q.e2 < 0.032346998155:
        z += 30.410564422607422 * Q.e2 - 0.983690471270591
    if Q.mass < 29.644699859619:
        z += 0.08413884043693542 * Q.mass - 3.3581408300334807
    if 29.644699859619 <= Q.mass < 41.377904891968:
        z += -0.02741040289402008 * Q.mass - 0.051296991919698165
    if 41.377904891968 <= Q.mass < 53.332374954224:
        z += 0.09916642308235168 * Q.mass - 5.288780858697193
    if Q.lam2 >= 0.001130644719:
        z += 997.0704345703125 * Q.lam2 - 1.127332421317959
    if Q.width < 0.000172198326:
        z += -12905.277099609375 * Q.width + 12.86690860603936
    if 0.000172198326 <= Q.width < 0.006096650059:
        z += -1796.730224609375 * Q.width + 10.954035429871828
    if Q.girth2 < 0.000964142894:
        z += -4038.0665283203125 * Q.girth2 + 10.709453569624461
    if 0.000964142894 <= Q.girth2 < 0.004372139461:
        z += -1525.5740966796875 * Q.girth2 + 8.287051845429371
    if 0.004372139461 <= Q.girth2 < 0.007520088344:
        z += -513.6770629882812 * Q.girth2 + 3.8628968939583275
    if Q.girth2 >= 0.018827652745:
        z += 138.78607177734375 * Q.girth2 - 2.613015965266473
    if Q.lam1 < 0.001503553356:
        z += 1016.0163116455078 * Q.lam1 - 2.4965796929098936
    if 0.001503553356 <= Q.lam1 < 0.005954149834:
        z += 217.7112579345703 * Q.lam1 - 1.296285450291053
    if Q.max_dr < 0.221586732566:
        z += -13.075275421142578 * Q.max_dr + 2.8973075579715135
    if Q.C2 >= 0.051192347892:
        z += 8.534798622131348 * Q.C2 - 0.4369163802523102
    if Q.centroid_offset < 0.018377780003:
        z += -137.01158142089844 * Q.centroid_offset + 2.5179687012163936
    if Q.log_sum_pt >= 6.377722943814:
        z += -2.2872304916381836 * Q.log_sum_pt + 14.587322384311818
    if Q.mass_over_sum_pt < 0.076373631775:
        z += 59.65558624267578 * Q.mass_over_sum_pt - 4.556113777019876
    if Q.pt_5 < 35.5:
        z += -0.07388392090797424 * Q.pt_5 + 2.6228791922330856
    if Q.sum_pt >= 840.01953125:
        z += -0.004977200645953417 * Q.sum_pt + 4.180945753550986
    if Q.girth2_top2 < 0.000759634834:
        z += -1996.1036376953125 * Q.girth2_top2 + 1.516309855467475
    if Q.mass_over_sum_pt_sq < 0.0030131544:
        z += -515.4534912109375 * Q.mass_over_sum_pt_sq + 1.5531409550375976
    if Q.mass < 53.332374954224 and Q.centroid_offset < 0.026856224803:
        z += 4.466829299926758 * (53.332374954224 - Q.mass) * (0.026856224803 - Q.centroid_offset)
    if Q.mass < 53.332374954224 and Q.lam1 < 0.000872228216:
        z += -27.17451286315918 * (53.332374954224 - Q.mass) * (0.000872228216 - Q.lam1)
    if Q.e2 < 0.032346998155 and Q.dr01 < 0.055953954317:
        z += 869.7527465820312 * (0.032346998155 - Q.e2) * (0.055953954317 - Q.dr01)
    if Q.mass < 53.332374954224 and Q.log_sum_pt < 6.842716632804:
        z += 0.21567653119564056 * (53.332374954224 - Q.mass) * (6.842716632804 - Q.log_sum_pt)
    if Q.max_dr < 0.221586732566 and Q.z_top5 < 0.90890302062:
        z += -36.52485656738281 * (0.221586732566 - Q.max_dr) * (0.90890302062 - Q.z_top5)
    if Q.width < 0.006096650059 and Q.C2 > 0.030867108516:
        z += -11169.1357421875 * (0.006096650059 - Q.width) * (Q.C2 - 0.030867108516)
    if Q.width < 0.006096650059 and Q.centroid_offset > 0.003343241496:
        z += -47078.44140625 * (0.006096650059 - Q.width) * (Q.centroid_offset - 0.003343241496)
    if Q.log_sum_pt > 6.377722943814 and Q.dr_5 < 0.021648628542:
        z += 163.74029541015625 * (Q.log_sum_pt - 6.377722943814) * (0.021648628542 - Q.dr_5)
    if Q.girth < 0.054649224505 and Q.dr_5 < 0.021648628542:
        z += -961.2159423828125 * (0.054649224505 - Q.girth) * (0.021648628542 - Q.dr_5)
    if Q.mass < 53.332374954224 and Q.planar_flow < 0.322073846732:
        z += 0.310873806476593 * (53.332374954224 - Q.mass) * (0.322073846732 - Q.planar_flow)
    if Q.centroid_offset < 0.018377780003 and Q.pt_4 > 47.34375:
        z += 2.0294907093048096 * (0.018377780003 - Q.centroid_offset) * (Q.pt_4 - 47.34375)
    if Q.centroid_offset < 0.018377780003 and Q.z_4 > 0.047491459878:
        z += -1578.69091796875 * (0.018377780003 - Q.centroid_offset) * (Q.z_4 - 0.047491459878)
    if Q.e2 < 0.020459658932 and Q.eccentricity > 0.903125533696:
        z += -951.6611328125 * (0.020459658932 - Q.e2) * (Q.eccentricity - 0.903125533696)
    if Q.mass < 41.377904891968 and Q.planar_flow < 0.322073846732:
        z += -0.24348866939544678 * (41.377904891968 - Q.mass) * (0.322073846732 - Q.planar_flow)
    if Q.e2 < 0.032346998155 and Q.tau21 < 0.446608647704:
        z += -222.02362060546875 * (0.032346998155 - Q.e2) * (0.446608647704 - Q.tau21)
    if Q.girth2 > 0.018827652745 and Q.pt_7 < 29.0421875:
        z += 54.92352294921875 * (Q.girth2 - 0.018827652745) * (29.0421875 - Q.pt_7)
    if Q.centroid_offset < 0.009480684835 and Q.mean_phi2 < 0.002127561159:
        z += -39520.390625 * (0.009480684835 - Q.centroid_offset) * (0.002127561159 - Q.mean_phi2)
    if Q.e2 < 0.020459658932 and Q.pt_7 < 53.4375:
        z += -4.629276752471924 * (0.020459658932 - Q.e2) * (53.4375 - Q.pt_7)
    if Q.sum_pt > 988.4078125 and Q.dr_3 < 0.060129364309:
        z += 0.4355114996433258 * (Q.sum_pt - 988.4078125) * (0.060129364309 - Q.dr_3)
    if Q.log_sum_pt > 6.377722943814 and Q.dr_2 < 0.027807975573:
        z += 82.92964172363281 * (Q.log_sum_pt - 6.377722943814) * (0.027807975573 - Q.dr_2)
    if Q.lam1 < 0.005954149834 and Q.dr_2 < 0.027807975573:
        z += -6549.06298828125 * (0.005954149834 - Q.lam1) * (0.027807975573 - Q.dr_2)
    if Q.girth2_top2 < 0.000759634834 and Q.z_7 > 0.023207568189:
        z += -53738.79296875 * (0.000759634834 - Q.girth2_top2) * (Q.z_7 - 0.023207568189)
    if Q.girth2_top2 < 0.000759634834 and Q.pt_7 > 33.21875:
        z += 108.72823333740234 * (0.000759634834 - Q.girth2_top2) * (Q.pt_7 - 33.21875)
    if Q.girth2 < 0.007520088344 and Q.pt_7 > 20.125:
        z += -4.501173973083496 * (0.007520088344 - Q.girth2) * (Q.pt_7 - 20.125)
    if Q.mass < 41.377904891968 and Q.centroid_offset < 0.026856224803:
        z += -9.102798461914062 * (41.377904891968 - Q.mass) * (0.026856224803 - Q.centroid_offset)
    if Q.e2 < 0.016554418951 and Q.centroid_offset < 0.023554160423:
        z += 11472.591796875 * (0.016554418951 - Q.e2) * (0.023554160423 - Q.centroid_offset)
    if Q.girth2 < 0.004372139461 and Q.centroid_offset > 0.010960638421:
        z += -59443.01171875 * (0.004372139461 - Q.girth2) * (Q.centroid_offset - 0.010960638421)
    if Q.mass_over_sum_pt < 0.076373631775 and Q.girth2_top2 < 0.000759634834:
        z += -64617.85546875 * (0.076373631775 - Q.mass_over_sum_pt) * (0.000759634834 - Q.girth2_top2)
    if Q.lam1 < 0.005954149834 and Q.mean_phi2 < 0.002127561159:
        z += -95750.734375 * (0.005954149834 - Q.lam1) * (0.002127561159 - Q.mean_phi2)
    return max(0.0, z)


def neuron_10(Q):
    z = -0.5241696834564209
    z += 63.79941177368164 * Q.e2
    if Q.lam2 < 0.000194798295:
        z += 553.4133911132812 * Q.lam2 - 1.8862480787513352
    if 0.000194798295 <= Q.lam2 < 0.003408388935:
        z += 1111.2244873046875 * Q.lam2 - 1.994908729221502
    if Q.lam2 >= 0.003408388935:
        z += 557.8110961914062 * Q.lam2 - 0.10866065047016693
    if 0.303313749495 <= Q.LHA < 0.346713497427:
        z += -10.305388450622559 * Q.LHA + 3.125766010960797
    if Q.LHA >= 0.346713497427:
        z += -36.7804536819458 * Q.LHA + 12.305028471920844
    if 0.00231612516 <= Q.centroid_offset < 0.037760993714:
        z += -22.884929656982422 * Q.centroid_offset + 0.053004361363367156
    if Q.centroid_offset >= 0.037760993714:
        z += 17.836158752441406 * Q.centroid_offset - 1.4846644020921242
    if Q.lam1 < 0.004183811014:
        z += 650.6764526367188 * Q.lam1 - 2.722307309091953
    if Q.lam1 >= 0.008375572068:
        z += -202.1414337158203 * Q.lam1 + 1.693050146015698
    if Q.n_dr_0p05_0p1 < 3.0:
        z += -0.1801043450832367 * Q.n_dr_0p05_0p1 + 0.5403130352497101
    if Q.tau21 < 0.391541349888:
        z += 3.013148069381714 * Q.tau21 - 1.1797720624981374
    if Q.mass_over_sum_pt_sq < 0.003904593248:
        z += -353.10101318359375 * Q.mass_over_sum_pt_sq + 1.378715831938619
    if Q.log_sum_pt >= 6.701242202626:
        z += -22.03592300415039 * Q.log_sum_pt + 147.66805720922972
    if 15.454033088684 <= Q.mass < 53.332374954224:
        z += 0.0561971552670002 * Q.mass - 0.8684726969861334
    if Q.mass >= 53.332374954224:
        z += 0.0068466514348983765 * Q.mass + 1.7635068775673894
    if Q.girth2 < 0.0016538364:
        z += 1010.096435546875 * Q.girth2 - 1.6705342526176759
    if Q.girth2 >= 0.008678044751:
        z += 394.14129638671875 * Q.girth2 - 3.4203758082610998
    if Q.girth2_top5 < 0.002270363079:
        z += -455.8287658691406 * Q.girth2_top5 + 1.0348968003754324
    if Q.sum_pt >= 813.415625:
        z += 0.011263282038271427 * Q.sum_pt - 9.161729598711826
    if Q.mass_over_sum_pt < 0.107985668755:
        z += -21.018348693847656 * Q.mass_over_sum_pt + 2.26968043983092
    if Q.girth2_top2 < 0.004007841607:
        z += -235.20639038085938 * Q.girth2_top2 + 0.9426699576006927
    if Q.lam2 > 0.000194798295 and Q.n_pt_above_50 < 8.0:
        z += -60.60557556152344 * (Q.lam2 - 0.000194798295) * (8.0 - Q.n_pt_above_50)
    if Q.eccentricity > 0.903125533696 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += 156.74400329589844 * (Q.eccentricity - 0.903125533696) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.lam2 > 0.000194798295 and Q.D2 < 2.055451202393:
        z += -200.43411254882812 * (Q.lam2 - 0.000194798295) * (2.055451202393 - Q.D2)
    if Q.lam1 < 0.004183811014 and Q.sum_pt > 988.4078125:
        z += -13.032033920288086 * (0.004183811014 - Q.lam1) * (Q.sum_pt - 988.4078125)
    if Q.lam1 < 0.004183811014 and Q.log_sum_pt > 6.701242202626:
        z += 807.4105834960938 * (0.004183811014 - Q.lam1) * (Q.log_sum_pt - 6.701242202626)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.1660556197166443
    if Q.planar_flow < 0.253403707141:
        z += -8.146034240722656 * Q.planar_flow + 2.0642352750966424
    if Q.girth < 0.02054281719:
        z += 141.29620361328125 * Q.girth - 7.797990861233436
    if 0.02054281719 <= Q.girth < 0.076081777364:
        z += 73.40077209472656 * Q.girth - 6.403227423511603
    if 0.076081777364 <= Q.girth < 0.087236513197:
        z += -110.39033508300781 * Q.girth + 7.579926674267846
    if 0.087236513197 <= Q.girth < 0.101940929517:
        z += -183.79110717773438 * Q.girth + 13.98315409777945
    if Q.girth >= 0.101940929517:
        z += -273.0926284790039 * Q.girth + 23.086634186513038
    if Q.centroid_offset < 0.014379521101:
        z += -116.94438934326172 * Q.centroid_offset + 3.4231589474200357
    if 0.014379521101 <= Q.centroid_offset < 0.049903668404:
        z += -196.35629272460938 * Q.centroid_offset + 4.565064087762698
    if Q.centroid_offset >= 0.049903668404:
        z += -127.76095581054688 * Q.centroid_offset + 1.1419051403426619
    if Q.width < 0.003562611091:
        z += -511.8034362792969 * Q.width + 6.76773845548721
    if 0.003562611091 <= Q.width < 0.008678044951:
        z += -966.5615844726562 * Q.width + 8.387864877963494
    if Q.e2_sq < 0.006390124748:
        z += 784.6260986328125 * Q.e2_sq - 5.013858650800224
    if Q.girth2 < 0.006679471442:
        z += -1307.0616455078125 * Q.girth2 + 10.828808451783845
    if 0.006679471442 <= Q.girth2 < 0.013238675334:
        z += -319.90582275390625 * Q.girth2 + 4.235129324895114
    if Q.lam1 < 0.004839980301:
        z += 532.2307281494141 * Q.lam1 - 3.9000030995998225
    if 0.004839980301 <= Q.lam1 < 0.008375572068:
        z += 374.4823913574219 * Q.lam1 - 3.136504257011067
    if Q.sum_pt_top5 < 687.4375:
        z += -0.001997138373553753 * Q.sum_pt_top5 + 1.372907810669858
    if Q.girth2_top3 < 0.002151567843:
        z += 283.59600830078125 * Q.girth2_top3 - 0.6101760518631221
    if Q.C2 < 0.035786485299:
        z += 28.105024337768555 * Q.C2 - 1.0057800402915915
    if Q.max_dr < 0.111761856824:
        z += 5.686001300811768 * Q.max_dr - 0.2378919684980506
    if 0.111761856824 <= Q.max_dr < 0.221586732566:
        z += -3.620182514190674 * Q.max_dr + 0.8021844146120783
    if Q.log_sum_pt < 6.701242202626:
        z += -2.4217636585235596 * Q.log_sum_pt + 16.22882483328402
    if Q.mass < 21.784077072144:
        z += -0.07632818818092346 * Q.mass + 1.6627391341103475
    if Q.LHA < 0.196739721581:
        z += 17.626785278320312 * Q.LHA - 3.467888828024808
    if Q.planar_flow < 0.253403707141 and Q.width > 0.006096650059:
        z += -4132.68017578125 * (0.253403707141 - Q.planar_flow) * (Q.width - 0.006096650059)
    if Q.planar_flow < 0.253403707141 and Q.D2 < 1.679198372364:
        z += 1.7183070182800293 * (0.253403707141 - Q.planar_flow) * (1.679198372364 - Q.D2)
    if Q.planar_flow < 0.253403707141 and Q.girth2 > 0.013238675334:
        z += 1373.97412109375 * (0.253403707141 - Q.planar_flow) * (Q.girth2 - 0.013238675334)
    if Q.planar_flow < 0.253403707141 and Q.mass < 69.611351776123:
        z += -0.11596707254648209 * (0.253403707141 - Q.planar_flow) * (69.611351776123 - Q.mass)
    if Q.centroid_offset < 0.049903668404 and Q.pt_7 < 48.71875:
        z += -1.0786479711532593 * (0.049903668404 - Q.centroid_offset) * (48.71875 - Q.pt_7)
    if Q.centroid_offset < 0.049903668404 and Q.log_sum_pt < 6.804164030582:
        z += -171.66868591308594 * (0.049903668404 - Q.centroid_offset) * (6.804164030582 - Q.log_sum_pt)
    if Q.centroid_offset < 0.049903668404 and Q.mean_phi2 < 0.001570267399:
        z += -8656.7353515625 * (0.049903668404 - Q.centroid_offset) * (0.001570267399 - Q.mean_phi2)
    if Q.LHA < 0.154689112391 and Q.z_7 < 0.028070914944:
        z += 881.0592651367188 * (0.154689112391 - Q.LHA) * (0.028070914944 - Q.z_7)
    if Q.girth > 0.076081777364 and Q.n_pt_above_50 < 7.0:
        z += 53.344871520996094 * (Q.girth - 0.076081777364) * (7.0 - Q.n_pt_above_50)
    if Q.sum_pt_top5 < 687.4375 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.0012279595248401165 * (687.4375 - Q.sum_pt_top5) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.girth > 0.101940929517 and Q.n_pt_above_50 < 7.0:
        z += -65.79352569580078 * (Q.girth - 0.101940929517) * (7.0 - Q.n_pt_above_50)
    if Q.girth > 0.076081777364 and Q.z_dr_0p1_0p2 < 0.20502409339:
        z += -809.5518798828125 * (Q.girth - 0.076081777364) * (0.20502409339 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt < 6.701242202626 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += 56.349365234375 * (6.701242202626 - Q.log_sum_pt) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.girth > 0.076081777364 and Q.pt_7 < 40.040625:
        z += -7.174622535705566 * (Q.girth - 0.076081777364) * (40.040625 - Q.pt_7)
    if Q.log_sum_pt < 6.701242202626 and Q.dr_0 < 0.111955475493:
        z += -22.59648895263672 * (6.701242202626 - Q.log_sum_pt) * (0.111955475493 - Q.dr_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.9238681793212891
    if Q.girth2 >= 0.018827652745:
        z += 259.0993347167969 * Q.girth2 - 4.878232300508374
    if Q.e2 >= 0.063441075385:
        z += -82.96912384033203 * Q.e2 + 5.263650440181905
    if Q.girth2 > 0.018827652745 and Q.pt_7 > 15.55390625:
        z += 4.818659782409668 * (Q.girth2 - 0.018827652745) * (Q.pt_7 - 15.55390625)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.402775287628174
    if Q.girth < 0.148408418149:
        z += -42.355857849121094 * Q.girth + 6.285965862731967
    if Q.lam1 < 0.006506575659:
        z += -443.32020568847656 * Q.lam1 + 5.491059189203535
    if 0.006506575659 <= Q.lam1 < 0.016433749775:
        z += -262.5684509277344 * Q.lam1 + 4.314984221355753
    if 658.125 <= Q.sum_pt_top5 < 902.40625:
        z += -0.007595764007419348 * Q.sum_pt_top5 + 4.998962187382858
    if Q.sum_pt_top5 >= 902.40625:
        z += 0.24361716071143746 * Q.sum_pt_top5 - 221.69715115969302
    if Q.lam2 < 0.000306123359:
        z += -3421.046142578125 * Q.lam2 + 1.0472621364600085
    if Q.e2 < 0.050284641981:
        z += 123.40970611572266 * Q.e2 - 6.20561288900954
    if Q.pt_6 < 31.90625:
        z += 0.09368681907653809 * Q.pt_6 - 2.9891950711607933
    if Q.z_top5_slots >= 0.930764273368:
        z += -609.7801513671875 * Q.z_top5_slots + 567.5615795015093
    if Q.z_7 < 0.028070914944:
        z += 207.82830810546875 * Q.z_7 - 5.833930759784039
    if Q.C2 >= 0.067292226106:
        z += -55.2353630065918 * Q.C2 + 3.716910536486563
    if Q.centroid_offset < 0.037760993714:
        z += -61.02037048339844 * Q.centroid_offset + 2.3041898262495595
    if Q.LHA >= 0.09323897448:
        z += -4.991696357727051 * Q.LHA + 0.4654206493100214
    if Q.width < 0.00752008842:
        z += 77.48333740234375 * Q.width + 0.6622405108089886
    if 0.00752008842 <= Q.width < 0.013238675006:
        z += -217.697509765625 * Q.width + 2.8820265814026205
    if Q.mass >= 53.332374954224:
        z += -0.05105486884713173 * Q.mass + 2.7228774085939595
    if Q.pt_7 < 25.578125:
        z += 0.09597271680831909 * Q.pt_7 - 2.4548021471127868
    if Q.sum_pt >= 988.4078125:
        z += 0.19783245027065277 * Q.sum_pt - 195.53913941353093
    if Q.girth < 0.148408418149 and Q.log_sum_pt < 6.804164030582:
        z += -47.55543518066406 * (0.148408418149 - Q.girth) * (6.804164030582 - Q.log_sum_pt)
    if Q.girth < 0.148408418149 and Q.pt_7 < 38.53125:
        z += -0.9023911952972412 * (0.148408418149 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.z_7 > 0.023207568189:
        z += -0.5466576218605042 * (Q.sum_pt_top5 - 658.125) * (Q.z_7 - 0.023207568189)
    if Q.lam2 < 0.000306123359 and Q.centroid_offset < 0.049903668404:
        z += -152474.328125 * (0.000306123359 - Q.lam2) * (0.049903668404 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 43.5:
        z += 0.00081919867079705 * (Q.sum_pt_top5 - 658.125) * (43.5 - Q.pt_7)
    if Q.lam1 < 0.016433749775 and Q.pt_6 < 56.53125:
        z += -1.8690171241760254 * (0.016433749775 - Q.lam1) * (56.53125 - Q.pt_6)
    if Q.e2 < 0.050284641981 and Q.pt_dispersion > 0.396830244362:
        z += 130.56898498535156 * (0.050284641981 - Q.e2) * (Q.pt_dispersion - 0.396830244362)
    if Q.sum_pt_top5 > 658.125 and Q.tau32 < 0.362731824815:
        z += -0.1395409256219864 * (Q.sum_pt_top5 - 658.125) * (0.362731824815 - Q.tau32)
    if Q.lam1 < 0.016433749775 and Q.centroid_offset < 0.037760993714:
        z += -3625.890869140625 * (0.016433749775 - Q.lam1) * (0.037760993714 - Q.centroid_offset)
    if Q.sum_pt_top5 > 902.40625 and Q.D2 < 3.885568320751:
        z += 0.09180247038602829 * (Q.sum_pt_top5 - 902.40625) * (3.885568320751 - Q.D2)
    if Q.tau21 < 0.501026660204 and Q.max_dr > 0.015595615841:
        z += -15.389939308166504 * (0.501026660204 - Q.tau21) * (Q.max_dr - 0.015595615841)
    if Q.sum_pt_top5 > 902.40625 and Q.n_pt_above_50 > 6.0:
        z += -0.24779272079467773 * (Q.sum_pt_top5 - 902.40625) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt_top5 > 839.9546875 and Q.n_pt_above_50 > 2.0:
        z += 0.011189950630068779 * (Q.sum_pt_top5 - 839.9546875) * (Q.n_pt_above_50 - 2.0)
    if Q.girth > 0.101940929517 and Q.pt_4 > 81.375:
        z += -25.00275230407715 * (Q.girth - 0.101940929517) * (Q.pt_4 - 81.375)
    if Q.sum_pt > 988.4078125 and Q.n_pt_above_50 > 6.0:
        z += -0.13308829069137573 * (Q.sum_pt - 988.4078125) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt > 988.4078125 and Q.D2 < 3.885568320751:
        z += 0.029379718005657196 * (Q.sum_pt - 988.4078125) * (3.885568320751 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.9318941831588745
    if Q.z_dr_0p05_0p1 < 0.588259786367:
        z += -1.8717608451843262 * Q.z_dr_0p05_0p1 + 1.101081634918247
    if 0.002464291268 <= Q.lam1 < 0.004183811014:
        z += -236.2156524658203 * Q.lam1 + 0.5821041697364437
    if 0.004183811014 <= Q.lam1 < 0.00543336053:
        z += 615.6523895263672 * Q.lam1 - 2.981950726825085
    if 0.00543336053 <= Q.lam1 < 0.005954149834:
        z += 762.9464874267578 * Q.lam1 - 3.7822526646590235
    if 0.005954149834 <= Q.lam1 < 0.007330079875:
        z += 234.12562561035156 * Q.lam1 - 0.6335740180591314
    if 0.007330079875 <= Q.lam1 < 0.012003726523:
        z += 278.95475006103516 * Q.lam1 - 0.962175081008958
    if Q.lam1 >= 0.012003726523:
        z += 9.014442443847656 * Q.lam1 + 2.2781145491622556
    if Q.max_dr < 0.080507021025:
        z += 91.8994836807251 * Q.max_dr - 8.584986066841092
    if 0.080507021025 <= Q.max_dr < 0.177304983139:
        z += 12.256791114807129 * Q.max_dr - 2.173190141949123
    if Q.girth2 < 0.013238675334:
        z += -546.9459838867188 * Q.girth2 + 7.2408403059114645
    if Q.width < 0.006096650059:
        z += 1493.4993896484375 * Q.width - 8.707654712288779
    if 0.006096650059 <= Q.width < 0.00752008842:
        z += 185.0418701171875 * Q.width - 0.7304470986395915
    if 0.00752008842 <= Q.width < 0.008678044951:
        z += -570.9058227539062 * Q.width + 4.954346392646038
    if Q.mass_over_sum_pt_sq < 0.008174660116:
        z += 437.7492446899414 * Q.mass_over_sum_pt_sq - 4.0081281914669304
    if 0.008174660116 <= Q.mass_over_sum_pt_sq < 0.011660904657:
        z += 123.24921417236328 * Q.mass_over_sum_pt_sq - 1.4371973355141012
    if Q.girth < 0.033604209498:
        z += 165.71249389648438 * Q.girth - 12.279640010607775
    if 0.033604209498 <= Q.girth < 0.087236513197:
        z += 125.1298599243164 * Q.girth - 10.91589267662639
    if Q.e2 < 0.038466955721:
        z += -209.63511657714844 * Q.e2 + 9.144102476928657
    if 0.038466955721 <= Q.e2 < 0.041109715588:
        z += -140.84815979003906 * Q.e2 + 6.49807765601658
    if 0.041109715588 <= Q.e2 < 0.04447356835:
        z += -210.42831420898438 * Q.e2 + 9.358498014748543
    if Q.D2 < 1.122624260187:
        z += 0.6169647574424744 * Q.D2 - 0.6926196043853097
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.15273815393447876 * Q.n_dr_0p05_0p1 - 0.7636907696723938
    if 0.012587644117 <= Q.centroid_offset < 0.049903668404:
        z += -26.85478401184082 * Q.centroid_offset + 0.33803846397995374
    if Q.centroid_offset >= 0.049903668404:
        z += -250.65461921691895 * Q.centroid_offset + 11.506471228924017
    if Q.e2_sq < 0.005834489329:
        z += -554.8969345092773 * Q.e2_sq + 4.499146382669702
    if 0.005834489329 <= Q.e2_sq < 0.017162483186:
        z += -111.37065887451172 * Q.e2_sq + 1.911397060347549
    if Q.LHA < 0.303313749495:
        z += 9.542763710021973 * Q.LHA - 2.8944514414315816
    if Q.z_dr_0_0p05 >= 0.608073231578:
        z += -1.3623102903366089 * Q.z_dr_0_0p05 + 0.8283844206569452
    if Q.C2 < 0.067292226106:
        z += 40.0904655456543 * Q.C2 - 2.6977766721929717
    if Q.mass < 69.611351776123:
        z += 0.021768510341644287 * Q.mass - 1.0832879159367454
    if 69.611351776123 <= Q.mass < 86.4:
        z += -0.02573450282216072 * Q.mass + 2.2234610438346865
    if Q.tau21 < 0.135767506063:
        z += -10.295787811279297 * Q.tau21 + 1.3978334340912235
    if Q.planar_flow < 0.111513564951 and Q.centroid_offset > 0.018377780003:
        z += -655.6102905273438 * (0.111513564951 - Q.planar_flow) * (Q.centroid_offset - 0.018377780003)
    if Q.planar_flow < 0.111513564951 and Q.max_dr < 0.15984864831:
        z += -143.30369567871094 * (0.111513564951 - Q.planar_flow) * (0.15984864831 - Q.max_dr)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.C2 < 0.067292226106:
        z += -23.247072219848633 * (0.588259786367 - Q.z_dr_0p05_0p1) * (0.067292226106 - Q.C2)
    if Q.lam1 > 0.00543336053 and Q.max_dr < 0.15984864831:
        z += 8633.546875 * (Q.lam1 - 0.00543336053) * (0.15984864831 - Q.max_dr)
    if Q.planar_flow < 0.111513564951 and Q.centroid_offset > 0.009480684835:
        z += 430.7784729003906 * (0.111513564951 - Q.planar_flow) * (Q.centroid_offset - 0.009480684835)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.n_dr_0p1_0p2 < 3.0:
        z += 0.26897019147872925 * (0.588259786367 - Q.z_dr_0p05_0p1) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.girth2 < 0.013238675334 and Q.eccentricity > 0.970449631164:
        z += 3483.619873046875 * (0.013238675334 - Q.girth2) * (Q.eccentricity - 0.970449631164)
    if Q.girth2 < 0.004372139461 and Q.D2 < 0.87567204833:
        z += -9430.443359375 * (0.004372139461 - Q.girth2) * (0.87567204833 - Q.D2)
    if Q.e2 < 0.038466955721 and Q.D2 < 1.002470755577:
        z += -202.4539337158203 * (0.038466955721 - Q.e2) * (1.002470755577 - Q.D2)
    if Q.width < 0.008678044951 and Q.D2 < 1.002470755577:
        z += -589.5438232421875 * (0.008678044951 - Q.width) * (1.002470755577 - Q.D2)
    if Q.width < 0.00752008842 and Q.n_dr_0p1_0p2 < 3.0:
        z += 160.65774536132812 * (0.00752008842 - Q.width) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.girth2 < 0.013238675334 and Q.n_dr_0p1_0p2 < 3.0:
        z += -41.30394744873047 * (0.013238675334 - Q.girth2) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.girth2 < 0.013238675334 and Q.centroid_offset > 0.031170772021:
        z += -6567.30126953125 * (0.013238675334 - Q.girth2) * (Q.centroid_offset - 0.031170772021)
    if Q.lam1 > 0.004183811014 and Q.D2 > 0.415245993435:
        z += -792.1480102539062 * (Q.lam1 - 0.004183811014) * (Q.D2 - 0.415245993435)
    if Q.lam1 > 0.002464291268 and Q.D2 > 0.415245993435:
        z += 331.54473876953125 * (Q.lam1 - 0.002464291268) * (Q.D2 - 0.415245993435)
    if Q.lam1 > 0.005954149834 and Q.D2 > 1.679198372364:
        z += -3529.808837890625 * (Q.lam1 - 0.005954149834) * (Q.D2 - 1.679198372364)
    if Q.e2 < 0.041109715588 and Q.D2 < 1.002470755577:
        z += 223.32017517089844 * (0.041109715588 - Q.e2) * (1.002470755577 - Q.D2)
    if Q.lam1 > 0.002464291268 and Q.D2 > 1.679198372364:
        z += -330.6229553222656 * (Q.lam1 - 0.002464291268) * (Q.D2 - 1.679198372364)
    return max(0.0, z)


def neuron_15(Q):
    z = -2.5768327713012695
    if Q.tau21 < 0.23799610585:
        z += -3.903501510620117 * Q.tau21 + 0.9290181587071803
    if Q.width < 0.006679471358:
        z += 835.1043701171875 * Q.width - 2.242202061976994
    if 0.006679471358 <= Q.width < 0.00752008842:
        z += -176.023681640625 * Q.width + 4.511598799009656
    if 0.00752008842 <= Q.width < 0.013238675006:
        z += -557.4603271484375 * Q.width + 7.380036099856603
    if Q.girth2 < 0.018827652745:
        z += -80.09904479980469 * Q.girth2 + 1.5080770006969206
    if Q.lam1 < 0.006506575659:
        z += -81.88116455078125 * Q.lam1 - 1.1938593092964815
    if 0.006506575659 <= Q.lam1 < 0.008375572068:
        z += 484.29742431640625 * Q.lam1 - 4.877743134266693
    if 0.008375572068 <= Q.lam1 < 0.012003726523:
        z += 226.41680908203125 * Q.lam1 - 2.717845456431006
    if Q.e2_sq < 0.008168570676:
        z += 814.3985595703125 * Q.e2_sq - 7.971121826107981
    if 0.008168570676 <= Q.e2_sq < 0.011657374702:
        z += 377.9660949707031 * Q.e2_sq - 4.406092393725204
    if Q.mass_over_sum_pt_sq < 0.007182835724:
        z += -425.1596374511719 * Q.mass_over_sum_pt_sq + 3.0538518322871657
    if Q.mass < 36.229410171509:
        z += 0.4693494737148285 * Q.mass - 17.004254596996404
    if Q.mass >= 80.4:
        z += -0.4978416860103607 * Q.mass + 40.026471555233
    if Q.e2 < 0.024547699839:
        z += -444.4417610168457 * Q.e2 + 12.92867598765484
    if 0.024547699839 <= Q.e2 < 0.032346998155:
        z += -101.05708694458008 * Q.e2 + 4.499372079216018
    if 0.032346998155 <= Q.e2 < 0.038466955721:
        z += -73.25964546203613 * Q.e2 + 3.6002082908664486
    if 0.038466955721 <= Q.e2 < 0.041109715588:
        z += -111.53983497619629 * Q.e2 + 5.072730645899136
    if 0.041109715588 <= Q.e2 < 0.063441075385:
        z += -21.824007034301758 * Q.e2 + 1.3845384754659082
    if Q.z_dr_0p05_0p1 >= 0.750909513235:
        z += -21.087724685668945 * Q.z_dr_0p05_0p1 + 15.83497307894936
    if Q.z_dr_0p1_0p2 < 0.328461505473:
        z += -3.616687536239624 * Q.z_dr_0p1_0p2 + 1.1879426329787022
    if Q.log_sum_pt >= 6.896095378249:
        z += 32.07989501953125 * Q.log_sum_pt - 221.22601577890256
    if Q.planar_flow < 0.083662731125:
        z += 9.232673645019531 * Q.planar_flow - 0.7724306927281428
    if Q.mass_over_sum_pt < 0.06813910019:
        z += 86.37085723876953 * Q.mass_over_sum_pt - 5.885232494888704
    if Q.D2 < 0.74595130682:
        z += 2.3336715698242188 * Q.D2 - 1.740805357199057
    if Q.LHA >= 0.346713497427:
        z += -42.44835662841797 * Q.LHA + 14.71741818666737
    if Q.girth >= 0.033604209498:
        z += 42.51485824584961 * Q.girth - 1.428678203271303
    if Q.girth2_top3 < 0.002151567843:
        z += 529.7582397460938 * Q.girth2_top3 - 1.1398107932019799
    if Q.girth2_top2 < 0.004007841607:
        z += 126.43302917480469 * Q.girth2_top2 - 0.2656349122443089
    if 0.004007841607 <= Q.girth2_top2 < 0.007639643088:
        z += -66.38265991210938 * Q.girth2_top2 + 0.5071398289606011
    if Q.tau21 < 0.23799610585 and Q.z_dr_0p05_0p1 < 0.588259786367:
        z += -12.53006362915039 * (0.23799610585 - Q.tau21) * (0.588259786367 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.23799610585 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += 36.85041809082031 * (0.23799610585 - Q.tau21) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.tau21 < 0.23799610585 and Q.girth2_top5 > 0.00832969537:
        z += -1687.306640625 * (0.23799610585 - Q.tau21) * (Q.girth2_top5 - 0.00832969537)
    if Q.width < 0.00752008842 and Q.e2 > 0.024547699839:
        z += -68520.140625 * (0.00752008842 - Q.width) * (Q.e2 - 0.024547699839)
    if Q.LHA > 0.176724128067 and Q.sum_pt_top3 > 353.0625:
        z += 0.044715240597724915 * (Q.LHA - 0.176724128067) * (Q.sum_pt_top3 - 353.0625)
    if Q.z_dr_0p05_0p1 > 0.750909513235 and Q.n_dr_0p2_0p4 < 2.0:
        z += 7.620892524719238 * (Q.z_dr_0p05_0p1 - 0.750909513235) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.mass > 80.4 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += 2.068425178527832 * (Q.mass - 80.4) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.896095378249 and Q.mean_phi > 0.01746432744:
        z += 48039.00390625 * (Q.log_sum_pt - 6.896095378249) * (Q.mean_phi - 0.01746432744)
    if Q.LHA > 0.176724128067 and Q.z_top5 > 0.865048766136:
        z += -212.14987182617188 * (Q.LHA - 0.176724128067) * (Q.z_top5 - 0.865048766136)
    if Q.width < 0.006096650059 and Q.log_sum_pt > 6.896095378249:
        z += -7484.3232421875 * (0.006096650059 - Q.width) * (Q.log_sum_pt - 6.896095378249)
    if Q.girth2_top2 < 0.007639643088 and Q.mean_phi < 0.001618889696:
        z += 7422.5888671875 * (0.007639643088 - Q.girth2_top2) * (0.001618889696 - Q.mean_phi)
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
