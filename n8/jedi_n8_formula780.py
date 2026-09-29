"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned), as if-statements.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.4% (the network: 65.8%); same class as the network for 90.0% of jets.

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
  Q.pt_balance01           min(pT0, pT1) / (pT0 + pT1)
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_1                   pT of particle 1 [GeV]
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_5                   pT of particle 5 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_3                    pT of particle 3 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.z_top2_slots           pT share of the 2 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_1st                  largest pT share
  Q.pt1_over_pt0           pT1 / pT0
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr01                   ΔR between particles 0 and 1
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_2                  Δη of particle 2
  Q.eta_7                  Δη of particle 7
  Q.phi_0                  Δφ of particle 0
  Q.phi_1                  Δφ of particle 1
  Q.phi_6                  Δφ of particle 6
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
        pt_balance01=min(pt[0], pt[1]) / max(pt[0] + pt[1], 1e-9),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        pt_0=pt[0],
        pt_1=pt[1],
        pt_2=pt[2],
        pt_4=pt[4],
        pt_5=pt[5],
        pt_6=pt[6],
        pt_7=pt[7],
        z_3=z[3],
        z_4=z[4],
        z_6=z[6],
        z_7=z[7],
        z_top2_slots=sum(pt[:2]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_1st=zs[0],
        pt1_over_pt0=pt[1] / max(pt[0], 1e-9),
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr01=math.sqrt(dist2(0, 1)),
        eta_0=eta[0],
        eta_1=eta[1],
        eta_2=eta[2],
        eta_7=eta[7],
        phi_0=phi[0],
        phi_1=phi[1],
        phi_6=phi[6],
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
    z = -0.6333608627319336
    if Q.planar_flow < 0.012569162668:
        z += -42.64367437362671 * Q.planar_flow + 0.7320103031736314
    if 0.012569162668 <= Q.planar_flow < 0.148419710734:
        z += -1.4428725242614746 * Q.planar_flow + 0.21415072267692448
    if Q.width < 0.004372139331:
        z += -301.76995849609375 * Q.width + 4.667667652516455
    if 0.004372139331 <= Q.width < 0.008678044951:
        z += -777.603515625 * Q.width + 6.748078262649381
    if Q.girth2 < 0.013238675334:
        z += -488.5999755859375 * Q.girth2 + 6.831256174231018
    if 0.013238675334 <= Q.girth2 < 0.018827652745:
        z += -64.92059326171875 * Q.girth2 + 1.2223023859310276
    if Q.mass < 21.784077072144:
        z += 0.3185160215944052 * Q.mass - 9.38602343647765
    if 21.784077072144 <= Q.mass < 29.644699859619:
        z += 0.12377526424825191 * Q.mass - 5.143775769361357
    if 29.644699859619 <= Q.mass < 56.920347213745:
        z += 0.04709099791944027 * Q.mass - 2.8704937100886476
    if 56.920347213745 <= Q.mass < 64.618731689453:
        z += 0.024688005447387695 * Q.mass - 1.5953075999524995
    if Q.girth2_top3 < 0.003952581551:
        z += -51.12089538574219 * Q.girth2_top3 - 0.41638367238172536
    if 0.003952581551 <= Q.girth2_top3 < 0.007929074034:
        z += 155.52479553222656 * Q.girth2_top3 - 1.2331676178977369
    if Q.lam1 < 0.000504949057:
        z += 7171.288131713867 * Q.lam1 - 5.327312859920971
    if 0.000504949057 <= Q.lam1 < 0.00543336053:
        z += 250.3716278076172 * Q.lam1 - 1.832602597697774
    if 0.00543336053 <= Q.lam1 < 0.006506575659:
        z += 11.4952392578125 * Q.lam1 - 0.5347010566023211
    if 0.006506575659 <= Q.lam1 < 0.008375572068:
        z += 246.07131958007812 * Q.lam1 - 2.0609880710108035
    if Q.sum_pt >= 901.59375:
        z += -0.021665889769792557 * Q.sum_pt + 19.533830804633908
    if Q.girth2_top5 < 0.00832969537:
        z += 108.38434600830078 * Q.girth2_top5 - 0.902808585125821
    if Q.z_dr_0_0p05 >= 0.847731333971:
        z += 8.718156814575195 * Q.z_dr_0_0p05 - 7.390654706188195
    if Q.girth < 0.076081777364:
        z += 71.718505859375 * Q.girth - 5.7293581375955895
    if 0.076081777364 <= Q.girth < 0.087236513197:
        z += 24.463756561279297 * Q.girth - 2.1341328221062366
    if Q.centroid_offset < 0.031170772021:
        z += -24.9677791595459 * Q.centroid_offset + 0.7782649520528802
    if Q.e2 < 0.020459658932:
        z += -62.341148376464844 * Q.e2 + 1.2754786332116763
    if Q.mass_over_sum_pt_sq < 0.008174660116:
        z += 276.44122314453125 * Q.mass_over_sum_pt_sq - 2.2598130412578556
    if 6.377722943814 <= Q.log_sum_pt < 6.638338705138:
        z += 0.5979770421981812 * Q.log_sum_pt - 3.8137319019013725
    if Q.log_sum_pt >= 6.638338705138:
        z += -2.098807215690613 * Q.log_sum_pt + 14.088435416648666
    if Q.sum_pt_top5 >= 687.4375:
        z += 0.007173945661634207 * Q.sum_pt_top5 - 4.931639270769665
    if Q.mass_over_sum_pt < 0.13092863437:
        z += 20.652650833129883 * Q.mass_over_sum_pt - 2.704023369702138
    if Q.z_dr_0p05_0p1 >= 0.846033477783:
        z += 3.0832738876342773 * Q.z_dr_0p05_0p1 - 2.6085529301127384
    if Q.planar_flow < 0.148419710734 and Q.centroid_offset < 0.049903668404:
        z += -39.59550857543945 * (0.148419710734 - Q.planar_flow) * (0.049903668404 - Q.centroid_offset)
    if Q.width < 0.004372139331 and Q.D2 < 0.74595130682:
        z += -4251.14990234375 * (0.004372139331 - Q.width) * (0.74595130682 - Q.D2)
    if Q.lam1 < 0.006506575659 and Q.D2 < 0.87567204833:
        z += -611.1583862304688 * (0.006506575659 - Q.lam1) * (0.87567204833 - Q.D2)
    if Q.girth2 < 0.013238675334 and Q.D2 < 1.002470755577:
        z += -76.21794128417969 * (0.013238675334 - Q.girth2) * (1.002470755577 - Q.D2)
    if Q.sum_pt > 901.59375 and Q.pt_7 < 25.578125:
        z += 0.0012645742390304804 * (Q.sum_pt - 901.59375) * (25.578125 - Q.pt_7)
    if Q.planar_flow < 0.148419710734 and Q.sum_pt_top2 < 380.5875:
        z += -0.014213231392204762 * (0.148419710734 - Q.planar_flow) * (380.5875 - Q.sum_pt_top2)
    if Q.planar_flow < 0.148419710734 and Q.dr_2 < 0.027807975573:
        z += -249.2783203125 * (0.148419710734 - Q.planar_flow) * (0.027807975573 - Q.dr_2)
    if Q.sum_pt > 901.59375 and Q.dr_2 < 0.038438041256:
        z += -0.10085105895996094 * (Q.sum_pt - 901.59375) * (0.038438041256 - Q.dr_2)
    if Q.girth2 < 0.018827652745 and Q.mass_top2 > 28.78966323496:
        z += -2.7408342361450195 * (0.018827652745 - Q.girth2) * (Q.mass_top2 - 28.78966323496)
    if Q.mass < 56.920347213745 and Q.C2 > 0.023843882605:
        z += 0.972002387046814 * (56.920347213745 - Q.mass) * (Q.C2 - 0.023843882605)
    if Q.mass < 29.644699859619 and Q.D2 < 0.87567204833:
        z += -3.0753562450408936 * (29.644699859619 - Q.mass) * (0.87567204833 - Q.D2)
    if Q.planar_flow < 0.148419710734 and Q.max_dr < 0.111761856824:
        z += -129.98863220214844 * (0.148419710734 - Q.planar_flow) * (0.111761856824 - Q.max_dr)
    if Q.girth2 < 0.013238675334 and Q.centroid_offset > 0.014379521101:
        z += -707.9732666015625 * (0.013238675334 - Q.girth2) * (Q.centroid_offset - 0.014379521101)
    if Q.mass < 64.618731689453 and Q.centroid_offset > 0.012587644117:
        z += 1.9668558835983276 * (64.618731689453 - Q.mass) * (Q.centroid_offset - 0.012587644117)
    if Q.girth2 < 0.018827652745 and Q.phi_0 > 0.021438598633:
        z += 493.88427734375 * (0.018827652745 - Q.girth2) * (Q.phi_0 - 0.021438598633)
    if Q.girth2 < 0.018827652745 and Q.eccentricity > 0.959856212153:
        z += 1921.3856201171875 * (0.018827652745 - Q.girth2) * (Q.eccentricity - 0.959856212153)
    if Q.sum_pt > 901.59375 and Q.pt_7 > 29.0421875:
        z += -0.0006689488072879612 * (Q.sum_pt - 901.59375) * (Q.pt_7 - 29.0421875)
    if Q.girth2_top3 < 0.007929074034 and Q.pt_6 < 35.28125:
        z += -3.116753101348877 * (0.007929074034 - Q.girth2_top3) * (35.28125 - Q.pt_6)
    if Q.log_sum_pt > 6.638338705138 and Q.dr_4 < 0.072321663733:
        z += 44.67527389526367 * (Q.log_sum_pt - 6.638338705138) * (0.072321663733 - Q.dr_4)
    if Q.sum_pt_top5 > 687.4375 and Q.z_7 > 0.023207568189:
        z += -0.5931612849235535 * (Q.sum_pt_top5 - 687.4375) * (Q.z_7 - 0.023207568189)
    if Q.mass < 64.618731689453 and Q.pt_7 < 40.040625:
        z += -0.001528036780655384 * (64.618731689453 - Q.mass) * (40.040625 - Q.pt_7)
    if Q.mass < 29.644699859619 and Q.phi_1 > -0.058901977539:
        z += -0.9678459167480469 * (29.644699859619 - Q.mass) * (Q.phi_1 - -0.058901977539)
    return max(0.0, z)


def neuron_1(Q):
    z = 0.5303021669387817
    if Q.lam1 < 0.008375572068:
        z += 25.667444229125977 * Q.lam1 - 0.21497952894241532
    if 34.53125 <= Q.pt_7 < 53.4375:
        z += 0.11768782138824463 * Q.pt_7 - 4.063907582312822
    if Q.pt_7 >= 53.4375:
        z += 0.0020506083965301514 * Q.pt_7 + 2.11545598693192
    if Q.e2 < 0.032346998155:
        z += 5.315937042236328 * Q.e2 - 0.1890395707705456
    if 0.032346998155 <= Q.e2 < 0.035560912266:
        z += -21.378719329833984 * Q.e2 + 0.6744524296451718
    if Q.e2 >= 0.035560912266:
        z += -26.694656372070312 * Q.e2 + 0.8634920004157174
    if 6.377722943814 <= Q.log_sum_pt < 6.572937922293:
        z += 3.7751243114471436 * Q.log_sum_pt - 24.076696936866476
    if Q.log_sum_pt >= 6.572937922293:
        z += 9.92626690864563 * Q.log_sum_pt - 64.50777537942426
    if Q.z_7 < 0.043044721986:
        z += 120.49181365966797 * Q.z_7 - 6.146548297834593
    if 0.043044721986 <= Q.z_7 < 0.055577157257:
        z += 76.60216522216797 * Q.z_7 - 4.257330582779126
    if Q.girth < 0.076081777364:
        z += -21.648239135742188 * Q.girth + 1.647036510248169
    if Q.mass < 53.332374954224:
        z += 0.023158278316259384 * Q.mass - 1.2350859824570208
    if Q.width < 0.008678044951:
        z += 364.983154296875 * Q.width - 3.16734021934605
    if Q.planar_flow < 0.083662731125:
        z += -5.074865818023682 * Q.planar_flow + 0.42457713442876843
    if Q.tau21 < 0.197783735394:
        z += -4.3067731857299805 * Q.tau21 + 0.8518096881683929
    if Q.e2_sq < 0.008168570676:
        z += -449.9858093261719 * Q.e2_sq + 3.6757408866778953
    if Q.C2 < 0.042322802544:
        z += 13.094454765319824 * Q.C2 - 0.5541940234539707
    if Q.LHA < 0.346713497427:
        z += 5.594720840454102 * Q.LHA - 1.9397652297215662
    if Q.mass_over_sum_pt >= 0.054892207095:
        z += 19.932071685791016 * Q.mass_over_sum_pt - 1.0941154068088261
    if Q.max_dr < 0.046566883102:
        z += 16.840356826782227 * Q.max_dr - 0.7842029277487357
    if Q.lam1 < 0.008375572068 and Q.centroid_offset > 0.02076709205:
        z += 11182.8037109375 * (0.008375572068 - Q.lam1) * (Q.centroid_offset - 0.02076709205)
    if Q.lam1 < 0.008375572068 and Q.z_7 > 0.039784489456:
        z += -1585.32958984375 * (0.008375572068 - Q.lam1) * (Q.z_7 - 0.039784489456)
    if Q.pt_7 > 34.53125 and Q.mass < 91.19:
        z += -0.00120707752648741 * (Q.pt_7 - 34.53125) * (91.19 - Q.mass)
    if Q.log_sum_pt > 6.377722943814 and Q.z_dr_0_0p05 < 0.90085350275:
        z += 0.6334074139595032 * (Q.log_sum_pt - 6.377722943814) * (0.90085350275 - Q.z_dr_0_0p05)
    if Q.lam1 < 0.008375572068 and Q.lam2 < 0.000537286005:
        z += -352540.96875 * (0.008375572068 - Q.lam1) * (0.000537286005 - Q.lam2)
    if Q.z_7 < 0.055577157257 and Q.lam2 < 0.003408388935:
        z += 4353.94677734375 * (0.055577157257 - Q.z_7) * (0.003408388935 - Q.lam2)
    if Q.pt_7 > 34.53125 and Q.centroid_offset > 0.014379521101:
        z += 0.8548842668533325 * (Q.pt_7 - 34.53125) * (Q.centroid_offset - 0.014379521101)
    if Q.log_sum_pt > 6.377722943814 and Q.centroid_offset < 0.023554160423:
        z += -132.0417022705078 * (Q.log_sum_pt - 6.377722943814) * (0.023554160423 - Q.centroid_offset)
    if Q.log_sum_pt > 6.377722943814 and Q.max_dr < 0.197968879342:
        z += 13.30794906616211 * (Q.log_sum_pt - 6.377722943814) * (0.197968879342 - Q.max_dr)
    if Q.pt_7 > 34.53125 and Q.pt_6 < 52.90625:
        z += -0.0022377558052539825 * (Q.pt_7 - 34.53125) * (52.90625 - Q.pt_6)
    if Q.z_7 < 0.055577157257 and Q.dr01 > 0.055953954317:
        z += -60.828189849853516 * (0.055577157257 - Q.z_7) * (Q.dr01 - 0.055953954317)
    if Q.log_sum_pt > 6.572937922293 and Q.D2 < 1.232133567333:
        z += -1.4443389177322388 * (Q.log_sum_pt - 6.572937922293) * (1.232133567333 - Q.D2)
    if Q.width < 0.008678044951 and Q.planar_flow < 0.083662731125:
        z += -1220.14404296875 * (0.008678044951 - Q.width) * (0.083662731125 - Q.planar_flow)
    if Q.e2 < 0.035560912266 and Q.eccentricity > 0.978160776925:
        z += 8622.8544921875 * (0.035560912266 - Q.e2) * (Q.eccentricity - 0.978160776925)
    if Q.z_7 < 0.055577157257 and Q.C2 < 0.042322802544:
        z += 178.35562133789062 * (0.055577157257 - Q.z_7) * (0.042322802544 - Q.C2)
    if Q.pt_7 > 34.53125 and Q.max_dr < 0.080507021025:
        z += 0.9115755558013916 * (Q.pt_7 - 34.53125) * (0.080507021025 - Q.max_dr)
    if Q.pt_7 > 53.4375 and Q.mass_top3 < 32.617988451746:
        z += 0.0008843201794661582 * (Q.pt_7 - 53.4375) * (32.617988451746 - Q.mass_top3)
    if Q.log_sum_pt > 6.572937922293 and Q.max_dr < 0.290267042816:
        z += 5.659153938293457 * (Q.log_sum_pt - 6.572937922293) * (0.290267042816 - Q.max_dr)
    if Q.tau21 < 0.197783735394 and Q.pt_6 < 50.25:
        z += -0.23375940322875977 * (0.197783735394 - Q.tau21) * (50.25 - Q.pt_6)
    if Q.log_sum_pt > 6.572937922293 and Q.girth2_top3 < 0.006756161242:
        z += -640.5792236328125 * (Q.log_sum_pt - 6.572937922293) * (0.006756161242 - Q.girth2_top3)
    if Q.z_7 < 0.055577157257 and Q.girth2_top3 < 0.007929074034:
        z += 4326.12109375 * (0.055577157257 - Q.z_7) * (0.007929074034 - Q.girth2_top3)
    if Q.log_sum_pt > 6.572937922293 and Q.lam2 < 0.001130644719:
        z += 4997.09912109375 * (Q.log_sum_pt - 6.572937922293) * (0.001130644719 - Q.lam2)
    if Q.LHA < 0.346713497427 and Q.planar_flow < 0.083662731125:
        z += -120.35140228271484 * (0.346713497427 - Q.LHA) * (0.083662731125 - Q.planar_flow)
    if Q.lam1 < 0.008375572068 and Q.n_pt_above_50 > 6.0:
        z += -35.77622604370117 * (0.008375572068 - Q.lam1) * (Q.n_pt_above_50 - 6.0)
    if Q.e2_sq < 0.008168570676 and Q.planar_flow < 0.083662731125:
        z += -6113.60205078125 * (0.008168570676 - Q.e2_sq) * (0.083662731125 - Q.planar_flow)
    if Q.e2_sq < 0.008168570676 and Q.n_pt_above_50 > 6.0:
        z += 18.361364364624023 * (0.008168570676 - Q.e2_sq) * (Q.n_pt_above_50 - 6.0)
    if Q.log_sum_pt > 6.572937922293 and Q.planar_flow < 0.033200121667:
        z += 45.06104278564453 * (Q.log_sum_pt - 6.572937922293) * (0.033200121667 - Q.planar_flow)
    if Q.log_sum_pt > 6.572937922293 and Q.n_pt_above_50 > 7.0:
        z += -2.6452300548553467 * (Q.log_sum_pt - 6.572937922293) * (Q.n_pt_above_50 - 7.0)
    if Q.log_sum_pt > 6.377722943814 and Q.n_pt_above_50 > 7.0:
        z += 1.0687122344970703 * (Q.log_sum_pt - 6.377722943814) * (Q.n_pt_above_50 - 7.0)
    return max(0.0, z)


def neuron_2(Q):
    z = 2.9385902881622314
    if Q.mass < 8.379955863953:
        z += -0.02256505936384201 * Q.mass - 0.1634957670151702
    if 8.379955863953 <= Q.mass < 36.229410171509:
        z += 0.05506011098623276 * Q.mass - 0.8139912684806299
    if 36.229410171509 <= Q.mass < 69.611351776123:
        z += -0.0353725403547287 * Q.mass + 2.462330349848126
    if Q.log_sum_pt < 6.267538488641:
        z += -3.204934000968933 * Q.log_sum_pt + 20.68411874131165
    if 6.267538488641 <= Q.log_sum_pt < 6.464150123592:
        z += -3.8819499611854553 * Q.log_sum_pt + 24.927342329392946
    if 6.464150123592 <= Q.log_sum_pt < 6.572937922293:
        z += -0.861589252948761 * Q.log_sum_pt + 5.403277283952298
    if 6.572937922293 <= Q.log_sum_pt < 6.638338705138:
        z += 0.13543760776519775 * Q.log_sum_pt - 1.1501183783792222
    if 6.638338705138 <= Q.log_sum_pt < 6.804164030582:
        z += -0.6770159602165222 * Q.log_sum_pt + 4.243223588081297
    if 6.804164030582 <= Q.log_sum_pt < 6.842716632804:
        z += -5.307611644268036 * Q.log_sum_pt + 35.750556181672856
    if 6.842716632804 <= Q.log_sum_pt < 6.896095378249:
        z += 8.34287816286087 * Q.log_sum_pt - 57.655877467489574
    if Q.log_sum_pt >= 6.896095378249:
        z += 1.459776222705841 * Q.log_sum_pt - 10.189349989969745
    if Q.z_7 < 0.016858545121:
        z += 193.86117935180664 * Q.z_7 - 3.67266502352775
    if 0.016858545121 <= Q.z_7 < 0.028070914944:
        z += 36.07155227661133 * Q.z_7 - 1.012561475854806
    if Q.z_7 >= 0.046240320761:
        z += -2.921682596206665 * Q.z_7 + 0.13509954041042743
    if Q.lam1 < 0.003377388461:
        z += -324.9339065551758 * Q.lam1 + 1.4170482907871125
    if 0.003377388461 <= Q.lam1 < 0.005954149834:
        z += -124.0395278930664 * Q.lam1 + 0.7385499344139397
    if Q.width < 9.1213921e-05:
        z += 3840.38720703125 * Q.width + 0.155417340851412
    if 9.1213921e-05 <= Q.width < 0.000172198326:
        z += -6244.58642578125 * Q.width + 1.0753073290818544
    if Q.pt_7 < 15.55390625:
        z += -0.14833103865385056 * Q.pt_7 + 2.059887389605865
    if 15.55390625 <= Q.pt_7 < 30.484375:
        z += -0.01184409111738205 * Q.pt_7 - 0.06301779672503471
    if 30.484375 <= Q.pt_7 < 43.5:
        z += 0.06200004369020462 * Q.pt_7 - 2.3141100937500596
    if 43.5 <= Q.pt_7 < 53.4375:
        z += 0.1320313811302185 * Q.pt_7 - 5.360473272390664
    if Q.pt_7 >= 53.4375:
        z += 0.07384413480758667 * Q.pt_7 - 2.251092297025025
    if Q.LHA >= 0.111565049159:
        z += -10.28906536102295 * Q.LHA + 1.1479000828026893
    if Q.planar_flow < 0.694781820497:
        z += 0.30738407373428345 * Q.planar_flow - 0.21356486634088953
    if Q.pt_6 < 56.53125:
        z += 0.01856943778693676 * Q.pt_6 - 1.0497535298927687
    if Q.sum_pt_top5 < 687.4375:
        z += 0.005039863754063845 * Q.sum_pt_top5 - 3.4645913394342642
    if Q.sum_pt_top5 >= 839.9546875:
        z += -0.0020333407446742058 * Q.sum_pt_top5 + 1.7079140897738398
    if Q.sum_pt < 788.4484375:
        z += -0.009516896679997444 * Q.sum_pt + 7.503582317192922
    if Q.girth2 < 4.8108519e-05:
        z += 15365.701171875 * Q.girth2 - 0.7392211267754707
    if Q.max_dr < 0.15984864831:
        z += 1.9605776071548462 * Q.max_dr - 0.3133956804105564
    if Q.mass < 69.611351776123 and Q.z_7 < 0.068101508468:
        z += -0.39586150646209717 * (69.611351776123 - Q.mass) * (0.068101508468 - Q.z_7)
    if Q.mass < 69.611351776123 and Q.centroid_offset > 0.010960638421:
        z += -0.18094247579574585 * (69.611351776123 - Q.mass) * (Q.centroid_offset - 0.010960638421)
    if Q.pt_7 > 30.484375 and Q.dr_0 > 0.031288303462:
        z += -0.1907491832971573 * (Q.pt_7 - 30.484375) * (Q.dr_0 - 0.031288303462)
    if Q.pt_7 > 30.484375 and Q.C2 < 0.051192347892:
        z += -0.8857061862945557 * (Q.pt_7 - 30.484375) * (0.051192347892 - Q.C2)
    if Q.z_7 < 0.028070914944 and Q.width < 0.00752008842:
        z += -4065.636474609375 * (0.028070914944 - Q.z_7) * (0.00752008842 - Q.width)
    if Q.lam1 < 0.005954149834 and Q.max_dr > 0.080507021025:
        z += -2543.572509765625 * (0.005954149834 - Q.lam1) * (Q.max_dr - 0.080507021025)
    if Q.pt_6 < 56.53125 and Q.m012 > 32.617988451746:
        z += 0.00030547555070370436 * (56.53125 - Q.pt_6) * (Q.m012 - 32.617988451746)
    if Q.mass < 36.229410171509 and Q.lam2 < 0.001130644719:
        z += 34.8690185546875 * (36.229410171509 - Q.mass) * (0.001130644719 - Q.lam2)
    if Q.pt_7 > 30.484375 and Q.max_dr > 0.093110798299:
        z += -0.37651774287223816 * (Q.pt_7 - 30.484375) * (Q.max_dr - 0.093110798299)
    if Q.log_sum_pt > 6.842716632804 and Q.n_pt_above_50 > 5.0:
        z += -1.890014886856079 * (Q.log_sum_pt - 6.842716632804) * (Q.n_pt_above_50 - 5.0)
    if Q.mass < 36.229410171509 and Q.pt_4 < 55.65625:
        z += 0.00022176551283337176 * (36.229410171509 - Q.mass) * (55.65625 - Q.pt_4)
    if Q.lam1 < 0.003377388461 and Q.centroid_offset < 0.006789738266:
        z += -24480.216796875 * (0.003377388461 - Q.lam1) * (0.006789738266 - Q.centroid_offset)
    if Q.mass < 8.379955863953 and Q.centroid_offset < 0.010960638421:
        z += 8.208698272705078 * (8.379955863953 - Q.mass) * (0.010960638421 - Q.centroid_offset)
    if Q.pt_7 > 30.484375 and Q.centroid_offset > 0.012587644117:
        z += -0.5703819990158081 * (Q.pt_7 - 30.484375) * (Q.centroid_offset - 0.012587644117)
    if Q.width < 9.1213921e-05 and Q.pt_2 < 154.25:
        z += 72.92163848876953 * (9.1213921e-05 - Q.width) * (154.25 - Q.pt_2)
    if Q.mass < 36.229410171509 and Q.mean_phi < -0.003096654534:
        z += 0.35139331221580505 * (36.229410171509 - Q.mass) * (-0.003096654534 - Q.mean_phi)
    if Q.pt_7 > 30.484375 and Q.z_dr_0p05_0p1 < 0.750909513235:
        z += -0.01526553463190794 * (Q.pt_7 - 30.484375) * (0.750909513235 - Q.z_dr_0p05_0p1)
    if Q.mass < 69.611351776123 and Q.max_pair_mass > 13.047927274731:
        z += -0.0006908411742188036 * (69.611351776123 - Q.mass) * (Q.max_pair_mass - 13.047927274731)
    if Q.width < 0.000172198326 and Q.dr_7 < 0.222994708167:
        z += -37591.54296875 * (0.000172198326 - Q.width) * (0.222994708167 - Q.dr_7)
    return max(0.0, z)


def neuron_3(Q):
    z = 2.8887221813201904
    if 0.06813910019 <= Q.mass_over_sum_pt < 0.090413827016:
        z += -56.85742950439453 * Q.mass_over_sum_pt + 3.874214085545801
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += -28.46988296508789 * Q.mass_over_sum_pt + 1.307587363332281
    if Q.mass_over_sum_pt >= 0.107985668755:
        z += -49.53963088989258 * Q.mass_over_sum_pt + 3.5828181834915886
    if Q.centroid_offset >= 0.014379521101:
        z += 60.75294876098633 * Q.centroid_offset - 0.8735983086565746
    if Q.width < 0.006679471358:
        z += -113.47566986083984 * Q.width + 0.7579574866653436
    if Q.width >= 0.018827653081:
        z += 295.0851745605469 * Q.width - 5.555761295972304
    if 36.229410171509 <= Q.mass < 69.611351776123:
        z += -0.02491580694913864 * Q.mass + 0.9026849897144782
    if Q.mass >= 69.611351776123:
        z += -0.08497373387217522 * Q.mass + 5.083398467698665
    if Q.e2 < 0.028531698044:
        z += -167.53952980041504 * Q.e2 + 7.365987421428229
    if 0.028531698044 <= Q.e2 < 0.038466955721:
        z += -224.5043430328369 * Q.e2 + 8.991290271708545
    if 0.038466955721 <= Q.e2 < 0.04447356835:
        z += -210.33773803710938 * Q.e2 + 8.446344104620996
    if Q.e2 >= 0.04447356835:
        z += -56.964813232421875 * Q.e2 + 1.6253028502803164
    if 0.312727471086 <= Q.LHA < 0.325582223496:
        z += -2.253824234008789 * Q.LHA + 0.7048327529739097
    if 0.325582223496 <= Q.LHA < 0.423592510895:
        z += -9.283518314361572 * Q.LHA + 2.9935761821518376
    if Q.LHA >= 0.423592510895:
        z += -47.62284326553345 * Q.LHA + 19.233827104238056
    if Q.girth2 < 0.004372139461:
        z += 995.9954833984375 * Q.girth2 - 14.088549224810421
    if 0.004372139461 <= Q.girth2 < 0.008678044751:
        z += 1730.0339965820312 * Q.girth2 - 17.29786797419418
    if 0.008678044751 <= Q.girth2 < 0.013238675334:
        z += 500.92974853515625 * Q.girth2 - 6.631646305999196
    if Q.lam1 < 0.007330079875:
        z += -236.58267211914062 * Q.lam1 + 1.7341698836742363
    if 0.012003726523 <= Q.lam1 < 0.016433749775:
        z += -232.5590362548828 * Q.lam1 + 2.7915750716560557
    if Q.lam1 >= 0.016433749775:
        z += -110.91461944580078 * Q.lam1 + 0.7925011642897977
    if Q.mass_top5 >= 53.607658247923:
        z += -0.03674808144569397 * Q.mass_top5 + 1.9699785914076025
    if Q.mean_eta < -0.012844925793:
        z += -28.232093811035156 * Q.mean_eta - 0.3626391499837612
    if Q.z_dr_0p1_0p2 < 0.468445876241:
        z += -1.5478767156600952 * Q.z_dr_0p1_0p2 + 0.7250964643804345
    if Q.max_dr >= 0.145231109113:
        z += 8.602799415588379 * Q.max_dr - 1.2493941006025684
    if Q.C2 >= 0.094821243733:
        z += -50.17066955566406 * Q.C2 + 4.757245286185425
    if Q.mass_over_sum_pt > 0.06813910019 and Q.tau32 < 0.518696343899:
        z += -13.926008224487305 * (Q.mass_over_sum_pt - 0.06813910019) * (0.518696343899 - Q.tau32)
    if Q.e2 > 0.028531698044 and Q.n_pt_above_50 > 3.0:
        z += -0.9309388995170593 * (Q.e2 - 0.028531698044) * (Q.n_pt_above_50 - 3.0)
    if Q.mass > 36.229410171509 and Q.lam2 < 0.001130644719:
        z += 37.69093322753906 * (Q.mass - 36.229410171509) * (0.001130644719 - Q.lam2)
    if Q.mass_over_sum_pt > 0.06813910019 and Q.n_dr_0_0p05 < 5.0:
        z += 14.551855087280273 * (Q.mass_over_sum_pt - 0.06813910019) * (5.0 - Q.n_dr_0_0p05)
    if Q.LHA > 0.312727471086 and Q.mass_top3 < 50.352200171245:
        z += -0.2616192698478699 * (Q.LHA - 0.312727471086) * (50.352200171245 - Q.mass_top3)
    if Q.e2 < 0.04447356835 and Q.z_dr_0p05_0p1 < 0.964120104909:
        z += -20.873794555664062 * (0.04447356835 - Q.e2) * (0.964120104909 - Q.z_dr_0p05_0p1)
    if Q.mass > 36.229410171509 and Q.eccentricity > 0.620723099573:
        z += 0.07604367285966873 * (Q.mass - 36.229410171509) * (Q.eccentricity - 0.620723099573)
    if Q.e2 < 0.04447356835 and Q.z_dr_0p1_0p2 > 0.0:
        z += -201.17576599121094 * (0.04447356835 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.0)
    if Q.centroid_offset > 0.014379521101 and Q.phi_7 > -0.041534423828:
        z += 54.247920989990234 * (Q.centroid_offset - 0.014379521101) * (Q.phi_7 - -0.041534423828)
    if Q.centroid_offset > 0.014379521101 and Q.eta_0 < -0.039672851562:
        z += -349.6746520996094 * (Q.centroid_offset - 0.014379521101) * (-0.039672851562 - Q.eta_0)
    if Q.width > 0.018827653081 and Q.pt_dispersion < 0.492494773865:
        z += 367.9317626953125 * (Q.width - 0.018827653081) * (0.492494773865 - Q.pt_dispersion)
    if Q.width > 0.018827653081 and Q.z_3 > 0.090493038582:
        z += -2111.0224609375 * (Q.width - 0.018827653081) * (Q.z_3 - 0.090493038582)
    if Q.mass > 36.229410171509 and Q.dr_6 < 0.046481671275:
        z += -1.5859696865081787 * (Q.mass - 36.229410171509) * (0.046481671275 - Q.dr_6)
    if Q.mass_over_sum_pt > 0.06813910019 and Q.dr_7 < 0.042151962757:
        z += -7069.494140625 * (Q.mass_over_sum_pt - 0.06813910019) * (0.042151962757 - Q.dr_7)
    if Q.LHA > 0.312727471086 and Q.max_dr < 0.145231109113:
        z += -2240.700927734375 * (Q.LHA - 0.312727471086) * (0.145231109113 - Q.max_dr)
    if Q.LHA > 0.312727471086 and Q.planar_flow > 0.00804883781:
        z += 10.170502662658691 * (Q.LHA - 0.312727471086) * (Q.planar_flow - 0.00804883781)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.max_dr < 0.145231109113:
        z += 11460.421875 * (Q.mass_over_sum_pt - 0.090413827016) * (0.145231109113 - Q.max_dr)
    if Q.mass > 69.611351776123 and Q.max_dr < 0.177304983139:
        z += 2.0978128910064697 * (Q.mass - 69.611351776123) * (0.177304983139 - Q.max_dr)
    if Q.max_dr > 0.145231109113 and Q.dr_3 < 0.04586879935:
        z += -509.6424255371094 * (Q.max_dr - 0.145231109113) * (0.04586879935 - Q.dr_3)
    if Q.lam1 > 0.016433749775 and Q.eccentricity > 0.959856212153:
        z += -2518.923583984375 * (Q.lam1 - 0.016433749775) * (Q.eccentricity - 0.959856212153)
    if Q.max_dr > 0.145231109113 and Q.eta_1 < -0.059631347656:
        z += -56.60622787475586 * (Q.max_dr - 0.145231109113) * (-0.059631347656 - Q.eta_1)
    if Q.mean_eta < -0.012844925793 and Q.pt_4 < 68.125:
        z += -1.383903980255127 * (-0.012844925793 - Q.mean_eta) * (68.125 - Q.pt_4)
    if Q.mean_eta < -0.012844925793 and Q.z_4 < 0.120257140434:
        z += 343.43353271484375 * (-0.012844925793 - Q.mean_eta) * (0.120257140434 - Q.z_4)
    if Q.mass_over_sum_pt > 0.107985668755 and Q.dr_7 < 0.04881348081:
        z += 12805.2041015625 * (Q.mass_over_sum_pt - 0.107985668755) * (0.04881348081 - Q.dr_7)
    if Q.LHA > 0.312727471086 and Q.eccentricity > 0.959856212153:
        z += 834.4568481445312 * (Q.LHA - 0.312727471086) * (Q.eccentricity - 0.959856212153)
    if Q.e2 > 0.028531698044 and Q.eccentricity > 0.959856212153:
        z += -1465.912353515625 * (Q.e2 - 0.028531698044) * (Q.eccentricity - 0.959856212153)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.pt_6 < 56.53125:
        z += 0.4581131935119629 * (Q.mass_over_sum_pt - 0.090413827016) * (56.53125 - Q.pt_6)
    if Q.lam1 > 0.012003726523 and Q.pt_6 < 38.25:
        z += 0.28903377056121826 * (Q.lam1 - 0.012003726523) * (38.25 - Q.pt_6)
    if Q.mass_over_sum_pt > 0.090413827016 and Q.pt_7 < 48.71875:
        z += 0.250757098197937 * (Q.mass_over_sum_pt - 0.090413827016) * (48.71875 - Q.pt_7)
    if Q.mass > 36.229410171509 and Q.pt_7 < 20.125:
        z += -0.005371446721255779 * (Q.mass - 36.229410171509) * (20.125 - Q.pt_7)
    return max(0.0, z)


def neuron_4(Q):
    z = -7.102276802062988
    if Q.tau21 < 0.23799610585:
        z += -27.141910552978516 * Q.tau21 + 6.459669016937907
    if Q.lam2 < 0.000306123359:
        z += 2674.7459716796875 * Q.lam2 - 1.26894589352001
    if 0.000306123359 <= Q.lam2 < 0.001130644719:
        z += 545.9454345703125 * Q.lam2 - 0.6172703224590839
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += -227.47259521484375 * Q.mass_over_sum_pt + 20.56666787463547
    if Q.mass_over_sum_pt >= 0.107985668755:
        z += 176.238525390625 * Q.mass_over_sum_pt - 23.028347467776534
    if 0.000319370692 <= Q.width < 0.001653836415:
        z += -502.4072265625 * Q.width + 0.16045414361306642
    if Q.width >= 0.001653836415:
        z += 481.15826416015625 * Q.width - 1.466202281481407
    if 0.007078157854 <= Q.e2 < 0.041109715588:
        z += 73.23894500732422 * Q.e2 - 0.518396813822266
    if 0.041109715588 <= Q.e2 < 0.063441075385:
        z += 53.35640525817871 * Q.e2 + 0.29896874043221067
    if Q.e2 >= 0.063441075385:
        z += 57.31969928741455 * Q.e2 + 0.04753310515053932
    if Q.sum_pt < 763.825:
        z += -0.004924036096781492 * Q.sum_pt + 3.7611018716241236
    if Q.girth2_top2 < 0.009530300104:
        z += 30.053926467895508 * Q.girth2_top2 - 0.28642293854259293
    if Q.mass < 56.920347213745:
        z += 0.0056081488728523254 * Q.mass - 1.2260265870363516
    if 56.920347213745 <= Q.mass < 69.611351776123:
        z += 0.07145287841558456 * Q.mass - 4.973931454803804
    if 0.014943876117 <= Q.C2 < 0.067292226106:
        z += -32.74215316772461 * Q.C2 + 0.4892946807423157
    if Q.C2 >= 0.067292226106:
        z += -174.46439743041992 * Q.C2 + 10.02609998591737
    if Q.girth2_top3 < 0.005011406868:
        z += -352.8853454589844 * Q.girth2_top3 + 1.7684520438497071
    if Q.girth < 0.047915700823:
        z += -24.445941925048828 * Q.girth + 1.1713444396170722
    if Q.girth >= 0.101940929517:
        z += -44.1640739440918 * Q.girth + 4.502126749118238
    if Q.e2_sq < 0.017162483186:
        z += -525.6851806640625 * Q.e2_sq + 9.701949014208852
    if 0.017162483186 <= Q.e2_sq < 0.023780909279:
        z += -102.72622680664062 * Q.e2_sq + 2.4429230802626987
    if Q.girth2 >= 0.013238675334:
        z += -340.14453125 * Q.girth2 + 4.503063015854367
    if Q.sum_pt_top5 < 430.75:
        z += 0.0027187378145754337 * Q.sum_pt_top5 - 1.171096313628368
    if Q.centroid_offset < 0.014379521101:
        z += -27.079221725463867 * Q.centroid_offset + 0.38938624019996526
    if 0.102758520097 <= Q.max_dr < 0.197968879342:
        z += 18.600749969482422 * Q.max_dr - 1.9113855395583315
    if Q.max_dr >= 0.197968879342:
        z += 7.895522117614746 * Q.max_dr + 0.20791642137667843
    if Q.mass_over_sum_pt_sq < 0.001101860861:
        z += -87.38382720947266 * Q.mass_over_sum_pt_sq + 0.09628481908650478
    if Q.tau21 < 0.23799610585 and Q.mass < 62.55:
        z += -0.4467792510986328 * (0.23799610585 - Q.tau21) * (62.55 - Q.mass)
    if Q.tau21 < 0.23799610585 and Q.e2_sq > 0.011657374702:
        z += -1726.7933349609375 * (0.23799610585 - Q.tau21) * (Q.e2_sq - 0.011657374702)
    if Q.tau21 < 0.23799610585 and Q.lam2 < 0.001130644719:
        z += -10138.16015625 * (0.23799610585 - Q.tau21) * (0.001130644719 - Q.lam2)
    if Q.tau21 < 0.23799610585 and Q.pt_6 < 62.25:
        z += -0.018405161798000336 * (0.23799610585 - Q.tau21) * (62.25 - Q.pt_6)
    if Q.sum_pt < 763.825 and Q.z_dr_0_0p05 > 0.151540452242:
        z += 0.0014922827249392867 * (763.825 - Q.sum_pt) * (Q.z_dr_0_0p05 - 0.151540452242)
    if Q.width > 0.000319370692 and Q.C2 < 0.067292226106:
        z += 2211.646728515625 * (Q.width - 0.000319370692) * (0.067292226106 - Q.C2)
    if Q.tau21 < 0.23799610585 and Q.mean_eta > 0.02644207105:
        z += -261.7283020019531 * (0.23799610585 - Q.tau21) * (Q.mean_eta - 0.02644207105)
    if Q.girth2_top2 < 0.009530300104 and Q.centroid_offset > 0.016278845848:
        z += 8791.552734375 * (0.009530300104 - Q.girth2_top2) * (Q.centroid_offset - 0.016278845848)
    if Q.tau21 < 0.23799610585 and Q.planar_flow > 0.045057236346:
        z += 29.358766555786133 * (0.23799610585 - Q.tau21) * (Q.planar_flow - 0.045057236346)
    if Q.tau21 < 0.23799610585 and Q.mean_phi > 0.004406178184:
        z += -157.07643127441406 * (0.23799610585 - Q.tau21) * (Q.mean_phi - 0.004406178184)
    if Q.mass < 69.611351776123 and Q.mean_eta2 > 0.004247450386:
        z += 5.289422035217285 * (69.611351776123 - Q.mass) * (Q.mean_eta2 - 0.004247450386)
    if Q.sum_pt < 763.825 and Q.n_dr_0p2_0p4 < 1.0:
        z += -0.005168015602976084 * (763.825 - Q.sum_pt) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.width > 0.000319370692 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += 1771.99462890625 * (Q.width - 0.000319370692) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.tau21 < 0.23799610585 and Q.dr_7 < 0.175465903809:
        z += -21.17687225341797 * (0.23799610585 - Q.tau21) * (0.175465903809 - Q.dr_7)
    if Q.tau21 < 0.23799610585 and Q.pt_7 < 23.21640625:
        z += -1.1343271732330322 * (0.23799610585 - Q.tau21) * (23.21640625 - Q.pt_7)
    if Q.mass_over_sum_pt > 0.107985668755 and Q.n_dr_0p1_0p2 > 6.0:
        z += -8.643866539001465 * (Q.mass_over_sum_pt - 0.107985668755) * (Q.n_dr_0p1_0p2 - 6.0)
    if Q.lam2 < 0.001130644719 and Q.n_dr_0p1_0p2 > 2.0:
        z += 56.68592834472656 * (0.001130644719 - Q.lam2) * (Q.n_dr_0p1_0p2 - 2.0)
    if Q.tau21 < 0.23799610585 and Q.phi_6 > 0.119201660156:
        z += -32.7449951171875 * (0.23799610585 - Q.tau21) * (Q.phi_6 - 0.119201660156)
    if Q.girth2_top3 < 0.005011406868 and Q.mean_eta < -0.009391680919:
        z += 9392.5087890625 * (0.005011406868 - Q.girth2_top3) * (-0.009391680919 - Q.mean_eta)
    if Q.mass < 69.611351776123 and Q.dr_3 < 0.104247858869:
        z += 0.1123090460896492 * (69.611351776123 - Q.mass) * (0.104247858869 - Q.dr_3)
    if Q.tau21 < 0.23799610585 and Q.dr_6 < 0.169508891404:
        z += -19.760766983032227 * (0.23799610585 - Q.tau21) * (0.169508891404 - Q.dr_6)
    if Q.tau21 < 0.23799610585 and Q.pt_7 > 33.21875:
        z += 0.5149714350700378 * (0.23799610585 - Q.tau21) * (Q.pt_7 - 33.21875)
    if Q.mass_over_sum_pt > 0.107985668755 and Q.D2 < 3.885568320751:
        z += -55.40217590332031 * (Q.mass_over_sum_pt - 0.107985668755) * (3.885568320751 - Q.D2)
    if Q.max_dr > 0.102758520097 and Q.pt_7 > 37.15625:
        z += -1.5663268566131592 * (Q.max_dr - 0.102758520097) * (Q.pt_7 - 37.15625)
    if Q.C2 > 0.014943876117 and Q.pt_7 > 38.53125:
        z += 5.417937755584717 * (Q.C2 - 0.014943876117) * (Q.pt_7 - 38.53125)
    if Q.lam2 < 0.000306123359 and Q.mass_top3 < 50.352200171245:
        z += -109.26939392089844 * (0.000306123359 - Q.lam2) * (50.352200171245 - Q.mass_top3)
    if Q.centroid_offset < 0.014379521101 and Q.z_dr_0p05_0p1 < 0.674770402908:
        z += -271.574462890625 * (0.014379521101 - Q.centroid_offset) * (0.674770402908 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.23799610585 and Q.pt_2 > 73.6875:
        z += 0.022424811497330666 * (0.23799610585 - Q.tau21) * (Q.pt_2 - 73.6875)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.4689365327358246
    if Q.LHA < 0.154689112391:
        z += -12.873782634735107 * Q.LHA + 3.1291921160370837
    if 0.154689112391 <= Q.LHA < 0.216055863061:
        z += -18.540302276611328 * Q.LHA + 4.005741009785083
    if Q.z_7 < 0.03243272066:
        z += -69.22884941101074 * Q.z_7 + 4.22365971008178
    if 0.03243272066 <= Q.z_7 < 0.049399692737:
        z += -41.3754825592041 * Q.z_7 + 3.3202992435366316
    if 0.049399692737 <= Q.z_7 < 0.071488645583:
        z += -57.78287124633789 * Q.z_7 + 4.130819203297571
    if 6.572937922293 <= Q.log_sum_pt < 6.896095378249:
        z += 2.0508697032928467 * Q.log_sum_pt - 13.480239246455344
    if Q.log_sum_pt >= 6.896095378249:
        z += 7.868241548538208 * Q.log_sum_pt - 53.59739034200774
    if Q.z_6 < 0.028865759995:
        z += -91.39573669433594 * Q.z_6 + 2.638207399984916
    if Q.dr_0 < 0.021588001063:
        z += 172.26665878295898 * Q.dr_0 - 3.936125384767021
    if 0.021588001063 <= Q.dr_0 < 0.026454043164:
        z += 44.64255905151367 * Q.dr_0 - 1.1809761841001616
    if Q.e2 < 0.035560912266:
        z += -64.74826049804688 * Q.e2 + 2.3025072109471583
    if Q.width < 0.002635417778:
        z += -345.9899597167969 * Q.width + 0.9118280908471503
    if Q.girth2 < 4.8108519e-05:
        z += -15912.79296875 * Q.girth2 + 0.7655409028801758
    if Q.mass_over_sum_pt < 0.084751611895:
        z += 10.340662956237793 * Q.mass_over_sum_pt - 0.8763878536040688
    if Q.sum_pt_top5 >= 752.1:
        z += 0.009726698510348797 * Q.sum_pt_top5 - 7.31544994963333
    if Q.sum_pt_top2 < 548.196875:
        z += 0.0011833219323307276 * Q.sum_pt_top2 - 0.6486933854226663
    if Q.pt_5 < 24.578125:
        z += -0.3327053189277649 * Q.pt_5 + 8.177272916771472
    if Q.sum_pt >= 868.509375:
        z += -0.02099069394171238 * Q.sum_pt + 18.230614476132903
    if Q.mean_phi2 < 0.01426135283:
        z += 37.785072326660156 * Q.mean_phi2 - 0.5388662481575696
    if Q.z_dr_0p1_0p2 < 0.045106684603:
        z += -9.088154792785645 * Q.z_dr_0p1_0p2 + 0.40993653186142487
    if Q.z_7 < 0.049399692737 and Q.mass_top5 < 62.55:
        z += 0.6343360543251038 * (0.049399692737 - Q.z_7) * (62.55 - Q.mass_top5)
    if Q.LHA < 0.216055863061 and Q.log_sum_pt < 6.804164030582:
        z += -119.958251953125 * (0.216055863061 - Q.LHA) * (6.804164030582 - Q.log_sum_pt)
    if Q.e2_sq < 0.00528466865 and Q.centroid_offset < 0.014379521101:
        z += -8767.326171875 * (0.00528466865 - Q.e2_sq) * (0.014379521101 - Q.centroid_offset)
    if Q.z_7 < 0.071488645583 and Q.centroid_offset < 0.031170772021:
        z += -873.2733764648438 * (0.071488645583 - Q.z_7) * (0.031170772021 - Q.centroid_offset)
    if Q.z_7 < 0.071488645583 and Q.sum_pt < 788.4484375:
        z += -0.33720436692237854 * (0.071488645583 - Q.z_7) * (788.4484375 - Q.sum_pt)
    if Q.z_7 < 0.071488645583 and Q.lam1 < 0.001503553356:
        z += -12025.833984375 * (0.071488645583 - Q.z_7) * (0.001503553356 - Q.lam1)
    if Q.log_sum_pt > 6.896095378249 and Q.dr_4 > 0.030068239644:
        z += -58.418033599853516 * (Q.log_sum_pt - 6.896095378249) * (Q.dr_4 - 0.030068239644)
    if Q.e2_sq < 0.00528466865 and Q.n_pt_above_50 > 6.0:
        z += -19.736957550048828 * (0.00528466865 - Q.e2_sq) * (Q.n_pt_above_50 - 6.0)
    if Q.LHA < 0.216055863061 and Q.n_dr_0p2_0p4 > 0.0:
        z += -10.810839653015137 * (0.216055863061 - Q.LHA) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.log_sum_pt > 6.896095378249 and Q.planar_flow < 0.045057236346:
        z += -464.3262634277344 * (Q.log_sum_pt - 6.896095378249) * (0.045057236346 - Q.planar_flow)
    if Q.z_7 < 0.071488645583 and Q.lam2 < 0.001130644719:
        z += 19999.908203125 * (0.071488645583 - Q.z_7) * (0.001130644719 - Q.lam2)
    if Q.width < 0.002635417778 and Q.centroid_offset < 0.023554160423:
        z += 31979.490234375 * (0.002635417778 - Q.width) * (0.023554160423 - Q.centroid_offset)
    if Q.log_sum_pt > 6.572937922293 and Q.centroid_offset > 0.018377780003:
        z += -109.20767974853516 * (Q.log_sum_pt - 6.572937922293) * (Q.centroid_offset - 0.018377780003)
    if Q.z_6 < 0.028865759995 and Q.phi_0 > 0.014526367188:
        z += -7389.86767578125 * (0.028865759995 - Q.z_6) * (Q.phi_0 - 0.014526367188)
    if Q.z_7 < 0.03243272066 and Q.pt_5 > 33.0265625:
        z += 1.5817875862121582 * (0.03243272066 - Q.z_7) * (Q.pt_5 - 33.0265625)
    if Q.sum_pt > 868.509375 and Q.centroid_offset < 0.012587644117:
        z += 0.5058428049087524 * (Q.sum_pt - 868.509375) * (0.012587644117 - Q.centroid_offset)
    if Q.log_sum_pt > 6.896095378249 and Q.centroid_offset < 0.018377780003:
        z += -821.4170532226562 * (Q.log_sum_pt - 6.896095378249) * (0.018377780003 - Q.centroid_offset)
    if Q.log_sum_pt > 6.572937922293 and Q.mean_phi2 < 0.000145482056:
        z += 13830.8564453125 * (Q.log_sum_pt - 6.572937922293) * (0.000145482056 - Q.mean_phi2)
    if Q.sum_pt_top2 < 548.196875 and Q.dr_0 < 0.021588001063:
        z += 0.39118096232414246 * (548.196875 - Q.sum_pt_top2) * (0.021588001063 - Q.dr_0)
    if Q.log_sum_pt > 6.572937922293 and Q.dr_0 < 0.021588001063:
        z += 545.17333984375 * (Q.log_sum_pt - 6.572937922293) * (0.021588001063 - Q.dr_0)
    if Q.log_sum_pt > 6.896095378249 and Q.dr_0 < 0.04118638065:
        z += -522.8942260742188 * (Q.log_sum_pt - 6.896095378249) * (0.04118638065 - Q.dr_0)
    if Q.z_7 < 0.071488645583 and Q.mean_phi2 < 0.004331280361:
        z += 2021.7418212890625 * (0.071488645583 - Q.z_7) * (0.004331280361 - Q.mean_phi2)
    if Q.log_sum_pt > 6.572937922293 and Q.lam1 < 0.012003726523:
        z += -1174.8094482421875 * (Q.log_sum_pt - 6.572937922293) * (0.012003726523 - Q.lam1)
    if Q.log_sum_pt > 6.572937922293 and Q.girth2_top2 < 0.006299534492:
        z += 689.0783081054688 * (Q.log_sum_pt - 6.572937922293) * (0.006299534492 - Q.girth2_top2)
    if Q.sum_pt_top2 < 548.196875 and Q.girth2_top3 < 0.003952581551:
        z += -1.5148147344589233 * (548.196875 - Q.sum_pt_top2) * (0.003952581551 - Q.girth2_top3)
    if Q.log_sum_pt > 6.572937922293 and Q.mean_eta2 < 9.0303693e-05:
        z += 20848.1875 * (Q.log_sum_pt - 6.572937922293) * (9.0303693e-05 - Q.mean_eta2)
    if Q.pt_5 < 24.578125 and Q.mean_eta2 > 1.7977892e-05:
        z += -21.51185417175293 * (24.578125 - Q.pt_5) * (Q.mean_eta2 - 1.7977892e-05)
    if Q.mass_over_sum_pt < 0.084751611895 and Q.max_pair_mass < 18.097979966098:
        z += -0.7357097268104553 * (0.084751611895 - Q.mass_over_sum_pt) * (18.097979966098 - Q.max_pair_mass)
    if Q.LHA < 0.216055863061 and Q.mass_top3 > 3.559569591142:
        z += -0.7625309228897095 * (0.216055863061 - Q.LHA) * (Q.mass_top3 - 3.559569591142)
    if Q.mean_phi2 < 0.01426135283 and Q.max_pair_mass < 40.046952646555:
        z += 1.0830191373825073 * (0.01426135283 - Q.mean_phi2) * (40.046952646555 - Q.max_pair_mass)
    if Q.z_7 < 0.03243272066 and Q.pt_5 < 29.875:
        z += -8.441235542297363 * (0.03243272066 - Q.z_7) * (29.875 - Q.pt_5)
    if Q.sum_pt > 868.509375 and Q.pt_5 < 33.0265625:
        z += 0.0003721854882314801 * (Q.sum_pt - 868.509375) * (33.0265625 - Q.pt_5)
    if Q.e2 < 0.035560912266 and Q.lam2 < 7.3007261e-05:
        z += 335180.375 * (0.035560912266 - Q.e2) * (7.3007261e-05 - Q.lam2)
    return max(0.0, z)


def neuron_6(Q):
    z = 8.167488098144531
    if 0.008092360237 <= Q.centroid_offset < 0.018377780003:
        z += 29.102014541625977 * Q.centroid_offset - 0.2355039852932498
    if Q.centroid_offset >= 0.018377780003:
        z += -106.77020835876465 * Q.centroid_offset + 2.261525835688708
    if Q.width < 0.013238675006:
        z += 655.52587890625 * Q.width - 8.678294068862353
    if Q.log_sum_pt < 6.327378592257:
        z += -4.60898756980896 * Q.log_sum_pt + 28.888191732427003
    if 6.327378592257 <= Q.log_sum_pt < 6.572937922293:
        z += -0.6790456771850586 * Q.log_sum_pt + 4.021961532224569
    if 6.572937922293 <= Q.log_sum_pt < 6.701242202626:
        z += 3.4399752616882324 * Q.log_sum_pt - 23.052107399614602
    if Q.e2 < 0.050284641981:
        z += -84.12153625488281 * Q.e2 + 4.2300213334684935
    if Q.lam2 < 0.000537286005:
        z += 1613.4793701171875 * Q.lam2 - 0.86689988492018
    if 0.000537286005 <= Q.lam2 < 0.003408388935:
        z += -203.4620819091797 * Q.lam2 + 0.10931732915796592
    if Q.lam2 >= 0.003408388935:
        z += -1154.784896850586 * Q.lam2 + 3.351795485217308
    z += 17.738208770751953 * Q.max_dr
    if Q.C2 >= 0.010539266048:
        z += -12.644725799560547 * Q.C2 + 0.13326612930557813
    if Q.mass_over_sum_pt >= 0.008374148675:
        z += -89.18032836914062 * Q.mass_over_sum_pt + 0.7468093286485039
    if 0.002270363079 <= Q.girth2_top5 < 0.011482925368:
        z += -74.46222686767578 * Q.girth2_top5 + 0.16905629066049294
    if Q.girth2_top5 >= 0.011482925368:
        z += 65.90906524658203 * Q.girth2_top5 - 1.4428167804972565
    if Q.girth2 < 0.003562611155:
        z += 1225.198226928711 * Q.girth2 - 10.92507254278071
    if 0.003562611155 <= Q.girth2 < 0.008678044751:
        z += 1282.426513671875 * Q.girth2 - 11.128954675513445
    if Q.lam1 < 0.007330079875:
        z += -483.4921569824219 * Q.lam1 + 3.855304310076714
    if 0.007330079875 <= Q.lam1 < 0.008375572068:
        z += -297.72406005859375 * Q.lam1 + 2.493609321398312
    if Q.girth >= 0.087236513197:
        z += 95.22792053222656 * Q.girth - 8.30735174623245
    if Q.D2 < 1.679198372364:
        z += -0.8138649463653564 * Q.D2 + 1.3666406932608208
    if Q.z_7 >= 0.06164517166:
        z += -8.214898109436035 * Q.z_7 + 0.5064088041255939
    if Q.centroid_offset > 0.008092360237 and Q.lam2 < 0.003408388935:
        z += 18047.484375 * (Q.centroid_offset - 0.008092360237) * (0.003408388935 - Q.lam2)
    if Q.log_sum_pt < 6.327378592257 and Q.z_7 < 0.071488645583:
        z += 163.3904571533203 * (6.327378592257 - Q.log_sum_pt) * (0.071488645583 - Q.z_7)
    if Q.centroid_offset > 0.008092360237 and Q.planar_flow > 0.00804883781:
        z += 55.498626708984375 * (Q.centroid_offset - 0.008092360237) * (Q.planar_flow - 0.00804883781)
    if Q.centroid_offset > 0.008092360237 and Q.C2 < 0.094821243733:
        z += 358.55145263671875 * (Q.centroid_offset - 0.008092360237) * (0.094821243733 - Q.C2)
    if Q.log_sum_pt < 6.327378592257 and Q.pt_6 > 27.578125:
        z += -0.0986853763461113 * (6.327378592257 - Q.log_sum_pt) * (Q.pt_6 - 27.578125)
    if Q.log_sum_pt < 6.701242202626 and Q.z_7 < 0.049399692737:
        z += 406.88568115234375 * (6.701242202626 - Q.log_sum_pt) * (0.049399692737 - Q.z_7)
    if Q.centroid_offset > 0.049903668404 and Q.n_dr_0p05_0p1 > 6.0:
        z += -44.30633544921875 * (Q.centroid_offset - 0.049903668404) * (Q.n_dr_0p05_0p1 - 6.0)
    if Q.centroid_offset > 0.018377780003 and Q.mean_phi2 < 0.008921136335:
        z += 5451.43212890625 * (Q.centroid_offset - 0.018377780003) * (0.008921136335 - Q.mean_phi2)
    if Q.LHA > 0.312727471086 and Q.eccentricity > 0.872657364787:
        z += 257.6095275878906 * (Q.LHA - 0.312727471086) * (Q.eccentricity - 0.872657364787)
    if Q.e2 < 0.050284641981 and Q.n_dr_0p05_0p1 < 4.0:
        z += -4.07758092880249 * (0.050284641981 - Q.e2) * (4.0 - Q.n_dr_0p05_0p1)
    if Q.log_sum_pt < 6.701242202626 and Q.mean_phi > 0.009050007537:
        z += -36.69248962402344 * (6.701242202626 - Q.log_sum_pt) * (Q.mean_phi - 0.009050007537)
    if Q.mass_over_sum_pt > 0.008374148675 and Q.tau32 < 0.518696343899:
        z += -20.19683265686035 * (Q.mass_over_sum_pt - 0.008374148675) * (0.518696343899 - Q.tau32)
    if Q.centroid_offset > 0.008092360237 and Q.mean_eta2 < 0.001101289818:
        z += -6312.3349609375 * (Q.centroid_offset - 0.008092360237) * (0.001101289818 - Q.mean_eta2)
    if Q.centroid_offset > 0.018377780003 and Q.pt_2 > 56.5:
        z += 0.4955327808856964 * (Q.centroid_offset - 0.018377780003) * (Q.pt_2 - 56.5)
    if Q.e2 < 0.050284641981 and Q.z_dr_0p1_0p2 > 0.15855820179:
        z += 623.8942260742188 * (0.050284641981 - Q.e2) * (Q.z_dr_0p1_0p2 - 0.15855820179)
    if Q.log_sum_pt < 6.701242202626 and Q.pt_6 < 36.8125:
        z += 0.19205421209335327 * (6.701242202626 - Q.log_sum_pt) * (36.8125 - Q.pt_6)
    if Q.centroid_offset > 0.018377780003 and Q.pt_5 < 24.578125:
        z += 22.22496795654297 * (Q.centroid_offset - 0.018377780003) * (24.578125 - Q.pt_5)
    if Q.log_sum_pt < 6.701242202626 and Q.z_4 < 0.037477688199:
        z += 2064.396484375 * (6.701242202626 - Q.log_sum_pt) * (0.037477688199 - Q.z_4)
    if Q.width < 0.013238675006 and Q.mean_eta > 0.02644207105:
        z += 7782.08056640625 * (0.013238675006 - Q.width) * (Q.mean_eta - 0.02644207105)
    if Q.lam2 < 0.000537286005 and Q.mean_eta > 0.02644207105:
        z += -197856.296875 * (0.000537286005 - Q.lam2) * (Q.mean_eta - 0.02644207105)
    if Q.lam1 < 0.007330079875 and Q.eta_7 > 0.008316040039:
        z += -690.1668090820312 * (0.007330079875 - Q.lam1) * (Q.eta_7 - 0.008316040039)
    if Q.log_sum_pt < 6.701242202626 and Q.mean_eta2 > 0.004247450386:
        z += 389.6140441894531 * (6.701242202626 - Q.log_sum_pt) * (Q.mean_eta2 - 0.004247450386)
    if Q.D2 < 1.679198372364 and Q.pt_4 < 90.625:
        z += -0.015521648339927197 * (1.679198372364 - Q.D2) * (90.625 - Q.pt_4)
    if Q.e2_sq < 0.008168570676 and Q.mass_top3 > 50.352200171245:
        z += 29.03057289123535 * (0.008168570676 - Q.e2_sq) * (Q.mass_top3 - 50.352200171245)
    if Q.girth2_top5 > 0.011482925368 and Q.mean_eta > 0.0127187056:
        z += -4757.43603515625 * (Q.girth2_top5 - 0.011482925368) * (Q.mean_eta - 0.0127187056)
    if Q.log_sum_pt < 6.701242202626 and Q.pt_7 < 45.75:
        z += 0.18615230917930603 * (6.701242202626 - Q.log_sum_pt) * (45.75 - Q.pt_7)
    if Q.C2 > 0.010539266048 and Q.pt_7 > 31.859375:
        z += 1.4182521104812622 * (Q.C2 - 0.010539266048) * (Q.pt_7 - 31.859375)
    if Q.mass < 49.668099212646 and Q.z_dr_0p05_0p1 < 0.750909513235:
        z += 0.06490819156169891 * (49.668099212646 - Q.mass) * (0.750909513235 - Q.z_dr_0p05_0p1)
    if Q.girth2_top5 > 0.002270363079 and Q.z_dr_0p05_0p1 > 0.291944718361:
        z += 102.22376251220703 * (Q.girth2_top5 - 0.002270363079) * (Q.z_dr_0p05_0p1 - 0.291944718361)
    return max(0.0, z)


def neuron_7(Q):
    z = 9.648682594299316
    if Q.planar_flow < 0.195013533663:
        z += 1.3604166507720947 * Q.planar_flow - 0.2652996583210496
    if 0.0016538364 <= Q.girth2 < 0.004372139461:
        z += -555.5516967773438 * Q.girth2 + 0.9187916182121338
    if 0.004372139461 <= Q.girth2 < 0.007520088344:
        z += -979.031494140625 * Q.girth2 + 2.7703043512004193
    if 0.007520088344 <= Q.girth2 < 0.008678044751:
        z += -2365.986083984375 * Q.girth2 + 13.200325395941706
    if 0.008678044751 <= Q.girth2 < 0.013238675334:
        z += -927.132568359375 * Q.girth2 + 0.7138901972142779
    if Q.girth2 >= 0.013238675334:
        z += -169.2742919921875 * Q.girth2 - 9.319149472795765
    if 0.072690732432 <= Q.mass_over_sum_pt < 0.084751611895:
        z += 85.8167953491211 * Q.mass_over_sum_pt - 6.238085708894664
    if 0.084751611895 <= Q.mass_over_sum_pt < 0.090413827016:
        z += 271.4699020385742 * Q.mass_over_sum_pt - 21.972485754140223
    if 0.090413827016 <= Q.mass_over_sum_pt < 0.107985668755:
        z += 1.5006637573242188 * Q.mass_over_sum_pt + 2.4364662554619994
    if Q.mass_over_sum_pt >= 0.107985668755:
        z += -81.46765899658203 * Q.mass_over_sum_pt + 11.395856073523246
    if Q.e2 < 0.024547699839:
        z += -154.4139404296875 * Q.e2 + 3.857535148640014
    if 0.024547699839 <= Q.e2 < 0.038466955721:
        z += -61.5745849609375 * Q.e2 + 1.5785425173469156
    if 0.038466955721 <= Q.e2 < 0.050284641981:
        z += 66.85270690917969 * Q.e2 - 3.361664432388826
    if Q.girth < 0.04081947431:
        z += 118.25749397277832 * Q.girth - 8.934166891559421
    if 0.04081947431 <= Q.girth < 0.087236513197:
        z += 88.47953796386719 * Q.girth - 7.718646381249362
    if Q.pt_7 < 48.71875:
        z += -0.014115940779447556 * Q.pt_7 + 0.6877109898487106
    if Q.width < 0.000561123155:
        z += 1988.7670288085938 * Q.width - 6.295467383608851
    if 0.000561123155 <= Q.width < 0.005590288644:
        z += 1029.8973388671875 * Q.width - 5.7574233979550575
    if Q.centroid_offset < 0.02076709205:
        z += 33.10216474533081 * Q.centroid_offset - 0.8069705459574066
    if 0.02076709205 <= Q.centroid_offset < 0.031170772021:
        z += 7.033984661102295 * Q.centroid_offset - 0.2656102505722562
    if 0.031170772021 <= Q.centroid_offset < 0.037760993714:
        z += 30.99238157272339 * Q.centroid_offset - 1.0124119786930277
    if Q.centroid_offset >= 0.037760993714:
        z += 23.958396911621094 * Q.centroid_offset - 0.7468017281207716
    if Q.e2_sq < 0.001101266364:
        z += -1536.0484619140625 * Q.e2_sq + 1.691598504579892
    if Q.mass < 29.644699859619:
        z += -0.02931332215666771 * Q.mass + 0.8689846372227338
    if Q.mass >= 80.4:
        z += -0.16583457589149475 * Q.mass + 13.33309990167618
    if Q.lam1 < 0.008375572068:
        z += 482.49420166015625 * Q.lam1 - 4.041164958396764
    if Q.z_dr_0p05_0p1 >= 0.674770402908:
        z += -0.5986657738685608 * Q.z_dr_0p05_0p1 + 0.4039619454405184
    if Q.LHA < 0.293190627853:
        z += -11.649070739746094 * Q.LHA + 3.4153983640901684
    if Q.girth2_top3 < 0.005884990035:
        z += -22.48561668395996 * Q.girth2_top3 + 0.13232763011593413
    if Q.max_dr < 0.197968879342:
        z += 1.6513280868530273 * Q.max_dr - 0.3269115707802627
    if Q.planar_flow < 0.195013533663 and Q.width > 0.00752008842:
        z += -1357.8944091796875 * (0.195013533663 - Q.planar_flow) * (Q.width - 0.00752008842)
    if Q.planar_flow < 0.195013533663 and Q.pt_6 < 35.28125:
        z += -0.26803553104400635 * (0.195013533663 - Q.planar_flow) * (35.28125 - Q.pt_6)
    if Q.planar_flow < 0.195013533663 and Q.mass_top3 > 23.663861485439:
        z += 0.0858401283621788 * (0.195013533663 - Q.planar_flow) * (Q.mass_top3 - 23.663861485439)
    if Q.planar_flow < 0.195013533663 and Q.sum_pt > 615.875:
        z += 0.027604175731539726 * (0.195013533663 - Q.planar_flow) * (Q.sum_pt - 615.875)
    if Q.girth2 > 0.004372139461 and Q.eccentricity > 0.945820652852:
        z += 5185.11083984375 * (Q.girth2 - 0.004372139461) * (Q.eccentricity - 0.945820652852)
    if Q.lam1 < 0.008375572068 and Q.D2 < 1.122624260187:
        z += -1226.2286376953125 * (0.008375572068 - Q.lam1) * (1.122624260187 - Q.D2)
    if Q.e2 < 0.050284641981 and Q.D2 < 1.122624260187:
        z += 190.80418395996094 * (0.050284641981 - Q.e2) * (1.122624260187 - Q.D2)
    if Q.centroid_offset < 0.02076709205 and Q.C2 > 0.023843882605:
        z += 1630.4295654296875 * (0.02076709205 - Q.centroid_offset) * (Q.C2 - 0.023843882605)
    if Q.mass > 80.4 and Q.eccentricity > 0.927072033478:
        z += -0.8854312300682068 * (Q.mass - 80.4) * (Q.eccentricity - 0.927072033478)
    if Q.pt_7 < 48.71875 and Q.planar_flow < 0.694781820497:
        z += -0.07013391703367233 * (48.71875 - Q.pt_7) * (0.694781820497 - Q.planar_flow)
    if Q.e2_sq < 0.001101266364 and Q.phi_0 > 0.040283203125:
        z += -116468.1796875 * (0.001101266364 - Q.e2_sq) * (Q.phi_0 - 0.040283203125)
    if Q.centroid_offset > 0.031170772021 and Q.pt_0 > 376.5:
        z += -3.622600793838501 * (Q.centroid_offset - 0.031170772021) * (Q.pt_0 - 376.5)
    if Q.e2_sq < 0.001101266364 and Q.phi_1 < -0.009460449219:
        z += -12668.7705078125 * (0.001101266364 - Q.e2_sq) * (-0.009460449219 - Q.phi_1)
    if Q.centroid_offset > 0.031170772021 and Q.pt_2 > 56.5:
        z += -0.8445028066635132 * (Q.centroid_offset - 0.031170772021) * (Q.pt_2 - 56.5)
    if Q.centroid_offset > 0.031170772021 and Q.n_pt_above_50 > 6.0:
        z += -34.78407669067383 * (Q.centroid_offset - 0.031170772021) * (Q.n_pt_above_50 - 6.0)
    if Q.girth2 > 0.013238675334 and Q.log_sum_pt > 6.19222188581:
        z += 234.33056640625 * (Q.girth2 - 0.013238675334) * (Q.log_sum_pt - 6.19222188581)
    if Q.z_dr_0p05_0p1 > 0.674770402908 and Q.pt_2 > 84.625:
        z += 0.019915636628866196 * (Q.z_dr_0p05_0p1 - 0.674770402908) * (Q.pt_2 - 84.625)
    if Q.e2_sq < 0.001101266364 and Q.eta_2 < -0.045445251465:
        z += -88692.0390625 * (0.001101266364 - Q.e2_sq) * (-0.045445251465 - Q.eta_2)
    if Q.width < 0.005590288644 and Q.mean_eta < -0.009391680919:
        z += 2613.806396484375 * (0.005590288644 - Q.width) * (-0.009391680919 - Q.mean_eta)
    if Q.e2 < 0.024547699839 and Q.z_dr_0p05_0p1 > 0.750909513235:
        z += -191.07730102539062 * (0.024547699839 - Q.e2) * (Q.z_dr_0p05_0p1 - 0.750909513235)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.3028097152709961
    if Q.girth2 < 0.000964142894:
        z += 377.9911689758301 * Q.girth2 - 0.6962901514397678
    if 0.000964142894 <= Q.girth2 < 0.006679471442:
        z += 58.06361770629883 * Q.girth2 - 0.3878342762884286
    if Q.e2 < 0.016554418951:
        z += -62.66361427307129 * Q.e2 + 0.9965187034657729
    if 0.016554418951 <= Q.e2 < 0.024547699839:
        z += 5.109418869018555 * Q.e2 - 0.12542448074839033
    if Q.width < 0.005019718802:
        z += -1284.406982421875 * Q.width + 6.447361879083169
    if Q.LHA < 0.196739721581:
        z += 15.387380599975586 * Q.LHA - 3.0273089751000777
    if Q.log_sum_pt >= 6.701242202626:
        z += -4.102076053619385 * Q.log_sum_pt + 27.489005168895737
    if Q.mass < 8.379955863953:
        z += 0.03617130406200886 * Q.mass - 1.747123648528644
    if 8.379955863953 <= Q.mass < 15.454033088684:
        z += 0.07020722143352032 * Q.mass - 2.0323431338910614
    if 15.454033088684 <= Q.mass < 21.784077072144:
        z += 0.11513605527579784 * Q.mass - 2.726674818725604
    if 21.784077072144 <= Q.mass < 29.644699859619:
        z += 0.027802137657999992 * Q.mass - 0.8241860263272205
    if Q.girth < 0.061086014472:
        z += 47.86677551269531 * Q.girth - 2.9239905416964813
    if Q.max_dr < 0.177304983139:
        z += -6.36794900894165 * Q.max_dr + 1.1290690916604111
    if Q.pt_7 >= 34.53125:
        z += -0.052581727504730225 * Q.pt_7 + 1.8157127778977156
    if Q.girth2 < 0.006679471442 and Q.tau21 < 0.501026660204:
        z += -551.404052734375 * (0.006679471442 - Q.girth2) * (0.501026660204 - Q.tau21)
    if Q.width < 0.005019718802 and Q.pt_7 < 48.71875:
        z += -7.399890899658203 * (0.005019718802 - Q.width) * (48.71875 - Q.pt_7)
    if Q.girth2 < 0.006679471442 and Q.centroid_offset < 0.023554160423:
        z += 25555.73828125 * (0.006679471442 - Q.girth2) * (0.023554160423 - Q.centroid_offset)
    if Q.girth2 < 0.006679471442 and Q.planar_flow < 0.40079469091:
        z += 380.7790222167969 * (0.006679471442 - Q.girth2) * (0.40079469091 - Q.planar_flow)
    if Q.girth2 < 0.006679471442 and Q.n_dr_0p05_0p1 > 0.0:
        z += -42.37726974487305 * (0.006679471442 - Q.girth2) * (Q.n_dr_0p05_0p1 - 0.0)
    if Q.LHA < 0.196739721581 and Q.mean_phi > -0.000855675264:
        z += 397.98553466796875 * (0.196739721581 - Q.LHA) * (Q.mean_phi - -0.000855675264)
    if Q.width < 0.005019718802 and Q.z_dr_0p2_0p4 < 0.1009733513:
        z += 6564.34228515625 * (0.005019718802 - Q.width) * (0.1009733513 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.061086014472 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += -63.09362030029297 * (0.061086014472 - Q.girth) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.girth2 < 0.006679471442 and Q.phi_0 > -0.040130615234:
        z += 1226.2559814453125 * (0.006679471442 - Q.girth2) * (Q.phi_0 - -0.040130615234)
    if Q.log_sum_pt > 6.701242202626 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += -23.77341079711914 * (Q.log_sum_pt - 6.701242202626) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.max_dr < 0.177304983139 and Q.pt1_over_pt0 < 0.465420272571:
        z += 7.129769325256348 * (0.177304983139 - Q.max_dr) * (0.465420272571 - Q.pt1_over_pt0)
    if Q.girth2 < 0.006679471442 and Q.girth2_top3 > 0.002915531053:
        z += 117475.375 * (0.006679471442 - Q.girth2) * (Q.girth2_top3 - 0.002915531053)
    if Q.width < 0.005019718802 and Q.centroid_offset > 0.006789738266:
        z += -23972.138671875 * (0.005019718802 - Q.width) * (Q.centroid_offset - 0.006789738266)
    if Q.girth < 0.061086014472 and Q.lam1 > 0.00027588256:
        z += -9670.8759765625 * (0.061086014472 - Q.girth) * (Q.lam1 - 0.00027588256)
    if Q.width < 0.005019718802 and Q.mass_over_sum_pt_sq > 0.00012320649:
        z += -78984.78125 * (0.005019718802 - Q.width) * (Q.mass_over_sum_pt_sq - 0.00012320649)
    if Q.max_dr < 0.177304983139 and Q.lam2 < 0.000194798295:
        z += 35039.984375 * (0.177304983139 - Q.max_dr) * (0.000194798295 - Q.lam2)
    if Q.mass < 15.454033088684 and Q.lam2 < 0.000306123359:
        z += -429.0279846191406 * (15.454033088684 - Q.mass) * (0.000306123359 - Q.lam2)
    if Q.log_sum_pt > 6.701242202626 and Q.pt_7 < 48.71875:
        z += 0.15141579508781433 * (Q.log_sum_pt - 6.701242202626) * (48.71875 - Q.pt_7)
    if Q.log_sum_pt > 6.701242202626 and Q.girth2 < 0.018827652745:
        z += -163.0625457763672 * (Q.log_sum_pt - 6.701242202626) * (0.018827652745 - Q.girth2)
    if Q.log_sum_pt > 6.701242202626 and Q.mass_over_sum_pt_sq < 0.00023679558:
        z += 19498.99609375 * (Q.log_sum_pt - 6.701242202626) * (0.00023679558 - Q.mass_over_sum_pt_sq)
    if Q.log_sum_pt > 6.701242202626 and Q.centroid_offset < 0.023554160423:
        z += 127.61711883544922 * (Q.log_sum_pt - 6.701242202626) * (0.023554160423 - Q.centroid_offset)
    if Q.mass < 21.784077072144 and Q.centroid_offset < 0.026856224803:
        z += 2.0524020195007324 * (21.784077072144 - Q.mass) * (0.026856224803 - Q.centroid_offset)
    if Q.LHA < 0.196739721581 and Q.width < 0.000561123155:
        z += 30981.44140625 * (0.196739721581 - Q.LHA) * (0.000561123155 - Q.width)
    if Q.girth2 < 0.006679471442 and Q.log_sum_pt > 6.670067010936:
        z += -398.5220031738281 * (0.006679471442 - Q.girth2) * (Q.log_sum_pt - 6.670067010936)
    if Q.LHA < 0.196739721581 and Q.z_6 > 0.02160287394:
        z += 37.0086784362793 * (0.196739721581 - Q.LHA) * (Q.z_6 - 0.02160287394)
    if Q.mass < 29.644699859619 and Q.centroid_offset < 0.023554160423:
        z += -6.564306735992432 * (29.644699859619 - Q.mass) * (0.023554160423 - Q.centroid_offset)
    if Q.e2 < 0.024547699839 and Q.sum_pt > 788.4484375:
        z += 0.2921964228153229 * (0.024547699839 - Q.e2) * (Q.sum_pt - 788.4484375)
    if Q.girth < 0.061086014472 and Q.width < 0.005019718802:
        z += -20337.24609375 * (0.061086014472 - Q.girth) * (0.005019718802 - Q.width)
    if Q.girth < 0.061086014472 and Q.lam2 < 0.000194798295:
        z += 151982.921875 * (0.061086014472 - Q.girth) * (0.000194798295 - Q.lam2)
    if Q.girth < 0.061086014472 and Q.centroid_offset > 0.006789738266:
        z += -1883.640380859375 * (0.061086014472 - Q.girth) * (Q.centroid_offset - 0.006789738266)
    if Q.e2 < 0.024547699839 and Q.centroid_offset < 0.031170772021:
        z += 2641.13232421875 * (0.024547699839 - Q.e2) * (0.031170772021 - Q.centroid_offset)
    if Q.girth2 < 0.006679471442 and Q.centroid_offset > 0.018377780003:
        z += -20309.263671875 * (0.006679471442 - Q.girth2) * (Q.centroid_offset - 0.018377780003)
    if Q.LHA < 0.196739721581 and Q.lam2 < 0.000306123359:
        z += -78873.6796875 * (0.196739721581 - Q.LHA) * (0.000306123359 - Q.lam2)
    if Q.mass < 15.454033088684 and Q.z_7 < 0.058613700176:
        z += -1.4062317609786987 * (15.454033088684 - Q.mass) * (0.058613700176 - Q.z_7)
    if Q.pt_7 > 34.53125 and Q.width < 0.013238675006:
        z += 2.89288330078125 * (Q.pt_7 - 34.53125) * (0.013238675006 - Q.width)
    if Q.z_dr_0_0p05 > 0.847731333971 and Q.lam2 < 0.000537286005:
        z += -9163.5322265625 * (Q.z_dr_0_0p05 - 0.847731333971) * (0.000537286005 - Q.lam2)
    if Q.mass < 21.784077072144 and Q.centroid_offset > 0.031170772021:
        z += -2.236372232437134 * (21.784077072144 - Q.mass) * (Q.centroid_offset - 0.031170772021)
    return max(0.0, z)


def neuron_9(Q):
    z = -1.5281912088394165
    if Q.girth < 0.054649224505:
        z += 84.28038024902344 * Q.girth - 4.60585742159565
    if Q.e2 < 0.020459658932:
        z += -185.45787143707275 * Q.e2 + 3.6594542312475755
    if 0.020459658932 <= Q.e2 < 0.032346998155:
        z += 11.352461814880371 * Q.e2 - 0.36721806138064333
    if Q.mass < 29.644699859619:
        z += 0.14135080948472023 * Q.mass - 5.73542592150568
    if 29.644699859619 <= Q.mass < 41.377904891968:
        z += 0.038800496608018875 * Q.mass - 2.695352675765847
    if 41.377904891968 <= Q.mass < 53.332374954224:
        z += 0.0911683589220047 * Q.mass - 4.862225101989628
    if Q.lam2 >= 0.001130644719:
        z += 859.0785522460938 * Q.lam2 - 0.9713126283032115
    if Q.width < 0.000172198326:
        z += -8338.86865234375 * Q.width + 10.452225775021635
    if 0.000172198326 <= Q.width < 0.006096650059:
        z += -1521.876953125 * Q.width + 9.27835121606027
    if Q.girth2 < 4.8108519e-05:
        z += -24804.981689453125 * Q.girth2 + 11.365101471865959
    if 4.8108519e-05 <= Q.girth2 < 0.000964142894:
        z += -4611.977783203125 * Q.girth2 + 10.393645959775057
    if 0.000964142894 <= Q.girth2 < 0.004372139461:
        z += -1340.99853515625 * Q.girth2 + 7.239954561349199
    if 0.004372139461 <= Q.girth2 < 0.007520088344:
        z += -437.40289306640625 * Q.girth2 + 3.28930839778056
    if Q.girth2 >= 0.018827652745:
        z += 197.82542419433594 * Q.girth2 - 3.7245883908632784
    if Q.lam1 < 0.001503553356:
        z += 785.2687301635742 * Q.lam1 - 1.5182911364141098
    if 0.001503553356 <= Q.lam1 < 0.005954149834:
        z += 75.85448455810547 * Q.lam1 - 0.45164896663979925
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.47418564558029175 * Q.n_dr_0p2_0p4 - 0.47418564558029175
    if Q.max_dr < 0.15984864831:
        z += -10.68047046661377 * Q.max_dr + 2.3666505529646065
    if 0.15984864831 <= Q.max_dr < 0.221586732566:
        z += -16.517516613006592 * Q.max_dr + 3.2996944895885933
    if Q.max_dr >= 0.221586732566:
        z += -5.837046146392822 * Q.max_dr + 0.9330439366239871
    if Q.C2 >= 0.051192347892:
        z += 18.68101692199707 * Q.C2 - 0.956325117247213
    if Q.centroid_offset < 0.009480684835:
        z += -122.58174133300781 * Q.centroid_offset + 2.5773535992993337
    if 0.009480684835 <= Q.centroid_offset < 0.018377780003:
        z += -159.06256103515625 * Q.centroid_offset + 2.9232167534178615
    if Q.log_sum_pt >= 6.377722943814:
        z += -3.794092893600464 * Q.log_sum_pt + 24.197673298477326
    if Q.z_dr_0p2_0p4 >= 0.1009733513:
        z += -3.1217141151428223 * Q.z_dr_0p2_0p4 + 0.31520993600648484
    if Q.mass_over_sum_pt < 0.076373631775:
        z += 40.241920471191406 * Q.mass_over_sum_pt - 3.073421615985607
    if Q.pt_5 < 35.5:
        z += -0.07125543802976608 * Q.pt_5 + 2.529568050056696
    if Q.mass < 53.332374954224 and Q.centroid_offset < 0.026856224803:
        z += 2.8021340370178223 * (53.332374954224 - Q.mass) * (0.026856224803 - Q.centroid_offset)
    if Q.mass < 53.332374954224 and Q.lam1 < 0.000872228216:
        z += -79.353271484375 * (53.332374954224 - Q.mass) * (0.000872228216 - Q.lam1)
    if Q.mass < 53.332374954224 and Q.n_dr_0p05_0p1 > 3.0:
        z += 0.014485448598861694 * (53.332374954224 - Q.mass) * (Q.n_dr_0p05_0p1 - 3.0)
    if Q.e2 < 0.032346998155 and Q.dr01 < 0.055953954317:
        z += 457.1907653808594 * (0.032346998155 - Q.e2) * (0.055953954317 - Q.dr01)
    if Q.mass < 53.332374954224 and Q.log_sum_pt < 6.842716632804:
        z += 0.12083146721124649 * (53.332374954224 - Q.mass) * (6.842716632804 - Q.log_sum_pt)
    if Q.girth < 0.054649224505 and Q.mean_phi < 0.002834883542:
        z += -1733.9052734375 * (0.054649224505 - Q.girth) * (0.002834883542 - Q.mean_phi)
    if Q.max_dr < 0.221586732566 and Q.z_top5 < 0.90890302062:
        z += -37.948543548583984 * (0.221586732566 - Q.max_dr) * (0.90890302062 - Q.z_top5)
    if Q.width < 0.006096650059 and Q.C2 > 0.030867108516:
        z += -10696.2421875 * (0.006096650059 - Q.width) * (Q.C2 - 0.030867108516)
    if Q.width < 0.006096650059 and Q.centroid_offset > 0.003343241496:
        z += -46014.05859375 * (0.006096650059 - Q.width) * (Q.centroid_offset - 0.003343241496)
    if Q.mass < 29.644699859619 and Q.mean_phi2 < 0.002127561159:
        z += -19.208194732666016 * (29.644699859619 - Q.mass) * (0.002127561159 - Q.mean_phi2)
    if Q.log_sum_pt > 6.377722943814 and Q.dr_5 < 0.021648628542:
        z += 162.35501098632812 * (Q.log_sum_pt - 6.377722943814) * (0.021648628542 - Q.dr_5)
    if Q.girth < 0.054649224505 and Q.dr_5 < 0.021648628542:
        z += -1036.6978759765625 * (0.054649224505 - Q.girth) * (0.021648628542 - Q.dr_5)
    if Q.mass < 53.332374954224 and Q.planar_flow < 0.322073846732:
        z += 0.2941550016403198 * (53.332374954224 - Q.mass) * (0.322073846732 - Q.planar_flow)
    if Q.centroid_offset < 0.018377780003 and Q.pt_4 > 47.34375:
        z += 2.5780608654022217 * (0.018377780003 - Q.centroid_offset) * (Q.pt_4 - 47.34375)
    if Q.centroid_offset < 0.018377780003 and Q.z_4 > 0.047491459878:
        z += -2136.375 * (0.018377780003 - Q.centroid_offset) * (Q.z_4 - 0.047491459878)
    if Q.girth < 0.054649224505 and Q.mean_phi2 > 0.000539434783:
        z += 17513.373046875 * (0.054649224505 - Q.girth) * (Q.mean_phi2 - 0.000539434783)
    if Q.e2 < 0.020459658932 and Q.eccentricity > 0.903125533696:
        z += -749.48779296875 * (0.020459658932 - Q.e2) * (Q.eccentricity - 0.903125533696)
    if Q.C2 > 0.051192347892 and Q.eccentricity < 0.620723099573:
        z += 30.11117935180664 * (Q.C2 - 0.051192347892) * (0.620723099573 - Q.eccentricity)
    if Q.girth2 < 0.004372139461 and Q.mass_top2 > 6.77991534008:
        z += -28.028560638427734 * (0.004372139461 - Q.girth2) * (Q.mass_top2 - 6.77991534008)
    if Q.mass < 41.377904891968 and Q.planar_flow < 0.322073846732:
        z += -0.2683734893798828 * (41.377904891968 - Q.mass) * (0.322073846732 - Q.planar_flow)
    if Q.e2 < 0.032346998155 and Q.tau21 < 0.446608647704:
        z += -185.23268127441406 * (0.032346998155 - Q.e2) * (0.446608647704 - Q.tau21)
    if Q.girth2 < 0.004372139461 and Q.n_dr_0p05_0p1 > 3.0:
        z += -276.0014953613281 * (0.004372139461 - Q.girth2) * (Q.n_dr_0p05_0p1 - 3.0)
    if Q.girth2 > 0.018827652745 and Q.pt_7 < 29.0421875:
        z += 35.572269439697266 * (Q.girth2 - 0.018827652745) * (29.0421875 - Q.pt_7)
    if Q.girth2 < 0.000964142894 and Q.mean_eta < 0.006823012256:
        z += -40853.0859375 * (0.000964142894 - Q.girth2) * (0.006823012256 - Q.mean_eta)
    if Q.lam2 > 0.001130644719 and Q.mass_top2 > 16.308019673264:
        z += 4.3150434494018555 * (Q.lam2 - 0.001130644719) * (Q.mass_top2 - 16.308019673264)
    if Q.centroid_offset < 0.009480684835 and Q.mean_phi2 < 0.002127561159:
        z += -71208.8125 * (0.009480684835 - Q.centroid_offset) * (0.002127561159 - Q.mean_phi2)
    if Q.e2 < 0.020459658932 and Q.pt_7 < 53.4375:
        z += -1.281410574913025 * (0.020459658932 - Q.e2) * (53.4375 - Q.pt_7)
    if Q.C2 > 0.051192347892 and Q.n_dr_0p05_0p1 > 1.0:
        z += -5.339889049530029 * (Q.C2 - 0.051192347892) * (Q.n_dr_0p05_0p1 - 1.0)
    return max(0.0, z)


def neuron_10(Q):
    z = -1.6939409971237183
    z += 71.1552963256836 * Q.e2
    if Q.lam2 >= 0.000194798295:
        z += 1410.705322265625 * Q.lam2 - 0.2748029915247693
    if Q.LHA >= 0.303313749495:
        z += -9.855120658874512 * Q.LHA + 2.989193598768863
    if 0.00231612516 <= Q.centroid_offset < 0.037760993714:
        z += -10.888427734375 * Q.centroid_offset + 0.025218961428427732
    if Q.centroid_offset >= 0.037760993714:
        z += 11.730125427246094 * Q.centroid_offset - 0.8288800823273211
    if Q.eccentricity >= 0.903125533696:
        z += 3.5384981632232666 * Q.eccentricity - 3.1957080421433286
    if Q.C2 >= 0.051192347892:
        z += 12.784676551818848 * Q.C2 - 0.6544776097274054
    if Q.lam1 < 0.004183811014:
        z += 665.8495025634766 * Q.lam1 - 3.1573374345043668
    if 0.004183811014 <= Q.lam1 < 0.006506575659:
        z += 159.9597930908203 * Q.lam1 - 1.0407904961434078
    if Q.mass_over_sum_pt < 0.090413827016:
        z += -30.094905853271484 * Q.mass_over_sum_pt + 2.7209956118804937
    if Q.n_dr_0p05_0p1 < 3.0:
        z += -0.1985996663570404 * Q.n_dr_0p05_0p1 + 0.5957989990711212
    if Q.tau32 < 0.269169217348:
        z += -4.241927146911621 * Q.tau32 + 1.1417962101814356
    if Q.girth2_top2 < 0.002412890926:
        z += -452.4322509765625 * Q.girth2_top2 + 1.0916696730111024
    if Q.mass_over_sum_pt_sq < 0.003904593248:
        z += -14.599488258361816 * Q.mass_over_sum_pt_sq + 0.057005063277854824
    if Q.log_sum_pt >= 6.701242202626:
        z += -17.091508865356445 * Q.log_sum_pt + 114.53434051508303
    if 15.454033088684 <= Q.mass < 53.332374954224:
        z += 0.04524412751197815 * Q.mass - 0.6992042436387484
    if Q.mass >= 53.332374954224:
        z += 0.02483793906867504 * Q.mass + 0.3891062498060456
    if Q.max_dr >= 0.121680960059:
        z += -2.628239393234253 * Q.max_dr + 0.31980669263362754
    if Q.girth2 < 0.0016538364:
        z += 814.3907318115234 * Q.girth2 - 0.8331891823850378
    if 0.0016538364 <= Q.girth2 < 0.006679471442:
        z += -102.21192932128906 * Q.girth2 + 0.6827216629332727
    if Q.girth2_top5 < 0.002270363079:
        z += -348.6498718261719 * Q.girth2_top5 + 0.791561796492223
    if Q.z_7 >= 0.06164517166:
        z += -9.261427879333496 * Q.z_7 + 0.5709223114382231
    if Q.sum_pt >= 813.415625:
        z += 0.010981127619743347 * Q.sum_pt - 8.932220786018297
    if Q.lam2 > 0.000194798295 and Q.planar_flow > 0.012569162668:
        z += -630.5682983398438 * (Q.lam2 - 0.000194798295) * (Q.planar_flow - 0.012569162668)
    if Q.centroid_offset > 0.00231612516 and Q.pt_7 < 34.53125:
        z += -0.6769360899925232 * (Q.centroid_offset - 0.00231612516) * (34.53125 - Q.pt_7)
    if Q.lam2 > 0.000194798295 and Q.n_pt_above_50 < 8.0:
        z += -28.23996353149414 * (Q.lam2 - 0.000194798295) * (8.0 - Q.n_pt_above_50)
    if Q.lam2 > 0.000194798295 and Q.z_top2_slots > 0.476619814198:
        z += 1380.3616943359375 * (Q.lam2 - 0.000194798295) * (Q.z_top2_slots - 0.476619814198)
    if Q.lam2 > 0.000194798295 and Q.pt_6 < 38.25:
        z += -14.695383071899414 * (Q.lam2 - 0.000194798295) * (38.25 - Q.pt_6)
    if Q.eccentricity > 0.903125533696 and Q.z_top2_slots > 0.55004856109:
        z += -23.00346565246582 * (Q.eccentricity - 0.903125533696) * (Q.z_top2_slots - 0.55004856109)
    if Q.mass_over_sum_pt < 0.090413827016 and Q.pt_7 > 33.21875:
        z += 0.14357584714889526 * (0.090413827016 - Q.mass_over_sum_pt) * (Q.pt_7 - 33.21875)
    if Q.lam2 > 0.000194798295 and Q.tau21 < 0.501026660204:
        z += 1044.37646484375 * (Q.lam2 - 0.000194798295) * (0.501026660204 - Q.tau21)
    if Q.LHA > 0.303313749495 and Q.tau21 < 0.553068161011:
        z += -18.32386016845703 * (Q.LHA - 0.303313749495) * (0.553068161011 - Q.tau21)
    if Q.eccentricity > 0.903125533696 and Q.z_dr_0p2_0p4 < 0.05643851608:
        z += 177.4636688232422 * (Q.eccentricity - 0.903125533696) * (0.05643851608 - Q.z_dr_0p2_0p4)
    if Q.tau32 < 0.269169217348 and Q.n_dr_0p2_0p4 < 2.0:
        z += -2.5418338775634766 * (0.269169217348 - Q.tau32) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.n_dr_0p2_0p4 > 2.0 and Q.dr_7 > 0.222994708167:
        z += 4.574099540710449 * (Q.n_dr_0p2_0p4 - 2.0) * (Q.dr_7 - 0.222994708167)
    if Q.lam1 < 0.004183811014 and Q.mean_phi > 0.009050007537:
        z += 9993.0498046875 * (0.004183811014 - Q.lam1) * (Q.mean_phi - 0.009050007537)
    if Q.tau21 < 0.391541349888 and Q.mean_eta < -0.006779838586:
        z += -36.644073486328125 * (0.391541349888 - Q.tau21) * (-0.006779838586 - Q.mean_eta)
    if Q.tau21 < 0.391541349888 and Q.pt_4 > 39.8125:
        z += -0.04639672860503197 * (0.391541349888 - Q.tau21) * (Q.pt_4 - 39.8125)
    if Q.n_dr_0p05_0p1 < 3.0 and Q.D2 < 1.432482242584:
        z += -0.1122458279132843 * (3.0 - Q.n_dr_0p05_0p1) * (1.432482242584 - Q.D2)
    if Q.n_dr_0p2_0p4 > 2.0 and Q.dr_7 < 0.042151962757:
        z += 37.06693649291992 * (Q.n_dr_0p2_0p4 - 2.0) * (0.042151962757 - Q.dr_7)
    if Q.lam2 > 0.000194798295 and Q.D2 < 2.055451202393:
        z += -128.8239288330078 * (Q.lam2 - 0.000194798295) * (2.055451202393 - Q.D2)
    if Q.lam1 < 0.004183811014 and Q.dr_7 < 0.175465903809:
        z += -411.668212890625 * (0.004183811014 - Q.lam1) * (0.175465903809 - Q.dr_7)
    if Q.tau21 < 0.391541349888 and Q.z_4 > 0.075444822386:
        z += 36.689537048339844 * (0.391541349888 - Q.tau21) * (Q.z_4 - 0.075444822386)
    if Q.lam1 < 0.004183811014 and Q.sum_pt > 988.4078125:
        z += -0.5860525369644165 * (0.004183811014 - Q.lam1) * (Q.sum_pt - 988.4078125)
    if Q.lam1 < 0.004183811014 and Q.log_sum_pt > 6.701242202626:
        z += 1741.4788818359375 * (0.004183811014 - Q.lam1) * (Q.log_sum_pt - 6.701242202626)
    if Q.mass > 15.454033088684 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.0025558697525411844 * (Q.mass - 15.454033088684) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.eccentricity > 0.903125533696 and Q.mass_top2 < 36.768271023571:
        z += -0.1313740462064743 * (Q.eccentricity - 0.903125533696) * (36.768271023571 - Q.mass_top2)
    if Q.tau21 < 0.391541349888 and Q.dr01 < 0.251215918102:
        z += 2.6207151412963867 * (0.391541349888 - Q.tau21) * (0.251215918102 - Q.dr01)
    return max(0.0, z)


def neuron_11(Q):
    z = -1.36906099319458
    if Q.planar_flow < 0.253403707141:
        z += -8.775918960571289 * Q.planar_flow + 2.223850398177756
    if Q.LHA < 0.154689112391:
        z += 23.616954803466797 * Q.LHA - 3.653285775926643
    if Q.girth < 0.076081777364:
        z += 65.16056823730469 * Q.girth - 5.68438077095765
    if 0.076081777364 <= Q.girth < 0.087236513197:
        z += -40.340476989746094 * Q.girth + 2.3423262636761235
    if 0.087236513197 <= Q.girth < 0.101940929517:
        z += -105.50104522705078 * Q.girth + 8.026707034633773
    if Q.girth >= 0.101940929517:
        z += -150.68946838378906 * Q.girth + 12.633256894639201
    if Q.centroid_offset < 0.014379521101:
        z += -70.36858558654785 * Q.centroid_offset + 4.513381172312717
    if 0.014379521101 <= Q.centroid_offset < 0.049903668404:
        z += -134.72073554992676 * Q.centroid_offset + 5.43873427060584
    if Q.centroid_offset >= 0.049903668404:
        z += -44.27886390686035 * Q.centroid_offset + 0.9253530982931233
    if Q.girth2_top2 < 0.001056655216:
        z += 307.09442138671875 * Q.girth2_top2 - 0.32449292216277836
    if Q.width < 0.003562611091:
        z += -593.8229370117188 * Q.width + 7.706363467098765
    if 0.003562611091 <= Q.width < 0.008678044951:
        z += -1092.928466796875 * Q.width + 9.484482363090793
    if Q.e2_sq < 0.006390124748:
        z += 484.3953552246094 * Q.e2_sq - 3.0953467472370275
    if Q.n_dr_0_0p05 < 7.0:
        z += 0.04834402725100517 * Q.n_dr_0_0p05 - 0.3384081907570362
    if Q.e2 < 0.007078157854:
        z += -31.906524658203125 * Q.e2 - 1.6414635094571632
    if 0.007078157854 <= Q.e2 < 0.04447356835:
        z += 49.93401336669922 * Q.e2 - 2.2207437564537114
    if Q.girth2 < 0.006679471442:
        z += -925.9129333496094 * Q.girth2 + 7.892680896730512
    if 0.006679471442 <= Q.girth2 < 0.013238675334:
        z += -260.4084167480469 * Q.girth2 + 3.4474624835683607
    if Q.lam1 < 0.004839980301:
        z += 627.3829650878906 * Q.lam1 - 4.168216231826703
    if 0.004839980301 <= Q.lam1 < 0.008375572068:
        z += 320.0864562988281 * Q.lam1 - 2.680907182721567
    if Q.mass_top5 < 9.257203159811:
        z += -0.03296806290745735 * Q.mass_top5 + 0.305192056119762
    if Q.mass < 15.454033088684:
        z += -0.01274068746715784 * Q.mass + 0.19689500569003882
    if Q.pt_7 < 29.0421875:
        z += 0.05821162834763527 * Q.pt_7 - 1.6905930251523387
    if Q.sum_pt_top5 < 687.4375:
        z += -0.00276312162168324 * Q.sum_pt_top5 + 1.8994734198058723
    if Q.girth2_top3 < 0.002151567843:
        z += 180.18235778808594 * Q.girth2_top3 - 0.38767456689276636
    if Q.C2 < 0.035786485299:
        z += 11.664087295532227 * Q.C2 - 0.41741668852781666
    if Q.girth2_top5 < 0.000657050184:
        z += 748.3506469726562 * Q.girth2_top5 - 0.49170393028990284
    if Q.max_dr < 0.111761856824:
        z += 1.6963233947753906 * Q.max_dr + 0.4993232871074289
    if 0.111761856824 <= Q.max_dr < 0.221586732566:
        z += -6.272782325744629 * Q.max_dr + 1.3899653396595066
    if Q.n_dr_0p1_0p2 < 3.0:
        z += 0.22522228956222534 * Q.n_dr_0p1_0p2 - 0.675666868686676
    if Q.log_sum_pt < 6.701242202626:
        z += 0.2620537281036377 * Q.log_sum_pt - 1.756085502123576
    if Q.planar_flow < 0.253403707141 and Q.width > 0.006096650059:
        z += -873.2092895507812 * (0.253403707141 - Q.planar_flow) * (Q.width - 0.006096650059)
    if Q.planar_flow < 0.253403707141 and Q.D2 < 1.679198372364:
        z += 1.006485104560852 * (0.253403707141 - Q.planar_flow) * (1.679198372364 - Q.D2)
    if Q.planar_flow < 0.253403707141 and Q.girth2 > 0.013238675334:
        z += -1371.681884765625 * (0.253403707141 - Q.planar_flow) * (Q.girth2 - 0.013238675334)
    if Q.planar_flow < 0.253403707141 and Q.mass < 69.611351776123:
        z += -0.1393420249223709 * (0.253403707141 - Q.planar_flow) * (69.611351776123 - Q.mass)
    if Q.planar_flow < 0.253403707141 and Q.max_dr > 0.102758520097:
        z += -9.344493865966797 * (0.253403707141 - Q.planar_flow) * (Q.max_dr - 0.102758520097)
    if Q.centroid_offset < 0.049903668404 and Q.pt_7 < 48.71875:
        z += -0.5699287056922913 * (0.049903668404 - Q.centroid_offset) * (48.71875 - Q.pt_7)
    if Q.centroid_offset < 0.049903668404 and Q.n_dr_0p05_0p1 > 2.0:
        z += 1.3796573877334595 * (0.049903668404 - Q.centroid_offset) * (Q.n_dr_0p05_0p1 - 2.0)
    if Q.centroid_offset < 0.049903668404 and Q.log_sum_pt < 6.804164030582:
        z += -81.70994567871094 * (0.049903668404 - Q.centroid_offset) * (6.804164030582 - Q.log_sum_pt)
    if Q.e2_sq < 0.006390124748 and Q.D2 < 1.332146394253:
        z += -100.77825927734375 * (0.006390124748 - Q.e2_sq) * (1.332146394253 - Q.D2)
    if Q.centroid_offset < 0.049903668404 and Q.mean_phi2 < 0.001570267399:
        z += -7269.0185546875 * (0.049903668404 - Q.centroid_offset) * (0.001570267399 - Q.mean_phi2)
    if Q.width < 0.003562611091 and Q.C2 > 0.030867108516:
        z += 7516.29150390625 * (0.003562611091 - Q.width) * (Q.C2 - 0.030867108516)
    if Q.centroid_offset < 0.049903668404 and Q.mean_phi < -0.000855675264:
        z += -674.93603515625 * (0.049903668404 - Q.centroid_offset) * (-0.000855675264 - Q.mean_phi)
    if Q.planar_flow < 0.253403707141 and Q.z_6 < 0.034484056668:
        z += -83.10964965820312 * (0.253403707141 - Q.planar_flow) * (0.034484056668 - Q.z_6)
    if Q.LHA < 0.154689112391 and Q.z_7 < 0.028070914944:
        z += 1018.6321411132812 * (0.154689112391 - Q.LHA) * (0.028070914944 - Q.z_7)
    if Q.girth > 0.076081777364 and Q.n_pt_above_50 < 7.0:
        z += 23.702573776245117 * (Q.girth - 0.076081777364) * (7.0 - Q.n_pt_above_50)
    if Q.sum_pt_top5 < 687.4375 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.0009188193362206221 * (687.4375 - Q.sum_pt_top5) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.pt_7 < 29.0421875 and Q.mass_top2 < 22.844978847276:
        z += 0.0017352867871522903 * (29.0421875 - Q.pt_7) * (22.844978847276 - Q.mass_top2)
    if Q.centroid_offset > 0.014379521101 and Q.pt_5 < 29.875:
        z += 5.4465765953063965 * (Q.centroid_offset - 0.014379521101) * (29.875 - Q.pt_5)
    if Q.mass_top5 < 9.257203159811 and Q.eta_0 < -0.053314208984:
        z += -9.07787799835205 * (9.257203159811 - Q.mass_top5) * (-0.053314208984 - Q.eta_0)
    if Q.girth2_top3 < 0.002151567843 and Q.phi_0 > -0.029769897461:
        z += 8289.640625 * (0.002151567843 - Q.girth2_top3) * (Q.phi_0 - -0.029769897461)
    if Q.max_dr < 0.221586732566 and Q.phi_0 < 0.021438598633:
        z += 34.29497146606445 * (0.221586732566 - Q.max_dr) * (0.021438598633 - Q.phi_0)
    if Q.n_dr_0_0p05 < 7.0 and Q.z_1st < 0.500737345219:
        z += -0.1642974466085434 * (7.0 - Q.n_dr_0_0p05) * (0.500737345219 - Q.z_1st)
    if Q.pt_7 < 29.0421875 and Q.n_dr_0p2_0p4 < 2.0:
        z += -0.020686086267232895 * (29.0421875 - Q.pt_7) * (2.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.7954257130622864
    if Q.girth2 >= 0.018827652745:
        z += 260.6923522949219 * Q.girth2 - 4.908225082285993
    if Q.mass >= 91.19:
        z += 0.05697167292237282 * Q.mass - 5.195246853791177
    if Q.mass_over_sum_pt >= 0.13092863437:
        z += -31.683631896972656 * Q.mass_over_sum_pt + 4.148294656152403
    if Q.girth2_top2 >= 0.0140332421:
        z += 12.319104194641113 * Q.girth2_top2 - 0.17287697161852425
    if Q.e2 >= 0.063441075385:
        z += -50.626304626464844 * Q.e2 + 3.2117872082715304
    if Q.mean_phi >= 0.026127964072:
        z += 8.319671630859375 * Q.mean_phi - 0.21737608146193138
    if Q.n_dr_0p2_0p4 >= 1.0:
        z += 0.1916208565235138 * Q.n_dr_0p2_0p4 - 0.1916208565235138
    if Q.girth2 > 0.018827652745 and Q.lam2 > 0.000537286005:
        z += 11113.287109375 * (Q.girth2 - 0.018827652745) * (Q.lam2 - 0.000537286005)
    if Q.girth2 > 0.018827652745 and Q.pt_7 > 15.55390625:
        z += 2.6988260746002197 * (Q.girth2 - 0.018827652745) * (Q.pt_7 - 15.55390625)
    if Q.girth2 > 0.018827652745 and Q.mass < 80.4:
        z += -3.8779118061065674 * (Q.girth2 - 0.018827652745) * (80.4 - Q.mass)
    if Q.girth2_top2 > 0.0140332421 and Q.z_6 < 0.089100391399:
        z += -1154.9747314453125 * (Q.girth2_top2 - 0.0140332421) * (0.089100391399 - Q.z_6)
    if Q.n_dr_0p2_0p4 > 1.0 and Q.lam2 < 0.003408388935:
        z += -55.60203552246094 * (Q.n_dr_0p2_0p4 - 1.0) * (0.003408388935 - Q.lam2)
    if Q.girth2_top2 > 0.0140332421 and Q.width > 0.013238675006:
        z += -1266.5501708984375 * (Q.girth2_top2 - 0.0140332421) * (Q.width - 0.013238675006)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.4233996868133545
    if Q.girth < 0.101940929517:
        z += -44.71196365356445 * Q.girth + 6.635631798161083
    if 0.101940929517 <= Q.girth < 0.148408418149:
        z += -40.31939697265625 * Q.girth + 6.187849467743897
    if Q.girth >= 0.148408418149:
        z += 4.392566680908203 * Q.girth - 0.44778233041718574
    if Q.lam1 < 0.006506575659:
        z += -446.355224609375 * Q.lam1 + 4.829927987161617
    if 0.006506575659 <= Q.lam1 < 0.016433749775:
        z += -193.9810791015625 * Q.lam1 + 3.18783651503956
    if 658.125 <= Q.sum_pt_top5 < 839.9546875:
        z += 0.001236681709997356 * Q.sum_pt_top5 - 0.8138911503920099
    if 839.9546875 <= Q.sum_pt_top5 < 902.40625:
        z += 0.021315949852578342 * Q.sum_pt_top5 - 17.679566548322327
    if Q.sum_pt_top5 >= 902.40625:
        z += 0.024813588126562536 * Q.sum_pt_top5 - 20.835857187004876
    if Q.lam2 < 0.000306123359:
        z += -1490.47705078125 * Q.lam2 + 0.45626984129756987
    if Q.e2 < 0.050284641981:
        z += 87.97148895263672 * Q.e2 - 4.423614826518834
    if Q.pt_6 < 31.90625:
        z += 0.04394249990582466 * Q.pt_6 - 1.402040387620218
    if Q.z_top5_slots >= 0.930764273368:
        z += 9.67802906036377 * Q.z_top5_slots - 9.007963686003873
    if Q.z_7 < 0.028070914944:
        z += 94.4925308227539 * Q.z_7 - 2.652491795568823
    if Q.C2 >= 0.067292226106:
        z += -55.08650207519531 * Q.C2 + 3.706893353032681
    if Q.z_6 < 0.067272114405:
        z += -4.472832679748535 * Q.z_6 + 0.30089691174646616
    if Q.centroid_offset < 0.037760993714:
        z += -43.35221862792969 * Q.centroid_offset + 1.6370228550972066
    if Q.LHA >= 0.09323897448:
        z += -5.284818649291992 * Q.LHA + 0.4927510711727641
    if Q.width < 0.003562611091:
        z += 258.9015884399414 * Q.width - 0.4129505593066103
    if 0.003562611091 <= Q.width < 0.00752008842:
        z += 134.04246520996094 * Q.width + 0.031873937925053664
    if 0.00752008842 <= Q.width < 0.013238675006:
        z += -181.84303283691406 * Q.width + 2.407360813833291
    if Q.z_top5 >= 0.877015459538:
        z += -8.293966293334961 * Q.z_top5 + 7.273936660141843
    if Q.pt_5 < 24.578125:
        z += 0.21056333184242249 * Q.pt_5 - 5.17525189043954
    if Q.mass >= 53.332374954224:
        z += -0.01802026852965355 * Q.mass + 0.9610637179992859
    if Q.sum_pt < 763.825:
        z += 0.0012835005763918161 * Q.sum_pt - 0.980369827762479
    if Q.sum_pt >= 988.4078125:
        z += -0.0407317578792572 * Q.sum_pt + 40.259587704716246
    if Q.pt_7 < 25.578125:
        z += 0.10502336174249649 * Q.pt_7 - 2.686300674569793
    if Q.girth < 0.148408418149 and Q.log_sum_pt < 6.804164030582:
        z += -45.33917236328125 * (0.148408418149 - Q.girth) * (6.804164030582 - Q.log_sum_pt)
    if Q.girth < 0.148408418149 and Q.pt_7 < 38.53125:
        z += -0.8842843770980835 * (0.148408418149 - Q.girth) * (38.53125 - Q.pt_7)
    if Q.sum_pt_top5 > 658.125 and Q.z_7 > 0.023207568189:
        z += -0.2682841122150421 * (Q.sum_pt_top5 - 658.125) * (Q.z_7 - 0.023207568189)
    if Q.lam2 < 0.000306123359 and Q.centroid_offset < 0.049903668404:
        z += -125115.015625 * (0.000306123359 - Q.lam2) * (0.049903668404 - Q.centroid_offset)
    if Q.sum_pt_top5 > 658.125 and Q.pt_7 < 43.5:
        z += 0.0005757395992986858 * (Q.sum_pt_top5 - 658.125) * (43.5 - Q.pt_7)
    if Q.lam1 < 0.016433749775 and Q.pt_6 < 56.53125:
        z += -1.8031131029129028 * (0.016433749775 - Q.lam1) * (56.53125 - Q.pt_6)
    if Q.e2 < 0.050284641981 and Q.pt_dispersion > 0.396830244362:
        z += 82.25191497802734 * (0.050284641981 - Q.e2) * (Q.pt_dispersion - 0.396830244362)
    if Q.sum_pt_top5 > 658.125 and Q.tau32 < 0.362731824815:
        z += -0.0229664109647274 * (Q.sum_pt_top5 - 658.125) * (0.362731824815 - Q.tau32)
    if Q.lam1 < 0.016433749775 and Q.centroid_offset < 0.037760993714:
        z += -2313.5 * (0.016433749775 - Q.lam1) * (0.037760993714 - Q.centroid_offset)
    if Q.sum_pt_top5 > 902.40625 and Q.D2 < 3.885568320751:
        z += -0.0028931419365108013 * (Q.sum_pt_top5 - 902.40625) * (3.885568320751 - Q.D2)
    if Q.tau21 < 0.501026660204 and Q.max_dr > 0.015595615841:
        z += -12.58998966217041 * (0.501026660204 - Q.tau21) * (Q.max_dr - 0.015595615841)
    if Q.sum_pt_top5 > 902.40625 and Q.n_pt_above_50 > 6.0:
        z += -0.013328051194548607 * (Q.sum_pt_top5 - 902.40625) * (Q.n_pt_above_50 - 6.0)
    if Q.sum_pt_top5 > 839.9546875 and Q.n_pt_above_50 > 2.0:
        z += 0.0005865641287527978 * (Q.sum_pt_top5 - 839.9546875) * (Q.n_pt_above_50 - 2.0)
    if Q.width < 0.003562611091 and Q.n_pt_above_50 > 7.0:
        z += 95.57559967041016 * (0.003562611091 - Q.width) * (Q.n_pt_above_50 - 7.0)
    if Q.girth > 0.101940929517 and Q.pt_1 < 116.0:
        z += -0.1404603123664856 * (Q.girth - 0.101940929517) * (116.0 - Q.pt_1)
    if Q.sum_pt_top5 < 531.1875 and Q.pt_5 < 29.875:
        z += 0.0014359570341184735 * (531.1875 - Q.sum_pt_top5) * (29.875 - Q.pt_5)
    if Q.lam1 < 0.006506575659 and Q.z_dr_0p05_0p1 > 0.163898047805:
        z += 153.5025177001953 * (0.006506575659 - Q.lam1) * (Q.z_dr_0p05_0p1 - 0.163898047805)
    if Q.sum_pt < 763.825 and Q.z_4 < 0.037477688199:
        z += 3.528134822845459 * (763.825 - Q.sum_pt) * (0.037477688199 - Q.z_4)
    if Q.width < 0.00752008842 and Q.m012 > 16.899120053094:
        z += -3.9407613277435303 * (0.00752008842 - Q.width) * (Q.m012 - 16.899120053094)
    if Q.girth > 0.101940929517 and Q.pt_4 > 81.375:
        z += -4.805459976196289 * (Q.girth - 0.101940929517) * (Q.pt_4 - 81.375)
    if Q.sum_pt > 988.4078125 and Q.n_pt_above_50 > 6.0:
        z += 0.014069809578359127 * (Q.sum_pt - 988.4078125) * (Q.n_pt_above_50 - 6.0)
    if Q.e2 < 0.050284641981 and Q.n_pt_above_50 < 5.0:
        z += 3.0710079669952393 * (0.050284641981 - Q.e2) * (5.0 - Q.n_pt_above_50)
    if Q.girth > 0.101940929517 and Q.pt_balance01 < 0.476255698625:
        z += -53.37581253051758 * (Q.girth - 0.101940929517) * (0.476255698625 - Q.pt_balance01)
    if Q.sum_pt > 988.4078125 and Q.D2 < 3.885568320751:
        z += 0.006110693793743849 * (Q.sum_pt - 988.4078125) * (3.885568320751 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.979142963886261
    if Q.planar_flow < 0.111513564951:
        z += -2.544314384460449 * Q.planar_flow + 0.2837255673672939
    if Q.z_dr_0p05_0p1 < 0.588259786367:
        z += -1.7361515760421753 * Q.z_dr_0p05_0p1 + 1.0213081552233005
    if Q.z_dr_0p05_0p1 >= 0.750909513235:
        z += -3.866259813308716 * Q.z_dr_0p05_0p1 + 2.9032112744516896
    if 0.002464291268 <= Q.lam1 < 0.004183811014:
        z += 290.8104248046875 * Q.lam1 - 0.7166415904895621
    if 0.004183811014 <= Q.lam1 < 0.00543336053:
        z += 95.79997253417969 * Q.lam1 + 0.0992452875649098
    if 0.00543336053 <= Q.lam1 < 0.005954149834:
        z += -56.79554748535156 * Q.lam1 + 0.9283517630938559
    if 0.005954149834 <= Q.lam1 < 0.008375572068:
        z += 108.77055358886719 * Q.lam1 - 0.057453610133230804
    if 0.008375572068 <= Q.lam1 < 0.012003726523:
        z += 51.07249069213867 * Q.lam1 + 0.42580067384231557
    if Q.lam1 >= 0.012003726523:
        z += -341.61833572387695 * Q.lam1 + 5.139553962231032
    if Q.max_dr < 0.080507021025:
        z += 26.407808303833008 * Q.max_dr - 4.063692471534866
    if 0.080507021025 <= Q.max_dr < 0.177304983139:
        z += 20.01776123046875 * Q.max_dr - 3.5492488174487895
    if Q.mass < 76.655700683594:
        z += 0.01249249093234539 * Q.mass - 0.9576206457023805
    if Q.girth2 < 0.013238675334:
        z += -636.7699584960938 * Q.girth2 + 8.42999074297444
    if Q.width < 0.00752008842:
        z += 709.8958740234375 * Q.width - 4.122460426180526
    if 0.00752008842 <= Q.width < 0.008678044951:
        z += -1050.1424560546875 * Q.width + 9.11318343859612
    if Q.e2 < 0.035560912266:
        z += -93.55570602416992 * Q.e2 + 3.714002054306907
    if 0.035560912266 <= Q.e2 < 0.038466955721:
        z += -133.19683837890625 * Q.e2 + 5.12367688409858
    if Q.mass_over_sum_pt_sq < 0.011660904657:
        z += 412.0372619628906 * Q.mass_over_sum_pt_sq - 4.8047272268806
    if Q.girth < 0.087236513197:
        z += 85.57461547851562 * Q.girth - 7.465231072519729
    if Q.D2 < 0.74595130682:
        z += 0.03776228427886963 * Q.D2 - 0.43620453408994553
    if 0.74595130682 <= Q.D2 < 1.122624260187:
        z += 1.08326256275177 * Q.D2 - 1.2160968330974795
    if Q.n_dr_0p05_0p1 < 5.0:
        z += 0.1278899908065796 * Q.n_dr_0p05_0p1 - 0.639449954032898
    if Q.centroid_offset >= 0.049903668404:
        z += -118.68329620361328 * Q.centroid_offset + 5.922731858838829
    if Q.e2_sq < 0.017162483186:
        z += -101.11784362792969 * Q.e2_sq + 1.7354332910689205
    if Q.LHA < 0.303313749495:
        z += 3.3993377685546875 * Q.LHA - 1.031065884380289
    if Q.planar_flow < 0.111513564951 and Q.centroid_offset > 0.018377780003:
        z += -839.3114013671875 * (0.111513564951 - Q.planar_flow) * (Q.centroid_offset - 0.018377780003)
    if Q.planar_flow < 0.111513564951 and Q.max_dr < 0.15984864831:
        z += -118.37430572509766 * (0.111513564951 - Q.planar_flow) * (0.15984864831 - Q.max_dr)
    if Q.planar_flow < 0.111513564951 and Q.sum_pt < 739.5:
        z += -0.02375268191099167 * (0.111513564951 - Q.planar_flow) * (739.5 - Q.sum_pt)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.C2 < 0.067292226106:
        z += -58.71162414550781 * (0.588259786367 - Q.z_dr_0p05_0p1) * (0.067292226106 - Q.C2)
    if Q.lam1 > 0.007330079875 and Q.max_dr < 0.13261153996:
        z += 22720.744140625 * (Q.lam1 - 0.007330079875) * (0.13261153996 - Q.max_dr)
    if Q.lam1 > 0.00543336053 and Q.max_dr < 0.15984864831:
        z += 5460.93896484375 * (Q.lam1 - 0.00543336053) * (0.15984864831 - Q.max_dr)
    if Q.planar_flow < 0.111513564951 and Q.centroid_offset > 0.009480684835:
        z += 892.7675170898438 * (0.111513564951 - Q.planar_flow) * (Q.centroid_offset - 0.009480684835)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.eccentricity > 0.970449631164:
        z += 35.617557525634766 * (0.588259786367 - Q.z_dr_0p05_0p1) * (Q.eccentricity - 0.970449631164)
    if Q.z_dr_0p05_0p1 > 0.750909513235 and Q.n_dr_0p2_0p4 < 1.0:
        z += 4.774363994598389 * (Q.z_dr_0p05_0p1 - 0.750909513235) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p05_0p1 < 0.588259786367 and Q.n_dr_0p1_0p2 < 3.0:
        z += 0.38745495676994324 * (0.588259786367 - Q.z_dr_0p05_0p1) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 76.655700683594 and Q.D2 < 0.74595130682:
        z += -0.048378899693489075 * (76.655700683594 - Q.mass) * (0.74595130682 - Q.D2)
    if Q.width < 0.00752008842 and Q.planar_flow < 0.111513564951:
        z += -4436.61181640625 * (0.00752008842 - Q.width) * (0.111513564951 - Q.planar_flow)
    if Q.girth2 < 0.013238675334 and Q.eccentricity > 0.970449631164:
        z += 4903.55224609375 * (0.013238675334 - Q.girth2) * (Q.eccentricity - 0.970449631164)
    if Q.girth2 < 0.004372139461 and Q.D2 < 0.87567204833:
        z += -5043.587890625 * (0.004372139461 - Q.girth2) * (0.87567204833 - Q.D2)
    if Q.width < 0.00752008842 and Q.D2 < 1.002470755577:
        z += 1262.0054931640625 * (0.00752008842 - Q.width) * (1.002470755577 - Q.D2)
    if Q.mass < 76.655700683594 and Q.n_dr_0p1_0p2 > 4.0:
        z += 0.013331599533557892 * (76.655700683594 - Q.mass) * (Q.n_dr_0p1_0p2 - 4.0)
    if Q.e2 < 0.038466955721 and Q.D2 < 1.002470755577:
        z += 57.562416076660156 * (0.038466955721 - Q.e2) * (1.002470755577 - Q.D2)
    if Q.D2 < 1.122624260187 and Q.centroid_offset < 0.031170772021:
        z += 27.79163360595703 * (1.122624260187 - Q.D2) * (0.031170772021 - Q.centroid_offset)
    if Q.planar_flow < 0.111513564951 and Q.sum_pt > 840.01953125:
        z += -0.02356705255806446 * (0.111513564951 - Q.planar_flow) * (Q.sum_pt - 840.01953125)
    if Q.width < 0.00752008842 and Q.log_sum_pt < 6.804164030582:
        z += 159.2598419189453 * (0.00752008842 - Q.width) * (6.804164030582 - Q.log_sum_pt)
    if Q.width < 0.008678044951 and Q.D2 < 1.002470755577:
        z += -1486.65576171875 * (0.008678044951 - Q.width) * (1.002470755577 - Q.D2)
    if Q.girth2 < 0.013238675334 and Q.D2 < 1.002470755577:
        z += 161.4713592529297 * (0.013238675334 - Q.girth2) * (1.002470755577 - Q.D2)
    if Q.width < 0.00752008842 and Q.n_dr_0p1_0p2 < 3.0:
        z += 166.02978515625 * (0.00752008842 - Q.width) * (3.0 - Q.n_dr_0p1_0p2)
    if Q.girth2 < 0.004372139461 and Q.n_dr_0p2_0p4 < 1.0:
        z += -94.65123748779297 * (0.004372139461 - Q.girth2) * (1.0 - Q.n_dr_0p2_0p4)
    if Q.girth2 < 0.013238675334 and Q.n_dr_0p1_0p2 < 3.0:
        z += -46.8410758972168 * (0.013238675334 - Q.girth2) * (3.0 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_15(Q):
    z = -2.0611205101013184
    if Q.tau21 < 0.23799610585:
        z += -5.139349937438965 * Q.tau21 + 1.2231452717109148
    if Q.width < 0.006096650059:
        z += 1043.2965545654297 * Q.width - 4.030889448880846
    if 0.006096650059 <= Q.width < 0.006679471358:
        z += 1184.2969055175781 * Q.width - 4.890519246832283
    if 0.006679471358 <= Q.width < 0.00752008842:
        z += -65.08517456054688 * Q.width + 3.454692572248014
    if 0.00752008842 <= Q.width < 0.013238675006:
        z += -518.52783203125 * Q.width + 6.864621449827475
    if Q.girth2 < 0.018827652745:
        z += -124.13481140136719 * Q.girth2 + 2.337167122631008
    if Q.lam1 < 0.006506575659:
        z += -37.15106201171875 * Q.lam1 - 1.6957243755403275
    if 0.006506575659 <= Q.lam1 < 0.008375572068:
        z += 681.4339599609375 * Q.lam1 - 6.371252188429592
    if 0.008375572068 <= Q.lam1 < 0.012003726523:
        z += 182.97262573242188 * Q.lam1 - 2.1963533604872247
    if Q.lam2 < 0.000306123359:
        z += 2366.2525634765625 * Q.lam2 - 1.2700318488883569
    if 0.000306123359 <= Q.lam2 < 0.001130644719:
        z += 661.798095703125 * Q.lam2 - 0.7482585219509948
    if 0.176724128067 <= Q.LHA < 0.346713497427:
        z += 2.3776962757110596 * Q.LHA - 0.42019630113319023
    if Q.LHA >= 0.346713497427:
        z += -51.330620527267456 * Q.LHA + 18.201202058544798
    if Q.e2_sq < 0.008168570676:
        z += 677.6780242919922 * Q.e2_sq - 6.342247509690482
    if 0.008168570676 <= Q.e2_sq < 0.011657374702:
        z += 231.19288635253906 * Q.e2_sq - 2.69510210464845
    if Q.mass_over_sum_pt_sq < 0.007182835724:
        z += -577.4603271484375 * Q.mass_over_sum_pt_sq + 4.147802667034524
    if Q.mass < 36.229410171509:
        z += 0.062203798443078995 * Q.mass - 2.253606928020182
    if Q.mass >= 80.4:
        z += -0.19141560792922974 * Q.mass + 15.389814877510071
    if Q.e2 < 0.024547699839:
        z += -354.948557138443 * Q.e2 + 10.317794928403291
    if 0.024547699839 <= Q.e2 < 0.032346998155:
        z += -109.57946228981018 * Q.e2 + 4.294548038291931
    if 0.032346998155 <= Q.e2 < 0.041109715588:
        z += -112.74347019195557 * Q.e2 + 4.396894196065033
    if 0.041109715588 <= Q.e2 < 0.063441075385:
        z += 10.655768394470215 * Q.e2 - 0.6760134059986853
    if Q.z_dr_0p05_0p1 >= 0.750909513235:
        z += -9.224932670593262 * Q.z_dr_0p05_0p1 + 6.927089701300835
    if Q.z_dr_0p1_0p2 < 0.328461505473:
        z += -3.815389633178711 * Q.z_dr_0p1_0p2 + 1.2532086228799566
    if Q.n_dr_0_0p05 < 2.0:
        z += -0.10891985893249512 * Q.n_dr_0_0p05 + 0.21783971786499023
    if Q.log_sum_pt >= 6.896095378249:
        z += 23.07244873046875 * Q.log_sum_pt - 159.10980705507257
    if Q.planar_flow < 0.083662731125:
        z += 2.1556692123413086 * Q.planar_flow - 0.18034917370655143
    if Q.mass_over_sum_pt < 0.06813910019:
        z += 37.41251754760742 * Q.mass_over_sum_pt - 2.549255281536555
    if Q.D2 < 0.74595130682:
        z += 0.6307396292686462 * Q.D2 - 0.470501050716109
    if Q.n_dr_0p05_0p1 >= 5.0:
        z += 0.22982318699359894 * Q.n_dr_0p05_0p1 - 1.1491159349679947
    if Q.n_dr_0p1_0p2 < 1.0:
        z += -0.3363337218761444 * Q.n_dr_0p1_0p2 + 0.3363337218761444
    if Q.girth >= 0.033604209498:
        z += 28.78110122680664 * Q.girth - 0.9671661552087552
    if Q.girth2_top3 < 0.002151567843:
        z += 702.5822143554688 * Q.girth2_top3 - 1.5116532994709597
    if Q.girth2_top2 < 0.007639643088:
        z += -34.186851501464844 * Q.girth2_top2 + 0.2611753437736483
    if Q.tau21 < 0.23799610585 and Q.z_dr_0p05_0p1 < 0.588259786367:
        z += -8.180723190307617 * (0.23799610585 - Q.tau21) * (0.588259786367 - Q.z_dr_0p05_0p1)
    if Q.tau21 < 0.23799610585 and Q.max_dr < 0.121680960059:
        z += -110.59868621826172 * (0.23799610585 - Q.tau21) * (0.121680960059 - Q.max_dr)
    if Q.tau21 < 0.23799610585 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += 36.49673080444336 * (0.23799610585 - Q.tau21) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.tau21 < 0.23799610585 and Q.mass < 76.655700683594:
        z += -0.0678974837064743 * (0.23799610585 - Q.tau21) * (76.655700683594 - Q.mass)
    if Q.tau21 < 0.23799610585 and Q.girth2_top5 > 0.00832969537:
        z += -2024.0169677734375 * (0.23799610585 - Q.tau21) * (Q.girth2_top5 - 0.00832969537)
    if Q.width < 0.00752008842 and Q.e2 > 0.024547699839:
        z += -44515.21875 * (0.00752008842 - Q.width) * (Q.e2 - 0.024547699839)
    if Q.LHA > 0.176724128067 and Q.sum_pt_top3 > 353.0625:
        z += 0.036749377846717834 * (Q.LHA - 0.176724128067) * (Q.sum_pt_top3 - 353.0625)
    if Q.z_dr_0p05_0p1 > 0.750909513235 and Q.n_dr_0p2_0p4 < 2.0:
        z += 1.1746230125427246 * (Q.z_dr_0p05_0p1 - 0.750909513235) * (2.0 - Q.n_dr_0p2_0p4)
    if Q.z_dr_0p1_0p2 < 0.328461505473 and Q.n_dr_0p2_0p4 > 0.0:
        z += -0.8615151047706604 * (0.328461505473 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.lam1 < 0.006506575659 and Q.pt1_dr01 > 1.21960336377:
        z += 6.627881050109863 * (0.006506575659 - Q.lam1) * (Q.pt1_dr01 - 1.21960336377)
    if Q.mass > 80.4 and Q.z_dr_0p2_0p4 < 0.20552001074:
        z += 0.6611398458480835 * (Q.mass - 80.4) * (0.20552001074 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt > 6.896095378249 and Q.mean_phi > 0.01746432744:
        z += 50422.56640625 * (Q.log_sum_pt - 6.896095378249) * (Q.mean_phi - 0.01746432744)
    if Q.LHA > 0.176724128067 and Q.z_top5 > 0.865048766136:
        z += -117.89598846435547 * (Q.LHA - 0.176724128067) * (Q.z_top5 - 0.865048766136)
    if Q.width < 0.006096650059 and Q.log_sum_pt > 6.896095378249:
        z += -4771.1767578125 * (0.006096650059 - Q.width) * (Q.log_sum_pt - 6.896095378249)
    if Q.girth2 < 0.018827652745 and Q.mass_top2 > 22.844978847276:
        z += -1.8455836772918701 * (0.018827652745 - Q.girth2) * (Q.mass_top2 - 22.844978847276)
    if Q.LHA > 0.176724128067 and Q.dr_7 < 0.042151962757:
        z += -339.25323486328125 * (Q.LHA - 0.176724128067) * (0.042151962757 - Q.dr_7)
    if Q.width < 0.00752008842 and Q.planar_flow < 0.061262048692:
        z += -1616.62744140625 * (0.00752008842 - Q.width) * (0.061262048692 - Q.planar_flow)
    if Q.D2 < 0.74595130682 and Q.pt_7 < 23.21640625:
        z += -0.16589275002479553 * (0.74595130682 - Q.D2) * (23.21640625 - Q.pt_7)
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
