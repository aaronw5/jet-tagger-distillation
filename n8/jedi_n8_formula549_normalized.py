"""JEDI-linear jet tagger, 8 particles, 3 features: the simplest formula at the network's accuracy (from the 931-term tuned formula), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  15.9%   (on for 92% of jets)
  neuron  9:  12.4%   (on for 68% of jets)
  neuron  4:  11.9%   (on for 91% of jets)
  neuron  3:   8.7%   (on for 23% of jets)
  neuron  6:   7.9%   (on for 48% of jets)
  neuron 11:   6.8%   (on for 80% of jets)
  neuron  5:   5.8%   (on for 62% of jets)
  neuron  7:   5.3%   (on for 51% of jets)
  neuron  2:   4.9%   (on for 89% of jets)
  neuron 10:   4.9%   (on for 84% of jets)
  neuron 14:   3.3%   (on for 32% of jets)
  neuron  8:   3.3%   (on for 42% of jets)
  neuron  0:   3.2%   (on for 46% of jets)
  neuron 15:   2.8%   (on for 33% of jets)
  neuron  1:   2.7%   (on for 63% of jets)
  neuron 12:   0.2%   (on for 4% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

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
    # scale S = 19.28;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.275173927523934 * (-0.026510785423851293
        + 0.12460493155293029 * max(0.0, 0.013 - Q.girth2) / 0.008032714809063365   # +12.5%  girth2 < 0.013
        - 0.11503506861963136 * max(0.0, 0.078 - Q.girth) / 0.029329642267302183   # -11.5%  girth < 0.078
        + 0.07919609678088013 * max(0.0, 0.0089 - Q.width) / 0.00462581375706813   # +7.9%  width < 0.0089
        - 0.06678940628213627 * max(0.0, 0.0044 - Q.width) / 0.0015796041995144003   # -6.7%  width < 0.0044
        - 0.06145320048472614 * max(0.0, 59.0 - Q.mass) / 23.2715349262494   # -6.1%  mass < 59
        - 0.05890922170928486 * max(0.0, 22.0 - Q.mass) / 4.505894818974343   # -5.9%  mass < 22
        + 0.0544156675127738 * max(0.0, 0.0006 - Q.lam1) / 0.00011170090049957705   # +5.4%  lam1 < 0.0006
        - 0.051282022224989864 * max(0.0, 30.0 - Q.mass) / 7.488408316225965   # -5.1%  mass < 30
        - 0.03797105457475891 * max(0.0, 0.0015 - Q.lam1) / 0.0003913896690588136   # -3.8%  lam1 < 0.0015
        + 0.03190708276764799 * max(0.0, Q.z_dr_0_0p05 - 0.85) / 0.05347952781446232   # +3.2%  z_dr_0_0p05 > 0.85
        - 0.030843418945050126 * max(0.0, 0.0085 - Q.girth2_top5) / 0.004681198934530141   # -3.1%  girth2_top5 < 0.0085
        + 0.02857735204059308 * max(0.0, 0.033 - Q.centroid_offset) / 0.01806011249083648   # +2.9%  centroid_offset < 0.033
        - 0.02461280676300585 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.phi_1 - -0.056) / 0.41983728424909306   # -2.5%  mass < 30 and phi_1 > -0.056
        + 0.022455967932551866 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.eccentricity - 0.96) / 0.00016842127922600672   # +2.2%  girth2 < 0.019 and eccentricity > 0.96
        + 0.02224461466672126 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) * max(0.0, 7.9 - Q.n_pt_above_50) / 0.011403425974781972   # +2.2%  mass_over_sum_pt_sq < 0.0081 and n_pt_above_50 < 7.9
        - 0.021396945406942855 * max(0.0, Q.sum_pt - 810.0) / 32.73252732036174   # -2.1%  sum_pt > 810
        - 0.020289027818711718 * max(0.0, 64.0 - Q.mass) * max(0.0, 40.0 - Q.pt_7) / 224.75548277358615   # -2.0%  mass < 64 and pt_7 < 40
        + 0.018626295834458725 * max(0.0, Q.sum_pt_top5 - 700.0) / 33.55374690043986   # +1.9%  sum_pt_top5 > 700
        - 0.01664895274072652 * max(0.0, Q.tau32 - 0.44) / 0.16979442316858728   # -1.7%  tau32 > 0.44
        - 0.016523492652900252 * max(0.0, Q.sum_pt - 900.0) / 12.638618840270484   # -1.7%  sum_pt > 900
        + 0.015402248523107776 * max(0.0, Q.n_dr_0_0p05 - 3.8) / 1.7260524369642503   # +1.5%  n_dr_0_0p05 > 3.8
        + 0.015005055573351675 * max(0.0, 63.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.012) / 0.15804647867132016   # +1.5%  mass < 63 and centroid_offset > 0.012
        + 0.011507981131346972 * max(0.0, 0.14 - Q.planar_flow) * max(0.0, 0.051 - Q.centroid_offset) / 0.0016430987989731595   # +1.2%  planar_flow < 0.14 and centroid_offset < 0.051
        + 0.009024319908834726 * max(0.0, Q.sum_pt - 900.0) * max(0.0, 26.0 - Q.pt_7) / 72.78047523866371   # +0.9%  sum_pt > 900 and pt_7 < 26
        - 0.008227181099462891 * max(0.0, 0.0065 - Q.lam1) * max(0.0, 0.88 - Q.D2) / 8.0497637880906e-05   # -0.8%  lam1 < 0.0065 and D2 < 0.88
        + 0.005652801734069642 * max(0.0, 56.0 - Q.mass) * max(0.0, Q.C2 - 0.024) / 0.0694004691732492   # +0.6%  mass < 56 and C2 > 0.024
        - 0.0054497314257498965 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 370.0 - Q.sum_pt_top2) / 2.793737263021868   # -0.5%  planar_flow < 0.15 and sum_pt_top2 < 370
        - 0.004941387246603121 * max(0.0, Q.sum_pt - 900.0) * max(0.0, Q.pt_7 - 26.0) / 105.36072856363248   # -0.5%  sum_pt > 900 and pt_7 > 26
        - 0.004399653524457328 * max(0.0, Q.D2 - 3.9) / 0.1199492035427987   # -0.4%  D2 > 3.9
        + 0.003893478065181435 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, 0.82 - Q.z_top5) / 0.0017252291238935983   # +0.4%  planar_flow < 0.18 and z_top5 < 0.82
        - 0.0027702287772585214 * max(0.0, 0.019 - Q.girth2) * max(0.0, Q.mass_top2 - 29.0) / 0.006356743035796423   # -0.3%  girth2 < 0.019 and mass_top2 > 29
        + 0.002178260836333635 * max(0.0, 0.013 - Q.planar_flow) / 0.0005177109307009194   # +0.2%  planar_flow < 0.013
        + 0.0020739543921617014 * max(0.0, 0.087 - Q.girth) * max(0.0, Q.m01 - 29.0) / 0.005287808416226054   # +0.2%  girth < 0.087 and m01 > 29
        - 0.001431675320858626 * max(0.0, 0.0053 - Q.lam1) * max(0.0, Q.mass_top3 - 15.0) / 0.0009483089627935962   # -0.1%  lam1 < 0.0053 and mass_top3 > 15
        - 0.0012107432138877639 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.87 - Q.D2) / 0.018232254710356218   # -0.1%  mass < 29 and D2 < 0.87
        - 0.0010056059759758634 * max(0.0, 0.15 - Q.planar_flow) * max(0.0, 0.028 - Q.dr_2) / 5.100850023550585e-05   # -0.1%  planar_flow < 0.15 and dr_2 < 0.028
        - 0.0009134187321961632 * max(0.0, Q.D2 - 3.9) * max(0.0, -0.01 - Q.phi_2) / 0.000139732578823329   # -0.1%  D2 > 3.9 and phi_2 < -0.01
        - 0.0006889115481504956 * max(0.0, 30.0 - Q.mass) * max(0.0, Q.dr_7 - 0.13) / 0.00299074097100914   # -0.1%  mass < 30 and dr_7 > 0.13
        + 0.000345464424426108 * max(0.0, 0.0039 - Q.girth2_top3) * max(0.0, Q.m01 - 29.0) / 1.0536213396495538e-05   # +0.0%  girth2_top3 < 0.0039 and m01 > 29
        - 9.527123516383776e-05 * max(0.0, Q.z_dr_0p05_0p1 - 0.84) * max(0.0, 0.0073 - Q.dr_7) / 1.5176608496471132e-08   # -0.0%  z_dr_0p05_0p1 > 0.84 and dr_7 < 0.0073
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 12.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.444398186524346 * (0.10285752519443427
        - 0.14674596677916468 * max(0.0, 0.0087 - Q.width) / 0.004464951694049875   # -14.7%  width < 0.0087
        + 0.12980890554684843 * max(0.0, 0.0081 - Q.e2_sq) / 0.004239878500739923   # +13.0%  e2_sq < 0.0081
        - 0.07451189443071976 * max(0.0, 0.25 - Q.max_dr) / 0.13096831693900318   # -7.5%  max_dr < 0.25
        - 0.06682492685215347 * max(0.0, 0.054 - Q.z_7) / 0.009703570577987873   # -6.7%  z_7 < 0.054
        + 0.05959198943183715 * max(0.0, Q.log_sum_pt - 6.6) / 0.07342440051652795   # +6.0%  log_sum_pt > 6.6
        + 0.053220724948516616 * max(0.0, Q.log_sum_pt - 6.4) / 0.19309034782356593   # +5.3%  log_sum_pt > 6.4
        + 0.051849773026973786 * max(0.0, Q.pt_7 - 34.0) / 4.744406039915966   # +5.2%  pt_7 > 34
        + 0.04545670155999385 * max(0.0, 0.054 - Q.z_7) * max(0.0, 0.014 - Q.girth2_top2) / 0.00011736126441049085   # +4.5%  z_7 < 0.054 and girth2_top2 < 0.014
        - 0.03523413341531555 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.027 - Q.centroid_offset) / 0.00332172413619174   # -3.5%  log_sum_pt > 6.4 and centroid_offset < 0.027
        - 0.03504232169227482 * max(0.0, 0.044 - Q.C2) * max(0.0, 0.22 - Q.tau21) / 0.00124594458433985   # -3.5%  C2 < 0.044 and tau21 < 0.22
        + 0.032648429511951016 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.19 - Q.max_dr) / 0.01838416547562856   # +3.3%  log_sum_pt > 6.4 and max_dr < 0.19
        + 0.028052811963816135 * max(0.0, Q.LHA - 0.28) / 0.02624814754356556   # +2.8%  LHA > 0.28
        - 0.025287001915014557 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 80.4 - Q.mass) / 204.338649853146   # -2.5%  pt_7 > 34 and mass < 80.4
        - 0.023321112099870294 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0068 - Q.girth2_top3) / 0.0003874729040365245   # -2.3%  log_sum_pt > 6.6 and girth2_top3 < 0.0068
        + 0.020704313581546094 * max(0.0, 0.21 - Q.tau21) * max(0.0, 0.00021 - Q.lam2) / 6.2842127411566726e-06   # +2.1%  tau21 < 0.21 and lam2 < 0.00021
        - 0.0187049177731075 * max(0.0, 0.0081 - Q.e2_sq) * max(0.0, 0.086 - Q.planar_flow) / 5.502871035809597e-05   # -1.9%  e2_sq < 0.0081 and planar_flow < 0.086
        - 0.018671336085276956 * max(0.0, 0.0085 - Q.lam1) * max(0.0, Q.z_7 - 0.037) / 6.600952867034293e-05   # -1.9%  lam1 < 0.0085 and z_7 > 0.037
        - 0.01684049252642592 * max(0.0, 0.044 - Q.z_7) / 0.0052655727300510435   # -1.7%  z_7 < 0.044
        + 0.014213286169097027 * max(0.0, 0.037 - Q.e2) * max(0.0, Q.eccentricity - 0.98) / 3.3754922257111184e-05   # +1.4%  e2 < 0.037 and eccentricity > 0.98
        - 0.013357280827270654 * max(0.0, 6.1 - Q.mass_top5) / 0.9183608911811235   # -1.3%  mass_top5 < 6.1
        - 0.012885057717884797 * max(0.0, Q.e2 - 0.029) * max(0.0, 0.64 - Q.tau32) / 0.002422156931989533   # -1.3%  e2 > 0.029 and tau32 < 0.64
        - 0.012302090295789424 * max(0.0, Q.e2 - 0.034) / 0.007360197604200993   # -1.2%  e2 > 0.034
        - 0.010105341876579318 * max(0.0, 0.33 - Q.LHA) * max(0.0, 0.082 - Q.planar_flow) / 0.0009901960482134819   # -1.0%  LHA < 0.33 and planar_flow < 0.082
        - 0.009497143473134224 * max(0.0, 0.048 - Q.max_dr) / 0.005497034186708515   # -0.9%  max_dr < 0.048
        + 0.00900465978153298 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02) / 7.94734552167218e-06   # +0.9%  lam1 < 0.0083 and centroid_offset > 0.02
        + 0.008834157438339346 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 1.2 - Q.D2) / 0.014619118724087863   # +0.9%  log_sum_pt > 6.6 and D2 < 1.2
        + 0.008430153222738893 * max(0.0, Q.pt_7 - 35.0) * max(0.0, 0.08 - Q.max_dr) / 0.08529120607900345   # +0.8%  pt_7 > 35 and max_dr < 0.08
        - 0.002935588849461666 * max(0.0, Q.pt_7 - 54.0) / 0.3044303046218487   # -0.3%  pt_7 > 54
        - 0.002465679062560616 * max(0.0, 0.042 - Q.z_7) * max(0.0, Q.z_dr_0p05_0p1 - 0.2) / 0.00034169144827038287   # -0.2%  z_7 < 0.042 and z_dr_0p05_0p1 > 0.2
        + 0.0024294815289939184 * max(0.0, Q.pt_7 - 54.0) * max(0.0, 52.0 - Q.mass_top3) / 12.441743017944985   # +0.2%  pt_7 > 54 and mass_top3 < 52
        + 0.0024045610828244326 * max(0.0, Q.n_pt_above_50 - 6.0) * max(0.0, 0.00075 - Q.girth2_top2) / 6.195303432399045e-05   # +0.2%  n_pt_above_50 > 6 and girth2_top2 < 0.00075
        + 0.0023257298973104424 * max(0.0, Q.pt_7 - 35.0) * max(0.0, Q.centroid_offset - 0.016) / 0.019424368400292297   # +0.2%  pt_7 > 35 and centroid_offset > 0.016
        + 0.0018799076978958482 * max(0.0, 46.0 - Q.mass) * max(0.0, 0.2 - Q.tau21) / 0.04542586397384126   # +0.2%  mass < 46 and tau21 < 0.2
        - 0.0018698750868060981 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.n_pt_above_50 - 7.0) / 0.0069254375414514475   # -0.2%  log_sum_pt > 6.6 and n_pt_above_50 > 7
        - 0.0016486099193677457 * max(0.0, Q.e2 - 0.025) * max(0.0, 81.0 - Q.pt_1) / 0.04804674072802347   # -0.2%  e2 > 0.025 and pt_1 < 81
        - 0.0008936429316060834 * max(0.0, Q.n_pt_above_50 - 6.1) * max(0.0, 0.16 - Q.tau32) / 0.0005762097656724893   # -0.1%  n_pt_above_50 > 6.1 and tau32 < 0.16
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 10.08;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.07668807939229 * (0.2302343767834397
        - 0.10247319448923706 * max(0.0, Q.LHA - 0.1) / 0.14523071974218607   # -10.2%  LHA > 0.1
        - 0.09382696918834037 * max(0.0, 54.0 - Q.pt_7) / 19.65623912568934   # -9.4%  pt_7 < 54
        - 0.0764045833034198 * max(0.0, 36.0 - Q.mass) / 10.011770530357682   # -7.6%  mass < 36
        + 0.0668904764023575 * max(0.0, 0.0057 - Q.lam1) / 0.0023567638677919832   # +6.7%  lam1 < 0.0057
        + 0.06455373700236686 * max(0.0, Q.log_sum_pt - 6.3) / 0.27103661338832297   # +6.5%  log_sum_pt > 6.3
        + 0.06244529169756606 * max(0.0, 790.0 - Q.sum_pt) / 113.17297238543856   # +6.2%  sum_pt < 790
        + 0.060646024803713304 * max(0.0, 69.0 - Q.mass) / 31.179136489903424   # +6.1%  mass < 69
        + 0.05742796691628134 * max(0.0, 0.012 - Q.mass_over_sum_pt_sq) / 0.007495902974728329   # +5.7%  mass_over_sum_pt_sq < 0.012
        + 0.05720785537309993 * max(0.0, Q.pt_7 - 31.0) / 6.477142857142857   # +5.7%  pt_7 > 31
        - 0.05452917187990936 * max(0.0, 0.68 - Q.planar_flow) / 0.42267188943185846   # -5.5%  planar_flow < 0.68
        + 0.052767398455237674 * max(0.0, 37.0 - Q.mass) * max(0.0, 0.0011 - Q.lam2) / 0.010895914241689278   # +5.3%  mass < 37 and lam2 < 0.0011
        - 0.040545147393236974 * max(0.0, Q.pt_7 - 30.0) * max(0.0, 0.051 - Q.C2) / 0.2184817130559542   # -4.1%  pt_7 > 30 and C2 < 0.051
        - 0.028978299492078367 * max(0.0, 0.16 - Q.max_dr) / 0.05782282872334372   # -2.9%  max_dr < 0.16
        - 0.02848382913969614 * max(0.0, 69.0 - Q.mass) * max(0.0, 0.068 - Q.z_7) / 0.6322085056110635   # -2.8%  mass < 69 and z_7 < 0.068
        + 0.01500098784888187 * max(0.0, 6.5 - Q.log_sum_pt) / 0.08351396432924671   # +1.5%  log_sum_pt < 6.5
        - 0.013658302237667452 * max(0.0, 0.0059 - Q.lam1) * max(0.0, Q.max_dr - 0.078) / 4.5422591202323655e-05   # -1.4%  lam1 < 0.0059 and max_dr > 0.078
        - 0.011804257315671512 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.max_dr - 0.092) / 0.26669914569261516   # -1.2%  pt_7 > 30 and max_dr > 0.092
        - 0.011065536617606696 * max(0.0, 0.0098 - Q.mass_over_sum_pt) / 0.0005994836608963242   # -1.1%  mass_over_sum_pt < 0.0098
        - 0.010810277806530601 * max(0.0, Q.z_7 - 0.044) * max(0.0, 0.066 - Q.dr_7) / 0.00022553167185918418   # -1.1%  z_7 > 0.044 and dr_7 < 0.066
        - 0.00955239070366137 * max(0.0, Q.sum_pt_top5 - 740.0) * max(0.0, 0.0015 - Q.mean_eta2) / 0.025066786857625593   # -1.0%  sum_pt_top5 > 740 and mean_eta2 < 0.0015
        + 0.00946858678235444 * max(0.0, 53.0 - Q.pt_6) * max(0.0, Q.lam2 - -0.00063) / 0.016535874446870432   # +0.9%  pt_6 < 53 and lam2 > -0.00063
        - 0.008672625446191838 * max(0.0, 68.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.01) / 0.21524960948442254   # -0.9%  mass < 68 and centroid_offset > 0.01
        + 0.008665331060662479 * max(0.0, 8.5 - Q.mass) / 0.6021919876066577   # +0.9%  mass < 8.5
        - 0.00801467669215798 * max(0.0, 0.004 - Q.lam1) * max(0.0, 0.0062 - Q.centroid_offset) / 2.3008945038191325e-06   # -0.8%  lam1 < 0.004 and centroid_offset < 0.0062
        + 0.007711553454319466 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, Q.eccentricity - 0.79) / 0.009581617603790064   # +0.8%  log_sum_pt < 6.5 and eccentricity > 0.79
        - 0.006448258814518305 * max(0.0, 0.017 - Q.z_7) / 0.00026413452328899543   # -0.6%  z_7 < 0.017
        - 0.005926497302770789 * max(0.0, Q.pt_7 - 30.0) * max(0.0, Q.centroid_offset - 0.013) / 0.04665583181514138   # -0.6%  pt_7 > 30 and centroid_offset > 0.013
        + 0.005905781888249908 * max(0.0, 790.0 - Q.sum_pt) * max(0.0, 0.064 - Q.dr_7) / 1.4374570520004528   # +0.6%  sum_pt < 790 and dr_7 < 0.064
        - 0.005900549012289787 * max(0.0, 71.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0) / 35.391661841672885   # -0.6%  mass < 71 and max_pair_mass > 13
        + 0.005710316117163077 * max(0.0, Q.log_sum_pt - 6.8) / 0.013197494116371295   # +0.6%  log_sum_pt > 6.8
        - 0.0029527066653932794 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # -0.3%  log_sum_pt > 6.9
        + 0.002634173019543084 * max(0.0, 14.0 - Q.pt_7) * max(0.0, 7.9 - Q.n_pt_above_10) / 0.21234991892069272   # +0.3%  pt_7 < 14 and n_pt_above_10 < 7.9
        + 0.0020738908783318505 * max(0.0, 3.7e-05 - Q.girth2_top2) / 1.953079578658595e-06   # +0.2%  girth2_top2 < 3.7e-05
        - 0.0008433547994933454 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.003934362617941143   # -0.1%  log_sum_pt > 6.9 and n_pt_above_50 > 5
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 19.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.740433579544685 * (0.06990727911011921
        - 0.17499155620344056 * max(0.0, 0.0088 - Q.girth2) / 0.00454527525291471   # -17.5%  girth2 < 0.0088
        - 0.15137306366299652 * max(0.0, 0.013 - Q.girth2) / 0.008032714809063365   # -15.1%  girth2 < 0.013
        + 0.12531877591605634 * max(0.0, 0.043 - Q.e2) / 0.019029592094159647   # +12.5%  e2 < 0.043
        + 0.06045482772559757 * max(0.0, Q.LHA - 0.32) / 0.013623339169862733   # +6.0%  LHA > 0.32
        - 0.04457612456018087 * max(0.0, Q.e2 - 0.028) / 0.00987600478242156   # -4.5%  e2 > 0.028
        + 0.03894922099791033 * max(0.0, Q.mass_over_sum_pt - 0.065) * max(0.0, 4.9 - Q.n_dr_0_0p05) / 0.07119208426706075   # +3.9%  mass_over_sum_pt > 0.065 and n_dr_0_0p05 < 4.9
        + 0.03253375953203211 * max(0.0, Q.centroid_offset - 0.013) / 0.007756407235929453   # +3.3%  centroid_offset > 0.013
        + 0.03225309144624065 * max(0.0, Q.mass - 37.0) * max(0.0, 0.0013 - Q.lam2) / 0.011555172584927302   # +3.2%  mass > 37 and lam2 < 0.0013
        + 0.030607182793031224 * max(0.0, Q.max_dr - 0.15) / 0.02466118608093126   # +3.1%  max_dr > 0.15
        + 0.023919700904517765 * max(0.0, Q.width - 0.019) * max(0.0, 0.49 - Q.pt_dispersion) / 6.813640215702856e-05   # +2.4%  width > 0.019 and pt_dispersion < 0.49
        - 0.023634250014548733 * max(0.0, Q.mass_over_sum_pt - 0.07) * max(0.0, 0.52 - Q.tau32) / 0.0026210693405311924   # -2.4%  mass_over_sum_pt > 0.07 and tau32 < 0.52
        + 0.020494433333639976 * max(0.0, Q.lam1 - 0.016) / 0.0007276420862826016   # +2.0%  lam1 > 0.016
        + 0.02046502253954166 * max(0.0, Q.LHA - 0.31) * max(0.0, Q.eccentricity - 0.96) / 0.0001970675210466864   # +2.0%  LHA > 0.31 and eccentricity > 0.96
        + 0.01934783573840682 * max(0.0, Q.e2 - 0.051) / 0.0032643988572817156   # +1.9%  e2 > 0.051
        + 0.019044163057562 * max(0.0, Q.lam1 - 0.0086) * max(0.0, 38.0 - Q.pt_7) / 0.00903702009413032   # +1.9%  lam1 > 0.0086 and pt_7 < 38
        - 0.01842423896384272 * max(0.0, Q.LHA - 0.32) * max(0.0, Q.planar_flow - 0.012) / 0.0046688378115455256   # -1.8%  LHA > 0.32 and planar_flow > 0.012
        + 0.014758194889337944 * max(0.0, Q.lam1 - 0.012) * max(0.0, 38.0 - Q.pt_6) / 0.0036971213955196893   # +1.5%  lam1 > 0.012 and pt_6 < 38
        - 0.013898778164918434 * max(0.0, Q.lam1 - 0.0028) * max(0.0, 0.077 - Q.z_7) / 6.859197680034957e-05   # -1.4%  lam1 > 0.0028 and z_7 < 0.077
        - 0.01286267661820811 * max(0.0, Q.e2 - 0.027) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.019531908725915356   # -1.3%  e2 > 0.027 and n_pt_above_50 > 3
        + 0.011306797994415021 * max(0.0, Q.max_dr - 0.14) * max(0.0, -0.041 - Q.phi_0) / 0.00031704700966772554   # +1.1%  max_dr > 0.14 and phi_0 < -0.041
        - 0.010942586648159497 * max(0.0, Q.C2 - 0.094) / 0.0008745401008761339   # -1.1%  C2 > 0.094
        - 0.010692859828159916 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.15 - Q.max_dr) / 6.0830457986505015e-05   # -1.1%  LHA > 0.31 and max_dr < 0.15
        - 0.010056079898780428 * max(0.0, Q.mass - 70.0) / 2.349247068786621   # -1.0%  mass > 70
        - 0.009855121987363618 * max(0.0, Q.mass_top5 - 53.0) / 2.0718251438749937   # -1.0%  mass_top5 > 53
        - 0.009470676018380574 * max(0.0, Q.centroid_offset - 0.012) * max(0.0, -0.0092 - Q.mean_phi) / 7.822395434904956e-05   # -0.9%  centroid_offset > 0.012 and mean_phi < -0.0092
        - 0.005317067938515413 * max(0.0, Q.mass_over_sum_pt - 0.069) * max(0.0, 0.042 - Q.dr_7) / 9.629470319100008e-06   # -0.5%  mass_over_sum_pt > 0.069 and dr_7 < 0.042
        + 0.0052144445544832045 * max(0.0, -0.016 - Q.mean_eta) * max(0.0, 0.13 - Q.z_4) / 6.019613823508432e-05   # +0.5%  mean_eta < -0.016 and z_4 < 0.13
        - 0.005068186718670316 * max(0.0, -0.015 - Q.mean_eta) * max(0.0, 69.0 - Q.pt_4) / 0.0318624851237713   # -0.5%  mean_eta < -0.015 and pt_4 < 69
        - 0.004974738077266663 * max(0.0, Q.max_dr - 0.16) * max(0.0, 0.047 - Q.dr_1) / 0.00015344294779674126   # -0.5%  max_dr > 0.16 and dr_1 < 0.047
        - 0.004658374969988456 * max(0.0, Q.mass_top5 - 54.0) * max(0.0, 0.035 - Q.eta_7) / 0.12052207297990901   # -0.5%  mass_top5 > 54 and eta_7 < 0.035
        + 0.0046444562522927274 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, Q.phi_7 - -0.03) / 0.00037269748033124235   # +0.5%  centroid_offset > 0.015 and phi_7 > -0.03
        + 0.004615097295416542 * max(0.0, Q.lam1 - 0.018) * max(0.0, Q.eccentricity - 0.96) / 7.0623272576206645e-06   # +0.5%  lam1 > 0.018 and eccentricity > 0.96
        - 0.004429851784572167 * max(0.0, Q.mass - 70.0) * max(0.0, Q.mean_phi - 0.00098) / 0.01721401474814456   # -0.4%  mass > 70 and mean_phi > 0.00098
        - 0.0038018803784409002 * max(0.0, Q.LHA - 0.42) * max(0.0, Q.pt_7 - 38.0) / 0.002493380966378299   # -0.4%  LHA > 0.42 and pt_7 > 38
        - 0.0036768691987351548 * max(0.0, Q.max_dr - 0.15) * max(0.0, 0.046 - Q.dr_3) / 0.00014316172031223867   # -0.4%  max_dr > 0.15 and dr_3 < 0.046
        + 0.0035766617886436223 * max(0.0, Q.mass_over_sum_pt - 0.11) * max(0.0, 0.049 - Q.dr_7) / 3.2239659577723704e-06   # +0.4%  mass_over_sum_pt > 0.11 and dr_7 < 0.049
        + 0.0026741156112913152 * max(0.0, Q.mass - 70.0) * max(0.0, 0.18 - Q.max_dr) / 0.013570231776020498   # +0.3%  mass > 70 and max_dr < 0.18
        + 0.002462636986325087 * max(0.0, Q.LHA - 0.31) * max(0.0, 30.0 - Q.pt_5) / 0.0030383451161925296   # +0.2%  LHA > 0.31 and pt_5 < 30
        + 0.0017600335960280657 * max(0.0, Q.mass_over_sum_pt - 0.091) * max(0.0, 0.15 - Q.max_dr) / 4.044682922020863e-06   # +0.2%  mass_over_sum_pt > 0.091 and max_dr < 0.15
        - 0.0013574007225150002 * max(0.0, Q.mass - 38.0) * max(0.0, 0.046 - Q.dr_6) / 0.02093412406533854   # -0.1%  mass > 38 and dr_6 < 0.046
        - 0.0008147644603215875 * max(0.0, Q.mass - 36.0) * max(0.0, 20.0 - Q.pt_7) / 3.6889458054935482   # -0.1%  mass > 36 and pt_7 < 20
        - 0.0007224002292242933 * max(0.0, 0.038 - Q.e2) * max(0.0, Q.phi_0 - 0.08) / 2.8464059366966104e-07   # -0.1%  e2 < 0.038 and phi_0 > 0.08
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 31.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.518322528345937 * (-0.21479569523128694
        + 0.19841174689370206 * max(0.0, 0.017 - Q.e2_sq) / 0.011888983711042285   # +19.8%  e2_sq < 0.017
        + 0.11462108345125804 * max(0.0, 0.13 - Q.girth) / 0.07342813570660987   # +11.5%  girth < 0.13
        + 0.11067454717026443 * max(0.0, Q.girth2 - 0.0034) / 0.004157659205472027   # +11.1%  girth2 > 0.0034
        + 0.07447653723682102 * max(0.0, 0.24 - Q.tau21) / 0.05711375964536474   # +7.4%  tau21 < 0.24
        - 0.045121020958754435 * max(0.0, Q.C2 - 0.013) / 0.017134203510678063   # -4.5%  C2 > 0.013
        - 0.0413645688180915 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.0013 - Q.lam2) / 7.714448646476741e-05   # -4.1%  tau21 < 0.26 and lam2 < 0.0013
        + 0.03893420293456442 * max(0.0, Q.e2 - 0.02) / 0.013960645796082756   # +3.9%  e2 > 0.02
        - 0.0373315017126134 * max(0.0, 0.003 - Q.e2_sq) / 0.0010600237040050882   # -3.7%  e2_sq < 0.003
        - 0.03732297889687975 * max(0.0, Q.mass_over_sum_pt - 0.089) / 0.008586552456865   # -3.7%  mass_over_sum_pt > 0.089
        + 0.03588920532500087 * max(0.0, 44.0 - Q.mass) / 13.879356426005003   # +3.6%  mass < 44
        - 0.03584816871547885 * max(0.0, Q.girth2 - 0.0081) / 0.002334450709969056   # -3.6%  girth2 > 0.0081
        - 0.030802726364543676 * max(0.0, 0.00033 - Q.lam2) / 0.00021816859872136456   # -3.1%  lam2 < 0.00033
        + 0.02924083410369489 * max(0.0, Q.max_dr - 0.094) / 0.050088154362940916   # +2.9%  max_dr > 0.094
        + 0.017699065508503024 * max(0.0, 760.0 - Q.sum_pt) / 94.38999241071429   # +1.8%  sum_pt < 760
        + 0.016415423349235368 * max(0.0, Q.mass_over_sum_pt - 0.11) / 0.005072417721181774   # +1.6%  mass_over_sum_pt > 0.11
        + 0.012589388505929751 * max(0.0, 0.014 - Q.centroid_offset) / 0.004099136439716375   # +1.3%  centroid_offset < 0.014
        - 0.012566255817428923 * max(0.0, 0.013 - Q.centroid_offset) * max(0.0, 0.65 - Q.z_dr_0p05_0p1) / 0.0017145770728460097   # -1.3%  centroid_offset < 0.013 and z_dr_0p05_0p1 < 0.65
        + 0.01085364623492529 * max(0.0, Q.width - -0.00013) * max(0.0, 0.063 - Q.C2) / 0.00015985454329016022   # +1.1%  width > -0.00013 and C2 < 0.063
        + 0.00975001514539853 * max(0.0, 430.0 - Q.sum_pt_top5) / 13.905163891806723   # +1.0%  sum_pt_top5 < 430
        + 0.009414635176019414 * max(0.0, 0.0087 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016) / 1.7352836723069638e-05   # +0.9%  girth2_top2 < 0.0087 and centroid_offset > 0.016
        - 0.009332677185204098 * max(0.0, Q.C2 - 0.067) / 0.0028838267605509654   # -0.9%  C2 > 0.067
        - 0.009232757682696602 * max(0.0, Q.max_dr - 0.2) / 0.011829310344280296   # -0.9%  max_dr > 0.2
        - 0.009229181551735542 * max(0.0, 0.23 - Q.tau21) * max(0.0, 65.0 - Q.mass) / 0.4706930757609402   # -0.9%  tau21 < 0.23 and mass < 65
        - 0.008937370886314553 * max(0.0, Q.e2 - 0.064) / 0.0017072178069715825   # -0.9%  e2 > 0.064
        - 0.006998557838982633 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.98) / 0.00023170462521254033   # -0.7%  max_dr > 0.11 and eccentricity > 0.98
        + 0.0065923258640640135 * max(0.0, Q.C2 - 0.015) * max(0.0, Q.pt_7 - 39.0) / 0.022010492880882034   # +0.7%  C2 > 0.015 and pt_7 > 39
        - 0.006445581860709356 * max(0.0, 0.25 - Q.tau21) * max(0.0, Q.e2_sq - 0.011) / 8.571895694881587e-05   # -0.6%  tau21 < 0.25 and e2_sq > 0.011
        + 0.0059869484993486785 * max(0.0, 0.23 - Q.tau21) * max(0.0, 410.0 - Q.sum_pt_top2) / 4.3378982474268595   # +0.6%  tau21 < 0.23 and sum_pt_top2 < 410
        + 0.005637342124646064 * max(0.0, 0.24 - Q.tau21) * max(0.0, Q.pt_7 - 32.0) / 0.3732764018639193   # +0.6%  tau21 < 0.24 and pt_7 > 32
        - 0.004652365885367591 * max(0.0, Q.C2 - 0.065) * max(0.0, 36.0 - Q.pt_7) / 0.015386649369872973   # -0.5%  C2 > 0.065 and pt_7 < 36
        - 0.004265295664104291 * max(0.0, Q.max_dr - 0.098) * max(0.0, Q.pt_7 - 38.0) / 0.08197253928048448   # -0.4%  max_dr > 0.098 and pt_7 > 38
        - 0.0016373007932033383 * max(0.0, 0.29 - Q.tau21) * max(0.0, 24.0 - Q.pt_7) / 0.044873890848782134   # -0.2%  tau21 < 0.29 and pt_7 < 24
        - 0.0013469311047510364 * max(0.0, 70.0 - Q.mass) * max(0.0, Q.mean_eta2 - 0.0044) / 0.006179477290102543   # -0.1%  mass < 70 and mean_eta2 > 0.0044
        - 0.00037781073976458375 * max(0.0, 0.22 - Q.tau21) * max(0.0, -0.028 - Q.mean_phi) / 2.8624905650416164e-05   # -0.0%  tau21 < 0.22 and mean_phi < -0.028
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.73;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.7298949922401 * (0.008405411047398675
        + 0.10830006242586529 * max(0.0, 53.0 - Q.pt_7) / 18.706220113084296   # +10.8%  pt_7 < 53
        - 0.09689986125766749 * max(0.0, 540.0 - Q.sum_pt_top2) / 188.32443642331933   # -9.7%  sum_pt_top2 < 540
        - 0.07315468203076925 * max(0.0, 0.071 - Q.z_7) * max(0.0, 0.031 - Q.centroid_offset) / 0.00040489192193148097   # -7.3%  z_7 < 0.071 and centroid_offset < 0.031
        + 0.07162703621802918 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.025 - Q.centroid_offset) / 1.3081845762710044e-05   # +7.2%  width < 0.0026 and centroid_offset < 0.025
        - 0.0684302460558343 * max(0.0, 0.023 - Q.dr_0) / 0.0040329159564070505   # -6.8%  dr_0 < 0.023
        + 0.06569044648343815 * max(0.0, 0.034 - Q.e2) / 0.01272804392279354   # +6.6%  e2 < 0.034
        + 0.05477807922038263 * max(0.0, 0.015 - Q.mean_phi2) * max(0.0, 41.0 - Q.max_pair_mass) / 0.3810487411760001   # +5.5%  mean_phi2 < 0.015 and max_pair_mass < 41
        - 0.05182835731873942 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.99 - Q.n_dr_0p2_0p4) / 0.062242410027155796   # -5.2%  log_sum_pt > 6.6 and n_dr_0p2_0p4 < 0.99
        + 0.045277792503873406 * max(0.0, Q.log_sum_pt - 6.6) / 0.07342440051652795   # +4.5%  log_sum_pt > 6.6
        + 0.04175846113626946 * max(0.0, 57.0 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 42.86942139534269   # +4.2%  mass < 57 and n_dr_0p2_0p4 < 2
        + 0.034906568489843706 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.021 - Q.dr_0) / 0.0005610567568373073   # +3.5%  log_sum_pt > 6.6 and dr_0 < 0.021
        - 0.03390114827093318 * max(0.0, Q.log_sum_pt - 6.8) / 0.013197494116371295   # -3.4%  log_sum_pt > 6.8
        + 0.03320706795894345 * max(0.0, 0.07 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.016384592562694998   # +3.3%  z_7 < 0.07 and n_dr_0p2_0p4 < 1
        + 0.03154942454899385 * max(0.0, Q.log_sum_pt - 6.3) * max(0.0, Q.centroid_offset - 0.00063) / 0.002931539135578786   # +3.2%  log_sum_pt > 6.3 and centroid_offset > 0.00063
        - 0.028709069220136923 * max(0.0, 570.0 - Q.sum_pt_top2) * max(0.0, 0.0036 - Q.girth2_top3) / 0.2555688367113954   # -2.9%  sum_pt_top2 < 570 and girth2_top3 < 0.0036
        - 0.023663092200007833 * max(0.0, 0.073 - Q.z_7) * max(0.0, 790.0 - Q.sum_pt) / 1.0423137678124412   # -2.4%  z_7 < 0.073 and sum_pt < 790
        + 0.02104477783292695 * max(0.0, Q.sum_pt - 870.0) * max(0.0, 0.013 - Q.centroid_offset) / 0.13880715644983535   # +2.1%  sum_pt > 870 and centroid_offset < 0.013
        - 0.02090329023668874 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt) / 0.004257547035285846   # -2.1%  LHA < 0.22 and log_sum_pt < 6.8
        + 0.015522584637776676 * max(0.0, 0.024 - Q.z_7) / 0.0007967777114800662   # +1.6%  z_7 < 0.024
        - 0.014279188860048413 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, 0.043 - Q.dr_0) / 0.00011151691703238093   # -1.4%  log_sum_pt > 6.9 and dr_0 < 0.043
        + 0.013534451026536143 * max(0.0, 550.0 - Q.sum_pt_top2) * max(0.0, 0.022 - Q.dr_0) / 0.42965621033770895   # +1.4%  sum_pt_top2 < 550 and dr_0 < 0.022
        + 0.00797580998642633 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.00014 - Q.mean_phi2) / 3.4770966988105176e-06   # +0.8%  log_sum_pt > 6.6 and mean_phi2 < 0.00014
        - 0.007801966334510478 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0028) / 0.00010795457844707693   # -0.8%  LHA < 0.21 and centroid_offset > 0.0028
        - 0.005778350133435028 * max(0.0, Q.sum_pt_top5 - 730.0) * max(0.0, 1.6 - Q.D2) / 6.129815868918706   # -0.6%  sum_pt_top5 > 730 and D2 < 1.6
        - 0.005142950605494258 * max(0.0, Q.sum_pt - 870.0) / 17.79054922777705   # -0.5%  sum_pt > 870
        + 0.004565855605583346 * max(0.0, 5.2e-05 - Q.girth2) / 1.0397649804795496e-06   # +0.5%  girth2 < 5.2e-05
        - 0.004344500438486333 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, Q.centroid_offset - 0.017) / 6.921781523863841e-05   # -0.4%  log_sum_pt > 6.6 and centroid_offset > 0.017
        - 0.0036719658265809575 * max(0.0, 0.15 - Q.LHA) * max(0.0, Q.centroid_offset - 0.0034) / 1.3870545812305552e-05   # -0.4%  LHA < 0.15 and centroid_offset > 0.0034
        + 0.002936905609001045 * max(0.0, 0.081 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 29.0) / 0.005522378139542727   # +0.3%  mass_over_sum_pt < 0.081 and mass_top3 > 29
        + 0.002345670363727249 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.pt_5 - 48.0) / 0.05591786033081933   # +0.2%  log_sum_pt > 6.9 and pt_5 > 48
        - 0.001970103047900881 * max(0.0, 0.22 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - -1.6e-05) / 0.001706068362154448   # -0.2%  LHA < 0.22 and n_dr_0p2_0p4 > -1.6e-05
        - 0.001878612639405786 * max(0.0, 0.21 - Q.LHA) * max(0.0, Q.mass_top3 - 3.5) / 0.013743989442948662   # -0.2%  LHA < 0.21 and mass_top3 > 3.5
        + 0.0017597482965636174 * max(0.0, 0.21 - Q.LHA) * max(0.0, 0.083 - Q.planar_flow) / 5.642672803029783e-05   # +0.2%  LHA < 0.21 and planar_flow < 0.083
        + 0.000861873179180361 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.dr_4 - 0.039) / 5.458485108255786e-05   # +0.1%  log_sum_pt > 6.9 and dr_4 > 0.039
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 29.22;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.21923861016119 * (0.35935227950630283
        - 0.1935874048012575 * max(0.0, 0.0086 - Q.girth2) / 0.00438486556031768   # -19.4%  girth2 < 0.0086
        - 0.18061708246296637 * max(0.0, 0.013 - Q.width) / 0.008032714809066324   # -18.1%  width < 0.013
        - 0.1477524268207778 * max(0.0, Q.mass_over_sum_pt - 0.0081) / 0.053497068333416185   # -14.8%  mass_over_sum_pt > 0.0081
        + 0.06382987966235079 * max(0.0, 0.0078 - Q.lam1) / 0.00384548553466413   # +6.4%  lam1 < 0.0078
        + 0.05017805027765751 * max(0.0, 0.051 - Q.e2) / 0.02563224517579615   # +5.0%  e2 < 0.051
        + 0.03000189206746221 * max(0.0, Q.girth - 0.083) / 0.008766324430754806   # +3.0%  girth > 0.083
        + 0.023621337392793523 * max(0.0, 0.01 - Q.girth2_top2) / 0.006511297109161857   # +2.4%  girth2_top2 < 0.01
        + 0.023551675667160696 * max(0.0, 0.0036 - Q.girth2) / 0.0012009808568724182   # +2.4%  girth2 < 0.0036
        + 0.022485334373912164 * max(0.0, 50.0 - Q.mass) * max(0.0, 0.76 - Q.z_dr_0p05_0p1) / 11.732220541082123   # +2.2%  mass < 50 and z_dr_0p05_0p1 < 0.76
        + 0.02201718169236722 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 46.0 - Q.pt_7) / 2.029417304077446   # +2.2%  log_sum_pt < 6.7 and pt_7 < 46
        - 0.02073807565232749 * max(0.0, 0.00054 - Q.lam2) * max(0.0, 0.2 - Q.z_dr_0p2_0p4) / 7.508683776963212e-05   # -2.1%  lam2 < 0.00054 and z_dr_0p2_0p4 < 0.2
        + 0.01755792468761495 * max(0.0, Q.max_dr - 0.12) / 0.0363850490033093   # +1.8%  max_dr > 0.12
        - 0.016548221952773653 * max(0.0, 50.0 - Q.mass) / 17.26880163614289   # -1.7%  mass < 50
        + 0.014321490179812322 * max(0.0, Q.centroid_offset - 0.0076) * max(0.0, 0.0035 - Q.lam2) / 2.8274529649798416e-05   # +1.4%  centroid_offset > 0.0076 and lam2 < 0.0035
        - 0.012998738452976808 * max(0.0, 0.11 - Q.max_dr) / 0.027522698586130937   # -1.3%  max_dr < 0.11
        + 0.01277405179887181 * max(0.0, 0.034 - Q.C2) / 0.013722355423889493   # +1.3%  C2 < 0.034
        - 0.010506268263969712 * max(0.0, Q.mass_over_sum_pt - 0.0086) * max(0.0, 42.0 - Q.pt_7) / 0.43298329944611424   # -1.1%  mass_over_sum_pt > 0.0086 and pt_7 < 42
        - 0.010221922043655757 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.52 - Q.tau32) / 0.0054803078761109   # -1.0%  mass_over_sum_pt > 0.015 and tau32 < 0.52
        + 0.009865360901117183 * max(0.0, Q.girth2_top5 - 0.011) / 0.0016194288435117854   # +1.0%  girth2_top5 > 0.011
        - 0.0097120581799382 * max(0.0, 6.8 - Q.log_sum_pt) / 0.27026566224322107   # -1.0%  log_sum_pt < 6.8
        + 0.009242909104773694 * max(0.0, Q.centroid_offset - 0.0084) * max(0.0, 0.096 - Q.C2) / 0.000566186093468373   # +0.9%  centroid_offset > 0.0084 and C2 < 0.096
        - 0.008772665673667067 * max(0.0, 1.7 - Q.D2) * max(0.0, 2.4 - Q.min_pair_mass) / 0.8487768594902269   # -0.9%  D2 < 1.7 and min_pair_mass < 2.4
        + 0.008095073911212548 * max(0.0, 0.05 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.16) / 0.00020390680705052608   # +0.8%  e2 < 0.05 and z_dr_0p1_0p2 > 0.16
        - 0.006968721158338929 * max(0.0, Q.C2 - 0.033) / 0.008519695662476321   # -0.7%  C2 > 0.033
        - 0.006925065767893939 * max(0.0, Q.centroid_offset - 0.019) / 0.005310896300870134   # -0.7%  centroid_offset > 0.019
        - 0.006669088068275234 * max(0.0, 1.8 - Q.D2) * max(0.0, 86.0 - Q.pt_4) / 17.55546626838856   # -0.7%  D2 < 1.8 and pt_4 < 86
        - 0.00608254823350558 * max(0.0, Q.girth2_top5 - 0.011) * max(0.0, Q.pt_7 - 17.0) / 0.029040429443237578   # -0.6%  girth2_top5 > 0.011 and pt_7 > 17
        - 0.005711441769382333 * max(0.0, Q.lam2 - 0.0034) / 0.0001574377168562487   # -0.6%  lam2 > 0.0034
        + 0.005256236385145707 * max(0.0, Q.C2 - 0.01) * max(0.0, Q.pt_7 - 32.0) / 0.0816932048558423   # +0.5%  C2 > 0.01 and pt_7 > 32
        - 0.004493609697630066 * max(0.0, 0.06 - Q.planar_flow) / 0.012504747997713065   # -0.4%  planar_flow < 0.06
        + 0.003804460738105269 * max(0.0, Q.sum_pt - 990.0) / 4.325425917886686   # +0.4%  sum_pt > 990
        + 0.003650820878930979 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7) / 0.00019681587893808614   # +0.4%  log_sum_pt < 6.7 and z_7 < 0.049
        + 0.0034031837987341038 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 37.0 - Q.pt_6) / 0.42495059593780576   # +0.3%  log_sum_pt < 6.7 and pt_6 < 37
        - 0.003207377362606743 * max(0.0, Q.centroid_offset - 0.0078) * max(0.0, 3.5 - Q.mass_top3) / 0.00927892321493425   # -0.3%  centroid_offset > 0.0078 and mass_top3 < 3.5
        + 0.002907787588894639 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409376186582   # +0.3%  centroid_offset > 0.05
        + 0.002505372555502545 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7) / 7.24802757490873e-05   # +0.3%  log_sum_pt < 6.3 and z_7 < 0.071
        - 0.0024462764649538986 * max(0.0, Q.sum_pt_top5 - 840.0) / 8.601484444754464   # -0.2%  sum_pt_top5 > 840
        - 0.0022004447371784734 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.0098) / 0.0008448793669394381   # -0.2%  log_sum_pt < 6.7 and mean_phi > 0.0098
        + 0.002036975927141754 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.0) / 0.23999550668673073   # +0.2%  log_sum_pt < 6.3 and pt_6 > 27
        - 0.0020251657433118554 * max(0.0, Q.girth2_top5 - 0.0082) * max(0.0, Q.mean_eta - 0.014) / 1.3540915578707908e-05   # -0.2%  girth2_top5 > 0.0082 and mean_eta > 0.014
        - 0.0020228008713037726 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, Q.pt_dispersion - 0.4) / 0.0029259753128448953   # -0.2%  log_sum_pt < 6.7 and pt_dispersion > 0.4
        - 0.0018715109419135344 * max(0.0, Q.centroid_offset - 0.0079) * max(0.0, 0.0012 - Q.mean_eta2) / 2.2229319013536214e-06   # -0.2%  centroid_offset > 0.0079 and mean_eta2 < 0.0012
        + 0.0017828720575330905 * max(0.0, 25.0 - Q.pt_6) / 0.6636199243369223   # +0.2%  pt_6 < 25
        + 0.0014045577522670623 * max(0.0, 0.0096 - Q.girth2_top2) * max(0.0, -0.0091 - Q.mean_phi) / 9.264132755133853e-06   # +0.1%  girth2_top2 < 0.0096 and mean_phi < -0.0091
        + 0.0007505363135127301 * max(0.0, 49.0 - Q.mass) * max(0.0, Q.dr_7 - 0.15) / 0.022561830895184373   # +0.1%  mass < 49 and dr_7 > 0.15
        - 0.0006255308416908625 * max(0.0, 0.0035 - Q.girth2) * max(0.0, Q.dr_7 - 0.13) / 2.483360723013625e-06   # -0.1%  girth2 < 0.0035 and dr_7 > 0.13
        + 0.0006139910907884047 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, Q.pt_5 - 59.0) / 0.0037769162497388506   # +0.1%  centroid_offset > 0.019 and pt_5 > 59
        + 0.0005170923465042382 * max(0.0, 0.014 - Q.width) * max(0.0, Q.mean_phi - 0.026) / 1.910119425536724e-06   # +0.1%  width < 0.014 and mean_phi > 0.026
        - 0.0004497475693631419 * max(0.0, Q.sum_pt - 970.0) * max(0.0, Q.pt_6 - 36.0) / 64.41804678216502   # -0.0%  sum_pt > 970 and pt_6 > 36
        + 0.0003665802444262327 * max(0.0, 0.014 - Q.width) * max(0.0, -0.026 - Q.mean_phi) / 1.9025214265828254e-06   # +0.0%  width < 0.014 and mean_phi < -0.026
        + 0.0003031770737477701 * max(0.0, Q.centroid_offset - 0.019) * max(0.0, 25.0 - Q.pt_5) / 0.0002785724295272494   # +0.0%  centroid_offset > 0.019 and pt_5 < 25
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 39.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.23619410081928 * (0.23549684702490198
        - 0.08894847861810014 * max(0.0, Q.girth2 - 0.0075) / 0.002475177143285355   # -8.9%  girth2 > 0.0075
        - 0.08300376676589333 * max(0.0, Q.girth2 - 0.0015) / 0.00532149003909432   # -8.3%  girth2 > 0.0015
        + 0.0828055111385491 * max(0.0, Q.girth2 - 0.0087) / 0.002210185787516779   # +8.3%  girth2 > 0.0087
        - 0.06715840328685536 * max(0.0, 0.0084 - Q.lam1) / 0.0043197379456789455   # -6.7%  lam1 < 0.0084
        - 0.059243200518010505 * max(0.0, 0.16 - Q.max_dr) / 0.05782282872334372   # -5.9%  max_dr < 0.16
        - 0.05840038967569392 * max(0.0, Q.mass_over_sum_pt - 0.091) / 0.008183603660282176   # -5.8%  mass_over_sum_pt > 0.091
        - 0.049883317308712864 * max(0.0, 0.088 - Q.girth) / 0.03699870548804189   # -5.0%  girth < 0.088
        + 0.04624616386771762 * max(0.0, Q.mass_over_sum_pt - 0.085) / 0.00950012283734065   # +4.6%  mass_over_sum_pt > 0.085
        - 0.043309438902201645 * max(0.0, 0.0055 - Q.width) / 0.0021757971205689595   # -4.3%  width < 0.0055
        + 0.04326735654632224 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 0.67 - Q.z_dr_0p05_0p1) / 0.029835613351157347   # +4.3%  max_dr < 0.16 and z_dr_0p05_0p1 < 0.67
        - 0.042219974129791456 * max(0.0, Q.girth2 - 0.0044) / 0.003624838292971698   # -4.2%  girth2 > 0.0044
        + 0.034203878232155545 * max(0.0, Q.mass_over_sum_pt - 0.072) / 0.013722188193431926   # +3.4%  mass_over_sum_pt > 0.072
        + 0.028364506916295013 * max(0.0, 0.039 - Q.centroid_offset) * max(0.0, Q.sum_pt - 560.0) / 4.902710568025472   # +2.8%  centroid_offset < 0.039 and sum_pt > 560
        - 0.023301749990174616 * max(0.0, 0.0056 - Q.width) * max(0.0, 0.96 - Q.n_dr_0p2_0p4) / 0.002054543787647764   # -2.3%  width < 0.0056 and n_dr_0p2_0p4 < 0.96
        + 0.019677802466053464 * max(0.0, 0.0053 - Q.width) * max(0.0, 3.1 - Q.n_dr_0p1_0p2) / 0.006031891226841048   # +2.0%  width < 0.0053 and n_dr_0p1_0p2 < 3.1
        + 0.017806734621319802 * max(0.0, 30.0 - Q.mass) / 7.488408316225965   # +1.8%  mass < 30
        + 0.016918420033468273 * max(0.0, Q.girth2 - 0.015) / 0.0011982209608526184   # +1.7%  girth2 > 0.015
        + 0.016801859811779375 * max(0.0, Q.girth2 - 0.004) * max(0.0, Q.eccentricity - 0.95) / 7.220602769219392e-05   # +1.7%  girth2 > 0.004 and eccentricity > 0.95
        + 0.016713620013111746 * max(0.0, 0.2 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.056) / 0.017440926568132717   # +1.7%  max_dr < 0.2 and z_dr_0p05_0p1 > 0.056
        - 0.015334644435804652 * max(0.0, 0.021 - Q.centroid_offset) / 0.00849820742302655   # -1.5%  centroid_offset < 0.021
        - 0.014756023723531578 * max(0.0, 47.0 - Q.pt_7) * max(0.0, 0.74 - Q.planar_flow) / 5.993480444852786   # -1.5%  pt_7 < 47 and planar_flow < 0.74
        + 0.01369952073088932 * max(0.0, 0.0011 - Q.girth2_top2) * max(0.0, 0.97 - Q.n_dr_0p2_0p4) / 0.00028591332685392077   # +1.4%  girth2_top2 < 0.0011 and n_dr_0p2_0p4 < 0.97
        - 0.012784274329884049 * max(0.0, Q.girth2 - 0.0075) * max(0.0, Q.log_sum_pt - 6.2) / 0.00034833768683711935   # -1.3%  girth2 > 0.0075 and log_sum_pt > 6.2
        - 0.011773570852697897 * max(0.0, 0.0081 - Q.lam1) * max(0.0, 1.1 - Q.D2) / 0.00036956008898896247   # -1.2%  lam1 < 0.0081 and D2 < 1.1
        + 0.010911801584495128 * max(0.0, 0.0012 - Q.e2_sq) / 0.0003425100519671026   # +1.1%  e2_sq < 0.0012
        + 0.009141379356384814 * max(0.0, 0.16 - Q.max_dr) * max(0.0, 1.2 - Q.D2) / 0.010366269791223612   # +0.9%  max_dr < 0.16 and D2 < 1.2
        - 0.008820265343023501 * max(0.0, 0.0011 - Q.girth2_top2) / 0.00031749875506385274   # -0.9%  girth2_top2 < 0.0011
        - 0.007643451358765596 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, 1.6 - Q.D2) / 0.06287210505507292   # -0.8%  planar_flow < 0.2 and D2 < 1.6
        - 0.006711553951048967 * max(0.0, 0.13 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.27) / 0.007588928920503683   # -0.7%  mass_over_sum_pt < 0.13 and z_dr_0p05_0p1 > 0.27
        - 0.00615245205139014 * max(0.0, 0.001 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.3) / 0.00011124368796512778   # -0.6%  girth2_top2 < 0.001 and log_sum_pt > 6.3
        + 0.006069839707768266 * max(0.0, 0.2 - Q.planar_flow) * max(0.0, Q.sum_pt - 620.0) / 10.221348022955189   # +0.6%  planar_flow < 0.2 and sum_pt > 620
        - 0.005856720514094331 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -0.6%  mass > 80.4
        + 0.005434174209583277 * max(0.0, 0.038 - Q.e2) * max(0.0, 1.1 - Q.D2) / 0.0006560501971226943   # +0.5%  e2 < 0.038 and D2 < 1.1
        - 0.0049613301636212016 * max(0.0, 0.00052 - Q.width) / 8.46363970861265e-05   # -0.5%  width < 0.00052
        - 0.0033220634491726605 * max(0.0, 0.19 - Q.planar_flow) * max(0.0, 2.9 - Q.n_dr_0p1_0p2) / 0.111406090860663   # -0.3%  planar_flow < 0.19 and n_dr_0p1_0p2 < 2.9
        + 0.0031794773471987407 * max(0.0, 0.025 - Q.e2) * max(0.0, 0.46 - Q.tau21) / 0.0003690845867865319   # +0.3%  e2 < 0.025 and tau21 < 0.46
        + 0.0028346179119537697 * max(0.0, 0.022 - Q.centroid_offset) * max(0.0, Q.C2 - 0.025) / 4.673093218280554e-05   # +0.3%  centroid_offset < 0.022 and C2 > 0.025
        - 0.002082934507257903 * max(0.0, Q.z_dr_0p05_0p1 - 0.66) / 0.03948136358747124   # -0.2%  z_dr_0p05_0p1 > 0.66
        - 0.0020463980258857766 * max(0.0, 0.0087 - Q.lam1) * max(0.0, 4.1 - Q.n_pt_above_50) / 0.0010189450526800474   # -0.2%  lam1 < 0.0087 and n_pt_above_50 < 4.1
        - 0.0014846660988154397 * max(0.0, 0.037 - Q.centroid_offset) * max(0.0, Q.C2 - 0.068) / 2.9569871689354688e-05   # -0.1%  centroid_offset < 0.037 and C2 > 0.068
        - 0.0012203304078021331 * max(0.0, 0.025 - Q.e2) * max(0.0, 1.1 - Q.D2) / 9.406899950423664e-05   # -0.1%  e2 < 0.025 and D2 < 1.1
        - 0.0011612921957311422 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.92) / 0.04339493904711782   # -0.1%  mass > 80.4 and eccentricity > 0.92
        - 0.001022216694614213 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_2 - 64.0) / 0.030384767153773612   # -0.1%  centroid_offset > 0.031 and pt_2 > 64
        + 0.0009333019270239972 * max(0.0, 0.006 - Q.width) * max(0.0, -0.0044 - Q.mean_phi) / 5.0232120114378916e-06   # +0.1%  width < 0.006 and mean_phi < -0.0044
        - 0.0008689388182744737 * max(0.0, Q.mass - 80.4) * max(0.0, 5.4 - Q.m012) / 0.6120978839417197   # -0.1%  mass > 80.4 and m012 < 5.4
        - 0.0006471049182405667 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 380.0) / 0.0023293517592368504   # -0.1%  centroid_offset > 0.031 and pt_0 > 380
        + 0.0005099954917224142 * max(0.0, 0.005 - Q.girth2_top3) * max(0.0, Q.n_dr_0p2_0p4 - 0.88) / 0.00013612436805281235   # +0.1%  girth2_top3 < 0.005 and n_dr_0p2_0p4 > 0.88
        - 0.0002647506745728943 * max(0.0, 0.025 - Q.e2) * max(0.0, Q.phi_0 - 0.055) / 2.733633909438137e-07   # -0.0%  e2 < 0.025 and phi_0 > 0.055
        + 9.633637654516189e-05 * max(0.0, 0.0059 - Q.girth2_top3) * max(0.0, Q.max_pair_mass - 46.0) / 1.4823030467041515e-05   # +0.0%  girth2_top3 < 0.0059 and max_pair_mass > 46
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 15.01;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.012905016870658 * (-0.029374727909383894
        + 0.3073870216275824 * max(0.0, 0.0049 - Q.width) / 0.0018385546450652038   # +30.7%  width < 0.0049
        - 0.19338912814047796 * max(0.0, 0.063 - Q.girth) * max(0.0, 0.0055 - Q.width) / 9.275823041752154e-05   # -19.3%  girth < 0.063 and width < 0.0055
        + 0.07003966464718585 * max(0.0, 0.0067 - Q.girth2) * max(0.0, 0.025 - Q.centroid_offset) / 4.274385499031198e-05   # +7.0%  girth2 < 0.0067 and centroid_offset < 0.025
        + 0.06433346076655506 * max(0.0, 0.18 - Q.max_dr) * max(0.0, 0.00021 - Q.lam2) / 1.1256784800639465e-05   # +6.4%  max_dr < 0.18 and lam2 < 0.00021
        - 0.05438302028447447 * max(0.0, 0.027 - Q.C2) / 0.008855174816283791   # -5.4%  C2 < 0.027
        - 0.0504200184842869 * max(0.0, 29.0 - Q.mass) * max(0.0, 0.024 - Q.centroid_offset) / 0.10146795555676436   # -5.0%  mass < 29 and centroid_offset < 0.024
        - 0.04296467117704498 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0071) / 1.0661562437389368e-05   # -4.3%  width < 0.0051 and centroid_offset > 0.0071
        + 0.042627286020544455 * max(0.0, 0.2 - Q.LHA) * max(0.0, 0.00067 - Q.width) / 1.2474842030280943e-05   # +4.3%  LHA < 0.2 and width < 0.00067
        - 0.04008609312239907 * max(0.0, Q.log_sum_pt - 6.7) / 0.0356099827540834   # -4.0%  log_sum_pt > 6.7
        - 0.02786652376441782 * max(0.0, Q.z_dr_0_0p05 - 0.87) * max(0.0, 0.00054 - Q.lam2) / 2.2372057455913032e-05   # -2.8%  z_dr_0_0p05 > 0.87 and lam2 < 0.00054
        - 0.02119011881470037 * max(0.0, 0.016 - Q.dr_0) / 0.002092929217508556   # -2.1%  dr_0 < 0.016
        + 0.017271900850034817 * max(0.0, 0.057 - Q.girth) * max(0.0, Q.log_sum_pt - 6.7) / 0.001184024689143289   # +1.7%  girth < 0.057 and log_sum_pt > 6.7
        - 0.015351237291523307 * max(0.0, 0.0048 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00045) / 5.733001675350303e-07   # -1.5%  width < 0.0048 and mass_over_sum_pt_sq > 0.00045
        + 0.011749931866360419 * max(0.0, 0.0065 - Q.girth2) * max(0.0, 0.38 - Q.planar_flow) / 0.00033728606322059413   # +1.2%  girth2 < 0.0065 and planar_flow < 0.38
        - 0.010965958247260014 * max(0.0, Q.pt_7 - 34.0) / 4.744406039915966   # -1.1%  pt_7 > 34
        + 0.008522340003253668 * max(0.0, Q.log_sum_pt - 6.7) * max(0.0, 50.0 - Q.pt_7) / 0.8254521354214482   # +0.9%  log_sum_pt > 6.7 and pt_7 < 50
        - 0.008381772322154877 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.031) / 0.004494098276626663   # -0.8%  mass < 22 and centroid_offset > 0.031
        + 0.006801129411578173 * max(0.0, 0.0034 - Q.centroid_offset) / 0.0002358076440264862   # +0.7%  centroid_offset < 0.0034
        - 0.0034222956878250865 * max(0.0, 0.2 - Q.LHA) * max(0.0, Q.mean_phi - -0.00075) / 5.633618432123252e-05   # -0.3%  LHA < 0.2 and mean_phi > -0.00075
        + 0.0017681673335110855 * max(0.0, 22.0 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0) / 0.00043163135336480305   # +0.2%  mass < 22 and max_pair_mass > 13
        + 0.0009812261665269652 * max(0.0, 0.00022 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.029) / 6.268534143888272e-07   # +0.1%  girth2_top5 < 0.00022 and n_dr_0p2_0p4 > 0.029
        + 9.70339703023746e-05 * max(0.0, 0.19 - Q.LHA) * max(0.0, Q.girth2 - 0.0075) / 8.184054941344932e-09   # +0.0%  LHA < 0.19 and girth2 > 0.0075
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 27.54;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.539678176335507 * (-0.06608646580205511
        + 0.28014131741508524 * max(0.0, 0.0061 - Q.width) / 0.002546205189936007   # +28.0%  width < 0.0061
        - 0.12409342018609745 * max(0.0, 0.076 - Q.mass_over_sum_pt) / 0.026909392564770866   # -12.4%  mass_over_sum_pt < 0.076
        + 0.06079580224348135 * max(0.0, 0.0033 - Q.mass_over_sum_pt_sq) / 0.0011959263058982947   # +6.1%  mass_over_sum_pt_sq < 0.0033
        + 0.05583054493200892 * max(0.0, 55.0 - Q.mass) * max(0.0, 0.026 - Q.centroid_offset) / 0.2973994661193356   # +5.6%  mass < 55 and centroid_offset < 0.026
        - 0.05196289275287906 * max(0.0, 0.055 - Q.girth) / 0.015504239908187712   # -5.2%  girth < 0.055
        - 0.0499073740407734 * max(0.0, 43.0 - Q.mass) * max(0.0, 0.027 - Q.centroid_offset) / 0.21576656510343808   # -5.0%  mass < 43 and centroid_offset < 0.027
        + 0.0497711345871682 * max(0.0, 0.23 - Q.max_dr) / 0.11327942388443746   # +5.0%  max_dr < 0.23
        - 0.03590738825188547 * max(0.0, 0.0059 - Q.width) * max(0.0, Q.centroid_offset - 0.0029) / 2.0558792445107196e-05   # -3.6%  width < 0.0059 and centroid_offset > 0.0029
        + 0.035490318473923484 * max(0.0, 0.018 - Q.centroid_offset) / 0.00647279436521528   # +3.5%  centroid_offset < 0.018
        + 0.02947460226803569 * max(0.0, 52.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt) / 4.272216109669429   # +2.9%  mass < 52 and log_sum_pt < 6.8
        + 0.024751796705863187 * max(0.0, 0.017 - Q.e2) * max(0.0, 0.024 - Q.centroid_offset) / 5.9794431189960836e-05   # +2.5%  e2 < 0.017 and centroid_offset < 0.024
        - 0.022815316526972173 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.043) / 0.00024259709445715918   # -2.3%  centroid_offset < 0.018 and z_4 > 0.043
        + 0.016121865321323484 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.056 - Q.dr01) / 0.0005394787151293713   # +1.6%  e2 < 0.032 and dr01 < 0.056
        - 0.014833784103364998 * max(0.0, 0.072 - Q.mass_over_sum_pt) * max(0.0, 0.00078 - Q.girth2_top2) / 1.1443071158092817e-05   # -1.5%  mass_over_sum_pt < 0.072 and girth2_top2 < 0.00078
        - 0.014443382888124712 * max(0.0, Q.log_sum_pt - 6.4) / 0.19309034782356593   # -1.4%  log_sum_pt > 6.4
        - 0.013132154703393214 * max(0.0, 0.0046 - Q.girth2) * max(0.0, Q.centroid_offset - 0.011) / 6.007563360353817e-06   # -1.3%  girth2 < 0.0046 and centroid_offset > 0.011
        + 0.010785851278040498 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.pt_4 - 49.0) / 0.09644119255034167   # +1.1%  centroid_offset < 0.018 and pt_4 > 49
        + 0.010668638561376474 * max(0.0, 51.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow) / 1.3416021578079962   # +1.1%  mass < 51 and planar_flow < 0.32
        + 0.009997082161130265 * max(0.0, Q.girth2 - 0.018) / 0.0008445289123310189   # +1.0%  girth2 > 0.018
        - 0.009266601700682749 * max(0.0, 0.018 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.033) / 0.00021266602385423852   # -0.9%  centroid_offset < 0.018 and z_5 > 0.033
        - 0.009012077357303664 * max(0.0, 0.00073 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023) / 3.5917468903094515e-06   # -0.9%  girth2_top2 < 0.00073 and z_7 > 0.023
        - 0.008587882297462268 * max(0.0, 0.019 - Q.e2) * max(0.0, 53.0 - Q.pt_7) / 0.10327839069360685   # -0.9%  e2 < 0.019 and pt_7 < 53
        - 0.007294417357768313 * max(0.0, Q.sum_pt - 850.0) / 22.05114231798188   # -0.7%  sum_pt > 850
        - 0.007050218478313414 * max(0.0, 0.032 - Q.e2) * max(0.0, 0.45 - Q.tau21) / 0.0007326820677947367   # -0.7%  e2 < 0.032 and tau21 < 0.45
        - 0.006930669229563543 * max(0.0, 0.00042 - Q.mean_phi) / 0.005597313786768654   # -0.7%  mean_phi < 0.00042
        + 0.006815288626857321 * max(0.0, Q.n_dr_0p2_0p4 - 0.9) / 0.17219344537843131   # +0.7%  n_dr_0p2_0p4 > 0.9
        - 0.006112962325942506 * max(0.0, 33.0 - Q.mass) * max(0.0, 0.32 - Q.planar_flow) / 0.5025343736134924   # -0.6%  mass < 33 and planar_flow < 0.32
        + 0.005156567754136083 * max(0.0, Q.C2 - 0.051) / 0.00483027947086316   # +0.5%  C2 > 0.051
        + 0.00391213741761622 * max(0.0, 0.00075 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.0) / 0.0009706216708355942   # +0.4%  girth2_top2 < 0.00075 and pt_7 > 33
        + 0.0035768041201525895 * max(0.0, Q.lam2 - 0.0017) / 0.0002619788148106202   # +0.4%  lam2 > 0.0017
        - 0.00295699689123604 * max(0.0, Q.C2 - 0.051) * max(0.0, 85.0 - Q.pt_2) / 0.07906285704181079   # -0.3%  C2 > 0.051 and pt_2 < 85
        - 0.0027483503987455666 * max(0.0, Q.C2 - 0.048) * max(0.0, 0.19 - Q.dr_5) / 0.00032908124129241856   # -0.3%  C2 > 0.048 and dr_5 < 0.19
        - 0.0021753229241934994 * max(0.0, Q.girth2 - 0.018) * max(0.0, 0.36 - Q.planar_flow) / 0.00010311134812718426   # -0.2%  girth2 > 0.018 and planar_flow < 0.36
        - 0.0019133434339451217 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, Q.mean_phi - 0.026) / 2.4855123779081747e-05   # -0.2%  log_sum_pt > 6.4 and mean_phi > 0.026
        - 0.001791806998853237 * max(0.0, Q.girth2 - 0.019) * max(0.0, 0.025 - Q.mean_eta) / 2.1738232644283576e-05   # -0.2%  girth2 > 0.019 and mean_eta < 0.025
        - 0.001650987107958767 * max(0.0, 0.006 - Q.width) * max(0.0, Q.C2 - 0.031) / 4.947513996350743e-06   # -0.2%  width < 0.006 and C2 > 0.031
        + 0.0011107883670657423 * max(0.0, Q.lam2 - 0.001) * max(0.0, Q.mass_top2 - 16.0) / 0.0028324772362044236   # +0.1%  lam2 > 0.001 and mass_top2 > 16
        + 0.001012109811276047 * max(0.0, Q.n_dr_0p2_0p4 - 0.87) * max(0.0, Q.dr_6 - 0.22) / 0.0028587875365798975   # +0.1%  n_dr_0p2_0p4 > 0.87 and dr_6 > 0.22
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 12.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.436054718051297 * (0.18655434159777795
        - 0.2033483231822408 * max(0.0, 0.0034 - Q.lam2) / 0.00302856392086027   # -20.3%  lam2 < 0.0034
        + 0.1505407745442083 * max(0.0, 0.1 - Q.mass_over_sum_pt) / 0.04533010434696364   # +15.1%  mass_over_sum_pt < 0.1
        + 0.08938850693408981 * max(0.0, Q.girth2 - 0.0085) / 0.0022502841364316776   # +8.9%  girth2 > 0.0085
        - 0.06688085850123698 * max(0.0, 0.037 - Q.e2) / 0.0146690302629917   # -6.7%  e2 < 0.037
        - 0.052239329446098125 * max(0.0, 0.0017 - Q.girth2) / 0.00044496654755204427   # -5.2%  girth2 < 0.0017
        - 0.050685204268825286 * max(0.0, Q.lam1 - 0.0085) / 0.0018164956013911194   # -5.1%  lam1 > 0.0085
        + 0.04988682623075647 * max(0.0, Q.mass - 17.0) / 26.1770169078314   # +5.0%  mass > 17
        + 0.04978338303505981 * max(0.0, 0.0038 - Q.girth2_top2) / 0.0017739509325894283   # +5.0%  girth2_top2 < 0.0038
        - 0.03535025913818545 * max(0.0, 0.0044 - Q.lam1) / 0.0016222057451651924   # -3.5%  lam1 < 0.0044
        + 0.024562073963452575 * max(0.0, 3.0 - Q.n_dr_0p05_0p1) / 1.6691546218487394   # +2.5%  n_dr_0p05_0p1 < 3
        + 0.02449953002975067 * max(0.0, Q.e2 - 0.036) / 0.0066378539437150375   # +2.4%  e2 > 0.036
        + 0.024175281253073208 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 0.044 - Q.z_dr_0p2_0p4) / 0.002017752486493276   # +2.4%  eccentricity > 0.89 and z_dr_0p2_0p4 < 0.044
        - 0.023594802736572912 * max(0.0, Q.log_sum_pt - 6.7) / 0.0356099827540834   # -2.4%  log_sum_pt > 6.7
        - 0.019878927589588022 * max(0.0, Q.LHA - 0.3) * max(0.0, 0.57 - Q.tau21) / 0.0060740892196632975   # -2.0%  LHA > 0.3 and tau21 < 0.57
        + 0.01926542384058771 * max(0.0, 0.0016 - Q.girth2_top3) / 0.000517464071377967   # +1.9%  girth2_top3 < 0.0016
        - 0.018243284495463568 * max(0.0, Q.eccentricity - 0.89) * max(0.0, 37.0 - Q.mass_top2) / 1.5646516153280123   # -1.8%  eccentricity > 0.89 and mass_top2 < 37
        - 0.01586414095211253 * max(0.0, Q.LHA - 0.33) / 0.011605136760902931   # -1.6%  LHA > 0.33
        - 0.012520087208008342 * max(0.0, 6.3 - Q.log_sum_pt) / 0.028104781515084085   # -1.3%  log_sum_pt < 6.3
        - 0.011842655316313502 * max(0.0, Q.mass - 8.0) * max(0.0, 1.9 - Q.n_dr_0p2_0p4) / 44.36021371105295   # -1.2%  mass > 8 and n_dr_0p2_0p4 < 1.9
        + 0.008888227676233344 * max(0.0, 0.27 - Q.tau32) / 0.01110899354051616   # +0.9%  tau32 < 0.27
        + 0.008296416168377832 * max(0.0, Q.C2 - 0.055) / 0.004263416753457529   # +0.8%  C2 > 0.055
        + 0.00697620423344141 * max(0.0, 0.0044 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.8) / 4.1911332160086226e-05   # +0.7%  lam1 < 0.0044 and log_sum_pt > 6.8
        - 0.006693859491037133 * max(0.0, Q.z_7 - 0.062) / 0.004624733494749155   # -0.7%  z_7 > 0.062
        - 0.005838382579373702 * max(0.0, Q.lam2 - 0.00048) * max(0.0, 8.1 - Q.n_pt_above_50) / 0.0014667968731718955   # -0.6%  lam2 > 0.00048 and n_pt_above_50 < 8.1
        + 0.005257836557120923 * max(0.0, Q.lam2 - 0.00023) * max(0.0, 0.51 - Q.tau21) / 4.772754972476368e-05   # +0.5%  lam2 > 0.00023 and tau21 < 0.51
        + 0.004196120639976979 * max(0.0, Q.n_dr_0p2_0p4 - 1.9) / 0.069025378151188   # +0.4%  n_dr_0p2_0p4 > 1.9
        - 0.00388080250806632 * max(0.0, Q.lam1 - 0.0083) * max(0.0, 2.8 - Q.min_pair_mass) / 0.0029608510638198444   # -0.4%  lam1 > 0.0083 and min_pair_mass < 2.8
        + 0.0026817579910803904 * max(0.0, Q.pt_7 - 46.0) / 1.0075676470588235   # +0.3%  pt_7 > 46
        - 0.0023970741359371174 * max(0.0, 0.27 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.008444800316640091   # -0.2%  tau32 < 0.27 and n_dr_0p2_0p4 < 2
        + 0.0023436453537305666 * max(0.0, Q.centroid_offset - 0.038) / 0.001665468677640001   # +0.2%  centroid_offset > 0.038
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 23.28;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.28157115364624 * (-0.01507630209677787
        + 0.244084302522546 * max(0.0, 0.0085 - Q.width) / 0.0043050500429293705   # +24.4%  width < 0.0085
        + 0.10257235021971466 * max(0.0, 0.04 - Q.centroid_offset) / 0.024293443235371397   # +10.3%  centroid_offset < 0.04
        - 0.10052157757386553 * max(0.0, 0.089 - Q.girth) / 0.03780775865529435   # -10.1%  girth < 0.089
        + 0.051563224781571104 * max(0.0, 690.0 - Q.sum_pt_top5) / 132.9427338497899   # +5.2%  sum_pt_top5 < 690
        + 0.050767405677240175 * max(0.0, 0.27 - Q.planar_flow) / 0.11963005744541409   # +5.1%  planar_flow < 0.27
        - 0.04764201266441808 * max(0.0, 0.0062 - Q.e2_sq) / 0.0028223432767164503   # -4.8%  e2_sq < 0.0062
        - 0.04696180112947796 * max(0.0, Q.girth - 0.076) / 0.010614995286401219   # -4.7%  girth > 0.076
        - 0.04284630400854558 * max(0.0, 0.05 - Q.centroid_offset) * max(0.0, 6.8 - Q.log_sum_pt) / 0.0075002201161331735   # -4.3%  centroid_offset < 0.05 and log_sum_pt < 6.8
        - 0.03231588630084757 * max(0.0, Q.girth - 0.089) * max(0.0, 7.1 - Q.n_pt_above_50) / 0.020224855008234528   # -3.2%  girth > 0.089 and n_pt_above_50 < 7.1
        + 0.03053756038420546 * max(0.0, Q.girth - 0.077) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.025947532297213433   # +3.1%  girth > 0.077 and n_pt_above_50 < 7
        + 0.0286010561582151 * max(0.0, 0.22 - Q.max_dr) / 0.10469772390203094   # +2.9%  max_dr < 0.22
        - 0.026466510589156007 * max(0.0, Q.centroid_offset - 0.014) / 0.0072834745800256185   # -2.6%  centroid_offset > 0.014
        - 0.02192364341748608 * max(0.0, 0.0036 - Q.width) / 0.0012009808568738123   # -2.2%  width < 0.0036
        - 0.016832675498697994 * max(0.0, 0.035 - Q.C2) / 0.01446092739222048   # -1.7%  C2 < 0.035
        - 0.016136016338170286 * max(0.0, 0.27 - Q.planar_flow) * max(0.0, 71.0 - Q.mass) / 2.5555905613163934   # -1.6%  planar_flow < 0.27 and mass < 71
        - 0.012760212774145416 * max(0.0, 0.21 - Q.planar_flow) * max(0.0, Q.width - 0.0056) / 0.00022677694781445204   # -1.3%  planar_flow < 0.21 and width > 0.0056
        - 0.012669469765857614 * max(0.0, Q.girth - 0.075) * max(0.0, 40.0 - Q.pt_7) / 0.06875644797966989   # -1.3%  girth > 0.075 and pt_7 < 40
        - 0.012391075714908236 * max(0.0, 2.8 - Q.n_dr_0p1_0p2) / 1.8030231932928378   # -1.2%  n_dr_0p1_0p2 < 2.8
        - 0.012358559521181083 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.12 - Q.dr_0) / 0.01034988067801977   # -1.2%  log_sum_pt < 6.7 and dr_0 < 0.12
        + 0.012306659058891807 * max(0.0, 0.23 - Q.planar_flow) * max(0.0, Q.girth2 - 0.015) / 7.722866807095813e-05   # +1.2%  planar_flow < 0.23 and girth2 > 0.015
        - 0.012073408322375514 * max(0.0, 0.002 - Q.girth2_top3) / 0.0007044809897854893   # -1.2%  girth2_top3 < 0.002
        - 0.009009418668037013 * max(0.0, 0.021 - Q.girth) / 0.0027064957648115155   # -0.9%  girth < 0.021
        - 0.007882815209330516 * max(0.0, 0.013 - Q.girth2) * max(0.0, -0.0014 - Q.mean_phi) / 2.7068484245881003e-05   # -0.8%  girth2 < 0.013 and mean_phi < -0.0014
        + 0.007517150395128452 * max(0.0, 0.0063 - Q.e2_sq) * max(0.0, -1.8e-05 - Q.mean_phi) / 1.1745709516566644e-05   # +0.8%  e2_sq < 0.0063 and mean_phi < -1.8e-05
        - 0.007439648950115423 * max(0.0, 0.26 - Q.planar_flow) * max(0.0, Q.max_dr - 0.11) / 0.005232831310884041   # -0.7%  planar_flow < 0.26 and max_dr > 0.11
        - 0.007260606950811349 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.mean_phi - 0.0032) / 2.4713207214031123e-05   # -0.7%  girth2 < 0.014 and mean_phi > 0.0032
        - 0.006792491811890365 * max(0.0, 30.0 - Q.pt_7) * max(0.0, 1.9 - Q.n_dr_0p2_0p4) / 3.8290528191061743   # -0.7%  pt_7 < 30 and n_dr_0p2_0p4 < 1.9
        + 0.006307723305335622 * max(0.0, 0.0062 - Q.e2_sq) * max(0.0, Q.mean_phi - 0.0014) / 9.725411188787015e-06   # +0.6%  e2_sq < 0.0062 and mean_phi > 0.0014
        + 0.003936896055685637 * max(0.0, Q.centroid_offset - 0.015) * max(0.0, 0.11 - Q.tau21) / 5.554977311815418e-05   # +0.4%  centroid_offset > 0.015 and tau21 < 0.11
        + 0.003795966639500388 * max(0.0, 0.16 - Q.LHA) * max(0.0, 0.028 - Q.z_7) / 6.312576243885407e-05   # +0.4%  LHA < 0.16 and z_7 < 0.028
        + 0.0031838734630989423 * max(0.0, 5.5 - Q.mass_top5) / 0.719665791993632   # +0.3%  mass_top5 < 5.5
        + 0.0015871052618963797 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top5 - 49.0) / 0.005278614869023845   # +0.2%  girth2 < 0.013 and mass_top5 > 49
        + 0.000954590867652757 * max(0.0, Q.m01 - 46.0) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.02387150935325302   # +0.1%  m01 > 46 and n_pt_above_50 > 6
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5843;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5843029526552805 * (-2.361788510102146
        - 0.36775500828061436 * max(0.0, Q.e2 - 0.063) / 0.0018057171192615976   # -36.8%  e2 > 0.063
        + 0.2953106319366727 * max(0.0, Q.girth2 - 0.019) / 0.000743753768065063   # +29.5%  girth2 > 0.019
        + 0.17138698676756056 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.pt_7 - 15.0) / 0.014576699041481386   # +17.1%  girth2 > 0.019 and pt_7 > 15
        + 0.09926363363269747 * max(0.0, Q.mass - 91.2) / 0.5835013503307559   # +9.9%  mass > 91.2
        + 0.06628373938245483 * max(0.0, Q.girth2 - 0.015) * max(0.0, Q.lam2 - 5.6e-06) / 4.5834064655859715e-06   # +6.6%  girth2 > 0.015 and lam2 > 5.6e-06
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.982808124516318 * (0.048107473982502445
        + 0.3498493606650852 * max(0.0, 0.15 - Q.girth) / 0.09211510952961005   # +35.0%  girth < 0.15
        + 0.10923542707253431 * max(0.0, 0.015 - Q.lam1) / 0.009920450793435537   # +10.9%  lam1 < 0.015
        - 0.06814139863633896 * max(0.0, 0.0076 - Q.width) / 0.003605085041673278   # -6.8%  width < 0.0076
        - 0.06778002034260354 * max(0.0, 0.049 - Q.e2) / 0.023931290647697473   # -6.8%  e2 < 0.049
        + 0.06454879916137794 * max(0.0, 0.0062 - Q.lam1) / 0.00268024418294773   # +6.5%  lam1 < 0.0062
        - 0.04454240491816137 * max(0.0, 0.14 - Q.girth) * max(0.0, 6.8 - Q.log_sum_pt) / 0.017674184956300153   # -4.5%  girth < 0.14 and log_sum_pt < 6.8
        + 0.04183075730304181 * max(0.0, 0.038 - Q.centroid_offset) / 0.022481130537366098   # +4.2%  centroid_offset < 0.038
        - 0.03691881249852332 * max(0.0, 0.017 - Q.lam1) * max(0.0, 0.037 - Q.centroid_offset) / 0.00028242572470604343   # -3.7%  lam1 < 0.017 and centroid_offset < 0.037
        - 0.03316951725799032 * max(0.0, 0.15 - Q.girth) * max(0.0, 39.0 - Q.pt_7) / 0.7094603868706324   # -3.3%  girth < 0.15 and pt_7 < 39
        + 0.03248600774827654 * max(0.0, Q.sum_pt_top5 - 640.0) * max(0.0, 45.0 - Q.pt_7) / 1105.6185096603826   # +3.2%  sum_pt_top5 > 640 and pt_7 < 45
        - 0.026418760556131404 * max(0.0, 0.00031 - Q.lam2) * max(0.0, 0.041 - Q.centroid_offset) / 5.81925734646329e-06   # -2.6%  lam2 < 0.00031 and centroid_offset < 0.041
        - 0.025446097659382255 * max(0.0, 0.53 - Q.tau21) * max(0.0, Q.max_dr - 0.013) / 0.031089654249423927   # -2.5%  tau21 < 0.53 and max_dr > 0.013
        - 0.01646547965649028 * max(0.0, 0.5 - Q.tau21) / 0.2086791654360457   # -1.6%  tau21 < 0.5
        - 0.010913089222499364 * max(0.0, 0.027 - Q.z_7) / 0.0011583431269464603   # -1.1%  z_7 < 0.027
        - 0.00995726571991829 * max(0.0, 25.0 - Q.pt_7) / 1.2165635479582457   # -1.0%  pt_7 < 25
        - 0.009482359348356664 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, Q.z_7 - 0.023) / 0.288080660788649   # -0.9%  sum_pt_top5 > 660 and z_7 > 0.023
        - 0.009225513638521324 * max(0.0, Q.C2 - 0.066) / 0.0029842881518688767   # -0.9%  C2 > 0.066
        + 0.0076902256992975844 * max(0.0, Q.sum_pt_top5 - 830.0) * max(0.0, Q.n_pt_above_50 - 0.88) / 34.5506951019564   # +0.8%  sum_pt_top5 > 830 and n_pt_above_50 > 0.88
        + 0.006586793287903166 * max(0.0, 0.053 - Q.e2) * max(0.0, Q.pt_dispersion - 0.4) / 0.0017209576394509453   # +0.7%  e2 < 0.053 and pt_dispersion > 0.4
        - 0.00634788696673159 * max(0.0, Q.sum_pt - 980.0) * max(0.0, 4.2 - Q.D2) / 9.540260739125708   # -0.6%  sum_pt > 980 and D2 < 4.2
        - 0.005324489700009207 * max(0.0, Q.sum_pt - 1000.0) / 3.8808921431853993   # -0.5%  sum_pt > 1000
        + 0.004052982832592967 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, 4.3 - Q.D2) / 6.682624250280097   # +0.4%  sum_pt_top5 > 900 and D2 < 4.3
        + 0.0037589609203294606 * max(0.0, Q.sum_pt - 990.0) * max(0.0, Q.n_pt_above_50 - 6.0) / 2.304610543592437   # +0.4%  sum_pt > 990 and n_pt_above_50 > 6
        - 0.003236483816812873 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.m012 - 16.0) / 0.00458038197158627   # -0.3%  width < 0.0074 and m012 > 16
        + 0.003168698100606972 * max(0.0, 0.007 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.18) / 0.0001942721727333123   # +0.3%  lam1 < 0.007 and z_dr_0p05_0p1 > 0.18
        + 0.0019932473758638575 * max(0.0, Q.sum_pt_top5 - 900.0) * max(0.0, Q.n_pt_above_50 - 6.1) / 0.9099714443277298   # +0.2%  sum_pt_top5 > 900 and n_pt_above_50 > 6.1
        + 0.0006593680890667426 * max(0.0, 520.0 - Q.sum_pt_top5) * max(0.0, 30.0 - Q.pt_5) / 7.51538371815401   # +0.1%  sum_pt_top5 < 520 and pt_5 < 30
        - 0.00048802351646158034 * max(0.0, Q.sum_pt_top5 - 650.0) * max(0.0, 0.37 - Q.tau32) / 0.20929317526057448   # -0.0%  sum_pt_top5 > 650 and tau32 < 0.37
        + 0.00028176828909126254 * max(0.0, 760.0 - Q.sum_pt) * max(0.0, 0.037 - Q.z_4) / 0.0006431742996250209   # +0.0%  sum_pt < 760 and z_4 < 0.037
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 37.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.05380155177853 * (0.0008528139806611354
        + 0.10557437963679309 * max(0.0, 0.013 - Q.girth2) / 0.008032714809063365   # +10.6%  girth2 < 0.013
        + 0.09436270951658644 * max(0.0, Q.lam1 - 0.0042) / 0.0032374973262182566   # +9.4%  lam1 > 0.0042
        - 0.08703884502186467 * max(0.0, 0.087 - Q.girth) / 0.036196634014996405   # -8.7%  girth < 0.087
        - 0.08314688779010557 * max(0.0, 0.0061 - Q.width) / 0.002546205189936007   # -8.3%  width < 0.0061
        - 0.0611449122115579 * max(0.0, Q.lam1 - 0.0025) / 0.004180168713999974   # -6.1%  lam1 > 0.0025
        + 0.06111441645307522 * max(0.0, 0.043 - Q.e2) / 0.019029592094159647   # +6.1%  e2 < 0.043
        - 0.05069061480266354 * max(0.0, 0.0081 - Q.mass_over_sum_pt_sq) / 0.00423990966464005   # -5.1%  mass_over_sum_pt_sq < 0.0081
        + 0.04830878651923619 * max(0.0, 0.0058 - Q.e2_sq) / 0.002557177412701445   # +4.8%  e2_sq < 0.0058
        + 0.0429868102936827 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 0.4) / 0.002095822023639565   # +4.3%  lam1 > 0.0025 and D2 > 0.4
        - 0.035798555009447404 * max(0.0, Q.lam1 - 0.0042) * max(0.0, Q.D2 - 0.41) / 0.0015478092802339452   # -3.6%  lam1 > 0.0042 and D2 > 0.41
        - 0.03134706015696791 * max(0.0, Q.lam1 - 0.0061) / 0.002429974364619147   # -3.1%  lam1 > 0.0061
        - 0.02701080390361172 * max(0.0, 0.067 - Q.C2) / 0.04135756064456336   # -2.7%  C2 < 0.067
        - 0.024444341317938528 * max(0.0, 0.18 - Q.max_dr) / 0.07246046178070677   # -2.4%  max_dr < 0.18
        - 0.024045726610245066 * max(0.0, 0.013 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.019496402231604832   # -2.4%  girth2 < 0.013 and n_dr_0p1_0p2 < 3
        + 0.02298056514945147 * max(0.0, 0.0077 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.009787555179258533   # +2.3%  width < 0.0077 and n_dr_0p1_0p2 < 3
        - 0.01900242929681824 * max(0.0, 0.58 - Q.z_dr_0p05_0p1) * max(0.0, 0.068 - Q.C2) / 0.015894181583882745   # -1.9%  z_dr_0p05_0p1 < 0.58 and C2 < 0.068
        + 0.018120632308691348 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.eccentricity - 0.97) / 5.9419319783981916e-05   # +1.8%  girth2 < 0.013 and eccentricity > 0.97
        - 0.016977407938822145 * max(0.0, 0.0043 - Q.girth2) * max(0.0, 1.1 - Q.n_dr_0p2_0p4) / 0.0016382226683039193   # -1.7%  girth2 < 0.0043 and n_dr_0p2_0p4 < 1.1
        - 0.016723920307239393 * max(0.0, 0.034 - Q.girth) / 0.006634741158803078   # -1.7%  girth < 0.034
        + 0.014197759688380933 * max(0.0, 0.6 - Q.z_dr_0p05_0p1) / 0.3812180941834117   # +1.4%  z_dr_0p05_0p1 < 0.6
        - 0.011525335868379968 * max(0.0, 0.0076 - Q.width) * max(0.0, 1.0 - Q.D2) / 0.0002190038502997674   # -1.2%  width < 0.0076 and D2 < 1
        - 0.010721181028912634 * max(0.0, 0.0075 - Q.width) * max(0.0, 0.11 - Q.planar_flow) / 6.092952672484992e-05   # -1.1%  width < 0.0075 and planar_flow < 0.11
        + 0.010094759070616159 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0094) / 0.00027914119351912494   # +1.0%  planar_flow < 0.11 and centroid_offset > 0.0094
        - 0.009887060140393575 * max(0.0, 4.8 - Q.n_dr_0p05_0p1) / 3.002894789939697   # -1.0%  n_dr_0p05_0p1 < 4.8
        + 0.007004267309655197 * max(0.0, 0.6 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.8187215486043322   # +0.7%  z_dr_0p05_0p1 < 0.6 and n_dr_0p1_0p2 < 3
        - 0.006829423694121171 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018) / 0.0001515306049550779   # -0.7%  planar_flow < 0.11 and centroid_offset > 0.018
        - 0.006403016899725769 * max(0.0, Q.centroid_offset - 0.05) / 0.0008015409376186582   # -0.6%  centroid_offset > 0.05
        - 0.006148372488297575 * max(0.0, 0.8 - Q.D2) / 0.11334356917800653   # -0.6%  D2 < 0.8
        + 0.005695238767753769 * max(0.0, 0.041 - Q.e2) * max(0.0, 0.99 - Q.D2) / 0.000694178444376131   # +0.6%  e2 < 0.041 and D2 < 0.99
        - 0.005651594186824669 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 0.16 - Q.max_dr) / 0.0010366982646029031   # -0.6%  planar_flow < 0.11 and max_dr < 0.16
        - 0.005559875442244629 * max(0.0, Q.centroid_offset - 0.031) / 0.0025308909249329155   # -0.6%  centroid_offset > 0.031
        + 0.004933544007109825 * max(0.0, 1.2 - Q.D2) * max(0.0, 0.03 - Q.centroid_offset) / 0.00440497736352804   # +0.5%  D2 < 1.2 and centroid_offset < 0.03
        + 0.004844631367745596 * max(0.0, 0.14 - Q.tau21) / 0.018132526191107747   # +0.5%  tau21 < 0.14
        + 0.003378298197242389 * max(0.0, Q.lam1 - 0.0054) * max(0.0, 0.16 - Q.max_dr) / 1.3135235150404054e-05   # +0.3%  lam1 > 0.0054 and max_dr < 0.16
        - 0.0031644841327786886 * max(0.0, Q.lam1 - 0.0025) * max(0.0, Q.D2 - 1.6) / 0.00023311365222611032   # -0.3%  lam1 > 0.0025 and D2 > 1.6
        - 0.0026566217004029722 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 750.0 - Q.sum_pt) / 2.0723775428395825   # -0.3%  planar_flow < 0.11 and sum_pt < 750
        + 0.002384873005288163 * max(0.0, 0.013 - Q.girth2) * max(0.0, Q.mass_top3 - 24.0) / 0.015341772754191192   # +0.2%  girth2 < 0.013 and mass_top3 > 24
        - 0.0023560720331216585 * max(0.0, 310.0 - Q.sum_pt_top3) / 11.323142095588235   # -0.2%  sum_pt_top3 < 310
        + 0.0023482527954809904 * max(0.0, 0.078 - Q.max_dr) / 0.014501948846193631   # +0.2%  max_dr < 0.078
        + 0.001618831795611641 * max(0.0, Q.z_dr_0p05_0p1 - 0.74) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.0221342701477133   # +0.2%  z_dr_0p05_0p1 > 0.74 and n_dr_0p2_0p4 < 1
        - 0.0012073451607949178 * max(0.0, 0.0074 - Q.width) * max(0.0, Q.mass_top3 - 24.0) / 0.0024182015131132446   # -0.1%  width < 0.0074 and mass_top3 > 24
        + 0.00029906226138867633 * max(0.0, 0.006 - Q.width) * max(0.0, -0.026 - Q.mean_phi) / 3.0953613645592545e-07   # +0.0%  width < 0.006 and mean_phi < -0.026
        + 0.0002702947129299023 * max(0.0, 0.006 - Q.width) * max(0.0, -0.027 - Q.mean_eta) / 2.8292222184744477e-07   # +0.0%  width < 0.006 and mean_eta < -0.027
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 22.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.577797282539994 * (-0.08238181859478326
        - 0.18165098428012597 * max(0.0, 0.0067 - Q.width) / 0.0029505605030579448   # -18.2%  width < 0.0067
        + 0.12505733106462177 * max(0.0, 0.0067 - Q.lam1) / 0.0030262798172267114   # +12.5%  lam1 < 0.0067
        - 0.12280817554260308 * max(0.0, 0.0083 - Q.lam1) / 0.0042396606911918626   # -12.3%  lam1 < 0.0083
        + 0.1120705059573125 * max(0.0, 0.013 - Q.width) / 0.008032714809066324   # +11.2%  width < 0.013
        + 0.07766042591501676 * max(0.0, 0.024 - Q.e2) / 0.007186071119610508   # +7.8%  e2 < 0.024
        + 0.049768449876920834 * max(0.0, 0.041 - Q.e2) / 0.017502522934382727   # +5.0%  e2 < 0.041
        + 0.04874011168909856 * max(0.0, Q.girth - 0.032) / 0.03265413534851113   # +4.9%  girth > 0.032
        + 0.03609432227633416 * max(0.0, 0.32 - Q.z_dr_0p1_0p2) / 0.22144844875155983   # +3.6%  z_dr_0p1_0p2 < 0.32
        - 0.034256778769429126 * max(0.0, 0.0038 - Q.girth2_top2) / 0.0017739509325894283   # -3.4%  girth2_top2 < 0.0038
        + 0.03219326141312072 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.22 - Q.z_dr_0p2_0p4) / 0.010428306026531659   # +3.2%  tau21 < 0.23 and z_dr_0p2_0p4 < 0.22
        - 0.029349162494744615 * max(0.0, 0.00031 - Q.lam2) / 0.00020202421988374075   # -2.9%  lam2 < 0.00031
        - 0.026625348601931983 * max(0.0, 0.068 - Q.mass_over_sum_pt) / 0.022182351413704057   # -2.7%  mass_over_sum_pt < 0.068
        + 0.017330966066823213 * max(0.0, 0.0074 - Q.girth2_top2) / 0.004352558827222625   # +1.7%  girth2_top2 < 0.0074
        - 0.015549599030695686 * max(0.0, Q.LHA - 0.34) / 0.009917392506774791   # -1.6%  LHA > 0.34
        + 0.013395960486543722 * max(0.0, Q.LHA - 0.18) * max(0.0, Q.sum_pt_top3 - 350.0) / 6.122495552026316   # +1.3%  LHA > 0.18 and sum_pt_top3 > 350
        - 0.012503664915766785 * max(0.0, 0.0076 - Q.width) * max(0.0, Q.e2 - 0.024) / 4.2515845144124996e-06   # -1.3%  width < 0.0076 and e2 > 0.024
        - 0.008903411418946733 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.log_sum_pt - 6.9) / 1.896409605094633e-05   # -0.9%  width < 0.0061 and log_sum_pt > 6.9
        + 0.008285397057011009 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # +0.8%  log_sum_pt > 6.9
        - 0.007780261754533343 * max(0.0, 0.27 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.0084) / 0.0001514320454301317   # -0.8%  tau21 < 0.27 and girth2_top5 > 0.0084
        - 0.0058964764138102095 * max(0.0, Q.z_dr_0p05_0p1 - 0.75) / 0.022953353302118156   # -0.6%  z_dr_0p05_0p1 > 0.75
        - 0.005785072656438201 * max(0.0, Q.LHA - 0.31) * max(0.0, 0.45 - Q.pt_dispersion) / 0.0008013140963302259   # -0.6%  LHA > 0.31 and pt_dispersion < 0.45
        - 0.005351366613852143 * max(0.0, 0.0082 - Q.lam1) * max(0.0, 0.76 - Q.D2) / 0.00014180994200951427   # -0.5%  lam1 < 0.0082 and D2 < 0.76
        - 0.0044852408573910554 * max(0.0, 0.23 - Q.tau21) * max(0.0, 0.63 - Q.z_dr_0p05_0p1) / 0.01269008256159663   # -0.4%  tau21 < 0.23 and z_dr_0p05_0p1 < 0.63
        + 0.004444082849265401 * max(0.0, 0.0065 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.3) / 0.0030405333841674836   # +0.4%  lam1 < 0.0065 and pt1_dr01 > 1.3
        + 0.0041310016170627495 * max(0.0, 1.8 - Q.n_dr_0_0p05) * max(0.0, 0.43 - Q.pt_dispersion) / 0.01605316989395657   # +0.4%  n_dr_0_0p05 < 1.8 and pt_dispersion < 0.43
        - 0.004023891637942148 * max(0.0, 0.0086 - Q.width) * max(0.0, 0.064 - Q.planar_flow) / 3.4543958056412786e-05   # -0.4%  width < 0.0086 and planar_flow < 0.064
        - 0.0029009746475108012 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.m01 - 17.0) / 0.0026733721433994605   # -0.3%  lam1 < 0.0083 and m01 > 17
        - 0.0026525063301142387 * max(0.0, Q.mass - 80.4) * max(0.0, 0.22 - Q.z_dr_0p2_0p4) / 0.0900567672360502   # -0.3%  mass > 80.4 and z_dr_0p2_0p4 < 0.22
        + 0.0003052677650322492 * max(0.0, Q.log_sum_pt - 6.9) * max(0.0, Q.mean_phi - 0.017) / 3.850432243459315e-07   # +0.0%  log_sum_pt > 6.9 and mean_phi > 0.017
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0606915966386554, 0.8437668067226891, 1.9565463235294118, 1.3222529411764705, 6.206889495798319, 1.9824300420168066, 1.4768453781512605, 1.3126792016806723, 1.1379882352941177, 4.367423949579832, 1.665191281512605, 3.0917281512605044, 0.07805189075630252, 5.1222955882352945, 0.5098701680672268, 0.5313573529411765]
T = [2.813884853433561, 2.280827217371324, 4.489019004398634, 3.225095671940651, 4.328460865283613]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +30%, n9 +27%, n5 -13%, n1 +12%, n4 -7%, n0 -6% ...
            + 0.2987696875249533 * h[2] / H_AVG[2]
            + 0.26676677633701806 * h[9] / H_AVG[9]
            - 0.1320969592713036 * h[5] / H_AVG[5]
            + 0.11713215928998305 * h[1] / H_AVG[1]
            - 0.0689314975014052 * h[4] / H_AVG[4]
            - 0.05889830984823667 * h[0] / H_AVG[0]
            + 0.05740461022710005 * h[6] / H_AVG[6]
        ),
        0.03125 + T[1] * (   # class q: n9 +49%, n4 -26%, n10 -9%, n6 +8%, n5 +4%, n8 +3% ...
            + 0.48619037371714696 * h[9] / H_AVG[9]
            - 0.2551249326556762 * h[4] / H_AVG[4]
            - 0.09126027109978516 * h[10] / H_AVG[10]
            + 0.08093803461433059 * h[6] / H_AVG[6]
            + 0.04074241464315583 * h[5] / H_AVG[5]
            + 0.03118353909677287 * h[8] / H_AVG[8]
            + 0.014560434173132235 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +26%, n3 -15%, n6 -10%, n14 -9%, n15 -8%, n0 +8% ...
            + 0.2582742589386757 * h[11] / H_AVG[11]
            - 0.14727638041639396 * h[3] / H_AVG[3]
            - 0.10280958494941704 * h[6] / H_AVG[6]
            - 0.08518623460397852 * h[14] / H_AVG[14]
            - 0.08137817634300634 * h[15] / H_AVG[15]
            + 0.08122325523399798 * h[0] / H_AVG[0]
            + 0.08023165154678215 * h[13] / H_AVG[13]
            + 0.06396688788492098 * h[7] / H_AVG[7]
            - 0.06337622062743789 * h[8] / H_AVG[8]
            - 0.030403524309127624 * h[9] / H_AVG[9]
            - 0.005873825146261851 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n3 -23%, n7 +19%, n6 -17%, n4 +15%, n13 +9%, n14 +6% ...
            - 0.23061867152741372 * h[3] / H_AVG[3]
            + 0.1907907356489542 * h[7] / H_AVG[7]
            - 0.17172111253167008 * h[6] / H_AVG[6]
            + 0.15035623472448947 * h[4] / H_AVG[4]
            + 0.0868580558458461 * h[13] / H_AVG[13]
            + 0.05928547009898706 * h[14] / H_AVG[14]
            - 0.0423187440955027 * h[9] / H_AVG[9]
            + 0.03270316963244402 * h[1] / H_AVG[1]
            - 0.025743294104233524 * h[15] / H_AVG[15]
            + 0.009604511790459102 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -48%, n4 +18%, n10 +14%, n5 -11%, n8 +5%, n3 +2% ...
            - 0.4807557807466141 * h[13] / H_AVG[13]
            + 0.17924643681027094 * h[4] / H_AVG[4]
            + 0.14426530584476274 * h[10] / H_AVG[10]
            - 0.11449970923364881 * h[5] / H_AVG[5]
            + 0.04929530397952813 * h[8] / H_AVG[8]
            + 0.01909242370338828 * h[3] / H_AVG[3]
            - 0.009016125267796354 * h[12] / H_AVG[12]
            + 0.003828914413990678 * h[0] / H_AVG[0]
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
