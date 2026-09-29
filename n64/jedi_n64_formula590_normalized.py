"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned), as if-statements with NORMALIZED weights (how much each one matters).

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.
  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1
             avg_k = the term's average size on the entire training set (term / avg is 1 on average), so share_k is the
             fraction of the neuron's average input that comes from that if-statement (sign: pushes it up / down).
  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1
             h_j = neuron j (after max(0, .) and the network's rounding), avg_j = its average on the training jets.

How much each neuron matters (share of all class scores, averaged over the training jets):
  neuron  5:  15.1%   (on for 83% of jets)
  neuron  1:  13.3%   (on for 99% of jets)
  neuron  8:  12.8%   (on for 52% of jets)
  neuron  4:   9.1%   (on for 66% of jets)
  neuron  9:   7.5%   (on for 53% of jets)
  neuron 13:   7.0%   (on for 86% of jets)
  neuron 10:   6.5%   (on for 78% of jets)
  neuron  0:   5.7%   (on for 58% of jets)
  neuron  3:   4.3%   (on for 89% of jets)
  neuron  6:   3.9%   (on for 52% of jets)
  neuron  7:   3.7%   (on for 28% of jets)
  neuron 11:   2.6%   (on for 90% of jets)
  neuron  2:   2.6%   (on for 74% of jets)
  neuron 14:   2.2%   (on for 40% of jets)
  neuron 15:   2.0%   (on for 57% of jets)
  neuron 12:   1.7%   (on for 41% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.9% of jets.

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
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_2                    pT of particle 2 / total pT
  Q.z_6                    pT of particle 6 / total pT
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.z_1st                  largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.phi_6                  Δφ of particle 6
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
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
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.pt_dispersion          √(Σ pTᵢ²) / Σ pTᵢ
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[0.0, 0.0, 0.75, -1.375, 0.125], [0.625, -0.1875, 0.0, 0.0, 0.0], [0.0, 0.125, 0.0, -0.375, 0.0], [-0.75, 0.0, 0.4375, 0.4375, 0.0], [-0.875, -1.0625, 0.34375, 0.0, 0.1875], [-0.3125, 0.0, 0.578125, 0.6875, -0.46875], [0.15625, 0.21875, 0.0625, -0.96875, 0.0], [0.0, 0.0, -0.625, 0.90625, -0.28125], [0.03125, 0.015625, -0.875, -0.9375, 0.2109375], [0.515625, 0.5625, -0.21875, 0.0, 0.0], [-0.015625, -0.015625, 0.0, 0.0, 0.984375], [0.0, -0.25, 0.59375, 0.0, 0.0], [0.234375, 0.34375, -0.40625, 0.3125, -0.375], [0.0, 0.0, 0.0, 0.0, -0.90625], [0.0, 0.0, -1.375, 0.0, 0.0], [0.0, 0.0, -0.1875, 0.5625, -0.375]]
B = [-1.078125, 1.359375, 0.09375, 0.984375, 0.78125]
INT_BITS = [2, 4, 3, 2, 2, 2, 3, 3, 4, 3, 3, 3, 3, 2, 2, 3]
FRAC_BITS = [5, 5, 3, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 5, 4]


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
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_2=pt[2],
        pt_6=pt[6],
        pt_9=pt[9],
        z_2=z[2],
        z_6=z[6],
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        z_1st=zs[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        phi_6=phi[6],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
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
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 15.92;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.916838125885295 * (0.0694092942906103
        - 0.12694726859948102 * max(0.0, Q.mass - 92.85979309082) / 14.482733665646967   # -12.7%  mass > 92.86
        - 0.09592075979354266 * max(0.0, Q.mass - 92.85979309082) * max(0.0, 0.390781164169 - Q.max_dr) / 0.34297144815280617   # -9.6%  mass > 92.86 and max_dr < 0.3908
        + 0.07316230397922774 * max(0.0, 0.008184367501 - Q.mass_over_sum_pt_sq) / 0.0024516083597474207   # +7.3%  mass_over_sum_pt_sq < 0.008184
        - 0.07270843136356722 * max(0.0, Q.mass - 78.261818313599) / 21.336576218229133   # -7.3%  mass > 78.26
        - 0.058366061194749214 * max(0.0, Q.mass - 74.251806640625) / 24.076479363593222   # -5.8%  mass > 74.25
        + 0.056746396445590964 * max(0.0, Q.mass_over_sum_pt - 0.083299446175) / 0.01700043986417834   # +5.7%  mass_over_sum_pt > 0.0833
        - 0.04509535985668713 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # -4.5%  mass > 80.4
        - 0.0441785466796555 * max(0.0, 0.056600876898 - Q.girth) / 0.010803815022120036   # -4.4%  girth < 0.0566
        - 0.04078939819992617 * max(0.0, 0.009614971338 - Q.width) / 0.003502544629791685   # -4.1%  width < 0.009615
        - 0.02964138974293577 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) / 0.020241059768600602   # -3.0%  mass_over_sum_pt > 0.07697
        + 0.028820253876656387 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # +2.9%  log_sum_pt < 7.017
        - 0.02617620303013482 * max(0.0, Q.mass - 92.85979309082) * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 1.7278724386184863   # -2.6%  mass > 92.86 and n_dr_0p2_0p4 < 7
        - 0.023510308082222733 * max(0.0, 0.006043208873 - Q.girth2_top20) / 0.0017975224219569485   # -2.4%  girth2_top20 < 0.006043
        + 0.02138170020186286 * max(0.0, 0.008840538245 - Q.girth2_top40) / 0.0031086216346541064   # +2.1%  girth2_top40 < 0.008841
        + 0.020893493467039215 * max(0.0, 0.006043208873 - Q.girth2_top20) * max(0.0, 68.434224049685 - Q.mass_top5) / 0.10844296886679751   # +2.1%  girth2_top20 < 0.006043 and mass_top5 < 68.43
        - 0.019927684531161914 * max(0.0, 0.00625977218 - Q.girth2_top40) / 0.0014618229210262657   # -2.0%  girth2_top40 < 0.00626
        + 0.018044771050312793 * max(0.0, 0.00742997247 - Q.girth2_top50) / 0.0020221555597961793   # +1.8%  girth2_top50 < 0.00743
        - 0.01657828640152594 * max(0.0, 0.007538018543 - Q.girth2_top20) / 0.0027408407225167327   # -1.7%  girth2_top20 < 0.007538
        - 0.015837514603980325 * max(0.0, Q.mass - 91.034691238403) / 15.06939693349851   # -1.6%  mass > 91.03
        - 0.015517759594592498 * max(0.0, 89.172933810464 - Q.mass_top30) / 20.17610316369032   # -1.6%  mass_top30 < 89.17
        - 0.01474009641855683 * max(0.0, 82.04491364955 - Q.mass_top50) / 11.943128521443713   # -1.5%  mass_top50 < 82.04
        + 0.014005959756604612 * max(0.0, 7.017257672702 - Q.log_sum_pt) * max(0.0, 0.390781164169 - Q.max_dr) / 0.003927553188539109   # +1.4%  log_sum_pt < 7.017 and max_dr < 0.3908
        - 0.013336265060705989 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # -1.3%  sum_pt < 1013
        + 0.012397332962619103 * max(0.0, 62.0 - Q.n_particles) / 16.64866050420168   # +1.2%  n_particles < 62
        + 0.01236226079891293 * max(0.0, 0.260146178237 - Q.LHA) / 0.039673711668022284   # +1.2%  LHA < 0.2601
        - 0.012008745159669073 * max(0.0, 0.006716736591 - Q.lam1) / 0.0019121054678470912   # -1.2%  lam1 < 0.006717
        - 0.011493243349130627 * max(0.0, 6.989450376716 - Q.log_sum_pt) * max(0.0, 0.387360095978 - Q.max_dr) / 0.0026132212469729453   # -1.1%  log_sum_pt < 6.989 and max_dr < 0.3874
        + 0.010482728387712543 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583193277311   # +1.0%  n_dr_0p2_0p4 < 10
        - 0.010220795719693705 * max(0.0, Q.z_top30_slots - 0.920388080863) / 0.04411613538673992   # -1.0%  z_top30_slots > 0.9204
        + 0.009130496257916435 * max(0.0, 0.00625977218 - Q.girth2_top40) * max(0.0, 0.002915531053 - Q.girth2_top3) / 3.6946461207137186e-06   # +0.9%  girth2_top40 < 0.00626 and girth2_top3 < 0.002916
        - 0.008717256635801711 * max(0.0, 858.826171875 - Q.sum_pt_top40) * max(0.0, 33.0 - Q.n_dr_0p1_0p2) / 60.82995928883272   # -0.9%  sum_pt_top40 < 858.8 and n_dr_0p1_0p2 < 33
        + 0.00816270847327903 * max(0.0, 70.421206773231 - Q.mass_top20) / 14.595357970697426   # +0.8%  mass_top20 < 70.42
        + 0.004120358659143166 * max(0.0, 82.04491364955 - Q.mass_top50) * max(0.0, Q.dr_1 - 0.161154452503) / 4.4732868840813177e-05   # +0.4%  mass_top50 < 82.04 and dr_1 > 0.1612
        + 0.003607945175837886 * max(0.0, 82.04491364955 - Q.mass_top50) * max(0.0, 0.212648361921 - Q.z_dr_0p05_0p1) / 1.9465026796824894   # +0.4%  mass_top50 < 82.04 and z_dr_0p05_0p1 < 0.2126
        - 0.0024941269988449245 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.088715460151) / 3.100400728225136   # -0.2%  sum_pt < 1013 and z_dr_0p1_0p2 > 0.08872
        - 0.0024757894907185196 * max(0.0, 1002.378515625 - Q.sum_pt) / 15.957354754845854   # -0.2%  sum_pt < 1002
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 21.26;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.261366853068544 * (0.061889557010111534
        + 0.07081795779027182 * max(0.0, Q.log_sum_pt - 6.910130970417) / 0.051750869608046556   # +7.1%  log_sum_pt > 6.91
        - 0.06495441147491533 * max(0.0, Q.z_top50_slots - 0.958653609576) / 0.03427111698918734   # -6.5%  z_top50_slots > 0.9587
        + 0.06472130827258898 * max(0.0, Q.log_sum_pt - 6.89371369877) / 0.06417423647849095   # +6.5%  log_sum_pt > 6.894
        - 0.06070252128256545 * max(0.0, Q.log_sum_pt - 6.811175180312) / 0.138314075182229   # -6.1%  log_sum_pt > 6.811
        - 0.0584392911398465 * max(0.0, Q.sum_pt_top50 - 959.095727539062) / 86.53899855624287   # -5.8%  sum_pt_top50 > 959.1
        + 0.05058115800120336 * max(0.0, Q.sum_pt - 949.91689453125) / 100.79265446346679   # +5.1%  sum_pt > 949.9
        + 0.05050788738438743 * max(0.0, 0.404204003833 - Q.LHA) / 0.14530548997384415   # +5.1%  LHA < 0.4042
        - 0.04848789317217458 * max(0.0, 31.0 - Q.n_pt_above_10) / 11.388505882352941   # -4.8%  n_pt_above_10 < 31
        + 0.04051942124739248 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.27338014705884   # +4.1%  sum_pt_top2 < 689.2
        + 0.034778099207107005 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, 0.016859196762 - Q.girth2_top5) / 0.0004650153521672225   # +3.5%  z_top30_slots > 0.9342 and girth2_top5 < 0.01686
        - 0.033606124375054815 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # -3.4%  mass < 120.6
        + 0.027193753907772022 * max(0.0, 80.4 - Q.mass_top30) / 14.655773101172624   # +2.7%  mass_top30 < 80.4
        - 0.02585678597225769 * max(0.0, 80.4 - Q.mass_top30) * max(0.0, 68.434224049685 - Q.mass_top5) / 891.7019121450287   # -2.6%  mass_top30 < 80.4 and mass_top5 < 68.43
        + 0.0245924478488943 * max(0.0, 1073.4734375 - Q.sum_pt_top30) / 97.85231293739916   # +2.5%  sum_pt_top30 < 1073
        - 0.023891355756397142 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, 91.19 - Q.mass_top10) / 1.6455577605891802   # -2.4%  z_top30_slots > 0.9342 and mass_top10 < 91.19
        + 0.020926745197683966 * max(0.0, 0.43572281599 - Q.max_dr) * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 0.0143775071962318   # +2.1%  max_dr < 0.4357 and z_dr_0p2_0p4 < 0.1937
        + 0.019508655255139715 * max(0.0, Q.n_particles - 26.0) / 20.249211764705883   # +2.0%  n_particles > 26
        - 0.018421846119590548 * max(0.0, Q.log_sum_pt - 6.959293500649) / 0.028460039078747168   # -1.8%  log_sum_pt > 6.959
        - 0.017372146321252585 * max(0.0, 0.43572281599 - Q.max_dr) / 0.09036112201580485   # -1.7%  max_dr < 0.4357
        + 0.01631478028652401 * max(0.0, 902.40625 - Q.sum_pt_top5) / 313.0793622488839   # +1.6%  sum_pt_top5 < 902.4
        + 0.015431870266927135 * max(0.0, Q.z_top50_slots - 0.958653609576) * max(0.0, 0.001155056594 - Q.girth2_top3) / 1.2668949446071925e-05   # +1.5%  z_top50_slots > 0.9587 and girth2_top3 < 0.001155
        + 0.01525265088265299 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.022605352903 - Q.girth2_top2) / 0.17051930939624896   # +1.5%  n_particles > 38 and girth2_top2 < 0.02261
        - 0.014445472956865853 * max(0.0, 0.957678701144 - Q.z_top20_slots) / 0.07565544239103991   # -1.4%  z_top20_slots < 0.9577
        + 0.013747785605880275 * max(0.0, 0.007678543663 - Q.girth2_top10) / 0.0034735798352083574   # +1.4%  girth2_top10 < 0.007679
        - 0.011904561958544271 * max(0.0, Q.n_pt_above_10 - 16.0) / 4.845095798319328   # -1.2%  n_pt_above_10 > 16
        + 0.011802047273471438 * max(0.0, 31.0 - Q.n_pt_above_10) * max(0.0, 0.004673423215 - Q.lam1) / 0.014237020923050172   # +1.2%  n_pt_above_10 < 31 and lam1 < 0.004673
        + 0.009847797633021555 * max(0.0, 0.003270031267 - Q.girth2_top15) * max(0.0, Q.n_real_top40 - 29.0) / 0.00524279632090315   # +1.0%  girth2_top15 < 0.00327 and n_real_top40 > 29
        + 0.00924835442815495 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, Q.mass_top5 - 22.1834155076) / 0.4186571122099573   # +0.9%  z_top30_slots > 0.9342 and mass_top5 > 22.18
        - 0.008854779953561969 * max(0.0, 0.003270031267 - Q.girth2_top15) / 0.0007969964936058968   # -0.9%  girth2_top15 < 0.00327
        + 0.008786054843941554 * max(0.0, 31.0 - Q.n_pt_above_10) * max(0.0, 0.017422899418 - Q.mean_phi2) / 0.1577871548170423   # +0.9%  n_pt_above_10 < 31 and mean_phi2 < 0.01742
        - 0.008347194910394626 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # -0.8%  n_dr_0p2_0p4 < 13
        - 0.00828489972879873 * max(0.0, 0.000823693417 - Q.girth2_top3) / 0.00020583156908661208   # -0.8%  girth2_top3 < 0.0008237
        + 0.008153021597852791 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.985099030959 - Q.z_top50_slots) / 0.09243197309849287   # +0.8%  n_particles > 38 and z_top50_slots < 0.9851
        - 0.007277323491524197 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, 0.001155056594 - Q.girth2_top3) / 1.546960382817174e-05   # -0.7%  z_top30_slots > 0.9342 and girth2_top3 < 0.001155
        - 0.007263666834713041 * max(0.0, Q.log_sum_pt - 6.989450376716) / 0.021142744446604637   # -0.7%  log_sum_pt > 6.989
        - 0.007091529019707584 * max(0.0, 0.967499609196 - Q.z_top40_slots) / 0.00873000890587478   # -0.7%  z_top40_slots < 0.9675
        - 0.006821526742058712 * max(0.0, 0.404204003833 - Q.LHA) * max(0.0, 0.001776308492 - Q.lam2) / 0.0001685717875281262   # -0.7%  LHA < 0.4042 and lam2 < 0.001776
        - 0.006107268030734334 * max(0.0, 34.0625 - Q.pt_9) * max(0.0, 28.393960910299 - Q.pt1_dr01) / 202.39411627265795   # -0.6%  pt_9 < 34.06 and pt1_dr01 < 28.39
        + 0.005713380554663091 * max(0.0, 0.000657050184 - Q.girth2_top5) / 0.0001395474066313899   # +0.6%  girth2_top5 < 0.0006571
        + 0.005362478566533792 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, Q.mass_top5 - 2.164373545539) / 8316.480273169645   # +0.5%  sum_pt_top2 < 689.2 and mass_top5 > 2.164
        + 0.0050561506983725895 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.199975347593 - Q.dr_4) / 1.307262581268358   # +0.5%  n_particles > 38 and dr_4 < 0.2
        - 0.004999247006317446 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 621.4389484506303   # -0.5%  sum_pt_top2 < 689.2 and n_dr_0p2_0p4 < 7
        + 0.004461437942025646 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, 1.976207274199 - Q.D2) / 0.01673775751856122   # +0.4%  z_top30_slots > 0.9342 and D2 < 1.976
        + 0.004383995859593215 * max(0.0, 40.2 - Q.mass_top20) * max(0.0, 0.005860335776 - Q.mean_phi2) / 0.01972420536905745   # +0.4%  mass_top20 < 40.2 and mean_phi2 < 0.00586
        + 0.004375050522733206 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.000823693417 - Q.girth2_top3) / 0.0013072722554786523   # +0.4%  n_dr_0p2_0p4 < 13 and girth2_top3 < 0.0008237
        + 0.003750668246161515 * max(0.0, 0.000412964127 - Q.girth2_top10) / 5.029872303525816e-05   # +0.4%  girth2_top10 < 0.000413
        - 0.00343674736668166 * max(0.0, Q.log_sum_pt - 7.062574317998) / 0.010878839937255731   # -0.3%  log_sum_pt > 7.063
        - 0.003231433515260728 * max(0.0, Q.z_dr_0_0p05 - 0.878906026483) / 0.015835254093555608   # -0.3%  z_dr_0_0p05 > 0.8789
        + 0.0027214758682114 * max(0.0, Q.sum_pt_top40 - 1032.405224609375) / 32.74345563439171   # +0.3%  sum_pt_top40 > 1032
        - 0.0016455376816200476 * max(0.0, Q.log_sum_pt - 7.139296169016) / 0.005452502053630527   # -0.2%  log_sum_pt > 7.139
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 8.657;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.656627436666966 * (-0.05278455307646211
        + 0.12179468169265183 * max(0.0, 0.015638355144 - Q.girth2_top15) / 0.009593305133886523   # +12.2%  girth2_top15 < 0.01564
        - 0.121093823966893 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # -12.1%  mass < 92.86
        - 0.10132802111800004 * max(0.0, 0.140939019879 - Q.mass_over_sum_pt) / 0.057960363993862694   # -10.1%  mass_over_sum_pt < 0.1409
        + 0.08280492189820064 * max(0.0, Q.sum_pt - 1017.43466796875) / 47.92777614496459   # +8.3%  sum_pt > 1017
        + 0.06863801662911206 * max(0.0, 125.1 - Q.mass) / 42.4577541956062   # +6.9%  mass < 125.1
        + 0.060506390556881284 * max(0.0, 91.19 - Q.mass) / 16.464230590854466   # +6.1%  mass < 91.19
        + 0.052039847356954896 * max(0.0, 0.402178311348 - Q.max_dr) / 0.059113559416633016   # +5.2%  max_dr < 0.4022
        - 0.04060858274749029 * max(0.0, Q.sum_pt - 1066.481811523438) / 29.84706589314436   # -4.1%  sum_pt > 1066
        - 0.034795960317304364 * max(0.0, 0.006189818106 - Q.lam1) / 0.0016087780966108156   # -3.5%  lam1 < 0.00619
        + 0.034139963063664756 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, 0.01976735495 - Q.girth2_top10) / 0.000859816573689493   # +3.4%  log_sum_pt > 6.903 and girth2_top10 < 0.01977
        + 0.030131622427032552 * max(0.0, 91.697531419407 - Q.mass_top30) / 22.012698823640722   # +3.0%  mass_top30 < 91.7
        - 0.028813823926183695 * max(0.0, Q.sum_pt_top50 - 997.018872070312) / 56.586436834715556   # -2.9%  sum_pt_top50 > 997
        - 0.027056958425339443 * max(0.0, 0.074356165682 - Q.mass_over_sum_pt) / 0.009452716665803248   # -2.7%  mass_over_sum_pt < 0.07436
        + 0.026613132038718464 * max(0.0, Q.sum_pt_top40 - 1069.67119140625) / 23.00895715528997   # +2.7%  sum_pt_top40 > 1070
        - 0.025390331531774304 * max(0.0, Q.log_sum_pt - 6.903422848462) / 0.056622750952925925   # -2.5%  log_sum_pt > 6.903
        + 0.020922679766594374 * max(0.0, 0.006142801866 - Q.girth2_top15) / 0.0020806505579742157   # +2.1%  girth2_top15 < 0.006143
        + 0.018796616800581305 * max(0.0, Q.sum_pt_top15 - 1003.329150390625) / 12.917826203040363   # +1.9%  sum_pt_top15 > 1003
        + 0.01866966299858938 * max(0.0, Q.sum_pt - 995.676940917969) / 62.241609981350926   # +1.9%  sum_pt > 995.7
        - 0.01657105203768727 * max(0.0, Q.sum_pt_top20 - 908.8125) / 62.53618744132419   # -1.7%  sum_pt_top20 > 908.8
        + 0.015846553185842948 * max(0.0, Q.sum_pt_top50 - 1038.855053710938) / 34.95068931446463   # +1.6%  sum_pt_top50 > 1039
        + 0.014733542827752447 * max(0.0, 0.09795414517 - Q.mass_over_sum_pt) / 0.023392533955001828   # +1.5%  mass_over_sum_pt < 0.09795
        - 0.013089696769985061 * max(0.0, Q.mass_top40 - 160.8) / 0.77818694636032   # -1.3%  mass_top40 > 160.8
        + 0.008148470278215395 * max(0.0, 86.4 - Q.mass_top50) / 14.253133905303311   # +0.8%  mass_top50 < 86.4
        + 0.007514683408885337 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, 42.78125 - Q.pt_6) / 0.23780334480065718   # +0.8%  log_sum_pt > 6.903 and pt_6 < 42.78
        - 0.004359978918551908 * max(0.0, Q.sum_pt_top50 - 1107.225842285156) / 19.65455559402422   # -0.4%  sum_pt_top50 > 1107
        - 0.00347393794674955 * max(0.0, Q.sum_pt - 1115.722741699219) / 20.458001746869385   # -0.3%  sum_pt > 1116
        + 0.0015239690439204493 * max(0.0, Q.mass_top20 - 125.1) / 1.3144709680417963   # +0.2%  mass_top20 > 125.1
        + 0.0005930783204428985 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168090372162   # +0.1%  sum_pt_top20 > 1129
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.219;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.2188919876586422 * (0.16472137195532813
        + 0.2638790300058271 * max(0.0, 0.00750911433 - Q.mass_over_sum_pt_sq) / 0.002014788498741022   # +26.4%  mass_over_sum_pt_sq < 0.007509
        - 0.22664886791986819 * max(0.0, 0.00750911433 - Q.mass_over_sum_pt_sq) * max(0.0, 136.785 - Q.mass_top20) / 0.20432244897571727   # -22.7%  mass_over_sum_pt_sq < 0.007509 and mass_top20 < 136.8
        - 0.13768612396804408 * max(0.0, 0.428145796061 - Q.tau21) / 0.08904985831436439   # -13.8%  tau21 < 0.4281
        + 0.13491897334835676 * max(0.0, 0.000615484055 - Q.lam2) / 0.00014995048124430673   # +13.5%  lam2 < 0.0006155
        + 0.1263102006726101 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741004761) / 0.021355506979636982   # +12.6%  n_dr_0p2_0p4 < 5 and z_top50_slots > 0.9787
        + 0.08816907030540093 * max(0.0, 45.595 - Q.mass_top20) / 5.371095251098325   # +8.8%  mass_top20 < 45.59
        + 0.022387733779892893 * max(0.0, Q.z_top20_slots - 0.967685186161) * max(0.0, 0.000823693417 - Q.girth2_top3) / 2.388392763623668e-06   # +2.2%  z_top20_slots > 0.9677 and girth2_top3 < 0.0008237
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 17.31;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.30820663982834 * (0.119000333893206
        - 0.07700851705677045 * max(0.0, Q.LHA - 0.115200825015) / 0.14925839084262735   # -7.7%  LHA > 0.1152
        + 0.06694509186089079 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +6.7%  mass < 101
        - 0.05477905895893724 * max(0.0, 62.55 - Q.mass_top40) / 6.231504653826083   # -5.5%  mass_top40 < 62.55
        + 0.043193684479718046 * max(0.0, 0.012926423095 - Q.girth2_top40) / 0.006292907975779176   # +4.3%  girth2_top40 < 0.01293
        - 0.04095798829918533 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # -4.1%  girth < 0.1207
        - 0.03834533178458788 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -3.8%  mass < 86.4
        - 0.03809856724277512 * max(0.0, 0.020485236462 - Q.lam1) / 0.013097460086155891   # -3.8%  lam1 < 0.02049
        + 0.03665797975519799 * max(0.0, Q.width - 0.009614971338) / 0.003192189674729974   # +3.7%  width > 0.009615
        + 0.03622662110090527 * max(0.0, 0.015638355144 - Q.girth2_top15) / 0.009593305133886523   # +3.6%  girth2_top15 < 0.01564
        - 0.03536522803858664 * max(0.0, 91.19 - Q.mass) / 16.464230590854466   # -3.5%  mass < 91.19
        + 0.03534729948745325 * max(0.0, 6.916121244431 - Q.D2) / 4.12932700422166   # +3.5%  D2 < 6.916
        - 0.03176060889498185 * max(0.0, 152.688263064041 - Q.mass_top30) / 74.18247342351319   # -3.2%  mass_top30 < 152.7
        - 0.03144171642296646 * max(0.0, 94.642533639752 - Q.mass_top40) / 21.020214245469763   # -3.1%  mass_top40 < 94.64
        + 0.030099954499380688 * max(0.0, Q.girth2_top30 - 0.000856488074) / 0.007661496842941363   # +3.0%  girth2_top30 > 0.0008565
        + 0.027385783341255698 * max(0.0, Q.z_top50_slots - 0.958653609576) / 0.03427111698918734   # +2.7%  z_top50_slots > 0.9587
        + 0.02723957294465526 * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 17.21061680672269   # +2.7%  n_dr_0p2_0p4 < 26
        - 0.026567982015440673 * max(0.0, 0.055571487173 - Q.e2) / 0.02580562231087303   # -2.7%  e2 < 0.05557
        - 0.025441627856519534 * max(0.0, 83.325535102591 - Q.mass_top40) * max(0.0, 6.916121244431 - Q.D2) / 32.02926829226556   # -2.5%  mass_top40 < 83.33 and D2 < 6.916
        + 0.02541204037480147 * max(0.0, 83.325535102591 - Q.mass_top40) / 13.716491940764705   # +2.5%  mass_top40 < 83.33
        - 0.022691156319167863 * max(0.0, Q.mass_over_sum_pt - 0.06030418859) / 0.03186976854304421   # -2.3%  mass_over_sum_pt > 0.0603
        - 0.020754081408543974 * max(0.0, Q.girth2_top30 - 0.006363915755) / 0.003802702866408852   # -2.1%  girth2_top30 > 0.006364
        - 0.019214089306753925 * max(0.0, 0.061710142531 - Q.girth) / 0.012950562309845613   # -1.9%  girth < 0.06171
        + 0.01722636008858708 * max(0.0, Q.girth2_top50 - 0.008124776277) / 0.003479883171406613   # +1.7%  girth2_top50 > 0.008125
        - 0.01605026605386643 * max(0.0, Q.lam1 - 0.00767124277) / 0.0027489268789339613   # -1.6%  lam1 > 0.007671
        - 0.014075199514231806 * max(0.0, Q.girth2_top50 - 0.009262053166) / 0.0031795754743621793   # -1.4%  girth2_top50 > 0.009262
        - 0.013984332640046037 * max(0.0, 0.007277630044 - Q.girth2_top15) / 0.0028214695977655667   # -1.4%  girth2_top15 < 0.007278
        - 0.01093101531612606 * max(0.0, 69.027163795459 - Q.mass_top15) * max(0.0, 0.004007841607 - Q.girth2_top2) / 0.0617874128736903   # -1.1%  mass_top15 < 69.03 and girth2_top2 < 0.004008
        - 0.010628090509723213 * max(0.0, 0.068491501734 - Q.z_dr_0p2_0p4) / 0.04053661392560968   # -1.1%  z_dr_0p2_0p4 < 0.06849
        + 0.010491961489477297 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.004007841607 - Q.girth2_top2) / 0.07347560540847334   # +1.0%  mass < 101 and girth2_top2 < 0.004008
        + 0.010396956287082478 * max(0.0, 993.56640625 - Q.sum_pt_top20) / 88.76680814814208   # +1.0%  sum_pt_top20 < 993.6
        - 0.010291392702610345 * max(0.0, 74.251806640625 - Q.mass) / 8.587064460866076   # -1.0%  mass < 74.25
        + 0.01011405457568266 * max(0.0, 69.027163795459 - Q.mass_top15) / 18.230290548016082   # +1.0%  mass_top15 < 69.03
        + 0.010035213365791284 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, 0.402178311348 - Q.max_dr) / 2.4354257611523584   # +1.0%  mass < 120.6 and max_dr < 0.4022
        - 0.00982535329311307 * max(0.0, 80.784643554688 - Q.mass) / 10.849517559889705   # -1.0%  mass < 80.78
        - 0.009257645016863278 * max(0.0, 1066.481811523438 - Q.sum_pt) / 52.533044976652775   # -0.9%  sum_pt < 1066
        + 0.008787278363176549 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # +0.9%  mass > 143.8
        - 0.008217072548727057 * max(0.0, Q.mass_over_sum_pt - 0.09795414517) / 0.012228974621468441   # -0.8%  mass_over_sum_pt > 0.09795
        - 0.006627661011532434 * max(0.0, Q.n_particles - 29.0) * max(0.0, 25.0 - Q.n_dr_0_0p05) / 223.37779663865547   # -0.7%  n_particles > 29 and n_dr_0_0p05 < 25
        + 0.006571843832001735 * max(0.0, 0.025159193203 - Q.e2) / 0.004550017304548827   # +0.7%  e2 < 0.02516
        - 0.006497280747543596 * max(0.0, 78.261818313599 - Q.mass) / 9.857172988469275   # -0.6%  mass < 78.26
        + 0.005570017581372402 * max(0.0, Q.e2 - 0.036805817112) / 0.004603798258454848   # +0.6%  e2 > 0.03681
        - 0.005361407214817947 * max(0.0, Q.mass_top20 - 103.674910639856) / 3.7955110428668526   # -0.5%  mass_top20 > 103.7
        + 0.004008840325968312 * max(0.0, 68.286969674465 - Q.mass_top30) / 9.510710607689377   # +0.4%  mass_top30 < 68.29
        - 0.0023956291762219583 * max(0.0, Q.mass_top40 - 136.785) / 3.2712129356961093   # -0.2%  mass_top40 > 136.8
        - 0.0017211468959613715 * max(0.0, Q.LHA - 0.404204003833) / 0.0031577141977621142   # -0.2%  LHA > 0.4042
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 17.73;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.729612289452067 * (0.039519876611763885
        + 0.06954903615729675 * max(0.0, Q.log_sum_pt - 6.856374501323) / 0.09628579359874964   # +7.0%  log_sum_pt > 6.856
        - 0.0650261370324825 * max(0.0, Q.log_sum_pt - 6.910130970417) / 0.051750869608046556   # -6.5%  log_sum_pt > 6.91
        - 0.05484988237291577 * max(0.0, Q.z_top40_slots - 0.930046498893) / 0.051778900853579946   # -5.5%  z_top40_slots > 0.93
        + 0.05375427122184074 * max(0.0, 0.025803959699 - Q.mass_over_sum_pt_sq) / 0.016976624856690506   # +5.4%  mass_over_sum_pt_sq < 0.0258
        + 0.04826285945068935 * max(0.0, Q.z_top50_slots - 0.958653609576) / 0.03427111698918734   # +4.8%  z_top50_slots > 0.9587
        - 0.03947280133652478 * max(0.0, Q.max_dr - 0.240474711359) / 0.11773588635266226   # -3.9%  max_dr > 0.2405
        + 0.03452265923698794 * max(0.0, Q.mass_top30 - 68.286969674465) / 20.391116049249682   # +3.5%  mass_top30 > 68.29
        - 0.0334597503527171 * max(0.0, Q.n_real_top50 - 22.0) / 19.586852100840336   # -3.3%  n_real_top50 > 22
        + 0.03176055115771574 * max(0.0, Q.mass_over_sum_pt - 0.079990613285) / 0.01856800181915088   # +3.2%  mass_over_sum_pt > 0.07999
        - 0.03021723436899254 * max(0.0, Q.mass - 89.741833496094) / 15.555716339810672   # -3.0%  mass > 89.74
        - 0.028477197736656388 * max(0.0, Q.mass_over_sum_pt - 0.078528833221) / 0.0193369167748192   # -2.8%  mass_over_sum_pt > 0.07853
        + 0.027931238051576903 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.038759447634 - Q.e2) / 0.2513692835663279   # +2.8%  n_particles < 64 and e2 < 0.03876
        + 0.02606081340508391 * max(0.0, Q.sum_pt - 907.937170410156) * max(0.0, 0.0140332421 - Q.girth2_top2) / 1.486733466180341   # +2.6%  sum_pt > 907.9 and girth2_top2 < 0.01403
        + 0.025094155473539076 * max(0.0, Q.sum_pt_top50 - 934.241552734375) / 108.3880964071825   # +2.5%  sum_pt_top50 > 934.2
        + 0.025018510743108194 * max(0.0, Q.sum_pt_top40 - 906.60234375) / 122.36719248432757   # +2.5%  sum_pt_top40 > 906.6
        + 0.024850934183946263 * max(0.0, Q.mass - 64.485443115234) / 31.191644704866818   # +2.5%  mass > 64.49
        - 0.02409219076792177 * max(0.0, Q.log_sum_pt - 6.89371369877) * max(0.0, 0.187280465662 - Q.z_dr_0p1_0p2) / 0.006505076767982554   # -2.4%  log_sum_pt > 6.894 and z_dr_0p1_0p2 < 0.1873
        - 0.024026790234398937 * max(0.0, Q.log_sum_pt - 6.89371369877) / 0.06417423647849095   # -2.4%  log_sum_pt > 6.894
        - 0.02170876592818894 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # -2.2%  mass > 172.8
        - 0.01968427725209137 * max(0.0, Q.mass_over_sum_pt_sq - 0.029200015571) / 0.00020282738822808687   # -2.0%  mass_over_sum_pt_sq > 0.0292
        - 0.018796123830360393 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -1.9%  girth < 0.08589
        - 0.018317577376317278 * max(0.0, Q.log_sum_pt - 6.92034855022) * max(0.0, 0.0140332421 - Q.girth2_top2) / 0.0005041067528121594   # -1.8%  log_sum_pt > 6.92 and girth2_top2 < 0.01403
        + 0.017550365288003463 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # +1.8%  n_dr_0p2_0p4 < 13
        + 0.01596594287881573 * max(0.0, Q.log_sum_pt - 6.935549248787) * max(0.0, 0.187280465662 - Q.z_dr_0p1_0p2) / 0.004033366591149925   # +1.6%  log_sum_pt > 6.936 and z_dr_0p1_0p2 < 0.1873
        + 0.015712077191729252 * max(0.0, Q.log_sum_pt - 6.959293500649) / 0.028460039078747168   # +1.6%  log_sum_pt > 6.959
        - 0.013895150074663713 * max(0.0, 0.00895655368 - Q.girth2_top10) / 0.0044733890107435115   # -1.4%  girth2_top10 < 0.008957
        - 0.01353230982543382 * max(0.0, 0.051804735139 - Q.z_dr_0p2_0p4) / 0.028368155573045024   # -1.4%  z_dr_0p2_0p4 < 0.0518
        - 0.013076369124638323 * max(0.0, Q.mass - 162.836349487305) / 1.5098322961374488   # -1.3%  mass > 162.8
        + 0.011390227096453692 * max(0.0, Q.sum_pt_top40 - 935.8189453125) / 96.68530102622462   # +1.1%  sum_pt_top40 > 935.8
        - 0.011185168200642748 * max(0.0, 0.070317784324 - Q.girth) / 0.017195212469713443   # -1.1%  girth < 0.07032
        - 0.011156651186779762 * max(0.0, Q.mass_over_sum_pt - 0.170880120467) / 0.0005536193928784717   # -1.1%  mass_over_sum_pt > 0.1709
        - 0.01087598713228253 * max(0.0, Q.mass_over_sum_pt_sq - 0.029200015571) * max(0.0, Q.phi_6 - -0.11682434082) / 2.682861956115054e-05   # -1.1%  mass_over_sum_pt_sq > 0.0292 and phi_6 > -0.1168
        - 0.010788702203903282 * max(0.0, Q.sum_pt_top50 - 1061.183898925781) / 28.411282622213932   # -1.1%  sum_pt_top50 > 1061
        + 0.010776298566984676 * max(0.0, Q.log_sum_pt - 6.989450376716) / 0.021142744446604637   # +1.1%  log_sum_pt > 6.989
        - 0.009758163648178022 * max(0.0, Q.mass_over_sum_pt - 0.090467494167) / 0.014210394977997521   # -1.0%  mass_over_sum_pt > 0.09047
        + 0.008807048789030884 * max(0.0, 0.018076787298 - Q.girth2_top30) * max(0.0, 0.286492615938 - Q.z_dr_0p1_0p2) / 0.002126628915073142   # +0.9%  girth2_top30 < 0.01808 and z_dr_0p1_0p2 < 0.2865
        + 0.0080431865586985 * max(0.0, 0.004752875822 - Q.girth2_top10) / 0.0016234175755221222   # +0.8%  girth2_top10 < 0.004753
        - 0.007946504708776725 * max(0.0, Q.mass_top30 - 101.927236862114) / 7.286523582862451   # -0.8%  mass_top30 > 101.9
        - 0.007926293090794729 * max(0.0, Q.sum_pt_top15 - 951.1375) / 24.48977481165237   # -0.8%  sum_pt_top15 > 951.1
        - 0.007771133964798948 * max(0.0, Q.z_top20_slots - 0.923451750505) / 0.020619026890972596   # -0.8%  z_top20_slots > 0.9235
        - 0.0070804843005787006 * max(0.0, 0.085894044489 - Q.girth) * max(0.0, Q.n_pt_above_50 - 2.0) / 0.09099878037845041   # -0.7%  girth < 0.08589 and n_pt_above_50 > 2
        - 0.006887973448695026 * max(0.0, Q.mass_top40 - 136.785) / 3.2712129356961093   # -0.7%  mass_top40 > 136.8
        + 0.006677938483660763 * max(0.0, 64.0 - Q.n_particles) * max(0.0, Q.z_dr_0p2_0p4 - 0.004744913615) / 0.4079057124357439   # +0.7%  n_particles < 64 and z_dr_0p2_0p4 > 0.004745
        - 0.006670236790682118 * max(0.0, Q.log_sum_pt - 6.935549248787) / 0.03715380406194132   # -0.7%  log_sum_pt > 6.936
        + 0.006268050185097634 * max(0.0, Q.sum_pt_top10 - 845.53671875) / 35.972808947321965   # +0.6%  sum_pt_top10 > 845.5
        + 0.00422527638644918 * max(0.0, Q.mass_top50 - 91.19) / 13.751447163946986   # +0.4%  mass_top50 > 91.19
        + 0.004113782057749244 * max(0.0, 0.070317784324 - Q.girth) * max(0.0, Q.n_pt_above_50 - 2.0) / 0.057205086388322265   # +0.4%  girth < 0.07032 and n_pt_above_50 > 2
        + 0.0037894799989262444 * max(0.0, Q.sum_pt_top20 - 1005.0126953125) / 20.50199904518448   # +0.4%  sum_pt_top20 > 1005
        + 0.002526343696774639 * max(0.0, Q.max_dr - 0.43572281599) / 0.007711927791734812   # +0.3%  max_dr > 0.4357
        + 0.0006690974494248504 * max(0.0, Q.mass_top50 - 157.544814151857) * max(0.0, Q.z_dr_0p05_0p1 - 0.591232848167) / 0.009747485253060875   # +0.1%  mass_top50 > 157.5 and z_dr_0p05_0p1 > 0.5912
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 19.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.735391676604113 * (0.03677396176693603
        - 0.08974286550597449 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # -9.0%  mass < 101
        - 0.08119451276815433 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) / 0.039273301687911995   # -8.1%  mass_over_sum_pt > 0.05077
        + 0.07937014487690543 * max(0.0, 172.8 - Q.mass) / 83.7879223272957   # +7.9%  mass < 172.8
        - 0.07623432818265691 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # -7.6%  mass < 120.6
        + 0.07234065462784452 * max(0.0, 0.019863807341 - Q.mass_over_sum_pt_sq) / 0.011777496153852728   # +7.2%  mass_over_sum_pt_sq < 0.01986
        + 0.05841425947889506 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # +5.8%  mass < 92.86
        - 0.05069265688786772 * max(0.0, 160.8 - Q.mass_top40) / 76.75884216591673   # -5.1%  mass_top40 < 160.8
        - 0.04528277437506118 * max(0.0, 0.019516409491 - Q.girth2_top50) / 0.011590985649215664   # -4.5%  girth2_top50 < 0.01952
        + 0.03375177808116459 * max(0.0, 0.047553086095 - Q.e2) / 0.018774568371458196   # +3.4%  e2 < 0.04755
        + 0.029517868572166758 * max(0.0, 1085.12490234375 - Q.sum_pt) / 67.04637030924575   # +3.0%  sum_pt < 1085
        - 0.028941500413828116 * max(0.0, 0.026753638475 - Q.girth2_top15) / 0.019599376803581564   # -2.9%  girth2_top15 < 0.02675
        + 0.02893274649072995 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # +2.9%  mass < 86.4
        - 0.027675440802197464 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063025210084   # -2.8%  n_dr_0p2_0p4 < 21
        + 0.024789209513798444 * max(0.0, Q.LHA - 0.260146178237) / 0.041583761487988195   # +2.5%  LHA > 0.2601
        - 0.024130745904289066 * max(0.0, 6.989450376716 - Q.log_sum_pt) / 0.06598628507448029   # -2.4%  log_sum_pt < 6.989
        + 0.023424388397559042 * max(0.0, Q.girth2_top20 - 0.0018230789) / 0.006286421415254086   # +2.3%  girth2_top20 > 0.001823
        - 0.021086384709064463 * max(0.0, 0.009614971338 - Q.width) / 0.003502544629791685   # -2.1%  width < 0.009615
        + 0.015961719929573175 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 0.32206342639914515   # +1.6%  mass_over_sum_pt > 0.05077 and n_dr_0p2_0p4 < 21
        + 0.014956210259615925 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.002396991421 - Q.lam2) / 3.6218014529889185e-05   # +1.5%  mass_over_sum_pt > 0.05077 and lam2 < 0.002397
        - 0.014738807271806499 * max(0.0, Q.girth2_top3 - 0.010023689877) / 0.001644746899335674   # -1.5%  girth2_top3 > 0.01002
        + 0.014296293014416024 * max(0.0, Q.girth2_top3 - 0.010023689877) * max(0.0, 4.450168704987 - Q.D2) / 0.0041143288114283754   # +1.4%  girth2_top3 > 0.01002 and D2 < 4.45
        - 0.013942559435261086 * max(0.0, Q.girth2_top20 - 0.016559833876) / 0.001314598562186081   # -1.4%  girth2_top20 > 0.01656
        + 0.013594691391742852 * max(0.0, Q.LHA - 0.228402115913) / 0.06082067856197019   # +1.4%  LHA > 0.2284
        + 0.012861228534558226 * max(0.0, 0.007259287357 - Q.lam1) / 0.0022497701299997253   # +1.3%  lam1 < 0.007259
        - 0.01244905056253674 * max(0.0, Q.LHA - 0.228402115913) * max(0.0, 3.345338582993 - Q.D2) / 0.10056812770798138   # -1.2%  LHA > 0.2284 and D2 < 3.345
        + 0.010383004362933792 * max(0.0, 94.642533639752 - Q.mass_top40) / 21.020214245469763   # +1.0%  mass_top40 < 94.64
        + 0.009534584804118182 * max(0.0, 0.043586218357 - Q.e2) / 0.015487320383570933   # +1.0%  e2 < 0.04359
        - 0.009322475052506603 * max(0.0, 0.030297144316 - Q.e2) / 0.0068225745792513896   # -0.9%  e2 < 0.0303
        + 0.009109235428126054 * max(0.0, Q.girth2_top20 - 0.0018230789) * max(0.0, 0.43572281599 - Q.max_dr) / 0.0005012106213039546   # +0.9%  girth2_top20 > 0.001823 and max_dr < 0.4357
        + 0.007646394780365046 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # +0.8%  mass_top50 < 71.8
        + 0.00651447376176128 * max(0.0, Q.lam1 - 0.008241985248) * max(0.0, 1167.4466796875 - Q.sum_pt) / 0.4321066405342491   # +0.7%  lam1 > 0.008242 and sum_pt < 1167
        + 0.006352186089482307 * max(0.0, Q.e2 - 0.055571487173) / 0.0011197307961334129   # +0.6%  e2 > 0.05557
        - 0.005300781934930467 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, Q.max_pair_mass - 13.047927274731) / 0.3749186717396242   # -0.5%  mass_over_sum_pt > 0.05077 and max_pair_mass > 13.05
        + 0.005268244989880771 * max(0.0, 0.004673423215 - Q.lam1) / 0.00095571431893026   # +0.5%  lam1 < 0.004673
        + 0.005080446903557091 * max(0.0, 0.008031209355 - Q.girth2_top20) / 0.0030984903862893757   # +0.5%  girth2_top20 < 0.008031
        - 0.004508811601134943 * max(0.0, Q.lam1 - 0.011744050682) / 0.0018275243697888329   # -0.5%  lam1 > 0.01174
        - 0.003997269527480165 * max(0.0, Q.e2 - 0.055571487173) * max(0.0, Q.n_real_top40 - 34.0) / 0.006474040958609148   # -0.4%  e2 > 0.05557 and n_real_top40 > 34
        + 0.0037526453507491955 * max(0.0, Q.lam1 - 0.008241985248) / 0.002595658483103056   # +0.4%  lam1 > 0.008242
        + 0.0017480185117342954 * max(0.0, Q.LHA - 0.404204003833) * max(0.0, Q.z_2 - 0.049737748174) / 6.972499774959441e-05   # +0.2%  LHA > 0.4042 and z_2 > 0.04974
        + 0.0012767866036239419 * max(0.0, Q.sum_pt_top40 - 1139.734106445312) / 13.31290013539202   # +0.1%  sum_pt_top40 > 1140
        + 0.0012315302606429929 * max(0.0, Q.mass_top10 - 99.066784770599) / 0.7458098208346458   # +0.1%  mass_top10 > 99.07
        - 0.0006502900493048018 * max(0.0, Q.e2 - 0.055571487173) * max(0.0, 0.368618160486 - Q.max_dr) / 1.040948113205381e-05   # -0.1%  e2 > 0.05557 and max_dr < 0.3686
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 31.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.160558948134764 * (-0.009694815190756607
        + 0.052394057031814316 * max(0.0, 0.320332145368 - Q.LHA) / 0.07499257253386664   # +5.2%  LHA < 0.3203
        + 0.05018772584984173 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # +5.0%  mass < 120.6
        - 0.049798279540829084 * max(0.0, 82.85408782959 - Q.mass) / 11.83453618826746   # -5.0%  mass < 82.85
        - 0.0458708746464213 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -4.6%  girth < 0.08589
        - 0.04575512201011994 * max(0.0, 87.363773345947 - Q.mass) / 14.19996044022135   # -4.6%  mass < 87.36
        + 0.044437786276008126 * max(0.0, 0.009606007381 - Q.e2_sq) / 0.003497865944155776   # +4.4%  e2_sq < 0.009606
        - 0.040413168524899346 * max(0.0, 0.008031209355 - Q.girth2_top20) / 0.0030984903862893757   # -4.0%  girth2_top20 < 0.008031
        - 0.03779995180610527 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # -3.8%  mass < 92.86
        - 0.03512930246534264 * max(0.0, 1260.540869140625 - Q.sum_pt) / 224.40051988701538   # -3.5%  sum_pt < 1261
        - 0.03405699623263932 * max(0.0, 92.165451466106 - Q.mass_top50) / 17.798728151002944   # -3.4%  mass_top50 < 92.17
        + 0.033740024833845327 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.390781164169 - Q.max_dr) / 1.1947278759545992   # +3.4%  mass < 101 and max_dr < 0.3908
        + 0.03302212535443186 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +3.3%  mass < 101
        - 0.03189183786805723 * max(0.0, 0.008031986041 - Q.girth2_top40) / 0.0025145928123983495   # -3.2%  girth2_top40 < 0.008032
        - 0.031598531591279656 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # -3.2%  mass_top50 < 71.8
        - 0.03142427595826644 * max(0.0, 0.333234539952 - Q.LHA) / 0.0850198895160223   # -3.1%  LHA < 0.3332
        - 0.029399723900685005 * max(0.0, 97.930041729355 - Q.mass_top50) * max(0.0, 1.601009327173 - Q.D2) / 2.279411140212284   # -2.9%  mass_top50 < 97.93 and D2 < 1.601
        - 0.02925777046864581 * max(0.0, 91.19 - Q.mass) * max(0.0, 0.390781164169 - Q.max_dr) / 0.7740281265073297   # -2.9%  mass < 91.19 and max_dr < 0.3908
        + 0.029023966947764806 * max(0.0, 0.008840538245 - Q.girth2_top40) / 0.0031086216346541064   # +2.9%  girth2_top40 < 0.008841
        + 0.026859087632573796 * max(0.0, 86.4 - Q.mass_top50) / 14.253133905303311   # +2.7%  mass_top50 < 86.4
        - 0.025129682118842922 * max(0.0, 0.090467494167 - Q.mass_over_sum_pt) / 0.0178873033081874   # -2.5%  mass_over_sum_pt < 0.09047
        + 0.024076507113172427 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 1.601009327173 - Q.D2) / 2.8196790984068856   # +2.4%  mass < 101 and D2 < 1.601
        - 0.02016226063996465 * max(0.0, 91.19 - Q.mass) / 16.464230590854466   # -2.0%  mass < 91.19
        - 0.019222609508982854 * max(0.0, 0.008376290695 - Q.girth2_top30) / 0.0029810771135580514   # -1.9%  girth2_top30 < 0.008376
        + 0.017661008042355597 * max(0.0, 79.21003612387 - Q.mass_top50) * max(0.0, 137.5 - Q.pt_2) / 397.7877576611584   # +1.8%  mass_top50 < 79.21 and pt_2 < 137.5
        + 0.016705469428704822 * max(0.0, 0.012157872869 - Q.girth2_top30) / 0.005935411821644211   # +1.7%  girth2_top30 < 0.01216
        + 0.01596182368197269 * max(0.0, 0.007820314762 - Q.girth2_top50) / 0.002264874199562952   # +1.6%  girth2_top50 < 0.00782
        + 0.015386163858115674 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # +1.5%  mass < 86.4
        + 0.013541558969730451 * max(0.0, 71.795159472175 - Q.mass_top50) * max(0.0, 0.39398368001 - Q.max_dr) / 0.376497726873284   # +1.4%  mass_top50 < 71.8 and max_dr < 0.394
        + 0.010862505859384287 * max(0.0, 76.415438713532 - Q.mass_top30) / 12.67595855468548   # +1.1%  mass_top30 < 76.42
        + 0.010426431836175587 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.499317836761 - Q.z_1st) / 1.5556551236660203   # +1.0%  n_dr_0p2_0p4 < 13 and z_1st < 0.4993
        + 0.01027730371215367 * max(0.0, 0.007709915821 - Q.girth2_top40) / 0.0022972993366394247   # +1.0%  girth2_top40 < 0.00771
        + 0.00769384772522983 * max(0.0, Q.z_dr_0_0p05 - 0.181086004525) / 0.3722859010670908   # +0.8%  z_dr_0_0p05 > 0.1811
        - 0.007490268180523271 * max(0.0, 0.007709915821 - Q.girth2_top40) * max(0.0, Q.n_pt_above_10 - 10.0) / 0.017632385918729388   # -0.7%  girth2_top40 < 0.00771 and n_pt_above_10 > 10
        + 0.007108056348312887 * max(0.0, 1038.26171875 - Q.sum_pt_top30) / 69.34980875418526   # +0.7%  sum_pt_top30 < 1038
        + 0.0066294755035625885 * max(0.0, 66.841467317407 - Q.mass_top20) / 12.904985456467061   # +0.7%  mass_top20 < 66.84
        + 0.00657303957793765 * max(0.0, 86.252206812802 - Q.mass_top30) / 18.207265524434344   # +0.7%  mass_top30 < 86.25
        + 0.0065155837768166615 * max(0.0, 1.788105106354 - Q.D2) * max(0.0, 1245.696667480468 - Q.sum_pt_top50) / 61.3686421801194   # +0.7%  D2 < 1.788 and sum_pt_top50 < 1246
        + 0.006472766847877478 * max(0.0, 0.006374177987 - Q.girth2_top20) / 0.0019877511397026312   # +0.6%  girth2_top20 < 0.006374
        - 0.0059619770630539425 * max(0.0, 0.056027559564 - Q.C2) / 0.009882646645223543   # -0.6%  C2 < 0.05603
        + 0.005689524494469628 * max(0.0, 1.788105106354 - Q.D2) * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 1.537145490198633   # +0.6%  D2 < 1.788 and n_dr_0p2_0p4 < 9
        + 0.005436845381343321 * max(0.0, 0.000615484055 - Q.lam2) / 0.00014995048124430673   # +0.5%  lam2 < 0.0006155
        - 0.0054122695558817034 * max(0.0, 91.19 - Q.mass) * max(0.0, 0.33780374676 - Q.pt_dispersion) / 0.4764307098452709   # -0.5%  mass < 91.19 and pt_dispersion < 0.3378
        + 0.004829599408495017 * max(0.0, 0.008031209355 - Q.girth2_top20) * max(0.0, Q.z_dr_0p2_0p4 - 0.000658670394) / 5.240807687245126e-05   # +0.5%  girth2_top20 < 0.008031 and z_dr_0p2_0p4 > 0.0006587
        - 0.003948223863299109 * max(0.0, 1.788105106354 - Q.D2) / 0.2865857032958301   # -0.4%  D2 < 1.788
        - 0.0031161686041720846 * max(0.0, 0.990637830118 - Q.z_top50_slots) / 0.004783315002560602   # -0.3%  z_top50_slots < 0.9906
        + 0.002586490706352662 * max(0.0, 1.788105106354 - Q.D2) * max(0.0, 0.007820314762 - Q.girth2_top50) / 0.00026955979916256045   # +0.3%  D2 < 1.788 and girth2_top50 < 0.00782
        - 0.0018884973350250481 * max(0.0, 0.383415880799 - Q.max_dr) * max(0.0, Q.eccentricity - 0.841527497033) / 0.00309141983654075   # -0.2%  max_dr < 0.3834 and eccentricity > 0.8415
        - 0.0011734319185204985 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.402664637566) / 0.4065322951460936   # -0.1%  mass < 91.19 and z_dr_0p05_0p1 > 0.4027
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 15.3;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.301076572766931 * (-0.023693861306851857
        + 0.08110605400175895 * max(0.0, Q.mass - 64.485443115234) / 31.191644704866818   # +8.1%  mass > 64.49
        - 0.08073632262305208 * max(0.0, Q.mass - 101.049709320068) / 12.310843098263781   # -8.1%  mass > 101
        - 0.06959776217399606 * max(0.0, 0.013977372691 - Q.mass_over_sum_pt_sq) / 0.006903769640131633   # -7.0%  mass_over_sum_pt_sq < 0.01398
        - 0.06057163548129763 * max(0.0, Q.mass_over_sum_pt - 0.09795414517) / 0.012228974621468441   # -6.1%  mass_over_sum_pt > 0.09795
        + 0.058549692723392034 * max(0.0, Q.mass - 80.784643554688) / 19.806095548570884   # +5.9%  mass > 80.78
        + 0.053452965928519044 * max(0.0, 0.097499583662 - Q.girth) / 0.03657082438923025   # +5.3%  girth < 0.0975
        + 0.044840947479016666 * max(0.0, 0.008241985248 - Q.lam1) / 0.0029450268440803277   # +4.5%  lam1 < 0.008242
        + 0.04426689393469931 * max(0.0, Q.mass - 87.363773345947) / 16.577408637801742   # +4.4%  mass > 87.36
        + 0.04107428693349365 * max(0.0, Q.girth2_top20 - 0.008031209355) / 0.00290685235713101   # +4.1%  girth2_top20 > 0.008031
        - 0.039784416795779626 * max(0.0, 0.020485236462 - Q.lam1) / 0.013097460086155891   # -4.0%  lam1 < 0.02049
        + 0.0368248699520354 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) * max(0.0, 1115.722741699219 - Q.sum_pt) / 2.3366981130955806   # +3.7%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        + 0.03291735767401736 * max(0.0, Q.mass_top50 - 97.930041729355) / 11.917644068610143   # +3.3%  mass_top50 > 97.93
        - 0.03242493397649058 * max(0.0, Q.mass_top50 - 82.04491364955) / 17.710574785086557   # -3.2%  mass_top50 > 82.04
        - 0.03175767895627917 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063025210084   # -3.2%  n_dr_0p2_0p4 < 21
        + 0.028587155180034244 * max(0.0, 6.941996527183 - Q.log_sum_pt) / 0.03180971936508312   # +2.9%  log_sum_pt < 6.942
        + 0.027533227147334874 * max(0.0, Q.mass_over_sum_pt - 0.088731426731) / 0.014771713814206479   # +2.8%  mass_over_sum_pt > 0.08873
        + 0.02481969449910135 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) / 0.05824141942126331   # +2.5%  z_dr_0p2_0p4 < 0.09123
        - 0.02317016048731204 * max(0.0, Q.girth2_top20 - 0.005718442372) / 0.0037490635294349373   # -2.3%  girth2_top20 > 0.005718
        - 0.01979968864752697 * max(0.0, 0.14021858573 - Q.girth) / 0.07284555213077804   # -2.0%  girth < 0.1402
        - 0.018688280950636176 * max(0.0, 0.006166777647 - Q.mass_over_sum_pt_sq) / 0.0012952540801658948   # -1.9%  mass_over_sum_pt_sq < 0.006167
        - 0.018297481365878138 * max(0.0, Q.mass - 125.1) / 7.098975738857251   # -1.8%  mass > 125.1
        + 0.015676141902058027 * max(0.0, 0.154838323593 - Q.z_dr_0p1_0p2) / 0.06710802455713627   # +1.6%  z_dr_0p1_0p2 < 0.1548
        - 0.013337278916071641 * max(0.0, 1011.52392578125 - Q.sum_pt_top30) / 51.02618157784598   # -1.3%  sum_pt_top30 < 1012
        - 0.011969934553762803 * max(0.0, Q.e2 - 0.025159193203) / 0.010276419760149801   # -1.2%  e2 > 0.02516
        + 0.011725541625956412 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # +1.2%  sum_pt < 1013
        + 0.011123720474150499 * max(0.0, Q.mass_top15 - 30.359943489662) / 32.64933730076125   # +1.1%  mass_top15 > 30.36
        - 0.010361874136823726 * max(0.0, Q.mass_top40 - 79.554505888974) / 17.06549122004638   # -1.0%  mass_top40 > 79.55
        - 0.010303142527128847 * max(0.0, 1031.422265625 - Q.sum_pt_top50) / 33.4418517165412   # -1.0%  sum_pt_top50 < 1031
        + 0.009199969419041375 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) / 0.020241059768600602   # +0.9%  mass_over_sum_pt > 0.07697
        + 0.008924220604304617 * max(0.0, 0.813850690953 - Q.z_top15_slots) / 0.039447636639057045   # +0.9%  z_top15_slots < 0.8139
        + 0.0066739271001494275 * max(0.0, Q.girth2_top40 - 0.005196965925) / 0.004725809282736856   # +0.7%  girth2_top40 > 0.005197
        - 0.0043485308724921495 * max(0.0, Q.sum_pt - 1260.540869140625) / 7.65548318400563   # -0.4%  sum_pt > 1261
        - 0.0042400021921000855 * max(0.0, Q.mass - 87.363773345947) * max(0.0, 6.903422848462 - Q.log_sum_pt) / 0.2991521451407768   # -0.4%  mass > 87.36 and log_sum_pt < 6.903
        + 0.004074496320093684 * max(0.0, Q.sum_pt_top40 - 1225.8421875) / 7.25975331858652   # +0.4%  sum_pt_top40 > 1226
        + 0.0038970922928431946 * max(0.0, 0.985099030959 - Q.z_top50_slots) / 0.0035560753163622707   # +0.4%  z_top50_slots < 0.9851
        - 0.003466903098688167 * max(0.0, Q.girth2_top40 - 0.005196965925) * max(0.0, 911.9328125 - Q.sum_pt_top30) / 0.19161403745216243   # -0.3%  girth2_top40 > 0.005197 and sum_pt_top30 < 911.9
        - 0.0018757170526841188 * max(0.0, 1001.52314453125 - Q.sum_pt_top40) * max(0.0, 6.811175180312 - Q.log_sum_pt) / 1.4602943926211116   # -0.2%  sum_pt_top40 < 1002 and log_sum_pt < 6.811
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 14.69;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.685359603452618 * (0.04013434966184868
        + 0.20346032408947315 * max(0.0, 136.785 - Q.mass_top50) / 53.22362615511637   # +20.3%  mass_top50 < 136.8
        - 0.1643455308582388 * max(0.0, 160.8 - Q.mass_top50) / 74.23410092497848   # -16.4%  mass_top50 < 160.8
        - 0.12579670970445503 * max(0.0, Q.mass - 62.55) / 32.65527990955991   # -12.6%  mass > 62.55
        + 0.0814258631378138 * max(0.0, Q.mass - 82.85408782959) / 18.721669902205264   # +8.1%  mass > 82.85
        - 0.06717975470141006 * max(0.0, 0.02146577947 - Q.girth2_top15) / 0.0147044065440157   # -6.7%  girth2_top15 < 0.02147
        + 0.034679006822216855 * max(0.0, Q.mass - 92.85979309082) / 14.482733665646967   # +3.5%  mass > 92.86
        - 0.032722880722826544 * max(0.0, 91.288535717504 - Q.mass_top40) / 18.56549606317863   # -3.3%  mass_top40 < 91.29
        + 0.03246007546991599 * max(0.0, 0.209102506978 - Q.LHA) / 0.020957818638470994   # +3.2%  LHA < 0.2091
        - 0.031232803172920504 * max(0.0, 0.043628720567 - Q.girth) / 0.0063254006084753804   # -3.1%  girth < 0.04363
        - 0.02745607825829364 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -2.7%  mass > 143.8
        + 0.024091734953239706 * max(0.0, 80.890431271924 - Q.mass_top40) / 12.434053057794845   # +2.4%  mass_top40 < 80.89
        + 0.02323957418655056 * max(0.0, 136.785 - Q.mass) / 52.08717491697615   # +2.3%  mass < 136.8
        + 0.023095805383981587 * max(0.0, 0.009962397174 - Q.girth2_top15) / 0.004894698509240601   # +2.3%  girth2_top15 < 0.009962
        + 0.01580768710221322 * max(0.0, 949.91689453125 - Q.sum_pt) / 6.913716555695057   # +1.6%  sum_pt < 949.9
        + 0.01572606096878508 * max(0.0, 76.415438713532 - Q.mass_top30) / 12.67595855468548   # +1.6%  mass_top30 < 76.42
        + 0.01498047288011006 * max(0.0, 120.60000000000001 - Q.mass_top40) / 41.60439949481591   # +1.5%  mass_top40 < 120.6
        - 0.012215950847371947 * max(0.0, 6.856374501323 - Q.log_sum_pt) / 0.008053458833961623   # -1.2%  log_sum_pt < 6.856
        - 0.011313833611555626 * max(0.0, Q.mass_top15 - 40.2) / 25.55355986978338   # -1.1%  mass_top15 > 40.2
        + 0.011152280332954062 * max(0.0, 0.00625977218 - Q.girth2_top40) / 0.0014618229210262657   # +1.1%  girth2_top40 < 0.00626
        + 0.009503189332122314 * max(0.0, 0.00363885588 - Q.girth2) / 0.0004964601451473658   # +1.0%  girth2 < 0.003639
        + 0.007357027034528463 * max(0.0, Q.girth - 0.097499583662) / 0.008315821681586878   # +0.7%  girth > 0.0975
        + 0.006920506877298203 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104090756302521   # +0.7%  n_dr_0p2_0p4 < 5
        - 0.00639313840156769 * max(0.0, 0.02783744745 - Q.girth) / 0.0024110754510684616   # -0.6%  girth < 0.02784
        + 0.004194373673036643 * max(0.0, 956.21328125 - Q.sum_pt_top40) / 14.004510341849162   # +0.4%  sum_pt_top40 < 956.2
        + 0.0033271639646360158 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +0.3%  mass > 172.8
        - 0.002808504122846874 * max(0.0, Q.width - 0.025866900997) / 0.00046535522878793676   # -0.3%  width > 0.02587
        - 0.0021462496404056138 * max(0.0, Q.e2 - 0.065240035206) / 0.0004075092339807838   # -0.2%  e2 > 0.06524
        - 0.0019424560023705985 * max(0.0, 956.21328125 - Q.sum_pt_top40) * max(0.0, Q.z_top15_slots - 0.794624168612) / 0.41323682564243985   # -0.2%  sum_pt_top40 < 956.2 and z_top15_slots > 0.7946
        + 0.0017496905328478185 * max(0.0, 6.856374501323 - Q.log_sum_pt) * max(0.0, Q.z_top20_slots - 0.896541111574) / 0.0002016422589223067   # +0.2%  log_sum_pt < 6.856 and z_top20_slots > 0.8965
        - 0.001275273214013459 * max(0.0, 956.21328125 - Q.sum_pt_top40) * max(0.0, Q.z_top30_slots - 0.97348863653) / 0.05156393724254791   # -0.1%  sum_pt_top40 < 956.2 and z_top30_slots > 0.9735
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.448415610312265 * (0.11126495812571907
        + 0.2351308490904863 * max(0.0, 87.363773345947 - Q.mass) / 14.19996044022135   # +23.5%  mass < 87.36
        - 0.19410070456085338 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -19.4%  mass < 86.4
        - 0.07479529721972361 * max(0.0, 0.065240035206 - Q.e2) / 0.03476194878141159   # -7.5%  e2 < 0.06524
        - 0.03844047438989351 * max(0.0, Q.z_top5 - 0.419766938686) / 0.16171630039159515   # -3.8%  z_top5 > 0.4198
        + 0.03629877929004164 * max(0.0, Q.girth2_top10 - 0.000237176831) / 0.006544219018306094   # +3.6%  girth2_top10 > 0.0002372
        + 0.03420992232034746 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +3.4%  mass < 80.4
        + 0.034070919584510355 * max(0.0, Q.mass - 74.251806640625) / 24.076479363593222   # +3.4%  mass > 74.25
        - 0.026133182672594197 * max(0.0, Q.lam1 - 0.004673423215) / 0.004174907990959959   # -2.6%  lam1 > 0.004673
        - 0.02292512122382815 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -2.3%  mass < 62.55
        - 0.02267356630412678 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -2.3%  mass > 143.8
        - 0.02233785841102092 * max(0.0, Q.mass - 160.8) / 1.7246633179459985   # -2.2%  mass > 160.8
        + 0.019449151573739974 * max(0.0, 3.345338582993 - Q.D2) / 1.1793772463327215   # +1.9%  D2 < 3.345
        + 0.01693886627835485 * max(0.0, Q.mass_top50 - 136.785) / 4.25098606820319   # +1.7%  mass_top50 > 136.8
        - 0.016061332704364058 * max(0.0, 0.065240035206 - Q.e2) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 0.004800315654529198   # -1.6%  e2 < 0.06524 and z_dr_0p1_0p2 < 0.2188
        + 0.01600953046184 * max(0.0, 0.024419631481 - Q.girth2_top5) * max(0.0, 656.384375 - Q.sum_pt_top3) / 3.538216833032187   # +1.6%  girth2_top5 < 0.02442 and sum_pt_top3 < 656.4
        + 0.014600451662877729 * max(0.0, 0.005925373826 - Q.girth2) / 0.001193203808700739   # +1.5%  girth2 < 0.005925
        + 0.01390156273005309 * max(0.0, Q.mass_top5 - 14.544037663713) / 16.71370088434276   # +1.4%  mass_top5 > 14.54
        - 0.013462152360188665 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # -1.3%  girth < 0.1207
        - 0.013338857211295149 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # -1.3%  mass < 101
        - 0.011680734612249674 * max(0.0, Q.mass_top20 - 80.4) / 8.58576931586636   # -1.2%  mass_top20 > 80.4
        - 0.011154194108558182 * max(0.0, 0.187029113551 - Q.LHA) / 0.01494324260337732   # -1.1%  LHA < 0.187
        - 0.01087352532810465 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # -1.1%  n_dr_0p2_0p4 < 13
        + 0.010007787726518424 * max(0.0, Q.mass_top40 - 136.785) / 3.2712129356961093   # +1.0%  mass_top40 > 136.8
        + 0.009795077733476836 * max(0.0, Q.z_top5 - 0.419766938686) * max(0.0, 0.175028083821 - Q.dr_2) / 0.021413576719416086   # +1.0%  z_top5 > 0.4198 and dr_2 < 0.175
        - 0.009786778951240131 * max(0.0, Q.girth2_top10 - 0.007678543663) / 0.002556946129466314   # -1.0%  girth2_top10 > 0.007679
        + 0.009527134836831206 * max(0.0, Q.mass - 162.836349487305) / 1.5098322961374488   # +1.0%  mass > 162.8
        + 0.00927239598257596 * max(0.0, 1052.07998046875 - Q.sum_pt_top30) / 80.11663973251567   # +0.9%  sum_pt_top30 < 1052
        + 0.008105671035424806 * max(0.0, 0.064132973195 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.14948521527659667   # +0.8%  dr_0 < 0.06413 and n_dr_0p2_0p4 > 2
        + 0.008038426261162015 * max(0.0, Q.lam1 - 0.001868040786) / 0.006218119096268822   # +0.8%  lam1 > 0.001868
        - 0.007931510598374153 * max(0.0, Q.z_dr_0_0p05 - 0.767473447323) / 0.05221087980608241   # -0.8%  z_dr_0_0p05 > 0.7675
        - 0.007711108497686184 * max(0.0, 0.002197764741 - Q.girth2_top15) / 0.00045099932175814557   # -0.8%  girth2_top15 < 0.002198
        - 0.006834412984524535 * max(0.0, 6.903422848462 - Q.log_sum_pt) / 0.015438763327039659   # -0.7%  log_sum_pt < 6.903
        - 0.003908508196563438 * max(0.0, Q.sum_pt_top15 - 935.104296875) / 29.657747695589798   # -0.4%  sum_pt_top15 > 935.1
        + 0.0038834240366301884 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, Q.mass_top3 - 16.899120053094) / 125.1697040938471   # +0.4%  mass < 120.6 and mass_top3 > 16.9
        - 0.00379174135431088 * max(0.0, Q.mass_top50 - 160.8) / 1.2464608384684717   # -0.4%  mass_top50 > 160.8
        + 0.0028189877056290467 * max(0.0, Q.mass_top50 - 157.544814151857) / 1.5571187949287246   # +0.3%  mass_top50 > 157.5
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 9.505;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.50546490247976 * (0.023949562963553474
        + 0.10860665136486472 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # +10.9%  mass < 92.86
        - 0.06779447441965795 * max(0.0, 0.006716736591 - Q.lam1) / 0.0019121054678470912   # -6.8%  lam1 < 0.006717
        - 0.05925938223341647 * max(0.0, 0.007538018543 - Q.girth2_top20) / 0.0027408407225167327   # -5.9%  girth2_top20 < 0.007538
        + 0.05815131960138664 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +5.8%  mass < 101
        - 0.05061109186520092 * max(0.0, 0.038759447634 - Q.e2) / 0.01184118794270979   # -5.1%  e2 < 0.03876
        - 0.04857219854412401 * max(0.0, 0.076787002012 - Q.girth) / 0.0210526739485986   # -4.9%  girth < 0.07679
        - 0.04806424301798082 * max(0.0, 86.4 - Q.mass_top50) / 14.253133905303311   # -4.8%  mass_top50 < 86.4
        + 0.04604220151439998 * max(0.0, 0.371981271173 - Q.LHA) / 0.11707573392405227   # +4.6%  LHA < 0.372
        + 0.04213942109538517 * max(0.0, 0.027935993578 - Q.e2) / 0.005715276962294592   # +4.2%  e2 < 0.02794
        + 0.04145957813479795 * max(0.0, 0.008184367501 - Q.mass_over_sum_pt_sq) / 0.0024516083597474207   # +4.1%  mass_over_sum_pt_sq < 0.008184
        + 0.037374137813310754 * max(0.0, 0.061710142531 - Q.girth) / 0.012950562309845613   # +3.7%  girth < 0.06171
        + 0.03219448007654721 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # +3.2%  girth < 0.08589
        - 0.03127132633210957 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -3.1%  mass < 80.4
        - 0.02802762352319394 * max(0.0, 0.006363915755 - Q.girth2_top30) / 0.001678623919074012   # -2.8%  girth2_top30 < 0.006364
        + 0.02781571797245177 * max(0.0, 0.005718442372 - Q.girth2_top20) / 0.0016279345755614207   # +2.8%  girth2_top20 < 0.005718
        + 0.027507732838959912 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581344537815   # +2.8%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        - 0.026880985646146296 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -2.7%  mass < 62.55
        - 0.026465241707177813 * max(0.0, 0.309838384344 - Q.LHA) / 0.06737135436028574   # -2.6%  LHA < 0.3098
        - 0.025891984562501566 * max(0.0, 80.4 - Q.mass) * max(0.0, 0.037001823448 - Q.z_dr_0p2_0p4) / 0.30159836215099584   # -2.6%  mass < 80.4 and z_dr_0p2_0p4 < 0.037
        - 0.023241394461957245 * max(0.0, 0.006170281901 - Q.width) / 0.0012963875716651668   # -2.3%  width < 0.00617
        + 0.023101447825410047 * max(0.0, 60.438206617337 - Q.mass_top30) / 6.992028296799824   # +2.3%  mass_top30 < 60.44
        + 0.022833365273873918 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.334047731757 - Q.z_dr_0p1_0p2) / 6.53582113499238   # +2.3%  mass < 101 and z_dr_0p1_0p2 < 0.334
        + 0.02126576392402128 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.02647292763 - Q.z_dr_0p2_0p4) / 0.39377280379101015   # +2.1%  mass < 101 and z_dr_0p2_0p4 < 0.02647
        - 0.014109842236495064 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.334047731757 - Q.z_dr_0p1_0p2) / 0.7658179227746299   # -1.4%  n_dr_0p2_0p4 < 10 and z_dr_0p1_0p2 < 0.334
        - 0.012430381811865525 * max(0.0, Q.z_top5_slots - 0.534625950898) / 0.08390497481820398   # -1.2%  z_top5_slots > 0.5346
        + 0.01130029890894705 * max(0.0, 86.4 - Q.mass_top50) * max(0.0, 0.00416995399 - Q.girth2_top15) / 0.04492769249291807   # +1.1%  mass_top50 < 86.4 and girth2_top15 < 0.00417
        + 0.010991529883779706 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104090756302521   # +1.1%  n_dr_0p2_0p4 < 5
        - 0.01033382941288209 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208341 - Q.girth2) / 0.005195774395551462   # -1.0%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        + 0.008941324426371034 * max(0.0, 0.005809484705 - Q.girth2_top30) / 0.0014021356702027855   # +0.9%  girth2_top30 < 0.005809
        - 0.007321029570783302 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 867.9265625 - Q.sum_pt_top10) / 1773.7051945073856   # -0.7%  mass < 101 and sum_pt_top10 < 867.9
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 7.348;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.3478859063953506 * (5.4645274520760036e-05
        - 0.12034770485215052 * max(0.0, 82.85408782959 - Q.mass) * max(0.0, Q.z_dr_0_0p05 - 0.328923654556) / 6.821195575793219   # -12.0%  mass < 82.85 and z_dr_0_0p05 > 0.3289
        + 0.11917678326120737 * max(0.0, 82.85408782959 - Q.mass) / 11.83453618826746   # +11.9%  mass < 82.85
        - 0.09172565443593418 * max(0.0, 77.376408295162 - Q.mass_top50) / 9.980854903106312   # -9.2%  mass_top50 < 77.38
        + 0.0669441536486316 * max(0.0, 78.261818313599 - Q.mass) / 9.857172988469275   # +6.7%  mass < 78.26
        - 0.0669368327507054 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -6.7%  mass < 62.55
        - 0.06377938065893987 * max(0.0, 0.050483809784 - Q.girth) / 0.008533024989277627   # -6.4%  girth < 0.05048
        + 0.05295121994798221 * max(0.0, 80.4 - Q.mass_top50) * max(0.0, 0.64696790278 - Q.z_dr_0p05_0p1) / 6.5876077202862655   # +5.3%  mass_top50 < 80.4 and z_dr_0p05_0p1 < 0.647
        + 0.04627929520901079 * max(0.0, 80.4 - Q.mass_top50) / 11.159877357156788   # +4.6%  mass_top50 < 80.4
        + 0.04316985268847631 * max(0.0, 82.85408782959 - Q.mass) * max(0.0, 0.64696790278 - Q.z_dr_0p05_0p1) / 6.926438037736949   # +4.3%  mass < 82.85 and z_dr_0p05_0p1 < 0.647
        + 0.03414978472068 * max(0.0, 74.251806640625 - Q.mass) / 8.587064460866076   # +3.4%  mass < 74.25
        + 0.03172145821370755 * max(0.0, 77.376408295162 - Q.mass_top50) * max(0.0, 0.850921532512 - Q.z_dr_0p05_0p1) / 7.991806220965258   # +3.2%  mass_top50 < 77.38 and z_dr_0p05_0p1 < 0.8509
        - 0.028519333151725983 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -2.9%  mass < 86.4
        - 0.027970832598775724 * max(0.0, 0.050483809784 - Q.girth) * max(0.0, 64.0 - Q.n_particles) / 0.24223643538921044   # -2.8%  girth < 0.05048 and n_particles < 64
        + 0.023314605438891058 * max(0.0, 0.228402115913 - Q.LHA) / 0.027166566418098437   # +2.3%  LHA < 0.2284
        - 0.022656819673493096 * max(0.0, 89.67879517394 - Q.mass_top40) * max(0.0, Q.n_particles - 22.0) / 314.97281238932567   # -2.3%  mass_top40 < 89.68 and n_particles > 22
        - 0.02084256589582069 * max(0.0, Q.planar_flow - 0.258818254187) / 0.2699773591449117   # -2.1%  planar_flow > 0.2588
        + 0.020219855439081 * max(0.0, 0.006936724595 - Q.e2_sq) / 0.0016886136779624735   # +2.0%  e2_sq < 0.006937
        - 0.018261692434391887 * max(0.0, Q.C2 - 0.061168736406) / 0.01742346976710858   # -1.8%  C2 > 0.06117
        - 0.017153111799323186 * max(0.0, 6.856374501323 - Q.log_sum_pt) / 0.008053458833961623   # -1.7%  log_sum_pt < 6.856
        + 0.01601954941938654 * max(0.0, 0.001000990214 - Q.girth2_top5) / 0.00025088670724764367   # +1.6%  girth2_top5 < 0.001001
        - 0.012746335429960162 * max(0.0, 0.000615484055 - Q.lam2) / 0.00014995048124430673   # -1.3%  lam2 < 0.0006155
        + 0.012509193470664077 * max(0.0, 43.666010696263 - Q.mass_top50) / 1.8550645269553439   # +1.3%  mass_top50 < 43.67
        + 0.012208478167493719 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.390781164169 - Q.max_dr) / 0.61981541359264   # +1.2%  mass < 86.4 and max_dr < 0.3908
        + 0.012040896756927735 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # +1.2%  mass > 143.8
        - 0.010637459924067895 * max(0.0, 43.666010696263 - Q.mass_top50) * max(0.0, 0.43572281599 - Q.max_dr) / 0.1851503022760109   # -1.1%  mass_top50 < 43.67 and max_dr < 0.4357
        + 0.004250076679377734 * max(0.0, 86.4 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.299250295758) / 0.29970876190808693   # +0.4%  mass < 86.4 and z_dr_0p05_0p1 > 0.2993
        - 0.003467073333193678 * max(0.0, Q.mass_top20 - 125.1) * max(0.0, Q.C2 - 0.056027559564) / 0.06998404251378426   # -0.3%  mass_top20 > 125.1 and C2 > 0.05603
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 11.66;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.658718966831286 * (0.1391520461304907
        + 0.06693369161306997 * max(0.0, 0.013978743181 - Q.girth2) / 0.006902497325099419   # +6.7%  girth2 < 0.01398
        - 0.053251496102620824 * max(0.0, Q.mass - 160.8) / 1.7246633179459985   # -5.3%  mass > 160.8
        + 0.05098674731154529 * max(0.0, Q.girth - 0.050483809784) / 0.027293796159757064   # +5.1%  girth > 0.05048
        - 0.05067393254898705 * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.08975042679559622   # -5.1%  z_dr_0p2_0p4 < 0.1292
        - 0.049878894794440534 * max(0.0, Q.LHA - 0.228402115913) / 0.06082067856197019   # -5.0%  LHA > 0.2284
        + 0.044869401971629294 * Q.n_particles / 45.8152268907563   # +4.5%  n_particles
        - 0.04264549741824446 * max(0.0, 1052.889428710937 - Q.sum_pt) / 42.62464270793601   # -4.3%  sum_pt < 1053
        - 0.03642585542017641 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # -3.6%  girth < 0.1207
        - 0.03575295784120052 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -3.6%  mass > 143.8
        + 0.033333697973004374 * max(0.0, 1053.04736328125 - Q.sum_pt_top40) / 57.73987569631368   # +3.3%  sum_pt_top40 < 1053
        + 0.029669495592305 * max(0.0, 0.009606007381 - Q.e2_sq) / 0.003497865944155776   # +3.0%  e2_sq < 0.009606
        - 0.02952215877001706 * max(0.0, Q.mass_top50 - 97.930041729355) / 11.917644068610143   # -3.0%  mass_top50 > 97.93
        - 0.028092085139294367 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # -2.8%  sum_pt < 1013
        + 0.025763148663764886 * max(0.0, 1013.915698242188 - Q.sum_pt_top50) / 24.159122871194317   # +2.6%  sum_pt_top50 < 1014
        + 0.025456867047813872 * max(0.0, Q.mass - 62.55) / 32.65527990955991   # +2.5%  mass > 62.55
        + 0.024725083707773828 * max(0.0, Q.mass - 74.251806640625) / 24.076479363593222   # +2.5%  mass > 74.25
        + 0.024236474406627996 * max(0.0, Q.mass_top40 - 136.785) / 3.2712129356961093   # +2.4%  mass_top40 > 136.8
        - 0.02410877125347557 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # -2.4%  mass > 172.8
        - 0.023156818646214564 * max(0.0, Q.z_top30_slots - 0.946075126916) / 0.026010155218494452   # -2.3%  z_top30_slots > 0.9461
        + 0.01845154288372337 * max(0.0, 988.455444335938 - Q.sum_pt_top50) / 15.571236316636032   # +1.8%  sum_pt_top50 < 988.5
        - 0.01812862141857043 * max(0.0, Q.girth2_top40 - 0.00663016737) / 0.003919751403490436   # -1.8%  girth2_top40 > 0.00663
        - 0.018034079692068546 * max(0.0, 6.903422848462 - Q.log_sum_pt) / 0.015438763327039659   # -1.8%  log_sum_pt < 6.903
        + 0.017804038601503164 * max(0.0, Q.mass_over_sum_pt - 0.170880120467) / 0.0005536193928784717   # +1.8%  mass_over_sum_pt > 0.1709
        + 0.017046581401225242 * max(0.0, Q.mass_top50 - 136.785) / 4.25098606820319   # +1.7%  mass_top50 > 136.8
        + 0.015591790764092315 * max(0.0, Q.mass - 62.55) * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 373.2570980937201   # +1.6%  mass > 62.55 and n_dr_0p2_0p4 < 26
        + 0.015467624066859099 * max(0.0, Q.mass - 160.8) * max(0.0, 5.378974604607 - Q.D2) / 5.667716793316475   # +1.5%  mass > 160.8 and D2 < 5.379
        - 0.014344667121595987 * max(0.0, 956.21328125 - Q.sum_pt_top40) / 14.004510341849162   # -1.4%  sum_pt_top40 < 956.2
        - 0.014048816940697352 * max(0.0, Q.mass - 120.60000000000001) / 7.969125673678914   # -1.4%  mass > 120.6
        - 0.013323553157718137 * max(0.0, Q.mass - 172.8) * max(0.0, 62.25 - Q.pt_6) / 13.368505683744926   # -1.3%  mass > 172.8 and pt_6 < 62.25
        + 0.011567492861199688 * max(0.0, Q.mass_over_sum_pt - 0.078528833221) / 0.0193369167748192   # +1.2%  mass_over_sum_pt > 0.07853
        + 0.011131536813994779 * max(0.0, 959.095727539062 - Q.sum_pt_top50) / 9.937940549204788   # +1.1%  sum_pt_top50 < 959.1
        - 0.01051823103983388 * max(0.0, Q.mass_over_sum_pt_sq - 0.029200015571) / 0.00020282738822808687   # -1.1%  mass_over_sum_pt_sq > 0.0292
        - 0.009271182810771045 * max(0.0, 966.06328125 - Q.sum_pt_top30) / 29.999231571738342   # -0.9%  sum_pt_top30 < 966.1
        - 0.009036112393031348 * max(0.0, Q.sum_pt_top20 - 956.50615234375) * max(0.0, 0.071710390673 - Q.dr_0) / 1.669913822919429   # -0.9%  sum_pt_top20 > 956.5 and dr_0 < 0.07171
        - 0.008632665964350422 * max(0.0, Q.mass - 74.251806640625) * max(0.0, Q.D2 - 0.603279101849) / 37.923504307700206   # -0.9%  mass > 74.25 and D2 > 0.6033
        + 0.007893646102104242 * max(0.0, Q.mass - 162.836349487305) / 1.5098322961374488   # +0.8%  mass > 162.8
        + 0.006627500369557584 * max(0.0, Q.mass_top5 - 37.764547629628) / 5.869287554842667   # +0.7%  mass_top5 > 37.76
        - 0.006419904340766945 * max(0.0, 1013.915698242188 - Q.sum_pt_top50) * max(0.0, 4.450168704987 - Q.D2) / 41.241641900364776   # -0.6%  sum_pt_top50 < 1014 and D2 < 4.45
        - 0.006158112599797849 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, 26.0 - Q.n_dr_0p1_0p2) / 242.9521402805676   # -0.6%  sum_pt < 1013 and n_dr_0p1_0p2 < 26
        - 0.006126457368537189 * max(0.0, 986.05654296875 - Q.sum_pt) / 11.984374324425932   # -0.6%  sum_pt < 986.1
        + 0.005826140506938081 * max(0.0, Q.sum_pt_top30 - 1111.24501953125) * max(0.0, 0.093385871589 - Q.dr_0) / 0.8369045086275705   # +0.6%  sum_pt_top30 > 1111 and dr_0 < 0.09339
        - 0.005769166942358072 * max(0.0, Q.mass_top15 - 86.4) / 4.307561877411397   # -0.6%  mass_top15 > 86.4
        + 0.005676063728008428 * max(0.0, Q.mass - 172.8) * max(0.0, 0.042471339685 - Q.z_6) / 0.004823052626121717   # +0.6%  mass > 172.8 and z_6 < 0.04247
        + 0.005109704771493776 * max(0.0, 6.811175180312 - Q.log_sum_pt) / 0.004882419406465069   # +0.5%  log_sum_pt < 6.811
        - 0.004844879033157755 * max(0.0, Q.sum_pt_top30 - 1111.24501953125) / 12.625727406952   # -0.5%  sum_pt_top30 > 1111
        - 0.0041470700001391475 * max(0.0, 1085.12490234375 - Q.sum_pt) * max(0.0, 800.73203125 - Q.sum_pt_top30) / 1062.2137747974023   # -0.4%  sum_pt < 1085 and sum_pt_top30 < 800.7
        + 0.0031195502741659825 * max(0.0, Q.mass_top50 - 160.8) / 1.2464608384684717   # +0.3%  mass_top50 > 160.8
        - 0.002693799931905082 * max(0.0, Q.mass_top50 - 168.969765712694) / 0.6651774989835844   # -0.3%  mass_top50 > 169
        - 0.0026600784426973425 * max(0.0, 6.811175180312 - Q.log_sum_pt) * max(0.0, 0.051086217058 - Q.dr_5) / 7.371257739198046e-05   # -0.3%  log_sum_pt < 6.811 and dr_5 < 0.05109
        - 0.002064694736517245 * max(0.0, Q.sum_pt - 1260.540869140625) * max(0.0, 0.111955475493 - Q.dr_0) / 0.6070481398991538   # -0.2%  sum_pt > 1261 and dr_0 < 0.112
        + 0.0015350743203271595 * max(0.0, Q.sum_pt - 1260.540869140625) / 7.65548318400563   # +0.2%  sum_pt > 1261
        - 0.0014465423780874048 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.21247899159663866   # -0.1%  n_dr_0_0p05 > 30
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 25.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.572932388880684 * (0.016592877552818824
        - 0.0898466087322483 * max(0.0, 0.090467494167 - Q.mass_over_sum_pt) / 0.0178873033081874   # -9.0%  mass_over_sum_pt < 0.09047
        + 0.07183625068932711 * max(0.0, 0.118225939153 - Q.mass_over_sum_pt) / 0.039171218898223295   # +7.2%  mass_over_sum_pt < 0.1182
        + 0.07149576774988543 * max(0.0, 0.09795414517 - Q.mass_over_sum_pt) / 0.023392533955001828   # +7.1%  mass_over_sum_pt < 0.09795
        - 0.04881061090534275 * max(0.0, 89.741833496094 - Q.mass) / 15.556328292327747   # -4.9%  mass < 89.74
        + 0.047700568419471005 * max(0.0, 0.140939019879 - Q.mass_over_sum_pt) / 0.057960363993862694   # +4.8%  mass_over_sum_pt < 0.1409
        + 0.04460682739540403 * max(0.0, 89.080094718389 - Q.mass_top50) / 15.79328170065322   # +4.5%  mass_top50 < 89.08
        - 0.04346533712750544 * max(0.0, 92.165451466106 - Q.mass_top50) / 17.798728151002944   # -4.3%  mass_top50 < 92.17
        - 0.03961676322672126 * max(0.0, 91.034691238403 - Q.mass) / 16.362866628552233   # -4.0%  mass < 91.03
        - 0.038658610695869386 * max(0.0, 0.009262053166 - Q.girth2_top50) / 0.0033075364251965024   # -3.9%  girth2_top50 < 0.009262
        + 0.03330256667292929 * max(0.0, 136.785 - Q.mass) / 52.08717491697615   # +3.3%  mass < 136.8
        - 0.032815854943221826 * max(0.0, 0.012157872869 - Q.girth2_top30) / 0.005935411821644211   # -3.3%  girth2_top30 < 0.01216
        - 0.02976050046013436 * max(0.0, 0.083299446175 - Q.mass_over_sum_pt) / 0.013509300202577342   # -3.0%  mass_over_sum_pt < 0.0833
        + 0.029008010370008018 * max(0.0, 0.007259287357 - Q.lam1) / 0.0022497701299997253   # +2.9%  lam1 < 0.007259
        - 0.028551130879530484 * max(0.0, 111.24867219155 - Q.mass_top40) / 33.99435635629078   # -2.9%  mass_top40 < 111.2
        - 0.028178687113024375 * max(0.0, 0.008241985248 - Q.lam1) / 0.0029450268440803277   # -2.8%  lam1 < 0.008242
        - 0.025547158052004738 * max(0.0, Q.sum_pt_top50 - 976.277001953125) / 72.2927679781981   # -2.6%  sum_pt_top50 > 976.3
        - 0.025288558764569976 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -2.5%  mass < 80.4
        - 0.022000413613322885 * max(0.0, Q.max_dr - 0.240474711359) / 0.11773588635266226   # -2.2%  max_dr > 0.2405
        + 0.02158466425798948 * max(0.0, 0.002396991421 - Q.lam2) / 0.0014662533145241962   # +2.2%  lam2 < 0.002397
        + 0.02044190929530881 * max(0.0, 91.034691238403 - Q.mass) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 2.9734071289695474   # +2.0%  mass < 91.03 and z_dr_0p1_0p2 < 0.2188
        + 0.02041359167436862 * max(0.0, 0.007877041167 - Q.girth2) / 0.0022422969111317915   # +2.0%  girth2 < 0.007877
        - 0.019401248313546178 * max(0.0, 0.010834353386 - Q.girth2_top20) / 0.005298250100903007   # -1.9%  girth2_top20 < 0.01083
        + 0.0188656374680229 * max(0.0, 91.288535717504 - Q.mass_top40) / 18.56549606317863   # +1.9%  mass_top40 < 91.29
        + 0.017901857506521405 * max(0.0, Q.log_sum_pt - 6.935549248787) / 0.03715380406194132   # +1.8%  log_sum_pt > 6.936
        - 0.016134862503726848 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 137.24421512605042   # -1.6%  n_dr_0p2_0p4 < 21 and n_dr_0p1_0p2 < 21
        + 0.013092289510263571 * max(0.0, 0.005718442372 - Q.girth2_top20) / 0.0016279345755614207   # +1.3%  girth2_top20 < 0.005718
        - 0.012144400876554969 * max(0.0, 80.890431271924 - Q.mass_top40) / 12.434053057794845   # -1.2%  mass_top40 < 80.89
        + 0.011955294560588851 * max(0.0, 117.048742792994 - Q.mass_top50) / 36.94196179171718   # +1.2%  mass_top50 < 117
        + 0.010225016767903817 * max(0.0, 0.00608841615 - Q.girth2_top30) / 0.0015338919490682538   # +1.0%  girth2_top30 < 0.006088
        + 0.010176671239497554 * max(0.0, 0.006876086349 - Q.girth2_top10) / 0.0028923874793807566   # +1.0%  girth2_top10 < 0.006876
        + 0.010015440546208607 * max(0.0, 67.726432644245 - Q.mass_top40) / 7.7029388850516805   # +1.0%  mass_top40 < 67.73
        - 0.008924900958732794 * max(0.0, 0.016559833876 - Q.girth2_top20) * max(0.0, Q.z_top50_slots - 0.970443639316) / 0.000256897965306318   # -0.9%  girth2_top20 < 0.01656 and z_top50_slots > 0.9704
        - 0.007136850099946975 * max(0.0, 45.595 - Q.mass_top10) / 11.863531158412167   # -0.7%  mass_top10 < 45.59
        - 0.0069183345034613325 * max(0.0, 0.001163277284 - Q.lam2) * max(0.0, 0.061084209235 - Q.dr_1) / 1.3408748634295094e-05   # -0.7%  lam2 < 0.001163 and dr_1 < 0.06108
        - 0.006568100944157195 * max(0.0, 0.011744050682 - Q.lam1) / 0.005678958164694959   # -0.7%  lam1 < 0.01174
        + 0.004732128696338664 * max(0.0, 0.006189818106 - Q.lam1) * max(0.0, Q.z_top50_slots - 0.985099030959) / 2.0496011611642913e-05   # +0.5%  lam1 < 0.00619 and z_top50_slots > 0.9851
        - 0.0034863262103996383 * max(0.0, 0.331585738063 - Q.max_dr) / 0.021666863543587846   # -0.3%  max_dr < 0.3316
        + 0.003324646152370792 * max(0.0, Q.max_dr - 0.240474711359) * max(0.0, 56.921923720802 - Q.mass_top10) / 2.2920324246063815   # +0.3%  max_dr > 0.2405 and mass_top10 < 56.92
        - 0.003021135823425055 * max(0.0, Q.sum_pt_top50 - 976.277001953125) * max(0.0, Q.eccentricity - 0.948956476603) / 0.2652630316967096   # -0.3%  sum_pt_top50 > 976.3 and eccentricity > 0.949
        + 0.0016783442964037438 * max(0.0, Q.log_sum_pt - 6.935549248787) * max(0.0, Q.eccentricity - 0.948956476603) / 0.00010163721555608554   # +0.2%  log_sum_pt > 6.936 and eccentricity > 0.949
        + 0.0013661219837415707 * max(0.0, Q.max_dr - 0.402178311348) / 0.010008869835123797   # +0.1%  max_dr > 0.4022
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 9.774;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.773546313128774 * (-0.06487735569057228
        + 0.10904781786823788 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # +10.9%  girth < 0.1207
        + 0.09354549093170052 * max(0.0, 0.006936724595 - Q.e2_sq) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.00019974727848153987   # +9.4%  e2_sq < 0.006937 and z_dr_0p2_0p4 < 0.1292
        - 0.08985871857754002 * max(0.0, 0.006936724595 - Q.e2_sq) / 0.0016886136779624735   # -9.0%  e2_sq < 0.006937
        - 0.08345809379822339 * max(0.0, 0.120745175332 - Q.girth) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.005981775287783369   # -8.3%  girth < 0.1207 and z_dr_0p2_0p4 < 0.1292
        + 0.0738470723561127 * max(0.0, 0.040828318335 - Q.e2) / 0.01334159186495724   # +7.4%  e2 < 0.04083
        - 0.061759321378611814 * max(0.0, 0.016493544356 - Q.lam1) / 0.009604222881411155   # -6.2%  lam1 < 0.01649
        + 0.056884575305701936 * max(0.0, 0.001501708498 - Q.girth2_top5) * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 7.533004185945094e-05   # +5.7%  girth2_top5 < 0.001502 and z_dr_0p2_0p4 < 0.1937
        - 0.04837688126616623 * max(0.0, 0.001501708498 - Q.girth2_top5) / 0.0004379639090356587   # -4.8%  girth2_top5 < 0.001502
        + 0.03864474653106826 * max(0.0, 0.001776308492 - Q.lam2) / 0.0009519164072663888   # +3.9%  lam2 < 0.001776
        + 0.03322496493560127 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # +3.3%  log_sum_pt < 7.017
        - 0.032229084164650285 * max(0.0, 0.056600876898 - Q.girth) / 0.010803815022120036   # -3.2%  girth < 0.0566
        - 0.032112076208163365 * max(0.0, 1002.378515625 - Q.sum_pt) / 15.957354754845854   # -3.2%  sum_pt < 1002
        + 0.026858809289515107 * max(0.0, 0.006189818106 - Q.lam1) / 0.0016087780966108156   # +2.7%  lam1 < 0.00619
        + 0.024128494136107494 * max(0.0, 986.05654296875 - Q.sum_pt) / 11.984374324425932   # +2.4%  sum_pt < 986.1
        + 0.024027470229254094 * max(0.0, 0.187280465662 - Q.z_dr_0p1_0p2) / 0.08737974006182235   # +2.4%  z_dr_0p1_0p2 < 0.1873
        + 0.022831241104976373 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 40.2 - Q.mass_top5) / 1.3263398907230615   # +2.3%  z_dr_0p1_0p2 < 0.1203 and mass_top5 < 40.2
        - 0.019228671901989482 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 0.051804735139 - Q.z_dr_0p2_0p4) / 0.0014190964277842602   # -1.9%  z_dr_0p1_0p2 < 0.1203 and z_dr_0p2_0p4 < 0.0518
        + 0.014615382677839923 * max(0.0, 13.0 - Q.n_dr_0p1_0p2) / 3.8210403361344536   # +1.5%  n_dr_0p1_0p2 < 13
        - 0.014434518204132782 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # -1.4%  mass_top50 < 71.8
        + 0.014406610252324531 * max(0.0, 0.013514311784 - Q.girth2_top50) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.0007407123072106741   # +1.4%  girth2_top50 < 0.01351 and z_dr_0p2_0p4 < 0.1292
        - 0.012384687844556043 * max(0.0, 0.012157872869 - Q.girth2_top30) * max(0.0, Q.z_dr_0p2_0p4 - 0.02647292763) / 3.9127155542309244e-05   # -1.2%  girth2_top30 < 0.01216 and z_dr_0p2_0p4 > 0.02647
        - 0.01235126307167171 * max(0.0, 0.006936724595 - Q.e2_sq) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 0.0003126700564572212   # -1.2%  e2_sq < 0.006937 and z_dr_0p1_0p2 < 0.2188
        - 0.011279966664583693 * max(0.0, 0.052014814497 - Q.dr_0) / 0.017315448851188676   # -1.1%  dr_0 < 0.05201
        - 0.009559694707727719 * max(0.0, 62.55 - Q.mass_top30) / 7.633530034750348   # -1.0%  mass_top30 < 62.55
        - 0.00921217518610927 * max(0.0, 1038.26171875 - Q.sum_pt_top30) / 69.34980875418526   # -0.9%  sum_pt_top30 < 1038
        - 0.009202242378632184 * max(0.0, 0.845900350809 - Q.z_dr_0_0p05) * max(0.0, 0.00339853589 - Q.z_dr_0p2_0p4) / 0.0002787951475897795   # -0.9%  z_dr_0_0p05 < 0.8459 and z_dr_0p2_0p4 < 0.003399
        + 0.00812772570376624 * max(0.0, 80.4 - Q.mass_top30) / 14.655773101172624   # +0.8%  mass_top30 < 80.4
        - 0.006749933484800238 * max(0.0, 0.001501708498 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.0032906615041510393   # -0.7%  girth2_top5 < 0.001502 and n_dr_0p2_0p4 > 0
        - 0.003663515267082932 * max(0.0, 0.007164202106 - Q.girth2_top5) * max(0.0, 984.70087890625 - Q.sum_pt_top40) / 0.05379473153393668   # -0.4%  girth2_top5 < 0.007164 and sum_pt_top40 < 984.7
        - 0.0020508167142510862 * max(0.0, 934.241552734375 - Q.sum_pt_top50) / 6.932863593743613   # -0.2%  sum_pt_top50 < 934.2
        - 0.0018979378589013339 * max(0.0, 1002.378515625 - Q.sum_pt) * max(0.0, 0.019523000158 - Q.z_dr_0p2_0p4) / 0.057971995942360716   # -0.2%  sum_pt < 1002 and z_dr_0p2_0p4 < 0.01952
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.5265952205882353, 3.381909348739496, 1.0597422268907564, 0.5455203781512605, 0.7645404411764706, 1.521089600840336, 0.5661378151260504, 0.42326355042016806, 1.2798324579831932, 1.1895046218487395, 1.3152607142857142, 0.6295125, 0.21572836134453782, 1.5887335084033614, 0.3313264180672269, 0.370521743697479]
T = [4.480051414784665, 2.6439220768776264, 4.442242497702206, 4.813591153492646, 4.26553731535583]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +47%, n4 -15%, n9 +14%, n5 -11%, n3 -9%, n6 +2% ...
            + 0.4718011351358074 * h[1] / H_AVG[1]
            - 0.149322591214406 * h[4] / H_AVG[4]
            + 0.13690430395880548 * h[9] / H_AVG[9]
            - 0.10610157256095963 * h[5] / H_AVG[5]
            - 0.0913249080720898 * h[3] / H_AVG[3]
            + 0.019745093398151815 * h[6] / H_AVG[6]
            + 0.011285882685021887 * h[12] / H_AVG[12]
            + 0.008927300293917978 * h[8] / H_AVG[8]
            - 0.004587212680839752 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -31%, n9 +25%, n1 -24%, n11 -6%, n2 +5%, n6 +5% ...
            - 0.3072421180087594 * h[4] / H_AVG[4]
            + 0.25306961791404 * h[9] / H_AVG[9]
            - 0.23983611636448582 * h[1] / H_AVG[1]
            - 0.05952449445327743 * h[11] / H_AVG[11]
            + 0.05010275435870034 * h[2] / H_AVG[2]
            + 0.04684050567975781 * h[6] / H_AVG[6]
            + 0.02804796134527576 * h[12] / H_AVG[12]
            - 0.007772902552780296 * h[10] / H_AVG[10]
            + 0.007563529322922996 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -25%, n5 +20%, n14 -10%, n0 +9%, n11 +8%, n7 -6% ...
            - 0.25209191108197027 * h[8] / H_AVG[8]
            + 0.19795855943044247 * h[5] / H_AVG[5]
            - 0.10255492019584411 * h[14] / H_AVG[14]
            + 0.08890699137777067 * h[0] / H_AVG[0]
            + 0.08414062200979298 * h[11] / H_AVG[11]
            - 0.059550940577746674 * h[7] / H_AVG[7]
            + 0.05916173572027039 * h[4] / H_AVG[4]
            - 0.05857495086411986 * h[9] / H_AVG[9]
            + 0.05372628026602069 * h[3] / H_AVG[3]
            - 0.019728694874615912 * h[12] / H_AVG[12]
            - 0.015639134283914672 * h[15] / H_AVG[15]
            + 0.007965259317491261 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -25%, n5 +22%, n0 -15%, n6 -11%, n2 -8%, n7 +8% ...
            - 0.249261495440605 * h[8] / H_AVG[8]
            + 0.21724925678803367 * h[5] / H_AVG[5]
            - 0.15042167172495607 * h[0] / H_AVG[0]
            - 0.11393697364709086 * h[6] / H_AVG[6]
            - 0.08255859760663008 * h[2] / H_AVG[2]
            + 0.07968740600039483 * h[7] / H_AVG[7]
            + 0.04958151987378608 * h[3] / H_AVG[3]
            + 0.04329791920084605 * h[15] / H_AVG[15]
            + 0.014005159717657575 * h[12] / H_AVG[12]
        ),
        0.78125 + T[4] * (   # class t: n13 -34%, n10 +30%, n5 -17%, n8 +6%, n4 +3%, n15 -3% ...
            - 0.3375400648371633 * h[13] / H_AVG[13]
            + 0.30352794264958755 * h[10] / H_AVG[10]
            - 0.16715613946854627 * h[5] / H_AVG[5]
            + 0.06328971924216058 * h[8] / H_AVG[8]
            + 0.03360686406482179 * h[4] / H_AVG[4]
            - 0.03257400970010359 * h[15] / H_AVG[15]
            - 0.02790806052197008 * h[7] / H_AVG[7]
            - 0.018965520524922005 * h[12] / H_AVG[12]
            + 0.015431678990724843 * h[0] / H_AVG[0]
        ),
    ]]


def classify(pt, eta, phi):
    s = logits(jet_layer_4(quantities(pt, eta, phi)))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3][:64] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4][:64] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36][:64] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
