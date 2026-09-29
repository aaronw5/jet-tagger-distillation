"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  5:  14.4%   (on for 89% of jets)
  neuron  8:  12.8%   (on for 49% of jets)
  neuron  1:  12.8%   (on for 99% of jets)
  neuron  4:   8.4%   (on for 61% of jets)
  neuron  0:   8.2%   (on for 60% of jets)
  neuron  9:   8.1%   (on for 76% of jets)
  neuron 13:   7.0%   (on for 88% of jets)
  neuron 10:   6.7%   (on for 77% of jets)
  neuron 11:   3.8%   (on for 68% of jets)
  neuron  3:   3.4%   (on for 50% of jets)
  neuron  7:   3.3%   (on for 25% of jets)
  neuron 12:   2.8%   (on for 41% of jets)
  neuron 14:   2.6%   (on for 47% of jets)
  neuron  6:   2.5%   (on for 41% of jets)
  neuron 15:   2.2%   (on for 66% of jets)
  neuron  2:   0.8%   (on for 55% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 93.7% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_0                    pT of particle 0 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.z_1st                  largest pT share
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_5                  Δη of particle 5
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
  Q.mean_phi               pT-weighted mean Δφ
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
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_4=pt[4],
        pt_7=pt[7],
        pt_9=pt[9],
        z_0=z[0],
        z_2=z[2],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        z_1st=zs[0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_5=eta[5],
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
        mean_phi=sum(z[i] * phi[i] for i in P),
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
    # scale S = 13.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.292533513839288 * (0.10425647344864195
        - 0.20984810901861725 * max(0.0, Q.mass - 78.261818313599) / 21.336576218229133   # -21.0%  mass > 78.26
        - 0.13720684367421404 * max(0.0, Q.mass - 92.85979309082) / 14.482733665646967   # -13.7%  mass > 92.86
        + 0.1062777534134759 * max(0.0, Q.mass_over_sum_pt - 0.083299446175) / 0.01700043986417834   # +10.6%  mass_over_sum_pt > 0.0833
        - 0.09674537443812788 * max(0.0, Q.mass - 91.034691238403) / 15.06939693349851   # -9.7%  mass > 91.03
        + 0.08892434699953554 * max(0.0, 0.008184367501 - Q.mass_over_sum_pt_sq) / 0.0024516083597474207   # +8.9%  mass_over_sum_pt_sq < 0.008184
        - 0.06567585274853197 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) / 0.020241059768600602   # -6.6%  mass_over_sum_pt > 0.07697
        - 0.050712082048423666 * max(0.0, 0.056600876898 - Q.girth) / 0.010803815022120036   # -5.1%  girth < 0.0566
        - 0.04969351032271727 * max(0.0, 0.007538018543 - Q.girth2_top20) / 0.0027408407225167327   # -5.0%  girth2_top20 < 0.007538
        + 0.024921759727407122 * max(0.0, 0.005312783396 - Q.girth2_top20) / 0.00143721407379855   # +2.5%  girth2_top20 < 0.005313
        - 0.023768479161288932 * max(0.0, 0.005913554513 - Q.lam1) / 0.0014639240999900342   # -2.4%  lam1 < 0.005914
        + 0.02243016713231239 * max(0.0, 0.00742997247 - Q.girth2_top50) / 0.0020221555597961793   # +2.2%  girth2_top50 < 0.00743
        + 0.02201197969369382 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # +2.2%  log_sum_pt < 7.017
        - 0.02052956454872947 * max(0.0, 0.00625977218 - Q.girth2_top40) / 0.0014618229210262657   # -2.1%  girth2_top40 < 0.00626
        + 0.018425242171928008 * max(0.0, 0.00625977218 - Q.girth2_top40) * max(0.0, 0.002915531053 - Q.girth2_top3) / 3.6946461207137186e-06   # +1.8%  girth2_top40 < 0.00626 and girth2_top3 < 0.002916
        + 0.013574477293428954 * max(0.0, 0.260146178237 - Q.LHA) / 0.039673711668022284   # +1.4%  LHA < 0.2601
        - 0.013030888163838017 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # -1.3%  sum_pt < 1013
        - 0.010502776059043135 * max(0.0, 0.006043208873 - Q.girth2_top20) / 0.0017975224219569485   # -1.1%  girth2_top20 < 0.006043
        - 0.008715789476198459 * max(0.0, 82.04491364955 - Q.mass_top50) / 11.943128521443713   # -0.9%  mass_top50 < 82.04
        - 0.008447272536478836 * max(0.0, 0.009614971338 - Q.width) / 0.003502544629791685   # -0.8%  width < 0.009615
        + 0.0056998899839686866 * max(0.0, Q.mass - 92.85979309082) * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 1.7278724386184863   # +0.6%  mass > 92.86 and n_dr_0p2_0p4 < 7
        - 0.0028578413880408836 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.088715460151) / 3.100400728225136   # -0.3%  sum_pt < 1013 and z_dr_0p1_0p2 > 0.08872
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 17.6;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.604543976855496 * (0.09954921507562801
        - 0.10123204403319606 * max(0.0, Q.z_top50_slots - 0.958653609576) / 0.03427111698918734   # -10.1%  z_top50_slots > 0.9587
        + 0.08848810425805677 * max(0.0, Q.log_sum_pt - 6.89371369877) / 0.06417423647849095   # +8.8%  log_sum_pt > 6.894
        + 0.07290233321079452 * max(0.0, 0.404204003833 - Q.LHA) / 0.14530548997384415   # +7.3%  LHA < 0.4042
        + 0.07085772128830853 * max(0.0, Q.log_sum_pt - 6.910130970417) / 0.051750869608046556   # +7.1%  log_sum_pt > 6.91
        + 0.06520427614499655 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.27338014705884   # +6.5%  sum_pt_top2 < 689.2
        - 0.050113772383870304 * max(0.0, 0.957678701144 - Q.z_top20_slots) / 0.07565544239103991   # -5.0%  z_top20_slots < 0.9577
        + 0.04601528781271444 * max(0.0, 80.4 - Q.mass_top30) / 14.655773101172624   # +4.6%  mass_top30 < 80.4
        - 0.042398422220416385 * max(0.0, Q.sum_pt_top50 - 959.095727539062) / 86.53899855624287   # -4.2%  sum_pt_top50 > 959.1
        - 0.03883407487316261 * max(0.0, Q.log_sum_pt - 6.811175180312) / 0.138314075182229   # -3.9%  log_sum_pt > 6.811
        - 0.036399978016836865 * max(0.0, 0.43572281599 - Q.max_dr) / 0.09036112201580485   # -3.6%  max_dr < 0.4357
        - 0.034907310894363185 * max(0.0, 117.048742792994 - Q.mass_top50) / 36.94196179171718   # -3.5%  mass_top50 < 117
        + 0.03484721664485169 * max(0.0, 902.40625 - Q.sum_pt_top5) / 313.0793622488839   # +3.5%  sum_pt_top5 < 902.4
        - 0.03384646301816602 * max(0.0, 31.0 - Q.n_pt_above_10) / 11.388505882352941   # -3.4%  n_pt_above_10 < 31
        - 0.03207798459642098 * max(0.0, 80.4 - Q.mass_top30) * max(0.0, 68.434224049685 - Q.mass_top5) / 891.7019121450287   # -3.2%  mass_top30 < 80.4 and mass_top5 < 68.43
        + 0.024668184759610277 * max(0.0, Q.n_particles - 38.0) / 10.799277310924369   # +2.5%  n_particles > 38
        + 0.024521452045049647 * max(0.0, 0.43572281599 - Q.max_dr) * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 0.0143775071962318   # +2.5%  max_dr < 0.4357 and z_dr_0p2_0p4 < 0.1937
        - 0.022796508960589438 * max(0.0, Q.log_sum_pt - 6.959293500649) / 0.028460039078747168   # -2.3%  log_sum_pt > 6.959
        + 0.018457396826329017 * max(0.0, 1073.4734375 - Q.sum_pt_top30) / 97.85231293739916   # +1.8%  sum_pt_top30 < 1073
        - 0.016925627880638477 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, 91.19 - Q.mass_top10) / 1.6455577605891802   # -1.7%  z_top30_slots > 0.9342 and mass_top10 < 91.19
        - 0.014965431973247291 * max(0.0, 40.2 - Q.mass_top20) / 3.9165662697562196   # -1.5%  mass_top20 < 40.2
        + 0.013892908573062924 * max(0.0, Q.z_top30_slots - 0.934183811419) / 0.034068179997830586   # +1.4%  z_top30_slots > 0.9342
        + 0.012223717738770127 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.144288109196 - Q.dr_0) / 0.8704373876584949   # +1.2%  n_particles > 38 and dr_0 < 0.1443
        - 0.012020100292164887 * max(0.0, 0.000829637219 - Q.lam2) / 0.000268103756224263   # -1.2%  lam2 < 0.0008296
        - 0.011959353782056568 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 621.4389484506303   # -1.2%  sum_pt_top2 < 689.2 and n_dr_0p2_0p4 < 7
        + 0.011716485653937939 * max(0.0, 31.0 - Q.n_pt_above_10) * max(0.0, 0.017422899418 - Q.mean_phi2) / 0.1577871548170423   # +1.2%  n_pt_above_10 < 31 and mean_phi2 < 0.01742
        + 0.010785930388788395 * max(0.0, 40.2 - Q.mass_top20) * max(0.0, 0.005860335776 - Q.mean_phi2) / 0.01972420536905745   # +1.1%  mass_top20 < 40.2 and mean_phi2 < 0.00586
        + 0.010527203348541537 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, Q.mass_top5 - 22.1834155076) / 0.4186571122099573   # +1.1%  z_top30_slots > 0.9342 and mass_top5 > 22.18
        - 0.009796806682582471 * max(0.0, Q.log_sum_pt - 6.989450376716) / 0.021142744446604637   # -1.0%  log_sum_pt > 6.989
        + 0.007021519464159439 * max(0.0, 0.000657050184 - Q.girth2_top5) / 0.0001395474066313899   # +0.7%  girth2_top5 < 0.0006571
        + 0.006522135511272536 * max(0.0, 0.00130722027 - Q.girth2_top10) / 0.00027941021155763345   # +0.7%  girth2_top10 < 0.001307
        + 0.006038887490903834 * max(0.0, 0.003270031267 - Q.girth2_top15) / 0.0007969964936058968   # +0.6%  girth2_top15 < 0.00327
        + 0.0060098181394590165 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.985099030959 - Q.z_top50_slots) / 0.09243197309849287   # +0.6%  n_particles > 38 and z_top50_slots < 0.9851
        + 0.005214487470610978 * max(0.0, 40.2 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 26.0) / 33.56032785174058   # +0.5%  mass_top20 < 40.2 and n_real_top40 > 26
        - 0.004252855964708083 * max(0.0, Q.z_dr_0_0p05 - 0.878906026483) / 0.015835254093555608   # -0.4%  z_dr_0_0p05 > 0.8789
        - 0.0015581976573622574 * max(0.0, Q.log_sum_pt - 7.139296169016) / 0.005452502053630527   # -0.2%  log_sum_pt > 7.139
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 11.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.45304387911873 * (-0.057950975437690774
        - 0.1443201321778588 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, Q.mass_top50 - 85.866695580031) / 1.0284575805457066   # -14.4%  log_sum_pt > 6.903 and mass_top50 > 85.87
        - 0.08177977110998798 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, Q.n_dr_0p2_0p4 - 4.0) / 0.29878998298885384   # -8.2%  log_sum_pt > 6.903 and n_dr_0p2_0p4 > 4
        + 0.07611155868246877 * max(0.0, 0.09795414517 - Q.mass_over_sum_pt) / 0.023392533955001828   # +7.6%  mass_over_sum_pt < 0.09795
        - 0.07160609638714964 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # -7.2%  mass < 92.86
        - 0.07028558604819043 * max(0.0, Q.sum_pt - 1066.481811523438) / 29.84706589314436   # -7.0%  sum_pt > 1066
        - 0.06417940974505934 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, Q.n_real_top30 - 22.0) / 0.41867376000846457   # -6.4%  log_sum_pt > 6.903 and n_real_top30 > 22
        - 0.05344650369501534 * max(0.0, Q.sum_pt_top50 - 1107.225842285156) * max(0.0, Q.z_top20_slots - 0.806826560894) / 1.529143479013972   # -5.3%  sum_pt_top50 > 1107 and z_top20_slots > 0.8068
        + 0.053391552810550386 * max(0.0, Q.z_top30_slots - 0.904849218002) / 0.056216409826696546   # +5.3%  z_top30_slots > 0.9048
        + 0.05272993677048771 * max(0.0, Q.sum_pt - 1017.43466796875) / 47.92777614496459   # +5.3%  sum_pt > 1017
        - 0.05151406352692324 * max(0.0, Q.sum_pt_top50 - 1038.855053710938) / 34.95068931446463   # -5.2%  sum_pt_top50 > 1039
        + 0.04376865820445129 * max(0.0, Q.sum_pt_top50 - 1156.659497070312) / 13.71463614316513   # +4.4%  sum_pt_top50 > 1157
        + 0.04345595403283877 * max(0.0, Q.sum_pt - 995.676940917969) / 62.241609981350926   # +4.3%  sum_pt > 995.7
        + 0.03749453101026936 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168090372162   # +3.7%  sum_pt_top20 > 1129
        + 0.02792453371767951 * max(0.0, Q.n_pt_above_10 - 18.0) / 3.550344537815126   # +2.8%  n_pt_above_10 > 18
        - 0.02003280939038357 * max(0.0, Q.sum_pt_top40 - 1013.04248046875) * max(0.0, -0.113525390625 - Q.eta_5) / 0.06120388842981403   # -2.0%  sum_pt_top40 > 1013 and eta_5 < -0.1135
        + 0.017406919526434597 * max(0.0, Q.sum_pt_top50 - 1038.855053710938) * max(0.0, Q.dr_3 - 0.004614387814) / 1.4544997558382704   # +1.7%  sum_pt_top50 > 1039 and dr_3 > 0.004614
        - 0.015827914678330383 * max(0.0, Q.mass_top40 - 160.8) * max(0.0, 41.4375 - Q.pt_9) / 7.659112980352017   # -1.6%  mass_top40 > 160.8 and pt_9 < 41.44
        + 0.014428043367165081 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, 0.01976735495 - Q.girth2_top10) / 0.000859816573689493   # +1.4%  log_sum_pt > 6.903 and girth2_top10 < 0.01977
        - 0.014058512385897398 * max(0.0, Q.sum_pt_top15 - 1003.329150390625) * max(0.0, 0.003952581551 - Q.girth2_top3) / 0.03832832268088158   # -1.4%  sum_pt_top15 > 1003 and girth2_top3 < 0.003953
        + 0.011658172135567883 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, Q.n_dr_0p05_0p1 - 9.0) / 0.2738882234939876   # +1.2%  log_sum_pt > 6.903 and n_dr_0p05_0p1 > 9
        + 0.01026598411494091 * max(0.0, 92.85979309082 - Q.mass) * max(0.0, 1069.67119140625 - Q.sum_pt_top40) / 1116.9379977034866   # +1.0%  mass < 92.86 and sum_pt_top40 < 1070
        - 0.007989085008064936 * max(0.0, Q.sum_pt_top50 - 1156.659497070312) * max(0.0, 5.0 - Q.n_pt_above_50) / 1.7969389997208596   # -0.8%  sum_pt_top50 > 1157 and n_pt_above_50 < 5
        - 0.005038094478003627 * max(0.0, Q.mass_top20 - 125.1) / 1.3144709680417963   # -0.5%  mass_top20 > 125.1
        - 0.0045944152187431374 * max(0.0, Q.sum_pt_top50 - 1107.225842285156) / 19.65455559402422   # -0.5%  sum_pt_top50 > 1107
        + 0.0026748676005032613 * max(0.0, Q.sum_pt_top50 - 997.018872070312) * max(0.0, Q.mean_phi - 0.000107912998) / 0.004139737697220853   # +0.3%  sum_pt_top50 > 997 and mean_phi > 0.0001079
        + 0.0014001283908204095 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.pt_7 - 45.75) / 58.04080342899353   # +0.1%  sum_pt_top20 > 1129 and pt_7 > 45.75
        - 0.0013552140254937505 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, 0.007929074034 - Q.girth2_top3) / 0.04346149526382533   # -0.1%  sum_pt_top20 > 1129 and girth2_top3 < 0.007929
        + 0.0007731916071294146 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.041473388672) / 0.012208788238965651   # +0.1%  sum_pt_top20 > 1129 and eta_1 > 0.04147
        - 0.0004883601535910629 * max(0.0, Q.sum_pt_top40 - 1013.04248046875) / 40.858647160172666   # -0.0%  sum_pt_top40 > 1013
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 2.181;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.180715852949879 * (-0.14022203024028257
        + 0.21147667308036738 * max(0.0, 0.00750911433 - Q.mass_over_sum_pt_sq) / 0.002014788498741022   # +21.1%  mass_over_sum_pt_sq < 0.007509
        + 0.16600962480577725 * max(0.0, 46.0 - Q.n_particles) / 6.392658823529412   # +16.6%  n_particles < 46
        - 0.11148403795143648 * max(0.0, 0.00750911433 - Q.mass_over_sum_pt_sq) * max(0.0, 136.785 - Q.mass_top20) / 0.20432244897571727   # -11.1%  mass_over_sum_pt_sq < 0.007509 and mass_top20 < 136.8
        + 0.09878073746643194 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139043697478992   # +9.9%  n_dr_0p1_0p2 < 15
        - 0.09749166715613682 * max(0.0, 0.00767124277 - Q.lam1) / 0.0025275527618478888   # -9.7%  lam1 < 0.007671
        - 0.08563107771147495 * max(0.0, 0.428145796061 - Q.tau21) * max(0.0, 0.016493544356 - Q.lam1) / 0.0007530527758270541   # -8.6%  tau21 < 0.4281 and lam1 < 0.01649
        + 0.07805726074340884 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.0366084033613445   # +7.8%  n_dr_0p2_0p4 < 7
        + 0.07092011747852052 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741004761) / 0.021355506979636982   # +7.1%  n_dr_0p2_0p4 < 5 and z_top50_slots > 0.9787
        - 0.03444271534995076 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top30 - 73.331387415761) / 43.91066245439405   # -3.4%  n_particles < 46 and mass_top30 > 73.33
        - 0.025835646455485355 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.460270001887473   # -2.6%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
        - 0.01987044180100988 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.04118638065 - Q.dr_0) / 0.009336793038502874   # -2.0%  n_dr_0p2_0p4 < 5 and dr_0 < 0.04119
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 13.75;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.74796456533005 * (0.18892357365585
        - 0.11544105050573472 * max(0.0, Q.LHA - 0.115200825015) / 0.14925839084262735   # -11.5%  LHA > 0.1152
        + 0.11462732102615648 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +11.5%  mass < 101
        - 0.0999406621185324 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # -10.0%  girth < 0.1207
        + 0.0780126374531129 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # +7.8%  mass < 120.6
        - 0.06527669786414501 * max(0.0, 94.642533639752 - Q.mass_top40) / 21.020214245469763   # -6.5%  mass_top40 < 94.64
        - 0.057767541106447294 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -5.8%  mass < 86.4
        - 0.05245105641884959 * max(0.0, 152.688263064041 - Q.mass_top30) / 74.18247342351319   # -5.2%  mass_top30 < 152.7
        + 0.04912650815867727 * max(0.0, 83.325535102591 - Q.mass_top40) / 13.716491940764705   # +4.9%  mass_top40 < 83.33
        + 0.04457225494415654 * max(0.0, 0.015638355144 - Q.girth2_top15) / 0.009593305133886523   # +4.5%  girth2_top15 < 0.01564
        + 0.038729276678777236 * max(0.0, Q.width - 0.009614971338) / 0.003192189674729974   # +3.9%  width > 0.009615
        + 0.03056274154951738 * max(0.0, 6.916121244431 - Q.D2) / 4.12932700422166   # +3.1%  D2 < 6.916
        - 0.029877192402564347 * max(0.0, 0.007277630044 - Q.girth2_top15) / 0.0028214695977655667   # -3.0%  girth2_top15 < 0.007278
        - 0.026067937632804624 * max(0.0, 83.325535102591 - Q.mass_top40) * max(0.0, 6.916121244431 - Q.D2) / 32.02926829226556   # -2.6%  mass_top40 < 83.33 and D2 < 6.916
        - 0.025818821033536932 * max(0.0, 80.784643554688 - Q.mass) / 10.849517559889705   # -2.6%  mass < 80.78
        + 0.023488990900435238 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.4828873949579835   # +2.3%  n_dr_0p2_0p4 < 15
        - 0.02274590433486113 * max(0.0, 0.068491501734 - Q.z_dr_0p2_0p4) / 0.04053661392560968   # -2.3%  z_dr_0p2_0p4 < 0.06849
        - 0.02045920464645308 * max(0.0, Q.lam1 - 0.00767124277) / 0.0027489268789339613   # -2.0%  lam1 > 0.007671
        + 0.017765021258104568 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, 0.402178311348 - Q.max_dr) / 2.4354257611523584   # +1.8%  mass < 120.6 and max_dr < 0.4022
        - 0.01683678082635273 * max(0.0, 0.061710142531 - Q.girth) / 0.012950562309845613   # -1.7%  girth < 0.06171
        - 0.012832523803196503 * max(0.0, 74.251806640625 - Q.mass) * max(0.0, 0.39398368001 - Q.max_dr) / 0.39183546981015077   # -1.3%  mass < 74.25 and max_dr < 0.394
        + 0.012594482859312847 * max(0.0, Q.e2 - 0.036805817112) / 0.004603798258454848   # +1.3%  e2 > 0.03681
        - 0.01223764634880422 * max(0.0, Q.n_particles - 29.0) * max(0.0, 25.0 - Q.n_dr_0_0p05) / 223.37779663865547   # -1.2%  n_particles > 29 and n_dr_0_0p05 < 25
        - 0.009874224961137814 * max(0.0, 1066.481811523438 - Q.sum_pt) / 52.533044976652775   # -1.0%  sum_pt < 1066
        - 0.00868616119343134 * max(0.0, Q.mass_top20 - 103.674910639856) / 3.7955110428668526   # -0.9%  mass_top20 > 103.7
        - 0.008201633563527412 * max(0.0, Q.mass_over_sum_pt - 0.09795414517) / 0.012228974621468441   # -0.8%  mass_over_sum_pt > 0.09795
        + 0.006005726411370381 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # +0.6%  mass > 143.8
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.779508728152875 * (0.06245582450833019
        - 0.16622035535361013 * max(0.0, Q.log_sum_pt - 6.910130970417) / 0.051750869608046556   # -16.6%  log_sum_pt > 6.91
        - 0.12346779062995586 * max(0.0, Q.z_top40_slots - 0.930046498893) / 0.051778900853579946   # -12.3%  z_top40_slots > 0.93
        + 0.10382603990415898 * max(0.0, Q.log_sum_pt - 6.856374501323) / 0.09628579359874964   # +10.4%  log_sum_pt > 6.856
        + 0.10054181069508979 * max(0.0, Q.z_top50_slots - 0.958653609576) / 0.03427111698918734   # +10.1%  z_top50_slots > 0.9587
        + 0.06868868583799032 * max(0.0, Q.sum_pt_top40 - 906.60234375) / 122.36719248432757   # +6.9%  sum_pt_top40 > 906.6
        + 0.06725098934368821 * max(0.0, Q.mass_top30 - 68.286969674465) / 20.391116049249682   # +6.7%  mass_top30 > 68.29
        - 0.05048612932168646 * max(0.0, Q.mass_over_sum_pt - 0.090467494167) / 0.014210394977997521   # -5.0%  mass_over_sum_pt > 0.09047
        + 0.0468424856057739 * max(0.0, Q.mass - 64.485443115234) / 31.191644704866818   # +4.7%  mass > 64.49
        + 0.034883856349997 * max(0.0, Q.sum_pt - 907.937170410156) * max(0.0, 0.0140332421 - Q.girth2_top2) / 1.486733466180341   # +3.5%  sum_pt > 907.9 and girth2_top2 < 0.01403
        + 0.03272523531261869 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.038759447634 - Q.e2) / 0.2513692835663279   # +3.3%  n_particles < 64 and e2 < 0.03876
        - 0.02930709698478584 * max(0.0, Q.mass_top50 - 91.19) / 13.751447163946986   # -2.9%  mass_top50 > 91.19
        - 0.021891226417452267 * max(0.0, Q.log_sum_pt - 6.92034855022) * max(0.0, 0.0140332421 - Q.girth2_top2) / 0.0005041067528121594   # -2.2%  log_sum_pt > 6.92 and girth2_top2 < 0.01403
        - 0.02105443023879019 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -2.1%  girth < 0.08589
        + 0.01955809904239357 * max(0.0, Q.log_sum_pt - 6.989450376716) / 0.021142744446604637   # +2.0%  log_sum_pt > 6.989
        - 0.018379335641259754 * max(0.0, Q.mass_over_sum_pt - 0.079990613285) / 0.01856800181915088   # -1.8%  mass_over_sum_pt > 0.07999
        + 0.017478965943270218 * max(0.0, Q.mass_over_sum_pt - 0.170880120467) / 0.0005536193928784717   # +1.7%  mass_over_sum_pt > 0.1709
        - 0.012563250719607264 * max(0.0, Q.mass_over_sum_pt_sq - 0.029200015571) / 0.00020282738822808687   # -1.3%  mass_over_sum_pt_sq > 0.0292
        - 0.011369565109954502 * max(0.0, Q.sum_pt_top15 - 951.1375) / 24.48977481165237   # -1.1%  sum_pt_top15 > 951.1
        + 0.010961924613619424 * max(0.0, Q.log_sum_pt - 6.92034855022) / 0.04509237421340201   # +1.1%  log_sum_pt > 6.92
        - 0.01070343468953262 * max(0.0, Q.sum_pt_top50 - 1061.183898925781) / 28.411282622213932   # -1.1%  sum_pt_top50 > 1061
        + 0.008751753649830907 * max(0.0, Q.sum_pt_top20 - 1005.0126953125) / 20.50199904518448   # +0.9%  sum_pt_top20 > 1005
        + 0.006599955935049161 * max(0.0, Q.log_sum_pt - 6.959293500649) / 0.028460039078747168   # +0.7%  log_sum_pt > 6.959
        + 0.005396640146435602 * max(0.0, Q.log_sum_pt - 6.989450376716) * max(0.0, 0.299250295758 - Q.z_dr_0p05_0p1) / 0.0036139904547987903   # +0.5%  log_sum_pt > 6.989 and z_dr_0p05_0p1 < 0.2993
        + 0.0050394843582880055 * max(0.0, 64.0 - Q.n_particles) * max(0.0, Q.z_dr_0p2_0p4 - 0.004744913615) / 0.4079057124357439   # +0.5%  n_particles < 64 and z_dr_0p2_0p4 > 0.004745
        - 0.004417551218890731 * max(0.0, Q.mass_top50 - 157.544814151857) / 1.5571187949287246   # -0.4%  mass_top50 > 157.5
        - 0.0015939069362707339 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # -0.2%  mass > 172.8
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 14.68;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.676707004285886 * (0.001244637558363823
        - 0.1320598106185053 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) / 0.039273301687911995   # -13.2%  mass_over_sum_pt > 0.05077
        - 0.11236288623356686 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # -11.2%  mass < 120.6
        - 0.10792702800639378 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # -10.8%  mass < 101
        + 0.08419839392677705 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # +8.4%  mass < 92.86
        + 0.07708059017911935 * max(0.0, Q.LHA - 0.228402115913) / 0.06082067856197019   # +7.7%  LHA > 0.2284
        + 0.05064761921142738 * max(0.0, Q.girth2_top20 - 0.0018230789) / 0.006286421415254086   # +5.1%  girth2_top20 > 0.001823
        + 0.041028539744121025 * max(0.0, 0.047553086095 - Q.e2) / 0.018774568371458196   # +4.1%  e2 < 0.04755
        + 0.03782105754019384 * max(0.0, 172.8 - Q.mass) / 83.7879223272957   # +3.8%  mass < 172.8
        + 0.03458182315678963 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # +3.5%  mass < 86.4
        + 0.029073397999235837 * max(0.0, 0.007259287357 - Q.lam1) / 0.0022497701299997253   # +2.9%  lam1 < 0.007259
        + 0.023452867289660884 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 0.32206342639914515   # +2.3%  mass_over_sum_pt > 0.05077 and n_dr_0p2_0p4 < 21
        + 0.02293559148737935 * max(0.0, 0.043586218357 - Q.e2) / 0.015487320383570933   # +2.3%  e2 < 0.04359
        - 0.022033401769105343 * max(0.0, 0.013514311784 - Q.girth2_top50) / 0.006620974830265835   # -2.2%  girth2_top50 < 0.01351
        - 0.021015621076824486 * max(0.0, Q.LHA - 0.228402115913) * max(0.0, 3.345338582993 - Q.D2) / 0.10056812770798138   # -2.1%  LHA > 0.2284 and D2 < 3.345
        + 0.021010856683592635 * max(0.0, Q.girth2_top3 - 0.010023689877) * max(0.0, 4.450168704987 - Q.D2) / 0.0041143288114283754   # +2.1%  girth2_top3 > 0.01002 and D2 < 4.45
        + 0.020950366041476688 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.002396991421 - Q.lam2) / 3.6218014529889185e-05   # +2.1%  mass_over_sum_pt > 0.05077 and lam2 < 0.002397
        - 0.02033501213236528 * max(0.0, Q.girth2_top3 - 0.010023689877) / 0.001644746899335674   # -2.0%  girth2_top3 > 0.01002
        - 0.020237168240932035 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063025210084   # -2.0%  n_dr_0p2_0p4 < 21
        - 0.01740168293805028 * max(0.0, 0.009614971338 - Q.width) / 0.003502544629791685   # -1.7%  width < 0.009615
        - 0.014565102068321904 * max(0.0, Q.girth2_top20 - 0.016559833876) / 0.001314598562186081   # -1.5%  girth2_top20 > 0.01656
        - 0.013365099770154345 * max(0.0, Q.lam1 - 0.011744050682) / 0.0018275243697888329   # -1.3%  lam1 > 0.01174
        + 0.01264913166882199 * max(0.0, 0.019863807341 - Q.mass_over_sum_pt_sq) / 0.011777496153852728   # +1.3%  mass_over_sum_pt_sq < 0.01986
        + 0.010158897614192895 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.908491230011 - Q.z_dr_0_0p05) / 0.02599640666253605   # +1.0%  mass_over_sum_pt > 0.05077 and z_dr_0_0p05 < 0.9085
        + 0.009605257225768512 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.286492615938 - Q.z_dr_0p1_0p2) / 0.0037357069815239703   # +1.0%  mass_over_sum_pt > 0.05077 and z_dr_0p1_0p2 < 0.2865
        + 0.007567346370848088 * max(0.0, Q.e2 - 0.055571487173) / 0.0011197307961334129   # +0.8%  e2 > 0.05557
        - 0.006529460053509618 * max(0.0, 92.85979309082 - Q.mass) * max(0.0, 1082.54765625 - Q.sum_pt_top15) / 2825.6490989367844   # -0.7%  mass < 92.86 and sum_pt_top15 < 1083
        - 0.006195178560820643 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, Q.n_dr_0p05_0p1 - 12.0) / 93.64873965170314   # -0.6%  mass < 120.6 and n_dr_0p05_0p1 > 12
        + 0.006137082615785529 * max(0.0, Q.lam1 - 0.008241985248) / 0.002595658483103056   # +0.6%  lam1 > 0.008242
        + 0.004552435633212042 * max(0.0, Q.lam1 - 0.008241985248) * max(0.0, 68.125 - Q.pt_4) / 0.046937438507826536   # +0.5%  lam1 > 0.008242 and pt_4 < 68.12
        - 0.004472920590509652 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, Q.max_pair_mass - 13.047927274731) / 0.3749186717396242   # -0.4%  mass_over_sum_pt > 0.05077 and max_pair_mass > 13.05
        + 0.0037553430854891058 * max(0.0, Q.LHA - 0.404204003833) * max(0.0, Q.z_2 - 0.049737748174) / 6.972499774959441e-05   # +0.4%  LHA > 0.4042 and z_2 > 0.04974
        - 0.002880998504913282 * max(0.0, Q.LHA - 0.404204003833) / 0.0031577141977621142   # -0.3%  LHA > 0.4042
        + 0.0014120319621352805 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # +0.1%  mass_top50 < 71.8
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 22.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 22.566858028066466 * (-0.022573805260000794
        - 0.1003243322494727 * max(0.0, 91.19 - Q.mass) / 16.464230590854466   # -10.0%  mass < 91.19
        - 0.08903397753825065 * max(0.0, 82.85408782959 - Q.mass) / 11.83453618826746   # -8.9%  mass < 82.85
        + 0.07133846958468375 * max(0.0, 0.009606007381 - Q.e2_sq) / 0.003497865944155776   # +7.1%  e2_sq < 0.009606
        - 0.06276797884248753 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # -6.3%  mass_top50 < 71.8
        - 0.06088783189462189 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -6.1%  girth < 0.08589
        - 0.06012905941740942 * max(0.0, 0.008031986041 - Q.girth2_top40) / 0.0025145928123983495   # -6.0%  girth2_top40 < 0.008032
        + 0.05680945615389325 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # +5.7%  mass < 120.6
        - 0.04890755623670681 * max(0.0, 0.008031209355 - Q.girth2_top20) / 0.0030984903862893757   # -4.9%  girth2_top20 < 0.008031
        + 0.043917441774196014 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +4.4%  mass < 101
        + 0.042787168203187335 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.390781164169 - Q.max_dr) / 1.1947278759545992   # +4.3%  mass < 101 and max_dr < 0.3908
        - 0.04023242384360875 * max(0.0, 97.930041729355 - Q.mass_top50) * max(0.0, 1.601009327173 - Q.D2) / 2.279411140212284   # -4.0%  mass_top50 < 97.93 and D2 < 1.601
        + 0.03490209705673319 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 1.601009327173 - Q.D2) / 2.8196790984068856   # +3.5%  mass < 101 and D2 < 1.601
        - 0.034366444250575834 * max(0.0, 91.19 - Q.mass) * max(0.0, 0.390781164169 - Q.max_dr) / 0.7740281265073297   # -3.4%  mass < 91.19 and max_dr < 0.3908
        + 0.03255985720400139 * max(0.0, 0.320332145368 - Q.LHA) / 0.07499257253386664   # +3.3%  LHA < 0.3203
        + 0.026597212244364385 * max(0.0, 0.007709915821 - Q.girth2_top40) / 0.0022972993366394247   # +2.7%  girth2_top40 < 0.00771
        + 0.025054036548413403 * max(0.0, 0.012926423095 - Q.girth2_top40) / 0.006292907975779176   # +2.5%  girth2_top40 < 0.01293
        - 0.023533077020874007 * max(0.0, 97.930041729355 - Q.mass_top50) / 22.035325884594407   # -2.4%  mass_top50 < 97.93
        + 0.02180010408246025 * max(0.0, 0.006374177987 - Q.girth2_top20) / 0.0019877511397026312   # +2.2%  girth2_top20 < 0.006374
        - 0.013885675232196292 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) / 0.05824141942126331   # -1.4%  z_dr_0p2_0p4 < 0.09123
        + 0.012025928209133556 * max(0.0, 66.841467317407 - Q.mass_top20) / 12.904985456467061   # +1.2%  mass_top20 < 66.84
        + 0.011528491992885059 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.499317836761 - Q.z_1st) / 1.5556551236660203   # +1.2%  n_dr_0p2_0p4 < 13 and z_1st < 0.4993
        + 0.011432550354116852 * max(0.0, 0.008840538245 - Q.girth2_top40) / 0.0031086216346541064   # +1.1%  girth2_top40 < 0.008841
        + 0.010863365807477213 * max(0.0, 1.788105106354 - Q.D2) * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 1.537145490198633   # +1.1%  D2 < 1.788 and n_dr_0p2_0p4 < 9
        + 0.010123761217717634 * max(0.0, 76.415438713532 - Q.mass_top30) / 12.67595855468548   # +1.0%  mass_top30 < 76.42
        - 0.009531546967385693 * max(0.0, 0.090467494167 - Q.mass_over_sum_pt) / 0.0178873033081874   # -1.0%  mass_over_sum_pt < 0.09047
        + 0.008556648012182946 * max(0.0, 71.795159472175 - Q.mass_top50) * max(0.0, 0.39398368001 - Q.max_dr) / 0.376497726873284   # +0.9%  mass_top50 < 71.8 and max_dr < 0.394
        - 0.007792102980241377 * max(0.0, 0.056027559564 - Q.C2) / 0.009882646645223543   # -0.8%  C2 < 0.05603
        - 0.007736993044150974 * max(0.0, 0.383415880799 - Q.max_dr) / 0.044460317722033585   # -0.8%  max_dr < 0.3834
        - 0.006585966060846974 * max(0.0, 91.19 - Q.mass) * max(0.0, 0.33780374676 - Q.pt_dispersion) / 0.4764307098452709   # -0.7%  mass < 91.19 and pt_dispersion < 0.3378
        - 0.0054106836084963274 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -0.5%  mass < 86.4
        - 0.00486605609121082 * max(0.0, 0.990637830118 - Q.z_top50_slots) / 0.004783315002560602   # -0.5%  z_top50_slots < 0.9906
        - 0.00371170627601774 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.402664637566) / 0.4065322951460936   # -0.4%  mass < 91.19 and z_dr_0p05_0p1 > 0.4027
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 14.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.886490186052427 * (-0.029461562518871325
        + 0.11025876108019651 * max(0.0, Q.mass - 64.485443115234) / 31.191644704866818   # +11.0%  mass > 64.49
        - 0.08863979298065915 * max(0.0, Q.mass - 101.049709320068) / 12.310843098263781   # -8.9%  mass > 101
        - 0.07864307039925687 * max(0.0, 0.020485236462 - Q.lam1) / 0.013097460086155891   # -7.9%  lam1 < 0.02049
        + 0.07022006803383637 * max(0.0, Q.mass - 80.784643554688) / 19.806095548570884   # +7.0%  mass > 80.78
        + 0.06838233763068739 * max(0.0, 0.097499583662 - Q.girth) / 0.03657082438923025   # +6.8%  girth < 0.0975
        - 0.05831015558143592 * max(0.0, Q.mass_over_sum_pt - 0.09795414517) / 0.012228974621468441   # -5.8%  mass_over_sum_pt > 0.09795
        - 0.05001117940403815 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063025210084   # -5.0%  n_dr_0p2_0p4 < 21
        + 0.044256804665001405 * max(0.0, Q.girth2_top20 - 0.008031209355) / 0.00290685235713101   # +4.4%  girth2_top20 > 0.008031
        + 0.038113922334792646 * max(0.0, Q.mass - 87.363773345947) / 16.577408637801742   # +3.8%  mass > 87.36
        + 0.03800297891633724 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) / 0.05824141942126331   # +3.8%  z_dr_0p2_0p4 < 0.09123
        - 0.03103535015033367 * max(0.0, Q.mass_top50 - 82.04491364955) / 17.710574785086557   # -3.1%  mass_top50 > 82.04
        - 0.030650848764157752 * max(0.0, Q.girth2_top20 - 0.005718442372) / 0.0037490635294349373   # -3.1%  girth2_top20 > 0.005718
        - 0.030232945617205065 * max(0.0, 0.013977372691 - Q.mass_over_sum_pt_sq) / 0.006903769640131633   # -3.0%  mass_over_sum_pt_sq < 0.01398
        + 0.02921813138073786 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) * max(0.0, 1115.722741699219 - Q.sum_pt) / 2.3366981130955806   # +2.9%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        + 0.0242550833301168 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) / 0.020241059768600602   # +2.4%  mass_over_sum_pt > 0.07697
        - 0.02202559733679003 * max(0.0, 0.006166777647 - Q.mass_over_sum_pt_sq) / 0.0012952540801658948   # -2.2%  mass_over_sum_pt_sq < 0.006167
        + 0.02190792339873291 * max(0.0, 0.008241985248 - Q.lam1) / 0.0029450268440803277   # +2.2%  lam1 < 0.008242
        - 0.01850363959948482 * max(0.0, 0.001411893759 - Q.lam2) / 0.0006681998875481002   # -1.9%  lam2 < 0.001412
        + 0.01737538772098501 * max(0.0, Q.girth2_top40 - 0.005196965925) / 0.004725809282736856   # +1.7%  girth2_top40 > 0.005197
        + 0.017303071894796906 * max(0.0, 0.154838323593 - Q.z_dr_0p1_0p2) / 0.06710802455713627   # +1.7%  z_dr_0p1_0p2 < 0.1548
        + 0.01698014872248305 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # +1.7%  sum_pt < 1013
        + 0.015384944060623228 * max(0.0, Q.mass_top50 - 97.930041729355) / 11.917644068610143   # +1.5%  mass_top50 > 97.93
        - 0.013991667669663617 * max(0.0, Q.e2 - 0.025159193203) / 0.010276419760149801   # -1.4%  e2 > 0.02516
        + 0.01152309666914612 * max(0.0, 1001.52314453125 - Q.sum_pt_top40) / 26.72885978092069   # +1.2%  sum_pt_top40 < 1002
        + 0.01087032710760364 * max(0.0, 0.007463984647 - Q.girth2_top30) / 0.0023370797259281294   # +1.1%  girth2_top30 < 0.007464
        - 0.010293061745419767 * max(0.0, 1011.52392578125 - Q.sum_pt_top30) / 51.02618157784598   # -1.0%  sum_pt_top30 < 1012
        + 0.008930300572708954 * max(0.0, Q.mass_over_sum_pt - 0.088731426731) / 0.014771713814206479   # +0.9%  mass_over_sum_pt > 0.08873
        - 0.00879326250994565 * max(0.0, Q.mass_top40 - 79.554505888974) / 17.06549122004638   # -0.9%  mass_top40 > 79.55
        - 0.00807690039486775 * max(0.0, Q.mass - 87.363773345947) * max(0.0, 6.903422848462 - Q.log_sum_pt) / 0.2991521451407768   # -0.8%  mass > 87.36 and log_sum_pt < 6.903
        - 0.0058513850935941235 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -0.6%  mass > 143.8
        - 0.001957855234361522 * max(0.0, 1001.52314453125 - Q.sum_pt_top40) * max(0.0, 6.811175180312 - Q.log_sum_pt) / 1.4602943926211116   # -0.2%  sum_pt_top40 < 1002 and log_sum_pt < 6.811
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 15.1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.09610885560571 * (0.06014324697331077
        + 0.23059764536224167 * max(0.0, 136.785 - Q.mass_top50) / 53.22362615511637   # +23.1%  mass_top50 < 136.8
        - 0.12477310335249693 * max(0.0, 160.8 - Q.mass_top50) / 74.23410092497848   # -12.5%  mass_top50 < 160.8
        - 0.11026196151272039 * max(0.0, Q.mass - 62.55) / 32.65527990955991   # -11.0%  mass > 62.55
        - 0.10428732560775694 * max(0.0, 0.02146577947 - Q.girth2_top15) / 0.0147044065440157   # -10.4%  girth2_top15 < 0.02147
        + 0.06073688866416072 * max(0.0, Q.mass - 82.85408782959) / 18.721669902205264   # +6.1%  mass > 82.85
        - 0.04387402598065533 * max(0.0, 0.047553086095 - Q.e2) / 0.018774568371458196   # -4.4%  e2 < 0.04755
        + 0.03625546901983992 * max(0.0, 0.209102506978 - Q.LHA) / 0.020957818638470994   # +3.6%  LHA < 0.2091
        - 0.03317998112762646 * max(0.0, 0.043628720567 - Q.girth) / 0.0063254006084753804   # -3.3%  girth < 0.04363
        + 0.0322679553108517 * max(0.0, Q.mass - 92.85979309082) / 14.482733665646967   # +3.2%  mass > 92.86
        - 0.02940637550530628 * max(0.0, 91.288535717504 - Q.mass_top40) / 18.56549606317863   # -2.9%  mass_top40 < 91.29
        - 0.02665134098357002 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -2.7%  mass > 143.8
        + 0.02547834980535657 * max(0.0, 136.785 - Q.mass) / 52.08717491697615   # +2.5%  mass < 136.8
        + 0.020132819544554544 * max(0.0, 0.009962397174 - Q.girth2_top15) / 0.004894698509240601   # +2.0%  girth2_top15 < 0.009962
        + 0.016353484246125795 * max(0.0, 120.60000000000001 - Q.mass_top40) / 41.60439949481591   # +1.6%  mass_top40 < 120.6
        + 0.014860605173586476 * max(0.0, 949.91689453125 - Q.sum_pt) / 6.913716555695057   # +1.5%  sum_pt < 949.9
        + 0.013582891730472656 * max(0.0, 0.00363885588 - Q.girth2) / 0.0004964601451473658   # +1.4%  girth2 < 0.003639
        + 0.01271268760909882 * max(0.0, 80.890431271924 - Q.mass_top40) / 12.434053057794845   # +1.3%  mass_top40 < 80.89
        + 0.012556351889595195 * max(0.0, 0.00625977218 - Q.girth2_top40) / 0.0014618229210262657   # +1.3%  girth2_top40 < 0.00626
        + 0.01250111006371603 * max(0.0, 0.02146577947 - Q.girth2_top15) * max(0.0, Q.n_particles - 41.0) / 0.10416207626617284   # +1.3%  girth2_top15 < 0.02147 and n_particles > 41
        + 0.011794510158186583 * max(0.0, 0.005240342196 - Q.lam1) / 0.0011655826082229445   # +1.2%  lam1 < 0.00524
        - 0.010573340308565224 * max(0.0, 0.02783744745 - Q.girth) / 0.0024110754510684616   # -1.1%  girth < 0.02784
        - 0.010559387854290524 * max(0.0, 6.856374501323 - Q.log_sum_pt) / 0.008053458833961623   # -1.1%  log_sum_pt < 6.856
        + 0.006602389189225214 * max(0.0, Q.girth - 0.097499583662) / 0.008315821681586878   # +0.7%  girth > 0.0975
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.143;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.142573389287582 * (0.4137258930781066
        - 0.1718476388961708 * max(0.0, 0.065240035206 - Q.e2) / 0.03476194878141159   # -17.2%  e2 < 0.06524
        - 0.13318122559241036 * max(0.0, 0.024419631481 - Q.girth2_top5) / 0.018899543128555206   # -13.3%  girth2_top5 < 0.02442
        + 0.09000570525750609 * max(0.0, Q.girth2_top10 - 0.000237176831) / 0.006544219018306094   # +9.0%  girth2_top10 > 0.0002372
        - 0.07056244784331389 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # -7.1%  mass < 120.6
        + 0.06404091370858463 * max(0.0, 0.024419631481 - Q.girth2_top5) * max(0.0, 656.384375 - Q.sum_pt_top3) / 3.538216833032187   # +6.4%  girth2_top5 < 0.02442 and sum_pt_top3 < 656.4
        + 0.062373922884374114 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +6.2%  mass < 80.4
        + 0.050056303379876266 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) / 0.05824141942126331   # +5.0%  z_dr_0p2_0p4 < 0.09123
        - 0.04043217627515067 * max(0.0, Q.lam1 - 0.001868040786) / 0.006218119096268822   # -4.0%  lam1 > 0.001868
        + 0.03786739216997328 * max(0.0, Q.mass_top50 - 136.785) / 4.25098606820319   # +3.8%  mass_top50 > 136.8
        - 0.036520642031782936 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -3.7%  mass > 143.8
        - 0.032445826394306015 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # -3.2%  n_dr_0p2_0p4 < 13
        - 0.027849419105716804 * max(0.0, 0.402178311348 - Q.max_dr) / 0.059113559416633016   # -2.8%  max_dr < 0.4022
        - 0.02773066928682161 * max(0.0, Q.girth2_top10 - 0.007678543663) / 0.002556946129466314   # -2.8%  girth2_top10 > 0.007679
        - 0.024793011936496207 * max(0.0, Q.z_dr_0_0p05 - 0.767473447323) / 0.05221087980608241   # -2.5%  z_dr_0_0p05 > 0.7675
        + 0.02104588540864262 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # +2.1%  mass < 86.4
        - 0.019945191366322612 * max(0.0, 0.065240035206 - Q.e2) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 0.004800315654529198   # -2.0%  e2 < 0.06524 and z_dr_0p1_0p2 < 0.2188
        + 0.01654718847557137 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, 0.470341457427 - Q.tau21) / 3.2648902511729423   # +1.7%  mass < 120.6 and tau21 < 0.4703
        - 0.015782955547590707 * max(0.0, Q.mass - 162.836349487305) / 1.5098322961374488   # -1.6%  mass > 162.8
        - 0.014618719495391879 * max(0.0, 0.002197764741 - Q.girth2_top15) / 0.00045099932175814557   # -1.5%  girth2_top15 < 0.002198
        - 0.013211379027032647 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.5418504201680672   # -1.3%  n_dr_0p2_0p4 < 6
        + 0.011528110805515041 * max(0.0, 0.064132973195 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.14948521527659667   # +1.2%  dr_0 < 0.06413 and n_dr_0p2_0p4 > 2
        - 0.01002650875839184 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -1.0%  mass < 62.55
        - 0.007586766353057618 * max(0.0, 959.095727539062 - Q.sum_pt_top50) / 9.937940549204788   # -0.8%  sum_pt_top50 < 959.1
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 7.035;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.035171038912912 * (0.12476783971033154
        - 0.19210707177897798 * max(0.0, 80.4 - Q.mass) * max(0.0, 0.037001823448 - Q.z_dr_0p2_0p4) / 0.30159836215099584   # -19.2%  mass < 80.4 and z_dr_0p2_0p4 < 0.037
        - 0.12374832372439022 * max(0.0, 86.4 - Q.mass_top50) / 14.253133905303311   # -12.4%  mass_top50 < 86.4
        + 0.09313475860809647 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # +9.3%  mass < 92.86
        - 0.06991024163894147 * max(0.0, 0.006363915755 - Q.girth2_top30) / 0.001678623919074012   # -7.0%  girth2_top30 < 0.006364
        + 0.0645333080357933 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.068491501734 - Q.z_dr_0p2_0p4) / 0.23346825157917758   # +6.5%  n_dr_0p2_0p4 < 10 and z_dr_0p2_0p4 < 0.06849
        + 0.06357091085313282 * max(0.0, 0.008184367501 - Q.mass_over_sum_pt_sq) / 0.0024516083597474207   # +6.4%  mass_over_sum_pt_sq < 0.008184
        - 0.054740679648007974 * max(0.0, Q.z_top5_slots - 0.534625950898) / 0.08390497481820398   # -5.5%  z_top5_slots > 0.5346
        - 0.05452695376244893 * max(0.0, 0.038759447634 - Q.e2) / 0.01184118794270979   # -5.5%  e2 < 0.03876
        + 0.05086100465588092 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +5.1%  mass < 101
        - 0.04790684712856336 * max(0.0, 0.006716736591 - Q.lam1) / 0.0019121054678470912   # -4.8%  lam1 < 0.006717
        + 0.04558861741781128 * max(0.0, 0.027935993578 - Q.e2) / 0.005715276962294592   # +4.6%  e2 < 0.02794
        - 0.04066843325266949 * max(0.0, 0.007164202106 - Q.girth2_top5) / 0.0036386067329905753   # -4.1%  girth2_top5 < 0.007164
        - 0.030279062915452088 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -3.0%  mass < 62.55
        + 0.026087948589375944 * max(0.0, 0.007164202106 - Q.girth2_top5) * max(0.0, 0.055209350586 - Q.eta_0) / 0.00020145990240281938   # +2.6%  girth2_top5 < 0.007164 and eta_0 < 0.05521
        + 0.024883835665421187 * max(0.0, Q.sum_pt_top3 - 439.125) / 81.65450839023109   # +2.5%  sum_pt_top3 > 439.1
        + 0.017452002325036676 * max(0.0, 0.006381743611 - Q.z_dr_0p2_0p4) / 0.0014538258888193599   # +1.7%  z_dr_0p2_0p4 < 0.006382
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 4.728;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.728324015460069 * (-0.04820088792881427
        + 0.20091911595159087 * max(0.0, 82.85408782959 - Q.mass) / 11.83453618826746   # +20.1%  mass < 82.85
        + 0.11764456486987042 * max(0.0, 80.4 - Q.mass_top50) / 11.159877357156788   # +11.8%  mass_top50 < 80.4
        + 0.09887247265922894 * max(0.0, 74.251806640625 - Q.mass) / 8.587064460866076   # +9.9%  mass < 74.25
        - 0.09085649810650764 * max(0.0, 0.050483809784 - Q.girth) / 0.008533024989277627   # -9.1%  girth < 0.05048
        - 0.0895103644762329 * max(0.0, 77.376408295162 - Q.mass_top50) / 9.980854903106312   # -9.0%  mass_top50 < 77.38
        - 0.08127399402675302 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -8.1%  mass < 86.4
        - 0.07244319173129579 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -7.2%  mass < 62.55
        + 0.051294128581247676 * max(0.0, 0.00742997247 - Q.girth2_top50) / 0.0020221555597961793   # +5.1%  girth2_top50 < 0.00743
        - 0.03820292113974394 * max(0.0, 77.936678808178 - Q.mass_top40) / 11.120277019467732   # -3.8%  mass_top40 < 77.94
        - 0.034832993118310326 * max(0.0, 0.00742997247 - Q.girth2_top50) * max(0.0, 0.428993919492 - Q.pt_dispersion) / 0.00017994021073038745   # -3.5%  girth2_top50 < 0.00743 and pt_dispersion < 0.429
        - 0.034715598089369186 * max(0.0, 6.856374501323 - Q.log_sum_pt) / 0.008053458833961623   # -3.5%  log_sum_pt < 6.856
        + 0.029125561365610588 * max(0.0, 0.228402115913 - Q.LHA) / 0.027166566418098437   # +2.9%  LHA < 0.2284
        - 0.024680157321471607 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.064725840837 - Q.z_dr_0p1_0p2) / 0.5993597398116846   # -2.5%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        + 0.023931943056546764 * max(0.0, 89.67879517394 - Q.mass_top40) * max(0.0, 501.625 - Q.sum_pt_top2) / 2080.229882440589   # +2.4%  mass_top40 < 89.68 and sum_pt_top2 < 501.6
        + 0.01169649550622059 * max(0.0, Q.LHA - 0.404204003833) / 0.0031577141977621142   # +1.2%  LHA > 0.4042
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 5.436;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.435878996717246 * (0.38100290545517385
        + 0.10707949239693652 * max(0.0, Q.mass - 74.251806640625) / 24.076479363593222   # +10.7%  mass > 74.25
        + 0.09274426403515888 * max(0.0, Q.mass_top50 - 136.785) / 4.25098606820319   # +9.3%  mass_top50 > 136.8
        - 0.09078201840561814 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # -9.1%  sum_pt < 1013
        - 0.0874620861232258 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -8.7%  mass > 143.8
        + 0.08739742552105509 * max(0.0, 1053.04736328125 - Q.sum_pt_top40) / 57.73987569631368   # +8.7%  sum_pt_top40 < 1053
        - 0.08286374157851091 * max(0.0, 1052.889428710937 - Q.sum_pt) / 42.62464270793601   # -8.3%  sum_pt < 1053
        + 0.0776228250593246 * max(0.0, 1013.915698242188 - Q.sum_pt_top50) / 24.159122871194317   # +7.8%  sum_pt_top50 < 1014
        - 0.06676700550852906 * max(0.0, Q.mass_top50 - 97.930041729355) / 11.917644068610143   # -6.7%  mass_top50 > 97.93
        - 0.05277811218462935 * max(0.0, 1053.04736328125 - Q.sum_pt_top40) * max(0.0, 5.378974604607 - Q.D2) / 152.36077532094276   # -5.3%  sum_pt_top40 < 1053 and D2 < 5.379
        - 0.042684586498506705 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # -4.3%  log_sum_pt < 7.017
        - 0.02999655105203307 * max(0.0, Q.mass_over_sum_pt_sq - 0.029200015571) / 0.00020282738822808687   # -3.0%  mass_over_sum_pt_sq > 0.0292
        + 0.0282270394241522 * max(0.0, Q.mass_over_sum_pt - 0.170880120467) / 0.0005536193928784717   # +2.8%  mass_over_sum_pt > 0.1709
        - 0.02770033105880644 * max(0.0, Q.mass - 74.251806640625) * max(0.0, Q.D2 - 0.603279101849) / 37.923504307700206   # -2.8%  mass > 74.25 and D2 > 0.6033
        - 0.0274236939550506 * max(0.0, Q.mass - 136.785) / 5.043396460386572   # -2.7%  mass > 136.8
        - 0.025669509811767844 * max(0.0, Q.mass - 160.8) / 1.7246633179459985   # -2.6%  mass > 160.8
        - 0.020457253399881276 * max(0.0, 956.21328125 - Q.sum_pt_top40) / 14.004510341849162   # -2.0%  sum_pt_top40 < 956.2
        + 0.018287859201945714 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +1.8%  mass > 172.8
        - 0.014770831698219366 * max(0.0, 986.05654296875 - Q.sum_pt) / 11.984374324425932   # -1.5%  sum_pt < 986.1
        + 0.010866110029920863 * max(0.0, 959.095727539062 - Q.sum_pt_top50) * max(0.0, 4.450168704987 - Q.D2) / 15.20393043139574   # +1.1%  sum_pt_top50 < 959.1 and D2 < 4.45
        + 0.008419263056727645 * max(0.0, Q.mass - 74.251806640625) * max(0.0, 1017.43466796875 - Q.sum_pt) / 595.5869900522073   # +0.8%  mass > 74.25 and sum_pt < 1017
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 25.55;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.549136064097862 * (0.028841614590303006
        - 0.11280735660025808 * max(0.0, 0.090467494167 - Q.mass_over_sum_pt) / 0.0178873033081874   # -11.3%  mass_over_sum_pt < 0.09047
        + 0.07588102462228664 * max(0.0, 0.09795414517 - Q.mass_over_sum_pt) / 0.023392533955001828   # +7.6%  mass_over_sum_pt < 0.09795
        - 0.0649398972416514 * max(0.0, 91.034691238403 - Q.mass) / 16.362866628552233   # -6.5%  mass < 91.03
        + 0.06182410752566496 * max(0.0, 0.118225939153 - Q.mass_over_sum_pt) / 0.039171218898223295   # +6.2%  mass_over_sum_pt < 0.1182
        - 0.055589451730393154 * max(0.0, 0.012157872869 - Q.girth2_top30) / 0.005935411821644211   # -5.6%  girth2_top30 < 0.01216
        + 0.044879604880814564 * max(0.0, 0.140939019879 - Q.mass_over_sum_pt) / 0.057960363993862694   # +4.5%  mass_over_sum_pt < 0.1409
        - 0.04394561319700583 * max(0.0, 0.009262053166 - Q.girth2_top50) / 0.0033075364251965024   # -4.4%  girth2_top50 < 0.009262
        + 0.04114746123843083 * max(0.0, 136.785 - Q.mass) / 52.08717491697615   # +4.1%  mass < 136.8
        - 0.037037795878883464 * max(0.0, Q.sum_pt_top50 - 976.277001953125) / 72.2927679781981   # -3.7%  sum_pt_top50 > 976.3
        - 0.035808068848219445 * max(0.0, 89.741833496094 - Q.mass) / 15.556328292327747   # -3.6%  mass < 89.74
        + 0.03218505653319541 * max(0.0, 0.007259287357 - Q.lam1) / 0.0022497701299997253   # +3.2%  lam1 < 0.007259
        + 0.031790341807111044 * max(0.0, 67.726432644245 - Q.mass_top40) / 7.7029388850516805   # +3.2%  mass_top40 < 67.73
        + 0.02957661253437154 * max(0.0, 89.080094718389 - Q.mass_top50) / 15.79328170065322   # +3.0%  mass_top50 < 89.08
        + 0.028800618155862316 * max(0.0, 0.002396991421 - Q.lam2) / 0.0014662533145241962   # +2.9%  lam2 < 0.002397
        - 0.027728060898951988 * max(0.0, 0.016559833876 - Q.girth2_top20) * max(0.0, Q.z_top50_slots - 0.970443639316) / 0.000256897965306318   # -2.8%  girth2_top20 < 0.01656 and z_top50_slots > 0.9704
        + 0.025188228029780328 * max(0.0, Q.log_sum_pt - 6.935549248787) / 0.03715380406194132   # +2.5%  log_sum_pt > 6.936
        - 0.02471980660949053 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -2.5%  mass < 80.4
        + 0.023917902554465926 * max(0.0, 0.007877041167 - Q.girth2) / 0.0022422969111317915   # +2.4%  girth2 < 0.007877
        - 0.021012937621474548 * max(0.0, 111.24867219155 - Q.mass_top40) / 33.99435635629078   # -2.1%  mass_top40 < 111.2
        - 0.020953232826566352 * max(0.0, 0.011744050682 - Q.lam1) / 0.005678958164694959   # -2.1%  lam1 < 0.01174
        - 0.018851526543493785 * max(0.0, 0.006189818106 - Q.lam1) / 0.0016087780966108156   # -1.9%  lam1 < 0.00619
        - 0.017770631987788266 * max(0.0, Q.max_dr - 0.240474711359) / 0.11773588635266226   # -1.8%  max_dr > 0.2405
        + 0.017551968925599022 * max(0.0, 0.006189818106 - Q.lam1) * max(0.0, Q.z_top50_slots - 0.985099030959) / 2.0496011611642913e-05   # +1.8%  lam1 < 0.00619 and z_top50_slots > 0.9851
        - 0.015808923310986 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 137.24421512605042   # -1.6%  n_dr_0p2_0p4 < 21 and n_dr_0p1_0p2 < 21
        + 0.015075110881013319 * max(0.0, 117.048742792994 - Q.mass_top50) / 36.94196179171718   # +1.5%  mass_top50 < 117
        + 0.013009705289056034 * max(0.0, 91.034691238403 - Q.mass) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 2.9734071289695474   # +1.3%  mass < 91.03 and z_dr_0p1_0p2 < 0.2188
        + 0.00958458668483732 * max(0.0, 91.19 - Q.mass_top30) / 21.633273586226487   # +1.0%  mass_top30 < 91.19
        - 0.00864942500208474 * max(0.0, 80.890431271924 - Q.mass_top40) / 12.434053057794845   # -0.9%  mass_top40 < 80.89
        - 0.008429795005670166 * max(0.0, 91.288535717504 - Q.mass_top40) * max(0.0, 0.068491501734 - Q.z_dr_0p2_0p4) / 1.0384437196530725   # -0.8%  mass_top40 < 91.29 and z_dr_0p2_0p4 < 0.06849
        + 0.006927075637907999 * max(0.0, 0.03263075389 - Q.e2) / 0.008034084923623714   # +0.7%  e2 < 0.03263
        - 0.0067079221631675895 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -0.7%  girth < 0.08589
        + 0.005086892652159744 * max(0.0, 91.288535717504 - Q.mass_top40) / 18.56549606317863   # +0.5%  mass_top40 < 91.29
        + 0.005023842479474361 * max(0.0, 0.019523000158 - Q.z_dr_0p2_0p4) * max(0.0, 0.358832142848 - Q.z_0) / 0.0010249432487276489   # +0.5%  z_dr_0p2_0p4 < 0.01952 and z_0 < 0.3588
        - 0.004011568448140468 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 40.0 - Q.n_real_top40) / 63.56472605042017   # -0.4%  n_dr_0p2_0p4 < 21 and n_real_top40 < 40
        - 0.003960688626970391 * max(0.0, 0.002396991421 - Q.lam2) * max(0.0, 462.25 - Q.sum_pt_top3) / 0.06946491493729062   # -0.4%  lam2 < 0.002397 and sum_pt_top3 < 462.2
        - 0.00265409366412932 * max(0.0, 0.331585738063 - Q.max_dr) / 0.021666863543587846   # -0.3%  max_dr < 0.3316
        + 0.0011630633626432063 * max(0.0, Q.max_dr - 0.402178311348) / 0.010008869835123797   # +0.1%  max_dr > 0.4022
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 12.48;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.48139128701637 * (-0.058708581733483776
        + 0.1229845794275643 * max(0.0, 0.013514311784 - Q.girth2_top50) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.0007407123072106741   # +12.3%  girth2_top50 < 0.01351 and z_dr_0p2_0p4 < 0.1292
        - 0.10717552706523388 * max(0.0, 0.013514311784 - Q.girth2_top50) / 0.006620974830265835   # -10.7%  girth2_top50 < 0.01351
        + 0.10466584679723172 * max(0.0, 0.006936724595 - Q.e2_sq) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.00019974727848153987   # +10.5%  e2_sq < 0.006937 and z_dr_0p2_0p4 < 0.1292
        + 0.1032794338091233 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # +10.3%  girth < 0.1207
        - 0.09912089272559674 * max(0.0, 0.019523000158 - Q.z_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.286492615938) / 8.047398982601161e-05   # -9.9%  z_dr_0p2_0p4 < 0.01952 and z_dr_0p1_0p2 > 0.2865
        - 0.07589996100347625 * max(0.0, 0.006936724595 - Q.e2_sq) / 0.0016886136779624735   # -7.6%  e2_sq < 0.006937
        - 0.06508524893841194 * max(0.0, 0.001501708498 - Q.girth2_top5) / 0.0004379639090356587   # -6.5%  girth2_top5 < 0.001502
        + 0.05369302377084013 * max(0.0, 0.001501708498 - Q.girth2_top5) * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 7.533004185945094e-05   # +5.4%  girth2_top5 < 0.001502 and z_dr_0p2_0p4 < 0.1937
        - 0.033904386095374356 * max(0.0, 0.120745175332 - Q.girth) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.005981775287783369   # -3.4%  girth < 0.1207 and z_dr_0p2_0p4 < 0.1292
        + 0.029921046383028418 * max(0.0, 0.001776308492 - Q.lam2) / 0.0009519164072663888   # +3.0%  lam2 < 0.001776
        - 0.026891811500016752 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # -2.7%  mass_top50 < 71.8
        - 0.0267011276165704 * max(0.0, 0.050483809784 - Q.girth) / 0.008533024989277627   # -2.7%  girth < 0.05048
        - 0.02566597029019545 * max(0.0, 0.016493544356 - Q.lam1) / 0.009604222881411155   # -2.6%  lam1 < 0.01649
        + 0.025410227834315006 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 40.2 - Q.mass_top5) / 1.3263398907230615   # +2.5%  z_dr_0p1_0p2 < 0.1203 and mass_top5 < 40.2
        + 0.02031264454688553 * max(0.0, 0.187280465662 - Q.z_dr_0p1_0p2) / 0.08737974006182235   # +2.0%  z_dr_0p1_0p2 < 0.1873
        + 0.018414714307429407 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # +1.8%  log_sum_pt < 7.017
        - 0.01749515130770634 * max(0.0, 1002.378515625 - Q.sum_pt) / 15.957354754845854   # -1.7%  sum_pt < 1002
        - 0.015029012281421915 * max(0.0, 0.019523000158 - Q.z_dr_0p2_0p4) / 0.0076644008476949655   # -1.5%  z_dr_0p2_0p4 < 0.01952
        + 0.011597321983995289 * max(0.0, 986.05654296875 - Q.sum_pt) / 11.984374324425932   # +1.2%  sum_pt < 986.1
        - 0.010873624635805601 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 0.037001823448 - Q.z_dr_0p2_0p4) / 0.0009116378981874741   # -1.1%  z_dr_0p1_0p2 < 0.1203 and z_dr_0p2_0p4 < 0.037
        - 0.00345773429328299 * max(0.0, 0.007164202106 - Q.girth2_top5) * max(0.0, 984.70087890625 - Q.sum_pt_top40) / 0.05379473153393668   # -0.3%  girth2_top5 < 0.007164 and sum_pt_top40 < 984.7
        - 0.0024207133864942287 * max(0.0, 0.002879910364 - Q.girth2_top20) * max(0.0, Q.sum_pt - 1115.722741699219) / 0.02115833407710715   # -0.2%  girth2_top20 < 0.00288 and sum_pt > 1116
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.7001486869747899, 3.02039506302521, 0.3060388655462185, 0.40825425420168066, 0.6571006302521009, 1.353483613445378, 0.3395171218487395, 0.3539188025210084, 1.1902862394957983, 1.2075529411764705, 1.271082668067227, 0.874551680672269, 0.3271928571428571, 1.484217962184874, 0.365484243697479, 0.37879285714285715]
T = [4.001301258862921, 2.4258356059611343, 4.485907198660714, 4.267458465730042, 4.056825237165179]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +47%, n9 +16%, n4 -14%, n5 -11%, n3 -8%, n12 +2% ...
            + 0.4717832505636457 * h[1] / H_AVG[1]
            + 0.15561049893830264 * h[9] / H_AVG[9]
            - 0.1436940170893255 * h[4] / H_AVG[4]
            - 0.10570651941410612 * h[5] / H_AVG[5]
            - 0.07652277867682297 * h[3] / H_AVG[3]
            + 0.019165221744550546 * h[12] / H_AVG[12]
            + 0.013258074525471002 * h[6] / H_AVG[6]
            + 0.009296087092131143 * h[8] / H_AVG[8]
            - 0.004963551955644142 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -29%, n9 +28%, n1 -23%, n11 -9%, n12 +5%, n6 +3% ...
            - 0.28780574327757763 * h[4] / H_AVG[4]
            + 0.280006001949436 * h[9] / H_AVG[9]
            - 0.2334552567888643 * h[1] / H_AVG[1]
            - 0.09012891047967006 * h[11] / H_AVG[11]
            + 0.04636445452712146 * h[12] / H_AVG[12]
            + 0.030615994843964576 * h[6] / H_AVG[6]
            + 0.015769765312732498 * h[2] / H_AVG[2]
            - 0.008187144520323535 * h[10] / H_AVG[10]
            + 0.007666728300309984 * h[8] / H_AVG[8]
        ),
        0.09375 + T[2] * (   # class W: n8 -23%, n5 +17%, n0 +12%, n11 +12%, n14 -11%, n9 -6% ...
            - 0.2321716463215663 * h[8] / H_AVG[8]
            + 0.1744313199918007 * h[5] / H_AVG[5]
            + 0.11705804243740635 * h[0] / H_AVG[0]
            + 0.11575474868365275 * h[11] / H_AVG[11]
            - 0.11202657853333867 * h[14] / H_AVG[14]
            - 0.05888490202410266 * h[9] / H_AVG[9]
            + 0.05035287883721638 * h[4] / H_AVG[4]
            - 0.04930981444325692 * h[7] / H_AVG[7]
            + 0.03981607917938214 * h[3] / H_AVG[3]
            - 0.029631040573904463 * h[12] / H_AVG[12]
            - 0.015832619260489858 * h[15] / H_AVG[15]
            + 0.004730329713882955 * h[6] / H_AVG[6]
        ),
        0.984375 + T[3] * (   # class Z: n8 -26%, n0 -23%, n5 +22%, n6 -8%, n7 +8%, n15 +5% ...
            - 0.2614889772187655 * h[8] / H_AVG[8]
            - 0.2255919893119908 * h[0] / H_AVG[0]
            + 0.2180501560158739 * h[5] / H_AVG[5]
            - 0.07707332465734958 * h[6] / H_AVG[6]
            + 0.07515923525919882 * h[7] / H_AVG[7]
            + 0.04992924567489767 * h[15] / H_AVG[15]
            + 0.041854241265047665 * h[3] / H_AVG[3]
            - 0.026892956428621963 * h[2] / H_AVG[2]
            + 0.02395987416825418 * h[12] / H_AVG[12]
        ),
        0.78125 + T[4] * (   # class t: n13 -33%, n10 +31%, n5 -16%, n8 +6%, n15 -4%, n4 +3% ...
            - 0.33155791773025683 * h[13] / H_AVG[13]
            + 0.30842393453778727 * h[10] / H_AVG[10]
            - 0.1563896413358584 * h[5] / H_AVG[5]
            + 0.06188977561653393 * h[8] / H_AVG[8]
            - 0.03501440489160213 * h[15] / H_AVG[15]
            + 0.030370144378800708 * h[4] / H_AVG[4]
            - 0.030244665287654748 * h[12] / H_AVG[12]
            - 0.0245363448977629 * h[7] / H_AVG[7]
            + 0.021573171323743003 * h[0] / H_AVG[0]
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
