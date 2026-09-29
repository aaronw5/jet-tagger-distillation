"""JEDI-linear jet tagger, 64 particles, 3 features: the tuned formula (start), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  19.5%   (on for 56% of jets)
  neuron  4:  11.2%   (on for 67% of jets)
  neuron  1:  10.4%   (on for 93% of jets)
  neuron  5:   9.5%   (on for 76% of jets)
  neuron 13:   7.9%   (on for 93% of jets)
  neuron 10:   5.8%   (on for 79% of jets)
  neuron  0:   5.5%   (on for 49% of jets)
  neuron  6:   5.3%   (on for 69% of jets)
  neuron  9:   4.7%   (on for 45% of jets)
  neuron  7:   4.2%   (on for 27% of jets)
  neuron 12:   3.9%   (on for 60% of jets)
  neuron  3:   3.9%   (on for 57% of jets)
  neuron 14:   3.5%   (on for 51% of jets)
  neuron 11:   2.8%   (on for 77% of jets)
  neuron 15:   1.0%   (on for 41% of jets)
  neuron  2:   0.9%   (on for 78% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.5% (the network: 81.1%); same class as the network for 92.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_0                    pT of particle 0 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_top10_slots          pT share of the 10 hardest particles
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
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr_8                   ΔR of particle 8 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_5                  Δη of particle 5
  Q.phi_0                  Δφ of particle 0
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
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top2=mass_of(2),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        pt_6=pt[6],
        pt_7=pt[7],
        pt_9=pt[9],
        z_0=z[0],
        z_4=z[4],
        z_top10_slots=sum(pt[:10]) / tot,
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
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr_8=dr[8] if pt[8] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_5=eta[5],
        phi_0=phi[0],
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
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 16.4;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.39531057229915 * (0.058806918983751184
        - 0.18243567342766623 * max(0.0, Q.mass - 78.261818313599) / 21.336576218229133   # -18.2%  mass > 78.26
        + 0.17653089229992708 * max(0.0, Q.mass - 91.034691238403) / 15.06939693349851   # +17.7%  mass > 91.03
        - 0.10826004345697682 * max(0.0, Q.mass - 92.85979309082) / 14.482733665646967   # -10.8%  mass > 92.86
        + 0.07046117841876784 * max(0.0, 0.008184367501 - Q.mass_over_sum_pt_sq) / 0.0024516083597474207   # +7.0%  mass_over_sum_pt_sq < 0.008184
        - 0.05774721682634226 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) / 0.020241059768600602   # -5.8%  mass_over_sum_pt > 0.07697
        + 0.05731885105683568 * max(0.0, Q.mass_over_sum_pt - 0.083299446175) / 0.01700043986417834   # +5.7%  mass_over_sum_pt > 0.0833
        - 0.03629615423435084 * max(0.0, 0.056600876898 - Q.girth) / 0.010803815022120036   # -3.6%  girth < 0.0566
        - 0.03297306944938328 * max(0.0, 82.04491364955 - Q.mass_top50) / 11.943128521443713   # -3.3%  mass_top50 < 82.04
        - 0.03047805544383902 * max(0.0, 0.00625977218 - Q.girth2_top40) / 0.0014618229210262657   # -3.0%  girth2_top40 < 0.00626
        + 0.02896114333766947 * max(0.0, 0.00742997247 - Q.girth2_top50) / 0.0020221555597961793   # +2.9%  girth2_top50 < 0.00743
        - 0.020723603597152966 * max(0.0, 0.009614971338 - Q.width) / 0.003502544629791685   # -2.1%  width < 0.009615
        - 0.02024924093900625 * max(0.0, 0.006043208873 - Q.girth2_top20) / 0.0017975224219569485   # -2.0%  girth2_top20 < 0.006043
        + 0.017506774561571282 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # +1.8%  log_sum_pt < 7.017
        - 0.017153753516268418 * max(0.0, 0.005913554513 - Q.lam1) / 0.0014639240999900342   # -1.7%  lam1 < 0.005914
        + 0.01535705829535058 * max(0.0, 0.028623861071 - Q.girth2_top40) / 0.019927913733935647   # +1.5%  girth2_top40 < 0.02862
        + 0.012823426613076816 * max(0.0, 0.260146178237 - Q.LHA) / 0.039673711668022284   # +1.3%  LHA < 0.2601
        + 0.011415983392694415 * max(0.0, 0.00625977218 - Q.girth2_top40) * max(0.0, 0.002915531053 - Q.girth2_top3) / 3.6946461207137186e-06   # +1.1%  girth2_top40 < 0.00626 and girth2_top3 < 0.002916
        + 0.010372097225532957 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583193277311   # +1.0%  n_dr_0p2_0p4 < 10
        - 0.009881743379647654 * max(0.0, Q.z_top30_slots - 0.920388080863) / 0.04411613538673992   # -1.0%  z_top30_slots > 0.9204
        - 0.008327920891280123 * max(0.0, Q.mass - 74.251806640625) / 24.076479363593222   # -0.8%  mass > 74.25
        + 0.007253742823697586 * max(0.0, 82.04491364955 - Q.mass_top50) * max(0.0, 0.212648361921 - Q.z_dr_0p05_0p1) / 1.9465026796824894   # +0.7%  mass_top50 < 82.04 and z_dr_0p05_0p1 < 0.2126
        + 0.006858675591039691 * max(0.0, 0.004278051991 - Q.girth2_top40) / 0.000735857317901877   # +0.7%  girth2_top40 < 0.004278
        + 0.0065332995613156175 * max(0.0, 62.0 - Q.n_particles) / 16.64866050420168   # +0.7%  n_particles < 62
        - 0.006332096100980546 * max(0.0, Q.mass - 78.261818313599) * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 11.579260522919679   # -0.6%  mass > 78.26 and n_dr_0p2_0p4 < 7
        + 0.00516986552544082 * max(0.0, 0.005312783396 - Q.girth2_top20) / 0.00143721407379855   # +0.5%  girth2_top20 < 0.005313
        + 0.004553780064953887 * max(0.0, 6.989450376716 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top40 - 956.21328125) / 1.6892767547563012   # +0.5%  log_sum_pt < 6.989 and sum_pt_top40 > 956.2
        - 0.004366660511313147 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 2.3517925596042555   # -0.4%  sum_pt < 1013 and z_dr_0p2_0p4 < 0.1937
        - 0.0043200856810396825 * max(0.0, 0.007538018543 - Q.girth2_top20) * max(0.0, 0.003111083776 - Q.girth2_top2) / 6.649790537932148e-06   # -0.4%  girth2_top20 < 0.007538 and girth2_top2 < 0.003111
        - 0.0036633417212026016 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.088715460151) / 3.100400728225136   # -0.4%  sum_pt < 1013 and z_dr_0p1_0p2 > 0.08872
        - 0.0035856166632565827 * max(0.0, Q.z_dr_0_0p05 - 0.878906026483) / 0.015835254093555608   # -0.4%  z_dr_0_0p05 > 0.8789
        + 0.003492346364230735 * max(0.0, Q.sum_pt_top20 - 942.0) / 44.31098723082983   # +0.3%  sum_pt_top20 > 942
        + 0.0029468549115981625 * max(0.0, 70.421206773231 - Q.mass_top20) / 14.595357970697426   # +0.3%  mass_top20 < 70.42
        - 0.002398741454970359 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # -0.2%  sum_pt < 1013
        + 0.0023262684013652586 * max(0.0, 0.037001823448 - Q.z_dr_0p2_0p4) * max(0.0, 126.75 - Q.pt_2) / 0.5887923700890885   # +0.2%  z_dr_0p2_0p4 < 0.037 and pt_2 < 126.8
        - 0.0022804594441210827 * max(0.0, 6.989450376716 - Q.log_sum_pt) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 0.33257796290773134   # -0.2%  log_sum_pt < 6.989 and n_dr_0p1_0p2 > 11
        + 0.0021256798857288255 * max(0.0, 62.55 - Q.mass_top50) / 5.6778583318492695   # +0.2%  mass_top50 < 62.55
        - 0.0018335252960323188 * max(0.0, Q.z_top30_slots - 0.920388080863) * max(0.0, Q.C2 - 0.061168736406) / 0.0003963218413190363   # -0.2%  z_top30_slots > 0.9204 and C2 > 0.06117
        - 0.0012756734597551824 * max(0.0, Q.sum_pt_top30 - 1073.4734375) / 17.186466467801473   # -0.1%  sum_pt_top30 > 1073
        + 0.0010410370039119414 * max(0.0, 62.0 - Q.n_particles) * max(0.0, 30.0 - Q.n_real_top30) / 39.198035294117645   # +0.1%  n_particles < 62 and n_real_top30 < 30
        - 0.0008656614883156342 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 30.0 - Q.n_real_top30) / 6.5523983193277315   # -0.1%  n_dr_0p2_0p4 < 10 and n_real_top30 < 30
        - 0.0008517856894243424 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.985099030959 - Q.z_top50_slots) / 0.007036197797037822   # -0.1%  n_dr_0p2_0p4 < 10 and z_top50_slots < 0.9851
        + 0.0006509219929660488 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, Q.n_dr_0p1_0p2 - 11.0) / 119.0108001529797   # +0.1%  sum_pt < 1013 and n_dr_0p1_0p2 > 11
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 18.67;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.6673076335677 * (0.07330316855685094
        + 0.08194664876821446 * max(0.0, Q.log_sum_pt - 6.910130970417) / 0.051750869608046556   # +8.2%  log_sum_pt > 6.91
        - 0.06965790907096565 * max(0.0, Q.z_top50_slots - 0.958653609576) / 0.03427111698918734   # -7.0%  z_top50_slots > 0.9587
        + 0.06702403682013641 * max(0.0, Q.log_sum_pt - 6.89371369877) / 0.06417423647849095   # +6.7%  log_sum_pt > 6.894
        + 0.04820172848258995 * max(0.0, 689.25 - Q.sum_pt_top2) / 323.27338014705884   # +4.8%  sum_pt_top2 < 689.2
        + 0.046676406990972315 * max(0.0, 0.404204003833 - Q.LHA) / 0.14530548997384415   # +4.7%  LHA < 0.4042
        - 0.04469415761726031 * max(0.0, Q.sum_pt_top50 - 959.095727539062) / 86.53899855624287   # -4.5%  sum_pt_top50 > 959.1
        + 0.04463912105591711 * max(0.0, Q.tau32 - 0.329314215481) / 0.3985957471124154   # +4.5%  tau32 > 0.3293
        + 0.04397230683017175 * max(0.0, 80.4 - Q.mass_top30) / 14.655773101172624   # +4.4%  mass_top30 < 80.4
        - 0.04351962874987543 * max(0.0, Q.log_sum_pt - 6.811175180312) / 0.138314075182229   # -4.4%  log_sum_pt > 6.811
        - 0.038680526596884736 * max(0.0, 0.957678701144 - Q.z_top20_slots) / 0.07565544239103991   # -3.9%  z_top20_slots < 0.9577
        - 0.036217034733317964 * max(0.0, 117.048742792994 - Q.mass_top50) / 36.94196179171718   # -3.6%  mass_top50 < 117
        - 0.03568072441997834 * max(0.0, 0.43572281599 - Q.max_dr) / 0.09036112201580485   # -3.6%  max_dr < 0.4357
        - 0.03373769313622573 * max(0.0, 31.0 - Q.n_pt_above_10) / 11.388505882352941   # -3.4%  n_pt_above_10 < 31
        - 0.030794771982529036 * max(0.0, 80.4 - Q.mass_top30) * max(0.0, 68.434224049685 - Q.mass_top5) / 891.7019121450287   # -3.1%  mass_top30 < 80.4 and mass_top5 < 68.43
        - 0.027475142347058733 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, 91.19 - Q.mass_top10) / 1.6455577605891802   # -2.7%  z_top30_slots > 0.9342 and mass_top10 < 91.19
        + 0.025957239287368945 * max(0.0, 902.40625 - Q.sum_pt_top5) / 313.0793622488839   # +2.6%  sum_pt_top5 < 902.4
        + 0.025747106923159842 * max(0.0, Q.z_top30_slots - 0.934183811419) / 0.034068179997830586   # +2.6%  z_top30_slots > 0.9342
        + 0.02354763644039701 * max(0.0, 1073.4734375 - Q.sum_pt_top30) / 97.85231293739916   # +2.4%  sum_pt_top30 < 1073
        + 0.021644342802460918 * max(0.0, 0.43572281599 - Q.max_dr) * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 0.0143775071962318   # +2.2%  max_dr < 0.4357 and z_dr_0p2_0p4 < 0.1937
        - 0.020139726273902418 * max(0.0, Q.log_sum_pt - 6.989450376716) / 0.021142744446604637   # -2.0%  log_sum_pt > 6.989
        - 0.018915910094199174 * max(0.0, 40.2 - Q.mass_top20) / 3.9165662697562196   # -1.9%  mass_top20 < 40.2
        - 0.015043574619287072 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # -1.5%  n_dr_0p2_0p4 < 13
        + 0.012578492537736548 * max(0.0, Q.n_particles - 38.0) / 10.799277310924369   # +1.3%  n_particles > 38
        + 0.012405366461525456 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.144288109196 - Q.dr_0) / 0.8704373876584949   # +1.2%  n_particles > 38 and dr_0 < 0.1443
        + 0.011168019527785327 * max(0.0, 0.007678543663 - Q.girth2_top10) / 0.0034735798352083574   # +1.1%  girth2_top10 < 0.007679
        - 0.010072852751142315 * max(0.0, 0.000829637219 - Q.lam2) / 0.000268103756224263   # -1.0%  lam2 < 0.0008296
        - 0.009384993778480073 * max(0.0, 34.0625 - Q.pt_9) / 8.573627858291754   # -0.9%  pt_9 < 34.06
        + 0.009327109639491477 * max(0.0, 40.2 - Q.mass_top20) * max(0.0, 0.005860335776 - Q.mean_phi2) / 0.01972420536905745   # +0.9%  mass_top20 < 40.2 and mean_phi2 < 0.00586
        + 0.008553096525589297 * max(0.0, 0.074356165682 - Q.mass_over_sum_pt) / 0.009452716665803248   # +0.9%  mass_over_sum_pt < 0.07436
        + 0.007769389805952767 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.985099030959 - Q.z_top50_slots) / 0.09243197309849287   # +0.8%  n_particles > 38 and z_top50_slots < 0.9851
        - 0.007697326857685976 * max(0.0, 689.25 - Q.sum_pt_top2) * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 621.4389484506303   # -0.8%  sum_pt_top2 < 689.2 and n_dr_0p2_0p4 < 7
        + 0.006615810585392436 * max(0.0, 0.003270031267 - Q.girth2_top15) / 0.0007969964936058968   # +0.7%  girth2_top15 < 0.00327
        + 0.006199398592319363 * max(0.0, 40.2 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 26.0) / 33.56032785174058   # +0.6%  mass_top20 < 40.2 and n_real_top40 > 26
        + 0.006056710296765402 * max(0.0, 0.000823693417 - Q.girth2_top3) / 0.00020583156908661208   # +0.6%  girth2_top3 < 0.0008237
        + 0.005979855472335625 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.161154452503 - Q.dr_1) / 1.0055977424208304   # +0.6%  n_particles > 38 and dr_1 < 0.1612
        + 0.005566582960437004 * max(0.0, 0.00115306291 - Q.girth2_top20) / 0.00012008616672638938   # +0.6%  girth2_top20 < 0.001153
        + 0.005547318553405515 * max(0.0, 0.00130722027 - Q.girth2_top10) / 0.00027941021155763345   # +0.6%  girth2_top10 < 0.001307
        - 0.005365672097808791 * max(0.0, Q.log_sum_pt - 6.959293500649) / 0.028460039078747168   # -0.5%  log_sum_pt > 6.959
        - 0.004906462072288019 * max(0.0, 0.000823693417 - Q.girth2_top3) * max(0.0, 10.0 - Q.n_dr_0p05_0p1) / 0.000986422207911248   # -0.5%  girth2_top3 < 0.0008237 and n_dr_0p05_0p1 < 10
        - 0.003815675254335825 * max(0.0, 7.0 - Q.n_dr_0p1_0p2) * max(0.0, 0.079220479673 - Q.dr_6) / 0.0469437715412463   # -0.4%  n_dr_0p1_0p2 < 7 and dr_6 < 0.07922
        + 0.0032692213716155666 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, Q.m012 - 16.899120053094) / 0.2366429075108053   # +0.3%  z_top30_slots > 0.9342 and m012 > 16.9
        - 0.0032656655211030024 * max(0.0, Q.n_particles - 38.0) * max(0.0, 57.873489696602 - Q.mass_top15) / 116.48355393422717   # -0.3%  n_particles > 38 and mass_top15 < 57.87
        + 0.002881516545797492 * max(0.0, 1.230420708656 - Q.D2) / 0.1020860510937745   # +0.3%  D2 < 1.23
        - 0.002332282262529965 * max(0.0, Q.z_top30_slots - 0.934183811419) * max(0.0, 0.057912331642 - Q.dr_5) / 0.0007378409260055227   # -0.2%  z_top30_slots > 0.9342 and dr_5 < 0.05791
        + 0.0012864665126354215 * max(0.0, 31.0 - Q.n_pt_above_10) * max(0.0, 0.017422899418 - Q.mean_phi2) / 0.1577871548170423   # +0.1%  n_pt_above_10 < 31 and mean_phi2 < 0.01742
        + 0.0011585663319165019 * max(0.0, 0.000657050184 - Q.girth2_top5) / 0.0001395474066313899   # +0.1%  girth2_top5 < 0.0006571
        - 0.0008474136802158194 * max(0.0, Q.log_sum_pt - 7.139296169016) / 0.005452502053630527   # -0.1%  log_sum_pt > 7.139
        + 0.0007393806570865773 * max(0.0, 0.43572281599 - Q.max_dr) * max(0.0, 26.0 - Q.n_real_top30) / 0.05166362526162354   # +0.1%  max_dr < 0.4357 and n_real_top30 < 26
        - 0.0007373271670005965 * max(0.0, Q.z_dr_0_0p05 - 0.878906026483) / 0.015835254093555608   # -0.1%  z_dr_0_0p05 > 0.8789
        - 0.0005586521367484354 * max(0.0, Q.n_dr_0_0p05 - 30.0) / 0.21247899159663866   # -0.1%  n_dr_0_0p05 > 30
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 5.682;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.682215937967667 * (-0.03739864442656243
        - 0.12465377218563665 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # -12.5%  mass < 92.86
        + 0.09651287685776576 * max(0.0, Q.z_top30_slots - 0.904849218002) / 0.056216409826696546   # +9.7%  z_top30_slots > 0.9048
        + 0.07150460891244055 * max(0.0, Q.sum_pt - 1017.43466796875) / 47.92777614496459   # +7.2%  sum_pt > 1017
        + 0.06539590952500668 * max(0.0, 0.015638355144 - Q.girth2_top15) / 0.009593305133886523   # +6.5%  girth2_top15 < 0.01564
        + 0.06429138550267846 * max(0.0, 0.09795414517 - Q.mass_over_sum_pt) / 0.023392533955001828   # +6.4%  mass_over_sum_pt < 0.09795
        - 0.05920667410383182 * max(0.0, 0.140939019879 - Q.mass_over_sum_pt) / 0.057960363993862694   # -5.9%  mass_over_sum_pt < 0.1409
        - 0.05564390203762526 * max(0.0, Q.sum_pt - 1066.481811523438) / 29.84706589314436   # -5.6%  sum_pt > 1066
        - 0.051143523697003666 * max(0.0, Q.sum_pt_top50 - 997.018872070312) / 56.586436834715556   # -5.1%  sum_pt_top50 > 997
        + 0.038282407179127634 * max(0.0, 86.4 - Q.mass_top50) / 14.253133905303311   # +3.8%  mass_top50 < 86.4
        + 0.03433847256970604 * max(0.0, 0.074356165682 - Q.mass_over_sum_pt) / 0.009452716665803248   # +3.4%  mass_over_sum_pt < 0.07436
        - 0.030912162939319837 * max(0.0, 0.000973194699 - Q.lam2) / 0.00035805739390115264   # -3.1%  lam2 < 0.0009732
        + 0.028004385928729558 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, 0.01976735495 - Q.girth2_top10) / 0.000859816573689493   # +2.8%  log_sum_pt > 6.903 and girth2_top10 < 0.01977
        + 0.026095656929921347 * max(0.0, Q.sum_pt_top50 - 1038.855053710938) / 34.95068931446463   # +2.6%  sum_pt_top50 > 1039
        + 0.02457824916749215 * max(0.0, Q.log_sum_pt - 7.062574317998) / 0.010878839937255731   # +2.5%  log_sum_pt > 7.063
        - 0.02432436176774637 * max(0.0, Q.sum_pt_top40 - 1013.04248046875) / 40.858647160172666   # -2.4%  sum_pt_top40 > 1013
        + 0.022825863964209006 * max(0.0, Q.sum_pt_top40 - 1069.67119140625) / 23.00895715528997   # +2.3%  sum_pt_top40 > 1070
        - 0.018792484427204343 * max(0.0, 0.006142801866 - Q.girth2_top15) / 0.0020806505579742157   # -1.9%  girth2_top15 < 0.006143
        - 0.01853865722361471 * max(0.0, Q.sum_pt_top50 - 1107.225842285156) / 19.65455559402422   # -1.9%  sum_pt_top50 > 1107
        + 0.014806573745688154 * max(0.0, 91.697531419407 - Q.mass_top30) / 22.012698823640722   # +1.5%  mass_top30 < 91.7
        - 0.014486060902812398 * max(0.0, Q.mass_top40 - 160.8) / 0.77818694636032   # -1.4%  mass_top40 > 160.8
        - 0.012454914483558873 * max(0.0, 0.006189818106 - Q.lam1) / 0.0016087780966108156   # -1.2%  lam1 < 0.00619
        - 0.011392076472223758 * max(0.0, Q.sum_pt_top30 - 1018.831689453125) / 29.594060059887628   # -1.1%  sum_pt_top30 > 1019
        + 0.009917977561281873 * max(0.0, Q.sum_pt_top20 - 908.8125) / 62.53618744132419   # +1.0%  sum_pt_top20 > 908.8
        + 0.009132192752205913 * max(0.0, Q.log_sum_pt - 6.903422848462) / 0.056622750952925925   # +0.9%  log_sum_pt > 6.903
        - 0.008850554116816884 * max(0.0, Q.sum_pt - 995.676940917969) / 62.241609981350926   # -0.9%  sum_pt > 995.7
        + 0.007985057113895702 * max(0.0, Q.sum_pt_top50 - 1107.225842285156) * max(0.0, Q.C2 - 0.097715596855) / 0.054736570108090826   # +0.8%  sum_pt_top50 > 1107 and C2 > 0.09772
        + 0.006722737273469393 * max(0.0, Q.log_sum_pt - 7.062574317998) * max(0.0, Q.mass_top20 - 66.841467317407) / 0.19896713560921692   # +0.7%  log_sum_pt > 7.063 and mass_top20 > 66.84
        + 0.00652776243797796 * max(0.0, Q.mass_top40 - 160.8) * max(0.0, 41.4375 - Q.pt_9) / 7.659112980352017   # +0.7%  mass_top40 > 160.8 and pt_9 < 41.44
        + 0.006225224247596071 * max(0.0, Q.n_pt_above_10 - 18.0) * max(0.0, Q.z_4 - 0.035928898346) / 0.05158404643022957   # +0.6%  n_pt_above_10 > 18 and z_4 > 0.03593
        - 0.005988246755243421 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, Q.mass_top50 - 85.866695580031) / 1.0284575805457066   # -0.6%  log_sum_pt > 6.903 and mass_top50 > 85.87
        + 0.00578273879058398 * max(0.0, 0.402178311348 - Q.max_dr) / 0.059113559416633016   # +0.6%  max_dr < 0.4022
        - 0.005254683636284663 * max(0.0, Q.sum_pt_top50 - 1038.855053710938) * max(0.0, Q.dr_3 - 0.004614387814) / 1.4544997558382704   # -0.5%  sum_pt_top50 > 1039 and dr_3 > 0.004614
        - 0.005003004796219255 * max(0.0, 92.85979309082 - Q.mass) * max(0.0, 1069.67119140625 - Q.sum_pt_top40) / 1116.9379977034866   # -0.5%  mass < 92.86 and sum_pt_top40 < 1070
        + 0.00496591149276832 * max(0.0, Q.log_sum_pt - 6.903422848462) * max(0.0, 18.0 - Q.n_dr_0_0p05) / 0.3049593532834459   # +0.5%  log_sum_pt > 6.903 and n_dr_0_0p05 < 18
        - 0.004186352921101793 * max(0.0, Q.sum_pt_top20 - 1129.275) / 6.499168090372162   # -0.4%  sum_pt_top20 > 1129
        + 0.0025942865663363584 * max(0.0, Q.sum_pt_top50 - 997.018872070312) * max(0.0, Q.mean_phi - 0.000107912998) / 0.004139737697220853   # +0.3%  sum_pt_top50 > 997 and mean_phi > 0.0001079
        + 0.001119278967630889 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, 0.007929074034 - Q.girth2_top3) / 0.04346149526382533   # +0.1%  sum_pt_top20 > 1129 and girth2_top3 < 0.007929
        - 0.0008347106042170284 * max(0.0, Q.sum_pt_top40 - 1013.04248046875) * max(0.0, -0.113525390625 - Q.eta_5) / 0.06120388842981403   # -0.1%  sum_pt_top40 > 1013 and eta_5 < -0.1135
        + 0.00045749916617045055 * max(0.0, Q.sum_pt - 1066.481811523438) * max(0.0, 36.8125 - Q.pt_6) / 40.34200870301723   # +0.0%  sum_pt > 1066 and pt_6 < 36.81
        - 0.00026680027685696935 * max(0.0, Q.sum_pt_top20 - 1129.275) * max(0.0, Q.eta_1 - 0.041473388672) / 0.012208788238965651   # -0.0%  sum_pt_top20 > 1129 and eta_1 > 0.04147
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 2.186;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.186399626935491 * (-0.08181320243871895
        + 0.18819673343645768 * max(0.0, 0.00750911433 - Q.mass_over_sum_pt_sq) / 0.002014788498741022   # +18.8%  mass_over_sum_pt_sq < 0.007509
        + 0.10929799327375672 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.978741004761) / 0.021355506979636982   # +10.9%  n_dr_0p2_0p4 < 5 and z_top50_slots > 0.9787
        - 0.0781841711049251 * max(0.0, 0.428145796061 - Q.tau21) * max(0.0, 0.016493544356 - Q.lam1) / 0.0007530527758270541   # -7.8%  tau21 < 0.4281 and lam1 < 0.01649
        - 0.07039294802959518 * max(0.0, 0.00750911433 - Q.mass_over_sum_pt_sq) * max(0.0, 136.785 - Q.mass_top20) / 0.20432244897571727   # -7.0%  mass_over_sum_pt_sq < 0.007509 and mass_top20 < 136.8
        + 0.06407700881331967 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 858.826171875) / 1128.32246451057   # +6.4%  n_particles < 46 and sum_pt_top40 > 858.8
        + 0.0443761324669163 * max(0.0, 0.000615484055 - Q.lam2) * max(0.0, 1.0 - Q.n_dr_0p4_up) / 0.00013830015832117198   # +4.4%  lam2 < 0.0006155 and n_dr_0p4_up < 1
        + 0.043472183347620444 * max(0.0, 15.0 - Q.n_dr_0p1_0p2) / 5.139043697478992   # +4.3%  n_dr_0p1_0p2 < 15
        + 0.04181769036643248 * max(0.0, 0.000615484055 - Q.lam2) / 0.00014995048124430673   # +4.2%  lam2 < 0.0006155
        + 0.03853251315157391 * max(0.0, 46.0 - Q.n_particles) / 6.392658823529412   # +3.9%  n_particles < 46
        - 0.036368133958755065 * max(0.0, 0.00767124277 - Q.lam1) / 0.0025275527618478888   # -3.6%  lam1 < 0.007671
        - 0.03491709900605911 * max(0.0, 0.000114064392 - Q.girth2_top5) / 1.0081866322349153e-05   # -3.5%  girth2_top5 < 0.0001141
        + 0.03278610737584964 * max(0.0, 0.00750911433 - Q.mass_over_sum_pt_sq) * max(0.0, Q.girth2_top15 - 0.004855288512) / 1.6969152412188185e-07   # +3.3%  mass_over_sum_pt_sq < 0.007509 and girth2_top15 > 0.004855
        - 0.02912347349634802 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 2.0366084033613445   # -2.9%  n_dr_0p2_0p4 < 7
        - 0.028145336270638546 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.mass_top30 - 73.331387415761) / 43.91066245439405   # -2.8%  n_particles < 46 and mass_top30 > 73.33
        - 0.027150134693364157 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.eta_0 - -0.02893371582) / 0.20595066643453763   # -2.7%  n_particles < 46 and eta_0 > -0.02893
        - 0.022112757811333296 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.04118638065 - Q.dr_0) / 0.009336793038502874   # -2.2%  n_dr_0p2_0p4 < 5 and dr_0 < 0.04119
        + 0.021162199128131987 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.003353110817) / 0.16586201546520213   # +2.1%  n_dr_0p2_0p4 < 5 and z_dr_0p1_0p2 > 0.003353
        + 0.02006429415038317 * max(0.0, 0.000349717384 - Q.lam2) / 4.1944372286634975e-05   # +2.0%  lam2 < 0.0003497
        + 0.015876752358042187 * max(0.0, 0.001393458078 - Q.z_dr_0p2_0p4) / 0.00013931799695728113   # +1.6%  z_dr_0p2_0p4 < 0.001393
        + 0.014755012940634665 * max(0.0, 0.050772907168 - Q.mass_over_sum_pt) / 0.0032556230196316804   # +1.5%  mass_over_sum_pt < 0.05077
        - 0.01236373507828065 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, 988.4375 - Q.sum_pt_top30) / 17.460270001887473   # -1.2%  n_dr_0p2_0p4 < 5 and sum_pt_top30 < 988.4
        - 0.01197752929571566 * max(0.0, Q.z_top20_slots - 0.923451750505) / 0.020619026890972596   # -1.2%  z_top20_slots > 0.9235
        - 0.009945598317993775 * max(0.0, 7.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.phi_0 - -0.040130615234) / 0.08958142834702756   # -1.0%  n_dr_0p2_0p4 < 7 and phi_0 > -0.04013
        - 0.004904462127872543 * max(0.0, 0.298200035095 - Q.max_dr) * max(0.0, Q.tau21 - 0.428145796061) / 0.0006732853594844003   # -0.5%  max_dr < 0.2982 and tau21 > 0.4281
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 14.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.35092241817143 * (0.1840187705662842
        - 0.08602340656017021 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # -8.6%  girth < 0.1207
        - 0.08322466728940314 * max(0.0, 80.784643554688 - Q.mass) / 10.849517559889705   # -8.3%  mass < 80.78
        - 0.07718706075324583 * max(0.0, Q.LHA - 0.115200825015) / 0.14925839084262735   # -7.7%  LHA > 0.1152
        + 0.062469028448961045 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +6.2%  mass < 101
        + 0.049722670332669354 * max(0.0, 6.916121244431 - Q.D2) / 4.12932700422166   # +5.0%  D2 < 6.916
        + 0.04847065751815062 * max(0.0, 0.015638355144 - Q.girth2_top15) / 0.009593305133886523   # +4.8%  girth2_top15 < 0.01564
        + 0.042559432342834644 * max(0.0, Q.width - 0.009614971338) / 0.003192189674729974   # +4.3%  width > 0.009615
        - 0.03761949616987181 * max(0.0, 94.642533639752 - Q.mass_top40) / 21.020214245469763   # -3.8%  mass_top40 < 94.64
        + 0.0341395520013932 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # +3.4%  mass < 120.6
        - 0.03033297990848635 * max(0.0, 152.688263064041 - Q.mass_top30) / 74.18247342351319   # -3.0%  mass_top30 < 152.7
        - 0.029827224252184252 * max(0.0, 0.068491501734 - Q.z_dr_0p2_0p4) / 0.04053661392560968   # -3.0%  z_dr_0p2_0p4 < 0.06849
        + 0.029226532706881036 * max(0.0, 0.012926423095 - Q.girth2_top40) / 0.006292907975779176   # +2.9%  girth2_top40 < 0.01293
        - 0.02829911603334004 * max(0.0, Q.lam1 - 0.00767124277) / 0.0027489268789339613   # -2.8%  lam1 > 0.007671
        - 0.02712358918345827 * max(0.0, 83.325535102591 - Q.mass_top40) * max(0.0, 6.916121244431 - Q.D2) / 32.02926829226556   # -2.7%  mass_top40 < 83.33 and D2 < 6.916
        - 0.0249406995293332 * max(0.0, Q.mass_over_sum_pt - 0.06030418859) / 0.03186976854304421   # -2.5%  mass_over_sum_pt > 0.0603
        - 0.024679406236753885 * max(0.0, 0.061710142531 - Q.girth) / 0.012950562309845613   # -2.5%  girth < 0.06171
        + 0.024556964000437977 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, 0.402178311348 - Q.max_dr) / 2.4354257611523584   # +2.5%  mass < 120.6 and max_dr < 0.4022
        - 0.024017157816115475 * max(0.0, 0.007277630044 - Q.girth2_top15) / 0.0028214695977655667   # -2.4%  girth2_top15 < 0.007278
        + 0.021127907512106088 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.4828873949579835   # +2.1%  n_dr_0p2_0p4 < 15
        - 0.02098740948595965 * max(0.0, 1066.481811523438 - Q.sum_pt) / 52.533044976652775   # -2.1%  sum_pt < 1066
        - 0.0209856678700254 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -2.1%  mass < 86.4
        + 0.020963414111234435 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.004007841607 - Q.girth2_top2) / 0.07347560540847334   # +2.1%  mass < 101 and girth2_top2 < 0.004008
        - 0.01853968098102775 * max(0.0, 74.251806640625 - Q.mass) * max(0.0, 0.39398368001 - Q.max_dr) / 0.39183546981015077   # -1.9%  mass < 74.25 and max_dr < 0.394
        - 0.01332734922047802 * max(0.0, Q.n_particles - 29.0) / 17.61555462184874   # -1.3%  n_particles > 29
        + 0.01287702448285902 * max(0.0, Q.e2 - 0.036805817112) / 0.004603798258454848   # +1.3%  e2 > 0.03681
        + 0.0123904485540293 * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 0.14637203090966402   # +1.2%  z_dr_0p2_0p4 < 0.1937
        - 0.011781871937731449 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.43572281599 - Q.max_dr) / 0.8636324988498371   # -1.2%  n_dr_0p2_0p4 < 15 and max_dr < 0.4357
        - 0.009801077208496819 * max(0.0, 0.470341457427 - Q.tau21) / 0.11135457214954626   # -1.0%  tau21 < 0.4703
        - 0.009150552288646549 * max(0.0, 0.004752875822 - Q.girth2_top10) / 0.0016234175755221222   # -0.9%  girth2_top10 < 0.004753
        + 0.0069359257941827525 * max(0.0, 993.56640625 - Q.sum_pt_top20) / 88.76680814814208   # +0.7%  sum_pt_top20 < 993.6
        + 0.0059870505591061015 * max(0.0, Q.eccentricity - 0.868081197276) / 0.0317658606467476   # +0.6%  eccentricity > 0.8681
        - 0.0055244030428455194 * max(0.0, Q.lam2 - 0.002396991421) / 0.00048126138920014886   # -0.6%  lam2 > 0.002397
        + 0.00496030093268629 * max(0.0, 0.577419030666 - Q.tau32) / 0.03226447382135008   # +0.5%  tau32 < 0.5774
        + 0.00480762134488886 * max(0.0, 45.595 - Q.mass_top10) / 11.863531158412167   # +0.5%  mass_top10 < 45.59
        - 0.004701086983625799 * max(0.0, Q.mass_over_sum_pt - 0.09795414517) / 0.012228974621468441   # -0.5%  mass_over_sum_pt > 0.09795
        + 0.0045261029684849656 * max(0.0, 83.325535102591 - Q.mass_top40) / 13.716491940764705   # +0.5%  mass_top40 < 83.33
        + 0.004515896718455535 * max(0.0, Q.C2 - 0.108888113871) / 0.004226185121665153   # +0.5%  C2 > 0.1089
        - 0.004216968632601086 * max(0.0, Q.mass_top20 - 103.674910639856) / 3.7955110428668526   # -0.4%  mass_top20 > 103.7
        + 0.004141858261462832 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # +0.4%  mass > 143.8
        - 0.003932280821914061 * max(0.0, 38.53125 - Q.pt_7) / 6.664217539554884   # -0.4%  pt_7 < 38.53
        - 0.002140315773812243 * max(0.0, 0.303602636177 - Q.planar_flow) * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 0.33274335411580497   # -0.2%  planar_flow < 0.3036 and n_dr_0p05_0p1 > 6
        - 0.001972404166534299 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 10.0) / 22.698038655462184   # -0.2%  n_dr_0p2_0p4 < 15 and n_dr_0p1_0p2 > 10
        + 0.0018254681566061773 * max(0.0, 74.251806640625 - Q.mass) / 8.587064460866076   # +0.2%  mass < 74.25
        + 0.0013296347265069855 * max(0.0, 69.027163795459 - Q.mass_top15) / 18.230290548016082   # +0.1%  mass_top15 < 69.03
        - 0.00120676309409543 * max(0.0, Q.n_particles - 29.0) * max(0.0, 25.0 - Q.n_dr_0_0p05) / 223.37779663865547   # -0.1%  n_particles > 29 and n_dr_0_0p05 < 25
        + 0.0009238732859062241 * max(0.0, 60.438206617337 - Q.mass_top30) / 6.992028296799824   # +0.1%  mass_top30 < 60.44
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.070323632149188 * (0.041759770961501866
        - 0.1131705112864893 * max(0.0, Q.log_sum_pt - 6.910130970417) / 0.051750869608046556   # -11.3%  log_sum_pt > 6.91
        + 0.09501898996139096 * max(0.0, Q.sum_pt_top40 - 906.60234375) / 122.36719248432757   # +9.5%  sum_pt_top40 > 906.6
        + 0.09321281245201475 * max(0.0, Q.log_sum_pt - 6.856374501323) / 0.09628579359874964   # +9.3%  log_sum_pt > 6.856
        - 0.07325240011106872 * max(0.0, Q.z_top40_slots - 0.930046498893) / 0.051778900853579946   # -7.3%  z_top40_slots > 0.93
        + 0.057267375872589295 * max(0.0, Q.sum_pt - 907.937170410156) * max(0.0, 0.0140332421 - Q.girth2_top2) / 1.486733466180341   # +5.7%  sum_pt > 907.9 and girth2_top2 < 0.01403
        + 0.047130636836058955 * max(0.0, Q.mass_top30 - 68.286969674465) / 20.391116049249682   # +4.7%  mass_top30 > 68.29
        - 0.03858625537508388 * max(0.0, Q.mass_over_sum_pt - 0.090467494167) / 0.014210394977997521   # -3.9%  mass_over_sum_pt > 0.09047
        + 0.038088801176091654 * max(0.0, Q.z_top50_slots - 0.958653609576) / 0.03427111698918734   # +3.8%  z_top50_slots > 0.9587
        - 0.0367517139343015 * max(0.0, Q.log_sum_pt - 6.92034855022) * max(0.0, 0.0140332421 - Q.girth2_top2) / 0.0005041067528121594   # -3.7%  log_sum_pt > 6.92 and girth2_top2 < 0.01403
        + 0.03175287873927574 * max(0.0, 64.0 - Q.n_particles) * max(0.0, 0.038759447634 - Q.e2) / 0.2513692835663279   # +3.2%  n_particles < 64 and e2 < 0.03876
        - 0.029905978203307315 * max(0.0, 0.0140332421 - Q.girth2_top2) / 0.010072191953219749   # -3.0%  girth2_top2 < 0.01403
        + 0.029438064906068245 * max(0.0, Q.mass - 64.485443115234) / 31.191644704866818   # +2.9%  mass > 64.49
        - 0.02539919634161911 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -2.5%  girth < 0.08589
        - 0.024709475061815818 * max(0.0, Q.sum_pt_top50 - 1061.183898925781) / 28.411282622213932   # -2.5%  sum_pt_top50 > 1061
        - 0.020363193774369237 * max(0.0, Q.mass_top50 - 157.544814151857) / 1.5571187949287246   # -2.0%  mass_top50 > 157.5
        + 0.020285883215780665 * max(0.0, Q.mass_over_sum_pt - 0.09795414517) / 0.012228974621468441   # +2.0%  mass_over_sum_pt > 0.09795
        - 0.01664737900813311 * max(0.0, Q.log_sum_pt - 6.935549248787) / 0.03715380406194132   # -1.7%  log_sum_pt > 6.936
        - 0.016446935661627547 * max(0.0, Q.mass_top50 - 91.19) / 13.751447163946986   # -1.6%  mass_top50 > 91.19
        - 0.016380586718935965 * max(0.0, 0.018076787298 - Q.girth2_top30) / 0.01083719354400756   # -1.6%  girth2_top30 < 0.01808
        - 0.016359617735994692 * max(0.0, Q.z_top30_slots - 0.934183811419) / 0.034068179997830586   # -1.6%  z_top30_slots > 0.9342
        + 0.013272608475315775 * max(0.0, Q.sum_pt_top10 - 845.53671875) / 35.972808947321965   # +1.3%  sum_pt_top10 > 845.5
        + 0.012846999491737838 * max(0.0, Q.max_dr - 0.43572281599) / 0.007711927791734812   # +1.3%  max_dr > 0.4357
        + 0.01233159712497986 * max(0.0, 0.004752875822 - Q.girth2_top10) / 0.0016234175755221222   # +1.2%  girth2_top10 < 0.004753
        - 0.011654440334925176 * max(0.0, Q.mass_top30 - 101.927236862114) / 7.286523582862451   # -1.2%  mass_top30 > 101.9
        - 0.01162119453837007 * max(0.0, Q.mass_over_sum_pt_sq - 0.029200015571) / 0.00020282738822808687   # -1.2%  mass_over_sum_pt_sq > 0.0292
        + 0.008545914895985947 * max(0.0, 0.018076787298 - Q.girth2_top30) * max(0.0, 0.864499151707 - Q.tau32) / 0.001101841236262874   # +0.9%  girth2_top30 < 0.01808 and tau32 < 0.8645
        - 0.007341790080770533 * max(0.0, Q.sum_pt_top3 - 331.25) * max(0.0, 27.0 - Q.n_dr_0p05_0p1) / 2847.1050371192227   # -0.7%  sum_pt_top3 > 331.2 and n_dr_0p05_0p1 < 27
        + 0.006843731566944544 * max(0.0, Q.log_sum_pt - 6.989450376716) / 0.021142744446604637   # +0.7%  log_sum_pt > 6.989
        - 0.006459914262838224 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # -0.6%  mass > 172.8
        - 0.006375202227667123 * max(0.0, Q.sum_pt_top15 - 951.1375) / 24.48977481165237   # -0.6%  sum_pt_top15 > 951.1
        - 0.006211710694721505 * max(0.0, 64.0 - Q.n_particles) * max(0.0, Q.mass_top5 - 3.066194584349) / 407.9645684579521   # -0.6%  n_particles < 64 and mass_top5 > 3.066
        + 0.006154090589484998 * max(0.0, Q.sum_pt_top3 - 331.25) / 148.84014507615547   # +0.6%  sum_pt_top3 > 331.2
        - 0.005463360545056339 * max(0.0, Q.max_dr - 0.240474711359) * max(0.0, 0.6418633461 - Q.tau21) / 0.023912457759965092   # -0.5%  max_dr > 0.2405 and tau21 < 0.6419
        - 0.005406234229333229 * max(0.0, Q.mass_top40 - 136.785) / 3.2712129356961093   # -0.5%  mass_top40 > 136.8
        + 0.005321670965490806 * max(0.0, Q.log_sum_pt - 6.92034855022) / 0.04509237421340201   # +0.5%  log_sum_pt > 6.92
        + 0.004343682320902273 * max(0.0, Q.max_dr - 0.240474711359) / 0.11773588635266226   # +0.4%  max_dr > 0.2405
        - 0.003976866054239087 * max(0.0, Q.sum_pt - 907.937170410156) * max(0.0, Q.pt1_dr01 - 0.253124156046) / 777.2786328712784   # -0.4%  sum_pt > 907.9 and pt1_dr01 > 0.2531
        + 0.0039045189554203007 * max(0.0, Q.log_sum_pt - 6.959293500649) / 0.028460039078747168   # +0.4%  log_sum_pt > 6.959
        + 0.0035660315896014305 * max(0.0, Q.sum_pt_top20 - 1005.0126953125) / 20.50199904518448   # +0.4%  sum_pt_top20 > 1005
        + 0.003401720187135049 * max(0.0, Q.log_sum_pt - 7.139296169016) / 0.005452502053630527   # +0.3%  log_sum_pt > 7.139
        + 0.00331642338637251 * max(0.0, Q.mass_over_sum_pt - 0.170880120467) / 0.0005536193928784717   # +0.3%  mass_over_sum_pt > 0.1709
        - 0.003163906025672102 * max(0.0, 0.051804735139 - Q.z_dr_0p2_0p4) / 0.028368155573045024   # -0.3%  z_dr_0p2_0p4 < 0.0518
        + 0.0026111736295467043 * max(0.0, Q.sum_pt_top3 - 331.25) * max(0.0, Q.n_dr_0p1_0p2 - 8.0) / 419.8329457195378   # +0.3%  sum_pt_top3 > 331.2 and n_dr_0p1_0p2 > 8
        - 0.0020867354773111387 * max(0.0, Q.sum_pt_top5 - 902.40625) / 4.122483656939338   # -0.2%  sum_pt_top5 > 902.4
        - 0.0013567409120846662 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # -0.1%  n_dr_0p2_0p4 < 13
        + 0.0012774583279607495 * max(0.0, Q.mass_top50 - 157.544814151857) * max(0.0, Q.z_dr_0p05_0p1 - 0.591232848167) / 0.009747485253060875   # +0.1%  mass_top50 > 157.5 and z_dr_0p05_0p1 > 0.5912
        + 0.000893046128091667 * max(0.0, Q.z_dr_0_0p05 - 0.878906026483) / 0.015835254093555608   # +0.1%  z_dr_0_0p05 > 0.8789
        - 8.255061002397326e-05 * max(0.0, Q.sum_pt_top5 - 631.275) / 56.51543623760633   # -0.0%  sum_pt_top5 > 631.3
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 13.96;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.96499940232324 * (0.018402880443163263
        - 0.11898269935679073 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # -11.9%  mass < 101
        - 0.10924842216977072 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) / 0.039273301687911995   # -10.9%  mass_over_sum_pt > 0.05077
        - 0.09621932663252708 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # -9.6%  mass < 120.6
        + 0.08513947288925558 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # +8.5%  mass < 92.86
        + 0.06328020994488918 * max(0.0, 172.8 - Q.mass) / 83.7879223272957   # +6.3%  mass < 172.8
        - 0.05213317305506012 * max(0.0, 0.009614971338 - Q.width) / 0.003502544629791685   # -5.2%  width < 0.009615
        + 0.044905366617120086 * max(0.0, 0.019863807341 - Q.mass_over_sum_pt_sq) / 0.011777496153852728   # +4.5%  mass_over_sum_pt_sq < 0.01986
        - 0.04301546873534632 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063025210084   # -4.3%  n_dr_0p2_0p4 < 21
        + 0.03459782104451727 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # +3.5%  mass < 86.4
        + 0.028707335725833002 * max(0.0, Q.girth2_top20 - 0.0018230789) / 0.006286421415254086   # +2.9%  girth2_top20 > 0.001823
        + 0.0230246888283734 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 0.32206342639914515   # +2.3%  mass_over_sum_pt > 0.05077 and n_dr_0p2_0p4 < 21
        + 0.022028674175002008 * max(0.0, 0.007259287357 - Q.lam1) / 0.0022497701299997253   # +2.2%  lam1 < 0.007259
        + 0.02169134078062232 * max(0.0, 0.008031209355 - Q.girth2_top20) / 0.0030984903862893757   # +2.2%  girth2_top20 < 0.008031
        + 0.020708609804759866 * max(0.0, 0.006427166767 - Q.lam2) / 0.00514670292238681   # +2.1%  lam2 < 0.006427
        + 0.020433295277931755 * max(0.0, Q.lam1 - 0.008241985248) / 0.002595658483103056   # +2.0%  lam1 > 0.008242
        + 0.02015927861846456 * max(0.0, 85.412093844921 - Q.mass_top10) / 39.981040136243024   # +2.0%  mass_top10 < 85.41
        + 0.019051028114251706 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # +1.9%  mass_top50 < 71.8
        + 0.018091403848718175 * max(0.0, 0.047553086095 - Q.e2) / 0.018774568371458196   # +1.8%  e2 < 0.04755
        - 0.014791386443918616 * max(0.0, 0.013514311784 - Q.girth2_top50) / 0.006620974830265835   # -1.5%  girth2_top50 < 0.01351
        + 0.012754467354626398 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.908491230011 - Q.z_dr_0_0p05) / 0.02599640666253605   # +1.3%  mass_over_sum_pt > 0.05077 and z_dr_0_0p05 < 0.9085
        + 0.012536548706012124 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.002396991421 - Q.lam2) / 3.6218014529889185e-05   # +1.3%  mass_over_sum_pt > 0.05077 and lam2 < 0.002397
        + 0.011923178835068595 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.286492615938 - Q.z_dr_0p1_0p2) / 0.0037357069815239703   # +1.2%  mass_over_sum_pt > 0.05077 and z_dr_0p1_0p2 < 0.2865
        - 0.011004763129669987 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 17.0 - Q.n_dr_0p1_0p2) / 0.1434593051377074   # -1.1%  mass_over_sum_pt > 0.05077 and n_dr_0p1_0p2 < 17
        + 0.009933103094743505 * max(0.0, Q.LHA - 0.228402115913) / 0.06082067856197019   # +1.0%  LHA > 0.2284
        + 0.009846119240704256 * max(0.0, Q.girth2_top20 - 0.0018230789) * max(0.0, 0.43572281599 - Q.max_dr) / 0.0005012106213039546   # +1.0%  girth2_top20 > 0.001823 and max_dr < 0.4357
        + 0.008094558426813685 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.142391438037 - Q.C2) / 0.0022362354021204   # +0.8%  mass_over_sum_pt > 0.05077 and C2 < 0.1424
        + 0.007401912865443321 * max(0.0, Q.girth2_top3 - 0.010023689877) * max(0.0, 4.450168704987 - Q.D2) / 0.0041143288114283754   # +0.7%  girth2_top3 > 0.01002 and D2 < 4.45
        - 0.006599751846532606 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, Q.max_pair_mass - 13.047927274731) / 0.3749186717396242   # -0.7%  mass_over_sum_pt > 0.05077 and max_pair_mass > 13.05
        + 0.006361424577981284 * max(0.0, Q.lam1 - 0.008241985248) * max(0.0, 68.125 - Q.pt_4) / 0.046937438507826536   # +0.6%  lam1 > 0.008242 and pt_4 < 68.12
        - 0.006339152580516528 * max(0.0, 92.85979309082 - Q.mass) * max(0.0, 1082.54765625 - Q.sum_pt_top15) / 2825.6490989367844   # -0.6%  mass < 92.86 and sum_pt_top15 < 1083
        - 0.005455106430137838 * max(0.0, Q.lam1 - 0.011744050682) / 0.0018275243697888329   # -0.5%  lam1 > 0.01174
        + 0.005248548391162567 * max(0.0, 0.043586218357 - Q.e2) / 0.015487320383570933   # +0.5%  e2 < 0.04359
        - 0.0051422607228987895 * max(0.0, 0.030297144316 - Q.e2) / 0.0068225745792513896   # -0.5%  e2 < 0.0303
        - 0.005078639820249189 * max(0.0, Q.e2 - 0.055571487173) / 0.0011197307961334129   # -0.5%  e2 > 0.05557
        - 0.005073177094337767 * max(0.0, Q.LHA - 0.228402115913) * max(0.0, 3.345338582993 - Q.D2) / 0.10056812770798138   # -0.5%  LHA > 0.2284 and D2 < 3.345
        - 0.0037368972398839203 * max(0.0, Q.girth2_top20 - 0.016559833876) / 0.001314598562186081   # -0.4%  girth2_top20 > 0.01656
        + 0.002177922613357512 * max(0.0, Q.mass_over_sum_pt - 0.050772907168) * max(0.0, 0.079528808594 - Q.eta_0) / 0.0033264609672938325   # +0.2%  mass_over_sum_pt > 0.05077 and eta_0 < 0.07953
        + 0.0020333591203010974 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, 1008.9353515625 - Q.sum_pt_top50) / 748.3773123240593   # +0.2%  mass < 120.6 and sum_pt_top50 < 1009
        + 0.0019703431212778847 * max(0.0, 6.811175180312 - Q.log_sum_pt) / 0.004882419406465069   # +0.2%  log_sum_pt < 6.811
        + 0.0018072571398670817 * max(0.0, Q.LHA - 0.404204003833) / 0.0031577141977621142   # +0.2%  LHA > 0.4042
        - 0.001152189239255593 * max(0.0, Q.e2 - 0.055571487173) * max(0.0, 0.368618160486 - Q.max_dr) / 1.040948113205381e-05   # -0.1%  e2 > 0.05557 and max_dr < 0.3686
        + 0.0010787083677502162 * max(0.0, Q.mass_top10 - 99.066784770599) / 0.7458098208346458   # +0.1%  mass_top10 > 99.07
        + 0.0010416079782557023 * max(0.0, Q.lam1 - 0.024570249917) / 0.0002026642440036589   # +0.1%  lam1 > 0.02457
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 24.51;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.50596202029161 * (-0.013739019606600752
        - 0.13903213183444887 * max(0.0, 91.19 - Q.mass) / 16.464230590854466   # -13.9%  mass < 91.19
        + 0.06569465992013601 * max(0.0, 0.009606007381 - Q.e2_sq) / 0.003497865944155776   # +6.6%  e2_sq < 0.009606
        + 0.05935341009173655 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +5.9%  mass < 101
        - 0.058610352042947456 * max(0.0, 0.008031986041 - Q.girth2_top40) / 0.0025145928123983495   # -5.9%  girth2_top40 < 0.008032
        - 0.05785906332032408 * max(0.0, 0.008031209355 - Q.girth2_top20) / 0.0030984903862893757   # -5.8%  girth2_top20 < 0.008031
        - 0.04764062094706919 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -4.8%  girth < 0.08589
        + 0.04485080555238176 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.390781164169 - Q.max_dr) / 1.1947278759545992   # +4.5%  mass < 101 and max_dr < 0.3908
        - 0.04322724457766265 * max(0.0, 91.19 - Q.mass) * max(0.0, 0.390781164169 - Q.max_dr) / 0.7740281265073297   # -4.3%  mass < 91.19 and max_dr < 0.3908
        + 0.04031612385236649 * max(0.0, 0.012926423095 - Q.girth2_top40) / 0.006292907975779176   # +4.0%  girth2_top40 < 0.01293
        + 0.04013495097425588 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # +4.0%  mass < 120.6
        - 0.038885559759719385 * max(0.0, 97.930041729355 - Q.mass_top50) * max(0.0, 1.601009327173 - Q.D2) / 2.279411140212284   # -3.9%  mass_top50 < 97.93 and D2 < 1.601
        + 0.035490981464871306 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 1.601009327173 - Q.D2) / 2.8196790984068856   # +3.5%  mass < 101 and D2 < 1.601
        + 0.03198218106752662 * max(0.0, 0.008840538245 - Q.girth2_top40) / 0.0031086216346541064   # +3.2%  girth2_top40 < 0.008841
        + 0.025591681277098485 * max(0.0, 0.006374177987 - Q.girth2_top20) / 0.0019877511397026312   # +2.6%  girth2_top20 < 0.006374
        - 0.02464819500888184 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -2.5%  mass < 86.4
        - 0.023891981299428797 * max(0.0, 0.090467494167 - Q.mass_over_sum_pt) / 0.0178873033081874   # -2.4%  mass_over_sum_pt < 0.09047
        + 0.02369303772223344 * max(0.0, 0.320332145368 - Q.LHA) / 0.07499257253386664   # +2.4%  LHA < 0.3203
        + 0.023678151783554525 * max(0.0, 82.85408782959 - Q.mass) / 11.83453618826746   # +2.4%  mass < 82.85
        - 0.023213639170442155 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) / 0.05824141942126331   # -2.3%  z_dr_0p2_0p4 < 0.09123
        + 0.02040652824968089 * max(0.0, 0.007709915821 - Q.girth2_top40) / 0.0022972993366394247   # +2.0%  girth2_top40 < 0.00771
        - 0.013535239218977778 * max(0.0, 0.013514311784 - Q.girth2_top50) / 0.006620974830265835   # -1.4%  girth2_top50 < 0.01351
        - 0.011831849879067073 * max(0.0, 86.252206812802 - Q.mass_top30) / 18.207265524434344   # -1.2%  mass_top30 < 86.25
        + 0.011348925715086769 * max(0.0, 0.027935993578 - Q.e2) / 0.005715276962294592   # +1.1%  e2 < 0.02794
        + 0.010493826847596257 * max(0.0, 76.415438713532 - Q.mass_top30) / 12.67595855468548   # +1.0%  mass_top30 < 76.42
        + 0.009597234206373774 * max(0.0, 66.841467317407 - Q.mass_top20) / 12.904985456467061   # +1.0%  mass_top20 < 66.84
        - 0.008514480316332226 * max(0.0, 97.930041729355 - Q.mass_top50) / 22.035325884594407   # -0.9%  mass_top50 < 97.93
        + 0.008164754389988064 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.499317836761 - Q.z_1st) / 1.5556551236660203   # +0.8%  n_dr_0p2_0p4 < 13 and z_1st < 0.4993
        + 0.008160529080657503 * max(0.0, 1.788105106354 - Q.D2) * max(0.0, 9.0 - Q.n_dr_0p2_0p4) / 1.537145490198633   # +0.8%  D2 < 1.788 and n_dr_0p2_0p4 < 9
        - 0.007169224742873784 * max(0.0, 0.990637830118 - Q.z_top50_slots) / 0.004783315002560602   # -0.7%  z_top50_slots < 0.9906
        + 0.006906346523629467 * max(0.0, 79.21003612387 - Q.mass_top50) / 10.657539313204703   # +0.7%  mass_top50 < 79.21
        + 0.006648399213563722 * max(0.0, 0.004744913615 - Q.z_dr_0p2_0p4) / 0.0009175254372549359   # +0.7%  z_dr_0p2_0p4 < 0.004745
        - 0.005177180525673896 * max(0.0, 1.788105106354 - Q.D2) * max(0.0, 0.007820314762 - Q.girth2_top50) / 0.00026955979916256045   # -0.5%  D2 < 1.788 and girth2_top50 < 0.00782
        + 0.0042346499861312505 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # +0.4%  n_dr_0p2_0p4 < 13
        - 0.003477112095531227 * max(0.0, 0.056027559564 - Q.C2) / 0.009882646645223543   # -0.3%  C2 < 0.05603
        - 0.003275559339031881 * max(0.0, 91.19 - Q.mass) * max(0.0, 0.33780374676 - Q.pt_dispersion) / 0.4764307098452709   # -0.3%  mass < 91.19 and pt_dispersion < 0.3378
        + 0.0025501347933610447 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.536493504079 - Q.planar_flow) / 1.1435727084895067   # +0.3%  n_dr_0p2_0p4 < 13 and planar_flow < 0.5365
        - 0.002331006369881627 * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.2000453781512605   # -0.2%  n_dr_0p2_0p4 < 2
        - 0.00227728642125976 * max(0.0, 91.19 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.402664637566) / 0.4065322951460936   # -0.2%  mass < 91.19 and z_dr_0p05_0p1 > 0.4027
        + 0.0015017337956262052 * max(0.0, 1.788105106354 - Q.D2) * max(0.0, 0.006118005947 - Q.girth2_top50) / 5.688007809906865e-05   # +0.2%  D2 < 1.788 and girth2_top50 < 0.006118
        - 0.001347612199842459 * max(0.0, 0.193997652829 - Q.max_dr) * max(0.0, Q.dr_8 - 0.008114792206) / 0.00010748644137982293   # -0.1%  max_dr < 0.194 and dr_8 > 0.008115
        + 0.0012699174723495496 * max(0.0, 76.415438713532 - Q.mass_top30) * max(0.0, 1.601009327173 - Q.D2) / 0.1607834566068122   # +0.1%  mass_top30 < 76.42 and D2 < 1.601
        + 0.0009233103647131782 * max(0.0, 0.320332145368 - Q.LHA) * max(0.0, 0.355009326339 - Q.pt_dispersion) / 0.002851920384284237   # +0.1%  LHA < 0.3203 and pt_dispersion < 0.355
        + 0.0007392857464862142 * max(0.0, 0.193997652829 - Q.max_dr) / 0.0017167136084653724   # +0.1%  max_dr < 0.194
        + 0.0003231008391987564 * max(0.0, 86.4 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.402664637566) / 0.2173575706421958   # +0.0%  mass < 86.4 and z_dr_0p05_0p1 > 0.4027
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 17.13;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.132566696985684 * (-0.03096513690531575
        - 0.06564241490614024 * max(0.0, Q.mass_over_sum_pt - 0.09795414517) / 0.012228974621468441   # -6.6%  mass_over_sum_pt > 0.09795
        - 0.06284424797152682 * max(0.0, Q.mass - 101.049709320068) / 12.310843098263781   # -6.3%  mass > 101
        + 0.05615721461216562 * max(0.0, Q.mass - 80.784643554688) / 19.806095548570884   # +5.6%  mass > 80.78
        + 0.05592496666745833 * max(0.0, Q.mass - 64.485443115234) / 31.191644704866818   # +5.6%  mass > 64.49
        + 0.05441962634861365 * max(0.0, 0.097499583662 - Q.girth) / 0.03657082438923025   # +5.4%  girth < 0.0975
        - 0.05070230246394119 * max(0.0, 0.013977372691 - Q.mass_over_sum_pt_sq) / 0.006903769640131633   # -5.1%  mass_over_sum_pt_sq < 0.01398
        + 0.04679105980774886 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) * max(0.0, 1115.722741699219 - Q.sum_pt) / 2.3366981130955806   # +4.7%  mass_over_sum_pt > 0.07697 and sum_pt < 1116
        + 0.04142562957841591 * max(0.0, Q.mass_over_sum_pt - 0.088731426731) / 0.014771713814206479   # +4.1%  mass_over_sum_pt > 0.08873
        + 0.040831324221398936 * max(0.0, Q.mass - 87.363773345947) / 16.577408637801742   # +4.1%  mass > 87.36
        - 0.036679682962707474 * max(0.0, Q.mass_top50 - 82.04491364955) / 17.710574785086557   # -3.7%  mass_top50 > 82.04
        + 0.03565357283661005 * max(0.0, Q.girth2_top20 - 0.008031209355) / 0.00290685235713101   # +3.6%  girth2_top20 > 0.008031
        - 0.03417496099097748 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063025210084   # -3.4%  n_dr_0p2_0p4 < 21
        + 0.031882550413757337 * max(0.0, Q.girth2_top40 - 0.005196965925) / 0.004725809282736856   # +3.2%  girth2_top40 > 0.005197
        - 0.030227588838556894 * max(0.0, 0.020485236462 - Q.lam1) / 0.013097460086155891   # -3.0%  lam1 < 0.02049
        + 0.02646340576528172 * max(0.0, Q.mass_over_sum_pt - 0.076966318366) / 0.020241059768600602   # +2.6%  mass_over_sum_pt > 0.07697
        + 0.024745170789484072 * max(0.0, 0.008241985248 - Q.lam1) / 0.0029450268440803277   # +2.5%  lam1 < 0.008242
        + 0.02438452549860836 * max(0.0, Q.mass_top50 - 97.930041729355) / 11.917644068610143   # +2.4%  mass_top50 > 97.93
        - 0.02325874453085869 * max(0.0, Q.mass_top40 - 79.554505888974) / 17.06549122004638   # -2.3%  mass_top40 > 79.55
        + 0.020717369254920386 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # +2.1%  sum_pt < 1013
        + 0.020489196723690706 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) / 0.05824141942126331   # +2.0%  z_dr_0p2_0p4 < 0.09123
        - 0.019312271223026756 * max(0.0, 0.001411893759 - Q.lam2) / 0.0006681998875481002   # -1.9%  lam2 < 0.001412
        - 0.018684385491177957 * max(0.0, Q.girth2_top20 - 0.005718442372) / 0.0037490635294349373   # -1.9%  girth2_top20 > 0.005718
        - 0.01694643671463322 * max(0.0, 0.006166777647 - Q.mass_over_sum_pt_sq) / 0.0012952540801658948   # -1.7%  mass_over_sum_pt_sq < 0.006167
        + 0.015834256822245107 * max(0.0, 0.007463984647 - Q.girth2_top30) / 0.0023370797259281294   # +1.6%  girth2_top30 < 0.007464
        + 0.015758751944147525 * max(0.0, Q.n_particles - 51.0) / 3.9737042016806723   # +1.6%  n_particles > 51
        + 0.015308643239235663 * max(0.0, 0.007709915821 - Q.girth2_top40) / 0.0022972993366394247   # +1.5%  girth2_top40 < 0.00771
        - 0.010254831166958371 * max(0.0, Q.e2 - 0.025159193203) / 0.010276419760149801   # -1.0%  e2 > 0.02516
        + 0.009467815172575882 * max(0.0, 1001.52314453125 - Q.sum_pt_top40) / 26.72885978092069   # +0.9%  sum_pt_top40 < 1002
        - 0.008492499124504518 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) / 7.4828873949579835   # -0.8%  n_dr_0p2_0p4 < 15
        + 0.007353904561609286 * max(0.0, Q.girth2_top40 - 0.005196965925) * max(0.0, 911.9328125 - Q.sum_pt_top30) / 0.19161403745216243   # +0.7%  girth2_top40 > 0.005197 and sum_pt_top30 < 911.9
        - 0.007313450948194624 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -0.7%  mass > 143.8
        - 0.007074398072135096 * max(0.0, Q.mass - 125.1) / 7.098975738857251   # -0.7%  mass > 125.1
        - 0.00682937630745561 * max(0.0, Q.n_particles - 51.0) * max(0.0, Q.z_top50_slots - 0.970443639316) / 0.04048127714655134   # -0.7%  n_particles > 51 and z_top50_slots > 0.9704
        + 0.005513348296731087 * max(0.0, 1001.52314453125 - Q.sum_pt_top40) * max(0.0, 6.811175180312 - Q.log_sum_pt) / 1.4602943926211116   # +0.6%  sum_pt_top40 < 1002 and log_sum_pt < 6.811
        + 0.00531774777087035 * max(0.0, 0.154838323593 - Q.z_dr_0p1_0p2) / 0.06710802455713627   # +0.5%  z_dr_0p1_0p2 < 0.1548
        + 0.004917681044096994 * max(0.0, 1.788105106354 - Q.D2) / 0.2865857032958301   # +0.5%  D2 < 1.788
        - 0.0049099531135970145 * max(0.0, 1001.52314453125 - Q.sum_pt_top40) * max(0.0, 0.39398368001 - Q.max_dr) / 0.8829598584890831   # -0.5%  sum_pt_top40 < 1002 and max_dr < 0.394
        + 0.0048314947619072296 * max(0.0, 0.001411893759 - Q.lam2) * max(0.0, Q.n_dr_0p05_0p1 - 4.0) / 0.004634859016466834   # +0.5%  lam2 < 0.001412 and n_dr_0p05_0p1 > 4
        - 0.003961706576395275 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, Q.z_0 - 0.088724280345) / 2.2982721947074065   # -0.4%  sum_pt < 1013 and z_0 > 0.08872
        + 0.003558324172876822 * max(0.0, Q.C2 - 0.072798889503) / 0.01280801443367811   # +0.4%  C2 > 0.0728
        - 0.0034722883305086517 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, 43.0 - Q.n_real_top50) / 70.7268566233913   # -0.3%  sum_pt < 1013 and n_real_top50 < 43
        + 0.003152194699854759 * max(0.0, Q.mass_top15 - 30.359943489662) / 32.64933730076125   # +0.3%  mass_top15 > 30.36
        - 0.0028321596139311393 * max(0.0, 1011.52392578125 - Q.sum_pt_top30) / 51.02618157784598   # -0.3%  sum_pt_top30 < 1012
        + 0.0024820106597371076 * max(0.0, Q.z_dr_0p1_0p2 - 0.218764226139) / 0.061986228098286786   # +0.2%  z_dr_0p1_0p2 > 0.2188
        + 0.002456281713334889 * max(0.0, Q.mass - 87.363773345947) * max(0.0, 6.903422848462 - Q.log_sum_pt) / 0.2991521451407768   # +0.2%  mass > 87.36 and log_sum_pt < 6.903
        - 0.002108192485694971 * max(0.0, 0.983076389702 - Q.z_top40_slots) / 0.013300445982205594   # -0.2%  z_top40_slots < 0.9831
        + 0.001970425813596971 * max(0.0, 0.813850690953 - Q.z_top15_slots) / 0.039447636639057045   # +0.2%  z_top15_slots < 0.8139
        - 0.001963190926891109 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.mean_eta2 - 0.006802603323) / 0.003966385337387449   # -0.2%  n_dr_0p2_0p4 < 21 and mean_eta2 > 0.006803
        - 0.0015786283753171954 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, 0.001225592976 - Q.mean_eta2) / 0.0024016579624223174   # -0.2%  sum_pt < 1013 and mean_eta2 < 0.001226
        - 0.0010632379788318657 * max(0.0, Q.n_particles - 51.0) * max(0.0, 0.899011841416 - Q.tau32) / 0.9885069347981912   # -0.1%  n_particles > 51 and tau32 < 0.899
        - 0.0009370004225105776 * max(0.0, Q.z_dr_0p1_0p2 - 0.218764226139) * max(0.0, Q.mean_phi - 0.000440474624) / 7.987388864459947e-06   # -0.1%  z_dr_0p1_0p2 > 0.2188 and mean_phi > 0.0004405
        + 0.0009275572725537543 * max(0.0, 15.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.334047731757) / 0.09686656938851342   # +0.1%  n_dr_0p2_0p4 < 15 and z_dr_0p1_0p2 > 0.334
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 15.11;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.109850828407499 * (0.039718974764988375
        + 0.18797208809842586 * max(0.0, 136.785 - Q.mass_top50) / 53.22362615511637   # +18.8%  mass_top50 < 136.8
        - 0.16832110803855171 * max(0.0, 160.8 - Q.mass_top50) / 74.23410092497848   # -16.8%  mass_top50 < 160.8
        - 0.12881420462475265 * max(0.0, Q.mass - 62.55) / 32.65527990955991   # -12.9%  mass > 62.55
        - 0.07086726742669902 * max(0.0, 0.02146577947 - Q.girth2_top15) / 0.0147044065440157   # -7.1%  girth2_top15 < 0.02147
        + 0.06879937948604178 * max(0.0, Q.mass - 82.85408782959) / 18.721669902205264   # +6.9%  mass > 82.85
        + 0.036987775923922245 * max(0.0, Q.mass - 92.85979309082) / 14.482733665646967   # +3.7%  mass > 92.86
        + 0.033275658183384973 * max(0.0, 136.785 - Q.mass) / 52.08717491697615   # +3.3%  mass < 136.8
        - 0.031357178597985515 * max(0.0, 0.043628720567 - Q.girth) / 0.0063254006084753804   # -3.1%  girth < 0.04363
        - 0.028083742122458896 * max(0.0, 91.288535717504 - Q.mass_top40) / 18.56549606317863   # -2.8%  mass_top40 < 91.29
        + 0.021743628557080525 * max(0.0, 0.00625977218 - Q.girth2_top40) / 0.0014618229210262657   # +2.2%  girth2_top40 < 0.00626
        + 0.021331428585073683 * max(0.0, 0.209102506978 - Q.LHA) / 0.020957818638470994   # +2.1%  LHA < 0.2091
        - 0.0199103813468826 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -2.0%  mass > 143.8
        + 0.019704602816648924 * max(0.0, 80.890431271924 - Q.mass_top40) / 12.434053057794845   # +2.0%  mass_top40 < 80.89
        - 0.019285970317263482 * max(0.0, 78.955574164508 - Q.mass_top15) / 24.97819492328375   # -1.9%  mass_top15 < 78.96
        + 0.015635063320562956 * max(0.0, 120.60000000000001 - Q.mass_top40) / 41.60439949481591   # +1.6%  mass_top40 < 120.6
        + 0.012890124698601046 * max(0.0, 949.91689453125 - Q.sum_pt) / 6.913716555695057   # +1.3%  sum_pt < 949.9
        + 0.011704010121718183 * max(0.0, Q.mass - 53.87361907959) * max(0.0, Q.eccentricity - 0.588471729833) / 9.47619031832216   # +1.2%  mass > 53.87 and eccentricity > 0.5885
        + 0.011237433612852711 * max(0.0, 0.009962397174 - Q.girth2_top15) / 0.004894698509240601   # +1.1%  girth2_top15 < 0.009962
        - 0.010755210835806583 * max(0.0, 6.856374501323 - Q.log_sum_pt) / 0.008053458833961623   # -1.1%  log_sum_pt < 6.856
        + 0.009821493500745826 * max(0.0, 0.02146577947 - Q.girth2_top15) * max(0.0, Q.n_particles - 41.0) / 0.10416207626617284   # +1.0%  girth2_top15 < 0.02147 and n_particles > 41
        + 0.008638704658359924 * max(0.0, 0.005240342196 - Q.lam1) / 0.0011655826082229445   # +0.9%  lam1 < 0.00524
        - 0.008050792135792266 * max(0.0, Q.width - 0.025866900997) / 0.00046535522878793676   # -0.8%  width > 0.02587
        + 0.006528218613885303 * max(0.0, 956.21328125 - Q.sum_pt_top40) / 14.004510341849162   # +0.7%  sum_pt_top40 < 956.2
        + 0.005736995720384323 * max(0.0, 52.713279629696 - Q.mass_top15) / 10.29374585732715   # +0.6%  mass_top15 < 52.71
        + 0.0053067971558146635 * max(0.0, Q.girth - 0.097499583662) / 0.008315821681586878   # +0.5%  girth > 0.0975
        - 0.004899632837684426 * max(0.0, 0.930046498893 - Q.z_top40_slots) / 0.002595351290643435   # -0.5%  z_top40_slots < 0.93
        - 0.004643312420901657 * max(0.0, 0.02783744745 - Q.girth) / 0.0024110754510684616   # -0.5%  girth < 0.02784
        + 0.004502169362164827 * max(0.0, 0.00363885588 - Q.girth2) / 0.0004964601451473658   # +0.5%  girth2 < 0.003639
        + 0.003211214853449399 * max(0.0, 0.00625977218 - Q.girth2_top40) * max(0.0, 1022.583984375 - Q.sum_pt) / 0.03084725782921442   # +0.3%  girth2_top40 < 0.00626 and sum_pt < 1023
        - 0.002837937156164676 * max(0.0, 949.91689453125 - Q.sum_pt) * max(0.0, 0.412129651204 - Q.z_0) / 1.510776463654286   # -0.3%  sum_pt < 949.9 and z_0 < 0.4121
        + 0.0027083977903087624 * max(0.0, 6.856374501323 - Q.log_sum_pt) * max(0.0, Q.z_top20_slots - 0.896541111574) / 0.0002016422589223067   # +0.3%  log_sum_pt < 6.856 and z_top20_slots > 0.8965
        + 0.0023678413103755563 * max(0.0, 120.60000000000001 - Q.mass_top40) * max(0.0, Q.max_dr - 0.379163956642) / 0.6764193058725948   # +0.2%  mass_top40 < 120.6 and max_dr > 0.3792
        - 0.002295961942033561 * max(0.0, Q.n_dr_0p1_0p2 - 21.0) / 1.3862773109243698   # -0.2%  n_dr_0p1_0p2 > 21
        + 0.0022551284886363067 * max(0.0, 6.856374501323 - Q.log_sum_pt) * max(0.0, 0.082575402011 - Q.dr_7) / 0.0002360890340260714   # +0.2%  log_sum_pt < 6.856 and dr_7 < 0.08258
        + 0.001503092769971388 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +0.2%  mass > 172.8
        - 0.0013877841381464513 * max(0.0, 956.21328125 - Q.sum_pt_top40) * max(0.0, Q.z_top15_slots - 0.794624168612) / 0.41323682564243985   # -0.1%  sum_pt_top40 < 956.2 and z_top15_slots > 0.7946
        - 0.001106438235592624 * max(0.0, Q.e2 - 0.065240035206) / 0.0004075092339807838   # -0.1%  e2 > 0.06524
        - 0.0010567069403954786 * max(0.0, 956.21328125 - Q.sum_pt_top40) * max(0.0, Q.z_top30_slots - 0.97348863653) / 0.05156393724254791   # -0.1%  sum_pt_top40 < 956.2 and z_top30_slots > 0.9735
        + 0.0009278353774959244 * max(0.0, 0.930046498893 - Q.z_top40_slots) * max(0.0, Q.n_pt_above_10 - 28.0) / 0.010703251084309638   # +0.1%  z_top40_slots < 0.93 and n_pt_above_10 > 28
        - 0.0006538732710344632 * max(0.0, Q.z_dr_0p05_0p1 - 0.710821145773) / 0.01402575874569485   # -0.1%  z_dr_0p05_0p1 > 0.7108
        - 0.0006043319025688007 * max(0.0, 80.890431271924 - Q.mass_top40) * max(0.0, Q.eta_1 - 0.057922363281) / 0.001437044328673227   # -0.1%  mass_top40 < 80.89 and eta_1 > 0.05792
        - 0.0002303762521580605 * max(0.0, Q.sum_pt_top50 - 1245.696667480468) / 7.470315304933702   # -0.0%  sum_pt_top50 > 1246
        + 4.87064512220916e-05 * max(0.0, 0.047553086095 - Q.e2) / 0.018774568371458196   # +0.0%  e2 < 0.04755
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 11.12;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.124778063250577 * (0.3678381075383563
        - 0.13167887798144848 * max(0.0, 0.024419631481 - Q.girth2_top5) / 0.018899543128555206   # -13.2%  girth2_top5 < 0.02442
        - 0.11066791459197478 * max(0.0, 120.60000000000001 - Q.mass) / 38.82790413032879   # -11.1%  mass < 120.6
        - 0.08606940511400236 * max(0.0, 0.065240035206 - Q.e2) / 0.03476194878141159   # -8.6%  e2 < 0.06524
        + 0.06503815256577947 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # +6.5%  mass < 86.4
        + 0.04991664125585304 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +5.0%  mass < 80.4
        - 0.049433995023254816 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # -4.9%  girth < 0.1207
        + 0.04718352826746929 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) / 0.05824141942126331   # +4.7%  z_dr_0p2_0p4 < 0.09123
        - 0.04717299597917104 * max(0.0, Q.z_dr_0_0p05 - 0.767473447323) / 0.05221087980608241   # -4.7%  z_dr_0_0p05 > 0.7675
        - 0.03795110631753931 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -3.8%  mass > 143.8
        + 0.0365950258075859 * max(0.0, Q.girth2_top10 - 0.000237176831) / 0.006544219018306094   # +3.7%  girth2_top10 > 0.0002372
        - 0.036040437932018866 * max(0.0, Q.lam1 - 0.001868040786) / 0.006218119096268822   # -3.6%  lam1 > 0.001868
        + 0.03472052006223841 * max(0.0, 0.024419631481 - Q.girth2_top5) * max(0.0, 656.384375 - Q.sum_pt_top3) / 3.538216833032187   # +3.5%  girth2_top5 < 0.02442 and sum_pt_top3 < 656.4
        - 0.03356839191744007 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # -3.4%  n_dr_0p2_0p4 < 13
        + 0.03275169682597524 * max(0.0, Q.mass_top50 - 136.785) / 4.25098606820319   # +3.3%  mass_top50 > 136.8
        + 0.028172282988299463 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, 0.470341457427 - Q.tau21) / 3.2648902511729423   # +2.8%  mass < 120.6 and tau21 < 0.4703
        - 0.017443291912268132 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -1.7%  mass < 62.55
        - 0.01707784833404246 * max(0.0, 0.402178311348 - Q.max_dr) / 0.059113559416633016   # -1.7%  max_dr < 0.4022
        - 0.01439498427212375 * max(0.0, 0.065240035206 - Q.e2) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 0.004800315654529198   # -1.4%  e2 < 0.06524 and z_dr_0p1_0p2 < 0.2188
        + 0.013008546498535336 * max(0.0, 0.069404718919 - Q.dr_1) / 0.02808839962534407   # +1.3%  dr_1 < 0.0694
        + 0.012183348878682571 * max(0.0, 0.005402330897 - Q.girth2_top30) / 0.0012312932478798142   # +1.2%  girth2_top30 < 0.005402
        + 0.011448245773818512 * max(0.0, 0.064132973195 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.14948521527659667   # +1.1%  dr_0 < 0.06413 and n_dr_0p2_0p4 > 2
        - 0.011181685173603555 * max(0.0, 6.0 - Q.n_dr_0p2_0p4) / 1.5418504201680672   # -1.1%  n_dr_0p2_0p4 < 6
        - 0.00979302118334073 * max(0.0, Q.mass - 162.836349487305) / 1.5098322961374488   # -1.0%  mass > 162.8
        - 0.008099802018837393 * max(0.0, 959.095727539062 - Q.sum_pt_top50) / 9.937940549204788   # -0.8%  sum_pt_top50 < 959.1
        + 0.007264843106576313 * max(0.0, 0.024419631481 - Q.girth2_top5) * max(0.0, Q.n_particles - 34.0) / 0.22539788738580874   # +0.7%  girth2_top5 < 0.02442 and n_particles > 34
        - 0.007087821378922066 * max(0.0, Q.girth2_top10 - 0.007678543663) / 0.002556946129466314   # -0.7%  girth2_top10 > 0.007679
        - 0.006112519119548642 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, 1.976207274199 - Q.D2) / 10.810678996261903   # -0.6%  mass < 120.6 and D2 < 1.976
        + 0.004518349869125069 * max(0.0, 0.091225683689 - Q.z_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.273806282878) / 0.004726553035726654   # +0.5%  z_dr_0p2_0p4 < 0.09123 and max_dr > 0.2738
        + 0.004458747624449803 * max(0.0, 120.60000000000001 - Q.mass) * max(0.0, Q.mass_top3 - 16.899120053094) / 125.1697040938471   # +0.4%  mass < 120.6 and mass_top3 > 16.9
        + 0.0038128127900853895 * max(0.0, 0.240474711359 - Q.max_dr) / 0.0051369759456107505   # +0.4%  max_dr < 0.2405
        + 0.0033028088824245743 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.065454679258 - Q.dr_2) / 0.1488249754985524   # +0.3%  n_dr_0p2_0p4 < 13 and dr_2 < 0.06545
        + 0.0028510034178058155 * max(0.0, Q.mass_top50 - 160.8) / 1.2464608384684717   # +0.3%  mass_top50 > 160.8
        - 0.0027165707890683694 * max(0.0, Q.sum_pt_top15 - 935.104296875) * max(0.0, 0.063364507347 - Q.dr_4) / 1.0308866503647645   # -0.3%  sum_pt_top15 > 935.1 and dr_4 < 0.06336
        + 0.0026900668624234153 * max(0.0, 0.428145796061 - Q.tau21) / 0.08904985831436439   # +0.3%  tau21 < 0.4281
        - 0.002669147618582166 * max(0.0, Q.lam1 - 0.001868040786) * max(0.0, Q.pt_dispersion - 0.274620002508) / 0.00020142990841830167   # -0.3%  lam1 > 0.001868 and pt_dispersion > 0.2746
        - 0.0016353904992029247 * max(0.0, Q.mass_top50 - 172.8) * max(0.0, 1260.540869140625 - Q.sum_pt) / 55.58852368215233   # -0.2%  mass_top50 > 172.8 and sum_pt < 1261
        + 0.0015962074183445436 * max(0.0, 0.005402330897 - Q.girth2_top30) * max(0.0, 631.275 - Q.sum_pt_top5) / 0.06578376550853171   # +0.2%  girth2_top30 < 0.005402 and sum_pt_top5 < 631.3
        - 0.001309812161188677 * max(0.0, 0.356311369374 - Q.planar_flow) / 0.06649230326828766   # -0.1%  planar_flow < 0.3563
        + 0.0012682597205080387 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, Q.D2 - 1.230420708656) / 0.978638625763552   # +0.1%  mass_top50 > 160.8 and D2 > 1.23
        - 0.001265220068349253 * max(0.0, 0.064132973195 - Q.dr_0) / 0.02518955799705094   # -0.1%  dr_0 < 0.06413
        - 0.0010748142421286454 * max(0.0, Q.mass_top50 - 172.8) * max(0.0, 33.376099042476 - Q.max_pair_mass) / 5.4577745330916   # -0.1%  mass_top50 > 172.8 and max_pair_mass < 33.38
        - 0.0008857496807938816 * max(0.0, Q.z_dr_0_0p05 - 0.767473447323) * max(0.0, Q.n_particles - 36.0) / 0.2867166657333033   # -0.1%  z_dr_0_0p05 > 0.7675 and n_particles > 36
        + 0.0007520457205179715 * max(0.0, Q.mass_top50 - 172.8) / 0.5062388854982302   # +0.1%  mass_top50 > 172.8
        + 0.0004963122032003485 * max(0.0, Q.mass_top50 - 160.8) * max(0.0, 41.4375 - Q.pt_9) / 13.004493341546693   # +0.0%  mass_top50 > 160.8 and pt_9 < 41.44
        - 0.00045559260460508475 * max(0.0, Q.lam1 - 0.004673423215) / 0.004174907990959959   # -0.0%  lam1 > 0.004673
        + 0.00018415754684611622 * max(0.0, 959.095727539062 - Q.sum_pt_top50) * max(0.0, Q.mass_top10 - 23.388939226434) / 224.53531236607003   # +0.0%  sum_pt_top50 < 959.1 and mass_top10 > 23.39
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 6.984;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.983653493449865 * (-0.015200799508666584
        + 0.16212762035919437 * max(0.0, 92.85979309082 - Q.mass) / 17.6013052131255   # +16.2%  mass < 92.86
        - 0.08092529543359 * max(0.0, 0.006363915755 - Q.girth2_top30) / 0.001678623919074012   # -8.1%  girth2_top30 < 0.006364
        - 0.07023278730032656 * max(0.0, 86.4 - Q.mass_top50) / 14.253133905303311   # -7.0%  mass_top50 < 86.4
        + 0.05576166142489567 * max(0.0, 0.008184367501 - Q.mass_over_sum_pt_sq) / 0.0024516083597474207   # +5.6%  mass_over_sum_pt_sq < 0.008184
        - 0.048015175400520395 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -4.8%  mass < 80.4
        + 0.046421166088678134 * max(0.0, 101.049709320068 - Q.mass) / 23.619330874804486   # +4.6%  mass < 101
        + 0.040703126032351475 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 43.23581344537815   # +4.1%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 21
        + 0.03836720504394535 * max(0.0, 0.027935993578 - Q.e2) / 0.005715276962294592   # +3.8%  e2 < 0.02794
        - 0.03429311807698588 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) / 3.813583193277311   # -3.4%  n_dr_0p2_0p4 < 10
        - 0.031684448645767534 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.334047731757 - Q.z_dr_0p1_0p2) / 6.53582113499238   # -3.2%  mass < 101 and z_dr_0p1_0p2 < 0.334
        - 0.029214954372090273 * max(0.0, 0.006716736591 - Q.lam1) / 0.0019121054678470912   # -2.9%  lam1 < 0.006717
        + 0.027296764463552513 * max(0.0, 0.005809484705 - Q.girth2_top30) / 0.0014021356702027855   # +2.7%  girth2_top30 < 0.005809
        + 0.026847925829216968 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.068491501734 - Q.z_dr_0p2_0p4) / 0.23346825157917758   # +2.7%  n_dr_0p2_0p4 < 10 and z_dr_0p2_0p4 < 0.06849
        - 0.02647303806030483 * max(0.0, 0.006170281901 - Q.width) / 0.0012963875716651668   # -2.6%  width < 0.00617
        - 0.02525454467227041 * max(0.0, 80.4 - Q.mass) * max(0.0, 0.037001823448 - Q.z_dr_0p2_0p4) / 0.30159836215099584   # -2.5%  mass < 80.4 and z_dr_0p2_0p4 < 0.037
        + 0.02341058168346701 * max(0.0, 0.006381743611 - Q.z_dr_0p2_0p4) / 0.0014538258888193599   # +2.3%  z_dr_0p2_0p4 < 0.006382
        - 0.023197496295578414 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.005532208341 - Q.girth2) / 0.005195774395551462   # -2.3%  n_dr_0p2_0p4 < 10 and girth2 < 0.005532
        + 0.023104773405617382 * max(0.0, 86.4 - Q.mass_top50) * max(0.0, 0.00416995399 - Q.girth2_top15) / 0.04492769249291807   # +2.3%  mass_top50 < 86.4 and girth2_top15 < 0.00417
        - 0.020827354148946146 * max(0.0, Q.z_top5_slots - 0.534625950898) / 0.08390497481820398   # -2.1%  z_top5_slots > 0.5346
        + 0.018666265706307715 * max(0.0, 0.025159193203 - Q.e2) / 0.004550017304548827   # +1.9%  e2 < 0.02516
        + 0.01774508090103965 * max(0.0, 0.095732276142 - Q.z_dr_0_0p05) / 0.021281262007401606   # +1.8%  z_dr_0_0p05 < 0.09573
        - 0.010980279833978887 * max(0.0, 0.038759447634 - Q.e2) / 0.01184118794270979   # -1.1%  e2 < 0.03876
        - 0.010587122218668255 * max(0.0, 0.007164202106 - Q.girth2_top5) / 0.0036386067329905753   # -1.1%  girth2_top5 < 0.007164
        + 0.010472196994897312 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, 0.356311369374 - Q.planar_flow) / 0.8671032650005658   # +1.0%  mass < 101 and planar_flow < 0.3563
        - 0.009940819118870936 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p05_0p1 - 0.591232848167) / 0.2158544762195853   # -1.0%  n_dr_0p2_0p4 < 10 and z_dr_0p05_0p1 > 0.5912
        + 0.009331093639667342 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.tau21 - 0.129504834861) / 1.0594061859795467   # +0.9%  n_dr_0p2_0p4 < 10 and tau21 > 0.1295
        - 0.009154642850185487 * max(0.0, 80.4 - Q.mass) * max(0.0, 464.75 - Q.sum_pt_top2) / 805.2282157106608   # -0.9%  mass < 80.4 and sum_pt_top2 < 464.8
        - 0.00834609254910024 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.345338582993 - Q.D2) / 3.422261309224861   # -0.8%  mass < 80.4 and D2 < 3.345
        + 0.008234002317121568 * max(0.0, 60.438206617337 - Q.mass_top30) / 6.992028296799824   # +0.8%  mass_top30 < 60.44
        + 0.00735016184638879 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) / 1.104090756302521   # +0.7%  n_dr_0p2_0p4 < 5
        + 0.00602859033406284 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, Q.mass_top10 - 44.192251085966) / 80.10308960257042   # +0.6%  mass < 101 and mass_top10 > 44.19
        + 0.005982725365879351 * max(0.0, 0.298200035095 - Q.max_dr) / 0.013500874805804353   # +0.6%  max_dr < 0.2982
        + 0.00586488428391836 * max(0.0, Q.z_top30_slots - 0.980193855091) / 0.006863737860514843   # +0.6%  z_top30_slots > 0.9802
        + 0.00510640014045318 * max(0.0, 0.00130722027 - Q.girth2_top10) / 0.00027941021155763345   # +0.5%  girth2_top10 < 0.001307
        - 0.004615787676541369 * max(0.0, 101.049709320068 - Q.mass) * max(0.0, Q.n_dr_0p2_0p4 - 10.0) / 10.717728124535183   # -0.5%  mass < 101 and n_dr_0p2_0p4 > 10
        + 0.004206192338315409 * max(0.0, 5.0 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_real_top30 - 22.0) / 7.061857142857143   # +0.4%  n_dr_0p2_0p4 < 5 and n_real_top30 > 22
        + 0.0038951764826624454 * max(0.0, 92.85979309082 - Q.mass) * max(0.0, 1.230420708656 - Q.D2) / 0.629877091073777   # +0.4%  mass < 92.86 and D2 < 1.23
        + 0.003296356743545812 * max(0.0, Q.z_dr_0p05_0p1 - 0.850921532512) / 0.0035564257207972684   # +0.3%  z_dr_0p05_0p1 > 0.8509
        - 0.0030809579915927194 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 9.0 - Q.n_dr_0p1_0p2) / 8.066541176470588   # -0.3%  n_dr_0p2_0p4 < 10 and n_dr_0p1_0p2 < 9
        + 0.002556118770760863 * max(0.0, 6.89371369877 - Q.log_sum_pt) / 0.013281099160517522   # +0.3%  log_sum_pt < 6.894
        + 0.0004000151587422551 * max(0.0, 0.095732276142 - Q.z_dr_0_0p05) * max(0.0, Q.max_dr - 0.193997652829) / 0.0034277142169341405   # +0.0%  z_dr_0_0p05 < 0.09573 and max_dr > 0.194
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 5.461;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.460998685518699 * (-0.008949680906300913
        + 0.1750187051671321 * max(0.0, 82.85408782959 - Q.mass) / 11.83453618826746   # +17.5%  mass < 82.85
        - 0.10373611432511691 * max(0.0, 77.376408295162 - Q.mass_top50) / 9.980854903106312   # -10.4%  mass_top50 < 77.38
        + 0.08639203922440758 * max(0.0, 80.4 - Q.mass_top50) / 11.159877357156788   # +8.6%  mass_top50 < 80.4
        + 0.08387053784548862 * max(0.0, 74.251806640625 - Q.mass) / 8.587064460866076   # +8.4%  mass < 74.25
        - 0.05380939233937532 * max(0.0, 0.050483809784 - Q.girth) / 0.008533024989277627   # -5.4%  girth < 0.05048
        - 0.0497923363243567 * max(0.0, 62.55 - Q.mass) / 5.464058366300095   # -5.0%  mass < 62.55
        - 0.042175432310760624 * max(0.0, Q.C2 - 0.061168736406) / 0.01742346976710858   # -4.2%  C2 > 0.06117
        - 0.0415008793620371 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # -4.2%  mass < 86.4
        + 0.03794488009085817 * max(0.0, 77.376408295162 - Q.mass_top50) * max(0.0, 0.850921532512 - Q.z_dr_0p05_0p1) / 7.991806220965258   # +3.8%  mass_top50 < 77.38 and z_dr_0p05_0p1 < 0.8509
        + 0.03539888043465528 * max(0.0, 0.00742997247 - Q.girth2_top50) / 0.0020221555597961793   # +3.5%  girth2_top50 < 0.00743
        - 0.029704692153222648 * max(0.0, 86.4 - Q.mass) * max(0.0, 0.064725840837 - Q.z_dr_0p1_0p2) / 0.5993597398116846   # -3.0%  mass < 86.4 and z_dr_0p1_0p2 < 0.06473
        - 0.028131614475717635 * max(0.0, Q.planar_flow - 0.258818254187) / 0.2699773591449117   # -2.8%  planar_flow > 0.2588
        + 0.024726889328260983 * max(0.0, 89.67879517394 - Q.mass_top40) / 17.478989415808297   # +2.5%  mass_top40 < 89.68
        - 0.02368503454221688 * max(0.0, 0.050483809784 - Q.girth) * max(0.0, 64.0 - Q.n_particles) / 0.24223643538921044   # -2.4%  girth < 0.05048 and n_particles < 64
        - 0.021912729365490886 * max(0.0, 77.936678808178 - Q.mass_top40) / 11.120277019467732   # -2.2%  mass_top40 < 77.94
        - 0.016618903979726922 * max(0.0, 89.67879517394 - Q.mass_top40) * max(0.0, 501.625 - Q.sum_pt_top2) / 2080.229882440589   # -1.7%  mass_top40 < 89.68 and sum_pt_top2 < 501.6
        + 0.016284949361465056 * max(0.0, 0.001000990214 - Q.girth2_top5) / 0.00025088670724764367   # +1.6%  girth2_top5 < 0.001001
        - 0.015008648490765313 * max(0.0, Q.mass_top20 - 125.1) * max(0.0, Q.C2 - 0.056027559564) / 0.06998404251378426   # -1.5%  mass_top20 > 125.1 and C2 > 0.05603
        + 0.014055190538784759 * max(0.0, 0.228402115913 - Q.LHA) / 0.027166566418098437   # +1.4%  LHA < 0.2284
        + 0.012964535863998858 * max(0.0, Q.LHA - 0.404204003833) / 0.0031577141977621142   # +1.3%  LHA > 0.4042
        - 0.01146247781035028 * max(0.0, Q.z_dr_0_0p05 - 0.908491230011) / 0.00922734725402288   # -1.1%  z_dr_0_0p05 > 0.9085
        - 0.01112839616904917 * max(0.0, 89.67879517394 - Q.mass_top40) * max(0.0, Q.n_particles - 22.0) / 314.97281238932567   # -1.1%  mass_top40 < 89.68 and n_particles > 22
        - 0.010168269380632073 * max(0.0, 6.856374501323 - Q.log_sum_pt) / 0.008053458833961623   # -1.0%  log_sum_pt < 6.856
        + 0.007824299159237443 * max(0.0, Q.mass_top20 - 125.1) / 1.3144709680417963   # +0.8%  mass_top20 > 125.1
        + 0.007489639548088607 * max(0.0, Q.mass_top30 - 138.381845700848) / 1.7582156343658408   # +0.7%  mass_top30 > 138.4
        + 0.007239622790656803 * max(0.0, 3.814159452915 - Q.D2) / 1.5199649835557993   # +0.7%  D2 < 3.814
        - 0.006003551304758823 * max(0.0, 0.000743190527 - Q.girth2_top10) * max(0.0, Q.n_particles - 51.0) / 0.00020012861546058106   # -0.6%  girth2_top10 < 0.0007432 and n_particles > 51
        + 0.004518130539685829 * max(0.0, 86.4 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.299250295758) / 0.29970876190808693   # +0.5%  mass < 86.4 and z_dr_0p05_0p1 > 0.2993
        - 0.003961665646608762 * max(0.0, Q.z_dr_0p1_0p2 - 0.688217741251) * max(0.0, Q.min_pair_mass - 0.740073079621) / 0.012527968941895694   # -0.4%  z_dr_0p1_0p2 > 0.6882 and min_pair_mass > 0.7401
        + 0.0037066534561054957 * max(0.0, Q.LHA - 0.404204003833) * max(0.0, 0.003687604901 - Q.lam2) / 1.6322723713619188e-06   # +0.4%  LHA > 0.4042 and lam2 < 0.003688
        + 0.003426542852664141 * max(0.0, Q.mass_top20 - 125.1) * max(0.0, Q.pt_2 - 73.6875) / 25.867315218426633   # +0.3%  mass_top20 > 125.1 and pt_2 > 73.69
        + 0.0031645184064614286 * max(0.0, Q.mass_top30 - 138.381845700848) * max(0.0, 0.186661871599 - Q.dr_3) / 0.08618385585768046   # +0.3%  mass_top30 > 138.4 and dr_3 < 0.1867
        - 0.0023015502983218363 * max(0.0, Q.mass_top20 - 125.1) * max(0.0, Q.pt_2 - 111.75) / 5.603265015950728   # -0.2%  mass_top20 > 125.1 and pt_2 > 111.8
        - 0.002042677856855953 * max(0.0, Q.girth2_top5 - 0.016859196762) / 0.0008909510557263302   # -0.2%  girth2_top5 > 0.01686
        + 0.0012095949634610575 * max(0.0, 6.856374501323 - Q.log_sum_pt) * max(0.0, Q.n_particles - 62.0) / 0.005383261230807502   # +0.1%  log_sum_pt < 6.856 and n_particles > 62
        + 0.0008337947038829246 * max(0.0, Q.mass_top20 - 125.1) * max(0.0, Q.pt_2 - 137.5) / 1.9734056133901783   # +0.1%  mass_top20 > 125.1 and pt_2 > 137.5
        - 0.0007862295893409361 * max(0.0, 0.005788041138 - Q.girth2_top15) / 0.0018776716250801934   # -0.1%  girth2_top15 < 0.005788
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 6.965;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.964848943888503 * (0.259521342189894
        - 0.09378343232940979 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # -9.4%  log_sum_pt < 7.017
        + 0.09332782366345725 * max(0.0, Q.mass - 74.251806640625) / 24.076479363593222   # +9.3%  mass > 74.25
        - 0.07624101353538162 * max(0.0, Q.mass - 143.787612915039) / 3.946978784615654   # -7.6%  mass > 143.8
        + 0.07160136569315376 * Q.n_particles / 45.8152268907563   # +7.2%  n_particles
        + 0.06071860629323951 * max(0.0, 0.009606007381 - Q.e2_sq) / 0.003497865944155776   # +6.1%  e2_sq < 0.009606
        - 0.05853174751201695 * max(0.0, Q.mass_top50 - 97.930041729355) / 11.917644068610143   # -5.9%  mass_top50 > 97.93
        + 0.058526745017852835 * max(0.0, Q.mass_top50 - 136.785) / 4.25098606820319   # +5.9%  mass_top50 > 136.8
        + 0.056852009710658395 * max(0.0, 1053.04736328125 - Q.sum_pt_top40) / 57.73987569631368   # +5.7%  sum_pt_top40 < 1053
        - 0.05677581056228534 * max(0.0, 1012.672900390625 - Q.sum_pt) / 19.544077209776376   # -5.7%  sum_pt < 1013
        - 0.05588479126804138 * max(0.0, 1052.889428710937 - Q.sum_pt) / 42.62464270793601   # -5.6%  sum_pt < 1053
        + 0.05121069156833914 * max(0.0, 1013.915698242188 - Q.sum_pt_top50) / 24.159122871194317   # +5.1%  sum_pt_top50 < 1014
        + 0.023217759840478212 * max(0.0, Q.mass_over_sum_pt - 0.170880120467) / 0.0005536193928784717   # +2.3%  mass_over_sum_pt > 0.1709
        - 0.022955272968414785 * max(0.0, Q.mass - 136.785) / 5.043396460386572   # -2.3%  mass > 136.8
        - 0.02278135291313071 * max(0.0, Q.mass_over_sum_pt_sq - 0.029200015571) / 0.00020282738822808687   # -2.3%  mass_over_sum_pt_sq > 0.0292
        + 0.018363010908314707 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +1.8%  mass > 172.8
        - 0.01661285316695428 * max(0.0, 1053.04736328125 - Q.sum_pt_top40) * max(0.0, 5.378974604607 - Q.D2) / 152.36077532094276   # -1.7%  sum_pt_top40 < 1053 and D2 < 5.379
        + 0.015363692356754242 * max(0.0, Q.mass_top10 - 56.921923720802) / 8.071231186963185   # +1.5%  mass_top10 > 56.92
        - 0.014354778001921925 * max(0.0, 4.0 - Q.n_dr_0p2_0p4) / 0.7297563025210084   # -1.4%  n_dr_0p2_0p4 < 4
        + 0.013302542060550307 * max(0.0, 6.811175180312 - Q.log_sum_pt) / 0.004882419406465069   # +1.3%  log_sum_pt < 6.811
        + 0.011280253934360484 * max(0.0, 1085.12490234375 - Q.sum_pt) * max(0.0, Q.tau21 - 0.185845967382) / 18.995761837088555   # +1.1%  sum_pt < 1085 and tau21 > 0.1858
        - 0.011258946261648691 * max(0.0, Q.z_top10_slots - 0.829873578817) / 0.021772983628397042   # -1.1%  z_top10_slots > 0.8299
        - 0.010764581555357757 * max(0.0, 1013.915698242188 - Q.sum_pt_top50) * max(0.0, 4.450168704987 - Q.D2) / 41.241641900364776   # -1.1%  sum_pt_top50 < 1014 and D2 < 4.45
        - 0.009077921443866089 * max(0.0, 986.05654296875 - Q.sum_pt) / 11.984374324425932   # -0.9%  sum_pt < 986.1
        - 0.008854316752827596 * max(0.0, 1085.12490234375 - Q.sum_pt) * max(0.0, 89.6875 - Q.pt_3) / 1742.3268874681942   # -0.9%  sum_pt < 1085 and pt_3 < 89.69
        + 0.0077049132919232224 * max(0.0, 959.095727539062 - Q.sum_pt_top50) * max(0.0, 4.450168704987 - Q.D2) / 15.20393043139574   # +0.8%  sum_pt_top50 < 959.1 and D2 < 4.45
        + 0.006095786464731102 * max(0.0, Q.mass_over_sum_pt - 0.078528833221) / 0.0193369167748192   # +0.6%  mass_over_sum_pt > 0.07853
        - 0.005587553103502715 * max(0.0, 956.21328125 - Q.sum_pt_top40) / 14.004510341849162   # -0.6%  sum_pt_top40 < 956.2
        + 0.0052969999025563605 * max(0.0, Q.z_top10_slots - 0.829873578817) * max(0.0, 3.814159452915 - Q.D2) / 0.019687885856715187   # +0.5%  z_top10_slots > 0.8299 and D2 < 3.814
        - 0.005090997810129589 * max(0.0, Q.mass - 160.8) / 1.7246633179459985   # -0.5%  mass > 160.8
        - 0.0047161373873584374 * max(0.0, Q.sum_pt_top20 - 956.50615234375) / 37.465638557006216   # -0.5%  sum_pt_top20 > 956.5
        - 0.0047126716221429065 * max(0.0, Q.mass_top50 - 168.969765712694) / 0.6651774989835844   # -0.5%  mass_top50 > 169
        - 0.0039528665553308345 * max(0.0, 1012.672900390625 - Q.sum_pt) * max(0.0, 0.051086217058 - Q.dr_5) / 0.2511134240217651   # -0.4%  sum_pt < 1013 and dr_5 < 0.05109
        - 0.0037342696131327366 * max(0.0, Q.C2 - 0.072798889503) / 0.01280801443367811   # -0.4%  C2 > 0.0728
        - 0.003655489380841703 * max(0.0, Q.mass - 160.8) * max(0.0, 5.378974604607 - Q.D2) / 5.667716793316475   # -0.4%  mass > 160.8 and D2 < 5.379
        + 0.003507834701722294 * max(0.0, 0.073744720221 - Q.girth) / 0.0191559325282986   # +0.4%  girth < 0.07374
        - 0.0027173592473692097 * max(0.0, 6.811175180312 - Q.log_sum_pt) * max(0.0, 0.051086217058 - Q.dr_5) / 7.371257739198046e-05   # -0.3%  log_sum_pt < 6.811 and dr_5 < 0.05109
        - 0.0026070825700495147 * max(0.0, Q.mass - 74.251806640625) * max(0.0, 1017.43466796875 - Q.sum_pt) / 595.5869900522073   # -0.3%  mass > 74.25 and sum_pt < 1017
        + 0.001972445576121245 * max(0.0, Q.mass - 143.787612915039) * max(0.0, 1007.788464355469 - Q.sum_pt) / 60.170216639904   # +0.2%  mass > 143.8 and sum_pt < 1008
        - 0.0019610708994150477 * max(0.0, Q.sum_pt - 1260.540869140625) / 7.65548318400563   # -0.2%  sum_pt > 1261
        - 0.0018698295763167297 * max(0.0, Q.sum_pt - 1260.540869140625) * max(0.0, 0.082575402011 - Q.dr_7) / 0.33734421106481194   # -0.2%  sum_pt > 1261 and dr_7 < 0.08258
        + 0.001734567749248898 * max(0.0, Q.sum_pt_top30 - 1111.24501953125) / 12.625727406952   # +0.2%  sum_pt_top30 > 1111
        + 0.000786729637429493 * max(0.0, 959.095727539062 - Q.sum_pt_top50) / 9.937940549204788   # +0.1%  sum_pt_top50 < 959.1
        + 0.0006540755922622661 * max(0.0, Q.mass - 74.251806640625) * max(0.0, Q.D2 - 0.603279101849) / 37.923504307700206   # +0.1%  mass > 74.25 and D2 > 0.6033
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 20.39;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.391058608774575 * (-0.0009516881758833224
        - 0.14779054017666704 * max(0.0, 0.090467494167 - Q.mass_over_sum_pt) / 0.0178873033081874   # -14.8%  mass_over_sum_pt < 0.09047
        + 0.10741666162225547 * max(0.0, 0.09795414517 - Q.mass_over_sum_pt) / 0.023392533955001828   # +10.7%  mass_over_sum_pt < 0.09795
        + 0.09257705799402964 * max(0.0, 0.118225939153 - Q.mass_over_sum_pt) / 0.039171218898223295   # +9.3%  mass_over_sum_pt < 0.1182
        - 0.06544414627414985 * max(0.0, 89.741833496094 - Q.mass) / 15.556328292327747   # -6.5%  mass < 89.74
        - 0.051562108594360376 * max(0.0, 91.034691238403 - Q.mass) / 16.362866628552233   # -5.2%  mass < 91.03
        - 0.043751951101161195 * max(0.0, 0.009262053166 - Q.girth2_top50) / 0.0033075364251965024   # -4.4%  girth2_top50 < 0.009262
        - 0.041005858361422294 * max(0.0, 0.012157872869 - Q.girth2_top30) / 0.005935411821644211   # -4.1%  girth2_top30 < 0.01216
        + 0.039728260968837505 * max(0.0, 0.140939019879 - Q.mass_over_sum_pt) / 0.057960363993862694   # +4.0%  mass_over_sum_pt < 0.1409
        + 0.035855848999407 * max(0.0, 117.048742792994 - Q.mass_top50) / 36.94196179171718   # +3.6%  mass_top50 < 117
        + 0.029536503471290186 * max(0.0, 136.785 - Q.mass) / 52.08717491697615   # +3.0%  mass < 136.8
        - 0.028230383717154438 * max(0.0, 0.011744050682 - Q.lam1) / 0.005678958164694959   # -2.8%  lam1 < 0.01174
        - 0.02665366991940039 * max(0.0, 80.890431271924 - Q.mass_top40) / 12.434053057794845   # -2.7%  mass_top40 < 80.89
        + 0.02104683358752382 * max(0.0, 0.002396991421 - Q.lam2) / 0.0014662533145241962   # +2.1%  lam2 < 0.002397
        - 0.01995478816976507 * max(0.0, 111.24867219155 - Q.mass_top40) / 33.99435635629078   # -2.0%  mass_top40 < 111.2
        - 0.01838457253223406 * max(0.0, 0.016559833876 - Q.girth2_top20) * max(0.0, Q.z_top50_slots - 0.970443639316) / 0.000256897965306318   # -1.8%  girth2_top20 < 0.01656 and z_top50_slots > 0.9704
        - 0.017415010663001256 * max(0.0, Q.sum_pt_top50 - 976.277001953125) / 72.2927679781981   # -1.7%  sum_pt_top50 > 976.3
        + 0.015941949823361626 * max(0.0, 0.006189818106 - Q.lam1) / 0.0016087780966108156   # +1.6%  lam1 < 0.00619
        + 0.01414886243523045 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) / 12.59063025210084   # +1.4%  n_dr_0p2_0p4 < 21
        + 0.014128039357839689 * max(0.0, 0.007877041167 - Q.girth2) / 0.0022422969111317915   # +1.4%  girth2 < 0.007877
        + 0.01392853891525137 * Q.sum_pt_top5 / 593.4493714080554   # +1.4%  sum_pt_top5
        + 0.01385012743523449 * max(0.0, 83.325535102591 - Q.mass_top40) / 13.716491940764705   # +1.4%  mass_top40 < 83.33
        - 0.013087762082927357 * max(0.0, 0.085894044489 - Q.girth) / 0.027459176528023727   # -1.3%  girth < 0.08589
        - 0.012610699250093558 * max(0.0, Q.max_dr - 0.240474711359) / 0.11773588635266226   # -1.3%  max_dr > 0.2405
        + 0.012525031595815084 * max(0.0, Q.log_sum_pt - 6.935549248787) / 0.03715380406194132   # +1.3%  log_sum_pt > 6.936
        - 0.011482942789926448 * max(0.0, 0.00634934989 - Q.girth2_top50) / 0.0014253049682818088   # -1.1%  girth2_top50 < 0.006349
        + 0.011149070777363015 * max(0.0, 91.19 - Q.mass_top30) / 21.633273586226487   # +1.1%  mass_top30 < 91.19
        + 0.010381366688959493 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +1.0%  mass < 80.4
        - 0.00980058101574176 * max(0.0, 0.016559833876 - Q.girth2_top20) * max(0.0, 0.6418633461 - Q.tau21) / 0.0019557440259633627   # -1.0%  girth2_top20 < 0.01656 and tau21 < 0.6419
        + 0.0079826154770955 * max(0.0, 0.006189818106 - Q.lam1) * max(0.0, Q.z_top50_slots - 0.985099030959) / 2.0496011611642913e-05   # +0.8%  lam1 < 0.00619 and z_top50_slots > 0.9851
        + 0.007939902491733895 * max(0.0, 0.007259287357 - Q.lam1) / 0.0022497701299997253   # +0.8%  lam1 < 0.007259
        + 0.007025020527457542 * max(0.0, 0.00608841615 - Q.girth2_top30) / 0.0015338919490682538   # +0.7%  girth2_top30 < 0.006088
        + 0.005086191002337647 * max(0.0, 0.006876086349 - Q.girth2_top10) / 0.0028923874793807566   # +0.5%  girth2_top10 < 0.006876
        + 0.005026158053959968 * max(0.0, 0.03263075389 - Q.e2) / 0.008034084923623714   # +0.5%  e2 < 0.03263
        - 0.004596859815351945 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 21.0 - Q.n_dr_0p1_0p2) / 137.24421512605042   # -0.5%  n_dr_0p2_0p4 < 21 and n_dr_0p1_0p2 < 21
        + 0.0037995312519140236 * max(0.0, Q.max_dr - 0.240474711359) * max(0.0, 56.921923720802 - Q.mass_top10) / 2.2920324246063815   # +0.4%  max_dr > 0.2405 and mass_top10 < 56.92
        + 0.0033808983854148955 * max(0.0, 0.001163277284 - Q.lam2) / 0.0004869715979290885   # +0.3%  lam2 < 0.001163
        + 0.0026617138305775395 * max(0.0, 0.958653609576 - Q.z_top50_slots) / 0.000623249323319539   # +0.3%  z_top50_slots < 0.9587
        - 0.0024515681050044477 * max(0.0, 0.016559833876 - Q.girth2_top20) / 0.010034861112355914   # -0.2%  girth2_top20 < 0.01656
        + 0.0024104097536980077 * max(0.0, 91.034691238403 - Q.mass) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 2.9734071289695474   # +0.2%  mass < 91.03 and z_dr_0p1_0p2 < 0.2188
        - 0.0022434524832189943 * max(0.0, 91.288535717504 - Q.mass_top40) * max(0.0, 0.068491501734 - Q.z_dr_0p2_0p4) / 1.0384437196530725   # -0.2%  mass_top40 < 91.29 and z_dr_0p2_0p4 < 0.06849
        - 0.002075829070233135 * max(0.0, 91.288535717504 - Q.mass_top40) / 18.56549606317863   # -0.2%  mass_top40 < 91.29
        - 0.0020379176031057843 * max(0.0, 21.0 - Q.n_dr_0p2_0p4) * max(0.0, 40.0 - Q.n_real_top40) / 63.56472605042017   # -0.2%  n_dr_0p2_0p4 < 21 and n_real_top40 < 40
        - 0.0010838499063939246 * max(0.0, 0.007877041167 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.009380839030992957   # -0.1%  girth2 < 0.007877 and n_dr_0p05_0p1 > 5
        - 0.0006146760424821643 * max(0.0, 117.048742792994 - Q.mass_top50) * max(0.0, Q.mass_top2 - 28.78966323496) / 13.565763507500906   # -0.1%  mass_top50 < 117 and mass_top2 > 28.79
        - 0.0001942378796165498 * max(0.0, Q.max_dr - 0.402178311348) / 0.010008869835123797   # -0.0%  max_dr > 0.4022
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 13.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.846380946360714 * (-0.04770301918245873
        + 0.13775888810935313 * max(0.0, 0.013514311784 - Q.girth2_top50) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.0007407123072106741   # +13.8%  girth2_top50 < 0.01351 and z_dr_0p2_0p4 < 0.1292
        - 0.0991828333834579 * max(0.0, 0.013514311784 - Q.girth2_top50) / 0.006620974830265835   # -9.9%  girth2_top50 < 0.01351
        + 0.09789574005975119 * max(0.0, 0.006936724595 - Q.e2_sq) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.00019974727848153987   # +9.8%  e2_sq < 0.006937 and z_dr_0p2_0p4 < 0.1292
        + 0.09629895865644761 * max(0.0, 0.120745175332 - Q.girth) / 0.05578613653830943   # +9.6%  girth < 0.1207
        - 0.07652472908909765 * max(0.0, 0.006936724595 - Q.e2_sq) / 0.0016886136779624735   # -7.7%  e2_sq < 0.006937
        - 0.0628659942560346 * max(0.0, 0.120745175332 - Q.girth) * max(0.0, 0.129185591638 - Q.z_dr_0p2_0p4) / 0.005981775287783369   # -6.3%  girth < 0.1207 and z_dr_0p2_0p4 < 0.1292
        - 0.05521633914188912 * max(0.0, 0.016493544356 - Q.lam1) / 0.009604222881411155   # -5.5%  lam1 < 0.01649
        + 0.03612754328569703 * max(0.0, 0.001501708498 - Q.girth2_top5) * max(0.0, 0.193744690716 - Q.z_dr_0p2_0p4) / 7.533004185945094e-05   # +3.6%  girth2_top5 < 0.001502 and z_dr_0p2_0p4 < 0.1937
        + 0.032769725303224764 * max(0.0, 7.017257672702 - Q.log_sum_pt) / 0.08901072718595586   # +3.3%  log_sum_pt < 7.017
        - 0.030878287813772096 * max(0.0, 0.001501708498 - Q.girth2_top5) / 0.0004379639090356587   # -3.1%  girth2_top5 < 0.001502
        - 0.029944561996009512 * max(0.0, 0.019523000158 - Q.z_dr_0p2_0p4) / 0.0076644008476949655   # -3.0%  z_dr_0p2_0p4 < 0.01952
        + 0.02856201365080468 * max(0.0, 0.007678543663 - Q.girth2_top10) / 0.0034735798352083574   # +2.9%  girth2_top10 < 0.007679
        - 0.017905995873167067 * max(0.0, 71.795159472175 - Q.mass_top50) / 8.207808294036079   # -1.8%  mass_top50 < 71.8
        + 0.017182331470758226 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 26.0 - Q.n_dr_0p2_0p4) / 0.8430573646645746   # +1.7%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p2_0p4 < 26
        - 0.017069419058483075 * max(0.0, 0.050483809784 - Q.girth) / 0.008533024989277627   # -1.7%  girth < 0.05048
        - 0.015568979366859224 * max(0.0, 0.006936724595 - Q.e2_sq) * max(0.0, 0.218764226139 - Q.z_dr_0p1_0p2) / 0.0003126700564572212   # -1.6%  e2_sq < 0.006937 and z_dr_0p1_0p2 < 0.2188
        + 0.01406859721279739 * max(0.0, 0.187280465662 - Q.z_dr_0p1_0p2) / 0.08737974006182235   # +1.4%  z_dr_0p1_0p2 < 0.1873
        + 0.01351798014091775 * max(0.0, 0.007164202106 - Q.girth2_top5) / 0.0036386067329905753   # +1.4%  girth2_top5 < 0.007164
        + 0.013450826781813923 * max(0.0, 0.001776308492 - Q.lam2) / 0.0009519164072663888   # +1.3%  lam2 < 0.001776
        - 0.01132376959837664 * max(0.0, 0.052014814497 - Q.dr_0) / 0.017315448851188676   # -1.1%  dr_0 < 0.05201
        - 0.010901877079827321 * max(0.0, 1002.378515625 - Q.sum_pt) / 15.957354754845854   # -1.1%  sum_pt < 1002
        + 0.009207481433799139 * max(0.0, 80.4 - Q.mass_top30) / 14.655773101172624   # +0.9%  mass_top30 < 80.4
        - 0.008373851953769995 * max(0.0, 0.05268713294 - Q.dr_3) / 0.016149627237586155   # -0.8%  dr_3 < 0.05269
        - 0.0063464851915337244 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 787.628125 - Q.sum_pt_top3) / 12.480147072204154   # -0.6%  z_dr_0p1_0p2 < 0.1203 and sum_pt_top3 < 787.6
        - 0.00632928635775059 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 0.037001823448 - Q.z_dr_0p2_0p4) / 0.0009116378981874741   # -0.6%  z_dr_0p1_0p2 < 0.1203 and z_dr_0p2_0p4 < 0.037
        - 0.006037270229000237 * max(0.0, 0.007678543663 - Q.girth2_top10) * max(0.0, 0.986057513941 - Q.z_top30_slots) / 8.865302557443164e-05   # -0.6%  girth2_top10 < 0.007679 and z_top30_slots < 0.9861
        + 0.0058968719310280564 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 40.2 - Q.mass_top5) / 1.3263398907230615   # +0.6%  z_dr_0p1_0p2 < 0.1203 and mass_top5 < 40.2
        - 0.005865480915842993 * max(0.0, 0.007164202106 - Q.girth2_top5) * max(0.0, 0.331585738063 - Q.max_dr) / 5.819031732268512e-05   # -0.6%  girth2_top5 < 0.007164 and max_dr < 0.3316
        + 0.0058415499426043805 * max(0.0, 7.017257672702 - Q.log_sum_pt) * max(0.0, 0.019523000158 - Q.z_dr_0p2_0p4) / 0.0005962116133511941   # +0.6%  log_sum_pt < 7.017 and z_dr_0p2_0p4 < 0.01952
        + 0.005569213278049292 * max(0.0, 0.007164202106 - Q.girth2_top5) * max(0.0, 984.70087890625 - Q.sum_pt_top40) / 0.05379473153393668   # +0.6%  girth2_top5 < 0.007164 and sum_pt_top40 < 984.7
        - 0.004282838546687507 * max(0.0, 0.001501708498 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.0032906615041510393   # -0.4%  girth2_top5 < 0.001502 and n_dr_0p2_0p4 > 0
        - 0.003569457623131178 * max(0.0, 0.087968891487 - Q.C2) / 0.030281422001049746   # -0.4%  C2 < 0.08797
        - 0.003148914033857514 * max(0.0, 0.00321372409 - Q.girth2_top10) / 0.0009423837681382483   # -0.3%  girth2_top10 < 0.003214
        + 0.003062452201075401 * max(0.0, 0.845900350809 - Q.z_dr_0_0p05) * max(0.0, Q.log_sum_pt - 6.910130970417) / 0.01458841971217905   # +0.3%  z_dr_0_0p05 < 0.8459 and log_sum_pt > 6.91
        + 0.0026767934203920612 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, 0.826191085385 - Q.planar_flow) / 0.012150672386848877   # +0.3%  z_dr_0p1_0p2 < 0.1203 and planar_flow < 0.8262
        - 0.002237071960389114 * max(0.0, 1002.378515625 - Q.sum_pt) * max(0.0, 0.661614120007 - Q.tau32) / 0.9182209027219576   # -0.2%  sum_pt < 1002 and tau32 < 0.6616
        - 0.001501495011125585 * max(0.0, Q.z_dr_0p05_0p1 - 0.850921532512) / 0.0035564257207972684   # -0.2%  z_dr_0p05_0p1 > 0.8509
        - 0.0013714755212638824 * max(0.0, 0.120343671367 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p4_up - 0.0) / 0.006297153133219203   # -0.1%  z_dr_0p1_0p2 < 0.1203 and n_dr_0p4_up > 0
        + 0.0012225733063584083 * max(0.0, 934.241552734375 - Q.sum_pt_top50) / 6.932863593743613   # +0.1%  sum_pt_top50 < 934.2
        - 0.000970765251795999 * max(0.0, 1002.378515625 - Q.sum_pt) * max(0.0, 0.019523000158 - Q.z_dr_0p2_0p4) / 0.057971995942360716   # -0.1%  sum_pt < 1002 and z_dr_0p2_0p4 < 0.01952
        - 0.0009449736697022002 * max(0.0, 0.845900350809 - Q.z_dr_0_0p05) * max(0.0, Q.sum_pt_top40 - 1095.686413574219) / 3.6164311505857505   # -0.1%  z_dr_0_0p05 < 0.8459 and sum_pt_top40 > 1096
        - 0.0005283068923028041 * max(0.0, 0.050483809784 - Q.girth) * max(0.0, Q.z_dr_0p2_0p4 - 0.068491501734) / 7.516294794887081e-07   # -0.1%  girth < 0.05048 and z_dr_0p2_0p4 > 0.06849
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.529638550420168, 2.756085241596639, 0.38296029411764704, 0.5158217962184874, 0.9832950630252101, 1.0016363445378151, 0.8091965336134453, 0.5001355042016806, 2.0380681722689076, 0.7904831932773109, 1.2247125, 0.7209241596638656, 0.5037800420168067, 1.8783017331932772, 0.5485674369747899, 0.19111974789915967]
T = [4.017743226431198, 2.6354319393382353, 5.282191501444329, 5.1989421152836135, 4.4590325293789395]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +43%, n4 -21%, n9 +10%, n3 -10%, n5 -8%, n6 +3% ...
            + 0.42873652668141643 * h[1] / H_AVG[1]
            - 0.2141458853037015 * h[4] / H_AVG[4]
            + 0.10144821945121218 * h[9] / H_AVG[9]
            - 0.09628946534433054 * h[3] / H_AVG[3]
            - 0.07790725788768309 * h[5] / H_AVG[5]
            + 0.031469646329143286 * h[6] / H_AVG[6]
            + 0.029388002341943848 * h[12] / H_AVG[12]
            + 0.015852090786791354 * h[8] / H_AVG[8]
            - 0.00476290587377777 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -40%, n1 -20%, n9 +17%, n11 -7%, n6 +7%, n12 +7% ...
            - 0.39642496126332366 * h[4] / H_AVG[4]
            - 0.1960839796641196 * h[1] / H_AVG[1]
            + 0.16871875519962753 * h[9] / H_AVG[9]
            - 0.06838766625907362 * h[11] / H_AVG[11]
            + 0.06716612145650376 * h[6] / H_AVG[6]
            + 0.06571005946249628 * h[12] / H_AVG[12]
            + 0.018164019358711342 * h[2] / H_AVG[2]
            + 0.012083338111056666 * h[8] / H_AVG[8]
            - 0.007261099225087612 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -34%, n14 -14%, n5 +11%, n11 +8%, n0 +8%, n4 +6% ...
            - 0.3376079133533264 * h[8] / H_AVG[8]
            - 0.14279683453999928 * h[14] / H_AVG[14]
            + 0.10962703861220989 * h[5] / H_AVG[5]
            + 0.0810361986465991 * h[11] / H_AVG[11]
            + 0.0752015357085237 * h[0] / H_AVG[0]
            + 0.06399004614325196 * h[4] / H_AVG[4]
            - 0.05917708398882911 * h[7] / H_AVG[7]
            + 0.04272318331962821 * h[3] / H_AVG[3]
            - 0.03874540368583127 * h[12] / H_AVG[12]
            - 0.03273607147376806 * h[9] / H_AVG[9]
            + 0.009574583454047715 * h[6] / H_AVG[6]
            - 0.006784107073985099 * h[15] / H_AVG[15]
        ),
        0.984375 + T[3] * (   # class Z: n8 -37%, n6 -15%, n0 -14%, n5 +13%, n7 +9%, n3 +4% ...
            - 0.3675149422966539 * h[8] / H_AVG[8]
            - 0.15078243314799078 * h[6] / H_AVG[6]
            - 0.14007715236660281 * h[0] / H_AVG[0]
            + 0.1324548286170295 * h[5] / H_AVG[5]
            + 0.08718077459457296 * h[7] / H_AVG[7]
            + 0.04340729918538002 * h[3] / H_AVG[3]
            + 0.03028140333923758 * h[12] / H_AVG[12]
            - 0.027622948497914444 * h[2] / H_AVG[2]
            + 0.020678217954618005 * h[15] / H_AVG[15]
        ),
        0.78125 + T[4] * (   # class t: n13 -38%, n10 +27%, n5 -11%, n8 +10%, n12 -4%, n4 +4% ...
            - 0.3817444556618863 * h[13] / H_AVG[13]
            + 0.27036725102236797 * h[10] / H_AVG[10]
            - 0.10529571906206656 * h[5] / H_AVG[5]
            + 0.09641217063465794 * h[8] / H_AVG[8]
            - 0.042367377791391714 * h[12] / H_AVG[12]
            + 0.04134704627124708 * h[4] / H_AVG[4]
            - 0.03154565696256861 * h[7] / H_AVG[7]
            - 0.016072972105491048 * h[15] / H_AVG[15]
            + 0.014847350488322657 * h[0] / H_AVG[0]
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
