"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  5:   8.4%   (on for 59% of jets)
  neuron  4:   7.7%   (on for 82% of jets)
  neuron 10:   7.0%   (on for 71% of jets)
  neuron  7:   6.9%   (on for 51% of jets)
  neuron  3:   6.6%   (on for 25% of jets)
  neuron  2:   6.2%   (on for 83% of jets)
  neuron 11:   6.0%   (on for 76% of jets)
  neuron  1:   4.1%   (on for 66% of jets)
  neuron  0:   3.4%   (on for 43% of jets)
  neuron 14:   2.8%   (on for 27% of jets)
  neuron  8:   2.3%   (on for 45% of jets)
  neuron 15:   2.2%   (on for 23% of jets)
  neuron 12:   0.2%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

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
    # scale S = 31.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.74302836450608 * (-0.028544307082556418
        + 0.1241442252998866 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # +12.4%  width < 0.008678
        + 0.12172959840524906 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # +12.2%  girth2 < 0.01324
        - 0.09142220979787437 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # -9.1%  mass < 29.64
        + 0.0669323611450582 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.111955475493 - Q.dr_0) / 0.7131988181169212   # +6.7%  mass < 29.64 and dr_0 < 0.112
        - 0.05586740567306695 * max(0.0, 0.076081777364 - Q.girth) / 0.027967843803613616   # -5.6%  girth < 0.07608
        - 0.040603022891362126 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -4.1%  lam1 < 0.008376
        - 0.0405087028896983 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -4.1%  girth < 0.08724
        - 0.03792060549973018 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) / 0.07219748739654334   # -3.8%  mass_over_sum_pt < 0.1309
        - 0.029684869130838728 * max(0.0, 64.618731689453 - Q.mass) / 27.572788342921132   # -3.0%  mass < 64.62
        - 0.02898222499106582 * max(0.0, 0.004372139331 - Q.width) / 0.0015657102477202767   # -2.9%  width < 0.004372
        - 0.024715232815111704 * max(0.0, 56.920347213745 - Q.mass) / 21.784746347219   # -2.5%  mass < 56.92
        + 0.02397726521300186 * max(0.0, 0.018827652745 - Q.girth2) / 0.013142955535859579   # +2.4%  girth2 < 0.01883
        + 0.02350079999295066 * max(0.0, 0.020459658932 - Q.e2) / 0.00553192518950076   # +2.4%  e2 < 0.02046
        + 0.021538753424493504 * max(0.0, 0.003952581551 - Q.girth2_top3) / 0.001773144907435322   # +2.2%  girth2_top3 < 0.003953
        - 0.02089463951065348 * max(0.0, 0.293190627853 - Q.LHA) / 0.07167605978374647   # -2.1%  LHA < 0.2932
        + 0.018213241823125247 * max(0.0, 0.000504949057 - Q.lam1) / 8.739618817810877e-05   # +1.8%  lam1 < 0.0005049
        - 0.01752237536901245 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) / 0.004299658381541595   # -1.8%  mass_over_sum_pt_sq < 0.008175
        - 0.016426705120958562 * max(0.0, 0.007929074034 - Q.girth2_top3) / 0.004560262122333358   # -1.6%  girth2_top3 < 0.007929
        - 0.015797102332700673 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7) / 229.29996254831684   # -1.6%  mass < 64.62 and pt_7 < 40.04
        + 0.015486377504239209 * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.016491452412829836   # +1.5%  centroid_offset < 0.03117
        - 0.013894672144152347 * max(0.0, Q.sum_pt - 813.415625) / 31.70376994639103   # -1.4%  sum_pt > 813.4
        - 0.011683575293868601 * max(0.0, 0.00832969537 - Q.girth2_top5) / 0.004544146438161959   # -1.2%  girth2_top5 < 0.00833
        - 0.01166475746740149 * max(0.0, 0.00543336053 - Q.lam1) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.004282963475752678   # -1.2%  lam1 < 0.005433 and n_dr_0p2_0p4 < 2
        - 0.01022082186781634 * max(0.0, 0.001503553356 - Q.lam1) / 0.0003926183331153187   # -1.0%  lam1 < 0.001504
        + 0.010065240945817969 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117) / 0.16024311424427862   # +1.0%  mass < 64.62 and centroid_offset > 0.01259
        + 0.010061754397542515 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619112969407   # +1.0%  sum_pt_top5 > 687.4
        + 0.010034576049158115 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539) / 0.4331515118014987   # +1.0%  mass < 29.64 and phi_1 > -0.0589
        + 0.008591657308319397 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.072321663733 - Q.dr_4) / 0.002398340732342072   # +0.9%  log_sum_pt > 6.638 and dr_4 < 0.07232
        + 0.008416316564016386 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) / 0.0544885966340469   # +0.8%  z_dr_0_0p05 > 0.8477
        + 0.008134793992880892 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153) / 0.0001670441522630915   # +0.8%  girth2 < 0.01883 and eccentricity > 0.9599
        + 0.00771419835998292 * max(0.0, 0.00543336053 - Q.lam1) / 0.0021941400339846296   # +0.8%  lam1 < 0.005433
        + 0.007327307730845413 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.01195259460622648   # +0.7%  mass_over_sum_pt_sq < 0.008175 and n_pt_above_50 < 8
        - 0.007287352459213004 * max(0.0, 0.000657050184 - Q.girth2_top5) / 0.0001395474066313899   # -0.7%  girth2_top5 < 0.0006571
        - 0.005726669827884321 * max(0.0, 21.784077072144 - Q.mass) / 4.431023211632268   # -0.6%  mass < 21.78
        - 0.00483560981353462 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, Q.pt_7 - 29.0421875) / 87.66880630219308   # -0.5%  sum_pt > 901.6 and pt_7 > 29.04
        - 0.004190939689408618 * max(0.0, Q.sum_pt - 901.59375) / 12.403615355829832   # -0.4%  sum_pt > 901.6
        + 0.004056384209381234 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.078464230803 - Q.dr_3) / 0.002829820778202598   # +0.4%  log_sum_pt > 6.638 and dr_3 < 0.07846
        - 0.003833873642840408 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2) / 7.962190634589057e-05   # -0.4%  lam1 < 0.006507 and D2 < 0.8757
        + 0.003702329633862779 * max(0.0, 0.000657050184 - Q.girth2_top5) * max(0.0, 0.222994708167 - Q.dr_7) / 2.7227213136631776e-05   # +0.4%  girth2_top5 < 0.0006571 and dr_7 < 0.223
        - 0.0033436688833601604 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, 0.063364507347 - Q.dr_4) / 1.3449142440644053   # -0.3%  sum_pt_top5 > 687.4 and dr_4 < 0.06336
        + 0.002732509849204119 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605) / 0.07430351940037334   # +0.3%  mass < 56.92 and C2 > 0.02384
        + 0.0027278045780447873 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7) / 68.64747957984208   # +0.3%  sum_pt > 901.6 and pt_7 < 25.58
        - 0.001993562286160509 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189) / 0.19805534351472132   # -0.2%  sum_pt_top5 > 687.4 and z_7 > 0.02321
        - 0.0018919041752253928 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2) / 0.021060647298465133   # -0.2%  mass < 29.64 and D2 < 0.8757
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 13.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.242221095479307 * (0.07358755611638855
        + 0.10974128436282315 * max(0.0, 0.008168570676 - Q.e2_sq) / 0.004294747328233826   # +11.0%  e2_sq < 0.008169
        - 0.07607714865113495 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # -7.6%  width < 0.008678
        - 0.07497247017986948 * max(0.0, 53.332374954224 - Q.mass) / 19.36003866872732   # -7.5%  mass < 53.33
        + 0.07324384670265231 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # +7.3%  log_sum_pt > 6.378
        - 0.07130858314076463 * max(0.0, 0.250761204958 - Q.max_dr) / 0.13165424828281402   # -7.1%  max_dr < 0.2508
        - 0.06486810587989836 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 0.000537286005 - Q.lam2) / 1.9807530097002534e-06   # -6.5%  lam1 < 0.008376 and lam2 < 0.0005373
        - 0.05914987182493234 * max(0.0, 0.055577157257 - Q.z_7) / 0.010554877533209049   # -5.9%  z_7 < 0.05558
        + 0.051860259836910394 * max(0.0, Q.pt_7 - 34.53125) / 4.473662132352941   # +5.2%  pt_7 > 34.53
        + 0.03329316666861077 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr) / 0.021256915339589413   # +3.3%  log_sum_pt > 6.378 and max_dr < 0.198
        + 0.03251046663806212 * max(0.0, 49.668099212646 - Q.mass) / 17.068932039075   # +3.3%  mass < 49.67
        - 0.029845191727590578 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass) / 238.71814489127775   # -3.0%  pt_7 > 34.53 and mass < 91.19
        + 0.029232996522134493 * max(0.0, Q.log_sum_pt - 6.638338705138) / 0.05708668543535571   # +2.9%  log_sum_pt > 6.638
        + 0.028915724366033655 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2) / 9.286869338661969e-05   # +2.9%  log_sum_pt > 6.573 and lam2 < 0.001131
        + 0.026834275265230006 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.0140332421 - Q.girth2_top2) / 0.0001273031329778418   # +2.7%  z_7 < 0.05558 and girth2_top2 < 0.01403
        - 0.022445912453575866 * max(0.0, Q.sum_pt_top5 - 579.875) / 79.44550428620667   # -2.2%  sum_pt_top5 > 579.9
        + 0.022316966784400726 * max(0.0, Q.log_sum_pt - 6.572937922293) / 0.08631512213470326   # +2.2%  log_sum_pt > 6.573
        - 0.021648995679980285 * max(0.0, 0.043044721986 - Q.z_7) / 0.004926447639348283   # -2.2%  z_7 < 0.04304
        - 0.021373071905007316 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow) / 5.359939441280181e-05   # -2.1%  e2_sq < 0.008169 and planar_flow < 0.08366
        - 0.01996703429788636 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.0029084479897083337   # -2.0%  log_sum_pt > 6.378 and centroid_offset < 0.02355
        + 0.01952895108949682 * max(0.0, Q.LHA - 0.266912960293) / 0.031878321592982185   # +2.0%  LHA > 0.2669
        + 0.019102707279187302 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925) / 3.5352354845618814e-05   # +1.9%  e2 < 0.03556 and eccentricity > 0.9782
        + 0.018205559168400658 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.003408388935 - Q.lam2) / 3.4475134450127075e-05   # +1.8%  z_7 < 0.05558 and lam2 < 0.003408
        - 0.016212199924829952 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.007929074034 - Q.girth2_top3) / 0.00036921026079211606   # -1.6%  log_sum_pt > 6.638 and girth2_top3 < 0.007929
        - 0.01218398186692031 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.00138715893483794   # -1.2%  lam1 < 0.008376 and n_pt_above_50 > 6
        - 0.009651723742441958 * max(0.0, Q.e2 - 0.032346998155) / 0.00800322490328291   # -1.0%  e2 > 0.03235
        + 0.00957138718488651 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0013761153368809544   # +1.0%  e2_sq < 0.008169 and n_pt_above_50 > 6
        - 0.00882364807355429 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow) / 0.0013308621908643308   # -0.9%  LHA < 0.3467 and planar_flow < 0.08366
        + 0.007711285729416624 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205) / 7.4336756001165145e-06   # +0.8%  lam1 < 0.008376 and centroid_offset > 0.02077
        + 0.00639333499577584 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr) / 0.08998780021469473   # +0.6%  pt_7 > 34.53 and max_dr < 0.08051
        - 0.0030098480575921098 * max(0.0, Q.pt_7 - 53.4375) / 0.3316342962184874   # -0.3%  pt_7 > 53.44
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 8.903;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.902912973016285 * (0.2610926948780995
        - 0.17569990096920354 * max(0.0, Q.LHA - 0.111565049159) / 0.1352185061481073   # -17.6%  LHA > 0.1116
        - 0.1342974947386804 * max(0.0, 53.4375 - Q.pt_7) / 19.120943117285975   # -13.4%  pt_7 < 53.44
        - 0.09984977098990909 * max(0.0, 36.229410171509 - Q.mass) / 10.113929790442333   # -10.0%  mass < 36.23
        + 0.08289899913540824 * max(0.0, 69.611351776123 - Q.mass) / 31.697292539607695   # +8.3%  mass < 69.61
        + 0.07428917207503637 * max(0.0, 788.4484375 - Q.sum_pt) / 112.15755791313566   # +7.4%  sum_pt < 788.4
        + 0.07335511359178078 * max(0.0, 43.5 - Q.pt_7) / 10.288609083672531   # +7.3%  pt_7 < 43.5
        + 0.04787588037122199 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2) / 0.010855764031987139   # +4.8%  mass < 36.23 and lam2 < 0.001131
        + 0.04711923030600479 * max(0.0, Q.pt_7 - 30.484375) / 6.808825630252101   # +4.7%  pt_7 > 30.48
        - 0.045043860961179245 * max(0.0, 6.638338705138 - Q.log_sum_pt) / 0.15249355870011386   # -4.5%  log_sum_pt < 6.638
        + 0.04298764730466308 * max(0.0, 6.572937922293 - Q.log_sum_pt) / 0.11632121255387393   # +4.3%  log_sum_pt < 6.573
        + 0.03501018959892588 * max(0.0, 0.005954149834 - Q.lam1) / 0.0025181540815307126   # +3.5%  lam1 < 0.005954
        + 0.03413735215541597 * max(0.0, 0.003377388461 - Q.lam1) / 0.001133379805322733   # +3.4%  lam1 < 0.003377
        - 0.024650106275197183 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2) / 0.21065871608184009   # -2.5%  pt_7 > 30.48 and C2 < 0.05119
        + 0.021462042996217724 * max(0.0, 6.464150123592 - Q.log_sum_pt) / 0.0700995140842757   # +2.1%  log_sum_pt < 6.464
        - 0.016454450993770547 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7) / 2.998515264059948e-06   # -1.6%  width < 0.0001722 and dr_7 < 0.223
        + 0.014843957109954975 * max(0.0, 0.000172198326 - Q.width) / 1.4418964712323033e-05   # +1.5%  width < 0.0001722
        - 0.014791743884842256 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299) / 0.24796742071553304   # -1.5%  pt_7 > 30.48 and max_dr > 0.09311
        - 0.011353011415763557 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025) / 4.493327216036786e-05   # -1.1%  lam1 < 0.005954 and max_dr > 0.08051
        - 0.0038800751268244227 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # -0.4%  log_sum_pt > 6.896
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 30.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 30.440356974492968 * (0.1197606574785116
        - 0.264010589949906 * max(0.0, 0.008678044751 - Q.girth2) / 0.004447347389619223   # -26.4%  girth2 < 0.008678
        - 0.12611094935120198 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # -12.6%  girth2 < 0.01324
        + 0.09738312671095598 * max(0.0, 0.04447356835 - Q.e2) / 0.020193337934351147   # +9.7%  e2 < 0.04447
        + 0.05953029195927053 * max(0.0, 0.004372139461 - Q.girth2) / 0.001565710312409871   # +6.0%  girth2 < 0.004372
        + 0.05566728866929277 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 0.06729416374778562   # +5.6%  mass_over_sum_pt > 0.06814 and n_dr_0_0p05 < 5
        + 0.050484242342685726 * max(0.0, 0.007330079875 - Q.lam1) / 0.00348685261907903   # +5.0%  lam1 < 0.00733
        - 0.04580620131058508 * max(0.0, Q.lam1 - 0.004183811014) / 0.0032456267321051657   # -4.6%  lam1 > 0.004184
        - 0.030711844610159432 * max(0.0, Q.e2 - 0.028531698044) / 0.009633336184298937   # -3.1%  e2 > 0.02853
        + 0.030297431132618055 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2) / 0.35204737477784626   # +3.0%  z_dr_0p1_0p2 < 0.4684
        + 0.025598228408661123 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +2.6%  e2 < 0.03847
        + 0.025568386459856672 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573) / 4.350681513338082   # +2.6%  mass > 36.23 and eccentricity > 0.6207
        - 0.01877743087534385 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) / 0.015391574669539157   # -1.9%  mass_over_sum_pt > 0.06814
        + 0.014735389419463568 * max(0.0, Q.width - 0.018827653081) / 0.0007605368512955965   # +1.5%  width > 0.01883
        + 0.014489671365423423 * max(0.0, Q.centroid_offset - 0.014379521101) / 0.0071112999721223424   # +1.4%  centroid_offset > 0.01438
        - 0.013719797690577107 * max(0.0, Q.mass - 36.229410171509) / 14.205281284755415   # -1.4%  mass > 36.23
        - 0.011751563489374972 * max(0.0, Q.lam1 - 0.012003726523) / 0.0012289213133753257   # -1.2%  lam1 > 0.012
        + 0.01169740438242849 * max(0.0, Q.LHA - 0.325582223496) / 0.01245304382641771   # +1.2%  LHA > 0.3256
        - 0.010177598250142798 * max(0.0, 0.006679471358 - Q.width) / 0.002936239797737678   # -1.0%  width < 0.006679
        + 0.008613840912118831 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.00829854327583745   # +0.9%  mass_over_sum_pt > 0.09041
        + 0.007308843483586822 * max(0.0, Q.e2 - 0.050284641981) / 0.00336902922782692   # +0.7%  e2 > 0.05028
        - 0.00716186196602925 * max(0.0, Q.mass_over_sum_pt_sq - 0.006387803907) / 0.002451625171049358   # -0.7%  mass_over_sum_pt_sq > 0.006388
        + 0.006832375256719933 * max(0.0, Q.max_dr - 0.145231109113) / 0.026281229857574792   # +0.7%  max_dr > 0.1452
        - 0.006244089444260726 * max(0.0, Q.lam1 - 0.008375572068) / 0.0018409335087098556   # -0.6%  lam1 > 0.008376
        + 0.00612072383217691 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6) / 0.14624935923720359   # +0.6%  mass_over_sum_pt > 0.09041 and pt_6 < 56.53
        - 0.005848357560454922 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481179386   # -0.6%  mass_over_sum_pt > 0.108
        - 0.005629868073626026 * max(0.0, Q.mass - 69.611351776123) / 2.4067024290003474   # -0.6%  mass > 69.61
        - 0.0049482851907769404 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, 0.075389597551 - Q.z_7) / 4.8230619117404544e-05   # -0.5%  lam1 > 0.004184 and z_7 < 0.07539
        + 0.004929214012554845 * max(0.0, Q.lam1 - 0.016433749775) / 0.0006837082011327267   # +0.5%  lam1 > 0.01643
        - 0.004717673692365788 * max(0.0, Q.C2 - 0.094821243733) / 0.0008333835716224887   # -0.5%  C2 > 0.09482
        - 0.004361217393281314 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.047732555332 - Q.dr_1) / 0.0001917605414039403   # -0.4%  max_dr > 0.1452 and dr_1 < 0.04773
        - 0.0030650497540285738 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr) / 4.2402205608029813e-05   # -0.3%  LHA > 0.3127 and max_dr < 0.1452
        + 0.0029611595144126705 * max(0.0, 0.008678044751 - Q.girth2) * max(0.0, Q.pt1_dr01 - 5.351076855015) / 0.004895578219593007   # +0.3%  girth2 < 0.008678 and pt1_dr01 > 5.351
        + 0.0029269779026887006 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.pt_7 - 25.578125) / 0.0673996460089272   # +0.3%  centroid_offset > 0.01438 and pt_7 > 25.58
        - 0.0027411215488403573 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3) / 0.28212928800771087   # -0.3%  LHA > 0.3127 and mass_top3 < 50.35
        - 0.0024798352811591885 * max(0.0, Q.LHA - 0.423592510895) / 0.0013397344097334583   # -0.2%  LHA > 0.4236
        - 0.0016895919601793992 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7) / 1.0188469566497442e-05   # -0.2%  mass_over_sum_pt > 0.06814 and dr_7 < 0.04215
        + 0.001362170168904883 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.068101508468 - Q.z_7) / 4.1348499599905125e-05   # +0.1%  mass_over_sum_pt > 0.108 and z_7 < 0.0681
        + 0.0013460041737540648 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.04881348081 - Q.dr_7) / 3.410777789559402e-06   # +0.1%  mass_over_sum_pt > 0.108 and dr_7 < 0.04881
        + 0.0011260865126432092 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr) / 3.112203819684481e-06   # +0.1%  mass_over_sum_pt > 0.09041 and max_dr < 0.1452
        - 0.0010682159874891074 * max(0.0, Q.width - 0.018827653081) * max(0.0, Q.z_3 - 0.090493038582) / 1.7485663937045227e-05   # -0.1%  width > 0.01883 and z_3 > 0.09049
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 85.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 85.43783046166995 * (-0.14493034245122594
        + 0.2814392066761224 * max(0.0, 0.017162483186 - Q.e2_sq) / 0.01203539920865141   # +28.1%  e2_sq < 0.01716
        - 0.21152296015281422 * max(0.0, 0.017142307326 - Q.mass_over_sum_pt_sq) / 0.012017282791920023   # -21.2%  mass_over_sum_pt_sq < 0.01714
        + 0.03517708097606204 * max(0.0, Q.girth2 - 0.003562611155) / 0.004066872434995929   # +3.5%  girth2 > 0.003563
        + 0.028825389489627533 * max(0.0, Q.e2 - 0.020459658932) / 0.013704419938894725   # +2.9%  e2 > 0.02046
        - 0.026262363981347736 * max(0.0, Q.width - 0.000319370692) / 0.006166221167135747   # -2.6%  width > 0.0003194
        - 0.026076629794921913 * max(0.0, Q.girth2 - 0.007520088344) / 0.002470136248856422   # -2.6%  girth2 > 0.00752
        + 0.024290292507640778 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481179386   # +2.4%  mass_over_sum_pt > 0.108
        - 0.02262684200167286 * max(0.0, 0.003013300392 - Q.e2_sq) / 0.0010659437602397771   # -2.3%  e2_sq < 0.003013
        + 0.022565725850048075 * max(0.0, Q.width - 0.001653836415) / 0.005220304336946791   # +2.3%  width > 0.001654
        + 0.022434443924128904 * max(0.0, 0.23799610585 - Q.tau21) / 0.05620226106196825   # +2.2%  tau21 < 0.238
        + 0.02219929861467774 * max(0.0, 0.023780909279 - Q.e2_sq) / 0.018171241308920595   # +2.2%  e2_sq < 0.02378
        + 0.021765568274601663 * max(0.0, Q.width - 0.002635417778) / 0.004603950979357016   # +2.2%  width > 0.002635
        + 0.020208557741877776 * max(0.0, 0.076081777364 - Q.girth) / 0.027967843803613616   # +2.0%  girth < 0.07608
        + 0.019292544774670663 * max(0.0, 0.124553743005 - Q.girth) / 0.06848161639571908   # +1.9%  girth < 0.1246
        - 0.018327770480340365 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.00829854327583745   # -1.8%  mass_over_sum_pt > 0.09041
        - 0.013382468581875034 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2) / 0.015907178094480428   # -1.3%  mass_over_sum_pt > 0.108 and D2 < 3.886
        + 0.01313669985607473 * max(0.0, Q.e2 - 0.007078157854) / 0.022325491981794805   # +1.3%  e2 > 0.007078
        - 0.011502271789162008 * max(0.0, Q.C2 - 0.014943876117) / 0.015904628092890307   # -1.2%  C2 > 0.01494
        - 0.009789702982362088 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3) / 0.0076573303900181015   # -1.0%  lam2 < 0.0003061 and mass_top3 < 50.35
        - 0.009689224395402075 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1) / 0.002138030313462078   # -1.0%  centroid_offset < 0.01438 and z_dr_0p05_0p1 < 0.6748
        + 0.009144414348685953 * max(0.0, 80.4 - Q.mass) / 41.295087132244156   # +0.9%  mass < 80.4
        - 0.008667493582122496 * max(0.0, 0.009530300104 - Q.girth2_top2) / 0.0061095731516265135   # -0.9%  girth2_top2 < 0.00953
        + 0.00864230030963515 * max(0.0, 0.001130644719 - Q.lam2) / 0.0009137232342242098   # +0.9%  lam2 < 0.001131
        + 0.008379641659629223 * max(0.0, 0.007182789718 - Q.e2_sq) / 0.0035287869395175283   # +0.8%  e2_sq < 0.007183
        + 0.007987962068815952 * max(0.0, Q.max_dr - 0.102758520097) / 0.04505724462361834   # +0.8%  max_dr > 0.1028
        - 0.0075047436830034215 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2) / 5.628323650212299e-05   # -0.8%  tau21 < 0.238 and lam2 < 0.001131
        - 0.007169215447921585 * max(0.0, Q.girth2 - 0.013238675334) / 0.0014426835012777079   # -0.7%  girth2 > 0.01324
        + 0.007159980479586748 * max(0.0, 56.920347213745 - Q.mass) / 21.784746347219   # +0.7%  mass < 56.92
        - 0.007116724537522612 * max(0.0, 0.047915700823 - Q.girth) / 0.012161121217552112   # -0.7%  girth < 0.04792
        + 0.006463186345889445 * max(0.0, 0.005011406868 - Q.girth2_top3) / 0.002432330601058206   # +0.6%  girth2_top3 < 0.005011
        - 0.006066048287458573 * max(0.0, 0.007639643088 - Q.girth2_top2) / 0.004543328598712764   # -0.6%  girth2_top2 < 0.00764
        + 0.005371561067008212 * max(0.0, 763.825 - Q.sum_pt) / 96.68366216443064   # +0.5%  sum_pt < 763.8
        - 0.005253669824186436 * max(0.0, Q.e2 - 0.041109715588) / 0.005106967369295564   # -0.5%  e2 > 0.04111
        - 0.00394683635844798 * max(0.0, Q.C2 - 0.067292226106) / 0.002854902335729804   # -0.4%  C2 > 0.06729
        + 0.003912631744257628 * max(0.0, 41.377904891968 - Q.mass) / 12.537841289638873   # +0.4%  mass < 41.38
        + 0.003563179382464081 * max(0.0, 0.014379521101 - Q.centroid_offset) / 0.0043064829328538275   # +0.4%  centroid_offset < 0.01438
        - 0.003398868826299207 * max(0.0, Q.max_dr - 0.197968879342) / 0.012226149933645198   # -0.3%  max_dr > 0.198
        + 0.003056329367826725 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3) / 2.3095576214583597   # +0.3%  mass < 69.61 and dr_3 < 0.1042
        + 0.0030111652117067395 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848) / 1.967884659027567e-05   # +0.3%  girth2_top2 < 0.00953 and centroid_offset > 0.01628
        + 0.00298010186398121 * max(0.0, 0.003111083776 - Q.girth2_top2) / 0.001348730243668823   # +0.3%  girth2_top2 < 0.003111
        + 0.0028236629713780286 * max(0.0, Q.girth - 0.101940929517) / 0.005400003353656977   # +0.3%  girth > 0.1019
        - 0.002817923856204489 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq) / 0.0003082102790484629   # -0.3%  mass_over_sum_pt_sq < 0.001102
        - 0.002234034836231256 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass) / 0.42797494392630053   # -0.2%  tau21 < 0.238 and mass < 62.55
        - 0.0021744662908203148 * max(0.0, Q.e2 - 0.063441075385) / 0.0017618611738475503   # -0.2%  e2 > 0.06344
        - 0.001729302188938893 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 68.70525244030448   # -0.2%  sum_pt < 763.8 and n_dr_0p2_0p4 < 1
        + 0.0016013122661795328 * max(0.0, Q.max_dr - 0.197968879342) * max(0.0, Q.z_7 - 0.028070914944) / 0.00031334476219344867   # +0.2%  max_dr > 0.198 and z_7 > 0.02807
        - 0.0014982978143809663 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7) / 0.004744492327402142   # -0.1%  tau21 < 0.238 and dr_7 < 0.1755
        - 0.0013991086996818754 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.eccentricity - 0.984196588116) / 0.0001760088494483609   # -0.1%  max_dr > 0.1028 and eccentricity > 0.9842
        - 0.001387645045517267 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625) / 0.08633888537871398   # -0.1%  max_dr > 0.1028 and pt_7 > 37.16
        + 0.000913448631988808 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875) / 0.3246332585626503   # +0.1%  tau21 < 0.238 and pt_7 > 33.22
        + 0.0008126631463336086 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125) / 0.023902004294371636   # +0.1%  C2 > 0.01494 and pt_7 > 38.53
        - 0.0006626345872348977 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702) / 7.209993946981458e-05   # -0.1%  tau21 < 0.238 and e2_sq > 0.01166
        + 0.0005227862290242324 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346) / 0.0019149421885925775   # +0.1%  tau21 < 0.238 and planar_flow > 0.04506
        - 0.00011161619420579021 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_6 - 62.25) / 6.0655998638359284e-05   # -0.0%  girth2 > 0.01883 and pt_6 > 62.25
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 15.6;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.598800003651505 * (0.006077586582865111
        + 0.06305688486358448 * max(0.0, 60.630975723267 - Q.mass) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 47.85276437032764   # +6.3%  mass < 60.63 and n_dr_0p2_0p4 < 2
        + 0.059469981528525076 * max(0.0, 0.216055863061 - Q.LHA) / 0.03222864900799722   # +5.9%  LHA < 0.2161
        + 0.05679542832793109 * max(0.0, 0.035560912266 - Q.e2) / 0.01371972344513943   # +5.7%  e2 < 0.03556
        - 0.051446603915568934 * max(0.0, 548.196875 - Q.sum_pt_top2) / 195.28859533251793   # -5.1%  sum_pt_top2 < 548.2
        + 0.04764733169766634 * max(0.0, 0.071488645583 - Q.z_7) / 0.021404874347378918   # +4.8%  z_7 < 0.07149
        - 0.04455452005352577 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt) / 0.05207765257754601   # -4.5%  mass_over_sum_pt < 0.108
        + 0.043869982913361326 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2) / 2.1406489096621576e-05   # +4.4%  z_7 < 0.07149 and lam2 < 0.001131
        + 0.041635453685102464 * max(0.0, 53.4375 - Q.pt_7) / 19.120943117285975   # +4.2%  pt_7 < 53.44
        - 0.041173010927434944 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1) / 0.0008217429754174446   # -4.1%  log_sum_pt > 6.573 and lam1 < 0.012
        - 0.036716703366176 * max(0.0, 0.021588001063 - Q.dr_0) / 0.0036021402838930334   # -3.7%  dr_0 < 0.02159
        + 0.03535044396578129 * max(0.0, Q.log_sum_pt - 6.572937922293) / 0.08631512213470326   # +3.5%  log_sum_pt > 6.573
        - 0.030031694230219484 * max(0.0, 0.004007841607 - Q.girth2_top2) / 0.0019072108050221064   # -3.0%  girth2_top2 < 0.004008
        + 0.024707950059266657 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0) / 0.0006616654182381468   # +2.5%  log_sum_pt > 6.573 and dr_0 < 0.02159
        + 0.02406447904250046 * max(0.0, 0.002635417778 - Q.width) / 0.0007941346639111337   # +2.4%  width < 0.002635
        - 0.02246295058389449 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.00407349257511869   # -2.2%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.02138537401851155 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00231612516) / 0.000134444866740672   # -2.1%  LHA < 0.2161 and centroid_offset > 0.002316
        + 0.021224064051158342 * max(0.0, 0.05096141791 - Q.z_6) / 0.005501945113704042   # +2.1%  z_6 < 0.05096
        - 0.021030306942805818 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) / 0.03304117548546515   # -2.1%  mass_over_sum_pt < 0.08475
        - 0.019672831467044665 * max(0.0, Q.sum_pt - 868.509375) / 18.084627120331504   # -2.0%  sum_pt > 868.5
        - 0.01765196969674981 * max(0.0, Q.log_sum_pt - 6.842716632804) / 0.007875918226781813   # -1.8%  log_sum_pt > 6.843
        - 0.017253111669101838 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset) / 1.318014685216707e-05   # -1.7%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.01636833247842131 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset) / 0.13441460519175305   # +1.6%  sum_pt > 868.5 and centroid_offset < 0.01259
        + 0.016198504023320116 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2) / 0.0004246061902383711   # +1.6%  log_sum_pt > 6.573 and girth2_top2 < 0.0063
        - 0.015947873133791354 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass) / 0.49027896930529   # -1.6%  mass_over_sum_pt < 0.08475 and max_pair_mass < 18.1
        - 0.015756504987155858 * max(0.0, 0.154689112391 - Q.LHA) / 0.012614340772605542   # -1.6%  LHA < 0.1547
        - 0.01567361190473601 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt) / 0.9285639647784526   # -1.6%  z_7 < 0.07149 and sum_pt < 788.4
        - 0.014209312268303817 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.0739931336990729   # -1.4%  log_sum_pt > 6.573 and n_dr_0p2_0p4 < 1
        + 0.013918445360786753 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset) / 1.2223894287536178e-05   # +1.4%  width < 0.002635 and centroid_offset < 0.02355
        + 0.01375004180174406 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.274147637091883   # +1.4%  sum_pt_top5 > 752.1
        - 0.013596145317394245 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 0.010960638421 - Q.centroid_offset) / 3.279841250636351e-05   # -1.4%  z_7 < 0.0494 and centroid_offset < 0.01096
        + 0.012332090873600276 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2) / 5.495977337611781e-07   # +1.2%  e2 < 0.03556 and lam2 < 7.301e-05
        - 0.011624888852925713 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 73.75 - Q.pt_5) / 0.27549178577716005   # -1.2%  z_7 < 0.0494 and pt_5 < 73.75
        + 0.010572365814762714 * max(0.0, 0.023207568189 - Q.z_7) / 0.0007159291123534974   # +1.1%  z_7 < 0.02321
        + 0.010472656643741538 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5) / 0.3059458585195602   # +1.0%  z_7 < 0.0494 and mass_top5 < 62.55
        - 0.009057031153164289 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset) / 5.203533472960053e-05   # -0.9%  log_sum_pt > 6.896 and centroid_offset < 0.01838
        + 0.008778449239373452 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2) / 4.16958841658888e-06   # +0.9%  log_sum_pt > 6.573 and mean_phi2 < 0.0001455
        + 0.008483960930559689 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # +0.8%  log_sum_pt > 6.896
        + 0.008266378234998045 * max(0.0, 0.028865759995 - Q.z_6) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.0013546644338533558   # +0.8%  z_6 < 0.02887 and n_dr_0p2_0p4 < 2
        + 0.008138666175199143 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0) / 0.40891493757937986   # +0.8%  sum_pt_top2 < 548.2 and dr_0 < 0.02159
        - 0.006649438889758426 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, 29.875 - Q.pt_5) / 0.011118081204044762   # -0.7%  z_7 < 0.03243 and pt_5 < 29.88
        + 0.006134308383313674 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2) / 2.0281431575878195e-06   # +0.6%  log_sum_pt > 6.573 and mean_eta2 < 9.03e-05
        + 0.0056397157300252755 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 24.578125 - Q.pt_5) / 0.003786120967577437   # +0.6%  mean_phi2 < 0.01426 and pt_5 < 24.58
        - 0.005061351772472099 * max(0.0, 0.028865759995 - Q.z_6) / 0.0008381782190062135   # -0.5%  z_6 < 0.02887
        - 0.004052817132522234 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0) / 0.00011031021395682367   # -0.4%  log_sum_pt > 6.896 and dr_0 < 0.04119
        - 0.0033780960803845147 * max(0.0, 0.05096141791 - Q.z_6) * max(0.0, Q.e2_sq - 0.003902458471) / 5.701478550833789e-06   # -0.3%  z_6 < 0.05096 and e2_sq > 0.003902
        - 0.00256572420760846 * max(0.0, Q.sum_pt_top5 - 752.1) * max(0.0, 1.679198372364 - Q.D2) / 5.264964520129756   # -0.3%  sum_pt_top5 > 752.1 and D2 < 1.679
        - 0.0021722116040259953 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.045057236346 - Q.planar_flow) / 2.1587905933501866e-05   # -0.2%  log_sum_pt > 6.896 and planar_flow < 0.04506
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 37.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.06584391297999 * (0.16911254246630342
        - 0.15818858012918582 * max(0.0, 0.013238675006 - Q.width) / 0.00823612446188605   # -15.8%  width < 0.01324
        - 0.14807333604761094 * max(0.0, 0.008678044751 - Q.girth2) / 0.004447347389619223   # -14.8%  girth2 < 0.008678
        + 0.08965340313473569 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # +9.0%  e2 < 0.05028
        - 0.08847717068619723 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) / 0.053262875947277605   # -8.8%  mass_over_sum_pt > 0.008374
        - 0.046327108738659015 * max(0.0, 0.024419631481 - Q.girth2_top5) / 0.018899543128555206   # -4.6%  girth2_top5 < 0.02442
        + 0.04466123744636837 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt) / 0.0937270068310905   # +4.5%  mass_over_sum_pt < 0.1542
        + 0.042424099770867175 * Q.max_dr / 0.1236972946900221   # +4.2%  max_dr
        + 0.03435297305124778 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2) / 2.6474453560607286e-05   # +3.4%  centroid_offset > 0.008092 and lam2 < 0.003408
        - 0.03153188139517475 * max(0.0, Q.centroid_offset - 0.018377780003) / 0.005523694064633852   # -3.2%  centroid_offset > 0.01838
        - 0.027971456455769 * max(0.0, 6.701242202626 - Q.log_sum_pt) / 0.19354909741955528   # -2.8%  log_sum_pt < 6.701
        + 0.026082738276529578 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # +2.6%  lam1 < 0.008376
        + 0.02346271000398212 * max(0.0, 0.009530300104 - Q.girth2_top2) / 0.0061095731516265135   # +2.3%  girth2_top2 < 0.00953
        + 0.01986705309395174 * max(0.0, 6.572937922293 - Q.log_sum_pt) / 0.11632121255387393   # +2.0%  log_sum_pt < 6.573
        + 0.018712227371931987 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1) / 11.464259601117858   # +1.9%  mass < 49.67 and z_dr_0p05_0p1 < 0.7509
        + 0.0164924656252242 * max(0.0, 1.679198372364 - Q.D2) / 0.5633062396133152   # +1.6%  D2 < 1.679
        - 0.015285704455174359 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 7.678122883839528e-05   # -1.5%  lam2 < 0.0005373 and z_dr_0p2_0p4 < 0.2055
        + 0.014052557017495327 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7) / 1.995212531081465   # +1.4%  log_sum_pt < 6.701 and pt_7 < 45.75
        + 0.013525697666712225 * max(0.0, Q.girth - 0.087236513197) / 0.007853436041615635   # +1.4%  girth > 0.08724
        - 0.012297442683100695 * max(0.0, Q.girth2_top5 - 0.002270363079) / 0.0043533388582812135   # -1.2%  girth2_top5 > 0.00227
        - 0.011944234530287489 * max(0.0, Q.lam2 - 0.003408388935) / 0.00015702993621826726   # -1.2%  lam2 > 0.003408
        + 0.010351019706059093 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787) / 0.0009017529775427706   # +1.0%  LHA > 0.3127 and eccentricity > 0.8727
        - 0.009938022451956791 * max(0.0, 5.351076855015 - Q.pt1_dr01) / 3.22124055682881   # -1.0%  pt1_dr01 < 5.351
        + 0.007957224874022159 * max(0.0, Q.girth2_top5 - 0.011482925368) / 0.0015471296538189059   # +0.8%  girth2_top5 > 0.01148
        - 0.0073374402067202605 * max(0.0, Q.z_dr_0_0p05 - 0.768138587475) / 0.09210424062684407   # -0.7%  z_dr_0_0p05 > 0.7681
        + 0.006653356936631881 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781) / 0.0027716438038630276   # +0.7%  centroid_offset > 0.008092 and planar_flow > 0.008049
        - 0.006469851163968381 * max(0.0, Q.sum_pt_top5 - 839.9546875) / 8.60588864380794   # -0.6%  sum_pt_top5 > 840
        - 0.006385904214306855 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4) / 17.624128555880276   # -0.6%  D2 < 1.679 and pt_4 < 90.62
        + 0.00571010957079483 * max(0.0, Q.sum_pt - 988.4078125) / 4.402477808070966   # +0.6%  sum_pt > 988.4
        + 0.005682319965001577 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.004247450386) / 0.0005461586200825784   # +0.6%  log_sum_pt < 6.701 and mean_eta2 > 0.004247
        + 0.005466703063712501 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2) / 2.1793621511481152e-05   # +0.5%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        - 0.004644218283520781 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 41.65625 - Q.pt_7) / 0.42048592691249276   # -0.5%  mass_over_sum_pt > 0.008374 and pt_7 < 41.66
        - 0.004306341368512121 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32) / 0.00581747934750154   # -0.4%  mass_over_sum_pt > 0.008374 and tau32 < 0.5187
        + 0.003759822787369996 * max(0.0, 24.421875 - Q.pt_6) / 0.6046572405133929   # +0.4%  pt_6 < 24.42
        + 0.003628292366912752 * max(0.0, Q.girth2_top5 - 0.002270363079) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361) / 0.0007317482583488287   # +0.4%  girth2_top5 > 0.00227 and z_dr_0p05_0p1 > 0.2919
        - 0.0032502607545965845 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.pt_7 - 15.55390625) / 0.029970388327448836   # -0.3%  girth2_top5 > 0.01148 and pt_7 > 15.55
        + 0.0031452260417216475 * max(0.0, Q.lam2 - 0.000537286005) / 0.00038257031407305515   # +0.3%  lam2 > 0.0005373
        + 0.0028746474269830662 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6) / 0.41268944603937285   # +0.3%  log_sum_pt < 6.701 and pt_6 < 36.81
        - 0.0028297005728345945 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.mean_eta - 0.0127187056) / 1.0990450553759857e-05   # -0.3%  girth2_top5 > 0.01148 and mean_eta > 0.01272
        + 0.0024703663283242696 * max(0.0, Q.centroid_offset - 0.049903668404) / 0.0008064001164412618   # +0.2%  centroid_offset > 0.0499
        + 0.00241722456656743 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179) / 0.0002133523951813162   # +0.2%  e2 < 0.05028 and z_dr_0p1_0p2 > 0.1586
        + 0.0023748574393968602 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375) / 0.08096163333875342   # +0.2%  C2 > 0.01054 and pt_7 > 31.86
        - 0.0023087036726835222 * max(0.0, 0.037477688199 - Q.z_4) / 0.00047222496069883454   # -0.2%  z_4 < 0.03748
        + 0.0021442726579590193 * max(0.0, 31.125 - Q.pt_4) / 0.31407923368566176   # +0.2%  pt_4 < 31.12
        - 0.0021357012508539937 * max(0.0, Q.LHA - 0.312727471086) / 0.015334441987817575   # -0.2%  LHA > 0.3127
        + 0.0017323179319618416 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7) / 0.00020927924246530186   # +0.2%  log_sum_pt < 6.701 and z_7 < 0.0494
        - 0.000642016816421821 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.pt_6 - 29.90625) / 71.98917715709486   # -0.1%  sum_pt > 988.4 and pt_6 > 29.91
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 40.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.992876279414304 * (0.24610136354774592
        - 0.07623906751451869 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -7.6%  lam1 < 0.008376
        - 0.06834415003546797 * max(0.0, Q.girth2 - 0.007520088344) / 0.002470136248856422   # -6.8%  girth2 > 0.00752
        - 0.06742235224565044 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -6.7%  girth < 0.08724
        - 0.06725136745225106 * max(0.0, Q.girth2 - 0.0016538364) / 0.005220304346757956   # -6.7%  girth2 > 0.001654
        - 0.06589884099939458 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481179386   # -6.6%  mass_over_sum_pt > 0.108
        - 0.05583572762950641 * max(0.0, 0.005590288644 - Q.width) / 0.0022293108396701207   # -5.6%  width < 0.00559
        + 0.04139248330331519 * max(0.0, Q.mass_over_sum_pt - 0.084751611895) / 0.009561895248528105   # +4.1%  mass_over_sum_pt > 0.08475
        - 0.041115923400160174 * max(0.0, 0.15984864831 - Q.max_dr) / 0.05771655139763449   # -4.1%  max_dr < 0.1598
        + 0.034874107101640406 * max(0.0, Q.mass_over_sum_pt - 0.072690732432) / 0.013441384325289518   # +3.5%  mass_over_sum_pt > 0.07269
        - 0.0332821452764362 * max(0.0, 0.067292226106 - Q.C2) / 0.04162086232651303   # -3.3%  C2 < 0.06729
        + 0.03178749460278363 * max(0.0, 0.15984864831 - Q.max_dr) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1) / 0.030025629725308974   # +3.2%  max_dr < 0.1598 and z_dr_0p05_0p1 < 0.6748
        - 0.030833219906128792 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.00829854327583745   # -3.1%  mass_over_sum_pt > 0.09041
        - 0.02961451883301828 * max(0.0, Q.girth2 - 0.004372139461) / 0.0036388049448567565   # -3.0%  girth2 > 0.004372
        + 0.027016606761608004 * max(0.0, 0.037760993714 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.6875) / 4.700370102048804   # +2.7%  centroid_offset < 0.03776 and sum_pt > 559.7
        + 0.02285661705809877 * max(0.0, Q.girth2 - 0.008678044751) / 0.002214536732122738   # +2.3%  girth2 > 0.008678
        + 0.022363492630857563 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # +2.2%  mass < 29.64
        + 0.02147909879057758 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +2.1%  e2 < 0.03847
        - 0.020684835646224867 * max(0.0, Q.e2 - 0.016554418951) / 0.015965081955307547   # -2.1%  e2 > 0.01655
        + 0.018803495068037267 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # +1.9%  e2 < 0.02455
        + 0.01594921415988006 * max(0.0, 0.197968879342 - Q.max_dr) * max(0.0, Q.z_dr_0p05_0p1 - 0.048758227378) / 0.017273610725895706   # +1.6%  max_dr < 0.198 and z_dr_0p05_0p1 > 0.04876
        - 0.015716440172100728 * max(0.0, Q.girth2 - 0.013238675334) / 0.0014426835012777079   # -1.6%  girth2 > 0.01324
        - 0.014430655227678515 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2) / 0.0004358510611765558   # -1.4%  lam1 < 0.008376 and D2 < 1.123
        - 0.01380110586358124 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow) / 6.147530334000904   # -1.4%  pt_7 < 48.72 and planar_flow < 0.6948
        + 0.013171362476316773 * max(0.0, 0.293190627853 - Q.LHA) / 0.07167605978374647   # +1.3%  LHA < 0.2932
        + 0.01159425018697739 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.006263581728545718   # +1.2%  width < 0.00559 and n_dr_0p1_0p2 < 3
        + 0.011592329196836736 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) / 0.07219748739654334   # +1.2%  mass_over_sum_pt < 0.1309
        - 0.011442885786593965 * max(0.0, 0.02076709205 - Q.centroid_offset) / 0.008334290847489842   # -1.1%  centroid_offset < 0.02077
        - 0.01017113519630382 * max(0.0, Q.girth2 - 0.007520088344) * max(0.0, Q.log_sum_pt - 6.19222188581) / 0.00036011410676096295   # -1.0%  girth2 > 0.00752 and log_sum_pt > 6.192
        - 0.009672872824008251 * max(0.0, 0.000561123155 - Q.width) / 9.465572315649017e-05   # -1.0%  width < 0.0005611
        + 0.009307164824926398 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852) / 7.473745705528229e-05   # +0.9%  girth2 > 0.004372 and eccentricity > 0.9458
        - 0.009235082225025623 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.log_sum_pt - 6.327378592257) / 0.00011178296636746688   # -0.9%  girth2_top2 < 0.001057 and log_sum_pt > 6.327
        - 0.008915041056870621 * max(0.0, 0.04081947431 - Q.girth) / 0.00917631455554571   # -0.9%  girth < 0.04082
        - 0.008269564694185006 * max(0.0, 0.005590288644 - Q.width) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.0021347966742966463   # -0.8%  width < 0.00559 and n_dr_0p2_0p4 < 1
        + 0.008197296978698488 * max(0.0, 0.002464291268 - Q.lam1) / 0.0007496296550128049   # +0.8%  lam1 < 0.002464
        - 0.00727847072910444 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -0.7%  mass > 80.4
        + 0.006919782384711622 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2) / 0.0025585111166936624   # +0.7%  e2 < 0.05028 and D2 < 1.123
        - 0.006234412066756829 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361) / 0.007346376310837921   # -0.6%  mass_over_sum_pt < 0.1309 and z_dr_0p05_0p1 > 0.2919
        - 0.006083304972887843 * max(0.0, 0.293190627853 - Q.LHA) * max(0.0, 16.899120053094 - Q.m012) / 0.9699540565722155   # -0.6%  LHA < 0.2932 and m012 < 16.9
        + 0.005394741880706537 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.000279382570519788   # +0.5%  girth2_top2 < 0.001057 and n_dr_0p2_0p4 < 1
        + 0.0042344786177554 * max(0.0, Q.centroid_offset - 0.031170772021) / 0.0025050185320603085   # +0.4%  centroid_offset > 0.03117
        + 0.0034379012518036332 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875) / 10.120048448944816   # +0.3%  planar_flow < 0.195 and sum_pt > 615.9
        + 0.0032217241050140357 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.122624260187 - Q.D2) / 0.0007488893454296128   # +0.3%  e2 < 0.03847 and D2 < 1.123
        + 0.0029534992274508325 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842) / 0.0001455029340999849   # +0.3%  planar_flow < 0.195 and width > 0.00752
        - 0.0015802508739548367 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5) / 0.042474336856843424   # -0.2%  centroid_offset > 0.03117 and pt_2 > 56.5
        + 0.0015314061731053858 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605) / 4.300822302519368e-05   # +0.2%  centroid_offset < 0.02077 and C2 > 0.02384
        + 0.0014978278498663051 * max(0.0, Q.mass_top5 - 53.607658247923) / 1.9788142552051857   # +0.1%  mass_top5 > 53.61
        - 0.0010702547412228965 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478) / 0.038560975699403234   # -0.1%  mass > 80.4 and eccentricity > 0.9271
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 23.85;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.846014355671315 * (-0.025885704742571992
        + 0.1498539814583799 * max(0.0, 0.005019718802 - Q.width) / 0.0019034243755815378   # +15.0%  width < 0.00502
        - 0.08990390669869548 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width) / 7.976303597876625e-05   # -9.0%  girth < 0.06109 and width < 0.00502
        + 0.0627754627325385 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset) / 3.883081060200359e-05   # +6.3%  girth2 < 0.006679 and centroid_offset < 0.02355
        + 0.053726550340501764 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4) / 0.00019011121076024945   # +5.4%  width < 0.00502 and z_dr_0p2_0p4 < 0.101
        - 0.038145758780289016 * max(0.0, 0.006679471442 - Q.girth2) / 0.0029362398563122583   # -3.8%  girth2 < 0.006679
        - 0.030657370501726382 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.10213245107017851   # -3.1%  mass < 29.64 and centroid_offset < 0.02355
        - 0.025856186246040063 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266) / 8.239263087841892e-05   # -2.6%  girth < 0.06109 and centroid_offset > 0.00679
        - 0.025045986960939005 * max(0.0, 0.061086014472 - Q.girth) / 0.018686846679786144   # -2.5%  girth < 0.06109
        + 0.023593469790941887 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2) / 2.869242750181301e-06   # +2.4%  girth < 0.06109 and lam2 < 0.0001948
        + 0.022693166366916845 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.log_sum_pt - 6.670067010936) / 0.0016069299943093157   # +2.3%  girth < 0.06109 and log_sum_pt > 6.67
        - 0.020822365982934694 * max(0.0, 21.784077072144 - Q.mass) / 4.431023211632268   # -2.1%  mass < 21.78
        - 0.020129715238015666 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.031170772021) / 0.004359226399952505   # -2.0%  mass < 21.78 and centroid_offset > 0.03117
        - 0.019438400936788915 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2) / 7.006679088185514e-06   # -1.9%  LHA < 0.1967 and lam2 < 0.0003061
        - 0.018828435846249537 * max(0.0, 0.196739721581 - Q.LHA) / 0.025065644276078353   # -1.9%  LHA < 0.1967
        + 0.01869800822960962 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width) / 9.585949093597567e-06   # +1.9%  LHA < 0.1967 and width < 0.0005611
        - 0.018625480082876383 * max(0.0, 0.027029510401 - Q.C2) / 0.00887441277984911   # -1.9%  C2 < 0.02703
        + 0.01749763970980794 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266) / 1.0643487980347305e-05   # +1.7%  width < 0.00502 and centroid_offset > 0.00679
        + 0.016603799525677043 * max(0.0, 0.000194798295 - Q.lam2) / 0.00011207599367586609   # +1.7%  lam2 < 0.0001948
        - 0.016589171922997702 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.004372139331 - Q.width) / 2.2864151287573696e-05   # -1.7%  C2 < 0.02703 and width < 0.004372
        + 0.016559068734122193 * max(0.0, 0.177304983139 - Q.max_dr) / 0.07042349604125828   # +1.7%  max_dr < 0.1773
        + 0.016052342184589063 * max(0.0, 0.000964142894 - Q.girth2) / 0.00020522880102919315   # +1.6%  girth2 < 0.0009641
        + 0.015098222436657552 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2) / 9.998101856927162e-06   # +1.5%  max_dr < 0.1773 and lam2 < 0.0001948
        - 0.013956665889549628 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.000194798295 - Q.lam2) / 0.000717852158419489   # -1.4%  mass < 21.78 and lam2 < 0.0001948
        - 0.01264821373104243 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.006978029732767499   # -1.3%  log_sum_pt > 6.701 and z_dr_0p2_0p4 < 0.2055
        - 0.012217586693509745 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003) / 5.841408434868268e-06   # -1.2%  girth2 < 0.006679 and centroid_offset > 0.01838
        - 0.01213692697585709 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.0016538364 - Q.girth2) / 0.04021089812658436   # -1.2%  sum_pt_top5 > 658.1 and girth2 < 0.001654
        + 0.012095537055587396 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.00015145615486565555   # +1.2%  e2 < 0.02455 and centroid_offset < 0.03117
        - 0.011774213089812137 * max(0.0, 0.000823693417 - Q.girth2_top3) / 0.00020583156908661208   # -1.2%  girth2_top3 < 0.0008237
        + 0.011718706654467025 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.000964142901 - Q.width) / 3.6464719955993426e-06   # +1.2%  C2 < 0.02703 and width < 0.0009641
        - 0.011270857400619698 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649) / 9.121997321495253e-07   # -1.1%  width < 0.00502 and mass_over_sum_pt_sq > 0.0001232
        + 0.011247301183187902 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55591824366466   # +1.1%  sum_pt_top5 > 658.1
        + 0.010172605916587071 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow) / 0.0003918103360084937   # +1.0%  girth2 < 0.006679 and planar_flow < 0.4008
        - 0.010153203331331748 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, 0.006096650059 - Q.width) / 3.403705720750153e-05   # -1.0%  C2 < 0.02703 and width < 0.006097
        - 0.009798954020610155 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000964142901 - Q.width) / 0.0016562102048156375   # -1.0%  mass < 15.45 and width < 0.0009641
        - 0.00967805257584273 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 48.71875 - Q.pt_7) / 0.030845495821782958   # -1.0%  width < 0.00502 and pt_7 < 48.72
        - 0.009565929189795935 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.0010110746822510478   # -1.0%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.009362511769962301 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset) / 0.07546093782964779   # +0.9%  mass < 21.78 and centroid_offset < 0.02686
        - 0.009200317170142355 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256) / 1.0102446095345114e-05   # -0.9%  girth < 0.06109 and lam1 > 0.0002759
        - 0.008238769745718168 * max(0.0, 0.000222950415 - Q.girth2_top5) / 2.934670561073984e-05   # -0.8%  girth2_top5 < 0.000223
        - 0.008210249790716698 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2) / 0.0005900482644939888   # -0.8%  log_sum_pt > 6.701 and girth2 < 0.01883
        + 0.00803767392950996 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 9.257203159811 - Q.mass_top5) / 0.01960549395734834   # +0.8%  e2 < 0.01655 and mass_top5 < 9.257
        - 0.007946771120653404 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936) / 0.00021206265198456234   # -0.8%  girth2 < 0.006679 and log_sum_pt > 6.67
        - 0.007390090411963765 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2) / 2.697123900716699e-05   # -0.7%  z_dr_0_0p05 > 0.8477 and lam2 < 0.0005373
        + 0.007272355776971919 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848) / 0.014297048836475   # +0.7%  mass < 21.78 and centroid_offset > 0.01628
        - 0.007131232041662956 * max(0.0, 0.027029510401 - Q.C2) * max(0.0, Q.centroid_offset - 0.016278845848) / 2.8295961573866287e-05   # -0.7%  C2 < 0.02703 and centroid_offset > 0.01628
        + 0.006039277677177769 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7) / 0.7769310447436848   # +0.6%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.005951596138388941 * max(0.0, 0.000319370692 - Q.width) / 4.035776571038976e-05   # +0.6%  width < 0.0003194
        - 0.005876797955501587 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.006679471358 - Q.width) / 0.00016927111692806288   # -0.6%  log_sum_pt > 6.701 and width < 0.006679
        - 0.005202496614253333 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.eccentricity - 0.872657364787) / 0.0002890798476729567   # -0.5%  e2 < 0.02455 and eccentricity > 0.8727
        + 0.004245613903669034 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7) / 0.03926630160825465   # +0.4%  mass < 15.45 and z_7 < 0.05861
        + 0.003891326564185854 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq) / 3.2228282972711254e-06   # +0.4%  log_sum_pt > 6.701 and mass_over_sum_pt_sq < 0.0002368
        + 0.00037367497442604864 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.girth2 - 0.006679471442) / 1.8560286120793675e-08   # +0.0%  LHA < 0.1967 and girth2 > 0.006679
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 34.91;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 34.9057072189751 * (-0.05402042305095521
        + 0.13095141708921704 * max(0.0, 0.006096650059 - Q.width) / 0.002544039033917776   # +13.1%  width < 0.006097
        - 0.05500148653255314 * max(0.0, 53.332374954224 - Q.mass) / 19.36003866872732   # -5.5%  mass < 53.33
        - 0.05274427818673328 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset) / 0.20225388264557495   # -5.3%  mass < 41.38 and centroid_offset < 0.02686
        + 0.05216855503443035 * max(0.0, 0.007520088344 - Q.girth2) / 0.0035449904994304922   # +5.2%  girth2 < 0.00752
        - 0.046401164563396824 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) / 0.02715027320795572   # -4.6%  mass_over_sum_pt < 0.07637
        + 0.045465348834854105 * max(0.0, 41.377904891968 - Q.mass) / 12.537841289638873   # +4.5%  mass < 41.38
        + 0.04538907092781467 * max(0.0, 0.004372139461 - Q.girth2) / 0.001565710312409871   # +4.5%  girth2 < 0.004372
        + 0.03972383945401723 * max(0.0, 0.221586732566 - Q.max_dr) / 0.10604661584056564   # +4.0%  max_dr < 0.2216
        + 0.03805302523826581 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset) / 0.2973625515049245   # +3.8%  mass < 53.33 and centroid_offset < 0.02686
        - 0.03729390444164318 * max(0.0, 0.054649224505 - Q.girth) / 0.015329780861188023   # -3.7%  girth < 0.05465
        + 0.032313004522101986 * max(0.0, 0.020459658932 - Q.e2) / 0.00553192518950076   # +3.2%  e2 < 0.02046
        + 0.0319781536778286 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt) / 5.175435934052333   # +3.2%  mass < 53.33 and log_sum_pt < 6.843
        - 0.028330947169806872 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496) / 2.100561780735258e-05   # -2.8%  width < 0.006097 and centroid_offset > 0.003343
        + 0.026366044104083688 * max(0.0, 0.018377780003 - Q.centroid_offset) / 0.006717135927308933   # +2.6%  centroid_offset < 0.01838
        - 0.023479767661434406 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # -2.3%  mass < 29.64
        - 0.021993318935448165 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) * max(0.0, 0.000759634834 - Q.girth2_top2) / 1.1880498756347003e-05   # -2.2%  mass_over_sum_pt < 0.07637 and girth2_top2 < 0.0007596
        + 0.018366072516025338 * max(0.0, 0.016554418951 - Q.e2) * max(0.0, 0.023554160423 - Q.centroid_offset) / 5.587933061311122e-05   # +1.8%  e2 < 0.01655 and centroid_offset < 0.02355
        + 0.015739849904152725 * max(0.0, 0.0030131544 - Q.mass_over_sum_pt_sq) / 0.0010658781088750734   # +1.6%  mass_over_sum_pt_sq < 0.003013
        - 0.015734681052506947 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7) / 0.11864275984569007   # -1.6%  e2 < 0.02046 and pt_7 < 53.44
        - 0.01570604168893906 * max(0.0, 0.005954149834 - Q.lam1) / 0.0025181540815307126   # -1.6%  lam1 < 0.005954
        + 0.014772249308852925 * max(0.0, 0.000964142894 - Q.girth2) / 0.00020522880102919315   # +1.5%  girth2 < 0.0009641
        - 0.013732149962285337 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # -1.4%  log_sum_pt > 6.378
        + 0.01365996241020428 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01) / 0.0005482140187387444   # +1.4%  e2 < 0.03235 and dr01 < 0.05595
        + 0.013626452282174451 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow) / 1.530012963091921   # +1.4%  mass < 53.33 and planar_flow < 0.3221
        - 0.011553960180620175 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.002127561159 - Q.mean_phi2) / 4.211969275399352e-06   # -1.2%  lam1 < 0.005954 and mean_phi2 < 0.002128
        + 0.010870926821343324 * max(0.0, 0.000759634834 - Q.girth2_top2) / 0.0001900990417826364   # +1.1%  girth2_top2 < 0.0007596
        - 0.010221556794553078 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878) / 0.0002260041308477699   # -1.0%  centroid_offset < 0.01838 and z_4 > 0.04749
        - 0.010209021162604996 * max(0.0, 0.032346998155 - Q.e2) / 0.011718069376880604   # -1.0%  e2 < 0.03235
        - 0.01015841059524619 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.90890302062 - Q.z_top5) / 0.009708087570272587   # -1.0%  max_dr < 0.2216 and z_top5 < 0.9089
        - 0.009286611047113856 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.centroid_offset - 0.010960638421) / 5.4532184170070955e-06   # -0.9%  girth2 < 0.004372 and centroid_offset > 0.01096
        - 0.008979310963083421 * max(0.0, 0.001503553356 - Q.lam1) / 0.0003926183331153187   # -0.9%  lam1 < 0.001504
        + 0.008910818640475139 * max(0.0, Q.lam2 - 0.001130644719) / 0.0003119523112525676   # +0.9%  lam2 > 0.001131
        - 0.008175798621786582 * max(0.0, 0.033604209498 - Q.girth) / 0.006496997164569461   # -0.8%  girth < 0.0336
        - 0.00669984835501823 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1) / 0.00860596641674667   # -0.7%  mass < 53.33 and lam1 < 0.0008722
        - 0.006660889972273581 * max(0.0, 0.007520088344 - Q.girth2) * max(0.0, Q.pt_7 - 20.125) / 0.0516538744292779   # -0.7%  girth2 < 0.00752 and pt_7 > 20.12
        + 0.006273902548577389 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375) / 0.10790638482697934   # +0.6%  centroid_offset < 0.01838 and pt_4 > 47.34
        - 0.005822549228283074 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.z_7 - 0.023207568189) / 3.782001556095723e-06   # -0.6%  girth2_top2 < 0.0007596 and z_7 > 0.02321
        - 0.005684180169044495 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow) / 0.8148647296533372   # -0.6%  mass < 41.38 and planar_flow < 0.3221
        - 0.00550029715973575 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, 0.027807975573 - Q.dr_2) / 2.931591322585266e-05   # -0.6%  lam1 < 0.005954 and dr_2 < 0.02781
        - 0.004724011827549657 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21) / 0.0007426911303479672   # -0.5%  e2 < 0.03235 and tau21 < 0.4466
        + 0.004588755196707179 * max(0.0, 0.000172198326 - Q.width) / 1.4418964712323033e-05   # +0.5%  width < 0.0001722
        + 0.004264445325554825 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.027807975573 - Q.dr_2) / 0.0017949369717670475   # +0.4%  log_sum_pt > 6.378 and dr_2 < 0.02781
        + 0.004013346175347979 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.021648628542 - Q.dr_5) / 0.0008555541335392073   # +0.4%  log_sum_pt > 6.378 and dr_5 < 0.02165
        - 0.003484803203990498 * max(0.0, Q.sum_pt - 840.01953125) / 24.43934432362789   # -0.3%  sum_pt > 840
        - 0.0034110674040870766 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696) / 0.00012511356827127218   # -0.3%  e2 < 0.02046 and eccentricity > 0.9031
        - 0.003292969057524928 * max(0.0, 0.009480684835 - Q.centroid_offset) * max(0.0, 0.002127561159 - Q.mean_phi2) / 2.9084584434850715e-06   # -0.3%  centroid_offset < 0.009481 and mean_phi2 < 0.002128
        + 0.003132285173500873 * max(0.0, 35.5 - Q.pt_5) / 1.4798162827435661   # +0.3%  pt_5 < 35.5
        - 0.003044688470922379 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.021648628542 - Q.dr_5) / 0.00011056517027334096   # -0.3%  girth < 0.05465 and dr_5 < 0.02165
        + 0.0030239160016082662 * max(0.0, Q.girth2 - 0.018827652745) / 0.0007605368842505339   # +0.3%  girth2 > 0.01883
        + 0.0030187279171596786 * max(0.0, 0.000759634834 - Q.girth2_top2) * max(0.0, Q.pt_7 - 33.21875) / 0.0009691211713441395   # +0.3%  girth2_top2 < 0.0007596 and pt_7 > 33.22
        + 0.0022151475925246992 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 0.060129364309 - Q.dr_3) / 0.17754133559001112   # +0.2%  sum_pt > 988.4 and dr_3 < 0.06013
        - 0.0016579354003774204 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516) / 5.181368461210433e-06   # -0.2%  width < 0.006097 and C2 > 0.03087
        + 0.001174009735062582 * max(0.0, Q.C2 - 0.051192347892) / 0.004801477093795494   # +0.1%  C2 > 0.05119
        + 0.0009550237595525264 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7) / 0.0006069490438354982   # +0.1%  girth2 > 0.01883 and pt_7 < 29.04
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 14.14;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.13926278675735 * (-0.037071924566488106
        + 0.129194470053617 * Q.e2 / 0.028632153681352513   # +12.9%  e2
        - 0.11885094241923004 * max(0.0, 0.003408388935 - Q.lam2) / 0.0030365450751719814   # -11.9%  lam2 < 0.003408
        + 0.10831638297913873 * max(0.0, Q.mass - 15.454033088684) / 27.25251475411285   # +10.8%  mass > 15.45
        + 0.07741466281092757 * max(0.0, 0.107985668755 - Q.mass_over_sum_pt) / 0.05207765257754601   # +7.7%  mass_over_sum_pt < 0.108
        - 0.06963049501037064 * max(0.0, 0.004183811014 - Q.lam1) / 0.0015130774487597748   # -7.0%  lam1 < 0.004184
        + 0.06173167524068897 * max(0.0, Q.girth2 - 0.008678044751) / 0.002214536732122738   # +6.2%  girth2 > 0.008678
        - 0.054919261298678704 * max(0.0, Q.log_sum_pt - 6.701242202626) / 0.035238726665107536   # -5.5%  log_sum_pt > 6.701
        + 0.03709599908171515 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq) / 0.001485439179640537   # +3.7%  mass_over_sum_pt_sq < 0.003905
        + 0.03172641854883442 * max(0.0, 0.004007841607 - Q.girth2_top2) / 0.0019072108050221064   # +3.2%  girth2_top2 < 0.004008
        - 0.03064071218990178 * max(0.0, 0.0016538364 - Q.girth2) / 0.00042890665324629343   # -3.1%  girth2 < 0.001654
        - 0.029394300901483927 * max(0.0, 0.391541349888 - Q.tau21) / 0.1379333956742398   # -2.9%  tau21 < 0.3915
        - 0.02631883602691344 * max(0.0, Q.lam1 - 0.008375572068) / 0.0018409335087098556   # -2.6%  lam1 > 0.008376
        + 0.02525510049343733 * max(0.0, Q.sum_pt - 813.415625) / 31.70376994639103   # +2.5%  sum_pt > 813.4
        + 0.024611199092315912 * max(0.0, 0.002270363079 - Q.girth2_top5) / 0.0007634099414501585   # +2.5%  girth2_top5 < 0.00227
        + 0.02447121672032952 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.0022074526402598485   # +2.4%  eccentricity > 0.9031 and z_dr_0p2_0p4 < 0.05644
        - 0.024224560148346768 * max(0.0, Q.centroid_offset - 0.00231612516) / 0.014966942305045673   # -2.4%  centroid_offset > 0.002316
        - 0.022158014585260743 * max(0.0, Q.mass - 53.332374954224) / 6.348425380208738   # -2.2%  mass > 53.33
        + 0.021261504545504555 * max(0.0, 3.0 - Q.n_dr_0p05_0p1) / 1.6691546218487394   # +2.1%  n_dr_0p05_0p1 < 3
        + 0.017601218541898004 * max(0.0, Q.lam2 - 0.000194798295) / 0.00044615149470896454   # +1.8%  lam2 > 0.0001948
        - 0.016662716221220022 * max(0.0, Q.LHA - 0.346713497427) / 0.008898883584779705   # -1.7%  LHA > 0.3467
        - 0.013077806818306002 * max(0.0, Q.LHA - 0.303313749495) / 0.017943093379202503   # -1.3%  LHA > 0.3033
        - 0.012904103675795657 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125) / 0.014000463321047197   # -1.3%  lam1 < 0.004184 and sum_pt > 988.4
        - 0.006957564857961736 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.0016231978158967296   # -0.7%  lam2 > 0.0001948 and n_pt_above_50 < 8
        + 0.00544885148039071 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626) / 9.541953566382599e-05   # +0.5%  lam1 < 0.004184 and log_sum_pt > 6.701
        - 0.0052665942783167215 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2) / 0.0003715223898038417   # -0.5%  lam2 > 0.0001948 and D2 < 2.055
        + 0.004865391979415953 * max(0.0, Q.centroid_offset - 0.037760993714) / 0.001689371734514427   # +0.5%  centroid_offset > 0.03776
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 35.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.026387440447834 * (0.004740872006825525
        + 0.12272562512817205 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # +12.3%  width < 0.008678
        + 0.08275264687484081 * max(0.0, 0.006679471442 - Q.girth2) / 0.0029362398563122583   # +8.3%  girth2 < 0.006679
        - 0.07624929807671463 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -7.6%  girth < 0.08724
        + 0.07522283781896422 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # +7.5%  girth2 < 0.01324
        - 0.06616152759560286 * max(0.0, 0.006390124748 - Q.e2_sq) / 0.0029535077959469682   # -6.6%  e2_sq < 0.00639
        + 0.0656564647044402 * max(0.0, 0.049903668404 - Q.centroid_offset) / 0.03352573038002464   # +6.6%  centroid_offset < 0.0499
        - 0.055569772729822775 * max(0.0, Q.girth - 0.076081777364) / 0.010590329529546494   # -5.6%  girth > 0.07608
        - 0.04597472948474591 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -4.6%  lam1 < 0.008376
        + 0.040414438707693366 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.026536230156759166   # +4.0%  girth > 0.07608 and n_pt_above_50 < 7
        - 0.03719660019274276 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.007589401193867531   # -3.7%  centroid_offset < 0.0499 and log_sum_pt < 6.804
        - 0.03173726485532471 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059) / 0.00026898808711045845   # -3.2%  planar_flow < 0.2534 and width > 0.006097
        - 0.026688889819471786 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.014208318908070106   # -2.7%  girth > 0.1019 and n_pt_above_50 < 7
        + 0.02546974516192291 * max(0.0, 0.253403707141 - Q.planar_flow) / 0.10951502727440561   # +2.5%  planar_flow < 0.2534
        - 0.023720586860975452 * Q.centroid_offset / 0.017184338140317994   # -2.4%  centroid_offset
        - 0.016122755087492387 * max(0.0, Q.centroid_offset - 0.014379521101) / 0.0071112999721223424   # -1.6%  centroid_offset > 0.01438
        - 0.015519359510480933 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7) / 0.5039522750508828   # -1.6%  centroid_offset < 0.0499 and pt_7 < 48.72
        - 0.015375467880140277 * max(0.0, 0.003562611091 - Q.width) / 0.001184249467958887   # -1.5%  width < 0.003563
        - 0.013767577810683862 * max(0.0, Q.girth - 0.101940929517) / 0.005400003353656977   # -1.4%  girth > 0.1019
        - 0.013748004932111991 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 40.040625 - Q.pt_7) / 0.06711753055841933   # -1.4%  girth > 0.07608 and pt_7 < 40.04
        + 0.013382201377965516 * max(0.0, 6.701242202626 - Q.log_sum_pt) / 0.19354909741955528   # +1.3%  log_sum_pt < 6.701
        + 0.012673302705628066 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.007877639949821663   # +1.3%  log_sum_pt < 6.701 and z_dr_0p2_0p4 < 0.05644
        - 0.012614110726331378 * max(0.0, 0.196739721581 - Q.LHA) / 0.025065644276078353   # -1.3%  LHA < 0.1967
        - 0.012074646115006585 * max(0.0, 0.035786485299 - Q.C2) / 0.015048242903037393   # -1.2%  C2 < 0.03579
        + 0.01096053953630917 * max(0.0, 0.221586732566 - Q.max_dr) / 0.10604661584056564   # +1.1%  max_dr < 0.2216
        + 0.009655919386677788 * max(0.0, 21.784077072144 - Q.mass) / 4.431023211632268   # +1.0%  mass < 21.78
        - 0.008354760118111839 * max(0.0, 0.004839980301 - Q.lam1) / 0.0018550881157933209   # -0.8%  lam1 < 0.00484
        - 0.0075441928283186055 * max(0.0, 0.111761856824 - Q.max_dr) / 0.028394648782258777   # -0.8%  max_dr < 0.1118
        + 0.007477030922792841 * max(0.0, 687.4375 - Q.sum_pt_top5) / 131.13431972163866   # +0.7%  sum_pt_top5 < 687.4
        - 0.007365537019390797 * max(0.0, 687.4375 - Q.sum_pt_top5) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 210.09499753807773   # -0.7%  sum_pt_top5 < 687.4 and n_dr_0p2_0p4 < 2
        - 0.007217457068136242 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass) / 2.179941616634416   # -0.7%  planar_flow < 0.2534 and mass < 69.61
        - 0.0067016677201508544 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 0.20502409339 - Q.z_dr_0p1_0p2) / 0.00028995696989441355   # -0.7%  girth > 0.07608 and z_dr_0p1_0p2 < 0.205
        - 0.0063339648941553766 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 0.001570267399 - Q.mean_phi2) / 2.5628126471122562e-05   # -0.6%  centroid_offset < 0.0499 and mean_phi2 < 0.00157
        - 0.006306959876319133 * max(0.0, 0.002151567843 - Q.girth2_top3) / 0.0007789602594300843   # -0.6%  girth2_top3 < 0.002152
        - 0.005903983374185869 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.111955475493 - Q.dr_0) / 0.009151652256226576   # -0.6%  log_sum_pt < 6.701 and dr_0 < 0.112
        - 0.005025106118895835 * max(0.0, 0.02054281719 - Q.girth) / 0.0025923881756567037   # -0.5%  girth < 0.02054
        + 0.0045079061990246855 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 1.679198372364 - Q.D2) / 0.09189025441465269   # +0.5%  planar_flow < 0.2534 and D2 < 1.679
        + 0.004347017190651895 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.girth2 - 0.013238675334) / 0.0001108174499013517   # +0.4%  planar_flow < 0.2534 and girth2 > 0.01324
        + 0.0014801035896036497 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7) / 5.884131049165044e-05   # +0.1%  LHA < 0.1547 and z_7 < 0.02807
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.4131;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.41305589871082116 * (-2.2366662289650185
        + 0.477065214059214 * max(0.0, Q.girth2 - 0.018827652745) / 0.0007605368842505339   # +47.7%  girth2 > 0.01883
        - 0.35389902039571225 * max(0.0, Q.e2 - 0.063441075385) / 0.0017618611738475503   # -35.4%  e2 > 0.06344
        + 0.16903576554507374 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625) / 0.014489759228566367   # +16.9%  girth2 > 0.01883 and pt_7 > 15.55
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 27.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.09400500920246 * (0.08868291294742409
        + 0.14163414768863972 * max(0.0, 0.148408418149 - Q.girth) / 0.09059989578347671   # +14.2%  girth < 0.1484
        - 0.11396979234208963 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # -11.4%  e2 < 0.05028
        + 0.1085500229754628 * max(0.0, 0.016433749775 - Q.lam1) / 0.011201097678927486   # +10.9%  lam1 < 0.01643
        + 0.0661764026714897 * max(0.0, 0.013238675006 - Q.width) / 0.00823612446188605   # +6.6%  width < 0.01324
        + 0.050146932321840705 * max(0.0, 0.037760993714 - Q.centroid_offset) / 0.02226602730794202   # +5.0%  centroid_offset < 0.03776
        - 0.042003898246471084 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset) / 7.463904537181694e-06   # -4.2%  lam2 < 0.0003061 and centroid_offset < 0.0499
        - 0.03862158124219834 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556396072   # -3.9%  width < 0.00752
        + 0.038223259212274874 * max(0.0, Q.sum_pt_top5 - 902.40625) / 4.122483656939338   # +3.8%  sum_pt_top5 > 902.4
        - 0.03731975388406383 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset) / 0.00027886707988999124   # -3.7%  lam1 < 0.01643 and centroid_offset < 0.03776
        - 0.035073029663413566 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.01998233930103558   # -3.5%  girth < 0.1484 and log_sum_pt < 6.804
        + 0.032145597217429954 * max(0.0, Q.sum_pt - 988.4078125) / 4.402477808070966   # +3.2%  sum_pt > 988.4
        + 0.02803688681962787 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7) / 927.286113873142   # +2.8%  sum_pt_top5 > 658.1 and pt_7 < 43.5
        - 0.02786576104607245 * max(0.0, Q.LHA - 0.09323897448) / 0.15125019938338352   # -2.8%  LHA > 0.09324
        + 0.025115458752295317 * max(0.0, 0.000306123359 - Q.lam2) / 0.00019890943789793307   # +2.5%  lam2 < 0.0003061
        - 0.02234226657320117 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708193581740943   # -2.2%  girth < 0.1484 and pt_7 < 38.53
        - 0.021497297265676896 * max(0.0, Q.z_top5_slots - 0.930764273368) / 0.0009551768428254974   # -2.1%  z_top5_slots > 0.9308
        + 0.019280524004302784 * max(0.0, 0.006506575659 - Q.lam1) / 0.0028900776904993344   # +1.9%  lam1 < 0.006507
        + 0.018081080800662765 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2) / 5.336336720841763   # +1.8%  sum_pt_top5 > 902.4 and D2 < 3.886
        - 0.01579972995524611 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841) / 0.027815441892245107   # -1.6%  tau21 < 0.501 and max_dr > 0.0156
        - 0.013272877005666626 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6) / 0.19240882891139893   # -1.3%  lam1 < 0.01643 and pt_6 < 56.53
        - 0.013051882436999491 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55591824366466   # -1.3%  sum_pt_top5 > 658.1
        - 0.01196272109133637 * max(0.0, Q.mass - 53.332374954224) / 6.348425380208738   # -1.2%  mass > 53.33
        - 0.01145382422314183 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0) / 2.3317601365545144   # -1.1%  sum_pt > 988.4 and n_pt_above_50 > 6
        - 0.01004711575362043 * max(0.0, 0.028070914944 - Q.z_7) / 0.0013098148516826897   # -1.0%  z_7 < 0.02807
        + 0.00884500485612024 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, Q.n_pt_above_50 - 2.0) / 21.416234423249538   # +0.9%  sum_pt_top5 > 840 and n_pt_above_50 > 2
        - 0.008740381396489078 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.9556856092436975   # -0.9%  sum_pt_top5 > 902.4 and n_pt_above_50 > 6
        + 0.008248442682657037 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2) / 7.606722001858439   # +0.8%  sum_pt > 988.4 and D2 < 3.886
        + 0.007948100291972751 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362) / 0.0016492880690503395   # +0.8%  e2 < 0.05028 and pt_dispersion > 0.3968
        - 0.006316973317767699 * max(0.0, 31.90625 - Q.pt_6) / 1.8268536428243172   # -0.6%  pt_6 < 31.91
        - 0.005849288005924118 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189) / 0.2899084037160227   # -0.6%  sum_pt_top5 > 658.1 and z_7 > 0.02321
        - 0.005820164527497604 * max(0.0, Q.C2 - 0.067292226106) / 0.002854902335729804   # -0.6%  C2 > 0.06729
        - 0.004701893202143012 * max(0.0, 25.578125 - Q.pt_7) / 1.3273888893448005   # -0.5%  pt_7 < 25.58
        - 0.0009402579383758722 * max(0.0, Q.girth - 0.101940929517) * max(0.0, Q.pt_4 - 81.375) / 0.0010189019585713386   # -0.1%  girth > 0.1019 and pt_4 > 81.38
        - 0.0009176505878281989 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.362731824815 - Q.tau32) / 0.17817589723224078   # -0.1%  sum_pt_top5 > 658.1 and tau32 < 0.3627
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 50.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 50.88819110116185 * (-0.018312582211978016
        - 0.08946939776190106 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -8.9%  girth < 0.08724
        + 0.08852182113019999 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # +8.9%  girth2 < 0.01324
        + 0.08350169200022577 * max(0.0, 0.04447356835 - Q.e2) / 0.020193337934351147   # +8.4%  e2 < 0.04447
        - 0.06541334898883308 * max(0.0, 0.006096650059 - Q.width) / 0.002544039033917776   # -6.5%  width < 0.006097
        + 0.05433177382586908 * max(0.0, Q.lam1 - 0.004183811014) / 0.0032456267321051657   # +5.4%  lam1 > 0.004184
        - 0.05266108647937854 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556396072   # -5.3%  width < 0.00752
        + 0.04989402368512714 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # +5.0%  width < 0.008678
        - 0.032789527608958285 * max(0.0, 0.067292226106 - Q.C2) / 0.04162086232651303   # -3.3%  C2 < 0.06729
        + 0.02989962323787244 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.009470677792468518   # +3.0%  width < 0.00752 and n_dr_0p1_0p2 < 3
        - 0.02958872917633704 * max(0.0, 69.611351776123 - Q.mass) / 31.697292539607695   # -3.0%  mass < 69.61
        - 0.026572819016534438 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) / 0.004299658381541595   # -2.7%  mass_over_sum_pt_sq < 0.008175
        + 0.026339909332219937 * max(0.0, 0.017162483186 - Q.e2_sq) / 0.01203539920865141   # +2.6%  e2_sq < 0.01716
        - 0.025775498948623424 * max(0.0, Q.lam1 - 0.005954149834) / 0.0024803645448479386   # -2.6%  lam1 > 0.005954
        - 0.024045951238132107 * max(0.0, 0.080507021025 - Q.max_dr) / 0.01536430929683225   # -2.4%  max_dr < 0.08051
        - 0.02404357938334256 * max(0.0, 0.041109715588 - Q.e2) / 0.01758452927609984   # -2.4%  e2 < 0.04111
        - 0.02395932710988583 * max(0.0, Q.lam1 - 0.004183811014) * max(0.0, Q.D2 - 0.415245993435) / 0.0015391654095455141   # -2.4%  lam1 > 0.004184 and D2 > 0.4152
        + 0.02371522923274038 * max(0.0, 86.4 - Q.mass) / 46.89521789262315   # +2.4%  mass < 86.4
        + 0.02248173879520692 * max(0.0, 0.005834489329 - Q.e2_sq) / 0.0025794526343665795   # +2.2%  e2_sq < 0.005834
        + 0.02118621088744422 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +2.1%  e2 < 0.03847
        - 0.019503680022793346 * max(0.0, Q.lam1 - 0.002464291268) / 0.004201698684296263   # -2.0%  lam1 > 0.002464
        - 0.01745289864355233 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq) / 0.007206102265286917   # -1.7%  mass_over_sum_pt_sq < 0.01166
        - 0.016962011458340916 * max(0.0, 0.177304983139 - Q.max_dr) / 0.07042349604125828   # -1.7%  max_dr < 0.1773
        - 0.016185665043001576 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.01994141641861341   # -1.6%  girth2 < 0.01324 and n_dr_0p1_0p2 < 3
        - 0.014719116587760588 * max(0.0, 0.303313749495 - Q.LHA) / 0.07849185419645244   # -1.5%  LHA < 0.3033
        + 0.013698193469976536 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) / 0.3724173891308251   # +1.4%  z_dr_0p05_0p1 < 0.5883
        + 0.013437240890747023 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 0.415245993435) / 0.0020624573469585736   # +1.3%  lam1 > 0.002464 and D2 > 0.4152
        - 0.009480200639607706 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.1585445378151262   # -0.9%  n_dr_0p05_0p1 < 5
        + 0.007748887972813539 * max(0.0, Q.lam1 - 0.00543336053) / 0.0026771398012749854   # +0.8%  lam1 > 0.005433
        - 0.007270276728442579 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2) / 0.015914745221096264   # -0.7%  z_dr_0p05_0p1 < 0.5883 and C2 < 0.06729
        - 0.0065189072394101965 * max(0.0, Q.lam1 - 0.012003726523) / 0.0012289213133753257   # -0.7%  lam1 > 0.012
        - 0.005181266068659999 * max(0.0, 0.033604209498 - Q.girth) / 0.006496997164569461   # -0.5%  girth < 0.0336
        - 0.00510121967924759 * max(0.0, Q.lam1 - 0.005954149834) * max(0.0, Q.D2 - 1.679198372364) / 7.354274800946109e-05   # -0.5%  lam1 > 0.005954 and D2 > 1.679
        - 0.00471175481713345 * max(0.0, Q.z_dr_0_0p05 - 0.608073231578) / 0.17600445453352792   # -0.5%  z_dr_0_0p05 > 0.6081
        + 0.004232951498449665 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.8008591717572943   # +0.4%  z_dr_0p05_0p1 < 0.5883 and n_dr_0p1_0p2 < 3
        - 0.004200573015252374 * max(0.0, Q.centroid_offset - 0.012587644117) / 0.0079598317469355   # -0.4%  centroid_offset > 0.01259
        + 0.0041026941182781065 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164) / 5.9931533843830415e-05   # +0.4%  girth2 < 0.01324 and eccentricity > 0.9704
        - 0.004099834644254968 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2) / 0.00035388916079662355   # -0.4%  width < 0.008678 and D2 < 1.002
        - 0.003546445830824388 * max(0.0, Q.centroid_offset - 0.049903668404) / 0.0008064001164412618   # -0.4%  centroid_offset > 0.0499
        + 0.0034051800402722294 * max(0.0, 0.135767506063 - Q.tau21) / 0.016830519023847676   # +0.3%  tau21 < 0.1358
        + 0.003216093986736217 * max(0.0, 0.041109715588 - Q.e2) * max(0.0, 1.002470755577 - Q.D2) / 0.0007328545451439236   # +0.3%  e2 < 0.04111 and D2 < 1.002
        - 0.0029761670960798486 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr) / 0.0010568587168460638   # -0.3%  planar_flow < 0.1115 and max_dr < 0.1598
        - 0.00289312857284049 * max(0.0, 1.122624260187 - Q.D2) / 0.23862964281013382   # -0.3%  D2 < 1.123
        - 0.0027435672134740663 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 0.87567204833 - Q.D2) / 1.4804730524080404e-05   # -0.3%  girth2 < 0.004372 and D2 < 0.8757
        + 0.0024027115495992265 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835) / 0.0002838341565068102   # +0.2%  planar_flow < 0.1115 and centroid_offset > 0.009481
        + 0.0021732683862282303 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr) / 1.2809763884266374e-05   # +0.2%  lam1 > 0.005433 and max_dr < 0.1598
        - 0.002017834920308576 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2) / 0.0005071967096445531   # -0.2%  e2 < 0.03847 and D2 < 1.002
        - 0.0019450135121579007 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003) / 0.00015097111917114598   # -0.2%  planar_flow < 0.1115 and centroid_offset > 0.01838
        + 0.0018262928413380682 * max(0.0, Q.lam1 - 0.007330079875) / 0.002073133041421207   # +0.2%  lam1 > 0.00733
        - 0.0013456413313309883 * max(0.0, Q.lam1 - 0.002464291268) * max(0.0, Q.D2 - 1.679198372364) / 0.00020711584637445052   # -0.1%  lam1 > 0.002464 and D2 > 1.679
        - 0.0007099943422552483 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.031170772021) / 5.50154870114671e-06   # -0.1%  girth2 < 0.01324 and centroid_offset > 0.03117
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 40.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 40.64798520689429 * (-0.06339386216033689
        - 0.11678235957258966 * max(0.0, 36.229410171509 - Q.mass) / 10.113929790442333   # -11.7%  mass < 36.23
        + 0.11295301879266367 * max(0.0, 0.013238675006 - Q.width) / 0.00823612446188605   # +11.3%  width < 0.01324
        - 0.07303964541092914 * max(0.0, 0.006679471358 - Q.width) / 0.002936239797737678   # -7.3%  width < 0.006679
        - 0.06697791954567518 * max(0.0, 0.011657374702 - Q.e2_sq) / 0.0072030732891324245   # -6.7%  e2_sq < 0.01166
        + 0.06298503000113256 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # +6.3%  e2 < 0.02455
        - 0.04729563281511414 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt) / 0.022258343201397777   # -4.7%  mass_over_sum_pt < 0.06814
        - 0.046112178789510225 * max(0.0, 0.008168570676 - Q.e2_sq) / 0.004294747328233826   # -4.6%  e2_sq < 0.008169
        - 0.04075307719383672 * max(0.0, 0.012003726523 - Q.lam1) / 0.007316287539015409   # -4.1%  lam1 < 0.012
        + 0.040255380437550475 * max(0.0, 0.006506575659 - Q.lam1) / 0.0028900776904993344   # +4.0%  lam1 < 0.006507
        + 0.03881153259980528 * max(0.0, 0.041109715588 - Q.e2) / 0.01758452927609984   # +3.9%  e2 < 0.04111
        + 0.03691017004916908 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq) / 0.0035288487287669316   # +3.7%  mass_over_sum_pt_sq < 0.007183
        - 0.033265838375655755 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556396072   # -3.3%  width < 0.00752
        + 0.03304823417609518 * max(0.0, Q.girth - 0.033604209498) / 0.03159705075660309   # +3.3%  girth > 0.0336
        - 0.02728115808438218 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -2.7%  lam1 < 0.008376
        + 0.025898901972885555 * max(0.0, 0.018827652745 - Q.girth2) / 0.013142955535859579   # +2.6%  girth2 < 0.01883
        + 0.020343306474785848 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2) / 0.22863861264226853   # +2.0%  z_dr_0p1_0p2 < 0.3285
        + 0.01963494669416067 * max(0.0, 0.063441075385 - Q.e2) / 0.03657078287722135   # +2.0%  e2 < 0.06344
        - 0.014891272288544952 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -1.5%  mass > 80.4
        - 0.014760438634011478 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # -1.5%  e2 < 0.03847
        - 0.011835229104613284 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) / 0.0228131875171655   # -1.2%  z_dr_0p05_0p1 > 0.7509
        - 0.010152055846493733 * max(0.0, 0.002151567843 - Q.girth2_top3) / 0.0007789602594300843   # -1.0%  girth2_top3 < 0.002152
        + 0.009391049147967867 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.010358830282549927   # +0.9%  tau21 < 0.238 and z_dr_0p2_0p4 < 0.2055
        - 0.009293030935698988 * max(0.0, Q.LHA - 0.346713497427) / 0.008898883584779705   # -0.9%  LHA > 0.3467
        - 0.009046946945404154 * max(0.0, 0.004007841607 - Q.girth2_top2) / 0.0019072108050221064   # -0.9%  girth2_top2 < 0.004008
        + 0.008013493070672085 * max(0.0, 0.032346998155 - Q.e2) / 0.011718069376880604   # +0.8%  e2 < 0.03235
        + 0.00799235716194619 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.04262928713838186   # +0.8%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 2
        + 0.0074197585858734256 * max(0.0, 0.007639643088 - Q.girth2_top2) / 0.004543328598712764   # +0.7%  girth2_top2 < 0.00764
        + 0.006826484043167694 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625) / 6.205553603034812   # +0.7%  LHA > 0.1767 and sum_pt_top3 > 353.1
        - 0.006338597919933853 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839) / 3.7602262945140036e-06   # -0.6%  width < 0.00752 and e2 > 0.02455
        - 0.005526955742022131 * max(0.0, 0.74595130682 - Q.D2) / 0.09626873727471312   # -0.6%  D2 < 0.746
        + 0.005397207508293654 * max(0.0, 0.23799610585 - Q.tau21) / 0.05620226106196825   # +0.5%  tau21 < 0.238
        - 0.005054838511330168 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00832969537) / 0.00012177336121647686   # -0.5%  tau21 < 0.238 and girth2_top5 > 0.00833
        - 0.0048750581773367805 * max(0.0, 0.083662731125 - Q.planar_flow) / 0.02146304529912963   # -0.5%  planar_flow < 0.08366
        + 0.004079952657427564 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.08017783625220609   # +0.4%  mass > 80.4 and z_dr_0p2_0p4 < 0.2055
        + 0.004066471397050795 * max(0.0, 0.007639643088 - Q.girth2_top2) * max(0.0, 0.001618889696 - Q.mean_phi) / 2.226903202496936e-05   # +0.4%  girth2_top2 < 0.00764 and mean_phi < 0.001619
        - 0.003840657678714078 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) / 0.012459234137161384   # -0.4%  tau21 < 0.238 and z_dr_0p05_0p1 < 0.5883
        - 0.003653314554137366 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249) / 1.9841456755321017e-05   # -0.4%  width < 0.006097 and log_sum_pt > 6.896
        + 0.0031843914380535326 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # +0.3%  log_sum_pt > 6.896
        - 0.0015655158354208572 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136) / 0.00029995334888293545   # -0.2%  LHA > 0.1767 and z_top5 > 0.865
        + 0.0004465918299439925 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.mean_phi - 0.01746432744) / 3.7788165076256915e-07   # +0.0%  log_sum_pt > 6.896 and mean_phi > 0.01746
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.8936149159663865, 1.0281884453781513, 1.9713068277310923, 0.8082214285714285, 3.2318747899159663, 2.2910531512605044, 1.3228779411764706, 1.371198949579832, 0.6289365546218487, 3.334394117647059, 1.906965231092437, 2.2067072478991596, 0.06439348739495798, 3.9511340336134455, 0.3441191176470588, 0.32994558823529413]
T = [2.636666661469275, 1.7206654149159664, 3.3084610720850836, 2.5111408941701683, 3.5116067194065126]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +32%, n9 +22%, n5 -16%, n1 +15%, n6 +5%, n0 -5% ...
            + 0.3212563479938302 * h[2] / H_AVG[2]
            + 0.21735739194700873 * h[9] / H_AVG[9]
            - 0.16292255374517708 * h[5] / H_AVG[5]
            + 0.15232722336316484 * h[1] / H_AVG[1]
            + 0.05487602089812465 * h[6] / H_AVG[6]
            - 0.052956004132104044 * h[0] / H_AVG[0]
            - 0.038304457920590604 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +49%, n4 -18%, n10 -14%, n6 +10%, n5 +6%, n8 +2% ...
            + 0.492032616623012 * h[9] / H_AVG[9]
            - 0.17608784306821157 * h[4] / H_AVG[4]
            - 0.13853399494183252 * h[10] / H_AVG[10]
            + 0.09610220628229146 * h[6] / H_AVG[6]
            + 0.06241371247098669 * h[5] / H_AVG[5]
            + 0.022844961212744135 * h[8] / H_AVG[8]
            + 0.011984665400921654 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +25%, n6 -12%, n3 -12%, n0 +9%, n7 +9%, n13 +8% ...
            + 0.2501208870022043 * h[11] / H_AVG[11]
            - 0.12495215981402234 * h[6] / H_AVG[6]
            - 0.1221446181414589 * h[3] / H_AVG[3]
            + 0.09284683140305228 * h[0] / H_AVG[0]
            + 0.09066141740381779 * h[7] / H_AVG[7]
            + 0.08397079659860084 * h[13] / H_AVG[13]
            - 0.0780088786333034 * h[14] / H_AVG[14]
            - 0.06856287167036407 * h[15] / H_AVG[15]
            - 0.047524856792819654 * h[8] / H_AVG[8]
            - 0.031494950040564025 * h[9] / H_AVG[9]
            - 0.009711732499792557 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +26%, n6 -20%, n3 -18%, n4 +10%, n13 +9%, n14 +5% ...
            + 0.2559591574920169 * h[7] / H_AVG[7]
            - 0.197551331784237 * h[6] / H_AVG[6]
            - 0.18104302893831203 * h[3] / H_AVG[3]
            + 0.10054800929265371 * h[4] / H_AVG[4]
            + 0.08604759811163058 * h[13] / H_AVG[13]
            + 0.05138886050449637 * h[14] / H_AVG[14]
            + 0.05118133991232731 * h[1] / H_AVG[1]
            - 0.04149500986518897 * h[9] / H_AVG[9]
            - 0.02053010975268325 * h[15] / H_AVG[15]
            + 0.014255554346453782 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -46%, n10 +20%, n5 -16%, n4 +12%, n8 +3%, n3 +1% ...
            - 0.4570979410321735 * h[13] / H_AVG[13]
            + 0.20364238333059206 * h[10] / H_AVG[10]
            - 0.1631057614310316 * h[5] / H_AVG[5]
            + 0.11504259474926969 * h[4] / H_AVG[4]
            + 0.03358166600487851 * h[8] / H_AVG[8]
            + 0.014384822482129063 * h[3] / H_AVG[3]
            - 0.0091686644519579 * h[12] / H_AVG[12]
            + 0.003976166517967762 * h[0] / H_AVG[0]
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
