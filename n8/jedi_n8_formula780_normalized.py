"""JEDI-linear jet tagger, 8 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  16.9%   (on for 88% of jets)
  neuron  9:  11.8%   (on for 69% of jets)
  neuron  6:  10.1%   (on for 37% of jets)
  neuron 10:   9.0%   (on for 74% of jets)
  neuron  7:   8.0%   (on for 55% of jets)
  neuron 11:   6.3%   (on for 78% of jets)
  neuron  5:   6.3%   (on for 69% of jets)
  neuron  2:   5.7%   (on for 82% of jets)
  neuron  3:   5.5%   (on for 23% of jets)
  neuron  4:   5.2%   (on for 80% of jets)
  neuron  1:   3.7%   (on for 61% of jets)
  neuron 14:   3.2%   (on for 29% of jets)
  neuron  0:   3.1%   (on for 44% of jets)
  neuron 15:   2.7%   (on for 27% of jets)
  neuron  8:   2.3%   (on for 50% of jets)
  neuron 12:   0.2%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

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
    # scale S = 25.04;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.037740143153666 * (-0.02529624714973009
        + 0.13936865800921283 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # +13.9%  girth2 < 0.01324
        + 0.13812241321696617 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # +13.8%  width < 0.008678
        - 0.05955287856271741 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) / 0.07219748739654334   # -6.0%  mass_over_sum_pt < 0.1309
        - 0.052784853576709215 * max(0.0, 0.076081777364 - Q.girth) / 0.027967843803613616   # -5.3%  girth < 0.07608
        - 0.047472448204236455 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) / 0.004299658381541595   # -4.7%  mass_over_sum_pt_sq < 0.008175
        - 0.042261898127804925 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -4.2%  lam1 < 0.008376
        - 0.035551553900750696 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -3.6%  girth < 0.08724
        - 0.034464005581893706 * max(0.0, 21.784077072144 - Q.mass) / 4.431023211632268   # -3.4%  mass < 21.78
        + 0.03407849373473537 * max(0.0, 0.018827652745 - Q.girth2) / 0.013142955535859579   # +3.4%  girth2 < 0.01883
        - 0.029755779569013468 * max(0.0, 0.004372139331 - Q.width) / 0.0015657102477202767   # -3.0%  width < 0.004372
        - 0.028326591381418513 * max(0.0, 0.007929074034 - Q.girth2_top3) / 0.004560262122333358   # -2.8%  girth2_top3 < 0.007929
        - 0.027187643330336287 * max(0.0, 64.618731689453 - Q.mass) / 27.572788342921132   # -2.7%  mass < 64.62
        + 0.027076848493035304 * max(0.0, 0.006506575659 - Q.lam1) / 0.0028900776904993344   # +2.7%  lam1 < 0.006507
        - 0.024157999790798333 * max(0.0, 0.000504949057 - Q.lam1) / 8.739618817810877e-05   # -2.4%  lam1 < 0.0005049
        - 0.02250270605338732 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # -2.3%  mass < 29.64
        - 0.02093352851711387 * max(0.0, 0.00543336053 - Q.lam1) / 0.0021941400339846296   # -2.1%  lam1 < 0.005433
        - 0.019670878324088953 * max(0.0, 0.00832969537 - Q.girth2_top5) / 0.004544146438161959   # -2.0%  girth2_top5 < 0.00833
        - 0.019492314627115936 * max(0.0, 56.920347213745 - Q.mass) / 21.784746347219   # -1.9%  mass < 56.92
        + 0.018972963508116374 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) / 0.0544885966340469   # +1.9%  z_dr_0_0p05 > 0.8477
        - 0.016743680525215323 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539) / 0.4331515118014987   # -1.7%  mass < 29.64 and phi_1 > -0.0589
        + 0.016445371647340393 * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.016491452412829836   # +1.6%  centroid_offset < 0.03117
        + 0.014634417978606665 * max(0.0, 0.003952581551 - Q.girth2_top3) / 0.001773144907435322   # +1.5%  girth2_top3 < 0.003953
        - 0.013994025601888754 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7) / 229.29996254831684   # -1.4%  mass < 64.62 and pt_7 < 40.04
        + 0.01377386964935295 * max(0.0, 0.020459658932 - Q.e2) / 0.00553192518950076   # +1.4%  e2 < 0.02046
        + 0.01281889780179432 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153) / 0.0001670441522630915   # +1.3%  girth2 < 0.01883 and eccentricity > 0.9599
        + 0.012588001563058799 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117) / 0.16024311424427862   # +1.3%  mass < 64.62 and centroid_offset > 0.01259
        - 0.01073321160415504 * max(0.0, Q.sum_pt - 901.59375) / 12.403615355829832   # -1.1%  sum_pt > 901.6
        + 0.010643323046627736 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619112969407   # +1.1%  sum_pt_top5 > 687.4
        - 0.006148736816378098 * max(0.0, Q.log_sum_pt - 6.638338705138) / 0.05708668543535571   # -0.6%  log_sum_pt > 6.638
        + 0.005005119258014659 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # +0.5%  log_sum_pt > 6.378
        - 0.00469206731012788 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189) / 0.19805534351472132   # -0.5%  sum_pt_top5 > 687.4 and z_7 > 0.02321
        + 0.004279400956273905 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.072321663733 - Q.dr_4) / 0.002398340732342072   # +0.4%  log_sum_pt > 6.638 and dr_4 < 0.07232
        + 0.0034671593264688698 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7) / 68.64747957984208   # +0.3%  sum_pt > 901.6 and pt_7 < 25.58
        - 0.003068848676258433 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2) / 0.0010081226860790134   # -0.3%  girth2 < 0.01324 and D2 < 1.002
        + 0.0029449944492503355 * max(0.0, 0.148419710734 - Q.planar_flow) / 0.051103617612443746   # +0.3%  planar_flow < 0.1484
        + 0.0028845733604632405 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605) / 0.07430351940037334   # +0.3%  mass < 56.92 and C2 > 0.02384
        - 0.00273954459951202 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.049903668404 - Q.centroid_offset) / 0.0017323178375768763   # -0.3%  planar_flow < 0.1484 and centroid_offset < 0.0499
        - 0.0025868545972448256 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2) / 0.021060647298465133   # -0.3%  mass < 29.64 and D2 < 0.8757
        - 0.0023423017843025077 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, Q.pt_7 - 29.0421875) / 87.66880630219308   # -0.2%  sum_pt > 901.6 and pt_7 > 29.04
        - 0.0023007947994787325 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.111761856824 - Q.max_dr) / 0.00044316723190441983   # -0.2%  planar_flow < 0.1484 and max_dr < 0.1118
        - 0.002051266703021733 * max(0.0, 0.007929074034 - Q.girth2_top3) * max(0.0, 35.28125 - Q.pt_6) / 0.01647839305985912   # -0.2%  girth2_top3 < 0.007929 and pt_6 < 35.28
        - 0.001943529867820521 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2) / 7.962190634589057e-05   # -0.2%  lam1 < 0.006507 and D2 < 0.8757
        - 0.0017347943000488486 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 380.5875 - Q.sum_pt_top2) / 3.055978453307198   # -0.2%  planar_flow < 0.1484 and sum_pt_top2 < 380.6
        + 0.0013261326701783913 * max(0.0, Q.z_dr_0p05_0p1 - 0.846033477783) / 0.01076886660132202   # +0.1%  z_dr_0p05_0p1 > 0.846
        - 0.0012214514912726993 * max(0.0, 0.004372139331 - Q.width) * max(0.0, 0.74595130682 - Q.D2) / 7.193908880769565e-06   # -0.1%  width < 0.004372 and D2 < 0.746
        + 0.0011924812857927623 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.phi_0 - 0.021438598633) / 6.0453506881880455e-05   # +0.1%  girth2 < 0.01883 and phi_0 > 0.02144
        - 0.0011368593102398688 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 0.038438041256 - Q.dr_2) / 0.2822418354616529   # -0.1%  sum_pt > 901.6 and dr_2 < 0.03844
        - 0.0008580706148583819 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.014379521101) / 3.0345989167683132e-05   # -0.1%  girth2 < 0.01324 and centroid_offset > 0.01438
        + 0.000779529680039348 * max(0.0, 0.012569162668 - Q.planar_flow) / 0.00047372042986104206   # +0.1%  planar_flow < 0.01257
        - 0.0007006301885855053 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 28.78966323496) / 0.006400312856178311   # -0.1%  girth2 < 0.01883 and mass_top2 > 28.79
        - 0.00048560202687738776 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.027807975573 - Q.dr_2) / 4.877430715476093e-05   # -0.0%  planar_flow < 0.1484 and dr_2 < 0.02781
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.048439272991018 * (0.037748119676063945
        + 0.13756512839559384 * max(0.0, 0.008168570676 - Q.e2_sq) / 0.004294747328233826   # +13.8%  e2_sq < 0.008169
        - 0.11554357786685739 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # -11.6%  width < 0.008678
        - 0.057552761341472976 * max(0.0, 0.055577157257 - Q.z_7) / 0.010554877533209049   # -5.8%  z_7 < 0.05558
        + 0.05631553647569725 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # +5.6%  log_sum_pt > 6.378
        - 0.049706346116093 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 0.000537286005 - Q.lam2) / 1.9807530097002534e-06   # -5.0%  lam1 < 0.008376 and lam2 < 0.0005373
        - 0.04494091090216072 * max(0.0, 0.346713497427 - Q.LHA) / 0.11284739233399547   # -4.5%  LHA < 0.3467
        + 0.04309763946061512 * max(0.0, 0.076081777364 - Q.girth) / 0.027967843803613616   # +4.3%  girth < 0.07608
        + 0.0377932818178544 * max(0.0, Q.log_sum_pt - 6.572937922293) / 0.08631512213470326   # +3.8%  log_sum_pt > 6.573
        + 0.0374771559852863 * max(0.0, Q.pt_7 - 34.53125) / 4.473662132352941   # +3.7%  pt_7 > 34.53
        + 0.03303385216545819 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2) / 9.286869338661969e-05   # +3.3%  log_sum_pt > 6.573 and lam2 < 0.001131
        - 0.03191423296151495 * max(0.0, 53.332374954224 - Q.mass) / 19.36003866872732   # -3.2%  mass < 53.33
        + 0.031388819691036914 * max(0.0, Q.mass_over_sum_pt - 0.054892207095) / 0.022123336411374987   # +3.1%  mass_over_sum_pt > 0.05489
        - 0.02733658992744185 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.0029084479897083337   # -2.7%  log_sum_pt > 6.378 and centroid_offset < 0.02355
        - 0.023325393037270218 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow) / 5.359939441280181e-05   # -2.3%  e2_sq < 0.008169 and planar_flow < 0.08366
        + 0.02169908029399542 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925) / 3.5352354845618814e-05   # +2.2%  e2 < 0.03556 and eccentricity > 0.9782
        - 0.02051126835256463 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass) / 238.71814489127775   # -2.1%  pt_7 > 34.53 and mass < 91.19
        - 0.020308791693679426 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006756161242 - Q.girth2_top3) / 0.0004453888235688722   # -2.0%  log_sum_pt > 6.573 and girth2_top3 < 0.006756
        + 0.02013646791262012 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr) / 0.021256915339589413   # +2.0%  log_sum_pt > 6.378 and max_dr < 0.198
        + 0.019405625481753456 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.007929074034 - Q.girth2_top3) / 6.301690249232185e-05   # +1.9%  z_7 < 0.05558 and girth2_top3 < 0.007929
        - 0.018759914572040355 * max(0.0, 0.042322802544 - Q.C2) / 0.020126650964483378   # -1.9%  C2 < 0.04232
        - 0.015391037448013472 * max(0.0, 0.043044721986 - Q.z_7) / 0.004926447639348283   # -1.5%  z_7 < 0.04304
        - 0.01520762089723912 * max(0.0, Q.e2 - 0.032346998155) / 0.00800322490328291   # -1.5%  e2 > 0.03235
        + 0.011937503024095462 * max(0.0, 0.197783735394 - Q.tau21) / 0.038939428447455314   # +1.2%  tau21 < 0.1978
        - 0.011401346996851606 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow) / 0.0013308621908643308   # -1.1%  LHA < 0.3467 and planar_flow < 0.08366
        + 0.010684667358473425 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.003408388935 - Q.lam2) / 3.4475134450127075e-05   # +1.1%  z_7 < 0.05558 and lam2 < 0.003408
        - 0.007856654891732167 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -0.8%  lam1 < 0.008376
        + 0.007753322118041694 * max(0.0, 0.083662731125 - Q.planar_flow) / 0.02146304529912963   # +0.8%  planar_flow < 0.08366
        + 0.006653876739000373 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.290267042816 - Q.max_dr) / 0.016517766492494906   # +0.7%  log_sum_pt > 6.573 and max_dr < 0.2903
        - 0.006411866978944334 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.z_7 - 0.039784489456) / 5.6818925513825684e-05   # -0.6%  lam1 < 0.008376 and z_7 > 0.03978
        - 0.0063216111331707 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 50.25 - Q.pt_6) / 0.37991528419887   # -0.6%  tau21 < 0.1978 and pt_6 < 50.25
        - 0.006159126636158378 * max(0.0, 0.046566883102 - Q.max_dr) / 0.005138021564075457   # -0.6%  max_dr < 0.04657
        + 0.005917335973876453 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205) / 7.4336756001165145e-06   # +0.6%  lam1 < 0.008376 and centroid_offset > 0.02077
        + 0.0058391311235379655 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr) / 0.08998780021469473   # +0.6%  pt_7 > 34.53 and max_dr < 0.08051
        - 0.005191550794647588 * max(0.0, 0.035560912266 - Q.e2) / 0.01371972344513943   # -0.5%  e2 < 0.03556
        - 0.00494125128299635 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 0.083662731125 - Q.planar_flow) / 5.689235544097237e-05   # -0.5%  width < 0.008678 and planar_flow < 0.08366
        - 0.0035325854101610718 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.00138715893483794   # -0.4%  lam1 < 0.008376 and n_pt_above_50 > 6
        + 0.0029263847006602037 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.042322802544 - Q.C2) / 0.00023050093654604115   # +0.3%  z_7 < 0.05558 and C2 < 0.04232
        - 0.0027297883417486354 * max(0.0, Q.pt_7 - 53.4375) / 0.3316342962184874   # -0.3%  pt_7 > 53.44
        + 0.0022194759836958664 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.90085350275 - Q.z_dr_0_0p05) / 0.04922609506557952   # +0.2%  log_sum_pt > 6.378 and z_dr_0_0p05 < 0.9009
        - 0.002159572007627546 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 52.90625 - Q.pt_6) / 13.557608088235295   # -0.2%  pt_7 > 34.53 and pt_6 < 52.91
        - 0.0019283526021602127 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 1.232133567333 - Q.D2) / 0.018756224107632988   # -0.2%  log_sum_pt > 6.573 and D2 < 1.232
        + 0.0017985880578774735 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0013761153368809544   # +0.2%  e2_sq < 0.008169 and n_pt_above_50 > 6
        + 0.0015862498717355278 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, Q.n_pt_above_50 - 7.0) / 0.02085157657557203   # +0.2%  log_sum_pt > 6.378 and n_pt_above_50 > 7
        - 0.0015399965013757108 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.n_pt_above_50 - 7.0) / 0.008178701618214594   # -0.2%  log_sum_pt > 6.573 and n_pt_above_50 > 7
        + 0.001427830292387642 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.centroid_offset - 0.014379521101) / 0.023463745833780993   # +0.1%  pt_7 > 34.53 and centroid_offset > 0.01438
        + 0.001373718378840506 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.033200121667 - Q.planar_flow) / 0.000428276800320312   # +0.1%  log_sum_pt > 6.573 and planar_flow < 0.0332
        - 0.0008296576464035926 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, Q.dr01 - 0.055953954317) / 0.0001916117361316086   # -0.1%  z_7 < 0.05558 and dr01 > 0.05595
        + 0.0004675223575403053 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 32.617988451746 - Q.mass_top3) / 7.427128319785132   # +0.0%  pt_7 > 53.44 and mass_top3 < 32.62
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 11.28;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.28369966050466 * (0.260427907209186
        - 0.12329928034574568 * max(0.0, Q.LHA - 0.111565049159) / 0.1352185061481073   # -12.3%  LHA > 0.1116
        + 0.09936579253499629 * max(0.0, 69.611351776123 - Q.mass) / 31.697292539607695   # +9.9%  mass < 69.61
        - 0.09860197103445328 * max(0.0, 53.4375 - Q.pt_7) / 19.120943117285975   # -9.9%  pt_7 < 53.44
        + 0.09459591469597864 * max(0.0, 788.4484375 - Q.sum_pt) / 112.15755791313566   # +9.5%  sum_pt < 788.4
        - 0.08105758872929168 * max(0.0, 36.229410171509 - Q.mass) / 10.113929790442333   # -8.1%  mass < 36.23
        + 0.06385539106903505 * max(0.0, 43.5 - Q.pt_7) / 10.288609083672531   # +6.4%  pt_7 < 43.5
        - 0.0585711357766986 * max(0.0, 687.4375 - Q.sum_pt_top5) / 131.13431972163866   # -5.9%  sum_pt_top5 < 687.4
        + 0.04455912979335718 * max(0.0, Q.pt_7 - 30.484375) / 6.808825630252101   # +4.5%  pt_7 > 30.48
        + 0.03354660695034302 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2) / 0.010855764031987139   # +3.4%  mass < 36.23 and lam2 < 0.001131
        - 0.02803545145065786 * max(0.0, 56.53125 - Q.pt_6) / 17.035713070345327   # -2.8%  pt_6 < 56.53
        + 0.027681580760994667 * max(0.0, 0.005954149834 - Q.lam1) / 0.0025181540815307126   # +2.8%  lam1 < 0.005954
        - 0.022607222393318204 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.068101508468 - Q.z_7) / 0.6443998809691351   # -2.3%  mass < 69.61 and z_7 < 0.0681
        + 0.020178632773739533 * max(0.0, 0.003377388461 - Q.lam1) / 0.001133379805322733   # +2.0%  lam1 < 0.003377
        + 0.01876386507766742 * max(0.0, 6.464150123592 - Q.log_sum_pt) / 0.0700995140842757   # +1.9%  log_sum_pt < 6.464
        - 0.017898699831353056 * max(0.0, Q.log_sum_pt - 6.267538488641) / 0.29831431617346454   # -1.8%  log_sum_pt > 6.268
        - 0.016535509951902547 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2) / 0.21065871608184009   # -1.7%  pt_7 > 30.48 and C2 < 0.05119
        - 0.01185636688549696 * max(0.0, 0.694781820497 - Q.planar_flow) / 0.4352329688894335   # -1.2%  planar_flow < 0.6948
        - 0.010979903718440185 * max(0.0, 6.638338705138 - Q.log_sum_pt) / 0.15249355870011386   # -1.1%  log_sum_pt < 6.638
        + 0.010278133668602368 * max(0.0, 6.572937922293 - Q.log_sum_pt) / 0.11632121255387393   # +1.0%  log_sum_pt < 6.573
        - 0.010128861922918031 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025) / 4.493327216036786e-05   # -1.0%  lam1 < 0.005954 and max_dr > 0.08051
        - 0.01002842876334968 * max(0.0, 0.15984864831 - Q.max_dr) / 0.05771655139763449   # -1.0%  max_dr < 0.1598
        - 0.009989526377230868 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7) / 2.998515264059948e-06   # -1.0%  width < 0.0001722 and dr_7 < 0.223
        + 0.009527915906231907 * max(0.0, Q.log_sum_pt - 6.842716632804) / 0.007875918226781813   # +1.0%  log_sum_pt > 6.843
        - 0.008274248372673144 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299) / 0.24796742071553304   # -0.8%  pt_7 > 30.48 and max_dr > 0.09311
        + 0.007979694074236307 * max(0.0, 0.000172198326 - Q.width) / 1.4418964712323033e-05   # +0.8%  width < 0.0001722
        - 0.005161264416464945 * max(0.0, Q.log_sum_pt - 6.804164030582) / 0.012576817653163173   # -0.5%  log_sum_pt > 6.804
        - 0.004904207331447751 * max(0.0, 0.003377388461 - Q.lam1) * max(0.0, 0.006789738266 - Q.centroid_offset) / 2.2605029628645087e-06   # -0.5%  lam1 < 0.003377 and centroid_offset < 0.00679
        - 0.0043657799623030066 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1) / 3.2270176620941267   # -0.4%  pt_7 > 30.48 and z_dr_0p05_0p1 < 0.7509
        - 0.00418719536292948 * max(0.0, 0.028070914944 - Q.z_7) / 0.0013098148516826897   # -0.4%  z_7 < 0.02807
        + 0.003977597037663243 * max(0.0, 8.379955863953 - Q.mass) / 0.5781888805022356   # +0.4%  mass < 8.38
        - 0.0037115396532387615 * max(0.0, 9.1213921e-05 - Q.width) / 4.152702847822947e-06   # -0.4%  width < 9.121e-05
        - 0.0035959803208623227 * max(0.0, 0.016858545121 - Q.z_7) / 0.0002571522772302325   # -0.4%  z_7 < 0.01686
        - 0.0033947100377342615 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.centroid_offset - 0.010960638421) / 0.21169649819279432   # -0.3%  mass < 69.61 and centroid_offset > 0.01096
        - 0.003207586672296728 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.dr_0 - 0.031288303462) / 0.18974364146477182   # -0.3%  pt_7 > 30.48 and dr_0 > 0.03129
        - 0.0031200814927755003 * max(0.0, Q.z_7 - 0.046240320761) / 0.01204992716405512   # -0.3%  z_7 > 0.04624
        - 0.0027681981746651035 * max(0.0, 0.028070914944 - Q.z_7) * max(0.0, 0.00752008842 - Q.width) / 7.682811042932545e-06   # -0.3%  z_7 < 0.02807 and width < 0.00752
        + 0.0026783570311198087 * max(0.0, 15.55390625 - Q.pt_7) / 0.22142612805288048   # +0.3%  pt_7 < 15.55
        - 0.0024613037683558336 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # -0.2%  log_sum_pt > 6.896
        - 0.0022928674698360755 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.centroid_offset - 0.012587644117) / 0.045359124123155334   # -0.2%  pt_7 > 30.48 and centroid_offset > 0.01259
        + 0.0019300620117975386 * max(0.0, 8.379955863953 - Q.mass) * max(0.0, 0.010960638421 - Q.centroid_offset) / 0.002653068652759248   # +0.2%  mass < 8.38 and centroid_offset < 0.01096
        - 0.0019205333982202678 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.max_pair_mass - 13.047927274731) / 31.36860231006198   # -0.2%  mass < 69.61 and max_pair_mass > 13.05
        - 0.0015507949121362121 * max(0.0, Q.sum_pt_top5 - 839.9546875) / 8.60588864380794   # -0.2%  sum_pt_top5 > 840
        + 0.001326333147134982 * max(0.0, 56.53125 - Q.pt_6) * max(0.0, Q.m012 - 32.617988451746) / 48.99228382620799   # +0.1%  pt_6 < 56.53 and m012 > 32.62
        + 0.001135279035910923 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 55.65625 - Q.pt_4) / 57.764381433425655   # +0.1%  mass < 36.23 and pt_4 < 55.66
        - 0.0011336774791924813 * max(0.0, 4.8108519e-05 - Q.girth2) / 8.325084578958415e-07   # -0.1%  girth2 < 4.811e-05
        - 0.0011050823364893159 * max(0.0, Q.log_sum_pt - 6.842716632804) * max(0.0, Q.n_pt_above_50 - 5.0) / 0.006597523263859726   # -0.1%  log_sum_pt > 6.843 and n_pt_above_50 > 5
        + 0.0010433333873536349 * max(0.0, 9.1213921e-05 - Q.width) * max(0.0, 154.25 - Q.pt_2) / 0.00016144262296696009   # +0.1%  width < 9.121e-05 and pt_2 < 154.2
        + 0.0008313827013595766 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, -0.003096654534 - Q.mean_phi) / 0.026696787841309855   # +0.1%  mass < 36.23 and mean_phi < -0.003097
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 23.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.068457004027966 * (0.12522390122650132
        - 0.23695791912596104 * max(0.0, 0.008678044751 - Q.girth2) / 0.004447347389619223   # -23.7%  girth2 < 0.008678
        - 0.1788468077892844 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # -17.9%  girth2 < 0.01324
        + 0.13425741045532857 * max(0.0, 0.04447356835 - Q.e2) / 0.020193337934351147   # +13.4%  e2 < 0.04447
        + 0.04982091648335582 * max(0.0, 0.004372139461 - Q.girth2) / 0.001565710312409871   # +5.0%  girth2 < 0.004372
        + 0.04244995315059429 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 0.06729416374778562   # +4.2%  mass_over_sum_pt > 0.06814 and n_dr_0_0p05 < 5
        - 0.03793601676879135 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) / 0.015391574669539157   # -3.8%  mass_over_sum_pt > 0.06814
        + 0.03576003846998959 * max(0.0, 0.007330079875 - Q.lam1) / 0.00348685261907903   # +3.6%  lam1 < 0.00733
        - 0.023788378930064618 * max(0.0, Q.e2 - 0.028531698044) / 0.009633336184298937   # -2.4%  e2 > 0.02853
        + 0.023622123236623157 * max(0.0, 0.468445876241 - Q.z_dr_0p1_0p2) / 0.35204737477784626   # +2.4%  z_dr_0p1_0p2 < 0.4684
        + 0.018728276570683308 * max(0.0, Q.centroid_offset - 0.014379521101) / 0.0071112999721223424   # +1.9%  centroid_offset > 0.01438
        + 0.016615287424243962 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.001130644719 - Q.lam2) / 0.01016926381848477   # +1.7%  mass > 36.23 and lam2 < 0.001131
        - 0.015572609315595657 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, 0.964120104909 - Q.z_dr_0p05_0p1) / 0.01720990725856626   # -1.6%  e2 < 0.04447 and z_dr_0p05_0p1 < 0.9641
        - 0.015342857395593367 * max(0.0, Q.mass - 36.229410171509) / 14.205281284755415   # -1.5%  mass > 36.23
        + 0.014443609204645182 * max(0.0, 0.006679471358 - Q.width) / 0.002936239797737678   # +1.4%  width < 0.006679
        + 0.014341739530265138 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573) / 4.350681513338082   # +1.4%  mass > 36.23 and eccentricity > 0.6207
        - 0.012389071198899338 * max(0.0, Q.lam1 - 0.012003726523) / 0.0012289213133753257   # -1.2%  lam1 > 0.012
        + 0.010212008692655612 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.00829854327583745   # +1.0%  mass_over_sum_pt > 0.09041
        + 0.009800922047807987 * max(0.0, Q.max_dr - 0.145231109113) / 0.026281229857574792   # +1.0%  max_dr > 0.1452
        + 0.009728572200780635 * max(0.0, Q.width - 0.018827653081) / 0.0007605368512955965   # +1.0%  width > 0.01883
        + 0.009625237323129617 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +1.0%  e2 < 0.03847
        - 0.008827394202984458 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.eccentricity - 0.959856212153) / 0.0001389130551637596   # -0.9%  e2 > 0.02853 and eccentricity > 0.9599
        + 0.006718583039657127 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.959856212153) / 0.00018573440235159712   # +0.7%  LHA > 0.3127 and eccentricity > 0.9599
        - 0.006265766218397667 * max(0.0, Q.mass - 69.611351776123) / 2.4067024290003474   # -0.6%  mass > 69.61
        - 0.00489953770890883 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481179386   # -0.5%  mass_over_sum_pt > 0.108
        - 0.004816733380243104 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.0) / 0.0005523260037536467   # -0.5%  e2 < 0.04447 and z_dr_0p1_0p2 > 0
        - 0.004118639639716969 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr) / 4.2402205608029813e-05   # -0.4%  LHA > 0.3127 and max_dr < 0.1452
        - 0.003794839353739909 * max(0.0, Q.LHA - 0.325582223496) / 0.01245304382641771   # -0.4%  LHA > 0.3256
        + 0.003605325028017911 * max(0.0, Q.lam1 - 0.016433749775) / 0.0006837082011327267   # +0.4%  lam1 > 0.01643
        - 0.003327538526216718 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.04586879935 - Q.dr_3) / 0.00015061771857077786   # -0.3%  max_dr > 0.1452 and dr_3 < 0.04587
        - 0.00319962701963069 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3) / 0.28212928800771087   # -0.3%  LHA > 0.3127 and mass_top3 < 50.35
        - 0.0031522536337598673 * max(0.0, Q.mass_top5 - 53.607658247923) / 1.9788142552051857   # -0.3%  mass_top5 > 53.61
        - 0.0031223295901287706 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7) / 1.0188469566497442e-05   # -0.3%  mass_over_sum_pt > 0.06814 and dr_7 < 0.04215
        + 0.0029043451409660826 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6) / 0.14624935923720359   # +0.3%  mass_over_sum_pt > 0.09041 and pt_6 < 56.53
        + 0.002433964755817117 * max(0.0, -0.012844925793 - Q.mean_eta) / 0.001988793735763939   # +0.2%  mean_eta < -0.01284
        + 0.002269497415612571 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.planar_flow - 0.00804883781) / 0.005147612196694049   # +0.2%  LHA > 0.3127 and planar_flow > 0.008049
        - 0.002226612420330878 * max(0.0, Q.LHA - 0.423592510895) / 0.0013397344097334583   # -0.2%  LHA > 0.4236
        - 0.0021177947297109224 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 68.125 - Q.pt_4) / 0.035301767581221215   # -0.2%  mean_eta < -0.01284 and pt_4 < 68.12
        + 0.001893308500553728 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 0.04881348081 - Q.dr_7) / 3.410777789559402e-06   # +0.2%  mass_over_sum_pt > 0.108 and dr_7 < 0.04881
        - 0.0018124927808431364 * max(0.0, Q.C2 - 0.094821243733) / 0.0008333835716224887   # -0.2%  C2 > 0.09482
        - 0.001654617792786853 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.046481671275 - Q.dr_6) / 0.024066966560401745   # -0.2%  mass > 36.23 and dr_6 < 0.04648
        - 0.001625679310471739 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.518696343899 - Q.tau32) / 0.0026929406238617812   # -0.2%  mass_over_sum_pt > 0.06814 and tau32 < 0.5187
        - 0.0016001343006626764 * max(0.0, Q.width - 0.018827653081) * max(0.0, Q.z_3 - 0.090493038582) / 1.7485663937045227e-05   # -0.2%  width > 0.01883 and z_3 > 0.09049
        + 0.001546144535299555 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr) / 3.112203819684481e-06   # +0.2%  mass_over_sum_pt > 0.09041 and max_dr < 0.1452
        - 0.0015162857832078265 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, -0.039672851562 - Q.eta_0) / 0.00010003119524312723   # -0.2%  centroid_offset > 0.01438 and eta_0 < -0.03967
        - 0.0014981989025581758 * max(0.0, Q.LHA - 0.312727471086) / 0.015334441987817575   # -0.1%  LHA > 0.3127
        + 0.0012685448818313251 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 48.71875 - Q.pt_7) / 0.11670007858005438   # +0.1%  mass_over_sum_pt > 0.09041 and pt_7 < 48.72
        + 0.0012083326440078563 * max(0.0, Q.mass - 69.611351776123) * max(0.0, 0.177304983139 - Q.max_dr) / 0.013287347868038578   # +0.1%  mass > 69.61 and max_dr < 0.1773
        + 0.0011392577735602264 * max(0.0, Q.width - 0.018827653081) * max(0.0, 0.492494773865 - Q.pt_dispersion) / 7.14287855263047e-05   # +0.1%  width > 0.01883 and pt_dispersion < 0.4925
        + 0.0010384951173139786 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.phi_7 - -0.041534423828) / 0.00044161102444959904   # +0.1%  centroid_offset > 0.01438 and phi_7 > -0.04153
        - 0.0009604198676319379 * max(0.0, Q.lam1 - 0.016433749775) * max(0.0, Q.eccentricity - 0.959856212153) / 8.795584178554827e-06   # -0.1%  lam1 > 0.01643 and eccentricity > 0.9599
        - 0.0008780446968101581 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 20.125 - Q.pt_7) / 3.7708902995959526   # -0.1%  mass > 36.23 and pt_7 < 20.12
        + 0.0008730092882875407 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 0.120257140434 - Q.z_4) / 5.864010154098662e-05   # +0.1%  mean_eta < -0.01284 and z_4 < 0.1203
        - 0.0007270572401988549 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.0180163152422735   # -0.1%  e2 > 0.02853 and n_pt_above_50 > 3
        - 0.000671195184114128 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, -0.059631347656 - Q.eta_1) / 0.0002735288647091144   # -0.1%  max_dr > 0.1452 and eta_1 < -0.05963
        + 4.824428172796357e-05 * max(0.0, Q.lam1 - 0.012003726523) * max(0.0, 38.25 - Q.pt_6) / 0.0038504882546104395   # +0.0%  lam1 > 0.012 and pt_6 < 38.25
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 37.34;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 37.33785992015965 * (-0.19021649385502917
        + 0.13751487653202185 * max(0.0, Q.width - 0.001653836415) / 0.005220304336946791   # +13.8%  width > 0.001654
        + 0.13633560866725428 * max(0.0, 0.017162483186 - Q.e2_sq) / 0.01203539920865141   # +13.6%  e2_sq < 0.01716
        - 0.08297085268347125 * max(0.0, Q.width - 0.000319370692) / 0.006166221167135747   # -8.3%  width > 0.0003194
        - 0.060658612860480195 * max(0.0, 69.611351776123 - Q.mass) / 31.697292539607695   # -6.1%  mass < 69.61
        + 0.058001016095164955 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481179386   # +5.8%  mass_over_sum_pt > 0.108
        - 0.050557026554117625 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.00829854327583745   # -5.1%  mass_over_sum_pt > 0.09041
        + 0.04999384164089479 * max(0.0, 0.023780909279 - Q.e2_sq) / 0.018171241308920595   # +5.0%  e2_sq < 0.02378
        + 0.043791890671090614 * max(0.0, Q.e2 - 0.007078157854) / 0.022325491981794805   # +4.4%  e2 > 0.007078
        + 0.0408549591722976 * max(0.0, 0.23799610585 - Q.tau21) / 0.05620226106196825   # +4.1%  tau21 < 0.238
        + 0.03841705803323732 * max(0.0, 56.920347213745 - Q.mass) / 21.784746347219   # +3.8%  mass < 56.92
        - 0.02360318134998456 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2) / 0.015907178094480428   # -2.4%  mass_over_sum_pt > 0.108 and D2 < 3.886
        + 0.022988297300923992 * max(0.0, 0.005011406868 - Q.girth2_top3) / 0.002432330601058206   # +2.3%  girth2_top3 < 0.005011
        + 0.0224463465059287 * max(0.0, Q.max_dr - 0.102758520097) / 0.04505724462361834   # +2.2%  max_dr > 0.1028
        - 0.02240920750569298 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3) / 0.0076573303900181015   # -2.2%  lam2 < 0.0003061 and mass_top3 < 50.35
        - 0.015550822550192257 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1) / 0.002138030313462078   # -1.6%  centroid_offset < 0.01438 and z_dr_0p05_0p1 < 0.6748
        - 0.015282302386659634 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2) / 5.628323650212299e-05   # -1.5%  tau21 < 0.238 and lam2 < 0.001131
        - 0.013947017054717245 * max(0.0, Q.C2 - 0.014943876117) / 0.015904628092890307   # -1.4%  C2 > 0.01494
        - 0.013360246925030368 * max(0.0, 0.001130644719 - Q.lam2) / 0.0009137232342242098   # -1.3%  lam2 < 0.001131
        - 0.013142716383143913 * max(0.0, Q.girth2 - 0.013238675334) / 0.0014426835012777079   # -1.3%  girth2 > 0.01324
        + 0.012750431960607342 * max(0.0, 763.825 - Q.sum_pt) / 96.68366216443064   # +1.3%  sum_pt < 763.8
        - 0.01134072812792944 * max(0.0, 0.000306123359 - Q.lam2) / 0.00019890943789793307   # -1.1%  lam2 < 0.0003061
        - 0.01083627093346031 * max(0.0, Q.C2 - 0.067292226106) / 0.002854902335729804   # -1.1%  C2 > 0.06729
        + 0.009630764338592444 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.067292226106 - Q.C2) / 0.0001625902207446026   # +1.0%  width > 0.0003194 and C2 < 0.06729
        - 0.009509645635212024 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 68.70525244030448   # -1.0%  sum_pt < 763.8 and n_dr_0p2_0p4 < 1
        + 0.008377306279240902 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.00017651898220235482   # +0.8%  width > 0.0003194 and z_dr_0p2_0p4 < 0.05644
        + 0.007962161293214441 * max(0.0, 0.047915700823 - Q.girth) / 0.012161121217552112   # +0.8%  girth < 0.04792
        + 0.006946949126428626 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3) / 2.3095576214583597   # +0.7%  mass < 69.61 and dr_3 < 0.1042
        - 0.006387247365521497 * max(0.0, Q.girth - 0.101940929517) / 0.005400003353656977   # -0.6%  girth > 0.1019
        - 0.00512108421171543 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass) / 0.42797494392630053   # -0.5%  tau21 < 0.238 and mass < 62.55
        - 0.004917707191623819 * max(0.0, 0.009530300104 - Q.girth2_top2) / 0.0061095731516265135   # -0.5%  girth2_top2 < 0.00953
        + 0.004633570802398158 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848) / 1.967884659027567e-05   # +0.5%  girth2_top2 < 0.00953 and centroid_offset > 0.01628
        + 0.004477408597893627 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875) / 0.3246332585626503   # +0.4%  tau21 < 0.238 and pt_7 > 33.22
        - 0.0036219246423844493 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625) / 0.08633888537871398   # -0.4%  max_dr > 0.1028 and pt_7 > 37.16
        - 0.0035053889288416684 * max(0.0, Q.max_dr - 0.197968879342) / 0.012226149933645198   # -0.4%  max_dr > 0.198
        + 0.0034683179961983853 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125) / 0.023902004294371636   # +0.3%  C2 > 0.01494 and pt_7 > 38.53
        - 0.0033344625319658783 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702) / 7.209993946981458e-05   # -0.3%  tau21 < 0.238 and e2_sq > 0.01166
        + 0.003123269690470682 * max(0.0, 0.014379521101 - Q.centroid_offset) / 0.0043064829328538275   # +0.3%  centroid_offset < 0.01438
        - 0.002719477815138098 * max(0.0, Q.e2 - 0.041109715588) / 0.005106967369295564   # -0.3%  e2 > 0.04111
        - 0.0026909284072403057 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7) / 0.004744492327402142   # -0.3%  tau21 < 0.238 and dr_7 < 0.1755
        - 0.002368289093987373 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.169508891404 - Q.dr_6) / 0.0044748691443844725   # -0.2%  tau21 < 0.238 and dr_6 < 0.1695
        + 0.0015057194173134945 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346) / 0.0019149421885925775   # +0.2%  tau21 < 0.238 and planar_flow > 0.04506
        + 0.001073142076176559 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, Q.z_dr_0_0p05 - 0.151540452242) / 26.85069514313372   # +0.1%  sum_pt < 763.8 and z_dr_0_0p05 > 0.1515
        - 0.0010232230441034333 * max(0.0, 430.75 - Q.sum_pt_top5) / 14.052461580882353   # -0.1%  sum_pt_top5 < 430.8
        + 0.0008973342416592065 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.mean_eta2 - 0.004247450386) / 0.0063342535334785025   # +0.1%  mass < 69.61 and mean_eta2 > 0.004247
        + 0.0008382846366688766 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_2 - 73.6875) / 1.3957644344475406   # +0.1%  tau21 < 0.238 and pt_2 > 73.69
        - 0.0008381906868780341 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.mean_phi - 0.004406178184) / 0.00019924215363894784   # -0.1%  tau21 < 0.238 and mean_phi > 0.004406
        - 0.0007752674694887798 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 23.21640625 - Q.pt_7) / 0.02551894097178784   # -0.1%  tau21 < 0.238 and pt_7 < 23.22
        + 0.0007320549923277402 * max(0.0, 0.005011406868 - Q.girth2_top3) * max(0.0, -0.009391680919 - Q.mean_eta) / 2.9101241607797263e-06   # +0.1%  girth2_top3 < 0.005011 and mean_eta < -0.009392
        + 0.0007213213029923187 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq) / 0.0003082102790484629   # +0.1%  mass_over_sum_pt_sq < 0.001102
        - 0.0005647904137232397 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.25 - Q.pt_6) / 1.1457690827873348   # -0.1%  tau21 < 0.238 and pt_6 < 62.25
        + 0.0004623858110550404 * max(0.0, 0.001130644719 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0) / 0.00030456406283498055   # +0.0%  lam2 < 0.001131 and n_dr_0p1_0p2 > 2
        - 0.00042162848226833796 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, Q.n_dr_0p1_0p2 - 6.0) / 0.0018212573202342978   # -0.0%  mass_over_sum_pt > 0.108 and n_dr_0p1_0p2 > 6
        - 0.0002845031612758112 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.mean_eta - 0.02644207105) / 4.05868952700406e-05   # -0.0%  tau21 < 0.238 and mean_eta > 0.02644
        + 0.00018701591054184303 * max(0.0, Q.e2 - 0.063441075385) / 0.0017618611738475503   # +0.0%  e2 > 0.06344
        - 0.0001259225111585081 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.phi_6 - 0.119201660156) / 0.00014358460172630337   # -0.0%  tau21 < 0.238 and phi_6 > 0.1192
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 12.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.267422964643428 * (0.03822616486668559
        + 0.1008227320459548 * max(0.0, 0.071488645583 - Q.z_7) / 0.021404874347378918   # +10.1%  z_7 < 0.07149
        - 0.07869553485923375 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1) / 0.0008217429754174446   # -7.9%  log_sum_pt > 6.573 and lam1 < 0.012
        + 0.07241359739102056 * max(0.0, 0.035560912266 - Q.e2) / 0.01371972344513943   # +7.2%  e2 < 0.03556
        + 0.0487085915515628 * max(0.0, 0.216055863061 - Q.LHA) / 0.03222864900799722   # +4.9%  LHA < 0.2161
        - 0.03983306437412588 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.00407349257511869   # -4.0%  LHA < 0.2161 and log_sum_pt < 6.804
        - 0.037474856142419126 * max(0.0, 0.021588001063 - Q.dr_0) / 0.0036021402838930334   # -3.7%  dr_0 < 0.02159
        + 0.03489957247887816 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2) / 2.1406489096621576e-05   # +3.5%  z_7 < 0.07149 and lam2 < 0.001131
        - 0.034873070744215494 * max(0.0, 0.01426135283 - Q.mean_phi2) / 0.011322003176195528   # -3.5%  mean_phi2 < 0.01426
        - 0.032432490565133455 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.003952581551 - Q.girth2_top3) / 0.2626480126636821   # -3.2%  sum_pt_top2 < 548.2 and girth2_top3 < 0.003953
        + 0.03186601693941496 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset) / 1.2223894287536178e-05   # +3.2%  width < 0.002635 and centroid_offset < 0.02355
        - 0.030944467638147 * max(0.0, Q.sum_pt - 868.509375) / 18.084627120331504   # -3.1%  sum_pt > 868.5
        + 0.030829086191059417 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 40.046952646555 - Q.max_pair_mass) / 0.3492029151333465   # +3.1%  mean_phi2 < 0.01426 and max_pair_mass < 40.05
        - 0.029539546988534098 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.00041496068339934707   # -3.0%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.0294049000315436 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0) / 0.0006616654182381468   # +2.9%  log_sum_pt > 6.573 and dr_0 < 0.02159
        - 0.02940332355117348 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass) / 0.49027896930529   # -2.9%  mass_over_sum_pt < 0.08475 and max_pair_mass < 18.1
        - 0.027851624612425914 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) / 0.03304117548546515   # -2.8%  mass_over_sum_pt < 0.08475
        - 0.025524172826884604 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt) / 0.9285639647784526   # -2.6%  z_7 < 0.07149 and sum_pt < 788.4
        + 0.023850723662487663 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2) / 0.0004246061902383711   # +2.4%  log_sum_pt > 6.573 and girth2_top2 < 0.0063
        + 0.022397745734228994 * max(0.0, 0.002635417778 - Q.width) / 0.0007941346639111337   # +2.2%  width < 0.002635
        - 0.0188376384067838 * max(0.0, 548.196875 - Q.sum_pt_top2) / 195.28859533251793   # -1.9%  sum_pt_top2 < 548.2
        - 0.01881488068212268 * max(0.0, 0.026454043164 - Q.dr_0) / 0.005170180748163715   # -1.9%  dr_0 < 0.02645
        + 0.01771584134480339 * max(0.0, 0.045106684603 - Q.z_dr_0p1_0p2) / 0.023913294162170258   # +1.8%  z_dr_0p1_0p2 < 0.04511
        + 0.016868026864895552 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.274147637091883   # +1.7%  sum_pt_top5 > 752.1
        + 0.015820151411567906 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5) / 0.3059458585195602   # +1.6%  z_7 < 0.0494 and mass_top5 < 62.55
        + 0.015016550340862592 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2) / 5.495977337611781e-07   # +1.5%  e2 < 0.03556 and lam2 < 7.301e-05
        + 0.014430175712721916 * max(0.0, Q.log_sum_pt - 6.572937922293) / 0.08631512213470326   # +1.4%  log_sum_pt > 6.573
        + 0.013039392156938476 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0) / 0.40891493757937986   # +1.3%  sum_pt_top2 < 548.2 and dr_0 < 0.02159
        - 0.01124462508081809 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001503553356 - Q.lam1) / 1.1470520225413071e-05   # -1.1%  z_7 < 0.07149 and lam1 < 0.001504
        + 0.010073096844150023 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.004331280361 - Q.mean_phi2) / 6.112102853578723e-05   # +1.0%  z_7 < 0.07149 and mean_phi2 < 0.004331
        - 0.009974237420761761 * max(0.0, 0.049399692737 - Q.z_7) / 0.007457505366847834   # -1.0%  z_7 < 0.0494
        - 0.009419634977876465 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset) / 1.318014685216707e-05   # -0.9%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.007701734135591276 * max(0.0, 24.578125 - Q.pt_5) / 0.2839763142561712   # +0.8%  pt_5 < 24.58
        - 0.007650371434344591 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, 29.875 - Q.pt_5) / 0.011118081204044762   # -0.8%  z_7 < 0.03243 and pt_5 < 29.88
        + 0.006244662471328264 * max(0.0, 0.028865759995 - Q.z_6) / 0.0008381782190062135   # +0.6%  z_6 < 0.02887
        - 0.0058267665477340965 * max(0.0, 0.154689112391 - Q.LHA) / 0.012614340772605542   # -0.6%  LHA < 0.1547
        + 0.005542538241883733 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset) / 0.13441460519175305   # +0.6%  sum_pt > 868.5 and centroid_offset < 0.01259
        - 0.004701930806598823 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0) / 0.00011031021395682367   # -0.5%  log_sum_pt > 6.896 and dr_0 < 0.04119
        + 0.004700985609780424 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2) / 4.16958841658888e-06   # +0.5%  log_sum_pt > 6.573 and mean_phi2 < 0.0001455
        + 0.004679585389975617 * max(0.0, 0.03243272066 - Q.z_7) / 0.0020610238461808515   # +0.5%  z_7 < 0.03243
        - 0.003484245341522338 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset) / 5.203533472960053e-05   # -0.3%  log_sum_pt > 6.896 and centroid_offset < 0.01838
        + 0.0034467800570746796 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2) / 2.0281431575878195e-06   # +0.3%  log_sum_pt > 6.573 and mean_eta2 < 9.03e-05
        + 0.0020684120679528924 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 33.0265625 - Q.pt_5) / 68.17591363736769   # +0.2%  sum_pt > 868.5 and pt_5 < 33.03
        + 0.0019134010954505946 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # +0.2%  log_sum_pt > 6.896
        - 0.0013523736279538376 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001534583791159281   # -0.1%  LHA < 0.2161 and n_dr_0p2_0p4 > 0
        + 0.0013247368419983723 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, Q.pt_5 - 33.0265625) / 0.010273887151027594   # +0.1%  z_7 < 0.03243 and pt_5 > 33.03
        - 0.001178102531313321 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0007322446740177455   # -0.1%  e2_sq < 0.005285 and n_pt_above_50 > 6
        + 0.0010798954901458321 * max(0.0, 4.8108519e-05 - Q.girth2) / 8.325084578958415e-07   # +0.1%  girth2 < 4.811e-05
        - 0.0009759250570179434 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.mass_top3 - 3.559569591142) / 0.01570045895432424   # -0.1%  LHA < 0.2161 and mass_top3 > 3.56
        - 0.0008255786259459323 * max(0.0, 0.028865759995 - Q.z_6) * max(0.0, Q.phi_0 - 0.014526367188) / 1.370487624323693e-06   # -0.1%  z_6 < 0.02887 and phi_0 > 0.01453
        - 0.0008171098140353145 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.045057236346 - Q.planar_flow) / 2.1587905933501866e-05   # -0.1%  log_sum_pt > 6.896 and planar_flow < 0.04506
        - 0.0006729884759627922 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.centroid_offset - 0.018377780003) / 7.559756148996543e-05   # -0.1%  log_sum_pt > 6.573 and centroid_offset > 0.01838
        - 0.0004777524551389391 * max(0.0, 24.578125 - Q.pt_5) * max(0.0, Q.mean_eta2 - 1.7977892e-05) / 0.0002724447364133756   # -0.0%  pt_5 < 24.58 and mean_eta2 > 1.798e-05
        - 0.000314754308503884 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.dr_4 - 0.030068239644) / 6.609644307456955e-05   # -0.0%  log_sum_pt > 6.896 and dr_4 > 0.03007
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 31.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.347506047266986 * (0.2605466631326038
        - 0.18194098756555918 * max(0.0, 0.008678044751 - Q.girth2) / 0.004447347389619223   # -18.2%  girth2 < 0.008678
        - 0.1722303751538737 * max(0.0, 0.013238675006 - Q.width) / 0.00823612446188605   # -17.2%  width < 0.01324
        - 0.15152723025878942 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) / 0.053262875947277605   # -15.2%  mass_over_sum_pt > 0.008374
        + 0.06999499208263607 * Q.max_dr / 0.1236972946900221   # +7.0%  max_dr
        + 0.06714564439836955 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # +6.7%  e2 < 0.05028
        + 0.04084078361720899 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # +4.1%  lam1 < 0.008376
        - 0.023941827782147564 * max(0.0, Q.centroid_offset - 0.018377780003) / 0.005523694064633852   # -2.4%  centroid_offset > 0.01838
        + 0.023857284919196944 * max(0.0, Q.girth - 0.087236513197) / 0.007853436041615635   # +2.4%  girth > 0.08724
        + 0.023737912584828415 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1) / 11.464259601117858   # +2.4%  mass < 49.67 and z_dr_0p05_0p1 < 0.7509
        - 0.021239460199526897 * max(0.0, 6.701242202626 - Q.log_sum_pt) / 0.19354909741955528   # -2.1%  log_sum_pt < 6.701
        + 0.02066339741105658 * max(0.0, 0.007330079875 - Q.lam1) / 0.00348685261907903   # +2.1%  lam1 < 0.00733
        - 0.020124160245704334 * max(0.0, 0.000537286005 - Q.lam2) / 0.00039098252303812735   # -2.0%  lam2 < 0.0005373
        + 0.01528445387081471 * max(0.0, 6.572937922293 - Q.log_sum_pt) / 0.11632121255387393   # +1.5%  log_sum_pt < 6.573
        + 0.015241955332946802 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2) / 2.6474453560607286e-05   # +1.5%  centroid_offset > 0.008092 and lam2 < 0.003408
        + 0.014624933855946461 * max(0.0, 1.679198372364 - Q.D2) / 0.5633062396133152   # +1.5%  D2 < 1.679
        + 0.011848260573082626 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7) / 1.995212531081465   # +1.2%  log_sum_pt < 6.701 and pt_7 < 45.75
        - 0.010340832384191086 * max(0.0, Q.girth2_top5 - 0.002270363079) / 0.0043533388582812135   # -1.0%  girth2_top5 > 0.00227
        - 0.009943617930238175 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.07644425178616783   # -1.0%  e2 < 0.05028 and n_dr_0p05_0p1 < 4
        + 0.009774919968178057 * max(0.0, Q.centroid_offset - 0.008092360237) / 0.010529146096595013   # +1.0%  centroid_offset > 0.008092
        - 0.008726548304339264 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4) / 17.624128555880276   # -0.9%  D2 < 1.679 and pt_4 < 90.62
        - 0.007614104094267455 * max(0.0, Q.C2 - 0.010539266048) / 0.01887610517800746   # -0.8%  C2 > 0.01054
        + 0.007410482932693137 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787) / 0.0009017529775427706   # +0.7%  LHA > 0.3127 and eccentricity > 0.8727
        + 0.006927906425710006 * max(0.0, Q.girth2_top5 - 0.011482925368) / 0.0015471296538189059   # +0.7%  girth2_top5 > 0.01148
        + 0.006788133908274866 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_eta2 - 0.004247450386) / 0.0005461586200825784   # +0.7%  log_sum_pt < 6.701 and mean_eta2 > 0.004247
        + 0.006490946378489525 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.094821243733 - Q.C2) / 0.0005674917208000974   # +0.6%  centroid_offset > 0.008092 and C2 < 0.09482
        + 0.004907006784173652 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781) / 0.0027716438038630276   # +0.5%  centroid_offset > 0.008092 and planar_flow > 0.008049
        - 0.004765487906054831 * max(0.0, Q.lam2 - 0.003408388935) / 0.00015702993621826726   # -0.5%  lam2 > 0.003408
        + 0.004246249359423386 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179) / 0.0002133523951813162   # +0.4%  e2 < 0.05028 and z_dr_0p1_0p2 > 0.1586
        + 0.004158309847596719 * max(0.0, 6.327378592257 - Q.log_sum_pt) / 0.033169101899090815   # +0.4%  log_sum_pt < 6.327
        + 0.0037899808786628756 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2) / 2.1793621511481152e-05   # +0.4%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        - 0.0037481341159665703 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32) / 0.00581747934750154   # -0.4%  mass_over_sum_pt > 0.008374 and tau32 < 0.5187
        + 0.0036629391562301977 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375) / 0.08096163333875342   # +0.4%  C2 > 0.01054 and pt_7 > 31.86
        + 0.002716411538230323 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7) / 0.00020927924246530186   # +0.3%  log_sum_pt < 6.701 and z_7 < 0.0494
        + 0.002528390816125046 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6) / 0.41268944603937285   # +0.3%  log_sum_pt < 6.701 and pt_6 < 36.81
        - 0.002483085973748105 * max(0.0, Q.lam2 - 0.000537286005) / 0.00038257031407305515   # -0.2%  lam2 > 0.0005373
        + 0.0023862204561786297 * max(0.0, Q.girth2_top5 - 0.002270363079) * max(0.0, Q.z_dr_0p05_0p1 - 0.291944718361) / 0.0007317482583488287   # +0.2%  girth2_top5 > 0.00227 and z_dr_0p05_0p1 > 0.2919
        + 0.0021619764475191874 * max(0.0, 0.003562611155 - Q.girth2) / 0.0011842494965263308   # +0.2%  girth2 < 0.003563
        + 0.001816659078022781 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, Q.pt_2 - 56.5) / 0.11492222841918831   # +0.2%  centroid_offset > 0.01838 and pt_2 > 56.5
        - 0.001667959340313092 * max(0.0, Q.girth2_top5 - 0.011482925368) * max(0.0, Q.mean_eta - 0.0127187056) / 1.0990450553759857e-05   # -0.2%  girth2_top5 > 0.01148 and mean_eta > 0.01272
        - 0.0012438638245667534 * max(0.0, Q.z_7 - 0.06164517166) / 0.004746501812091229   # -0.1%  z_7 > 0.06165
        - 0.0010392407099372597 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.009050007537) / 0.0008878548382281195   # -0.1%  log_sum_pt < 6.701 and mean_phi > 0.00905
        - 0.0008819520345971785 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.578125) / 0.2801529239851139   # -0.1%  log_sum_pt < 6.327 and pt_6 > 27.58
        - 0.0008180526982618686 * max(0.0, 0.007330079875 - Q.lam1) * max(0.0, Q.eta_7 - 0.008316040039) / 3.715610714438028e-05   # -0.1%  lam1 < 0.00733 and eta_7 > 0.008316
        - 0.0007795066338716338 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, Q.mean_eta - 0.02644207105) / 1.2350169949159355e-07   # -0.1%  lam2 < 0.0005373 and mean_eta > 0.02644
        + 0.0005121552446712426 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, 0.071488645583 - Q.z_7) / 9.82602650680265e-05   # +0.1%  log_sum_pt < 6.327 and z_7 < 0.07149
        + 0.0004181208318364173 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_eta - 0.02644207105) / 1.6842597802265111e-06   # +0.0%  width < 0.01324 and mean_eta > 0.02644
        - 0.0003806605035888821 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.001101289818 - Q.mean_eta2) / 1.8903872357933204e-06   # -0.0%  centroid_offset > 0.008092 and mean_eta2 < 0.001101
        + 0.00017891792579811429 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 24.578125 - Q.pt_5) / 0.00025235720347887783   # +0.0%  centroid_offset > 0.01838 and pt_5 < 24.58
        - 0.0001725354434939205 * max(0.0, Q.centroid_offset - 0.049903668404) * max(0.0, Q.n_dr_0p05_0p1 - 6.0) / 0.00012207183924051954   # -0.0%  centroid_offset > 0.0499 and n_dr_0p05_0p1 > 6
        + 0.0001407992753101332 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.037477688199 - Q.z_4) / 2.138012812771994e-06   # +0.0%  log_sum_pt < 6.701 and z_4 < 0.03748
        + 0.000134230997751323 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.mass_top3 - 50.352200171245) / 0.0001449439881708522   # +0.0%  e2_sq < 0.008169 and mass_top3 > 50.35
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 35.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.44133499057814 * (0.27224376838130837
        - 0.09666585101271186 * max(0.0, Q.girth2 - 0.007520088344) / 0.002470136248856422   # -9.7%  girth2 > 0.00752
        - 0.09083711716205117 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -9.1%  girth < 0.08724
        + 0.08990614950996015 * max(0.0, Q.girth2 - 0.008678044751) / 0.002214536732122738   # +9.0%  girth2 > 0.008678
        - 0.08182956252371734 * max(0.0, Q.girth2 - 0.0016538364) / 0.005220304346757956   # -8.2%  girth2 > 0.001654
        - 0.06478202082101026 * max(0.0, 0.005590288644 - Q.width) / 0.0022293108396701207   # -6.5%  width < 0.00559
        - 0.06321295198438233 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.00829854327583745   # -6.3%  mass_over_sum_pt > 0.09041
        - 0.05854167638277403 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -5.9%  lam1 < 0.008376
        + 0.05679517742578276 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +5.7%  e2 < 0.03847
        + 0.05008828135848404 * max(0.0, Q.mass_over_sum_pt - 0.084751611895) / 0.009561895248528105   # +5.0%  mass_over_sum_pt > 0.08475
        - 0.04719788851409346 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # -4.7%  e2 < 0.05028
        - 0.04347918556403419 * max(0.0, Q.girth2 - 0.004372139461) / 0.0036388049448567565   # -4.3%  girth2 > 0.004372
        + 0.03254664442405752 * max(0.0, Q.mass_over_sum_pt - 0.072690732432) / 0.013441384325289518   # +3.3%  mass_over_sum_pt > 0.07269
        + 0.030849561166710088 * max(0.0, Q.girth2 - 0.013238675334) / 0.0014426835012777079   # +3.1%  girth2 > 0.01324
        + 0.023558917602542386 * max(0.0, 0.293190627853 - Q.LHA) / 0.07167605978374647   # +2.4%  LHA < 0.2932
        + 0.01953068614041855 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # +2.0%  e2 < 0.02455
        - 0.015079935705770264 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2) / 0.0004358510611765558   # -1.5%  lam1 < 0.008376 and D2 < 1.123
        + 0.013774160197492584 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2) / 0.0025585111166936624   # +1.4%  e2 < 0.05028 and D2 < 1.123
        + 0.01334907553805541 * max(0.0, 0.001101266364 - Q.e2_sq) / 0.00030800399185922496   # +1.3%  e2_sq < 0.001101
        - 0.012557886386463315 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481179386   # -1.3%  mass_over_sum_pt > 0.108
        - 0.01216518459367916 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow) / 6.147530334000904   # -1.2%  pt_7 < 48.72 and planar_flow < 0.6948
        + 0.01093418176326375 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852) / 7.473745705528229e-05   # +1.1%  girth2 > 0.004372 and eccentricity > 0.9458
        + 0.007882197323284625 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875) / 10.120048448944816   # +0.8%  planar_flow < 0.195 and sum_pt > 615.9
        - 0.0077099773818230435 * max(0.0, 0.04081947431 - Q.girth) / 0.00917631455554571   # -0.8%  girth < 0.04082
        - 0.006130124464681137 * max(0.0, 0.02076709205 - Q.centroid_offset) / 0.008334290847489842   # -0.6%  centroid_offset < 0.02077
        + 0.006076850842894609 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # +0.6%  mass < 29.64
        + 0.005873389321685115 * max(0.0, 48.71875 - Q.pt_7) / 14.746502676109506   # +0.6%  pt_7 < 48.72
        - 0.005689113287454264 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -0.6%  mass > 80.4
        - 0.0055747793017992915 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842) / 0.0001455029340999849   # -0.6%  planar_flow < 0.195 and width > 0.00752
        - 0.004419102570187752 * max(0.0, 0.037760993714 - Q.centroid_offset) / 0.02226602730794202   # -0.4%  centroid_offset < 0.03776
        - 0.00403021326960873 * max(0.0, 0.197968879342 - Q.max_dr) / 0.08649773458639719   # -0.4%  max_dr < 0.198
        - 0.002906988928762252 * max(0.0, 0.195013533663 - Q.planar_flow) / 0.07573236359588242   # -0.3%  planar_flow < 0.195
        - 0.002560922266003015 * max(0.0, 0.000561123155 - Q.width) / 9.465572315649017e-05   # -0.3%  width < 0.0005611
        + 0.001978533776888232 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605) / 4.300822302519368e-05   # +0.2%  centroid_offset < 0.02077 and C2 > 0.02384
        + 0.0019161271006134584 * max(0.0, 0.005884990035 - Q.girth2_top3) / 0.0030201574371678344   # +0.2%  girth2_top3 < 0.005885
        + 0.0016933963768018965 * max(0.0, Q.centroid_offset - 0.031170772021) / 0.0025050185320603085   # +0.2%  centroid_offset > 0.03117
        + 0.0013202169088860644 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.mass_top3 - 23.663861485439) / 0.5450859711047729   # +0.1%  planar_flow < 0.195 and mass_top3 > 23.66
        - 0.001278300295405205 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.16902486327670005   # -0.1%  planar_flow < 0.195 and pt_6 < 35.28
        + 0.0012321149800140763 * max(0.0, Q.girth2 - 0.013238675334) * max(0.0, Q.log_sum_pt - 6.19222188581) / 0.00018635127471114103   # +0.1%  girth2 > 0.01324 and log_sum_pt > 6.192
        - 0.001012086500023532 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5) / 0.042474336856843424   # -0.1%  centroid_offset > 0.03117 and pt_2 > 56.5
        - 0.0009633692454087738 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478) / 0.038560975699403234   # -0.1%  mass > 80.4 and eccentricity > 0.9271
        - 0.0006148554248121738 * max(0.0, Q.z_dr_0p05_0p1 - 0.674770402908) / 0.03639977101200851   # -0.1%  z_dr_0p05_0p1 > 0.6748
        + 0.00031338178139971924 * max(0.0, Q.z_dr_0p05_0p1 - 0.674770402908) * max(0.0, Q.pt_2 - 84.625) / 0.5576858476335781   # +0.0%  z_dr_0p05_0p1 > 0.6748 and pt_2 > 84.62
        - 0.00028586810991133334 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0002912696960907807   # -0.0%  centroid_offset > 0.03117 and n_pt_above_50 > 6
        - 0.0002462903629701205 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_0 - 376.5) / 0.002409555939429373   # -0.0%  centroid_offset > 0.03117 and pt_0 > 376.5
        + 0.00018376351594010713 * max(0.0, 0.005590288644 - Q.width) * max(0.0, -0.009391680919 - Q.mean_eta) / 2.491701120725571e-06   # +0.0%  width < 0.00559 and mean_eta < -0.009392
        - 0.0001742829178994505 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.009460449219 - Q.phi_1) / 4.875626464778701e-07   # -0.0%  e2_sq < 0.001101 and phi_1 < -0.00946
        - 0.00010438173984209831 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) / 1.936089838400846e-05   # -0.0%  e2 < 0.02455 and z_dr_0p05_0p1 > 0.7509
        - 8.658835659106246e-05 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, Q.phi_0 - 0.040283203125) / 2.6348887399644323e-08   # -0.0%  e2_sq < 0.001101 and phi_0 > 0.04028
        - 6.068787095428786e-05 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.045445251465 - Q.eta_2) / 2.42508706203069e-08   # -0.0%  e2_sq < 0.001101 and eta_2 < -0.04545
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 15.66;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.664070601808646 * (-0.01933148304605013
        + 0.15607511103956848 * max(0.0, 0.005019718802 - Q.width) / 0.0019034243755815378   # +15.6%  width < 0.00502
        - 0.10355931948477698 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width) / 7.976303597876625e-05   # -10.4%  girth < 0.06109 and width < 0.00502
        + 0.07966990773979675 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4) / 0.00019011121076024945   # +8.0%  width < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.06335198928936199 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset) / 3.883081060200359e-05   # +6.3%  girth2 < 0.006679 and centroid_offset < 0.02355
        - 0.057103872792695305 * max(0.0, 0.061086014472 - Q.girth) / 0.018686846679786144   # -5.7%  girth < 0.06109
        - 0.04280041590504447 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.10213245107017851   # -4.3%  mass < 29.64 and centroid_offset < 0.02355
        - 0.035280903420522 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2) / 7.006679088185514e-06   # -3.5%  LHA < 0.1967 and lam2 < 0.0003061
        + 0.02862941844569804 * max(0.0, 0.177304983139 - Q.max_dr) / 0.07042349604125828   # +2.9%  max_dr < 0.1773
        + 0.027839244844240132 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2) / 2.869242750181301e-06   # +2.8%  girth < 0.06109 and lam2 < 0.0001948
        + 0.02553715164379916 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.00015145615486565555   # +2.6%  e2 < 0.02455 and centroid_offset < 0.03117
        - 0.024704856481083545 * max(0.0, 21.784077072144 - Q.mass) / 4.431023211632268   # -2.5%  mass < 21.78
        - 0.024622884961657605 * max(0.0, 0.196739721581 - Q.LHA) / 0.025065644276078353   # -2.5%  LHA < 0.1967
        + 0.02236540818489002 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2) / 9.998101856927162e-06   # +2.2%  max_dr < 0.1773 and lam2 < 0.0001948
        + 0.018959728139395445 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width) / 9.585949093597567e-06   # +1.9%  LHA < 0.1967 and width < 0.0005611
        - 0.018197578183730686 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000306123359 - Q.lam2) / 0.0006644045602408312   # -1.8%  mass < 15.45 and lam2 < 0.0003061
        + 0.016819211239159353 * max(0.0, 0.016554418951 - Q.e2) / 0.0038873472250323077   # +1.7%  e2 < 0.01655
        - 0.016288688700615273 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266) / 1.0643487980347305e-05   # -1.6%  width < 0.00502 and centroid_offset > 0.00679
        - 0.015778262503742608 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2) / 2.697123900716699e-05   # -1.6%  z_dr_0_0p05 > 0.8477 and lam2 < 0.0005373
        - 0.015017353354143566 * max(0.0, Q.pt_7 - 34.53125) / 4.473662132352941   # -1.5%  pt_7 > 34.53
        - 0.014571774453104269 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 48.71875 - Q.pt_7) / 0.030845495821782958   # -1.5%  width < 0.00502 and pt_7 < 48.72
        - 0.01304058773847788 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # -1.3%  mass < 29.64
        - 0.010884061547272863 * max(0.0, 0.006679471442 - Q.girth2) / 0.0029362398563122583   # -1.1%  girth2 < 0.006679
        - 0.010590578375740891 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.006978029732767499   # -1.1%  log_sum_pt > 6.701 and z_dr_0p2_0p4 < 0.2055
        - 0.010309122658558066 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.501026660204 - Q.tau21) / 0.0002928575232002313   # -1.0%  girth2 < 0.006679 and tau21 < 0.501
        + 0.010293753270504254 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.sum_pt - 788.4484375) / 0.5518276932797608   # +1.0%  e2 < 0.02455 and sum_pt > 788.4
        - 0.009907902648875384 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266) / 8.239263087841892e-05   # -1.0%  girth < 0.06109 and centroid_offset > 0.00679
        + 0.009887352089507667 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset) / 0.07546093782964779   # +1.0%  mass < 21.78 and centroid_offset < 0.02686
        + 0.009524545722011897 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow) / 0.0003918103360084937   # +1.0%  girth2 < 0.006679 and planar_flow < 0.4008
        + 0.009313935343895957 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.phi_0 - -0.040130615234) / 0.00011897527352772684   # +0.9%  girth2 < 0.006679 and phi_0 > -0.04013
        - 0.009228248549664088 * max(0.0, Q.log_sum_pt - 6.701242202626) / 0.035238726665107536   # -0.9%  log_sum_pt > 6.701
        - 0.007573682929209742 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003) / 5.841408434868268e-06   # -0.8%  girth2 < 0.006679 and centroid_offset > 0.01838
        + 0.007510157152553185 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7) / 0.7769310447436848   # +0.8%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.007191865990064111 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.013238675006 - Q.width) / 0.038941735602223373   # +0.7%  pt_7 > 34.53 and width < 0.01324
        + 0.006843086541411083 * max(0.0, 15.454033088684 - Q.mass) / 2.385786176762162   # +0.7%  mass < 15.45
        - 0.00633184430240206 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.0023404635737413245   # -0.6%  girth2 < 0.006679 and n_dr_0p05_0p1 > 0
        - 0.006237172043690218 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256) / 1.0102446095345114e-05   # -0.6%  girth < 0.06109 and lam1 > 0.0002759
        - 0.006142386266326434 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2) / 0.0005900482644939888   # -0.6%  log_sum_pt > 6.701 and girth2 < 0.01883
        - 0.005395253572049405 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.log_sum_pt - 6.670067010936) / 0.00021206265198456234   # -0.5%  girth2 < 0.006679 and log_sum_pt > 6.67
        + 0.004872147709668493 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.0005980205978877513   # +0.5%  log_sum_pt > 6.701 and centroid_offset < 0.02355
        - 0.004599691748824193 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649) / 9.121997321495253e-07   # -0.5%  width < 0.00502 and mass_over_sum_pt_sq > 0.0001232
        - 0.004191652951032435 * max(0.0, 0.000964142894 - Q.girth2) / 0.00020522880102919315   # -0.4%  girth2 < 0.0009641
        - 0.004072527743192195 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.0010110746822510478   # -0.4%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        + 0.004011850940716561 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq) / 3.2228282972711254e-06   # +0.4%  log_sum_pt > 6.701 and mass_over_sum_pt_sq < 0.0002368
        - 0.0035251067146825155 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7) / 0.03926630160825465   # -0.4%  mass < 15.45 and z_7 < 0.05861
        + 0.002884684571357053 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.girth2_top3 - 0.002915531053) / 3.8464148584062824e-07   # +0.3%  girth2 < 0.006679 and girth2_top3 > 0.002916
        - 0.002431993064711807 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # -0.2%  e2 < 0.02455
        + 0.0016101275109258275 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_6 - 0.02160287394) / 0.0006814928842293515   # +0.2%  LHA < 0.1967 and z_6 > 0.0216
        + 0.0013842970170163656 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.mean_phi - -0.000855675264) / 5.448370435500312e-05   # +0.1%  LHA < 0.1967 and mean_phi > -0.0008557
        + 0.0012563266255725734 * max(0.0, 8.379955863953 - Q.mass) / 0.5781888805022356   # +0.1%  mass < 8.38
        + 0.0011586055175580293 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.465420272571 - Q.pt1_over_pt0) / 0.002545451023553486   # +0.1%  max_dr < 0.1773 and pt1_over_pt0 < 0.4654
        - 0.000622370335501107 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.031170772021) / 0.004359226399952505   # -0.1%  mass < 21.78 and centroid_offset > 0.03117
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 25.51;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 25.506741989658426 * (-0.05991322645044254
        + 0.1517918037176061 * max(0.0, 0.006096650059 - Q.width) / 0.002544039033917776   # +15.2%  width < 0.006097
        - 0.06919829097773607 * max(0.0, 53.332374954224 - Q.mass) / 19.36003866872732   # -6.9%  mass < 53.33
        + 0.060791342970125305 * max(0.0, 0.007520088344 - Q.girth2) / 0.0035449904994304922   # +6.1%  girth2 < 0.00752
        + 0.05546647296790386 * max(0.0, 0.004372139461 - Q.girth2) / 0.001565710312409871   # +5.5%  girth2 < 0.004372
        - 0.05065326495398601 * max(0.0, 0.054649224505 - Q.girth) / 0.015329780861188023   # -5.1%  girth < 0.05465
        + 0.044405034128965404 * max(0.0, 0.221586732566 - Q.max_dr) / 0.10604661584056564   # +4.4%  max_dr < 0.2216
        - 0.04283491539800157 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) / 0.02715027320795572   # -4.3%  mass_over_sum_pt < 0.07637
        + 0.042684402441987394 * max(0.0, 0.020459658932 - Q.e2) / 0.00553192518950076   # +4.3%  e2 < 0.02046
        + 0.04188872274837034 * max(0.0, 0.018377780003 - Q.centroid_offset) / 0.006717135927308933   # +4.2%  centroid_offset < 0.01838
        - 0.03789404891370776 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496) / 2.100561780735258e-05   # -3.8%  width < 0.006097 and centroid_offset > 0.003343
        + 0.03266782277580771 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset) / 0.2973625515049245   # +3.3%  mass < 53.33 and centroid_offset < 0.02686
        - 0.031172958238584784 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # -3.1%  log_sum_pt > 6.378
        - 0.02953966791289194 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # -3.0%  mass < 29.64
        - 0.026773767881856307 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1) / 0.00860596641674667   # -2.7%  mass < 53.33 and lam1 < 0.0008722
        + 0.026318498440146006 * max(0.0, 0.000964142894 - Q.girth2) / 0.00020522880102919315   # +2.6%  girth2 < 0.0009641
        + 0.02574142736993309 * max(0.0, 41.377904891968 - Q.mass) / 12.537841289638873   # +2.6%  mass < 41.38
        + 0.024517263616925217 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt) / 5.175435934052333   # +2.5%  mass < 53.33 and log_sum_pt < 6.843
        - 0.018929488338246616 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878) / 0.0002260041308477699   # -1.9%  centroid_offset < 0.01838 and z_4 > 0.04749
        + 0.01764478449856471 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow) / 1.530012963091921   # +1.8%  mass < 53.33 and planar_flow < 0.3221
        - 0.014443545321598701 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.90890302062 - Q.z_top5) / 0.009708087570272587   # -1.4%  max_dr < 0.2216 and z_top5 < 0.9089
        - 0.010919820285586006 * max(0.0, 0.001503553356 - Q.lam1) / 0.0003926183331153187   # -1.1%  lam1 < 0.001504
        + 0.010906497896213396 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375) / 0.10790638482697934   # +1.1%  centroid_offset < 0.01838 and pt_4 > 47.34
        + 0.010506694270453448 * max(0.0, Q.lam2 - 0.001130644719) / 0.0003119523112525676   # +1.1%  lam2 > 0.001131
        - 0.010469530646749391 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.002127561159 - Q.mean_phi2) / 0.01390258796706784   # -1.0%  mass < 29.64 and mean_phi2 < 0.002128
        + 0.00982635833778 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01) / 0.0005482140187387444   # +1.0%  e2 < 0.03235 and dr01 < 0.05595
        - 0.008573736738244615 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow) / 0.8148647296533372   # -0.9%  mass < 41.38 and planar_flow < 0.3221
        - 0.008119730542228447 * max(0.0, 0.009480684835 - Q.centroid_offset) * max(0.0, 0.002127561159 - Q.mean_phi2) / 2.9084584434850715e-06   # -0.8%  centroid_offset < 0.009481 and mean_phi2 < 0.002128
        - 0.007488736898261919 * max(0.0, 0.005954149834 - Q.lam1) / 0.0025181540815307126   # -0.7%  lam1 < 0.005954
        - 0.00596038832261578 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7) / 0.11864275984569007   # -0.6%  e2 < 0.02046 and pt_7 < 53.44
        + 0.0058985789640755 * max(0.0, Q.girth2 - 0.018827652745) / 0.0007605368842505339   # +0.6%  girth2 > 0.01883
        + 0.005445756294805276 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.021648628542 - Q.dr_5) / 0.0008555541335392073   # +0.5%  log_sum_pt > 6.378 and dr_5 < 0.02165
        - 0.005393502215565462 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21) / 0.0007426911303479672   # -0.5%  e2 < 0.03235 and tau21 < 0.4466
        - 0.005215442066222804 * max(0.0, 0.032346998155 - Q.e2) / 0.011718069376880604   # -0.5%  e2 < 0.03235
        - 0.004935050295162172 * max(0.0, Q.max_dr - 0.15984864831) / 0.0215651977777285   # -0.5%  max_dr > 0.1598
        - 0.004717514392767323 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.002834883542 - Q.mean_phi) / 6.939734499466795e-05   # -0.5%  girth < 0.05465 and mean_phi < 0.002835
        - 0.004493818819582395 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, 0.021648628542 - Q.dr_5) / 0.00011056517027334096   # -0.4%  girth < 0.05465 and dr_5 < 0.02165
        + 0.004134003373430644 * max(0.0, 35.5 - Q.pt_5) / 1.4798162827435661   # +0.4%  pt_5 < 35.5
        + 0.0038536463337844936 * max(0.0, 0.000172198326 - Q.width) / 1.4418964712323033e-05   # +0.4%  width < 0.0001722
        - 0.0036763257413314414 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696) / 0.00012511356827127218   # -0.4%  e2 < 0.02046 and eccentricity > 0.9031
        + 0.003516579062749053 * max(0.0, Q.C2 - 0.051192347892) / 0.004801477093795494   # +0.4%  C2 > 0.05119
        + 0.002854670674215875 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, Q.n_dr_0p05_0p1 - 3.0) / 5.026654704942342   # +0.3%  mass < 53.33 and n_dr_0p05_0p1 > 3
        + 0.002842237207635874 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857142857143   # +0.3%  n_dr_0p2_0p4 > 1
        - 0.002802241984217636 * max(0.0, 0.009480684835 - Q.centroid_offset) / 0.0019592778854094204   # -0.3%  centroid_offset < 0.009481
        - 0.002380702849581964 * max(0.0, 0.000964142894 - Q.girth2) * max(0.0, 0.006823012256 - Q.mean_eta) / 1.4863986880019704e-06   # -0.2%  girth2 < 0.0009641 and mean_eta < 0.006823
        - 0.0021728048194571935 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516) / 5.181368461210433e-06   # -0.2%  width < 0.006097 and C2 > 0.03087
        - 0.001692313714209334 * max(0.0, Q.z_dr_0p2_0p4 - 0.1009733513) / 0.013827470319755132   # -0.2%  z_dr_0p2_0p4 > 0.101
        + 0.0013949408264632647 * max(0.0, 0.054649224505 - Q.girth) * max(0.0, Q.mean_phi2 - 0.000539434783) / 2.031612965486861e-06   # +0.1%  girth < 0.05465 and mean_phi2 > 0.0005394
        - 0.0011653078642439307 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 0.005566259290425861   # -0.1%  C2 > 0.05119 and n_dr_0p05_0p1 > 1
        + 0.0008464646301059047 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7) / 0.0006069490438354982   # +0.1%  girth2 > 0.01883 and pt_7 < 29.04
        - 0.0007014539089196771 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 3.0) / 6.48250251290431e-05   # -0.1%  girth2 < 0.004372 and n_dr_0p05_0p1 > 3
        + 0.0006590746301151578 * max(0.0, 4.8108519e-05 - Q.girth2) / 8.325084578958415e-07   # +0.1%  girth2 < 4.811e-05
        + 0.00045463702517158485 * max(0.0, Q.lam2 - 0.001130644719) * max(0.0, Q.mass_top2 - 16.308019673264) / 0.002687414260360442   # +0.0%  lam2 > 0.001131 and mass_top2 > 16.31
        - 0.0003825011732847414 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, Q.mass_top2 - 6.77991534008) / 0.00034808632749907815   # -0.0%  girth2 < 0.004372 and mass_top2 > 6.78
        + 0.00024141358182359018 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.620723099573 - Q.eccentricity) / 0.00020449793322373317   # +0.0%  C2 > 0.05119 and eccentricity < 0.6207
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 11.72;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.72397070809741 * (-0.14448526350835744
        + 0.17377469036424883 * Q.e2 / 0.028632153681352513   # +17.4%  e2
        + 0.10517053336763615 * max(0.0, Q.mass - 15.454033088684) / 27.25251475411285   # +10.5%  mass > 15.45
        + 0.09610689636448563 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt) / 0.037440038633677876   # +9.6%  mass_over_sum_pt < 0.09041
        - 0.06528934010676399 * max(0.0, 0.004183811014 - Q.lam1) / 0.0015130774487597748   # -6.5%  lam1 < 0.004184
        + 0.053683884393194506 * max(0.0, Q.lam2 - 0.000194798295) / 0.00044615149470896454   # +5.4%  lam2 > 0.0001948
        - 0.05137193056824839 * max(0.0, Q.log_sum_pt - 6.701242202626) / 0.035238726665107536   # -5.1%  log_sum_pt > 6.701
        - 0.03943171139700773 * max(0.0, 0.006506575659 - Q.lam1) / 0.0028900776904993344   # -3.9%  lam1 < 0.006507
        + 0.03661431606982963 * max(0.0, 0.002412890926 - Q.girth2_top2) / 0.0009487943624114899   # +3.7%  girth2_top2 < 0.002413
        - 0.03353275008368902 * max(0.0, 0.0016538364 - Q.girth2) / 0.00042890665324629343   # -3.4%  girth2 < 0.001654
        + 0.03341381977553901 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.0022074526402598485   # +3.3%  eccentricity > 0.9031 and z_dr_0p2_0p4 < 0.05644
        + 0.029694985809530478 * max(0.0, Q.sum_pt - 813.415625) / 31.70376994639103   # +3.0%  sum_pt > 813.4
        + 0.02827485322600802 * max(0.0, 3.0 - Q.n_dr_0p05_0p1) / 1.6691546218487394   # +2.8%  n_dr_0p05_0p1 < 3
        + 0.025598728292323104 * max(0.0, 0.006679471442 - Q.girth2) / 0.0029362398563122583   # +2.6%  girth2 < 0.006679
        + 0.022702443128213567 * max(0.0, 0.002270363079 - Q.girth2_top5) / 0.0007634099414501585   # +2.3%  girth2_top5 < 0.00227
        - 0.016334943038736137 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.planar_flow - 0.012569162668) / 0.00030371078629989727   # -1.6%  lam2 > 0.0001948 and planar_flow > 0.01257
        - 0.015082889120778913 * max(0.0, Q.LHA - 0.303313749495) / 0.017943093379202503   # -1.5%  LHA > 0.3033
        - 0.014633473255133905 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 36.768271023571 - Q.mass_top2) / 1.3059079533204017   # -1.5%  eccentricity > 0.9031 and mass_top2 < 36.77
        + 0.014303408099383822 * max(0.0, Q.eccentricity - 0.903125533696) / 0.047390935319967936   # +1.4%  eccentricity > 0.9031
        + 0.014173620048229432 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626) / 9.541953566382599e-05   # +1.4%  lam1 < 0.004184 and log_sum_pt > 6.701
        - 0.013900279500058246 * max(0.0, Q.centroid_offset - 0.00231612516) / 0.014966942305045673   # -1.4%  centroid_offset > 0.002316
        - 0.011049768704839326 * max(0.0, Q.mass - 53.332374954224) / 6.348425380208738   # -1.1%  mass > 53.33
        - 0.010845672370835351 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.pt_4 - 39.8125) / 2.740588593384243   # -1.1%  tau21 < 0.3915 and pt_4 > 39.81
        - 0.008474067054720235 * max(0.0, Q.LHA - 0.303313749495) * max(0.0, 0.553068161011 - Q.tau21) / 0.005421876887000885   # -0.8%  LHA > 0.3033 and tau21 < 0.5531
        - 0.008273903829200152 * max(0.0, Q.mass - 15.454033088684) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 37.95303185489477   # -0.8%  mass > 15.45 and n_dr_0p2_0p4 < 2
        - 0.007985629797453727 * max(0.0, Q.max_dr - 0.121680960059) / 0.035622055613376453   # -0.8%  max_dr > 0.1217
        - 0.0076205286618300115 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, 0.175465903809 - Q.dr_7) / 0.0002170263625266813   # -0.8%  lam1 < 0.004184 and dr_7 < 0.1755
        + 0.006680924081708729 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.z_4 - 0.075444822386) / 0.0021348581786076234   # +0.7%  tau21 < 0.3915 and z_4 > 0.07544
        + 0.005488804954499911 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, 0.251215918102 - Q.dr01) / 0.024554591033188237   # +0.5%  tau21 < 0.3915 and dr01 < 0.2512
        + 0.0052358823766717084 * max(0.0, Q.C2 - 0.051192347892) / 0.004801477093795494   # +0.5%  C2 > 0.05119
        + 0.004169439043355283 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 0.501026660204 - Q.tau21) / 4.680532629659398e-05   # +0.4%  lam2 > 0.0001948 and tau21 < 0.501
        - 0.004082317765507798 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2) / 0.0003715223898038417   # -0.4%  lam2 > 0.0001948 and D2 < 2.055
        + 0.003989006862479916 * max(0.0, 0.269169217348 - Q.tau32) / 0.011024941728233898   # +0.4%  tau32 < 0.2692
        - 0.003909856845144186 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.0016231978158967296   # -0.4%  lam2 > 0.0001948 and n_pt_above_50 < 8
        - 0.00374953036870411 * max(0.0, Q.z_7 - 0.06164517166) / 0.004746501812091229   # -0.4%  z_7 > 0.06165
        - 0.0034945113776109144 * max(0.0, 3.0 - Q.n_dr_0p05_0p1) * max(0.0, 1.432482242584 - Q.D2) / 0.36499841278621575   # -0.3%  n_dr_0p05_0p1 < 3 and D2 < 1.432
        + 0.003259232331624922 * max(0.0, Q.centroid_offset - 0.037760993714) / 0.001689371734514427   # +0.3%  centroid_offset > 0.03776
        - 0.0029043117370988205 * max(0.0, Q.centroid_offset - 0.00231612516) * max(0.0, 34.53125 - Q.pt_7) / 0.05030026650419268   # -0.3%  centroid_offset > 0.002316 and pt_7 < 34.53
        - 0.0026332160447873365 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, Q.z_top2_slots - 0.55004856109) / 0.001342047682883366   # -0.3%  eccentricity > 0.9031 and z_top2_slots > 0.55
        + 0.0024832440021315652 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt) * max(0.0, Q.pt_7 - 33.21875) / 0.2027742166957715   # +0.2%  mass_over_sum_pt < 0.09041 and pt_7 > 33.22
        + 0.001849770218778718 * max(0.0, 0.003904593248 - Q.mass_over_sum_pt_sq) / 0.001485439179640537   # +0.2%  mass_over_sum_pt_sq < 0.003905
        - 0.0018140195324667608 * max(0.0, 0.269169217348 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.008366995203849923   # -0.2%  tau32 < 0.2692 and n_dr_0p2_0p4 < 2
        - 0.001793962078205837 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 38.25 - Q.pt_6) / 0.0014312222249272951   # -0.2%  lam2 > 0.0001948 and pt_6 < 38.25
        - 0.0014035641637502587 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, -0.006779838586 - Q.mean_eta) / 0.0004490588402755699   # -0.1%  tau21 < 0.3915 and mean_eta < -0.00678
        + 0.0013065623977749372 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.mean_phi - 0.009050007537) / 1.5328753062583084e-06   # +0.1%  lam1 < 0.004184 and mean_phi > 0.00905
        + 0.0011794759878097551 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.z_top2_slots - 0.476619814198) / 1.0017767074185762e-05   # +0.1%  lam2 > 0.0001948 and z_top2_slots > 0.4766
        - 0.0006998488184817801 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125) / 0.014000463321047197   # -0.1%  lam1 < 0.004184 and sum_pt > 988.4
        + 0.00040474050671460154 * max(0.0, Q.n_dr_0p2_0p4 - 2.0) * max(0.0, Q.dr_7 - 0.222994708167) / 0.0010373989028593532   # +0.0%  n_dr_0p2_0p4 > 2 and dr_7 > 0.223
        + 0.0001287120767749307 * max(0.0, Q.n_dr_0p2_0p4 - 2.0) * max(0.0, 0.042151962757 - Q.dr_7) / 4.0710583626890944e-05   # +0.0%  n_dr_0p2_0p4 > 2 and dr_7 < 0.04215
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.998206858684817 * (-0.047211918994382686
        + 0.167618389743922 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # +16.8%  width < 0.008678
        + 0.10456266549675063 * max(0.0, 0.049903668404 - Q.centroid_offset) / 0.03352573038002464   # +10.5%  centroid_offset < 0.0499
        - 0.08176064115603104 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -8.2%  girth < 0.08724
        + 0.07396168372042569 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # +7.4%  girth2 < 0.01324
        + 0.06738626618273445 * max(0.0, 0.006679471442 - Q.girth2) / 0.0029362398563122583   # +6.7%  girth2 < 0.006679
        - 0.04933634224172409 * max(0.0, 0.006390124748 - Q.e2_sq) / 0.0029535077959469682   # -4.9%  e2_sq < 0.00639
        - 0.047465633676972596 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -4.7%  lam1 < 0.008376
        - 0.03852965254406526 * max(0.0, Q.girth - 0.076081777364) / 0.010590329529546494   # -3.9%  girth > 0.07608
        - 0.03477230199943117 * max(0.0, 0.04447356835 - Q.e2) / 0.020193337934351147   # -3.5%  e2 < 0.04447
        + 0.03314325637473322 * max(0.0, 0.253403707141 - Q.planar_flow) / 0.10951502727440561   # +3.3%  planar_flow < 0.2534
        + 0.022939602465471217 * max(0.0, 0.221586732566 - Q.max_dr) / 0.10604661584056564   # +2.3%  max_dr < 0.2216
        + 0.02169020160795319 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.026536230156759166   # +2.2%  girth > 0.07608 and n_pt_above_50 < 7
        - 0.021385100199708875 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.007589401193867531   # -2.1%  centroid_offset < 0.0499 and log_sum_pt < 6.804
        - 0.020382827841176964 * max(0.0, 0.003562611091 - Q.width) / 0.001184249467958887   # -2.0%  width < 0.003563
        - 0.019658529379330807 * max(0.0, 0.004839980301 - Q.lam1) / 0.0018550881157933209   # -2.0%  lam1 < 0.00484
        - 0.015781232421394776 * max(0.0, Q.centroid_offset - 0.014379521101) / 0.0071112999721223424   # -1.6%  centroid_offset > 0.01438
        - 0.015221255960260513 * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 1.9597932773109243   # -1.5%  n_dr_0p1_0p2 < 3
        + 0.01249525792864891 * max(0.0, 687.4375 - Q.sum_pt_top5) / 131.13431972163866   # +1.2%  sum_pt_top5 < 687.4
        + 0.011895429840319095 * Q.centroid_offset / 0.017184338140317994   # +1.2%  centroid_offset
        - 0.010475043527852222 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass) / 2.179941616634416   # -1.0%  planar_flow < 0.2534 and mass < 69.61
        - 0.010273473713528264 * max(0.0, 0.154689112391 - Q.LHA) / 0.012614340772605542   # -1.0%  LHA < 0.1547
        - 0.009904642354270785 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7) / 0.5039522750508828   # -1.0%  centroid_offset < 0.0499 and pt_7 < 48.72
        - 0.008414921577117285 * max(0.0, Q.girth - 0.101940929517) / 0.005400003353656977   # -0.8%  girth > 0.1019
        - 0.008099911059604047 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059) / 0.00026898808711045845   # -0.8%  planar_flow < 0.2534 and width > 0.006097
        - 0.007803239667389486 * max(0.0, 0.111761856824 - Q.max_dr) / 0.028394648782258777   # -0.8%  max_dr < 0.1118
        - 0.006656940793682058 * max(0.0, 687.4375 - Q.sum_pt_top5) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 210.09499753807773   # -0.7%  sum_pt_top5 < 687.4 and n_dr_0p2_0p4 < 2
        + 0.006653002121259089 * max(0.0, 0.002151567843 - Q.girth2_top3) * max(0.0, Q.phi_0 - -0.029769897461) / 2.327303926321172e-05   # +0.7%  girth2_top3 < 0.002152 and phi_0 > -0.02977
        - 0.006424236082882982 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 0.001570267399 - Q.mean_phi2) / 2.5628126471122562e-05   # -0.6%  centroid_offset < 0.0499 and mean_phi2 < 0.00157
        - 0.00605292664200831 * max(0.0, 0.035786485299 - Q.C2) / 0.015048242903037393   # -0.6%  C2 < 0.03579
        - 0.005241920277566446 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.girth2 - 0.013238675334) / 0.0001108174499013517   # -0.5%  planar_flow < 0.2534 and girth2 > 0.01324
        - 0.005186274559272739 * max(0.0, 7.0 - Q.n_dr_0_0p05) / 3.1108840336134453   # -0.5%  n_dr_0_0p05 < 7
        - 0.004840123282495176 * max(0.0, 0.002151567843 - Q.girth2_top3) / 0.0007789602594300843   # -0.5%  girth2_top3 < 0.002152
        - 0.004376146467407865 * max(0.0, 29.0421875 - Q.pt_7) / 2.1799836923983107   # -0.4%  pt_7 < 29.04
        - 0.0037069723769343034 * max(0.0, 7.0 - Q.n_dr_0_0p05) * max(0.0, 0.500737345219 - Q.z_1st) / 0.6542740257058977   # -0.4%  n_dr_0_0p05 < 7 and z_1st < 0.5007
        - 0.0036012706766618777 * max(0.0, 0.000657050184 - Q.girth2_top5) / 0.0001395474066313899   # -0.4%  girth2_top5 < 0.0006571
        + 0.0033107353411116485 * max(0.0, 0.221586732566 - Q.max_dr) * max(0.0, 0.021438598633 - Q.phi_0) / 0.0027994013166307264   # +0.3%  max_dr < 0.2216 and phi_0 < 0.02144
        + 0.0031893755628878088 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 1.679198372364 - Q.D2) / 0.09189025441465269   # +0.3%  planar_flow < 0.2534 and D2 < 1.679
        - 0.0031796974677414497 * max(0.0, 0.001056655216 - Q.girth2_top2) / 0.0003002513836012997   # -0.3%  girth2_top2 < 0.001057
        - 0.002538411499408534 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 3.558400598421145   # -0.3%  pt_7 < 29.04 and n_dr_0p2_0p4 < 2
        + 0.0024562619673496417 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 22.844978847276 - Q.mass_top2) / 41.04635219704124   # +0.2%  pt_7 < 29.04 and mass_top2 < 22.84
        + 0.0023809066456309887 * max(0.0, 9.257203159811 - Q.mass_top5) / 2.0942092841495867   # +0.2%  mass_top5 < 9.257
        - 0.0023591775884861292 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, -0.000855675264 - Q.mean_phi) / 0.00010136059739565747   # -0.2%  centroid_offset < 0.0499 and mean_phi < -0.0008557
        + 0.0021773642998372932 * max(0.0, 0.007078157854 - Q.e2) / 0.0007714961545607776   # +0.2%  e2 < 0.007078
        + 0.0020669433246031984 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7) / 5.884131049165044e-05   # +0.2%  LHA < 0.1547 and z_7 < 0.02807
        - 0.0018060978491632798 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.max_dr - 0.102758520097) / 0.005604755034171544   # -0.2%  planar_flow < 0.2534 and max_dr > 0.1028
        - 0.0017490827207716866 * max(0.0, 6.701242202626 - Q.log_sum_pt) / 0.19354909741955528   # -0.2%  log_sum_pt < 6.701
        + 0.0017227935630918518 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.036210384231276795   # +0.2%  centroid_offset < 0.0499 and n_dr_0p05_0p1 > 2
        - 0.0011010888813517445 * max(0.0, 0.006390124748 - Q.e2_sq) * max(0.0, 1.332146394253 - Q.D2) / 0.0003168302705384586   # -0.1%  e2_sq < 0.00639 and D2 < 1.332
        + 0.001048221919021468 * max(0.0, 15.454033088684 - Q.mass) / 2.385786176762162   # +0.1%  mass < 15.45
        + 0.0003557169916932081 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, 29.875 - Q.pt_5) / 0.0018938786093925187   # +0.0%  centroid_offset > 0.01438 and pt_5 < 29.88
        - 0.00034541854686693943 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 0.034484056668 - Q.z_6) / 0.00012052172661138358   # -0.0%  planar_flow < 0.2534 and z_6 < 0.03448
        + 0.0003397132121634051 * max(0.0, 0.003562611091 - Q.width) * max(0.0, Q.C2 - 0.030867108516) / 1.3106295829297012e-06   # +0.0%  width < 0.003563 and C2 > 0.03087
        - 0.00017567665783235878 * max(0.0, 9.257203159811 - Q.mass_top5) * max(0.0, -0.053314208984 - Q.eta_0) / 0.0005611782913352576   # -0.0%  mass_top5 < 9.257 and eta_0 < -0.05331
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5899847429695153 * (-1.3482140386524983
        + 0.3360530110734862 * max(0.0, Q.girth2 - 0.018827652745) / 0.0007605368842505339   # +33.6%  girth2 > 0.01883
        - 0.15118445275008718 * max(0.0, Q.e2 - 0.063441075385) / 0.0017618611738475503   # -15.1%  e2 > 0.06344
        - 0.13646786819725776 * max(0.0, Q.mass_over_sum_pt - 0.13092863437) / 0.002541184685006073   # -13.6%  mass_over_sum_pt > 0.1309
        + 0.06628195133302738 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625) / 0.014489759228566367   # +6.6%  girth2 > 0.01883 and pt_7 > 15.55
        + 0.05638594298445372 * max(0.0, Q.mass - 91.19) / 0.5839190666579981   # +5.6%  mass > 91.19
        + 0.05328614586676711 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.lam2 - 0.000537286005) / 2.828867171668773e-06   # +5.3%  girth2 > 0.01883 and lam2 > 0.0005373
        + 0.049655676474249975 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857142857143   # +5.0%  n_dr_0p2_0p4 > 1
        - 0.038866152782883585 * max(0.0, Q.girth2_top2 - 0.0140332421) * max(0.0, Q.width - 0.013238675006) / 1.8104641795245737e-05   # -3.9%  girth2_top2 > 0.01403 and width > 0.01324
        - 0.030806283976405325 * max(0.0, Q.girth2_top2 - 0.0140332421) * max(0.0, 0.089100391399 - Q.z_6) / 1.57364806682145e-05   # -3.1%  girth2_top2 > 0.01403 and z_6 < 0.0891
        - 0.025505866429375284 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) * max(0.0, 0.003408388935 - Q.lam2) / 0.0002706388697491293   # -2.6%  n_dr_0p2_0p4 > 1 and lam2 < 0.003408
        + 0.023585959930558304 * max(0.0, Q.girth2_top2 - 0.0140332421) / 0.0011295753560857935   # +2.4%  girth2_top2 > 0.01403
        - 0.022050466564554252 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 80.4 - Q.mass) / 0.003354753666125259   # -2.2%  girth2 > 0.01883 and mass < 80.4
        + 0.00987022163689414 * max(0.0, Q.mean_phi - 0.026127964072) / 0.000699941107518642   # +1.0%  mean_phi > 0.02613
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 20.16;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 20.157564498046888 * (0.12022284175492348
        + 0.20096174057535765 * max(0.0, 0.148408418149 - Q.girth) / 0.09059989578347671   # +20.1%  girth < 0.1484
        - 0.10919871559607099 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # -10.9%  e2 < 0.05028
        + 0.10779085018385472 * max(0.0, 0.016433749775 - Q.lam1) / 0.011201097678927486   # +10.8%  lam1 < 0.01643
        + 0.07429875028388325 * max(0.0, 0.013238675006 - Q.width) / 0.00823612446188605   # +7.4%  width < 0.01324
        - 0.055552897156159055 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556396072   # -5.6%  width < 0.00752
        + 0.04788682104541367 * max(0.0, 0.037760993714 - Q.centroid_offset) / 0.02226602730794202   # +4.8%  centroid_offset < 0.03776
        - 0.04632734936224456 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset) / 7.463904537181694e-06   # -4.6%  lam2 < 0.0003061 and centroid_offset < 0.0499
        - 0.044945049084625474 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.01998233930103558   # -4.5%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.039654089882132655 * max(0.0, Q.LHA - 0.09323897448) / 0.15125019938338352   # -4.0%  LHA > 0.09324
        + 0.03618397885625682 * max(0.0, 0.006506575659 - Q.lam1) / 0.0028900776904993344   # +3.6%  lam1 < 0.006507
        - 0.03200580057119528 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset) / 0.00027886707988999124   # -3.2%  lam1 < 0.01643 and centroid_offset < 0.03776
        - 0.029427914188035524 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708193581740943   # -2.9%  girth < 0.1484 and pt_7 < 38.53
        + 0.02648511111986206 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7) / 927.286113873142   # +2.6%  sum_pt_top5 > 658.1 and pt_7 < 43.5
        - 0.017372938377848125 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841) / 0.027815441892245107   # -1.7%  tau21 < 0.501 and max_dr > 0.0156
        - 0.01721115071019049 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6) / 0.19240882891139893   # -1.7%  lam1 < 0.01643 and pt_6 < 56.53
        + 0.014707627620360243 * max(0.0, 0.000306123359 - Q.lam2) / 0.00019890943789793307   # +1.5%  lam2 < 0.0003061
        - 0.008895948722601101 * max(0.0, Q.sum_pt - 988.4078125) / 4.402477808070966   # -0.9%  sum_pt > 988.4
        + 0.008572461504511382 * max(0.0, Q.sum_pt_top5 - 839.9546875) / 8.60588864380794   # +0.9%  sum_pt_top5 > 840
        - 0.007801864330231846 * max(0.0, Q.C2 - 0.067292226106) / 0.002854902335729804   # -0.8%  C2 > 0.06729
        - 0.007335427366200137 * max(0.0, 0.003562611091 - Q.width) / 0.001184249467958887   # -0.7%  width < 0.003563
        - 0.006915857494199614 * max(0.0, 25.578125 - Q.pt_7) / 1.3273888893448005   # -0.7%  pt_7 < 25.58
        + 0.0067298359403958515 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362) / 0.0016492880690503395   # +0.7%  e2 < 0.05028 and pt_dispersion > 0.3968
        - 0.006156177058381337 * max(0.0, 763.825 - Q.sum_pt) / 96.68366216443064   # -0.6%  sum_pt < 763.8
        - 0.006140013604159348 * max(0.0, 0.028070914944 - Q.z_7) / 0.0013098148516826897   # -0.6%  z_7 < 0.02807
        - 0.005675305174040936 * max(0.0, Q.mass - 53.332374954224) / 6.348425380208738   # -0.6%  mass > 53.33
        - 0.003982451155522343 * max(0.0, 31.90625 - Q.pt_6) / 1.8268536428243172   # -0.4%  pt_6 < 31.91
        - 0.0038584928611871358 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189) / 0.2899084037160227   # -0.4%  sum_pt_top5 > 658.1 and z_7 > 0.02321
        - 0.0029663801348572593 * max(0.0, 24.578125 - Q.pt_5) / 0.2839763142561712   # -0.3%  pt_5 < 24.58
        - 0.0029630031827433756 * max(0.0, Q.z_top5 - 0.877015459538) / 0.007201250361008148   # -0.3%  z_top5 > 0.877
        + 0.0029303550921216262 * max(0.0, 0.067272114405 - Q.z_6) / 0.01320613266824519   # +0.3%  z_6 < 0.06727
        + 0.002856240523980507 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55591824366466   # +0.3%  sum_pt_top5 > 658.1
        + 0.0023059506485515616 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2) / 7.606722001858439   # +0.2%  sum_pt > 988.4 and D2 < 3.886
        + 0.002056283392098405 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 5.0 - Q.n_pt_above_50) / 0.01349708810525873   # +0.2%  e2 < 0.05028 and n_pt_above_50 < 5
        + 0.0016275488592339357 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0) / 2.3317601365545144   # +0.2%  sum_pt > 988.4 and n_pt_above_50 > 6
        + 0.0011767232499920548 * max(0.0, Q.girth - 0.101940929517) / 0.005400003353656977   # +0.1%  girth > 0.1019
        + 0.0011300263736497144 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.z_dr_0p05_0p1 - 0.163898047805) / 0.00014839222087436268   # +0.1%  lam1 < 0.006507 and z_dr_0p05_0p1 > 0.1639
        - 0.0010503743344157906 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 0.476255698625 - Q.pt_balance01) / 0.00039667758464517484   # -0.1%  girth > 0.1019 and pt_balance01 < 0.4763
        - 0.0010382846005466456 * max(0.0, Q.girth - 0.101940929517) * max(0.0, 116.0 - Q.pt_1) / 0.14900499970582204   # -0.1%  girth > 0.1019 and pt_1 < 116
        - 0.0009029134808900353 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.m012 - 16.899120053094) / 0.004618533124313442   # -0.1%  width < 0.00752 and m012 > 16.9
        - 0.0007659050058307261 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2) / 5.336336720841763   # -0.1%  sum_pt_top5 > 902.4 and D2 < 3.886
        + 0.0007153124388505586 * max(0.0, Q.sum_pt_top5 - 902.40625) / 4.122483656939338   # +0.1%  sum_pt_top5 > 902.4
        - 0.0006318931400233165 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.9556856092436975   # -0.1%  sum_pt_top5 > 902.4 and n_pt_above_50 > 6
        + 0.0006231901124193949 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, Q.n_pt_above_50 - 2.0) / 21.416234423249538   # +0.1%  sum_pt_top5 > 840 and n_pt_above_50 > 2
        + 0.0005988630154577823 * max(0.0, 0.003562611091 - Q.width) * max(0.0, Q.n_pt_above_50 - 7.0) / 0.0001263044113896617   # +0.1%  width < 0.003563 and n_pt_above_50 > 7
        + 0.0005522066952481044 * max(0.0, 531.1875 - Q.sum_pt_top5) * max(0.0, 29.875 - Q.pt_5) / 7.7517236318636344   # +0.1%  sum_pt_top5 < 531.2 and pt_5 < 29.88
        + 0.00045859851985329756 * max(0.0, Q.z_top5_slots - 0.930764273368) / 0.0009551768428254974   # +0.0%  z_top5_slots > 0.9308
        - 0.00024290100036921574 * max(0.0, Q.girth - 0.101940929517) * max(0.0, Q.pt_4 - 81.375) / 0.0010189019585713386   # -0.0%  girth > 0.1019 and pt_4 > 81.38
        - 0.0002030037349125767 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 0.362731824815 - Q.tau32) / 0.17817589723224078   # -0.0%  sum_pt_top5 > 658.1 and tau32 < 0.3627
        + 0.0001294226390326893 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.037477688199 - Q.z_4) / 0.0007394403345688553   # +0.0%  sum_pt < 763.8 and z_4 < 0.03748
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 39.96;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 39.95569447855095 * (-0.02450571756203325
        - 0.15615594574063685 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556396072   # -15.6%  width < 0.00752
        + 0.1312583069428434 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741761488   # +13.1%  girth2 < 0.01324
        + 0.11688818177277092 * max(0.0, 0.008678044951 - Q.width) / 0.0044473475499813035   # +11.7%  width < 0.008678
        - 0.07792859420207016 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -7.8%  girth < 0.08724
        - 0.07431187683165723 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq) / 0.007206102265286917   # -7.4%  mass_over_sum_pt_sq < 0.01166
        + 0.052249175419250574 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +5.2%  e2 < 0.03847
        + 0.03935395491653188 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.009470677792468518   # +3.9%  width < 0.00752 and n_dr_0p1_0p2 < 3
        - 0.03528209801297627 * max(0.0, 0.177304983139 - Q.max_dr) / 0.07042349604125828   # -3.5%  max_dr < 0.1773
        + 0.030581317512512092 * max(0.0, Q.lam1 - 0.002464291268) / 0.004201698684296263   # +3.1%  lam1 > 0.002464
        + 0.030458577458425347 * max(0.0, 0.017162483186 - Q.e2_sq) / 0.01203539920865141   # +3.0%  e2_sq < 0.01716
        - 0.023385416071146387 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2) / 0.015914745221096264   # -2.3%  z_dr_0p05_0p1 < 0.5883 and C2 < 0.06729
        - 0.023377829171852042 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.01994141641861341   # -2.3%  girth2 < 0.01324 and n_dr_0p1_0p2 < 3
        + 0.01618225000273961 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) / 0.3724173891308251   # +1.6%  z_dr_0p05_0p1 < 0.5883
        - 0.015840824322772045 * max(0.0, Q.lam1 - 0.004183811014) / 0.0032456267321051657   # -1.6%  lam1 > 0.004184
        - 0.013611711173012134 * max(0.0, 0.035560912266 - Q.e2) / 0.01371972344513943   # -1.4%  e2 < 0.03556
        - 0.013167368676085483 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2) / 0.00035388916079662355   # -1.3%  width < 0.008678 and D2 < 1.002
        - 0.012078031240545054 * max(0.0, Q.lam1 - 0.012003726523) / 0.0012289213133753257   # -1.2%  lam1 > 0.012
        - 0.011843815363003114 * max(0.0, 76.655700683594 - Q.mass) / 37.88098551900848   # -1.2%  mass < 76.66
        + 0.0102779914676156 * max(0.0, Q.lam1 - 0.005954149834) / 0.0024803645448479386   # +1.0%  lam1 > 0.005954
        - 0.010224313341865267 * max(0.0, Q.lam1 - 0.00543336053) / 0.0026771398012749854   # -1.0%  lam1 > 0.005433
        - 0.010109853856255593 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.1585445378151262   # -1.0%  n_dr_0p05_0p1 < 5
        + 0.00776602333713931 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.8008591717572943   # +0.8%  z_dr_0p05_0p1 < 0.5883 and n_dr_0p1_0p2 < 3
        + 0.007355081953325019 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164) / 5.9931533843830415e-05   # +0.7%  girth2 < 0.01324 and eccentricity > 0.9704
        - 0.0069807404188903304 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.111513564951 - Q.planar_flow) / 6.286786921042492e-05   # -0.7%  width < 0.00752 and planar_flow < 0.1115
        + 0.0066899135919215825 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 1.002470755577 - Q.D2) / 0.0002118058479258733   # +0.7%  width < 0.00752 and D2 < 1.002
        - 0.006677904813721189 * max(0.0, 0.303313749495 - Q.LHA) / 0.07849185419645244   # -0.7%  LHA < 0.3033
        - 0.006469629968709778 * max(0.0, 1.122624260187 - Q.D2) / 0.23862964281013382   # -0.6%  D2 < 1.123
        + 0.0063419724892006145 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835) / 0.0002838341565068102   # +0.6%  planar_flow < 0.1115 and centroid_offset > 0.009481
        + 0.004074086123125155 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2) / 0.0010081226860790134   # +0.4%  girth2 < 0.01324 and D2 < 1.002
        - 0.0035987349429303164 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.001519155562097681   # -0.4%  girth2 < 0.004372 and n_dr_0p2_0p4 < 1
        - 0.0031713072004174205 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003) / 0.00015097111917114598   # -0.3%  planar_flow < 0.1115 and centroid_offset > 0.01838
        - 0.0031310910369316517 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr) / 0.0010568587168460638   # -0.3%  planar_flow < 0.1115 and max_dr < 0.1598
        + 0.002849613096960218 * max(0.0, 1.122624260187 - Q.D2) * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.004096854179158968   # +0.3%  D2 < 1.123 and centroid_offset < 0.03117
        + 0.0027756222396049064 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.0006963583088947689   # +0.3%  width < 0.00752 and log_sum_pt < 6.804
        - 0.0026584019815062043 * max(0.0, Q.lam1 - 0.008375572068) / 0.0018409335087098556   # -0.3%  lam1 > 0.008376
        + 0.0025190149474933424 * max(0.0, 0.74595130682 - Q.D2) / 0.09626873727471312   # +0.3%  D2 < 0.746
        + 0.002470242816030237 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067296656828152   # +0.2%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        - 0.0024571881664875207 * max(0.0, 0.080507021025 - Q.max_dr) / 0.01536430929683225   # -0.2%  max_dr < 0.08051
        - 0.002395308732015748 * max(0.0, Q.centroid_offset - 0.049903668404) / 0.0008064001164412618   # -0.2%  centroid_offset > 0.0499
        - 0.002207487850284808 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) / 0.0228131875171655   # -0.2%  z_dr_0p05_0p1 > 0.7509
        + 0.0021288875665946426 * max(0.0, 0.111513564951 - Q.planar_flow) / 0.03343186742548701   # +0.2%  planar_flow < 0.1115
        + 0.00210351841242051 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, Q.eccentricity - 0.970449631164) / 0.0023597221386163136   # +0.2%  z_dr_0p05_0p1 < 0.5883 and eccentricity > 0.9704
        - 0.0018687939371270865 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 0.87567204833 - Q.D2) / 1.4804730524080404e-05   # -0.2%  girth2 < 0.004372 and D2 < 0.8757
        - 0.0017550350768834404 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, 0.74595130682 - Q.D2) / 1.4494675524944256   # -0.2%  mass < 76.66 and D2 < 0.746
        + 0.0017507726905758296 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr) / 1.2809763884266374e-05   # +0.2%  lam1 > 0.005433 and max_dr < 0.1598
        - 0.0011557693890688263 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 739.5 - Q.sum_pt) / 1.9441833461309348   # -0.1%  planar_flow < 0.1115 and sum_pt < 739.5
        + 0.0007306960475670242 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2) / 0.0005071967096445531   # +0.1%  e2 < 0.03847 and D2 < 1.002
        + 0.0005195521646290752 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, Q.n_dr_0p1_0p2 - 4.0) / 1.557132548373888   # +0.1%  mass < 76.66 and n_dr_0p1_0p2 > 4
        + 0.0004314404298506088 * max(0.0, Q.lam1 - 0.007330079875) * max(0.0, 0.13261153996 - Q.max_dr) / 7.5871203399465e-07   # +0.0%  lam1 > 0.00733 and max_dr < 0.1326
        - 0.0003987350820203501 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.sum_pt - 840.01953125) / 0.6760173795951988   # -0.0%  planar_flow < 0.1115 and sum_pt > 840
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 35.62;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.62473382124418 * (-0.05785644660374152
        + 0.11987906444411346 * max(0.0, 0.013238675006 - Q.width) / 0.00823612446188605   # +12.0%  width < 0.01324
        - 0.10297579778457283 * max(0.0, 0.006679471358 - Q.width) / 0.002936239797737678   # -10.3%  width < 0.006679
        + 0.06091042067737335 * max(0.0, 0.041109715588 - Q.e2) / 0.01758452927609984   # +6.1%  e2 < 0.04111
        - 0.06016763982254911 * max(0.0, 0.008375572068 - Q.lam1) / 0.004300145279369157   # -6.0%  lam1 < 0.008376
        + 0.058295636709900275 * max(0.0, 0.006506575659 - Q.lam1) / 0.0028900776904993344   # +5.8%  lam1 < 0.006507
        + 0.05720099275958414 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq) / 0.0035288487287669316   # +5.7%  mass_over_sum_pt_sq < 0.007183
        - 0.053826110333435984 * max(0.0, 0.008168570676 - Q.e2_sq) / 0.004294747328233826   # -5.4%  e2_sq < 0.008169
        + 0.051352748079524314 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # +5.1%  e2 < 0.02455
        - 0.046745592898446604 * max(0.0, 0.011657374702 - Q.e2_sq) / 0.0072030732891324245   # -4.7%  e2_sq < 0.01166
        + 0.04579678587598509 * max(0.0, 0.018827652745 - Q.girth2) / 0.013142955535859579   # +4.6%  girth2 < 0.01883
        - 0.04512173892067671 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556396072   # -4.5%  width < 0.00752
        - 0.037577272811193614 * max(0.0, 0.012003726523 - Q.lam1) / 0.007316287539015409   # -3.8%  lam1 < 0.012
        + 0.02552714978468239 * max(0.0, Q.girth - 0.033604209498) / 0.03159705075660309   # +2.6%  girth > 0.0336
        + 0.024487071167938572 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2) / 0.22863861264226853   # +2.4%  z_dr_0p1_0p2 < 0.3285
        - 0.02337535095087146 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt) / 0.022258343201397777   # -2.3%  mass_over_sum_pt < 0.06814
        - 0.017659776864829783 * max(0.0, 36.229410171509 - Q.mass) / 10.113929790442333   # -1.8%  mass < 36.23
        - 0.016974170233622354 * max(0.0, 0.001130644719 - Q.lam2) / 0.0009137232342242098   # -1.7%  lam2 < 0.001131
        - 0.015362462122845005 * max(0.0, 0.002151567843 - Q.girth2_top3) / 0.0007789602594300843   # -1.5%  girth2_top3 < 0.002152
        - 0.013416073820014343 * max(0.0, Q.LHA - 0.346713497427) / 0.008898883584779705   # -1.3%  LHA > 0.3467
        - 0.010938742568561822 * max(0.0, 0.063441075385 - Q.e2) / 0.03657078287722135   # -1.1%  e2 < 0.06344
        + 0.010612386387731806 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.010358830282549927   # +1.1%  tau21 < 0.238 and z_dr_0p2_0p4 < 0.2055
        + 0.010069138998154717 * max(0.0, 0.006096650059 - Q.width) / 0.002544039033917776   # +1.0%  width < 0.006097
        - 0.009516761074163013 * max(0.0, 0.000306123359 - Q.lam2) / 0.00019890943789793307   # -1.0%  lam2 < 0.0003061
        + 0.008107936702687963 * max(0.0, 0.23799610585 - Q.tau21) / 0.05620226106196825   # +0.8%  tau21 < 0.238
        - 0.0069185457093288985 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.girth2_top5 - 0.00832969537) / 0.00012177336121647686   # -0.7%  tau21 < 0.238 and girth2_top5 > 0.00833
        - 0.006532889145250851 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -0.7%  mass > 80.4
        + 0.006401457909841108 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625) / 6.205553603034812   # +0.6%  LHA > 0.1767 and sum_pt_top3 > 353.1
        - 0.005907415895469012 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) / 0.0228131875171655   # -0.6%  z_dr_0p05_0p1 > 0.7509
        + 0.005650349751298653 * max(0.0, Q.LHA - 0.176724128067) / 0.08465850240975396   # +0.6%  LHA > 0.1767
        + 0.004898174221294533 * max(0.0, 1.0 - Q.n_dr_0p1_0p2) / 0.518818487394958   # +0.5%  n_dr_0p1_0p2 < 1
        - 0.004698625872959487 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839) / 3.7602262945140036e-06   # -0.5%  width < 0.00752 and e2 > 0.02455
        + 0.004359951176222629 * max(0.0, 0.007639643088 - Q.girth2_top2) / 0.004543328598712764   # +0.4%  girth2_top2 < 0.00764
        - 0.0028610893249275864 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) / 0.012459234137161384   # -0.3%  tau21 < 0.238 and z_dr_0p05_0p1 < 0.5883
        - 0.002657341884633996 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249) / 1.9841456755321017e-05   # -0.3%  width < 0.006097 and log_sum_pt > 6.896
        + 0.0026132117033114536 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # +0.3%  log_sum_pt > 6.896
        + 0.0019442226075454883 * max(0.0, 2.0 - Q.n_dr_0_0p05) / 0.6359025210084034   # +0.2%  n_dr_0_0p05 < 2
        - 0.001939634549531358 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 76.655700683594 - Q.mass) / 1.0176955133751828   # -0.2%  tau21 < 0.238 and mass < 76.66
        + 0.0018739888126109597 * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.2904857142857143   # +0.2%  n_dr_0p05_0p1 > 5
        - 0.0017044480378013004 * max(0.0, 0.74595130682 - Q.D2) / 0.09626873727471312   # -0.2%  D2 < 0.746
        - 0.0016679820473374368 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.121680960059 - Q.max_dr) / 0.0005372705453095934   # -0.2%  tau21 < 0.238 and max_dr < 0.1217
        - 0.0015018648491016593 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.062104001645998504   # -0.2%  z_dr_0p1_0p2 < 0.3285 and n_dr_0p2_0p4 > 0
        + 0.0014879763752397661 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.08017783625220609   # +0.1%  mass > 80.4 and z_dr_0p2_0p4 < 0.2055
        + 0.0014055779878185245 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.04262928713838186   # +0.1%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 2
        - 0.0012987388533645678 * max(0.0, 0.083662731125 - Q.planar_flow) / 0.02146304529912963   # -0.1%  planar_flow < 0.08366
        - 0.0010407394001138736 * max(0.0, 0.032346998155 - Q.e2) / 0.011718069376880604   # -0.1%  e2 < 0.03235
        - 0.0009926613553715613 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136) / 0.00029995334888293545   # -0.1%  LHA > 0.1767 and z_top5 > 0.865
        - 0.0009660469858926975 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.061262048692 - Q.planar_flow) / 2.1288248516495536e-05   # -0.1%  width < 0.00752 and planar_flow < 0.06126
        - 0.0009156333218241754 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, 0.042151962757 - Q.dr_7) / 9.614998477757584e-05   # -0.1%  LHA > 0.1767 and dr_7 < 0.04215
        - 0.0006185999634899541 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 22.844978847276) / 0.011940644746868265   # -0.1%  girth2 < 0.01883 and mass_top2 > 22.84
        + 0.0005819057100107367 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.21960336377) / 0.0031277320566655276   # +0.1%  lam1 < 0.006507 and pt1_dr01 > 1.22
        + 0.000534846455973143 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, Q.mean_phi - 0.01746432744) / 3.7788165076256915e-07   # +0.1%  log_sum_pt > 6.896 and mean_phi > 0.01746
        - 0.00012925829897586546 * max(0.0, 0.74595130682 - Q.D2) * max(0.0, 23.21640625 - Q.pt_7) / 0.027757647603730364   # -0.0%  D2 < 0.746 and pt_7 < 23.22
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.8109609243697479, 0.9003199579831933, 1.759858193277311, 0.6534636554621849, 2.1004378151260505, 1.6846193277310924, 1.470729831932773, 1.5531443277310923, 0.6179550420168067, 3.233074369747899, 2.4174855042016805, 2.262026260504202, 0.06178403361344538, 4.241951890756303, 0.3825840336134454, 0.3919016806722689]
T = [2.3326397337841387, 1.6459233275341385, 3.391402893579307, 2.48781124113708, 3.5138309086134454]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +32%, n9 +24%, n1 +15%, n5 -14%, n6 +7%, n0 -5% ...
            + 0.3241773928788876 * h[2] / H_AVG[2]
            + 0.23822138037533871 * h[9] / H_AVG[9]
            + 0.15076802409459852 * h[1] / H_AVG[1]
            - 0.13541144797236396 * h[5] / H_AVG[5]
            + 0.06896095999646255 * h[6] / H_AVG[6]
            - 0.054321566505777026 * h[0] / H_AVG[0]
            - 0.028139228176571586 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +50%, n10 -18%, n4 -12%, n6 +11%, n5 +5%, n8 +2% ...
            + 0.49874606882426364 * h[9] / H_AVG[9]
            - 0.18359645493203713 * h[10] / H_AVG[10]
            - 0.1196386501569788 * h[4] / H_AVG[4]
            + 0.11169489241459428 * h[6] / H_AVG[6]
            + 0.047977041011806845 * h[5] / H_AVG[5]
            + 0.023465364078600646 * h[8] / H_AVG[8]
            + 0.014881528581718683 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +25%, n6 -14%, n7 +10%, n3 -10%, n13 +9%, n14 -8% ...
            + 0.2501206357094946 * h[11] / H_AVG[11]
            - 0.13552004491979536 * h[6] / H_AVG[6]
            + 0.10017987610212892 * h[7] / H_AVG[7]
            - 0.09634120096720733 * h[3] / H_AVG[3]
            + 0.08794656715174728 * h[13] / H_AVG[13]
            - 0.08460747195602229 * h[14] / H_AVG[14]
            + 0.08219837822273238 * h[0] / H_AVG[0]
            - 0.07944570843301495 * h[15] / H_AVG[15]
            - 0.04555305440019641 * h[8] / H_AVG[8]
            - 0.029791085643614112 * h[9] / H_AVG[9]
            - 0.008295976494046375 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +29%, n6 -22%, n3 -15%, n13 +9%, n4 +7%, n14 +6% ...
            + 0.2926413353173824 * h[7] / H_AVG[7]
            - 0.22169032676398323 * h[6] / H_AVG[6]
            - 0.1477496765508125 * h[3] / H_AVG[3]
            + 0.09324732527525104 * h[13] / H_AVG[13]
            + 0.06596027125905284 * h[4] / H_AVG[4]
            + 0.05766876933133722 * h[14] / H_AVG[14]
            + 0.045236548853465904 * h[1] / H_AVG[1]
            - 0.04061143079667226 * h[9] / H_AVG[9]
            - 0.024613860003725237 * h[15] / H_AVG[15]
            + 0.010580455848317292 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -49%, n10 +26%, n5 -12%, n4 +7%, n8 +3%, n3 +1% ...
            - 0.49043138399046016 * h[13] / H_AVG[13]
            + 0.257996781192114 * h[10] / H_AVG[10]
            - 0.11985631718942344 * h[5] / H_AVG[5]
            + 0.07472036467297176 * h[4] / H_AVG[4]
            + 0.03297442972970834 * h[8] / H_AVG[8]
            + 0.011623063126421916 * h[3] / H_AVG[3]
            - 0.008791549055760527 * h[12] / H_AVG[12]
            + 0.003606111043139916 * h[0] / H_AVG[0]
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
