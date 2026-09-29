"""JEDI-linear jet tagger, 8 particles, 3 features: the tuned formula (start), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  17.1%   (on for 93% of jets)
  neuron  9:  12.8%   (on for 68% of jets)
  neuron 10:   9.6%   (on for 98% of jets)
  neuron  6:   8.9%   (on for 45% of jets)
  neuron  4:   7.7%   (on for 90% of jets)
  neuron  5:   7.2%   (on for 82% of jets)
  neuron  7:   6.6%   (on for 56% of jets)
  neuron  2:   6.1%   (on for 94% of jets)
  neuron 11:   5.8%   (on for 81% of jets)
  neuron  3:   4.1%   (on for 19% of jets)
  neuron 15:   3.5%   (on for 33% of jets)
  neuron  0:   3.3%   (on for 46% of jets)
  neuron 14:   3.3%   (on for 28% of jets)
  neuron  8:   2.1%   (on for 43% of jets)
  neuron  1:   1.6%   (on for 43% of jets)
  neuron 12:   0.2%   (on for 4% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

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
    # scale S = 27.97;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.972164356575842 * (-0.024508311267739585
        + 0.12109125435045387 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741758945   # +12.1%  girth2 < 0.01324
        + 0.12008178608492921 * max(0.0, 0.008678044951 - Q.width) / 0.004447347549975801   # +12.0%  width < 0.008678
        - 0.059664931582901606 * max(0.0, 0.13092863437 - Q.mass_over_sum_pt) / 0.07219748739649169   # -6.0%  mass_over_sum_pt < 0.1309
        - 0.05108241108897399 * max(0.0, 0.008174660116 - Q.mass_over_sum_pt_sq) / 0.004299658381533023   # -5.1%  mass_over_sum_pt_sq < 0.008175
        - 0.04822746762540378 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, Q.phi_1 - -0.058901977539) / 0.4331515118014987   # -4.8%  mass < 29.64 and phi_1 > -0.0589
        - 0.04581812864028042 * max(0.0, 0.076081777364 - Q.girth) / 0.027967843803613616   # -4.6%  girth < 0.07608
        - 0.0446091812559878 * max(0.0, 21.784077072144 - Q.mass) / 4.431023211632268   # -4.5%  mass < 21.78
        - 0.037902536797027694 * max(0.0, 0.008375572068 - Q.lam1) / 0.0043001452793647   # -3.8%  lam1 < 0.008376
        - 0.03632465940596532 * max(0.0, 0.004372139331 - Q.width) / 0.0015657102477179045   # -3.6%  width < 0.004372
        - 0.03505980073777901 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -3.5%  girth < 0.08724
        + 0.03141416764632467 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) / 0.0544885966340469   # +3.1%  z_dr_0_0p05 > 0.8477
        - 0.03081519010193054 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # -3.1%  mass < 29.64
        + 0.028698803095026953 * max(0.0, 0.018827652745 - Q.girth2) / 0.013142955535857035   # +2.9%  girth2 < 0.01883
        + 0.02559405988374419 * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.016491452412822304   # +2.6%  centroid_offset < 0.03117
        + 0.02512879698260702 * max(0.0, 0.006506575659 - Q.lam1) / 0.00289007769049566   # +2.5%  lam1 < 0.006507
        - 0.021356054383960817 * max(0.0, 0.007929074034 - Q.girth2_top3) / 0.004560262122333382   # -2.1%  girth2_top3 < 0.007929
        - 0.01969981010326053 * max(0.0, 64.618731689453 - Q.mass) / 27.572788342921132   # -2.0%  mass < 64.62
        - 0.019658193248845178 * max(0.0, 0.00832969537 - Q.girth2_top5) / 0.004544146438161959   # -2.0%  girth2_top5 < 0.00833
        + 0.019008389893944187 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, Q.centroid_offset - 0.012587644117) / 0.1602431142445217   # +1.9%  mass < 64.62 and centroid_offset > 0.01259
        - 0.01786880850240391 * max(0.0, 64.618731689453 - Q.mass) * max(0.0, 40.040625 - Q.pt_7) / 229.29996254831684   # -1.8%  mass < 64.62 and pt_7 < 40.04
        + 0.015382645166184657 * max(0.0, 0.003952581551 - Q.girth2_top3) / 0.0017731449074353464   # +1.5%  girth2_top3 < 0.003953
        - 0.014995812443536989 * max(0.0, 0.00543336053 - Q.lam1) / 0.002194140033981738   # -1.5%  lam1 < 0.005433
        + 0.014894448986013465 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.eccentricity - 0.959856212153) / 0.0001670441522630475   # +1.5%  girth2 < 0.01883 and eccentricity > 0.9599
        - 0.013960386133750013 * max(0.0, Q.sum_pt - 901.59375) / 12.403615355829832   # -1.4%  sum_pt > 901.6
        + 0.0130392334216136 * max(0.0, Q.sum_pt_top5 - 687.4375) / 37.14619112969407   # +1.3%  sum_pt_top5 > 687.4
        - 0.011207349566891074 * max(0.0, 56.920347213745 - Q.mass) / 21.784746347219   # -1.1%  mass < 56.92
        - 0.011097397624355874 * max(0.0, 0.020459658932 - Q.e2) / 0.00553192518950076   # -1.1%  e2 < 0.02046
        - 0.009786962974369692 * max(0.0, Q.log_sum_pt - 6.638338705138) / 0.05708668543535571   # -1.0%  log_sum_pt > 6.638
        + 0.006683509996026337 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.049903668404 - Q.centroid_offset) / 0.0017323178375765194   # +0.7%  planar_flow < 0.1484 and centroid_offset < 0.0499
        + 0.00604873859200825 * max(0.0, 0.148419710734 - Q.planar_flow) / 0.051103617612443746   # +0.6%  planar_flow < 0.1484
        - 0.0056059392781424655 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 380.5875 - Q.sum_pt_top2) / 3.055978453307198   # -0.6%  planar_flow < 0.1484 and sum_pt_top2 < 380.6
        - 0.005526030580647682 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, 0.87567204833 - Q.D2) / 7.962190634474389e-05   # -0.6%  lam1 < 0.006507 and D2 < 0.8757
        + 0.0053104815240418105 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 25.578125 - Q.pt_7) / 68.64747957984208   # +0.5%  sum_pt > 901.6 and pt_7 < 25.58
        - 0.005227960863927747 * max(0.0, Q.sum_pt_top5 - 687.4375) * max(0.0, Q.z_7 - 0.023207568189) / 0.19805534351481954   # -0.5%  sum_pt_top5 > 687.4 and z_7 > 0.02321
        + 0.005134175085403818 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # +0.5%  log_sum_pt > 6.378
        + 0.0025184065638364527 * max(0.0, 56.920347213745 - Q.mass) * max(0.0, Q.C2 - 0.023843882605) / 0.07430351940037334   # +0.3%  mass < 56.92 and C2 > 0.02384
        - 0.0023583150540277278 * max(0.0, 0.148419710734 - Q.planar_flow) * max(0.0, 0.111761856824 - Q.max_dr) / 0.00044316723190441983   # -0.2%  planar_flow < 0.1484 and max_dr < 0.1118
        - 0.0022507787838348305 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.87567204833 - Q.D2) / 0.021060647298465133   # -0.2%  mass < 29.64 and D2 < 0.8757
        - 0.002092980417296823 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.centroid_offset - 0.014379521101) / 3.0345989167714716e-05   # -0.2%  girth2 < 0.01324 and centroid_offset > 0.01438
        - 0.0020099809646286562 * max(0.0, 0.007929074034 - Q.girth2_top3) * max(0.0, 35.28125 - Q.pt_6) / 0.01647839305985912   # -0.2%  girth2_top3 < 0.007929 and pt_6 < 35.28
        + 0.0018527591828790713 * max(0.0, 0.012569162668 - Q.planar_flow) / 0.00047372042986104206   # +0.2%  planar_flow < 0.01257
        - 0.0015545918793695107 * max(0.0, Q.sum_pt - 901.59375) * max(0.0, 0.038438041256 - Q.dr_2) / 0.2822418354616529   # -0.2%  sum_pt > 901.6 and dr_2 < 0.03844
        + 0.0015183429284522669 * max(0.0, Q.log_sum_pt - 6.638338705138) * max(0.0, 0.093979107928 - Q.dr_7) / 0.002872406160327696   # +0.2%  log_sum_pt > 6.638 and dr_7 < 0.09398
        - 0.0008383405770104536 * max(0.0, 0.004372139331 - Q.width) * max(0.0, Q.n_dr_0p1_0p2 - 1.0) / 4.107866360800222e-05   # -0.1%  width < 0.004372 and n_dr_0p1_0p2 > 1
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 15.45;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.453946284927408 * (0.053037418577847995
        - 0.13583568054861642 * max(0.0, 0.008678044951 - Q.width) / 0.004447347549975801   # -13.6%  width < 0.008678
        + 0.13031445965785815 * max(0.0, 0.008168570676 - Q.e2_sq) / 0.004294747328227963   # +13.0%  e2_sq < 0.008169
        - 0.05629201083440356 * max(0.0, 0.055577157257 - Q.z_7) / 0.010554877533196528   # -5.6%  z_7 < 0.05558
        + 0.045773713689197054 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # +4.6%  log_sum_pt > 6.378
        - 0.042512213214287296 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.0029084479897071897   # -4.3%  log_sum_pt > 6.378 and centroid_offset < 0.02355
        + 0.03973181225130407 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.001130644719 - Q.lam2) / 9.286869338660042e-05   # +4.0%  log_sum_pt > 6.573 and lam2 < 0.001131
        - 0.03913664168859385 * max(0.0, 0.042322802544 - Q.C2) / 0.020126650964483378   # -3.9%  C2 < 0.04232
        - 0.039044845302991925 * max(0.0, 0.008375572068 - Q.lam1) / 0.0043001452793647   # -3.9%  lam1 < 0.008376
        + 0.036664350160049125 * max(0.0, Q.log_sum_pt - 6.572937922293) / 0.08631512213470326   # +3.7%  log_sum_pt > 6.573
        - 0.03661544534957612 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006756161242 - Q.girth2_top3) / 0.0004453888235688722   # -3.7%  log_sum_pt > 6.573 and girth2_top3 < 0.006756
        - 0.03275732307621143 * max(0.0, 0.346713497427 - Q.LHA) / 0.11284739233372   # -3.3%  LHA < 0.3467
        + 0.03255831548492603 * max(0.0, Q.mass_over_sum_pt - 0.054892207095) / 0.022123336411487685   # +3.3%  mass_over_sum_pt > 0.05489
        + 0.032421380646003924 * max(0.0, Q.pt_7 - 34.53125) / 4.473662132352941   # +3.2%  pt_7 > 34.53
        + 0.029191858946723617 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.007929074034 - Q.girth2_top3) / 6.301690249224892e-05   # +2.9%  z_7 < 0.05558 and girth2_top3 < 0.007929
        - 0.025161890385621152 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 91.19 - Q.mass) / 238.71814489127775   # -2.5%  pt_7 > 34.53 and mass < 91.19
        + 0.02356934119774816 * max(0.0, 0.076081777364 - Q.girth) / 0.027967843803613616   # +2.4%  girth < 0.07608
        + 0.02329442785947115 * max(0.0, Q.log_sum_pt - 6.377722943814) * max(0.0, 0.197968879342 - Q.max_dr) / 0.021256915339589413   # +2.3%  log_sum_pt > 6.378 and max_dr < 0.198
        - 0.022359854728142683 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, 0.083662731125 - Q.planar_flow) / 5.3599394412614046e-05   # -2.2%  e2_sq < 0.008169 and planar_flow < 0.08366
        - 0.01971363153354538 * max(0.0, 53.332374954224 - Q.mass) / 19.36003866872732   # -2.0%  mass < 53.33
        - 0.019188132551309568 * max(0.0, 0.046566883102 - Q.max_dr) / 0.005138021564075457   # -1.9%  max_dr < 0.04657
        - 0.017131528221454542 * max(0.0, Q.e2 - 0.032346998155) / 0.00800322490328291   # -1.7%  e2 > 0.03235
        + 0.015188446873900096 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, Q.eccentricity - 0.978160776925) / 3.5352354845618814e-05   # +1.5%  e2 < 0.03556 and eccentricity > 0.9782
        + 0.012191363456516923 * max(0.0, 0.055577157257 - Q.z_7) * max(0.0, 0.042322802544 - Q.C2) / 0.00023050093654597824   # +1.2%  z_7 < 0.05558 and C2 < 0.04232
        - 0.010764993204809283 * max(0.0, Q.e2 - 0.032346998155) * max(0.0, 0.641386964917 - Q.tau32) / 0.002183834013729681   # -1.1%  e2 > 0.03235 and tau32 < 0.6414
        + 0.008484840064783752 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 0.000194798295 - Q.lam2) / 5.097887295163036e-06   # +0.8%  tau21 < 0.1978 and lam2 < 0.0001948
        - 0.007749147420454391 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 0.083662731125 - Q.planar_flow) / 5.689235544083617e-05   # -0.8%  width < 0.008678 and planar_flow < 0.08366
        + 0.007524081749332554 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.centroid_offset - 0.02076709205) / 7.4336756001124556e-06   # +0.8%  lam1 < 0.008376 and centroid_offset > 0.02077
        - 0.007017823082329803 * max(0.0, 0.197783735394 - Q.tau21) / 0.038939428447455314   # -0.7%  tau21 < 0.1978
        - 0.006704746954178734 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0013871589348363625   # -0.7%  lam1 < 0.008376 and n_pt_above_50 > 6
        - 0.0066743882546839335 * max(0.0, 0.346713497427 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow) / 0.0013308621908605532   # -0.7%  LHA < 0.3467 and planar_flow < 0.08366
        + 0.006664660933454248 * max(0.0, 0.008168570676 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0013761153368793523   # +0.7%  e2_sq < 0.008169 and n_pt_above_50 > 6
        + 0.005112811195880157 * max(0.0, 0.083662731125 - Q.planar_flow) / 0.02146304529912963   # +0.5%  planar_flow < 0.08366
        - 0.005010884957918865 * max(0.0, 0.197783735394 - Q.tau21) * max(0.0, 50.25 - Q.pt_6) / 0.37991528419887   # -0.5%  tau21 < 0.1978 and pt_6 < 50.25
        + 0.004048670973425527 * max(0.0, 0.035560912266 - Q.e2) / 0.01371972344513943   # +0.4%  e2 < 0.03556
        - 0.0036961159916288323 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.n_pt_above_50 - 7.0) / 0.008178701618214594   # -0.4%  log_sum_pt > 6.573 and n_pt_above_50 > 7
        + 0.0031569524461955312 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 45.595 - Q.mass_top3) / 11.461427234693382   # +0.3%  pt_7 > 53.44 and mass_top3 < 45.59
        + 0.0028081127160006544 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.033200121667 - Q.planar_flow) / 0.000428276800320312   # +0.3%  log_sum_pt > 6.573 and planar_flow < 0.0332
        + 0.0020045683977494757 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, Q.centroid_offset - 0.014379521101) / 0.023463745833839495   # +0.2%  pt_7 > 34.53 and centroid_offset > 0.01438
        - 0.0019716807340002768 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 32.617988451746 - Q.mass_top3) / 7.427128319785132   # -0.2%  pt_7 > 53.44 and mass_top3 < 32.62
        + 0.0019099822833238173 * max(0.0, Q.pt_7 - 34.53125) * max(0.0, 0.080507021025 - Q.max_dr) / 0.08998780021469473   # +0.2%  pt_7 > 34.53 and max_dr < 0.08051
        - 0.0013714529262098506 * max(0.0, 0.083662731125 - Q.planar_flow) * max(0.0, 0.269169217348 - Q.tau32) / 5.796886858637436e-05   # -0.1%  planar_flow < 0.08366 and tau32 < 0.2692
        - 0.0006754180551881107 * max(0.0, Q.pt_7 - 53.4375) * max(0.0, 0.553068161011 - Q.tau21) / 0.07407243315184071   # -0.1%  pt_7 > 53.44 and tau21 < 0.5531
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 11.87;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.873373823806498 * (0.2362730837864581
        - 0.11777569506732433 * max(0.0, 53.4375 - Q.pt_7) / 19.120943117285975   # -11.8%  pt_7 < 53.44
        + 0.10664761128166464 * max(0.0, 788.4484375 - Q.sum_pt) / 112.15755791313566   # +10.7%  sum_pt < 788.4
        + 0.08974018737174297 * max(0.0, Q.pt_7 - 30.484375) / 6.808825630252101   # +9.0%  pt_7 > 30.48
        - 0.07095419272836698 * max(0.0, 36.229410171509 - Q.mass) / 10.113929790442333   # -7.1%  mass < 36.23
        - 0.06757513718606048 * max(0.0, 687.4375 - Q.sum_pt_top5) / 131.13431972163866   # -6.8%  sum_pt_top5 < 687.4
        - 0.0664615436900694 * max(0.0, Q.LHA - 0.111565049159) / 0.1352185061485581   # -6.6%  LHA > 0.1116
        + 0.059115200360378574 * max(0.0, 0.005954149834 - Q.lam1) / 0.002518154081527821   # +5.9%  lam1 < 0.005954
        + 0.055072201247465175 * max(0.0, 69.611351776123 - Q.mass) / 31.697292539607695   # +5.5%  mass < 69.61
        + 0.044240660413503316 * max(0.0, 36.229410171509 - Q.mass) * max(0.0, 0.001130644719 - Q.lam2) / 0.01085576403198656   # +4.4%  mass < 36.23 and lam2 < 0.001131
        + 0.039739841936767405 * max(0.0, 43.5 - Q.pt_7) / 10.288609083672531   # +4.0%  pt_7 < 43.5
        - 0.03435493215171168 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, 0.051192347892 - Q.C2) / 0.21065871608184009   # -3.4%  pt_7 > 30.48 and C2 < 0.05119
        - 0.03326215806406785 * max(0.0, Q.z_7 - 0.046240320761) / 0.012049927164073902   # -3.3%  z_7 > 0.04624
        - 0.03240194613797369 * max(0.0, 0.694781820497 - Q.planar_flow) / 0.4352329688894335   # -3.2%  planar_flow < 0.6948
        + 0.023027276304883836 * max(0.0, 6.464150123592 - Q.log_sum_pt) / 0.0700995140842757   # +2.3%  log_sum_pt < 6.464
        + 0.019735954421958683 * max(0.0, Q.log_sum_pt - 6.267538488641) / 0.29831431617346454   # +2.0%  log_sum_pt > 6.268
        - 0.016320519951626603 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.068101508468 - Q.z_7) / 0.6443998809685784   # -1.6%  mass < 69.61 and z_7 < 0.0681
        + 0.012389493922370542 * max(0.0, 0.003377388461 - Q.lam1) / 0.0011333798053217976   # +1.2%  lam1 < 0.003377
        - 0.011541510189764275 * max(0.0, 0.028070914944 - Q.z_7) * max(0.0, 0.00752008842 - Q.width) / 7.682811042932545e-06   # -1.2%  z_7 < 0.02807 and width < 0.00752
        - 0.01092385737051716 * max(0.0, Q.pt_7 - 30.484375) * max(0.0, Q.max_dr - 0.093110798299) / 0.24796742071553304   # -1.1%  pt_7 > 30.48 and max_dr > 0.09311
        + 0.010121846368219271 * max(0.0, 8.379955863953 - Q.mass) * max(0.0, 0.010960638421 - Q.centroid_offset) / 0.0026530686527581372   # +1.0%  mass < 8.38 and centroid_offset < 0.01096
        - 0.009141707859901517 * max(0.0, 0.15984864831 - Q.max_dr) / 0.05771655139763449   # -0.9%  max_dr < 0.1598
        - 0.007672856370775871 * max(0.0, 0.005954149834 - Q.lam1) * max(0.0, Q.max_dr - 0.080507021025) / 4.4933272160277494e-05   # -0.8%  lam1 < 0.005954 and max_dr > 0.08051
        + 0.007598062560913257 * max(0.0, 0.028070914944 - Q.z_7) / 0.0013098148516826897   # +0.8%  z_7 < 0.02807
        - 0.0069711810474670155 * max(0.0, 0.016858545121 - Q.z_7) / 0.0002571522772302325   # -0.7%  z_7 < 0.01686
        - 0.006455841016629759 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # -0.6%  log_sum_pt > 6.896
        - 0.006190829144852076 * max(0.0, 0.003377388461 - Q.lam1) * max(0.0, 0.006789738266 - Q.centroid_offset) / 2.260502962862742e-06   # -0.6%  lam1 < 0.003377 and centroid_offset < 0.00679
        - 0.006070728247447403 * max(0.0, 0.000172198326 - Q.width) * max(0.0, 0.222994708167 - Q.dr_7) / 2.998515264059948e-06   # -0.6%  width < 0.0001722 and dr_7 < 0.223
        + 0.005663786009444125 * max(0.0, Q.log_sum_pt - 6.842716632804) / 0.007875918226781813   # +0.6%  log_sum_pt > 6.843
        - 0.00529585692799577 * max(0.0, 4.8108519e-05 - Q.girth2) / 8.325084578958415e-07   # -0.5%  girth2 < 4.811e-05
        - 0.005179036392270009 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, Q.max_pair_mass - 13.047927274731) / 31.36860231006198   # -0.5%  mass < 69.61 and max_pair_mass > 13.05
        + 0.003869005679275899 * max(0.0, Q.log_sum_pt - 6.804164030582) / 0.012576817653163173   # +0.4%  log_sum_pt > 6.804
        - 0.0033851482690925587 * max(0.0, 9.1213921e-05 - Q.width) / 4.152702847822947e-06   # -0.3%  width < 9.121e-05
        - 0.0028968723536716167 * max(0.0, 56.53125 - Q.pt_6) * max(0.0, Q.m012 - 32.617988451746) / 48.99228382620799   # -0.3%  pt_6 < 56.53 and m012 > 32.62
        + 0.002207321953826199 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.04881348081 - Q.dr_7) / 0.15471315475506517   # +0.2%  sum_pt_top5 > 840 and dr_7 < 0.04881
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 24.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.273038673391433 * (0.1260960194531668
        - 0.22840926513694004 * max(0.0, 0.008678044751 - Q.girth2) / 0.00444734738961668   # -22.8%  girth2 < 0.008678
        - 0.15891558702904343 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741758945   # -15.9%  girth2 < 0.01324
        + 0.12981269621995523 * max(0.0, 0.04447356835 - Q.e2) / 0.020193337934351147   # +13.0%  e2 < 0.04447
        + 0.04980688383054662 * max(0.0, 0.007330079875 - Q.lam1) / 0.0034868526190745728   # +5.0%  lam1 < 0.00733
        + 0.04204516382334186 * max(0.0, 0.004372139461 - Q.girth2) / 0.0015657103124088927   # +4.2%  girth2 < 0.004372
        + 0.03875234589120184 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 5.0 - Q.n_dr_0_0p05) / 0.06729405587917689   # +3.9%  mass_over_sum_pt > 0.06814 and n_dr_0_0p05 < 5
        - 0.0362811776837787 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) / 0.015391574669651855   # -3.6%  mass_over_sum_pt > 0.06814
        - 0.023124095305072048 * max(0.0, Q.e2 - 0.028531698044) / 0.009633336184298937   # -2.3%  e2 > 0.02853
        + 0.022604990739407464 * max(0.0, Q.mass - 36.229410171509) * max(0.0, Q.eccentricity - 0.620723099573) / 4.350681513338082   # +2.3%  mass > 36.23 and eccentricity > 0.6207
        - 0.021577625304232784 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, 0.964120104909 - Q.z_dr_0p05_0p1) / 0.01720990725856626   # -2.2%  e2 < 0.04447 and z_dr_0p05_0p1 < 0.9641
        + 0.02029409693185037 * max(0.0, Q.centroid_offset - 0.014379521101) / 0.007111299972137995   # +2.0%  centroid_offset > 0.01438
        + 0.017235119952189022 * max(0.0, Q.max_dr - 0.145231109113) / 0.026281229857574792   # +1.7%  max_dr > 0.1452
        - 0.016680198067742673 * max(0.0, Q.lam1 - 0.012003726523) / 0.0012289213133878476   # -1.7%  lam1 > 0.012
        - 0.015798558551445137 * max(0.0, Q.mass - 36.229410171509) / 14.205281284755415   # -1.6%  mass > 36.23
        + 0.014099821775147483 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.008298543275912581   # +1.4%  mass_over_sum_pt > 0.09041
        - 0.013297506634982236 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.518696343899 - Q.tau32) / 0.0026929406238906054   # -1.3%  mass_over_sum_pt > 0.06814 and tau32 < 0.5187
        + 0.011687344650582663 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.001130644719 - Q.lam2) / 0.010169263818483524   # +1.2%  mass > 36.23 and lam2 < 0.001131
        + 0.010093240039074159 * max(0.0, 0.006679471358 - Q.width) / 0.002936239797733741   # +1.0%  width < 0.006679
        + 0.0097739879734283 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.959856212153) / 0.0001857344023569209   # +1.0%  LHA > 0.3127 and eccentricity > 0.9599
        - 0.009577387329686647 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 56.53125 - Q.pt_6) / 0.14624935923838925   # -1.0%  mass_over_sum_pt > 0.09041 and pt_6 < 56.53
        + 0.009088759938396614 * max(0.0, Q.LHA - 0.312727471086) / 0.01533444198816819   # +0.9%  LHA > 0.3127
        - 0.008976970242915593 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481254519   # -0.9%  mass_over_sum_pt > 0.108
        - 0.007857734112605194 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.eccentricity - 0.959856212153) / 0.0001389130551637596   # -0.8%  e2 > 0.02853 and eccentricity > 0.9599
        - 0.006938447649047406 * max(0.0, Q.mass_top5 - 53.607658247923) / 1.9788142552051857   # -0.7%  mass_top5 > 53.61
        - 0.006530991377664576 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 50.352200171245 - Q.mass_top3) / 0.2821292880184888   # -0.7%  LHA > 0.3127 and mass_top3 < 50.35
        - 0.006517108867682806 * max(0.0, Q.mass - 69.611351776123) / 2.4067024290003474   # -0.7%  mass > 69.61
        - 0.005465783826329787 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.0) / 0.0005523260037536467   # -0.5%  e2 < 0.04447 and z_dr_0p1_0p2 > 0
        + 0.004526564236743145 * max(0.0, Q.lam1 - 0.012003726523) * max(0.0, 38.25 - Q.pt_6) / 0.003850488254637342   # +0.5%  lam1 > 0.012 and pt_6 < 38.25
        - 0.00446194928794542 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 0.145231109113 - Q.max_dr) / 4.2402205610446514e-05   # -0.4%  LHA > 0.3127 and max_dr < 0.1452
        + 0.004317594711591236 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 48.71875 - Q.pt_7) / 0.11670007858083505   # +0.4%  mass_over_sum_pt > 0.09041 and pt_7 < 48.72
        + 0.004159790246214378 * max(0.0, Q.LHA - 0.325582223496) / 0.012453043826768326   # +0.4%  LHA > 0.3256
        - 0.003799665159271731 * max(0.0, Q.mass_over_sum_pt - 0.06813910019) * max(0.0, 0.042151962757 - Q.dr_7) / 1.0188469566612644e-05   # -0.4%  mass_over_sum_pt > 0.06814 and dr_7 < 0.04215
        - 0.0037560079064777106 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.planar_flow - 0.00804883781) / 0.005147612196763805   # -0.4%  LHA > 0.3127 and planar_flow > 0.008049
        + 0.0036137852809486767 * max(0.0, Q.width - 0.018827653081) / 0.0007605368512987271   # +0.4%  width > 0.01883
        - 0.0034975237076321176 * max(0.0, Q.lam1 - 0.016433749775) * max(0.0, Q.eccentricity - 0.959856212153) / 8.795584178662102e-06   # -0.3%  lam1 > 0.01643 and eccentricity > 0.9599
        - 0.003324584305609439 * max(0.0, Q.max_dr - 0.145231109113) * max(0.0, 0.04586879935 - Q.dr_3) / 0.00015061771857077786   # -0.3%  max_dr > 0.1452 and dr_3 < 0.04587
        - 0.0030299667209141125 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 68.125 - Q.pt_4) / 0.035301767581221215   # -0.3%  mean_eta < -0.01284 and pt_4 < 68.12
        + 0.002929953843379657 * max(0.0, Q.width - 0.018827653081) * max(0.0, 0.492494773865 - Q.pt_dispersion) / 7.142878552670272e-05   # +0.3%  width > 0.01883 and pt_dispersion < 0.4925
        - 0.0027871352148428805 * max(0.0, Q.e2 - 0.028531698044) * max(0.0, Q.n_pt_above_50 - 3.0) / 0.0180163152422735   # -0.3%  e2 > 0.02853 and n_pt_above_50 > 3
        + 0.002478364134554624 * max(0.0, -0.012844925793 - Q.mean_eta) * max(0.0, 0.120257140434 - Q.z_4) / 5.864010154098662e-05   # +0.2%  mean_eta < -0.01284 and z_4 < 0.1203
        + 0.001850236165189996 * max(0.0, Q.mass - 69.611351776123) * max(0.0, 0.177304983139 - Q.max_dr) / 0.013287347868038578   # +0.2%  mass > 69.61 and max_dr < 0.1773
        - 0.0016874031934315894 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, -0.039672851562 - Q.eta_0) / 0.00010003119524352961   # -0.2%  centroid_offset > 0.01438 and eta_0 < -0.03967
        - 0.0016628076438576012 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 0.046481671275 - Q.dr_6) / 0.024066966560401745   # -0.2%  mass > 36.23 and dr_6 < 0.04648
        + 0.001456976900904601 * max(0.0, -0.012844925793 - Q.mean_eta) / 0.001988793735763939   # +0.1%  mean_eta < -0.01284
        + 0.0014250150983140292 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) * max(0.0, 0.145231109113 - Q.max_dr) / 3.112203819684481e-06   # +0.1%  mass_over_sum_pt > 0.09041 and max_dr < 0.1452
        - 0.0013891264339127344 * max(0.0, Q.LHA - 0.423592510895) / 0.0013397344097334583   # -0.1%  LHA > 0.4236
        + 0.0011355124306399228 * max(0.0, Q.centroid_offset - 0.014379521101) * max(0.0, Q.phi_7 - -0.041534423828) / 0.0004416110244511484   # +0.1%  centroid_offset > 0.01438 and phi_7 > -0.04153
        - 0.0009049506712840508 * max(0.0, Q.mass - 36.229410171509) * max(0.0, 20.125 - Q.pt_7) / 3.7708902995959526   # -0.1%  mass > 36.23 and pt_7 < 20.12
        + 0.00056219782204963 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, 29.875 - Q.pt_5) / 0.0027959887902143485   # +0.1%  LHA > 0.3127 and pt_5 < 29.88
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 38.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 38.74152039216561 * (-0.18090912741152998
        + 0.1348547972968518 * max(0.0, Q.width - 0.001653836415) / 0.005220304336956379   # +13.5%  width > 0.001654
        + 0.1250266899144102 * max(0.0, 0.017162483186 - Q.e2_sq) / 0.012035399208639285   # +12.5%  e2_sq < 0.01716
        - 0.07865196421475702 * max(0.0, Q.width - 0.000319370692) / 0.006166221167145921   # -7.9%  width > 0.0003194
        - 0.0609174025758651 * max(0.0, 69.611351776123 - Q.mass) / 31.697292539607695   # -6.1%  mass < 69.61
        + 0.05134071595220564 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) / 0.005364315481254519   # +5.1%  mass_over_sum_pt > 0.108
        + 0.049496040845810454 * max(0.0, 0.023780909279 - Q.e2_sq) / 0.018171241308908473   # +4.9%  e2_sq < 0.02378
        - 0.04945916721074214 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.008298543275912581   # -4.9%  mass_over_sum_pt > 0.09041
        + 0.0428038241417192 * max(0.0, Q.e2 - 0.007078157854) / 0.022325491981794805   # +4.3%  e2 > 0.007078
        + 0.042655404180444056 * max(0.0, 56.920347213745 - Q.mass) / 21.784746347219   # +4.3%  mass < 56.92
        + 0.03994990227811267 * max(0.0, 0.23799610585 - Q.tau21) / 0.05620226106196825   # +4.0%  tau21 < 0.238
        - 0.025898575364854095 * max(0.0, Q.mass_over_sum_pt - 0.107985668755) * max(0.0, 3.885568320751 - Q.D2) / 0.015907178094692793   # -2.6%  mass_over_sum_pt > 0.108 and D2 < 3.886
        + 0.025334377696540126 * max(0.0, 0.005011406868 - Q.girth2_top3) / 0.002432330601058231   # +2.5%  girth2_top3 < 0.005011
        - 0.024926782074336015 * max(0.0, 0.000306123359 - Q.lam2) / 0.00019890943789781996   # -2.5%  lam2 < 0.0003061
        - 0.020123706198105034 * max(0.0, Q.C2 - 0.014943876117) / 0.015904628092890307   # -2.0%  C2 > 0.01494
        + 0.02002769604497597 * max(0.0, 763.825 - Q.sum_pt) / 96.68366216443064   # +2.0%  sum_pt < 763.8
        - 0.017087724239794374 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.001130644719 - Q.lam2) / 5.6283236502119514e-05   # -1.7%  tau21 < 0.238 and lam2 < 0.001131
        + 0.016781160599523254 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.067292226106 - Q.C2) / 0.0001625902207448854   # +1.7%  width > 0.0003194 and C2 < 0.06729
        + 0.01673403955641412 * max(0.0, Q.max_dr - 0.102758520097) / 0.04505724462361834   # +1.7%  max_dr > 0.1028
        - 0.013900076327202944 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 50.352200171245 - Q.mass_top3) / 0.007657330390013242   # -1.4%  lam2 < 0.0003061 and mass_top3 < 50.35
        + 0.011560695575966872 * max(0.0, 0.014379521101 - Q.centroid_offset) / 0.004306482932849425   # +1.2%  centroid_offset < 0.01438
        - 0.010886265692272537 * max(0.0, Q.C2 - 0.067292226106) / 0.002854902335729804   # -1.1%  C2 > 0.06729
        + 0.010504748509315029 * max(0.0, 0.047915700823 - Q.girth) / 0.012161121217552112   # +1.1%  girth < 0.04792
        + 0.009817474228226476 * max(0.0, 69.611351776123 - Q.mass) * max(0.0, 0.104247858869 - Q.dr_3) / 2.3095576214583597   # +1.0%  mass < 69.61 and dr_3 < 0.1042
        - 0.007730528326144109 * max(0.0, 0.001130644719 - Q.lam2) / 0.0009137232342239988   # -0.8%  lam2 < 0.001131
        + 0.007721609814042792 * max(0.0, Q.C2 - 0.014943876117) * max(0.0, Q.pt_7 - 38.53125) / 0.023902004294371636   # +0.8%  C2 > 0.01494 and pt_7 > 38.53
        - 0.007711485319804499 * max(0.0, 0.014379521101 - Q.centroid_offset) * max(0.0, 0.674770402908 - Q.z_dr_0p05_0p1) / 0.002138030313460975   # -0.8%  centroid_offset < 0.01438 and z_dr_0p05_0p1 < 0.6748
        + 0.007687761577140404 * max(0.0, 0.001101860861 - Q.mass_over_sum_pt_sq) / 0.0003082102790481083   # +0.8%  mass_over_sum_pt_sq < 0.001102
        - 0.00719434089894134 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 62.55 - Q.mass) / 0.42797494392630053   # -0.7%  tau21 < 0.238 and mass < 62.55
        - 0.007153496246849803 * max(0.0, Q.max_dr - 0.197968879342) / 0.012226149933645198   # -0.7%  max_dr > 0.198
        - 0.007074468253413443 * max(0.0, Q.girth2 - 0.013238675334) / 0.0014426835012777079   # -0.7%  girth2 > 0.01324
        + 0.007019185688498786 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_7 - 33.21875) / 0.3246332585626503   # +0.7%  tau21 < 0.238 and pt_7 > 33.22
        - 0.0059390672418860625 * max(0.0, Q.max_dr - 0.102758520097) * max(0.0, Q.pt_7 - 37.15625) / 0.08633888537871398   # -0.6%  max_dr > 0.1028 and pt_7 > 37.16
        + 0.005572360374123693 * max(0.0, 0.009530300104 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.016278845848) / 1.967884659033273e-05   # +0.6%  girth2_top2 < 0.00953 and centroid_offset > 0.01628
        - 0.0054488196943270365 * max(0.0, Q.e2 - 0.063441075385) / 0.0017618611738475503   # -0.5%  e2 > 0.06344
        - 0.005025948540971251 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 68.70525244030448   # -0.5%  sum_pt < 763.8 and n_dr_0p2_0p4 < 1
        + 0.004373648569919567 * max(0.0, 430.75 - Q.sum_pt_top5) / 14.052461580882353   # +0.4%  sum_pt_top5 < 430.8
        - 0.0033744550392170725 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.e2_sq - 0.011657374702) / 7.209993947021868e-05   # -0.3%  tau21 < 0.238 and e2_sq > 0.01166
        - 0.0033342116933557267 * max(0.0, Q.width - 0.000319370692) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.00017651898220290152   # -0.3%  width > 0.0003194 and z_dr_0p2_0p4 < 0.05644
        + 0.00279401342465816 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.175465903809 - Q.dr_7) / 0.004744492327402142   # +0.3%  tau21 < 0.238 and dr_7 < 0.1755
        - 0.0019846886285910122 * max(0.0, 0.001130644719 - Q.lam2) * max(0.0, Q.n_dr_0p1_0p2 - 2.0) / 0.0003045640628348002   # -0.2%  lam2 < 0.001131 and n_dr_0p1_0p2 > 2
        + 0.001321309104837421 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.pt_2 - 73.6875) / 1.3957644344475406   # +0.1%  tau21 < 0.238 and pt_2 > 73.69
        + 0.0012870418706913739 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, Q.planar_flow - 0.045057236346) / 0.0019149421885925775   # +0.1%  tau21 < 0.238 and planar_flow > 0.04506
        + 0.0010616838556444217 * max(0.0, Q.girth - 0.101940929517) / 0.005400003353656977   # +0.1%  girth > 0.1019
        - 0.00045064511849702443 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, -0.025945045147 - Q.mean_phi) / 3.862186358450691e-05   # -0.0%  tau21 < 0.238 and mean_phi < -0.02595
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 16.05;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.053128720370946 * (0.0241785789834099
        + 0.09656356446069324 * max(0.0, 0.071488645583 - Q.z_7) / 0.021404874347360138   # +9.7%  z_7 < 0.07149
        - 0.08251759260672088 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.00041496068339887457   # -8.3%  z_7 < 0.07149 and centroid_offset < 0.03117
        + 0.06289235580683786 * max(0.0, 0.002635417778 - Q.width) * max(0.0, 0.023554160423 - Q.centroid_offset) / 1.2223894287524528e-05   # +6.3%  width < 0.002635 and centroid_offset < 0.02355
        - 0.057394228752014394 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.012003726523 - Q.lam1) / 0.0008217429754172258   # -5.7%  log_sum_pt > 6.573 and lam1 < 0.012
        + 0.04433240529947854 * max(0.0, 0.035560912266 - Q.e2) / 0.01371972344513943   # +4.4%  e2 < 0.03556
        - 0.03932497812432389 * max(0.0, 0.021588001063 - Q.dr_0) / 0.0036021402838930334   # -3.9%  dr_0 < 0.02159
        + 0.03892105998867212 * max(0.0, 0.01426135283 - Q.mean_phi2) * max(0.0, 40.046952646555 - Q.max_pair_mass) / 0.3492029151333389   # +3.9%  mean_phi2 < 0.01426 and max_pair_mass < 40.05
        - 0.03521519564489591 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.003952581551 - Q.girth2_top3) / 0.26264801266369175   # -3.5%  sum_pt_top2 < 548.2 and girth2_top3 < 0.003953
        - 0.03315457252397464 * max(0.0, Q.sum_pt - 868.509375) / 18.084627120331504   # -3.3%  sum_pt > 868.5
        + 0.030779344197447777 * max(0.0, 0.216055863061 - Q.LHA) / 0.032228649007922085   # +3.1%  LHA < 0.2161
        - 0.03046595425904495 * max(0.0, 0.026454043164 - Q.dr_0) / 0.005170180748163715   # -3.0%  dr_0 < 0.02645
        + 0.02893612530622297 * max(0.0, 0.01426135283 - Q.mean_phi2) / 0.011322003176195332   # +2.9%  mean_phi2 < 0.01426
        - 0.02734665317317976 * max(0.0, 0.002270363079 - Q.girth2_top5) / 0.0007634099414501585   # -2.7%  girth2_top5 < 0.00227
        - 0.027189577116922888 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.00407349257511083   # -2.7%  LHA < 0.2161 and log_sum_pt < 6.804
        + 0.026701670432832555 * max(0.0, 0.00528466865 - Q.e2_sq) / 0.002237349690661603   # +2.7%  e2_sq < 0.005285
        + 0.023685380474367676 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, 18.097979966098 - Q.max_pair_mass) / 0.4902789693049119   # +2.4%  mass_over_sum_pt < 0.08475 and max_pair_mass < 18.1
        - 0.023176370038571432 * max(0.0, 548.196875 - Q.sum_pt_top2) / 195.28859533251793   # -2.3%  sum_pt_top2 < 548.2
        + 0.02203578658221643 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.021588001063 - Q.dr_0) / 0.0006616654182381468   # +2.2%  log_sum_pt > 6.573 and dr_0 < 0.02159
        + 0.020545335434750806 * max(0.0, 0.049399692737 - Q.z_7) * max(0.0, 62.55 - Q.mass_top5) / 0.30594585851903483   # +2.1%  z_7 < 0.0494 and mass_top5 < 62.55
        + 0.020506343328814805 * max(0.0, 0.002635417778 - Q.width) / 0.0007941346639099353   # +2.1%  width < 0.002635
        - 0.018790002031970496 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001503553356 - Q.lam1) / 1.1470520225408143e-05   # -1.9%  z_7 < 0.07149 and lam1 < 0.001504
        + 0.0187270978682692 * max(0.0, Q.sum_pt_top5 - 752.1) / 21.274147637091883   # +1.9%  sum_pt_top5 > 752.1
        + 0.01781976620834652 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.006299534492 - Q.girth2_top2) / 0.0004246061902383332   # +1.8%  log_sum_pt > 6.573 and girth2_top2 < 0.0063
        - 0.016664701628630946 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) / 0.03304117548543854   # -1.7%  mass_over_sum_pt < 0.08475
        + 0.015788640269939636 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.001130644719 - Q.lam2) / 2.1406489096600342e-05   # +1.6%  z_7 < 0.07149 and lam2 < 0.001131
        + 0.01458367190726065 * max(0.0, Q.sum_pt - 868.509375) * max(0.0, 0.012587644117 - Q.centroid_offset) / 0.13441460519173168   # +1.5%  sum_pt > 868.5 and centroid_offset < 0.01259
        + 0.014310580175561055 * max(0.0, 0.035560912266 - Q.e2) * max(0.0, 7.3007261e-05 - Q.lam2) / 5.495977337610575e-07   # +1.4%  e2 < 0.03556 and lam2 < 7.301e-05
        - 0.013991265346993931 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 788.4484375 - Q.sum_pt) / 0.9285639647758638   # -1.4%  z_7 < 0.07149 and sum_pt < 788.4
        + 0.011583457327742956 * max(0.0, 0.03243272066 - Q.z_7) / 0.0020610238461745904   # +1.2%  z_7 < 0.03243
        + 0.010429575534534653 * max(0.0, 548.196875 - Q.sum_pt_top2) * max(0.0, 0.021588001063 - Q.dr_0) / 0.40891493757937986   # +1.0%  sum_pt_top2 < 548.2 and dr_0 < 0.02159
        - 0.009844583914151198 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 0.004331280361 - Q.mean_phi2) / 6.11210285357367e-05   # -1.0%  z_7 < 0.07149 and mean_phi2 < 0.004331
        + 0.008061163480964613 * max(0.0, 0.049399692737 - Q.z_7) / 0.007457505366835312   # +0.8%  z_7 < 0.0494
        + 0.007488705719907729 * max(0.0, 0.071488645583 - Q.z_7) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.01734541169321625   # +0.7%  z_7 < 0.07149 and n_dr_0p2_0p4 < 1
        - 0.007173853869507995 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.04118638065 - Q.dr_0) / 0.00011031021395682367   # -0.7%  log_sum_pt > 6.896 and dr_0 < 0.04119
        - 0.006993959659362204 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # -0.7%  log_sum_pt > 6.896
        + 0.00577734718040606 * max(0.0, 0.028865759995 - Q.z_6) / 0.0008381782190046483   # +0.6%  z_6 < 0.02887
        + 0.004361698838178044 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 0.000145482056 - Q.mean_phi2) / 4.16958841658888e-06   # +0.4%  log_sum_pt > 6.573 and mean_phi2 < 0.0001455
        - 0.004334116427713713 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, 0.014379521101 - Q.centroid_offset) / 1.3180146852159648e-05   # -0.4%  e2_sq < 0.005285 and centroid_offset < 0.01438
        + 0.003920837413380188 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, 9.0303693e-05 - Q.mean_eta2) / 2.028143157584815e-06   # +0.4%  log_sum_pt > 6.573 and mean_eta2 < 9.03e-05
        - 0.0031375832927047547 * max(0.0, Q.log_sum_pt - 6.896095378249) * max(0.0, 0.018377780003 - Q.centroid_offset) / 5.203533472960018e-05   # -0.3%  log_sum_pt > 6.896 and centroid_offset < 0.01838
        - 0.002604893938239898 * max(0.0, 0.00528466865 - Q.e2_sq) * max(0.0, Q.n_pt_above_50 - 6.0) / 0.0007322446740177089   # -0.3%  e2_sq < 0.005285 and n_pt_above_50 > 6
        - 0.0022171449325354008 * max(0.0, 0.084751611895 - Q.mass_over_sum_pt) * max(0.0, Q.mass_top3 - 28.345095968085) / 0.009446570356548488   # -0.2%  mass_over_sum_pt < 0.08475 and mass_top3 > 28.35
        + 0.0019784346828390803 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, 0.083662731125 - Q.planar_flow) / 7.059062637079571e-05   # +0.2%  LHA < 0.2161 and planar_flow < 0.08366
        + 0.0016781672131419617 * max(0.0, 0.03243272066 - Q.z_7) * max(0.0, Q.pt_5 - 33.0265625) / 0.010273887151011912   # +0.2%  z_7 < 0.03243 and pt_5 > 33.03
        - 0.0016025974195096263 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.mass_top3 - 3.559569591142) / 0.015700458953858076   # -0.2%  LHA < 0.2161 and mass_top3 > 3.56
        + 0.0015367572631094794 * max(0.0, 4.8108519e-05 - Q.girth2) / 8.325084578958415e-07   # +0.2%  girth2 < 4.811e-05
        - 0.0012587320493930327 * max(0.0, 24.578125 - Q.pt_5) / 0.2839763142561712   # -0.1%  pt_5 < 24.58
        - 0.0009051767012823915 * max(0.0, Q.log_sum_pt - 6.572937922293) * max(0.0, Q.centroid_offset - 0.018377780003) / 7.559756148996543e-05   # -0.1%  log_sum_pt > 6.573 and centroid_offset > 0.01838
        - 0.0007509941524390773 * max(0.0, 0.216055863061 - Q.LHA) * max(0.0, Q.n_dr_0p2_0p4 - 0.0) / 0.001534583791159281   # -0.1%  LHA < 0.2161 and n_dr_0p2_0p4 > 0
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 31.21;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 31.214157194975087 * (0.2729485312408842
        - 0.19359790981161218 * max(0.0, 0.008678044751 - Q.girth2) / 0.00444734738961668   # -19.4%  girth2 < 0.008678
        - 0.17341292946111647 * max(0.0, 0.013238675006 - Q.width) / 0.008236124461880547   # -17.3%  width < 0.01324
        - 0.1356736908685471 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) / 0.05326287594740439   # -13.6%  mass_over_sum_pt > 0.008374
        + 0.06439321896353556 * Q.max_dr / 0.1236972946900221   # +6.4%  max_dr
        + 0.057402903376473365 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # +5.7%  e2 < 0.05028
        + 0.04084738080675716 * max(0.0, 0.008375572068 - Q.lam1) / 0.0043001452793647   # +4.1%  lam1 < 0.008376
        + 0.02823976965480499 * max(0.0, 0.007330079875 - Q.lam1) / 0.0034868526190745728   # +2.8%  lam1 < 0.00733
        - 0.02547463003011059 * max(0.0, 0.000537286005 - Q.lam2) / 0.0003909825230380142   # -2.5%  lam2 < 0.0005373
        + 0.0236147031416693 * max(0.0, 0.003562611155 - Q.girth2) / 0.0011842494965261352   # +2.4%  girth2 < 0.003563
        + 0.022898704361794617 * max(0.0, 49.668099212646 - Q.mass) * max(0.0, 0.750909513235 - Q.z_dr_0p05_0p1) / 11.464259601117858   # +2.3%  mass < 49.67 and z_dr_0p05_0p1 < 0.7509
        + 0.022398617035530116 * max(0.0, Q.girth - 0.087236513197) / 0.007853436041615635   # +2.2%  girth > 0.08724
        - 0.018566269757346234 * max(0.0, Q.centroid_offset - 0.018377780003) / 0.005523694064646374   # -1.9%  centroid_offset > 0.01838
        + 0.015659729551810084 * max(0.0, 0.154170806525 - Q.mass_over_sum_pt) / 0.09372700683098877   # +1.6%  mass_over_sum_pt < 0.1542
        + 0.015107724060805409 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 45.75 - Q.pt_7) / 1.995212531081465   # +1.5%  log_sum_pt < 6.701 and pt_7 < 45.75
        - 0.014335784439750617 * max(0.0, 6.701242202626 - Q.log_sum_pt) / 0.19354909741955528   # -1.4%  log_sum_pt < 6.701
        - 0.013118977768510509 * max(0.0, Q.C2 - 0.010539266048) / 0.01887610517800746   # -1.3%  C2 > 0.01054
        + 0.012021857903230111 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.003408388935 - Q.lam2) / 2.6474453560625107e-05   # +1.2%  centroid_offset > 0.008092 and lam2 < 0.003408
        + 0.01113481757667922 * max(0.0, 6.572937922293 - Q.log_sum_pt) / 0.11632121255387393   # +1.1%  log_sum_pt < 6.573
        - 0.010888187281352232 * max(0.0, Q.mass_over_sum_pt - 0.008374148675) * max(0.0, 0.518696343899 - Q.tau32) / 0.005817482410631842   # -1.1%  mass_over_sum_pt > 0.008374 and tau32 < 0.5187
        - 0.009599960720382408 * max(0.0, 1.679198372364 - Q.D2) * max(0.0, 90.625 - Q.pt_4) / 17.624128555880276   # -1.0%  D2 < 1.679 and pt_4 < 90.62
        + 0.009126265764584559 * max(0.0, Q.girth2_top5 - 0.011482925368) / 0.0015471296538189059   # +0.9%  girth2_top5 > 0.01148
        + 0.007273946071276067 * max(0.0, Q.centroid_offset - 0.008092360237) / 0.010529146096613796   # +0.7%  centroid_offset > 0.008092
        + 0.0071409559430663655 * max(0.0, Q.C2 - 0.010539266048) * max(0.0, Q.pt_7 - 31.859375) / 0.08096163333875342   # +0.7%  C2 > 0.01054 and pt_7 > 31.86
        - 0.007011656709522488 * max(0.0, 49.668099212646 - Q.mass) / 17.068932039075   # -0.7%  mass < 49.67
        + 0.006508474024000655 * max(0.0, 1.679198372364 - Q.D2) / 0.5633062396133152   # +0.7%  D2 < 1.679
        - 0.005899438896461892 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 4.0 - Q.n_dr_0p05_0p1) / 0.07644425178616783   # -0.6%  e2 < 0.05028 and n_dr_0p05_0p1 < 4
        + 0.005413291974954628 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.15855820179) / 0.0002133523951813162   # +0.5%  e2 < 0.05028 and z_dr_0p1_0p2 > 0.1586
        - 0.005291901565776659 * max(0.0, Q.lam2 - 0.003408388935) / 0.00015702993621826726   # -0.5%  lam2 > 0.003408
        + 0.005195102739828041 * max(0.0, Q.LHA - 0.312727471086) * max(0.0, Q.eccentricity - 0.872657364787) / 0.0009017529775732308   # +0.5%  LHA > 0.3127 and eccentricity > 0.8727
        + 0.004627055634621334 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 0.049399692737 - Q.z_7) / 0.00020927924246377964   # +0.5%  log_sum_pt < 6.701 and z_7 < 0.0494
        - 0.004539296007959636 * max(0.0, Q.girth2_top5 - 0.002270363079) / 0.0043533388582812135   # -0.5%  girth2_top5 > 0.00227
        + 0.0038330696425107227 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, 0.094821243733 - Q.C2) / 0.0005674917208011498   # +0.4%  centroid_offset > 0.008092 and C2 < 0.09482
        + 0.0029685257260576555 * max(0.0, Q.centroid_offset - 0.008092360237) * max(0.0, Q.planar_flow - 0.00804883781) / 0.002771643803870943   # +0.3%  centroid_offset > 0.008092 and planar_flow > 0.008049
        + 0.0027431741312447886 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, 0.071488645583 - Q.z_7) / 9.826026506744865e-05   # +0.3%  log_sum_pt < 6.327 and z_7 < 0.07149
        + 0.0026405799151244806 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 0.008921136335 - Q.mean_phi2) / 2.1793621511525516e-05   # +0.3%  centroid_offset > 0.01838 and mean_phi2 < 0.008921
        + 0.0019238256122947014 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, Q.pt_2 - 56.5) / 0.11492222841935226   # +0.2%  centroid_offset > 0.01838 and pt_2 > 56.5
        + 0.0016156449697721405 * max(0.0, 6.327378592257 - Q.log_sum_pt) / 0.033169101899090815   # +0.2%  log_sum_pt < 6.327
        + 0.0015533871638646892 * max(0.0, 0.007330079875 - Q.lam1) * max(0.0, 1.232133567333 - Q.D2) / 0.0003628155854250704   # +0.2%  lam1 < 0.00733 and D2 < 1.232
        + 0.001541534040485476 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, 36.8125 - Q.pt_6) / 0.41268944603937285   # +0.2%  log_sum_pt < 6.701 and pt_6 < 36.81
        - 0.0013178292481936754 * max(0.0, 6.701242202626 - Q.log_sum_pt) * max(0.0, Q.mean_phi - 0.009050007537) / 0.0008878548382281195   # -0.1%  log_sum_pt < 6.701 and mean_phi > 0.00905
        - 0.0006871128785613718 * max(0.0, Q.z_7 - 0.06164517166) / 0.004746501812103751   # -0.1%  z_7 > 0.06165
        + 0.0005798589740257847 * max(0.0, 6.327378592257 - Q.log_sum_pt) * max(0.0, Q.pt_6 - 27.578125) / 0.2801529239851139   # +0.1%  log_sum_pt < 6.327 and pt_6 > 27.58
        - 0.000519834182254008 * max(0.0, 0.000537286005 - Q.lam2) * max(0.0, Q.mean_eta - 0.02644207105) / 1.235016994909338e-07   # -0.1%  lam2 < 0.0005373 and mean_eta > 0.02644
        + 0.0004931388128480108 * max(0.0, Q.centroid_offset - 0.018377780003) * max(0.0, 24.578125 - Q.pt_5) / 0.00025235720347887783   # +0.0%  centroid_offset > 0.01838 and pt_5 < 24.58
        + 0.0004895035336844517 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_eta - 0.02644207105) / 1.6842597802487237e-06   # +0.0%  width < 0.01324 and mean_eta > 0.02644
        - 0.0003480357468310043 * max(0.0, Q.centroid_offset - 0.049903668404) / 0.0008064001164412618   # -0.0%  centroid_offset > 0.0499
        + 0.00032979351837645867 * max(0.0, 0.013238675006 - Q.width) * max(0.0, Q.mean_phi - 0.026127964072) / 1.6737490359453807e-06   # +0.0%  width < 0.01324 and mean_phi > 0.02613
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 36.51;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 36.50898539825622 * (0.2705391143076064
        - 0.0957866906492 * max(0.0, Q.girth2 - 0.007520088344) / 0.0024701362488572048   # -9.6%  girth2 > 0.00752
        + 0.08961539547737438 * max(0.0, Q.girth2 - 0.008678044751) / 0.002214536732122738   # +9.0%  girth2 > 0.008678
        - 0.07996644649329128 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -8.0%  girth < 0.08724
        - 0.0773523006934809 * max(0.0, Q.girth2 - 0.0016538364) / 0.005220304346760304   # -7.7%  girth2 > 0.001654
        - 0.0724382586827243 * max(0.0, Q.mass_over_sum_pt - 0.090413827016) / 0.008298543275912581   # -7.2%  mass_over_sum_pt > 0.09041
        - 0.06199898651698133 * max(0.0, 0.005590288644 - Q.width) / 0.0022293108396669657   # -6.2%  width < 0.00559
        + 0.060717270768514825 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +6.1%  e2 < 0.03847
        - 0.06047219755863292 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # -6.0%  e2 < 0.05028
        - 0.05275343217219072 * max(0.0, 0.008375572068 - Q.lam1) / 0.0043001452793647   # -5.3%  lam1 < 0.008376
        + 0.05258456213544009 * max(0.0, Q.mass_over_sum_pt - 0.084751611895) / 0.00956189524862828   # +5.3%  mass_over_sum_pt > 0.08475
        - 0.04892541178157586 * max(0.0, Q.girth2 - 0.004372139461) / 0.0036388049448583216   # -4.9%  girth2 > 0.004372
        + 0.04005123249464082 * max(0.0, Q.mass_over_sum_pt - 0.072690732432) / 0.013441384325389693   # +4.0%  mass_over_sum_pt > 0.07269
        + 0.026168285379329068 * max(0.0, 0.293190627853 - Q.LHA) / 0.07167605978362125   # +2.6%  LHA < 0.2932
        + 0.017954227591679164 * max(0.0, Q.girth2 - 0.004372139461) * max(0.0, Q.eccentricity - 0.945820652852) / 7.473745705530461e-05   # +1.8%  girth2 > 0.004372 and eccentricity > 0.9458
        - 0.0139106650477616 * max(0.0, 48.71875 - Q.pt_7) * max(0.0, 0.694781820497 - Q.planar_flow) / 6.147530334000904   # -1.4%  pt_7 < 48.72 and planar_flow < 0.6948
        - 0.013888486450862944 * max(0.0, 0.195013533663 - Q.planar_flow) / 0.07573236359588242   # -1.4%  planar_flow < 0.195
        + 0.013186860270404634 * max(0.0, 0.001101266364 - Q.e2_sq) / 0.0003080039918590354   # +1.3%  e2_sq < 0.001101
        + 0.011172275243711763 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # +1.1%  e2 < 0.02455
        + 0.010941789324194217 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, 1.122624260187 - Q.D2) / 0.0025585111166936624   # +1.1%  e2 < 0.05028 and D2 < 1.123
        - 0.010006550057079238 * max(0.0, 0.02076709205 - Q.centroid_offset) / 0.00833429084748231   # -1.0%  centroid_offset < 0.02077
        + 0.009830131394029343 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.sum_pt - 615.875) / 10.120048448944816   # +1.0%  planar_flow < 0.195 and sum_pt > 615.9
        - 0.009398237584556535 * max(0.0, 0.197968879342 - Q.max_dr) / 0.08649773458639719   # -0.9%  max_dr < 0.198
        + 0.008881034886977206 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # +0.9%  mass < 29.64
        - 0.008756164006271997 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, 1.122624260187 - Q.D2) / 0.0004358510611742372   # -0.9%  lam1 < 0.008376 and D2 < 1.123
        - 0.00872213105217463 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.width - 0.00752008842) / 0.00014550293410035644   # -0.9%  planar_flow < 0.195 and width > 0.00752
        + 0.0064744592445435106 * max(0.0, 0.005884990035 - Q.girth2_top3) / 0.0030201574371678586   # +0.6%  girth2_top3 < 0.005885
        - 0.006143441939454718 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -0.6%  mass > 80.4
        - 0.0060877533022818025 * max(0.0, 0.001056655216 - Q.girth2_top2) / 0.0003002513836013012   # -0.6%  girth2_top2 < 0.001057
        - 0.004941713108828809 * max(0.0, 0.04081947431 - Q.girth) / 0.00917631455554571   # -0.5%  girth < 0.04082
        + 0.004657386150666057 * max(0.0, 48.71875 - Q.pt_7) / 14.746502676109506   # +0.5%  pt_7 < 48.72
        - 0.0038562381686436434 * max(0.0, 0.000561123155 - Q.width) / 9.465572315646571e-05   # -0.4%  width < 0.0005611
        - 0.002950314923000089 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.927072033478) / 0.038560975699403234   # -0.3%  mass > 80.4 and eccentricity > 0.9271
        + 0.0020038786296363515 * max(0.0, 0.02076709205 - Q.centroid_offset) * max(0.0, Q.C2 - 0.023843882605) / 4.3008223025190735e-05   # +0.2%  centroid_offset < 0.02077 and C2 > 0.02384
        + 0.0017936502523717404 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, Q.mass_top3 - 23.663861485439) / 0.5450859711047729   # +0.2%  planar_flow < 0.195 and mass_top3 > 23.66
        + 0.0014107761972460462 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21) / 0.0003205445922054599   # +0.1%  e2 < 0.02455 and tau21 < 0.4466
        - 0.001263378232990782 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_2 - 56.5) / 0.04247433685700738   # -0.1%  centroid_offset > 0.03117 and pt_2 > 56.5
        + 0.0010068535215830938 * max(0.0, 0.001056655216 - Q.girth2_top2) * max(0.0, Q.centroid_offset - 0.006789738266) / 7.886294926840798e-07   # +0.1%  girth2_top2 < 0.001057 and centroid_offset > 0.00679
        - 0.0009309266125354562 * max(0.0, 0.195013533663 - Q.planar_flow) * max(0.0, 35.28125 - Q.pt_6) / 0.16902486327670005   # -0.1%  planar_flow < 0.195 and pt_6 < 35.28
        - 0.0006120581600062809 * max(0.0, Q.centroid_offset - 0.031170772021) * max(0.0, Q.pt_0 - 376.5) / 0.002409555939429373   # -0.1%  centroid_offset > 0.03117 and pt_0 > 376.5
        + 0.00030514561673725626 * max(0.0, Q.centroid_offset - 0.031170772021) / 0.00250501853207283   # +0.0%  centroid_offset > 0.03117
        - 8.300222639463159e-05 * max(0.0, 0.001101266364 - Q.e2_sq) * max(0.0, -0.009460449219 - Q.phi_1) / 4.875626464770658e-07   # -0.0%  e2_sq < 0.001101 and phi_1 < -0.00946
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 18.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.366123627067683 * (-0.023576587381384562
        + 0.13550162204582403 * max(0.0, 0.005019718802 - Q.width) / 0.0019034243755791656   # +13.6%  width < 0.00502
        - 0.07906943485656431 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.005019718802 - Q.width) / 7.976303597873066e-05   # -7.9%  girth < 0.06109 and width < 0.00502
        + 0.06522255170123326 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 0.1009733513 - Q.z_dr_0p2_0p4) / 0.0001901112107600099   # +6.5%  width < 0.00502 and z_dr_0p2_0p4 < 0.101
        + 0.055120237393364344 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.023554160423 - Q.centroid_offset) / 3.88308106019661e-05   # +5.5%  girth2 < 0.006679 and centroid_offset < 0.02355
        - 0.046580012908627795 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.10213245107008513   # -4.7%  mass < 29.64 and centroid_offset < 0.02355
        - 0.04305760665349854 * max(0.0, 0.061086014472 - Q.girth) / 0.018686846679786144   # -4.3%  girth < 0.06109
        + 0.03909541887722651 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.00015145615486558198   # +3.9%  e2 < 0.02455 and centroid_offset < 0.03117
        - 0.032920417701244306 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000306123359 - Q.lam2) / 7.006679088170391e-06   # -3.3%  LHA < 0.1967 and lam2 < 0.0003061
        - 0.03122438611271418 * max(0.0, 21.784077072144 - Q.mass) / 4.431023211632268   # -3.1%  mass < 21.78
        + 0.028431214570935986 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.000194798295 - Q.lam2) / 2.869242750180312e-06   # +2.8%  girth < 0.06109 and lam2 < 0.0001948
        - 0.027591919486148957 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) * max(0.0, 0.000537286005 - Q.lam2) / 2.697123900716615e-05   # -2.8%  z_dr_0_0p05 > 0.8477 and lam2 < 0.0005373
        + 0.025761920378728223 * max(0.0, 0.177304983139 - Q.max_dr) / 0.07042349604125828   # +2.6%  max_dr < 0.1773
        + 0.02486548489223057 * max(0.0, 0.016554418951 - Q.e2) / 0.0038873472250323077   # +2.5%  e2 < 0.01655
        + 0.024157717060181882 * max(0.0, 0.177304983139 - Q.max_dr) * max(0.0, 0.000194798295 - Q.lam2) / 9.998101856924045e-06   # +2.4%  max_dr < 0.1773 and lam2 < 0.0001948
        - 0.022896969667351094 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.centroid_offset - 0.006789738266) / 1.0643487980325792e-05   # -2.3%  width < 0.00502 and centroid_offset > 0.00679
        - 0.02285260476331762 * max(0.0, 0.196739721581 - Q.LHA) / 0.02506564427600322   # -2.3%  LHA < 0.1967
        - 0.01747538070655813 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.006978029732767499   # -1.7%  log_sum_pt > 6.701 and z_dr_0p2_0p4 < 0.2055
        + 0.016248802770487743 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, 0.000561123155 - Q.width) / 9.58594909358334e-06   # +1.6%  LHA < 0.1967 and width < 0.0005611
        - 0.01593901505987304 * max(0.0, Q.log_sum_pt - 6.701242202626) / 0.035238726665107536   # -1.6%  log_sum_pt > 6.701
        + 0.01558567973166847 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # +1.6%  e2 < 0.02455
        - 0.015255788988970205 * max(0.0, 0.000222950415 - Q.girth2_top5) / 2.934670561073984e-05   # -1.5%  girth2_top5 < 0.000223
        - 0.014788039898988001 * max(0.0, Q.pt_7 - 34.53125) / 4.473662132352941   # -1.5%  pt_7 > 34.53
        + 0.014395727208464453 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.40079469091 - Q.planar_flow) / 0.0003918103360080332   # +1.4%  girth2 < 0.006679 and planar_flow < 0.4008
        + 0.0120835646661676 * max(0.0, 0.000964142894 - Q.girth2) / 0.00020522880102919315   # +1.2%  girth2 < 0.0009641
        - 0.01132702863909978 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.centroid_offset - 0.006789738266) / 8.239263087855558e-05   # -1.1%  girth < 0.06109 and centroid_offset > 0.00679
        + 0.011157770127710005 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.023554160423 - Q.centroid_offset) / 0.0005980205978877087   # +1.1%  log_sum_pt > 6.701 and centroid_offset < 0.02355
        - 0.010864674273387996 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.000306123359 - Q.lam2) / 0.0006644045602408043   # -1.1%  mass < 15.45 and lam2 < 0.0003061
        - 0.010731174581918152 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, Q.centroid_offset - 0.016278845848) / 0.01429704883653226   # -1.1%  mass < 21.78 and centroid_offset > 0.01628
        - 0.010703559337423105 * max(0.0, Q.z_dr_0_0p05 - 0.847731333971) / 0.0544885966340469   # -1.1%  z_dr_0_0p05 > 0.8477
        - 0.00964637124028923 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.0010110746822510478   # -1.0%  girth < 0.06109 and z_dr_0p2_0p4 < 0.05644
        - 0.009332906399462583 * max(0.0, 21.784077072144 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset) / 0.07546093782958285   # -0.9%  mass < 21.78 and centroid_offset < 0.02686
        - 0.00861060600548116 * max(0.0, 0.061086014472 - Q.girth) * max(0.0, Q.lam1 - 0.00027588256) / 1.010244609535702e-05   # -0.9%  girth < 0.06109 and lam1 > 0.0002759
        + 0.008572019462344234 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 48.71875 - Q.pt_7) / 0.7769310447436848   # +0.9%  log_sum_pt > 6.701 and pt_7 < 48.72
        + 0.008354593030394085 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_7 - 0.016858545121) / 0.0005923452122172064   # +0.8%  LHA < 0.1967 and z_7 > 0.01686
        - 0.008015849303644457 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, 0.501026660204 - Q.tau21) / 0.0002928575231997225   # -0.8%  girth2 < 0.006679 and tau21 < 0.501
        + 0.0077638517486010705 * max(0.0, 0.024547699839 - Q.e2) * max(0.0, Q.sum_pt - 788.4484375) / 0.5518276932797608   # +0.8%  e2 < 0.02455 and sum_pt > 788.4
        - 0.006649391283173945 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.centroid_offset - 0.018377780003) / 5.841408434865172e-06   # -0.7%  girth2 < 0.006679 and centroid_offset > 0.01838
        - 0.00651099213000244 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.018827652745 - Q.girth2) / 0.000590048264493985   # -0.7%  log_sum_pt > 6.701 and girth2 < 0.01883
        - 0.006421233726872479 * max(0.0, 0.006679471442 - Q.girth2) * max(0.0, Q.n_dr_0p05_0p1 - 0.0) / 0.0023404635737340855   # -0.6%  girth2 < 0.006679 and n_dr_0p05_0p1 > 0
        + 0.006295204992519088 * max(0.0, 8.379955863953 - Q.mass) / 0.5781888805022356   # +0.6%  mass < 8.38
        - 0.004977254301534093 * max(0.0, 0.005019718802 - Q.width) * max(0.0, Q.mass_over_sum_pt_sq - 0.00012320649) / 9.121997321478234e-07   # -0.5%  width < 0.00502 and mass_over_sum_pt_sq > 0.0001232
        - 0.004890439938492241 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.z_6 - 0.02160287394) / 0.0006814928842297751   # -0.5%  LHA < 0.1967 and z_6 > 0.0216
        - 0.004755406992679532 * max(0.0, 15.454033088684 - Q.mass) * max(0.0, 0.058613700176 - Q.z_7) / 0.03926630160825465   # -0.5%  mass < 15.45 and z_7 < 0.05861
        + 0.004610817323827036 * max(0.0, 0.003343241496 - Q.centroid_offset) / 0.00022720523783792328   # +0.5%  centroid_offset < 0.003343
        + 0.004198220694076512 * max(0.0, Q.log_sum_pt - 6.701242202626) * max(0.0, 0.00023679558 - Q.mass_over_sum_pt_sq) / 3.22282829726687e-06   # +0.4%  log_sum_pt > 6.701 and mass_over_sum_pt_sq < 0.0002368
        - 0.004049177510337074 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55591824366466   # -0.4%  sum_pt_top5 > 658.1
        + 0.003116499071131781 * max(0.0, 0.005019718802 - Q.width) * max(0.0, 248.125 - Q.sum_pt_top2) / 0.012572105241288054   # +0.3%  width < 0.00502 and sum_pt_top2 < 248.1
        - 0.002323439785228609 * max(0.0, 0.196739721581 - Q.LHA) * max(0.0, Q.mean_phi - -0.000855675264) / 5.448370435494377e-05   # -0.2%  LHA < 0.1967 and mean_phi > -0.0008557
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 26.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.29252832990791 * (-0.06230833545849684
        + 0.15818105690136536 * max(0.0, 0.006096650059 - Q.width) / 0.002544039033914621   # +15.8%  width < 0.006097
        + 0.06563049849267968 * max(0.0, 0.007520088344 - Q.girth2) / 0.003544990499428731   # +6.6%  girth2 < 0.00752
        - 0.06494256066868985 * max(0.0, 53.332374954224 - Q.mass) / 19.36003866872732   # -6.5%  mass < 53.33
        + 0.05335189903217329 * max(0.0, 0.004372139461 - Q.girth2) / 0.0015657103124088927   # +5.3%  girth2 < 0.004372
        - 0.050475876848218026 * max(0.0, 0.054649224505 - Q.girth) / 0.015329780861188023   # -5.0%  girth < 0.05465
        + 0.04958903597993407 * max(0.0, 0.018377780003 - Q.centroid_offset) / 0.0067171359273014   # +5.0%  centroid_offset < 0.01838
        + 0.04664395150712717 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.026856224803 - Q.centroid_offset) / 0.2973625515047354   # +4.7%  mass < 53.33 and centroid_offset < 0.02686
        + 0.04434442885752136 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 6.842716632804 - Q.log_sum_pt) / 5.175435934052333   # +4.4%  mass < 53.33 and log_sum_pt < 6.843
        - 0.04390155744948745 * max(0.0, Q.log_sum_pt - 6.377722943814) / 0.20956803777448815   # -4.4%  log_sum_pt > 6.378
        - 0.04156794909619069 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.centroid_offset - 0.003343241496) / 2.1005617807319642e-05   # -4.2%  width < 0.006097 and centroid_offset > 0.003343
        - 0.039970662523783555 * max(0.0, 0.076373631775 - Q.mass_over_sum_pt) / 0.027150273207929114   # -4.0%  mass_over_sum_pt < 0.07637
        + 0.034050282752900805 * max(0.0, 0.221586732566 - Q.max_dr) / 0.10604661584056564   # +3.4%  max_dr < 0.2216
        + 0.033139472517118705 * max(0.0, 0.000964142894 - Q.girth2) / 0.00020522880102919315   # +3.3%  girth2 < 0.0009641
        - 0.025673055469304266 * max(0.0, 29.644699859619 - Q.mass) / 7.347229538150994   # -2.6%  mass < 29.64
        + 0.024498323473743852 * max(0.0, 0.020459658932 - Q.e2) / 0.00553192518950076   # +2.4%  e2 < 0.02046
        + 0.023723933150046658 * max(0.0, 41.377904891968 - Q.mass) / 12.537841289638873   # +2.4%  mass < 41.38
        - 0.020892637516620377 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.000872228216 - Q.lam1) / 0.00860596641673995   # -2.1%  mass < 53.33 and lam1 < 0.0008722
        - 0.020533418155010683 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.047491459878) / 0.00022600413084766904   # -2.1%  centroid_offset < 0.01838 and z_4 > 0.04749
        + 0.019333323454970457 * max(0.0, 53.332374954224 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow) / 1.530012963091921   # +1.9%  mass < 53.33 and planar_flow < 0.3221
        - 0.018357858058712276 * max(0.0, 0.032346998155 - Q.e2) / 0.011718069376880604   # -1.8%  e2 < 0.03235
        - 0.012537732326481322 * max(0.0, 41.377904891968 - Q.mass) * max(0.0, 0.322073846732 - Q.planar_flow) / 0.8148647296533372   # -1.3%  mass < 41.38 and planar_flow < 0.3221
        - 0.012275342375028 * max(0.0, 0.001503553356 - Q.lam1) / 0.0003926183331151659   # -1.2%  lam1 < 0.001504
        + 0.010589590135736869 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.pt_4 - 47.34375) / 0.10790638482690211   # +1.1%  centroid_offset < 0.01838 and pt_4 > 47.34
        + 0.010077884449598574 * max(0.0, Q.lam2 - 0.001130644719) / 0.0003119523112529589   # +1.0%  lam2 > 0.001131
        + 0.008398768149900749 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.055953954317 - Q.dr01) / 0.0005482140187387444   # +0.8%  e2 < 0.03235 and dr01 < 0.05595
        - 0.00812236891619622 * max(0.0, 29.644699859619 - Q.mass) * max(0.0, 0.002127561159 - Q.mean_phi2) / 0.01390258796706784   # -0.8%  mass < 29.64 and mean_phi2 < 0.002128
        - 0.007530377003007933 * max(0.0, 0.018377780003 - Q.centroid_offset) * max(0.0, Q.z_5 - 0.036727111752) / 0.00019974565532721066   # -0.8%  centroid_offset < 0.01838 and z_5 > 0.03673
        + 0.006698691179601154 * max(0.0, Q.girth2 - 0.018827652745) / 0.0007605368842505339   # +0.7%  girth2 > 0.01883
        + 0.006074929976670493 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857142857143   # +0.6%  n_dr_0p2_0p4 > 1
        - 0.005588858422950938 * max(0.0, 0.032346998155 - Q.e2) * max(0.0, 0.446608647704 - Q.tau21) / 0.0007426911303479672   # -0.6%  e2 < 0.03235 and tau21 < 0.4466
        + 0.004225643197212728 * max(0.0, Q.C2 - 0.051192347892) / 0.004801477093795494   # +0.4%  C2 > 0.05119
        + 0.0035610075487303616 * max(0.0, 35.5 - Q.pt_5) / 1.4798162827435661   # +0.4%  pt_5 < 35.5
        + 0.0035223565192078823 * max(0.0, 0.000172198326 - Q.width) / 1.4418964712323033e-05   # +0.4%  width < 0.0001722
        - 0.0031700041114902846 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, Q.eccentricity - 0.903125533696) / 0.00012511356827127218   # -0.3%  e2 < 0.02046 and eccentricity > 0.9031
        - 0.00276785025753484 * max(0.0, Q.max_dr - 0.15984864831) / 0.0215651977777285   # -0.3%  max_dr > 0.1598
        + 0.0026768385867613656 * max(0.0, 0.020459658932 - Q.e2) * max(0.0, 53.4375 - Q.pt_7) / 0.11864275984569007   # +0.3%  e2 < 0.02046 and pt_7 < 53.44
        - 0.002108172756057974 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.C2 - 0.030867108516) / 5.18136846120527e-06   # -0.2%  width < 0.006097 and C2 > 0.03087
        - 0.002034057750395307 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.40079469091 - Q.planar_flow) / 0.0001085455191037657   # -0.2%  girth2 > 0.01883 and planar_flow < 0.4008
        - 0.00194032765453026 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, Q.n_dr_0p05_0p1 - 1.0) / 0.005566259290425861   # -0.2%  C2 > 0.05119 and n_dr_0p05_0p1 > 1
        - 0.0016355669983335278 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 0.02644207105 - Q.mean_eta) / 2.309501513169841e-05   # -0.2%  girth2 > 0.01883 and mean_eta < 0.02644
        + 0.001577256795425378 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.620723099573 - Q.eccentricity) / 0.00020449793322373317   # +0.2%  C2 > 0.05119 and eccentricity < 0.6207
        + 0.0014556415431414029 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, 29.0421875 - Q.pt_7) / 0.0006069490438354982   # +0.1%  girth2 > 0.01883 and pt_7 < 29.04
        - 0.0013200263707946627 * max(0.0, 35.5 - Q.pt_5) * max(0.0, Q.min_pair_mass - 0.173071536962) / 1.3593246346767114   # -0.1%  pt_5 < 35.5 and min_pair_mass > 0.1731
        + 0.0013089250696131374 * max(0.0, Q.lam2 - 0.001130644719) * max(0.0, Q.mass_top2 - 16.308019673264) / 0.002687414260360442   # +0.1%  lam2 > 0.001131 and mass_top2 > 16.31
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 12.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.88925099773224 * (-0.13440323455965583
        + 0.14969945338004928 * Q.e2 / 0.028632153681352513   # +15.0%  e2
        + 0.10942302991311333 * max(0.0, 0.090413827016 - Q.mass_over_sum_pt) / 0.03744003863362622   # +10.9%  mass_over_sum_pt < 0.09041
        + 0.08180322142097853 * max(0.0, Q.mass - 15.454033088684) / 27.25251475411285   # +8.2%  mass > 15.45
        - 0.05460750568984018 * max(0.0, 0.004183811014 - Q.lam1) / 0.001513077448758448   # -5.5%  lam1 < 0.004184
        + 0.054072362024949916 * max(0.0, Q.lam2 - 0.000194798295) / 0.0004461514947095026   # +5.4%  lam2 > 0.0001948
        - 0.053777389602934066 * max(0.0, 0.0016538364 - Q.girth2) / 0.0004289066532460978   # -5.4%  girth2 < 0.001654
        + 0.04860782654623052 * max(0.0, 0.006679471442 - Q.girth2) / 0.002936239856310497   # +4.9%  girth2 < 0.006679
        - 0.04813039911300447 * max(0.0, Q.log_sum_pt - 6.701242202626) / 0.035238726665107536   # -4.8%  log_sum_pt > 6.701
        + 0.03960234258059301 * max(0.0, 0.002412890926 - Q.girth2_top2) / 0.0009487943624112958   # +4.0%  girth2_top2 < 0.002413
        + 0.038647911318964154 * max(0.0, Q.eccentricity - 0.903125533696) * max(0.0, 0.05643851608 - Q.z_dr_0p2_0p4) / 0.0022074526402598485   # +3.9%  eccentricity > 0.9031 and z_dr_0p2_0p4 < 0.05644
        - 0.03706213239984003 * max(0.0, 0.006506575659 - Q.lam1) / 0.00289007769049566   # -3.7%  lam1 < 0.006507
        + 0.03217026499967766 * max(0.0, 3.0 - Q.n_dr_0p05_0p1) / 1.6691546218487394   # +3.2%  n_dr_0p05_0p1 < 3
        + 0.028935559693838896 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.701242202626) / 9.541953566382002e-05   # +2.9%  lam1 < 0.004184 and log_sum_pt > 6.701
        + 0.02710061990094588 * max(0.0, Q.sum_pt - 813.415625) / 31.70376994639103   # +2.7%  sum_pt > 813.4
        + 0.020556073478783472 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, 0.175465903809 - Q.dr_7) / 0.00021702636252652282   # +2.1%  lam1 < 0.004184 and dr_7 < 0.1755
        - 0.016945809096210594 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.pt_4 - 39.8125) / 2.740588593384243   # -1.7%  tau21 < 0.3915 and pt_4 > 39.81
        - 0.016574945124020555 * max(0.0, Q.LHA - 0.303313749495) * max(0.0, 0.553068161011 - Q.tau21) / 0.005421876887105311   # -1.7%  LHA > 0.3033 and tau21 < 0.5531
        - 0.015692568519464833 * max(0.0, Q.LHA - 0.303313749495) / 0.017943093379553122   # -1.6%  LHA > 0.3033
        + 0.015100649324450767 * max(0.0, 0.391541349888 - Q.tau21) / 0.1379333956742398   # +1.5%  tau21 < 0.3915
        - 0.013834694214114278 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, Q.planar_flow - 0.012569162668) / 0.0003037107863002691   # -1.4%  lam2 > 0.0001948 and planar_flow > 0.01257
        - 0.013311256472826513 * max(0.0, Q.max_dr - 0.121680960059) / 0.035622055613376453   # -1.3%  max_dr > 0.1217
        + 0.013096470199537363 * max(0.0, 0.391541349888 - Q.tau21) * max(0.0, Q.z_4 - 0.075444822386) / 0.0021348581786107975   # +1.3%  tau21 < 0.3915 and z_4 > 0.07544
        + 0.010484046219009198 * max(0.0, Q.C2 - 0.051192347892) / 0.004801477093795494   # +1.0%  C2 > 0.05119
        - 0.009826716306360919 * max(0.0, 0.004183811014 - Q.lam1) * max(0.0, Q.sum_pt - 988.4078125) / 0.014000463321047197   # -1.0%  lam1 < 0.004184 and sum_pt > 988.4
        + 0.00936558873433424 * max(0.0, 0.269169217348 - Q.tau32) / 0.011024941728233898   # +0.9%  tau32 < 0.2692
        - 0.008184998254198005 * max(0.0, Q.mass - 15.454033088684) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 37.95303185489477   # -0.8%  mass > 15.45 and n_dr_0p2_0p4 < 2
        - 0.007665274682808413 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 2.055451202393 - Q.D2) / 0.0003715223898042748   # -0.8%  lam2 > 0.0001948 and D2 < 2.055
        + 0.0071796815670234744 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 0.501026660204 - Q.tau21) / 4.680532629671513e-05   # +0.7%  lam2 > 0.0001948 and tau21 < 0.501
        - 0.006151502305078232 * max(0.0, Q.lam2 - 0.000194798295) * max(0.0, 8.0 - Q.n_pt_above_50) / 0.0016231978158997133   # -0.6%  lam2 > 0.0001948 and n_pt_above_50 < 8
        - 0.0045401746841728634 * max(0.0, 6.267538488641 - Q.log_sum_pt) / 0.02292097294024541   # -0.5%  log_sum_pt < 6.268
        + 0.002569038756988978 * max(0.0, Q.centroid_offset - 0.00231612516) * max(0.0, 34.53125 - Q.pt_7) / 0.050300266504218406   # +0.3%  centroid_offset > 0.002316 and pt_7 < 34.53
        - 0.0025203712889707974 * max(0.0, 0.269169217348 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.008366995203849923   # -0.3%  tau32 < 0.2692 and n_dr_0p2_0p4 < 2
        + 0.0020170286455556572 * max(0.0, Q.n_dr_0p2_0p4 - 2.0) / 0.05970756302521008   # +0.2%  n_dr_0p2_0p4 > 2
        - 0.0007430935411309028 * max(0.0, Q.C2 - 0.051192347892) * max(0.0, 0.222994708167 - Q.dr_7) / 0.0003521456689843323   # -0.1%  C2 > 0.05119 and dr_7 < 0.223
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 26.27;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 26.271690071901123 * (-0.06160256269482947
        + 0.1836511030048299 * max(0.0, 0.008678044951 - Q.width) / 0.004447347549975801   # +18.4%  width < 0.008678
        + 0.11089184536608793 * max(0.0, 0.049903668404 - Q.centroid_offset) / 0.03352573038000459   # +11.1%  centroid_offset < 0.0499
        - 0.09319750129952982 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -9.3%  girth < 0.08724
        + 0.07995122503782505 * max(0.0, 0.006679471442 - Q.girth2) / 0.002936239856310497   # +8.0%  girth2 < 0.006679
        + 0.0706860897090133 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741758945   # +7.1%  girth2 < 0.01324
        - 0.05461848654057433 * max(0.0, 0.008375572068 - Q.lam1) / 0.0043001452793647   # -5.5%  lam1 < 0.008376
        - 0.05183871234733885 * max(0.0, 0.006390124748 - Q.e2_sq) / 0.0029535077959458003   # -5.2%  e2_sq < 0.00639
        + 0.03589444735173601 * max(0.0, 0.253403707141 - Q.planar_flow) / 0.10951502727440561   # +3.6%  planar_flow < 0.2534
        + 0.034231813702052985 * max(0.0, 687.4375 - Q.sum_pt_top5) / 131.13431972163866   # +3.4%  sum_pt_top5 < 687.4
        - 0.033278208928264406 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.007589401193857488   # -3.3%  centroid_offset < 0.0499 and log_sum_pt < 6.804
        - 0.028189537853698206 * max(0.0, 0.04447356835 - Q.e2) / 0.020193337934351147   # -2.8%  e2 < 0.04447
        - 0.021955291839059633 * max(0.0, Q.girth - 0.076081777364) / 0.010590329529546494   # -2.2%  girth > 0.07608
        - 0.020265900224895012 * max(0.0, 0.003562611091 - Q.width) / 0.0011842494679572973   # -2.0%  width < 0.003563
        - 0.019521223725038783 * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 1.9597932773109243   # -2.0%  n_dr_0p1_0p2 < 3
        - 0.015401992143108192 * max(0.0, 0.154689112391 - Q.LHA) / 0.0126143407725805   # -1.5%  LHA < 0.1547
        - 0.01377122171038334 * max(0.0, 0.035786485299 - Q.C2) / 0.015048242903037393   # -1.4%  C2 < 0.03579
        - 0.012710576602948984 * max(0.0, Q.centroid_offset - 0.014379521101) / 0.007111299972137995   # -1.3%  centroid_offset > 0.01438
        + 0.012338315145698532 * max(0.0, 0.221586732566 - Q.max_dr) / 0.10604661584056564   # +1.2%  max_dr < 0.2216
        - 0.011995886051733516 * max(0.0, 0.004839980301 - Q.lam1) / 0.001855088115791994   # -1.2%  lam1 < 0.00484
        - 0.011563871254222707 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.width - 0.006096650059) / 0.0002689880871109671   # -1.2%  planar_flow < 0.2534 and width > 0.006097
        + 0.010742956184178658 * max(0.0, Q.girth - 0.076081777364) * max(0.0, 7.0 - Q.n_pt_above_50) / 0.026536230156759166   # +1.1%  girth > 0.07608 and n_pt_above_50 < 7
        - 0.010246015054362147 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, 69.611351776123 - Q.mass) / 2.179941616634416   # -1.0%  planar_flow < 0.2534 and mass < 69.61
        - 0.010190679180135425 * max(0.0, 0.253403707141 - Q.planar_flow) * max(0.0, Q.max_dr - 0.102758520097) / 0.005604755034171544   # -1.0%  planar_flow < 0.2534 and max_dr > 0.1028
        - 0.007804211399788081 * max(0.0, 7.0 - Q.n_dr_0_0p05) / 3.110831932773109   # -0.8%  n_dr_0_0p05 < 7
        - 0.007158854838081576 * max(0.0, 0.111761856824 - Q.max_dr) / 0.028394648782258777   # -0.7%  max_dr < 0.1118
        + 0.006917605503645713 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, 48.71875 - Q.pt_7) / 0.5039522750506547   # +0.7%  centroid_offset < 0.0499 and pt_7 < 48.72
        + 0.004729317385748709 * max(0.0, 0.049903668404 - Q.centroid_offset) * max(0.0, Q.n_dr_0p05_0p1 - 2.0) / 0.036210384231240796   # +0.5%  centroid_offset < 0.0499 and n_dr_0p05_0p1 > 2
        - 0.004370251392115138 * max(0.0, 29.0421875 - Q.pt_7) / 2.1799836923983107   # -0.4%  pt_7 < 29.04
        + 0.004357485101555264 * max(0.0, 15.454033088684 - Q.mass) / 2.385786176762162   # +0.4%  mass < 15.45
        - 0.003337380938129096 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 3.558400598421145   # -0.3%  pt_7 < 29.04 and n_dr_0p2_0p4 < 2
        + 0.0030451646550342516 * max(0.0, 29.0421875 - Q.pt_7) * max(0.0, 22.844978847276 - Q.mass_top2) / 41.04635219703177   # +0.3%  pt_7 < 29.04 and mass_top2 < 22.84
        + 0.0030397847518856902 * max(0.0, 9.257203159811 - Q.mass_top5) / 2.0942092841499877   # +0.3%  mass_top5 < 9.257
        + 0.0025689408900325685 * Q.centroid_offset / 0.01718433814033805   # +0.3%  centroid_offset
        + 0.002298756401681671 * max(0.0, 0.154689112391 - Q.LHA) * max(0.0, 0.028070914944 - Q.z_7) / 5.884131049147776e-05   # +0.2%  LHA < 0.1547 and z_7 < 0.02807
        - 0.0016401276839957128 * max(0.0, 0.04447356835 - Q.e2) * max(0.0, Q.pt_5 - 43.0625) / 0.18225838803915703   # -0.2%  e2 < 0.04447 and pt_5 > 43.06
        + 0.0015992188015910322 * max(0.0, 0.006390124748 - Q.e2_sq) * max(0.0, 1.332146394253 - Q.D2) / 0.0003168302705384398   # +0.2%  e2_sq < 0.00639 and D2 < 1.332
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5773;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5773436514554158 * (-2.35461098438797
        - 0.3537507980409894 * max(0.0, Q.e2 - 0.063441075385) / 0.0017618611738475503   # -35.4%  e2 > 0.06344
        + 0.2574304265218027 * max(0.0, Q.girth2 - 0.018827652745) / 0.0007605368842505339   # +25.7%  girth2 > 0.01883
        + 0.16819346334273585 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.pt_7 - 15.55390625) / 0.014489759228566367   # +16.8%  girth2 > 0.01883 and pt_7 > 15.55
        + 0.08340921984800312 * max(0.0, Q.mass - 91.19) / 0.5839190666579981   # +8.3%  mass > 91.19
        + 0.0740262652684805 * max(0.0, Q.girth2 - 0.018827652745) * max(0.0, Q.lam2 - 0.000537286005) / 2.828867171668773e-06   # +7.4%  girth2 > 0.01883 and lam2 > 0.0005373
        + 0.03355933996520212 * max(0.0, Q.n_dr_0p2_0p4 - 1.0) / 0.1528857142857143   # +3.4%  n_dr_0p2_0p4 > 1
        + 0.02963048701278621 * max(0.0, Q.mean_phi - 0.026127964072) / 0.000699941107518642   # +3.0%  mean_phi > 0.02613
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 19.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 19.760570350366038 * (0.12912085560823813
        + 0.21281674111551388 * max(0.0, 0.148408418149 - Q.girth) / 0.09059989578347671   # +21.3%  girth < 0.1484
        + 0.10045117821997729 * max(0.0, 0.016433749775 - Q.lam1) / 0.011201097678923027   # +10.0%  lam1 < 0.01643
        - 0.0993783835126554 * max(0.0, 0.050284641981 - Q.e2) / 0.02502151752730684   # -9.9%  e2 < 0.05028
        - 0.06782662499071573 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556391352   # -6.8%  width < 0.00752
        + 0.06096849492592769 * max(0.0, 0.013238675006 - Q.width) / 0.008236124461880547   # +6.1%  width < 0.01324
        + 0.04892173926804816 * max(0.0, 0.037760993714 - Q.centroid_offset) / 0.022266027307928225   # +4.9%  centroid_offset < 0.03776
        + 0.046891708517957195 * max(0.0, 0.006506575659 - Q.lam1) / 0.00289007769049566   # +4.7%  lam1 < 0.006507
        - 0.03798738163896033 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.01998233930103558   # -3.8%  girth < 0.1484 and log_sum_pt < 6.804
        - 0.03696398921596371 * max(0.0, 0.148408418149 - Q.girth) * max(0.0, 38.53125 - Q.pt_7) / 0.6708193581740943   # -3.7%  girth < 0.1484 and pt_7 < 38.53
        - 0.03448748272390308 * max(0.0, 0.000306123359 - Q.lam2) * max(0.0, 0.049903668404 - Q.centroid_offset) / 7.4639045371768525e-06   # -3.4%  lam2 < 0.0003061 and centroid_offset < 0.0499
        - 0.03302939797277099 * max(0.0, Q.LHA - 0.09323897448) / 0.1512501993838343   # -3.3%  LHA > 0.09324
        + 0.03015414168968139 * max(0.0, 0.000306123359 - Q.lam2) / 0.00019890943789781996   # +3.0%  lam2 < 0.0003061
        + 0.02608939769769803 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, 43.5 - Q.pt_7) / 927.286113873142   # +2.6%  sum_pt_top5 > 658.1 and pt_7 < 43.5
        - 0.022715309452071744 * max(0.0, 0.501026660204 - Q.tau21) * max(0.0, Q.max_dr - 0.015595615841) / 0.027815441892245107   # -2.3%  tau21 < 0.501 and max_dr > 0.0156
        - 0.021289497157128117 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 0.037760993714 - Q.centroid_offset) / 0.0002788670798897511   # -2.1%  lam1 < 0.01643 and centroid_offset < 0.03776
        - 0.01623655714157982 * max(0.0, 0.016433749775 - Q.lam1) * max(0.0, 56.53125 - Q.pt_6) / 0.19240882891134978   # -1.6%  lam1 < 0.01643 and pt_6 < 56.53
        + 0.013316908874967904 * max(0.0, 0.067272114405 - Q.z_6) / 0.013206132668212321   # +1.3%  z_6 < 0.06727
        - 0.012785336699253018 * max(0.0, 25.578125 - Q.pt_7) / 1.3273888893448005   # -1.3%  pt_7 < 25.58
        - 0.009796420327059914 * max(0.0, 763.825 - Q.sum_pt) / 96.68366216443064   # -1.0%  sum_pt < 763.8
        - 0.009518877979249376 * max(0.0, Q.sum_pt - 988.4078125) / 4.402477808070966   # -1.0%  sum_pt > 988.4
        + 0.008409595644864341 * max(0.0, 0.050284641981 - Q.e2) * max(0.0, Q.pt_dispersion - 0.396830244362) / 0.0016492880690503395   # +0.8%  e2 < 0.05028 and pt_dispersion > 0.3968
        - 0.008133095661651895 * max(0.0, Q.C2 - 0.067292226106) / 0.002854902335729804   # -0.8%  C2 > 0.06729
        + 0.007216770913965499 * max(0.0, Q.sum_pt_top5 - 839.9546875) / 8.60588864380794   # +0.7%  sum_pt_top5 > 840
        - 0.00629561504107489 * max(0.0, 0.501026660204 - Q.tau21) / 0.20939941990711786   # -0.6%  tau21 < 0.501
        - 0.006216533612824003 * max(0.0, 0.028070914944 - Q.z_7) / 0.0013098148516826897   # -0.6%  z_7 < 0.02807
        + 0.005752730300956766 * max(0.0, 531.1875 - Q.sum_pt_top5) / 43.88237834164916   # +0.6%  sum_pt_top5 < 531.2
        - 0.0045696147845417446 * max(0.0, Q.sum_pt_top5 - 658.125) * max(0.0, Q.z_7 - 0.023207568189) / 0.28990840371630444   # -0.5%  sum_pt_top5 > 658.1 and z_7 > 0.02321
        + 0.002567547734858754 * max(0.0, Q.sum_pt_top5 - 658.125) / 46.55591824366466   # +0.3%  sum_pt_top5 > 658.1
        + 0.0021542979922659447 * max(0.0, Q.sum_pt_top5 - 902.40625) * max(0.0, 3.885568320751 - Q.D2) / 5.336336720841763   # +0.2%  sum_pt_top5 > 902.4 and D2 < 3.886
        + 0.002073138893712935 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, Q.n_pt_above_50 - 6.0) / 2.3317601365545144   # +0.2%  sum_pt > 988.4 and n_pt_above_50 > 6
        + 0.0019476406755629576 * max(0.0, Q.sum_pt_top5 - 839.9546875) * max(0.0, 0.15855820179 - Q.z_dr_0p1_0p2) / 1.2171684842632062   # +0.2%  sum_pt_top5 > 840 and z_dr_0p1_0p2 < 0.1586
        - 0.0015327550940799648 * max(0.0, Q.sum_pt - 988.4078125) * max(0.0, 3.885568320751 - Q.D2) / 7.606722001858439   # -0.2%  sum_pt > 988.4 and D2 < 3.886
        - 0.001258420690770128 * max(0.0, Q.sum_pt_top5 - 902.40625) / 4.122483656939338   # -0.1%  sum_pt_top5 > 902.4
        + 0.00024667383778756457 * max(0.0, 763.825 - Q.sum_pt) * max(0.0, 0.037477688199 - Q.z_4) / 0.0007394403345688553   # +0.0%  sum_pt < 763.8 and z_4 < 0.03748
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 41.86;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 41.85764761010851 * (-0.022901128066013316
        - 0.15278465472092315 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556391352   # -15.3%  width < 0.00752
        + 0.1290437225110917 * max(0.0, 0.013238675334 - Q.girth2) / 0.008236124741758945   # +12.9%  girth2 < 0.01324
        + 0.11195008955749255 * max(0.0, 0.008678044951 - Q.width) / 0.004447347549975801   # +11.2%  width < 0.008678
        - 0.07757598442834344 * max(0.0, 0.087236513197 - Q.girth) / 0.0363856861485123   # -7.8%  girth < 0.08724
        - 0.06779754318116353 * max(0.0, 0.011660904657 - Q.mass_over_sum_pt_sq) / 0.007206102265273649   # -6.8%  mass_over_sum_pt_sq < 0.01166
        + 0.05213851502195705 * max(0.0, 0.038466955721 - Q.e2) / 0.01567343576030707   # +5.2%  e2 < 0.03847
        - 0.036419431219359255 * max(0.0, 0.177304983139 - Q.max_dr) / 0.07042349604125828   # -3.6%  max_dr < 0.1773
        + 0.031994595335866086 * max(0.0, 0.017162483186 - Q.e2_sq) / 0.012035399208639285   # +3.2%  e2_sq < 0.01716
        - 0.03134742073796904 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 0.067292226106 - Q.C2) / 0.015914745221096264   # -3.1%  z_dr_0p05_0p1 < 0.5883 and C2 < 0.06729
        + 0.03048887579304951 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.00947067779246101   # +3.0%  width < 0.00752 and n_dr_0p1_0p2 < 3
        + 0.029893009817498532 * max(0.0, Q.lam1 - 0.002464291268) / 0.00420169868431309   # +3.0%  lam1 > 0.002464
        - 0.023393869984090446 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.019941416418612044   # -2.3%  girth2 < 0.01324 and n_dr_0p1_0p2 < 3
        + 0.01866712804173958 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) / 0.3724173891308251   # +1.9%  z_dr_0p05_0p1 < 0.5883
        - 0.016277899812870342 * max(0.0, Q.lam1 - 0.004183811014) / 0.003245626732120818   # -1.6%  lam1 > 0.004184
        - 0.013373354698306264 * max(0.0, 76.655700683594 - Q.mass) / 37.88098551900848   # -1.3%  mass < 76.66
        - 0.012394792233249855 * max(0.0, 0.008678044951 - Q.width) * max(0.0, 1.002470755577 - Q.D2) / 0.0003538891607959911   # -1.2%  width < 0.008678 and D2 < 1.002
        + 0.01216051283651043 * max(0.0, Q.lam1 - 0.005954149834) / 0.002480364544862026   # +1.2%  lam1 > 0.005954
        - 0.011769763042031813 * max(0.0, 0.035560912266 - Q.e2) / 0.01371972344513943   # -1.2%  e2 < 0.03556
        - 0.01053237988747021 * max(0.0, 5.0 - Q.n_dr_0p05_0p1) / 3.1585445378151262   # -1.1%  n_dr_0p05_0p1 < 5
        - 0.009868902310887722 * max(0.0, Q.lam1 - 0.008375572068) / 0.0018409335087223775   # -1.0%  lam1 > 0.008376
        - 0.009546527132269882 * max(0.0, 0.004372139461 - Q.girth2) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.0015191555620967026   # -1.0%  girth2 < 0.004372 and n_dr_0p2_0p4 < 1
        - 0.00935989821286088 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.111513564951 - Q.planar_flow) / 6.286786921022877e-05   # -0.9%  width < 0.00752 and planar_flow < 0.1115
        + 0.009254761674494913 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.009480684835) / 0.0002838341565069535   # +0.9%  planar_flow < 0.1115 and centroid_offset > 0.009481
        - 0.009217830618822809 * max(0.0, 1.122624260187 - Q.D2) / 0.23862964281013382   # -0.9%  D2 < 1.123
        - 0.008560548432678824 * max(0.0, Q.lam1 - 0.007330079875) / 0.0020731330414337293   # -0.9%  lam1 > 0.00733
        + 0.008426223767384368 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, Q.eccentricity - 0.970449631164) / 5.9931533843805085e-05   # +0.8%  girth2 < 0.01324 and eccentricity > 0.9704
        + 0.005870340894836004 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, 3.0 - Q.n_dr_0p1_0p2) / 0.8008591717572943   # +0.6%  z_dr_0p05_0p1 < 0.5883 and n_dr_0p1_0p2 < 3
        + 0.005556152699878603 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 6.804164030582 - Q.log_sum_pt) / 0.0006963583088931786   # +0.6%  width < 0.00752 and log_sum_pt < 6.804
        - 0.005279429981417313 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 0.15984864831 - Q.max_dr) / 0.0010568587168460638   # -0.5%  planar_flow < 0.1115 and max_dr < 0.1598
        + 0.004858708450395395 * max(0.0, 0.111513564951 - Q.planar_flow) / 0.03343186742548701   # +0.5%  planar_flow < 0.1115
        - 0.004804202796669109 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018377780003) / 0.00015097111917114598   # -0.5%  planar_flow < 0.1115 and centroid_offset > 0.01838
        + 0.0038592587524239608 * max(0.0, 1.122624260187 - Q.D2) * max(0.0, 0.031170772021 - Q.centroid_offset) / 0.0040968541791565226   # +0.4%  D2 < 1.123 and centroid_offset < 0.03117
        - 0.0037366018876961904 * max(0.0, Q.lam1 - 0.00543336053) / 0.0026771398012890727   # -0.4%  lam1 > 0.005433
        + 0.003734673303443321 * max(0.0, 0.013238675334 - Q.girth2) * max(0.0, 1.002470755577 - Q.D2) / 0.0010081226860786433   # +0.4%  girth2 < 0.01324 and D2 < 1.002
        + 0.003448857118021752 * max(0.0, 0.74595130682 - Q.D2) / 0.09626873727471312   # +0.3%  D2 < 0.746
        - 0.002943045870169281 * max(0.0, 0.004372139461 - Q.girth2) / 0.0015657103124088927   # -0.3%  girth2 < 0.004372
        - 0.0028894143651936295 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) / 0.0228131875171655   # -0.3%  z_dr_0p05_0p1 > 0.7509
        + 0.002764634272162657 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 1.0 - Q.n_dr_0p2_0p4) / 0.02067296656828152   # +0.3%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 1
        + 0.0027092552324235446 * max(0.0, Q.lam1 - 0.00543336053) * max(0.0, 0.15984864831 - Q.max_dr) / 1.2809763884348644e-05   # +0.3%  lam1 > 0.005433 and max_dr < 0.1598
        + 0.002678609666497656 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 1.002470755577 - Q.D2) / 0.00021180584792542705   # +0.3%  width < 0.00752 and D2 < 1.002
        + 0.0024143707688052637 * max(0.0, 0.038466955721 - Q.e2) * max(0.0, 1.002470755577 - Q.D2) / 0.0005071967096445531   # +0.2%  e2 < 0.03847 and D2 < 1.002
        - 0.002365749675326244 * max(0.0, 76.655700683594 - Q.mass) * max(0.0, 0.74595130682 - Q.D2) / 1.4494675524944256   # -0.2%  mass < 76.66 and D2 < 0.746
        - 0.0022473142179307588 * max(0.0, 0.111513564951 - Q.planar_flow) * max(0.0, 739.5 - Q.sum_pt) / 1.9441833461309348   # -0.2%  planar_flow < 0.1115 and sum_pt < 739.5
        + 0.0022090710220564937 * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) * max(0.0, Q.eccentricity - 0.970449631164) / 0.0023597221386163136   # +0.2%  z_dr_0p05_0p1 < 0.5883 and eccentricity > 0.9704
        + 0.0007887445068432747 * max(0.0, Q.lam1 - 0.007330079875) * max(0.0, 0.13261153996 - Q.max_dr) / 7.5871203399465e-07   # +0.1%  lam1 > 0.00733 and max_dr < 0.1326
        + 0.000603329507427364 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, Q.eccentricity - 0.984196588116) / 0.00015706677923478843   # +0.1%  z_dr_0p05_0p1 > 0.7509 and eccentricity > 0.9842
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 35.38;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 35.37668995060162 * (-0.055319313531149425
        + 0.12062726239993844 * max(0.0, 0.013238675006 - Q.width) / 0.008236124461880547   # +12.1%  width < 0.01324
        - 0.1094878235249815 * max(0.0, 0.006679471358 - Q.width) / 0.002936239797733741   # -10.9%  width < 0.006679
        + 0.06283901716947163 * max(0.0, 0.041109715588 - Q.e2) / 0.01758452927609984   # +6.3%  e2 < 0.04111
        - 0.062304510049899633 * max(0.0, 0.008375572068 - Q.lam1) / 0.0043001452793647   # -6.2%  lam1 < 0.008376
        + 0.05793526575852149 * max(0.0, 0.006506575659 - Q.lam1) / 0.00289007769049566   # +5.8%  lam1 < 0.006507
        + 0.05543422191366248 * max(0.0, 0.007182835724 - Q.mass_over_sum_pt_sq) / 0.0035288487287591422   # +5.5%  mass_over_sum_pt_sq < 0.007183
        - 0.054669637975121255 * max(0.0, 0.008168570676 - Q.e2_sq) / 0.004294747328227963   # -5.5%  e2_sq < 0.008169
        + 0.05294454062001371 * max(0.0, 0.024547699839 - Q.e2) / 0.007455820719602151   # +5.3%  e2 < 0.02455
        - 0.05170537243681972 * max(0.0, 0.00752008842 - Q.width) / 0.003544990556391352   # -5.2%  width < 0.00752
        + 0.04949044958965644 * max(0.0, 0.018827652745 - Q.girth2) / 0.013142955535857035   # +4.9%  girth2 < 0.01883
        - 0.045988163422482735 * max(0.0, 0.011657374702 - Q.e2_sq) / 0.007203073289124996   # -4.6%  e2_sq < 0.01166
        - 0.04116519391354959 * max(0.0, 0.012003726523 - Q.lam1) / 0.007316287539010952   # -4.1%  lam1 < 0.012
        + 0.028469002028194752 * max(0.0, Q.girth - 0.033604209498) / 0.03159705075660309   # +2.8%  girth > 0.0336
        + 0.025695757191608628 * max(0.0, 0.328461505473 - Q.z_dr_0p1_0p2) / 0.22863861264226853   # +2.6%  z_dr_0p1_0p2 < 0.3285
        - 0.015643797617567772 * max(0.0, 0.001130644719 - Q.lam2) / 0.0009137232342239988   # -1.6%  lam2 < 0.001131
        - 0.015563067407784644 * max(0.0, Q.LHA - 0.346713497427) / 0.008898883584980057   # -1.6%  LHA > 0.3467
        - 0.014499973150851444 * max(0.0, 0.06813910019 - Q.mass_over_sum_pt) / 0.02225834320138369   # -1.4%  mass_over_sum_pt < 0.06814
        + 0.014170485938160915 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.010358830282549927   # +1.4%  tau21 < 0.238 and z_dr_0p2_0p4 < 0.2055
        - 0.013131573844782257 * max(0.0, 0.002151567843 - Q.girth2_top3) / 0.0007789602594301087   # -1.3%  girth2_top3 < 0.002152
        - 0.012802950685378804 * max(0.0, 0.000306123359 - Q.lam2) / 0.00019890943789781996   # -1.3%  lam2 < 0.0003061
        + 0.009689765509092724 * max(0.0, 0.006096650059 - Q.width) / 0.002544039033914621   # +1.0%  width < 0.006097
        + 0.00911773164853807 * max(0.0, 0.23799610585 - Q.tau21) / 0.05620226106196825   # +0.9%  tau21 < 0.238
        - 0.008232346034005569 * max(0.0, 36.229410171509 - Q.mass) / 10.113929790442333   # -0.8%  mass < 36.23
        + 0.008006573878658664 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.sum_pt_top3 - 353.0625) / 6.205553603061063   # +0.8%  LHA > 0.1767 and sum_pt_top3 > 353.1
        - 0.005789896850778927 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) / 0.0228131875171655   # -0.6%  z_dr_0p05_0p1 > 0.7509
        + 0.005424257004410945 * max(0.0, 2.0 - Q.n_dr_0_0p05) / 0.6358991596638656   # +0.5%  n_dr_0_0p05 < 2
        + 0.0052651328213199535 * max(0.0, 0.007639643088 - Q.girth2_top2) / 0.004543328598712961   # +0.5%  girth2_top2 < 0.00764
        - 0.00515915466700015 * max(0.0, 0.063441075385 - Q.e2) / 0.03657078287722135   # -0.5%  e2 < 0.06344
        - 0.004819931827895652 * max(0.0, 0.00752008842 - Q.width) * max(0.0, Q.e2 - 0.024547699839) / 3.760226294488998e-06   # -0.5%  width < 0.00752 and e2 > 0.02455
        + 0.004383688159160787 * max(0.0, Q.log_sum_pt - 6.896095378249) / 0.0040348977447761045   # +0.4%  log_sum_pt > 6.896
        - 0.004315829379512221 * max(0.0, 0.006096650059 - Q.width) * max(0.0, Q.log_sum_pt - 6.896095378249) / 1.9841456755321017e-05   # -0.4%  width < 0.006097 and log_sum_pt > 6.896
        - 0.004177368856205747 * max(0.0, 0.74595130682 - Q.D2) / 0.09626873727471312   # -0.4%  D2 < 0.746
        - 0.003993088835028547 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.588259786367 - Q.z_dr_0p05_0p1) / 0.012459234137161384   # -0.4%  tau21 < 0.238 and z_dr_0p05_0p1 < 0.5883
        - 0.003183037002818944 * max(0.0, 0.00752008842 - Q.width) * max(0.0, 0.061262048692 - Q.planar_flow) / 2.128824851640754e-05   # -0.3%  width < 0.00752 and planar_flow < 0.06126
        + 0.003015670401216117 * max(0.0, 0.006506575659 - Q.lam1) * max(0.0, Q.pt1_dr01 - 1.21960336377) / 0.003127732056631688   # +0.3%  lam1 < 0.006507 and pt1_dr01 > 1.22
        - 0.002297506474260096 * max(0.0, 0.008375572068 - Q.lam1) * max(0.0, Q.m01 - 16.308019673264) / 0.0029524506110795517   # -0.2%  lam1 < 0.008376 and m01 > 16.31
        - 0.002256297053785694 * max(0.0, 0.23799610585 - Q.tau21) * max(0.0, 0.121680960059 - Q.max_dr) / 0.0005372705453095934   # -0.2%  tau21 < 0.238 and max_dr < 0.1217
        + 0.0019920856145774235 * max(0.0, Q.n_dr_0p05_0p1 - 5.0) / 0.2904857142857143   # +0.2%  n_dr_0p05_0p1 > 5
        - 0.001679291952300467 * max(0.0, Q.mass - 80.4) * max(0.0, 0.20552001074 - Q.z_dr_0p2_0p4) / 0.08017783625220609   # -0.2%  mass > 80.4 and z_dr_0p2_0p4 < 0.2055
        + 0.0012392646021847661 * max(0.0, Q.z_dr_0p05_0p1 - 0.750909513235) * max(0.0, 2.0 - Q.n_dr_0p2_0p4) / 0.04262928713838186   # +0.1%  z_dr_0p05_0p1 > 0.7509 and n_dr_0p2_0p4 < 2
        + 0.0007460893545357841 * max(0.0, 0.018827652745 - Q.girth2) * max(0.0, Q.mass_top2 - 22.844978847276) / 0.011940644746868265   # +0.1%  girth2 < 0.01883 and mass_top2 > 22.84
        - 0.0006479254342651715 * max(0.0, Q.LHA - 0.176724128067) * max(0.0, Q.z_top5 - 0.865048766136) / 0.00029995334888511626   # -0.1%  LHA > 0.1767 and z_top5 > 0.865
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0838271008403362, 0.5021306722689076, 2.370731512605042, 0.6077867647058823, 3.941731512605042, 2.3874502100840336, 1.6046012605042017, 1.6071586134453781, 0.7084941176470588, 4.366464285714286, 3.220442857142857, 2.5599957983193278, 0.07600987394957984, 5.376722899159664, 0.49238466386554625, 0.6404726890756303]
T = [2.8809818113182772, 2.277562570575105, 4.006392635569853, 2.8201887949711133, 4.707307796087185]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +35%, n9 +26%, n5 -16%, n1 +7%, n6 +6%, n0 -6% ...
            + 0.3535856050255156 * h[2] / H_AVG[2]
            + 0.2604966286697021 * h[9] / H_AVG[9]
            - 0.15537998630609973 * h[5] / H_AVG[5]
            + 0.06808262137735964 * h[1] / H_AVG[1]
            + 0.06091786563114066 * h[6] / H_AVG[6]
            - 0.058781344554484506 * h[0] / H_AVG[0]
            - 0.04275594843569781 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +49%, n10 -18%, n4 -16%, n6 +9%, n5 +5%, n8 +2% ...
            + 0.4867802917329701 * h[9] / H_AVG[9]
            - 0.1767483196043252 * h[10] / H_AVG[10]
            - 0.1622512303639637 * h[4] / H_AVG[4]
            + 0.0880657068018018 * h[6] / H_AVG[6]
            + 0.049136620896184806 * h[5] / H_AVG[5]
            + 0.019442224299356946 * h[8] / H_AVG[8]
            + 0.017575606301397494 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +24%, n6 -13%, n15 -11%, n13 +9%, n0 +9%, n14 -9% ...
            + 0.23961666059552392 * h[11] / H_AVG[11]
            - 0.12515944879082988 * h[6] / H_AVG[6]
            - 0.10990559682797185 * h[15] / H_AVG[15]
            + 0.09436190189916109 * h[13] / H_AVG[13]
            + 0.09299277424936495 * h[0] / H_AVG[0]
            - 0.09217481447537496 * h[14] / H_AVG[14]
            + 0.08775124623829367 * h[7] / H_AVG[7]
            - 0.07585212184519619 * h[3] / H_AVG[3]
            - 0.044210227384908166 * h[8] / H_AVG[8]
            - 0.03405857122367715 * h[9] / H_AVG[9]
            - 0.003916636469698247 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +27%, n6 -21%, n3 -12%, n4 +11%, n13 +10%, n14 +7% ...
            + 0.2671294919673055 * h[7] / H_AVG[7]
            - 0.21336354281034542 * h[6] / H_AVG[6]
            - 0.12122594620498116 * h[3] / H_AVG[3]
            + 0.1091940280634379 * h[4] / H_AVG[4]
            + 0.10426235792161068 * h[13] / H_AVG[13]
            + 0.06547230074767782 * h[14] / H_AVG[14]
            - 0.04838399796917464 * h[9] / H_AVG[9]
            - 0.03548480791304338 * h[15] / H_AVG[15]
            + 0.02225607524770566 * h[1] / H_AVG[1]
            + 0.013227451154717861 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -46%, n10 +26%, n5 -13%, n4 +10%, n8 +3%, n12 -1% ...
            - 0.4640218512159424 * h[13] / H_AVG[13]
            + 0.2565513290701767 * h[10] / H_AVG[10]
            - 0.12679488539439324 * h[5] / H_AVG[5]
            + 0.10467053789964334 * h[4] / H_AVG[4]
            + 0.028220514318023818 * h[8] / H_AVG[8]
            - 0.00807360355878586 * h[12] / H_AVG[12]
            + 0.008069723595659706 * h[3] / H_AVG[3]
            + 0.0035975549473749773 * h[0] / H_AVG[0]
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
