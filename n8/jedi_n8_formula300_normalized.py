"""JEDI-linear jet tagger, 8 particles, 3 features: the formula with the fewest quantities (24) at the main result's accuracy (from the 931-term tuned formula), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  15.7%   (on for 92% of jets)
  neuron  9:  12.4%   (on for 72% of jets)
  neuron  4:  11.9%   (on for 91% of jets)
  neuron  3:   8.4%   (on for 26% of jets)
  neuron  6:   8.1%   (on for 53% of jets)
  neuron 11:   7.0%   (on for 82% of jets)
  neuron  5:   5.7%   (on for 64% of jets)
  neuron  7:   5.3%   (on for 52% of jets)
  neuron  2:   4.9%   (on for 92% of jets)
  neuron 10:   4.8%   (on for 88% of jets)
  neuron  8:   3.4%   (on for 43% of jets)
  neuron 14:   3.3%   (on for 33% of jets)
  neuron  0:   3.3%   (on for 47% of jets)
  neuron  1:   2.8%   (on for 69% of jets)
  neuron 15:   2.7%   (on for 36% of jets)
  neuron 12:   0.2%   (on for 4% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.5% (the network: 65.8%); same class as the network for 87.5% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.centroid_offset        distance of the pT centroid from the jet axis
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        pt_7=pt[7],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 16.37;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.368683193482717 * (-0.055899426312087976
        + 0.24834027891640806 * max(0.0, 0.0128 - Q.girth2) / 0.007862675724881661   # +24.8%  girth2 < 0.0128
        - 0.1093887432401612 * max(0.0, 0.00142 - Q.lam1) / 0.0003639328624047607   # -10.9%  lam1 < 0.00142
        - 0.09480076140582776 * max(0.0, 0.0768 - Q.girth) / 0.028472727155099777   # -9.5%  girth < 0.0768
        - 0.09207314881433701 * max(0.0, 29.3 - Q.mass) / 7.211082313723786   # -9.2%  mass < 29.3
        - 0.08753383957238203 * max(0.0, 0.00459 - Q.width) / 0.0016758054838239326   # -8.8%  width < 0.00459
        - 0.08220992428431874 * max(0.0, 59.8 - Q.mass) / 23.859365354081824   # -8.2%  mass < 59.8
        + 0.057892664974229964 * max(0.0, 0.000706 - Q.lam1) / 0.00014038913958364498   # +5.8%  lam1 < 0.000706
        + 0.04630623532494626 * max(0.0, 0.0252 - Q.e2) / 0.007782054372863486   # +4.6%  e2 < 0.0252
        + 0.03676258274955117 * max(0.0, 0.158 - Q.planar_flow) * max(0.0, 0.0558 - Q.centroid_offset) / 0.002212334817652926   # +3.7%  planar_flow < 0.158 and centroid_offset < 0.0558
        - 0.030054046293244513 * max(0.0, Q.sum_pt - 816.0) / 30.939947324300814   # -3.0%  sum_pt > 816
        + 0.027074717975719424 * max(0.0, Q.sum_pt_top5 - 699.0) / 33.83034206850709   # +2.7%  sum_pt_top5 > 699
        - 0.01930974091238658 * max(0.0, 64.6 - Q.mass) * max(0.0, 38.6 - Q.pt_7) / 202.61219970710758   # -1.9%  mass < 64.6 and pt_7 < 38.6
        - 0.013966280242356991 * max(0.0, Q.sum_pt - 890.0) / 14.199355073201156   # -1.4%  sum_pt > 890
        + 0.01157559396080773 * max(0.0, 67.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0133) / 0.16917609850076207   # +1.2%  mass < 67.8 and centroid_offset > 0.0133
        + 0.010379026211429892 * max(0.0, 56.8 - Q.mass) * max(0.0, Q.C2 - 0.0184) / 0.10296423752227217   # +1.0%  mass < 56.8 and C2 > 0.0184
        + 0.009468232285015744 * max(0.0, Q.sum_pt - 905.0) * max(0.0, 26.0 - Q.pt_7) / 68.57632507775561   # +0.9%  sum_pt > 905 and pt_7 < 26
        - 0.00929954958668274 * max(0.0, 0.00646 - Q.lam1) * max(0.0, 0.866 - Q.D2) / 7.425433220804532e-05   # -0.9%  lam1 < 0.00646 and D2 < 0.866
        - 0.0068193581108730115 * max(0.0, Q.D2 - 3.7) / 0.13040176693900363   # -0.7%  D2 > 3.7
        - 0.004415493722884309 * max(0.0, Q.sum_pt - 901.0) * max(0.0, Q.pt_7 - 25.9) / 105.05206089055933   # -0.4%  sum_pt > 901 and pt_7 > 25.9
        + 0.0023297814164368134 * max(0.0, Q.eccentricity - 0.997) / 0.00010391131857143768   # +0.2%  eccentricity > 0.997
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 6.121;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.121283308529243 * (0.17806723607796773
        + 0.10346463461000427 * max(0.0, Q.log_sum_pt - 6.4) / 0.19309034782356593   # +10.3%  log_sum_pt > 6.4
        - 0.09869079989526165 * max(0.0, 0.252 - Q.max_dr) / 0.13277238375917907   # -9.9%  max_dr < 0.252
        - 0.0904237902373169 * max(0.0, 0.00911 - Q.lam1) * max(0.0, Q.z_7 - 0.0272) / 0.00010917349859440583   # -9.0%  lam1 < 0.00911 and z_7 > 0.0272
        - 0.0872702884853544 * max(0.0, 0.0515 - Q.z_7) / 0.008439275833111087   # -8.7%  z_7 < 0.0515
        + 0.07696435184133545 * max(0.0, Q.log_sum_pt - 6.61) * max(0.0, 0.00151 - Q.lam2) / 0.00010045215400386751   # +7.7%  log_sum_pt > 6.61 and lam2 < 0.00151
        + 0.07626790753488577 * max(0.0, Q.pt_7 - 33.5) / 5.0092003151260505   # +7.6%  pt_7 > 33.5
        - 0.07376947738944009 * max(0.0, Q.log_sum_pt - 6.47) * max(0.0, 0.0272 - Q.centroid_offset) / 0.002610195783947428   # -7.4%  log_sum_pt > 6.47 and centroid_offset < 0.0272
        - 0.06203008124356494 * max(0.0, 0.00946 - Q.width) * max(0.0, 0.0841 - Q.planar_flow) / 7.137287611709529e-05   # -6.2%  width < 0.00946 and planar_flow < 0.0841
        + 0.057778050300573834 * max(0.0, Q.LHA - 0.282) / 0.025444303230522707   # +5.8%  LHA > 0.282
        - 0.055909111341188644 * max(0.0, Q.e2 - 0.0316) / 0.008306687137075754   # -5.6%  e2 > 0.0316
        + 0.044508030070534665 * max(0.0, Q.log_sum_pt - 6.36) * max(0.0, 0.195 - Q.max_dr) / 0.021971472706958185   # +4.5%  log_sum_pt > 6.36 and max_dr < 0.195
        - 0.04206260777295826 * max(0.0, 0.0389 - Q.C2) * max(0.0, 0.263 - Q.tau21) / 0.0013480478475069203   # -4.2%  C2 < 0.0389 and tau21 < 0.263
        + 0.03560291998753881 * max(0.0, 0.223 - Q.tau21) * max(0.0, 0.000218 - Q.lam2) / 7.36268783292647e-06   # +3.6%  tau21 < 0.223 and lam2 < 0.000218
        + 0.025358235613973323 * max(0.0, 0.00989 - Q.lam1) * max(0.0, Q.centroid_offset - 0.0201) / 1.0931334112504698e-05   # +2.5%  lam1 < 0.00989 and centroid_offset > 0.0201
        + 0.024432902345444745 * max(0.0, Q.log_sum_pt - 6.55) * max(0.0, 1.2 - Q.D2) / 0.02085923532860473   # +2.4%  log_sum_pt > 6.55 and D2 < 1.2
        - 0.023034212255847418 * max(0.0, 0.00749 - Q.e2) / 0.0008757698074958914   # -2.3%  e2 < 0.00749
        + 0.01865500181559008 * max(0.0, 0.0346 - Q.e2) * max(0.0, Q.eccentricity - 0.979) / 3.0451346995827807e-05   # +1.9%  e2 < 0.0346 and eccentricity > 0.979
        + 0.0037775972591867447 * max(0.0, Q.pt_7 - 35.4) * max(0.0, Q.centroid_offset - 0.0188) / 0.01482291221090105   # +0.4%  pt_7 > 35.4 and centroid_offset > 0.0188
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 10.78;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.778258347687586 * (0.47874154001022334
        - 0.22163204958818383 * Q.LHA / 0.24276498867773896   # -22.2%  LHA
        - 0.11974251685793455 * max(0.0, 36.7 - Q.mass) / 10.324926255177237   # -12.0%  mass < 36.7
        + 0.093803937246968 * max(0.0, 794.0 - Q.sum_pt) / 115.81249366301208   # +9.4%  sum_pt < 794
        + 0.0899869711032665 * max(0.0, 38.0 - Q.mass) * max(0.0, 0.00111 - Q.lam2) / 0.011464572369703356   # +9.0%  mass < 38 and lam2 < 0.00111
        + 0.08708292638684356 * max(0.0, 0.00591 - Q.lam1) / 0.0024896612155704515   # +8.7%  lam1 < 0.00591
        - 0.07423501202095205 * max(0.0, 714.0 - Q.sum_pt_top5) / 150.399274061187   # -7.4%  sum_pt_top5 < 714
        - 0.06674718018376104 * max(0.0, 54.0 - Q.pt_7) / 19.65623912568934   # -6.7%  pt_7 < 54
        + 0.05877244272628496 * max(0.0, Q.pt_7 - 31.0) / 6.477142857142857   # +5.9%  pt_7 > 31
        - 0.05403569021808764 * max(0.0, 0.748 - Q.planar_flow) / 0.48133109848443273   # -5.4%  planar_flow < 0.748
        - 0.04018815046328254 * max(0.0, Q.pt_7 - 31.2) * max(0.0, 0.05 - Q.C2) / 0.19081862035638758   # -4.0%  pt_7 > 31.2 and C2 < 0.05
        - 0.02461740334956919 * max(0.0, 72.6 - Q.mass) * max(0.0, 0.0637 - Q.z_7) / 0.5831488640678822   # -2.5%  mass < 72.6 and z_7 < 0.0637
        + 0.019769349346478754 * max(0.0, 6.43 - Q.log_sum_pt) / 0.05886164492321457   # +2.0%  log_sum_pt < 6.43
        - 0.01014272342774761 * max(0.0, Q.pt_7 - 29.5) * max(0.0, Q.max_dr - 0.111) / 0.21310115682925365   # -1.0%  pt_7 > 29.5 and max_dr > 0.111
        - 0.009317847348748706 * max(0.0, 0.00581 - Q.lam1) * max(0.0, Q.max_dr - 0.0819) / 4.1159904085708767e-05   # -0.9%  lam1 < 0.00581 and max_dr > 0.0819
        - 0.006950415945100593 * max(0.0, 0.00343 - Q.lam1) * max(0.0, 0.0071 - Q.centroid_offset) / 2.4888165674478863e-06   # -0.7%  lam1 < 0.00343 and centroid_offset < 0.0071
        - 0.006722218980163286 * max(0.0, 73.7 - Q.mass) * max(0.0, Q.max_pair_mass - 12.4) / 46.44475181918517   # -0.7%  mass < 73.7 and max_pair_mass > 12.4
        - 0.005644450911935856 * max(0.0, Q.pt_7 - 28.1) * max(0.0, Q.centroid_offset - 0.0132) / 0.05685733659783687   # -0.6%  pt_7 > 28.1 and centroid_offset > 0.0132
        - 0.005567808641762153 * max(0.0, 0.0301 - Q.z_7) * max(0.0, 0.00468 - Q.width) / 5.218372171426087e-06   # -0.6%  z_7 < 0.0301 and width < 0.00468
        - 0.002737147571942708 * max(0.0, 4.88e-05 - Q.girth2) / 8.676965784160088e-07   # -0.3%  girth2 < 4.88e-05
        - 0.00230375768098632 * max(0.0, Q.log_sum_pt - 6.93) / 0.0027256306757563334   # -0.2%  log_sum_pt > 6.93
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 27.76;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.764493850169547 * (0.26904866482751977
        - 0.33861789895687316 * max(0.0, 0.121 - Q.girth) / 0.06528857342462109   # -33.9%  girth < 0.121
        - 0.13592857934507335 * max(0.0, 0.00863 - Q.girth2) / 0.004408864725804407   # -13.6%  girth2 < 0.00863
        + 0.088984099253772 * max(0.0, 0.00539 - Q.width) / 0.002111622629482247   # +8.9%  width < 0.00539
        + 0.06556280346104047 * max(0.0, 0.0397 - Q.e2) / 0.016548345940853935   # +6.6%  e2 < 0.0397
        - 0.053835275767363834 * max(0.0, Q.e2 - 0.0263) / 0.01067649416403682   # -5.4%  e2 > 0.0263
        + 0.04941830876326355 * max(0.0, Q.width - 0.0185) / 0.0007931065489846303   # +4.9%  width > 0.0185
        + 0.04678068709758317 * max(0.0, Q.LHA - 0.326) / 0.012369924754548131   # +4.7%  LHA > 0.326
        + 0.030543003280375357 * max(0.0, Q.lam1 - 0.0088) * max(0.0, 36.3 - Q.pt_7) / 0.006894398591412113   # +3.1%  lam1 > 0.0088 and pt_7 < 36.3
        + 0.027656364715697965 * max(0.0, Q.mass - 35.4) * max(0.0, Q.eccentricity - 0.705) / 3.3678288073115885   # +2.8%  mass > 35.4 and eccentricity > 0.705
        - 0.023763728560437795 * max(0.0, Q.lam1 - 0.00553) * max(0.0, 0.0813 - Q.z_7) / 4.851381584363041e-05   # -2.4%  lam1 > 0.00553 and z_7 < 0.0813
        + 0.023129063274876918 * max(0.0, Q.max_dr - 0.122) / 0.03547882514118798   # +2.3%  max_dr > 0.122
        - 0.01954649864520888 * max(0.0, Q.LHA - 0.322) * max(0.0, Q.planar_flow - 0.00227) / 0.004678436564028011   # -2.0%  LHA > 0.322 and planar_flow > 0.00227
        + 0.01787954088600239 * max(0.0, Q.LHA - 0.313) * max(0.0, Q.eccentricity - 0.957) / 0.00020344934548084765   # +1.8%  LHA > 0.313 and eccentricity > 0.957
        - 0.01756522513631804 * max(0.0, Q.mass - 75.6) / 1.653185034827613   # -1.8%  mass > 75.6
        + 0.015063196355736526 * max(0.0, Q.centroid_offset - 0.0106) / 0.009013405659110843   # +1.5%  centroid_offset > 0.0106
        - 0.013440800290185509 * max(0.0, Q.e2 - 0.0319) * max(0.0, Q.n_pt_above_50 - 1.68) / 0.02487846779988084   # -1.3%  e2 > 0.0319 and n_pt_above_50 > 1.68
        - 0.008104082863241623 * max(0.0, Q.LHA - 0.312) * max(0.0, 0.149 - Q.max_dr) / 5.1606825416912026e-05   # -0.8%  LHA > 0.312 and max_dr < 0.149
        - 0.007877588984957834 * max(0.0, Q.C2 - 0.0894) / 0.0011274086130259027   # -0.8%  C2 > 0.0894
        + 0.0060870006143026074 * max(0.0, Q.lam1 - 0.0147) * max(0.0, Q.eccentricity - 0.957) / 1.2246557327665436e-05   # +0.6%  lam1 > 0.0147 and eccentricity > 0.957
        - 0.0052724194181020334 * max(0.0, Q.centroid_offset - 0.0495) / 0.0008270398672847934   # -0.5%  centroid_offset > 0.0495
        - 0.00234331477206986 * max(0.0, Q.LHA - 0.424) * max(0.0, Q.pt_7 - 38.5) / 0.0019479325921600325   # -0.2%  LHA > 0.424 and pt_7 > 38.5
        + 0.0013604320410437632 * max(0.0, Q.mass - 75.1) * max(0.0, 0.185 - Q.max_dr) / 0.007584680127938386   # +0.1%  mass > 75.1 and max_dr < 0.185
        + 0.0012400875164733497 * max(0.0, Q.mass_over_sum_pt - 0.0901) * max(0.0, 0.147 - Q.max_dr) / 3.746507314994162e-06   # +0.1%  mass_over_sum_pt > 0.0901 and max_dr < 0.147
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 15.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.572584397547358 * (0.11879852134828503
        + 0.16509249075450802 * max(0.0, Q.width - 0.000872) / 0.005751491601064615   # +16.5%  width > 0.000872
        - 0.1506717220764195 * max(0.0, Q.mass_over_sum_pt - 0.0906) / 0.008261789113939584   # -15.1%  mass_over_sum_pt > 0.0906
        + 0.10141753837461161 * max(0.0, 0.238 - Q.tau21) / 0.0562040276060547   # +10.1%  tau21 < 0.238
        + 0.09016768724099554 * max(0.0, 47.9 - Q.mass) / 16.029040176849968   # +9.0%  mass < 47.9
        + 0.08633805677179182 * max(0.0, 0.0602 - Q.C2) / 0.03528888912858171   # +8.6%  C2 < 0.0602
        - 0.07208733488448155 * max(0.0, Q.girth - 0.0651) / 0.014560131082786625   # -7.2%  girth > 0.0651
        - 0.06006697430909562 * max(0.0, 0.0429 - Q.centroid_offset) / 0.02695671547359367   # -6.0%  centroid_offset < 0.0429
        + 0.05274766339276489 * max(0.0, Q.e2 - 0.0328) / 0.0078230232376881   # +5.3%  e2 > 0.0328
        + 0.0435802529803569 * max(0.0, 762.0 - Q.sum_pt) / 95.58551656381303   # +4.4%  sum_pt < 762
        + 0.03684635462836903 * max(0.0, Q.mass_over_sum_pt - 0.109) / 0.005216299701747603   # +3.7%  mass_over_sum_pt > 0.109
        - 0.032522568216171305 * max(0.0, Q.C2 - 0.0604) / 0.003591918002633467   # -3.3%  C2 > 0.0604
        - 0.02617420639141132 * max(0.0, 0.278 - Q.tau21) * max(0.0, 63.5 - Q.mass) / 0.6804675092972893   # -2.6%  tau21 < 0.278 and mass < 63.5
        - 0.01908657764282253 * max(0.0, Q.max_dr - 0.11) * max(0.0, Q.eccentricity - 0.982) / 0.0001955443034231542   # -1.9%  max_dr > 0.11 and eccentricity > 0.982
        + 0.01890404975947257 * max(0.0, 439.0 - Q.sum_pt_top5) / 15.742508574054622   # +1.9%  sum_pt_top5 < 439
        + 0.015572581014938282 * max(0.0, 0.294 - Q.tau21) * max(0.0, Q.planar_flow - 0.00357) / 0.005461831805017345   # +1.6%  tau21 < 0.294 and planar_flow > 0.00357
        + 0.012018155448113051 * max(0.0, Q.C2 - 0.00455) * max(0.0, Q.pt_7 - 38.6) / 0.04282694279601466   # +1.2%  C2 > 0.00455 and pt_7 > 38.6
        - 0.011877774594223109 * max(0.0, Q.C2 - 0.067) * max(0.0, 36.4 - Q.pt_7) / 0.015038020107608387   # -1.2%  C2 > 0.067 and pt_7 < 36.4
        - 0.004828011519453456 * max(0.0, 0.267 - Q.tau21) * max(0.0, 26.3 - Q.pt_7) / 0.060632755531467575   # -0.5%  tau21 < 0.267 and pt_7 < 26.3
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 8.864;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.86409293035832 * (-0.07197577970047407
        + 0.2757941551455237 * max(0.0, 54.4 - Q.pt_7) / 20.038237875898215   # +27.6%  pt_7 < 54.4
        + 0.1582888420139528 * max(0.0, 0.0405 - Q.e2) / 0.017131709468259874   # +15.8%  e2 < 0.0405
        - 0.11981770365949566 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.0298 - Q.centroid_offset) / 0.0003948235167062839   # -12.0%  z_7 < 0.072 and centroid_offset < 0.0298
        + 0.1056950576038957 * max(0.0, 0.00234 - Q.width) * max(0.0, 0.0257 - Q.centroid_offset) / 1.1996041138034663e-05   # +10.6%  width < 0.00234 and centroid_offset < 0.0257
        - 0.0892094512411124 * max(0.0, 721.0 - Q.sum_pt_top5) / 155.66158780856094   # -8.9%  sum_pt_top5 < 721
        - 0.06407552194752165 * max(0.0, 0.217 - Q.LHA) * max(0.0, 6.83 - Q.log_sum_pt) / 0.004693978356231786   # -6.4%  LHA < 0.217 and log_sum_pt < 6.83
        - 0.04349491401697448 * max(0.0, Q.sum_pt - 935.0) / 8.291246448266806   # -4.3%  sum_pt > 935
        - 0.04083475195237333 * max(0.0, 0.076 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt) / 1.402957504240186   # -4.1%  z_7 < 0.076 and sum_pt < 803
        + 0.039649452864785356 * max(0.0, Q.log_sum_pt - 6.27) * max(0.0, Q.centroid_offset - 0.00241) / 0.002811651478650554   # +4.0%  log_sum_pt > 6.27 and centroid_offset > 0.00241
        + 0.021495326891739296 * max(0.0, Q.sum_pt - 864.0) * max(0.0, 0.0114 - Q.centroid_offset) / 0.12136087588331679   # +2.1%  sum_pt > 864 and centroid_offset < 0.0114
        + 0.015804282788429647 * max(0.0, 0.0243 - Q.z_7) / 0.0008289386469485379   # +1.6%  z_7 < 0.0243
        - 0.009505658567532476 * max(0.0, 0.16 - Q.LHA) * max(0.0, Q.centroid_offset - 0.00367) / 1.932546809790473e-05   # -1.0%  LHA < 0.16 and centroid_offset > 0.00367
        + 0.009194268797090592 * max(0.0, 5.65e-05 - Q.girth2) / 1.3060713628862948e-06   # +0.9%  girth2 < 5.65e-05
        - 0.0071406125095729725 * max(0.0, Q.log_sum_pt - 6.58) * max(0.0, Q.centroid_offset - 0.0151) / 0.0001091294014905758   # -0.7%  log_sum_pt > 6.58 and centroid_offset > 0.0151
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 32.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 32.65169477047986 * (0.01623192926816065
        - 0.2275437042447854 * max(0.0, 0.00868 - Q.girth2) / 0.00444891471733238   # -22.8%  girth2 < 0.00868
        - 0.21840557185450532 * max(0.0, 0.0135 - Q.width) / 0.008459444920955414   # -21.8%  width < 0.0135
        + 0.17209081380927663 * max(0.0, 0.132 - Q.mass_over_sum_pt) / 0.07316480111072891   # +17.2%  mass_over_sum_pt < 0.132
        + 0.14241921367611693 * max(0.0, 0.00819 - Q.lam1) / 0.004151989905718143   # +14.2%  lam1 < 0.00819
        + 0.07875811246103427 * max(0.0, 0.0508 - Q.e2) / 0.02546124602749324   # +7.9%  e2 < 0.0508
        + 0.023428223416800373 * max(0.0, Q.max_dr - 0.0866) / 0.054640800001426744   # +2.3%  max_dr > 0.0866
        + 0.021909762112753202 * max(0.0, 0.00371 - Q.girth2) / 0.0012506833304186038   # +2.2%  girth2 < 0.00371
        + 0.01776197650973791 * max(0.0, 6.62 - Q.log_sum_pt) * max(0.0, 46.1 - Q.pt_7) / 1.5424431795648812   # +1.8%  log_sum_pt < 6.62 and pt_7 < 46.1
        + 0.015241697165103518 * max(0.0, Q.LHA - 0.313) / 0.015265866368682459   # +1.5%  LHA > 0.313
        - 0.01147409902257788 * max(0.0, 6.72 - Q.log_sum_pt) / 0.20698827571904652   # -1.1%  log_sum_pt < 6.72
        - 0.010081672665517913 * max(0.0, Q.lam2 - 0.000765) / 0.00035206812689880313   # -1.0%  lam2 > 0.000765
        + 0.009972248977262833 * max(0.0, Q.centroid_offset - 0.00712) * max(0.0, 0.0973 - Q.C2) / 0.0006322540384093514   # +1.0%  centroid_offset > 0.00712 and C2 < 0.0973
        - 0.009438998916582238 * max(0.0, Q.mass_over_sum_pt - 0.00716) * max(0.0, 44.9 - Q.pt_7) / 0.5728611739091697   # -0.9%  mass_over_sum_pt > 0.00716 and pt_7 < 44.9
        + 0.007413425656397518 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159) / 0.00021048774933594437   # +0.7%  e2 < 0.0502 and z_dr_0p1_0p2 > 0.159
        + 0.006269691225332107 * max(0.0, Q.C2 - 0.00343) * max(0.0, Q.pt_7 - 31.0) / 0.1287522290532703   # +0.6%  C2 > 0.00343 and pt_7 > 31
        - 0.0052193623216118 * max(0.0, 0.0747 - Q.planar_flow) / 0.01792019194761426   # -0.5%  planar_flow < 0.0747
        + 0.004963232021431426 * max(0.0, 6.72 - Q.log_sum_pt) * max(0.0, 0.0495 - Q.z_7) / 0.0002504759459642205   # +0.5%  log_sum_pt < 6.72 and z_7 < 0.0495
        - 0.004773946058374799 * max(0.0, Q.C2 - 0.0419) / 0.006414709034929603   # -0.5%  C2 > 0.0419
        - 0.003839662770766976 * max(0.0, 0.0452 - Q.max_dr) / 0.004803505625006064   # -0.4%  max_dr < 0.0452
        + 0.002807470752678625 * max(0.0, 6.32 - Q.log_sum_pt) * max(0.0, 0.0711 - Q.z_7) / 8.814295970529982e-05   # +0.3%  log_sum_pt < 6.32 and z_7 < 0.0711
        + 0.0028064889663670825 * max(0.0, 6.33 - Q.log_sum_pt) / 0.033689934230344665   # +0.3%  log_sum_pt < 6.33
        + 0.001846411141049525 * max(0.0, Q.centroid_offset - 0.0525) / 0.0006843184222288595   # +0.2%  centroid_offset > 0.0525
        + 0.0015342142539354997 * max(0.0, Q.sum_pt - 997.0) / 4.007575642561712   # +0.2%  sum_pt > 997
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 29.74;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.735675435746156 * (0.1832815269919307
        - 0.14365461905415444 * max(0.0, Q.girth2 - 0.00741) / 0.0024980509514854338   # -14.4%  girth2 > 0.00741
        - 0.12877125776470266 * max(0.0, 0.0501 - Q.e2) / 0.02486428783340263   # -12.9%  e2 < 0.0501
        + 0.11088516528922761 * max(0.0, 0.0384 - Q.e2) / 0.015626754908433828   # +11.1%  e2 < 0.0384
        + 0.10516827009683502 * max(0.0, Q.girth2 - 0.00874) / 0.0022022884124918126   # +10.5%  girth2 > 0.00874
        - 0.08689320111688766 * max(0.0, Q.mass_over_sum_pt - 0.0905) / 0.008281500083284573   # -8.7%  mass_over_sum_pt > 0.0905
        - 0.06363386921484468 * max(0.0, 0.00845 - Q.lam1) / 0.004359898805745024   # -6.4%  lam1 < 0.00845
        + 0.06342328509809343 * max(0.0, Q.mass_over_sum_pt - 0.0849) / 0.009524920306796475   # +6.3%  mass_over_sum_pt > 0.0849
        - 0.03992374144608715 * max(0.0, 0.00524 - Q.width) / 0.002025869313688557   # -4.0%  width < 0.00524
        + 0.03509702860634344 * max(0.0, 0.0397 - Q.centroid_offset) * max(0.0, Q.sum_pt - 559.0) / 5.041709427040227   # +3.5%  centroid_offset < 0.0397 and sum_pt > 559
        - 0.0295331959204181 * max(0.0, 0.236 - Q.LHA) / 0.04065692261434514   # -3.0%  LHA < 0.236
        + 0.02731856298497852 * max(0.0, 27.6 - Q.mass) / 6.551096147518629   # +2.7%  mass < 27.6
        - 0.02304640999575052 * max(0.0, Q.girth2 - 0.00779) * max(0.0, Q.log_sum_pt - 6.13) / 0.000453841435491903   # -2.3%  girth2 > 0.00779 and log_sum_pt > 6.13
        + 0.02109038982365929 * max(0.0, Q.girth2 - 0.00457) * max(0.0, Q.eccentricity - 0.942) / 7.878605359418295e-05   # +2.1%  girth2 > 0.00457 and eccentricity > 0.942
        - 0.018363603961185533 * max(0.0, 47.2 - Q.pt_7) * max(0.0, 0.801 - Q.planar_flow) / 6.791718497766113   # -1.8%  pt_7 < 47.2 and planar_flow < 0.801
        - 0.017848385623755618 * max(0.0, 0.0204 - Q.centroid_offset) / 0.0080781400602745   # -1.8%  centroid_offset < 0.0204
        - 0.01723943159410052 * max(0.0, 0.00849 - Q.lam1) * max(0.0, 1.17 - Q.D2) / 0.0005075506362167535   # -1.7%  lam1 < 0.00849 and D2 < 1.17
        + 0.010888179453888055 * max(0.0, 0.0492 - Q.e2) * max(0.0, 1.14 - Q.D2) / 0.0024527831085376897   # +1.1%  e2 < 0.0492 and D2 < 1.14
        - 0.009068349178277488 * max(0.0, Q.eccentricity - 0.955) / 0.01654315876707207   # -0.9%  eccentricity > 0.955
        - 0.00830104061132594 * max(0.0, 0.000555 - Q.girth2) / 9.314605637635322e-05   # -0.8%  girth2 < 0.000555
        + 0.007188371028563234 * max(0.0, 0.174 - Q.planar_flow) * max(0.0, Q.sum_pt - 627.0) / 8.127417027265288   # +0.7%  planar_flow < 0.174 and sum_pt > 627
        + 0.0068992852930962485 * max(0.0, 0.197 - Q.max_dr) * max(0.0, 1.21 - Q.D2) / 0.01899582483464137   # +0.7%  max_dr < 0.197 and D2 < 1.21
        - 0.005817249766130343 * max(0.0, Q.centroid_offset - 0.0284) / 0.00296198374956031   # -0.6%  centroid_offset > 0.0284
        - 0.005397289250313983 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -0.5%  mass > 80.4
        + 0.0042129233548588805 * max(0.0, 0.0249 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0213) / 7.369065971515206e-05   # +0.4%  centroid_offset < 0.0249 and C2 > 0.0213
        - 0.0029315505351304016 * max(0.0, 0.0458 - Q.centroid_offset) * max(0.0, Q.C2 - 0.065) / 5.2513033274774446e-05   # -0.3%  centroid_offset < 0.0458 and C2 > 0.065
        + 0.002782034495409784 * max(0.0, 0.0219 - Q.e2) * max(0.0, 0.469 - Q.tau21) / 0.0002593281341898281   # +0.3%  e2 < 0.0219 and tau21 < 0.469
        - 0.0025217526594702162 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.934) / 0.033930325163455226   # -0.3%  mass > 80.4 and eccentricity > 0.934
        - 0.002101556782511323 * max(0.0, 0.00831 - Q.lam1) * max(0.0, 3.92 - Q.n_pt_above_50) / 0.0007574692169036082   # -0.2%  lam1 < 0.00831 and n_pt_above_50 < 3.92
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 9.353;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.352696207260609 * (-0.13472050968808835
        + 0.29312449706092525 * max(0.0, 0.0055 - Q.width) / 0.0021757971205689595   # +29.3%  width < 0.0055
        + 0.10254666343742368 * max(0.0, 0.188 - Q.max_dr) / 0.07861375329495267   # +10.3%  max_dr < 0.188
        - 0.09187617703971682 * max(0.0, 0.00581 - Q.width) * max(0.0, Q.centroid_offset - 0.00627) / 1.4764432517817246e-05   # -9.2%  width < 0.00581 and centroid_offset > 0.00627
        - 0.08728179661589396 * max(0.0, 0.0262 - Q.C2) / 0.00833830570145417   # -8.7%  C2 < 0.0262
        - 0.08247692216837263 * max(0.0, 0.234 - Q.LHA) / 0.03976193799745707   # -8.2%  LHA < 0.234
        - 0.07899149980726194 * max(0.0, Q.log_sum_pt - 6.67) * max(0.0, 0.0159 - Q.girth2) / 0.0006208264711371477   # -7.9%  log_sum_pt > 6.67 and girth2 < 0.0159
        + 0.07491124466533135 * max(0.0, 0.184 - Q.max_dr) * max(0.0, 0.000195 - Q.lam2) / 1.0663959115108311e-05   # +7.5%  max_dr < 0.184 and lam2 < 0.000195
        - 0.06398868984441111 * max(0.0, 24.1 - Q.mass) / 5.249708568556134   # -6.4%  mass < 24.1
        + 0.03855178983875651 * max(0.0, 0.0441 - Q.girth) * max(0.0, Q.log_sum_pt - 6.66) / 0.0010731046982382325   # +3.9%  girth < 0.0441 and log_sum_pt > 6.66
        + 0.03644380836619963 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 55.8 - Q.pt_7) / 1.981673652817952   # +3.6%  log_sum_pt > 6.6 and pt_7 < 55.8
        + 0.02273010090364613 * max(0.0, 0.00659 - Q.girth2) * max(0.0, 0.401 - Q.planar_flow) / 0.00038030005100569243   # +2.3%  girth2 < 0.00659 and planar_flow < 0.401
        + 0.014593960562481927 * max(0.0, 0.00353 - Q.centroid_offset) / 0.0002560842018792413   # +1.5%  centroid_offset < 0.00353
        + 0.009810976428424958 * max(0.0, 31.3 - Q.mass) * max(0.0, Q.centroid_offset - 0.00794) / 0.04986906632155073   # +1.0%  mass < 31.3 and centroid_offset > 0.00794
        + 0.0026718732611540792 * max(0.0, 21.8 - Q.mass) * max(0.0, Q.max_pair_mass - 13.0) / 0.0003740901035310897   # +0.3%  mass < 21.8 and max_pair_mass > 13
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 21.59;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.590349242193344 * (-0.10236054892901252
        + 0.28809688185671756 * max(0.0, 0.00617 - Q.width) / 0.0025917134561972695   # +28.8%  width < 0.00617
        - 0.09091103748302672 * max(0.0, 38.7 - Q.mass) * max(0.0, 0.0262 - Q.centroid_offset) / 0.17682892335393485   # -9.1%  mass < 38.7 and centroid_offset < 0.0262
        + 0.09082191121124067 * max(0.0, 0.0186 - Q.e2) * max(0.0, 0.0237 - Q.centroid_offset) / 6.738408185203302e-05   # +9.1%  e2 < 0.0186 and centroid_offset < 0.0237
        + 0.07599427568616503 * max(0.0, 50.9 - Q.mass) * max(0.0, 0.0277 - Q.centroid_offset) / 0.2893726547569357   # +7.6%  mass < 50.9 and centroid_offset < 0.0277
        - 0.06950631192179446 * max(0.0, 0.0383 - Q.girth) / 0.008200358190865406   # -7.0%  girth < 0.0383
        - 0.062455870398410634 * max(0.0, Q.log_sum_pt - 6.37) / 0.2154063984228196   # -6.2%  log_sum_pt > 6.37
        + 0.057894849913222315 * max(0.0, 0.194 - Q.max_dr) / 0.08333133526338912   # +5.8%  max_dr < 0.194
        - 0.056972328201393294 * max(0.0, 0.00619 - Q.width) * max(0.0, Q.centroid_offset - 0.00326) / 2.1770840053255615e-05   # -5.7%  width < 0.00619 and centroid_offset > 0.00326
        + 0.0535601928129259 * max(0.0, Q.sum_pt_top5 - 446.0) / 164.72696129776128   # +5.4%  sum_pt_top5 > 446
        + 0.03142942686144753 * max(0.0, 54.3 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt) / 4.523815349470833   # +3.1%  mass < 54.3 and log_sum_pt < 6.79
        + 0.023419445162997436 * max(0.0, Q.mass - 40.0) / 12.183951810301451   # +2.3%  mass > 40
        - 0.02211640983049002 * max(0.0, 57.9 - Q.mass) * max(0.0, 0.000614 - Q.lam1) / 0.005858908125444879   # -2.2%  mass < 57.9 and lam1 < 0.000614
        + 0.012830971350074378 * max(0.0, 60.5 - Q.mass) * max(0.0, 0.317 - Q.planar_flow) / 2.114695821104449   # +1.3%  mass < 60.5 and planar_flow < 0.317
        - 0.011950164595749619 * max(0.0, 0.0182 - Q.e2) * max(0.0, 58.1 - Q.pt_7) / 0.11835239776326996   # -1.2%  e2 < 0.0182 and pt_7 < 58.1
        - 0.010216688472264104 * max(0.0, 0.00397 - Q.girth2) * max(0.0, Q.centroid_offset - 0.012) / 4.047373802107758e-06   # -1.0%  girth2 < 0.00397 and centroid_offset > 0.012
        + 0.009447633077886881 * max(0.0, Q.lam2 - 0.00101) / 0.0003242888675098184   # +0.9%  lam2 > 0.00101
        - 0.008813588633317939 * max(0.0, 0.0327 - Q.e2) * max(0.0, 0.43 - Q.tau21) / 0.0006995899142292642   # -0.9%  e2 < 0.0327 and tau21 < 0.43
        - 0.00840548495320168 * max(0.0, Q.sum_pt - 863.0) / 19.20395298302915   # -0.8%  sum_pt > 863
        - 0.007792521978376919 * max(0.0, 0.0199 - Q.e2) * max(0.0, Q.eccentricity - 0.907) / 0.00010996292221609482   # -0.8%  e2 < 0.0199 and eccentricity > 0.907
        + 0.003945637955119217 * max(0.0, Q.girth2 - 0.0241) / 0.0003505666725690432   # +0.4%  girth2 > 0.0241
        - 0.0034183676441777214 * max(0.0, 0.00673 - Q.width) * max(0.0, Q.C2 - 0.0308) / 6.531305422655815e-06   # -0.3%  width < 0.00673 and C2 > 0.0308
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.113;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.112767374669437 * (0.5859910365805543
        - 0.3219392346529608 * max(0.0, 0.00362 - Q.lam2) / 0.003238142775023785   # -32.2%  lam2 < 0.00362
        + 0.1354819923687612 * max(0.0, 0.0634 - Q.girth) / 0.019977603234842406   # +13.5%  girth < 0.0634
        - 0.11563102560233622 * max(0.0, 0.00516 - Q.lam1) / 0.0020342058641091426   # -11.6%  lam1 < 0.00516
        + 0.07948828672574865 * max(0.0, Q.girth2 - 0.00809) / 0.002336639567557329   # +7.9%  girth2 > 0.00809
        - 0.06851967749553918 * max(0.0, Q.LHA - 0.24) / 0.045246658087262394   # -6.9%  LHA > 0.24
        - 0.0676051275477268 * max(0.0, 0.00157 - Q.girth2) / 0.00040004532511512386   # -6.8%  girth2 < 0.00157
        - 0.04619626935642299 * max(0.0, Q.max_dr - 0.113) / 0.039714703417231406   # -4.6%  max_dr > 0.113
        - 0.04509181522517494 * max(0.0, Q.LHA - 0.299) * max(0.0, 0.663 - Q.tau21) / 0.007871862502846681   # -4.5%  LHA > 0.299 and tau21 < 0.663
        + 0.043282097972493244 * max(0.0, Q.mass - 39.2) / 12.601268061053757   # +4.3%  mass > 39.2
        + 0.023767439005132267 * max(0.0, Q.C2 - 0.0492) / 0.0051081873289013004   # +2.4%  C2 > 0.0492
        - 0.02114192873333085 * max(0.0, Q.sum_pt - 792.0) / 38.53229567973674   # -2.1%  sum_pt > 792
        - 0.020446456384003088 * max(0.0, 6.28 - Q.log_sum_pt) / 0.02481009329743608   # -2.0%  log_sum_pt < 6.28
        + 0.011408648930369714 * max(0.0, Q.lam2 - 0.000223) * max(0.0, 0.562 - Q.tau21) / 6.378181825873036e-05   # +1.1%  lam2 > 0.000223 and tau21 < 0.562
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 23.35;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 23.347086856444403 * (0.06381952528645865
        - 0.26148052812960104 * max(0.0, 0.124 - Q.girth) / 0.06798227841326002   # -26.1%  girth < 0.124
        + 0.2067559226948178 * max(0.0, 0.00909 - Q.width) / 0.0047793549358814014   # +20.7%  width < 0.00909
        + 0.1920462618368213 * max(0.0, 0.0501 - Q.centroid_offset) / 0.03371218613202805   # +19.2%  centroid_offset < 0.0501
        - 0.05920865377174214 * max(0.0, 0.0554 - Q.centroid_offset) * max(0.0, 6.82 - Q.log_sum_pt) / 0.009468147823712384   # -5.9%  centroid_offset < 0.0554 and log_sum_pt < 6.82
        + 0.04780583112461831 * max(0.0, 0.289 - Q.planar_flow) / 0.13146370924746487   # +4.8%  planar_flow < 0.289
        + 0.042841106606230465 * max(0.0, 674.0 - Q.sum_pt_top5) / 121.82887173713236   # +4.3%  sum_pt_top5 < 674
        - 0.03666091920766531 * max(0.0, Q.girth - 0.079) * max(0.0, 41.8 - Q.pt_7) / 0.07574563406888962   # -3.7%  girth > 0.079 and pt_7 < 41.8
        - 0.028513335467436736 * max(0.0, Q.girth - 0.0709) / 0.01230505212061335   # -2.9%  girth > 0.0709
        - 0.019071383154095706 * max(0.0, Q.centroid_offset - 0.0143) / 0.007147050384770515   # -1.9%  centroid_offset > 0.0143
        - 0.018486682134038722 * max(0.0, 0.195 - Q.planar_flow) * max(0.0, Q.width - 0.00586) / 0.00019354716299142648   # -1.8%  planar_flow < 0.195 and width > 0.00586
        - 0.017319368202604414 * max(0.0, 0.276 - Q.planar_flow) * max(0.0, 80.4 - Q.mass) / 3.6759708520449697   # -1.7%  planar_flow < 0.276 and mass < 80.4
        - 0.01634301597343641 * max(0.0, 0.0034 - Q.width) / 0.0011124251120352131   # -1.6%  width < 0.0034
        - 0.011422388071516174 * max(0.0, Q.mass_over_sum_pt - 0.0933) / 0.007752310651561123   # -1.1%  mass_over_sum_pt > 0.0933
        + 0.0089223115221581 * max(0.0, Q.mass - 91.2) / 0.5835013503307559   # +0.9%  mass > 91.2
        - 0.008115869307705359 * max(0.0, 0.0204 - Q.girth) / 0.002557110737416307   # -0.8%  girth < 0.0204
        - 0.006864669425576515 * max(0.0, 0.361 - Q.planar_flow) * max(0.0, Q.max_dr - 0.121) / 0.006938096680507064   # -0.7%  planar_flow < 0.361 and max_dr > 0.121
        + 0.006451071683547432 * max(0.0, 0.188 - Q.planar_flow) * max(0.0, Q.girth2 - 0.0161) / 4.9381551118993744e-05   # +0.6%  planar_flow < 0.188 and girth2 > 0.0161
        - 0.006308532672061326 * max(0.0, 0.0424 - Q.z_7) / 0.004705618537742271   # -0.6%  z_7 < 0.0424
        + 0.00328868800140958 * max(0.0, Q.centroid_offset - 0.0186) * max(0.0, 0.103 - Q.tau21) / 3.709240792881933e-05   # +0.3%  centroid_offset > 0.0186 and tau21 < 0.103
        + 0.002093461012917315 * max(0.0, 0.139 - Q.LHA) * max(0.0, 0.0257 - Q.z_7) / 3.6749034661023024e-05   # +0.2%  LHA < 0.139 and z_7 < 0.0257
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5852;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5852013262835938 * (-2.289810258137768
        - 0.37233319987217806 * max(0.0, Q.e2 - 0.0629) / 0.0018157490198717752   # -37.2%  e2 > 0.0629
        + 0.2908351773493471 * max(0.0, Q.girth2 - 0.0193) / 0.0007151139979611856   # +29.1%  girth2 > 0.0193
        + 0.1700925663821598 * max(0.0, Q.girth2 - 0.0188) * max(0.0, Q.pt_7 - 15.4) / 0.01465955750188809   # +17.0%  girth2 > 0.0188 and pt_7 > 15.4
        + 0.09911124875812363 * max(0.0, Q.mass - 91.2) / 0.5835013503307559   # +9.9%  mass > 91.2
        + 0.06762780763819143 * max(0.0, Q.girth2 - 0.0145) * max(0.0, Q.lam2 - 5.56e-06) / 4.779695981101616e-06   # +6.8%  girth2 > 0.0145 and lam2 > 5.56e-06
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 16.95;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 16.952356345735343 * (0.060168626661543635
        + 0.3403829354942794 * max(0.0, 0.143 - Q.girth) / 0.08548581950379956   # +34.0%  girth < 0.143
        + 0.10634713042005876 * max(0.0, 0.0157 - Q.lam1) / 0.010542891527644604   # +10.6%  lam1 < 0.0157
        + 0.09260191200876769 * max(0.0, 0.00647 - Q.lam1) / 0.002864636150855947   # +9.3%  lam1 < 0.00647
        - 0.09130760963015931 * max(0.0, 0.00741 - Q.width) / 0.0034628168580039297   # -9.1%  width < 0.00741
        - 0.056127154202058424 * max(0.0, 0.0489 - Q.e2) / 0.02384680498008349   # -5.6%  e2 < 0.0489
        - 0.05501133961235031 * max(0.0, 0.133 - Q.girth) * max(0.0, 6.88 - Q.log_sum_pt) / 0.021243094126761477   # -5.5%  girth < 0.133 and log_sum_pt < 6.88
        - 0.043141531398634865 * max(0.0, 0.0168 - Q.lam1) * max(0.0, 0.0384 - Q.centroid_offset) / 0.00029371510585156157   # -4.3%  lam1 < 0.0168 and centroid_offset < 0.0384
        + 0.041584029566244074 * max(0.0, 0.0387 - Q.centroid_offset) / 0.023113025819625044   # +4.2%  centroid_offset < 0.0387
        - 0.03801290097609219 * max(0.0, 0.498 - Q.tau21) * max(0.0, Q.max_dr - -0.0292) / 0.036823328176106604   # -3.8%  tau21 < 0.498 and max_dr > -0.0292
        + 0.03329706318109426 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 44.9 - Q.pt_7) / 1018.8875095891225   # +3.3%  sum_pt_top5 > 653 and pt_7 < 44.9
        - 0.03158132463202297 * max(0.0, 0.147 - Q.girth) * max(0.0, 38.2 - Q.pt_7) / 0.6427105270497029   # -3.2%  girth < 0.147 and pt_7 < 38.2
        - 0.02532281536179464 * max(0.0, 0.000307 - Q.lam2) * max(0.0, 0.0423 - Q.centroid_offset) / 6.003935520145508e-06   # -2.5%  lam2 < 0.000307 and centroid_offset < 0.0423
        - 0.018521067154056367 * max(0.0, 0.0286 - Q.z_7) / 0.0013892731429153004   # -1.9%  z_7 < 0.0286
        - 0.010244165946973598 * max(0.0, Q.sum_pt_top5 - 667.0) * max(0.0, Q.z_7 - 0.023) / 0.2635246609983972   # -1.0%  sum_pt_top5 > 667 and z_7 > 0.023
        - 0.009176729074494258 * max(0.0, Q.C2 - 0.0657) / 0.0030148678557964485   # -0.9%  C2 > 0.0657
        + 0.004686085521564433 * max(0.0, Q.sum_pt_top5 - 878.0) * max(0.0, Q.n_pt_above_50 - 6.08) / 1.0837679621848748   # +0.5%  sum_pt_top5 > 878 and n_pt_above_50 > 6.08
        - 0.0026542058193544066 * max(0.0, Q.sum_pt - 977.0) * max(0.0, 4.88 - Q.D2) / 12.710464086050944   # -0.3%  sum_pt > 977 and D2 < 4.88
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 27.9;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 27.89933403585069 * (-0.08745728471025853
        + 0.13518768490181526 * max(0.0, 0.0137 - Q.girth2) / 0.008630769745100852   # +13.5%  girth2 < 0.0137
        + 0.12631693183028092 * max(0.0, 0.0437 - Q.e2) / 0.019578657086204376   # +12.6%  e2 < 0.0437
        - 0.12518076305290035 * max(0.0, 0.0879 - Q.girth) / 0.036918181007141046   # -12.5%  girth < 0.0879
        + 0.10890002962777547 * max(0.0, Q.lam1 - 0.00419) / 0.003242516865634306   # +10.9%  lam1 > 0.00419
        - 0.10794337668100853 * max(0.0, 0.00735 - Q.width) / 0.0034183295380035335   # -10.8%  width < 0.00735
        - 0.08050264287766459 * max(0.0, Q.lam1 - 0.00616) / 0.002409839189284078   # -8.1%  lam1 > 0.00616
        - 0.04727833934468143 * max(0.0, Q.lam1 - 0.00417) * max(0.0, Q.D2 - 0.395) / 0.0015911148154856085   # -4.7%  lam1 > 0.00417 and D2 > 0.395
        - 0.03585605876677179 * max(0.0, 0.0356 - Q.C2) / 0.014908497179482245   # -3.6%  C2 < 0.0356
        + 0.03429676189271797 * max(0.0, Q.lam1 - 0.00255) * max(0.0, Q.D2 - 0.419) / 0.0020186852666518426   # +3.4%  lam1 > 0.00255 and D2 > 0.419
        - 0.025977297244016075 * max(0.0, 0.0332 - Q.girth) / 0.006357449939994631   # -2.6%  girth < 0.0332
        + 0.022364494588967724 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.968) / 7.010724776186357e-05   # +2.2%  girth2 < 0.0135 and eccentricity > 0.968
        + 0.016827046310932774 * max(0.0, Q.C2 - 0.0359) / 0.007759725386205662   # +1.7%  C2 > 0.0359
        + 0.014798559690514307 * max(0.0, Q.lam1 - 0.00709) * max(0.0, Q.D2 - 0.378) / 0.00111285703518904   # +1.5%  lam1 > 0.00709 and D2 > 0.378
        - 0.013914962418236716 * max(0.0, 0.00787 - Q.width) * max(0.0, 0.117 - Q.planar_flow) / 7.748865960133629e-05   # -1.4%  width < 0.00787 and planar_flow < 0.117
        - 0.013828913225929013 * max(0.0, 0.00766 - Q.width) * max(0.0, 1.0 - Q.D2) / 0.00022562425113624974   # -1.4%  width < 0.00766 and D2 < 1
        - 0.013074043426565821 * max(0.0, 0.00457 - Q.girth2) / 0.0016655575559679351   # -1.3%  girth2 < 0.00457
        + 0.012646368853074972 * max(0.0, 0.104 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00913) / 0.0002594303448327328   # +1.3%  planar_flow < 0.104 and centroid_offset > 0.00913
        - 0.0088739719892155 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0184) / 0.00014736780282851528   # -0.9%  planar_flow < 0.11 and centroid_offset > 0.0184
        - 0.007305803823116133 * max(0.0, Q.centroid_offset - 0.0496) / 0.0008218833115383548   # -0.7%  centroid_offset > 0.0496
        + 0.006820275508805456 * max(0.0, 1.09 - Q.D2) * max(0.0, 0.0317 - Q.centroid_offset) / 0.003964190513264478   # +0.7%  D2 < 1.09 and centroid_offset < 0.0317
        - 0.006036607918625785 * max(0.0, Q.C2 - 0.0669) / 0.00289376874167014   # -0.6%  C2 > 0.0669
        + 0.005585389980609813 * max(0.0, 0.136 - Q.tau21) / 0.01690115626784455   # +0.6%  tau21 < 0.136
        - 0.005537560316851748 * max(0.0, Q.centroid_offset - 0.0296) / 0.0027539081109361515   # -0.6%  centroid_offset > 0.0296
        - 0.005055089879377419 * max(0.0, 0.1 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr) / 0.0010765926803511358   # -0.5%  planar_flow < 0.1 and max_dr < 0.17
        + 0.00470129826878032 * max(0.0, 0.0411 - Q.e2) * max(0.0, 0.984 - Q.D2) / 0.0006903320568572036   # +0.5%  e2 < 0.0411 and D2 < 0.984
        - 0.004575277096527007 * max(0.0, 457.0 - Q.sum_pt_top5) / 19.88273894432773   # -0.5%  sum_pt_top5 < 457
        - 0.003378676917712053 * max(0.0, 75.6 - Q.mass) * max(0.0, 0.734 - Q.D2) / 1.3055794449649167   # -0.3%  mass < 75.6 and D2 < 0.734
        - 0.0028953324962263686 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, 741.0 - Q.sum_pt) / 1.9232821061207823   # -0.3%  planar_flow < 0.11 and sum_pt < 741
        - 0.002273940973188018 * max(0.0, Q.lam1 - 0.00268) * max(0.0, Q.D2 - 1.65) / 0.00019887598366388686   # -0.2%  lam1 > 0.00268 and D2 > 1.65
        + 0.0020665000971105832 * max(0.0, Q.lam1 - 0.00655) * max(0.0, 0.165 - Q.max_dr) / 7.576081011091466e-06   # +0.2%  lam1 > 0.00655 and max_dr < 0.165
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 28.58;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 28.582029811680155 * (-0.20502392722315646
        + 0.29771123157867324 * max(0.0, 0.0627 - Q.e2) / 0.03590376074368629   # +29.8%  e2 < 0.0627
        + 0.1632356149160335 * max(0.0, 0.0136 - Q.width) / 0.008545064490582439   # +16.3%  width < 0.0136
        - 0.14619178042229553 * max(0.0, 0.00855 - Q.lam1) / 0.0044404440236478744   # -14.6%  lam1 < 0.00855
        - 0.11535409857972719 * max(0.0, 0.00323 - Q.lam2) / 0.0028670037256567425   # -11.5%  lam2 < 0.00323
        + 0.0597362260284422 * max(0.0, Q.e2 - 0.0244) / 0.011614847572668022   # +6.0%  e2 > 0.0244
        + 0.05608226659735791 * max(0.0, Q.girth - 0.0329) / 0.03205890031584556   # +5.6%  girth > 0.0329
        - 0.04493303023874003 * max(0.0, 80.4 - Q.mass) / 41.295087132244156   # -4.5%  mass < 80.4
        - 0.025702168048167932 * max(0.0, 0.305 - Q.LHA) / 0.07967680405396409   # -2.6%  LHA < 0.305
        - 0.024279208411346263 * max(0.0, Q.LHA - 0.343) / 0.009454346847644337   # -2.4%  LHA > 0.343
        - 0.019204062138300133 * max(0.0, 0.000335 - Q.lam2) / 0.00022222310791184316   # -1.9%  lam2 < 0.000335
        + 0.013369227950585774 * max(0.0, 0.25 - Q.tau21) / 0.061731772510951556   # +1.3%  tau21 < 0.25
        - 0.009964609390822873 * max(0.0, Q.e2 - 0.0497) / 0.0034564170226971773   # -1.0%  e2 > 0.0497
        - 0.00856222749144944 * max(0.0, 0.00762 - Q.width) * max(0.0, Q.e2 - 0.0244) / 4.099260325209302e-06   # -0.9%  width < 0.00762 and e2 > 0.0244
        - 0.0066483530018288895 * max(0.0, 0.00626 - Q.width) * max(0.0, Q.log_sum_pt - 6.9) / 1.9549734948235244e-05   # -0.7%  width < 0.00626 and log_sum_pt > 6.9
        + 0.005965808164953815 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # +0.6%  log_sum_pt > 6.9
        - 0.0030600870412752675 * max(0.0, 0.00818 - Q.width) * max(0.0, 0.0618 - Q.planar_flow) / 2.8033172769251865e-05   # -0.3%  width < 0.00818 and planar_flow < 0.0618
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0935579831932773, 0.8909138655462185, 1.9498713235294118, 1.2707930672268908, 6.208876470588235, 1.9468523109243698, 1.5047441176470588, 1.3274747899159665, 1.15376743697479, 4.341210504201681, 1.6509269957983193, 3.1840957983193277, 0.0776, 5.062475210084034, 0.5067075630252101, 0.509103781512605]
T = [2.826506139213498, 2.273989691110819, 4.503901217830882, 3.2102859703256303, 4.290193618697479]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +30%, n9 +26%, n5 -13%, n1 +12%, n4 -7%, n0 -6% ...
            + 0.29642084363637 * h[2] / H_AVG[2]
            + 0.2639815796109631 * h[9] / H_AVG[9]
            - 0.12914700705369556 * h[5] / H_AVG[5]
            + 0.12312487982984872 * h[1] / H_AVG[1]
            - 0.06864566363895191 * h[4] / H_AVG[4]
            - 0.060452171853939564 * h[0] / H_AVG[0]
            + 0.058227854376231215 * h[6] / H_AVG[6]
        ),
        0.03125 + T[1] * (   # class q: n9 +48%, n4 -26%, n10 -9%, n6 +8%, n5 +4%, n8 +3% ...
            + 0.48472536348395484 * h[9] / H_AVG[9]
            - 0.25597397006373684 * h[4] / H_AVG[4]
            - 0.09075057608285922 * h[10] / H_AVG[10]
            + 0.08271498126889087 * h[6] / H_AVG[6]
            + 0.040131537285026546 * h[5] / H_AVG[5]
            + 0.031710990200531294 * h[8] / H_AVG[8]
            + 0.013992581615000457 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +27%, n3 -14%, n6 -10%, n14 -8%, n0 +8%, n13 +8% ...
            + 0.26511148149576996 * h[11] / H_AVG[11]
            - 0.1410769248441594 * h[3] / H_AVG[3]
            - 0.10440560616717388 * h[6] / H_AVG[6]
            - 0.08437810997372044 * h[14] / H_AVG[14]
            + 0.08346332180520843 * h[0] / H_AVG[0]
            + 0.07903265879806856 * h[13] / H_AVG[13]
            - 0.07771237264357304 * h[15] / H_AVG[15]
            + 0.06447412948234457 * h[7] / H_AVG[7]
            - 0.06404266996393264 * h[8] / H_AVG[8]
            - 0.030121181992006234 * h[9] / H_AVG[9]
            - 0.006181542834042844 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n3 -22%, n7 +19%, n6 -18%, n4 +15%, n13 +9%, n14 +6% ...
            - 0.22266586432566918 * h[3] / H_AVG[3]
            + 0.1938312703369513 * h[7] / H_AVG[7]
            - 0.17577220513486227 * h[6] / H_AVG[6]
            + 0.15109821328954806 * h[4] / H_AVG[4]
            + 0.08623970437854431 * h[13] / H_AVG[13]
            + 0.05918953572699316 * h[14] / H_AVG[14]
            - 0.04225879859623277 * h[9] / H_AVG[9]
            + 0.034689817113701324 * h[1] / H_AVG[1]
            - 0.024778934523791277 * h[15] / H_AVG[15]
            + 0.009475656573706335 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -48%, n4 +18%, n10 +14%, n5 -11%, n8 +5%, n3 +2% ...
            - 0.47937942593859906 * h[13] / H_AVG[13]
            + 0.18090315444997548 * h[4] / H_AVG[4]
            + 0.14430528746446886 * h[10] / H_AVG[10]
            - 0.11344781168148317 * h[5] / H_AVG[5]
            + 0.050424622676692205 * h[8] / H_AVG[8]
            + 0.018513049470665687 * h[3] / H_AVG[3]
            - 0.009043880870761222 * h[12] / H_AVG[12]
            + 0.003982767447354181 * h[0] / H_AVG[0]
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
