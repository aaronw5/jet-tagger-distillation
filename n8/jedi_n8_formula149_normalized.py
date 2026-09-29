"""JEDI-linear jet tagger, 8 particles, 3 features: the simpler version of the simplified formula, as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron 13:  17.3%   (on for 94% of jets)
  neuron  9:  12.8%   (on for 69% of jets)
  neuron 10:   9.6%   (on for 100% of jets)
  neuron  6:   9.2%   (on for 50% of jets)
  neuron  4:   7.5%   (on for 91% of jets)
  neuron  5:   7.1%   (on for 91% of jets)
  neuron  7:   6.9%   (on for 60% of jets)
  neuron  2:   6.3%   (on for 95% of jets)
  neuron 11:   5.9%   (on for 82% of jets)
  neuron  0:   3.4%   (on for 51% of jets)
  neuron 14:   3.3%   (on for 31% of jets)
  neuron 15:   3.2%   (on for 40% of jets)
  neuron  3:   2.9%   (on for 25% of jets)
  neuron  8:   2.2%   (on for 82% of jets)
  neuron  1:   2.1%   (on for 58% of jets)
  neuron 12:   0.2%   (on for 5% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 86.5% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau32                  N-subjettiness τ3/τ2
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
        max_dr=max(dr[i] for i in real),
        pt_0=pt[0],
        pt_7=pt[7],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def neuron_0(Q):
    # scale S = 11.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.44451469253545 * (-0.012145556516316165
        - 0.23242276201643988 * max(0.0, 29.0 - Q.mass) / 7.093241906071511   # -23.2%  mass < 29
        + 0.18389344909021063 * max(0.0, 0.013 - Q.girth2) / 0.008032714809060822   # +18.4%  girth2 < 0.013
        - 0.16071702885623812 * max(0.0, 74.0 - Q.mass) / 35.50827023331778   # -16.1%  mass < 74
        - 0.12287518970676084 * max(0.0, 0.005 - Q.width) / 0.0018926607186367497   # -12.3%  width < 0.005
        + 0.09405228036060484 * max(0.0, 0.033 - Q.centroid_offset) / 0.018060112490828947   # +9.4%  centroid_offset < 0.033
        + 0.04837925840980048 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008) / 0.224160782868393   # +4.8%  mass < 65 and centroid_offset > 0.008
        - 0.04616685771856322 * max(0.0, Q.sum_pt - 880.0) / 15.914375947840073   # -4.6%  sum_pt > 880
        + 0.04387834483177192 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97) / 0.00010103950947914501   # +4.4%  girth2 < 0.018 and eccentricity > 0.97
        - 0.026820554696838392 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7) / 217.69378176589902   # -2.7%  mass < 66 and pt_7 < 39
        + 0.026457626912791417 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7) / 196.61993502146927   # +2.6%  sum_pt > 860 and pt_7 < 33
        - 0.01433664739998027 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2) / 8.286665242968401e-05   # -1.4%  lam1 < 0.0062 and D2 < 0.95
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 4.635;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.634604753993108 * (0.11867247137442044
        - 0.30860162824039405 * max(0.0, 0.01 - Q.width) / 0.005522187541826039   # -30.9%  width < 0.01
        + 0.2640213970673998 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2) / 0.000181011068343857   # +26.4%  log_sum_pt > 6.5 and lam2 < 0.0015
        + 0.15556370284046564 * max(0.0, Q.pt_7 - 32.0) / 5.861595745798319   # +15.6%  pt_7 > 32
        - 0.12360207173859704 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset) / 0.00240691911463487   # -12.4%  log_sum_pt > 6.5 and centroid_offset < 0.028
        - 0.10284725863775884 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass) / 283.7240439391166   # -10.3%  pt_7 > 34 and mass < 98
        + 0.045363941475384695 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018) / 1.928843475428285e-05   # +4.5%  lam1 < 0.012 and centroid_offset > 0.018
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 5.927;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.927363968902115 * (0.22438304902108905
        + 0.3225365904196348 * max(0.0, Q.pt_7 - 28.0) / 8.534784663865546   # +32.3%  pt_7 > 28
        - 0.15520582105869032 * max(0.0, Q.LHA - 0.13) / 0.11994281505960237   # -15.5%  LHA > 0.13
        + 0.12566235743168927 * max(0.0, 780.0 - Q.sum_pt) / 106.71153720456932   # +12.6%  sum_pt < 780
        - 0.10559766773354154 * max(0.0, Q.pt_7 - 28.0) * max(0.0, 0.054 - Q.C2) / 0.28067973584031014   # -10.6%  pt_7 > 28 and C2 < 0.054
        + 0.10394413176368833 * max(0.0, 0.005 - Q.lam1) / 0.0019435794996684541   # +10.4%  lam1 < 0.005
        - 0.0837424255825955 * max(0.0, Q.z_7 - 0.047) / 0.01159747280529774   # -8.4%  z_7 > 0.047
        - 0.043955979657113606 * max(0.0, Q.pt_7 - 28.0) * max(0.0, Q.max_dr - 0.07) / 0.44613542814618073   # -4.4%  pt_7 > 28 and max_dr > 0.07
        + 0.03638744300411421 * max(0.0, 6.4 - Q.log_sum_pt) / 0.05015851594955021   # +3.6%  log_sum_pt < 6.4
        - 0.022967583348932362 * max(0.0, 0.018 - Q.z_7) / 0.0003173361911310439   # -2.3%  z_7 < 0.018
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 11.15;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.15365298431444 * (0.04554561637468984
        - 0.5149053006737486 * max(0.0, 0.01 - Q.girth2) / 0.005522187541826039   # -51.5%  girth2 < 0.01
        + 0.3751120012070375 * max(0.0, 0.047 - Q.e2) / 0.02225462282827147   # +37.5%  e2 < 0.047
        + 0.05413539730557757 * max(0.0, Q.LHA - 0.27) / 0.03049532503608097   # +5.4%  LHA > 0.27
        - 0.02787050096542728 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32) / 0.002851907305205289   # -2.8%  mass_over_sum_pt > 0.071 and tau32 < 0.55
        + 0.015646816612834732 * max(0.0, Q.centroid_offset - 0.013) / 0.007756407235945105   # +1.6%  centroid_offset > 0.013
        - 0.01232998323537419 * max(0.0, Q.lam1 - 0.016) / 0.0007276420862951236   # -1.2%  lam1 > 0.016
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 12.52;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.521490072100413 * (0.06556727636028717
        + 0.23534887857542755 * max(0.0, 55.0 - Q.mass) / 20.464712823348485   # +23.5%  mass < 55
        - 0.21020863710592153 * max(0.0, 0.00042 - Q.lam2) / 0.0002921337805318041   # -21.0%  lam2 < 0.00042
        + 0.1792219122134112 * max(0.0, Q.width - 0.0016) / 0.005255562984737892   # +17.9%  width > 0.0016
        + 0.1674123345013955 * max(0.0, 0.27 - Q.tau21) / 0.07130108450361827   # +16.7%  tau21 < 0.27
        - 0.13603556329673203 * max(0.0, Q.mass_over_sum_pt - 0.086) / 0.009257434539525095   # -13.6%  mass_over_sum_pt > 0.086
        - 0.027670966163901786 * max(0.0, Q.C2 - 0.063) / 0.0032998259819687873   # -2.8%  C2 > 0.063
        - 0.023264256800764337 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass) / 0.3435178780254306   # -2.3%  tau21 < 0.26 and mass < 57
        + 0.020837451342446037 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0) / 0.06287131087043697   # +2.1%  C2 > -0.00087 and pt_7 > 38
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 6.208;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.207715402280074 * (0.049937856346658226
        + 0.25407715001803965 * max(0.0, 0.066 - Q.z_7) / 0.017218762418497922   # +25.4%  z_7 < 0.066
        - 0.21343742227708676 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset) / 0.00043872807075908196   # -21.3%  z_7 < 0.072 and centroid_offset < 0.032
        + 0.16096648887237047 * max(0.0, 0.04 - Q.e2) / 0.016765673695032856   # +16.1%  e2 < 0.04
        + 0.093784792655386 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset) / 1.0213847401166365e-05   # +9.4%  width < 0.0026 and centroid_offset < 0.021
        - 0.09053189011375885 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt) / 0.004257547035278091   # -9.1%  LHA < 0.22 and log_sum_pt < 6.8
        + 0.0782346130978973 * max(0.0, 0.035 - Q.z_7) / 0.002611065659780857   # +7.8%  z_7 < 0.035
        + 0.06475806617048666 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2) / 4.599538269890186e-07   # +6.5%  e2 < 0.025 and lam2 < 9.2e-05
        - 0.04420957679497434 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # -4.4%  log_sum_pt > 6.9
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 17.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.092218392182833 * (0.5528831766110582
        - 0.4166546529765687 * max(0.0, 0.012 - Q.width) / 0.0071862283781984485   # -41.7%  width < 0.012
        - 0.2178599945710652 * max(0.0, Q.mass_over_sum_pt - 0.0089) / 0.05281859015784983   # -21.8%  mass_over_sum_pt > 0.0089
        + 0.09335798694399318 * Q.max_dr / 0.1236972946900221   # +9.3%  max_dr
        - 0.06536641391985402 * max(0.0, 0.00066 - Q.lam2) / 0.0004965586765475399   # -6.5%  lam2 < 0.00066
        + 0.05818546125546637 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1) / 11.27572121347306   # +5.8%  mass < 42 and z_dr_0p05_0p1 < 0.94
        + 0.05316065143635672 * max(0.0, Q.girth - 0.09) / 0.0073276892275880405   # +5.3%  girth > 0.09
        - 0.0243786861006695 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32) / 0.0061914684538947285   # -2.4%  mass_over_sum_pt > 0.015 and tau32 < 0.55
        + 0.022389440863878427 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2) / 2.25108948778984e-05   # +2.2%  centroid_offset > 0.0059 and lam2 < 0.0026
        - 0.014578641634168698 * max(0.0, Q.lam2 - 0.0027) / 0.0001946729114629536   # -1.5%  lam2 > 0.0027
        + 0.012653638379829489 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7) / 0.8100327739466373   # +1.3%  log_sum_pt < 6.5 and pt_7 < 44
        + 0.00964952019477387 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7) / 0.00019681587893657948   # +1.0%  log_sum_pt < 6.7 and z_7 < 0.049
        + 0.006464233800415996 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18) / 0.00014850550518123685   # +0.6%  e2 < 0.049 and z_dr_0p1_0p2 > 0.18
        + 0.005300677922959805 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7) / 7.248027574868087e-05   # +0.5%  log_sum_pt < 6.3 and z_7 < 0.071
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 15.36;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 15.363553354885669 * (0.2577528719123239
        - 0.14839117207351316 * max(0.0, 0.0056 - Q.width) / 0.0022351134211229797   # -14.8%  width < 0.0056
        + 0.13188001686875225 * max(0.0, 0.038 - Q.e2) / 0.015349588451562859   # +13.2%  e2 < 0.038
        + 0.1300240689330709 * max(0.0, Q.mass_over_sum_pt - 0.073) / 0.013317544803150444   # +13.0%  mass_over_sum_pt > 0.073
        - 0.11019595053730831 * max(0.0, Q.mass_over_sum_pt - 0.09) / 0.008381194879070684   # -11.0%  mass_over_sum_pt > 0.09
        - 0.10389989367620617 * max(0.0, 0.087 - Q.girth) / 0.036196634014996405   # -10.4%  girth < 0.087
        - 0.1014458680144802 * max(0.0, Q.girth2 - 0.0046) / 0.0035261742214324795   # -10.1%  girth2 > 0.0046
        + 0.06171136578188618 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94) / 8.034795430448085e-05   # +6.2%  girth2 > 0.0047 and eccentricity > 0.94
        - 0.051846571141567606 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078) / 0.000166991103144937   # -5.2%  planar_flow < 0.22 and width > 0.0078
        - 0.03481647050332944 * max(0.0, 0.19 - Q.planar_flow) / 0.07297472062847271   # -3.5%  planar_flow < 0.19
        + 0.02862736967577302 * max(0.0, 0.0011 - Q.e2_sq) / 0.0003075651198767656   # +2.9%  e2_sq < 0.0011
        - 0.02603657145011346 * max(0.0, Q.mass - 80.4) / 1.215848798334684   # -2.6%  mass > 80.4
        + 0.02476029102448052 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0) / 10.47950557127006   # +2.5%  planar_flow < 0.18 and sum_pt > 590
        - 0.02270330949284083 * max(0.0, 0.00069 - Q.width) / 0.0001277668522804917   # -2.3%  width < 0.00069
        - 0.022311496157858303 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow) / 3.868892343663461   # -2.2%  pt_7 < 45 and planar_flow < 0.6
        - 0.0013495846688199409 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0) / 0.002696282973517668   # -0.1%  centroid_offset > 0.031 and pt_0 > 370
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 8.818;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.818387252099694 * (0.015762519384358353
        + 0.4816142803304423 * max(0.0, 0.0049 - Q.width) / 0.0018385546450628313   # +48.2%  width < 0.0049
        - 0.23965521705077542 * max(0.0, 0.063 - Q.girth) / 0.01975114496205367   # -24.0%  girth < 0.063
        - 0.14683737290334212 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014) / 1.808476001875197e-05   # -14.7%  width < 0.0051 and centroid_offset > 0.0014
        - 0.08143951005766652 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2) / 6.138163566750181e-06   # -8.1%  LHA < 0.17 and lam2 < 0.00039
        + 0.03524206216516599 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow) / 0.00036778479495266827   # +3.5%  girth2 < 0.0051 and planar_flow < 0.54
        - 0.015211557492607528 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015) / 0.013549636836100182   # -1.5%  mass < 20 and centroid_offset > 0.015
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 12.98;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.978319381635472 * (-0.09092086311804891
        + 0.509245561114166 * max(0.0, 0.0065 - Q.width) / 0.0028124049088594947   # +50.9%  width < 0.0065
        - 0.13186373631098589 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016) / 2.4588644895111168e-05   # -13.2%  width < 0.0061 and centroid_offset > 0.0016
        - 0.10483667100492665 * max(0.0, Q.log_sum_pt - 6.3) / 0.27103661338832297   # -10.5%  log_sum_pt > 6.3
        + 0.07083552004426615 * max(0.0, 0.019 - Q.centroid_offset) / 0.0071265581604552505   # +7.1%  centroid_offset < 0.019
        + 0.06613905737622036 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt) / 4.990545408307566   # +6.6%  mass < 56 and log_sum_pt < 6.8
        - 0.043401778973377866 * max(0.0, 25.0 - Q.mass) / 5.577050982649988   # -4.3%  mass < 25
        + 0.03130108952548527 * max(0.0, Q.lam2 - 0.0015) / 0.0002782435183937757   # +3.1%  lam2 > 0.0015
        + 0.01961795746536556 * max(0.0, 0.00029 - Q.width) / 3.4640560217803286e-05   # +2.0%  width < 0.00029
        + 0.00982838860993422 * max(0.0, Q.n_dr_0p2_0p4 - 1.5) / 0.10629663865546218   # +1.0%  n_dr_0p2_0p4 > 1.5
        - 0.009477416381784634 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026) / 1.675762080593257e-05   # -0.9%  width < 0.0091 and C2 > 0.026
        + 0.003452823193487317 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7) / 0.0007739523691432977   # +0.3%  girth2 > 0.019 and pt_7 < 30
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 3.612;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.6118250500681417 * (0.7392384633772965
        - 0.1701722424256835 * max(0.0, 0.037 - Q.e2) / 0.0146690302629917   # -17.0%  e2 < 0.037
        + 0.15417964250245866 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4) / 0.0016524329228514273   # +15.4%  eccentricity > 0.9 and z_dr_0p2_0p4 < 0.041
        + 0.13817218759787414 * max(0.0, Q.e2 - 0.037) / 0.006301183944301975   # +13.8%  e2 > 0.037
        + 0.12501929087863276 * max(0.0, 0.34 - Q.z_dr_0p05_0p1) / 0.19804728356894793   # +12.5%  z_dr_0p05_0p1 < 0.34
        - 0.11499542608810046 * max(0.0, 0.0018 - Q.girth2) / 0.0004801657347841164   # -11.5%  girth2 < 0.0018
        - 0.11006371452732075 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21) / 0.009888827891376266   # -11.0%  LHA > 0.29 and tau21 < 0.69
        + 0.06705137443689345 * max(0.0, Q.lam2 - 0.00016) / 0.00045522149216667364   # +6.7%  lam2 > 0.00016
        - 0.036338783815945484 * max(0.0, 6.3 - Q.log_sum_pt) / 0.028104781515084085   # -3.6%  log_sum_pt < 6.3
        + 0.033611935504325625 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21) / 5.100858430075548e-05   # +3.4%  lam2 > 0.00022 and tau21 < 0.52
        + 0.032766728166855205 * max(0.0, Q.C2 - 0.059) / 0.00375706951116892   # +3.3%  C2 > 0.059
        + 0.017628674055909933 * max(0.0, Q.n_dr_0p2_0p4 - 1.5) / 0.10629663865546218   # +1.8%  n_dr_0p2_0p4 > 1.5
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 17.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.471515135883834 * (-0.1264916054968236
        + 0.3512071817302648 * max(0.0, 0.0088 - Q.width) / 0.004545275252912167   # +35.1%  width < 0.0088
        - 0.2689426509628021 * max(0.0, 0.088 - Q.girth) / 0.03699870548804189   # -26.9%  girth < 0.088
        + 0.2616796279443124 * max(0.0, 0.05 - Q.centroid_offset) / 0.033617202796922825   # +26.2%  centroid_offset < 0.05
        - 0.04939320212714525 * max(0.0, 0.0038 - Q.width) / 0.0012918773631350117   # -4.9%  width < 0.0038
        - 0.03320401352907217 * max(0.0, Q.girth - 0.074) / 0.01124272141366814   # -3.3%  girth > 0.074
        - 0.021054909947046816 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078) / 0.0003835882980443165   # -2.1%  planar_flow < 0.39 and width > 0.0078
        - 0.014518413759356622 * max(0.0, 0.15 - Q.LHA) / 0.01147776858577481   # -1.5%  LHA < 0.15
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 0.5487;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.5487088315473567 * (-2.0411554099495803
        + 0.47712729916216173 * max(0.0, Q.width - 0.018) / 0.0008445289123310189   # +47.7%  width > 0.018
        - 0.37197982018837306 * max(0.0, Q.e2 - 0.062) / 0.001907557126119234   # -37.2%  e2 > 0.062
        + 0.08783749917311244 * max(0.0, Q.mass - 91.2) / 0.5835013503307559   # +8.8%  mass > 91.2
        + 0.0630553814763528 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016) / 2.2321964317846704e-06   # +6.3%  girth2 > 0.019 and lam2 > 0.0016
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 11.66;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.664822294370994 * (0.18602940921330285
        + 0.5889638553057913 * max(0.0, 0.14 - Q.girth) / 0.08267339001142825   # +58.9%  girth < 0.14
        - 0.15720607436381462 * max(0.0, 0.047 - Q.e2) / 0.02225462282827147   # -15.7%  e2 < 0.047
        - 0.0720190042729927 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7) / 0.48842377131511927   # -7.2%  girth < 0.15 and pt_7 < 35
        - 0.0558352832184268 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031) / 0.02615697415657926   # -5.6%  tau21 < 0.52 and max_dr > 0.031
        - 0.05577331475924989 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt) / 0.013553870946555602   # -5.6%  girth < 0.15 and log_sum_pt < 6.7
        + 0.04565531710100336 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7) / 1109.502418284086   # +4.6%  sum_pt_top5 > 660 and pt_7 < 48
        - 0.01749289682746366 * max(0.0, Q.C2 - 0.065) / 0.0030870126006978787   # -1.7%  C2 > 0.065
        - 0.007054254151257704 * max(0.0, Q.sum_pt - 980.0) / 4.840389476102941   # -0.7%  sum_pt > 980
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 24.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 24.23220703365125 * (-0.002125270716302563
        + 0.27179996204585755 * max(0.0, 0.014 - Q.girth2) / 0.008888411541206169   # +27.2%  girth2 < 0.014
        - 0.17043570825112742 * max(0.0, 0.0075 - Q.width) / 0.003529943049801968   # -17.0%  width < 0.0075
        - 0.14454714200824084 * max(0.0, 0.091 - Q.girth) / 0.03944477782957522   # -14.5%  girth < 0.091
        - 0.08708158318089937 * max(0.0, 80.4 - Q.mass) / 41.295087132244156   # -8.7%  mass < 80.4
        + 0.07347874908416885 * max(0.0, 0.038 - Q.e2) / 0.015349588451562859   # +7.3%  e2 < 0.038
        - 0.06159923900491888 * max(0.0, 0.18 - Q.max_dr) / 0.07246046178070677   # -6.2%  max_dr < 0.18
        - 0.05169218485685859 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2) / 0.017112236686791932   # -5.2%  z_dr_0p05_0p1 < 0.63 and C2 < 0.067
        + 0.028545690533605502 * max(0.0, 0.64 - Q.z_dr_0p05_0p1) / 0.41174112079099234   # +2.9%  z_dr_0p05_0p1 < 0.64
        + 0.027550911545132912 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt) / 0.001167166770152925   # +2.8%  width < 0.0083 and log_sum_pt < 6.9
        + 0.022887732979873204 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97) / 6.755423679649392e-05   # +2.3%  girth2 < 0.014 and eccentricity > 0.97
        + 0.012676623982781075 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011) / 0.0002870865204098128   # +1.3%  planar_flow < 0.12 and centroid_offset > 0.011
        - 0.011573133265192563 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr) / 0.001445580212939085   # -1.2%  planar_flow < 0.12 and max_dr < 0.17
        - 0.010705350569597039 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow) / 5.4728749234238766e-05   # -1.1%  width < 0.0078 and planar_flow < 0.097
        - 0.00880471427689468 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018) / 0.00017488332723758759   # -0.9%  planar_flow < 0.12 and centroid_offset > 0.018
        - 0.006643044069226641 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2) / 0.0003545718485003723   # -0.7%  width < 0.0087 and D2 < 1
        - 0.005164104701019629 * max(0.0, Q.lam1 - 0.0075) / 0.002031455426275314   # -0.5%  lam1 > 0.0075
        + 0.004814125644605305 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr) / 6.90277451515299e-06   # +0.5%  lam1 > 0.0064 and max_dr < 0.16
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 21.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.989784418360692 * (0.08140143468188862
        - 0.28821763952233664 * max(0.0, 0.0069 - Q.width) / 0.003091631101787794   # -28.8%  width < 0.0069
        + 0.2860592747575048 * max(0.0, 0.0065 - Q.lam1) / 0.0028854962306376786   # +28.6%  lam1 < 0.0065
        - 0.22453834294296895 * max(0.0, 0.0081 - Q.lam1) / 0.004080619632208122   # -22.5%  lam1 < 0.0081
        + 0.14008523129855235 * max(0.0, 0.041 - Q.e2) / 0.017502522934382727   # +14.0%  e2 < 0.041
        + 0.021675348191848205 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4) / 0.014531592497916045   # +2.2%  tau21 < 0.29 and z_dr_0p2_0p4 < 0.2
        - 0.018130199514257983 * max(0.0, Q.LHA - 0.34) / 0.009917392506975143   # -1.8%  LHA > 0.34
        - 0.007764074507207878 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1) / 0.026428842820679765   # -0.8%  tau21 < 0.31 and z_dr_0p05_0p1 < 0.68
        - 0.006930875954833917 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9) / 2.0430089554584135e-05   # -0.7%  width < 0.0065 and log_sum_pt > 6.9
        + 0.006599013310489125 * max(0.0, Q.log_sum_pt - 6.9) / 0.0038490949621100413   # +0.7%  log_sum_pt > 6.9
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [1.0972802521008402, 0.6188720588235294, 2.429671953781513, 0.4310033613445378, 3.778063025210084, 2.3545018907563025, 1.64279012605042, 1.6582336134453781, 0.7163237394957983, 4.337557983193277, 3.1657599789915967, 2.602124474789916, 0.07484873949579832, 5.361462394957983, 0.4898024159663866, 0.5860214285714286]
T = [2.9419281274619222, 2.248359102547269, 3.9257877051601886, 2.7490873834690124, 4.641955225840336]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -0.4375 + T[0] * (   # class g: n2 +35%, n9 +25%, n5 -15%, n1 +8%, n6 +6%, n0 -6% ...
            + 0.3548691954419632 * h[2] / H_AVG[2]
            + 0.25341128200997964 * h[9] / H_AVG[9]
            - 0.15006114540863158 * h[5] / H_AVG[5]
            + 0.08217294492048059 * h[1] / H_AVG[1]
            + 0.061075649115799255 * h[6] / H_AVG[6]
            - 0.05827811964212419 * h[0] / H_AVG[0]
            - 0.04013166346102154 * h[4] / H_AVG[4]
        ),
        0.03125 + T[1] * (   # class q: n9 +49%, n10 -18%, n4 -16%, n6 +9%, n5 +5%, n8 +2% ...
            + 0.4898386029270935 * h[9] / H_AVG[9]
            - 0.17600391188650436 * h[10] / H_AVG[10]
            - 0.1575341804661735 * h[4] / H_AVG[4]
            + 0.09133272595274193 * h[6] / H_AVG[6]
            + 0.049087921944569055 * h[5] / H_AVG[5]
            + 0.01991240352475952 * h[8] / H_AVG[8]
            + 0.016290253298158034 * h[15] / H_AVG[15]
        ),
        -0.125 + T[2] * (   # class W: n11 +25%, n6 -13%, n15 -10%, n0 +10%, n13 +10%, n14 -9% ...
            + 0.24856073515223412 * h[11] / H_AVG[11]
            - 0.13076914824404864 * h[6] / H_AVG[6]
            - 0.10262646948873094 * h[15] / H_AVG[15]
            + 0.0960801028960054 * h[0] / H_AVG[0]
            + 0.09602603425294005 * h[13] / H_AVG[13]
            - 0.09357403903729442 * h[14] / H_AVG[14]
            + 0.09239893498682583 * h[7] / H_AVG[7]
            - 0.05489387018788769 * h[3] / H_AVG[3]
            - 0.04561656113970695 * h[8] / H_AVG[8]
            - 0.034527767967845055 * h[9] / H_AVG[9]
            - 0.0049263366464810275 * h[1] / H_AVG[1]
        ),
        -0.09375 + T[3] * (   # class Z: n7 +28%, n6 -22%, n4 +11%, n13 +11%, n3 -9%, n14 +7% ...
            + 0.28274728951018907 * h[7] / H_AVG[7]
            - 0.22409120240169755 * h[6] / H_AVG[6]
            + 0.1073669668048458 * h[4] / H_AVG[4]
            + 0.10665538552444115 * h[13] / H_AVG[13]
            - 0.08818904492238207 * h[3] / H_AVG[3]
            + 0.06681341127673374 * h[14] / H_AVG[14]
            - 0.049306794607505 * h[9] / H_AVG[9]
            - 0.033307725598282295 * h[15] / H_AVG[15]
            + 0.028139886646790964 * h[1] / H_AVG[1]
            + 0.013382292707132462 * h[5] / H_AVG[5]
        ),
        1.34375 + T[4] * (   # class t: n13 -47%, n10 +26%, n5 -13%, n4 +10%, n8 +3%, n12 -1% ...
            - 0.4692191096171935 * h[13] / H_AVG[13]
            + 0.2557456792157956 * h[10] / H_AVG[10]
            - 0.1268055041574677 * h[5] / H_AVG[5]
            + 0.10173684475075206 * h[4] / H_AVG[4]
            + 0.0289340794171809 * h[8] / H_AVG[8]
            - 0.008062199639403933 * h[12] / H_AVG[12]
            + 0.005803095629634615 * h[3] / H_AVG[3]
            + 0.003693487572571719 * h[0] / H_AVG[0]
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
