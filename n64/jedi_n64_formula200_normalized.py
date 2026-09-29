"""JEDI-linear jet tagger, 64 particles, 3 features: the formula with the fewest quantities (31) at the network's accuracy, as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  19.7%   (on for 58% of jets)
  neuron  4:  11.1%   (on for 67% of jets)
  neuron  1:  10.5%   (on for 92% of jets)
  neuron  5:   9.4%   (on for 78% of jets)
  neuron 13:   7.9%   (on for 94% of jets)
  neuron 10:   5.9%   (on for 83% of jets)
  neuron  0:   5.5%   (on for 49% of jets)
  neuron  6:   5.3%   (on for 76% of jets)
  neuron  9:   4.8%   (on for 48% of jets)
  neuron  7:   4.1%   (on for 28% of jets)
  neuron 12:   4.1%   (on for 68% of jets)
  neuron  3:   3.7%   (on for 59% of jets)
  neuron 14:   3.1%   (on for 49% of jets)
  neuron 11:   2.9%   (on for 89% of jets)
  neuron 15:   1.1%   (on for 61% of jets)
  neuron  2:   1.0%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 92.0% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau32                  N-subjettiness τ3/τ2
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
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        e2=e2,
        lam1=lam1,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def neuron_0(Q):
    # scale S = 10.5;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.504328107961232 * (0.1237585104576779
        - 0.30306270717003064 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # -30.3%  mass > 80.4
        + 0.22008543525154223 * max(0.0, Q.mass - 91.2) / 15.012010543283488   # +22.0%  mass > 91.2
        - 0.12432128176707541 * max(0.0, Q.mass_over_sum_pt - 0.0782) / 0.01952035178600386   # -12.4%  mass_over_sum_pt > 0.0782
        + 0.121333119874026 * max(0.0, Q.mass_over_sum_pt - 0.0857) / 0.015951475613508896   # +12.1%  mass_over_sum_pt > 0.0857
        + 0.07184157597497288 * max(0.0, 0.00784 - Q.mass_over_sum_pt_sq) / 0.0022195514289239412   # +7.2%  mass_over_sum_pt_sq < 0.00784
        - 0.028292869644643014 * max(0.0, 0.00611 - Q.girth2_top20) / 0.00183455299915498   # -2.8%  girth2_top20 < 0.00611
        - 0.027565468517748545 * max(0.0, 0.0562 - Q.girth) / 0.010645467858827445   # -2.8%  girth < 0.0562
        - 0.02668531683345658 * max(0.0, 0.00583 - Q.lam1) / 0.0014229001202209592   # -2.7%  lam1 < 0.00583
        - 0.016844598756245017 * max(0.0, 80.8 - Q.mass_top50) / 11.34238411426624   # -1.7%  mass_top50 < 80.8
        + 0.016409269508013195 * max(0.0, 10.8 - Q.n_dr_0p2_0p4) / 4.352736134447832   # +1.6%  n_dr_0p2_0p4 < 10.8
        + 0.012082068578892715 * max(0.0, 6.99 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top40 - 936.0) / 2.4836401678195297   # +1.2%  log_sum_pt < 6.99 and sum_pt_top40 > 936
        - 0.008476457410638426 * max(0.0, 1010.0 - Q.sum_pt) / 18.51132844792214   # -0.8%  sum_pt < 1010
        - 0.007999848386992716 * max(0.0, Q.mass - 80.4) * max(0.0, 7.0 - Q.n_dr_0p2_0p4) / 9.104337190781788   # -0.8%  mass > 80.4 and n_dr_0p2_0p4 < 7
        + 0.006384316555792848 * max(0.0, 70.3 - Q.mass_top50) * max(0.0, 0.231 - Q.z_dr_0p05_0p1) / 1.48698349993652   # +0.6%  mass_top50 < 70.3 and z_dr_0p05_0p1 < 0.231
        - 0.005044440805938299 * max(0.0, Q.z_dr_0_0p05 - 0.871) / 0.01784123277668831   # -0.5%  z_dr_0_0p05 > 0.871
        - 0.0025309843240446665 * max(0.0, Q.z_top30_slots - 0.913) * max(0.0, Q.C2 - 0.0639) / 0.0004128305865818579   # -0.3%  z_top30_slots > 0.913 and C2 > 0.0639
        - 0.0010402406399469394 * max(0.0, 9.75 - Q.n_dr_0p2_0p4) * max(0.0, 0.98 - Q.z_top50_slots) / 0.005130060560205736   # -0.1%  n_dr_0p2_0p4 < 9.75 and z_top50_slots < 0.98
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 14.03;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 14.025270758652399 * (0.050123809521916626
        + 0.12124102435283617 * max(0.0, Q.log_sum_pt - 6.91) / 0.051842627853807825   # +12.1%  log_sum_pt > 6.91
        + 0.09676257480057021 * max(0.0, 761.0 - Q.sum_pt_top3) / 302.9288640362395   # +9.7%  sum_pt_top3 < 761
        - 0.09188043905258629 * max(0.0, Q.log_sum_pt - 6.8) / 0.14897665146085345   # -9.2%  log_sum_pt > 6.8
        - 0.08558385931350615 * max(0.0, Q.z_top50_slots - 0.958) / 0.03489351161169589   # -8.6%  z_top50_slots > 0.958
        + 0.08430482381852333 * max(0.0, Q.log_sum_pt - 6.89) / 0.0671817034270044   # +8.4%  log_sum_pt > 6.89
        - 0.06234150326374294 * max(0.0, Q.sum_pt_top50 - 962.0) / 84.0727368053276   # -6.2%  sum_pt_top50 > 962
        + 0.058434402180722704 * max(0.0, Q.tau32 - 0.326) / 0.4017442706885412   # +5.8%  tau32 > 0.326
        + 0.057889234655885656 * max(0.0, 0.398 - Q.LHA) / 0.13974392255765236   # +5.8%  LHA < 0.398
        + 0.050926819038090404 * max(0.0, Q.n_particles - 26.7) / 19.62259411775032   # +5.1%  n_particles > 26.7
        - 0.04649588311036383 * max(0.0, 114.0 - Q.mass_top50) / 34.50356348071459   # -4.6%  mass_top50 < 114
        - 0.03962790704876061 * max(0.0, 0.997 - Q.z_top30_slots) / 0.046316010496464784   # -4.0%  z_top30_slots < 0.997
        - 0.03298846138579495 * max(0.0, Q.log_sum_pt - 6.98) / 0.023133605142356184   # -3.3%  log_sum_pt > 6.98
        + 0.03289278528759809 * max(0.0, 79.3 - Q.mass_top30) / 14.064945721487234   # +3.3%  mass_top30 < 79.3
        + 0.02323234491441682 * max(0.0, Q.n_particles - 34.8) * max(0.0, 0.131 - Q.dr_0) / 0.9101673960421702   # +2.3%  n_particles > 34.8 and dr_0 < 0.131
        - 0.016475560318694217 * max(0.0, 7.51 - Q.n_dr_0p2_0p4) / 2.315372689080118   # -1.6%  n_dr_0p2_0p4 < 7.51
        - 0.01634965904006219 * max(0.0, 0.429 - Q.max_dr) / 0.08399574902876229   # -1.6%  max_dr < 0.429
        + 0.013632062980240504 * max(0.0, Q.sum_pt_top40 - 1120.0) / 15.4188205078125   # +1.4%  sum_pt_top40 > 1120
        + 0.01321354390097499 * max(0.0, Q.n_particles - 38.8) * max(0.0, 0.985 - Q.z_top50_slots) / 0.08909785139063185   # +1.3%  n_particles > 38.8 and z_top50_slots < 0.985
        - 0.012722517980009711 * max(0.0, 0.000885 - Q.lam2) / 0.000301923450763891   # -1.3%  lam2 < 0.000885
        - 0.01219921926615218 * max(0.0, Q.sum_pt_top50 - 1150.0) / 14.377928844701943   # -1.2%  sum_pt_top50 > 1150
        + 0.011700868564323027 * max(0.0, 0.00133 - Q.girth2_top20) / 0.00015629319021527124   # +1.2%  girth2_top20 < 0.00133
        + 0.007898609338738247 * max(0.0, 1.15 - Q.D2) / 0.0826717422333017   # +0.8%  D2 < 1.15
        - 0.007651825566611035 * max(0.0, Q.z_dr_0_0p05 - 0.877) / 0.01630986707746179   # -0.8%  z_dr_0_0p05 > 0.877
        - 0.0035540708207958383 * max(0.0, 0.0156 - Q.girth) / 0.0005377217427949039   # -0.4%  girth < 0.0156
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.04875;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.048747668553705145 * (7.795244598854103
        + 1.0 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936) / 0.07652695220361876   # +100.0%  sum_pt_top50 > 1090 and C2 > 0.0936
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.309;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.3092373031755864 * (-0.16574535378243602
        + 0.20272828587585828 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979) / 0.019956348441918646   # +20.3%  n_dr_0p2_0p4 < 4.84 and z_top50_slots > 0.979
        + 0.1986252948371307 * max(0.0, 0.000569 - Q.lam2) / 0.0001274743359583435   # +19.9%  lam2 < 0.000569
        + 0.19357553520333107 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq) / 0.0024136791587655235   # +19.4%  mass_over_sum_pt_sq < 0.00813
        - 0.14880354497527232 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1) / 0.0010474147953031861   # -14.9%  tau21 < 0.419 and lam1 < 0.0208
        + 0.13610955105540343 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0) / 1360.3030653451615   # +13.6%  n_particles < 46 and sum_pt_top40 > 822
        - 0.06581666960931737 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0) / 0.014195986659522444   # -6.6%  n_dr_0p2_0p4 < 5.2 and dr_0 < 0.0493
        - 0.05434111844368688 * max(0.0, 0.000115 - Q.girth2_top5) / 1.0222043011890478e-05   # -5.4%  girth2_top5 < 0.000115
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 10;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.99966856649959 * (-0.04310142852573289
        - 0.1399205376421016 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -14.0%  mass < 80.4
        + 0.1194441200077314 * max(0.0, 0.0231 - Q.girth2_top20) / 0.01577809263268846   # +11.9%  girth2_top20 < 0.0231
        + 0.11417330118425718 * max(0.0, 103.0 - Q.mass) / 25.09220156012543   # +11.4%  mass < 103
        + 0.10936257848140132 * max(0.0, Q.mass_over_sum_pt_sq - 0.00841) / 0.003493896288791736   # +10.9%  mass_over_sum_pt_sq > 0.00841
        - 0.10097975480460811 * max(0.0, Q.mass_over_sum_pt - 0.0909) / 0.01408318103169422   # -10.1%  mass_over_sum_pt > 0.0909
        + 0.09036296318477995 * max(0.0, 6.81 - Q.D2) / 4.033927154172338   # +9.0%  D2 < 6.81
        - 0.04882740892796858 * max(0.0, 84.6 - Q.mass_top40) * max(0.0, 6.72 - Q.D2) / 32.55052708270859   # -4.9%  mass_top40 < 84.6 and D2 < 6.72
        - 0.04193980704899001 * max(0.0, 0.0698 - Q.z_dr_0p2_0p4) / 0.04152318517156865   # -4.2%  z_dr_0p2_0p4 < 0.0698
        - 0.03488674921908435 * max(0.0, 1080.0 - Q.sum_pt) / 62.97038439592634   # -3.5%  sum_pt < 1080
        - 0.03341845275748328 * max(0.0, 0.00777 - Q.girth2_top20) / 0.002905856100696149   # -3.3%  girth2_top20 < 0.00777
        - 0.029077044467200514 * max(0.0, Q.n_particles - 23.7) / 22.36621596656737   # -2.9%  n_particles > 23.7
        - 0.028126477611155706 * max(0.0, 0.0616 - Q.girth) / 0.01290162633278111   # -2.8%  girth < 0.0616
        + 0.026985953230067312 * max(0.0, 118.0 - Q.mass) * max(0.0, 0.411 - Q.max_dr) / 2.594717194728194   # +2.7%  mass < 118 and max_dr < 0.411
        + 0.025608082535962695 * max(0.0, 18.0 - Q.n_dr_0p2_0p4) / 9.963904201680672   # +2.6%  n_dr_0p2_0p4 < 18
        - 0.015369065161640447 * max(0.0, 0.454 - Q.tau21) / 0.10245703852889329   # -1.5%  tau21 < 0.454
        - 0.012418459433529495 * max(0.0, 74.3 - Q.mass) * max(0.0, 0.36 - Q.max_dr) / 0.22017815326562978   # -1.2%  mass < 74.3 and max_dr < 0.36
        + 0.00938572381925902 * max(0.0, Q.e2 - 0.0356) / 0.005045920830607038   # +0.9%  e2 > 0.0356
        + 0.007855445873892664 * max(0.0, Q.C2 - 0.107) / 0.004514474435689849   # +0.8%  C2 > 0.107
        + 0.00712410460539352 * max(0.0, 0.573 - Q.tau32) / 0.03138268056696412   # +0.7%  tau32 < 0.573
        - 0.00473397000349285 * max(0.0, Q.mass_over_sum_pt_sq - 0.0265) / 0.00039779942049310426   # -0.5%  mass_over_sum_pt_sq > 0.0265
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 8.715;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.714688313999218 * (-0.06265284314562453
        + 0.3701025556868541 * max(0.0, Q.log_sum_pt - 6.85) / 0.1020673549691605   # +37.0%  log_sum_pt > 6.85
        - 0.2623455717492836 * max(0.0, Q.log_sum_pt - 6.91) / 0.051842627853807825   # -26.2%  log_sum_pt > 6.91
        + 0.08768991796565201 * max(0.0, Q.mass_over_sum_pt - 0.0786) / 0.019297734933101458   # +8.8%  mass_over_sum_pt > 0.0786
        - 0.06455640101104645 * max(0.0, Q.mass_over_sum_pt - 0.0906) / 0.01417100537745123   # -6.5%  mass_over_sum_pt > 0.0906
        + 0.049266398583274106 * max(0.0, 65.0 - Q.n_particles) * max(0.0, 0.0378 - Q.e2) / 0.24961703953865497   # +4.9%  n_particles < 65 and e2 < 0.0378
        - 0.04461791742065771 * max(0.0, Q.mass_top50 - 155.0) / 1.8254987959661404   # -4.5%  mass_top50 > 155
        - 0.03342127912688425 * max(0.0, Q.z_top30_slots - 0.936) / 0.032799102550221185   # -3.3%  z_top30_slots > 0.936
        + 0.02179641504997557 * max(0.0, 0.881 - Q.tau32) / 0.16662189782727965   # +2.2%  tau32 < 0.881
        + 0.018102958188481462 * max(0.0, Q.sum_pt_top3 - 239.0) / 227.322245207458   # +1.8%  sum_pt_top3 > 239
        + 0.017288237562638507 * max(0.0, Q.max_dr - 0.437) / 0.0076477970485669105   # +1.7%  max_dr > 0.437
        - 0.01638238416982451 * max(0.0, 0.00636 - Q.mass_over_sum_pt_sq) / 0.001386090989128305   # -1.6%  mass_over_sum_pt_sq < 0.00636
        - 0.010219273473550764 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293) / 0.00019790618470558898   # -1.0%  mass_over_sum_pt_sq > 0.0293
        - 0.0023739236995620463 * max(0.0, Q.sum_pt_top3 - 791.0) / 4.274381223739495   # -0.2%  sum_pt_top3 > 791
        + 0.0018367663123149021 * max(0.0, Q.mass_top50 - 155.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.581) / 0.013013695867868389   # +0.2%  mass_top50 > 155 and z_dr_0p05_0p1 > 0.581
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 7.277;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.27739347926504 * (0.350400182052205
        - 0.2249582119704429 * max(0.0, Q.mass_over_sum_pt - 0.0491) / 0.040623062652625906   # -22.5%  mass_over_sum_pt > 0.0491
        - 0.2151580844270641 * max(0.0, 110.0 - Q.mass) / 30.46284125721154   # -21.5%  mass < 110
        + 0.10660055198338107 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # +10.7%  mass < 91.2
        - 0.08829540342508374 * max(0.0, 0.0101 - Q.mass_over_sum_pt_sq) / 0.0038708457417757866   # -8.8%  mass_over_sum_pt_sq < 0.0101
        - 0.084127609188949 * max(0.0, 18.5 - Q.n_dr_0p2_0p4) / 10.39439243697479   # -8.4%  n_dr_0p2_0p4 < 18.5
        + 0.08382785465714472 * max(0.0, Q.lam1 - 0.00793) / 0.002675650363432818   # +8.4%  lam1 > 0.00793
        + 0.06479977937499723 * max(0.0, Q.mass_over_sum_pt - 0.0497) * max(0.0, 18.6 - Q.n_dr_0p2_0p4) / 0.26642570162791984   # +6.5%  mass_over_sum_pt > 0.0497 and n_dr_0p2_0p4 < 18.6
        + 0.05973210967036369 * max(0.0, 0.0477 - Q.e2) / 0.018899741974689084   # +6.0%  e2 < 0.0477
        + 0.05711544688116133 * max(0.0, 70.5 - Q.mass_top50) / 7.827713384148253   # +5.7%  mass_top50 < 70.5
        - 0.010278697452187898 * max(0.0, Q.e2 - 0.0526) / 0.001441274100460316   # -1.0%  e2 > 0.0526
        + 0.00510625096922425 * max(0.0, 6.82 - Q.log_sum_pt) / 0.00534679100819041   # +0.5%  log_sum_pt < 6.82
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 21.57;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 21.567529039290882 * (-0.026474984638239014
        - 0.14510006706184064 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -14.5%  mass < 91.2
        + 0.12487976339318252 * max(0.0, 0.0095 - Q.mass_over_sum_pt_sq) / 0.003417954217515531   # +12.5%  mass_over_sum_pt_sq < 0.0095
        + 0.1146820388931956 * max(0.0, 120.0 - Q.mass) / 38.3474140172726   # +11.5%  mass < 120
        - 0.10504369128379021 * max(0.0, 0.0906 - Q.mass_over_sum_pt) / 0.017980419540932158   # -10.5%  mass_over_sum_pt < 0.0906
        - 0.08028681446973727 * max(0.0, 0.00795 - Q.girth2_top20) / 0.0030378740395582775   # -8.0%  girth2_top20 < 0.00795
        + 0.05435527977753267 * max(0.0, 0.0799 - Q.mass_over_sum_pt) / 0.011723090750407164   # +5.4%  mass_over_sum_pt < 0.0799
        + 0.04946986700836294 * max(0.0, 101.0 - Q.mass) * max(0.0, 0.397 - Q.max_dr) / 1.3107405322760761   # +4.9%  mass < 101 and max_dr < 0.397
        - 0.048699334326070834 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.392 - Q.max_dr) / 0.7897175242644119   # -4.9%  mass < 91.2 and max_dr < 0.392
        + 0.048126466201166826 * max(0.0, 0.0063 - Q.girth2_top20) / 0.0019437620924196942   # +4.8%  girth2_top20 < 0.0063
        - 0.042745835505529794 * max(0.0, 97.4 - Q.mass_top50) * max(0.0, 1.65 - Q.D2) / 2.376087754057384   # -4.3%  mass_top50 < 97.4 and D2 < 1.65
        + 0.035575889985758176 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.64 - Q.D2) / 2.810564250426583   # +3.6%  mass < 100 and D2 < 1.64
        - 0.03350441734292424 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq) / 0.001405851155618754   # -3.4%  mass_over_sum_pt_sq < 0.0064
        - 0.027117727690886943 * max(0.0, 0.0901 - Q.z_dr_0p2_0p4) / 0.05733944896595948   # -2.7%  z_dr_0p2_0p4 < 0.0901
        - 0.026419342391711703 * max(0.0, 0.0884 - Q.girth) / 0.029371130630526328   # -2.6%  girth < 0.0884
        + 0.021316172243376372 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) / 5.93209243697479   # +2.1%  n_dr_0p2_0p4 < 13
        + 0.01854698028646389 * max(0.0, 0.0286 - Q.e2) / 0.006015226104052112   # +1.9%  e2 < 0.0286
        - 0.00818036254317514 * max(0.0, 0.991 - Q.z_top50_slots) / 0.004873762616073404   # -0.8%  z_top50_slots < 0.991
        + 0.006115467862985412 * max(0.0, 1.8 - Q.D2) * max(0.0, 9.09 - Q.n_dr_0p2_0p4) / 1.5814811837384652   # +0.6%  D2 < 1.8 and n_dr_0p2_0p4 < 9.09
        + 0.005922021052892487 * max(0.0, 0.0059 - Q.z_dr_0p2_0p4) / 0.001287533881346277   # +0.6%  z_dr_0p2_0p4 < 0.0059
        - 0.0026405164460792875 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.422) / 0.3822108397955718   # -0.3%  mass < 91.2 and z_dr_0p05_0p1 > 0.422
        + 0.0012719442333368417 * max(0.0, 75.6 - Q.mass_top30) * max(0.0, 1.61 - Q.D2) / 0.14990543272596127   # +0.1%  mass_top30 < 75.6 and D2 < 1.61
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 29.09;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 29.093543408952108 * (0.20279413604135155
        - 0.37897179965046485 * max(0.0, 0.161 - Q.mass_over_sum_pt) / 0.07551803084862671   # -37.9%  mass_over_sum_pt < 0.161
        + 0.25318849391910137 * max(0.0, 0.0255 - Q.mass_over_sum_pt_sq) / 0.01670328897615098   # +25.3%  mass_over_sum_pt_sq < 0.0255
        - 0.07565512428169663 * max(0.0, Q.mass - 65.2) / 30.655649615587844   # -7.6%  mass > 65.2
        + 0.051975457336104806 * max(0.0, Q.mass - 86.5) / 16.97138298774527   # +5.2%  mass > 86.5
        + 0.042327143333096104 * max(0.0, Q.mass_over_sum_pt - 0.0757) * max(0.0, 1090.0 - Q.sum_pt) / 1.9392859558084554   # +4.2%  mass_over_sum_pt > 0.0757 and sum_pt < 1090
        - 0.028607718351533935 * max(0.0, Q.mass - 105.0) / 11.354705261820305   # -2.9%  mass > 105
        - 0.02489546714991368 * max(0.0, 20.6 - Q.n_dr_0p2_0p4) / 12.234752604935048   # -2.5%  n_dr_0p2_0p4 < 20.6
        + 0.0245159611075115 * max(0.0, Q.girth2_top20 - 0.00792) / 0.002935210611907685   # +2.5%  girth2_top20 > 0.00792
        + 0.024089833971447518 * max(0.0, 64.5 - Q.mass) / 5.939479918328454   # +2.4%  mass < 64.5
        - 0.02004322201336688 * max(0.0, Q.e2 - 0.0203) / 0.01340524941841733   # -2.0%  e2 > 0.0203
        - 0.012893983993267475 * max(0.0, 1040.0 - Q.sum_pt_top40) / 48.27949588448661   # -1.3%  sum_pt_top40 < 1040
        - 0.012048876366884275 * max(0.0, 0.00148 - Q.lam2) / 0.0007198039170616937   # -1.2%  lam2 < 0.00148
        + 0.010596262141806265 * max(0.0, 1000.0 - Q.sum_pt_top40) / 26.125662084345457   # +1.1%  sum_pt_top40 < 1000
        + 0.010420917227905306 * max(0.0, Q.n_particles - 50.7) / 4.0970460504210795   # +1.0%  n_particles > 50.7
        + 0.006902624072113654 * max(0.0, Q.log_sum_pt - 6.94) / 0.035231893522406375   # +0.7%  log_sum_pt > 6.94
        + 0.004629754808083955 * max(0.0, Q.C2 - 0.0695) / 0.01400166034114296   # +0.5%  C2 > 0.0695
        - 0.004341475498791525 * max(0.0, 46.1 - Q.n_particles) / 6.4443319328057775   # -0.4%  n_particles < 46.1
        + 0.0041172252924640395 * max(0.0, 1.8 - Q.D2) / 0.291446892386224   # +0.4%  D2 < 1.8
        - 0.00383818040469807 * max(0.0, Q.n_particles - 51.3) * max(0.0, Q.z_top50_slots - 0.97) / 0.04002375204855651   # -0.4%  n_particles > 51.3 and z_top50_slots > 0.97
        - 0.0030306067974740112 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, 0.39 - Q.max_dr) / 0.9521716027837525   # -0.3%  sum_pt_top40 < 1010 and max_dr < 0.39
        + 0.0029098722822743653 * max(0.0, 883.0 - Q.sum_pt_top40) * max(0.0, 6.89 - Q.log_sum_pt) / 1.2523446088588137   # +0.3%  sum_pt_top40 < 883 and log_sum_pt < 6.89
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 5.033;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.033190452898176 * (0.07728696214465892
        - 0.3901911188852714 * max(0.0, Q.mass_over_sum_pt - 0.0588) / 0.03300682713242026   # -39.0%  mass_over_sum_pt > 0.0588
        + 0.16398626028489627 * max(0.0, Q.mass_over_sum_pt_sq - 0.00745) / 0.0038211762947797   # +16.4%  mass_over_sum_pt_sq > 0.00745
        + 0.14584419870571602 * max(0.0, 80.4 - Q.mass_top40) / 12.19371475973745   # +14.6%  mass_top40 < 80.4
        + 0.13026524598108216 * max(0.0, Q.mass - 84.5) / 17.913928754546863   # +13.0%  mass > 84.5
        + 0.05097154928456538 * max(0.0, 1010.0 - Q.sum_pt_top40) / 30.43291995593159   # +5.1%  sum_pt_top40 < 1010
        - 0.03366349613130439 * max(0.0, 0.0417 - Q.girth) / 0.005763088004736598   # -3.4%  girth < 0.0417
        - 0.02969726876701806 * max(0.0, Q.mass_over_sum_pt_sq - 0.0257) / 0.0004745143163025603   # -3.0%  mass_over_sum_pt_sq > 0.0257
        - 0.02184032017865381 * max(0.0, Q.mass - 151.0) / 2.9235768886021205   # -2.2%  mass > 151
        - 0.021141248814595504 * max(0.0, 0.979 - Q.z_top50_slots) / 0.0025037160399051053   # -2.1%  z_top50_slots < 0.979
        + 0.0076729343646458085 * max(0.0, 115.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.375) / 0.6658506894801658   # +0.8%  mass_top40 < 115 and max_dr > 0.375
        + 0.004726358602251072 * max(0.0, 931.0 - Q.sum_pt_top40) * max(0.0, Q.z_top30_slots - 0.977) / 0.030934542254646633   # +0.5%  sum_pt_top40 < 931 and z_top30_slots > 0.977
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 9.577;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.576964794260252 * (0.49180508659934064
        - 0.20544327748245184 * max(0.0, 0.0254 - Q.girth2_top5) / 0.019833901569222594   # -20.5%  girth2_top5 < 0.0254
        - 0.14299664731950193 * max(0.0, 123.0 - Q.mass) / 40.75815050821785   # -14.3%  mass < 123
        + 0.10884731039006737 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +10.9%  mass < 80.4
        - 0.10765290901303293 * max(0.0, 0.0611 - Q.e2) / 0.0308679077729796   # -10.8%  e2 < 0.0611
        + 0.07312868695426061 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4) / 0.055583401619954995   # +7.3%  z_dr_0p2_0p4 < 0.0879
        - 0.0662818109140953 * max(0.0, Q.z_dr_0_0p05 - 0.769) / 0.05160801379057769   # -6.6%  z_dr_0_0p05 > 0.769
        + 0.05426471499149037 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3) / 4.060088008156796   # +5.4%  girth2_top5 < 0.0261 and sum_pt_top3 < 668
        - 0.052508511080309135 * max(0.0, 11.3 - Q.n_dr_0p2_0p4) / 4.699739831917243   # -5.3%  n_dr_0p2_0p4 < 11.3
        - 0.037938842405070915 * max(0.0, Q.mass - 144.0) / 3.9152904962106914   # -3.8%  mass > 144
        + 0.035145673343747526 * max(0.0, Q.mass_top50 - 137.0) / 4.2179057178401145   # +3.5%  mass_top50 > 137
        - 0.03211689357076344 * max(0.0, Q.lam1 - 0.00248) / 0.005738476847541136   # -3.2%  lam1 > 0.00248
        + 0.027904829783904243 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21) / 3.391415893785227   # +2.8%  mass < 122 and tau21 < 0.47
        - 0.024129978314293116 * max(0.0, 0.436 - Q.max_dr) / 0.09062429521656806   # -2.4%  max_dr < 0.436
        + 0.014382903413691977 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94) / 0.20286385807389345   # +1.4%  dr_0 < 0.0612 and n_dr_0p2_0p4 > -0.94
        - 0.009267638082307917 * max(0.0, Q.mass - 162.0) / 1.5963281230253332   # -0.9%  mass > 162
        - 0.007989372941011376 * max(0.0, 954.0 - Q.sum_pt_top50) / 9.21854739569656   # -0.8%  sum_pt_top50 < 954
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 1.773;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.7734976609175848 * (0.017930669264907342
        - 0.27333017563616663 * max(0.0, 0.0797 - Q.mass_over_sum_pt) / 0.011624710483187868   # -27.3%  mass_over_sum_pt < 0.0797
        + 0.24468023997793437 * max(0.0, 14.3 - Q.n_dr_0p2_0p4) / 6.931946218428433   # +24.5%  n_dr_0p2_0p4 < 14.3
        + 0.16171995583671483 * max(0.0, 0.0234 - Q.e2) / 0.0038810549851164933   # +16.2%  e2 < 0.0234
        + 0.11240990076332373 * max(0.0, 0.000827 - Q.lam2) / 0.0002665223209461664   # +11.2%  lam2 < 0.000827
        + 0.09954229011308864 * max(0.0, 0.0049 - Q.z_dr_0p2_0p4) / 0.0009646886266554227   # +10.0%  z_dr_0p2_0p4 < 0.0049
        + 0.060259947692640016 * max(0.0, 98.1 - Q.mass) * max(0.0, 1.27 - Q.D2) / 1.1744052338451987   # +6.0%  mass < 98.1 and D2 < 1.27
        - 0.0480574899801317 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.24 - Q.D2) / 3.0880379010628203   # -4.8%  mass < 80.4 and D2 < 3.24
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.912;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.9124711490227593 * (0.04271959869394491
        + 0.4685574461947894 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +46.9%  mass < 80.4
        - 0.22671149157147713 * max(0.0, Q.C2 - 0.0556) / 0.02007311049918363   # -22.7%  C2 > 0.0556
        - 0.20765642005360163 * max(0.0, 62.5 - Q.mass_top50) / 5.665291187758406   # -20.8%  mass_top50 < 62.5
        + 0.05615151718201521 * max(0.0, Q.mass_top30 - 128.0) / 2.9747411796249263   # +5.6%  mass_top30 > 128
        - 0.020870263077624143 * max(0.0, 6.85 - Q.log_sum_pt) / 0.007460518880088058   # -2.1%  log_sum_pt < 6.85
        + 0.020052861920492448 * max(0.0, Q.LHA - 0.406) * max(0.0, 0.0046 - Q.lam2) / 2.4427082725018427e-06   # +2.0%  LHA > 0.406 and lam2 < 0.0046
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 3.672;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.671681808522402 * (0.8361290983532864
        - 0.28427850906159313 * max(0.0, 7.0 - Q.log_sum_pt) / 0.0745557307339516   # -28.4%  log_sum_pt < 7
        + 0.19066267907533593 * max(0.0, 1040.0 - Q.sum_pt_top40) / 48.27949588448661   # +19.1%  sum_pt_top40 < 1040
        - 0.16054375288382486 * max(0.0, Q.mass - 143.0) / 4.065279840934177   # -16.1%  mass > 143
        - 0.1400462581122037 * max(0.0, 1020.0 - Q.sum_pt) / 22.752446825761556   # -14.0%  sum_pt < 1020
        + 0.06858137074148878 * max(0.0, Q.mass_top50 - 137.0) / 4.2179057178401145   # +6.9%  mass_top50 > 137
        - 0.04410511008918627 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 5.03 - Q.D2) / 110.91776053242677   # -4.4%  sum_pt_top40 < 1040 and D2 < 5.03
        - 0.03752624057364565 * max(0.0, 4.41 - Q.n_dr_0p2_0p4) / 0.8832334285672435   # -3.8%  n_dr_0p2_0p4 < 4.41
        + 0.022374631931735896 * max(0.0, 6.8 - Q.log_sum_pt) / 0.00436981537426272   # +2.2%  log_sum_pt < 6.8
        + 0.02065292326014219 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +2.1%  mass > 172.8
        + 0.01991820496876293 * max(0.0, Q.e2 - 0.0369) / 0.00457083192763921   # +2.0%  e2 > 0.0369
        + 0.011310319302080672 * max(0.0, 961.0 - Q.sum_pt_top50) * max(0.0, 5.16 - Q.D2) / 21.517043331621448   # +1.1%  sum_pt_top50 < 961 and D2 < 5.16
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 6.088;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.087515218829918 * (0.00809854238187448
        + 0.39730057146055925 * max(0.0, 135.0 - Q.mass) / 50.59776726393259   # +39.7%  mass < 135
        - 0.365265044117372 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -36.5%  mass < 91.2
        - 0.14314107229052747 * max(0.0, 0.0888 - Q.mass_over_sum_pt) / 0.016757181846311916   # -14.3%  mass_over_sum_pt < 0.0888
        - 0.04849999717025152 * max(0.0, Q.max_dr - 0.225) / 0.1318055673603188   # -4.8%  max_dr > 0.225
        - 0.02093310571312617 * max(0.0, 0.0129 - Q.girth2_top20) * max(0.0, 0.6 - Q.tau21) / 0.001079920335644331   # -2.1%  girth2_top20 < 0.0129 and tau21 < 0.6
        + 0.014700416562426743 * max(0.0, Q.log_sum_pt - 6.94) / 0.035231893522406375   # +1.5%  log_sum_pt > 6.94
        + 0.01015979268573689 * max(0.0, 0.96 - Q.z_top50_slots) / 0.0006918108791340068   # +1.0%  z_top50_slots < 0.96
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 11.07;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 11.070140857414565 * (0.19782946108885152
        - 0.4623463996291915 * max(0.0, Q.girth - 0.0293) / 0.04265199807344696   # -46.2%  girth > 0.0293
        + 0.33855613666325657 * max(0.0, Q.LHA - 0.163) / 0.10863374263780712   # +33.9%  LHA > 0.163
        - 0.12964816049653802 * max(0.0, 0.0176 - Q.lam1) / 0.010553113225010048   # -13.0%  lam1 < 0.0176
        + 0.035529309728787546 * max(0.0, Q.mass_over_sum_pt - 0.0926) / 0.013609496998767744   # +3.6%  mass_over_sum_pt > 0.0926
        + 0.02771940009950174 * max(0.0, 7.05 - Q.log_sum_pt) / 0.11756998604770712   # +2.8%  log_sum_pt < 7.05
        - 0.006200593382724695 * max(0.0, 0.0105 - Q.girth2_top5) * max(0.0, 0.328 - Q.max_dr) / 0.00011773832272095202   # -0.6%  girth2_top5 < 0.0105 and max_dr < 0.328
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.5217939075630252, 2.7636597163865546, 0.41227563025210084, 0.488697006302521, 0.9620025210084033, 0.9874659663865546, 0.8024293067226891, 0.4876982142857143, 2.043034243697479, 0.7931175420168067, 1.2523205882352941, 0.7316513655462185, 0.520090756302521, 1.8734215861344539, 0.48036680672268905, 0.21547258403361344]
T = [3.9837848099396007, 2.6266923155199584, 5.169440477284664, 5.183163967305672, 4.482969685530462]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +43%, n4 -21%, n9 +10%, n3 -9%, n5 -8%, n6 +3% ...
            + 0.433579474080023 * h[1] / H_AVG[1]
            - 0.21129459698279107 * h[4] / H_AVG[4]
            + 0.1026539464636937 * h[9] / H_AVG[9]
            - 0.09200365286107101 * h[3] / H_AVG[3]
            - 0.07745978490752789 * h[5] / H_AVG[5]
            + 0.03147247784634258 * h[6] / H_AVG[6]
            + 0.030598106279302636 * h[12] / H_AVG[12]
            + 0.01602617188464911 * h[8] / H_AVG[8]
            - 0.004911788694598978 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -39%, n1 -20%, n9 +17%, n11 -7%, n12 +7%, n6 +7% ...
            - 0.389131103225197 * h[4] / H_AVG[4]
            - 0.19727708257291762 * h[1] / H_AVG[1]
            + 0.16984426183016485 * h[9] / H_AVG[9]
            - 0.06963618856529326 * h[11] / H_AVG[11]
            + 0.06806324304626503 * h[12] / H_AVG[12]
            + 0.06682602671369277 * h[6] / H_AVG[6]
            + 0.019619524326095754 * h[2] / H_AVG[2]
            + 0.012153083126317372 * h[8] / H_AVG[8]
            - 0.007449486594056239 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -35%, n14 -13%, n5 +11%, n11 +8%, n0 +8%, n4 +6% ...
            - 0.34581207987412405 * h[8] / H_AVG[8]
            - 0.12777095744617964 * h[14] / H_AVG[14]
            + 0.11043337558982604 * h[5] / H_AVG[5]
            + 0.08403578689066415 * h[11] / H_AVG[11]
            + 0.07570363415381266 * h[0] / H_AVG[0]
            + 0.06396985670881315 * h[4] / H_AVG[4]
            - 0.05896409587613218 * h[7] / H_AVG[7]
            + 0.04135939686255128 * h[3] / H_AVG[3]
            - 0.04087228988830163 * h[12] / H_AVG[12]
            - 0.03356155527441286 * h[9] / H_AVG[9]
            + 0.009701597666235469 * h[6] / H_AVG[6]
            - 0.007815373768946825 * h[15] / H_AVG[15]
        ),
        0.984375 + T[3] * (   # class Z: n8 -37%, n6 -15%, n0 -14%, n5 +13%, n7 +9%, n3 +4% ...
            - 0.36953193368914905 * h[8] / H_AVG[8]
            - 0.14997661578738192 * h[6] / H_AVG[6]
            - 0.1384225209591653 * h[0] / H_AVG[0]
            + 0.1309784633812492 * h[5] / H_AVG[5]
            + 0.08527156568542402 * h[7] / H_AVG[7]
            + 0.041249889373747455 * h[3] / H_AVG[3]
            + 0.03135697855011594 * h[12] / H_AVG[12]
            - 0.02982798968347208 * h[2] / H_AVG[2]
            + 0.023384042890295022 * h[15] / H_AVG[15]
        ),
        0.78125 + T[4] * (   # class t: n13 -38%, n10 +27%, n5 -10%, n8 +10%, n12 -4%, n4 +4% ...
            - 0.3787195612574062 * h[13] / H_AVG[13]
            + 0.27498581643838355 * h[10] / H_AVG[10]
            - 0.10325179606672408 * h[5] / H_AVG[5]
            + 0.09613103946941882 * h[8] / H_AVG[8]
            - 0.043505543712006454 * h[12] / H_AVG[12]
            + 0.04023571100006938 * h[4] / H_AVG[4]
            - 0.03059693292385639 * h[7] / H_AVG[7]
            - 0.01802426174627229 * h[15] / H_AVG[15]
            + 0.014549337385862844 * h[0] / H_AVG[0]
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
