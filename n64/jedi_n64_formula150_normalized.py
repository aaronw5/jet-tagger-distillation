"""JEDI-linear jet tagger, 64 particles, 3 features: the formula simplified by hand with the training data (main result), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  19.3%   (on for 59% of jets)
  neuron  1:  11.0%   (on for 96% of jets)
  neuron  4:  10.8%   (on for 67% of jets)
  neuron  5:   9.6%   (on for 78% of jets)
  neuron 13:   7.8%   (on for 93% of jets)
  neuron 10:   6.0%   (on for 83% of jets)
  neuron  0:   5.5%   (on for 50% of jets)
  neuron  6:   5.2%   (on for 76% of jets)
  neuron  9:   4.8%   (on for 54% of jets)
  neuron 12:   4.0%   (on for 68% of jets)
  neuron  7:   4.0%   (on for 29% of jets)
  neuron  3:   3.7%   (on for 59% of jets)
  neuron 14:   3.1%   (on for 62% of jets)
  neuron 11:   2.9%   (on for 92% of jets)
  neuron 15:   1.2%   (on for 83% of jets)
  neuron  2:   1.0%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.0% (the network: 81.1%); same class as the network for 91.8% of jets.

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
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
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
        mass_top10=mass_of(10),
        mass_top20=mass_of(20),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_9=pt[9],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
    )


def neuron_0(Q):
    # scale S = 9.981;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.980738899947054 * (0.11522193011241887
        - 0.3992032191049005 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # -39.9%  mass > 80.4
        + 0.3053319168887099 * max(0.0, Q.mass - 91.2) / 15.012010543283488   # +30.5%  mass > 91.2
        + 0.1153875510641846 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq) / 0.0024766731601632993   # +11.5%  mass_over_sum_pt_sq < 0.00822
        - 0.04275075278492192 * max(0.0, 0.00578 - Q.lam1) / 0.0013989642666311151   # -4.3%  lam1 < 0.00578
        - 0.03793194869514506 * max(0.0, 0.0555 - Q.girth) / 0.010372297969655622   # -3.8%  girth < 0.0555
        - 0.03174299696306849 * max(0.0, 0.00622 - Q.girth2_top20) / 0.0018971171532347238   # -3.2%  girth2_top20 < 0.00622
        - 0.014331872501479305 * max(0.0, 81.8 - Q.mass_top50) / 11.82170887476   # -1.4%  mass_top50 < 81.8
        + 0.013398629856873302 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.4875243697478995   # +1.3%  n_dr_0p2_0p4 < 11
        + 0.01112375426806101 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1) / 1.7994049746627003   # +1.1%  mass_top50 < 72 and z_dr_0p05_0p1 < 0.259
        - 0.009173727143616665 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4) / 2.49483856566199   # -0.9%  sum_pt < 1010 and z_dr_0p2_0p4 < 0.212
        - 0.006861657205832693 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771) / 3.701859945532777   # -0.7%  sum_pt < 1020 and z_dr_0p1_0p2 > 0.0771
        - 0.006525162488951537 * max(0.0, Q.z_dr_0_0p05 - 0.883) / 0.014835066761265139   # -0.7%  z_dr_0_0p05 > 0.883
        - 0.006236811034255043 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4) / 8.892568928744048   # -0.6%  mass > 80.4 and n_dr_0p2_0p4 < 6.91
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 9.135;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.135148961993325 * (0.1258873834224883
        + 0.24005554450614697 * max(0.0, Q.log_sum_pt - 6.91) / 0.051842627853807825   # +24.0%  log_sum_pt > 6.91
        + 0.11531509724372026 * max(0.0, 662.0 - Q.sum_pt_top2) / 297.5764381039916   # +11.5%  sum_pt_top2 < 662
        - 0.11118162967992631 * max(0.0, Q.sum_pt_top50 - 930.0) / 112.22770706776852   # -11.1%  sum_pt_top50 > 930
        - 0.10654473218085356 * max(0.0, Q.z_top50_slots - 0.96) / 0.0329932881216197   # -10.7%  z_top50_slots > 0.96
        + 0.10585192400410237 * max(0.0, Q.tau32 - 0.276) / 0.44975492729817307   # +10.6%  tau32 > 0.276
        - 0.05157604135335676 * max(0.0, Q.log_sum_pt - 6.99) / 0.021033697349680438   # -5.2%  log_sum_pt > 6.99
        + 0.045770190801306565 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0) / 0.7602136563432582   # +4.6%  n_particles > 38.6 and dr_0 < 0.136
        - 0.035089244538897564 * max(0.0, 0.931 - Q.z_top20_slots) / 0.0586006354344863   # -3.5%  z_top20_slots < 0.931
        + 0.03251729436581276 * max(0.0, 1070.0 - Q.sum_pt_top30) / 94.90425810628939   # +3.3%  sum_pt_top30 < 1070
        - 0.030725086398370495 * max(0.0, 33.6 - Q.pt_9) / 8.206966114597352   # -3.1%  pt_9 < 33.6
        - 0.03062745316297118 * max(0.0, 0.435 - Q.max_dr) / 0.08967511120840235   # -3.1%  max_dr < 0.435
        - 0.024418679270321816 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4) / 921.7697214443317   # -2.4%  sum_pt_top2 < 748 and n_dr_0p2_0p4 < 7.95
        + 0.022298541323407767 * max(0.0, 0.00073 - Q.girth2_top3) / 0.00017410298856794336   # +2.2%  girth2_top3 < 0.00073
        + 0.017558735934783118 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0) / 49.65995924103681   # +1.8%  mass_top20 < 48.5 and n_real_top40 > 27
        + 0.01495056101438882 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots) / 0.09228081211608913   # +1.5%  n_particles > 37.9 and z_top50_slots < 0.985
        - 0.010003064393465126 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1) / 0.0011422435413839416   # -1.0%  girth2_top3 < 0.000994 and n_dr_0p05_0p1 < 9.47
        - 0.005516179828168789 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6) / 0.03625260750465081   # -0.6%  n_dr_0p1_0p2 < 6.42 and dr_6 < 0.0751
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
    # scale S = 6.352;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.351604323721392 * (0.0777756256250179
        - 0.2337368247383768 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -23.4%  mass < 80.4
        + 0.21594164885274905 * max(0.0, 109.0 - Q.mass) / 29.68779027109627   # +21.6%  mass < 109
        + 0.15332988780963647 * max(0.0, 6.52 - Q.D2) / 3.7747704587883835   # +15.3%  D2 < 6.52
        - 0.07377598309871326 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2) / 28.748211855005128   # -7.4%  mass_top40 < 84.9 and D2 < 6.27
        - 0.0653173544377374 * max(0.0, Q.n_particles - 22.7) / 23.307302857346937   # -6.5%  n_particles > 22.7
        - 0.051884338990217445 * max(0.0, 1070.0 - Q.sum_pt) / 55.20080265723477   # -5.2%  sum_pt < 1070
        - 0.04836519480347916 * max(0.0, 0.063 - Q.girth) / 0.013532889005788796   # -4.8%  girth < 0.063
        + 0.04452264292033041 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr) / 2.570820102511584   # +4.5%  mass < 124 and max_dr < 0.401
        - 0.04195920806167882 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4) / 0.04144763411267258   # -4.2%  z_dr_0p2_0p4 < 0.0697
        - 0.02620103165765784 * max(0.0, 0.451 - Q.tau21) / 0.10085974906832765   # -2.6%  tau21 < 0.451
        - 0.019210077162497157 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr) / 0.1871392778594149   # -1.9%  mass < 74.4 and max_dr < 0.35
        + 0.013316194192701806 * max(0.0, Q.C2 - 0.106) / 0.0046728837906010205   # +1.3%  C2 > 0.106
        + 0.012439613274224466 * max(0.0, 0.578 - Q.tau32) / 0.03238176289261727   # +1.2%  tau32 < 0.578
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 7.89;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.889776650653197 * (-0.10963554968674384
        + 0.39068712037791986 * max(0.0, Q.log_sum_pt - 6.85) / 0.1020673549691605   # +39.1%  log_sum_pt > 6.85
        - 0.2878037237985425 * max(0.0, Q.log_sum_pt - 6.91) / 0.051842627853807825   # -28.8%  log_sum_pt > 6.91
        + 0.11332365053379272 * max(0.0, Q.mass - 63.2) / 32.16180906288851   # +11.3%  mass > 63.2
        + 0.054596664282832254 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2) / 0.293030943579756   # +5.5%  n_particles < 67.6 and e2 < 0.0387
        - 0.042844210393591396 * max(0.0, Q.mass_top50 - 102.0) / 10.939522678933024   # -4.3%  mass_top50 > 102
        - 0.042721348968647974 * max(0.0, Q.mass_top50 - 157.0) / 1.6127363711829947   # -4.3%  mass_top50 > 157
        - 0.021036300730143297 * max(0.0, Q.z_top30_slots - 0.943) / 0.02803576255351409   # -2.1%  z_top30_slots > 0.943
        + 0.019220945189371338 * max(0.0, Q.max_dr - 0.436) / 0.007697916982669384   # +1.9%  max_dr > 0.436
        + 0.014629872360208228 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32) / 0.0014932267186236413   # +1.5%  girth2_top30 < 0.0226 and tau32 < 0.858
        - 0.011237576762171526 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293) / 0.00019790618470558898   # -1.1%  mass_over_sum_pt_sq > 0.0293
        + 0.0018985866027788356 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587) / 0.011348048672611692   # +0.2%  mass_top50 > 156 and z_dr_0p05_0p1 > 0.587
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 6.26;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.25993267762414 * (0.19489027483647509
        - 0.2836643141598435 * max(0.0, 101.0 - Q.mass) / 23.58193239953578   # -28.4%  mass < 101
        + 0.1702350653219159 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # +17.0%  mass < 91.2
        - 0.15028497770269492 * max(0.0, Q.mass_over_sum_pt - 0.0534) / 0.03718473687261325   # -15.0%  mass_over_sum_pt > 0.0534
        - 0.09152081577990948 * max(0.0, 17.7 - Q.n_dr_0p2_0p4) / 9.710409243787701   # -9.2%  n_dr_0p2_0p4 < 17.7
        + 0.08660829598499864 * max(0.0, Q.lam1 - 0.00717) / 0.002914850011773315   # +8.7%  lam1 > 0.00717
        + 0.07391540259195194 * max(0.0, 73.1 - Q.mass_top50) / 8.600472938013086   # +7.4%  mass_top50 < 73.1
        + 0.07135261522534886 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4) / 0.25378554981994494   # +7.1%  mass_over_sum_pt > 0.0538 and n_dr_0p2_0p4 < 19.4
        + 0.0605777873462342 * max(0.0, 92.5 - Q.mass_top10) / 46.47216550819333   # +6.1%  mass_top10 < 92.5
        - 0.01184072588710253 * max(0.0, Q.e2 - 0.0521) / 0.0015004483179648425   # -1.2%  e2 > 0.0521
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 13.72;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.71689434717793 * (-0.052927432524065836
        - 0.2725740250175945 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -27.3%  mass < 91.2
        + 0.15283589253204713 * max(0.0, 101.0 - Q.mass) / 23.58193239953578   # +15.3%  mass < 101
        + 0.12299402116970544 * max(0.0, 0.0129 - Q.girth2_top40) / 0.006271732318659536   # +12.3%  girth2_top40 < 0.0129
        + 0.07632310723475025 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr) / 1.2782857120725286   # +7.6%  mass < 102 and max_dr < 0.393
        - 0.07453745568106111 * max(0.0, 0.00812 - Q.girth2_top20) / 0.003165394441128708   # -7.5%  girth2_top20 < 0.00812
        - 0.07421848912769775 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr) / 0.7771352473067206   # -7.4%  mass < 91.2 and max_dr < 0.391
        - 0.06992133463064143 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2) / 2.184746149754401   # -7.0%  mass_top50 < 97.2 and D2 < 1.61
        + 0.05983811888212177 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2) / 2.6735933375893715   # +6.0%  mass < 100 and D2 < 1.61
        + 0.036638105217874965 * max(0.0, 66.0 - Q.mass_top20) / 12.532693724548116   # +3.7%  mass_top20 < 66
        + 0.02525208368460516 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4) / 1.8723252105287815   # +2.5%  D2 < 2.06 and n_dr_0p2_0p4 < 8.39
        - 0.01580893328126615 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50) / 0.00038448487156154737   # -1.6%  D2 < 1.99 and girth2_top50 < 0.00792
        - 0.013962194561304323 * max(0.0, 0.998 - Q.z_top50_slots) / 0.006939056074353395   # -1.4%  z_top50_slots < 0.998
        + 0.0050962389793300835 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50) / 9.776863167474217e-05   # +0.5%  D2 < 2 and girth2_top50 < 0.00623
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 6.763;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 6.7625443318057705 * (0.10661667630169527
        + 0.23308699059187749 * max(0.0, Q.mass - 91.2) / 15.012010543283488   # +23.3%  mass > 91.2
        + 0.18700822943304296 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt) / 2.651260884598444   # +18.7%  mass_over_sum_pt > 0.0762 and sum_pt < 1130
        - 0.1868537097588137 * max(0.0, Q.mass - 104.0) / 11.59272014501235   # -18.7%  mass > 104
        - 0.1189197135285951 * max(0.0, 20.2 - Q.n_dr_0p2_0p4) / 11.87887495809109   # -11.9%  n_dr_0p2_0p4 < 20.2
        + 0.0528583706668918 * max(0.0, Q.n_particles - 49.3) / 4.684889579841121   # +5.3%  n_particles > 49.3
        + 0.038870113238918154 * max(0.0, 1010.0 - Q.sum_pt) / 18.51132844792214   # +3.9%  sum_pt < 1010
        - 0.03458358310853995 * max(0.0, 0.00143 - Q.lam2) / 0.0006818455216448707   # -3.5%  lam2 < 0.00143
        + 0.03126623440140411 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30) / 0.3077719013509839   # +3.1%  girth2_top40 > 0.00145 and sum_pt_top30 < 932
        + 0.030901465684962808 * max(0.0, 0.00815 - Q.girth2_top30) / 0.00281255089653211   # +3.1%  girth2_top30 < 0.00815
        - 0.026659896260715594 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr) / 2.7995144463421133   # -2.7%  sum_pt_top40 < 1040 and max_dr < 0.415
        - 0.017771992199928966 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972) / 0.038520475998886534   # -1.8%  n_particles > 50.6 and z_top50_slots > 0.972
        + 0.017357885086376645 * max(0.0, Q.C2 - 0.0661) / 0.015324212454701346   # +1.7%  C2 > 0.0661
        + 0.013426184417306997 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt) / 1.3196971995499767   # +1.3%  sum_pt_top40 < 958 and log_sum_pt < 6.82
        - 0.0104356316226258 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50) / 67.2108775984773   # -1.0%  sum_pt < 1010 and n_real_top50 < 43.2
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 8.643;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.643461037461963 * (0.0014808882627599083
        - 0.2829326853770728 * max(0.0, Q.mass - 61.8) / 33.227141878820824   # -28.3%  mass > 61.8
        + 0.2571218314474514 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # +25.7%  mass > 80.4
        + 0.24095250395766846 * max(0.0, 128.0 - Q.mass_top50) / 45.87364713297371   # +24.1%  mass_top50 < 128
        - 0.15154212117157875 * max(0.0, 0.0191 - Q.girth2_top15) / 0.012594696345008467   # -15.2%  girth2_top15 < 0.0191
        - 0.02356641414783652 * max(0.0, Q.mass - 142.0) / 4.217295703509675   # -2.4%  mass > 142
        + 0.02248043802034487 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1) / 0.08832217733360428   # +2.2%  girth2_top15 < 0.0145 and n_particles > 35.1
        - 0.007011324289492962 * max(0.0, Q.width - 0.026) / 0.00045225453968091863   # -0.7%  width > 0.026
        + 0.006733040414296017 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt) / 0.015857431194720537   # +0.7%  girth2_top40 < 0.00576 and sum_pt < 997
        + 0.0042792284292184485 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7) / 0.00023117090123968262   # +0.4%  log_sum_pt < 6.85 and dr_7 < 0.0848
        + 0.00338041274503989 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386) / 0.4960690297502909   # +0.3%  mass_top40 < 113 and max_dr > 0.386
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
    # scale S = 1.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.2398943688200665 * (0.15807798222885833
        + 0.4114517759243579 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2) / 71.65122752944202   # +41.1%  n_dr_0p2_0p4 < 10.6 and n_dr_0p1_0p2 < 26.9
        - 0.22135412161326995 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2) / 0.007055417195460298   # -22.1%  n_dr_0p2_0p4 < 10 and girth2 < 0.00648
        + 0.20559861033938492 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4) / 0.0014321379730170478   # +20.6%  z_dr_0p2_0p4 < 0.00632
        + 0.08360496838641136 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow) / 1.0064206748319529   # +8.4%  mass < 103 and planar_flow < 0.362
        - 0.07799052373657583 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2) / 3.4049299719122574   # -7.8%  mass < 80.4 and D2 < 3.34
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.873;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.8733469645817176 * (0.045373335322844946
        + 0.4600987878816537 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +46.0%  mass < 80.4
        - 0.2183213423497926 * max(0.0, Q.C2 - 0.0528) / 0.021525874947073156   # -21.8%  C2 > 0.0528
        - 0.20404027630185723 * max(0.0, 63.0 - Q.mass_top50) / 5.791488367613624   # -20.4%  mass_top50 < 63
        + 0.0735520566490851 * max(0.0, Q.mass_top30 - 136.0) / 2.008579038809127   # +7.4%  mass_top30 > 136
        - 0.026489720335770354 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065) / 0.03938447395526507   # -2.6%  mass_top20 > 131 and C2 > 0.065
        + 0.01749781648184093 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2) / 2.664998487241029e-06   # +1.7%  LHA > 0.405 and lam2 < 0.00471
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 2.678;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.677702002207887 * (1.1315672907222787
        - 0.34234230868677 * max(0.0, 6.99 - Q.log_sum_pt) / 0.06642686126166916   # -34.2%  log_sum_pt < 6.99
        - 0.19458141043339622 * max(0.0, 1020.0 - Q.sum_pt) / 22.752446825761556   # -19.5%  sum_pt < 1020
        + 0.17423643486261342 * max(0.0, 1020.0 - Q.sum_pt_top40) / 35.614751945739236   # +17.4%  sum_pt_top40 < 1020
        - 0.1348132777482234 * max(0.0, Q.mass - 144.0) / 3.9152904962106914   # -13.5%  mass > 144
        - 0.05703985155184067 * max(0.0, 4.7 - Q.n_dr_0p2_0p4) / 0.9917904201688598   # -5.7%  n_dr_0p2_0p4 < 4.7
        + 0.040734382122113465 * max(0.0, 6.82 - Q.log_sum_pt) / 0.00534679100819041   # +4.1%  log_sum_pt < 6.82
        + 0.02820521856281281 * max(0.0, Q.mass_top10 - 66.7) / 4.841357065279162   # +2.8%  mass_top10 > 66.7
        + 0.028047116032229962 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +2.8%  mass > 172.8
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 4.857;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.856943244731604 * (0.07185589421465143
        - 0.4747657824643652 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -47.5%  mass < 91.2
        + 0.3660494303128023 * max(0.0, 131.0 - Q.mass) / 47.28407733498982   # +36.6%  mass < 131
        - 0.09283912097192128 * max(0.0, 0.089 - Q.mass_over_sum_pt) / 0.016888177582449187   # -9.3%  mass_over_sum_pt < 0.089
        - 0.05275824559044771 * max(0.0, Q.max_dr - 0.237) / 0.12086971911529051   # -5.3%  max_dr > 0.237
        + 0.013587420660463406 * max(0.0, 0.961 - Q.z_top50_slots) / 0.0007465308935538956   # +1.4%  z_top50_slots < 0.961
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 0.4223;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.4222823970804453 * (-0.48308904517546764
        + 0.5200397046641811 * max(0.0, 7.04 - Q.log_sum_pt) / 0.10871465993197885   # +52.0%  log_sum_pt < 7.04
        + 0.4799602953358188 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow) / 0.06050112955756658   # +48.0%  z_dr_0p1_0p2 < 0.231 and planar_flow < 1.09
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.5234828256302521, 2.886987394957983, 0.41227563025210084, 0.488697006302521, 0.9381055672268908, 1.0027632352941176, 0.7881045168067227, 0.4732985294117647, 1.9954369747899159, 0.7874355042016806, 1.2523205882352941, 0.7423908613445378, 0.5167867647058824, 1.8370240546218488, 0.4804859243697479, 0.2322189075630252]
T = [4.037305334164917, 2.6189014673056725, 5.126888714548319, 5.132841117384453, 4.443836367515757]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -20%, n9 +10%, n3 -9%, n5 -8%, n6 +3% ...
            + 0.44692362169876804 * h[1] / H_AVG[1]
            - 0.20331441478485893 * h[4] / H_AVG[4]
            + 0.10056743254420557 * h[9] / H_AVG[9]
            - 0.09078400675452082 * h[3] / H_AVG[3]
            - 0.07761699576636762 * h[5] / H_AVG[5]
            + 0.0305008713879009 * h[6] / H_AVG[6]
            + 0.030000678163469702 * h[12] / H_AVG[12]
            + 0.015445303315183364 * h[8] / H_AVG[8]
            - 0.004846675584724841 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -38%, n1 -21%, n9 +17%, n11 -7%, n12 +7%, n6 +7% ...
            - 0.3805936105736789 * h[4] / H_AVG[4]
            - 0.20669358634233842 * h[1] / H_AVG[1]
            + 0.16912910876678935 * h[9] / H_AVG[9]
            - 0.07086853692402467 * h[11] / H_AVG[11]
            + 0.06783204812604454 * h[12] / H_AVG[12]
            + 0.06582831206278013 * h[6] / H_AVG[6]
            + 0.019677889536841294 * h[2] / H_AVG[2]
            + 0.011905259942127225 * h[8] / H_AVG[8]
            - 0.007471647725375303 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -34%, n14 -13%, n5 +11%, n11 +9%, n0 +8%, n4 +6% ...
            - 0.34055885550755516 * h[8] / H_AVG[8]
            - 0.12886336778359514 * h[14] / H_AVG[14]
            + 0.11307491300900717 * h[5] / H_AVG[5]
            + 0.08597701227110671 * h[11] / H_AVG[11]
            + 0.07657902113392728 * h[0] / H_AVG[0]
            + 0.06289853489879266 * h[4] / H_AVG[4]
            - 0.05769806940473332 * h[7] / H_AVG[7]
            + 0.041702668452827774 * h[3] / H_AVG[3]
            - 0.040949713335110084 * h[12] / H_AVG[12]
            - 0.0335976702703392 * h[9] / H_AVG[9]
            + 0.009607490047647677 * h[6] / H_AVG[6]
            - 0.00849268388535779 * h[15] / H_AVG[15]
        ),
        0.984375 + T[3] * (   # class Z: n8 -36%, n6 -15%, n0 -14%, n5 +13%, n7 +8%, n3 +4% ...
            - 0.36446134238006805 * h[8] / H_AVG[8]
            - 0.14874340218143317 * h[6] / H_AVG[6]
            - 0.14023206033082514 * h[0] / H_AVG[0]
            + 0.13431152620909567 * h[5] / H_AVG[5]
            + 0.08356518007671751 * h[7] / H_AVG[7]
            + 0.041654307111360915 * h[3] / H_AVG[3]
            + 0.03146325013326768 * h[12] / H_AVG[12]
            - 0.030120426058174814 * h[2] / H_AVG[2]
            + 0.02544850551905714 * h[15] / H_AVG[15]
        ),
        0.78125 + T[4] * (   # class t: n13 -37%, n10 +28%, n5 -11%, n8 +9%, n12 -4%, n4 +4% ...
            - 0.3746319422719265 * h[13] / H_AVG[13]
            + 0.277407396918457 * h[10] / H_AVG[10]
            - 0.10577465677632221 * h[5] / H_AVG[5]
            + 0.09471826864431804 * h[8] / H_AVG[8]
            - 0.04360984985435982 * h[12] / H_AVG[12]
            + 0.03958174408509391 * h[4] / H_AVG[4]
            - 0.029955020929691517 * h[7] / H_AVG[7]
            - 0.01959615141833318 * h[15] / H_AVG[15]
            + 0.014724969101497749 * h[0] / H_AVG[0]
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
