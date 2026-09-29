"""JEDI-linear jet tagger, 64 particles, 3 features: the simpler version of the simplified formula, as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  19.7%   (on for 62% of jets)
  neuron  1:  10.9%   (on for 94% of jets)
  neuron  4:  10.7%   (on for 69% of jets)
  neuron  5:  10.0%   (on for 85% of jets)
  neuron 13:   8.3%   (on for 93% of jets)
  neuron 10:   6.1%   (on for 87% of jets)
  neuron  0:   5.4%   (on for 50% of jets)
  neuron  6:   5.1%   (on for 88% of jets)
  neuron  9:   4.9%   (on for 58% of jets)
  neuron 12:   4.1%   (on for 69% of jets)
  neuron  7:   3.7%   (on for 31% of jets)
  neuron  3:   3.1%   (on for 59% of jets)
  neuron 11:   2.8%   (on for 93% of jets)
  neuron 14:   2.8%   (on for 65% of jets)
  neuron 15:   1.3%   (on for 100% of jets)
  neuron  2:   1.0%   (on for 100% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 80.5% (the network: 81.1%); same class as the network for 91.3% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
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
        mass_top10=mass_of(10),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top50=mass_of(50),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
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
    # scale S = 9.687;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.687477636514261 * (0.13625837906707788
        - 0.4133547526782006 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # -41.3%  mass > 80.4
        + 0.316124616307425 * max(0.0, Q.mass - 91.2) / 15.012010543283488   # +31.6%  mass > 91.2
        + 0.12402839117843395 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq) / 0.0022585004996878225   # +12.4%  mass_over_sum_pt_sq < 0.0079
        - 0.05182972780452952 * max(0.0, 0.057 - Q.girth) / 0.010962867445698715   # -5.2%  girth < 0.057
        - 0.045943992620663676 * max(0.0, 0.0058 - Q.lam1) / 0.0014084854463444797   # -4.6%  lam1 < 0.0058
        - 0.037955390953764316 * max(0.0, 0.0062 - Q.girth2_top20) / 0.0018856000053833206   # -3.8%  girth2_top20 < 0.0062
        - 0.010763128456982941 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4) / 2.0051455043452404   # -1.1%  sum_pt < 1000 and z_dr_0p2_0p4 < 0.21
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 10.23;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.233184627384672 * (0.1925109408002032
        + 0.21588053622292264 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922641781844428   # +21.6%  log_sum_pt > 6.9
        - 0.1595952600766834 * max(0.0, Q.z_top50_slots - 0.96) / 0.0329932881216197   # -16.0%  z_top50_slots > 0.96
        + 0.15109741141843108 * max(0.0, 760.0 - Q.sum_pt_top3) / 301.99369292279414   # +15.1%  sum_pt_top3 < 760
        - 0.14558190522839018 * max(0.0, Q.log_sum_pt - 6.8) / 0.14897665146085345   # -14.6%  log_sum_pt > 6.8
        + 0.09237273902833983 * max(0.0, Q.tau32 - 0.28) / 0.44588079859161467   # +9.2%  tau32 > 0.28
        - 0.05053303758168005 * max(0.0, 0.96 - Q.z_top20_slots) / 0.0772965475868311   # -5.1%  z_top20_slots < 0.96
        + 0.03962444329750335 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0) / 0.5531845075316328   # +4.0%  n_particles > 38 and dr_0 < 0.11
        - 0.03678212782702564 * max(0.0, 43.0 - Q.n_particles) / 4.9591344537815125   # -3.7%  n_particles < 43
        - 0.03539196507495791 * max(0.0, Q.log_sum_pt - 7.0) / 0.019162566822116114   # -3.5%  log_sum_pt > 7
        - 0.03058398972384055 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4) / 1043.2387116203197   # -3.1%  sum_pt_top2 < 820 and n_dr_0p2_0p4 < 7.7
        + 0.016637710119331513 * max(0.0, 0.0021 - Q.girth2_top20) / 0.0003405135188560513   # +1.7%  girth2_top20 < 0.0021
        + 0.015825030934191138 * max(0.0, 73.0 - Q.mass_top30) / 11.245865505809142   # +1.6%  mass_top30 < 73
        + 0.010093843466702501 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots) / 0.06376059493499212   # +1.0%  n_particles > 40 and z_top50_slots < 0.98
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 0.04558;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.04557978913638735 * (8.358968025497935
        + 1.0 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094) / 0.06874779658580293   # +100.0%  sum_pt_top50 > 1100 and C2 > 0.094
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 0.7303;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.7302582702999295 * (-0.3930116394054999
        + 0.4011319426136199 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq) / 0.002324840623611688   # +40.1%  mass_over_sum_pt_sq < 0.008
        + 0.3012208240157531 * max(0.0, 0.00052 - Q.lam2) / 0.00010524832436558056   # +30.1%  lam2 < 0.00052
        + 0.2066536298858357 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0) / 1207.2841784930574   # +20.7%  n_particles < 45 and sum_pt_top40 > 830
        - 0.09099360348479131 * max(0.0, 0.00011 - Q.girth2_top5) / 9.479148571920305e-06   # -9.1%  girth2_top5 < 0.00011
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 5.329;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.329128616867864 * (-0.09982870338616018
        - 0.27858284784534676 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -27.9%  mass < 80.4
        + 0.24980184541362366 * max(0.0, 110.0 - Q.mass) / 30.46284125721154   # +25.0%  mass < 110
        + 0.18990622304657293 * max(0.0, 7.7 - Q.D2) / 4.842271233295614   # +19.0%  D2 < 7.7
        - 0.11113295834445763 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2) / 48.945605668647396   # -11.1%  mass_top40 < 85 and D2 < 8.2
        - 0.06343928926807234 * max(0.0, 1100.0 - Q.sum_pt) / 79.17473814338236   # -6.3%  sum_pt < 1100
        + 0.05156515510354965 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr) / 2.328791048267427   # +5.2%  mass < 120 and max_dr < 0.4
        + 0.02134783307428004 * max(0.0, 0.61 - Q.tau32) / 0.03936517236825745   # +2.1%  tau32 < 0.61
        + 0.01811494379802782 * max(0.0, Q.C2 - 0.1) / 0.005712240555445153   # +1.8%  C2 > 0.1
        - 0.016108904106069116 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr) / 0.13691614331424748   # -1.6%  mass < 75 and max_dr < 0.33
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 2.609;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.6091442127596975 * (0.33382593255692117
        + 0.20380991636910742 * max(0.0, Q.mass_top30 - 49.0) / 34.087786140865035   # +20.4%  mass_top30 > 49
        - 0.1670687396319465 * max(0.0, Q.log_sum_pt - 6.9) / 0.05922641781844428   # -16.7%  log_sum_pt > 6.9
        + 0.15868904829867259 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2) / 0.2637214089151756   # +15.9%  n_particles < 66 and e2 < 0.038
        - 0.13297555685057577 * max(0.0, Q.mass_top50 - 150.0) / 2.4093916985776245   # -13.3%  mass_top50 > 150
        - 0.1293294364026319 * max(0.0, Q.z_top30_slots - 0.93) / 0.0370405214631614   # -12.9%  z_top30_slots > 0.93
        - 0.10131704110886809 * max(0.0, Q.mass_over_sum_pt_sq - 0.029) / 0.00021318610601866095   # -10.1%  mass_over_sum_pt_sq > 0.029
        + 0.05692539454510647 * max(0.0, Q.max_dr - 0.44) / 0.007501341602849848   # +5.7%  max_dr > 0.44
        + 0.04611648337342459 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32) / 0.0012520765422122746   # +4.6%  girth2_top30 < 0.02 and tau32 < 0.86
        + 0.0037683834196668127 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55) / 0.013109674387844352   # +0.4%  mass_top50 > 160 and z_dr_0p05_0p1 > 0.55
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 5.394;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.394164700100113 * (0.18260473210647774
        - 0.5206090253569051 * max(0.0, 100.0 - Q.mass) / 22.8313075376727   # -52.1%  mass < 100
        + 0.479390974643095 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # +47.9%  mass < 91.2
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 10.64;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.64068380813684 * (-0.05751510063028175
        - 0.2941023308639468 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -29.4%  mass < 91.2
        + 0.15784857103564634 * max(0.0, 120.0 - Q.mass) / 38.3474140172726   # +15.8%  mass < 120
        + 0.1268228326131516 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr) / 1.3230212367537362   # +12.7%  mass < 100 and max_dr < 0.4
        - 0.1169185254929242 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2) / 2.1122123278162865   # -11.7%  mass_top50 < 97 and D2 < 1.6
        - 0.11499723546428159 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr) / 0.7647807633658005   # -11.5%  mass < 91.2 and max_dr < 0.39
        + 0.11339229001768379 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2) / 2.6286960879275045   # +11.3%  mass < 100 and D2 < 1.6
        + 0.039669850459845814 * max(0.0, 9.4 - Q.n_dr_0p2_0p4) / 3.4318238655226923   # +4.0%  n_dr_0p2_0p4 < 9.4
        + 0.020810663550663865 * max(0.0, 0.026 - Q.e2) / 0.004888293392497407   # +2.1%  e2 < 0.026
        - 0.015437700501855812 * max(0.0, 0.99 - Q.z_top50_slots) / 0.004627258866618679   # -1.5%  z_top50_slots < 0.99
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 5.162;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.162155029111463 * (-0.006741370571737741
        + 0.2525710454996398 * max(0.0, Q.mass - 71.0) / 26.39293305129364   # +25.3%  mass > 71
        + 0.24844051305526216 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt) / 1.6919372610860863   # +24.8%  mass_over_sum_pt > 0.085 and sum_pt < 1100
        - 0.12246300773522431 * max(0.0, 19.0 - Q.n_dr_0p2_0p4) / 10.824880672268907   # -12.2%  n_dr_0p2_0p4 < 19
        + 0.11579557292989888 * max(0.0, 0.096 - Q.girth) / 0.03537010054135625   # +11.6%  girth < 0.096
        - 0.0739581814797775 * max(0.0, Q.mass - 120.0) / 8.088635560800247   # -7.4%  mass > 120
        - 0.07233984719236346 * max(0.0, 0.0015 - Q.lam2) / 0.0007350974527346728   # -7.2%  lam2 < 0.0015
        + 0.0487992524735328 * max(0.0, 1000.0 - Q.sum_pt) / 15.267230701401655   # +4.9%  sum_pt < 1000
        + 0.032846587426060635 * max(0.0, Q.n_particles - 58.0) / 1.5414470588235294   # +3.3%  n_particles > 58
        + 0.01991517969853422 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt) / 1.8292748227125788   # +2.0%  sum_pt_top40 < 930 and log_sum_pt < 6.9
        - 0.012870812509706318 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50) / 56.787290192932645   # -1.3%  sum_pt < 1000 and n_real_top50 < 44
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 1.043;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.043302651154608 * (-0.1485666693441867
        + 0.6583086394639923 * max(0.0, 0.0056 - Q.girth2_top40) / 0.0011760533370389831   # +65.8%  girth2_top40 < 0.0056
        + 0.12273758701992594 * max(0.0, Q.mass - 91.2) / 15.012010543283488   # +12.3%  mass > 91.2
        + 0.11698610468131931 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt) / 0.041094920256656876   # +11.7%  girth2_top40 < 0.0094 and sum_pt < 1000
        - 0.06780555738095143 * max(0.0, Q.mass_over_sum_pt_sq - 0.026) / 0.00044491646401611644   # -6.8%  mass_over_sum_pt_sq > 0.026
        + 0.03416211145381101 * max(0.0, 920.0 - Q.sum_pt_top40) / 8.425867954799108   # +3.4%  sum_pt_top40 < 920
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 7.671;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 7.671285453209044 * (0.4014972482495196
        - 0.21262268662080308 * max(0.0, 0.021 - Q.girth2_top5) / 0.015683551181696075   # -21.3%  girth2_top5 < 0.021
        - 0.16446134469425913 * max(0.0, 120.0 - Q.mass) / 38.3474140172726   # -16.4%  mass < 120
        + 0.12781161224341886 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +12.8%  mass < 80.4
        + 0.11029155380087216 * max(0.0, 0.088 - Q.z_dr_0p2_0p4) / 0.055663025808187716   # +11.0%  z_dr_0p2_0p4 < 0.088
        - 0.0881248222586204 * max(0.0, Q.z_dr_0_0p05 - 0.77) / 0.051214444474180845   # -8.8%  z_dr_0_0p05 > 0.77
        - 0.08423666583458687 * max(0.0, 11.0 - Q.n_dr_0p2_0p4) / 4.4875243697478995   # -8.4%  n_dr_0p2_0p4 < 11
        - 0.06340076480197712 * max(0.0, Q.mass - 150.0) / 3.0589016650800946   # -6.3%  mass > 150
        + 0.055201269669430925 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3) / 3.7474751948121683   # +5.5%  girth2_top5 < 0.021 and sum_pt_top3 < 720
        + 0.04870874029160495 * max(0.0, Q.mass_top50 - 140.0) / 3.766720270596031   # +4.9%  mass_top50 > 140
        + 0.03451873080478208 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21) / 2.8908628546502584   # +3.5%  mass < 120 and tau21 < 0.45
        - 0.010621808979644504 * max(0.0, 960.0 - Q.sum_pt_top50) / 10.072055465057117   # -1.1%  sum_pt_top50 < 960
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 0.65;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 0.6500187759470666 * (0.3707585210117461
        + 0.2960558900951422 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4) / 0.0009341839188910855   # +29.6%  z_dr_0p2_0p4 < 0.0048
        + 0.25922238783126256 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2) / 2.6286960879275045   # +25.9%  mass < 100 and D2 < 1.6
        + 0.24881251360806167 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow) / 2.307172689526749   # +24.9%  mass < 120 and planar_flow < 0.39
        - 0.19590920846553345 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2) / 2.6696994524847955   # -19.6%  mass < 80.4 and D2 < 3.1
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 1.833;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.832505077552807 * (0.04643916191143415
        + 0.4761816386999315 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +47.6%  mass < 80.4
        - 0.23463115857579564 * max(0.0, Q.C2 - 0.055) / 0.020377383385888306   # -23.5%  C2 > 0.055
        - 0.2126963640727839 * max(0.0, 63.0 - Q.mass_top50) / 5.791488367613624   # -21.3%  mass_top50 < 63
        + 0.058093179483553824 * max(0.0, Q.mass_top30 - 130.0) / 2.715715468744871   # +5.8%  mass_top30 > 130
        + 0.018397659167935166 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2) / 1.9047346802444715e-06   # +1.8%  LHA > 0.41 and lam2 < 0.0044
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 2.482;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.482315541225112 * (1.2488339812230473
        - 0.4535247495276327 * max(0.0, 7.0 - Q.log_sum_pt) / 0.0745557307339516   # -45.4%  log_sum_pt < 7
        + 0.15787071577200557 * max(0.0, 1000.0 - Q.sum_pt_top40) / 26.125662084345457   # +15.8%  sum_pt_top40 < 1000
        - 0.14822461278318716 * max(0.0, 1000.0 - Q.sum_pt) / 15.267230701401655   # -14.8%  sum_pt < 1000
        - 0.14135220067821075 * max(0.0, Q.mass - 140.0) / 4.527493735869592   # -14.1%  mass > 140
        + 0.04013663407353045 * max(0.0, 6.8 - Q.log_sum_pt) / 0.00436981537426272   # +4.0%  log_sum_pt < 6.8
        + 0.034158569097737276 * max(0.0, Q.mass_top10 - 67.0) / 4.763614996478938   # +3.4%  mass_top10 > 67
        + 0.024732518067696045 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +2.5%  mass > 172.8
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 3.884;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 3.8841636313908725 * (0.01583352449494451
        - 0.5851887555922864 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -58.5%  mass < 91.2
        + 0.3971248576286356 * max(0.0, 130.0 - Q.mass) / 46.460780997663385   # +39.7%  mass < 130
        + 0.017686386779077936 * max(0.0, 0.96 - Q.z_top50_slots) / 0.0006918108791340068   # +1.8%  z_top50_slots < 0.96
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 1;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.0 * (0.265
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.510664968487395, 2.8501269957983193, 0.4092661764705882, 0.40548324579831935, 0.9152868697478992, 1.0369552521008403, 0.7727060924369747, 0.42787888655462186, 2.017024369747899, 0.7956326680672269, 1.2765771008403362, 0.7070176470588235, 0.5198649159663865, 1.929614600840336, 0.42390514705882354, 0.25]
T = [3.9461716238839286, 2.5815427159926467, 4.98990809480042, 5.076312204569327, 4.561372357536764]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +45%, n4 -20%, n9 +10%, n5 -8%, n3 -8%, n12 +3% ...
            + 0.4514069691223204 * h[1] / H_AVG[1]
            - 0.202950121627292 * h[4] / H_AVG[4]
            + 0.10396103706923589 * h[9] / H_AVG[9]
            - 0.0821171877878376 * h[5] / H_AVG[5]
            - 0.07706518198755478 * h[3] / H_AVG[3]
            + 0.03087634074052267 * h[12] / H_AVG[12]
            + 0.030595559050836805 * h[6] / H_AVG[6]
            + 0.015972952411173653 * h[8] / H_AVG[8]
            - 0.005054650203226172 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -38%, n1 -21%, n9 +17%, n12 +7%, n11 -7%, n6 +7% ...
            - 0.37670974533272567 * h[4] / H_AVG[4]
            - 0.20700754180885186 * h[1] / H_AVG[1]
            + 0.17336276212486654 * h[9] / H_AVG[9]
            + 0.06922355526266426 * h[12] / H_AVG[12]
            - 0.06846852103965316 * h[11] / H_AVG[11]
            + 0.06547614210427409 * h[6] / H_AVG[6]
            + 0.019816938043247644 * h[2] / H_AVG[2]
            + 0.01220820619471814 * h[8] / H_AVG[8]
            - 0.0077265880889987445 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -35%, n5 +12%, n14 -12%, n11 +8%, n0 +8%, n4 +6% ...
            - 0.35369315225834874 * h[8] / H_AVG[8]
            + 0.12014044021080832 * h[5] / H_AVG[5]
            - 0.1168096818883786 * h[14] / H_AVG[14]
            + 0.08412814824758146 * h[11] / H_AVG[11]
            + 0.07675466543454744 * h[0] / H_AVG[0]
            + 0.06305323775475759 * h[4] / H_AVG[4]
            - 0.053593031978945646 * h[7] / H_AVG[7]
            - 0.0423244512922821 * h[12] / H_AVG[12]
            + 0.03555154056276463 * h[3] / H_AVG[3]
            - 0.034879329004288424 * h[9] / H_AVG[9]
            + 0.009678360775348616 * h[6] / H_AVG[6]
            - 0.009393960591948507 * h[15] / H_AVG[15]
        ),
        0.984375 + T[3] * (   # class Z: n8 -37%, n6 -15%, n5 +14%, n0 -14%, n7 +8%, n3 +3% ...
            - 0.37250670771126926 * h[8] / H_AVG[8]
            - 0.14746118774462313 * h[6] / H_AVG[6]
            + 0.14043792168212602 * h[5] / H_AVG[5]
            - 0.13832173896596248 * h[0] / H_AVG[0]
            + 0.07638719316575682 * h[7] / H_AVG[7]
            + 0.034946416392018426 * h[3] / H_AVG[3]
            + 0.032003111647322065 * h[12] / H_AVG[12]
            - 0.030233525833640352 * h[2] / H_AVG[2]
            + 0.02770219685728147 * h[15] / H_AVG[15]
        ),
        0.78125 + T[4] * (   # class t: n13 -38%, n10 +28%, n5 -11%, n8 +9%, n12 -4%, n4 +4% ...
            - 0.3833743652000154 * h[13] / H_AVG[13]
            + 0.27549397092376654 * h[10] / H_AVG[10]
            - 0.10656283598928948 * h[5] / H_AVG[5]
            + 0.09327589257007247 * h[8] / H_AVG[8]
            - 0.04273918641289606 * h[12] / H_AVG[12]
            + 0.037623827792565366 * h[4] / H_AVG[4]
            - 0.02638261632919484 * h[7] / H_AVG[7]
            - 0.020553024978348173 * h[15] / H_AVG[15]
            + 0.013994279803851747 * h[0] / H_AVG[0]
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
