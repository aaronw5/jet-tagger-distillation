"""JEDI-linear jet tagger, 64 particles, 3 features: the simplified formula closest to the start formula (within 0.1 point on validation jets), as if-statements with NORMALIZED weights (how much each one matters).

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
  neuron  8:  19.4%   (on for 58% of jets)
  neuron  4:  11.3%   (on for 67% of jets)
  neuron  1:  10.4%   (on for 93% of jets)
  neuron  5:   9.5%   (on for 76% of jets)
  neuron 13:   7.9%   (on for 93% of jets)
  neuron 10:   5.8%   (on for 79% of jets)
  neuron  0:   5.5%   (on for 49% of jets)
  neuron  6:   5.3%   (on for 70% of jets)
  neuron  9:   4.8%   (on for 47% of jets)
  neuron  7:   4.2%   (on for 28% of jets)
  neuron 12:   3.9%   (on for 64% of jets)
  neuron  3:   3.8%   (on for 58% of jets)
  neuron 14:   3.5%   (on for 52% of jets)
  neuron 11:   2.9%   (on for 78% of jets)
  neuron 15:   1.0%   (on for 44% of jets)
  neuron  2:   0.9%   (on for 88% of jets)

1. quantities():  physics quantities of the particles.
2. neuron_0 ... neuron_15: each neuron, its if-statements sorted by how much they matter.
3. logits():      each class score from the 16 neurons, with normalized weights.
4. classify():    the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.4% (the network: 81.1%); same class as the network for 92.4% of jets.

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
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.z_1st                  largest pT share
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
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
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi               pT-weighted mean Δφ
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
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
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_9=pt[9],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        z_1st=zs[0],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        sum_pt_top10=sum(pt[:10]),
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
        n_pt_above_10=sum(1 for x in pt if x > 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
        e2=e2,
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    # scale S = 12.44;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.443269218594189 * (0.07811451981988173
        - 0.24779348064980655 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # -24.8%  mass > 80.4
        + 0.18820404859944062 * max(0.0, Q.mass - 91.2) / 15.012010543283488   # +18.8%  mass > 91.2
        + 0.10820005569374122 * max(0.0, 0.00825 - Q.mass_over_sum_pt_sq) / 0.002497889466538232   # +10.8%  mass_over_sum_pt_sq < 0.00825
        - 0.1072464484512572 * max(0.0, Q.mass_over_sum_pt - 0.0769) / 0.020281100772296008   # -10.7%  mass_over_sum_pt > 0.0769
        + 0.09772172871299122 * max(0.0, Q.mass_over_sum_pt - 0.0836) / 0.016865156433872062   # +9.8%  mass_over_sum_pt > 0.0836
        - 0.042524808190805154 * max(0.0, 0.0567 - Q.girth) / 0.01084318927842762   # -4.3%  girth < 0.0567
        - 0.02749135063893993 * max(0.0, 0.00621 - Q.girth2_top40) / 0.0014373204923659702   # -2.7%  girth2_top40 < 0.00621
        - 0.02365968041196977 * max(0.0, 81.9 - Q.mass_top50) / 11.871119878711262   # -2.4%  mass_top50 < 81.9
        - 0.022875774522242334 * max(0.0, 0.00631 - Q.girth2_top20) / 0.0019496535682473918   # -2.3%  girth2_top20 < 0.00631
        - 0.022048529182986082 * max(0.0, 0.00592 - Q.lam1) / 0.0014671432326092322   # -2.2%  lam1 < 0.00592
        + 0.018573688771632575 * max(0.0, 7.02 - Q.log_sum_pt) / 0.09135075484893446   # +1.9%  log_sum_pt < 7.02
        - 0.012326962256682627 * max(0.0, Q.z_top30_slots - 0.918) / 0.04592446407405727   # -1.2%  z_top30_slots > 0.918
        + 0.011971119143616848 * max(0.0, 0.255 - Q.LHA) / 0.037427100088414864   # +1.2%  LHA < 0.255
        + 0.011708732098823347 * max(0.0, 10.2 - Q.n_dr_0p2_0p4) / 3.9483714285651565   # +1.2%  n_dr_0p2_0p4 < 10.2
        + 0.011584005128779237 * max(0.0, 0.00596 - Q.girth2_top40) * max(0.0, 0.00294 - Q.girth2_top3) / 3.440164545273888e-06   # +1.2%  girth2_top40 < 0.00596 and girth2_top3 < 0.00294
        - 0.007441374425589824 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.207 - Q.z_dr_0p2_0p4) / 2.4176246823492393   # -0.7%  sum_pt < 1010 and z_dr_0p2_0p4 < 0.207
        - 0.007275020361223411 * max(0.0, Q.mass - 80.4) * max(0.0, 6.83 - Q.n_dr_0p2_0p4) / 8.704330473601658   # -0.7%  mass > 80.4 and n_dr_0p2_0p4 < 6.83
        + 0.005767186827808255 * max(0.0, 47.8 - Q.n_particles) / 7.337695126006985   # +0.6%  n_particles < 47.8
        + 0.0056122341999861065 * max(0.0, 80.0 - Q.mass_top50) * max(0.0, 0.208 - Q.z_dr_0p05_0p1) / 1.7769603325249033   # +0.6%  mass_top50 < 80 and z_dr_0p05_0p1 < 0.208
        + 0.005556545430612771 * max(0.0, 6.99 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top40 - 961.0) / 1.539901797738621   # +0.6%  log_sum_pt < 6.99 and sum_pt_top40 > 961
        - 0.004168863080789193 * max(0.0, Q.z_dr_0_0p05 - 0.878) / 0.016060150355949817   # -0.4%  z_dr_0_0p05 > 0.878
        - 0.0036173181711979785 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0965) / 3.4624049118099407   # -0.4%  sum_pt < 1020 and z_dr_0p1_0p2 > 0.0965
        + 0.002945507928342645 * max(0.0, 0.00422 - Q.girth2_top40) / 0.0007186617281935523   # +0.3%  girth2_top40 < 0.00422
        - 0.002600564376915627 * max(0.0, Q.z_top30_slots - 0.916) * max(0.0, Q.C2 - 0.0591) / 0.000465604642622256   # -0.3%  z_top30_slots > 0.916 and C2 > 0.0591
        - 0.0010849727438193481 * max(0.0, 9.29 - Q.n_dr_0p2_0p4) * max(0.0, 0.987 - Q.z_top50_slots) / 0.006716720371234316   # -0.1%  n_dr_0p2_0p4 < 9.29 and z_top50_slots < 0.987
    )
    return max(0.0, z)


def neuron_1(Q):
    # scale S = 17.24;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 17.24241158617177 * (0.12701239551411456
        + 0.09020077196519029 * max(0.0, Q.log_sum_pt - 6.91) / 0.051842627853807825   # +9.0%  log_sum_pt > 6.91
        + 0.0728609132084818 * max(0.0, Q.log_sum_pt - 6.89) / 0.0671817034270044   # +7.3%  log_sum_pt > 6.89
        - 0.07185050800086225 * max(0.0, Q.z_top50_slots - 0.959) / 0.033941809085654646   # -7.2%  z_top50_slots > 0.959
        + 0.0656217132243956 * max(0.0, 0.411 - Q.LHA) / 0.1514694228118822   # +6.6%  LHA < 0.411
        + 0.06084298044945269 * max(0.0, 668.0 - Q.sum_pt_top2) / 303.20222862394957   # +6.1%  sum_pt_top2 < 668
        - 0.052158228935746474 * max(0.0, Q.log_sum_pt - 6.81) / 0.13943157378541313   # -5.2%  log_sum_pt > 6.81
        - 0.05123523806445924 * max(0.0, Q.sum_pt_top50 - 960.0) / 85.76884101193211   # -5.1%  sum_pt_top50 > 960
        + 0.04846616197737178 * max(0.0, Q.tau32 - 0.328) / 0.399843786036322   # +4.8%  tau32 > 0.328
        - 0.044526102726679506 * max(0.0, 0.952 - Q.z_top20_slots) / 0.07175115790108161   # -4.5%  z_top20_slots < 0.952
        - 0.04273055874486249 * max(0.0, 31.3 - Q.n_pt_above_10) / 11.676353109128534   # -4.3%  n_pt_above_10 < 31.3
        + 0.04264003265233972 * max(0.0, 80.7 - Q.mass_top30) / 14.822923246763002   # +4.3%  mass_top30 < 80.7
        - 0.042590667027574504 * max(0.0, 117.0 - Q.mass_top50) / 36.90280455372033   # -4.3%  mass_top50 < 117
        - 0.03755010137003784 * max(0.0, 0.435 - Q.max_dr) / 0.08967511120840235   # -3.8%  max_dr < 0.435
        + 0.030538397983236636 * max(0.0, 1080.0 - Q.sum_pt_top30) / 103.44904267372965   # +3.1%  sum_pt_top30 < 1080
        - 0.0278790873393492 * max(0.0, 80.8 - Q.mass_top30) * max(0.0, 68.2 - Q.mass_top5) / 900.1923193855605   # -2.8%  mass_top30 < 80.8 and mass_top5 < 68.2
        - 0.02488558075470165 * max(0.0, Q.log_sum_pt - 6.99) / 0.021033697349680438   # -2.5%  log_sum_pt > 6.99
        + 0.023008284731080856 * max(0.0, 0.433 - Q.max_dr) * max(0.0, 0.191 - Q.z_dr_0p2_0p4) / 0.013774941500872485   # +2.3%  max_dr < 0.433 and z_dr_0p2_0p4 < 0.191
        - 0.0160083381284558 * max(0.0, 13.3 - Q.n_dr_0p2_0p4) / 6.16121327726432   # -1.6%  n_dr_0p2_0p4 < 13.3
        + 0.014975259671158401 * max(0.0, Q.n_particles - 37.8) * max(0.0, 0.143 - Q.dr_0) / 0.8693925618178872   # +1.5%  n_particles > 37.8 and dr_0 < 0.143
        - 0.014685103486264292 * max(0.0, Q.z_top30_slots - 0.94) * max(0.0, 92.7 - Q.mass_top10) / 1.49826389642423   # -1.5%  z_top30_slots > 0.94 and mass_top10 < 92.7
        - 0.014276013483760347 * max(0.0, 35.2 - Q.pt_9) / 9.503972984429877   # -1.4%  pt_9 < 35.2
        + 0.011828502104659052 * max(0.0, 0.0742 - Q.mass_over_sum_pt) / 0.009398705149144258   # +1.2%  mass_over_sum_pt < 0.0742
        - 0.01159285132248494 * max(0.0, 41.1 - Q.mass_top20) / 4.147068754348155   # -1.2%  mass_top20 < 41.1
        - 0.011490440364454986 * max(0.0, 0.000842 - Q.lam2) / 0.00027555341038983924   # -1.1%  lam2 < 0.000842
        + 0.009888665860537639 * max(0.0, Q.n_particles - 37.7) / 11.000286890678414   # +1.0%  n_particles > 37.7
        + 0.009558049567246249 * max(0.0, Q.n_particles - 38.2) * max(0.0, 0.986 - Q.z_top50_slots) / 0.09637650561373731   # +1.0%  n_particles > 38.2 and z_top50_slots < 0.986
        + 0.009473862081370934 * max(0.0, 0.000845 - Q.girth2_top3) / 0.00021325356307783758   # +0.9%  girth2_top3 < 0.000845
        - 0.00896869377600685 * max(0.0, 701.0 - Q.sum_pt_top2) * max(0.0, 6.95 - Q.n_dr_0p2_0p4) / 636.3864587499883   # -0.9%  sum_pt_top2 < 701 and n_dr_0p2_0p4 < 6.95
        + 0.007484104406308062 * max(0.0, 0.00115 - Q.girth2_top20) / 0.00011948519308096789   # +0.7%  girth2_top20 < 0.00115
        - 0.00639585843649138 * max(0.0, 0.000838 - Q.girth2_top3) * max(0.0, 10.4 - Q.n_dr_0p05_0p1) / 0.0010811767020477787   # -0.6%  girth2_top3 < 0.000838 and n_dr_0p05_0p1 < 10.4
        + 0.005506882126119305 * max(0.0, 40.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 26.6) / 32.40680142494257   # +0.6%  mass_top20 < 40.5 and n_real_top40 > 26.6
        + 0.004826623749584983 * max(0.0, Q.z_top30_slots - 0.943) * max(0.0, Q.m012 - 13.7) / 0.22311697925451993   # +0.5%  z_top30_slots > 0.943 and m012 > 13.7
        + 0.004720642607888356 * max(0.0, 0.00292 - Q.girth2_top15) / 0.0006782938566369192   # +0.5%  girth2_top15 < 0.00292
        - 0.00454631431502027 * max(0.0, 7.22 - Q.n_dr_0p1_0p2) * max(0.0, 0.0781 - Q.dr_6) / 0.04930152366017866   # -0.5%  n_dr_0p1_0p2 < 7.22 and dr_6 < 0.0781
        + 0.00338647322145615 * max(0.0, 1.18 - Q.D2) / 0.08969426284162205   # +0.3%  D2 < 1.18
        - 0.0008019921349094046 * max(0.0, Q.log_sum_pt - 7.13) / 0.005934883467373627   # -0.1%  log_sum_pt > 7.13
    )
    return max(0.0, z)


def neuron_2(Q):
    # scale S = 1.809;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.8094983290924194 * (0.061895608412180884
        + 0.19768254638154373 * max(0.0, Q.z_top30_slots - 0.919) / 0.04516492896062349   # +19.8%  z_top30_slots > 0.919
        - 0.17658668231880117 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -17.7%  mass < 91.2
        - 0.1478475355400876 * max(0.0, Q.sum_pt - 1090.0) / 24.771284122242648   # -14.8%  sum_pt > 1090
        + 0.14453497979765106 * max(0.0, Q.sum_pt - 1010.0) / 52.30716088785123   # +14.5%  sum_pt > 1010
        + 0.11866678562319612 * max(0.0, 0.0732 - Q.mass_over_sum_pt) / 0.00906022575122117   # +11.9%  mass_over_sum_pt < 0.0732
        - 0.07938675948869615 * max(0.0, 0.000964 - Q.lam2) / 0.00035208384472268985   # -7.9%  lam2 < 0.000964
        + 0.07873387705606581 * max(0.0, Q.log_sum_pt - 7.05) / 0.012176822134693939   # +7.9%  log_sum_pt > 7.05
        - 0.024398473458818758 * max(0.0, Q.mass_top40 - 158.0) / 0.9660612025413192   # -2.4%  mass_top40 > 158
        + 0.023705734675987765 * max(0.0, Q.sum_pt_top50 - 1110.0) * max(0.0, Q.C2 - 0.0963) / 0.056890566692451044   # +2.4%  sum_pt_top50 > 1110 and C2 > 0.0963
        + 0.008456625659151786 * max(0.0, Q.sum_pt_top50 - 1030.0) * max(0.0, Q.mean_phi - 3.25e-05) / 0.0038937022900751234   # +0.8%  sum_pt_top50 > 1030 and mean_phi > 3.25e-05
    )
    return max(0.0, z)


def neuron_3(Q):
    # scale S = 1.412;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 1.411512512621099 * (-0.13106507972534048
        + 0.15517693611523628 * max(0.0, 4.94 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979) / 0.02066360254687367   # +15.5%  n_dr_0p2_0p4 < 4.94 and z_top50_slots > 0.979
        + 0.14748548141053822 * max(0.0, 0.000624 - Q.lam2) / 0.00015420563143771942   # +14.7%  lam2 < 0.000624
        + 0.126183267943992 * max(0.0, 0.00728 - Q.mass_over_sum_pt_sq) / 0.0018807736175962572   # +12.6%  mass_over_sum_pt_sq < 0.00728
        - 0.12460712507566511 * max(0.0, 0.424 - Q.tau21) * max(0.0, 0.0159 - Q.lam1) / 0.0006897432008080141   # -12.5%  tau21 < 0.424 and lam1 < 0.0159
        + 0.11605997714021772 * max(0.0, 46.1 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 841.0) / 1250.5351904422596   # +11.6%  n_particles < 46.1 and sum_pt_top40 > 841
        + 0.05799723363568237 * max(0.0, 14.4 - Q.n_dr_0p1_0p2) / 4.732012773073696   # +5.8%  n_dr_0p1_0p2 < 14.4
        - 0.05235899102214124 * max(0.0, 0.000115 - Q.girth2_top5) / 1.0222043011890478e-05   # -5.2%  girth2_top5 < 0.000115
        + 0.04824901048246682 * max(0.0, 0.00749 - Q.mass_over_sum_pt_sq) * max(0.0, Q.girth2_top15 - 0.00486) / 1.653011699456031e-07   # +4.8%  mass_over_sum_pt_sq < 0.00749 and girth2_top15 > 0.00486
        - 0.04667696041969219 * max(0.0, 45.4 - Q.n_particles) * max(0.0, Q.mass_top30 - 73.0) / 42.50652495710665   # -4.7%  n_particles < 45.4 and mass_top30 > 73
        - 0.04035995463448942 * max(0.0, 5.19 - Q.n_dr_0p2_0p4) * max(0.0, 0.042 - Q.dr_0) / 0.010608674297095295   # -4.0%  n_dr_0p2_0p4 < 5.19 and dr_0 < 0.042
        + 0.03438813667337082 * max(0.0, 0.000347 - Q.lam2) / 4.113498745778593e-05   # +3.4%  lam2 < 0.000347
        + 0.023902132786612445 * max(0.0, 0.00143 - Q.z_dr_0p2_0p4) / 0.00014479896783963297   # +2.4%  z_dr_0p2_0p4 < 0.00143
        - 0.019919015729526496 * max(0.0, 5.16 - Q.n_dr_0p2_0p4) * max(0.0, 991.0 - Q.sum_pt_top30) / 19.661496462463735   # -2.0%  n_dr_0p2_0p4 < 5.16 and sum_pt_top30 < 991
        - 0.006635776930369106 * max(0.0, 0.293 - Q.max_dr) * max(0.0, Q.tau21 - 0.43) / 0.0006042891721405432   # -0.7%  max_dr < 0.293 and tau21 > 0.43
    )
    return max(0.0, z)


def neuron_4(Q):
    # scale S = 12.56;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.555546360268258 * (0.1489381621748913
        - 0.10888551988354774 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -10.9%  mass < 80.4
        - 0.09435525435730072 * max(0.0, Q.LHA - 0.114) / 0.15034032619517673   # -9.4%  LHA > 0.114
        - 0.09054688041002834 * max(0.0, 0.121 - Q.girth) / 0.05600322929880672   # -9.1%  girth < 0.121
        + 0.07118409763279393 * max(0.0, 101.0 - Q.mass) / 23.58193239953578   # +7.1%  mass < 101
        + 0.060895125184063134 * max(0.0, 6.92 - Q.D2) / 4.132819283042399   # +6.1%  D2 < 6.92
        + 0.05652821059612791 * max(0.0, 0.0159 - Q.girth2_top15) / 0.009816633040147876   # +5.7%  girth2_top15 < 0.0159
        + 0.04113680616059072 * max(0.0, Q.width - 0.00971) / 0.0031686814531451876   # +4.1%  width > 0.00971
        + 0.03909402947006457 * max(0.0, 120.0 - Q.mass) / 38.3474140172726   # +3.9%  mass < 120
        + 0.03676458876485558 * max(0.0, 0.0129 - Q.girth2_top40) / 0.006271732318659536   # +3.7%  girth2_top40 < 0.0129
        - 0.035481825577036365 * max(0.0, 84.3 - Q.mass_top40) * max(0.0, 6.61 - Q.D2) / 30.937062915238343   # -3.5%  mass_top40 < 84.3 and D2 < 6.61
        - 0.03498265359591937 * max(0.0, Q.lam1 - 0.00757) / 0.0027799134748656374   # -3.5%  lam1 > 0.00757
        - 0.0335684859865541 * max(0.0, 0.069 - Q.z_dr_0p2_0p4) / 0.04091948369399954   # -3.4%  z_dr_0p2_0p4 < 0.069
        - 0.029748518018780562 * max(0.0, 95.9 - Q.mass_top40) / 21.971111596122057   # -3.0%  mass_top40 < 95.9
        - 0.02946037667128962 * max(0.0, 0.00725 - Q.girth2_top15) / 0.0028022054930859266   # -2.9%  girth2_top15 < 0.00725
        + 0.028512239483600477 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.403 - Q.max_dr) / 2.5389130827760065   # +2.9%  mass < 122 and max_dr < 0.403
        - 0.026580166549311363 * max(0.0, 0.0609 - Q.girth) / 0.012593528806548316   # -2.7%  girth < 0.0609
        + 0.0261752392944826 * max(0.0, 15.4 - Q.n_dr_0p2_0p4) / 7.806281008384601   # +2.6%  n_dr_0p2_0p4 < 15.4
        - 0.023829178103231513 * max(0.0, 1070.0 - Q.sum_pt) / 55.20080265723477   # -2.4%  sum_pt < 1070
        - 0.01895174087606728 * max(0.0, Q.n_particles - 28.2) / 18.303804705942593   # -1.9%  n_particles > 28.2
        - 0.015113882747314204 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.388 - Q.max_dr) / 0.3546973000328263   # -1.5%  mass < 74.4 and max_dr < 0.388
        + 0.014853192150826902 * max(0.0, Q.e2 - 0.0369) / 0.00457083192763921   # +1.5%  e2 > 0.0369
        - 0.01388453675620926 * max(0.0, 14.6 - Q.n_dr_0p2_0p4) * max(0.0, 0.436 - Q.max_dr) / 0.8341049996815025   # -1.4%  n_dr_0p2_0p4 < 14.6 and max_dr < 0.436
        + 0.012727132109400128 * max(0.0, 0.194 - Q.z_dr_0p2_0p4) / 0.14660192406681838   # +1.3%  z_dr_0p2_0p4 < 0.194
        - 0.01070676283135236 * max(0.0, 0.465 - Q.tau21) / 0.10841069120761418   # -1.1%  tau21 < 0.465
        + 0.007493494387360419 * max(0.0, 0.425 - Q.planar_flow) / 0.09620134578825534   # +0.7%  planar_flow < 0.425
        - 0.006645556853860895 * max(0.0, Q.lam2 - 0.00239) / 0.00048230402987542735   # -0.7%  lam2 > 0.00239
        + 0.005715528596424166 * max(0.0, 0.577 - Q.tau32) / 0.032180082630422685   # +0.6%  tau32 < 0.577
        + 0.0049832803578649006 * max(0.0, Q.C2 - 0.11) / 0.004062844646713438   # +0.5%  C2 > 0.11
        + 0.004791997091916304 * max(0.0, Q.mass - 143.0) / 4.065279840934177   # +0.5%  mass > 143
        - 0.004377753133005987 * max(0.0, Q.mass_top20 - 106.0) / 3.4569234223437912   # -0.4%  mass_top20 > 106
        - 0.004106409395373612 * max(0.0, Q.z_top5_slots - 0.569) / 0.06635548717870596   # -0.4%  z_top5_slots > 0.569
        + 0.0035355537451779574 * max(0.0, 0.548 - Q.z_top5_slots) / 0.05597832150920796   # +0.4%  z_top5_slots < 0.548
        - 0.0026870824181204627 * max(0.0, 0.303 - Q.planar_flow) * max(0.0, Q.n_dr_0p05_0p1 - 6.21) / 0.3244018064862808   # -0.3%  planar_flow < 0.303 and n_dr_0p05_0p1 > 6.21
        - 0.0016969008101466341 * max(0.0, 15.2 - Q.n_dr_0p2_0p4) * max(0.0, Q.n_dr_0p1_0p2 - 11.9) / 17.46353835292855   # -0.2%  n_dr_0p2_0p4 < 15.2 and n_dr_0p1_0p2 > 11.9
    )
    return max(0.0, z)


def neuron_5(Q):
    # scale S = 10.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.98740659106522 * (-0.09647408523699985
        + 0.16114403245980555 * max(0.0, Q.sum_pt_top40 - 905.0) / 123.81503526990875   # +16.1%  sum_pt_top40 > 905
        - 0.15523430777352518 * max(0.0, Q.log_sum_pt - 6.91) / 0.051842627853807825   # -15.5%  log_sum_pt > 6.91
        + 0.11797646672713762 * max(0.0, Q.log_sum_pt - 6.85) / 0.1020673549691605   # +11.8%  log_sum_pt > 6.85
        - 0.07528153281914499 * max(0.0, Q.sum_pt_top40 - 1010.0) / 42.417887686285454   # -7.5%  sum_pt_top40 > 1010
        + 0.06204352240566148 * max(0.0, 1010.0 - Q.sum_pt_top40) / 30.43291995593159   # +6.2%  sum_pt_top40 < 1010
        + 0.05802638353329199 * max(0.0, Q.mass - 61.6) / 33.38007690520266   # +5.8%  mass > 61.6
        - 0.053223610636064185 * max(0.0, 1020.0 - Q.sum_pt_top50) / 27.073585662175027   # -5.3%  sum_pt_top50 < 1020
        + 0.047672325540503616 * max(0.0, Q.mass_top30 - 68.9) / 19.99218411660825   # +4.8%  mass_top30 > 68.9
        + 0.044765729276283195 * max(0.0, Q.sum_pt_top50 - 1020.0) / 42.770371209050026   # +4.5%  sum_pt_top50 > 1020
        - 0.026096670424807483 * max(0.0, Q.mass_top50 - 90.5) / 13.987059933189585   # -2.6%  mass_top50 > 90.5
        - 0.02594797736344058 * max(0.0, Q.mass_top50 - 156.0) / 1.717475768119748   # -2.6%  mass_top50 > 156
        + 0.024594144363462817 * max(0.0, 61.7 - Q.n_particles) * max(0.0, 0.0374 - Q.e2) / 0.21277627077222117   # +2.5%  n_particles < 61.7 and e2 < 0.0374
        - 0.01753474031147019 * max(0.0, Q.z_top30_slots - 0.933) / 0.034902413273707356   # -1.8%  z_top30_slots > 0.933
        - 0.015910868300171122 * max(0.0, Q.sum_pt_top50 - 1060.0) / 28.705940760405724   # -1.6%  sum_pt_top50 > 1060
        + 0.014386848379507638 * max(0.0, Q.max_dr - 0.435) / 0.0077487329759636375   # +1.4%  max_dr > 0.435
        - 0.013972372765903154 * max(0.0, Q.mass_top30 - 107.0) / 6.317701260119526   # -1.4%  mass_top30 > 107
        - 0.011807709339220885 * max(0.0, Q.mass_over_sum_pt - 0.0895) / 0.014511868391402453   # -1.2%  mass_over_sum_pt > 0.0895
        + 0.011563642661204108 * max(0.0, Q.sum_pt_top10 - 840.0) / 37.92669957983193   # +1.2%  sum_pt_top10 > 840
        + 0.010550074459343296 * max(0.0, 62.5 - Q.mass_top10) / 21.387077057346374   # +1.1%  mass_top10 < 62.5
        + 0.009823547964037092 * max(0.0, 0.0186 - Q.girth2_top30) * max(0.0, 0.866 - Q.tau32) / 0.0011643507621111808   # +1.0%  girth2_top30 < 0.0186 and tau32 < 0.866
        - 0.008196265963927151 * max(0.0, Q.mass_over_sum_pt_sq - 0.0292) / 0.0002028281681850827   # -0.8%  mass_over_sum_pt_sq > 0.0292
        - 0.007749936948932964 * max(0.0, 64.0 - Q.n_particles) * max(0.0, Q.mass_top5 - 11.1) / 321.3272011813054   # -0.8%  n_particles < 64 and mass_top5 > 11.1
        - 0.0071007105812175035 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # -0.7%  mass > 172.8
        + 0.005621026180051699 * max(0.0, Q.log_sum_pt - 7.14) / 0.005417587728004402   # +0.6%  log_sum_pt > 7.14
        + 0.0032881042259637677 * max(0.0, Q.sum_pt_top3 - 318.0) * max(0.0, Q.n_dr_0p1_0p2 - 7.87) / 473.4959114608611   # +0.3%  sum_pt_top3 > 318 and n_dr_0p1_0p2 > 7.87
        - 0.0032377365357119953 * max(0.0, Q.max_dr - 0.278) * max(0.0, 0.582 - Q.tau21) / 0.01363001063318569   # -0.3%  max_dr > 0.278 and tau21 < 0.582
        - 0.0027172045980084093 * max(0.0, 0.864 - Q.z_top30_slots) / 0.0038374076747339737   # -0.3%  z_top30_slots < 0.864
        - 0.001902230445822706 * max(0.0, Q.sum_pt_top10 - 994.0) / 6.593242693425026   # -0.2%  sum_pt_top10 > 994
        + 0.001464671567402987 * max(0.0, Q.mass_top50 - 157.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.589) / 0.010382543247373812   # +0.1%  mass_top50 > 157 and z_dr_0p05_0p1 > 0.589
        - 0.0011656054489747822 * max(0.0, Q.sum_pt_top3 - 794.0) / 4.13128419117647   # -0.1%  sum_pt_top3 > 794
    )
    return max(0.0, z)


def neuron_6(Q):
    # scale S = 10.61;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.607152070101732 * (0.05722553952167279
        - 0.13629459622118073 * max(0.0, 120.0 - Q.mass) / 38.3474140172726   # -13.6%  mass < 120
        - 0.1218319382011242 * max(0.0, 101.0 - Q.mass) / 23.58193239953578   # -12.2%  mass < 101
        + 0.11453827245206409 * max(0.0, 172.8 - Q.mass) / 83.7879223272957   # +11.5%  mass < 172.8
        - 0.1117136031947643 * max(0.0, Q.mass_over_sum_pt - 0.0513) / 0.03885125171756943   # -11.2%  mass_over_sum_pt > 0.0513
        + 0.08416177668506157 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # +8.4%  mass < 91.2
        - 0.05813322501331307 * max(0.0, 0.00959 - Q.width) / 0.00348377377424663   # -5.8%  width < 0.00959
        + 0.037715772341000174 * max(0.0, 86.1 - Q.mass) / 13.515436914267756   # +3.8%  mass < 86.1
        - 0.03623709725310209 * max(0.0, 16.1 - Q.n_dr_0p2_0p4) / 8.374126386551625   # -3.6%  n_dr_0p2_0p4 < 16.1
        + 0.03051124652153157 * max(0.0, 0.00743 - Q.lam1) / 0.002362317019724437   # +3.1%  lam1 < 0.00743
        + 0.028637480642818 * max(0.0, Q.lam1 - 0.00815) / 0.0026186388972669417   # +2.9%  lam1 > 0.00815
        + 0.028479880484945846 * max(0.0, 0.0481 - Q.e2) / 0.019241428231983645   # +2.8%  e2 < 0.0481
        + 0.028010011223467576 * max(0.0, 85.3 - Q.mass_top10) / 39.88006020571501   # +2.8%  mass_top10 < 85.3
        + 0.027298254992502085 * max(0.0, 71.7 - Q.mass_top50) / 8.179568981748119   # +2.7%  mass_top50 < 71.7
        + 0.02704254105837519 * max(0.0, Q.mass_over_sum_pt - 0.0403) * max(0.0, 0.945 - Q.z_dr_0_0p05) / 0.03204964752716821   # +2.7%  mass_over_sum_pt > 0.0403 and z_dr_0_0p05 < 0.945
        + 0.02357267169935725 * max(0.0, Q.mass_over_sum_pt - 0.0481) * max(0.0, 0.00253 - Q.lam2) / 4.259606702106742e-05   # +2.4%  mass_over_sum_pt > 0.0481 and lam2 < 0.00253
        + 0.022658611942616418 * max(0.0, Q.mass_over_sum_pt - 0.0528) * max(0.0, 19.0 - Q.n_dr_0p2_0p4) / 0.251932224919031   # +2.3%  mass_over_sum_pt > 0.0528 and n_dr_0p2_0p4 < 19
        - 0.016172597701039893 * max(0.0, Q.mass_over_sum_pt - 0.0497) * max(0.0, 17.3 - Q.n_dr_0p1_0p2) / 0.15454522809325025   # -1.6%  mass_over_sum_pt > 0.0497 and n_dr_0p1_0p2 < 17.3
        + 0.014865810240916598 * max(0.0, Q.mass_over_sum_pt - 0.0519) * max(0.0, 0.282 - Q.z_dr_0p1_0p2) / 0.00351973013104192   # +1.5%  mass_over_sum_pt > 0.0519 and z_dr_0p1_0p2 < 0.282
        + 0.013558203697798376 * max(0.0, Q.girth2_top20 - 0.00238) * max(0.0, 0.425 - Q.max_dr) / 0.0004028401356301484   # +1.4%  girth2_top20 > 0.00238 and max_dr < 0.425
        + 0.010279245111930879 * max(0.0, Q.girth2_top3 - 0.00959) * max(0.0, 5.09 - Q.D2) / 0.005371109165916292   # +1.0%  girth2_top3 > 0.00959 and D2 < 5.09
        + 0.007582218450517837 * max(0.0, Q.sum_pt_top10 - 625.0) / 164.8068527323234   # +0.8%  sum_pt_top10 > 625
        - 0.007170960494093678 * max(0.0, Q.e2 - 0.0555) / 0.0011268661992524963   # -0.7%  e2 > 0.0555
        - 0.006134522745578864 * max(0.0, Q.m012 - 20.4) / 5.758390764588762   # -0.6%  m012 > 20.4
        + 0.0036678712755384285 * max(0.0, 6.83 - Q.log_sum_pt) / 0.005948878959203994   # +0.4%  log_sum_pt < 6.83
        + 0.002224624255208736 * max(0.0, Q.lam1 - 0.0237) / 0.0002507643759174906   # +0.2%  lam1 > 0.0237
        - 0.0015069661001523812 * max(0.0, Q.e2 - 0.0547) * max(0.0, 0.366 - Q.max_dr) / 1.0379622460262638e-05   # -0.2%  e2 > 0.0547 and max_dr < 0.366
    )
    return max(0.0, z)


def neuron_7(Q):
    # scale S = 18.99;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 18.987509919818326 * (-0.012534555663435682
        - 0.20992383702656162 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -21.0%  mass < 91.2
        + 0.10829985152993644 * max(0.0, 101.0 - Q.mass) / 23.58193239953578   # +10.8%  mass < 101
        + 0.07725079115739288 * max(0.0, 0.0127 - Q.girth2_top40) / 0.006111667347561712   # +7.7%  girth2_top40 < 0.0127
        - 0.07429885665153606 * max(0.0, 0.00798 - Q.girth2_top20) / 0.0030601958301566174   # -7.4%  girth2_top20 < 0.00798
        + 0.06094139187471186 * max(0.0, 0.00943 - Q.width) / 0.0033637362870599635   # +6.1%  width < 0.00943
        - 0.05768462563482884 * max(0.0, 98.1 - Q.mass_top50) * max(0.0, 1.61 - Q.D2) / 2.345369168013531   # -5.8%  mass_top50 < 98.1 and D2 < 1.61
        + 0.057028296128628406 * max(0.0, 101.0 - Q.mass) * max(0.0, 0.391 - Q.max_dr) / 1.196492086688032   # +5.7%  mass < 101 and max_dr < 0.391
        - 0.054900676098711826 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.392 - Q.max_dr) / 0.7897175242644119   # -5.5%  mass < 91.2 and max_dr < 0.392
        + 0.05440717040862898 * max(0.0, 101.0 - Q.mass) * max(0.0, 1.61 - Q.D2) / 2.8537477564726212   # +5.4%  mass < 101 and D2 < 1.61
        - 0.041856844876340824 * max(0.0, 0.00801 - Q.girth2_top40) / 0.0024992366581818182   # -4.2%  girth2_top40 < 0.00801
        + 0.028715752256207674 * max(0.0, 0.00639 - Q.girth2_top20) / 0.001997218427911314   # +2.9%  girth2_top20 < 0.00639
        - 0.028093647985945193 * max(0.0, 0.093 - Q.z_dr_0p2_0p4) / 0.05966760848065084   # -2.8%  z_dr_0p2_0p4 < 0.093
        - 0.02351200991536487 * max(0.0, 0.0858 - Q.girth) / 0.02738862095109554   # -2.4%  girth < 0.0858
        + 0.020587748155714344 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +2.1%  mass < 80.4
        + 0.017359298104254482 * max(0.0, 66.0 - Q.mass_top20) / 12.532693724548116   # +1.7%  mass_top20 < 66
        + 0.013671023676746647 * max(0.0, 0.0275 - Q.e2) / 0.005522951014389322   # +1.4%  e2 < 0.0275
        + 0.011727837370551466 * max(0.0, 1.82 - Q.D2) * max(0.0, 9.16 - Q.n_dr_0p2_0p4) / 1.6373707971423677   # +1.2%  D2 < 1.82 and n_dr_0p2_0p4 < 9.16
        + 0.011663320089951582 * max(0.0, 13.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.504 - Q.z_1st) / 1.5818386136140858   # +1.2%  n_dr_0p2_0p4 < 13 and z_1st < 0.504
        - 0.00929190859922644 * max(0.0, 0.991 - Q.z_top50_slots) / 0.004873762616073404   # -0.9%  z_top50_slots < 0.991
        + 0.008346315809902964 * max(0.0, 0.00474 - Q.z_dr_0p2_0p4) / 0.000916044822164561   # +0.8%  z_dr_0p2_0p4 < 0.00474
        - 0.0074964001562726175 * max(0.0, 1.79 - Q.D2) * max(0.0, 0.00789 - Q.girth2_top50) / 0.00028241661176617853   # -0.7%  D2 < 1.79 and girth2_top50 < 0.00789
        - 0.005722878144329349 * max(0.0, 0.0562 - Q.C2) / 0.009969101425262843   # -0.6%  C2 < 0.0562
        + 0.0037499830003971486 * max(0.0, 12.9 - Q.n_dr_0p2_0p4) * max(0.0, 0.549 - Q.planar_flow) / 1.1769064366808422   # +0.4%  n_dr_0p2_0p4 < 12.9 and planar_flow < 0.549
        - 0.0031155255067490326 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.34 - Q.pt_dispersion) / 0.49296726220703563   # -0.3%  mass < 91.2 and pt_dispersion < 0.34
        - 0.0030127980028996962 * max(0.0, 1.99 - Q.n_dr_0p2_0p4) / 0.19863031932800984   # -0.3%  n_dr_0p2_0p4 < 1.99
        - 0.002807621422934295 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.393) / 0.419761729283932   # -0.3%  mass < 91.2 and z_dr_0p05_0p1 > 0.393
        + 0.0019262987788801176 * max(0.0, 1.77 - Q.D2) * max(0.0, 0.00613 - Q.girth2_top50) / 5.5501695254203584e-05   # +0.2%  D2 < 1.77 and girth2_top50 < 0.00613
        + 0.0017618962944223068 * max(0.0, 76.1 - Q.mass_top30) * max(0.0, 1.61 - Q.D2) / 0.1593048731811176   # +0.2%  mass_top30 < 76.1 and D2 < 1.61
        - 0.0008453953419721095 * max(0.0, 0.196 - Q.max_dr) / 0.0018178881587614536   # -0.1%  max_dr < 0.196
    )
    return max(0.0, z)


def neuron_8(Q):
    # scale S = 12.79;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 12.794583612710344 * (0.052522225055641865
        - 0.10305747430482155 * max(0.0, Q.mass - 101.0) / 12.323153943063431   # -10.3%  mass > 101
        + 0.0907622993198697 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # +9.1%  mass > 80.4
        - 0.08770709654612267 * max(0.0, 0.0988 - Q.mass_over_sum_pt) / 0.024029459961186835   # -8.8%  mass_over_sum_pt < 0.0988
        + 0.06923608556126724 * max(0.0, 0.0968 - Q.girth) / 0.03601003600530086   # +6.9%  girth < 0.0968
        + 0.0645621510431424 * max(0.0, Q.mass_over_sum_pt - 0.0754) * max(0.0, 1110.0 - Q.sum_pt) / 2.3268896894025914   # +6.5%  mass_over_sum_pt > 0.0754 and sum_pt < 1110
        - 0.06009566467489601 * max(0.0, Q.mass_top50 - 82.6) / 17.435351620053524   # -6.0%  mass_top50 > 82.6
        - 0.05771689337491678 * max(0.0, 20.5 - Q.n_dr_0p2_0p4) / 12.14578319327731   # -5.8%  n_dr_0p2_0p4 < 20.5
        + 0.055966878218941335 * max(0.0, Q.mass - 91.2) / 15.012010543283488   # +5.6%  mass > 91.2
        + 0.039006971337021905 * max(0.0, Q.mass_over_sum_pt - 0.089) / 0.01467876341912125   # +3.9%  mass_over_sum_pt > 0.089
        - 0.03614007710145223 * max(0.0, 0.0135 - Q.mass_over_sum_pt_sq) / 0.006521822824320578   # -3.6%  mass_over_sum_pt_sq < 0.0135
        - 0.033260479536900964 * max(0.0, 0.00149 - Q.lam2) / 0.0007274427118525141   # -3.3%  lam2 < 0.00149
        + 0.03295046872634217 * max(0.0, Q.mass_top50 - 98.1) / 11.875704991469924   # +3.3%  mass_top50 > 98.1
        + 0.030975927241053622 * max(0.0, 0.00763 - Q.girth2_top30) / 0.002446445006585757   # +3.1%  girth2_top30 < 0.00763
        + 0.029288158683717893 * max(0.0, Q.girth2_top20 - 0.00813) / 0.0028825368857012088   # +2.9%  girth2_top20 > 0.00813
        + 0.0282515239684042 * max(0.0, 0.0913 - Q.z_dr_0p2_0p4) / 0.0583010460645545   # +2.8%  z_dr_0p2_0p4 < 0.0913
        + 0.021702146402247144 * max(0.0, 1010.0 - Q.sum_pt) / 18.51132844792214   # +2.2%  sum_pt < 1010
        + 0.021614662639641753 * max(0.0, Q.n_particles - 50.7) / 4.0970460504210795   # +2.2%  n_particles > 50.7
        - 0.016353687213738664 * max(0.0, Q.e2 - 0.0256) / 0.010011417149870402   # -1.6%  e2 > 0.0256
        + 0.015569756914200863 * max(0.0, 64.6 - Q.mass) / 5.964328043961615   # +1.6%  mass < 64.6
        - 0.013057440860179566 * max(0.0, Q.mass - 140.0) / 4.527493735869592   # -1.3%  mass > 140
        - 0.011257000991954244 * max(0.0, 1080.0 - Q.sum_pt_top50) / 68.58506686662946   # -1.1%  sum_pt_top50 < 1080
        + 0.010667622882001127 * max(0.0, Q.girth2_top40 - 0.00516) * max(0.0, 912.0 - Q.sum_pt_top30) / 0.19223632804595142   # +1.1%  girth2_top40 > 0.00516 and sum_pt_top30 < 912
        - 0.008837079655730292 * max(0.0, Q.n_particles - 51.1) * max(0.0, Q.z_top50_slots - 0.971) / 0.038458079778034955   # -0.9%  n_particles > 51.1 and z_top50_slots > 0.971
        + 0.008419847250207441 * max(0.0, 995.0 - Q.sum_pt_top40) / 24.263162083114494   # +0.8%  sum_pt_top40 < 995
        + 0.007301682804994908 * max(0.0, 967.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt) / 1.3678183186236712   # +0.7%  sum_pt_top40 < 967 and log_sum_pt < 6.82
        + 0.007210499423820662 * max(0.0, 0.144 - Q.z_dr_0p1_0p2) / 0.060694301162811326   # +0.7%  z_dr_0p1_0p2 < 0.144
        + 0.007045661805607195 * max(0.0, 1.81 - Q.D2) / 0.2955616691105606   # +0.7%  D2 < 1.81
        - 0.006568962033990066 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, 0.392 - Q.max_dr) / 0.9946406389657481   # -0.7%  sum_pt_top40 < 1010 and max_dr < 0.392
        - 0.006421447917713614 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.5 - Q.n_real_top50) / 69.04180865365677   # -0.6%  sum_pt < 1010 and n_real_top50 < 43.5
        + 0.0060659545910532586 * max(0.0, 0.00142 - Q.lam2) * max(0.0, Q.n_dr_0p05_0p1 - 4.52) / 0.004434935040350577   # +0.6%  lam2 < 0.00142 and n_dr_0p05_0p1 > 4.52
        + 0.0045155550821198666 * max(0.0, Q.C2 - 0.0718) / 0.013160511857854613   # +0.5%  C2 > 0.0718
        + 0.00310320552001881 * max(0.0, Q.mass - 85.7) * max(0.0, 6.91 - Q.log_sum_pt) / 0.3545019865473656   # +0.3%  mass > 85.7 and log_sum_pt < 6.91
        + 0.002544136736644767 * max(0.0, Q.sum_pt_top40 - 1060.0) / 25.039361691668855   # +0.3%  sum_pt_top40 > 1060
        + 0.0016345778777196918 * max(0.0, 15.9 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p1_0p2 - 0.3) / 0.13942495551980816   # +0.2%  n_dr_0p2_0p4 < 15.9 and z_dr_0p1_0p2 > 0.3
        - 0.0011309217575453437 * max(0.0, Q.z_dr_0p1_0p2 - 0.212) * max(0.0, Q.mean_phi - 0.000543) / 7.420345121203711e-06   # -0.1%  z_dr_0p1_0p2 > 0.212 and mean_phi > 0.000543
    )
    return max(0.0, z)


def neuron_9(Q):
    # scale S = 9.267;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 9.267082987888045 * (-0.10445573879776011
        + 0.2276937630942669 * max(0.0, 135.0 - Q.mass_top50) / 51.71708329458557   # +22.8%  mass_top50 < 135
        + 0.2091394481986718 * max(0.0, Q.mass - 80.4) / 20.02182461258476   # +20.9%  mass > 80.4
        - 0.18410857092330768 * max(0.0, Q.mass - 63.0) / 32.31343571075311   # -18.4%  mass > 63
        - 0.11324763156758365 * max(0.0, 0.0208 - Q.girth2_top15) / 0.014105849461271068   # -11.3%  girth2_top15 < 0.0208
        + 0.050523617226460246 * max(0.0, 0.00603 - Q.girth2_top40) / 0.0013531981320401652   # +5.1%  girth2_top40 < 0.00603
        - 0.047965256551467134 * max(0.0, 0.0435 - Q.girth) / 0.006287100608172371   # -4.8%  girth < 0.0435
        + 0.032753869123612465 * max(0.0, 0.21 - Q.LHA) / 0.021226071562443397   # +3.3%  LHA < 0.21
        - 0.0218308083118054 * max(0.0, Q.mass - 145.0) / 3.7673726688673517   # -2.2%  mass > 145
        + 0.015379925295174906 * max(0.0, 951.0 - Q.sum_pt) / 7.021036653098739   # +1.5%  sum_pt < 951
        - 0.014088476446383398 * max(0.0, 6.85 - Q.log_sum_pt) / 0.007460518880088058   # -1.4%  log_sum_pt < 6.85
        - 0.012224086760768286 * max(0.0, Q.width - 0.0258) / 0.0004720067769299303   # -1.2%  width > 0.0258
        + 0.011818883391636953 * max(0.0, 0.0183 - Q.girth2_top15) * max(0.0, Q.n_particles - 40.2) / 0.0876212585715771   # +1.2%  girth2_top15 < 0.0183 and n_particles > 40.2
        + 0.011145796494710609 * max(0.0, Q.girth - 0.0981) / 0.0081975413557615   # +1.1%  girth > 0.0981
        - 0.006842852764170125 * max(0.0, 0.0277 - Q.girth) / 0.0023839580616339716   # -0.7%  girth < 0.0277
        - 0.006478692661603785 * max(0.0, 0.973 - Q.z_top50_slots) / 0.0017252466249455046   # -0.6%  z_top50_slots < 0.973
        - 0.006279204493828414 * max(0.0, Q.lam2 - 0.00153) / 0.0006401530158660886   # -0.6%  lam2 > 0.00153
        + 0.005208889860337832 * max(0.0, 0.00614 - Q.girth2_top40) * max(0.0, 1020.0 - Q.sum_pt) / 0.02822878047398787   # +0.5%  girth2_top40 < 0.00614 and sum_pt < 1020
        + 0.0048376352077585906 * max(0.0, 965.0 - Q.sum_pt_top40) / 15.841260401211265   # +0.5%  sum_pt_top40 < 965
        - 0.0039674780167298735 * max(0.0, Q.n_dr_0p1_0p2 - 21.2) / 1.3567139495814469   # -0.4%  n_dr_0p1_0p2 > 21.2
        + 0.003715251174708034 * max(0.0, 120.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.38) / 0.6558007801308171   # +0.4%  mass_top40 < 120 and max_dr > 0.38
        + 0.0034440837811700736 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0782 - Q.dr_7) / 0.00020073339759334792   # +0.3%  log_sum_pt < 6.85 and dr_7 < 0.0782
        + 0.0034086712478652655 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, Q.z_top20_slots - 0.898) / 0.00018365371704880996   # +0.3%  log_sum_pt < 6.85 and z_top20_slots > 0.898
        - 0.002056944359285016 * max(0.0, 965.0 - Q.sum_pt_top40) * max(0.0, Q.z_top30_slots - 0.97) / 0.06931590574168162   # -0.2%  sum_pt_top40 < 965 and z_top30_slots > 0.97
        - 0.0018401630466934674 * max(0.0, Q.e2 - 0.0659) / 0.00037561549922804555   # -0.2%  e2 > 0.0659
    )
    return max(0.0, z)


def neuron_10(Q):
    # scale S = 10.29;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 10.286411019084115 * (0.4646907450160983
        - 0.13297822204851628 * max(0.0, 0.0253 - Q.girth2_top5) / 0.019738364337345916   # -13.3%  girth2_top5 < 0.0253
        - 0.12178773798821188 * max(0.0, 121.0 - Q.mass) / 39.14871031347724   # -12.2%  mass < 121
        - 0.08624344932684196 * max(0.0, 0.0663 - Q.e2) / 0.035771595462881035   # -8.6%  e2 < 0.0663
        + 0.06607876031053163 * max(0.0, 86.4 - Q.mass) / 13.676323705949105   # +6.6%  mass < 86.4
        - 0.05297697898781383 * max(0.0, Q.z_dr_0_0p05 - 0.767) / 0.05239836350173421   # -5.3%  z_dr_0_0p05 > 0.767
        + 0.05233141021585133 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +5.2%  mass < 80.4
        + 0.05033251170622376 * max(0.0, 0.089 - Q.z_dr_0p2_0p4) / 0.05646029476914725   # +5.0%  z_dr_0p2_0p4 < 0.089
        - 0.040727138194002026 * max(0.0, Q.mass - 144.0) / 3.9152904962106914   # -4.1%  mass > 144
        - 0.03680528817419693 * max(0.0, 0.123 - Q.girth) / 0.05771254906030878   # -3.7%  girth < 0.123
        - 0.03604000993407791 * max(0.0, 13.1 - Q.n_dr_0p2_0p4) / 6.008466050466775   # -3.6%  n_dr_0p2_0p4 < 13.1
        + 0.03571504068253582 * max(0.0, Q.mass_top50 - 137.0) / 4.2179057178401145   # +3.6%  mass_top50 > 137
        + 0.03496729357962245 * max(0.0, 0.0236 - Q.girth2_top5) * max(0.0, 655.0 - Q.sum_pt_top3) / 3.361569663411006   # +3.5%  girth2_top5 < 0.0236 and sum_pt_top3 < 655
        + 0.03081709920930505 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.464 - Q.tau21) / 3.107817145909842   # +3.1%  mass < 120 and tau21 < 0.464
        - 0.0298711137106149 * max(0.0, Q.lam1 - 0.00223) / 0.005931786741798914   # -3.0%  lam1 > 0.00223
        - 0.028634400960714776 * max(0.0, 81.3 - Q.mass_top10) / 36.31876912048943   # -2.9%  mass_top10 < 81.3
        + 0.023155510057991102 * max(0.0, 0.00533 - Q.girth2_top30) / 0.001202965120267842   # +2.3%  girth2_top30 < 0.00533
        - 0.018183773720117417 * max(0.0, 0.0672 - Q.e2) * max(0.0, 0.218 - Q.z_dr_0p1_0p2) / 0.004987887209683945   # -1.8%  e2 < 0.0672 and z_dr_0p1_0p2 < 0.218
        - 0.017823915546347255 * max(0.0, 0.402 - Q.max_dr) / 0.058953093658897456   # -1.8%  max_dr < 0.402
        - 0.017442944468364783 * max(0.0, 63.0 - Q.mass) / 5.572214167225461   # -1.7%  mass < 63
        - 0.01250720961838865 * max(0.0, 5.96 - Q.n_dr_0p2_0p4) / 1.5243400336088604   # -1.3%  n_dr_0p2_0p4 < 5.96
        - 0.01232294891837287 * max(0.0, Q.sum_pt_top10 - 677.0) / 125.50387875402114   # -1.2%  sum_pt_top10 > 677
        + 0.01119747919141245 * max(0.0, 0.0642 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - 2.93) / 0.1301490094243052   # +1.1%  dr_0 < 0.0642 and n_dr_0p2_0p4 > 2.93
        - 0.00918713287926872 * max(0.0, Q.mass - 162.0) / 1.5963281230253332   # -0.9%  mass > 162
        - 0.008335479516903107 * max(0.0, 959.0 - Q.sum_pt_top50) / 9.923862077780331   # -0.8%  sum_pt_top50 < 959
        - 0.007661724593986538 * max(0.0, 47.4 - Q.m012) / 33.114137936458356   # -0.8%  m012 < 47.4
        - 0.0064830201727365455 * max(0.0, 114.0 - Q.mass) * max(0.0, 2.16 - Q.D2) / 10.86107657032277   # -0.6%  mass < 114 and D2 < 2.16
        + 0.005251282577177659 * max(0.0, 0.0847 - Q.z_dr_0p2_0p4) * max(0.0, Q.max_dr - 0.265) / 0.004656625083293509   # +0.5%  z_dr_0p2_0p4 < 0.0847 and max_dr > 0.265
        + 0.004027496366777874 * max(0.0, 0.24 - Q.max_dr) / 0.0050894942268483   # +0.4%  max_dr < 0.24
        + 0.0031332484613231276 * max(0.0, Q.mass_top50 - 162.0) * max(0.0, Q.D2 - -0.418) / 2.6858234581735476   # +0.3%  mass_top50 > 162 and D2 > -0.418
        + 0.0025301150057675022 * max(0.0, 671.0 - Q.sum_pt_top10) / 27.984734274061186   # +0.3%  sum_pt_top10 < 671
        - 0.0016804208381920897 * max(0.0, Q.mass_top50 - 172.0) * max(0.0, 1240.0 - Q.sum_pt) / 53.84890787127002   # -0.2%  mass_top50 > 172 and sum_pt < 1240
        - 0.0016052497427150785 * max(0.0, Q.lam1 - 0.00517) * max(0.0, Q.pt_dispersion - 0.276) / 8.51147352672491e-05   # -0.2%  lam1 > 0.00517 and pt_dispersion > 0.276
        + 0.0011645932950967433 * max(0.0, Q.mass_top10 - 89.1) / 1.4076951002860891   # +0.1%  mass_top10 > 89.1
    )
    return max(0.0, z)


def neuron_11(Q):
    # scale S = 4.304;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 4.304327999211462 * (-0.008642464051720712
        + 0.2456654451008299 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # +24.6%  mass < 91.2
        - 0.13994891030939138 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # -14.0%  mass < 80.4
        - 0.12233056591991372 * max(0.0, 0.00644 - Q.girth2_top30) / 0.0017207545099623138   # -12.2%  girth2_top30 < 0.00644
        + 0.08051516157401324 * max(0.0, 0.0258 - Q.e2) / 0.00480670824305215   # +8.1%  e2 < 0.0258
        + 0.05174463829650759 * max(0.0, 9.59 - Q.n_dr_0p2_0p4) * max(0.0, 22.0 - Q.n_dr_0p1_0p2) / 43.671744201711235   # +5.2%  n_dr_0p2_0p4 < 9.59 and n_dr_0p1_0p2 < 22
        + 0.04135831278132122 * max(0.0, 0.00633 - Q.z_dr_0p2_0p4) / 0.001435643094393437   # +4.1%  z_dr_0p2_0p4 < 0.00633
        + 0.04030978198128765 * max(0.0, 89.6 - Q.mass_top50) * max(0.0, 0.00355 - Q.girth2_top15) / 0.03970400989111352   # +4.0%  mass_top50 < 89.6 and girth2_top15 < 0.00355
        - 0.03760603218160714 * max(0.0, 9.84 - Q.n_dr_0p2_0p4) * max(0.0, 0.0057 - Q.girth2) / 0.005307170401919308   # -3.8%  n_dr_0p2_0p4 < 9.84 and girth2 < 0.0057
        + 0.03503245897939614 * max(0.0, 8.15 - Q.n_dr_0p2_0p4) * max(0.0, 0.0608 - Q.z_dr_0p2_0p4) / 0.14783450398651157   # +3.5%  n_dr_0p2_0p4 < 8.15 and z_dr_0p2_0p4 < 0.0608
        - 0.033935614357057574 * max(0.0, 80.4 - Q.mass) * max(0.0, 0.034 - Q.z_dr_0p2_0p4) / 0.27050002786578775   # -3.4%  mass < 80.4 and z_dr_0p2_0p4 < 0.034
        - 0.03275680199086795 * max(0.0, Q.z_top5_slots - 0.545) / 0.07833112220773261   # -3.3%  z_top5_slots > 0.545
        + 0.027538386687806282 * max(0.0, 0.0985 - Q.z_dr_0_0p05) / 0.022114598670417308   # +2.8%  z_dr_0_0p05 < 0.0985
        + 0.020214216020156644 * max(0.0, 106.0 - Q.mass) * max(0.0, 0.368 - Q.planar_flow) / 1.2169037202471207   # +2.0%  mass < 106 and planar_flow < 0.368
        - 0.01794880222571069 * max(0.0, 10.7 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_dr_0p05_0p1 - 0.593) / 0.2334064410043369   # -1.8%  n_dr_0p2_0p4 < 10.7 and z_dr_0p05_0p1 > 0.593
        - 0.014730374670744315 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.33 - Q.D2) / 3.3725725603276637   # -1.5%  mass < 80.4 and D2 < 3.33
        + 0.013620910203119552 * max(0.0, 103.0 - Q.mass) * max(0.0, Q.mass_top10 - 45.1) / 86.09231301326369   # +1.4%  mass < 103 and mass_top10 > 45.1
        - 0.012870900463287175 * max(0.0, 80.4 - Q.mass) * max(0.0, 467.0 - Q.sum_pt_top2) / 818.3246268713553   # -1.3%  mass < 80.4 and sum_pt_top2 < 467
        + 0.010010451976145125 * max(0.0, 98.7 - Q.mass) * max(0.0, 1.3 - Q.D2) / 1.3217260345301587   # +1.0%  mass < 98.7 and D2 < 1.3
        + 0.009927926924304727 * max(0.0, 0.296 - Q.max_dr) / 0.013068212181776808   # +1.0%  max_dr < 0.296
        - 0.006470839822906323 * max(0.0, 98.9 - Q.mass) * max(0.0, Q.n_dr_0p2_0p4 - 10.1) / 9.377985531363038   # -0.6%  mass < 98.9 and n_dr_0p2_0p4 > 10.1
        + 0.005463467533625638 * max(0.0, Q.z_dr_0p05_0p1 - 0.851) / 0.0035523498909014535   # +0.5%  z_dr_0p05_0p1 > 0.851
    )
    return max(0.0, z)


def neuron_12(Q):
    # scale S = 2.709;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 2.7089391469119084 * (0.04392863536106215
        + 0.4455279655127612 * max(0.0, 80.4 - Q.mass) / 10.680603069217145   # +44.6%  mass < 80.4
        - 0.09440917935193348 * max(0.0, Q.C2 - 0.0576) / 0.019085725505543286   # -9.4%  C2 > 0.0576
        - 0.08563608186505355 * max(0.0, 60.1 - Q.mass_top50) / 5.076213009899267   # -8.6%  mass_top50 < 60.1
        - 0.06904839461159568 * max(0.0, 0.0488 - Q.girth) / 0.007959485072117991   # -6.9%  girth < 0.0488
        - 0.06656339661076803 * max(0.0, 84.7 - Q.mass) * max(0.0, 0.0652 - Q.z_dr_0p1_0p2) / 0.57793650907158   # -6.7%  mass < 84.7 and z_dr_0p1_0p2 < 0.0652
        - 0.04793048322292862 * max(0.0, Q.planar_flow - 0.266) / 0.2649811476183588   # -4.8%  planar_flow > 0.266
        - 0.033290751278563686 * max(0.0, 94.4 - Q.mass_top40) * max(0.0, 471.0 - Q.sum_pt_top2) / 2112.005137438146   # -3.3%  mass_top40 < 94.4 and sum_pt_top2 < 471
        - 0.028570392388482295 * max(0.0, Q.mass_top20 - 127.0) * max(0.0, Q.C2 - 0.0598) / 0.05775780177895053   # -2.9%  mass_top20 > 127 and C2 > 0.0598
        + 0.024846488262306086 * max(0.0, Q.mass_top30 - 137.0) / 1.901345330990062   # +2.5%  mass_top30 > 137
        - 0.02039073886903518 * max(0.0, 6.86 - Q.log_sum_pt) / 0.00842031566414446   # -2.0%  log_sum_pt < 6.86
        + 0.019990161463555336 * max(0.0, Q.LHA - 0.4) / 0.0035862338371996623   # +2.0%  LHA > 0.4
        - 0.019949309998941626 * max(0.0, Q.z_dr_0_0p05 - 0.918) / 0.007423278407968975   # -2.0%  z_dr_0_0p05 > 0.918
        + 0.01709687262905068 * max(0.0, Q.mass_top20 - 127.0) / 1.16660925830232   # +1.7%  mass_top20 > 127
        + 0.011226704597201846 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.32) / 0.5234502508464022   # +1.1%  mass < 91.2 and z_dr_0p05_0p1 > 0.32
        - 0.00840652377542184 * max(0.0, Q.z_dr_0p1_0p2 - 0.685) * max(0.0, Q.min_pair_mass - 0.762) / 0.012793686148699954   # -0.8%  z_dr_0p1_0p2 > 0.685 and min_pair_mass > 0.762
        + 0.007116555562400977 * max(0.0, Q.LHA - 0.406) * max(0.0, 0.00368 - Q.lam2) / 1.5179776341859605e-06   # +0.7%  LHA > 0.406 and lam2 < 0.00368
    )
    return max(0.0, z)


def neuron_13(Q):
    # scale S = 5.632;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 5.632047003736996 * (0.420806147112666
        - 0.11693356497102322 * max(0.0, Q.mass - 143.0) / 4.065279840934177   # -11.7%  mass > 143
        + 0.10359902086576434 * max(0.0, Q.mass - 74.9) / 23.62245162174553   # +10.4%  mass > 74.9
        - 0.09602129888952646 * max(0.0, 7.02 - Q.log_sum_pt) / 0.09135075484893446   # -9.6%  log_sum_pt < 7.02
        - 0.08644244940734998 * max(0.0, 1010.0 - Q.sum_pt) / 18.51132844792214   # -8.6%  sum_pt < 1010
        - 0.07770044022886428 * max(0.0, 1060.0 - Q.sum_pt) / 47.722195374048056   # -7.8%  sum_pt < 1060
        - 0.07000534715977899 * max(0.0, Q.mass_top50 - 98.1) / 11.875704991469924   # -7.0%  mass_top50 > 98.1
        + 0.06725226781856061 * max(0.0, Q.mass_top50 - 137.0) / 4.2179057178401145   # +6.7%  mass_top50 > 137
        + 0.06430845567920543 * max(0.0, 1050.0 - Q.sum_pt_top40) / 55.46527490398503   # +6.4%  sum_pt_top40 < 1050
        + 0.06269960929733218 * max(0.0, 0.00947 - Q.mass_over_sum_pt_sq) / 0.003395453333447307   # +6.3%  mass_over_sum_pt_sq < 0.00947
        + 0.03982620909815721 * max(0.0, 1010.0 - Q.sum_pt_top50) / 22.475258679506958   # +4.0%  sum_pt_top50 < 1010
        + 0.030338632968194463 * max(0.0, Q.mass_over_sum_pt - 0.171) / 0.000547655791365372   # +3.0%  mass_over_sum_pt > 0.171
        - 0.0293507772504666 * max(0.0, Q.mass_over_sum_pt_sq - 0.0292) / 0.0002028281681850827   # -2.9%  mass_over_sum_pt_sq > 0.0292
        + 0.021164999729543273 * max(0.0, Q.mass_top10 - 57.1) / 8.00015257119923   # +2.1%  mass_top10 > 57.1
        + 0.02058467826669904 * max(0.0, Q.mass - 172.8) / 0.7291438714141659   # +2.1%  mass > 172.8
        - 0.01871576578374679 * max(0.0, 1030.0 - Q.sum_pt_top40) * max(0.0, 4.94 - Q.D2) / 90.86902810775386   # -1.9%  sum_pt_top40 < 1030 and D2 < 4.94
        - 0.01694604715738006 * max(0.0, 3.96 - Q.n_dr_0p2_0p4) / 0.7176010084053247   # -1.7%  n_dr_0p2_0p4 < 3.96
        - 0.015584275434141131 * max(0.0, 46.4 - Q.n_particles) / 6.599351260471175   # -1.6%  n_particles < 46.4
        + 0.01488028368499936 * max(0.0, Q.n_particles - 46.0) / 6.207885714285714   # +1.5%  n_particles > 46
        + 0.01482018209144419 * max(0.0, 6.81 - Q.log_sum_pt) / 0.0048247376961245625   # +1.5%  log_sum_pt < 6.81
        + 0.00949403496906368 * max(0.0, 1070.0 - Q.sum_pt) * max(0.0, Q.tau21 - 0.241) / 13.367712800222339   # +0.9%  sum_pt < 1070 and tau21 > 0.241
        - 0.007005801635217082 * max(0.0, Q.mass - 161.0) * max(0.0, 6.47 - Q.D2) / 7.444717756301908   # -0.7%  mass > 161 and D2 < 6.47
        + 0.0061107287715090955 * max(0.0, 953.0 - Q.sum_pt_top50) * max(0.0, 4.76 - Q.D2) / 15.933292438994098   # +0.6%  sum_pt_top50 < 953 and D2 < 4.76
        - 0.005396629128975892 * max(0.0, Q.mass_top50 - 169.0) / 0.6636259588674337   # -0.5%  mass_top50 > 169
        - 0.002474350029027892 * max(0.0, Q.sum_pt - 1250.0) * max(0.0, 0.0801 - Q.dr_7) / 0.3440902633872367   # -0.2%  sum_pt > 1250 and dr_7 < 0.0801
        - 0.0023441496840287553 * max(0.0, Q.sum_pt - 1250.0) / 8.200224350462841   # -0.2%  sum_pt > 1250
    )
    return max(0.0, z)


def neuron_14(Q):
    # scale S = 13.47;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 13.471914013238363 * (0.015216842966675255
        - 0.22108257243626042 * max(0.0, 0.0907 - Q.mass_over_sum_pt) / 0.01805094185264753   # -22.1%  mass_over_sum_pt < 0.0907
        + 0.20430995543055008 * max(0.0, 0.0984 - Q.mass_over_sum_pt) / 0.02372798406559425   # +20.4%  mass_over_sum_pt < 0.0984
        - 0.16138346383708158 * max(0.0, 91.2 - Q.mass) / 16.470788999785803   # -16.1%  mass < 91.2
        + 0.11567555833879518 * max(0.0, 136.0 - Q.mass) / 51.431391926521016   # +11.6%  mass < 136
        - 0.051838517090098464 * max(0.0, 0.00935 - Q.girth2_top50) / 0.0033737393469159124   # -5.2%  girth2_top50 < 0.00935
        + 0.04666226865454676 * max(0.0, 0.00238 - Q.lam2) / 0.0014518015495997266   # +4.7%  lam2 < 0.00238
        + 0.03179698049022665 * max(0.0, 0.00626 - Q.lam1) / 0.0016475622578651973   # +3.2%  lam1 < 0.00626
        - 0.026411234091399317 * max(0.0, 0.0166 - Q.girth2_top20) * max(0.0, Q.z_top50_slots - 0.969) / 0.00027161059134568036   # -2.6%  girth2_top20 < 0.0166 and z_top50_slots > 0.969
        - 0.024089820263999844 * max(0.0, 80.7 - Q.mass_top40) / 12.339771376082622   # -2.4%  mass_top40 < 80.7
        - 0.01986165745483582 * max(0.0, Q.max_dr - 0.249) / 0.11011297999668429   # -2.0%  max_dr > 0.249
        + 0.01788168453790086 * max(0.0, 21.4 - Q.n_dr_0p2_0p4) / 12.951640672390022   # +1.8%  n_dr_0p2_0p4 < 21.4
        - 0.016476110269135653 * max(0.0, Q.sum_pt_top50 - 978.0) / 70.91525265764508   # -1.6%  sum_pt_top50 > 978
        + 0.014462115107582526 * max(0.0, Q.log_sum_pt - 6.94) / 0.035231893522406375   # +1.4%  log_sum_pt > 6.94
        - 0.013198872904606057 * max(0.0, 0.0163 - Q.girth2_top20) * max(0.0, 0.637 - Q.tau21) / 0.0018736994820075289   # -1.3%  girth2_top20 < 0.0163 and tau21 < 0.637
        - 0.01083658041498731 * max(0.0, 19.9 - Q.n_dr_0p2_0p4) * max(0.0, 21.3 - Q.n_dr_0p1_0p2) / 130.34774959665353   # -1.1%  n_dr_0p2_0p4 < 19.9 and n_dr_0p1_0p2 < 21.3
        + 0.010775924564334313 * max(0.0, 0.00623 - Q.lam1) * max(0.0, Q.z_top50_slots - 0.987) / 1.7944663676619893e-05   # +1.1%  lam1 < 0.00623 and z_top50_slots > 0.987
        + 0.007540025135614823 * max(0.0, Q.max_dr - 0.201) * max(0.0, 52.1 - Q.mass_top10) / 2.533131428545108   # +0.8%  max_dr > 0.201 and mass_top10 < 52.1
        + 0.004258766285006811 * max(0.0, 0.959 - Q.z_top50_slots) / 0.0006403318436840441   # +0.4%  z_top50_slots < 0.959
        - 0.0014578926930374817 * max(0.0, Q.m012 - 36.2) / 1.852887264257497   # -0.1%  m012 > 36.2
    )
    return max(0.0, z)


def neuron_15(Q):
    # scale S = 8.658;  each line: share (fraction of this neuron's average input) * term / its average size
    z = 8.657801799522408 * (-0.3534384444061539
        + 0.2797997203227498 * Q.sum_pt_top30 / 992.8075910312172   # +28.0%  sum_pt_top30
        - 0.18703672208056424 * max(0.0, 0.0135 - Q.girth2_top50) / 0.006609497424513803   # -18.7%  girth2_top50 < 0.0135
        + 0.14490129554473016 * max(0.0, 0.0128 - Q.girth2_top50) * max(0.0, 0.129 - Q.z_dr_0p2_0p4) / 0.0006781225390920503   # +14.5%  girth2_top50 < 0.0128 and z_dr_0p2_0p4 < 0.129
        + 0.07195962237251728 * max(0.0, 7.02 - Q.log_sum_pt) / 0.09135075484893446   # +7.2%  log_sum_pt < 7.02
        + 0.06119350068776812 * max(0.0, 0.123 - Q.girth) / 0.05771254906030878   # +6.1%  girth < 0.123
        + 0.04245115128534548 * max(0.0, 0.177 - Q.z_dr_0p1_0p2) / 0.0807766272505631   # +4.2%  z_dr_0p1_0p2 < 0.177
        - 0.03718553888737814 * max(0.0, 0.0498 - Q.girth) / 0.00829755220348848   # -3.7%  girth < 0.0498
        + 0.028634122026182294 * max(0.0, 0.00711 - Q.girth2_top5) / 0.003598092209085998   # +2.9%  girth2_top5 < 0.00711
        - 0.027132342616061417 * max(0.0, 0.0195 - Q.z_dr_0p2_0p4) / 0.0076516757239933255   # -2.7%  z_dr_0p2_0p4 < 0.0195
        + 0.026203856289133905 * max(0.0, 0.00212 - Q.lam2) / 0.0012329771420352723   # +2.6%  lam2 < 0.00212
        - 0.023618758746871072 * max(0.0, Q.mass_top10 - 26.6) / 25.18307044102783   # -2.4%  mass_top10 > 26.6
        - 0.013260822727643604 * max(0.0, 1000.0 - Q.sum_pt) / 15.267230701401655   # -1.3%  sum_pt < 1000
        - 0.009927488736416442 * max(0.0, 0.00177 - Q.girth2_top5) * max(0.0, Q.n_dr_0p2_0p4 - 2.31) / 0.002974056396085976   # -1.0%  girth2_top5 < 0.00177 and n_dr_0p2_0p4 > 2.31
        - 0.00849506195334143 * max(0.0, 0.127 - Q.z_dr_0p1_0p2) * max(0.0, 758.0 - Q.sum_pt_top3) / 12.17691434879036   # -0.8%  z_dr_0p1_0p2 < 0.127 and sum_pt_top3 < 758
        - 0.007609547329177589 * max(0.0, 0.00599 - Q.girth2_top5) * max(0.0, 0.332 - Q.max_dr) / 4.0171922292746744e-05   # -0.8%  girth2_top5 < 0.00599 and max_dr < 0.332
        + 0.007584425519461683 * max(0.0, 0.117 - Q.z_dr_0p1_0p2) * max(0.0, 0.818 - Q.planar_flow) / 0.011400078630336638   # +0.8%  z_dr_0p1_0p2 < 0.117 and planar_flow < 0.818
        + 0.006777696703028478 * max(0.0, 0.00727 - Q.girth2_top5) * max(0.0, 975.0 - Q.sum_pt_top40) / 0.047707280253737445   # +0.7%  girth2_top5 < 0.00727 and sum_pt_top40 < 975
        + 0.0046922426830752146 * max(0.0, Q.girth - 0.0858) / 0.010833201905419853   # +0.5%  girth > 0.0858
        - 0.00439585061403712 * max(0.0, Q.sum_pt_top50 - 1120.0) / 17.867795003118434   # -0.4%  sum_pt_top50 > 1120
        - 0.003508317189724184 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.664 - Q.tau32) / 0.8829742697235301   # -0.4%  sum_pt < 1000 and tau32 < 0.664
        - 0.002720095799454662 * max(0.0, Q.z_dr_0p05_0p1 - 0.849) / 0.003656840109843465   # -0.3%  z_dr_0p05_0p1 > 0.849
        - 0.0009118198853377849 * max(0.0, 0.051 - Q.girth) * max(0.0, Q.z_dr_0p2_0p4 - 0.0672) / 9.892676496388209e-07   # -0.1%  girth < 0.051 and z_dr_0p2_0p4 > 0.0672
    )
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


H_AVG = [0.524445955882353, 2.7630600315126053, 0.38618655462184875, 0.5078138130252101, 0.9913307773109243, 1.002253361344538, 0.8098634453781512, 0.4985296218487395, 2.0291856092436973, 0.792392962184874, 1.2277453781512606, 0.7371721638655462, 0.5027090336134454, 1.8962013655462184, 0.5546384978991596, 0.19645693277310924]
T = [4.0239282710740545, 2.6505035139837188, 5.2881566636029405, 5.183463665309874, 4.478660905495011]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    # rounded to a multiple of 2^-20: the formula's class scores are exact binary fractions; this removes the 1e-15 floating-point noise
    # of the normalized form, so exact ties between two classes are broken exactly as in the formula
    return [round(v * 2 ** 20) / 2 ** 20 for v in [
        -1.078125 + T[0] * (   # class g: n1 +43%, n4 -22%, n9 +10%, n3 -9%, n5 -8%, n6 +3% ...
            + 0.4291608605723074 * h[1] / H_AVG[1]
            - 0.21556408855059714 * h[4] / H_AVG[4]
            + 0.10153700404245013 * h[9] / H_AVG[9]
            - 0.09464889384503107 * h[3] / H_AVG[3]
            - 0.0778354270556042 * h[5] / H_AVG[5]
            + 0.03144717172271069 * h[6] / H_AVG[6]
            + 0.02928044980327208 * h[12] / H_AVG[12]
            + 0.015758742705406076 * h[8] / H_AVG[8]
            - 0.0047673617026212645 * h[10] / H_AVG[10]
        ),
        1.359375 + T[1] * (   # class q: n4 -40%, n1 -20%, n9 +17%, n11 -7%, n6 +7%, n12 +7% ...
            - 0.397392022057635 * h[4] / H_AVG[4]
            - 0.19546239164570897 * h[1] / H_AVG[1]
            + 0.16816466715755107 * h[9] / H_AVG[9]
            - 0.06953133244082868 * h[11] / H_AVG[11]
            + 0.06683923554215625 * h[6] / H_AVG[6]
            + 0.06519751035715222 * h[12] / H_AVG[12]
            + 0.01821288637160721 * h[2] / H_AVG[2]
            + 0.011962264896898202 * h[8] / H_AVG[8]
            - 0.007237689530462281 * h[10] / H_AVG[10]
        ),
        0.09375 + T[2] * (   # class W: n8 -34%, n14 -14%, n5 +11%, n11 +8%, n0 +7%, n4 +6% ...
            - 0.33575733871668645 * h[8] / H_AVG[8]
            - 0.14421432327455838 * h[14] / H_AVG[14]
            + 0.10957083183926207 * h[5] / H_AVG[5]
            + 0.08276910086793002 * h[11] / H_AVG[11]
            + 0.07438025987750846 * h[0] / H_AVG[0]
            + 0.06444021544332539 * h[4] / H_AVG[4]
            - 0.05892053384121473 * h[7] / H_AVG[7]
            + 0.04201247378461004 * h[3] / H_AVG[3]
            - 0.03861942031920037 * h[12] / H_AVG[12]
            - 0.03277814397424517 * h[9] / H_AVG[9]
            + 0.009571665242921966 * h[6] / H_AVG[6]
            - 0.006965692818537075 * h[15] / H_AVG[15]
        ),
        0.984375 + T[3] * (   # class Z: n8 -37%, n6 -15%, n0 -14%, n5 +13%, n7 +9%, n3 +4% ...
            - 0.36700585390372187 * h[8] / H_AVG[8]
            - 0.15135732849073272 * h[6] / H_AVG[6]
            - 0.1391180175843147 * h[0] / H_AVG[0]
            + 0.13293219175737728 * h[5] / H_AVG[5]
            + 0.08716034276926131 * h[7] / H_AVG[7]
            + 0.042861020650223444 * h[3] / H_AVG[3]
            + 0.03030725845645728 * h[12] / H_AVG[12]
            - 0.027938839226827255 * h[2] / H_AVG[2]
            + 0.02131914716108417 * h[15] / H_AVG[15]
        ),
        0.78125 + T[4] * (   # class t: n13 -38%, n10 +27%, n5 -10%, n8 +10%, n12 -4%, n4 +4% ...
            - 0.38369336812658916 * h[13] / H_AVG[13]
            + 0.26984893076741406 * h[10] / H_AVG[10]
            - 0.10489882423423304 * h[5] / H_AVG[5]
            + 0.09557127643324753 * h[8] / H_AVG[8]
            - 0.04209202071399197 * h[12] / H_AVG[12]
            + 0.041502253612847304 * h[4] / H_AVG[4]
            - 0.031306557719725576 * h[7] / H_AVG[7]
            - 0.01644941453360004 * h[15] / H_AVG[15]
            + 0.01463735385835121 * h[0] / H_AVG[0]
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
