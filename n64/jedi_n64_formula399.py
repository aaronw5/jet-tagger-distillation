"""JEDI-linear jet tagger, 64 particles, 3 features: the simplified formula closest to the start formula (within 0.1 point on validation jets), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

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
    z = 0.972
    if Q.LHA < 0.255:
        z += -3.98 * Q.LHA + 1.0149
    if Q.girth < 0.0567:
        z += 48.8 * Q.girth - 2.7669599999999996
    if Q.girth2_top20 < 0.00631:
        z += 146.0 * Q.girth2_top20 - 0.92126
    if Q.girth2_top40 < 0.00422:
        z += 187.0 * Q.girth2_top40 - 1.26276
    if 0.00422 <= Q.girth2_top40 < 0.00621:
        z += 238.0 * Q.girth2_top40 - 1.47798
    if Q.lam1 < 0.00592:
        z += 187.0 * Q.lam1 - 1.10704
    if Q.log_sum_pt < 7.02:
        z += -2.53 * Q.log_sum_pt + 17.760599999999997
    if 80.4 <= Q.mass < 91.2:
        z += -0.154 * Q.mass + 12.3816
    if Q.mass >= 91.2:
        z += 0.0020000000000000018 * Q.mass - 1.8455999999999992
    if 0.0769 <= Q.mass_over_sum_pt < 0.0836:
        z += -65.8 * Q.mass_over_sum_pt + 5.06002
    if Q.mass_over_sum_pt >= 0.0836:
        z += 6.299999999999997 * Q.mass_over_sum_pt - 0.9675399999999996
    if Q.mass_over_sum_pt_sq < 0.00825:
        z += -539.0 * Q.mass_over_sum_pt_sq + 4.446750000000001
    if Q.mass_top50 < 81.9:
        z += 0.0248 * Q.mass_top50 - 2.03112
    if Q.n_dr_0p2_0p4 < 10.2:
        z += -0.0369 * Q.n_dr_0p2_0p4 + 0.37638
    if Q.n_particles < 47.8:
        z += -0.00978 * Q.n_particles + 0.467484
    if Q.z_dr_0_0p05 >= 0.878:
        z += -3.23 * Q.z_dr_0_0p05 + 2.83594
    if Q.z_top30_slots >= 0.918:
        z += -3.34 * Q.z_top30_slots + 3.06612
    if Q.girth2_top40 < 0.00596 and Q.girth2_top3 < 0.00294:
        z += 41900.0 * (0.00596 - Q.girth2_top40) * (0.00294 - Q.girth2_top3)
    if Q.log_sum_pt < 6.99 and Q.sum_pt_top40 > 961.0:
        z += 0.0449 * (6.99 - Q.log_sum_pt) * (Q.sum_pt_top40 - 961.0)
    if Q.mass > 80.4 and Q.n_dr_0p2_0p4 < 6.83:
        z += -0.0104 * (Q.mass - 80.4) * (6.83 - Q.n_dr_0p2_0p4)
    if Q.mass_top50 < 80.0 and Q.z_dr_0p05_0p1 < 0.208:
        z += 0.0393 * (80.0 - Q.mass_top50) * (0.208 - Q.z_dr_0p05_0p1)
    if Q.n_dr_0p2_0p4 < 9.29 and Q.z_top50_slots < 0.987:
        z += -2.01 * (9.29 - Q.n_dr_0p2_0p4) * (0.987 - Q.z_top50_slots)
    if Q.sum_pt < 1020.0 and Q.z_dr_0p1_0p2 > 0.0965:
        z += -0.013 * (1020.0 - Q.sum_pt) * (Q.z_dr_0p1_0p2 - 0.0965)
    if Q.sum_pt < 1010.0 and Q.z_dr_0p2_0p4 < 0.207:
        z += -0.0383 * (1010.0 - Q.sum_pt) * (0.207 - Q.z_dr_0p2_0p4)
    if Q.z_top30_slots > 0.916 and Q.C2 > 0.0591:
        z += -69.5 * (Q.z_top30_slots - 0.916) * (Q.C2 - 0.0591)
    return max(0.0, z)


def neuron_1(Q):
    z = 2.19
    if Q.D2 < 1.18:
        z += -0.651 * Q.D2 + 0.76818
    if Q.LHA < 0.411:
        z += -7.47 * Q.LHA + 3.0701699999999996
    if Q.girth2_top15 < 0.00292:
        z += -120.0 * Q.girth2_top15 + 0.3504
    if Q.girth2_top20 < 0.00115:
        z += -1080.0 * Q.girth2_top20 + 1.242
    if Q.girth2_top3 < 0.000845:
        z += -766.0 * Q.girth2_top3 + 0.64727
    if Q.lam2 < 0.000842:
        z += 719.0 * Q.lam2 - 0.605398
    if 6.81 <= Q.log_sum_pt < 6.89:
        z += -6.45 * Q.log_sum_pt + 43.9245
    if 6.89 <= Q.log_sum_pt < 6.91:
        z += 12.25 * Q.log_sum_pt - 84.9185
    if 6.91 <= Q.log_sum_pt < 6.99:
        z += 42.25 * Q.log_sum_pt - 292.2185
    if 6.99 <= Q.log_sum_pt < 7.13:
        z += 21.85 * Q.log_sum_pt - 149.6225
    if Q.log_sum_pt >= 7.13:
        z += 19.520000000000003 * Q.log_sum_pt - 133.0096
    if Q.mass_over_sum_pt < 0.0742:
        z += -21.7 * Q.mass_over_sum_pt + 1.61014
    if Q.mass_top20 < 41.1:
        z += 0.0482 * Q.mass_top20 - 1.98102
    if Q.mass_top30 < 80.7:
        z += -0.0496 * Q.mass_top30 + 4.00272
    if Q.mass_top50 < 117.0:
        z += 0.0199 * Q.mass_top50 - 2.3283
    if Q.max_dr < 0.435:
        z += 7.22 * Q.max_dr - 3.1407
    if Q.n_dr_0p2_0p4 < 13.3:
        z += 0.0448 * Q.n_dr_0p2_0p4 - 0.59584
    if Q.n_particles >= 37.7:
        z += 0.0155 * Q.n_particles - 0.58435
    if Q.n_pt_above_10 < 31.3:
        z += 0.0631 * Q.n_pt_above_10 - 1.97503
    if Q.pt_9 < 35.2:
        z += 0.0259 * Q.pt_9 - 0.91168
    if Q.sum_pt_top2 < 668.0:
        z += -0.00346 * Q.sum_pt_top2 + 2.31128
    if Q.sum_pt_top30 < 1080.0:
        z += -0.00509 * Q.sum_pt_top30 + 5.4972
    if Q.sum_pt_top50 >= 960.0:
        z += -0.0103 * Q.sum_pt_top50 + 9.888
    if Q.tau32 >= 0.328:
        z += 2.09 * Q.tau32 - 0.68552
    if Q.z_top20_slots < 0.952:
        z += 10.7 * Q.z_top20_slots - 10.186399999999999
    if Q.z_top50_slots >= 0.959:
        z += -36.5 * Q.z_top50_slots + 35.003499999999995
    if Q.girth2_top3 < 0.000838 and Q.n_dr_0p05_0p1 < 10.4:
        z += -102.0 * (0.000838 - Q.girth2_top3) * (10.4 - Q.n_dr_0p05_0p1)
    if Q.mass_top20 < 40.5 and Q.n_real_top40 > 26.6:
        z += 0.00293 * (40.5 - Q.mass_top20) * (Q.n_real_top40 - 26.6)
    if Q.mass_top30 < 80.8 and Q.mass_top5 < 68.2:
        z += -0.000534 * (80.8 - Q.mass_top30) * (68.2 - Q.mass_top5)
    if Q.max_dr < 0.433 and Q.z_dr_0p2_0p4 < 0.191:
        z += 28.8 * (0.433 - Q.max_dr) * (0.191 - Q.z_dr_0p2_0p4)
    if Q.n_dr_0p1_0p2 < 7.22 and Q.dr_6 < 0.0781:
        z += -1.59 * (7.22 - Q.n_dr_0p1_0p2) * (0.0781 - Q.dr_6)
    if Q.n_particles > 37.8 and Q.dr_0 < 0.143:
        z += 0.297 * (Q.n_particles - 37.8) * (0.143 - Q.dr_0)
    if Q.n_particles > 38.2 and Q.z_top50_slots < 0.986:
        z += 1.71 * (Q.n_particles - 38.2) * (0.986 - Q.z_top50_slots)
    if Q.sum_pt_top2 < 701.0 and Q.n_dr_0p2_0p4 < 6.95:
        z += -0.000243 * (701.0 - Q.sum_pt_top2) * (6.95 - Q.n_dr_0p2_0p4)
    if Q.z_top30_slots > 0.943 and Q.m012 > 13.7:
        z += 0.373 * (Q.z_top30_slots - 0.943) * (Q.m012 - 13.7)
    if Q.z_top30_slots > 0.94 and Q.mass_top10 < 92.7:
        z += -0.169 * (Q.z_top30_slots - 0.94) * (92.7 - Q.mass_top10)
    return max(0.0, z)


def neuron_2(Q):
    z = 0.112
    if Q.lam2 < 0.000964:
        z += 408.0 * Q.lam2 - 0.393312
    if Q.log_sum_pt >= 7.05:
        z += 11.7 * Q.log_sum_pt - 82.485
    if Q.mass < 91.2:
        z += 0.0194 * Q.mass - 1.7692800000000002
    if Q.mass_over_sum_pt < 0.0732:
        z += -23.7 * Q.mass_over_sum_pt + 1.73484
    if Q.mass_top40 >= 158.0:
        z += -0.0457 * Q.mass_top40 + 7.220599999999999
    if 1010.0 <= Q.sum_pt < 1090.0:
        z += 0.005 * Q.sum_pt - 5.05
    if Q.sum_pt >= 1090.0:
        z += -0.0058000000000000005 * Q.sum_pt + 6.722
    if Q.z_top30_slots >= 0.919:
        z += 7.92 * Q.z_top30_slots - 7.27848
    if Q.sum_pt_top50 > 1110.0 and Q.C2 > 0.0963:
        z += 0.754 * (Q.sum_pt_top50 - 1110.0) * (Q.C2 - 0.0963)
    if Q.sum_pt_top50 > 1030.0 and Q.mean_phi > 3.25e-05:
        z += 3.93 * (Q.sum_pt_top50 - 1030.0) * (Q.mean_phi - 3.25e-05)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.185
    if Q.girth2_top5 < 0.000115:
        z += 7230.0 * Q.girth2_top5 - 0.83145
    if Q.lam2 < 0.000347:
        z += -2530.0 * Q.lam2 + 1.25186
    if 0.000347 <= Q.lam2 < 0.000624:
        z += -1350.0 * Q.lam2 + 0.8424
    if Q.mass_over_sum_pt_sq < 0.00728:
        z += -94.7 * Q.mass_over_sum_pt_sq + 0.689416
    if Q.n_dr_0p1_0p2 < 14.4:
        z += -0.0173 * Q.n_dr_0p1_0p2 + 0.24912
    if Q.z_dr_0p2_0p4 < 0.00143:
        z += -233.0 * Q.z_dr_0p2_0p4 + 0.33319000000000004
    if Q.mass_over_sum_pt_sq < 0.00749 and Q.girth2_top15 > 0.00486:
        z += 412000.0 * (0.00749 - Q.mass_over_sum_pt_sq) * (Q.girth2_top15 - 0.00486)
    if Q.max_dr < 0.293 and Q.tau21 > 0.43:
        z += -15.5 * (0.293 - Q.max_dr) * (Q.tau21 - 0.43)
    if Q.n_dr_0p2_0p4 < 5.19 and Q.dr_0 < 0.042:
        z += -5.37 * (5.19 - Q.n_dr_0p2_0p4) * (0.042 - Q.dr_0)
    if Q.n_dr_0p2_0p4 < 5.16 and Q.sum_pt_top30 < 991.0:
        z += -0.00143 * (5.16 - Q.n_dr_0p2_0p4) * (991.0 - Q.sum_pt_top30)
    if Q.n_dr_0p2_0p4 < 4.94 and Q.z_top50_slots > 0.979:
        z += 10.6 * (4.94 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.979)
    if Q.n_particles < 45.4 and Q.mass_top30 > 73.0:
        z += -0.00155 * (45.4 - Q.n_particles) * (Q.mass_top30 - 73.0)
    if Q.n_particles < 46.1 and Q.sum_pt_top40 > 841.0:
        z += 0.000131 * (46.1 - Q.n_particles) * (Q.sum_pt_top40 - 841.0)
    if Q.tau21 < 0.424 and Q.lam1 < 0.0159:
        z += -255.0 * (0.424 - Q.tau21) * (0.0159 - Q.lam1)
    return max(0.0, z)


def neuron_4(Q):
    z = 1.87
    if Q.C2 >= 0.11:
        z += 15.4 * Q.C2 - 1.694
    if Q.D2 < 6.92:
        z += -0.185 * Q.D2 + 1.2802
    if Q.LHA >= 0.114:
        z += -7.88 * Q.LHA + 0.89832
    if Q.e2 >= 0.0369:
        z += 40.8 * Q.e2 - 1.50552
    if Q.girth < 0.0609:
        z += 46.8 * Q.girth - 4.07015
    if 0.0609 <= Q.girth < 0.121:
        z += 20.3 * Q.girth - 2.4563
    if Q.girth2_top15 < 0.00725:
        z += 59.7 * Q.girth2_top15 + 0.1925699999999999
    if 0.00725 <= Q.girth2_top15 < 0.0159:
        z += -72.3 * Q.girth2_top15 + 1.14957
    if Q.girth2_top40 < 0.0129:
        z += -73.6 * Q.girth2_top40 + 0.94944
    if Q.lam1 >= 0.00757:
        z += -158.0 * Q.lam1 + 1.1960600000000001
    if Q.lam2 >= 0.00239:
        z += -173.0 * Q.lam2 + 0.41347000000000006
    if Q.mass < 80.4:
        z += 0.0773 * Q.mass - 4.927300000000001
    if 80.4 <= Q.mass < 101.0:
        z += -0.0507 * Q.mass + 5.363900000000001
    if 101.0 <= Q.mass < 120.0:
        z += -0.0128 * Q.mass + 1.536
    if Q.mass >= 143.0:
        z += 0.0148 * Q.mass - 2.1164
    if Q.mass_top20 >= 106.0:
        z += -0.0159 * Q.mass_top20 + 1.6854
    if Q.mass_top40 < 95.9:
        z += 0.017 * Q.mass_top40 - 1.6303000000000003
    if Q.n_dr_0p2_0p4 < 15.4:
        z += -0.0421 * Q.n_dr_0p2_0p4 + 0.64834
    if Q.n_particles >= 28.2:
        z += -0.013 * Q.n_particles + 0.3666
    if Q.planar_flow < 0.425:
        z += -0.978 * Q.planar_flow + 0.41564999999999996
    if Q.sum_pt < 1070.0:
        z += 0.00542 * Q.sum_pt - 5.7994
    if Q.tau21 < 0.465:
        z += 1.24 * Q.tau21 - 0.5766
    if Q.tau32 < 0.577:
        z += -2.23 * Q.tau32 + 1.2867099999999998
    if Q.width >= 0.00971:
        z += 163.0 * Q.width - 1.58273
    if Q.z_dr_0p2_0p4 < 0.069:
        z += 9.21 * Q.z_dr_0p2_0p4 - 0.49924000000000013
    if 0.069 <= Q.z_dr_0p2_0p4 < 0.194:
        z += -1.09 * Q.z_dr_0p2_0p4 + 0.21146
    if Q.z_top5_slots < 0.548:
        z += -0.793 * Q.z_top5_slots + 0.43456400000000006
    if Q.z_top5_slots >= 0.569:
        z += -0.777 * Q.z_top5_slots + 0.442113
    if Q.mass < 74.4 and Q.max_dr < 0.388:
        z += -0.535 * (74.4 - Q.mass) * (0.388 - Q.max_dr)
    if Q.mass < 122.0 and Q.max_dr < 0.403:
        z += 0.141 * (122.0 - Q.mass) * (0.403 - Q.max_dr)
    if Q.mass_top40 < 84.3 and Q.D2 < 6.61:
        z += -0.0144 * (84.3 - Q.mass_top40) * (6.61 - Q.D2)
    if Q.n_dr_0p2_0p4 < 14.6 and Q.max_dr < 0.436:
        z += -0.209 * (14.6 - Q.n_dr_0p2_0p4) * (0.436 - Q.max_dr)
    if Q.n_dr_0p2_0p4 < 15.2 and Q.n_dr_0p1_0p2 > 11.9:
        z += -0.00122 * (15.2 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 11.9)
    if Q.planar_flow < 0.303 and Q.n_dr_0p05_0p1 > 6.21:
        z += -0.104 * (0.303 - Q.planar_flow) * (Q.n_dr_0p05_0p1 - 6.21)
    return max(0.0, z)


def neuron_5(Q):
    z = -1.06
    if 6.85 <= Q.log_sum_pt < 6.91:
        z += 12.7 * Q.log_sum_pt - 86.99499999999999
    if 6.91 <= Q.log_sum_pt < 7.14:
        z += -20.2 * Q.log_sum_pt + 140.344
    if Q.log_sum_pt >= 7.14:
        z += -8.799999999999999 * Q.log_sum_pt + 58.94799999999999
    if 61.6 <= Q.mass < 172.8:
        z += 0.0191 * Q.mass - 1.17656
    if Q.mass >= 172.8:
        z += -0.0879 * Q.mass + 17.31304
    if Q.mass_over_sum_pt >= 0.0895:
        z += -8.94 * Q.mass_over_sum_pt + 0.8001299999999999
    if Q.mass_over_sum_pt_sq >= 0.0292:
        z += -444.0 * Q.mass_over_sum_pt_sq + 12.9648
    if Q.mass_top10 < 62.5:
        z += -0.00542 * Q.mass_top10 + 0.33875
    if 68.9 <= Q.mass_top30 < 107.0:
        z += 0.0262 * Q.mass_top30 - 1.8051800000000002
    if Q.mass_top30 >= 107.0:
        z += 0.0019000000000000024 * Q.mass_top30 + 0.7949199999999996
    if 90.5 <= Q.mass_top50 < 156.0:
        z += -0.0205 * Q.mass_top50 + 1.85525
    if Q.mass_top50 >= 156.0:
        z += -0.1865 * Q.mass_top50 + 27.751250000000002
    if Q.max_dr >= 0.435:
        z += 20.4 * Q.max_dr - 8.873999999999999
    if 840.0 <= Q.sum_pt_top10 < 994.0:
        z += 0.00335 * Q.sum_pt_top10 - 2.814
    if Q.sum_pt_top10 >= 994.0:
        z += 0.00018000000000000004 * Q.sum_pt_top10 + 0.33698000000000006
    if Q.sum_pt_top3 >= 794.0:
        z += -0.0031 * Q.sum_pt_top3 + 2.4614
    if Q.sum_pt_top40 < 905.0:
        z += -0.0224 * Q.sum_pt_top40 + 22.624
    if 905.0 <= Q.sum_pt_top40 < 1010.0:
        z += -0.0081 * Q.sum_pt_top40 + 9.6825
    if Q.sum_pt_top40 >= 1010.0:
        z += -0.0052 * Q.sum_pt_top40 + 6.753500000000001
    if Q.sum_pt_top50 < 1020.0:
        z += 0.0216 * Q.sum_pt_top50 - 22.032
    if 1020.0 <= Q.sum_pt_top50 < 1060.0:
        z += 0.0115 * Q.sum_pt_top50 - 11.73
    if Q.sum_pt_top50 >= 1060.0:
        z += 0.00541 * Q.sum_pt_top50 - 5.2746
    if Q.z_top30_slots < 0.864:
        z += 7.78 * Q.z_top30_slots - 6.72192
    if Q.z_top30_slots >= 0.933:
        z += -5.52 * Q.z_top30_slots + 5.15016
    if Q.girth2_top30 < 0.0186 and Q.tau32 < 0.866:
        z += 92.7 * (0.0186 - Q.girth2_top30) * (0.866 - Q.tau32)
    if Q.mass_top50 > 157.0 and Q.z_dr_0p05_0p1 > 0.589:
        z += 1.55 * (Q.mass_top50 - 157.0) * (Q.z_dr_0p05_0p1 - 0.589)
    if Q.max_dr > 0.278 and Q.tau21 < 0.582:
        z += -2.61 * (Q.max_dr - 0.278) * (0.582 - Q.tau21)
    if Q.n_particles < 61.7 and Q.e2 < 0.0374:
        z += 1.27 * (61.7 - Q.n_particles) * (0.0374 - Q.e2)
    if Q.n_particles < 64.0 and Q.mass_top5 > 11.1:
        z += -0.000265 * (64.0 - Q.n_particles) * (Q.mass_top5 - 11.1)
    if Q.sum_pt_top3 > 318.0 and Q.n_dr_0p1_0p2 > 7.87:
        z += 7.63e-05 * (Q.sum_pt_top3 - 318.0) * (Q.n_dr_0p1_0p2 - 7.87)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.607
    if Q.e2 < 0.0481:
        z += -15.7 * Q.e2 + 0.7551699999999999
    if Q.e2 >= 0.0555:
        z += -67.5 * Q.e2 + 3.74625
    if Q.lam1 < 0.00743:
        z += -137.0 * Q.lam1 + 1.01791
    if 0.00815 <= Q.lam1 < 0.0237:
        z += 116.0 * Q.lam1 - 0.9453999999999999
    if Q.lam1 >= 0.0237:
        z += 210.1 * Q.lam1 - 3.1755699999999996
    if Q.log_sum_pt < 6.83:
        z += -6.54 * Q.log_sum_pt + 44.6682
    if Q.m012 >= 20.4:
        z += -0.0113 * Q.m012 + 0.23051999999999997
    if Q.mass < 86.1:
        z += -0.005800000000000001 * Q.mass - 0.06159999999999943
    if 86.1 <= Q.mass < 91.2:
        z += 0.0238 * Q.mass - 2.6101599999999996
    if 91.2 <= Q.mass < 101.0:
        z += 0.078 * Q.mass - 7.5531999999999995
    if 101.0 <= Q.mass < 120.0:
        z += 0.0232 * Q.mass - 2.0183999999999997
    if 120.0 <= Q.mass < 172.8:
        z += -0.0145 * Q.mass + 2.5056000000000003
    if Q.mass_over_sum_pt >= 0.0513:
        z += -30.5 * Q.mass_over_sum_pt + 1.5646499999999999
    if Q.mass_top10 < 85.3:
        z += -0.00745 * Q.mass_top10 + 0.635485
    if Q.mass_top50 < 71.7:
        z += -0.0354 * Q.mass_top50 + 2.53818
    if Q.n_dr_0p2_0p4 < 16.1:
        z += 0.0459 * Q.n_dr_0p2_0p4 - 0.7389900000000001
    if Q.sum_pt_top10 >= 625.0:
        z += 0.000488 * Q.sum_pt_top10 - 0.305
    if Q.width < 0.00959:
        z += 177.0 * Q.width - 1.69743
    if Q.e2 > 0.0547 and Q.max_dr < 0.366:
        z += -1540.0 * (Q.e2 - 0.0547) * (0.366 - Q.max_dr)
    if Q.girth2_top20 > 0.00238 and Q.max_dr < 0.425:
        z += 357.0 * (Q.girth2_top20 - 0.00238) * (0.425 - Q.max_dr)
    if Q.girth2_top3 > 0.00959 and Q.D2 < 5.09:
        z += 20.3 * (Q.girth2_top3 - 0.00959) * (5.09 - Q.D2)
    if Q.mass_over_sum_pt > 0.0481 and Q.lam2 < 0.00253:
        z += 5870.0 * (Q.mass_over_sum_pt - 0.0481) * (0.00253 - Q.lam2)
    if Q.mass_over_sum_pt > 0.0497 and Q.n_dr_0p1_0p2 < 17.3:
        z += -1.11 * (Q.mass_over_sum_pt - 0.0497) * (17.3 - Q.n_dr_0p1_0p2)
    if Q.mass_over_sum_pt > 0.0528 and Q.n_dr_0p2_0p4 < 19.0:
        z += 0.954 * (Q.mass_over_sum_pt - 0.0528) * (19.0 - Q.n_dr_0p2_0p4)
    if Q.mass_over_sum_pt > 0.0403 and Q.z_dr_0_0p05 < 0.945:
        z += 8.95 * (Q.mass_over_sum_pt - 0.0403) * (0.945 - Q.z_dr_0_0p05)
    if Q.mass_over_sum_pt > 0.0519 and Q.z_dr_0p1_0p2 < 0.282:
        z += 44.8 * (Q.mass_over_sum_pt - 0.0519) * (0.282 - Q.z_dr_0p1_0p2)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.238
    if Q.C2 < 0.0562:
        z += 10.9 * Q.C2 - 0.61258
    if Q.e2 < 0.0275:
        z += -47.0 * Q.e2 + 1.2925
    if Q.girth < 0.0858:
        z += 16.3 * Q.girth - 1.3985400000000001
    if Q.girth2_top20 < 0.00639:
        z += 188.0 * Q.girth2_top20 - 1.9343099999999998
    if 0.00639 <= Q.girth2_top20 < 0.00798:
        z += 461.0 * Q.girth2_top20 - 3.6787799999999997
    if Q.girth2_top40 < 0.00801:
        z += 78.0 * Q.girth2_top40 + 0.50082
    if 0.00801 <= Q.girth2_top40 < 0.0127:
        z += -240.0 * Q.girth2_top40 + 3.048
    if Q.mass < 80.4:
        z += 0.1182 * Q.mass - 10.320559999999999
    if 80.4 <= Q.mass < 91.2:
        z += 0.1548 * Q.mass - 13.2632
    if 91.2 <= Q.mass < 101.0:
        z += -0.0872 * Q.mass + 8.8072
    if Q.mass_top20 < 66.0:
        z += -0.0263 * Q.mass_top20 + 1.7358
    if Q.max_dr < 0.196:
        z += 8.83 * Q.max_dr - 1.73068
    if Q.n_dr_0p2_0p4 < 1.99:
        z += 0.288 * Q.n_dr_0p2_0p4 - 0.57312
    if Q.width < 0.00943:
        z += -344.0 * Q.width + 3.2439199999999997
    if Q.z_dr_0p2_0p4 < 0.00474:
        z += -164.06 * Q.z_dr_0p2_0p4 - 0.011399999999999855
    if 0.00474 <= Q.z_dr_0p2_0p4 < 0.093:
        z += 8.94 * Q.z_dr_0p2_0p4 - 0.8314199999999999
    if Q.z_top50_slots < 0.991:
        z += 36.2 * Q.z_top50_slots - 35.8742
    if Q.D2 < 1.77 and Q.girth2_top50 < 0.00613:
        z += 659.0 * (1.77 - Q.D2) * (0.00613 - Q.girth2_top50)
    if Q.D2 < 1.79 and Q.girth2_top50 < 0.00789:
        z += -504.0 * (1.79 - Q.D2) * (0.00789 - Q.girth2_top50)
    if Q.D2 < 1.82 and Q.n_dr_0p2_0p4 < 9.16:
        z += 0.136 * (1.82 - Q.D2) * (9.16 - Q.n_dr_0p2_0p4)
    if Q.mass < 101.0 and Q.D2 < 1.61:
        z += 0.362 * (101.0 - Q.mass) * (1.61 - Q.D2)
    if Q.mass < 91.2 and Q.max_dr < 0.392:
        z += -1.32 * (91.2 - Q.mass) * (0.392 - Q.max_dr)
    if Q.mass < 101.0 and Q.max_dr < 0.391:
        z += 0.905 * (101.0 - Q.mass) * (0.391 - Q.max_dr)
    if Q.mass < 91.2 and Q.pt_dispersion < 0.34:
        z += -0.12 * (91.2 - Q.mass) * (0.34 - Q.pt_dispersion)
    if Q.mass < 91.2 and Q.z_dr_0p05_0p1 > 0.393:
        z += -0.127 * (91.2 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.393)
    if Q.mass_top30 < 76.1 and Q.D2 < 1.61:
        z += 0.21 * (76.1 - Q.mass_top30) * (1.61 - Q.D2)
    if Q.mass_top50 < 98.1 and Q.D2 < 1.61:
        z += -0.467 * (98.1 - Q.mass_top50) * (1.61 - Q.D2)
    if Q.n_dr_0p2_0p4 < 12.9 and Q.planar_flow < 0.549:
        z += 0.0605 * (12.9 - Q.n_dr_0p2_0p4) * (0.549 - Q.planar_flow)
    if Q.n_dr_0p2_0p4 < 13.0 and Q.z_1st < 0.504:
        z += 0.14 * (13.0 - Q.n_dr_0p2_0p4) * (0.504 - Q.z_1st)
    return max(0.0, z)


def neuron_8(Q):
    z = 0.672
    if Q.C2 >= 0.0718:
        z += 4.39 * Q.C2 - 0.315202
    if Q.D2 < 1.81:
        z += -0.305 * Q.D2 + 0.55205
    if Q.e2 >= 0.0256:
        z += -20.9 * Q.e2 + 0.53504
    if Q.girth < 0.0968:
        z += -24.6 * Q.girth + 2.3812800000000003
    if Q.girth2_top20 >= 0.00813:
        z += 130.0 * Q.girth2_top20 - 1.0569
    if Q.girth2_top30 < 0.00763:
        z += -162.0 * Q.girth2_top30 + 1.23606
    if Q.lam2 < 0.00149:
        z += 585.0 * Q.lam2 - 0.87165
    if Q.mass < 64.6:
        z += -0.0334 * Q.mass + 2.15764
    if 80.4 <= Q.mass < 91.2:
        z += 0.058 * Q.mass - 4.663200000000001
    if 91.2 <= Q.mass < 101.0:
        z += 0.1057 * Q.mass - 9.013440000000001
    if 101.0 <= Q.mass < 140.0:
        z += -0.0012999999999999956 * Q.mass + 1.7935599999999994
    if Q.mass >= 140.0:
        z += -0.0382 * Q.mass + 6.95956
    if Q.mass_over_sum_pt < 0.089:
        z += 46.7 * Q.mass_over_sum_pt - 4.6139600000000005
    if 0.089 <= Q.mass_over_sum_pt < 0.0988:
        z += 80.7 * Q.mass_over_sum_pt - 7.63996
    if Q.mass_over_sum_pt >= 0.0988:
        z += 34.0 * Q.mass_over_sum_pt - 3.026
    if Q.mass_over_sum_pt_sq < 0.0135:
        z += 70.9 * Q.mass_over_sum_pt_sq - 0.9571500000000001
    if 82.6 <= Q.mass_top50 < 98.1:
        z += -0.0441 * Q.mass_top50 + 3.64266
    if Q.mass_top50 >= 98.1:
        z += -0.008600000000000003 * Q.mass_top50 + 0.16011000000000042
    if Q.n_dr_0p2_0p4 < 20.5:
        z += 0.0608 * Q.n_dr_0p2_0p4 - 1.2464
    if Q.n_particles >= 50.7:
        z += 0.0675 * Q.n_particles - 3.4222500000000005
    if Q.sum_pt < 1010.0:
        z += -0.015 * Q.sum_pt + 15.149999999999999
    if Q.sum_pt_top40 < 995.0:
        z += -0.00444 * Q.sum_pt_top40 + 4.417800000000001
    if Q.sum_pt_top40 >= 1060.0:
        z += 0.0013 * Q.sum_pt_top40 - 1.378
    if Q.sum_pt_top50 < 1080.0:
        z += 0.0021 * Q.sum_pt_top50 - 2.268
    if Q.z_dr_0p1_0p2 < 0.144:
        z += -1.52 * Q.z_dr_0p1_0p2 + 0.21888
    if Q.z_dr_0p2_0p4 < 0.0913:
        z += -6.2 * Q.z_dr_0p2_0p4 + 0.56606
    if Q.girth2_top40 > 0.00516 and Q.sum_pt_top30 < 912.0:
        z += 0.71 * (Q.girth2_top40 - 0.00516) * (912.0 - Q.sum_pt_top30)
    if Q.lam2 < 0.00142 and Q.n_dr_0p05_0p1 > 4.52:
        z += 17.5 * (0.00142 - Q.lam2) * (Q.n_dr_0p05_0p1 - 4.52)
    if Q.mass > 85.7 and Q.log_sum_pt < 6.91:
        z += 0.112 * (Q.mass - 85.7) * (6.91 - Q.log_sum_pt)
    if Q.mass_over_sum_pt > 0.0754 and Q.sum_pt < 1110.0:
        z += 0.355 * (Q.mass_over_sum_pt - 0.0754) * (1110.0 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 15.9 and Q.z_dr_0p1_0p2 > 0.3:
        z += 0.15 * (15.9 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.3)
    if Q.n_particles > 51.1 and Q.z_top50_slots > 0.971:
        z += -2.94 * (Q.n_particles - 51.1) * (Q.z_top50_slots - 0.971)
    if Q.sum_pt < 1010.0 and Q.n_real_top50 < 43.5:
        z += -0.00119 * (1010.0 - Q.sum_pt) * (43.5 - Q.n_real_top50)
    if Q.sum_pt_top40 < 967.0 and Q.log_sum_pt < 6.82:
        z += 0.0683 * (967.0 - Q.sum_pt_top40) * (6.82 - Q.log_sum_pt)
    if Q.sum_pt_top40 < 1010.0 and Q.max_dr < 0.392:
        z += -0.0845 * (1010.0 - Q.sum_pt_top40) * (0.392 - Q.max_dr)
    if Q.z_dr_0p1_0p2 > 0.212 and Q.mean_phi > 0.000543:
        z += -1950.0 * (Q.z_dr_0p1_0p2 - 0.212) * (Q.mean_phi - 0.000543)
    return max(0.0, z)


def neuron_9(Q):
    z = -0.968
    if Q.LHA < 0.21:
        z += -14.3 * Q.LHA + 3.003
    if Q.e2 >= 0.0659:
        z += -45.4 * Q.e2 + 2.99186
    if Q.girth < 0.0277:
        z += 97.30000000000001 * Q.girth - 3.81227
    if 0.0277 <= Q.girth < 0.0435:
        z += 70.7 * Q.girth - 3.07545
    if Q.girth >= 0.0981:
        z += 12.6 * Q.girth - 1.2360600000000002
    if Q.girth2_top15 < 0.0208:
        z += 74.4 * Q.girth2_top15 - 1.54752
    if Q.girth2_top40 < 0.00603:
        z += -346.0 * Q.girth2_top40 + 2.08638
    if Q.lam2 >= 0.00153:
        z += -90.9 * Q.lam2 + 0.139077
    if Q.log_sum_pt < 6.85:
        z += 17.5 * Q.log_sum_pt - 119.875
    if 63.0 <= Q.mass < 80.4:
        z += -0.0528 * Q.mass + 3.3264
    if 80.4 <= Q.mass < 145.0:
        z += 0.044 * Q.mass - 4.45632
    if Q.mass >= 145.0:
        z += -0.0097 * Q.mass + 3.3301799999999995
    if Q.mass_top50 < 135.0:
        z += -0.0408 * Q.mass_top50 + 5.508
    if Q.n_dr_0p1_0p2 >= 21.2:
        z += -0.0271 * Q.n_dr_0p1_0p2 + 0.5745199999999999
    if Q.sum_pt < 951.0:
        z += -0.0203 * Q.sum_pt + 19.3053
    if Q.sum_pt_top40 < 965.0:
        z += -0.00283 * Q.sum_pt_top40 + 2.73095
    if Q.width >= 0.0258:
        z += -240.0 * Q.width + 6.192
    if Q.z_top50_slots < 0.973:
        z += 34.8 * Q.z_top50_slots - 33.8604
    if Q.girth2_top15 < 0.0183 and Q.n_particles > 40.2:
        z += 1.25 * (0.0183 - Q.girth2_top15) * (Q.n_particles - 40.2)
    if Q.girth2_top40 < 0.00614 and Q.sum_pt < 1020.0:
        z += 1.71 * (0.00614 - Q.girth2_top40) * (1020.0 - Q.sum_pt)
    if Q.log_sum_pt < 6.85 and Q.dr_7 < 0.0782:
        z += 159.0 * (6.85 - Q.log_sum_pt) * (0.0782 - Q.dr_7)
    if Q.log_sum_pt < 6.85 and Q.z_top20_slots > 0.898:
        z += 172.0 * (6.85 - Q.log_sum_pt) * (Q.z_top20_slots - 0.898)
    if Q.mass_top40 < 120.0 and Q.max_dr > 0.38:
        z += 0.0525 * (120.0 - Q.mass_top40) * (Q.max_dr - 0.38)
    if Q.sum_pt_top40 < 965.0 and Q.z_top30_slots > 0.97:
        z += -0.275 * (965.0 - Q.sum_pt_top40) * (Q.z_top30_slots - 0.97)
    return max(0.0, z)


def neuron_10(Q):
    z = 4.78
    if Q.e2 < 0.0663:
        z += 24.8 * Q.e2 - 1.64424
    if Q.girth < 0.123:
        z += 6.56 * Q.girth - 0.8068799999999999
    if Q.girth2_top30 < 0.00533:
        z += -198.0 * Q.girth2_top30 + 1.05534
    if Q.girth2_top5 < 0.0253:
        z += 69.3 * Q.girth2_top5 - 1.7532899999999998
    if Q.lam1 >= 0.00223:
        z += -51.8 * Q.lam1 + 0.115514
    if Q.m012 < 47.4:
        z += 0.00238 * Q.m012 - 0.11281200000000001
    if Q.mass < 63.0:
        z += -0.0359 * Q.mass + 2.445640000000001
    if 63.0 <= Q.mass < 80.4:
        z += -0.0681 * Q.mass + 4.474240000000002
    if 80.4 <= Q.mass < 86.4:
        z += -0.0177 * Q.mass + 0.42208000000000023
    if 86.4 <= Q.mass < 121.0:
        z += 0.032 * Q.mass - 3.872
    if 144.0 <= Q.mass < 162.0:
        z += -0.107 * Q.mass + 15.408
    if Q.mass >= 162.0:
        z += -0.16620000000000001 * Q.mass + 24.9984
    if Q.mass_top10 < 81.3:
        z += 0.00811 * Q.mass_top10 - 0.6593429999999999
    if Q.mass_top10 >= 89.1:
        z += 0.00851 * Q.mass_top10 - 0.7582409999999999
    if Q.mass_top50 >= 137.0:
        z += 0.0871 * Q.mass_top50 - 11.932699999999999
    if Q.max_dr < 0.24:
        z += -5.030000000000001 * Q.max_dr + 0.7033799999999999
    if 0.24 <= Q.max_dr < 0.402:
        z += 3.11 * Q.max_dr - 1.25022
    if Q.n_dr_0p2_0p4 < 5.96:
        z += 0.1461 * Q.n_dr_0p2_0p4 - 1.311294
    if 5.96 <= Q.n_dr_0p2_0p4 < 13.1:
        z += 0.0617 * Q.n_dr_0p2_0p4 - 0.8082699999999999
    if Q.sum_pt_top10 < 671.0:
        z += -0.00093 * Q.sum_pt_top10 + 0.6240300000000001
    if Q.sum_pt_top10 >= 677.0:
        z += -0.00101 * Q.sum_pt_top10 + 0.68377
    if Q.sum_pt_top50 < 959.0:
        z += 0.00864 * Q.sum_pt_top50 - 8.28576
    if Q.z_dr_0_0p05 >= 0.767:
        z += -10.4 * Q.z_dr_0_0p05 + 7.976800000000001
    if Q.z_dr_0p2_0p4 < 0.089:
        z += -9.17 * Q.z_dr_0p2_0p4 + 0.8161299999999999
    if Q.dr_0 < 0.0642 and Q.n_dr_0p2_0p4 > 2.93:
        z += 0.885 * (0.0642 - Q.dr_0) * (Q.n_dr_0p2_0p4 - 2.93)
    if Q.e2 < 0.0672 and Q.z_dr_0p1_0p2 < 0.218:
        z += -37.5 * (0.0672 - Q.e2) * (0.218 - Q.z_dr_0p1_0p2)
    if Q.girth2_top5 < 0.0236 and Q.sum_pt_top3 < 655.0:
        z += 0.107 * (0.0236 - Q.girth2_top5) * (655.0 - Q.sum_pt_top3)
    if Q.lam1 > 0.00517 and Q.pt_dispersion > 0.276:
        z += -194.0 * (Q.lam1 - 0.00517) * (Q.pt_dispersion - 0.276)
    if Q.mass < 114.0 and Q.D2 < 2.16:
        z += -0.00614 * (114.0 - Q.mass) * (2.16 - Q.D2)
    if Q.mass < 120.0 and Q.tau21 < 0.464:
        z += 0.102 * (120.0 - Q.mass) * (0.464 - Q.tau21)
    if Q.mass_top50 > 162.0 and Q.D2 > -0.418:
        z += 0.012 * (Q.mass_top50 - 162.0) * (Q.D2 - -0.418)
    if Q.mass_top50 > 172.0 and Q.sum_pt < 1240.0:
        z += -0.000321 * (Q.mass_top50 - 172.0) * (1240.0 - Q.sum_pt)
    if Q.z_dr_0p2_0p4 < 0.0847 and Q.max_dr > 0.265:
        z += 11.6 * (0.0847 - Q.z_dr_0p2_0p4) * (Q.max_dr - 0.265)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.0372
    if Q.e2 < 0.0258:
        z += -72.1 * Q.e2 + 1.86018
    if Q.girth2_top30 < 0.00644:
        z += 306.0 * Q.girth2_top30 - 1.9706400000000002
    if Q.mass < 80.4:
        z += -0.0077999999999999944 * Q.mass + 1.3204799999999999
    if 80.4 <= Q.mass < 91.2:
        z += -0.0642 * Q.mass + 5.85504
    if Q.max_dr < 0.296:
        z += -3.27 * Q.max_dr + 0.96792
    if Q.z_dr_0_0p05 < 0.0985:
        z += -5.36 * Q.z_dr_0_0p05 + 0.5279600000000001
    if Q.z_dr_0p05_0p1 >= 0.851:
        z += 6.62 * Q.z_dr_0p05_0p1 - 5.63362
    if Q.z_dr_0p2_0p4 < 0.00633:
        z += -124.0 * Q.z_dr_0p2_0p4 + 0.78492
    if Q.z_top5_slots >= 0.545:
        z += -1.8 * Q.z_top5_slots + 0.9810000000000001
    if Q.mass < 80.4 and Q.D2 < 3.33:
        z += -0.0188 * (80.4 - Q.mass) * (3.33 - Q.D2)
    if Q.mass < 98.7 and Q.D2 < 1.3:
        z += 0.0326 * (98.7 - Q.mass) * (1.3 - Q.D2)
    if Q.mass < 103.0 and Q.mass_top10 > 45.1:
        z += 0.000681 * (103.0 - Q.mass) * (Q.mass_top10 - 45.1)
    if Q.mass < 98.9 and Q.n_dr_0p2_0p4 > 10.1:
        z += -0.00297 * (98.9 - Q.mass) * (Q.n_dr_0p2_0p4 - 10.1)
    if Q.mass < 106.0 and Q.planar_flow < 0.368:
        z += 0.0715 * (106.0 - Q.mass) * (0.368 - Q.planar_flow)
    if Q.mass < 80.4 and Q.sum_pt_top2 < 467.0:
        z += -6.77e-05 * (80.4 - Q.mass) * (467.0 - Q.sum_pt_top2)
    if Q.mass < 80.4 and Q.z_dr_0p2_0p4 < 0.034:
        z += -0.54 * (80.4 - Q.mass) * (0.034 - Q.z_dr_0p2_0p4)
    if Q.mass_top50 < 89.6 and Q.girth2_top15 < 0.00355:
        z += 4.37 * (89.6 - Q.mass_top50) * (0.00355 - Q.girth2_top15)
    if Q.n_dr_0p2_0p4 < 9.84 and Q.girth2 < 0.0057:
        z += -30.5 * (9.84 - Q.n_dr_0p2_0p4) * (0.0057 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 9.59 and Q.n_dr_0p1_0p2 < 22.0:
        z += 0.0051 * (9.59 - Q.n_dr_0p2_0p4) * (22.0 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 10.7 and Q.z_dr_0p05_0p1 > 0.593:
        z += -0.331 * (10.7 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p05_0p1 - 0.593)
    if Q.n_dr_0p2_0p4 < 8.15 and Q.z_dr_0p2_0p4 < 0.0608:
        z += 1.02 * (8.15 - Q.n_dr_0p2_0p4) * (0.0608 - Q.z_dr_0p2_0p4)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.119
    if Q.C2 >= 0.0576:
        z += -13.4 * Q.C2 + 0.77184
    if Q.LHA >= 0.4:
        z += 15.1 * Q.LHA - 6.04
    if Q.girth < 0.0488:
        z += 23.5 * Q.girth - 1.1468
    if Q.log_sum_pt < 6.86:
        z += 6.56 * Q.log_sum_pt - 45.001599999999996
    if Q.mass < 80.4:
        z += -0.113 * Q.mass + 9.0852
    if Q.mass_top20 >= 127.0:
        z += 0.0397 * Q.mass_top20 - 5.0419
    if Q.mass_top30 >= 137.0:
        z += 0.0354 * Q.mass_top30 - 4.8498
    if Q.mass_top50 < 60.1:
        z += 0.0457 * Q.mass_top50 - 2.7465699999999997
    if Q.planar_flow >= 0.266:
        z += -0.49 * Q.planar_flow + 0.13034
    if Q.z_dr_0_0p05 >= 0.918:
        z += -7.28 * Q.z_dr_0_0p05 + 6.68304
    if Q.LHA > 0.406 and Q.lam2 < 0.00368:
        z += 12700.0 * (Q.LHA - 0.406) * (0.00368 - Q.lam2)
    if Q.mass < 91.2 and Q.z_dr_0p05_0p1 > 0.32:
        z += 0.0581 * (91.2 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.32)
    if Q.mass < 84.7 and Q.z_dr_0p1_0p2 < 0.0652:
        z += -0.312 * (84.7 - Q.mass) * (0.0652 - Q.z_dr_0p1_0p2)
    if Q.mass_top20 > 127.0 and Q.C2 > 0.0598:
        z += -1.34 * (Q.mass_top20 - 127.0) * (Q.C2 - 0.0598)
    if Q.mass_top40 < 94.4 and Q.sum_pt_top2 < 471.0:
        z += -4.27e-05 * (94.4 - Q.mass_top40) * (471.0 - Q.sum_pt_top2)
    if Q.z_dr_0p1_0p2 > 0.685 and Q.min_pair_mass > 0.762:
        z += -1.78 * (Q.z_dr_0p1_0p2 - 0.685) * (Q.min_pair_mass - 0.762)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.37
    if Q.log_sum_pt < 6.81:
        z += -11.38 * Q.log_sum_pt + 76.25460000000001
    if 6.81 <= Q.log_sum_pt < 7.02:
        z += 5.92 * Q.log_sum_pt - 41.5584
    if 74.9 <= Q.mass < 143.0:
        z += 0.0247 * Q.mass - 1.85003
    if 143.0 <= Q.mass < 172.8:
        z += -0.1373 * Q.mass + 21.31597
    if Q.mass >= 172.8:
        z += 0.021699999999999997 * Q.mass - 6.159230000000001
    if Q.mass_over_sum_pt >= 0.171:
        z += 312.0 * Q.mass_over_sum_pt - 53.352000000000004
    if Q.mass_over_sum_pt_sq < 0.00947:
        z += -104.0 * Q.mass_over_sum_pt_sq + 0.98488
    if Q.mass_over_sum_pt_sq >= 0.0292:
        z += -815.0 * Q.mass_over_sum_pt_sq + 23.798000000000002
    if Q.mass_top10 >= 57.1:
        z += 0.0149 * Q.mass_top10 - 0.85079
    if 98.1 <= Q.mass_top50 < 137.0:
        z += -0.0332 * Q.mass_top50 + 3.25692
    if 137.0 <= Q.mass_top50 < 169.0:
        z += 0.056600000000000004 * Q.mass_top50 - 9.04568
    if Q.mass_top50 >= 169.0:
        z += 0.010800000000000004 * Q.mass_top50 - 1.305480000000001
    if Q.n_dr_0p2_0p4 < 3.96:
        z += 0.133 * Q.n_dr_0p2_0p4 - 0.52668
    if Q.n_particles < 46.0:
        z += 0.0133 * Q.n_particles - 0.61712
    if 46.0 <= Q.n_particles < 46.4:
        z += 0.026799999999999997 * Q.n_particles - 1.2381199999999999
    if Q.n_particles >= 46.4:
        z += 0.0135 * Q.n_particles - 0.621
    if Q.sum_pt < 1010.0:
        z += 0.03547 * Q.sum_pt - 36.2832
    if 1010.0 <= Q.sum_pt < 1060.0:
        z += 0.00917 * Q.sum_pt - 9.7202
    if Q.sum_pt >= 1250.0:
        z += -0.00161 * Q.sum_pt + 2.0125
    if Q.sum_pt_top40 < 1050.0:
        z += -0.00653 * Q.sum_pt_top40 + 6.8565000000000005
    if Q.sum_pt_top50 < 1010.0:
        z += -0.00998 * Q.sum_pt_top50 + 10.079799999999999
    if Q.mass > 161.0 and Q.D2 < 6.47:
        z += -0.0053 * (Q.mass - 161.0) * (6.47 - Q.D2)
    if Q.sum_pt > 1250.0 and Q.dr_7 < 0.0801:
        z += -0.0405 * (Q.sum_pt - 1250.0) * (0.0801 - Q.dr_7)
    if Q.sum_pt < 1070.0 and Q.tau21 > 0.241:
        z += 0.004 * (1070.0 - Q.sum_pt) * (Q.tau21 - 0.241)
    if Q.sum_pt_top40 < 1030.0 and Q.D2 < 4.94:
        z += -0.00116 * (1030.0 - Q.sum_pt_top40) * (4.94 - Q.D2)
    if Q.sum_pt_top50 < 953.0 and Q.D2 < 4.76:
        z += 0.00216 * (953.0 - Q.sum_pt_top50) * (4.76 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.205
    if Q.girth2_top50 < 0.00935:
        z += 207.0 * Q.girth2_top50 - 1.9354500000000001
    if Q.lam1 < 0.00626:
        z += -260.0 * Q.lam1 + 1.6276
    if Q.lam2 < 0.00238:
        z += -433.0 * Q.lam2 + 1.03054
    if Q.log_sum_pt >= 6.94:
        z += 5.53 * Q.log_sum_pt - 38.37820000000001
    if Q.m012 >= 36.2:
        z += -0.0106 * Q.m012 + 0.38372
    if Q.mass < 91.2:
        z += 0.10170000000000001 * Q.mass - 7.917600000000001
    if 91.2 <= Q.mass < 136.0:
        z += -0.0303 * Q.mass + 4.1208
    if Q.mass_over_sum_pt < 0.0907:
        z += 49.0 * Q.mass_over_sum_pt - 3.5511
    if 0.0907 <= Q.mass_over_sum_pt < 0.0984:
        z += -116.0 * Q.mass_over_sum_pt + 11.4144
    if Q.mass_top40 < 80.7:
        z += 0.0263 * Q.mass_top40 - 2.12241
    if Q.max_dr >= 0.249:
        z += -2.43 * Q.max_dr + 0.60507
    if Q.n_dr_0p2_0p4 < 21.4:
        z += -0.0186 * Q.n_dr_0p2_0p4 + 0.39803999999999995
    if Q.sum_pt_top50 >= 978.0:
        z += -0.00313 * Q.sum_pt_top50 + 3.06114
    if Q.z_top50_slots < 0.959:
        z += -89.6 * Q.z_top50_slots + 85.92639999999999
    if Q.girth2_top20 < 0.0163 and Q.tau21 < 0.637:
        z += -94.9 * (0.0163 - Q.girth2_top20) * (0.637 - Q.tau21)
    if Q.girth2_top20 < 0.0166 and Q.z_top50_slots > 0.969:
        z += -1310.0 * (0.0166 - Q.girth2_top20) * (Q.z_top50_slots - 0.969)
    if Q.lam1 < 0.00623 and Q.z_top50_slots > 0.987:
        z += 8090.0 * (0.00623 - Q.lam1) * (Q.z_top50_slots - 0.987)
    if Q.max_dr > 0.201 and Q.mass_top10 < 52.1:
        z += 0.0401 * (Q.max_dr - 0.201) * (52.1 - Q.mass_top10)
    if Q.n_dr_0p2_0p4 < 19.9 and Q.n_dr_0p1_0p2 < 21.3:
        z += -0.00112 * (19.9 - Q.n_dr_0p2_0p4) * (21.3 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_15(Q):
    z = -3.06
    if Q.girth < 0.0498:
        z += 29.619999999999997 * Q.girth - 0.8030999999999997
    if 0.0498 <= Q.girth < 0.0858:
        z += -9.18 * Q.girth + 1.12914
    if 0.0858 <= Q.girth < 0.123:
        z += -5.43 * Q.girth + 0.80739
    if Q.girth >= 0.123:
        z += 3.75 * Q.girth - 0.32175
    if Q.girth2_top5 < 0.00711:
        z += -68.9 * Q.girth2_top5 + 0.48987900000000006
    if Q.girth2_top50 < 0.0135:
        z += 245.0 * Q.girth2_top50 - 3.3075
    if Q.lam2 < 0.00212:
        z += -184.0 * Q.lam2 + 0.39008
    if Q.log_sum_pt < 7.02:
        z += -6.82 * Q.log_sum_pt + 47.8764
    if Q.mass_top10 >= 26.6:
        z += -0.00812 * Q.mass_top10 + 0.21599200000000002
    if Q.sum_pt < 1000.0:
        z += 0.00752 * Q.sum_pt - 7.52
    z += 0.00244 * Q.sum_pt_top30
    if Q.sum_pt_top50 >= 1120.0:
        z += -0.00213 * Q.sum_pt_top50 + 2.3856
    if Q.z_dr_0p05_0p1 >= 0.849:
        z += -6.44 * Q.z_dr_0p05_0p1 + 5.46756
    if Q.z_dr_0p1_0p2 < 0.177:
        z += -4.55 * Q.z_dr_0p1_0p2 + 0.8053499999999999
    if Q.z_dr_0p2_0p4 < 0.0195:
        z += 30.7 * Q.z_dr_0p2_0p4 - 0.59865
    if Q.girth < 0.051 and Q.z_dr_0p2_0p4 > 0.0672:
        z += -7980.0 * (0.051 - Q.girth) * (Q.z_dr_0p2_0p4 - 0.0672)
    if Q.girth2_top5 < 0.00599 and Q.max_dr < 0.332:
        z += -1640.0 * (0.00599 - Q.girth2_top5) * (0.332 - Q.max_dr)
    if Q.girth2_top5 < 0.00177 and Q.n_dr_0p2_0p4 > 2.31:
        z += -28.9 * (0.00177 - Q.girth2_top5) * (Q.n_dr_0p2_0p4 - 2.31)
    if Q.girth2_top5 < 0.00727 and Q.sum_pt_top40 < 975.0:
        z += 1.23 * (0.00727 - Q.girth2_top5) * (975.0 - Q.sum_pt_top40)
    if Q.girth2_top50 < 0.0128 and Q.z_dr_0p2_0p4 < 0.129:
        z += 1850.0 * (0.0128 - Q.girth2_top50) * (0.129 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 1000.0 and Q.tau32 < 0.664:
        z += -0.0344 * (1000.0 - Q.sum_pt) * (0.664 - Q.tau32)
    if Q.z_dr_0p1_0p2 < 0.117 and Q.planar_flow < 0.818:
        z += 5.76 * (0.117 - Q.z_dr_0p1_0p2) * (0.818 - Q.planar_flow)
    if Q.z_dr_0p1_0p2 < 0.127 and Q.sum_pt_top3 < 758.0:
        z += -0.00604 * (0.127 - Q.z_dr_0p1_0p2) * (758.0 - Q.sum_pt_top3)
    return max(0.0, z)


def jet_layer_4(Q):
    return [neuron_0(Q), neuron_1(Q), neuron_2(Q), neuron_3(Q), neuron_4(Q), neuron_5(Q), neuron_6(Q), neuron_7(Q), neuron_8(Q), neuron_9(Q), neuron_10(Q), neuron_11(Q), neuron_12(Q), neuron_13(Q), neuron_14(Q), neuron_15(Q)]


def logits(h):
    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]
    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(5)]


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
