"""JEDI-linear jet tagger, 64 particles, 3 features: the tuned formula (start), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.5% (the network: 81.1%); same class as the network for 92.5% of jets.

Quantities:
  Q.mass_over_sum_pt_sq    (jet mass / total pT) squared
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top2              mass of the 2 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.min_pair_mass          smallest pair mass among particles 0, 1, 2 [GeV]
  Q.m012                   mass of particles 0, 1 and 2 [GeV]
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_3                   pT of particle 3 [GeV]
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_0                    pT of particle 0 / total pT
  Q.z_4                    pT of particle 4 / total pT
  Q.z_top10_slots          pT share of the 10 hardest particles
  Q.z_top15_slots          pT share of the 15 hardest particles
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.z_1st                  largest pT share
  Q.pt1_dr01               pT1 · ΔR01
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_1                   ΔR of particle 1 from the jet axis
  Q.dr_2                   ΔR of particle 2 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.dr_6                   ΔR of particle 6 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.dr_8                   ΔR of particle 8 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_5                  Δη of particle 5
  Q.phi_0                  Δφ of particle 0
  Q.sum_pt_top10           total pT of the 10 hardest particles [GeV]
  Q.sum_pt_top15           total pT of the 15 hardest particles [GeV]
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top20           total pT of the 20 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top30           total pT of the 30 hardest particles [GeV]
  Q.sum_pt_top40           total pT of the 40 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.sum_pt_top50           total pT of the 50 hardest particles [GeV]
  Q.girth2_top10           pT-weighted mean ΔR² of the 10 hardest particles
  Q.girth2_top15           pT-weighted mean ΔR² of the 15 hardest particles
  Q.girth2_top2            pT-weighted mean ΔR² of the 2 hardest particles
  Q.girth2_top20           pT-weighted mean ΔR² of the 20 hardest particles
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.girth2_top30           pT-weighted mean ΔR² of the 30 hardest particles
  Q.girth2_top40           pT-weighted mean ΔR² of the 40 hardest particles
  Q.girth2_top5            pT-weighted mean ΔR² of the 5 hardest particles
  Q.girth2_top50           pT-weighted mean ΔR² of the 50 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_dr_0p4_up            number of particles with 0.4 ≤ ΔR < 10
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top2=mass_of(2),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        min_pair_mass=min(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        m012=math.sqrt(pair_mass(0, 1) ** 2 + pair_mass(0, 2) ** 2 + pair_mass(1, 2) ** 2),
        n_particles=len(real),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_2=pt[2],
        pt_3=pt[3],
        pt_4=pt[4],
        pt_6=pt[6],
        pt_7=pt[7],
        pt_9=pt[9],
        z_0=z[0],
        z_4=z[4],
        z_top10_slots=sum(pt[:10]) / tot,
        z_top15_slots=sum(pt[:15]) / tot,
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        z_1st=zs[0],
        pt1_dr01=pt[1] * math.sqrt(dist2(0, 1)),
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_1=dr[1] if pt[1] > 0 else 0.0,
        dr_2=dr[2] if pt[2] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        dr_6=dr[6] if pt[6] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        dr_8=dr[8] if pt[8] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_5=eta[5],
        phi_0=phi[0],
        sum_pt_top10=sum(pt[:10]),
        sum_pt_top15=sum(pt[:15]),
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top20=sum(pt[:20]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top30=sum(pt[:30]),
        sum_pt_top40=sum(pt[:40]),
        sum_pt_top5=sum(pt[:5]),
        sum_pt_top50=sum(pt[:50]),
        girth2_top10=sum(pt[i] * dr[i] ** 2 for i in range(10)) / max(sum(pt[:10]), 1e-9),
        girth2_top15=sum(pt[i] * dr[i] ** 2 for i in range(15)) / max(sum(pt[:15]), 1e-9),
        girth2_top2=sum(pt[i] * dr[i] ** 2 for i in range(2)) / max(sum(pt[:2]), 1e-9),
        girth2_top20=sum(pt[i] * dr[i] ** 2 for i in range(20)) / max(sum(pt[:20]), 1e-9),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        girth2_top30=sum(pt[i] * dr[i] ** 2 for i in range(30)) / max(sum(pt[:30]), 1e-9),
        girth2_top40=sum(pt[i] * dr[i] ** 2 for i in range(40)) / max(sum(pt[:40]), 1e-9),
        girth2_top5=sum(pt[i] * dr[i] ** 2 for i in range(5)) / max(sum(pt[:5]), 1e-9),
        girth2_top50=sum(pt[i] * dr[i] ** 2 for i in range(50)) / max(sum(pt[:50]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_dr_0p4_up=sum(1 for i in real if 0.4 <= dr[i] < 10),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
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
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 0.9641577005386353
    if 74.251806640625 <= Q.mass < 78.261818313599:
        z += -0.005671047139912844 * Q.mass + 0.42108549568267795
    if 78.261818313599 <= Q.mass < 91.034691238403:
        z += -0.1458570589311421 * Q.mass + 11.392297680595908
    if 91.034691238403 <= Q.mass < 92.85979309082:
        z += 0.04620628757402301 * Q.mass - 6.092129766716203
    if Q.mass >= 92.85979309082:
        z += -0.07635048078373075 * Q.mass + 5.288466384864364
    if Q.girth2_top20 < 0.005312783396:
        z += 125.71826171875 * Q.girth2_top20 - 0.8028194978954202
    if 0.005312783396 <= Q.girth2_top20 < 0.006043208873:
        z += 184.69454956054688 * Q.girth2_top20 - 1.1161477406990352
    if Q.sum_pt < 1012.672900390625:
        z += 0.0020122777204960585 * Q.sum_pt - 2.037779115606179
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.044591594487428665 * Q.n_dr_0p2_0p4 + 0.44591594487428665
    if Q.girth < 0.056600876898:
        z += 55.0811653137207 * Q.girth - 3.117642257320293
    if Q.mass_over_sum_pt_sq < 0.008184367501:
        z += -471.21429443359375 * Q.mass_over_sum_pt_sq + 3.8565909573689496
    if Q.lam1 < 0.005913554513:
        z += 192.1145477294922 * Q.lam1 - 1.1360798507386924
    if Q.girth2_top40 < 0.004278051991:
        z += 176.38170719146729 * Q.girth2_top40 - 1.1243819438247178
    if 0.004278051991 <= Q.girth2_top40 < 0.00625977218:
        z += 329.196816444397 * Q.girth2_top40 - 1.7781329262190961
    if 0.00625977218 <= Q.girth2_top40 < 0.028623861071:
        z += -12.634726524353027 * Q.girth2_top40 + 0.36165465670315977
    if Q.z_top30_slots >= 0.920388080863:
        z += -3.6724488735198975 * Q.z_top30_slots + 3.3800781707664647
    if Q.width < 0.009614971338:
        z += 97.006591796875 * Q.width - 0.932715599724019
    if Q.n_particles < 62.0:
        z += -0.006433879490941763 * Q.n_particles + 0.3989005284383893
    if Q.LHA < 0.260146178237:
        z += -5.2993292808532715 * Q.LHA + 1.3786002596334082
    if Q.mass_top50 < 62.55:
        z += 0.03912674682214856 * Q.mass_top50 - 3.329812018094756
    if 62.55 <= Q.mass_top50 < 82.04491364955:
        z += 0.045264832675457 * Q.mass_top50 - 3.7137492882191987
    if Q.log_sum_pt < 7.017257672702:
        z += -3.224656343460083 * Q.log_sum_pt + 22.628244467972443
    if Q.girth2_top50 < 0.00742997247:
        z += -234.81227111816406 * Q.girth2_top50 + 1.744648710026135
    if Q.sum_pt_top30 >= 1073.4734375:
        z += -0.0012169495457783341 * Q.sum_pt_top30 + 1.306363012170732
    if Q.z_dr_0_0p05 >= 0.878906026483:
        z += -3.7124316692352295 * Q.z_dr_0_0p05 + 3.2628785669971863
    if 0.076966318366 <= Q.mass_over_sum_pt < 0.083299446175:
        z += -46.775394439697266 * Q.mass_over_sum_pt + 3.600129900140966
    if Q.mass_over_sum_pt >= 0.083299446175:
        z += 8.503196716308594 * Q.mass_over_sum_pt - 1.0045461284885748
    if Q.sum_pt_top20 >= 942.0:
        z += 0.001292187487706542 * Q.sum_pt_top20 - 1.2172406134195626
    if Q.mass_top20 < 70.421206773231:
        z += -0.0033102717716246843 * Q.mass_top20 + 0.2331133329051716
    if Q.sum_pt < 1012.672900390625 and Q.z_dr_0p2_0p4 < 0.193744690716:
        z += -0.030441781505942345 * (1012.672900390625 - Q.sum_pt) * (0.193744690716 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt < 6.989450376716 and Q.sum_pt_top40 > 956.21328125:
        z += 0.044196806848049164 * (6.989450376716 - Q.log_sum_pt) * (Q.sum_pt_top40 - 956.21328125)
    if Q.z_dr_0p2_0p4 < 0.037001823448 and Q.pt_2 < 126.75:
        z += 0.0647764727473259 * (0.037001823448 - Q.z_dr_0p2_0p4) * (126.75 - Q.pt_2)
    if Q.z_top30_slots > 0.920388080863 and Q.C2 > 0.061168736406:
        z += -75.85051727294922 * (Q.z_top30_slots - 0.920388080863) * (Q.C2 - 0.061168736406)
    if Q.girth2_top40 < 0.00625977218 and Q.girth2_top3 < 0.002915531053:
        z += 50659.41015625 * (0.00625977218 - Q.girth2_top40) * (0.002915531053 - Q.girth2_top3)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_top50_slots < 0.985099030959:
        z += -1.9847780466079712 * (10.0 - Q.n_dr_0p2_0p4) * (0.985099030959 - Q.z_top50_slots)
    if Q.sum_pt < 1012.672900390625 and Q.n_dr_0p1_0p2 > 11.0:
        z += 8.967310714069754e-05 * (1012.672900390625 - Q.sum_pt) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.mass_top50 < 82.04491364955 and Q.z_dr_0p05_0p1 < 0.212648361921:
        z += 0.061097972095012665 * (82.04491364955 - Q.mass_top50) * (0.212648361921 - Q.z_dr_0p05_0p1)
    if Q.girth2_top20 < 0.007538018543 and Q.girth2_top2 < 0.003111083776:
        z += -10651.3349609375 * (0.007538018543 - Q.girth2_top20) * (0.003111083776 - Q.girth2_top2)
    if Q.log_sum_pt < 6.989450376716 and Q.n_dr_0p1_0p2 > 11.0:
        z += -0.11242128163576126 * (6.989450376716 - Q.log_sum_pt) * (Q.n_dr_0p1_0p2 - 11.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_real_top30 < 30.0:
        z += -0.0021660448983311653 * (10.0 - Q.n_dr_0p2_0p4) * (30.0 - Q.n_real_top30)
    if Q.n_particles < 62.0 and Q.n_real_top30 < 30.0:
        z += 0.00043543317588046193 * (62.0 - Q.n_particles) * (30.0 - Q.n_real_top30)
    if Q.sum_pt < 1012.672900390625 and Q.z_dr_0p1_0p2 > 0.088715460151:
        z += -0.019372213631868362 * (1012.672900390625 - Q.sum_pt) * (Q.z_dr_0p1_0p2 - 0.088715460151)
    if Q.mass > 78.261818313599 and Q.n_dr_0p2_0p4 < 7.0:
        z += -0.008965743705630302 * (Q.mass - 78.261818313599) * (7.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.3683727979660034
    if Q.n_particles >= 38.0:
        z += 0.0217428058385849 * Q.n_particles - 0.8262266218662262
    if 6.811175180312 <= Q.log_sum_pt < 6.89371369877:
        z += -5.873547554016113 * Q.log_sum_pt + 40.00576132029681
    if 6.89371369877 <= Q.log_sum_pt < 6.910130970417:
        z += 13.622723579406738 * Q.log_sum_pt - 94.39595006711443
    if 6.910130970417 <= Q.log_sum_pt < 6.959293500649:
        z += 43.18209743499756 * Q.log_sum_pt - 298.6550948127671
    if 6.959293500649 <= Q.log_sum_pt < 6.989450376716:
        z += 39.662683725357056 * Q.log_sum_pt - 274.16246185717097
    if 6.989450376716 <= Q.log_sum_pt < 7.139296169016:
        z += 21.880958795547485 * Q.log_sum_pt - 149.87797784785317
    if Q.log_sum_pt >= 7.139296169016:
        z += 18.979734420776367 * Q.log_sum_pt - 129.1652777835939
    if Q.sum_pt_top50 >= 959.095727539062:
        z += -0.009640966542065144 * Q.sum_pt_top50 + 9.246609819841723
    if Q.mass_top20 < 40.2:
        z += 0.09015782922506332 * Q.mass_top20 - 3.6243447348475457
    if Q.sum_pt_top2 < 689.25:
        z += -0.002783391857519746 * Q.sum_pt_top2 + 1.9184528377954848
    if Q.z_top30_slots >= 0.934183811419:
        z += 14.107861518859863 * Q.z_top30_slots - 13.17933584465995
    if Q.sum_pt_top30 < 1073.4734375:
        z += -0.00449218787252903 * Q.sum_pt_top30 + 4.8222443574195495
    if Q.girth2_top15 < 0.003270031267:
        z += -154.9559783935547 * Q.girth2_top15 + 0.5067108943555002
    if Q.z_top20_slots < 0.957678701144:
        z += 9.544075965881348 * Q.z_top20_slots - 9.140158274624916
    if Q.max_dr < 0.43572281599:
        z += 7.371124267578125 * Q.max_dr - 3.211767022881367
    if Q.D2 < 1.230420708656:
        z += -0.5269099473953247 * Q.D2 + 0.6483209108720511
    if Q.lam2 < 0.000829637219:
        z += 701.3442993164062 * Q.lam2 - 0.5818613340463669
    if Q.z_top50_slots >= 0.958653609576:
        z += -37.942317962646484 * Q.z_top50_slots + 36.37354007057135
    if Q.mass_over_sum_pt < 0.074356165682:
        z += -16.890729904174805 * Q.mass_over_sum_pt + 1.2559299112447337
    if Q.pt_9 < 34.0625:
        z += 0.02043388970196247 * Q.pt_9 - 0.6960293679730967
    if Q.tau32 >= 0.329314215481:
        z += 2.0905697345733643 * Q.tau32 - 0.6884543320493498
    if Q.girth2_top3 < 0.000823693417:
        z += -549.2960815429688 * Q.girth2_top3 + 0.45245156635083855
    if Q.z_dr_0_0p05 >= 0.878906026483:
        z += -0.8691943287849426 * Q.z_dr_0_0p05 + 0.7639401337539322
    if Q.n_dr_0p2_0p4 < 13.0:
        z += 0.04733962565660477 * Q.n_dr_0p2_0p4 - 0.615415133535862
    if Q.girth2_top20 < 0.00115306291:
        z += -865.3212890625 * Q.girth2_top20 + 0.9977698836513575
    if Q.mass_top50 < 117.048742792994:
        z += 0.018300991505384445 * Q.mass_top50 - 2.142108047570512
    if Q.LHA < 0.404204003833:
        z += -5.996489524841309 * Q.LHA + 2.4238050748835005
    if Q.mass_top30 < 80.4:
        z += -0.05600827559828758 * Q.mass_top30 + 4.503065358102322
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.049080293625593185 * Q.n_dr_0_0p05 + 1.4724088087677956
    if Q.n_pt_above_10 < 31.0:
        z += 0.05530066043138504 * Q.n_pt_above_10 - 1.7143204733729362
    if Q.girth2_top10 < 0.00130722027:
        z += -430.63246154785156 * Q.girth2_top10 + 0.945324729309279
    if 0.00130722027 <= Q.girth2_top10 < 0.007678543663:
        z += -60.01786804199219 * Q.girth2_top10 + 0.46084982032060934
    if Q.girth2_top5 < 0.000657050184:
        z += -154.98184204101562 * Q.girth2_top5 + 0.10183084782970825
    if Q.sum_pt_top5 < 902.40625:
        z += -0.0015476963017135859 * Q.sum_pt_top5 + 1.3966508157682256
    if Q.n_particles > 38.0 and Q.mass_top15 < 57.873489696602:
        z += -0.0005233458359725773 * (Q.n_particles - 38.0) * (57.873489696602 - Q.mass_top15)
    if Q.sum_pt_top2 < 689.25 and Q.n_dr_0p2_0p4 < 7.0:
        z += -0.00023121880076359957 * (689.25 - Q.sum_pt_top2) * (7.0 - Q.n_dr_0p2_0p4)
    if Q.mass_top20 < 40.2 and Q.n_real_top40 > 26.0:
        z += 0.003448300063610077 * (40.2 - Q.mass_top20) * (Q.n_real_top40 - 26.0)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.144288109196:
        z += 0.26604416966438293 * (Q.n_particles - 38.0) * (0.144288109196 - Q.dr_0)
    if Q.z_top30_slots > 0.934183811419 and Q.m012 > 16.899120053094:
        z += 0.2578888237476349 * (Q.z_top30_slots - 0.934183811419) * (Q.m012 - 16.899120053094)
    if Q.n_particles > 38.0 and Q.z_top50_slots < 0.985099030959:
        z += 1.569084644317627 * (Q.n_particles - 38.0) * (0.985099030959 - Q.z_top50_slots)
    if Q.n_particles > 38.0 and Q.dr_1 < 0.161154452503:
        z += 0.11100641638040543 * (Q.n_particles - 38.0) * (0.161154452503 - Q.dr_1)
    if Q.max_dr < 0.43572281599 and Q.z_dr_0p2_0p4 < 0.193744690716:
        z += 28.102340698242188 * (0.43572281599 - Q.max_dr) * (0.193744690716 - Q.z_dr_0p2_0p4)
    if Q.mass_top20 < 40.2 and Q.mean_phi2 < 0.005860335776:
        z += 8.827327728271484 * (40.2 - Q.mass_top20) * (0.005860335776 - Q.mean_phi2)
    if Q.max_dr < 0.43572281599 and Q.n_real_top30 < 26.0:
        z += 0.2671559751033783 * (0.43572281599 - Q.max_dr) * (26.0 - Q.n_real_top30)
    if Q.z_top30_slots > 0.934183811419 and Q.dr_5 < 0.057912331642:
        z += -59.006526947021484 * (Q.z_top30_slots - 0.934183811419) * (0.057912331642 - Q.dr_5)
    if Q.n_dr_0p1_0p2 < 7.0 and Q.dr_6 < 0.079220479673:
        z += -1.51731276512146 * (7.0 - Q.n_dr_0p1_0p2) * (0.079220479673 - Q.dr_6)
    if Q.n_pt_above_10 < 31.0 and Q.mean_phi2 < 0.017422899418:
        z += 0.15219785273075104 * (31.0 - Q.n_pt_above_10) * (0.017422899418 - Q.mean_phi2)
    if Q.z_top30_slots > 0.934183811419 and Q.mass_top10 < 91.19:
        z += -0.3116796910762787 * (Q.z_top30_slots - 0.934183811419) * (91.19 - Q.mass_top10)
    if Q.mass_top30 < 80.4 and Q.mass_top5 < 68.434224049685:
        z += -0.0006446722545661032 * (80.4 - Q.mass_top30) * (68.434224049685 - Q.mass_top5)
    if Q.girth2_top3 < 0.000823693417 and Q.n_dr_0p05_0p1 < 10.0:
        z += -92.85115051269531 * (0.000823693417 - Q.girth2_top3) * (10.0 - Q.n_dr_0p05_0p1)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.21250717341899872
    if 6.903422848462 <= Q.log_sum_pt < 7.062574317998:
        z += 0.9164353609085083 * Q.log_sum_pt - 6.326540809634316
    if Q.log_sum_pt >= 7.062574317998:
        z += 13.754101872444153 * Q.log_sum_pt - 96.99351461702894
    if 995.676940917969 <= Q.sum_pt < 1017.43466796875:
        z += -0.0008079925901256502 * Q.sum_pt + 0.8044995904206937
    if 1017.43466796875 <= Q.sum_pt < 1066.481811523438:
        z += 0.007669442857149988 * Q.sum_pt - 7.820737129104707
    if Q.sum_pt >= 1066.481811523438:
        z += -0.0029239155701361597 * Q.sum_pt + 3.4768869565445026
    if Q.lam2 < 0.000973194699:
        z += 490.5626525878906 * Q.lam2 - 0.4774129730259138
    if Q.girth2_top15 < 0.006142801866:
        z += 12.58721923828125 * Q.girth2_top15 + 0.29048649793607834
    if 0.006142801866 <= Q.girth2_top15 < 0.015638355144:
        z += -38.73468780517578 * Q.girth2_top15 + 0.6057468042893047
    if Q.mass < 92.85979309082:
        z += 0.04024188220500946 * Q.mass - 3.73685285514233
    if Q.mass_top50 < 86.4:
        z += -0.01526182983070612 * Q.mass_top50 + 1.3186220973730087
    if Q.mass_over_sum_pt < 0.074356165682:
        z += -30.45398235321045 * Q.mass_over_sum_pt + 2.246493818621635
    if 0.074356165682 <= Q.mass_over_sum_pt < 0.09795414517:
        z += -9.812443733215332 * Q.mass_over_sum_pt + 0.7116681530618763
    if 0.09795414517 <= Q.mass_over_sum_pt < 0.140939019879:
        z += 5.804399490356445 * Q.mass_over_sum_pt - 0.8180663751570045
    if 908.8125 <= Q.sum_pt_top20 < 1129.275:
        z += 0.0009011756628751755 * Q.sum_pt_top20 - 0.8189997071167454
    if Q.sum_pt_top20 >= 1129.275:
        z += -0.002758948365226388 * Q.sum_pt_top20 + 3.314286854717648
    if 997.018872070312 <= Q.sum_pt_top50 < 1038.855053710938:
        z += -0.005135657265782356 * Q.sum_pt_top50 + 5.120347214470027
    if 1038.855053710938 <= Q.sum_pt_top50 < 1107.225842285156:
        z += -0.0008930754847824574 * Q.sum_pt_top50 + 0.7129196904963306
    if Q.sum_pt_top50 >= 1107.225842285156:
        z += -0.006252680439502001 * Q.sum_pt_top50 + 6.647212800801372
    if Q.max_dr < 0.402178311348:
        z += -0.5558584332466125 * Q.max_dr + 0.2235542060316676
    if Q.mass_top40 >= 160.8:
        z += -0.10577525943517685 * Q.mass_top40 + 17.008661717176437
    if Q.mass_top30 < 91.697531419407:
        z += -0.0038220733404159546 * Q.mass_top30 + 0.3504746902200699
    if Q.z_top30_slots >= 0.904849218002:
        z += 9.75528335571289 * Q.z_top30_slots - 8.827060515804735
    if Q.lam1 < 0.006189818106:
        z += 43.990848541259766 * Q.lam1 - 0.2722953507989934
    if 1013.04248046875 <= Q.sum_pt_top40 < 1069.67119140625:
        z += -0.0033827912993729115 * Q.sum_pt_top40 + 3.42691128882484
    if Q.sum_pt_top40 >= 1069.67119140625:
        z += 0.0022542085498571396 * Q.sum_pt_top40 - 2.6028250558579202
    if Q.sum_pt_top30 >= 1018.831689453125:
        z += -0.0021873388905078173 * Q.sum_pt_top30 + 2.2285301772226034
    if Q.log_sum_pt > 6.903422848462 and Q.mass_top50 > 85.866695580031:
        z += -0.033084992319345474 * (Q.log_sum_pt - 6.903422848462) * (Q.mass_top50 - 85.866695580031)
    if Q.log_sum_pt > 6.903422848462 and Q.girth2_top10 < 0.01976735495:
        z += 185.07083129882812 * (Q.log_sum_pt - 6.903422848462) * (0.01976735495 - Q.girth2_top10)
    if Q.log_sum_pt > 7.062574317998 and Q.mass_top20 > 66.841467317407:
        z += 0.19199173152446747 * (Q.log_sum_pt - 7.062574317998) * (Q.mass_top20 - 66.841467317407)
    if Q.mass < 92.85979309082 and Q.sum_pt_top40 < 1069.67119140625:
        z += -2.5451863621128723e-05 * (92.85979309082 - Q.mass) * (1069.67119140625 - Q.sum_pt_top40)
    if Q.sum_pt_top50 > 1107.225842285156 and Q.C2 > 0.097715596855:
        z += 0.8289306163787842 * (Q.sum_pt_top50 - 1107.225842285156) * (Q.C2 - 0.097715596855)
    if Q.sum_pt > 1066.481811523438 and Q.pt_6 < 36.8125:
        z += 6.44392566755414e-05 * (Q.sum_pt - 1066.481811523438) * (36.8125 - Q.pt_6)
    if Q.sum_pt_top50 > 997.018872070312 and Q.mean_phi > 0.000107912998:
        z += 3.560925245285034 * (Q.sum_pt_top50 - 997.018872070312) * (Q.mean_phi - 0.000107912998)
    if Q.sum_pt_top50 > 1038.855053710938 and Q.dr_3 > 0.004614387814:
        z += -0.020528189837932587 * (Q.sum_pt_top50 - 1038.855053710938) * (Q.dr_3 - 0.004614387814)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.041473388672:
        z += -0.12417422235012054 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.041473388672)
    if Q.sum_pt_top40 > 1013.04248046875 and Q.eta_5 < -0.113525390625:
        z += -0.07749517261981964 * (Q.sum_pt_top40 - 1013.04248046875) * (-0.113525390625 - Q.eta_5)
    if Q.log_sum_pt > 6.903422848462 and Q.n_dr_0_0p05 < 18.0:
        z += 0.092528335750103 * (Q.log_sum_pt - 6.903422848462) * (18.0 - Q.n_dr_0_0p05)
    if Q.n_pt_above_10 > 18.0 and Q.z_4 > 0.035928898346:
        z += 0.6857365965843201 * (Q.n_pt_above_10 - 18.0) * (Q.z_4 - 0.035928898346)
    if Q.sum_pt_top20 > 1129.275 and Q.girth2_top3 < 0.007929074034:
        z += 0.14633607864379883 * (Q.sum_pt_top20 - 1129.275) * (0.007929074034 - Q.girth2_top3)
    if Q.mass_top40 > 160.8 and Q.pt_9 < 41.4375:
        z += 0.00484287878498435 * (Q.mass_top40 - 160.8) * (41.4375 - Q.pt_9)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.1788763552904129
    if Q.lam2 < 0.000349717384:
        z += -1655.6107177734375 * Q.lam2 + 0.7410433134867516
    if 0.000349717384 <= Q.lam2 < 0.000615484055:
        z += -609.73583984375 * Q.lam2 + 0.37528268718586183
    if Q.n_particles < 46.0:
        z += -0.013178784400224686 * Q.n_particles + 0.6062240824103355
    if Q.mass_over_sum_pt < 0.050772907168:
        z += -9.90911865234375 * Q.mass_over_sum_pt + 0.5031147614521465
    if Q.z_top20_slots >= 0.923451750505:
        z += -1.2700728178024292 * Q.z_top20_slots + 1.172850966868471
    if Q.mass_over_sum_pt_sq < 0.00750911433:
        z += -204.22653198242188 * Q.mass_over_sum_pt_sq + 1.5335603778754074
    if Q.z_dr_0p2_0p4 < 0.001393458078:
        z += -249.1632537841797 * Q.z_dr_0p2_0p4 + 0.34719854872632927
    if Q.girth2_top5 < 0.000114064392:
        z += 7572.28173828125 * Q.girth2_top5 - 0.8637277125297539
    if Q.lam1 < 0.00767124277:
        z += 31.459392547607422 * Q.lam1 - 0.24133263762942533
    if Q.n_dr_0p1_0p2 < 15.0:
        z += -0.018495185300707817 * Q.n_dr_0p1_0p2 + 0.27742777951061726
    if Q.n_dr_0p2_0p4 < 7.0:
        z += 0.031265486031770706 * Q.n_dr_0p2_0p4 - 0.21885840222239494
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += -0.0015482043381780386 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_particles < 46.0 and Q.mass_top30 > 73.331387415761:
        z += -0.001401412533596158 * (46.0 - Q.n_particles) * (Q.mass_top30 - 73.331387415761)
    if Q.n_particles < 46.0 and Q.sum_pt_top40 > 858.826171875:
        z += 0.00012416481331456453 * (46.0 - Q.n_particles) * (Q.sum_pt_top40 - 858.826171875)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_top50_slots > 0.978741004761:
        z += 11.190045356750488 * (5.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741004761)
    if Q.max_dr < 0.298200035095 and Q.tau21 > 0.428145796061:
        z += -15.926551818847656 * (0.298200035095 - Q.max_dr) * (Q.tau21 - 0.428145796061)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.dr_0 < 0.04118638065:
        z += -5.1781511306762695 * (5.0 - Q.n_dr_0p2_0p4) * (0.04118638065 - Q.dr_0)
    if Q.mass_over_sum_pt_sq < 0.00750911433 and Q.mass_top20 < 136.785:
        z += -0.7532560229301453 * (0.00750911433 - Q.mass_over_sum_pt_sq) * (136.785 - Q.mass_top20)
    if Q.mass_over_sum_pt_sq < 0.00750911433 and Q.girth2_top15 > 0.004855288512:
        z += 422434.375 * (0.00750911433 - Q.mass_over_sum_pt_sq) * (Q.girth2_top15 - 0.004855288512)
    if Q.tau21 < 0.428145796061 and Q.lam1 < 0.016493544356:
        z += -226.9984893798828 * (0.428145796061 - Q.tau21) * (0.016493544356 - Q.lam1)
    if Q.n_particles < 46.0 and Q.eta_0 > -0.02893371582:
        z += -0.28822943568229675 * (46.0 - Q.n_particles) * (Q.eta_0 - -0.02893371582)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_dr_0p1_0p2 > 0.003353110817:
        z += 0.2789609432220459 * (5.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.003353110817)
    if Q.lam2 < 0.000615484055 and Q.n_dr_0p4_up < 1.0:
        z += 701.5462646484375 * (0.000615484055 - Q.lam2) * (1.0 - Q.n_dr_0p4_up)
    if Q.n_dr_0p2_0p4 < 7.0 and Q.phi_0 > -0.040130615234:
        z += -0.24274063110351562 * (7.0 - Q.n_dr_0p2_0p4) * (Q.phi_0 - -0.040130615234)
    return max(0.0, z)


def neuron_4(Q):
    z = 2.640839099884033
    if Q.mass_top15 < 69.027163795459:
        z += -0.0010466912062838674 * Q.mass_top15 + 0.07225012533942309
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.04051978141069412 * Q.n_dr_0p2_0p4 + 0.6077967211604118
    if Q.mass_top40 < 83.325535102591:
        z += 0.02094813622534275 * Q.mass_top40 - 2.0361757605264126
    if 83.325535102591 <= Q.mass_top40 < 94.642533639752:
        z += 0.025683585554361343 * Q.mass_top40 - 2.430759609818092
    if Q.tau32 < 0.577419030666:
        z += -2.2062933444976807 * Q.tau32 + 1.273955764344698
    if Q.mass_top30 < 60.438206617337:
        z += 0.003971825586631894 * Q.mass_top30 - 0.7813776833714602
    if 60.438206617337 <= Q.mass_top30 < 152.688263064041:
        z += 0.005868047010153532 * Q.mass_top30 - 0.8959819055584818
    if Q.mass < 74.251806640625:
        z += 0.07847955310717225 * Q.mass - 5.211959050328924
    if 74.251806640625 <= Q.mass < 80.784643554688:
        z += 0.0815303223207593 * Q.mass - 5.438484176081361
    if 80.784643554688 <= Q.mass < 86.4:
        z += -0.02855298202484846 * Q.mass + 3.454556326800799
    if 86.4 <= Q.mass < 101.049709320068:
        z += -0.05057378951460123 * Q.mass + 5.3571540939154385
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += -0.012618091888725758 * Q.mass + 1.5217418817803265
    if Q.mass >= 143.787612915039:
        z += 0.015059489756822586 * Q.mass - 2.165368083852001
    if Q.C2 >= 0.108888113871:
        z += 15.334700584411621 * Q.C2 - 1.6697666234131028
    if Q.girth2_top10 < 0.004752875822:
        z += 80.890380859375 * Q.girth2_top10 - 0.38446193541889506
    if Q.n_particles >= 29.0:
        z += -0.010857435874640942 * Q.n_particles + 0.3148656403645873
    if Q.pt_7 < 38.53125:
        z += 0.00846788939088583 * Q.pt_7 - 0.32627836309256963
    if Q.lam2 >= 0.002396991421:
        z += -164.73434448242188 * Q.lam2 + 0.39486681046842387
    if Q.D2 < 6.916121244431:
        z += -0.17280447483062744 * Q.D2 + 1.1951366995088446
    if Q.tau21 < 0.470341457427:
        z += 1.263122797012329 * Q.tau21 - 0.5940990172560475
    if Q.girth2_top40 < 0.012926423095:
        z += -66.6508560180664 * Q.girth2_top40 + 0.8615571645334533
    if Q.mass_top10 < 45.595:
        z += -0.0058156209997832775 * Q.mass_top10 + 0.26516323948511855
    if Q.mass_top20 >= 103.674910639856:
        z += -0.01594446413218975 * Q.mass_top20 + 1.6530408941051615
    if Q.girth2_top15 < 0.007277630044:
        z += 49.65040588378906 * Q.girth2_top15 + 0.24488851197971007
    if 0.007277630044 <= Q.girth2_top15 < 0.015638355144:
        z += -72.50875854492188 * Q.girth2_top15 + 1.1339177171760328
    if Q.z_dr_0p2_0p4 < 0.068491501734:
        z += 9.344733476638794 * Q.z_dr_0p2_0p4 - 0.4878758665047884
    if 0.068491501734 <= Q.z_dr_0p2_0p4 < 0.193744690716:
        z += -1.214811086654663 * Q.z_dr_0p2_0p4 + 0.23536319826227559
    if Q.girth < 0.061710142531:
        z += 49.477447509765625 * Q.girth - 4.359671784122936
    if 0.061710142531 <= Q.girth < 0.120745175332:
        z += 22.129426956176758 * Q.girth - 2.6720215378202496
    if Q.LHA >= 0.115200825015:
        z += -7.421395301818848 * Q.LHA + 0.8549508615319761
    if Q.e2 >= 0.036805817112:
        z += 40.14015579223633 * Q.e2 - 1.4773912329362378
    if 0.06030418859 <= Q.mass_over_sum_pt < 0.09795414517:
        z += -11.230770111083984 * Q.mass_over_sum_pt + 0.6772624787897439
    if Q.mass_over_sum_pt >= 0.09795414517:
        z += -16.747580528259277 * Q.mass_over_sum_pt + 1.217656927269101
    if Q.lam1 >= 0.00767124277:
        z += -147.73707580566406 * Q.lam1 + 1.1333269746351424
    if Q.width >= 0.009614971338:
        z += 191.3317108154297 * Q.width - 1.8396489155408609
    if Q.eccentricity >= 0.868081197276:
        z += 2.7047810554504395 * Q.eccentricity - 2.3479695769848603
    if Q.sum_pt < 1066.481811523438:
        z += 0.005733318626880646 * Q.sum_pt - 6.114480035236741
    if Q.sum_pt_top20 < 993.56640625:
        z += -0.0011213305406272411 * Q.sum_pt_top20 + 1.1141163554693776
    if Q.planar_flow < 0.303602636177 and Q.n_dr_0p05_0p1 > 6.0:
        z += -0.09230989962816238 * (0.303602636177 - Q.planar_flow) * (Q.n_dr_0p05_0p1 - 6.0)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.max_dr < 0.43572281599:
        z += -0.19577856361865997 * (15.0 - Q.n_dr_0p2_0p4) * (0.43572281599 - Q.max_dr)
    if Q.n_particles > 29.0 and Q.n_dr_0_0p05 < 25.0:
        z += -7.752858073217794e-05 * (Q.n_particles - 29.0) * (25.0 - Q.n_dr_0_0p05)
    if Q.mass_top40 < 83.325535102591 and Q.D2 < 6.916121244431:
        z += -0.0121529009193182 * (83.325535102591 - Q.mass_top40) * (6.916121244431 - Q.D2)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.n_dr_0p1_0p2 > 10.0:
        z += -0.0012470601359382272 * (15.0 - Q.n_dr_0p2_0p4) * (Q.n_dr_0p1_0p2 - 10.0)
    if Q.mass < 74.251806640625 and Q.max_dr < 0.39398368001:
        z += -0.6790133714675903 * (74.251806640625 - Q.mass) * (0.39398368001 - Q.max_dr)
    if Q.mass < 120.60000000000001 and Q.max_dr < 0.402178311348:
        z += 0.1447036862373352 * (120.60000000000001 - Q.mass) * (0.402178311348 - Q.max_dr)
    if Q.mass < 101.049709320068 and Q.girth2_top2 < 0.004007841607:
        z += 4.094479084014893 * (101.049709320068 - Q.mass) * (0.004007841607 - Q.girth2_top2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.5040539503097534
    if 0.090467494167 <= Q.mass_over_sum_pt < 0.09795414517:
        z += -32.775203704833984 * Q.mass_over_sum_pt + 2.9650905499893057
    if 0.09795414517 <= Q.mass_over_sum_pt < 0.170880120467:
        z += -12.752496719360352 * Q.mass_over_sum_pt + 1.0037834032378483
    if Q.mass_over_sum_pt >= 0.170880120467:
        z += 59.55404472351074 * Q.mass_over_sum_pt - 11.351967109072092
    if 6.856374501323 <= Q.log_sum_pt < 6.910130970417:
        z += 11.685096740722656 * Q.log_sum_pt - 80.11739933858331
    if 6.910130970417 <= Q.log_sum_pt < 6.92034855022:
        z += -14.710685729980469 * Q.log_sum_pt + 102.2809146006125
    if 6.92034855022 <= Q.log_sum_pt < 6.935549248787:
        z += -13.286181211471558 * Q.log_sum_pt + 92.42284682116752
    if 6.935549248787 <= Q.log_sum_pt < 6.959293500649:
        z += -18.69449019432068 * Q.log_sum_pt + 129.93244012437472
    if 6.959293500649 <= Q.log_sum_pt < 6.989450376716:
        z += -17.038525938987732 * Q.log_sum_pt + 118.40809884492909
    if 6.989450376716 <= Q.log_sum_pt < 7.139296169016:
        z += -13.13146197795868 * Q.log_sum_pt + 91.09986917066108
    if Q.log_sum_pt >= 7.139296169016:
        z += -5.600999236106873 * Q.log_sum_pt + 37.33766536684075
    if Q.z_top30_slots >= 0.934183811419:
        z += -5.796196937561035 * Q.z_top30_slots + 5.414713346865903
    if 0.240474711359 <= Q.max_dr < 0.43572281599:
        z += 0.44531580805778503 * Q.max_dr - 0.1070871904062957
    if Q.max_dr >= 0.43572281599:
        z += 20.552796810865402 * Q.max_dr - 8.868375435415059
    if Q.sum_pt_top40 >= 906.60234375:
        z += 0.009372691623866558 * Q.sum_pt_top40 - 8.497304193443416
    if Q.sum_pt_top20 >= 1005.0126953125:
        z += 0.0020994613878428936 * Q.sum_pt_top20 - 2.1099853481005084
    if Q.z_top40_slots >= 0.930046498893:
        z += -17.07607078552246 * Q.z_top40_slots + 15.881539848924204
    if Q.sum_pt_top3 >= 331.25:
        z += 0.0004990714369341731 * Q.sum_pt_top3 - 0.16531741348444484
    if Q.n_dr_0p2_0p4 < 13.0:
        z += 0.0027606282383203506 * Q.n_dr_0p2_0p4 - 0.03588816709816456
    if Q.girth2_top30 < 0.018076787298:
        z += 18.244482040405273 * Q.girth2_top30 - 0.3298016212065872
    if Q.sum_pt_top50 >= 1061.183898925781:
        z += -0.010497638024389744 * Q.sum_pt_top50 + 11.139924448233442
    if 68.286969674465 <= Q.mass_top30 < 101.927236862114:
        z += 0.027898523956537247 * Q.mass_top30 - 1.9051056593823943
    if Q.mass_top30 >= 101.927236862114:
        z += 0.008592627942562103 * Q.mass_top30 + 0.06269097646939259
    if 64.485443115234 <= Q.mass < 172.8:
        z += 0.011391735635697842 * Q.mass - 0.7346011203195772
    if Q.mass >= 172.8:
        z += -0.0955463582649827 * Q.mass + 17.74430150571802
    if 91.19 <= Q.mass_top50 < 157.544814151857:
        z += -0.014436286874115467 * Q.mass_top50 + 1.3164450000505894
    if Q.mass_top50 >= 157.544814151857:
        z += -0.1722857328131795 * Q.mass_top50 + 26.18480662449403
    if Q.mass_over_sum_pt_sq >= 0.029200015571:
        z += -691.5810546875 * Q.mass_over_sum_pt_sq + 20.194177565483603
    if Q.z_dr_0_0p05 >= 0.878906026483:
        z += 0.6807188391685486 * Q.z_dr_0_0p05 - 0.5982878900857493
    if Q.z_top50_slots >= 0.958653609576:
        z += 13.414916038513184 * Q.z_top50_slots - 12.860257682479638
    if Q.girth2_top10 < 0.004752875822:
        z += -91.68704986572266 * Q.girth2_top10 + 0.4357771624973016
    if Q.girth2_top2 < 0.0140332421:
        z += 35.8387565612793 * Q.girth2_top2 - 0.5029339473873958
    if 631.275 <= Q.sum_pt_top5 < 902.40625:
        z += -1.76308039954165e-05 * Q.sum_pt_top5 + 0.011129885792206551
    if Q.sum_pt_top5 >= 902.40625:
        z += -0.006127436116003082 * Q.sum_pt_top5 + 5.524656385631124
    if Q.girth < 0.085894044489:
        z += 11.164811134338379 * Q.girth - 0.9589907842841432
    if Q.z_dr_0p2_0p4 < 0.051804735139:
        z += 1.3462055921554565 * Q.z_dr_0p2_0p4 - 0.06973982414425407
    if Q.mass_top40 >= 136.785:
        z += -0.019948257133364677 * Q.mass_top40 + 2.7286223519872874
    if Q.sum_pt_top15 >= 951.1375:
        z += -0.0031421585008502007 * Q.sum_pt_top15 + 2.988624781102408
    if Q.sum_pt_top10 >= 845.53671875:
        z += 0.004453493747860193 * Q.sum_pt_top10 - 3.7655924905393476
    if Q.n_particles < 64.0 and Q.mass_top5 > 3.066194584349:
        z += -0.0001837839954532683 * (64.0 - Q.n_particles) * (Q.mass_top5 - 3.066194584349)
    if Q.n_particles < 64.0 and Q.e2 < 0.038759447634:
        z += 1.524718999862671 * (64.0 - Q.n_particles) * (0.038759447634 - Q.e2)
    if Q.sum_pt_top3 > 331.25 and Q.n_dr_0p1_0p2 > 8.0:
        z += 7.507202826673165e-05 * (Q.sum_pt_top3 - 331.25) * (Q.n_dr_0p1_0p2 - 8.0)
    if Q.girth2_top30 < 0.018076787298 and Q.tau32 < 0.864499151707:
        z += 93.61780548095703 * (0.018076787298 - Q.girth2_top30) * (0.864499151707 - Q.tau32)
    if Q.log_sum_pt > 6.92034855022 and Q.girth2_top2 < 0.0140332421:
        z += -879.982421875 * (Q.log_sum_pt - 6.92034855022) * (0.0140332421 - Q.girth2_top2)
    if Q.sum_pt > 907.937170410156 and Q.pt1_dr01 > 0.253124156046:
        z += -6.175656744744629e-05 * (Q.sum_pt - 907.937170410156) * (Q.pt1_dr01 - 0.253124156046)
    if Q.mass_top50 > 157.544814151857 and Q.z_dr_0p05_0p1 > 0.591232848167:
        z += 1.5818783044815063 * (Q.mass_top50 - 157.544814151857) * (Q.z_dr_0p05_0p1 - 0.591232848167)
    if Q.sum_pt > 907.937170410156 and Q.girth2_top2 < 0.0140332421:
        z += 0.46493589878082275 * (Q.sum_pt - 907.937170410156) * (0.0140332421 - Q.girth2_top2)
    if Q.sum_pt_top3 > 331.25 and Q.n_dr_0p05_0p1 < 27.0:
        z += -3.11255753331352e-05 * (Q.sum_pt_top3 - 331.25) * (27.0 - Q.n_dr_0p05_0p1)
    if Q.max_dr > 0.240474711359 and Q.tau21 < 0.6418633461:
        z += -2.7577478885650635 * (Q.max_dr - 0.240474711359) * (0.6418633461 - Q.tau21)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.256996214389801
    if Q.mass_top50 < 71.795159472175:
        z += -0.03241396322846413 * Q.mass_top50 + 2.3271656591127985
    if Q.mass < 86.4:
        z += -0.008469819091260433 * Q.mass - 0.13472660102678158
    if 86.4 <= Q.mass < 92.85979309082:
        z += 0.026858282275497913 * Q.mass - 3.187074559114703
    if 92.85979309082 <= Q.mass < 101.049709320068:
        z += 0.09440853912383318 * Q.mass - 9.459777433282863
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += 0.024059665389358997 * Q.mass - 2.351044191420079
    if 120.60000000000001 <= Q.mass < 172.8:
        z += -0.010546962730586529 * Q.mass + 1.8225151598453524
    if Q.mass_over_sum_pt >= 0.050772907168:
        z += -38.84710693359375 * Q.mass_over_sum_pt + 1.9723805540847248
    if Q.e2 < 0.030297144316:
        z += -7.663891792297363 * Q.e2 + 0.5272970155417699
    if 0.030297144316 <= Q.e2 < 0.043586218357:
        z += -18.189488410949707 * Q.e2 + 0.8461925353090816
    if 0.043586218357 <= Q.e2 < 0.047553086095:
        z += -13.456844329833984 * Q.e2 + 0.639914476983608
    if Q.e2 >= 0.055571487173:
        z += -63.33951187133789 * Q.e2 + 3.519870871502135
    if Q.girth2_top20 < 0.0018230789:
        z += -97.76359558105469 * Q.girth2_top20 + 0.7851599034090031
    if 0.0018230789 <= Q.girth2_top20 < 0.008031209355:
        z += -33.991554260253906 * Q.girth2_top20 + 0.6688984404671231
    if 0.008031209355 <= Q.girth2_top20 < 0.016559833876:
        z += 63.77204132080078 * Q.girth2_top20 - 0.11626146294188003
    if Q.girth2_top20 >= 0.016559833876:
        z += 24.074928283691406 * Q.girth2_top20 + 0.5411161343094452
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.047710955142974854 * Q.n_dr_0p2_0p4 - 1.001930058002472
    if Q.log_sum_pt < 6.811175180312:
        z += -5.635697841644287 * Q.log_sum_pt + 38.38572526274548
    if Q.lam1 < 0.007259287357:
        z += -136.7386016845703 * Q.lam1 + 0.9926248024226603
    if 0.008241985248 <= Q.lam1 < 0.011744050682:
        z += 109.9339370727539 * Q.lam1 - 0.906073887608198
    if 0.011744050682 <= Q.lam1 < 0.024570249917:
        z += 68.24882507324219 * Q.lam1 - 0.416521819601086
    if Q.lam1 >= 0.024570249917:
        z += 140.02297973632812 * Q.lam1 - 2.1800307372545182
    if Q.lam2 < 0.006427166767:
        z += -56.19048309326172 * Q.lam2 + 0.3611456055586871
    if 0.228402115913 <= Q.LHA < 0.404204003833:
        z += 2.280733823776245 * Q.LHA - 0.5209244311848417
    if Q.LHA >= 0.404204003833:
        z += 10.273333311080933 * Q.LHA - 3.751565144986979
    if Q.girth2_top50 < 0.013514311784:
        z += 31.19808006286621 * Q.girth2_top50 - 0.4216205810317683
    if Q.mass_top10 < 85.412093844921:
        z += -0.007041445467621088 * Q.mass_top10 + 0.601424601084346
    if Q.mass_top10 >= 99.066784770599:
        z += 0.0201983954757452 * Q.mass_top10 - 2.00099009730709
    if Q.width < 0.009614971338:
        z += 207.8602294921875 * Q.width - 1.998570148877485
    if Q.mass_over_sum_pt_sq < 0.019863807341:
        z += -53.24590301513672 * Q.mass_over_sum_pt_sq + 1.0576663591902469
    if Q.mass < 120.60000000000001 and Q.sum_pt_top50 < 1008.9353515625:
        z += 3.794323856709525e-05 * (120.60000000000001 - Q.mass) * (1008.9353515625 - Q.sum_pt_top50)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.C2 < 0.142391438037:
        z += 50.54946517944336 * (Q.mass_over_sum_pt - 0.050772907168) * (0.142391438037 - Q.C2)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.n_dr_0p2_0p4 < 21.0:
        z += 0.9983740448951721 * (Q.mass_over_sum_pt - 0.050772907168) * (21.0 - Q.n_dr_0p2_0p4)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.lam2 < 0.002396991421:
        z += 4833.8623046875 * (Q.mass_over_sum_pt - 0.050772907168) * (0.002396991421 - Q.lam2)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.z_dr_0_0p05 < 0.908491230011:
        z += 6.851567268371582 * (Q.mass_over_sum_pt - 0.050772907168) * (0.908491230011 - Q.z_dr_0_0p05)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.z_dr_0p1_0p2 < 0.286492615938:
        z += 44.571800231933594 * (Q.mass_over_sum_pt - 0.050772907168) * (0.286492615938 - Q.z_dr_0p1_0p2)
    if Q.girth2_top20 > 0.0018230789 and Q.max_dr < 0.43572281599:
        z += 274.3378601074219 * (Q.girth2_top20 - 0.0018230789) * (0.43572281599 - Q.max_dr)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.eta_0 < 0.079528808594:
        z += 9.143257141113281 * (Q.mass_over_sum_pt - 0.050772907168) * (0.079528808594 - Q.eta_0)
    if Q.mass < 92.85979309082 and Q.sum_pt_top15 < 1082.54765625:
        z += -3.1329531338997185e-05 * (92.85979309082 - Q.mass) * (1082.54765625 - Q.sum_pt_top15)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.n_dr_0p1_0p2 < 17.0:
        z += -1.071255087852478 * (Q.mass_over_sum_pt - 0.050772907168) * (17.0 - Q.n_dr_0p1_0p2)
    if Q.LHA > 0.228402115913 and Q.D2 < 3.345338582993:
        z += -0.7044668793678284 * (Q.LHA - 0.228402115913) * (3.345338582993 - Q.D2)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.max_pair_mass > 13.047927274731:
        z += -0.2458280622959137 * (Q.mass_over_sum_pt - 0.050772907168) * (Q.max_pair_mass - 13.047927274731)
    if Q.girth2_top3 > 0.010023689877 and Q.D2 < 4.450168704987:
        z += 25.12383270263672 * (Q.girth2_top3 - 0.010023689877) * (4.450168704987 - Q.D2)
    if Q.lam1 > 0.008241985248 and Q.pt_4 < 68.125:
        z += 1.892674446105957 * (Q.lam1 - 0.008241985248) * (68.125 - Q.pt_4)
    if Q.e2 > 0.055571487173 and Q.max_dr < 0.368618160486:
        z += -1545.7371826171875 * (Q.e2 - 0.055571487173) * (0.368618160486 - Q.max_dr)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.3366878926753998
    if Q.mass < 82.85408782959:
        z += 0.11516338400542736 * Q.mass - 9.346753198948795
    if 82.85408782959 <= Q.mass < 86.4:
        z += 0.1641941089183092 * Q.mass - 13.409149187229172
    if 86.4 <= Q.mass < 91.19:
        z += 0.1200281661003828 * Q.mass - 9.59321172776033
    if 91.19 <= Q.mass < 101.049709320068:
        z += -0.08691233582794666 * Q.mass + 9.277692643084032
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += -0.02533089555799961 * Q.mass + 3.0549060042947533
    if Q.n_dr_0p2_0p4 < 2.0:
        z += 0.26805929094552994 * Q.n_dr_0p2_0p4 - 0.3436880186200142
    if 2.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += -0.017493687570095062 * Q.n_dr_0p2_0p4 + 0.2274179384112358
    if Q.max_dr < 0.193997652829:
        z += -10.553250312805176 * Q.max_dr + 2.047305790401114
    if Q.z_top50_slots < 0.990637830118:
        z += 36.729496002197266 * Q.z_top50_slots - 36.385628220944454
    if Q.mass_top50 < 79.21003612387:
        z += -0.006411326117813587 * Q.mass_top50 + 0.3305790658084675
    if 79.21003612387 <= Q.mass_top50 < 97.930041729355:
        z += 0.009469137527048588 * Q.mass_top50 - 0.9273130331648696
    if Q.mass_top30 < 76.415438713532:
        z += -0.0043623186647892 * Q.mass_top30 + 0.1766978759981901
    if 76.415438713532 <= Q.mass_top30 < 86.252206812802:
        z += 0.015925008803606033 * Q.mass_top30 - 1.37356715282432
    if Q.C2 < 0.056027559564:
        z += 8.62218189239502 * Q.C2 - 0.48307980954780416
    if Q.z_dr_0p2_0p4 < 0.004744913615:
        z += -167.80297374725342 * Q.z_dr_0p2_0p4 - 0.048489558058602
    if 0.004744913615 <= Q.z_dr_0p2_0p4 < 0.091225683689:
        z += 9.767491340637207 * Q.z_dr_0p2_0p4 - 0.8910460754760163
    if Q.girth2_top50 < 0.013514311784:
        z += 50.09746551513672 * Q.girth2_top50 - 0.6770327685597458
    if Q.girth2_top20 < 0.006374177987:
        z += 142.10067749023438 * Q.girth2_top20 - 1.6640447615353549
    if 0.006374177987 <= Q.girth2_top20 < 0.008031209355:
        z += 457.60736083984375 * Q.girth2_top20 - 3.6751405172938143
    if Q.mass_top20 < 66.841467317407:
        z += -0.018224697560071945 * Q.mass_top20 + 1.218165526331176
    if Q.girth2_top40 < 0.007709915821:
        z += -55.61778259277344 * Q.girth2_top40 + 1.348892333808978
    if 0.007709915821 <= Q.girth2_top40 < 0.008031986041:
        z += 162.06460571289062 * Q.girth2_top40 - 0.3294205557419265
    if 0.008031986041 <= Q.girth2_top40 < 0.008840538245:
        z += -409.1225280761719 * Q.girth2_top40 + 4.258346529650623
    if 0.008840538245 <= Q.girth2_top40 < 0.012926423095:
        z += -156.99981689453125 * Q.girth2_top40 + 2.02944605901624
    if Q.girth < 0.085894044489:
        z += 42.51690673828125 * Q.girth - 3.6519490789125935
    if Q.LHA < 0.320332145368:
        z += -7.74237585067749 * Q.LHA + 2.4801318664929144
    if Q.e2_sq < 0.009606007381:
        z += -460.2551574707031 * Q.e2_sq + 4.421214439806891
    if Q.mass_over_sum_pt < 0.090467494167:
        z += 32.73249053955078 * Q.mass_over_sum_pt - 2.9612263969581933
    if Q.e2 < 0.027935993578:
        z += -48.66191864013672 * Q.e2 + 1.359419046624018
    if Q.D2 < 1.788105106354 and Q.girth2_top50 < 0.007820314762:
        z += -470.6628723144531 * (1.788105106354 - Q.D2) * (0.007820314762 - Q.girth2_top50)
    if Q.D2 < 1.788105106354 and Q.girth2_top50 < 0.006118005947:
        z += 647.0003662109375 * (1.788105106354 - Q.D2) * (0.006118005947 - Q.girth2_top50)
    if Q.D2 < 1.788105106354 and Q.n_dr_0p2_0p4 < 9.0:
        z += 0.1300993412733078 * (1.788105106354 - Q.D2) * (9.0 - Q.n_dr_0p2_0p4)
    if Q.mass < 91.19 and Q.z_dr_0p05_0p1 > 0.402664637566:
        z += -0.13727591931819916 * (91.19 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.402664637566)
    if Q.n_dr_0p2_0p4 < 13.0 and Q.planar_flow < 0.536493504079:
        z += 0.05464760214090347 * (13.0 - Q.n_dr_0p2_0p4) * (0.536493504079 - Q.planar_flow)
    if Q.mass_top50 < 97.930041729355 and Q.D2 < 1.601009327173:
        z += -0.4180588722229004 * (97.930041729355 - Q.mass_top50) * (1.601009327173 - Q.D2)
    if Q.mass_top30 < 76.415438713532 and Q.D2 < 1.601009327173:
        z += 0.19355566799640656 * (76.415438713532 - Q.mass_top30) * (1.601009327173 - Q.D2)
    if Q.mass < 101.049709320068 and Q.D2 < 1.601009327173:
        z += 0.308453768491745 * (101.049709320068 - Q.mass) * (1.601009327173 - Q.D2)
    if Q.n_dr_0p2_0p4 < 13.0 and Q.z_1st < 0.499317836761:
        z += 0.12861794233322144 * (13.0 - Q.n_dr_0p2_0p4) * (0.499317836761 - Q.z_1st)
    if Q.mass < 91.19 and Q.max_dr < 0.390781164169:
        z += -1.3685874938964844 * (91.19 - Q.mass) * (0.390781164169 - Q.max_dr)
    if Q.mass < 101.049709320068 and Q.max_dr < 0.390781164169:
        z += 0.9199686050415039 * (101.049709320068 - Q.mass) * (0.390781164169 - Q.max_dr)
    if Q.mass < 86.4 and Q.z_dr_0p05_0p1 > 0.402664637566:
        z += 0.03642797842621803 * (86.4 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.402664637566)
    if Q.mass < 91.19 and Q.pt_dispersion < 0.33780374676:
        z += -0.16848354041576385 * (91.19 - Q.mass) * (0.33780374676 - Q.pt_dispersion)
    if Q.LHA < 0.320332145368 and Q.pt_dispersion < 0.355009326339:
        z += 7.933815002441406 * (0.320332145368 - Q.LHA) * (0.355009326339 - Q.pt_dispersion)
    if Q.max_dr < 0.193997652829 and Q.dr_8 > 0.008114792206:
        z += -307.24371337890625 * (0.193997652829 - Q.max_dr) * (Q.dr_8 - 0.008114792206)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.530512273311615
    if 0.076966318366 <= Q.mass_over_sum_pt < 0.088731426731:
        z += 22.399324417114258 * Q.mass_over_sum_pt - 1.7239935342709334
    if 0.088731426731 <= Q.mass_over_sum_pt < 0.09795414517:
        z += 70.44570350646973 * Q.mass_over_sum_pt - 5.987217300127929
    if Q.mass_over_sum_pt >= 0.09795414517:
        z += -21.518102645874023 * Q.mass_over_sum_pt + 3.021018718104491
    if Q.sum_pt_top40 < 1001.52314453125:
        z += -0.0060686455108225346 * Q.sum_pt_top40 + 6.077888935044438
    if Q.girth2_top40 < 0.005196965925:
        z += -114.16725158691406 * Q.girth2_top40 + 0.8802198992500361
    if 0.005196965925 <= Q.girth2_top40 < 0.007709915821:
        z += 1.4171676635742188 * Q.girth2_top40 + 0.27953161094433443
    if Q.girth2_top40 >= 0.007709915821:
        z += 115.58441925048828 * Q.girth2_top40 - 0.6006882883057016
    if Q.n_dr_0p2_0p4 < 15.0:
        z += 0.06594735570251942 * Q.n_dr_0p2_0p4 - 1.2682296317070723
    if 15.0 <= Q.n_dr_0p2_0p4 < 21.0:
        z += 0.0465032160282135 * Q.n_dr_0p2_0p4 - 0.9765675365924835
    if 64.485443115234 <= Q.mass < 80.784643554688:
        z += 0.03071778453886509 * Q.mass - 1.9808499475069994
    if 80.784643554688 <= Q.mass < 87.363773345947:
        z += 0.07929460890591145 * Q.mass - 5.905111389017522
    if 87.363773345947 <= Q.mass < 101.049709320068:
        z += 0.12149332650005817 * Q.mass - 9.591750588402181
    if 101.049709320068 <= Q.mass < 125.1:
        z += 0.034035200253129005 * Q.mass - 0.7541323534721784
    if 125.1 <= Q.mass < 143.787612915039:
        z += 0.016961948946118355 * Q.mass + 1.3817313850348545
    if Q.mass >= 143.787612915039:
        z += -0.01478339172899723 * Q.mass + 5.946318141884417
    if Q.n_particles >= 51.0:
        z += 0.06794362515211105 * Q.n_particles - 3.4651248827576637
    if 0.005718442372 <= Q.girth2_top20 < 0.008031209355:
        z += -85.38438415527344 * Q.girth2_top20 + 0.48826568026064104
    if Q.girth2_top20 >= 0.008031209355:
        z += 124.75260925292969 * Q.girth2_top20 - 1.1993885070308934
    if Q.sum_pt < 1012.672900390625:
        z += -0.018161088228225708 * Q.sum_pt + 18.391241890327365
    if Q.girth2_top30 < 0.007463984647:
        z += -116.07711029052734 * Q.girth2_top30 + 0.8663977690766218
    if Q.mass_over_sum_pt_sq < 0.006166777647:
        z += 349.97776794433594 * Q.mass_over_sum_pt_sq - 3.1409961422482278
    if 0.006166777647 <= Q.mass_over_sum_pt_sq < 0.013977372691:
        z += 125.8240966796875 * Q.mass_over_sum_pt_sq - 1.7586902928004078
    if Q.girth < 0.097499583662:
        z += -25.494308471679688 * Q.girth + 2.485684461739369
    if Q.D2 < 1.788105106354:
        z += -0.29398709535598755 * Q.D2 + 0.5256798264082216
    if Q.lam2 < 0.001411893759:
        z += 495.16436767578125 * Q.lam2 - 0.6991194804006169
    if 82.04491364955 <= Q.mass_top50 < 97.930041729355:
        z += -0.03548259288072586 * Q.mass_top50 + 2.9111662689612907
    if Q.mass_top50 >= 97.930041729355:
        z += -0.00042788684368133545 * Q.mass_top50 - 0.5217425560567523
    if Q.e2 >= 0.025159193203:
        z += -17.096574783325195 * Q.e2 + 0.43013602808321644
    if Q.lam1 < 0.008241985248:
        z += -104.41376495361328 * Q.lam1 + 0.37647611760178834
    if 0.008241985248 <= Q.lam1 < 0.020485236462:
        z += 39.540199279785156 * Q.lam1 - 0.8099903320010011
    if Q.z_dr_0p2_0p4 < 0.091225683689:
        z += -6.027197360992432 * Q.z_dr_0p2_0p4 + 0.5498351999850711
    if Q.z_dr_0p1_0p2 < 0.154838323593:
        z += -1.357612133026123 * Q.z_dr_0p1_0p2 + 0.21021038676728182
    if Q.z_dr_0p1_0p2 >= 0.218764226139:
        z += 0.6860106587409973 * Q.z_dr_0p1_0p2 - 0.1500745908825799
    if Q.C2 >= 0.072798889503:
        z += 4.759771823883057 * Q.C2 - 0.3465061030663554
    if Q.mass_top40 >= 79.554505888974:
        z += -0.02335016243159771 * Q.mass_top40 + 1.8576106346730397
    if Q.z_top40_slots < 0.983076389702:
        z += 2.715604305267334 * Q.z_top40_slots - 2.6696464762814185
    if Q.z_top15_slots < 0.813850690953:
        z += -0.8557788133621216 * Q.z_top15_slots + 0.696476178557701
    if Q.sum_pt_top30 < 1011.52392578125:
        z += 0.0009509267983958125 * Q.sum_pt_top30 - 0.9618852082439275
    if Q.mass_top15 >= 30.359943489662:
        z += 0.001654097461141646 * Q.mass_top15 - 0.05021830544665376
    if Q.mass_over_sum_pt > 0.076966318366 and Q.sum_pt < 1115.722741699219:
        z += 0.34306997060775757 * (Q.mass_over_sum_pt - 0.076966318366) * (1115.722741699219 - Q.sum_pt)
    if Q.n_dr_0p2_0p4 < 15.0 and Q.z_dr_0p1_0p2 > 0.334047731757:
        z += 0.16405491530895233 * (15.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.334047731757)
    if Q.girth2_top40 > 0.005196965925 and Q.sum_pt_top30 < 911.9328125:
        z += 0.6575262546539307 * (Q.girth2_top40 - 0.005196965925) * (911.9328125 - Q.sum_pt_top30)
    if Q.n_particles > 51.0 and Q.z_top50_slots > 0.970443639316:
        z += -2.8903422355651855 * (Q.n_particles - 51.0) * (Q.z_top50_slots - 0.970443639316)
    if Q.sum_pt_top40 < 1001.52314453125 and Q.log_sum_pt < 6.811175180312:
        z += 0.06468408554792404 * (1001.52314453125 - Q.sum_pt_top40) * (6.811175180312 - Q.log_sum_pt)
    if Q.mass > 87.363773345947 and Q.log_sum_pt < 6.903422848462:
        z += 0.14067226648330688 * (Q.mass - 87.363773345947) * (6.903422848462 - Q.log_sum_pt)
    if Q.sum_pt < 1012.672900390625 and Q.n_real_top50 < 43.0:
        z += -0.0008411120506934822 * (1012.672900390625 - Q.sum_pt) * (43.0 - Q.n_real_top50)
    if Q.sum_pt_top40 < 1001.52314453125 and Q.max_dr < 0.39398368001:
        z += -0.09527058154344559 * (1001.52314453125 - Q.sum_pt_top40) * (0.39398368001 - Q.max_dr)
    if Q.lam2 < 0.001411893759 and Q.n_dr_0p05_0p1 > 4.0:
        z += 17.85942268371582 * (0.001411893759 - Q.lam2) * (Q.n_dr_0p05_0p1 - 4.0)
    if Q.sum_pt < 1012.672900390625 and Q.z_0 > 0.088724280345:
        z += -0.02953270822763443 * (1012.672900390625 - Q.sum_pt) * (Q.z_0 - 0.088724280345)
    if Q.n_particles > 51.0 and Q.tau32 < 0.899011841416:
        z += -0.018427787348628044 * (Q.n_particles - 51.0) * (0.899011841416 - Q.tau32)
    if Q.sum_pt < 1012.672900390625 and Q.mean_eta2 < 0.001225592976:
        z += -11.261368751525879 * (1012.672900390625 - Q.sum_pt) * (0.001225592976 - Q.mean_eta2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.mean_eta2 > 0.006802603323:
        z += -8.479887008666992 * (21.0 - Q.n_dr_0p2_0p4) * (Q.mean_eta2 - 0.006802603323)
    if Q.z_dr_0p1_0p2 > 0.218764226139 and Q.mean_phi > 0.000440474624:
        z += -2009.821044921875 * (Q.z_dr_0p1_0p2 - 0.218764226139) * (Q.mean_phi - 0.000440474624)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.6001477837562561
    if Q.mass_top40 < 80.890431271924:
        z += -0.006766905542463064 * Q.mass_top40 + 0.5351982549107985
    if 80.890431271924 <= Q.mass_top40 < 91.288535717504:
        z += 0.017178111243993044 * Q.mass_top40 - 1.401724479759096
    if 91.288535717504 <= Q.mass_top40 < 120.60000000000001:
        z += -0.005678329151123762 * Q.mass_top40 + 0.6848064956255258
    if Q.sum_pt_top40 < 956.21328125:
        z += -0.007043474353849888 * Q.sum_pt_top40 + 6.735063723295025
    if Q.girth2_top40 < 0.00625977218:
        z += -224.7488250732422 * Q.girth2_top40 + 1.406876442681168
    if Q.z_top40_slots < 0.930046498893:
        z += 28.52512550354004 * Q.z_top40_slots - 26.529693105050836
    if Q.mass < 62.55:
        z += -0.009652860462665558 * Q.mass + 1.3203665183857083
    if 62.55 <= Q.mass < 82.85408782959:
        z += -0.06925619021058083 * Q.mass + 5.048554794117808
    if 82.85408782959 <= Q.mass < 92.85979309082:
        z += -0.01372971385717392 * Q.mass + 0.4479592454649799
    if 92.85979309082 <= Q.mass < 136.785:
        z += 0.02485967054963112 * Q.mass - 3.135443006053052
    if 136.785 <= Q.mass < 143.787612915039:
        z += 0.03451253101229668 * Q.mass - 4.455809524438761
    if 143.787612915039 <= Q.mass < 172.8:
        z += -0.04170852527022362 * Q.mass + 6.5038342122876704
    if Q.mass >= 172.8:
        z += -0.01056034117937088 * Q.mass + 1.1214280013883169
    if Q.girth2_top15 < 0.009962397174:
        z += 38.13152313232422 * Q.girth2_top15 - 1.2175725056672435
    if 0.009962397174 <= Q.girth2_top15 < 0.02146577947:
        z += 72.8212890625 * Q.girth2_top15 - 1.5631657317367482
    if Q.girth < 0.02783744745:
        z += 104.0036563873291 * Q.girth - 4.078037034684154
    if 0.02783744745 <= Q.girth < 0.043628720567:
        z += 74.90470886230469 * Q.girth - 3.2679966121059794
    if Q.girth >= 0.097499583662:
        z += 9.642452239990234 * Q.girth - 0.9401350788797671
    if Q.sum_pt < 949.91689453125:
        z += -0.028171224519610405 * Q.sum_pt + 26.76032211081092
    if Q.log_sum_pt < 6.856374501323:
        z += 20.178861618041992 * Q.log_sum_pt - 138.3538322636685
    if Q.lam1 < 0.005240342196:
        z += -111.98651885986328 * Q.lam1 + 0.5868476801644914
    if Q.LHA < 0.209102506978:
        z += -15.379210472106934 * Q.LHA + 3.2158314650598707
    if Q.width >= 0.025866900997:
        z += -261.4051818847656 * Q.width + 6.7617419599160105
    if Q.mass_top15 < 52.713279629696:
        z += 0.0032453536987304688 * Q.mass_top15 - 0.4772289909007031
    if 52.713279629696 <= Q.mass_top15 < 78.955574164508:
        z += 0.011666500940918922 * Q.mass_top15 - 0.9211352802810263
    if Q.mass_top50 < 136.785:
        z += -0.019103460013866425 * Q.mass_top50 + 1.790297823417931
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.03426062688231468 * Q.mass_top50 - 5.509108802676201
    if Q.n_dr_0p1_0p2 >= 21.0:
        z += -0.025025038048624992 * Q.n_dr_0p1_0p2 + 0.5255257990211248
    if Q.sum_pt_top50 >= 1245.696667480468:
        z += -0.00046597106847912073 * Q.sum_pt_top50 + 0.5804586071467537
    if Q.girth2 < 0.00363885588:
        z += -137.02430725097656 * Q.girth2 + 0.49861170614314265
    if Q.e2 < 0.047553086095:
        z += -0.03919915482401848 * Q.e2 + 0.0018640407841977855
    if Q.e2 >= 0.065240035206:
        z += -41.025123596191406 * Q.e2 + 2.6764805077460285
    if Q.z_dr_0p05_0p1 >= 0.710821145773:
        z += -0.7044130563735962 * Q.z_dr_0p05_0p1 + 0.5007116958289405
    if Q.girth2_top40 < 0.00625977218 and Q.sum_pt < 1022.583984375:
        z += 1.5729429721832275 * (0.00625977218 - Q.girth2_top40) * (1022.583984375 - Q.sum_pt)
    if Q.z_top40_slots < 0.930046498893 and Q.n_pt_above_10 > 28.0:
        z += 1.3098313808441162 * (0.930046498893 - Q.z_top40_slots) * (Q.n_pt_above_10 - 28.0)
    if Q.sum_pt_top40 < 956.21328125 and Q.z_top15_slots > 0.794624168612:
        z += -0.05074381083250046 * (956.21328125 - Q.sum_pt_top40) * (Q.z_top15_slots - 0.794624168612)
    if Q.log_sum_pt < 6.856374501323 and Q.z_top20_slots > 0.896541111574:
        z += 202.95094299316406 * (6.856374501323 - Q.log_sum_pt) * (Q.z_top20_slots - 0.896541111574)
    if Q.sum_pt_top40 < 956.21328125 and Q.z_top30_slots > 0.97348863653:
        z += -0.3096482753753662 * (956.21328125 - Q.sum_pt_top40) * (Q.z_top30_slots - 0.97348863653)
    if Q.girth2_top15 < 0.02146577947 and Q.n_particles > 41.0:
        z += 1.424715280532837 * (0.02146577947 - Q.girth2_top15) * (Q.n_particles - 41.0)
    if Q.mass > 53.87361907959 and Q.eccentricity > 0.588471729833:
        z += 0.01866212487220764 * (Q.mass - 53.87361907959) * (Q.eccentricity - 0.588471729833)
    if Q.sum_pt < 949.91689453125 and Q.z_0 < 0.412129651204:
        z += -0.028383290395140648 * (949.91689453125 - Q.sum_pt) * (0.412129651204 - Q.z_0)
    if Q.mass_top40 < 120.60000000000001 and Q.max_dr > 0.379163956642:
        z += 0.05289282649755478 * (120.60000000000001 - Q.mass_top40) * (Q.max_dr - 0.379163956642)
    if Q.log_sum_pt < 6.856374501323 and Q.dr_7 < 0.082575402011:
        z += 144.32968139648438 * (6.856374501323 - Q.log_sum_pt) * (0.082575402011 - Q.dr_7)
    if Q.mass_top40 < 80.890431271924 and Q.eta_1 > 0.057922363281:
        z += -6.3542680740356445 * (80.890431271924 - Q.mass_top40) * (Q.eta_1 - 0.057922363281)
    return max(0.0, z)


def neuron_10(Q):
    z = 4.0921173095703125
    if Q.girth < 0.120745175332:
        z += 9.858044624328613 * Q.girth - 1.1903113265952383
    if Q.mass < 62.55:
        z += -0.03767431154847145 * Q.mass + 2.7057094663381585
    if 62.55 <= Q.mass < 80.4:
        z += -0.07318871095776558 * Q.mass + 4.927135149389506
    if 80.4 <= Q.mass < 86.4:
        z += -0.021196182817220688 * Q.mass + 0.7469358868896965
    if 86.4 <= Q.mass < 120.60000000000001:
        z += 0.031708020716905594 * Q.mass - 3.823987298458815
    if 143.787612915039 <= Q.mass < 162.836349487305:
        z += -0.10696729272603989 * Q.mass + 15.380571681061491
    if Q.mass >= 162.836349487305:
        z += -0.17912443727254868 * Q.mass + 27.13037768844278
    if Q.girth2_top30 < 0.005402330897:
        z += -110.07698822021484 * Q.girth2_top30 + 0.5946723145107716
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.0857107862830162 * Q.mass_top50 - 11.72394990172237
    if 160.8 <= Q.mass_top50 < 172.8:
        z += 0.11115625500679016 * Q.mass_top50 - 15.815581272505224
    if Q.mass_top50 >= 172.8:
        z += 0.12768272496759892 * Q.mass_top50 - 18.67135528173298
    if Q.max_dr < 0.240474711359:
        z += -5.043196201324463 * Q.max_dr + 0.693055922111125
    if 0.240474711359 <= Q.max_dr < 0.402178311348:
        z += 3.213937282562256 * Q.max_dr - 1.292575869079268
    if Q.z_dr_0_0p05 >= 0.767473447323:
        z += -10.051336288452148 * Q.z_dr_0_0p05 + 7.714133711501138
    if Q.girth2_top5 < 0.024419631481:
        z += 77.50971984863281 * Q.girth2_top5 - 1.8927587948991644
    if 0.001868040786 <= Q.lam1 < 0.004673423215:
        z += -64.47960662841797 * Q.lam1 + 0.12045053504712071
    if Q.lam1 >= 0.004673423215:
        z += -65.69361340999603 * Q.lam1 + 0.12612410252331507
    if Q.e2 < 0.065240035206:
        z += 27.544572830200195 * Q.e2 - 1.7970089011764918
    if Q.sum_pt_top50 < 959.095727539062:
        z += 0.009067120030522346 * Q.sum_pt_top50 - 8.696236082357832
    if Q.dr_0 < 0.064132973195:
        z += 0.5587748885154724 * Q.dr_0 - 0.03583589494720191
    if Q.n_dr_0p2_0p4 < 6.0:
        z += 0.14363087713718414 * Q.n_dr_0p2_0p4 - 1.3024537786841393
    if 6.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += 0.06295264512300491 * Q.n_dr_0p2_0p4 - 0.8183843865990639
    if 0.000237176831 <= Q.girth2_top10 < 0.007678543663:
        z += 62.2093391418457 * Q.girth2_top10 - 0.014754613916267223
    if Q.girth2_top10 >= 0.007678543663:
        z += 31.371599197387695 * Q.girth2_top10 + 0.22203431871549278
    if Q.planar_flow < 0.356311369374:
        z += 0.21914370357990265 * Q.planar_flow - 0.07808339311224506
    if Q.z_dr_0p2_0p4 < 0.091225683689:
        z += -9.012594223022461 * Q.z_dr_0p2_0p4 + 0.8221800698067557
    if Q.tau21 < 0.428145796061:
        z += -0.3360633850097656 * Q.tau21 + 0.14388412550196045
    if Q.dr_1 < 0.069404718919:
        z += -5.152204990386963 * Q.dr_1 + 0.35758733917087626
    if Q.girth2_top30 < 0.005402330897 and Q.sum_pt_top5 < 631.275:
        z += 0.2699367105960846 * (0.005402330897 - Q.girth2_top30) * (631.275 - Q.sum_pt_top5)
    if Q.mass_top50 > 160.8 and Q.pt_9 < 41.4375:
        z += 0.00042457348899915814 * (Q.mass_top50 - 160.8) * (41.4375 - Q.pt_9)
    if Q.girth2_top5 < 0.024419631481 and Q.sum_pt_top3 < 656.384375:
        z += 0.1091674417257309 * (0.024419631481 - Q.girth2_top5) * (656.384375 - Q.sum_pt_top3)
    if Q.z_dr_0_0p05 > 0.767473447323 and Q.n_particles > 36.0:
        z += -0.03436761721968651 * (Q.z_dr_0_0p05 - 0.767473447323) * (Q.n_particles - 36.0)
    if Q.dr_0 < 0.064132973195 and Q.n_dr_0p2_0p4 > 2.0:
        z += 0.8519852161407471 * (0.064132973195 - Q.dr_0) * (Q.n_dr_0p2_0p4 - 2.0)
    if Q.e2 < 0.065240035206 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += -33.36051559448242 * (0.065240035206 - Q.e2) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.mass_top50 > 160.8 and Q.D2 > 1.230420708656:
        z += 0.014417076483368874 * (Q.mass_top50 - 160.8) * (Q.D2 - 1.230420708656)
    if Q.lam1 > 0.001868040786 and Q.pt_dispersion > 0.274620002508:
        z += -147.4144287109375 * (Q.lam1 - 0.001868040786) * (Q.pt_dispersion - 0.274620002508)
    if Q.sum_pt_top50 < 959.095727539062 and Q.mass_top10 > 23.388939226434:
        z += 9.124230018642265e-06 * (959.095727539062 - Q.sum_pt_top50) * (Q.mass_top10 - 23.388939226434)
    if Q.mass_top50 > 172.8 and Q.sum_pt < 1260.540869140625:
        z += -0.00032728619407862425 * (Q.mass_top50 - 172.8) * (1260.540869140625 - Q.sum_pt)
    if Q.mass < 120.60000000000001 and Q.tau21 < 0.470341457427:
        z += 0.09599415957927704 * (120.60000000000001 - Q.mass) * (0.470341457427 - Q.tau21)
    if Q.girth2_top5 < 0.024419631481 and Q.n_particles > 34.0:
        z += 0.3585648834705353 * (0.024419631481 - Q.girth2_top5) * (Q.n_particles - 34.0)
    if Q.mass < 120.60000000000001 and Q.D2 < 1.976207274199:
        z += -0.006290115416049957 * (120.60000000000001 - Q.mass) * (1.976207274199 - Q.D2)
    if Q.z_dr_0p2_0p4 < 0.091225683689 and Q.max_dr > 0.273806282878:
        z += 10.634735107421875 * (0.091225683689 - Q.z_dr_0p2_0p4) * (Q.max_dr - 0.273806282878)
    if Q.sum_pt_top15 > 935.104296875 and Q.dr_4 < 0.063364507347:
        z += -0.029315780848264694 * (Q.sum_pt_top15 - 935.104296875) * (0.063364507347 - Q.dr_4)
    if Q.mass_top50 > 172.8 and Q.max_pair_mass < 33.376099042476:
        z += -0.0021908325143158436 * (Q.mass_top50 - 172.8) * (33.376099042476 - Q.max_pair_mass)
    if Q.mass < 120.60000000000001 and Q.mass_top3 > 16.899120053094:
        z += 0.0003962826158385724 * (120.60000000000001 - Q.mass) * (Q.mass_top3 - 16.899120053094)
    if Q.n_dr_0p2_0p4 < 13.0 and Q.dr_2 < 0.065454679258:
        z += 0.2468874305486679 * (13.0 - Q.n_dr_0p2_0p4) * (0.065454679258 - Q.dr_2)
    return max(0.0, z)


def neuron_11(Q):
    z = -0.10615711659193039
    if Q.n_dr_0p2_0p4 < 5.0:
        z += 0.016307901591062546 * Q.n_dr_0p2_0p4 - 0.39553718641400337
    if 5.0 <= Q.n_dr_0p2_0p4 < 10.0:
        z += 0.06279953569173813 * Q.n_dr_0p2_0p4 - 0.6279953569173813
    if Q.z_dr_0p2_0p4 < 0.006381743611:
        z += -112.45596313476562 * Q.z_dr_0p2_0p4 + 0.7176651242541421
    if Q.mass < 80.4:
        z += -0.046657461673021317 * Q.mass + 4.836193353090776
    if 80.4 <= Q.mass < 92.85979309082:
        z += -0.078052818775177 * Q.mass + 7.360380064104094
    if 92.85979309082 <= Q.mass < 101.049709320068:
        z += -0.01372559368610382 * Q.mass + 1.3869672522261518
    if Q.girth2_top10 < 0.00130722027:
        z += -127.63072967529297 * Q.girth2_top10 + 0.1668414769064335
    if Q.lam1 < 0.006716736591:
        z += 106.7028579711914 * Q.lam1 - 0.7166949904993773
    if Q.e2 < 0.025159193203:
        z += -69.05619049072266 * Q.e2 + 1.779505797820483
    if 0.025159193203 <= Q.e2 < 0.027935993578:
        z += -40.406028747558594 * Q.e2 + 1.058690843227019
    if 0.027935993578 <= Q.e2 < 0.038759447634:
        z += 6.475910186767578 * Q.e2 - 0.25100270176650513
    if Q.girth2_top30 < 0.005809484705:
        z += 200.71939086914062 * Q.girth2_top30 - 1.3527404657145912
    if 0.005809484705 <= Q.girth2_top30 < 0.006363915755:
        z += 336.6770935058594 * Q.girth2_top30 - 2.142584659709547
    if Q.z_top30_slots >= 0.980193855091:
        z += 5.967349052429199 * Q.z_top30_slots - 5.849158872374202
    if Q.mass_top50 < 86.4:
        z += 0.03441218286752701 * Q.mass_top50 - 2.9732125997543335
    if Q.mass_top30 < 60.438206617337:
        z += -0.008224139921367168 * Q.mass_top30 + 0.49705226781747863
    if Q.girth2_top5 < 0.007164202106:
        z += 20.320083618164062 * Q.girth2_top5 - 0.14557718585134707
    if Q.z_dr_0_0p05 < 0.095732276142:
        z += -5.823221206665039 * Q.z_dr_0_0p05 + 0.5574702205924079
    if Q.mass_over_sum_pt_sq < 0.008184367501:
        z += -158.84271240234375 * Q.mass_over_sum_pt_sq + 1.3000271331564317
    if Q.log_sum_pt < 6.89371369877:
        z += -1.344094157218933 * Q.log_sum_pt + 9.265800304056878
    if Q.z_top5_slots >= 0.534625950898:
        z += -1.7335208654403687 * Q.z_top5_slots + 0.9267852410875811
    if Q.z_dr_0p05_0p1 >= 0.850921532512:
        z += 6.472963333129883 * Q.z_dr_0p05_0p1 - 5.5079838793208635
    if Q.width < 0.006170281901:
        z += 142.61053466796875 * Q.width - 0.8799472009537006
    if Q.max_dr < 0.298200035095:
        z += -3.0947091579437256 * Q.max_dr + 0.9228423795076368
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208341:
        z += -31.179813385009766 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208341 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.006574561819434166 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.049709320068 and Q.planar_flow < 0.356311369374:
        z += 0.08434312045574188 * (101.049709320068 - Q.mass) * (0.356311369374 - Q.planar_flow)
    if Q.mass < 101.049709320068 and Q.z_dr_0p1_0p2 < 0.334047731757:
        z += -0.03385545685887337 * (101.049709320068 - Q.mass) * (0.334047731757 - Q.z_dr_0p1_0p2)
    if Q.mass < 101.049709320068 and Q.mass_top10 > 44.192251085966:
        z += 0.0005255925352685153 * (101.049709320068 - Q.mass) * (Q.mass_top10 - 44.192251085966)
    if Q.mass < 80.4 and Q.D2 < 3.345338582993:
        z += -0.017031492665410042 * (80.4 - Q.mass) * (3.345338582993 - Q.D2)
    if Q.mass < 80.4 and Q.sum_pt_top2 < 464.75:
        z += -7.939718489069492e-05 * (80.4 - Q.mass) * (464.75 - Q.sum_pt_top2)
    if Q.mass < 92.85979309082 and Q.D2 < 1.230420708656:
        z += 0.04318709671497345 * (92.85979309082 - Q.mass) * (1.230420708656 - Q.D2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_dr_0p05_0p1 > 0.591232848167:
        z += -0.32162055373191833 * (10.0 - Q.n_dr_0p2_0p4) * (Q.z_dr_0p05_0p1 - 0.591232848167)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.tau21 > 0.129504834861:
        z += 0.06151099130511284 * (10.0 - Q.n_dr_0p2_0p4) * (Q.tau21 - 0.129504834861)
    if Q.mass_top50 < 86.4 and Q.girth2_top15 < 0.00416995399:
        z += 3.591453790664673 * (86.4 - Q.mass_top50) * (0.00416995399 - Q.girth2_top15)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.n_real_top30 > 22.0:
        z += 0.0041596125811338425 * (5.0 - Q.n_dr_0p2_0p4) * (Q.n_real_top30 - 22.0)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 9.0:
        z += -0.0026673567481338978 * (10.0 - Q.n_dr_0p2_0p4) * (9.0 - Q.n_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_dr_0p2_0p4 < 0.068491501734:
        z += 0.8030925393104553 * (10.0 - Q.n_dr_0p2_0p4) * (0.068491501734 - Q.z_dr_0p2_0p4)
    if Q.mass < 80.4 and Q.z_dr_0p2_0p4 < 0.037001823448:
        z += -0.5847809910774231 * (80.4 - Q.mass) * (0.037001823448 - Q.z_dr_0p2_0p4)
    if Q.z_dr_0_0p05 < 0.095732276142 and Q.max_dr > 0.193997652829:
        z += 0.814994215965271 * (0.095732276142 - Q.z_dr_0_0p05) * (Q.max_dr - 0.193997652829)
    if Q.mass < 101.049709320068 and Q.n_dr_0p2_0p4 > 10.0:
        z += -0.003007639432325959 * (101.049709320068 - Q.mass) * (Q.n_dr_0p2_0p4 - 10.0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.04887419566512108
    if Q.mass < 62.55:
        z += -0.06776379235088825 * Q.mass + 6.107339691402389
    if 62.55 <= Q.mass < 74.251806640625:
        z += -0.11752824671566486 * Q.mass + 9.220106311919166
    if 74.251806640625 <= Q.mass < 82.85408782959:
        z += -0.06419024057686329 * Q.mass + 5.259662993504402
    if 82.85408782959 <= Q.mass < 86.4:
        z += 0.01657143048942089 * Q.mass - 1.431771594285965
    if Q.mass_top20 >= 125.1:
        z += 0.032506223767995834 * Q.mass_top20 - 4.066528593376279
    if Q.mass_top40 < 77.936678808178:
        z += 0.003035531844943762 * Q.mass_top40 - 0.14586583433275258
    if 77.936678808178 <= Q.mass_top40 < 89.67879517394:
        z += -0.007725475821644068 * Q.mass_top40 + 0.6928113638304442
    if Q.LHA < 0.228402115913:
        z += -2.825361728668213 * Q.LHA + 0.6453185970474312
    if Q.LHA >= 0.404204003833:
        z += 22.421064376831055 * Q.LHA - 9.062683991312559
    if Q.log_sum_pt < 6.856374501323:
        z += 6.89503812789917 * Q.log_sum_pt - 47.27496360577774
    if Q.z_dr_0_0p05 >= 0.908491230011:
        z += -6.783810615539551 * Q.z_dr_0_0p05 + 6.163032450273206
    if Q.planar_flow >= 0.258818254187:
        z += -0.569035530090332 * Q.planar_flow + 0.14727678246835385
    if Q.girth2_top15 < 0.005788041138:
        z += 2.286661148071289 * Q.girth2_top15 - 0.013235288793702931
    if Q.girth2_top5 < 0.001000990214:
        z += -354.4710998535156 * Q.girth2_top5 + 0.35482210209918597
    if Q.girth2_top5 >= 0.016859196762:
        z += -12.520397186279297 * Q.girth2_top5 + 0.21108383970187386
    if Q.girth < 0.050483809784:
        z += 34.4371452331543 * Q.girth - 1.738518289454544
    if Q.girth2_top50 < 0.00742997247:
        z += -95.59761047363281 * Q.girth2_top50 + 0.7102876140168755
    if Q.D2 < 3.814159452915:
        z += -0.026010842993855476 * Q.D2 + 0.09920950268330175
    if Q.mass_top50 < 77.376408295162:
        z += 0.014483675360679626 * Q.mass_top50 - 0.9928716269106621
    if 77.376408295162 <= Q.mass_top50 < 80.4:
        z += -0.04227526858448982 * Q.mass_top50 + 3.398931594192982
    if Q.mass_top30 >= 138.381845700848:
        z += 0.023262739181518555 * Q.mass_top30 - 3.219140783995972
    if Q.C2 >= 0.061168736406:
        z += -13.218950271606445 * Q.C2 + 0.8085864847279167
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.064725840837:
        z += -0.2706509530544281 * (86.4 - Q.mass) * (0.064725840837 - Q.z_dr_0p1_0p2)
    if Q.mass_top20 > 125.1 and Q.C2 > 0.056027559564:
        z += -1.1711556911468506 * (Q.mass_top20 - 125.1) * (Q.C2 - 0.056027559564)
    if Q.mass_top40 < 89.67879517394 and Q.sum_pt_top2 < 501.625:
        z += -4.362778054201044e-05 * (89.67879517394 - Q.mass_top40) * (501.625 - Q.sum_pt_top2)
    if Q.LHA > 0.404204003833 and Q.lam2 < 0.003687604901:
        z += 12401.134765625 * (Q.LHA - 0.404204003833) * (0.003687604901 - Q.lam2)
    if Q.mass < 86.4 and Q.z_dr_0p05_0p1 > 0.299250295758:
        z += 0.08232493698596954 * (86.4 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.299250295758)
    if Q.mass_top20 > 125.1 and Q.pt_2 > 111.75:
        z += -0.0022431141696870327 * (Q.mass_top20 - 125.1) * (Q.pt_2 - 111.75)
    if Q.mass_top20 > 125.1 and Q.pt_2 > 137.5:
        z += 0.0023073572665452957 * (Q.mass_top20 - 125.1) * (Q.pt_2 - 137.5)
    if Q.mass_top20 > 125.1 and Q.pt_2 > 73.6875:
        z += 0.0007233973010443151 * (Q.mass_top20 - 125.1) * (Q.pt_2 - 73.6875)
    if Q.z_dr_0p1_0p2 > 0.688217741251 and Q.min_pair_mass > 0.740073079621:
        z += -1.7269080877304077 * (Q.z_dr_0p1_0p2 - 0.688217741251) * (Q.min_pair_mass - 0.740073079621)
    if Q.mass_top50 < 77.376408295162 and Q.z_dr_0p05_0p1 < 0.850921532512:
        z += 0.025928674265742302 * (77.376408295162 - Q.mass_top50) * (0.850921532512 - Q.z_dr_0p05_0p1)
    if Q.log_sum_pt < 6.856374501323 and Q.n_particles > 62.0:
        z += 1.2270622253417969 * (6.856374501323 - Q.log_sum_pt) * (Q.n_particles - 62.0)
    if Q.girth2_top10 < 0.000743190527 and Q.n_particles > 51.0:
        z += -163.8215789794922 * (0.000743190527 - Q.girth2_top10) * (Q.n_particles - 51.0)
    if Q.mass_top40 < 89.67879517394 and Q.n_particles > 22.0:
        z += -0.00019294413505122066 * (89.67879517394 - Q.mass_top40) * (Q.n_particles - 22.0)
    if Q.girth < 0.050483809784 and Q.n_particles < 64.0:
        z += -0.5339574217796326 * (0.050483809784 - Q.girth) * (64.0 - Q.n_particles)
    if Q.mass_top30 > 138.381845700848 and Q.dr_3 < 0.186661871599:
        z += 0.2005181908607483 * (Q.mass_top30 - 138.381845700848) * (0.186661871599 - Q.dr_3)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.80752694606781
    if 74.251806640625 <= Q.mass < 136.785:
        z += 0.026997892186045647 * Q.mass - 2.004642270302702
    if 136.785 <= Q.mass < 143.787612915039:
        z += -0.004702968522906303 * Q.mass + 2.33155996177129
    if 143.787612915039 <= Q.mass < 160.8:
        z += -0.1392380576580763 * Q.mass + 21.676039281829382
    if 160.8 <= Q.mass < 172.8:
        z += -0.1597974505275488 * Q.mass + 24.98198965524056
    if Q.mass >= 172.8:
        z += 0.015607709065079689 * Q.mass - 5.328021922365643
    if Q.z_top10_slots >= 0.829873578817:
        z += -3.601567029953003 * Q.z_top10_slots + 2.988845320496412
    if Q.sum_pt < 986.05654296875:
        z += 0.0346402651630342 * Q.sum_pt - 35.30607632283993
    if 986.05654296875 <= Q.sum_pt < 1012.672900390625:
        z += 0.02936453279107809 * Q.sum_pt - 30.103905898520562
    if 1012.672900390625 <= Q.sum_pt < 1052.889428710937:
        z += 0.00913155172020197 * Q.sum_pt - 9.614514273927824
    if Q.sum_pt >= 1260.540869140625:
        z += -0.0017841542139649391 * Q.sum_pt + 2.248999303552273
    if Q.log_sum_pt < 6.811175180312:
        z += -11.637987613677979 * Q.log_sum_pt + 77.75607718214509
    if 6.811175180312 <= Q.log_sum_pt < 7.017257672702:
        z += 7.3383002281188965 * Q.log_sum_pt - 51.49474358035816
    if Q.sum_pt_top40 < 956.21328125:
        z += -0.004078898346051574 * Q.sum_pt_top40 + 4.564360743087809
    if 956.21328125 <= Q.sum_pt_top40 < 1053.04736328125:
        z += -0.006857750471681356 * Q.sum_pt_top40 + 7.221536052244801
    if 0.078528833221 <= Q.mass_over_sum_pt < 0.170880120467:
        z += 2.1956050395965576 * Q.mass_over_sum_pt - 0.17241830197366517
    if Q.mass_over_sum_pt >= 0.170880120467:
        z += 294.2883174419403 * Q.mass_over_sum_pt - 50.085256184818945
    if Q.mass_top10 >= 56.921923720802:
        z += 0.013257679529488087 * Q.mass_top10 - 0.754652622892359
    if Q.C2 >= 0.072798889503:
        z += -2.0306522846221924 * Q.C2 + 0.14782923128722547
    if 97.930041729355 <= Q.mass_top50 < 136.785:
        z += -0.034206826239824295 * Q.mass_top50 + 3.349875921094789
    if 136.785 <= Q.mass_top50 < 168.969765712694:
        z += 0.06168385222554207 * Q.mass_top50 - 9.766530532790348
    if Q.mass_top50 >= 168.969765712694:
        z += 0.012339059263467789 * Q.mass_top50 - 1.4287524268472662
    z += 0.01088486798107624 * Q.n_particles
    if Q.e2_sq < 0.009606007381:
        z += -120.901123046875 * Q.e2_sq + 1.1613770803594705
    if Q.mass_over_sum_pt_sq >= 0.029200015571:
        z += -782.2843017578125 * Q.mass_over_sum_pt_sq + 22.84271379227699
    if Q.sum_pt_top50 < 959.095727539062:
        z += -0.015314930002205074 * Q.sum_pt_top50 + 15.49782202052259
    if 959.095727539062 <= Q.sum_pt_top50 < 1013.915698242188:
        z += -0.014763562940061092 * Q.sum_pt_top50 + 14.969008226914532
    if Q.sum_pt_top30 >= 1111.24501953125:
        z += 0.0009568559471517801 * Q.sum_pt_top30 - 1.0633014056812726
    if Q.n_dr_0p2_0p4 < 4.0:
        z += 0.1370030790567398 * Q.n_dr_0p2_0p4 - 0.5480123162269592
    if Q.sum_pt_top20 >= 956.50615234375:
        z += -0.0008767282706685364 * Q.sum_pt_top20 + 0.8385959848281517
    if Q.girth < 0.073744720221:
        z += -1.2754032611846924 * Q.girth + 0.09405425666501613
    if Q.sum_pt < 1085.12490234375 and Q.tau21 > 0.185845967382:
        z += 0.004135936498641968 * (1085.12490234375 - Q.sum_pt) * (Q.tau21 - 0.185845967382)
    if Q.z_top10_slots > 0.829873578817 and Q.D2 < 3.814159452915:
        z += 1.8738834857940674 * (Q.z_top10_slots - 0.829873578817) * (3.814159452915 - Q.D2)
    if Q.sum_pt < 1085.12490234375 and Q.pt_3 < 89.6875:
        z += -3.5394608858041465e-05 * (1085.12490234375 - Q.sum_pt) * (89.6875 - Q.pt_3)
    if Q.mass > 143.787612915039 and Q.sum_pt < 1007.788464355469:
        z += 0.00022831537353340536 * (Q.mass - 143.787612915039) * (1007.788464355469 - Q.sum_pt)
    if Q.mass > 74.251806640625 and Q.sum_pt < 1017.43466796875:
        z += -3.0487462936434895e-05 * (Q.mass - 74.251806640625) * (1017.43466796875 - Q.sum_pt)
    if Q.log_sum_pt < 6.811175180312 and Q.dr_5 < 0.051086217058:
        z += -256.75396728515625 * (6.811175180312 - Q.log_sum_pt) * (0.051086217058 - Q.dr_5)
    if Q.sum_pt < 1012.672900390625 and Q.dr_5 < 0.051086217058:
        z += -0.10963618755340576 * (1012.672900390625 - Q.sum_pt) * (0.051086217058 - Q.dr_5)
    if Q.sum_pt > 1260.540869140625 and Q.dr_7 < 0.082575402011:
        z += -0.038604725152254105 * (Q.sum_pt - 1260.540869140625) * (0.082575402011 - Q.dr_7)
    if Q.mass > 160.8 and Q.D2 < 5.378974604607:
        z += -0.0044920966029167175 * (Q.mass - 160.8) * (5.378974604607 - Q.D2)
    if Q.mass > 74.251806640625 and Q.D2 > 0.603279101849:
        z += 0.0001201243867399171 * (Q.mass - 74.251806640625) * (Q.D2 - 0.603279101849)
    if Q.sum_pt_top40 < 1053.04736328125 and Q.D2 < 5.378974604607:
        z += -0.0007594212656840682 * (1053.04736328125 - Q.sum_pt_top40) * (5.378974604607 - Q.D2)
    if Q.sum_pt_top50 < 959.095727539062 and Q.D2 < 4.450168704987:
        z += 0.003529584500938654 * (959.095727539062 - Q.sum_pt_top50) * (4.450168704987 - Q.D2)
    if Q.sum_pt_top50 < 1013.915698242188 and Q.D2 < 4.450168704987:
        z += -0.0018179122125729918 * (1013.915698242188 - Q.sum_pt_top50) * (4.450168704987 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = -0.019405929371714592
    if Q.girth2 < 0.007877041167:
        z += -128.47793579101562 * Q.girth2 + 1.0120259892770127
    if Q.mass_over_sum_pt < 0.090467494167:
        z += 12.67429256439209 * Q.mass_over_sum_pt + 1.597564120833695
    if 0.090467494167 <= Q.mass_over_sum_pt < 0.09795414517:
        z += -155.80306339263916 * Q.mass_over_sum_pt + 16.839288338148002
    if 0.09795414517 <= Q.mass_over_sum_pt < 0.118225939153:
        z += -62.168938636779785 * Q.mass_over_sum_pt + 7.667437688946662
    if 0.118225939153 <= Q.mass_over_sum_pt < 0.140939019879:
        z += -13.976815223693848 * Q.mass_over_sum_pt + 1.969878638657297
    if Q.girth2_top50 < 0.00634934989:
        z += 434.0122375488281 * Q.girth2_top50 - 3.541344994914235
    if 0.00634934989 <= Q.girth2_top50 < 0.009262053166:
        z += 269.7320556640625 * Q.girth2_top50 - 2.4982726401350184
    if Q.mass_top40 < 80.890431271924:
        z += 0.03737016557715833 * Q.mass_top40 - 3.359834192686276
    if 80.890431271924 <= Q.mass_top40 < 83.325535102591:
        z += -0.0063401630613952875 * Q.mass_top40 + 0.1759131419238571
    if 83.325535102591 <= Q.mass_top40 < 91.288535717504:
        z += 0.014249559259042144 * Q.mass_top40 - 1.5397364880403537
    if 91.288535717504 <= Q.mass_top40 < 111.24867219155:
        z += 0.011969611980021 * Q.mass_top40 - 1.331603439425406
    if Q.n_dr_0p2_0p4 < 21.0:
        z += -0.02291468158364296 * Q.n_dr_0p2_0p4 + 0.48120831325650215
    if Q.mass_top50 < 117.048742792994:
        z += -0.019791550934314728 * Q.mass_top50 + 2.3165761547850447
    if Q.mass < 80.4:
        z += 0.1186563540250063 * Q.mass - 10.37270804945619
    if 80.4 <= Q.mass < 89.741833496094:
        z += 0.13847612030804157 * Q.mass - 11.966217258612225
    if 89.741833496094 <= Q.mass < 91.034691238403:
        z += 0.05269267596304417 * Q.mass - 4.267853679482021
    if 91.034691238403 <= Q.mass < 136.785:
        z += -0.011562934145331383 * Q.mass + 1.5816359470691532
    if 0.240474711359 <= Q.max_dr < 0.402178311348:
        z += -2.1840877532958984 * Q.max_dr + 0.525217872056558
    if Q.max_dr >= 0.402178311348:
        z += -2.5798083543777466 * Q.max_dr + 0.6843681151652712
    if Q.lam2 < 0.001163277284:
        z += -434.2655334472656 * Q.lam2 + 0.8662750434969639
    if 0.001163277284 <= Q.lam2 < 0.002396991421:
        z += -292.6965026855469 * Q.lam2 + 0.7015910058939593
    if Q.lam1 < 0.006189818106:
        z += -172.6614761352539 * Q.lam1 + 0.5827020786015108
    if 0.006189818106 <= Q.lam1 < 0.007259287357:
        z += 29.400718688964844 * Q.lam1 - 0.6680261534595379
    if 0.007259287357 <= Q.lam1 < 0.011744050682:
        z += 101.3649673461914 * Q.lam1 - 1.1904353138929469
    if Q.sum_pt_top50 >= 976.277001953125:
        z += -0.004912116564810276 * Q.sum_pt_top50 + 4.7955864331372595
    if Q.girth2_top30 < 0.00608841615:
        z += 47.486961364746094 * Q.girth2_top30 - 1.1441568574153074
    if 0.00608841615 <= Q.girth2_top30 < 0.012157872869:
        z += 140.8752899169922 * Q.girth2_top30 - 1.7127438651943085
    if Q.mass_top30 < 91.19:
        z += -0.01050887443125248 * Q.mass_top30 + 0.9583042593859136
    if Q.girth < 0.085894044489:
        z += 9.718912124633789 * Q.girth - 0.8347966704179761
    if Q.e2 < 0.03263075389:
        z += -12.756733894348145 * Q.e2 + 0.4162618441466956
    if Q.log_sum_pt >= 6.935549248787:
        z += 6.874091625213623 * Q.log_sum_pt - 47.67560100734335
    z += 0.0004785878409165889 * Q.sum_pt_top5
    if Q.girth2_top20 < 0.016559833876:
        z += 4.981640338897705 * Q.girth2_top20 - 0.08249513644212635
    if Q.girth2_top10 < 0.006876086349:
        z += -35.8571662902832 * Q.girth2_top10 + 0.2465569716424393
    if Q.z_top50_slots < 0.958653609576:
        z += -87.08419036865234 * Q.z_top50_slots + 83.4835734339121
    if Q.max_dr > 0.240474711359 and Q.mass_top10 < 56.921923720802:
        z += 0.033802516758441925 * (Q.max_dr - 0.240474711359) * (56.921923720802 - Q.mass_top10)
    if Q.girth2 < 0.007877041167 and Q.n_dr_0p05_0p1 > 5.0:
        z += -2.3559563159942627 * (0.007877041167 - Q.girth2) * (Q.n_dr_0p05_0p1 - 5.0)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.n_real_top40 < 40.0:
        z += -0.0006537477602250874 * (21.0 - Q.n_dr_0p2_0p4) * (40.0 - Q.n_real_top40)
    if Q.mass_top50 < 117.048742792994 and Q.mass_top2 > 28.78966323496:
        z += -0.0009239358478225768 * (117.048742792994 - Q.mass_top50) * (Q.mass_top2 - 28.78966323496)
    if Q.mass_top40 < 91.288535717504 and Q.z_dr_0p2_0p4 < 0.068491501734:
        z += -0.044052816927433014 * (91.288535717504 - Q.mass_top40) * (0.068491501734 - Q.z_dr_0p2_0p4)
    if Q.mass < 91.034691238403 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += 0.016530130058526993 * (91.034691238403 - Q.mass) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += -0.0006829784251749516 * (21.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.girth2_top20 < 0.016559833876 and Q.tau21 < 0.6418633461:
        z += -102.18321990966797 * (0.016559833876 - Q.girth2_top20) * (0.6418633461 - Q.tau21)
    if Q.lam1 < 0.006189818106 and Q.z_top50_slots > 0.985099030959:
        z += 7941.7392578125 * (0.006189818106 - Q.lam1) * (Q.z_top50_slots - 0.985099030959)
    if Q.girth2_top20 < 0.016559833876 and Q.z_top50_slots > 0.970443639316:
        z += -1459.2598876953125 * (0.016559833876 - Q.girth2_top20) * (Q.z_top50_slots - 0.970443639316)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.6605141758918762
    if Q.log_sum_pt < 7.017257672702:
        z += -5.097611427307129 * Q.log_sum_pt + 35.77125290072434
    if Q.girth2_top5 < 0.001501708498:
        z += 924.7863616943359 * Q.girth2_top5 - 1.097472817710934
    if 0.001501708498 <= Q.girth2_top5 < 0.007164202106:
        z += -51.44142150878906 * Q.girth2_top5 + 0.3685367403089003
    if Q.z_dr_0p05_0p1 >= 0.850921532512:
        z += -5.8458333015441895 * Q.z_dr_0p05_0p1 + 4.974345431759666
    if Q.sum_pt < 1002.378515625:
        z += 0.00945968460291624 * Q.sum_pt - 9.48218461055185
    if Q.z_dr_0p2_0p4 < 0.019523000158:
        z += 54.097354888916016 * Q.z_dr_0p2_0p4 - 1.0561426680436894
    if Q.girth < 0.050483809784:
        z += 3.796375274658203 * Q.girth + 1.4877213831178615
    if 0.050483809784 <= Q.girth < 0.120745175332:
        z += -23.901853561401367 * Q.girth + 2.8860334990311967
    if Q.lam1 < 0.016493544356:
        z += 79.60523986816406 * Q.lam1 - 1.3129725547355835
    if Q.girth2_top10 < 0.00321372409:
        z += -67.58709335327148 * Q.girth2_top10 + 0.7255432761017403
    if 0.00321372409 <= Q.girth2_top10 < 0.007678543663:
        z += -113.85387420654297 * Q.girth2_top10 + 0.8742319442966496
    if Q.dr_0 < 0.052014814497:
        z += 9.055106163024902 * Q.dr_0 - 0.4709996673203818
    if Q.girth2_top50 < 0.013514311784:
        z += 207.42010498046875 * Q.girth2_top50 - 2.803139968976066
    if Q.sum_pt_top50 < 934.241552734375:
        z += -0.0024417350068688393 * Q.sum_pt_top50 + 2.281170304183024
    if Q.e2_sq < 0.006936724595:
        z += 627.4913940429688 * Q.e2_sq - 4.352734986208698
    if Q.z_dr_0p1_0p2 < 0.187280465662:
        z += -2.229340076446533 * Q.z_dr_0p1_0p2 + 0.4175118476358654
    if Q.mass_top50 < 71.795159472175:
        z += 0.030206996947526932 * Q.mass_top50 - 2.1687161630231997
    if Q.mass_top30 < 80.4:
        z += -0.008698981255292892 * Q.mass_top30 + 0.6993980929255486
    if Q.lam2 < 0.001776308492:
        z += -195.65296936035156 * Q.lam2 + 0.3475400309598083
    if Q.dr_3 < 0.05268713294:
        z += 7.179580211639404 * Q.dr_3 - 0.3782714970640386
    if Q.C2 < 0.087968891487:
        z += 1.6321581602096558 * Q.C2 - 0.14357914408510478
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.mass_top5 < 40.2:
        z += 0.06156064197421074 * (0.120343671367 - Q.z_dr_0p1_0p2) * (40.2 - Q.mass_top5)
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.planar_flow < 0.826191085385:
        z += 3.0503580570220947 * (0.120343671367 - Q.z_dr_0p1_0p2) * (0.826191085385 - Q.planar_flow)
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.sum_pt_top3 < 787.628125:
        z += -0.007041251286864281 * (0.120343671367 - Q.z_dr_0p1_0p2) * (787.628125 - Q.sum_pt_top3)
    if Q.girth2_top5 < 0.001501708498 and Q.n_dr_0p2_0p4 > 0.0:
        z += -18.021244049072266 * (0.001501708498 - Q.girth2_top5) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.girth2_top5 < 0.007164202106 and Q.sum_pt_top40 < 984.70087890625:
        z += 1.4334758520126343 * (0.007164202106 - Q.girth2_top5) * (984.70087890625 - Q.sum_pt_top40)
    if Q.z_dr_0_0p05 < 0.845900350809 and Q.log_sum_pt > 6.910130970417:
        z += 2.9066808223724365 * (0.845900350809 - Q.z_dr_0_0p05) * (Q.log_sum_pt - 6.910130970417)
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.z_dr_0p2_0p4 < 0.037001823448:
        z += -96.13214874267578 * (0.120343671367 - Q.z_dr_0p1_0p2) * (0.037001823448 - Q.z_dr_0p2_0p4)
    if Q.girth2_top5 < 0.001501708498 and Q.z_dr_0p2_0p4 < 0.193744690716:
        z += 6640.58740234375 * (0.001501708498 - Q.girth2_top5) * (0.193744690716 - Q.z_dr_0p2_0p4)
    if Q.z_dr_0_0p05 < 0.845900350809 and Q.sum_pt_top40 > 1095.686413574219:
        z += -0.0036180601455271244 * (0.845900350809 - Q.z_dr_0_0p05) * (Q.sum_pt_top40 - 1095.686413574219)
    if Q.girth < 0.050483809784 and Q.z_dr_0p2_0p4 > 0.068491501734:
        z += -9732.373046875 * (0.050483809784 - Q.girth) * (Q.z_dr_0p2_0p4 - 0.068491501734)
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.n_dr_0p4_up > 0.0:
        z += -3.015644073486328 * (0.120343671367 - Q.z_dr_0p1_0p2) * (Q.n_dr_0p4_up - 0.0)
    if Q.girth2_top5 < 0.007164202106 and Q.max_dr < 0.331585738063:
        z += -1395.690673828125 * (0.007164202106 - Q.girth2_top5) * (0.331585738063 - Q.max_dr)
    if Q.girth2_top50 < 0.013514311784 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += 2575.172607421875 * (0.013514311784 - Q.girth2_top50) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.e2_sq < 0.006936724595 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += -689.461669921875 * (0.006936724595 - Q.e2_sq) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.e2_sq < 0.006936724595 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += 6786.08349609375 * (0.006936724595 - Q.e2_sq) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.log_sum_pt < 7.017257672702 and Q.z_dr_0p2_0p4 < 0.019523000158:
        z += 135.66378784179688 * (7.017257672702 - Q.log_sum_pt) * (0.019523000158 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.120745175332 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += -145.51976013183594 * (0.120745175332 - Q.girth) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.n_dr_0p2_0p4 < 26.0:
        z += 0.28220275044441223 * (0.120343671367 - Q.z_dr_0p1_0p2) * (26.0 - Q.n_dr_0p2_0p4)
    if Q.sum_pt < 1002.378515625 and Q.tau32 < 0.661614120007:
        z += -0.033734094351530075 * (1002.378515625 - Q.sum_pt) * (0.661614120007 - Q.tau32)
    if Q.girth2_top10 < 0.007678543663 and Q.z_top30_slots < 0.986057513941:
        z += -942.9384155273438 * (0.007678543663 - Q.girth2_top10) * (0.986057513941 - Q.z_top30_slots)
    if Q.sum_pt < 1002.378515625 and Q.z_dr_0p2_0p4 < 0.019523000158:
        z += -0.23186342418193817 * (1002.378515625 - Q.sum_pt) * (0.019523000158 - Q.z_dr_0p2_0p4)
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
