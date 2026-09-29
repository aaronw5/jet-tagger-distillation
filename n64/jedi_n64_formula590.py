"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 100 if-statements per neuron, pruned), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.2% (the network: 81.1%); same class as the network for 93.9% of jets.

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
  Q.mass_top10             mass of the 10 hardest particles [GeV]
  Q.mass_top15             mass of the 15 hardest particles [GeV]
  Q.mass_top20             mass of the 20 hardest particles [GeV]
  Q.mass_top3              mass of the 3 hardest particles [GeV]
  Q.mass_top30             mass of the 30 hardest particles [GeV]
  Q.mass_top40             mass of the 40 hardest particles [GeV]
  Q.mass_top5              mass of the 5 hardest particles [GeV]
  Q.mass_top50             mass of the 50 hardest particles [GeV]
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.n_real_top50           number of real particles among the 50 hardest
  Q.pt_2                   pT of particle 2 [GeV]
  Q.pt_6                   pT of particle 6 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_2                    pT of particle 2 / total pT
  Q.z_6                    pT of particle 6 / total pT
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
  Q.dr_4                   ΔR of particle 4 from the jet axis
  Q.dr_5                   ΔR of particle 5 from the jet axis
  Q.phi_6                  Δφ of particle 6
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
  Q.n_dr_0p1_0p2           number of particles with 0.1 ≤ ΔR < 0.2
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_10          number of particles with pT > 10 GeV
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi2              pT-weighted mean Δφ²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
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
        z_top5=sum(zs[:5]),
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        mass_top10=mass_of(10),
        mass_top15=mass_of(15),
        mass_top20=mass_of(20),
        mass_top3=mass_of(3),
        mass_top30=mass_of(30),
        mass_top40=mass_of(40),
        mass_top5=mass_of(5),
        mass_top50=mass_of(50),
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        n_real_top50=sum(1 for x in pt[:50] if x > 0),
        pt_2=pt[2],
        pt_6=pt[6],
        pt_9=pt[9],
        z_2=z[2],
        z_6=z[6],
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
        dr_4=dr[4] if pt[4] > 0 else 0.0,
        dr_5=dr[5] if pt[5] > 0 else 0.0,
        phi_6=phi[6],
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
        n_dr_0p1_0p2=sum(1 for i in real if 0.1 <= dr[i] < 0.2),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_10=sum(1 for x in pt if x > 10),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi2=sum(z[i] * phi[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        pt_dispersion=math.sqrt(sum(x * x for x in z)),
    )


def neuron_0(Q):
    z = 1.1047765016555786
    if 74.251806640625 <= Q.mass < 78.261818313599:
        z += -0.03858550637960434 * Q.mass + 2.865043558828984
    if 78.261818313599 <= Q.mass < 80.4:
        z += -0.09282515197992325 * Q.mass + 7.109936848195141
    if 80.4 <= Q.mass < 91.034691238403:
        z += -0.12867480888962746 * Q.mass + 9.99224926373536
    if 91.034691238403 <= Q.mass < 92.85979309082:
        z += -0.1454029604792595 * Q.mass + 11.515091378686712
    if Q.mass >= 92.85979309082:
        z += -0.28492075949907303 * Q.mass + 24.470685328153206
    if Q.sum_pt < 1002.378515625:
        z += 0.013330654241144657 * Q.sum_pt - 13.47417027712072
    if 1002.378515625 <= Q.sum_pt < 1012.672900390625:
        z += 0.010861150920391083 * Q.sum_pt - 10.998793204132744
    if Q.n_dr_0p2_0p4 < 10.0:
        z += -0.043751999735832214 * Q.n_dr_0p2_0p4 + 0.43751999735832214
    if Q.girth2_top20 < 0.006043208873:
        z += 304.4556655883789 * Q.girth2_top20 - 1.98380168646215
    if 0.006043208873 <= Q.girth2_top20 < 0.007538018543:
        z += 96.2748031616211 * Q.girth2_top20 - 0.7257212514559748
    if Q.girth < 0.056600876898:
        z += 65.0865249633789 * Q.girth - 3.6839543871708136
    if Q.mass_over_sum_pt_sq < 0.008184367501:
        z += -474.9994201660156 * Q.mass_over_sum_pt_sq + 3.8875698174005824
    if Q.girth2_top40 < 0.00625977218:
        z += 107.50049591064453 * Q.girth2_top40 - 0.3903886989398966
    if 0.00625977218 <= Q.girth2_top40 < 0.008840538245:
        z += -109.47908782958984 * Q.girth2_top40 + 0.967854062985203
    if Q.z_top30_slots >= 0.920388080863:
        z += -3.6876020431518555 * Q.z_top30_slots + 3.3940249674830136
    if Q.width < 0.009614971338:
        z += 185.36187744140625 * Q.width - 1.7822491387569899
    if Q.n_particles < 62.0:
        z += -0.011852385476231575 * Q.n_particles + 0.7348478995263577
    if Q.LHA < 0.260146178237:
        z += -4.959659576416016 * Q.LHA + 1.2902364841611647
    if Q.mass_top50 < 82.04491364955:
        z += 0.01964441128075123 * Q.mass_top50 - 1.6117240272254805
    if Q.log_sum_pt < 7.017257672702:
        z += -5.153618335723877 * Q.log_sum_pt + 36.16426780853609
    if Q.girth2_top50 < 0.00742997247:
        z += -142.034423828125 * Q.girth2_top50 + 1.0553118588352808
    if 0.076966318366 <= Q.mass_over_sum_pt < 0.083299446175:
        z += -23.308917999267578 * Q.mass_over_sum_pt + 1.7940016034986161
    if Q.mass_over_sum_pt >= 0.083299446175:
        z += 29.820484161376953 * Q.mass_over_sum_pt - 2.6316481720919214
    if Q.mass_top20 < 70.421206773231:
        z += -0.008901769295334816 * Q.mass_top20 + 0.6268733361943719
    if Q.mass_top30 < 89.172933810464:
        z += 0.012241891585290432 * Q.mass_top30 - 1.09164538804998
    if Q.lam1 < 0.006716736591:
        z += 99.96376037597656 * Q.lam1 - 0.6714302470912776
    if Q.girth2_top40 < 0.00625977218 and Q.girth2_top3 < 0.002915531053:
        z += 39334.92578125 * (0.00625977218 - Q.girth2_top40) * (0.002915531053 - Q.girth2_top3)
    if Q.log_sum_pt < 7.017257672702 and Q.max_dr < 0.390781164169:
        z += 56.76068115234375 * (7.017257672702 - Q.log_sum_pt) * (0.390781164169 - Q.max_dr)
    if Q.mass_top50 < 82.04491364955 and Q.z_dr_0p05_0p1 < 0.212648361921:
        z += 0.02950269728899002 * (82.04491364955 - Q.mass_top50) * (0.212648361921 - Q.z_dr_0p05_0p1)
    if Q.mass_top50 < 82.04491364955 and Q.dr_1 > 0.161154452503:
        z += 1466.10498046875 * (82.04491364955 - Q.mass_top50) * (Q.dr_1 - 0.161154452503)
    if Q.sum_pt < 1012.672900390625 and Q.z_dr_0p1_0p2 > 0.088715460151:
        z += -0.012804349884390831 * (1012.672900390625 - Q.sum_pt) * (Q.z_dr_0p1_0p2 - 0.088715460151)
    if Q.mass > 92.85979309082 and Q.n_dr_0p2_0p4 < 7.0:
        z += -0.2411302924156189 * (Q.mass - 92.85979309082) * (7.0 - Q.n_dr_0p2_0p4)
    if Q.girth2_top20 < 0.006043208873 and Q.mass_top5 < 68.434224049685:
        z += 3.0666658878326416 * (0.006043208873 - Q.girth2_top20) * (68.434224049685 - Q.mass_top5)
    if Q.log_sum_pt < 6.989450376716 and Q.max_dr < 0.387360095978:
        z += -70.00405883789062 * (6.989450376716 - Q.log_sum_pt) * (0.387360095978 - Q.max_dr)
    if Q.mass > 92.85979309082 and Q.max_dr < 0.390781164169:
        z += -4.451551914215088 * (Q.mass - 92.85979309082) * (0.390781164169 - Q.max_dr)
    if Q.sum_pt_top40 < 858.826171875 and Q.n_dr_0p1_0p2 < 33.0:
        z += -0.002280967542901635 * (858.826171875 - Q.sum_pt_top40) * (33.0 - Q.n_dr_0p1_0p2)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.3158565759658813
    if 6.811175180312 <= Q.log_sum_pt < 6.89371369877:
        z += -9.331071853637695 * Q.log_sum_pt + 63.55556501520496
    if 6.89371369877 <= Q.log_sum_pt < 6.910130970417:
        z += 12.111543655395508 * Q.log_sum_pt - 84.2636872568753
    if 6.910130970417 <= Q.log_sum_pt < 6.959293500649:
        z += 41.20644760131836 * Q.log_sum_pt - 285.3132840949046
    if 6.959293500649 <= Q.log_sum_pt < 6.989450376716:
        z += 27.44421672821045 * Q.log_sum_pt - 189.53788022525367
    if 6.989450376716 <= Q.log_sum_pt < 7.062574317998:
        z += 20.1397967338562 * Q.log_sum_pt - 138.48399914402245
    if 7.062574317998 <= Q.log_sum_pt < 7.139296169016:
        z += 13.42309284210205 * Q.log_sum_pt - 91.04677873652237
    if Q.log_sum_pt >= 7.139296169016:
        z += 7.006519317626953 * Q.log_sum_pt - 45.23695995502783
    if Q.sum_pt_top50 >= 959.095727539062:
        z += -0.014357679523527622 * Q.sum_pt_top50 + 13.770389088390417
    if Q.sum_pt_top2 < 689.25:
        z += -0.0026649218052625656 * Q.sum_pt_top2 + 1.8367973542772233
    if Q.sum_pt_top30 < 1073.4734375:
        z += -0.005343451164662838 * Q.sum_pt_top30 + 5.736052889843996
    if Q.girth2_top15 < 0.003270031267:
        z += 236.21775817871094 * Q.girth2_top15 - 0.7724394550650298
    if Q.z_top20_slots < 0.957678701144:
        z += 4.059595584869385 * Q.z_top20_slots - 3.8877882268876296
    if Q.max_dr < 0.43572281599:
        z += 4.087549686431885 * Q.max_dr - 1.7810386598711423
    if Q.z_top50_slots >= 0.958653609576:
        z += -40.29689407348633 * Q.z_top50_slots + 38.63076295824939
    if Q.girth2_top3 < 0.000823693417:
        z += 855.7885131835938 * Q.girth2_top3 - 0.7049073646535439
    if Q.z_dr_0_0p05 >= 0.878906026483:
        z += -4.338717460632324 * Q.z_dr_0_0p05 + 3.813324923356768
    if Q.n_dr_0p2_0p4 < 13.0:
        z += 0.02991739846765995 * Q.n_dr_0p2_0p4 - 0.38892618007957935
    if Q.LHA < 0.404204003833:
        z += -7.390407085418701 * Q.LHA + 2.987232133882011
    if Q.mass_top30 < 80.4:
        z += -0.03945041820406914 * Q.mass_top30 + 3.171813623607159
    if Q.n_pt_above_10 < 16.0:
        z += 0.09052275121212006 * Q.n_pt_above_10 - 2.8062052875757217
    if 16.0 <= Q.n_pt_above_10 < 31.0:
        z += 0.038282863795757294 * Q.n_pt_above_10 - 1.9703670889139175
    if Q.n_pt_above_10 >= 31.0:
        z += -0.05223988741636276 * Q.n_pt_above_10 + 0.8358381986618042
    if Q.girth2_top10 < 0.000412964127:
        z += -1669.5632247924805 * Q.girth2_top10 + 1.300857716563698
    if 0.000412964127 <= Q.girth2_top10 < 0.007678543663:
        z += -84.14855194091797 * Q.girth2_top10 + 0.646138330256562
    if Q.girth2_top5 < 0.000657050184:
        z += -870.487548828125 * Q.girth2_top5 + 0.5719540041272285
    if Q.sum_pt_top5 < 902.40625:
        z += -0.0011079444084316492 * Q.sum_pt_top5 + 0.9998159588212729
    if Q.sum_pt >= 949.91689453125:
        z += 0.010669671930372715 * Q.sum_pt - 10.135301625766898
    if Q.z_top40_slots < 0.967499609196:
        z += 17.27095603942871 * Q.z_top40_slots - 16.709643218588575
    if Q.mass < 120.60000000000001:
        z += 0.01840202696621418 * Q.mass - 2.2192844521254305
    if Q.sum_pt_top40 >= 1032.405224609375:
        z += 0.0017671408131718636 * Q.sum_pt_top40 - 1.8244054081390912
    if Q.n_particles >= 26.0:
        z += 0.020483793690800667 * Q.n_particles - 0.5325786359608173
    if Q.sum_pt_top2 < 689.25 and Q.n_dr_0p2_0p4 < 7.0:
        z += -0.00017103985010180622 * (689.25 - Q.sum_pt_top2) * (7.0 - Q.n_dr_0p2_0p4)
    if Q.n_particles > 38.0 and Q.z_top50_slots < 0.985099030959:
        z += 1.8753725290298462 * (Q.n_particles - 38.0) * (0.985099030959 - Q.z_top50_slots)
    if Q.z_top30_slots > 0.934183811419 and Q.mass_top5 > 22.1834155076:
        z += 0.4696747064590454 * (Q.z_top30_slots - 0.934183811419) * (Q.mass_top5 - 22.1834155076)
    if Q.max_dr < 0.43572281599 and Q.z_dr_0p2_0p4 < 0.193744690716:
        z += 30.946338653564453 * (0.43572281599 - Q.max_dr) * (0.193744690716 - Q.z_dr_0p2_0p4)
    if Q.mass_top20 < 40.2 and Q.mean_phi2 < 0.005860335776:
        z += 4.725652694702148 * (40.2 - Q.mass_top20) * (0.005860335776 - Q.mean_phi2)
    if Q.n_pt_above_10 < 31.0 and Q.mean_phi2 < 0.017422899418:
        z += 1.1838957071304321 * (31.0 - Q.n_pt_above_10) * (0.017422899418 - Q.mean_phi2)
    if Q.z_top30_slots > 0.934183811419 and Q.mass_top10 < 91.19:
        z += -0.30868735909461975 * (Q.z_top30_slots - 0.934183811419) * (91.19 - Q.mass_top10)
    if Q.mass_top30 < 80.4 and Q.mass_top5 < 68.434224049685:
        z += -0.0006165183731354773 * (80.4 - Q.mass_top30) * (68.434224049685 - Q.mass_top5)
    if Q.pt_9 < 34.0625 and Q.pt1_dr01 < 28.393960910299:
        z += -0.0006415644311346114 * (34.0625 - Q.pt_9) * (28.393960910299 - Q.pt1_dr01)
    if Q.z_top50_slots > 0.958653609576 and Q.girth2_top3 < 0.001155056594:
        z += 25898.173828125 * (Q.z_top50_slots - 0.958653609576) * (0.001155056594 - Q.girth2_top3)
    if Q.z_top30_slots > 0.934183811419 and Q.girth2_top3 < 0.001155056594:
        z += -10001.9267578125 * (Q.z_top30_slots - 0.934183811419) * (0.001155056594 - Q.girth2_top3)
    if Q.z_top30_slots > 0.934183811419 and Q.girth2_top5 < 0.016859196762:
        z += 1590.119384765625 * (Q.z_top30_slots - 0.934183811419) * (0.016859196762 - Q.girth2_top5)
    if Q.z_top30_slots > 0.934183811419 and Q.D2 < 1.976207274199:
        z += 5.667202949523926 * (Q.z_top30_slots - 0.934183811419) * (1.976207274199 - Q.D2)
    if Q.n_particles > 38.0 and Q.dr_4 < 0.199975347593:
        z += 0.08223342150449753 * (Q.n_particles - 38.0) * (0.199975347593 - Q.dr_4)
    if Q.girth2_top15 < 0.003270031267 and Q.n_real_top40 > 29.0:
        z += 39.93625259399414 * (0.003270031267 - Q.girth2_top15) * (Q.n_real_top40 - 29.0)
    if Q.sum_pt_top2 < 689.25 and Q.mass_top5 > 2.164373545539:
        z += 1.3709360246139113e-05 * (689.25 - Q.sum_pt_top2) * (Q.mass_top5 - 2.164373545539)
    if Q.n_dr_0p2_0p4 < 13.0 and Q.girth2_top3 < 0.000823693417:
        z += 71.15545654296875 * (13.0 - Q.n_dr_0p2_0p4) * (0.000823693417 - Q.girth2_top3)
    if Q.n_particles > 38.0 and Q.girth2_top2 < 0.022605352903:
        z += 1.9017916917800903 * (Q.n_particles - 38.0) * (0.022605352903 - Q.girth2_top2)
    if Q.n_pt_above_10 < 31.0 and Q.lam1 < 0.004673423215:
        z += 17.625011444091797 * (31.0 - Q.n_pt_above_10) * (0.004673423215 - Q.lam1)
    if Q.LHA < 0.404204003833 and Q.lam2 < 0.001776308492:
        z += -860.3751831054688 * (0.404204003833 - Q.LHA) * (0.001776308492 - Q.lam2)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.45693621039390564
    if Q.log_sum_pt >= 6.903422848462:
        z += -3.881737232208252 * Q.log_sum_pt + 26.79727350055209
    if 995.676940917969 <= Q.sum_pt < 1017.43466796875:
        z += 0.002596596023067832 * Q.sum_pt - 2.585370785047943
    if 1017.43466796875 <= Q.sum_pt < 1066.481811523438:
        z += 0.017552669858559966 * Q.sum_pt - 17.80219880197799
    if 1066.481811523438 <= Q.sum_pt < 1115.722741699219:
        z += 0.005774849792942405 * Q.sum_pt - 5.241367922601077
    if Q.sum_pt >= 1115.722741699219:
        z += 0.004304882837459445 * Q.sum_pt - 3.6012923608223755
    if Q.girth2_top15 < 0.006142801866:
        z += -196.95243072509766 * Q.girth2_top15 + 2.253427738591218
    if 0.006142801866 <= Q.girth2_top15 < 0.015638355144:
        z += -109.9028091430664 * Q.girth2_top15 + 1.7186991607025226
    if Q.mass < 91.19:
        z += 0.013748289085924625 * Q.mass - 0.8786002993861488
    if 91.19 <= Q.mass < 92.85979309082:
        z += 0.04556157533079386 * Q.mass - 3.779653872055774
    if 92.85979309082 <= Q.mass < 125.1:
        z += -0.013994469307363033 * Q.mass + 1.7507081103511153
    if Q.mass_top50 < 86.4:
        z += -0.004948965739458799 * Q.mass_top50 + 0.4275906398892403
    if Q.mass_over_sum_pt < 0.074356165682:
        z += 34.4597601890564 * Q.mass_over_sum_pt - 3.441282426670241
    if 0.074356165682 <= Q.mass_over_sum_pt < 0.09795414517:
        z += 9.681485652923584 * Q.mass_over_sum_pt - 1.5988649399474681
    if 0.09795414517 <= Q.mass_over_sum_pt < 0.140939019879:
        z += 15.133771896362305 * Q.mass_over_sum_pt - 2.1329389781456585
    if 908.8125 <= Q.sum_pt_top20 < 1129.275:
        z += -0.002293862635269761 * Q.sum_pt_top20 + 2.0846910362160997
    if Q.sum_pt_top20 >= 1129.275:
        z += -0.0015039064455777407 * Q.sum_pt_top20 + 1.1926132601016435
    if 997.018872070312 <= Q.sum_pt_top50 < 1038.855053710938:
        z += -0.004407956264913082 * Q.sum_pt_top50 + 4.394815583378906
    if 1038.855053710938 <= Q.sum_pt_top50 < 1107.225842285156:
        z += -0.0004830635152757168 * Q.sum_pt_top50 + 0.3174209151447105
    if Q.sum_pt_top50 >= 1107.225842285156:
        z += -0.0024033670779317617 * Q.sum_pt_top50 + 2.443630644749736
    if Q.max_dr < 0.402178311348:
        z += -7.620748519897461 * Q.max_dr + 3.064899770940131
    if Q.mass_top40 >= 160.8:
        z += -0.14561106264591217 * Q.mass_top40 + 23.41425887346268
    if Q.sum_pt_top15 >= 1003.329150390625:
        z += 0.012596183456480503 * Q.sum_pt_top15 - 12.63811804555503
    if Q.mass_top30 < 91.697531419407:
        z += -0.011849443428218365 * Q.mass_top30 + 1.0865647110615393
    if Q.lam1 < 0.006189818106:
        z += 187.23257446289062 * Q.lam1 - 1.1589355794433935
    if Q.sum_pt_top40 >= 1069.67119140625:
        z += 0.010012621060013771 * Q.sum_pt_top40 - 10.71021229836424
    if Q.mass_top20 >= 125.1:
        z += 0.010036305524408817 * Q.mass_top20 - 1.255541821103543
    if Q.log_sum_pt > 6.903422848462 and Q.girth2_top10 < 0.01976735495:
        z += 343.7209167480469 * (Q.log_sum_pt - 6.903422848462) * (0.01976735495 - Q.girth2_top10)
    if Q.log_sum_pt > 6.903422848462 and Q.pt_6 < 42.78125:
        z += 0.2735529839992523 * (Q.log_sum_pt - 6.903422848462) * (42.78125 - Q.pt_6)
    return max(0.0, z)


def neuron_3(Q):
    z = 0.2007775604724884
    if Q.lam2 < 0.000615484055:
        z += -1096.7064208984375 * Q.lam2 + 0.6750053150791071
    if Q.tau21 < 0.428145796061:
        z += 1.8846129179000854 * Q.tau21 - 0.8068890980011761
    if Q.mass_over_sum_pt_sq < 0.00750911433:
        z += -159.6396026611328 * Q.mass_over_sum_pt_sq + 1.1987520279782187
    if Q.mass_top20 < 45.595:
        z += -0.020008688792586327 * Q.mass_top20 + 0.9122961654979735
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_top50_slots > 0.978741004761:
        z += 7.209311008453369 * (5.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741004761)
    if Q.mass_over_sum_pt_sq < 0.00750911433 and Q.mass_top20 < 136.785:
        z += -1.3520809412002563 * (0.00750911433 - Q.mass_over_sum_pt_sq) * (136.785 - Q.mass_top20)
    if Q.z_top20_slots > 0.967685186161 and Q.girth2_top3 < 0.000823693417:
        z += 11425.3525390625 * (Q.z_top20_slots - 0.967685186161) * (0.000823693417 - Q.girth2_top3)
    return max(0.0, z)


def neuron_4(Q):
    z = 2.0596823692321777
    if Q.mass_top15 < 69.027163795459:
        z += -0.00960248801857233 * Q.mass_top15 + 0.6628325133019248
    if Q.mass_top40 < 62.55:
        z += 0.14597368985414505 * Q.mass_top40 - 9.295315111678123
    if 62.55 <= Q.mass_top40 < 83.325535102591:
        z += -0.0061769261956214905 * Q.mass_top40 + 0.22170592223477348
    if 83.325535102591 <= Q.mass_top40 < 94.642533639752:
        z += 0.025889351963996887 * Q.mass_top40 - 2.450233864163955
    if Q.mass_top40 >= 136.785:
        z += -0.012675434350967407 * Q.mass_top40 + 1.7338092876970768
    if Q.mass_top30 < 68.286969674465:
        z += 0.00011481763795018196 * Q.mass_top30 - 0.6332849129838287
    if 68.286969674465 <= Q.mass_top30 < 152.688263064041:
        z += 0.00741036469116807 * Q.mass_top30 - 1.131475713365551
    if Q.mass < 74.251806640625:
        z += 0.08447556011378765 * Q.mass - 6.325239064419335
    if 74.251806640625 <= Q.mass < 78.261818313599:
        z += 0.06373208574950695 * Q.mass - 4.784998616868002
    if 78.261818313599 <= Q.mass < 80.784643554688:
        z += 0.052323512732982635 * Q.mass - 3.8921429482313483
    if 80.784643554688 <= Q.mass < 86.4:
        z += 0.036649152636528015 * Q.mass - 2.6258953548914366
    if 86.4 <= Q.mass < 91.19:
        z += -0.01187916100025177 * Q.mass + 1.5669509433263369
    if 91.19 <= Q.mass < 101.049709320068:
        z += -0.0490572527050972 * Q.mass + 4.957221125891191
    if Q.mass >= 143.787612915039:
        z += 0.038533784449100494 * Q.mass - 5.540680882518812
    if Q.D2 < 6.916121244431:
        z += -0.14815934002399445 * Q.D2 + 1.0246879591008242
    if Q.girth2_top40 < 0.012926423095:
        z += -118.80123138427734 * Q.girth2_top40 + 1.5356749810801615
    if Q.mass_top20 >= 103.674910639856:
        z += -0.024448972195386887 * Q.mass_top20 + 2.534745007593059
    if Q.girth2_top15 < 0.007277630044:
        z += 20.42646026611328 * Q.girth2_top15 + 0.3978002664155804
    if 0.007277630044 <= Q.girth2_top15 < 0.015638355144:
        z += -65.35993957519531 * Q.girth2_top15 + 1.0221219472672847
    if Q.z_dr_0p2_0p4 < 0.068491501734:
        z += 4.537951469421387 * Q.z_dr_0p2_0p4 - 0.31081111093668273
    if Q.girth < 0.061710142531:
        z += 38.386929512023926 * Q.girth - 3.1190579731827337
    if 0.061710142531 <= Q.girth < 0.120745175332:
        z += 12.707625389099121 * Q.girth - 1.534384455660148
    if 0.115200825015 <= Q.LHA < 0.404204003833:
        z += -8.930012702941895 * Q.LHA + 1.0287448307733362
    if Q.LHA >= 0.404204003833:
        z += -18.364041328430176 * Q.LHA + 4.842016973470833
    if Q.e2 < 0.025159193203:
        z += -7.179676055908203 * Q.e2 - 0.36129800056116246
    if 0.025159193203 <= Q.e2 < 0.036805817112:
        z += 17.81953239440918 * Q.e2 - 0.9902579158847676
    if 0.036805817112 <= Q.e2 < 0.055571487173:
        z += 38.76028823852539 * Q.e2 - 1.760999545670354
    if Q.e2 >= 0.055571487173:
        z += 20.94075584411621 * Q.e2 - 0.7707416297855864
    if 0.06030418859 <= Q.mass_over_sum_pt < 0.09795414517:
        z += -12.323378562927246 * Q.mass_over_sum_pt + 0.7431513449247279
    if Q.mass_over_sum_pt >= 0.09795414517:
        z += -23.9533634185791 * Q.mass_over_sum_pt + 1.8823565698001512
    if Q.lam1 < 0.00767124277:
        z += 50.34700393676758 * Q.lam1 - 1.0313702807979288
    if 0.00767124277 <= Q.lam1 < 0.020485236462:
        z += -50.71109390258789 * Q.lam1 - 0.2561290783978205
    if Q.lam1 >= 0.020485236462:
        z += -101.05809783935547 * Q.lam1 + 0.7752412024001083
    if Q.width >= 0.009614971338:
        z += 198.76133728027344 * Q.width - 1.91108456105238
    if Q.sum_pt < 1066.481811523438:
        z += 0.0030501417350023985 * Q.sum_pt - 3.2529206829486
    if Q.sum_pt_top20 < 993.56640625:
        z += -0.0020272517576813698 * Q.sum_pt_top20 + 2.0142092434434744
    if 0.008124776277 <= Q.girth2_top50 < 0.009262053166:
        z += 85.68029022216797 * Q.girth2_top50 - 0.6961331894035453
    if Q.girth2_top50 >= 0.009262053166:
        z += 9.061111450195312 * Q.girth2_top50 + 0.013517717917724048
    if 0.000856488074 <= Q.girth2_top30 < 0.006363915755:
        z += 67.99927520751953 * Q.girth2_top30 - 0.058240568255884355
    if Q.girth2_top30 >= 0.006363915755:
        z += -26.46404266357422 * Q.girth2_top30 + 0.5429160286135423
    if Q.z_top50_slots >= 0.958653609576:
        z += 13.830853462219238 * Q.z_top50_slots - 13.25899759507319
    if Q.n_dr_0p2_0p4 < 26.0:
        z += -0.02739403024315834 * Q.n_dr_0p2_0p4 + 0.7122447863221169
    if Q.n_particles > 29.0 and Q.n_dr_0_0p05 < 25.0:
        z += -0.0005135377286933362 * (Q.n_particles - 29.0) * (25.0 - Q.n_dr_0_0p05)
    if Q.mass_top40 < 83.325535102591 and Q.D2 < 6.916121244431:
        z += -0.013748330064117908 * (83.325535102591 - Q.mass_top40) * (6.916121244431 - Q.D2)
    if Q.mass < 120.60000000000001 and Q.max_dr < 0.402178311348:
        z += 0.07131876051425934 * (120.60000000000001 - Q.mass) * (0.402178311348 - Q.max_dr)
    if Q.mass < 101.049709320068 and Q.girth2_top2 < 0.004007841607:
        z += 2.4715282917022705 * (101.049709320068 - Q.mass) * (0.004007841607 - Q.girth2_top2)
    if Q.mass_top15 < 69.027163795459 and Q.girth2_top2 < 0.004007841607:
        z += -3.062052011489868 * (69.027163795459 - Q.mass_top15) * (0.004007841607 - Q.girth2_top2)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.7006720900535583
    if 0.078528833221 <= Q.mass_over_sum_pt < 0.079990613285:
        z += -26.110143661499023 * Q.mass_over_sum_pt + 2.050399116970207
    if 0.079990613285 <= Q.mass_over_sum_pt < 0.090467494167:
        z += 4.216342926025391 * Q.mass_over_sum_pt - 0.3754351439451975
    if 0.090467494167 <= Q.mass_over_sum_pt < 0.170880120467:
        z += -7.958438873291016 * Q.mass_over_sum_pt + 0.7259868574689574
    if Q.mass_over_sum_pt >= 0.170880120467:
        z += -365.249210357666 * Q.mass_over_sum_pt + 61.77987693046632
    if 6.856374501323 <= Q.log_sum_pt < 6.89371369877:
        z += 12.806431770324707 * Q.log_sum_pt - 87.80569224298708
    if 6.89371369877 <= Q.log_sum_pt < 6.910130970417:
        z += 6.1684770584106445 * Q.log_sum_pt - 42.045532913650234
    if 6.910130970417 <= Q.log_sum_pt < 6.935549248787:
        z += -16.10918140411377 * Q.log_sum_pt + 111.89600477661209
    if 6.935549248787 <= Q.log_sum_pt < 6.959293500649:
        z += -19.29218554496765 * Q.log_sum_pt + 133.97188675459716
    if 6.959293500649 <= Q.log_sum_pt < 6.989450376716:
        z += -9.504109144210815 * Q.log_sum_pt + 65.85379027495425
    if Q.log_sum_pt >= 6.989450376716:
        z += -0.46745848655700684 * Q.log_sum_pt + 2.692568931564935
    if 0.240474711359 <= Q.max_dr < 0.43572281599:
        z += -5.9441304206848145 * Q.max_dr + 1.429413047194432
    if Q.max_dr >= 0.43572281599:
        z += -0.13610219955444336 * Q.max_dr - 1.101277364665884
    if 906.60234375 <= Q.sum_pt_top40 < 935.8189453125:
        z += 0.0036248972173780203 * Q.sum_pt_top40 - 3.2863403131277664
    if Q.sum_pt_top40 >= 935.8189453125:
        z += 0.005713573656976223 * Q.sum_pt_top40 - 5.240963295931624
    if Q.sum_pt_top20 >= 1005.0126953125:
        z += 0.003277046838775277 * Q.sum_pt_top20 - 3.293473676102849
    if Q.z_top40_slots >= 0.930046498893:
        z += -18.781147003173828 * Q.z_top40_slots + 17.46734001549658
    if Q.n_dr_0p2_0p4 < 13.0:
        z += -0.052453864365816116 * Q.n_dr_0p2_0p4 + 0.6819002367556095
    if 934.241552734375 <= Q.sum_pt_top50 < 1061.183898925781:
        z += 0.004104783292859793 * Q.sum_pt_top50 - 3.8348591171594535
    if Q.sum_pt_top50 >= 1061.183898925781:
        z += -0.002627735957503319 * Q.sum_pt_top50 + 3.3095819105337494
    if 68.286969674465 <= Q.mass_top30 < 101.927236862114:
        z += 0.030016668140888214 * Q.mass_top30 - 2.049747307065313
    if Q.mass_top30 >= 101.927236862114:
        z += 0.010681185871362686 * Q.mass_top30 - 0.07893502593617918
    if 64.485443115234 <= Q.mass < 89.741833496094:
        z += 0.014125495217740536 * Q.mass - 0.9108888183381173
    if 89.741833496094 <= Q.mass < 162.836349487305:
        z += -0.02031456772238016 * Q.mass + 2.179825575629192
    if 162.836349487305 <= Q.mass < 172.8:
        z += -0.17386735323816538 * Q.mass + 27.18380062262678
    if Q.mass >= 172.8:
        z += -0.701730259694159 * Q.mass + 118.39851085822248
    if Q.mass_over_sum_pt_sq < 0.025803959699:
        z += -56.13850784301758 * Q.mass_over_sum_pt_sq + 1.448595793943221
    if Q.mass_over_sum_pt_sq >= 0.029200015571:
        z += -1720.6483154296875 * Q.mass_over_sum_pt_sq + 50.24295760276179
    if Q.mass_top50 >= 91.19:
        z += 0.0054476093500852585 * Q.mass_top50 - 0.4967674966342747
    if Q.z_top50_slots >= 0.958653609576:
        z += 24.968015670776367 * Q.z_top50_slots - 23.9356783467399
    if Q.girth2_top10 < 0.004752875822:
        z += -32.76961135864258 * Q.girth2_top10 - 0.0757523696001301
    if 0.004752875822 <= Q.girth2_top10 < 0.00895655368:
        z += 55.07136154174805 * Q.girth2_top10 - 0.49324960587935396
    if Q.girth < 0.070317784324:
        z += 23.668910026550293 * Q.girth - 1.853380744765376
    if 0.070317784324 <= Q.girth < 0.085894044489:
        z += 12.136124610900879 * Q.girth - 1.042420827252768
    if Q.z_dr_0p2_0p4 < 0.051804735139:
        z += 8.457462310791016 * Q.z_dr_0p2_0p4 - 0.43813659495860346
    if Q.mass_top40 >= 136.785:
        z += -0.03733205422759056 * Q.mass_top40 + 5.106465037520975
    if Q.sum_pt_top15 >= 951.1375:
        z += -0.005738317500799894 * Q.sum_pt_top15 + 5.45792896191706
    if Q.sum_pt_top10 >= 845.53671875:
        z += 0.003089280566200614 * Q.sum_pt_top10 - 2.6121001532434094
    if Q.n_real_top50 >= 22.0:
        z += -0.03028707206249237 * Q.n_real_top50 + 0.6663155853748322
    if Q.z_top20_slots >= 0.923451750505:
        z += -6.682138442993164 * Q.z_top20_slots + 6.170632442298792
    if Q.n_particles < 64.0 and Q.e2 < 0.038759447634:
        z += 1.9700498580932617 * (64.0 - Q.n_particles) * (0.038759447634 - Q.e2)
    if Q.n_particles < 64.0 and Q.z_dr_0p2_0p4 > 0.004744913615:
        z += 0.29025644063949585 * (64.0 - Q.n_particles) * (Q.z_dr_0p2_0p4 - 0.004744913615)
    if Q.log_sum_pt > 6.92034855022 and Q.girth2_top2 < 0.0140332421:
        z += -644.2356567382812 * (Q.log_sum_pt - 6.92034855022) * (0.0140332421 - Q.girth2_top2)
    if Q.mass_top50 > 157.544814151857 and Q.z_dr_0p05_0p1 > 0.591232848167:
        z += 1.217015266418457 * (Q.mass_top50 - 157.544814151857) * (Q.z_dr_0p05_0p1 - 0.591232848167)
    if Q.sum_pt > 907.937170410156 and Q.girth2_top2 < 0.0140332421:
        z += 0.31078073382377625 * (Q.sum_pt - 907.937170410156) * (0.0140332421 - Q.girth2_top2)
    if Q.girth < 0.070317784324 and Q.n_pt_above_50 > 2.0:
        z += 1.2749873399734497 * (0.070317784324 - Q.girth) * (Q.n_pt_above_50 - 2.0)
    if Q.girth < 0.085894044489 and Q.n_pt_above_50 > 2.0:
        z += -1.3795156478881836 * (0.085894044489 - Q.girth) * (Q.n_pt_above_50 - 2.0)
    if Q.mass_over_sum_pt_sq > 0.029200015571 and Q.phi_6 > -0.11682434082:
        z += -7187.36328125 * (Q.mass_over_sum_pt_sq - 0.029200015571) * (Q.phi_6 - -0.11682434082)
    if Q.log_sum_pt > 6.89371369877 and Q.z_dr_0p1_0p2 < 0.187280465662:
        z += -65.66336059570312 * (Q.log_sum_pt - 6.89371369877) * (0.187280465662 - Q.z_dr_0p1_0p2)
    if Q.log_sum_pt > 6.935549248787 and Q.z_dr_0p1_0p2 < 0.187280465662:
        z += 70.18206024169922 * (Q.log_sum_pt - 6.935549248787) * (0.187280465662 - Q.z_dr_0p1_0p2)
    if Q.girth2_top30 < 0.018076787298 and Q.z_dr_0p1_0p2 < 0.286492615938:
        z += 73.42398071289062 * (0.018076787298 - Q.girth2_top30) * (0.286492615938 - Q.z_dr_0p1_0p2)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.7257485389709473
    if Q.mass_top50 < 71.795159472175:
        z += -0.018385492265224457 * Q.mass_top50 + 1.3199893491562298
    if Q.mass < 86.4:
        z += -0.012208584696054459 * Q.mass + 0.6694415890969081
    if 86.4 <= Q.mass < 92.85979309082:
        z += 0.029542334377765656 * Q.mass - 2.9378378188811487
    if 92.85979309082 <= Q.mass < 101.049709320068:
        z += 0.09503908455371857 * Q.mass - 9.019852488341265
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += 0.020053446292877197 * Q.mass - 1.4425755389034753
    if 120.60000000000001 <= Q.mass < 172.8:
        z += -0.01869482919573784 * Q.mass + 3.2304664850234985
    if Q.mass_over_sum_pt >= 0.050772907168:
        z += -40.801395416259766 * Q.mass_over_sum_pt + 2.071605461794618
    if Q.e2 < 0.030297144316:
        z += -20.662192344665527 * Q.e2 + 1.399690878842538
    if 0.030297144316 <= Q.e2 < 0.043586218357:
        z += -47.628947257995605 * Q.e2 + 2.2167065441859015
    if 0.043586218357 <= Q.e2 < 0.047553086095:
        z += -35.479087829589844 * Q.e2 + 1.6871401181325527
    if Q.e2 >= 0.055571487173:
        z += 111.95805358886719 * Q.e2 - 6.221675538927779
    if Q.girth2_top20 < 0.0018230789:
        z += -32.35918045043945 * Q.girth2_top20 + 0.2598833527537025
    if 0.0018230789 <= Q.girth2_top20 < 0.008031209355:
        z += 41.1786003112793 * Q.girth2_top20 + 0.12581817629418712
    if 0.008031209355 <= Q.girth2_top20 < 0.016559833876:
        z += 73.53778076171875 * Q.girth2_top20 - 0.13406517645951538
    if Q.girth2_top20 >= 0.016559833876:
        z += -135.77468872070312 * Q.girth2_top20 + 3.3321145463447106
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.04338032752275467 * Q.n_dr_0p2_0p4 - 0.910986877977848
    if Q.lam1 < 0.004673423215:
        z += -221.60968780517578 * Q.lam1 + 1.3274157147269032
    if 0.004673423215 <= Q.lam1 < 0.007259287357:
        z += -112.82102966308594 * Q.lam1 + 0.8190002742369618
    if 0.008241985248 <= Q.lam1 < 0.011744050682:
        z += 28.532230377197266 * Q.lam1 - 0.23516222186139732
    if Q.lam1 >= 0.011744050682:
        z += -20.15831756591797 * Q.lam1 + 0.3366620409168989
    if 0.228402115913 <= Q.LHA < 0.260146178237:
        z += 4.411272048950195 * Q.LHA - 1.0075438698480994
    if Q.LHA >= 0.260146178237:
        z += 16.176074981689453 * Q.LHA - 4.068112390511667
    if Q.width < 0.009614971338:
        z += 118.81306457519531 * Q.width - 1.142384210470446
    if Q.girth2_top3 >= 0.010023689877:
        z += -176.8516082763672 * Q.girth2_top3 + 1.7727056756109911
    if Q.log_sum_pt < 6.989450376716:
        z += 7.217101573944092 * Q.log_sum_pt - 50.44357331480117
    if Q.mass_over_sum_pt_sq < 0.019863807341:
        z += -121.22026062011719 * Q.mass_over_sum_pt_sq + 2.407895902783817
    if Q.mass_top10 >= 99.066784770599:
        z += 0.032588377594947815 * Q.mass_top10 - 3.228425789221706
    if Q.sum_pt < 1085.12490234375:
        z += -0.008688713423907757 * Q.sum_pt + 9.428339305610734
    if Q.girth2_top50 < 0.019516409491:
        z += 77.10071563720703 * Q.girth2_top50 - 1.5047291384248793
    if Q.mass_top40 < 94.642533639752:
        z += 0.0032851779833436012 * Q.mass_top40 - 1.1731835511188775
    if 94.642533639752 <= Q.mass_top40 < 160.8:
        z += 0.013033539988100529 * Q.mass_top40 - 2.0957932300865654
    if Q.sum_pt_top40 >= 1139.734106445312:
        z += 0.0018927418859675527 * Q.sum_pt_top40 - 2.157222482134843
    if Q.girth2_top15 < 0.026753638475:
        z += 29.14234733581543 * Q.girth2_top15 - 0.7796638249352854
    if Q.mass_over_sum_pt > 0.050772907168 and Q.n_dr_0p2_0p4 < 21.0:
        z += 0.9781017303466797 * (Q.mass_over_sum_pt - 0.050772907168) * (21.0 - Q.n_dr_0p2_0p4)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.lam2 < 0.002396991421:
        z += 8149.7197265625 * (Q.mass_over_sum_pt - 0.050772907168) * (0.002396991421 - Q.lam2)
    if Q.girth2_top20 > 0.0018230789 and Q.max_dr < 0.43572281599:
        z += 358.6802062988281 * (Q.girth2_top20 - 0.0018230789) * (0.43572281599 - Q.max_dr)
    if Q.LHA > 0.228402115913 and Q.D2 < 3.345338582993:
        z += -2.4429895877838135 * (Q.LHA - 0.228402115913) * (3.345338582993 - Q.D2)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.max_pair_mass > 13.047927274731:
        z += -0.2790285348892212 * (Q.mass_over_sum_pt - 0.050772907168) * (Q.max_pair_mass - 13.047927274731)
    if Q.girth2_top3 > 0.010023689877 and Q.D2 < 4.450168704987:
        z += 68.57569122314453 * (Q.girth2_top3 - 0.010023689877) * (4.450168704987 - Q.D2)
    if Q.e2 > 0.055571487173 and Q.max_dr < 0.368618160486:
        z += -1232.888427734375 * (Q.e2 - 0.055571487173) * (0.368618160486 - Q.max_dr)
    if Q.LHA > 0.404204003833 and Q.z_2 > 0.049737748174:
        z += 494.7698974609375 * (Q.LHA - 0.404204003833) * (Q.z_2 - 0.049737748174)
    if Q.lam1 > 0.008241985248 and Q.sum_pt < 1167.4466796875:
        z += 0.297532320022583 * (Q.lam1 - 0.008241985248) * (1167.4466796875 - Q.sum_pt)
    if Q.e2 > 0.055571487173 and Q.n_real_top40 > 34.0:
        z += -12.185230255126953 * (Q.e2 - 0.055571487173) * (Q.n_real_top40 - 34.0)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.30209586024284363
    if Q.D2 < 1.788105106354:
        z += 0.429291695356369 * Q.D2 - 0.7676186725820893
    if Q.mass < 82.85408782959:
        z += 0.21770529076457024 * Q.mass - 17.04093455260194
    if 82.85408782959 <= Q.mass < 86.4:
        z += 0.08658547326922417 * Q.mass - 6.177121677642724
    if 86.4 <= Q.mass < 87.363773345947:
        z += 0.12164178490638733 * Q.mass - 9.205987003093622
    if 87.363773345947 <= Q.mass < 91.19:
        z += 0.02123621106147766 * Q.mass - 0.4341772070371883
    if 91.19 <= Q.mass < 92.85979309082:
        z += -0.016923319548368454 * Q.mass + 3.045590389274679
    if 92.85979309082 <= Q.mass < 101.049709320068:
        z += -0.08384265378117561 * Q.mass + 9.25970591990858
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += -0.040277156978845596 * Q.mass + 4.857425131648779
    if Q.z_top50_slots < 0.990637830118:
        z += 20.3000545501709 * Q.z_top50_slots - 20.110001990858333
    if Q.mass_top50 < 71.795159472175:
        z += 0.12086653709411621 * Q.mass_top50 - 9.034598396940556
    if 71.795159472175 <= Q.mass_top50 < 86.4:
        z += 0.0009041950106620789 * Q.mass_top50 - 0.42188291640335507
    if 86.4 <= Q.mass_top50 < 92.165451466106:
        z += 0.05962420627474785 * Q.mass_top50 - 5.495291889620366
    if Q.mass_top30 < 76.415438713532:
        z += -0.03795198816806078 * Q.mass_top30 + 3.0107749028438997
    if 76.415438713532 <= Q.mass_top30 < 86.252206812802:
        z += -0.011249332688748837 * Q.mass_top30 + 0.9702797695759786
    if Q.C2 < 0.056027559564:
        z += 18.798460006713867 * Q.C2 - 1.053231837737633
    if Q.girth2_top20 < 0.006374177987:
        z += 304.95377349853516 * Q.girth2_top20 - 2.6172848415236802
    if 0.006374177987 <= Q.girth2_top20 < 0.008031209355:
        z += 406.4227294921875 * Q.girth2_top20 - 3.2640660271822908
    if Q.mass_top20 < 66.841467317407:
        z += -0.016007624566555023 * Q.mass_top20 + 1.0699731142947089
    if Q.girth2_top40 < 0.007709915821:
        z += -35.134918212890625 * Q.girth2_top40 + 0.4725413097913749
    if 0.007709915821 <= Q.girth2_top40 < 0.008031986041:
        z += 104.26638793945312 * Q.girth2_top40 - 0.6022310259806449
    if 0.008031986041 <= Q.girth2_top40 < 0.008840538245:
        z += -290.93377685546875 * Q.girth2_top40 + 2.572011181053067
    if Q.girth < 0.085894044489:
        z += 52.0540771484375 * Q.girth - 4.471135218421729
    if Q.LHA < 0.320332145368:
        z += -10.253249168395996 * Q.LHA + 3.1358447922403605
    if 0.320332145368 <= Q.LHA < 0.333234539952:
        z += 11.517281532287598 * Q.LHA - 3.837956012909523
    if Q.e2_sq < 0.009606007381:
        z += -395.8717346191406 * Q.e2_sq + 3.802746804680738
    if Q.mass_over_sum_pt < 0.090467494167:
        z += 43.777137756347656 * Q.mass_over_sum_pt - 3.9604079546203375
    if Q.girth2_top30 < 0.008376290695:
        z += 113.22708892822266 * Q.girth2_top30 - 0.6167679664199801
    if 0.008376290695 <= Q.girth2_top30 < 0.012157872869:
        z += -87.70272064208984 * Q.girth2_top30 + 1.0662785278319504
    if Q.z_dr_0_0p05 >= 0.181086004525:
        z += 0.6439797878265381 * Q.z_dr_0_0p05 - 0.11661572677236502
    if Q.sum_pt < 1260.540869140625:
        z += 0.004878102336078882 * Q.sum_pt - 6.149047358477787
    if Q.sum_pt_top30 < 1038.26171875:
        z += -0.0031938229221850634 * Q.sum_pt_top30 + 3.3160240765710114
    if Q.lam2 < 0.000615484055:
        z += -1129.8072509765625 * Q.lam2 + 0.6953783481994574
    if Q.girth2_top50 < 0.007820314762:
        z += -219.60572814941406 * Q.girth2_top50 + 1.7173859176666217
    if Q.D2 < 1.788105106354 and Q.girth2_top50 < 0.007820314762:
        z += 298.9930114746094 * (1.788105106354 - Q.D2) * (0.007820314762 - Q.girth2_top50)
    if Q.D2 < 1.788105106354 and Q.sum_pt_top50 < 1245.696667480468:
        z += 0.0033083546441048384 * (1.788105106354 - Q.D2) * (1245.696667480468 - Q.sum_pt_top50)
    if Q.D2 < 1.788105106354 and Q.n_dr_0p2_0p4 < 9.0:
        z += 0.1153363585472107 * (1.788105106354 - Q.D2) * (9.0 - Q.n_dr_0p2_0p4)
    if Q.mass < 91.19 and Q.z_dr_0p05_0p1 > 0.402664637566:
        z += -0.08994314819574356 * (91.19 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.402664637566)
    if Q.mass_top50 < 97.930041729355 and Q.D2 < 1.601009327173:
        z += -0.4019072353839874 * (97.930041729355 - Q.mass_top50) * (1.601009327173 - Q.D2)
    if Q.mass < 101.049709320068 and Q.D2 < 1.601009327173:
        z += 0.2660719156265259 * (101.049709320068 - Q.mass) * (1.601009327173 - Q.D2)
    if Q.n_dr_0p2_0p4 < 13.0 and Q.z_1st < 0.499317836761:
        z += 0.20884670317173004 * (13.0 - Q.n_dr_0p2_0p4) * (0.499317836761 - Q.z_1st)
    if Q.mass < 91.19 and Q.max_dr < 0.390781164169:
        z += -1.177849292755127 * (91.19 - Q.mass) * (0.390781164169 - Q.max_dr)
    if Q.mass_top50 < 71.795159472175 and Q.max_dr < 0.39398368001:
        z += 1.1207572221755981 * (71.795159472175 - Q.mass_top50) * (0.39398368001 - Q.max_dr)
    if Q.mass < 101.049709320068 and Q.max_dr < 0.390781164169:
        z += 0.8799979090690613 * (101.049709320068 - Q.mass) * (0.390781164169 - Q.max_dr)
    if Q.mass < 91.19 and Q.pt_dispersion < 0.33780374676:
        z += -0.3539850413799286 * (91.19 - Q.mass) * (0.33780374676 - Q.pt_dispersion)
    if Q.girth2_top20 < 0.008031209355 and Q.z_dr_0p2_0p4 > 0.000658670394:
        z += 2871.5615234375 * (0.008031209355 - Q.girth2_top20) * (Q.z_dr_0p2_0p4 - 0.000658670394)
    if Q.girth2_top40 < 0.007709915821 and Q.n_pt_above_10 > 10.0:
        z += -13.237059593200684 * (0.007709915821 - Q.girth2_top40) * (Q.n_pt_above_10 - 10.0)
    if Q.mass_top50 < 79.21003612387 and Q.pt_2 < 137.5:
        z += 0.0013834686251357198 * (79.21003612387 - Q.mass_top50) * (137.5 - Q.pt_2)
    if Q.max_dr < 0.383415880799 and Q.eccentricity > 0.841527497033:
        z += -19.035470962524414 * (0.383415880799 - Q.max_dr) * (Q.eccentricity - 0.841527497033)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.3625415861606598
    if 0.076966318366 <= Q.mass_over_sum_pt < 0.088731426731:
        z += 6.954647541046143 * Q.mass_over_sum_pt - 0.5352736167674764
    if 0.088731426731 <= Q.mass_over_sum_pt < 0.09795414517:
        z += 35.47456216812134 * Q.mass_over_sum_pt - 3.0658863318741743
    if Q.mass_over_sum_pt >= 0.09795414517:
        z += -40.31357717514038 * Q.mass_over_sum_pt + 4.357876071519873
    if Q.girth2_top40 >= 0.005196965925:
        z += 21.608631134033203 * Q.girth2_top40 - 0.11229931968946466
    if 64.485443115234 <= Q.mass < 80.784643554688:
        z += 0.03978661447763443 * Q.mass - 2.565657464645241
    if 80.784643554688 <= Q.mass < 87.363773345947:
        z += 0.0850188173353672 * Q.mass - 6.219724849700523
    if 87.363773345947 <= Q.mass < 101.049709320068:
        z += 0.1258775033056736 * Q.mass - 9.789293830023595
    if 101.049709320068 <= Q.mass < 125.1:
        z += 0.02553078904747963 * Q.mass + 0.35071247699082786
    if Q.mass >= 125.1:
        z += -0.013907458633184433 * Q.mass + 5.284437261841902
    if 0.005718442372 <= Q.girth2_top20 < 0.008031209355:
        z += -94.56452178955078 * Q.girth2_top20 + 0.5407617682892845
    if Q.girth2_top20 >= 0.008031209355:
        z += 121.64212799072266 * Q.girth2_top20 - 1.1956391000392566
    if Q.sum_pt < 1012.672900390625:
        z += -0.009179937653243542 * Q.sum_pt + 9.296274088715245
    if Q.sum_pt >= 1260.540869140625:
        z += -0.00869144406169653 * Q.sum_pt + 10.955920451618068
    if Q.mass_over_sum_pt_sq < 0.006166777647:
        z += 375.0201873779297 * Q.mass_over_sum_pt_sq - 3.517466483562969
    if 0.006166777647 <= Q.mass_over_sum_pt_sq < 0.013977372691:
        z += 154.25205993652344 * Q.mass_over_sum_pt_sq - 2.1560385300872578
    if Q.girth < 0.097499583662:
        z += -18.20560359954834 * Q.girth + 1.5973751913044796
    if 0.097499583662 <= Q.girth < 0.14021858573:
        z += 4.158888816833496 * Q.girth - 0.5831535081047059
    if 82.04491364955 <= Q.mass_top50 < 97.930041729355:
        z += -0.028013568371534348 * Q.mass_top50 + 2.2983707980583006
    if Q.mass_top50 >= 97.930041729355:
        z += 0.014249064028263092 * Q.mass_top50 - 1.840410556446253
    if Q.e2 >= 0.025159193203:
        z += -17.822635650634766 * Q.e2 + 0.44840313372099566
    if Q.lam1 < 0.008241985248:
        z += -186.49597930908203 * Q.lam1 + 0.9680546653502933
    if 0.008241985248 <= Q.lam1 < 0.020485236462:
        z += 46.478050231933594 * Q.lam1 - 0.9521138492938737
    if Q.z_dr_0p2_0p4 < 0.091225683689:
        z += -6.520583629608154 * Q.z_dr_0p2_0p4 + 0.594844699662305
    if Q.z_dr_0p1_0p2 < 0.154838323593:
        z += -3.5742647647857666 * Q.z_dr_0p1_0p2 + 0.5534331642569565
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.038594309240579605 * Q.n_dr_0p2_0p4 - 0.8104804940521717
    if Q.mass_top40 >= 79.554505888974:
        z += -0.009290551766753197 * Q.mass_top40 + 0.739105255239985
    if Q.z_top15_slots < 0.813850690953:
        z += -3.4615554809570312 * Q.z_top15_slots + 2.817189319949024
    if Q.sum_pt_top30 < 1011.52392578125:
        z += 0.0039994120597839355 * Q.sum_pt_top30 - 4.045500987529522
    if Q.mass_top15 >= 30.359943489662:
        z += 0.005213119555264711 * Q.mass_top15 - 0.15827001510268854
    if Q.sum_pt_top40 >= 1225.8421875:
        z += 0.008587644435465336 * Q.sum_pt_top40 - 10.52709684024303
    if Q.log_sum_pt < 6.941996527183:
        z += -13.750962257385254 * Q.log_sum_pt + 95.45913223619294
    if Q.sum_pt_top50 < 1031.422265625:
        z += 0.004714128095656633 * Q.sum_pt_top50 - 4.862256680868631
    if Q.z_top50_slots < 0.985099030959:
        z += -16.768404006958008 * Q.z_top50_slots + 16.518538537983346
    if Q.mass_over_sum_pt > 0.076966318366 and Q.sum_pt < 1115.722741699219:
        z += 0.24113519489765167 * (Q.mass_over_sum_pt - 0.076966318366) * (1115.722741699219 - Q.sum_pt)
    if Q.girth2_top40 > 0.005196965925 and Q.sum_pt_top30 < 911.9328125:
        z += -0.2768447995185852 * (Q.girth2_top40 - 0.005196965925) * (911.9328125 - Q.sum_pt_top30)
    if Q.sum_pt_top40 < 1001.52314453125 and Q.log_sum_pt < 6.811175180312:
        z += -0.019653907045722008 * (1001.52314453125 - Q.sum_pt_top40) * (6.811175180312 - Q.log_sum_pt)
    if Q.mass > 87.363773345947 and Q.log_sum_pt < 6.903422848462:
        z += -0.21686823666095734 * (Q.mass - 87.363773345947) * (6.903422848462 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.5893873572349548
    if Q.mass_top40 < 80.890431271924:
        z += -0.007857637014240026 * Q.mass_top40 + 0.5764384893757515
    if 80.890431271924 <= Q.mass_top40 < 91.288535717504:
        z += 0.02059614146128297 * Q.mass_top40 - 1.7251999228250918
    if 91.288535717504 <= Q.mass_top40 < 120.60000000000001:
        z += -0.0052877492271363735 * Q.mass_top40 + 0.6377025567926466
    if Q.sum_pt_top40 < 956.21328125:
        z += -0.0043982891365885735 * Q.sum_pt_top40 + 4.205702487183589
    if Q.girth2_top40 < 0.00625977218:
        z += -112.03494262695312 * Q.girth2_top40 + 0.7013132170440973
    if Q.mass < 62.55:
        z += -0.006552121601998806 * Q.mass + 0.8962319533294066
    if 62.55 <= Q.mass < 82.85408782959:
        z += -0.0631239814683795 * Q.mass + 4.4348017879715185
    if 82.85408782959 <= Q.mass < 92.85979309082:
        z += 0.0007468210533261299 * Q.mass - 0.8571552939082774
    if 92.85979309082 <= Q.mass < 136.785:
        z += 0.03591101709753275 * Q.mass - 4.122495262778335
    if 136.785 <= Q.mass < 143.787612915039:
        z += 0.042463138699531555 * Q.mass - 5.018727216107742
    if 143.787612915039 <= Q.mass < 172.8:
        z += -0.059691548347473145 * Q.mass + 9.66985138246392
    if Q.mass >= 172.8:
        z += 0.007319368422031403 * Q.mass - 1.9096350353064664
    if Q.girth2_top15 < 0.009962397174:
        z += -2.2006454467773438 * Q.girth2_top15 - 0.7498696793513162
    if 0.009962397174 <= Q.girth2_top15 < 0.02146577947:
        z += 67.09273529052734 * Q.girth2_top15 - 1.4401978597855465
    if Q.girth < 0.02783744745:
        z += 111.45088195800781 * Q.girth - 4.247558617558744
    if 0.02783744745 <= Q.girth < 0.043628720567:
        z += 72.51160430908203 * Q.girth - 3.163588522265813
    if Q.girth >= 0.097499583662:
        z += 12.992172241210938 * Q.girth - 1.2667313843830599
    if Q.sum_pt < 949.91689453125:
        z += -0.03357695788145065 * Q.sum_pt + 31.895319558554185
    if Q.log_sum_pt < 6.856374501323:
        z += 22.27560043334961 * Q.log_sum_pt - 152.72985881287784
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.09204871207475662 * Q.n_dr_0p2_0p4 + 0.4602435603737831
    if Q.LHA < 0.209102506978:
        z += -22.74510955810547 * Q.LHA + 4.756059430089123
    if Q.width >= 0.025866900997:
        z += -88.62883758544922 * Q.width + 2.2925533673020078
    if Q.mass_top50 < 136.785:
        z += -0.023626726120710373 * Q.mass_top50 + 2.4510144212655725
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.03251165151596069 * Q.mass_top50 - 5.2278735637664795
    if Q.girth2 < 0.00363885588:
        z += -281.10565185546875 * Q.girth2 + 1.0229029541555053
    if Q.e2 >= 0.065240035206:
        z += -77.34413146972656 * Q.e2 + 5.045933860062453
    if Q.mass_top15 >= 40.2:
        z += -0.006501940079033375 * Q.mass_top15 + 0.26137799117714167
    if Q.mass_top30 < 76.415438713532:
        z += -0.01821896620094776 * Q.mass_top30 + 1.3922102951524347
    if Q.sum_pt_top40 < 956.21328125 and Q.z_top15_slots > 0.794624168612:
        z += -0.06902982294559479 * (956.21328125 - Q.sum_pt_top40) * (Q.z_top15_slots - 0.794624168612)
    if Q.log_sum_pt < 6.856374501323 and Q.z_top20_slots > 0.896541111574:
        z += 127.42782592773438 * (6.856374501323 - Q.log_sum_pt) * (Q.z_top20_slots - 0.896541111574)
    if Q.sum_pt_top40 < 956.21328125 and Q.z_top30_slots > 0.97348863653:
        z += -0.36319658160209656 * (956.21328125 - Q.sum_pt_top40) * (Q.z_top30_slots - 0.97348863653)
    return max(0.0, z)


def neuron_10(Q):
    z = 1.6076023578643799
    if Q.girth < 0.120745175332:
        z += 3.4866507053375244 * Q.girth - 0.4209962507374208
    if Q.mass < 62.55:
        z += -0.011684797704219818 * Q.mass + 2.2887277233466987
    if 62.55 <= Q.mass < 74.251806640625:
        z += -0.07230488210916519 * Q.mass + 6.080514002876032
    if 74.251806640625 <= Q.mass < 80.4:
        z += -0.051858752965927124 * Q.mass + 4.562351975183071
    if 80.4 <= Q.mass < 86.4:
        z += -0.005580544471740723 * Q.mass + 0.8415840122504836
    if 86.4 <= Q.mass < 87.363773345947:
        z += -0.2106391340494156 * Q.mass + 18.558646151761597
    if 87.363773345947 <= Q.mass < 101.049709320068:
        z += 0.02860577404499054 * Q.mass - 2.342691773170023
    if 101.049709320068 <= Q.mass < 143.787612915039:
        z += 0.020446129143238068 * Q.mass - 1.5181620276929608
    if 143.787612915039 <= Q.mass < 160.8:
        z += -0.06255333125591278 * Q.mass + 10.41613225633725
    if 160.8 <= Q.mass < 162.836349487305:
        z += -0.24968937039375305 * Q.mass + 40.50760734970197
    if Q.mass >= 162.836349487305:
        z += -0.1585189774632454 * Q.mass + 25.661753383574904
    if 136.785 <= Q.mass_top50 < 157.544814151857:
        z += 0.05757247284054756 * Q.mass_top50 - 7.875050697494298
    if 157.544814151857 <= Q.mass_top50 < 160.8:
        z += 0.0837296973913908 * Q.mass_top50 - 11.995985778085288
    if Q.mass_top50 >= 160.8:
        z += 0.03977753035724163 * Q.mass_top50 - 4.9284773189941
    if Q.z_dr_0_0p05 >= 0.767473447323:
        z += -2.194901943206787 * Q.z_dr_0_0p05 + 1.6845289608888645
    if 0.001868040786 <= Q.lam1 < 0.004673423215:
        z += 18.67807960510254 * Q.lam1 - 0.034891414506486314
    if Q.lam1 >= 0.004673423215:
        z += -71.76297569274902 * Q.lam1 + 0.3877779129115919
    if Q.e2 < 0.065240035206:
        z += 31.08782958984375 * Q.e2 - 2.0281710969195346
    if Q.girth2_top15 < 0.002197764741:
        z += 247.03651428222656 * Q.girth2_top15 - 0.5429281408290205
    if 0.000237176831 <= Q.girth2_top10 < 0.007678543663:
        z += 80.14093780517578 * Q.girth2_top10 - 0.019007573661999688
    if Q.girth2_top10 >= 0.007678543663:
        z += 24.839244842529297 * Q.girth2_top10 + 0.4056288903895012
    if Q.n_dr_0p2_0p4 < 13.0:
        z += 0.026483945548534393 * Q.n_dr_0p2_0p4 - 0.3442912921309471
    if Q.sum_pt_top15 >= 935.104296875:
        z += -0.0019041146151721478 * Q.sum_pt_top15 + 1.7805457583899624
    if Q.z_top5 >= 0.419766938686:
        z += -3.434433937072754 * Q.z_top5 + 1.4416618198843363
    if Q.D2 < 3.345338582993:
        z += -0.23826932907104492 * Q.D2 + 0.7970915796852223
    if Q.log_sum_pt < 6.903422848462:
        z += 6.396007061004639 * Q.log_sum_pt - 44.15434128386371
    if Q.sum_pt_top30 < 1052.07998046875:
        z += -0.0016722048167139292 * Q.sum_pt_top30 + 1.75929321090814
    if Q.LHA < 0.187029113551:
        z += 10.784836769104004 * Q.LHA - 2.0170784607177525
    if Q.mass_top5 >= 14.544037663713:
        z += 0.012017419561743736 * Q.mass_top5 - 0.1747818027266423
    if Q.mass_top20 >= 80.4:
        z += -0.019656725227832794 * Q.mass_top20 + 1.5804007083177567
    if Q.girth2 < 0.005925373826:
        z += -176.7957763671875 * Q.girth2 + 1.047581065833482
    if Q.mass_top40 >= 136.785:
        z += 0.044202771037817 * Q.mass_top40 - 6.046276036407798
    if Q.girth2_top5 < 0.024419631481 and Q.sum_pt_top3 < 656.384375:
        z += 0.06537540256977081 * (0.024419631481 - Q.girth2_top5) * (656.384375 - Q.sum_pt_top3)
    if Q.dr_0 < 0.064132973195 and Q.n_dr_0p2_0p4 > 2.0:
        z += 0.7834494113922119 * (0.064132973195 - Q.dr_0) * (Q.n_dr_0p2_0p4 - 2.0)
    if Q.e2 < 0.065240035206 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += -48.34282302856445 * (0.065240035206 - Q.e2) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.mass < 120.60000000000001 and Q.mass_top3 > 16.899120053094:
        z += 0.0004482660151552409 * (120.60000000000001 - Q.mass) * (Q.mass_top3 - 16.899120053094)
    if Q.z_top5 > 0.419766938686 and Q.dr_2 < 0.175028083821:
        z += 6.609047889709473 * (Q.z_top5 - 0.419766938686) * (0.175028083821 - Q.dr_2)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.22765173017978668
    if Q.mass < 62.55:
        z += -0.007461171597242355 * Q.mass + 2.6486529527680527
    if 62.55 <= Q.mass < 80.4:
        z += -0.05422426387667656 * Q.mass + 5.573684374846662
    if 80.4 <= Q.mass < 92.85979309082:
        z += -0.08205494843423367 * Q.mass + 7.811271413274254
    if 92.85979309082 <= Q.mass < 101.049709320068:
        z += -0.023402666673064232 * Q.mass + 2.3648326646275835
    if Q.lam1 < 0.006716736591:
        z += 337.0201110839844 * Q.lam1 - 2.2636753120206823
    if Q.e2 < 0.027935993578:
        z += -29.45708465576172 * Q.e2 + 0.38317930574732006
    if 0.027935993578 <= Q.e2 < 0.038759447634:
        z += 40.627845764160156 * Q.e2 - 1.5747128603781944
    if Q.girth2_top30 < 0.005809484705:
        z += 98.09500122070312 * Q.girth2_top30 - 0.6578755540540439
    if 0.005809484705 <= Q.girth2_top30 < 0.006363915755:
        z += 158.71070861816406 * Q.girth2_top30 - 1.0100215790623486
    if Q.n_dr_0p2_0p4 < 5.0:
        z += -0.09462954103946686 * Q.n_dr_0p2_0p4 + 0.4731477051973343
    if Q.mass_top50 < 86.4:
        z += 0.03205421194434166 * Q.mass_top50 - 2.7694839119911197
    if Q.mass_top30 < 60.438206617337:
        z += -0.03140576556324959 * Q.mass_top30 + 1.8981081480873259
    if Q.mass_over_sum_pt_sq < 0.008184367501:
        z += -160.7485809326172 * Q.mass_over_sum_pt_sq + 1.3156254616167804
    if Q.z_top5_slots >= 0.534625950898:
        z += -1.4082187414169312 * Q.z_top5_slots + 0.7528702837024116
    if Q.width < 0.006170281901:
        z += 170.41220092773438 * Q.width - 1.0514913190939748
    if Q.LHA < 0.309838384344:
        z += -0.004203319549560547 * Q.LHA + 0.23360490094545505
    if 0.309838384344 <= Q.LHA < 0.371981271173:
        z += -3.7382001876831055 * Q.LHA + 1.3905404577135088
    if Q.girth < 0.061710142531:
        z += -16.645804405212402 * Q.girth + 0.9660894818300672
    if 0.061710142531 <= Q.girth < 0.076787002012:
        z += 10.786097526550293 * Q.girth - 0.7267370962754232
    if 0.076787002012 <= Q.girth < 0.085894044489:
        z += -11.144671440124512 * Q.girth + 0.9572609044933424
    if Q.girth2_top20 < 0.005718442372:
        z += 43.101287841796875 * Q.girth2_top20 - 0.6204251472786232
    if 0.005718442372 <= Q.girth2_top20 < 0.007538018543:
        z += 205.51649475097656 * Q.girth2_top20 - 1.5491871483252235
    if Q.n_dr_0p2_0p4 < 10.0 and Q.girth2 < 0.005532208341:
        z += -18.90533447265625 * (10.0 - Q.n_dr_0p2_0p4) * (0.005532208341 - Q.girth2)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += 0.006047620438039303 * (10.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.mass < 101.049709320068 and Q.z_dr_0p1_0p2 < 0.334047731757:
        z += 0.033208031207323074 * (101.049709320068 - Q.mass) * (0.334047731757 - Q.z_dr_0p1_0p2)
    if Q.mass < 101.049709320068 and Q.sum_pt_top10 < 867.9265625:
        z += -3.9234135329024866e-05 * (101.049709320068 - Q.mass) * (867.9265625 - Q.sum_pt_top10)
    if Q.mass_top50 < 86.4 and Q.girth2_top15 < 0.00416995399:
        z += 2.3908326625823975 * (86.4 - Q.mass_top50) * (0.00416995399 - Q.girth2_top15)
    if Q.mass < 80.4 and Q.z_dr_0p2_0p4 < 0.037001823448:
        z += -0.8160367608070374 * (80.4 - Q.mass) * (0.037001823448 - Q.z_dr_0p2_0p4)
    if Q.mass < 101.049709320068 and Q.z_dr_0p2_0p4 < 0.02647292763:
        z += 0.5133441686630249 * (101.049709320068 - Q.mass) * (0.02647292763 - Q.z_dr_0p2_0p4)
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_dr_0p1_0p2 < 0.334047731757:
        z += -0.1751338094472885 * (10.0 - Q.n_dr_0p2_0p4) * (0.334047731757 - Q.z_dr_0p1_0p2)
    return max(0.0, z)


def neuron_12(Q):
    z = 0.0004015272425021976
    if Q.mass < 62.55:
        z += -0.047782283276319504 * Q.mass + 5.251746903826746
    if 62.55 <= Q.mass < 74.251806640625:
        z += -0.13779673352837563 * Q.mass + 10.882150767092856
    if 74.251806640625 <= Q.mass < 78.261818313599:
        z += -0.10857502184808254 * Q.mass + 8.71238588169964
    if 78.261818313599 <= Q.mass < 82.85408782959:
        z += -0.058672478422522545 * Q.mass + 4.806922094741981
    if 82.85408782959 <= Q.mass < 86.4:
        z += 0.015322597697377205 * Q.mass - 1.3238724410533906
    if Q.mass >= 143.787612915039:
        z += 0.02241591364145279 * Q.mass - 3.223130713814156
    if Q.log_sum_pt < 6.856374501323:
        z += 15.650307655334473 * Q.log_sum_pt - 107.30437034589542
    if Q.planar_flow >= 0.258818254187:
        z += -0.5672653317451477 * Q.planar_flow + 0.14681862282308852
    if Q.mass_top50 < 43.666010696263:
        z += -0.012491712346673012 * Q.mass_top50 - 0.611611688731045
    if 43.666010696263 <= Q.mass_top50 < 77.376408295162:
        z += 0.03705703653395176 * Q.mass_top50 - 2.7752078873388557
    if 77.376408295162 <= Q.mass_top50 < 80.4:
        z += -0.030471211299300194 * Q.mass_top50 + 2.449885388463736
    if Q.girth < 0.050483809784:
        z += 54.92115783691406 * Q.girth - 2.7726292853558103
    if Q.girth2_top5 < 0.001000990214:
        z += -469.1752014160156 * Q.girth2_top5 + 0.4696397852689106
    if Q.LHA < 0.228402115913:
        z += -6.306025505065918 * Q.LHA + 1.4403095683584002
    if Q.C2 >= 0.061168736406:
        z += -7.7013840675354 * Q.C2 + 0.471083931988441
    if Q.lam2 < 0.000615484055:
        z += 624.5969848632812 * Q.lam2 - 0.384429484984426
    if Q.e2_sq < 0.006936724595:
        z += -87.98530578613281 * Q.e2_sq + 0.6103298346452632
    if Q.mass_top20 > 125.1 and Q.C2 > 0.056027559564:
        z += -0.364020973443985 * (Q.mass_top20 - 125.1) * (Q.C2 - 0.056027559564)
    if Q.mass < 86.4 and Q.z_dr_0p05_0p1 > 0.299250295758:
        z += 0.10419808328151703 * (86.4 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.299250295758)
    if Q.mass_top50 < 77.376408295162 and Q.z_dr_0p05_0p1 < 0.850921532512:
        z += 0.02916557900607586 * (77.376408295162 - Q.mass_top50) * (0.850921532512 - Q.z_dr_0p05_0p1)
    if Q.mass_top40 < 89.67879517394 and Q.n_particles > 22.0:
        z += -0.0005285526858642697 * (89.67879517394 - Q.mass_top40) * (Q.n_particles - 22.0)
    if Q.girth < 0.050483809784 and Q.n_particles < 64.0:
        z += -0.8484540581703186 * (0.050483809784 - Q.girth) * (64.0 - Q.n_particles)
    if Q.mass < 86.4 and Q.max_dr < 0.390781164169:
        z += 0.14473100006580353 * (86.4 - Q.mass) * (0.390781164169 - Q.max_dr)
    if Q.mass < 82.85408782959 and Q.z_dr_0_0p05 > 0.328923654556:
        z += -0.12964020669460297 * (82.85408782959 - Q.mass) * (Q.z_dr_0_0p05 - 0.328923654556)
    if Q.mass_top50 < 43.666010696263 and Q.max_dr < 0.43572281599:
        z += -0.4221588671207428 * (43.666010696263 - Q.mass_top50) * (0.43572281599 - Q.max_dr)
    if Q.mass < 82.85408782959 and Q.z_dr_0p05_0p1 < 0.64696790278:
        z += 0.045796576887369156 * (82.85408782959 - Q.mass) * (0.64696790278 - Q.z_dr_0p05_0p1)
    if Q.mass_top50 < 80.4 and Q.z_dr_0p05_0p1 < 0.64696790278:
        z += 0.05906233936548233 * (80.4 - Q.mass_top50) * (0.64696790278 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_13(Q):
    z = 1.622334599494934
    if 62.55 <= Q.mass < 74.251806640625:
        z += 0.009088712744414806 * Q.mass - 0.5684989821631461
    if 74.251806640625 <= Q.mass < 120.60000000000001:
        z += 0.02106150984764099 * Q.mass - 1.457500797619332
    if 120.60000000000001 <= Q.mass < 143.787612915039:
        z += 0.0005082879215478897 * Q.mass + 1.0212177666674962
    if 143.787612915039 <= Q.mass < 160.8:
        z += -0.1051000040024519 * Q.mass + 16.206381966454018
    if 160.8 <= Q.mass < 162.836349487305:
        z += -0.4650799613445997 * Q.mass + 74.09115910707139
    if 162.836349487305 <= Q.mass < 172.8:
        z += -0.4041263032704592 * Q.mass + 64.16568793838096
    if Q.mass >= 172.8:
        z += -0.7896159160882235 * Q.mass + 130.77829303329062
    if Q.sum_pt < 986.05654296875:
        z += 0.034382306039333344 * Q.sum_pt - 35.12849919558686
    if 986.05654296875 <= Q.sum_pt < 1012.672900390625:
        z += 0.028422324918210506 * Q.sum_pt - 29.251620815133464
    if 1012.672900390625 <= Q.sum_pt < 1052.889428710937:
        z += 0.011664423160254955 * Q.sum_pt - 12.281347837443462
    if Q.sum_pt >= 1260.540869140625:
        z += 0.0023378015030175447 * Q.sum_pt - 2.9468943384919957
    if Q.log_sum_pt < 6.811175180312:
        z += 1.4171419143676758 * Q.log_sum_pt - 10.908685477723495
    if 6.811175180312 <= Q.log_sum_pt < 6.903422848462:
        z += 13.618595123291016 * Q.log_sum_pt - 94.01492073808036
    if Q.sum_pt_top40 < 956.21328125:
        z += 0.005211225710809231 * Q.sum_pt_top40 - 4.331284704217888
    if 956.21328125 <= Q.sum_pt_top40 < 1053.04736328125:
        z += -0.006730672903358936 * Q.sum_pt_top40 + 7.0877173539906835
    if 0.078528833221 <= Q.mass_over_sum_pt < 0.170880120467:
        z += 6.974335670471191 * Q.mass_over_sum_pt - 0.5476864426937034
    if Q.mass_over_sum_pt >= 0.170880120467:
        z += 381.9111337661743 * Q.mass_over_sum_pt - 64.6169316687987
    if 97.930041729355 <= Q.mass_top50 < 136.785:
        z += -0.028880754485726357 * Q.mass_top50 + 2.8282934919624387
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.017871057614684105 * Q.mass_top50 - 3.566653126192206
    if 160.8 <= Q.mass_top50 < 168.969765712694:
        z += 0.047049639746546745 * Q.mass_top50 - 8.258569132995719
    if Q.mass_top50 >= 168.969765712694:
        z += -0.0001652110368013382 * Q.mass_top50 - 0.2806868579735866
    z += 0.011418032459914684 * Q.n_particles
    if Q.e2_sq < 0.009606007381:
        z += -98.89124298095703 * Q.e2_sq + 0.9499500099913377
    if Q.mass_over_sum_pt_sq >= 0.029200015571:
        z += -604.5983276367188 * Q.mass_over_sum_pt_sq + 17.654280581192747
    if Q.sum_pt_top50 < 959.095727539062:
        z += -0.03930708300322294 * Q.sum_pt_top50 + 38.786433967104294
    if 959.095727539062 <= Q.sum_pt_top50 < 988.455444335938:
        z += -0.02624809369444847 * Q.sum_pt_top50 + 26.261613115080415
    if 988.455444335938 <= Q.sum_pt_top50 < 1013.915698242188:
        z += -0.012432790361344814 * Q.sum_pt_top50 + 12.605801320321673
    if Q.sum_pt_top30 < 966.06328125:
        z += 0.00360309612005949 * Q.sum_pt_top30 - 3.480818860403815
    if Q.sum_pt_top30 >= 1111.24501953125:
        z += -0.004473808221518993 * Q.sum_pt_top30 + 4.971497104500941
    if Q.mass_top15 >= 86.4:
        z += -0.015614655800163746 * Q.mass_top15 + 1.3491062611341478
    if Q.mass_top5 >= 37.764547629628:
        z += 0.013164828531444073 * Q.mass_top5 - 0.4971637941116053
    if Q.mass_top40 >= 136.785:
        z += 0.08637965470552444 * Q.mass_top40 - 11.815441068895161
    if Q.n_dr_0_0p05 >= 30.0:
        z += -0.07937175780534744 * Q.n_dr_0_0p05 + 2.3811527341604233
    if Q.girth2 < 0.013978743181:
        z += -113.05489349365234 * Q.girth2 + 1.580365321503074
    if Q.girth < 0.050483809784:
        z += 7.6126227378845215 * Q.girth - 0.9191874672222363
    if 0.050483809784 <= Q.girth < 0.120745175332:
        z += 29.391936779022217 * Q.girth - 2.018690214501032
    if Q.girth >= 0.120745175332:
        z += 21.779314041137695 * Q.girth - 1.0995027472787957
    if Q.LHA >= 0.228402115913:
        z += -9.561287879943848 * Q.LHA + 2.183818382632497
    if Q.girth2_top40 >= 0.00663016737:
        z += -53.920894622802734 * Q.girth2_top40 + 0.35750455608931514
    if Q.z_dr_0p2_0p4 < 0.129185591638:
        z += 6.582622051239014 * Q.z_dr_0p2_0p4 - 0.8503799242186572
    if Q.z_top30_slots >= 0.946075126916:
        z += -10.37974739074707 * Q.z_top30_slots + 9.820020830057056
    if Q.sum_pt < 1085.12490234375 and Q.sum_pt_top30 < 800.73203125:
        z += -4.551769598037936e-05 * (1085.12490234375 - Q.sum_pt) * (800.73203125 - Q.sum_pt_top30)
    if Q.mass > 172.8 and Q.pt_6 < 62.25:
        z += -0.011619515717029572 * (Q.mass - 172.8) * (62.25 - Q.pt_6)
    if Q.log_sum_pt < 6.811175180312 and Q.dr_5 < 0.051086217058:
        z += -420.7301940917969 * (6.811175180312 - Q.log_sum_pt) * (0.051086217058 - Q.dr_5)
    if Q.mass > 172.8 and Q.z_6 < 0.042471339685:
        z += 13.720694541931152 * (Q.mass - 172.8) * (0.042471339685 - Q.z_6)
    if Q.mass > 160.8 and Q.D2 < 5.378974604607:
        z += 0.031817518174648285 * (Q.mass - 160.8) * (5.378974604607 - Q.D2)
    if Q.mass > 74.251806640625 and Q.D2 > 0.603279101849:
        z += -0.002653916832059622 * (Q.mass - 74.251806640625) * (Q.D2 - 0.603279101849)
    if Q.sum_pt_top50 < 1013.915698242188 and Q.D2 < 4.450168704987:
        z += -0.0018148613162338734 * (1013.915698242188 - Q.sum_pt_top50) * (4.450168704987 - Q.D2)
    if Q.sum_pt > 1260.540869140625 and Q.dr_0 < 0.111955475493:
        z += -0.03965368494391441 * (Q.sum_pt - 1260.540869140625) * (0.111955475493 - Q.dr_0)
    if Q.sum_pt_top30 > 1111.24501953125 and Q.dr_0 < 0.093385871589:
        z += 0.08116258680820465 * (Q.sum_pt_top30 - 1111.24501953125) * (0.093385871589 - Q.dr_0)
    if Q.sum_pt_top20 > 956.50615234375 and Q.dr_0 < 0.071710390673:
        z += -0.06308678537607193 * (Q.sum_pt_top20 - 956.50615234375) * (0.071710390673 - Q.dr_0)
    if Q.sum_pt < 1012.672900390625 and Q.n_dr_0p1_0p2 < 26.0:
        z += -0.0002955137751996517 * (1012.672900390625 - Q.sum_pt) * (26.0 - Q.n_dr_0p1_0p2)
    if Q.mass > 62.55 and Q.n_dr_0p2_0p4 < 26.0:
        z += 0.0004870109842158854 * (Q.mass - 62.55) * (26.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.4243285357952118
    if Q.girth2 < 0.007877041167:
        z += -232.8127899169922 * Q.girth2 + 1.83387593038027
    if Q.mass_over_sum_pt < 0.083299446175:
        z += 38.68291091918945 * Q.mass_over_sum_pt - 0.1465137766255742
    if 0.083299446175 <= Q.mass_over_sum_pt < 0.090467494167:
        z += -17.653335571289062 * Q.mass_over_sum_pt + 4.546264355609574
    if 0.090467494167 <= Q.mass_over_sum_pt < 0.09795414517:
        z += -146.1042938232422 * Q.mass_over_sum_pt + 16.166900672013703
    if 0.09795414517 <= Q.mass_over_sum_pt < 0.118225939153:
        z += -67.94446563720703 * Q.mass_over_sum_pt + 8.51082151541656
    if 0.118225939153 <= Q.mass_over_sum_pt < 0.140939019879:
        z += -21.046165466308594 * Q.mass_over_sum_pt + 2.9662259330327903
    if Q.mass_top40 < 67.726432644245:
        z += -0.012781184166669846 * Q.mass_top40 + 0.21433984954468732
    if 67.726432644245 <= Q.mass_top40 < 80.890431271924:
        z += 0.020469006150960922 * Q.mass_top40 - 2.03757692541066
    if 80.890431271924 <= Q.mass_top40 < 91.288535717504:
        z += -0.004508202895522118 * Q.mass_top40 - 0.01715971367164526
    if 91.288535717504 <= Q.mass_top40 < 111.24867219155:
        z += 0.021478157490491867 * Q.mass_top40 - 2.389416501938214
    if Q.mass_top50 < 89.080094718389:
        z += -0.018054342828691006 * Q.mass_top50 + 1.647069808602204
    if 89.080094718389 <= Q.mass_top50 < 92.165451466106:
        z += 0.05417430493980646 * Q.mass_top50 - 4.787064975996707
    if 92.165451466106 <= Q.mass_top50 < 117.048742792994:
        z += -0.008276007138192654 * Q.mass_top50 + 0.9686962308712943
    if Q.mass < 80.4:
        z += 0.18635390140116215 * Q.mass - 15.468974853434645
    if 80.4 <= Q.mass < 89.741833496094:
        z += 0.12580464221537113 * Q.mass - 10.600814414897048
    if 89.741833496094 <= Q.mass < 91.034691238403:
        z += 0.04556524194777012 * Q.mass - 3.399983516255557
    if 91.034691238403 <= Q.mass < 136.785:
        z += -0.016350364312529564 * Q.mass + 2.236484582489356
    if Q.max_dr < 0.240474711359:
        z += 4.114835739135742 * Q.max_dr - 1.3644208455693352
    if 0.240474711359 <= Q.max_dr < 0.331585738063:
        z += -0.6637845039367676 * Q.max_dr - 0.2152835219221989
    if 0.331585738063 <= Q.max_dr < 0.402178311348:
        z += -4.77862024307251 * Q.max_dr + 1.1491373236471363
    if Q.max_dr >= 0.402178311348:
        z += -1.2881417274475098 * Q.max_dr - 0.2546574315633998
    if Q.lam2 < 0.002396991421:
        z += -376.458251953125 * Q.lam2 + 0.902367200296297
    if Q.sum_pt_top50 >= 976.277001953125:
        z += -0.009037083014845848 * Q.sum_pt_top50 + 8.822696312135212
    if Q.girth2_top30 < 0.00608841615:
        z += -29.082443237304688 * Q.girth2_top30 - 0.6810839994881861
    if 0.00608841615 <= Q.girth2_top30 < 0.012157872869:
        z += 141.38827514648438 * Q.girth2_top30 - 1.7189806743981495
    if Q.girth2_top50 < 0.009262053166:
        z += 298.89739990234375 * Q.girth2_top50 - 2.768403609074671
    if Q.log_sum_pt >= 6.935549248787:
        z += 12.321833610534668 * Q.log_sum_pt - 85.45868384122213
    if Q.lam1 < 0.007259287357:
        z += -55.46694374084473 * Q.lam1 + 0.029551359112870212
    if 0.007259287357 <= Q.lam1 < 0.008241985248:
        z += 274.26447105407715 * Q.lam1 - 2.364063731513629
    if 0.008241985248 <= Q.lam1 < 0.011744050682:
        z += 29.576833724975586 * Q.lam1 - 0.34735183427920013
    if Q.girth2_top10 < 0.006876086349:
        z += -89.97664642333984 * Q.girth2_top10 + 0.6186871902003268
    if Q.mass_top10 < 45.595:
        z += 0.015384136699140072 * Q.mass_top10 - 0.7014397127972916
    if Q.girth2_top20 < 0.005718442372:
        z += -112.02090454101562 * Q.girth2_top20 + 0.16151315038643244
    if 0.005718442372 <= Q.girth2_top20 < 0.010834353386:
        z += 93.64352416992188 * Q.girth2_top20 - 1.014567033167366
    if Q.max_dr > 0.240474711359 and Q.mass_top10 < 56.921923720802:
        z += 0.037094131112098694 * (Q.max_dr - 0.240474711359) * (56.921923720802 - Q.mass_top10)
    if Q.mass < 91.034691238403 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += 0.17581163346767426 * (91.034691238403 - Q.mass) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += -0.0030064345337450504 * (21.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.lam1 < 0.006189818106 and Q.z_top50_slots > 0.985099030959:
        z += 5904.29052734375 * (0.006189818106 - Q.lam1) * (Q.z_top50_slots - 0.985099030959)
    if Q.girth2_top20 < 0.016559833876 and Q.z_top50_slots > 0.970443639316:
        z += -888.4301147460938 * (0.016559833876 - Q.girth2_top20) * (Q.z_top50_slots - 0.970443639316)
    if Q.log_sum_pt > 6.935549248787 and Q.eccentricity > 0.948956476603:
        z += 422.2880859375 * (Q.log_sum_pt - 6.935549248787) * (Q.eccentricity - 0.948956476603)
    if Q.sum_pt_top50 > 976.277001953125 and Q.eccentricity > 0.948956476603:
        z += -0.2912554442882538 * (Q.sum_pt_top50 - 976.277001953125) * (Q.eccentricity - 0.948956476603)
    if Q.lam2 < 0.001163277284 and Q.dr_1 < 0.061084209235:
        z += -13194.52734375 * (0.001163277284 - Q.lam2) * (0.061084209235 - Q.dr_1)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.6340818405151367
    if Q.sum_pt < 986.05654296875:
        z += -9.393319487571716e-06 * Q.sum_pt - 0.3117578158800818
    if 986.05654296875 <= Q.sum_pt < 1002.378515625:
        z += 0.01966797560453415 * Q.sum_pt - 19.714756191821653
    if Q.log_sum_pt < 7.017257672702:
        z += -3.6481640338897705 * Q.log_sum_pt + 25.60010705808847
    if Q.girth2_top5 < 0.001501708498:
        z += 1079.572265625 * Q.girth2_top5 - 1.6212028454941758
    if Q.girth < 0.056600876898:
        z += 10.050851821899414 * Q.girth + 0.6565780203554399
    if 0.056600876898 <= Q.girth < 0.120745175332:
        z += -19.104816436767578 * Q.girth + 2.3068144103431765
    if Q.lam1 < 0.006189818106:
        z += -100.32278442382812 * Q.lam1 - 0.02659028909948824
    if 0.006189818106 <= Q.lam1 < 0.016493544356:
        z += 62.84814453125 * Q.lam1 - 1.0365886595184706
    if Q.dr_0 < 0.052014814497:
        z += 6.366873741149902 * Q.dr_0 - 0.3311717565717326
    if Q.n_dr_0p1_0p2 < 13.0:
        z += -0.03738356754183769 * Q.n_dr_0p1_0p2 + 0.48598637804389
    if Q.sum_pt_top50 < 934.241552734375:
        z += 0.0028911216650158167 * Q.sum_pt_top50 - 2.701005993468368
    if Q.e2_sq < 0.006936724595:
        z += 520.0942993164062 * Q.e2_sq - 3.6077509177874068
    if Q.z_dr_0p1_0p2 < 0.187280465662:
        z += -2.6875061988830566 * Q.z_dr_0p1_0p2 + 0.5033174123963304
    if Q.mass_top50 < 71.795159472175:
        z += 0.017188075929880142 * Q.mass_top50 - 1.2340206524055974
    if Q.mass_top30 < 62.55:
        z += 0.0068195355124771595 * Q.mass_top30 - 0.3298120069317519
    if 62.55 <= Q.mass_top30 < 80.4:
        z += -0.005420164670795202 * Q.mass_top30 + 0.4357812395319343
    if Q.lam2 < 0.001776308492:
        z += -396.7745666503906 * Q.lam2 + 0.7047940321507089
    if Q.e2 < 0.040828318335:
        z += -54.09757614135742 * Q.e2 + 2.208713059851242
    if Q.sum_pt_top30 < 1038.26171875:
        z += 0.0012982821790501475 * Q.sum_pt_top30 - 1.3479566866431014
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.mass_top5 < 40.2:
        z += 0.1682390719652176 * (0.120343671367 - Q.z_dr_0p1_0p2) * (40.2 - Q.mass_top5)
    if Q.girth2_top5 < 0.001501708498 and Q.n_dr_0p2_0p4 > 0.0:
        z += -20.047880172729492 * (0.001501708498 - Q.girth2_top5) * (Q.n_dr_0p2_0p4 - 0.0)
    if Q.girth2_top5 < 0.007164202106 and Q.sum_pt_top40 < 984.70087890625:
        z += -0.6655955910682678 * (0.007164202106 - Q.girth2_top5) * (984.70087890625 - Q.sum_pt_top40)
    if Q.girth2_top5 < 0.001501708498 and Q.z_dr_0p2_0p4 < 0.193744690716:
        z += 7380.3759765625 * (0.001501708498 - Q.girth2_top5) * (0.193744690716 - Q.z_dr_0p2_0p4)
    if Q.girth2_top50 < 0.013514311784 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += 190.09225463867188 * (0.013514311784 - Q.girth2_top50) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.e2_sq < 0.006936724595 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += -386.0799560546875 * (0.006936724595 - Q.e2_sq) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.e2_sq < 0.006936724595 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += 4577.1396484375 * (0.006936724595 - Q.e2_sq) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.120745175332 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += -136.36111450195312 * (0.120745175332 - Q.girth) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.sum_pt < 1002.378515625 and Q.z_dr_0p2_0p4 < 0.019523000158:
        z += -0.3199748992919922 * (1002.378515625 - Q.sum_pt) * (0.019523000158 - Q.z_dr_0p2_0p4)
    if Q.girth2_top30 < 0.012157872869 and Q.z_dr_0p2_0p4 > 0.02647292763:
        z += -3093.56298828125 * (0.012157872869 - Q.girth2_top30) * (Q.z_dr_0p2_0p4 - 0.02647292763)
    if Q.z_dr_0_0p05 < 0.845900350809 and Q.z_dr_0p2_0p4 < 0.00339853589:
        z += -322.59722900390625 * (0.845900350809 - Q.z_dr_0_0p05) * (0.00339853589 - Q.z_dr_0p2_0p4)
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.z_dr_0p2_0p4 < 0.051804735139:
        z += -132.43096923828125 * (0.120343671367 - Q.z_dr_0p1_0p2) * (0.051804735139 - Q.z_dr_0p2_0p4)
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
