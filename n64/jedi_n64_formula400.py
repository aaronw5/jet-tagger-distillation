"""JEDI-linear jet tagger, 64 particles, 3 features: tuned on the network's predictions (from 60 if-statements per neuron, pruned), as if-statements.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities():  physics quantities of the particles.
2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.
3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid
                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.
4. classify():    softmax of the logits; the class is the largest logit.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 81.1% (the network: 81.1%); same class as the network for 93.7% of jets.

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
  Q.max_pair_mass          largest pair mass among particles 0, 1, 2 [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.n_particles            number of real particles (pT > 0)
  Q.n_real_top30           number of real particles among the 30 hardest
  Q.n_real_top40           number of real particles among the 40 hardest
  Q.pt_4                   pT of particle 4 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.pt_9                   pT of particle 9 [GeV]
  Q.z_0                    pT of particle 0 / total pT
  Q.z_2                    pT of particle 2 / total pT
  Q.z_top20_slots          pT share of the 20 hardest particles
  Q.z_top30_slots          pT share of the 30 hardest particles
  Q.z_top40_slots          pT share of the 40 hardest particles
  Q.z_top5_slots           pT share of the 5 hardest particles
  Q.z_top50_slots          pT share of the 50 hardest particles
  Q.z_1st                  largest pT share
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_3                   ΔR of particle 3 from the jet axis
  Q.eta_0                  Δη of particle 0
  Q.eta_1                  Δη of particle 1
  Q.eta_5                  Δη of particle 5
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
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.mean_phi               pT-weighted mean Δφ
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
        max_pair_mass=max(pair_mass(0, 1), pair_mass(0, 2), pair_mass(1, 2)),
        max_dr=max(dr[i] for i in real),
        n_particles=len(real),
        n_real_top30=sum(1 for x in pt[:30] if x > 0),
        n_real_top40=sum(1 for x in pt[:40] if x > 0),
        pt_4=pt[4],
        pt_7=pt[7],
        pt_9=pt[9],
        z_0=z[0],
        z_2=z[2],
        z_top20_slots=sum(pt[:20]) / tot,
        z_top30_slots=sum(pt[:30]) / tot,
        z_top40_slots=sum(pt[:40]) / tot,
        z_top5_slots=sum(pt[:5]) / tot,
        z_top50_slots=sum(pt[:50]) / tot,
        z_1st=zs[0],
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_3=dr[3] if pt[3] > 0 else 0.0,
        eta_0=eta[0],
        eta_1=eta[1],
        eta_5=eta[5],
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
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        mean_phi=sum(z[i] * phi[i] for i in P),
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
    z = 1.385832667350769
    if 78.261818313599 <= Q.mass < 91.034691238403:
        z += -0.13073386251926422 * Q.mass + 10.231469795917686
    if 91.034691238403 <= Q.mass < 92.85979309082:
        z += -0.21607179194688797 * Q.mass + 18.00018185228604
    if Q.mass >= 92.85979309082:
        z += -0.34200289100408554 * Q.mass + 29.694117654436965
    if Q.girth2_top20 < 0.005312783396:
        z += 88.17398834228516 * Q.girth2_top20 - 1.061469209838797
    if 0.005312783396 <= Q.girth2_top20 < 0.006043208873:
        z += 318.6708450317383 * Q.girth2_top20 - 2.286049082888715
    if 0.006043208873 <= Q.girth2_top20 < 0.007538018543:
        z += 241.003662109375 * Q.girth2_top20 - 1.8166900739113752
    if Q.sum_pt < 1012.672900390625:
        z += 0.008862711489200592 * Q.sum_pt - 8.97502774909408
    if Q.girth < 0.056600876898:
        z += 62.393890380859375 * Q.girth - 3.531548908634328
    if Q.mass_over_sum_pt_sq < 0.008184367501:
        z += -482.1446533203125 * Q.mass_over_sum_pt_sq + 3.946049031415677
    if Q.lam1 < 0.005913554513:
        z += 215.8194580078125 * Q.lam1 - 1.2762601298953136
    if Q.girth2_top40 < 0.00625977218:
        z += 186.67782592773438 * Q.girth2_top40 - 1.1685606613653143
    if Q.width < 0.009614971338:
        z += 32.05830764770508 * Q.width - 0.3082397091774705
    if Q.LHA < 0.260146178237:
        z += -4.548079490661621 * Q.LHA + 1.183165497813702
    if Q.mass_top50 < 82.04491364955:
        z += 0.009700550697743893 * Q.mass_top50 - 0.7958808443494797
    if Q.log_sum_pt < 7.017257672702:
        z += -3.2871878147125244 * Q.log_sum_pt + 23.067043914403982
    if Q.girth2_top50 < 0.00742997247:
        z += -147.4435272216797 * Q.girth2_top50 + 1.0955013481367757
    if 0.076966318366 <= Q.mass_over_sum_pt < 0.083299446175:
        z += -43.13007736206055 * Q.mass_over_sum_pt + 3.3195632653985614
    if Q.mass_over_sum_pt >= 0.083299446175:
        z += 39.96780776977539 * Q.mass_over_sum_pt - 3.602444544397139
    if Q.girth2_top40 < 0.00625977218 and Q.girth2_top3 < 0.002915531053:
        z += 66290.015625 * (0.00625977218 - Q.girth2_top40) * (0.002915531053 - Q.girth2_top3)
    if Q.sum_pt < 1012.672900390625 and Q.z_dr_0p1_0p2 > 0.088715460151:
        z += -0.012252594344317913 * (1012.672900390625 - Q.sum_pt) * (Q.z_dr_0p1_0p2 - 0.088715460151)
    if Q.mass > 92.85979309082 and Q.n_dr_0p2_0p4 < 7.0:
        z += 0.043849289417266846 * (Q.mass - 92.85979309082) * (7.0 - Q.n_dr_0p2_0p4)
    return max(0.0, z)


def neuron_1(Q):
    z = 1.7525185346603394
    if Q.n_particles >= 38.0:
        z += 0.040213074535131454 * Q.n_particles - 1.5280968323349953
    if 6.811175180312 <= Q.log_sum_pt < 6.89371369877:
        z += -4.9427809715271 * Q.log_sum_pt + 33.66614707498382
    if 6.89371369877 <= Q.log_sum_pt < 6.910130970417:
        z += 19.331644535064697 * Q.log_sum_pt - 133.67479256957995
    if 6.910130970417 <= Q.log_sum_pt < 6.959293500649:
        z += 43.435932636260986 * Q.log_sum_pt - 300.2385802975104
    if 6.959293500649 <= Q.log_sum_pt < 6.989450376716:
        z += 29.33468198776245 * Q.log_sum_pt - 202.10383830839203
    if 6.989450376716 <= Q.log_sum_pt < 7.139296169016:
        z += 21.177353382110596 * Q.log_sum_pt - 145.08859481262243
    if Q.log_sum_pt >= 7.139296169016:
        z += 16.14638614654541 * Q.log_sum_pt - 109.17102970130692
    if Q.sum_pt_top50 >= 959.095727539062:
        z += -0.008625069633126259 * Q.sum_pt_top50 + 8.2722674348583
    if Q.mass_top20 < 40.2:
        z += 0.067268006503582 * Q.mass_top20 - 2.7041738614439965
    if Q.sum_pt_top2 < 689.25:
        z += -0.0035508384462445974 * Q.sum_pt_top2 + 2.447415399074089
    if Q.z_top30_slots >= 0.934183811419:
        z += 7.179083824157715 * Q.z_top30_slots - 6.706583889348144
    if Q.sum_pt_top30 < 1073.4734375:
        z += -0.003320657880976796 * Q.sum_pt_top30 + 3.5646380302536276
    if Q.girth2_top15 < 0.003270031267:
        z += -133.390625 * Q.girth2_top15 + 0.43619151447467186
    if Q.z_top20_slots < 0.957678701144:
        z += 11.661158561706543 * Q.z_top20_slots - 11.167643185209357
    if Q.max_dr < 0.43572281599:
        z += 7.0916008949279785 * Q.max_dr - 3.089972311815223
    if Q.lam2 < 0.000829637219:
        z += 789.2779541015625 * Q.lam2 - 0.65481436685883
    if Q.z_top50_slots >= 0.958653609576:
        z += -52.001338958740234 * Q.z_top50_slots + 49.8512712955814
    if Q.z_dr_0_0p05 >= 0.878906026483:
        z += -4.728032112121582 * Q.z_dr_0_0p05 + 4.155495916748806
    if Q.mass_top50 < 117.048742792994:
        z += 0.01663493923842907 * Q.mass_top50 - 1.9470987242959676
    if Q.LHA < 0.404204003833:
        z += -8.832510948181152 * Q.LHA + 3.570136289153629
    if Q.mass_top30 < 80.4:
        z += -0.05527365580201149 * Q.mass_top30 + 4.444001926481724
    if Q.n_pt_above_10 < 31.0:
        z += 0.05232043191790581 * Q.n_pt_above_10 - 1.62193338945508
    if Q.girth2_top10 < 0.00130722027:
        z += -410.9342346191406 * Q.girth2_top10 + 0.5371815611310764
    if Q.girth2_top5 < 0.000657050184:
        z += -885.7968139648438 * Q.girth2_top5 + 0.5820129596022143
    if Q.sum_pt_top5 < 902.40625:
        z += -0.0019594691693782806 * Q.sum_pt_top5 + 1.768237225129269
    if Q.sum_pt_top2 < 689.25 and Q.n_dr_0p2_0p4 < 7.0:
        z += -0.0003387926844879985 * (689.25 - Q.sum_pt_top2) * (7.0 - Q.n_dr_0p2_0p4)
    if Q.mass_top20 < 40.2 and Q.n_real_top40 > 26.0:
        z += 0.002735333051532507 * (40.2 - Q.mass_top20) * (Q.n_real_top40 - 26.0)
    if Q.n_particles > 38.0 and Q.dr_0 < 0.144288109196:
        z += 0.24722395837306976 * (Q.n_particles - 38.0) * (0.144288109196 - Q.dr_0)
    if Q.n_particles > 38.0 and Q.z_top50_slots < 0.985099030959:
        z += 1.1446267366409302 * (Q.n_particles - 38.0) * (0.985099030959 - Q.z_top50_slots)
    if Q.z_top30_slots > 0.934183811419 and Q.mass_top5 > 22.1834155076:
        z += 0.44266921281814575 * (Q.z_top30_slots - 0.934183811419) * (Q.mass_top5 - 22.1834155076)
    if Q.max_dr < 0.43572281599 and Q.z_dr_0p2_0p4 < 0.193744690716:
        z += 30.025300979614258 * (0.43572281599 - Q.max_dr) * (0.193744690716 - Q.z_dr_0p2_0p4)
    if Q.mass_top20 < 40.2 and Q.mean_phi2 < 0.005860335776:
        z += 9.62682056427002 * (40.2 - Q.mass_top20) * (0.005860335776 - Q.mean_phi2)
    if Q.n_pt_above_10 < 31.0 and Q.mean_phi2 < 0.017422899418:
        z += 1.3072254657745361 * (31.0 - Q.n_pt_above_10) * (0.017422899418 - Q.mean_phi2)
    if Q.z_top30_slots > 0.934183811419 and Q.mass_top10 < 91.19:
        z += -0.1810741424560547 * (Q.z_top30_slots - 0.934183811419) * (91.19 - Q.mass_top10)
    if Q.mass_top30 < 80.4 and Q.mass_top5 < 68.434224049685:
        z += -0.0006333038909360766 * (80.4 - Q.mass_top30) * (68.434224049685 - Q.mass_top5)
    return max(0.0, z)


def neuron_2(Q):
    z = -0.6637150645256042
    if 995.676940917969 <= Q.sum_pt < 1017.43466796875:
        z += 0.00799630582332611 * Q.sum_pt - 7.961737320813883
    if 1017.43466796875 <= Q.sum_pt < 1066.481811523438:
        z += 0.020596896298229694 * Q.sum_pt - 20.782014906857604
    if Q.sum_pt >= 1066.481811523438:
        z += -0.0063733896240592 * Q.sum_pt + 7.981304480850131
    if Q.mass < 92.85979309082:
        z += 0.04659357666969299 * Q.mass - 4.32666988890895
    if Q.mass_over_sum_pt < 0.09795414517:
        z += -37.26441192626953 * Q.mass_over_sum_pt + 3.650203615500485
    if Q.sum_pt_top20 >= 1129.275:
        z += 0.06607407331466675 * Q.sum_pt_top20 - 74.6157991424203
    if 1038.855053710938 <= Q.sum_pt_top50 < 1107.225842285156:
        z += -0.016880720853805542 * Q.sum_pt_top50 + 17.536622169259505
    if 1107.225842285156 <= Q.sum_pt_top50 < 1156.659497070312:
        z += -0.0195579647552222 * Q.sum_pt_top50 + 20.500935803008364
    if Q.sum_pt_top50 >= 1156.659497070312:
        z += 0.016993086086586118 * Q.sum_pt_top50 - 21.776184281069053
    if Q.n_pt_above_10 >= 18.0:
        z += 0.09008165448904037 * Q.n_pt_above_10 - 1.6214697808027267
    if Q.z_top30_slots >= 0.904849218002:
        z += 10.877532005310059 * Q.z_top30_slots - 9.842526328796534
    if Q.sum_pt_top40 >= 1013.04248046875:
        z += -0.00013689171464648098 * Q.sum_pt_top40 + 0.1386771221610914
    if Q.mass_top20 >= 125.1:
        z += -0.04389714077115059 * Q.mass_top20 + 5.491532310470938
    if Q.log_sum_pt > 6.903422848462 and Q.mass_top50 > 85.866695580031:
        z += -1.6071686744689941 * (Q.log_sum_pt - 6.903422848462) * (Q.mass_top50 - 85.866695580031)
    if Q.log_sum_pt > 6.903422848462 and Q.girth2_top10 < 0.01976735495:
        z += 192.1863555908203 * (Q.log_sum_pt - 6.903422848462) * (0.01976735495 - Q.girth2_top10)
    if Q.log_sum_pt > 6.903422848462 and Q.n_dr_0p2_0p4 > 4.0:
        z += -3.134734630584717 * (Q.log_sum_pt - 6.903422848462) * (Q.n_dr_0p2_0p4 - 4.0)
    if Q.mass < 92.85979309082 and Q.sum_pt_top40 < 1069.67119140625:
        z += 0.00010526704863877967 * (92.85979309082 - Q.mass) * (1069.67119140625 - Q.sum_pt_top40)
    if Q.sum_pt_top20 > 1129.275 and Q.pt_7 > 45.75:
        z += 0.0002762837684713304 * (Q.sum_pt_top20 - 1129.275) * (Q.pt_7 - 45.75)
    if Q.log_sum_pt > 6.903422848462 and Q.n_dr_0p05_0p1 > 9.0:
        z += 0.4875038266181946 * (Q.log_sum_pt - 6.903422848462) * (Q.n_dr_0p05_0p1 - 9.0)
    if Q.sum_pt_top50 > 1107.225842285156 and Q.z_top20_slots > 0.806826560894:
        z += -0.4003058969974518 * (Q.sum_pt_top50 - 1107.225842285156) * (Q.z_top20_slots - 0.806826560894)
    if Q.sum_pt_top50 > 1156.659497070312 and Q.n_pt_above_50 < 5.0:
        z += -0.050919558852910995 * (Q.sum_pt_top50 - 1156.659497070312) * (5.0 - Q.n_pt_above_50)
    if Q.sum_pt_top50 > 997.018872070312 and Q.mean_phi > 0.000107912998:
        z += 7.400318145751953 * (Q.sum_pt_top50 - 997.018872070312) * (Q.mean_phi - 0.000107912998)
    if Q.sum_pt_top50 > 1038.855053710938 and Q.dr_3 > 0.004614387814:
        z += 0.1370658278465271 * (Q.sum_pt_top50 - 1038.855053710938) * (Q.dr_3 - 0.004614387814)
    if Q.sum_pt_top20 > 1129.275 and Q.eta_1 > 0.041473388672:
        z += 0.7253297567367554 * (Q.sum_pt_top20 - 1129.275) * (Q.eta_1 - 0.041473388672)
    if Q.sum_pt_top40 > 1013.04248046875 and Q.eta_5 < -0.113525390625:
        z += -3.7487266063690186 * (Q.sum_pt_top40 - 1013.04248046875) * (-0.113525390625 - Q.eta_5)
    if Q.sum_pt_top20 > 1129.275 and Q.girth2_top3 < 0.007929074034:
        z += -0.35712820291519165 * (Q.sum_pt_top20 - 1129.275) * (0.007929074034 - Q.girth2_top3)
    if Q.sum_pt_top15 > 1003.329150390625 and Q.girth2_top3 < 0.003952581551:
        z += -4.2008819580078125 * (Q.sum_pt_top15 - 1003.329150390625) * (0.003952581551 - Q.girth2_top3)
    if Q.log_sum_pt > 6.903422848462 and Q.n_real_top30 > 22.0:
        z += -1.755661964416504 * (Q.log_sum_pt - 6.903422848462) * (Q.n_real_top30 - 22.0)
    if Q.mass_top40 > 160.8 and Q.pt_9 < 41.4375:
        z += -0.02366825006902218 * (Q.mass_top40 - 160.8) * (41.4375 - Q.pt_9)
    return max(0.0, z)


def neuron_3(Q):
    z = -0.3057844042778015
    if Q.n_particles < 46.0:
        z += -0.05663055554032326 * Q.n_particles + 2.60500555485487
    if Q.mass_over_sum_pt_sq < 0.00750911433:
        z += -228.8927764892578 * Q.mass_over_sum_pt_sq + 1.718782027968973
    if Q.lam1 < 0.00767124277:
        z += 84.1136245727539 * Q.lam1 - 0.6452560343622328
    if Q.n_dr_0p1_0p2 < 15.0:
        z += -0.04191688820719719 * Q.n_dr_0p1_0p2 + 0.6287533231079578
    if Q.n_dr_0p2_0p4 < 7.0:
        z += -0.08358047902584076 * Q.n_dr_0p2_0p4 + 0.5850633531808853
    if Q.n_dr_0p2_0p4 < 5.0 and Q.sum_pt_top30 < 988.4375:
        z += -0.0032267658971250057 * (5.0 - Q.n_dr_0p2_0p4) * (988.4375 - Q.sum_pt_top30)
    if Q.n_particles < 46.0 and Q.mass_top30 > 73.331387415761:
        z += -0.001710513373836875 * (46.0 - Q.n_particles) * (Q.mass_top30 - 73.331387415761)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.z_top50_slots > 0.978741004761:
        z += 7.242002010345459 * (5.0 - Q.n_dr_0p2_0p4) * (Q.z_top50_slots - 0.978741004761)
    if Q.n_dr_0p2_0p4 < 5.0 and Q.dr_0 < 0.04118638065:
        z += -4.6409711837768555 * (5.0 - Q.n_dr_0p2_0p4) * (0.04118638065 - Q.dr_0)
    if Q.mass_over_sum_pt_sq < 0.00750911433 and Q.mass_top20 < 136.785:
        z += -1.1898595094680786 * (0.00750911433 - Q.mass_over_sum_pt_sq) * (136.785 - Q.mass_top20)
    if Q.tau21 < 0.428145796061 and Q.lam1 < 0.016493544356:
        z += -247.973388671875 * (0.428145796061 - Q.tau21) * (0.016493544356 - Q.lam1)
    return max(0.0, z)


def neuron_4(Q):
    z = 2.5973145961761475
    if Q.n_dr_0p2_0p4 < 15.0:
        z += -0.0431552417576313 * Q.n_dr_0p2_0p4 + 0.6473286263644695
    if Q.mass_top40 < 83.325535102591:
        z += -0.006545957177877426 * Q.mass_top40 + 0.062285685197699614
    if 83.325535102591 <= Q.mass_top40 < 94.642533639752:
        z += 0.04269327223300934 * Q.mass_top40 - 4.040599453503677
    if Q.mass < 80.784643554688:
        z += -0.0035562608391046524 * Q.mass + 2.4130916265743165
    if 80.784643554688 <= Q.mass < 86.4:
        z += -0.036272576078772545 * Q.mass + 5.056067491633694
    if 86.4 <= Q.mass < 101.049709320068:
        z += -0.09434272162616253 * Q.mass + 10.07332806692819
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += -0.027622273191809654 * Q.mass + 3.3312461469322447
    if Q.mass >= 143.787612915039:
        z += 0.020918915048241615 * Q.mass - 3.0078808595591497
    if Q.mass_top30 < 152.688263064041:
        z += 0.00972056109458208 * Q.mass_top30 - 1.484215589539631
    if Q.D2 < 6.916121244431:
        z += -0.10175398737192154 * Q.D2 + 0.7037429137685103
    if Q.mass_top20 >= 103.674910639856:
        z += -0.03146270290017128 * Q.mass_top20 + 3.2618929116635953
    if Q.girth2_top15 < 0.007277630044:
        z += 81.70480346679688 * Q.girth2_top15 - 0.060571296428570065
    if 0.007277630044 <= Q.girth2_top15 < 0.015638355144:
        z += -63.87556457519531 * Q.girth2_top15 + 0.9989087638504097
    if Q.z_dr_0p2_0p4 < 0.068491501734:
        z += 7.714257717132568 * Q.z_dr_0p2_0p4 - 0.5283610958095082
    if Q.girth < 0.061710142531:
        z += 42.50289726257324 * Q.girth - 4.076858988881962
    if 0.061710142531 <= Q.girth < 0.120745175332:
        z += 24.62942886352539 * Q.girth - 2.9738847064533944
    if Q.LHA >= 0.115200825015:
        z += -10.633100509643555 * Q.LHA + 1.2249419511783544
    if Q.e2 >= 0.036805817112:
        z += 37.60992431640625 * Q.e2 - 1.3842639959858098
    if Q.mass_over_sum_pt >= 0.09795414517:
        z += -9.220377922058105 * Q.mass_over_sum_pt + 0.9031742374995426
    if Q.lam1 >= 0.00767124277:
        z += -102.32080841064453 * Q.lam1 + 0.7849277617407121
    if Q.width >= 0.009614971338:
        z += 166.79733276367188 * Q.width - 1.6037515737775534
    if Q.sum_pt < 1066.481811523438:
        z += 0.002584097208455205 * Q.sum_pt - 2.755892672025966
    if Q.n_particles > 29.0 and Q.n_dr_0_0p05 < 25.0:
        z += -0.000753175700083375 * (Q.n_particles - 29.0) * (25.0 - Q.n_dr_0_0p05)
    if Q.mass_top40 < 83.325535102591 and Q.D2 < 6.916121244431:
        z += -0.011189174838364124 * (83.325535102591 - Q.mass_top40) * (6.916121244431 - Q.D2)
    if Q.mass < 74.251806640625 and Q.max_dr < 0.39398368001:
        z += -0.4502427577972412 * (74.251806640625 - Q.mass) * (0.39398368001 - Q.max_dr)
    if Q.mass < 120.60000000000001 and Q.max_dr < 0.402178311348:
        z += 0.10028344392776489 * (120.60000000000001 - Q.mass) * (0.402178311348 - Q.max_dr)
    return max(0.0, z)


def neuron_5(Q):
    z = 0.6732431054115295
    if 0.079990613285 <= Q.mass_over_sum_pt < 0.090467494167:
        z += -10.6699800491333 * Q.mass_over_sum_pt + 0.8534982478688872
    if 0.090467494167 <= Q.mass_over_sum_pt < 0.170880120467:
        z += -48.96699237823486 * Q.mass_over_sum_pt + 4.31813298736541
    if Q.mass_over_sum_pt >= 0.170880120467:
        z += 291.365496635437 * Q.mass_over_sum_pt - 53.83792373412479
    if 6.856374501323 <= Q.log_sum_pt < 6.910130970417:
        z += 11.623663902282715 * Q.log_sum_pt - 79.6961927915598
    if 6.910130970417 <= Q.log_sum_pt < 6.92034855022:
        z += -22.999402046203613 * Q.log_sum_pt + 159.55372750986584
    if 6.92034855022 <= Q.log_sum_pt < 6.959293500649:
        z += -20.378911018371582 * Q.log_sum_pt + 141.4190162245439
    if 6.959293500649 <= Q.log_sum_pt < 6.989450376716:
        z += -17.879115343093872 * Q.log_sum_pt + 124.02220442863324
    if Q.log_sum_pt >= 6.989450376716:
        z += -7.90752911567688 * Q.log_sum_pt + 54.3262973149575
    if Q.sum_pt_top40 >= 906.60234375:
        z += 0.006050888914614916 * Q.sum_pt_top40 - 5.485750071760776
    if Q.sum_pt_top20 >= 1005.0126953125:
        z += 0.004601483233273029 * Q.sum_pt_top20 - 4.624549066707004
    if Q.z_top40_slots >= 0.930046498893:
        z += -25.703947067260742 * Q.z_top40_slots + 23.905865977636847
    if Q.sum_pt_top50 >= 1061.183898925781:
        z += -0.004060984123498201 * Q.sum_pt_top50 + 4.309450965649517
    if 64.485443115234 <= Q.mass < 172.8:
        z += 0.016188276931643486 * Q.mass - 1.043908211209151
    if Q.mass >= 172.8:
        z += -0.007375705987215042 * Q.mass + 3.0279480371696037
    if 91.19 <= Q.mass_top50 < 157.544814151857:
        z += -0.022973299026489258 * Q.mass_top50 + 2.0949351382255554
    if Q.mass_top50 >= 157.544814151857:
        z += -0.053554801270365715 * Q.mass_top50 + 6.91289222572167
    if Q.mass_over_sum_pt_sq >= 0.029200015571:
        z += -667.6892700195312 * Q.mass_over_sum_pt_sq + 19.496537081159936
    if Q.mass_top30 >= 68.286969674465:
        z += 0.035551395267248154 * Q.mass_top30 - 2.4276970504994932
    if Q.z_top50_slots >= 0.958653609576:
        z += 31.62404441833496 * Q.z_top50_slots - 30.316504331028565
    if Q.girth < 0.085894044489:
        z += 8.265230178833008 * Q.girth - 0.7099340486925078
    if Q.sum_pt_top15 >= 951.1375:
        z += -0.005004469305276871 * Q.sum_pt_top15 + 4.75993842384778
    if Q.n_particles < 64.0 and Q.e2 < 0.038759447634:
        z += 1.403361439704895 * (64.0 - Q.n_particles) * (0.038759447634 - Q.e2)
    if Q.n_particles < 64.0 and Q.z_dr_0p2_0p4 > 0.004744913615:
        z += 0.133175790309906 * (64.0 - Q.n_particles) * (Q.z_dr_0p2_0p4 - 0.004744913615)
    if Q.log_sum_pt > 6.92034855022 and Q.girth2_top2 < 0.0140332421:
        z += -468.1085205078125 * (Q.log_sum_pt - 6.92034855022) * (0.0140332421 - Q.girth2_top2)
    if Q.sum_pt > 907.937170410156 and Q.girth2_top2 < 0.0140332421:
        z += 0.2529241740703583 * (Q.sum_pt - 907.937170410156) * (0.0140332421 - Q.girth2_top2)
    if Q.log_sum_pt > 6.989450376716 and Q.z_dr_0p05_0p1 < 0.299250295758:
        z += 16.096647262573242 * (Q.log_sum_pt - 6.989450376716) * (0.299250295758 - Q.z_dr_0p05_0p1)
    return max(0.0, z)


def neuron_6(Q):
    z = 0.018267180770635605
    if Q.mass_top50 < 71.795159472175:
        z += -0.0025249102618545294 * Q.mass_top50 + 0.1812763349027771
    if Q.mass < 86.4:
        z += -0.004407708998769522 * Q.mass - 1.028281191585033
    if 86.4 <= Q.mass < 92.85979309082:
        z += 0.03270367393270135 * Q.mass - 4.234704676864115
    if 92.85979309082 <= Q.mass < 101.049709320068:
        z += 0.10291182855144143 * Q.mass - 10.754219388048618
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += 0.035847550723701715 * Q.mass - 3.977393607795239
    if 120.60000000000001 <= Q.mass < 172.8:
        z += -0.006624923553317785 * Q.mass + 1.1447867900133133
    if Q.mass_over_sum_pt >= 0.050772907168:
        z += -49.3516731262207 * Q.mass_over_sum_pt + 2.505727918223084
    if Q.e2 < 0.043586218357:
        z += -53.80851173400879 * Q.e2 + 2.4725403963567407
    if 0.043586218357 <= Q.e2 < 0.047553086095:
        z += -32.07337951660156 * Q.e2 + 1.5251881775105636
    if Q.e2 >= 0.055571487173:
        z += 99.18788146972656 * Q.e2 - 5.512018082811954
    if 0.0018230789 <= Q.girth2_top20 < 0.016559833876:
        z += 118.24537658691406 * Q.girth2_top20 - 0.21557065107815704
    if Q.girth2_top20 >= 0.016559833876:
        z += -44.36528015136719 * Q.girth2_top20 + 2.4772348109750406
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.02359016053378582 * Q.n_dr_0p2_0p4 - 0.4953933712095022
    if Q.lam1 < 0.007259287357:
        z += -189.6645965576172 * Q.lam1 + 1.3768298078612162
    if 0.008241985248 <= Q.lam1 < 0.011744050682:
        z += 34.70108413696289 * Q.lam1 - 0.28600582354645493
    if Q.lam1 >= 0.011744050682:
        z += -72.63299942016602 * Q.lam1 + 0.97453109365449
    if 0.228402115913 <= Q.LHA < 0.404204003833:
        z += 18.600404739379883 * Q.LHA - 4.248371799312559
    if Q.LHA >= 0.404204003833:
        z += 5.209841728210449 * Q.LHA + 1.1641473833801985
    if Q.girth2_top50 < 0.013514311784:
        z += 48.84141540527344 * Q.girth2_top50 - 0.660058115758726
    if Q.width < 0.009614971338:
        z += 72.91824340820312 * Q.width - 0.7011068203871804
    if Q.girth2_top3 >= 0.010023689877:
        z += -181.4571075439453 * Q.girth2_top3 + 1.8188697719979448
    if Q.mass_over_sum_pt_sq < 0.019863807341:
        z += -15.762908935546875 * Q.mass_over_sum_pt_sq + 0.3131113862294305
    if Q.mass_over_sum_pt > 0.050772907168 and Q.n_dr_0p2_0p4 < 21.0:
        z += 1.0687673091888428 * (Q.mass_over_sum_pt - 0.050772907168) * (21.0 - Q.n_dr_0p2_0p4)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.lam2 < 0.002396991421:
        z += 8489.763671875 * (Q.mass_over_sum_pt - 0.050772907168) * (0.002396991421 - Q.lam2)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.z_dr_0_0p05 < 0.908491230011:
        z += 5.735375881195068 * (Q.mass_over_sum_pt - 0.050772907168) * (0.908491230011 - Q.z_dr_0_0p05)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.z_dr_0p1_0p2 < 0.286492615938:
        z += 37.736778259277344 * (Q.mass_over_sum_pt - 0.050772907168) * (0.286492615938 - Q.z_dr_0p1_0p2)
    if Q.mass < 120.60000000000001 and Q.n_dr_0p05_0p1 > 12.0:
        z += -0.000970913446508348 * (120.60000000000001 - Q.mass) * (Q.n_dr_0p05_0p1 - 12.0)
    if Q.mass < 92.85979309082 and Q.sum_pt_top15 < 1082.54765625:
        z += -3.391467544133775e-05 * (92.85979309082 - Q.mass) * (1082.54765625 - Q.sum_pt_top15)
    if Q.LHA > 0.228402115913 and Q.D2 < 3.345338582993:
        z += -3.06697678565979 * (Q.LHA - 0.228402115913) * (3.345338582993 - Q.D2)
    if Q.mass_over_sum_pt > 0.050772907168 and Q.max_pair_mass > 13.047927274731:
        z += -0.17509862780570984 * (Q.mass_over_sum_pt - 0.050772907168) * (Q.max_pair_mass - 13.047927274731)
    if Q.girth2_top3 > 0.010023689877 and Q.D2 < 4.450168704987:
        z += 74.95030212402344 * (Q.girth2_top3 - 0.010023689877) * (4.450168704987 - Q.D2)
    if Q.lam1 > 0.008241985248 and Q.pt_4 < 68.125:
        z += 1.423485517501831 * (Q.lam1 - 0.008241985248) * (68.125 - Q.pt_4)
    if Q.LHA > 0.404204003833 and Q.z_2 > 0.049737748174:
        z += 790.4779052734375 * (Q.LHA - 0.404204003833) * (Q.z_2 - 0.049737748174)
    return max(0.0, z)


def neuron_7(Q):
    z = -0.509419858455658
    if Q.max_dr < 0.383415880799:
        z += 3.927088975906372 * Q.max_dr - 1.5057082786731846
    if Q.mass < 82.85408782959:
        z += 0.24123599380254745 * Q.mass - 19.155538600754873
    if 82.85408782959 <= Q.mass < 86.4:
        z += 0.07146025449037552 * Q.mass - 5.088924584450605
    if 86.4 <= Q.mass < 91.19:
        z += 0.06253226101398468 * Q.mass - 4.317545948090436
    if 91.19 <= Q.mass < 101.049709320068:
        z += -0.07497826218605042 * Q.mass + 8.222038662520765
    if 101.049709320068 <= Q.mass < 120.60000000000001:
        z += -0.03301777318120003 * Q.mass + 3.9819434456527234
    if Q.z_top50_slots < 0.990637830118:
        z += 22.957216262817383 * Q.z_top50_slots - 22.742286904147072
    if Q.mass_top50 < 71.795159472175:
        z += 0.19667738862335682 * Q.mass_top50 - 14.75035442148004
    if 71.795159472175 <= Q.mass_top50 < 97.930041729355:
        z += 0.024100737646222115 * Q.mass_top50 - 2.3601862434027687
    if Q.mass_top30 < 76.415438713532:
        z += -0.018023211508989334 * Q.mass_top30 + 1.377251614486199
    if Q.C2 < 0.056027559564:
        z += 17.793136596679688 * Q.C2 - 0.9969060205008594
    if Q.girth2_top20 < 0.006374177987:
        z += 108.70675659179688 * Q.girth2_top20 - 1.283154853901371
    if 0.006374177987 <= Q.girth2_top20 < 0.008031209355:
        z += 356.20245361328125 * Q.girth2_top20 - 2.8607364777329383
    if Q.mass_top20 < 66.841467317407:
        z += -0.021029656752943993 * Q.mass_top20 + 1.4056531145481932
    if Q.girth2_top40 < 0.007709915821:
        z += 105.51004028320312 * Q.girth2_top40 - 0.42475329393140693
    if 0.007709915821 <= Q.girth2_top40 < 0.008031986041:
        z += 366.78009033203125 * Q.girth2_top40 - 2.4391233863563286
    if 0.008031986041 <= Q.girth2_top40 < 0.008840538245:
        z += -172.83966064453125 * Q.girth2_top40 + 1.8950949209353176
    if 0.008840538245 <= Q.girth2_top40 < 0.012926423095:
        z += -89.8457260131836 * Q.girth2_top40 + 1.1613838677238586
    if Q.girth < 0.085894044489:
        z += 50.03963088989258 * Q.girth - 4.298106281869572
    if Q.LHA < 0.320332145368:
        z += -9.797952651977539 * Q.LHA + 3.13859919322205
    if Q.z_dr_0p2_0p4 < 0.091225683689:
        z += 5.380295753479004 * Q.z_dr_0p2_0p4 - 0.4908211585601455
    if Q.e2_sq < 0.009606007381:
        z += -460.247802734375 * Q.e2_sq + 4.421143790155439
    if Q.mass_over_sum_pt < 0.090467494167:
        z += 12.025125503540039 * Q.mass_over_sum_pt - 1.0878829713489515
    if Q.D2 < 1.788105106354 and Q.n_dr_0p2_0p4 < 9.0:
        z += 0.15948525071144104 * (1.788105106354 - Q.D2) * (9.0 - Q.n_dr_0p2_0p4)
    if Q.mass < 91.19 and Q.z_dr_0p05_0p1 > 0.402664637566:
        z += -0.20603910088539124 * (91.19 - Q.mass) * (Q.z_dr_0p05_0p1 - 0.402664637566)
    if Q.mass_top50 < 97.930041729355 and Q.D2 < 1.601009327173:
        z += -0.39831313490867615 * (97.930041729355 - Q.mass_top50) * (1.601009327173 - Q.D2)
    if Q.mass < 101.049709320068 and Q.D2 < 1.601009327173:
        z += 0.2793334424495697 * (101.049709320068 - Q.mass) * (1.601009327173 - Q.D2)
    if Q.n_dr_0p2_0p4 < 13.0 and Q.z_1st < 0.499317836761:
        z += 0.16723619401454926 * (13.0 - Q.n_dr_0p2_0p4) * (0.499317836761 - Q.z_1st)
    if Q.mass < 91.19 and Q.max_dr < 0.390781164169:
        z += -1.0019567012786865 * (91.19 - Q.mass) * (0.390781164169 - Q.max_dr)
    if Q.mass_top50 < 71.795159472175 and Q.max_dr < 0.39398368001:
        z += 0.512876033782959 * (71.795159472175 - Q.mass_top50) * (0.39398368001 - Q.max_dr)
    if Q.mass < 101.049709320068 and Q.max_dr < 0.390781164169:
        z += 0.8081940412521362 * (101.049709320068 - Q.mass) * (0.390781164169 - Q.max_dr)
    if Q.mass < 91.19 and Q.pt_dispersion < 0.33780374676:
        z += -0.31195420026779175 * (91.19 - Q.mass) * (0.33780374676 - Q.pt_dispersion)
    return max(0.0, z)


def neuron_8(Q):
    z = -0.438579261302948
    if 0.076966318366 <= Q.mass_over_sum_pt < 0.088731426731:
        z += 17.83864402770996 * Q.mass_over_sum_pt - 1.3729747554544693
    if 0.088731426731 <= Q.mass_over_sum_pt < 0.09795414517:
        z += 26.838333129882812 * Q.mass_over_sum_pt - 2.171530009625699
    if Q.mass_over_sum_pt >= 0.09795414517:
        z += -44.14337921142578 * Q.mass_over_sum_pt + 4.781422945470024
    if Q.sum_pt_top40 < 1001.52314453125:
        z += -0.006417724769562483 * Q.sum_pt_top40 + 6.427499891948309
    if Q.girth2_top40 >= 0.005196965925:
        z += 54.73317337036133 * Q.girth2_top40 - 0.28444643697288524
    if 64.485443115234 <= Q.mass < 80.784643554688:
        z += 0.05262197554111481 * Q.mass - 3.393351410367794
    if 80.784643554688 <= Q.mass < 87.363773345947:
        z += 0.1054001897573471 * Q.mass - 7.657020633279087
    if 87.363773345947 <= Q.mass < 101.049709320068:
        z += 0.13962643966078758 * Q.mass - 10.647154972325001
    if 101.049709320068 <= Q.mass < 143.787612915039:
        z += 0.032441627234220505 * Q.mass + 0.18383916690561541
    if Q.mass >= 143.787612915039:
        z += 0.010372446849942207 * Q.mass + 3.357113933352395
    if 0.005718442372 <= Q.girth2_top20 < 0.008031209355:
        z += -121.70600891113281 * Q.girth2_top20 + 0.6959687982844315
    if Q.girth2_top20 >= 0.008031209355:
        z += 104.94068908691406 * Q.girth2_top20 - 1.1242782829573423
    if Q.sum_pt < 1012.672900390625:
        z += -0.012933576479554176 * Q.sum_pt + 13.097482405974098
    if Q.girth2_top30 < 0.007463984647:
        z += -69.24069213867188 * Q.girth2_top30 + 0.5168114630707005
    if Q.mass_over_sum_pt_sq < 0.006166777647:
        z += 318.33331298828125 * Q.mass_over_sum_pt_sq - 2.4722699045507834
    if 0.006166777647 <= Q.mass_over_sum_pt_sq < 0.013977372691:
        z += 65.19082641601562 * Q.mass_over_sum_pt_sq - 0.9111964768509382
    if Q.girth < 0.097499583662:
        z += -27.83565902709961 * Q.girth + 2.713965166099604
    if Q.lam2 < 0.001411893759:
        z += 412.2333068847656 * Q.lam2 - 0.5820296332425323
    if 82.04491364955 <= Q.mass_top50 < 97.930041729355:
        z += -0.026086529716849327 * Q.mass_top50 + 2.140267078035323
    if Q.mass_top50 >= 97.930041729355:
        z += -0.006868988275527954 * Q.mass_top50 + 0.25829244275111174
    if Q.e2 >= 0.025159193203:
        z += -20.268423080444336 * Q.e2 + 0.5099371722010434
    if Q.lam1 < 0.008241985248:
        z += -21.354721069335938 * Q.lam1 - 0.918360355329803
    if 0.008241985248 <= Q.lam1 < 0.020485236462:
        z += 89.38521575927734 * Q.lam1 - 1.8310772810356855
    if Q.z_dr_0p2_0p4 < 0.091225683689:
        z += -9.713550567626953 * Q.z_dr_0p2_0p4 + 0.8861252915794428
    if Q.z_dr_0p1_0p2 < 0.154838323593:
        z += -3.8383190631866455 * Q.z_dr_0p1_0p2 + 0.5943188891588744
    if Q.n_dr_0p2_0p4 < 21.0:
        z += 0.059130553156137466 * Q.n_dr_0p2_0p4 - 1.2417416162788868
    if Q.mass_top40 >= 79.554505888974:
        z += -0.007670497987419367 * Q.mass_top40 + 0.6102226773115172
    if Q.sum_pt_top30 < 1011.52392578125:
        z += 0.0030029204208403826 * Q.sum_pt_top30 - 3.037525852897147
    if Q.mass_over_sum_pt > 0.076966318366 and Q.sum_pt < 1115.722741699219:
        z += 0.18614104390144348 * (Q.mass_over_sum_pt - 0.076966318366) * (1115.722741699219 - Q.sum_pt)
    if Q.sum_pt_top40 < 1001.52314453125 and Q.log_sum_pt < 6.811175180312:
        z += -0.01995871029794216 * (1001.52314453125 - Q.sum_pt_top40) * (6.811175180312 - Q.log_sum_pt)
    if Q.mass > 87.363773345947 and Q.log_sum_pt < 6.903422848462:
        z += -0.40192490816116333 * (Q.mass - 87.363773345947) * (6.903422848462 - Q.log_sum_pt)
    return max(0.0, z)


def neuron_9(Q):
    z = 0.907929003238678
    if Q.mass_top40 < 80.890431271924:
        z += 0.0025428817607462406 * Q.mass_top40 - 0.21869479878573728
    if 80.890431271924 <= Q.mass_top40 < 91.288535717504:
        z += 0.017977279145270586 * Q.mass_top40 - 1.467189859642167
    if 91.288535717504 <= Q.mass_top40 < 120.60000000000001:
        z += -0.005933843087404966 * Q.mass_top40 + 0.715621476341039
    if Q.girth2_top40 < 0.00625977218:
        z += -129.66827392578125 * Q.girth2_top40 + 0.8116938537492249
    if Q.mass < 62.55:
        z += -0.007384235039353371 * Q.mass + 1.0100525898579509
    if 62.55 <= Q.mass < 82.85408782959:
        z += -0.05835689790546894 * Q.mass + 4.19839265213348
    if 82.85408782959 <= Q.mass < 92.85979309082:
        z += -0.009382063522934914 * Q.mass + 0.14062742276338103
    if 92.85979309082 <= Q.mass < 136.785:
        z += 0.02425250969827175 * Q.mass - 2.9826720872559047
    if 136.785 <= Q.mass < 143.787612915039:
        z += 0.03163674473762512 * Q.mass - 3.9927246771138556
    if Q.mass >= 143.787612915039:
        z += -0.07029730826616287 * Q.mass + 10.66412947905588
    if Q.girth2_top15 < 0.009962397174:
        z += 44.97222900390625 * Q.girth2_top15 - 1.679645161170923
    if 0.009962397174 <= Q.girth2_top15 < 0.02146577947:
        z += 107.06537628173828 * Q.girth2_top15 - 2.298241756136363
    if Q.girth < 0.02783744745:
        z += 145.38814544677734 * Q.girth - 5.297696178540427
    if 0.02783744745 <= Q.girth < 0.043628720567:
        z += 79.18685913085938 * Q.girth - 3.454821349598656
    if Q.girth >= 0.097499583662:
        z += 11.98563289642334 * Q.girth - 1.1685942173268469
    if Q.sum_pt < 949.91689453125:
        z += -0.03244815021753311 * Q.sum_pt + 30.82304608792256
    if Q.log_sum_pt < 6.856374501323:
        z += 19.793441772460938 * Q.log_sum_pt - 135.71124946212268
    if Q.lam1 < 0.005240342196:
        z += -152.75726318359375 * Q.lam1 + 0.8005003320064636
    if Q.LHA < 0.209102506978:
        z += -26.11514663696289 * Q.LHA + 5.460742631887026
    if Q.mass_top50 < 136.785:
        z += -0.040032047778367996 * Q.mass_top50 + 4.866436014957726
    if 136.785 <= Q.mass_top50 < 160.8:
        z += 0.02537362650036812 * Q.mass_top50 - 4.080079141259194
    if Q.girth2 < 0.00363885588:
        z += -413.0216979980469 * Q.girth2 + 1.502926434327777
    if Q.e2 < 0.047553086095:
        z += 35.27788543701172 * Q.e2 - 1.6775723234357651
    if Q.girth2_top15 < 0.02146577947 and Q.n_particles > 41.0:
        z += 1.8117737770080566 * (0.02146577947 - Q.girth2_top15) * (Q.n_particles - 41.0)
    return max(0.0, z)


def neuron_10(Q):
    z = 3.7825193405151367
    if Q.mass < 62.55:
        z += -0.034069569781422615 * Q.mass + 2.4551499495282774
    if 62.55 <= Q.mass < 80.4:
        z += -0.05084612779319286 * Q.mass + 3.5045236531645063
    if 80.4 <= Q.mass < 86.4:
        z += 0.0025458168238401413 * Q.mass - 0.7881886940449476
    if 86.4 <= Q.mass < 120.60000000000001:
        z += 0.016614915803074837 * Q.mass - 2.0037588458508253
    if 143.787612915039 <= Q.mass < 162.836349487305:
        z += -0.0845944881439209 * Q.mass + 12.163639515983954
    if Q.mass >= 162.836349487305:
        z += -0.18016591668128967 * Q.mass + 27.72614205429593
    if Q.max_dr < 0.402178311348:
        z += 4.307224273681641 * Q.max_dr - 1.732272184986398
    if Q.z_dr_0_0p05 >= 0.767473447323:
        z += -4.3414692878723145 * Q.z_dr_0_0p05 + 3.331962400810295
    if Q.girth2_top5 < 0.024419631481:
        z += 64.42584991455078 * Q.girth2_top5 - 1.5732555127635455
    if Q.lam1 >= 0.001868040786:
        z += -59.44790267944336 * Q.lam1 + 0.11105110684735887
    if Q.e2 < 0.065240035206:
        z += 45.19682312011719 * Q.e2 - 2.9486423315558
    if Q.sum_pt_top50 < 959.095727539062:
        z += 0.006979571655392647 * Q.sum_pt_top50 - 6.694077354739826
    if Q.girth2_top15 < 0.002197764741:
        z += 296.3479309082031 * Q.girth2_top15 - 0.651303033618353
    if Q.n_dr_0p2_0p4 < 6.0:
        z += 0.12834402546286583 * Q.n_dr_0p2_0p4 - 1.1201039515435696
    if 6.0 <= Q.n_dr_0p2_0p4 < 13.0:
        z += 0.05000568553805351 * Q.n_dr_0p2_0p4 - 0.6500739119946957
    if 0.000237176831 <= Q.girth2_top10 < 0.007678543663:
        z += 125.7420883178711 * Q.girth2_top10 - 0.029823110030554788
    if Q.girth2_top10 >= 0.007678543663:
        z += 26.588775634765625 * Q.girth2_top10 + 0.7315299307377622
    if Q.mass_top50 >= 136.785:
        z += 0.08144120126962662 * Q.mass_top50 - 11.139934715665877
    if Q.z_dr_0p2_0p4 < 0.091225683689:
        z += -7.857697010040283 * Q.z_dr_0p2_0p4 + 0.716823781961936
    if Q.girth2_top5 < 0.024419631481 and Q.sum_pt_top3 < 656.384375:
        z += 0.16547848284244537 * (0.024419631481 - Q.girth2_top5) * (656.384375 - Q.sum_pt_top3)
    if Q.dr_0 < 0.064132973195 and Q.n_dr_0p2_0p4 > 2.0:
        z += 0.7050637006759644 * (0.064132973195 - Q.dr_0) * (Q.n_dr_0p2_0p4 - 2.0)
    if Q.e2 < 0.065240035206 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += -37.98716354370117 * (0.065240035206 - Q.e2) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.mass < 120.60000000000001 and Q.tau21 < 0.470341457427:
        z += 0.046336591243743896 * (120.60000000000001 - Q.mass) * (0.470341457427 - Q.tau21)
    return max(0.0, z)


def neuron_11(Q):
    z = 0.8777630925178528
    if Q.z_dr_0p2_0p4 < 0.006381743611:
        z += -84.45153045654297 * Q.z_dr_0p2_0p4 + 0.538948014930215
    if Q.mass < 62.55:
        z += -0.01338949054479599 * Q.mass + 2.5490554315041156
    if 62.55 <= Q.mass < 92.85979309082:
        z += -0.05237486958503723 * Q.mass + 4.987590890471205
    if 92.85979309082 <= Q.mass < 101.049709320068:
        z += -0.015149280428886414 * Q.mass + 1.5308303837471673
    if Q.lam1 < 0.006716736591:
        z += 176.2626953125 * Q.lam1 - 1.183910095233753
    if Q.e2 < 0.027935993578:
        z += -23.72097396850586 * Q.e2 + 0.31203297916383477
    if 0.027935993578 <= Q.e2 < 0.038759447634:
        z += 32.39594268798828 * Q.e2 - 1.255648844169147
    if Q.girth2_top30 < 0.006363915755:
        z += 292.9962463378906 * Q.girth2_top30 - 1.8646034282255632
    if Q.sum_pt_top3 >= 439.125:
        z += 0.002143935998901725 * Q.sum_pt_top3 - 0.9414558955177199
    if Q.mass_top50 < 86.4:
        z += 0.06108064576983452 * Q.mass_top50 - 5.277367794513703
    if Q.girth2_top5 < 0.007164202106:
        z += 78.63157653808594 * Q.girth2_top5 - 0.5633325062322555
    if Q.mass_over_sum_pt_sq < 0.008184367501:
        z += -182.42401123046875 * Q.mass_over_sum_pt_sq + 1.4930251489167075
    if Q.z_top5_slots >= 0.534625950898:
        z += -4.5898356437683105 * Q.z_top5_slots + 2.453845245515167
    if Q.n_dr_0p2_0p4 < 10.0 and Q.z_dr_0p2_0p4 < 0.068491501734:
        z += 1.944602131843567 * (10.0 - Q.n_dr_0p2_0p4) * (0.068491501734 - Q.z_dr_0p2_0p4)
    if Q.mass < 80.4 and Q.z_dr_0p2_0p4 < 0.037001823448:
        z += -4.48114538192749 * (80.4 - Q.mass) * (0.037001823448 - Q.z_dr_0p2_0p4)
    if Q.girth2_top5 < 0.007164202106 and Q.eta_0 < 0.055209350586:
        z += 911.0159301757812 * (0.007164202106 - Q.girth2_top5) * (0.055209350586 - Q.eta_0)
    return max(0.0, z)


def neuron_12(Q):
    z = -0.2279094159603119
    if Q.mass < 62.55:
        z += -0.04392927326261997 * Q.mass + 4.344591570965589
    if 62.55 <= Q.mass < 74.251806640625:
        z += -0.10661800391972065 * Q.mass + 8.265771673567237
    if 74.251806640625 <= Q.mass < 82.85408782959:
        z += -0.05217552371323109 * Q.mass + 4.223319160238919
    if 82.85408782959 <= Q.mass < 86.4:
        z += 0.028098909184336662 * Q.mass - 2.427745753526688
    if Q.LHA < 0.228402115913:
        z += -5.069285869598389 * Q.LHA + 1.157835618784144
    if Q.LHA >= 0.404204003833:
        z += 17.51419448852539 * Q.LHA - 7.079307536171824
    if Q.log_sum_pt < 6.856374501323:
        z += 20.382123947143555 * Q.log_sum_pt - 139.74747491399995
    if Q.mass_top40 < 77.936678808178:
        z += 0.016243821009993553 * Q.mass_top40 - 1.2659894606734012
    if Q.girth < 0.050483809784:
        z += 50.3454475402832 * Q.girth - 2.541629997114008
    if Q.girth2_top50 < 0.00742997247:
        z += -119.93897247314453 * Q.girth2_top50 + 0.8911432635555516
    if Q.mass_top50 < 77.376408295162:
        z += -0.007440194487571716 * Q.mass_top50 + 0.7264057870039733
    if 77.376408295162 <= Q.mass_top50 < 80.4:
        z += -0.04984477907419205 * Q.mass_top50 + 4.007520237565041
    if Q.mass < 86.4 and Q.z_dr_0p1_0p2 < 0.064725840837:
        z += -0.19470073282718658 * (86.4 - Q.mass) * (0.064725840837 - Q.z_dr_0p1_0p2)
    if Q.mass_top40 < 89.67879517394 and Q.sum_pt_top2 < 501.625:
        z += 5.439686356112361e-05 * (89.67879517394 - Q.mass_top40) * (501.625 - Q.sum_pt_top2)
    if Q.girth2_top50 < 0.00742997247 and Q.pt_dispersion < 0.428993919492:
        z += -915.3133544921875 * (0.00742997247 - Q.girth2_top50) * (0.428993919492 - Q.pt_dispersion)
    return max(0.0, z)


def neuron_13(Q):
    z = 2.0710856914520264
    if 74.251806640625 <= Q.mass < 136.785:
        z += 0.02417592518031597 * Q.mass - 1.7951061218470388
    if 136.785 <= Q.mass < 143.787612915039:
        z += -0.005381910130381584 * Q.mass + 2.247962381126726
    if 143.787612915039 <= Q.mass < 160.8:
        z += -0.1258369069546461 * Q.mass + 19.567898838176326
    if 160.8 <= Q.mass < 172.8:
        z += -0.20674333535134792 * Q.mass + 32.57765252436598
    if Q.mass >= 172.8:
        z += -0.07040455006062984 * Q.mass + 9.018310426129892
    if Q.sum_pt < 986.05654296875:
        z += 0.042516890447586775 * Q.sum_pt - 43.30236895611026
    if 986.05654296875 <= Q.sum_pt < 1012.672900390625:
        z += 0.035817128606140614 * Q.sum_pt - 36.69602495601991
    if 1012.672900390625 <= Q.sum_pt < 1052.889428710937:
        z += 0.010567531920969486 * Q.sum_pt - 11.126442647154153
    if Q.sum_pt_top40 < 956.21328125:
        z += -0.0002874433994293213 * Q.sum_pt_top40 + 1.071604891204788
    if 956.21328125 <= Q.sum_pt_top40 < 1053.04736328125:
        z += -0.008227967657148838 * Q.sum_pt_top40 + 8.664439646523988
    if Q.mass_over_sum_pt >= 0.170880120467:
        z += 277.15570068359375 * Q.mass_over_sum_pt - 47.36039952092829
    if 97.930041729355 <= Q.mass_top50 < 136.785:
        z += -0.03045378439128399 * Q.mass_top50 + 2.982340376255221
    if Q.mass_top50 >= 136.785:
        z += 0.0881414283066988 * Q.mass_top50 - 13.239705792638352
    if Q.mass_over_sum_pt_sq >= 0.029200015571:
        z += -803.923095703125 * Q.mass_over_sum_pt_sq + 23.474566912417774
    if Q.sum_pt_top50 < 1013.915698242188:
        z += -0.0174653809517622 * Q.sum_pt_top50 + 17.70842392277178
    if Q.log_sum_pt < 7.017257672702:
        z += 2.6067447662353516 * Q.log_sum_pt - 18.2921997116408
    if Q.mass > 74.251806640625 and Q.sum_pt < 1017.43466796875:
        z += 7.684199954383075e-05 * (Q.mass - 74.251806640625) * (1017.43466796875 - Q.sum_pt)
    if Q.mass > 74.251806640625 and Q.D2 > 0.603279101849:
        z += -0.003970509860664606 * (Q.mass - 74.251806640625) * (Q.D2 - 0.603279101849)
    if Q.sum_pt_top40 < 1053.04736328125 and Q.D2 < 5.378974604607:
        z += -0.0018830006010830402 * (1053.04736328125 - Q.sum_pt_top40) * (5.378974604607 - Q.D2)
    if Q.sum_pt_top50 < 959.095727539062 and Q.D2 < 4.450168704987:
        z += 0.0038849730044603348 * (959.095727539062 - Q.sum_pt_top50) * (4.450168704987 - Q.D2)
    return max(0.0, z)


def neuron_14(Q):
    z = 0.7368783354759216
    if Q.girth2 < 0.007877041167:
        z += -272.52490234375 * Q.girth2 + 2.1466898747943737
    if Q.mass_over_sum_pt < 0.090467494167:
        z += 18.14312171936035 * Q.mass_over_sum_pt + 1.0969287445334062
    if 0.090467494167 <= Q.mass_over_sum_pt < 0.09795414517:
        z += -142.98404502868652 * Q.mass_over_sum_pt + 15.673699762457574
    if 0.09795414517 <= Q.mass_over_sum_pt < 0.118225939153:
        z += -60.107404708862305 * Q.mass_over_sum_pt + 7.555589305367636
    if 0.118225939153 <= Q.mass_over_sum_pt < 0.140939019879:
        z += -19.783090591430664 * Q.mass_over_sum_pt + 2.7882093981337044
    if Q.mass_top40 < 67.726432644245:
        z += -0.07887742994353175 * Q.mass_top40 + 4.5857391684151585
    if 67.726432644245 <= Q.mass_top40 < 80.890431271924:
        z += 0.02656489284709096 * Q.mass_top40 - 2.555493203916689
    if 80.890431271924 <= Q.mass_top40 < 91.288535717504:
        z += 0.0087923021055758 * Q.mass_top40 - 1.117860674016124
    if 91.288535717504 <= Q.mass_top40 < 111.24867219155:
        z += 0.015792692080140114 * Q.mass_top40 - 1.7569160242455955
    if Q.mass_top50 < 89.080094718389:
        z += -0.05827270820736885 * Q.mass_top50 + 5.482538859592375
    if 89.080094718389 <= Q.mass_top50 < 117.048742792994:
        z += -0.010425977408885956 * Q.mass_top50 + 1.2203475480982584
    if Q.mass < 80.4:
        z += 0.19915689155459404 * Q.mass - 16.5019142350682
    if 80.4 <= Q.mass < 89.741833496094:
        z += 0.14002449065446854 * Q.mass - 11.747669202698109
    if 89.741833496094 <= Q.mass < 91.034691238403:
        z += 0.08121464774012566 * Q.mass - 6.469966071947707
    if 91.034691238403 <= Q.mass < 136.785:
        z += -0.020183127373456955 * Q.mass + 2.7607490777783097
    if Q.max_dr < 0.240474711359:
        z += 3.1296546459198 * Q.max_dr - 1.0377488456496138
    if 0.240474711359 <= Q.max_dr < 0.331585738063:
        z += -0.7266402244567871 * Q.max_dr - 0.11040744978061168
    if 0.331585738063 <= Q.max_dr < 0.402178311348:
        z += -3.856294870376587 * Q.max_dr + 0.9273413958690021
    if Q.max_dr >= 0.402178311348:
        z += -0.887401819229126 * Q.max_dr - 0.26668299801429496
    if Q.lam2 < 0.002396991421:
        z += -501.8443298339844 * Q.lam2 + 1.2029165532895547
    if Q.lam1 < 0.006189818106:
        z += 28.14521026611328 * Q.lam1 - 0.30689776862041507
    if 0.006189818106 <= Q.lam1 < 0.007259287357:
        z += -271.23741912841797 * Q.lam1 + 1.5462262514277423
    if 0.007259287357 <= Q.lam1 < 0.011744050682:
        z += 94.26676177978516 * Q.lam1 - 1.1070736279698175
    if Q.sum_pt_top50 >= 976.277001953125:
        z += -0.013089603744447231 * Q.sum_pt_top50 + 12.779079100383342
    if Q.girth2_top30 < 0.012157872869:
        z += 239.2862548828125 * Q.girth2_top30 - 2.909211866164365
    if Q.mass_top30 < 91.19:
        z += -0.01131950318813324 * Q.mass_top30 + 1.03222549572587
    if Q.girth < 0.085894044489:
        z += 6.241323947906494 * Q.girth - 0.5360925568517415
    if Q.e2 < 0.03263075389:
        z += -22.028743743896484 * Q.e2 + 0.7188145156129634
    if Q.girth2_top50 < 0.009262053166:
        z += 339.4588317871094 * Q.girth2_top50 - 3.1440857476804576
    if Q.log_sum_pt >= 6.935549248787:
        z += 17.320903778076172 * Q.log_sum_pt - 120.1299811863481
    if Q.lam2 < 0.002396991421 and Q.sum_pt_top3 < 462.25:
        z += -1.4567378759384155 * (0.002396991421 - Q.lam2) * (462.25 - Q.sum_pt_top3)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.n_real_top40 < 40.0:
        z += -0.001612405409105122 * (21.0 - Q.n_dr_0p2_0p4) * (40.0 - Q.n_real_top40)
    if Q.mass_top40 < 91.288535717504 and Q.z_dr_0p2_0p4 < 0.068491501734:
        z += -0.20740072429180145 * (91.288535717504 - Q.mass_top40) * (0.068491501734 - Q.z_dr_0p2_0p4)
    if Q.mass < 91.034691238403 and Q.z_dr_0p1_0p2 < 0.218764226139:
        z += 0.11178648471832275 * (91.034691238403 - Q.mass) * (0.218764226139 - Q.z_dr_0p1_0p2)
    if Q.n_dr_0p2_0p4 < 21.0 and Q.n_dr_0p1_0p2 < 21.0:
        z += -0.00294296070933342 * (21.0 - Q.n_dr_0p2_0p4) * (21.0 - Q.n_dr_0p1_0p2)
    if Q.z_dr_0p2_0p4 < 0.019523000158 and Q.z_0 < 0.358832142848:
        z += 125.23116302490234 * (0.019523000158 - Q.z_dr_0p2_0p4) * (0.358832142848 - Q.z_0)
    if Q.lam1 < 0.006189818106 and Q.z_top50_slots > 0.985099030959:
        z += 21879.263671875 * (0.006189818106 - Q.lam1) * (Q.z_top50_slots - 0.985099030959)
    if Q.girth2_top20 < 0.016559833876 and Q.z_top50_slots > 0.970443639316:
        z += -2757.6240234375 * (0.016559833876 - Q.girth2_top20) * (Q.z_top50_slots - 0.970443639316)
    return max(0.0, z)


def neuron_15(Q):
    z = -0.7327647805213928
    if Q.sum_pt < 986.05654296875:
        z += 0.0016059251502156258 * Q.sum_pt - 1.8068863397309531
    if 986.05654296875 <= Q.sum_pt < 1002.378515625:
        z += 0.013684212230145931 * Q.sum_pt - 13.716760342751149
    if Q.log_sum_pt < 7.017257672702:
        z += -2.582174777984619 * Q.log_sum_pt + 18.119785773070152
    if Q.girth2_top5 < 0.001501708498:
        z += 1854.8433837890625 * Q.girth2_top5 - 2.7854340718951107
    if Q.z_dr_0p2_0p4 < 0.019523000158:
        z += 24.474578857421875 * Q.z_dr_0p2_0p4 - 0.4778172069004307
    if Q.girth < 0.050483809784:
        z += 15.94879150390625 * Q.girth + 0.8183997743274405
    if 0.050483809784 <= Q.girth < 0.120745175332:
        z += -23.107372283935547 * Q.girth + 2.7901037178855947
    if Q.lam1 < 0.016493544356:
        z += 33.35480880737305 * Q.lam1 - 0.5501390185503068
    if Q.girth2_top50 < 0.013514311784:
        z += 202.03968811035156 * Q.girth2_top50 - 2.7304273378654087
    if Q.e2_sq < 0.006936724595:
        z += 561.0147094726562 * Q.e2_sq - 3.891604533355754
    if Q.z_dr_0p1_0p2 < 0.187280465662:
        z += -2.9014742374420166 * Q.z_dr_0p1_0p2 + 0.5433894462944372
    if Q.mass_top50 < 71.795159472175:
        z += 0.04089364781975746 * Q.mass_top50 - 2.9359659666184483
    if Q.lam2 < 0.001776308492:
        z += -392.3204650878906 * Q.lam2 + 0.6968821737210097
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.mass_top5 < 40.2:
        z += 0.2391204535961151 * (0.120343671367 - Q.z_dr_0p1_0p2) * (40.2 - Q.mass_top5)
    if Q.girth2_top5 < 0.007164202106 and Q.sum_pt_top40 < 984.70087890625:
        z += -0.8022595047950745 * (0.007164202106 - Q.girth2_top5) * (984.70087890625 - Q.sum_pt_top40)
    if Q.z_dr_0p1_0p2 < 0.120343671367 and Q.z_dr_0p2_0p4 < 0.037001823448:
        z += -148.87266540527344 * (0.120343671367 - Q.z_dr_0p1_0p2) * (0.037001823448 - Q.z_dr_0p2_0p4)
    if Q.girth2_top5 < 0.001501708498 and Q.z_dr_0p2_0p4 < 0.193744690716:
        z += 8896.3662109375 * (0.001501708498 - Q.girth2_top5) * (0.193744690716 - Q.z_dr_0p2_0p4)
    if Q.girth2_top20 < 0.002879910364 and Q.sum_pt > 1115.722741699219:
        z += -1.42798912525177 * (0.002879910364 - Q.girth2_top20) * (Q.sum_pt - 1115.722741699219)
    if Q.girth2_top50 < 0.013514311784 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += 2072.354736328125 * (0.013514311784 - Q.girth2_top50) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.e2_sq < 0.006936724595 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += 6540.14111328125 * (0.006936724595 - Q.e2_sq) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.girth < 0.120745175332 and Q.z_dr_0p2_0p4 < 0.129185591638:
        z += -70.74386596679688 * (0.120745175332 - Q.girth) * (0.129185591638 - Q.z_dr_0p2_0p4)
    if Q.z_dr_0p2_0p4 < 0.019523000158 and Q.z_dr_0p1_0p2 > 0.286492615938:
        z += -15373.4970703125 * (0.019523000158 - Q.z_dr_0p2_0p4) * (Q.z_dr_0p1_0p2 - 0.286492615938)
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
