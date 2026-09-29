"""JEDI-linear jet tagger, 64 particles, 3 features: the formula simplified by hand with the training data (main result), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

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


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 1.15
            + 1170.0 * max(0.0, 0.00073 - Q.girth2_top3)
            + 42.3 * max(0.0, Q.log_sum_pt - 6.91)
            - 22.4 * max(0.0, Q.log_sum_pt - 6.99)
            - 3.12 * max(0.0, 0.435 - Q.max_dr)
            - 0.0342 * max(0.0, 33.6 - Q.pt_9)
            + 0.00354 * max(0.0, 662.0 - Q.sum_pt_top2)
            + 0.00313 * max(0.0, 1070.0 - Q.sum_pt_top30)
            - 0.00905 * max(0.0, Q.sum_pt_top50 - 930.0)
            + 2.15 * max(0.0, Q.tau32 - 0.276)
            - 5.47 * max(0.0, 0.931 - Q.z_top20_slots)
            - 29.5 * max(0.0, Q.z_top50_slots - 0.96)
            - 80.0 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1)
            + 0.00323 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0)
            - 1.39 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6)
            + 0.55 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0)
            + 1.48 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots)
            - 0.000242 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4)
        ))
        - 0.75 * grid(3, max(0.0, -0.217
            - 6960.0 * max(0.0, 0.000115 - Q.girth2_top5)
            + 2040.0 * max(0.0, 0.000569 - Q.lam2)
            + 105.0 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
            - 6.07 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
            + 13.3 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
            + 0.000131 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
            - 186.0 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        ))
        - 0.875 * grid(4, max(0.0, 0.494
            + 18.1 * max(0.0, Q.C2 - 0.106)
            + 0.258 * max(0.0, 6.52 - Q.D2)
            - 22.7 * max(0.0, 0.063 - Q.girth)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0462 * max(0.0, 109.0 - Q.mass)
            - 0.0178 * max(0.0, Q.n_particles - 22.7)
            - 0.00597 * max(0.0, 1070.0 - Q.sum_pt)
            - 1.65 * max(0.0, 0.451 - Q.tau21)
            + 2.44 * max(0.0, 0.578 - Q.tau32)
            - 6.43 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
            - 0.652 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
            + 0.11 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
            - 0.0163 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        ))
        - 0.3125 * grid(5, max(0.0, -0.865
            + 30.2 * max(0.0, Q.log_sum_pt - 6.85)
            - 43.8 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.0278 * max(0.0, Q.mass - 63.2)
            - 448.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 0.0309 * max(0.0, Q.mass_top50 - 102.0)
            - 0.209 * max(0.0, Q.mass_top50 - 157.0)
            + 19.7 * max(0.0, Q.max_dr - 0.436)
            - 5.92 * max(0.0, Q.z_top30_slots - 0.943)
            + 77.3 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
            + 1.32 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
            + 1.47 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        ))
        + 0.15625 * grid(6, max(0.0, 1.22
            - 49.4 * max(0.0, Q.e2 - 0.0521)
            + 186.0 * max(0.0, Q.lam1 - 0.00717)
            + 0.0647 * max(0.0, 91.2 - Q.mass)
            - 0.0753 * max(0.0, 101.0 - Q.mass)
            - 25.3 * max(0.0, Q.mass_over_sum_pt - 0.0534)
            + 0.00816 * max(0.0, 92.5 - Q.mass_top10)
            + 0.0538 * max(0.0, 73.1 - Q.mass_top50)
            - 0.059 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
            + 1.76 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        ))
        + 0.03125 * grid(8, max(0.0, 0.721
            + 7.66 * max(0.0, Q.C2 - 0.0661)
            + 74.3 * max(0.0, 0.00815 - Q.girth2_top30)
            - 343.0 * max(0.0, 0.00143 - Q.lam2)
            + 0.105 * max(0.0, Q.mass - 91.2)
            - 0.109 * max(0.0, Q.mass - 104.0)
            - 0.0677 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
            + 0.0763 * max(0.0, Q.n_particles - 49.3)
            + 0.0142 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.687 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
            + 0.477 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
            - 3.12 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
            - 0.00105 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
            + 0.0688 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
            - 0.0644 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        ))
        + 0.515625 * grid(9, max(0.0, 0.0128
            - 104.0 * max(0.0, 0.0191 - Q.girth2_top15)
            - 0.0736 * max(0.0, Q.mass - 61.8)
            + 0.111 * max(0.0, Q.mass - 80.4)
            - 0.0483 * max(0.0, Q.mass - 142.0)
            + 0.0454 * max(0.0, 128.0 - Q.mass_top50)
            - 134.0 * max(0.0, Q.width - 0.026)
            + 2.2 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
            + 3.67 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
            + 160.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
            + 0.0589 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        ))
        - 0.015625 * grid(10, max(0.0, 4.71
            - 33.4 * max(0.0, 0.0611 - Q.e2)
            - 99.2 * max(0.0, 0.0254 - Q.girth2_top5)
            - 53.6 * max(0.0, Q.lam1 - 0.00248)
            - 0.0928 * max(0.0, Q.mass - 144.0)
            - 0.0556 * max(0.0, Q.mass - 162.0)
            + 0.0976 * max(0.0, 80.4 - Q.mass)
            - 0.0336 * max(0.0, 123.0 - Q.mass)
            + 0.0798 * max(0.0, Q.mass_top50 - 137.0)
            - 2.55 * max(0.0, 0.436 - Q.max_dr)
            - 0.107 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
            - 0.0083 * max(0.0, 954.0 - Q.sum_pt_top50)
            - 12.3 * max(0.0, Q.z_dr_0_0p05 - 0.769)
            + 12.6 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
            + 0.679 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
            + 0.128 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
            + 0.0788 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        ))
        + 0.234375 * grid(12, max(0.0, 0.085
            - 19.0 * max(0.0, Q.C2 - 0.0528)
            + 0.0807 * max(0.0, 80.4 - Q.mass)
            + 0.0686 * max(0.0, Q.mass_top30 - 136.0)
            - 0.066 * max(0.0, 63.0 - Q.mass_top50)
            + 12300.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
            - 1.26 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 1.15
            + 1170.0 * max(0.0, 0.00073 - Q.girth2_top3)
            + 42.3 * max(0.0, Q.log_sum_pt - 6.91)
            - 22.4 * max(0.0, Q.log_sum_pt - 6.99)
            - 3.12 * max(0.0, 0.435 - Q.max_dr)
            - 0.0342 * max(0.0, 33.6 - Q.pt_9)
            + 0.00354 * max(0.0, 662.0 - Q.sum_pt_top2)
            + 0.00313 * max(0.0, 1070.0 - Q.sum_pt_top30)
            - 0.00905 * max(0.0, Q.sum_pt_top50 - 930.0)
            + 2.15 * max(0.0, Q.tau32 - 0.276)
            - 5.47 * max(0.0, 0.931 - Q.z_top20_slots)
            - 29.5 * max(0.0, Q.z_top50_slots - 0.96)
            - 80.0 * max(0.0, 0.000994 - Q.girth2_top3) * max(0.0, 9.47 - Q.n_dr_0p05_0p1)
            + 0.00323 * max(0.0, 48.5 - Q.mass_top20) * max(0.0, Q.n_real_top40 - 27.0)
            - 1.39 * max(0.0, 6.42 - Q.n_dr_0p1_0p2) * max(0.0, 0.0751 - Q.dr_6)
            + 0.55 * max(0.0, Q.n_particles - 38.6) * max(0.0, 0.136 - Q.dr_0)
            + 1.48 * max(0.0, Q.n_particles - 37.9) * max(0.0, 0.985 - Q.z_top50_slots)
            - 0.000242 * max(0.0, 748.0 - Q.sum_pt_top2) * max(0.0, 7.95 - Q.n_dr_0p2_0p4)
        ))
        + 0.125 * grid(2, max(0.0, 0.38
            + 0.637 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        ))
        - 1.0625 * grid(4, max(0.0, 0.494
            + 18.1 * max(0.0, Q.C2 - 0.106)
            + 0.258 * max(0.0, 6.52 - Q.D2)
            - 22.7 * max(0.0, 0.063 - Q.girth)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0462 * max(0.0, 109.0 - Q.mass)
            - 0.0178 * max(0.0, Q.n_particles - 22.7)
            - 0.00597 * max(0.0, 1070.0 - Q.sum_pt)
            - 1.65 * max(0.0, 0.451 - Q.tau21)
            + 2.44 * max(0.0, 0.578 - Q.tau32)
            - 6.43 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
            - 0.652 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
            + 0.11 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
            - 0.0163 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        ))
        + 0.21875 * grid(6, max(0.0, 1.22
            - 49.4 * max(0.0, Q.e2 - 0.0521)
            + 186.0 * max(0.0, Q.lam1 - 0.00717)
            + 0.0647 * max(0.0, 91.2 - Q.mass)
            - 0.0753 * max(0.0, 101.0 - Q.mass)
            - 25.3 * max(0.0, Q.mass_over_sum_pt - 0.0534)
            + 0.00816 * max(0.0, 92.5 - Q.mass_top10)
            + 0.0538 * max(0.0, 73.1 - Q.mass_top50)
            - 0.059 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
            + 1.76 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        ))
        + 0.015625 * grid(8, max(0.0, 0.721
            + 7.66 * max(0.0, Q.C2 - 0.0661)
            + 74.3 * max(0.0, 0.00815 - Q.girth2_top30)
            - 343.0 * max(0.0, 0.00143 - Q.lam2)
            + 0.105 * max(0.0, Q.mass - 91.2)
            - 0.109 * max(0.0, Q.mass - 104.0)
            - 0.0677 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
            + 0.0763 * max(0.0, Q.n_particles - 49.3)
            + 0.0142 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.687 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
            + 0.477 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
            - 3.12 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
            - 0.00105 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
            + 0.0688 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
            - 0.0644 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        ))
        + 0.5625 * grid(9, max(0.0, 0.0128
            - 104.0 * max(0.0, 0.0191 - Q.girth2_top15)
            - 0.0736 * max(0.0, Q.mass - 61.8)
            + 0.111 * max(0.0, Q.mass - 80.4)
            - 0.0483 * max(0.0, Q.mass - 142.0)
            + 0.0454 * max(0.0, 128.0 - Q.mass_top50)
            - 134.0 * max(0.0, Q.width - 0.026)
            + 2.2 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
            + 3.67 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
            + 160.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
            + 0.0589 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        ))
        - 0.015625 * grid(10, max(0.0, 4.71
            - 33.4 * max(0.0, 0.0611 - Q.e2)
            - 99.2 * max(0.0, 0.0254 - Q.girth2_top5)
            - 53.6 * max(0.0, Q.lam1 - 0.00248)
            - 0.0928 * max(0.0, Q.mass - 144.0)
            - 0.0556 * max(0.0, Q.mass - 162.0)
            + 0.0976 * max(0.0, 80.4 - Q.mass)
            - 0.0336 * max(0.0, 123.0 - Q.mass)
            + 0.0798 * max(0.0, Q.mass_top50 - 137.0)
            - 2.55 * max(0.0, 0.436 - Q.max_dr)
            - 0.107 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
            - 0.0083 * max(0.0, 954.0 - Q.sum_pt_top50)
            - 12.3 * max(0.0, Q.z_dr_0_0p05 - 0.769)
            + 12.6 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
            + 0.679 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
            + 0.128 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
            + 0.0788 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        ))
        - 0.25 * grid(11, max(0.0, 0.196
            + 178.0 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4)
            - 0.0284 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2)
            + 0.103 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow)
            - 38.9 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
            + 0.00712 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        ))
        + 0.34375 * grid(12, max(0.0, 0.085
            - 19.0 * max(0.0, Q.C2 - 0.0528)
            + 0.0807 * max(0.0, 80.4 - Q.mass)
            + 0.0686 * max(0.0, Q.mass_top30 - 136.0)
            - 0.066 * max(0.0, 63.0 - Q.mass_top50)
            + 12300.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
            - 1.26 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.15
            - 36.5 * max(0.0, 0.0555 - Q.girth)
            - 167.0 * max(0.0, 0.00622 - Q.girth2_top20)
            - 305.0 * max(0.0, 0.00578 - Q.lam1)
            - 0.199 * max(0.0, Q.mass - 80.4)
            + 0.203 * max(0.0, Q.mass - 91.2)
            + 465.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
            - 0.0121 * max(0.0, 81.8 - Q.mass_top50)
            + 0.0298 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 4.39 * max(0.0, Q.z_dr_0_0p05 - 0.883)
            - 0.007 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
            + 0.0617 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
            - 0.0185 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
            - 0.0367 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        ))
        + 0.4375 * grid(3, max(0.0, -0.217
            - 6960.0 * max(0.0, 0.000115 - Q.girth2_top5)
            + 2040.0 * max(0.0, 0.000569 - Q.lam2)
            + 105.0 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
            - 6.07 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
            + 13.3 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
            + 0.000131 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
            - 186.0 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        ))
        + 0.34375 * grid(4, max(0.0, 0.494
            + 18.1 * max(0.0, Q.C2 - 0.106)
            + 0.258 * max(0.0, 6.52 - Q.D2)
            - 22.7 * max(0.0, 0.063 - Q.girth)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0462 * max(0.0, 109.0 - Q.mass)
            - 0.0178 * max(0.0, Q.n_particles - 22.7)
            - 0.00597 * max(0.0, 1070.0 - Q.sum_pt)
            - 1.65 * max(0.0, 0.451 - Q.tau21)
            + 2.44 * max(0.0, 0.578 - Q.tau32)
            - 6.43 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
            - 0.652 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
            + 0.11 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
            - 0.0163 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        ))
        + 0.578125 * grid(5, max(0.0, -0.865
            + 30.2 * max(0.0, Q.log_sum_pt - 6.85)
            - 43.8 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.0278 * max(0.0, Q.mass - 63.2)
            - 448.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 0.0309 * max(0.0, Q.mass_top50 - 102.0)
            - 0.209 * max(0.0, Q.mass_top50 - 157.0)
            + 19.7 * max(0.0, Q.max_dr - 0.436)
            - 5.92 * max(0.0, Q.z_top30_slots - 0.943)
            + 77.3 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
            + 1.32 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
            + 1.47 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        ))
        + 0.0625 * grid(6, max(0.0, 1.22
            - 49.4 * max(0.0, Q.e2 - 0.0521)
            + 186.0 * max(0.0, Q.lam1 - 0.00717)
            + 0.0647 * max(0.0, 91.2 - Q.mass)
            - 0.0753 * max(0.0, 101.0 - Q.mass)
            - 25.3 * max(0.0, Q.mass_over_sum_pt - 0.0534)
            + 0.00816 * max(0.0, 92.5 - Q.mass_top10)
            + 0.0538 * max(0.0, 73.1 - Q.mass_top50)
            - 0.059 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
            + 1.76 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        ))
        - 0.625 * grid(7, max(0.0, -0.726
            - 323.0 * max(0.0, 0.00812 - Q.girth2_top20)
            + 269.0 * max(0.0, 0.0129 - Q.girth2_top40)
            - 0.227 * max(0.0, 91.2 - Q.mass)
            + 0.0889 * max(0.0, 101.0 - Q.mass)
            + 0.0401 * max(0.0, 66.0 - Q.mass_top20)
            - 27.6 * max(0.0, 0.998 - Q.z_top50_slots)
            - 564.0 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
            + 715.0 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
            + 0.185 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
            + 0.307 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
            - 1.31 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
            + 0.819 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
            - 0.439 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        ))
        - 0.875 * grid(8, max(0.0, 0.721
            + 7.66 * max(0.0, Q.C2 - 0.0661)
            + 74.3 * max(0.0, 0.00815 - Q.girth2_top30)
            - 343.0 * max(0.0, 0.00143 - Q.lam2)
            + 0.105 * max(0.0, Q.mass - 91.2)
            - 0.109 * max(0.0, Q.mass - 104.0)
            - 0.0677 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
            + 0.0763 * max(0.0, Q.n_particles - 49.3)
            + 0.0142 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.687 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
            + 0.477 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
            - 3.12 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
            - 0.00105 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
            + 0.0688 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
            - 0.0644 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        ))
        - 0.21875 * grid(9, max(0.0, 0.0128
            - 104.0 * max(0.0, 0.0191 - Q.girth2_top15)
            - 0.0736 * max(0.0, Q.mass - 61.8)
            + 0.111 * max(0.0, Q.mass - 80.4)
            - 0.0483 * max(0.0, Q.mass - 142.0)
            + 0.0454 * max(0.0, 128.0 - Q.mass_top50)
            - 134.0 * max(0.0, Q.width - 0.026)
            + 2.2 * max(0.0, 0.0145 - Q.girth2_top15) * max(0.0, Q.n_particles - 35.1)
            + 3.67 * max(0.0, 0.00576 - Q.girth2_top40) * max(0.0, 997.0 - Q.sum_pt)
            + 160.0 * max(0.0, 6.85 - Q.log_sum_pt) * max(0.0, 0.0848 - Q.dr_7)
            + 0.0589 * max(0.0, 113.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.386)
        ))
        + 0.59375 * grid(11, max(0.0, 0.196
            + 178.0 * max(0.0, 0.00632 - Q.z_dr_0p2_0p4)
            - 0.0284 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.34 - Q.D2)
            + 0.103 * max(0.0, 103.0 - Q.mass) * max(0.0, 0.362 - Q.planar_flow)
            - 38.9 * max(0.0, 10.0 - Q.n_dr_0p2_0p4) * max(0.0, 0.00648 - Q.girth2)
            + 0.00712 * max(0.0, 10.6 - Q.n_dr_0p2_0p4) * max(0.0, 26.9 - Q.n_dr_0p1_0p2)
        ))
        - 0.40625 * grid(12, max(0.0, 0.085
            - 19.0 * max(0.0, Q.C2 - 0.0528)
            + 0.0807 * max(0.0, 80.4 - Q.mass)
            + 0.0686 * max(0.0, Q.mass_top30 - 136.0)
            - 0.066 * max(0.0, 63.0 - Q.mass_top50)
            + 12300.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
            - 1.26 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        ))
        - 1.375 * grid(14, max(0.0, 0.349
            - 0.14 * max(0.0, 91.2 - Q.mass)
            + 0.0376 * max(0.0, 131.0 - Q.mass)
            - 26.7 * max(0.0, 0.089 - Q.mass_over_sum_pt)
            - 2.12 * max(0.0, Q.max_dr - 0.237)
            + 88.4 * max(0.0, 0.961 - Q.z_top50_slots)
        ))
        - 0.1875 * grid(15, max(0.0, -0.204
            + 2.02 * max(0.0, 7.04 - Q.log_sum_pt)
            + 3.35 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.15
            - 36.5 * max(0.0, 0.0555 - Q.girth)
            - 167.0 * max(0.0, 0.00622 - Q.girth2_top20)
            - 305.0 * max(0.0, 0.00578 - Q.lam1)
            - 0.199 * max(0.0, Q.mass - 80.4)
            + 0.203 * max(0.0, Q.mass - 91.2)
            + 465.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
            - 0.0121 * max(0.0, 81.8 - Q.mass_top50)
            + 0.0298 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 4.39 * max(0.0, Q.z_dr_0_0p05 - 0.883)
            - 0.007 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
            + 0.0617 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
            - 0.0185 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
            - 0.0367 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        ))
        - 0.375 * grid(2, max(0.0, 0.38
            + 0.637 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        ))
        + 0.4375 * grid(3, max(0.0, -0.217
            - 6960.0 * max(0.0, 0.000115 - Q.girth2_top5)
            + 2040.0 * max(0.0, 0.000569 - Q.lam2)
            + 105.0 * max(0.0, 0.00813 - Q.mass_over_sum_pt_sq)
            - 6.07 * max(0.0, 5.2 - Q.n_dr_0p2_0p4) * max(0.0, 0.0493 - Q.dr_0)
            + 13.3 * max(0.0, 4.84 - Q.n_dr_0p2_0p4) * max(0.0, Q.z_top50_slots - 0.979)
            + 0.000131 * max(0.0, 46.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 822.0)
            - 186.0 * max(0.0, 0.419 - Q.tau21) * max(0.0, 0.0208 - Q.lam1)
        ))
        + 0.6875 * grid(5, max(0.0, -0.865
            + 30.2 * max(0.0, Q.log_sum_pt - 6.85)
            - 43.8 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.0278 * max(0.0, Q.mass - 63.2)
            - 448.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 0.0309 * max(0.0, Q.mass_top50 - 102.0)
            - 0.209 * max(0.0, Q.mass_top50 - 157.0)
            + 19.7 * max(0.0, Q.max_dr - 0.436)
            - 5.92 * max(0.0, Q.z_top30_slots - 0.943)
            + 77.3 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
            + 1.32 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
            + 1.47 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        ))
        - 0.96875 * grid(6, max(0.0, 1.22
            - 49.4 * max(0.0, Q.e2 - 0.0521)
            + 186.0 * max(0.0, Q.lam1 - 0.00717)
            + 0.0647 * max(0.0, 91.2 - Q.mass)
            - 0.0753 * max(0.0, 101.0 - Q.mass)
            - 25.3 * max(0.0, Q.mass_over_sum_pt - 0.0534)
            + 0.00816 * max(0.0, 92.5 - Q.mass_top10)
            + 0.0538 * max(0.0, 73.1 - Q.mass_top50)
            - 0.059 * max(0.0, 17.7 - Q.n_dr_0p2_0p4)
            + 1.76 * max(0.0, Q.mass_over_sum_pt - 0.0538) * max(0.0, 19.4 - Q.n_dr_0p2_0p4)
        ))
        + 0.90625 * grid(7, max(0.0, -0.726
            - 323.0 * max(0.0, 0.00812 - Q.girth2_top20)
            + 269.0 * max(0.0, 0.0129 - Q.girth2_top40)
            - 0.227 * max(0.0, 91.2 - Q.mass)
            + 0.0889 * max(0.0, 101.0 - Q.mass)
            + 0.0401 * max(0.0, 66.0 - Q.mass_top20)
            - 27.6 * max(0.0, 0.998 - Q.z_top50_slots)
            - 564.0 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
            + 715.0 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
            + 0.185 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
            + 0.307 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
            - 1.31 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
            + 0.819 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
            - 0.439 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        ))
        - 0.9375 * grid(8, max(0.0, 0.721
            + 7.66 * max(0.0, Q.C2 - 0.0661)
            + 74.3 * max(0.0, 0.00815 - Q.girth2_top30)
            - 343.0 * max(0.0, 0.00143 - Q.lam2)
            + 0.105 * max(0.0, Q.mass - 91.2)
            - 0.109 * max(0.0, Q.mass - 104.0)
            - 0.0677 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
            + 0.0763 * max(0.0, Q.n_particles - 49.3)
            + 0.0142 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.687 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
            + 0.477 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
            - 3.12 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
            - 0.00105 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
            + 0.0688 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
            - 0.0644 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        ))
        + 0.3125 * grid(12, max(0.0, 0.085
            - 19.0 * max(0.0, Q.C2 - 0.0528)
            + 0.0807 * max(0.0, 80.4 - Q.mass)
            + 0.0686 * max(0.0, Q.mass_top30 - 136.0)
            - 0.066 * max(0.0, 63.0 - Q.mass_top50)
            + 12300.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
            - 1.26 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        ))
        + 0.5625 * grid(15, max(0.0, -0.204
            + 2.02 * max(0.0, 7.04 - Q.log_sum_pt)
            + 3.35 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.15
            - 36.5 * max(0.0, 0.0555 - Q.girth)
            - 167.0 * max(0.0, 0.00622 - Q.girth2_top20)
            - 305.0 * max(0.0, 0.00578 - Q.lam1)
            - 0.199 * max(0.0, Q.mass - 80.4)
            + 0.203 * max(0.0, Q.mass - 91.2)
            + 465.0 * max(0.0, 0.00822 - Q.mass_over_sum_pt_sq)
            - 0.0121 * max(0.0, 81.8 - Q.mass_top50)
            + 0.0298 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 4.39 * max(0.0, Q.z_dr_0_0p05 - 0.883)
            - 0.007 * max(0.0, Q.mass - 80.4) * max(0.0, 6.91 - Q.n_dr_0p2_0p4)
            + 0.0617 * max(0.0, 72.0 - Q.mass_top50) * max(0.0, 0.259 - Q.z_dr_0p05_0p1)
            - 0.0185 * max(0.0, 1020.0 - Q.sum_pt) * max(0.0, Q.z_dr_0p1_0p2 - 0.0771)
            - 0.0367 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 0.212 - Q.z_dr_0p2_0p4)
        ))
        + 0.1875 * grid(4, max(0.0, 0.494
            + 18.1 * max(0.0, Q.C2 - 0.106)
            + 0.258 * max(0.0, 6.52 - Q.D2)
            - 22.7 * max(0.0, 0.063 - Q.girth)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0462 * max(0.0, 109.0 - Q.mass)
            - 0.0178 * max(0.0, Q.n_particles - 22.7)
            - 0.00597 * max(0.0, 1070.0 - Q.sum_pt)
            - 1.65 * max(0.0, 0.451 - Q.tau21)
            + 2.44 * max(0.0, 0.578 - Q.tau32)
            - 6.43 * max(0.0, 0.0697 - Q.z_dr_0p2_0p4)
            - 0.652 * max(0.0, 74.4 - Q.mass) * max(0.0, 0.35 - Q.max_dr)
            + 0.11 * max(0.0, 124.0 - Q.mass) * max(0.0, 0.401 - Q.max_dr)
            - 0.0163 * max(0.0, 84.9 - Q.mass_top40) * max(0.0, 6.27 - Q.D2)
        ))
        - 0.46875 * grid(5, max(0.0, -0.865
            + 30.2 * max(0.0, Q.log_sum_pt - 6.85)
            - 43.8 * max(0.0, Q.log_sum_pt - 6.91)
            + 0.0278 * max(0.0, Q.mass - 63.2)
            - 448.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 0.0309 * max(0.0, Q.mass_top50 - 102.0)
            - 0.209 * max(0.0, Q.mass_top50 - 157.0)
            + 19.7 * max(0.0, Q.max_dr - 0.436)
            - 5.92 * max(0.0, Q.z_top30_slots - 0.943)
            + 77.3 * max(0.0, 0.0226 - Q.girth2_top30) * max(0.0, 0.858 - Q.tau32)
            + 1.32 * max(0.0, Q.mass_top50 - 156.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.587)
            + 1.47 * max(0.0, 67.6 - Q.n_particles) * max(0.0, 0.0387 - Q.e2)
        ))
        - 0.28125 * grid(7, max(0.0, -0.726
            - 323.0 * max(0.0, 0.00812 - Q.girth2_top20)
            + 269.0 * max(0.0, 0.0129 - Q.girth2_top40)
            - 0.227 * max(0.0, 91.2 - Q.mass)
            + 0.0889 * max(0.0, 101.0 - Q.mass)
            + 0.0401 * max(0.0, 66.0 - Q.mass_top20)
            - 27.6 * max(0.0, 0.998 - Q.z_top50_slots)
            - 564.0 * max(0.0, 1.99 - Q.D2) * max(0.0, 0.00792 - Q.girth2_top50)
            + 715.0 * max(0.0, 2.0 - Q.D2) * max(0.0, 0.00623 - Q.girth2_top50)
            + 0.185 * max(0.0, 2.06 - Q.D2) * max(0.0, 8.39 - Q.n_dr_0p2_0p4)
            + 0.307 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.61 - Q.D2)
            - 1.31 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.391 - Q.max_dr)
            + 0.819 * max(0.0, 102.0 - Q.mass) * max(0.0, 0.393 - Q.max_dr)
            - 0.439 * max(0.0, 97.2 - Q.mass_top50) * max(0.0, 1.61 - Q.D2)
        ))
        + 0.2109375 * grid(8, max(0.0, 0.721
            + 7.66 * max(0.0, Q.C2 - 0.0661)
            + 74.3 * max(0.0, 0.00815 - Q.girth2_top30)
            - 343.0 * max(0.0, 0.00143 - Q.lam2)
            + 0.105 * max(0.0, Q.mass - 91.2)
            - 0.109 * max(0.0, Q.mass - 104.0)
            - 0.0677 * max(0.0, 20.2 - Q.n_dr_0p2_0p4)
            + 0.0763 * max(0.0, Q.n_particles - 49.3)
            + 0.0142 * max(0.0, 1010.0 - Q.sum_pt)
            + 0.687 * max(0.0, Q.girth2_top40 - 0.00145) * max(0.0, 932.0 - Q.sum_pt_top30)
            + 0.477 * max(0.0, Q.mass_over_sum_pt - 0.0762) * max(0.0, 1130.0 - Q.sum_pt)
            - 3.12 * max(0.0, Q.n_particles - 50.6) * max(0.0, Q.z_top50_slots - 0.972)
            - 0.00105 * max(0.0, 1010.0 - Q.sum_pt) * max(0.0, 43.2 - Q.n_real_top50)
            + 0.0688 * max(0.0, 958.0 - Q.sum_pt_top40) * max(0.0, 6.82 - Q.log_sum_pt)
            - 0.0644 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 0.415 - Q.max_dr)
        ))
        + 0.984375 * grid(10, max(0.0, 4.71
            - 33.4 * max(0.0, 0.0611 - Q.e2)
            - 99.2 * max(0.0, 0.0254 - Q.girth2_top5)
            - 53.6 * max(0.0, Q.lam1 - 0.00248)
            - 0.0928 * max(0.0, Q.mass - 144.0)
            - 0.0556 * max(0.0, Q.mass - 162.0)
            + 0.0976 * max(0.0, 80.4 - Q.mass)
            - 0.0336 * max(0.0, 123.0 - Q.mass)
            + 0.0798 * max(0.0, Q.mass_top50 - 137.0)
            - 2.55 * max(0.0, 0.436 - Q.max_dr)
            - 0.107 * max(0.0, 11.3 - Q.n_dr_0p2_0p4)
            - 0.0083 * max(0.0, 954.0 - Q.sum_pt_top50)
            - 12.3 * max(0.0, Q.z_dr_0_0p05 - 0.769)
            + 12.6 * max(0.0, 0.0879 - Q.z_dr_0p2_0p4)
            + 0.679 * max(0.0, 0.0612 - Q.dr_0) * max(0.0, Q.n_dr_0p2_0p4 - -0.94)
            + 0.128 * max(0.0, 0.0261 - Q.girth2_top5) * max(0.0, 668.0 - Q.sum_pt_top3)
            + 0.0788 * max(0.0, 122.0 - Q.mass) * max(0.0, 0.47 - Q.tau21)
        ))
        - 0.375 * grid(12, max(0.0, 0.085
            - 19.0 * max(0.0, Q.C2 - 0.0528)
            + 0.0807 * max(0.0, 80.4 - Q.mass)
            + 0.0686 * max(0.0, Q.mass_top30 - 136.0)
            - 0.066 * max(0.0, 63.0 - Q.mass_top50)
            + 12300.0 * max(0.0, Q.LHA - 0.405) * max(0.0, 0.00471 - Q.lam2)
            - 1.26 * max(0.0, Q.mass_top20 - 131.0) * max(0.0, Q.C2 - 0.065)
        ))
        - 0.90625 * grid(13, max(0.0, 3.03
            + 20.4 * max(0.0, 6.82 - Q.log_sum_pt)
            - 13.8 * max(0.0, 6.99 - Q.log_sum_pt)
            - 0.0922 * max(0.0, Q.mass - 144.0)
            + 0.103 * max(0.0, Q.mass - 172.8)
            + 0.0156 * max(0.0, Q.mass_top10 - 66.7)
            - 0.154 * max(0.0, 4.7 - Q.n_dr_0p2_0p4)
            - 0.0229 * max(0.0, 1020.0 - Q.sum_pt)
            + 0.0131 * max(0.0, 1020.0 - Q.sum_pt_top40)
        ))
        - 0.375 * grid(15, max(0.0, -0.204
            + 2.02 * max(0.0, 7.04 - Q.log_sum_pt)
            + 3.35 * max(0.0, 0.231 - Q.z_dr_0p1_0p2) * max(0.0, 1.09 - Q.planar_flow)
        ))
    )


def logits(Q):
    return [logit_g(Q), logit_q(Q), logit_W(Q), logit_Z(Q), logit_t(Q)]


def classify(pt, eta, phi):
    s = logits(quantities(pt, eta, phi))
    m = max(s)
    e = [math.exp(x - m) for x in s]
    p = [x / sum(e) for x in e]
    return CLASSES[s.index(m)], s, p


if __name__ == '__main__':
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3] + [0.0] * 56
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4] + [0.0] * 56
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36] + [0.0] * 56
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
