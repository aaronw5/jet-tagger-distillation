"""JEDI-linear jet tagger, 64 particles, 3 features: the simpler version of the simplified formula, with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

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


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 1.97
            + 500.0 * max(0.0, 0.0021 - Q.girth2_top20)
            - 10.0 * max(0.0, Q.log_sum_pt - 6.8)
            + 37.3 * max(0.0, Q.log_sum_pt - 6.9)
            - 18.9 * max(0.0, Q.log_sum_pt - 7.0)
            + 0.0144 * max(0.0, 73.0 - Q.mass_top30)
            - 0.0759 * max(0.0, 43.0 - Q.n_particles)
            + 0.00512 * max(0.0, 760.0 - Q.sum_pt_top3)
            + 2.12 * max(0.0, Q.tau32 - 0.28)
            - 6.69 * max(0.0, 0.96 - Q.z_top20_slots)
            - 49.5 * max(0.0, Q.z_top50_slots - 0.96)
            + 0.733 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0)
            + 1.62 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots)
            - 0.0003 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4)
        ))
        - 0.75 * grid(3, max(0.0, -0.287
            - 7010.0 * max(0.0, 0.00011 - Q.girth2_top5)
            + 2090.0 * max(0.0, 0.00052 - Q.lam2)
            + 126.0 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
            + 0.000125 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        ))
        - 0.875 * grid(4, max(0.0, -0.532
            + 16.9 * max(0.0, Q.C2 - 0.1)
            + 0.209 * max(0.0, 7.7 - Q.D2)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0437 * max(0.0, 110.0 - Q.mass)
            - 0.00427 * max(0.0, 1100.0 - Q.sum_pt)
            + 2.89 * max(0.0, 0.61 - Q.tau32)
            - 0.627 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
            + 0.118 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
            - 0.0121 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        ))
        - 0.3125 * grid(5, max(0.0, 0.871
            - 7.36 * max(0.0, Q.log_sum_pt - 6.9)
            - 1240.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
            + 0.0156 * max(0.0, Q.mass_top30 - 49.0)
            - 0.144 * max(0.0, Q.mass_top50 - 150.0)
            + 19.8 * max(0.0, Q.max_dr - 0.44)
            - 9.11 * max(0.0, Q.z_top30_slots - 0.93)
            + 96.1 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
            + 0.75 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
            + 1.57 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        ))
        + 0.15625 * grid(6, max(0.0, 0.985
            + 0.157 * max(0.0, 91.2 - Q.mass)
            - 0.123 * max(0.0, 100.0 - Q.mass)
        ))
        + 0.03125 * grid(8, max(0.0, -0.0348
            + 16.9 * max(0.0, 0.096 - Q.girth)
            - 508.0 * max(0.0, 0.0015 - Q.lam2)
            + 0.0494 * max(0.0, Q.mass - 71.0)
            - 0.0472 * max(0.0, Q.mass - 120.0)
            - 0.0584 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
            + 0.11 * max(0.0, Q.n_particles - 58.0)
            + 0.0165 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.758 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
            - 0.00117 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
            + 0.0562 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        ))
        + 0.515625 * grid(9, max(0.0, -0.155
            + 584.0 * max(0.0, 0.0056 - Q.girth2_top40)
            + 0.00853 * max(0.0, Q.mass - 91.2)
            - 159.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
            + 0.00423 * max(0.0, 920.0 - Q.sum_pt_top40)
            + 2.97 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        ))
        - 0.015625 * grid(10, max(0.0, 3.08
            - 104.0 * max(0.0, 0.021 - Q.girth2_top5)
            - 0.159 * max(0.0, Q.mass - 150.0)
            + 0.0918 * max(0.0, 80.4 - Q.mass)
            - 0.0329 * max(0.0, 120.0 - Q.mass)
            + 0.0992 * max(0.0, Q.mass_top50 - 140.0)
            - 0.144 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.00809 * max(0.0, 960.0 - Q.sum_pt_top50)
            - 13.2 * max(0.0, Q.z_dr_0_0p05 - 0.77)
            + 15.2 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
            + 0.113 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
            + 0.0916 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        ))
        + 0.234375 * grid(12, max(0.0, 0.0851
            - 21.1 * max(0.0, Q.C2 - 0.055)
            + 0.0817 * max(0.0, 80.4 - Q.mass)
            + 0.0392 * max(0.0, Q.mass_top30 - 130.0)
            - 0.0673 * max(0.0, 63.0 - Q.mass_top50)
            + 17700.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 1.97
            + 500.0 * max(0.0, 0.0021 - Q.girth2_top20)
            - 10.0 * max(0.0, Q.log_sum_pt - 6.8)
            + 37.3 * max(0.0, Q.log_sum_pt - 6.9)
            - 18.9 * max(0.0, Q.log_sum_pt - 7.0)
            + 0.0144 * max(0.0, 73.0 - Q.mass_top30)
            - 0.0759 * max(0.0, 43.0 - Q.n_particles)
            + 0.00512 * max(0.0, 760.0 - Q.sum_pt_top3)
            + 2.12 * max(0.0, Q.tau32 - 0.28)
            - 6.69 * max(0.0, 0.96 - Q.z_top20_slots)
            - 49.5 * max(0.0, Q.z_top50_slots - 0.96)
            + 0.733 * max(0.0, Q.n_particles - 38.0) * max(0.0, 0.11 - Q.dr_0)
            + 1.62 * max(0.0, Q.n_particles - 40.0) * max(0.0, 0.98 - Q.z_top50_slots)
            - 0.0003 * max(0.0, 820.0 - Q.sum_pt_top2) * max(0.0, 7.7 - Q.n_dr_0p2_0p4)
        ))
        + 0.125 * grid(2, max(0.0, 0.381
            + 0.663 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094)
        ))
        - 1.0625 * grid(4, max(0.0, -0.532
            + 16.9 * max(0.0, Q.C2 - 0.1)
            + 0.209 * max(0.0, 7.7 - Q.D2)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0437 * max(0.0, 110.0 - Q.mass)
            - 0.00427 * max(0.0, 1100.0 - Q.sum_pt)
            + 2.89 * max(0.0, 0.61 - Q.tau32)
            - 0.627 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
            + 0.118 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
            - 0.0121 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        ))
        + 0.21875 * grid(6, max(0.0, 0.985
            + 0.157 * max(0.0, 91.2 - Q.mass)
            - 0.123 * max(0.0, 100.0 - Q.mass)
        ))
        + 0.015625 * grid(8, max(0.0, -0.0348
            + 16.9 * max(0.0, 0.096 - Q.girth)
            - 508.0 * max(0.0, 0.0015 - Q.lam2)
            + 0.0494 * max(0.0, Q.mass - 71.0)
            - 0.0472 * max(0.0, Q.mass - 120.0)
            - 0.0584 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
            + 0.11 * max(0.0, Q.n_particles - 58.0)
            + 0.0165 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.758 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
            - 0.00117 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
            + 0.0562 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        ))
        + 0.5625 * grid(9, max(0.0, -0.155
            + 584.0 * max(0.0, 0.0056 - Q.girth2_top40)
            + 0.00853 * max(0.0, Q.mass - 91.2)
            - 159.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
            + 0.00423 * max(0.0, 920.0 - Q.sum_pt_top40)
            + 2.97 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        ))
        - 0.015625 * grid(10, max(0.0, 3.08
            - 104.0 * max(0.0, 0.021 - Q.girth2_top5)
            - 0.159 * max(0.0, Q.mass - 150.0)
            + 0.0918 * max(0.0, 80.4 - Q.mass)
            - 0.0329 * max(0.0, 120.0 - Q.mass)
            + 0.0992 * max(0.0, Q.mass_top50 - 140.0)
            - 0.144 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.00809 * max(0.0, 960.0 - Q.sum_pt_top50)
            - 13.2 * max(0.0, Q.z_dr_0_0p05 - 0.77)
            + 15.2 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
            + 0.113 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
            + 0.0916 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        ))
        - 0.25 * grid(11, max(0.0, 0.241
            + 206.0 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4)
            - 0.0477 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2)
            + 0.0641 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
            + 0.0701 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow)
        ))
        + 0.34375 * grid(12, max(0.0, 0.0851
            - 21.1 * max(0.0, Q.C2 - 0.055)
            + 0.0817 * max(0.0, 80.4 - Q.mass)
            + 0.0392 * max(0.0, Q.mass_top30 - 130.0)
            - 0.0673 * max(0.0, 63.0 - Q.mass_top50)
            + 17700.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.32
            - 45.8 * max(0.0, 0.057 - Q.girth)
            - 195.0 * max(0.0, 0.0062 - Q.girth2_top20)
            - 316.0 * max(0.0, 0.0058 - Q.lam1)
            - 0.2 * max(0.0, Q.mass - 80.4)
            + 0.204 * max(0.0, Q.mass - 91.2)
            + 532.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
            - 0.052 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        ))
        + 0.4375 * grid(3, max(0.0, -0.287
            - 7010.0 * max(0.0, 0.00011 - Q.girth2_top5)
            + 2090.0 * max(0.0, 0.00052 - Q.lam2)
            + 126.0 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
            + 0.000125 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        ))
        + 0.34375 * grid(4, max(0.0, -0.532
            + 16.9 * max(0.0, Q.C2 - 0.1)
            + 0.209 * max(0.0, 7.7 - Q.D2)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0437 * max(0.0, 110.0 - Q.mass)
            - 0.00427 * max(0.0, 1100.0 - Q.sum_pt)
            + 2.89 * max(0.0, 0.61 - Q.tau32)
            - 0.627 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
            + 0.118 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
            - 0.0121 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        ))
        + 0.578125 * grid(5, max(0.0, 0.871
            - 7.36 * max(0.0, Q.log_sum_pt - 6.9)
            - 1240.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
            + 0.0156 * max(0.0, Q.mass_top30 - 49.0)
            - 0.144 * max(0.0, Q.mass_top50 - 150.0)
            + 19.8 * max(0.0, Q.max_dr - 0.44)
            - 9.11 * max(0.0, Q.z_top30_slots - 0.93)
            + 96.1 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
            + 0.75 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
            + 1.57 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        ))
        + 0.0625 * grid(6, max(0.0, 0.985
            + 0.157 * max(0.0, 91.2 - Q.mass)
            - 0.123 * max(0.0, 100.0 - Q.mass)
        ))
        - 0.625 * grid(7, max(0.0, -0.612
            + 45.3 * max(0.0, 0.026 - Q.e2)
            - 0.19 * max(0.0, 91.2 - Q.mass)
            + 0.0438 * max(0.0, 120.0 - Q.mass)
            + 0.123 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
            - 35.5 * max(0.0, 0.99 - Q.z_top50_slots)
            + 0.459 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
            - 1.6 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
            + 1.02 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
            - 0.589 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        ))
        - 0.875 * grid(8, max(0.0, -0.0348
            + 16.9 * max(0.0, 0.096 - Q.girth)
            - 508.0 * max(0.0, 0.0015 - Q.lam2)
            + 0.0494 * max(0.0, Q.mass - 71.0)
            - 0.0472 * max(0.0, Q.mass - 120.0)
            - 0.0584 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
            + 0.11 * max(0.0, Q.n_particles - 58.0)
            + 0.0165 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.758 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
            - 0.00117 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
            + 0.0562 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        ))
        - 0.21875 * grid(9, max(0.0, -0.155
            + 584.0 * max(0.0, 0.0056 - Q.girth2_top40)
            + 0.00853 * max(0.0, Q.mass - 91.2)
            - 159.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.026)
            + 0.00423 * max(0.0, 920.0 - Q.sum_pt_top40)
            + 2.97 * max(0.0, 0.0094 - Q.girth2_top40) * max(0.0, 1000.0 - Q.sum_pt)
        ))
        + 0.59375 * grid(11, max(0.0, 0.241
            + 206.0 * max(0.0, 0.0048 - Q.z_dr_0p2_0p4)
            - 0.0477 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.1 - Q.D2)
            + 0.0641 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
            + 0.0701 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.39 - Q.planar_flow)
        ))
        - 0.40625 * grid(12, max(0.0, 0.0851
            - 21.1 * max(0.0, Q.C2 - 0.055)
            + 0.0817 * max(0.0, 80.4 - Q.mass)
            + 0.0392 * max(0.0, Q.mass_top30 - 130.0)
            - 0.0673 * max(0.0, 63.0 - Q.mass_top50)
            + 17700.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        ))
        - 1.375 * grid(14, max(0.0, 0.0615
            - 0.138 * max(0.0, 91.2 - Q.mass)
            + 0.0332 * max(0.0, 130.0 - Q.mass)
            + 99.3 * max(0.0, 0.96 - Q.z_top50_slots)
        ))
        - 0.1875 * grid(15, max(0.0, 0.265

        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.32
            - 45.8 * max(0.0, 0.057 - Q.girth)
            - 195.0 * max(0.0, 0.0062 - Q.girth2_top20)
            - 316.0 * max(0.0, 0.0058 - Q.lam1)
            - 0.2 * max(0.0, Q.mass - 80.4)
            + 0.204 * max(0.0, Q.mass - 91.2)
            + 532.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
            - 0.052 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        ))
        - 0.375 * grid(2, max(0.0, 0.381
            + 0.663 * max(0.0, Q.sum_pt_top50 - 1100.0) * max(0.0, Q.C2 - 0.094)
        ))
        + 0.4375 * grid(3, max(0.0, -0.287
            - 7010.0 * max(0.0, 0.00011 - Q.girth2_top5)
            + 2090.0 * max(0.0, 0.00052 - Q.lam2)
            + 126.0 * max(0.0, 0.008 - Q.mass_over_sum_pt_sq)
            + 0.000125 * max(0.0, 45.0 - Q.n_particles) * max(0.0, Q.sum_pt_top40 - 830.0)
        ))
        + 0.6875 * grid(5, max(0.0, 0.871
            - 7.36 * max(0.0, Q.log_sum_pt - 6.9)
            - 1240.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
            + 0.0156 * max(0.0, Q.mass_top30 - 49.0)
            - 0.144 * max(0.0, Q.mass_top50 - 150.0)
            + 19.8 * max(0.0, Q.max_dr - 0.44)
            - 9.11 * max(0.0, Q.z_top30_slots - 0.93)
            + 96.1 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
            + 0.75 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
            + 1.57 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        ))
        - 0.96875 * grid(6, max(0.0, 0.985
            + 0.157 * max(0.0, 91.2 - Q.mass)
            - 0.123 * max(0.0, 100.0 - Q.mass)
        ))
        + 0.90625 * grid(7, max(0.0, -0.612
            + 45.3 * max(0.0, 0.026 - Q.e2)
            - 0.19 * max(0.0, 91.2 - Q.mass)
            + 0.0438 * max(0.0, 120.0 - Q.mass)
            + 0.123 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
            - 35.5 * max(0.0, 0.99 - Q.z_top50_slots)
            + 0.459 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
            - 1.6 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
            + 1.02 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
            - 0.589 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        ))
        - 0.9375 * grid(8, max(0.0, -0.0348
            + 16.9 * max(0.0, 0.096 - Q.girth)
            - 508.0 * max(0.0, 0.0015 - Q.lam2)
            + 0.0494 * max(0.0, Q.mass - 71.0)
            - 0.0472 * max(0.0, Q.mass - 120.0)
            - 0.0584 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
            + 0.11 * max(0.0, Q.n_particles - 58.0)
            + 0.0165 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.758 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
            - 0.00117 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
            + 0.0562 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        ))
        + 0.3125 * grid(12, max(0.0, 0.0851
            - 21.1 * max(0.0, Q.C2 - 0.055)
            + 0.0817 * max(0.0, 80.4 - Q.mass)
            + 0.0392 * max(0.0, Q.mass_top30 - 130.0)
            - 0.0673 * max(0.0, 63.0 - Q.mass_top50)
            + 17700.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        ))
        + 0.5625 * grid(15, max(0.0, 0.265

        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.32
            - 45.8 * max(0.0, 0.057 - Q.girth)
            - 195.0 * max(0.0, 0.0062 - Q.girth2_top20)
            - 316.0 * max(0.0, 0.0058 - Q.lam1)
            - 0.2 * max(0.0, Q.mass - 80.4)
            + 0.204 * max(0.0, Q.mass - 91.2)
            + 532.0 * max(0.0, 0.0079 - Q.mass_over_sum_pt_sq)
            - 0.052 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
        ))
        + 0.1875 * grid(4, max(0.0, -0.532
            + 16.9 * max(0.0, Q.C2 - 0.1)
            + 0.209 * max(0.0, 7.7 - Q.D2)
            - 0.139 * max(0.0, 80.4 - Q.mass)
            + 0.0437 * max(0.0, 110.0 - Q.mass)
            - 0.00427 * max(0.0, 1100.0 - Q.sum_pt)
            + 2.89 * max(0.0, 0.61 - Q.tau32)
            - 0.627 * max(0.0, 75.0 - Q.mass) * max(0.0, 0.33 - Q.max_dr)
            + 0.118 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
            - 0.0121 * max(0.0, 85.0 - Q.mass_top40) * max(0.0, 8.2 - Q.D2)
        ))
        - 0.46875 * grid(5, max(0.0, 0.871
            - 7.36 * max(0.0, Q.log_sum_pt - 6.9)
            - 1240.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.029)
            + 0.0156 * max(0.0, Q.mass_top30 - 49.0)
            - 0.144 * max(0.0, Q.mass_top50 - 150.0)
            + 19.8 * max(0.0, Q.max_dr - 0.44)
            - 9.11 * max(0.0, Q.z_top30_slots - 0.93)
            + 96.1 * max(0.0, 0.02 - Q.girth2_top30) * max(0.0, 0.86 - Q.tau32)
            + 0.75 * max(0.0, Q.mass_top50 - 160.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.55)
            + 1.57 * max(0.0, 66.0 - Q.n_particles) * max(0.0, 0.038 - Q.e2)
        ))
        - 0.28125 * grid(7, max(0.0, -0.612
            + 45.3 * max(0.0, 0.026 - Q.e2)
            - 0.19 * max(0.0, 91.2 - Q.mass)
            + 0.0438 * max(0.0, 120.0 - Q.mass)
            + 0.123 * max(0.0, 9.4 - Q.n_dr_0p2_0p4)
            - 35.5 * max(0.0, 0.99 - Q.z_top50_slots)
            + 0.459 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.6 - Q.D2)
            - 1.6 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.39 - Q.max_dr)
            + 1.02 * max(0.0, 100.0 - Q.mass) * max(0.0, 0.4 - Q.max_dr)
            - 0.589 * max(0.0, 97.0 - Q.mass_top50) * max(0.0, 1.6 - Q.D2)
        ))
        + 0.2109375 * grid(8, max(0.0, -0.0348
            + 16.9 * max(0.0, 0.096 - Q.girth)
            - 508.0 * max(0.0, 0.0015 - Q.lam2)
            + 0.0494 * max(0.0, Q.mass - 71.0)
            - 0.0472 * max(0.0, Q.mass - 120.0)
            - 0.0584 * max(0.0, 19.0 - Q.n_dr_0p2_0p4)
            + 0.11 * max(0.0, Q.n_particles - 58.0)
            + 0.0165 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.758 * max(0.0, Q.mass_over_sum_pt - 0.085) * max(0.0, 1100.0 - Q.sum_pt)
            - 0.00117 * max(0.0, 1000.0 - Q.sum_pt) * max(0.0, 44.0 - Q.n_real_top50)
            + 0.0562 * max(0.0, 930.0 - Q.sum_pt_top40) * max(0.0, 6.9 - Q.log_sum_pt)
        ))
        + 0.984375 * grid(10, max(0.0, 3.08
            - 104.0 * max(0.0, 0.021 - Q.girth2_top5)
            - 0.159 * max(0.0, Q.mass - 150.0)
            + 0.0918 * max(0.0, 80.4 - Q.mass)
            - 0.0329 * max(0.0, 120.0 - Q.mass)
            + 0.0992 * max(0.0, Q.mass_top50 - 140.0)
            - 0.144 * max(0.0, 11.0 - Q.n_dr_0p2_0p4)
            - 0.00809 * max(0.0, 960.0 - Q.sum_pt_top50)
            - 13.2 * max(0.0, Q.z_dr_0_0p05 - 0.77)
            + 15.2 * max(0.0, 0.088 - Q.z_dr_0p2_0p4)
            + 0.113 * max(0.0, 0.021 - Q.girth2_top5) * max(0.0, 720.0 - Q.sum_pt_top3)
            + 0.0916 * max(0.0, 120.0 - Q.mass) * max(0.0, 0.45 - Q.tau21)
        ))
        - 0.375 * grid(12, max(0.0, 0.0851
            - 21.1 * max(0.0, Q.C2 - 0.055)
            + 0.0817 * max(0.0, 80.4 - Q.mass)
            + 0.0392 * max(0.0, Q.mass_top30 - 130.0)
            - 0.0673 * max(0.0, 63.0 - Q.mass_top50)
            + 17700.0 * max(0.0, Q.LHA - 0.41) * max(0.0, 0.0044 - Q.lam2)
        ))
        - 0.90625 * grid(13, max(0.0, 3.1
            + 22.8 * max(0.0, 6.8 - Q.log_sum_pt)
            - 15.1 * max(0.0, 7.0 - Q.log_sum_pt)
            - 0.0775 * max(0.0, Q.mass - 140.0)
            + 0.0842 * max(0.0, Q.mass - 172.8)
            + 0.0178 * max(0.0, Q.mass_top10 - 67.0)
            - 0.0241 * max(0.0, 1000.0 - Q.sum_pt)
            + 0.015 * max(0.0, 1000.0 - Q.sum_pt_top40)
        ))
        - 0.375 * grid(15, max(0.0, 0.265

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
