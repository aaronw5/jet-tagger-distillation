"""JEDI-linear jet tagger, 64 particles, 3 features: the formula with the fewest quantities (31) at the network's accuracy, with each class score (logit) written directly in terms of the jet quantities.

Input:  the 64 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

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


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-1.078125
        + 0.625 * grid(1, max(0.0, 0.703
            + 1.34 * max(0.0, 1.15 - Q.D2)
            + 5.81 * max(0.0, 0.398 - Q.LHA)
            - 92.7 * max(0.0, 0.0156 - Q.girth)
            + 1050.0 * max(0.0, 0.00133 - Q.girth2_top20)
            - 591.0 * max(0.0, 0.000885 - Q.lam2)
            - 8.65 * max(0.0, Q.log_sum_pt - 6.8)
            + 17.6 * max(0.0, Q.log_sum_pt - 6.89)
            + 32.8 * max(0.0, Q.log_sum_pt - 6.91)
            - 20.0 * max(0.0, Q.log_sum_pt - 6.98)
            + 0.0328 * max(0.0, 79.3 - Q.mass_top30)
            - 0.0189 * max(0.0, 114.0 - Q.mass_top50)
            - 2.73 * max(0.0, 0.429 - Q.max_dr)
            - 0.0998 * max(0.0, 7.51 - Q.n_dr_0p2_0p4)
            + 0.0364 * max(0.0, Q.n_particles - 26.7)
            + 0.00448 * max(0.0, 761.0 - Q.sum_pt_top3)
            + 0.0124 * max(0.0, Q.sum_pt_top40 - 1120.0)
            - 0.0104 * max(0.0, Q.sum_pt_top50 - 962.0)
            - 0.0119 * max(0.0, Q.sum_pt_top50 - 1150.0)
            + 2.04 * max(0.0, Q.tau32 - 0.326)
            - 6.58 * max(0.0, Q.z_dr_0_0p05 - 0.877)
            - 12.0 * max(0.0, 0.997 - Q.z_top30_slots)
            - 34.4 * max(0.0, Q.z_top50_slots - 0.958)
            + 0.358 * max(0.0, Q.n_particles - 34.8) * max(0.0, 0.131 - Q.dr_0)
            + 2.08 * max(0.0, Q.n_particles - 38.8) * max(0.0, 0.985 - Q.z_top50_slots)
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
        - 0.875 * grid(4, max(0.0, -0.431
            + 17.4 * max(0.0, Q.C2 - 0.107)
            + 0.224 * max(0.0, 6.81 - Q.D2)
            + 18.6 * max(0.0, Q.e2 - 0.0356)
            - 21.8 * max(0.0, 0.0616 - Q.girth)
            - 115.0 * max(0.0, 0.00777 - Q.girth2_top20)
            + 75.7 * max(0.0, 0.0231 - Q.girth2_top20)
            - 0.131 * max(0.0, 80.4 - Q.mass)
            + 0.0455 * max(0.0, 103.0 - Q.mass)
            - 71.7 * max(0.0, Q.mass_over_sum_pt - 0.0909)
            + 313.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00841)
            - 119.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0265)
            + 0.0257 * max(0.0, 18.0 - Q.n_dr_0p2_0p4)
            - 0.013 * max(0.0, Q.n_particles - 23.7)
            - 0.00554 * max(0.0, 1080.0 - Q.sum_pt)
            - 1.5 * max(0.0, 0.454 - Q.tau21)
            + 2.27 * max(0.0, 0.573 - Q.tau32)
            - 10.1 * max(0.0, 0.0698 - Q.z_dr_0p2_0p4)
            - 0.564 * max(0.0, 74.3 - Q.mass) * max(0.0, 0.36 - Q.max_dr)
            + 0.104 * max(0.0, 118.0 - Q.mass) * max(0.0, 0.411 - Q.max_dr)
            - 0.015 * max(0.0, 84.6 - Q.mass_top40) * max(0.0, 6.72 - Q.D2)
        ))
        - 0.3125 * grid(5, max(0.0, -0.546
            + 31.6 * max(0.0, Q.log_sum_pt - 6.85)
            - 44.1 * max(0.0, Q.log_sum_pt - 6.91)
            + 39.6 * max(0.0, Q.mass_over_sum_pt - 0.0786)
            - 39.7 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            - 450.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 103.0 * max(0.0, 0.00636 - Q.mass_over_sum_pt_sq)
            - 0.213 * max(0.0, Q.mass_top50 - 155.0)
            + 19.7 * max(0.0, Q.max_dr - 0.437)
            + 0.000694 * max(0.0, Q.sum_pt_top3 - 239.0)
            - 0.00484 * max(0.0, Q.sum_pt_top3 - 791.0)
            + 1.14 * max(0.0, 0.881 - Q.tau32)
            - 8.88 * max(0.0, Q.z_top30_slots - 0.936)
            + 1.23 * max(0.0, Q.mass_top50 - 155.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.581)
            + 1.72 * max(0.0, 65.0 - Q.n_particles) * max(0.0, 0.0378 - Q.e2)
        ))
        + 0.15625 * grid(6, max(0.0, 2.55
            - 51.9 * max(0.0, Q.e2 - 0.0526)
            + 23.0 * max(0.0, 0.0477 - Q.e2)
            + 228.0 * max(0.0, Q.lam1 - 0.00793)
            + 6.95 * max(0.0, 6.82 - Q.log_sum_pt)
            + 0.0471 * max(0.0, 91.2 - Q.mass)
            - 0.0514 * max(0.0, 110.0 - Q.mass)
            - 40.3 * max(0.0, Q.mass_over_sum_pt - 0.0491)
            - 166.0 * max(0.0, 0.0101 - Q.mass_over_sum_pt_sq)
            + 0.0531 * max(0.0, 70.5 - Q.mass_top50)
            - 0.0589 * max(0.0, 18.5 - Q.n_dr_0p2_0p4)
            + 1.77 * max(0.0, Q.mass_over_sum_pt - 0.0497) * max(0.0, 18.6 - Q.n_dr_0p2_0p4)
        ))
        + 0.03125 * grid(8, max(0.0, 5.9
            + 9.62 * max(0.0, Q.C2 - 0.0695)
            + 0.411 * max(0.0, 1.8 - Q.D2)
            - 43.5 * max(0.0, Q.e2 - 0.0203)
            + 243.0 * max(0.0, Q.girth2_top20 - 0.00792)
            - 487.0 * max(0.0, 0.00148 - Q.lam2)
            + 5.7 * max(0.0, Q.log_sum_pt - 6.94)
            - 0.0718 * max(0.0, Q.mass - 65.2)
            + 0.0891 * max(0.0, Q.mass - 86.5)
            - 0.0733 * max(0.0, Q.mass - 105.0)
            + 0.118 * max(0.0, 64.5 - Q.mass)
            - 146.0 * max(0.0, 0.161 - Q.mass_over_sum_pt)
            + 441.0 * max(0.0, 0.0255 - Q.mass_over_sum_pt_sq)
            - 0.0592 * max(0.0, 20.6 - Q.n_dr_0p2_0p4)
            + 0.074 * max(0.0, Q.n_particles - 50.7)
            - 0.0196 * max(0.0, 46.1 - Q.n_particles)
            + 0.0118 * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.00777 * max(0.0, 1040.0 - Q.sum_pt_top40)
            + 0.635 * max(0.0, Q.mass_over_sum_pt - 0.0757) * max(0.0, 1090.0 - Q.sum_pt)
            - 2.79 * max(0.0, Q.n_particles - 51.3) * max(0.0, Q.z_top50_slots - 0.97)
            + 0.0676 * max(0.0, 883.0 - Q.sum_pt_top40) * max(0.0, 6.89 - Q.log_sum_pt)
            - 0.0926 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, 0.39 - Q.max_dr)
        ))
        + 0.515625 * grid(9, max(0.0, 0.389
            - 29.4 * max(0.0, 0.0417 - Q.girth)
            + 0.0366 * max(0.0, Q.mass - 84.5)
            - 0.0376 * max(0.0, Q.mass - 151.0)
            - 59.5 * max(0.0, Q.mass_over_sum_pt - 0.0588)
            + 216.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00745)
            - 315.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0257)
            + 0.0602 * max(0.0, 80.4 - Q.mass_top40)
            + 0.00843 * max(0.0, 1010.0 - Q.sum_pt_top40)
            - 42.5 * max(0.0, 0.979 - Q.z_top50_slots)
            + 0.058 * max(0.0, 115.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.375)
            + 0.769 * max(0.0, 931.0 - Q.sum_pt_top40) * max(0.0, Q.z_top30_slots - 0.977)
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
        + 0.234375 * grid(12, max(0.0, 0.0817
            - 21.6 * max(0.0, Q.C2 - 0.0556)
            - 5.35 * max(0.0, 6.85 - Q.log_sum_pt)
            + 0.0839 * max(0.0, 80.4 - Q.mass)
            + 0.0361 * max(0.0, Q.mass_top30 - 128.0)
            - 0.0701 * max(0.0, 62.5 - Q.mass_top50)
            + 15700.0 * max(0.0, Q.LHA - 0.406) * max(0.0, 0.0046 - Q.lam2)
        ))
    )


def logit_q(Q):
    return (1.359375
        - 0.1875 * grid(1, max(0.0, 0.703
            + 1.34 * max(0.0, 1.15 - Q.D2)
            + 5.81 * max(0.0, 0.398 - Q.LHA)
            - 92.7 * max(0.0, 0.0156 - Q.girth)
            + 1050.0 * max(0.0, 0.00133 - Q.girth2_top20)
            - 591.0 * max(0.0, 0.000885 - Q.lam2)
            - 8.65 * max(0.0, Q.log_sum_pt - 6.8)
            + 17.6 * max(0.0, Q.log_sum_pt - 6.89)
            + 32.8 * max(0.0, Q.log_sum_pt - 6.91)
            - 20.0 * max(0.0, Q.log_sum_pt - 6.98)
            + 0.0328 * max(0.0, 79.3 - Q.mass_top30)
            - 0.0189 * max(0.0, 114.0 - Q.mass_top50)
            - 2.73 * max(0.0, 0.429 - Q.max_dr)
            - 0.0998 * max(0.0, 7.51 - Q.n_dr_0p2_0p4)
            + 0.0364 * max(0.0, Q.n_particles - 26.7)
            + 0.00448 * max(0.0, 761.0 - Q.sum_pt_top3)
            + 0.0124 * max(0.0, Q.sum_pt_top40 - 1120.0)
            - 0.0104 * max(0.0, Q.sum_pt_top50 - 962.0)
            - 0.0119 * max(0.0, Q.sum_pt_top50 - 1150.0)
            + 2.04 * max(0.0, Q.tau32 - 0.326)
            - 6.58 * max(0.0, Q.z_dr_0_0p05 - 0.877)
            - 12.0 * max(0.0, 0.997 - Q.z_top30_slots)
            - 34.4 * max(0.0, Q.z_top50_slots - 0.958)
            + 0.358 * max(0.0, Q.n_particles - 34.8) * max(0.0, 0.131 - Q.dr_0)
            + 2.08 * max(0.0, Q.n_particles - 38.8) * max(0.0, 0.985 - Q.z_top50_slots)
        ))
        + 0.125 * grid(2, max(0.0, 0.38
            + 0.637 * max(0.0, Q.sum_pt_top50 - 1090.0) * max(0.0, Q.C2 - 0.0936)
        ))
        - 1.0625 * grid(4, max(0.0, -0.431
            + 17.4 * max(0.0, Q.C2 - 0.107)
            + 0.224 * max(0.0, 6.81 - Q.D2)
            + 18.6 * max(0.0, Q.e2 - 0.0356)
            - 21.8 * max(0.0, 0.0616 - Q.girth)
            - 115.0 * max(0.0, 0.00777 - Q.girth2_top20)
            + 75.7 * max(0.0, 0.0231 - Q.girth2_top20)
            - 0.131 * max(0.0, 80.4 - Q.mass)
            + 0.0455 * max(0.0, 103.0 - Q.mass)
            - 71.7 * max(0.0, Q.mass_over_sum_pt - 0.0909)
            + 313.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00841)
            - 119.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0265)
            + 0.0257 * max(0.0, 18.0 - Q.n_dr_0p2_0p4)
            - 0.013 * max(0.0, Q.n_particles - 23.7)
            - 0.00554 * max(0.0, 1080.0 - Q.sum_pt)
            - 1.5 * max(0.0, 0.454 - Q.tau21)
            + 2.27 * max(0.0, 0.573 - Q.tau32)
            - 10.1 * max(0.0, 0.0698 - Q.z_dr_0p2_0p4)
            - 0.564 * max(0.0, 74.3 - Q.mass) * max(0.0, 0.36 - Q.max_dr)
            + 0.104 * max(0.0, 118.0 - Q.mass) * max(0.0, 0.411 - Q.max_dr)
            - 0.015 * max(0.0, 84.6 - Q.mass_top40) * max(0.0, 6.72 - Q.D2)
        ))
        + 0.21875 * grid(6, max(0.0, 2.55
            - 51.9 * max(0.0, Q.e2 - 0.0526)
            + 23.0 * max(0.0, 0.0477 - Q.e2)
            + 228.0 * max(0.0, Q.lam1 - 0.00793)
            + 6.95 * max(0.0, 6.82 - Q.log_sum_pt)
            + 0.0471 * max(0.0, 91.2 - Q.mass)
            - 0.0514 * max(0.0, 110.0 - Q.mass)
            - 40.3 * max(0.0, Q.mass_over_sum_pt - 0.0491)
            - 166.0 * max(0.0, 0.0101 - Q.mass_over_sum_pt_sq)
            + 0.0531 * max(0.0, 70.5 - Q.mass_top50)
            - 0.0589 * max(0.0, 18.5 - Q.n_dr_0p2_0p4)
            + 1.77 * max(0.0, Q.mass_over_sum_pt - 0.0497) * max(0.0, 18.6 - Q.n_dr_0p2_0p4)
        ))
        + 0.015625 * grid(8, max(0.0, 5.9
            + 9.62 * max(0.0, Q.C2 - 0.0695)
            + 0.411 * max(0.0, 1.8 - Q.D2)
            - 43.5 * max(0.0, Q.e2 - 0.0203)
            + 243.0 * max(0.0, Q.girth2_top20 - 0.00792)
            - 487.0 * max(0.0, 0.00148 - Q.lam2)
            + 5.7 * max(0.0, Q.log_sum_pt - 6.94)
            - 0.0718 * max(0.0, Q.mass - 65.2)
            + 0.0891 * max(0.0, Q.mass - 86.5)
            - 0.0733 * max(0.0, Q.mass - 105.0)
            + 0.118 * max(0.0, 64.5 - Q.mass)
            - 146.0 * max(0.0, 0.161 - Q.mass_over_sum_pt)
            + 441.0 * max(0.0, 0.0255 - Q.mass_over_sum_pt_sq)
            - 0.0592 * max(0.0, 20.6 - Q.n_dr_0p2_0p4)
            + 0.074 * max(0.0, Q.n_particles - 50.7)
            - 0.0196 * max(0.0, 46.1 - Q.n_particles)
            + 0.0118 * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.00777 * max(0.0, 1040.0 - Q.sum_pt_top40)
            + 0.635 * max(0.0, Q.mass_over_sum_pt - 0.0757) * max(0.0, 1090.0 - Q.sum_pt)
            - 2.79 * max(0.0, Q.n_particles - 51.3) * max(0.0, Q.z_top50_slots - 0.97)
            + 0.0676 * max(0.0, 883.0 - Q.sum_pt_top40) * max(0.0, 6.89 - Q.log_sum_pt)
            - 0.0926 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, 0.39 - Q.max_dr)
        ))
        + 0.5625 * grid(9, max(0.0, 0.389
            - 29.4 * max(0.0, 0.0417 - Q.girth)
            + 0.0366 * max(0.0, Q.mass - 84.5)
            - 0.0376 * max(0.0, Q.mass - 151.0)
            - 59.5 * max(0.0, Q.mass_over_sum_pt - 0.0588)
            + 216.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00745)
            - 315.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0257)
            + 0.0602 * max(0.0, 80.4 - Q.mass_top40)
            + 0.00843 * max(0.0, 1010.0 - Q.sum_pt_top40)
            - 42.5 * max(0.0, 0.979 - Q.z_top50_slots)
            + 0.058 * max(0.0, 115.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.375)
            + 0.769 * max(0.0, 931.0 - Q.sum_pt_top40) * max(0.0, Q.z_top30_slots - 0.977)
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
        - 0.25 * grid(11, max(0.0, 0.0318
            + 73.9 * max(0.0, 0.0234 - Q.e2)
            + 748.0 * max(0.0, 0.000827 - Q.lam2)
            - 41.7 * max(0.0, 0.0797 - Q.mass_over_sum_pt)
            + 0.0626 * max(0.0, 14.3 - Q.n_dr_0p2_0p4)
            + 183.0 * max(0.0, 0.0049 - Q.z_dr_0p2_0p4)
            - 0.0276 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.24 - Q.D2)
            + 0.091 * max(0.0, 98.1 - Q.mass) * max(0.0, 1.27 - Q.D2)
        ))
        + 0.34375 * grid(12, max(0.0, 0.0817
            - 21.6 * max(0.0, Q.C2 - 0.0556)
            - 5.35 * max(0.0, 6.85 - Q.log_sum_pt)
            + 0.0839 * max(0.0, 80.4 - Q.mass)
            + 0.0361 * max(0.0, Q.mass_top30 - 128.0)
            - 0.0701 * max(0.0, 62.5 - Q.mass_top50)
            + 15700.0 * max(0.0, Q.LHA - 0.406) * max(0.0, 0.0046 - Q.lam2)
        ))
    )


def logit_W(Q):
    return (0.09375
        + 0.75 * grid(0, max(0.0, 1.3
            - 27.2 * max(0.0, 0.0562 - Q.girth)
            - 162.0 * max(0.0, 0.00611 - Q.girth2_top20)
            - 197.0 * max(0.0, 0.00583 - Q.lam1)
            - 0.159 * max(0.0, Q.mass - 80.4)
            + 0.154 * max(0.0, Q.mass - 91.2)
            - 66.9 * max(0.0, Q.mass_over_sum_pt - 0.0782)
            + 79.9 * max(0.0, Q.mass_over_sum_pt - 0.0857)
            + 340.0 * max(0.0, 0.00784 - Q.mass_over_sum_pt_sq)
            - 0.0156 * max(0.0, 80.8 - Q.mass_top50)
            + 0.0396 * max(0.0, 10.8 - Q.n_dr_0p2_0p4)
            - 0.00481 * max(0.0, 1010.0 - Q.sum_pt)
            - 2.97 * max(0.0, Q.z_dr_0_0p05 - 0.871)
            + 0.0511 * max(0.0, 6.99 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top40 - 936.0)
            - 0.00923 * max(0.0, Q.mass - 80.4) * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 0.0451 * max(0.0, 70.3 - Q.mass_top50) * max(0.0, 0.231 - Q.z_dr_0p05_0p1)
            - 2.13 * max(0.0, 9.75 - Q.n_dr_0p2_0p4) * max(0.0, 0.98 - Q.z_top50_slots)
            - 64.4 * max(0.0, Q.z_top30_slots - 0.913) * max(0.0, Q.C2 - 0.0639)
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
        + 0.34375 * grid(4, max(0.0, -0.431
            + 17.4 * max(0.0, Q.C2 - 0.107)
            + 0.224 * max(0.0, 6.81 - Q.D2)
            + 18.6 * max(0.0, Q.e2 - 0.0356)
            - 21.8 * max(0.0, 0.0616 - Q.girth)
            - 115.0 * max(0.0, 0.00777 - Q.girth2_top20)
            + 75.7 * max(0.0, 0.0231 - Q.girth2_top20)
            - 0.131 * max(0.0, 80.4 - Q.mass)
            + 0.0455 * max(0.0, 103.0 - Q.mass)
            - 71.7 * max(0.0, Q.mass_over_sum_pt - 0.0909)
            + 313.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00841)
            - 119.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0265)
            + 0.0257 * max(0.0, 18.0 - Q.n_dr_0p2_0p4)
            - 0.013 * max(0.0, Q.n_particles - 23.7)
            - 0.00554 * max(0.0, 1080.0 - Q.sum_pt)
            - 1.5 * max(0.0, 0.454 - Q.tau21)
            + 2.27 * max(0.0, 0.573 - Q.tau32)
            - 10.1 * max(0.0, 0.0698 - Q.z_dr_0p2_0p4)
            - 0.564 * max(0.0, 74.3 - Q.mass) * max(0.0, 0.36 - Q.max_dr)
            + 0.104 * max(0.0, 118.0 - Q.mass) * max(0.0, 0.411 - Q.max_dr)
            - 0.015 * max(0.0, 84.6 - Q.mass_top40) * max(0.0, 6.72 - Q.D2)
        ))
        + 0.578125 * grid(5, max(0.0, -0.546
            + 31.6 * max(0.0, Q.log_sum_pt - 6.85)
            - 44.1 * max(0.0, Q.log_sum_pt - 6.91)
            + 39.6 * max(0.0, Q.mass_over_sum_pt - 0.0786)
            - 39.7 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            - 450.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 103.0 * max(0.0, 0.00636 - Q.mass_over_sum_pt_sq)
            - 0.213 * max(0.0, Q.mass_top50 - 155.0)
            + 19.7 * max(0.0, Q.max_dr - 0.437)
            + 0.000694 * max(0.0, Q.sum_pt_top3 - 239.0)
            - 0.00484 * max(0.0, Q.sum_pt_top3 - 791.0)
            + 1.14 * max(0.0, 0.881 - Q.tau32)
            - 8.88 * max(0.0, Q.z_top30_slots - 0.936)
            + 1.23 * max(0.0, Q.mass_top50 - 155.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.581)
            + 1.72 * max(0.0, 65.0 - Q.n_particles) * max(0.0, 0.0378 - Q.e2)
        ))
        + 0.0625 * grid(6, max(0.0, 2.55
            - 51.9 * max(0.0, Q.e2 - 0.0526)
            + 23.0 * max(0.0, 0.0477 - Q.e2)
            + 228.0 * max(0.0, Q.lam1 - 0.00793)
            + 6.95 * max(0.0, 6.82 - Q.log_sum_pt)
            + 0.0471 * max(0.0, 91.2 - Q.mass)
            - 0.0514 * max(0.0, 110.0 - Q.mass)
            - 40.3 * max(0.0, Q.mass_over_sum_pt - 0.0491)
            - 166.0 * max(0.0, 0.0101 - Q.mass_over_sum_pt_sq)
            + 0.0531 * max(0.0, 70.5 - Q.mass_top50)
            - 0.0589 * max(0.0, 18.5 - Q.n_dr_0p2_0p4)
            + 1.77 * max(0.0, Q.mass_over_sum_pt - 0.0497) * max(0.0, 18.6 - Q.n_dr_0p2_0p4)
        ))
        - 0.625 * grid(7, max(0.0, -0.571
            + 66.5 * max(0.0, 0.0286 - Q.e2)
            - 19.4 * max(0.0, 0.0884 - Q.girth)
            + 534.0 * max(0.0, 0.0063 - Q.girth2_top20)
            - 570.0 * max(0.0, 0.00795 - Q.girth2_top20)
            - 0.19 * max(0.0, 91.2 - Q.mass)
            + 0.0645 * max(0.0, 120.0 - Q.mass)
            + 100.0 * max(0.0, 0.0799 - Q.mass_over_sum_pt)
            - 126.0 * max(0.0, 0.0906 - Q.mass_over_sum_pt)
            - 514.0 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq)
            + 788.0 * max(0.0, 0.0095 - Q.mass_over_sum_pt_sq)
            + 0.0775 * max(0.0, 13.0 - Q.n_dr_0p2_0p4)
            + 99.2 * max(0.0, 0.0059 - Q.z_dr_0p2_0p4)
            - 10.2 * max(0.0, 0.0901 - Q.z_dr_0p2_0p4)
            - 36.2 * max(0.0, 0.991 - Q.z_top50_slots)
            + 0.0834 * max(0.0, 1.8 - Q.D2) * max(0.0, 9.09 - Q.n_dr_0p2_0p4)
            + 0.273 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.64 - Q.D2)
            - 1.33 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.392 - Q.max_dr)
            + 0.814 * max(0.0, 101.0 - Q.mass) * max(0.0, 0.397 - Q.max_dr)
            - 0.149 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.422)
            + 0.183 * max(0.0, 75.6 - Q.mass_top30) * max(0.0, 1.61 - Q.D2)
            - 0.388 * max(0.0, 97.4 - Q.mass_top50) * max(0.0, 1.65 - Q.D2)
        ))
        - 0.875 * grid(8, max(0.0, 5.9
            + 9.62 * max(0.0, Q.C2 - 0.0695)
            + 0.411 * max(0.0, 1.8 - Q.D2)
            - 43.5 * max(0.0, Q.e2 - 0.0203)
            + 243.0 * max(0.0, Q.girth2_top20 - 0.00792)
            - 487.0 * max(0.0, 0.00148 - Q.lam2)
            + 5.7 * max(0.0, Q.log_sum_pt - 6.94)
            - 0.0718 * max(0.0, Q.mass - 65.2)
            + 0.0891 * max(0.0, Q.mass - 86.5)
            - 0.0733 * max(0.0, Q.mass - 105.0)
            + 0.118 * max(0.0, 64.5 - Q.mass)
            - 146.0 * max(0.0, 0.161 - Q.mass_over_sum_pt)
            + 441.0 * max(0.0, 0.0255 - Q.mass_over_sum_pt_sq)
            - 0.0592 * max(0.0, 20.6 - Q.n_dr_0p2_0p4)
            + 0.074 * max(0.0, Q.n_particles - 50.7)
            - 0.0196 * max(0.0, 46.1 - Q.n_particles)
            + 0.0118 * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.00777 * max(0.0, 1040.0 - Q.sum_pt_top40)
            + 0.635 * max(0.0, Q.mass_over_sum_pt - 0.0757) * max(0.0, 1090.0 - Q.sum_pt)
            - 2.79 * max(0.0, Q.n_particles - 51.3) * max(0.0, Q.z_top50_slots - 0.97)
            + 0.0676 * max(0.0, 883.0 - Q.sum_pt_top40) * max(0.0, 6.89 - Q.log_sum_pt)
            - 0.0926 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, 0.39 - Q.max_dr)
        ))
        - 0.21875 * grid(9, max(0.0, 0.389
            - 29.4 * max(0.0, 0.0417 - Q.girth)
            + 0.0366 * max(0.0, Q.mass - 84.5)
            - 0.0376 * max(0.0, Q.mass - 151.0)
            - 59.5 * max(0.0, Q.mass_over_sum_pt - 0.0588)
            + 216.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00745)
            - 315.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0257)
            + 0.0602 * max(0.0, 80.4 - Q.mass_top40)
            + 0.00843 * max(0.0, 1010.0 - Q.sum_pt_top40)
            - 42.5 * max(0.0, 0.979 - Q.z_top50_slots)
            + 0.058 * max(0.0, 115.0 - Q.mass_top40) * max(0.0, Q.max_dr - 0.375)
            + 0.769 * max(0.0, 931.0 - Q.sum_pt_top40) * max(0.0, Q.z_top30_slots - 0.977)
        ))
        + 0.59375 * grid(11, max(0.0, 0.0318
            + 73.9 * max(0.0, 0.0234 - Q.e2)
            + 748.0 * max(0.0, 0.000827 - Q.lam2)
            - 41.7 * max(0.0, 0.0797 - Q.mass_over_sum_pt)
            + 0.0626 * max(0.0, 14.3 - Q.n_dr_0p2_0p4)
            + 183.0 * max(0.0, 0.0049 - Q.z_dr_0p2_0p4)
            - 0.0276 * max(0.0, 80.4 - Q.mass) * max(0.0, 3.24 - Q.D2)
            + 0.091 * max(0.0, 98.1 - Q.mass) * max(0.0, 1.27 - Q.D2)
        ))
        - 0.40625 * grid(12, max(0.0, 0.0817
            - 21.6 * max(0.0, Q.C2 - 0.0556)
            - 5.35 * max(0.0, 6.85 - Q.log_sum_pt)
            + 0.0839 * max(0.0, 80.4 - Q.mass)
            + 0.0361 * max(0.0, Q.mass_top30 - 128.0)
            - 0.0701 * max(0.0, 62.5 - Q.mass_top50)
            + 15700.0 * max(0.0, Q.LHA - 0.406) * max(0.0, 0.0046 - Q.lam2)
        ))
        - 1.375 * grid(14, max(0.0, 0.0493
            + 2.54 * max(0.0, Q.log_sum_pt - 6.94)
            - 0.135 * max(0.0, 91.2 - Q.mass)
            + 0.0478 * max(0.0, 135.0 - Q.mass)
            - 52.0 * max(0.0, 0.0888 - Q.mass_over_sum_pt)
            - 2.24 * max(0.0, Q.max_dr - 0.225)
            + 89.4 * max(0.0, 0.96 - Q.z_top50_slots)
            - 118.0 * max(0.0, 0.0129 - Q.girth2_top20) * max(0.0, 0.6 - Q.tau21)
        ))
        - 0.1875 * grid(15, max(0.0, 2.19
            + 34.5 * max(0.0, Q.LHA - 0.163)
            - 120.0 * max(0.0, Q.girth - 0.0293)
            - 136.0 * max(0.0, 0.0176 - Q.lam1)
            + 2.61 * max(0.0, 7.05 - Q.log_sum_pt)
            + 28.9 * max(0.0, Q.mass_over_sum_pt - 0.0926)
            - 583.0 * max(0.0, 0.0105 - Q.girth2_top5) * max(0.0, 0.328 - Q.max_dr)
        ))
    )


def logit_Z(Q):
    return (0.984375
        - 1.375 * grid(0, max(0.0, 1.3
            - 27.2 * max(0.0, 0.0562 - Q.girth)
            - 162.0 * max(0.0, 0.00611 - Q.girth2_top20)
            - 197.0 * max(0.0, 0.00583 - Q.lam1)
            - 0.159 * max(0.0, Q.mass - 80.4)
            + 0.154 * max(0.0, Q.mass - 91.2)
            - 66.9 * max(0.0, Q.mass_over_sum_pt - 0.0782)
            + 79.9 * max(0.0, Q.mass_over_sum_pt - 0.0857)
            + 340.0 * max(0.0, 0.00784 - Q.mass_over_sum_pt_sq)
            - 0.0156 * max(0.0, 80.8 - Q.mass_top50)
            + 0.0396 * max(0.0, 10.8 - Q.n_dr_0p2_0p4)
            - 0.00481 * max(0.0, 1010.0 - Q.sum_pt)
            - 2.97 * max(0.0, Q.z_dr_0_0p05 - 0.871)
            + 0.0511 * max(0.0, 6.99 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top40 - 936.0)
            - 0.00923 * max(0.0, Q.mass - 80.4) * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 0.0451 * max(0.0, 70.3 - Q.mass_top50) * max(0.0, 0.231 - Q.z_dr_0p05_0p1)
            - 2.13 * max(0.0, 9.75 - Q.n_dr_0p2_0p4) * max(0.0, 0.98 - Q.z_top50_slots)
            - 64.4 * max(0.0, Q.z_top30_slots - 0.913) * max(0.0, Q.C2 - 0.0639)
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
        + 0.6875 * grid(5, max(0.0, -0.546
            + 31.6 * max(0.0, Q.log_sum_pt - 6.85)
            - 44.1 * max(0.0, Q.log_sum_pt - 6.91)
            + 39.6 * max(0.0, Q.mass_over_sum_pt - 0.0786)
            - 39.7 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            - 450.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 103.0 * max(0.0, 0.00636 - Q.mass_over_sum_pt_sq)
            - 0.213 * max(0.0, Q.mass_top50 - 155.0)
            + 19.7 * max(0.0, Q.max_dr - 0.437)
            + 0.000694 * max(0.0, Q.sum_pt_top3 - 239.0)
            - 0.00484 * max(0.0, Q.sum_pt_top3 - 791.0)
            + 1.14 * max(0.0, 0.881 - Q.tau32)
            - 8.88 * max(0.0, Q.z_top30_slots - 0.936)
            + 1.23 * max(0.0, Q.mass_top50 - 155.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.581)
            + 1.72 * max(0.0, 65.0 - Q.n_particles) * max(0.0, 0.0378 - Q.e2)
        ))
        - 0.96875 * grid(6, max(0.0, 2.55
            - 51.9 * max(0.0, Q.e2 - 0.0526)
            + 23.0 * max(0.0, 0.0477 - Q.e2)
            + 228.0 * max(0.0, Q.lam1 - 0.00793)
            + 6.95 * max(0.0, 6.82 - Q.log_sum_pt)
            + 0.0471 * max(0.0, 91.2 - Q.mass)
            - 0.0514 * max(0.0, 110.0 - Q.mass)
            - 40.3 * max(0.0, Q.mass_over_sum_pt - 0.0491)
            - 166.0 * max(0.0, 0.0101 - Q.mass_over_sum_pt_sq)
            + 0.0531 * max(0.0, 70.5 - Q.mass_top50)
            - 0.0589 * max(0.0, 18.5 - Q.n_dr_0p2_0p4)
            + 1.77 * max(0.0, Q.mass_over_sum_pt - 0.0497) * max(0.0, 18.6 - Q.n_dr_0p2_0p4)
        ))
        + 0.90625 * grid(7, max(0.0, -0.571
            + 66.5 * max(0.0, 0.0286 - Q.e2)
            - 19.4 * max(0.0, 0.0884 - Q.girth)
            + 534.0 * max(0.0, 0.0063 - Q.girth2_top20)
            - 570.0 * max(0.0, 0.00795 - Q.girth2_top20)
            - 0.19 * max(0.0, 91.2 - Q.mass)
            + 0.0645 * max(0.0, 120.0 - Q.mass)
            + 100.0 * max(0.0, 0.0799 - Q.mass_over_sum_pt)
            - 126.0 * max(0.0, 0.0906 - Q.mass_over_sum_pt)
            - 514.0 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq)
            + 788.0 * max(0.0, 0.0095 - Q.mass_over_sum_pt_sq)
            + 0.0775 * max(0.0, 13.0 - Q.n_dr_0p2_0p4)
            + 99.2 * max(0.0, 0.0059 - Q.z_dr_0p2_0p4)
            - 10.2 * max(0.0, 0.0901 - Q.z_dr_0p2_0p4)
            - 36.2 * max(0.0, 0.991 - Q.z_top50_slots)
            + 0.0834 * max(0.0, 1.8 - Q.D2) * max(0.0, 9.09 - Q.n_dr_0p2_0p4)
            + 0.273 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.64 - Q.D2)
            - 1.33 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.392 - Q.max_dr)
            + 0.814 * max(0.0, 101.0 - Q.mass) * max(0.0, 0.397 - Q.max_dr)
            - 0.149 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.422)
            + 0.183 * max(0.0, 75.6 - Q.mass_top30) * max(0.0, 1.61 - Q.D2)
            - 0.388 * max(0.0, 97.4 - Q.mass_top50) * max(0.0, 1.65 - Q.D2)
        ))
        - 0.9375 * grid(8, max(0.0, 5.9
            + 9.62 * max(0.0, Q.C2 - 0.0695)
            + 0.411 * max(0.0, 1.8 - Q.D2)
            - 43.5 * max(0.0, Q.e2 - 0.0203)
            + 243.0 * max(0.0, Q.girth2_top20 - 0.00792)
            - 487.0 * max(0.0, 0.00148 - Q.lam2)
            + 5.7 * max(0.0, Q.log_sum_pt - 6.94)
            - 0.0718 * max(0.0, Q.mass - 65.2)
            + 0.0891 * max(0.0, Q.mass - 86.5)
            - 0.0733 * max(0.0, Q.mass - 105.0)
            + 0.118 * max(0.0, 64.5 - Q.mass)
            - 146.0 * max(0.0, 0.161 - Q.mass_over_sum_pt)
            + 441.0 * max(0.0, 0.0255 - Q.mass_over_sum_pt_sq)
            - 0.0592 * max(0.0, 20.6 - Q.n_dr_0p2_0p4)
            + 0.074 * max(0.0, Q.n_particles - 50.7)
            - 0.0196 * max(0.0, 46.1 - Q.n_particles)
            + 0.0118 * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.00777 * max(0.0, 1040.0 - Q.sum_pt_top40)
            + 0.635 * max(0.0, Q.mass_over_sum_pt - 0.0757) * max(0.0, 1090.0 - Q.sum_pt)
            - 2.79 * max(0.0, Q.n_particles - 51.3) * max(0.0, Q.z_top50_slots - 0.97)
            + 0.0676 * max(0.0, 883.0 - Q.sum_pt_top40) * max(0.0, 6.89 - Q.log_sum_pt)
            - 0.0926 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, 0.39 - Q.max_dr)
        ))
        + 0.3125 * grid(12, max(0.0, 0.0817
            - 21.6 * max(0.0, Q.C2 - 0.0556)
            - 5.35 * max(0.0, 6.85 - Q.log_sum_pt)
            + 0.0839 * max(0.0, 80.4 - Q.mass)
            + 0.0361 * max(0.0, Q.mass_top30 - 128.0)
            - 0.0701 * max(0.0, 62.5 - Q.mass_top50)
            + 15700.0 * max(0.0, Q.LHA - 0.406) * max(0.0, 0.0046 - Q.lam2)
        ))
        + 0.5625 * grid(15, max(0.0, 2.19
            + 34.5 * max(0.0, Q.LHA - 0.163)
            - 120.0 * max(0.0, Q.girth - 0.0293)
            - 136.0 * max(0.0, 0.0176 - Q.lam1)
            + 2.61 * max(0.0, 7.05 - Q.log_sum_pt)
            + 28.9 * max(0.0, Q.mass_over_sum_pt - 0.0926)
            - 583.0 * max(0.0, 0.0105 - Q.girth2_top5) * max(0.0, 0.328 - Q.max_dr)
        ))
    )


def logit_t(Q):
    return (0.78125
        + 0.125 * grid(0, max(0.0, 1.3
            - 27.2 * max(0.0, 0.0562 - Q.girth)
            - 162.0 * max(0.0, 0.00611 - Q.girth2_top20)
            - 197.0 * max(0.0, 0.00583 - Q.lam1)
            - 0.159 * max(0.0, Q.mass - 80.4)
            + 0.154 * max(0.0, Q.mass - 91.2)
            - 66.9 * max(0.0, Q.mass_over_sum_pt - 0.0782)
            + 79.9 * max(0.0, Q.mass_over_sum_pt - 0.0857)
            + 340.0 * max(0.0, 0.00784 - Q.mass_over_sum_pt_sq)
            - 0.0156 * max(0.0, 80.8 - Q.mass_top50)
            + 0.0396 * max(0.0, 10.8 - Q.n_dr_0p2_0p4)
            - 0.00481 * max(0.0, 1010.0 - Q.sum_pt)
            - 2.97 * max(0.0, Q.z_dr_0_0p05 - 0.871)
            + 0.0511 * max(0.0, 6.99 - Q.log_sum_pt) * max(0.0, Q.sum_pt_top40 - 936.0)
            - 0.00923 * max(0.0, Q.mass - 80.4) * max(0.0, 7.0 - Q.n_dr_0p2_0p4)
            + 0.0451 * max(0.0, 70.3 - Q.mass_top50) * max(0.0, 0.231 - Q.z_dr_0p05_0p1)
            - 2.13 * max(0.0, 9.75 - Q.n_dr_0p2_0p4) * max(0.0, 0.98 - Q.z_top50_slots)
            - 64.4 * max(0.0, Q.z_top30_slots - 0.913) * max(0.0, Q.C2 - 0.0639)
        ))
        + 0.1875 * grid(4, max(0.0, -0.431
            + 17.4 * max(0.0, Q.C2 - 0.107)
            + 0.224 * max(0.0, 6.81 - Q.D2)
            + 18.6 * max(0.0, Q.e2 - 0.0356)
            - 21.8 * max(0.0, 0.0616 - Q.girth)
            - 115.0 * max(0.0, 0.00777 - Q.girth2_top20)
            + 75.7 * max(0.0, 0.0231 - Q.girth2_top20)
            - 0.131 * max(0.0, 80.4 - Q.mass)
            + 0.0455 * max(0.0, 103.0 - Q.mass)
            - 71.7 * max(0.0, Q.mass_over_sum_pt - 0.0909)
            + 313.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.00841)
            - 119.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0265)
            + 0.0257 * max(0.0, 18.0 - Q.n_dr_0p2_0p4)
            - 0.013 * max(0.0, Q.n_particles - 23.7)
            - 0.00554 * max(0.0, 1080.0 - Q.sum_pt)
            - 1.5 * max(0.0, 0.454 - Q.tau21)
            + 2.27 * max(0.0, 0.573 - Q.tau32)
            - 10.1 * max(0.0, 0.0698 - Q.z_dr_0p2_0p4)
            - 0.564 * max(0.0, 74.3 - Q.mass) * max(0.0, 0.36 - Q.max_dr)
            + 0.104 * max(0.0, 118.0 - Q.mass) * max(0.0, 0.411 - Q.max_dr)
            - 0.015 * max(0.0, 84.6 - Q.mass_top40) * max(0.0, 6.72 - Q.D2)
        ))
        - 0.46875 * grid(5, max(0.0, -0.546
            + 31.6 * max(0.0, Q.log_sum_pt - 6.85)
            - 44.1 * max(0.0, Q.log_sum_pt - 6.91)
            + 39.6 * max(0.0, Q.mass_over_sum_pt - 0.0786)
            - 39.7 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            - 450.0 * max(0.0, Q.mass_over_sum_pt_sq - 0.0293)
            - 103.0 * max(0.0, 0.00636 - Q.mass_over_sum_pt_sq)
            - 0.213 * max(0.0, Q.mass_top50 - 155.0)
            + 19.7 * max(0.0, Q.max_dr - 0.437)
            + 0.000694 * max(0.0, Q.sum_pt_top3 - 239.0)
            - 0.00484 * max(0.0, Q.sum_pt_top3 - 791.0)
            + 1.14 * max(0.0, 0.881 - Q.tau32)
            - 8.88 * max(0.0, Q.z_top30_slots - 0.936)
            + 1.23 * max(0.0, Q.mass_top50 - 155.0) * max(0.0, Q.z_dr_0p05_0p1 - 0.581)
            + 1.72 * max(0.0, 65.0 - Q.n_particles) * max(0.0, 0.0378 - Q.e2)
        ))
        - 0.28125 * grid(7, max(0.0, -0.571
            + 66.5 * max(0.0, 0.0286 - Q.e2)
            - 19.4 * max(0.0, 0.0884 - Q.girth)
            + 534.0 * max(0.0, 0.0063 - Q.girth2_top20)
            - 570.0 * max(0.0, 0.00795 - Q.girth2_top20)
            - 0.19 * max(0.0, 91.2 - Q.mass)
            + 0.0645 * max(0.0, 120.0 - Q.mass)
            + 100.0 * max(0.0, 0.0799 - Q.mass_over_sum_pt)
            - 126.0 * max(0.0, 0.0906 - Q.mass_over_sum_pt)
            - 514.0 * max(0.0, 0.0064 - Q.mass_over_sum_pt_sq)
            + 788.0 * max(0.0, 0.0095 - Q.mass_over_sum_pt_sq)
            + 0.0775 * max(0.0, 13.0 - Q.n_dr_0p2_0p4)
            + 99.2 * max(0.0, 0.0059 - Q.z_dr_0p2_0p4)
            - 10.2 * max(0.0, 0.0901 - Q.z_dr_0p2_0p4)
            - 36.2 * max(0.0, 0.991 - Q.z_top50_slots)
            + 0.0834 * max(0.0, 1.8 - Q.D2) * max(0.0, 9.09 - Q.n_dr_0p2_0p4)
            + 0.273 * max(0.0, 100.0 - Q.mass) * max(0.0, 1.64 - Q.D2)
            - 1.33 * max(0.0, 91.2 - Q.mass) * max(0.0, 0.392 - Q.max_dr)
            + 0.814 * max(0.0, 101.0 - Q.mass) * max(0.0, 0.397 - Q.max_dr)
            - 0.149 * max(0.0, 91.2 - Q.mass) * max(0.0, Q.z_dr_0p05_0p1 - 0.422)
            + 0.183 * max(0.0, 75.6 - Q.mass_top30) * max(0.0, 1.61 - Q.D2)
            - 0.388 * max(0.0, 97.4 - Q.mass_top50) * max(0.0, 1.65 - Q.D2)
        ))
        + 0.2109375 * grid(8, max(0.0, 5.9
            + 9.62 * max(0.0, Q.C2 - 0.0695)
            + 0.411 * max(0.0, 1.8 - Q.D2)
            - 43.5 * max(0.0, Q.e2 - 0.0203)
            + 243.0 * max(0.0, Q.girth2_top20 - 0.00792)
            - 487.0 * max(0.0, 0.00148 - Q.lam2)
            + 5.7 * max(0.0, Q.log_sum_pt - 6.94)
            - 0.0718 * max(0.0, Q.mass - 65.2)
            + 0.0891 * max(0.0, Q.mass - 86.5)
            - 0.0733 * max(0.0, Q.mass - 105.0)
            + 0.118 * max(0.0, 64.5 - Q.mass)
            - 146.0 * max(0.0, 0.161 - Q.mass_over_sum_pt)
            + 441.0 * max(0.0, 0.0255 - Q.mass_over_sum_pt_sq)
            - 0.0592 * max(0.0, 20.6 - Q.n_dr_0p2_0p4)
            + 0.074 * max(0.0, Q.n_particles - 50.7)
            - 0.0196 * max(0.0, 46.1 - Q.n_particles)
            + 0.0118 * max(0.0, 1000.0 - Q.sum_pt_top40)
            - 0.00777 * max(0.0, 1040.0 - Q.sum_pt_top40)
            + 0.635 * max(0.0, Q.mass_over_sum_pt - 0.0757) * max(0.0, 1090.0 - Q.sum_pt)
            - 2.79 * max(0.0, Q.n_particles - 51.3) * max(0.0, Q.z_top50_slots - 0.97)
            + 0.0676 * max(0.0, 883.0 - Q.sum_pt_top40) * max(0.0, 6.89 - Q.log_sum_pt)
            - 0.0926 * max(0.0, 1010.0 - Q.sum_pt_top40) * max(0.0, 0.39 - Q.max_dr)
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
        - 0.375 * grid(12, max(0.0, 0.0817
            - 21.6 * max(0.0, Q.C2 - 0.0556)
            - 5.35 * max(0.0, 6.85 - Q.log_sum_pt)
            + 0.0839 * max(0.0, 80.4 - Q.mass)
            + 0.0361 * max(0.0, Q.mass_top30 - 128.0)
            - 0.0701 * max(0.0, 62.5 - Q.mass_top50)
            + 15700.0 * max(0.0, Q.LHA - 0.406) * max(0.0, 0.0046 - Q.lam2)
        ))
        - 0.90625 * grid(13, max(0.0, 3.07
            + 16.0 * max(0.0, Q.e2 - 0.0369)
            + 18.8 * max(0.0, 6.8 - Q.log_sum_pt)
            - 14.0 * max(0.0, 7.0 - Q.log_sum_pt)
            - 0.145 * max(0.0, Q.mass - 143.0)
            + 0.104 * max(0.0, Q.mass - 172.8)
            + 0.0597 * max(0.0, Q.mass_top50 - 137.0)
            - 0.156 * max(0.0, 4.41 - Q.n_dr_0p2_0p4)
            - 0.0226 * max(0.0, 1020.0 - Q.sum_pt)
            + 0.0145 * max(0.0, 1040.0 - Q.sum_pt_top40)
            - 0.00146 * max(0.0, 1040.0 - Q.sum_pt_top40) * max(0.0, 5.03 - Q.D2)
            + 0.00193 * max(0.0, 961.0 - Q.sum_pt_top50) * max(0.0, 5.16 - Q.D2)
        ))
        - 0.375 * grid(15, max(0.0, 2.19
            + 34.5 * max(0.0, Q.LHA - 0.163)
            - 120.0 * max(0.0, Q.girth - 0.0293)
            - 136.0 * max(0.0, 0.0176 - Q.lam1)
            + 2.61 * max(0.0, 7.05 - Q.log_sum_pt)
            + 28.9 * max(0.0, Q.mass_over_sum_pt - 0.0926)
            - 583.0 * max(0.0, 0.0105 - Q.girth2_top5) * max(0.0, 0.328 - Q.max_dr)
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
