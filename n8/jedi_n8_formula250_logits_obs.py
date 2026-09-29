"""JEDI-linear jet tagger, 8 particles, 3 features: the formula simplified by hand with the training data (main result), with each class score (logit) written directly in terms of the jet quantities.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 65.3% (the network: 65.8%); same class as the network for 87.6% of jets.

Quantities:
  Q.eccentricity           1 − λ2/λ1 of the pT-weighted (Δη, Δφ) tensor
  Q.C2                     energy correlation ratio e3/e2²
  Q.D2                     energy correlation ratio e3/e2³
  Q.LHA                    Les Houches angularity
  Q.log_sum_pt             natural log of the total pT
  Q.mass_over_sum_pt       jet mass / total pT
  Q.mass                   invariant mass of all particles (massless four-vectors) [GeV]
  Q.max_dr                 largest distance ΔR of a particle from the jet axis
  Q.pt_0                   pT of particle 0 [GeV]
  Q.pt_7                   pT of particle 7 [GeV]
  Q.z_4                    pT of particle 4 / total pT
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.dr_0                   ΔR of particle 0 from the jet axis
  Q.dr_7                   ΔR of particle 7 from the jet axis
  Q.phi_1                  Δφ of particle 1
  Q.sum_pt_top2            total pT of the 2 hardest particles [GeV]
  Q.sum_pt_top3            total pT of the 3 hardest particles [GeV]
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.girth2_top3            pT-weighted mean ΔR² of the 3 hardest particles
  Q.n_dr_0_0p05            number of particles with 0 ≤ ΔR < 0.05
  Q.n_dr_0p05_0p1          number of particles with 0.05 ≤ ΔR < 0.1
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.n_pt_above_50          number of particles with pT > 50 GeV
  Q.sum_pt                 total pT of the particles [GeV]
  Q.z_dr_0_0p05            pT share of the particles with 0 ≤ ΔR < 0.05
  Q.z_dr_0p05_0p1          pT share of the particles with 0.05 ≤ ΔR < 0.1
  Q.z_dr_0p1_0p2           pT share of the particles with 0.1 ≤ ΔR < 0.2
  Q.z_dr_0p2_0p4           pT share of the particles with 0.2 ≤ ΔR < 0.4
  Q.girth                  pT-weighted mean ΔR
  Q.girth2                 pT-weighted mean ΔR²
  Q.e2                     energy correlation e2 = Σ_{i<j} zᵢzⱼΔRᵢⱼ
  Q.e2_sq                  Σ_{i<j} zᵢzⱼΔRᵢⱼ²
  Q.lam1                   larger eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.width                  λ1 + λ2 of the pT-weighted (Δη, Δφ) tensor
  Q.lam2                   smaller eigenvalue of the pT-weighted (Δη, Δφ) tensor
  Q.tau21                  N-subjettiness τ2/τ1 (axes from pT-weighted k-means)
  Q.tau32                  N-subjettiness τ3/τ2
  Q.centroid_offset        distance of the pT centroid from the jet axis
"""
import math
from types import SimpleNamespace

CLASSES = ['g', 'q', 'W', 'Z', 't']
W = [[-0.15625, 0.0, 0.34375, 0.0, 0.015625], [0.390625, 0.0, -0.03125, 0.125, 0.0], [0.4296875, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.5, -0.5625, 0.0625], [-0.03125, -0.09375, 0.0, 0.078125, 0.125], [-0.1875, 0.046875, 0.0, 0.015625, -0.25], [0.109375, 0.125, -0.3125, -0.375, 0.0], [0.0, 0.0, 0.21875, 0.46875, 0.0], [0.0, 0.0625, -0.25, 0.0, 0.1875], [0.171875, 0.25390625, -0.03125, -0.03125, 0.0], [0.0, -0.125, 0.0, 0.0, 0.375], [0.0, 0.0, 0.375, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.5], [0.0, 0.0, 0.0703125, 0.0546875, -0.40625], [0.0, 0.0, -0.75, 0.375, 0.0], [0.0, 0.0625, -0.6875, -0.15625, 0.0]]
B = [-0.4375, 0.03125, -0.125, -0.09375, 1.34375]
INT_BITS = [3, 5, 4, 4, 5, 5, 4, 4, 3, 4, 5, 4, 5, 4, 3, 3]
FRAC_BITS = [3, 3, 4, 3, 2, 3, 3, 3, 3, 2, 4, 4, 3, 3, 4, 3]


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
        eccentricity=1 - lam2 / max(lam1, 1e-12),
        C2=e3 / max(e2 ** 2, 1e-12),
        D2=e3 / max(e2 ** 3, 1e-12),
        LHA=sum(z[i] * math.sqrt(dr[i] / 0.8) for i in P),
        log_sum_pt=math.log(tot),
        mass_over_sum_pt=mass_of(n) / tot,
        mass=mass_of(n),
        max_dr=max(dr[i] for i in real),
        pt_0=pt[0],
        pt_7=pt[7],
        z_4=z[4],
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        dr_0=dr[0] if pt[0] > 0 else 0.0,
        dr_7=dr[7] if pt[7] > 0 else 0.0,
        phi_1=phi[1],
        sum_pt_top2=sum(pt[:2]),
        sum_pt_top3=sum(pt[:3]),
        sum_pt_top5=sum(pt[:5]),
        girth2_top3=sum(pt[i] * dr[i] ** 2 for i in range(3)) / max(sum(pt[:3]), 1e-9),
        n_dr_0_0p05=sum(1 for i in real if 0 <= dr[i] < 0.05),
        n_dr_0p05_0p1=sum(1 for i in real if 0.05 <= dr[i] < 0.1),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        n_pt_above_50=sum(1 for x in pt if x > 50),
        sum_pt=tot,
        z_dr_0_0p05=sum(z[i] for i in real if 0 <= dr[i] < 0.05),
        z_dr_0p05_0p1=sum(z[i] for i in real if 0.05 <= dr[i] < 0.1),
        z_dr_0p1_0p2=sum(z[i] for i in real if 0.1 <= dr[i] < 0.2),
        z_dr_0p2_0p4=sum(z[i] for i in real if 0.2 <= dr[i] < 0.4),
        girth=sum(z[i] * dr[i] for i in P),
        girth2=sum(z[i] * dr[i] ** 2 for i in P),
        e2=e2,
        e2_sq=sum(z[i] * z[j] * dist2(i, j) for i in P for j in P if i < j),
        lam1=lam1,
        width=ta + tc,
        lam2=lam2,
        tau21=tau(2) / max(tau(1), 1e-12),
        tau32=tau(3) / max(tau(2), 1e-12),
        centroid_offset=math.hypot(sum(z[i] * eta[i] for i in P), sum(z[i] * phi[i] for i in P)),
    )


def grid(j, v):
    return (math.floor(v * 2 ** FRAC_BITS[j] + 0.5) / 2 ** FRAC_BITS[j]) % 2 ** INT_BITS[j]


def logit_g(Q):
    return (-0.4375
        - 0.15625 * grid(0, max(0.0, -0.48
            + 71.0 * max(0.0, 0.0338 - Q.centroid_offset)
            + 499.0 * max(0.0, Q.eccentricity - 0.997)
            - 56.8 * max(0.0, 0.0775 - Q.girth)
            + 301.0 * max(0.0, 0.0121 - Q.girth2)
            - 0.543 * max(0.0, 22.5 - Q.mass)
            - 0.0265 * max(0.0, 71.3 - Q.mass)
            - 0.0276 * max(0.0, Q.sum_pt - 895.0)
            - 799.0 * max(0.0, 0.00449 - Q.width)
            + 15.4 * max(0.0, Q.z_dr_0_0p05 - 0.852)
            + 3910.0 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
            - 2060.0 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
            - 2.9 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
            + 2.55 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
            - 4.03 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
            - 0.0022 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
            - 0.0518 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
            + 0.00176 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
            - 0.514 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        ))
        + 0.390625 * grid(1, max(0.0, 1.2
            - 23.7 * max(0.0, 0.0491 - Q.C2)
            + 470.0 * max(0.0, 0.00775 - Q.e2_sq)
            + 4.58 * max(0.0, Q.log_sum_pt - 6.41)
            - 63.3 * max(0.0, 0.0455 - Q.max_dr)
            + 0.115 * max(0.0, Q.pt_7 - 30.4)
            - 657.0 * max(0.0, 0.009 - Q.width)
            - 33.7 * max(0.0, 0.0562 - Q.z_7)
            + 6000.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
            - 82.7 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
            - 8390.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
            + 14300.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
            - 235.0 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
            - 1090.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
            + 10300.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
            + 19.0 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
            - 7.05 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
            + 1.53 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
            - 0.00155 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        ))
        + 0.4296875 * grid(2, max(0.0, 3.38
            - 7.91 * max(0.0, Q.LHA - 0.116)
            - 54100.0 * max(0.0, 4.87e-05 - Q.girth2)
            + 267.0 * max(0.0, 0.0053 - Q.lam1)
            + 11.4 * max(0.0, Q.log_sum_pt - 6.83)
            - 13.7 * max(0.0, Q.log_sum_pt - 6.89)
            + 4.86 * max(0.0, 6.46 - Q.log_sum_pt)
            - 0.791 * max(0.0, 0.654 - Q.planar_flow)
            + 0.173 * max(0.0, Q.pt_7 - 30.2)
            - 0.0503 * max(0.0, 54.2 - Q.pt_7)
            + 0.00563 * max(0.0, 788.0 - Q.sum_pt)
            - 45.7 * max(0.0, Q.z_7 - 0.045)
            - 303.0 * max(0.0, 0.0185 - Q.z_7)
            - 32100.0 * max(0.0, 0.00374 - Q.lam1) * max(0.0, 0.00583 - Q.centroid_offset)
            - 1.91 * max(0.0, Q.pt_7 - 29.6) * max(0.0, 0.0514 - Q.C2)
            - 0.564 * max(0.0, Q.pt_7 - 29.2) * max(0.0, Q.max_dr - 0.0755)
        ))
        - 0.03125 * grid(4, max(0.0, 2.33
            - 57.4 * max(0.0, Q.C2 - 0.00878)
            - 6470.0 * max(0.0, 0.000389 - Q.lam2)
            - 174.0 * max(0.0, Q.mass_over_sum_pt - 0.0885)
            + 41.6 * max(0.0, 0.111 - Q.mass_over_sum_pt)
            + 0.0167 * max(0.0, 486.0 - Q.sum_pt_top5)
            + 15.5 * max(0.0, 0.279 - Q.tau21)
            + 393.0 * max(0.0, Q.width - 0.00219)
            + 11.0 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
            - 2.35 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
            - 0.609 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
            + 54.6 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
            + 0.67 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        ))
        - 0.1875 * grid(5, max(0.0, 0.465
            - 169.0 * max(0.0, 0.0238 - Q.dr_0)
            + 106.0 * max(0.0, 0.0374 - Q.e2)
            + 31500.0 * max(0.0, 6.4e-05 - Q.girth2)
            - 67.9 * max(0.0, Q.log_sum_pt - 6.89)
            + 185.0 * max(0.0, 0.0333 - Q.z_7)
            + 107.0 * max(0.0, 0.0683 - Q.z_7)
            - 90.9 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
            + 685000.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
            + 757.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
            - 982.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
            - 1.79 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
            + 75200.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
            - 3370.0 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
            - 0.278 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        ))
        + 0.109375 * grid(6, max(0.0, 6.56
            - 20.8 * max(0.0, Q.C2 - 0.0101)
            - 59.4 * max(0.0, Q.centroid_offset - 0.0212)
            + 85.7 * max(0.0, 0.0508 - Q.e2)
            + 626.0 * max(0.0, 0.00328 - Q.e2_sq)
            + 126.0 * max(0.0, Q.girth - 0.0868)
            - 1680.0 * max(0.0, 0.00862 - Q.girth2)
            + 919.0 * max(0.0, 0.00802 - Q.lam1)
            - 1090.0 * max(0.0, Q.lam2 - 0.0029)
            - 2400.0 * max(0.0, 0.000518 - Q.lam2)
            - 53.2 * max(0.0, Q.mass_over_sum_pt - 0.00941)
            + 14.7 * max(0.0, Q.max_dr - 0.0279)
            - 540.0 * max(0.0, 0.0133 - Q.width)
            + 2.56 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
            + 19800.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
            + 766.0 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
            + 0.343 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
            + 1090.0 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
            + 833.0 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
            + 0.0576 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
            - 68.7 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        ))
        + 0.171875 * grid(9, max(0.0, -2.22
            + 170.0 * max(0.0, 0.0174 - Q.centroid_offset)
            + 1500.0 * max(0.0, Q.lam2 - 0.00136)
            - 4.6 * max(0.0, Q.log_sum_pt - 6.36)
            - 0.136 * max(0.0, 31.3 - Q.mass)
            + 14.7 * max(0.0, 0.257 - Q.max_dr)
            + 1.23 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            + 1550.0 * max(0.0, 0.00662 - Q.width)
            - 1420.0 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
            + 59.4 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
            + 5.63 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
            + 0.208 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
            - 43400.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        ))
    )


def logit_q(Q):
    return (0.03125
        - 0.09375 * grid(4, max(0.0, 2.33
            - 57.4 * max(0.0, Q.C2 - 0.00878)
            - 6470.0 * max(0.0, 0.000389 - Q.lam2)
            - 174.0 * max(0.0, Q.mass_over_sum_pt - 0.0885)
            + 41.6 * max(0.0, 0.111 - Q.mass_over_sum_pt)
            + 0.0167 * max(0.0, 486.0 - Q.sum_pt_top5)
            + 15.5 * max(0.0, 0.279 - Q.tau21)
            + 393.0 * max(0.0, Q.width - 0.00219)
            + 11.0 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
            - 2.35 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
            - 0.609 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
            + 54.6 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
            + 0.67 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        ))
        + 0.046875 * grid(5, max(0.0, 0.465
            - 169.0 * max(0.0, 0.0238 - Q.dr_0)
            + 106.0 * max(0.0, 0.0374 - Q.e2)
            + 31500.0 * max(0.0, 6.4e-05 - Q.girth2)
            - 67.9 * max(0.0, Q.log_sum_pt - 6.89)
            + 185.0 * max(0.0, 0.0333 - Q.z_7)
            + 107.0 * max(0.0, 0.0683 - Q.z_7)
            - 90.9 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
            + 685000.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
            + 757.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
            - 982.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
            - 1.79 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
            + 75200.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
            - 3370.0 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
            - 0.278 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        ))
        + 0.125 * grid(6, max(0.0, 6.56
            - 20.8 * max(0.0, Q.C2 - 0.0101)
            - 59.4 * max(0.0, Q.centroid_offset - 0.0212)
            + 85.7 * max(0.0, 0.0508 - Q.e2)
            + 626.0 * max(0.0, 0.00328 - Q.e2_sq)
            + 126.0 * max(0.0, Q.girth - 0.0868)
            - 1680.0 * max(0.0, 0.00862 - Q.girth2)
            + 919.0 * max(0.0, 0.00802 - Q.lam1)
            - 1090.0 * max(0.0, Q.lam2 - 0.0029)
            - 2400.0 * max(0.0, 0.000518 - Q.lam2)
            - 53.2 * max(0.0, Q.mass_over_sum_pt - 0.00941)
            + 14.7 * max(0.0, Q.max_dr - 0.0279)
            - 540.0 * max(0.0, 0.0133 - Q.width)
            + 2.56 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
            + 19800.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
            + 766.0 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
            + 0.343 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
            + 1090.0 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
            + 833.0 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
            + 0.0576 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
            - 68.7 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        ))
        + 0.0625 * grid(8, max(0.0, 0.104
            + 477.0 * max(0.0, 0.00366 - Q.centroid_offset)
            - 11.6 * max(0.0, Q.log_sum_pt - 6.71)
            - 0.0426 * max(0.0, Q.pt_7 - 37.1)
            - 169000.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
            + 0.561 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
            - 32700.0 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
            + 1240.0 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
            - 15.2 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
            - 76700.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
            + 12900.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        ))
        + 0.25390625 * grid(9, max(0.0, -2.22
            + 170.0 * max(0.0, 0.0174 - Q.centroid_offset)
            + 1500.0 * max(0.0, Q.lam2 - 0.00136)
            - 4.6 * max(0.0, Q.log_sum_pt - 6.36)
            - 0.136 * max(0.0, 31.3 - Q.mass)
            + 14.7 * max(0.0, 0.257 - Q.max_dr)
            + 1.23 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            + 1550.0 * max(0.0, 0.00662 - Q.width)
            - 1420.0 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
            + 59.4 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
            + 5.63 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
            + 0.208 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
            - 43400.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        ))
        - 0.125 * grid(10, max(0.0, 1.73
            + 27.9 * max(0.0, Q.C2 - 0.0596)
            + 61.3 * max(0.0, Q.e2 - 0.0458)
            - 70.6 * max(0.0, 0.0467 - Q.e2)
            - 1260.0 * max(0.0, 0.00182 - Q.girth2)
            + 514.0 * max(0.0, Q.lam2 - 0.000227)
            - 6.91 * max(0.0, Q.log_sum_pt - 6.69)
            - 3.17 * max(0.0, 6.24 - Q.log_sum_pt)
            + 0.0236 * max(0.0, Q.mass - 9.7)
            + 31.9 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
            + 0.324 * max(0.0, 2.95 - Q.n_dr_0p05_0p1)
            + 0.519 * max(0.0, Q.n_dr_0p2_0p4 - 1.56)
            + 12.1 * max(0.0, 0.271 - Q.tau32)
            - 42.4 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
            + 259.0 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
            + 3650.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
            - 9.18 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
            + 2350.0 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
            - 4.11 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        ))
        + 0.0625 * grid(15, max(0.0, 0.951
            - 1.98 * max(0.0, 0.711 - Q.D2)
            - 54.8 * max(0.0, Q.LHA - 0.342)
            + 199.0 * max(0.0, 0.0244 - Q.e2)
            + 104.0 * max(0.0, 0.041 - Q.e2)
            - 665.0 * max(0.0, 0.00198 - Q.girth2_top3)
            + 1610.0 * max(0.0, 0.00679 - Q.lam1)
            - 790.0 * max(0.0, 0.00813 - Q.lam1)
            + 37.6 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0463 * max(0.0, 37.4 - Q.mass)
            + 0.454 * max(0.0, 2.12 - Q.n_dr_0_0p05)
            - 2070.0 * max(0.0, 0.00695 - Q.width)
            - 6.11 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
            + 3.61 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
            + 0.0311 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
            - 0.572 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
            - 9.02 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
            + 62.2 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
            - 35800.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
            - 7700.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
            - 3420.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
        ))
    )


def logit_W(Q):
    return (-0.125
        + 0.34375 * grid(0, max(0.0, -0.48
            + 71.0 * max(0.0, 0.0338 - Q.centroid_offset)
            + 499.0 * max(0.0, Q.eccentricity - 0.997)
            - 56.8 * max(0.0, 0.0775 - Q.girth)
            + 301.0 * max(0.0, 0.0121 - Q.girth2)
            - 0.543 * max(0.0, 22.5 - Q.mass)
            - 0.0265 * max(0.0, 71.3 - Q.mass)
            - 0.0276 * max(0.0, Q.sum_pt - 895.0)
            - 799.0 * max(0.0, 0.00449 - Q.width)
            + 15.4 * max(0.0, Q.z_dr_0_0p05 - 0.852)
            + 3910.0 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
            - 2060.0 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
            - 2.9 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
            + 2.55 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
            - 4.03 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
            - 0.0022 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
            - 0.0518 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
            + 0.00176 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
            - 0.514 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        ))
        - 0.03125 * grid(1, max(0.0, 1.2
            - 23.7 * max(0.0, 0.0491 - Q.C2)
            + 470.0 * max(0.0, 0.00775 - Q.e2_sq)
            + 4.58 * max(0.0, Q.log_sum_pt - 6.41)
            - 63.3 * max(0.0, 0.0455 - Q.max_dr)
            + 0.115 * max(0.0, Q.pt_7 - 30.4)
            - 657.0 * max(0.0, 0.009 - Q.width)
            - 33.7 * max(0.0, 0.0562 - Q.z_7)
            + 6000.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
            - 82.7 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
            - 8390.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
            + 14300.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
            - 235.0 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
            - 1090.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
            + 10300.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
            + 19.0 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
            - 7.05 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
            + 1.53 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
            - 0.00155 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        ))
        - 0.5 * grid(3, max(0.0, 2.9
            + 49.3 * max(0.0, Q.centroid_offset - 0.0103)
            - 126.0 * max(0.0, Q.e2 - 0.0278)
            + 179.0 * max(0.0, 0.0433 - Q.e2)
            - 833.0 * max(0.0, 0.00905 - Q.girth2)
            - 441.0 * max(0.0, 0.0126 - Q.girth2)
            - 0.0685 * max(0.0, Q.mass - 71.9)
            + 1090.0 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
            - 1610.0 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
            - 328.0 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
            - 12100.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
            + 2.06 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
            - 7530.0 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
            + 9260.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
            + 10.8 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
            - 133.0 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        ))
        - 0.3125 * grid(6, max(0.0, 6.56
            - 20.8 * max(0.0, Q.C2 - 0.0101)
            - 59.4 * max(0.0, Q.centroid_offset - 0.0212)
            + 85.7 * max(0.0, 0.0508 - Q.e2)
            + 626.0 * max(0.0, 0.00328 - Q.e2_sq)
            + 126.0 * max(0.0, Q.girth - 0.0868)
            - 1680.0 * max(0.0, 0.00862 - Q.girth2)
            + 919.0 * max(0.0, 0.00802 - Q.lam1)
            - 1090.0 * max(0.0, Q.lam2 - 0.0029)
            - 2400.0 * max(0.0, 0.000518 - Q.lam2)
            - 53.2 * max(0.0, Q.mass_over_sum_pt - 0.00941)
            + 14.7 * max(0.0, Q.max_dr - 0.0279)
            - 540.0 * max(0.0, 0.0133 - Q.width)
            + 2.56 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
            + 19800.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
            + 766.0 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
            + 0.343 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
            + 1090.0 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
            + 833.0 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
            + 0.0576 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
            - 68.7 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        ))
        + 0.21875 * grid(7, max(0.0, 4.74
            - 37.9 * max(0.0, 0.0201 - Q.centroid_offset)
            + 56.5 * max(0.0, 0.0253 - Q.e2)
            + 81.6 * max(0.0, 0.0372 - Q.e2)
            + 1320.0 * max(0.0, 0.00115 - Q.e2_sq)
            - 40.9 * max(0.0, 0.089 - Q.girth)
            - 765.0 * max(0.0, Q.girth2 - 0.00445)
            - 2290.0 * max(0.0, 0.000708 - Q.girth2)
            - 0.206 * max(0.0, Q.mass - 80.4)
            + 183.0 * max(0.0, Q.mass_over_sum_pt - 0.0729)
            - 185.0 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            - 7.43 * max(0.0, 0.184 - Q.planar_flow)
            - 1070.0 * max(0.0, 0.00564 - Q.width)
            + 1100.0 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245)
            - 8.88 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
            + 120.0 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
            + 10600.0 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
            - 754.0 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
            - 2.23 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928)
            + 0.0407 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
            - 3780.0 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
            - 0.0953 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow)
        ))
        - 0.25 * grid(8, max(0.0, 0.104
            + 477.0 * max(0.0, 0.00366 - Q.centroid_offset)
            - 11.6 * max(0.0, Q.log_sum_pt - 6.71)
            - 0.0426 * max(0.0, Q.pt_7 - 37.1)
            - 169000.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
            + 0.561 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
            - 32700.0 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
            + 1240.0 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
            - 15.2 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
            - 76700.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
            + 12900.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        ))
        - 0.03125 * grid(9, max(0.0, -2.22
            + 170.0 * max(0.0, 0.0174 - Q.centroid_offset)
            + 1500.0 * max(0.0, Q.lam2 - 0.00136)
            - 4.6 * max(0.0, Q.log_sum_pt - 6.36)
            - 0.136 * max(0.0, 31.3 - Q.mass)
            + 14.7 * max(0.0, 0.257 - Q.max_dr)
            + 1.23 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            + 1550.0 * max(0.0, 0.00662 - Q.width)
            - 1420.0 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
            + 59.4 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
            + 5.63 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
            + 0.208 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
            - 43400.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        ))
        + 0.375 * grid(11, max(0.0, -1.15
            - 19.1 * max(0.0, 0.0355 - Q.C2)
            - 22.8 * max(0.0, 0.157 - Q.LHA)
            - 73.1 * max(0.0, Q.centroid_offset - 0.0144)
            + 101.0 * max(0.0, 0.0498 - Q.centroid_offset)
            - 81.5 * max(0.0, Q.girth - 0.0766)
            - 84.9 * max(0.0, 0.0883 - Q.girth)
            + 8.98 * max(0.0, 0.271 - Q.planar_flow)
            + 0.00545 * max(0.0, 688.0 - Q.sum_pt_top5)
            - 514.0 * max(0.0, 0.00365 - Q.width)
            + 1070.0 * max(0.0, 0.0087 - Q.width)
            - 101.0 * max(0.0, 0.0437 - Q.centroid_offset) * max(0.0, 6.84 - Q.log_sum_pt)
            + 12.6 * max(0.0, Q.girth - 0.0771) * max(0.0, 7.47 - Q.n_pt_above_50)
            - 0.128 * max(0.0, 0.254 - Q.planar_flow) * max(0.0, 66.7 - Q.mass)
            - 57.9 * max(0.0, 0.281 - Q.planar_flow) * max(0.0, Q.max_dr - 0.114)
            - 1310.0 * max(0.0, 0.217 - Q.planar_flow) * max(0.0, Q.width - 0.0053)
        ))
        + 0.0703125 * grid(13, max(0.0, 1.68
            - 52.4 * max(0.0, Q.C2 - 0.0664)
            - 88.6 * max(0.0, 0.0509 - Q.e2)
            + 67.0 * max(0.0, 0.15 - Q.girth)
            + 0.0022 * max(0.0, Q.pt_0 - 179.0)
            - 0.246 * max(0.0, 24.8 - Q.pt_7)
            - 0.0241 * max(0.0, Q.sum_pt - 998.0)
            + 217.0 * max(0.0, 0.0159 - Q.width)
            - 41.1 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
            - 1.09 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
            + 0.019 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
            + 6.47 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
            + 0.000594 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
            - 0.566 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
            - 14.9 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        ))
        - 0.75 * grid(14, max(0.0, -1.51
            + 106.0 * max(0.0, 0.0385 - Q.e2)
            - 80.1 * max(0.0, 0.0869 - Q.girth)
            + 632.0 * max(0.0, 0.0131 - Q.girth2)
            - 101.0 * max(0.0, Q.lam1 - 0.00732)
            + 0.0187 * max(0.0, Q.mass - 5.61)
            - 20.1 * max(0.0, 0.177 - Q.max_dr)
            - 1040.0 * max(0.0, 0.00741 - Q.width)
            + 1.85 * max(0.0, 0.572 - Q.z_dr_0p05_0p1)
            + 38.5 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
            + 192.0 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2)
            + 11600.0 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
            - 486.0 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
            + 6850.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
            + 43000.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
            - 0.0486 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2)
            + 1270.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
            - 1310.0 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
            - 187.0 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr)
            - 0.0507 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt)
            - 986.0 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
            + 381.0 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
            - 6890.0 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
            - 77.0 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        ))
        - 0.6875 * grid(15, max(0.0, 0.951
            - 1.98 * max(0.0, 0.711 - Q.D2)
            - 54.8 * max(0.0, Q.LHA - 0.342)
            + 199.0 * max(0.0, 0.0244 - Q.e2)
            + 104.0 * max(0.0, 0.041 - Q.e2)
            - 665.0 * max(0.0, 0.00198 - Q.girth2_top3)
            + 1610.0 * max(0.0, 0.00679 - Q.lam1)
            - 790.0 * max(0.0, 0.00813 - Q.lam1)
            + 37.6 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0463 * max(0.0, 37.4 - Q.mass)
            + 0.454 * max(0.0, 2.12 - Q.n_dr_0_0p05)
            - 2070.0 * max(0.0, 0.00695 - Q.width)
            - 6.11 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
            + 3.61 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
            + 0.0311 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
            - 0.572 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
            - 9.02 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
            + 62.2 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
            - 35800.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
            - 7700.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
            - 3420.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
        ))
    )


def logit_Z(Q):
    return (-0.09375
        + 0.125 * grid(1, max(0.0, 1.2
            - 23.7 * max(0.0, 0.0491 - Q.C2)
            + 470.0 * max(0.0, 0.00775 - Q.e2_sq)
            + 4.58 * max(0.0, Q.log_sum_pt - 6.41)
            - 63.3 * max(0.0, 0.0455 - Q.max_dr)
            + 0.115 * max(0.0, Q.pt_7 - 30.4)
            - 657.0 * max(0.0, 0.009 - Q.width)
            - 33.7 * max(0.0, 0.0562 - Q.z_7)
            + 6000.0 * max(0.0, 0.0384 - Q.e2) * max(0.0, Q.eccentricity - 0.979)
            - 82.7 * max(0.0, Q.e2 - 0.031) * max(0.0, 0.629 - Q.tau32)
            - 8390.0 * max(0.0, 0.00778 - Q.e2_sq) * max(0.0, 0.0847 - Q.planar_flow)
            + 14300.0 * max(0.0, 0.0083 - Q.lam1) * max(0.0, Q.centroid_offset - 0.021)
            - 235.0 * max(0.0, Q.log_sum_pt - 6.4) * max(0.0, 0.0235 - Q.centroid_offset)
            - 1090.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0063 - Q.girth2_top3)
            + 10300.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0012 - Q.lam2)
            + 19.0 * max(0.0, Q.log_sum_pt - 6.33) * max(0.0, 0.191 - Q.max_dr)
            - 7.05 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, Q.n_pt_above_50 - 7.01)
            + 1.53 * max(0.0, Q.pt_7 - 32.9) * max(0.0, Q.centroid_offset - 0.0144)
            - 0.00155 * max(0.0, Q.pt_7 - 29.2) * max(0.0, 97.0 - Q.mass)
        ))
        - 0.5625 * grid(3, max(0.0, 2.9
            + 49.3 * max(0.0, Q.centroid_offset - 0.0103)
            - 126.0 * max(0.0, Q.e2 - 0.0278)
            + 179.0 * max(0.0, 0.0433 - Q.e2)
            - 833.0 * max(0.0, 0.00905 - Q.girth2)
            - 441.0 * max(0.0, 0.0126 - Q.girth2)
            - 0.0685 * max(0.0, Q.mass - 71.9)
            + 1090.0 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
            - 1610.0 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
            - 328.0 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
            - 12100.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
            + 2.06 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
            - 7530.0 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
            + 9260.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
            + 10.8 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
            - 133.0 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        ))
        + 0.078125 * grid(4, max(0.0, 2.33
            - 57.4 * max(0.0, Q.C2 - 0.00878)
            - 6470.0 * max(0.0, 0.000389 - Q.lam2)
            - 174.0 * max(0.0, Q.mass_over_sum_pt - 0.0885)
            + 41.6 * max(0.0, 0.111 - Q.mass_over_sum_pt)
            + 0.0167 * max(0.0, 486.0 - Q.sum_pt_top5)
            + 15.5 * max(0.0, 0.279 - Q.tau21)
            + 393.0 * max(0.0, Q.width - 0.00219)
            + 11.0 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
            - 2.35 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
            - 0.609 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
            + 54.6 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
            + 0.67 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        ))
        + 0.015625 * grid(5, max(0.0, 0.465
            - 169.0 * max(0.0, 0.0238 - Q.dr_0)
            + 106.0 * max(0.0, 0.0374 - Q.e2)
            + 31500.0 * max(0.0, 6.4e-05 - Q.girth2)
            - 67.9 * max(0.0, Q.log_sum_pt - 6.89)
            + 185.0 * max(0.0, 0.0333 - Q.z_7)
            + 107.0 * max(0.0, 0.0683 - Q.z_7)
            - 90.9 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
            + 685000.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
            + 757.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
            - 982.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
            - 1.79 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
            + 75200.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
            - 3370.0 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
            - 0.278 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        ))
        - 0.375 * grid(6, max(0.0, 6.56
            - 20.8 * max(0.0, Q.C2 - 0.0101)
            - 59.4 * max(0.0, Q.centroid_offset - 0.0212)
            + 85.7 * max(0.0, 0.0508 - Q.e2)
            + 626.0 * max(0.0, 0.00328 - Q.e2_sq)
            + 126.0 * max(0.0, Q.girth - 0.0868)
            - 1680.0 * max(0.0, 0.00862 - Q.girth2)
            + 919.0 * max(0.0, 0.00802 - Q.lam1)
            - 1090.0 * max(0.0, Q.lam2 - 0.0029)
            - 2400.0 * max(0.0, 0.000518 - Q.lam2)
            - 53.2 * max(0.0, Q.mass_over_sum_pt - 0.00941)
            + 14.7 * max(0.0, Q.max_dr - 0.0279)
            - 540.0 * max(0.0, 0.0133 - Q.width)
            + 2.56 * max(0.0, Q.C2 - 0.0103) * max(0.0, Q.pt_7 - 32.6)
            + 19800.0 * max(0.0, Q.centroid_offset - 0.00781) * max(0.0, 0.00351 - Q.lam2)
            + 766.0 * max(0.0, 0.0502 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.159)
            + 0.343 * max(0.0, 6.57 - Q.log_sum_pt) * max(0.0, 45.4 - Q.pt_7)
            + 1090.0 * max(0.0, 6.31 - Q.log_sum_pt) * max(0.0, 0.0712 - Q.z_7)
            + 833.0 * max(0.0, 6.71 - Q.log_sum_pt) * max(0.0, 0.0492 - Q.z_7)
            + 0.0576 * max(0.0, 46.6 - Q.mass) * max(0.0, 0.8 - Q.z_dr_0p05_0p1)
            - 68.7 * max(0.0, Q.mass_over_sum_pt - 0.0171) * max(0.0, 0.53 - Q.tau32)
        ))
        + 0.46875 * grid(7, max(0.0, 4.74
            - 37.9 * max(0.0, 0.0201 - Q.centroid_offset)
            + 56.5 * max(0.0, 0.0253 - Q.e2)
            + 81.6 * max(0.0, 0.0372 - Q.e2)
            + 1320.0 * max(0.0, 0.00115 - Q.e2_sq)
            - 40.9 * max(0.0, 0.089 - Q.girth)
            - 765.0 * max(0.0, Q.girth2 - 0.00445)
            - 2290.0 * max(0.0, 0.000708 - Q.girth2)
            - 0.206 * max(0.0, Q.mass - 80.4)
            + 183.0 * max(0.0, Q.mass_over_sum_pt - 0.0729)
            - 185.0 * max(0.0, Q.mass_over_sum_pt - 0.0906)
            - 7.43 * max(0.0, 0.184 - Q.planar_flow)
            - 1070.0 * max(0.0, 0.00564 - Q.width)
            + 1100.0 * max(0.0, 0.0228 - Q.centroid_offset) * max(0.0, Q.C2 - 0.0245)
            - 8.88 * max(0.0, Q.centroid_offset - 0.0311) * max(0.0, Q.pt_0 - 375.0)
            + 120.0 * max(0.0, 0.0513 - Q.e2) * max(0.0, 1.1 - Q.D2)
            + 10600.0 * max(0.0, Q.girth2 - 0.0044) * max(0.0, Q.eccentricity - 0.945)
            - 754.0 * max(0.0, 0.00797 - Q.lam1) * max(0.0, 1.06 - Q.D2)
            - 2.23 * max(0.0, Q.mass - 80.4) * max(0.0, Q.eccentricity - 0.928)
            + 0.0407 * max(0.0, 0.179 - Q.planar_flow) * max(0.0, Q.sum_pt - 607.0)
            - 3780.0 * max(0.0, 0.205 - Q.planar_flow) * max(0.0, Q.width - 0.00768)
            - 0.0953 * max(0.0, 44.5 - Q.pt_7) * max(0.0, 0.596 - Q.planar_flow)
        ))
        - 0.03125 * grid(9, max(0.0, -2.22
            + 170.0 * max(0.0, 0.0174 - Q.centroid_offset)
            + 1500.0 * max(0.0, Q.lam2 - 0.00136)
            - 4.6 * max(0.0, Q.log_sum_pt - 6.36)
            - 0.136 * max(0.0, 31.3 - Q.mass)
            + 14.7 * max(0.0, 0.257 - Q.max_dr)
            + 1.23 * max(0.0, Q.n_dr_0p2_0p4 - 1.0)
            + 1550.0 * max(0.0, 0.00662 - Q.width)
            - 1420.0 * max(0.0, 0.019 - Q.centroid_offset) * max(0.0, Q.z_4 - 0.031)
            + 59.4 * max(0.0, Q.girth2 - 0.0184) * max(0.0, 29.8 - Q.pt_7)
            + 5.63 * max(0.0, 51.1 - Q.mass) * max(0.0, 0.0241 - Q.centroid_offset)
            + 0.208 * max(0.0, 49.9 - Q.mass) * max(0.0, 6.79 - Q.log_sum_pt)
            - 43400.0 * max(0.0, 0.00676 - Q.width) * max(0.0, Q.centroid_offset - 0.00285)
        ))
        + 0.0546875 * grid(13, max(0.0, 1.68
            - 52.4 * max(0.0, Q.C2 - 0.0664)
            - 88.6 * max(0.0, 0.0509 - Q.e2)
            + 67.0 * max(0.0, 0.15 - Q.girth)
            + 0.0022 * max(0.0, Q.pt_0 - 179.0)
            - 0.246 * max(0.0, 24.8 - Q.pt_7)
            - 0.0241 * max(0.0, Q.sum_pt - 998.0)
            + 217.0 * max(0.0, 0.0159 - Q.width)
            - 41.1 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
            - 1.09 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
            + 0.019 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
            + 6.47 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
            + 0.000594 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
            - 0.566 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
            - 14.9 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
        ))
        + 0.375 * grid(14, max(0.0, -1.51
            + 106.0 * max(0.0, 0.0385 - Q.e2)
            - 80.1 * max(0.0, 0.0869 - Q.girth)
            + 632.0 * max(0.0, 0.0131 - Q.girth2)
            - 101.0 * max(0.0, Q.lam1 - 0.00732)
            + 0.0187 * max(0.0, Q.mass - 5.61)
            - 20.1 * max(0.0, 0.177 - Q.max_dr)
            - 1040.0 * max(0.0, 0.00741 - Q.width)
            + 1.85 * max(0.0, 0.572 - Q.z_dr_0p05_0p1)
            + 38.5 * max(0.0, 1.04 - Q.D2) * max(0.0, 0.0307 - Q.centroid_offset)
            + 192.0 * max(0.0, 0.0382 - Q.e2) * max(0.0, 0.961 - Q.D2)
            + 11600.0 * max(0.0, 0.0135 - Q.girth2) * max(0.0, Q.eccentricity - 0.971)
            - 486.0 * max(0.0, 0.00453 - Q.girth2) * max(0.0, 0.93 - Q.n_dr_0p2_0p4)
            + 6850.0 * max(0.0, Q.lam1 - 0.0055) * max(0.0, 0.168 - Q.max_dr)
            + 43000.0 * max(0.0, Q.lam1 - 0.00727) * max(0.0, 0.134 - Q.max_dr)
            - 0.0486 * max(0.0, 75.3 - Q.mass) * max(0.0, 0.865 - Q.D2)
            + 1270.0 * max(0.0, 0.11 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.00946)
            - 1310.0 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.0186)
            - 187.0 * max(0.0, 0.116 - Q.planar_flow) * max(0.0, 0.164 - Q.max_dr)
            - 0.0507 * max(0.0, 0.106 - Q.planar_flow) * max(0.0, 740.0 - Q.sum_pt)
            - 986.0 * max(0.0, 0.00857 - Q.width) * max(0.0, 1.04 - Q.D2)
            + 381.0 * max(0.0, 0.00793 - Q.width) * max(0.0, 6.82 - Q.log_sum_pt)
            - 6890.0 * max(0.0, 0.00758 - Q.width) * max(0.0, 0.109 - Q.planar_flow)
            - 77.0 * max(0.0, 0.597 - Q.z_dr_0p05_0p1) * max(0.0, 0.0662 - Q.C2)
        ))
        - 0.15625 * grid(15, max(0.0, 0.951
            - 1.98 * max(0.0, 0.711 - Q.D2)
            - 54.8 * max(0.0, Q.LHA - 0.342)
            + 199.0 * max(0.0, 0.0244 - Q.e2)
            + 104.0 * max(0.0, 0.041 - Q.e2)
            - 665.0 * max(0.0, 0.00198 - Q.girth2_top3)
            + 1610.0 * max(0.0, 0.00679 - Q.lam1)
            - 790.0 * max(0.0, 0.00813 - Q.lam1)
            + 37.6 * max(0.0, Q.log_sum_pt - 6.9)
            - 0.0463 * max(0.0, 37.4 - Q.mass)
            + 0.454 * max(0.0, 2.12 - Q.n_dr_0_0p05)
            - 2070.0 * max(0.0, 0.00695 - Q.width)
            - 6.11 * max(0.0, Q.z_dr_0p05_0p1 - 0.757)
            + 3.61 * max(0.0, 0.342 - Q.z_dr_0p1_0p2)
            + 0.0311 * max(0.0, Q.LHA - 0.186) * max(0.0, Q.sum_pt_top3 - 338.0)
            - 0.572 * max(0.0, Q.mass - 80.4) * max(0.0, 0.236 - Q.z_dr_0p2_0p4)
            - 9.02 * max(0.0, 0.26 - Q.tau21) * max(0.0, 0.513 - Q.z_dr_0p05_0p1)
            + 62.2 * max(0.0, 0.24 - Q.tau21) * max(0.0, 0.21 - Q.z_dr_0p2_0p4)
            - 35800.0 * max(0.0, 0.00788 - Q.width) * max(0.0, Q.e2 - 0.0243)
            - 7700.0 * max(0.0, 0.00614 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
            - 3420.0 * max(0.0, 0.00868 - Q.width) * max(0.0, 0.0755 - Q.planar_flow)
        ))
    )


def logit_t(Q):
    return (1.34375
        + 0.015625 * grid(0, max(0.0, -0.48
            + 71.0 * max(0.0, 0.0338 - Q.centroid_offset)
            + 499.0 * max(0.0, Q.eccentricity - 0.997)
            - 56.8 * max(0.0, 0.0775 - Q.girth)
            + 301.0 * max(0.0, 0.0121 - Q.girth2)
            - 0.543 * max(0.0, 22.5 - Q.mass)
            - 0.0265 * max(0.0, 71.3 - Q.mass)
            - 0.0276 * max(0.0, Q.sum_pt - 895.0)
            - 799.0 * max(0.0, 0.00449 - Q.width)
            + 15.4 * max(0.0, Q.z_dr_0_0p05 - 0.852)
            + 3910.0 * max(0.0, 0.0197 - Q.girth2) * max(0.0, Q.eccentricity - 0.961)
            - 2060.0 * max(0.0, 0.00645 - Q.lam1) * max(0.0, 0.886 - Q.D2)
            - 2.9 * max(0.0, 30.0 - Q.mass) * max(0.0, 0.91 - Q.D2)
            + 2.55 * max(0.0, 64.8 - Q.mass) * max(0.0, Q.centroid_offset - 0.0106)
            - 4.03 * max(0.0, 29.3 - Q.mass) * max(0.0, Q.phi_1 - -0.0771)
            - 0.0022 * max(0.0, 64.9 - Q.mass) * max(0.0, 40.1 - Q.pt_7)
            - 0.0518 * max(0.0, 0.148 - Q.planar_flow) * max(0.0, 388.0 - Q.sum_pt_top2)
            + 0.00176 * max(0.0, Q.sum_pt - 871.0) * max(0.0, 27.8 - Q.pt_7)
            - 0.514 * max(0.0, Q.sum_pt_top5 - 655.0) * max(0.0, Q.z_7 - 0.0266)
        ))
        + 0.0625 * grid(3, max(0.0, 2.9
            + 49.3 * max(0.0, Q.centroid_offset - 0.0103)
            - 126.0 * max(0.0, Q.e2 - 0.0278)
            + 179.0 * max(0.0, 0.0433 - Q.e2)
            - 833.0 * max(0.0, 0.00905 - Q.girth2)
            - 441.0 * max(0.0, 0.0126 - Q.girth2)
            - 0.0685 * max(0.0, Q.mass - 71.9)
            + 1090.0 * max(0.0, Q.LHA - 0.316) * max(0.0, Q.eccentricity - 0.953)
            - 1610.0 * max(0.0, Q.LHA - 0.308) * max(0.0, 0.156 - Q.max_dr)
            - 328.0 * max(0.0, 0.0435 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.000595)
            - 12100.0 * max(0.0, Q.lam1 - 0.0152) * max(0.0, Q.eccentricity - 0.951)
            + 2.06 * max(0.0, Q.mass - 62.0) * max(0.0, 0.176 - Q.max_dr)
            - 7530.0 * max(0.0, Q.mass_over_sum_pt - 0.0682) * max(0.0, 0.0419 - Q.dr_7)
            + 9260.0 * max(0.0, Q.mass_over_sum_pt - 0.0909) * max(0.0, 0.148 - Q.max_dr)
            + 10.8 * max(0.0, Q.mass_over_sum_pt - 0.0616) * max(0.0, 5.99 - Q.n_dr_0_0p05)
            - 133.0 * max(0.0, Q.mass_over_sum_pt - 0.0749) * max(0.0, 0.529 - Q.tau32)
        ))
        + 0.125 * grid(4, max(0.0, 2.33
            - 57.4 * max(0.0, Q.C2 - 0.00878)
            - 6470.0 * max(0.0, 0.000389 - Q.lam2)
            - 174.0 * max(0.0, Q.mass_over_sum_pt - 0.0885)
            + 41.6 * max(0.0, 0.111 - Q.mass_over_sum_pt)
            + 0.0167 * max(0.0, 486.0 - Q.sum_pt_top5)
            + 15.5 * max(0.0, 0.279 - Q.tau21)
            + 393.0 * max(0.0, Q.width - 0.00219)
            + 11.0 * max(0.0, Q.C2 - 0.0157) * max(0.0, Q.pt_7 - 39.7)
            - 2.35 * max(0.0, Q.max_dr - 0.109) * max(0.0, Q.pt_7 - 39.2)
            - 0.609 * max(0.0, 0.285 - Q.tau21) * max(0.0, 66.2 - Q.mass)
            + 54.6 * max(0.0, 0.243 - Q.tau21) * max(0.0, Q.planar_flow - 0.0325)
            + 0.67 * max(0.0, 0.246 - Q.tau21) * max(0.0, Q.pt_7 - 34.2)
        ))
        - 0.25 * grid(5, max(0.0, 0.465
            - 169.0 * max(0.0, 0.0238 - Q.dr_0)
            + 106.0 * max(0.0, 0.0374 - Q.e2)
            + 31500.0 * max(0.0, 6.4e-05 - Q.girth2)
            - 67.9 * max(0.0, Q.log_sum_pt - 6.89)
            + 185.0 * max(0.0, 0.0333 - Q.z_7)
            + 107.0 * max(0.0, 0.0683 - Q.z_7)
            - 90.9 * max(0.0, 0.221 - Q.LHA) * max(0.0, 6.79 - Q.log_sum_pt)
            + 685000.0 * max(0.0, 0.0338 - Q.e2) * max(0.0, 7.09e-05 - Q.lam2)
            + 757.0 * max(0.0, Q.log_sum_pt - 6.6) * max(0.0, 0.0225 - Q.dr_0)
            - 982.0 * max(0.0, Q.log_sum_pt - 6.59) * max(0.0, 0.0124 - Q.lam1)
            - 1.79 * max(0.0, 546.0 - Q.sum_pt_top2) * max(0.0, 0.00383 - Q.girth2_top3)
            + 75200.0 * max(0.0, 0.00267 - Q.width) * max(0.0, 0.0234 - Q.centroid_offset)
            - 3370.0 * max(0.0, 0.0707 - Q.z_7) * max(0.0, 0.0301 - Q.centroid_offset)
            - 0.278 * max(0.0, 0.0684 - Q.z_7) * max(0.0, 803.0 - Q.sum_pt)
        ))
        + 0.1875 * grid(8, max(0.0, 0.104
            + 477.0 * max(0.0, 0.00366 - Q.centroid_offset)
            - 11.6 * max(0.0, Q.log_sum_pt - 6.71)
            - 0.0426 * max(0.0, Q.pt_7 - 37.1)
            - 169000.0 * max(0.0, 0.198 - Q.LHA) * max(0.0, 0.000321 - Q.lam2)
            + 0.561 * max(0.0, 0.0218 - Q.e2) * max(0.0, Q.sum_pt - 830.0)
            - 32700.0 * max(0.0, 0.0603 - Q.girth) * max(0.0, Q.lam1 - 0.00119)
            + 1240.0 * max(0.0, 0.00621 - Q.girth2) * max(0.0, 0.453 - Q.planar_flow)
            - 15.2 * max(0.0, 20.2 - Q.mass) * max(0.0, Q.centroid_offset - 0.0163)
            - 76700.0 * max(0.0, 0.00544 - Q.width) * max(0.0, Q.centroid_offset - 0.00598)
            + 12900.0 * max(0.0, 0.00506 - Q.width) * max(0.0, 0.0996 - Q.z_dr_0p2_0p4)
        ))
        + 0.375 * grid(10, max(0.0, 1.73
            + 27.9 * max(0.0, Q.C2 - 0.0596)
            + 61.3 * max(0.0, Q.e2 - 0.0458)
            - 70.6 * max(0.0, 0.0467 - Q.e2)
            - 1260.0 * max(0.0, 0.00182 - Q.girth2)
            + 514.0 * max(0.0, Q.lam2 - 0.000227)
            - 6.91 * max(0.0, Q.log_sum_pt - 6.69)
            - 3.17 * max(0.0, 6.24 - Q.log_sum_pt)
            + 0.0236 * max(0.0, Q.mass - 9.7)
            + 31.9 * max(0.0, 0.0966 - Q.mass_over_sum_pt)
            + 0.324 * max(0.0, 2.95 - Q.n_dr_0p05_0p1)
            + 0.519 * max(0.0, Q.n_dr_0p2_0p4 - 1.56)
            + 12.1 * max(0.0, 0.271 - Q.tau32)
            - 42.4 * max(0.0, Q.LHA - 0.301) * max(0.0, 0.699 - Q.tau21)
            + 259.0 * max(0.0, Q.eccentricity - 0.901) * max(0.0, 0.0539 - Q.z_dr_0p2_0p4)
            + 3650.0 * max(0.0, 0.004 - Q.lam1) * max(0.0, Q.log_sum_pt - 6.69)
            - 9.18 * max(0.0, 0.00395 - Q.lam1) * max(0.0, Q.sum_pt - 989.0)
            + 2350.0 * max(0.0, Q.lam2 - 8.42e-05) * max(0.0, 0.518 - Q.tau21)
            - 4.11 * max(0.0, 0.29 - Q.tau32) * max(0.0, 2.0 - Q.n_dr_0p2_0p4)
        ))
        - 0.5 * grid(12, max(0.0, -0.91
            - 124.0 * max(0.0, Q.e2 - 0.0622)
            + 220.0 * max(0.0, Q.girth2 - 0.0188)
            + 0.0778 * max(0.0, Q.mass - 91.2)
            + 15300.0 * max(0.0, Q.girth2 - 0.0186) * max(0.0, Q.lam2 - 0.000489)
            + 6.92 * max(0.0, Q.girth2 - 0.0197) * max(0.0, Q.pt_7 - 16.1)
        ))
        - 0.40625 * grid(13, max(0.0, 1.68
            - 52.4 * max(0.0, Q.C2 - 0.0664)
            - 88.6 * max(0.0, 0.0509 - Q.e2)
            + 67.0 * max(0.0, 0.15 - Q.girth)
            + 0.0022 * max(0.0, Q.pt_0 - 179.0)
            - 0.246 * max(0.0, 24.8 - Q.pt_7)
            - 0.0241 * max(0.0, Q.sum_pt - 998.0)
            + 217.0 * max(0.0, 0.0159 - Q.width)
            - 41.1 * max(0.0, 0.156 - Q.girth) * max(0.0, 6.91 - Q.log_sum_pt)
            - 1.09 * max(0.0, 0.148 - Q.girth) * max(0.0, 39.5 - Q.pt_7)
            + 0.019 * max(0.0, Q.sum_pt - 1070.0) * max(0.0, Q.n_pt_above_50 - 6.01)
            + 6.47 * max(0.0, 764.0 - Q.sum_pt) * max(0.0, 0.0379 - Q.z_4)
            + 0.000594 * max(0.0, Q.sum_pt_top5 - 653.0) * max(0.0, 42.6 - Q.pt_7)
            - 0.566 * max(0.0, Q.sum_pt_top5 - 677.0) * max(0.0, Q.z_7 - 0.0289)
            - 14.9 * max(0.0, 0.518 - Q.tau21) * max(0.0, Q.max_dr - -0.0382)
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
    pt = [412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3]
    eta = [0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4]
    phi = [-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36]
    c, s, p = classify(pt, eta, phi)
    print('class:', c)
    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))
    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))
