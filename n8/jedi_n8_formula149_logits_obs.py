"""JEDI-linear jet tagger, 8 particles, 3 features: the simpler version of the simplified formula, with each class score (logit) written directly in terms of the jet quantities.

Input:  the 8 hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;
        empty slots have pT = 0.
Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.

1. quantities(): physics quantities of the particles.
2. logit_g() ... logit_t(): each class score as one formula of the quantities:
       B[c] + sum over the 16 groups j of W[j][c] * grid(j, max(0, intercept_j + terms of the quantities)),
   every term being coef * max(0, Q.x - t)  (only counts when x > t),  coef * max(0, t - Q.x)  (only when x < t),
   coef * Q.x, or a product of two of these.  grid(j, v) is the network's rounding: to a multiple of 2^-f, wrapped at 2^i.
3. classify(): the class with the largest score, and the softmax probabilities.

Test set (50,000 jets of the hls4ml LHC jet dataset, test file): accuracy 64.6% (the network: 65.8%); same class as the network for 86.5% of jets.

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
  Q.z_7                    pT of particle 7 / total pT
  Q.planar_flow            planar flow of the pT-weighted (Δη, Δφ) tensor
  Q.sum_pt_top5            total pT of the 5 hardest particles [GeV]
  Q.n_dr_0p2_0p4           number of particles with 0.2 ≤ ΔR < 0.4
  Q.sum_pt                 total pT of the particles [GeV]
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
        z_7=z[7],
        planar_flow=4 * (ta * tc - tb ** 2) / max((ta + tc) ** 2, 1e-12),
        sum_pt_top5=sum(pt[:5]),
        n_dr_0p2_0p4=sum(1 for i in real if 0.2 <= dr[i] < 0.4),
        sum_pt=tot,
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
        - 0.15625 * grid(0, max(0.0, -0.139
            + 59.6 * max(0.0, 0.033 - Q.centroid_offset)
            + 262.0 * max(0.0, 0.013 - Q.girth2)
            - 0.375 * max(0.0, 29.0 - Q.mass)
            - 0.0518 * max(0.0, 74.0 - Q.mass)
            - 0.0332 * max(0.0, Q.sum_pt - 880.0)
            - 743.0 * max(0.0, 0.005 - Q.width)
            + 4970.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
            - 1980.0 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
            + 2.47 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
            - 0.00141 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
            + 0.00154 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        ))
        + 0.390625 * grid(1, max(0.0, 0.55
            + 0.123 * max(0.0, Q.pt_7 - 32.0)
            - 259.0 * max(0.0, 0.01 - Q.width)
            + 10900.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
            - 238.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
            + 6760.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
            - 0.00168 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        ))
        + 0.4296875 * grid(2, max(0.0, 1.33
            - 7.67 * max(0.0, Q.LHA - 0.13)
            + 317.0 * max(0.0, 0.005 - Q.lam1)
            + 4.3 * max(0.0, 6.4 - Q.log_sum_pt)
            + 0.224 * max(0.0, Q.pt_7 - 28.0)
            + 0.00698 * max(0.0, 780.0 - Q.sum_pt)
            - 42.8 * max(0.0, Q.z_7 - 0.047)
            - 429.0 * max(0.0, 0.018 - Q.z_7)
            - 2.23 * max(0.0, Q.pt_7 - 28.0) * max(0.0, 0.054 - Q.C2)
            - 0.584 * max(0.0, Q.pt_7 - 28.0) * max(0.0, Q.max_dr - 0.07)
        ))
        - 0.03125 * grid(4, max(0.0, 0.821
            - 105.0 * max(0.0, Q.C2 - 0.063)
            - 9010.0 * max(0.0, 0.00042 - Q.lam2)
            + 0.144 * max(0.0, 55.0 - Q.mass)
            - 184.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 29.4 * max(0.0, 0.27 - Q.tau21)
            + 427.0 * max(0.0, Q.width - 0.0016)
            + 4.15 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
            - 0.848 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        ))
        - 0.1875 * grid(5, max(0.0, 0.31
            + 59.6 * max(0.0, 0.04 - Q.e2)
            - 71.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 186.0 * max(0.0, 0.035 - Q.z_7)
            + 91.6 * max(0.0, 0.066 - Q.z_7)
            - 132.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 874000.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
            + 57000.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
            - 3020.0 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        ))
        + 0.109375 * grid(6, max(0.0, 9.45
            + 124.0 * max(0.0, Q.girth - 0.09)
            - 1280.0 * max(0.0, Q.lam2 - 0.0027)
            - 2250.0 * max(0.0, 0.00066 - Q.lam2)
            - 70.5 * max(0.0, Q.mass_over_sum_pt - 0.0089)
            + 12.9 * Q.max_dr
            - 991.0 * max(0.0, 0.012 - Q.width)
            + 17000.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
            + 744.0 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
            + 0.267 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
            + 1250.0 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
            + 838.0 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
            + 0.0882 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
            - 67.3 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        ))
        + 0.171875 * grid(9, max(0.0, -1.18
            + 129.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 1460.0 * max(0.0, Q.lam2 - 0.0015)
            - 5.02 * max(0.0, Q.log_sum_pt - 6.3)
            - 0.101 * max(0.0, 25.0 - Q.mass)
            + 1.2 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
            + 7350.0 * max(0.0, 0.00029 - Q.width)
            + 2350.0 * max(0.0, 0.0065 - Q.width)
            + 57.9 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
            + 0.172 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
            - 7340.0 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
            - 69600.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        ))
    )


def logit_q(Q):
    return (0.03125
        - 0.09375 * grid(4, max(0.0, 0.821
            - 105.0 * max(0.0, Q.C2 - 0.063)
            - 9010.0 * max(0.0, 0.00042 - Q.lam2)
            + 0.144 * max(0.0, 55.0 - Q.mass)
            - 184.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 29.4 * max(0.0, 0.27 - Q.tau21)
            + 427.0 * max(0.0, Q.width - 0.0016)
            + 4.15 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
            - 0.848 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        ))
        + 0.046875 * grid(5, max(0.0, 0.31
            + 59.6 * max(0.0, 0.04 - Q.e2)
            - 71.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 186.0 * max(0.0, 0.035 - Q.z_7)
            + 91.6 * max(0.0, 0.066 - Q.z_7)
            - 132.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 874000.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
            + 57000.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
            - 3020.0 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        ))
        + 0.125 * grid(6, max(0.0, 9.45
            + 124.0 * max(0.0, Q.girth - 0.09)
            - 1280.0 * max(0.0, Q.lam2 - 0.0027)
            - 2250.0 * max(0.0, 0.00066 - Q.lam2)
            - 70.5 * max(0.0, Q.mass_over_sum_pt - 0.0089)
            + 12.9 * Q.max_dr
            - 991.0 * max(0.0, 0.012 - Q.width)
            + 17000.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
            + 744.0 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
            + 0.267 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
            + 1250.0 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
            + 838.0 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
            + 0.0882 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
            - 67.3 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        ))
        + 0.0625 * grid(8, max(0.0, 0.139
            - 107.0 * max(0.0, 0.063 - Q.girth)
            + 2310.0 * max(0.0, 0.0049 - Q.width)
            - 117000.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
            + 845.0 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
            - 9.9 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
            - 71600.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        ))
        + 0.25390625 * grid(9, max(0.0, -1.18
            + 129.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 1460.0 * max(0.0, Q.lam2 - 0.0015)
            - 5.02 * max(0.0, Q.log_sum_pt - 6.3)
            - 0.101 * max(0.0, 25.0 - Q.mass)
            + 1.2 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
            + 7350.0 * max(0.0, 0.00029 - Q.width)
            + 2350.0 * max(0.0, 0.0065 - Q.width)
            + 57.9 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
            + 0.172 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
            - 7340.0 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
            - 69600.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        ))
        - 0.125 * grid(10, max(0.0, 2.67
            + 31.5 * max(0.0, Q.C2 - 0.059)
            + 79.2 * max(0.0, Q.e2 - 0.037)
            - 41.9 * max(0.0, 0.037 - Q.e2)
            - 865.0 * max(0.0, 0.0018 - Q.girth2)
            + 532.0 * max(0.0, Q.lam2 - 0.00016)
            - 4.67 * max(0.0, 6.3 - Q.log_sum_pt)
            + 0.599 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
            + 2.28 * max(0.0, 0.34 - Q.z_dr_0p05_0p1)
            - 40.2 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21)
            + 337.0 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4)
            + 2380.0 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21)
        ))
        + 0.0625 * grid(15, max(0.0, 1.79
            - 40.2 * max(0.0, Q.LHA - 0.34)
            + 176.0 * max(0.0, 0.041 - Q.e2)
            + 2180.0 * max(0.0, 0.0065 - Q.lam1)
            - 1210.0 * max(0.0, 0.0081 - Q.lam1)
            + 37.7 * max(0.0, Q.log_sum_pt - 6.9)
            - 2050.0 * max(0.0, 0.0069 - Q.width)
            - 6.46 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
            + 32.8 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
            - 7460.0 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        ))
    )


def logit_W(Q):
    return (-0.125
        + 0.34375 * grid(0, max(0.0, -0.139
            + 59.6 * max(0.0, 0.033 - Q.centroid_offset)
            + 262.0 * max(0.0, 0.013 - Q.girth2)
            - 0.375 * max(0.0, 29.0 - Q.mass)
            - 0.0518 * max(0.0, 74.0 - Q.mass)
            - 0.0332 * max(0.0, Q.sum_pt - 880.0)
            - 743.0 * max(0.0, 0.005 - Q.width)
            + 4970.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
            - 1980.0 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
            + 2.47 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
            - 0.00141 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
            + 0.00154 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        ))
        - 0.03125 * grid(1, max(0.0, 0.55
            + 0.123 * max(0.0, Q.pt_7 - 32.0)
            - 259.0 * max(0.0, 0.01 - Q.width)
            + 10900.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
            - 238.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
            + 6760.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
            - 0.00168 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        ))
        - 0.5 * grid(3, max(0.0, 0.508
            + 19.8 * max(0.0, Q.LHA - 0.27)
            + 22.5 * max(0.0, Q.centroid_offset - 0.013)
            + 188.0 * max(0.0, 0.047 - Q.e2)
            - 1040.0 * max(0.0, 0.01 - Q.girth2)
            - 189.0 * max(0.0, Q.lam1 - 0.016)
            - 109.0 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        ))
        - 0.3125 * grid(6, max(0.0, 9.45
            + 124.0 * max(0.0, Q.girth - 0.09)
            - 1280.0 * max(0.0, Q.lam2 - 0.0027)
            - 2250.0 * max(0.0, 0.00066 - Q.lam2)
            - 70.5 * max(0.0, Q.mass_over_sum_pt - 0.0089)
            + 12.9 * Q.max_dr
            - 991.0 * max(0.0, 0.012 - Q.width)
            + 17000.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
            + 744.0 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
            + 0.267 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
            + 1250.0 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
            + 838.0 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
            + 0.0882 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
            - 67.3 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        ))
        + 0.21875 * grid(7, max(0.0, 3.96
            + 132.0 * max(0.0, 0.038 - Q.e2)
            + 1430.0 * max(0.0, 0.0011 - Q.e2_sq)
            - 44.1 * max(0.0, 0.087 - Q.girth)
            - 442.0 * max(0.0, Q.girth2 - 0.0046)
            - 0.329 * max(0.0, Q.mass - 80.4)
            + 150.0 * max(0.0, Q.mass_over_sum_pt - 0.073)
            - 202.0 * max(0.0, Q.mass_over_sum_pt - 0.09)
            - 7.33 * max(0.0, 0.19 - Q.planar_flow)
            - 2730.0 * max(0.0, 0.00069 - Q.width)
            - 1020.0 * max(0.0, 0.0056 - Q.width)
            - 7.69 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0)
            + 11800.0 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
            + 0.0363 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0)
            - 4770.0 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
            - 0.0886 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow)
        ))
        - 0.25 * grid(8, max(0.0, 0.139
            - 107.0 * max(0.0, 0.063 - Q.girth)
            + 2310.0 * max(0.0, 0.0049 - Q.width)
            - 117000.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
            + 845.0 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
            - 9.9 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
            - 71600.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        ))
        - 0.03125 * grid(9, max(0.0, -1.18
            + 129.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 1460.0 * max(0.0, Q.lam2 - 0.0015)
            - 5.02 * max(0.0, Q.log_sum_pt - 6.3)
            - 0.101 * max(0.0, 25.0 - Q.mass)
            + 1.2 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
            + 7350.0 * max(0.0, 0.00029 - Q.width)
            + 2350.0 * max(0.0, 0.0065 - Q.width)
            + 57.9 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
            + 0.172 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
            - 7340.0 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
            - 69600.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        ))
        + 0.375 * grid(11, max(0.0, -2.21
            - 22.1 * max(0.0, 0.15 - Q.LHA)
            + 136.0 * max(0.0, 0.05 - Q.centroid_offset)
            - 51.6 * max(0.0, Q.girth - 0.074)
            - 127.0 * max(0.0, 0.088 - Q.girth)
            - 668.0 * max(0.0, 0.0038 - Q.width)
            + 1350.0 * max(0.0, 0.0088 - Q.width)
            - 959.0 * max(0.0, 0.39 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
        ))
        + 0.0703125 * grid(13, max(0.0, 2.17
            - 66.1 * max(0.0, Q.C2 - 0.065)
            - 82.4 * max(0.0, 0.047 - Q.e2)
            + 83.1 * max(0.0, 0.14 - Q.girth)
            - 0.017 * max(0.0, Q.sum_pt - 980.0)
            - 48.0 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
            - 1.72 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
            + 0.00048 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
            - 24.9 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
        ))
        - 0.75 * grid(14, max(0.0, -0.0515
            + 116.0 * max(0.0, 0.038 - Q.e2)
            - 88.8 * max(0.0, 0.091 - Q.girth)
            + 741.0 * max(0.0, 0.014 - Q.girth2)
            - 61.6 * max(0.0, Q.lam1 - 0.0075)
            - 0.0511 * max(0.0, 80.4 - Q.mass)
            - 20.6 * max(0.0, 0.18 - Q.max_dr)
            - 1170.0 * max(0.0, 0.0075 - Q.width)
            + 1.68 * max(0.0, 0.64 - Q.z_dr_0p05_0p1)
            + 8210.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
            + 16900.0 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr)
            + 1070.0 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
            - 1220.0 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
            - 194.0 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
            - 454.0 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
            + 572.0 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
            - 4740.0 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow)
            - 73.2 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2)
        ))
        - 0.6875 * grid(15, max(0.0, 1.79
            - 40.2 * max(0.0, Q.LHA - 0.34)
            + 176.0 * max(0.0, 0.041 - Q.e2)
            + 2180.0 * max(0.0, 0.0065 - Q.lam1)
            - 1210.0 * max(0.0, 0.0081 - Q.lam1)
            + 37.7 * max(0.0, Q.log_sum_pt - 6.9)
            - 2050.0 * max(0.0, 0.0069 - Q.width)
            - 6.46 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
            + 32.8 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
            - 7460.0 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        ))
    )


def logit_Z(Q):
    return (-0.09375
        + 0.125 * grid(1, max(0.0, 0.55
            + 0.123 * max(0.0, Q.pt_7 - 32.0)
            - 259.0 * max(0.0, 0.01 - Q.width)
            + 10900.0 * max(0.0, 0.012 - Q.lam1) * max(0.0, Q.centroid_offset - 0.018)
            - 238.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.028 - Q.centroid_offset)
            + 6760.0 * max(0.0, Q.log_sum_pt - 6.5) * max(0.0, 0.0015 - Q.lam2)
            - 0.00168 * max(0.0, Q.pt_7 - 34.0) * max(0.0, 98.0 - Q.mass)
        ))
        - 0.5625 * grid(3, max(0.0, 0.508
            + 19.8 * max(0.0, Q.LHA - 0.27)
            + 22.5 * max(0.0, Q.centroid_offset - 0.013)
            + 188.0 * max(0.0, 0.047 - Q.e2)
            - 1040.0 * max(0.0, 0.01 - Q.girth2)
            - 189.0 * max(0.0, Q.lam1 - 0.016)
            - 109.0 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        ))
        + 0.078125 * grid(4, max(0.0, 0.821
            - 105.0 * max(0.0, Q.C2 - 0.063)
            - 9010.0 * max(0.0, 0.00042 - Q.lam2)
            + 0.144 * max(0.0, 55.0 - Q.mass)
            - 184.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 29.4 * max(0.0, 0.27 - Q.tau21)
            + 427.0 * max(0.0, Q.width - 0.0016)
            + 4.15 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
            - 0.848 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        ))
        + 0.015625 * grid(5, max(0.0, 0.31
            + 59.6 * max(0.0, 0.04 - Q.e2)
            - 71.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 186.0 * max(0.0, 0.035 - Q.z_7)
            + 91.6 * max(0.0, 0.066 - Q.z_7)
            - 132.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 874000.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
            + 57000.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
            - 3020.0 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        ))
        - 0.375 * grid(6, max(0.0, 9.45
            + 124.0 * max(0.0, Q.girth - 0.09)
            - 1280.0 * max(0.0, Q.lam2 - 0.0027)
            - 2250.0 * max(0.0, 0.00066 - Q.lam2)
            - 70.5 * max(0.0, Q.mass_over_sum_pt - 0.0089)
            + 12.9 * Q.max_dr
            - 991.0 * max(0.0, 0.012 - Q.width)
            + 17000.0 * max(0.0, Q.centroid_offset - 0.0059) * max(0.0, 0.0026 - Q.lam2)
            + 744.0 * max(0.0, 0.049 - Q.e2) * max(0.0, Q.z_dr_0p1_0p2 - 0.18)
            + 0.267 * max(0.0, 6.5 - Q.log_sum_pt) * max(0.0, 44.0 - Q.pt_7)
            + 1250.0 * max(0.0, 6.3 - Q.log_sum_pt) * max(0.0, 0.071 - Q.z_7)
            + 838.0 * max(0.0, 6.7 - Q.log_sum_pt) * max(0.0, 0.049 - Q.z_7)
            + 0.0882 * max(0.0, 42.0 - Q.mass) * max(0.0, 0.94 - Q.z_dr_0p05_0p1)
            - 67.3 * max(0.0, Q.mass_over_sum_pt - 0.015) * max(0.0, 0.55 - Q.tau32)
        ))
        + 0.46875 * grid(7, max(0.0, 3.96
            + 132.0 * max(0.0, 0.038 - Q.e2)
            + 1430.0 * max(0.0, 0.0011 - Q.e2_sq)
            - 44.1 * max(0.0, 0.087 - Q.girth)
            - 442.0 * max(0.0, Q.girth2 - 0.0046)
            - 0.329 * max(0.0, Q.mass - 80.4)
            + 150.0 * max(0.0, Q.mass_over_sum_pt - 0.073)
            - 202.0 * max(0.0, Q.mass_over_sum_pt - 0.09)
            - 7.33 * max(0.0, 0.19 - Q.planar_flow)
            - 2730.0 * max(0.0, 0.00069 - Q.width)
            - 1020.0 * max(0.0, 0.0056 - Q.width)
            - 7.69 * max(0.0, Q.centroid_offset - 0.031) * max(0.0, Q.pt_0 - 370.0)
            + 11800.0 * max(0.0, Q.girth2 - 0.0047) * max(0.0, Q.eccentricity - 0.94)
            + 0.0363 * max(0.0, 0.18 - Q.planar_flow) * max(0.0, Q.sum_pt - 590.0)
            - 4770.0 * max(0.0, 0.22 - Q.planar_flow) * max(0.0, Q.width - 0.0078)
            - 0.0886 * max(0.0, 45.0 - Q.pt_7) * max(0.0, 0.6 - Q.planar_flow)
        ))
        - 0.03125 * grid(9, max(0.0, -1.18
            + 129.0 * max(0.0, 0.019 - Q.centroid_offset)
            + 1460.0 * max(0.0, Q.lam2 - 0.0015)
            - 5.02 * max(0.0, Q.log_sum_pt - 6.3)
            - 0.101 * max(0.0, 25.0 - Q.mass)
            + 1.2 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
            + 7350.0 * max(0.0, 0.00029 - Q.width)
            + 2350.0 * max(0.0, 0.0065 - Q.width)
            + 57.9 * max(0.0, Q.girth2 - 0.019) * max(0.0, 30.0 - Q.pt_7)
            + 0.172 * max(0.0, 56.0 - Q.mass) * max(0.0, 6.8 - Q.log_sum_pt)
            - 7340.0 * max(0.0, 0.0091 - Q.width) * max(0.0, Q.C2 - 0.026)
            - 69600.0 * max(0.0, 0.0061 - Q.width) * max(0.0, Q.centroid_offset - 0.0016)
        ))
        + 0.0546875 * grid(13, max(0.0, 2.17
            - 66.1 * max(0.0, Q.C2 - 0.065)
            - 82.4 * max(0.0, 0.047 - Q.e2)
            + 83.1 * max(0.0, 0.14 - Q.girth)
            - 0.017 * max(0.0, Q.sum_pt - 980.0)
            - 48.0 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
            - 1.72 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
            + 0.00048 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
            - 24.9 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
        ))
        + 0.375 * grid(14, max(0.0, -0.0515
            + 116.0 * max(0.0, 0.038 - Q.e2)
            - 88.8 * max(0.0, 0.091 - Q.girth)
            + 741.0 * max(0.0, 0.014 - Q.girth2)
            - 61.6 * max(0.0, Q.lam1 - 0.0075)
            - 0.0511 * max(0.0, 80.4 - Q.mass)
            - 20.6 * max(0.0, 0.18 - Q.max_dr)
            - 1170.0 * max(0.0, 0.0075 - Q.width)
            + 1.68 * max(0.0, 0.64 - Q.z_dr_0p05_0p1)
            + 8210.0 * max(0.0, 0.014 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
            + 16900.0 * max(0.0, Q.lam1 - 0.0064) * max(0.0, 0.16 - Q.max_dr)
            + 1070.0 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.011)
            - 1220.0 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, Q.centroid_offset - 0.018)
            - 194.0 * max(0.0, 0.12 - Q.planar_flow) * max(0.0, 0.17 - Q.max_dr)
            - 454.0 * max(0.0, 0.0087 - Q.width) * max(0.0, 1.0 - Q.D2)
            + 572.0 * max(0.0, 0.0083 - Q.width) * max(0.0, 6.9 - Q.log_sum_pt)
            - 4740.0 * max(0.0, 0.0078 - Q.width) * max(0.0, 0.097 - Q.planar_flow)
            - 73.2 * max(0.0, 0.63 - Q.z_dr_0p05_0p1) * max(0.0, 0.067 - Q.C2)
        ))
        - 0.15625 * grid(15, max(0.0, 1.79
            - 40.2 * max(0.0, Q.LHA - 0.34)
            + 176.0 * max(0.0, 0.041 - Q.e2)
            + 2180.0 * max(0.0, 0.0065 - Q.lam1)
            - 1210.0 * max(0.0, 0.0081 - Q.lam1)
            + 37.7 * max(0.0, Q.log_sum_pt - 6.9)
            - 2050.0 * max(0.0, 0.0069 - Q.width)
            - 6.46 * max(0.0, 0.31 - Q.tau21) * max(0.0, 0.68 - Q.z_dr_0p05_0p1)
            + 32.8 * max(0.0, 0.29 - Q.tau21) * max(0.0, 0.2 - Q.z_dr_0p2_0p4)
            - 7460.0 * max(0.0, 0.0065 - Q.width) * max(0.0, Q.log_sum_pt - 6.9)
        ))
    )


def logit_t(Q):
    return (1.34375
        + 0.015625 * grid(0, max(0.0, -0.139
            + 59.6 * max(0.0, 0.033 - Q.centroid_offset)
            + 262.0 * max(0.0, 0.013 - Q.girth2)
            - 0.375 * max(0.0, 29.0 - Q.mass)
            - 0.0518 * max(0.0, 74.0 - Q.mass)
            - 0.0332 * max(0.0, Q.sum_pt - 880.0)
            - 743.0 * max(0.0, 0.005 - Q.width)
            + 4970.0 * max(0.0, 0.018 - Q.girth2) * max(0.0, Q.eccentricity - 0.97)
            - 1980.0 * max(0.0, 0.0062 - Q.lam1) * max(0.0, 0.95 - Q.D2)
            + 2.47 * max(0.0, 65.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.008)
            - 0.00141 * max(0.0, 66.0 - Q.mass) * max(0.0, 39.0 - Q.pt_7)
            + 0.00154 * max(0.0, Q.sum_pt - 860.0) * max(0.0, 33.0 - Q.pt_7)
        ))
        + 0.0625 * grid(3, max(0.0, 0.508
            + 19.8 * max(0.0, Q.LHA - 0.27)
            + 22.5 * max(0.0, Q.centroid_offset - 0.013)
            + 188.0 * max(0.0, 0.047 - Q.e2)
            - 1040.0 * max(0.0, 0.01 - Q.girth2)
            - 189.0 * max(0.0, Q.lam1 - 0.016)
            - 109.0 * max(0.0, Q.mass_over_sum_pt - 0.071) * max(0.0, 0.55 - Q.tau32)
        ))
        + 0.125 * grid(4, max(0.0, 0.821
            - 105.0 * max(0.0, Q.C2 - 0.063)
            - 9010.0 * max(0.0, 0.00042 - Q.lam2)
            + 0.144 * max(0.0, 55.0 - Q.mass)
            - 184.0 * max(0.0, Q.mass_over_sum_pt - 0.086)
            + 29.4 * max(0.0, 0.27 - Q.tau21)
            + 427.0 * max(0.0, Q.width - 0.0016)
            + 4.15 * max(0.0, Q.C2 - -0.00087) * max(0.0, Q.pt_7 - 38.0)
            - 0.848 * max(0.0, 0.26 - Q.tau21) * max(0.0, 57.0 - Q.mass)
        ))
        - 0.25 * grid(5, max(0.0, 0.31
            + 59.6 * max(0.0, 0.04 - Q.e2)
            - 71.3 * max(0.0, Q.log_sum_pt - 6.9)
            + 186.0 * max(0.0, 0.035 - Q.z_7)
            + 91.6 * max(0.0, 0.066 - Q.z_7)
            - 132.0 * max(0.0, 0.22 - Q.LHA) * max(0.0, 6.8 - Q.log_sum_pt)
            + 874000.0 * max(0.0, 0.025 - Q.e2) * max(0.0, 9.2e-05 - Q.lam2)
            + 57000.0 * max(0.0, 0.0026 - Q.width) * max(0.0, 0.021 - Q.centroid_offset)
            - 3020.0 * max(0.0, 0.072 - Q.z_7) * max(0.0, 0.032 - Q.centroid_offset)
        ))
        + 0.1875 * grid(8, max(0.0, 0.139
            - 107.0 * max(0.0, 0.063 - Q.girth)
            + 2310.0 * max(0.0, 0.0049 - Q.width)
            - 117000.0 * max(0.0, 0.17 - Q.LHA) * max(0.0, 0.00039 - Q.lam2)
            + 845.0 * max(0.0, 0.0051 - Q.girth2) * max(0.0, 0.54 - Q.planar_flow)
            - 9.9 * max(0.0, 20.0 - Q.mass) * max(0.0, Q.centroid_offset - 0.015)
            - 71600.0 * max(0.0, 0.0051 - Q.width) * max(0.0, Q.centroid_offset - 0.0014)
        ))
        + 0.375 * grid(10, max(0.0, 2.67
            + 31.5 * max(0.0, Q.C2 - 0.059)
            + 79.2 * max(0.0, Q.e2 - 0.037)
            - 41.9 * max(0.0, 0.037 - Q.e2)
            - 865.0 * max(0.0, 0.0018 - Q.girth2)
            + 532.0 * max(0.0, Q.lam2 - 0.00016)
            - 4.67 * max(0.0, 6.3 - Q.log_sum_pt)
            + 0.599 * max(0.0, Q.n_dr_0p2_0p4 - 1.5)
            + 2.28 * max(0.0, 0.34 - Q.z_dr_0p05_0p1)
            - 40.2 * max(0.0, Q.LHA - 0.29) * max(0.0, 0.69 - Q.tau21)
            + 337.0 * max(0.0, Q.eccentricity - 0.9) * max(0.0, 0.041 - Q.z_dr_0p2_0p4)
            + 2380.0 * max(0.0, Q.lam2 - 0.00022) * max(0.0, 0.52 - Q.tau21)
        ))
        - 0.5 * grid(12, max(0.0, -1.12
            - 107.0 * max(0.0, Q.e2 - 0.062)
            + 0.0826 * max(0.0, Q.mass - 91.2)
            + 310.0 * max(0.0, Q.width - 0.018)
            + 15500.0 * max(0.0, Q.girth2 - 0.019) * max(0.0, Q.lam2 - 0.0016)
        ))
        - 0.40625 * grid(13, max(0.0, 2.17
            - 66.1 * max(0.0, Q.C2 - 0.065)
            - 82.4 * max(0.0, 0.047 - Q.e2)
            + 83.1 * max(0.0, 0.14 - Q.girth)
            - 0.017 * max(0.0, Q.sum_pt - 980.0)
            - 48.0 * max(0.0, 0.15 - Q.girth) * max(0.0, 6.7 - Q.log_sum_pt)
            - 1.72 * max(0.0, 0.15 - Q.girth) * max(0.0, 35.0 - Q.pt_7)
            + 0.00048 * max(0.0, Q.sum_pt_top5 - 660.0) * max(0.0, 48.0 - Q.pt_7)
            - 24.9 * max(0.0, 0.52 - Q.tau21) * max(0.0, Q.max_dr - 0.031)
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
