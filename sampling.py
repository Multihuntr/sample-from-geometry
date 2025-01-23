import shapely
import numpy as np

def interp1d(a, b, t=0.5):
    """
    Calculates the value at t between a and b.
    t=0 returns a, t=1 returns b, t=0.5 returns the average of a and b
    """
    return a * (1 - t) + b * t


def interp1d_inv(a, c, b):
    """
    Calculates where c is between a and b
    c=a returns 0, c=b returns 1, c=(a+b)/2 returns 0.5
    """
    return (c - a) / (b - a)


def density_1d(shp):
    xlo, ylo, xhi, yhi = shp.bounds
    xs = shapely.get_coordinates(shp)[:, 0]
    xs = np.unique(xs)
    xs = np.sort(xs)

    line_coords = np.array([[[x, ylo], [x, yhi]] for x in xs])

    lines = shapely.linestrings(line_coords)
    inter_lines = shapely.intersection(shp, lines)

    point_mass = shapely.length(inter_lines)
    h = xs[1:] - xs[:-1]
    a = point_mass[:-1]
    b = point_mass[1:]
    #  - The mass changes from x_0 to x_1 linearly; hence the mass for the segment is a trapezoid
    densities = (a + b) / 2 * h
    return xs, point_mass, densities


def sample_uniform_from_geom(shp, n=1, rng=None):
    """Sample uniformly from an arbitrary 2D geometry, n times."""

    shapely.prepare(shp)

    # Calculate an unnormalised density function along one dimension
    xs, point_mass, densities = density_1d(shp)

    # Sample a point along this density function
    #  - sample density
    if rng is None:
        rng = np.random.default_rng()
    d = rng.uniform(0, densities.sum(), size=(n,))

    #  - interpolate to x_t from trapezoidal distribution
    cumul_density = np.concatenate([[0], np.cumsum(densities)])
    hi_idx = np.searchsorted(cumul_density, d)
    d_lo = point_mass[hi_idx - 1]
    d_hi = point_mass[hi_idx]
    cd_lo = cumul_density[hi_idx - 1]
    cd_hi = cumul_density[hi_idx]
    x_lo = xs[hi_idx - 1]
    x_hi = xs[hi_idx]
    # We can define the piecewise cdf as follows. Given some x \in [0, x_hi-x_lo]:
    #   slope(x) = (d_hi-d_lo)/(x_hi-x_lo)
    #   pdf(x) = d_lo + x*slope
    #   => cdf(x) = integral(d_lo + x*slope)
    #             = 1/2*slope*x^2 + x*d_lo + C
    #   at x=0, cdf(0) = C = cd_lo, by construction
    #   => cdf(x) = 1/2*slope*x^2 + x*d_lo + cd_lo

    # As an aside, we can verify this derivation using:
    #   at x=x_diff, cdf(x_diff) = cd_hi = 1/2*x_diff^2*slope + x_diff*d_lo + cd_lo
    # Or, in code (left off for performance reasons, but does work):
    #   assert all((1 / 2 * (x_hi - x_lo) ** 2 * slope + (x_hi - x_lo) * d_lo + cd_lo) == cd_hi)

    # In any case, to sample from that cdf, we use `d` from above as the cdf(x)
    # and re-arrange to solve for what x would give us that d
    #   => cdf(x) = d = 1/2*slope*x^2 + x*d_lo + cd_lo
    #   => 1/2*slope*x^2 + x*d_lo + cd_lo - d = 0
    #   => x = (-d_lo \pm sqrt(d_lo**2 - 2*slope*(cd_lo - d)))/ slope
    # We only want the positive case, so:
    #   => x = sqrt(d_lo**2 - 2*slope*(cd_lo-d) - d_lo) / slope
    slope = (d_hi - d_lo) / (x_hi - x_lo)
    with np.errstate(invalid="ignore"):
        x_t = x_lo + (np.sqrt(d_lo**2 + 2 * slope * (d - cd_lo)) - d_lo) / slope
    # Note: at the start we assumed x \in [0, x_hi-x_lo]; we add in x_lo to make x_t \in [x_lo, x_hi]
    # For perfectly square cases, the slope is zero, and we get invalid values
    # In such cases it is sufficient to do simple linear interpolation.
    square = slope == 0
    t = interp1d_inv(cd_lo[square], d[square], cd_hi[square])
    x_t[square] = interp1d(x_lo[square], x_hi[square], t)

    # Create vertical lines at x_t
    xlo, ylo, xhi, yhi = shp.bounds
    v_lines = shapely.linestrings([[[x, ylo], [x, yhi]] for x in x_t])

    # Sample along the intersection of those lines and the polygon
    v_valid = shapely.intersection(shp, v_lines)
    e = rng.uniform(0, shapely.length(v_valid))
    y_u = np.array([line.interpolate(u).y for u, line in zip(e, v_valid)])

    points = np.stack([x_t, y_u], axis=1)

    return points
