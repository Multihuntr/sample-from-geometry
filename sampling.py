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
    """
    Calculate the point masses and densities between those point masses
    across the first dimension
    """
    xlo, ylo, xhi, yhi = shp.bounds
    xs = shapely.get_coordinates(shp)[:, 0]
    xs = np.unique(xs)
    xs = np.sort(xs)

    line_coords = np.array([[[x, ylo], [x, yhi]] for x in xs])

    lines = shapely.linestrings(line_coords)
    shapely.prepare(lines)
    inter_lines = shapely.intersection(shp, lines)

    point_mass = shapely.length(inter_lines)
    h = xs[1:] - xs[:-1]
    a = point_mass[:-1]
    b = point_mass[1:]
    #  - The mass changes from x_0 to x_1 linearly; hence the mass for the segment is a trapezoid
    densities = (a + b) / 2 * h
    return xs, point_mass, densities


def _coverage(shp):
    return shp.area / shapely.box(*shp.bounds).area


def est_rejection_time(c, slope=0.04, bias=0):
    # Slope and bias estimated through an external process
    return 1 / c * slope + bias


def est_closed_time(n, slope=0.017, bias=2.4):
    # Slope and bias estimated through an external process
    return n * np.log(n) * slope + bias


def sample_uniform_from_geom(shp, n=1, rng=None):
    # Choose algorithm based on estimated times for each.
    rej_time = est_rejection_time(_coverage(shp))
    n_coords = len(shapely.get_coordinates(shp))
    closed_time = est_closed_time(n_coords)
    if rej_time < closed_time:
        return sample_uniform_from_geom_rejection(shp, n, rng)
    else:
        return sample_uniform_from_geom_closed(shp, n, rng)


def sample_uniform_from_geom_closed(shp, n=1, rng=None):
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
    square = slope == 0
    x_t = np.zeros_like(x_lo)

    # Default case; not square
    d_x = np.sqrt(d_lo[~square] ** 2 + 2 * slope[~square] * (d[~square] - cd_lo[~square]))
    x_t[~square] = x_lo[~square] + (d_x - d_lo[~square]) / slope[~square]
    # Note: at the start we assumed x \in [0, x_hi-x_lo]; we add in x_lo to make x_t \in [x_lo, x_hi]

    # Edge case; square; above formula gives invalid values
    # In such cases it is sufficient to do simple linear interpolation.
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


def sample_uniform_from_geom_rejection(shp, n=1, rng=None):
    shapely.prepare(shp)  # Very important optimisation; like 10x faster

    # Repeatedly sample points until we have enough points
    found_points = np.array([]).reshape((0, 2))
    xlo, ylo, xhi, yhi = shp.bounds
    c = _coverage(shp)
    while len(found_points) < n:
        # Estimate the number to sample based on geometry coverage and missing points
        to_sample = int((n - len(found_points)) / c * 1.05)

        # Sample new points
        new_points = rng.uniform((xlo, ylo), (xhi, yhi), size=(to_sample, 2))

        # Add all new points that are inside the geometry to our list of found points
        new_points_shp = shapely.points(new_points)
        include = shapely.contains(shp, new_points_shp)
        found_points = np.concatenate([found_points, new_points[include]])

    # Return just the first n (we may have sampled more than we needed)
    return found_points[:n]
