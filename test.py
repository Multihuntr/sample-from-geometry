import time
import shapely
import matplotlib.pyplot as plt
import numpy as np

import sampling

RANDOM_SHAPE = shapely.Polygon([
    [16, 16], [57, 79], [77, 25], [127, 44], [180, 26],
    [170, 94], [120, 80], [83, 113], [153, 143], [172, 180],
    [88, 146], [116, 181], [20, 170], [49, 132], [18, 103]
])

def complicate_shape(shp, seg_size, rng):
    shp = shp.segmentize(seg_size)
    # add small noise to coords
    coords = shapely.get_coordinates(shp)
    coords += rng.normal(0, 2, size=coords.shape)
    shp = shapely.Polygon(coords)
    shp = shp.simplify(seg_size / 4)
    shp = shp.buffer(0, cap_style="flat")
    if shp.geom_type == "MultiPolygon":
        shp = sorted(shp.geoms, key=lambda g: g.area)[-1]
    shp = shapely.Polygon(shp.exterior.coords)
    return shp


def debug_draw_polygons_and_points(polygons=None, points=None, fname="debug.png"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if polygons is not None:
        for shp in polygons:
            ax.plot(*shp.exterior.xy)
            for interior in shp.interiors:
                ax.plot(*interior.xy)
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], s=1)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def test_sample_uniform_from_geom_triangle():
    tri = shapely.Polygon([[0, 0], [0, 1], [1, 0]])
    rng = np.random.default_rng(seed=1474)
    points = sampling.sample_uniform_from_geom(tri, 10000, rng)
    debug_draw_polygons_and_points([tri], points, "img/tri.png")


def test_sample_uniform_from_geom_trapezoid():
    trap = shapely.Polygon([[0, 0], [0, 2], [1, 1], [1, 0]])
    rng = np.random.default_rng(seed=7777)
    points = sampling.sample_uniform_from_geom(trap, 10000, rng)
    debug_draw_polygons_and_points([trap], points, "img/trap.png")


def test_sample_uniform_from_geom_star():
    star = shapely.Polygon([
        [0, 39], [25, 51], [29, 78], [47, 58], [74, 63],
        [61, 39], [74, 15], [47, 20], [29, 0], [25, 28]
    ])
    rng = np.random.default_rng(seed=9911)
    points = sampling.sample_uniform_from_geom(star, 10000, rng)
    debug_draw_polygons_and_points([star], points, 'img/star.png')


def test_sample_uniform_from_geom_random_shape():
    rng = np.random.default_rng(seed=6451)
    points = sampling.sample_uniform_from_geom(RANDOM_SHAPE, 10000, rng)
    debug_draw_polygons_and_points([RANDOM_SHAPE], points, 'img/random_simple.png')
    assert all(shapely.contains(RANDOM_SHAPE, shapely.points(points))), "points not contained!"


def test_sample_uniform_from_geom_as_hole():
    # Use random shape to cut a hole in outer
    outer = shapely.box(0, 0, 200, 200)
    diff = shapely.difference(outer, RANDOM_SHAPE)

    rng = np.random.default_rng(seed=6451)
    points = sampling.sample_uniform_from_geom(diff, 10000, rng)
    debug_draw_polygons_and_points([outer, RANDOM_SHAPE], points, 'img/as_hole.png')
    assert all(shapely.contains(diff, shapely.points(points))), "points not contained!"
    assert not any(shapely.contains(RANDOM_SHAPE, shapely.points(points))), "hole has points!"


def test_sample_uniform_from_geom_complex():
    # Making RANDOM_SHAPE more "complex" a shape
    rng = np.random.default_rng(seed=7545)
    shp = complicate_shape(RANDOM_SHAPE, 1, rng)
    points = sampling.sample_uniform_from_geom(shp, 50000, rng)
    debug_draw_polygons_and_points([shp], points, 'img/random_complex.png')
    assert all(shapely.contains(shp, shapely.points(points))), "points not contained!"


def test_sample_uniform_from_geom_time():
    rng = np.random.default_rng(seed=19443)
    samples = np.linspace(2, 5, 6)
    seg_sizes = np.linspace(-1, 0.5, 6)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for seg_size_e in seg_sizes:
        seg_size = 10**seg_size_e
        shp = complicate_shape(RANDOM_SHAPE, seg_size, rng)
        n_coords = len(shapely.get_coordinates(shp))

        ns = []
        times = []
        for e in samples:
            n = int(10**e)

            start = time.perf_counter()
            sampling.sample_uniform_from_geom(shp, n, rng)
            end = time.perf_counter()
            ns.append(n)
            times.append(end - start)

        ax.plot(ns, times, label=f"{n_coords} vertices")
    ax.set_title("Algorithm performance")
    ax.set_xlabel("# samples")
    ax.set_ylabel("Time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig("img/timing.png", dpi=400)


if __name__ == '__main__':
    # test_sample_uniform_from_geom_triangle()
    # test_sample_uniform_from_geom_trapezoid()
    # test_sample_uniform_from_geom_star()
    # test_sample_uniform_from_geom_random_shape()
    # test_sample_uniform_from_geom_as_hole()
    # test_sample_uniform_from_geom_complex()
    test_sample_uniform_from_geom_time()
