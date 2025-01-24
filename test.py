import time
import shapely
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import sampling

# fmt: off
RANDOM_SHAPE = shapely.Polygon([
    [16, 16], [57, 79], [77, 25], [127, 44], [180, 26],
    [170, 94], [120, 80], [83, 113], [153, 143], [172, 180],
    [88, 146], [116, 181], [20, 170], [49, 132], [18, 103],
])
# fmt: on


def complicate_shape(shp, seg_size, rng, std=2):
    shp = shp.segmentize(seg_size)
    # add small noise to coords
    coords = shapely.get_coordinates(shp)
    coords += rng.normal(0, std, size=coords.shape)
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
    points = sampling.sample_uniform_from_geom_closed(tri, 10000, rng)
    debug_draw_polygons_and_points([tri], points, "img/tri.png")


def test_sample_uniform_from_geom_trapezoid():
    trap = shapely.Polygon([[0, 0], [0, 2], [1, 1], [1, 0]])
    rng = np.random.default_rng(seed=7777)
    points = sampling.sample_uniform_from_geom_closed(trap, 10000, rng)
    debug_draw_polygons_and_points([trap], points, "img/trap.png")


def test_sample_uniform_from_geom_star():
    # fmt: off
    star = shapely.Polygon([
        [0, 39], [25, 51], [29, 78], [47, 58], [74, 63],
        [61, 39], [74, 15], [47, 20], [29, 0], [25, 28],
    ])
    # fmt: on
    rng = np.random.default_rng(seed=9911)
    points = sampling.sample_uniform_from_geom_closed(star, 10000, rng)
    debug_draw_polygons_and_points([star], points, "img/star.png")


def test_sample_uniform_from_geom_random_shape():
    rng = np.random.default_rng(seed=6451)
    points = sampling.sample_uniform_from_geom_closed(RANDOM_SHAPE, 10000, rng)
    debug_draw_polygons_and_points([RANDOM_SHAPE], points, "img/random_simple.png")
    assert all(shapely.contains(RANDOM_SHAPE, shapely.points(points))), "points not contained!"


def test_sample_uniform_from_geom_as_hole():
    # Use random shape to cut a hole in outer
    outer = shapely.box(0, 0, 200, 200)
    diff = shapely.difference(outer, RANDOM_SHAPE)

    rng = np.random.default_rng(seed=6451)
    points = sampling.sample_uniform_from_geom_closed(diff, 10000, rng)
    debug_draw_polygons_and_points([outer, RANDOM_SHAPE], points, "img/as_hole.png")
    assert all(shapely.contains(diff, shapely.points(points))), "points not contained!"
    assert not any(shapely.contains(RANDOM_SHAPE, shapely.points(points))), "hole has points!"


def test_sample_uniform_from_geom_complex():
    rng = np.random.default_rng(seed=7545)
    shp = complicate_shape(RANDOM_SHAPE, 1, rng)
    points = sampling.sample_uniform_from_geom_closed(shp, 50000, rng)
    debug_draw_polygons_and_points([shp], points, "img/random_complex.png")
    assert all(shapely.contains(shp, shapely.points(points))), "points not contained!"


def visualise_speed_relationship():
    rng = np.random.default_rng(seed=3474)
    coverages = np.linspace(0.002, 0.01, 10)
    seg_sizes = np.linspace(1.5, 3, 10)
    closed_times = []
    rejection_times = []
    est_correct = 0
    total = 0
    for target_coverage in coverages:
        for seg_size_e in seg_sizes:
            total += 1
            seg_size = 10**seg_size_e
            base_shp = shapely.Polygon(
                [
                    [0, 0],
                    [1000, 0],
                    [1000, 1000],
                    [1000 * (1 - target_coverage), 1000],
                    [1000 * (1 - target_coverage), 1000 * target_coverage],
                    [0, 1000 * target_coverage],
                ]
            )
            d = (1000 * target_coverage) * 0.95 / 2
            shp = shapely.segmentize(base_shp, seg_size)
            coords = shapely.get_coordinates(shp)
            coords += rng.uniform(-d, d, size=coords.shape)
            shp = shapely.Polygon(coords)
            shp = shp.buffer(0)
            c = sampling._coverage(shp)
            n = len(shapely.get_coordinates(shp))
            print(n, c)

            start = time.perf_counter()
            rng_inner = np.random.default_rng(seed=1234)
            for _ in tqdm.tqdm(range(1000)):
                sampling.sample_uniform_from_geom_closed(shp, 100, rng_inner)
            end = time.perf_counter()
            closed_time = end - start
            closed_times.append((n, c, closed_time))

            start = time.perf_counter()
            rng_inner = np.random.default_rng(seed=1234)
            for _ in tqdm.tqdm(range(1000)):
                sampling.sample_uniform_from_geom_rejection(shp, 100, rng_inner)
            end = time.perf_counter()
            rejection_time = end - start
            rejection_times.append((n, c, rejection_time))

            est_clo = sampling.est_closed_time(n)
            est_rej = sampling.est_rejection_time(c)
            if (est_clo < est_rej) == (closed_time < rejection_time):
                est_correct += 1
            print(f"Correct: {est_correct/total*100:.2f}%")
    closed_times = np.array(closed_times)
    rejection_times = np.array(rejection_times)
    is_closed_faster = closed_times[:, -1] < rejection_times[:, -1]

    closed_faster = closed_times[is_closed_faster][:, :2]
    rejection_faster = closed_times[~is_closed_faster][:, :2]

    nlogn = closed_times[:, 0] * np.log(closed_times[:, 0])
    c_inv = 1 / closed_times[:, 1]
    print("n log(n)", nlogn)
    print("1/c", c_inv)
    print("closed_times", closed_times[:, 2])
    print("rejection_times", rejection_times[:, 2])

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    if len(closed_faster) > 0:
        ax.scatter(closed_faster[:, 0], closed_faster[:, 1], label="Closed faster")
    ax.scatter(rejection_faster[:, 0], rejection_faster[:, 1], label="Rejection faster")
    ax.set_title("Relative performance")
    ax.set_xlabel("# vertices (n)")
    ax.set_ylabel("coverage (c)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("img/relative_performance.png", dpi=400)


if __name__ == "__main__":
    test_sample_uniform_from_geom_triangle()
    test_sample_uniform_from_geom_trapezoid()
    test_sample_uniform_from_geom_star()
    test_sample_uniform_from_geom_random_shape()
    test_sample_uniform_from_geom_as_hole()
    test_sample_uniform_from_geom_complex()
    # visualise_speed_relationship()
