import numpy as np
import pyvista as pv


def visualize_cylinders_pyvista(obstacles, radius=0.02):
    plotter = pv.Plotter()
    for start, end in obstacles:
        start, end = np.array(start), np.array(end)
        center = (start + end) / 2
        direction = end - start
        height = np.linalg.norm(direction)
        cyl = pv.Cylinder(center=center,
                          direction=direction,
                          radius=radius,
                          height=height,
                          resolution=50)
        plotter.add_mesh(cyl, color="skyblue", opacity=0.8)
    plotter.show()


def sample_cylinders(num_cyls,
                     length,
                     bbox=[[0.3, 0.6], [-0.6, 0.6], [-0.1, 0.8]],
                     min_gap=0.08,
                     max_trials=5000):
    """
    Sample new cylinders with given lengths and spacing.

    Args:
        num_cyls: number of cylinders to sample
        lengths: array of target lengths
        bbox: [[xmin,xmax],[ymin,ymax],[zmin,zmax]] sampling box
        min_gap: minimum allowed distance between cylinder axes
        max_trials: max sampling attempts per cylinder
    Returns:
        Array of shape (num_cyls, 2, 3)
    """
    cylinders = []
    L = length
    for i in range(num_cyls):
        for trial in range(max_trials):
            # sample center point
            center = np.array([
                np.random.uniform(*bbox[0]),
                np.random.uniform(*bbox[1]),
                np.random.uniform(*bbox[2]),
            ])
            # sample random direction
            v = np.random.normal(size=3)
            v /= np.linalg.norm(v)
            start = center - 0.5 * L * v
            end = center + 0.5 * L * v

            # check inside bounds
            if not all(bbox[d][0] <= start[d] <= bbox[d][1]
                       and bbox[d][0] <= end[d] <= bbox[d][1]
                       for d in range(3)):
                continue

            # check distance from existing cylinders
            ok = True
            for c in cylinders:
                d = min_distance_between_segments(start, end, c[0], c[1])
                if d < min_gap:
                    ok = False
                    break
            if ok:
                cylinders.append([start, end])
                break
        else:
            raise RuntimeError(
                f"Could not place cylinder {i} after {max_trials} trials")

    return np.array(cylinders, dtype=np.float32)


def min_distance_between_segments(p1, q1, p2, q2):
    """Return shortest distance between two 3D segments."""

    def dot(u, v):
        return np.dot(u, v)

    def norm2(v):
        return np.dot(v, v)

    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a, b, c = norm2(u), dot(u, v), norm2(v)
    d, e = dot(u, w), dot(v, w)
    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D
    SMALL_NUM = 1e-9
    if D < SMALL_NUM:
        sN, sD = 0.0, 1.0
        tN, tD = e, c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0
        elif (-d + b) > a: sN = sD
        else:
            sN = (-d + b)
            sD = a
    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)


def create_obstacles() -> np.ndarray:
    while True:
        try:
            cylinders = sample_cylinders(8,
                                         length=1.0,
                                         min_gap=0.015,
                                         max_trials=100)
        except RuntimeError:
            continue
        break
    return cylinders


if __name__ == '__main__':
    new_cyls = sample_cylinders(8, length=1.0, min_gap=0.03)
    print(new_cyls.shape)  # (12, 2, 3)
    visualize_cylinders_pyvista(new_cyls, radius=0.03)
