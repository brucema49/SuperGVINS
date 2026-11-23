#!/usr/bin/env python3
"""Compare world positions reconstructed in one shared ENU frame.

Each pose uses the anchor and yaw recorded at its exact image timestamp in
the GNSS result. Export the reconstructed ECEF and shared ENU coordinates.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NANOSECONDS_PER_SECOND = 1_000_000_000


class TrajectoryError(ValueError):
    """Raised when an input trajectory cannot be read reliably."""


@dataclass
class Trajectory:
    path: str
    timestamps_ns: np.ndarray
    positions: np.ndarray
    invalid_rows: int
    empty_lines: int

    @property
    def count(self):
        return int(len(self.timestamps_ns))


def _parse_timestamp_ns(value, line_number):
    try:
        timestamp_ns = int(value.strip())
    except ValueError as exc:
        raise ValueError("line {}: timestamp is not an integer".format(line_number)) from exc
    if timestamp_ns < np.iinfo(np.int64).min or timestamp_ns > np.iinfo(np.int64).max:
        raise ValueError("line {}: timestamp is outside int64 range".format(line_number))
    return timestamp_ns


def read_trajectory(path):
    """Read valid GVINS rows while preserving their original positions."""
    timestamps = []
    positions = []
    invalid_rows = 0
    empty_lines = 0
    previous_timestamp = None

    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise TrajectoryError("cannot open trajectory {}: {}".format(path, exc)) from exc

    with handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                empty_lines += 1
                continue

            fields = stripped.split(",")
            if fields and fields[-1].strip() == "":
                fields.pop()  # GVINS output ends in a comma.
            if len(fields) != 11:
                invalid_rows += 1
                continue

            try:
                timestamp_ns = _parse_timestamp_ns(fields[0], line_number)
                values = [float(value.strip()) for value in fields[1:]]
            except (ValueError, OverflowError):
                invalid_rows += 1
                continue

            if not all(math.isfinite(value) for value in values):
                invalid_rows += 1
                continue

            # Keep the previous timestamp for each accepted row, matching the
            # existing strict ordering check for duplicate or reset CSVs.
            if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
                order = "repeated" if timestamp_ns == previous_timestamp else "went backwards"
                raise TrajectoryError(
                    "{}: timestamps {} at line {}; trajectory reset or duplicate rows are not supported".format(
                        path, order, line_number
                    )
                )

            quaternion = np.asarray(values[3:7], dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if not math.isfinite(norm) or norm < 1e-12:
                invalid_rows += 1
                continue

            timestamps.append(timestamp_ns)
            positions.append(values[:3])
            previous_timestamp = timestamp_ns

    if not timestamps:
        raise TrajectoryError("{}: no valid trajectory rows".format(path))

    return Trajectory(
        path=str(path),
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64),
        invalid_rows=invalid_rows,
        empty_lines=empty_lines,
    )


def path_length(positions):
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def describe_trajectory(trajectory):
    timestamps = trajectory.timestamps_ns
    span_ns = int(timestamps[-1]) - int(timestamps[0])
    intervals = np.diff(timestamps).astype(np.float64) / NANOSECONDS_PER_SECOND
    median_interval = float(np.median(intervals)) if len(intervals) else None
    return {
        "input": trajectory.path,
        "valid_samples": trajectory.count,
        "invalid_rows_ignored": trajectory.invalid_rows,
        "empty_lines_ignored": trajectory.empty_lines,
        "coverage_start_ns": int(timestamps[0]),
        "coverage_end_ns": int(timestamps[-1]),
        "duration_s": span_ns / NANOSECONDS_PER_SECOND,
        "median_sample_rate_hz": (1.0 / median_interval) if median_interval and median_interval > 0 else None,
    }


def _mark_ends(axis, positions, color, label_prefix):
    axis.scatter([positions[0, 0]], [positions[0, 1]], marker="o", s=34,
                 color=color, zorder=4, label="{} start".format(label_prefix))
    axis.scatter([positions[-1, 0]], [positions[-1, 1]], marker="s", s=34,
                 color=color, zorder=4, label="{} end".format(label_prefix))


def make_shared_enu_plot(out_path, gvins, supergvins):
    figure, axis = plt.subplots(figsize=(8, 8))
    for trajectory, label, color in (
        (gvins, "GVINS", "tab:blue"),
        (supergvins, "SuperGVINS", "tab:orange"),
    ):
        positions = trajectory.enu_positions
        axis.plot(positions[:, 0], positions[:, 1], color=color, linewidth=1.3,
                  label=label)
        _mark_ends(axis, positions, color, label)

    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("East [m]")
    axis.set_ylabel("North [m]")
    axis.set_title("World trajectories in a shared ENU frame")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


@dataclass
class GnssAlignment:
    path: Path
    timestamps_ns: np.ndarray
    logged_ecef: np.ndarray
    yaw_rad: np.ndarray
    anchors_ecef: np.ndarray


def read_gnss_alignment(path):
    """Read the image stamp, ECEF, yaw and anchor columns of gnss_result.csv."""
    path = Path(path)
    stamps, positions, yaws, anchors = [], [], [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            try:
                if len(row) != 14:
                    raise ValueError("expected 14 GNSS result columns")
                stamp = _parse_timestamp_ns(row[0], line)
                position = [float(value) for value in row[2:5]]
                yaw = float(row[5])
                anchor = [float(value) for value in row[11:14]]
                if not all(math.isfinite(value) for value in [*position, yaw, *anchor]):
                    raise ValueError("non-finite position, yaw or anchor")
                if not any(anchor):
                    raise ValueError("ECEF anchor is zero")
                if stamps and stamp <= stamps[-1]:
                    raise ValueError("image timestamps must be strictly increasing")
            except (ValueError, OverflowError) as exc:
                raise TrajectoryError(f"{path}:{line}: {exc}") from exc
            stamps.append(stamp)
            positions.append(position)
            yaws.append(yaw)
            anchors.append(anchor)
    if not stamps:
        raise TrajectoryError(f"{path}: no GNSS alignment rows")
    return GnssAlignment(path, np.asarray(stamps, dtype=np.int64), np.asarray(positions),
                         np.asarray(yaws), np.asarray(anchors))


def ecef_to_enu_rotation(origins_ecef):
    """WGS84 rotation at each ECEF anchor, following gnss_comm::ecef2rotation."""
    x, y, z = np.asarray(origins_ecef).T
    semimajor = 6378137.0
    eccentricity_sq = 6.6943799901413165e-3
    semiminor = semimajor * np.sqrt(1.0 - eccentricity_sq)
    horizontal = np.hypot(x, y)
    theta = np.arctan2(z * semimajor, horizontal * semiminor)
    latitude = np.arctan2(
        z + eccentricity_sq / (1.0 - eccentricity_sq) * semiminor * np.sin(theta)**3,
        horizontal - eccentricity_sq * semimajor * np.cos(theta)**3,
    )
    longitude = np.arctan2(y, x)
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    east = np.stack((-sin_lon, cos_lon, np.zeros_like(latitude)), axis=-1)
    north = np.stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), axis=-1)
    up = np.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), axis=-1)
    return np.stack((east, north, up), axis=-2)


@dataclass
class SharedEnuTrajectory:
    timestamps_ns: np.ndarray
    ecef_positions: np.ndarray
    enu_positions: np.ndarray
    max_ecef_reconstruction_diff_m: float


def world_to_shared_enu(world, alignment, reference_ecef):
    stamps, world_indices, gnss_indices = np.intersect1d(
        world.timestamps_ns, alignment.timestamps_ns, assume_unique=True, return_indices=True
    )
    if not len(stamps):
        raise TrajectoryError(f"{world.path}: no image timestamps match {alignment.path}")

    anchors = alignment.anchors_ecef[gnss_indices]
    yaw = alignment.yaw_rad[gnss_indices]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_enu_to_ecef = np.swapaxes(ecef_to_enu_rotation(anchors), -1, -2)
    ecef_to_shared = ecef_to_enu_rotation(reference_ecef)

    x, y, z = world.positions[world_indices].T
    local_enu = np.stack((cosine * x - sine * y, sine * x + cosine * y, z), axis=-1)
    ecef = anchors + np.einsum("nij,nj->ni", local_enu_to_ecef, local_enu)
    enu = (ecef - reference_ecef) @ ecef_to_shared.T
    reconstruction_diff = np.linalg.norm(ecef - alignment.logged_ecef[gnss_indices], axis=1)
    return SharedEnuTrajectory(stamps, ecef, enu, float(reconstruction_diff.max()))


def horizontal_difference(reference, other):
    stamps, i, j = np.intersect1d(reference.timestamps_ns, other.timestamps_ns,
                                  assume_unique=True, return_indices=True)
    differences = other.enu_positions[j, :2] - reference.enu_positions[i, :2]
    return stamps, np.linalg.norm(differences, axis=1)


def difference_stats(distances):
    return {
        "pairs": len(distances),
        "rms_m": float(np.sqrt(np.mean(distances**2))) if len(distances) else None,
        "max_m": float(np.max(distances)) if len(distances) else None,
    }


def write_shared_enu(path, trajectory):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp_ns", "ecef_x_m", "ecef_y_m", "ecef_z_m",
                         "east_m", "north_m", "up_m"))
        for stamp, ecef, enu in zip(trajectory.timestamps_ns, trajectory.ecef_positions,
                                    trajectory.enu_positions):
            writer.writerow((int(stamp), *ecef, *enu))


def run(args):
    worlds = {name: read_trajectory(getattr(args, name)) for name in ("gvins", "supergvins")}
    alignments = {}
    for name, world in worlds.items():
        gnss_path = getattr(args, f"{name}_gnss") or Path(world.path).with_name("gnss_result.csv")
        alignments[name] = read_gnss_alignment(gnss_path)

    reference = alignments["gvins"].anchors_ecef[0]
    trajectories = {}
    descriptions = {}
    for name, world in worlds.items():
        alignment = alignments[name]
        trajectory = world_to_shared_enu(world, alignment, reference)
        trajectories[name] = trajectory
        descriptions[name] = {
            **describe_trajectory(world),
            "gnss_input": str(alignment.path),
            "shared_enu_samples": len(trajectory.timestamps_ns),
            "shared_enu_path_length_m": path_length(trajectory.enu_positions),
            "world_samples_missing_exact_alignment": world.count - len(trajectory.timestamps_ns),
            "shared_enu_start_ns": int(trajectory.timestamps_ns[0]),
            "shared_enu_end_ns": int(trajectory.timestamps_ns[-1]),
            "max_ecef_reconstruction_diff_m": trajectory.max_ecef_reconstruction_diff_m,
        }

    stamps, distances = horizontal_difference(trajectories["gvins"], trajectories["supergvins"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "traj_compare_shared_enu.png"
    outputs = [plot_name, "summary.json"]
    make_shared_enu_plot(out_dir / plot_name, trajectories["gvins"], trajectories["supergvins"])
    for name, trajectory in trajectories.items():
        filename = f"{name}_shared_enu.csv"
        write_shared_enu(out_dir / filename, trajectory)
        outputs.append(filename)

    summary = {
        "comparison_type": "estimate_vs_estimate_shared_enu",
        "ground_truth_used": False,
        "reference_ecef_m": reference.tolist(),
        "reference_origin": "first GVINS GNSS anchor",
        "association": "exact image timestamp for each world pose and GNSS alignment",
        "transformations_applied": ["world_to_ecef_with_per_pose_anchor_and_yaw", "ecef_to_shared_enu"],
        "trajectories": descriptions,
        "horizontal_difference": {
            **difference_stats(distances),
            "start_ns": int(stamps[0]) if len(stamps) else None,
            "end_ns": int(stamps[-1]) if len(stamps) else None,
        },
        "outputs": outputs,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("supergvins", "gvins"):
        parser.add_argument(f"--{name}", required=True, help=f"{name} vins_result_no_loop.csv")
        parser.add_argument(f"--{name}-gnss", help="GNSS CSV (defaults to sibling gnss_result.csv)")
    parser.add_argument("--dataset", help="bag identifier retained for existing commands")
    parser.add_argument("--out-dir", required=True, help="output directory for ENU plot, CSV and statistics")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (TrajectoryError, OSError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
