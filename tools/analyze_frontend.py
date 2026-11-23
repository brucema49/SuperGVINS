#!/usr/bin/env python3
"""Audit frontend frame completeness, budgets, timing, and backend accounting.

This CLI reuses the coverage, timing, and backend event summaries from
``run_sports_field.py`` and adds frontend policy checks for one extractor call
per image and the configured 512/128 feature limits.

The dataset's ``camera_timestamps_csv`` field is optional. When present, the
index is checked against the run window; when omitted, only the run's expected
stamp file and observed frontend sequence can be checked independently.

Example:
    python3 src/SuperGVINS/tools/analyze_frontend.py --run-dir RUN_DIR \
        --dataset output/supergvins/sports_field/dataset.json --out SUMMARY.json
"""

from __future__ import print_function

import argparse
import collections
import csv
import json
import math
import os
import pathlib
import sys

import run_sports_field as runner


SAMPLE_LIMIT = 20
FRONTEND_REQUIRED_COLUMNS = (
    "frame_id",
    "stamp_ns",
    "superpoint_calls",
    "raw_keypoints",
    "kept_keypoints",
    "prev_track_count",
    "matcher_raw_pairs",
    "matches_before_ransac",
    "matches_after_ransac",
    "retained_tracks",
    "published_points",
    "extract_ms",
    "match_ms",
    "total_ms",
    "receive_steady_ns",
    "publish_steady_ns",
    "generation",
)
R200_FRONTEND_COLUMNS = (
    "prev_candidate_count",
    "ransac_inliers_before_cap",
)
LEGACY_FRONTEND_BUDGETS = {
    "kept_keypoints": 512,
    "matcher_raw_pairs": 512,
    "prev_track_count": 128,
    "matches_before_ransac": 128,
    "matches_after_ransac": 128,
    "retained_tracks": 128,
    "published_points": 128,
}
R200_FRONTEND_BUDGETS = {
    "kept_keypoints": 512,
    "prev_candidate_count": 512,
    "matcher_raw_pairs": 512,
    "matches_before_ransac": 512,
    "ransac_inliers_before_cap": 512,
    "prev_track_count": 200,
    "matches_after_ransac": 200,
    "retained_tracks": 200,
    "published_points": 200,
}
MATCH_OUTPUT_TARGET = 200


def detect_frontend_schema(rows):
    """Identify new R200 telemetry while retaining support for old R128 CSVs."""
    columns = set()
    for row in rows:
        columns.update(row.keys())
    new_columns = columns.intersection(R200_FRONTEND_COLUMNS)
    if len(new_columns) == len(R200_FRONTEND_COLUMNS):
        return "k512_r200", R200_FRONTEND_BUDGETS
    if not new_columns:
        return "legacy_k512_r128", LEGACY_FRONTEND_BUDGETS
    return "partial_k512_r200", R200_FRONTEND_BUDGETS


class AuditInputError(RuntimeError):
    pass


def read_integer_csv_column(path, column):
    path = pathlib.Path(path)
    if not path.is_file():
        return [], {"present": False, "path": str(path), "invalid_rows": 0, "invalid_sample": []}
    values = []
    invalid = []
    invalid_count = 0
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as source:
            reader = csv.DictReader(source)
            if column not in (reader.fieldnames or []):
                return [], {
                    "present": True,
                    "path": str(path),
                    "invalid_rows": 1,
                    "invalid_sample": [{"row": 1, "reason": "missing column {!r}".format(column)}],
                }
            for line_number, row in enumerate(reader, 2):
                try:
                    values.append(int(row.get(column, "")))
                except (TypeError, ValueError):
                    invalid_count += 1
                    if len(invalid) < SAMPLE_LIMIT:
                        invalid.append({"row": line_number, "value": row.get(column, "")})
    except (OSError, csv.Error) as error:
        return [], {
            "present": True,
            "path": str(path),
            "read_error": str(error),
            "invalid_rows": 1,
            "invalid_sample": [{"reason": str(error)}],
        }
    return values, {
        "present": True,
        "path": str(path),
        "row_count": len(values) + invalid_count,
        "valid_count": len(values),
        "invalid_rows": invalid_count,
        "invalid_sample": invalid,
    }


def load_json_object(path, description):
    path = pathlib.Path(path)
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditInputError("cannot read {} {}: {}".format(description, path, error))
    if not isinstance(value, dict):
        raise AuditInputError("{} must contain a JSON object: {}".format(description, path))
    return value


def load_run_metadata(run_dir):
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run_manifest.json"
    summary = load_json_object(summary_path, "runner summary") if summary_path.is_file() else {}
    manifest = load_json_object(manifest_path, "run manifest") if manifest_path.is_file() else {}
    if not summary and not manifest:
        raise AuditInputError("run directory must contain summary.json or run_manifest.json")
    # The final runner summary contains exit and artifact state; the manifest
    # carries the same window/config metadata before playback starts.
    metadata = dict(manifest)
    metadata.update(summary)
    return metadata, summary, manifest


def camera_index_audit(dataset_path, dataset, run_metadata, expected_stamps, failures):
    index_name = dataset.get("camera_timestamps_csv")
    if not index_name:
        return {
            "available": False,
            "optional": True,
            "reason": (
                "dataset omits camera_timestamps_csv; independent dataset-index verification is skipped, "
                "while run expected stamps are still checked against frontend output"
            ),
        }
    index_path = pathlib.Path(index_name)
    if not index_path.is_absolute():
        index_path = pathlib.Path(dataset_path).resolve().parent / index_path
    if not index_path.is_file():
        result = {"available": True, "path": str(index_path), "read_error": "camera timestamp index does not exist"}
        failures.append({"component": "camera_index", "reason": result["read_error"]})
        return result

    bag_window = run_metadata.get("bag_window") or {}
    try:
        window_start = float(bag_window["window_start_time"])
        window_end = float(bag_window["window_end_time"])
        if not (math.isfinite(window_start) and math.isfinite(window_end) and window_end >= window_start):
            raise ValueError("invalid run bag window")
    except (KeyError, TypeError, ValueError):
        reason = "run metadata has no valid bag_window.window_start_time/window_end_time"
        failures.append({"component": "camera_index", "reason": reason})
        return {"available": True, "path": str(index_path), "read_error": reason}

    selected_stamps = []
    invalid_rows = []
    invalid_row_count = 0
    try:
        with index_path.open("r", newline="", encoding="utf-8", errors="replace") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or [])
            if not {"bag_record_ns", "image_stamp_ns"}.issubset(fields):
                reason = "camera index requires bag_record_ns and image_stamp_ns columns"
                failures.append({"component": "camera_index", "reason": reason})
                return {"available": True, "path": str(index_path), "read_error": reason}
            for line_number, row in enumerate(reader, 2):
                try:
                    record_ns = int(row["bag_record_ns"])
                    image_stamp_ns = int(row["image_stamp_ns"])
                except (TypeError, ValueError):
                    invalid_row_count += 1
                    if len(invalid_rows) < SAMPLE_LIMIT:
                        invalid_rows.append({"row": line_number, "bag_record_ns": row.get("bag_record_ns"),
                                             "image_stamp_ns": row.get("image_stamp_ns")})
                    continue
                if window_start <= record_ns / 1e9 <= window_end:
                    selected_stamps.append(image_stamp_ns)
    except (OSError, csv.Error) as error:
        failures.append({"component": "camera_index", "reason": "cannot read camera index: {}".format(error)})
        return {"available": True, "path": str(index_path), "read_error": str(error)}

    coverage = runner.stamp_coverage(selected_stamps, expected_stamps)
    passed = coverage["complete"] and coverage["exact_sequence"] and not invalid_row_count
    if not passed:
        failures.append({
            "component": "camera_index",
            "reason": "per-run expected_image_stamps.csv does not exactly match the camera index window",
            "missing_count": coverage["missing_count"],
            "unexpected_count": coverage["unexpected_count"],
            "invalid_index_rows": invalid_row_count,
            "missing_stamp_ns_sample": coverage["missing_stamp_ns_sample"][:SAMPLE_LIMIT],
            "unexpected_stamp_ns_sample": coverage["unexpected_stamp_ns_sample"][:SAMPLE_LIMIT],
            "invalid_row_sample": invalid_rows[:SAMPLE_LIMIT],
        })
    return {
        "available": True,
        "path": str(index_path),
        "window_start_time": window_start,
        "window_end_time": window_end,
        "window_sample_count": len(selected_stamps),
        "invalid_rows": invalid_row_count,
        "invalid_row_sample": invalid_rows[:SAMPLE_LIMIT],
        "expected_image_stamps_coverage": coverage,
        "passed": passed,
    }


def frontend_policy_audit(rows, expected_stamps, playback_rate, failures):
    schema_profile, frontend_budgets = detect_frontend_schema(rows)
    required_columns = list(FRONTEND_REQUIRED_COLUMNS)
    if schema_profile != "legacy_k512_r128":
        required_columns.extend(R200_FRONTEND_COLUMNS)

    observed_stamps = []
    invalid_stamp_samples = []
    invalid_stamp_count = 0
    duplicate_stamps = collections.Counter()
    budget_stats = {
        name: {"limit": limit, "max_observed": None, "invalid_count": 0,
               "violation_count": 0, "violation_sample": []}
        for name, limit in frontend_budgets.items()
    }
    superpoint_violations = []
    superpoint_violation_count = 0
    raw_keypoint_max = None
    raw_keypoint_invalid_count = 0
    frame_ids = []
    malformed_required_rows = 0
    timing_invalid = collections.Counter()
    clock_invalid = []
    clock_invalid_count = 0
    generations = []
    exact_required_columns = []
    r200_frame_data = []

    for row_index, row in enumerate(rows, 1):
        for column in required_columns:
            if column not in row:
                if column not in exact_required_columns:
                    exact_required_columns.append(column)
        try:
            stamp = int(row.get("stamp_ns", ""))
            observed_stamps.append(stamp)
            duplicate_stamps[stamp] += 1
        except (TypeError, ValueError):
            invalid_stamp_count += 1
            if len(invalid_stamp_samples) < SAMPLE_LIMIT:
                invalid_stamp_samples.append({"row": row_index, "value": row.get("stamp_ns")})

        try:
            frame_id = int(row.get("frame_id", ""))
            frame_ids.append(frame_id)
        except (TypeError, ValueError):
            malformed_required_rows += 1

        try:
            calls = int(row.get("superpoint_calls", ""))
        except (TypeError, ValueError):
            calls = None
            malformed_required_rows += 1
        if calls != 1:
            superpoint_violation_count += 1
            if len(superpoint_violations) < SAMPLE_LIMIT:
                superpoint_violations.append({"row": row_index, "frame_id": row.get("frame_id"),
                                              "stamp_ns": row.get("stamp_ns"), "superpoint_calls": row.get("superpoint_calls")})

        try:
            raw_keypoints = int(row.get("raw_keypoints", ""))
            if raw_keypoints < 0:
                raise ValueError("negative raw keypoint count")
            raw_keypoint_max = raw_keypoints if raw_keypoint_max is None else max(raw_keypoint_max, raw_keypoints)
        except (TypeError, ValueError):
            raw_keypoint_invalid_count += 1

        for column, limit in frontend_budgets.items():
            stat = budget_stats[column]
            try:
                value = int(row.get(column, ""))
            except (TypeError, ValueError):
                stat["invalid_count"] += 1
                if len(stat["violation_sample"]) < SAMPLE_LIMIT:
                    stat["violation_sample"].append({"row": row_index, "frame_id": row.get("frame_id"),
                                                     "stamp_ns": row.get("stamp_ns"), "value": row.get(column)})
                continue
            stat["max_observed"] = value if stat["max_observed"] is None else max(stat["max_observed"], value)
            if value < 0 or value > limit:
                stat["violation_count"] += 1
                if len(stat["violation_sample"]) < SAMPLE_LIMIT:
                    stat["violation_sample"].append({"row": row_index, "frame_id": row.get("frame_id"),
                                                     "stamp_ns": row.get("stamp_ns"), "value": value})

        for column in ("extract_ms", "match_ms", "total_ms"):
            value = runner.finite_float(row.get(column))
            if value is None or value < 0:
                timing_invalid[column] += 1

        try:
            receive_ns = int(row.get("receive_steady_ns", ""))
            publish_ns = int(row.get("publish_steady_ns", ""))
            generation = int(row.get("generation", ""))
            generations.append(generation)
            if receive_ns <= 0 or publish_ns < receive_ns or generation < 0:
                raise ValueError("invalid steady-clock order")
        except (TypeError, ValueError):
            clock_invalid_count += 1
            if len(clock_invalid) < SAMPLE_LIMIT:
                clock_invalid.append({"row": row_index, "receive_steady_ns": row.get("receive_steady_ns"),
                                      "publish_steady_ns": row.get("publish_steady_ns"),
                                      "generation": row.get("generation")})

        if schema_profile != "legacy_k512_r128":
            frame_data = {"row": row_index}
            for column in (
                "generation", "kept_keypoints", "prev_candidate_count",
                "matches_before_ransac", "matches_after_ransac",
                "ransac_inliers_before_cap", "published_points",
            ):
                try:
                    frame_data[column] = int(row.get(column, ""))
                except (TypeError, ValueError):
                    frame_data[column] = None
            r200_frame_data.append(frame_data)

    coverage = runner.stamp_coverage(expected_stamps, observed_stamps)
    duplicate_values = [stamp for stamp, count in duplicate_stamps.items() if count > 1]
    duplicate_samples = duplicate_values[:SAMPLE_LIMIT]
    duplicate_count = sum(duplicate_stamps[stamp] - 1 for stamp in duplicate_values)
    duplicate_frame_ids = [frame_id for frame_id, count in collections.Counter(frame_ids).items() if count > 1]
    frame_id_monotone = all(right > left for left, right in zip(frame_ids, frame_ids[1:]))
    frame_id_sequence_valid = frame_ids == list(range(1, len(rows) + 1))

    match_target = None
    cross_field_errors = []
    if schema_profile != "legacy_k512_r128":
        no_previous_count = 0
        eligible_count = 0
        final_target_count = 0
        pre_cap_target_count = 0
        capped_count = 0
        below_target_count = 0
        thresholding_below_target_count = 0
        geometry_below_target_count = 0
        last_kept_by_generation = {}

        for frame_data in r200_frame_data:
            row_index = frame_data["row"]
            generation = frame_data["generation"]
            kept_keypoints = frame_data["kept_keypoints"]
            prev_candidates = frame_data["prev_candidate_count"]
            matches_before = frame_data["matches_before_ransac"]
            final_matches = frame_data["matches_after_ransac"]
            inliers_before_cap = frame_data["ransac_inliers_before_cap"]
            published_points = frame_data["published_points"]

            if (generation is not None and prev_candidates is not None and
                    generation in last_kept_by_generation and
                    prev_candidates > last_kept_by_generation[generation]):
                cross_field_errors.append({
                    "row": row_index,
                    "field": "prev_candidate_count",
                    "value": prev_candidates,
                    "previous_kept_keypoints": last_kept_by_generation[generation],
                    "generation": generation,
                })

            if prev_candidates == 0:
                no_previous_count += 1
            elif prev_candidates is not None and prev_candidates > 0:
                eligible_count += 1
                if final_matches == MATCH_OUTPUT_TARGET:
                    final_target_count += 1
                if inliers_before_cap == MATCH_OUTPUT_TARGET:
                    pre_cap_target_count += 1
                if inliers_before_cap is not None and inliers_before_cap > MATCH_OUTPUT_TARGET:
                    capped_count += 1
                if final_matches is not None and final_matches < MATCH_OUTPUT_TARGET:
                    below_target_count += 1
                if matches_before is not None and matches_before < MATCH_OUTPUT_TARGET:
                    thresholding_below_target_count += 1
                elif (matches_before is not None and matches_before >= MATCH_OUTPUT_TARGET and
                      inliers_before_cap is not None and inliers_before_cap < MATCH_OUTPUT_TARGET):
                    geometry_below_target_count += 1

            if (matches_before is not None and inliers_before_cap is not None and
                    inliers_before_cap > matches_before):
                cross_field_errors.append({
                    "row": row_index,
                    "field": "ransac_inliers_before_cap",
                    "value": inliers_before_cap,
                    "matches_before_ransac": matches_before,
                })
            if (matches_before is not None and kept_keypoints is not None and
                    prev_candidates is not None and
                    matches_before > min(kept_keypoints, prev_candidates)):
                cross_field_errors.append({
                    "row": row_index,
                    "field": "matches_before_ransac",
                    "value": matches_before,
                    "maximum_from_one_to_one_candidates": min(kept_keypoints, prev_candidates),
                })

            if inliers_before_cap is not None and final_matches is not None:
                expected_final = min(inliers_before_cap, MATCH_OUTPUT_TARGET)
                if final_matches != expected_final:
                    cross_field_errors.append({
                        "row": row_index,
                        "field": "matches_after_ransac",
                        "value": final_matches,
                        "expected_from_pre_cap_inliers": expected_final,
                    })
            if final_matches is not None and published_points is not None and published_points != final_matches:
                cross_field_errors.append({
                    "row": row_index,
                    "field": "published_points",
                    "value": published_points,
                    "expected_from_matches_after_ransac": final_matches,
                })

            if generation is not None and kept_keypoints is not None:
                last_kept_by_generation[generation] = kept_keypoints

        match_target = {
            "target": MATCH_OUTPUT_TARGET,
            "frame_count": len(rows),
            "eligible_frames_with_previous_candidates": eligible_count,
            "no_previous_or_bootstrapping_frames": no_previous_count,
            "final_count_exactly_at_target_frames": final_target_count,
            "pre_cap_count_exactly_at_target_frames": pre_cap_target_count,
            "capped_frames": capped_count,
            "below_target_frames": below_target_count,
            "thresholding_below_target_frames": thresholding_below_target_count,
            "geometry_below_target_frames": geometry_below_target_count,
            "shortage_measurement_available": eligible_count > 0,
            "below_target_fraction_of_eligible": (
                float(below_target_count) / eligible_count if eligible_count else None
            ),
            "shortage_observed": eligible_count > 0 and below_target_count > 0,
        }

    if exact_required_columns:
        failures.append({"component": "frontend_schema", "reason": "frontend_frames.csv is missing required columns",
                         "columns": exact_required_columns[:SAMPLE_LIMIT]})
    if cross_field_errors:
        failures.append({
            "component": "frontend_policy",
            "reason": "K512/R200 cross-field accounting does not match the frame telemetry",
            "violation_count": len(cross_field_errors),
            "sample": cross_field_errors[:SAMPLE_LIMIT],
        })
    if malformed_required_rows:
        failures.append({"component": "frontend_schema", "reason": "frontend rows contain invalid frame_id or superpoint_calls values",
                         "count": malformed_required_rows})
    if invalid_stamp_count:
        failures.append({"component": "frontend_coverage", "reason": "frontend rows contain invalid stamp_ns values",
                         "count": invalid_stamp_count, "sample": invalid_stamp_samples})
    if not coverage["complete"] or not coverage["exact_sequence"]:
        failures.append({"component": "frontend_coverage", "reason": "frontend stamp sequence does not exactly match expected image stamps",
                         "missing_count": coverage["missing_count"], "unexpected_count": coverage["unexpected_count"],
                         "missing_stamp_ns_sample": coverage["missing_stamp_ns_sample"][:SAMPLE_LIMIT],
                         "unexpected_stamp_ns_sample": coverage["unexpected_stamp_ns_sample"][:SAMPLE_LIMIT]})
    if duplicate_count:
        failures.append({"component": "frontend_coverage", "reason": "duplicate frontend image timestamps",
                         "duplicate_rows": duplicate_count, "stamp_ns_sample": duplicate_samples})
    if len(rows) != len(expected_stamps):
        failures.append({"component": "frontend_coverage", "reason": "frontend row count does not equal expected image count",
                         "observed_rows": len(rows), "expected_rows": len(expected_stamps)})
    if superpoint_violation_count:
        failures.append({"component": "frontend_policy", "reason": "SuperPoint call count must equal one for every frontend row",
                         "violation_count": superpoint_violation_count, "sample": superpoint_violations})
    if raw_keypoint_invalid_count:
        failures.append({"component": "frontend_telemetry", "reason": "raw_keypoints must be a nonnegative integer",
                         "invalid_count": raw_keypoint_invalid_count})
    for column, stat in budget_stats.items():
        if stat["invalid_count"] or stat["violation_count"]:
            failures.append({"component": "frontend_policy", "reason": "{} must be an integer in [0, {}]".format(column, stat["limit"]),
                             "invalid_count": stat["invalid_count"], "violation_count": stat["violation_count"],
                             "max_observed": stat["max_observed"], "sample": stat["violation_sample"][:SAMPLE_LIMIT]})
    if duplicate_frame_ids or not frame_id_monotone or not frame_id_sequence_valid:
        failures.append({"component": "frontend_policy", "reason": "frame_id values must be unique and strictly increasing",
                         "duplicate_frame_id_sample": duplicate_frame_ids[:SAMPLE_LIMIT],
                         "strictly_increasing": frame_id_monotone,
                         "one_based_contiguous": frame_id_sequence_valid})
    if timing_invalid:
        failures.append({"component": "frontend_timing", "reason": "timing fields must be finite and nonnegative",
                         "invalid_counts": dict(timing_invalid)})
    if clock_invalid_count:
        failures.append({"component": "frontend_timing", "reason": "steady-clock receive/publish values or generation are invalid",
                         "invalid_count": clock_invalid_count, "sample": clock_invalid})
    if generations and any(right < left for left, right in zip(generations, generations[1:])):
        failures.append({"component": "frontend_policy", "reason": "generation values must not decrease"})

    timing = runner.frontend_metrics(rows, playback_rate)
    return {
        "row_count": len(rows),
        "expected_image_count": len(expected_stamps),
        "exact_coverage": coverage,
        "invalid_stamp_row_count": invalid_stamp_count,
        "duplicate_stamp_row_count": duplicate_count,
        "duplicate_stamp_ns_sample": duplicate_samples,
        "once_per_frame_superpoint": {
            "passed": not superpoint_violation_count and not malformed_required_rows,
            "expected_calls": 1,
            "violation_count": superpoint_violation_count,
            "violation_sample": superpoint_violations,
        },
        "budget_checks": budget_stats,
        "budget_profile": schema_profile,
        "match_target_200": match_target,
        "raw_detector_matcher_telemetry": {
            "raw_keypoints_max_observed": raw_keypoint_max,
            "raw_keypoints_invalid_count": raw_keypoint_invalid_count,
            "raw_matcher_pairs_max_observed": budget_stats["matcher_raw_pairs"]["max_observed"],
            "raw_matcher_pairs_limit": budget_stats["matcher_raw_pairs"]["limit"],
        },
        "frame_ids": {
            "unique": not duplicate_frame_ids,
            "strictly_increasing": frame_id_monotone and frame_id_sequence_valid,
            "one_based_contiguous": frame_id_sequence_valid,
            "duplicate_sample": duplicate_frame_ids[:SAMPLE_LIMIT],
        },
        "timing_ms": timing,
        "invalid_timing_counts": dict(timing_invalid),
        "invalid_steady_clock_rows": clock_invalid_count,
        "invalid_steady_clock_sample": clock_invalid,
        "generation_count": len(set(generations)),
        "generation_values": sorted(set(generations))[:SAMPLE_LIMIT],
    }


def backend_audit(run_dir, mode, stack, expected_stamps, failures):
    if mode != "full" or stack != "supergvins":
        return {"available": False, "applicable": False,
                "reason": "backend event audit applies to full SuperGVINS runs"}
    result = runner.backend_event_summary(run_dir / "backend.csv", expected_stamps)
    error = runner.supergvins_backend_failure(result)
    if error:
        failures.append({"component": "backend", "reason": error,
                         "received_coverage": result.get("received_coverage"),
                         "unaccounted_received_timestamp_count": result.get("unaccounted_received_timestamp_count"),
                         "unaccounted_received_stamp_ns_sample": result.get("unaccounted_received_stamp_ns_sample", [])[:SAMPLE_LIMIT]})
    return {
        "available": result.get("available", False),
        "path": result.get("path"),
        "row_count": result.get("row_count"),
        "event_counts": result.get("event_counts", {}),
        "terminal_event_counts": result.get("terminal_event_counts", {}),
        "phase_counts": result.get("phase_counts", {}),
        "terminal_phase_counts": result.get("terminal_phase_counts", {}),
        "generation_counts": result.get("generation_counts", {}),
        "received_rows": result.get("received_rows"),
        "received_coverage": result.get("received_coverage"),
        "accounted_received_identity_count": result.get("accounted_received_identity_count"),
        "unaccounted_received_timestamp_count": result.get("unaccounted_received_timestamp_count"),
        "unaccounted_received_generation_stamp_sample": result.get("unaccounted_received_generation_stamp_sample", [])[:SAMPLE_LIMIT],
        "pending_shutdown_or_reset_count": result.get("pending_shutdown_or_reset_count"),
        "pending_shutdown_or_reset_sample": result.get("pending_shutdown_or_reset_rows", [])[:SAMPLE_LIMIT],
        "selected_enqueue_phase_counts_by_generation": result.get("selected_enqueue_phase_counts_by_generation", {}),
        "parity_summary_by_generation": result.get("parity_summary_by_generation", {}),
        "backend_queue_depth": result.get("backend_queue_depth", {}),
        "failure": error,
    }


def analyze(args):
    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    dataset_path = pathlib.Path(args.dataset).expanduser().resolve()
    out_path = pathlib.Path(args.out).expanduser().resolve()
    if not run_dir.is_dir():
        raise AuditInputError("run directory does not exist: {}".format(run_dir))
    run_metadata, run_summary, manifest = load_run_metadata(run_dir)
    dataset = load_json_object(dataset_path, "dataset metadata")

    failures = []
    if run_summary.get("success") is False:
        failures.append({"component": "runner", "reason": "runner summary marks this run unsuccessful",
                         "runner_failure": run_summary.get("failure")})

    expected_stamps, expected_file = read_integer_csv_column(
        run_dir / "expected_image_stamps.csv", "stamp_ns"
    )
    if not expected_file.get("present"):
        failures.append({"component": "expected_stamps", "reason": "expected_image_stamps.csv is missing"})
    if expected_file.get("read_error") or expected_file.get("invalid_rows"):
        failures.append({"component": "expected_stamps", "reason": "expected stamp CSV contains invalid data",
                         "invalid_rows": expected_file.get("invalid_rows"),
                         "sample": expected_file.get("invalid_sample", [])[:SAMPLE_LIMIT]})
    if not expected_stamps and not expected_file.get("read_error"):
        failures.append({"component": "expected_stamps", "reason": "expected stamp CSV contains no valid stamps"})

    camera = camera_index_audit(dataset_path, dataset, run_metadata, expected_stamps, failures)
    frontend_path = run_dir / "frontend_frames.csv"
    frontend_rows = runner.read_frontend_rows(frontend_path)
    if not frontend_path.is_file():
        failures.append({"component": "frontend", "reason": "frontend_frames.csv is missing"})
    if frontend_path.is_file() and not frontend_rows:
        failures.append({"component": "frontend", "reason": "frontend_frames.csv has no readable data rows"})
    try:
        playback_rate = float(run_metadata.get("playback_rate", 1.0))
        if not (math.isfinite(playback_rate) and playback_rate > 0):
            raise ValueError("playback_rate must be positive")
    except (TypeError, ValueError):
        playback_rate = 1.0
        failures.append({"component": "run_metadata", "reason": "invalid playback_rate; timing uses fallback 1.0"})

    frontend = frontend_policy_audit(frontend_rows, expected_stamps, playback_rate, failures)
    mode = run_metadata.get("mode", "full")
    stack = run_metadata.get("stack", "supergvins")
    backend = backend_audit(run_dir, mode, stack, expected_stamps, failures)

    report = {
        "schema_version": 1,
        "run_directory": str(run_dir),
        "run": {
            "stack": stack,
            "mode": mode,
            "device": run_metadata.get("device"),
            "gnss": run_metadata.get("gnss"),
            "playback_rate": playback_rate,
            "expected_image_count_in_metadata": run_metadata.get("expected_image_count"),
            "runner_success": run_summary.get("success"),
            "runner_failure": run_summary.get("failure"),
        },
        "dataset": {
            "path": str(dataset_path),
            "camera_topic": dataset.get("camera_topic"),
            "camera_index": camera,
        },
        "expected_image_stamps": expected_file,
        "frontend": frontend,
        "backend": backend,
        "failure_count": len(failures),
        "failure_causes": failures[:SAMPLE_LIMIT],
        "passed": not failures,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_json(out_path, report)
    return out_path, report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path, help="archived sports_field run directory")
    parser.add_argument("--dataset", required=True, type=pathlib.Path, help="sports_field dataset.json metadata")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="path for frontend_budget_summary.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        out_path, report = analyze(args)
    except AuditInputError as error:
        print("frontend audit input error: {}".format(error), file=sys.stderr)
        return 2
    except Exception as error:
        print("frontend audit failed: {}".format(error), file=sys.stderr)
        return 2
    print("summary={}".format(out_path))
    print("passed={}".format(report["passed"]))
    print("frontend_rows={}/{}".format(report["frontend"]["row_count"], report["frontend"]["expected_image_count"]))
    match_target = report["frontend"].get("match_target_200")
    if match_target is not None:
        print(
            "match_target_200 eligible={} no_previous={} exact={} capped={} below={} shortage_fraction={}".format(
                match_target["eligible_frames_with_previous_candidates"],
                match_target["no_previous_or_bootstrapping_frames"],
                match_target["final_count_exactly_at_target_frames"],
                match_target["capped_frames"],
                match_target["below_target_frames"],
                match_target["below_target_fraction_of_eligible"],
            )
        )
    print("failure_count={}".format(report["failure_count"]))
    for failure in report["failure_causes"]:
        print("failure[{}]={}".format(failure["component"], failure["reason"]), file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
