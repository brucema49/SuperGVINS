#!/usr/bin/env python3
"""Run an isolated, timestamp-accounted sports_field ROS bag replay.

Run from a sourced catkin workspace, for example:

    python3 src/SuperGVINS/tools/run_sports_field.py --mode full --device cpu \
        --gnss on --duration 2 --rate 0.1

Every invocation creates a new directory below output/supergvins/replay_runs.
The runner starts ROS processes directly so their exit codes remain observable.
"""

from __future__ import print_function

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import uuid


WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
SUPERGVINS_ROOT = WORKSPACE / "src" / "SuperGVINS"
GVINS_ROOT = WORKSPACE / "src" / "GVINS"
DEFAULT_BAG = WORKSPACE / "data" / "gvins" / "sports_field.bag"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "output" / "supergvins" / "replay_runs"
ROS_ENV_KEYS = ("ROS_MASTER_URI", "ROS_LOG_DIR", "ROS_HOME", "ROS_NAMESPACE")


class ReplayError(RuntimeError):
    pass


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def yaml_scalar(text, key):
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*:\s*([^#\r\n]*?)\s*(?:#.*)?$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def replace_yaml_scalar(text, key, value):
    """Replace one existing top-level OpenCV-YAML scalar without reformatting the file."""
    pattern = re.compile(
        r"^(\s*" + re.escape(key) + r"\s*:\s*)([^#\r\n]*?)(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ReplayError("expected exactly one configuration key {!r}; found {}".format(key, len(matches)))
    rendered = '"{}"'.format(str(value).replace('"', '\\"')) if isinstance(value, str) else str(value)
    match = matches[0]
    return text[: match.start()] + match.group(1) + rendered + match.group(3) + text[match.end() :]


def make_config_snapshot(template, run_dir, stack, device, gnss_enabled):
    template = pathlib.Path(template).resolve()
    if not template.is_file():
        raise ReplayError("configuration file does not exist: {}".format(template))
    text = template.read_text(encoding="utf-8")
    # Keep estimator and frontend artifacts at the run root. This is also the
    # backend contract location (`<run>/backend.csv`).
    text = replace_yaml_scalar(text, "output_dir", str(run_dir) + os.sep)

    if yaml_scalar(text, "gnss_enable") is not None:
        text = replace_yaml_scalar(text, "gnss_enable", 1 if gnss_enabled else 0)
    elif not gnss_enabled:
        raise ReplayError("--gnss off requested but config has no gnss_enable key")

    if stack == "supergvins":
        if yaml_scalar(text, "inference_device") is None:
            raise ReplayError("SuperGVINS config has no inference_device setting")
        text = replace_yaml_scalar(text, "inference_device", device)

    # GNSS trace destinations are always isolated, even if a future config enables
    # diagnostics that the current baseline leaves disabled.
    for key, filename in (
        ("gnss_spp_trace_file", "gnss_spp_trace.csv"),
        ("gnss_online_residual_trace_file", "gnss_online_residual_trace.csv"),
    ):
        if yaml_scalar(text, key) is not None:
            text = replace_yaml_scalar(text, key, str(run_dir / filename))

    snapshot = run_dir / "config.snapshot.yaml"
    snapshot.write_text(text, encoding="utf-8")
    return snapshot, template, text


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", choices=("supergvins", "gvins"), default="supergvins",
                        help="implementation to replay; gvins runs the original GVINS baseline")
    parser.add_argument("--mode", choices=("frontend", "full"), default="full",
                        help="frontend starts only the tracker; full starts tracker and estimator")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="SuperGVINS inference device (ignored for the original GVINS stack only when cpu)")
    parser.add_argument("--gnss", choices=("on", "off"), default="on", help="GNSS estimator input mode")
    parser.add_argument("--bag", type=pathlib.Path, default=DEFAULT_BAG, help="input ROS bag")
    parser.add_argument("--config", type=pathlib.Path, default=None, help="config template; defaults to the selected stack config")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT,
                        help="parent directory for unique run directories")
    parser.add_argument("--start-offset", type=float, default=0.0, help="seconds from bag start")
    parser.add_argument("--duration", type=float, default=2.0, help="bag window length in seconds (default: 2)")
    parser.add_argument("--full-bag", action="store_true",
                        help="replay from --start-offset to bag end; overrides --duration (default remains 2 s)")
    parser.add_argument("--rate", type=float, default=0.1, help="rosbag playback rate")
    parser.add_argument("--delay", type=float, default=2.0, help="rosbag --delay seconds")
    parser.add_argument("--ros-master-port", type=int, default=None, help="fixed local ROS master port; otherwise allocated automatically")
    parser.add_argument("--startup-timeout", type=float, default=45.0, help="seconds to await roscore and node subscriptions")
    parser.add_argument("--playback-timeout", type=float, default=None,
                        help="playback timeout; default is finite and derived from duration/rate")
    parser.add_argument("--drain-timeout", type=float, default=120.0,
                        help="maximum seconds to wait for expected frontend stamps after playback")
    parser.add_argument("--drain-seconds", type=float, default=3.0,
                        help="minimum post-playback processing time before tracker shutdown")
    parser.add_argument("--backend-drain-seconds", type=float, default=5.0,
                        help="additional processing time after tracker shutdown")
    parser.add_argument("--shutdown-timeout", type=float, default=15.0, help="seconds before killing a non-exiting process")
    parser.add_argument("--cuda-lib-path", action="append", default=[], metavar="DIR",
                        help="directory to prepend to LD_LIBRARY_PATH; may be repeated for CUDA runtime libraries")
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.start_offset < 0 or args.rate <= 0 or args.delay < 0:
        parser.error("duration/rate must be positive and start-offset/delay must be nonnegative")
    if min(args.startup_timeout, args.drain_timeout, args.shutdown_timeout) <= 0:
        parser.error("startup, drain and shutdown timeouts must be positive")
    if args.drain_seconds < 0 or args.backend_drain_seconds < 0:
        parser.error("drain durations must be nonnegative")
    if args.ros_master_port is not None and not (1024 <= args.ros_master_port <= 65535):
        parser.error("ros-master-port must be in 1024..65535")
    if args.stack == "gvins" and args.device != "cpu":
        parser.error("--device cuda applies only to the SuperGVINS learned frontend")
    return args


def make_run_dir(output_root, args):
    output_root = pathlib.Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    duration_label = "fullbag" if args.full_bag else _number_label(args.duration)
    label = "{}_{}_{}_gnss-{}_{}s".format(args.stack, args.mode, args.device, args.gnss, duration_label)
    for _ in range(10):
        candidate = output_root / "{}_{}_{}".format(stamp, label, uuid.uuid4().hex[:8])
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise ReplayError("could not allocate a unique run directory under {}".format(output_root))


def _number_label(value):
    return ("{:.3f}".format(value)).rstrip("0").rstrip(".").replace(".", "p")


def allocate_port(requested):
    if requested is not None:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def append_event(events, event_path, started_monotonic, name, **details):
    event = {
        "event": name,
        "utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
    }
    event.update(details)
    events.append(event)
    with pathlib.Path(event_path).open("a", encoding="utf-8") as target:
        target.write(json.dumps(event, sort_keys=True) + "\n")
        target.flush()


def import_ros_modules():
    try:
        import rosbag
        import rosgraph
        import rospy
    except Exception as error:
        raise ReplayError("ROS Python modules are unavailable; source the workspace setup.bash first: {}".format(error))
    return rosbag, rosgraph, rospy


def topic_value(config_text, key, default=None):
    value = yaml_scalar(config_text, key)
    return value if value is not None else default


def bag_window(bag_path, image_topic, offset, duration, full_bag=False):
    rosbag, _, rospy = import_ros_modules()
    image_stamps = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        bag_start = bag.get_start_time()
        bag_end = bag.get_end_time()
        window_start = bag_start + offset
        effective_duration = bag_end - window_start if full_bag else duration
        if effective_duration <= 0:
            raise ReplayError("start offset {:.3f}s leaves no bag data to replay".format(offset))
        window_end = window_start + effective_duration
        if window_end > bag_end + 1e-9:
            raise ReplayError(
                "requested window ends at {:.6f}, past bag end {:.6f}".format(window_end, bag_end)
            )
        for _, message, _ in bag.read_messages(
            topics=[image_topic],
            start_time=rospy.Time.from_sec(window_start),
            end_time=rospy.Time.from_sec(window_end),
        ):
            image_stamps.append(int(message.header.stamp.to_nsec()))
        topic_names = set(bag.get_type_and_topic_info()[1].keys())
    return {
        "bag_start_time": bag_start,
        "bag_end_time": bag_end,
        "window_start_time": window_start,
        "window_end_time": window_end,
        "window_duration_seconds": effective_duration,
        "image_stamps_ns": image_stamps,
        "bag_topics": topic_names,
    }


def write_expected_stamps(path, stamps):
    with pathlib.Path(path).open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(("ordinal", "stamp_ns"))
        writer.writerows((index, stamp) for index, stamp in enumerate(stamps, 1))


def build_child_environment(run_dir, port, args):
    env = os.environ.copy()
    for key in ROS_ENV_KEYS:
        env.pop(key, None)
    env.pop("ROS_HOSTNAME", None)
    env["ROS_MASTER_URI"] = "http://127.0.0.1:{}".format(port)
    env["ROS_IP"] = "127.0.0.1"
    env["ROS_LOG_DIR"] = str(run_dir / "ros_log")
    env["ROS_HOME"] = str(run_dir / "ros_home")
    (run_dir / "ros_log").mkdir(parents=True, exist_ok=True)
    (run_dir / "ros_home").mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        dirs = [str(pathlib.Path(item).expanduser().resolve()) for item in args.cuda_lib_path]
        dirs = [item for item in dirs if pathlib.Path(item).is_dir()]
        if dirs:
            inherited = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([inherited] if inherited else []))
    return env


def make_process_specs(stack, mode, config_path):
    if stack == "supergvins":
        root = SUPERGVINS_ROOT
        frontend = (
            "supergvins_feature_tracker",
            WORKSPACE / "devel" / "lib" / "supergvins_feature_tracker" / "supergvins_feature_tracker",
        )
        backend = ("supergvins", WORKSPACE / "devel" / "lib" / "supergvins" / "supergvins")
        feature_topic = "/supergvins_feature_tracker/feature"
        restart_topic = "/supergvins_feature_tracker/restart"
    else:
        root = GVINS_ROOT
        frontend = ("gvins_feature_tracker", WORKSPACE / "devel" / "lib" / "gvins_feature_tracker" / "gvins_feature_tracker")
        backend = ("gvins", WORKSPACE / "devel" / "lib" / "gvins" / "gvins")
        feature_topic = "/gvins_feature_tracker/feature"
        restart_topic = "/gvins_feature_tracker/restart"

    specs = []
    for name, executable in ((frontend[0], frontend[1]),):
        if not executable.is_file():
            raise ReplayError("built node executable is missing: {}".format(executable))
        specs.append(
            {
                "name": name,
                "kind": "frontend",
                "command": [
                    str(executable),
                    "__name:=" + name,
                    "_config_file:=" + str(config_path),
                    "_gvins_folder:=" + str(root) + os.sep,
                ],
                "node_name": "/" + name,
            }
        )
    if mode == "full":
        if not backend[1].is_file():
            raise ReplayError("built node executable is missing: {}".format(backend[1]))
        specs.append(
            {
                "name": backend[0],
                "kind": "backend",
                "command": [
                    str(backend[1]),
                    "__name:=" + backend[0],
                    "_config_file:=" + str(config_path),
                    "_gvins_folder:=" + str(root) + os.sep,
                ],
                "node_name": "/" + backend[0],
            }
        )
    return specs, feature_topic, restart_topic


def start_logged_process(name, command, env, run_dir, processes, events, event_path, run_started):
    log_path = run_dir / (name + ".log")
    log_file = log_path.open("w", buffering=1, encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    record = {
        "name": name,
        "command": command,
        "pid": process.pid,
        "log": str(log_path),
        "started_utc": utc_now(),
        "started_elapsed_seconds": round(time.monotonic() - run_started, 6),
        "exit_code": None,
        "stopped_utc": None,
        "stop_signal": None,
        "expected_exit": False,
    }
    processes[name] = {"process": process, "log_file": log_file, "record": record}
    append_event(events, event_path, run_started, "process_started", process=name, pid=process.pid, command=command)
    return process


def process_exit_error(processes, names=None):
    names = names if names is not None else list(processes.keys())
    for name in names:
        process = processes[name]["process"]
        code = process.poll()
        if code is not None and not processes[name]["record"].get("expected_exit", False):
            return "{} exited unexpectedly with status {}".format(name, code)
    return None


def wait_until_master(process, rosgraph, env, timeout):
    # ROS child processes receive a per-run ROS_MASTER_URI. The Python process
    # itself may have been started with a different URI, so pass it explicitly.
    master = rosgraph.Master("/sports_field_replay_runner", master_uri=env["ROS_MASTER_URI"])
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ReplayError("roscore exited with status {}".format(process.returncode))
        try:
            if master.getPid():
                return master
        except Exception as error:
            last_error = error
        time.sleep(0.1)
    raise ReplayError("roscore did not become ready within {:.1f}s: {}".format(timeout, last_error))


def subscriber_state(master):
    # rosgraph.Master.getSystemState unwraps the XML-RPC response and returns
    # [publishers, subscribers, services], raising on an RPC failure.
    state = master.getSystemState()
    subscribers = {}
    for topic, nodes in state[1]:
        subscribers[topic] = set(nodes)
    return subscribers


def expected_subscribers(stack, mode, gnss_enabled, config_text, image_topic, imu_topic, feature_topic, restart_topic):
    checks = {image_topic: ["/" + ("supergvins_feature_tracker" if stack == "supergvins" else "gvins_feature_tracker")]}
    if mode == "full":
        backend_node = "/supergvins" if stack == "supergvins" else "/gvins"
        checks[imu_topic] = [backend_node]
        checks[feature_topic] = [backend_node]
        checks[restart_topic] = [backend_node]
        if gnss_enabled:
            for key in ("gnss_meas_topic", "gnss_ephem_topic", "gnss_glo_ephem_topic", "gnss_iono_params_topic"):
                topic = topic_value(config_text, key)
                if topic:
                    checks[topic] = [backend_node]
            if topic_value(config_text, "gnss_local_online_sync", "0") not in ("0", "false", "False"):
                for key in ("gnss_tp_info_topic", "local_trigger_info_topic"):
                    topic = topic_value(config_text, key)
                    if topic:
                        checks[topic] = [backend_node]
    return checks


def wait_for_subscribers(master, checks, processes, timeout):
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        error = process_exit_error(processes)
        if error:
            raise ReplayError(error)
        active = subscriber_state(master)
        last = {
            topic: {
                "required_nodes": nodes,
                "registered_nodes": sorted(active.get(topic, set())),
                "ready": all(node in active.get(topic, set()) for node in nodes),
            }
            for topic, nodes in checks.items()
        }
        if all(item["ready"] for item in last.values()):
            return last
        time.sleep(0.1)
    return last


def read_frontend_rows(path):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error):
        return []


def stamp_coverage(expected, observed):
    expected_counts = collections.Counter(expected)
    observed_counts = collections.Counter(observed)
    missing_counts = expected_counts - observed_counts
    unexpected_counts = observed_counts - expected_counts
    missing = [stamp for stamp in expected for _ in range(missing_counts[stamp])]
    unexpected = [stamp for stamp in observed for _ in range(unexpected_counts[stamp])]
    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "complete": not missing_counts,
        "exact_sequence": list(expected) == list(observed),
        "missing_count": sum(missing_counts.values()),
        "unexpected_count": sum(unexpected_counts.values()),
        "missing_stamp_ns_sample": missing[:20],
        "unexpected_stamp_ns_sample": unexpected[:20],
        "first_expected_stamp_ns": expected[0] if expected else None,
        "last_expected_stamp_ns": expected[-1] if expected else None,
        "first_observed_stamp_ns": observed[0] if observed else None,
        "last_observed_stamp_ns": observed[-1] if observed else None,
    }


def wait_frontend_coverage(frontend_csv, expected, processes, timeout):
    deadline = time.monotonic() + timeout
    last_rows = []
    while True:
        error = process_exit_error(processes)
        if error:
            raise ReplayError(error)
        last_rows = read_frontend_rows(frontend_csv)
        stamps = []
        for row in last_rows:
            try:
                stamps.append(int(row["stamp_ns"]))
            except (KeyError, TypeError, ValueError):
                continue
        coverage = stamp_coverage(expected, stamps)
        if coverage["complete"] and coverage["exact_sequence"]:
            return last_rows, coverage, True
        if time.monotonic() >= deadline:
            return last_rows, coverage, False
        time.sleep(0.2)


def wait_duration(seconds, processes):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        error = process_exit_error(processes)
        if error:
            raise ReplayError(error)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def stop_process(name, item, timeout, events, event_path, run_started):
    process = item["process"]
    record = item["record"]
    if process.poll() is None:
        record["stop_signal"] = "SIGINT"
        append_event(events, event_path, run_started, "sigint_sent", process=name, pid=process.pid)
        try:
            # Each child runs in its own session. Signal its process group too so
            # helper children started by roscore cannot outlive this run.
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record["stop_signal"] = "SIGKILL_after_SIGINT_timeout"
            append_event(events, event_path, run_started, "sigkill_sent", process=name, pid=process.pid)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    record["exit_code"] = process.returncode
    record["stopped_utc"] = utc_now()
    record["stopped_elapsed_seconds"] = round(time.monotonic() - run_started, 6)
    record["expected_exit"] = True
    append_event(events, event_path, run_started, "process_stopped", process=name,
                 exit_code=process.returncode, stop_signal=record["stop_signal"])
    item["log_file"].close()


def stop_all(processes, order, timeout, events, event_path, run_started):
    for name in order:
        item = processes.get(name)
        if not item or item["record"].get("stopped_utc"):
            continue
        try:
            stop_process(name, item, timeout, events, event_path, run_started)
        except Exception as error:
            item["record"]["cleanup_error"] = str(error)
            try:
                if item["process"].poll() is None:
                    item["process"].kill()
                    item["process"].wait()
            except Exception:
                pass
            try:
                item["log_file"].close()
            except Exception:
                pass


def log_evidence(processes):
    patterns = re.compile(r"gnss enabled|gnss|time.?pulse|external.?trigger|trigger|time diff|synchron", re.IGNORECASE)
    evidence = {}
    for name, item in processes.items():
        path = pathlib.Path(item["record"]["log"])
        hits = []
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    if patterns.search(line):
                        hits.append(line.rstrip()[:500])
                        if len(hits) >= 40:
                            break
        evidence[name] = hits
    return evidence


def csv_file_summary(path, timestamp_column=None, event_column=None):
    path = pathlib.Path(path)
    if not path.is_file():
        return {"present": False, "path": str(path)}
    result = {"present": True, "path": str(path), "size_bytes": path.stat().st_size}
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        result["row_count"] = len(rows)
        if event_column:
            result["event_counts"] = dict(collections.Counter(row.get(event_column, "") for row in rows))
        if timestamp_column:
            result["timestamps_ns"] = [row.get(timestamp_column, "") for row in rows if row.get(timestamp_column, "")]
    except (OSError, csv.Error) as error:
        result["read_error"] = str(error)
    return result


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values, percentage):
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def series_summary(values):
    values = [value for value in values if math.isfinite(value)]
    return {
        "sample_count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def linear_trend(values):
    values = [value for value in values if math.isfinite(value)]
    if len(values) < 2:
        return None
    x_values = list(range(len(values)))
    x_mean = sum(x_values) / float(len(x_values))
    y_mean = sum(values) / float(len(values))
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values)) / denominator if denominator else 0.0
    return {
        "first_ms": values[0],
        "last_ms": values[-1],
        "endpoint_change_ms": values[-1] - values[0],
        "linear_slope_ms_per_frame": slope,
        "trend": "growing" if slope > 0.05 else "shrinking" if slope < -0.05 else "flat",
        "sample_count": len(values),
    }


def frontend_metrics(rows, playback_rate=1.0):
    timing_values = {name: [] for name in ("extract_ms", "match_ms", "total_ms")}
    receive_publish_latency_ms = []
    receive_timeline = []
    for row in rows:
        for name in timing_values:
            value = finite_float(row.get(name))
            if value is not None:
                timing_values[name].append(value)
        if finite_float(row.get("total_ms")) is None:
            legacy_processing = finite_float(row.get("processing_ms"))
            if legacy_processing is not None:
                timing_values["total_ms"].append(legacy_processing)
        try:
            receive_ns = int(row["receive_steady_ns"])
            publish_ns = int(row["publish_steady_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        receive_publish_latency_ms.append((publish_ns - receive_ns) / 1e6)
        try:
            stamp_ns = int(row["stamp_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        receive_timeline.append((stamp_ns, receive_ns))

    relative_playback_lag_ms = []
    if playback_rate > 0 and len(receive_timeline) >= 2:
        first_stamp, first_receive = receive_timeline[0]
        for stamp_ns, receive_ns in receive_timeline:
            expected_elapsed_ms = (stamp_ns - first_stamp) / 1e6 / playback_rate
            actual_elapsed_ms = (receive_ns - first_receive) / 1e6
            relative_playback_lag_ms.append(actual_elapsed_ms - expected_elapsed_ms)

    timing_summary = {name: series_summary(values) for name, values in timing_values.items()}
    # Keep the original processing_ms field for existing summary readers.
    processing_summary = dict(timing_summary["total_ms"])
    processing_summary["source_column"] = (
        "total_ms" if any(finite_float(row.get("total_ms")) is not None for row in rows)
        else "processing_ms" if timing_values["total_ms"] else None
    )
    callback_latency_summary = series_summary(receive_publish_latency_ms)
    callback_latency_summary["available"] = bool(receive_publish_latency_ms)
    return {
        "frame_count": len(rows),
        "playback_rate": playback_rate,
        "timings_ms": timing_summary,
        "processing_ms": processing_summary,
        "receive_to_publish_wall_lag_ms": callback_latency_summary,
        "callback_processing_latency_ms": dict(callback_latency_summary),
        "relative_playback_lag_ms": series_summary(relative_playback_lag_ms),
        "relative_playback_lag_trend": linear_trend(relative_playback_lag_ms),
        "relative_playback_lag_definition": (
            "callback receive wall elapsed minus image header-stamp elapsed divided by playback rate; "
            "this tracks arrival drift relative to replay time, not subscription queue depth"
        ),
        "queue_growth_trend": {
            "available": False,
            "reason": "callback receive/publish clocks do not directly measure subscriber queue depth",
        },
        "steady_clock_columns_available": bool(receive_publish_latency_ms),
    }


def backend_row_is_terminal(row):
    event = row.get("event", "")
    if event == "wait_sync":
        return row.get("detail", "").strip() in ("clock_uncalibrated", "phase_uncalibrated")
    return event in {
        "skip_parity", "empty", "rejected", "drop_sync_stale", "processed",
        "reset_pending", "reset_drop", "shutdown_pending",
    }


def backend_event_summary(path, expected_stamps):
    """Summarize the new optional backend event contract when that file exists."""
    path = pathlib.Path(path)
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "backend.csv was not produced by this run"}
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as error:
        return {"available": True, "path": str(path), "read_error": str(error)}

    events = collections.Counter(row.get("event", "") for row in rows)
    phase_counts = collections.Counter(row.get("phase", "") for row in rows if row.get("phase", ""))
    terminal_event_counts = collections.Counter(row.get("event", "") for row in rows if backend_row_is_terminal(row))
    terminal_phase_counts = collections.Counter(
        row.get("phase", "") for row in rows
        if row.get("phase", "") and backend_row_is_terminal(row)
    )
    generation_counts = collections.Counter(row.get("generation", "") for row in rows if row.get("generation", ""))
    # `received` is an arrival. Only wait_sync records for the two sync
    # calibration states resolve a callback; IMU/GNSS waits remain outstanding.
    decision_events = {"skip_parity", "empty", "rejected", "drop_sync_stale"}
    received_sequence = []
    received_identities = set()
    accounted_identities = set()
    decision_counts = collections.Counter()
    wait_sync_detail_counts = collections.Counter()
    queued_wait_sync_counts = collections.Counter()
    received_by_generation_phase = collections.Counter()
    enqueue_by_generation_phase = collections.Counter()
    processed_by_generation_phase = collections.Counter()
    skipped_by_generation_phase = collections.Counter()
    queue_depth_samples = []
    received_rows = 0
    pending_rows = []
    for row in rows:
        event = row.get("event", "")
        raw_stamp = row.get("feature_stamp_ns", "")
        generation = row.get("generation", "")
        phase = row.get("phase", "")
        try:
            stamp = int(raw_stamp)
        except (TypeError, ValueError):
            stamp = None
        if event == "received":
            received_rows += 1
            if stamp is not None:
                received_sequence.append(stamp)
                received_identities.add((generation, stamp))
                received_by_generation_phase[(generation, phase)] += 1
        if backend_row_is_terminal(row) and stamp is not None:
            accounted_identities.add((generation, stamp))
        if event in decision_events:
            decision_counts[event] += 1
        if event == "wait_sync":
            detail = row.get("detail", "").strip() or "unspecified"
            wait_sync_detail_counts[detail] += 1
            if backend_row_is_terminal(row):
                decision_counts["wait_sync_" + detail] += 1
            else:
                queued_wait_sync_counts[detail] += 1
        if event == "enqueue":
            enqueue_by_generation_phase[(generation, phase)] += 1
        elif event == "processed":
            processed_by_generation_phase[(generation, phase)] += 1
        elif event == "skip_parity":
            skipped_by_generation_phase[(generation, phase)] += 1
        try:
            queue_depth = int(row.get("queue_depth", ""))
        except (TypeError, ValueError):
            queue_depth = -1
        if queue_depth >= 0:
            queue_depth_samples.append({
                "generation": generation,
                "sequence": row.get("sequence", ""),
                "event": event,
                "queue_depth": queue_depth,
            })
        if event in ("shutdown_pending", "reset_pending", "reset_drop"):
            pending_rows.append({
                "event": event,
                "feature_stamp_ns": stamp,
                "detail": row.get("detail", ""),
                "phase": row.get("phase", ""),
                "queue_depth": row.get("queue_depth", ""),
                "terminal_for_received": backend_row_is_terminal(row),
                "classification": (
                    "shutdown_pending_terminal; may be a bag-tail sensor wait and is not automatic loss"
                    if event == "shutdown_pending" else
                    "reset_pending_terminal; queued pre-barrier observation cancelled at ordered reset barrier"
                    if event == "reset_pending" else
                    "reset_drop_terminal" if event == "reset_drop" else
                    "reset_drop_terminal"
                ),
            })

    unaccounted_identities = sorted(received_identities - accounted_identities)
    accounted_received = received_identities.intersection(accounted_identities)
    selected_by_generation = collections.defaultdict(list)
    for (generation, phase), count in sorted(enqueue_by_generation_phase.items()):
        selected_by_generation[generation].append({"phase": phase, "enqueue_count": count})
    parity_by_generation = collections.defaultdict(dict)
    for (generation, phase), count in sorted(received_by_generation_phase.items()):
        parity_by_generation[generation].setdefault("received_by_phase", {})[phase] = count
    for (generation, phase), count in sorted(enqueue_by_generation_phase.items()):
        parity_by_generation[generation].setdefault("enqueued_by_phase", {})[phase] = count
    for (generation, phase), count in sorted(processed_by_generation_phase.items()):
        parity_by_generation[generation].setdefault("processed_by_phase", {})[phase] = count
    for (generation, phase), count in sorted(skipped_by_generation_phase.items()):
        parity_by_generation[generation].setdefault("skip_parity_by_phase", {})[phase] = count
    for generation, entries in selected_by_generation.items():
        selected_count = max(item["enqueue_count"] for item in entries)
        parity_by_generation[generation]["selected_enqueue_phases"] = [
            item["phase"] for item in entries if item["enqueue_count"] == selected_count
        ]
        parity_by_generation[generation]["selected_enqueue_count"] = sum(
            item["enqueue_count"] for item in entries
        )
    return {
        "available": True,
        "path": str(path),
        "schema_columns": fieldnames,
        "row_count": len(rows),
        "schema_versions": sorted(set(row.get("schema_version", "") for row in rows if row.get("schema_version", ""))),
        "event_counts": dict(events),
        "terminal_event_counts": dict(terminal_event_counts),
        "phase_counts": dict(phase_counts),
        "terminal_phase_counts": dict(terminal_phase_counts),
        "generation_counts": dict(generation_counts),
        "callback_decision_event_counts": dict(decision_counts),
        "wait_sync_detail_counts": dict(wait_sync_detail_counts),
        "queued_wait_sync_observation_counts": dict(queued_wait_sync_counts),
        "parity_summary_by_generation": dict(parity_by_generation),
        "selected_enqueue_phase_counts_by_generation": {
            generation: entries for generation, entries in selected_by_generation.items()
        },
        "received_rows": received_rows,
        "received_unique_stamp_count": len(set(received_sequence)),
        "received_coverage": stamp_coverage(expected_stamps, received_sequence),
        "accounted_received_identity_count": len(accounted_received),
        "unaccounted_received_timestamp_count": len(unaccounted_identities),
        "unaccounted_received_stamp_ns_sample": [stamp for _, stamp in unaccounted_identities[:20]],
        "unaccounted_received_generation_stamp_sample": [
            {"generation": generation, "stamp_ns": stamp} for generation, stamp in unaccounted_identities[:20]
        ],
        "pending_shutdown_or_reset_rows": pending_rows,
        "pending_shutdown_or_reset_count": len(pending_rows),
        "shutdown_pending_count": events.get("shutdown_pending", 0),
        "reset_pending_count": events.get("reset_pending", 0),
        "reset_drop_count": events.get("reset_drop", 0),
        "backend_queue_depth": {
            "available": bool(queue_depth_samples),
            "sample_count": len(queue_depth_samples),
            "max": max((item["queue_depth"] for item in queue_depth_samples), default=None),
            "last": queue_depth_samples[-1]["queue_depth"] if queue_depth_samples else None,
            "sample": queue_depth_samples[:20],
        },
    }


def artifact_summary(run_dir, backend_enabled, expected_stamps, playback_rate=1.0):
    frontend_rows = read_frontend_rows(run_dir / "frontend_frames.csv")
    result = {
        "frontend_frames_csv": csv_file_summary(run_dir / "frontend_frames.csv"),
        "frontend_metrics": frontend_metrics(frontend_rows, playback_rate),
        "vins_result_no_loop_csv": csv_file_summary(run_dir / "vins_result_no_loop.csv"),
        "gnss_result_csv": csv_file_summary(run_dir / "gnss_result.csv"),
        "factor_graph_result": {
            "present": (run_dir / "factor_graph_result.txt").is_file(),
            "path": str(run_dir / "factor_graph_result.txt"),
            "size_bytes": (run_dir / "factor_graph_result.txt").stat().st_size
            if (run_dir / "factor_graph_result.txt").is_file() else 0,
        },
        "gnss_spp_trace_csv": csv_file_summary(run_dir / "gnss_spp_trace.csv"),
        "gnss_online_residual_trace_csv": csv_file_summary(run_dir / "gnss_online_residual_trace.csv"),
    }
    if backend_enabled:
        result["backend_csv"] = backend_event_summary(run_dir / "backend.csv", expected_stamps)
    return result


def supergvins_backend_failure(backend_summary):
    if not backend_summary.get("available"):
        return "SuperGVINS full run did not produce backend.csv"
    coverage = backend_summary.get("received_coverage", {})
    if not coverage.get("complete") or not coverage.get("exact_sequence"):
        return "backend receive sequence did not exactly match expected image stamps"
    if backend_summary.get("unaccounted_received_timestamp_count") != 0:
        return "backend has received timestamps without a terminal accounting event"
    return None


def execute(args):
    run_dir = make_run_dir(args.output_root, args)
    run_started = time.monotonic()
    events = []
    event_path = run_dir / "lifecycle.jsonl"
    processes = {}
    process_specs = []
    failure = None
    ros_env = None
    port = None
    master = None
    metadata = {
        "runner": str(pathlib.Path(__file__).resolve()),
        "workspace": str(WORKSPACE),
        "run_directory": str(run_dir),
        "started_utc": utc_now(),
        "stack": args.stack,
        "mode": args.mode,
        "device": args.device,
        "gnss": args.gnss,
        "bag": str(pathlib.Path(args.bag).expanduser().resolve()),
        "start_offset_seconds": args.start_offset,
        "duration_seconds": args.duration,
        "full_bag": args.full_bag,
        "playback_rate": args.rate,
        "playback_delay_seconds": args.delay,
        "playback_topics_filter": None,
        "processes": {},
    }
    try:
        bag_path = pathlib.Path(args.bag).expanduser().resolve()
        if not bag_path.is_file():
            raise ReplayError("bag does not exist: {}".format(bag_path))
        config_default = (
            SUPERGVINS_ROOT / "config" / "visensor_f9p" / "visensor_left_f9p_config.yaml"
            if args.stack == "supergvins"
            else GVINS_ROOT / "config" / "visensor_f9p" / "visensor_left_f9p_config.yaml"
        )
        config_path, template_path, config_text = make_config_snapshot(
            args.config or config_default,
            run_dir,
            args.stack,
            args.device,
            args.gnss == "on",
        )
        image_topic = topic_value(config_text, "image_topic", "/cam1/image_raw")
        imu_topic = topic_value(config_text, "imu_topic", "/imu0")
        window = bag_window(bag_path, image_topic, args.start_offset, args.duration, args.full_bag)
        effective_duration = window["window_duration_seconds"]
        metadata["duration_seconds"] = effective_duration
        expected_stamps = window["image_stamps_ns"]
        if not expected_stamps:
            raise ReplayError("no {} messages exist in the selected bag window".format(image_topic))
        write_expected_stamps(run_dir / "expected_image_stamps.csv", expected_stamps)

        port = allocate_port(args.ros_master_port)
        ros_env = build_child_environment(run_dir, port, args)
        process_specs, feature_topic, restart_topic = make_process_specs(args.stack, args.mode, config_path)
        metadata.update(
            {
                "config_template": str(template_path),
                "config_template_sha256": sha256_file(template_path),
                "config_snapshot": str(config_path),
                "config_snapshot_sha256": sha256_file(config_path),
                "ros_master_uri": ros_env["ROS_MASTER_URI"],
                "ros_master_port": port,
                "image_topic": image_topic,
                "imu_topic": imu_topic,
                "feature_topic": feature_topic,
                "restart_topic": restart_topic,
                "expected_image_count": len(expected_stamps),
                "bag_window": {key: value for key, value in window.items() if key != "bag_topics" and key != "image_stamps_ns"},
                "gnss_topics_present_in_bag": {},
                "cuda_library_path": ros_env.get("LD_LIBRARY_PATH", "") if args.device == "cuda" else None,
            }
        )
        gnss_topics = [
            topic_value(config_text, key)
            for key in (
                "gnss_meas_topic", "gnss_ephem_topic", "gnss_glo_ephem_topic",
                "gnss_iono_params_topic", "gnss_tp_info_topic", "local_trigger_info_topic",
            )
        ]
        metadata["gnss_topics_present_in_bag"] = {
            topic: topic in window["bag_topics"] for topic in gnss_topics if topic
        }
        write_json(run_dir / "run_manifest.json", metadata)
        append_event(events, event_path, run_started, "run_prepared", run_directory=str(run_dir))

        roscore = start_logged_process(
            "roscore",
            ["roscore", "-p", str(port)],
            ros_env,
            run_dir,
            processes,
            events,
            event_path,
            run_started,
        )
        _, rosgraph, _ = import_ros_modules()
        master = wait_until_master(roscore, rosgraph, ros_env, args.startup_timeout)
        append_event(events, event_path, run_started, "roscore_ready", ros_master_uri=ros_env["ROS_MASTER_URI"])

        subprocess.run(
            ["rosparam", "set", "/use_sim_time", "true"],
            env=ros_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=min(10.0, args.startup_timeout),
            text=True,
        )
        append_event(events, event_path, run_started, "use_sim_time_set", value=True)

        for spec in process_specs:
            start_logged_process(
                spec["name"], spec["command"], ros_env, run_dir, processes, events, event_path, run_started
            )
        checks = expected_subscribers(
            args.stack, args.mode, args.gnss == "on", config_text, image_topic, imu_topic, feature_topic, restart_topic
        )
        readiness = wait_for_subscribers(master, checks, processes, args.startup_timeout)
        metadata["subscriber_readiness"] = readiness
        if not readiness or not all(value.get("ready") for value in readiness.values()):
            raise ReplayError("required subscribers were not ready before timeout")
        append_event(events, event_path, run_started, "subscribers_ready", subscriptions=readiness)
        metadata["node_startup_log_evidence"] = log_evidence(processes)
        write_json(run_dir / "run_manifest.json", metadata)

        playback_command = [
            "rosbag", "play", "--clock", "--delay", str(args.delay),
            "--start", str(args.start_offset), "--duration", str(effective_duration),
            "--rate", str(args.rate), str(bag_path),
        ]
        playback_timeout = args.playback_timeout
        if playback_timeout is None:
            playback_timeout = args.delay + effective_duration / args.rate + args.startup_timeout + 60.0
        playback = start_logged_process(
            "rosbag_play",
            playback_command,
            ros_env,
            run_dir,
            processes,
            events,
            event_path,
            run_started,
        )
        metadata["playback_command"] = playback_command
        metadata["playback_timeout_seconds"] = playback_timeout
        append_event(events, event_path, run_started, "playback_started", command=playback_command,
                     delay_seconds=args.delay, start_offset_seconds=args.start_offset,
                     duration_seconds=effective_duration, rate=args.rate)
        deadline = time.monotonic() + playback_timeout
        while playback.poll() is None:
            error = process_exit_error(processes, [name for name in processes if name != "rosbag_play"])
            if error:
                raise ReplayError(error)
            if time.monotonic() >= deadline:
                raise ReplayError("rosbag playback exceeded {:.1f}s timeout".format(playback_timeout))
            time.sleep(0.1)
        if playback.returncode != 0:
            raise ReplayError("rosbag play exited with status {}".format(playback.returncode))
        processes["rosbag_play"]["record"]["exit_code"] = playback.returncode
        processes["rosbag_play"]["record"]["stopped_utc"] = utc_now()
        processes["rosbag_play"]["record"]["stopped_elapsed_seconds"] = round(time.monotonic() - run_started, 6)
        processes["rosbag_play"]["record"]["expected_exit"] = True
        processes["rosbag_play"]["log_file"].close()
        append_event(events, event_path, run_started, "playback_finished", exit_code=playback.returncode)

        append_event(events, event_path, run_started, "drain_started", minimum_seconds=args.drain_seconds,
                     coverage_timeout_seconds=args.drain_timeout)
        frontend_csv = run_dir / "frontend_frames.csv"
        if args.stack == "supergvins":
            rows, frontend_coverage, coverage_wait_satisfied = wait_frontend_coverage(
                frontend_csv, expected_stamps, processes, args.drain_timeout
            )
        else:
            rows = read_frontend_rows(frontend_csv)
            frontend_coverage = {
                "available": False,
                "reason": "original GVINS frontend does not emit frontend_frames.csv",
                "expected_count": len(expected_stamps),
            }
            coverage_wait_satisfied = True
        if coverage_wait_satisfied:
            append_event(events, event_path, run_started, "frontend_coverage_reached",
                         observed_count=frontend_coverage.get("observed_count"),
                         expected_count=frontend_coverage.get("expected_count"))
        wait_duration(args.drain_seconds, processes)

        # Stop the tracker first so the estimator can finish work queued from its
        # final feature messages. Its eventual backend.csv may include shutdown
        # accounting records, so inspect artifacts only after graceful shutdown.
        tracker_names = [spec["name"] for spec in process_specs if spec["kind"] == "frontend"]
        backend_names = [spec["name"] for spec in process_specs if spec["kind"] == "backend"]
        stop_all(processes, tracker_names, args.shutdown_timeout, events, event_path, run_started)
        if backend_names:
            wait_duration(args.backend_drain_seconds, processes)
            append_event(events, event_path, run_started, "backend_drain_finished",
                         seconds=args.backend_drain_seconds)
            stop_all(processes, backend_names, args.shutdown_timeout, events, event_path, run_started)
        stop_all(processes, ["rosbag_play"], args.shutdown_timeout, events, event_path, run_started)
        metadata["frontend_coverage"] = frontend_coverage
        metadata["frontend_coverage_wait_satisfied"] = coverage_wait_satisfied
        metadata["frontend_log_row_count"] = len(rows)
        metadata["node_shutdown_log_evidence"] = log_evidence(processes)
    except KeyboardInterrupt:
        failure = "interrupted by user"
        append_event(events, event_path, run_started, "interrupted")
    except Exception as error:
        failure = str(error)
        append_event(events, event_path, run_started, "run_error", error=failure)
    finally:
        # SIGINT all remaining children individually, preserving their true exit
        # status and making a failed startup/replay leave no ROS processes behind.
        cleanup_order = ["rosbag_play"] + [
            spec["name"] for spec in reversed(process_specs)
        ] + ["roscore"]
        stop_all(processes, cleanup_order, args.shutdown_timeout, events, event_path, run_started)

    for name, item in processes.items():
        metadata["processes"][name] = item["record"]
    metadata["node_startup_log_evidence"] = metadata.get("node_startup_log_evidence", log_evidence(processes))
    metadata["node_shutdown_log_evidence"] = metadata.get("node_shutdown_log_evidence", log_evidence(processes))
    metadata["finished_utc"] = utc_now()
    metadata["elapsed_seconds"] = round(time.monotonic() - run_started, 6)
    metadata["failure"] = failure
    metadata["success"] = failure is None
    expected_stamps = window.get("image_stamps_ns", []) if "window" in locals() else []
    metadata["artifacts"] = artifact_summary(
        run_dir, args.mode == "full", expected_stamps, playback_rate=args.rate
    )
    if "frontend_coverage" not in metadata:
        if args.stack == "supergvins":
            rows = read_frontend_rows(run_dir / "frontend_frames.csv")
            observed = []
            for row in rows:
                try:
                    observed.append(int(row["stamp_ns"]))
                except (KeyError, TypeError, ValueError):
                    continue
            metadata["frontend_coverage"] = stamp_coverage(window["image_stamps_ns"], observed) if "window" in locals() else {"available": False}
        else:
            metadata["frontend_coverage"] = {"available": False, "reason": "original GVINS frontend does not emit frontend_frames.csv"}
    if metadata.get("frontend_coverage_wait_satisfied") is False:
        metadata["success"] = False
        if not metadata["failure"]:
            metadata["failure"] = "frontend did not cover the exact expected image stamp sequence before drain timeout"
    if args.stack == "supergvins" and args.mode == "full":
        backend = metadata["artifacts"].get("backend_csv", {})
        backend_failure = supergvins_backend_failure(backend)
        if backend_failure:
            metadata["success"] = False
            if not metadata["failure"]:
                metadata["failure"] = backend_failure
    nonzero_nodes = {
        name: item["record"].get("exit_code")
        for name, item in processes.items()
        if name != "rosbag_play" and item["record"].get("exit_code") not in (None, 0)
    }
    metadata["nonzero_node_exit_codes"] = nonzero_nodes
    if nonzero_nodes:
        metadata["success"] = False
        if not metadata["failure"]:
            metadata["failure"] = "one or more ROS nodes exited nonzero"
    metadata["processes"] = {name: item["record"] for name, item in processes.items()}
    append_event(events, event_path, run_started, "run_finished", success=metadata["success"], failure=metadata["failure"])
    write_json(run_dir / "summary.json", metadata)
    return run_dir, metadata


def main(argv=None):
    args = parse_args(argv)
    try:
        run_dir, summary = execute(args)
    except Exception as error:
        print("sports_field replay runner failed before a run could be completed: {}".format(error), file=sys.stderr)
        return 2
    print("run_directory={}".format(run_dir))
    print("summary={}".format(run_dir / "summary.json"))
    print("success={}".format(summary.get("success")))
    if summary.get("failure"):
        print("failure={}".format(summary["failure"]), file=sys.stderr)
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
