import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import yaml

from emg_pipeline import (
    apply_filters,
    resolve_collection_output_dir,
    save_gesture_samples,
    save_metadata_json,
    segment_continuous_signal,
)
from myo_runtime import MyoEMGCollector


def banner(text, char="*", width=72):
    line = char * width
    print("")
    print(line)
    print(text.center(width))
    print(line)


def phase_banner(title, detail=None):
    banner(title, char="=")
    if detail:
        print(detail)


def countdown(seconds, title):
    total = int(seconds)
    if total <= 0:
        return
    banner(f"{title.upper()} START", char="-", width=56)
    for remaining in range(total, 0, -1):
        print(f">>> {title.upper()} : {remaining}s")
        time.sleep(1.0)
    print(f">>> {title.upper()} : done")


def write_protocol_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gesture_id",
                "repetition",
                "rest_frames",
                "action_frames",
                "windows",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def collect_protocol(cfg, gestures, repetitions, action_seconds, rest_seconds, output_dir):
    fs = cfg["data"]["fs"]
    window = cfg["data"]["window"]
    step = cfg["data"]["step"]
    apply_window_filters = cfg["collect"].get("apply_filters", False)

    collector = MyoEMGCollector(cfg)
    print("waiting for Myo connection...")
    if not collector.wait_for_connection(timeout_seconds=cfg["myo"].get("connect_timeout", 15.0)):
        raise RuntimeError("Myo connection timeout")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(output_dir, f"guided_session_{timestamp}")
    raw_dir = os.path.join(session_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    protocol_rows = []

    banner("GUIDED MYO COLLECTION", char="#")
    print(f"gestures={gestures}")
    print(f"repetitions per gesture={repetitions}")
    print(f"action_seconds={action_seconds}")
    print(f"rest_seconds={rest_seconds}")
    print(f"window={window}, step={step}, fs={fs}")
    print(f"dataset_dir={output_dir}")
    print(f"session_dir={session_dir}")

    for gesture_id in gestures:
        gesture_title = "REST CLASS (0)" if gesture_id == 0 else f"GESTURE {gesture_id}"
        phase_banner(gesture_title, "press Enter when you are ready")
        input()

        gesture_action_segments = []

        for rep in range(1, repetitions + 1):
            rep_title = f"REST CLASS (0) | REP {rep}/{repetitions}" if gesture_id == 0 else f"GESTURE {gesture_id} | REP {rep}/{repetitions}"
            banner(rep_title, char="*", width=72)

            if gesture_id != 0 and rest_seconds > 0:
                countdown(rest_seconds, "rest")
                rest_segment = collector.collect_duration(rest_seconds)
            else:
                rest_segment = np.empty((0, cfg["data"]["channel"]), dtype=np.float32)

            countdown(1, "prepare")
            hold_title = "STAY RELAXED NOW" if gesture_id == 0 else "HOLD GESTURE NOW"
            banner(hold_title, char="!", width=72)
            action_segment = collector.collect_duration(action_seconds)
            banner("RELEASE", char=".", width=56)

            if action_segment.shape[0] < window:
                banner("REPETITION SKIPPED", char="x", width=56)
                print(f"not enough frames: {action_segment.shape[0]}")
                protocol_rows.append(
                    {
                        "gesture_id": gesture_id,
                        "repetition": rep,
                        "rest_frames": rest_segment.shape[0],
                        "action_frames": action_segment.shape[0],
                        "windows": 0,
                    }
                )
                continue

            gesture_action_segments.append(action_segment)

            raw_path = os.path.join(raw_dir, f"gesture_{gesture_id}_rep_{rep:03d}.npz")
            np.savez(
                raw_path,
                rest=rest_segment.astype(np.float32),
                action=action_segment.astype(np.float32),
            )

            segmented = segment_continuous_signal(action_segment, window=window, step=step)
            print(
                f"saved repetition {rep}: rest_frames={rest_segment.shape[0]}, "
                f"action_frames={action_segment.shape[0]}, windows={segmented.shape[0]}"
            )
            protocol_rows.append(
                {
                    "gesture_id": gesture_id,
                    "repetition": rep,
                    "rest_frames": rest_segment.shape[0],
                    "action_frames": action_segment.shape[0],
                    "windows": segmented.shape[0],
                }
            )

        if not gesture_action_segments:
            print(f"gesture {gesture_id} produced no valid action segments")
            continue

        all_action = np.concatenate(gesture_action_segments, axis=0)
        np.save(os.path.join(session_dir, f"gesture_{gesture_id}_action_raw.npy"), all_action.astype(np.float32))

        gesture_windows = []
        for segment in gesture_action_segments:
            segmented = segment_continuous_signal(segment, window=window, step=step)
            if segmented.shape[0] > 0:
                gesture_windows.append(segmented)

        if not gesture_windows:
            print(f"gesture {gesture_id} produced no windows")
            continue

        gesture_windows = np.concatenate(gesture_windows, axis=0).astype(np.float32)
        if apply_window_filters:
            gesture_windows = np.asarray([apply_filters(item, cfg) for item in gesture_windows], dtype=np.float32)

        save_path, total_count = save_gesture_samples(session_dir, gesture_id, gesture_windows)
        phase_banner(
            f"GESTURE {gesture_id} COMPLETE",
            f"saved to {save_path}, total samples={total_count}",
        )

    write_protocol_csv(os.path.join(session_dir, "protocol.csv"), protocol_rows)
    save_metadata_json(
        os.path.join(session_dir, "session.json"),
        {
            "timestamp": timestamp,
            "gestures": gestures,
            "repetitions": repetitions,
            "action_seconds": action_seconds,
            "rest_seconds": rest_seconds,
            "fs": fs,
            "window": window,
            "step": step,
            "output_dir": output_dir,
        },
    )
    banner("GUIDED COLLECTION FINISHED", char="#")
    print(f"session_dir={session_dir}")


def parse_gestures(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item) for item in str(value).split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Guided Myo EMG collection with post segmentation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--gestures", default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--action-seconds", type=float, default=None)
    parser.add_argument("--rest-seconds", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--include-rest-class", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(args.config)

    guided_cfg = cfg["collect"].get("guided", {})
    gestures = parse_gestures(args.gestures) if args.gestures else parse_gestures(guided_cfg.get("gestures", [1, 2, 3, 4, 5]))
    include_rest_class = args.include_rest_class or guided_cfg.get("include_rest_class", False)
    if include_rest_class and 0 not in gestures:
        gestures = [0] + gestures
    repetitions = args.repetitions if args.repetitions is not None else guided_cfg.get("repetitions", 50)
    action_seconds = args.action_seconds if args.action_seconds is not None else guided_cfg.get("action_seconds", 1.0)
    rest_seconds = args.rest_seconds if args.rest_seconds is not None else guided_cfg.get("rest_seconds", 1.0)

    output_dir = resolve_collection_output_dir(
        cfg,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split="train",
    )

    collect_protocol(
        cfg=cfg,
        gestures=gestures,
        repetitions=repetitions,
        action_seconds=action_seconds,
        rest_seconds=rest_seconds,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
