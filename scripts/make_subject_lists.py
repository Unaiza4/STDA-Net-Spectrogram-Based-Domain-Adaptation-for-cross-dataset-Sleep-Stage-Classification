import argparse
import csv
import os
import re
from pathlib import Path


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def extract_subject_id(name: str) -> str:
    if name.startswith("SC"):
        match = re.match(r"^(SC\d{3})", name)
        if match:
            return match.group(1)
        return name[:5]
    return name


def list_recording_folders(root_dir: str):
    return sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith("_")],
        key=natural_key,
    )


def write_list_file(subject_ids, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sid in subject_ids:
            f.write(f"{sid}\n")


def read_shhs_healthy_ids(
    csv_path: str,
    ahi_column="nsrr_ahi_hp3u",
    scoring_column="nsrr_flag_spsw",
    valid_scoring="full scoring",
    ahi_threshold=5.0,
):
    healthy = set()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        delim = "\t" if "\t" in first else ","
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            sid = str(row.get("nsrrid", "")).strip()
            if not sid:
                continue
            scoring = str(row.get(scoring_column, "")).strip().lower()
            if scoring != valid_scoring.lower():
                continue
            try:
                ahi = float(str(row.get(ahi_column, "")).strip())
            except Exception:
                continue
            if ahi < ahi_threshold:
                healthy.add(sid)
    return healthy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_root", type=str, required=True)
    parser.add_argument("--shhs_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--edf_count", type=int, default=20)
    parser.add_argument("--shhs_count", type=int, default=42)
    parser.add_argument("--shhs_csv", type=str, default=None)
    parser.add_argument("--ahi_threshold", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_recordings = list_recording_folders(args.edf_root)
    edf_subjects = sorted({extract_subject_id(r) for r in edf_recordings}, key=natural_key)[:args.edf_count]

    shhs_recordings = list_recording_folders(args.shhs_root)
    if args.shhs_csv:
        healthy_ids = read_shhs_healthy_ids(args.shhs_csv, ahi_threshold=args.ahi_threshold)
        shhs_recordings = [
            r for r in shhs_recordings
            if (re.search(r"(\d+)$", r) and re.search(r"(\d+)$", r).group(1) in healthy_ids)
        ]
    shhs_subjects = sorted(shhs_recordings, key=natural_key)[:args.shhs_count]

    edf_path = out_dir / "edf20.txt"
    shhs_path = out_dir / "shhs1_42.txt"

    write_list_file(edf_subjects, str(edf_path))
    write_list_file(shhs_subjects, str(shhs_path))

    print(f"Saved EDF list : {edf_path}")
    print(f"Saved SHHS list: {shhs_path}")


if __name__ == "__main__":
    main()
