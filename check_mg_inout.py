"""Check that every row whose image path contains an 'mg' directory has in_or_out == 0."""
import argparse
import csv
import glob
import os


def check_file(csv_path):
    total, bad = 0, []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header (has fewer columns than rows, so read in_or_out as the last field)
        for line_no, row in enumerate(reader, start=2):
            if not row or "mg" not in row[0].split("/"):
                continue
            total += 1
            if float(row[-1]) != 0:
                bad.append((line_no, row[0], row[-1]))
    return total, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    all_ok = True
    for csv_path in sorted(glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True)):
        total, bad = check_file(csv_path)
        status = "OK" if not bad else "FAIL"
        print(f"[{status}] {csv_path}: {total} mg rows, {len(bad)} with in_or_out != 0")
        for line_no, path, value in bad[:20]:
            print(f"    line {line_no}: in_or_out={value} {path}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
        all_ok &= not bad

    print("All mg rows have in_or_out == 0" if all_ok else "Some mg rows have in_or_out != 0")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
