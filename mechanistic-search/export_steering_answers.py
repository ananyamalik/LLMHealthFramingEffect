"""
Writes the written answers of run_diffmeans_steering.py --answer full to a CSV, one row per answer.

For each <dir>/pairs.jsonl given (<dir> = outputs/<model_tag>/diffmeans_steering/full/<positions>), writes
<out_dir>/<model_tag>_steering_full_<positions>_responses.csv with columns:
    review_id, framing, question, prompt (bare / doctor / friend / enemy / steered), condition (e.g. framing|-1.0,
    empty for unsteered), layer (empty for unsteered), d (yes/no follow-up D), unchanged (steered answer identical to
    the bare claim's answer), answer
Usage: python export_steering_answers.py outputs/olmo2-7B-instruct/diffmeans_steering/full/*/
"""
import argparse
import csv
import json
import os

COLUMNS = ["review_id", "framing", "question", "prompt", "condition", "layer", "d", "unchanged", "answer"]


def rows(record: dict):
    base = {"review_id": record["review_id"], "framing": record["framing"], "question": record["question"]}
    yield base | {"prompt": "bare", "condition": "", "layer": "", "d": record["d_clean"], "unchanged": "",
                  "answer": record["clean_answer"]}
    for role, answer in record["role_answers"].items():
        yield base | {"prompt": role, "condition": "", "layer": "", "d": record["d_roles"][role], "unchanged": "",
                      "answer": answer}
    for cond, by_layer in record["steered_answers"].items():
        for layer, answer in by_layer.items():
            yield base | {"prompt": "steered", "condition": cond, "layer": layer, "d": record["d_steered"][cond][layer],
                          "unchanged": answer == record["clean_answer"], "answer": answer}


def main(args):
    for d in args.dirs:
        with open(os.path.join(d, "pairs.jsonl"), "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        parts = os.path.normpath(d).split(os.sep)
        name = f"{parts[-4]}_steering_{parts[-2]}_{parts[-1]}_responses.csv"   # e.g. olmo2-7B-instruct_steering_full_last
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            n = 0
            for record in records:
                for row in rows(record):
                    writer.writerow(row)
                    n += 1
        print(f"Saved {path} ({n} answers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the steered written answers to CSV")
    parser.add_argument("dirs", nargs="+", help="output dirs of run_diffmeans_steering.py --answer full (with pairs.jsonl).")
    parser.add_argument("--out_dir", default="outputs", help="where the CSVs go.")
    main(parser.parse_args())
