"""
Writes the written answers of run_diffmeans_steering.py --answer full to a CSV, one row per answer.

For each <dir>/pairs.jsonl given (<dir> = outputs/<model_tag>/diffmeans_steering/full/<positions>), writes
<out_dir>/<model_tag>_steering_full_<positions>_responses.csv with columns:
    review_id, framing, question, prompt (bare / doctor / friend / enemy / steered), condition (e.g. framing|-1.0,
    empty for unsteered), layer (empty for unsteered), d (yes/no follow-up D: log p(Yes) - log p(No) after the answer;
    > 0 = says it works), d_change (d minus the bare claim's d for the same question), unchanged (answer identical to
    the bare claim's answer), answer
Usage: python export_steering_answers.py outputs/olmo2-7B-instruct/diffmeans_steering/full/*/
"""
import argparse
import csv
import json
import os

COLUMNS = ["review_id", "framing", "question", "prompt", "condition", "layer", "d", "d_change", "unchanged", "answer"]


def rows(record: dict):
    base = {"review_id": record["review_id"], "framing": record["framing"], "question": record["question"]}
    clean_d, clean_answer = record["d_clean"], record["clean_answer"]
    yield base | {"prompt": "bare", "condition": "", "layer": "", "d": clean_d, "d_change": 0.0, "unchanged": True,
                  "answer": clean_answer}
    for role, answer in record["role_answers"].items():
        d = record["d_roles"][role]
        yield base | {"prompt": role, "condition": "", "layer": "", "d": d, "d_change": d - clean_d,
                      "unchanged": answer == clean_answer, "answer": answer}
    for cond, by_layer in record["steered_answers"].items():
        for layer, answer in by_layer.items():
            d = record["d_steered"][cond][layer]
            yield base | {"prompt": "steered", "condition": cond, "layer": layer, "d": d, "d_change": d - clean_d,
                          "unchanged": answer == clean_answer, "answer": answer}


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
