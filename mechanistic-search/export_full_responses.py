"""
Exports the full-answer role-patching responses to one CSV: one row per (model, role, review, framing), with the clean
answers (run 1 with the role, run 2 bare claim) and run 2's patched answers at each saved setup/layer.

    python export_full_responses.py [--prompt question_only|articles] [--model_tags ...] [--output_path ...]
"""
import argparse
import csv
import json
import os

from run_role_patching import ROLES, SETUPS, PROMPTS


def main(args):
    run_dir = "role_patching" if args.prompt == "articles" else f"role_patching_{args.prompt}"
    output_path = args.output_path or os.path.join(args.outputs_dir, f"{run_dir}_full_responses.csv")
    rows, patched_columns = [], []
    for model_tag in args.model_tags:
        for role in ROLES:
            path = os.path.join(args.outputs_dir, model_tag, run_dir, "full", role, "pairs.jsonl")
            if not os.path.exists(path):
                print(f"Missing {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            for r in records:
                row = {
                    "model": model_tag,
                    "role": role,
                    "review_id": r["review_id"],
                    "framing": r["framing"],
                    "run1_question": r["source_question"],
                    "run2_question": r["target_question"],
                    "run1_answer": r["run1_answer"],
                    "run2_answer": r["run2_answer"],
                    "d_run1": r["d_source"],
                    "d_run2": r["d_target"],
                }
                for setup in SETUPS:
                    for layer, answer in sorted(r.get("patched_run2_answers", {}).get(setup, {}).items(), key=lambda kv: int(kv[0])):
                        column = f"patched_{setup}_L{layer}"
                        row[column] = answer
                        if column not in patched_columns:
                            patched_columns.append(column)
                rows.append(row)

    columns = ["model", "role", "review_id", "framing", "run1_question", "run2_question", "run1_answer", "run2_answer",
               "d_run1", "d_run2"] + patched_columns
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export full-answer role-patching responses to CSV")
    parser.add_argument("--model_tags", nargs="+", default=["llama3-8B-instruct", "olmo2-7B-instruct"])
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--prompt", default="question_only", choices=PROMPTS, help="question_only (role_patching_question_only/) or articles (role_patching/).")
    parser.add_argument("--output_path", default=None, help="default: <outputs_dir>/<run dir>_full_responses.csv")
    main(parser.parse_args())
