"""
Exports the role-patching outputs to three CSVs (in --outputs_dir, named after the run dir, e.g.
role_patching_question_only_*.csv):

    *_full_responses.csv   one row per (model, role, review, framing): the clean full answers (run 1 with the role,
                           run 2 bare claim) and run 2's patched answers at each saved setup/layer
    *_yes_no_pairs.csv     one row per (model, role, review, framing): yes/no scores and top tokens for both runs
    *_recovery.csv         one row per (model, answer, role, framing group, setup, layer): mean/SEM/median recovery

    python export_full_responses.py [--prompt question_only|articles] [--patch_generation] [--model_tags ...]
"""
import argparse
import csv
import json
import os

from run_role_patching import ROLES, SETUPS, ANSWERS, PROMPTS


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


def base_row(model_tag: str, role: str, r: dict) -> dict:
    return {
        "model": model_tag,
        "role": role,
        "review_id": r["review_id"],
        "framing": r["framing"],
        "run1_question": r["source_question"],
        "run2_question": r["target_question"],
    }


def main(args):
    run_dir = "role_patching" if args.prompt == "articles" else f"role_patching_{args.prompt}"
    if args.patch_generation:
        run_dir += "_every_token"
    full_rows, yes_no_rows, recovery_rows, patched_columns = [], [], [], []

    for model_tag in args.model_tags:
        for answer in ANSWERS:
            for role in ROLES:
                role_dir = os.path.join(args.outputs_dir, model_tag, run_dir, answer, role)
                pairs_path, summary_path = os.path.join(role_dir, "pairs.jsonl"), os.path.join(role_dir, "summary.json")
                if not os.path.exists(pairs_path):
                    print(f"Missing {pairs_path}")
                    continue

                for r in load_records(pairs_path):
                    row = base_row(model_tag, role, r)
                    if answer == "full":
                        row |= {"run1_answer": r["run1_answer"], "run2_answer": r["run2_answer"],
                                "d_run1": r["d_source"], "d_run2": r["d_target"]}
                        for setup in SETUPS:
                            patched = r.get("patched_run2_answers", {}).get(setup, {})
                            for layer, text in sorted(patched.items(), key=lambda kv: int(kv[0])):
                                column = f"patched_{setup}_L{layer}"
                                row[column] = text
                                if column not in patched_columns:
                                    patched_columns.append(column)
                        full_rows.append(row)
                    else:
                        row |= {"run1_top_token": r["source_top_token"].strip(), "run2_top_token": r["target_top_token"].strip(),
                                "d_run1": r["d_source"], "d_run2": r["d_target"], "role_effect": r["d_source"] - r["d_target"]}
                        yes_no_rows.append(row)

                if os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    for group, by_setup in summary["recovery"].items():
                        for setup, by_layer in by_setup.items():
                            for layer, stats in by_layer.items():
                                recovery_rows.append({"model": model_tag, "answer": answer, "role": role, "framing": group,
                                                      "setup": setup, "layer": int(layer), **stats})

    write_csv(os.path.join(args.outputs_dir, f"{run_dir}_full_responses.csv"),
              ["model", "role", "review_id", "framing", "run1_question", "run2_question", "run1_answer", "run2_answer",
               "d_run1", "d_run2"] + patched_columns, full_rows)
    write_csv(os.path.join(args.outputs_dir, f"{run_dir}_yes_no_pairs.csv"),
              ["model", "role", "review_id", "framing", "run1_question", "run2_question", "run1_top_token", "run2_top_token",
               "d_run1", "d_run2", "role_effect"], yes_no_rows)
    write_csv(os.path.join(args.outputs_dir, f"{run_dir}_recovery.csv"),
              ["model", "answer", "role", "framing", "setup", "layer", "n", "mean", "sem", "median"], recovery_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export role-patching outputs to CSV")
    parser.add_argument("--model_tags", nargs="+", default=["llama3-8B-instruct", "olmo2-7B-instruct"])
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--prompt", default="question_only", choices=PROMPTS, help="question_only (role_patching_question_only/) or articles (role_patching/).")
    parser.add_argument("--patch_generation", action="store_true", help="read the every-token run (<run_dir>_every_token/).")
    main(parser.parse_args())
