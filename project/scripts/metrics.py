import os
import json
import argparse
import pandas as pd

def parse_judges(judge_str):
    """
    Parse a judge string into a list of uppercase judgment labels.
    Expected format per line: "1. TRUE" (or FALSE / UNKNOWN).
    """
    lines = judge_str.strip().splitlines()
    results = []
    for line in lines:
        # Split on the period and take the judgment part
        parts = line.split('.')
        if len(parts) >= 2:
            result = parts[1].strip().upper()
            results.append(result)
    return results

def compute_metrics(judges):
    """
    Compute metrics for a single entry given a list of judge labels.
    Returns a dictionary with counts and rates.
    """
    count_true = judges.count("TRUE")
    count_false = judges.count("FALSE")
    count_unknown = judges.count("UNKNOWN")
    total = len(judges)
    
    accuracy = count_true / total if total > 0 else 0
    false_rate = count_false / total if total > 0 else 0
    unknown_rate = count_unknown / total if total > 0 else 0

    return {
        "true": count_true,
        "false": count_false,
        "unknown": count_unknown,
        "total": total,
        "accuracy": accuracy,
        "false_rate": false_rate,
        "unknown_rate": unknown_rate
    }

def process_file(file_path):
    """
    Load the JSON file from the given path and compute metrics for each entry.
    Each entry is expected to have a 'judge' key.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metrics_per_entry = []
    for entry in data:
        judge_str = entry.get("judge", "")
        judges = parse_judges(judge_str)
        entry_metrics = compute_metrics(judges)
        # Retain the entry id for reference if available
        entry_metrics["id"] = entry.get("id", None)
        metrics_per_entry.append(entry_metrics)
    return metrics_per_entry

def aggregate_metrics(metrics_list):
    """
    Aggregate per-entry metrics into overall metrics.
    Computes overall (micro) accuracy and the macro average of per-entry accuracy.
    """
    total_entries = len(metrics_list)
    if total_entries == 0:
        return None

    sum_true = sum(m["true"] for m in metrics_list)
    sum_false = sum(m["false"] for m in metrics_list)
    sum_unknown = sum(m["unknown"] for m in metrics_list)
    sum_total = sum(m["total"] for m in metrics_list)

    overall_accuracy = sum_true / sum_total if sum_total > 0 else 0
    overall_false_rate = sum_false / sum_total if sum_total > 0 else 0
    overall_unknown_rate = sum_unknown / sum_total if sum_total > 0 else 0

    # Macro accuracy: average accuracy per entry
    macro_accuracy = sum(m["accuracy"] for m in metrics_list) / total_entries

    return {
        "total_entries": total_entries,
        "sum_true": sum_true,
        "sum_false": sum_false,
        "sum_unknown": sum_unknown,
        "sum_total_judgments": sum_total,
        "overall_accuracy": overall_accuracy,
        "overall_false_rate": overall_false_rate,
        "overall_unknown_rate": overall_unknown_rate,
        "macro_accuracy": macro_accuracy
    }

def main():
    parser = argparse.ArgumentParser(description="Llama Q&A Benchmark Metrics Calculation")
    parser.add_argument("--topic", 
                        choices=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science", "test"],
                        required=True,
                        help="Select the topic file to process")
    parser.add_argument("--results-dir", 
                        default="./results",
                        help="Directory containing the results JSON files")
    parser.add_argument("--output-excel", 
                        action="store_true",
                        help="Export metrics to Excel files")
    args = parser.parse_args()

    file_name = f"{args.topic}.json"
    file_path = os.path.join(args.results_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Process the file and compute per-entry metrics
    metrics_list = process_file(file_path)
    agg_metrics = aggregate_metrics(metrics_list)

    # Display per-entry metrics
    print("Per-entry Metrics:")
    for m in metrics_list:
        print(f"ID: {m['id']} - Accuracy: {m['accuracy']:.2f}, False Rate: {m['false_rate']:.2f}, Unknown Rate: {m['unknown_rate']:.2f}")

    # Display aggregated metrics
    print("\nAggregate Metrics:")
    print(f"Total entries: {agg_metrics['total_entries']}")
    print(f"Total Judgments: {agg_metrics['sum_total_judgments']}")
    print(f"Overall Accuracy: {agg_metrics['overall_accuracy']*100:.2f}%")
    print(f"Overall False Rate: {agg_metrics['overall_false_rate']*100:.2f}%")
    print(f"Overall Unknown Rate: {agg_metrics['overall_unknown_rate']*100:.2f}%")
    print(f"Macro Accuracy: {agg_metrics['macro_accuracy']*100:.2f}%")

    # Optionally export metrics to Excel
    if args.output_excel:
        df_entries = pd.DataFrame(metrics_list)
        df_entries.to_excel(f"{args.topic}_per_entry_metrics.xlsx", index=False)
        df_agg = pd.DataFrame([agg_metrics])
        df_agg.to_excel(f"{args.topic}_aggregate_metrics.xlsx", index=False)
        print("Metrics exported to Excel files.")

if __name__ == "__main__":
    main()
