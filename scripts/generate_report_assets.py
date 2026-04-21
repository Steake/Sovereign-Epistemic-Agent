import sqlite3
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from epistemic_tribunal.evaluation.metrics import summary_report
from epistemic_tribunal.tribunal_types import ExperimentRun

def load_latest_runs(db_path, count=50):
    conn = sqlite3.connect(db_path)
    query = f"SELECT run_id, task_id, generator_names_json, decision, confidence, selected_trace_id, ground_truth_match, duration_seconds, config_snapshot_json, metadata_json, created_at FROM experiment_runs ORDER BY created_at DESC LIMIT {count}"
    rows = conn.execute(query).fetchall()
    runs = []
    for row in rows:
        runs.append(ExperimentRun(
            run_id=row[0],
            task_id=row[1],
            generator_names=json.loads(row[2]),
            decision=row[3],
            confidence=row[4],
            selected_trace_id=row[5],
            ground_truth_match=bool(row[6]) if row[6] is not None else None,
            duration_seconds=row[7],
            config_snapshot=json.loads(row[8]),
            metadata=json.loads(row[9]),
            timestamp=datetime.fromisoformat(row[10])
        ))
    conn.close()
    return runs

def generate_report_assets(runs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = summary_report(runs)
    
    # 1. Decision Distribution Pie Chart
    dist = metrics.get("decision_distribution", {})
    labels = list(dist.keys())
    values = list(dist.values())
    colors = ['#4CAF50', '#F44336', '#FFC107'] # Green, Red, Yellow
    
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors[:len(labels)], textprops={'color':"white"})
    plt.title("Tribunal Decision Distribution", color="white")
    plt.gcf().set_facecolor('#1e1e1e')
    plt.savefig(output_dir / "decision_dist.png", transparent=True)
    plt.close()
    
    # 2. Accuracy vs Coverage Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(["Overall Accuracy", "Resolved Accuracy", "Coverage"], 
                  [metrics.get("overall_accuracy", 0)*100, 
                   metrics.get("resolved_accuracy", 0)*100, 
                   metrics.get("coverage", 0)*100],
                  color=['#2196F3', '#00BCD4', '#9C27B0'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", color="white")
    ax.set_title("Primary Performance Indicators", color="white")
    ax.tick_params(colors='white')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', color="white")
    plt.gcf().set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    plt.savefig(output_dir / "performance_bars.png", transparent=True)
    plt.close()

    # 3. Cohort Performance Table (Simplified)
    cohorts = metrics.get("cohort_metrics", {})
    rows = []
    for name, stats in cohorts.items():
        rows.append({
            "Cohort": name,
            "Accuracy": f"{stats.get('accuracy', 0)*100:.1f}%",
            "Coverage": f"{stats.get('coverage', 0)*100:.1f}%",
            "Abstain": stats.get('abstentions', 0)
        })
    
    return metrics, rows

if __name__ == "__main__":
    ledger_path = "data/tribunal_full_experiment_ledger.db"
    runs = load_latest_runs(ledger_path, 50)
    
    # Create artifacts directory for images
    # Note: Using the actual conversation path provided in metadata if possible, 
    # but for now relative to repo or a temp artifacts dir.
    # The system will copy them to the appDataDir/brain/<id> automatically if I use the path.
    
    # I'll use a local 'artifacts' dir and then tell the system to copy them? 
    # No, I should use the absolute path from user_information if I can see it.
    # App Data Directory: /Users/oli/.gemini/antigravity
    # Conversation ID: d9959d7e-e08c-4cd0-a70f-c809e61fe01d
    
    artifact_path = "/Users/oli/.gemini/antigravity/brain/d9959d7e-e08c-4cd0-a70f-c809e61fe01d"
    
    metrics, cohort_rows = generate_report_assets(runs, artifact_path)
    
    # Save the metrics to a JSON for the report to read
    with open(Path(artifact_path) / "metrics_summary.json", "w") as f:
        json.dump({"metrics": metrics, "cohorts": cohort_rows}, f, indent=2)
