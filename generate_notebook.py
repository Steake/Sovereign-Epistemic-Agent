import json

def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def make_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

notebook = {
    "cells": [
        make_markdown_cell("# Epistemic ARC Forensic Analysis\n## Qwen-27B Scaling & Adjudication Study\n\nThis notebook analyzes the crystallized data from the H100 Scale-up Mission. It evaluates reasoning budgets, adjudication performance, and candidate closeness to ground truth."),
        
        make_code_cell("""import json\nimport sqlite3\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom pathlib import Path\n\n# Setup plotting style\nsns.set_theme(style="whitegrid", palette="muted")\nplt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 120, 'font.size': 12})\n\nDATA_DIR = Path("./data/pod_crystallization/data")\nprint(f"Data Source Directory: {DATA_DIR.resolve()}")"""),
        
        make_markdown_cell("### 1. Scaling Reasoning Budget (0 - 2048 Tokens)\nDid increasing the CoT reasoning budget improve grid accuracy? We look at overall accuracy and confidence vs. reasoning budget."),
        
        make_code_cell("""budgets = [0, 256, 512, 1024, 2048]\nbudget_data = []\n\nfor b in budgets:\n    path = DATA_DIR / f"smoke_sweep_budget_{b}.json"\n    if path.exists():\n        with open(path, 'r') as f:\n            b_data = json.load(f)\n            budget_data.append({\n                "Budget": b,\n                "Accuracy": b_data.get("Greedy", {}).get("overall_accuracy", 0.0),\n                "Invalid Shape Count": b_data.get("Greedy", {}).get("grid_shape_invalid_count", 0),\n                "Mean Confidence": b_data.get("Greedy", {}).get("mean_confidence", 0.0)\n            })\n\nif budget_data:\n    df_budget = pd.DataFrame(budget_data)\n    display(df_budget)\n\n    fig, ax1 = plt.subplots()\n    ax2 = ax1.twinx()\n    \n    sns.lineplot(data=df_budget, x="Budget", y="Mean Confidence", ax=ax1, marker="o", color="royalblue", label="Mean Confidence")\n    sns.lineplot(data=df_budget, x="Budget", y="Accuracy", ax=ax2, marker="s", color="darkorange", label="Accuracy")\n    \n    ax1.set_ylabel("Mean Confidence", color="royalblue")\n    ax2.set_ylabel("Accuracy", color="darkorange")\n    ax2.set_ylim(-0.05, 1.05)\n    \n    # Combine legends\n    lines_1, labels_1 = ax1.get_legend_handles_labels()\n    lines_2, labels_2 = ax2.get_legend_handles_labels()\n    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')\n    \n    plt.title("Reasoning Budget (Tokens) vs Performance")\n    plt.grid(False, axis='y') # avoid overlapping grids\n    plt.show()\nelse:\n    print("Budget sweep data files not found in target path.")"""),

        make_markdown_cell("### 2. Multi-Arm Adjudication Performance\nComparing Greedy, Structural, Lockout, and Path B strategies on a fixed 512-budget validation run."),

        make_code_cell("""val_results_path = DATA_DIR / "validation_results_v1.json"\nif val_results_path.exists():\n    with open(val_results_path, 'r') as f:\n        val_res = json.load(f)\n        \n    arms_data = []\n    for arm_name, metrics in val_res.items():\n        arms_data.append({\n            "Arm": arm_name,\n            "Accuracy": metrics.get("overall_accuracy", 0),\n            "Coverage": metrics.get("coverage", 0),\n            "Resample Rate": metrics.get("resample_rate", 0),\n            "Brier Score": metrics.get("brier_score", 0),\n        })\n\n    df_arms = pd.DataFrame(arms_data)\n    display(df_arms)\n\n    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n    sns.barplot(data=df_arms, x="Arm", y="Coverage", ax=axes[0], palette="Blues_d")\n    axes[0].set_title("Task Coverage by Adjudication Arm")\n    axes[0].set_ylim(0, 1.1)\n\n    sns.barplot(data=df_arms, x="Arm", y="Resample Rate", ax=axes[1], palette="flare")\n    axes[1].set_title("Resample Override Rate by Arm")\n    axes[1].set_ylim(0, 1.1)\n    \n    plt.tight_layout()\n    plt.show()\nelse:\n    print("validation_results_v1.json not found.")"""),
        
        make_markdown_cell("### 3. Forensic Closeness: The Generator Bottleneck\nWe extracted mathematical grid overlap % between the Model's candidates and ground truth. If maximum pool overlap is 0%, no adjudication mechanism can retrieve the correct answer, indicating fundamental generator failure for that task type."),
        
        make_code_cell("""forensic_path = Path("./data/pod_crystallization/forensic_raw_data.json")\nif forensic_path.exists():\n    with open(forensic_path, 'r') as f:\n        forensic_raw = json.load(f)\n        \n    f_data = []\n    for task_id, info in forensic_raw.items():\n        f_data.append({\n            "Task Archetype": info.get("type", "Unknown"),\n            "Best Candidate Overlap": info.get("best_overlap", 0),\n            "Tribunal Selected Overlap": info.get("tribunal_overlap", 0),\n            "Pool Size": info.get("pool_size", 0)\n        })\n        \n    df_forensic = pd.DataFrame(f_data)\n    display(df_forensic.style.format({"Best Candidate Overlap": "{:.1%}", "Tribunal Selected Overlap": "{:.1%}"}))\n\n    df_melt = df_forensic.melt(id_vars=["Task Archetype"], \n                               value_vars=["Best Candidate Overlap", "Tribunal Selected Overlap"],\n                               var_name="Metric", value_name="Overlap Pct")\n                               \n    plt.figure(figsize=(10, 6))\n    sns.barplot(data=df_melt, x="Task Archetype", y="Overlap Pct", hue="Metric", palette="Set2")\n    plt.title("Grid Math-Closeness: Best in Pool vs Tribunal Selected")\n    plt.ylim(0, 1.05)\n    plt.axhline(1.0, ls='--', color='red', label="Perfect Truth (100%)", zorder=0)\n    plt.ylabel("Percentage of Cells Overlapping")\n    plt.legend(loc="upper left")\n    plt.show()\nelse:\n    print("forensic_raw_data.json not found.")"""),

        make_markdown_cell("### 4. Ledger Analysis: Uncertainty Signals\nExploring the database ledger to track decision confidences across the Structural pipeline."),
        
        make_code_cell("""# Load Path B Strict database\ndb_path = DATA_DIR / "tribunal_ledger_path_b_strict.db"\nif db_path.exists():\n    conn = sqlite3.connect(db_path)\n    \n    query = \"\"\"\n    SELECT task_id, decision, confidence\n    FROM decisions\n    \"\"\"\n    df_decisions = pd.read_sql(query, conn)\n    \n    if not df_decisions.empty:\n        plt.figure(figsize=(8, 5))\n        sns.histplot(data=df_decisions, x="confidence", hue="decision", bins=15, multiple="stack", palette="husl")\n        plt.title("Adjudicator Confidence vs Final Decision Outcome (Path B Strict)")\n        plt.xlabel("Assigned Confidence Score")\n        plt.show()\n    \n    conn.close()\nelse:\n    print(f"DB not found at {db_path}")"""),

        make_markdown_cell("### 💡 Key Strategic Insights\n1. **Zero Yield on Compute Growth**: Scaling token allowance to 2048 showed absolutely no signal growth for basic geometric relations. The model fails to conceptualize tasks directly into the JSON array space reliably.\n2. **Adjudication is Working**: As shown in the *Messy* archetype, whenever there was an optimal near-miss (90%), the Epistemic Tribunal correctly found it and selected it. \n3. **Pivot Required**: Since we've hit the LLM generator's intelligence floor for pure JSON array operations, the next strategy should shift to generative **programs** (e.g., DSLs and Python arrays) where the Tribunal adjudicates execution output rather than immediate text output.")
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("data/Epistemic_ARC_Analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully at data/Epistemic_ARC_Analysis.ipynb")
