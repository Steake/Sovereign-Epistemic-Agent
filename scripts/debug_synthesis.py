import os
import sys
import json
from pathlib import Path

# Ensure src/ is in PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

from epistemic_tribunal.config import load_config
from epistemic_tribunal.generators.synthesis import ProgramSynthesisGenerator
from epistemic_tribunal.tasks.arc_like import load_task_from_file

def dry_run():
    # 1. Load config
    config_path = "configs/deepseek_synthesis_chat.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Config {config_path} not found.")
        return
    config = load_config(config_path)

    # 2. Setup Generator
    synth_cfg = config.generators.configs.get("synthesis", {})
    generator = ProgramSynthesisGenerator(**synth_cfg)

    # 3. Load Task
    dataset_path = os.environ.get("ARC_DATASET_PATH", "/Users/oli/Documents/Oli Works/arc_epistemic/Kaggle_Submission_Bundle/dataset")
    task_id = "control_rotate90_a"
    task_file = Path(dataset_path) / f"{task_id}.json"
    
    if not task_file.exists():
        # Search recursively
        matches = list(Path(dataset_path).glob(f"**/{task_id}.json"))
        if not matches:
            print(f"ERROR: Task {task_id} not found in {dataset_path}")
            return
        task_file = matches[0]

    task = load_task_from_file(task_file)
    print(f"Loaded Task: {task.task_id}")

    # 4. Generate candidate
    print("Starting Synthesis...")
    try:
        def on_token(t, text):
            # Print a dot for each reasoning token to show activity
            if t == "reasoning": sys.stdout.write("."); sys.stdout.flush()

        trace = generator.generate(task, on_token=on_token)
        print("\n\n--- SUCCESS ---")
        print(f"Generator: {trace.generator_name}")
        print(f"Confidence: {trace.confidence_score}")
        print(f"Reasoning Steps: {len(trace.reasoning_steps)}")
        print(f"Code Found:\n{trace.metadata.get('code')}")
        print("Answer Grid Shape:", len(trace.answer), "x", len(trace.answer[0]) if trace.answer else 0)
    except Exception as e:
        print("\n\n--- FAILED ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dry_run()
