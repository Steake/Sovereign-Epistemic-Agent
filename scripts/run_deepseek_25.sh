#!/usr/bin/env bash
# Run 25-task ARC benchmark against DeepSeek Reasoner API
set -euo pipefail

export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_API_KEY="sk-08e83c1210624090a59fee70318f2d95"
export ARC_DATASET_PATH="$(pwd)/data/pod_crystallization/data/validation_set/tasks"
export PYTHONPATH="$(pwd)/src"

echo "=== DeepSeek 25-Task Benchmark ==="
echo "API: $OPENAI_BASE_URL"
echo "Dataset: $ARC_DATASET_PATH"
echo "Config: configs/deepseek_25.yaml"
echo "Manifest: data/deepseek_manifest_25.txt"
echo ""

# Single-arm run using the full_validation infrastructure
python3 -c "
import json, time, sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

from epistemic_tribunal.config import load_config
from epistemic_tribunal.evaluation.benchmark import BenchmarkRunner
from epistemic_tribunal.evaluation.metrics import summary_report

console = Console()
config = load_config('configs/deepseek_25.yaml')

# Clear old ledger
db = Path(config.ledger.path)
if db.exists():
    db.unlink()

runner = BenchmarkRunner(config=config)

console.print('[bold blue]Running 25 tasks against DeepSeek Reasoner...[/bold blue]')
start = time.monotonic()
runs = runner.run(
    dataset_path=Path('$ARC_DATASET_PATH'),
    manifest_path='data/deepseek_manifest_25.txt'
)
elapsed = time.monotonic() - start

if not runs:
    console.print('[bold red]FATAL: 0 tasks ran.[/bold red]')
    sys.exit(1)

metrics = summary_report(runs)

# Display results
table = Table(title=f'DeepSeek Reasoner — {len(runs)} Tasks ({elapsed:.0f}s)')
table.add_column('Metric', style='cyan')
table.add_column('Value', justify='right')

for key in ['overall_accuracy', 'resolved_accuracy', 'coverage', 'resample_rate',
            'wrong_pick_count', 'json_not_found_count', 'json_invalid_count',
            'grid_shape_invalid_count', 'reasoning_bleed_count', 'parse_failure_count']:
    table.add_row(key, str(metrics.get(key, '-')))

console.print(table)

with open('data/deepseek_25_results.json', 'w') as f:
    json.dump({'DeepSeek Reasoner': metrics}, f, indent=2)
console.print('[bold green]Results saved to data/deepseek_25_results.json[/bold green]')
"
