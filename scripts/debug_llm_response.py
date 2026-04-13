#!/usr/bin/env python3
"""Debug: probe raw LLM response for the first manifest task."""
import json
import sys
sys.path.insert(0, "/workspace/Sovereign-Epistemic-Agent/src")

from openai import OpenAI
from pathlib import Path
from epistemic_tribunal.tasks.arc_like import load_task_from_file

client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

task = load_task_from_file(Path("/workspace/arc_dataset/auto_control_0a7d41f9.json"))
print(f"Task: {task.task_id}  test_input shape: {len(task.test_input)}x{len(task.test_input[0]) if task.test_input else 0}")
print(f"Train examples: {len(task.train)}")

prompt = (
    'Solve the ARC-style task and return JSON only.\n'
    'Use this schema exactly:\n'
    '{"answer": [[0]], "reasoning_steps": ["short step"], "confidence": 0.0}\n'
    'You may reason inside <think>...</think> tags.\n'
    'After </think>, you MUST output ONLY the JSON object on its own line — nothing else.\n'
    f'Task ID: {task.task_id}\n'
    f'Test Input: {json.dumps(task.test_input)}\n'
)

print(f"\nSending prompt ({len(prompt)} chars)...")
response = client.chat.completions.create(
    model="Qwen3.5-27B.Q8_0.gguf",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_tokens=2048,
)

msg = response.choices[0].message
text = msg.content or ""
reasoning = getattr(msg, "reasoning_content", None)

print(f"\nFinish reason: {response.choices[0].finish_reason}")
print(f"Content length: {len(text)} chars")
print(f"Reasoning content: {'YES (' + str(len(reasoning)) + ' chars)' if reasoning else 'None'}")
print("\n=== CONTENT (first 800 chars) ===")
print(text[:800])
print("\n=== CONTENT (last 300 chars) ===")
print(text[-300:])
if reasoning:
    print("\n=== REASONING_CONTENT (last 200 chars) ===")
    print(reasoning[-200:])
