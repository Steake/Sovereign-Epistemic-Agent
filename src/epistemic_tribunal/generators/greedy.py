"""Greedy generator — applies the most-frequent transformation seen in training.

Mock logic (no LLM required):
- Find the most common per-cell colour mapping across training pairs.
- Apply that mapping to the test input.
- Report low confidence when mapping coverage is partial.
"""

from __future__ import annotations

import random
from collections import Counter

from epistemic_tribunal.generators.base import BaseGenerator
from epistemic_tribunal.tasks.base import colour_counts, grid_shape, object_count
from epistemic_tribunal.types import CandidateTrace, Task


class GreedyGenerator(BaseGenerator):
    """Applies the most frequent colour-to-colour mapping seen in training."""

    name = "greedy"

    def generate(self, task: Task) -> CandidateTrace:
        rng = random.Random(self.seed)

        # Build colour mapping from training pairs
        mapping: dict[int, Counter] = {}  # src_colour → Counter of dst_colour
        for example in task.train:
            inp = example.input
            out = example.output
            if grid_shape(inp) != grid_shape(out):
                continue
            rows, cols = grid_shape(inp)
            for r in range(rows):
                for c in range(cols):
                    src = inp[r][c]
                    dst = out[r][c]
                    mapping.setdefault(src, Counter())[dst] += 1

        # Derive best mapping
        best_map: dict[int, int] = {}
        for src, counter in mapping.items():
            best_map[src] = counter.most_common(1)[0][0]

        # Apply mapping to test input
        rows, cols = grid_shape(task.test_input)
        answer: list[list[int]] = []
        coverage_hits = 0
        for r in range(rows):
            row_out = []
            for c in range(cols):
                src = task.test_input[r][c]
                dst = best_map.get(src, src)  # identity if no mapping found
                if src in best_map:
                    coverage_hits += 1
                row_out.append(dst)
            answer.append(row_out)

        total_cells = rows * cols
        coverage = coverage_hits / total_cells if total_cells > 0 else 0.0
        confidence = round(0.4 + 0.5 * coverage + rng.uniform(-0.02, 0.02), 4)

        steps = [
            f"Identified {len(mapping)} distinct colour mappings from {len(task.train)} training pairs.",
            f"Derived greedy best mapping: {best_map}.",
            f"Applied mapping to {rows}x{cols} test grid (coverage={coverage:.2f}).",
        ]

        return CandidateTrace(
            generator_name=self.name,
            answer=answer,
            reasoning_steps=steps,
            raw_trace="\n".join(steps),
            token_count=len(" ".join(steps).split()),
            confidence_score=min(1.0, confidence),
            derived_features={
                "colour_mapping": best_map,
                "coverage": coverage,
                "mapping_size": len(best_map),
                "object_count_in": object_count(task.test_input),
                "object_count_out": object_count(answer),
                "colour_counts_out": colour_counts(answer),
            },
        )
