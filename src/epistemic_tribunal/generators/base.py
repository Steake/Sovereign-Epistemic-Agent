"""Abstract base class for candidate-trace generators.

All generators must implement :meth:`BaseGenerator.generate` and expose a
``name`` attribute.  The interface is kept intentionally thin so that
concrete implementations can range from simple heuristics to full LLM calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from epistemic_tribunal.tribunal_types import CandidateTrace, Task


class BaseGenerator(ABC):
    """Abstract generator that produces a :class:`CandidateTrace` from a task.

    Subclasses override :meth:`generate`.  The orchestrator calls each
    registered generator in turn to build the candidate pool.
    """

    #: Unique, stable name used in configuration and logs.
    name: str = "base"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self.seed = seed
        self._extra_kwargs = kwargs

    @abstractmethod
    def generate(self, task: Task) -> CandidateTrace:
        """Produce one candidate reasoning trace for *task*.

        Parameters
        ----------
        task:
            The task to solve.

        Returns
        -------
        CandidateTrace
            A fully populated candidate including answer, reasoning steps, and
            any derived feature metadata.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, seed={self.seed})"


def build_generators(
    enabled: list[str],
    seed: int = 42,
    generator_configs: Optional[dict[str, dict[str, Any]]] = None,
) -> list[BaseGenerator]:
    """Instantiate all enabled generators by name.

    Parameters
    ----------
    enabled:
        List of generator names (as registered in :data:`GENERATOR_REGISTRY`).
    seed:
        Shared RNG seed.

    Returns
    -------
    list[BaseGenerator]
        Instantiated generator objects in the same order as *enabled*.
    """
    from epistemic_tribunal.generators.greedy import GreedyGenerator
    from epistemic_tribunal.generators.diverse import DiverseGenerator
    from epistemic_tribunal.generators.adversarial import AdversarialGenerator
    from epistemic_tribunal.generators.rule_first import RuleFirstGenerator
    from epistemic_tribunal.generators.minimal import MinimalDescriptionGenerator

    from epistemic_tribunal.generators.llm import LLMGenerator, OpenAIGenerator

    REGISTRY: dict[str, type[BaseGenerator]] = {
        "greedy": GreedyGenerator,
        "diverse": DiverseGenerator,
        "adversarial": AdversarialGenerator,
        "rule_first": RuleFirstGenerator,
        "minimal_description": MinimalDescriptionGenerator,
        "llm": LLMGenerator,
        "openai": OpenAIGenerator,
        "vllm": OpenAIGenerator,
    }

    generators: list[BaseGenerator] = []
    generator_configs = generator_configs or {}
    for name in enabled:
        config = generator_configs.get(name, {})
        # Allow overriding the registry lookup via 'type' key
        lookup_name = str(config.get("type", name))
        cls = REGISTRY.get(lookup_name)
        if cls is None:
            raise ValueError(
                f"Unknown generator {name!r} (resolved as {lookup_name!r}). "
                f"Available types: {sorted(REGISTRY.keys())}"
            )
        generators.append(cls(seed=seed, **config))
    return generators
