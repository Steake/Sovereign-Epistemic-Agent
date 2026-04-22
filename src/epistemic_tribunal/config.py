"""Configuration loader for Epistemic Tribunal.

Reads YAML config files and exposes typed Pydantic settings.
Environment variables override YAML values when prefixed with TRIBUNAL_.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------


class TribunalWeights(BaseModel):
    """Scoring weights for the tribunal aggregator.  Must be positive; they are
    normalised internally so they do not need to sum to 1.0.
    """

    uncertainty: float = Field(default=0.25, ge=0.0)
    critic: float = Field(default=0.35, ge=0.0)
    memory: float = Field(default=0.15, ge=0.0)
    invariant: float = Field(default=0.25, ge=0.0)


class StructuralOverrideConfig(BaseModel):
    """Configuration for bypassing the diversity floor when structural
    evidence is overwhelming (Path B).
    """

    enabled: bool = Field(default=False)
    v_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    c_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    margin_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    confidence_cap: float = Field(default=0.70, ge=0.0, le=1.0)


class EqbslSourceTrustConfig(BaseModel):
    """Trust weights for EQBSL evidence sources."""

    u: float = Field(default=1.0, ge=0.0)
    c: float = Field(default=1.0, ge=0.0)
    m: float = Field(default=1.0, ge=0.0)
    v: float = Field(default=1.0, ge=0.0)
    verification: float = Field(default=0.9, ge=0.0)
    generator_trust: float = Field(default=0.8, ge=0.0)


class EqbslVerificationConfig(BaseModel):
    """Configuration for the optional answer-conditioned verification source."""

    enabled: bool = Field(default=False)
    model_name: str = Field(default="deepseek-chat")
    api_base: str = Field(default="https://api.deepseek.com")
    api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1)
    enable_decision_semantics: bool = Field(default=False)
    contradiction_abstain_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    support_select_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    support_advantage_margin: float = Field(default=0.05, ge=0.0, le=1.0)


class EQBSLConfig(BaseModel):
    """Configuration for the additive EQBSL coalition-opinion layer."""

    enabled: bool = Field(default=False)
    k: float = Field(default=2.0, gt=0.0)
    default_base_rate: float = Field(default=0.25, ge=0.0, le=0.5)
    selection_expectation_threshold: float = Field(default=0.52, ge=0.0, le=1.0)
    resample_expectation_threshold: float = Field(default=0.38, ge=0.0, le=1.0)
    max_select_uncertainty: float = Field(default=0.55, ge=0.0, le=1.0)
    abstain_uncertainty_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    distinct_answer_gap_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    enable_generator_trust: bool = Field(default=True)
    source_trust: EqbslSourceTrustConfig = Field(default_factory=EqbslSourceTrustConfig)
    verification: EqbslVerificationConfig = Field(default_factory=EqbslVerificationConfig)


class TribunalConfig(BaseModel):
    adjudication_strategy: str = Field(
        default="standard",
        description="standard | greedy — greedy bypasses tribunal entirely",
    )
    fusion_mode: str = Field(
        default="weighted_sum",
        description="weighted_sum | eqbsl",
    )
    weights: TribunalWeights = Field(default_factory=TribunalWeights)
    selection_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    resample_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    max_resample_attempts: int = Field(default=2, ge=0)
    single_source_resampling_mode: bool = Field(default=False)
    resample_temperature_schedule: list[float] = Field(default_factory=lambda: [0.1, 0.4, 0.7])
    diversity_floor: float = Field(default=0.90, ge=0.0, le=1.0)
    ledger_warmup_tasks: int = Field(default=150, ge=0)
    # Discordant-resample guardrail parameters (FIX B)
    # These prevent a SELECT when the top candidate barely edges out rivals.
    guardrail_margin_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Minimum score margin between top-2 candidates to allow SELECT."
    )
    guardrail_min_coalition_mass: float = Field(
        default=0.40, ge=0.0, le=1.0,
        description="Minimum coalition_mass to allow SELECT (0 to disable)."
    )
    structural_override: StructuralOverrideConfig = Field(
        default_factory=StructuralOverrideConfig
    )


class LLMGeneratorConfig(BaseModel):
    model_name: str = Field(
        default="deepseek-chat"
    )
    max_new_tokens: int = Field(default=8192, ge=1)
    temperature: float = Field(default=0.1, ge=0.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    trust_remote_code: bool = Field(default=False)
    device: Optional[str] = Field(default=None)
    # BF16 is natively accelerated on H200/B100 Tensor Cores (halves memory vs FP32,
    # same numerical range).  Use "float16" for older Volta/Turing, "float32" to disable.
    torch_dtype: str = Field(default="bfloat16")
    # "auto" detects flash_attention_2 at runtime and falls back to "sdpa".
    # Set explicitly to "flash_attention_2", "sdpa", or "eager" to override.
    attn_implementation: str = Field(default="auto")
    # Use full json_schema constrained decoding (if supported by remote API) or fall back
    # to json_object (DeepSeek, older OpenAI cloud APIs that lack schema support).
    use_json_schema: bool = Field(default=False)
    api_base: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))


class GeneratorsConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "greedy",
            "diverse",
            "adversarial",
            "rule_first",
            "minimal_description",
        ]
    )
    seed: int = Field(default=42)
    llm: LLMGeneratorConfig = Field(default_factory=LLMGeneratorConfig)
    configs: dict[str, dict[str, Any]] = Field(default_factory=dict)


class InvariantsConfig(BaseModel):
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled_checks: list[str] = Field(
        default_factory=lambda: [
            "object_count_preserved",
            "colour_count_preserved",
            "symmetry_expected",
            "shape_transform_expected",
            "size_relation_preserved",
            "bounding_box_consistent",
            "grid_dimensions_consistent",
        ]
    )


class CriticConfig(BaseModel):
    failure_similarity_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    consistency_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    rule_coherence_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    morphology_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    use_llm_judge_for_math: bool = Field(default=False)


class UncertaintyConfig(BaseModel):
    entropy_bins: int = Field(default=10, ge=1)
    min_coalition_mass: float = Field(default=0.6, ge=0.0, le=1.0)


class FailureMemoryConfig(BaseModel):
    """Configuration for the metacognitive failure-memory layer."""

    enabled: bool = Field(
        default=False,
        description="Enable failure-memory lookup during adjudication.",
    )
    penalty_scale: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Maximum penalty from a single failure-memory match.",
    )


class StrangeLoopConfig(BaseModel):
    """Configuration for Strange Loop memory v1.

    When enabled, failure memory is queried *before* generation and
    structured constraints (bad-answer avoidance + structural warnings)
    are injected into generator prompts.  This makes failure memory a
    live participant during generation rather than only a post-hoc penalty.
    """

    enabled: bool = Field(
        default=False,
        description="Enable pre-generation failure constraint injection.",
    )
    mode: str = Field(
        default="full_memory",
        description="off | bad_answers_only | warnings_only | full_memory",
    )
    max_bad_answers: int = Field(
        default=5, ge=0,
        description="Maximum number of bad-answer signatures to inject.",
    )
    max_warnings: int = Field(
        default=3, ge=0,
        description="Maximum number of structural warnings to inject.",
    )
    min_similarity: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum match similarity to include a failure signature.",
    )
    same_task_boost: float = Field(
        default=1.5, ge=1.0,
        description="Similarity multiplier for exact task_id matches.",
    )


class LedgerConfig(BaseModel):
    path: str = Field(default="data/tribunal_ledger.db")
    always_record: bool = False


class BenchmarkConfig(BaseModel):
    checkpoint_every_n_tasks: int = Field(default=0, ge=0)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(default="rich")


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class TribunalSettings(BaseModel):
    """Full application settings."""

    tribunal: TribunalConfig = Field(default_factory=TribunalConfig)
    generators: GeneratorsConfig = Field(default_factory=GeneratorsConfig)
    invariants: InvariantsConfig = Field(default_factory=InvariantsConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    failure_memory: FailureMemoryConfig = Field(default_factory=FailureMemoryConfig)
    strange_loop: StrangeLoopConfig = Field(default_factory=StrangeLoopConfig)
    eqbsl: EQBSLConfig = Field(default_factory=EQBSLConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _apply_env_overrides(cls, data: Any) -> Any:
        """Apply TRIBUNAL_LEDGER_PATH env override if set."""
        if isinstance(data, dict):
            ledger_path = os.environ.get("TRIBUNAL_LEDGER_PATH")
            if ledger_path:
                ledger = data.setdefault("ledger", {})
                if isinstance(ledger, dict):
                    ledger["path"] = ledger_path
            log_level = os.environ.get("LOG_LEVEL")
            if log_level:
                logging_cfg = data.setdefault("logging", {})
                if isinstance(logging_cfg, dict):
                    logging_cfg["level"] = log_level
        return data


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: Optional[Path | str] = None) -> TribunalSettings:
    """Load settings from *path* (YAML), falling back to the bundled default.

    Environment variable ``TRIBUNAL_CONFIG_PATH`` can override the path.
    """
    if path is None:
        env_path = os.environ.get("TRIBUNAL_CONFIG_PATH")
        path = Path(env_path) if env_path else _DEFAULT_CONFIG_PATH

    path = Path(path)
    if path.exists():
        with open(path) as fh:
            content = os.path.expandvars(fh.read())
            raw: dict[str, Any] = yaml.safe_load(content) or {}
    else:
        raw = {}

    return TribunalSettings(**raw)
