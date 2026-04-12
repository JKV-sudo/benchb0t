"""
framework/config.py
~~~~~~~~~~~~~~~~~~~
Shared framework config loading, validation, and path resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

KNOWN_TOOLS = {
    "bash",
    "read_file",
    "write_file",
    "http_request",
    "list_dir",
    "run_background",
    "patch_file",
}
KNOWN_EVAL_TYPES = {"script", "exact_match", "llm_judge"}
LEVEL_FILENAME_RE = re.compile(r"^l\d{1,3}-[a-z0-9-]+\.yaml$")


class FrameworkConfigError(ValueError):
    """Raised when the framework config cannot be loaded or validated."""


class LevelValidationError(ValueError):
    """Raised when a level YAML file cannot be loaded or validated."""


class HarnessValidationError(ValueError):
    """Raised when a harness YAML file cannot be loaded or validated."""


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys so config mistakes fail fast."""

    model_config = ConfigDict(extra="forbid")


class FrameworkSettings(_StrictModel):
    version: str = "0.1.0"
    name: str = "benchb0t"
    log_level: str = "INFO"
    runs_dir: str = "runs"
    max_parallel_levels: int = 1
    preview_linger_seconds: int = 60

    @field_validator("runs_dir")
    @classmethod
    def _validate_runs_dir(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("framework.runs_dir must not be empty")
        return value

    @field_validator("max_parallel_levels")
    @classmethod
    def _validate_parallelism(cls, value: int) -> int:
        if value < 1:
            raise ValueError("framework.max_parallel_levels must be >= 1")
        return value

    @field_validator("preview_linger_seconds")
    @classmethod
    def _validate_preview_linger(cls, value: int) -> int:
        if value < 0:
            raise ValueError("framework.preview_linger_seconds must be >= 0")
        return value


class ScoringWeights(_StrictModel):
    completion: float = 0.40
    efficiency: float = 0.25
    self_correction: float = 0.20
    path_quality: float = 0.15

    @field_validator("completion", "efficiency", "self_correction", "path_quality")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("scoring weights must be between 0.0 and 1.0")
        return value


class ScoringPenalties(_StrictModel):
    extra_tool_call: float = 0.5
    backtrack: float = 1.0
    timeout: float = 5.0

    @field_validator("extra_tool_call", "backtrack", "timeout")
    @classmethod
    def _validate_penalty(cls, value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("scoring penalties must be non-negative")
        return value


class ScoringSettings(_StrictModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    penalties: ScoringPenalties = Field(default_factory=ScoringPenalties)


class ContainerSettings(_StrictModel):
    default_cpu_limit: str = "2"
    default_memory_limit: str = "4g"
    pull_policy: Literal["always", "if_not_present", "never"] = "if_not_present"
    network_mode: str = "bridge"


class AgentSettings(_StrictModel):
    default_max_turns: int = 20
    default_timeout_s: int = 120
    default_temperature: float = 0.2
    default_max_tokens: int = 4096

    @field_validator("default_max_turns", "default_timeout_s", "default_max_tokens")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("agent defaults must be >= 1")
        return value

    @field_validator("default_temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("agent.default_temperature must be >= 0.0")
        return value

    def apply_task_defaults(self, task_cfg: dict[str, Any]) -> dict[str, Any]:
        """Fill missing task-level runtime limits from framework defaults."""
        resolved = dict(task_cfg or {})
        resolved.setdefault("max_turns", self.default_max_turns)
        resolved.setdefault("timeout_s", self.default_timeout_s)
        return resolved

    def api_defaults(self, task_cfg: dict[str, Any]) -> dict[str, Any]:
        """Map framework agent defaults to the keys AgentAPI expects."""
        resolved_task = self.apply_task_defaults(task_cfg)
        return {
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
            "timeout_s": resolved_task["timeout_s"],
        }


class RecorderSettings(_StrictModel):
    format: str = "agentlog"
    compress: bool = False

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        if value != "agentlog":
            raise ValueError("recorder.format currently only supports 'agentlog'")
        return value


class FrameworkConfig(_StrictModel):
    framework: FrameworkSettings = Field(default_factory=FrameworkSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    container: ContainerSettings = Field(default_factory=ContainerSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    recorder: RecorderSettings = Field(default_factory=RecorderSettings)


class LevelModeSettings(_StrictModel):
    system_prompt: str


class LevelMetadata(_StrictModel):
    id: str
    name: str
    difficulty: int
    category: str
    tags: list[str] = Field(default_factory=list)
    deprecated: bool = False
    replaced_by: str | None = None

    @field_validator("id", "name", "category")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("difficulty")
    @classmethod
    def _validate_difficulty(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("level.difficulty must be between 1 and 5")
        return value

    @field_validator("replaced_by")
    @classmethod
    def _normalize_replaced_by(cls, value: str | None) -> str | None:
        return value.strip() if value else None


class LevelPackages(_StrictModel):
    apt: list[str] = Field(default_factory=list)
    pip: list[str] = Field(default_factory=list)
    npm: list[str] = Field(default_factory=list)
    gem: list[str] = Field(default_factory=list)


class LevelContainerConfig(_StrictModel):
    image: str
    working_dir: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    volumes: list[str] = Field(default_factory=list)
    packages: LevelPackages = Field(default_factory=LevelPackages)
    setup_script: str = ""

    @field_validator("image", "working_dir")
    @classmethod
    def _validate_container_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class PreviewConfig(_StrictModel):
    port: int
    path: str = "/"
    linger_seconds: int | None = None

    @field_validator("port")
    @classmethod
    def _validate_preview_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("preview.port must be between 1 and 65535")
        return value

    @field_validator("path")
    @classmethod
    def _validate_preview_path(cls, value: str) -> str:
        value = value.strip() or "/"
        if not value.startswith("/"):
            raise ValueError("preview.path must start with '/'")
        return value

    @field_validator("linger_seconds")
    @classmethod
    def _validate_preview_linger(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("preview.linger_seconds must be >= 0")
        return value


class TaskConfig(_StrictModel):
    instruction: str
    context: str | None = None
    max_turns: int | None = None
    timeout_s: int | None = None

    @field_validator("instruction")
    @classmethod
    def _validate_instruction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task.instruction must not be empty")
        return value

    @field_validator("max_turns", "timeout_s")
    @classmethod
    def _validate_task_limits(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("task limits must be >= 1")
        return value


class EvaluationCriterion(_StrictModel):
    id: str
    description: str
    type: Literal["script", "exact_match", "llm_judge"] = "script"
    check: str = ""
    weight: float = 1.0

    @field_validator("id", "description")
    @classmethod
    def _validate_criterion_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("weight")
    @classmethod
    def _validate_weight_positive(cls, value: float) -> float:
        value = float(value)
        if value <= 0.0:
            raise ValueError("criterion.weight must be > 0")
        return value


class EvaluationConfig(_StrictModel):
    type: Literal["script", "exact_match", "llm_judge"] = "script"
    efficiency_target: int = 0
    criteria: list[EvaluationCriterion] = Field(default_factory=list)

    @field_validator("efficiency_target")
    @classmethod
    def _validate_efficiency_target(cls, value: int) -> int:
        if value < 0:
            raise ValueError("evaluation.efficiency_target must be >= 0")
        return value


class ForcedRetryConfig(_StrictModel):
    enabled: bool = False
    max_retries: int = 2
    penalty_per_retry: float = 10.0
    completion_threshold: float = 0.5

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("forced_retry.max_retries must be >= 0")
        return value

    @field_validator("penalty_per_retry")
    @classmethod
    def _validate_retry_penalty(cls, value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("forced_retry.penalty_per_retry must be >= 0")
        return value

    @field_validator("completion_threshold")
    @classmethod
    def _validate_retry_threshold(cls, value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("forced_retry.completion_threshold must be between 0.0 and 1.0")
        return value


class LevelFileConfig(_StrictModel):
    level: LevelMetadata
    container: LevelContainerConfig
    task: TaskConfig
    tools: list[str]
    evaluation: EvaluationConfig
    modes: dict[str, LevelModeSettings] = Field(default_factory=dict)
    preview: PreviewConfig | None = None
    forced_retry: ForcedRetryConfig | None = None

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("tools must contain at least one tool")
        unknown = sorted(set(value) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(unknown)}")
        if len(set(value)) != len(value):
            raise ValueError("tools must not contain duplicates")
        return value

    @property
    def is_deprecated(self) -> bool:
        return self.level.deprecated


class HarnessEndpointConfig(_StrictModel):
    base_url: str
    api_key_env: str = "BENCHBOT_API_KEY"

    @field_validator("base_url", "api_key_env")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class HarnessModelDefaults(_StrictModel):
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("harness.model_defaults.model must not be empty")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temp(cls, value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("harness.model_defaults.temperature must be >= 0.0")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int) -> int:
        if value < 1:
            raise ValueError("harness.model_defaults.max_tokens must be >= 1")
        return value


class HarnessContainerConfig(_StrictModel):
    cpu_limit: str = "2"
    memory_limit: str = "4g"
    max_parallel: int = 1

    @field_validator("max_parallel")
    @classmethod
    def _validate_max_parallel(cls, value: int) -> int:
        if value < 1:
            raise ValueError("harness.container.max_parallel must be >= 1")
        return value


class HarnessSettings(_StrictModel):
    name: str
    type: str
    endpoint: HarnessEndpointConfig
    model_defaults: HarnessModelDefaults
    container: HarnessContainerConfig

    @field_validator("name", "type")
    @classmethod
    def _validate_name_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class HarnessFileConfig(_StrictModel):
    harness: HarnessSettings


@dataclass(frozen=True)
class LoadedFrameworkConfig:
    """Validated framework config plus project-relative path helpers."""

    path: Path
    project_dir: Path
    config: FrameworkConfig

    @property
    def runs_dir(self) -> Path:
        return self.project_dir / self.config.framework.runs_dir

    @property
    def db_path(self) -> Path:
        return self.project_dir / "benchb0t.db"

    @property
    def levels_dir(self) -> Path:
        return self.project_dir / "levels"

    @property
    def harnesses_dir(self) -> Path:
        return self.project_dir / "harnesses"

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.project_dir / candidate


def load_framework_config(path: str | Path) -> LoadedFrameworkConfig:
    """
    Load ``config.yaml`` and return a validated config object plus resolved paths.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise FrameworkConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise FrameworkConfigError(f"Framework config must be a YAML mapping: {config_path}")

    try:
        config = FrameworkConfig.model_validate(raw)
    except ValidationError as exc:
        raise FrameworkConfigError(f"Invalid framework config in {config_path}:\n{exc}") from exc

    return LoadedFrameworkConfig(
        path=config_path,
        project_dir=config_path.parent,
        config=config,
    )


def _load_yaml_mapping(path: str | Path, *, error_type: type[ValueError]) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path).expanduser()
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    file_path = file_path.resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        raw = yaml.safe_load(file_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise error_type(f"Invalid YAML in {file_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise error_type(f"YAML root must be a mapping: {file_path}")

    return file_path, raw


def load_level_config(path: str | Path) -> LevelFileConfig:
    """Load and validate one level YAML file."""
    file_path, raw = _load_yaml_mapping(path, error_type=LevelValidationError)
    try:
        model = LevelFileConfig.model_validate(raw)
    except ValidationError as exc:
        raise LevelValidationError(f"Invalid level config in {file_path}:\n{exc}") from exc
    return model


def load_harness_config(path: str | Path) -> HarnessFileConfig:
    """Load and validate one harness YAML file."""
    file_path, raw = _load_yaml_mapping(path, error_type=HarnessValidationError)
    try:
        model = HarnessFileConfig.model_validate(raw)
    except ValidationError as exc:
        raise HarnessValidationError(f"Invalid harness config in {file_path}:\n{exc}") from exc
    return model
