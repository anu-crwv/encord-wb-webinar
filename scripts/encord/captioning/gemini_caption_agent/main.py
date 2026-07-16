# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click<8.2",
#     "encord==0.1.197",
#     "encord_agents==0.2.4",
#     "google-genai>=1.25.0",
#     "pydantic>=2",
#     "python-dotenv>=1.0",
#     "pyyaml>=6",
#     "typer>=0.12.0",
# ]
# ///
"""Local Encord task agent for Gemini-generated robotics captions."""

from __future__ import annotations

import atexit
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import time
from typing import Annotated, Any, Iterable
from uuid import UUID, uuid4
import warnings

from dotenv import load_dotenv
from encord.exceptions import ResourceExistsError
from encord.objects import Classification
from encord.objects.ontology_labels_impl import LabelRowV2
from encord.project import Project
from encord.storage import StorageItem
from encord.workflow import AgentStage
from encord.workflow.stages.agent import AgentTask
from encord_agents.core.utils import download_asset
from encord_agents.tasks import Runner
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError, field_validator, model_validator
import typer
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
CLASSIFICATION_TITLES = (
    "Language Instruction 1",
    "Language Instruction 2",
    "Language Instruction 3",
)
FENCE_RE = re.compile(r"^```(?:json)?|```$", re.IGNORECASE | re.MULTILINE)
ARM_PREFIX_RE = re.compile(
    r"^use\s+(?:the\s+)?(?:left|right|robot|both)\s+arms?\s+to\s+",
    re.IGNORECASE,
)
BAD_CAPTION_PATTERNS = (
    r"\bvideo\b",
    r"\bclip\b",
    r"\bfootage\b",
    r"\bcamera\b",
    r"\bframe\b",
    r"\btimestamp\b",
    r"\bsuccess(?:ful|fully)?\b",
    r"\bsucceed(?:s|ed)?\b",
    r"\bfail(?:s|ed|ure)?\b",
    r"\buncertain\b",
    r"\bunsure\b",
    r"\bmaybe\b",
    r"\bprobably\b",
    r"\bmight\b",
    r"\bappears? to\b",
    r"\bseems? to\b",
)

app = typer.Typer(add_completion=False, help=__doc__)


class AgentConfig(BaseModel):
    project_hash: str
    agent_stage_name: str
    success_pathway: str
    failure_pathway: str
    metadata_mismatch_pathway: str = "human_review"
    gemini_api_key_env_var: str = "GEMINI_API_KEY"
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    file_processing_timeout_seconds: int = 300
    keep_uploaded_files: bool = False
    caption_titles: tuple[str, str, str] = CLASSIFICATION_TITLES
    metadata_task_key: str = "task_name"
    video_layout: str = "camera_cam_high"
    local_video_cache_dir: str = ".cache/videos"
    use_gemini_video_proxy: bool = True
    gemini_video_proxy_width: int = 640
    gemini_video_proxy_fps: int = 4
    gemini_video_proxy_crf: int = 32
    gemini_video_proxy_preset: str = "veryfast"
    overwrite: bool = False
    task_batch_size: int = 1
    parallel_worker_count: int = 1
    parallel_worker_index: int = 0
    worker_lock_dir: str = ".cache/worker-locks"
    runner_init_retries: int = 3
    runner_init_retry_delay_seconds: float = 5.0
    max_tasks_per_stage: int | None = None
    refresh_every: int | None = None

    @field_validator("caption_titles")
    @classmethod
    def validate_caption_titles(cls, value: tuple[str, str, str]) -> tuple[str, str, str]:
        if tuple(value) != CLASSIFICATION_TITLES:
            raise ValueError(f"caption_titles must be exactly {CLASSIFICATION_TITLES}")
        return value

    @model_validator(mode="after")
    def validate_parallel_worker(self) -> AgentConfig:
        if self.parallel_worker_count < 1:
            raise ValueError("parallel_worker_count must be at least 1")
        if not 0 <= self.parallel_worker_index < self.parallel_worker_count:
            raise ValueError(
                "parallel_worker_index must satisfy "
                "0 <= parallel_worker_index < parallel_worker_count"
            )
        return self


class GeminiCaptionResponse(BaseModel):
    language_instruction_1: str
    language_instruction_2: str
    language_instruction_3_action: str
    metadata_mismatch: bool

    @field_validator(
        "language_instruction_1",
        "language_instruction_2",
        "language_instruction_3_action",
    )
    @classmethod
    def strip_text(cls, value: str) -> str:
        return " ".join(str(value or "").strip().split())

    @property
    def captions(self) -> tuple[str, str, str]:
        return (
            self.language_instruction_1,
            self.language_instruction_2,
            build_language_instruction_3(action=self.language_instruction_3_action),
        )


class SelectedVideo(BaseModel):
    layout_key: str
    storage_item: Any
    fallback: bool = False
    fallback_reason: str = ""

    model_config = {"arbitrary_types_allowed": True}


def clean_language_instruction_3_action(action: str) -> str:
    cleaned = " ".join(str(action or "").strip().split())
    cleaned = ARM_PREFIX_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"^to\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.rstrip(".")


def build_language_instruction_3(
    *,
    action: str,
) -> str:
    return f"use the robot arm to {clean_language_instruction_3_action(action)}"


def load_config(path: Path) -> AgentConfig:
    if not path.exists():
        raise typer.BadParameter(f"Config does not exist: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter("Config must contain a YAML object.")
    try:
        return AgentConfig.model_validate(data)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


def load_env_files() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    load_dotenv(Path.cwd() / ".env")


def env_path_status(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        return "unset"
    path = Path(value).expanduser()
    return f"set: {path} (exists={path.exists()})"


def secret_env_status(name: str) -> str:
    return "set" if os.environ.get(name) else "unset"


def echo_auth_context(config: AgentConfig, config_path: Path) -> None:
    typer.echo(f"[auth] config: {config_path}")
    typer.echo(f"[auth] project_hash: {config.project_hash}")
    typer.echo(f"[auth] script .env exists: {(SCRIPT_DIR / '.env').exists()}")
    typer.echo(f"[auth] cwd .env exists: {(Path.cwd() / '.env').exists()}")
    typer.echo(f"[auth] ENCORD_SSH_KEY_FILE: {env_path_status('ENCORD_SSH_KEY_FILE')}")
    typer.echo(f"[auth] {config.gemini_api_key_env_var}: {secret_env_status(config.gemini_api_key_env_var)}")


def runner_or_exit(config: AgentConfig) -> Runner:
    load_env_files()
    attempts = max(1, config.runner_init_retries)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return Runner(project_hash=config.project_hash)
        except Exception as exc:
            last_exc = exc
            exc_name = exc.__class__.__name__
            typer.echo(
                f"[auth] failed to initialize Encord task Runner "
                f"(attempt {attempt}/{attempts}): {exc_name}: {exc}",
                err=True,
            )
            if exc_name in {"AuthenticationError", "AuthorisationError", "AuthorizationError"} or (
                "ENCORD_SSH_KEY_FILE" in str(exc) or "ENCORD_SSH_KEY" in str(exc)
            ):
                typer.echo(
                    "[auth] This means ENCORD_SSH_KEY_FILE is missing, points to the wrong key, "
                    "or that key's user does not have editor/task-agent access to this project.",
                    err=True,
                )
                raise typer.Exit(code=1) from exc
            if exc_name != "UnknownException" or attempt == attempts:
                break
            typer.echo(
                f"[auth] Encord returned UnknownException; retrying in "
                f"{config.runner_init_retry_delay_seconds:g}s...",
                err=True,
            )
            time.sleep(config.runner_init_retry_delay_seconds)

    typer.echo(
        "[auth] Encord task Runner could not start. This was not classified as an auth failure. "
        "If the same key worked moments ago, it is likely a transient Encord/API issue; "
        "retry the command and keep the trace_id above if it persists.",
        err=True,
    )
    raise typer.Exit(code=1) from last_exc


def gemini_client(config: AgentConfig) -> genai.Client:
    load_env_files()
    api_key = os.environ.get(config.gemini_api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"Set {config.gemini_api_key_env_var} in the environment or in an ignored .env file."
        )
    return genai.Client(api_key=api_key)


def cache_root(config: AgentConfig) -> Path:
    path = Path(config.local_video_cache_dir).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def worker_lock_root(config: AgentConfig) -> Path:
    path = Path(config.worker_lock_dir).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def unique_tmp_path(cache_path: Path, suffix: str) -> Path:
    return cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{uuid4().hex}{suffix}")


def worker_owns_data_hash(data_hash: Any, config: AgentConfig) -> bool:
    if config.parallel_worker_count == 1:
        return True
    return int(UUID(str(data_hash))) % config.parallel_worker_count == config.parallel_worker_index


def process_is_live(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class WorkerLock:
    def __init__(self, path: Path, token: str) -> None:
        self.path = path
        self.token = token
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        with suppress(Exception):
            metadata = json.loads(self.path.read_text())
            if metadata.get("token") == self.token and metadata.get("pid") == os.getpid():
                self.path.unlink()


def acquire_worker_lock(config: AgentConfig) -> WorkerLock:
    root = worker_lock_root(config)
    root.mkdir(parents=True, exist_ok=True)
    key_source = (
        f"{config.project_hash}|{config.agent_stage_name}|"
        f"{config.parallel_worker_count}|{config.parallel_worker_index}"
    )
    key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:24]
    lock_path = root / f"gemini-caption-agent-{key}.lock"
    token = uuid4().hex

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                metadata = json.loads(lock_path.read_text())
            except Exception:
                metadata = {}
            pid = int(metadata.get("pid") or -1)
            if process_is_live(pid):
                typer.echo(
                    "[agent] another live Gemini caption worker already owns "
                    f"shard {config.parallel_worker_index}/{config.parallel_worker_count}: "
                    f"pid={pid}, host={metadata.get('host', 'unknown')}, lock={lock_path}",
                    err=True,
                )
                raise typer.Exit(code=1)
            with suppress(FileNotFoundError):
                lock_path.unlink()
            continue

        metadata = {
            "token": token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "project_hash": config.project_hash,
            "agent_stage_name": config.agent_stage_name,
            "parallel_worker_count": config.parallel_worker_count,
            "parallel_worker_index": config.parallel_worker_index,
            "created_at_epoch": time.time(),
        }
        with os.fdopen(fd, "w") as handle:
            json.dump(metadata, handle, sort_keys=True)
            handle.write("\n")
        lock = WorkerLock(lock_path, token)
        atexit.register(lock.release)
        typer.echo(
            f"[agent] acquired worker shard {config.parallel_worker_index}/"
            f"{config.parallel_worker_count} lock: {lock_path}"
        )
        return lock


def metadata_dict(item: Any) -> dict[str, Any]:
    metadata = getattr(item, "client_metadata", None) or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def metadata_sources_for(storage_item: StorageItem, label_row: LabelRowV2 | None = None) -> list[dict[str, Any]]:
    sources = [metadata_dict(storage_item)]
    if label_row is not None:
        sources.append({
            "data_title": getattr(label_row, "data_title", None),
            "source_uri": getattr(label_row, "data_title", None),
        })
    with suppress(Exception):
        sources.extend(metadata_dict(child) for child in storage_item.get_child_items())
    return [source for source in sources if source]


def find_task_name(metadata_sources: Iterable[dict[str, Any]], config: AgentConfig) -> str | None:
    for metadata in metadata_sources:
        value = metadata.get(config.metadata_task_key)
        if value not in (None, ""):
            return str(value)
    return None


def item_type_name(storage_item: StorageItem) -> str:
    item_type = getattr(storage_item, "item_type", "")
    return getattr(item_type, "name", str(item_type)).upper()


def child_for_layout(
    layout_key: str,
    layout_to_file_name: dict[str, str],
    storage_children: list[StorageItem],
) -> StorageItem | None:
    file_name = layout_to_file_name.get(layout_key)
    if file_name:
        for child in storage_children:
            if str(getattr(child, "name", "")) == file_name:
                return child
    return next(
        (
            child
            for child in storage_children
            if layout_key == str(getattr(child, "name", ""))
            or layout_key in str(getattr(child, "name", ""))
        ),
        None,
    )


def is_video_storage_item(storage_item: StorageItem) -> bool:
    if item_type_name(storage_item) == "VIDEO":
        return True
    name = str(getattr(storage_item, "name", "")).lower()
    if name.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return True
    metadata = metadata_dict(storage_item)
    return any("observation.images." in str(metadata.get(key, "")) for key in ("source_uri", "s3_uri", "objectUrl"))


def select_videos(
    label_row: LabelRowV2,
    storage_item: StorageItem,
    config: AgentConfig,
) -> list[SelectedVideo]:
    if item_type_name(storage_item) != "GROUP":
        return [
            SelectedVideo(
                layout_key=str(getattr(storage_item, "name", "video")),
                storage_item=storage_item,
            )
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        children_meta = list(getattr(label_row.metadata, "children", []) or [])

    layout_to_file_name = {
        str(getattr(child, "layout_key", "")): str(getattr(child, "name", ""))
        for child in children_meta
        if getattr(child, "layout_key", None)
    }
    storage_children = list(storage_item.get_child_items())

    child = child_for_layout(config.video_layout, layout_to_file_name, storage_children)
    if child is not None:
        return [SelectedVideo(layout_key=config.video_layout, storage_item=child)]

    video_children = [child for child in storage_children if is_video_storage_item(child)]
    if not video_children:
        available = sorted(layout_to_file_name) or [str(getattr(child, "name", "")) for child in storage_children]
        raise RuntimeError(f"No video child found. Wanted layout {config.video_layout!r}; available {available}")

    fallback_child = random.choice(video_children)
    fallback_name = str(getattr(fallback_child, "name", "random_video_child"))
    return [
        SelectedVideo(
            layout_key=fallback_name,
            storage_item=fallback_child,
            fallback=True,
            fallback_reason=f"configured layout {config.video_layout!r} was not found",
        )
    ]


def cached_video_path(selected: SelectedVideo, config: AgentConfig) -> Path:
    root = cache_root(config)
    root.mkdir(parents=True, exist_ok=True)
    name = str(getattr(selected.storage_item, "name", "") or selected.layout_key)
    suffix = Path(name).suffix or ".mp4"
    cache_path = root / f"{selected.storage_item.uuid}{suffix}"
    if cache_path.exists():
        return cache_path

    tmp_path = unique_tmp_path(cache_path, ".tmp")
    try:
        with download_asset(selected.storage_item) as asset_path:
            shutil.copy2(asset_path, tmp_path)
        if cache_path.exists():
            return cache_path
        tmp_path.replace(cache_path)
    finally:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
    return cache_path


def cached_gemini_proxy_path(selected: SelectedVideo, source_path: Path, config: AgentConfig) -> Path:
    if not config.use_gemini_video_proxy:
        return source_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        typer.echo("[agent] ffmpeg not found; uploading original video to Gemini", err=True)
        return source_path

    root = cache_root(config) / "_gemini_proxy"
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / (
        f"{selected.storage_item.uuid}"
        f".w{config.gemini_video_proxy_width}"
        f".fps{config.gemini_video_proxy_fps}"
        f".crf{config.gemini_video_proxy_crf}.mp4"
    )
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    tmp_path = unique_tmp_path(cache_path, ".tmp.mp4")
    filter_spec = f"fps={config.gemini_video_proxy_fps},scale={config.gemini_video_proxy_width}:-2"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        filter_spec,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        config.gemini_video_proxy_preset,
        "-crf",
        str(config.gemini_video_proxy_crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        stderr = " ".join(result.stderr.strip().split())[:500]
        typer.echo(f"[agent] Gemini proxy encode failed; uploading original video. ffmpeg: {stderr}", err=True)
        return source_path

    if cache_path.exists() and cache_path.stat().st_size > 0:
        with suppress(FileNotFoundError):
            tmp_path.unlink()
        return cache_path

    tmp_path.replace(cache_path)
    return cache_path


def gemini_video_path(selected: SelectedVideo, config: AgentConfig) -> Path:
    source_path = cached_video_path(selected, config)
    return cached_gemini_proxy_path(selected, source_path, config)


def wait_until_ready(
    client: genai.Client,
    file_name: str,
    timeout_seconds: int,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while True:
        file_obj = client.files.get(name=file_name)
        state = getattr(getattr(file_obj, "state", None), "name", None) or str(getattr(file_obj, "state", "UNKNOWN"))
        if state == "ACTIVE":
            return file_obj
        if state == "FAILED":
            raise RuntimeError(f"Gemini file processing failed for {file_name}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Gemini file {file_name}; last state={state}")
        time.sleep(2)


def build_prompt(
    *,
    task_name: str | None,
    selected_videos: list[SelectedVideo],
) -> str:
    view_lines = "\n".join(f"- {video.layout_key}" for video in selected_videos)
    return f"""
You are captioning a robot manipulation episode for imitation-learning training.

Return exactly one compact JSON object with exactly these keys:
{{
  "language_instruction_1": "detailed whole-episode instruction",
  "language_instruction_2": "short paraphrase of the same instruction",
  "language_instruction_3_action": "action phrase for the robot-arm instruction",
  "metadata_mismatch": false
}}

Metadata task name: {task_name or "unknown"}
Provided video view(s):
{view_lines}

Caption rules:
- Each instruction must be an imperative command for the whole episode.
- Return the JSON object once. Do not repeat it.
- Keep language_instruction_1 concise, one sentence, under 30 words.
- Keep language_instruction_2 under 14 words.
- Keep language_instruction_3_action under 24 words.
- Mention the main object(s), action, and final placement/state when visible.
- Do not mention camera, video, frames, timestamps, uncertainty, success, or failure.
- Do not invent details that are not visible.
- language_instruction_2 must be a shorter safe paraphrase, not a duplicate.
- language_instruction_3_action must be the action phrase that follows "use the robot arm to"; do not start it with "use the".

Metadata mismatch rule:
- Set metadata_mismatch=true only when the visible task is clearly a different task than the metadata task name.
- More specific wording, object color differences, or a partial view are NOT enough to mark a mismatch.
- If metadata says "Coil wire" but the video shows opening or closing an air fryer tray, that IS a mismatch.
- If unsure, set metadata_mismatch=false.
""".strip()


def extract_json_object(text: str) -> str:
    stripped = FENCE_RE.sub("", text.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in Gemini response: {text!r}")
    return stripped[start : end + 1]


def gemini_response_text(response: Any) -> str:
    with suppress(Exception):
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()

    texts: list[str] = []
    for candidate in list(getattr(response, "candidates", None) or []):
        content = getattr(candidate, "content", None)
        for part in list(getattr(content, "parts", None) or []):
            text = getattr(part, "text", None)
            if text:
                texts.append(str(text))

    unique_texts = list(dict.fromkeys(texts))
    return "\n".join(unique_texts).strip()


def parsed_gemini_response(response: Any) -> GeminiCaptionResponse | None:
    parsed = getattr(response, "parsed", None)
    if parsed is None:
        return None
    if isinstance(parsed, GeminiCaptionResponse):
        return parsed
    return GeminiCaptionResponse.model_validate(parsed)


def gemini_response_summary(response: Any) -> str:
    finish_reasons = []
    safety_ratings = []
    for candidate in list(getattr(response, "candidates", None) or []):
        finish_reasons.append(str(getattr(candidate, "finish_reason", "unknown")))
        ratings = getattr(candidate, "safety_ratings", None)
        if ratings:
            safety_ratings.append(str(ratings))
    prompt_feedback = getattr(response, "prompt_feedback", None)
    return (
        f"finish_reasons={finish_reasons or ['none']}; "
        f"prompt_feedback={prompt_feedback}; "
        f"safety_ratings={safety_ratings or ['none']}"
    )


def run_gemini_caption(
    *,
    client: genai.Client,
    config: AgentConfig,
    video_paths: list[tuple[SelectedVideo, Path]],
    task_name: str | None,
) -> GeminiCaptionResponse:
    uploaded_files = []
    try:
        for _, path in video_paths:
            uploaded_files.append(client.files.upload(file=str(path)))

        ready_files = [
            wait_until_ready(client, uploaded.name, config.file_processing_timeout_seconds)
            for uploaded in uploaded_files
        ]
        prompt = build_prompt(
            task_name=task_name,
            selected_videos=[selected for selected, _ in video_paths],
        )

        parts: list[Any] = [types.Part.from_text(text=prompt)]
        for (selected, _), ready_file in zip(video_paths, ready_files, strict=True):
            parts.append(types.Part.from_text(text=f"View: {selected.layout_key}"))
            parts.append(types.Part.from_uri(
                file_uri=ready_file.uri,
                mime_type=ready_file.mime_type or "video/mp4",
            ))

        response = client.models.generate_content(
            model=config.gemini_model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
                response_mime_type="application/json",
                response_schema=GeminiCaptionResponse,
            ),
        )
        parsed = parsed_gemini_response(response)
        if parsed is not None:
            return parsed

        text = gemini_response_text(response)
        if not text:
            raise ValueError(f"Gemini returned no text. {gemini_response_summary(response)}")
        try:
            return GeminiCaptionResponse.model_validate_json(extract_json_object(text))
        except ValueError as exc:
            preview = " ".join(text.split())[:1000]
            raise ValueError(
                f"No valid JSON object found in Gemini response. {gemini_response_summary(response)}; "
                f"text_preview={preview!r}"
            ) from exc
    finally:
        if not config.keep_uploaded_files:
            for uploaded in uploaded_files:
                with suppress(Exception):
                    client.files.delete(name=uploaded.name)


def validate_captions(result: GeminiCaptionResponse) -> None:
    action = clean_language_instruction_3_action(result.language_instruction_3_action)
    raw_values = (result.language_instruction_1, result.language_instruction_2, action)
    missing = [index + 1 for index, value in enumerate(raw_values) if not value]
    if missing:
        raise ValueError(f"Gemini returned empty language instruction value(s): {missing}")

    captions = result.captions
    normalized = [caption.lower().strip() for caption in captions]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Gemini returned duplicate language instructions.")

    if not normalized[2].startswith("use the robot arm to "):
        raise ValueError('Language Instruction 3 must start with "use the robot arm to ".')

    for caption in captions:
        lowered = caption.lower()
        for pattern in BAD_CAPTION_PATTERNS:
            if re.search(pattern, lowered):
                raise ValueError(f"Rejected caption {caption!r}; matched banned pattern {pattern!r}.")


def ensure_label_row_initialized(label_row: LabelRowV2) -> None:
    try:
        label_row.initialise_labels()
    except Exception as exc:
        message = str(exc).lower()
        if "already been initialized" in message or "already been initialised" in message:
            return
        raise


def label_row_for_task(project: Project, task: AgentTask) -> LabelRowV2:
    label_rows = project.list_label_rows_v2(data_hashes=[task.data_hash])
    if not label_rows:
        raise RuntimeError(f"No label row found for task data_hash={task.data_hash}")
    return label_rows[0]


def initialized_label_row_for_task(
    project: Project,
    task: AgentTask,
    *,
    max_attempts: int = 3,
) -> LabelRowV2:
    last_error: ResourceExistsError | None = None
    for attempt in range(1, max_attempts + 1):
        label_row = label_row_for_task(project, task)
        try:
            ensure_label_row_initialized(label_row)
            return label_row
        except ResourceExistsError as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            typer.echo(
                f"[agent] {task.data_hash}: label row create raced; refetching "
                f"(attempt {attempt + 1}/{max_attempts})",
                err=True,
            )
            time.sleep(0.5 * attempt)

    raise RuntimeError(
        f"Label row for task data_hash={task.data_hash} already exists, "
        "but could not be refetched initialized."
    ) from last_error


def existing_caption_answers(label_row: LabelRowV2, config: AgentConfig) -> dict[str, str]:
    answers: dict[str, str] = {}
    titles = set(config.caption_titles)
    for instance in label_row.get_classification_instances():
        ontology_item = getattr(instance, "ontology_item", None)
        title = str(getattr(ontology_item, "title", "") or "")
        if title not in titles:
            continue
        with suppress(Exception):
            answer = instance.get_answer()
            if answer not in (None, ""):
                answers[title] = str(answer)
    return answers


def add_caption_instances(
    label_row: LabelRowV2,
    captions: tuple[str, str, str],
    config: AgentConfig,
) -> int:
    existing = existing_caption_answers(label_row, config)
    added = 0
    for title, caption in zip(config.caption_titles, captions, strict=True):
        if title in existing and not config.overwrite:
            continue
        classification = label_row.ontology_structure.get_child_by_title(
            title=title,
            type_=Classification,
        )
        instance = classification.create_instance()
        instance.set_answer(answer=caption)
        label_row.add_classification_instance(instance, force=config.overwrite)
        added += 1
    return added


def validate_project_setup(runner: Runner, config: AgentConfig) -> None:
    project = getattr(runner, "project", None) or runner.client.get_project(config.project_hash)
    failures: list[str] = []

    try:
        project.workflow.get_stage(name=config.agent_stage_name, type_=AgentStage)
    except Exception as exc:
        stage_titles = [getattr(stage, "title", "<unknown>") for stage in project.workflow.stages]
        failures.append(
            f"missing agent stage {config.agent_stage_name!r}; available stages: {stage_titles}; error={exc!r}"
        )

    for title in config.caption_titles:
        try:
            project.ontology_structure.get_child_by_title(title=title, type_=Classification)
        except Exception as exc:
            failures.append(f"missing classification {title!r}: {exc!r}")

    try:
        gemini_client(config)
    except Exception as exc:
        failures.append(str(exc))

    if failures:
        for failure in failures:
            typer.echo(f"[check] FAIL: {failure}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[check] project: {project.title} ({config.project_hash})")
    typer.echo(f"[check] agent stage: {config.agent_stage_name}")
    typer.echo(f"[check] success pathway: {config.success_pathway}")
    typer.echo(f"[check] failure pathway: {config.failure_pathway}")
    typer.echo(f"[check] metadata mismatch pathway: {config.metadata_mismatch_pathway}")
    typer.echo(f"[check] caption classifications: {', '.join(config.caption_titles)}")
    typer.echo(
        f"[check] worker shard: {config.parallel_worker_index}/{config.parallel_worker_count}"
    )

    stage = project.workflow.get_stage(name=config.agent_stage_name, type_=AgentStage)
    sample_task = next(iter(stage.get_tasks()), None)
    if sample_task is None:
        typer.echo("[check] no pending task at the agent stage; skipping sample video-layout check")
        return

    label_row = initialized_label_row_for_task(project, sample_task)
    storage_item = label_row.get_storage_item()
    selected = select_videos(label_row, storage_item, config)
    task_name = find_task_name(metadata_sources_for(storage_item, label_row), config)
    if not task_name:
        raise typer.BadParameter(
            f"Sample task is missing metadata key {config.metadata_task_key!r} "
            "on the group or child storage items."
        )
    typer.echo(f"[check] sample task data_hash: {sample_task.data_hash}")
    typer.echo(f"[check] metadata task name: {task_name}")
    typer.echo(f"[check] selected video: {', '.join(item.layout_key for item in selected)}")
    for item in selected:
        if item.fallback:
            typer.echo(
                f"[check] WARNING: {item.fallback_reason}; this task would route to "
                f"{config.metadata_mismatch_pathway!r}"
            )


def build_runner(config: AgentConfig) -> Runner:
    runner = runner_or_exit(config)
    client = gemini_client(config)

    @runner.stage(config.agent_stage_name)
    def caption_with_gemini(
        task: AgentTask,
        project: Project,
    ) -> str | None:
        try:
            if not worker_owns_data_hash(task.data_hash, config):
                return None

            label_row = initialized_label_row_for_task(project, task)
            storage_item = label_row.get_storage_item()
            existing = existing_caption_answers(label_row, config)
            if len(existing) == len(config.caption_titles) and not config.overwrite:
                typer.echo(f"[agent] {task.data_hash}: all captions already present; skipping")
                return config.success_pathway

            selected = select_videos(label_row, storage_item, config)
            video_paths = [(video, gemini_video_path(video, config)) for video in selected]
            upload_mb = sum(path.stat().st_size for _, path in video_paths) / 1_000_000
            typer.echo(f"[agent] {task.data_hash}: Gemini upload size={upload_mb:.1f} MB")
            task_name = find_task_name(metadata_sources_for(storage_item, label_row), config)
            if not task_name:
                raise RuntimeError(
                    f"Missing metadata key {config.metadata_task_key!r} "
                    "on the group or child storage items."
                )
            result = run_gemini_caption(
                client=client,
                config=config,
                video_paths=video_paths,
                task_name=task_name,
            )
            validate_captions(result)
            added = add_caption_instances(label_row, result.captions, config)
            if added:
                label_row.save()

            layout_fallback = any(video.fallback for video in selected)
            route = (
                config.metadata_mismatch_pathway
                if result.metadata_mismatch or layout_fallback
                else config.success_pathway
            )
            typer.echo(
                f"[agent] {task.data_hash}: saved={added}, "
                f"metadata_mismatch={result.metadata_mismatch}, "
                f"layout_fallback={layout_fallback}, route={route}"
            )
            for video in selected:
                if video.fallback:
                    typer.echo(f"[agent] layout fallback: {video.fallback_reason}; selected {video.layout_key}")
            return route
        except errors.APIError as exc:
            typer.echo(f"[agent] {task.data_hash}: Gemini API failed: {exc}", err=True)
            return config.failure_pathway
        except Exception as exc:
            typer.echo(f"[agent] {task.data_hash}: failed: {exc}", err=True)
            return config.failure_pathway

    return runner


@app.command()
def check(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to YAML config."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Validate config, project access, ontology titles, and sample video layout."""

    config = load_config(config_path)
    load_env_files()
    echo_auth_context(config, config_path)
    validate_project_setup(runner_or_exit(config), config)


@app.command("debug-auth")
def debug_auth(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to YAML config."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Print auth context and test Encord task-runner project access."""

    config = load_config(config_path)
    load_env_files()
    echo_auth_context(config, config_path)
    runner = runner_or_exit(config)
    project = getattr(runner, "project", None) or runner.client.get_project(config.project_hash)
    typer.echo(f"[auth] Encord project access OK: {project.title} ({config.project_hash})")


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to YAML config."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the local Encord task agent."""

    config = load_config(config_path)
    load_env_files()
    worker_lock = acquire_worker_lock(config)
    try:
        runner = build_runner(config)
        kwargs: dict[str, Any] = {"task_batch_size": config.task_batch_size}
        if config.refresh_every is not None:
            kwargs["refresh_every"] = config.refresh_every
        if config.max_tasks_per_stage is not None:
            kwargs["max_tasks_per_stage"] = config.max_tasks_per_stage
        runner(**kwargs)
    finally:
        worker_lock.release()


if __name__ == "__main__":
    app()
