"""Callable local HGDN experiment runners used by CLI entrypoints and pipelines."""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from _repo_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

from hgdn_local_runner import (  # noqa: E402
    RECURRENCE_MODES,
    REPO_ROOT,
    CommandSpec,
    bool_flag_value,
    check_cuda_jobs,
    create_7z_archive,
    csv_ints,
    csv_items,
    diagnostic_env,
    filtered_config_env,
    log_completion_state,
    resolve_grad_accum_steps,
    resolve_val_batch_size,
    run_command,
    write_command_log,
)
from screen_hgdn_arch_sizes import run_size_screen  # noqa: E402


DEFAULT_SEARCH_CONFIGS: tuple[str, ...] = (
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml",
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m1p75.toml",
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_v1p5_m1p75.toml",
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml",
    "configs/hgdn/naive_contract_l8_d512_boundary2_dk48_m2.toml",
    "configs/hgdn/naive_contract_l8_d512_mid3_dk48_m1p75.toml",
    "configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml",
    "configs/hgdn/naive_contract_l9_d512_mid2_dk48_m1p75.toml",
    "configs/hgdn/naive_contract_l9_d512_mid2_dk48_v1p5_m1p75.toml",
    "configs/hgdn/naive_contract_l9_d512_mid2_dk48_m2.toml",
    "configs/hgdn/naive_contract_l9_d512_mid3_dk48_m1p75.toml",
    "configs/hgdn/naive_contract_l9_d512_mid3_dk48_v1p5_m1p75.toml",
    "configs/hgdn/naive_contract_l9_d512_tail2_dk48_m1p75.toml",
    "configs/hgdn/naive_contract_l8_d512_r0_m1p25.toml",
    "configs/hgdn/naive_contract_l8_d512_r0_m1p5.toml",
    "configs/hgdn/naive_contract_l8_d512_r0_m1p75.toml",
    "configs/hgdn/naive_contract_l8_d512_r0_m2.toml",
    "configs/hgdn/naive_contract_l9_d512_r0_m1p75.toml",
    "configs/hgdn/naive_contract_l9_d512_r0_m2.toml",
)

DEFAULT_SEARCH_LABELS: tuple[str, ...] = (
    "HGDN 8Lx512d mid2 dk48 mlp2.0",
    "HGDN 8Lx512d mid2 dk48 mlp1.75",
    "HGDN 8Lx512d mid2 dk48 v1.5 mlp1.75",
    "HGDN 8Lx512d mid2 dk48 v2.0 mlp1.5",
    "HGDN 8Lx512d boundary2 dk48 mlp2.0",
    "HGDN 8Lx512d mid3 dk48 mlp1.75",
    "HGDN 8Lx512d OLMo-ish 6G2A dk48 v2.0 mlp1.25",
    "HGDN 9Lx512d mid2 dk48 mlp1.75",
    "HGDN 9Lx512d mid2 dk48 v1.5 mlp1.75",
    "HGDN 9Lx512d mid2 dk48 mlp2.0",
    "HGDN 9Lx512d mid3 dk48 mlp1.75",
    "HGDN 9Lx512d mid3 dk48 v1.5 mlp1.75",
    "HGDN 9Lx512d tail2 dk48 mlp1.75",
    "Attention-only baseline 8Lx512d mlp1.25",
    "Attention-only baseline 8Lx512d mlp1.5",
    "Attention-only baseline 8Lx512d mlp1.75",
    "Attention-only baseline 8Lx512d mlp2.0",
    "Attention-only baseline 9Lx512d mlp1.75",
    "Attention-only baseline 9Lx512d mlp2.0",
)


@dataclass(frozen=True)
class LocalTrainContract:
    """Resolved trainer launch contract shared by local HGDN runners."""

    python_bin: str
    use_wandb: bool
    wandb_mode: str
    wandb_project: str
    wandb_watch: str
    wandb_watch_log_freq: int
    ngpu: int
    grad_accum_steps: int
    iterations: int
    train_batch_tokens: int
    train_seq_len: int
    val_loss_every: int
    train_log_every: int
    min_val_seqs: int
    val_max_seqs: int
    val_batch_size: int
    max_wallclock_seconds: float
    compile: bool
    compile_strategy: str
    distributed_mode: str
    weight_decay: float
    perf_skip_final_eval: bool
    torchinductor_max_autotune: int
    torchinductor_max_autotune_gemm: int
    torch_logs: str
    torch_trace: str
    data_path: Path
    tokenizer_path: Path
    vocab_size: int
    allow_existing_logs: bool

    @classmethod
    def resolve(
        cls,
        *,
        python_bin: str,
        use_wandb: bool,
        wandb_mode: str,
        wandb_project: str,
        wandb_watch: str,
        wandb_watch_log_freq: int,
        ngpu: int,
        grad_accum_steps: int | None,
        iterations: int,
        train_batch_tokens: int,
        train_seq_len: int,
        val_loss_every: int,
        train_log_every: int,
        min_val_seqs: int,
        val_max_seqs: int,
        val_batch_size: int | None,
        val_batch_seqs: int | None,
        max_wallclock_seconds: float,
        compile: bool,
        compile_strategy: str,
        distributed_mode: str,
        weight_decay: float,
        perf_skip_final_eval: bool,
        torchinductor_max_autotune: int,
        torchinductor_max_autotune_gemm: int,
        torch_logs: str,
        torch_trace: str,
        data_path: Path,
        tokenizer_path: Path,
        vocab_size: int,
        allow_existing_logs: bool,
    ) -> "LocalTrainContract":
        """Resolve dependent defaults and return a concrete contract."""
        resolved_grad_accum = resolve_grad_accum_steps(ngpu, grad_accum_steps)
        resolved_val_batch_size = resolve_val_batch_size(
            ngpu,
            resolved_grad_accum,
            train_seq_len,
            requested_tokens=val_batch_size,
            requested_seqs=val_batch_seqs,
        )
        return cls(
            python_bin=python_bin,
            use_wandb=use_wandb,
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            wandb_watch=wandb_watch,
            wandb_watch_log_freq=wandb_watch_log_freq,
            ngpu=ngpu,
            grad_accum_steps=resolved_grad_accum,
            iterations=iterations,
            train_batch_tokens=train_batch_tokens,
            train_seq_len=train_seq_len,
            val_loss_every=val_loss_every,
            train_log_every=train_log_every,
            min_val_seqs=min_val_seqs,
            val_max_seqs=val_max_seqs,
            val_batch_size=resolved_val_batch_size,
            max_wallclock_seconds=max_wallclock_seconds,
            compile=compile,
            compile_strategy=compile_strategy,
            distributed_mode=distributed_mode,
            weight_decay=weight_decay,
            perf_skip_final_eval=perf_skip_final_eval,
            torchinductor_max_autotune=torchinductor_max_autotune,
            torchinductor_max_autotune_gemm=torchinductor_max_autotune_gemm,
            torch_logs=torch_logs,
            torch_trace=torch_trace,
            data_path=data_path,
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
            allow_existing_logs=allow_existing_logs,
        )

    def base_env(self) -> dict[str, str]:
        """Build the trainer environment overlay."""
        env = {
            "NGPU": str(self.ngpu),
            "USE_WANDB": bool_flag_value(self.use_wandb),
            "WANDB_MODE": self.wandb_mode,
            "WANDB_PROJECT": self.wandb_project,
            "WANDB_WATCH": self.wandb_watch,
            "WANDB_WATCH_LOG_FREQ": str(self.wandb_watch_log_freq),
            "TORCHINDUCTOR_MAX_AUTOTUNE": str(self.torchinductor_max_autotune),
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": str(
                self.torchinductor_max_autotune_gemm
            ),
            "COMPILE": bool_flag_value(self.compile),
            "COMPILE_STRATEGY": self.compile_strategy,
            "DISTRIBUTED_MODE": self.distributed_mode,
            "DATA_PATH": str(self.data_path),
            "TOKENIZER_PATH": str(self.tokenizer_path),
            "VOCAB_SIZE": str(self.vocab_size),
            "GRAD_ACCUM_STEPS": str(self.grad_accum_steps),
            "ITERATIONS": str(self.iterations),
            "MAX_WALLCLOCK_SECONDS": f"{self.max_wallclock_seconds:g}",
            "TRAIN_BATCH_TOKENS": str(self.train_batch_tokens),
            "TRAIN_SEQ_LEN": str(self.train_seq_len),
            "VAL_LOSS_EVERY": str(self.val_loss_every),
            "TRAIN_LOG_EVERY": str(self.train_log_every),
            "MIN_VAL_SEQS": str(self.min_val_seqs),
            "VAL_MAX_SEQS": str(self.val_max_seqs),
            "VAL_BATCH_SIZE": str(self.val_batch_size),
            "WEIGHT_DECAY": f"{self.weight_decay:g}",
            "PERF_SKIP_FINAL_EVAL": bool_flag_value(self.perf_skip_final_eval),
        }
        env.update(diagnostic_env(self.torch_logs, self.torch_trace))
        return env

    def manifest_contract(self) -> dict[str, Any]:
        """Return the serializable contract fields used by bundle manifests."""
        return {
            "ngpu": self.ngpu,
            "grad_accum_steps": self.grad_accum_steps,
            "iterations": self.iterations,
            "train_batch_tokens": self.train_batch_tokens,
            "train_seq_len": self.train_seq_len,
            "val_loss_every": self.val_loss_every,
            "train_log_every": self.train_log_every,
            "min_val_seqs": self.min_val_seqs,
            "val_max_seqs": self.val_max_seqs,
            "val_batch_size": self.val_batch_size,
            "max_wallclock_seconds": self.max_wallclock_seconds,
            "weight_decay": self.weight_decay,
            "perf_skip_final_eval": self.perf_skip_final_eval,
            "compile": self.compile,
            "compile_strategy": self.compile_strategy,
            "distributed_mode": self.distributed_mode,
            "torch_logs": self.torch_logs or None,
            "torch_trace": self.torch_trace or None,
            "torchinductor_max_autotune": self.torchinductor_max_autotune,
            "torchinductor_max_autotune_gemm": self.torchinductor_max_autotune_gemm,
            "data_path": str(self.data_path),
            "tokenizer_path": str(self.tokenizer_path),
            "vocab_size": self.vocab_size,
        }


@dataclass(frozen=True)
class CandidateSpec:
    """One config/training leg in a local experiment."""

    config: str
    label: str
    run_prefix: str
    recurrence_mode: str


@dataclass(frozen=True)
class NaiveSearchPlan:
    """Plan for the local naive-contract sparse candidate search."""

    run_prefix_base: str
    bundle_stage_dir: Path
    archive_output: Path
    command_log: Path
    size_screen_output: Path
    size_screen_config: Path
    gdn_fla_recurrence_mode: str
    contract: LocalTrainContract
    candidates: tuple[CandidateSpec, ...]


@dataclass(frozen=True)
class RecurrenceMatrixPlan:
    """Plan for the local recurrence implementation matrix."""

    run_prefix_base: str
    bundle_stage_dir: Path
    archive_output: Path
    command_log: Path
    contract: LocalTrainContract
    candidates: tuple[CandidateSpec, ...]
    check_cuda_idle: bool
    allow_active_cuda_jobs: bool


@dataclass(frozen=True)
class ExperimentResult:
    """Result metadata for one local experiment run."""

    status: int
    bundle_stage_dir: Path
    archive_output: Path
    run_ids: tuple[str, ...]


def _suffix(index: int) -> str:
    """Return the stable alphabetical suffix used for default run prefixes."""
    return chr(ord("a") + index)


def _run_id(run_prefix: str, config: str, contract: LocalTrainContract) -> str:
    """Build the trainer RUN_ID for one candidate."""
    config_stem = Path(config).stem
    return (
        f"{run_prefix}_{config_stem}_seq{contract.train_seq_len}"
        f"_it{contract.iterations}"
    )


def _check_fresh_logs(run_ids: Sequence[str], *, allow_existing_logs: bool) -> None:
    """Refuse to append to existing logs unless explicitly allowed."""
    if allow_existing_logs:
        return
    conflicts = [
        run_id for run_id in run_ids if (REPO_ROOT / "logs" / f"{run_id}.txt").exists()
    ]
    if not conflicts:
        return
    rendered = "\n".join(f"logs/{run_id}.txt" for run_id in conflicts)
    raise SystemExit(
        "Refusing to append to existing run logs:\n"
        f"{rendered}\n"
        "Use a fresh --run-prefix-base or --allow-existing-logs 1."
    )


def _train_spec(
    contract: LocalTrainContract,
    candidate: CandidateSpec,
    run_id: str,
) -> CommandSpec:
    """Build one torchrun command specification."""
    if candidate.recurrence_mode not in RECURRENCE_MODES:
        raise SystemExit(f"Unsupported recurrence mode: {candidate.recurrence_mode}")
    env = contract.base_env()
    env["RUN_ID"] = run_id
    env.update(filtered_config_env(Path(candidate.config), candidate.recurrence_mode))
    return CommandSpec(
        label=candidate.label,
        run_id=run_id,
        env=env,
        command=[
            "torchrun",
            "--standalone",
            f"--nproc_per_node={contract.ngpu}",
            "train_gpt_hybrid.py",
        ],
    )


def _run_specs(specs: Sequence[CommandSpec]) -> int:
    """Run specs sequentially and return the first nonzero exit status."""
    for spec in specs:
        status = run_command(spec)
        if status != 0:
            return status
    return 0


def _copy_existing_files(paths: Sequence[Path], output_dir: Path) -> None:
    """Copy existing files into a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def _copy_tree_contents(src: Path, dst: Path) -> None:
    """Copy directory contents when the source exists."""
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _bundle_common(
    *,
    bundle_stage_dir: Path,
    archive_output: Path,
    command_log: Path,
    configs: Sequence[str],
    run_ids: Sequence[str],
    manifest: dict[str, Any],
    size_screen_output: Path | None = None,
) -> None:
    """Stage logs/configs/manifest and write the archive."""
    shutil.rmtree(bundle_stage_dir, ignore_errors=True)
    (bundle_stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    (bundle_stage_dir / "configs").mkdir(parents=True, exist_ok=True)
    _copy_existing_files(
        [Path(config) for config in configs], bundle_stage_dir / "configs"
    )
    if command_log.is_file():
        shutil.copy2(command_log, bundle_stage_dir / "commands.sh")
    if size_screen_output is not None:
        _copy_tree_contents(size_screen_output, bundle_stage_dir / "size_screen")

    missing_run_ids: list[str] = []
    completed_log_count = 0
    for run_id in run_ids:
        log_path = REPO_ROOT / "logs" / f"{run_id}.txt"
        if log_path.is_file():
            shutil.copy2(log_path, bundle_stage_dir / "logs" / log_path.name)
            if log_completion_state(log_path) == "complete":
                completed_log_count += 1
        else:
            missing_run_ids.append(run_id)
    manifest.update(
        {
            "archive_output": str(archive_output),
            "completed_log_count": completed_log_count,
            "matched_logs": not missing_run_ids and bool(run_ids),
            "missing_run_ids": missing_run_ids,
        }
    )
    _write_json(bundle_stage_dir / "bundle_manifest.json", manifest)
    create_7z_archive(archive_output, bundle_stage_dir)


def resolve_search_candidates(
    *,
    run_prefix_base: str,
    candidate_configs: str = "",
    candidate_indexes: str = "",
    run_prefixes: str = "",
    allow_custom_candidate_configs: bool = False,
) -> tuple[CandidateSpec, ...]:
    """Resolve default/index/config-filtered search candidates."""
    if candidate_configs and candidate_indexes:
        raise SystemExit("Set only one of --candidate-configs or --candidate-indexes.")
    default_prefixes = tuple(
        f"{run_prefix_base}_{_suffix(index)}"
        for index, _config in enumerate(DEFAULT_SEARCH_CONFIGS)
    )
    resolved_prefixes = (
        tuple(csv_items(run_prefixes)) if run_prefixes else default_prefixes
    )
    if len(resolved_prefixes) != len(DEFAULT_SEARCH_CONFIGS):
        raise SystemExit(
            f"--run-prefixes count ({len(resolved_prefixes)}) must match default "
            f"config count ({len(DEFAULT_SEARCH_CONFIGS)})."
        )
    defaults = [
        CandidateSpec(
            config=config,
            label=label,
            run_prefix=resolved_prefixes[index],
            recurrence_mode="direct",
        )
        for index, (config, label) in enumerate(
            zip(DEFAULT_SEARCH_CONFIGS, DEFAULT_SEARCH_LABELS, strict=True)
        )
    ]
    if candidate_indexes:
        selected: list[CandidateSpec] = []
        for index in csv_ints(candidate_indexes):
            if index < 0 or index >= len(defaults):
                raise SystemExit(f"--candidate-indexes entry out of range: {index}")
            selected.append(defaults[index])
        return tuple(selected)
    if candidate_configs:
        selected = []
        custom_index = 0
        for config in csv_items(candidate_configs):
            match = next((item for item in defaults if item.config == config), None)
            if match is not None:
                selected.append(match)
                continue
            if not allow_custom_candidate_configs:
                raise SystemExit(
                    "--candidate-configs entry is not in the helper candidate list: "
                    f"{config}"
                )
            if not Path(config).is_file():
                raise SystemExit(f"Custom candidate config does not exist: {config}")
            selected.append(
                CandidateSpec(
                    config=config,
                    label=f"Custom {Path(config).stem}",
                    run_prefix=f"{run_prefix_base}_custom{custom_index}",
                    recurrence_mode="direct",
                )
            )
            custom_index += 1
        return tuple(selected)
    return tuple(defaults)


def build_naive_search_plan(
    *,
    run_prefix_base: str,
    contract: LocalTrainContract,
    gdn_fla_recurrence_mode: str,
    bundle_stage_dir: Path | None = None,
    archive_output: Path | None = None,
    command_log: Path | None = None,
    size_screen_output: Path | None = None,
    size_screen_config: Path = Path("configs/hgdn/naive_contract_search.toml"),
    candidate_configs: str = "",
    candidate_indexes: str = "",
    run_prefixes: str = "",
    allow_custom_candidate_configs: bool = False,
) -> NaiveSearchPlan:
    """Build a concrete local naive-contract search plan."""
    candidates = tuple(
        CandidateSpec(
            config=item.config,
            label=item.label,
            run_prefix=item.run_prefix,
            recurrence_mode=gdn_fla_recurrence_mode,
        )
        for item in resolve_search_candidates(
            run_prefix_base=run_prefix_base,
            candidate_configs=candidate_configs,
            candidate_indexes=candidate_indexes,
            run_prefixes=run_prefixes,
            allow_custom_candidate_configs=allow_custom_candidate_configs,
        )
    )
    return NaiveSearchPlan(
        run_prefix_base=run_prefix_base,
        bundle_stage_dir=bundle_stage_dir
        or Path(f"local-scratch/{run_prefix_base}_bundle"),
        archive_output=archive_output
        or Path(f"local-scratch/{run_prefix_base}_bundle.7z"),
        command_log=command_log or Path(f"local-scratch/{run_prefix_base}_commands.sh"),
        size_screen_output=size_screen_output
        or Path(f"local-scratch/{run_prefix_base}_size_screen"),
        size_screen_config=size_screen_config,
        gdn_fla_recurrence_mode=gdn_fla_recurrence_mode,
        contract=contract,
        candidates=candidates,
    )


def build_recurrence_matrix_plan(
    *,
    run_prefix_base: str,
    contract: LocalTrainContract,
    hgdn_config: str = "configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml",
    attn_config: str = "configs/hgdn/naive_contract_l8_d512_r0_m1p5.toml",
    bundle_stage_dir: Path | None = None,
    archive_output: Path | None = None,
    command_log: Path | None = None,
    check_cuda_idle: bool = True,
    allow_active_cuda_jobs: bool = False,
) -> RecurrenceMatrixPlan:
    """Build a concrete recurrence matrix plan."""
    candidates = (
        CandidateSpec(
            hgdn_config,
            "HGDN recurrence A direct",
            f"{run_prefix_base}_a_direct",
            "direct",
        ),
        CandidateSpec(
            hgdn_config,
            "HGDN recurrence B direct_fused",
            f"{run_prefix_base}_b_direct_fused",
            "direct_fused",
        ),
        CandidateSpec(
            hgdn_config,
            "HGDN recurrence C compile_visible",
            f"{run_prefix_base}_c_compile_visible",
            "compile_visible",
        ),
        CandidateSpec(
            attn_config,
            "Attention-only baseline diagnostic control",
            f"{run_prefix_base}_d_attention_only",
            "direct",
        ),
    )
    return RecurrenceMatrixPlan(
        run_prefix_base=run_prefix_base,
        bundle_stage_dir=bundle_stage_dir
        or Path(f"local-scratch/{run_prefix_base}_bundle"),
        archive_output=archive_output
        or Path(f"local-scratch/{run_prefix_base}_bundle.7z"),
        command_log=command_log or Path(f"local-scratch/{run_prefix_base}_commands.sh"),
        contract=contract,
        candidates=candidates,
        check_cuda_idle=check_cuda_idle,
        allow_active_cuda_jobs=allow_active_cuda_jobs,
    )


def _print_contract(contract: LocalTrainContract) -> None:
    """Print common contract fields."""
    print(f"python_bin={contract.python_bin}")
    print(f"use_wandb={int(contract.use_wandb)}")
    print(f"wandb_mode={contract.wandb_mode}")
    print(f"wandb_project={contract.wandb_project}")
    print(f"wandb_watch={contract.wandb_watch}")
    print(f"wandb_watch_log_freq={contract.wandb_watch_log_freq}")
    print(f"TORCH_LOGS={contract.torch_logs or '<unset>'}")
    print(f"TORCH_TRACE={contract.torch_trace or '<unset>'}")
    print(f"TORCHINDUCTOR_MAX_AUTOTUNE={contract.torchinductor_max_autotune}")
    print(f"TORCHINDUCTOR_MAX_AUTOTUNE_GEMM={contract.torchinductor_max_autotune_gemm}")
    print(f"ngpu={contract.ngpu}")
    print(f"grad_accum_steps={contract.grad_accum_steps}")
    print(f"iterations={contract.iterations}")
    print(f"train_batch_tokens={contract.train_batch_tokens}")
    print(f"train_seq_len={contract.train_seq_len}")
    print(f"val_loss_every={contract.val_loss_every}")
    print(f"train_log_every={contract.train_log_every}")
    print(f"min_val_seqs={contract.min_val_seqs}")
    print(f"val_max_seqs={contract.val_max_seqs}")
    print(f"val_batch_size={contract.val_batch_size}")
    print(f"weight_decay={contract.weight_decay:g}")
    print(f"perf_skip_final_eval={int(contract.perf_skip_final_eval)}")
    print(f"compile={int(contract.compile)}")
    print(f"compile_strategy={contract.compile_strategy}")
    print(f"distributed_mode={contract.distributed_mode}")
    print(f"allow_existing_logs={int(contract.allow_existing_logs)}")
    print(f"max_wallclock_seconds={contract.max_wallclock_seconds:g}")
    print(f"data_path={contract.data_path}")
    print(f"tokenizer_path={contract.tokenizer_path}")
    print(f"vocab_size={contract.vocab_size}")


def run_naive_contract_search(plan: NaiveSearchPlan) -> ExperimentResult:
    """Run the local naive-contract search from a structured Python plan."""
    print()
    print(
        ">>> Local HGDN naive-contract search (sparse exact-contract candidate screen)"
    )
    _print_contract(plan.contract)
    print(f"gdn_fla_recurrence_mode={plan.gdn_fla_recurrence_mode}")
    print(f"size_screen_config={plan.size_screen_config}")
    print(f"size_screen_output={plan.size_screen_output}")
    print(f"archive_output={plan.archive_output}")
    print("batch:")
    for candidate in plan.candidates:
        print(f"  - {candidate.run_prefix} :: {candidate.label} :: {candidate.config}")

    run_ids = tuple(
        _run_id(candidate.run_prefix, candidate.config, plan.contract)
        for candidate in plan.candidates
    )
    specs = tuple(
        _train_spec(plan.contract, candidate, run_id)
        for candidate, run_id in zip(plan.candidates, run_ids, strict=True)
    )
    screen_command = shlex.join(
        [
            plan.contract.python_bin,
            "scripts/screen_hgdn_arch_sizes.py",
            "--config",
            str(plan.size_screen_config),
            "--gdn-fla-recurrence-mode",
            plan.gdn_fla_recurrence_mode,
            "--output-dir",
            str(plan.size_screen_output),
        ]
    )
    write_command_log(plan.command_log, specs, prelude=[screen_command])
    _check_fresh_logs(run_ids, allow_existing_logs=plan.contract.allow_existing_logs)

    status = 0
    try:
        print()
        print(">>> artifact-size screen")
        run_size_screen(
            config=plan.size_screen_config,
            output_dir=plan.size_screen_output,
            gdn_fla_recurrence_mode=plan.gdn_fla_recurrence_mode,
        )
        status = _run_specs(specs)
    except KeyboardInterrupt:
        status = 130
        raise
    except BaseException:
        status = 1
        raise
    finally:
        print()
        print(">>> bundle outputs")
        manifest = {
            "run_prefix_base": plan.run_prefix_base,
            "wandb_project": plan.contract.wandb_project,
            "wandb_mode": plan.contract.wandb_mode,
            "exit_status": status,
            "size_screen": {
                "config": str(plan.size_screen_config),
                "output_dir": str(plan.size_screen_output),
            },
            "contract": {
                **plan.contract.manifest_contract(),
                "gdn_fla_recurrence_mode": plan.gdn_fla_recurrence_mode,
                "muon_distributed_mode": "per_config",
                "gdn_w_g_optimizer": "per_config",
            },
            "candidates": [
                {
                    "config": candidate.config,
                    "label": candidate.label,
                    "run_id": run_id,
                    "gdn_fla_recurrence_mode": candidate.recurrence_mode,
                }
                for candidate, run_id in zip(plan.candidates, run_ids, strict=True)
            ],
        }
        _bundle_common(
            bundle_stage_dir=plan.bundle_stage_dir,
            archive_output=plan.archive_output,
            command_log=plan.command_log,
            configs=[candidate.config for candidate in plan.candidates]
            + [str(plan.size_screen_config)],
            run_ids=run_ids,
            manifest=manifest,
            size_screen_output=plan.size_screen_output,
        )
        print(f"bundle_archive={plan.archive_output}")
    if status != 0:
        raise SystemExit(status)
    return ExperimentResult(status, plan.bundle_stage_dir, plan.archive_output, run_ids)


def run_recurrence_matrix(plan: RecurrenceMatrixPlan) -> ExperimentResult:
    """Run the local recurrence matrix from a structured Python plan."""
    print()
    print(">>> Local HGDN recurrence implementation matrix")
    _print_contract(plan.contract)
    print(f"check_cuda_idle={int(plan.check_cuda_idle)}")
    print(f"allow_active_cuda_jobs={int(plan.allow_active_cuda_jobs)}")
    print(f"archive_output={plan.archive_output}")
    print("batch:")
    for candidate in plan.candidates:
        print(
            f"  - {candidate.run_prefix} :: mode={candidate.recurrence_mode} :: "
            f"{candidate.label} :: {candidate.config}"
        )

    run_ids = tuple(
        _run_id(candidate.run_prefix, candidate.config, plan.contract)
        for candidate in plan.candidates
    )
    specs = tuple(
        _train_spec(plan.contract, candidate, run_id)
        for candidate, run_id in zip(plan.candidates, run_ids, strict=True)
    )
    write_command_log(plan.command_log, specs, prelude=[])
    _check_fresh_logs(run_ids, allow_existing_logs=plan.contract.allow_existing_logs)

    status = 0
    try:
        check_cuda_jobs(plan.check_cuda_idle, plan.allow_active_cuda_jobs)
        status = _run_specs(specs)
    except KeyboardInterrupt:
        status = 130
        raise
    except BaseException:
        status = 1
        raise
    finally:
        print()
        print(">>> bundle outputs")
        manifest = {
            "run_prefix_base": plan.run_prefix_base,
            "wandb_project": plan.contract.wandb_project,
            "wandb_mode": plan.contract.wandb_mode,
            "exit_status": status,
            "command_log": str(plan.command_log),
            "contract": plan.contract.manifest_contract(),
            "runs": [
                {
                    "config": candidate.config,
                    "label": candidate.label,
                    "run_id": run_id,
                    "gdn_fla_recurrence_mode": candidate.recurrence_mode,
                }
                for candidate, run_id in zip(plan.candidates, run_ids, strict=True)
            ],
        }
        _bundle_common(
            bundle_stage_dir=plan.bundle_stage_dir,
            archive_output=plan.archive_output,
            command_log=plan.command_log,
            configs=[candidate.config for candidate in plan.candidates],
            run_ids=run_ids,
            manifest=manifest,
        )
        print(f"bundle_archive={plan.archive_output}")
    if status != 0:
        raise SystemExit(status)
    return ExperimentResult(status, plan.bundle_stage_dir, plan.archive_output, run_ids)
