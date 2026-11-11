"""Single-run launcher and RankSim-style SLURM grid submission (standard library only)."""
import argparse
import itertools
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from source_contract import read_source_args

DEFAULT_GRID = ("0.1", "w2", "10.0", "64", "1e-5")
DATASETS = ("agedb", "imdb_wiki")


def validate_parameters(parameters):
    h, kind, l2sp, batch, lr = parameters
    if kind not in ("w2", "l1", "kl", "skl"):
        raise ValueError("loss must be w2, l1, kl, or skl")
    for name, value, zero_allowed in (("bandwidth", h, False), ("L2-SP", l2sp, True), ("learning rate", lr, False)):
        number = float(value)
        if not math.isfinite(number) or number < 0 or (number == 0 and not zero_allowed):
            raise ValueError(f"invalid {name}: {value}")
    if int(batch) < 2:
        raise ValueError("batch size must be at least 2")


def validate_overrides(overrides):
    for value in overrides:
        if "=" not in value or value.startswith("-") or not value.split("=", 1)[0]:
            raise ValueError(f"expected key=value override, got {value!r}")


def resolve_checkpoint(path):
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def run_command(repo, dataset, checkpoint, parameters, overrides, environ=None):
    environ = os.environ if environ is None else environ
    validate_parameters(parameters)
    validate_overrides(overrides)
    source_dataset = read_source_args(checkpoint).get("dataset")
    if source_dataset is not None and source_dataset != dataset:
        raise ValueError(f"requested {dataset}, but source training.log says {source_dataset}: {checkpoint}")
    h, kind, l2sp, batch, lr = parameters
    command = [environ.get("PYTHON_BIN") or environ.get("PYTHON") or sys.executable,
               str(repo / "postadapt.py"), "-c", str(repo / "PostTrainAdpt.yaml"),
               f"model.pretrained={checkpoint}", f"dataset.name={dataset}",
               f"model.bandwidth={h}", f"model.condLoss={kind}", f"model.l2spWeight={l2sp}",
               f"dataset.batch_size={batch}", f"model.optim.lr={lr}",
               f"model.optim.name={environ.get('OPTIMIZER', 'sgd')}",
               f"model.optim.param={environ.get('OPTIM_PARAM', 'all')}",
               "model.optim.weight_decay=0.0",
               f"trainer.max_steps={2000 if dataset == 'agedb' else 10000}"]
    if dataset == "imdb_wiki":
        command.append("logger.save_train_predictions=false")
    # CSV, normalization and architecture stay at the corrected YAML's auto defaults.
    # Explicit overrides are appended last and take precedence.
    return command + list(overrides)


def finished_source(log):
    if not log.is_file():
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    start = text.rfind("Test best model on testset")
    if start < 0 or text.rfind("Args: Namespace(") > start:
        return False
    tail = text[start:]
    return "Test loss:" in tail and "Done" in tail


def find_sources(repo, datasets, pattern, roots=None):
    """Scan both ConR output layouts recursively, deduplicating overlapping roots."""
    found = {}
    for dataset in datasets:
        folder = "agedb-dir" if dataset == "agedb" else "imdb-wiki-dir"
        scan_roots = roots if roots is not None else [repo / "checkpoint" / dataset, repo / folder / "checkpoint"]
        for root in scan_roots:
            root = Path(root).expanduser().resolve()
            if not root.is_dir():
                continue
            for candidate in root.rglob("ckpt.best.pth.tar"):
                candidate = candidate.resolve()
                if candidate in found or not candidate.parent.match(pattern):
                    continue
                if not finished_source(candidate.parent / "training.log"):
                    continue
                actual = read_source_args(candidate).get("dataset")
                if actual is not None and actual != dataset:
                    continue
                # A shared custom root needs dataset metadata to avoid assigning the wrong dataset.
                if actual is None and roots is not None and len(datasets) > 1:
                    raise ValueError(f"cannot identify dataset for {candidate}; select one --datasets value")
                found[candidate] = dataset
    return sorted(found.items(), key=lambda item: str(item[0]))


def grid_values(environ):
    keys = ("BANDWIDTHS", "COND_LOSSES", "L2SP_WEIGHTS", "BATCH_SIZES", "LRS")
    axes = [shlex.split(environ.get(key, default)) for key, default in zip(keys, DEFAULT_GRID)]
    if not all(axes):
        raise ValueError("grid axes must not be empty")
    grid = list(dict.fromkeys(itertools.product(*axes)))
    for parameters in grid:
        validate_parameters(parameters)
    return grid


def submission_command(repo, dataset, checkpoint, parameters, overrides, local=False, environ=None):
    environ = os.environ if environ is None else environ
    runner = str(repo / "postadapt" / "run_postadapt.sh")
    arguments = [dataset, str(checkpoint), *parameters, *overrides]
    if local:
        return ["bash", runner, *arguments]
    return [environ.get("SBATCH_BIN", "sbatch"), "--export=ALL", f"--chdir={repo}",
            f"--output={repo / 'postadapt/log/output'}/%x_%j.log",
            f"--error={repo / 'postadapt/log/error'}/%x_%j.log",
            f"--job-name=ConR_PostAdpt_{dataset}_h{parameters[0]}_{parameters[1]}_w{parameters[2]}",
            runner, *arguments]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    single = actions.add_parser("run", help="run one source checkpoint")
    single.add_argument("--dry-run", action="store_true")
    single.add_argument("dataset", choices=DATASETS)
    single.add_argument("checkpoint")
    single.add_argument("parameters", nargs="*", help="[bandwidth loss l2sp batch lr] [key=value ...]")
    submit = actions.add_parser("submit", help="submit a grid over finished source runs")
    submit.add_argument("mode", nargs="?", choices=("slurm", "local"), default="slurm")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--datasets", nargs="+", choices=DATASETS)
    submit.add_argument("--runs", default=os.environ.get("RUNS", "*diraligned*"), help="source run folder glob")
    submit.add_argument("--roots", nargs="+", type=Path, help="custom checkpoint roots (recursive)")
    submit.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    submit.add_argument("--limit", type=int, help="maximum jobs to submit")
    args = parser.parse_args(argv)
    repo = Path(os.environ.get("CONR_ROOT", REPO)).expanduser().resolve()
    if not (repo / "postadapt.py").is_file() or not (repo / "source_contract.py").is_file():
        parser.error(f"ConR with source-contract support not found: {repo}")
    try:
        if args.action == "run":
            parameters = list(DEFAULT_GRID)
            extra = list(args.parameters)
            for index in range(5):
                if not extra or "=" in extra[0]:
                    break
                parameters[index] = extra.pop(0)
            checkpoint = resolve_checkpoint(args.checkpoint)
            command = run_command(repo, args.dataset, checkpoint, parameters, extra)
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=repo, check=True)
            return 0

        datasets = args.datasets or shlex.split(os.environ.get("DATASETS", "agedb"))
        if not datasets or any(dataset not in DATASETS for dataset in datasets):
            raise ValueError("DATASETS must contain agedb and/or imdb_wiki")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        validate_overrides(args.override)
        sources = find_sources(repo, datasets, args.runs, args.roots)
        grid = grid_values(os.environ)
        jobs = [(checkpoint, dataset, parameters) for checkpoint, dataset in sources for parameters in grid]
        if args.limit is not None:
            jobs = jobs[:args.limit]
        # Validate all jobs before submitting any; do not train or create logs during dry runs.
        for checkpoint, dataset, parameters in jobs:
            run_command(repo, dataset, checkpoint, parameters, args.override)
        if not jobs:
            print(f"No finished source runs matched {args.runs!r}. Use RUNS='*' for original ConR runs.")
            return 0
        if not args.dry_run:
            for stream in ("output", "error"):
                (repo / "postadapt/log" / stream).mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, CONR_ROOT=str(repo))
        for checkpoint, dataset, parameters in jobs:
            command = submission_command(repo, dataset, checkpoint, parameters, args.override,
                                         local=args.mode == "local")
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=repo, env=environment, check=True)
        print(f"{'Planned' if args.dry_run else 'Launched'} {len(jobs)} jobs from {len(sources)} source runs.")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
