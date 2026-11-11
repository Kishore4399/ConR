"""Submission tests use temporary source runs and never submit real SLURM jobs."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("conr_launcher", ROOT / "postadapt/launch_postadapt.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class SubmissionTests(unittest.TestCase):
    def source(self, repo, name="source_diraligned_seed0", dataset="agedb", finished=True):
        run = repo / "checkpoint" / dataset / "diraligned" / name
        run.mkdir(parents=True)
        checkpoint = run / "ckpt.best.pth.tar"
        checkpoint.write_text("placeholder for discovery; no model inference")
        text = f"Args: Namespace(dataset={dataset!r}, imagenet_pretrained=True)\n"
        text += "Test best model on testset...\n"
        if finished:
            text += "Test loss: MSE [10], L1 [6.31], G-Mean [4]\nDone\n"
        (run / "training.log").write_text(text)
        return checkpoint.resolve()

    def fake_repo(self, folder):
        repo = folder / "ConR with spaces"
        repo.mkdir()
        (repo / "postadapt.py").touch()
        (repo / "source_contract.py").touch()
        return repo

    def test_default_grid_and_custom_grid_deduplication(self):
        self.assertEqual(launcher.grid_values({}), [launcher.DEFAULT_GRID])
        grid = launcher.grid_values({"BANDWIDTHS": "0.1 0.5 0.1", "BATCH_SIZES": "64 256"})
        self.assertEqual(len(grid), 4)

    def test_recursive_discovery_requires_finished_test_and_run_filter(self):
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name)
            good = self.source(repo)
            self.source(repo, "unfinished_diraligned", finished=False)
            self.source(repo, "original_source")
            self.assertEqual(launcher.find_sources(repo, ["agedb"], "*diraligned*"), [(good, "agedb")])

    def test_overlapping_roots_are_deduplicated_and_dataset_is_checked(self):
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name)
            good = self.source(repo)
            self.source(repo, dataset="imdb_wiki")
            sources = launcher.find_sources(repo, ["agedb"], "*", [repo / "checkpoint", good.parent])
            self.assertEqual(sources, [(good, "agedb")])

    def test_restarted_incomplete_run_is_skipped(self):
        with tempfile.TemporaryDirectory() as name:
            checkpoint = self.source(Path(name))
            log = checkpoint.parent / "training.log"
            log.write_text(log.read_text() + "Args: Namespace(dataset='agedb')\n")
            self.assertFalse(launcher.finished_source(log))

    def test_single_run_preserves_auto_input_contract_and_path_argument(self):
        with tempfile.TemporaryDirectory() as name:
            checkpoint = self.source(Path(name), "with spaces_diraligned")
            command = launcher.run_command(ROOT, "agedb", checkpoint, launcher.DEFAULT_GRID, [], {})
            self.assertIn(f"model.pretrained={checkpoint}", command)
            self.assertIn("model.optim.name=sgd", command)
            self.assertIn("model.optim.param=all", command)
            self.assertFalse(any(value.startswith(("dataset.data_csv=", "dataset.normalization=", "model.backbone.params.name="))
                                 for value in command))

    def test_imdb_defaults_and_explicit_overrides_last(self):
        with tempfile.TemporaryDirectory() as name:
            checkpoint = self.source(Path(name), dataset="imdb_wiki")
            command = launcher.run_command(ROOT, "imdb_wiki", checkpoint, launcher.DEFAULT_GRID,
                                           ["trainer.max_steps=5", "dataset.data_csv=/chosen split.csv"], {})
            self.assertIn("trainer.max_steps=10000", command)
            self.assertIn("logger.save_train_predictions=false", command)
            self.assertEqual(command[-2:], ["trainer.max_steps=5", "dataset.data_csv=/chosen split.csv"])
            with self.assertRaises(ValueError):
                launcher.run_command(ROOT, "agedb", checkpoint, launcher.DEFAULT_GRID, [], {})

    def test_dry_run_does_not_launch_or_create_log_directories(self):
        with tempfile.TemporaryDirectory() as name:
            repo = self.fake_repo(Path(name))
            self.source(repo)
            with patch.dict(os.environ, {"CONR_ROOT": str(repo)}, clear=True), patch.object(subprocess, "run") as run:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(launcher.main(["submit", "--dry-run"]), 0)
                self.assertIn("Planned 1 jobs", output.getvalue())
                run.assert_not_called()
            self.assertFalse((repo / "postadapt/log").exists())

    def test_slurm_submission_has_absolute_paths_and_explicit_working_directory(self):
        with tempfile.TemporaryDirectory() as name:
            repo = self.fake_repo(Path(name))
            checkpoint = self.source(repo)
            with patch.dict(os.environ, {"CONR_ROOT": str(repo)}, clear=True), patch.object(subprocess, "run") as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(["submit"]), 0)
                command = run.call_args.args[0]
                self.assertEqual(command[0], "sbatch")
                self.assertIn("--export=ALL", command)
                self.assertIn(f"--chdir={repo}", command)
                self.assertIn(str(checkpoint), command)
                self.assertEqual(run.call_args.kwargs["env"]["CONR_ROOT"], str(repo))
            self.assertTrue((repo / "postadapt/log/output").is_dir())

    def test_local_mode_and_limit(self):
        with tempfile.TemporaryDirectory() as name:
            repo = self.fake_repo(Path(name))
            self.source(repo)
            self.source(repo, "second_diraligned")
            with patch.dict(os.environ, {"CONR_ROOT": str(repo)}, clear=True), patch.object(subprocess, "run") as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.main(["submit", "local", "--limit", "1"]), 0)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][:2], ["bash", str(repo / "postadapt/run_postadapt.sh")])

    def test_invalid_grid_fails_before_any_jobs_launch(self):
        with tempfile.TemporaryDirectory() as name:
            repo = self.fake_repo(Path(name))
            self.source(repo)
            with patch.dict(os.environ, {"CONR_ROOT": str(repo), "BANDWIDTHS": "0.1 nan"}, clear=True):
                with patch.object(subprocess, "run") as run, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        launcher.main(["submit"])
                    run.assert_not_called()

    def test_python_cli_single_and_batch_dry_runs(self):
        with tempfile.TemporaryDirectory() as name:
            checkpoint = self.source(Path(name), "space in source_diraligned")
            environment = dict(os.environ, CONR_ROOT=str(ROOT))
            script = str(ROOT / "postadapt/launch_postadapt.py")
            single = subprocess.run([sys.executable, script, "run", "--dry-run", "agedb", str(checkpoint),
                                     "trainer.max_steps=1"], env=environment, check=True, capture_output=True, text=True)
            self.assertIn("trainer.max_steps=1", single.stdout)
            self.assertIn("model.bandwidth=0.1", single.stdout)
            batch = subprocess.run([sys.executable, script, "submit", "--dry-run", "--roots", name],
                                    env=environment, check=True, capture_output=True, text=True)
            self.assertIn("Planned 1 jobs", batch.stdout)

    def test_shell_entrypoints_and_spooled_slurm_runner_dry_run(self):
        bash = shutil.which("bash")
        if bash is None and Path("C:/Program Files/Git/bin/bash.exe").exists():
            bash = "C:/Program Files/Git/bin/bash.exe"
        if bash is None:
            self.skipTest("bash unavailable")
        probe = subprocess.run([bash, "--version"], capture_output=True, text=True)
        if probe.returncode != 0 and (probe.returncode in (3221225794, -1073741502)
                                     or "NtCreateDirectoryObject" in probe.stderr):
            self.skipTest("Git Bash cannot start in this Windows environment")
        probe.check_returncode()
        with tempfile.TemporaryDirectory() as name:
            checkpoint = self.source(Path(name))
            environment = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
            environment.update(CONR_ROOT=ROOT.as_posix(), PYTHON_BIN=Path(sys.executable).as_posix())
            for script in ("run_postadapt.sh", "submit_postadapt.sh"):
                subprocess.run([bash, "-n", (ROOT / "postadapt" / script).as_posix()], check=True,
                               capture_output=True, text=True)
            result = subprocess.run([bash, (ROOT / "postadapt/run_postadapt.sh").as_posix(), "--dry-run", "agedb",
                                     checkpoint.as_posix(), "trainer.max_steps=1"], env=environment, check=True,
                                    capture_output=True, text=True)
            self.assertIn("trainer.max_steps=1", result.stdout)
            self.assertIn("postadapt.py", result.stdout)
            result = subprocess.run([bash, (ROOT / "postadapt/submit_postadapt.sh").as_posix(), "--dry-run",
                                     "--roots", Path(name).as_posix()], env=environment, check=True,
                                    capture_output=True, text=True)
            self.assertIn("Planned 1 jobs", result.stdout)
            copied = Path(name) / "spooled_job"
            shutil.copyfile(ROOT / "postadapt/run_postadapt.sh", copied)
            environment.update(SLURM_JOB_ID="123", SLURM_SUBMIT_DIR=(ROOT / "postadapt").as_posix())
            environment.pop("CONR_ROOT")
            result = subprocess.run([bash, copied.as_posix(), "--dry-run", "agedb", checkpoint.as_posix()],
                                    env=environment, check=True, capture_output=True, text=True)
            self.assertIn("postadapt.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
