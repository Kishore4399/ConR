"""CPU checkpoint parity checks; skipped when the training libraries are unavailable."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "scipy", "pandas"))


@unittest.skipUnless(AVAILABLE, "requires torch, numpy, scipy and pandas")
class CheckpointParityTests(unittest.TestCase):
    def test_source_and_wrapped_checkpoint_predictions_match(self):
        import torch
        from conr_backbone import Encoder, load_checkpoint, read_checkpoint
        from resnet import resnet18, resnet50
        torch.set_num_threads(1)
        for name, builder in (("resnet18", resnet18), ("resnet50", resnet50)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                source = builder(fds=False, bucket_num=100, bucket_start=3, start_update=0,
                                 start_smooth=1, kernel="gaussian", ks=5, sigma=2, momentum=.9).eval()
                path = Path(directory) / "source.pt"
                state = {f"module.{key}": value for key, value in source.state_dict().items()}
                torch.save({"model": name, "state_dict": state}, path)
                self.assertEqual(read_checkpoint(path)[1], name)
                encoder = Encoder(name).eval()
                head = torch.nn.Linear(encoder.dim, 1).eval()
                load_checkpoint(encoder, head, path)
                images = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    torch.testing.assert_close(head(encoder(images)), source(images)[0], rtol=1e-5, atol=1e-5)
                # Original checkpoints can omit architecture metadata and module prefixes.
                torch.save({"state_dict": source.state_dict()}, path)
                self.assertEqual(read_checkpoint(path)[1], name)
                load_checkpoint(encoder, head, path)
                with torch.no_grad():
                    torch.testing.assert_close(head(encoder(images)), source(images)[0], rtol=1e-5, atol=1e-5)

    def test_wrong_architecture_is_rejected(self):
        import torch
        from conr_backbone import Encoder, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            encoder = Encoder("resnet18")
            head = torch.nn.Linear(512, 1)
            state = {key: value for key, value in encoder.encoder.state_dict().items()}
            state.update({f"linear.{key}": value for key, value in head.state_dict().items()})
            path = Path(directory) / "source.pt"
            torch.save({"state_dict": state}, path)
            wrong = Encoder("resnet50")
            with self.assertRaises(ValueError):
                load_checkpoint(wrong, torch.nn.Linear(2048, 1), path)


if __name__ == "__main__":
    unittest.main()
