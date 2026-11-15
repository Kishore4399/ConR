"""Source-input regression checks: python -m unittest discover -s tests -v."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_contract", ROOT / "source_contract.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def state_keys(layout, bottleneck=False):
    state = {"conv1.weight": None}
    for layer, blocks in enumerate(layout, 1):
        for block in range(blocks):
            state[f"layer{layer}.{block}.conv1.weight"] = None
            if bottleneck:
                state[f"layer{layer}.{block}.conv3.weight"] = None
    return state


# Transform descriptions, to compare the real functions without torch.
NAMES = ("Compose", "Normalize", "Resize", "RandomCrop", "RandomHorizontalFlip", "ToTensor", "ColorJitter",
         "RandomGrayscale", "RandomResizedCrop", "RandomApply", "CenterCrop")
FAKE_TRANSFORMS = SimpleNamespace(**{name: (lambda name: lambda *args, **kwargs: (name, args, kwargs))(name)
                                     for name in NAMES})
T = FAKE_TRANSFORMS
NORM_05 = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# PostTrainingAdpt condition-variance-cosine utils.get_transforms, aug="crop,flip,color,grayscale"
PTA_TRAIN = T.Compose([T.RandomResizedCrop(size=224, scale=(0.2, 1.)), T.RandomHorizontalFlip(),
                       T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(p=0.2),
                       T.ToTensor(), NORM_05])
PTA_EVAL = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM_05])


def extract(path, function, cls=None, extra=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    if cls:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    # The training datasets call a module-level pta_transform; load it alongside when present.
    module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    helpers = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "pta_transform"]
    scope = {"transforms": FAKE_TRANSFORMS, **(extra or {})}
    exec(compile(ast.Module(body=helpers + [fn], type_ignores=[]), path, "exec"), scope)
    return scope[function]


class SourceContractTests(unittest.TestCase):
    def test_infer_supported_backbones(self):
        self.assertEqual(contract.infer_backbone(state_keys((2, 2, 2, 2))), "resnet18")
        self.assertEqual(contract.infer_backbone(state_keys((3, 4, 6, 3), True)), "resnet50")

    def test_reject_unknown_or_missing_architecture(self):
        for state in ({}, state_keys((3, 4, 6, 3))):
            with self.assertRaises(ValueError):
                contract.infer_backbone(state)

    def test_explicit_architecture_must_match(self):
        self.assertEqual(contract.resolve_backbone("auto", "resnet18"), "resnet18")
        self.assertEqual(contract.resolve_backbone("resnet50", "resnet50"), "resnet50")
        with self.assertRaises(ValueError):
            contract.resolve_backbone("resnet50", "resnet18")

    def fixture(self, folder, **overrides):
        run = folder / "agedb-dir/checkpoint/run"
        run.mkdir(parents=True)
        args = dict(dataset="agedb", data_dir="images", data_csv="../../external/agedb.csv", img_size=224,
                    imagenet_pretrained=True)
        args.update(overrides)
        (run / "training.log").write_text("Args: Namespace(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")\n",
                                          encoding="utf-8")
        return run / "ckpt.best.pth.tar"

    def config(self):
        return dict(name="auto", normalization="auto", data_folder="auto", data_csv="auto", img_size="auto",
                    transforms="auto")

    def test_aligned_source_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = contract.resolve_source_data(self.config(), self.fixture(root))
            self.assertEqual(config["normalization"], "imagenet")
            self.assertEqual(config["name"], "agedb")
            self.assertEqual(Path(config["data_folder"]), root / "agedb-dir/images")
            self.assertEqual(Path(config["data_csv"]), (root / "agedb-dir/../../external/agedb.csv").resolve())

    def test_legacy_source_uses_conr_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.fixture(Path(directory), imagenet_pretrained=False)
            self.assertEqual(contract.resolve_source_data(self.config(), checkpoint)["normalization"], "conr")
            log = checkpoint.parent / "training.log"
            log.write_text(log.read_text().replace(", imagenet_pretrained=False", ""))
            self.assertEqual(contract.resolve_source_data(self.config(), checkpoint)["normalization"], "conr")

    def test_default_source_manifest_is_under_image_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = contract.resolve_source_data(self.config(), self.fixture(root, data_csv=None))
            self.assertEqual(Path(config["data_csv"]), root / "agedb-dir/images/agedb.csv")

    def test_overrides_work_without_log(self):
        config = dict(name="agedb", normalization="imagenet", data_csv="custom.csv", data_folder="images", img_size=224,
                      transforms="dir")
        self.assertEqual(contract.resolve_source_data(config, ROOT / "absent/ckpt"), config)

    def test_missing_metadata_and_bad_size_are_rejected(self):
        with self.assertRaises(ValueError):
            contract.resolve_source_data(self.config(), ROOT / "absent/ckpt")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "img_size=224"):
                contract.resolve_source_data(self.config(), self.fixture(Path(directory), img_size=128))

    def test_log_expressions_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.fixture(Path(directory))
            (checkpoint.parent / "training.log").write_text("Args: Namespace(data_csv='safe.csv', data_dir=print('do not run'))\n")
            self.assertEqual(contract.read_source_args(checkpoint), {"data_csv": "safe.csv"})

    def test_transforms_match_source_models(self):
        postadapt = extract("pta_data.py", "get_transforms")
        for path, cls in (("agedb-dir/datasets.py", "AgeDB"), ("imdb-wiki-dir/datasets.py", "IMDBWIKI")):
            source = extract(path, "get_transform", cls)
            for aug in ("pta", "dir"):
                for imagenet in (False, True):
                    for train in (False, True):
                        with self.subTest(path=path, aug=aug, imagenet=imagenet, train=train):
                            module = SimpleNamespace(img_size=224, split="train" if train else "test",
                                                     imagenet_normalize=imagenet, aug=aug)
                            self.assertEqual(source(module), postadapt(module.split, "crop,flip", 224,
                                                                      "imagenet" if imagenet else "conr", aug))

    def test_pta_preset_is_posttrainingadpt_exactly(self):
        postadapt = extract("pta_data.py", "get_transforms")
        for normalization in ("conr", "imagenet"):
            self.assertEqual(postadapt("train", "crop,flip", 224, normalization, "pta"), PTA_TRAIN)
            for split in ("val", "test"):
                self.assertEqual(postadapt(split, "", 224, normalization, "pta"), PTA_EVAL)

    def test_transforms_follow_source_aug_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.fixture(Path(directory), aug="pta")
            self.assertEqual(contract.resolve_source_data(self.config(), checkpoint)["transforms"], "pta")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.fixture(Path(directory))   # log from before --aug existed
            self.assertEqual(contract.resolve_source_data(self.config(), checkpoint)["transforms"], "dir")

    def test_data_module_forwards_normalization(self):
        fn = extract("pta_data.py", "_transform", "AgeDBPTADataModule",
                     {"get_transforms": extract("pta_data.py", "get_transforms")})
        for preset in ("pta", "dir"):
            actual = fn(SimpleNamespace(img_size=224, normalization="imagenet", transforms=preset), "val")
            expected = extract("pta_data.py", "get_transforms")("val", "", 224, "imagenet", preset)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
