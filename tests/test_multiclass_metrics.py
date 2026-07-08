import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evenet_lite.data import EvenetTensorDataset
from evenet_lite.metrics import compute_classification_metrics, compute_loss
from evenet_lite.trainer import Trainer, TrainerConfig


class PassthroughClassifier(torch.nn.Module):
    num_classes = 3

    def forward(self, globals: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return globals


class MultiHeadPassthroughClassifier(torch.nn.Module):
    num_classes = {"cls1": 3, "cls2": 2}

    def forward(self, globals: torch.Tensor, **_: torch.Tensor):  # type: ignore[override]
        return {
            "cls1": globals[:, :3],
            "cls2": globals[:, 3:5],
        }


class MultiHeadLinearClassifier(torch.nn.Module):
    num_classes = {"cls1": 3, "cls2": 2}

    def __init__(self):
        super().__init__()
        self.cls1 = torch.nn.Linear(5, 3)
        self.cls2 = torch.nn.Linear(5, 2)

    def forward(self, globals: torch.Tensor, **_: torch.Tensor):  # type: ignore[override]
        return {"cls1": self.cls1(globals), "cls2": self.cls2(globals)}


class ThreeHeadLinearClassifier(torch.nn.Module):
    num_classes = {"kl1": 2, "kl5": 2, "vbf": 2}

    def __init__(self):
        super().__init__()
        self.kl1 = torch.nn.Linear(6, 2)
        self.kl5 = torch.nn.Linear(6, 2)
        self.vbf = torch.nn.Linear(6, 2)

    def forward(self, globals: torch.Tensor, **_: torch.Tensor):  # type: ignore[override]
        return {
            "kl1": self.kl1(globals),
            "kl5": self.kl5(globals),
            "vbf": self.vbf(globals),
        }


class MulticlassMetricsTest(unittest.TestCase):
    def test_compute_loss_ignores_ignore_index(self):
        logits = torch.tensor(
            [
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 4.0],
            ]
        )
        targets = torch.tensor([0, -100, 1])

        actual = compute_loss(logits, targets, None, gamma=0.0, ignore_index=-100)
        expected = compute_loss(logits[[0, 2]], targets[[0, 2]], None, gamma=0.0, ignore_index=-100)

        torch.testing.assert_close(actual, expected)
        all_ignored = compute_loss(logits, torch.full_like(targets, -100), None, ignore_index=-100)
        torch.testing.assert_close(all_ignored, torch.tensor(0.0))

    def test_compute_classification_metrics_weighted_multiclass(self):
        logits = np.array(
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [0.0, 1.0, 3.0],
                [0.0, 3.0, 1.0],
                [2.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
            ]
        )
        targets = np.array([0, 1, 2, 0, 1, 2])
        weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 3.0])

        metrics = compute_classification_metrics(logits, targets, weights, class_labels=["a", "b", "c"])

        np.testing.assert_allclose(
            metrics["confusion_matrix"],
            np.array(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            ),
        )
        self.assertAlmostEqual(metrics["accuracy"], 7.0 / 9.0)
        self.assertGreater(metrics["macro_f1"], 0.0)
        self.assertTrue(np.isfinite(metrics["class_auc"]).all())

    def test_trainer_evaluate_multiclass_skips_binary_physics_and_exports_by_class(self):
        features = {
            "globals": torch.tensor(
                [
                    [4.0, 1.0, 0.0],
                    [0.0, 3.0, 1.0],
                    [0.0, 1.0, 3.0],
                    [0.0, 3.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [1.0, 0.0, 2.0],
                ]
            )
        }
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 3.0])
        dataset = EvenetTensorDataset(features, labels, weights)
        trainer = Trainer(
            PassthroughClassifier(),
            {"globals": ["a", "b", "c"]},
            TrainerConfig(device="cpu", num_workers=0, use_wandb=False),
            class_labels=["a", "b", "c"],
        )
        preds = trainer.predict(dataset, batch_size=2)
        self.assertIsInstance(preds, torch.Tensor)
        self.assertEqual(tuple(preds.shape), (6, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = trainer.evaluate(dataset, batch_size=2, output_path=tmpdir)

            self.assertIn("macro_f1", metrics)
            self.assertIn("weighted_auc", metrics)
            self.assertNotIn("max_sic", metrics)
            for label in ["a", "b", "c"]:
                self.assertTrue((Path(tmpdir) / f"eval_output-{label}.npz").exists())

    def test_trainer_requires_explicit_class_labels(self):
        with self.assertRaisesRegex(ValueError, "class_labels is required"):
            Trainer(
                PassthroughClassifier(),
                {"globals": ["a", "b", "c"]},
                TrainerConfig(device="cpu", num_workers=0, use_wandb=False),
            )

    def test_trainer_evaluate_multi_head_metrics_and_exports_by_head(self):
        features = {
            "globals": torch.tensor(
                [
                    [4.0, 1.0, 0.0, 3.0, 0.0],
                    [0.0, 3.0, 1.0, 0.0, 3.0],
                    [0.0, 1.0, 3.0, 4.0, 0.0],
                    [0.0, 3.0, 1.0, 0.0, 4.0],
                ]
            )
        }
        labels = {
            "cls1": torch.tensor([0, 1, 2, 1]),
            "cls2": torch.tensor([0, 1, 0, 1]),
        }
        class_labels = {
            "cls1": {"name": ["a", "b", "c"], "lambda": 1.0},
            "cls2": {"name": ["x", "y"], "lambda": 0.5},
        }
        dataset = EvenetTensorDataset(features, labels)
        trainer = Trainer(
            MultiHeadPassthroughClassifier(),
            {"globals": ["a", "b", "c", "x", "y"]},
            TrainerConfig(
                device="cpu",
                num_workers=0,
                use_wandb=False,
                compute_physics_metrics=False,
                loss_gamma={"cls1": 0.0, "cls2": 1.0},
            ),
            class_labels=class_labels,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = trainer.evaluate(dataset, batch_size=2, output_path=tmpdir)

            self.assertIn("loss", metrics)
            self.assertIn("cls1/loss", metrics)
            self.assertIn("cls2/loss", metrics)
            self.assertIn("cls1/accuracy", metrics)
            self.assertIn("cls2/accuracy", metrics)
            self.assertNotIn("accuracy", metrics)
            for expected in [
                "eval_output-cls1-a.npz",
                "eval_output-cls1-b.npz",
                "eval_output-cls1-c.npz",
                "eval_output-cls2-x.npz",
                "eval_output-cls2-y.npz",
            ]:
                self.assertTrue((Path(tmpdir) / expected).exists())

    def test_multi_head_label_keys_must_match_class_labels(self):
        features = {"globals": torch.randn(2, 5)}
        labels = {
            "cls1": torch.tensor([0, 1]),
            "missing": torch.tensor([0, 1]),
        }
        class_labels = {
            "cls1": {"name": ["a", "b", "c"], "lambda": 1.0},
            "cls2": {"name": ["x", "y"], "lambda": 1.0},
        }
        trainer = Trainer(
            MultiHeadPassthroughClassifier(),
            {"globals": ["a", "b", "c", "x", "y"]},
            TrainerConfig(
                device="cpu",
                num_workers=0,
                use_wandb=False,
                compute_physics_metrics=False,
                loss_gamma={"cls1": 0.0, "cls2": 0.0},
            ),
            class_labels=class_labels,
        )

        with self.assertRaisesRegex(ValueError, "do not match class_labels heads"):
            trainer.setup_datasets((features, labels, None), None, None)

    def test_multi_head_label_keys_are_not_cast_to_strings(self):
        class NumericHeadClassifier(torch.nn.Module):
            num_classes = {"1": 3, "cls2": 2}

            def forward(self, globals: torch.Tensor, **_: torch.Tensor):  # type: ignore[override]
                return {
                    "1": globals[:, :3],
                    "cls2": globals[:, 3:5],
                }

        features = {"globals": torch.randn(2, 5)}
        labels = {
            1: torch.tensor([0, 1]),
            "cls2": torch.tensor([0, 1]),
        }
        class_labels = {
            "1": {"name": ["a", "b", "c"], "lambda": 1.0},
            "cls2": {"name": ["x", "y"], "lambda": 1.0},
        }
        trainer = Trainer(
            NumericHeadClassifier(),
            {"globals": ["a", "b", "c", "x", "y"]},
            TrainerConfig(
                device="cpu",
                num_workers=0,
                use_wandb=False,
                compute_physics_metrics=False,
                loss_gamma={"1": 0.0, "cls2": 0.0},
            ),
            class_labels=class_labels,
        )

        with self.assertRaisesRegex(ValueError, "do not match class_labels heads"):
            trainer.setup_datasets((features, labels, None), None, None)

    def test_multi_head_loss_gamma_must_be_named(self):
        class_labels = {
            "cls1": {"name": ["a", "b", "c"], "lambda": 1.0},
            "cls2": {"name": ["x", "y"], "lambda": 0.5},
        }
        with self.assertRaisesRegex(ValueError, "loss_gamma must be a dictionary"):
            Trainer(
                MultiHeadPassthroughClassifier(),
                {"globals": ["a", "b", "c", "x", "y"]},
                TrainerConfig(device="cpu", num_workers=0, use_wandb=False, loss_gamma=0.0),
                class_labels=class_labels,
            )

    def test_multi_head_head_name_and_lambda_are_validated(self):
        with self.assertRaisesRegex(ValueError, "without '.' or '/'"):
            Trainer(
                MultiHeadPassthroughClassifier(),
                {"globals": ["a", "b", "c", "x", "y"]},
                TrainerConfig(device="cpu", num_workers=0, use_wandb=False),
                class_labels={"bad/head": {"name": ["a", "b"], "lambda": 1.0}},
            )
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            Trainer(
                MultiHeadPassthroughClassifier(),
                {"globals": ["a", "b", "c", "x", "y"]},
                TrainerConfig(device="cpu", num_workers=0, use_wandb=False),
                class_labels={"cls1": {"name": ["a", "b"], "lambda": -1.0}},
            )

    def test_multi_head_train_loop_uses_named_losses(self):
        features = {"globals": torch.randn(4, 5)}
        labels = {
            "cls1": torch.tensor([0, 1, 2, 1]),
            "cls2": torch.tensor([0, 1, 0, 1]),
        }
        class_labels = {
            "cls1": {"name": ["a", "b", "c"], "lambda": 1.0},
            "cls2": {"name": ["x", "y"], "lambda": 0.5},
        }
        trainer = Trainer(
            MultiHeadLinearClassifier(),
            {"globals": ["a", "b", "c", "x", "y"]},
            TrainerConfig(
                device="cpu",
                num_workers=0,
                use_wandb=False,
                compute_physics_metrics=False,
                loss_gamma={"cls1": 0.0, "cls2": 1.0},
                lr=[1e-3],
                weight_decay=[0.0],
                module_lists=[["cls1", "cls2"]],
            ),
            class_labels=class_labels,
        )

        trainer.train((features, labels, None), None, None, epochs=1, batch_size=2, sampler=None)

        self.assertEqual(trainer.global_step, 2)

    def test_multi_head_ignore_index_skips_head_specific_rows(self):
        features = {"globals": torch.randn(4, 6)}
        labels = {
            "kl1": torch.tensor([1, -100, -100, 0]),
            "kl5": torch.tensor([-100, 1, -100, 0]),
            "vbf": torch.tensor([-100, -100, 1, 0]),
        }
        class_labels = {
            "kl1": {"name": ["bkg", "sig"], "lambda": 1.0},
            "kl5": {"name": ["bkg", "sig"], "lambda": 1.0},
            "vbf": {"name": ["bkg", "sig"], "lambda": 1.0},
        }
        trainer = Trainer(
            ThreeHeadLinearClassifier(),
            {"globals": [f"g{i}" for i in range(6)]},
            TrainerConfig(
                device="cpu",
                num_workers=0,
                use_wandb=False,
                compute_physics_metrics=False,
                loss_gamma={"kl1": 0.0, "kl5": 0.0, "vbf": 0.0},
                lr=[1e-3],
                weight_decay=[0.0],
                module_lists=[["kl1", "kl5", "vbf"]],
                ignore_index=-100,
            ),
            class_labels=class_labels,
        )

        trainer.train((features, labels, None), None, None, epochs=1, batch_size=2, sampler=None)
        metrics = trainer.evaluate(EvenetTensorDataset(features, labels), batch_size=2)

        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(metrics["kl1/support_bkg"], 1.0)
        self.assertEqual(metrics["kl1/support_sig"], 1.0)
        self.assertEqual(metrics["kl5/support_bkg"], 1.0)
        self.assertEqual(metrics["kl5/support_sig"], 1.0)
        self.assertEqual(metrics["vbf/support_bkg"], 1.0)
        self.assertEqual(metrics["vbf/support_sig"], 1.0)


if __name__ == "__main__":
    unittest.main()
