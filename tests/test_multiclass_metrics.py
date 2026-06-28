import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evenet_lite.data import EvenetTensorDataset
from evenet_lite.metrics import compute_classification_metrics
from evenet_lite.trainer import Trainer, TrainerConfig


class PassthroughClassifier(torch.nn.Module):
    num_classes = 3

    def forward(self, globals: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return globals


class MulticlassMetricsTest(unittest.TestCase):
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

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = trainer.evaluate(dataset, batch_size=2, output_path=tmpdir)

            self.assertIn("macro_f1", metrics)
            self.assertIn("weighted_auc", metrics)
            self.assertNotIn("max_sic", metrics)
            for label in ["a", "b", "c"]:
                self.assertTrue((Path(tmpdir) / f"eval_output-{label}.npz").exists())


if __name__ == "__main__":
    unittest.main()
