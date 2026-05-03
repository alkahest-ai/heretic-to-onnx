from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
    from safetensors.torch import save_file

    from scripts.merge_lora_scaled import load_base_tensors
except ModuleNotFoundError:
    torch = None
    save_file = None
    load_base_tensors = None


@unittest.skipIf(load_base_tensors is None, "torch/safetensors not installed")
class MergeLoraScaledTests(unittest.TestCase):
    def test_load_base_tensors_accepts_sharded_safetensors(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({"model.layers.0.weight": torch.ones(1, 1)}, root / "model-00001-of-00002.safetensors")
            save_file({"model.layers.1.weight": torch.ones(1, 1) * 2}, root / "model-00002-of-00002.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.weight": "model-00001-of-00002.safetensors",
                            "model.layers.1.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                )
                + "\n"
            )

            tensors, _, resolved = load_base_tensors(root, "model.safetensors")

            self.assertEqual(resolved, "model.safetensors.index.json")
            self.assertEqual(set(tensors), {"model.layers.0.weight", "model.layers.1.weight"})
            self.assertEqual(tensors["model.layers.1.weight"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
