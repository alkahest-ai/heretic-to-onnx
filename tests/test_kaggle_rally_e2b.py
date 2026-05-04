from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.kaggle_rally_e2b_two_stage_export import _find_artifacts, _has_merged_checkpoint
from scripts.kaggle_rally_e2b_two_stage_sft import _parser as sft_parser


class KaggleRallyE2BTests(unittest.TestCase):
    def test_find_artifacts_accepts_sharded_rally_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "rally-e2b-two-stage-sft"
            for subdir in ["stage-a-adapter", "stage-b-adapter", "stage-a-merged", "stage-ab-merged"]:
                (root / subdir).mkdir(parents=True)
            (root / "stage-a-adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
            (root / "stage-b-adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
            (root / "stage-a-merged" / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
            (root / "stage-ab-merged" / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')

            self.assertTrue(_has_merged_checkpoint(root / "stage-a-merged"))
            self.assertEqual(_find_artifacts(str(root), "rally-e2b-two-stage-sft"), root.resolve())

    def test_sft_defaults_use_v8_stage_mix(self) -> None:
        args = sft_parser().parse_args([])

        self.assertEqual(args.stage_a_repeats, 18)
        self.assertEqual(args.stage_b_boundary_repeats, 80)
        self.assertEqual(args.stage_b_adult_repeats, 40)
        self.assertEqual(args.stage_b_max_steps, 450)


if __name__ == "__main__":
    unittest.main()
