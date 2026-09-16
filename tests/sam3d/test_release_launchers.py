"""Argument-level launcher checks: no Docker process or accelerator is started."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


class ReleaseLaunchers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scripts = self.root / "projects/sam3d/scripts"
        scripts.mkdir(parents=True)
        for name in ("train_v93.sh", "train_complete106k.sh"):
            shutil.copy2(REPO / "projects/sam3d" / name, scripts.parent / name)
        self.output = self.root / "args.txt"
        runner = scripts / "run_cosmos_sam3d_musa.sh"
        runner.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$TEST_OUTPUT"\n')
        runner.chmod(0o755)
        self.env = dict(os.environ)
        for name in ("COSMOS_SAM3D_WORKSPACE", "SAM3D_DATASET_ROOT", "SAM3D_CACHE_ROOT", "SAM3D_OBJECTS_ROOT"):
            path = self.root / name
            path.mkdir()
            self.env[name] = str(path)
        for name in ("SAM3D_NATIVE_INDEX", "SAM3D_BASE_CHECKPOINT", "ACTION_NORM_PATH"):
            path = self.root / (name + ".pt")
            path.write_text("fixture")
            self.env[name] = str(path)
        self.env.update(COSMOS_PROJECT=str(self.root), COSMOS_MUSA_IMAGE="fixture",
                        MASTER_ADDR="localhost", TEST_OUTPUT=str(self.output),
                        NNODES="1", NPROC_PER_NODE="1", FSDP_SHARD_SIZE="1")

    def launch(self, variant):
        return subprocess.run(["bash", str(self.root / "projects/sam3d/train_complete106k.sh"), variant, "0"],
                              env=self.env, capture_output=True, text=True)

    def test_action_recipe_and_consolidated_checkpoint(self):
        result = self.launch("action")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.output.read_text().splitlines()
        for expected in ("--nproc-per-node=1", "model.config.fsdp_shard_size=1",
                         "model.config.net.action_conditioning_enabled=true",
                         "model.config.net.action_conditioning_temporal_radius=4",
                         "model.config.net.action_conditioning_output_norm=tanh",
                         "dataloader_train.dataset.sam3d_require_complete_conditions=true",
                         "dataloader_train.dataset.expected_num_samples=105498",
                         "checkpoint.load_path=" + self.env["SAM3D_BASE_CHECKPOINT"]):
            self.assertIn(expected, args)

    def test_semantic_disables_inherited_action_conditioning(self):
        self.env["ACTION_CONDITIONING_ENABLED"] = "true"
        result = self.launch("semantic")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("model.config.net.action_conditioning_enabled=true", self.output.read_text())

    def test_missing_action_statistics_fails_before_runtime(self):
        del self.env["ACTION_NORM_PATH"]
        result = self.launch("action")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ACTION_NORM_PATH", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
