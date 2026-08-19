import json

import h5py
import numpy as np

from cosmos_predict2._src.predict2.datasets.local_datasets.worldarena_action_hdf5 import (
    WorldArenaActionIndex,
    aligned_action_indices,
    read_action_sequence,
    resample_action_sequence,
)


def _write_episode(path, length=9, corrupt_vector=False):
    left_arm = np.arange(length * 6, dtype=np.float64).reshape(length, 6)
    left_gripper = np.arange(length, dtype=np.float64) + 100
    right_arm = np.arange(length * 6, dtype=np.float64).reshape(length, 6) + 200
    right_gripper = np.arange(length, dtype=np.float64) + 300
    vector = np.concatenate(
        [left_arm, left_gripper[:, None], right_arm, right_gripper[:, None]], axis=1
    )
    if corrupt_vector:
        vector[0, 0] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("joint_action/left_arm", data=left_arm)
        handle.create_dataset("joint_action/left_gripper", data=left_gripper)
        handle.create_dataset("joint_action/right_arm", data=right_arm)
        handle.create_dataset("joint_action/right_gripper", data=right_gripper)
        handle.create_dataset("joint_action/vector", data=vector)
    return vector.astype(np.float32)


def test_worldarena_action_order_and_video_time_alignment(tmp_path):
    path = tmp_path / "data" / "fixed_scene_task" / "episode1.hdf5"
    expected = _write_episode(path)
    actions = read_action_sequence(path)
    np.testing.assert_allclose(actions, expected)

    indices = aligned_action_indices(
        action_frames=9,
        target_frames=3,
        video_frame_indices=[0, 5, 10],
        video_frame_count=11,
    )
    np.testing.assert_array_equal(indices, [0, 4, 8])
    sampled, sampled_indices = resample_action_sequence(
        actions,
        target_frames=3,
        video_frame_indices=[0, 5, 10],
        video_frame_count=11,
        alignment_offset=1,
    )
    np.testing.assert_array_equal(sampled_indices, [1, 5, 8])
    np.testing.assert_allclose(sampled, expected[[1, 5, 8]])


def test_vector_mismatch_is_rejected(tmp_path):
    path = tmp_path / "episode2.hdf5"
    _write_episode(path, corrupt_vector=True)
    try:
        read_action_sequence(path)
    except ValueError as error:
        assert "ordering differs" in str(error)
    else:
        raise AssertionError("corrupt joint_action/vector was accepted")


def test_questions_manifest_index_resolves_episode_video(tmp_path):
    path = tmp_path / "data" / "fixed_scene_task" / "episode7.hdf5"
    _write_episode(path)
    record = {
        "question_id": "worldarena2_track1/test/episode7",
        "task_name": "fixed_scene_task",
        "episode": 7,
        "trajectory_hdf5": "data/fixed_scene_task/episode7.hdf5",
    }
    (tmp_path / "questions.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    index = WorldArenaActionIndex(tmp_path)
    assert index.resolve("/paired/videos/episode7.mp4") == path.resolve()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_worldarena_action_order_and_video_time_alignment(root / "alignment")
        test_vector_mismatch_is_rejected(root / "mismatch")
        test_questions_manifest_index_resolves_episode_video(root / "index")
    print("WorldArena action HDF5 tests passed")
