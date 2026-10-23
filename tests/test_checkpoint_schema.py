import pytest

from humanoid_sim.checkpoint_schema import checkpoint_identity, validate_resume


def test_known_legacy_is_humanoid_only():
    identity = checkpoint_identity({'environment': 'Humanoid-v5'})
    assert identity['robot_id'] == 'humanoid'
    validate_resume({'environment': 'Humanoid-v5'}, identity)
    with pytest.raises(ValueError, match='robot_id'):
        validate_resume({'environment': 'Humanoid-v5'}, {**identity, 'robot_id': 'g1_29dof'})


def test_unknown_metadata_does_not_guess_robot():
    with pytest.raises(ValueError):
        checkpoint_identity({})
    with pytest.raises(ValueError):
        checkpoint_identity({'environment': 'OtherRobot'})


def test_reordered_actions_and_changed_normalization_are_incompatible():
    base = checkpoint_identity({'environment': 'Humanoid-v5'})
    for key in ('action_schema', 'observation_schema', 'normalization_schema', 'model_revision', 'control_dt'):
        with pytest.raises(ValueError, match=key):
            validate_resume(base, {**base, key: 'different'})


def test_explicit_schema_cannot_silently_fall_back_to_legacy():
    with pytest.raises(ValueError):
        checkpoint_identity({'schema_version': 999, 'environment': 'Humanoid-v5'})
    with pytest.raises(ValueError):
        checkpoint_identity({'schema_version': 1, 'environment': 'Humanoid-v5'})


def test_loader_rejects_wrong_robot_before_loading_weights(tmp_path):
    from humanoid_sim.core import load_stage, write_json
    identity = checkpoint_identity({'environment': 'Humanoid-v5'})
    write_json(tmp_path / 'metadata.json', {**identity, 'robot_id': 'g1_29dof'})
    with pytest.raises(ValueError, match='robot_id'):
        load_stage(tmp_path)


def test_branch_rejects_unknown_source_without_creating_experiment(tmp_path):
    from humanoid_sim.core import write_json
    from humanoid_sim.lab import Laboratory
    original = tmp_path / 'original'
    stage = original / 'stage_000000000001'
    stage.mkdir(parents=True)
    (stage / 'policy.zip').touch()
    (stage / 'normalize.pkl').touch()
    write_json(stage / 'metadata.json', {'environment': 'Unknown'})
    lab = Laboratory(original)
    with pytest.raises(ValueError, match='Unknown'):
        lab.create({'source_id': 'baseline'})
    assert not list(lab.root.glob('exp-*'))
