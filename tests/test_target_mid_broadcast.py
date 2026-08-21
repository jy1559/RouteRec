from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import routerec.session_data as MOD


class FakeInteraction:
    def __init__(self, interaction: dict[str, torch.Tensor]) -> None:
        self.interaction = interaction
        self.columns = list(interaction)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.interaction[key]
        return FakeInteraction({name: value[key] for name, value in self.interaction.items()})

    def __len__(self) -> int:
        return len(next(iter(self.interaction.values())))


class DatasetLike:
    uid_field = "session_id"
    iid_field = "item_id"
    time_field = "timestamp"

    def __init__(self, *, training_contract: bool = True, model: str = "RouteRec") -> None:
        self.config = {
            "model": model,
            "MAX_ITEM_LIST_LENGTH": 4,
            "LIST_SUFFIX": "_list",
            "ITEM_LIST_LENGTH_FIELD": "item_length",
            "routerec_feature_fp16": False,
            "routerec_causal_mid_target_broadcast": training_contract,
            "sequence_convert_chunk_size": 1024,
        }


@pytest.fixture(autouse=True)
def fake_recbole(monkeypatch):
    recbole = types.ModuleType("recbole")
    data = types.ModuleType("recbole.data")
    interaction = types.ModuleType("recbole.data.interaction")
    interaction.Interaction = FakeInteraction
    monkeypatch.setitem(sys.modules, "recbole", recbole)
    monkeypatch.setitem(sys.modules, "recbole.data", data)
    monkeypatch.setitem(sys.modules, "recbole.data.interaction", interaction)


def raw_interaction() -> FakeInteraction:
    return FakeInteraction(
        {
            "session_id": torch.tensor([1, 1, 1, 1]),
            "item_id": torch.tensor([10, 11, 12, 13]),
            "timestamp": torch.tensor([0.0, 1.0, 2.0, 3.0]),
            # Row j is the strict-prefix mid cue for target j.
            "mid_valid_r": torch.tensor([0.0, 10.0, 20.0, 30.0]),
            # Macro is constant; micro remains event-local/position-wise.
            "mac5_ctx_valid_r": torch.tensor([5.0, 5.0, 5.0, 5.0]),
            "mic_valid_r": torch.tensor([100.0, 101.0, 102.0, 103.0]),
        }
    )


def test_training_targets_broadcast_target_mid_and_preserve_macro_micro() -> None:
    converted = MOD._convert_interactions(
        DatasetLike(), raw_interaction(), training=True
    )
    assert converted is not None
    assert converted["item_id"].tolist() == [11, 12, 13]
    assert converted["item_length"].tolist() == [1, 2, 3]
    assert converted["mid_valid_r_list"].tolist() == [
        [10.0, 0.0, 0.0, 0.0],
        [20.0, 20.0, 0.0, 0.0],
        [30.0, 30.0, 30.0, 0.0],
    ]
    assert converted["mac5_ctx_valid_r_list"].tolist() == [
        [5.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 0.0, 0.0],
        [5.0, 5.0, 5.0, 0.0],
    ]
    assert converted["mic_valid_r_list"].tolist() == [
        [100.0, 0.0, 0.0, 0.0],
        [100.0, 101.0, 0.0, 0.0],
        [100.0, 101.0, 102.0, 0.0],
    ]


def test_validation_last_target_uses_same_causal_alignment_and_zero_padding() -> None:
    converted = MOD._convert_interactions(
        DatasetLike(), raw_interaction(), training=False
    )
    assert converted is not None
    assert converted["item_id"].tolist() == [13]
    assert converted["item_length"].tolist() == [3]
    assert converted["mid_valid_r_list"].tolist() == [[30.0, 30.0, 30.0, 0.0]]
    assert converted["mic_valid_r_list"].tolist() == [[100.0, 101.0, 102.0, 0.0]]


def test_flag_off_preserves_per_position_mid_history() -> None:
    converted = MOD._convert_interactions(
        DatasetLike(training_contract=False), raw_interaction(), training=False
    )
    assert converted is not None
    assert converted["mid_valid_r_list"].tolist() == [[0.0, 10.0, 20.0, 0.0]]


def test_history_profile_changes_only_for_opted_in_routerec() -> None:
    standard_route = DatasetLike(training_contract=False).config
    aligned_route = DatasetLike(training_contract=True).config
    baseline_off = DatasetLike(training_contract=False, model="SASRec").config
    baseline_on = DatasetLike(training_contract=True, model="SASRec").config
    assert MOD._history_profile(standard_route) == "routerec_route_features"
    assert MOD._history_profile(aligned_route).endswith("target_mid_broadcast_v1")
    assert MOD._history_profile(baseline_off) == MOD._history_profile(baseline_on)
    assert MOD._history_profile(baseline_on) == "baseline_item_only"


def test_cache_identity_versions_and_separates_target_mid_semantics(tmp_path: Path) -> None:
    dataset = write_contract_fixture(tmp_path)
    assert MOD._PATCH_VERSION == 6
    enabled_key = MOD._cache_key(dataset, ["train", "valid", "test"])
    dataset.config["routerec_causal_mid_target_broadcast"] = False
    standard_key = MOD._cache_key(dataset, ["train", "valid", "test"])
    assert enabled_key != standard_key


def write_contract_fixture(root: Path, *, mid_scope: str = "strict_prefix") -> DatasetLike:
    dataset = DatasetLike(training_contract=True)
    dataset.dataset_name = "toy"
    dataset.config.update({"data_path": str(root), "dataset": "toy"})
    directory = root / "toy"
    directory.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (directory / f"toy.{split}.inter").write_text("header\n", encoding="utf-8")
    (directory / "feature_metadata.json").write_text(
        json.dumps(
            {
                "reconstruction_contract": "core5-features-leakage-safe-v1",
                "mid_scope": mid_scope,
                "all_features": [f"mid_{index}" for index in range(16)],
            }
        ),
        encoding="utf-8",
    )
    return dataset


def test_contract_guard_accepts_only_leakage_safe_strict_prefix_metadata(tmp_path: Path) -> None:
    valid = write_contract_fixture(tmp_path / "valid")
    MOD._validate_target_mid_broadcast_contract(valid, ["train", "valid", "test"])

    invalid = write_contract_fixture(tmp_path / "invalid", mid_scope="session_constant_last")
    with pytest.raises(ValueError, match="strict_prefix"):
        MOD._validate_target_mid_broadcast_contract(invalid, ["train", "valid", "test"])


def test_contract_guard_rejects_flag_on_baseline(tmp_path: Path) -> None:
    dataset = DatasetLike(training_contract=True, model="SASRec")
    with pytest.raises(ValueError, match="RouteRec-only"):
        MOD._validate_target_mid_broadcast_contract(dataset, ["train", "valid", "test"])
