from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FDSA_PATH = ROOT / "src" / "routerec" / "models" / "fdsa.py"


class _SequentialRecommenderStub(nn.Module):
    """Minimal RecBole contract needed to exercise the real FDSA constructor."""

    def __init__(self, config, dataset):
        super().__init__()
        self.n_items = int(dataset.item_num)
        self.max_seq_length = int(config["MAX_ITEM_LIST_LENGTH"])
        self.ITEM_SEQ = "item_id_list"
        self.ITEM_SEQ_LEN = "item_length"
        self.POS_ITEM_ID = "item_id"
        self.NEG_ITEM_ID = "neg_item_id"
        self.ITEM_ID = "item_id"

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return {}

    def load_other_parameter(self, parameters):
        if parameters is None:
            return
        for key, value in parameters.items():
            setattr(self, key, value)


class _FeatureSeqEmbLayerStub(nn.Module):
    def __init__(self, dataset, embedding_size, selected_features, pooling_mode, device):
        super().__init__()
        self.embedding = nn.Embedding(7, int(embedding_size))


class _TransformerEncoderStub(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = int(kwargs["hidden_size"])
        self.projection = nn.Linear(hidden_size, hidden_size)


class _VanillaAttentionStub(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super().__init__()
        self.projection = nn.Linear(int(hidden_size), int(attention_size))


class _BPRLossStub(nn.Module):
    pass


def _recbole_stubs() -> dict[str, ModuleType]:
    modules = {
        name: ModuleType(name)
        for name in (
            "recbole",
            "recbole.model",
            "recbole.model.abstract_recommender",
            "recbole.model.layers",
            "recbole.model.loss",
        )
    }
    modules["recbole.model.abstract_recommender"].SequentialRecommender = (
        _SequentialRecommenderStub
    )
    modules["recbole.model.layers"].FeatureSeqEmbLayer = _FeatureSeqEmbLayerStub
    modules["recbole.model.layers"].TransformerEncoder = _TransformerEncoderStub
    modules["recbole.model.layers"].VanillaAttention = _VanillaAttentionStub
    modules["recbole.model.loss"].BPRLoss = _BPRLossStub
    return modules


def _load_fdsa_with_stubs():
    module_name = "_routerec_fdsa_checkpoint_surface_test"
    spec = importlib.util.spec_from_file_location(module_name, FDSA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {FDSA_PATH}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, _recbole_stubs()):
        spec.loader.exec_module(module)
    return module.FDSA


class FDSACheckpointSurfaceTest(unittest.TestCase):
    def _build_model(self):
        fdsa_cls = _load_fdsa_with_stubs()
        config = {
            "MAX_ITEM_LIST_LENGTH": 5,
            "n_layers": 1,
            "n_heads": 1,
            "hidden_size": 4,
            "inner_size": 8,
            "hidden_dropout_prob": 0.0,
            "attn_dropout_prob": 0.0,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "selected_features": ["category"],
            "pooling_mode": "mean",
            "device": "cpu",
            "initializer_range": 0.02,
            "loss_type": "CE",
        }

        class DatasetStub:
            item_num = 11

        return fdsa_cls(config, DatasetStub())

    def test_feature_module_is_state_dict_only_and_round_trips_strictly(self) -> None:
        torch.manual_seed(7)
        model = self._build_model()

        self.assertEqual(model.other_parameter_name, [])
        self.assertEqual(model.other_parameter(), {})
        feature_keys = [
            key for key in model.state_dict() if key.startswith("feature_embed_layer.")
        ]
        self.assertTrue(feature_keys)

        buffer = io.BytesIO()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "other_parameter": model.other_parameter(),
            },
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
        restored = self._build_model()
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        restored.load_other_parameter(checkpoint["other_parameter"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_old_extra_parameter_remains_loadable(self) -> None:
        model = self._build_model()
        current_feature_layer = model.feature_embed_layer
        legacy_feature_layer = _FeatureSeqEmbLayerStub(None, 4, [], "mean", "cpu")
        model.load_other_parameter({"feature_embed_layer": legacy_feature_layer})
        # State-dict parameters are loaded separately.  Do not replace the
        # current data lookup with a stale, pickled dataset from the old run.
        self.assertIs(model.feature_embed_layer, current_feature_layer)
        self.assertIsNot(model.feature_embed_layer, legacy_feature_layer)
        # Loading an old checkpoint does not make future checkpoints duplicate it.
        self.assertEqual(model.other_parameter(), {})


if __name__ == "__main__":
    unittest.main()
