import unittest
from parameterized import parameterized

import torch
import copy

from transformers import AutoModel
from collaborative_attention import (
    swap_to_collaborative,
    BERTCollaborativeAdapter,
    DistilBERTCollaborativeAdapter,
    ALBERTCollaborativeAdapter,
)


class TestReparametrization(unittest.TestCase):
    @parameterized.expand(
        [
            ["bert-base-cased-finetuned-mrpc", BERTCollaborativeAdapter, 768, 1e-5],
            ["distilbert-base-cased", DistilBERTCollaborativeAdapter, 768, 1e-5],
            ["albert-base-v2", ALBERTCollaborativeAdapter, 768, 5e-2],
            ["bert-base-cased-finetuned-mrpc", BERTCollaborativeAdapter, 2, 1e100],
            ["distilbert-base-cased", DistilBERTCollaborativeAdapter, 2, 1e100],
            ["albert-base-v2", ALBERTCollaborativeAdapter, 2, 1e100],
        ]
    )
    def test_model(
        self, path_or_model_name, AdapterClass, dim_shared_query_key, tolerance
    ):
        original_model = AutoModel.from_pretrained(path_or_model_name)
        collab_model = copy.deepcopy(original_model)
        swap_to_collaborative(
            collab_model, AdapterClass, dim_shared_query_key=dim_shared_query_key
        )

        any_input = torch.LongTensor(3, 25).random_(1000, 10000)
        collab_model.eval()
        out_collab = collab_model(any_input)

        original_model.eval()
        out_original = original_model(any_input)

        diff = (out_collab[0] - out_original[0]).abs().max().item()
        print("Output max difference was {}".format(diff))
        assert diff < tolerance


if __name__ == "__main__":
    unittest.main()
