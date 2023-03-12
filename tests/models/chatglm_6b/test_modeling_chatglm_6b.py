# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the PyTorch ChatGLM-6B model. """


import unittest

from ...test_modeling_common import floats_tensor
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from transformers import ChatGLM6BConfig, ChatGLM6BTokenizer
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        ChatGLM6BForConditionalGeneration,
        ChatGLM6BModel,
    )
    from transformers.models.chatglm_6b.modeling_chatglm_6b import (
        CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class ChatGLM6BModelTester:
    def __init__(
            self,
            parent,
            batch_size=2,
            seq_length=7,
            is_training=True,
            use_input_mask=True,
            use_token_type_ids=True,
            use_labels=True,
            vocab_size=150528,
            hidden_size=4096,
            num_hidden_layers=5,
            num_attention_heads=32,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=2048,
            type_vocab_size=16,
            type_sequence_label_size=2,
            initializer_range=0.02,
            num_labels=3,
            num_choices=4,
            scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = {'input_ids': torch.tensor([[ 20005,  92306, 150001, 150004], [ 20005,  92306, 150001, 150004]])}
        config = self.get_config()
        return config, input_ids

    def get_config(self):
        return ChatGLM6BConfig()

    def create_and_check_model(
            self, config, input_ids
    ):
        model = ChatGLM6BModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_conditional_generation(
            self, config, input_ids
    ):
        model = ChatGLM6BForConditionalGeneration(config=config)
        input_ids = input_ids.to('cuda')
        outputs = model.generate(**input_ids, max_length=512, num_beams=2)

        self.parent.assertEqual(outputs.dim(), 2)


@require_torch
class ChatGLM6BModelTest(unittest.TestCase):

    all_model_classes = (
        (
            ChatGLM6BModel,
            ChatGLM6BForConditionalGeneration
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (ChatGLM6BForConditionalGeneration,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = ChatGLM6BModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ChatGLM6BConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ChatGLM6BModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class ChatGLM6BModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = ChatGLM6BForConditionalGeneration.from_pretrained("THUDM/ChatGLM-6B")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        # TODO Replace vocab size
        vocab_size = 150528

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        # TODO Replace values below with what was printed above.
        # expected_slice = torch.tensor(
        #     [[[-0.0483, 0.1188, -0.0313], [-0.0606, 0.1435, 0.0199], [-0.0235, 0.1519, 0.0175]]]
        # )

        # self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


