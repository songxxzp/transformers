# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from ...utils import  _LazyModule, OptionalDependencyNotAvailable, is_tokenizers_available
from ...utils import is_torch_available




_import_structure = {
    "configuration_chatglm_6b": ["CHATGLM_6B_PRETRAINED_CONFIG_ARCHIVE_MAP", "ChatGLM6BConfig"],
    "tokenization_chatglm_6b": ["ChatGLM6BTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_chatglm_6b_fast"] = ["ChatGLM6BTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_chatglm_6b"] = [
        "CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ChatGLM6BForMaskedLM",
        "ChatGLM6BForCausalLM",
        "ChatGLM6BForMultipleChoice",
        "ChatGLM6BForQuestionAnswering",
        "ChatGLM6BForSequenceClassification",
        "ChatGLM6BForTokenClassification",
        "ChatGLM6BLayer",
        "ChatGLM6BModel",
        "ChatGLM6BPreTrainedModel",
        "load_tf_weights_in_chatglm_6b",
    ]




if TYPE_CHECKING:
    from .configuration_chatglm_6b import CHATGLM_6B_PRETRAINED_CONFIG_ARCHIVE_MAP, ChatGLM6BConfig
    from .tokenization_chatglm_6b import ChatGLM6BTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_chatglm_6b_fast import ChatGLM6BTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_chatglm_6b import (
            CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST,
            ChatGLM6BForMaskedLM,
            ChatGLM6BForCausalLM,
            ChatGLM6BForMultipleChoice,
            ChatGLM6BForQuestionAnswering,
            ChatGLM6BForSequenceClassification,
            ChatGLM6BForTokenClassification,
            ChatGLM6BLayer,
            ChatGLM6BModel,
            ChatGLM6BPreTrainedModel,
            load_tf_weights_in_chatglm_6b,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
