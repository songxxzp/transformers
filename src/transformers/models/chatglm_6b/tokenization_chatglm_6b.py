# coding=utf-8
# Copyright 2022 THUDM and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ChatGLM-6B."""
import sys
import unicodedata
from typing import List, Optional, Union
from functools import lru_cache
import os
import collections
import re

from tokenizers import ByteLevelBPETokenizer

from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_punctuation
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .icetk_glm_6b import _IceTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "ice_text.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "THUDM/ChatGLM-6B": "https://cloud.tsinghua.edu.cn/f/2c73ea6d3e7f4aed82ec/?dl=1",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "THUDM/ChatGLM-6B": 2048,
}

class ChatGLM6BTokenizer(PreTrainedTokenizer):
    """
    Construct a ChatGLM-6B tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=False,
        bos_token='sop',
        eos_token='eos',
        eop_token='eop',
        unk_token="<unk>",
        pad_token='[pad]',
        mask_token='[MASK]',
        gMASK_token='[gMASK]',
        padding_side="left",
        **kwargs
    ) -> None:
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            bos_token=bos_token, 
            eos_token=eos_token,
            eop_token=eop_token,
            unk_token=unk_token,
            pad_token=pad_token, 
            mask_token=mask_token,
            gMASK_token=gMASK_token,
            padding_side=padding_side,
            **kwargs
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.vocab_file = vocab_file

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.eop_token = eop_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.gMASK_token = gMASK_token

        self.icetokenizer = _IceTokenizer(vocab_file)

        """ Initialisation """

    @property
    def eop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self.eop_token is None:
            return None
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def vocab_size(self):
        """ Returns vocab size """
        return self.icetokenizer.vocab_size

    def get_vocab(self):
        """ Returns vocab as a dict """
        return self.icetokenizer.vocab

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs

        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text):
        """ Returns a tokenized string. """
        text = self.preprocess_text(text)

        mask_pattern = r"\[g?MASK\]"
        text_list = re.split(mask_pattern, text)
        pattern_list = re.compile(mask_pattern).findall(text)
        seq = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            seq.extend(self.icetokenizer.tokenize(sub_text))
            seq.append(pattern)


        seq.extend(self.icetokenizer.tokenize(text_list[-1]))
        
        return seq

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        if isinstance(token_ids[0], list):
            tokens = []
            for single_token_ids in token_ids:
                if 0 in single_token_ids:  # remove pad
                    single_token_ids = list(filter((0).__ne__, single_token_ids))
                tokens.append(self.icetokenizer.decode(single_token_ids))
            return(tokens)
        else:
            if 0 in token_ids:  # remove pad
                token_ids = list(filter((0).__ne__, token_ids))
            return self.icetokenizer.decode(token_ids)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.icetokenizer.TokenToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.icetokenizer.IdToToken(index)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is not None:
            token_ids_0 += token_ids_1
        mask_ids = self.icetokenizer.get_command(self.mask_token)
        gmask_ids = self.icetokenizer.get_command(self.gMASK_token)
        if mask_ids not in token_ids_0 and gmask_ids not in token_ids_0:
            token_ids_0 += [gmask_ids]

        if token_ids_0[-1] != mask_ids and token_ids_0[-1] != gmask_ids:
            token_ids_0 += [self.icetokenizer.get_command(self.eos_token)]

        token_ids_0 += [self.icetokenizer.get_command(self.bos_token)]

        return token_ids_0
