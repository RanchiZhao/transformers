# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for CPMAnt."""
import collections
import os
import re
from typing import List, Optional, Tuple, Dict, Union

from ...tokenization_utils import PreTrainedTokenizer, _insert_one_token_to_ordered_list
from ...utils import logging
from ...tokenization_utils_base import TextInput, AddedToken


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openbmb/cpm-bee-10b": "https://huggingface.co/openbmb/cpm-bee-10b/blob/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openbmb/cpm-bee-10b": 1024,
}


class CpmBeeTokenizer(PreTrainedTokenizer):
    """
    Construct a CPMAnt tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bod_token (`str`, *optional*, defaults to `"<d>"`):
            The beginning of document token.
        eod_token (`str`, *optional*, defaults to `"</d>"`):
            The end of document token.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        line_token (`str`, *optional*, defaults to `"</n>"`):
            The line token.
        space_token (`str`, *optional*, defaults to `"</_>"`):
            The space token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    add_prefix_space = False

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        line_token="\n",
        space_token=" ",
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>",
        padding_side="left",
        **kwargs,
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            line_token=line_token,
            space_token=space_token,
            unk_token=unk_token,
            mask_token=mask_token,
            pad_token=pad_token,
            padding_side=padding_side,
            **kwargs,
        )

        self.encoder: Dict[str, int] = {}

        with open(vocab_file, "r", encoding="utf-8") as reader:
            for token in reader.readlines():
                token = token.rstrip("\n")
                if len(token) == 0:
                    continue
                self.encoder[token] = len(self.encoder)

        self.encoder[" "] = self.encoder["</_>"]
        self.encoder["\n"] = self.encoder["</n>"]
        del self.encoder["</_>"]
        del self.encoder["</n>"]

        self.decoder = {v: k for k, v in self.encoder.items()}

        self._max_word_len = max([len(x) for x in self.encoder.keys()])

        self.cpm_bee_special_tok = self.added_tokens_encoder.copy()

    @property
    def bod_token_id(self):
        return self.encoder[self.bod_token]

    @property
    def eod_token_id(self):
        return self.encoder[self.eod_token]

    @property
    def newline_id(self):
        return self.encoder[self.line_token]

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)
    
    @staticmethod
    def escape(text: str) -> str:
        return text.replace("<", "<<")

    @staticmethod
    def unescape(text: str) -> str:
        return text.replace("<<", "<")

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder) - len(self.cpm_bee_special_tok)

    def get_vocab(self):
        new = self.added_tokens_encoder.copy()
        for k in self.cpm_bee_special_tok:
            del new[k]
        return dict(self.encoder, **new)
    
    def get_piece(self, text: str) -> str:
        text = text[: self._max_word_len]
        len_text = len(text)
        for i in range(len(text)):
            sub = text[: len_text - i]
            if ((sub in self.encoder) or (sub in self.added_tokens_encoder)) and (sub not in self.all_special_tokens):
                return sub
        return text[0]
    
    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.
        This method overrides the method:`tokenize` in `PreTrainedTokenizer` for CPMBee.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        # add special token
        new_text = text
        for sp_token in self.all_special_tokens:
            new_text = new_text.replace(sp_token, "")

        sentence_split = [""]
        is_escape = False
        is_special_token = False
        for i, c in enumerate(new_text):  # "ahdsfjksssss<mask>fsg" -> ["ahdsfjksssss", "<mask>", "fsg"]
            if is_special_token:
                if c == "<":
                    raise ValueError("Invalid special token at pos {}".format(i))
                elif c == ">":
                    # end of special token
                    sentence_split[-1] += c
                    is_special_token = False
                    cur_token = sentence_split[-1]
                    if cur_token not in self.all_special_tokens:
                        self.add_tokens(cur_token, special_tokens=False)
                else:
                    sentence_split[-1] += c
            else:
                if c == "<":
                    if is_escape:
                        # case: <<
                        sentence_split[-1] += c
                        is_escape = False
                    else:
                        # case: x<
                        is_escape = True
                else:
                    if is_escape:
                        # case <x
                        is_special_token = True
                        is_escape = False
                        sentence_split.append("<" + c)
                    else:
                        # case xx
                        sentence_split[-1] += c
        del sentence_split
        if is_escape or is_special_token:
            raise ValueError("Unexpected end of text `{}`".format(text))
        
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = {
            str(t): t for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        }

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                elif token not in self.cpm_bee_special_tok:
                # else:
                    # We strip left and right by default
                    if right:
                        tokens[i + 1] = right.lstrip()
                    if left:
                        tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        print(tokens)
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        output_tokens = []
        part_pos = 0

        if text in self.all_special_tokens:
            # special token
            output_tokens.append(text)
        else:
            part_st = 0
            last_unk = None
            while part_st < len(text):
                piece = self.get_piece(text[part_st:])
                if piece in self.encoder or piece in self.added_tokens_encoder:
                    if last_unk is None:
                        output_tokens.append(piece)
                    else:
                        if last_unk not in self.added_tokens_encoder:
                            print("CHECK: ", last_unk, piece)
                            self._add_tokens([last_unk], special_tokens=True, for_cpmbee=True)
                            self.cpm_bee_special_tok[last_unk] = self.added_tokens_encoder[last_unk]
                        output_tokens.append(last_unk)
                        output_tokens.append(piece)
                        last_unk = None
                else:
                    if last_unk is None:
                        last_unk = piece
                    else:
                        last_unk += piece
                part_st += len(piece)
            
            if last_unk is not None:
                # part end with UNK
                if last_unk not in self.added_tokens_encoder:
                    self.add_tokens(last_unk, special_tokens=True)
                    self.cpm_bee_special_tok[last_unk] = self.added_tokens_encoder[last_unk]
                output_tokens.append(last_unk)
        part_pos += len(text)
        return output_tokens

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False, for_cpmbee: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is only added if it's not already in the vocabulary (tested by
                checking if the tokenizer assign the index of the `unk_token` to them).
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            if not isinstance(token, str):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                if self.verbose:
                    logger.info(f"Adding {token} to the vocabulary")

        added_tok_encoder = {tok: len(self) + i for i, tok in enumerate(tokens_to_add)}
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
        # if not for_cpmbee:
        if special_tokens:
            if len(new_tokens) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, new_tokens[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            # Or on the newly added tokens
            if len(tokens_to_add) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, tokens_to_add[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))
        self._create_trie(self.unique_no_split_tokens)

        return len(tokens_to_add)

    def check(self, token):
        return token in self.encoder

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def _convert_token_to_id(self, token: str):
        """Converts a token (str) in an id using the vocab."""
        if token in self.encoder:
            return self.encoder.get(token)
        elif token in self.added_tokens_encoder:
            return self.added_tokens_encoder.get(token)
        else:
            return self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index]
        else:
            if index >= 0:
                return self.decoder[index]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        index = 0
        self.encoder["</n>"] = self.encoder["\n"]
        del self.encoder["\n"]
        self.encoder["</_>"] = self.encoder[" "]
        del self.encoder[" "]
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for (token, token_index) in sorted(self.encoder.items(), key=lambda x: x[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
    
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        for k in self.cpm_bee_special_tok:
            del self.added_tokens_encoder[k]
            del self.added_tokens_decoder[self.cpm_bee_special_tok[k]]
        ret = super().save_pretrained(
            save_directory,
            legacy_format,
            filename_prefix,
            push_to_hub,
            **kwargs
        )
        for k, v in self.cpm_bee_special_tok.items():
            self.added_tokens_encoder[k] = v
            self.added_tokens_decoder[v] = k
        return ret

    # def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
    #     """
    #     Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
    #     adding special tokens. A CPMAnt sequence has the following format:

    #     - single sequence: `[BOS] Sequence`.

    #     Args:
    #         token_ids_0 (`List[int]`): The first tokenized sequence that special tokens will be added.
    #         token_ids_1 (`List[int]`): The optional second tokenized sequence that special tokens will be added.

    #     Returns:
    #         `List[int]`: The model input with special tokens.
    #     """
    #     if token_ids_1 is None:
    #         return [self.bos_token_id] + token_ids_0
    #     return [self.bos_token_id] + token_ids_0 + [self.bos_token_id] + token_ids_1

    # def get_special_tokens_mask(
    #     self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    # ) -> List[int]:
    #     """
    #     Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
    #     special tokens using the tokenizer `prepare_for_model` method.

    #     Args:
    #         token_ids_0 (`List[int]`): List of IDs.
    #         token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.
    #         already_has_special_tokens (`bool`, *optional*, defaults to `False`):
    #             Whether or not the token list is already formatted with special tokens for the model.

    #     Returns:
    #         `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
    #     """

    #     if already_has_special_tokens:
    #         return super().get_special_tokens_mask(
    #             token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
    #         )

    #     if token_ids_1 is not None:
    #         return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
    #     return [1] + ([0] * len(token_ids_0))
