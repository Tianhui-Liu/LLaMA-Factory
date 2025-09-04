from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX

from .processor_utils import DatasetProcessor, infer_seqlen

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import Template


logger = logging.get_logger(__name__)

# This function is modified from 

# This function is modified from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/processors/supervised.py

class COTRAFTDatasetProcessor(DatasetProcessor):
      def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, original_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            # TODO: I think the sequence is not yet padded here. Verify this

            # Step 1: Use the attention mask to calculate the sequence length
            # logger.info(f"\n\nSequence length: {seq_len}")
            # logger.info(f"The input ids correspond to the following tokens: {tokenizer.convert_ids_to_tokens(input_ids)}\n\n")
            # Step 2: Construct another score_label

            # The following is customized for Mistral-7B-Instruct-v0.2
            # indices_to_scores = {
            #     28740: 1.0,
            #     28750: 2.0,
            #     28770: 3.0,
            #     28781: 4.0,
            #     28782: 5.0,
            # }

            # The following is customized for LLama-3.1-8B-Instruct and Qwen-2.5-VL-7B
             indices_to_scores = {
                 15: 0,
                 16: 1,
                 17: 2,
                 18: 3,
                 19: 4,
                 20: 5,
                 21: 6,
                 22: 7,
                 23: 8,
                 24: 9,
             }
            
            possible_scores = tokenizer.convert_ids_to_tokens(list(indices_to_scores.keys()))
            possible_scores = [int(score) for score in possible_scores]
            for idx, score in enumerate(possible_scores):
                # 如果从0开始，idx；如果从1开始，idx+1
                if score != idx:
                    raise ValueError(f"Indices and scores do not match: {possible_scores} and {list(indices_to_scores.keys())}")
            score_label = indices_to_scores[original_labels[-2]]

            # Step 3: Mask out the second-to-last token in the input_ids. This token corresponds to the score token, whose loss will be calculated using raft
            lm_loss_labels = original_labels.copy()
            lm_loss_labels[-2] = IGNORE_INDEX

            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(lm_loss_labels)
            model_inputs["score_labels"].append(score_label)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        valid_score_labels = list(filter(lambda x: x != IGNORE_INDEX, example["score_labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
        print("score_label_ids:\n{}".format(example["score_labels"]))
        print(f"score_labels:\n{self.tokenizer.decode(valid_score_labels, skip_special_tokens=False)}")
        
        return model_inputs