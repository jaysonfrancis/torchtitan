# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp

from generation import generate
from torchtitan import utils

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config


@dataclass
class ModelArgs:
    config_path: str
    checkpoint_path: str
    device: str = "cuda"


@dataclass
class GenerationArgs:
    prompt: str = "Hello! How are you?"
    temperature: float = 1.0
    max_generated_tokens: int = 32
    top_k: Optional[int] = None


def example_generate(args: ModelArgs, gen_args: GenerationArgs):
    init_logger()
    color = utils.Color

    # Read toml file
    config = JobConfig()
    config.parse_args(args_list=[f"--job.config_file={args.config_path}"])
    config._validate_config()

    # Load tokenizer
    tokenizer_type = model_name_to_tokenizer[config.model.name]
    tokenizer = build_tokenizer(tokenizer_type, config.model.tokenizer_path)

    # Load model
    model_cls = model_name_to_cls[config.model.name]
    model_config = models_config[config.model.name][config.model.flavor]
    model_config.vocab_size = tokenizer.n_words
    with torch.device(args.device):
        model = model_cls.from_model_args(model_config)
    state_dict = model.state_dict()

    precompute = False
    if "freqs_cis" in state_dict:
        del state_dict["freqs_cis"]
        precompute = True

    # Load checkpoint
    logger.info(f"Loading checkpoint at : {args.checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=args.checkpoint_path)

    if precompute:
        model.freqs_cis = model._precompute_freqs_cis().to(args.device)

    # Encode & generate
    input_ids = tokenizer.encode(gen_args.prompt, bos=False, eos=False)
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(args.device)
    logger.info(f"{color.red}Input tokens: {len(input_ids)}{color.reset}")

    responses, logits = generate(
        model,
        input_ids,
        temperature=gen_args.temperature,
        max_generated_tokens=gen_args.max_generated_tokens,
        top_k=gen_args.top_k,
    )

    prompt_length = len(input_ids)

    logger.info(
        f"{color.blue}Output tokens: {len(responses[0]) - prompt_length}{color.reset}"
    )

    # Single prompt generation
    response = ""
    for token in responses[0][prompt_length:]:
        response += tokenizer.decode([token.item()])
    logger.info(f"{color.red}{gen_args.prompt}{color.blue}{response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")

    # Group for model-related arguments
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    model_group.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on. Options: 'cpu', 'cuda'. Default is 'cuda'.",
    )

    # Group for generation-related arguments
    generation_group = parser.add_argument_group("Generation Options")
    generation_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (range 0-1). Default is 1.0.",
    )
    generation_group.add_argument(
        "--max_generated_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate. Default is 32.",
    )
    generation_group.add_argument(
        "--top_k", type=int, help="Prune to select from top_k probabilities. (Optional)"
    )
    generation_group.add_argument(
        "--prompt",
        type=str,
        default="Hello! How are you?",
        help="Input prompt for generation.",
    )

    args = parser.parse_args()

    model_args = ModelArgs(
        config_path=args.config, checkpoint_path=args.checkpoint, device=args.device
    )
    generation_args = GenerationArgs(
        prompt=args.prompt,
        temperature=args.temperature,
        max_generated_tokens=args.max_generated_tokens,
        top_k=args.top_k,
    )

    example_generate(args=model_args, gen_args=generation_args)
