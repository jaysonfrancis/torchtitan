from typing import Callable, List, Optional, Tuple

import torch


def multinomial_sample_one(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Sample from a multinomial distribution

    q: noise sampled from an exponential distribution used to perturb the `probs`

    The Gumbel-Softmax trick provides randomness which enables differentiable sampling and
    stochasticity.

    """
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample from a probability distribution

    Args:
        logits (torch.Tensor): logits from which to sample (vocab_size,)
        temperature (float): value to scale logits by, default 1.0.
        top_k (Optional[int]): if specified, prune sampling to only tokens within the top_k probs.
        q (Optional[torch.Tensor]): randomly sampled tensor for softmax sampling. If None, we use
            default softmax sampling from an exponential.

    Returns:
        torch.Tensor: sampled token id
    """

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))  # (k,)
        # select last value from top_k above as the pivot
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)  # (1,)
        # mask values smaller than pivot to -inf since these should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)  # (vocab_size, )

    # convert to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    if q is None:
        q = torch.empty_like(probs).exponential_(lambd=1)

    return multinomial_sample_one(probs, q)


def generate_next_token(
    model,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # take the last token's logits as the input to the next model call.
    logits = model(x)  # (B, T, vocab_size)
    return (
        sample(logits[0, -1, :].clone(), temperature=temperature, top_k=top_k, q=q),
        logits,
    )


@torch.inference_mode()
def generate(
    model,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    custom_generate_next_token: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """

    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    B, T = prompt.size()

    generated_tokens = prompt.clone()

    # from model_config
    vocab_size = 128256  # 34107 # 256

    q = torch.empty((B, vocab_size), device=prompt.device).exponential_(
        1, generator=rng
    )

    tokens, generated_logits = generate_next_token(
        model,
        x=prompt,
        temperature=temperature,
        top_k=top_k,
        q=q,
    )

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    for _ in range(max_generated_tokens - 1):
        tokens = generated_tokens.clone()

        q = torch.empty((B, vocab_size), device=prompt.device).exponential_(
            1, generator=rng
        )

        tokens, logits = custom_generate_next_token(
            model,
            x=tokens.clone(),
            temperature=temperature,
            top_k=top_k,
            q=q,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        generated_logits = logits

    return generated_tokens, generated_logits
