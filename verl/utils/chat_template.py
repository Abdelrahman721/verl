# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    t1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )

    # 2) user -> assistant (valid alternation)
    t2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )

    # 3) user -> assistant -> user (valid alternation)
    t3 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )

    # token cost of adding one (empty) user turn at the end
    user_turn_len = len(t3) - len(t2)

    # prefix/system prompt tokens are the part before the first user turn
    system_prompt = t1[: -user_turn_len] if user_turn_len > 0 else []
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    t1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )

    t2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )

    t3 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **apply_chat_template_kwargs,
    )

    user_turn_len = len(t3) - len(t2)
    system_prompt = t1[: -user_turn_len] if user_turn_len > 0 else []

    # generation prompt suffix (this part was already fine)
    t_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=True,
        **apply_chat_template_kwargs,
    )
    generate_prompt = t_gen[len(t1):]

    return system_prompt, generate_prompt
