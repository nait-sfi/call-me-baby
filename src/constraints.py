from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


def get_logits(
    logits: Sequence[float] | NDArray[np.float32],
    allowed_ids: Iterable[int],
) -> NDArray[np.float32]:
    """Mask logits so only allowed token IDs can be selected."""
    logits_array = np.asarray(logits, dtype=np.float32)
    logits_copy = np.full(logits_array.shape[0], -np.inf, dtype=np.float32)
    idx = list(allowed_ids)
    logits_copy[idx] = logits_array[idx]
    return logits_copy


def constrained_function_name(
    function_name_paths: dict[str, list[int]],
    gen_ids: list[int],
) -> set[int]:
    """Return allowed next-token IDs for function-name decoding."""
    nb_tk = len(gen_ids)
    return {
        ids[nb_tk]
        for ids in function_name_paths.values()
        if len(ids) > nb_tk and gen_ids == ids[:nb_tk]
    }


def constrained_number_value(
    gen_ids: list[int],
    digit_tokens: set[int],
    minus_token: int,
    dot_token: int,
    comma_token: int,
    close_brace_token: int,
) -> set[int]:
    """Return allowed next-token IDs while writing a numeric value."""
    has_digit = any(t in digit_tokens for t in gen_ids)
    allowed = set(digit_tokens)
    if not gen_ids:
        allowed.add(minus_token)
    if dot_token not in gen_ids:
        allowed.add(dot_token)
    if has_digit:
        allowed.add(comma_token)
        allowed.add(close_brace_token)
    return allowed


def constrained_decoding(
    logits: Sequence[float] | NDArray[np.float32],
    state: int,
    functions_name_paths: dict[str, list[int]],
    gen_ids: list[int],
    digit_tokens: set[int],
    minus_token: int,
    dot_token: int,
    comma_token: int,
    close_brace_token: int,
) -> NDArray[np.float32] | None:
    """Apply constrained decoding according to the generation state."""
    if state == 1:
        allowed_ids = constrained_function_name(functions_name_paths, gen_ids)
        if not allowed_ids:
            return np.asarray(logits, dtype=np.float32)
        return get_logits(logits, allowed_ids)

    if state == 3:
        allowed_ids = constrained_number_value(
            gen_ids, digit_tokens, minus_token, dot_token,
            comma_token, close_brace_token,
        )
        return get_logits(logits, allowed_ids)

    return None
