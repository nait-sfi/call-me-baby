from typing import Any, cast

from llm_sdk import Small_LLM_Model


def get_prompt_prefix(
    functions: list[dict[str, Any]],
    model: Small_LLM_Model,
) -> list[int]:
    """Pre-compute the teaching prompt prefix.

    Args:
        functions: Function definitions used to seed the prompt.
        model: Tokenizer/model used to encode the prompt text.

    Returns:
        The token IDs for the static teaching prefix and encoded functions.
    """
    first = [
        198,
        2610,
        525,
        264,
        729,
        1786,
        16740,
        17847,
        382,
        7771,
        2618,
        374,
        311,
        23643,
        279,
        1196,
        594,
        1681,
        323,
        8253,
        3425,
        825,
        315,
        279,
        2500,
        5746,
        1265,
        387,
        2598,
        382,
        16485,
        5746,
        510,
    ]
    second = [
        198,
        91916,
        50,
        198,
        262,
        2677,
        421,
        279,
        5733,
        943,
        374,
        1372,
        68424,
        432,
        311,
        2224,
        271,
        13314,
        271,
        1474,
        510,
        3838,
        594,
        279,
        2629,
        315,
        220,
        23,
        323,
        220,
        24,
        1939,
        5097,
        510,
        515,
        220,
        330,
        40581,
        788,
        330,
        3838,
        594,
        279,
        2629,
        315,
        220,
        23,
        323,
        220,
        24,
        35718,
        220,
        330,
        606,
        788,
        330,
        8822,
        2891,
        32964,
        756,
        220,
        330,
        13786,
        788,
        341,
        262,
        330,
        64,
        788,
        220,
        23,
        13,
        15,
        345,
        262,
        330,
        65,
        788,
        220,
        24,
        13,
        15,
        198,
        220,
        456,
        630,
        7039,
        1882,
        279,
        2701,
        1681,
        624,
        1474,
        510,
    ]
    func_ids = cast(list[int], model.encode(f"{functions}").tolist()[0])
    return first + func_ids + second


def build_full_prompt(
    prefix: list[int],
    user_prompt: str,
    model: Small_LLM_Model,
) -> list[int]:
    """Build the complete prompt using the cached prefix.

    Args:
        prefix: Precomputed static prefix token IDs.
        user_prompt: The user prompt to append.
        model: Tokenizer/model used to encode the prompt text.

    Returns:
        The full prompt as token IDs.
    """
    end = [198, 5370, 510]
    prompt_ids = cast(list[int], model.encode(user_prompt).tolist()[0])
    return prefix + prompt_ids + end
