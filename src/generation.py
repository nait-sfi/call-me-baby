"""Generation helpers for constrained function calling."""

import json
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from llm_sdk import Small_LLM_Model

from .constraints import constrained_decoding
from .prompting import build_full_prompt, get_prompt_prefix


class GenerationMetadata(TypedDict):
    """Cached values used by the generation loop."""

    function_name_paths: dict[str, list[int]]
    teaching_prefix: list[int]
    function_parameters: dict[str, list[tuple[str, str]]]
    digit_tokens: set[int]
    minus_token: int
    dot_token: int
    comma_token: int
    close_brace_token: int
    bool_paths: dict[str, list[int]]


def prepare_model_metadata(
    model: Small_LLM_Model,
    valid_functions: list[dict[str, Any]],
) -> GenerationMetadata:
    """Encode tokens and build lookup structures used during constrained
    decoding.

    Args:
        model: Tokenizer/model used for encoding and decoding.
        valid_functions: Validated function definitions.

    Returns:
        Cached metadata used by the generation loop.
    """
    function_name_paths: dict[str, list[int]] = {
        func["name"]: model.encode(func["name"] + '",').tolist()[0]
        for func in valid_functions
    }
    teaching_prefix = get_prompt_prefix(valid_functions, model)
    function_parameters: dict[str, list[tuple[str, str]]] = {
        function["name"]: [
            (name, param_info.get("type", "string"))
            for name, param_info in function["parameters"].items()
            if name
        ]
        for function in valid_functions
    }

    digit_tokens = {model.encode(str(d)).tolist()[0][0] for d in range(10)}
    minus_token = model.encode("-").tolist()[0][0]
    dot_token = model.encode(".").tolist()[0][0]
    comma_token = model.encode(",").tolist()[0][0]
    close_brace_token = model.encode("}").tolist()[0][0]
    bool_paths = {
        "true": model.encode("true").tolist()[0],
        "false": model.encode("false").tolist()[0],
    }

    return {
        "function_name_paths": function_name_paths,
        "teaching_prefix": teaching_prefix,
        "function_parameters": function_parameters,
        "digit_tokens": digit_tokens,
        "minus_token": minus_token,
        "dot_token": dot_token,
        "comma_token": comma_token,
        "close_brace_token": close_brace_token,
        "bool_paths": bool_paths,
    }


def _postprocess_result(output: str, prompt: str) -> dict[str, Any]:
    """Parse and clean parameters from generated JSON output.

    Args:
        output: Generated JSON string.
        prompt: Prompt associated with the output.

    Returns:
        A parsed result dictionary, or a fallback result on JSON errors.
    """
    try:
        result_dict: dict[str, Any] = json.loads(output)
        if "parameters" in result_dict:
            dict_res = result_dict["parameters"].items()
            clean_parameters = {}
            for param_name, param_value in dict_res:
                if not param_name:
                    continue
                if isinstance(param_value, str):
                    clean_parameters[param_name] = param_value.strip()
                else:
                    clean_parameters[param_name] = param_value
            result_dict["parameters"] = clean_parameters
        return result_dict
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON. Error: {e}")
        return {"prompt": prompt, "name": "", "parameters": {}}


def generate_result_for_prompt(
    prompt: str,
    model: Small_LLM_Model,
    metadata: GenerationMetadata,
    max_tokens: int = 300,
) -> dict[str, Any]:
    """Generate a single result dict for a prompt.

    Args:
        prompt: Prompt to decode.
        model: Tokenizer/model used for generation.
        metadata: Precomputed decoding metadata.
        max_tokens: Maximum number of generated tokens.

    Returns:
        The parsed result dictionary for the prompt.
    """
    function_name_paths = metadata["function_name_paths"]
    teaching_prefix = metadata["teaching_prefix"]
    function_parameters = metadata["function_parameters"]
    digit_tokens = metadata["digit_tokens"]
    minus_token = metadata["minus_token"]
    dot_token = metadata["dot_token"]
    comma_token = metadata["comma_token"]
    close_brace_token = metadata["close_brace_token"]
    bool_paths = metadata["bool_paths"]

    p = json.dumps({"prompt": prompt, "name": ""})[:-2]
    print(p, end="", flush=True)
    output = p
    state = 1
    param_index = 0
    function_param: list[tuple[str, str]] = []
    ids = build_full_prompt(teaching_prefix, prompt, model)
    ids += model.encode(p).tolist()[0]
    gen_ids: list[int] = []
    fn_name = ""
    braket_track = 1
    token_count = 0
    past_key_values = None
    processed_len = 0

    while token_count < max_tokens:
        token_count += 1
        new_token_ids = ids[processed_len:]
        logits, past_key_values = model.get_logits_from_input_ids(
            ids, past_key_values, new_token_ids
        )
        processed_len = len(ids)
        logits_for_sampling: list[float] | NDArray[np.float32] = logits

        if state in (1, 3, 5):
            constrained_logits = constrained_decoding(
                logits,
                state,
                function_name_paths,
                gen_ids,
                digit_tokens,
                minus_token,
                dot_token,
                comma_token,
                close_brace_token,
                bool_paths,
            )
            if constrained_logits is not None:
                logits_for_sampling = constrained_logits

        new_id = int(np.argmax(logits_for_sampling))
        new_token = model.decode([new_id])
        ids.append(new_id)
        output += new_token
        print(new_token, end="", flush=True)

        if new_token == '\",' and state == 1:
            gen_ids = []
            function_param = function_parameters.get(fn_name, [])
            if function_param:
                name, param_type = function_param[0]
                quote = '"' if param_type == "string" else ""
                new_s = f' "parameters":{{"{name}":{quote}'
                state = (
                    3
                    if param_type in ("number", "integer")
                    else (5 if param_type == "boolean" else 2)
                )
            else:
                new_s = ' "parameters":{}'
                state = 2
            print(new_s, end="", flush=True)
            output += new_s
            ids.extend(model.encode(new_s).tolist()[0])
            braket_track += new_s.count("{") - new_s.count("}")

        elif state == 1:
            gen_ids.append(new_id)
            fn_name += new_token

        elif state in (3, 5):
            if "," in new_token or "}" in new_token:
                param_index += 1
                if param_index < len(function_param):
                    next_name, next_type = function_param[param_index]
                    quote = '"' if next_type == "string" else ""
                    new_s = f' "{next_name}":{quote}'
                    print(new_s, end="", flush=True)
                    output += new_s
                    ids.extend(model.encode(new_s).tolist()[0])
                    braket_track += new_s.count("{") - new_s.count("}")
                    gen_ids = []
                    state = (
                        3
                        if next_type in ("number", "integer")
                        else (5 if next_type == "boolean" else 2)
                    )
                else:
                    state = 2
            else:
                gen_ids.append(new_id)

        braket_track += new_token.count("{") - new_token.count("}")
        if braket_track == 0:
            break
    print()

    return _postprocess_result(output, prompt)
