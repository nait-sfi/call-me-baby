import argparse
import json
from collections.abc import Iterable, Sequence
from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
from .schema import Prompt
from .schema import FunctionDef
from llm_sdk import Small_LLM_Model


def get_prompt_prefix(
    functions: list[dict[str, Any]],
    model: Small_LLM_Model,
) -> list[int]:
    """Pre-compute the teaching prompt prefix.

    Args:
        functions: Function definitions injected in the prompt template.
        model: Model used to tokenize the prompt.

    Returns:
        list[int]: Prefix token IDs.
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
        prefix: Precomputed prompt prefix token IDs.
        user_prompt: Current user prompt text.
        model: Model used to tokenize prompt text.

    Returns:
        list[int]: Full prompt token IDs.
    """
    end = [198, 5370, 510]
    prompt_ids = cast(list[int], model.encode(user_prompt).tolist()[0])
    return prefix + prompt_ids + end


def get_logits(
    logits: Sequence[float] | NDArray[np.float32],
    allowed_ids: Iterable[int],
) -> NDArray[np.float32]:
    """Mask logits so only allowed token IDs can be selected.

    Args:
        logits: Raw logits returned by the model.
        allowed_ids: Token IDs that remain available for decoding.

    Returns:
        NDArray[np.float32]: Masked logits array.
    """
    logits_array = np.asarray(logits, dtype=np.float32)
    logits_copy = np.full(logits_array.shape[0], -np.inf, dtype=np.float32)
    idx = list(allowed_ids)
    logits_copy[idx] = logits_array[idx]
    return logits_copy


def constrained_function_name(
    function_name_paths: dict[str, list[int]],
    gen_ids: list[int],
) -> set[int]:
    """Return allowed next-token IDs for function-name decoding.

    Args:
        function_name_paths: Tokenized function names keyed by name.
        gen_ids: Already generated function-name token IDs.

    Returns:
        set[int]: Candidate next token IDs.
    """
    nb_tk = len(gen_ids)
    return {
        ids[nb_tk]
        for ids in function_name_paths.values()
        if len(ids) > nb_tk and gen_ids == ids[:nb_tk]
    }


def constrained_decoding(
    logits: Sequence[float] | NDArray[np.float32],
    state: int,
    functions_name_paths: dict[str, list[int]],
    gen_ids: list[int],
    digit_tokens,
    minus_token,
    dot_token,
    comma_token,
    close_brace_token,
    bool_paths
) -> NDArray[np.float32] | None:
    """Apply constrained decoding according to the generation state.

    Args:
        logits: Raw logits from the model.
        state: Decoding state machine value.
        functions_name_paths: Tokenized function names keyed by name.
        gen_ids: Generated token IDs for the function name fragment.

    Returns:
        NDArray[np.float32] | None: Constrained logits for state 1, otherwise
        None.
    """
    if state == 1:
        allowed_ids = constrained_function_name(functions_name_paths, gen_ids)
        if not allowed_ids:
            return np.asarray(logits, dtype=np.float32)
        return get_logits(logits, allowed_ids)
    if state == 3:
        allowed_ids = constrained_number_value(
            gen_ids,
            digit_tokens,
            minus_token,
            dot_token,
            comma_token,
            close_brace_token,
            bool_paths
        )
        return get_logits(logits, allowed_ids)
    if state == 5:
        allowed_ids = constrained_function_name(
            bool_paths, gen_ids
        )
        if not allowed_ids:
            return np.asarray(logits, dtype=np.float32)
        return get_logits(logits, allowed_ids)

    return None


def get_args() -> tuple[str, str, str]:
    """Parse command line arguments.

    Returns:
        tuple[str, str, str]: Function definitions path, prompts path, and
        output path.
    """
    arg_parser = argparse.ArgumentParser(description="call me baby")
    arg_parser.add_argument("--functions_definition",
                            default="data/input/functions_definition.json",
                            )
    arg_parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    arg_parser.add_argument(
        "--output",
        default="data/output/function_calls.json"
    )

    args = arg_parser.parse_args()
    fn_df = args.functions_definition
    input_path = args.input
    output = args.output

    return fn_df, input_path, output


def constrained_number_value(
    gen_ids: list[int],
    digit_tokens: set[int],
    minus_token: int,
    dot_token: int,
    comma_token: int,
    close_brace_token: int,
    bool_paths
) -> set[int]:
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


def main() -> None:
    fn_df, input_path, output_path = get_args()
    try:
        with open(input_path, encoding="utf-8") as file_obj:
            prompts = json.load(file_obj)
            prompts = [Prompt.model_validate(prompt).prompt
                       for prompt in prompts]

        with open(fn_df, encoding="utf-8") as file_obj:
            functions = json.load(file_obj)
            [FunctionDef.model_validate(func) for func in functions]
    except Exception as e:
        print(f"Error loading input files: {e}")
        return

    if not functions:
        print("Warning: No function definitions provided.")
        results = [
            {"prompt": prompt, "name": "", "parameters": {}}
            for prompt in prompts
        ]
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=2)
        print(f"\n Successfully wrote {len(results)} results to {output_path}")
        return

    valid_functions = []
    for function in functions:
        function_name = function.get("name", "")
        if not function_name:
            print("Warning: Skipping function definition with empty name.")
            continue
        valid_functions.append(function)

    if not valid_functions:
        print("Warning: No valid function definitions with non-empty names.")
        results = [
            {"prompt": prompt, "name": "", "parameters": {}}
            for prompt in prompts
        ]
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=2)
        print(f"\n Successfully wrote {len(results)} results to {output_path}")
        return

    model = Small_LLM_Model()
    results = []
    function_name_paths = {
        func["name"]: model.encode(func["name"] + '",').tolist()[0]
        for func in valid_functions
    }

    teaching_prefix = get_prompt_prefix(valid_functions, model)
    function_parameters = {
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

    for prompt in prompts:
        p = json.dumps({"prompt": prompt, "name": ""})[:-2]
        print(p, end="", flush=True)
        output = p
        state = 1
        param_index = 0
        ids = build_full_prompt(teaching_prefix, prompt, model)
        ids += model.encode(p).tolist()[0]
        gen_ids: list[int] = []
        fn_name = ""
        braket_track = 1
        token_count = 0
        MAX_TOKENS = 300
        past_key_values = None
        processed_len = 0

        while token_count < MAX_TOKENS:
            token_count += 1
            new_token_ids = ids[processed_len:]
            logits, past_key_values = model.get_logits_from_input_ids(
                ids, past_key_values, new_token_ids
            )
            processed_len = len(ids)
            logits_for_sampling: Sequence[float] | NDArray[np.float32] = logits
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
            if new_token == '",' and state == 1:
                gen_ids = []
                function_param = function_parameters.get(fn_name, [])
                if function_param:
                    name, param_type = function_param.pop()
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
            elif state == 1:
                gen_ids.append(new_id)
                fn_name += new_token
            elif state in (3, 5):
                if new_token in (",", "}"):
                    param_index += 1
                    if param_index < len(function_param):
                        next_name, next_type = function_param[param_index]
                        quote = '"' if next_type == "string" else ""
                        new_s = f' "{next_name}":{quote}'
                        output += new_s
                        ids.extend(model.encode(new_s).tolist()[0])
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
            if "}" in new_token:
                braket_track -= 1
                if braket_track == 0:
                    break
            elif "{" in new_token:
                braket_track += 1
        print()

        try:
            result_dict = json.loads(output)
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
            results.append(result_dict)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON. Error: {e}")
    try:
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=2)
        print(f"\n Successfully wrote {len(results)} results to {output_path}")
    except Exception as e:
        print(f"\n Error writing to output file: {e}")


if __name__ == "__main__":
    main()
