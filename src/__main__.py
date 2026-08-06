import json
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from llm_sdk import Small_LLM_Model

from .constraints import constrained_decoding
from .io_utils import (
    build_empty_results,
    filter_valid_functions,
    get_args,
    load_input_files,
    write_results,
)
from .prompting import build_full_prompt, get_prompt_prefix


def main() -> None:
    fn_df, input_path, output_path = get_args()
    try:
        prompts, functions = load_input_files(fn_df, input_path)
    except Exception as e:
        print(f"Error loading input files: {e}")
        return

    if not functions:
        print("Warning: No function definitions provided.")
        results = build_empty_results(prompts)
        write_results(output_path, results)
        print(f"\n Successfully wrote {len(results)} results to {output_path}")
        return

    valid_functions = filter_valid_functions(functions)
    if not valid_functions:
        print("Warning: No valid function definitions with non-empty names.")
        results = build_empty_results(prompts)
        write_results(output_path, results)
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
        function_param: list[tuple[str, str]] = []
        ids = build_full_prompt(teaching_prefix, prompt, model)
        ids += model.encode(p).tolist()[0]
        gen_ids: list[int] = []
        fn_name = ""
        braket_track = 1
        token_count = 0
        max_tokens = 300
        past_key_values = None
        processed_len = 0

        while token_count < max_tokens:
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
            results.append({"prompt": prompt, "name": "", "parameters": {}})
    try:
        write_results(output_path, results)
        print(f"\n Successfully wrote {len(results)} results to {output_path}")
    except Exception as e:
        print(f"\n Error writing to output file: {e}")


if __name__ == "__main__":
    main()
