import argparse
import json
from llm_sdk import Small_LLM_Model
import numpy as np
from .schema import Prompt
from .schema import FunctionDef


def get_prompt(user_prompt: str, functions: list[dict]) -> str:
    return f"""
You are a function-calling assistant.

Your job is to analyze the user's request and determine whether one of the available functions should be called.

Available functions:
{functions}

Example

User:
What's the sum of 8 and 9?

Output:
{{
  "prompt": "What's the sum of 8 and 9?",
  "name": "fn_add_numbers",
  "parameters": {{
    "a": 8.0,
    "b": 9.0
  }}
}}

Now process the following request.
User:
{user_prompt}

JSON:
""" # noqa


def get_logits(logits, allowed_id):
    logits = np.asarray(logits)
    logits_cpy = np.full(151643, -np.inf, dtype=np.float32)
    idx = list(allowed_id)
    logits_cpy[idx] = logits[idx]
    return logits_cpy


def constrained_function_name(functions, gen_ids, model):
    nb_tk = len(gen_ids)
    fn_ids = [model.encode(func["name"] + '",').tolist()[0]
              for func in functions]
    return {ids[nb_tk] for ids in fn_ids if gen_ids == ids[:nb_tk]}


def constrained_decoding(
    logits: list[int],
    state: int,
    functions: list[dict],
    gen_ids: list[int],
    model
):
    if state == 1:
        allowed_ids = constrained_function_name(functions, gen_ids, model)
        return get_logits(logits, allowed_ids)


def get_args():
    arg_parser = argparse.ArgumentParser(description="call me baby")
    arg_parser.add_argument("--functions_definition", required=True)
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
    input = args.input
    output = args.output

    return fn_df, input, output


def main():

    fn_df, input, ouput = get_args()

    with open(input) as f:
        prompts = json.load(f)
        prompts = [Prompt.model_validate(prompt).prompt for prompt in prompts]

    with open(fn_df) as f:
        functions = json.load(f)
        [FunctionDef.model_validate(func) for func in functions]

    model = Small_LLM_Model()
    results = []

    for prompt in prompts:
        p = json.dumps({"prompt": prompt, "name": ""})[:-2]
        function_parameters = {
            function["name"]: [
                (name, type["type"])
                for name, type in function["parameters"].items()
            ]
            for function in functions
        }
        print(p, end="", flush=True)
        output = p
        state = 1
        ids = model.encode(get_prompt(prompt, functions)).tolist()[0]
        ids += model.encode(p).tolist()[0]
        gen_ids = []
        fn_name = ""
        while 1:
            logits = model.get_logits_from_input_ids(ids)
            if state == 1:
                logits = constrained_decoding(
                        logits,
                        state,
                        functions,
                        gen_ids, model
                    )
            new_id = int(np.argmax(logits))
            new_token = model.decode(new_id)
            ids.append(new_id)
            output += new_token
            print(new_token, end="", flush=True)
            if new_token == '",' and state == 1:
                state = 2
                gen_ids = []
                function_param = function_parameters[fn_name]
                if function_parameters:
                    name, type = function_param.pop(0)
                quote = '"' if type == "string" else ""
                new_s = f' "parameters":{{"{name}":{quote}'
                print(new_s, end="", flush=True)
                output += new_s
                ids.extend(model.encode(new_s).tolist()[0])
            elif state == 1:
                gen_ids.append(new_id)
                fn_name += new_token
            if "}" in new_token:
                break
        print()

        try:
            result_dict = json.loads(output)
            results.append(result_dict)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON. Error: {e}")

    try:
        with open(ouput, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n Successfully wrote {len(results)} results to {ouput}")
    except Exception as e:
        print(f"\n Error writing to output file: {e}")


if __name__ == "__main__":
    main()
