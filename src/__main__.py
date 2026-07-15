# import argparse
# import json
# from src.schema import FunctionDef
# from src.decoder import mask_logits
# from llm_sdk import Small_LLM_Model
# import numpy as np


# def main():
#     prompt = "Question: What is the sum of 2 and 3?\nFunction name:"
#     model = Small_LLM_Model()
#     for _ in range(2):
#         input_ids = model.encode(prompt).tolist()[0]
#         add_number_id = model.encode(" fn_add_numbers")
#         greeting_id = model.encode(" fn_greet")
#         allowed = np.array([add_number_id, greeting_id])
#         logits = model.get_logits_from_input_ids(input_ids)
#         inf_arr = np.full(len(logits), -np.inf)
#         np_logits = np.array(logits)

#         inf_arr[allowed] = np_logits[allowed]

#         win = inf_arr.argmax()
#         input_ids.append(win)
#     print(model.decode(input_ids))




# def main():
#     model = Small_LLM_Model()
#     input_ids = model.encode("The capital of France is").tolist()[0]
#     london_id = model.encode(" London").tolist()[0][0]
#     berlin_id = model.encode(" Berlin").tolist()[0][0]
#     allowed_ids = [london_id, berlin_id]

#     for _ in range(5):
#         logits = model.get_logits_from_input_ids(input_ids)
#         inf_arr = np.full(len(logits), -np.inf)
#         np_arr_logits = np.array(logits)
#         inf_arr[allowed_ids] = np_arr_logits[allowed_ids]
#         win = inf_arr.argmax()
#         input_ids.append(win)
#     print("The result is : ", model.decode(input_ids))

# def main() -> None:
#     # Fake AI outputting 5 scores (for tokens 0, 1, 2, 3, 4)
#     fake_logits = [0.5, 2.1, -1.0, 5.5, 0.0]

#     # We ONLY want to allow token ID 1 and 3.
#     allowed = [1, 3]

#     # Run your function!
#     masked = mask_logits(fake_logits, allowed, 5)
#     print("Masked Logits:", masked)
#     return
#     arg_parser = argparse.ArgumentParser(description="call me baby")
#     arg_parser.add_argument("--functions_definition", required=True)
#     arg_parser.add_argument(
#         "--input",
#         default="data/input/function_calling_tests.json"
#         )
#     arg_parser.add_argument(
#         "--output",
#         default="data/output/function_calls.json"
#         )

#     args = arg_parser.parse_args()
#     fn_df_file = args.functions_definition

#     with open(fn_df_file, "r") as file:
#         raw_data = json.load(file)
#         print(raw_data)
#         validated_functions = []
#         validated_functions = [FunctionDef.model_validate(item)
#                                for item in raw_data
#                                ]
#         print(validated_functions)



# if __name__ == "__main__":
#     main()
# import argparse
import json
# from src.schema import FunctionDef
# from src.decoder import mask_logits
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
"""


def get_logits(logits, allowed_id):
    logits_cpy = [-np.inf] * len(logits)
    for i in allowed_id:
        logits_cpy[i] = logits[i]
    return logits_cpy


def constrained_function_name(functions, gen_ids, model):
    nb_tk = len(gen_ids)
    fn_ids = [model.encode(func["name"] + '",').tolist()[0]
              for func in functions]
    return {ids[nb_tk] for ids in fn_ids if gen_ids == ids[:nb_tk]}


def constrained_decoding(logits: list[int], state: int, functions: list[dict], gen_ids: list[int], model):
    if state == 1:
        allowed_ids = constrained_function_name(functions, gen_ids, model)
        return get_logits(logits, allowed_ids)


def main():
    with open("data/input/function_calling_tests.json") as f:
        prompts = json.load(f)
        prompts = [Prompt.model_validate(prompt).prompt for prompt in prompts]
    with open("data/input/functions_definition.json") as f:
        functions = json.load(f)
        [FunctionDef.model_validate(func) for func in functions]
    model = Small_LLM_Model()
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
                logits = constrained_decoding(logits, state, functions, gen_ids, model)
            if state == 2:
                # constrained decoding of params 
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
                new_s = f'"parameters":{{"{name}":{quote}'
                print(new_s, end="", flush=True)
                output += new_s
                ids.extend(model.encode(new_s).tolist()[0])
            elif state == 1:
                gen_ids.append(new_id)
                fn_name += new_token
            if "}" in new_token:
                break
        print()


    # for _ in range(2):
    #     input_ids = model.encode(prompt).tolist()[0]
    #     add_number_id = model.encode(" fn_add_numbers")
    #     greeting_id = model.encode(" fn_greet")
    #     allowed = np.array([add_number_id, greeting_id])
    #     logits = model.get_logits_from_input_ids(input_ids)
    #     inf_arr = np.full(len(logits), -np.inf)
    #     np_logits = np.array(logits)

    #     inf_arr[allowed] = np_logits[allowed]

    #     win = inf_arr.argmax()
    #     input_ids.append(win)
    # print(model.decode(input_ids))




# def main():
#     model = Small_LLM_Model()
#     input_ids = model.encode("The capital of France is").tolist()[0]
#     london_id = model.encode(" London").tolist()[0][0]
#     berlin_id = model.encode(" Berlin").tolist()[0][0]
#     allowed_ids = [london_id, berlin_id]

#     for _ in range(5):
#         logits = model.get_logits_from_input_ids(input_ids)
#         inf_arr = np.full(len(logits), -np.inf)
#         np_arr_logits = np.array(logits)
#         inf_arr[allowed_ids] = np_arr_logits[allowed_ids]
#         win = inf_arr.argmax()
#         input_ids.append(win)
#     print("The result is : ", model.decode(input_ids))

# def main() -> None:
#     # Fake AI outputting 5 scores (for tokens 0, 1, 2, 3, 4)
#     fake_logits = [0.5, 2.1, -1.0, 5.5, 0.0]

#     # We ONLY want to allow token ID 1 and 3.
#     allowed = [1, 3]

#     # Run your function!
#     masked = mask_logits(fake_logits, allowed, 5)
#     print("Masked Logits:", masked)
#     return
#     arg_parser = argparse.ArgumentParser(description="call me baby")
#     arg_parser.add_argument("--functions_definition", required=True)
#     arg_parser.add_argument(
#         "--input",
#         default="data/input/function_calling_tests.json"
#         )
#     arg_parser.add_argument(
#         "--output",
#         default="data/output/function_calls.json"
#         )

#     args = arg_parser.parse_args()
#     fn_df_file = args.functions_definition

#     with open(fn_df_file, "r") as file:
#         raw_data = json.load(file)
#         print(raw_data)
#         validated_functions = []
#         validated_functions = [FunctionDef.model_validate(item)
#                                for item in raw_data
#                                ]
#         print(validated_functions)



if __name__ == "__main__":
    main()
