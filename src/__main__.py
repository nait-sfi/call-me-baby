# import argparse
# import json
# from src.schema import FunctionDef
# from src.decoder import mask_logits
from llm_sdk import Small_LLM_Model
import numpy as np


def main():
    prompt = "Question: What is the sum of 2 and 3?\nFunction name:"
    model = Small_LLM_Model()
    for _ in range(2):
        input_ids = model.encode(prompt).tolist()[0]
        add_number_id = model.encode(" fn_add_numbers")
        greeting_id = model.encode(" fn_greet")
        allowed = np.array([add_number_id, greeting_id])
        logits = model.get_logits_from_input_ids(input_ids)
        inf_arr = np.full(len(logits), -np.inf)
        np_logits = np.array(logits)

        inf_arr[allowed] = np_logits[allowed]

        win = inf_arr.argmax()
        input_ids.append(win)
    print(model.decode(input_ids))




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
