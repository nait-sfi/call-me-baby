# from llm_sdk import Small_LLM_Model
import argparse
import json
from src.schema import FunctionDef


def main() -> None:
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
    fn_df_file = args.functions_definition

    with open(fn_df_file, "r") as file:
        raw_data = json.load(file)
        print(raw_data)
        validated_functions = []
        validated_functions = [FunctionDef.model_validate(item)
                               for item in raw_data
                               ]
        print(validated_functions)


if __name__ == "__main__":
    main()
