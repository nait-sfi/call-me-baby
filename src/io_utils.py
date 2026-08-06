import argparse
import json
from typing import Any

from .schema import FunctionDef, Prompt


def get_args() -> tuple[str, str, str]:
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(description="call me baby")
    arg_parser.add_argument("--functions_definition", required=True)
    arg_parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    arg_parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
    )
    args = arg_parser.parse_args()
    return args.functions_definition, args.input, args.output


def load_input_files(
    functions_definition_path: str,
    input_path: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Load and validate prompts and function definitions."""
    with open(input_path, encoding="utf-8") as file_obj:
        prompts_raw = json.load(file_obj)
        prompts = [
            Prompt.model_validate(prompt).prompt
            for prompt in prompts_raw
        ]

    with open(functions_definition_path, encoding="utf-8") as file_obj:
        functions = json.load(file_obj)
        [FunctionDef.model_validate(func) for func in functions]

    return prompts, functions


def filter_valid_functions(
    functions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only functions with a non-empty name."""
    valid_functions = []
    for function in functions:
        function_name = function.get("name", "")
        if not function_name:
            print("Warning: Skipping function definition with empty name.")
            continue
        valid_functions.append(function)
    return valid_functions


def build_empty_results(prompts: list[str]) -> list[dict[str, Any]]:
    """Build empty fallback results matching the prompt count."""
    return [
        {"prompt": prompt, "name": "", "parameters": {}}
        for prompt in prompts
    ]


def write_results(output_path: str, results: list[dict[str, Any]]) -> None:
    """Persist output results to disk."""
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)
