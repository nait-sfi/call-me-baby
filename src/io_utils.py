import argparse
import json
from typing import Any

from .schema import FunctionDef, Prompt


def get_args() -> tuple[str, str, str]:
    """Parse command line arguments.

    Returns:
        tuple[str, str, str]: Function-definition, input, and output paths.
    """
    arg_parser = argparse.ArgumentParser(description="call me baby")
    arg_parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
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


def prevent_duplicates(
    ordered_pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate keys while loading JSON objects.

    Args:
        ordered_pairs (list[tuple[str, Any]]): Ordered JSON key/value pairs.

    Returns:
        dict[str, Any]: Dictionary with unique keys.
    """
    d: dict[str, Any] = {}
    for key, value in ordered_pairs:
        if key in d:
            raise ValueError(f"Duplicate key found: {key}")
        d[key] = value
    return d


def load_input_files(
    functions_definition_path: str,
    input_path: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Load and validate prompts and function definitions.

    Args:
        functions_definition_path (str): Path to function definition JSON file.
        input_path (str): Path to prompt input JSON file.

    Returns:
        tuple[list[str], list[dict[str, Any]]]: Validated prompts and raw
        functions.
    """
    with open(input_path, encoding="utf-8") as file_obj:
        prompts_raw = json.load(file_obj, object_pairs_hook=prevent_duplicates)
        prompts = [Prompt.model_validate(prompt).prompt
                   for prompt in prompts_raw]

    with open(functions_definition_path, encoding="utf-8") as file_obj:
        functions = json.load(file_obj, object_pairs_hook=prevent_duplicates)
        [FunctionDef.model_validate(func) for func in functions]

    return prompts, functions


def filter_valid_functions(
    functions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only functions with a non-empty name.

    Args:
        functions (list[dict[str, Any]]): Raw function definitions.

    Returns:
        list[dict[str, Any]]: Functions with non-empty names.
    """
    valid_functions = []
    for function in functions:
        function_name = function.get("name", "")
        if not function_name:
            print("Warning: Skipping function definition with empty name.")
            continue
        valid_functions.append(function)
    return valid_functions


def build_empty_results(prompts: list[str]) -> list[dict[str, Any]]:
    """Build empty fallback results matching the prompt count.

    Args:
        prompts (list[str]): Prompts to mirror in the fallback output.

    Returns:
        list[dict[str, Any]]: Fallback results.
    """
    return [{"prompt": prompt, "name": "", "parameters": {}}
            for prompt in prompts]


def write_results(output_path: str, results: list[dict[str, Any]]) -> None:
    """Persist output results to disk.

    Args:
        output_path (str): Destination JSON file path.
        results (list[dict[str, Any]]): Results to serialize.

    Returns:
        None: This function does not return a value.
    """
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)
