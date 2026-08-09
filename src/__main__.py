from llm_sdk import Small_LLM_Model

from .io_utils import (
    build_empty_results,
    filter_valid_functions,
    get_args,
    load_input_files,
    write_results,
)
from .generation import generate_result_for_prompt, prepare_model_metadata


def main() -> None:
    """Run the function-calling generation pipeline.

    Returns:
        None: This function does not return a value.
    """
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
    metadata = prepare_model_metadata(model, valid_functions)

    results = []
    for prompt in prompts:
        result = generate_result_for_prompt(prompt, model, metadata)
        results.append(result)

    try:
        write_results(output_path, results)
        print(f"\n Successfully wrote {len(results)} results to {output_path}")
    except Exception as e:
        print(f"\n Error writing to output file: {e}")


if __name__ == "__main__":
    main()
