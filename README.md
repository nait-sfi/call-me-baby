*This project has been created as part of the 42 curriculum by nait-sfi.*

# call-me-baby

## Description
`call-me-baby` is a constrained-decoding function-calling prototype built around `Small_LLM_Model` (from `llm_sdk`).  
Its goal is to convert natural-language prompts into structured JSON function calls (`name` + `parameters`) while forcing valid function-name generation against a provided function catalog.

At runtime, the program:
1. Loads function definitions and prompt inputs from JSON files.
2. Builds a teaching prompt and runs token-by-token decoding.
3. Constrains function-name tokens to legal prefixes of known function names.
4. Writes generated function calls to an output JSON file.

## Instructions
### Requirements
- Python 3.12+
- `uv` installed

### Installation
```bash
make install
```

### Execution
Use the Makefile target:
```bash
make run
```

Equivalent command:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Debug
```bash
make debug
```

### Linting
```bash
make lint
```

## Algorithm explanation (constrained decoding)
The constrained decoding is implemented in `src/generation.py` around a state-driven loop:
1. **Pre-tokenize allowed function names**: each function name is encoded as tokens (`function_name_paths`).
2. **Track generated function-name prefix**: while in state `1`, generated IDs are stored in `gen_ids`.
3. **Compute legal next-token set**: `constrained_function_name()` keeps only token IDs that continue at least one valid function-name path matching `gen_ids`.
4. **Mask logits**: `get_logits()` replaces all non-allowed token logits with `-inf`, so `argmax` can only select legal continuations.
5. **Transition state**: once `",` closes the function name, state changes to `2` and parameter emission begins.
6. **Stop condition**: bracket tracking and max-token guard (`MAX_TOKENS`) prevent runaway generation.

This creates a deterministic constrained step for function-name decoding while leaving the rest of generation model-driven.

## Design decisions
- **Deterministic decoding (`argmax`)**: chosen for reproducibility and easier debugging.
- **Prefix-path constraint over regex/string filtering**: operates directly in token space, which matches model logits and avoids post-hoc string correction.
- **Prompt-prefix caching**: `get_prompt_prefix()` precomputes static prompt parts once per run to reduce repeated work.
- **Strict schema validation**: input prompts and function definitions are validated with Pydantic models (`src/schema.py`) before generation.
- **Separation of concerns**: CLI wiring lives in `src/__main__.py`, while decoding helpers live in `src/generation.py`.
- **Graceful fallback paths**: empty/invalid function sets are handled explicitly and still produce structured output.

## Performance analysis
- **Accuracy**: function-name validity is improved because impossible names are masked out during state `1`.
- **Speed**: per-step masking is lightweight (`NumPy` array fill + indexed assignment), but full generation remains token-iterative and model-bound.
- **Reliability**: max-token limit, bracket tracking, and JSON parse checks reduce infinite loops and malformed output risks.

Current trade-off: constraints are strongest for function-name selection; parameter value quality still depends on model behavior and prompt quality.

## Challenges faced
- **Token-level control**: enforcing valid function names required moving from text-level reasoning to token-prefix path matching.
- **State synchronization**: switching correctly from constrained name generation to free-form parameter generation required explicit state transitions.
- **JSON integrity during streaming generation**: bracket counting and post-generation parsing were used to detect/limit malformed outputs.

## Testing strategy
- **Schema-level validation**: verify malformed prompts/function definitions are rejected early by Pydantic.
- **Golden-input runs**: execute with `data/input/functions_definition.json` and `data/input/function_calling_tests.json`, then inspect `data/output/function_calls.json`.
- **Edge-case checks**:
  - empty function list,
  - function definitions with empty names,
  - prompts that stress parameter extraction.
- **Static quality checks**: run `make lint` (`flake8` + `mypy`) to keep typing/style issues controlled.

## Example usage
Run:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Expected output file shape:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Resources
### Classic references
- OpenAI tokenizer and token-level generation concepts: https://platform.openai.com/docs
- NumPy documentation (masking/indexing operations): https://numpy.org/doc/
- Pydantic documentation (data validation): https://docs.pydantic.dev/
- Constrained decoding overview (general concept): https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/

### AI usage disclosure
AI was used as a development assistant for:
- drafting and refining README wording and structure,
- reviewing phrasing for algorithm/design/performance explanations,
- improving clarity of command examples and section organization.

Core implementation logic and integration decisions were produced and adjusted manually in the project code.
