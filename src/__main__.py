from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    print("Model loaded successfully!")
    print("Tokens for 'Hello World':", model.encode("Hello World"))


if __name__ == "__main__":
    main()
