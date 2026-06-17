install:
	uv sync

run: uv run python main.py

debug: uv run python db main.py

clean: rf -rf __pycach__ 
