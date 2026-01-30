format-check:
    ruff format --check

format-fix:
    ruff format

lint-check:
    ruff check
    torchfix .

lint-fix:
    ruff check --fix
    torchfix --fix .

type-check:
    mypy

spell-check:
    codespell

check: format-check lint-check spell-check

fix: format-fix lint-fix

test:
    pytest
