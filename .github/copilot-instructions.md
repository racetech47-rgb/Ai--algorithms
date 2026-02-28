# Copilot Instructions for Ai--algorithms

## Repository Overview

This repository is a comprehensive AI algorithms suite implementing core machine learning and artificial intelligence techniques, including:

- **Neural Networks** – Feedforward/backpropagation architectures for classification and regression.
- **Simulated Annealing** – Metaheuristic optimization for combinatorial and continuous problems.
- **Natural Language Processing (NLP)** – Text processing, tokenization, and language modeling utilities.

The project is language-agnostic at the repository level; individual algorithm implementations may be in Python, JavaScript, or other languages. Check each subdirectory for its own language and runtime requirements.

## Repository Layout

```
/
├── .github/
│   └── copilot-instructions.md   # This file
└── README.md                     # Project summary
```

As the codebase grows, expect subdirectories named after algorithm families (e.g., `neural-networks/`, `simulated-annealing/`, `nlp/`). Each subdirectory may contain its own `README.md`, dependency manifest, and test suite.

## Build & Validation

Because the repository currently contains only source and documentation files, there is no global build step. Follow the per-language instructions below when working in each area:

### Python implementations
```bash
# Install dependencies (always run before any other step)
pip install -r requirements.txt   # if present in the relevant directory

# Run tests
pytest                             # from the directory containing tests

# Lint
flake8 .                           # PEP-8 style check
```

### JavaScript/Node.js implementations
```bash
# Install dependencies (always run before any other step)
npm install

# Run tests
npm test

# Lint
npm run lint
```

If a `requirements.txt`, `package.json`, or equivalent manifest is absent, create one when adding new code so that future agents can reproduce the environment reliably.

## Coding Conventions

- Follow PEP 8 for Python and the existing ESLint/Prettier config for JavaScript.
- Every new algorithm must include at least one unit test that exercises the happy path.
- Keep algorithm implementations self-contained in their own module/file; avoid cross-algorithm dependencies unless explicitly documented.
- Add or update the relevant `README.md` when introducing a new algorithm or changing a public API.

## Tips for Coding Agents

- Trust these instructions first; only search the codebase if the information here appears incomplete or incorrect.
- When adding a new algorithm, create its directory, implementation file, and test file together in the same pull request.
- Always run the applicable lint and test commands (see above) before marking a task as complete.
- If CI workflows are added in the future, check `.github/workflows/` for the authoritative list of checks that must pass.
