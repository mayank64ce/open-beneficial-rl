# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

`open-beneficial-rl` is an open implementation of the OpenAI paper **"Reinforcement Learning Towards Broadly and Persistently Beneficial Models"** (arXiv:2606.24014, June 2026), a copy of which lives at `paper/2606.24014.pdf`.

The paper's core idea, which this repo aims to reproduce/build on: train models with RL on a dataset of realistic situations designed to elicit *beneficial traits* (truthfulness, fairness, risk awareness, corrigibility) across domains like health, science, and education, then measure whether that alignment **generalizes out-of-distribution** and **persists** under adversarial steering toward misalignment. Read the paper before making design decisions — the dataset construction, the >50 alignment/beneficial-behavior benchmarks, and the persistence evaluations are the substance of the work.

## Current state

The repository is a fresh scaffold. As of this writing it contains only:
- `README.md` — project name
- `.gitignore` — standard Python ignore
- `paper/2606.24014.pdf` — the reference paper (untracked)

There is **no source code, dependency manifest, build system, or test suite yet.** Do not document or assume commands (build/lint/test/run) that don't exist — when the first code lands, add the real commands here. The Python `.gitignore` indicates this will be a Python project; confirm the tooling (package manager, test runner) once it's chosen rather than guessing.
