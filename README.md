# Pilot: Opponent-Type Classification of Kuhn Poker Matches

A small pilot for my final project in Large Language Models (UMass Amherst). The project asks whether a strategic LLM can infer which opponent it is playing from the text of an interaction. This pilot tests the measurement step on its own: an LLM reads the full log of a Kuhn Poker match and outputs a probability distribution over five opponent types, which is scored against the known true type.

## What the script does

`pilot.py` runs the whole pipeline:

1. **Simulates the data.** It generates eight-hand Kuhn Poker matches between a fixed near-equilibrium learner and five scripted opponents, so every document has a ground-truth label. Matches are seeded, so the same documents are rebuilt on every run.
2. **Writes each match as text.** Each match becomes one line per hand from the learner's point of view. The opponent's card is shown only at showdown.
3. **Queries the model.** It sends each match to a local Ollama model through `/api/chat`, one request at a time.
4. **Scores the answers.** It parses the JSON reply, checks that it is valid, and computes accuracy, log loss and Brier score on the true type.
5. **Saves everything** to a pandas DataFrame (`results.csv`) and records the full configuration (`run_config.json`).

## Opponent types

The model sees neutral labels (Type A to E), because an earlier assignment showed that names such as "Nash" or "Maniac" shift the answer.

| Label | Bot | Behavior |
| --- | --- | --- |
| Type A | Rock | Never bets; calls only with the King |
| Type B | Nash | Mixed equilibrium strategy (α = 0.2) |
| Type C | Maniac | Bets and calls about 90% of the time with any card |
| Type D | Exploiter | Plays Nash until the learner folds to half its bets (after at least two), then bets every card |
| Type E | Cautious | Bets and calls only with the King |

## Requirements

- Python 3.10 or later
- [Ollama](https://ollama.com) running locally, with the model pulled

```bash
pip install -r requirements.txt     # requests, pandas
ollama pull qwen3.8:27b-mlx         # Apple Silicon; use qwen3.8:27b on NVIDIA GPUs
```

## Usage

```bash
# Build the matches and an example prompt without calling the model
python pilot.py --dry-run

# Full run (temperature 0, top-p 0.95, thinking on by default)
python pilot.py --model qwen3.8:27b-mlx --num-predict 16000 --num-ctx 20000 --out run_think

# Same matches, thinking off
python pilot.py --model qwen3.8:27b-mlx --think false --out run_nothink
```

Main options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen3.8:27b` | Ollama model tag |
| `--matches-per-type` | 3 | Matches per opponent type (3 × 5 = 15 documents) |
| `--repeats` | 1 | Calls per match, to test consistency |
| `--temperature`, `--top-p` | 0, 0.95 | Decoding settings |
| `--num-predict` | 4096 | Maximum output tokens, including thinking |
| `--num-ctx` | 8192 | Context window |
| `--think` | `default` | `true`, `false`, or the model's default |
| `--seed` | 2026 | Seed for both match simulation and decoding |
| `--out` | `pilot_output` | Output folder |

## Output

Each run folder contains:

- `matches/` – the text of every match
- `prompt_example.txt` – one complete prompt
- `run_config.json` – start time, model tag and digest, Ollama version, decoding options, prompt template and type definitions
- `results.csv` – one row per call: raw response, thinking text, stop reason, token counts, latency, parsed probabilities, validity checks (`parse_ok`, `label_valid`, `unique_max`) and scores (`correct`, `p_true`, `log_loss`, `brier`)

Results are saved after every call, so an interrupted run keeps what it finished.

## Pilot runs (October 27, 2026)

All runs used `qwen3.8:27b-mlx` (digest `5642e974`), Ollama 0.40.11, temperature 0, top-p 0.95 and seed 2026, on the same 15 matches.

| Folder | Thinking | Output limit | Parsed | Accuracy | Log loss |
| --- | --- | --- | --- | --- | --- |
| `pilot_output` | On | 4,096 | 9 / 15 | 44% | 1.22 |
| `run2_nothink` | Off | 4,096 | 15 / 15 | 27% | 2.82 |
| `run2_think` | On | 16,000 | 15 / 15 | 47% | 1.33 |

Chance accuracy is 20%, and a uniform guess has a log loss of 1.61. `pilot_nothink` and `pilot_think12k` were stopped after one match and are not part of the analysis.

**Note on definitions.** These runs used the first version of the type definitions, which described Type B as "value-bets the King". The equilibrium bot actually checks the King 40% of the time as Player 1, so some Nash errors come from the prompt, not the model. The current `pilot.py` uses corrected definitions that match the bot code and records `definitions_version` in `run_config.json`. Rerunning with them is the next step.

## Reproducing

Because the matches are seeded and the model is pinned by digest, `python pilot.py` with the settings above rebuilds the same documents and prompt. Outputs may still differ slightly between runs, since local inference at temperature 0 is not guaranteed to be bit-identical.
