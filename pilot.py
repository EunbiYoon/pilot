"""
Assignment Six pilot: opponent-type classification of Kuhn Poker matches with a local LLM (Ollama).

What it does
  1. Simulates best-of-8 Kuhn Poker matches between a fixed learner and five scripted opponent
     bots, so every document has a known ground-truth opponent type.
  2. Writes each match as a plain-text log from the learner's point of view.
  3. Sends every log to an Ollama model through its HTTP API, one request at a time.
  4. Parses the JSON answer and collects everything into a pandas DataFrame (results.csv),
     with a run_config.json that records the prompt, model, digest, settings and date.

Usage
  python pilot.py --dry-run                 # simulate matches and build prompts only, no API calls
  python pilot.py                           # full run with the defaults below
  python pilot.py --model qwen3:8b --matches-per-type 2 --repeats 3 --temperature 0.6

Requirements
  pip install requests pandas
  Ollama running locally (ollama serve) with the model pulled (ollama pull <model>).
"""

import argparse
import hashlib
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen3.8:27b"          # check the exact tag with `ollama list`
DEFAULT_HOST = "http://localhost:11434"
HANDS_PER_MATCH = 8
CARDS = ["Jack", "Queen", "King"]
RANK = {"Jack": 0, "Queen": 1, "King": 2}

# Neutral labels (Assignment 5 showed that names like "Nash" or "Maniac" shift the answer).
# Key = internal bot name, value = label shown to the model.
LABELS = {"Rock": "Type A", "Nash": "Type B", "Maniac": "Type C",
          "Exploiter": "Type D", "Cautious": "Type E"}

# Definitions shown to the model, written to match the bot code below.
DEFINITIONS = {
    "Type A": "very passive: never bets, calls only with the King, otherwise checks or folds.",
    "Type B": "plays the equilibrium strategy: value-bets the King, sometimes bluffs with the Jack, "
              "checks the Queen, and calls with the Queen only some of the time.",
    "Type C": "very aggressive: bets and calls with almost any card.",
    "Type D": "adaptive: starts close to equilibrium, but bluffs with any card once the learner "
              "has folded to its bets often.",
    "Type E": "cautious: bets and calls only with the King, otherwise checks or folds.",
}

PROMPT_TEMPLATE = """Below is the full log of an {n_hands}-hand Kuhn Poker match, written from the learner's point of view. \
Kuhn Poker uses three cards (Jack < Queen < King). Each player antes 1 chip and receives one card. \
Player 1 checks or bets 1 chip. If Player 1 checks, Player 2 checks (showdown) or bets; facing a bet, a player calls or folds. \
The opponent's card is revealed only at showdown.

The opponent is exactly one of these bot types:
{definitions}

Based only on the log, estimate the probability of each type.
Output JSON only. Report each probability with two decimal places (e.g., 0.37), summing to 1:
{{"probs": {{{prob_keys}}}, "prediction": "<one of the type labels>", "evidence": "one sentence"}}

<match>
{match_log}
</match>"""


def build_prompt(match_log: str, label_order: list[str]) -> str:
    definitions = "\n".join(f"- {lab}: {DEFINITIONS[lab]}" for lab in label_order)
    prob_keys = ", ".join(f'"{lab}": p' for lab in label_order)
    return PROMPT_TEMPLATE.format(n_hands=HANDS_PER_MATCH, definitions=definitions,
                                  prob_keys=prob_keys, match_log=match_log)


# ---------------------------------------------------------------------------
# Kuhn Poker simulator
# ---------------------------------------------------------------------------
# Decisions: "open" (Player 1 first action), "after_check" (Player 2 after a check),
# "facing_bet" (either player facing a bet). Each policy returns "bet"/"check" or "call"/"fold".

ALPHA = 0.2  # Nash family parameter, 0 <= alpha <= 1/3


class Bot:
    def __init__(self, rng):
        self.rng = rng

    def observe(self, event):  # used only by the adaptive bot
        pass


class Learner(Bot):
    """Fixed, roughly equilibrium learner so that the opponent is the only thing that varies."""
    def act(self, card, situation):
        r = self.rng.random()
        if situation == "open":
            return "bet" if (card == "King" or (card == "Jack" and r < 1 / 3)) else "check"
        if situation == "after_check":
            return "bet" if (card == "King" or (card == "Jack" and r < 1 / 3)) else "check"
        # facing_bet
        if card == "King":
            return "call"
        if card == "Queen":
            return "call" if r < 1 / 3 else "fold"
        return "fold"


class Cautious(Bot):
    def act(self, card, situation):
        if situation in ("open", "after_check"):
            return "bet" if card == "King" else "check"
        return "call" if card == "King" else "fold"


class Rock(Bot):
    def act(self, card, situation):
        if situation in ("open", "after_check"):
            return "check"
        return "call" if card == "King" else "fold"


class Maniac(Bot):
    def act(self, card, situation):
        r = self.rng.random()
        if situation in ("open", "after_check"):
            return "bet" if r < 0.9 else "check"
        return "call" if r < 0.9 else "fold"


class Nash(Bot):
    def act(self, card, situation, alpha=ALPHA):
        r = self.rng.random()
        if situation == "open":
            p_bet = {"Jack": alpha, "Queen": 0.0, "King": 3 * alpha}[card]
            return "bet" if r < p_bet else "check"
        if situation == "after_check":
            p_bet = {"Jack": 1 / 3, "Queen": 0.0, "King": 1.0}[card]
            return "bet" if r < p_bet else "check"
        # facing_bet: as Player 2 (after open bet) or as Player 1 (after check-bet); same mix used
        p_call = {"Jack": 0.0, "Queen": alpha + 1 / 3, "King": 1.0}[card]
        return "call" if r < p_call else "fold"


class Exploiter(Nash):
    """Plays Nash until the learner has folded to at least half of its bets (min. 2 bets)."""
    def __init__(self, rng):
        super().__init__(rng)
        self.bets_faced_by_learner = 0
        self.learner_folds = 0

    def observe(self, event):
        if event == "learner_faced_bet":
            self.bets_faced_by_learner += 1
        elif event == "learner_folded":
            self.learner_folds += 1

    def exploiting(self):
        n = self.bets_faced_by_learner
        return n >= 2 and self.learner_folds / n >= 0.5

    def act(self, card, situation):
        if self.exploiting() and situation in ("open", "after_check"):
            return "bet"
        return super().act(card, situation)


BOTS = {"Cautious": Cautious, "Maniac": Maniac, "Nash": Nash, "Rock": Rock, "Exploiter": Exploiter}


def play_hand(hand_no, learner, opp, learner_is_p1, rng):
    """Plays one hand; returns (text line, learner chip result)."""
    cards = rng.sample(CARDS, 2)
    l_card, o_card = cards
    p1, p2 = (learner, opp) if learner_is_p1 else (opp, learner)
    c1, c2 = (l_card, o_card) if learner_is_p1 else (o_card, l_card)
    name = {id(learner): "You", id(opp): "Opponent"}
    lines = []

    def say(player, action):
        verb = {"bet": "bet", "check": "checked", "call": "called", "fold": "folded"}[action]
        lines.append(f"{name[id(player)]} {verb}.")

    def facing(player, card):
        if player is learner:
            opp.observe("learner_faced_bet")
        a = player.act(card, "facing_bet")
        if player is learner and a == "fold":
            opp.observe("learner_folded")
        say(player, a)
        return a

    a1 = p1.act(c1, "open")
    say(p1, a1)
    winner, pot_per_player, showdown = None, 1, False
    if a1 == "bet":
        a2 = facing(p2, c2)
        if a2 == "fold":
            winner = p1
        else:
            pot_per_player, showdown = 2, True
    else:
        a2 = p2.act(c2, "after_check")
        say(p2, a2)
        if a2 == "check":
            showdown = True
        else:
            a3 = facing(p1, c1)
            if a3 == "fold":
                winner = p2
            else:
                pot_per_player, showdown = 2, True

    if showdown:
        winner = p1 if RANK[c1] > RANK[c2] else p2
        reveal = f"Showdown: the opponent had the {o_card}."
    else:
        reveal = "No showdown; the opponent's card was not shown."
    result = pot_per_player if winner is learner else -pot_per_player
    seat = "Player 1" if learner_is_p1 else "Player 2"
    outcome = f"You {'won' if result > 0 else 'lost'} {abs(result)} chip{'s' if abs(result) > 1 else ''}."
    text = f"Hand {hand_no}: You are {seat} and hold the {l_card}. " + " ".join(lines) + f" {reveal} {outcome}"
    return text, result


def simulate_match(opp_type, seed):
    rng = random.Random(seed)
    learner, opp = Learner(rng), BOTS[opp_type](rng)
    lines, total = [], 0
    for h in range(1, HANDS_PER_MATCH + 1):
        text, res = play_hand(h, learner, opp, learner_is_p1=(h % 2 == 1), rng=rng)
        lines.append(text)
        total += res
    if total == 0:
        lines.append(f"Final result: you broke even over {HANDS_PER_MATCH} hands.")
    else:
        lines.append(f"Final result: you {'won' if total > 0 else 'lost'} {abs(total)} chips "
                     f"over {HANDS_PER_MATCH} hands.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama client and parsing
# ---------------------------------------------------------------------------

def ollama_info(host, model):
    info = {"ollama_version": None, "model_digest": None}
    try:
        info["ollama_version"] = requests.get(f"{host}/api/version", timeout=10).json().get("version")
        for m in requests.get(f"{host}/api/tags", timeout=10).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                info["model_digest"] = m.get("digest")
    except requests.RequestException as e:
        print(f"Warning: could not query Ollama metadata ({e}).")
    return info


def call_ollama(host, model, prompt, options, think, timeout):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "stream": False, "options": options}
    if think is not None:
        payload["think"] = think
    t0 = time.time()
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data.get("message", {})
    return {"content": msg.get("content", ""), "thinking": msg.get("thinking", ""),
            "latency_s": round(time.time() - t0, 2),
            "eval_count": data.get("eval_count"), "prompt_eval_count": data.get("prompt_eval_count"),
            "done_reason": data.get("done_reason")}


def extract_json(text):
    """Returns the last parseable {...} object in the text, ignoring any <think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in reversed(starts):
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[s:i + 1])
                        if isinstance(obj, dict) and "probs" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def score(obj, true_label, labels):
    """Validates the answer and computes metrics. Mirrors the checks proposed in Assignment 4."""
    out = {"parse_ok": obj is not None, "schema_ok": False, "label_valid": False,
           "unique_max": False, "prob_sum": None, "prediction": None, "argmax": None,
           "p_true": None, "log_loss": None, "brier": None, "correct": None, "evidence": None}
    if obj is None:
        return out
    probs = obj.get("probs", {})
    try:
        p = {lab: float(probs.get(lab, 0.0)) for lab in labels}
    except (TypeError, ValueError):
        return out
    out["schema_ok"] = set(probs.keys()) == set(labels)
    s = sum(p.values())
    out["prob_sum"] = round(s, 4)
    if s > 0:
        p = {k: v / s for k, v in p.items()}  # renormalise for scoring
    for lab in labels:
        out[f"p_{lab.replace(' ', '_')}"] = round(p[lab], 4)
    top = max(p.values())
    out["unique_max"] = sum(1 for v in p.values() if math.isclose(v, top)) == 1
    out["argmax"] = max(p, key=p.get)
    pred = str(obj.get("prediction", "")).strip()
    out["prediction"] = pred
    out["label_valid"] = pred in labels
    out["p_true"] = round(p[true_label], 4)
    out["log_loss"] = round(-math.log(max(p[true_label], 1e-6)), 4)
    out["brier"] = round(sum((p[lab] - (1.0 if lab == true_label else 0.0)) ** 2 for lab in labels), 4)
    final = pred if out["label_valid"] else out["argmax"]
    out["correct"] = final == true_label
    out["evidence"] = obj.get("evidence")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--matches-per-type", type=int, default=3, help="3 x 5 types = 15 documents")
    ap.add_argument("--repeats", type=int, default=1, help="calls per document (use >1 to test consistency)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-predict", type=int, default=4096, help="max output tokens, incl. thinking")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--think", choices=["true", "false", "default"], default="default",
                    help="Ollama thinking mode for reasoning models")
    ap.add_argument("--seed", type=int, default=2026, help="seed for match simulation and decoding")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="pilot_output")
    ap.add_argument("--dry-run", action="store_true", help="simulate and save prompts, no API calls")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "matches").mkdir(parents=True, exist_ok=True)
    labels = sorted(LABELS.values())  # Type A..E, shown in this order
    think = None if args.think == "default" else (args.think == "true")
    options = {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
               "num_predict": args.num_predict, "num_ctx": args.num_ctx}

    # 1. Build the sample
    docs = []
    for opp_type in BOTS:
        for k in range(args.matches_per_type):
            match_seed = int(hashlib.sha256(f"{args.seed}-{opp_type}-{k}".encode()).hexdigest()[:8], 16)
            log = simulate_match(opp_type, match_seed)
            doc_id = f"{opp_type.lower()}_{k + 1:02d}"
            (out / "matches" / f"{doc_id}.txt").write_text(log)
            docs.append({"doc_id": doc_id, "true_type": opp_type, "true_label": LABELS[opp_type],
                         "match_seed": match_seed, "match_log": log})
    example_prompt = build_prompt(docs[0]["match_log"], labels)
    (out / "prompt_example.txt").write_text(example_prompt)

    config = {"run_started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model": args.model, "host": args.host, "options": options, "think": args.think,
              "repeats": args.repeats, "n_documents": len(docs), "hands_per_match": HANDS_PER_MATCH,
              "labels": LABELS, "definitions": DEFINITIONS, "prompt_template": PROMPT_TEMPLATE,
              "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:12],
              "nash_alpha": ALPHA, "dry_run": args.dry_run}
    if not args.dry_run:
        config.update(ollama_info(args.host, args.model))
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Built {len(docs)} documents in {out}/matches. Example prompt: {out}/prompt_example.txt")
    if args.dry_run:
        return

    # 2. Query the model, one document at a time
    rows = []
    for i, d in enumerate(docs, 1):
        prompt = build_prompt(d["match_log"], labels)
        for rep in range(1, args.repeats + 1):
            row = {"doc_id": d["doc_id"], "repeat": rep, "true_type": d["true_type"],
                   "true_label": d["true_label"], "match_seed": d["match_seed"], "model": args.model,
                   "temperature": args.temperature, "top_p": args.top_p, "think": args.think,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                resp = call_ollama(args.host, args.model, prompt, options, think, args.timeout)
                row.update(resp)
                row.update(score(extract_json(resp["content"]), d["true_label"], labels))
                row["error"] = None
            except requests.RequestException as e:
                row.update(score(None, d["true_label"], labels))
                row["error"] = str(e)
            rows.append(row)
            status = "ERROR" if row["error"] else ("ok" if row["correct"] else "wrong" if row["parse_ok"] else "unparsed")
            print(f"[{i}/{len(docs)} rep {rep}] {d['doc_id']}: {status} "
                  f"pred={row.get('prediction')} p_true={row.get('p_true')}")
            # Save after every call so a crash loses nothing
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    # 3. Summary
    df = pd.DataFrame(rows)
    ok = df[df["parse_ok"]]
    print("\n=== Summary ===")
    print(f"Calls: {len(df)} | parsed: {df['parse_ok'].mean():.0%} | valid label: {df['label_valid'].mean():.0%} "
          f"| unique max: {df['unique_max'].mean():.0%} | truncated: {(df['done_reason'] == 'length').mean():.0%}")
    if len(ok):
        print(f"Accuracy: {ok['correct'].mean():.0%} | mean log loss: {ok['log_loss'].mean():.3f} "
              f"| mean Brier: {ok['brier'].mean():.3f}")
        print("\nBy true type:")
        print(ok.groupby("true_type")[["correct", "p_true", "log_loss"]].mean().round(3))
        print("\nConfusion (rows = true, cols = predicted, invalid labels use argmax):")
        pred = ok["prediction"].where(ok["label_valid"], ok["argmax"])
        inv = {v: k for k, v in LABELS.items()}
        print(pd.crosstab(ok["true_type"], pred.map(inv)))
        if args.repeats > 1:
            agree = ok.groupby("doc_id")["argmax"].nunique().eq(1).mean()
            print(f"\nDocuments with the same top type across repeats: {agree:.0%}")
    print(f"\nSaved {out}/results.csv and {out}/run_config.json")


if __name__ == "__main__":
    main()
