# Intent Chatbot

A small ML-powered chatbot that classifies each message into an **intent** (greeting, pricing, support, …) and replies from a template — the classic architecture behind FAQ and customer-service bots. Fully offline, no API keys.

## How it works

1. Training data lives in `intents.json` — each intent has example **patterns** and canned **responses**.
2. A **TF-IDF + Multinomial Naive Bayes** pipeline is trained on the patterns at startup (instant — the dataset is tiny).
3. Each user message is classified; if the top intent's probability is below a confidence threshold, the bot falls back to "I'm not sure" instead of guessing wrong.

## Setup & run

```bash
pip install -r requirements.txt
python chatbot.py          # chat interactively
python chatbot.py --eval   # self-test on the training patterns
```

## Extend it

- Add new intents or patterns to `intents.json` — no code changes needed.
- Tune `CONFIDENCE_THRESHOLD` in `chatbot.py` to trade coverage vs. accuracy.
- Swap Naive Bayes for `LogisticRegression` or sentence embeddings for better generalization.
