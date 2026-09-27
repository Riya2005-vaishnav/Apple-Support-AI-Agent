# AppleSupport AI Support Agent

An AI agent that classifies AppleSupport customer messages (from the Kaggle
"Customer Support on Twitter" dataset), drafts a reply grounded in real past
resolutions, and decides whether to auto-handle or escalate to a human.

Built for the Hiver SDE Intern take-home assignment.

## What this does

For any AppleSupport customer message, the agent:
1. **Classifies** it into one of 7 intents, discovered by clustering real
   customer messages (see `notebooks/` and `src/discover_intents.py`).
2. **Retrieves** the 3 most similar past customer messages and how
   AppleSupport actually replied to them, using a local embedding index.
3. **Drafts a reply**, grounded in those real past resolutions rather than
   generic LLM output.
4. **Decides auto-handle vs. escalate**, with a stated reason, using simple
   explainable rules (see `src/reply_agent.py`).

## Setup (one-time)

1. Clone this repo and `cd` into it.
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Get a free Gemini API key from https://aistudio.google.com/apikey
5. Create a `.env` file in the project root with:
   ```
   GEMINI_API_KEY=your_key_here
   ```
6. Download `twcs.csv` from
   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
   and place it at `data/raw/twcs.csv` (not included in this repo - ~350MB).

## Reproducing the headline results (should take well under 15 minutes)

```
python src/data_prep.py --brand AppleSupport --sample_size 5000
python src/retrieval.py --build
python eval/run_eval.py --n 50
```

- Step 1 reconstructs 5,000 real customer<->brand reply pairs from the raw
  dataset (~1-2 minutes).
- Step 2 builds the local retrieval index over 1,500 of those pairs
  (~20-30 seconds, no API calls, no rate limits - uses a local
  sentence-transformers model).
- Step 3 runs the full agent pipeline plus two baselines against a 50-row
  sample of the golden evaluation set and prints an accuracy comparison
  table. This step DOES call the Gemini API (classification + reply
  drafting for each message) and may take 10-20+ minutes depending on
  free-tier rate limits; it checkpoints progress after every row, so it's
  safe to interrupt and resume.

The full 206-row golden evaluation set is already included at
`eval/golden_set_final.csv` (no need to regenerate it) - see the note below
on how it was built.

## Repo structure

```
hiver-agent/
├── data/
│   ├── raw/                    # twcs.csv goes here (not included)
│   └── processed/               # pairs.csv, retrieval index (generated)
├── src/
│   ├── data_prep.py             # reconstructs customer<->brand pairs
│   ├── discover_intents.py       # clusters messages to find intents
│   ├── intents.py                # intent taxonomy + classifier
│   ├── embed_messages.py         # embeddings for intent discovery
│   ├── retrieval.py              # local embedding index + similarity search
│   ├── reply_agent.py            # full pipeline: classify->retrieve->draft->escalate
│   ├── build_golden_set.py       # stratified sampling for golden set
│   ├── prelabel_golden_set.py    # AI-assisted pre-labeling (see note below)
│   └── ...                       # supporting fix/finalize scripts (see comments)
├── eval/
│   ├── golden_set_final.csv      # the 206-row hand-reviewed golden set
│   ├── run_eval.py               # evaluation harness (pipeline vs baselines)
│   └── judge.py                  # LLM-as-judge for reply quality
├── notebooks/                    # exploration (clustering, intent discovery)
├── report.md                     # full report (framing, results, failure analysis)
├── decision_log.md               # 15 non-obvious decisions and why
└── requirements.txt
```

## Golden evaluation set - how it was sampled and labeled

206 examples were sampled from the 5,000 reconstructed pairs using
keyword-based stratification (`src/build_golden_set.py`) to ensure all 7
intents were represented, rather than a plain random sample (which would
have been ~54% dominated by a single event - see report Section 4).

Labeling used a mixed approach under time constraints: the first ~9 examples
were labeled fully independently by hand. The remaining examples were
pre-labeled by the project's own classifier (`src/prelabel_golden_set.py`)
and then reviewed/corrected by hand; ~11 examples with no AI suggestion
available were labeled fully manually. This is an honest tradeoff between
labeling rigor and time - see `report.md` Section 4 ("What is misleading
about my headline number") for a full discussion of how this affects
interpretation of the reported accuracy.

## Model / provider used

Google Gemini (`gemini-3.6-flash` for classification and reply generation).
Retrieval embeddings use a local `sentence-transformers` model
(`all-MiniLM-L6-v2`) instead of an API, after repeatedly hitting Gemini's
free-tier embedding quota (see decision log entry #10).

## Citations / borrowed material

- Dataset: Kaggle "Customer Support on Twitter"
  (thoughtvector/customer-support-on-twitter).
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face,
  open weights).
- No code was copied from external sources; built from scratch with AI
  coding assistance (Claude), per the assignment's allowance for AI coding
  assistants.