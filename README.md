# Grounded Diet Coach

A Streamlit application that accepts meal photos or free-form text logs, links them to a curated nutrition table, and produces transparent dietary recommendations using only open-source models.

## Features

- **Dual input**: upload meal photos (analyzed by open-source ViT/ResNet classifiers such as [`nateraw/foods-101`](https://huggingface.co/nateraw/foods-101) with automatic fallbacks) or paste text describing what you ate.
- **Traceable knowledge base**: every detected food maps to deterministic nutrition facts in `nutrition_db.py`, preventing hallucinated nutrients.
- **India-first coverage**: includes dozens of regional staples (idli, dosa, khichdi, biryani, chole, pav bhaji, mithai, beverages, and more) with macros, targets, and synonyms for reliable matching.
- **Macro & food-group tracking**: aggregates calories, protein, carbs, fat, fiber, plus servings across vegetables, fruits, grains, proteins, fats, and hydration.
- **Evidence-backed recommendations**: suggestions are generated from observed gaps versus targets, with ready-to-eat examples pulled straight from the knowledge base.
- **Manual overrides**: if an item is missed, add it from the curated list, remove mistakes, or reset the log entirely.

## Getting started

```powershell
# Install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Launch Streamlit
streamlit run app.py
```

Open the provided URL in your browser. The sidebar shows instructions and includes a reset button to clear the session log.

## Hallucination safeguards

1. **Open-source deterministic stack**: image understanding relies on a fine-tuned ViT classifier, while text parsing uses rule-based synonym matching.
2. **Confidence gating**: predictions under 30 % are discarded automatically.
3. **Curated nutrition DB**: `nutrition_db.py` is the single source of truth for macros, categories, and micronutrients, so recommendations can always be traced back.
4. **User-in-the-loop controls**: each auto-detected item requires an explicit “Add” action and can be removed later.

## Project structure

```
Gen-AI-LAB/
├── app.py              # Streamlit interface + business logic
├── nutrition_db.py     # Food facts, targets, and lookup helpers
├── requirements.txt    # Python dependencies
└── README.md           # This guide
```

Feel free to expand `nutrition_db.py` with local cuisines, adjust category targets, or integrate additional open-source models as long as they keep outputs grounded in verifiable data.
