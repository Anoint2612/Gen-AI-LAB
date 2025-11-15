"""Streamlit app: grounded diet coach using open-source models only."""

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from nutrition_db import CATEGORY_TARGETS, FOOD_DB, FoodItem, all_food_items, best_guess

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
IMG_MODEL_CANDIDATES = [
    "nateraw/foods-101",
    "microsoft/resnet-50",
    "google/vit-base-patch16-224",
]
CONFIDENCE_THRESHOLD = 0.30
MAX_PREDICTIONS = 5

MODEL_PRETTY_NAMES = {
    "nateraw/foods-101": "ViT fine-tuned on Food-101",
    "microsoft/resnet-50": "ResNet-50 (ImageNet)",
    "google/vit-base-patch16-224": "ViT Base Patch16",
}

CATEGORY_SUGGESTIONS: Dict[str, List[str]] = {
    "vegetable": ["spinach", "mixed salad", "lentil soup"],
    "fruit": ["berries", "banana", "apple"],
    "whole_grain": ["oats", "quinoa", "brown rice"],
    "lean_protein": ["grilled chicken", "eggs", "greek yogurt"],
    "plant_protein": ["chickpeas", "tofu", "lentil soup"],
    "healthy_fat": ["almonds", "salmon", "paneer"],
    "hydration": ["water", "turmeric milk"],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_classifier():
    """Load an open-source image classifier, trying multiple fallbacks."""
    errors: list[str] = []
    for model_id in IMG_MODEL_CANDIDATES:
        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)
            model.eval()
            return processor, model, model_id
        except Exception as exc:  # pragma: no cover - informative logging
            errors.append(f"{model_id}: {exc}")
            continue
    error_msg = "\n".join(errors) or "Unknown error"
    raise RuntimeError(
        "Unable to load any image classifier.\n" + error_msg
    )


def ensure_session_state():
    if "meal_log" not in st.session_state:
        st.session_state.meal_log = []


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def classify_image(image_bytes: bytes) -> Tuple[List[Dict[str, float]], str]:
    processor, model, model_id = load_classifier()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    topk = torch.topk(probs, k=min(MAX_PREDICTIONS, probs.shape[-1]))
    clean_preds: List[Dict[str, float]] = []
    for score, idx in zip(topk.values[0], topk.indices[0]):
        score_val = float(score.item())
        if score_val < CONFIDENCE_THRESHOLD:
            continue
        label = model.config.id2label.get(idx.item(), str(idx.item()))
        clean_preds.append({
            "label": label,
            "score": score_val,
        })
    return clean_preds, model_id


def foods_from_text(text: str) -> List[str]:
    matches: List[str] = []
    normalized_text = normalize_label(text)
    if not normalized_text:
        return matches
    unique_tokens = set(normalized_text.split())
    for item in FOOD_DB.values():
        for synonym in item.synonyms:
            norm_syn = normalize_label(synonym)
            if not norm_syn:
                continue
            if norm_syn in normalized_text or norm_syn in unique_tokens:
                matches.append(item.key)
                break
    return matches


def add_food_entry(food_key: str, source: str, confidence: float | None, note: str):
    food = FOOD_DB.get(food_key)
    if not food:
        return
    st.session_state.meal_log.append({
        "food_key": food_key,
        "source": source,
        "confidence": confidence,
        "note": note,
    })


def build_summary() -> pd.DataFrame:
    rows = []
    for entry in st.session_state.meal_log:
        food = FOOD_DB[entry["food_key"]]
        rows.append({
            "Food": food.name,
            "Calories": food.calories,
            "Protein (g)": food.protein_g,
            "Carbs (g)": food.carbs_g,
            "Fat (g)": food.fat_g,
            "Fiber (g)": food.fiber_g,
            "Source": entry["source"],
        })
    return pd.DataFrame(rows)


def category_counts() -> Counter:
    counts: Counter = Counter()
    for entry in st.session_state.meal_log:
        food = FOOD_DB[entry["food_key"]]
        for cat in food.categories:
            counts[cat] += 1
    return counts


def macro_totals(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"Calories": 0, "Protein": 0, "Carbs": 0, "Fat": 0, "Fiber": 0}
    return {
        "Calories": float(df["Calories"].sum()),
        "Protein": float(df["Protein (g)"].sum()),
        "Carbs": float(df["Carbs (g)"].sum()),
        "Fat": float(df["Fat (g)"].sum()),
        "Fiber": float(df["Fiber (g)"].sum()),
    }


def recommendation_gaps(cat_counts: Counter) -> Dict[str, int]:
    gaps: Dict[str, int] = {}
    for cat, target in CATEGORY_TARGETS.items():
        current = cat_counts.get(cat, 0)
        if current < target:
            gaps[cat] = target - current
    return gaps


def macro_recommendations(totals: Dict[str, float]) -> List[str]:
    recs: List[str] = []
    if totals["Protein"] < 80:
        recs.append("Aim for ≥80 g protein. Add lean protein such as Greek yogurt, tofu, or grilled chicken.")
    if totals["Fiber"] < 25:
        recs.append("Fiber is below 25 g. Include legumes, oats, or mixed salads to boost gut health.")
    if totals["Calories"] < 1600:
        recs.append("Total calories seem low (<1600 kcal). Ensure next meal contains a balanced plate with whole grains and healthy fats.")
    if totals["Calories"] > 2400:
        recs.append("Calories exceed 2400 kcal. Choose lighter meals with vegetables and lean protein next.")
    return recs


def format_recommendations(gaps: Dict[str, int]) -> List[str]:
    formatted: List[str] = []
    for cat, gap in gaps.items():
        suggestions = ", ".join(CATEGORY_SUGGESTIONS.get(cat, [])[:2])
        formatted.append(
            f"Need {gap} more {cat.replace('_', ' ')} servings. Consider {suggestions}."
        )
    return formatted


def render_log_table():
    if not st.session_state.meal_log:
        st.info("No meals logged yet. Add something from the left panel.")
        return
    for idx, entry in enumerate(st.session_state.meal_log):
        food = FOOD_DB[entry["food_key"]]
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(
                f"**{food.name}** · Source: {entry['source']}"
                + (f" · Confidence: {entry['confidence']:.2f}" if entry["confidence"] else "")
            )
            st.caption(
                f"Macros → {food.protein_g} g protein · {food.fiber_g} g fiber · Categories: {', '.join(food.categories)}"
            )
        with cols[1]:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.meal_log.pop(idx)
                st.experimental_rerun()


# -----------------------------------------------------------------------------
# Streamlit layout
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Diet Coach - Streamlit", layout="wide", page_icon="🥗")
st.title("🥗 Grounded Diet Coach")
st.caption("Open-source food recognition + deterministic nutrition guidance to avoid hallucinations.")

ensure_session_state()

with st.sidebar:
    st.markdown("### How it works")
    st.write(
        "1. Add meals via an image or text description.\n"
        "2. We map foods to a curated nutrition table.\n"
        "3. You get transparent macro counts and evidence-backed suggestions."
    )
    if st.button("Reset log"):
        st.session_state.meal_log = []
        st.experimental_rerun()


log_tab, summary_tab, recs_tab = st.tabs(["Log meals", "Nutrition summary", "Recommendations"])


with log_tab:
    st.header("Capture what you ate")
    left, right = st.columns(2)

    with left:
        uploaded = st.file_uploader("Upload a meal photo", type=["png", "jpg", "jpeg"])
        if uploaded:
            image_bytes = uploaded.read()
            st.image(image_bytes, caption="Your photo", use_container_width=True)
            with st.spinner("Classifying foods…"):
                predictions, model_id = classify_image(image_bytes)
            st.caption(
                f"Image model: {MODEL_PRETTY_NAMES.get(model_id, model_id)}"
            )
            if not predictions:
                st.warning("No confident predictions. Try adding manually.")
            else:
                for pred in predictions:
                    food = best_guess(pred["label"])
                    label_display = pred["label"].replace("_", " ")
                    cols = st.columns([3, 1])
                    cols[0].write(f"**{label_display}** · confidence {pred['score']*100:.1f}%")
                    if food:
                        if cols[1].button("Add", key=f"add_img_{label_display}_{pred['score']:.2f}"):
                            add_food_entry(food.key, "image", pred["score"], label_display)
                            st.success(f"Added {food.name} from photo.")
                    else:
                        cols[1].write("No DB match")

    with right:
        st.subheader("Describe meals in text")
        text_log = st.text_area(
            "List foods eaten in the last 1-2 days",
            placeholder="Example: breakfast of oats with banana, dal rice for lunch, salad and paneer for dinner",
        )
        if st.button("Parse text log"):
            matches = foods_from_text(text_log)
            if not matches:
                st.warning("No matches found. Try using common food names or add manually.")
            else:
                added = []
                for key in matches:
                    add_food_entry(key, "text", None, "text log")
                    added.append(FOOD_DB[key].name)
                st.success("Added: " + ", ".join(added))

        st.divider()
        st.subheader("Add from curated list")
        human_choice = st.selectbox(
            "Quick add",
            options=["- select a food -"] + [item.name for item in all_food_items()],
        )
        if human_choice != "- select a food -" and st.button("Add selected"):
            key = next(k for k, v in FOOD_DB.items() if v.name == human_choice)
            add_food_entry(key, "manual", 1.0, "manual")
            st.success(f"Added {human_choice} manually.")

    st.divider()
    st.subheader("Logged meals")
    render_log_table()


with summary_tab:
    st.header("Nutrition snapshot")
    df = build_summary()
    if df.empty:
        st.info("Log at least one meal to see the summary.")
    else:
        st.dataframe(df, use_container_width=True)
        totals = macro_totals(df)
        st.metric("Calories", f"{totals['Calories']:.0f} kcal")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Protein", f"{totals['Protein']:.1f} g")
        col2.metric("Carbs", f"{totals['Carbs']:.1f} g")
        col3.metric("Fat", f"{totals['Fat']:.1f} g")
        col4.metric("Fiber", f"{totals['Fiber']:.1f} g")


with recs_tab:
    st.header("Next-meal guidance")
    if not st.session_state.meal_log:
        st.info("Add meals first to unlock recommendations.")
    else:
        counts = category_counts()
        gaps = recommendation_gaps(counts)
        totals = macro_totals(build_summary())
        st.subheader("Food group coverage")
        coverage_rows = []
        for cat, target in CATEGORY_TARGETS.items():
            coverage_rows.append({
                "Category": cat.replace("_", " "),
                "Logged": counts.get(cat, 0),
                "Target": target,
            })
        st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True)

        st.subheader("What to eat next")
        rec_items = format_recommendations(gaps) + macro_recommendations(totals)
        if not rec_items:
            st.success("You already have a balanced intake logged. Keep going with similar meals!")
        else:
            for rec in rec_items:
                st.write(f"- {rec}")

        st.caption(
            "All tips are drawn from the curated food table so you can trace every suggestion back to evidence-based entries."
        )
