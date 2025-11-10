# app.py
"""
SmartFood Advisor — Multi-model food recognizer + nutrition estimator + deterministic suggestions.
- Uses 2 image-classification models: general food + Indian-food (both Hugging Face)
- Uses BLIP image captioning (Salesforce/blip-image-captioning-base)
- Optional CLIP text-image matching for label verification
- Nutritionix API integration (natural/nutrients). Fallback deterministic table when API missing.
- Streamlit UI with manual-confirm step to remove hallucinations.
"""
import os
import io
import math
import requests
from PIL import Image
import streamlit as st
from transformers import pipeline
import numpy as np

# ---------- CONFIG ----------
# Put your Nutritionix keys in environment variables or set them here (not recommended for public repo)
from dotenv import load_dotenv
load_dotenv()

CALORIE_NINJAS_API_KEY = os.getenv("CALORIE_NINJAS_API_KEY")
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")


# Model names (Hugging Face). You can swap models to other HF models easily.
IMG_CLASS_GENERAL = "ZachBeesley/food-classifier"
IMG_CLASS_INDIAN = "dwililiya/food101-model-classification"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"  # image captioning

# Confidence thresholds and misc
MIN_CONFIDENCE = 0.25
PREFER_INDIAN_CONF = 0.65  # if Indian model > this, prefer it
PREFER_GENERAL_CONF = 0.65

COMMON_TOPPINGS = [
    "pepperoni","chicken","cheese","mushroom","onion","tomato","basil","olive","spinach",
    "beef","pork","sausage","bacon","shrimp","fish","tofu","egg","avocado","lettuce","rice",
    "beans","lentils","dal","sambar","curry","paneer","masala","potato","aloo","naan","roti",
    "idli","dosa","biryani","chicken","mutton","lamb","fish","prawn","vegetable","paneer","palak",
    "chole","rajma","tikka","kabab","samosa","paratha","ghee","curd","yogurt","pickles"
]

# Fallback nutrition (very approximate per serving)
FALLBACK_NUTRITION = {
    "pizza": {"calories": 285, "protein": 12, "fat": 10, "carbs": 34},
    "pepperoni pizza": {"calories": 320, "protein": 14, "fat": 15, "carbs": 30},
    "chicken pizza": {"calories": 300, "protein": 20, "fat": 12, "carbs": 30},
    "chicken biryani": {"calories": 450, "protein": 20, "fat": 18, "carbs": 50},
    "veg biryani": {"calories": 380, "protein": 7, "fat": 14, "carbs": 58},
    "dosa": {"calories": 168, "protein": 4, "fat": 6, "carbs": 24},
    "idli": {"calories": 39, "protein": 1.6, "fat": 0.2, "carbs": 8.2},
    "salad": {"calories": 120, "protein": 3, "fat": 8, "carbs": 8},
    "burger": {"calories": 354, "protein": 17, "fat": 21, "carbs": 29},
    "pasta": {"calories": 350, "protein": 12, "fat": 10, "carbs": 48},
    "samosa": {"calories": 262, "protein": 5, "fat": 14, "carbs": 26},
    "rajma chawal": {"calories": 420, "protein": 13, "fat": 8, "carbs": 70},
    # add common items you expect in demos
}

# ---------- Utilities & Model Load ----------
@st.cache_resource(show_spinner=False)
def load_pipelines(general_name=IMG_CLASS_GENERAL, indian_name=IMG_CLASS_INDIAN, caption_name=CAPTION_MODEL):
    """Load pipelines (cached by Streamlit)"""
    # image classification pipelines
    try:
        clf_general = pipeline("image-classification", model=general_name)
    except Exception as e:
        clf_general = None
    try:
        clf_indian = pipeline("image-classification", model=indian_name)
    except Exception as e:
        clf_indian = None
    try:
        captioner = pipeline("image-to-text", model=caption_name)
    except Exception as e:
        captioner = None
    return clf_general, clf_indian, captioner

def run_models_on_image(image_bytes):
    """Run all pipelines and return outputs"""
    clf_general, clf_indian, captioner = load_pipelines()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preds_general = []
    preds_indian = []
    caption = ""
    if clf_general:
        try:
            preds_general = clf_general(img)
        except Exception:
            preds_general = []
    if clf_indian:
        try:
            preds_indian = clf_indian(img)
        except Exception:
            preds_indian = []
    if captioner:
        try:
            out = captioner(img)
            caption = out[0].get("generated_text","").lower() if out else ""
        except Exception:
            caption = ""
    return preds_general, preds_indian, caption

def extract_top(preds):
    if not preds:
        return None, 0.0
    top = preds[0]
    return top.get("label","").lower(), float(top.get("score",0.0))

def extract_ingredients(label, caption):
    text = (label or "") + " " + (caption or "")
    text = text.lower()
    found = set()
    for kw in COMMON_TOPPINGS:
        if kw in text:
            found.add(kw)
    return sorted(found)

def make_query_text(dish_label, ingredients):
    parts = []
    if dish_label:
        parts.append(dish_label)
    if ingredients:
        parts.append("with " + ", ".join(ingredients))
    parts.append("1 serving")
    return " ".join(parts)

def calorieninjas_query(query_text):
    """Fetch nutrition info from CalorieNinjas API."""
    if not CALORIE_NINJAS_API_KEY:
        return None

    url = "https://api.calorieninjas.com/v1/nutrition"
    headers = {"X-Api-Key": CALORIE_NINJAS_API_KEY}
    params = {"query": query_text}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            return {
                "calories": item.get("calories"),
                "protein": item.get("protein_g"),
                "fat": item.get("fat_total_g"),
                "carbs": item.get("carbohydrates_total_g"),
                "description": item.get("name", query_text)
            }
    except Exception as e:
        st.error(f"CalorieNinjas API error: {e}")
        return None

    return None


def fallback_estimate(dish_label, ingredients):
    # exact match attempts
    candidates = []
    if dish_label:
        candidates.append(dish_label)
    if ingredients:
        candidates += [dish_label + " " + " ".join(ingredients), " ".join(ingredients)]
    for c in candidates:
        if not c: continue
        key = c.lower().strip()
        if key in FALLBACK_NUTRITION:
            return FALLBACK_NUTRITION[key].copy()
    # partial matching
    if dish_label:
        dl = dish_label.lower()
        for k,v in FALLBACK_NUTRITION.items():
            if k in dl or dl in k:
                return v.copy()
    # last resort average
    return {"calories": 300, "protein": 10, "fat": 12, "carbs": 35}

def suggest_goal_action(nutrition, user_goal="maintain", user_bmi=None):
    cal = float(nutrition.get("calories") or 0)
    protein = float(nutrition.get("protein") or 0)
    fat = float(nutrition.get("fat") or 0)
    carbs = float(nutrition.get("carbs") or 0)
    if cal < 250:
        baseline = "low"
    elif cal < 500:
        baseline = "medium"
    else:
        baseline = "high"
    suggestion = []
    if user_goal == "lose":
        if baseline == "low":
            suggestion.append("Good choice for weight loss (low calories).")
        elif baseline == "medium":
            suggestion.append("Moderate — can fit weight loss if portion controlled.")
        else:
            suggestion.append("High calorie — not ideal for weight loss; consider a smaller portion.")
    elif user_goal == "gain":
        if baseline == "high":
            suggestion.append("Good for weight gain (high calories).")
        elif baseline == "medium":
            suggestion.append("May help for weight gain with additional meals.")
        else:
            suggestion.append("Low calorie — may not be sufficient alone for weight gain.")
    else:
        suggestion.append("For maintenance, moderate portions should be fine.")
        if baseline == "low":
            suggestion.append("You may need extra calories to maintain weight.")
        if baseline == "high":
            suggestion.append("Watch portion size to avoid weight gain.")
    if protein >= 18:
        suggestion.append("High protein — helps satiety and muscle maintenance.")
    elif protein < 8:
        suggestion.append("Low protein — consider adding lean protein.")
    suggestion.append(f"Macros — Calories: {cal:.0f} kcal; Protein: {protein:.0f} g; Fat: {fat:.0f} g; Carbs: {carbs:.0f} g")
    if user_goal == "lose":
        label = "Good" if baseline == "low" else "Moderate" if baseline=="medium" else "Poor"
    elif user_goal == "gain":
        label = "Good" if baseline == "high" else "Moderate" if baseline=="medium" else "Poor"
    else:
        label = "Suitable" if baseline!="high" else "Be careful with portions"
    return label, suggestion

# ---------- Streamlit UI ----------
st.set_page_config(page_title="SmartFood Advisor", layout="centered")
st.title("🍽️ SmartFood Advisor — Multi-model food photo → nutrition → advice")
st.write("Upload a food photo. The app uses multiple vision models and deterministic rules to estimate nutrition and give weight-goal suggestions. This is a demo; accuracy depends on images and model labels.")

with st.sidebar:
    st.header("Settings & Personalize")
    user_goal = st.selectbox("Your goal", ["maintain", "lose", "gain"])
    height_cm = st.number_input("Height (cm) — optional", min_value=50, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg) — optional", min_value=20.0, max_value=300.0, value=70.0)
    use_bmi = st.checkbox("Use BMI for extra hint", value=False)
    auto_ensemble = st.checkbox("Auto ensemble models (recommended)", value=True)
    show_raw = st.checkbox("Show raw predictions", value=False)

uploaded = st.file_uploader("Upload a dish image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Upload an image to start. Try pizza, biryani, dosa, salad.")
    st.stop()

image_bytes = uploaded.read()
st.image(image_bytes, caption="Uploaded image", use_container_width=True)

with st.spinner("Running vision models... (first run may download models and take ~20-40s)"):
    preds_general, preds_indian, caption = run_models_on_image(image_bytes)

# Show raw outputs (optional)
if show_raw:
    st.subheader("Raw model outputs")
    st.write("Caption:", caption or "_none_")
    st.write("General classifier (top-5):")
    if preds_general:
        for p in preds_general[:5]:
            st.write(f"- {p.get('label')}  ({p.get('score'):.3f})")
    else:
        st.write("_No general classifier available or returned no preds._")
    st.write("Indian classifier (top-5):")
    if preds_indian:
        for p in preds_indian[:5]:
            st.write(f"- {p.get('label')}  ({p.get('score'):.3f})")
    else:
        st.write("_No Indian classifier available or returned no preds._")

# Extract top labels + confidences
label_gen, conf_gen = extract_top(preds_general)
label_ind, conf_ind = extract_top(preds_indian)

# Ensemble / selection logic
final_label = None
source = None
if auto_ensemble:
    # If Indian model is confident, prefer it for Indian-sounding labels.
    if conf_ind and conf_ind >= PREFER_INDIAN_CONF:
        final_label = label_ind
        source = "Indian model (high confidence)"
    elif conf_gen and conf_gen >= PREFER_GENERAL_CONF:
        final_label = label_gen
        source = "General model (high confidence)"
    else:
        # fallback: if models agree, pick that. else prefer whichever has higher confidence.
        if label_gen and label_ind and label_gen == label_ind:
            final_label = label_gen
            source = "Both models agree"
        else:
            # pick higher confidence
            if (conf_ind or 0) > (conf_gen or 0):
                final_label = label_ind
                source = "Indian model (higher confidence)"
            else:
                final_label = label_gen
                source = "General model (higher confidence)"
else:
    # let user pick which model to trust
    choice = st.selectbox("Choose which model to prefer", ["Auto (ensemble)", "Prefer Indian model", "Prefer General model"], index=0)
    if choice == "Prefer Indian model":
        final_label = label_ind
        source = "Indian model"
    else:
        final_label = label_gen
        source = "General model"

st.markdown("---")
st.subheader("Detected (initial)")
st.write(f"**Dish (auto):** {final_label or 'unknown'} — source: {source}")
st.write(f"**Caption hint:** {caption or 'none'}")
detected_ings = extract_ingredients(final_label or "", caption or "")
st.write(f"**Detected ingredients/toppings (auto):** {', '.join(detected_ings) if detected_ings else 'none detected'}")

# Manual-confirm / edit step (important to avoid hallucination)
st.markdown("### Confirm or edit before nutrition lookup")
dish_input = st.text_input("Detected dish (edit if wrong)", value=final_label or "")
ing_input = st.text_input("Detected ingredients/toppings (comma separated — edit/add)", value=", ".join(detected_ings))
manual_confirm = st.button("Confirm and fetch nutrition")

if not manual_confirm:
    st.info("Edit the detected dish/ingredients if needed, then press *Confirm and fetch nutrition*.")
    st.stop()

# Build query
user_ingredients = [i.strip() for i in ing_input.split(",") if i.strip()]
query_text = make_query_text(dish_input or "food", user_ingredients)
st.write("Nutritionix query:", query_text)

nutrition = calorieninjas_query(query_text)
used_fallback = False
if nutrition is None:
    used_fallback = True
    st.warning("Nutritionix API not configured or returned no reliable result. Using fallback estimator.")
    fallback = fallback_estimate(dish_input or "", user_ingredients)
    nutrition = {
        "calories": fallback.get("calories"),
        "protein": fallback.get("protein"),
        "fat": fallback.get("fat"),
        "carbs": fallback.get("carbs"),
        "description": dish_input or "food (fallback)"
    }

# Show nutrition
st.subheader("Estimated nutrition (per serving)")
st.write(f"- Calories: {nutrition.get('calories', 'N/A')} kcal")
st.write(f"- Protein: {nutrition.get('protein','N/A')} g")
st.write(f"- Fat: {nutrition.get('fat','N/A')} g")
st.write(f"- Carbs: {nutrition.get('carbs','N/A')} g")
if used_fallback:
    st.caption("Fallback estimates are approximate — add more items to the fallback table for better coverage.")

# BMI if selected
bmi = None
if use_bmi:
    try:
        h = (height_cm or 170)/100.0
        bmi = (weight_kg or 70.0) / (h*h)
        st.write(f"Your BMI: {bmi:.1f}")
    except Exception:
        st.write("BMI calc failed — check height/weight inputs.")

label, suggestions = suggest_goal_action(nutrition, user_goal, bmi)
st.subheader("Recommendation")
st.write(f"**Quick label:** {label}")
for s in suggestions:
    st.write("- " + s)

st.markdown("---")
st.write("⚠️ This demo is deterministic and uses explicit thresholds. It is not a substitute for professional dietary advice.")

# Provide a small 'export' button to copy results as text for submission/screenshots
if st.button("Copy result summary to clipboard (browser feature)"):
    summary = f"Dish: {dish_input}\nIngredients: {', '.join(user_ingredients)}\nCalories: {nutrition.get('calories')} kcal\nProtein: {nutrition.get('protein')} g\nFat: {nutrition.get('fat')} g\nCarbs: {nutrition.get('carbs')} g\nRecommendation: {label}\nNotes: {', '.join(suggestions)}"
    st.write("Summary (copy manually):")
    st.code(summary)
