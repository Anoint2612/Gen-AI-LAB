import io
from PIL import Image
import torch
import streamlit as st
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

# -----------------------------
# Model identifiers (from Hugging Face)
# -----------------------------
IMG_CLASS_GENERAL = "ZachBeesley/food-classifier"  # General food classifier
IMG_CLASS_INDIAN = "microsoft/resnet-50"  # Indian food classifier
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"  # Image captioning model

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_pipelines():
    """Load pipelines (cached by Streamlit)"""
    st.info("Loading AI models... this may take 1–2 minutes the first time ⏳")

    # General food classifier
    try:
        processor_general = AutoImageProcessor.from_pretrained(IMG_CLASS_GENERAL)
        model_general = AutoModelForImageClassification.from_pretrained(IMG_CLASS_GENERAL,from_tf=True)
    except Exception as e:
        st.error(f"Failed to load general food classifier: {e}")
        processor_general, model_general = None, None

    # Indian food classifier
    try:
        clf_indian = pipeline("image-classification", model=IMG_CLASS_INDIAN)
    except Exception as e:
        st.warning(f"Indian classifier not loaded: {e}")
        clf_indian = None

    # Caption model
    try:
        captioner = pipeline("image-to-text", model=CAPTION_MODEL)
    except Exception as e:
        st.warning(f"Caption model not loaded: {e}")
        captioner = None

    return (processor_general, model_general), clf_indian, captioner


# -----------------------------
# Run all models on uploaded image
# -----------------------------
def run_models_on_image(image_bytes):
    (processor_general, model_general), clf_indian, captioner = load_pipelines()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    preds_general, preds_indian, caption = [], [], ""

    # General model
    if processor_general and model_general:
        inputs = processor_general(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model_general(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            topk = torch.topk(probs, k=5)
            labels = [model_general.config.id2label[i.item()] for i in topk.indices[0]]
            scores = [round(s.item(), 3) for s in topk.values[0]]
            preds_general = [{"label": l, "score": s} for l, s in zip(labels, scores)]

    # Indian classifier
    if clf_indian:
        preds_indian = clf_indian(img)

    # Caption model
    if captioner:
        try:
            out = captioner(img)
            caption = out[0].get("generated_text", "").capitalize() if out else ""
        except Exception:
            caption = ""

    return preds_general, preds_indian, caption, img


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🍱 Food Recognition AI", layout="wide", page_icon="🍛")
st.title("🍱 Food Recognition & Captioning App")

st.markdown(
    "Upload a food image to identify dishes and generate a caption using multiple AI models."
)

uploaded_file = st.file_uploader("📸 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    preds_general, preds_indian, caption, img = run_models_on_image(image_bytes)

    st.image(img, caption="Uploaded image", use_container_width=True)

    st.subheader("🍽️ General Food Classification")
    if preds_general:
        for p in preds_general:
            st.write(f"- **{p['label']}** ({p['score']*100:.1f}%)")
    else:
        st.write("No general predictions available.")

    st.subheader("🇮🇳 Indian Food Classification")
    if preds_indian:
        for p in preds_indian:
            st.write(f"- **{p['label']}** ({p['score']*100:.1f}%)")
    else:
        st.write("No Indian-specific predictions available.")

    st.subheader("📝 Caption")
    st.write(caption if caption else "No caption generated.")
else:
    st.info("👆 Upload an image to get started.")
