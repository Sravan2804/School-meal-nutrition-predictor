import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.ops as ops
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# ---------------------------------------------------------
# Configuration & Knowledge Base
# ---------------------------------------------------------
ALLOWED_CLASSES = [
    "sandwich half", "bread slice", "bread roll", "quesadilla",
    "salad", "baby carrots", "apple", "green apple", "orange slice", "orange",
    "grapes", "strawberries", "banana", "boiled egg", "hummus"
]
NON_FOOD_CONTAINERS = ["metal tray", "plastic water bottle"]
CANDIDATE_LABELS = ALLOWED_CLASSES + NON_FOOD_CONTAINERS

# OWL-ViT zero-shot text matching performs significantly better with descriptive
# noun phrases than bare keywords (CLIP embedding alignment).
PROMPT_TEXT = {
    "sandwich half": "half of a sandwich with visible filling",
    "bread slice": "a flat rectangular slice of sandwich bread",
    "bread roll": "a round bread roll",
    "quesadilla": "a folded toasted quesadilla slice",
    "salad": "a mixed green leafy salad",
    "baby carrots": "small orange baby carrots",
    "apple": "a whole red or yellow apple fruit",
    "green apple": "a whole green granny smith apple",
    "orange slice": "a cut slice of orange fruit",
    "orange": "a whole orange fruit",
    "grapes": "a bunch of grapes",
    "strawberries": "fresh red strawberries",
    "banana": "a ripe yellow banana",
    "boiled egg": "a peeled whole boiled egg",
    "hummus": "a container of beige hummus dip",
    "metal tray": "an empty metal compartment lunch tray",
    "plastic water bottle": "a clear plastic water bottle",
}
PROMPT_LABELS = [PROMPT_TEXT.get(c, c) for c in CANDIDATE_LABELS]

# Distinct palette per food group for high visual contrast
CLASS_COLORS = {
    # Breads & Grains (Warm Terracotta / Amber)
    "sandwich half": "#E07A5F",
    "bread slice": "#D4A373",
    "bread roll": "#BC6C25",
    "quesadilla": "#E76F51",
    # Fresh Fruits (Ruby, Citrus & Berry)
    "apple": "#E63946",
    "green apple": "#4EBA6F",
    "orange slice": "#F4A261",
    "orange": "#F39C12",
    "grapes": "#8E44AD",
    "strawberries": "#D81159",
    "banana": "#F1C40F",
    # Vegetables & Greens (Teal & Emerald)
    "salad": "#2A9D8F",
    "baby carrots": "#E67E22",
    # Proteins & Dips (Gold & Warm Saffron)
    "boiled egg": "#F4D03F",
    "hummus": "#C89666",
    # Containers & Utensils
    "metal tray": "#7F8C8D",
    "plastic water bottle": "#3498DB",
}

DEFAULT_TYPICAL_WEIGHT_G = 100.0
DEFAULT_EXPECTED_AREA_RATIO = 0.035

# School Meal Nutritional Standards (Irish HSE & National School Food Standards)
GUIDELINE_PROFILES = {
    "HSE Primary School (Ages 5–11)": {
        "calories": 500.0,
        "cal_min": 450.0,
        "cal_max": 575.0,
        "protein": 15.0,
        "carbs": 65.0,
        "fats": 15.0,
        "fiber_min": 4.0,
        "sodium_max": 400.0,
        "sugar_max": 10.0,
        "badge": "Primary Level (500 kcal Target)",
        "description": "Standard hot lunch benchmarks for primary pupils under HSE & Department of Social Protection Guidelines."
    },
    "HSE Post-Primary School (Ages 12–18)": {
        "calories": 650.0,
        "cal_min": 585.0,
        "cal_max": 745.0,
        "protein": 22.0,
        "carbs": 85.0,
        "fats": 20.0,
        "fiber_min": 6.0,
        "sodium_max": 500.0,
        "sugar_max": 12.0,
        "badge": "Post-Primary Level (650 kcal Target)",
        "description": "Nutrient allowances for adolescent students requiring higher caloric and protein density for growth and learning."
    }
}

SAMPLE_PRESETS = {
    "None (Use upload or select below)": None,
    "🥪 Sample 1: Primary School Lunch Tray (Bread, Salad, Apple)": "Images/school_lunch_tray.jpg",
    "🥗 Sample 2: Mediterranean Roll & Orange Lunch": "Images/images.jpg",
    "🍓 Sample 3: Fresh Bento Box (Berries, Carrots & Sandwich)": "Images/Lunchboxes.jpg",
}


# ---------------------------------------------------------
# Helpers & Model Caching
# ---------------------------------------------------------
@st.cache_data
def load_nutrition_db():
    try:
        with open("nutrition_db.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("nutrition_db.json not found in root directory. Using empty database.")
        return {}


@st.cache_resource(show_spinner=False)
def load_model():
    """Load OWL-ViT zero-shot detector with offline cache preference."""
    try:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", local_files_only=True)
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", local_files_only=True)
    except Exception:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    return processor, model


def estimate_item_weight(class_name, box_area, image_area, nutrition_db):
    """
    Reference-anchored serving weight estimation.
    Prevents camera perspective and tray depth from arbitrarily inflating portions.
    Damps bounding box ratio with a bounded 0.6x - 1.4x adjustment around reference weight.
    """
    clean_key = str(class_name).strip().lower()
    item_data = nutrition_db.get(clean_key, {})

    reference_weight = float(item_data.get("typical_weight_g", DEFAULT_TYPICAL_WEIGHT_G))
    expected_ratio = float(item_data.get("expected_area_ratio", DEFAULT_EXPECTED_AREA_RATIO))

    relative_ratio = (box_area / image_area) if image_area > 0 else 0.0
    size_factor = (relative_ratio / expected_ratio) if expected_ratio > 0 else 1.0
    size_factor = max(0.6, min(1.4, size_factor))

    estimated_weight_g = reference_weight * size_factor
    return round(max(10.0, min(400.0, estimated_weight_g)), 1)


def calculate_item_nutrition(item_key, weight_g, nutrition_db):
    clean_key = str(item_key).strip().lower()

    if clean_key not in nutrition_db:
        return {
            "Item": str(item_key).title(),
            "Serving Weight (g)": float(weight_g),
            "Calories (kcal)": 0.0,
            "Protein (g)": 0.0,
            "Carbs (g)": 0.0,
            "Fats (g)": 0.0,
            "Fiber (g)": 0.0,
            "Sodium (mg)": 0.0,
            "Sugar (g)": 0.0,
            "Allergens": "None"
        }

    data = nutrition_db[clean_key]
    factor = float(weight_g) / 100.0

    cals = data.get("calories_kcal", data.get("calories", 0.0))
    prot = data.get("protein_g", data.get("protein", 0.0))
    carbs = data.get("carbs_g", data.get("carbs", 0.0))
    fats = data.get("fat_g", data.get("fats_g", data.get("fats", 0.0)))
    fiber = data.get("fiber_g", data.get("fiber", 0.0))
    sodium = data.get("sodium_mg", data.get("sodium", 0.0))
    sugar = data.get("sugar_g", data.get("sugar", 0.0))

    allergens = data.get("allergens", [])
    allergens_str = ", ".join(allergens) if isinstance(allergens, list) and allergens else "None"

    return {
        "Item": data.get("name", clean_key.title()),
        "Serving Weight (g)": round(float(weight_g), 1),
        "Calories (kcal)": round(float(cals) * factor, 1),
        "Protein (g)": round(float(prot) * factor, 1),
        "Carbs (g)": round(float(carbs) * factor, 1),
        "Fats (g)": round(float(fats) * factor, 1),
        "Fiber (g)": round(float(fiber) * factor, 1),
        "Sodium (mg)": round(float(sodium) * factor, 1),
        "Sugar (g)": round(float(sugar) * factor, 1),
        "Allergens": allergens_str
    }


def draw_annotated_tray(image, detections, class_colors, show_boxes=True, show_scores=True):
    """Draw clean, high-contrast bounding boxes with rounded pill labels."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    img_w, img_h = annotated.size

    for det in detections:
        class_name = det.get("class_name", "")
        box = det.get("box", [0, 0, 0, 0])
        score = det.get("score", 0.0)
        color = class_colors.get(class_name, "#10B981")

        x1, y1, x2, y2 = box

        if show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = class_name.title()
        if show_scores:
            label_text += f" · {int(score * 100)}%"

        try:
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw = len(label_text) * 8
            th = 14

        pad_x, pad_y = 6, 3
        if y1 >= (th + pad_y * 2 + 5):
            pill_y1 = y1 - (th + pad_y * 2 + 2)
            pill_y2 = y1 - 2
        else:
            pill_y1 = y1 + 3
            pill_y2 = y1 + 3 + th + pad_y * 2

        pill_x1 = max(2, min(x1, img_w - tw - pad_x * 2 - 4))
        pill_x2 = pill_x1 + tw + pad_x * 2

        draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=4, fill=color)
        draw.text((pill_x1 + pad_x, pill_y1 + pad_y), label_text, fill="white", font=font)

    return annotated


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="NutriTray AI · School Meal Nutrition & Compliance",
        page_icon="🍱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Clean, adaptive styling (works across light & dark themes seamlessly)
    st.markdown("""
        <style>
        /* Card metrics */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.06));
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #10B981;
            transform: translateY(-1px);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            opacity: 0.85;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
        }
        
        /* Modern banner styling */
        .status-card {
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 20px;
            border-left: 5px solid;
        }
        .status-card-pass {
            background-color: rgba(16, 185, 129, 0.12);
            border-left-color: #10B981;
        }
        .status-card-warning {
            background-color: rgba(245, 158, 11, 0.12);
            border-left-color: #F59E0B;
        }
        .status-card-danger {
            background-color: rgba(239, 68, 68, 0.12);
            border-left-color: #EF4444;
        }
        
        /* Badge pills */
        .pill-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 4px;
        }
        .pill-allergen {
            background-color: rgba(239, 68, 68, 0.15);
            color: #DC2626;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .pill-safe {
            background-color: rgba(16, 185, 129, 0.15);
            color: #059669;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .pill-guideline {
            background-color: rgba(59, 130, 246, 0.15);
            color: #2563EB;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # ------------------ SIDEBAR CONTROLS ------------------
    with st.sidebar:
        st.markdown("### 🍱 NutriTray AI")
        st.caption("Zero-Shot Cafeteria Tray Auditing & HSE Standards Compliance")
        st.divider()

        st.subheader("1. Meal Image Source")
        sample_choice = st.selectbox(
            "Quick Sample Tray Presets",
            options=list(SAMPLE_PRESETS.keys()),
            key="preset_select",
            help="Select one of the pre-loaded cafeteria meal images to immediately test the system."
        )

        uploaded_file = st.file_uploader(
            "Or Upload Tray Photo",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a top-down or 45-degree photo of a school meal tray."
        )

        st.divider()
        st.subheader("2. Guideline & Demographics")
        guideline_name = st.selectbox(
            "School Nutrition Standard",
            options=list(GUIDELINE_PROFILES.keys()),
            index=0,
            help="Irish HSE School Food Policy targets for Primary vs Post-Primary pupils."
        )
        active_guideline = GUIDELINE_PROFILES[guideline_name]

        st.divider()
        st.subheader("3. Detection & Portion Calibration")
        conf_threshold = st.slider(
            "Model Sensitivity (Confidence)",
            min_value=0.05,
            max_value=0.45,
            value=0.12,
            step=0.01,
            help="Lower values detect partially occluded items; higher values enforce strict confidence."
        )

        weight_mode = st.radio(
            "Serving Portion Calculation",
            options=["Adaptive (Size-Adjusted from Bounding Box)", "Standard Reference Serving (Fixed)"],
            index=0,
            help="Adaptive mode applies bounded scaling (0.6x-1.4x) relative to expected tray area."
        )
        use_fixed_weight = (weight_mode == "Standard Reference Serving (Fixed)")

        st.divider()
        with st.expander("ℹ️ About the Standards"):
            st.markdown(f"""
            **{guideline_name}**  
            {active_guideline['description']}
            
            - **Target Energy:** {int(active_guideline['calories'])} kcal (Range: {int(active_guideline['cal_min'])}-{int(active_guideline['cal_max'])})
            - **Min Protein:** {int(active_guideline['protein'])}g
            - **Min Fiber:** {int(active_guideline['fiber_min'])}g
            - **Max Sodium:** {int(active_guideline['sodium_max'])}mg
            - **Max Free Sugar:** {int(active_guideline['sugar_max'])}g
            """)

    nutrition_db = load_nutrition_db()

    # Determine Active Image Path / Source
    active_image = None
    image_source_id = None

    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file).convert("RGB")
            image_source_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
        except Exception as e:
            st.error(f"Could not load uploaded image: {e}")
    elif SAMPLE_PRESETS.get(sample_choice):
        sample_path = SAMPLE_PRESETS[sample_choice]
        if os.path.exists(sample_path):
            try:
                active_image = Image.open(sample_path).convert("RGB")
                image_source_id = f"sample_{sample_path}"
            except Exception as e:
                st.error(f"Could not load sample image '{sample_path}': {e}")
        else:
            st.error(f"Sample file not found at: {sample_path}")

    # ------------------ EMPTY STATE (WELCOME SCREEN) ------------------
    if active_image is None:
        st.markdown("""
        <div style="padding: 24px 0 10px 0;">
            <span class="pill-badge pill-guideline">HSE Healthy Eating & School Food Standards 2024</span>
            <span class="pill-badge pill-safe">Open-Vocabulary Vision AI</span>
            <h1 style="margin-top: 8px; margin-bottom: 4px;">School Meal Nutrition & Compliance Analyzer</h1>
            <p style="font-size: 1.1rem; opacity: 0.85; max-width: 850px;">
                Automated zero-shot vision pipeline for school cafeteria tray auditing, portion estimation, 
                macronutrient tracking, and real-time dietary guideline verification.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🚀 Quick Start: Try a Sample Meal Tray")
        st.write("Click any sample cafeteria tray below to test detection and compliance instantly:")

        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.image("Images/school_lunch_tray.jpg", use_container_width=True)
            st.markdown("**Primary School Meal Tray**")
            st.caption("Standard primary lunch: Bread slices, tossed salad, and fresh apple.")
            if st.button("Load Tray 1", key="btn_sample_1", use_container_width=True):
                st.session_state["preset_select"] = "🥪 Sample 1: Primary School Lunch Tray (Bread, Salad, Apple)"
                st.rerun()

        with col_s2:
            st.image("Images/images.jpg", use_container_width=True)
            st.markdown("**Fresh Roll & Orange Lunch**")
            st.caption("Balanced tray: Whole bread roll, mixed garden salad, and sliced oranges.")
            if st.button("Load Tray 2", key="btn_sample_2", use_container_width=True):
                st.session_state["preset_select"] = "🥗 Sample 2: Mediterranean Roll & Orange Lunch"
                st.rerun()

        with col_s3:
            st.image("Images/Lunchboxes.jpg", use_container_width=True)
            st.markdown("**Fruit & Veg Bento Box**")
            st.caption("Packed lunch: Fresh strawberries, baby carrots, and sandwich halves.")
            if st.button("Load Tray 3", key="btn_sample_3", use_container_width=True):
                st.session_state["preset_select"] = "🍓 Sample 3: Fresh Bento Box (Berries, Carrots & Sandwich)"
                st.rerun()

        st.divider()
        st.markdown("### 📋 How the Pipeline Works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. Zero-Shot Vision")
            st.write("Utilizes Google OWL-ViT (ViT-B/32) with descriptive text embeddings to detect open-vocabulary food items without custom re-training.")
        with c2:
            st.markdown("#### 2. Damped Portion Modeling")
            st.write("Anchors serving weights to reference portions while dynamically scaling for visible tray geometry to prevent camera-depth distortion.")
        with c3:
            st.markdown("#### 3. HSE Compliance Audit")
            st.write("Evaluates energy density, sodium ceiling, free sugars, dietary fiber, and flags common EU/FDA allergens automatically.")
        return

    # ------------------ INFERENCE & SESSION STATE ------------------
    processor, model = load_model()

    if "last_image_source_id" not in st.session_state:
        st.session_state["last_image_source_id"] = None
    if "detections" not in st.session_state:
        st.session_state["detections"] = []
    if "debug_scores" not in st.session_state:
        st.session_state["debug_scores"] = {}
    if "confirmed_meal" not in st.session_state:
        st.session_state["confirmed_meal"] = {}

    img_w, img_h = active_image.size
    image_area = img_w * img_h

    # Re-run inference only if a new image was loaded or threshold changed significantly
    if st.session_state["last_image_source_id"] != image_source_id:
        with st.spinner("Analyzing tray contents with zero-shot OWL-ViT..."):
            inputs = processor(text=[PROMPT_LABELS], images=active_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[img_h, img_w]])
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
            )[0]

            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]

            # Non-Maximum Suppression to remove overlapping duplicate boxes
            keep_indices = ops.nms(boxes, scores, iou_threshold=0.38)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

            detected_items = []
            for box, score, label in zip(boxes, scores, labels):
                class_name = CANDIDATE_LABELS[label.item()]
                box_coords = [round(i, 1) for i in box.tolist()]
                x1, y1, x2, y2 = box_coords
                b_area = (x2 - x1) * (y2 - y1)
                est_wt = estimate_item_weight(class_name, b_area, image_area, nutrition_db)

                detected_items.append({
                    "class_name": class_name,
                    "score": float(score.item()),
                    "box": box_coords,
                    "est_weight_g": est_wt
                })

            st.session_state["detections"] = detected_items
            st.session_state["last_image_source_id"] = image_source_id

            # Raw score logging for transparency
            debug_results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.0
            )[0]
            raw_scores = {}
            for score, label in zip(debug_results["scores"], debug_results["labels"]):
                cls = CANDIDATE_LABELS[label.item()]
                s = float(score.item())
                if cls not in raw_scores or s > raw_scores[cls]:
                    raw_scores[cls] = s
            st.session_state["debug_scores"] = raw_scores

            # Initialize confirmed meal state from detections
            initial_meal = {}
            for d in detected_items:
                c = d["class_name"]
                if c in ALLOWED_CLASSES:
                    if c not in initial_meal:
                        initial_meal[c] = {"qty": 0, "weights": []}
                    initial_meal[c]["qty"] += 1
                    initial_meal[c]["weights"].append(d["est_weight_g"])

            meal_state = {}
            for c, val in initial_meal.items():
                ref_wt = float(nutrition_db.get(c, {}).get("typical_weight_g", DEFAULT_TYPICAL_WEIGHT_G))
                avg_wt = sum(val["weights"]) / len(val["weights"]) if val["weights"] else ref_wt
                meal_state[c] = {
                    "qty": val["qty"],
                    "unit_weight": round(ref_wt if use_fixed_weight else avg_wt, 1)
                }
            st.session_state["confirmed_meal"] = meal_state

    # ------------------ MEAL DATA AGGREGATION ------------------
    # Gather active meal items from session state
    current_meal = st.session_state.get("confirmed_meal", {})

    aggregated_rows = []
    detected_allergens = set()

    for food_name, specs in current_meal.items():
        qty = specs.get("qty", 1)
        unit_wt = specs.get("unit_weight", DEFAULT_TYPICAL_WEIGHT_G)
        if qty > 0 and unit_wt > 0:
            total_wt = qty * unit_wt
            nutr = calculate_item_nutrition(food_name, total_wt, nutrition_db)
            nutr["Quantity"] = qty
            nutr["Unit Weight (g)"] = unit_wt
            aggregated_rows.append(nutr)

            food_db_entry = nutrition_db.get(food_name.strip().lower(), {})
            for a in food_db_entry.get("allergens", []):
                detected_allergens.add(a)

    if aggregated_rows:
        df_meal = pd.DataFrame(aggregated_rows)
        total_cals = float(df_meal["Calories (kcal)"].sum())
        total_prot = float(df_meal["Protein (g)"].sum())
        total_carbs = float(df_meal["Carbs (g)"].sum())
        total_fats = float(df_meal["Fats (g)"].sum())
        total_fiber = float(df_meal["Fiber (g)"].sum())
        total_sodium = float(df_meal["Sodium (mg)"].sum())
        total_sugar = float(df_meal["Sugar (g)"].sum())
    else:
        df_meal = pd.DataFrame()
        total_cals = total_prot = total_carbs = total_fats = total_fiber = total_sodium = total_sugar = 0.0

    # ------------------ HEADER & EXECUTIVE SUMMARY ------------------
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
        <div>
            <h2 style="margin: 0; padding: 0;">Tray Inspection & Nutrition Report</h2>
            <div style="margin-top: 4px;">
                <span class="pill-badge pill-guideline">{guideline_name}</span>
                <span class="pill-badge pill-safe">Model: Google OWL-ViT Zero-Shot</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compliance Evaluation
    status_type = "pass"
    compliance_alerts = []

    if total_cals > active_guideline["cal_max"]:
        status_type = "danger"
        compliance_alerts.append(f"Caloric density ({total_cals:.0f} kcal) exceeds maximum guideline threshold ({active_guideline['cal_max']:.0f} kcal).")
    elif total_cals < active_guideline["cal_min"] and total_cals > 0:
        status_type = "warning"
        compliance_alerts.append(f"Caloric provision ({total_cals:.0f} kcal) is below the recommended minimum ({active_guideline['cal_min']:.0f} kcal).")

    if total_sodium > active_guideline["sodium_max"]:
        status_type = "warning" if status_type != "danger" else "danger"
        compliance_alerts.append(f"Sodium content ({total_sodium:.0f} mg) exceeds the {active_guideline['sodium_max']:.0f} mg upper limit.")

    if total_sugar > active_guideline["sugar_max"]:
        status_type = "warning" if status_type != "danger" else "danger"
        compliance_alerts.append(f"Free sugars ({total_sugar:.1f} g) exceed the {active_guideline['sugar_max']:.0f} g guideline ceiling.")

    if total_fiber < active_guideline["fiber_min"]:
        status_type = "warning" if status_type != "danger" else "danger"
        compliance_alerts.append(f"Dietary fiber ({total_fiber:.1f} g) is below the minimum recommendation of {active_guideline['fiber_min']:.0f} g.")

    # Executive Banner Render
    if not aggregated_rows:
        st.warning("⚠️ No active food items registered on this tray. Please confirm or add items in Tab 1.")
    elif status_type == "pass":
        st.markdown(f"""
        <div class="status-card status-card-pass">
            <h4 style="margin: 0 0 4px 0; color: #059669;">🟢 Fully Compliant with {guideline_name}</h4>
            <div style="font-size: 0.95rem;">
                This meal tray meets energy targets ({total_cals:.0f} kcal), delivers adequate protein ({total_prot:.1f}g), 
                and keeps sodium ({total_sodium:.0f}mg) and sugars ({total_sugar:.1f}g) strictly within HSE school food thresholds.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        card_class = "status-card-danger" if status_type == "danger" else "status-card-warning"
        status_title = "🔴 Nutritional Guideline Breach" if status_type == "danger" else "🟡 Nutritional Attention Required"
        alerts_html = "".join([f"<li>{a}</li>" for a in compliance_alerts])
        st.markdown(f"""
        <div class="status-card {card_class}">
            <h4 style="margin: 0 0 4px 0;">{status_title}</h4>
            <ul style="margin: 4px 0 0 0; padding-left: 20px; font-size: 0.93rem;">
                {alerts_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Allergen Warning Bar
    if detected_allergens:
        allergen_tags = "".join([f'<span class="pill-badge pill-allergen">⚠️ {a}</span>' for a in sorted(detected_allergens)])
        st.markdown(f"""
        <div style="margin-bottom: 16px; padding: 10px 14px; background: rgba(239, 68, 68, 0.08); border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.2);">
            <strong style="font-size: 0.88rem; color: #DC2626;">Allergen Notice (EU FIC 1169/2011):</strong>
            <div style="margin-top: 4px;">{allergen_tags}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-bottom: 16px; padding: 8px 14px; background: rgba(16, 185, 129, 0.08); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.2);">
            <span style="font-size: 0.88rem; color: #059669;">✅ <strong>Allergens:</strong> No major declared allergens identified in current items.</span>
        </div>
        """, unsafe_allow_html=True)

    # ------------------ TOP KPI METRICS ROW ------------------
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

    target_cal = active_guideline["calories"]
    target_prot = active_guideline["protein"]
    target_carbs = active_guideline["carbs"]
    target_fats = active_guideline["fats"]
    min_fiber = active_guideline["fiber_min"]
    max_sodium = active_guideline["sodium_max"]
    max_sugar = active_guideline["sugar_max"]

    k1.metric("Energy", f"{total_cals:.0f} kcal", f"{int((total_cals/target_cal)*100)}% RDA", delta_color="off")
    k2.metric("Protein", f"{total_prot:.1f} g", f"{int((total_prot/target_prot)*100)}% target", delta_color="off")
    k3.metric("Carbs", f"{total_carbs:.1f} g", f"{int((total_carbs/target_carbs)*100)}% target", delta_color="off")
    k4.metric("Total Fats", f"{total_fats:.1f} g", f"{int((total_fats/target_fats)*100)}% target", delta_color="off")
    k5.metric("Fiber", f"{total_fiber:.1f} g", f"{'Meets' if total_fiber >= min_fiber else 'Low'} (>={min_fiber:.0f}g)")
    k6.metric("Sodium", f"{total_sodium:.0f} mg", f"{'OK' if total_sodium <= max_sodium else 'High'} (<={max_sodium:.0f}mg)")
    k7.metric("Free Sugar", f"{total_sugar:.1f} g", f"{'OK' if total_sugar <= max_sugar else 'High'} (<={max_sugar:.0f}g)")

    st.write("")

    # ------------------ MAIN TABS ------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Tray Inspection & Portion Adjuster",
        "📊 Dietary Analytics & Energy Split",
        "📋 Official Procurement Audit & Export",
        "🔬 Model Diagnostics & Calibration"
    ])

    # --- TAB 1: Visual Detection & Interactive Portion Editor ---
    with tab1:
        c_left, c_right = st.columns([1.15, 1.0])

        with c_left:
            st.markdown("#### 🔍 Visual Bounding Box Inspection")
            col_opt1, col_opt2 = st.columns(2)
            show_boxes = col_opt1.checkbox("Show Bounding Boxes", value=True)
            show_scores = col_opt2.checkbox("Show Confidence Scores", value=True)

            annotated_img = draw_annotated_tray(
                active_image,
                st.session_state["detections"],
                CLASS_COLORS,
                show_boxes=show_boxes,
                show_scores=show_scores
            )
            st.image(annotated_img, use_container_width=True, caption=f"Zero-shot detection overlay ({len(st.session_state['detections'])} candidates identified)")

        with c_right:
            st.markdown("#### ✏️ Meal Portions & Overrides")
            st.caption("Review auto-detected items, adjust portions, or add cafeteria items below:")

            # Add / Remove item multiselect
            all_db_items = sorted(list(nutrition_db.keys()))
            currently_selected = list(st.session_state["confirmed_meal"].keys())

            new_selected = st.multiselect(
                "Active Items on Tray",
                options=all_db_items,
                default=currently_selected,
                format_func=lambda x: f"{x.title()} ({nutrition_db.get(x, {}).get('typical_weight_g', 100)}g ref)"
            )

            # Sync additions/removals with session state
            updated_meal = {}
            for item in new_selected:
                if item in st.session_state["confirmed_meal"]:
                    updated_meal[item] = st.session_state["confirmed_meal"][item]
                else:
                    ref_w = float(nutrition_db.get(item, {}).get("typical_weight_g", DEFAULT_TYPICAL_WEIGHT_G))
                    updated_meal[item] = {"qty": 1, "unit_weight": ref_w}
            st.session_state["confirmed_meal"] = updated_meal

            st.write("")
            # Portion adjustment cards
            for item_name in new_selected:
                specs = st.session_state["confirmed_meal"][item_name]
                item_info = nutrition_db.get(item_name, {})
                cals_per_100g = item_info.get("calories_kcal", item_info.get("calories", 0.0))

                with st.container():
                    st.markdown(f"**{item_name.title()}**")
                    col_q, col_w, col_sub = st.columns([1, 1.2, 1.2])

                    new_q = col_q.number_input(
                        "Quantity",
                        min_value=0,
                        max_value=10,
                        value=int(specs["qty"]),
                        step=1,
                        key=f"q_{item_name}"
                    )
                    new_w = col_w.number_input(
                        "Unit Weight (g)",
                        min_value=5.0,
                        max_value=800.0,
                        value=float(specs["unit_weight"]),
                        step=5.0,
                        key=f"w_{item_name}"
                    )

                    sub_wt = new_q * new_w
                    sub_cal = (sub_wt / 100.0) * cals_per_100g
                    col_sub.metric("Total", f"{sub_cal:.0f} kcal", f"{sub_wt:.0f} g")

                    st.session_state["confirmed_meal"][item_name]["qty"] = new_q
                    st.session_state["confirmed_meal"][item_name]["unit_weight"] = new_w
                    st.divider()

    # --- TAB 2: Dietary Analytics & Energy Split ---
    with tab2:
        if total_cals > 0:
            c_donut, c_bench = st.columns([1.0, 1.2])

            with c_donut:
                st.markdown("#### ⚡ Caloric Energy Contribution")
                cal_prot = total_prot * 4
                cal_carbs = total_carbs * 4
                cal_fats = total_fats * 9

                macro_df = pd.DataFrame({
                    "Macronutrient": ["Protein (4 kcal/g)", "Carbohydrates (4 kcal/g)", "Fats (9 kcal/g)"],
                    "Calories": [cal_prot, cal_carbs, cal_fats],
                    "Grams": [total_prot, total_carbs, total_fats]
                })

                fig_donut = px.pie(
                    macro_df,
                    names="Macronutrient",
                    values="Calories",
                    hole=0.55,
                    color="Macronutrient",
                    color_discrete_map={
                        "Protein (4 kcal/g)": "#3B82F6",
                        "Carbohydrates (4 kcal/g)": "#10B981",
                        "Fats (9 kcal/g)": "#F59E0B"
                    }
                )
                fig_donut.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>%{value:.1f} kcal (%{percent})<extra></extra>"
                )
                fig_donut.update_layout(
                    annotations=[dict(
                        text=f"<b>{int(total_cals)}</b><br><span style='font-size:12px;'>Total kcal</span>",
                        x=0.5, y=0.5, font_size=16, showarrow=False
                    )],
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=290,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_donut, use_container_width=True)

                pct_prot_energy = (cal_prot / total_cals) * 100
                st.caption(f"**Energy Distribution:** Protein accounts for **{pct_prot_energy:.1f}%** of total energy (HSE target: 12%–20%).")

            with c_bench:
                st.markdown("#### 🎯 Guideline Benchmark Progress (% of Standard)")

                benchmark_data = [
                    {"Nutrient": "Energy (Target)", "Pct": round((total_cals / target_cal) * 100, 1), "Label": f"{total_cals:.0f} / {target_cal:.0f} kcal"},
                    {"Nutrient": "Protein (Min Target)", "Pct": round((total_prot / target_prot) * 100, 1), "Label": f"{total_prot:.1f} / {target_prot:.0f} g"},
                    {"Nutrient": "Dietary Fiber (Min Target)", "Pct": round((total_fiber / min_fiber) * 100, 1), "Label": f"{total_fiber:.1f} / {min_fiber:.0f} g"},
                    {"Nutrient": "Sodium (Max Ceiling)", "Pct": round((total_sodium / max_sodium) * 100, 1), "Label": f"{total_sodium:.0f} / {max_sodium:.0f} mg"},
                    {"Nutrient": "Free Sugar (Max Ceiling)", "Pct": round((total_sugar / max_sugar) * 100, 1), "Label": f"{total_sugar:.1f} / {max_sugar:.0f} g"},
                ]
                df_bench = pd.DataFrame(benchmark_data)

                # Color logic: Green if healthy, Amber if warning, Red if high breach
                bar_colors = []
                for b in benchmark_data:
                    if "Max" in b["Nutrient"]:
                        bar_colors.append("#EF4444" if b["Pct"] > 100 else "#10B981")
                    else:
                        bar_colors.append("#10B981" if b["Pct"] >= 90 else "#F59E0B")

                fig_bar = go.Figure(go.Bar(
                    x=df_bench["Pct"],
                    y=df_bench["Nutrient"],
                    orientation="h",
                    text=df_bench["Label"],
                    textposition="inside",
                    marker=dict(color=bar_colors),
                    hovertemplate="<b>%{y}</b>: %{x}% of standard (%{text})<extra></extra>"
                ))
                fig_bar.add_vline(x=100, line_dash="dash", line_color="#94A3B8", annotation_text="100% Target", annotation_position="top right")
                fig_bar.update_layout(
                    xaxis=dict(title="% of Guideline Target / Ceiling", range=[0, max(130.0, float(df_bench["Pct"].max() + 15))]),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=20, b=10, l=10, r=20),
                    height=290,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No active meal data to chart. Please confirm food items in Tab 1.")

    # --- TAB 3: Procurement Audit & Export ---
    with tab3:
        st.markdown("#### 📋 Detailed Food Item Procurement Breakdown")

        if not df_meal.empty:
            display_cols = ["Item", "Quantity", "Serving Weight (g)", "Calories (kcal)", "Protein (g)", "Carbs (g)", "Fats (g)", "Fiber (g)", "Sodium (mg)", "Sugar (g)", "Allergens"]
            st.dataframe(df_meal[display_cols], use_container_width=True, hide_index=True)

            st.write("")
            st.markdown("#### 📑 School Food Standards Compliance Checklist")

            checklist = [
                {"Standard": "Total Caloric Energy", "Requirement": f"{active_guideline['cal_min']:.0f} – {active_guideline['cal_max']:.0f} kcal", "Actual Value": f"{total_cals:.1f} kcal", "Status": "PASS" if active_guideline['cal_min'] <= total_cals <= active_guideline['cal_max'] else "ALERT"},
                {"Standard": "Protein Provision", "Requirement": f">= {active_guideline['protein']:.0f} g", "Actual Value": f"{total_prot:.1f} g", "Status": "PASS" if total_prot >= active_guideline['protein'] else "ALERT"},
                {"Standard": "Dietary Fiber", "Requirement": f">= {active_guideline['fiber_min']:.0f} g", "Actual Value": f"{total_fiber:.1f} g", "Status": "PASS" if total_fiber >= active_guideline['fiber_min'] else "ALERT"},
                {"Standard": "Sodium Ceiling", "Requirement": f"<= {active_guideline['sodium_max']:.0f} mg", "Actual Value": f"{total_sodium:.0f} mg", "Status": "PASS" if total_sodium <= active_guideline['sodium_max'] else "ALERT"},
                {"Standard": "Added / Free Sugars", "Requirement": f"<= {active_guideline['sugar_max']:.0f} g", "Actual Value": f"{total_sugar:.1f} g", "Status": "PASS" if total_sugar <= active_guideline['sugar_max'] else "ALERT"},
                {"Standard": "Allergen Identification", "Requirement": "Mandatory EU FIC 1169/2011 Declaration", "Actual Value": ", ".join(detected_allergens) if detected_allergens else "None Detected", "Status": "PASS"}
            ]
            st.dataframe(pd.DataFrame(checklist), use_container_width=True, hide_index=True)

            st.write("")
            # Export CSV
            csv_data = df_meal[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Cafeteria Audit Report (CSV)",
                data=csv_data,
                file_name="school_meal_nutrition_audit.csv",
                mime="text/csv",
                help="Export itemized nutritional data for school district auditing and procurement tracking."
            )
        else:
            st.info("No items confirmed for export.")

    # --- TAB 4: Model Diagnostics & Calibration ---
    with tab4:
        st.markdown("#### 🔬 Zero-Shot Vision Model Diagnostics")
        st.write("""
        **NutriTray AI** operates using Google's **OWL-ViT (Open-Vocabulary Object Detection with Vision Transformers)**. 
        Unlike closed-set models (such as fixed YOLO checkpoints), OWL-ViT projects both image patches and natural language 
        prompt queries into a shared multimodal latent space.
        """)

        st.markdown("##### 📐 Bounded Reference-Anchored Weight Formula")
        st.latex(r"""
        W_{\text{estimated}} = W_{\text{typical}} \times \mathrm{clamp}\left(\frac{\text{Area}_{\text{bbox}} / \text{Area}_{\text{image}}}{\text{Area}_{\text{expected\_ratio}}}, \, 0.6, \, 1.4\right)
        """)
        st.caption("Damping bounding-box area prevents cafeteria tray angle variations from causing wild swings in calculated caloric output.")

        st.write("")
        st.markdown("##### 🔍 Raw Candidate Class Logits (Threshold = 0.0)")
        st.caption("Inspect raw model scores across all candidate food categories to evaluate sensitivity:")

        if st.session_state.get("debug_scores"):
            debug_rows = [
                {"Food / Class": k.title(), "Raw Score": round(v, 3), "Detected (> Threshold)": "✅ Yes" if v >= conf_threshold else "❌ No"}
                for k, v in sorted(st.session_state["debug_scores"].items(), key=lambda x: -x[1])
            ]
            st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run an image detection to inspect raw logits.")


if __name__ == "__main__":
    main()
