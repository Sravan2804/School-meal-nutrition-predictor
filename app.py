import streamlit as st
import pandas as pd
import json
import plotly.express as px
from PIL import Image, ImageDraw
import torch
import torchvision.ops as ops
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Define Unified Class Vocabulary
ALLOWED_CLASSES = [
    "bread roll", "bread slice", "sandwich half", "baby carrots", "salad", 
    "apple", "orange slice", "grapes", "boiled egg", "hummus", 
    "strawberries"
]
NON_FOOD_CONTAINERS = ["metal tray", "plastic water bottle"]
CANDIDATE_LABELS = ALLOWED_CLASSES + NON_FOOD_CONTAINERS

# OWL-ViT's zero-shot text matching is noticeably weaker with bare single/double-word
# labels than with a fuller descriptive phrase (a well-known property of CLIP-style
# vision-language models -- this is why "green apple" only matched green-colored
# apples, and why single-word "bread slice" vs "quesadilla" prompts were easily
# confused: both are visually similar tan, flat/folded shapes, and bare labels give
# the model very little to discriminate on). PROMPT_TEXT maps each short class name
# (used for lookup, display, and the nutrition_db key) to a more descriptive phrase
# that is what actually gets sent to the model as the text query.
PROMPT_TEXT = {
    "bread roll": "a round bread roll",
    "bread slice": "a flat rectangular slice of sandwich bread",
    "sandwich half": "half of a sandwich with visible filling",
    "baby carrots": "small orange baby carrots",
    "salad": "a mixed green leafy salad",
    "apple": "a whole apple fruit, any color",
    "orange slice": "a cut slice of orange fruit",
    "grapes": "a bunch of grapes",
    "boiled egg": "a peeled whole boiled egg",
    "hummus": "a container of beige hummus dip",
    "strawberries": "fresh red strawberries",
    "metal tray": "an empty metal compartment lunch tray",
    "plastic water bottle": "a clear plastic water bottle",
}
PROMPT_LABELS = [PROMPT_TEXT.get(c, c) for c in CANDIDATE_LABELS]

# Default fallbacks used only if a class is missing these fields in nutrition_db.json
DEFAULT_TYPICAL_WEIGHT_G = 100.0
DEFAULT_EXPECTED_AREA_RATIO = 0.035

# Load Nutrition Database
def load_nutrition_db():
    try:
        with open("nutrition_db.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("nutrition_db.json not found. Please ensure it's in the same directory.")
        return {}

# Load OwL-ViT Model
@st.cache_resource
def load_model():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    return processor, model


def estimate_item_weight(class_name, box_area, image_area, nutrition_db):
    """
    Estimate a single detection's serving weight.

    Previous approach estimated weight purely from bounding-box area relative to
    the whole image (relative_ratio * 1200, clamped 20-250g). That made the
    estimate highly sensitive to occlusion, camera distance, and how items were
    arranged on the tray -- rearranging the same physical items could change the
    box areas (and therefore the estimated weight and downstream calories)
    without any real change in what was served.

    This version anchors the estimate to a realistic per-item reference weight
    (nutrition_db[class]["typical_weight_g"]), and lets the detected box size
    only apply a bounded adjustment (0.6x-1.4x) around that reference, instead
    of letting box area alone determine the whole estimate. This keeps output
    far more stable across rearrangements of the same tray, while still letting
    a visibly larger/smaller item shift the estimate somewhat.
    """
    clean_key = str(class_name).strip().lower()
    item_data = nutrition_db.get(clean_key, {})

    reference_weight = float(item_data.get("typical_weight_g", DEFAULT_TYPICAL_WEIGHT_G))
    expected_ratio = float(item_data.get("expected_area_ratio", DEFAULT_EXPECTED_AREA_RATIO))

    relative_ratio = (box_area / image_area) if image_area > 0 else 0.0
    size_factor = (relative_ratio / expected_ratio) if expected_ratio > 0 else 1.0
    size_factor = max(0.6, min(1.4, size_factor))  # damped, bounded adjustment

    estimated_weight_g = reference_weight * size_factor
    # Keep a sane absolute floor/ceiling as a final safety net
    estimated_weight_g = max(10.0, min(400.0, estimated_weight_g))
    return round(estimated_weight_g, 1)


def calculate_item_nutrition(item_key, weight_g, nutrition_db):
    # Normalize key to lowercase to eliminate case mismatches
    clean_key = str(item_key).strip().lower()
    
    # Fallback if key missing
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
            "Allergen Badges": "None"
        }
    
    data = nutrition_db[clean_key]
    factor = float(weight_g) / 100.0  # Scale per 100g base values
    
    cals = data.get("calories_kcal", data.get("calories", 0.0))
    prot = data.get("protein_g", data.get("protein", 0.0))
    carbs = data.get("carbs_g", data.get("carbs", 0.0))
    fats = data.get("fat_g", data.get("fats_g", data.get("fats", 0.0)))
    fiber = data.get("fiber_g", data.get("fiber", 0.0))
    sodium = data.get("sodium_mg", data.get("sodium", 0.0))
    sugar = data.get("sugar_g", data.get("sugar", 0.0))

    allergens = data.get("allergens", [])
    allergens_str = ", ".join(allergens) if isinstance(allergens, list) and len(allergens) > 0 else "None"
    
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
        "Allergen Badges": allergens_str
    }

def main():
    st.set_page_config(
        page_title="School Meal Nutrition Predictor", 
        page_icon="🍱", 
        layout="wide"
    )
    
    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
            background-color: #1E1E1E;
            border: 1px solid #4CAF50;
            padding: 5% 5% 5% 10%;
            border-radius: 5px;
            color: #FFFFFF;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("School Meal Nutrition Predictor (Zero-Shot)")
    st.markdown("Upload a photo of a school meal tray to detect food items using Open-Vocabulary detection!")
    
    # ------------------ SIDEBAR ------------------
    st.sidebar.header("⚙️ Control Panel")
    
    uploaded_file = st.sidebar.file_uploader("Upload Image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])
    
    st.sidebar.subheader("Model Fine-Tuning")
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=0.50, value=0.10, step=0.01)
    
    st.sidebar.subheader("Weight Estimation Mode")
    weight_mode = st.sidebar.radio(
        "Weight Calculation Method",
        options=["Size-Adjusted (Estimated from Bounding Box)", "Standard Reference Weight (Fixed per item)"],
        index=0,
        help="Choose whether to estimate weights based on the bounding box size (which can vary slightly by position/perspective) or use the exact typical serving weight from the database."
    )
    use_typical_weight = (weight_mode == "Standard Reference Weight (Fixed per item)")
    
    nutrition_db = load_nutrition_db()
    processor, model = load_model()
    
    # Initialize Session State
    if 'detections' not in st.session_state:
        st.session_state['detections'] = []
    if 'last_uploaded' not in st.session_state:
        st.session_state['last_uploaded'] = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_width, image_height = image.size
            image_area = image_width * image_height
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        # Run inference if a new image was uploaded
        if st.session_state['last_uploaded'] != uploaded_file.name:
            with st.spinner("Analyzing food items with OwL-ViT..."):
                inputs = processor(text=[PROMPT_LABELS], images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                target_sizes = torch.tensor([[image_height, image_width]])
                results = processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
                )[0]
                
                boxes = results["boxes"]
                scores = results["scores"]
                labels = results["labels"]
                
                keep_indices = ops.nms(boxes, scores, iou_threshold=0.4)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]
                
                detections = []
                for box, score, label in zip(boxes, scores, labels):
                    box_list = [round(i, 2) for i in box.tolist()]
                    class_name = CANDIDATE_LABELS[label.item()]

                    # Reference-weight-anchored estimate (see estimate_item_weight docstring)
                    x1, y1, x2, y2 = box_list
                    box_area = (x2 - x1) * (y2 - y1)
                    estimated_weight_g = estimate_item_weight(class_name, box_area, image_area, nutrition_db)

                    detections.append({
                        "class_name": class_name,
                        "score": float(score.item()),
                        "box": box_list,
                        "est_weight_g": estimated_weight_g
                    })
                
                st.session_state['detections'] = detections
                st.session_state['last_uploaded'] = uploaded_file.name

                # --- Debug: raw best score per class, before threshold filtering ---
                # This runs the same model outputs through post-processing at threshold=0
                # so you can see every candidate class's best score, even ones that never
                # made it into the main detection list -- the fastest way to tell whether
                # an item is "not detected" because it scored near-zero (a real semantic
                # miss) or because it scored just under your current slider value.
                debug_results = processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=0.0
                )[0]
                debug_scores = {}
                for score, label in zip(debug_results["scores"], debug_results["labels"]):
                    cls = CANDIDATE_LABELS[label.item()]
                    s = float(score.item())
                    if cls not in debug_scores or s > debug_scores[cls]:
                        debug_scores[cls] = s
                st.session_state['debug_scores'] = debug_scores

        if 'debug_scores' in st.session_state and st.session_state['debug_scores']:
            with st.sidebar.expander("🔍 Debug: raw best score per class"):
                debug_df = pd.DataFrame(
                    sorted(st.session_state['debug_scores'].items(), key=lambda x: -x[1]),
                    columns=["Class", "Best raw score"]
                )
                debug_df["Above current threshold?"] = debug_df["Best raw score"] >= conf_threshold
                st.dataframe(debug_df, use_container_width=True, hide_index=True)

        # Draw bounding boxes
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        for det in st.session_state['detections']:
            box = det['box']
            class_name = det['class_name']
            score = det['score']
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], max(0, box[1] - 15)), f"{class_name} {score:.2f}", fill="red")

        # Extract detected food items (excluding non-food containers)
        available_options = list(nutrition_db.keys())
        detected_food_classes = list(dict.fromkeys([
            d['class_name'] for d in st.session_state['detections'] 
            if d['class_name'] in ALLOWED_CLASSES and d['class_name'] in available_options
        ]))

        # Sidebar: Manual Override & Weight Adjustments
        st.sidebar.markdown("---")
        st.sidebar.subheader("✏️ Manual Override & Serving Weights")
        
        final_foods = st.sidebar.multiselect(
            "Confirmed Food Items",
            options=available_options,
            default=detected_food_classes
        )
        
        item_weights = {}
        if final_foods:
            st.sidebar.markdown("**Adjust Quantities & Unit Weights:**")
            for food in final_foods:
                matching_dets = [d for d in st.session_state['detections'] if d['class_name'] == food]
                detected_qty = len(matching_dets)
                
                # Input quantity
                qty = st.sidebar.number_input(
                    f"{food.title()} Quantity",
                    min_value=0,
                    max_value=10,
                    value=detected_qty,
                    step=1,
                    key=f"qty_{food}"
                )
                
                if qty > 0:
                    if use_typical_weight:
                        # Use typical reference weight from nutrition database
                        ref_data = nutrition_db.get(food.strip().lower(), {})
                        unit_weight = float(ref_data.get("typical_weight_g", DEFAULT_TYPICAL_WEIGHT_G))
                    else:
                        # Use size-adjusted weight (average of detected weights if multiple, or typical if none detected)
                        if matching_dets:
                            unit_weight = sum(d['est_weight_g'] for d in matching_dets) / len(matching_dets)
                        else:
                            ref_data = nutrition_db.get(food.strip().lower(), {})
                            unit_weight = float(ref_data.get("typical_weight_g", DEFAULT_TYPICAL_WEIGHT_G))
                    
                    unit_weight = round(unit_weight, 1)
                    
                    # Input unit weight
                    adjusted_unit_weight = st.sidebar.number_input(
                        f"{food.title()} Unit Weight (g)",
                        min_value=5.0,
                        max_value=1000.0,
                        value=float(unit_weight),
                        step=5.0,
                        key=f"unit_wt_{food}"
                    )
                    
                    item_weights[food] = qty * adjusted_unit_weight
        
        if not final_foods:
            st.warning("No food items confirmed. Please select items manually in the sidebar.")
        elif not item_weights:
            st.warning("All confirmed food items have a quantity of 0. Please increase the quantity of at least one item.")
        else:
            # Aggregation Loop using calculate_item_nutrition
            item_rows = []
            for item_name, weight in item_weights.items():
                row = calculate_item_nutrition(item_name, weight, nutrition_db)
                item_rows.append(row)
            
            df_breakdown = pd.DataFrame(item_rows)
            
            total_cals = float(df_breakdown["Calories (kcal)"].sum())
            total_prot = float(df_breakdown["Protein (g)"].sum())
            total_carbs = float(df_breakdown["Carbs (g)"].sum())
            total_fats = float(df_breakdown["Fats (g)"].sum())
            total_fiber = float(df_breakdown["Fiber (g)"].sum())
            total_sodium = float(df_breakdown["Sodium (mg)"].sum())
            total_sugar = float(df_breakdown["Sugar (g)"].sum())

            # TABS LAYOUT
            tab1, tab2, tab3 = st.tabs([
                "🖼️ Detection & Overview", 
                "📊 In-Depth Nutritional Analysis", 
                "📋 Itemized Breakdown & Procurement Audit"
            ])
            
            # --- TAB 1: Detection & Overview ---
            with tab1:
                # Compliance Status Card
                if total_cals > 800:
                    status_text = "🔴 High Calorie Alert - Exceeds HSE Guidelines"
                elif total_fiber < 4.0:
                    status_text = "🟡 Low Fiber Alert - Consider adding fruits/veg"
                elif total_sodium > 400.0:
                    status_text = "🟡 High Sodium Alert - Exceeds 400mg Guideline"
                elif total_sugar > 10.0:
                    status_text = "🟡 High Sugar Alert - Exceeds 10g Guideline"
                else:
                    status_text = "🟢 Meets HSE Hot School Meal Guidelines"
                st.info(f"**School Meal Compliance Status:** {status_text}")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(draw_image, use_container_width=True, caption="OwL-ViT Zero-Shot Detection")
                
                with col2:
                    st.markdown("### Target RDA Progress")
                    pct_cals = min(total_cals / 500.0, 1.0)
                    pct_prot = min(total_prot / 15.0, 1.0)
                    pct_carbs = min(total_carbs / 65.0, 1.0)
                    pct_fats = min(total_fats / 15.0, 1.0)

                    st.metric("Calories", f"{total_cals:.1f} kcal")
                    st.progress(pct_cals, text=f"{int(pct_cals*100)}% of 500kcal Target")
                    
                    st.metric("Protein", f"{total_prot:.1f} g")
                    st.progress(pct_prot, text=f"{int(pct_prot*100)}% of 15g Target")
                    
                    st.metric("Carbs", f"{total_carbs:.1f} g")
                    st.progress(pct_carbs, text=f"{int(pct_carbs*100)}% of 65g Target")
                    
                    st.metric("Fats", f"{total_fats:.1f} g")
                    st.progress(pct_fats, text=f"{int(pct_fats*100)}% of 15g Target")

            # --- TAB 2: In-Depth Nutritional Analysis ---
            with tab2:
                st.markdown("### Micronutrients")
                m1, m2, m3 = st.columns(3)
                m1.metric("Fiber (g)", f"{total_fiber:.1f}")
                m2.metric("Sodium (mg)", f"{total_sodium:.0f}")
                m3.metric("Sugar (g)", f"{total_sugar:.1f}")
                
                st.divider()
                
                c3, c4 = st.columns(2)
                with c3:
                    st.markdown("#### Caloric Distribution")
                    cals_from_prot = total_prot * 4
                    cals_from_carbs = total_carbs * 4
                    cals_from_fats = total_fats * 9
                    
                    if total_cals > 0:
                        fig1 = px.pie(
                            values=[cals_from_prot, cals_from_carbs, cals_from_fats],
                            names=["Protein", "Carbs", "Fats"],
                            hole=0.4,
                            color_discrete_sequence=["#ef553b", "#00cc96", "#ab63fa"]
                        )
                        fig1.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                        st.plotly_chart(fig1, use_container_width=True)
                
                with c4:
                    st.markdown("#### Micronutrient Progress vs Targets")
                    fiber_pct = (total_fiber / 6.0) * 100
                    sodium_pct = (total_sodium / 400.0) * 100
                    sugar_pct = (total_sugar / 10.0) * 100

                    micro_df = pd.DataFrame({
                        "Nutrient": ["Fiber (6g Target)", "Sodium (400mg Max)", "Sugar (10g Max)"],
                        "Percentage": [round(fiber_pct, 1), round(sodium_pct, 1), round(sugar_pct, 1)],
                        "Value_Text": [f"{fiber_pct:.1f}%", f"{sodium_pct:.1f}%", f"{sugar_pct:.1f}%"]
                    })
                    
                    max_x = max(110.0, float(max(fiber_pct, sodium_pct, sugar_pct) + 15.0))
                    
                    fig2 = px.bar(
                        micro_df, 
                        x="Percentage", 
                        y="Nutrient", 
                        orientation='h',
                        text="Value_Text",
                        color="Nutrient",
                        color_discrete_sequence=["#00cc96", "#ef553b", "#ab63fa"]
                    )
                    fig2.update_traces(textposition='outside')
                    fig2.update_layout(
                        margin=dict(t=20, b=20, l=10, r=40), 
                        height=300, 
                        showlegend=False,
                        xaxis_title="% of Recommended Intake / Guideline Limit",
                        yaxis_title="",
                        xaxis=dict(range=[0, max_x])
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # --- TAB 3: Itemized Breakdown & Procurement Audit ---
            with tab3:
                st.markdown("### Itemized Breakdown")
                st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
                
                # CSV Export
                csv = df_breakdown.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Audit Report (CSV)",
                    data=csv,
                    file_name="school_meal_audit_report.csv",
                    mime="text/csv",
                )

main()