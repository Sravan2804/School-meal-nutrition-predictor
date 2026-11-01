# NutriTray AI · School Meal Nutrition & Compliance Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Vision Model OWL-ViT](<https://img.shields.io/badge/model-Google%20OWL--ViT%20(ViT--B%2F32)-8A2BE2.svg>)](https://huggingface.co/google/owlvit-base-patch32)
[![Standards HSE Aligned](https://img.shields.io/badge/standards-Irish%20HSE%20%26%20School%20Food%20Policy-059669.svg)](https://www.hse.ie)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An open-vocabulary computer vision system and interactive clinical audit dashboard designed to monitor, analyze, and verify dietary compliance in school meal programmes.

By pairing Google's **OWL-ViT (Open-Vocabulary Object Detection with Vision Transformers)** with a **perspective-damped portion estimation algorithm**, NutriTray AI converts cafeteria tray photos into instant macronutrient and micronutrient audits evaluated against official school food standards (e.g., Irish HSE and National School Food Policy guidelines).

---

## 📌 Background & Motivation

Universal hot school meal programmes play a vital role in child development, cognitive performance, and public health. However, ensuring nutritional quality control across hundreds of decentralized school kitchens currently relies on sporadic spot checks or manual paper logs.

**NutriTray AI** bridges the gap between vision AI and dietary auditing:

- **Zero-Shot Generalization**: Identifies un-trained dishes and cafeteria items on the fly using natural-language prompt embeddings rather than requiring costly re-training for every menu change.
- **Robust Portion Modeling**: Anchors volumetric serving calculations to verified dietitian reference weights with bounded geometric damping, preventing camera distance and tray angle from distorting caloric counts.
- **Automated Regulatory Compliance**: Real-time evaluation against Primary (ages 5–11) and Post-Primary (ages 12–18) nutritional benchmarks, covering caloric allowances, protein density, sodium ceilings, free sugars, and mandatory EU FIC 1169/2011 allergen warnings.
- **Human-in-the-Loop Workflow**: Dietitians and cafeteria staff retain full control to adjust portions, correct false detections, or add un-plated items before generating official procurement reports.

---

## 🏗️ System Architecture

```text
 ┌──────────────────────┐
 │ Cafeteria Tray Photo │  (Upload or Preloaded Test Sample)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────────────────────────┐
 │  OWL-ViT Zero-Shot Detector (ViT-B/32)   │
 │  Prompt Queries: Rich Descriptors        │  <-- e.g. "half of a sandwich with visible filling"
 └──────────┬───────────────────────────────┘
            │ Bounding Boxes & Confidence Scores
            ▼
 ┌──────────────────────────────────────────┐
 │  Non-Maximum Suppression (IoU = 0.38)    │
 └──────────┬───────────────────────────────┘
            │ Deduplicated Detections
            ▼
 ┌──────────────────────────────────────────┐
 │  Bounded Portion Estimation Engine       │
 │  W_est = W_ref * clamp(ratio/exp, .6,1.4)│  <-- Damps tray perspective and camera height
 └──────────┬───────────────────────────────┘
            │ Grams per detected food item
            ▼
 ┌──────────────────────────────────────────┐
 │  Nutritional Knowledge Base Integration  │  <-- nutrition_db.json (Macros, Micros, Allergens)
 └──────────┬───────────────────────────────┘
            │ Aggregated Nutrient Profile
            ▼
 ┌──────────────────────────────────────────┐
 │  HSE Compliance & Audit Dashboard (UI)   │
 │  • Executive Pass/Warning Verdict Card   │
 │  • EU FIC 1169/2011 Allergen Badges      │
 │  • Interactive Portion Adjuster          │
 │  • Plotly Energy Split & RDA Charts      │
 │  • Official CSV Audit Report Export      │
 └──────────────────────────────────────────┘
```

---

## 🔬 Core Technical Innovations

### 1. Open-Vocabulary Zero-Shot Detection

Traditional object detectors (e.g., YOLOv8, Faster R-CNN) are constrained to closed label sets determined at training time. School cafeteria menus change seasonally, making static detectors fragile.

NutriTray AI uses **Google OWL-ViT** (`google/owlvit-base-patch32`), which projects visual image patches and text embeddings into a shared multimodal latent space.

> **Prompt Engineering Insight:** CLIP-style vision-language models demonstrate poor zero-shot discrimination with single-word prompts (e.g., `apple` vs `quesadilla`). NutriTray AI utilizes expanded descriptive queries:
>
> - `"half of a sandwich with visible filling"` (avoids confusion with plain bread slices)
> - `"a flat rectangular slice of sandwich bread"`
> - `"small orange baby carrots"`
> - `"fresh red strawberries"`

### 2. Perspective-Damped Portion Modeling

Estimating portion weight strictly from 2D bounding-box area relative to image dimensions produces severe artifacts: moving the camera closer or tilting the tray inflates estimated calories even when physical food quantity is unchanged.

NutriTray AI anchors serving weight to a verified reference baseline (`typical_weight_g`) in `nutrition_db.json`, using the detected bounding-box area ratio only to apply a bounded adjustment factor ($0.6\times$ to $1.4\times$):

$$\text{Estimated Weight} = W_{\text{typical}} \times \mathrm{clamp}\left(\frac{\text{Area}_{\text{bbox}} / \text{Area}_{\text{image}}}{\text{Area}_{\text{expected}}}, \, 0.6, \, 1.4\right)$$

This provides stable nutritional predictions across different smartphone cameras and cafeteria lighting setups, while still allowing a visibly larger serving to appropriately register higher grams.

### 3. Standards-Aligned Compliance Framework

Nutrient benchmarks are configured according to **Irish Health Service Executive (HSE)** School Food Policy and National Guidelines:

| Nutrient Metric      | Primary School (Ages 5–11) | Post-Primary School (Ages 12–18) | Regulatory Purpose                              |
| :------------------- | :------------------------: | :------------------------------: | :---------------------------------------------- |
| **Energy Target**    | 500 kcal (450 – 575 range) |    650 kcal (585 – 745 range)    | Prevents under-nourishment & excess energy      |
| **Minimum Protein**  |        $\ge 15.0$ g        |           $\ge 22.0$ g           | Muscle synthesis & growth (12–20% of energy)    |
| **Target Carbs**     |      $\approx 65.0$ g      |         $\approx 85.0$ g         | Sustained cognitive energy                      |
| **Fat Ceiling**      |        $\le 15.0$ g        |           $\le 20.0$ g           | Restricts saturated fats to $<35\%$ of calories |
| **Dietary Fiber**    |        $\ge 4.0$ g         |           $\ge 6.0$ g            | Promotes whole grains, fruits & vegetables      |
| **Sodium Ceiling**   |       $\le 400.0$ mg       |          $\le 500.0$ mg          | Cardiovascular health protection                |
| **Free Sugar Limit** |        $\le 10.0$ g        |           $\le 12.0$ g           | WHO dental and metabolic guidelines             |

### 4. Food Allergen Management (EU FIC 1169/2011)

School caterers are legally required to declare major allergens. Every database food item is tagged with allergen metadata (Gluten, Dairy, Eggs, Sesame, etc.). The dashboard flags detected allergens prominently to ensure student safety before meal distribution.

---

## 🖥️ User Interface Overview

The web dashboard is organized into four purpose-built clinical views:

1. **🖼️ Tray Inspection & Portion Adjuster**:
   - High-contrast color-coded bounding boxes mapped to food categories.
   - Anti-aliased pill labels displaying item name and model confidence.
   - Interactive portion editor allowing dietitians to tweak quantities (e.g. 1 slice vs 2 slices) or adjust portion grams with real-time recalculation.
2. **📊 Dietary Analytics & Energy Split**:
   - Donut chart illustrating caloric distribution across Protein (4 kcal/g), Carbohydrates (4 kcal/g), and Fats (9 kcal/g).
   - Horizontal benchmark progress chart comparing meal provisions against HSE targets and statutory ceilings.
3. **📋 Official Procurement Audit & Export**:
   - Itemized nutritional spreadsheet breaking down weight, macros, sodium, sugar, and allergens per item.
   - Official School Food Standards Pass/Alert checklist.
   - One-click CSV export (`school_meal_nutrition_audit.csv`) for municipal and district record-keeping.
4. **🔬 Model Diagnostics & Calibration**:
   - Mathematical documentation of the bounded portion scaling formula.
   - Unfiltered class logit distribution table (threshold = 0.0) for model sensitivity transparency and troubleshooting.

---

## 📁 Repository Structure

```text
.
├── app.py                  # Main Streamlit application and inference pipeline
├── nutrition_db.json       # Nutritional reference database (macros, micros, allergens)
├── requirements.txt        # Production dependency specifications
├── Images/                 # Sample cafeteria trays and evaluation test images
│   ├── school_lunch_tray.jpg
│   ├── images.jpg
│   ├── Lunchboxes.jpg
│   └── ...
└── README.md               # Project documentation and engineering guide
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.10 – 3.14
- CUDA-compatible GPU (optional; CPU inference runs in ~2–4 seconds per tray)

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/your-username/school-meal-nutrition-predictor.git
cd "school-meal-nutrition-predictor"

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

> **Note on Model Weights:** The application uses `google/owlvit-base-patch32` (~600 MB). The model weights are automatically cached locally in `~/.cache/huggingface/` on first run. Subsequent launches execute entirely offline with zero network requests.

---

## 🛠️ Extending the Nutrition Database

To add regional dishes, hot entrees, or cafeteria-specific recipes, append an entry to `nutrition_db.json`:

```json
"lentil soup": {
  "name": "Red Lentil & Vegetable Soup",
  "serving_unit_g": 100,
  "typical_weight_g": 200,
  "expected_area_ratio": 0.045,
  "calories_kcal": 68.0,
  "protein_g": 4.2,
  "carbs_g": 10.5,
  "fat_g": 1.1,
  "fiber_g": 3.4,
  "sodium_mg": 180.0,
  "sugar_g": 1.8,
  "allergens": ["Celery"]
}
```

Then add the corresponding descriptive query in `PROMPT_TEXT` within `app.py`:

```python
"lentil soup": "a bowl of warm orange-red lentil soup",
```

---

## 🛡️ Cafeteria Deployment & Child Privacy Guidelines

When integrating NutriTray AI into an active school cafeteria, observe the following operational and privacy safeguards:

- **Privacy-by-Design (GDPR / COPPA Compliance)**: Cameras should be mounted strictly overhead (45° to 90° top-down) focused entirely on the tray checkout conveyor. Under no circumstances should facial imagery or student biometric identifiers be captured or stored.
- **Consistent Illumination**: Diffused overhead LED lighting (500–800 lux) minimizes harsh shadows that could affect bounding box regression.
- **Compartmentalized Trays**: High-contrast, standard compartment trays (metal or matte neutral composite) provide the cleanest visual separation between food items.
- **Edge Deployment**: For air-gapped school environments, the model runs self-contained on edge hardware (e.g., Apple Silicon M-series, NVIDIA Jetson Orin, or local desktop CPU) without sending data to external cloud APIs.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
