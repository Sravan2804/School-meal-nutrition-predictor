# School Meal Nutrition Predictor

![Demo UI](file:///C:/Users/ramas/.gemini/antigravity-ide/brain/706b7d95-4484-4444-90f9-4969a54cee85/app_demo_1793545073888.png)

## 🚀 Overview

A **cutting‑edge AI‑powered web app** that instantly analyses a school lunch tray, identifies the food items, estimates portion weights, and delivers a **comprehensive nutritional breakdown**. Designed to empower nutritionists, educators, and policy makers with data‑driven insights for healthier school meals.

## 🛠️ Tech Stack

- **Python 3.11**
- **Streamlit** – Rapid web UI development.
- **PyTorch** – Model inference.
- **OWL‑ViT (Google)** – Zero‑shot object detection.
- **Plotly** – Interactive charts.
- **Pandas** – Data manipulation.
- **Pillow** – Image handling.

## 📸 Demo

The screenshot above (generated mock‑up) illustrates the sleek dashboard with:

- Detection image with bounding boxes.
- Real‑time nutrition cards.
- Sidebar controls for confidence thresholds and weight estimation modes.

## ⚙️ Getting Started

```bash
# Clone the repo
git clone <repository-url>
cd "School meal nutrition predictor"

# (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🎯 Features

- **Zero‑shot detection** – No need for a custom training dataset.
- **Dynamic weight estimation** – Size‑adjusted or fixed reference weights.
- **Compliance scoring** – Instant alerts for calories, sodium, sugar, and fiber.
- **Exportable reports** – CSV download of itemized nutrition data.
- **Manual overrides** – Fine‑tune quantities and portion sizes in the sidebar.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request. Follow the existing coding style and run the linter (`flake8`) before committing.

## 📄 License

MIT License – free to use, modify, and distribute.

## 📞 Contact

**Ramas** – _Data Scientist & AI Engineer_  
Email: [ramas@example.com](mailto:ramas@example.com)  
LinkedIn: [Ramas Profile](https://www.linkedin.com/in/ramas)
