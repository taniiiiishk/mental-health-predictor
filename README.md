# 🧠 Mental Health Treatment Predictor

A machine learning web app that predicts whether a person working in tech is likely to seek mental health treatment, based on survey responses. Built with scikit-learn, served with FastAPI, and containerised with Docker.

🔗 **Live demo:** `<https://lnkd.in/p/gJ2AyWnf>`



> ⚠️ This is a learning project and **not medical advice**.

---

## 📌 Problem Statement

**Type:** Binary classification (supervised learning)

Predict the target `treatment` (Yes / No) using demographic details, workplace factors and personal factors from a mental health survey of tech workers.

## 📊 Dataset

Mental Health in Tech Survey (OSMI), available on Kaggle. It contains responses about age, gender, country, family history, work environment, company benefits and more.

## 🔧 Workflow

1. **Data cleaning:** removed duplicates and useless columns, stripped text, fixed impossible ages, encoded the target
2. **EDA:** target distribution, gender vs treatment, top countries, correlation heatmap
3. **Preprocessing (sklearn Pipeline + ColumnTransformer):**
   - Skewed numeric columns: median imputation + `PowerTransformer`
   - Other numeric columns: median imputation + `StandardScaler`
   - Categorical columns: most-frequent imputation + `OneHotEncoder`
4. **Modelling:** Logistic Regression baseline, then Random Forest
5. **Tuning:** `RandomizedSearchCV` (3-fold CV)
6. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, feature importance
7. **Deployment:** FastAPI backend, HTML UI, Docker, Render

## 📈 Results

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression (baseline) | XX% | XX% |
| Random Forest (default) | XX% | XX% |
| Random Forest (tuned) | XX% | XX% |

*Replace the XX values with your own results.*

## 🛠️ Tech Stack

- **Python**, pandas, NumPy
- **scikit-learn** (pipelines, Random Forest, hyperparameter tuning)
- **matplotlib**, **seaborn** (EDA)
- **FastAPI**, **Uvicorn** (API)
- **HTML / CSS / JavaScript** (UI)
- **Docker**, **Render** (deployment)

## 📁 Project Structure

```
mental-health-api/
├── main.py            # FastAPI app (API + serves the UI)
├── index.html         # Frontend form, built automatically from the model schema
├── model.pkl          # Trained sklearn pipeline
├── columns.pkl        # Input column names
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container setup
├── .dockerignore
├── notebook/
│   └── model.ipynb    # Training notebook (EDA, modelling, tuning)
└── README.md
```

## 🚀 Run Locally

**Without Docker (macOS):**

```bash
git clone https://github.com/<your-username>/mental-health-predictor.git
cd mental-health-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

**With Docker:**

```bash
docker build -t mental-health-api .
docker run -p 8000:8000 mental-health-api
```

Open http://localhost:8000

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/docs` | Interactive API docs (Swagger) |
| GET | `/features` | List of input columns the model expects |
| GET | `/schema` | Field types and dropdown options used by the UI |
| POST | `/predict` | Returns the prediction and probability |

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"Age": 29, "Gender": "Male", "Country": "India", "family_history": "Yes"}}'
```

**Example response:**

```json
{
  "treatment": "Yes",
  "probability_yes": 0.78
}
```

Any field that is left out is imputed by the pipeline.

## 🔮 Future Improvements

- Add SHAP explanations for each prediction
- Try Gradient Boosting and XGBoost
- Handle class imbalance (if present)
- Add input validation and unit tests
- Add CI/CD with GitHub Actions

## 👤 Author

**Tanishk Yadav**
B.Tech CSE, LNCT Bhopal

- LinkedIn: `<https://www.linkedin.com/in/tanishk-yadav-a10569331/>`
- GitHub: `<https://github.com/taniiiiishk>`

---

⭐ If you found this useful, consider giving the repo a star.
