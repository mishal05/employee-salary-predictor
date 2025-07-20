# Employee Salary Prediction using ML algorithm

This project is a **Machine Learning web app** built using **Streamlit** that predicts whether a person's annual salary is greater than or less than \$50,000 based on personal and professional attributes. It uses a pre-trained **Random Forest classifier** trained on the **Adult Census Income dataset** (UCI).

---

## Features

* Cleaned and preprocessed dataset (nulls, outliers removed)
* One-hot encoding for categorical features
* Random Forest classification pipeline
* Input validation to prevent **illogical combinations** (e.g., "Husband" cannot be "Female")
* Dynamic form interface using Streamlit
* Real-time prediction and model feedback
* Model serialized using `joblib` for easy deployment

---

## Input Fields

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| Age             | Person's age (18–90)                      |
| Workclass       | Type of employment (e.g., Private, Gov)   |
| Education Level | Educational numeric representation (1–16) |
| Marital Status  | Married, Never-married, Divorced, etc.    |
| Occupation      | Profession category                       |
| Relationship    | Role in family (e.g., Husband, Wife)      |
| Gender          | Male or Female                            |
| Race            | Race of individual                        |
| Native Country  | Origin country (e.g., US, India)          |
| Capital Gain    | Investment gain                           |
| Capital Loss    | Investment loss                           |
| Hours Per Week  | Weekly working hours                      |

---

## Model Training (Notebook)

The training pipeline includes:

1. Data Cleaning (replacing `"?"`, dropping nulls)
2. Feature Engineering:

   * One-hot encoding for categorical columns
3. Model:

   * `RandomForestClassifier`
4. Evaluation:

   * Accuracy score
5. Saving:

   * Model saved as `salary_model_pipeline.pkl`
   * Column info saved for consistent prediction

---

## Running the App

### 🔧 1. Install Requirements

```bash
pip install -r requirements.txt
```



### 2. Start the App

```bash
streamlit run app.py
```

Then visit `http://localhost:8501` in your browser.

---

## Example Prediction

**Input:**

* Age: 35
* Gender: Male
* Workclass: Private
* Relationship: Husband
* Marital Status: Married-civ-spouse
* Education Level: 13
* Capital Gain: 0
* Hours Per Week: 40

**Output:**

> Predicted Salary Category: **>50K**

---

## Logical Input Handling

To prevent user errors:

* Gender mismatch with relationship (e.g., Female + Husband) is blocked
* "Never-married" disables "Husband"/"Wife" as relationship options

---

## Dataset Source

UCI Machine Learning Repository – [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

---

## Future Improvements

* Add SHAP or LIME explanations
* Use advanced models like XGBoost or LightGBM
* Add feature importances
* Allow user to upload CSV for batch predictions

