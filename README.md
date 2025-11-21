# ✅ Customer Lifetime Value (CLV) Prediction Using Machine Learning

## 📌 Project Overview

Customer Lifetime Value (CLV) is one of the most important metrics in modern business analytics.
It measures the total revenue a customer is expected to generate throughout their relationship with a business.

This project uses machine learning models to:

Estimate long-term customer value

Identify high-value vs. low-value customers

Optimize marketing spend and resource allocation

Improve retention strategies

Increase profitability through data-driven decisions

The model analyzes behavioral, demographic, and transactional features to generate accurate CLV predictions.

## 📊 Dataset Overview

The dataset contains typical customer attributes such as:

Demographics (gender, seniority status)

Engagement features (tenure, service usage)

Behavioral metrics (charges, contract type)

Subscription indicators (InternetService, Add-ons)

Financial data (MonthlyCharges, TotalCharges)

Churn indicator

Historical transactions for CLV calculation (if available)

These attributes allow the model to capture spending patterns and predict long-term value.

## 🧼 Preprocessing & Feature Engineering

The following steps were performed:

✔ Data Cleaning

Handling missing values

Fixing incorrect data types

Removing noise and inconsistent records

✔ Feature Engineering

To improve CLV prediction accuracy:

Created revenue-based features (AvgMonthlyRevenue, RevenuePerService)

Calculated engagement-related metrics (ServiceCount, ContractLengthScore)

Transformed categorical variables using one-hot encoding

Computed Raw CLV (if historical revenue existed)

✔ Feature Scaling

Standardized numerical features for algorithms that rely on distance or gradient calculations.

33 📈 Exploratory Data Analysis (EDA)

Several plots were generated to understand customer value drivers:

🔹 Distribution Analysis

Study of MonthlyCharges, Tenure, TotalCharges, raw CLV distribution.

🔹 Correlation Heatmap

Identifies strongest relationships between features and CLV.

🔹 Boxplots & Segment Comparisons

To see CLV variation across:

Payment methods

Contract types

Internet service types

Add-on subscriptions

🔹 Scatter & Revenue Trend Plots

To observe patterns like:

Revenue increases with tenure

Long-term contract customers producing higher CLV

## 🤖 Modeling

Several machine learning models were evaluated:

Models Used:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

LightGBM Regressor

CatBoost Regressor

Neural Network (optional)

Model Selection Criteria:

✔ MAE (Mean Absolute Error)
✔ RMSE (Root Mean Squared Error)
✔ R² Score
✔ Cross-validation error

Final Model

LightGBM / XGBoost achieved the best performance with:

High accuracy

Low overfitting

Strong generalization on test data

## 🎯 Results & Insights

The optimized model identifies:

⭐ High-Value Customers

Long tenure

High monthly charges

Multi-service users

Long-term contract customers
These customers generate strong long-term profit.

⚠️ Medium-Value Customers

Medium engagement

Irregular spending

Mixed service usage

❗ Low-Value or At-Risk Customers

Short tenure

Low spending

High churn likelihood

Minimal service subscriptions

These insights help businesses choose where to focus marketing and retention efforts.

## 💰 Business Impact

Predicting CLV enables:

Smarter marketing allocation (focus on high-return segments)

Targeted retention strategies

Reduction of unnecessary marketing costs

Higher customer lifetime value through personalization

Better forecasting and revenue planning

CLV is essential for subscription-based, telecom, retail, and e-commerce platforms.

## 🌐 Deployment on Hugging Face Spaces

This project can be deployed using Hugging Face Spaces with a Gradio app.

Steps to deploy:

Open:
https://huggingface.co/spaces

Create a New Space

Choose Gradio template

Upload:

app.py

model.pkl

requirements.txt

#### Example Gradio Code:

```python
import gradio as gr

def predict_clv(input_data):
    # load model & scaler
    return model.predict([input_data])[0]

iface = gr.Interface(
    fn=predict_clv,
    inputs=[gr.Number(), gr.Number(), gr.Number(), ...],
    outputs="number",
    title="Customer Lifetime Value Prediction",
)

iface.launch()
```

## 🖥 How to Run Locally
#### 1️⃣ Install dependencies
```python
pip install -r requirements.txt
```

#### 2️⃣ Run the Gradio app
```python
python app.py
```

## 📁 Project Structure
````markdown
├── data/
├── models/
│   ├── clv_model.pkl
├── notebooks/
│   ├── EDA.ipynb
│   ├── CLV_Modeling.ipynb
├── app.py
├── requirements.txt
├── README.md
````

## 📌 Conclusion
This CLV prediction project provides a robust, data-driven system to forecast customer profitability.
By combining machine learning, feature engineering, and visual analytics, it becomes easier for businesses to:

Understand customer value

Prioritize retention

Personalize offers

Increase long-term revenue

## 🔮 Future Improvements

Use deep learning models for more complex CLV patterns

Add survival analysis for more accurate lifetime estimation

Integrate model with CRM tools (Salesforce, HubSpot, Zoho)

Deploy REST API for third-party applications

Build an advanced dashboard on Hugging Face
