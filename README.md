🌱 Fertilizer / Crop Prediction using Machine Learning

A Machine Learning project that predicts the suitable crop/fertilizer label based on soil nutrients (N, P, K), temperature, humidity, pH and rainfall. The project includes EDA, preprocessing, outlier removal, feature scaling, multiple ML model comparison, and final prediction.

📌 Project Overview

This project builds a classification model using multiple ML algorithms to identify the correct fertilizer/crop label for given soil and environmental conditions.
It includes:

✔ Data cleaning & preprocessing
✔ Label Encoding
✔ Exploratory Data Analysis (EDA)
✔ Outlier Detection using Z-Score
✔ Feature Scaling
✔ Training ML classification models
✔ Model Performance Comparison
✔ Final Prediction using Random Forest

📂 Dataset

The dataset used: Crop.csv

Columns include:

N – Nitrogen

P – Phosphorus

K – Potassium

temperature

humidity

ph

rainfall

label – Target crop/fertilizer label

🧹 Data Preprocessing Steps

Handling missing values

Converting categorical values to numeric using LabelEncoder

Checking correlations

Visualizations:

Heatmap

Scatter plot

Histogram

Detecting and removing outliers using Z-Score (threshold = 3)

📊 Exploratory Data Analysis

Heatmap shows relationships between nutrient values and target crop label

Boxplots used to detect outliers

Distribution of labels visualized using histograms

Scatterplot of Phosphorus (P) vs Label

🤖 Machine Learning Models Used

The following models were trained and evaluated:

Model	Type
Logistic Regression	Linear Model
Decision Tree Classifier	Tree-Based
Random Forest Classifier	Ensemble
Gradient Boosting Classifier	Ensemble
Support Vector Machine	Kernel-based
K-Nearest Neighbors	Distance-based
Gaussian Naive Bayes	Probabilistic

Each model was evaluated on:

✔ Accuracy
✔ Precision
✔ Recall
✔ F1-Score

🏆 Model Evaluation

The script prints evaluation metrics for every model.
The best-performing model (usually Random Forest) is used for the final prediction.

🔮 Final Prediction

A sample input is processed using StandardScaler and passed to the Random Forest model:

res=np.array([[80,67,50,35.89,85.63,7.95]])
ress=sc.transform(res)
print(rf.predict(ress))


This predicts the most suitable crop/fertilizer label for given nutrient values.

📦 Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

SciPy (Z-Score)

▶ How to Run the Project

Install required packages

pip install numpy pandas matplotlib seaborn scikit-learn scipy


Place Crop.csv in the same folder.

Run the Python file in Jupyter Notebook or VS Code.

📁 Project Structure
├── Crop.csv
├── fertilizer_prediction.ipynb
├── README.md

🚀 Future Improvements

Add Deep Learning (ANN) for improved accuracy

Build a Flask/Streamlit Web App

Add SHAP explainability

Hyperparameter tuning using GridSearchCV

👨‍💻 Author

Prasanna
Data Science & Machine Learning Enthusiast
