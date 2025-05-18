# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 2. Load Dataset
df = pd.read_csv("personalized_learning_dataset.csv")

# 3. Data Cleaning
df = df.drop(columns=["Student_ID"])
assert not df.isnull().any().any(), "Missing values found!"
assert not df.duplicated().any(), "Duplicate rows found!"

# 4. Exploratory Data Analysis (5 charts)
sns.set(style="whitegrid")
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Dropout_Likelihood', palette='Set2')
plt.title('Dropout Likelihood Distribution')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(data=df, x='Learning_Style', y='Final_Exam_Score', palette='viridis')
plt.title('Average Final Exam Score by Learning Style')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Engagement_Level', y='Assignment_Completion_Rate', palette='pastel')
plt.title('Assignment Completion Rate by Engagement Level')
plt.ylabel('Completion Rate (%)')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Dropout_Likelihood', bins=20, kde=True, multiple="stack", palette='coolwarm')
plt.title('Age Distribution by Dropout Likelihood')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 5. Encode Categorical Variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 6. Split Data
X = df.drop(columns='Dropout_Likelihood')
y = df['Dropout_Likelihood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=1,
                           scoring='accuracy')

grid_search.fit(X_train, y_train)

# 8. Best Model from Grid Search
best_model = grid_search.best_estimator_

# 9. Evaluate Best Model
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


import pickle

# Save trained model
with open("trained_dropout_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Save feature names
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

