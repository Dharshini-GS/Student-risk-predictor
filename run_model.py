import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.abspath(os.path.join(BASE_DIR, 'data', 'StudentsPerformance.csv'))
df = pd.read_csv(csv_path)

df["average_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
df["pass_fail"] = (df["average_score"] >= 50).astype(int)

categorical_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[categorical_cols + ["math score", "reading score", "writing score"]]
y = df["pass_fail"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\n Model trained and saved successfully\n")
print("Enter student data as numbers (one by one):")

print("Gender: 0 = male, 1 = female")
gender = int(input())

print("Race: 0=A, 1=B, 2=C, 3=D, 4=E")
race = int(input())

print("Parental Education: 0=some high school, 1=high school, 2=some college, 3=associate's degree, 4=bachelor's degree, 5=master's degree")
parent_edu = int(input())

print("Lunch: 0=standard, 1=free/reduced")
lunch = int(input())

print("Test Preparation: 0=none, 1=completed")
prep = int(input())

print("Math score:")
math = int(input())

print("Reading score:")
reading = int(input())

print("Writing score:")
writing = int(input())

# Create DataFrame for new student
student_dict = {
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": parent_edu,
    "lunch": lunch,
    "test preparation course": prep,
    "math score": math,
    "reading score": reading,
    "writing score": writing
}

student_df = pd.DataFrame([student_dict])

# **DO NOT use LabelEncoder.transform** on numeric inputs
student_scaled = scaler.transform(student_df)
prediction = model.predict(student_scaled)[0]
prob = model.predict_proba(student_scaled)[0][1]

if prediction == 1:
    print(f"\nPrediction: PASS (Confidence: {prob:.2f})")
else:
    print(f"\nPrediction: AT RISK (Confidence: {1 - prob:.2f})")


