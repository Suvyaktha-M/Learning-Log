import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create dataset
data = {
    "Height": [158, 165, 170, 180, 155, 175, 162, 178],
    "Weight": [55, 60, 65, 80, 50, 75, 58, 78],
    "ShoeSize": [6, 7, 8, 10, 5, 9, 6, 10],
    "Gender": ["Female", "Female", "Male", "Male", "Female", "Male", "Female", "Male"],
}

df = pd.DataFrame(data)

# Separate features and target
X = df[["Height", "Weight", "ShoeSize"]]
y = df["Gender"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Test with new data
new_person = pd.DataFrame([[172, 68, 8]], columns=["Height", "Weight", "ShoeSize"])

prediction = model.predict(new_person)
print("Predicted Gender:", prediction[0])
