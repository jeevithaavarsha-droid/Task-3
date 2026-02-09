# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load dataset
data = pd.read_csv("bank.csv")

# Step 3: View dataset
data.head()

# Step 4: Dataset information
data.info()

# Step 5: Convert target variable (purchase: yes/no → 1/0)
data["y"] = data["y"].map({"yes": 1, "no": 0})

# Step 6: Convert categorical columns to numeric
data_encoded = pd.get_dummies(data, drop_first=True)

# Step 7: Split features and target
X = data_encoded.drop("y", axis=1)
y = data_encoded["y"]

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 9: Build Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test)

# Step 11: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
