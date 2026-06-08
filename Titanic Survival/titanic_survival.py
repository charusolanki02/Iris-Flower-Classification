import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")
# Display first rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unnecessary columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Encode categorical columns
encoder = LabelEncoder()

if 'Sex' in df.columns:
    df['Sex'] = encoder.fit_transform(df['Sex'])

if 'Embarked' in df.columns:
    df['Embarked'] = encoder.fit_transform(df['Embarked'])

# Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Sample Prediction
sample = [[3, 1, 25, 0, 0, 7.25, 2]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPassenger Survived")
else:
    print("\nPassenger Did Not Survive")