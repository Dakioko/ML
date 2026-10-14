import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the data
df = pd.read_csv('train.csv')

# 2. Feature Selection
# We'll stick to the most impactful features for now
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

# 3. THE SPLIT (Before any cleaning/imputation)
# This creates our internal "test" set (X_val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Data Cleaning (Handling Missing Values)
# We calculate the median on TRAIN only to avoid leakage
age_median = X_train['Age'].median()

# Apply that median to both sets
X_train['Age'] = X_train['Age'].fillna(age_median)
X_val['Age'] = X_val['Age'].fillna(age_median)

# 5. Encoding (Converting Text to Numbers)
# Sex: male=0, female=1
X_train['Sex'] = X_train['Sex'].map({'male': 0, 'female': 1})
X_val['Sex'] = X_val['Sex'].map({'male': 0, 'female': 1})

# 6. Fit the Model
# We train ONLY on the X_train portion
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 7. Performance Check
# Now we test it on the X_val "secret" data
val_predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)

print(f"Validation Accuracy: {accuracy:.2%}")