import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load the data
data = pd.read_excel('epl-training_basic.csv')

# Step 2: Preprocess the data
# Encode categorical variables
le = LabelEncoder()
data['HomeTeam'] = le.fit_transform(data['HomeTeam'])
data['AwayTeam'] = le.fit_transform(data['AwayTeam'])
data['Result'] = le.fit_transform(data['Result'])

# Step 3: Define features and target variable
X = data[['HomeTeam', 'AwayTeam']]
y = data['Result']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Step 7: Function to predict future matches
def predict_match(home_team, away_team):
    home_team_enc = le.transform([home_team])[0]
    away_team_enc = le.transform([away_team])[0]
    prediction = svm_model.predict([[home_team_enc, away_team_enc]])[0]
    result = le.inverse_transform([prediction])[0]
    return result

# Example prediction
print(predict_match('TeamA', 'TeamB'))
