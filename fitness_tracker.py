import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Define file paths
calories_file = "E:/Documents/calories.csv"
exercise_file = "E:/Documents/exercise.csv"

# Check if files exist
if not os.path.exists(calories_file) or not os.path.exists(exercise_file):
    st.error("Error: One or both of the required CSV files (`calories.csv`, `exercise.csv`) are missing!")
    st.stop()

# Load data with caching
@st.cache_data
def load_data():
    calories = pd.read_csv(calories_file)
    exercise = pd.read_csv(exercise_file)
    data = exercise.merge(calories, on="User_ID")  # Ensure 'User_ID' exists in both files
    return data

data = load_data()

# Show column names for debugging
st.write("Columns in data:", data.columns)

# Convert categorical data
data = pd.get_dummies(data, drop_first=False)

# Detect gender column dynamically
gender_cols = [col for col in data.columns if 'Gender' in col]
if not gender_cols:
    st.error("No gender-related column found after encoding!")
    st.stop()

gender_column = gender_cols[0]  # Use the first detected gender column

# Create SQLite Database
conn = sqlite3.connect("fitness_tracker.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY, 
        age INTEGER, 
        gender TEXT, 
        height REAL, 
        weight REAL, 
        bmi REAL, 
        calories REAL)
''')
conn.commit()

# Streamlit UI
st.write("# Personal Fitness Tracker")
st.write("Track your calories burned based on your health data.")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    height = st.sidebar.slider("Height (cm): ", 140, 200, 170)
    weight = st.sidebar.slider("Weight (kg): ", 40, 120, 70)
    duration = st.sidebar.slider("Workout Duration (min): ", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm): ", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 35, 42, 37)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0  # Convert to binary
    bmi = round(weight / ((height / 100) ** 2), 2)

    return pd.DataFrame({
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender": [gender_button]  # Store as "Male"/"Female" for SQLite
    })

df = user_input_features()

# Show input
st.write("### Your Input Data:")
st.write(df)

# Add user data to database
cursor.execute("INSERT INTO users (age, gender, height, weight, bmi, calories) VALUES (?, ?, ?, ?, ?, ?)", 
               (df["Age"].values[0], df["Gender"].values[0], df["Height"].values[0], df["Weight"].values[0], df["BMI"].values[0], 0))
conn.commit()

# Train Machine Learning Model
data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)

# Select columns dynamically
feature_columns = gender_cols + ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
data = data[feature_columns]

X = data.drop("Calories", axis=1)
y = data["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
model.fit(X_train, y_train)

# Align input columns
df_ml = df.copy()
df_ml["Gender"] = 1 if df_ml["Gender"].values[0] == "Male" else 0  # Convert gender to match encoded format
df_ml = df_ml.reindex(columns=X_train.columns, fill_value=0)

# Predict calories burned
prediction = model.predict(df_ml)
st.write("### Predicted Calories Burned: ")
st.write(f"🔥 {round(prediction[0], 2)} kilocalories")

# Update calories in database
cursor.execute("UPDATE users SET calories = ? WHERE id = (SELECT MAX(id) FROM users)", (round(prediction[0], 2),))
conn.commit()

# Show similar results
st.write("### Similar Results from Dataset:")
similar_data = data[(data["Calories"] >= prediction[0] - 10) & (data["Calories"] <= prediction[0] + 10)]
st.write(similar_data.sample(min(5, len(similar_data))))  # Prevent error if less than 5 samples exist

# Show fitness trends
st.write("### Fitness Trends")

fig, ax = plt.subplots()
sns.scatterplot(data=data, x="Age", y="Calories", hue=gender_column, ax=ax)
st.pyplot(fig)

# Show general fitness stats
st.write("### General Information:")
st.write(f"You are older than **{round((data['Age'] < df['Age'].values[0]).mean() * 100, 2)}%** of people.")
st.write(f"Your workout duration is longer than **{round((data['Duration'] < df['Duration'].values[0]).mean() * 100, 2)}%** of people.")
st.write(f"Your heart rate is higher than **{round((data['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}%** of people.")
st.write(f"Your body temperature is higher than **{round((data['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100, 2)}%** of people.")

conn.close()
