import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# 🌟 UI Styling
st.markdown(
    """
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; color: #FF4B4B; }
    .stApp { background-color: #f7f7f7; }
    </style>
    """,
    unsafe_allow_html=True
)

 # 📚 **Defining File Paths**
calories_file = r"E:\Documents\Fitness-Tracker\calories.csv"
exercise_file = r"E:\Documents\Fitness-Tracker\exercise.csv"

# ❗ Check if files exist
if not os.path.exists(calories_file) or not os.path.exists(exercise_file):
    st.error("Error: Required CSV files (`calories.csv`, `exercise.csv`) are missing!")
    st.stop()

# 🔄 Load Data
@st.cache_data
def load_data():
    calories = pd.read_csv(calories_file)
    exercise = pd.read_csv(exercise_file)
    data = exercise.merge(calories, on="User_ID")  # Ensure 'User_ID' exists in both files
    return data

data = load_data()

# 🏗 Convert Categorical Data
data = pd.get_dummies(data, drop_first=False)

# 🏷 Detect Gender Column Dynamically
gender_cols = [col for col in data.columns if 'Gender' in col]
if not gender_cols:
    st.error("No gender-related column found after encoding!")
    st.stop()

gender_column = gender_cols[0]  # Use the first detected gender column

# 🔗 SQLite Database
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

# 🎨 Streamlit UI
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.write("Track your calories burned based on your health data.")

st.sidebar.header("📊 User Input Parameters")

# 📥 User Input Function
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    height = st.sidebar.slider("Height (cm)", 140, 200, 170)
    weight = st.sidebar.slider("Weight (kg)", 40, 120, 70)
    duration = st.sidebar.slider("Workout Duration (min)", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    
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
        "Gender": [gender_button]  # Store as text in DB
    })

df = user_input_features()

# 🔎 Display User Input
st.write("### 📝 Your Input Data")
st.write(df)

# 📤 Save User Data to Database
cursor.execute("INSERT INTO users (age, gender, height, weight, bmi, calories) VALUES (?, ?, ?, ?, ?, ?)", 
               (df["Age"].values[0], df["Gender"].values[0], df["Height"].values[0], df["Weight"].values[0], df["BMI"].values[0], 0))
conn.commit()

# 📊 Machine Learning Model Training
data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
feature_columns = gender_cols + ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
data = data[feature_columns]

X = data.drop("Calories", axis=1)
y = data["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 📈 Improve Model with Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔥 Gradient Boosting Model for Better Accuracy
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(X_train_scaled, y_train)

# 🎯 Align Input Data for Prediction
df_ml = df.copy()
df_ml["Gender"] = 1 if df_ml["Gender"].values[0] == "Male" else 0
df_ml = df_ml.reindex(columns=X_train.columns, fill_value=0)

df_ml_scaled = scaler.transform(df_ml)
prediction = model.predict(df_ml_scaled)

# 🔥 Display Prediction
st.markdown(f"<p class='big-font'>🔥 Predicted Calories Burned: {round(prediction[0], 2)} kcal</p>", unsafe_allow_html=True)

# 🔄 Update Database
cursor.execute("UPDATE users SET calories = ? WHERE id = (SELECT MAX(id) FROM users)", (round(prediction[0], 2),))
conn.commit()

# 📊 Show Similar Results
st.write("### 📍 Similar Results from Dataset")
similar_data = data[(data["Calories"] >= prediction[0] - 10) & (data["Calories"] <= prediction[0] + 10)]
st.write(similar_data.sample(min(5, len(similar_data))))  # Avoid error if fewer than 5 results exist

# 📈 Fitness Trends
st.write("### 📊 Fitness Trends Over Age")

fig, ax = plt.subplots()
sns.scatterplot(data=data, x="Age", y="Calories", hue=gender_column, ax=ax)
st.pyplot(fig)

# 📜 Fitness Insights
st.write("### 📈 General Statistics")
st.write(f"📌 You are older than **{round((data['Age'] < df['Age'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"📌 Your workout duration is longer than **{round((data['Duration'] < df['Duration'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"📌 Your heart rate is higher than **{round((data['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}%** of users.")
st.write(f"📌 Your body temperature is higher than **{round((data['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100, 2)}%** of users.")

# 📂 Show Past Records
users_df = pd.read_sql_query("SELECT * FROM users", conn)
st.write("### 📜 Past User Records")
st.write(users_df)

conn.close()
