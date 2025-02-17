import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Loading dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic_Data.csv", delimiter=";", decimal=",")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove extra unnamed columns
    return df
df = load_data()

# Title
st.title("🚢 Titanic Data Dashboard")

# Sidebar filters
st.sidebar.header("Filter Data")
gender_filter = st.sidebar.selectbox("Select Gender", ["All"] + df["Sex"].unique().tolist())

# Apply filter
if gender_filter != "All":
    df = df[df["Sex"] == gender_filter]

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show key statistics
st.subheader("Key Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Passengers", len(df))
col2.metric("Survivors", df["Survived"].sum())
col3.metric("Survival Rate", f"{df['Survived'].mean() * 100:.2f}%")

# Age distribution histogram
st.subheader("Age Distribution")
fig, ax = plt.subplots()
df["Age"].dropna().hist(bins=20, ax=ax)
st.pyplot(fig)

# Survival rate by class
st.subheader("Survival Rate by Passenger Class")
st.bar_chart(df.groupby("Pclass")["Survived"].mean())



