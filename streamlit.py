from LLM.reveal_rag import getLLMRecommendation
import streamlit as st
import openai

# Set up API key (or use a local LLM instead)
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🏡 Property Purchase Advisor")

# Inputs
salary = st.number_input("Your Annual Salary (€)", min_value=10000)
budget = st.number_input("Your Maximum Budget (€)", min_value=50000)
year = st.number_input("Year of Purchase", min_value=2024, max_value=2030, value=2025)

if st.button("Get Recommendation"):
    recommendation = getLLMRecommendation(year=year, salary=salary, budget=budget)

    st.markdown("### 🧠 AI Recommendation")
    st.write(recommendation)

