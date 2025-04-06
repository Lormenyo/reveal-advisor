import pandas as pd
import openai  # Use llama-cpp-python for local models
import os

from model.data_preprocessing import load_and_clean_data

def getLLMRecommendation(year, salary, budget):
    # Load the dataset
    df = load_and_clean_data("PPR-ALL.csv")

    # Summarize real estate trends (Average price per county)
    county_summary = df.groupby("County")["Price (€)"].mean().reset_index()
    county_summary = county_summary.rename(columns={"Price (€)": "Average Price (€)"})

    # Convert to a readable string
    trend_summary = county_summary.to_string(index=False)

    # # Define user input
    # year = 2025
    # salary = 60000  # User's salary
    # budget = 300000  # User's max budget

    # Define the LLM prompt
    prompt = f"""
    Given the real estate market trends in Ireland, help a user predict the **best county** to buy a house, **affordability**, and the expected **property price**.

    ### 🔹 Real Estate Trends:
    {trend_summary}

    ### 🔹 User Info:
    - **Year of Purchase**: {year}
    - **Salary**: €{salary}
    - **Budget**: €{budget}

    ### 🔹 Task:
    1️⃣ Predict the **average house price** in {year}.  
    2️⃣ Determine if the user can **afford a house** based on their salary & budget.  
    3️⃣ Recommend **the best county** to buy property in.  
    4️⃣ Consider factors like price trends, salary multipliers, and investment potential.

    Give a structured response with clear justifications. 
    The response should start with initial clear cut bullet point summary of the Predicted price, affordability range, and the recommended county to buy from.
    """

    # Call OpenAI API (Replace with your API key)
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"))

    response = client.completions.create(
            model="gpt-3.5-turbo",  # Use a free LLM like Mistral-7B if needed
            prompt=prompt,
    )

    # return the LLM response
    return response.choices[0].text.strip()
