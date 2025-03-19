import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    
    # Convert 'Date of Sale' to datetime
    df["Date of Sale"] = pd.to_datetime(df["Date of Sale (dd/mm/yyyy)"], format="%d/%m/%Y", errors="coerce")
    
    # Convert 'Price (€)' to numeric
    df["Price (€)"] = df["Price"].replace('[^\d.]', '', regex=True).astype(float)
    
    # Drop missing values
    df = df.dropna(subset=["Price (€)", "County"])

    # Extract year and month
    df["Year"] = df["Date of Sale"].dt.year
    df["Month"] = df["Date of Sale"].dt.month

    return df
