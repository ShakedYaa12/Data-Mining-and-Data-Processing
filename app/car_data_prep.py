import pandas as pd 
import numpy as np
import re
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def prepare_data(df):
    
    # After reviewing the 'manufactor' column, we found duplicates
    df['manufactor'] = df['manufactor'].replace('Lexsus', 'לקסוס')

    # After revewing the model column, we saw that some of the model names are written with the name of the manufactor and
    # the year saw we build a function to help us clean the column:
    def clean_model_name(manufactor, model_name):
        model_name = re.sub(r'\(\d{4}\)', '', model_name)
        model_name = re.sub(manufactor, '', model_name, flags=re.IGNORECASE)
        model_name = model_name.strip()
        return model_name

    df['model'] = df.apply(lambda row: clean_model_name(row['manufactor'], row['model']), axis=1)
    
    # Further cleaning 'model', 'Gear', and 'Engine_type' columns for remaining duplicates
    df['model'] = df['model'].replace('CIVIC', 'סיוויק')
    df['model'] = df['model'].replace('ACCORD', 'אקורד')
    df['model'] = df['model'].replace("ג`טה", "ג'טה")
    df['model'] = df['model'].replace("ג'אז", "ג`אז")
    df['model'] = df['model'].replace('מיטו / MITO', 'מיטו')
    df['model'] = df['model'].replace("ג'וק JUKE", "ג'וק")
    df['Gear'] = df['Gear'].replace('לא מוגדר', np.nan)
    df['Gear'] = df['Gear'].replace('אוטומט', 'אוטומטית')
    df['Engine_type'] = df['Engine_type'].replace('היבריד', 'היברידי')
    
    # Converting 'capacity_Engine' to numeric and removing outliers, assuming minimum capacity is 800
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
    df.loc[df['capacity_Engine'] < 800, 'capacity_Engine'] = np.nan
    
    # We assume that cars with the same manufactor and model will have the same capacity engine, gear and engine type
    def fill_missing_values(row, reference_data, columns_to_fill, reference_columns):
        for col in columns_to_fill:
            if pd.isnull(row[col]):
                # find a not missing values from similar cars
                similar_rows = reference_data[
                    (reference_data[reference_columns[0]] == row[reference_columns[0]]) &
                    (reference_data[reference_columns[1]] == row[reference_columns[1]])
                ]
                if not similar_rows.empty:
                    non_na_values = similar_rows[col].dropna()
                    if not non_na_values.empty:
                        row[col] = non_na_values.mode().values[0]
        return row
    
    columns_to_fill = ['Gear', 'Engine_type', 'capacity_Engine']
    reference_columns = ['manufactor', 'model']
    df = df.apply(lambda row: fill_missing_values(row, df, columns_to_fill, reference_columns), axis=1)
    
    # If there are still missing values, fill them with the most common value
    df['Engine_type'] = df['Engine_type'].replace(np.nan, 'דיזל')
    df['Gear'] = df['Gear'].replace(np.nan, 'אוטומטית')
    
    # Correcting engine capacity for Mitsubishi models based on additional research
    df.loc[(df['model'] == 'לנסר') | (df['model'] == 'לנסר הדור החדש'), 'capacity_Engine'] = 1500
    df.loc[df['model'] == "אטראז'", 'capacity_Engine'] = 1200
    df.loc[df['model'] == 'גרנדיס', 'capacity_Engine'] = 2400
    df.loc[df['model'] == 'IS300H', 'capacity_Engine'] = 2500
    df.loc[df['model'] == 'GT3000', 'capacity_Engine'] = 1600
    
    # Removing rows with missing 'capacity_Engine' because most of the data is non-null
    df = df.dropna(subset=['capacity_Engine'])
    
    # Assuming the previous owner of a first-hand car is the manufacturer
    df.loc[df['Hand'] == 1, 'Prev_ownership'] = 'יצרן'
    
    # Merging categories in 'Prev_ownership' with small value counts
    categories_to_merge = ["ליסינג", "השכרה", "חברה", "ממשלתי", "אחר", "מונית"]
    df['Prev_ownership'] = df['Prev_ownership'].replace('לא מוגדר', np.nan)
    df['Prev_ownership'] = df['Prev_ownership'].replace(categories_to_merge, 'לא פרטית')
    
    # We assume that the previous owner of a car which is first hand is the manufactor
    df.loc[df['Hand'] == 1, 'Prev_ownership'] = 'יצרן'
    
    # Function to fill missing values in the "Prev_ownership column" by the distribution
    def fill_na_proportionally(series):
        # finding the distribution
        distribution = series[series != 'יצרן'].value_counts(normalize=True)
        # Creating a copy
        series_copy = series.copy()
        # finding missing indices
        na_indices = series_copy[series_copy.isna()].index
        # creating random values by the distribution
        fill_values = np.random.choice(distribution.index, size=len(na_indices), p=distribution.values)
        # filling missing values
        series_copy.loc[na_indices] = fill_values
        return series_copy

    df['Prev_ownership'] = fill_na_proportionally(df['Prev_ownership'])
    
    # Dropping 'Curr_ownership' column because we assume that the current ownership is private
    df = df.drop(columns=['Curr_ownership'])
    
    # Dropping 'City' and 'Area' columns due to messiness and assumed irrelevance to price
    df = df.drop(columns=['City', 'Area'])
    
    # Converting 'Km' column from string to numeric format
    df['Km'] = df['Km'].str.replace(',', '')
    df['Km'] = df['Km'].replace({'0': np.nan, 'None': np.nan})
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    
    # Correcting mileage written in thousands (140 instead of 140,000)
    df.loc[df['Km'] < 1000, 'Km'] = df['Km'] * 1000
    np.set_printoptions(suppress=True)
   
    # Filling missing 'Km' values based on average annual mileage data
    def fill_km(row):
        if pd.isna(row['Km']):
            age = 2024 - row['Year']
            if row['Prev_ownership'] in ['פרטית', 'יצרן']:
                return age * 15300
            elif row['Prev_ownership'] in ['לא פרטית']:
                return age * 27400
        return row['Km']
    
    df['Km'] = df.apply(fill_km, axis=1)
    
    # Feature Engineering:
    
    # Authorized Service - A binary column indicating whether the car is serviced at an authorized garage based on the description
    df['Authorized_service'] = np.where(df['Description'].str.contains('מוסך מורשה', na=False), 1, 0)
    
    # Km Per Year, removing outliers
    df["KM_Per_Year"] = df["Km"] / (2024 - df["Year"])
    pd.set_option('display.float_format', '{:.3f}'.format)
    df = df[df['KM_Per_Year'] <= 100000]

    # Engine efficiency metric based on engine capacity and annual mileage
    df['capacity_Engine'] = df['capacity_Engine'].astype(str)
    df['capacity_Engine'] = df['capacity_Engine'].str.replace(',', '').astype(float)
    df['Engine_Efficiency'] = df['capacity_Engine'] / df['KM_Per_Year']
    
    # Vintage Car - A car that is at least 30 years old
    df['Vintage_Car'] = np.where(df['Year'] < 1995, 1, 0)

    # Dropping columns deemed irrelevant to the data
    df = df.drop(columns=['Cre_date', 'Repub_date', 'Pic_num', 'Description'])
    
    # Dropping columns with too many missing values
    df = df.drop(columns=['Color', 'Test', 'Supply_score'])
    
    # Preparing categorical columns for OneHotEncoder
    categorical_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership']


    # Creating ColumnTransformer with OneHotEncoder for categorical columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), categorical_columns)
        ],
        remainder='passthrough'
    )

    # Fitting the encoder and transforming categorical columns
    df_encoded = column_transformer.fit_transform(df)

    # Getting names of the encoded columns
    encoded_columns = column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_columns)

    # Creating a new DataFrame with encoded columns and remaining original columns
    df_encoded = pd.DataFrame(df_encoded, columns=encoded_columns.tolist() + df.drop(columns=categorical_columns).columns.tolist())

    # Converting encoded columns to numeric type
    for col in encoded_columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col])

    df = df_encoded

    # Changing all columns to their correct type
    # Adjust the types according to the columns available after transformation
    columns_to_convert = {
        'Year': 'int',  
        'Hand': 'int', 
        'capacity_Engine': 'float', 
        'Price': 'float', 
        'Km': 'float',        
        "Authorized_service": 'int',
        "KM_Per_Year": 'float',               
        "Engine_Efficiency": 'float',
        "Vintage_Car": 'int'
    }

    for col, col_type in columns_to_convert.items():
        if col in df.columns:
            df[col] = df[col].astype(col_type)

    return df

