 
# Import required libraries
import pandas as pd
from openpyxl import Workbook

# Load datasets
inflation = pd.read_excel("data/Inflation_Data_in_Excel.xlsx")
exchange = pd.read_excel("data/Monthly_Average_Exchange_Rates_Data_in_Excel.xlsx")

# Ensure join keys are integers
inflation['tyear'] = inflation['tyear'].astype(int)
inflation['tmonth'] = inflation['tmonth'].astype(int)
exchange['tyear'] = exchange['tyear'].astype(int)
exchange['tmonth'] = exchange['tmonth'].astype(int)

# Perform joins
left_join = pd.merge(inflation, exchange, on=['tyear', 'tmonth'], how='left')
right_join = pd.merge(inflation, exchange, on=['tyear', 'tmonth'], how='right')
inner_join = pd.merge(inflation, exchange, on=['tyear', 'tmonth'], how='inner')
union_join = pd.concat([inflation, exchange], ignore_index=True).drop_duplicates()

# --------------------------
# Add date column (year-month-01)
# --------------------------
for df in [left_join, right_join, inner_join, union_join]:
    df['date'] = pd.to_datetime(
        dict(year=df['tyear'], month=df['tmonth'], day=1)
    )
 

left_join.to_excel("data/clean_data/left_join.xlsx", index=False)
right_join.to_excel("data/clean_data/right_join.xlsx", index=False)
inner_join.to_excel("data/inner_join.xlsx", index=False)
union_join.to_excel("data/clean_data/union_join.xlsx", index=False)


# output_path


 