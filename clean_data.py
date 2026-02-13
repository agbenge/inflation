
# combine the table  using tyear	tmonth 

# four output 
# left join 
# rigth join
#  intersect join
#  union join

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



# Save all results to one Excel file with multiple sheets
output_path = "data/combined_join_results.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    left_join.to_excel(writer, sheet_name='Left_Join', index=False)
    right_join.to_excel(writer, sheet_name='Right_Join', index=False)
    inner_join.to_excel(writer, sheet_name='Inner_Join', index=False)
    union_join.to_excel(writer, sheet_name='Union', index=False)



left_join.to_excel("data/left_join.xlsx", index=False)
right_join.to_excel("data/right_join.xlsx", index=False)
inner_join.to_excel("data/inner_join.xlsx", index=False)
union_join.to_excel("data/union_join.xlsx", index=False)


output_path


 