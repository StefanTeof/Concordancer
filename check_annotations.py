import pandas as pd

df = pd.read_excel('./Orwell chapter 1.xlsx', header=None)

column_to_check = df.iloc[:, 1]

grouped = column_to_check.groupby(column_to_check.str[0])

for group_name, group_data in grouped:
    if not group_data.str.len().eq(group_data.str.len().iloc[0]).all():
        print(f'Values in group {group_name} do not have the same string length.')
        print(group_data)

all_same_length = grouped.apply(lambda x: x.str.len().eq(x.str.len().iloc[0]).all())

print(all_same_length)
