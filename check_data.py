import pandas as pd

# sp500_data.xlsx 파일 확인
df = pd.read_excel('sp500_data.xlsx')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('Date column type:', type(df.iloc[0,0]))
print('Sample dates:')
for i in range(5):
    print(f'  {i}: {df.iloc[i,0]} (type: {type(df.iloc[i,0])})')

print('\nFirst few rows:')
print(df.head())

print('\nDate column info:')
print(df.iloc[:5, 0])