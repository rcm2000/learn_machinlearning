import pandas as pd
import numpy as np
df_covid = pd.read_csv('./Covid19InfState.csv')

print(df_covid)

df01 = df_covid[['decideCnt','stateDt']]
df02 = df01.astype({'decideCnt': int,

                  'stateDt': pd.datetime})


# j = df[df['stateDt'].dt.month == 7]
# print(j)
