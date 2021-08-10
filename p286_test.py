import pickle

import pandas as pd

with open("myClass.pickle", "rb") as r:
    read_data = pickle.load(r)

result = read_data.predict(pd.DataFrame({'weight':[2278.7]}))
print(result)
print('-------------',read_data.intercept_)