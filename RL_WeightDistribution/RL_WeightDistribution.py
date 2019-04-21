import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

date = pd.date_range(start = "05/04/2017", periods = 2776, freq = "30min")
h    = pd.DataFrame(0, index = np.arange(9), columns = ["bin", "dd3", "dd4", "dd5"])

for j in range(3):
    df  = pd.read_csv("<path>/" + str(j + 3) + "0weights1.csv")
    v   = np.sort(df.values, 1)[:, 0:5]
    res = np.sum(v, axis = 1) * 50
    d   = pd.DataFrame(data = [date, res]).T
    d.columns = ['date', 'value']
    for i in range(9):
        h.iloc[i, 0]     = i + 1
        h.iloc[i, j + 1] = pd.DataFrame.mean(d.iloc[336 * i : 336 * (i + 1) - 1,1])
    

h.plot(x = "bin",y = ["dd3", "dd4", "dd5"], kind = "bar",
       color = ["blue", "red", "gray"], legend = False)
pl.xlabel("Week number") 
pl.ylabel("Value (%)")
pl.title("Av. investment (in %) for 5 least invested coins")
