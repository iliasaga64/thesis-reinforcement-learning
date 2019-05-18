[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **RL_WeightDistribution** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: RL_WeightDistribution

Published in: 'A leveraged investment strategy using Deep Reinforcement Learning'

Description: 'Outputs a plot that displays the weight (in %) for the 5 least held coins for different target drawdowns in Experiment 1. A higher value means better diversification.'

Keywords: 'reinforcement learning, neural network, machine learning, portfolio management, cryptocurrency'
 
Author: Ilyas Agakishiev

See also: RL_MainComputation, RL_Experiment1Performance, RL_DrawdownFigures

Submitted: 23.04.2019

Input: 
- df: Tables "30weights1.csv", "40weights1.csv" or "50weights1.csv", which contain weights for the coins for a target drawdown of 30%, 40% and 50%, respectively.
```

![Picture1](RL_WeightDistribution.png)

### PYTHON Code
```python

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

```

automatically created on 2019-05-13