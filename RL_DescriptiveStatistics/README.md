[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **RL_DescriptiveStatistics** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: RL_DescriptiveStatistics

Published in: 'A leveraged investment strategy using Deep Reinforcement Learning'

Description: 'Outputs descriptive statistics for all coins used in the experiments (prices relative to Bitcoin, from 07/2015 to 10/2018, if available).'

Keywords: 'reinforcement learning, neural network, machine learning, portfolio management, cryptocurrency'
 
Author: Ilyas Agakishiev

See also: RL_MainComputation, RL_CoinFigures

Submitted: 23.04.2019

Input: 
- df: Coin prices, contained in the CSV-files of the "Database" folder

Output:
- ds: Table with descriptive statistics (Min, max, median, mean, quartile1, quartile3)
```

### PYTHON Code
```python

import pandas as pd
import numpy as np

coin = ["BCH", "DASH", "DGB", "ETC", "ETH", "FCT", "GNT", "LTC", "rev_USDT",
        "STR", "XEM", "XMR", "XRP", "ZEC"]
ds = pd.DataFrame(index = coin, columns = ["Min", "Quartile1", "Mean", 
                                           "Median", "Quartile3", "Max"])

print("Processing...")

for i in coin:
    print(i)
    # Import from "Database"    
    df      = pd.read_csv("<path>/" + i + ".csv")
    df.date = df.date.astype(int)
    df      = df[df.date % (900*48) == 0]
    df      = df.sort_values(by = "date")
    df.date = pd.to_datetime(df.date, unit = 's')
    df      = df[df.date >= "07/01/2015"]
    df      = df[df.date < "11/01/2018"]
    df["Returns"] = 0
    for j in range(30,len(df.iloc[:,0])):
        df.Returns.iloc[j] = (df.close.iloc[j] - df.close.iloc[j-30]) / df.close.iloc[j-30] * 100
    df      = df.iloc[30:, :]    
    df      = df.sort_values(by = "Returns")

    ds.Min[i]       = min(df.Returns.values)   
    ds.Quartile1[i] = df.Returns.iloc[round(len(df.Returns.values) / 4)]
    ds.Mean[i]      = np.mean(df.Returns.values)
    ds.Median[i]    = np.median(df.Returns.values)
    ds.Quartile3[i] = df.Returns.iloc[round(len(df.Returns.values) * 3 / 4)]
    ds.Max[i]       = max(df.Returns.values)
        
print(ds)

```

automatically created on 2019-05-13