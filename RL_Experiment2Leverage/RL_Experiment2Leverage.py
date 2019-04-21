import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

df = pd.read_csv('C:\\Users\\Ilyas Agakishiev\\Desktop\\FOR ANALYSE\\Pythonize\\weightsrec.csv', sep=";")

df.iloc[0, 0] = datetime(2018, 9, 2, 0, 0, 0)
for i in range(1, len(df.iloc[:,0])):
    df.iloc[i, 0] = df.iloc[i-1, 0] + timedelta(minutes = 30)
    
new_x = df.iloc[:, 0]
fig   = plt.figure()
ax    = fig.add_subplot(111, label="1")
ax.xaxis.set_major_formatter(DateFormatter('%d.%m'))
ax.plot_date(new_x, df.iloc[:, 1], fmt="b-", tz=None, xdate=True, linewidth=0.8)
plt.title("Leverage values")
plt.xlabel("Time")
plt.ylabel("Leverage")
plt.savefig("C:\\Users\\Ilyas Agakishiev\\Desktop\\FOR ANALYSE\\Pythonize\\weightsrec.png", dpi=500, bbox_inches="tight")
plt.show()
