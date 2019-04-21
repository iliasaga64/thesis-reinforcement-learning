import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

df = pd.read_csv('C:\\Users\\Ilyas Agakishiev\\Desktop\\FOR ANALYSE\\Pythonize\\weights.csv', sep=";")

df.iloc[0, 0] = datetime(2017, 5, 4, 0, 0, 0)
for i in range(1, len(df.iloc[:,0])):
    df.iloc[i, 0] = df.iloc[i-1, 0] + timedelta(minutes = 30)
    
new_x = df.iloc[:, 0]
fig   = plt.figure()
ax    = fig.add_subplot(111, label="1")
ax.xaxis_date()
ax.set_xlim(["05/01/2017","07/03/2017"])
ax.xaxis.set_major_formatter(DateFormatter('%d.%m'))
plt.scatter(new_x.tolist(), df.iloc[:, 1], c="blue", s=2, marker="x")
plt.title("Leverage values for " + r"$D_{target} = 0.1$")
plt.xlabel("Time")
plt.ylabel("Leverage")
plt.savefig("C:\\Users\\Ilyas Agakishiev\\Desktop\\FOR ANALYSE\\Pythonize\\weightsrec.png", dpi=500, bbox_inches="tight")
plt.show()
