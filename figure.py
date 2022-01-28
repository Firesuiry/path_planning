import matplotlib.pyplot as plt
import json
import numpy as np

with open('ep_rs.json', 'r') as f:
    data = json.loads(f.read())

x = [i for i in range(len(data))]
data2 = []
for i in range(len(data)):
    data2.append(np.mean(data[i:i + 1 + int(0.02 * len(data))]))
data2_mean = np.mean(data)
data2_mean = [data2_mean for i in range(len(data))]
plt.plot(x, data, x, data2, x, data2_mean)
plt.show()
