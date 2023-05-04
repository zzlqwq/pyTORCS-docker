import json
import matplotlib.pyplot as plt

plt.ion()
plt.figure(1)

with open('../data.json', 'r') as f:
    data = json.load(f)
tick = 0

for points in data["data"]:
    plt.clf()
    x = points[::2]
    y = points[1::2]
    y = [-i for i in y]
    plt.xlim(-20, 20)
    plt.ylim(-10, 200)
    plt.plot(y, x, 'ro')
    tick += 1
    print(tick)
    plt.pause(0.0000001)
