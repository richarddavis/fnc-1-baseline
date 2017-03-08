import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

with open('fnc-1/train_bodies.csv') as bodiesFile:
    reader = csv.DictReader(bodiesFile)
    bodies = list(reader)

with open('fnc-1/train_stances.csv') as stancesFile:
    reader = csv.DictReader(stancesFile)
    stances = list(reader)

btl = [len(b['articleBody'].split()) for b in bodies]
bl = [len(b['articleBody']) for b in bodies]

print("{} bodies. Body lengths: token mean={} std={}, char mean={}, std={}".format(
    len(bodies), np.mean(btl), np.std(btl), np.mean(bl), np.std(bl))
)

hpa = Counter(s['Body ID'] for s in stances)

print("{} headlines".format(len(stances)))

plt.hist(btl, 50)
plt.title("Article bodies, by token")
plt.xlabel('Article body lengths')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.hist(bl, 50)
plt.title("Article bodies, by character")
plt.xlabel('Article body lengths')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.hist(list(hpa.values()), 50)
plt.title("Headlines per article")
plt.xlabel("Headlines per article")
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
