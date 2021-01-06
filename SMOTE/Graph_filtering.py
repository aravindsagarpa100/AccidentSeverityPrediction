import numpy as np
import pandas as pd
from IPython.display import HTML
import base64
import smote_variants as sv
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from numpy import inf
import csv

file1 = open("D:\Final year project\Dataset\Original\Acc.csv")
file2 = open("D:\Final year project\Dataset\Original\Cas.csv")
file3 = open("D:\Final year project\Dataset\Original\Veh.csv")
file4 = open("D:\Final year project\Dataset\After filtering\Acc_new1.csv")
file5 = open("D:\Final year project\Dataset\After filtering\Cas_new1.csv")
file6 = open("D:\Final year project\Dataset\After filtering\Veh_new1.csv")
a = []
b = []
a.append(len(list(csv.reader(file1))))
a.append(len(list(csv.reader(file2))))
a.append(len(list(csv.reader(file3))))
b.append(len(list(csv.reader(file4))))
b.append(len(list(csv.reader(file5))))
b.append(len(list(csv.reader(file6))))

N = 3
ind = np.arange(N) 
width = 0.35       
plt.bar(ind, a, width, label='Before filtering')
plt.bar(ind + width, b, width, label='After filtering')

plt.ylabel('No. of data points')

plt.xticks(ind + width / 2, ('Acc', 'Cas', 'Veh'))
plt.legend(loc='best')
plt.show()

