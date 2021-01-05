import numpy as np
import pandas as pd
from IPython.display import HTML
import base64
import smote_variants as sv
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from numpy import inf

dataset = pd.read_csv('Acc_new1.csv', delimiter=',')
X1 = dataset.iloc[:,1:]
X = X1.values
y = dataset.Index.to_list()
y.sort()
#print(X)
#print(y)

for i in range(len(y)):
    y[i] = y[i] - 1
    
X[X == inf] = np.finfo(np.float64).max


colors= ['orange', 'red', 'blue']

'''

#plotting graph

plt.figure(figsize=(10, 5))
for i in range(2,-1,-1):
    i = np.int32(i)
    plt.scatter(X[y == i][:,2], X[y == i][:,1], label='class %d' % i, c= colors[i])
plt.title('original dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.legend()


a = []
for i in np.unique(y):
    a.append(np.sum(y == i))
    print("class %d - samples: %d" % (i, np.sum(y == i)))
print("****")


courses = ['ACC','CAS','VEH']
values = a
   
fig = plt.figure(figsize = (10, 5)) 
   
plt.bar(courses, values, color ='maroon', width = 0.4)  
plt.xlabel("Datasets") 
plt.ylabel("No. of data points")
plt.show() 

''' 

oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())

X_samp, y_samp= oversampler.sample(X, y)

b = []
for i in np.unique(y_samp):
    b.append(np.sum(y_samp == i))
    print("class %d - samples: %d" % (i, np.sum(y_samp == i)))
print("****")

courses2 = ['ACC','CAS','VEH']
values2 = b
   
fig = plt.figure(figsize = (10, 5)) 
   
plt.bar(courses2, values2, color ='maroon', width = 0.4)  
plt.xlabel("Datasets") 
plt.ylabel("No. of data points")
plt.show() 

''' 
 
X_samp, y_samp= X_samp[len(X):], y_samp[len(y):]

for i in np.unique(y_samp):
    print("class %d - samples: %d" % (i, np.sum(y_samp == i)))
print("****")

    
#plotting upsampled graph
 
plt.figure(figsize=(10, 5))
for i in range(2,-1,-1):
    i = np.int32(i)
    plt.scatter(X[y == i][:,2], 
                X[y == i][:,1], 
                label='class %d' % i, 
                c=colors[i],
                marker='o')
    
    if len(y_samp[y_samp == i]):
        plt.scatter(X_samp[y_samp == i][:, 2], 
                    X_samp[y_samp == i][:, 1], 
                    label='class %d new samples' % i, 
                    c=colors[i], 
                    marker='x')
plt.title('oversampled dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.legend()

'''

#download csv
'''
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

df = pd.DataFrame(data=X_samp)
df.to_csv('abc.csv')
'''