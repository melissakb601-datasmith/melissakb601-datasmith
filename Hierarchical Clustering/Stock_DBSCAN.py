import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

ccdata = np.loadtxt("StockData.csv", delimiter=',', skiprows=1)

#mms = MinMaxScaler()
#mms.fit(ccdata)
#ccdata = mms.transform(ccdata)

# eps is the radius and min_samples is the number of core points
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan.fit(ccdata)
print(dbscan.labels_)
