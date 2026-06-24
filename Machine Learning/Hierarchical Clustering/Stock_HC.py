import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Dataset: Stocks with Attributes

ccdata = np.loadtxt('StockData.csv', delimiter=',', skiprows=1)

mms = MinMaxScaler()
mms.fit(ccdata)
ccdata = mms.transform(ccdata)

# Perform hierarchical clustering using the Ward method
linkage_matrix_stocks = linkage(ccdata, method='ward')

# Plot the dendrogram
plt.figure(figsize=(8, 6))

dendrogram(linkage_matrix_stocks, labels=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'BRK.B', 'LLY', 'AVGO','JPM', 'TSLA', 'JNJ', 'V', 'XOM', 'UNH', 'PG', 'MA', 'COST', 'HD','MRK', 'ORCL', 'ABBV', 'WMT', 'BAC', 'PEP', 'KO', 'CRM', 'CVX', 'TMO','ADBE', 'ACN', 'MCD', 'AMD', 'NFLX', 'LIN', 'DHR', 'WFC', 'IBM', 'TXN','PM', 'CAT', 'GE', 'AMGN', 'HON', 'MS', 'UNP', 'VZ', 'NKE', 'UPS','SBUX', 'BKNG', 'SPGI', 'GS', 'BLK', 'ISRG', 'AMAT', 'LMT', 'ADI', 'DE','MDLZ', 'SYK', 'C', 'CB', 'SCHW', 'CI', 'PGR', 'PLD', 'T', 'GILD','NOW', 'PYPL', 'MO', 'INTU', 'REGN', 'ZTS', 'EQIX', 'ICE', 'CME', 'FISV','ADP', 'CL', 'SO', 'DUKE', 'EOG', 'COP', 'SLB', 'KMI', 'AEP', 'D','EXC', 'NEE', 'MDT', 'BMY', 'CVS'], leaf_rotation=90)
plt.title("Hierarchicalal Clustering Dendrogram")
plt.xlabel("Stocks")
plt.ylabel("Distance")
plt.show()

