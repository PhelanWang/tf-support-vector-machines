'''
By adidinchuk park. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-support-vector-machines
'''

import data as d
import numpy as np
from network import Network
import hyperparams as hp

# load and parse data
data = d.load_data('seeds//seeds_dataset.txt')
data = [row[0].replace('\t\t', '\t') for row in data]
data = [row.split('\t') for row in data]

# extract desired features and targets
inputs = np.array([list(map(float, [row[6], row[4]])) for row in data])
targets = np.transpose(np.array([list(map(float, [1 if int(row[7]) == 1 else -1,
                                                  1 if int(row[7]) == 2 else -1,
                                                  1 if int(row[7]) == 3 else -1])) for row in data]))


# extract desired features and targets
#inputs = np.array([list(map(float, [row[6], row[4]])) for row in data if int(row[7]) == 3 or int(row[7]) == 2])
#targets = np.transpose(np.array([list(map(float, [1 if int(row[7]) == 3 else -1,
#                                                  1 if int(row[7]) == 2 else -1])) for row in data if int(row[7]) == 3 or int(row[7]) == 2]))


# extract number of features and number of data clusters from the data
feature_count = len(inputs[0])
cluster_count = len(targets)

# init the network and train
net = Network(feature_count, cluster_count, gamma=hp.gamma)
net.train(inputs, targets, lr=hp.learning_rate, batch_size=hp.batch_size,
          epochs=hp.epochs, plot=True, kernel='gaussian')

# Example generating predictions for data
test_inx = [0, 85, 190, 10, 95, 205]
tmp = np.transpose(targets)
print(tmp[test_inx])
print(net.predict(inputs[test_inx]))
