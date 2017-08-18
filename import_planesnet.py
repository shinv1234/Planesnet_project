import numpy as np
import gzip
import pickle

def import_planesnet(file, train_size, seed=None): # file: r'planesnet.pklz'
    with gzip.open(file, 'rb') as f:
        planesnet = pickle.load(f, encoding='latin1')
        
    lst = list()
    for row in planesnet['data']:
        lst.append(row.reshape((3,400)).T.reshape((20,20,3)))
    data = np.array(lst)
    
    labels = np.array(planesnet['labels'])
    
    plane_data = data[labels == 1]
    plane_labels = labels[labels == 1]
    no_plane_data = data[labels == 0]
    no_plane_labels = labels[labels == 0]
    
    test_size = len(labels) - train_size
    
    pct = 1 / (train_size + test_size)
    train_ratio = pct * train_size
    test_ratio = pct * test_size
    train_ratio, test_ratio
    
    train_data = np.vstack((plane_data[0:round(len(plane_labels) * train_ratio)], 
                        no_plane_data[0:round(len(no_plane_labels) * train_ratio)]))
    train_labels = np.hstack((plane_labels[0:round(len(plane_labels) * train_ratio)],
                              no_plane_labels[0:round(len(no_plane_labels) * train_ratio)]))
    test_data = np.vstack((plane_data[round(len(plane_labels) * train_ratio):len(labels)],                        
                           no_plane_data[round(len(no_plane_labels) * train_ratio):len(labels)]))
    test_labels = np.hstack((plane_labels[round(len(plane_labels) * train_ratio):len(labels)],                         
                             no_plane_labels[round(len(no_plane_labels) * train_ratio):len(labels)]))
    
    if seed != None:
        np.random.seed(seed)
    else:
        pass
    
    train_idx = np.random.permutation(len(train_labels))
    test_idx = np.random.permutation(len(test_labels))
    train_data = train_data[train_idx]
    train_labels = train_labels[train_idx]
    test_data = test_data[test_idx]
    test_labels = test_labels[test_idx]
    
    
    del planesnet, lst, data, labels, plane_data, plane_labels, no_plane_data, no_plane_labels, pct, train_ratio, test_ratio, train_idx, test_idx

    return train_data, train_labels, test_data, test_labels

def import_planesnet2(file, train_size): # file: r'planesnet.pklz'
    with gzip.open(file, 'rb') as f:
        planesnet = pickle.load(f, encoding='latin1')
        
    lst = list()
    for row in planesnet['data']:
        lst.append(row.reshape((3,400)).T.reshape((20,20,3)))
    data = np.array(lst)
    
    labels = np.array(planesnet['labels'])
    
    index =  np.random.permutation(len(labels))
    
    train_index = index[:train_size]
    test_index = index[train_size:len(index)]
    
    train_data = data[train_index]
    train_labels = labels[train_index]
    train_labels = train_labels.reshape(len(train_labels), 1)
    
    test_data = data[test_index]
    test_labels = labels[test_index]
    test_labels = test_labels.reshape(len(test_labels), 1)
    
    del lst, data, labels, index, train_index, test_index

    return train_data, train_labels, test_data, test_labels