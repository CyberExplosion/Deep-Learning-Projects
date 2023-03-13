'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None, dDataset='cora'):
        super(Dataset_Loader, self).__init__(dName, dDescription)

        # ! Testing purposes
        self.dataset_name = dDataset
        self.dataset_source_folder_path = 'data/cora'
        # self.dataset_source_folder_path = 'P5/data/cora'

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # ! load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        # Create sparse matrix, for all features (one hot encoded), not including the id and the label
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        # one hot encode the target label
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])
        # Both features and labels use index to access

        # ! load link data from file and build graph
        # created array of node ID
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        # Create a map of [node ID to value from 0 to 2708 (num Nodes)]
        idx_map = {j: i for i, j in enumerate(idx)}
        # Create a map of [value from 0 to 2708 (numNodes) to node ID]
        reverse_idx_map = {i: j for i, j in enumerate(idx)}

        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        # edge_unordered is in the form of array = [node ID 0 cited, node ID 0 cited from, node ID 1 cited, node ID 1 cited from, ....]
    
        # ? Turn all the node ID to the index of the idx list -> You can use the idx list to get the node ID
        # Then reshape it back to the 2d array version (before being flatten)
        # So now we have [index of the cited, index of the cited from] as one row (out of 5429)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        
        # print(f'Shape of the edges: {edges.shape}') # 5429 links
        # print(f'Value of edges: {edges[0:2]}')
        # return

        #? https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html
        # coo matrix let you create a sparse matrix using row,col,data (3 array) to indicate position of value that matters
        # np.ones(edges.shape[0]) = the data array. The data is array of 1 with size = 5429 links
        # row array is all index of the cited
        # col array is all index of the cited from
        # Combine all we get an adjacency matrix, for ex: at row 163 and col 402 = 1
            # -> This means that index of the cited is 163 and index of the cited from si 402
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)

        # Symmetric normalized the Laplcian matrix -> just make it better in general
        # The adj matrices now both symmetric and normalized
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # Symmetrize the matrix ???
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)   # The normalized link matrix


        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        if self.dataset_name == 'cora':
            idx_train = range(200)
            idx_test = range(200, 1200)
            # idx_train = range(2708)
            # idx_test = range(200, 1200)
            # idx_val = range(1200, 1500)
        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'pubmed':
            idx_train = range(60)
            idx_test = range(6300, 7300)
            idx_val = range(6000, 6300)
        #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        elif self.dataset_name == 'cora-small':
            print('got in here')
            idx_train = range(5)
            idx_test = range(5, 10)
            # idx_val = range(5, 10)

        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)
        # idx_val = torch.LongTensor(idx_val)
        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test = {'idx_train': idx_train, 'idx_test': idx_test}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test': train_test}

# obj = Dataset_Loader()
# res = obj.load()
# print(res)