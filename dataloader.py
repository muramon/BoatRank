import os
import torch
import pickle
import pandas as pd
import numpy as np
import torch.utils.data as data
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

## due to the restriction of 4GB ##
max_bytes = 2 ** 31 - 1

SCALER_LEVEL = ['QUERY', 'DATASET']
SCALER_ID = ['MinMaxScaler', 'RobustScaler', 'StandardScaler']

MSLETOR = ['MQ2007_Super', 'MQ2008_Super', 'MQ2007_Semi', 'MQ2008_Semi', 'MQ2007_List', 'MQ2008_List']
MSLETOR_SUPER = ['MQ2007_Super', 'MQ2008_Super']
MSLETOR_SEMI = ['MQ2007_Semi', 'MQ2008_Semi']
MSLETOR_LIST = ['MQ2007_List', 'MQ2008_List']
MSLRWEB = ['MSLRWEB10K', 'MSLRWEB30K']
BOATRACE = ['BOATRACE']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pickle_save(target, file):
    bytes_out = pickle.dumps(target, protocol=4)
    with open(file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load(file):
    file_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        bytes_in = bytearray(0)
        for _ in range(0, file_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data


class L2RDataLoader():
    """
	An abstract loader for learning-to-rank datasets
	"""

    def __init__(self, file, buffer=True):
        '''
		:param file:   the specified data file, e.g., the fold path when performing k-fold cross validation
		:param buffer: buffer the primarily parsed data
		'''
        self.df = None
        self.file = file
        self.buffer = buffer

    def load_data(self):
        pass

    def filter(self):
        pass


class MSL2RDataLoader(L2RDataLoader):
    """
	The data loader for MS learning-to-rank datasets
	"""

    def __init__(self, file, data_id=None, buffer=True):
        super(MSL2RDataLoader, self).__init__(file=file, buffer=buffer)

        self.data_id = data_id
        # origianl data as dataframe
        self.df_file = file[:file.find('.txt')].replace('Fold',
                                                        'BufferedFold') + '.df'  # the original data file buffer as a dataframe

        pq_suffix = 'PerQ'

        # plus scaling
        self.scale_data = True
        self.scaler_id = 'StandardScaler'

        if self.scale_data:
            pq_suffix = '_'.join([pq_suffix, 'QS', self.scaler_id])

        self.perquery_file = file[:file.find('.txt')].replace('Fold', 'BufferedFold') + '_' + pq_suffix + 'nosort' + '.np' #+

    def load_data(self):
        '''
		Load data at a per-query unit consisting of {scaled} {des-sorted} document vectors and standard labels
		:param given_scaler: scaler learned over entire training data, which is only needed for dataset-level scaling
		:return:
		'''
        if self.data_id in MSLETOR:
            self.num_features = 46
        elif self.data_id in MSLRWEB:
            self.num_features = 136
        elif self.data_id in BOATRACE:
            self.num_features = 10

        self.feature_cols = [str(f_index) for f_index in range(1, self.num_features + 1)]

        if os.path.exists(self.perquery_file):
            list_Qs = pickle_load(self.perquery_file)
            return list_Qs
        else:
            self.get_df_file()

            self.ini_scaler()

            list_Qs = []
            qids = self.df.qid.unique()
            np.random.shuffle(qids)
            for qid in qids:
                sorted_qdf = self.df[self.df.qid == qid].sort_values('rele_truth', ascending=False)

                doc_reprs = sorted_qdf[self.feature_cols].values
                if self.scale_data:
                    doc_reprs = self.scaler.fit_transform(doc_reprs) #normalization

                #print(doc_reprs)

                doc_labels = sorted_qdf['rele_truth'].values

                # doc_ids    = sorted_qdf['#docid'].values # commented due to rare usage

                list_Qs.append((qid, doc_reprs, doc_labels))

            if self.buffer: pickle_save(list_Qs, file=self.perquery_file)

            return list_Qs

    def load_data_shuffle(self):
        '''
        Load data at a per-query unit consisting of {scaled} {des-sorted} document vectors and standard labels
        :param given_scaler: scaler learned over entire training data, which is only needed for dataset-level scaling
        :return:
        '''
        if self.data_id in MSLETOR:
            self.num_features = 46
        elif self.data_id in MSLRWEB:
            self.num_features = 136
        elif self.data_id in BOATRACE:
            self.num_features = 10

        self.feature_cols = [str(f_index) for f_index in range(1, self.num_features + 1)]

        if os.path.exists(self.perquery_file):
            list_Qs = pickle_load(self.perquery_file)
            return list_Qs
        else:
            self.get_df_file()

            self.ini_scaler()

            list_Qs = []
            qids = self.df.qid.unique()
            np.random.shuffle(qids)
            for qid in qids:
                df = self.df[self.df.qid == qid]#.sample(flac=1)#.sort_values('rele_truth', ascending=False)
                shuffled_qdf = df.take(np.random.permutation(len(df)))

                doc_reprs = shuffled_qdf[self.feature_cols].values
                if self.scale_data:
                    doc_reprs = self.scaler.fit_transform(doc_reprs)  # normalization

                doc_labels = shuffled_qdf['rele_truth'].values

                # doc_ids    = sorted_qdf['#docid'].values # commented due to rare usage

                list_Qs.append((qid, doc_reprs, doc_labels))

            if self.buffer: pickle_save(list_Qs, file=self.perquery_file)

            return list_Qs

    def load_data_nosort(self):
        '''
        Load data at a per-query unit consisting of {scaled} {des-sorted} document vectors and standard labels
        :param given_scaler: scaler learned over entire training data, which is only needed for dataset-level scaling
        :return:
        '''
        if self.data_id in MSLETOR:
            self.num_features = 46
        elif self.data_id in MSLRWEB:
            self.num_features = 136
        elif self.data_id in BOATRACE:
            self.num_features = 10

        self.feature_cols = [str(f_index) for f_index in range(1, self.num_features + 1)]

        if os.path.exists(self.perquery_file):
            list_Qs = pickle_load(self.perquery_file)
            return list_Qs
        else:
            self.get_df_file()

            self.ini_scaler()

            list_Qs = []
            qids = self.df.qid.unique()
            np.random.shuffle(qids)
            for qid in qids:
                #sorted_qdf = self.df[self.df.qid == qid].sort_values('rele_truth', ascending=False)
                no_sorted_qdf = self.df[self.df.qid == qid]
                #print(no_sorted_qdf)

                doc_reprs = no_sorted_qdf[self.feature_cols].values
                if self.scale_data:
                    doc_reprs = self.scaler.fit_transform(doc_reprs)  # normalization

                print(doc_reprs)

                doc_labels = no_sorted_qdf['rele_truth'].values

                # doc_ids    = sorted_qdf['#docid'].values # commented due to rare usage

                list_Qs.append((qid, doc_reprs, doc_labels))
                #print(list_Qs)

            if self.buffer: pickle_save(list_Qs, file=self.perquery_file)

            return list_Qs

    def get_df_file(self):
        ''' Load original data file as a dataframe. If buffer exists, load buffered file. '''

        if os.path.exists(self.df_file):
            self.df = pd.read_pickle(self.df_file)
        else:
            if self.data_id in MSLETOR:
                self.df = self.load_LETOR4()
            elif self.data_id in MSLRWEB:
                self.df = self.load_MSLRWEB()
            elif self.data_id in BOATRACE:
                self.df = self.load_BOATRACE()

            if self.buffer:
                parent_dir = Path(self.df_file).parent
                if not os.path.exists(parent_dir): os.makedirs(parent_dir)
                self.df.to_pickle(self.df_file)

    def load_LETOR4(self):
        '''  '''
        df = pd.read_csv(self.file, sep=" ", header=None)
        df.drop(columns=df.columns[[-2, -3, -5, -6, -8, -9]], axis=1, inplace=True)  # remove redundant keys
        # print(self.num_features, len(df.columns) - 5)
        assert self.num_features == len(df.columns) - 5

        for c in range(1, self.num_features + 2):  # remove keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

        df.columns = ['rele_truth', 'qid'] + self.feature_cols + ['#docid', 'inc', 'prob']

        if self.data_id in MSLETOR_SEMI and self.data_dict['unknown_as_zero']:
            self.df[self.df[self.feature_cols] < 0] = 0

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)  # additional binarized column for later filtering

        return df

    def load_BOATRACE(self):
        '''  '''
        df = pd.read_csv(self.file, sep=" ", header=None)
        df.drop(columns=df.columns[[1,3,5,7,9,11,13,15,17,19,21]], axis=1, inplace=True)  # remove redundant keys
        #print(df)
        #print(self.num_features, len(df.columns))
        assert self.num_features == len(df.columns) - 2

        #for c in range(1, self.num_features + 2):  # remove keys per column from key:value
        #    df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

        df.columns = ['rele_truth', 'qid'] + self.feature_cols #+ ['#docid', 'inc', 'prob']

        #if self.data_id in MSLETOR_SEMI and self.data_dict['unknown_as_zero']:
        #    self.df[self.df[self.feature_cols] < 0] = 0

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)  # additional binarized column for later filtering
        #print(df)
        return df


    def load_MSLRWEB(self):
        '''  '''
        df = pd.read_csv(self.file, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True)  # remove the line-break
        assert self.num_features == len(df.columns) - 2

        for c in range(1, len(df.columns)):  # remove the keys per column from key:value
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.split(":")[1])

        df.columns = ['rele_truth', 'qid'] + self.feature_cols

        for c in ['rele_truth'] + self.feature_cols:
            df[c] = df[c].astype(np.float32)

        df['rele_binary'] = (df['rele_truth'] > 0).astype(np.float32)  # additional binarized column for later filtering

        return df

    def ini_scaler(self):
        assert self.scaler_id in SCALER_ID
        if self.scaler_id == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif self.scaler_id == 'RobustScaler':
            self.scaler = RobustScaler()
        elif self.scaler_id == 'StandardScaler':
            self.scaler = StandardScaler()


class L2RDataset(data.Dataset):
    '''
	Buffering tensored objects can save much time.
	'''

    def __init__(self, file, data_id):
        loader = MSL2RDataLoader(file=file, data_id=data_id)
        perquery_file = loader.perquery_file

        torch_perquery_file = perquery_file.replace('.np', '.torch')

        if os.path.exists(torch_perquery_file):
            self.list_torch_Qs = pickle_load(torch_perquery_file)
        else:
            self.list_torch_Qs = []

            # sorted
            #list_Qs = loader.load_data()
            # no sorted
            list_Qs = loader.load_data_nosort()
            # shuffle
            #list_Qs = loader.load_data_shuffle()
            #print(list_Qs)
            list_inds = list(range(len(list_Qs)))
            for ind in list_inds:
                qid, doc_reprs, doc_labels = list_Qs[ind]

                torch_batch_rankings = torch.from_numpy(doc_reprs).type(torch.FloatTensor)

                torch_batch_std_labels = torch.from_numpy(doc_labels).type(torch.FloatTensor)

                self.list_torch_Qs.append((qid, torch_batch_rankings, torch_batch_std_labels))

            # buffer
            pickle_save(self.list_torch_Qs, torch_perquery_file)

    def __getitem__(self, index):
        qid, torch_batch_rankings, torch_batch_std_labels = self.list_torch_Qs[index]
        return qid, torch_batch_rankings, torch_batch_std_labels

    def __len__(self):
        return len(self.list_torch_Qs)


def transform_ls(q_sample_ls, cols_to_drop):
    """
        input dataframe
        transforms datafram into tensor
    """

    label_tensor_ls = torch.tensor(np.asarray([q_sample['y'] for q_sample in q_sample_ls]))
    data_tensor_ls = torch.tensor(np.asarray([q_sample[feature_cols].values.astype('float') \
                    for q_sample in q_sample_ls])).float()
    return {'y': label_tensor_ls, 'data': data_tensor_ls}

class RANKNET_TEST_DS(Dataset):
    """Document Ranking Dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the txt file with q_id.
            root_dir (string): Directory with all the query_details.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feats_to_drop = feats_to_drop

    def __len__(self):
        return len(self.meta_file)

    def __getitem__(self, idx):
        q_fname = os.path.join(self.root_dir,str(self.meta_file.iloc[idx]['qid']))
        q_data = pd.read_csv("{}.csv".format(q_fname))
        z_ls = [q_data.iloc[i] for i in range(len(q_data))]
        sample_ls = transform_ls(z_ls,self.feats_to_drop)
        return sample_ls

feats_to_drop = ['doc_id','inc','prob','qid','y']
feature_cols = [str(i) for i in range(1, 47)]

def torch_batch_triu(batch_mats=None, k=0, pair_type="All", batch_std_labels=None):
    """
	Get unique document pairs being consistent with the specified pair_type. This function is used to avoid duplicate computation.
	All:        pairs including both pairs of documents across different relevance levels and
			   pairs of documents having the same relevance level.
	UpNoTies:   the pairs consisting of two documents of the same relevance level are removed
	UpNo00:     the pairs consisting of two non-relevant documents are removed
	"""

    assert batch_mats.size(0) == batch_mats.size(1)
    #assert pair_type in PAIR_TYPE

    m = batch_mats.size(0)  # the number of documents

    if pair_type == "All":
        row_inds, col_inds = np.triu_indices(m,k=k)


    elif pair_type == "UpNo00":
        assert batch_std_labels.size(0) == 1

        row_inds, col_inds = np.triu_indices(m, k=k)
        std_labels = torch.squeeze(batch_std_labels, 0)
        labels = std_labels.cpu().numpy() if torch.cuda.is_available() else std_labels.data.numpy()

        pairs = [e for e in zip(row_inds, col_inds) if
                 not (0 == labels[e[0]] and 0 == labels[e[1]])]  # remove pairs of 00 comparisons
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    elif pair_type == "UpNoTies":
        assert batch_std_labels.size(0) == 1
        std_labels = torch.squeeze(batch_std_labels, 0)

        row_inds, col_inds = np.triu_indices(m, k=k)
        labels = std_labels.cpu().numpy() if torch.cuda.is_available() else std_labels.data.numpy()
        pairs = [e for e in zip(row_inds, col_inds) if
                 labels[e[0]] != labels[e[1]]]  # remove pairs of documents of the same level
        row_inds = [e[0] for e in pairs]
        col_inds = [e[1] for e in pairs]

    tor_row_inds = torch.LongTensor(row_inds).to(device) if torch.cuda.is_available() else torch.LongTensor(row_inds)
    tor_col_inds = torch.LongTensor(col_inds).to(device) if torch.cuda.is_available() else torch.LongTensor(col_inds)
    batch_triu = batch_mats[tor_row_inds, tor_col_inds]

    return batch_triu, tor_row_inds, tor_col_inds