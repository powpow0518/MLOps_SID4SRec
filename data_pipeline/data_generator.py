from tqdm import tqdm
from collections import defaultdict
import time
from training.utils import generate_rating_matrix_valid, generate_rating_matrix_test, neg_sample
from torch.utils.data import Dataset, DataLoader
import random
import copy 
import torch
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
import pickle as pkl

def get_user_sample(sample_file):
    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)
    return sample_seq

class DataGenerator(object):

    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.bs = args.train_batch_size
        self.data_file = args.data_path + args.dataset 
        print("========dataset :===========", self.data_file)
        self.create_dataset()
       
    def get_rating_matrix(self, seq_dic):
        n_items = seq_dic['n_items'] + 1
        valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['n_users'], n_items)
        test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['n_users'], n_items)    
        return valid_rating_matrix, test_rating_matrix

    def get_data_dic_from_db(self, db_url):
        """Build data_dic directly from PostgreSQL (interaction + item + category + brand).

        Snapshot isolation: if args.snapshot_item_id / snapshot_interaction_id are set,
        only items and interactions up to those IDs are used. This ensures training vocab
        and inference vocab are identical for a given model version.

        Remap: DB item_ids are remapped to consecutive train_ids (1-indexed) to eliminate
        gaps from sequence drift. Mappings are stored in the returned dict and later saved
        to model_args.pkl for use by inference scripts and serving.
        """
        import psycopg2

        if not db_url:
            raise ValueError("--use_db is set but --db_url (or DATABASE_URL env) is empty")

        snapshot_item_id = getattr(self.args, 'snapshot_item_id', None)
        snapshot_interaction_id = getattr(self.args, 'snapshot_interaction_id', None)

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # 1. item features — filtered by snapshot
        if snapshot_item_id is not None:
            cur.execute("""
                SELECT item_id,
                       COALESCE(price, 0.0),
                       COALESCE(category_id1, 0),
                       COALESCE(category_id2, 0),
                       COALESCE(brand_id, 0)
                FROM item
                WHERE item_id <= %s
                ORDER BY item_id
            """, (snapshot_item_id,))
        else:
            cur.execute("""
                SELECT item_id,
                       COALESCE(price, 0.0),
                       COALESCE(category_id1, 0),
                       COALESCE(category_id2, 0),
                       COALESCE(brand_id, 0)
                FROM item
                ORDER BY item_id
            """)
        item_rows = cur.fetchall()

        # Build remap: sorted DB item_ids → consecutive train_ids (1-indexed; 0 = padding)
        sorted_db_ids = [row[0] for row in item_rows]
        db2train = {db_id: train_id for train_id, db_id in enumerate(sorted_db_ids, start=1)}
        train2db = [0] + sorted_db_ids  # train2db[train_id] = db_id; train2db[0] = 0 (padding)
        n_items = len(sorted_db_ids) + 1  # 0=padding, 1..N=items

        # 2. user sequences — filtered by snapshot, remapped to train_ids
        if snapshot_item_id is not None and snapshot_interaction_id is not None:
            cur.execute("""
                SELECT user_id, item_id
                FROM interaction
                WHERE item_id <= %s AND interaction_id <= %s
                ORDER BY user_id, timestamp
            """, (snapshot_item_id, snapshot_interaction_id))
        elif snapshot_item_id is not None:
            cur.execute("""
                SELECT user_id, item_id
                FROM interaction
                WHERE item_id <= %s
                ORDER BY user_id, timestamp
            """, (snapshot_item_id,))
        else:
            cur.execute("""
                SELECT user_id, item_id
                FROM interaction
                ORDER BY user_id, timestamp
            """)

        user_items = defaultdict(list)
        for uid, iid in cur.fetchall():
            train_id = db2train.get(iid)
            if train_id:
                user_items[uid].append(train_id)

        filter_num = getattr(self.args, 'filter_num', 5)
        user_seq = [items for items in user_items.values() if len(items) >= filter_num]
        n_users = len(user_seq)

        # 3. vocab sizes
        cur.execute("SELECT COALESCE(MAX(category_id), 0) FROM category")
        n_categories = cur.fetchone()[0] + 1
        cur.execute("SELECT COALESCE(MAX(brand_id), 0) FROM brand")
        n_brands = cur.fetchone()[0] + 1

        conn.close()

        # 4. build items_feat matrix: shape (n_items, 4) indexed by train_id
        #    [price, cat1, cat2, brand] — matches get_feats_vec expectations
        items_feat = np.zeros((n_items, 4), dtype=np.float32)
        for db_id, price, cat1, cat2, brand in item_rows:
            train_id = db2train[db_id]
            items_feat[train_id] = [float(price), float(cat1), float(cat2), float(brand)]

        feature_size = 2 + 1 + n_categories + n_brands - 2

        return {
            'user_seq_wt': [],
            'user_seq': user_seq,
            'user_seq_wt_dic': {},
            'items_feat': items_feat,
            'category2id': {},
            'n_items': n_items,
            'n_users': n_users,
            'n_categories': n_categories,
            'n_brands': n_brands,
            'feature_size': feature_size,
            'db2train': db2train,
            'train2db': train2db,
            'snapshot_max_item_id': snapshot_item_id if snapshot_item_id is not None else (sorted_db_ids[-1] if sorted_db_ids else 0),
        }

    def get_data_dic(self, data_file):
        dat = pkl.load(open(f'{data_file}_all_multi_word.dat', 'rb'))
        data = {}

        user_reviews = dat['user_seq_token']
        data['user_seq_wt'] = []
        data['user_seq'] = []
        for u in user_reviews:
            data['user_seq_wt'].append(user_reviews[u])
            items = [item for item, time in user_reviews[u]]
            data['user_seq'].append(items)

        data['user_seq_wt_dic'] = user_reviews
        data['items_feat'] = dat['items_feat']
        data['category2id'] = dat['category2id']
        data['n_items'] = len(dat['item2id']) + 2
        data['n_users'] = len(dat['user2id']) - 1
        data['n_categories'] = len(dat['category2id'])
        data['n_brands'] = len(dat['brand2id'])
        data['feature_size'] = 6 + 1 + data['n_categories'] + data['n_brands'] - 2

        return data
    
    def get_feats_vec(self, feats, args):
        feats = torch.tensor(feats)
        
        # 取出類別資料 (去掉第一列price和最後一列brand)
        categories = feats[:, 1:-1].long()
        # categories = torch.zeros_like(feats[:, 1:-1]).long()
        
        # 取得最具體的類別 (最後一個非零值)
        item_to_category = {}
        for idx in range(1, len(categories)):  # 從1開始，跳過padding item
            # 找出該項目的所有非零類別
            item_cats = categories[idx]
            non_zero_cats = item_cats[item_cats > 0]
            if len(non_zero_cats) > 0:
                # 使用最後一個非零類別（最具體的類別）
                item_to_category[idx] = non_zero_cats[-1].item()
                # item_to_category[idx] = non_zero_cats[-2].item()
            # item_to_category[idx] = 0
        
        # 建立category_items映射
        category_items = defaultdict(list)
        for item_id, category_id in item_to_category.items():
            category_items[category_id].append(item_id)
        category_items = dict(category_items)
        
        # 取出品牌資料
        brands = feats[:, -1].long()
        # brands = torch.zeros_like(feats[:, -1]).long()
        item_to_brand = {
            idx: brand.item() 
            for idx, brand in enumerate(brands) 
            if brand > 0
        }
        
        # 建立brand_items映射
        brand_items = defaultdict(list)
        for item_id, brand_id in item_to_brand.items():
            brand_items[brand_id].append(item_id)
        brand_items = dict(brand_items)
        
        # 建立lookup tables
        max_item_id = max(item_to_category.keys())
        category_lookup = torch.zeros(max_item_id + 1, dtype=torch.long)
        brand_lookup = torch.zeros(max_item_id + 1, dtype=torch.long)
        
        for item_id, category in item_to_category.items():
            category_lookup[item_id] = category
        
        for item_id, brand in item_to_brand.items():
            brand_lookup[item_id] = brand
        
        # 測試輸出
        # for item_id in list(item_to_category.keys())[:5]:
        #     cat_id = item_to_category[item_id]
        #     category_name = list(args['category2id'].keys())[list(args['category2id'].values()).index(cat_id)]
        #     print(f"Item {item_id}: Category {cat_id} ({category_name})")

        return feats, item_to_category, category_items, category_lookup, item_to_brand, brand_items, brand_lookup    
        
    def create_dataset(self):
        if getattr(self.args, 'use_db', False):
            print("========dataset: PostgreSQL (via --use_db)===========")
            data_dic = self.get_data_dic_from_db(self.args.db_url)
            # Store remap mappings so they get saved to model_args.pkl
            self.args.db2train = data_dic.get('db2train', {})
            self.args.train2db = data_dic.get('train2db', [0])
            self.args.snapshot_max_item_id = data_dic.get('snapshot_max_item_id', 0)
        else:
            data_dic = self.get_data_dic(self.data_file)
            self.args.db2train = {}
            self.args.train2db = []
            self.args.snapshot_max_item_id = 0
        self.item_size = data_dic['n_items'] + 1
        self.args.feature_size = data_dic['feature_size']
        self.args.n_categories = data_dic['n_categories']
        self.args.n_brands = data_dic['n_brands']
        (items_feature, item_to_category, category_items, category_lookup,
            item_to_brand, brand_items, brand_lookup) = self.get_feats_vec(data_dic['items_feat'], data_dic)
        self.args.items_feature = items_feature
        self.args.item_to_category = item_to_category
        self.args.category_items = category_items
        self.args.category_lookup = category_lookup
        self.args.item_to_brand = item_to_brand
        self.args.brand_items = brand_items
        self.args.brand_lookup = brand_lookup

        self.args.item_size = self.item_size
        self.args.valid_rating_matrix, self.args.test_rating_matrix = self.get_rating_matrix(data_dic)

        self.train_dataset = SASRecDataset(self.args, data_dic['user_seq'], data_type="train")
        self.valid_dataset = SASRecDataset(self.args, data_dic['user_seq'], data_type="valid")
        self.test_dataset = SASRecDataset(self.args, data_dic['user_seq'], data_type="test")
       
        self.train_dataloader = self.make_train_dataloader()
        self.valid_dataloader = self.make_valid_dataloader()
        self.test_dataloader = self.make_test_dataloader()

        
        
    def make_train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                        sampler=RandomSampler(self.train_dataset),
                                        batch_size=self.args.train_batch_size,
                                        drop_last=False,
                                        num_workers=4, 
                                        ) 
     
        return train_dataloader
    
    
    def make_valid_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset,
                                        sampler=SequentialSampler(self.valid_dataset),
                                        batch_size=self.args.test_batch_size,
                                        drop_last=True) 
        return valid_dataloader
        
       
    def make_test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                        sampler=SequentialSampler(self.test_dataset),
                                        batch_size=self.args.test_batch_size,
                                        drop_last=True) 
        return test_dataloader


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        
    def mask_input_ids(self, input_ids, p):
        mask_indices = []
        for token in input_ids:
            if token == 0:
                mask_indices.append(0)  # 元素为0的位置不参与mask
            else:
                mask_indices.append(1 if random.random() < p else 0)
        return mask_indices

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]
        attention_mask = [1.0 if token > 0 else 0.0 for token in input_ids]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        
        masked_indices0 = self.mask_input_ids(input_ids, self.args.mlm_probability_train) # random mask some items and return the indices. for diffusion train
        # masked_indices1 = self.mask_input_ids(input_ids, self.args.mlm_probability) # random mask some items and return the indices.  for sas aug
        # masked_indices2 = self.mask_input_ids(input_ids, self.args.mlm_probability) # random mask some items and return the indices.  for sas aug
        

        dict1 = {"user_id": torch.tensor(user_id, dtype=torch.long),
                "input_ids":torch.tensor(input_ids, dtype=torch.long),
                "target_pos":torch.tensor(target_pos, dtype=torch.long), 
                "target_neg":torch.tensor(target_neg, dtype=torch.long), 
                "answer": torch.tensor(answer if isinstance(answer, int) else answer[0], dtype=torch.long), 
                "masked_indices0": torch.tensor(masked_indices0, dtype=torch.bool), 
                # "masked_indices1": torch.tensor(masked_indices1, dtype=torch.bool),  
                # "masked_indices2": torch.tensor(masked_indices2, dtype=torch.bool), 
                "attention_mask":torch.tensor(attention_mask, dtype=torch.float), 
             }
        return dict1
        

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = 0  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
    