# B
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import TransformerEncoder, info_nce
from transformers.models.deberta.modeling_deberta import DebertaEncoder
from torch.nn import  CrossEntropyLoss
import torch as th
from .utils import (
    SiLU,
    linear,
    timestep_embedding
)

class SID4SRec(nn.Module):

    def __init__(self, device, args):
        super(SID4SRec, self).__init__()
        self.item_size = args.item_size
        self.batch_size = args.train_batch_size
        self.hidden_size = args.hidden_size
        self.device = device
        self.args = args
        
        #************params for sasrec****************
        self.max_seq_length = args.max_seq_length
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.inner_size = args.inner_size
        self.hidden_dropout_prob = args.sasrec_dropout_prob
        self.attn_dropout_prob = args.sasrec_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.batch_size = args.train_batch_size

        self.temperature = args.temperature
        self.psi_seq = args.psi_seq
        self.psi_item = args.psi_item
        self.item_to_category = args.item_to_category
        self.item_to_brand = args.item_to_brand
        self.category_items = args.category_items
        self.brand_items = args.brand_items
        self.item_temp = args.item_temp

        self._category_lookup = args.category_lookup.to(device)
        self._brand_lookup = args.brand_lookup.to(device)        
        
        self.item_embedding = torch.nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size * 3)
        # self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size * 2)

        self.nce_fct = nn.CrossEntropyLoss()
        self.rec_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size * 3,
            # hidden_size=self.hidden_size * 2,
            inner_size=self.inner_size * 3,
            # inner_size=self.inner_size * 2,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        
        self.LayerNorm = nn.LayerNorm(self.hidden_size * 3, eps=self.layer_norm_eps)
        # self.LayerNorm = nn.LayerNorm(self.hidden_size * 2, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
                
        self.hidden_t_dim = self.hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            linear(self.hidden_size, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )
        self.bert_encoder = DebertaEncoder(args,)
        self.ln_bert = nn.LayerNorm(args.hidden_size)
        self.bias = nn.Parameter(torch.zeros(self.item_size))
        self.bert_loss_fct = CrossEntropyLoss()  # -100 index = padding token
        self.logits_mode = 1
        self.register_buffer("position_ids", torch.arange(self.max_seq_length).expand((1, -1)))
        self.softmax_loss_fct = nn.CrossEntropyLoss()

        self.category_emb = torch.nn.Embedding(
            num_embeddings=self.args.n_categories,
            embedding_dim=self.args.hidden_size
        ).to(self.device)

        self.brand_emb = torch.nn.Embedding(
            num_embeddings=self.args.n_brands,
            embedding_dim=self.args.hidden_size
        ).to(self.device)

        # parameters initialization
        self.apply(self._init_weights)

    def get_att_emb(self, ids):
        categories = self.get_item_categories(ids)
        cat_emb = self.category_emb(categories)
        brands = self.get_item_brands(ids)  # New method needed in CategoryContrastiveLearning
        brand_emb = self.brand_emb(brands)
        return cat_emb, brand_emb

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_embeds(self, item_ids):
        return self.item_embedding(item_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(hidden_repr, test_items_emb.transpose(0, 1))  # [B n_items+2]
            return scores
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError
        

    def diffusion_reverse(self, x, timesteps, attention_mask):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        emb_x = x
    
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
      
        # emb_inputs = self.position_embedding(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.ln_bert(emb_inputs))

        input_trans_hidden_states = self.bert_encoder(emb_inputs,attention_mask,output_hidden_states=False,output_attentions=False,return_dict=False,)
        h = input_trans_hidden_states[0]
        # print(" debug: ", h.shape)
        h = h.type(x.dtype)
        return h

  
    
    def add_position_embedding(self, sequence, seq_emb=None):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        
        if seq_emb == None:
            item_embeddings = self.item_embedding(sequence)

        else:
            item_embeddings = seq_emb

        category_emb, brand_emb = self.get_att_emb(sequence)
        item_embeddings = torch.cat([item_embeddings, category_emb, brand_emb], dim=-1)
        # item_embeddings = torch.cat([item_embeddings, category_emb], dim=-1)
        position_embeddings = self.position_embedding(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    

    def get_extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.to(input_ids.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self, seq_emb, extended_attention_mask):
        item_encoded_layers = self.rec_trm_encoder(seq_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output
    
    
    def calculate_rec_loss(self, item_seq, target_pos, target_neg):
        extended_attention_mask = self.get_extended_attention_mask(item_seq)
        sequence_emb = self.add_position_embedding(item_seq)
        seq_output = self.forward(sequence_emb, extended_attention_mask)
        loss = self.cross_entropy(seq_output, target_pos, target_neg)

        return loss
    
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        seq_emb = seq_out.view(-1, self.hidden_size * 3)  # [batch*seq_len hidden_size]
        # seq_emb = seq_out.view(-1, self.hidden_size * 2)

        test_items_emb = self.items_emb()
        
        logits = torch.matmul(seq_emb, test_items_emb.transpose(0, 1))
        loss = self.nce_fct(logits, torch.squeeze(pos_ids.view(-1)))

        return loss


    def full_sort_predict(self, item_seq):
        extended_attention_mask = self.get_extended_attention_mask(item_seq)
        sequence_emb = self.add_position_embedding(item_seq)
        seq_output = self.forward(sequence_emb, extended_attention_mask)
        seq_output = seq_output[:,-1,:]

        test_items_emb = self.items_emb()

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) 
        return scores
    
    def items_emb(self):
        all_items = torch.arange(self.args.item_size).to(self.device)
        items_emb = self.item_embedding.weight

        all_categories = self.get_item_categories(all_items.unsqueeze(0))
        all_brands = self.get_item_brands(all_items.unsqueeze(0))
        cat_emb = self.category_emb(all_categories[0])
        brand_emb = self.brand_emb(all_brands[0])
        test_items_emb = torch.cat([items_emb, cat_emb, brand_emb], dim=-1)
        # test_items_emb = torch.cat([items_emb, cat_emb], dim=-1)
        return test_items_emb

    def calculate_cl_loss(self, aug_seq1, aug_seq2, emb1=None, emb2=None):
        extended_attention_mask = self.get_extended_attention_mask(aug_seq1)
        aug_seq1_full_emb = self.add_position_embedding(aug_seq1, emb1)
        aug_seq2_full_emb = self.add_position_embedding(aug_seq2, emb2)

        seq1_output  = self.forward(aug_seq1_full_emb, extended_attention_mask)[:,-1,:]
        seq2_output  = self.forward(aug_seq2_full_emb, extended_attention_mask)[:,-1,:]

        test_items_emb = self.items_emb()

        # nce_logits, nce_labels = info_nce(seq1_output, seq2_output, temp=self.args.temperature, batch_size=aug_seq1.shape[0], sim="dot")
        nce_logits, nce_labels = info_nce(seq1_output, seq2_output, self.psi_seq, temp=self.args.temperature, batch_size=aug_seq1.shape[0], sim="dot")
        nce_loss = self.nce_fct(nce_logits, nce_labels)
        return nce_loss
    
    def get_item_embeddings(self):
        return self.item_embedding.weight
    
    def get_item_categories(self, item_ids):
        # 使用預先生成的 lookup 表
        flattened_ids = item_ids.flatten()
        valid_mask = flattened_ids < len(self._category_lookup)
        
        categories = torch.zeros_like(flattened_ids, device=self.device)
        if valid_mask.any():
            categories[valid_mask] = self._category_lookup[flattened_ids[valid_mask]]
        
        return categories.reshape(item_ids.shape)
    
    def get_item_brands(self, item_ids):
        # Similar logic for brand lookup
        flattened_ids = item_ids.flatten()
        valid_mask = flattened_ids < len(self._brand_lookup)
        
        brands = torch.zeros_like(flattened_ids, device=self.device)
        if valid_mask.any():
            brands[valid_mask] = self._brand_lookup[flattened_ids[valid_mask]]
        
        return brands.reshape(item_ids.shape)

    def get_category_pooling(self, item_embeddings):
        # 預先計算每個類別的項目索引和數量
        if not hasattr(self, '_category_indices'):
            self._category_indices = {}
            self._category_counts = {}
            for cat_id, items in self.category_items.items():
                if items:  # 只處理非空類別
                    self._category_indices[cat_id] = torch.tensor(items, device=self.device)
                    self._category_counts[cat_id] = len(items)
        
        # 直接使用快取的索引進行 pooling
        category_embeddings = {}
        for cat_id, indices in self._category_indices.items():
            indices = indices.to(self.device)  # 確保索引在正確的設備上
            cat_items = torch.index_select(item_embeddings, 0, indices)
            category_embeddings[cat_id] = torch.mean(cat_items, dim=0)
        
        return category_embeddings
    
    def compute_similarity(self, x, y):
        """計算兩個向量的余弦相似度"""
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return torch.sum(x_norm * y_norm, dim=-1)

    def get_contrastive_loss(self, item_embeddings, batch_items=None):
        """使用InfoNCE計算對比學習損失，使用psi閾值過濾負例（完全向量化版本）"""
        category_embeddings = self.get_category_pooling(item_embeddings)
        
        if not category_embeddings:
            return torch.tensor(0.0, device=self.device)
        
        # 將category_embeddings轉換為tensor形式
        categories = list(category_embeddings.keys())
        category_embeds = torch.stack([category_embeddings[c] for c in categories])
        
        # 建立類別ID到index的映射
        category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        category_to_idx_tensor = torch.tensor(
            [category_to_idx.get(i, -1) for i in range(max(category_to_idx.keys()) + 1)],
            device=self.device
        )
        
        # 處理valid items
        if batch_items is not None:
            valid_items = torch.tensor(batch_items, device=self.device)
            valid_mask = torch.zeros(len(item_embeddings), dtype=torch.bool, device=self.device)
            valid_mask[valid_items] = True
        else:
            valid_mask = torch.ones(len(item_embeddings), dtype=torch.bool, device=self.device)
        
        # 獲取有效的embeddings和對應的類別
        valid_item_embeddings = item_embeddings[valid_mask]
        indices = torch.arange(len(item_embeddings), device=self.device)
        category_indices = torch.tensor([self.item_to_category.get(i.item(), -1) for i in indices[valid_mask]],
            device=self.device
        )
        
        if len(valid_item_embeddings) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 正則化embeddings
        item_embeds_normalized = F.normalize(valid_item_embeddings, dim=1)
        category_embeds_normalized = F.normalize(category_embeds, dim=1)
        
        # 計算相似度矩陣 [batch_size, n_categories]
        raw_similarities = torch.matmul(item_embeds_normalized, category_embeds_normalized.t())
        
        # 創建正例mask - 使用高級索引
        batch_size = len(valid_item_embeddings)
        n_categories = len(categories)
        positive_mask = torch.zeros((batch_size, n_categories), dtype=torch.bool, device=self.device)
        valid_category_positions = category_to_idx_tensor[category_indices]
        valid_positions = valid_category_positions != -1
        batch_indices = torch.arange(batch_size, device=self.device)
        positive_mask[batch_indices[valid_positions], valid_category_positions[valid_positions]] = True
        
        # 創建負例mask（基於psi閾值）
        negative_mask = (~positive_mask) & (raw_similarities < self.psi_item)
        
        # 應用溫度係數的相似度
        similarities = raw_similarities / self.item_temp
        
        # 將不符合條件的負例設置為很大的負值
        similarities = torch.where(
            positive_mask | negative_mask,
            similarities,
            torch.tensor(-1e9, device=self.device)
        )
        
        # 創建標籤 - 使用argmax找到每個item的正例位置
        labels = valid_category_positions
        labels = torch.where(labels >= 0, labels, 0)  # 將無效的標籤設為0
        
        # 計算InfoNCE損失
        loss = F.cross_entropy(similarities, labels)
        
        return loss



    # def get_contrastive_loss(self, item_embeddings, batch_items=None):
    #     """使用InfoNCE計算對比學習損失，同一類別內的item互為正樣本"""
        
    #     # 處理valid items
    #     if batch_items is not None:
    #         valid_items = torch.tensor(batch_items, device=self.device)
    #         valid_mask = torch.zeros(len(item_embeddings), dtype=torch.bool, device=self.device)
    #         valid_mask[valid_items] = True
    #     else:
    #         valid_mask = torch.ones(len(item_embeddings), dtype=torch.bool, device=self.device)
        
    #     # 獲取有效的embeddings和對應的類別
    #     valid_item_embeddings = item_embeddings[valid_mask]
    #     indices = torch.arange(len(item_embeddings), device=self.device)
    #     valid_indices = indices[valid_mask]
        
    #     # 獲取每個有效item的category
    #     category_indices = torch.tensor([self.item_to_category.get(i.item(), -1) for i in valid_indices],
    #         device=self.device
    #     )
        
    #     if len(valid_item_embeddings) == 0:
    #         return torch.tensor(0.0, device=self.device)
        
    #     batch_size = len(valid_item_embeddings)
        
    #     # 正則化embeddings
    #     item_embeds_normalized = F.normalize(valid_item_embeddings, dim=1)
        
    #     # 計算item之間的相似度矩陣 [batch_size, batch_size]
    #     raw_similarities = torch.matmul(item_embeds_normalized, item_embeds_normalized.t())
        
    #     # 創建正例mask - 同一類別的item互為正例（但不包括自己）
    #     # 向量化操作：category_indices[i] == category_indices[j] 的矩陣
    #     category_matrix = category_indices.unsqueeze(1) == category_indices.unsqueeze(0)  # [batch_size, batch_size]
    #     valid_category_mask = (category_indices != -1).unsqueeze(1) & (category_indices != -1).unsqueeze(0)
    #     identity_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        
    #     positive_mask = category_matrix & valid_category_mask & (~identity_mask)
        
    #     # 創建負例mask - 不同類別的item
    #     different_category_mask = (~category_matrix) & valid_category_mask
        
    #     # negative_mask = different_category_mask & (raw_similarities < self.psi_item)
    #     # print(f"使用psi_item閾值: {self.psi_item}")

    #     negative_mask = different_category_mask
    #     print("不使用psi_item閾值")
        
    #     # 統計正負樣本數量
    #     positive_counts = positive_mask.sum(dim=1)  # 每個item的正樣本數量
    #     negative_counts = negative_mask.sum(dim=1)  # 每個item的負樣本數量
        
    #     # 計算平均值
    #     avg_positive_samples = positive_counts.float().mean().item()
    #     avg_negative_samples = negative_counts.float().mean().item()
        
    #     # 打印統計信息
    #     print(f"正樣本統計:")
    #     print(f"  平均正樣本數量: {avg_positive_samples:.4f}")
    #     print(f"  正樣本數量範圍: {positive_counts.min().item()} ~ {positive_counts.max().item()}")
    #     print(f"負樣本統計:")
    #     print(f"  平均負樣本數量: {avg_negative_samples:.4f}")
    #     print(f"  負樣本數量範圍: {negative_counts.min().item()} ~ {negative_counts.max().item()}")
    #     print(f"處理的項目總數: {batch_size}")
        
    #     # 類別統計
    #     unique_categories = torch.unique(category_indices[category_indices != -1])
    #     print(f"涉及的類別數: {len(unique_categories)}")
        
    #     exit()
        
    #     # 每個類別的item數量統計
    #     category_counts = {}
    #     for cat in unique_categories:
    #         count = (category_indices == cat).sum().item()
    #         category_counts[cat.item()] = count
        
        
    #     # 應用溫度係數
    #     similarities = raw_similarities / self.item_temp
        
    #     # 將diagonal設為很大的負值（避免自己和自己比較）
    #     similarities.fill_diagonal_(-1e9)
        
    #     # 將不符合條件的位置設置為很大的負值
    #     similarities = torch.where(
    #         positive_mask | negative_mask,
    #         similarities,
    #         torch.tensor(-1e9, device=self.device)
    #     )
        
    #     # 計算損失 - 簡化版本，使用向量化操作
    #     # 只對有正樣本的item計算損失
    #     has_positive = positive_counts > 0
    #     has_negative = negative_counts > 0
    #     valid_items_mask = has_positive & has_negative
        
    #     if valid_items_mask.sum() == 0:
    #         return torch.tensor(0.0, device=self.device)
        
    #     # 對每個有效的item計算損失
    #     valid_similarities = similarities[valid_items_mask]  # [valid_items, batch_size]
    #     valid_positive_mask = positive_mask[valid_items_mask]  # [valid_items, batch_size]
    #     valid_negative_mask = negative_mask[valid_items_mask]  # [valid_items, batch_size]
        
    #     # 簡化的損失計算：對每個item，隨機選擇一個正樣本作為anchor
    #     losses = []
    #     for i in range(valid_similarities.shape[0]):
    #         pos_indices = torch.where(valid_positive_mask[i])[0]
    #         neg_indices = torch.where(valid_negative_mask[i])[0]
            
    #         if len(pos_indices) > 0 and len(neg_indices) > 0:
    #             # 隨機選擇一個正樣本
    #             pos_idx = pos_indices[torch.randint(len(pos_indices), (1,)).item()]
                
    #             # 構建logits: [pos_sim, neg_sim1, neg_sim2, ...]
    #             pos_sim = valid_similarities[i, pos_idx]  # 單個值
    #             neg_sims = valid_similarities[i, neg_indices]  # 1D tensor
                
    #             logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
    #             target = torch.zeros(1, dtype=torch.long, device=self.device)
                
    #             loss = F.cross_entropy(logits.unsqueeze(0), target)
    #             losses.append(loss)
        
    #     if len(losses) > 0:
    #         final_loss = torch.stack(losses).mean()
    #     else:
    #         final_loss = torch.tensor(0.0, device=self.device)
        
    #     return final_loss