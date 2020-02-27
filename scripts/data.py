import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class BertDataset(Dataset):
    def __init__(self, split, tokenizer, prefix=None):
        assert split in ('train', 'dev', 'test', 'test_hard')
        self.split = split
        self.fname_list = os.listdir('data/%s/answer' % split)
        self.tokenizer = tokenizer
        if prefix:
             self.fname_list = [fname for fname in self.fname_list if fname.startswith(prefix)]
    
    def __len__(self):
        return len(self.fname_list)
        
    def __getitem__(self, i):
        fname = self.fname_list[i]
        answer = torch.load('data/%s/answer/%s' % (self.split, fname))
        qcp_ids_1 = torch.load('data/%s/qcp_1/%s' % (self.split, fname))
        qcp_ids_2 = torch.load('data/%s/qcp_2/%s' % (self.split, fname))
        qcp_ids_3 = torch.load('data/%s/qcp_3/%s' % (self.split, fname))
        qcp_ids_4 = torch.load('data/%s/qcp_4/%s' % (self.split, fname))
        attention_mask_1 = torch.load('data/%s/amask_1/%s' % (self.split, fname))
        attention_mask_2 = torch.load('data/%s/amask_2/%s' % (self.split, fname))
        attention_mask_3 = torch.load('data/%s/amask_3/%s' % (self.split, fname))
        attention_mask_4 = torch.load('data/%s/amask_4/%s' % (self.split, fname))
        token_type_ids_1 = torch.load('data/%s/ttype_1/%s' % (self.split, fname))
        token_type_ids_2 = torch.load('data/%s/ttype_1/%s' % (self.split, fname))
        token_type_ids_3 = torch.load('data/%s/ttype_1/%s' % (self.split, fname))
        token_type_ids_4 = torch.load('data/%s/ttype_1/%s' % (self.split, fname))
        return answer, qcp_ids_1, qcp_ids_2, qcp_ids_3, qcp_ids_4, \
               attention_mask_1, attention_mask_2, attention_mask_3, attention_mask_4, \
               token_type_ids_1, token_type_ids_2, token_type_ids_3, token_type_ids_4
    
    
def get_dataloader(split, tokenizer, maxlen=512, batch_size=16, num_workers=0, prefix=None):
    def collate_fn(batch):
        answer, qcp_ids_1, qcp_ids_2, qcp_ids_3, qcp_ids_4, \
            attention_mask_1, attention_mask_2, attention_mask_3, attention_mask_4, \
            token_type_ids_1, token_type_ids_2, token_type_ids_3, token_type_ids_4 = zip(*batch)
        answer = torch.stack(answer)
        qcp_ids_1 = pad_sequence(qcp_ids_1, batch_first=True)[:, :maxlen]
        qcp_ids_2 = pad_sequence(qcp_ids_2, batch_first=True)[:, :maxlen]
        qcp_ids_3 = pad_sequence(qcp_ids_3, batch_first=True)[:, :maxlen]
        qcp_ids_4 = pad_sequence(qcp_ids_4, batch_first=True)[:, :maxlen]
        attention_mask_1 = pad_sequence(attention_mask_1, batch_first=True)[:, :maxlen]
        attention_mask_2 = pad_sequence(attention_mask_2, batch_first=True)[:, :maxlen]
        attention_mask_3 = pad_sequence(attention_mask_3, batch_first=True)[:, :maxlen]
        attention_mask_4 = pad_sequence(attention_mask_4, batch_first=True)[:, :maxlen]
        token_type_ids_1 = pad_sequence(token_type_ids_1, batch_first=True)[:, :maxlen]
        token_type_ids_2 = pad_sequence(token_type_ids_2, batch_first=True)[:, :maxlen]
        token_type_ids_3 = pad_sequence(token_type_ids_3, batch_first=True)[:, :maxlen]
        token_type_ids_4 = pad_sequence(token_type_ids_4, batch_first=True)[:, :maxlen]
        return answer, qcp_ids_1, qcp_ids_2, qcp_ids_3, qcp_ids_4, \
               attention_mask_1, attention_mask_2, attention_mask_3, attention_mask_4, \
               token_type_ids_1, token_type_ids_2, token_type_ids_3, token_type_ids_4
    
    shuffle = split == 'train'
    dataset = BertDataset(split, tokenizer, prefix)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
                            batch_size=batch_size, num_workers=num_workers)
    return dataloader