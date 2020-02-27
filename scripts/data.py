import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class BertDataset(Dataset):
    def __init__(self, split, tokenizer, prefix=None):
        assert split in ('train', 'dev', 'test', 'test_hard')
        self.split = split
        self.fname_list = os.listdir('data/%s/qcp1' % split)
        self.tokenizer = tokenizer
        if prefix:
             self.fname_list = [fname for fname in self.fname_list if fname.startswith(prefix)]
    
    def __len__(self):
        return len(self.fname_list)
        
    def __getitem__(self, i):
        fname = self.fname_list[i]
        qcp1_ids = torch.load('data/%s/qcp1/%s' % (self.split, fname))
        qcp2_ids = torch.load('data/%s/qcp2/%s' % (self.split, fname))
        qcp3_ids = torch.load('data/%s/qcp3/%s' % (self.split, fname))
        qcp4_ids = torch.load('data/%s/qcp4/%s' % (self.split, fname))
        answer = torch.load('data/%s/answer/%s' % (self.split, fname))
        return qcp1_ids, qcp2_ids, qcp3_ids, qcp4_ids, answer
    
    
def get_dataloader(split, tokenizer, maxlen=512, batch_size=16, num_workers=0, prefix=None):
    def collate_fn(batch):
        qcp1_ids, qcp2_ids, qcp3_ids, qcp4_ids, answer = zip(*batch)
        qcp1_ids = pad_sequence(qcp1_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :maxlen]
        qcp2_ids = pad_sequence(qcp2_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :maxlen]
        qcp3_ids = pad_sequence(qcp3_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :maxlen]
        qcp4_ids = pad_sequence(qcp4_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :maxlen]
        answer = torch.stack(answer)
        return qcp1_ids, qcp2_ids, qcp3_ids, qcp4_ids, answer
    
    shuffle = split == 'train'
    dataset = BertDataset(split, tokenizer, prefix)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
                            batch_size=batch_size, num_workers=num_workers)
    return dataloader