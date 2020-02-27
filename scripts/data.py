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
        qcp_ids = torch.load('data/%s/qcp/%s' % (self.split, fname))
        attention_mask = torch.load('data/%s/amask/%s' % (self.split, fname))
        token_type_ids = torch.load('data/%s/ttype/%s' % (self.split, fname))
        answer = torch.load('data/%s/answer/%s' % (self.split, fname))
        return qcp_ids, attention_mask, token_type_ids, answer
    
    
def get_dataloader(split, tokenizer, maxlen=512, batch_size=16, num_workers=0, prefix=None):
    def collate_fn(batch):
        qcp_ids, attention_mask, token_type_ids, answers = zip(*batch)
        qcp_ids = pad_sequence(qcp_ids, batch_first=True).transpose(1, 2)[:, :, :maxlen]
        attention_mask = pad_sequence(attention_mask, batch_first=True).transpose(1, 2)[:, :, :maxlen]
        token_type_ids = pad_sequence(token_type_ids, batch_first=True).transpose(1, 2)[:, :, :maxlen]
        answers = torch.stack(answers)
        return qcp_ids, attention_mask, token_type_ids, answers
    
    shuffle = split == 'train'
    dataset = BertDataset(split, tokenizer, prefix)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, shuffle=shuffle, \
                            batch_size=batch_size, num_workers=num_workers)
    return dataloader