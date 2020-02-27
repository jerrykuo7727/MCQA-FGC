import sys
import json
import torch
from os.path import join, exists
from transformers import BertTokenizer


if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print('Usage: python3 prepare_bert_data.py <pretrained_model> <split> <dataset_1> <dataset_2> ... <dataset_n>')
        exit(1)

    model_path = sys.argv[1]
    split = sys.argv[2]
    datasets = sys.argv[3:]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    for dataset in datasets:
        data = json.load(open('dataset/%s.json' % dataset))
        n_data = len(data)
        for i, PQCA in enumerate(data):
            p_tokens = tokenizer.tokenize(PQCA['passage']) + [tokenizer.sep_token]
            q_tokens = tokenizer.tokenize(PQCA['question']) + [tokenizer.sep_token]
            answer = torch.LongTensor([PQCA['answer']]).squeeze(0)
            torch.save(answer, 'data/%s/answer/%s|%s.pt' % (split, dataset, i))
            
            # Save QCP id tensor
            all_qcp_ids = []
            for ci, key in enumerate(['c1', 'c2', 'c3', 'c4'], start=1):
                c_tokens = tokenizer.tokenize(PQCA[key]) + [tokenizer.sep_token]
                qcp_tokens = [tokenizer.cls_token] + q_tokens + c_tokens + p_tokens
                qcp_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(qcp_tokens))
                torch.save(qcp_ids, 'data/%s/qcp%d/%s|%s.pt' % (split, ci, dataset, i))

            print('%s: %d/%d (%.2f%%) \r' % (dataset, i+1, n_data, 100*(i+1)/n_data), end='')
    print()
