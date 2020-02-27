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
            
            # Individually process QCP input
            all_qcp_ids = []
            for ci, key in enumerate(['c1', 'c2', 'c3', 'c4'], start=1):
                c_tokens = tokenizer.tokenize(PQCA[key]) + [tokenizer.sep_token]
                qc_tokens = q_tokens + c_tokens
                qcp_tokens = [tokenizer.cls_token] + qc_tokens + p_tokens
                
                # Convert input to tensors
                qcp_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(qcp_tokens))
                attention_mask = torch.FloatTensor([1 for token in qcp_tokens])
                token_type_ids = torch.LongTensor([0, *(0 for _ in qc_tokens), *(1 for _ in p_tokens)])
                
                # Save tensors
                torch.save(qcp_ids, 'data/%s/qcp_%d/%s|%s.pt' % (split, ci, dataset, i))
                torch.save(attention_mask, 'data/%s/amask_%d/%s|%s.pt' % (split, ci, dataset, i))
                torch.save(token_type_ids, 'data/%s/ttype_%d/%s|%s.pt' % (split, ci, dataset, i))

            print('%s: %d/%d (%.2f%%) \r' % (dataset, i+1, n_data, 100*(i+1)/n_data), end='')
    print()
