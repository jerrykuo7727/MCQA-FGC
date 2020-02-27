import sys
import numpy as np
from time import time
from os.path import join
from copy import deepcopy
from datetime import timedelta

import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertForMultipleChoice

from data import get_dataloader

np.random.seed(42)
torch.manual_seed(42)

global start_time


def validate_dataset(model, split, tokenizer, maxlen=512, prefix=None):
    assert split in ('dev', 'test', 'test_hard')
    dataloader = get_dataloader(split, tokenizer, maxlen, \
                    batch_size=8, num_workers=16, prefix=prefix)
    total_loss, correct, count = 0, 0, 0
    
    model.eval()
    for batch in dataloader:
        batch = [tensor.cuda(device=device) for tensor in batch]
        input_ids, attention_mask, token_type_ids, answers = batch
        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask=attention_mask, \
                                 token_type_ids=token_type_ids, labels=answers)
        
        batch_size = len(answers)
        total_loss += loss * batch_size
        preds = logits.argmax(-1)
        correct += (preds == answers).sum().item()
        count += batch_size
    
    del dataloader
    loss = total_loss / count
    acc = correct / count
    return loss, acc

def validate(model, tokenizer, maxlen=512, prefix=None):
    if prefix:
        print('---- Validation results on %s dataset ----' % prefix)

    # Valid set
    val_loss, val_acc = validate_dataset(model, 'dev', tokenizer, maxlen, prefix)
    print('  val_loss=%.5f, val_acc=%.2f%%' % (val_loss, 100*val_acc))

    # Test set
    test_loss, test_acc = validate_dataset(model, 'test', tokenizer, maxlen, prefix)
    print('  test_loss=%.5f, test_acc=%.2f%%' % (test_loss, 100*test_acc))
    
    # Test-hard set
    test_hard_loss, test_hard_acc = validate_dataset(model, 'test_hard', tokenizer, maxlen, prefix)
    print('  test_hard_loss=%.5f, test_hard_acc=%.2f%%' % (test_hard_loss, 100*test_hard_acc))

    return val_loss


if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print('Usage: python3 train_bert.py cuda:<n> <model_path> <save_path>')
        exit(1)

    # Config
    lr = 3e-5
    maxlen = 376
    batch_size = 2
    accumulate_batch_size = 64
    
    assert accumulate_batch_size % batch_size == 0
    update_stepsize = accumulate_batch_size // batch_size
    
    model_path = sys.argv[2]
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMultipleChoice.from_pretrained(model_path)
    
    device = torch.device(sys.argv[1])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    patience, best_val = 0, 0
    best_state_dict = model.state_dict()
    dataloader = get_dataloader('train', tokenizer, maxlen, batch_size, num_workers=16)
    
    n_step_per_epoch = len(dataloader)
    n_step_per_validation = n_step_per_epoch // 5
    print('%d steps per epoch.' % n_step_per_epoch)
    print('%d steps per validation.' % n_step_per_validation)

    
    print('Start training...')
    start_time = time()
    while True:
        for batch in dataloader:
            batch = [tensor.cuda(device=device) for tensor in batch]
            input_ids, attention_mask, token_type_ids, answers = batch
    
            model.train()
            loss = model(input_ids, attention_mask=attention_mask, \
                         token_type_ids=token_type_ids, labels=answers)[0]
            
            loss.backward()
            step += 1
            elapsed_time = str(timedelta(seconds=int(time()-start_time)))
            print('step %d | loss=%.5f | %s\r' % (step, loss, elapsed_time), end='')
            
            if step % update_stepsize == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if step % n_step_per_validation == 0:
                print("\nstep %d | Validating..." % (step))
                val_metric = validate(model, tokenizer, maxlen)
                if val_metric > best_val:
                    patience = 0
                    best_val = val_metric
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    patience += 1

            if patience >= 10 or step >= 200000:
                print('Finish training. Scoring best results...')
                save_path = join(sys.argv[3], 'finetune.ckpt')
                torch.save(best_state_dict, save_path)
                model.load_state_dict(best_state_dict)
                validate(model, tokenizer, maxlen)
                del model, dataloader
                exit(0)
