import argparse
from src.config import Config
import time
from src.model import SRLer
import torch
from collections import Counter
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
# from src.config.utils import get_metric
from src.config.eval import decode_output
from src.config.metric import Metric
from tqdm import tqdm
from src.data import SRLDataset, batch_iter, batch_variable
from transformers import set_seed, AutoTokenizer
# from sklearn.metrics import f1_score, recall_score, precision_score
from logger import get_logger
import numpy as np
import json

logger = get_logger()

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:3", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="conll12")
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--pretr_lr', type=float, default=2e-5, help=" bert/roberta, 2e-5 to 5e-5, frozen is 0")
    parser.add_argument('--pretr_frozen', type=bool, default=False)
    parser.add_argument('--other_lr', type=float, default=1e-3, help=" LSTM/Transformer、MLP、Biaffine, 1e-3 to 1e-4, bert frozened set 1e-3 or 5e-4")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=25, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")

    parser.add_argument('--max_no_incre', type=int, default=80, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")

    ##model hyperparameter
    parser.add_argument('--embedder_type', type=str, default="roberta-base", help="you can use 'bert-base-cased' and bert-base-multilan-cased")
    parser.add_argument('--pred_embed_dim', type=int, default=48, help='pos_tag embedding size, heads | pos_embed_dim')
    parser.add_argument('--enc_type', type=str, default='naivetrans', choices=['lstm', 'naivetrans', 'adatrans'], help='type of word encoder used')
    parser.add_argument('--enc_nlayers', type=int, default=3, help='number of encoder layers, 3 for LSTM or 6 for Transformer')
    parser.add_argument('--enc_dropout', type=float, default=0.33, help='dropout used in transformer or lstm')
    parser.add_argument('--enc_dim', type=int, default=400, help="hidden size of the encoder, usually we set to 400 for LSTM, 512 for transformer (d_model)")
    parser.add_argument('--heads', type=int, default=8, help='transformer heads')
    # parser.add_argument('--ff_dim', type=int, default=2048, help='transformer forward feed dim')
    parser.add_argument('--non_linearity', type=str, default="relu", help='nonlinearity used, default relu')
    parser.add_argument('--mlp_dim', type=int, default=300)


    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args


# def train_model(config: Config, epoch: int, train_loader: DataLoader, dev_loader: DataLoader, test_loader: DataLoader):
def train_model(config, epoch, train_data, dev_data, test_data, test_brown_data=None):
    ### Data Processing Info
    train_num = len(train_data)
    logger.info(f"[Data Info] number of training instances: {train_num}")
    logger.info(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}")
    logger.info(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.")
    model = SRLer(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(model=model, pretr_lr=config.pretr_lr, other_lr=config.other_lr,
                                                                   num_training_steps=train_num // config.batch_size * epoch,
                                                                   weight_decay=1e-6, eps=1e-8, warmup_step=int(0.2 * train_num // config.batch_size * epoch))
    logger.info(f"[Optimizer Info] Modify the optimizer info as you need.")
    logger.info(optimizer)

    model.to(config.device)
    best_test_metric = 0
    no_incre_dev = 0
    logger.info(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs")
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        # for iter, batch in enumerate(train_loader, 1):
        for iter, (batch_data, _) in enumerate(batch_iter(train_data, config.batch_size, False)):
            batcher = batch_variable(batch_data, config)
            loss = model(batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"], batcher["attention_mask"], batcher["pred_ids"], batcher["label_adjs"])
            epoch_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
        end_time = time.time()
        logger.info(f"Epoch {i}: {epoch_loss:.5f}, Time is {(end_time - start_time):.2f}s")

        model.eval()
        dev_metrics = evaluate_model_srl(config, model, dev_data, "dev")
        test_metrics = evaluate_model_srl(config, model, test_data, "test")
        if test_brown_data != None:
            test_brown_metrics = evaluate_model_srl(config, model, test_brown_data, "test_brown")
        if test_metrics > best_test_metric:
            no_incre_dev = 0
            # best_dev[0] = dev_metrics[2]
            # best_dev[1] = i
            best_test_metric = test_metrics
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

def evaluate_model_srl(config, model, dataset, name):
    metric = Metric()
    with torch.no_grad():
        for iter, (batch_data, batch_insts) in enumerate(batch_iter(dataset, config.batch_size, False)):
            batcher = batch_variable(batch_data, config)
            preds = model(batcher["input_ids"], batcher["word_seq_lens"], batcher["orig_to_tok_index"], batcher["attention_mask"], batcher["pred_ids"], is_train=False)
            result = decode_output(config.idx2label, preds, batch_insts)
            srl_set_batch = [item['srl_set'] for item in batch_insts]
            metric.step(y_preds=result, y_trues=srl_set_batch)

    logger.info(f"[{name} set Total] Prec.: {metric.precision*100:.2f}, Rec.: {metric.recall*100:.2f}, Micro F1: {metric.f1*100:.2f}")
    return  metric

def get_labels(label_file):
    label_file_path = label_file
    with open(label_file_path, 'r') as f:
        predefined_roles =  json.load(f)
    
    roles_to_idx = {}
    idx_to_roles = []
    for role, idx in predefined_roles.items():
        roles_to_idx[role] = idx
        idx_to_roles.append(role)
    print("srl labels: {}".format(len(roles_to_idx)))
    print("srl label 2idx: {}".format(roles_to_idx))

    return len(roles_to_idx), roles_to_idx, idx_to_roles

def main():

    parser = argparse.ArgumentParser(description="Roberta Deep SRL implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    logger.info(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True, use_fast=True) # for roberta
    # tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, use_fast=True)
    logger.info(f"[Data Info] Reading dataset from: \t{conf.train_file}\t{conf.dev_file}\t{conf.test_file}")
    conf.label_size, conf.label2idx, conf.idx2label = get_labels(conf.label_file) 
    train_dataset = SRLDataset(conf.train_file, tokenizer, srllabel2idx=conf.label2idx, is_train=True)
    dev_dataset = SRLDataset(conf.dev_file, tokenizer, srllabel2idx=conf.label2idx, is_train=False)
    test_dataset = SRLDataset(conf.test_file, tokenizer, srllabel2idx=conf.label2idx, is_train=False)
    if '05' in  conf.test_file:
        test_dataset_brown = SRLDataset(conf.test_file_brown, tokenizer, srllabel2idx=conf.label2idx, is_train=False)
        train_model(conf, conf.num_epochs, train_dataset, dev_dataset, test_dataset, test_dataset_brown)
    train_model(conf, conf.num_epochs, train_dataset, dev_dataset, test_dataset)

if __name__ == "__main__":
    main()
