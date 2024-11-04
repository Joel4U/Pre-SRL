from tqdm import tqdm
from typing import List, Dict, Set
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import torch
import numpy as np
from src.data.data_utils import build_srllabel_idx, group_pa_by_p_
from logger import get_logger
from src.data import Instance
import json
from transformers.tokenization_utils_base import BatchEncoding

logger = get_logger()

PAD_INDEX = 0
PUNCT_LABEL = 'punct'  # EN 依存关系标签中的标点符号标签
# PUNCT_LABEL = 'P'  # CN 依存关系标签中的标点符号标签


def convert_instances_to_feature_tensors(instances, tokenizer: PreTrainedTokenizerFast, srllabel2idx: Dict[str, int]) -> List[Dict]:
    features = []
    # logger.info("[Data Info] We are not limiting the max length in tokenizer. You should be aware of that")
    for idx, inst in enumerate(instances):
        words = inst['token']
        orig_to_tok_index = []
        res = tokenizer.encode_plus(words, is_split_into_words=True)
        subword_idx2word_idx = res.word_ids(batch_index=0)
        prev_word_idx = -1
        for i, mapped_word_idx in enumerate(subword_idx2word_idx):
            if mapped_word_idx is None: ## cls and sep token
                continue
            if mapped_word_idx != prev_word_idx:
                orig_to_tok_index.append(i)
                prev_word_idx = mapped_word_idx
        assert len(orig_to_tok_index) == len(words)
        srllabels = inst['srl']
        assert len(srllabels) == len(words)
        # 获取 SRL 标签并转换为 ID
        srl_ids = []
        for srl_vector in srllabels:  # 假设 inst['srl'] 是一个包含多个标签的列表
            srl_ids.append([])  # 为每个 srl_vector 初始化一个列表
            for srl in srl_vector:
                srl_id = srllabel2idx[srl]# if srl in srllabel2idx else srllabel2idx["O"]
                srl_ids[-1].append(srl_id)

        predicate_flags = [0] * len(words)
        for pos in inst['pred_pos']:
            predicate_flags[pos] = 1
        features.append({"input_ids": res["input_ids"],
                         "attention_mask": res["attention_mask"],
                         "orig_to_tok_index": orig_to_tok_index, "predicate_flags": predicate_flags,
                         "srl_ids": srl_ids, "srl_set": inst['srl_set']})
    return features

def batch_iter(dataset, batch_size, shuffle=False):
    insts_ids = dataset.insts_ids
    insts = dataset.insts
    if shuffle:
        combined = list(zip(insts, insts_ids))
        np.random.shuffle(combined)
        insts, insts_ids = zip(*combined)
        # np.random.shuffle(insts_ids)

    nb_batch = int(np.ceil(len(dataset) / batch_size))
    for i in range(nb_batch):
        # batch_data = dataset[i*batch_size: (i+1)*batch_size]
        batch_insts = insts[i*batch_size: (i+1)*batch_size]
        batch_insts_ids = insts_ids[i*batch_size: (i+1)*batch_size]
        yield  batch_insts_ids, batch_insts


def batch_variable(batch:List[Dict], config):
    device = config.device
    batch_size = len(batch)
    word_seq_lens = [len(feature["orig_to_tok_index"]) for feature in batch]
    max_seq_len = max(word_seq_lens)
    max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])

    input_ids = torch.zeros((batch_size, max_wordpiece_length), dtype=torch.long, device=device)
    input_mask = torch.zeros((batch_size, max_wordpiece_length), dtype=torch.long, device=device)
    orig_to_tok_index = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    # pos_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    pred_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    srl_matrix = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.long, device=device)
    # punc_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
    
    for i, feature in enumerate(batch):
        input_ids[i, :len(feature["input_ids"])] = torch.tensor(feature["input_ids"], dtype=torch.long, device=device)
        input_mask[i, :len(feature["attention_mask"])] = torch.tensor(feature["attention_mask"], dtype=torch.long, device=device)
        orig_to_tok_index[i, :len(feature["orig_to_tok_index"])] = torch.tensor(feature["orig_to_tok_index"], dtype=torch.long, device=device)
        # pos_ids[i, :len(feature["tag_ids"])] = torch.tensor(feature["tag_ids"], dtype=torch.long, device=device)
        pred_ids[i, :len(feature["predicate_flags"])] = torch.tensor(feature["predicate_flags"], dtype=torch.long, device=device)
        srl = torch.tensor(feature["srl_ids"], dtype=torch.long, device=device)
        w = srl.size(0)
        srl_matrix[i][:w, :w] = srl
        # punc_mask[i, :len(feature["deplabel_ids"])] = torch.tensor(
        #     [deplabel_id == config.punctid for deplabel_id in feature['deplabel_ids']], dtype=torch.bool, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "orig_to_tok_index": orig_to_tok_index,
        "word_seq_lens": torch.tensor(word_seq_lens, dtype=torch.long, device=device),
        "pred_ids": pred_ids,
        "label_adjs": srl_matrix,
        # "true_srl": [feature["srl_set"] for feature in batch]
    }
    
class SRLDataset(Dataset):

    def __init__(self, file: str, tokenizer: PreTrainedTokenizerFast, is_train: bool, srllabel2idx: Dict[str, int] = None):
        self.insts = self.read_jsonlines(file)
        self.insts_ids = convert_instances_to_feature_tensors(self.insts, tokenizer, srllabel2idx)
        self.tokenizer = tokenizer

    def read_jsonlines_SRLMM(self, data_file):
        insts = []
        with open(data_file, "r") as f:
            for line in f:
                # 解析每一行json数据
                entry = json.loads(line)

                sentences = entry["sentences"][0]  # 句子
                pos_tags = entry["pos"][0]  # 词性标签
                srl_data = entry["srl"][0]  # SRL谓词和语义角色

                text = sentences  # 直接用原文作为text
                pos = pos_tags  # 词性标签直接取

                predicate_positions = []  # 谓词的位置
                predicates = []  # 谓词词汇
                labels = []  # 语义角色标签

                # 初始化每个单词的标签为"O"
                labels_per_predicate = [["O"] * len(sentences) for _ in range(len(srl_data))]

                for srl in srl_data:
                    predicate_idx = srl[0]  # 谓词的索引位置
                    start_idx = srl[1]  # 角色开始位置
                    end_idx = srl[2]  # 角色结束位置
                    role_label = srl[3]  # 语义角色标签

                    # 添加谓词的位置和词汇信息
                    if predicate_idx not in predicate_positions:
                        predicate_positions.append(predicate_idx)
                        predicates.append(sentences[predicate_idx])

                    # 为当前谓词设置对应角色标签，B-开头，I-表示延续
                    labels_per_predicate[predicate_positions.index(predicate_idx)][start_idx] = "B-" + role_label
                    for idx in range(start_idx + 1, end_idx + 1):
                        labels_per_predicate[predicate_positions.index(predicate_idx)][idx] = "I-" + role_label

                    # 转换为 IOBES 格式
                    # labels_per_predicate = [convert_iobes(labels) for labels in labels_per_predicate]

                # 将每个谓词及其对应的信息存入data中
                for p_idx, predicate_position in enumerate(predicate_positions):
                    insts.append(Instance(words=text,ori_words=text, pos=pos, predicate_position=predicate_position, 
                                          labels=labels_per_predicate[p_idx]))
                    #                       {
                    #     "text": text,
                    #     "pos": pos,
                    #     "predicate_position": predicate_position,
                    #     "predicates": predicates[p_idx] + "." + pos[predicate_position],  # 谓词词汇.词性
                    #     "labels": labels_per_predicate[p_idx],  # 该谓词的角色标签
                    # })
        return insts

    def read_jsonlines(self, data_file, doc_level_offset=True):
        num_docs, num_sentences = 0, 0
        insts = []
        with open(data_file, 'r') as f:
            for doc_line in tqdm(f, desc='load file'):
                doc = json.loads(doc_line)
                num_tokens_in_doc = 0
                num_docs += 1
                for sid, (sentence, srl) in enumerate(zip(doc['sentences'], doc['srl'])):
                    # chinese conll2012里面是按照句子序号，造成srl进行了为了递增便宜，所以要去除
                    # srl的标注格式是：位置以0开始 p谓词，b开始位置， e结束位置就（b和e是这个词/词组的在句子中开始和结束的位置）， l标签
                    if doc_level_offset:
                        srl = [(x[0] - num_tokens_in_doc, x[1] - num_tokens_in_doc, x[2] - num_tokens_in_doc, x[3])
                               for x in srl]
                    else:
                        srl = [(x[0], x[1], x[2], x[3]) for x in srl]

                    for x in srl:
                        if any([o < 0 for o in x[:3]]):  # 判断长度是否小于0
                            raise ValueError(f'Negative offset occurred, maybe doc_level_offset=False')
                        if any([o >= len(sentence) for o in x[:3]]):  # 判断长度是否大于句子长度
                            raise ValueError('Offset exceeds sentence length, maybe doc_level_offset=True')
                    # 去重
                    deduplicated_srl = set()
                    pa_set = set()
                    for p, b, e, l in srl:
                        pa = (p, b, e)
                        if pa in pa_set:
                            continue
                        pa_set.add(pa)
                        deduplicated_srl.add((p, b, e, l))
                    inst = self.build_sample(sentence, deduplicated_srl, doc, sid)
                    if len(inst['pred_pos']) != 0:
                        insts.append(inst) # 有srl标注的才进行计算
                    num_sentences += 1
                    num_tokens_in_doc += len(sentence)
        return insts

    def build_sample(self, tokens: List[str], deduplicated_srl: Set, doc: Dict, sid: int):
        """
        返回bio格式的sample，用序列标注的方式来做semantic role labeling
        """
        # 注意这里，把谓词给忽略掉了，市面上对于谓词，一般会单独分出一个二分类任务来做
        # 那能不能做？
        # 我觉得可以，可以在biaffine那一层来学习这个规律
        # 但是在crf那层没啥子希望，为啥子？
        # 因为谓词是单个词。。。
        # 额外插个话题进来，amr貌似是近来的热点，谓词可以不一定为词，也可以为多个词组成的，边界更具有语义一些。
        predicates = [x[2] for x in deduplicated_srl if x[3] == 'V']
        deduplicated_srl = set((x[0], x[1], x[2] + 1, x[3]) for x in deduplicated_srl if x[3] != 'V')
        labels = [['O'] * len(tokens) for _ in range(len(tokens))]
        srl = group_pa_by_p_(deduplicated_srl)
        for p, args in sorted(srl.items()):
            labels_per_p = labels[p]
            for start, end, label in args:
                assert end > start
                assert label != 'V'  # We don't predict predicate（谓词）
                labels_per_p[start] = 'B-' + label
                for j in range(start + 1, end):
                    labels_per_p[j] = 'I-' + label
         # 提取所有B-和I-开头的标签
        srl_bio_labels = [label for seq in labels for label in seq if label.startswith('B-') or label.startswith('I-')]
        sample = {
            'token': tokens,
            'srl': labels,
            'srl_set': deduplicated_srl,
            'pred_pos' : predicates,
            'bio_labels': srl_bio_labels
        }
        if 'pos' in doc:
            sample['pos'] = doc['pos'][sid]
        return sample

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]
