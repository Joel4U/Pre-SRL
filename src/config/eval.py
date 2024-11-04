
from typing import List, Dict, Tuple, Any
import torch
from collections import defaultdict, Counter
from src.data import Instance
from src.model.module.eisner import eisner

# PAD_INDEX = 0

class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))

def from_label_id_tensor_to_label_sequence(batch_ids: torch.Tensor,
                                           word_seq_lens: torch.Tensor,
                                           need_to_reverse: bool,
                                           idx2label: List[str]) -> List[List[str]]:
    all_results = []
    for idx in range(len(batch_ids)):
        length = word_seq_lens[idx]
        output = batch_ids[idx][:length].tolist()
        if need_to_reverse:
            output = output[::-1]
        output = [idx2label[l] for l in output]
        all_results.append(output)
    return all_results

def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.Tensor,
                         batch_gold_ids: torch.Tensor,
                         word_seq_lens: torch.Tensor,
                         idx2label: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    batch_p_dict = defaultdict(int)
    batch_total_entity_dict = defaultdict(int)
    batch_total_predict_dict = defaultdict(int)

    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        #convert to span
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
                batch_total_entity_dict[output[i][2:]] += 1
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))
                batch_total_entity_dict[output[i][2:]] += 1
        predict_spans = set()
        start = -1
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
                batch_total_predict_dict[prediction[i][2:]] += 1
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))
                batch_total_predict_dict[prediction[i][2:]] += 1

        correct_spans = predict_spans.intersection(output_spans)
        for span in correct_spans:
            batch_p_dict[span.type] += 1

    return Counter(batch_p_dict), Counter(batch_total_predict_dict), Counter(batch_total_entity_dict)


def calc_train_acc(pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
    '''a
    :param pred_arcs: (bz, seq_len, seq_len)
    :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    :param true_heads: (bz, seq_len)  包含padding
    :param true_rels: (bz, seq_len)
    :param non_pad_mask: (bz, seq_len) 非填充部分mask
    :return:
    '''
    # non_pad_mask[:, 0] = 0  # mask out <root>
    _mask = non_pad_mask.byte()

    bz, seq_len, seq_len, rel_size = pred_rels.size()

    # (bz, seq_len)
    pred_heads = pred_arcs.data.argmax(dim=2)
    masked_pred_heads = pred_heads[_mask]
    masked_true_heads = true_heads[_mask]
    arc_acc = masked_true_heads.eq(masked_pred_heads).sum().item()

    valid_arc_count = non_pad_mask.sum().item()

    out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                         torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                         true_heads].contiguous()
    pred_rels = out_rels.argmax(dim=2)
    masked_pred_rels = pred_rels[_mask]
    masked_true_rels = true_rels[_mask]
    rel_acc = masked_true_rels.eq(masked_pred_rels).sum().item()

    return arc_acc, rel_acc, valid_arc_count

def decode(pred_arc_score, pred_rel_score, mask):
    '''
    :param pred_arc_score: (bz, seq_len, seq_len)
    :param pred_rel_score: (bz, seq_len, seq_len, rel_size)
    :param mask: (bz, seq_len)  pad部分为0
    :return: pred_heads (bz, seq_len)
             pred_rels (bz, seq_len)
    '''
    bz, seq_len, _ = pred_arc_score.size()
    # pred_heads = mst_decode(pred_arc_score, mask)
    # mask[:, 0] = 0  # mask out <root>
    pred_heads = eisner(pred_arc_score, mask)
    pred_rels = pred_rel_score.argmax(dim=-1)
    # pred_rels = pred_rels.gather(dim=-1, index=pred_heads.unsqueeze(-1)).squeeze(-1)
    pred_rels = pred_rels[torch.arange(bz, dtype=torch.long, device=pred_arc_score.device).unsqueeze(1),
                          torch.arange(seq_len, dtype=torch.long, device=pred_arc_score.device).unsqueeze(0),
                          pred_heads].contiguous()
    return pred_heads, pred_rels

def calc_evalu_acc(pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask, punc_mask=None):
    '''
    :param pred_arcs: (bz, seq_len, seq_len)
    :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    :param true_heads: (bz, seq_len)  包含padding
    :param true_rels: (bz, seq_len)
    :param non_pad_mask: (bz, seq_len)
    :param punc_mask: (bz, seq_len)  含标点符号为1
    if punc_mask is not None, we will omit punctuation
    :return:
    '''

    non_pad_mask = non_pad_mask.byte()
    non_pad_mask[:, 0] = 0  # mask out <root>
    # 解码过程
    pred_heads, pred_rels = decode(pred_arcs, pred_rels, non_pad_mask)

    # 统计过程
    non_punc_mask = (~punc_mask if punc_mask is not None else 1)
    pred_heads_correct = (pred_heads == true_heads) * non_pad_mask * non_punc_mask
    pred_rels_correct = (pred_rels == true_rels) * pred_heads_correct
    arc_acc = pred_heads_correct.sum().item()
    rel_acc = pred_rels_correct.sum().item()
    valid_arc_count = (non_pad_mask * non_punc_mask).sum().item()

    return arc_acc, rel_acc, valid_arc_count

def prediction_to_result(prediction: List, batch_tokens, delimiter='') -> List:
    for matrix, tokens in zip(prediction, batch_tokens):
        # result = []
        srls = []
        for i, arguments in enumerate(matrix):
            if arguments:
                # pas = [(delimiter.join(tokens[x[1]:x[2]]),) + x for x in arguments]
                # pas.insert(bisect([a[1] for a in arguments], i), (tokens[i], 'PRED', i, i + 1))
                # result.append(pas)
                srls.extend([(i, start, end, label) for (label, start, end) in arguments])

        yield srls

def decode_output(idx2label, pred, batch_insts):
    offset = 0
    results = []
    for inst in batch_insts:
        sent = inst['token']
        results.append([])
        for token in sent:
            token_pred = get_entities([idx2label[i] for i in pred[offset]])
            results[-1].append(token_pred)
            offset += 1
    return results

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
      prev_tag: previous chunk tag.
      tag: current chunk tag.
      prev_type: previous type.
      type_: current type.

    Returns:
      chunk_end: boolean.

    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
      prev_tag: previous chunk tag.
      tag: current chunk tag.
      prev_type: previous type.
      type_: current type.

    Returns:
      chunk_start: boolean.

    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
      seq(list): sequence of labels.
      suffix:  (Default value = False)

    Returns:
      list: list of (chunk_type, chunk_start, chunk_end).
      Example:

    >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-2]
        else:
            tag = chunk[0]
            type_ = chunk[2:]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks