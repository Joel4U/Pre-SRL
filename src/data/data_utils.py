from typing import List, Dict, Tuple
from src.data import Instance

B_PREF="B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"
root_dep_label = "root"
self_label = "self"

punc_tags = ['``', "''", ':', ',', '.', 'PU']  # 标点符号的标签

from logger import get_logger

logger = get_logger()

def convert_iobes(labels: List[str]) -> List[str]:
	"""
	Use IOBES tagging schema to replace the IOB tagging schema in the instance
	:param insts:
	:return:
	"""
	for pos in range(len(labels)):
		curr_entity = labels[pos]
		if pos == len(labels) - 1:
			if curr_entity.startswith(B_PREF):
				labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				labels[pos] = curr_entity.replace(I_PREF, E_PREF)
		else:
			next_entity = labels[pos + 1]
			if curr_entity.startswith(B_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(I_PREF, E_PREF)
	return labels


def build_pos_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
	"""
	Build the mapping from label to index and index to labels.
	:param insts: list of instances.
	:return:
	"""
	pos2idx = {}
	idx2pos = []
	pos2idx[root_dep_label] = len(pos2idx)
	idx2pos.append(root_dep_label)
	for inst in insts:
		for tag in inst.pos:
			if tag not in pos2idx:
				idx2pos.append(tag)
				pos2idx[tag] = len(pos2idx)

	tag_size = len(pos2idx)
	logger.info("#pos_labels: {}".format(tag_size))
	logger.info("pos_label 2idx: {}".format(pos2idx))
	return idx2pos, pos2idx

def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
	for inst in insts:
		for label in inst.labels:
			if label not in label2idx:
				raise ValueError(f"The label {label} does not exist in label2idx dict. The label might not appear in the training set.")


def build_word_idx(trains:List[Instance], devs:List[Instance], tests:List[Instance]) -> Tuple[Dict, List, Dict, List]:
	"""
	Build the vocab 2 idx for all instances
	:param train_insts:
	:param dev_insts:
	:param test_insts:
	:return:
	"""
	word2idx = dict()
	idx2word = []
	word2idx[PAD] = 0
	idx2word.append(PAD)
	word2idx[UNK] = 1
	idx2word.append(UNK)

	char2idx = {}
	idx2char = []
	char2idx[PAD] = 0
	idx2char.append(PAD)
	char2idx[UNK] = 1
	idx2char.append(UNK)

	# extract char on train, dev, test
	for inst in trains + devs + tests:
		for word in inst.words:
			if word not in word2idx:
				word2idx[word] = len(word2idx)
				idx2word.append(word)
	# extract char only on train (doesn't matter for dev and test)
	for inst in trains:
		for word in inst.words:
			for c in word:
				if c not in char2idx:
					char2idx[c] = len(idx2char)
					idx2char.append(c)
	return word2idx, idx2word, char2idx, idx2char

def build_srllabel_idx(insts):
	roles_to_idx = {}
	idx_to_roles = []
	roles_to_idx['O']=len(roles_to_idx)
	idx_to_roles.append(['O'])
	for inst in insts:
		for role in inst['bio_labels']:
			if role not in roles_to_idx:
				idx_to_roles.append(role)
				roles_to_idx[role] = len(roles_to_idx)
	# roles_to_idx[START_TAG] = len(roles_to_idx)
	# idx_to_roles.append(START_TAG)
	# roles_to_idx[STOP_TAG] = len(roles_to_idx)
	# idx_to_roles.append(STOP_TAG)
	print("srl labels: {}".format(len(roles_to_idx)))
	print("srl label 2idx: {}".format(roles_to_idx))
	return roles_to_idx, idx_to_roles

def build_deplabel_idx(insts: List[Instance]) -> Tuple[Dict[str, int], int]:
	deplabel2idx = {}
	deplabels = []
	deplabel2idx[PAD]=len(deplabel2idx)
	deplabels.append(PAD)
	deplabel2idx[root_dep_label]=len(deplabel2idx)
	deplabels.append(root_dep_label)
	root_dep_label_id = deplabel2idx[root_dep_label]
	for inst in insts:
		for label in inst.deplabels:
			if label not in deplabels:
				deplabels.append(label)
				deplabel2idx[label] = len(deplabel2idx)
	print("dep labels: {}".format(len(deplabel2idx)))
	print("dep label 2idx: {}".format(deplabel2idx))
	return deplabel2idx, root_dep_label_id

def check_all_obj_is_None(objs):
	for obj in objs:
		if obj is not None:
			return False
	return [None] * len(objs)

def group_pa_by_p_(srl):
    grouped_srl = {}
    for p, b, e, l in srl:
        bel = grouped_srl.get(p, None)
        if not bel:
            bel = grouped_srl[p] = set()
        bel.add((b, e, l))
    return dict(sorted(grouped_srl.items(), key=lambda x: x[0]))