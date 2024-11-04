from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	pos: List[str]
	predicate_position: List[int]
	labels: List[str]
	prediction: List[str]  = None
