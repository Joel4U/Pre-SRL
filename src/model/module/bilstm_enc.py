
import torch
import torch.nn as nn
from torch.nn.init import orthogonal_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def init_lstm_weights(lstm, initializer=orthogonal_):
    for layer_p in lstm._all_weights:
        for p in layer_p:
            if 'weight' in p:
                initializer(lstm.__getattr__(p))

class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder.
    output the score of all labels.
    """

    def __init__(self,input_dim:int,
                 hidden_dim: int,
                 drop_lstm:float=0.5,
                 num_lstm_layers: int =1):
        super(BiLSTMEncoder, self).__init__()

        print("[Model Info] Input size to LSTM: {}".format(input_dim))
        print("[Model Info] LSTM Hidden Size: {}".format(hidden_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        init_lstm_weights(self.lstm)
        self.drop_lstm = nn.Dropout(drop_lstm)

    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)
        return feature_out[recover_idx]


