import torch
import torch.nn as nn
from torch.functional import F
from torch.nn.parameter import Parameter
# from opt_einsum import contract
from typing import Tuple, Union
from src.model.embedder import TransformersEmbedder
from src.model.module.bilstm_enc import BiLSTMEncoder
from src.model.module.transformer_enc import TransformerEncoder
from src.model.module.crf import CRF



PAD_INDEX = 0

def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    # drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, time_step)).transpose(1, 2)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x):

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        # x: dep, y: head
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        # s = torch.einsum('bxi,oij,bj->boxy', x, self.weight, y) / self.n_in ** self.scale
        s = torch.einsum('bxi,oji,byj->boxy', x, self.weight, y) / self.n_in ** self.scale
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s
    
class SRLer(nn.Module):

    def __init__(self, config):
        super(SRLer, self).__init__()
        self.word_embedder = TransformersEmbedder(transformer_model_name=config.embedder_type)
        # self.tag_embedding = nn.Embedding(num_embeddings=config.pos_size,
        #                                   embedding_dim=config.pos_embed_dim,
        #                                   padding_idx=0)
        self.enc_dropout = config.enc_dropout
        input_dim = self.word_embedder.get_output_dim() + config.pred_embed_dim

        self.indicator_embeddings = nn.Embedding(2, config.pred_embed_dim)
        self.pred_embed_dim = config.pred_embed_dim
        self.tagset_size = config.label_size

        self.enc_type = config.enc_type
        # if config.enc_type == 'lstm':
        #     self.encoder = BiLSTMEncoder(input_dim=input_dim * 2,
        #                                  hidden_dim=config.enc_dim, drop_lstm=self.enc_dropout, num_lstm_layers=config.enc_nlayers)
        # elif config.enc_type == 'adatrans':
        #     self.encoder = TransformerEncoder(d_model=input_dim * 2, num_layers=config.enc_nlayers, n_head=config.heads,
        #                                       feedforward_dim=2 * input_dim, attn_type='adatrans', dropout=self.enc_dropout, output_dim = config.enc_dim)
        # elif config.enc_type == 'naivetrans':
        #     self.encoder = TransformerEncoder(d_model=input_dim * 2, num_layers=config.enc_nlayers, n_head=config.heads,
        #                                       feedforward_dim=2 * input_dim, attn_type='naivetrans', dropout=self.enc_dropout, output_dim = config.enc_dim)
        self.encoder_ls = BiLSTMEncoder(input_dim=input_dim, hidden_dim=input_dim, drop_lstm=self.enc_dropout, num_lstm_layers=1)
        self.encoder_tr = TransformerEncoder(d_model=input_dim, num_layers=1, n_head=config.heads,
                                              feedforward_dim=2 * input_dim, attn_type='naivetrans', dropout=self.enc_dropout, output_dim = config.enc_dim)
        
        self.mlp_pre_h = MLP(config.enc_dim, config.mlp_dim, dropout=0.1)
        self.mlp_arg_h = MLP(config.enc_dim, config.mlp_dim, dropout=0.1)
        self.srl_biaf = Biaffine(n_in=config.mlp_dim, n_out=config.label_size)
        self.crf = CRF(num_tags=config.label_size)


    def forward(self, subword_input_ids: torch.Tensor, word_seq_lens: torch.Tensor, orig_to_tok_index: torch.Tensor, attention_mask: torch.Tensor,
                    pred_indices, label_adjs: torch.Tensor = None, pred_idx: torch.Tensor =None, 
                    is_train: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        word_emb = self.word_embedder(subword_input_ids, orig_to_tok_index, attention_mask)
        batch_size, sent_len, _ = word_emb.shape
        pred_emb = self.indicator_embeddings(pred_indices.long()) 
        word_rep = torch.cat((word_emb, pred_emb), dim=-1)  # (batch_size, seq_len, hidden_size + pred_embed_dim)

        
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=word_rep.device).view(1, sent_len).expand(batch_size, sent_len)
        non_pad_mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        adj_mask = non_pad_mask.unsqueeze(1) & non_pad_mask.unsqueeze(2)
        # word_rep = torch.cat((word_emb, tags_emb), dim=-1).contiguous()
        if self.training:
            word_rep = timestep_dropout(word_rep, 0.2)
        
        
        enc_out_ls = self.encoder_ls(word_rep, word_seq_lens)
        enc_out = self.encoder_tr(enc_out_ls, non_pad_mask)

        _, _, hidden_dim = enc_out.shape
        pred_mask = (pred_indices > 0).float()  # 谓词位置为 1，非谓词位置为 0, (batch_size, sent_len)
        pred_mask = pred_mask.unsqueeze(-1).expand(-1, -1, hidden_dim)  # (batch_size, sent_len, hidden_dim)
        pred_enc = enc_out * pred_mask  # (batch_size, sent_len, hidden_dim)

        pre_h = self.mlp_pre_h(enc_out) #  [batch_size, hidden_dim]
        arg_h = self.mlp_arg_h(enc_out)  #  [batch_size, seq_len, hidden_dim]
        # [batch_size, seq_len, n_labels]
        s_labels = self.srl_biaf(arg_h, pre_h).permute(0, 2, 3, 1)
        logits = F.log_softmax(s_labels, dim=-1)
        preds = logits.flatten(end_dim=1)
        mask = adj_mask.flatten(end_dim=1)
        first_mask = mask[:, 0]
        preds = preds[first_mask]
        mask = mask[first_mask]
        
        if is_train:
            # ValueError: mask of the first timestep must all be on
            trues = label_adjs.flatten(end_dim=1)
            trues = trues[first_mask]
            return - self.crf(preds, trues, mask, reduction='mean')
        else:
            preds = self.crf.decode(preds, mask)
            return preds
            # pred_labels = self.crf.decode(s_labels, non_pad_mask)
            # return pred_labels


    def get_enhanced_output(self, valid_output, predicates_):
        batch_size, max_len, hidden_size = valid_output.shape
        v_o = torch.zeros((batch_size, max_len, hidden_size * 2), device=valid_output.device)
        for i_ in range(batch_size):
            for j_ in range(max_len):
                v_o[i_][j_] = torch.cat([valid_output[i_][j_], predicates_[i_]], dim=-1)
        return v_o