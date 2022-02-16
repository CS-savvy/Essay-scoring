import numpy as np
import torch
import torch.nn as nn
from networks.layers.graph_transformer_edge import GraphTransformerLayer
from networks.layers.mlp_readout import MLPReadout
import dgl
from copy import deepcopy


class GraphTransformerNet(nn.Module):
    """
    Graph Transformer Network
    """
    def __init__(self, device=None, word_embedding: np.array = None, **kwargs):
        """

        :param net_params: network configurable parameters.
        """
        super().__init__()
        if kwargs['mode'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.L1Loss()
        num_vocab = kwargs['num_vocab']
        num_edge_type = kwargs['num_edge_type']
        hidden_dim = kwargs['hidden_dim']
        num_heads = kwargs['num_heads']
        out_dim = kwargs['out_dim']
        in_feat_dropout = kwargs['in_feat_dropout']
        dropout = kwargs['dropout']
        n_layers = kwargs['num_layers']
        self.readout = kwargs['readout']
        self.layer_norm = kwargs['layer_norm']
        self.batch_norm = kwargs['batch_norm']
        self.residual = kwargs['residual']
        self.edge_feat = kwargs['edge_feat']
        self.device = device
        self.lap_pos_enc = kwargs['lap_pos_enc']
        self.wl_pos_enc = kwargs['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = kwargs['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(num_vocab, hidden_dim)
        # if word_embedding is not None:
        #     self.embedding_h.from_pretrained(word_embedding)

        # self.post_embedding = nn.Linear(300, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_edge_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        if kwargs['mode'] == 'classification':
            self.MLP_layer = MLPReadout(out_dim, kwargs['n_class'])   # 1 out dim since regression problem
        else:
            self.MLP_layer = MLPReadout(out_dim, 1)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        h = self.embedding_h(h)
        # h = self.post_embedding(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)

    def get_node_embedding(self, h):
        # return self.post_embedding(self.embedding_h(h))
        return self.embedding_h(h)

    def infer(self, h, e, g, h_lap_pos_enc=None, h_wl_pos_enc=None):

        graphs = []
        for ht in h:
            g_new = deepcopy(g)
            g_new.ndata['h_init'] = ht
            g_new.ndata['lap_pos_enc'] = h_lap_pos_enc
            g_new.edata['type'] = e
            graphs.append(g_new)

        g = dgl.batch(graphs)
        h = g.ndata['h_init']
        h_lap_pos_enc = g.ndata['lap_pos_enc']
        e = g.edata['type']

        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc

        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        loss = self.criterion(scores, targets)
        return loss

