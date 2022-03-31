import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
import sys

sys.path.append('..')
from utils import reset

drug_num = 87
cline_num = 55


class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = HypergraphConv(256, 256)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = HypergraphConv(256, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x = self.batch1(self.act(self.conv1(x, edge)))
        x = self.batch2(self.act(self.conv2(x, edge)))
        x = self.act(self.conv3(x, edge))
        return x


class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        # -------drug_layer
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)
        # -------cell line_layer
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.reset_para()
        self.act = nn.ReLU()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        # -----drug_train
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.act(x_drug)
        x_drug = self.batch_conv2(x_drug)
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        # ----cellline_train
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)

        self.reset_parameters()
        self.drop_out = nn.Dropout(0.4)
        self.act = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        h1 = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h1))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return torch.sigmoid(h.squeeze(dim=1))


class HypergraphSynergy(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder):
        super(HypergraphSynergy, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.drug_rec_weight = nn.Parameter(torch.rand(256, 256))
        self.cline_rec_weight = nn.Parameter(torch.rand(256, 256))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.decoder)

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id):
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        drug_emb, cline_emb = graph_embed[:drug_num], graph_embed[drug_num:]
        rec_drug = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        rec_cline = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        res, tsne = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        return res, rec_drug, rec_cline, tsne
