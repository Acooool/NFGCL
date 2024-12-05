# @Time   : 2024/1/15
# @Author : Yuxi Xiao
# @Email  : 202224131082@stu.cqu.edu.cn

import torch
import torch.nn.functional as F
from recbole_gnn.model.general_recommender import LightGCN

class NFGCL(LightGCN):
    def __init__(self, config, dataset):
        super(NFGCL, self).__init__(config, dataset)
        self.eps = config['eps']
        self.off_rate = config['off_rate']
        self.gamma = config['gamma']
    def forward(self, idx=None):
        all_embs = self.get_ego_embeddings()
        all_embs_cl = all_embs
        embeddings_list = []
        for layer_idx in range(self.n_layers):
            all_embs = self.gcn_conv(all_embs, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embs)
            if layer_idx == 0:
                all_embs_cl = all_embs
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings,[self.n_users, self.n_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embs_cl, [self.n_users, self.n_items])
        if idx is not None:
            user_clone_all = user_all_embeddings.clone()
            item_clone_all = item_all_embeddings.clone()
            user_clone_all[idx[0]] += F.normalize(user_all_embeddings_cl[idx[0]], p=2, dim=1) * self.eps
            item_clone_all[idx[1]] += F.normalize(item_all_embeddings_cl[idx[1]], p=2, dim=1) * self.eps
            return user_clone_all, item_clone_all, user_all_embeddings, item_all_embeddings
        return user_all_embeddings, item_all_embeddings
    def UICC_Loss(self, x1, x2):
        y1 = F.normalize(x1, dim=1, p=2).cuda()
        y2 = F.normalize(x2, dim=1, p=2).cuda()
        c = y1 @ y2.T
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(x1.size(0))
        off_diag = off_diagonal(c).pow_(2).sum().div(x1.size(0))
        loss = on_diag + self.off_rate * off_diag
        return loss
    def UNIF_Loss(self, x, t=2):
        x = F.normalize(x, dim=-1).cuda()
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        idx = [user, pos_item]
        u_idx = torch.unique(user.type(torch.long)).cuda()
        i_idx = torch.unique(pos_item.type(torch.long)).cuda()
        user_view_1, item_view_1, user_view_2, item_view_2 = self.forward(idx)
        uicc_loss = self.UICC_Loss(user_view_1[idx[0]], item_view_1[idx[1]])
        unif_loss = self.gamma * (self.UNIF_Loss(user_view_2[u_idx]) + self.UNIF_Loss(item_view_2[i_idx])) / 2
        return uicc_loss, unif_loss
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

