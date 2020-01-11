import torch
import torch.nn as nn
from ndcg import idcg_std

class LambdaRank(nn.Module):
    def __init__(self, layers):
        super(LambdaRank, self).__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, batch_rankings=None, batch_stds_labels=None, device=None, sigma=1.0):
        #if batch_stds_labels.sum().item()==0.0:
        #    print("000000000000")
        batch_pred = self.model(batch_rankings)  # batch_pred = [40,1]
        batch_pred_dim = torch.squeeze(batch_pred, 1)  # batch_pred_dim = [40]
        batch_pred_diffs = batch_pred - torch.unsqueeze(batch_pred_dim, 0)  # batch_pred_diffs = [40, 40]
        #stds_labels = batch_stds_labels.view(1, -1)

        batch_s_ij = batch_pred_diffs

        # Create a pair from　label
        batch_std = batch_stds_labels  # batch_std = [40]
        batch_s_ij_label = torch.unsqueeze(batch_std, 1) - torch.unsqueeze(batch_std, 0)  # batch_std_diffs = [40, 40]
        #batch_s_ij_label = batch_std_diffs
        # print("batch_s_ij_label", batch_s_ij_label.get().size())

        # Align to -1 ~ 1
        batch_Sij = torch.clamp(batch_s_ij_label, -1, 1)

        # print("batch_s_ij", batch_s_ij.get().size()) # ok
        # print("batch_Sij", batch_Sij.get().size()) # ng
        batch_loss_1st = 0.5 * sigma * batch_s_ij * (1.0 - batch_Sij)
        batch_loss_2nd = torch.log(torch.exp(-sigma * batch_s_ij) + 1.0)
        batch_loss_12 = batch_loss_1st + batch_loss_2nd
        #batch_rank_loss = torch.sum(batch_loss_half)

        ''' delta nDCG'''
        #print(batch_stds_labels)
        batch_idcgs = idcg_std(batch_stds_labels, device)  # use original input ideal ranking
        #batch_idcgs = torch.unsqueeze(batch_idcgs, 1)
        #print("batch_idcgs", batch_idcgs)

        # G ### ここ修正
        #label_ar = batch_stds_labels#.detach()
        _, argsort = torch.sort(batch_pred_dim, descending=True, dim=0)
        pred_ar_sorted = batch_stds_labels[argsort]
        #print(pred_ar_sorted)
        batch_gains = torch.pow(2.0, pred_ar_sorted) - 1.0
        if batch_idcgs == 0.0:
            batch_n_gains = batch_gains
        else:
            batch_n_gains = batch_gains / batch_idcgs
        #print("batch_n_gains", batch_n_gains)

        # G_i-G_j
        batch_g_diffs = batch_n_gains.view(batch_n_gains.size(0), 1) - batch_n_gains.view(1, batch_n_gains.size(0))
        #batch_g_diffs = torch.unsqueeze(batch_n_gains, dim=2) - torch.unsqueeze(batch_n_gains, dim=1)
        #print("batch_g_diffs", batch_g_diffs.size())

        # 1/D
        batch_std_ranks = torch.arange(batch_stds_labels.size(0), dtype=torch.float)
        batch_d = 1.0 / torch.log2(batch_std_ranks + 2.0)

        # 1/D_i-1/D_j
        batch_d = torch.unsqueeze(batch_d, dim=0)
        batch_d_diffs = torch.unsqueeze(batch_d, dim=2) - torch.unsqueeze(batch_d, dim=1)

        # |G_i-Gj|*|1/D_i-1/D_j|
        batch_g_diffs = batch_g_diffs.to(device)
        batch_d_diffs = batch_d_diffs.to(device)
        batch_delta_ndcg = torch.abs(batch_g_diffs) * torch.abs(batch_d_diffs)
        #batch_delta_ndcg = torch_batch_triu(batch_ndg_diffs_abs, k=1)

        #print("batch_delta_ndcg", batch_delta_ndcg.size())

        # weighting with delta-nDCG
        #one_eye = (1 - torch.eye(batch_loss_1st.size(0))) # G_ij = 0, D_ij = 0だからone_eyeしない
        loss_mul_dndcg = batch_loss_12 * batch_delta_ndcg
        batch_loss_half = loss_mul_dndcg * 0.5 #one_eye *
        #print(batch_loss_half)
        batch_loss = torch.sum(batch_loss_half)

        return batch_loss


    def predict(self, x):
        return self.model(x)
