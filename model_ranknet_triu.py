import torch
import torch.nn as nn
from dataloader import torch_batch_triu

class RankNet(nn.Module):
    def __init__(self, layers):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, batch_rankings=None, batch_stds_labels=None, sigma=1.0):
        batch_pred = self.model(batch_rankings)  # batch_pred = [40,1]
        batch_pred_dim = torch.squeeze(batch_pred, 1)  # batch_pred_dim = [40]
        batch_pred_diffs = batch_pred - torch.unsqueeze(batch_pred_dim, 0)  # batch_pred_diffs = [40, 40]
        stds_labels = batch_stds_labels.view(1, -1)
        batch_s_ij, tor_row_inds, tor_col_inds = torch_batch_triu(batch_mats=batch_pred_diffs, k=1,
                                                                  pair_type="All", batch_std_labels=stds_labels)

        # Create a pair from　label
        batch_std = batch_stds_labels  # batch_std = [40]
        batch_std_diffs = torch.unsqueeze(batch_std, 1) - torch.unsqueeze(batch_std, 0)  # batch_std_diffs = [40, 40]
        batch_s_ij_label = batch_std_diffs[tor_row_inds, tor_col_inds]

        # Align to -1 ~ 1
        batch_Sij = torch.clamp(batch_s_ij_label, -1, 1)

        batch_loss_1st = 0.5 * sigma * batch_s_ij * (1.0 - batch_Sij)
        batch_loss_2nd = torch.log(torch.exp(-sigma * batch_s_ij) + 1.0)
        batch_rank_loss = torch.sum(batch_loss_1st + batch_loss_2nd)
        return batch_rank_loss

    def predict(self, x):
        return self.model(x)