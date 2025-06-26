import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

def logit_poisson_nll(logits,
                      label,
                      reduction='mean'):
    pass

def exp_poisson_nll(logits,
                    label,
                    reduction='mean',
                    fill_mask=False):
    # integral term
    int_loss = torch.exp(logits).mean()
    
    # summation term
    inst_loss = torch.tensor(0.).to(int_loss.device)
    for b, l in enumerate(label):

        batch_ids = l.unique()
        batch_ids = batch_ids[batch_ids>100]
        # if b == 0:
        #     print("num difference ", len(batch_ids) - torch.mean(torch.exp(logits[b])))
        if batch_ids.any():
            instance_mask = l.repeat((len(batch_ids), 1, 1))
            binary_masks = instance_mask.eq(batch_ids[:, None, None])
            # if fill_mask:
            #     inst_loss += 
            com_x, com_y = torch.arange(label.size(2)).repeat((len(batch_ids), 1)).to(label.device), torch.arange(label.size(1)).repeat((len(batch_ids), 1)).to(label.device)
            batch_coords = (com_x * binary_masks.float().sum(1) / binary_masks.int().sum((1,2))[:, None]).sum(1), (com_y * binary_masks.float().sum(2) / binary_masks.int().sum((1,2))[:, None]).sum(1)
        
            inst_loss -= torch.sum(logits[b].view(-1)[batch_coords[1].int()*label.size(2) + batch_coords[0].int()]) / len(label)
    return int_loss + inst_loss
        
        
    # # label_ids = label.unique()
    # batch_ids = [l.unique() for l in label]
    # max_len = max(((v > 100).int().sum() for v in batch_ids))
    
    # label_ids = - torch.ones((label.size(0), max_len)).to(label.device)
    # for i, l in enumerate(batch_ids):
    #     insert = l[l.gt(100)]
    #     label_ids[i, :len(insert)] = insert #torch.where(l.gt(100), l, -1)
    
    
    # instance_mask = label.float().repeat((max_len, 1, 1, 1))
    # binary_masks = instance_mask.eq(label_ids.permute(1, 0)[:, :, None, None]).float()

    # com_x, com_y = torch.arange(binary_masks.size(3)).to(label.device), torch.arange(binary_masks.size(2)).to(label.device)
    # coords = torch.stack([
    #     torch.sum(com_x[None, None, :] * binary_masks.sum(dim=2), dim=-1)/binary_masks.sum(dim=(2, 3)),
    #     torch.sum(com_y[None, None, :] * binary_masks.sum(dim=3), dim=-1)/binary_masks.sum(dim=(2, 3))
    # ]).permute(2, 0, 1).int()
    
    # # integral term
    # loss = torch.exp(logits).sum()
    # for i, c in enumerate(coords):
    #     c = torch.nan_to_num(c, -1.)
    #     zero_mask = (torch.logical_and(c[0, :] > 0, c[1, :] > 0))
    #     if torch.any(zero_mask):
    #         # c = torch.masked_select(c, zero_mask[None, :]).view(-1, 2)
    #         c = (c * zero_mask[None, :].float()).view(-1, 2)
    #         select_ids = c[:, 1] * logits.size(2) + c[:, 0]
    #         flat = logits.view(len(coords), -1)[i, :]
    #         batch_contrib = flat.index_select(0, select_ids.int()).sum()
    #         loss -= batch_contrib/len(label)
    
    # return loss
    

@MODELS.register_module()
class PoissonNLLLoss(nn.Module):
    """PoissonNLLLoss
    
    Args:
        use_exponential (bool, optional): Whether the prediction uses exponential activation. Defaults to True.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_nll'.
    """
    
    def __init__(self, 
                 use_sigmoid=True,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_nll') -> None:
        super().__init__()
        self.use_exponential = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        
        # if self.use_exponential:
        self.criterion = exp_poisson_nll
        # else:
        #     self.criterion = logit_poisson_nll
            
    def forward(self,
                logit_score,
                label,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * self.criterion(
            logit_score,
            label,
            reduction = reduction
        )
        
        return loss
            
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
