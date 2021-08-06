import torch
import torch.nn.functional as F
import torchmetrics

class FleissKappa(torchmetrics.Metric):
    def __init__(self, num_classes: int, num_raters: int, mode: str = 'logit'):
        """Calculates Fleiss Kappa. See https://en.wikipedia.org/wiki/Fleiss%27_kappa

        Args:
            num_classes (int): Number of categories / classes. `k` on Wiki.
            num_raters (int): Number of raters. `n` on Wiki.
            mode (str, optional): Whether `preds` will be provided as logits or probabilities. Defaults to 'logit'.
        """
        super().__init__()
        
        assert mode in ['logit', 'p'], f"Mode must be either 'logit' or 'p'. Found {mode=}"
        
        self.num_classes = num_classes
        self.num_raters = num_raters
        self.mode = mode

        if self.mode == 'logit':
            self.pred_transform = lambda x: torch.nn.functional.softmax(x, dim=1)
        else:
            self.pred_transform = lambda x: x

        self.eps = 1e-5
        self.add_state("counts", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = self.pred_transform(preds.detach())
        y_hat = preds.argmax(dim=1)
        one_hot = F.one_hot(y_hat, num_classes=self.num_classes).permute(0,2,1)
        counts = one_hot.sum(dim=-1)
        self.counts.append(counts)

    def compute(self):
        self.counts = torch.cat(self.counts, dim=0)

        # For notation see: https://en.wikipedia.org/wiki/Fleiss%27_kappa        
        N, n, k = self.counts.shape[0], self.num_raters, self.num_classes
        p_i = self.counts.sum(dim=0)/(N*n)
        P_i = ((self.counts**2).sum(dim=1)-n)/(n*(n-1))
        P_bar = P_i.mean()
        Pe_bar = (p_i**2).sum()
        kappa = (P_bar - Pe_bar)/(1-Pe_bar + self.eps)
        
        self.reset()
        return kappa
    
__all__ = [FleissKappa]
