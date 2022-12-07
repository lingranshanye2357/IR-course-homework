import torch


class SearchMetric:

    def __init__(self):
        """
        metric[(raw, filter), (left, right, average), (mrr, mr, hit1, hit3, hit10)]
        left: predict head, right: predict tail
        """
        self.mrr_ttl = 0.
        self.mr_ttl = 0.
        self.hit1_ttl = 0.
        self.hit3_ttl = 0.
        self.hit10_ttl = 0.
        self.multiple = 0

    def clear(self):
        self.__init__()

    def cal_metric(self, pred_top: torch.Tensor, y_true: torch.Tensor):
        mrr = (1.0 / (pred_top == y_true).nonzero()[:, 1].float().add(1.0)).sum().item()
        mr = (pred_top == y_true).nonzero()[:, 1].float().add(1.0).sum().item()
        hit1 = torch.where(pred_top[:, :1] == y_true, torch.tensor([1]), torch.tensor([0])).sum().item()
        hit3 = torch.where(pred_top[:, :3] == y_true, torch.tensor([1]), torch.tensor([0])).sum().item()
        hit10 = torch.where(pred_top[:, :10] == y_true, torch.tensor([1]), torch.tensor([0])).sum().item()
        return mrr, mr, hit1, hit3, hit10

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.size(0) == y_true.size(0)
        self.multiple += y_pred.size(0)
        y_true = y_true.unsqueeze(1)

        mrr, mr, hit1, hit3, hit10 = self.cal_metric(y_pred, y_true)
        self.mrr_ttl += mrr
        self.mr_ttl += mr
        self.hit1_ttl += hit1
        self.hit3_ttl += hit3
        self.hit10_ttl += hit10

    def get(self):
        mtr = {"mrr": self.mrr_ttl / self.multiple,
               "mr": self.mr_ttl / self.multiple,
               "hit1": self.hit1_ttl / self.multiple,
               'hit3': self.hit3_ttl / self.multiple,
               'hit10': self.hit10_ttl / self.multiple}
        return mtr
