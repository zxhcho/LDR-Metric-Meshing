from copy import deepcopy
from utils.registry import Register

CRITERION_REGISTRY = Register()


class CalcCriterion():
    def __init__(self, config):
        self.config = config
        self.cri_types = self.config.criterion.types
        self.cri_weights = self.config.criterion.weights
        assert len(self.cri_types) == len(self.cri_weights)

        # remove loss with weight 0
        n = len(self.cri_types)
        remove_idx = []
        for i in range(n):
            weight = self.cri_weights[i]
            if weight == 0.0:
                remove_idx.append(i)
        self.cri_types = [self.cri_types[i] for i in range(
            n) if i not in remove_idx]
        self.cri_weights = [self.cri_weights[i] for i in range(
            n) if i not in remove_idx]

    def weight_udpate(self, epoch):
        weights = deepcopy(self.cri_weights)
        if "SmoothCoeff" in self.cri_types:
            idx = self.cri_types.index("SmoothCoeff")
            weights[idx] *= (epoch // 10 + 1)
        # update smooth weights by epoch
        return weights

    def __call__(self, epoch, **data):
        loss_dict = {}
        weights = self.weight_udpate(epoch)
        for cri_type, cri_weight in zip(self.cri_types, weights):
            loss_dict[cri_type] = CRITERION_REGISTRY[cri_type](
                **data) * cri_weight

        return loss_dict
