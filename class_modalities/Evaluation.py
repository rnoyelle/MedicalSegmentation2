import numpy as np
from .metrics import hausdorff_distance


class MeanHausdorffDistance(object):

    def __init__(self):
        self._sum = None
        self._num_examples = None

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        y_true, y_pred = output['true'], output['pred']
        y_pred = np.round(y_pred)

        for i in range(y_true.shape[0]):
            self._sum += hausdorff_distance(y_true[i], y_pred[i])

        self._num_examples += y_true.shape[0]

    def compute(self):
        if self._num_examples == 0:
            print('CustomAccuracy must have at least one example before it can be computed.')
            raise
        return self._sum / self._num_examples







# from ignite.metrics import Metric
# from ignite.exceptions import NotComputableError
#
# # These decorators helps with distributed settings
# from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
#
#
# class CustomAccuracy(Metric):
#
#     def __init__(self, ignored_class, output_transform=lambda x: x):
#         self.ignored_class = ignored_class
#         self._num_correct = None
#         self._num_examples = None
#         super(CustomAccuracy, self).__init__(output_transform=output_transform)
#
#     @reinit__is_reduced
#     def reset(self):
#         self._num_correct = 0
#         self._num_examples = 0
#         super(CustomAccuracy, self).reset()
#
#     @reinit__is_reduced
#     def update(self, output):
#         y_pred, y = output
#
#         indices = torch.argmax(y_pred, dim=1)
#
#         mask = (y != self.ignored_class)
#         mask &= (indices != self.ignored_class)
#         y = y[mask]
#         indices = indices[mask]
#         correct = torch.eq(indices, y).view(-1)
#
#         self._num_correct += torch.sum(correct).item()
#         self._num_examples += correct.shape[0]
#
#     @sync_all_reduce("_num_examples", "_num_correct")
#     def compute(self):
#         if self._num_examples == 0:
#             raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
#         return self._num_correct / self._num_examples
