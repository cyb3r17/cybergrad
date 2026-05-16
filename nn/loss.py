from tensor import Tensor
import cupy as cp

class MSELoss:
    def __init__(self):
        pass
    
    def forward(self, pred, target):
        diff = pred - target
        return (diff*diff).mean()
    
class CrossEntropyLoss:
    def __init__(self):
        pass
    
    def forward(self, pred, target ):
        probs = pred.softmax()
        loss = -(target * probs.log()).sum(axis=1).mean()
        return loss
    
