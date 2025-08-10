import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class SamDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _dice_loss(self, score, target):
        score = score
        target = target

        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        dice = self._dice_loss(inputs, target)
        return dice

class SamDiceLossDevice(nn.Module):
    def __init__(self,device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.celoss = torch.nn.BCEWithLogitsLoss()

    def _dice_loss(self, score, target):
        score = score.to(self.device)
        target = target.to(self.device)

        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target).to(self.device)
        y_sum = torch.sum(target * target).to(self.device)
        z_sum = torch.sum(score * score).to(self.device)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss.to(self.device)

    def forward(self, inputs, target,alpha=0.5, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        dice = self._dice_loss(inputs, target)
        cels = self.celoss(inputs,target)
        loss = alpha*dice+(1-alpha)*cels
        return loss

if __name__ == '__main__':
    inputs = torch.randn((1,1,8,8))
    labels = torch.randint(0,2,(1,8,8))
    # dl = DiceLoss(1)
    dl = SamDiceLoss()
    print(dl(inputs,labels))



