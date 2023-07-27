import torchvision.transforms as transforms
import torch

# takes an rgb PIL image and turns it into imagenet format
def imageNetTransformPIL(size=224):
    return transforms.Compose([
        # resize smallest edge to size
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def imageNetTransformPIL(size=224):
    return transforms.Compose([
        # resize smallest edge to size
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_device(model):
    return next(model.parameters()).device

def to_imgnet(im):
    x = im.float()
    if len(im.shape) == 3:
        x = x.unsqueeze(0)
    x = x / 255
    x = x.permute(0,3, 1, 2)
    x = x - (torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1,3, 1, 1).float())
    x = x / (torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1,3, 1, 1).float())
    if len(im.shape) == 3:
        return x.squeeze()
    return x


def numpy_to_imgnet(im):
    return to_imgnet(torch.tensor(im))
    # x = torch.tensor(im).float()
    # if len(im.shape) == 3:
        # x = x.unsqueeze(0)
    # x = x / 255
    # x = x.permute(0,3, 1, 2)
    # x = x - (torch.tensor([0.485, 0.456, 0.406]).view(1,3, 1, 1).float())
    # x = x / (torch.tensor([0.229, 0.224, 0.225]).view(1,3, 1, 1).float())
    # if len(im.shape) == 3:
        # return x.squeeze()
    # return x
