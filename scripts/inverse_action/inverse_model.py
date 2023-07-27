import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from absl import flags

FLAGS = flags.FLAGS
batch_size = 128


flags.DEFINE_integer('bottleneck_size', 3, 'Output dimension of CNN')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_float('lr_decay', 0.9, 'Learning rate decay gamma')
flags.DEFINE_float('lr_decay_every', 1, 'Learning rate decay rate')
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay in optimizer')
flags.DEFINE_integer('gpu', 5, 'Which GPU to use.')
flags.DEFINE_string('model_path', '0', 'path from which to load model')
flags.DEFINE_string('logdir', 'runs_gibson_wd=0', 'Name of tensorboard logdir')

'''
EXPERIMENTS
- learning rate
- weight decay

python gibson_action_learning --lr=0.001 --lr_decay=0.9 --lr_decay_every=1 --weight_decay=0.007 --gpu=1 --logdir=gibson_inverse_1
python gibson_action_learning --lr=0.001 --lr_decay=0.9 --lr_decay_every=1 --weight_decay=0.0 --gpu=1 --logdir=gibson_inverse_2
python gibson_action_learning --lr=0.0001 --lr_decay=0.1 --lr_decay_every=200 --weight_decay=0.007 --gpu=2 --logdir=gibson_inverse_3
python gibson_action_learning --lr=0.0001 --lr_decay=0.1 --lr_decay_every=200 --weight_decay=0.0 --gpu=2 --logdir=gibson_inverse_4

'''


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # resnet
        self.resnet18 = models.resnet18(pretrained=True)
        self.modules = list((self.resnet18).children())[:-2]    #  converts [batch_size, 1000] to [batch_size, 512, 7, 7]
        self.resnet18 = nn.Sequential(*self.modules)

        # freeze resnet model
        (self.resnet18).eval()
        for param in (self.resnet18).parameters():
            param.requires_grad = False

        # CNN
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64*3*3, 128)
        self.fc2 = nn.Linear(128, 3)

        # accuracy_learned_fc
        self.fc_accuracy = nn.Linear(3, 3)


    def forward(self, k, k_plus_one):

        # freeze resnet
        self.resnet18.eval()

        # resnet
        resnet_k = self.resnet18(k)
        resnet_k_plus_one = self.resnet18(k_plus_one)
        resnet_output = torch.cat([resnet_k, resnet_k_plus_one], dim=1)

        # CNN
        x = self.conv1(resnet_output)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        encoding = torch.softmax(x, dim=1)
        
        # accuracy_learned_fc
        y = self.fc_accuracy(x)                                               # [batch_size, 3]                          

        return encoding, y
