import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.convLayer1 = nn.Sequential(nn.Conv2d(3, 32, 5), nn.BatchNorm2d(32), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(32, 64, 5), nn.BatchNorm2d(64), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2))
        
        self.convLayer2 = nn.Sequential(nn.Conv2d(64, 256, 5), nn.BatchNorm2d(256), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(256, 256, 5), nn.BatchNorm2d(256), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2))
        
        self.convLayer3 = nn.Sequential(nn.Conv2d(256, 256, 5), nn.BatchNorm2d(256), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(256, 512, 5), nn.BatchNorm2d(512), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2))

        self.linearLayer = nn.Sequential(nn.Linear(4608, 256),
                                        nn.PReLU(),
                                        nn.Linear(256, 256))
        
    def forward(self, x):
        output = self.convLayer1(x)
        output = self.convLayer2(output)
        output = self.convLayer3(output)
        output = output.view(output.size()[0], -1)
        output = self.linearLayer(output)
        return output

class MultiViewNetwork(nn.Module):
    def __init__(self, embeddingGenerator):
        super(MultiViewNetwork, self).__init__()
        self.embeddingGenerator = embeddingGenerator
        self.classifierLayer = nn.Sequential(nn.Linear(1024, 256),
                                        nn.PReLU(),
                                        nn.Linear(256, 2))

    def forward(self, t2, t3, t4, t8, tM, tB):
        outputT2 = self.embeddingGenerator(t2)
        outputT3 = self.embeddingGenerator(t3)
        outputT4 = self.embeddingGenerator(t4)
        outputT8 = self.embeddingGenerator(t8)
        finalView = torch.cat((outputT2, outputT3, outputT4, outputT8), 1)
        finalView = self.classifierLayer(finalView)
        finalClassifierOutput = F.log_softmax(finalView,dim = 1)
        return finalClassifierOutput