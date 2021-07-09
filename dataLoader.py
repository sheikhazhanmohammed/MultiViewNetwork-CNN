import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class CellStageDataset(Dataset):
    def __init__(self, dataFrame, transform=None):
        self.t2Path = dataFrame['t2']
        self.t3Path = dataFrame['t3']
        self.t4Path = dataFrame['t4']
        self.t8Path = dataFrame['t8']
        self.label = dataFrame['label']
        self.transform = transform
    
    def __len__(self):
        return len(self.t2Path)
    
    def __getitem__(self, index):
        imgT2 = Image.open(self.t2Path[index])
        imgT3 = Image.open( self.t3Path[index])
        imgT4 = Image.open(self.t4Path[index])
        imgT8 = Image.open(self.t8Path[index])
        if self.transform is not None:
            imgT2 = self.transform(imgT2)
            imgT3 = self.transform(imgT3)
            imgT4 = self.transform(imgT4)
            imgT8 = self.transform(imgT8)
        label = torch.tensor(self.label[index])
        return imgT2, imgT3, imgT4, imgT8, label

def createDataFrame(imageFilePath, excelFilePath):
    imageT2 = []
    imageT3 = []
    imageT4 = []
    imageT8 = []
    projectId = []
    for folder in os.listdir('./images'):
        projectId.append(folder)
        for image in os.listdir(os.path.join('./images',folder)): 
            if image == 't2.jpg':
                imageT2.append(os.path.join('./image',folder,image))
            if image == 't3.jpg':
                imageT3.append(os.path.join('./image',folder,image))
            if image == 't4.jpg':
                imageT4.append(os.path.join('./image',folder,image))
            if image == 't8.jpg':
                imageT8.append(os.path.join('./image',folder,image))
    excelFile = pd.read_excel(excelFilePath)
    data = {'Projestk_Id':projectId, 't2': imageT2, 't3': imageT3, 't4': imageT4, 't8': imageT8}
    data = pd.DataFrame(data)
    return data
    
def creatingTrainAndValidationLoader(dataFrame, imageFilePath, batchSize, validationSize, randomState):
    trainData, validationData = train_test_split(dataFrame, test_size=validationSize, randomState=randomState)
    trainData = trainData.reset_index(drop=True)
    validationData = validationData.reset_index(drop=True)
    transformTrain = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomCrop(500),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transformValidation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    trainClass = CellStageDataset(trainData, imageFilePath, transformTrain)
    validationClass = CellStageDataset(validationData, imageFilePath, transformValidation)
    trainLoader = DataLoader(trainClass, batch_size = batchSize)
    validationLoader = DataLoader(validationClass, batch_size = batchSize)
    return trainLoader, validationLoader

    