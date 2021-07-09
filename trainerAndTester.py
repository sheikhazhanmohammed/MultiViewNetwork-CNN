import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def trainOneEpoch(model, trainLoader, criterion, optimizer, currentEpoch, totalSteps, device):
    currentLR = get_lr(optimizer)
    print("Training started for Epoch: ", currentEpoch)
    print("Current Learning Rate is set to:", currentLR)
    print("Training Stats for current epoch")
    runningLoss = 0.0
    correct = 0
    total = 0
    model.train()
    for batchIndex, (imgT2, imgT3, imgT4, imgT8, label) in enumerate(trainLoader):
        imgT2, imgT3, imgT4, imgT8, label = imgT2.to(device), imgT3.to(device), imgT4.to(device), imgT8.to(device), label.to(device)
        optimizer.zero_grad()
        prediction = model(imgT2, imgT3, imgT4, imgT8)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
        _, predictionValue = torch.max(prediction, dim=1)
        correct += torch.sum(predictionValue==label).item()
        total += label.size(0)
        if batchIndex%20==0:
            print('Epoch [{}], Step [{}/{}], Loss: {:.4f}'.format(currentEpoch, batchIndex, totalSteps, loss.item()))
    trainAccuracy = 100 * correct / total
    trainLoss = runningLoss/totalSteps
    print("Training Complete for Epoch:", currentEpoch)
    return model, trainAccuracy, trainLoss, currentLR

def validateEpoch(model, validationLoader, criterion, scheduler, device):
    print("Validation started")
    batchLoss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for imgT2, imgT3, imgT4, imgT8, label in validationLoader:
            imgT2, imgT3, imgT4, imgT8, label = imgT2.to(device), imgT3.to(device), imgT4.to(device), imgT8.to(device), label.to(device)
            prediction = model(imgT2, imgT3, imgT4, imgT8)
            loss = criterion(prediction, label)
            batchLoss += loss.item()
            _, predictionValue = torch.max(prediction, dim=1)
            correct += torch.sum(predictionValue==label).item()
            total += label.size(0)
        validationAccuracy = 100*correct/total
        validationLoss = batchLoss/len(validationLoader)
        scheduler.step(validationLoss)
        return validationAccuracy, validationLoss

def trainModel(model, trainLoader, validationLoader, criterion, optimizer, scheduler, device, totalEpochs):
    trainingAccuracyList = []
    trainingLossList = []
    validationAccuracyList = []
    validationLossList = []
    learningRateList = []
    maxValidationAccuracy = 0
    totalTrainStep = len(trainLoader)
    print("Training started for model")
    for epoch in range(1,totalEpochs+1):
        model, trainAccuracy, trainLoss, currentLR = trainOneEpoch(model, trainLoader, criterion, optimizer, epoch, totalTrainStep, device)
        print("Training Accuracy: ",trainAccuracy)
        print("Training Loss: ",trainLoss)
        trainingAccuracyList.append(trainAccuracy)
        trainingLossList.append(trainLoss)
        learningRateList.append(currentLR)
        validationAccuracy, validationLoss = validateEpoch(model, validationLoader, criterion, scheduler, device)
        print("Validation Accuracy:", validationAccuracy)
        print("Validation Loss: ",validationLoss)
        validationAccuracyList.append(validationAccuracy)
        validationLossList.append(validationLoss)
        if maxValidationAccuracy <= validationAccuracy:
            print("Model improvement detected, saving model")
            torch.save(model.state_dict(), './bestClassificationModel.pt')
    return model, trainingAccuracyList, trainingLossList, validationAccuracyList, validationLossList, learningRateList
