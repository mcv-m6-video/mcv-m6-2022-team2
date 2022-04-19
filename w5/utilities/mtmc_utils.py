import os
from os.path import join
import random
import torch
import wandb
import numpy as np
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch.nn.functional as F

from utilities.dataset_utils import load_annot

# --- DATASET ---
class TripletDataset(Dataset):
    def __init__(self, data_path, sequences, transform=None):
        self.data_root = data_path
        self.sequences = sequences
        self.data = {}
        self.counter = 0

        # Iterate through the sequences
        for seq in sequences:

            # Iterate through the cameras
            for cam in sorted(os.listdir(os.path.join(data_path, seq))):
                cam_gt = load_annot(join(data_path, seq, cam, 'gt'), 'gt.txt')

                for frame_num, frame_annotations in cam_gt.items():

                    frame_path = join(data_path, seq, cam, 'frames', frame_num + '.jpg')
                    for annot in frame_annotations:

                        if self.data.get(annot['obj_id']) is None:
                            self.data[annot['obj_id']] = [{'path': frame_path, 'bbox': annot['bbox']}]
                            self.counter += 1
                        else:
                            self.data[annot['obj_id']].append({'path': frame_path, 'bbox': annot['bbox']})
                            self.counter += 1

        self.transform = transform

    def __getitem__(self, index):
        counter = 0
        max_id = max(list(self.data.keys()))

        for id in range(1, max_id+1):
            if self.data.get(id) is not None:
                if index >= (len(self.data[id]) + counter):
                    counter += len(self.data[id])
                else:
                    anchor_id = id
                    anchor_img = self.data[id][index - counter]
                    break

        # Get the positive image
        positive_img = random.choice(self.data[anchor_id])

        # Get negative image
        filtered_list = [x for x in list(self.data.keys()) if x != anchor_id]
        negative_id = random.choice(filtered_list)
        negative_img = random.choice(self.data[negative_id])


        anchor_img = Image.open(anchor_img['path']).crop(anchor_img['bbox']).resize((224, 224))
        positive_img = Image.open(positive_img['path']).crop(positive_img['bbox']).resize((224, 224))
        negative_img = Image.open(negative_img['path']).crop(negative_img['bbox']).resize((224, 224))

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return (anchor_img, positive_img, negative_img), []

    def __len__(self):
        return self.counter

# --- MODELS ---

class EmbeddingNet(nn.Module):
    def __init__(self, model_id, backbone='resnet50'):
        super(EmbeddingNet, self).__init__()

        basemodel = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=True)

        self.model_id = model_id
        self.base_resnet = torch.nn.Sequential(*(list(basemodel.children())[:-1]))

    def forward(self, x):
        return self.base_resnet(x).squeeze()

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net, num_classes=2):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

# --- TRAINING ---
def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, output_path, model_id,
        metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        PATH = join(output_path, model_id + '.pth')
        torch.save(model.state_dict(), PATH)
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    """
    Train for one epoch on the training set.
    :param train_loader:    Train data loader
    :param model:
    :param loss_fn:
    :param optimizer:
    :param cuda:
    :param log_interval:
    :param metrics:
    :return:
    """
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

# --- LOSS FUNCTION ---
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()