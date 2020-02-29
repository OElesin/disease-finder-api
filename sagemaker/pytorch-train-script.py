from __future__ import print_function, division

import time
import os, argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.distributed as dist
import pickle
from torchvision import datasets, models, transforms
import copy


print(torch.backends.cudnn.version())
print(torch.backends.cudnn.enabled == True)
print(torch.cuda.is_available())


def model_fn():
    """
    Load PyTorch model
    :return:
    """
    return


def input_fn(request_body, request_content_type):
    """
    :param request_body:
    :param request_content_type:
    :return:
    """
    return


def predict_fn(input_object, model):
    """
    :param input_object:
    :param model:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return


def output_fn(prediction, content_type):
    """
    :param prediction:
    :param content_type:
    :return:
    """
    return


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.squeeze()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--step-size', type=int, default=7)
    parser.add_argument('--use-cuda', type=bool, default=True)
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--valid', type=str, default=os.environ['SM_CHANNEL_VALID'])

    args, _ = parser.parse_known_args()
    learning_rate = args.learning_rate
    epochs = args.epochs
    gamma = args.gamma
    momentum = args.momentum
    batch_size = args.batch_size
    step_size = args.step_size
    data_dir = '/opt/ml/input/data/'
    num_gpus = torch.cuda.device_count()
    num_workers = 4 * num_gpus

    # if args.distributed:
    #     # Initialize the distributed environment.
    #     world_size = len(args.hosts)
    #     os.environ['WORLD_SIZE'] = str(world_size)
    #     host_rank = args.hosts.index(args.current_host)
    #     dist.init_process_group(backend=args.backend, rank=host_rank)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'valid']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for x in ['train', 'valid']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    print("Prediction Classes")
    print(class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Show Device")
    print(device)
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    print(f"Number of features: {num_ftrs}")
    model_ft.fc = nn.Linear(num_ftrs, len(class_names) - 1)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)
    model_ft_cpu = model_ft.cpu()

    # save model
    print("Saving Pickled Model in GPU and CPU")
    pickle.dump(model_ft_cpu, open(os.path.join(args.model_dir, 'plant-disease-model-cpu.pt'), 'wb'))
    pickle.dump(model_ft, open(os.path.join(args.model_dir, 'plant-disease-model-gpu.pt'), 'wb'))
    print("Done...")
