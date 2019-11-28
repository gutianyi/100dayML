import time
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import logging
from torchvision import models

total_start_time = time.time()
logging.basicConfig(level=logging.INFO, filename='vgg.log', filemode='w', format='%(asctime)s - %(name)s - %('
                                                                                 'levelname)s - %(message)s')
logger = logging.getLogger(__name__)
model = models.vgg16(pretrained=False)


model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))


def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}'.format(test_loss))
    logging.info('Test Loss: {:.6f}'.format(test_loss))

    print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
    logging.info('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
    ac = round(correct/total, 2)
    return ac


train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    logging.info('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    logging.info('CUDA is available!  Training on GPU ...')

# train on small sample
# PATH_train = "C:\\Users\\Lin\\Desktop\\PolyU\\ai_concept\\project\\cat_dog\\train"
# PATH_test = "C:\\Users\\Lin\\Desktop\\PolyU\\ai_concept\\project\\cat_dog\\test"

# train on 20000 sample
PATH_train = "D:\\Development\\pytorch\\train"
PATH_test = "D:\\Development\\pytorch\\test"

TRAIN = Path(PATH_train)
TEST = Path(PATH_test)
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 25
# learning rate
LR = 0.0001

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# choose the training and test datasets
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
test_data = datasets.ImageFolder(TEST, transform=test_transforms)

print(train_data.class_to_idx)
logging.info(train_data.class_to_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


#  Train model
if train_on_gpu:
    model.cuda()
# number of epochs to train the model

n_epochs = 1

train_loss_min = np.Inf  # track change in validation loss

# train_losses,valid_losses=[],[]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

Loss_list = []
Accuracy_list = []
for epoch in range(1, n_epochs + 1):

    start = time.time()
    # keep track of training and validation loss
    train_loss = 0.0
    print('Running epoch: {}'.format(epoch))
    logging.info('running epoch: {}'.format(epoch))

    # train the model #

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        print('Running batch : {}'.format(batch_idx))
        # logging.info('Running batch : {}'.format(batch_idx))
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # print(output)
        # print(target)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        # Loss_list.append(loss.item())
        # _, predicted = torch.max(output.data, 1)
        # correct = 0
        # total = 0
        # correct += np.sum(np.squeeze(predicted.eq(target.data.view_as(predicted))).cpu().numpy())
        # total += data.size(0)
        # # print('target: {}'.format(target))
        # # print('predicted: {}'.format(predicted.t()))
        # accuracy = round(correct / total, 4)
        # Accuracy_list.append(accuracy)
        # print('train accuracy : {}'.format(accuracy))
        # logging.info('train accuracy : {}'.format(accuracy))
        # train_loss += loss.item() * data.size(0)

    end = time.time()

    logging.info('epoch {} finish with:{} second'.format(epoch, end - start))
    # calculate average losses
    # train_losses.append(train_loss/len(train_loader.dataset))
    # valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
    # train_loss = train_loss / len(train_loader.dataset)
    # print('\tTraining Loss: {:.6f}'.format(train_loss))
    # logging.info('\tTraining Loss: {:.6f}'.format(train_loss))
torch.save(model.state_dict(), 'model_CNN.pth')
logging.info('Saving model ...')

# test_model = models.vgg16(pretrained=False)
# test_model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
#                                        torch.nn.ReLU(),
#                                        torch.nn.Dropout(p=0.5),
#                                        torch.nn.Linear(4096, 4096),
#                                        torch.nn.ReLU(),
#                                        torch.nn.Dropout(p=0.5),
#                                        torch.nn.Linear(4096, 2))
# test_model.load_state_dict(torch.load('model_CNN.pth'))
# final_accuracy = test(test_loader, test_model, criterion, train_on_gpu)
final_accuracy = test(test_loader, model, criterion, train_on_gpu)
logging.info('Final accuracy :{}'.format(final_accuracy))
# x1 = range(0, 80)
# x2 = range(0, 80)
# y1 = Accuracy_list
# y2 = Loss_list
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Test accuracy')
# plt.ylabel('Test accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('Test loss')
# plt.ylabel('Test loss')
# plt.show()
# plt.savefig("accuracy_loss.jpg")
total_end_time = time.time()
logging.info('Toal runtime :{} seconds'.format(total_end_time-total_start_time))








