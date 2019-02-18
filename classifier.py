
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy
import argparse
import os
from PIL import Image
import numpy as np
import json
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
# In[2]:


def load_model(checkpoint_path, arch , num_classes=102):



    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained="imagenet")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained="imagenet")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained="imagenet")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained="imagenet")
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained="imagenet")
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained="imagenet")
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    chpt = torch.load(checkpoint_path)
    model.class_to_idx = chpt['class_to_idx']
    model.load_state_dict(chpt['state_dict'])


    return model

# In[3]:


def train_model(model, epochs, learning_rate, device , image_path):



    since =time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer= optim.SGD(model.parameters(), lr=0.001, momentum = 0.95, weight_decay = 0.01,  nesterov =True)

    scheduler =  lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_acc = 0.0




    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)




    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)


        for phase in ['train','valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            corrects = 0



            for inputs, labels in dataloaders[phase]:

                model.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)


    return model


# In[4]:


def check_accuracy_on_test(testloader, checkpoint_path, loaded_model):
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model =loaded_model
    model.cuda()


    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images= images.to(device)
            labels = labels.to(device)
            chpt = torch.load(checkpoint_path)
            model.class_to_idx = chpt['class_to_idx']
            model.load_state_dict(chpt['state_dict'])
            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# In[5]:


def process_image(image_test):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image_test)
    pil_image = img_loader(pil_image).float()

    tensor_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor_image = (np.transpose(tensor_image, (1, 2, 0)) - mean)/std
    tensor_image = np.transpose(tensor_image, (2, 0, 1))


    return tensor_image


# In[6]:



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# In[7]:


def predict(image_test,model,checkpoint_path, topk = 5):



    image = process_image(image_test)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    input= image_tensor.unsqueeze(0)



    chpt = torch.load(checkpoint_path)

    model.class_to_idx = chpt['class_to_idx']
    model.load_state_dict(chpt['state_dict'])
    model.cpu()
    model.eval()
    probs = torch.exp(model.forward(input))
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return  top_probs, top_labels, top_flowers



# In[8]:



def sanity_check(image_test):
    img = mpimg.imread(image_test)
    plt.rcdefaults()
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    flower_num = image_test.split('\\')[6]
    title_ = cat_to_name[flower_num]
    img = process_image(image_test)
    imshow(img, ax, title = title_)

    probs, labs, flowers= predict(image_test, loaded_model,checkpoint_path, topk=5)



    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()





if __name__ == '__main__':


    image_path = "C://Users//Sahan//ipthw//flowers"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)



    with open('C:\\Users\\sahan\\ipthw\\cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(45),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ]),
                'valid': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ]),
                'test': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])}



    image_datasets = {x: datasets.ImageFolder(os.path.join(image_path, x),data_transforms[x]) for x in ['train', 'valid','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,shuffle=True,  num_workers=0) for x in ['train', 'valid','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
    class_names = image_datasets['train'].classes



    arch = 'Densenet-161'

    epochs=25
    learning_rate =0.001


    checkpoint_path = 'Densenet-161(test15).pth.tar'
    loaded_model =load_model(checkpoint_path, arch , num_classes=102)

    trained_model =train_model(loaded_model, epochs, learning_rate, device, image_path)

    trained_model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
            'arch': arch,
            'class_to_idx': trained_model.class_to_idx,
            'state_dict': trained_model.state_dict(),
            'hidden_units':1000}

    torch.save(checkpoint, 'Densenet-161(test16).pth.tar')
    checkpoint_path = 'Densenet-161(Test16).pth.tar'
    image_test = 'C:\\Users\\sahan\\ipthw\\flowers\\train\\10\\image_07087.jpg'

    check_accuracy_on_test(dataloaders['test'], checkpoint_path,loaded_model)

    top_probs, top_labels, top_flowers =predict(image_test,loaded_model, checkpoint_path, topk=5)
    sanity_check(image_test)
