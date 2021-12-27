import torch
from torch._C import TensorType
from vit_pytorch import ViT
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import resnet50
from vit_pytorch import distill
import time
import copy
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from vit_pytorch.distill import DistillableViT, DistillWrapper
from vit_pytorch.extractor import Extractor
# from tsnecuda import TSNE
from tqdm import tqdm
from sklearn.manifold import TSNE

number_of_classes = 7
img_size = 224

dataset = 'Sydney-captions/'
data_dir = '/home/antonio/PycharmProjects/ViT-with-RS-images/' + dataset
train_dir = data_dir + 'train'
val_dir = data_dir + 'val'
test_dir = data_dir + 'test'

teacher = resnet50(pretrained=True)

v = DistillableViT(
    image_size=img_size,
    patch_size=32,
    num_classes=number_of_classes,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

distiller = DistillWrapper(
    student=v,
    teacher=teacher,
    temperature=3,  # temperature of distillation
    alpha=0.5,  # trade between main loss and distillation loss
    hard=False  # whether to use soft or hard distillation
)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    since = time.time()
    writer = SummaryWriter(data_dir + 'runs/')

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phase_list = ['train']
        if dataset == 'UCM-captions':
            phase_list.append('val')
        for phase in phase_list:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # for batch_idx, labels in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    #   In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    # print(min(labels))
                    # print(max(labels))
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            writer.add_scalar('loss/' + phase, epoch_loss, epoch)
            writer.add_scalar('acc/' + phase, epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

if dataset == 'UCM-captions':
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
else:
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms[x]) for x in ['train']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train']}


# Initialize the model for this run
model_ft, input_size = initialize_model(
    model_name, number_of_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                             num_epochs=num_epochs)

test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

classes = os.listdir(test_dir)

y_pred = []
y_true = []

# iterate over test data to plot confusion matrix, as well as extract test data embeddings
for inputs, labels in test_loader:
    # v = Extractor(model_ft)

    inputs = inputs.to(device)
    labels = labels.to(device)

    output = model_ft(inputs)  # Feed Network
    # logits, embeddings = v(inputs)  # Get embeddings
    # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(embeddings)

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)  # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 7))

sns.heatmap(df_cm, annot=True)

plt.savefig(data_dir + 'confusion_matrix.png')

colors_per_class = {
    '0': [254, 202, 87],
    '1': [255, 107, 107],
    '2': [10, 189, 227],
    '3': [255, 159, 243],
    '4': [16, 172, 132],
    '5': [128, 80, 128],
    '6': [87, 101, 116],
    '7': [52, 31, 151],
    '8': [0, 0, 0],
    '9': [100, 100, 100],
    '10': [0, 100, 255],
    '11': [100, 0, 255],
    '12': [0, 100, 150],
    '13': [100, 150, 0],
    '14': [50, 255, 180],
    '15': [255, 255, 255],
    '16': [10, 80, 255],
    '17': [255, 76, 100],
    '18': [100, 200, 150],
    '19': [78, 80, 90],
    '20': [200, 210, 220]
}

# we'll store the features as NumPy array of size num_images x feature_size
features = None

# we'll also store the image labels and paths to visualize them later
labels = []

for inputs, test_label in tqdm(test_loader, desc='Running the model inference'):
    images = inputs.to(device)
    labels += test_label

    with torch.no_grad():
        output = model_ft.forward(images)

    current_features = output.cpu().numpy()
    if features is not None:
        features = np.concatenate((features, current_features))
    else:
        features = current_features


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne_points():
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()


tsne = TSNE(n_components=2).fit_transform(features)

print(len(labels))
print(labels)

list_labels = [x.item() for x in labels]
print(list_labels)

palette = sns.color_palette("viridis", number_of_classes)

plt.figure()
sns.scatterplot(tsne[:,0], tsne[:,1], hue=list_labels, legend='full', palette=palette)
plt.savefig(data_dir + 't-SNE plot.png')

tx = tsne[:, 0]
ty = tsne[:, 1]

# scale and move the coordinates so they fit [0; 1] range
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# visualize the plot: samples as colored points
visualize_tsne_points()

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
