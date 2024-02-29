import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10,FashionMNIST



from model import get_model
from logger import setup_logger
from pickler import pickle_results
from seeder import seed_everything
def parse_arguments():
    parser = argparse.ArgumentParser(description='Base training Configuration')

    parser.add_argument('--dataset', type=str, default="F-MNIST", choices=["CIFAR10", "CIFAR100", "STL10","F-MNIST"], help='Dataset to use')
    parser.add_argument('--architecture', type=str, default="LeNet", choices=["CNN", "LeNet", "Resnet9"], help='Model architecture to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate')
    parser.add_argument('--mom', type=float, default=0.0, help='momentum')
    parser.add_argument('--local_epochs', type=int, default=500, help='Number of local epochs')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use for computation')

    args = parser.parse_args()

    # Construct the folder name based on arguments
    args.folder = f"{args.dataset}/E_{args.local_epochs}/bs{args.batch_size}/lr_{args.lr}_m{args.mom}/base"

    return args


def create_dataloaders(dataset,batch_size):
    if dataset == "CIFAR10":
        transform = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.CenterCrop((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        )
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
        num_classes = len(set(trainset.targets))
    elif dataset == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.Resize((32,32)),
                transforms.CenterCrop((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), 
                    (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR100("./dataset", train=False, download=True, transform=transform)
        num_classes = len(set(trainset.targets))
    elif dataset == "STL10":
        transform = transforms.Compose(
        [
            transforms.Resize((96,96)),
            transforms.CenterCrop((96,96)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4467, 0.4398, 0.4066), 
                (0.2241, 0.2215, 0.2239)
            ),
        ]
        )
        trainset = STL10("./dataset", split = "train", download=True, transform=transform)
        testset = STL10("./dataset", split = "test", download=True, transform=transform)
        num_classes = len(set(trainset.labels))
    
    elif dataset == "F-MNIST":
        transform = transforms.Compose(
        [

            transforms.ToTensor(),
        ]
        )
        trainset = FashionMNIST("./dataset", train=True, download=True, transform=transform)
        testset = FashionMNIST("./dataset", train=False, download=True, transform=transform)
        num_classes = len(set(trainset.targets))

    else:
        print("THE DATASET IS NOT ADDED YET. EXITING NOW")
        import sys
        sys.exit()
    


    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader,valloader,testloader,num_classes



def train(trainloader, global_model, criterion, optimizer, device):
    global_model.train()
    epoch_loss = 0.0
    for images, labels in trainloader:
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = global_model(images)
        loss = criterion(outputs, labels)
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
    return global_model, epoch_loss/len(trainloader)

def evaluate(test_loader, global_model, criterion, device):
    correct = 0
    total = 0
    loss = 0.0
    y_true = []
    y_pred = []
    global_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            y_true+=labels
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            y_pred += predicted.cpu().detach().numpy().tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    precision,recall,fscore,support = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    accuracy = correct / total
    return loss, accuracy,precision,recall,fscore,support


def main(args):
    log = setup_logger(f"{args.dataset}_base.log")
    log.info(f"Logging arguments: {args}")
    log.info(
        f"Training on {args.device} using PyTorch {torch.__version__}"
    )
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_fscores= []
    test_supports = []

    seed_everything(args.seed)
    trainloaders, valloaders, testloader, num_classes = create_dataloaders(args.dataset,args.batch_size)
    global_model = get_model(args.dataset,args.architecture)
    global_model = global_model.to(args.device)
    optimizer = optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.mom)
    criterion=nn.CrossEntropyLoss()
    for epoch in range(args.local_epochs):
        log.info(f"Epoch {epoch+1}")
        global_model, epoch_loss = train(trainloaders, global_model, criterion, optimizer, args.device)
        log.info(f"Epoch {epoch+1} global model Train loss {epoch_loss}")
        log.info(f"Evaluating global model for Epoch {epoch+1} ")
        loss, accuracy,precision,recall,fscore,support = evaluate(valloaders, global_model, criterion, args.device)

        print(f"Epoch {epoch + 1}, Client Validation Loss: {loss}")
        log.info(f"Epoch {epoch + 1}, Client Validation Loss: {loss}")
        print(f"Epoch {epoch + 1}, Client Validation Accuracy: {accuracy * 100}%")
        log.info(f"Epoch {epoch + 1}, Client Validation Accuracy: {accuracy * 100}%")
        log.info("Evaluating global model on test set")
        test_loss,test_accuracy,test_precision,test_recall,test_fscore,test_support = evaluate(testloader, global_model, criterion, args.device)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_fscores.append(test_fscore)
        test_supports.append(test_support)
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy*100}%")
        log.info(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy*100}%")
    pickle_results(args.folder,test_accuracies,test_losses,test_precisions,test_recalls,test_fscores)



if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
