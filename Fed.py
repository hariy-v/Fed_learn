import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

from logger import setup_logger
from data import load_datasets
from model import get_model
from pickler import pickle_results
from seeder import seed_everything
from client import Client
from server import Server


def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Configuration')

    parser.add_argument('--method', type=str, default="FedAvg", choices=["FedAvg", "FedProx", "Stochastic", "Stochastic_each_epoch"], help='Federated learning method')
    parser.add_argument('--dataset', type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "STL10","F-MNIST"], help='Dataset to use')
    parser.add_argument('--architecture', type=str, default="Resnet9", choices=["CNN", "LeNet", "Resnet9", "Resnet9_new","Resnet18","Resnet34"], help='Model architecture to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    # parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--classes_per_client', type=int, default=2, help='Number of classes per client for N-IID')
    parser.add_argument('--split', type=str, default="DIR", choices=["IID", "N-IID", "DIR-N-IID","DIR"], help='Data distribution among clients')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for dirichlet distribution')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate')
    parser.add_argument('--mom', type=float, default=0.0, help='momentum')
    parser.add_argument('--mu', type=float, default=0.25, help='mu for FedProx')
    parser.add_argument('--min_fit_clients', type=int, default=10, help='Minimum number of clients for training')
    parser.add_argument('--min_evaluate_clients', type=int, default=5, help='Minimum number of clients for evaluation')
    parser.add_argument('--server_round', type=int, default=100, help='Number of server rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use for computation')

    args = parser.parse_args()

    # Construct the folder name based on arguments
    args.folder = f"{args.dataset}/{args.num_clients}_clients_{args.min_fit_clients}_E_{args.local_epochs}_R_{args.server_round}/{args.split}/class_per_client_{args.classes_per_client}/bs{args.batch_size}/lr_{args.lr}_m{args.mom}_{args.architecture}/{args.method}"

    return args


def main(args):
    log = setup_logger(f"./logs/{args.dataset}_{args.method}_{args.split}_{args.architecture}_{args.lr}.log")
    log.info(f"Logging arguments: {args}")
    log.info(
        f"Training on {args.device} using PyTorch {torch.__version__} Aggregating Algorithm {args.method}"
    )
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_fscores= []
    test_supports = []

    seed_everything(args.seed)
    trainloaders, valloaders, testloader, client_samples, num_classes = load_datasets(args.dataset, args.num_clients, args.batch_size, args.split, args.classes_per_client, args.alpha, args.seed)

    global_model = get_model(args.dataset,args.architecture)
    global_model = global_model.to(args.device)
    server = Server(global_model,args.device)
    for r in range(args.server_round):
        # lr =args.lr


        if args.method == "FedAvg" or args.method == "FedProx":
            lr = args.lr
            for i in range(r):
                lr*=args.decay_rate
        if args.method == "Stochastic":
            lr = args.lr/(r+1)
        else:
            lr = args.lr 




        log.info(f"Starting round {r + 1}/{args.server_round}")
        client_models = []
        selected_train_indices = np.random.choice(range(args.num_clients), args.min_fit_clients, replace=False)
        selected_client_samples = [client_samples[i] for i in selected_train_indices]
        selected_val_indices = np.random.choice(range(args.num_clients), args.min_evaluate_clients, replace=False)
        # selected_trainloaders = [trainloaders[i] for i in selected_train_indices]
        selected_valloaders = [valloaders[i] for i in selected_val_indices]

        for index in selected_train_indices:
            log.info(f"Round {r + 1}/{args.server_round} for Client {index} Learning Rate {lr}")
            client = Client(name = index, method = args.method, model=global_model, 
                            data_loader=trainloaders[index], learning_rate=lr, mom=args.mom,optimizer=optim.SGD, 
                            criterion=nn.CrossEntropyLoss(),mu = args.mu, device = args.device, epochs=args.local_epochs, 
                            current_round = r,
                            )
            client_weights = client.train(server.global_model.state_dict())
            # client_weights = client.train(server.global_model)
            client_models.append(client_weights)

        server.aggregate_weights(client_models,selected_client_samples)

        # Evaluate on validation set of selected clients
        client_accuracies = []
        client_losses = []
        for index in selected_val_indices:
            log.info(f"Evaluating global model on Client {index}")
            test_loss,test_accuracy,_,_,_,_ = server.evaluate(valloaders[index],nn.CrossEntropyLoss())
            client_accuracies.append(test_accuracy)
            client_losses.append(test_loss)
        avg_accuracy = sum(client_accuracies) / len(client_accuracies)
        avg_loss = sum(client_losses) / len(client_losses)
        print(f"Round {r + 1}, Average Client Validation Loss: {avg_loss}")
        log.info(f"Round {r + 1}, Average Client Validation Loss: {avg_loss}")
        print(f"Round {r + 1}, Average Client Validation Accuracy: {avg_accuracy * 100}%")
        log.info(f"Round {r + 1}, Average Client Validation Accuracy: {avg_accuracy * 100}%")
        log.info("Evaluating global model on test set")
        test_loss,test_accuracy,test_precision,test_recall,test_fscore,test_support = server.evaluate(testloader,nn.CrossEntropyLoss())
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_fscores.append(test_fscore)
        test_supports.append(test_support)
        print(f"Round {r + 1}, Test Accuracy: {test_accuracy*100}%")
        log.info(f"Round {r + 1}, Test Accuracy: {test_accuracy*100}%")
    try:
        torch.save(server.global_model.state_dict(), f"{args.folder}/model.pth")
    except Exception as e:
        log.info(f"Exception while saving model {e}")
        pass
    pickle_results(args.folder,test_accuracies,test_losses,test_precisions,test_recalls,test_fscores)

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
