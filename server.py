import copy
import torch
from typing import List
from sklearn.metrics import precision_recall_fscore_support
import logging
log = logging.getLogger(__name__)
class Server:
    def __init__(self, model, device):
        self.global_model = model
        self.device = device

    # def aggregate_weights(self, client_weights: List[dict]):
    #     log.info("Aggregating weights from clients")
    #     global_state_dict = copy.deepcopy(client_weights[0])
    #     for key in global_state_dict.keys():
    #         for i in range(1, len(client_weights)):
    #             global_state_dict[key] += client_weights[i][key]
    #         global_state_dict[key] = torch.div(global_state_dict[key], len(client_weights))
    #     self.global_model.load_state_dict(global_state_dict)

    def aggregate_weights(self, client_updates, client_weights):
        """Aggregate weights from clients using weighted averaging and update the global model.
        
        :param client_updates: A list of client model state dictionaries.
        :param client_weights: A list of weights indicating the importance or contribution size of each client's update.
        """
        with torch.no_grad():
            global_state_dict = self.global_model.state_dict()
            
            # Initialize a new state dict for the aggregated model
            aggregated_state_dict = {}
            
            # Calculate the weighted average of each parameter
            for key in global_state_dict.keys():
                # Collect all clients' updates for this parameter
                stacked_weights = torch.stack([client_updates[i][key].float() for i in range(len(client_updates))])
                
                # Calculate the weighted average
                weighted_sum = torch.zeros_like(stacked_weights[0])
                for i, weights in enumerate(stacked_weights):
                    weighted_sum += weights * client_weights[i]
                
                aggregated_state_dict[key] = weighted_sum / sum(client_weights)
            
            # Update the global model with the aggregated weights
            self.global_model.load_state_dict(aggregated_state_dict)

        
    # def aggregate_weights(self, client_weights: List[dict], client_samples: List[int]):
    #     """
    #     Aggregate weights from clients with weighted updates based on the number of samples.

    #     Parameters:
    #     - client_weights: List[dict] - The list of client model state dicts.
    #     - client_samples: List[int] - The list of the number of samples for each client.
    #     """
    #     log.info("Aggregating weights from clients with weighted updates")

    #     # Calculate the total number of samples across all clients
    #     total_samples = sum(client_samples)

    #     # Initialize the global state dict with zero weights
    #     global_state_dict = {key: torch.zeros_like(client_weights[0][key], dtype = torch.float) for key in client_weights[0]}

    #     # Perform weighted aggregation of weights
    #     for i, client_weight in enumerate(client_weights):
    #         weight = client_samples[i] / total_samples
    #         for key in client_weight:
    #             global_state_dict[key] += client_weight[key] * weight

    #     # Load the aggregated weights into the global model
    #     self.global_model.load_state_dict(global_state_dict)
    def global_model_state(self):
        return self.global_model.state_dict()

    def evaluate(self, test_loader, criterion):
        self.global_model.eval()
        correct = 0
        total = 0
        loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                y_true+=labels
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss += criterion(outputs, labels).item()
                y_pred += predicted.cpu().detach().numpy().tolist()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(test_loader.dataset)
        precision,recall,fscore,support = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        accuracy = correct / total
        return loss, accuracy,precision,recall,fscore,support
