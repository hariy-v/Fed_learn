import copy
import torch
import logging
log = logging.getLogger(__name__)
class Client:
    def __init__(self, name, method ,model, data_loader, learning_rate, mom, optimizer, criterion, mu, device ,epochs,current_round):
        self.client_name = name
        self.method = method
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.lr = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, momentum=mom)
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.mu = mu
        self.r = current_round

    def reset_optimizer(self):
        self.optimizer.zero_grad()  # Clear existing gradients
        self.optimizer.state_dict()['state'] = {}  # Clear state

    def train(self, global_model_weights):
        self.model.load_state_dict(global_model_weights)
        self.model.train()
        self.reset_optimizer()

        lambda1 = lambda e: (1/((self.r)*self.epochs+e+1))
        scheduler =  torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        for epoch in range(self.epochs):
            log.info(f"Client {self.client_name} Epoch {epoch+1}")
            for inputs, labels in self.data_loader:
                inputs,labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # FedProx term
                if self.method == "FedProx":
                    fed_prox_reg = 0.0
                    for name,w in self.model.named_parameters():
                        w_t = global_model_weights[name]
                        fed_prox_reg += (self.mu / 2) * torch.norm((w - w_t))**2
                    loss += fed_prox_reg
                loss.backward()
                self.optimizer.step()
            # log.info(f"Client {self.client_name} Epoch {epoch+1} Old LR {self.optimizer.param_groups[0]['lr']}")
            if self.method == "Stochastic_each_epoch":
                if epoch!=self.epochs-1:
                    scheduler.step()
                log.info(f"Client {self.client_name} Epoch {epoch+1} New LR {self.optimizer.param_groups[0]['lr']}")

        return self.model.state_dict()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)




