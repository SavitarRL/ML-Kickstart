import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random


class NeuralNetTrain:
    def __init__(self, training_data, nn_model, layer_size_params, criterion_instance, lr, momentum, epochs, name):
        
        train_imgs, labels = next(iter(training_data))
        self.train_imgs = train_imgs
        self.labels = labels

        self.training_data = training_data

        self.model = nn_model
        self.layer_size_params = layer_size_params

        self.criterion_instance = criterion_instance
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs

        self.name = name

    def flatten_input(self, data):
        data = data.view(data.shape[0], -1)
        return data 

    def optimiser(self):
        return optim.SGD(self.model.parameters(), lr = self.lr, momentum = self.momentum)
         
    def train(self, show = True):
        loss_list = []
        optimiser = self.optimiser()
        for epoch in range(self.epochs):
            running_loss = 0
            for images, labels in self.training_data:
                
                # Flatten MNIST images into a 784 long vector
                images = self.flatten_input(images)    
                # labels = labels.view(-1)

                optimiser.zero_grad()

                ## def loss function
                output = self.model(images)
                loss = self.criterion_instance(output, labels)

                # backpropagation
                loss.backward()

                # optimising weights
                optimiser.step()

                running_loss += loss.item()
            loss_list.append(running_loss)
            # print("Epoch {}: Training loss: {}".format(epoch, running_loss/len(self.training_data)))
        
        print("\nTraining complete")

        torch.save({'model_state_dict': self.model.state_dict(),}, 
                   './saved_models/{}.pth'.format(self.name))
        
        print("model saved: /saved_models/{}.pth".format(self.name))

        if show:
            self.plot_loss_vs_epochs(loss_list)


    def plot_loss_vs_epochs(self, loss_list):

        plt.plot(range(1, self.epochs + 1), loss_list, marker='o')
        plt.xticks(range(1,self.epochs,5))
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


class NeuralNetTest:

    def __init__(self, saved_model, test_data):
        self.model = saved_model ## trained model
        self.test_data = test_data

    def flatten_input(self, data):
        data = data.view(data.shape[0], -1)
        return data 

    def run_test(self, show = True):
        correct_count = 0
        total_counts = 0
        predictions = []
        self.model.eval()

        with torch.no_grad():
            for images, labels in self.test_data:
                images = self.flatten_input(images)
                labels = labels.view(-1)

                # Forward pass
                logps = self.model(images)

                # Calculate probabilities using softmax
                ps = nn.functional.softmax(logps, dim=1)
                
                # Get the predicted label with max probability
                _, pred_labels = torch.max(ps, 1)

                for i in range(len(labels)):
                    true_label = labels.numpy()[i].item()
                    pred_label = pred_labels.numpy()[i].item()

                    predictions.append((images[i], true_label, pred_label))

                    if true_label == pred_label:
                        correct_count += 1

                    total_counts += 1

        accuracy = correct_count / total_counts
        print("Number of Correct Predictions:", correct_count)
        print("Number Of Images Tested =", total_counts)
        print("Model Accuracy =", accuracy)

        if show:
            self.show_results(len(labels), predictions)
        

    def show_results(self, num_images, predictions):
        
        fig = plt.figure(figsize=(12, 10))
        
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            rand_idx = random.randint(0, num_images - 1)
            image, label, pred_label = predictions[rand_idx]

            plt.imshow(image.view(28, 28), cmap='gray')
            plt.title("True: {}\nPred: {}".format(label, pred_label))
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()




