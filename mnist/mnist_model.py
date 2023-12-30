import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MnistModel:
    def __init__(self, batch_size, shuffle=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        ## transforms.ToTensor() --> TorchTensor, converting image to numbers
        ## transforms.Normalise() --> nornmalising tensor with mean and s.d.
        self.size = batch_size
        self.train_data, self.test_data = self.read_data(batch_size, shuffle)

    def download_data(self):
        train_set = datasets.MNIST('trainset', download=True, train=True, transform=self.transform)
        test_set = datasets.MNIST('testset', download=True, train=False, transform=self.transform)
        return train_set, test_set

    def read_data(self, batch_size, shuffle):
        train_set, test_set = self.download_data()
        load_train_set = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        load_test_set = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
        return load_train_set, load_test_set

    def show_data(self, num_of_images):
        raise NotImplementedError("show_data method must be implemented in subclasses")

class MnistTrainingModel(MnistModel):
    def __init__(self, mnist_model):
        super().__init__(mnist_model.size)
        training_images, training_labels = next(iter(self.train_data))
        self.images = training_images
        self.labels = training_labels

    def show_images(self, num_images):

        num_row_images = num_images // 10
        for index in range(1, num_images + 1):
            plt.subplot(num_row_images, 10, index)
            plt.axis('off')
            plt.imshow(self.images[index].numpy().squeeze(), cmap='gray')

class MnistTestingModel(MnistModel):
    def __init__(self, mnist_model):
        super().__init__(mnist_model.size)
        testing_images, testing_labels = next(iter(self.test_data))
        self.images = testing_images
        self.labels = testing_labels

    def show_images(self, num_images):

        num_row_images = num_images // 10
        for index in range(1, num_images + 1):
            plt.subplot(num_row_images, 10, index)
            plt.axis('off')
            plt.imshow(self.images[index].numpy().squeeze(), cmap='gray')

