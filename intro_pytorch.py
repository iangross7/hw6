import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)

    if training:
        return torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        return torch.utils.data.DataLoader(test_set, batch_size = 64)
    



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )



def build_deeper_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    n = len(train_loader.dataset) # Dataset Size

    for epoch in range(T):
        runningLoss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # backwards pass
            loss.backward()
            optimizer.step()
            
            # batch loss times batch size here
            runningLoss += loss.item() * inputs.size(0)
            
            # calculate predictions & keep counts
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avgLoss = runningLoss / n
        accuracy = (correct / total) * 100
        
        # Print the training status in the required format:
        # "Train Epoch: ? Accuracy: ?/?(??.??%) Loss: ?.???"
        print("Train Epoch: {} Accuracy: {}/{}({:.2f}%) Loss: {:.3f}".format(
            epoch, correct, total, accuracy, avgLoss
        ))

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()

    n = len(test_loader.dataset)
    correct = 0
    total = 0
    runningLoss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            runningLoss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100

    if (show_loss):
        print("Average loss: {:.4f}".format(runningLoss / total))
    
    print("Accuracy: {:.2f}%".format(accuracy))

    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    
    model.eval()
    img = test_images[index].unsqueeze(0) # adding batch dimension to evaluate
    logits = model(img)

    allProbs = F.softmax(logits, dim=1)

    topProbs, topClasses = allProbs.topk(3, dim=1)

    for i in range(3):
        label = class_names[topClasses[0][i].item()]
        probPercent = topProbs[0][i].item() * 100
        print("{}: {:.2f}%".format(label, probPercent))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    loader = get_data_loader()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    train_model(model, loader, criterion, 5)
    evaluate_model(model, loader, criterion, True)
    test_images = next(iter(loader))[0]
    predict_label(model, test_images, 2)
