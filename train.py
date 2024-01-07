import torch
import torchvision
from torchvision.models import ResNet50_Weights
import swanlab
from torch.utils.data import DataLoader
from load_datasets import DatasetLoader
import os


# Define train function
def train(model, device, train_dataloader, optimizer, criterion, epoch):
    model.train()
    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(TrainDataLoader),
                                                                      loss.item()))
        swanlab.log({"train_loss": loss.item()})


# Define test function
def test(model, device, test_dataloader, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print('Accuracy: {:.2f}%'.format(accuracy))
    swanlab.log({"test_acc": accuracy})


if __name__ == "__main__":
    num_epochs = 20
    lr = 1e-4
    batch_size = 16
    num_classes = 2

    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    # Initialize swanlab
    swanlab.init(
        experiment_name="ResNet50",
        description="Train ResNet50 for cat and dog classification.",
        config={
            "model": "resnet50",
            "optim": "Adam",
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_class": num_classes,
            "device": device,
        }
    )

    TrainDataset = DatasetLoader("datasets/train.csv")
    ValDataset = DatasetLoader("datasets/val.csv")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained ResNet50 model
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Replace the last fully connected layer.
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    # Train
    model.to(torch.device(device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch in range(1, num_epochs + 1):
        train(model, device, TrainDataLoader, optimizer, criterion, epoch)  # Train for one epoch

        if epoch % 4 == 0:  # Test every 4 epochs
            accuracy = test(model, device, ValDataLoader, epoch)

    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    torch.save(model.state_dict(), 'checkpoint/latest_checkpoint.pth')
    print("Training complete")