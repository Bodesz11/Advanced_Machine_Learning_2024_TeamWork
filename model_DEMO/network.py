from matplotlib import pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, ReLU, BatchNorm2d, Conv2d
import torch.optim as optim
from tqdm import tqdm
import os
import shutil
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
from time import sleep
from enum import Enum
import numpy as np
import cv2



class TrafficSignEUSpeedLimit(Enum):
    EU_SPEEDLIMIT_5 = 0
    EU_SPEEDLIMIT_10 = 1
    EU_SPEEDLIMIT_20 = 2
    EU_SPEEDLIMIT_30 = 3
    EU_SPEEDLIMIT_40 = 4
    EU_SPEEDLIMIT_50 = 5
    EU_SPEEDLIMIT_60 = 6
    EU_SPEEDLIMIT_70 = 7
    EU_SPEEDLIMIT_80 = 8
    EU_SPEEDLIMIT_90 = 9
    EU_SPEEDLIMIT_100 = 10
    EU_SPEEDLIMIT_110 = 11  # Not present in the training data
    EU_SPEEDLIMIT_120 = 12
    EU_SPEEDLIMIT_130 = 13  # Not present in the training data


class BasicBlock(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 expansion: int = 1,
                 downsample: Module = None
                 ) -> None:
        super().__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = Conv2d(in_channels,
                            out_channels,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels,
                            out_channels * self.expansion,
                            kernel_size=3,
                            padding=1,
                            bias=False)
        self.bn2 = BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ClassifierNet(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super(ClassifierNet, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)


def validate_model_loss_acc(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # Disable gradient computation during validation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            correct_val += calculate_accuracy(outputs, labels) * inputs.size(0)
            total_val += labels.size(0)

    val_loss = running_loss / len(data_loader.dataset)
    val_accuracy = correct_val / total_val

    return val_loss, val_accuracy


def train_model(model, train_data_loader, validation_data_loader, epochs=10, patience=3):
    print(f'Training for {epochs} epochs began')
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0
    epochs_without_improvement = 0

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Number of epochs
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Create a tqdm progress bar for the training loop
        with tqdm(train_data_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # Zero the parameter gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item() * inputs.size(0)

                # Update training accuracy
                correct_train += calculate_accuracy(outputs, labels) * inputs.size(0)
                total_train += labels.size(0)

                # Update the progress bar with the current loss
                tepoch.set_postfix(loss=loss.item())

        # Calculate average loss and accuracy over the epoch
        epoch_loss = running_loss / len(train_data_loader.dataset)
        epoch_accuracy = correct_train / total_train

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')

        # Validate the model and compute validation loss and accuracy
        val_loss, val_accuracy = validate_model_loss_acc(model, validation_data_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Check for early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved the best model.')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch + 1}. No improvement in validation accuracy.')
                break
        sleep(0.5)

    # Plot and save loss and accuracy charts
    plot_training_(train_losses, val_losses, train_accuracies, val_accuracies)

    return model

def plot_training_(train_losses, val_losses, train_accuracies, val_accuracies,
                   output_path="loss_accuracy_plot_during_training.png"):
    """
    Plot training and validation loss and accuracy over epochs.

    Parameters:
    - train_losses: List of training loss values per epoch
    - val_losses: List of validation loss values per epoch
    - train_accuracies: List of training accuracy values per epoch
    - val_accuracies: List of validation accuracy values per epoch
    - output_path: Path to save the plot image
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns for loss and accuracy

    # Plot losses
    ax[0].plot(train_losses, label='Training Loss', marker='o', color='red')
    ax[0].plot(val_losses, label='Validation Loss', marker='o', color='blue')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    # Plot accuracies
    ax[1].plot(train_accuracies, label='Training Accuracy', marker='o', color='red')
    ax[1].plot(val_accuracies, label='Validation Accuracy', marker='o', color='blue')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid displaying
    print(f"Loss and accuracy plots saved to {output_path}")



def get_grad_cam_heatmap(model, input_image):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks for the target layer
    target_layer = model.layer4[-1]
    h_grad = target_layer.register_backward_hook(backward_hook)# save_gradients)
    h_act = target_layer.register_forward_hook(forward_hook)# save_activations)

    model.eval()
    input_image = input_image.unsqueeze(0)  # Add batch dimension
    output = model(input_image)

    # Zero out previous gradients
    model.zero_grad()

    # Compute the class score for the target class
    pred_class = output.argmax(dim=1).item()
    class_score = output[:, pred_class]

    # Backward pass to compute gradients
    class_score.backward()

    # Use recorded gradients to compute the weights
    weight = torch.mean(gradients, dim=(1, 2), keepdim=True)

    # Get the output of the target layer (activations)
    target_layer_output = activations  # Use the activations stored from the forward hook

    # Compute the Grad-CAM
    cam = weight * target_layer_output
    cam = torch.sum(cam, dim=1).squeeze()  # Sum across the channel dimension

    # Apply ReLU to the CAM
    cam = torch.clamp(cam, min=0)

    # Resize the CAM to match the original image size
    cam = cam.cpu().detach().numpy()
    cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))  # H x W

    # Normalize the heatmap
    cam -= cam.min()
    cam /= cam.max()
    cam *= 255.0
    heatmap = np.uint8(cam)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Clean up hooks
    h_grad.remove()
    h_act.remove()

    return heatmap


def generate_confusion_matrix(gt_labels, prediction_labels, mapper):
    # Generate confusion matrix
    cm = confusion_matrix([mapper[code] for code in gt_labels],
                          [mapper[code] for code in prediction_labels])

    # Plot confusion matrix
    plt.figure(figsize=(16, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[mapper[code] for code in range(len(cm))],
                yticklabels=[mapper[code] for code in range(len(cm[0]))])
    plt.xlabel('Predicted Labels', rotation=45)
    plt.xticks(rotation=45)
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    return 0


def save_image(tensor, path):
    # Denormalize the tensor
    denormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    tensor = denormalize(tensor)
    tensor = tensor[[2, 1, 0], :, :]
    transform = transforms.ToPILImage()
    image = transform(tensor.cpu().clamp(0, 1))
    image.save(path)


def evaluate_model(model, dataset, other_validation_data, class_mapper,
                   other_training_data=None, device='cpu', num_examples_to_save=1000):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    # Create directories for saving examples
    correct_dir = 'correctly_classified'
    incorrect_dir = 'incorrectly_classified'
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    # Counters to track saved examples
    correct_saved = 0
    incorrect_saved = 0

    # Clear old images
    shutil.rmtree(correct_dir)
    shutil.rmtree(incorrect_dir)
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    gt_labels = []
    pred_labels = []

    # Loop through the dataset
    with torch.no_grad():  # Disable gradient computation for evaluation
        times = []
        for inputs, labels in tqdm(dataset, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            import time

            # Start the timer
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            times.append((end_time - start_time) / len(labels))
            values, predicted = torch.max(outputs, 1)

            # Collect labels and predictions for confusion matrix
            gt_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save examples
            for i in range(inputs.size(0)):
                img = torch.tensor(np.array(inputs[i]))
                if predicted[i] == labels[i]:
                    if correct_saved < num_examples_to_save:
                        save_image(img, os.path.join(correct_dir, f'correct_pred_{str(TrafficSignEUSpeedLimit(int(predicted[i])))}_gt_{str(TrafficSignEUSpeedLimit(int(labels[i])))}_{correct_saved}.png'))
                        # heatmap = get_grad_cam_heatmap(model, inputs[i])  # Get heatmap
                        # overlayed_image = cv2.addWeighted(inputs[i].cpu().numpy().transpose(1, 2, 0), 0.6, heatmap, 0.4,
                        #                                   0)
                        # cv2.imwrite(os.path.join(correct_dir, f'correct_heatmap_{correct_saved}.png'), overlayed_image)
                        correct_saved += 1
                else:
                    if incorrect_saved < num_examples_to_save:
                        save_image(img, os.path.join(incorrect_dir, f'incorrect_pred_{str(TrafficSignEUSpeedLimit(int(predicted[i])))}_gt_{str(TrafficSignEUSpeedLimit(int(labels[i])))}_{incorrect_saved}.png'))
                        incorrect_saved += 1

    # Calculate and print overall accuracy
    accuracy = correct / total
    # generate confusion matrix
    code_to_name = {x.value: x.name for x in class_mapper}
    generate_confusion_matrix(gt_labels, pred_labels, code_to_name)

    print('Average time per iamge:', sum(times) / len(times))
    return accuracy
