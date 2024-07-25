# Landmark Classification & Tagging for Social Media

## Project Overview

The goal of this project is to predict the most likely locations where an image was taken based on visible landmarks. Photo sharing and storage services often lack location metadata for uploaded images, which can occur when cameras lack GPS capabilities or metadata is removed for privacy reasons. Inferring the location from the image content can enable advanced features like automatic tag suggestions and photo organization.

To address this, we'll develop a Convolutional Neural Network (CNN) that automatically predicts the location of an image by detecting and classifying recognizable landmarks. Given the vast number of global landmarks and the sheer volume of images uploaded daily, manual classification is impractical.
## Project Structure

The project is divided into two parts:

1. **Building a CNN from Scratch**:
    - Implementing a CNN architecture similar to VGG, ResNet, or AlexNet.
    - Achieving an accuracy higher than 50% on the test dataset.
    - Code available in `cnn_from_scratch.ipynb`.

2. **Using a Pretrained Model**:
    - Utilizing a pretrained ResNet18 model.
    - Achieving an accuracy higher than 60% on the test dataset.
    - Code available in `transfer_learning.ipynb`.

## Dataset

We will use a subset of the Google Landmarks Dataset v2 for training and evaluating our model.

## Getting Started
  **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

## Project Files

- `cnn_from_scratch.ipynb`: Contains the code for building and training a CNN from scratch.
- `transfer_learning.ipynb`: Contains the code for building and training a model using a pretrained ResNet18.
- `src/`: Contains various utility files used throughout the project.

### `src/data.py`

- **Data Initialization and Augmentation**: Initializes the transforms for the train, validation, and test datasets. Various data augmentations are applied to help the model generalize better.
    - **Benefits of Image Augmentation**:
        - Improves model generalization by providing diverse training examples.
        - Helps prevent overfitting by artificially expanding the training dataset.
        - Enhances model robustness by introducing variations in the training images.

- **Dataset Loading**: Utilizes `datasets.ImageFolder` for loading the dataset.
    - **Benefits of `datasets.ImageFolder`**:
        - Simplifies loading of images from a directory where each class has its own folder.
        - Automatically applies specified transforms while loading the dataset.
        - Facilitates easy splitting of datasets into train, validation, and test sets.

- **Data Splitting and Dataloaders**: Splits the dataset into training, validation, and test sets, and initializes dataloaders for use during training and testing.

- **Visualization**: The `visualize_one_batch` function performs an inverse transform on input batch images and plots them using subplots.

### `src/model.py`

- **ResidualBlock Class**: Contains two convolutional layers. The first layer includes a convolutional layer followed by batch normalization and ReLU activation. The second layer includes a convolutional layer followed by batch normalization. If downsampling is required, it ensures the output channels match.

- **Helper Functions**:
    - `get_conv_layer`: Creates a convolutional layer, which includes a convolutional layer, batch normalization, and ReLU activation.
    - `get_fc_layer`: Creates a fully connected layer, which includes a fully connected layer, dropout, and ReLU activation.

- **Model Class**: Designed to achieve high accuracy for this problem.
    - **Architecture**: Combines elements from ResNet and VGG architectures. Initially, more complex architectures were tested, but a simpler architecture was chosen for better performance.
        - Two convolutional layers followed by pooling.
        - Another set of two convolutional layers followed by pooling.
        - Three convolutional layers followed by pooling.
        - Another set of three convolutional layers followed by pooling.
        - Two residual blocks.
        - Global Average Pooling (GAP) layer.
        - Fully connected layer with dropout to reduce overfitting.
        - Final output layer with 50-dimensional vector matching the number of landmark classes.

### `src/optimization.py`

- **Loss Function**: `get_loss()` initializes and returns `nn.CrossEntropyLoss`. If a GPU is available, it moves the loss function to the GPU.

- **Optimizer Initialization**: `get_optimizer()` accepts hyperparameters such as learning rate, momentum, weight decay, the model, and the type of optimizer (e.g., SGD or Adam). It initializes and returns the specified optimizer.

### `src/train.py`

- **Training for One Epoch**: `train_one_epoch()` trains the model using the training data loader and computes the training loss.

- **Validation for One Epoch**: `valid_one_epoch()` validates the model using the validation data loader and returns the validation loss.

- **Optimization and Training Loop**: `optimize()`:
    - Starts with an initial validation.
    - Initializes the learning rate scheduler.
    - Iterates through the specified number of epochs, performing training and validation.
    - Saves the model if the validation loss is the lowest observed.
    - Provides interactive tracking to monitor the model's performance over the epochs.

- **Testing**: `one_epoch_test()` tests the model using the test data loader and returns the test loss.

### `src/predictor.py`

- **Predictor Class**: Accepts a model and sets it to evaluation mode. The `forward` method takes an image, applies necessary transforms, performs inference, and then applies the softmax function to get prediction probabilities.

### `src/transfer.py`

- **Model for Transfer Learning**: `get_model_transfer_learning()` initializes a pretrained model, freezes its layers (disables gradient calculations), replaces the fully connected layer with one suitable for the current problem, and returns the modified model.


## Usage

1. Upload an image through the app interface.
2. The app will process the image and return the top k most relevant landmarks.
3. View the suggested tags and use them for organizing or sharing your photos.

## Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Google Landmarks Dataset v2
- TensorFlow / PyTorch
