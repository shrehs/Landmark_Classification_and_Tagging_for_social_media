# Landmark Classification & Tagging for Social Media 2.0

## Project Overview
This project involves building a Convolutional Neural Network (CNN) from scratch and using transfer learning to classify landmarks. The model is then deployed as an application to predict the class of new images.

## File Structure
- `cnn_from_scratch.ipynb`: Notebook for building and training a CNN from scratch.
- `transfer_learning.ipynb`: Notebook for applying transfer learning.
- `app.ipynb`: Notebook for running the final application.
- `src/data.py`: Data loading and preprocessing.
- `src/model.py`: CNN model definition.
- `src/helpers.py`: Helper functions.
- `src/predictor.py`: Predictor class for TorchScript model.
- `src/transfer.py`: Transfer learning model definition.
- `src/optimization.py`: Loss and optimizer functions.
- `src/train.py`: Training and validation functions.

## Data Preprocessing
- Images are resized to 256x256 pixels and cropped to 224x224 pixels.
- Normalization using the dataset's mean and standard deviation.
- Data augmentation techniques like random cropping and horizontal flipping are applied to the training set.

## Model Architecture
### CNN from Scratch
- Three convolutional layers with ReLU activation and max pooling.
- Fully connected layers with dropout for regularization.
- The architecture is designed to balance complexity and performance.

### Transfer Learning
- Utilizes a pre-trained ResNet-18 model.
- The final fully connected layer is replaced to match the number of classes.

## Training and Evaluation
- The model is trained using CrossEntropyLoss and optimized with SGD/Adam.
- Training includes validation checks and learning rate scheduling.
- The best model weights are saved based on validation loss.

## Application
- The application uses TorchScript to load the model and predict the class of new images.
- The model is evaluated on a test set with a minimum accuracy requirement.

## Additional Use Cases
- The model can be extended to other classification tasks by retraining with appropriate datasets.
- Features from the CNN can be used for image retrieval or object detection tasks.

## Running the Project
1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Run the notebooks in the following order:
    - `cnn_from_scratch.ipynb`
    - `transfer_learning.ipynb`

## Notes
- Experiment with different hyperparameters to improve accuracy.
- Avoid overfitting by using regularization techniques and data augmentation.
