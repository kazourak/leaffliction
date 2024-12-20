
# leaffliction

An innovative computer vision project utilizing leaf image analysis for disease recognition.

## Setup project

- ### Download the project
    `git clone git@github.com:kazourak/leaffliction.git`

- ### Setup the environment
    `make create_environment` We develop the project with the python virtual environment.

    `source .venv/bin/activate` Source the virtual environment.

    `make setup` Download required dependencies and set up the project as a python usable package.

- ### Download data
    `make data` Download from the school intranet the given dataset.

## Commands and usage
_Don't forget ! If you want a script usage you can call the option `--help`._

### Data Understanding

- ### Get a quick look to data repartition
    `python leaffliction/Distribution.py <directory_path> --plant_type <plant_type>` Get a quick look to the
    data set repartition. You can see the repartition of a single plant type by using the option `--plant_type`.

- ### Augment data
    Data augmentation is essential for training CNNs as it enhances generalization, expands dataset size, and makes
    models more robust to real-world variations. It helps prevent overfitting, reduces dependency on large labeled
    datasets, balances class distributions, encourages invariance to transformations, and mitigates dataset biases.
    By simulating diverse conditions through techniques like geometric transformations, color adjustments, and noise
    injection, data augmentation improves model performance and reliability.

    `python leaffliction/Augmentation <image_path>` Augment a single image by using six methods:
    clahe, zoom blur, rotate, blur contrast and flip.
    
    `python leaffliction/tools/improve_dataset.py <dataset_directory_path> <destination_path> -v` Augment the entire
    data set and save the produced images into the destination directory.

- ### Transform data
    `python leaffliction/Transform.py <source_path> --dest <destination_path>`Transformations ensure to focus on
    essential features of the data rather than overfitting to specific, irrelevant details.


### Model training and prediction
For the training, predict and evaluate processes we use tensorflow.keras that provides usefully features to create data
set (training and evaluate), create the model, save it and train. This part of the project is more implementation and
understanding of CNNs concepts than development.

- ### Train a model
    `python leaffliction/train.py <dataset path> --save_model` _The script provides more options to customise the
    learning process._

- ### Evaluate model performance
    `python leaffliction/evaluate.py <model path> <dataset path>` It's important to use the same parameters as your training
    phase (dataset and options) to have a realistic evaluation. (You can modify the number of images used to evaluate
    the model by modifying the --evaluation_ratio).

- ### Predict class
    `python leafflication/predict.py <dir or file path> -p` The predict script can be used to predict the class of an
    image (or more) even if data are not structured. the option `-p` plot the predicted images.

## Clean project
`make clean` Suppress data and virtual environment.

--------

# What's a CNN (Convolutional Neural Network)
A Convolutional Neural Network (CNN) processes images by learning features hierarchically:

1. **Convolution Layers**: Extract features using small filters.
2. **ReLU Activation**: Adds non-linearity to capture complex patterns.
3. **Pooling Layers**: Reduce spatial size, making the model efficient and robust.
4. **Fully Connected Layers**: Flatten features and map them to outputs.
5. **Output Layer**: Predicts class probabilities (e.g., for classification tasks).

CNNs excel in tasks like image recognition by learning spatial hierarchies of features efficiently.

# CNN history
The history of CNNs is marked by key milestones:

1. **1980s - Concept Origin**:  
   - **Neocognitron (1980)** by Kunihiko Fukushima introduced the idea of convolution and hierarchical feature extraction.
   - Inspired by biological vision systems.

2. **1990s - LeNet**:  
   - **LeNet-5 (1998)** by Yann LeCun popularized CNNs for digit recognition (e.g., reading checks).  
   - Limited by computing power and data availability.

3. **2000s - Foundations**:  
   - Techniques like ReLU activation and backpropagation were refined.  
   - Datasets like ImageNet emerged, providing large-scale data.

4. **2012 - AlexNet Breakthrough**:  
   - Alex Krizhevsky's **AlexNet** won the ImageNet competition, outperforming traditional methods.  
   - Leveraged GPUs for faster training, ReLU activation, and dropout for regularization.

5. **2014-2015 - Deeper Networks**:  
   - **VGGNet** and **GoogLeNet** (Inception) explored deeper architectures for improved performance.  
   - **ResNet (2015)** introduced residual connections, enabling very deep networks.

6. **2016+ - Modern CNNs**:  
   - Architectures like DenseNet, MobileNet, and EfficientNet focused on efficiency and scalability.  
   - Applications expanded to object detection, medical imaging, and more.

CNNs have evolved from small-scale experiments to the backbone of modern computer vision systems.

# Model explanation


```python
def _load_model(input_size: int, output_size: int) -> tf.keras.Model:
    """
    Build and compile the CNN model from tensorflow tutorial.
    Parameters
    ----------
    input_size : Length of input image side.
    output_size : Number of possible labels.

    Returns
    -------
    The compiled CNN model.
    """
    model = tf.keras.models.Sequential(
        [
            # Extract features layers.
            tf.keras.layers.Input(shape=(input_size, input_size, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            # Hidden layer (learn).
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation="relu"),
            # Use soft max to extract the highest result (the predicted label).
            tf.keras.layers.Dense(output_size, activation="softmax"),
        ]
    )
    model.compile(
        # Function used to optimise weights during the training session.
        optimizer="adam",
        # Function used to evaluate the result.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
```


### Model Architecture
The model is built using the `Sequential` API, where layers are stacked in order.

#### Input Layer
```python
tf.keras.layers.Input(shape=(input_size, input_size, 3))
```
- Specifies the input shape:
  - `(input_size, input_size, 3)` means the input is an image with `input_size x input_size` resolution and 3 color channels (RGB).


#### Convolutional and Pooling Layers:
The model uses multiple pairs of `Conv2D` and `MaxPooling2D` layers to extract hierarchical features.

- Conv2D
  - Each convolution layer applies 3x3 filters to the input to extract spatial features.
  - The first layer has 32 filters, followed by layers with 64 filters, meaning more features are captured as the network goes deeper.
  - The activation function is ReLU (`activation="relu"`) to introduce non-linearity.

- MaxPooling2D
  - Reduces the spatial dimensions (height and width) by taking the maximum value in each 2x2 region (`pool_size=(2, 2)`).
  - This helps reduce computational complexity and makes the model more robust to small shifts in the input.


#### Dropout Layers
```python
tf.keras.layers.Dropout(0.5)
```
- Regularization technique that randomly drops 50% of neurons during training.
- Prevents overfitting by forcing the model to generalize better.


#### Flatten Layer
```python
tf.keras.layers.Flatten()
```
- Flattens the 3D feature maps from the last convolutional layer into a 1D vector.
- Prepares the data for the dense (fully connected) layers.


#### Dense Layers
```python
tf.keras.layers.Dense(512, activation="relu"),
tf.keras.layers.Dense(output_size, activation="softmax"),
```
- The first dense layer has 512 neurons with ReLU activation to learn high-level features.
- The final dense layer has `output_size` neurons with a **softmax** activation, producing a probability distribution
  over the classes.


### Compilation
The model is compiled with the following configurations:

#### Optimizer
```python
optimizer="adam"
```
- Adam optimizer is used, combining momentum and adaptive learning rates for efficient training.

#### Loss Function
```python
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
- `SparseCategoricalCrossentropy` is used for multi-class classification when labels are integers.

#### Metrics
```python
metrics=["accuracy"]
```
- Tracks the classification accuracy during training and evaluation.


### Model Summary
The architecture extracts features with convolution and pooling, reduces overfitting with dropout, and uses fully
connected layers for classification. It's suitable for image-based multi-class classification tasks. 

Key Points
- Depth: Deeper layers extract higher-level features.
- Pooling and Dropout: Improve robustness and prevent overfitting.
- Final Dense Layers: Map features to class probabilities.
