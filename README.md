# Handwritten Digit Recognition using Machine Learning and Deep Learning

## Requirements

- Python 3.5 or higher
- Scikit-Learn
- Numpy (+ mkl for Windows)
- Matplotlib

## Steps to Run the Code

### 1. Download the MNIST Dataset
You can download the dataset directly using the following commands:

```bash
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

Alternatively, you can [download the dataset as a ZIP file](https://github.com/anujdutt9/Handwritten-Digit-Recognition-using-Deep-Learning/blob/master/dataset.zip) and extract it into the correct folders.

### 2. Organize Dataset
Place the dataset files into the `dataset` folder for each algorithm like so:

```
KNN
|_ MNIST_Dataset_Loader
   |_ dataset
      |_ train-images-idx3-ubyte
      |_ train-labels-idx1-ubyte
      |_ t10k-images-idx3-ubyte
      |_ t10k-labels-idx1-ubyte
```

Repeat this for the SVM and RFC folders as well.

### 3. Run the Code
Navigate to the directory of the algorithm you want to use. For example:

```bash
cd 1. K Nearest Neighbors/
```

Then run the Python file:

```bash
python knn.py
```

Or:

```bash
python3 knn.py
```

To see the output directly in the command prompt, uncomment lines 16, 17, 18, 106, and 107 in `knn.py`. Otherwise, results will be saved in a `summary.log` file.

### 4. Running the CNN Code
For CNN, the dataset downloads automatically. Just run:

```bash
python CNN_MNIST.py
```

Or:

```bash
python3 CNN_MNIST.py
```

### 5. Saving CNN Model Weights
To save the model after training:

```bash
python CNN_MNIST.py --save_model 1 --save_weights cnn_weights.hdf5
```

### 6. Loading Saved Model Weights
To load saved weights and skip training:

```bash
python CNN_MNIST.py --load_model 1 --save_weights cnn_weights.hdf5
```

## Accuracy Results

### Machine Learning Algorithms:

- **K Nearest Neighbors:** 96.67%
- **SVM:** 97.91%
- **Random Forest Classifier:** 96.82%

### Deep Neural Networks:

- **Three-Layer CNN (TensorFlow):** 99.70%
- **Three-Layer CNN (Keras + Theano):** 98.75%

## Additional Resources
- **Hardware Used:** Intel Xeon Processor / AWS EC2 Server
