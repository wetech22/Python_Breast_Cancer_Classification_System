# Python_Breast_Cancer_Classification_System
# Breast Cancer Tumor Classifier
This project uses a simple neural network to predict whether a tumor is malignant or benign, based on 30 numerical features derived from a digitized image of a breast mass. It's built using TensorFlow and works on the well-known Breast Cancer Wisconsin dataset from sklearn.datasets.

# Dataset Info
Source: sklearn.datasets.load_breast_cancer()  
Instances: 569  
Features: 30 numeric features like mean radius, texture, area error, worst perimeter, etc. 
Target: 0 = Malignant, 1 = Benign

# Tech Stack
Language: Python  
Framework: TensorFlow / Keras  
Libraries: pandas, numpy, matplotlib, scikit-learn (for preprocessing & splitting)

# Model Architecture
Input Layer → 30 features  
Hidden Layer → Dense (20 units, ReLU)  
Output Layer → Dense (2 units, Sigmoid)  
Compiled with:  
loss = sparse_categorical_crossentropy  
optimizer = adam  
metrics = accuracy

# Training & Evaluation
Standardized features using StandardScaler  
Trained on 455 samples, tested on 114  
Achieved ~96.4% accuracy on test data after 10 epochs


# Prediction Strategy
Converts softmax probabilities to class labels using np.argmax()  
Predicts class for new input after reshaping and scaling  
Interprets 0 as Malignant, 1 as Benign

