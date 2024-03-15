# Machine Learning Practicals Repository

Welcome to my machine learning practicals repository! This repository is dedicated to storing the experiments and projects I work on as part of my university machine learning coursework.

## Experiment 1: Perceptron Binary Classification

### Overview
In this experiment, I implemented a Perceptron algorithm for binary classification. The Perceptron is a fundamental building block in the field of machine learning, and this experiment serves as a starting point for understanding its functionality and applications.

### Files
- [Perceptron.ipynb](./Perceptron_BinaryClassifier/Perceptron_BinaryClassifier.ipynb): Jupyter Notebook containing the implementation of the Perceptron algorithm.
- [Sklearn make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html): Sample dataset used for training and testing the Perceptron.

### Instructions
1. Clone this repository to your local machine.
2. Navigate to the `Perceptron_Binary_Classification` folder.
3. Run the [Perceptron.ipynb](./Perceptron_BinaryClassifier/Perceptron_BinaryClassifier.ipynb) script to train and test the Perceptron on the provided dataset.
4. Experiment with different parameters and datasets to gain a deeper understanding of the Perceptron's behavior.

## Experiment 2: Linear and Ridge Regression

### Overview
This experiment implements linear regression, a fundamental statistical method for modeling the relationship between a dependent variable (y) and one or more independent variables (X). It aims to find a linear function that best fits the data points, enabling prediction of the dependent variable based on the independent variables.

### Files
- [Linear_Regression.ipynb](./Regression/Linear_Regression.ipynb): Jupyter Notebook containing the implementation of the Linear Regression.
- [Ridge_Regression.ipynb](./Regression/Ridge_Regression.ipynb): Jupyter Notebook containing the implementation of the Linear Regression.
- [Sklearn make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html): Sample dataset used for training and testing Regression Model.

### Instructions
1. Import make_regression dataset(x,y): 100 samples, 1 feature, No Bias term and noise with std dev of 5. 
2. Range of x is [-5 5] and y is [15 -15].
3. Split the data set into training (80%) and testing(20%) dataset.
4. Display overall dataset in one scatter plot, training dataset and testing dataset on one scatter plot.
5. Add bias to each instance of input training data set. (use np.c_) to train the model: y = wTx + b = w1x1 +w0x0; w = [w1] and x = [x1] are 1 x 1; b is bias term =w0 with x0=1 ; x_b = [1, x1]T; w_b = [w0, w1]T
6. Obtain the linear regression co-efficients using close form solution formula and using training data set.
7. Test the algorithm using testing dataset and obtain prediction.
8. Find and display Root Mean Square Error (RMSE) for training as well as testing. Use the function mean_squared_error in sklearn.metrics for performance evaluation.
9. Plot straight line equation using co-efficients obtained on same scatter plot showing the training and testing datasets.
Verify the results with sklearn in-built function: LinearRegression().

10. Repeat the same experiment as in part (a) with same dataset for Ridge Regression algorithm with hyper-parameter values of 0.1, 0.5, 1.0 and 1.5 (denoted by α or λ).
11. Sklearn in-built function is Ridge()

### Learning Objectives
1. Understand the core concepts of regression.
2. Implement linear regression from scratch using matrix operations.
3. Evaluate model performance using MSE and regularization terms.
4. Visualize the relationship between variables and the fitted model.
Compare the custom implementation with a popular machine learning library (scikit-learn).




## Experiment [N]: [Experiment Name]

[Repeat the above sections for each experiment you add to the repository.]

## Future Work
I plan to continue adding experiments to this repository on a weekly basis. Stay tuned for updates!

## Contributing
If you would like to contribute to this repository by adding your own experiments or improvements, feel free to fork the repository and submit a pull request.

## Contact
If you have any questions or suggestions regarding this repository, please feel free to reach out to me via email or GitHub.

Happy learning!


