<h1 align="center">Machine Learning</h1>

<br>

Welcome to my repository dedicated to the captivating realm of Machine Learning!

<br>

<h2 align="center">🌅 Journey Highlights 🌅</h2>
<p>


---

Before diving into the projects, you'll find a comprehensive list of Models and data cleaning Methods

<br>

<details>
  <h2 align="center"> 📚 Models & Methods 📚 </h2>
  
  <summary> 📚 Models & Methods 📚</summary> 
<p>

<h3>Data Modification</h3>

**Binning:**
<p>Binning is a data preprocessing technique that involves grouping continuous numerical data into discrete intervals or "bins," simplifying complex distributions and reducing noise.</p>

**Mapping:**
<p>Mapping involves transforming values from one range to another, often used to normalize or scale features within a specific desired range.

**Standard Scaling:**
<p>Standard Scaling, or Z-score normalization, standardizes numerical features by rescaling them to have a mean of 0 and a standard deviation of 1, facilitating comparison between different scales of data.

**One-Hot Encoding:**
<p>One-Hot Encoding is a method for representing categorical variables as binary vectors, creating binary columns for each category and indicating the presence or absence of that category in the data.

**Box-Cox Transformation:**
<p>The Box-Cox Transformation is a statistical technique that stabilizes the variance and makes a distribution more closely approximate a normal distribution by applying a power transformation.

<h1></h1>
<h3>Upgrading Models</h3>

**GridSearchCV for Hyperparameter Tuning:**
<p>GridSearchCV is a technique for systematically searching and selecting the optimal combination of hyperparameters for a machine learning model by evaluating performance across different parameter values.

**Cross-validation:**
<p>Cross-validation is a validation technique that partitions the dataset into subsets, training the model on some subsets and testing it on others to assess its performance and generalization.

<h1></h1>
<h3>Supervised Learning Models</h3>

**RandomForestClassifier (Random Forest):**
<p>RandomForestClassifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks.

**LogisticRegression (Logistic Regression):**
<p>LogisticRegression is a linear model for binary classification that estimates the probability of an instance belonging to a particular class.

**AdaBoostClassifier (AdaBoost):**
<p>AdaBoostClassifier is an ensemble learning algorithm that combines weak learners sequentially, with each focusing on the mistakes of its predecessor, to improve overall accuracy.

**AdaBoostClassifier with SVM (AdaBoost with SVM):**
<p>AdaBoostClassifier with SVM involves boosting the performance of a Support Vector Machine using the AdaBoost algorithm.

**GradientBoostingClassifier (Gradient Boosting):**
<p>GradientBoostingClassifier is an ensemble learning method that builds a series of weak learners, typically decision trees, to progressively correct errors and improve model accuracy.

**DecisionTreeClassifier (Decision Tree):**
<p>DecisionTreeClassifier is a model that partitions the dataset into subsets based on feature values, creating a tree-like structure to make decisions.

**Model Blending:**
<p>Model Blending combines predictions from multiple models to produce a final prediction, often enhancing overall model performance.

<h1></h1>
<h3>Specific Models</h3>

**XGBoost Classifier:**
<p>XGBoost Classifier is an implementation of gradient-boosted decision trees designed for speed and performance.

**Support Vector Machines (SVM): AdaBoostClassifier with SVM:**
<p>AdaBoostClassifier with SVM boosts the performance of a Support Vector Machine using the AdaBoost algorithm.

**Logistic Regression: Lasso | Ridge | Elastic Net-Regularization:**
<p>Logistic Regression with Lasso, Ridge, or Elastic Net regularization introduces penalties to control the magnitude of coefficients, preventing overfitting.

**GradientBoost Classifier:**
<p>GradientBoost Classifier is an ensemble learning method that builds a series of weak learners, typically decision trees, to improve model accuracy.

**Monte Carlo Simulations:**
<p>Monte Carlo Simulations involve using random sampling and probability distributions to model and analyze various outcomes in a system.


<h1></h1>
<h3>Dimensionality Reduction</h3>

**Principal Component Analysis (PCA):**
<p>Principal Component Analysis is a technique for reducing the dimensionality of data while preserving its variance, often used for feature extraction and visualization in high-dimensional datasets.


</p>
  <br>
</details>

<br>

<h2 align="center">🔎 Repository Overview 🔍</h2>

This repository is a testament to my exploration and experimentation within the domain of Deep Learning. It is divided into two primary sections:

<br>


<details>
  <h2 align="center"> Logistic Regression using Lasso | Ridge | Elasitc Net-Reglarization </h2>
  
  <summary> Logistic Regression using Lasso | Ridge | Elasitc Net-Reglarization </summary> 

  <p>
The provided code fits Logistic Regression models with different regularization techniques on a breast cancer dataset. The L1-regularized model (Lasso), L2-regularized model (Ridge), and elastic net-regularized model are trained on a standardized training set. The models are then evaluated using a comprehensive evaluation function, including metrics such as confusion matrix, accuracy, precision, recall, F1 score, ROC curve, and the distribution of predicted probabilities. Additionally, the code extracts and analyzes the coefficients of the features from each model, providing insights into the importance of individual features in making predictions. The elastic net model, which combines L1 and L2 regularization, aims to strike a balance between feature selection and regularization. The overall approach demonstrates a thorough analysis of logistic regression models with different regularization techniques applied to a breast cancer classification task.
<a href="https://github.com/trystan-geoffre/Machine-Learning/blob/master/Python/LASSO(L1)%20%7C%C2%A0Ridge(L2)%20%7C%20Elastic%20Net%20Regularization.ipynb"> Code Link</a>
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> GradiantBoost & GridSearchCV </h2>
  
  <summary> GradiantBoost & GridSearchCV </summary> 

  <p>
The code begins by loading the Boston Housing dataset and organizing its features and target variable into Pandas DataFrames. Subsequently, it splits the dataset into training and testing sets using the train_test_split function from scikit-learn. Then, a Gradient Boosting Regressor model is created and trained on the training set. Predictions are made on the test set, and the R-squared score is calculated to evaluate the model's performance.
Following this, the code visualizes the feature importances using a horizontal bar chart. It normalizes and sorts the importances before plotting. Finally, hyperparameter tuning is performed using GridSearchCV to optimize the Gradient Boosting Regressor model. The grid includes different combinations of learning rates and numbers of estimators. The best hyperparameters and their corresponding R-squared score on the training set are printed, providing insights into the optimal configuration for the model. <a href="https://github.com/trystan-geoffre/Machine-Learning/blob/master/Python/GradiantBoost%20%26%20GridSearchCV.ipynb"> Code Link</a>
  </p>
  <br>
</details>

<br>


<details>
  <h2 align="center"> Support Vector Machines (SVM) & One Hot Encoder (OHE) & Principal Component Analysis (PCA) </h2>
  
  <summary> Support Vector Machines (SVM) & One Hot Encoder (OHE) & Principal Component Analysis (PCA) </summary> 

  <p>
The code reads data from an Excel file into a Pandas DataFrame and performs several data processing steps. It renames columns, drops unnecessary columns, and conducts exploratory data analysis. It handles missing values and class imbalances through resampling. The code then prepares the data for modeling by encoding categorical features, splitting into training and testing sets, and scaling the features.

The Support Vector Classification (SVC) model is trained, and hyperparameter tuning is performed using grid search. The tuned model is then evaluated on the test set. Principal Component Analysis (PCA) is applied to reduce dimensionality, and the first two principal components are used to train an SVM model. The decision surface of the model is visualized in a 2D plot.

Overall, the code covers data preprocessing, model training and tuning, dimensionality reduction, and visualization to analyze the performance of an SVM classifier on credit card default prediction.
<a href="https://github.com/trystan-geoffre/Machine-Learning/blob/master/Python/SVM/Support%20Vector%20Machines%20(SVM)%20%26%20One%20Hot%20Encoder%20(OHE)%20%26%20Principal%20Component%20Analysis%20(PCA).ipynb"> Code Link</a>
  </p>
  <br>
</details>

<br>


<details>
  <h2 align="center"> Case Study 1 & 2: Data Cleaning</h2>
  
  <summary> Case Study 1 & 2 </summary> 

  <p>
In the context of a case study focused on data manipulation, the two examples illustrate common practices in data cleaning and enhancement. For both, the code addresses fundamental tasks such as handling missing values, eliminating irrelevant rows, and removing duplicate entries. It also encompasses actions like altering data types, concatenating information, and rectifying spelling errors. <a href="https://github.com/trystan-geoffre/Machine-Learning/blob/master/Python/Case%20Study%201.ipynb"> Code Link for Case Study 1</a>

For the second case, the code expands its scope to advanced operations. Apart from the foundational cleaning steps, it involves sorting data for improved organization, ranking data to identify patterns or outliers, extracting insightful information to address specific queries, and employing data visualization techniques for enhanced comprehension.  <a href="https://github.com/trystan-geoffre/Machine-Learning/blob/master/Python/Case%20Study%202.ipynb"> Code Link for Case Study 2</a>
  </p>
  <br>
</details>

<br>


<details>
  <h2 align="center"> Monte Carlo Simulations to Predict Stock Price </h2>
  
  <summary> Monte Carlo Simulations to Predict Stock Price </summary> 

  <p>
The code implements a Stock Price Prediction Model using Monte Carlo simulation. It begins by extracting historical stock price data for Microsoft (MSFT) from Yahoo Finance. It analyzes the data by calculating and visualizing historical log returns and their distribution, along with computing key statistical measures like mean, variance, and standard deviation.

The Monte Carlo simulation is then applied to simulate future daily returns using random numbers sampled from a normal distribution. This simulation generates a spectrum of potential future stock prices through iterative simulations, and the results are visualized. The model is further enhanced by including drift, adjusting daily returns based on the mean and variance. The quantification and analysis section calculates worst, average, and best-case scenarios for future stock prices. Confidence intervals are established to provide a range of possible future prices, offering insights into the potential variability of future scenarios. <a href="https://github.com/trystan-geoffre/Machine-Learning/blob/master/Python/Modeling%20Risk%20with%20Monte%20Carlo%20in%20Python%20-%20Downloads/Stocks%20Price%20Prediction.ipynb"> Code Link</a>
  </p>
  <br>
</details>

<br>


<details>
  <h2 align="center"> Predicting Titanic Survivors </h2>
  
  <summary> Predicting Titanic Survivors </summary> 

  <p>

<a href=""> Code Link</a>
  </p>
  <br>
</details>

<br>
This marks the conclusion of the repository on Machine Learning! For those interested in exploring Deep-Learning with the use of TensorFlow, I would invite you to visit the repository <a href="https://github.com/trystan-geoffre/Deep-Learning-TensorFlow"> Deep Learning with TensorFlow </a> to witness the exploration of Deep-Learning.
