
# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

##  Objective
To build a basic classification model that predicts the **species** of an iris flower based on:
- Sepal length
- Sepal width
- Petal length
- Petal width

---

##  Dataset
The Iris dataset is a built-in dataset provided by `sklearn.datasets`.

### Features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

### Target:
- Species: `setosa`, `versicolor`, `virginica`

---

##  Tools Used
- Python
- Jupyter Notebook / Google Colab
- Libraries:
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`

---

##  ML Algorithm
- **K-Nearest Neighbors (KNN)** using `sklearn.neighbors.KNeighborsClassifier`

---

##  Steps Performed
1. Imported the Iris dataset using `sklearn.datasets`.
2. Converted it into a pandas DataFrame.
3. Displayed the first few rows of the data.
4. Visualized the dataset using pair plots (Seaborn).
5. Split the data into training and testing sets.
6. Applied and trained the **KNN algorithm** (with `k = 3`).
7. Made predictions and calculated the accuracy.
8. Visualized the **confusion matrix** using Seaborn heatmap.

---

##  Accuracy & Evaluation
- The model was evaluated using `accuracy_score`.
- Confusion matrix shows how predictions match actual species classes.

---

##  Example Code Snippet
```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

---

##  How to Run the Code
1. Clone or download the repository.
2. Open `iris_flower_classification_knn.ipynb` in Jupyter or Colab.
3. Run all cells to see data analysis, model training, and prediction results.

---

##  GitHub Link (replace with your own)
[👉 View the Notebook on GitHub](https://github.com/shafsoft2010/iris-flower-classification-knnS)

---
