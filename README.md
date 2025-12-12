# Breast Cancer Classification

This repository contains two components:

1. **Jupyter Notebook** (`notebook.ipynb`): Performs exploratory data analysis and builds a Support Vector Classifier (SVC) model using GridSearchCV on the scikit-learn breast cancer dataset.
2. **Streamlit App** (`app.py`): A web application for real-time breast cancer prediction using the trained model and scaler.

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)

  * [1. Jupyter Notebook](#1-jupyter-notebook)
  * [2. Streamlit App](#2-streamlit-app)
* [Usage](#usage)

  * [Notebook Usage](#notebook-usage)
  * [App Usage](#app-usage)
* [File Structure](#file-structure)
* [Model and Files](#model-and-files)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* Load and inspect the built-in breast cancer dataset from scikit-learn.
* Perform data cleaning, visualization (pairplots), and feature scaling.
* Train an SVC model with hyperparameter optimization using `GridSearchCV`.
* Save the best model and scaler via pickle for future inference.
* Streamlit-based web interface for manual input and live prediction.
* Post-prediction guidance for both malignant and benign outcomes.

---

## Prerequisites

* Python 3.7 or higher
* pip
* Recommended to use a virtual environment (venv or conda)

Install required packages:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```
pandas
numpy
scikit-learn
seaborn
matplotlib
streamlit
```

---

## Getting Started

### 1. Jupyter Notebook

1. Clone the repository:

   ```bash
   ```

git clone \<repo\_url>
cd \<repo\_folder>

````
2. Launch Jupyter Lab/Notebook:
```bash
jupyter notebook notebook.ipynb
````

3. Execute cells sequentially to:

   * Load and explore data
   * Visualize relationships with `pairplot`
   * Preprocess features (scaling)
   * Train and evaluate an SVC model using GridSearchCV
   * Perform manual prediction and save the model

### 2. Streamlit App

1. Ensure the pickled model (`best_svc_model.pkl`) and scaler (`scaler.pkl`) exist in the project root.
2. Run the Streamlit application:

   ```bash
   ```

streamlit run app.py

```
3. A browser window will open with the prediction interface.

---

## Usage

### Notebook Usage

- Walk through each section of the notebook:
1. **Data Loading**: Import the breast cancer dataset.
2. **EDA & Visualization**: Understand feature distributions and correlations.
3. **Preprocessing**: Scale features with `StandardScaler`.
4. **Model Training**: Use `GridSearchCV` for hyperparameter tuning.
5. **Evaluation**: Check accuracy, precision, recall, F1-score, and confusion matrix.
6. **Serialization**: Save the best model and scaler to `.pkl` files.

### App Usage

- Input 30 numerical feature values in the web form.
- Click the **🔮 Predict** button.
- See instant classification: **Benign** or **Malignant**.
- Read recommended next steps and precautions based on the result.

---

## File Structure

```

├── notebook.ipynb           # Jupyter notebook for model development
├── app.py                   # Streamlit application code
├── best\_svc\_model.pkl       # Pickled best SVC model
├── scaler.pkl               # Pickled StandardScaler instance
├── requirements.txt         # Python dependencies
└── README.md                # This file

```

---

## Model and Files

- **best_svc_model.pkl**: Contains the trained SVC model (best estimator from GridSearchCV).
- **scaler.pkl**: Contains the fitted `StandardScaler` for input normalization.

Load these files in any Python script to perform inference without retraining.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:

- Additional visualizations
- Deployment scripts (Docker, CI/CD)
- Integration with external data sources

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

```