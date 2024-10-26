# 🔥 GitHub Actions and CML for Machine Learning - Churn Prediction Model 🔥

## Overview

This project demonstrates how to build a machine learning pipeline for predicting customer churn using a Gradient Boosting Classifier. It leverages GitHub Actions for automation and Continuous Machine Learning (CML) for tracking metrics and results directly in GitHub.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Workflow Overview](#workflow-overview)
- [How to Use](#how-to-use)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Processing:** Efficient loading and preprocessing of data.
- **Model Training:** Uses Gradient Boosting for churn prediction.
- **Model Evaluation:** Confusion matrix and ROC curve visualization.
- **CI/CD Integration:** Automates model training and evaluation using GitHub Actions.
- **CML Integration:** Generates reports of model metrics and plots.

<img src="https://github.com/Abonia1/Github-Action-for-ML/blob/main/actions.png">

## Technologies Used

- Python
- Scikit-learn
- Pandas
- Matplotlib
- GitHub Actions
- Continuous Machine Learning (CML)

## Getting Started

### Prerequisites

To run this project, ensure you have:

- Python 3.6 or higher
- Git
- Access to GitHub for hosting the repository

### Clone the Repository

```bash
git clone https://github.com/Abonia1/Github-Action-for-ML
cd Github-Action-for-ML
```

### Install Dependencies

Make sure you have `requirements.txt` in the repository with necessary libraries. Install them using:

```bash
pip install -r requirements.txt
```

## Workflow Overview

This project utilizes GitHub Actions to automate the training and evaluation of the churn prediction model. The workflow triggers on pushes and pull requests to the `main` branch.

### GitHub Actions Configuration

The `.github/workflows/main.yml` file defines the workflow. Key steps include:

1. **Checkout Repository:** Retrieves the code from the repository.
2. **Install Packages:** Installs required Python packages.
3. **Format Code:** Automatically formats Python scripts using Black.
4. **Train Model:** Runs the `train.py` script to train the model.
5. **Evaluate Model:** Generates a report with metrics and plots, posting results in GitHub comments using CML.

### CML Integration

CML (Continuous Machine Learning) is used to create dynamic reports that include model performance metrics and visualizations. 

## How to Use

1. **Update the Dataset:** Place your dataset (`Churn_Modelling.csv`) in the root directory of the project.
2. **Push Changes:** Commit and push changes to the `main` branch to trigger the GitHub Actions workflow.
3. **Review Results:** After the workflow completes, check the GitHub Actions tab for logs and results.

## Results

Upon successful execution of the workflow, you will find:
- **Model Metrics:** Accuracy and F1 Score.
- **Visualizations:** Confusion matrix and ROC curve plots.
- **Comments in PRs:** CML will post the metrics report in GitHub comments for easy review.

## 🤝Contributing

Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.

## 📜License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.