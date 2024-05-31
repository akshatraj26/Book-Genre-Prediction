# Book Genre Prediction

This project aims to predict the genre of books based on their descriptions using machine learning models. Two models were employed in this project: Naive Bayes and XGBoost. Among these, XGBoost provided better accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
  - [Naive Bayes](#naive-bayes)
  - [XGBoost](#xgboost)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Models](#training-the-models)
  - [Running the API](#running-the-api)
  - [API Endpoints](#api-endpoints)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to classify books into different genres based on their descriptions. Accurate genre classification can help in better cataloging and recommendations for readers.

## Dataset

The dataset used for this project includes book descriptions and their corresponding genres. Ensure that your dataset is in a CSV format with at least two columns: `description` and `genre`.

## Models Used

### Naive Bayes

Naive Bayes is a simple yet effective classification algorithm based on Bayes' theorem. It assumes independence between features.

### XGBoost

XGBoost stands for eXtreme Gradient Boosting. It is an advanced implementation of gradient boosting that is designed for speed and performance.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- fastapi
- uvicorn

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn
```

## Usage

### Training the Models

1. Clone the repository:

```bash
git clone https://github.com/yourusername/book-genre-prediction.git
cd book-genre-prediction
```

2. Prepare your dataset and place it in the project directory.



The script will train both Naive Bayes and XGBoost models and save the trained models to disk.

### Running the API

1. Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

2. The API will be accessible at `http://127.0.0.1:8000/docs`.

### API Endpoints

- **`POST /predict`**: Predict the genre of a book based on its description.

#### Request

- **Body**: JSON object containing the book description.

```json
{
  "description": "A young wizard's journey and adventures in a magical school."
}
```

#### Response

- **Body**: JSON object containing the predicted genre.

```json
{
  "genre": "Fantasy"
}
```

## Results

The XGBoost model outperformed the Naive Bayes model in terms of accuracy. Below are the results:

- **Naive Bayes Accuracy**: *accuracy_value*
- **XGBoost Accuracy**: *accuracy_value*

Replace `*accuracy_value*` with the actual accuracy values obtained from your training.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

