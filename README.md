# Personalized Learning Recommendation System

An ML-based system that recommends suitable learning resources based on students' academic performance, interests, and learning preferences.

The system uses a **Random Forest Classification Model** to evaluate the suitability of learning resources and generate personalized recommendations through an interactive Streamlit application.

## Problem Statement

Traditional learning approaches often provide the same educational resources to all students without considering their individual academic performance, interests, learning preferences, or previously completed topics.

This can make it difficult for students to identify learning materials that match their individual needs.

The proposed system addresses this problem by using Machine Learning to recommend suitable learning resources based on individual student profiles.

## Objectives

* To develop a personalized learning recommendation system using Machine Learning.
* To analyze students' academic performance and learning interests.
* To consider individual learning preferences and completed topics.
* To implement a Random Forest classification model for evaluating resource suitability.
* To generate five suitable learning resource recommendations.
* To provide an interactive and user-friendly Streamlit interface.

## Key Features

* **Personalized Recommendations:** Suggests learning resources based on individual student profiles.
* **Academic Performance Analysis:** Considers students' recent academic performance.
* **Interest-Based Learning:** Uses students' interests to improve resource relevance.
* **Learning Preference Support:** Considers individual learning preferences.
* **Completed Topic Tracking:** Takes previously completed topics into account.
* **Random Forest Classification:** Evaluates the suitability of learning resources.
* **Five Recommendations:** Generates five recommended learning resources.
* **Interactive Dashboard:** Provides a user-friendly interface built with Streamlit.
* **Model Evaluation:** Reports accuracy, precision, recall, and F1-score.

## Technologies Used

| Technology    | Purpose                                           |
| ------------- | ------------------------------------------------- |
| Python        | Core programming language                         |
| Pandas        | Data manipulation and analysis                    |
| NumPy         | Numerical operations                              |
| Scikit-learn  | Machine Learning model development and evaluation |
| Random Forest | Resource suitability classification               |
| Streamlit     | Interactive web application                       |
| Joblib        | Saving and loading the trained model              |
| CSV           | Dataset and learning resource storage             |

## Machine Learning Algorithm

The system uses the **Random Forest Classifier**, an ensemble learning algorithm that combines multiple decision trees to perform classification.

The model evaluates the suitability of learning resources using student-related features, including academic performance, interests, learning preferences, and completed topics.

The trained model is saved using Joblib and loaded by the application to support personalized recommendations.

## Model Performance

The Random Forest model was evaluated using 500 test records.

| Evaluation Metric |  Score |
| ----------------- | -----: |
| Accuracy          | 86.80% |
| Precision         | 94.06% |
| Recall            | 90.62% |
| F1-Score          | 92.31% |

These evaluation metrics measure the model's classification performance.

## Project Structure

```text
Personalized-Learning-Recommendation-System/
│
├── dataset/
├── .gitignore
├── README.md
├── app.py
├── learning_resources.csv
├── recommender.py
├── requirements.txt
└── suitability_model.joblib
```

## Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/karmakarpriyasmita05-commits/Personalized-Learning-Recommendation-System.git
```

### 2. Navigate to the Project Folder

```bash
cd Personalized-Learning-Recommendation-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## How the System Works

1. The user enters their learning profile and academic information.
2. The system processes the provided information.
3. The trained Random Forest model evaluates learning resource suitability.
4. The application generates five personalized learning recommendations.
5. The user can view the recommended resources through the Streamlit dashboard.

## Future Improvements

* Expand the learning resource dataset.
* Support additional learning preferences.
* Improve recommendation relevance.
* Add more learning resources across different subjects.
* Enhance the dashboard with additional features.

## Author

**Priyasmita Karmakar**

## License

This project is available for educational and learning purposes.
