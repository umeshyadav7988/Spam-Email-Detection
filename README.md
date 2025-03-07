# Spam Email Detection

## Project Description
This project is a **Spam Email Detector** built using **Python, Flask, and Machine Learning**. It uses the **Naive Bayes classifier** with **TF-IDF vectorization** to detect whether a given email message is spam or not.

## Features
- Preprocesses email messages by removing punctuation and stopwords.
- Uses **TF-IDF Vectorization** for text representation.
- Trains a **Multinomial Naive Bayes** model for classification.
- Provides a **Flask-based API** to predict whether an email is spam or not.
- Includes a **frontend UI** for easy interaction.

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, NLTK, Pandas, NumPy
- **Frontend:** HTML, JavaScript (Fetch API)
- **Dataset:** SMS Spam Collection (Kaggle)

## Installation and Setup
### 1. Clone the Repository
```bash
git clone https://github.com/umeshyadav7988/Spam-Email-Detection.git
cd Spam-Email-Detection
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Activate virtual environment:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask App
```bash
python spam_email_detector.py
```

### 5. Access the Frontend
Open a browser and go to:
```
http://127.0.0.1:5000/
```

## API Endpoints
### 1. Predict Spam Email
**POST** `/predict`
- **Request Body:** `{ "email": "Your email text here" }`
- **Response:** `{ "prediction": "Spam" or "Not Spam" }`

### 2. Get Dataset
**GET** `/dataset`
- Returns the dataset used for training.

## Example Usage
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"email": "Congratulations! You won a free iPhone."}'
```

## Contact
For any queries, reach out to **Umesh Yadav** at [umeshyadav7988@gmail.com](mailto:umeshyadav7988@gmail.com).

## Repository Link
[GitHub: Spam Email Detection](https://github.com/umeshyadav7988/Spam-Email-Detection.git)

