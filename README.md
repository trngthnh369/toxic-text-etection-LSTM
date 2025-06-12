# 🧠 Toxic Text Detection

**Using LSTM, Django, and REST API**  
Version: 1.0.0  

---

## 📘 Overview
This project classifies user input text (comments) as **toxic** or **non-toxic** using a deep learning model based on **LSTM**. It includes a **Django** web application and provides **REST API** access.

- **Goal**: Detect and filter harmful content (e.g., harassment, hate speech, offensive language)
- **Core Technologies**:
  - LSTM (Long Short-Term Memory) neural networks
  - Django + Django REST Framework
  - Tokenization & Preprocessing (nltk, keras.preprocessing)
  - Dataset: Kaggle “Toxic Comment Classification Challenge”

---

## 🗂️ Project Structure

```
/
├── machine_learning/           # Model training and data processing
│   ├── data_preprocessing.ipynb
│   ├── train_model.py
│   └── toxic_lstm_model.h5     # Pretrained model
├── website/                    # Django application
│   ├── manage.py
│   ├── toxic_detection/        # Main Django app
│   │   ├── models.py
│   │   ├── views.py            # API endpoint
│   │   ├── serializers.py
│   │   └── urls.py
│   └── project/                # Global settings
└── requirements.txt
```

---

## ⚙️ Technologies & Libraries

- Python 3.8+
- Django 4.x, Django REST Framework
- Keras/TensorFlow (LSTM), numpy, pandas
- nltk (tokenization), scikit-learn (metrics, splitting)
- Gunicorn / uWSGI (for deployment)

---

## 🔧 Installation & Usage

### 1. Clone the project
```bash
git clone https://github.com/trngthnh369/toxic-text-etection-LSTM.git
cd toxic-text-etection-LSTM
```

### 2. Create virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare data & model
- If `toxic_lstm_model.h5` is missing, run `train_model.py` or use the provided notebook to retrain.
- Ensure tokenizer and input shape match the training phase.

### 4. Run Django server
```bash
cd website
python manage.py migrate
python manage.py runserver
```
- Access the web app at: `http://localhost:8000/`

### 5. Use REST API
- Endpoint: `POST /api/predict/`
- JSON Payload:
```json
{
  "text": "Your input text here"
}
```
- Response:
```json
{
  "text": "...",
  "prediction": "toxic",
  "confidence": 0.98
}
```

---

## 📊 Core Logic

1. User submits a comment via the web form or API.
2. Server performs:
   - Load tokenizer and convert text to sequences.
   - Pad the sequences to match model input shape.
   - Load the trained `.h5` model and run prediction.
   - Apply threshold (e.g., 0.5) → label as *toxic* or *non-toxic*.
3. Return result in JSON or render to template.

---

## 🧪 Performance & Evaluation

- Dataset: Kaggle Toxic Comment Classification Challenge
- LSTM model achieves ~96–98% accuracy/precision on test set
- Easily extensible to multi-label classification

---

## 🔄 Future Improvements

- Add multi-label support (insult, obscene, threat, etc.)
- Tune probability threshold, include more metrics: recall, F1
- Deploy on Heroku / AWS, add Swagger/OpenAPI docs
- Add frontend with React/VueJS and support live streaming input

---

## 🛠️ Common Issues

- Missing `toxic_lstm_model.h5`: retrain the model
- TensorFlow/Keras version mismatches: double-check `requirements.txt`
- Inconsistent tokenization may lead to shape errors

---

## ✅ Checklist Before Running

- [x] Create and activate virtual environment
- [x] Install all required packages
- [x] Ensure `.h5` model and tokenizer exist
- [x] Run Django migrations
- [x] Launch the server & test with sample API input

---

## 📞 Contact

For questions or contributions, feel free to open an issue or contact via email: **truongthinhnguyen30303@gmail.com**
