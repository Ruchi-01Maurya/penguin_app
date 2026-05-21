
## Tech Stack
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Python, Flask
- **ML**: scikit-learn (Random Forest Classifier)
- **Dataset**: Palmer Penguins

## Project Structure
```
penguin_app/
├── app.py              # Flask web server & prediction API
├── train_model.py      # Model training script
├── model.pkl           # Trained Random Forest model
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend (HTML/CSS/JS)
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (already done — model.pkl included)
```bash
python train_model.py
```

### 3. Start the Flask app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Input Features
| Feature          | Range        | Unit |
|-----------------|-------------|------|
| Bill Length      | 32 – 60     | mm   |
| Bill Depth       | 13 – 22     | mm   |
| Flipper Length   | 170 – 235   | mm   |
| Body Mass        | 2700 – 6300 | g    |

## Model Performance
- Algorithm: Random Forest (100 trees)
- Test Accuracy: ~98.6%
- Dataset: Palmer Penguins (344 samples, 3 species)

## API Endpoint
`POST /predict`
```json
{
  "bill_length": 43.9,
  "bill_depth": 17.2,
  "flipper_length": 200,
  "body_mass": 4050
}
```
Response:
```json
{
  "success": true,
  "species": "Adelie",
  "probabilities": {"Adelie": 95.0, "Chinstrap": 3.0, "Gentoo": 2.0},
  "info": { ... }
}
```