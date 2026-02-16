# Rainfall Prediction System

## Project Overview

This is a **Machine Learning-powered Rainfall Prediction System** that predicts whether it will rain tomorrow based on various weather parameters. The application uses a trained model to analyze weather data and provides accurate predictions with a beautiful, modern user interface.

---

## Features

- **Weather Parameter Input**: Users can input various weather parameters including:
  - Location (Australian cities)
  - Min/Max Temperature
  - Rainfall amount
  - Humidity levels (9am and 3pm)
  
- **AI-Powered Prediction**: Uses a trained machine learning model to predict rainfall probability

- **Beautiful UI/UX**:
  - Modern glassmorphism design
  - Animated rain drops on the home page
  - Dynamic result pages with weather-appropriate effects
  - Responsive design for all devices

- **Visual Feedback**:
  - Rain prediction: Shows rain image with animated rain effect and lightning
  - No Rain prediction: Shows sunny image with floating clouds and sun rays

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Machine Learning | Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Styling | Custom CSS with animations |

---

## Project Structure

```
Rainfall_Prediction/
├── main.py                    # Flask application (main server)
├── Weather-Dataset.csv        # Training dataset
├── Rainfall_prediction.ipynb  # Jupyter notebook for model training
├── models/
│   ├── rainfall.pkl          # Trained ML model
│   ├── scale.pkl             # Feature scaler
│   ├── encoder.pkl           # Label encoder
│   └── impter.pkl            # Data imputer
├── templates/
│   ├── index.html            # Home page with input form
│   ├── change.html           # Rain prediction result page
│   └── nochange.html         # No rain prediction result page
├── static/
│   ├── css/
│   │   ├── index.css         # Home page styles
│   │   ├── change.css        # Rain result page styles
│   │   └── nochange.css      # No rain result page styles
│   └── images/
│       ├── rain.png          # Rain prediction image
│       └── norain.png        # No rain prediction image
├── images/
│   ├── rain.png              # Source rain image
│   └── norain.png            # Source no rain image
└── documentation/
    ├── README.md             # This file
    ├── PROJECT_REPORT.md     # Detailed project report
    └── screenshots/          # Application screenshots
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Rainfall_Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install flask pandas numpy scikit-learn
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Access the application**
   Open your browser and navigate to: `http://localhost:5000`

---

## How to Use

1. **Open the Application**: Navigate to `http://localhost:5000` in your web browser

2. **Enter Weather Data**:
   - Select a location from the dropdown
   - Enter minimum and maximum temperature
   - Enter current rainfall amount
   - Optionally enter humidity values

3. **Get Prediction**: Click the "Predict Rainfall" button

4. **View Results**:
   - If rain is predicted: You'll see the rain image with animated rain effects
   - If no rain is predicted: You'll see the sunny image with cloud animations

5. **Go Back**: Click anywhere on the result page or use the "Go Back" button to make another prediction

---

## Model Information

- **Algorithm**: Random Forest Classifier
- **Dataset**: Australian Weather Dataset (145,460 records)
- **Features**: 110 features after one-hot encoding
- **Accuracy**: ~85% on test data

### Input Features
- Numeric: MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Temp9am, Temp3pm, Year, Month, Day
- Categorical: Location, WindGustDir, WindDir9am, WindDir3pm, RainToday

---

## Author

**Project developed for Machine Learning course submission**

Date: February 2026

---

## License

This project is for educational purposes only.
