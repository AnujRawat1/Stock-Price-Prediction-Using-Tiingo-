# Stock Price Prediction Using Tiingo

An end-to-end machine learning project that predicts stock prices using historical data sourced from Tiingo. The model leverages deep learning techniques to forecast future stock prices.

## 📂 Project Structure

The project is organized into the following components:

- **Stock_Price_Model.ipynb**: Jupyter Notebook containing the data preprocessing, model training, and evaluation.
- **Stock Predictions Model.keras**: The trained Keras model file.
- **app.py**: Flask application for serving the model and making predictions.
- **CSV Files**: Historical stock data for various companies (e.g., AAPL.csv, GOOG.csv).

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Flask
- **Deployment**: Docker (for containerization)

## ⚙️ Features

- Predicts future stock prices based on historical data.
- Utilizes deep learning models for accurate forecasting.
- Provides a web interface for user interaction.

## 🧪 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AnujRawat1/Stock-Price-Prediction-Using-Tiingo-.git
cd Stock-Price-Prediction-Using-Tiingo-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Application
```bash
python app.py
```
### The application will be accessible at http://127.0.0.1:5000/.
