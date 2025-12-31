# Hospital Treatment Duration Prediction Model

## Overview
This project contains a machine learning model that predicts the duration category of hospital antibiotic treatment based on patient and prescription characteristics. The model classifies treatment duration into three categories: `<5 days`, `5-10 days`, and `>10 days`.

## Model Details

### Model Type
**Decision Tree Classifier** (Scikit-learn)

### Model File
`best_dt_classifier_model.joblib`

### Hyperparameters (Optimized via GridSearchCV)
The model was tuned using GridSearchCV with 5-fold cross-validation and weighted F1-score as the optimization metric.

**Best Parameters:**
- **criterion**: `entropy` (information gain)
- **max_depth**: `None` (nodes are expanded until all leaves are pure or contain less than min_samples_split samples)
- **min_samples_split**: `2` (minimum number of samples required to split an internal node)
- **min_samples_leaf**: `1` (minimum number of samples required to be at a leaf node)
- **random_state**: `42` (for reproducibility)
- **splitter**: `best` (best split at each node)

### Model Performance
- **Initial Model Accuracy**: 77.84%
- **After Hyperparameter Tuning**: Optimized using weighted F1-score
- **Evaluation Metrics**: Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted)

## Dataset Information

### Source
`Hopsital Dataset.csv` - Hospital antibiotic prescription records

### Original Columns
- Age
- Date of Data Entry
- Gender
- Diagnosis
- Name of Drug
- Dosage (gram)
- Route (administration route)
- Frequency (dosing frequency)
- Duration (days)
- Indication

### Data Preprocessing
1. **Cleaning**: Removed rows with placeholder values in Gender, Route, Frequency, and Duration columns
2. **Target Creation**: Converted 'Duration (days)' into categorical bins
3. **Numeric Conversion**:
   - Age converted to numeric, missing values filled with median
   - Dosage (gram) converted to numeric, missing values filled with median
4. **One-Hot Encoding**: Categorical features encoded as binary variables
5. **Train-Test Split**: 80% training, 20% testing with stratification

## Model Features

### Input Features (8 total)

#### Numeric Features
1. **Age** (float)
   - Patient age in years
   - Missing values imputed with median

2. **Dosage (gram)** (float)
   - Medication dosage in grams
   - Missing values imputed with median

#### Binary Features (One-Hot Encoded)

3. **Gender_Male** (int: 0 or 1)
   - 1 = Male
   - 0 = Female

4. **Route_IV** (int: 0 or 1)
   - 1 = Intravenous administration
   - 0 = Other route

5. **Route_Oral** (int: 0 or 1)
   - 1 = Oral administration
   - 0 = Other route

6. **Frequency_OD** (int: 0 or 1)
   - 1 = Once Daily dosing
   - 0 = Other frequency

7. **Frequency_QID** (int: 0 or 1)
   - 1 = Four times daily (Quarter In Die)
   - 0 = Other frequency

8. **Frequency_TDS** (int: 0 or 1)
   - 1 = Three times daily (Ter Die Sumendum)
   - 0 = Other frequency

**Note**: Binary features are mutually exclusive within their category. For example:
- If Route_IV = 0 and Route_Oral = 0, the route is "BD" (Twice Daily) or another unlisted route
- If Frequency_OD = 0, Frequency_QID = 0, Frequency_TDS = 0, the frequency is "BD" (Twice Daily)

### Output (Target Variable)

**Duration_Category** (string) - Three possible classes:
- `<5 days` - Treatment duration less than 5 days
- `5-10 days` - Treatment duration between 5 and 10 days (exclusive upper bound)
- `>10 days` - Treatment duration 10 days or more

## Usage

### Loading the Model

```python
import joblib

# Load the trained model
model = joblib.load('best_dt_classifier_model.joblib')
```

### Making Predictions

```python
import pandas as pd

# Example: Prepare input data
input_data = {
    'Age': 60.0,
    'Dosage (gram)': 0.5,
    'Gender_Male': 1,        # Male patient
    'Route_IV': 0,           # Not IV
    'Route_Oral': 1,         # Oral route
    'Frequency_OD': 1,       # Once daily
    'Frequency_QID': 0,      # Not QID
    'Frequency_TDS': 0       # Not TDS
}

# Convert to DataFrame (required format)
input_df = pd.DataFrame([input_data])

# Make prediction
prediction = model.predict(input_df)
print(f"Predicted Duration Category: {prediction[0]}")
# Output: Predicted Duration Category: <5 days
```

### Input Data Validation

Ensure your input data meets these requirements:
1. All 8 features must be present
2. Features must be in the exact order shown above
3. Numeric features (Age, Dosage) must be non-negative numbers
4. Binary features must be 0 or 1 (integers)
5. Only one route should be selected (Route_IV or Route_Oral)
6. Only one frequency should be selected (or none if frequency is BD)

## API Integration

### Flask API Example

The project includes Flask API code for serving predictions via HTTP POST requests.

#### Endpoint
`POST /predict`

#### Request Format
```json
{
    "Age": 60,
    "Dosage (gram)": 0.5,
    "Gender_Male": 1,
    "Route_IV": 0,
    "Route_Oral": 1,
    "Frequency_OD": 1,
    "Frequency_QID": 0,
    "Frequency_TDS": 0
}
```

#### Response Format
```json
{
    "prediction": "<5 days"
}
```

#### Error Responses
- **400 Bad Request**: Missing required fields or invalid JSON
- **500 Internal Server Error**: Model prediction failure

### Running the API Locally

1. Save the Flask application code to `app.py`
2. Ensure the model file is in the same directory
3. Set environment variable:
   ```bash
   # Linux/Mac
   export FLASK_APP=app.py

   # Windows
   set FLASK_APP=app.py
   ```
4. Run the server:
   ```bash
   flask run
   ```
5. The API will be available at `http://127.0.0.1:5000/`

### Testing the API

**Using curl:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "Age": 60,
    "Dosage (gram)": 0.5,
    "Gender_Male": 1,
    "Route_IV": 0,
    "Route_Oral": 1,
    "Frequency_OD": 1,
    "Frequency_QID": 0,
    "Frequency_TDS": 0
  }' \
  http://127.0.0.1:5000/predict
```

**Using Python requests:**
```python
import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "Age": 60,
    "Dosage (gram)": 0.5,
    "Gender_Male": 1,
    "Route_IV": 0,
    "Route_Oral": 1,
    "Frequency_OD": 1,
    "Frequency_QID": 0,
    "Frequency_TDS": 0
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

## Feature Engineering Details

### Categorical Encoding Strategy
The model uses one-hot encoding with `drop_first=True` for categorical variables to avoid multicollinearity:

**Gender** (original values: Male, Female)
- Encoded as: `Gender_Male` (1 for Male, 0 for Female)

**Route** (original values: IV, Oral, BD)
- Encoded as: `Route_IV`, `Route_Oral`
- BD (Twice Daily) represented when both are 0

**Frequency** (original values: OD, BD, TDS, QID)
- Encoded as: `Frequency_OD`, `Frequency_QID`, `Frequency_TDS`
- BD (Twice Daily) represented when all three are 0

### Target Variable Creation
Duration categories created using `pd.cut()` with the following bins:
- `<5 days`: [0, 5)
- `5-10 days`: [5, 10)
- `>10 days`: [10, max]

## Dependencies

```
pandas
scikit-learn (v1.6.1 or compatible)
joblib
flask (for API)
seaborn (for visualization during training)
matplotlib (for visualization during training)
```

## Model Training Information

### Training Process
1. Data loaded from CSV and cleaned
2. Features engineered (numeric conversion, one-hot encoding)
3. Train-test split (80-20) with stratification
4. Hyperparameter tuning via GridSearchCV:
   - 5-fold cross-validation
   - Parameter grid search over max_depth, min_samples_split, min_samples_leaf, criterion
   - Scoring metric: weighted F1-score
   - Parallel processing enabled (n_jobs=-1)
5. Best model selected and saved

### GridSearchCV Parameter Grid
```python
{
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
```

## Deployment Considerations

### For Production Use
1. **Containerization**: Use Docker for consistent environments
2. **Web Server**: Replace Flask development server with Gunicorn or uWSGI
3. **Security**:
   - Implement API authentication (API keys, OAuth)
   - Add rate limiting
   - Validate and sanitize all inputs
4. **Monitoring**:
   - Log all predictions and errors
   - Monitor model performance metrics
   - Set up alerts for anomalies
5. **Scalability**:
   - Consider cloud deployment (AWS, GCP, Azure)
   - Use load balancers for high traffic
   - Cache frequent predictions if applicable
6. **Model Versioning**:
   - Track model versions
   - Implement A/B testing for model updates
   - Maintain rollback capabilities

## Files in This Project

- `best_dt_classifier_model.joblib` - Trained and optimized Decision Tree model
- `decision_tree_classifier.py` - Complete training pipeline and API code
- `Hopsital Dataset.csv` - Original training dataset
- `README.md` - This documentation file

## Limitations and Considerations

1. **Model Scope**: Predictions are based on limited features and may not capture all clinical factors affecting treatment duration
2. **Data Quality**: Model performance depends on the quality and representativeness of training data
3. **Class Imbalance**: Check training data for class imbalance which may affect predictions
4. **Clinical Validation**: Model outputs should be validated by healthcare professionals before clinical use
5. **Scikit-learn Version**: Model was trained with scikit-learn 1.6.1; version compatibility warnings may appear with different versions

## Future Improvements

1. Feature expansion (e.g., diagnosis, drug name, patient comorbidities)
2. Ensemble methods (Random Forest, Gradient Boosting)
3. Feature importance analysis
4. Cross-validation with larger datasets
5. Integration with hospital information systems
6. Real-time model retraining pipeline

## Contact and Support

For questions about model implementation or integration into applications, refer to the original training code in `decision_tree_classifier.py` for detailed preprocessing and prediction logic.

---

**Model Version**: 1.0
**Last Updated**: 2024
**Training Framework**: Scikit-learn
**Model Type**: Classification (Multi-class)
