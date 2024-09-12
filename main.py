import uvicorn
from fastapi import FastAPI
import joblib # type: ignore
from Diabetics import Diabetics
import sklearn

app = FastAPI()

# Load the model
try:
    model = joblib.load("random_forest_model_v1_new.pkl")
except Exception as e:
    raise ValueError(f"Failed to load the model: {e}")

@app.get('/')
def index():
    return {'message': 'Diabetics Prediction ML API'}

@app.post('/Diabetics/predict')
def predict_Diab_type(data: Diabetics):
    data = data.dict()
    Pregnancies = data['Pregnancies']
    Glucose = data['Glucose']
    BloodPressure = data['BloodPressure']
    SkinThickness = data['SkinThickness']
    Insulin = data['Insulin']
    BMI = data['BMI']
    DiabetesPedigreeFunction = data['DiabetesPedigreeFunction']
    Age = data['Age']

    # Ensure the features are passed in the correct order expected by the model
    prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    return {
        'prediction': prediction[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
