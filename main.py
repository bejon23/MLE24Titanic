from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = FastAPI()

# Get the absolute path of the current file.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the templates directory.
templates_dir = os.path.join(current_dir, "templates")

# Initialize Jinja2Templates with the correct directory
templates = Jinja2Templates(directory=templates_dir)

# Construct the absolute path to the stacking_clf.pkl file
model_path = os.path.join(current_dir, 'MLE24Titanic', 'stacking_clf.pkl')

# Load the trained model
with open(model_path, 'rb') as f:
    stacking_clf = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, 
                  pclass: int = Form(...), 
                  sex_female: int = Form(...), 
                  sex_male: int = Form(...), 
                  age: int = Form(...), 
                  sibsp: int = Form(...),
                  parch: int = Form(...), 
                  fare: int = Form(...), 
                  embarked_c: int = Form(...), 
                  embarked_q: int = Form(...), 
                  embarked_s: int = Form(...)):
    
    # Make prediction
    prediction = stacking_clf.predict([[pclass, sex_female, sex_male, age, sibsp, parch, fare, embarked_c, embarked_q, embarked_s]])
    result = "likely" if prediction == 1 else "unlikely"
    
    return templates.TemplateResponse("results.html", {"request": request, "prediction": result})
