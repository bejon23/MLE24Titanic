from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import pdb

app = FastAPI()

# Load the trained model
with open('best_rf_clf.pkl', 'rb') as file:
    model = pickle.load(file)

# Assuming main.py is in the same directory as the templates and static directories
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request,
                  pclass: int = Form(...),
                  sex: str = Form(...),
                  age: int = Form(...),
                  sibsp: int = Form(...),
                  parch: int = Form(...),
                  fare: int = Form(...),
                  embarked: str = Form(...),
                  survived: int = Form(...),
                  familysize: int = Form(...)):

    # Label encoding for 'sex' and 'embarked'
    sex_encoder = LabelEncoder()
    embarked_encoder = LabelEncoder()
    sex_encoded = sex_encoder.fit_transform([sex])
    embarked_encoded = embarked_encoder.fit_transform([embarked])

    features = [pclass, sex_encoded[0], age, sibsp, parch, fare, embarked_encoded[0], survived, familysize]  # Update features list

    # Make prediction
    prediction = model.predict([features])[0]
    result = "likely" if prediction == 1 else "unlikely"

    return templates.TemplateResponse("results.html", {"request": request, "prediction": result})

# Mounting the static files directory
@app.get("/static/{filename}")
async def get_static_file(filename: str):
    return FileResponse(os.path.join(static_dir, filename), media_type="text/css")
