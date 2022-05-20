import mlflow
from sklearn.preprocessing import StandardScaler
import joblib,pickle,json
import pandas as pd
from utils import read_config
from predict import *
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse



config_dict = read_config("app_config.yaml")
model_path = config_dict["config"]["model_path"]
scalar_path = config_dict["config"]["scalar_path"]
cat_dict_path = config_dict["config"]["cat_dict_path"]

model = mlflow.sklearn.load_model(model_path)

#loading cat cols
with open(cat_dict_path , 'rb') as handle:
    cat_dict = pickle.load(handle)
#loading std scalar
with open(scalar_path,'rb') as handle:
    std_scalar = joblib.load(handle)

label_map = {0:'Fully Paid',1:'Charged Off'}

app = FastAPI()

@app.get("/")
def read_root():
    return {"greetings": "testing server"}

@app.post("/predict")
async def predict_data(data_json):
    data = json.loads(data_json)
    df = pd.DataFrame.from_dict(data,orient='index').T

    x = feature_engineer(df,cat_dict,std_scalar)

    out = model.predict(x)[0]
    prob = model.predict_proba(x)[0][1]
    prob = round(prob,2)
    predicted_label = label_map[out]
    
    print(out,predicted_label,prob)
    return {"predicted_class":str(out),
            "predicted_label":str(predicted_label),
            "predicted_probability":str(prob)}



