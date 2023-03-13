#import relevant libraries for flask,html rendering and loading the ML model
from flask import Flask,request,url_for,render_template
import pickle
import pandas as pd
import joblib
app = Flask(__name__)
# model=pickle.load(open("model.pkl","rb"))
model=joblib.load(open("model.pkl","rb"))
# scale=pickle.load(open("scale.pkl","rb"))
scale=joblib.load(open("scale.pk1","rb"))


@app.route("/")
def landingPage():
    return render_template("index.html") 

@app.route("/predict",methods=["POST"])
def predict():
    lost_reason = request.form['1']
    budget = request.form['2']
    lease = request.form['3']
    utm_source = request.form['4']
    utm_medium = request.form['5']
    des_city = request.form['6']
    des_country = request.form['7']
    rowDf=pd.DataFrame([pd.Series([lost_reason,budget,lease,utm_source,utm_medium,des_city,des_country])])
    rowDf_new=pd.DataFrame(scale.transform(rowDf))
    
    print(rowDf_new)

#  model prediction 
    prediction= model.predict_proba(rowDf_new)
    print(f"The  Predicted values is :{prediction[0][1]}")

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round val {valPred*100}%")
        return render_template('result.html',pred=f'The probability of won the lead is {valPred*100}%.')
    else:
        valPred = round(prediction[0][0],3)
        return render_template('result.html',pred=f'Probability of loss the lead is {valPred*100}%.')
if __name__ == "__main__":
    app.run(debug=True)