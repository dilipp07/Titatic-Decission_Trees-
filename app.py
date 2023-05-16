from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import os
import sys

application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    try:
        return render_template('index.html')
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict',methods=['GET','POST'])


def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:

        data=CustomData(
            Sex=(request.form.get('Sex')),
            Embarked =(request.form.get('Embarked')),
            Pclass = float(request.form.get('Pclass')),
            SibSp = float(request.form.get('SibSp')),
            
            Parch= float(request.form.get('Parch')))


        final_new_data=data.get_data_as_dataframe()
        print(final_new_data)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('result.html',final_result=results)





if __name__=="__main__":
    app.run(host='0.0.0.0',port=5001,debug=True)