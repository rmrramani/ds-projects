#import required libraries
from flask import Flask,render_template,session,url_for,redirect
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
import numpy as np
import pickle

##input to output
def return_prediction(model,price):

    first_price=price['firstday']
    second_price=price['secondday']
    third_price=price['thirdday']
    fourth_price=price['fourthday']
    fifth_price=price['fifthday']

    price_list=[first_price,second_price,third_price,fourth_price,fifth_price]

    array=np.array(price_list).reshape(-1,5)
    predict=model.predict(array)
    return round(predict[0],0)

app=Flask(__name__)
app.config['SECRET_KEY']='secretkey'

class PriceForm(FlaskForm):

    firstday_price=TextAreaField("First Day Price")
    secondday_price=TextAreaField("Second Day Price")
    thirdday_price=TextAreaField("Third Day Price")
    fourthday_price=TextAreaField("Fourth Day Price")
    fifthday_price=TextAreaField("Fifth Day Price")

    submit=SubmitField("Analyze")

@app.route("/",methods=['GET','POST'])
def index():

    form=PriceForm()

    if form.validate_on_submit():

        session['firstday_price']=form.firstday_price.data
        session['secondday_price']=form.secondday_price.data
        session['thirdday_price']=form.thirdday_price.data
        session['fourthday_price']=form.fourthday_price.data
        session['fifthday_price']=form.fifthday_price.data

        return redirect(url_for("prediction"))

    return render_template('home.html',form=form)

price_model=pickle.load(open('tsreg.pkl','rb'))

@app.route('/prediction')
def prediction():
    content={}

    content['firstday']=float(session['firstday_price'])
    content['secondday']=float(session['secondday_price'])
    content['thirdday']=float(session['thirdday_price'])
    content['fourthday']=float(session['fourthday_price'])
    content['fifthday']=float(session['fifthday_price'])
    
    results=return_prediction(model=price_model,price=content)
    return render_template('prediction.html',results=results)

if __name__=="__main__":
    app.run(port=8000,host="0.0.0.0")