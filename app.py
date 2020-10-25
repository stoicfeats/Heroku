import numpy as np 
from flask import Flask, request, render_template
import pickle 

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1 :
        return render_template('index.html', prediction_text='Passenger will survive')
    else :
        return render_template('index.html', prediction_text='Passenger will not survive, RIP')

if __name__ == "__main__":
    app.run(debug=True)