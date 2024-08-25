from flask import Flask, render_template, request
import pickle
import numpy as np 

model = pickle.load(open('proj.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        d1 = request.form['a']
        d2 = request.form['b']
        d3 = request.form['c']
        d4 = request.form['d']
        d5 = request.form['f']  # Corrected variable name
        d6 = request.form['e']  # Corrected variable name
        d7 = request.form['g']  # Corrected variable name
        arr = np.array([[d1, d2, d3, d4, d5, d6, d7]])  # Updated array
        
        arr_numeric = arr.astype(np.float64)  # Convert string values to float64

        pred = model.predict(arr_numeric)  # Corrected function name
        return render_template('crop.html', data=pred[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
