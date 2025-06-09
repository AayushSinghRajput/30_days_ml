import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained pipeline and encoders
pipeline = joblib.load('mental_health_model.pkl')
le_dict = joblib.load('label_encoders.pkl')  # Optional: Only if needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    form_data = {
        'Age': int(request.form['age']),
        'Gender': request.form['gender'],
        'self_employed': request.form['self_employed'],
        'family_history': request.form['family_history'],
        'work_interfere': request.form['work_interfere'],
        'no_employees': request.form['no_employees'],
        'remote_work': request.form['remote_work'],
        'tech_company': request.form['tech_company'],
        'benefits': request.form['benefits'],
        'care_options': request.form['care_options'],
        'wellness_program': request.form['wellness_program'],
        'seek_help': request.form['seek_help'],
        'anonymity': request.form['anonymity'],
        'leave': request.form['leave'],
        'mental_health_consequence': request.form['mental_health_consequence'],
        'phys_health_consequence': request.form['phys_health_consequence'],
        'coworkers': request.form['coworkers'],
        'supervisor': request.form['supervisor'],
        'mental_health_interview': request.form['mental_health_interview'],
        'phys_health_interview': request.form['phys_health_interview'],
        'mental_vs_physical': request.form['mental_vs_physical'],
        'obs_consequence': request.form['obs_consequence']
    }

    # Convert to dataframe with 1 row
    input_df = pd.DataFrame([form_data])

    # Apply same LabelEncoder used during training
    for col, le in le_dict.items():
        input_df[col] = le.transform(input_df[col])

    # Predict
    prediction = pipeline.predict(input_df)[0]

    # Decode target if needed
    result = "Will Seek Treatment" if prediction == 1 else "Will Not Seek Treatment"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
