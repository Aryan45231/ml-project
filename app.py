
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

# Route For Home Page 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        input_data = request.get_json()
        
        # Validate JSON data exists
        if not input_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract features
        gender = input_data.get('gender')
        race = input_data.get('race')
        parental_education = input_data.get('parental_education')
        lunch = input_data.get('lunch')
        test_preparation_course = input_data.get('test_preparation_course')
        reading_score = input_data.get('reading_score')
        writing_score = input_data.get('writing_score')
        
        # Validate all required fields
        if not all([gender, race, parental_education, lunch, test_preparation_course, 
                   reading_score is not None, writing_score is not None]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Convert scores to float
        try:
            reading_score = float(reading_score)
            writing_score = float(writing_score)
        except ValueError:
            return jsonify({'error': 'Reading and writing scores must be numbers'}), 400
        
        # Create CustomData instance
        data = CustomData(
            gender=gender,
            race=race,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        # Get data as dataframe
        pred_df = data.get_data_as_data_frame()
        
        # Make prediction
        predictor = PredictPipeline()
        results = predictor.predict(pred_df)
        
        return jsonify({
            'predicted_math_score': float(results[0]),
            'reading_score': reading_score,
            'writing_score': writing_score
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(
        debug=True,           # Enable debug mode
        host='127.0.0.1',     # Only localhost can access (default)
        port=3000             # Port number (default)
    )
