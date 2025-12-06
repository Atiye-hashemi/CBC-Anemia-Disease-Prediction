import os
import joblib
import shap
from openai import OpenAI
import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Load the full pipeline model 
model = joblib.load('xgb_model3.joblib')  

# Initialize OpenAI client for DeepSeek API

load_dotenv()
api_key = os.environ.get("API_KEY")
client = OpenAI(
    api_key=api_key,  
    base_url="https://openrouter.ai/api/v1"
)

def extract_cbc_data_from_pdf(pdf_path):
    import pdfplumber

    features = {
        "Hemoglobin": None, "Hematocrit": None, "MCV": None,
        "MCH": None, "MCHC": None, "RBC": None, "WBC": None,
        "Platelets": None, "Serum Iron": None, "Ferritin": None,
        "Vitamin B12": None, "Folate": None
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    # Convert row to lower case strings for easier matching
                    row_lower = [str(cell).lower() if cell else '' for cell in row]

                    for key in features:
                        key_lower = key.lower()
                        # If key appears anywhere in the row
                        if any(key_lower in cell for cell in row_lower):
                            # Find the index of the key cell
                            try:
                                key_idx = next(i for i, cell in enumerate(row_lower) if key_lower in cell)
                                # Assuming the next column after key cell contains the result value
                                # Sometimes it's key_idx + 1, sometimes key_idx + 2, so check which one is numeric
                                possible_indices = [key_idx + 1, key_idx + 2]
                                value = None
                                for idx in possible_indices:
                                    if idx < len(row):
                                        try:
                                            value = float(row[idx])
                                            break
                                        except:
                                            continue
                                if value is not None:
                                    features[key] = value
                            except StopIteration:
                                pass
    return features

def explain_with_shap(input_df, background_df):


    # Preprocess input and background
    preprocessor = model.named_steps["preprocessor"]
    model_step = model.named_steps["model"]

    background_processed = preprocessor.transform(background_df)
    input_processed = preprocessor.transform(input_df)

    # Get processed feature names
    try:
        feature_names = preprocessor.get_feature_names_out()

        feature_names = [name.split("_", 1)[1] if "_" in name else name for name in feature_names]
    except AttributeError:
        # fallback if get_feature_names_out is not available
        feature_names = input_df.columns.tolist()

    # Create TreeExplainer with background dataset
    explainer = shap.TreeExplainer(model_step, data=background_processed)

    # Compute SHAP values
    shap_values_all = explainer.shap_values(input_processed)  # returns list [class0, class1] for binary
    # Select class 0 (No Anemia)
    shap_values = shap_values_all[0] if isinstance(shap_values_all, list) else shap_values_all

    # Identify top 2 features by absolute SHAP value
    shap_array = shap_values[0]  # single row
    top_indices = np.argsort(-np.abs(shap_array))[:2]
    top_features = [feature_names[i] for i in top_indices]

    # Plot summary
    shap.summary_plot(
        shap_values, 
        input_processed, 
        feature_names=feature_names, 
        show=False,
        plot_size=(10,6)
    )
    shap_plot_path = os.path.join(app.config['STATIC_FOLDER'], 'shap_summary_plot.png')
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.clf()

    shap_text = f"Top influencing features: {top_features[0]}, {top_features[1]}"

    return shap_text, 'shap_summary_plot.png', top_features



def get_ai_explanation(input_df, prediction, top_features):
    disease_name = "Anemia" if prediction == 1 else "No Anemia"
    feature_names = input_df.columns.tolist()
    inputs = input_df.iloc[0].tolist()

    system_prompt = """
    You are a medical explanation assistant.

    Rules you MUST follow:
    - Output exactly 4_5 bullet points.
    - Each bullet must be one short sentence.
    - No paragraphs, no introductions, no extra text before or after the bullets.
    - Use a dash "-" and a space at the start of each bullet.
    - If you cannot comply, output "ERROR".
    """

    user_prompt = f"""
    Patient CBC values:
    {dict(zip(feature_names, inputs))}

    Predicted disease: {disease_name}
    Top contributing features: {', '.join(top_features)}

    Follow this exact format (example):

    - Anemia predicted due to low hemoglobin and microcytosis.
    - Low hemoglobin reduces oxygen transport efficiency.
    - Low MCV suggests microcytic red blood cells.
    - Recommend ferritin test to confirm iron deficiency.
    - Consider hemoglobin electrophoresis to rule out thalassemia.

    Now produce your response in exactly the same format for the given data(each line has one sentence).    """


    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pdf_file = request.files['pdf_file']
    age = int(request.form['age'])
    gender = request.form['gender']

    filename = secure_filename(pdf_file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(pdf_path)

    # Extract CBC data
    cbc_data = extract_cbc_data_from_pdf(pdf_path)
    # Normalize keys: replace spaces with underscores
    cbc_data = {key.replace(" ", "_"): value for key, value in cbc_data.items()}

    cbc_data['Age'] = age
    cbc_data['Gender'] = gender

    
    input_df = pd.DataFrame([cbc_data])

    # Predict
    prediction = model.predict(input_df)[0]

    # SHAP explanation and top features
    shap_text, shap_plot_filename, top_features = explain_with_shap(input_df , model.background_data)

    # DeepSeek AI explanation
    ai_explanation = get_ai_explanation(input_df, prediction, top_features)

    result = {
        "prediction": "Anemia" if prediction == 1 else "No Anemia",
        "shap_text": shap_text,
        "shap_plot_path": shap_plot_filename,
        "deepseek_explanation": ai_explanation
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
