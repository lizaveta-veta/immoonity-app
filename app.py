from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Загружаем модели
with open('earth_model.pkl', 'rb') as f:
    earth_model = pickle.load(f)
with open('agg_model_v2.pkl', 'rb') as f:
    agg_model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    for key in data:
        if key != 'sex': # Пол оставляем текстом ('M' или 'F')
            data[key] = float(data[key])
    results = []

    # 1. ЗЕМНАЯ МОДЕЛЬ
    if all(k in data for k in ['sex', 'vo2max', 'pullups', 'pushups']):
        is_male = (data['sex'] == 'M')
        vo2 = 1 if (is_male and data['vo2max'] > 43) or (not is_male and data['vo2max'] > 36) else 0
        pull = 1 if (is_male and data['pullups'] > 10) or (not is_male and data['pullups'] > 3) else 0
        push = 1 if (is_male and data['pushups'] > 45) or (not is_male and data['pushups'] > 20) else 0
        
        earth_features = np.array([[vo2, pull, push, int(is_male)]])
        results.append(earth_model.predict_proba(earth_features)[0][1])

    # 2. ПОЛЕТНАЯ МОДЕЛЬ (Математика порогов)
    if all(k in data for k in ['il6_b', 'il6_a', 'nlr_b', 'nlr_a', 'ebv', 'cortb', 'corta', 'days']):
        # Расчет процентов роста
        il6_growth = (data['il6_a'] - data['il6_b']) / data['il6_b']
        nlr_growth = (data['nlr_a'] - data['nlr_b']) / data['nlr_b']
        cort_growth = (data['corta'] - data['cortb']) / data['cortb']
        
        # Превращение в 0/1 по твоим правилам
        cytokine = 1 if il6_growth > 0.20 or data['il6_a'] > 7 else 0
        immune_shift = 1 if data['nlr_a'] > 3.5 or nlr_growth > 0.5 else 0 # рост 1.5x = рост 50%
        viral = 1 if data['ebv'] > 500 else 0
        neuro = 1 if cort_growth > 0.25 or data['corta'] > 550 else 0
        
        agg_features = np.array([[cytokine, immune_shift, viral, neuro, data['days']]])
        results.append(agg_model.predict_proba(agg_features)[0][1])

    # Итоговый результат
    if not results:
        return jsonify({"error": "Insufficient data"}), 400
    
    final_risk = sum(results) / len(results)
    return jsonify({"risk": round(final_risk * 100, 1)})

if __name__ == '__main__':
    app.run(debug=True)