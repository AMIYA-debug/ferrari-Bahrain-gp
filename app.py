from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import threading

BASE_DIR = os.path.dirname(__file__)
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
PIT_IN_PATH = os.path.join(BASE_DIR, 'pit_in.pkl')
PIT_OUT_PATH = os.path.join(BASE_DIR, 'pit_out.pkl')

COL_TO_SCALE = [
    'lap_time_s','s1_s','s2_s','s3_s','tyre_life','position','prev_position',
    'avg_speed','max_speed','std_speed','avg_throttle','avg_brake',
    'session_time_sec','stint','overtakes'
]

MODEL_FEATURE_ORDER = COL_TO_SCALE + ['pit_in', 'pit_out']

scaler = None
le_pit_in = None
le_pit_out = None
model = None
_lock = threading.Lock()

def load_assets():
    global scaler, le_pit_in, le_pit_out, model
    with _lock:
        if scaler is None:
            scaler = pickle.load(open(SCALER_PATH, 'rb'))
        if le_pit_in is None:
            le_pit_in = pickle.load(open(PIT_IN_PATH, 'rb'))
        if le_pit_out is None:
            le_pit_out = pickle.load(open(PIT_OUT_PATH, 'rb'))
        if model is None:
            model = tf.keras.models.load_model(MODEL_PATH)

def assets():
    load_assets()
    return scaler, le_pit_in, le_pit_out, model

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        scaler, le_pit_in, le_pit_out, model = assets()

        values = {}
        for col in COL_TO_SCALE:
            val = request.form.get(col, '').strip()
            if val == '':
                return render_template('index.html', error=f'Missing value for {col}')
            values[col] = float(val)

        pit_in_raw = request.form.get('pit_in', 'no')
        pit_out_raw = request.form.get('pit_out', 'no')

        def encode(labeler, raw):
            if raw.isdigit():
                return int(raw)
            try:
                return int(labeler.transform([raw])[0])
            except:
                if raw.lower() in ['yes', 'y', 'true', '1']:
                    if 'yes' in labeler.classes_:
                        return int(labeler.transform(['yes'])[0])
                    if 'Y' in labeler.classes_:
                        return int(labeler.transform(['Y'])[0])
                    return 1
                return 0

        pit_in = encode(le_pit_in, pit_in_raw)
        pit_out = encode(le_pit_out, pit_out_raw)

        df = pd.DataFrame([values])
        scaled = pd.DataFrame(scaler.transform(df[COL_TO_SCALE]), columns=COL_TO_SCALE)

        merged = np.concatenate([scaled.values[0], np.array([pit_in, pit_out])]).reshape(1, -1)

        prediction = model.predict(merged)
        prob = float(prediction[0][0])
        tyre = 'soft' if prob >= 0.5 else 'hard'

        return render_template('index.html', result=tyre, prob=round(prob, 4))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

