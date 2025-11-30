from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Load artifacts
BASE_DIR = os.path.dirname(__file__)
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
PIT_IN_PATH = os.path.join(BASE_DIR, 'pit_in.pkl')
PIT_OUT_PATH = os.path.join(BASE_DIR, 'pit_out.pkl')

# Features that were scaled in the notebook
COL_TO_SCALE = ['lap_time_s','s1_s','s2_s','s3_s','tyre_life','position','prev_position','avg_speed','max_speed','std_speed','avg_throttle','avg_brake','session_time_sec','stint','overtakes']
# final input order used for model: scaled cols followed by pit_in and pit_out
MODEL_FEATURE_ORDER = COL_TO_SCALE + ['pit_in','pit_out']

# Load scaler, label encoders and model
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(PIT_IN_PATH, 'rb') as f:
    le_pit_in = pickle.load(f)
with open(PIT_OUT_PATH, 'rb') as f:
    le_pit_out = pickle.load(f)
# Keras model
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse numeric fields
        data = {}
        for col in COL_TO_SCALE:
            val = request.form.get(col, '')
            if val == '':
                return render_template('index.html', error=f'Missing value for {col}')
            # cast to float
            data[col] = float(val)

        # Parse pit_in / pit_out (expected values: yes/no or 1/0)
        pit_in_raw = request.form.get('pit_in', 'no')
        pit_out_raw = request.form.get('pit_out', 'no')

        # Convert to label-encoded value using saved encoders
        # Try to map common values (Yes/No, 1/0)
        def map_encoder(le, raw):
            # if raw is numeric string
            if raw.isdigit():
                val = int(raw)
                # try inverse_transform if possible
                try:
                    return val
                except Exception:
                    pass
            # try to transform by matching label
            try:
                return int(le.transform([raw])[0])
            except Exception:
                # fallback try lower-case yes/no
                if raw.lower() in ['yes','y','true','1']:
                    # find which class corresponds to positive
                    classes = list(le.classes_)
                    if 'yes' in classes:
                        return int(le.transform(['yes'])[0])
                    if 'Y' in classes:
                        return int(le.transform(['Y'])[0])
                    # else fallback to 1
                    return 1
                return 0

        pit_in = map_encoder(le_pit_in, pit_in_raw)
        pit_out = map_encoder(le_pit_out, pit_out_raw)

        # Build dataframe for scaling
        df = pd.DataFrame([data])
        # Apply scaler to scaled columns
        df_scaled = df[COL_TO_SCALE].copy()
        df_scaled = pd.DataFrame(scaler.transform(df_scaled), columns=COL_TO_SCALE)

        # Assemble final feature vector
        final_features = np.concatenate([df_scaled.values[0], np.array([pit_in, pit_out])])
        final_features = final_features.reshape(1, -1)

        # Predict
        pred = model.predict(final_features)
        # model outputs probability near 0..1
        prob = float(pred[0][0])
        label = 'soft' if prob >= 0.5 else 'hard'

        return render_template('index.html', result=label, prob=round(prob,4))

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
