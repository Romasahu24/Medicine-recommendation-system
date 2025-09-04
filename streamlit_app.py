import streamlit as st
import pandas as pd
import pickle

# Load model and helpers
svc = pickle.load(open('svc.pkl', 'rb'))
description  = pd.read_csv("description.csv")
precautions  = pd.read_csv("precautions_df.csv")
medications  = pd.read_csv("medications.csv")
diets        = pd.read_csv("diets.csv")
workout      = pd.read_csv("workout_df.csv")

def normalize_symptom(s):
    return s.strip().lower().replace(" ", "_")

def get_predicted_value(symptoms):
    inp = pd.DataFrame(0, index=[0], columns=feature_cols)
    for s in symptoms:
        key = normalize_symptom(s)
        if key in inp.columns:
            inp.at[0, key] = 1
    y_pred = svc.predict(inp)[0]
    return le.inverse_transform([y_pred])[0]

def helper(dis):
    desc_series = description.loc[description['Disease'] == dis, 'Description']
    desc = desc_series.iloc[0] if not desc_series.empty else "No description available."
    pre_df = precautions.loc[precautions['Disease'] == dis, ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = pre_df.iloc[0].dropna().tolist() if not pre_df.empty else []
    med = medications.loc[medications['Disease'] == dis, 'Medication'].dropna().tolist()
    die = diets.loc[diets['Disease'] == dis, 'Diet'].dropna().tolist()
    wrkout = workout.loc[workout['disease'] == dis, 'workout'].dropna().tolist()
    return desc, pre, med, die, wrkout

# --- Streamlit UI ---
st.title("🩺 Disease Prediction App")

symptoms_text = st.text_input("Enter symptoms (comma separated)")
if symptoms_text:
    user_symptoms = [s.strip() for s in symptoms_text.split(",")]
    disease = get_predicted_value(user_symptoms)
    desc, pre, med, die, wrkout = helper(disease)

    st.subheader(f"Predicted Disease: {disease}")
    st.write(desc)
    st.write("**Precautions:**", pre)
    st.write("**Medications:**", med)
    st.write("**Diet:**", die)
    st.write("**Workout:**", wrkout)

