# run_deploy.py (Streamlit example)
import streamlit as st
import joblib
model = joblib.load("models/best_model.pkl")

st.title("Big Five Personality Predictor")
text = st.text_area("Enter Reddit comment:")
if st.button("Predict"):
    feats = preprocess_and_feature_engineer(text)  # reuse your funcs
    pred = model.predict([feats])[0]
    st.write({trait: round(val,2) for trait,val in zip(traits,pred)})
