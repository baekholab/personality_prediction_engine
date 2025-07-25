# run_interpret.py
import shap
import joblib

model = joblib.load("models/best_model.pkl")
explainer = shap.Explainer(model.predict, X_sample)
shap_values = explainer(X_sample)
shap.summary_plot(shap_values, X_sample)
# Save plots under docs/ or notebooks/
