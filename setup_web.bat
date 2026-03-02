@echo off
echo Setting up Wild Animal Detection System for Web...
pip install -r requirements.txt
echo Done! Running Streamlit App.
streamlit run streamlit_app.py
pause
