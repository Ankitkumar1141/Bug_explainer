import streamlit as st
from bug_explainer.infer import explain_error

st.set_page_config(page_title="Bug Explainer", page_icon="🐞")
st.title("🐞 Beginner-Friendly Bug Explainer")
error = st.text_area("Paste an error message", "NameError: name 'x' is not defined")
model_path = st.text_input("Model path", "outputs/bug_explainer_final")
if st.button("Explain") and error.strip():
    with st.spinner("Generating explanation..."):
        st.write(explain_error(error, model_path=model_path))
