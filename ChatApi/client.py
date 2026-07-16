import streamlit as st
import requests

def get_gemini_response(input_text1):
    response = requests.post(
                        "http://localhost:8000/googlegenai/invoke",
                        json = {'input' : {'prompt':input_text1}})
    st.write(response.status_code)
    return response.text

def get_groq_response(input_text2):
    response = requests.post(
                        "http://localhost:8000/groq/invoke",
                        json = {'input' : {'prompt':input_text2}})
    return response.json()['output']['content']

# StreamLit Framework
st.title("Chatbot eith gemini and groq")
input_text1 = st.text_input("Chat with Gemini")
input_text2 = st.text_input("Chat with Groq")

if input_text1:
    st.write(get_gemini_response(input_text1))
if input_text2:
    st.write(get_groq_response(input_text2))



