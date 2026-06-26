import streamlit as st

def render():
    st.title("Intro Page")
    st.image("assets/car.jpg", use_container_width=True)
    st.write("This should show up")
