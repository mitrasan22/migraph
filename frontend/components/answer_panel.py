import streamlit as st


def render_answer_panel():
    st.subheader("Answer")
    answer_box = st.empty()
    status_box = st.empty()
    return answer_box, status_box
