"""
Title: Development and international evaluation of an artificial intelligence-based model (PROGRxN-BCa) using the 
WHO 2004/2022 grading system to predict progression risk and improve substratification in non-muscle invasive 
bladder cancer
"""

# Import packages and libraries
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="PROGRxN-BCa: PROGression Risk assessment in Non-muscle invasive Bladder Cancer",
                       page_icon="https://bladdercancercanada.org/wp-content/uploads/2017/03/bcc-fav-icon.png",
                       layout="wide",
                       initial_sidebar_state="auto"
                       )

st.sidebar.image('https://uofturology.ca/wp-content/themes/uofturology22/img/UofT-Urology-logo@2x.png',
                 use_column_width=True)

def disclaimer():

    st.title("PROGRxN-BCa (PROGression Risk assessment in Non-muscle invasive Bladder Cancer)")
    st.header("Legal Disclaimer")
    st.markdown(
        """
        This app predicts the probability of progression to muscle-invasive, nodal, or metastatic disease in patients 
        with non-muscle invasive bladder cancer (NMIBC) following transurethral resection of bladder tumour (TURBT). 
        This tool uses the WHO 2004/2022 grading classification system.
        \n
        This predictive model is for general health information only and should not to be used as a substitute for 
        professional advice or clinical expertise. Patients should not rely on information provided by PROGRxN-BCa for 
        their own health problems and should discuss any questions with their healthcare provider.
        \n
        The authors of PROGRxN-BCa do not make any warranties, express or implied representations whatsoever, regarding 
        the accuracy, completeness, timeliness, comparative or controversial nature, or usefulness of any information 
        contained or referenced in PROGRxN-BCa. The authors do not assume any risk whatsoever for your use of 
        PROGRxN-BCa or the information contained herein. Health-related information gets updated frequently and 
        therefore information contained in PROGRxN-BCa may be outdated, incomplete, or incorrect.
        \n
        PROGRxN-BCa does not record any user information and does not contact any users of the tool.
        \n
        You are advised to consult with a physician or other professional healthcare provider prior to making 
        any decisions, or undertaking any actions or not undertaking any actions related to any healthcare problem or 
        issue you might have at any time, now or in the future. In using PROGRxN-BCa, you agree that neither the authors
        nor any other party is or will be liable or otherwise responsible for any decision made or any action taken or 
        any action not taken due to your use of any information presented in PROGRxN-BCa.
    """)


    agree = st.button("Yes, I agree to the above terms")
    if agree:
        switch_page("use progrxn-bca")

if __name__ == "__main__":
    disclaimer()
