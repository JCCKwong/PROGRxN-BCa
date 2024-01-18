import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="PROGRxN-BCa: PROGression Risk assessment in Non-muscle invasive Bladder Cancer",
                   page_icon="https://bladdercancercanada.org/wp-content/uploads/2017/03/bcc-fav-icon.png",
                   layout="wide",
                   initial_sidebar_state="auto"
                   )

st.title("About PROGRxN-BCa")
st.sidebar.image('https://uofturology.ca/wp-content/themes/uofturology22/img/UofT-Urology-logo@2x.png',
                 use_column_width=True)

st.markdown(
        """
    **An artificial intelligence-based model to predict 5-year progression risk in non-muscle invasive bladder cancer 
    (PROGRxN-BCa) and improve substratification of intermediate-risk patients: a retrospective, multi-institutional 
    model development and validation study.**\n

    *Jethro C.C. Kwong$^{1,2,3}$, Zizo Al-Daqqaq$^{4}$, Yashan Chelliahpillai$^{4}$, Soomin Lee$^{4}$, 
    Kellie Kim$^{4}$, Maximiliano Ringa$^{5}$, Amna Ali$^{5}$, Andrew Feifer$^{1,2,5}$, Katherine Lajkosz$^{2}$, 
    Marian S. Wettstein$^{1,2}$, Amy Chan$^{6}$, Wassim Kassouf$^{7}$, Peter C. Black$^{8}$, Rodney H. Breau$^{9}$, 
    Michele Lodde$^{10}$, Adrian Fairey$^{11}$, Jean-Baptiste Lattouf$^{12}$, Claudio Jeldres$^{13}$, 
    Ricardo Rendon$^{14}$, Nimira Alimohamed$^{15}$, Peter Chung$^{16}$, Neil E. Fleshner$^{1,2}$, 
    Antonio Finelli$^{1,2}$, Alexandre R. Zlotta$^{1,2,6}$, Alistair E.W. Johnson$^{3,17,18}$, 
    Girish S. Kulkarni$^{1,2,3}$*\n

    1. Division of Urology, Department of Surgery, University of Toronto, Toronto, Ontario, Canada
    1. Division of Urology, Department of Surgery, University Health Network, Toronto, Ontario, Canada
    1. Temerty Centre for AI Research and Education in Medicine, University of Toronto, Toronto, Ontario, Canada
    1. Temerty Faculty of Medicine, University of Toronto, Toronto, Ontario, Canada
    1. Division of Urology, Department of Surgery, Trillium Health Partners, Mississauga, Canada
    1. Division of Urology, Department of Surgery, Mount Sinai Hospital, Sinai Health System, Toronto, Canada
    1. Department of Urology, McGill University Health Centre, Montreal, Canada
    1. Department of Urologic Sciences, University of British Columbia, Vancouver, Canada
    1. Division of Urology, Department of Surgery, The Ottawa Hospital, Ottawa, Canada
    1. Division of Urology, Department of Surgery, CHU de Québec-Université Laval, Quebec City, Canada
    1. Division of Urology, Department of Surgery, University of Alberta, Edmonton, Canada
    1. Division of Urology, Department of Surgery, Centre Hospitalier de l'Université de Montréal, Montreal, Canada
    1. Division of Urology, Department of Surgery, Université de Sherbrooke, Sherbrooke, Canada
    1. Department of Urology, Dalhousie University, Halifax, Canada
    1. Department of Oncology, Cumming School of Medicine, University of Calgary, Calgary, Canada
    1. Radiation Medicine Program, Princess Margaret Cancer Centre, University Health Network, University of Toronto, 
    Toronto, Canada
    1. Division of Biostatistics, Dalla Lana School of Public Health, University of Toronto, Toronto, Canada
    1. Vector Institute, Toronto, Canada

    For more information, the full manuscript is available [here](#).
    """)

st.header('Abstract', divider='gray')
st.markdown(
    """
    **Background**: Accurate prediction of progression in non-muscle invasive bladder cancer (NMIBC) is essential for 
    patient counselling and treatment planning. We aimed to develop, externally validate, and conduct an algorithmic 
    audit of a progression risk assessment tool using artificial intelligence approaches (PROGRxN-BCa).\n
    
    **Methods**: PROGRxN-BCa, based on a random survival forest, was trained on NMIBC patients treated from Jan 1, 2005 
    to Jun 30, 2022 at four Canadian academic or community hospitals. External validation was performed on patients 
    treated from Nov 1, 2011 to Sep 11, 2023 across 13 institutions from the Canadian Bladder Cancer Information 
    System. Primary outcome was time to progression, defined as first development of muscle-invasive or metastatic 
    disease. PROGRxN-BCa was compared to the European Association of Urology risk calculator and a LASSO Cox model 
    using identical variables as PROGRxN-BCa. Model performance in predicting five-year progression risk was evaluated 
    using concordance index, calibration plots, instability assessments, decision curve analysis, and an algorithmic 
    audit.\n
    
    **Findings**: Overall, 999 out of 7032 patients developed progression during a median follow-up of 3.0 years 
    (IQR 1.4-5.4). PROGRxN-BCa had the highest concordance index overall (training: 0.83, 95% CI 0.81-0.84; validation: 
    0.76, 95% CI 0.74-0.77) and across different subgroups. It was well calibrated, more stable, and had the highest 
    net benefit for clinically relevant thresholds between 15 and 40%. PROGRxN-BCa could better substratify 
    intermediate-risk patients compared to current guideline-endorsed substratification tools, reclassifying 5% of 
    these patients with an observed five-year progression risk of 13.1% who otherwise would not have been considered 
    for treatment intensification or clinical trial enrollment.\n
    
    **Interpretation**: PROGRxN-BCa outperformed current prognostication tools and improved substratification of the 
    heterogenous intermediate-risk group. Future work should examine the clinical impact of implementing PROGRxN-BCa 
    in practice.
    """
)

st.header('Contributing Institutions', divider='gray')
st.write('')
st.image('https://bladdercancercanada.org/wp-content/uploads/2017/03/BCCCentersMap.png',
         caption='Canadian Bladder Cancer Information System')

st.header('Funding', divider='gray')
st.write('')
col1, col2 = st.columns([1, 2])
col1.image('https://lh3.googleusercontent.com/p/AF1QipMcP_iOErUmGpGCsxGVZWFN3rMn0Uj4Qg1lgp3s=s680-w680-h510',
           caption='CUASF Bladder Cancer Canada Research Grant')
col1.image('https://pbs.twimg.com/profile_images/1324121822263185409/XujIy-bC_400x400.jpg',
           caption='Temerty Innovation Grant')
