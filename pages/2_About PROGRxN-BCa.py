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
    **An artificial intelligence-based model to predict progression risk in non-muscle invasive bladder cancer (PROGRxN-BCa) and improve 
    substratification of intermediate-risk patients: an international model development and evaluation study**\n

    *Jethro C.C. Kwong$^{1,2,3}$, Zizo Al-Daqqaq$^{4}$, Yashan Chelliahpillai$^{4}$, Soomin Lee$^{4}$, 
    Kellie Kim$^{4}$, Maximiliano Ringa$^{5}$, Andrew Feifer$^{1,2,5}$, Katherine Lajkosz$^{2}$, 
    Marian S. Wettstein$^{1,2}$, Amy Chan$^{6}$, Taeweon Lee$^{7}$, Myky Nguyen$^{8}$, Wassim Kassouf$^{9}$, 
    Peter C. Black$^{10}$, Rodney H. Breau$^{11}$, Michele Lodde$^{12}$, Adrian Fairey$^{13}$, 
    Jean-Baptiste Lattouf$^{14}$, Claudio Jeldres$^{15}$, Ricardo Rendon$^{16}$, Nimira Alimohamed$^{17}$, 
    Neil E. Fleshner$^{1,2}$, Romain Diamand$^{8}$, Paolo Gontero$^{18}$, Richard J. Sylvester$^{19}$, 
    Bas W.G. van Rhijn$^{20}$, Ashish M. Kamat$^{7}$, Alistair E.W. Johnson$^{3,21,22}$, 
    Alexandre R. Zlotta$^{1,2,6}$, Girish S. Kulkarni$^{1,2,3}$, on behalf of the PROGRxN-BCa consortium*\n

    1. Division of Urology, Department of Surgery, University of Toronto, Toronto, Canada
    1. Division of Urology, Department of Surgery, University Health Network, Toronto, Canada
    1. Temerty Centre for AI Research and Education in Medicine, University of Toronto, Toronto, Canada
    1. Temerty Faculty of Medicine, University of Toronto, Toronto, Canada
    1. Division of Urology, Department of Surgery, Trillium Health Partners, Mississauga, Canada
    1. Division of Urology, Department of Surgery, Mount Sinai Hospital, Sinai Health System, Toronto, Canada
    1. Department of Urology, University of Texas MD Anderson Cancer Center, Houston, United States
    1. Department of Urology, Jules Bordet Institute-Erasme Hospital, Hôpital Universitaire de Bruxelles, Université Libre de Bruxelles, Brussels, Belgium
    1. Department of Urology, McGill University Health Centre, Montreal, Canada
    1. Department of Urologic Sciences, University of British Columbia, Vancouver, Canada
    1. Division of Urology, Department of Surgery, The Ottawa Hospital Research Institute, Ottawa, Canada
    1. Division of Urology, Department of Surgery, CHU de Québec-Université Laval, Quebec City, Canada
    1. Division of Urology, Department of Surgery, University of Alberta, Edmonton, Canada
    1. Division of Urology, Department of Surgery, Centre Hospitalier de l'Université de Montréal, Montreal, Canada
    1. Division of Urology, Department of Surgery, Université de Sherbrooke, Sherbrooke, Canada
    1. Department of Urology, Dalhousie University, Halifax, Canada
    1. Department of Oncology, Cumming School of Medicine, University of Calgary, Calgary, Canada
    1. Department of Urology, Città della Salute e della Scienza, University of Torino School of Medicine, Torino, Italy
    1. Department of Biostatistics, European Organization for Research and Treatment of Cancer, Brussels, Belgium
    1. Department of Surgical Oncology (Urology), Netherlands Cancer Institute, Antoni van Leeuwenhoek Hospital, Amsterdam, The Netherlands
    1. Division of Biostatistics, Dalla Lana School of Public Health, University of Toronto, Toronto, Canada
    1. Vector Institute, Toronto, Canada

    For more information, the full manuscript is available [here](#).
    """)

st.header('Abstract', divider='gray')
st.markdown(
    """
    **Background**: Non-muscle invasive bladder cancer (NMIBC) is a heterogenous disease with varying risks of 
    progression to potentially lethal muscle-invasive disease. Current tools perform poorly, which impacts downstream 
    management and patient outcomes. To address these limitations, we developed a progression risk assessment tool 
    using artificial intelligence approaches (PROGRxN-BCa).\n
    
    **Methods**: PROGRxN-BCa was trained using 14 clinicopathological features on 3324 NMIBC patients treated from 
    2005-2022 at four Canadian academic or community hospitals. External testing was performed on 9335 patients 
    treated from 2005-2023 across 30 North American and European institutions. Primary outcome was time to progression, 
    defined as development of muscle-invasive or metastatic disease. PROGRxN-BCa was compared to the guideline-endorsed 
    European Association of Urology (EAU) risk calculator. Performance for predicting five-year progression risk was 
    characterized using concordance index, calibration plots, instability assessments, decision curve analysis, and an 
    algorithmic audit.\n
    
    **Findings**: Among 12659 patients, 1405 (11%) progressed over a median follow-up of 3.3 years (IQR 1.6-5.8). 
    PROGRxN-BCa had significantly higher concordance index and net benefit overall and across different subgroups 
    (training: 0.83, 95% CI 0.81-0.84; testing: 0.79, 95% CI 0.77-0.80) compared to the EAU risk calculator 
    (training 0.76, 95% CI 0.74-0.78; testing 0.71, 95% CI 0.70-0.72). This improvement was consistent among patients 
    who received guideline-concordant or non-guideline concordant care. Furthermore, it outperformed other 
    guideline-endorsed tools and a previously published AI model using the World Health Organization 1973 grade. 
    Compared to current guideline recommendations, PROGRxN-BCa improved substratification of intermediate-risk 
    patients into distinct risk tertiles with five-year progression risks of 2, 7, and 17% - including 9% of patients 
    in the highest risk tertile who may otherwise not be considered for treatment intensification.\n
    
    **Interpretation**: PROGRxN-BCa outperformed current tools and improved substratification, especially in the 
    heterogenous intermediate-risk group, in the largest NMIBC cohort of its kind. PROGRxN-BCa may better inform 
    risk adapted management for current NMIBC guidelines.
    """
)

st.header('Contributing Institutions', divider='gray')
st.write('')

# Sample hospital data (Replace with your own list)
hospital_data = [
    {"name": "Toronto General Hospital", "lat": 43.6583, "lon": -79.3891},
    {"name": "Mayo Clinic", "lat": 44.0216, "lon": -92.4668},
    {"name": "Charité - Universitätsmedizin Berlin", "lat": 52.525, "lon": 13.378},
]

# Convert to DataFrame
df = pd.DataFrame(hospital_data)

# Pydeck Map Configuration
view_state = pdk.ViewState(
    latitude=df["lat"].mean(),
    longitude=df["lon"].mean(),
    zoom=3,
    pitch=0
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["lon", "lat"],
    get_color=[255, 0, 0, 200],  # Red markers
    get_radius=50000,
    pickable=True
)

tooltip = {"html": "<b>{name}</b>", "style": {"color": "white"}}

# Render Map
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

#data = pd.DataFrame({
#    "name": ["Toronto General Hospital", "Credit Valley Hospital", "Mayo Clinic Rochester"],
#    "lat": [43.6584, 43.5600, 44.0217],
#    "lon": [-79.3892, -79.7125, -92.4668]
#})

# Display Map
#st.map(data)
#st.image('https://bladdercancercanada.org/wp-content/uploads/2017/03/BCCCentersMap.png',
#         caption='Canadian Bladder Cancer Information System')

st.header('Funding', divider='gray')
st.write('')
col1, col2 = st.columns([1, 2])
col1.image('https://lh3.googleusercontent.com/p/AF1QipMcP_iOErUmGpGCsxGVZWFN3rMn0Uj4Qg1lgp3s=s680-w680-h510',
           caption='CUASF Bladder Cancer Canada Research Grant')
col1.image('https://pbs.twimg.com/profile_images/1324121822263185409/XujIy-bC_400x400.jpg',
           caption='Temerty Innovation Grant')
