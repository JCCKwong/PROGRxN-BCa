import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

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
    **Development and international evaluation of an artificial intelligence-based model (PROGRxN-BCa) using the 
    WHO 2004/2022 grading system to predict progression risk and improve substratification in non-muscle invasive 
    bladder cancer**\n

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
    **Background**: Current non–muscle-invasive bladder cancer (NMIBC) tools perform suboptimally in predicting 
    progression risk to potentially lethal muscle-invasive disease. We aimed to improve risk assessment using 
    artificial intelligence approaches (PROGression Risk assessment in NMIBC [PROGRxN-BCa]).\n
    
    **Methods**: PROGRxN-BCa was trained using 14 clinicopathological features on 3324 NMIBC patients treated 
    from 2005 to 2022 at four Canadian institutions. External testing was performed on 9335 patients treated 
    from 2005 to 2023 across 30 North American and European institutions. The primary outcome was time to 
    progression (muscle-invasive or metastatic disease). PROGRxN-BCa was compared with the European Association 
    of Urology (EAU) risk calculator. Performance was characterised using concordance index (c-index), 
    calibration plots, instability assessments, decision curve analysis, and an algorithmic audit.\n
    
    **Findings**: During a median follow-up of 3.3 yr (interquartile range 1.6–5.8), 1405 of 12659 patients 
    progressed. In the external testing cohort, PROGRxN-BCa had significantly higher c-index (0.79, 95% 
    confidence interval [CI] 0.77–0.80) and net benefit overall and across different subgroups compared with 
    the EAU risk calculator (0.71, 95% CI 0.70–0.73, p < 0.001). This improvement was consistent regardless 
    of treatment with bacillus Calmette-Guérin, adherence to guideline-concordant care, and World Health 
    Organization 1973 or 2004/2022 grading system, and was consistent among patients with at least 5 yr of 
    follow-up. It also outperformed other guideline-endorsed tools and a previously published artificial 
    intelligence model. Compared with guideline recommendations, PROGRxN-BCa improved substratification of 
    intermediate-risk patients into distinct risk tertiles, with estimated 5-yr progression risks of 2%, 7%, 
    and 15%—the latter in line with high-risk NMIBC.\n
    
    **Conclusion**: PROGRxN-BCa outperformed current tools in the largest NMIBC cohort of its kind. Its 
    integration into guidelines could improve risk stratification and patient management.
    """
)

st.header('Contributing Institutions', divider='gray')
st.write('')

# Hospital data
hospital_data = [
    {"name": "Princess Margaret Cancer Center, Toronto", "lat": 43.6582, "lon": -79.3906},
    {"name": "Mount Sinai Hospital, Toronto", "lat": 43.6574, "lon": -79.3903},
    {"name": "Credit Valley Hospital, Mississauga", "lat": 43.5586, "lon": -79.7033},
    {"name": "Mississauga Hospital, Mississauga", "lat": 43.5717, "lon": -79.6076},
    {"name": "MD Anderson Cancer Center, Houston", "lat": 29.7070, "lon": -95.3971},
    {"name": "Fundacio Puigvert, Universitat Autònoma de Barcelona, Barcelona", "lat": 41.4125, "lon": 2.1728},
    {"name": "McGill University Health Center, Montreal", "lat": 45.4732, "lon": -73.6009},
    {"name": "Jules Bordet Institute-Erasme Hospital, Hôpital Universitaire de Bruxelles, Belgium", "lat": 50.8127, "lon": 4.2624},
    {"name": "Hospital Universitario Fundación Alcorcón, Madrid", "lat": 40.3488, "lon": -3.8374},
    {"name": "Vancouver General Hospital, Vancouver", "lat": 49.2614, "lon": -123.1222},
    {"name": "The Ottawa Hospital, Ottawa", "lat": 45.4017, "lon": -75.6495},
    {"name": "Città della Salute e della Scienza, University of Torino School of Medicine, Torino", "lat": 45.0414, "lon": 7.6743},
    {"name": "General Teaching Hospital and 1st Faculty of Medicine, Charles University, Prague", "lat": 50.0884, "lon": 14.4037},
    {"name": "Radboud University Medical Center, Nijmegen", "lat": 51.8234, "lon": 5.8614},
    {"name": "Centre Hospitalier de l'Université Laval, Quebec City", "lat": 46.7696, "lon": -71.2830},
    {"name": "Caritas St. Josef Medical Center, Regensburg", "lat": 49.0070, "lon": 12.1188},
    {"name": "University of Alberta Hospital, Edmonton", "lat": 53.5208, "lon": -113.5232},
    {"name": "Teaching Hospital Motol and 2nd Faculty of Medicine, Charles University, Prague", "lat": 50.074, "lon": 14.3438},
    {"name": "Centre Hospitalier de l'Université de Montréal, Montreal", "lat": 45.5116, "lon": -73.5576},
    {"name": "Fundación Instituto Valenciano de Oncología, Valencia", "lat": 39.4820, "lon": -0.3906},
    {"name": "Centre Hospitalier de l'Université de Sherbrooke, Sherbrooke", "lat": 45.4478, "lon": -71.8683},
    {"name": "Medical University of Graz, Graz", "lat": 47.0802, "lon": 15.4696},
    {"name": "Amsterdam University Medical Center, Vrije Universiteit, Amsterdam", "lat": 52.3346, "lon": 4.8598},
    {"name": "The Stokes Centre for Urology, Royal Surrey Hospital, Guildford", "lat": 51.2407, "lon": -0.6101},
    {"name": "Comprehensive Cancer Center, Medical University Vienna, Vienna", "lat": 48.2199, "lon": 16.3506},
    {"name": "Pitié Salpétrière Hospital, Sorbonne University, Paris", "lat": 48.8371, "lon": 2.3650},
    {"name": "Queen Elizabeth II Health Sciences Centre, Dalhousie University, Halifax", "lat": 44.6389, "lon": -63.5792},
    {"name": "Paul Albrechtsen Research Institute CancerCare Manitoba, Winnipeg", "lat": 49.8830, "lon": -97.1260},
    {"name": "Victoria Hospital, London", "lat": 42.9605, "lon": -81.2252},
    {"name": "Netherlands Cancer Institute, Antoni van Leeuwenhoek Hospital, Amsterdam", "lat": 52.3495, "lon": 4.8258},
    {"name": "Royal Free Hospital, London", "lat": 51.5538, "lon": -0.1662},
    {"name": "Southern Alberta Institute of Urology, Calgary", "lat": 50.9899, "lon": -114.0973},
    {"name": "Kingston General Hospital, Kingston", "lat": 44.2238, "lon": -76.4943},
    {"name": "St. Joseph's Healthcare Hamilton, Hamilton", "lat": 43.2488, "lon": -79.8708},
]

# Convert to DataFrame
df = pd.DataFrame(hospital_data)

# Pydeck Map Configuration
view_state = pdk.ViewState(
    latitude=df["lat"].mean(),
    longitude=df["lon"].mean(),
    zoom=3,
    min_zoom=1,
    max_zoom=15,
    pitch=0
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["lon", "lat"],
    get_color=[255, 0, 0, 200],
    get_radius=50000,
    radius_min_pixels=3,  
    radius_max_pixels=30,
    pickable=True
)

tooltip = {"html": "<b>{name}</b>", "style": {"color": "white"}}

# Render Map
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='light', tooltip=tooltip))

st.header('Funding', divider='gray')
st.write('')
col1, col2 = st.columns([1, 2])
col1.image('https://lh3.googleusercontent.com/p/AF1QipMcP_iOErUmGpGCsxGVZWFN3rMn0Uj4Qg1lgp3s=s680-w680-h510',
           caption='CUASF Bladder Cancer Canada Research Grant')
col1.image('https://pbs.twimg.com/profile_images/1324121822263185409/XujIy-bC_400x400.jpg',
           caption='Temerty Innovation Grant')

