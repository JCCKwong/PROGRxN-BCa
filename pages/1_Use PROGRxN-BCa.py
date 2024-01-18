import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import joblib
from sksurv.ensemble import RandomSurvivalForest

st.set_page_config(page_title="PROGRxN-BCa: PROGression Risk assessment in Non-muscle invasive Bladder Cancer",
                   page_icon="https://bladdercancercanada.org/wp-content/uploads/2017/03/bcc-fav-icon.png",
                   layout="wide",
                   initial_sidebar_state="auto"
                   )

st.title("Use PROGRxN-BCa")
st.sidebar.image('https://uofturology.ca/wp-content/themes/uofturology22/img/UofT-Urology-logo@2x.png',
                 use_column_width=True)

st.header("Instructions", divider='gray')
st.markdown(
    """
    1. Enter your information below
    1. Press the SUBMIT button
    1. PROGRxN-BCa will output the following:
        * Plot showing the individualized risk of progression over time
        * Table showing the probability of progression within one-, five-, and ten-years
    """
)

@st.cache_data()
def load_model():
    model = joblib.load(r'model\PROGRxN-BCa_model.joblib')
    return model

model = load_model()

col1, col2 = st.columns([1, 1])

# Enter information column
col1.header("Enter your information", divider='gray')

# Results column
col2.header("Your results", divider='gray')
col2.write("This graph displays your personalized risk of progressing to muscle-invasive or metastatic disease over "
           "time.")

choices = {'Male': 0,
           'Female': 1,
           'No (primary tumour)': 0,
           'Yes (recurrent tumour)': 1,
           'Ta (Non-invasive papillary carcinoma)': 0,
           'T1 (Tumour infiltrating the lamina propria)': 2,
           'CIS only (carcinoma-in-situ)': 1,
           'No': 0,
           'Yes': 1,
           'No (Ta, primary CIS, or T1 superficial invasion)': 0,
           'Yes (T1 extensive invasion)': 1,
           'Low grade': 0,
           'High grade': 1,
           'No (urothelial, squamous, or glandular)': 0,
           'Yes (micropapillary, plasmacytoid, sarcomatoid, neuroendocrine, or nested variant)': 1,
           'Single tumour': 0,
           'Multiple tumours': 1,
           'Less than 3 cm': 0,
           '3 cm or greater': 1
           }

age = col1.number_input("Age (years)", 0, 100, 72)
sex = col1.radio("Sex", options=('Male', 'Female'), horizontal=True, index=0)
history = col1.radio("Prior history of bladder cancer", options=('No (primary tumour)',
                                                                 'Yes (recurrent tumour)'
                                                                 ), horizontal=True, index=1)
stage = col1.radio("Tumour stage", options=('Ta (Non-invasive papillary carcinoma)',
                                            'T1 (Tumour infiltrating the lamina propria)',
                                            'CIS only (carcinoma-in-situ)'), horizontal=True, index=0)
cis = col1.radio("Concomitant CIS", options=('No', 'Yes'), horizontal=True, index=0)
t1substratification = col1.radio("T1 extensive invasion",
                                 options=('No (Ta, primary CIS, or T1 superficial invasion)',
                                          'Yes (T1 extensive invasion)'), horizontal=True, index=0)
grade = col1.radio("Tumour grade (WHO 2004/2022 classification system)",
                   options=('Low grade', 'High grade'), horizontal=True, index=0)
variant = col1.radio("Variant histology", options=('No (urothelial, squamous, or glandular)',
                     'Yes (micropapillary, plasmacytoid, sarcomatoid, neuroendocrine, or nested variant)'),
                     horizontal=True, index=0)
lvi = col1.radio("Lymphovascular invasion", options=('No', 'Yes'), horizontal=True, index=0)
number = col1.radio("Number of tumours", options=('Single tumour', 'Multiple tumours'), horizontal=True,
                    index=1)
size = col1.radio("Tumour size", options=('Less than 3 cm', '3 cm or greater'), horizontal=True, index=1)
returbt = col1.radio("Was a repeat TURBT planned/completed within 6 weeks of initial TURBT?",
                     options=('No', 'Yes'), horizontal=True, index=1)
bcg = col1.radio("Was intravesical bacillus Calmette-Guérin (BCG) planned?", options=('No', 'Yes'),
                 horizontal=True, index=1)
sic = col1.radio("Was single instillation chemotherapy given immediately after TURBT?",
                 options=('No', 'Yes'), horizontal=True, index=0)

submit = st.button("SUBMIT")
if submit:
    ### DATA STORAGE ###
    sex = choices[sex]
    history = choices[history]
    stage = choices[stage]
    cis = choices[cis]
    t1substratification = choices[t1substratification]
    grade = choices[grade]
    variant = choices[variant]
    lvi = choices[lvi]
    number = choices[number]
    size = choices[size]
    returbt = choices[returbt]
    bcg = choices[bcg]
    sic = choices[sic]

    # Store a dictionary into a variable
    pt_data = {'Age': age,
               'Sex': sex,
               'Recurrent Tumour': history,
               'Stage': stage,
               'Concomitant CIS': cis,
               'T1 Substratification': t1substratification,
               'Grade': grade,
               'Variant Histology': variant,
               'Lymphovascular Invasion': lvi,
               'Number of Tumours': number,
               'Tumour Diameter': size,
               'Repeat TURBT': returbt,
               'BCG': bcg,
               'SIC': sic,
               }

    pt_features = pd.DataFrame(pt_data, index=[0])

    prog_surv = model.predict_survival_function(pt_features)
    fig_individual, ax_ind = plt.subplots(1, 1, figsize=(6, 3))

    for fn in prog_surv:
        ax_ind.step(fn.x, 1-fn(fn.x), where="post", label=None, color='red', lw=3, ls='-')
    ax_ind.set_ylabel("Risk of progression (%)")
    ax_ind.set_xlabel("Time (years)")
    ax_ind.legend(loc="upper left")
    ax_ind.set_ylim([0, 1])
    ax_ind.set_yticks(np.arange(0, 1.1, 0.1))
    ax_ind.set_yticklabels(np.arange(0, 110, 10))
    ax_ind.set_xlim([0, 10])
    ax_ind.set_xticks(range(0, 11, 1))


    ax_ind.grid(which='major', axis='both', color='k', linestyle='-', linewidth=1, alpha=.1)
    ax_ind.legend().remove()
    col2.pyplot(fig_individual, use_container_width=True)

    # Print Survival probabilities at 1, 5, 10 years
    risk_1yr = round(np.interp(1, fn.x, 1-fn(fn.x))*100, 0)
    risk_5yr = round(np.interp(5, fn.x, 1-fn(fn.x))*100, 0)
    risk_10yr = round(np.interp(10, fn.x, 1-fn(fn.x))*100, 0)

    individual_risk = pd.DataFrame({"Time (years)": [1, 5, 10],
                                    "Risk of progression (%)": [risk_1yr, risk_5yr, risk_10yr]
                                    })
    col2.dataframe(data=individual_risk, use_container_width=True, hide_index=True)

    ibcg_conditions = [(pt_features['Stage'][0] == 0) & (pt_features['Grade'][0] == 0)
                       & (pt_features['Recurrent Tumour'][0] == 1) & (pt_features['Concomitant CIS'][0] == 0)
                       & (pt_features['Variant Histology'][0] == 0),
                       (pt_features['Stage'][0] == 0) & (pt_features['Grade'][0] == 0)
                       & (pt_features['Recurrent Tumour'][0] == 0) & (pt_features['Number of Tumours'][0] == 1)
                       & (pt_features['Concomitant CIS'][0] == 0) & (pt_features['Variant Histology'][0] == 0),
                       (pt_features['Stage'][0] == 0) & (pt_features['Grade'][0] == 0)
                       & (pt_features['Recurrent Tumour'][0] == 0) & (pt_features['Tumour Diameter'][0] == 1)
                       & (pt_features['Concomitant CIS'][0] == 0) & (pt_features['Variant Histology'][0] == 0),
                       (pt_features['Stage'][0] == 2) & (pt_features['Grade'][0] == 0)
                       & (pt_features['Concomitant CIS'][0] == 0) & (pt_features['Variant Histology'][0] == 0)
                       ]

    if pd.DataFrame(ibcg_conditions).any(axis=0).sum() == 1:
        prediction = model.predict(pt_features)
        risk_tertile = str(np.where(prediction >= 45.866148561624186, 'upper',
                           np.where(prediction <= 7.681456768856299, 'lower', 'middle'))[0])
        col2.write('You are considered *intermediate-risk* based on the [International Bladder Cancer Group]'
                   '(https://doi.org/10.1016/j.euo.2022.05.005) consensus definition for intermediate-risk non-muscle '
                   f'invasive bladder cancer, and in the **{risk_tertile} tertile** based on PROGRxN-BCa.''')
