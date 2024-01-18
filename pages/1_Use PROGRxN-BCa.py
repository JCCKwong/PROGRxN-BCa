import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
#import PIL.Image
from lifelines import KaplanMeierFitter
import joblib
from sksurv.ensemble import RandomSurvivalForest
#from pathlib import Path
#from google_drive_downloader import GoogleDriveDownloader as gdd
#from persist import persist, load_widget_state

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
    #uhnthp = pd.read_excel(r'models\PROGRxN-BCa training set (Nov 28, 2023).xlsx')
    #cbcis = pd.read_excel(r'models\PROGRxN-BCa CBCIS conservative set (Nov 28, 2023).xlsx')
    #data = pd.concat([uhnthp, cbcis], ignore_index=True)
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

    #query = data[(data['Sex'] == pt_features['Sex'][0]) &
    #             (data['Age'].between(pt_features['Age'][0] - 5, pt_features['Age'][0] + 5)) &
    #             (data['Recurrent Tumour'] == pt_features['Recurrent Tumour'][0]) &
    #             (data['Stage'] == pt_features['Stage'][0]) &
    #             (data['Concomitant CIS'] == pt_features['Concomitant CIS'][0]) &
    #             #(data['T1 Substratification'] == pt_features['T1 Substratification'][0]) &
    #             (data['Grade'] == pt_features['Grade'][0]) &
    #             (data['Variant Histology'] == pt_features['Variant Histology'][0]) &
    #             (data['Lymphovascular Invasion'] == pt_features['Lymphovascular Invasion'][0]) #&
    #             #(data['Number of Tumours'] == pt_features['Number of Tumours'][0]) &
    #             #(data['Tumour Diameter'] == pt_features['Tumour Diameter'][0]) &
    #             #(data['Repeat TURBT'] == pt_features['Repeat TURBT'][0]) &
    #             #(data['BCG'] == pt_features['BCG'][0]) &
    #             #(data['SIC'] == pt_features['SIC'][0])
    #             ]
    #progression_cases = sum(query['Progression'])
    #similar_cases = len(query['Progression'])
    #if similar_cases == 0:
    #    col2.write("There are no patients with similar characteristics.")
    #else:
    #    progression_rate = round((progression_cases/similar_cases)*100)
    #    median_fu = round(np.median(query[query['Progression'] == 0]['Time']),1)
    #    fu_quartile1 = round(np.percentile(query[query['Progression'] == 0]['Time'], 25),1)
    #    fu_quartile3 = round(np.percentile(query[query['Progression'] == 0]['Time'], 75),1)
    #
    #    col2.write(f"There are {similar_cases} patients with similar characteristics. During a median follow-up of "
    #             f"{median_fu} years (IQR {fu_quartile1}-{fu_quartile3}), {progression_cases} patients "
    #             f"({progression_rate}%) progressed to muscle-invasive or metastatic disease.")
    #
    #
    #    fig, ax = plt.subplots(1, 1)
    #
    #    # Plot cumulative incidence of progression for similar patients
    #    kmf = KaplanMeierFitter()
    #    ax = kmf.fit(query['Time'], query['Progression'],
    #                 label='Similar patients').plot_cumulative_density(ax=ax, color='r', lw=3)
    #    ax.set_ylim([0, 0.8])
    #    ax.set_yticks(np.arange(0, 0.9, 0.1))
    #    ax.set_yticklabels(np.arange(0, 90, 10))
    #    ax.set_xlim([0, 5])
    #    ax.set_xticks(range(0, 6, 1))
    #    ax.set_xlabel('Years')
    #    ax.set_ylabel('Incidence of Progression (%)')
    #    ax.grid(which='major', axis='both', color='k', linestyle='-', linewidth=1, alpha=.1)
    #    ax.legend().remove()
    #    plt.title('Similar patients' + ' (n=%d)' % len(query['Progression']))
    #
    #    # Create another axes where we can put size ticks
    #    ax2 = plt.twiny(ax=ax)
    #    ax2.set_xticks(range(0, 11, 1))
    #
    #    # Set ticks and labels on bottom
    #    ax2.xaxis.tick_bottom()
    #
    #    # Set limit
    #    min_time, max_time = ax.get_xlim()
    #    ax2.set_xlim(min_time, max_time)
    #    ticklabels = []
    #
    #    for tick in ax2.get_xticks():
    #        lbl = ""
    #
    #        # Get counts at tick
    #        counts = []
    #        for f in [kmf]:
    #            event_table_slice = f.event_table.assign(
    #                at_risk=lambda x: x.at_risk - x.removed
    #            )
    #            if not event_table_slice.loc[:tick].empty:
    #                event_table_slice = (
    #                    event_table_slice.loc[:tick, ["at_risk"]]
    #                        .agg(
    #                        {
    #                            "at_risk": lambda x: x.tail(1).values
    #                        }
    #                    )  # see #1385
    #                        .rename(
    #                        {
    #                            "at_risk": "At risk"
    #                        }
    #                    )
    #                        .fillna(0)
    #                )
    #                counts.extend([int(c) for c in event_table_slice])
    #
    #        # Create tick label
    #        lbl += ""
    #        for i, c in enumerate(counts):
    #            s = "\n{}"
    #            lbl += s.format(c)
    #        ticklabels.append(lbl)
    #
    #    # Align labels to the right so numbers can be compared easily
    #    ax2.set_xticklabels(ticklabels, ha="center", y=-0.1)
    #
    #    plt.text(-0.7, -0.16, 'Number at Risk', ha='right')
    #
    #    col2.pyplot(fig, use_container_width=True)

print(model)