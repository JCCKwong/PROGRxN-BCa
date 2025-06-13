import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="PROGRxN-BCa: PROGression Risk assessment in Non-muscle invasive Bladder Cancer",
                   page_icon="https://bladdercancercanada.org/wp-content/uploads/2017/03/bcc-fav-icon.png",
                   layout="wide",
                   initial_sidebar_state="auto"
                   )

st.title("Validate PROGRxN-BCa (for researchers)")
st.sidebar.image('https://uofturology.ca/wp-content/themes/uofturology22/img/UofT-Urology-logo@2x.png',
                 use_column_width=True)

# Step 1: Download database template
st.header("Step 1", divider="gray")
st.markdown("Download the database template (CSV file) and review the data dictionary below.")
column_names = ['ID', 'Age', 'Sex', 'Recurrent Tumour', 'Stage', 'Concomitant CIS', 'T1 Substratification', 'Grade',
                'Variant Histology', 'Lymphovascular Invasion', 'Number of Tumours', 'Tumour Diameter', 'Repeat TURBT',
                'BCG', 'SIC', 'Progression', 'Time']

df = pd.DataFrame(columns=column_names)
template = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download database template",
    data=template,
    file_name='template.csv',
    mime='text/csv',
)

dictionary = pd.read_csv(r'dictionary/dictionary.csv')
st.dataframe(data=dictionary, use_container_width=True, hide_index=True)

# Step 2: Populate database template
st.header("Step 2", divider="gray")
st.markdown("Populate the database template with your institutional data (missing data can be left blank). Please "
            "ensure that you have received institutional research ethics board approval prior to starting Step 2.")

# Step 3: Upload completed database
st.header("Step 3", divider="gray")
st.markdown("Upload your completed database. A data quality check will automatically be executed to identify any "
            "potential errors. You can only proceed to validate PROGRxN-BCa once your database is error-free.")

uploaded_file = st.file_uploader("Choose a file", type='csv')
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)

    with st.status("Executing data quality check...", expanded=True) as status:
        # Check if DataFrame columns match the list of column names
        columns_match = set(column_names).issubset(set(uploaded_df.columns))

        if columns_match:
            st.write(":white_check_mark: Database columns match the required list of columns")
        else:
            missing_columns = set(column_names) - set(uploaded_df.columns)
            st.error(f"The following required columns are missing: {', '.join(missing_columns)}. Please ensure that "
                     f"you use the database template provided in Step 1.", icon="🚨")
            st.stop()

        # Check if there are any missing values
        selected_df = uploaded_df[column_names]
        missing_values = selected_df.drop(columns='ID').isna().sum()
        columns_with_missing_values = missing_values[missing_values > 0]

        # Display columns with missing values and the number of missing values
        if not columns_with_missing_values.empty:
            st.error("There are missing values in the database. These must be corrected before validating "
                     "PROGRxN-BCa.", icon="🚨")
            st.write("Columns with missing values:")
            for column, count in columns_with_missing_values.items():
                st.write(f"{column}: {count} missing value(s)")
            st.stop()
        else:
            st.write(":white_check_mark: No missing values")

        # Check if there are any negative values for Age or Time
        negative_count_age = (selected_df['Age'] <= 0).sum()
        negative_count_time = (selected_df['Time'] <= 0).sum()

        if negative_count_age == 0 and negative_count_time == 0:
            st.write(":white_check_mark: No zero or negative values for the 'Age' and 'Time' columns")
        else:
            st.error(f"Age and Time columns must only contain positive values. There are {negative_count_age} values "
                     f"in the 'Age' column and {negative_count_time} values in the 'Time' column with zero or negative "
                     f"values.", icon="🚨")
            st.stop()

        # Check if other variables contain only the allowed values
        allowed_values_stage = [0, 1, 2]
        allowed_values = [0, 1]

        # Check if variables contain only allowed values
        valid_values_count_stage = selected_df['Stage'].isin(allowed_values_stage).sum()
        valid_values_count = (selected_df[['Recurrent Tumour', 'Concomitant CIS', 'T1 Substratification', 'Grade',
                              'Variant Histology', 'Lymphovascular Invasion', 'Number of Tumours', 'Tumour Diameter',
                              'Repeat TURBT', 'BCG', 'SIC']].isin(allowed_values)).all(axis=1).sum()

        if (valid_values_count == len(selected_df)) & (valid_values_count_stage == len(selected_df)):
            # subtracting 4 for ID, Stage, Progression, and Time
            st.write(":white_check_mark: Only allowed values entered in the feature columns")
        else:
            st.error("There are incorrect values in the uploaded database. Please refer to the data dictionary in "
                     "Step 1 for the allowed coded values for each categorical feature.", icon="🚨")
            st.stop()

        # Check if Concomitant CIS is 0 if patient has Primary CIS
        error1_filter = (selected_df['Stage'] == 1) & (selected_df['Concomitant CIS'] != 0)

        # Check if T1 Substratification is 0 if patient has Ta or CIS
        error2_filter = (selected_df['Stage'] < 2) & (selected_df['T1 Substratification'] != 0)

        # Display rows where errors are present
        error1_rows = selected_df[error1_filter]
        error2_rows = selected_df[error2_filter]

        if not error1_rows.empty:
            st.error("'Concomitant CIS' must be 0 if 'Stage' is 1")
            st.dataframe(error1_rows, use_container_width=True, hide_index=True)
            st.stop()

        if not error2_rows.empty:
            st.error("'T1 Substratification' must be 0 if 'Stage' < 2")
            st.dataframe(error2_rows, use_container_width=True, hide_index=True)
            st.stop()

        # If there are no errors, you can print a success message
        if error1_rows.empty and error2_rows.empty:
            st.write(":white_check_mark: No other coding errors found")

        status.update(label="Data upload and quality check complete!", state="complete", expanded=False)

    st.write(f"This validation cohort includes **{len(selected_df)} patients** with "
                 f"**{selected_df['Progression'].sum()} progression events "
                 f"({((selected_df['Progression'].sum()/len(selected_df))*100).round()}%)** during a median follow-up "
                 f"of **{round(np.median(selected_df[selected_df['Progression']==0]['Time']), 1)} years**.")
    if (len(selected_df) >= 1956) and (selected_df['Progression'].sum() >= 382):
        st.write(":white_check_mark: Validation cohort is adequately powered")
    else:
        st.write(f":warning: This validation cohort is **not sufficiently powered**, therefore **please interpret "
                 f"the performance metrics with caution**. A minimum sample size of 1956 patients with 382 "
                 f"progression events is required, assuming a shrinkage of 0.9, 14 features in PROGRxN-BCa, "
                 f"prediction timepoint of 5 years, and a 5% progression rate during a median follow-up of 3.9 "
                 f"years from the European Association of Urology (EAU) Prognostic Risk Group study "
                 "(https://doi.org/10.1016/j.eururo.2020.12.033).")

# Step 4: Validate PROGRxN-BCa on the uploaded database
st.header("Step 4", divider="gray")
st.markdown("Once the data quality check is complete, please reach out to the [PROGRxN-BCa study team]"
            "(mailto:jethro.kwong@mail.utoronto.ca) if you are interested in validating PROGRxN-BCa on your data.")
