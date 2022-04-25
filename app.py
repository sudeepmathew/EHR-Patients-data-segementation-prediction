import warnings
from fun import P_COLS, D_COLS, PRE_COLS, pat_adm_combine, diagnoses_combine, dru_consump_days, drug_records_agg, final_merge, UVA_Anal, UVA_category, UVA_outlier, clean_test, gen_prediction, BVA_categorical_plot, corr_plot, segmentation, display_clusters, preprocess_df, gen_graphs
import pandas as pd
import numpy as np
import streamlit as st
import re
r = re.compile("\d+")
warnings.filterwarnings('ignore')

st.title("Clinical Data Analytics")

nav = st.sidebar.radio(
    "Navigation", ["Patient Profiling", "Segmentation", "SAE Prediction"])

if nav == "Patient Profiling":

    with st.sidebar:
        upload_file1 = st.file_uploader(
            "Upload .csv, .xlsx file for patient data")
        upload_file2 = st.file_uploader(
            "Upload .csv, .xlsx file for Admission data")
        upload_file3 = st.file_uploader(
            "Upload .csv, .xlsx file for patient d_icd_diagnoses")
        upload_file4 = st.file_uploader(
            "Upload .csv, .xlsx file for diagnoses_icd")
        upload_file5 = st.file_uploader(
            "Upload .csv, .xlsx file for presecription data")

    if upload_file1 is not None and upload_file2 is not None and upload_file3 is not None and upload_file4 is not None and upload_file5 is not None:

        patients = pd.read_csv(upload_file1)
        admissions = pd.read_csv(upload_file2)
        d_icd_diagnoses = pd.read_csv(upload_file3)
        diagnoses_icd = pd.read_csv(upload_file4)
        presc = pd.read_csv(upload_file5)
        padm = pat_adm_combine(patients, admissions, P_COLS)
        diagnoses = diagnoses_combine(diagnoses_icd, d_icd_diagnoses, D_COLS)
        presc = dru_consump_days(presc, 'startdate',  'enddate')
        presc_final = drug_records_agg(presc)
        df = final_merge(padm, diagnoses, presc_final)

        uni_out = st.checkbox('Univariate Analyisis', False)
        if uni_out:
            with st.container():
                vals = ['disease_count', 'drg_num_of_days']
                choice = st.selectbox(
                    "Select Test name for the Analysis", vals)
                # col1 = st.columns(1)
                # with col1:
                with st.spinner("Generaring Vizualizations"):
                    UVA_Anal(df, choice)
            with st.container():
                cats = ['gender', 'drug', 'diagnosis']
                categories = st.selectbox(
                    "Select Test name for the Analysis", cats)
                with st.spinner("Generaring Vizualizations"):
                    UVA_category(df, categories)

            with st.container():
                outliers = ['disease_count', 'drg_num_of_days']
                out = st.selectbox(
                    "Select Variable for the Analysis", outliers)
                with st.spinner("Generaring Vizualizations"):
                    UVA_outlier(df, out, include_outlier=True)
        multy_out = st.checkbox('Multivariate Analyisis with Gender', False)
        if multy_out:
            with st.container():
                select = ['drug', 'diagnosis']
                value = st.multiselect(
                    "select any one of the categories: ", select)
                if len(value) > 0:
                    val = value[0]
                    with st.spinner("Generaring Vizualizations"):
                        BVA_categorical_plot(df, 'gender', val)
        multy_out2 = st.checkbox(
            'Multivariate Analyisis with Is_expire', False)
        if multy_out2:
            with st.container():
                select2 = ['drug', 'diagnosis', 'gender']
                df['hospital_expire_flag'] = df['hospital_expire_flag'].map(
                    {1: 'Yes', 0: 'No'})
                value2 = st.multiselect(
                    "select any one of the categories: ", select2)
                if len(value2) > 0:
                    val2 = value2[0]
                    with st.spinner("Generaring Vizualizations"):
                        BVA_categorical_plot(df, 'hospital_expire_flag', val2)

        corr = st.checkbox("Correlation Analysis")
        if corr:
            with st.container():
                nums = numerical = df.select_dtypes(
                    include=['int64', 'float64', 'Int64'])[:]
                correlation = ['pearson', 'spearman']
                cor_method = st.selectbox(
                    "Select Variable for the Analysis", correlation)
                with st.spinner("Generaring Vizualizations"):
                    corr_plot(numerical, cor_method)


if nav == "Segmentation":
    st.title("CHF NOS Patients Segmentation")
    cluster = st.file_uploader(
        "Upload .csv, .xlsx file for patient data")

    if cluster is not None:
        cluster_data = pd.read_csv(cluster)
        try:
            df = segmentation(cluster_data)
        except:
            st.write(
                "Clear the existing csv files loaded in Chache and Load correct file for segementation")
        else:
            st.write(df)
            df_n = df.copy()
            cols = ['is_hyper', 'is_diabetics', 'is_resp', 'is_kideney']
            # pre_df = preprocess_df(df, cols)
            df_n['is_hyper'] = df_n['is_hyper'] .map({1: 'Yes', 0: 'No'})
            df_n['is_diabetics'] = df_n['is_diabetics'] .map(
                {1: 'Yes', 0: 'No'})
            df_n['is_resp'] = df_n['is_resp'] .map({1: 'Yes', 0: 'No'})
            df_n['is_kideney'] = df_n['is_kideney'] .map(
                {1: 'Yes', 0: 'No'})
            df_n['gender'] = df_n['gender'].map({1: 'M', 0: 'F'})
            df_n['age_group'] = df_n['age_group'].map(
                {0: 'Adult', 1: 'Senior citizen', 2: 'Very old', 3: 'Young'})
            df_n['predicted_cluster'] = df_n['predicted_cluster'].map(
                {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6'})
            with st.container():
                seg_button = st.button("Display Clusters")
                if seg_button:
                    with st.spinner("Generaring Vizualizations"):
                        display_clusters(df)
                cats = ['gender', 'age_group',
                        'is_kideney', 'is_diabetics', 'is_resp']
                nums = ['no_of_days_admitted', 'count_of_diagnosis',
                        'drug_adm_days', 'no_of_drugs']
                df_n = df_n.dropna()
                cluster_ch = df_n['predicted_cluster'].unique()
                cluster_ch = sorted(
                    cluster_ch, key=lambda x: int(r.search(x).group()))
                cluster_choice = st.selectbox(
                    "Select Clusters", cluster_ch)
                if cluster_choice:
                    st.write(cluster_choice)
                    df_n = df_n[df_n['predicted_cluster'] == cluster_choice]
                    st.title(cluster_choice)
                    clust_select = st.checkbox(
                        'More Information on Clusters', False)
                    cats_choice = st.selectbox("Select Field", cats)
                    if cats_choice:
                        UVA_category(df_n, cats_choice)
                    nums_choice = st.selectbox("Select Field", nums)
                    if nums_choice:
                        UVA_Anal(df_n, nums_choice)


if nav == "SAE Prediction":
    st.title("Serious Adverse Event Prediction")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            Diag = st.text_area(
                "Diagnosis", value="Enter the Diagnosis", height=7)
            text = clean_test(Diag)
        with col2:
            submit = st.button("Predict")
        if submit:
            pred = gen_prediction(text)
            if pred == 0:
                event = "No Serious Event"
            else:
                event = "Serious Event"
            st.title("This Patient has" + ' ' + event)
