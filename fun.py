import warnings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
warnings.filterwarnings('ignore')

P_COLS = ['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'EXPIRE_FLAG',
          'INSURANCE', 'MARITAL_STATUS', 'ETHNICITY', 'EDREGTIME', 'EDOUTTIME']
P_COLS = [i.lower() for i in P_COLS]

D_COLS = ['SUBJECT_ID', 'ICD9_CODE']
D_COLS = [i.lower() for i in D_COLS]

PRE_COLS = ['ROW_ID', 'ICUSTAY_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN',
            'NDC', 'PROD_STRENGTH', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'ROUTE']

PRE_COLS = [i.lower() for i in PRE_COLS]

# Function to combine patients data with admission_Data


def pat_adm_combine(df1, df2, cols):
    combined_df = pd.merge(df1, df2, on='subject_id')
    combined_df[cols]
    final_df = combined_df.groupby(
        ['subject_id'], as_index=False).agg(lambda x: x.head(1))
    return final_df


def diagnoses_combine(df1, df2, cols):
    cmbine_df = pd.merge(df1, df2, on='icd9_code')
    cmbine_df = cmbine_df[D_COLS]
    cmbine_df = cmbine_df.groupby(["subject_id"])[
        "icd9_code"].count().reset_index(name="disease_count")
    return cmbine_df

# Function to combine prescription dataset


def dru_consump_days(df, start_date, end_date):
    df[start_date] = df[start_date].apply(pd.to_datetime)
    df[end_date] = df[end_date].apply(pd.to_datetime)
    #day_diff = df[end_date].apply(lambda x: x.day) - df[start_date].apply(lambda x: x.day)
    day_diff = (df[end_date] - df[start_date]).dt.days
    df['drg_num_of_days'] = round(day_diff, 1)
    return df


def drug_records_agg(df):
    combined_df = df.groupby(['subject_id'], as_index=False).agg(
        lambda x: x.sum() if x.dtype == 'float64' else x.head(1))
    combined_df = combined_df.drop(PRE_COLS, axis=1)
    return combined_df


def final_merge(df1, df2, df3):
    df = pd.merge(df1, df2, on='subject_id')
    final_df = pd.merge(df, df3, on='subject_id')
    final_df = final_df.drop(
        ['row_id_x', 'row_id_y', 'hadm_id_x', 'hadm_id_y'], axis=1)
    return final_df


def UVA_Anal(df, col):
    size = len(col)
    # Calculating discriptive statistics
    mini = df[col].min()
    maxi = df[col].max()
    ran = df[col].max() - df[col].min()
    mean = df[col].mean()
    median = df[col].median()
    std_dev = df[col].std()
    skew = df[col].skew()
    kurt = df[col].kurt()
    points = mean - std_dev, mean+std_dev

# Plotting the variables with all the information

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12, 6))
    sns.kdeplot(df[col], shade=True)
    sns.lineplot(points, [0, 0], color='black', label="std_dev")
    sns.scatterplot([mini, maxi], [0, 0], color='orange', label="min/max")
    sns.scatterplot([mean], [0], color='red', label="mean")
    sns.scatterplot([median], [0], color='blue', label="median")
    plt.xlabel('{}'.format(col), fontsize=20)
    plt.ylabel('density')
    plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0], 2), round(points[1], 2)),
                                                                                                   round(
        kurt, 2),
        round(
        skew, 2),
        (round(mini, 2), round(
            maxi, 2), round(ran, 2)),
        round(
        mean, 2),
        round(median, 2)))
    st.pyplot(fig)


def UVA_category(data, col):
    fig = plt.figure(figsize=(12, 6))
    norm_count = data[col].value_counts(normalize=True)
    n_uni = data[col].nunique()

  # Plotting the variable with every information
    sns.barplot(norm_count, norm_count.index, order=norm_count.index)
    plt.xlabel('fraction/percent', fontsize=20)
    plt.ylabel('{}'.format(col), fontsize=20)
    plt.title('n_uniques = {} \n value counts \n {};'.format(n_uni, norm_count))
    st.pyplot(fig)


def UVA_outlier(data, i, include_outlier=True):

    fig = plt.figure(figsize=(12, 6))

    # calculating descriptives of variable
    quant25 = data[i].quantile(0.25)
    quant75 = data[i].quantile(0.75)
    IQR = quant75 - quant25
    med = data[i].median()
    whis_low = med-(1.5*IQR)
    whis_high = med+(1.5*IQR)

    # Calculating Number of Outliers
    outlier_high = len(data[i][data[i] > whis_high])
    outlier_low = len(data[i][data[i] < whis_low])

    if include_outlier == True:
        sns.boxplot(data[i], orient="v")
        plt.ylabel('{}'.format(i))
        plt.title('With Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
            round(IQR, 2),
            round(med, 2),
            (round(quant25, 2), round(quant75, 2)),
            (outlier_low, outlier_high)
        ))

    else:
        data2 = data
        data2[i][data2[i] > whis_high] = whis_high+1
        data2[i][data2[i] < whis_low] = whis_low-1

      # plotting without outliers
        sns.boxplot(data2[i], orient="v")
        plt.ylabel('{}'.format(i))
        plt.title('Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
            round(IQR, 2),
            round(med, 2),
            (round(quant25, 2), round(quant75, 2)),
            (outlier_low, outlier_high)
        ))
    st.pyplot(fig)


def BVA_categorical_plot(data, tar, cat):
    '''
    take data and two categorical variables,
    calculates the chi2 significance between the two variables 
    and prints the result with countplot & CrossTab
    '''
    fig = plt.figure(figsize=(12, 6))
    # isolating the variables
    data = data[[cat, tar]][:]

    # forming a crosstab
    table = pd.crosstab(data[tar], data[cat],)
    f_obs = np.array([table.iloc[0][:].values,
                      table.iloc[1][:].values])

    # performing chi2 test
    from scipy.stats import chi2_contingency
    chi, p, dof, expected = chi2_contingency(f_obs)

    # checking whether results are significant
    if p < 0.05:
        sig = True
    else:
        sig = False

    # plotting grouped plot
    sns.countplot(x=cat, hue=tar, data=data)
    plt.title(
        "p-value = {}\n difference significant? = {}\n".format(round(p, 8), sig))

    # plotting percent stacked bar plot
    #sns.catplot(ax, kind='stacked')
    # ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack()
    # ax1.plot(kind='bar', stacked='True', title=str(ax1))
    # int_level = data[cat].value_counts()
    st.pyplot(fig)


def corr_plot(df, i):
    fig = plt.figure(figsize=(12, 6))
    correlation = df.dropna().corr(method=i)
    sns.heatmap(correlation, linewidth=2)
    plt.title(i, fontsize=18)
    st.pyplot(fig)


def clean_test(data):
    text = re.sub('[^a-zA-Z]', ' ', data)
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    text = text.lower()
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.split()
    text = [word for word in text if not word in stop_words]
    text = [" ".join(text)]
    return text


def gen_prediction(text):
    loaded_vectorizer = pickle.load(open('tfidfvector.pickle', 'rb'))
    xtrain_tfidf = loaded_vectorizer.transform(text)
    loaded_model = pickle.load(open('SAE_PredictionLR.pk1', 'rb'))
    prediction = loaded_model.predict(xtrain_tfidf)
    return prediction[0]


def segmentation(data):
    model = pickle.load(open('Segmentation.pkl', 'rb'))
    preprocessed_data = model["preprocessor"].transform(data)
    predicted_labels = model["clusterer"]["kmeans"].labels_
    pcadf = pd.DataFrame(
        model["preprocessor"].transform(data),
        columns=["component_1", "component_2"],
    )
    pcadf["predicted_cluster"] = model["clusterer"]["kmeans"].labels_
    pcadf_n = data.join(pcadf)
    return pcadf_n


def display_clusters(data):
    fig = plt.figure(figsize=(8, 8))
    plt.style.use("fivethirtyeight")
    scat = sns.scatterplot(
        "component_1",
        "component_2",
        s=50,
        data=data,
        hue="predicted_cluster",
        style="predicted_cluster",
        palette="Set2",)

    scat.set_title("CHF Patient's Segmentation")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    st.pyplot(fig)


def preprocess_df(df, cols):
    for i in cols:
        df[i] = df[i] .map({1: 'Yes', 0: 'No'})
    return df


def gen_graphs(df, cats, nums):
    UVA_category(df, cats)
    UVA_Anal(df, nums)
