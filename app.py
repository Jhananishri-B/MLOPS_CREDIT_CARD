import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="Credit Card Clustering", layout="wide")

# ---------------------------- PREPROCESSING FUNCTIONS ----------------------------
def load_data():
    # Load dataset from CSV file
    df = pd.read_csv("CC_GENERAL.csv")
    return df

def clean_data(df):
    # Placeholder for cleaning data if needed
    # You can add more cleaning steps if required
    return df

def transform_data(df):
    # Scale numeric data using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data

# ---------------------------- LOAD DATA ----------------------------
@st.cache_data
def load_and_prepare_data():
    df = load_data()
    df = clean_data(df)
    df_transformed = transform_data(df)
    return df, df_transformed

df, df_scaled = load_and_prepare_data()
features = df.columns.tolist()

# ---------------------------- HEADER ----------------------------
st.title("💳 Credit Card Customer Clustering")
st.markdown("""
This app allows you to explore clustering of credit card customers using **KMeans**, **Hierarchical**, or **DBSCAN**.
""")

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.header("Clustering Settings")
model_option = st.sidebar.selectbox("Select Clustering Model", ["KMeans", "Hierarchical", "DBSCAN"])

if model_option == "KMeans":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
elif model_option == "Hierarchical":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    linkage = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
elif model_option == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Minimum Samples", 2, 20, 5)

# ---------------------------- MAIN ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# ---------------------------- USER INPUT FOR PREDICTION ----------------------------
st.subheader("Predict Cluster for a New Customer")
with st.form("predict_form"):
    st.write("Enter customer details:")
    customer_data = {}
    for col in features:
        customer_data[col] = st.number_input(col, value=float(df[col].median()))
    submitted = st.form_submit_button("Predict Cluster")

# ---------------------------- CLUSTERING LOGIC ----------------------------
def run_clustering(model_option, df_scaled):
    if model_option == "KMeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
    elif model_option == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)

    labels = model.fit_predict(df_scaled)
    return labels, model

labels, model = run_clustering(model_option, df_scaled)

# ---------------------------- METRICS ----------------------------
st.subheader(f"{model_option} Clustering Metrics")
if len(set(labels)) > 1 and -1 not in labels:  # valid clustering
    sil = silhouette_score(df_scaled, labels)
    dbi = davies_bouldin_score(df_scaled, labels)
    ch = calinski_harabasz_score(df_scaled, labels)

    st.metric("Silhouette Score", round(sil, 3))
    st.metric("Davies-Bouldin Index", round(dbi, 3))
    st.metric("Calinski-Harabasz Score", round(ch, 3))
else:
    st.warning("Cannot calculate metrics (DBSCAN may have all points as noise)")

# ---------------------------- NEW CUSTOMER PREDICTION ----------------------------
if submitted:
    customer_df = pd.DataFrame([customer_data])
    scaler = StandardScaler()
    customer_scaled = scaler.fit_transform(customer_df)
    if model_option != "DBSCAN":
        new_cluster = model.fit_predict(customer_scaled)[0]
    else:
        # DBSCAN doesn't have predict method; assign cluster 0 by default or customize
        new_cluster = 0
    st.success(f"Predicted Cluster for new customer: {new_cluster}")

# ---------------------------- VISUALIZATION ----------------------------
st.subheader("Cluster Distribution")
st.bar_chart(pd.Series(labels).value_counts().sort_index())

st.markdown("---")
