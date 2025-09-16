import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
mlflow.set_tracking_uri("http://127.0.0.1:5000")
df = pd.read_csv(r"D:\AI WORKSHOP\TASK\MLOPS\CC_GENERAL_preprocessed.csv")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

def train_models(experiment_name="ClusteringModels"):
    mlflow.set_experiment(experiment_name)

    
    with mlflow.start_run(run_name="KMeans"):
        best_k, best_sil = 2, -1
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=50)
            labels = km.fit_predict(df_scaled)
            sil = silhouette_score(df_scaled, labels)
            if sil > best_sil:
                best_sil, best_k = sil, k

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50)
        kmeans_labels = kmeans.fit_predict(df_scaled)

        metrics = {
            "Silhouette": silhouette_score(df_scaled, kmeans_labels),
            "Davies-Bouldin": davies_bouldin_score(df_scaled, kmeans_labels),
            "Calinski-Harabasz": calinski_harabasz_score(df_scaled, kmeans_labels),
            "Inertia": kmeans.inertia_
        }

        mlflow.log_param("Best_K", best_k)
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

    with mlflow.start_run(run_name="Hierarchical"):
        linkages = ['ward', 'complete', 'average', 'single']
        best_link, best_sil_h = "ward", -1

        for link in linkages:
            model = AgglomerativeClustering(n_clusters=best_k, linkage=link)
            labels = model.fit_predict(df_scaled)
            sil = silhouette_score(df_scaled, labels)
            if sil > best_sil_h:
                best_sil_h, best_link = sil, link

        hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage=best_link)
        hier_labels = hierarchical.fit_predict(df_scaled)

        metrics = {
            "Silhouette": silhouette_score(df_scaled, hier_labels),
            "Davies-Bouldin": davies_bouldin_score(df_scaled, hier_labels),
            "Calinski-Harabasz": calinski_harabasz_score(df_scaled, hier_labels),
            "Inertia": np.nan
        }

        mlflow.log_param("Best_Linkage", best_link)
        for name, val in metrics.items():
            if not np.isnan(val):
                mlflow.log_metric(name, val)

    with mlflow.start_run(run_name="DBSCAN"):
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        db_labels = dbscan.fit_predict(df_scaled)
        mask = db_labels != -1  

        if np.any(mask):
            metrics = {
                "Silhouette": silhouette_score(df_scaled[mask], db_labels[mask]),
                "Davies-Bouldin": davies_bouldin_score(df_scaled[mask], db_labels[mask]),
                "Calinski-Harabasz": calinski_harabasz_score(df_scaled[mask], db_labels[mask]),
                "Inertia": np.nan
            }
        else:
            metrics = {
                "Silhouette": np.nan,
                "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan,
                "Inertia": np.nan
            }

        for name, val in metrics.items():
            if not np.isnan(val):
                mlflow.log_metric(name, val)

    print("✅ Each clustering model logged in MLflow as separate runs.")

if __name__ == "__main__":
    train_models()
