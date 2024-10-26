import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import hdbscan
from langchain_huggingface import HuggingFaceEmbeddings


def process_data(input_csv_path, output_dir):
    
    print("\nloading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",   # sentence-transformers/all-MiniLM-L6-v2 is a more cheaper option
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    print("embedding model loaded successfully...\n")
    

    print("loading data from CSV file...")
    data = pd.read_csv(input_csv_path)
    print("data loaded successfully.")

    missing_values_count = int(data["name"].isnull().sum().sum())
    if missing_values_count > 0:
        print(f"missing values found: {missing_values_count}")
        data["name"].dropna(inplace=True)
        print("rows with missing values have been removed.")
    else:
        print("no missing values found.")
        
    print(f"total rows: {data.shape[0]}, total columns: {data.shape[1]}\n")
    
    
    print("generating embeddings for each name in the data...")
    embeddings = embedding_model.embed_documents(data["name"].tolist())
    numpy_embeddings = np.array(embeddings)
    print("Embeddings generated successfully.\n")
    
    
    print("starting clustering process using HDBSCAN...")
    cluster_function = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', alpha=0.85, cluster_selection_epsilon=0.85)
    labels = cluster_function.fit_predict(numpy_embeddings)
    print("clustering completed.")
    
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)    # removing the noise labels (-1)
    num_clusters = len(unique_labels)
    print(f"total clusters: {num_clusters}")
    
    clustered_points = np.sum(labels != -1)
    print(f"total clustered points: {clustered_points}")
    
    non_clustered_points = np.sum(labels == -1)
    print(f"total non-clustered points: {non_clustered_points}")
    
    # a high score (close to 1) means clusters are well-defined and separated.
    # a low or negative score means clusters are overlapping or poorly defined.
    # calculating for more than one cluster (num_clusters > 1), because with just one cluster, there's nothing to compare it to.
    if num_clusters > 1:
        silhouette = silhouette_score(numpy_embeddings, labels)
        print(f"silhouette score: {silhouette}\n")
    else:
        print("not enough clusters to calculate silhouette score.\n")
    
    
    def save_clusters_no_noise(data, labels, output_file):
        data["Cluster ID"] = labels
        clustered_data = data[data["Cluster ID"] != -1]  # exclude noise points (label -1)
        clustered_data.to_csv(output_file, index=False)
        print(f"clusters (without noise) saved to {output_file}")

    output_file_no_noise = f"{output_dir}/output_clusters_no_noise.csv"
    save_clusters_no_noise(data, labels, output_file_no_noise)


    def save_clusters_with_noise(data, labels, output_file):
        data["Cluster ID"] = labels
        clustered_data = data[data["Cluster ID"] > -2]  # save all points, including noise (label -1)
        clustered_data.to_csv(output_file, index=False)
        print(f"clusters (with noise) saved to {output_file}")

    output_file_with_noise = f"{output_dir}/output_clusters_with_noise.csv"
    save_clusters_with_noise(data, labels, output_file_with_noise)


if __name__ == "__main__":
    input_csv_path = input("Enter the path to the input CSV file: ")
    output_dir = input("Enter the directory to save the output files: ")
    process_data(input_csv_path, output_dir)