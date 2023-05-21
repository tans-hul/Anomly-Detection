from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import hashlib
import joblib

# For encoding the Country, Region and City columns
def consistent_hash(value, buckets):
    sha256 = hashlib.sha256()
    sha256.update(value.encode('utf-8'))
    hash_value = int(sha256.hexdigest(), 16)
    return hash_value % buckets

# For encoding the Browser Name and Version column
def custom_hash_function(browser_name, version):
    string_to_hash = browser_name + version
    hash_object = hashlib.md5(string_to_hash.encode())
    hash_integer = int(hash_object.hexdigest(), 16) & 0xffffffff
    return hash_integer

def get_prediction(json_data):
    # Convert JSON / Dict to DataFrame
    data = pd.DataFrame.from_dict(json_data, orient='index').T
    
    # Preprocessing the data
    ## Login Timestamp
    data['Timestamp'] = pd.to_datetime(data['Login Timestamp'])
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour
    data['Minute'] = data['Timestamp'].dt.minute
    data['Second'] = data['Timestamp'].dt.second
    data.drop(['Login Timestamp', 'Timestamp'], inplace=True, axis=1)

    ## Browser Name and Version
    data[['Browser Name', 'Version']] = data['Browser Name and Version'].str.extract(r'([a-zA-Z]+[a-zA-Z\s]*[a-zA-Z]+) (\d.*\d)')
    data.drop(["Browser Name and Version"], inplace=True, axis=1)
    data['Browser Info'] = data.apply(lambda row: custom_hash_function(row['Browser Name'], row['Version']), axis=1)
    data = data.drop(['Browser Name', 'Version'], axis=1)
    
    ## Login Successful
    data['Login Successful'] = data['Login Successful'].astype(int)

    ## Device Type
    data['Device_bot'] = data['Device Type'].apply(lambda x: 1 if x == 'bot' else 0)
    data['Device_desktop'] = data['Device Type'].apply(lambda x: 1 if x == 'desktop' else 0)
    data['Device_mobile'] = data['Device Type'].apply(lambda x: 1 if x == 'mobile' else 0)
    data['Device_tablet'] = data['Device Type'].apply(lambda x: 1 if x == 'tablet' else 0)
    data['Device_unknown'] = data['Device Type'].apply(lambda x: 1 if x == 'unknown' else 0)
    data = data.drop('Device Type', axis=1)

    ## IP address
    data[['IP_Octet1', 'IP_Octet2', 'IP_Octet3', 'IP_Octet4']] = data['IP Address'].str.split('.', expand=True)
    data.drop(["IP Address"], inplace=True, axis=1)
    data['IP_Octet1'] = data['IP_Octet1'].astype(int)
    data['IP_Octet2'] = data['IP_Octet2'].astype(int)
    data['IP_Octet3'] = data['IP_Octet3'].astype(int)
    data['IP_Octet4'] = data['IP_Octet4'].astype(int)

    ## Country, Region and City
    country_buckets = 200
    region_buckets = 4000
    city_buckets = 10000
    data['Country'] = data['Country'].apply(lambda x: consistent_hash(x, country_buckets))
    data['Region'] = data['Region'].apply(lambda x: consistent_hash(x, region_buckets))
    data['City'] = data['City'].apply(lambda x: consistent_hash(x, city_buckets))

    ## User ID
    data['User ID'] = data['User ID'].map(freq_encoding)

    # Scaling the data
    scaled_data = scaler.transform(data)

    # Making the predictions
    ## KMeans
    kmeans_model = joblib.load('kmeans_1_crore.pkl')
    predicted_cluster = kmeans_model.predict(scaled_data)[0]
    centroid = kmeans_model.cluster_centers_[predicted_cluster]
    distance = np.linalg.norm(scaled_data - centroid)
    differences = np.abs(scaled_data - centroid)[0]
    feature_contributions = [(i, diff) for i, diff in enumerate(differences)]
    sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

    ## Evaluating KMeans Predictions
    top_features, contributions = zip(*sorted_contributions[:5])
    is_anomaly = distance > 4.148080343148985
    kmeans_result = {
        "is_anomaly": bool(is_anomaly),
        "top_features": [data.columns[idx] for idx in top_features],
        "top_features_contributions": contributions
    }

    ## GMM
    log_likelihood = gmm_model.score_samples(scaled_data)[0]
    is_anomaly = log_likelihood < 19.844993514473185
    component_probabilities = gmm_model.predict_proba(scaled_data)[0]
    non_anomaly_component = np.argmax(component_probabilities)
    non_anomaly_means = gmm_model.means_[non_anomaly_component]
    differences = np.abs(scaled_data - non_anomaly_means)[0]
    feature_contributions = [(i, diff) for i, diff in enumerate(differences)]
    sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

    ## Evaluating GMM Predictions
    top_features, contributions = zip(*sorted_contributions[:5])
    gmm_result = {
        "is_anomaly": bool(is_anomaly),
        "top_features": [data.columns[idx] for idx in top_features],
        "top_features_contributions": contributions
    }

    return {"kmeans": kmeans_result, "gmm": gmm_result}

app = Flask(__name__)

# Loading the models
freq_encoding = joblib.load('freq_encoder_1_crore.joblib')
scaler = joblib.load('scaler_1_crore.pkl')
kmeans_model = joblib.load('kmeans_1_crore.pkl')
gmm_model = joblib.load('gmm_1_crore.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    json_data = request.json

    # Getting the predictions
    result = get_prediction(json_data)

    # Return the prediction as a response
    return jsonify(result)

if __name__ == '__main__':
    app.run()
