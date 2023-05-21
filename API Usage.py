import requests
import json

# Test record
data = {'Login Timestamp': '2020-02-29 03:34:31.674', 'User ID': 5200729356629656445, 'IP Address': '255.255.255.255', 'Country': 'AU', 'Region': 'New South Wales', 'City': 'Mosman', 'Browser Name and Version': 'Firefox 20.0.0.1618', 'Device Type': 'mobile', 'Login Successful': False}

# Set the headers for the request
headers = {'Content-Type': 'application/json'}

# Make the POST request
response = requests.post('http://localhost:5000/predict', data=json.dumps(data), headers=headers)

# Get the response data
prediction = response.json()

# Print the formatted JSON
print(json.dumps(data, indent=4, sort_keys=True))
print(json.dumps(prediction, indent=4, sort_keys=True))
