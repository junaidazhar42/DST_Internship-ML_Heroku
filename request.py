import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Days_spend_hsptl':2, 'ccs_diagnosis_code':9, 'ccs_procedure_code':6})

print(r.json())