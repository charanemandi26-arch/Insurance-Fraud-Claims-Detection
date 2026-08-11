import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from app import app

def run_tests():
    client = app.test_client()
    
    base_payload = {
        'months_as_customer': '228', 'age': '42', 'policy_state': 'OH', 'policy_csl': '250/500',
        'policy_deductable': '1000', 'policy_annual_premium': '1406.91', 'umbrella_limit': '0',
        'insured_zip': '466132', 'insured_sex': 'MALE', 'insured_education_level': 'MD',
        'insured_occupation': 'machine-op-inspct', 'insured_hobbies': 'reading',
        'insured_relationship': 'husband', 'capital-gains': '53300', 'capital-loss': '0',
        'incident_type': 'Single Vehicle Collision', 'collision_type': 'Side Collision',
        'incident_severity': 'Minor Damage', 'authorities_contacted': 'Police',
        'incident_state': 'SC', 'incident_city': 'Columbus', 'incident_hour_of_the_day': '5',
        'number_of_vehicles_involved': '1', 'property_damage': 'YES', 'bodily_injuries': '1',
        'witnesses': '2', 'police_report_available': 'YES', 'total_claim_amount': '71610',
        'injury_claim': '6510', 'property_claim': '13020', 'vehicle_claim': '52080',
        'auto_make': 'Saab', 'auto_model': '92x', 'auto_year': '2004'
    }

    print("TC01 - Valid ordinary claim:")
    res = client.post('/predict', data=base_payload)
    print("Outcome:", "RISK" in res.get_data(as_text=True))
    
    print("\nTC02 - Potentially suspicious claim:")
    suspicious = base_payload.copy()
    suspicious['incident_severity'] = 'Major Damage'
    suspicious['total_claim_amount'] = '112000'
    suspicious['witnesses'] = '0'
    suspicious['authorities_contacted'] = 'Other'
    res = client.post('/predict', data=suspicious)
    print("Outcome:", "RISK" in res.get_data(as_text=True))
    
    print("\nTC03 - Missing input:")
    missing = base_payload.copy()
    del missing['age']
    res = client.post('/predict', data=missing)
    print("Outcome:", "System Error" in res.get_data(as_text=True))
    
    print("\nTC04 - Invalid input:")
    invalid = base_payload.copy()
    invalid['age'] = 'text_instead_of_number'
    res = client.post('/predict', data=invalid)
    print("Outcome:", "System Error" in res.get_data(as_text=True))
    
    print("\nTC05 - Boundary/edge input:")
    boundary = base_payload.copy()
    boundary['age'] = '150'
    boundary['total_claim_amount'] = '99999999'
    res = client.post('/predict', data=boundary)
    print("Outcome:", "RISK" in res.get_data(as_text=True))

if __name__ == '__main__':
    run_tests()
