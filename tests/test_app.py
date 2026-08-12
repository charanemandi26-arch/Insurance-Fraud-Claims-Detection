import sys
import os

# Add app directory to path so we can import the Flask app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
from app import app

# Minimal valid payload — exactly the 22 model columns the backend expects
BASE_PAYLOAD = {
    # Numerical
    'months_as_customer': '228',
    'age': '42',
    'policy_deductable': '1000',
    'policy_annual_premium': '1406.91',
    'umbrella_limit': '0',
    'incident_hour_of_the_day': '5',
    'number_of_vehicles_involved': '1',
    'bodily_injuries': '1',
    'witnesses': '2',
    'total_claim_amount': '71610',
    'injury_claim': '6510',
    'property_claim': '13020',
    'vehicle_claim': '52080',
    'auto_year': '2004',
    # Categorical
    'policy_csl': '250/500',
    'incident_type': 'Single Vehicle Collision',
    'collision_type': 'Side Collision',
    'incident_severity': 'Minor Damage',
    'authorities_contacted': 'Police',
    'property_damage': 'YES',
    'police_report_available': 'YES',
    'auto_make': 'Saab',
}

def run_tests():
    client  = app.test_client()
    passed  = 0
    total   = 5

    # TC01 — Valid ordinary claim: expect a RISK level in the result
    print('TC01 - Valid ordinary claim:')
    res  = client.post('/predict', data=BASE_PAYLOAD)
    body = res.get_data(as_text=True)
    ok   = 'RISK' in body and res.status_code == 200
    print(f'  Status: {res.status_code}  |  RISK in response: {"RISK" in body}  |  {"PASS" if ok else "FAIL"}')
    if ok: passed += 1

    # TC02 — Suspicious-looking claim: should still return a RISK level (HIGH expected)
    print('\nTC02 - Suspicious-looking claim:')
    suspicious = BASE_PAYLOAD.copy()
    suspicious.update({
        'incident_severity': 'Major Damage',
        'total_claim_amount': '112000',
        'witnesses': '0',
        'police_report_available': 'NO',
        'authorities_contacted': 'Other',
    })
    res  = client.post('/predict', data=suspicious)
    body = res.get_data(as_text=True)
    ok   = 'RISK' in body and res.status_code == 200
    print(f'  Status: {res.status_code}  |  RISK in response: {"RISK" in body}  |  {"PASS" if ok else "FAIL"}')
    if ok: passed += 1

    # TC03 — Missing required field: expect user-friendly error, no traceback
    print('\nTC03 - Missing required field (age removed):')
    missing = BASE_PAYLOAD.copy()
    del missing['age']
    res  = client.post('/predict', data=missing)
    body = res.get_data(as_text=True)
    ok   = res.status_code == 200 and 'error' in body.lower()
    print(f'  Status: {res.status_code}  |  Error message shown: {ok}  |  {"PASS" if ok else "FAIL"}')
    if ok: passed += 1

    # TC04 — Invalid numeric input (text in age field): expect user-friendly error
    print('\nTC04 - Invalid numeric input (age = notanumber):')
    invalid = BASE_PAYLOAD.copy()
    invalid['age'] = 'notanumber'
    res  = client.post('/predict', data=invalid)
    body = res.get_data(as_text=True)
    ok   = res.status_code == 200 and 'error' in body.lower()
    print(f'  Status: {res.status_code}  |  Error message shown: {ok}  |  {"PASS" if ok else "FAIL"}')
    if ok: passed += 1

    # TC05 — Out-of-range age triggers range validation
    print('\nTC05 - Age out-of-range boundary (age=150):')
    boundary = BASE_PAYLOAD.copy()
    boundary['age'] = '150'
    res  = client.post('/predict', data=boundary)
    body = res.get_data(as_text=True)
    ok   = res.status_code == 200 and 'error' in body.lower()
    print(f'  Status: {res.status_code}  |  Validation error shown: {ok}  |  {"PASS" if ok else "FAIL"}')
    if ok: passed += 1

    print(f'\n{"="*40}')
    print(f'  Results: {passed}/{total} tests passed')
    print(f'{"="*40}')
    return passed == total


if __name__ == '__main__':
    run_tests()
