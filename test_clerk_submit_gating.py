import web_app

# Test 1: Record at NEEDS_REVIEW with only WARNING checks (no critical fail)
rec_warning = {
    'verification_id': 'test-warn-1',
    'status': 'NEEDS_REVIEW',
    'clerk_submitted': False,
    'document_payload': {
        'document_type': 'Sale Deed',
        'document_number': 'TEST-123',
        'property': {'survey_number': '10', 'area': '100 sq yd', 'village': 'V', 'district': 'D'},
        'stamp_information': {'stamp_number': 'ST-1'},
        'parties': [{'name': 'A', 'role': 'Seller'}, {'name': 'B', 'role': 'Buyer'}]
    },
    'checks': [
        {'status': 'WARNING', 'severity': 'warning', 'name': 'Boundary check'}
    ]
}

# Unsubmitted: button should be enabled!
html_unsub = web_app._clerk_panel(rec_warning, message="", role='clerk')
assert 'value="submit_to_officer"' in html_unsub, "Submit to officer button must be present"
assert 'disabled' not in html_unsub.split('value="submit_to_officer"')[1].split('>')[0], "Submit button must not be disabled for non-critical warning"

# Submitted: should be locked and show pending approval banner!
rec_warning['clerk_submitted'] = True
html_sub = web_app._clerk_panel(rec_warning, message="", role='clerk')
assert 'Record Sent to Officer (Pending Approval)' in html_sub, "Should show pending approval banner"
assert 'Save Corrections' not in html_sub, "Should not show correction button when submitted"
assert 'Submit to Officer' not in html_sub, "Should not show submit button when submitted"

# Test 2: Record with critical failure
rec_critical = {
    'verification_id': 'test-crit-1',
    'status': 'NEEDS_REVIEW',
    'clerk_submitted': False,
    'document_payload': {
        'document_type': 'Sale Deed',
        'document_number': 'TEST-123',
        'property': {'survey_number': '10', 'area': '100 sq yd', 'village': 'V', 'district': 'D'},
        'stamp_information': {'stamp_number': 'ST-1'},
        'parties': [{'name': 'A', 'role': 'Seller'}, {'name': 'B', 'role': 'Buyer'}]
    },
    'checks': [
        {'status': 'FAIL', 'severity': 'critical', 'name': 'Missing Survey Number'}
    ]
}
html_crit = web_app._clerk_panel(rec_critical, message="", role='clerk')
assert 'value="submit_to_officer" class="btn btn-green" disabled' in html_crit, "Button must be disabled for critical fail"

print("PASSED: All clerk submit and lock tests passed successfully!")
