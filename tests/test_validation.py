from svm_model import config
from svm_model.processing.validation import validate_inputs

'''
def test_validate_inputs_identifies_errors(sample_input_data):
    # Given
    test_inputs = sample_input_data.copy()

    # introduce errors intentionally
    test_inputs.at[1, config.FEATURES] = 50  # string expected

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors
    assert errors[1] == {"EmailText": ["Not a valid string."]}
 ''' 
    
def test_validate_inputs(sample_input_data):
    # When
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)

    # Then
    assert not errors
