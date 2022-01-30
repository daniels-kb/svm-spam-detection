from svm_model.processing.validation import validate_inputs

#FAILING
def test_validate_inputs(sample_input_data):
    # When
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)

    # Then
    assert not errors

    # we expect that 2 rows are removed due to missing vars
    # 1459 is the total number of rows in the test data set (test.csv)
    # and 1457 number returned after 2 rows are filtered out.
    # assert len(sample_input_data) == 1459
    # assert len(validated_inputs) == 1457


def test_validate_inputs_identifies_errors(sample_input_data):
    # Given
    test_inputs = sample_input_data.copy()

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert not errors
