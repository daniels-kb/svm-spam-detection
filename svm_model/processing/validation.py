import typing as t

from svm_model import config

import numpy as np
import pandas as pd
from marshmallow import fields, Schema, ValidationError


class EmailDataSchema(Schema):
    EmailText = fields.Str()
    Label = fields.Str()


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    if input_data[config.FEATURES, config.TARGET].isnull().any().any():
        validated_data = validated_data.dropna(axis=0)

    return validated_data


def validate_inputs(
    *, input_data: pd.DataFrame
) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:
    """Check model inputs for unprocessable values."""

    data = drop_na_inputs(input_data=input_data)

    # set many=True to allow passing in a list
    schema = EmailDataSchema(many=True)
    errors = None

    try:
        schema.load(data)
    except ValidationError as exc:
        errors = exc.messages

    return data, errors
