import typing as t
import numpy as np
import pandas as pd
from marshmallow import fields, Schema, ValidationError

from svm_model import config

class EmailDataSchema(Schema):
    EmailText = fields.Str()
    Label = fields.Str()


def validate_inputs(
    *, input_data: pd.DataFrame
) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:
    """Check model inputs for unprocessable values."""

    data = input_data.copy().dropna()

    # set many=True to allow passing in a list
    schema = EmailDataSchema(many=True)
    errors = None

    try:
        schema.dump(data)
    except ValidationError as exc:
        errors = exc.messages


    return data, errors