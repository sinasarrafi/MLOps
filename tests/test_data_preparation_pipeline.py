from src.data_preparation import data_preparation


def test_data_preparation_pipeline():
    X_train, X_test, y_train, y_test = data_preparation(test_size=0.2)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


# Run with `python -m pytest tests`
