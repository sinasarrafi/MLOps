from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

def data_preparation(test_size: float = 0.2, random_state: int = 42):
    """Load and split the Heart Disease dataset into train/test sets."""
    heart = fetch_ucirepo(id=45)
    X = heart.data.features.dropna()
    y = heart.data.targets.squeeze().loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test