from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
def test_dummy():
    X, y = load_iris(return_X_y=True, as_frame=True)
    clf = RandomForestClassifier().fit(X, y)
    assert clf.predict(X).shape[0] == X.shape[0]
