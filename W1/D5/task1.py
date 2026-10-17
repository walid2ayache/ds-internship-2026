import numpy as np
def train_test_split(X,y,test_size=0.2,random_state=None):
    n=len(X)
    assert len(X)==len(y)
    assert 0.0<test_size<1.0
    rng=np.random.default_rng(random_state)
    indices=np.arange(n)
    rng.shuffle(indices)
    n_test=int(n*test_size)
    test_idx=indices[:n_test]
    train_idx=indices[n_test:]
    return X[train_idx],X[test_idx],y[train_idx],y[test_idx]
    