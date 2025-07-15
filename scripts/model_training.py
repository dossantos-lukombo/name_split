from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def training_cv(X,y):

    # 2) Prépare ton splitter stratifié
    skf = StratifiedKFold(
        n_splits=5,      # 5 folds
        shuffle=True,    # mélange avant de split
        random_state=42  # reproductibilité
    )

    # 3) Instancie ton modèle
    model = LogisticRegression(
        max_iter=1000,   # pour être sûr de converger
        solver='lbfgs',  # par défaut
        random_state=42
    )

    # 4) Évalue ta LR avec cross_val_score
    scores = cross_val_score(
        estimator = model,   # ton modèle
        X         = X,       # features
        y         = y,       # target
        cv        = skf,     # ton StratifiedKFold
        scoring   = 'accuracy',  # métrique (ici accuracy)
        n_jobs    = -1       # parallélisation
    )
    
    print(scores)
    
    return model