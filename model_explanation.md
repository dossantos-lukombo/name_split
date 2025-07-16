# Challenge Entretien Technique

## Choix du modèle

J'avais fait l'erreur de prendre le modèle 'LinearRegression' mais j'avais fait un l'absus dans ma tête je voulais bien prendre le model de 'LogisticRegression' qui est un model de Machine Learning qui sert à faire de la classification.

J'ai fait le choix d'utiliser le Vectorizer de scikit-learn, mais pendant l'entretien technique je n'avais immédiatement pensé qu'il fallait que je transforme les prénoms de string en une représentation numérique pour bien entrainer le model de classification.

1. J'ai dans un premier temps choisis de prendre l'index des prénoms, mais ça peut poser problème si on mélange les prénoms.

2. Puis j'ai pensé à convertir les prénoms par des représentation de vecteurs, mais la valeur résultante est trop grande et je pense que c'est overkill pour ce genre de projet.

3. Ensuite je me suis dit que faire une conversion en passant en base26 chaque prénoms mais l'appretissage ne fontionnait pas et là pour cette solution je n'arrive pas à l'expliquer.

4. Et puis on m'a conseillé (chatGPT) d'utiliser tout simplement la class 'TfidfVectorizer' pour trouver les petits schémas dans les prénoms qui peuvent indiquer si le prénom est féminin ou masculin.

## Entrainement

J'ai choisi de séparer le dataset en deux parties : test et train.

30% de test et 70% pour l'entrainement. j'ai pris ces pourcentages car ce sont les plus recommandés pour faire un entrainement avec autant de données: cela permet d'avoir suffisemment de données pour bien entrainer le model et aussi avec suffisamment d'exemple de test.

## Résultat

Score : 80% de bonnes réponses
outil d'analyse : sklearn.metrics : accuracy_score

## Amélioration

Dans le fichier 'scripts/model_training', on peut améliorer l'entrainement du model en découpant le dataset en n morceaux (folds) avec la fonction 'StratifiedKFold', en contrôlant les paramètres pour éliminer certains biais dans chacun des dataset.

En utilisant 'cross_val_score', on peut entrainer le mode selon ces différents morceaux (folds) pour affiner l'entrainement du model.
