from sklearn.base import ClassifierMixin, clone
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split 
from sklearn.exceptions import NotFittedError
from sklearn.utils import shuffle

class AWE_OUR(ClassifierMixin, BaseEnsemble):

    def __init__(self, base_estimator=GaussianNB(), normalize=False, n_estimators=5, 
                 n_splits=5, random_state=None):
        self.base_estimator = base_estimator 
        self.n_estimators = n_estimators 
        self.n_splits = n_splits 
        self.random_state = random_state 
        self.normalize = normalize 
        self.shape = None
        self.classes_=None
        np.random.seed(self.random_state)

    def __compute_mser(self, y):
        _, numbers = np.unique(y, return_counts=True)
        posterior = numbers/len(y)
        return np.sum([(1-x)**2  for x in posterior])
    
    def __compute_msei(self, clf, X, y):
        probas = clf.predict_proba(X)
        scores = np.zeros(len(y))
        for label in self.classes_:
            scores[y == label] = probas[y==label, label]
        return np.sum([(1-score)**2 for score in scores])/len(y)
        


    def __normalize_X(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X


    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, "ensemble"):
            self.ensemble = []
        X, y = check_X_y(X, y)
        if not self.shape:
            self.shape = X.shape[1]
        
        if self.shape != X.shape[1]:
            raise Exception("chunks do not have the same size!")
       
        if not hasattr(self, "ensemble"):
            self.ensemble = []


        self.X, self.y = X, y

        
        if self.normalize:
            self.X = self.__normalize_X(self.X)

        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)
        
        # train_X, test_X, train_y, test_y = train_test_split(self.X, self.y, test_size=0.3, random_state=self.random_state)
        clf = clone(self.base_estimator).fit(self.X, self.y)
        
        #msei = self.__compute_msei(clf, test_X, test_y)
        mser = self.__compute_mser(self.y)
        
        mseis = np.zeros(self.n_splits)
        kf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        
        for id_fold, (train, test) in enumerate(kf.split(self.X, self.y)):
            fold_clf = clone(self.base_estimator).fit(self.X[train], self.y[train])
            msei = self.__compute_msei(fold_clf, self.X[test], self.y[test])
            mseis[id_fold] = msei
        
        
        new_weight = mser - np.mean(mseis)

        self.weights = [mser - self.__compute_msei(clf, self.X, self.y ) for clf in self.ensemble]
        self.ensemble.append(clf)
        self.weights.append(new_weight)
     
        if len(self.ensemble) > self.n_estimators:
            worst = np.argmin(self.weights)
            del self.ensemble[worst]
            del self.weights[worst]

    def __ensemble_suport_matrix(self, X):
        probas = []
        for i, member_clf in enumerate(self.ensemble):
            probas.append(member_clf.predict_proba(X))
        return np.array(probas)
    
    def predict(self, X):
        try: 
            X = check_array(X)
            if self.shape != X.shape[1]:
                raise Exception('chunks do not have the same size!')

            if self.normalize:
                X = self.__normalize_X(X)
     
            esm = self.__ensemble_suport_matrix(X)
            avg_esm = np.mean(esm, axis=0)
            predict = np.argmax(avg_esm, axis=1)

            return self.classes_[predict]
        except NotFittedError as e:
            raise Exception("Predict error: "+str(e))
        
        









