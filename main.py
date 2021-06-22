from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import accuracy_score
from strlearn.ensembles import  SEA, WAE, AUE, AWE
from strlearn.metrics import f1_score, geometric_mean_score_1, balanced_accuracy_score
from AWE import AWE_OUR
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate

random_state=1111
clfs = {
    'SEA': SEA(base_estimator=GaussianNB(), n_estimators=5),
    'AUE': AUE(base_estimator=GaussianNB(), n_estimators=5),
    'WAE': WAE(base_estimator=GaussianNB(), n_estimators=5),
    'AWE' : AWE(base_estimator=GaussianNB(), n_estimators=5),
    'AWE_OUR': AWE_OUR(normalize=True, random_state=random_state, n_estimators=5)
}
  
names_column = np.array([["SEA"], ["AUE"], ["WAE"], ["AWE"], ["AWE_OUR"]])
accuracy_header = ["F1_score", "G-mean", "Balanced accuracy", "accuracy"]

n_chunks =200              
chunk_size = 250            
n_drifts = 3                
n_features = 10             
n_classes = 2               

metrics = [
    f1_score,
    geometric_mean_score_1,
    balanced_accuracy_score,
    accuracy_score
]

alfa = .05 

streams = {
    "strumien z dryftem nagłym" : StreamGenerator(n_chunks=n_chunks,
                                                   chunk_size=chunk_size,
                                                   random_state=random_state,
                                                   n_features=n_features,
                                                   n_classes=n_classes,
                                                   n_drifts=n_drifts),

    "strumien z dryftem dualnym" : StreamGenerator(n_chunks=n_chunks,
                                                   chunk_size=chunk_size,
                                                   random_state=random_state,
                                                   n_features=n_features,
                                                   n_classes=n_classes,
                                                   n_drifts=n_drifts,
                                                   concept_sigmoid_spacing=5),

    "strumien z dryftem inkrementalnym" : StreamGenerator(n_chunks=n_chunks,
                                                   chunk_size=chunk_size,
                                                   random_state=random_state,
                                                   n_features=n_features,
                                                   n_classes=n_classes,
                                                   n_drifts=n_drifts,
                                                   concept_sigmoid_spacing=5,
                                                   incremental=True)
} 


def create_table(scores):
    mean_accuracy_table = np.concatenate((names_column, scores), axis=1)
    mean_accuracy_table = tabulate(mean_accuracy_table, accuracy_header, floatfmt=".3f")
    return mean_accuracy_table 

def compare_models(scores):
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for y in range(len(clfs)):
        for x in range(len(clfs)):
            t_statistic[y, x], p_value[y, x] = ttest_ind(scores[y], scores[x])

    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, clfs.keys(), floatfmt=".3f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, clfs.keys(), floatfmt=".3f")

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), clfs.keys())

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), clfs.keys())

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), clfs.keys())
    return t_statistic_table, p_value_table, advantage_table, significance_table, stat_better_table




scores = np.zeros((len(streams), len(clfs), len(metrics)))
evaluator = TestThenTrain(metrics, verbose=True)
for id_stream, key in enumerate(streams):
    evaluator.process(streams[key], tuple(clfs.values()))
    
    fig, ax = plt.subplots(len(clfs), figsize=(10,20))
    
    for id_clf, clf_name in enumerate(clfs):
        a = ax[id_clf]
        for metric_id in range(len(metrics)):
            scores[id_stream, id_clf, metric_id] = np.mean(evaluator.scores[id_clf,:,metric_id], axis=0)
            a.plot(evaluator.scores[id_clf, :, metric_id], label=metrics[metric_id].__name__)

        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlabel('chunk', fontsize=8)
        a.set_ylabel('quality', fontsize=8)
        a.grid(ls=":")
        a.legend()
        a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        a.set_title(clf_name)
        a.set_xlim(xmin=0, xmax=n_chunks)
    

        plt.tight_layout()
        plt.savefig(f'scalers{id_stream}.png')

table_1 = create_table(scores[0,:,:])
print("Średnia dokładność predykcji algorytmów dla strumienia nagłego:\n", table_1, "\n")

table_t, table_p, advantage, significance, state = compare_models(scores[0,:,:])
print("t-statistic:\n", table_t, "\n\np-value:\n", table_p, "\n", "\n\nadvantage:\n", advantage, "\n\nsignificance:\n", significance)
print("\n\nStatystycznie lepszy:\n", state)


table_1 = create_table(scores[1,:,:])
print("Średnia dokładność predykcji algorytmów dla strumienia gradualnego:\n", table_1, "\n")

table_t, table_p, advantage, significance, state = compare_models(scores[1,:,:])
print("t-statistic:\n", table_t, "\n\np-value:\n", table_p, "\n", "\n\nadvantage:\n", advantage, "\n\nsignificance:\n", significance)
print("\n\nStatystycznie lepszy:\n", state)


table_1 = create_table(scores[2,:,:])
print("Średnia dokładność predykcji algorytmów dla strumienia gradualnego:\n", table_1, "\n")

table_t, table_p, advantage, significance, state = compare_models(scores[2,:,:])
print("t-statistic:\n", table_t, "\n\np-value:\n", table_p, "\n", "\n\nadvantage:\n", advantage, "\n\nsignificance:\n", significance)
print("\n\nStatystycznie lepszy:\n", state)
