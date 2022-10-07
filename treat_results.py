import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import seaborn as sns


if __name__ == "__main__":
    f1_neg = joblib.load('results/result_consolidado_f1_neg.pkl')
    accuracy_results = joblib.load('results/resultado_consolidado_accuracy.pkl')
    clustered_resulst = joblib.load('results/resutlado_consolidado_clustering1663534627.pkl')
    classic_results = joblib.load('results/resultado_consolidado_classic.pkl')
    gboost_results = joblib.load('results/resultado_consolidado_gboost.pkl')
    adaboost_results = joblib.load('results/resultado_consolidado_adaboost.pkl')

    precision_results = classic_results[0]
    precision_results.update(gboost_results[0])
    precision_results.update(adaboost_results[0])
    precision_results.pop('NaiveBayes')

    recall_results = classic_results[1]
    recall_results.update(gboost_results[1])
    recall_results.update(adaboost_results[1])
    recall_results.pop('NaiveBayes')

    f1_results = classic_results[2]
    f1_results.update(gboost_results[2])
    f1_results.update(adaboost_results[2])
    f1_results.pop('NaiveBayes')

    precision_results = pd.DataFrame({key: val[0].tolist() for key, val in precision_results.items()})
    precision_results = precision_results[['KNN', 'AdaBoost', 'DecisionTree', 'RandomForest', 'LightGBM']]
    precision_results.boxplot()
    # plt.title("Precisão (Valor preditivo positivo)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/precision_pos")

    recall_results = pd.DataFrame({key: val[0].tolist() for key, val in recall_results.items()})
    recall_results = recall_results[['KNN', 'AdaBoost', 'DecisionTree', 'RandomForest', 'LightGBM']]
    recall_results.boxplot()
    # plt.title("Sensibilidade (Taxa de verdadeiro positivo)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/recall_pos")

    f1_results = pd.DataFrame({key: val[0].tolist() for key, val in f1_results.items()})
    f1_results = f1_results[['KNN', 'AdaBoost', 'DecisionTree', 'RandomForest', 'LightGBM']]
    f1_results.boxplot()
    # plt.title("Medida F1")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/f1_pos")

    print(f1_results.mean())
    print(f1_results.std())


    acc_results = pd.DataFrame({key: val[0].tolist() for key, val in accuracy_results.items()})
    acc_results = acc_results[['KNN', 'AdaBoost', 'DecisionTree', 'RandomForest', 'LightGBM']]
    acc_results.boxplot()
    # plt.title("Medida F1")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/accuracy.png")

    f1_neg = pd.DataFrame({key: val[0].tolist() for key, val in f1_neg.items()})
    f1_neg = f1_neg[['KNN', 'AdaBoost', 'DecisionTree', 'RandomForest', 'LightGBM']]
    f1_neg.boxplot()
    # plt.title("Medida F1")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/f1_neg.png")

    print(f1_results.mean())
    print(f1_results.std())

    f1_results = pd.DataFrame({key: val[0].tolist() for key, val in clustered_resulst[-1].items()})
    f1_results = f1_results[['DecisionTree', 'LightGBM']]
    f1_results.boxplot()
    # plt.title("Medida F1")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/f1_pos_clustered")

    print(f1_results.mean())
    print(f1_results.std())

