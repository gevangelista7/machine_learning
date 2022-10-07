import pandas as pd
import joblib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    justaposto = joblib.load('results/resutlado_consolidado_clusterized_FINAL_1663641705.pkl')
    membership = joblib.load('results/resutlado_consolidado_clusterized_somente_membership_1663647864.pkl')

    f1_pos = pd.DataFrame()

    f1_pos['DecisionTree \n Pertinencia'] = membership['f1_pos']['DecisionTree'][0]
    f1_pos['LGBM \n Pertinencia'] = membership['f1_pos']['LightGBM'][0]
    f1_pos['DecisionTree \n Justaposto'] = justaposto['f1_pos']['DecisionTree'][0]
    f1_pos['LGBM \n Justaposto'] = justaposto['f1_pos']['LightGBM'][0]

    f1_pos.boxplot()
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/f1_pos_cluster")

    acc = pd.DataFrame()

    acc['DecisionTree \n Pertinencia'] = membership['accuracy']['DecisionTree'][0]
    acc['LGBM \n Pertinencia'] = membership['accuracy']['LightGBM'][0]
    acc['DecisionTree \n Justaposto'] = justaposto['accuracy']['DecisionTree'][0]
    acc['LGBM \n Justaposto'] = justaposto['accuracy']['LightGBM'][0]

    acc.boxplot()
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/f1_pos_cluster")

