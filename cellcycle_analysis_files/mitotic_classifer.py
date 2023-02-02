import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV



def df_confusion(label,pred):
    """
    transfer confusion matrix to data frame
    :param label: ground truth label
    :param pred: predicted label
    :return: DataFrame
    """
    return pd.DataFrame(confusion_matrix(label, pred,labels=['G2',"M"]),columns=['pred_G2','pred_M'],index=['ground_true_G2','ground_true_M'])

def check_classifier(clf,x_train,x_test,y_train,y_test):
    """
    check the accuracy of selected classifier with train and test set, also get the confusion matrix to show the prediction label with ground truth label
    :param clf: classifier
    :param x_train: train features
    :param x_test: test features
    :param y_train: ground truth label of training set
    :param y_test:  ground truth label of test set
    :return: hemtmap of confusion matrix
    """
    clf_hp=make_pipeline(StandardScaler(),clf)
    clf_hp.fit(x_train,y_train)
    predicted_train = clf_hp.predict(x_train)
    accuracy_train = accuracy_score(y_train,predicted_train)
    print(f'the accuracy of model run with train data {accuracy_train}')
    predicted_test = clf_hp.predict(x_test)
    accuracy_test = accuracy_score(y_test,predicted_test)
    print(f'the accuracy of model run with test data {accuracy_test}')
    cm_train = df_confusion(y_train,predicted_train)
    cm_test = df_confusion(y_test, predicted_test)
    CM=[cm_train,cm_test]
    title=['train_data','test_data']
    fig, axs= plt.subplots(ncols=1,nrows=2,figsize=(8,5))
    for index,cm  in enumerate(CM):
        axs[index].set_title(f'{title[index]}')
        sns.heatmap(cm, annot=True,fmt='g',ax=axs[index])
    plt.show()

def preprecess_classifier(data,features=[ 'intensity_max_DAPI_nucleus', 'intensity_mean_DAPI_nucleus',
       'integrated_int_DAPI',  'intensity_max_DAPI_cell','intensity_mean_DAPI_cell',
       'intensity_max_Tub_nucleus','intensity_mean_Tub_nucleus', 'intensity_max_Tub_cell','intensity_mean_Tub_cell', 'intensity_max_Tub_cyto','intensity_mean_Tub_cyto',
       'DAPI_total_norm',
       'area_cell_norm',
       'area_nucleus_norm',
       ],label='cell_cycle_detailed'):
    """
    preprocessing of data
    :param data: DataFrame
    :param features: the interesting columns using to classifier
    :param label: ground truth label
    :return: x_train,x_test,y_train,y_test
    """
    X,Y=data[features],data[[label]]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=2/3,random_state=21,shuffle=True,stratify=Y)
    y_train=np.ravel(y_train)
    y_test=np.ravel(y_test)
    # define the classifier
    clf_svm = svm.SVC(random_state=42,C=100,gamma='scale',decision_function_shape='ovo',kernel='rbf')
    # clf_log=LogisticRegression(random_state=42,C=10,solver='newton-cg',max_iter=10000)
    # clf_rf = RandomForestClassifier(random_state=11,n_estimators=40)
    # clf_sgd=SGDClassifier(loss="log_loss", penalty="l2", max_iter=10000)
    x=[i for i in range(1,11)]
    # using the cross validation to compare different  clf  clf_log,clf_rf,clf_sgd

    CLF=[clf_svm]
    for clf in CLF:
        clf_tem=make_pipeline(StandardScaler(),clf)
        score=cross_val_score(clf_tem,x_train,y_train,cv=10)
        plt.plot(x, score, label =f"{clf}")
        print(f'the selected model: {clf}  Score:{score}  Mean:{score.mean()}')
    plt.legend()
    plt.show()
    return x_train,x_test,y_train,y_test

def merge_data(df1,df2,merge_clue_columns='well_id',merge_key_columns=['experiment','plate_id','well_id','cell_line','condition','Cyto_ID','intensity_mean_EdU_cyto','intensity_mean_H3P_cyto','area_cell','area_nucleus',]):
    """
    :param df1: Dataframe, original analysis data
    :param df2: Dataframe, cell cycle data
    :param merge_clue_columns: the columns using to split df2 to multiple dataframes , default 'well_id'
    :param merge_key_columns: key columns using to merge two dataframe,
                             default ['experiment','plate_id','well_id','cell_line','condition','Cyto_ID','intensity_mean_EdU_cyto','intensity_mean_H3P_cyto','area_cell','area_nucleus',]
    :return: merged Dataframe
    """
    all_merged_df=pd.DataFrame()
    for i in df1[merge_clue_columns].unique().tolist():
        # merge two data based on the same well id, hwo=inner:use intersection of keys from both frames, drop NAN columns before merge
        merged_df = pd.merge(df1[df1[merge_clue_columns]==i], df2[df2[merge_clue_columns]==i],how='right',on=merge_key_columns).dropna()
        all_merged_df=pd.concat([all_merged_df,merged_df])
    return all_merged_df
