import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek

import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils

sepscores = []
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
#parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
#parameters = {'kernel': ['rbf'], 'C':[2**-15: 2**15],'gamma':[0.001, 0.0001]}
#parameters = {'kernel': ['rbf'], 'C':[1, 10, 100, 1000], 'gamma':[0.001, 0.0001]}
#clf = GridSearchCV(svm.SVC(), parameters, cv=5, n_jobs=12, scoring='accuracy')
#C_range = 2. ** np.arange(-15, 15)
#gamma_range = 2. ** np.arange(-15, -5)
#param_grid = dict(gamma=gamma_range, C=C_range)
#clf = GridSearchCV(SVC(), param_grid=param_grid, cv=10)
#clf.fit(X,Y)
#C=clf.best_params_['C']
#gamma=clf.best_params_['gamma']

def training(X, y):
     sepscores = []
     ypredict=[]
     ytest = np.ones((1, 2)) * 0.5
     yscore = np.ones((1, 2)) * 0.5
     #C_range = 2. ** np.arange(-15, 15)
     #gamma_range = 2. ** np.arange(-15, -5)
     #param_grid = dict(gamma=gamma_range, C=C_range)
     #clf = GridSearchCV(SVC(), param_grid=param_grid, cv=10)
     #clf.fit(X, y)
     #C = clf.best_params_['C']
     #gamma = clf.best_params_['gamma']
     #print(C, gamma)
     skf = StratifiedKFold(n_splits=10, shuffle=False)
     for train, test in skf.split(X, y):
          svc = svm.SVC( kernel='poly', C=2, gamma=0.0005, probability=True)
          # y_train = utils.to_categorical(y[train])
          y_train = utils.to_categorical(y[train])
          hist = svc.fit(X[train], y[train])
          # hist=svc.fit_transform(X[train], y[train])
          y_score = svc.predict_proba(X[test])
          y_predict= svc.predict(X[test])
          ypredict = np.concatenate((ypredict, y_predict))
          yscore = np.vstack((yscore, y_score))
          y_test = utils.to_categorical(y[test])
          ytest = np.vstack((ytest, y_test))
          fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
          roc_auc = auc(fpr, tpr)
          y_class = utils.categorical_probas_to_classes(y_score)
          y_test_tmp = y[test]
          acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                              y_test_tmp)
          sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
          print('SVM:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
                % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))
     scores = np.array(sepscores)
     print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
     print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
     print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
     print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
     print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
     print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
     print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
     print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))
     result1 = np.mean(scores, axis=0)
     H1 = result1.tolist()
     sepscores.append(H1)
     result = sepscores
     row = yscore.shape[0]
     ypredict_sum = pd.DataFrame(data=ypredict)
     ypredict_sum.to_csv('train\Train_ypredict_SVM.csv')
     yscore = yscore[np.array(range(1, row)), :]
     yscore_sum = pd.DataFrame(data=yscore)
     yscore_sum.to_csv('train\MARSA_Train_DDE_SVM_yscore.csv')
     ytest = ytest[np.array(range(1, row)), :]
     ytest_sum = pd.DataFrame(data=ytest)
     ytest_sum.to_csv('train\MARSA_Train_DDE_SVM_ytest.csv')
     fpr, tpr, _ = roc_curve(ytest[:, 0], yscore[:, 0])
     auc_score = np.mean(scores, axis=0)[7]
     lw = 2
     plt.plot(fpr, tpr, color='darkorange',
              lw=lw, label='SVM ROC (area = %0.2f%%)' % auc_score)
     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
     plt.xlim([0.0, 1.05])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('Receiver operating characteristic')
     plt.legend(loc="lower right")
     plt.grid()
     plt.show()
     plt.savefig('MARSA_Train_DDE_SVM.png', dpi=300)
     colum = ['ACC', 'precision', 'npv', 'Sn', 'Sp', 'MCC', 'F1', 'AUC']
     ro = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
     data_csv = pd.DataFrame(columns=colum, data=result, index=ro)
     data_csv.to_csv(r'train\MARSA_Train_DDE_SVM_results.csv')


def dumpmodel(X, y):
    svc = svm.SVC( kernel='poly', C=5, gamma=0.000625, probability=True)
    #yy = utils.to_categorical(y)
    svc.fit(X, y)
    #joblib.dump(svc, 'model\saved_model_PELM_RF.pkl')
    pickle.dump(svc, open('model\MARSA_DDE_SVMModle.pkl', 'wb'))
    print("Saved model to disk")

def indepTesting(Xtest, Ytest):
     Sepscores = []
     Ypredict=[]
     ytest = np.ones((1, 2)) * 0.5
     yscore = np.ones((1, 2)) * 0.5
     # xtest=np.asarray(pd.read_csv(Xtest).iloc[:].values)
     # ytest=np.asarray(pd.read_csv(Xtest).iloc[:].values.ravel())
     xtest = np.vstack(Xtest)
     y_test = np.vstack(Ytest)
     ldmodel = pickle.load(open("model\MARSA_DDE_SVMModle.pkl", 'rb'))
     print("Loaded model from disk")
     y_score = ldmodel.predict_proba(xtest)
     Y_predict= ldmodel.predict(xtest)
     Ypredict = np.concatenate((Ypredict, Y_predict))
     yscore = np.vstack((yscore, y_score))
     y_class = utils.categorical_probas_to_classes(y_score)

     fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
     roc_auc = auc(fpr, tpr)
     acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                         y_test)
     Sepscores.append([acc, precision,npv,sensitivity, specificity, mcc,f1,roc_auc])
     print('SVM:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
           % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))
     fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 1])
     auc_score = auc(fpr, tpr)
     scores = np.array(Sepscores)
     result1 = np.mean(scores, axis=0)
     H1 = result1.tolist()
     Sepscores.append(H1)
     result = Sepscores
     row = y_score.shape[0]
     ypredict_sum = pd.DataFrame(data=Ypredict)
     ypredict_sum.to_csv('ind\Test_ypredict_SVM.csv')
     yscore = y_score[np.array(range(1, row)), :]
     yscore_sum = pd.DataFrame(data=yscore)
     yscore_sum.to_csv('ind\MARSA_Test_DDE_SVM_yscore.csv')
     y_test = y_test[np.array(range(1, row)), :]
     ytest_sum = pd.DataFrame(data=y_test)
     ytest_sum.to_csv('ind\MARSA_Test_DDE_SVM_ytest.csv')
     colum = ['ACC', 'precision', 'npv', 'Sn', 'Sp', 'MCC', 'F1', 'AUC']
     # ro = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
     data_csv = pd.DataFrame(columns=colum, data=result)  # , index=ro)
     data_csv.to_csv('ind\MARSA_Test_DDE_SVM_results.csv')
     lw = 2
     plt.plot(fpr, tpr, color='blue',
              lw=lw, label='SVM ROC (area = %0.2f%%)' % auc_score)
     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
     plt.xlim([0.0, 1.05])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('Receiver operating characteristic')
     plt.legend(loc="lower right")
     plt.grid()
     plt.show()
     plt.savefig('MARSA_Test_DDE_SVM.png', dpi=300)


if __name__ == '__main__':
     ###data for training#######
     data_train = sio.loadmat('TR_SCMRSA_DDE.mat')
     data = data_train.get('TR_SCMRSA_DDE')  # Remove the data in the dictionary
     label1 = np.ones((118 , 1))  # Value can be changed
     label2 = np.zeros((678, 1))
     label = np.append(label1, label2)
     X = data
     y = label
     #sm = SMOTE(random_state=2)
     #sm = SMOTE() # this give best results
     #sme = SMOTEENN()
     #ru = RandomUnderSampler(sampling_strategy=0.5)
     #ada = ADASYN()
     #svmsmo = SVMSMOTE()
     #bordersm= BorderlineSMOTE()
     #smotemok= SMOTETomek()
     #X_res, y_res = smotemok.fit_resample(X,y)


     #X = X_res
     #y = y_res


###############################2nd Loadiing method###
     ############data for testing#############
     data_test = sio.loadmat('TS_SCMRSA_DDE.mat')
     data_test = data_test.get('TS_SCMRSA_DDE')  # Remove the data in the dictionary
     label1_test = np.ones((30, 1))  # Value can be changed
     label2_test = np.zeros((169, 1))
     label_test = np.append(label1_test, label2_test)
     indepXtest = data_test
     indepYtest = label_test



     #########function calling###########
     training(X, y)
     dumpmodel(X, y)
     indepTesting(indepXtest, indepYtest)
