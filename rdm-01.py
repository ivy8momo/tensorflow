# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# 加载数据
data= pd.read_csv('C:/Users/hey/Desktop/assessment.csv')
print(data.columns.values)
print(data.dtypes)

# 接着选择好样本特征和类别输出，样本特征为除去ID和输出类别的列
x_columns = [x for x in data.columns if x not in ['orig_id', 'group']]
x = data[x_columns]
y = data['group']

#拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=123)

#分类型决策树
clf = RandomForestClassifier( random_state=71)

#训练模型
s = clf.fit(x_train , y_train)
print(s)
y_pred = clf.predict_proba(x_train)[:,1]
metrics.roc_auc_score(y_train,y_pred)

#评估模型准确率
r = clf.score(x_test,y_test)
#直接给出预测结果，每个点在所有label的概率和为1，内部还是调用predict——proba()
predict_y_validation = clf.predict(x_test)
#给出带有概率值的结果，每个点所有label的概率和为1
prob_predict_y_validation = clf.predict_proba(x_test)
predictions_validation = prob_predict_y_validation[:, 1]
fpr, tpr, _ = roc_curve(y_test, predictions_validation)
roc_auc = auc(fpr, tpr)
plt.title('ROC Validation')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#保存树
joblib.dump(clf, 'filename.pkl')

# 读取树： clf = joblib.load('filename.pkl')


from sklearn.model_selection import GridSearchCV

# 调整n_estimators 最大的弱学习器的个数，与learning_rate一起考虑
param_test1 = {'n_estimators':range(10,201,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(),
                        param_grid = param_test1,
                        scoring='roc_auc',
                        cv=5)
gsearch1.fit(x,y)
gsearch1.scorer_,gsearch1.best_params_, gsearch1.best_score_

# max_depth 特征少时可以不调整，默认不限制深度
# min_samples_split
# min_samples_leaf 限制叶子节点最少的样本数
# max_features
param_test2 = {'max_depth':range(1,21,1),
               'min_samples_split':range(10,100,10),
               'min_samples_leaf':range(10,100,10),
               'max_features': range(3, 39, 2)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=100),
                        param_grid = param_test2,
                        scoring='roc_auc',
                        cv=5)
gsearch2.fit(x,y)
gsearch2.scorer_,gsearch2.best_params_, gsearch2.best_score_

# 最佳模型袋外误差
rf1= RandomForestClassifier(n_estimators= 100,
                            max_depth=15,
                            max_features= 6,
                            min_samples_split=8,
                            min_samples_leaf=10,
                            oob_score=True,
                            random_state=123)
rf1.fit(x,y)
print(rf1.oob_score_)