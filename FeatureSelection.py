from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib as plt

from Preprocessing import Preprocessing

# columns
stud_id_col = ['ITEST_id']
label_col = ['isSTEM']

cat_cols = ['skill', 'problemType', 'SY ASSISTments Usage']
per_action_cols = ['Prev5count', 'skill', 'problemType', 'endTime', 'timeTaken', 'correct', 'original', 'hint',
                   'hintCount',
                   'hintTotal', 'scaffold', 'bottomHint', 'attemptCount', 'frIsHelpRequest',
                   'frPast5HelpRequest',
                   'frPast8HelpRequest',
                   'stlHintUsed',
                   'past8BottomOut',
                   'totalFrPercentPastWrong',
                   'totalFrPastWrongCount',
                   'frPast5WrongCount',
                   'frPast8WrongCount',
                   'totalFrTimeOnSkill',
                   'timeSinceSkill',
                   'frWorkingInSchool',
                   'totalFrAttempted',
                   'totalFrSkillOpportunities',
                   'responseIsFillIn',
                   'responseIsChosen',
                   'endsWithScaffolding',
                   'endsWithAutoScaffolding',
                   'frTimeTakenOnScaffolding',
                   'frTotalSkillOpportunitiesScaffolding',
                   'totalFrSkillOpportunitiesByScaffolding',
                   'frIsHelpRequestScaffolding',
                   'timeGreater5Secprev2wrong',
                   'sumRight',
                   'helpAccessUnder2Sec',
                   'timeGreater10SecAndNextActionRight',
                   'consecutiveErrorsInRow',
                   'sumTime3SDWhen3RowRight',
                   'sumTimePerSkill',
                   'totalTimeByPercentCorrectForskill',
                   'prev5count',
                   'timeOver80',
                   'manywrong',
                   'RES_BORED',
                   'RES_CONCENTRATING',
                   'RES_CONFUSED',
                   'RES_FRUSTRATED',
                   'RES_OFFTASK',
                   'RES_GAMING',
                   ]
per_stud_cols = ['SY ASSISTments Usage', 'NumActions', 'SchoolId', 'AveKnow',
                 'AveCarelessness'] + label_col

id_cols = ['actionId', 'problemId', 'assignmentId', 'assistmentId']
conf_cols = ['confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)', 'confidence(FRUSTRATED)',
             'confidence(OFF TASK)', 'confidence(GAMING)']
ave_res_cols = ['AveCorrect', 'AveResBored', 'AveResEngcon', 'AveResConf', 'AveResFrust', 'AveResOfftask',
                'AveResGaming']
excluded_cols = ['MCAS', 'Ln-1'] + ave_res_cols + id_cols + conf_cols + cat_cols

pre = Preprocessing()

X, y = pre.load_data(summarize=True)

scaler = RobustScaler()

scaled_features = scaler.fit_transform(X)

# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(scaled_features, y)

# # summarize scores
# df = DataFrame({'score':fit.scores_, 'feature':col_names})
# df.sort_values(by='score', inplace=True)
# print(df)

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=10, cv=StratifiedKFold(2), scoring='accuracy', n_jobs=-1, verbose=1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
print("here")
