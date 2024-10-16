from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import classification_report

t=['svm','lr','rf']
l=[]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
model_type = {
    'svm':{
        'model' : SVC(gamma='auto'),
        'params':{
        'kernel' : ['linear','rbf'],
        'C' : [1,5,10,20,30]

        }
    },

    'lr' :{
        'model' : LogisticRegression(),
        'params':{
            'C' : [1,5,10,20,30]
        }
    },
    'rf' :{
        'model' : RandomForestClassifier(),
        'params':{
            'n_estimators' :[1,5,10,20,30]
        }
    }



}
from sklearn.model_selection import GridSearchCV




def pick_best_model(train_df,test_df):
    x_train = train_df.iloc[:,:-1]
    y_train = train_df.iloc[:,-1]

    x_test = test_df.iloc[:,:-1]
    y_test = test_df.iloc[:,-1]

    for model,model_params in model_type.items():
        gsv = GridSearchCV(model_params['model'],model_params['params'],cv=5,scoring='recall')
        gsv.fit(x_train,y_train)
        l.append({
            'model': model,
            'score' : gsv.best_score_,
            'bestparamas' : gsv.best_params_,
            'best_estimator' : gsv.best_estimator_,
            })
    l.sort(key =lambda x:x['score'])
    best_estimator = l[-1]['best_estimator'] 
    y_pred = best_estimator.predict(x_test)
    print(y_test,y_pred)
    return classification_report(y_true=y_test,y_pred=y_pred),best_estimator

    
    


  
  




