# SpaceshipTitanic

### import
***
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
***
### Dataframe
![image](https://user-images.githubusercontent.com/116700688/233049509-591ef56b-167e-42c2-a042-7a9731be962d.png)
***
### 테이블 컬럼 내용
***
- PassengerId  
승객 고유 ID. gggg-pp 형태로 gggg는 그룹, pp는 그룹 내 번호. 그룹이 같으면 가족인 경우가 많음
- HomePlanet  고향 행성
- CryoSleep  
냉동수면 상태 여부. 냉동수면 상태인 탑승객은 객실에 갇혀있음
- Cabin  
객실 번호. deck/num/side 형태로 side는 P(Port) 혹은 S(Starboard)
- Destination  
목적 행성
- Age  
나이
- VIP  
VIP 서비스 신청했는지 여부
- RoomService, FoodCourt, ShoppingMall, Spa, VRDeck  
우주선 내 있는 해당 편의시설에 지불한 금액
- Name  
이름과 성
- Transported  
다른 차원으로 전이됐는지 여부. target
***
### Data 분포 확인
---
* HomePlanet

![image](https://user-images.githubusercontent.com/116700688/233046039-a964eab1-0f89-4a60-9e8b-b7ea414a2851.png)

---
* CryoSleep

![image](https://user-images.githubusercontent.com/116700688/233047152-23cf7101-066d-4d08-bb8a-92fc9c4996b0.png)
---
* Destination

![image](https://user-images.githubusercontent.com/116700688/233047215-752b60ca-b36a-4fdc-8535-115db6ef20c4.png)
---
* 편의시설 돈 쓴 여부

![image](https://user-images.githubusercontent.com/116700688/233047287-0fa945ed-a7e9-4e4e-a39b-7cde9d241c75.png)
---
* 연령대는 0세부터 79세까지 분포하기 때문에 10대, 20대와 같이 범위를 나누었습니다.
* 0-19세까지는 도착률이 더 높고, 20-39세는 도착률이 더 낮으며 40세 이후로는 도착률이 거의 동일한 것을 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/116700688/233047492-0eca6c3a-5a9a-4bd2-b808-f53d4e45feac.png)
---
### Permutation Feature Importance 사용
```
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(lgbm_model1, random_state=42).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist(), top=25)
```

![image](https://user-images.githubusercontent.com/116700688/233050353-fbc7be74-e2c4-4d6a-8d2b-e224d47f7965.png)
---

### HyperOpt 사용한 하이퍼파라미터 최적화
```
from hyperopt import hp
```
### Random Forest
### 검색 공간 설정
```
rfc_search_space = {'max_depth': hp.quniform('max_depth', 5, 15, 1),
                    'min_samples_leaf': hp.quniform ('min_samples_leaf', 1, 4, 1),
                    'min_samples_split' : hp.quniform ('min_samples_split', 2, 4, 1),
                    'n_estimators' : hp.quniform('n_estimators', 10, 100, 10)}
```
### LGBM 검색 공간 설정
```
lgbm_search_space = {'n_estimators' : hp.quniform('n_estimators', 100, 1000, 100),
                     'max_depth' : hp.quniform('max_depth', 5, 15, 1),                 # 5부터 20까지 간격 1
                     'num_leaves' : hp.quniform('num_leaves', 5, 15, 1),
                     'min_child_samples' : hp.quniform('min_child_samples', 10, 30, 2),    # 1부터 1까지 간격 1
                     'learning_rate' : hp.uniform('learning_rate', 0.01, 0.1),         # 0.01부터 0.2까지 균등분포
                     'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1)}      # 0.5부터 1까지 균등분포
```

---
### 목적함수(=손실함수) 정의
```
from sklearn.model_selection import cross_val_score
from hyperopt import STATUS_OK

def objective_func(search_space):
    rfc_hp = RandomForestClassifier(max_depth = int(search_space['max_depth']),
                                    min_samples_leaf = int(search_space['min_samples_leaf']),
                                    min_samples_split = int(search_space['min_samples_split']))

    
    accuracy = cross_val_score(rfc_hp, X_train, y_train, scoring='accuracy', cv = 3)

    # accuracy는 높을수록 좋기 때문에 -1을 곱해줌
    return {'loss': -1 * np.mean(accuracy), 'status': STATUS_OK }
```
---

```
from hyperopt import fmin, tpe, Trials

# 반복 결과를 저장할 변수 생성
trial_val = Trials()

# 100회 반복
best = fmin(fn = objective_func,
            space = rfc_search_space,
            algo = tpe.suggest,     # 최적화에 적용할 알고리즘 기본값
            max_evals = 100,
            trials = trial_val)

print('best :', best)
```

---

### randomforest hyperopt

![image](https://user-images.githubusercontent.com/116700688/233058596-10be84b3-e26e-429e-9d25-bb74163cca6f.png)

### LGBM hyperopt

![image](https://user-images.githubusercontent.com/116700688/233058731-a6bd5192-875b-4d6f-9cc8-181aeee1d5f7.png)

---

# Grid Search 를 활용한 하이퍼파라미터 최적화
---
### Random Forest
```
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_estimators=100, random_state=100, n_jobs=-1)
param_grid = {'n_estimators': range(1, 20), 'max_depth': [8,16,24], "min_samples_leaf" : [1, 6, 12], 'min_samples_split' : [2, 8, 16]}
grid_cv = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_cv.fit(X_train, y_train)

# 그리드 서치 학습 결과 출력
print('베스트 하이퍼 파라미터: {0}'.format(grid_cv.best_params_))
print('베스트 하이퍼 파라미터 일 때 정확도: {0:.4f}'.format(grid_cv.best_score_))

# 최적화 모델 추출
model = grid_cv.best_estimator_

# 테스트세트 정확도 출력
score = model.score(X_test, y_test)
print('테스트세트에서의 정확도: {0:.4f}'.format(score))
```

![image](https://user-images.githubusercontent.com/116700688/233059070-0793e62f-37a0-452f-b683-a80c41fee155.png)
---

### LGBM

```
from sklearn.model_selection import RandomizedSearchCV

lgbm_model3 = LGBMClassifier(random_state=100)

parameters = {'num_leaves':[20,40,60,80,100], 'min_child_samples':[5,10,15], 'max_depth':[-1,5,10,20],
             'learning_rate':[0.05,0.1,0.2], 'reg_alpha':[0,0.01,0.03]}

grid_lgbm = RandomizedSearchCV(lgbm_model3, parameters,scoring='accuracy', n_iter=100)
grid_lgbm.fit(X_train, y_train)
print(grid_lgbm.best_params_)
lgbm_pred3 = grid_lgbm.predict(X_test)
print('Classification of the result is:')
print(accuracy_score(y_test, lgbm_pred3))
```

![image](https://user-images.githubusercontent.com/116700688/233059519-d4f12ee7-8a1d-4d1c-ae16-efa7722fd060.png)

---

### XGBoost
```
import xgboost as xgb

param_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.3], # 경사하강법 : '매개변수' -> 최소오차 -> 보폭 크기
    'max_depth': [3], # 트리의 깊이 (오버피팅 방지)
    'subsample': [0.3, 0.5, 0.7, 1], # 추출할 데이터 비율
    'n_estimators': [300, 600, 1000] # 트리 개수
}
```

![image](https://user-images.githubusercontent.com/116700688/233059937-547233e6-9f65-420c-a4cf-18f6589d2478.png)

---

# Result

```
results = pd.DataFrame({'Random Forest(최초)' : [0.797, 0.88, 0.8474],
                        'LGBM (최초)' : [0.8074, 0.8924, 0.8441],
                        'Random Forest(H.O 조정)' : [0.8039, 0.8854, 0.8198],
                        'LGBM (H.O 조정)' : [0.8056, 0.886, 0.8181],
                        'Random Forest(GS CV 조정)' : [0.8056, 0.8832, 0.82],
                        'LGBM (Rando GS CV 조정)' : [0.8028, 0.8832, 0.82],
                        'XGBoost (GS CV 조정)' : [0.8, 0.8832, 0.82]}, 
                       index = ['Accuracy', 'ROC-AUC Score', '훈련 set Accuracy'])
results
```

![image](https://user-images.githubusercontent.com/116700688/233060144-97640b09-76dd-42bb-ae99-d09efd8fafff.png)
