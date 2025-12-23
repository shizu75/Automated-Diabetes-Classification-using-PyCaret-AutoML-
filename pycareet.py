from pycaret.classification import *
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.datasets import get_data

data = get_data('diabetes')
train_data, test_data = train_test_split(data, test_size = 0.3, random_state = 42)
new = setup(data = train_data, target = 'Class variable')

compare_models()
best_model = automl(optimize = 'Accuracy')
print(best_model)
tuned = tune_model(best_model)
print(tuned)

predict = predict_model(tuned, data = test_data)
print(predict)

from sklearn.metrics import accuracy_score
accuracy_score(test_data['Class variable'], predict['Class variable'])
