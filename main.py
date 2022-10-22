import numpy as np
from sklearn.datasets import load_digits

import dann
import lr
import plotly.express as px

digits = load_digits()
X = digits.data
target = digits.target

print("нормализация или стандартизация?(0 - нормализация, 1 - стандартизация)")
k = int(input())
# k = 0
d = dann.dann()
if k == 0:
    X = d.normaniz(X)
else:
    X = d.standortiz(X)
print(X)

# делаю нужный вид ответов

target = d.one_hot_encoding(target)
print(target)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def preobr(x):
    return np.array([x[k * 8:(k + 1) * 8] for k in range(8)])
# for i in range(10):
#     fig = px.imshow(preobr(X[i]), title=str(np.argmax(target[i])))
#     fig.show()

# перемешиваю данные

#
X_train, X_val, X_test, t_train, t_val, t_test = d.peremeshat_razd(X, target)#d.peremeshat_razd(X, target)
print(X_train)
# for i in range(10):
#     fig = px.imshow(preobr(X_train[i]), title=str(np.argmax(t_train[i])))
#     fig.show()

ll = lr.Lr(10, 64, len(X_train), X_train, X_val, t_train, t_val)

ll.save()
ll.load()

ll.grad_sp(X_train, t_train, 0)

ll.top3_val()
