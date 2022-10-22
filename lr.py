import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class Lr(object):

    def softmax(self, z):
        h = np.sum(np.exp(z))
        return np.exp(z) / h

    def y(self, x):
        return self.softmax(self.W.dot(x) + self.b)

    def __init__(self, K, D, N, x_train, x_val, t_train, t_val, sigma=1):
        self.K = K
        self.N = N
        self.D = D
        self.W = np.random.sample((K, D)) * sigma
        self.b = np.random.sample(K) * sigma
        self.x_train = x_train
        self.x_val = x_val
        self.t_train = t_train
        self.t_val = t_val

    def accuracy(self, x, t):
        a = 0
        for i in range(len(x)):
            if np.argmax(self.y(x[i])) == np.argmax(t[i]):
                a += 1
        return a / self.N

    def grad_sp(self, x, T, lambd, gamma=0.0001, n=1000, l=None):  # l-длина батча
        def E():  # ц.ф. наверное надо вытащить от сюда
            h = 0
            for i in range(self.N):
                for k in range(self.K):
                    h += T[i][k] * np.log(self.y(x[i])[k])
            return lambd / 2 * np.sum(self.W ** 2) - h

        acc = [[], [], []]  # 0 - ц.ф., 1 - обуч, 2 - вал

        for i in range(n):
            self.W -= gamma * ((np.array([self.y(x[j]) for j in range(len(x))]) - T).T.dot(x) + lambd * self.W)
            self.b -= gamma * (np.array([self.y(x[j]) for j in range(len(x))]) - T).T.dot(np.array([1 for j in range(self.N)]).T)
            if i % 10 == 0:
                print("осталось "+str(n-i)+"; E = " + str(round(E(), 6)), end='')
                print("; accuracy на обучающей = " + str(round(self.accuracy(self.x_train, self.t_train), 6)), end='')
                print("; accuracy на валидационной = " + str(round(self.accuracy(self.x_val, self.t_val), 6)))
            acc[0].append(E())
            acc[1].append(self.accuracy(self.x_train, self.t_train))
            acc[2].append(self.accuracy(self.x_val, self.t_val))

        for i in range(3):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[j for j in range(self.N)], y=acc[i], mode='markers', name="t"))
            if i == 0:
                fig.update_layout(title="целевая функция")
            elif i == 1:
                fig.update_layout(title="ошибка на обучающей")
            else:
                fig.update_layout(title="ошибка на валидационной")
            fig.show()

        print(self.accuracy(self.x_val, self.t_val))#точность на вал выборке

    def save(self):
        np.savez('param', self.W, self.b)

    def load(self):
        ex = np.load('param.npz')
        self.W = ex['arr_0']
        self.b = ex['arr_1']

    def top3_val(self):
        def preobr(x):
            return np.array([x[k*8:(k+1)*8] for k in range(8)])

        top = []
        ne_top = []
        for i in range(len(self.x_val)):
            z = self.y(self.x_val[i])
            j = np.argmax(z)
            e = z[j]
            top.append([e, i])
            ne_top.append([e, i])
            top.sort()
            ne_top.sort()
            if len(top) == 4:
                top.pop(0)
                ne_top.pop(3)
        for i in range(3):
            fig = px.imshow([preobr(self.x_val[top[i][1]])][0], title="это - " + str(np.argmax(self.t_val[top[i][1]])) +
                                                                     " | думает что это - "+ str(np.argmax(self.y(self.x_val[top[i][1]]))) + " | ошибка - " +
                                                                                                str(top[i][0]))
            fig.show()
        for i in range(3):
            fig = px.imshow([preobr(self.x_val[ne_top[i][1]])][0], title="это - " + str(np.argmax(self.t_val[ne_top[i][1]])) +
                                                                     " | думает что это - "+ str(np.argmax(self.y(self.x_val[ne_top[i][1]]))) + " | ошибка - " +
                                                                                                str(ne_top[i][0]))
            fig.show()
        preobr(self.x_val[top[1][1]])
