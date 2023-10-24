import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Charger les données à partir du fichier CSV
data = pd.read_csv('./dataset/advertising.csv')
# Afficher les premières lignes du jeu de données pour l'examiner
print(data.head(7))


# Initialiser X (predictors) avec la variable "TV"
X = data['TV']
# Initialiser y (cible) avec la variable "Sales"
y = data['Sales']

plt.scatter(data['TV'], data['Sales'])
plt.title('Relation TV --> SALES')
plt.xlabel('TV')
plt.ylabel('SALES')
plt.show()

# Divisez les données en base d'apprentissage et base de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def mse(X, y,W):
 y_pred = W[0] + W[1] * X
 mse = ((y - y_pred) ** 2).mean()
 return mse

def gradient(X, y, w):
 grad = []
 y_pred = w[0] + w[1] * X
 grad.append((-2 * (y - y_pred).mean()) / len(y))
 grad.append((-2 * (X * (y - y_pred)).mean()) / len(y))
 return grad

def miseJour(grad,w,alfa):
 # Mise à jour des coefficients
 w[0] = w[0] - alfa * grad[0]
 w[1] = w[1] - alfa * grad[1]
 return w

def batchGradDesc(X,y,W,N_max,alfa,eps):
    erreurs = []
    for i in range(N_max):
        mq=mse(X,y,W)
        grad = gradient(X,y,W)
        W = miseJour(grad,W,alfa)
        erreurs.append(mq)
        #print('mse: ',mq)
        if mq < eps:
            break
    return W, mq,erreurs

W = [0,0]
w,mse,errs = batchGradDesc(X_train,y_train,W,100,0.001,10)
print('best errors: ',mse)
print('poids: ',w)

plt.plot(errs)
plt.title("Error evolution")
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.show()