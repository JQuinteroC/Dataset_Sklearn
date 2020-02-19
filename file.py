import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# import some data to play with
digits = load_digits()
X = digits.data
y = digits.target
class_names = digits.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Modelo vecinos
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Dataset digits')
print('Matriz númerica')
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize = (5,5))
plt.title('Confusión dataset digits grafica')
sn.heatmap(confusion_matrix(y_test, y_pred), annot = True)
plt.show()
print("\n"*3)
#################################
vino = load_wine()
datosX = vino.data
datosY = vino.target

datosX_train, x_test, datosY_train, y_testD=train_test_split(datosX, datosY, random_state = 0)

knnDiabetes = KNeighborsClassifier(n_neighbors = 5)
knnDiabetes.fit(datosX_train, datosY_train)

y_diabetesP = knnDiabetes.predict(x_test)
print('Dataset wine')
print('Matriz númerica')
print(confusion_matrix(y_testD, y_diabetesP, normalize=all))

plt.figure(figsize = (5,5))
plt.title('Confusión dataset digits grafica')
sn.heatmap(confusion_matrix(y_testD, y_diabetesP), annot = True)
