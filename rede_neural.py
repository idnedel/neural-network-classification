import numpy as np

np.random.seed(1)

X = np.load('X.npy') # (100, 6)
Y = np.load('Y.npy') # (100, 5) - classes em one-hot

NOMES_CLASSES = ['otimo', 'bom', 'medio', 'regular', 'insuficiente']

# 6 entradas, vai para 10 neuronios na camada oculta e retorna 5 saidas
N_ENTRADAS = 6
N_OCULTA = 10
N_SAIDAS = 5

w0 = np.random.random((N_ENTRADAS, N_OCULTA))
w1 = np.random.random((N_OCULTA, N_SAIDAS))

# biases
b0 = np.zeros((1, N_OCULTA))
b1 = np.zeros((1, N_SAIDAS))

# taxa de aprendizado e ciclos
LEARNING_RATE = 1
CICLOS = 500

# ativações
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# loss
def cross_entropy(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# treinamento
N = X.shape[0]  # numero de alunos (100)

for i in range(CICLOS):
    # forward
    l1 = sigmoid(X.dot(w0) + b0)
    l2 = softmax(l1.dot(w1) + b1)

    # backprop
    d_l2 = l2 - Y
    d_l1 = d_l2.dot(w1.T) * sigmoid_deriv(l1)

    # gradients
    dw1 = l1.T.dot(d_l2) / N
    dw0 = X.T.dot(d_l1) / N
    db1 = np.sum(d_l2, axis=0, keepdims=True) / N
    db0 = np.sum(d_l1, axis=0, keepdims=True) / N

    # update com learning rate
    w1 -= LEARNING_RATE * dw1
    w0 -= LEARNING_RATE * dw0
    b1 -= LEARNING_RATE * db1
    b0 -= LEARNING_RATE * db0

    # loss
    if i % 10 == 0 or i == CICLOS - 1:
        loss = cross_entropy(Y, l2)
        print(f"ciclo {i:>5}: loss = {loss:.4f}")

# acurácia
l1 = sigmoid(X.dot(w0) + b0)
l2 = softmax(l1.dot(w1) + b1)
previsoes = l2.argmax(axis=1)
reais = Y.argmax(axis=1)
acuracia = np.mean(previsoes == reais)
print(f"\nacuracia no conjunto de treinamento: {acuracia*100:.1f}%")


# teste com adição de 1 aluno
print("\nteste com um aluno novo:")
x = np.array([[0.9, 0.9, 0.9, 0.8, 0.9, 0.8]])  # entrada do aluno
l1 = sigmoid(x.dot(w0) + b0)
l2 = softmax(l1.dot(w1) + b1)

print("entrada:", x[0])
print("\nprobabilidades:")
for nome, prob in zip(NOMES_CLASSES, l2[0]):
    print(f"  {nome:<13} {prob*100:5.1f}%")
print("\nclasse prevista:", NOMES_CLASSES[l2.argmax()])