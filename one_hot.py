import numpy as np
import pandas as pd

NOMES_CLASSES = ['otimo', 'bom', 'medio', 'regular', 'insuficiente']

df = pd.read_csv('dataset.csv')
X = df.values

def media_para_classe(media):
    if media >= 0.80:
        return 0  # otimo
    elif media >= 0.65:
        return 1  # bom
    elif media >= 0.45:
        return 2  # medio
    elif media >= 0.30:
        return 3  # regular
    else:
        return 4  # insuficiente

PESOS = np.array([0.05, 0.15, 0.35, 0.25, 0.15, 0.05]) 
# diferentes pesos para cada atributo (diferente da criação
# do dataset que supoe que quem dorme bem tem maior desemmpenho),
# agora dando mais valor para as avaliações e para quem estuda

scores = X.dot(PESOS)
classes_idx = np.array([media_para_classe(s) for s in scores])

Y = np.zeros((len(X), 5))
Y[np.arange(len(X)), classes_idx] = 1

print(f"atributos de entrada - X: {X.shape}   atributos de saída - Y: {Y.shape}")

# salvar
df_out = df.copy()
df_out['classe'] = [NOMES_CLASSES[i] for i in classes_idx]
for i, nome in enumerate(NOMES_CLASSES):
    df_out[f'y_{nome}'] = Y[:, i].astype(int)
df_out.to_csv('dataset_classificado.csv', index=False)

# geração de matrizes para a rede neural
np.save('X.npy', X)
np.save('Y.npy', Y)

print("\narquivos salvos")