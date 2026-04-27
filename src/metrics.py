import numpy as np
from sklearn.metrics import f1_score

def calculate_macro_f1(y_true, y_pred):
    # Преобразуем предсказанные и реальные классы в numpy-массивы
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average='macro'
    )