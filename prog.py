import Krichevsky
import time
import predictor


def cross_val_split(data, n_splits):
    """Делает разбиение временного ряда на n_splits датасетов, сохраняя порядок.
    :param data: временной ряд в форме массива
    :param n_splits: сколько разбиений хочется получить
    :return: res data - n_splits массивов, для которых верное следующее значение - соответствующий элемент из answ"""
    res_data = []
    answ = []
    for i in range(n_splits):
        res_data.append(data[:-(i+1)])
        answ.append(data[-(i+1)])
    return res_data, answ


def accuracy_metrics(y_true, y_pred):
    """Измеряет точность предсказания по числу совпадений
    :param y_true: верные значения
    :param y_pred: предсказанные значение
    :return: точность предсказания - число верных ответов, деленное на длину y_true"""
    tr = 0
    for i in range(len(y_true)):
        if y_pred[i]==y_true[i]:
            tr += 1
    return tr/(len(y_true))


def mean_absolute_error(y_true, y_pred):
    """Вычисляет среднюю абсолютную ошибку.
    :param y_true: верные значения
    :param y_pred: предсказанные значение
    :return: точность предсказания - сумму модулей разности верных и предсказанных значений, деленное на длину y_true"""
    tr = 0
    for i in range(len(y_true)):
        tr += abs(float(y_true[i])-float(y_pred[i]))
    return tr



pred = Krichevsky.KrichevskyPredictor()
res, alpha, asize = pred.load_data('data/eur_usd.txt')
X, y = cross_val_split(res, 50)
y_pred = []
stt = time.time()
for i, x in enumerate(X):
    y_pred.append(pred.fit_predict(x, weights_type='r', sort_weights=True))
y = [pred.mapping[char] for char in y]
print('TOTAL CALCULATION TIME: ', time.time()-stt, 'Dataset size: ', len(X))
print("Accuracy on usd: ", accuracy_metrics(y, y_pred))
print("MAE on usd: ", mean_absolute_error(y, y_pred))