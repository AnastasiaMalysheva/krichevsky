import predictor
import timeseries
from decimal import *
import copy
import time


class KrichevskyPredictor(object):

    def __init__(self):
        self.P1 = 0
        self.P2 = 0
        self.vx = {}
        self.vxa = {}

    def load_data(self, file_name):
        """Загружает данные из файла. Данные в файле - последовательность символов из некоторого конечного алфавита с
        единым произвольным разделителем. Последовательность символов преобразуется к последовательности подряд идущих
        целых чисел кроме 0. Преобразование сохраняется в self.mapping.
        : file_name - имя файла для загрузки временного ряда
        : return: res - временной ряд с целочисленными элементами в виде списка, alphabet - алфавит временного ряда,
        asize - размер алфавита"""
        res = timeseries.load(file_name)
        res, alphabet, asize, mapping = timeseries.preprocess(res)
        self.mapping = mapping
        print('SEQUENCE LOADED: ', [self.mapping[i] for i in res], '\nALPHABET: ', [self.mapping[i] for i in alphabet], '\nALPHABET SIZE: ', asize)
        self.alplabet = alphabet
        self.asize = asize
        self.P1 = predictor.getRandPrime(self.asize)
        self.P2 = predictor.getRandPrime(self.P1 * 1000000000)
        return res, alphabet, asize

    def r_measure(self, seq, weights='r', max_step=20):
        if weights == 'r':
            w = predictor.calculate_wi(len(seq) + 1)
        elif weights == 'l':
            w = predictor.calculate_linear_knn_weights(len(seq) + 1)
        elif weights == 'e':
            w = predictor.calculate_exp_knn_weights(len(seq) + 1)
        res = Decimal(0)
        if len(seq) < max_step:
            max_step = len(seq)
        for i in range(1, max_step):
            if i not in self.vx.keys():
                self.vx[i] = {}
            if i not in self.vxa.keys():
                self.vxa[i] = {}
            curr_res, curr_vx, curr_vxa = predictor.calcKrichm(seq, i, self.P1, self.P2, self.asize, self.vx[i], self.vxa[i])
            self.vx[i] = {**self.vx[i], **curr_vx}
            self.vxa[i] = {**self.vxa[i], **curr_vxa}
            if weights == 'r':
                res += Decimal(w[i]) * curr_res
            elif weights == 'l' or weights == 'e':
                res += Decimal(w[i]) * curr_res
        return res

    def fit_predict(self, seq, weights_type='r'):
        """Осуществляет предсказание. Для всех возможных последующих элементов ряда рассчитывает аналог R-меры
        с различными весами - классические веса R-меры, веса, линейно убывающие в зависимости от номера и веса, убывающие
        экспоненциально."""
        st = time.time()
        res_proba = {}
        max_proba = Decimal(0)
        for ch in self.alplabet:
            temp_seq = copy.deepcopy(seq)
            temp_seq.append(ch)
            res_proba[ch] = self.r_measure(temp_seq, weights_type)
            if res_proba[ch] >= max_proba:
                max_char = ch
                max_proba = res_proba[ch]
        print('NEXT MOST PROBABLE CHARACTER IS: ', self.mapping[max_char])
        print('CALCULATION TIME: ', time.time()-st)
        #return res_proba
        return self.mapping[max_char]
