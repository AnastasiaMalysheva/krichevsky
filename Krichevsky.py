import predictor
import timeseries
from decimal import *
import copy
import time


class KrichevskyPredictor(object):

    def __init__(self):
        pass

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
        return res, alphabet, asize

    def fit_predict(self, seq, weights_type='r'):
        """Осуществляет предсказание. Для всех возможных последующих элементов ряда рассчитывает аналог R-меры
        с различными весами - классические веса R-меры, веса, линейно убывающие в зависимости от номера и веса, убывающие
        экспоненциально."""
        st = time.time()
        P1 = predictor.getRandPrime(self.asize)
        P2 = predictor.getRandPrime(P1 * 1000000000)
        res_proba = {}
        max_proba = Decimal(0)
        for ch in self.alplabet:
            temp_seq = copy.deepcopy(seq)
            temp_seq.append(ch)
            res_proba[ch] = predictor.r_measure(temp_seq, self.asize, P1, P2, weights_type)
            if res_proba[ch] >= max_proba:
                max_char = ch
                max_proba = res_proba[ch]
        print('NEXT MOST PROBABLE CHARACTER IS: ', self.mapping[max_char])
        print('CALCULATION TIME: ', time.time()-st)
        #return res_proba
        return self.mapping[max_char]
