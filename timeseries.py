def load(fileName):
  f = open(fileName, 'r')
  res = []
  for b in f:
    res.extend(b.split())
  f.closed
  return res


def listToInt(lst):
  n = len(lst)
  res = [0] * n
  for i in range(n):
    res[i] = int(lst[i])
  return res


def listToFloat(lst):
  n = len(lst)
  res = [0]*n
  for i in range(n):
    res[i] = float(lst[i])
  return res


def refactor(lst):
  n = len(lst)
  res = [0]*n
  for i in range(n):
    res[i] = lst[i] + 1
  return res

def numerate(lst):
  alphabet = list(set(lst))
  mapping = {i+1: alphabet[i] for i in range(len(alphabet))}
  new_lst = [0]*len(lst)
  for i in range(len(lst)):
    new_lst[i] = alphabet.index(lst[i])+1
  return new_lst, mapping, list(mapping.keys())


def preprocess(res):
  res, mapping, alphabet = numerate(res)
  return res, alphabet, len(alphabet), mapping
