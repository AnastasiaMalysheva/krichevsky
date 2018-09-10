import random
import math
from decimal import *


def calcT1(n):
  T1 = [0] * int(n)
  T1[0] = 1
  for i in range(1, n):
    T1[i] = T1[i-1]*(i*2+1)
  return T1


def calcT2(n, Asize):
  T2 = [0] * int(n)
  T2[0] = Asize
  for i in range(1,n):
    T2[i] = T2[i-1]*(Asize+i*2)
  return T2


def getHash(s, P1, P2):
  h = 0
  P1m = 1
  for i in range(int(len(s))):
    h = (h * P1 + int(s[i])) % P2
    if (i > 0):
      P1m = (P1m * P1) % P2
  return (h, P1m)


def moveHashRight(h, s0, sn, P1, P2, P1m):
  #print(s0,' ',sn,' ',h)
  h = (h - P1m * s0) % P2
  #print(h)
  h = (h * P1 + sn) % P2
  #print(h)
  return h


def removeHashRight(h, sn, P1inv, P2):
  h = (h - sn) % P2
  h = h * P1inv
  return h


def expandHash(h, sn, P1, P2):
  h=(h * P1 + sn) % P2
  return h


def calcVxm(t, m, P1, P2):
  if m > len(t):
    return None
  #	P1inv=pow(P1,P2-2,P2)
  h, P1m = getHash(t[:m], P1, P2)
  vx={}
  vxa={}
  for i in range(m, len(t)):
    h2 = expandHash(h, t[i], P1, P2)
    x = vxa.get(h2, 0)
    vxa[h2] = x+1
    x = vx.get(h, 0)
    vx[h] = x + 1
    h = moveHashRight(h, t[i-m], t[i], P1, P2, P1m)
  return vx, vxa


def calcKrichm(t, m, P1, P2, Asize, vx, vxa):
  if m > len(t):
    return None
  #	P1inv=pow(P1,P2-2,P2)
  h, P1m = getHash(t[:m], P1, P2)
  #print(P1,' ',P2,' ',P1m)
  c = getcontext()
  c.prec = 100
  d = Decimal(1)
  for i in range(m, len(t)):
    h2 = expandHash(h, t[i], P1, P2)
    x = vxa.get(h2, 0)
    if h2 not in vxa.keys():
    #print(i,' ',t[i-m:i],' ',h,' ',h2)
      vxa[h2] = x + 1
    d = d * Decimal(x*2 + 1)
    x = vx.get(h, 0)
    if h not in vx.keys():
      vx[h] = x + 1
    if m > 0:
      h = moveHashRight(h, t[i-m], t[i], P1, P2, P1m)
    d = d / Decimal(x*2 + int(Asize))
  for i in range(m):
    d = d / Decimal(Asize)
  return (d, vx, vxa)


def calcPredictors(t, m, Asize, P1, P2):
  ans = [0] * (m + 1)
  for i in range(m+1):
    ans[i]=calcKrichm(t,i,P1,P2,Asize)
    print(i,' - ',ans[i][0])
  return ans


def _try_composite(a, d, n, s):
  if pow(a, d, n) == 1:
    return False
  for i in range(s):
    if pow(a, 2**i * d, n) == n-1:
      return False
  return True # n  is definitely composite


def is_prime(n, _precision_for_huge_n=16):
  if n in _known_primes or n in (0, 1):
    return True
  if any((n % p) == 0 for p in _known_primes):
    return False
  d, s = n - 1, 0
  while not d % 2:
    d, s = d >> 1, s + 1
  # Returns exact according to http://primes.utm.edu/prove/prove2_3.html
  if n < 1373653:
    return not any(_try_composite(a, d, n, s) for a in (2, 3))
  if n < 25326001:
    return not any(_try_composite(a, d, n, s) for a in (2, 3, 5))
  if n < 118670087467:
    if n == 3215031751:
      return False
    return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7))
  if n < 2152302898747:
    return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11))
  if n < 3474749660383:
    return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13))
  if n < 341550071728321:
    return not any(_try_composite(a, d, n, s) for a in (2, 3, 5, 7, 11, 13, 17))
  # otherwise
  return not any(_try_composite(a, d, n, s) for a in _known_primes[:_precision_for_huge_n])


_known_primes = [2, 3]
_known_primes += [x for x in range(5, 1000, 2) if is_prime(x)]


def getRandPrime(minSize):
  while 1>0:
    p=random.randint(int(minSize)+1,int(minSize)+1000000)
    if is_prime(p):
      return p;


def init_and_run(t,maxm,Asize):
  P1=getRandPrime(Asize)
  P2=getRandPrime(P1*1000000000)
  return calcPredictors(t,maxm,Asize,P1,P2)


def calculate_wi(max_n):
    w = []
    for i in range(1, max_n+1):
        w.append(1/math.log(i+1, 2) + 1/math.log(i+2, 2))
    return w


def calculate_linear_knn_weights(max_n):
  w = []
  for i in range(max_n):
    w.append(Decimal((max_n+1-i)/max_n))
  return w


def calculate_exp_knn_weights(max_n):
  w = []
  for i in range(max_n):
    w.append(Decimal((0.5)**i))
  return w


def r_measure(seq, Asize, p1, p2, weights='r', max_step=20):
    if weights=='r':
      w = calculate_wi(len(seq)+1)
    elif weights=='l':
      w = calculate_linear_knn_weights(len(seq)+1)
    elif weights=='e':
      w = calculate_exp_knn_weights(len(seq)+1)
    res = 0
    if len(seq)<max_step:
      max_step = len(seq)
    for i in range(max_step):
        if weights=='r':
          res += Decimal(w[i])*calcKrichm(seq, i, p1, p2, Asize)[0]
        elif weights=='l' or weights=='e':
          res += w[i] * calcKrichm(seq, i, p1, p2, Asize)[0]
    return res