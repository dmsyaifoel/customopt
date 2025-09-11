import math

def zeros(n):
  return n*[0]

def isconst(x):
  return isinstance(x, (int, float))

class Sym:
  '''
  Base class for symbolic automatic differentation
  '''
  def __init__(self, name, bounds=None):
    self.name = name
    self.key = id(self)

  def __repr__(self):
    return str(self.name)

  def __add__(self, other):
    if other == 0:
      return self
    if isinstance(self, Add):
      if isinstance(other, Add):
        return Add(self.l + other.l, self.c + other.c)
      if isconst(other):
        return Add(self.l, self.c + other)
      return Add(self.l + [other], self.c)
    if isconst(other):
      return Add([self], other)
    return Add([self, other], 0)

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if other == 1:
      return self
    if other == 0:
      return 0
    if isinstance(self, Mul):
      if isinstance(other, Mul):
        return Mul(self.l + other.l, self.c*other.c)
      if isconst(other):
        return Mul(self.l, self.c * other)
      return Mul(self.l + [other], self.c)
    if isconst(other):
      return Mul([self], other)
    return Mul([self, other], 1)

  def __rmul__(self, other):
    return self*other

  def __neg__(self):
    return self*(-1)

  def __sub__(self, other):
    if other == 0:
      return self
    return self + -other

  def __rsub__(self, other):
    if other == 0:
      return self
    return -self + (other)

  def __pow__(self, other):
    if other == 0:
      return 1
    if other == 1:
      return self
    if isinstance(self, Pow):
      return self.a**(self.b*other)
    return Pow(self, other)

  def __truediv__(self, other):
    if other == 1:
      return self
    if isconst(other):
      return self*(1/other)
    return self*(other**-1)

  def sin(x):
    if isconst(x):
      return math.sin(x)
    return Sin(x)

  def cos(x):
    if isconst(x):
      return math.cos(x)
    return Cos(x)

  def acos(x):
    if isconst(x):
      return math.acos(x)
    return Acos(x)

  def val(self, dic):
    return dic[self.name]

  def valgrad(self, dic, n, cache):
    if self.key in cache:
      return cache[self.key]
    grad = zeros(n)
    if self.name in dic['map']:
      grad[dic['map'][self.name]] = 1
    val = dic[self.name]
    cache[self.key] = (val, grad)
    return val, grad

  def vg(self, dic):
    dic, n = parsedic(dic)
    return self.valgrad(dic, n, {})

class Add(Sym):
  def __init__(self, l, c):
    self.l = l
    self.c = c
    self.key = id(self)

  def __repr__(self):
    s = '('
    for i in self.l:
      s += str(i) + ' + '
    if self.c != 0:
      return s + str(self.c) + ')'
    return s[:-3] + ')'

  def val(self, dic):
    val = self.c
    for i in self.l:
      val += i.val(dic)
    return val

  def valgrad(self, dic, n, cache):
    if self.key in cache:
      return cache[self.key]
    val = self.c
    grad = zeros(n)
    for i in self.l:
      v_, g_ = i.valgrad(dic, n, cache)
      val += v_
      for i in range(n):
        grad[i] += g_[i]
    cache[self.key] = (val, grad)
    return val, grad

class Mul(Sym):
  def __init__(self, l, c):
    self.l = l
    self.c = c
    self.key = id(self)

  def __repr__(self):
    s = '('
    for i in self.l:
      s += str(i) + '*'
    if self.c != 1:
      return s + str(self.c) + ')'
    return s[:-1] + ')'

  def val(self, dic):
    val = self.c
    for i in self.l:
      val *= i.val(dic)
    return val

  def valgrad(self, dic, n, cache):
    if self.key in cache:
      return cache[self.key]
    grad = zeros(n)
    vals = []
    grads = []
    val = self.c
    for i in self.l:
      v_, g_ = i.valgrad(dic, n, cache)
      val *= v_
      vals.append(v_)
      grads.append(g_)
    for k in range(len(self.l)):
      prod = self.c
      for j in range(len(self.l)):
        if j != k:
          prod *= vals[j]
      for i in range(n):
        grad[i] += grads[k][i]*prod
    cache[self.key] = (val, grad)
    return val, grad

class Pow(Sym):
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.key = id(self)

  def __repr__(self):
    return '(' + str(self.a) + '**' + str(self.b) + ')'

  def val(self, dic):
    return self.a.val(dic)**self.b

  def valgrad(self, dic, n, cache):
    if self.key in cache:
      return cache[self.key]
    va, ga = self.a.valgrad(dic, n, cache)
    val = va**self.b
    grad = [self.b*va**(self.b - 1)*ga[i] for i in range(n)]
    cache[self.key] = (val, grad)
    return val, grad

  def interval(self):
    return self.a.interval() ** self.b

class Sin(Sym):
  def __init__(self, a):
    self.a = a
    self.key = id(self)

  def __repr__(self):
    return 'sin(' + str(self.a) + ')'

  def val(self, dic):
    return math.sin(self.a.val(dic))

  def valgrad(self, dic, n, cache):
    if self.key in cache:
      return cache[self.key]
    va, ga = self.a.valgrad(dic, n, cache)
    val = math.sin(va)
    grad = [math.cos(va)*ga[i] for i in range(n)]
    cache[self.key] = (val, grad)
    return val, grad

class Cos(Sym):
  def __init__(self, a):
    self.a = a
    self.key = id(self)

  def __repr__(self):
    return 'cos(' + str(self.a) + ')'

  def val(self, dic):
    return math.cos(self.a.val(dic))

  def valgrad(self, dic, n, cache):
    if self.key in cache:
      return cache[self.key]
    va, ga = self.a.valgrad(dic, n, cache)
    val = math.cos(va)
    grad = [-math.sin(va)*ga[i] for i in range(n)]
    cache[self.key] = (val, grad)
    return val, grad

class Acos(Sym):
  def __init__(self, a):
    self.a = a
    self.key = id(self)

  def __repr__(self):
    return 'acos(' + str(self.a) + ')'

  def val(self, dic):
    v = self.a.val(dic)
    if v >= 1:
      return 0
    elif v <= -1:
      return math.pi
    return math.acos(v)

  def valgrad(self, dic, n, cache):
    key = self.key
    if key in cache:
      return cache[key]
    va, ga = self.a.valgrad(dic, n, cache)
    if va >= 1:
      val, grad = 0, zeros(n)
    elif va <= -1:
      val, grad = math.pi, zeros(n)
    else:
      val = math.acos(va)
      den = max(1e-9, (1 - va**2)**.5)
      grad = [-ga[i]/den for i in range(n)]
    cache[key] = (val, grad)
    return val, grad

def sin(x):
  return Sin(x)

def cos(x):
  return Cos(x)

def acos(x):
  return Acos(x)

def syms(s, bounds=None):
  names = s.replace(' ', '').split(',')
  if bounds is None: return [Sym(name) for name in names]
  assert len(bounds) == len(names)
  return [Sym(name, bounds[i]) for i, name in enumerate(names)]

def symlist(name, n):
  return [Sym(name + str(j)) for j in range(n)]

def parsedic(dic_):
  '''
  Turns a dictionary containing symlists into a dictionary of individual syms
  and adds the 'map' dictionary to speed up lookups
  '''
  dic = dict()
  for key, item in dic_.items():
    dic[str(key)] = item
  dic2 = dict()
  dic3 = dict()
  for key, item in dic.items():
    if not isinstance(item, (list, tuple)):
      dic2[key] = item
      n = len(dic)
    else:
      item = [float(i) for i in item]
      n = len(item)
      for i in range(n):
        dic2[key + str(i)] = item[i]
        dic3[key + str(i)] = i
  if len(dic3) == 0:
    dic3 = {name: idx for idx, name in enumerate(dic.keys())}
  dic2['map'] = dic3
  return dic2, n

if __name__ == '__main__':
  a, b, c, d, e = syms('a, b, c, d, e')
  f = a+b-c*d/e + a**2 + sin(a) + cos(b) - acos(e)
  print(f'{f = }')
  print(f'{f.vg({a:1, b:2, c:-3, d:34, e:25}) = }')

  s = symlist('s', 5)
  print(f'{s = }')
  from matrix import matrix
  sm = matrix(s)

  A = matrix([[1, 2, 3, 4, 5],
              [2, 5, 6, 2, 3],
              [4, 6, 2, 3, 5],
              [7, 2, 5, 2, 3],
              [6, 1, 5, 3, 2]])

  b = A@sm
  t = (b.T()@b)[0]
  print(f'{t = }')
  print(f'{t.vg({'s':(1, 2, 3, 4, 5)}) = }')