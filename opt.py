'''
Custom optimization functions for use with custom objects
'''

def root_bisection(f, a, b, atol=1e-6, maxloop=1000):
  assert a < b
  fa = f(a)
  fb = f(b)
  assert fa*fb < 0
  for i in range(maxloop):
    c = (a + b)/2
    fc = f(c)
    if abs(fc) < atol: return c, i
    if fa*fc < 0:
      b = c
      fb = f(b)
    else:
      a = c
      fa = f(a)
  return c, 'maxloop'
  
def deriv(f, x, dx=1e-6):
  # first derivative using central finite difference
  return (f(x + dx) - f(x - dx))/2/dx

def root_newton(f, x=0, atol=1e-6, maxloop=1000, divtol=1e-50):
  # find the root of a scalar function using the newton's algorithm
  fx, d = f(x), deriv(f, x)
  for i in range(maxloop):
    if abs(d) < divtol: return x, 'divtol'
    xn = x - fx/d
    fxn, dn = f(xn), deriv(f, xn)
    if abs(fx - fxn) < atol: return xn, i
    x = xn
    fx, d = fxn, dn
  return xn, 'maxloop'

def deriv2(f, x, dx=1e-6):
  # second derivative using central finite difference
  return (f(x + dx) - 2*f(x) + f(x - dx))/dx**2

def derivs(f, x, dx=1e-6):
  # function evaluation and both derivatives using 3 function evaluations
  fp = f(x + dx)
  fc = f(x)
  fm = f(x - dx)
  return  fc, (fp - fm)/2/dx, (fp - 2*fc + fm)/dx**2

def minimize_newton(f, x=0, atol=1e-6, maxloop=1000, divtol=1e-50):
  # find the minimum of a scalar function using newton's algorithm
  xn = 0
  fx, d1, d2 = derivs(f, x)
  for i in range(maxloop):
    if abs(d2) < divtol: return x, 'divtol'
    xn = x - d1/d2
    fxn, d1n, d2n = derivs(f, xn)
    if abs(fxn - fx) < atol: return xn, i
    x = xn
    fx, d1, d2 = fxn, d1n, d2n
  return xn, 'maxloop'

def minimize_root_adaptive(f, fmin=0, x=0, atol=1e-6, maxloop=100, divtol=1e-50):
  # find the minimum of a scalar function using adaptive rootfinding
  fx = f(x)
  if fx < fmin: fmin -= 2*(fmin - fx)
  for i in range(maxloop):
    d = deriv(f, x)
    if abs(d) < divtol: return x, 'divtol'
    xn = x - (fx - fmin)/d
    fxn = f(xn)
    if abs(fxn - fmin) < atol: return xn, i
    if fxn > fx:
      fmin = (fmin + fx)/2
      continue
    if fxn < fmin: fmin -= 2*(fmin - fx)
    else: fmin -= 2*(fxn - fmin)
    x = xn
    fx = f(x)
  return xn, 'maxloop'


def gradient(f, x):
  # find the gradient of a function using finite difference
  d = len(x)
  g = [0]*d
  for i in range(d):
    def fl(y):
      z = x[:]
      z[i] = y
      return f(z)
    g[i] = deriv(fl, x[i])
  return g

def gradient_descent(fgrad, x, step, atol=1e-6, maxloop=1000):
  # find the minimum of a function using fixed step gradient descent
  x = list(x)
  dim = len(x)
  xn = dim*[0]
  fx, gx = fgrad(x)
  for i in range(maxloop):
    xn = [x[i] - step*gx[i] for i in range(dim)]
    fxn, gxn = fgrad(xn)
    if abs(fxn - fx) < atol: return xn, i
    x = xn
    fx = fxn
    gx = gxn
  return xn, 'maxloop'

def gradient_descent_root(fgrad, x, fmin, atol=1e-6, divtol=1e-50, maxloop=1000):
  # find the minimum of a function using adaptive step gradient descent if you know the value of the minimum
  x = list(x)
  dim = len(x)
  xn = dim*[0]
  fx, gx = fgrad(x)
  for i in range(maxloop):
    div = sum([j**2 for j in gx])
    if div < divtol: return xn, i
    xn = [x[i] - (fx - fmin)/div*gx[i] for i in range(dim)]
    fxn, gxn = fgrad(xn)
    if abs(fxn - fx) < atol: return xn, i
    x = xn
    fx = fxn
    gx = gxn
  return xn, 'maxloop'

def gradient_descent_root_adaptive(fgrad, x, fmin=0, atol=1e-6, divtol=1e-50, maxloop=1000):
  # find the minimum of a function using adaptive step gradient descent
  x = list(x)
  dim = len(x)
  xn = dim*[0]
  fx, gx = fgrad(x)
  if fx < fmin: fmin -= 2*(fmin - fx)
  for i in range(maxloop):
    div = sum([j**2 for j in gx])
    if div < divtol: return xn, i
    xn = [x[i] - (fx - fmin)/div*gx[i] for i in range(dim)]
    fxn, gxn = fgrad(xn)
    if abs(fxn - fx) < atol: return xn, i
    if fxn > fx:
      fmin = (fmin + fx)/2
      continue
    if fxn < fmin: fmin -= 2*(fmin - fx)
    else: fmin -= 2*(fxn - fmin)
    x = xn
    fx = fxn
    gx = gxn
  return xn, 'maxloop'

def line_search(fgrad, x, atol=1e-6, innertol=1e-6, maxloop=1000):
  # line search using finite difference gradients
  x = list(x)
  dim = len(x)
  xn = dim*[0]
  fx, gx = fgrad(x)
  for i in range(maxloop):
    xt = lambda t: [x[j] + gx[j]*t for j in range(dim)]
    func = lambda t: fgrad(xt(t))[0]
    tn, dummy = minimize_newton(func, 0, innertol)
    xn = xt(tn)
    fxn, gxn = fgrad(xn)
    if abs(fxn - fx) < atol: return xn, i
    x = xn
    fx, gx = fxn, gxn
  return xn, 'maxloop'

if __name__ == '__main__':
  def fs(x): return 2*x**2 - 6*x - 9 + 0.1*x**3
  print('Scalar')
  print(f'{root_bisection(fs, -5, 0) = }')
  print(f'{deriv(fs, 3) = }')
  print(f'{root_newton(fs) = }')
  print(f'{deriv2(fs, 3) = }')
  print(f'{derivs(fs, 3) = }')
  print(f'{minimize_newton(fs) = }')
  print(f'{minimize_root_adaptive(fs) = }')

  def fv(x): return (x[0] - 4)**2 + (x[1] - 1)**2 + (x[2] + 3)**2

  print('\nVectorial')
  print(f'{gradient(fv, [1, 3, 4]) = }')

  def fgrad(x): return fv(x), gradient(fv, x)

  print(f'{gradient_descent(fgrad, [1, 3, 4], .1) = }')
  print(f'{line_search(fgrad, [1, 3, 4]) = }')
  print(f'{gradient_descent_root(fgrad, [1, 3, 4], 0) = }')
  print(f'{gradient_descent_root_adaptive(fgrad, [1, 3, 4], 300) = }')
