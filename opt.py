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

def root_newton(f, x0=0, atol=1e-6, maxloop=1000, divtol=1e-50):
  # find the root of a scalar function using the newton's algorithm
  xn = 0
  fx0, d = f(x0), deriv(f, x0)
  for i in range(maxloop):
    if abs(d) < divtol: return x0, 'divtol'
    xn = x0 - fx0/d
    fxn, dn = f(xn), deriv(f, xn)
    if abs(fx0 - fxn) < atol: return xn, i
    x0 = xn
    fx0, d = fxn, dn
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

def minimize_newton(f, x0=0, atol=1e-6, maxloop=1000, divtol=1e-50):
  # find the minimum of a scalar using newton's algorithm
  xn = 0
  fx0, d1, d2 = derivs(f, x0)
  for i in range(maxloop):
    if abs(d2) < divtol: return x0, 'divtol'
    xn = x0 - d1/d2
    fxn, d1n, d2n = derivs(f, xn)
    if abs(fxn - fx0) < atol: return xn, i
    x0 = xn
    fx0, d1, d2 = fxn, d1n, d2n
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

def gradient_descent(fgrad, x0, step, atol=1e-6, maxloop=1000):
  # find the minimum of a function using fixed step gradient descent
  x0 = list(x0)
  dim = len(x0)
  xn = dim*[0]
  fx0, gx0 = fgrad(x0)
  for i in range(maxloop):
    xn = [x0[i] - step*gx0[i] for i in range(dim)]
    fxn, gxn = fgrad(xn)
    if abs(fxn - fx0) < atol: return xn, i
    x0 = xn
    fx0 = fxn
    gx0 = gxn
  return xn, 'maxloop'

def gradient_descent_root(fgrad, x0, fmin, atol=1e-6, divtol=1e-50, maxloop=1000):
  # find the minimum of a function using adaptive step gradient descent if you know the value of the minimum
  x0 = list(x0)
  dim = len(x0)
  xn = dim*[0]
  fx0, gx0 = fgrad(x0)
  for i in range(maxloop):
    div = sum([j**2 for j in gx0])
    if div < divtol: return xn, i
    xn = [x0[i] - (fx0 - fmin)/div*gx0[i] for i in range(dim)]
    fxn, gxn = fgrad(xn)
    if abs(fxn - fx0) < atol: return xn, i
    x0 = xn
    fx0 = fxn
    gx0 = gxn
  return xn, 'maxloop'

def line_search(fgrad, x0, atol=1e-6, innertol=1e-6, maxloop=1000):
  # line search using finite difference gradients
  x0 = list(x0)
  dim = len(x0)
  xn = dim*[0]
  fx0, gx0 = fgrad(x0)
  for i in range(maxloop):
    xt = lambda t: [x0[j] + gx0[j]*t for j in range(dim)]
    func = lambda t: fgrad(xt(t))[0]
    tn, dummy = minimize_newton(func, 0, innertol)
    xn = xt(tn)
    fxn, gxn = fgrad(xn)
    if abs(fxn - fx0) < atol: return xn, i
    x0 = xn
    fx0, gx0 = fxn, gxn
  return xn, 'maxloop'

if __name__ == '__main__':

  def fs(x):
    return 2*x**2 - 6*x - 9

  print(f'{root_bisection(fs, -5, 0) = }')
  print(f'{deriv(fs, 3) = }')
  print(f'{root_newton(fs) = }')
  print(f'{deriv2(fs, 3) = }')
  print(f'{derivs(fs, 3) = }')
  print(f'{minimize_newton(fs) = }')


  def fv(x):
    return (x[0] - 4)**2 + (x[1] - 1)**2 + (x[2] + 3)**2

  print(f'{gradient(fv, [1, 3, 4]) = }')

  def fgrad(x):
    return fv(x), gradient(fv, x)

  print(f'{gradient_descent(fgrad, [1, 3, 4], .1) = }')
  print(f'{gradient_descent_root(fgrad, [1, 3, 4], 0) = }')
  print(f'{line_search(fgrad, [1, 3, 4]) = }')
