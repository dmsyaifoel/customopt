'''
Custom optimization functions for use with custom objects
'''

def deriv(f, x, dx=1e-6):
  # first derivative using central finite difference
  return (f(x + dx) - f(x - dx))/2/dx

def root_scalar(f, x0=0, atol=1e-6, maxloop=1000, divtol=1e-50):
  # find the root of a scalar function using the newton method
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
  # both derivatives using 3 function evaluations
  fp = f(x + dx)
  fc = f(x)
  fm = f(x - dx)
  return  fc, (fp - fm)/2/dx, (fp - 2*fc + fm)/dx**2

def minimize_scalar(f, x0=0, atol=1e-6, maxloop=1000, divtol=1e-50):
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

# def gradient_descent(fgrad, x0, frac, atol=1e-6, maxloop=1000):
#   x0 = list(x0)
#   dim = len(x0)
#   xn = dim*[0]
#   fx0, gx0 = fgrad(x0)
#   for i in range(maxloop):
#     xn = [x0[i] - frac*gx0[i] for i in range(dim)]
#     fxn, gxn = fgrad(xn)
#     if abs(fxn - fx0) < atol: return xn, i
#     x0 = xn
#     fx0 = fxn
#     gx0 = gxn
#   return xn, 'maxloop'

def line_search(f, fgrad, x0, atol=1e-6, innertol=1e-3, maxloop=1000):
  x0 = list(x0)
  dim = len(x0)
  xn = dim*[0]
  fx0, gx0 = fgrad(x0)
  for i in range(maxloop):
    xt = lambda t: [x0[j] + gx0[j]*t for j in range(dim)]
    func = lambda t: f(xt(t))
    tn, dummy = minimize_scalar(func, 0, innertol)
    xn = xt(tn)
    fxn, gxn = fgrad(xn)
    print(i, fxn)
    if abs(fxn - fx0) < atol: return xn, i
    x0 = xn
    fx0, gx0 = fxn, gxn
  return xn, 'maxloop'

if __name__ == '__main__':
  def f(x):
    return (x[0] - 7)**2 + (x[1] - 3)**2

  def fgrad(x):
    return f(x), 2*x[0] + 2*x[1]

  print(line_search(f, fgrad, (0, 0)))