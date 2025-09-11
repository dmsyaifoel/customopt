'''
Test functions for optimizations
'''

def beale(x):
  return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def booth(x):
  return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def rosen(x, n=5):
  return sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n-1)])

if __name__ == '__main__':

  print('interval arithmetic to predict optimum value within given bounds (for gradient_descent_root)')

  from interval import *

  x = interlist((-10, 10), 2)
  print('beale min', beale(x)[0])
  print('booth min', booth(x)[0])
  x = interlist((-10, 10), 5)
  print('rosen min', rosen(x)[0])
  print()

  from opt import *

  print('finite differences')

  def fgrad(x):
    return beale(x), gradient(beale, x)

  print(f'{gradient_descent(fgrad, [0, 0], .01) = }')
  print(f'{gradient_descent_root(fgrad, [0, 0], 0) = }')
  print(f'{line_search(fgrad, [0, 0]) = }')
  print()

  def fgrad(x):
    return beale(x), gradient(booth, x)

  print(f'{gradient_descent(fgrad, [0, 0], .01) = }')
  print(f'{gradient_descent_root(fgrad, [0, 0], 0) = }')
  print(f'{line_search(fgrad, [0, 0]) = }')
  print()

  def fgrad(x):
    return rosen(x), gradient(rosen, x)

  print(f'{gradient_descent(fgrad, [0]*5, .001) = }')
  print(f'{gradient_descent_root(fgrad, [0]*5, 0) = }')
  print(f'{line_search(fgrad, [0]*5) = }')
  print()

  print('automatic differentiation')

  from sym import *

  x = symlist('x', 2)
  beale_sym = beale(x)
  print(f'{beale_sym = }')

  def fgrad(x):
    return beale_sym.vg({'x':x})

  print(f'{gradient_descent(fgrad, [0, 0], .01) = }')
  print(f'{gradient_descent_root(fgrad, [0, 0], 0) = }')
  print(f'{line_search(fgrad, [0, 0]) = }')
  print()

  booth_sym = booth(x)
  print(f'{booth_sym = }')

  def fgrad(x):
    return booth_sym.vg({'x': x})

  print(f'{gradient_descent(fgrad, [0, 0], .01) = }')
  print(f'{gradient_descent_root(fgrad, [0, 0], 0) = }')
  print(f'{line_search(fgrad, [0, 0]) = }')
  print()

  x = symlist('x', 5)
  rosen_sym = rosen(x)
  print(f'{rosen_sym = }')

  def fgrad(x):
    return rosen_sym.vg({'x': x})

  print(f'{gradient_descent(fgrad, [0]*5, .001) = }')
  print(f'{gradient_descent_root(fgrad, [0]*5, 0) = }')
  print(f'{line_search(fgrad, [0]*5) = }')