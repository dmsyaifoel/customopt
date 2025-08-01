'''
Test functions for optimizations
'''

def rosen(x, n=2):
  return sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n-1)])

def beale(x):
  return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 _ x[0] + x[0]*x[1]**3)**2

def booth(x):
  return (x[0] + 2*x[1] - 7)**2 +(2*x[0] + x[1] -5)**2
