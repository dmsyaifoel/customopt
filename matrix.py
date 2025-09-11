class Matrix:
  '''
  A simple matrix implementation that can deal with custom datatypes (intended for the Sym class).
  '''
  def __init__(self, l):
    self.l = l
    self.dim = (len(l), len(l[0]))

  def __repr__(self):
    return str(self.l)

  def __getitem__(self, i):
    if self.dim[0] == 1:
      return self.l[0][i]
    if self.dim[1] == 1:
      return self.l[i][0]
    return self.l[i]

  def __add__(self, other):
    return Matrix([[self.l[i][j] + other.l[i][j] for j in range(self.dim[1])] for i in range(self.dim[0])])

  def __mul__(self, other):
    return Matrix([[other*self.l[i][j]for j in range(self.dim[1])] for i in range(self.dim[0])])

  def __rmul__(self, other):
    return self*other

  def __neg__(self):
    return self*-1

  def __sub__(self, other):
    return self + (other*-1)

  def __truediv__(self, other):
    return self*(other**(-1))

  def __len__(self):
    if self.dim[0] == 1:
      return self.dim[1]
    if self.dim[1] == 1:
      return self.dim[0]

  def T(self):
    return Matrix([[self.l[i][j] for i in range(self.dim[0])] for j in range(self.dim[1])])

  def __matmul__(self, other):
    return Matrix([[sum([self.l[i][k]*other.l[k][j] for k in range(self.dim[1])]) for j in range(other.dim[1])]for i in range(self.dim[0])])

  def norm(self):
      return sum([self[i]**2 for i in range(len(self))])**.5

  def angle(self, other, tol=1e-6):
      d = (self.T()@other)[0]
      if isinstance(d, (int, float)) and d < tol:
          return 0
      return d/self.norm()/other.norm()

def matrix(l):
  if isinstance(l, Matrix):
    return l
  if not isinstance(l[0], (list, tuple)):
    l = [[i] for i in l]
  return Matrix(l)

if __name__ == '__main__':


  A = matrix([[1, 2],
              [3, 4]])
  b = matrix([[5],
              [6]])

  print(A@b)

  c = matrix([7,
              8])

  print((A.T() @ (b-c)).norm())