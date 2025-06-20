import math

class Interval:
    '''
    Simple custom interval arithmetic class
    '''
    def __init__(self, min_val, max_val=None):
        self.min_val = float(min_val)
        if max_val is None:
          self.max_val = self.min_val
        else:
          self.max_val = float(max_val)

    def __add__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)

        new_min = self.min_val + other.min_val
        new_max = self.max_val + other.max_val
        return Interval(new_min, new_max)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)

        new_min = self.min_val - other.max_val
        new_max = self.max_val - other.min_val
        return Interval(new_min, new_max)

    def __rsub__(self, other):
        return Interval(other) - self

    def __mul__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)

        products = [
            self.min_val*other.min_val,
            self.min_val*other.max_val,
            self.max_val*other.min_val,
            self.max_val*other.max_val
        ]
        new_min = min(products)
        new_max = max(products)
        return Interval(new_min, new_max)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)

        if other.min_val <= 0 <= other.max_val:
            raise ValueError(f'Cannot divide by an interval which includes zero.')

        reciprocal_min = 1/other.max_val
        reciprocal_max = 1/other.min_val

        products = [
            self.min_val*reciprocal_min,
            self.min_val*reciprocal_max,
            self.max_val*reciprocal_min,
            self.max_val*reciprocal_max
        ]
        new_min = min(products)
        new_max = max(products)
        return Interval(new_min, new_max)

    def __rtruediv__(self, other):
        return Interval(other)/self

    def __pow__(self, power):
        # print(f"Raising interval [{self.min_val}, {self.max_val}] to power {power}")
        if not isinstance(power, (int, float)):
            raise TypeError('Power must be a scalar (int or float).')

        if power == 0:
            return Interval(1, 1)

        # Integer powers
        if isinstance(power, int) or power.is_integer():
            power = int(power)

            if power > 0:
                if self.min_val >= 0:
                    new_min = self.min_val ** power
                    new_max = self.max_val ** power
                elif self.max_val <= 0:
                    new_min = self.max_val ** power
                    new_max = self.min_val ** power
                    if new_min > new_max:
                        new_min, new_max = new_max, new_min
                else:  # spans zero
                    if power % 2 == 0:
                        new_min = 1e-9
                        new_max = max(abs(self.min_val), abs(self.max_val)) ** power
                    else:
                        new_min = self.min_val ** power
                        new_max = self.max_val ** power

            else:  # Negative integer powers
                if self.min_val <= 0 <= self.max_val:
                    raise ValueError("Cannot raise to a negative power if the interval includes zero.")

                # safe reciprocal
                reciprocals = [1 / self.min_val, 1 / self.max_val]
                recip_min, recip_max = min(reciprocals), max(reciprocals)
                return Interval(recip_min, recip_max) ** abs(power)

        else:  # Fractional powers
            if self.min_val < 0:
                raise ValueError("Cannot raise negative numbers to fractional powers for real results.")

            if self.min_val == 0 and power < 0:
                raise ValueError("Cannot raise zero to a negative fractional power.")

            new_min = self.min_val ** power
            new_max = self.max_val ** power

        return Interval(min(new_min, new_max), max(new_min, new_max))

    def __neg__(self):
        new_min = -self.max_val
        new_max = -self.min_val
        return Interval(new_min, new_max)


    def l(self):
        return [self.min_val, self.max_val]

    def __repr__(self):
        return f'{self.l()}'

def sin(x):
    if x.max_val - x.min_val >= 2*math.pi:
        new_min = -1
        new_max = 1
    else:
        val1 = math.sin(x.min_val)
        val2 = math.sin(x.max_val)

        temp_min = min(val1, val2)
        temp_max = max(val1, val2)

        k_start = int(math.floor(x.min_val/(math.pi/2)))
        k_end = int(math.ceil(x.max_val/(math.pi/2)))

        for k in range(k_start, k_end + 1):
            crit_point_val = k*(math.pi/2)
            if x.min_val <= crit_point_val <= x.max_val:
                if math.sin(crit_point_val) == 1:
                    temp_max = 1
                elif math.sin(crit_point_val) == -1:
                    temp_min = -1

        new_min = temp_min
        new_max = temp_max

    return Interval(new_min, new_max)

def cos(x):
    if x.max_val - x.min_val >= 2*math.pi:
        new_min = -1
        new_max = 1
    else:
        val1 = math.cos(x.min_val)
        val2 = math.cos(x.max_val)

        temp_min = min(val1, val2)
        temp_max = max(val1, val2)

        k_start = int(math.floor(x.min_val/(math.pi/2)))
        k_end = int(math.ceil(x.max_val/(math.pi/2)))

        for k in range(k_start, k_end + 1):
            crit_point_val = k*(math.pi/2)
            if x.min_val <= crit_point_val <= x.max_val:
                if math.cos(crit_point_val) == 1:
                    temp_max = 1
                elif math.cos(crit_point_val) == -1:
                    temp_min = -1

        new_min = temp_min
        new_max = temp_max

    return Interval(new_min, new_max)

def acos(x):
    if x.max_val < -1.0 or x.min_val > 1.0:
        raise ValueError(f"Domain error for acos: interval [{x.min_val}, {x.max_val}] is outside [-1, 1].")

    new_min = math.acos(x.max_val)
    new_max = math.acos(x.min_val)

    return Interval(new_min, new_max)

def interlist(bounds, n=None):
  if isinstance(bounds[0], (int, float)):
    assert isinstance(n, int)
    return [Interval(bounds[0], bounds[1]) for i in range(n)]
  n = len(bounds)
  return [Interval(bounds[i][0], bounds[i][1]) for i in range(n)]

if __name__ == '__main__':
  a = Interval(2, 4)
  b = Interval(3, 6)
  c = Interval(-2, 5)
  d = Interval(9, 12)
  print(a**2 + 2*b*c/d - 4*a + b)