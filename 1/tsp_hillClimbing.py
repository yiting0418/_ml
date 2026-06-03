import math
import random


class TspSolution:
    def __init__(self, cities, path=None):
        self.cities = cities
        self.n = len(cities)
        if path is None:
            self.path = list(range(self.n))
        else:
            self.path = path[:]

    def neighbor(self):
        n = self.n
        new_path = self.path[:]
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i > j:
            i, j = j, i
        if j - i < 2:
            return TspSolution(self.cities, new_path)
        new_path[i + 1:j + 1] = reversed(new_path[i + 1:j + 1])
        return TspSolution(self.cities, new_path)

    def height(self):
        return -self.total_distance()

    def total_distance(self):
        dist = 0.0
        n = self.n
        for k in range(n):
            a = self.path[k]
            b = self.path[(k + 1) % n]
            x1, y1 = self.cities[a]
            x2, y2 = self.cities[b]
            dist += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist

    def str(self):
        path_str = "=>".join(str(p + 1) for p in self.path) + "=>1"
        return f"path: {path_str}, distance={self.total_distance():.2f}"


def hillClimbing(s, maxGens=1000, maxFails=100):
    print("start: ", s.str())
    fails = 0
    for gens in range(maxGens):
        snew = s.neighbor()
        sheight = s.height()
        nheight = snew.height()
        if nheight >= sheight:
            print(gens, ':', snew.str())
            s = snew
            fails = 0
        else:
            fails += 1
        if fails >= maxFails:
            break
    print("solution: ", s.str())
    return s


def main():
    random.seed(42)
    n = 20
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]

    s = TspSolution(cities)
    print(f"Initial distance: {s.total_distance():.2f}")

    result = hillClimbing(s, maxGens=10000, maxFails=500)
    print(f"Final distance: {result.total_distance():.2f}")


if __name__ == "__main__":
    main()
