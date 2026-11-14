"""Walu Season: Gaussian Thompson sampling."""

import json
import math
import random
import sys


class Strategy:
    def __init__(self, n_spots, n_casts):
        self.n_spots = n_spots
        self.n = [0] * n_spots      # casts per spot
        self.s = [0.0] * n_spots    # sum of catches
        self.sq = [0.0] * n_spots   # sum of squared catches
        self.rng = random.Random(0)

    def choose(self):
        best, best_val = 0, -1.0
        for i in range(self.n_spots):
            if self.n[i] < 5:        # first 5 casts per spot are a forced look
                return i
            mean = self.s[i] / self.n[i]
            var = max(self.sq[i] / self.n[i] - mean * mean, 4.0)
            draw = self.rng.gauss(mean, math.sqrt(var / self.n[i]))
            if draw > best_val:
                best, best_val = i, draw
        return best

    def update(self, spot, catch):
        self.n[spot] += 1
        self.s[spot] += catch
        self.sq[spot] += catch * catch


def main():
    with open("test-input.json") as f:
        rules = json.load(f)
    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"])
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            catch = float(sys.stdin.readline())
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
