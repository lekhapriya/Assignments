# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:06:00 2021

@author: lekha.dk
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy import stats


def maximize(g, a, b, args):
    """
    Maximize the function g over the interval [a, b].

    We use the fact that the maximizer of g on any interval is
    also the minimizer of -g.  The tuple args collects any extra
    arguments to g.

    Returns the maximal value and the maximizer.
    """

    objective = lambda x: -g(x, *args)
    result = minimize_scalar(objective, bounds=(a, b), method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum

class OptimalGrowthModel:

    def __init__(self,
                 r,            # reward function
                 f,            # renewal function
                 β=0.909,      # discount factor
                 μ=0.05,       # income location parameter
                 s=0.25,       # income scale parameter
                 grid_max=4,
                 grid_size=120,
                 shock_size=250,
                 seed=1234):

        self.r, self.f, self.β, self.μ, self.s = r, f, β, μ, s
       

        # Set up grid
        self.grid = np.linspace(1e-5, grid_max, grid_size)

        # Store shock (with a seed, so results are reproducible)
        np.random.seed(seed)
        self.shocks = np.exp(μ + s * np.random.randn(shock_size))

    def objective(self, a, s, v_array,y):
        """
        Right hand side of the Bellman equation.
        """

        r, f, β, shocks = self.r, self.f, self.β, self.shocks
        
        v = interp1d(self.grid, v_array,fill_value="extrapolate")

        return r(a) + β * np.mean(v(f(s)+(y)-a))
    
def T(og, v, y):
    """
    The Bellman operator.  Updates the guess of the value function
    and also computes a v-greedy policy.

      * og is an instance of OptimalGrowthModel
      * v is an array representing a guess of the value function

    """
    v_new = np.empty_like(v)
    v_greedy = np.empty_like(v)

    for i in range(len(grid)):
        s = grid[i]

        # Maximize RHS of Bellman equation at state s
        a_star, v_max = maximize(og.objective, 1e-10, (s), (s, v, y))
        v_new[i] = v_max
        v_greedy[i] = a_star


    return v_greedy, v_new


def fcd(k):
    return k*1.10

#reward

g = 0.5
def rwd(m):
    return (m**(1-g)/(1-g))



og = OptimalGrowthModel(r=rwd, f=fcd)
grid = og.grid

y =[]
y.append(np.ones(250))
for i in range(10):
    y.append(og.shocks*y[i])
    
y.reverse()

v = rwd(grid) # An initial condition
n = 10


fig, ax = plt.subplots()

ax.plot(grid, v, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='Initial condition')

for i in range(n):
    v_greedy, v = T(og, v,y)  # Apply the Bellman operator
    ax.plot(grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)



ax.legend()
ax.set(ylim=(0, 50), xlim=(0, 4.5))
plt.show()

opt_pi = interp1d(grid, v_greedy,fill_value="extrapolate")

T=np.array([10,15,20])
for k in range(len(T)):
    s = []
    y = []
    policy=[]
    np.random.seed(1234)
    noise = np.exp(0.05 + 0.25 * np.random.randn(T[k]))
    s.append(1)
    y.append(1)
    policy.append(0)
    for i in range(T[k]):
        y.append(y[i]*noise[i])
        policy.append(opt_pi(s[i]))
        res = fcd(s[i])+(y[i])-opt_pi(s[i])
        s.append(res)
        k = np.array(s)
    std,mean = np.std(k),np.mean(k)
    print("std=",std,"mean=",mean)
    plt.plot(s)
    plt.xlabel("time")
    plt.ylabel("state")
    plt.title("Optimal path")
    plt.show()
        



