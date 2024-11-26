import numpy as np
import matplotlib.pyplot as plt

# very simple model of a hopfield net with hebbian learning
# number of neurons
N = 25

# 3 patterns to be recognized
smiley = np.array([-1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1])
house = np.array([-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,-1])
cross = np.array([1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1])

# init weightmatrix J (we don't have an external field)
J = np.zeros((25,25))
# use hebbian learning rule to compute weights
J = 1/N * (np.outer(smiley, smiley) + np.outer(house, house) + np.outer(cross, cross))
np.fill_diagonal(J,0)

# neural dynamics
# sign function (ambiguous case is set to 1, cannot happen in this case tho)
def sgn(x):
    return np.where(x>=0,1,-1)

# evolve network
random_start = np.random.choice([-1,1], size=25)
def run_network(init_pattern=random_start, max_iter=20):
    states, iter = np.array([np.zeros(25), init_pattern]), 0
    while iter<max_iter and not(np.array_equal(states[-1],states[-2])):
        iter += 1
        new_state = sgn(J @ states[-1])
        states = np.vstack([states, new_state])
    # print('Number of Iterations: ', iter)
    return states[-1]

# small test function to test the ability to recognize the patterns (or their inverses)
def test(num=100, max_iterations=20):
    counter_h, counter_s, counter_c, counter_miss, results = 0, 0, 0, 0, []
    for _ in range(num):
        random_init = np.random.choice([-1,1], size=25)
        result = run_network(init_pattern=random_init, max_iter=max_iterations)
        results.append(result)
        if np.array_equal(result, house) or np.array_equal(result, house*-1):
            counter_h += 1
        elif np.array_equal(result, smiley) or np.array_equal(result, smiley*-1):
            counter_s += 1
        elif np.array_equal(result, cross) or np.array_equal(result, cross*-1):
            counter_c += 1
        else :
            counter_miss += 1
    print('House: ', counter_h, 'Smiley: ', counter_s, 'Cross: ', counter_c, 'Missed: ', counter_miss)
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
            ax.imshow(results[i].reshape(5, 5))
            ax.axis('off')
    plt.show()
    return counter_h, counter_s, counter_c, counter_miss
