import gym
import numpy as np


def value_iteration(env, gamma=1.0):  # Remember gamma is the discount factor e[0:1]
    # Initialisation "State - value" table : on a un vecteur de taille observation.space.n
    value_table = np.zeros(env.observation_space.n)
    # Iterations
    no_of_iterations = 100000
    # Seuil a partir duquel on n'update plus la valeur
    threshold = 1e-20
    # Pour chaque Itération
    for i in range(no_of_iterations):
        print('value_table :\n', value_table)
        # On récupère les valeurs actuelles de la Q table
        updated_value_table = np.copy(value_table)
        # Pour chaque Etat S
        for state in range(env.observation_space.n):
            # On initialise une liste de Q value
            Q_value = []
            # Pour chaque Action A
            for action in range(env.action_space.n):
                # On initialise une liste de récompenses
                next_states_rewards = []
                # On récupère  trans_prob, next_state et reward_prob pour chaque couple [state][action]
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    # On ajoute a la liste de récompense le calcul suivant :
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))
                # On ajoute a la liste de Q value la somme des valeurs de la liste de récompense
                Q_value.append(np.sum(next_states_rewards))
            # On rentre dans la value_table a l'indice [state] l'espérance maximum
            value_table[state] = max(Q_value)
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return value_table


def extract_policy(value_table, gamma=1.0):
    # initialize the policy with zeros
    policy = np.zeros(my_env.observation_space.n)
    # Pour chaque état
    for state in range(my_env.observation_space.n):
        # on initialise la Q_table de dimensions 4 ( 4 actions, on boucle 16 fois, 16 états )
        Q_table = np.zeros(my_env.action_space.n)
        # pour chacune des 4 actions on calcul la Q_value
        for action in range(my_env.action_space.n):
            # pour chaque couple état/action on récupere la probabilité de transition, l'état suivant et la reward prob
            for next_sr in my_env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        # select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)
    return policy


my_env = gym.make('FrozenLake-v1')
final_value_table = value_iteration(env=my_env, gamma=1.0)
final_policy = extract_policy(final_value_table, gamma=1.0)
print("Final policy :\n", final_policy)


def simulation(env, policy, epochs):
    state = env.reset()
    for i in range(epochs):
        env.render()
        state, reward, done, _ = env.step(int(policy[state]))
        if done:
            print(f"Simulation ended within {i} epochs.")
            if reward == 1:
                print('The agent won !')
            else:
                print('The agent lost !')
            break
        elif i == epochs-1 and not done:
            print('Last epoch reached...')


simulation(my_env, final_policy, 30)
my_env.close()














