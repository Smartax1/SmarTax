from env.env_core import economic_society
from omegaconf import OmegaConf
from basic_dqn import *

# Load environment parameters
yaml_cfg = OmegaConf.load(f'./cfg/default.yaml')
env = economic_society(yaml_cfg.Environment)

# Get action max
gov_action_max = env.government.action_space.high[0]
house_action_max = env.households.action_space.high[0]

# Get observation and action sizes
global_obs, private_obs = env.reset()
state_size = len(global_obs) + len(private_obs[0])  # assuming all households have the same state size
action_size = env.government.action_space.shape[0] # only generate government actions because households actions are not considered

# Create DQN agent
agent = Agent(state_size, action_size)
batch_size = 32
EPISODES = 500

# Define model path
model_path = "/home/data/michael/TaxAI/data"

# Define lists
gov_rew = []
house_rew = []
epochs = []

# Training loop
for e in range(EPISODES):
    global_obs, private_obs = env.reset()
    state = np.concatenate((global_obs, private_obs[0]))  # Assuming single household for simplicity

    # Define var for culm rew
    culm_govr = 0
    culm_houser = 0

    for time in range(500):
        # Choose action
        action_idx = agent.act(state)
        
        # Map action index to environment actions
        gov_action = np.zeros(env.government.action_space.shape)
        house_action = np.zeros(env.households.action_space.shape)
        if action_idx < gov_action.size:
            gov_action[action_idx] = 1
        else:
            house_action[action_idx - gov_action.size] = 1

        action = {
            env.government.name: gov_action * gov_action_max,
            env.households.name: house_action * house_action_max
        }

        # Take action
        next_global_obs, next_private_obs, gov_reward, house_reward, done = env.step(action)
        
        # Process next state and rewards
        next_state = np.concatenate((next_global_obs, next_private_obs[0]))  # Assuming single household
        reward = gov_reward + np.sum(house_reward)  # Simplified reward

        # Remember experience
        agent.remember(state, action_idx, reward, next_state, done)

        state = next_state

        culm_govr += gov_reward
        culm_houser += np.sum(house_reward)


        if done:
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    print(f"Episode {e}/{EPISODES}, Time: {time}, Gov Reward: {gov_reward}, House Reward: {np.sum(house_reward)}, Total reward: {reward}")

    # save information
    gov_rew.append(culm_govr)
    house_rew.append(culm_houser)
    np.savetxt(model_path + "/gov_reward.txt", gov_rew)
    np.savetxt(model_path + "/house_reward.txt", house_rew)
    epochs.append(e)
    np.savetxt(model_path + "/steps.txt", epochs)


    # save the model
    if e % 50 == 0:
        torch.save(agent.model.state_dict(), model_path+ '/house_net.pt')

env.close()
