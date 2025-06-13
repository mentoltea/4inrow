from __future__ import annotations
import conv_nn
import game
import numpy as np
import tensorflow as tf
import keras
import Q
import random

def reward_func(o1: float, o2: float) -> float:
    return 1 + 2.5*(o2 - o1)

class Enviroment:
    def __init__(self, rows, columns, agent_move, opponent_agent):
        self.rows = rows
        self.columns = columns
        
        self.agent_move = agent_move
        self.opponent_agent = opponent_agent
    
    def reset(self):
        self.game = game.Game(self.rows, self.columns)
        
        if self.agent_move != self.game.turn:
            m = self.opponent_agent.get_move(self.game)
            self.game.move(m)
        
        return self.game.gamemap * self.agent_move
        
    def get_actions(self) -> list[int]:
        return list(range(self.columns))
    
    def step(self, action: int):
        o1 = Q.Q(self.game, self.agent_move, 7)
        if self.game.move(action):
            
            m = self.opponent_agent.get_move(self.game)
            self.game.move(m)
            
            o2 = Q.Q(self.game, self.agent_move, 7)
            reward = reward_func(o1, o2)
        else:
            reward = 0
        done = self.game.ended
        
        
        return self.game.gamemap * self.agent_move, reward, done

    
def q_learn(
    agent,      # model
    env: Enviroment,        # game envirement,
    max_steps_per_episode: int,
    epsilon_random_frames: int, # Number of frames to take random action and observe output
    epsilon_greedy_frames: float, # Number of frames for exploration
    epsilon_min: float, # Minimum epsilon greedy parameter
    epsilon_max: float,  # Maximum epsilon greedy parameter
    update_after_actions: int, # Train the model after n actions,
    batch_size: int,
    gamma: float,  # Discount factor for past rewards
    epsilon: float,  # Epsilon greedy parameter
    loss_function,
    optimizer,
    update_target_network: int, # How often to update the target network
    max_memory_length: int,
    max_episodes: int  # Limit training episodes, will run until solved if smaller than 1
) -> tuple[keras.Model, keras.Model, list]:
    model = keras.models.clone_model(agent)
    model.set_weights(agent.get_weights())
    
    model_target = keras.models.clone_model(agent)
    model_target.set_weights(agent.get_weights())
    
    running_reward = 0
    episode_count = 0
    frame_count = 0
    actions = env.get_actions()
    num_actions = len(actions)
    
    epsilon_interval = (
        epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    
    while True:
        observation = env.reset()
        state = np.array(observation)
        episode_reward = 0
        print(f"Episode: {episode_count}")
        for timestep in range(1, max_steps_per_episode):
            print(f"\tTimestep: {timestep}")
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = keras.ops.convert_to_tensor(state)
                state_tensor = keras.ops.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor)
                # Take best action
                action = keras.ops.argmax(action_probs[0]).numpy() # type: ignore

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = keras.ops.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * keras.ops.amax(
                    future_rewards, axis=1
                ) # type: ignore

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample # type: ignore

                # Create a mask so we only calculate loss on the updated Q-values
                masks = keras.ops.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables)) # type: ignore

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1

        if running_reward > 40:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break

        if (
            max_episodes > 0 and episode_count >= max_episodes
        ):  # Maximum number of episodes reached
            print("Stopped at episode {}!".format(episode_count))
            break
    return (model, model_target, [rewards_history, state_history, state_next_history, action_history, done_history])


if __name__=="__main__":
    NN_MSE = conv_nn.game_conv_NN.load_from("NN4/Conv2D_Regression_MSE.keras")
    NN_MAE = conv_nn.game_conv_NN.load_from("NN4/Conv2D_Regression_MAE.keras")
    
    mse = NN_MSE.model
    mae = NN_MAE.model
    
    opponent_depth = 5
    
    gamma = 0.99  # Discount factor for past rewards,
    epsilon= 1.0
    epsilon_max = 1  # Maximum epsilon greedy parameter
    epsilon_min = 0.1 # Minimum epsilon greedy parameter
    
    max_episodes = 10  # Limit training episodes, will run until solved if smaller than 1
    max_steps_per_episode = 25
    epsilon_random_frames = 10 # Number of frames to take random action and observe output
    epsilon_greedy_frames = 20 # Number of frames for exploration
    update_after_actions = 4 # Train the model after n actions,
    update_target_network = 5 # How often to update the target network
    
    batch_size = 16
    max_memory_length = 50
    
    while True:
        try:
            (mse, _, _) = q_learn(
                agent = mse,
                env = Enviroment(6, 7, random.choice([-1,1]), Q.Q_based_opponent(opponent_depth)),
                loss_function = keras.losses.MeanSquaredError(),
                optimizer = keras.optimizers.Adam(),
                
                gamma = gamma,  # Discount factor for past rewards,
                epsilon = epsilon,
                epsilon_max = epsilon_max,  # Maximum epsilon greedy parameter
                epsilon_min = epsilon_min, # Minimum epsilon greedy parameter
                
                max_episodes = max_episodes,  # Limit training episodes, will run until solved if smaller than 1
                max_steps_per_episode = max_steps_per_episode,
                epsilon_random_frames = epsilon_random_frames, # Number of frames to take random action and observe output
                epsilon_greedy_frames = epsilon_greedy_frames, # Number of frames for exploration
                update_after_actions = update_after_actions, # Train the model after n actions,
                update_target_network = update_target_network, # How often to update the target network
                
                batch_size = batch_size,
                max_memory_length = max_memory_length,
            )
            
            (mae, _, _) = q_learn(
                agent = mae,
                env = Enviroment(6, 7, random.choice([-1,1]), Q.Q_based_opponent(opponent_depth)),
                loss_function = keras.losses.MeanAbsoluteError(),
                optimizer = keras.optimizers.Adam(),
                
                gamma = gamma,  # Discount factor for past rewards,
                epsilon = epsilon,
                epsilon_max = epsilon_max,  # Maximum epsilon greedy parameter
                epsilon_min = epsilon_min, # Minimum epsilon greedy parameter
                
                max_episodes = max_episodes,  # Limit training episodes, will run until solved if smaller than 1
                max_steps_per_episode = max_steps_per_episode,
                epsilon_random_frames = epsilon_random_frames, # Number of frames to take random action and observe output
                epsilon_greedy_frames = epsilon_greedy_frames, # Number of frames for exploration
                update_after_actions = update_after_actions, # Train the model after n actions,
                update_target_network = update_target_network, # How often to update the target network
                
                batch_size = batch_size,
                max_memory_length = max_memory_length,
            )
        except:
            print("Interrupred")
            break
    
    print("Saving...")
    
    NN_MSE.model = mse
    NN_MSE.save("NN4/Conv2D_Q_MSE.keras")
    
    NN_MAE.model = mae
    NN_MAE.save("NN4/Conv2D_Q_MAE.keras")