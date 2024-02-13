import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import flappy_bird_gymnasium
import gymnasium

from flappybird import GameState


rgb_shape = (84, 84, 3)
image_shape = (84, 84)
fully_connected_layer_size = 3136
device = torch.device("cpu")


def make_env(type_env):
    width = 288
    height = 512

    if type_env == 0:
        ENVIRONMENT = {
            "render_mode": "rgb_array",
            "screen_size": (width, height),
            "background": "night",
        }

        env = gymnasium.make("FlappyBird-v0", **ENVIRONMENT)
        env.reset()
        return env

    return GameState()


class NeuralNetwork(nn.Module):
    """"
    References:
        https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # hyperparameters
        self.number_of_actions = 2
        self.minibatch_size = 32
        self.gamma = 0.98
        self.epsilon = 0.1
        self.final_epsilon = 0.0001
        self.replay_memory_size = 12000
        self.epochs = 2_000_000

        # architecture
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(fully_connected_layer_size, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


class FlappyBirdTrainer:
    def __init__(self, environment, path, agent=None):
        if agent is not None:
            self.model = agent
        else:
            self.model = NeuralNetwork()
            self.model.apply(self.init_weights)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
            self.loss_function = nn.MSELoss()
            self.replay = []
            self.epsilon_annealing = np.linspace(
                self.model.epsilon,
                self.model.final_epsilon,
                self.model.epochs,
            )
            self.best_score = 0
            self.minibatch_size = self.model.minibatch_size

        self.env = make_env(environment)
        self.type_env = environment
        self.path = path

    def update_optimizer(self, q_value, output_y_batch):
        self.optimizer.zero_grad()

        output_y_batch = output_y_batch.detach()

        # calculate loss
        loss = self.loss_function(q_value, output_y_batch)

        # do backward pass
        loss.backward()
        self.optimizer.step()

    def batchify(self):
        # get a batch from memory
        minibatch = random.sample(
            self.replay, min(len(self.replay), self.minibatch_size)
        )

        state_batch_list = []
        action_batch_list = []
        reward_batch_list = []
        state_1_batch_list = []
        done_batch_list = []

        # extracting the batches from replay members
        for data_tuple in minibatch:
            state, action, reward, state_1, done = data_tuple

            state_batch_list.append(state)
            action_batch_list.append(action)
            reward_batch_list.append(reward)
            state_1_batch_list.append(state_1)
            done_batch_list.append(done)

        # concatenate into tensors
        state_batch = torch.cat(state_batch_list).to(device)
        action_batch = torch.cat(action_batch_list).to(device)
        reward_batch = torch.cat(reward_batch_list).to(device)
        state_1_batch = torch.cat(state_1_batch_list).to(device)
        return state_batch, action_batch, reward_batch, state_1_batch, done_batch_list

    def compute_predicted_y_batches(
        self, done_batch, reward_batch, output_state_1_batch
    ):
        """
        Compute the Y batches in the following way:
            - if state is final, give the reward of the state
            - otherwise apply discount formula based on prediction made by the model
        """
        output_y_batch = []

        size = (
            self.minibatch_size
            if len(done_batch) >= self.minibatch_size
            else len(done_batch)
        )

        for index in range(size):
            done = done_batch[index]
            reward = reward_batch[index]
            if done:
                output_y_batch.append(reward)
            else:
                output_y_batch.append(
                    reward + self.model.gamma * torch.max(output_state_1_batch[index])
                )
        return torch.cat(output_y_batch)

    def reset_env(self, done):
        if done and self.type_env == 0:
            self.env.reset()

    def act(self, output, skip_frame, initial_action=False):
        """
        Act based on epsilon greedy exploration
        """
        action = torch.zeros([self.model.number_of_actions], dtype=torch.float32).to(
            device
        )

        if initial_action:
            action[0] = 1
            skip_frame = 0
            return 0, action, skip_frame

        random_action = random.random() <= self.model.epsilon

        if skip_frame == 1:
            action_index = 0
            skip_frame = 0
        else:
            skip_frame = 1
            action_index = [
                torch.randint(
                    self.model.number_of_actions, torch.Size([]), dtype=torch.int
                )
                if random_action
                else torch.argmax(output)
            ][0].to(device)

        action[action_index] = 1

        return action_index, action, skip_frame

    def process_image(self, image):
        """
        Helper method for manipulation of image input.

        The goal is to transform the input image in a BLACK AND WHITE image, where whie is the
        bird and pipes; black anything else.

        Parameters used:
            - cropping the initial 512x288 image to 400x200. By this, I shrunk the width a bit to concentrate a lot
            of passing one pipe; by shrinking the height, I remove necessary features, such as the floor. It turned out
             that this process of image input increases the accuracy of training over same epochs.
        """
        if self.type_env == 0:
            image = image[0:400, 0:200]
        else:
            image = image[0:288, 25:325]
        image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
        image_data = np.reshape(image_data, (84, 84, 1))
        image_data[image_data > 0] = 255  # anything not black, white. AKA pipes + bird
        image_tensor = image_data.transpose(2, 0, 1)
        image_tensor = image_tensor.astype(np.float32)
        image_tensor = torch.from_numpy(image_tensor)
        return image_tensor.to(device)

    @staticmethod
    def init_weights(sub_module):
        if isinstance(sub_module, nn.Conv2d) or isinstance(sub_module, nn.Linear):
            print("DEVICE HERE: ", device)
            nn.init.uniform_(sub_module.weight, -0.01, 0.01)
            sub_module.weight.data = sub_module.weight.data.to(device)
            sub_module.bias.data.fill_(0.01)
            sub_module.bias.data = sub_module.bias.data.to(device)

    def update_epsilon(self, iteration):
        """
        Update the epsilon based on the iteration where is
        """
        self.model.epsilon = self.epsilon_annealing[iteration]

    def flush_replay_memory(self):
        """
        When memory is full, flush the memory by removing the oldest transition
        """
        if len(self.replay) > self.model.replay_memory_size:
            self.replay.pop(0)

    def debug(self, epoch, info, best_score):
        """
        Util function for debugging purposes
        """
        if epoch % 50000 == 0:
            # save a pre-trained model
            torch.save(
                self.model,
                f"{self.path}/trained_{self.type_env}_{epoch}.pth",
            )

        if info["score"] > best_score:
            best_score = info["score"]

        if epoch % 1000 == 0:
            print(
                "epoch:",
                epoch,
                "epsilon:",
                self.model.epsilon,
                "best training score:",
                best_score,
            )

        return best_score

    def initialize_state(self):
        """
        Getting the first initial position of the bird. State 1 from which transitions to state 2
        """
        action_index, action, skip_frame = self.act(None, 0, True)

        image_data, _, _, _ = self.make_step(action_index, action)

        processed_image = self.process_image(image_data)
        state = torch.cat(
            (processed_image, processed_image, processed_image, processed_image)
        ).unsqueeze(0)
        return state

    def make_step(self, action_index, action):
        if self.type_env == 0:
            state_res, reward, done, truncated, info = self.env.step(action_index)
            image_data_state_next = self.env.render()
        else:
            image_data_state_next, reward, done, info = self.env.frame_step(action)

        return image_data_state_next, reward, done, info

    def play_step(self, action):
        if self.type_env == 0:
            state_res, _, done, _, _ = self.env.step(action)
            image_data_state_next = self.env.render()
        else:
            image_data_state_next, _, done, _ = self.env.frame_step(action, True)
        return image_data_state_next, done

    def train(self):
        """
        Main training method
        """

        state = self.initialize_state()

        # main training loop
        skip_frame = 0
        best_score = 0
        for epoch in range(self.model.epochs):
            # get prediction output
            output = self.model(state)[0]

            action_index, action, skip_frame = self.act(output, skip_frame)

            # get next state and reward

            image_data_state_next, reward, done, info = self.make_step(action_index, action)

            processed_image_state_next = self.process_image(image_data_state_next)

            # prepare next_state, action and reward for adding to replay
            next_state = torch.cat(
                (state.squeeze(0)[1:, :, :], processed_image_state_next)
            ).unsqueeze(0)
            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            # save transition to replay memory
            self.replay.append((state, action, reward, next_state, done))

            # flush memory
            self.flush_replay_memory()

            # update epsilon
            self.update_epsilon(epoch)

            # batchify
            (
                state_batch,
                action_batch,
                reward_batch,
                state_1_batch,
                done_batch,
            ) = self.batchify()

            # get output for the next state
            output_1_batch = self.model(state_1_batch)

            # get predicted action
            y_batch = self.compute_predicted_y_batches(
                done_batch, reward_batch, output_1_batch
            )

            # extract Q-value
            q_value = torch.sum(self.model(state_batch) * action_batch, dim=1)

            self.update_optimizer(q_value, y_batch)
            # set state to be state_1
            state = next_state

            # reset env in case of death
            self.reset_env(done)

            best_score = self.debug(epoch, info, best_score)

    def get_model_action(self, state):
        # get output from the neural network
        output = self.model(state)[0]
        # get predicted action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(device)

        action_index = torch.argmax(output).to(device)
        if self.type_env == 0:
            return action_index

        action[action_index] = 1
        return action

    def play(self):
        state = self.initialize_state()

        while True:
            # get output from the neural network
            action = self.get_model_action(state)

            # get next state
            next_state_image, done = self.play_step(action)
            processed_next_state_image = self.process_image(next_state_image)
            next_state = torch.cat(
                (state.squeeze(0)[1:, :, :], processed_next_state_image)
            ).unsqueeze(0)

            self.reset_env(done)

            state = next_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        help="The environment to use for training / testing. 0 is flappy bird gymnasium. 1 an "
        "easier env. default is gymnasium",
        choices=[0, 1],
        type=int,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="The path where to save the trained model. Default is current dir.",
        default=".",
    )
    parser.add_argument(
        "-a",
        "--agent",
        help="The relative / full path to agent to use for playing the game."
        " If no agent supplied for training, creates a new one",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="The device you want to train the model on. Devices available are: mps, cuda and cpu. Default is on CPU.",
        choices=["mps", "cuda", "cpu"],
        default="cpu",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=int,
        choices=[0, 1],
        help="Whether to play (1) or train (0) the agent",
    )

    args = parser.parse_args()

    agent = args.agent
    path = args.path
    env = args.env or 0
    test_or_play = args.type
    device = torch.device(args.device)

    
    if agent:
        model = torch.load(agent).eval().to(device)
    else:
        model = None

    if test_or_play == 1 and model is None:
        raise Exception("Supply agent to play")

    flappy_bird_gym = FlappyBirdTrainer(agent=model, path=path, environment=env)

    if test_or_play == 0:
        flappy_bird_gym.train()
    elif test_or_play == 1:
        flappy_bird_gym.play()
