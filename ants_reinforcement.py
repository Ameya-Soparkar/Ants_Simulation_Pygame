import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------- CONSTANTS -------------------

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

ANT_SIZE = 5
FOOD_SIZE = 20

ANT_SPEED = 5

NUM_ANTS = 50
NUM_FOOD = 10

MAX_MEMORY = 10000
BATCH_SIZE = 128

# ------------------- PYGAME -------------------

pygame.init()

window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

pygame.display.set_caption("Learning Ants")

clock = pygame.time.Clock()

# ------------------- NEURAL NETWORK -------------------


class AntNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(4, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, 4)

        )

    def forward(self, x):

        return self.model(x)


network = AntNetwork()

optimizer = optim.Adam(network.parameters(), lr=0.001)

loss_function = nn.MSELoss()

memory = []

epsilon = 1.0

# ------------------- FOOD -------------------


class Food:

    def __init__(self):

        self.x = random.randint(0, WINDOW_WIDTH - FOOD_SIZE)

        self.y = random.randint(0, WINDOW_HEIGHT - FOOD_SIZE)

    def draw(self):

        pygame.draw.rect(
            window,
            (0, 255, 0),
            (self.x, self.y, FOOD_SIZE, FOOD_SIZE)
        )


# ------------------- ANT -------------------


class Ant:

    def __init__(self):

        self.x = random.randint(0, WINDOW_WIDTH - ANT_SIZE)

        self.y = random.randint(0, WINDOW_HEIGHT - ANT_SIZE)

    def get_nearest_food(self, foods):

        nearest = foods[0]

        nearest_distance = 999999

        for food in foods:

            distance = math.sqrt(
                (self.x - food.x) ** 2 +
                (self.y - food.y) ** 2
            )

            if distance < nearest_distance:

                nearest_distance = distance

                nearest = food

        return nearest

    def get_state(self, foods):

        food = self.get_nearest_food(foods)

        return np.array([

            self.x / WINDOW_WIDTH,

            self.y / WINDOW_HEIGHT,

            food.x / WINDOW_WIDTH,

            food.y / WINDOW_HEIGHT

        ], dtype=np.float32)

    def move(self, action):

        old_x = self.x

        old_y = self.y

        if action == 0:

            self.y -= ANT_SPEED

        elif action == 1:

            self.y += ANT_SPEED

        elif action == 2:

            self.x -= ANT_SPEED

        elif action == 3:

            self.x += ANT_SPEED

        self.x = max(
            0,
            min(self.x, WINDOW_WIDTH - ANT_SIZE)
        )

        self.y = max(
            0,
            min(self.y, WINDOW_HEIGHT - ANT_SIZE)
        )

        return old_x, old_y

    def draw(self):

        pygame.draw.rect(
            window,
            (255, 0, 0),
            (self.x, self.y, ANT_SIZE, ANT_SIZE)
        )


# ------------------- TRAINING -------------------


def train():

    if len(memory) < BATCH_SIZE:

        return

    batch = random.sample(memory, BATCH_SIZE)

    states = torch.FloatTensor([x[0] for x in batch])

    actions = [x[1] for x in batch]

    rewards = [x[2] for x in batch]

    next_states = torch.FloatTensor([x[3] for x in batch])

    current_q = network(states)

    next_q = network(next_states).detach()

    target_q = current_q.clone().detach()

    gamma = 0.9

    for i in range(BATCH_SIZE):

        target_q[i][actions[i]] = rewards[i] + gamma * torch.max(
            next_q[i]
        )

    loss = loss_function(current_q, target_q)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


# ------------------- CREATE OBJECTS -------------------

ants = [Ant() for _ in range(NUM_ANTS)]

foods = [Food() for _ in range(NUM_FOOD)]

# ------------------- MAIN LOOP -------------------

running = True

while running:

    clock.tick(60)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


    if epsilon > 0.05:

        epsilon *= 0.9995

    for ant in ants:

        state = ant.get_state(foods)

        nearest_food = ant.get_nearest_food(foods)

        old_distance = math.sqrt(

            (ant.x - nearest_food.x) ** 2 +

            (ant.y - nearest_food.y) ** 2

        )

        if random.random() < epsilon:

            action = random.randint(0, 3)

        else:

            with torch.no_grad():

                q_values = network(
                    torch.FloatTensor(state)
                )

                action = torch.argmax(q_values).item()

        ant.move(action)

        nearest_food = ant.get_nearest_food(foods)

        new_distance = math.sqrt(

            (ant.x - nearest_food.x) ** 2 +

            (ant.y - nearest_food.y) ** 2

        )

        reward = 1 if new_distance < old_distance else -1

        for food in foods:

            if (

                ant.x < food.x + FOOD_SIZE

                and ant.x + ANT_SIZE > food.x

                and ant.y < food.y + FOOD_SIZE

                and ant.y + ANT_SIZE > food.y

            ):

                reward = 100

                foods.remove(food)

                foods.append(Food())

                break

        next_state = ant.get_state(foods)

        memory.append(

            (

                state,

                action,

                reward,

                next_state

            )

        )

        if len(memory) > MAX_MEMORY:

            memory.pop(0)

    train()

    window.fill((255, 255, 255))

    for food in foods:

        food.draw()

    for ant in ants:

        ant.draw()

    pygame.display.update()

pygame.quit()