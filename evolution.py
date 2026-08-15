# ==========================================================
# ANT COLONY EVOLUTION
#
# Features
# 1. Pheromone grid (800x600)
# 2. Ants leave pheromone trails
# 3. Pheromone evaporation
# 4. Food scent diffusion
# 5. Shared neural network
# 6. Ant reproduction
# 7. Genetic evolution
#
# pip install pygame torch numpy
# ==========================================================

import pygame
import random
import numpy as np
import torch
import torch.nn as nn

# ==========================================================
# CONFIG
# ==========================================================

WIDTH = 800
HEIGHT = 600

ANT_SIZE = 4
FOOD_SIZE = 15

INITIAL_ANTS = 50
FOOD_COUNT = 10

ANT_SPEED = 3

PHEROMONE_DEPOSIT = 10
PHEROMONE_EVAPORATION = 0.995

MAX_ANTS = 300

# ==========================================================
# PYGAME
# ==========================================================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

# ==========================================================
# GRIDS
# ==========================================================

pheromone_grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

food_scent_grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

# ==========================================================
# FOOD
# ==========================================================


class Food:

    def __init__(self):

        self.x = random.randint(0, WIDTH - FOOD_SIZE)

        self.y = random.randint(0, HEIGHT - FOOD_SIZE)

    def draw(self):

        pygame.draw.rect(
            screen,
            (0, 255, 0),
            (self.x, self.y, FOOD_SIZE, FOOD_SIZE),
        )


foods = [Food() for _ in range(FOOD_COUNT)]

# ==========================================================
# FOOD SCENT DIFFUSION
# ==========================================================


def update_food_scent():

    global food_scent_grid

    food_scent_grid *= 0.95

    for food in foods:

        x = food.x
        y = food.y

        radius = 100

        xmin = max(0, x - radius)
        xmax = min(WIDTH, x + radius)

        ymin = max(0, y - radius)
        ymax = min(HEIGHT, y + radius)

        for py in range(ymin, ymax):

            for px in range(xmin, xmax):

                distance = np.sqrt((px - x) ** 2 + (py - y) ** 2)

                if distance < radius:

                    strength = (radius - distance) / radius

                    food_scent_grid[py, px] += strength * 5


# ==========================================================
# SHARED NEURAL NETWORK
# ==========================================================


class SharedBrain(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(4, 32),

            nn.ReLU(),

            nn.Linear(32, 32),

            nn.ReLU(),

            nn.Linear(32, 8),

            nn.Tanh(),
        )

    def forward(self, x):

        return self.net(x)


shared_brain = SharedBrain()

# ==========================================================
# ANT
# ==========================================================


class Ant:

    def __init__(self, brain=None):

        self.x = random.randint(0, WIDTH - ANT_SIZE)

        self.y = random.randint(0, HEIGHT - ANT_SIZE)

        self.energy = 100

        self.food_collected = 0

        if brain is None:

            self.weights = []

            for parameter in shared_brain.parameters():

                self.weights.append(
                    parameter.detach().numpy().copy()
                )

        else:

            self.weights = brain

    def mutate(self):

        for weight in self.weights:

            mask = np.random.random(weight.shape) < 0.05

            weight += mask * np.random.normal(
                0,
                0.1,
                weight.shape,
            )

    def sense(self):

        px = int(self.x)

        py = int(self.y)

        pheromone = pheromone_grid[py, px]

        scent = food_scent_grid[py, px]

        return np.array(
            [
                self.x / WIDTH,
                self.y / HEIGHT,
                pheromone / 100,
                scent / 100,
            ],
            dtype=np.float32,
        )

    def move(self):

        state = self.sense()

        x = torch.FloatTensor(state)

        with torch.no_grad():

            output = shared_brain(x).numpy()

        dx = output[0] * ANT_SPEED

        dy = output[1] * ANT_SPEED

        dx += random.uniform(-1, 1)

        dy += random.uniform(-1, 1)

        self.x += dx

        self.y += dy

        self.x = max(
            0,
            min(self.x, WIDTH - ANT_SIZE),
        )

        self.y = max(
            0,
            min(self.y, HEIGHT - ANT_SIZE),
        )

        self.energy -= 0.1

        pheromone_grid[
            int(self.y),
            int(self.x),
        ] += PHEROMONE_DEPOSIT

    def check_food(self):

        for food in foods:

            if (

                self.x < food.x + FOOD_SIZE

                and self.x + ANT_SIZE > food.x

                and self.y < food.y + FOOD_SIZE

                and self.y + ANT_SIZE > food.y

            ):

                self.energy += 50

                self.food_collected += 1

                foods.remove(food)

                foods.append(Food())

                break

    def reproduce(self):

        if (

            self.food_collected >= 3

            and len(ants) < MAX_ANTS

        ):

            child = Ant(
                brain=[
                    w.copy()
                    for w in self.weights
                ]
            )

            child.mutate()

            self.food_collected = 0

            ants.append(child)

    def dead(self):

        return self.energy <= 0

    def draw(self):

        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (
                int(self.x),
                int(self.y),
                ANT_SIZE,
                ANT_SIZE,
            ),
        )


# ==========================================================
# EVOLUTION
# ==========================================================


def evolve():

    if len(ants) < 10:

        return

    ants.sort(
        key=lambda a: a.food_collected,
        reverse=True,
    )

    survivors = ants[: len(ants) // 2]

    while len(survivors) < INITIAL_ANTS:

        parent = random.choice(survivors)

        child = Ant(
            brain=[
                w.copy()
                for w in parent.weights
            ]
        )

        child.mutate()

        survivors.append(child)

    ants.clear()

    ants.extend(survivors)


# ==========================================================
# CREATE ANTS
# ==========================================================

ants = [Ant() for _ in range(INITIAL_ANTS)]

generation_timer = 0

# ==========================================================
# MAIN LOOP
# ==========================================================

running = True

while running:

    clock.tick(60)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

    pheromone_grid *= PHEROMONE_EVAPORATION

    update_food_scent()

    for ant in ants[:]:

        ant.move()

        ant.check_food()

        ant.reproduce()

        if ant.dead():

            ants.remove(ant)

    generation_timer += 1

    if generation_timer > 2000:

        evolve()

        generation_timer = 0

    screen.fill((255, 255, 255))

    for food in foods:

        food.draw()

    for ant in ants:

        ant.draw()

    pygame.display.set_caption(

        f"Generation Ants: {len(ants)}"
    )

    pygame.display.flip()

pygame.quit()