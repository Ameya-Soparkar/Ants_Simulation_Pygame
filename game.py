# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:56:53 2023

@author: Dell
"""


import pygame
import random

from datetime import datetime, timedelta



# Define constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ANT_SIZE = 5
FOOD_SIZE = 20
ANT_SPEED = 5
FOOD_COLOR = (0, 255, 0)
NUM_ANTS = 50
NUM_FOOD = 10

# Initialize Pygame
pygame.init()

# Create the window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Ants and Food")

# Define the Ant class
class Ant:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = (255, 0, 0)
        self._starttime = datetime.now()

    def move(self):
        self.x += random.randint(-ANT_SPEED, ANT_SPEED)
        self.y += random.randint(-ANT_SPEED, ANT_SPEED)

        # Keep the ant inside the window
        if self.x < 0:
            self.x = 0
        elif self.x > WINDOW_WIDTH - ANT_SIZE:
            self.x = WINDOW_WIDTH - ANT_SIZE
        if self.y < 0:
            self.y = 0
        elif self.y > WINDOW_HEIGHT - ANT_SIZE:
            self.y = WINDOW_HEIGHT - ANT_SIZE

    def draw(self):
        pygame.draw.rect(window, self.color, (self.x, self.y, ANT_SIZE, ANT_SIZE))

# Define the Food class
class Food:
    def __init__(self):
        self.x = random.randint(0, WINDOW_WIDTH - FOOD_SIZE)
        self.y = random.randint(0, WINDOW_HEIGHT - FOOD_SIZE)
        self.color = FOOD_COLOR

    def draw(self):
        pygame.draw.rect(window, self.color, (self.x, self.y, FOOD_SIZE, FOOD_SIZE))


# Create a list of Ant objects
ants = []
for i in range(NUM_ANTS):
    x = random.randint(0, WINDOW_WIDTH - ANT_SIZE)
    y = random.randint(0, WINDOW_HEIGHT - ANT_SIZE)
    ants.append(Ant(x, y))

# Create a list of Food objects
foods = []
for i in range(NUM_FOOD):
    foods.append(Food())

# Game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    for ant in ants:
        if (datetime.now() - ant._starttime).total_seconds() > 5:
            ants.remove(ant)
    print("Number of ants ", len(ants))
    
    # Move the ants
    for ant in ants:
        ant.move()

    # Check if an ant has touched a food
    for food in foods:
        for ant in ants:
            if (ant.x < food.x + FOOD_SIZE and ant.x + ANT_SIZE > food.x and ant.y < food.y + FOOD_SIZE and ant.y + ANT_SIZE > food.y):
                foods.remove(food)
                foods.append(Food())
                ant._starttime  = ant._starttime + timedelta(seconds=10)
                
                
    

    # Draw the ants and the foods
    window.fill((255, 255, 255))
    for ant in ants:
            ant.draw()
            
    for food in foods:
        food.draw()

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()























