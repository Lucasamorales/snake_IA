import torch
import random
import numpy as np 
from collections import deque
from entorno import SnakeGameIA, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 

class Agente:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma)
        
    
    def getState(self, game): # vamos a tener 11 estados 
        head = game.snake[0]
        # creamos puntos para cheuquear si la cabeza choca con algo 
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        #chequeamos la direccion 
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # chequeamos si el peligro esta delante de la cabeza de la serpiente
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # chequeamos si el peligro esta hacia la derecha 
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # chequeamos si el peligro esta hacia la izquierda 
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # direccion del movimiento 
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # chequiamos donde esta la comida
            game.food.x < game.head.x,  # la comida esta hacia la izquierda
            game.food.x > game.head.x,  # la comida esta hacia la derecha
            game.food.y < game.head.y,  # la comida esta hacia arriba
            game.food.y > game.head.y  # la comida esta hacia abajo
            ]

        return np.array(state, dtype=int) #convertimos los booleanos en 0 o 1 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else: 
            mini_sample = self.memory 

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def getAction(self, state):
        # movimientos aleatorios 
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]

        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0) #equivalente al predict de tensorflow 
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move 

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agente()
    game = SnakeGameIA()

    while True:
        #obtenemos el estado anterior 
        state_old = agent.getState(game)

        #obtenemos una accion
        final_move = agent.getAction(state_old)

        #realizamos la accion y obtenemos un nuevo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.getState(game)

        # entrenamos con memoria a corto plazo 
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #recordar
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # entrenar la memoria de largo plazo, y graficar los resultados
            game.reset()
            agent.n_games +=1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
if __name__ == '__name__':
    train()
