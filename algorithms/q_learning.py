import numpy as np
import random

class QLearningOptimizer:
    def __init__(self, env, source, target, w_delay, w_rel, w_res, 
                 episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.source = source
        self.target = target
        self.weights = (w_delay, w_rel, w_res)
        
        # Гиперпараметры RL
        self.episodes = episodes  # Сколько раз агент попытается пройти путь
        self.alpha = alpha        # Скорость обучения (Learning Rate)
        self.gamma = gamma        # Коэффициент дисконтирования (важность будущего)
        self.epsilon = epsilon    # Вероятность случайного действия (Exploration)
        
        # Q-Таблица: Словарь словарей. Q[state][next_node] = value
        # Инициализируем нулями
        self.q_table = {} 
        for node in self.env.graph.nodes():
            self.q_table[node] = {}
            for neighbor in self.env.graph.neighbors(node):
                self.q_table[node][neighbor] = 0.0

    def get_valid_actions(self, state):
        """Возвращает список соседей текущего узла."""
        return list(self.env.graph.neighbors(state))

    def choose_action(self, state):
        """Epsilon-Greedy стратегия: иногда исследуем, иногда используем знания."""
        actions = self.get_valid_actions(state)
        if not actions:
            return None

        # Случайное действие (Exploration)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        
        # Лучшее действие (Exploitation)
        # Ищем соседа с максимальным Q-значением
        max_q = -float('inf')
        best_actions = []
        
        for action in actions:
            q_val = self.q_table[state].get(action, 0.0)
            if q_val > max_q:
                max_q = q_val
                best_actions = [action]
            elif q_val == max_q:
                best_actions.append(action)
        
        return random.choice(best_actions)

    def train(self):
        """Основной цикл обучения."""
        print(f"QL: Старт обучения ({self.episodes} эпизодов)...")
        
        for episode in range(self.episodes):
            state = self.source
            path = [state]
            
            # Ограничиваем длину пути, чтобы не зациклился навечно
            max_steps = self.env.num_nodes * 2 
            
            for _ in range(max_steps):
                action = self.choose_action(state)
                if action is None:
                    break # Тупик
                
                next_state = action
                
                # Если дошли до цели
                if next_state == self.target:
                    # Считаем итоговую награду согласно PDF 
                    # Сначала соберем весь путь
                    full_path = path + [next_state]
                    cost = self.env.calculate_weighted_cost(full_path, *self.weights)
                    
                    # Избегаем деления на ноль
                    if cost == 0: cost = 0.0001
                    
                    reward = 1000.0 / cost
                    
                    # Обновляем Q-значение для последнего шага
                    # Q(s,a) = Q(s,a) + alpha * (R - Q(s,a))  <-- gamma тут 0, т.к. это конец
                    old_q = self.q_table[state][action]
                    new_q = old_q + self.alpha * (reward - old_q)
                    self.q_table[state][action] = new_q
                    break
                
                else:
                    # Мы еще не у цели. Награда пока 0 (или маленькая отрицательная за шаг)
                    # Но мы обновляем Q на основе прогноза будущего (Bootstrap)
                    reward = 0 
                    
                    # Max Q для следующего состояния
                    next_actions = self.get_valid_actions(next_state)
                    if next_actions:
                        max_next_q = max([self.q_table[next_state][a] for a in next_actions])
                    else:
                        max_next_q = 0
                    
                    # Формула Q-Learning [cite: 97]
                    old_q = self.q_table[state][action]
                    td_target = reward + self.gamma * max_next_q
                    new_q = old_q + self.alpha * (td_target - old_q)
                    self.q_table[state][action] = new_q
                    
                    # Переход
                    state = next_state
                    path.append(state)
                    
                    # Прерываем, если вернулись в начало (простой способ борьбы с циклами)
                    if len(path) > len(set(path)):
                        break

    def get_best_path(self):
        """Восстанавливает лучший путь по Q-таблице после обучения."""
        path = [self.source]
        state = self.source
        
        visited = {state} # Чтобы не попасть в бесконечный цикл при выводе
        
        while state != self.target:
            actions = self.get_valid_actions(state)
            if not actions:
                return None # Тупик
            
            # Выбираем соседа с максимальным Q
            best_action = None
            max_q = -float('inf')
            
            for action in actions:
                if action not in visited: # Не ходим назад
                    q_val = self.q_table[state].get(action, -float('inf'))
                    if q_val > max_q:
                        max_q = q_val
                        best_action = action
            
            if best_action is None or max_q == 0:
                # Если агент ничего не выучил для этого состояния, путь не найден
                return None
                
            state = best_action
            path.append(state)
            visited.add(state)
            
            if len(path) > self.env.num_nodes: # Защита
                return None
                
        # Считаем стоимость найденного пути
        cost = self.env.calculate_weighted_cost(path, *self.weights)
        return path, cost