import random
import networkx as nx

class GeneticOptimizer:
    def __init__(self, env, source, target, w_delay, w_rel, w_res, 
                 pop_size=50, generations=100, mutation_rate=0.2):
        self.env = env
        self.graph = env.graph
        self.source = source
        self.target = target
        
        # Веса для расчета стоимости (cite: 66)
        self.weights = (w_delay, w_rel, w_res)
        
        # Параметры GA
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.population = []

    def get_fitness(self, path):

        return self.env.calculate_weighted_cost(path, *self.weights)

    def create_random_path(self):

        try:
            # Используем генератор all_simple_paths с ограничением cutoff, 
            # чтобы не искать слишком долго, или делаем свой random walk.
            # Для скорости и разнообразия используем 'randomized shortest path' эвристику:
            # Временно меняем веса ребер на случайные и ищем кратчайший путь.
            
            for u, v in self.graph.edges():
                self.graph[u][v]['temp_weight'] = random.random()
                
            path = nx.shortest_path(self.graph, self.source, self.target, weight='temp_weight')
            return path
        except nx.NetworkXNoPath:
            return None

    def initialize_population(self):
        """Создает стартовую популяцию путей."""
        print("GA: Инициализация популяции...")
        self.population = []
        attempts = 0
        while len(self.population) < self.pop_size and attempts < self.pop_size * 5:
            path = self.create_random_path()
            if path and path not in self.population:
                self.population.append(path)
            attempts += 1
        
        # Сортируем популяцию по стоимости (от лучшего к худшему)
        self.population.sort(key=self.get_fitness)

    def crossover(self, parent1, parent2):

        # Ищем общие узлы (исключая S и D, чтобы было интереснее, но можно и с ними)
        common_nodes = [node for node in parent1 if node in parent2 and node != self.source and node != self.target]
        
        if not common_nodes:
            return parent1, parent2  # Скрещивание невозможно, возвращаем как есть

        # Выбираем точку разрыва
        pivot = random.choice(common_nodes)
        
        # Индексы точки разрыва
        idx1 = parent1.index(pivot)
        idx2 = parent2.index(pivot)
        
        # Создаем потомков: начало от одного, конец от другого
        child1 = parent1[:idx1] + parent2[idx2:]
        child2 = parent2[:idx2] + parent1[idx1:]
        
        # Важно: Проверка на циклы! Путь не должен содержать повторяющихся узлов.
        if len(child1) != len(set(child1)): child1 = parent1 # Откат, если цикл
        if len(child2) != len(set(child2)): child2 = parent2 # Откат
            
        return child1, child2

    def mutate(self, path):
        """
        Мутация (cite: 82).
        Выбираем узел и пытаемся перепроложить маршрут от него до D.
        """
        if random.random() > self.mutation_rate:
            return path
            
        if len(path) < 3: return path

        # Выбираем случайный узел разрыва (кроме последнего)
        cut_idx = random.randint(1, len(path) - 2)
        cut_node = path[cut_idx]
        
        # Пытаемся найти новый кусок пути от cut_node до target
        # Опять используем трюк со случайными весами для разнообразия
        try:
            # Временные веса для разнообразия
            for u, v in self.graph.edges():
                self.graph[u][v]['mut_weight'] = random.random()
            
            # Ищем путь от точки разрыва до конца
            new_tail = nx.shortest_path(self.graph, cut_node, self.target, weight='mut_weight')
            
            # Склеиваем: начало старого пути + новый хвост
            new_path = path[:cut_idx] + new_tail
            
            # Проверка на циклы
            if len(new_path) == len(set(new_path)):
                return new_path
        except:
            pass
            
        return path

    def run(self):
        """Запуск основного цикла эволюции."""
        self.initialize_population()
        
        if not self.population:
            print("GA: Не удалось создать начальную популяцию.")
            return None, float('inf')

        for generation in range(self.generations):
            new_population = []
            
            # Элитизм: сохраняем 2 лучших пути без изменений
            new_population.extend(self.population[:2])
            
            while len(new_population) < self.pop_size:
                # Селекция: Турнирный отбор (берем случайных и выбираем лучшего)
                parent1 = min(random.sample(self.population, 5), key=self.get_fitness)
                parent2 = min(random.sample(self.population, 5), key=self.get_fitness)
                
                # Скрещивание
                child1, child2 = self.crossover(parent1, parent2)
                
                # Мутация
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # Обновляем популяцию и сортируем
            self.population = new_population
            self.population.sort(key=self.get_fitness)
            
            # (Опционально) Вывод прогресса
            # best_cost = self.get_fitness(self.population[0])
            # print(f"Gen {generation}: Best Cost = {best_cost:.4f}")

        best_path = self.population[0]
        best_cost = self.get_fitness(best_path)
        return best_path, best_cost