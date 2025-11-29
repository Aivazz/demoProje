import networkx as nx
import random
import math
import numpy as np

class NetworkEnvironment:
    def __init__(self, num_nodes=250, connection_prob=0.4, seed=42):
        self.num_nodes = num_nodes
        self.prob = connection_prob
        self.seed = seed
        self.graph = None
        
        # Генерация сети при инициализации
        self.generate_network()

    def generate_network(self):
        """
        Создает граф согласно требованиям Раздела 2.1 PDF.
        """
        print(f"Генерация сети: {self.num_nodes} узлов, вероятность связи {self.prob}...")
        
        # Используем seed для воспроизводимости (требование 7.2)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # 1. Топология Erdős-Rényi (cite: 25)
        while True:
            # Создаем граф
            G = nx.erdos_renyi_graph(self.num_nodes, self.prob, seed=self.seed)
            
            # 2. Проверка на связность (cite: 26)
            if nx.is_connected(G):
                self.graph = G
                break
            else:
                print("Граф не связный, пересоздаем...")
                self.seed += 1  # меняем seed для новой попытки

        # 3. Назначение свойств УЗЛАМ (cite: 27-30)
        for node in self.graph.nodes():
            # Задержка обработки: 0.5 - 2.0 ms
            proc_delay = random.uniform(0.5, 2.0)
            # Надежность узла: 0.95 - 0.999
            reliability = random.uniform(0.95, 0.999)
            
            self.graph.nodes[node]['proc_delay'] = proc_delay
            self.graph.nodes[node]['reliability'] = reliability
            # Предрасчет стоимости надежности для узла (-log)
            self.graph.nodes[node]['rel_cost'] = -math.log(reliability)

        # 4. Назначение свойств СВЯЗЯМ (cite: 31-35)
        for u, v in self.graph.edges():
            # Пропускная способность: 100 - 1000 Mbps
            bw = random.uniform(100, 1000)
            # Задержка канала: 3 - 15 ms
            link_delay = random.uniform(3, 15)
            # Надежность канала: 0.95 - 0.999
            link_rel = random.uniform(0.95, 0.999)

            self.graph[u][v]['bandwidth'] = bw
            self.graph[u][v]['delay'] = link_delay
            self.graph[u][v]['reliability'] = link_rel
            
            # Предрасчет метрик для быстрого доступа
            self.graph[u][v]['rel_cost'] = -math.log(link_rel)
            # Стоимость ресурсов: 1000 / Bandwidth (cite: 57)
            # 1 Gbps = 1000 Mbps
            self.graph[u][v]['res_cost'] = 1000.0 / bw

        print("Сеть успешно создана.")

    def calculate_path_metrics(self, path):
        """
        Считает метрики для конкретного пути (список узлов).
        Соответствует Разделу 3 PDF.
        """
        if not path or len(path) < 2:
            return float('inf'), float('inf'), float('inf')

        total_delay = 0.0
        total_rel_cost = 0.0
        total_res_cost = 0.0

        # Проходим по всем сегментам пути
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # Данные ребра
            edge_data = self.graph[u][v]
            
            # 3.1 Задержка: Link Delay (cite: 42)
            total_delay += edge_data['delay']
            
            # 3.2 Надежность (Cost): Link Reliability Cost (cite: 52)
            total_rel_cost += edge_data['rel_cost']
            
            # 3.3 Ресурсы: Resource Cost (cite: 57)
            total_res_cost += edge_data['res_cost']

            # Добавляем Node Processing Delay и Node Reliability
            # ВАЖНО: Не учитываем для начального (S) узла, 
            # но учитываем для текущего 'v', если он не конечный.
            # Однако, в формуле (cite: 42) сказано: "промежуточные узлы".
            # Если мы идем u -> v, то 'v' становится промежуточным, 
            # если это не самый последний узел пути.
            if i < len(path) - 2: # Если v - не последний узел
                node_data = self.graph.nodes[v]
                total_delay += node_data['proc_delay']
                total_rel_cost += node_data['rel_cost']

        return total_delay, total_rel_cost, total_res_cost

    def calculate_weighted_cost(self, path, w_delay, w_rel, w_res):
        """
        Считает общую взвешенную стоимость (Fitness).
        Раздел 4, пункт 4 (cite: 66).
        """
        d, r, res = self.calculate_path_metrics(path)
        return (w_delay * d) + (w_rel * r) + (w_res * res)

# Простой тест (чтобы проверить, что работает)
if __name__ == "__main__":
    env = NetworkEnvironment()
    # Берем случайные S и D
    nodes = list(env.graph.nodes())
    S, D = nodes[0], nodes[-1]
    
    # Пытаемся найти кратчайший путь (по количеству прыжков) для теста
    try:
        path = nx.shortest_path(env.graph, S, D)
        print(f"Тестовый путь от {S} к {D}: {path}")
        
        d, r, res = env.calculate_path_metrics(path)
        print(f"Metrics -> Delay: {d:.2f}ms, RelCost: {r:.4f}, ResCost: {res:.4f}")
        
        cost = env.calculate_weighted_cost(path, 0.33, 0.33, 0.34)
        print(f"Weighted Cost: {cost:.4f}")
        
    except nx.NetworkXNoPath:
        print("Путь не найден (теоретически невозможно при связном графе).")