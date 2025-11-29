import time
import random
import numpy as np
import matplotlib.pyplot as plt # Импортируем библиотеку для графиков

# Импортируем наши модули
from network_model import NetworkEnvironment
from algorithms.genetic import GeneticOptimizer
from algorithms.q_learning import QLearningOptimizer
from utils import save_results_to_csv, generate_report_name

def plot_results(results):

    print("\nГенерация графиков...")

    # Разделяем данные по алгоритмам
    ga_times = [r['Time_ms'] for r in results if r['Algorithm'] == 'Genetic Algorithm']
    ql_times = [r['Time_ms'] for r in results if r['Algorithm'] == 'Q-Learning']

    # Фильтруем стоимость (убираем "Not Found" и бесконечность)
    ga_costs = [r['Cost'] for r in results if r['Algorithm'] == 'Genetic Algorithm' and isinstance(r['Cost'], (int, float)) and r['Cost'] != float('inf')]
    ql_costs = [r['Cost'] for r in results if r['Algorithm'] == 'Q-Learning' and isinstance(r['Cost'], (int, float)) and r['Cost'] != float('inf')]

    # Считаем средние значения
    avg_ga_time = np.mean(ga_times) if ga_times else 0
    avg_ql_time = np.mean(ql_times) if ql_times else 0

    avg_ga_cost = np.mean(ga_costs) if ga_costs else 0
    avg_ql_cost = np.mean(ql_costs) if ql_costs else 0

    algorithms = ['Genetic Algorithm', 'Q-Learning']
    colors = ['#4CAF50', '#2196F3'] # Зеленый и Синий

    # --- График 1: Сравнение Времени ---
    plt.figure(figsize=(10, 6))
    times = [avg_ga_time, avg_ql_time]
    bars = plt.bar(algorithms, times, color=colors, alpha=0.7)
    
    # Добавляем подписи значений над столбиками
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f} ms',
                 ha='center', va='bottom')

    plt.ylabel('Average Execution Time (ms)')
    plt.title('Performance Comparison: Execution Time')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('benchmark_time.png') # Сохраняем картинку
    print("График времени сохранен как 'benchmark_time.png'")
    plt.close()

    # --- График 2: Сравнение Стоимости (Качество) ---
    plt.figure(figsize=(10, 6))
    costs = [avg_ga_cost, avg_ql_cost]
    bars = plt.bar(algorithms, costs, color=colors, alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.ylabel('Average Weighted Cost (Lower is Better)')
    plt.title('Quality Comparison: Path Cost')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('benchmark_cost.png') # Сохраняем картинку
    plt.close()
    print("График стоимости сохранен как 'benchmark_cost.png'")

def run_benchmark():
    # 1. Создаем среду
    print("Инициализация сети для тестов...")
    env = NetworkEnvironment(num_nodes=250, connection_prob=0.4, seed=42)
    
    # Параметры (чуть уменьшил для скорости демонстрации)
    NUM_TEST_CASES = 20  
    REPEATS = 5           
    
    W_DELAY = 0.33
    W_REL = 0.33
    W_RES = 0.34
    
    results = []
    
    print(f"Начинаем тестирование: {NUM_TEST_CASES} сценариев x {REPEATS} повторов.")

    # Генерируем пары S->D
    test_cases = []
    nodes = list(env.graph.nodes())
    for _ in range(NUM_TEST_CASES):
        s = random.choice(nodes)
        d = random.choice(nodes)
        while s == d: d = random.choice(nodes)
        test_cases.append((s, d))

    # Цикл тестов
    for i, (s, d) in enumerate(test_cases):
        print(f"Тест {i+1}/{NUM_TEST_CASES} (S={s} -> D={d})...")
        
        # GA
        for r in range(REPEATS):
            start = time.time()
            ga = GeneticOptimizer(env, s, d, W_DELAY, W_REL, W_RES, pop_size=50, generations=50)
            path, cost = ga.run()
            duration = (time.time() - start) * 1000
            
            results.append({
                "Test_ID": i+1, "Source": s, "Destination": d,
                "Algorithm": "Genetic Algorithm", "Run_ID": r+1,
                "Time_ms": round(duration, 2), "Cost": round(cost, 4) if cost != float('inf') else float('inf'),
                "Path_Length": len(path) if path else 0
            })

        # Q-Learning
        for r in range(REPEATS):
            start = time.time()
            ql = QLearningOptimizer(env, s, d, W_DELAY, W_REL, W_RES, episodes=500)
            ql.train()
            path, cost = ql.get_best_path()
            duration = (time.time() - start) * 1000
            
            if path is None: cost = float('inf')
                
            results.append({
                "Test_ID": i+1, "Source": s, "Destination": d,
                "Algorithm": "Q-Learning", "Run_ID": r+1,
                "Time_ms": round(duration, 2), "Cost": round(cost, 4) if cost != float('inf') else float('inf'),
                "Path_Length": len(path) if path else 0
            })

    # Сохраняем CSV
    filename = generate_report_name()
    save_results_to_csv(results, filename)
    print(f"\nДанные сохранены в {filename}")

    # Рисуем графики
    plot_results(results)

if __name__ == "__main__":
    run_benchmark()