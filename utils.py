import csv
import os
import time

def save_results_to_csv(results, filename="experiment_results.csv"):

    file_exists = os.path.isfile(filename)
    
    # Определяем заголовки таблицы по ключам первого результата
    if not results:
        return
    fieldnames = results[0].keys()
    
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Если файл новый - пишем заголовки
        if not file_exists:
            writer.writeheader()
            
        # Пишем данные
        for row in results:
            writer.writerow(row)
            
    print(f"Результаты успешно добавлены в {filename}")

def generate_report_name():
    """Генерирует имя файла с текущей датой."""
    return f"report_{time.strftime('%Y%m%d_%H%M%S')}.csv"