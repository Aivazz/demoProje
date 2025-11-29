import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
# NavigationToolbar2Tk: Araç çubuğu (Zoom/Pan) için
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import networkx as nx
import time
import numpy as np
import random

# Modüllerimizi içe aktarıyoruz
from network_model import NetworkEnvironment
from algorithms.genetic import GeneticOptimizer
from algorithms.q_learning import QLearningOptimizer
from utils import save_results_to_csv, generate_report_name

# --- RENK PALETİ (CYBERPUNK / DARK MODE) ---
BG_COLOR = "#2b2b2b"        # Koyu gri arka plan
PANEL_COLOR = "#333333"     # Panel rengi
TEXT_COLOR = "#ffffff"      # Beyaz metin
ACCENT_COLOR = "#00d4ff"    # Vurgu rengi (Cyan) - Düğmeler ve düğümler için
PATH_COLOR = "#ff3366"      # Yol rengi (Neon Kırmızı/Pembe)
EDGE_COLOR = "#ffffff"      # Bağlantı rengi
INPUT_BG = "#4d4d4d"        # Giriş kutusu arka planı

class NetworkVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BSM307 - QoS Rotalama Projesi")
        self.root.geometry("1280x850")
        self.root.configure(bg=BG_COLOR)

        self.setup_styles()

        # 1. Ağın Başlatılması (Network Initialization)
        self.env = NetworkEnvironment()
        try:
            # Kamada-Kawai düzeni daha estetik görünür
            self.pos = nx.kamada_kawai_layout(self.env.graph)
        except:
            self.pos = nx.spring_layout(self.env.graph, seed=42, k=0.15)

        # --- GUI (Arayüz) ---
        # Sol Panel (Kontrol Paneli)
        control_frame = tk.Frame(self.root, bg=PANEL_COLOR, padx=20, pady=20, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        control_frame.pack_propagate(False)

        # Başlık
        tk.Label(control_frame, text="BSM307 PROJESİ", font=("Segoe UI", 16, "bold"), 
                 bg=PANEL_COLOR, fg=ACCENT_COLOR).pack(pady=(0, 20))

        # Ayarlar Bölümü
        self.create_label(control_frame, "Kaynak Düğüm (S):")
        self.s_entry = self.create_entry(control_frame, "0")
        
        self.create_label(control_frame, "Hedef Düğüm (D):")
        self.d_entry = self.create_entry(control_frame, str(self.env.num_nodes - 1))

        self.create_separator(control_frame)

        self.create_label(control_frame, "Ağırlıklar (Toplam=1.0):")
        self.w_delay_entry = self.create_labeled_entry(control_frame, "Gecikme:", "0.33")
        self.w_rel_entry = self.create_labeled_entry(control_frame, "Güvenilirlik:", "0.33")
        self.w_res_entry = self.create_labeled_entry(control_frame, "Kaynaklar:", "0.34")

        self.create_separator(control_frame)

        self.create_label(control_frame, "Algoritma:")
        self.algo_combo = ttk.Combobox(control_frame, values=[
            "Genetik Algoritma (GA)", 
            "Pekiştirmeli Öğrenme (Q-Learning)"
        ], state="readonly", font=("Segoe UI", 10))
        self.algo_combo.current(0)
        self.algo_combo.pack(fill='x', pady=5)

        # Hesaplama Butonu
        tk.Button(control_frame, text="ROTA HESAPLA", command=self.calculate_path,
                  bg=ACCENT_COLOR, fg="black", font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=5).pack(fill='x', pady=15)

        self.create_separator(control_frame)
        
        # Benchmark Bölümü
        tk.Label(control_frame, text="KIYASLAMA (BENCHMARK)", font=("Segoe UI", 12, "bold"), bg=PANEL_COLOR, fg=PATH_COLOR).pack(pady=5)
        
        tk.Button(control_frame, text="TESTİ BAŞLAT (20x5)", command=self.run_full_benchmark,
                  bg=PATH_COLOR, fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=5).pack(fill='x', pady=5)

        # Log Alanı
        tk.Label(control_frame, text="Log (Kayıt):", bg=PANEL_COLOR, fg="gray", font=("Segoe UI", 9)).pack(anchor="w", pady=(10,0))
        self.result_text = tk.Text(control_frame, height=10, bg=INPUT_BG, fg=TEXT_COLOR, 
                                   font=("Consolas", 9), relief="flat", bd=5)
        self.result_text.pack(fill='both', expand=True, pady=5)

        # --- Sağ Panel (Grafik) ---
        right_frame = tk.Frame(self.root, bg=BG_COLOR)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        plt.style.use('dark_background')
        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.figure.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().configure(highlightthickness=0, borderwidth=0)

        # --- ARAÇ ÇUBUĞU (ZOOM/PAN) EKLENMESİ ---
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        self.toolbar.update()
        # Araç çubuğunu koyu tema için stilize etme
        self.toolbar.config(background=BG_COLOR)
        self.toolbar._message_label.config(background=BG_COLOR, foreground=TEXT_COLOR)
        for button in self.toolbar.winfo_children():
            button.config(background=BG_COLOR)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # --- MOUSE TEKERLEĞİ İLE ZOOM ÖZELLİĞİ ---
        self.canvas.mpl_connect('scroll_event', self.zoom_with_scroll)

        self.draw_network()

    # --- Mouse Tekerleği ile Zoom Fonksiyonu ---
    def zoom_with_scroll(self, event):
        ax = self.ax
        # Mevcut sınırları al
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata = event.xdata # İmleç koordinatları
        ydata = event.ydata
        
        if xdata is None or ydata is None: return

        # Ölçekleme faktörü
        base_scale = 1.2
        if event.button == 'up':
            # Yakınlaştır (Zoom In)
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # Uzaklaştır (Zoom Out)
            scale_factor = base_scale
        else:
            scale_factor = 1
            
        # Yeni sınırları hesapla
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
        
        self.canvas.draw()

    # --- Diğer Metodlar ---
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground=INPUT_BG, background=PANEL_COLOR, foreground=TEXT_COLOR, arrowcolor="white")
        style.map('TCombobox', fieldbackground=[('readonly', INPUT_BG)], selectbackground=[('readonly', INPUT_BG)], selectforeground=[('readonly', TEXT_COLOR)])

    def create_label(self, parent, text):
        lbl = tk.Label(parent, text=text, bg=PANEL_COLOR, fg=TEXT_COLOR, font=("Segoe UI", 10))
        lbl.pack(anchor="w", pady=(5, 0))
        return lbl

    def create_entry(self, parent, default_val):
        ent = tk.Entry(parent, bg=INPUT_BG, fg=TEXT_COLOR, insertbackground="white", relief="flat", font=("Segoe UI", 10))
        ent.insert(0, default_val)
        ent.pack(fill='x', pady=2, ipady=3)
        return ent
    
    def create_labeled_entry(self, parent, label_text, default_val):
        frame = tk.Frame(parent, bg=PANEL_COLOR)
        frame.pack(fill='x', pady=1)
        tk.Label(frame, text=label_text, bg=PANEL_COLOR, fg="gray", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ent = tk.Entry(frame, bg=INPUT_BG, fg=TEXT_COLOR, insertbackground="white", relief="flat", width=10, justify="right")
        ent.insert(0, default_val)
        ent.pack(side=tk.RIGHT)
        return ent

    def create_separator(self, parent):
        tk.Frame(parent, height=1, bg="#555555").pack(fill='x', pady=10)

    def draw_network(self, path=None, title_suffix=""):
        self.ax.clear()
        
        # Bağlantıları çiz (Şeffaf)
        nx.draw_networkx_edges(
            self.env.graph, self.pos, ax=self.ax, 
            width=0.8, alpha=0.15, edge_color=EDGE_COLOR
        )
        
        # Düğümleri çiz
        nx.draw_networkx_nodes(
            self.env.graph, self.pos, ax=self.ax, 
            node_size=40, node_color=ACCENT_COLOR, 
            linewidths=0
        )
        
        # Yol bulunduysa vurgula (Neon Efekti)
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(
                self.env.graph, self.pos, ax=self.ax, edgelist=path_edges, 
                width=6.0, alpha=0.4, edge_color=PATH_COLOR
            )
            nx.draw_networkx_edges(
                self.env.graph, self.pos, ax=self.ax, edgelist=path_edges, 
                width=2.0, alpha=1.0, edge_color="white"
            )
            nx.draw_networkx_nodes(
                self.env.graph, self.pos, ax=self.ax, nodelist=path, 
                node_size=80, node_color="white"
            )
            labels = {path[0]: 'S', path[-1]: 'D'}
            text_items = nx.draw_networkx_labels(
                self.env.graph, self.pos, ax=self.ax, labels=labels, 
                font_color='black', font_weight='bold', font_size=10
            )
            for _, t in text_items.items():
                t.set_bbox(dict(facecolor=ACCENT_COLOR, alpha=0.8, edgecolor='none', pad=2))

        self.ax.set_title(f"AĞ TOPOLOJİSİ {title_suffix}", color="white", fontsize=12, pad=10)
        self.ax.axis('off')
        self.canvas.draw()
        self.root.update()

    def calculate_path(self):
        try:
            s = int(self.s_entry.get())
            d = int(self.d_entry.get())
            w_d = float(self.w_delay_entry.get())
            w_r = float(self.w_rel_entry.get())
            w_res = float(self.w_res_entry.get())

            if not (0.99 <= w_d + w_r + w_res <= 1.01):
                messagebox.showwarning("Hata", "Ağırlıkların toplamı 1.0 olmalıdır")
                return

            selected_algo = self.algo_combo.get()
            self.log(f"{s} -> {d} rotası hesaplanıyor...", clear=True)

            start_time = time.time()
            path = None
            algo_name = ""
            
            if "Genetik" in selected_algo:
                ga = GeneticOptimizer(self.env, s, d, w_d, w_r, w_res, pop_size=50, generations=50)
                path, cost = ga.run()
                algo_name = "GA"
            elif "Q-Learning" in selected_algo:
                ql = QLearningOptimizer(self.env, s, d, w_d, w_r, w_res, episodes=1500)
                ql.train()
                path, cost = ql.get_best_path()
                algo_name = "QL"

            duration = (time.time() - start_time) * 1000
            
            if not path:
                self.log("Yol bulunamadı!")
                return

            delay, rel_cost, res_cost = self.env.calculate_path_metrics(path)
            total_cost = self.env.calculate_weighted_cost(path, w_d, w_r, w_res)

            self.log(f"Algoritma: {algo_name}")
            self.log(f"Süre: {duration:.1f} ms")
            self.log(f"Maliyet (Cost): {total_cost:.4f}")
            self.log(f"Uzunluk: {len(path)} düğüm")
            
            self.draw_network(path)

        except ValueError: messagebox.showerror("Hata", "Girişi kontrol edin")

    def run_full_benchmark(self):
        if not messagebox.askyesno("Kıyaslama", "Görsel test başlatılsın mı (20 senaryo)?"):
            return

        self.log("Kıyaslama başlıyor...", clear=True)
        
        NUM_SCENARIOS = 20
        REPEATS = 5
        w_d, w_r, w_res = 0.33, 0.33, 0.34
        nodes = list(self.env.graph.nodes())

        scenarios = []
        for _ in range(NUM_SCENARIOS):
            s = random.choice(nodes)
            d = random.choice(nodes)
            while s == d: d = random.choice(nodes)
            scenarios.append((s, d))

        all_results_csv = []
        ga_total_times = []
        ga_total_costs = []
        ql_total_times = []
        ql_total_costs = []

        start_total = time.time()

        for i, (s, d) in enumerate(scenarios):
            self.log(f"Test {i+1}/{NUM_SCENARIOS}: {s}->{d}")
            self.result_text.see(tk.END)

            # GA (Genetik Algoritma)
            for r in range(REPEATS):
                st = time.time()
                ga = GeneticOptimizer(self.env, s, d, w_d, w_r, w_res, pop_size=30, generations=30)
                path, cost = ga.run()
                dur = (time.time() - st) * 1000
                ga_total_times.append(dur)
                ga_total_costs.append(cost)
                all_results_csv.append({"Test_ID": i+1, "Source": s, "Destination": d, "Algorithm": "Genetic Algorithm", "Run_ID": r+1, "Time_ms": dur, "Cost": cost, "Path_Length": len(path) if path else 0})
                
                if path and r == 0: 
                    self.draw_network(path, title_suffix=f"| Test {i+1} | GA")

            # QL (Q-Learning)
            for r in range(REPEATS):
                st = time.time()
                ql = QLearningOptimizer(self.env, s, d, w_d, w_r, w_res, episodes=400)
                ql.train()
                path, cost = ql.get_best_path()
                dur = (time.time() - st) * 1000
                if path: ql_total_costs.append(cost)
                ql_total_times.append(dur)
                all_results_csv.append({"Test_ID": i+1, "Source": s, "Destination": d, "Algorithm": "Q-Learning", "Run_ID": r+1, "Time_ms": dur, "Cost": cost if path else 0, "Path_Length": len(path) if path else 0})
                
                if path and r == 0:
                    self.draw_network(path, title_suffix=f"| Test {i+1} | Q-Learning")

        total_time = time.time() - start_total
        self.log(f"Tamamlandı! {total_time:.1f} sn.")

        filename = generate_report_name()
        save_results_to_csv(all_results_csv, filename)
        
        self.show_charts(ga_total_times, ql_total_times, ga_total_costs, ql_total_costs)

    def show_charts(self, ga_times, ql_times, ga_costs, ql_costs):
        top = tk.Toplevel(self.root)
        top.title("Sonuçlar")
        top.geometry("1000x500")
        top.configure(bg=BG_COLOR)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor(BG_COLOR)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor(BG_COLOR)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')

        labels = ['GA', 'Q-Learning']
        avg_times = [np.mean(ga_times), np.mean(ql_times)]
        
        bars1 = ax1.bar(labels, avg_times, color=[ACCENT_COLOR, PATH_COLOR])
        ax1.set_title('Ortalama Süre (ms)')
        ax1.bar_label(bars1, fmt='%.1f', color='white')

        avg_c_ga = np.mean(ga_costs) if ga_costs else 0
        avg_c_ql = np.mean(ql_costs) if ql_costs else 0
        avg_costs = [avg_c_ga, avg_c_ql]
        
        bars2 = ax2.bar(labels, avg_costs, color=[ACCENT_COLOR, PATH_COLOR])
        ax2.set_title('Ortalama Maliyet (Fitness)')
        ax2.bar_label(bars2, fmt='%.2f', color='white')

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, msg, clear=False):
        if clear: self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, msg + "\n")
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkVisualizerApp(root)
    root.mainloop()