#!/usr/bin/env python3
import os
import csv
import random
import numpy as np
import EndSy as m

# ---- Ayarlar ----
SEED_START = 42
SEED_END = 141  # dahil
OUTPUT_FILE = "simulation_results.csv"
MAX_ITER = 1000  # iterasyon sınırı (eğer normal hücreler ölmezse)

# Başlangıç pozisyon setleri
initial_position_sets = {
    1: [(25, 25)],
    5: [(25, 23), (23, 25), (27, 25), (25, 27), (25, 25)],
    25: [(x, y) for x in [21, 23, 25, 27, 29] for y in [21, 23, 25, 27, 29]]
}

# Senaryolar
scenarios = {
    1: m.SCENARIO_1,
    2: m.SCENARIO_2
}

# CSV başlık
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "seed",
        "scenario",
        "mito_enabled",
        "initial_cell_count",
        "last_alive_iteration"
    ])

# Simülasyon döngüsü
for seed in range(SEED_START, SEED_END + 1):
    for scenario_id, schedule in scenarios.items():
        for mito_enabled in (True, False):
            # Modülün MITO_ENABLED bayrağını güncelle
            m.MITO_ENABLED = mito_enabled

            # tohumları ayarla
            random.seed(seed)
            np.random.seed(seed)

            for init_count, positions in initial_position_sets.items():
                # Yeni environment
                env = m.Environment(
                    width=50,
                    height=50,
                    capacity=5,
                    daily_increment=0.5,
                    daily_increment_iterations=100,
                    supply_schedule=schedule
                )

                # Başlangıç hücrelerini ekle
                cells = []
                for x, y in positions:
                    c = m.ProkaryoticCell(
                        x=x,
                        y=y,
                        birth_iteration=env.iteration
                    )
                    cells.append(c)
                    env.add_agent(c)

                last_alive_iter = None

                # Simülasyonu çalıştır
                while env.iteration < MAX_ITER:
                    env.iterate()

                    # Hücre güncellemeleri
                    for cell in list(cells):
                        cell.update(env)

                    # MITO_ENABLED == True ise mito-tag'li hücreleri öldür
                    if mito_enabled:
                        for cell in list(cells):
                            if cell.alive and cell.mito_tag == 1:
                                cell.alive = False
                                env.remove_agent(cell)

                    # Hasar/enerji eşiğine göre ölümleri uygula
                    surviving = []
                    for cell in cells:
                        if cell.alive and \
                           cell.damage < cell.death_damage_threshold and \
                           cell.energy_deficit < cell.death_energy_threshold:
                            surviving.append(cell)
                        else:
                            env.remove_agent(cell)
                    cells = surviving

                    # Bölünme adımı
                    next_gen = []
                    for cell in cells:
                        if cell.alive and cell.proliferation >= cell.proliferation_threshold:
                            daughters = cell.attempt_division(env)
                            if daughters:
                                next_gen.extend(daughters)
                            else:
                                next_gen.append(cell)
                        else:
                            next_gen.append(cell)
                    cells = next_gen

                    # Normal hücre sayısını kontrol et
                    normal_cells = [c for c in cells if not c.is_dys and c.alive]
                    if not normal_cells:
                        last_alive_iter = env.iteration
                        break

                # Eğer hiç ölmediyse, MAX_ITER kaydet
                if last_alive_iter is None:
                    last_alive_iter = env.iteration

                # Sonucu CSV'ye yaz
                with open(OUTPUT_FILE, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        seed,
                        scenario_id,
                        mito_enabled,
                        init_count,
                        last_alive_iter
                    ])

print(f"Tüm simülasyonlar tamamlandı. Sonuçlar: {os.path.abspath(OUTPUT_FILE)}")
