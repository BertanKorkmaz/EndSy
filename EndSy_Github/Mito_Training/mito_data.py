# run_multiple_sims.py

import os
import random
import numpy as np
import csv

from EndSy_no_mito import Environment, ProkaryoticCell

# Başlangıç hücre koordinatları
INITIAL_POSITIONS = [
    (25, 23),
    (23, 25),
    (27, 25),
    (25, 27),
    (25, 25)
]

# Simülasyon parametreleri
NUM_SIMULATIONS = 50
ITERATIONS_PER_SIM = 500
BASE_SEED = 42
OUTPUT_DIR = "mito_sim_data"

def run_simulation(sim_index: int, seed: int):
    # 1) Rastgele sayı üreticilerini ayarla
    random.seed(seed)
    np.random.seed(seed)

    # 2) Ortamı ve başlangıç hücrelerini oluştur
    env = Environment(width=50, height=50,
                      capacity=5,
                      daily_increment=0.5,
                      daily_increment_iterations=100)
    cells = []
    for x, y in INITIAL_POSITIONS:
        cell = ProkaryoticCell(x=x, y=y, birth_iteration=env.iteration)
        cells.append(cell)
        env.add_agent(cell)

    # 3) Çıktı klasörünü ve CSV dosyasını hazırla
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"sim_{sim_index+1:02d}_seed_{seed}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cell_id",
            "iteration",
            "damage_rate",
            "prolif_rate",
            "energy_req_rate",
            "is_dys"
        ])

        # 4) Ana iterasyon döngüsü
        for it in range(1, ITERATIONS_PER_SIM + 1):
            # 4.1) Ortamı güncelle
            env.iterate()

            # 4.2) Hücreleri güncelle
            for cell in cells:
                cell.update(env)

            # 4.3) Ölüm kontrolü
            alive_cells = []
            for cell in cells:
                if cell.alive:
                    alive_cells.append(cell)
                else:
                    env.remove_agent(cell)
            cells = alive_cells

            # 4.4) Bölünme kontrolü
            next_cells = []
            for cell in cells:
                if cell.alive and cell.proliferation >= cell.proliferation_threshold:
                    daughters = cell.attempt_division(env)
                    if daughters:
                        next_cells.extend(daughters)
                    else:
                        next_cells.append(cell)
                else:
                    next_cells.append(cell)
            cells = next_cells

            # 4.5) Her hücre için istenen verileri CSV'ye yaz
            for cell in cells:
                writer.writerow([
                    cell.id,
                    env.iteration,
                    cell.original_damage_rate,
                    cell.original_prolif_rate,
                    cell.original_energy_req_rate,
                    cell.is_dys
                ])

    print(f"✔ Simulation {sim_index+1} complete, seed={seed}, output → {csv_path}")

if __name__ == "__main__":
    for i in range(NUM_SIMULATIONS):
        seed = BASE_SEED + i
        run_simulation(i, seed)
