import random
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import csv
import math
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



MITO_ENABLED = True          # <<< sadece bunu True / False yapman yeter

MITO_MODEL   = joblib.load("mito.pkl") if MITO_ENABLED else None



# Senaryo-1  : 0-99 arası 0.5 → sonra kıtlık
SCENARIO_1 = [(0, 100, 0.5)]

# Senaryo-2  : 0-99 arası 0.5 → 100-299 arası yok → 300-399 arası 0.5
SCENARIO_2 = [(0, 100, 0.5),
              (300, 400, 0.5)]




def get_mito_tag(cell, model=MITO_MODEL):
    if not MITO_ENABLED:
        return 0
    p = model.predict_proba([[cell.damage_rate,
                              cell.prolif_rate,
                              cell.energy_req_rate]])[0, 1]
    return int(p >= 0.5)





SEED = 42          # İstediğiniz tek bir sayı
random.seed(SEED)  # Python random
np.random.seed(SEED)  # NumPy random



dead_cells = []
cells = []


pro_slowdown_factor = 0.99




class Environment: 
    def __init__(self, width, height, capacity=5, daily_increment=0.5, save_interval=1, diffusion_rate=0.1,
                 daily_increment_iterations=100,         #genel besin verme nereye kadar olmalı
                 external_supply=False,                   # Bu besin verme açık mı değil mi?
                 supply_schedule=None,
                 external_supply_iterations=80,         # Kaç iterasyona kadar ek besin verilecek
                 external_supply_amount=2,               # Eklenen besin miktarı
                 external_supply_mode="uniform",         # "uniform" veya "range"
                 external_supply_ranges = {"x": (0,5), "y": (5,10)}):  # Eğer localized ise, bu koordinatlar kullanılacak
        
    
        self.width = width #grid'in yatayı
        self.height = height #grid'in dikeyi
        self.capacity = capacity #grid'in besin kapasitesi
        self.daily_increment = daily_increment #gridlerin her biriminin + olarak günlük besin ekleme miktarı
        self.save_interval = save_interval #kaç iterasyonda bir kayıt edilecek (history)
        self.diffusion_rate = diffusion_rate  # Difüzyon katsayısı
        self.daily_increment_iterations = daily_increment_iterations
        self.supply_schedule = supply_schedule or []
        
        
        # External nutrient supply parametreleri
        self.external_supply = external_supply
        self.external_supply_iterations = external_supply_iterations
        self.external_supply_amount = external_supply_amount
        self.external_supply_mode = external_supply_mode
        self.external_supply_ranges = external_supply_ranges
        
        
        
        # Grid'i oluşuyor her hücre başlangıçta 0 besin değeriyle başlıyor
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        
        
        #Grid'in history'sini buralara kaydeder
        self.history = []
        self.iteration = 0 #iterasyonun sayısı (sayaç)
        
        #bunun sayesinde nerede ajan var ona bakabilcez
        self.occupancy = [[None for _ in range(width)] for _ in range(height)]
        
        
        self.save_snapshot()




    def save_snapshot(self):
        snapshot = [row[:] for row in self.grid]  # Her satırın kopyasını alıyoruz.
        self.history.append(snapshot) #history listesine appendliyor bu snepshotları
        
    
        
    def apply_external_nutrients(self):
        if self.external_supply and self.iteration < self.external_supply_iterations:
            if self.external_supply_mode == "uniform":
                # Grid'in her hücresine eşit miktarda besin ekle
                for i in range(self.height):
                    for j in range(self.width):
                        self.grid[i][j] = min(self.capacity, self.grid[i][j] + self.external_supply_amount)
            elif self.external_supply_mode == "localized":
                # Sadece belirli koordinatlara besin ekle
                for (x, y) in self.external_supply_locations:
                    if 0 <= x < self.width and 0 <= y < self.height:
                        self.grid[y][x] = min(self.capacity, self.grid[y][x] + self.external_supply_amount)
            elif self.external_supply_mode == "range":
                # "range" modu: Belirli bir aralıkta besin ekleyelim
                x_start, x_end = self.external_supply_ranges["x"]
                y_start, y_end = self.external_supply_ranges["y"]
                # x ve y aralıklarını döngüye alalım (dahil olacak şekilde)
                for i in range(y_start, y_end + 1):
                    for j in range(x_start, x_end + 1):
                        if 0 <= i < self.height and 0 <= j < self.width:
                            self.grid[i][j] = min(self.capacity, self.grid[i][j] + self.external_supply_amount)

    
    
    def _current_increment(self):
        """O iterasyonda kaç birim temel besin eklenecek?"""
        for start, end, inc in self.supply_schedule:
            if start <= self.iteration < end:
                return inc
        return 0  # takvimde yoksa kaynak yok
    
    
        
    def iterate(self):
        # 1) O iterasyonda ne kadar temel besin eklenecek?
        if self.supply_schedule:                     # takvim varsa onu kullan
            current_increment = self._current_increment()
        else:                                        # aksi hâlde eski “ilk 100 iterasyon” kuralı
            current_increment = (self.daily_increment
                                 if self.iteration < self.daily_increment_iterations
                                 else 0)
    
        # 2) Grid’e besini ekle
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = min(self.capacity,
                                      self.grid[i][j] + current_increment)
    
        # 3) Harici besin, difüzyon, sayaç vs.
        self.apply_external_nutrients()
        self.diffuse()
        self.iteration += 1
        if self.iteration % self.save_interval == 0:
            self.save_snapshot()
    
    
    
    def diffuse(self):
        # Yeni grid oluşturuluyor
        new_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        #burda hücre ve komşularıyla difüzyon işlemi
        for i in range(self.height):
            for j in range(self.width):
                sum_diff = 0
                count = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            if di == 0 and dj == 0:
                                continue
                            sum_diff += (self.grid[ni][nj] - self.grid[i][j])
                            count += 1
                # Difüzyon güncellemesi: basit formülle
                new_value = self.grid[i][j] + self.diffusion_rate * sum_diff
                # Kapasite ve negatif değer kontrolü
                new_grid[i][j] = max(0, min(self.capacity, new_value))
        self.grid = new_grid
            
                        
                        
    
        
        
        
    def is_empty(self, x, y):
        return self.occupancy[y][x] is None
        
        """
        boş mu değil mi occupancy'den (o griddeki yerleşim) kontrol ediyor
        
        if env.is_empty(2, 3):
            print("Burası boş")
            
        """
    
        
    
    def add_agent(self, agent):
        self.occupancy[agent.y][agent.x] = agent
        
        """
        Bu kod var olan bir hücreyi env'a eklemeye yarar
        
        hücre1 = ProkaryoticCell(x=2, y=3)
        env.add_agent(hücre1)
        
        sonuç olarak
        env.occupancy[3][2] = hücre1 yapar
        
        """
        
        
        
    def remove_agent(self, agent):
        if self.occupancy[agent.y][agent.x] == agent:
            self.occupancy[agent.y][agent.x] = None
    
        """
        if ile kontrol ederk belirli yerdeki agent'ı kaldırırız
            aslında if olmasa sadece adına göre kaldırır ama yeri de önemli 
            o yüzden if ile de yerine bakıyoruz. 
            
            ! Eğer hücre ölmüyorsa yeri ve adı uyuşmuyor olabilir buraya bak !
        
        """

        
                
    def display(self):
        #burada row (i de olabilirdi) ile beraber bütün gridleri dolaşıp teker teker çıktısını alabiliyoruz ne kadar besin var diye
        #bunu çağırırsan console'da yazdırır
        for row in self.grid:
            print(row)


    def display_occupancy(self):
        for row in self.occupancy:
            print("".join(["[X]" if a else "[ ]" for a in row]))
            




class ProkaryoticCell:
    
    _ID_COUNTER = 0  # Sınıf düzeyinde sayaç
    
    def __init__(self, x, y, birth_iteration=0, dysregulation_params=None, dysregulation_threshold_interval = (0.85,3),
                 base_damage_rate=3, damage_std=0.5,
                 base_prolif_rate=2, prolif_std=0.5,
                 base_energy_req=0.05, energy_req_std=1,
                 death_energy_threshold=0.15,
                 death_damage_threshold=16,
                 proliferation_threshold=8):
        
        
        self.id = ProkaryoticCell._ID_COUNTER
        ProkaryoticCell._ID_COUNTER += 1
        
        self.has_slowed_down = False
        #1 kere değerleri düşürmek için bunu koyduk (bölünmedeki hata için)
        
        self.stuck_counter = 0
        # Stuck counter: Division-ready olup da bölünemeyen iterasyon sayısı

        
        self.x = x #konum x
        self.y = y #konum y
        self.alive = True #Yaşama durumu
        
        # Başlangıç değerleri
        self.damage = 0.0   
        self.proliferation = 0.0 #bölünme isteği
        self.energy_deficit = 0.0 #enerji açlığı
        
        # Rastgele belirlenen oranlar, rate = ortalama, std = standard sapma, max ile negatif değerleri engelliyoruz:
        self.damage_rate = max(0, random.gauss(base_damage_rate, damage_std))
        self.prolif_rate = max(0, random.gauss(base_prolif_rate, prolif_std))
        self.energy_req_rate = max(0, random.gauss(base_energy_req, energy_req_std))
        
        
        
        # Proliferasyondan önce standart sapma değerlerini saklamak için (kullanılacak)
        self.damage_std = damage_std
        self.prolif_std = prolif_std
        self.energy_req_std = energy_req_std
        
        
        
        # Ölüm için eşik değerleri
        self.death_energy_threshold = death_energy_threshold
        self.death_damage_threshold = death_damage_threshold
        
        # Bölünme için eşik değeri
        self.proliferation_threshold = proliferation_threshold
        
        
        
        # Eğer daughter hücre oluşturulurken dysregulation_params verilmişse, onu kullan.
        if dysregulation_params is not None:
            self.dysregulation_params = dysregulation_params
        else:
            self.dysregulation_params = self.generate_dysregulation_params()
            
            
        
        
        
        
        #kanser mi değil mi
        self.is_dys = False
        
        
        self.original_damage_rate = self.damage_rate
        self.original_prolif_rate = self.prolif_rate
        self.original_energy_req_rate = self.energy_req_rate
                
        # Simülasyonun her iterasyonundaki durumu saklamak için history listesi
        self.iteration = birth_iteration
        self.history = []
        self.save_snapshot()
        
        
        
        
        
        # --- Bozulma indeksi hesaplaması için ---

        self.dysregulation_params = self.generate_dysregulation_params()  # a,b,c'yi belirlemek için

        self.dysregulation_index = self.compute_dysregulation_index()  # score'u elde eder 

        self.is_dys = (dysregulation_threshold_interval[0] <= self.dysregulation_index <= dysregulation_threshold_interval[1])  # Belirlediğimiz aralıkta ise hücre kanser olarak etiketlensin:
        

        #write_cell_data(self) #cvc'ye yazdırıyoruz
        
        self.mito_tag = get_mito_tag(self)      # 0-ya da-1
        
    def generate_dysregulation_params(self):
        a = min(self.energy_deficit / 0.15, 1.0)
        b = min(self.prolif_rate / 8.0, 1.0)
        c = min(self.damage_rate / 16.0, 1.0)
        return (a,b,c)
    
    def compute_dysregulation_index(self):
        a, b, c = self.dysregulation_params
        score = 0.0
        score += min(self.energy_deficit / 0.15, 1.0)
        score += min(self.prolif_rate / 8.0, 1.0)
        score += min(self.damage_rate / 16.0, 1.0)
        score /= 3.0

        return score

    
        
    def save_snapshot(self):
        snapshot = {
            "iteration": self.iteration,
            "x": self.x,
            "y": self.y,
            "alive": self.alive,
            "damage": self.damage,
            "proliferation": self.proliferation,
            "energy_deficit": self.energy_deficit,
            "damage_rate": self.damage_rate,
            "prolif_rate": self.prolif_rate,
            "energy_req_rate": self.energy_req_rate,
        }
        self.history.append(snapshot)
        
        
    def update(self, environment):
        
        #yaşamıyorsa buraya girme diyor:
        if not self.alive:
            return
        
        
        # Hücrenin bulunduğu koordinattaki var olan besin değeri
        available_energy = environment.grid[self.y][self.x]
        
        # Hücrenin, o birimden alabileceği ve aldığı max enerji (istek 4 ama o koordinatta 3 var o zaman 3 alabilir.)
        energy_to_consume = min(available_energy, self.energy_req_rate)
        
        # Environment'dan tüketilen enerjiyi düşer
        environment.grid[self.y][self.x] -= energy_to_consume
        
        
        
        # Enerji eksikliği hesaplaması
        if energy_to_consume < self.energy_req_rate:    # yeterli enerji alınmadı, energy deficit'e ekleyeceğiz
            self.energy_deficit += (self.energy_req_rate - energy_to_consume)
        else:                                           #yeterli enerji alındı ve energy deficit = 0 oldu
            self.energy_deficit = 0
        
        
        
        
        # Rate'lere göre damage ve proliferasyon ihtimali artar
        self.damage += self.damage_rate
        self.proliferation += self.prolif_rate
        
        # Ölüm kontrolü:
        self.check_death()
        

    
        
        # iterasyon günceller ve kaydeder.
        self.iteration += 1
        self.save_snapshot()



    def apply_slowdown(self, factor):
        
        if not self.has_slowed_down:                 
            self.damage_rate = max(self.damage_rate * factor, 1.5)  # Minimum limit
            self.prolif_rate = max(self.prolif_rate * factor, 1.5)
            self.energy_req_rate = max(self.energy_req_rate * factor, 1.5)
            self.has_slowed_down = True
            #!!! print(f"[DEBUG] Cell id={self.id} slowdown applied with factor {factor}")
 





    def attempt_division(self, environment):
               
        #cancer_probability = 0.000135

        
        #!!!print(f"[DEBUG] iteration={environment.iteration}, cell_id={self.id} -> attempt_division çağrıldı")

        
        if self.proliferation < self.proliferation_threshold or not self.alive: # Yeterli proliferation olup olmadığını kontrol et
            return []  # Bölünme gerçekleşmez
        
        
        
        possible_positions = []        
 
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx = self.x + dx
                ny = self.y + dy
                if 0 <= nx < environment.width and 0 <= ny < environment.height:
                    # Ebeveynin konumu da dahil: çünkü division sonrası ebeveyn silinecek
                    if (nx == self.x and ny == self.y) or environment.is_empty(nx, ny):
                        possible_positions.append((nx, ny))
    
        #!!!print(f"[DEBUG] iteration={environment.iteration}, cell_id={self.id}, possible_positions={possible_positions}")




        # Bölünme için 2 alan en az yoksa bölünemez
        if len(possible_positions) < 2:
            #print(f"yerim yok {self.stuck_counter} Cell {self.id}")
            # Eğer boş alan yoksa ve slowdown henüz uygulanmadıysa, uygulayalım.
            self.stuck_counter += 1
            if not self.has_slowed_down:
                self.apply_slowdown(pro_slowdown_factor)
            return []
            
            
        if len(possible_positions) >= 2:
            #print(f"şu anda bu{self.stuck_counter} Cell {self.id}")
            self.stuck_counter = 0
            #print(f"sonra da bu{self.stuck_counter} Cell {self.id}")
            # Eğer boş alan yeterliyse, ve hücre daha önce slowdown uygulandıysa, resetleyelim:
            if self.has_slowed_down:
                #!!!print(f"[DEBUG] Cell {self.id} had slowdown but now enough space available. Resetting slowdown.")
                self.has_slowed_down = False
            new_positions = random.sample(possible_positions, 2)
        #print(f"[DEBUG] iteration={environment.iteration}, cell_id={self.id}, new_positions={new_positions}")


        target_interval = (0.85,3)
                    

        if self.is_dys:
            # Eğer hücre zaten kanserse, daughter'lar otomatik olarak kanser hücresi olarak üretilecek
            cell1 = DysCell(
                x=new_positions[0][0],
                y=new_positions[0][1],
                birth_iteration=environment.iteration,
                dysregulation_params=self.dysregulation_params,  # ebeveyndeki parametreleri kullanıyoruz
                dysregulation_threshold_interval = target_interval,
                base_damage_rate=0.1, damage_std=0.05,
                base_prolif_rate=6, prolif_std=0.5,
                base_energy_req=0.5, energy_req_std=1,
                death_energy_threshold=self.death_energy_threshold,
                death_damage_threshold=1e9,
                proliferation_threshold=8
            )
            cell2 = DysCell(
                x=new_positions[1][0],
                y=new_positions[1][1],
                birth_iteration=environment.iteration,
                dysregulation_params=self.dysregulation_params,  # ebeveyndeki parametreleri kullanıyoruz
                dysregulation_threshold_interval = target_interval,
                base_damage_rate=0.1, damage_std=0.05,
                base_prolif_rate=6, prolif_std=0.5,
                base_energy_req=0.5, energy_req_std=1,
                death_energy_threshold=self.death_energy_threshold,
                death_damage_threshold=1e9,
                proliferation_threshold=8
            )
        else:
            
            params1 = (
                min(self.energy_deficit / 0.15, 1.0),
                min(self.prolif_rate / 8.0, 1.0),
                min(self.damage_rate / 16.0, 1.0)
            )
            total1 = params1[0] + params1[1] + params1[2]

            
            if target_interval[0] <= total1 <= target_interval[1]:
                cell1 = DysCell(
                    x=new_positions[0][0],
                    y=new_positions[0][1],
                    birth_iteration=environment.iteration,
                    dysregulation_params=params1,
                    dysregulation_threshold_interval = target_interval,
                    base_damage_rate=0.1, damage_std=0.05,
                    base_prolif_rate=6, prolif_std=0.5,
                    base_energy_req=0.5, energy_req_std=1,
                    death_energy_threshold=self.death_energy_threshold,
                    death_damage_threshold=1e9,
                    proliferation_threshold=8         
                )
            else:
                
                
                cell1 = ProkaryoticCell(
                    x=new_positions[0][0],
                    y=new_positions[0][1],
                    birth_iteration=environment.iteration,
                    dysregulation_params=params1,
                    dysregulation_threshold_interval = target_interval,
                    base_damage_rate=3.0,  damage_std=0.5,
                    base_prolif_rate=2.0 ,   prolif_std=0.5,
                    base_energy_req=0.05,    energy_req_std=1,
                    death_energy_threshold=self.death_energy_threshold,
                    death_damage_threshold=self.death_damage_threshold,
                    proliferation_threshold=self.proliferation_threshold
                )
            
            params2 = (
                min(self.energy_deficit / 0.15, 1.0),
                min(self.prolif_rate / 8.0, 1.0),
                min(self.damage_rate / 16.0, 1.0)
            )
            total2 = params2[0] + params2[1] + params2[2]



            if target_interval[0] <= total2 <= target_interval[1]:
                cell2 = DysCell(
                    x=new_positions[1][0],
                    y=new_positions[1][1],
                    birth_iteration=environment.iteration,
                    dysregulation_params = params2,
                    dysregulation_threshold_interval = target_interval,
                    base_damage_rate=0.1, damage_std=0.05,
                    base_prolif_rate=6, prolif_std=0.5,
                    base_energy_req=0.5, energy_req_std=1,
                    death_energy_threshold=self.death_energy_threshold,
                    death_damage_threshold=1e9,
                    proliferation_threshold=8 
                )
            else:
                cell2 = ProkaryoticCell(
                    x=new_positions[1][0],
                    y=new_positions[1][1],
                    birth_iteration=environment.iteration,
                    dysregulation_params = params2,
                    dysregulation_threshold_interval = target_interval,
                    base_damage_rate=3.0,  damage_std=0.5,
                    base_prolif_rate=2.0 ,   prolif_std=0.5,
                    base_energy_req=0.05,    energy_req_std=1,
                    death_energy_threshold=self.death_energy_threshold,
                    death_damage_threshold=self.death_damage_threshold,
                    proliferation_threshold=self.proliferation_threshold
                )
            
              
            


        


        # Ebeveyn hücreyi kaldır
        self.alive = False
        environment.remove_agent(self)
        
        # Yeni hücreleri occupancy grid'e ekle
        environment.add_agent(cell1)
        environment.add_agent(cell2)
        
        

                
        return [cell1, cell2]
    
    
    
    
    


    def check_death(self):
        if self.energy_deficit >= self.death_energy_threshold or self.damage >= self.death_damage_threshold:
            self.alive = False
            #print(f"öldüm cell_id={self.id}")
            



    def __str__(self):
        
        #istersen hücrenin genel durumunu burdan bakabilirsin
        
        return (f"Prokaryotik Hücre ({self.x}, {self.y}) - Alive: {self.alive}, "
                f"Damage: {self.damage:.2f}, Proliferation: {self.proliferation:.2f}, "
                f"Energy Req: {self.energy_req_rate:.2f}, Energy Deficit: {self.energy_deficit:.2f}")


class DysCell(ProkaryoticCell):
    def __init__(self, x, y, birth_iteration=0, dysregulation_params=None, dysregulation_threshold_interval = (0.85,3),
                 base_damage_rate=0.1, damage_std=0.05,
                 base_prolif_rate=6, prolif_std=0.5,
                 base_energy_req=0.5, energy_req_std=1,
                 death_energy_threshold=0.15,
                 death_damage_threshold=1e9,  # çok yüksek, hasardan ölmez
                 proliferation_threshold=8):
        # DysCell için farklı parametreler kullanarak parent __init__ çağırılır.
        super().__init__(x, y, birth_iteration, dysregulation_params, dysregulation_threshold_interval,
                         base_damage_rate, damage_std,
                         base_prolif_rate, prolif_std,
                         base_energy_req, energy_req_std,
                         death_energy_threshold,
                         death_damage_threshold,
                         proliferation_threshold)
        #print(f"[DEBUG] Cell id={self.id} kanserim")
        
        
        #değiştir cancer state'i
        self.is_dys = True
        

    def check_death(self):
        # Kanser hücreleri hasardan etkilenmez, yalnızca enerji eksikliğine bağlı ölür.
        if self.energy_deficit >= self.death_energy_threshold:
            self.alive = False
        #print(f"[DEBUG] DysCell id={self.id} died (energy) at iteration={self.iteration}")
        


    remaining_cells = []




def visualize_environment(environment, iteration, output_dir="output"):
    #return ile kapatabilirsin
    # Eğer klasör yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    dys_found = False

    
    
    
    
    for y in range(environment.height):
        for x in range(environment.width):
            agent = environment.occupancy[y][x]
            if agent is not None and agent.alive and hasattr(agent, 'is_dys') and agent.is_dys:
                dys_found = True
                break
        if dys_found:
            break
        
                
    # ListedColormap: 0->beyaz, 1->yeşil, 2->kırmızı
    from matplotlib.colors import ListedColormap
    
    
    if dys_found:
        cmap = ListedColormap(["white", "green", "red"])
    else:
        cmap = ListedColormap(["white", "green", "green"])
    
    grid_vis = np.zeros((environment.height, environment.width))
    
    for y in range(environment.height):
        for x in range(environment.width):
            agent = environment.occupancy[y][x]
            if agent is not None and agent.alive:
                # Eğer kanser hücresi varsa, is_dys bayrağına göre renk belirleyelim
                if hasattr(agent, 'is_dys') and agent.is_dys:
                    grid_vis[y, x] = 2
                else:
                    grid_vis[y, x] = 1

    prokaryotic_count = int((grid_vis == 1).sum())
    dys_count    = int((grid_vis == 2).sum())
    
    
    plt.figure(figsize=(5, 5))
    plt.imshow(grid_vis, cmap=cmap, origin='upper')
    plt.title(f"Iteration {iteration}   Prokaryotic: {prokaryotic_count}   Dysregulated: {dys_count}")
    plt.xticks(np.arange(-0.5, environment.width, 1), [])
    plt.yticks(np.arange(-0.5, environment.height, 1), [])
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"iteration_{iteration:04d}.png")
    plt.savefig(filename)
    plt.close()



def create_gif(image_folder="output", gif_name="simulation.gif", duration=0.5):

    # PNG dosyalarını listele ve sıralayın (isimde iterasyon numarası varsa doğru sıralama olur)
    filenames = sorted([os.path.join(image_folder, fn) 
                        for fn in os.listdir(image_folder) if fn.endswith('.png')])
    if not filenames:
        #print("PNG dosyası bulunamadı!")
        return

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    
    gif_path = os.path.join(image_folder, gif_name)
    imageio.mimsave(gif_path, images, duration=duration)



def visualize_nutrients(environment, iteration, output_dir="n_output"):
    #return ile durudr
    # Çıkış klasörü yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Grid besin değerleri numpy array olarak
    grid_array = np.array(environment.grid)
    
    plt.figure(figsize=(5, 5))
    # Renk skalasını 'viridis' gibi bir colormap ile kullanabilirsiniz.
    plt.imshow(grid_array, cmap='viridis', origin='upper', vmin=0, vmax=environment.capacity)
    plt.colorbar(label='Nutrient Level')
    plt.title(f"Nutrient Distribution at Iteration {iteration}")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"nutrient_iter_{iteration:04d}.png")
    plt.savefig(filename)
    plt.close()









#environment oluşturuyoruz
env = Environment(width=50, height=50, capacity=5, daily_increment=0.5, daily_increment_iterations=100, supply_schedule=SCENARIO_2)


# Başlangıç hücresi oluşturuluyor ve hem cells listesine hem de environment'e ekleniyor.
initial_positions = [


(25, 23),
(23, 25),
(27, 25),
(25, 27),
(25, 25) 

]

    
"""

(21, 21), (23, 21), (25, 21), (27, 21), (29, 21),
(21, 23), (23, 23), (25, 23), (27, 23), (29, 23),
(21, 25), (23, 25), (25, 25), (27, 25), (29, 25),
(21, 27), (23, 27), (25, 27), (27, 27), (29, 27),
(21, 29), (23, 29), (25, 29), (27, 29), (29, 29),
  


(25, 25)

    
   

"""
    




for x, y in initial_positions:
    c = ProkaryoticCell(x=x, y=y, birth_iteration=env.iteration)
    cells.append(c)
    env.add_agent(c)







# 40 iterasyonluk simülasyon

for day in range(1, 500):
    # 1. Environment güncellemesi
    env.iterate()
    
    
    # 2. Tüm hücreleri güncelle
    for cell in cells:
        cell.update(env)
    
    for cell in cells:
        if MITO_ENABLED and cell.alive and cell.mito_tag == 1:
            cell.alive = False
            env.remove_agent(cell)
    
    
    # 3. Ölüm kontrollerini (damage ve energy) tekrar yapalım
    surviving_cells = []
    for cell in cells:
        if cell.damage >= cell.death_damage_threshold:
            cell.alive = False
            env.remove_agent(cell)
            # (Opsiyonel: Ölen hücreleri global dead_cells listesine ekleyebilirsiniz)
        elif cell.energy_deficit >= cell.death_energy_threshold:
            cell.alive = False
            env.remove_agent(cell)
        else:
            surviving_cells.append(cell)
    cells = surviving_cells  # Sadece hayatta olanlar devam eder.
    
    # 4. Proliferasyon kontrolleri: Her hücrenin proliferation değeri kontrol edilir.
    # Eğer hücre proliferation threshold'unu aştıysa, etrafında en az 2 boş alan varsa division denemesi yapın.
    next_cells = []
    for cell in cells:
        if cell.alive and cell.proliferation >= cell.proliferation_threshold:
            # Hücrenin etrafındaki 3x3 bölgeyi kontrol edin:
            possible_positions = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx = cell.x + dx
                    ny = cell.y + dy
                    if 0 <= nx < env.width and 0 <= ny < env.height:
                        # Ebeveyn konumu da dahil (çünkü division sonrası ebeveyn silinecek)
                        if (nx == cell.x and ny == cell.y) or env.is_empty(nx, ny):
                            possible_positions.append((nx, ny))
            if len(possible_positions) >= 2:
                daughters = cell.attempt_division(env)
                if daughters:
                    next_cells.extend(daughters)
                else:
                    # Eğer division denemesi başarısızsa (örneğin boş alan hala yoksa)
                    next_cells.append(cell)
            else:
                next_cells.append(cell)
        else:
            if cell.alive:
                next_cells.append(cell)
    cells = next_cells
    

    
    #BURDA OUTPUT OLUŞTURULUYOR
    #visualize_environment(env, env.iteration) 
    #visualize_nutrients(env, env.iteration)





    
#create_gif(image_folder="n_output", gif_name="simulation_nutrient.gif", duration=0.1)
#create_gif(image_folder="output", gif_name="simulation_cells.gif", duration=0.1)
