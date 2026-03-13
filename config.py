"""
config.py — Hiperparâmetros centralizados para RL em Zelda: Link's Awakening.

Otimizado para combate em tempo real e exploração de mundo aberto.
Referências:
  - OpenAI Five / DeepMind IMPALA: reward scaling, GAE tuning
  - PokemonRedExperiments V2: delta-reward, parallel envs
  - Google Research on exploration: count-based bonuses
"""
import os

# ===========================
# CAMINHOS
# ===========================
ROM_PATH = "zelda.gb"
INIT_STATE_PATH = "init.state"
MODEL_DIR = "models"
LOG_DIR = "logs"
SESSION_DIR = "session"

# ===========================
# EMULADOR
# ===========================
# Zelda é ação em tempo real — precisamos de respostas mais rápidas que em RPG.
# 16 frames ≈ 267ms de jogo a 60fps — bom balanço entre reatividade e eficiência
ACTION_FREQ = 16
MAX_STEPS_PER_EPISODE = 2048 * 50   # ~102400 steps por episódio

# ===========================
# HIPERPARÂMETROS DO PPO
# Tuning baseado em DeepMind IMPALA / OpenAI baselines para jogos Atari
# ===========================
NUM_ENVS = 8                  # Processos paralelos (8 Game Boys simultâneos)
N_STEPS = 2048                # Steps coletados por env antes de cada atualização
BATCH_SIZE = 512              # Mini-batch para PPO update
N_EPOCHS = 3                  # Mais epochs = melhor sample efficiency para Zelda
GAMMA = 0.998                 # Desconto alto — objetivos distantes importam muito em Zelda
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.015              # Entropia ligeiramente mais alta para incentivar exploração
VF_COEF = 0.5                 # Coeficiente da value function
MAX_GRAD_NORM = 0.5           # Gradient clipping
LEARNING_RATE = 2.5e-4
LR_SCHEDULE = "linear"        # Decaimento linear do LR (melhora convergência final)
TOTAL_TIMESTEPS = 20_000_000  # 20M steps — Zelda precisa de mais treino que Pokémon

# ===========================
# RECOMPENSAS
# ===========================
REWARD_SCALE = 1.0

# Exploração
EXPLORE_NEW_SCREEN = 2.0      # Bônus por visitar tela nova no overworld
EXPLORE_NEW_ROOM = 1.5        # Bônus por sala nova em dungeon
EXPLORE_COUNT_BONUS = 0.02    # Bônus intrínseco baseado em contagem (1/sqrt(n))

# Progresso principal (instrumentos)
INSTRUMENT_REWARD = 50.0      # Coletar instrumento de dungeon (objetivo principal)

# Itens e upgrades
NEW_ITEM_REWARD = 10.0        # Novo item no inventário
EQUIPMENT_UPGRADE = 5.0       # Upgrade de sword/shield/bracelet level
HEART_CONTAINER = 8.0         # Novo heart container (max health up)
DUNGEON_KEY_REWARD = 3.0      # Chave de dungeon entrance
SMALL_KEY_REWARD = 1.5        # Small key dentro de dungeon
DUNGEON_ITEM_REWARD = 2.0     # Mapa, bússola, owl beak, nightmare key

# Colecionáveis
SHELL_REWARD = 1.0            # Secret shell
GOLDEN_LEAF_REWARD = 2.0      # Golden leaf
TRADING_STEP_REWARD = 3.0     # Progresso no trading sequence

# Combate
KILL_BONUS = 0.1              # Baseado em kill counters (PoP/Guardian Acorn)

# Economia
RUPEE_REWARD_SCALE = 0.001    # Pequeno bônus por rupees (evita farming)

# Penalidades
TIME_PENALTY = -0.0005        # Penalidade por step (cria urgência) — ~-0.05 por 100 steps
DEATH_PENALTY = -5.0          # Morrer
HEALTH_LOSS_PENALTY = -0.3    # Penalidade por perder HP (por heart perdido)
STUCK_PENALTY = -0.1          # Ficar parado no mesmo tile muito tempo

# Thresholds
STUCK_VISIT_THRESHOLD = 400   # Visitas ao mesmo (x,y,room) = "preso"
STUCK_SAME_SCREEN_STEPS = 800 # Steps na mesma tela sem progresso

# ===========================
# OBSERVAÇÃO
# ===========================
FRAME_STACKS = 4              # 4 frames para captar movimento/animação de combate
SCREEN_SIZE = (72, 80)        # Tela GB 144×160 / 2
COORDS_PAD = 8                # Raio do mapa de exploração local
ENC_FREQS = 8                 # Frequências para Fourier encoding

# ===========================
# NORMALIZAÇÃO (VecNormalize do SB3)
# ===========================
NORMALIZE_OBS = True
NORMALIZE_REWARD = True
REWARD_CLIP = 10.0            # Clipa rewards normalizadas em [-10, 10]

# ===========================
# LOGGING
# ===========================
SAVE_FREQUENCY = 100_000
VERBOSE = 1
TB_LOG_NAME = "zelda_ppo"

# Cria pastas
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
