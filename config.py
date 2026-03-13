"""
config.py — Hiperparâmetros para RL em Zelda: Link's Awakening.

V3 — Episódios curtos, reward manual (sem VecNormalize reward), anti-oscillation,
auto Game Over, detecção de interação com o mundo, observação de combate/ammo.
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
ACTION_FREQ = 16
# ~30k steps ≈ 8 min de jogo por episódio. Com 20M steps / 8 envs = 2500 eps/env.
MAX_STEPS_PER_EPISODE = 2048 * 15

# ===========================
# PPO
# ===========================
NUM_ENVS = 8
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 2              # 2 para evitar overfitting no rollout com rewards ruidosos
GAMMA = 0.998
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.015
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 2.5e-4
TOTAL_TIMESTEPS = 20_000_000

# ===========================
# RECOMPENSAS DENSAS (~a cada step)
# ===========================
REWARD_SCALE = 1.0

RUPEE_REWARD_SCALE = 0.05
HEALTH_RECOVERY_REWARD = 0.5
SCREEN_TRANSITION_REWARD = 0.15
SCREEN_TRANSITION_COOLDOWN = 3  # ignora transição se room está nas últimas N visitadas
KILL_BONUS = 0.3
EXPLORE_COUNT_BONUS = 0.02
EXPLORE_COUNT_MAX_VISITS = 100  # corta bônus após 100 visitas (reduz ruído)
WORLD_INTERACTION_REWARD = 0.3  # mudança nos tile data do mapa → interagiu com o mundo

# ===========================
# RECOMPENSAS MÉDIAS (~a cada minuto)
# ===========================
EXPLORE_NEW_SCREEN = 3.0
EXPLORE_NEW_ROOM = 2.0
DUNGEON_FLAGS_REWARD = 0.5      # novo bit em dungeon flags (chest, porta, etc.)

# ===========================
# MILESTONES
# ===========================
INSTRUMENT_REWARD = 50.0
NEW_ITEM_REWARD = 10.0
EQUIPMENT_UPGRADE = 5.0
HEART_CONTAINER = 8.0
DUNGEON_KEY_REWARD = 3.0
SMALL_KEY_REWARD = 1.5
DUNGEON_ITEM_REWARD = 2.0
SHELL_REWARD = 1.0
GOLDEN_LEAF_REWARD = 2.0
TRADING_STEP_REWARD = 3.0

# ===========================
# PENALIDADES
# ===========================
TIME_PENALTY_BASE = -0.0003
IDLE_PENALTY = -0.002
DEATH_PENALTY = -3.0
HEALTH_LOSS_PENALTY = -0.5
STUCK_PENALTY = -0.15
STUCK_VISIT_THRESHOLD = 300
STUCK_SAME_SCREEN_STEPS = 500

# ===========================
# GAME OVER AUTO-NAVIGATE
# ===========================
GAME_OVER_WAIT_STEPS = 20       # espera pela animação de morte antes de apertar A

# ===========================
# OBSERVAÇÃO
# ===========================
FRAME_STACKS = 4
SCREEN_SIZE = (72, 80)
COORDS_PAD = 8
ENC_FREQS = 8

# ===========================
# NORMALIZAÇÃO
# ===========================
NORMALIZE_OBS = True
NORMALIZE_REWARD = False        # reward manual — VecNormalize destruía hierarquia via clipping

# ===========================
# LOGGING
# ===========================
SAVE_FREQUENCY = 100_000
VERBOSE = 1
TB_LOG_NAME = "zelda_ppo"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
