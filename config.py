"""
config.py — Hiperparâmetros para RL em Zelda: Link's Awakening.

V4 — Correções estruturais:
  - Recompensas rebalanceadas (hierarquia comprimida, max ~8x entre dense e milestone)
  - Gamma reduzido para 0.995 (horizonte efetivo ~920 steps, adequado ao episódio)
  - Time penalty 10x mais forte → pressão por eficiência
  - Exploration decay contínuo (sem cutoff abrupto)
  - N_EPOCHS=3, ENT_COEF=0.005 para convergência mais estável
  - Overworld map 2D (16×16×1) processado por mini-CNN dedicada
  - Curriculum learning em 3 fases baseado no progresso do agente
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
# ~30k steps ≈ 8 min de jogo por episódio.
MAX_STEPS_PER_EPISODE = 2048 * 15

# ===========================
# PPO
# ===========================
NUM_ENVS = 8
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 3              # 3 épocas para melhor extração de sinal
GAMMA = 0.995             # horizonte efetivo ~920 steps, melhor para episódios de 30k
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.005          # menor entropia → menos aleatoriedade após convergência inicial
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
SCREEN_TRANSITION_REWARD = 0.2
SCREEN_TRANSITION_COOLDOWN = 3  # ignora transição se room está nas últimas N visitadas
KILL_BONUS = 0.4
EXPLORE_COUNT_BONUS = 0.03      # bônus por posição: 0.03/√(visits), decai continuamente
WORLD_INTERACTION_REWARD = 0.4  # mudança nos tile data do mapa → interagiu com o mundo

# ===========================
# RECOMPENSAS MÉDIAS (~a cada minuto)
# ===========================
EXPLORE_NEW_SCREEN = 2.0
EXPLORE_NEW_ROOM = 1.5
DUNGEON_FLAGS_REWARD = 0.5      # novo bit em dungeon flags (chest, porta, etc.)

# ===========================
# MILESTONES (comprimidos: max ~8x vs dense rewards)
# ===========================
INSTRUMENT_REWARD = 8.0         # era 50 → 8 (principal milestone, mas alcançável)
NEW_ITEM_REWARD = 3.0           # era 10 → 3
EQUIPMENT_UPGRADE = 2.0         # era 5 → 2
HEART_CONTAINER = 3.0           # era 8 → 3
DUNGEON_KEY_REWARD = 1.5        # era 3 → 1.5
SMALL_KEY_REWARD = 1.0
DUNGEON_ITEM_REWARD = 1.5
SHELL_REWARD = 0.8
GOLDEN_LEAF_REWARD = 1.5
TRADING_STEP_REWARD = 2.0

# ===========================
# PENALIDADES (10x mais fortes)
# ===========================
TIME_PENALTY_BASE = -0.003      # era -0.0003 → 10x mais forte
IDLE_PENALTY = -0.01            # era -0.002 → 5x mais forte
DEATH_PENALTY = -2.0
HEALTH_LOSS_PENALTY = -0.5
STUCK_PENALTY = -0.1
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
# CURRICULUM LEARNING
# ===========================
# Fase 1 (exploração): bônus de exploração amplificado, penalidades leves
# Fase 2 (combate/itens): recompensas de itens/kills amplificadas
# Fase 3 (dungeons): recompensas de dungeon/instrumentos amplificadas
CURRICULUM_PHASE_2_SCREENS = 15   # avança para fase 2 após explorar N telas
CURRICULUM_PHASE_3_ITEMS = 3      # avança para fase 3 após coletar N itens
CURRICULUM_EXPLORE_MULTIPLIER = 1.5   # multiplica explore rewards na fase 1
CURRICULUM_COMBAT_MULTIPLIER = 1.5    # multiplica combat rewards na fase 2
CURRICULUM_DUNGEON_MULTIPLIER = 1.5   # multiplica dungeon rewards na fase 3

# ===========================
# LOGGING
# ===========================
SAVE_FREQUENCY = 100_000
VERBOSE = 1
TB_LOG_NAME = "zelda_ppo"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
