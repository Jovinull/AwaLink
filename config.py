"""
config.py — Hiperparâmetros centralizados para RL em Zelda: Link's Awakening.

V2 — Reward denso: prioriza sinais frequentes (rupees, HP recovery, transições
de tela, movimento) para que o agente receba feedback a cada poucos steps.
Rewards raras (instrumentos, itens) continuam altas mas servem como "bônus",
não como fonte primária de aprendizado.

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
ACTION_FREQ = 16
MAX_STEPS_PER_EPISODE = 2048 * 50   # ~102400 steps por episódio

# ===========================
# HIPERPARÂMETROS DO PPO
# ===========================
NUM_ENVS = 8
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 3
GAMMA = 0.998
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.015
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 2.5e-4
LR_SCHEDULE = "linear"
TOTAL_TIMESTEPS = 20_000_000

# ===========================
# RECOMPENSAS — SINAIS DENSOS (acontecem frequentemente)
# Estes são o "pão e manteiga" do aprendizado.
# O agente precisa ver reward positiva a cada ~10-50 steps.
# ===========================
REWARD_SCALE = 1.0

# Rupees — sinal MAIS frequente (inimigos dropam, chests dão, grama dá)
RUPEE_REWARD_SCALE = 0.05        # 1 rupee = +0.05 (antes: 0.001 — invisível)

# HP Recovery — pegar corações no chão = combate bem-sucedido
HEALTH_RECOVERY_REWARD = 0.5     # por fração de HP recuperada

# Transição de tela — qualquer mudança de room, não só rooms novas
SCREEN_TRANSITION_REWARD = 0.15  # incentiva MOVIMENTO entre telas

# Combate — kill counters (PoP/Guardian) + proxy via rupee bursts
KILL_BONUS = 0.3                 # por incremento nos kill counters

# Bônus de exploração baseado em contagem (1/sqrt(n))
EXPLORE_COUNT_BONUS = 0.02

# ===========================
# RECOMPENSAS — SINAIS MÉDIOS (a cada poucos minutos)
# ===========================

# Exploração de telas NOVAS (nunca visitadas)
EXPLORE_NEW_SCREEN = 3.0
EXPLORE_NEW_ROOM = 2.0

# ===========================
# RECOMPENSAS — MILESTONES (raras, alto valor)
# ===========================

INSTRUMENT_REWARD = 50.0
NEW_ITEM_REWARD = 10.0
EQUIPMENT_UPGRADE = 5.0
HEART_CONTAINER = 8.0
DUNGEON_KEY_REWARD = 3.0
SMALL_KEY_REWARD = 1.5
DUNGEON_ITEM_REWARD = 2.0

# Colecionáveis
SHELL_REWARD = 1.0
GOLDEN_LEAF_REWARD = 2.0
TRADING_STEP_REWARD = 3.0

# ===========================
# PENALIDADES
# ===========================

# Time penalty — base por step
TIME_PENALTY_BASE = -0.0003     # penalidade base (suave)

# Inatividade — penalidade EXTRA quando NÃO se move entre steps
IDLE_PENALTY = -0.002           # 4x o time penalty base quando parado

# Morte — detectada via HP chegando a 0
DEATH_PENALTY = -3.0

# Health loss — tomar dano
HEALTH_LOSS_PENALTY = -0.5      # por fração de HP perdida

# Stuck — mesma posição por muito tempo
STUCK_PENALTY = -0.15

# Thresholds
STUCK_VISIT_THRESHOLD = 300     # visitas ao mesmo (x,y,room) = "preso"
STUCK_SAME_SCREEN_STEPS = 500   # steps na mesma tela sem progresso

# ===========================
# OBSERVAÇÃO
# ===========================
FRAME_STACKS = 4
SCREEN_SIZE = (72, 80)
COORDS_PAD = 8
ENC_FREQS = 8

# ===========================
# NORMALIZAÇÃO (VecNormalize do SB3)
# ===========================
NORMALIZE_OBS = True
NORMALIZE_REWARD = True
REWARD_CLIP = 10.0

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
