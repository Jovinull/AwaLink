"""
train.py — Treinamento Paralelo com PPO para Zelda: Link's Awakening.

Técnicas avançadas de RL:
  - PPO com GAE (Schulman et al., 2017)
  - VecNormalize para normalização de observações e rewards
  - TensorBoard com métricas detalhadas de progresso do jogo
  - Linear LR decay para convergência suave
  - Checkpointing automático com salvamento de normalizador

Uso:
    python train.py                       → Treina (continua se existir checkpoint)
    python train.py --fresh               → Força treino do zero
    python train.py --timesteps 5000000   → Define total de timesteps
    python train.py --num-envs 4          → Usa 4 processos paralelos
"""
import argparse
import os
import glob
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)

import config


def make_env(rank, seed=0):
    """Factory para SubprocVecEnv — cada processo recebe seu próprio PyBoy."""
    def _init():
        from environment import ZeldaLinksAwakeningEnv
        env = ZeldaLinksAwakeningEnv(render_mode="null")
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


class ZeldaLogCallback(BaseCallback):
    """
    Callback de logging detalhado para TensorBoard.

    Registra todas as métricas de progresso do Zelda para
    visualização em tempo real durante o treinamento.
    """

    def __init__(self, num_envs, verbose=0):
        super().__init__(verbose)
        self.num_envs = num_envs
        self.episode_count = 0
        self.start_time = time.time()
        self.best_instruments = 0
        self.best_screens = 0

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue

            self.episode_count += 1
            infos = self.locals.get("infos", [])
            if i >= len(infos):
                continue

            info = infos[i]
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / max(elapsed, 1)

            instruments = info.get("instruments", 0)
            screens = info.get("screens_explored", 0)
            self.best_instruments = max(self.best_instruments, instruments)
            self.best_screens = max(self.best_screens, screens)

            # Console output
            print(
                f"[EP {self.episode_count:>4}] "
                f"Env {i:>2}/{self.num_envs} | "
                f"Steps: {self.num_timesteps:>9,} | "
                f"Scr: {screens:>3} | "
                f"Trans: {info.get('screen_transitions', 0):>4} | "
                f"Inst: {instruments}/8 | "
                f"Items: {info.get('inventory_count', 0):>2} | "
                f"Kills: {info.get('kills', 0):>3} | "
                f"Deaths: {info.get('deaths', 0):>2} | "
                f"Rupees: {info.get('rupees', 0):>3} | "
                f"HP: {info.get('hp_fraction', 0):.0%} | "
                f"R: {info.get('total_reward', 0):>8.1f} | "
                f"FPS: {fps:>5.0f}"
            )

            # --- TensorBoard: Progresso Principal ---
            self.logger.record("zelda/instruments", instruments)
            self.logger.record("zelda/best_instruments", self.best_instruments)
            self.logger.record("zelda/screens_explored", screens)
            self.logger.record("zelda/best_screens", self.best_screens)
            self.logger.record("zelda/total_screens_visited", info.get("total_screens_visited", 0))
            self.logger.record("zelda/dungeon_rooms", info.get("dungeon_rooms", 0))

            # --- TensorBoard: Inventário e Equipamento ---
            self.logger.record("zelda/inventory_count", info.get("inventory_count", 0))
            self.logger.record("zelda/sword_level", info.get("sword_level", 0))
            self.logger.record("zelda/shield_level", info.get("shield_level", 0))
            self.logger.record("zelda/max_hearts", info.get("max_hearts", 0))
            self.logger.record("zelda/dungeon_keys", info.get("dungeon_keys", 0))

            # --- TensorBoard: Colecionáveis ---
            self.logger.record("zelda/shells", info.get("shells", 0))
            self.logger.record("zelda/leaves", info.get("leaves", 0))
            self.logger.record("zelda/trading_item", info.get("trading_item", 0))
            self.logger.record("zelda/rupees", info.get("rupees", 0))

            # --- TensorBoard: Saúde e Sobrevivência ---
            self.logger.record("zelda/hp_fraction", info.get("hp_fraction", 0))
            self.logger.record("zelda/deaths", info.get("deaths", 0))
            self.logger.record("zelda/kills", info.get("kills", 0))
            self.logger.record("zelda/screen_transitions", info.get("screen_transitions", 0))

            # --- TensorBoard: Componentes de Reward ---
            self.logger.record("zelda/total_reward", info.get("total_reward", 0))
            self.logger.record("zelda/unique_positions", info.get("unique_positions", 0))

            for key in [
                "explore_screen", "explore_room", "explore_count",
                "screen_transition", "instruments", "new_items",
                "equipment", "hearts", "dungeon_keys", "small_keys",
                "dungeon_items", "shells", "leaves", "trading",
                "kills", "rupees", "hp_recovery",
                "time", "idle", "death", "health_loss", "stuck",
            ]:
                val = info.get(f"rew_{key}", 0)
                self.logger.record(f"reward/{key}", val)

            # --- TensorBoard: Performance ---
            self.logger.record("zelda/fps", fps)
            self.logger.record("zelda/episode", self.episode_count)

        return True


def find_latest_checkpoint() -> str | None:
    if not os.path.exists(config.MODEL_DIR):
        return None
    zips = glob.glob(os.path.join(config.MODEL_DIR, "zelda_ppo_*.zip"))
    if not zips:
        return None
    zips.sort(key=os.path.getmtime)
    return zips[-1]


def find_latest_vecnormalize() -> str | None:
    if not os.path.exists(config.MODEL_DIR):
        return None
    pkls = glob.glob(os.path.join(config.MODEL_DIR, "zelda_vecnorm_*.pkl"))
    if not pkls:
        return None
    pkls.sort(key=os.path.getmtime)
    return pkls[-1]


class SaveVecNormalizeCallback(BaseCallback):
    """Salva o VecNormalize junto com os checkpoints do modelo."""

    def __init__(self, save_freq, save_path, name_prefix="zelda_vecnorm", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.pkl",
            )
            if hasattr(self.model, "get_env"):
                env = self.model.get_env()
                if isinstance(env, VecNormalize):
                    env.save(path)
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Treinar Agente Zelda: Link's Awakening com RL"
    )
    parser.add_argument("--fresh", action="store_true", help="Forçar treino do zero")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument(
        "--num-envs", type=int, default=config.NUM_ENVS,
        help=f"Processos paralelos (default: {config.NUM_ENVS})",
    )
    args = parser.parse_args()

    num_envs = args.num_envs
    n_steps = config.N_STEPS

    print("=" * 70)
    print("   ZELDA: LINK'S AWAKENING — TREINAMENTO RL (PPO)")
    print("=" * 70)
    print(f"  ROM: {config.ROM_PATH}")
    print(f"  Init state: {config.INIT_STATE_PATH}")
    print(f"  Timesteps alvo: {args.timesteps:,}")
    print(f"  Processos paralelos: {num_envs}")
    print(f"  Episódio: {config.MAX_STEPS_PER_EPISODE:,} steps")
    print(f"  PPO: n_steps={n_steps} × {num_envs} envs")
    print(f"       batch={config.BATCH_SIZE} epochs={config.N_EPOCHS}")
    print(f"       gamma={config.GAMMA} ent={config.ENT_COEF} lr={config.LEARNING_RATE}")
    print(f"  Reward: scale={config.REWARD_SCALE} time_penalty={config.TIME_PENALTY}")
    print(f"  Normalização: obs={config.NORMALIZE_OBS} reward={config.NORMALIZE_REWARD}")
    print("=" * 70)

    # 1. Cria ambientes paralelos
    print(f"\n[SETUP] Criando {num_envs} ambientes paralelos...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # VecNormalize — normaliza observações e rewards (técnica essencial de RL moderno)
    latest_vecnorm = None if args.fresh else find_latest_vecnormalize()
    if latest_vecnorm:
        print(f"[SETUP] Carregando VecNormalize: {latest_vecnorm}")
        env = VecNormalize.load(latest_vecnorm, vec_env)
        env.training = True
    else:
        # Apenas espaços Box podem ser normalizados; MultiBinary/MultiDiscrete ficam intactos
        norm_keys = [
            "screens", "health", "position", "equipment",
            "overworld_map", "dungeon_state", "held_items", "game_progress",
        ]
        env = VecNormalize(
            vec_env,
            norm_obs=config.NORMALIZE_OBS,
            norm_obs_keys=norm_keys,
            norm_reward=config.NORMALIZE_REWARD,
            clip_reward=config.REWARD_CLIP,
            gamma=config.GAMMA,
        )

    print(f"[SETUP] {num_envs} Game Boys rodando!")

    # 2. Cria ou carrega modelo
    latest_model = None if args.fresh else find_latest_checkpoint()

    if latest_model:
        print(f"[SETUP] Continuando do checkpoint: {latest_model}")
        model = PPO.load(latest_model, env=env)
        model.n_steps = n_steps
        model.n_envs = num_envs
        model.rollout_buffer.buffer_size = n_steps
        model.rollout_buffer.n_envs = num_envs
        model.rollout_buffer.reset()
    else:
        print("[SETUP] Criando modelo PPO do zero...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=n_steps,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            vf_coef=config.VF_COEF,
            max_grad_norm=config.MAX_GRAD_NORM,
            tensorboard_log=config.LOG_DIR,
            verbose=config.VERBOSE,
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[SETUP] Política: {type(model.policy).__name__}")
    print(f"[SETUP] Parâmetros: {total_params:,}")

    # 3. Callbacks
    save_freq = max(config.SAVE_FREQUENCY // num_envs, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=config.MODEL_DIR,
        name_prefix="zelda_ppo",
    )
    vecnorm_cb = SaveVecNormalizeCallback(
        save_freq=save_freq,
        save_path=config.MODEL_DIR,
    )
    log_cb = ZeldaLogCallback(num_envs=num_envs)

    # 4. TREINA!
    print(f"\n[TREINO] Iniciando... (Ctrl+C para parar e salvar)\n")
    print(f"[TREINO] TensorBoard: tensorboard --logdir {config.LOG_DIR}\n")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoint_cb, vecnorm_cb, log_cb]),
            progress_bar=True,
            tb_log_name=config.TB_LOG_NAME,
        )
    except KeyboardInterrupt:
        print("\n\n[TREINO] Interrompido pelo usuário.")

    # 5. Salva modelo final + normalizador
    final_path = os.path.join(config.MODEL_DIR, "zelda_ppo_final")
    model.save(final_path)
    env.save(os.path.join(config.MODEL_DIR, "zelda_vecnorm_final.pkl"))
    print(f"\n[TREINO] Modelo salvo: {final_path}.zip")
    print(f"[TREINO] VecNormalize salvo: zelda_vecnorm_final.pkl")
    print("[TREINO] Para continuar: python train.py")

    env.close()


if __name__ == "__main__":
    main()
