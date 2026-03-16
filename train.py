"""
train.py — Treinamento Paralelo com PPO para Zelda: Link's Awakening.

V4:
  - Custom feature extractor: mini-CNN para overworld_map (16×16×1)
  - Normalização apenas em observações Box escalares (sem screens/overworld_map)
  - Sem normalização de reward (escala manual preserva hierarquia)
  - Métricas completas no TensorBoard (world interactions, entities, dungeon flags)
  - Curriculum phase logging

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

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

import config


class ZeldaFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor que processa:
      - screens (72×80×4) via NatureCNN padrão
      - overworld_map (16×16×1) via mini-CNN dedicada
      - demais observações via MLP (flatten + linear)
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        # Calcular dimensão total das features antes de chamar super()
        # NatureCNN para screens: 512 features
        # Mini-CNN para overworld_map: 64 features
        # MLP para escalares: soma das dimensões flat
        scalar_keys = [
            k for k in observation_space.spaces
            if k not in ("screens", "overworld_map")
        ]
        scalar_dim = sum(
            get_flattened_obs_dim(observation_space.spaces[k])
            for k in scalar_keys
        )
        features_dim = 512 + 64 + scalar_dim

        super().__init__(observation_space, features_dim=features_dim)

        self.scalar_keys = scalar_keys

        # NatureCNN para screens
        # SB3 aplica VecTransposeImage que converte HWC→CHW no obs_space e nos dados
        # Então observation_space já tem screens como (C, H, W) = (4, 72, 80)
        screens_shape = observation_space.spaces["screens"].shape  # (C, H, W) after VecTransposeImage
        n_channels = screens_shape[0]
        self.screens_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *screens_shape)
            cnn_out = self.screens_cnn(sample)
            cnn_flat = cnn_out.shape[1]
        self.screens_linear = nn.Sequential(
            nn.Linear(cnn_flat, 512),
            nn.ReLU(),
        )

        # Mini-CNN para overworld_map (16×16×1)
        # overworld_map é float32 → SB3 NÃO transpõe, chega como HWC
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            map_shape = observation_space.spaces["overworld_map"].shape  # (16, 16, 1) HWC
            sample = torch.zeros(1, 1, map_shape[0], map_shape[1])  # CHW
            map_out = self.map_cnn(sample)
            map_flat = map_out.shape[1]
        self.map_linear = nn.Sequential(
            nn.Linear(map_flat, 64),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Screens: já CHW via VecTransposeImage, já /255 via preprocess_obs
        screens = observations["screens"]
        screen_features = self.screens_linear(self.screens_cnn(screens))

        # Overworld map: float32, NÃO transposto pelo SB3 → HWC→CHW
        ow_map = observations["overworld_map"]
        if ow_map.dim() == 4:
            ow_map = ow_map.permute(0, 3, 1, 2)
        map_features = self.map_linear(self.map_cnn(ow_map))

        # Escalares: flatten e concatenar
        scalar_parts = []
        for key in self.scalar_keys:
            obs = observations[key].float()
            if obs.dim() > 2:
                obs = obs.reshape(obs.shape[0], -1)
            scalar_parts.append(obs)
        scalars = torch.cat(scalar_parts, dim=1)

        return torch.cat([screen_features, map_features, scalars], dim=1)


def make_env(rank, seed=0):
    def _init():
        from environment import ZeldaLinksAwakeningEnv
        env = ZeldaLinksAwakeningEnv(render_mode="null")
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


class ZeldaLogCallback(BaseCallback):
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

            print(
                f"[EP {self.episode_count:>4}] "
                f"Env {i:>2}/{self.num_envs} | "
                f"Steps: {self.num_timesteps:>9,} | "
                f"Scr: {screens:>3} | "
                f"Trans: {info.get('screen_transitions', 0):>4} | "
                f"WI: {info.get('world_interactions', 0):>3} | "
                f"Inst: {instruments}/8 | "
                f"Items: {info.get('inventory_count', 0):>2} | "
                f"Kills: {info.get('kills', 0):>3} | "
                f"Ent: {info.get('entities', 0):>2} | "
                f"Deaths: {info.get('deaths', 0):>2} | "
                f"Rupees: {info.get('rupees', 0):>3} | "
                f"HP: {info.get('hp_fraction', 0):.0%} | "
                f"R: {info.get('total_reward', 0):>8.1f} | "
                f"FPS: {fps:>5.0f}"
            )

            # --- TensorBoard ---
            self.logger.record("zelda/instruments", instruments)
            self.logger.record("zelda/best_instruments", self.best_instruments)
            self.logger.record("zelda/screens_explored", screens)
            self.logger.record("zelda/best_screens", self.best_screens)
            self.logger.record("zelda/total_screens_visited", info.get("total_screens_visited", 0))
            self.logger.record("zelda/dungeon_rooms", info.get("dungeon_rooms", 0))
            self.logger.record("zelda/screen_transitions", info.get("screen_transitions", 0))
            self.logger.record("zelda/world_interactions", info.get("world_interactions", 0))
            self.logger.record("zelda/inventory_count", info.get("inventory_count", 0))
            self.logger.record("zelda/sword_level", info.get("sword_level", 0))
            self.logger.record("zelda/shield_level", info.get("shield_level", 0))
            self.logger.record("zelda/max_hearts", info.get("max_hearts", 0))
            self.logger.record("zelda/dungeon_keys", info.get("dungeon_keys", 0))
            self.logger.record("zelda/dungeon_flag_bits", info.get("dungeon_flag_bits", 0))
            self.logger.record("zelda/shells", info.get("shells", 0))
            self.logger.record("zelda/leaves", info.get("leaves", 0))
            self.logger.record("zelda/trading_item", info.get("trading_item", 0))
            self.logger.record("zelda/rupees", info.get("rupees", 0))
            self.logger.record("zelda/hp_fraction", info.get("hp_fraction", 0))
            self.logger.record("zelda/deaths", info.get("deaths", 0))
            self.logger.record("zelda/kills", info.get("kills", 0))
            self.logger.record("zelda/entities", info.get("entities", 0))
            self.logger.record("zelda/total_reward", info.get("total_reward", 0))
            self.logger.record("zelda/unique_positions", info.get("unique_positions", 0))
            self.logger.record("zelda/curriculum_phase", info.get("curriculum_phase", 1))

            for key in [
                "explore_screen", "explore_room", "explore_count",
                "screen_transition", "instruments", "new_items",
                "equipment", "hearts", "dungeon_keys", "small_keys",
                "dungeon_items", "dungeon_flags", "shells", "leaves",
                "trading", "kills", "rupees", "hp_recovery",
                "world_interact", "time", "idle", "death",
                "health_loss", "stuck",
            ]:
                self.logger.record(f"reward/{key}", info.get(f"rew_{key}", 0))

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
    parser = argparse.ArgumentParser(description="Treinar Zelda RL Agent")
    parser.add_argument("--fresh", action="store_true", help="Forçar treino do zero")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=config.NUM_ENVS)
    args = parser.parse_args()

    num_envs = args.num_envs

    if args.fresh:
        import shutil
        for d in (config.MODEL_DIR, config.LOG_DIR):
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
        print("[FRESH] Modelos e logs antigos removidos.\n")

    print("=" * 70)
    print("   ZELDA: LINK'S AWAKENING — RL TRAINING V4")
    print("=" * 70)
    print(f"  ROM: {config.ROM_PATH}")
    print(f"  Init state: {config.INIT_STATE_PATH}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Envs: {num_envs}")
    print(f"  Episode: {config.MAX_STEPS_PER_EPISODE:,} steps")
    print(f"  PPO: n_steps={config.N_STEPS} × {num_envs} envs")
    print(f"       batch={config.BATCH_SIZE} epochs={config.N_EPOCHS}")
    print(f"       gamma={config.GAMMA} ent={config.ENT_COEF} lr={config.LEARNING_RATE}")
    print(f"  Reward: scale={config.REWARD_SCALE} time_penalty={config.TIME_PENALTY_BASE}")
    print(f"  Normalização: obs={config.NORMALIZE_OBS} reward={config.NORMALIZE_REWARD}")
    print(f"  Anti-oscillation cooldown: {config.SCREEN_TRANSITION_COOLDOWN}")
    print(f"  Game Over auto-continue after {config.GAME_OVER_WAIT_STEPS} steps")
    print(f"  Curriculum: phase2@{config.CURRICULUM_PHASE_2_SCREENS}scr, phase3@{config.CURRICULUM_PHASE_3_ITEMS}items")
    print(f"  Feature extractor: ZeldaFeaturesExtractor (NatureCNN + MapCNN + MLP)")
    print("=" * 70)

    print(f"\n[SETUP] Criando {num_envs} ambientes paralelos...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    latest_vecnorm = None if args.fresh else find_latest_vecnormalize()
    if latest_vecnorm:
        print(f"[SETUP] Carregando VecNormalize: {latest_vecnorm}")
        env = VecNormalize.load(latest_vecnorm, vec_env)
        env.training = True
    else:
        norm_keys = [
            "health", "position", "equipment",
            "dungeon_state", "held_items", "game_progress",
            "combat_info", "ammo",
        ]
        env = VecNormalize(
            vec_env,
            norm_obs=config.NORMALIZE_OBS,
            norm_obs_keys=norm_keys,
            norm_reward=config.NORMALIZE_REWARD,
            gamma=config.GAMMA,
        )

    print(f"[SETUP] {num_envs} Game Boys rodando!")

    latest_model = None if args.fresh else find_latest_checkpoint()

    policy_kwargs = dict(
        features_extractor_class=ZeldaFeaturesExtractor,
    )

    if latest_model:
        print(f"[SETUP] Continuando: {latest_model}")
        model = PPO.load(latest_model, env=env, custom_objects={"policy_kwargs": policy_kwargs})
        model.n_steps = config.N_STEPS
        model.n_envs = num_envs
        model.rollout_buffer.buffer_size = config.N_STEPS
        model.rollout_buffer.n_envs = num_envs
        model.rollout_buffer.reset()
    else:
        print("[SETUP] Criando modelo PPO do zero...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
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
            policy_kwargs=policy_kwargs,
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n[SETUP] Política: {type(model.policy).__name__}")
    print(f"[SETUP] Parâmetros: {total_params:,}")

    save_freq = max(config.SAVE_FREQUENCY // num_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq, save_path=config.MODEL_DIR, name_prefix="zelda_ppo",
    )
    vecnorm_cb = SaveVecNormalizeCallback(save_freq=save_freq, save_path=config.MODEL_DIR)
    log_cb = ZeldaLogCallback(num_envs=num_envs)

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

    final_path = os.path.join(config.MODEL_DIR, "zelda_ppo_final")
    model.save(final_path)
    env.save(os.path.join(config.MODEL_DIR, "zelda_vecnorm_final.pkl"))
    print(f"\n[TREINO] Modelo salvo: {final_path}.zip")
    print(f"[TREINO] VecNormalize salvo: zelda_vecnorm_final.pkl")

    env.close()


if __name__ == "__main__":
    main()
