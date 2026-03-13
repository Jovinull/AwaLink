"""
play.py — Assistir o agente jogar ou criar save state para Zelda: Link's Awakening.

Uso:
    python play.py                        → Assiste o melhor modelo jogar
    python play.py --model models/x.zip   → Carrega modelo específico
    python play.py --create-save          → Joga manualmente para criar save state
"""
import argparse
import os
import glob
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from environment import ZeldaLinksAwakeningEnv
import memory_map as mem


def find_best_model() -> str | None:
    final = os.path.join(config.MODEL_DIR, "zelda_ppo_final.zip")
    if os.path.exists(final):
        return final

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


def create_save_state():
    """Jogue manualmente para criar o save state inicial."""
    from pyboy import PyBoy

    print("=" * 60)
    print("   MODO: CRIAR SAVE STATE")
    print("=" * 60)
    print("  Jogue até um ponto inicial bom.")
    print("  Recomendado: após pegar a espada na praia.")
    print("  Quando pronto, FECHE A JANELA do jogo.")
    print("  O estado será salvo automaticamente.\n")

    pyboy = PyBoy(config.ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)

    try:
        while pyboy.tick():
            pass
    except KeyboardInterrupt:
        pass

    with open(config.INIT_STATE_PATH, "wb") as f:
        pyboy.save_state(f)

    m = pyboy.memory
    health = m[mem.CURRENT_HEALTH]
    max_hp = m[mem.MAX_HEALTH]
    sword = m[mem.SWORD_LEVEL]
    room = m[mem.MAP_ROOM]
    cat = m[mem.MAP_CATEGORY]
    area = ["Overworld", "Dungeon", "Side-scroll"][cat] if cat <= 2 else f"Unknown({cat})"

    print(f"\n[OK] Save state salvo: {config.INIT_STATE_PATH}")
    print(f"  Área: {area}")
    print(f"  Sala: {room:#04x}")
    print(f"  Saúde: {health}/{max_hp * 8} ({max_hp} corações)")
    print(f"  Espada nível: {sword}")

    pyboy.stop()


def watch_agent(model_path: str):
    """Assiste o agente treinado jogar via SDL2."""
    print("=" * 60)
    print("   ZELDA: LINK'S AWAKENING — ASSISTINDO O AGENTE")
    print("=" * 60)
    print(f"  Modelo: {model_path}\n")

    env = ZeldaLinksAwakeningEnv(render_mode="human")
    env.pyboy.set_emulation_speed(1)

    # Wrap com VecNormalize se disponível (para manter consistência com treino)
    vecnorm_path = find_latest_vecnormalize()
    vec_env = DummyVecEnv([lambda: env])

    if vecnorm_path:
        print(f"  VecNormalize: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    total_reward = 0
    step_count = 0

    print("[PLAY] Agente jogando! Feche a janela ou Ctrl+C para parar.\n")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env.step(action)
            total_reward += float(reward[0])
            step_count += 1

            if step_count % 100 == 0:
                i = info[0] if isinstance(info, list) else info
                print(
                    f"  Step {step_count:>6} | "
                    f"Reward: {total_reward:>8.1f} | "
                    f"Screens: {i.get('screens_explored', 0):>3} | "
                    f"Instruments: {i.get('instruments', 0)}/8 | "
                    f"Items: {i.get('inventory_count', 0):>2} | "
                    f"HP: {i.get('hp_fraction', 0):.0%} | "
                    f"Deaths: {i.get('deaths', 0)}"
                )

            if done[0]:
                print(f"\n[EP FIM] Reward: {total_reward:.1f} | Steps: {step_count}")
                obs = vec_env.reset()
                total_reward = 0
                step_count = 0

    except KeyboardInterrupt:
        print("\n[PLAY] Interrompido.")
    except SystemExit:
        pass
    finally:
        vec_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Assistir o Agente Zelda jogar"
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--create-save", action="store_true")
    args = parser.parse_args()

    if args.create_save:
        create_save_state()
        return

    model_path = args.model or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print("[ERRO] Nenhum modelo treinado encontrado!")
        print("       Execute 'python train.py' primeiro.")
        return

    watch_agent(model_path)


if __name__ == "__main__":
    main()
