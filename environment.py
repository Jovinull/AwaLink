"""
environment.py — Gymnasium Environment para Zelda: Link's Awakening via PyBoy.

V2 — Reward Denso:
  - Morte detectada via HP=0 (não pelo contador de save file)
  - Rupees como sinal denso de combate/exploração (50x mais reward)
  - HP recovery como sinal de combate bem-sucedido
  - Transição de tela recompensada (não só telas "novas" nunca visitadas)
  - Time penalty proporcional à inatividade (parado = penalidade alta)
  - Kill counters + rupee burst como proxy de kills
  - Bônus intrínseco de exploração 1/sqrt(n) (Bellemare et al.)

Referências:
  - Bellemare et al. (2016): Count-based exploration with neural density models
  - Schulman et al. (2017): PPO — Proximal Policy Optimization
  - Burda et al. (2019): Exploration by Random Network Distillation
"""
import math
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import gymnasium as gym
from gymnasium import spaces

import memory_map as mem
import config
from global_map import (
    OVERWORLD_SHAPE,
    DUNGEON_SHAPE,
    GLOBAL_MAP_SHAPE,
    get_overworld_pos,
    get_dungeon_pos,
    build_overworld_exploration_map,
    count_explored_screens,
)

VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

ACTION_NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]


class ZeldaLinksAwakeningEnv(gym.Env):
    """
    Gymnasium Environment para treinar agente via RL em Zelda: Link's Awakening.

    V2: Rewards densos para feedback rápido ao agente.
    """

    metadata = {"render_modes": ["human", "null"]}

    def __init__(self, render_mode="null"):
        super().__init__()

        self.render_mode = render_mode
        self.headless = render_mode != "human"

        self.frame_stacks = config.FRAME_STACKS
        self.output_shape = (config.SCREEN_SIZE[0], config.SCREEN_SIZE[1], self.frame_stacks)
        self.enc_freqs = config.ENC_FREQS

        self.action_space = spaces.Discrete(len(VALID_ACTIONS))

        self.observation_space = spaces.Dict({
            "screens": spaces.Box(
                low=0, high=255, shape=self.output_shape, dtype=np.uint8
            ),
            "health": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "position": spaces.Box(
                low=-1, high=1,
                shape=(self.enc_freqs * 3,),
                dtype=np.float32,
            ),
            "inventory": spaces.MultiBinary(mem.NUM_ITEMS),
            "equipment": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "instruments": spaces.MultiBinary(mem.NUM_INSTRUMENTS),
            "overworld_map": spaces.Box(
                low=0, high=255,
                shape=(OVERWORLD_SHAPE[0], OVERWORLD_SHAPE[1], 1),
                dtype=np.uint8,
            ),
            "dungeon_state": spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            "held_items": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "game_progress": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "recent_actions": spaces.MultiDiscrete(
                [len(VALID_ACTIONS)] * self.frame_stacks
            ),
        })

        window = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            config.ROM_PATH,
            window=window,
            sound_emulated=not self.headless,
        )
        if self.headless:
            self.pyboy.set_emulation_speed(0)
        else:
            self.pyboy.set_emulation_speed(6)

        self.step_count = 0
        self.reset_count = 0

    # ===========================
    # RESET
    # ===========================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        with open(config.INIT_STATE_PATH, "rb") as f:
            self.pyboy.load_state(f)

        # Frame buffer
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        # Exploração
        self.visited_positions = {}
        self.visited_screens = set()
        self.visited_dungeon_rooms = set()
        self.prev_explored_count = count_explored_screens(self._read_m)

        # Posição atual (para detectar movimento)
        self.prev_x = self._read_m(mem.LINK_X)
        self.prev_y = self._read_m(mem.LINK_Y)
        self.prev_room = self._read_m(mem.MAP_ROOM)

        # Inventário / progresso snapshot
        self.prev_inventory = self._get_inventory_set()
        self.prev_instruments = self._count_instruments()
        self.prev_sword_level = self._read_m(mem.SWORD_LEVEL)
        self.prev_shield_level = self._read_m(mem.SHIELD_LEVEL)
        self.prev_bracelet_level = self._read_m(mem.BRACELET_LEVEL)
        self.prev_max_health = self._read_m(mem.MAX_HEALTH)
        self.prev_shells = self._read_m(mem.SECRET_SHELLS)
        self.prev_leaves = self._read_m(mem.GOLDEN_LEAVES)
        self.prev_trading = self._read_m(mem.TRADING_ITEM)
        self.prev_dungeon_keys_total = self._count_dungeon_entrance_keys()
        self.prev_small_keys = self._read_m(mem.DUNGEON_SMALL_KEYS)
        self.prev_dungeon_items = self._count_current_dungeon_items()
        self.prev_kill_counter = self._get_kill_counter()
        self.prev_rupees = self._read_rupees()

        # Saúde — usado para detecção de morte e HP recovery
        self.prev_health_raw = self._read_m(mem.CURRENT_HEALTH)
        self.prev_health_frac = self._get_health_fraction()
        self.is_dead = False

        # Contadores
        self.died_count = 0
        self.total_kills_detected = 0
        self.step_count = 0
        self.steps_on_current_screen = 0
        self.last_room = self._read_m(mem.MAP_ROOM)
        self.screen_transitions = 0

        # Reward acumulativa
        self.total_reward = 0.0
        self.cumulative_rewards = {
            "explore_screen": 0.0,
            "explore_room": 0.0,
            "explore_count": 0.0,
            "screen_transition": 0.0,
            "instruments": 0.0,
            "new_items": 0.0,
            "equipment": 0.0,
            "hearts": 0.0,
            "dungeon_keys": 0.0,
            "small_keys": 0.0,
            "dungeon_items": 0.0,
            "shells": 0.0,
            "leaves": 0.0,
            "trading": 0.0,
            "kills": 0.0,
            "rupees": 0.0,
            "hp_recovery": 0.0,
            "time": 0.0,
            "idle": 0.0,
            "death": 0.0,
            "health_loss": 0.0,
            "stuck": 0.0,
        }

        self.reset_count += 1
        return self._get_obs(), {}

    # ===========================
    # STEP
    # ===========================

    def step(self, action):
        self._run_action(action)
        self._update_recent_actions(action)
        self._update_position_tracking()

        step_reward = self._compute_reward()
        self.total_reward += step_reward

        # Atualiza estado anterior para próximo step
        self.prev_health_raw = self._read_m(mem.CURRENT_HEALTH)
        self.prev_health_frac = self._get_health_fraction()
        self.prev_x = self._read_m(mem.LINK_X)
        self.prev_y = self._read_m(mem.LINK_Y)
        self.prev_room = self._read_m(mem.MAP_ROOM)

        self.step_count += 1
        truncated = self.step_count >= config.MAX_STEPS_PER_EPISODE
        terminated = False

        info = self._build_info()
        obs = self._get_obs()

        return obs, step_reward, terminated, truncated, info

    # ===========================
    # OBSERVAÇÃO
    # ===========================

    def _get_obs(self):
        screen = self._render_screen()
        self._update_recent_screens(screen)

        return {
            "screens": self.recent_screens,
            "health": self._get_health_obs(),
            "position": self._get_position_encoding(),
            "inventory": self._get_inventory_binary(),
            "equipment": self._get_equipment_obs(),
            "instruments": self._get_instruments_binary(),
            "overworld_map": self._get_overworld_map_obs(),
            "dungeon_state": self._get_dungeon_state_obs(),
            "held_items": self._get_held_items_obs(),
            "game_progress": self._get_game_progress_obs(),
            "recent_actions": self.recent_actions,
        }

    def _render_screen(self):
        game_pixels = self.pyboy.screen.ndarray[:, :, 0:1]
        return game_pixels[::2, ::2, :].astype(np.uint8)

    def _update_recent_screens(self, screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = screen[:, :, 0]

    def _update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def _get_health_obs(self):
        hp_frac = self._get_health_fraction()
        max_hearts = self._read_m(mem.MAX_HEALTH)
        return np.array([hp_frac, max_hearts / 14.0], dtype=np.float32)

    def _get_position_encoding(self):
        x = self._read_m(mem.LINK_X) / 160.0
        y = self._read_m(mem.LINK_Y) / 144.0
        room = self._read_m(mem.MAP_ROOM) / 255.0
        return np.concatenate([
            self._fourier_encode(x),
            self._fourier_encode(y),
            self._fourier_encode(room),
        ])

    def _get_inventory_binary(self):
        inv = self._get_inventory_set()
        return np.array(
            [1 if item_id in inv else 0 for item_id in mem.ALL_ITEMS],
            dtype=np.int8,
        )

    def _get_equipment_obs(self):
        sword = self._read_m(mem.SWORD_LEVEL) / 2.0
        shield = self._read_m(mem.SHIELD_LEVEL) / 3.0
        bracelet = self._read_m(mem.BRACELET_LEVEL) / 2.0
        flippers = float(self._read_m(mem.FLIPPERS) > 0)
        potion = float(self._read_m(mem.POTION) > 0)
        has_boots = 1.0 if mem.ITEM_BOOTS in self._get_inventory_set() else 0.0
        return np.array(
            [sword, shield, bracelet, flippers, potion, has_boots],
            dtype=np.float32,
        )

    def _get_instruments_binary(self):
        return np.array([
            1 if self._read_m(mem.INSTRUMENTS_START + i) == mem.INSTRUMENT_COLLECTED else 0
            for i in range(mem.NUM_INSTRUMENTS)
        ], dtype=np.int8)

    def _get_overworld_map_obs(self):
        explore = build_overworld_exploration_map(self._read_m)
        return explore[:, :, np.newaxis]

    def _get_dungeon_state_obs(self):
        cat = self._read_m(mem.MAP_CATEGORY)
        in_dungeon = float(cat == 0x01)
        dungeon_id = self._read_m(mem.MAP_DUNGEON_ID) / 9.0
        has_map = float(self._read_m(mem.DUNGEON_MAP_FLAG) > 0)
        has_compass = float(self._read_m(mem.DUNGEON_COMPASS) > 0)
        has_owl = float(self._read_m(mem.DUNGEON_OWL_BEAK) > 0)
        has_nk = float(self._read_m(mem.DUNGEON_NIGHTMARE_KEY) > 0)
        small_keys = min(self._read_m(mem.DUNGEON_SMALL_KEYS), 9) / 9.0
        grid = self._read_m(mem.DUNGEON_GRID_POS) / 63.0
        return np.array(
            [in_dungeon, dungeon_id, has_map, has_compass, has_owl, has_nk, small_keys, grid],
            dtype=np.float32,
        )

    def _get_held_items_obs(self):
        a = self._read_m(mem.HELD_ITEM_A) / 13.0
        b = self._read_m(mem.HELD_ITEM_B) / 13.0
        return np.array([a, b], dtype=np.float32)

    def _get_game_progress_obs(self):
        shells = min(self._read_m(mem.SECRET_SHELLS), 26) / 26.0
        leaves = min(self._read_m(mem.GOLDEN_LEAVES), 6) / 6.0
        trading = self._read_m(mem.TRADING_ITEM) / 14.0
        rupees = min(self._read_rupees(), 999) / 999.0
        songs = bin(self._read_m(mem.OCARINA_SONGS) & 0x07).count("1") / 3.0
        kills = min(self.total_kills_detected, 100) / 100.0
        return np.array(
            [shells, leaves, trading, rupees, songs, kills],
            dtype=np.float32,
        )

    # ===========================
    # REWARD SYSTEM V2 — DENSO
    # ===========================

    def _compute_reward(self) -> float:
        """
        Sistema de recompensa V2 — Rewards densos.

        Prioridade de FREQUÊNCIA (não só importância):
        1. Movimento e transição de tela (~todo step)
        2. Rupees e HP recovery (~a cada 10-50 steps)
        3. Kill counters (~a cada 20-100 steps)
        4. Exploração de telas novas (~a cada minuto)
        5. Itens e milestones (~raro)
        6. Penalidades (morte, stuck, inatividade)
        """
        r = 0.0
        cur_health_raw = self._read_m(mem.CURRENT_HEALTH)
        cur_health_frac = self._get_health_fraction()
        cur_room = self._read_m(mem.MAP_ROOM)
        cur_x = self._read_m(mem.LINK_X)
        cur_y = self._read_m(mem.LINK_Y)

        # ---- DETECÇÃO DE MORTE (via HP=0) ----
        if cur_health_raw == 0 and self.prev_health_raw > 0:
            self.is_dead = True
            self.died_count += 1
            r += config.DEATH_PENALTY
            self.cumulative_rewards["death"] += config.DEATH_PENALTY

        # Detectar respawn: HP volta após morte
        if self.is_dead and cur_health_raw > 0:
            self.is_dead = False

        # Durante Game Over / tela de morte, não computar outros rewards
        if self.is_dead:
            r += config.TIME_PENALTY_BASE
            self.cumulative_rewards["time"] += config.TIME_PENALTY_BASE
            return r * config.REWARD_SCALE

        # ---- 1. TRANSIÇÃO DE TELA (DENSO — incentiva movimento entre telas) ----
        if cur_room != self.prev_room:
            r += config.SCREEN_TRANSITION_REWARD
            self.cumulative_rewards["screen_transition"] += config.SCREEN_TRANSITION_REWARD
            self.screen_transitions += 1

        # ---- 2. RUPEES (DENSO — sinal mais frequente de progresso) ----
        cur_rupees = self._read_rupees()
        delta_rupees = cur_rupees - self.prev_rupees
        if delta_rupees > 0:
            rupee_reward = delta_rupees * config.RUPEE_REWARD_SCALE
            r += rupee_reward
            self.cumulative_rewards["rupees"] += rupee_reward
        self.prev_rupees = cur_rupees

        # ---- 3. HP RECOVERY (DENSO — pegar corações = combate/exploração) ----
        if cur_health_frac > self.prev_health_frac and self.prev_health_frac > 0:
            hp_gained = cur_health_frac - self.prev_health_frac
            recovery_reward = hp_gained * config.HEALTH_RECOVERY_REWARD
            r += recovery_reward
            self.cumulative_rewards["hp_recovery"] += recovery_reward

        # ---- 4. COMBATE — Kill counters ----
        cur_kills = self._get_kill_counter()
        delta_kills = cur_kills - self.prev_kill_counter
        if delta_kills > 0:
            kill_reward = delta_kills * config.KILL_BONUS
            r += kill_reward
            self.cumulative_rewards["kills"] += kill_reward
            self.total_kills_detected += delta_kills
        self.prev_kill_counter = cur_kills

        # ---- 5. EXPLORAÇÃO DE TELAS NOVAS (overworld) ----
        cur_explored = count_explored_screens(self._read_m)
        delta_screens = cur_explored - self.prev_explored_count
        if delta_screens > 0:
            screen_reward = delta_screens * config.EXPLORE_NEW_SCREEN
            r += screen_reward
            self.cumulative_rewards["explore_screen"] += screen_reward
        self.prev_explored_count = cur_explored

        # Novas salas de dungeon
        if self._read_m(mem.MAP_CATEGORY) == 0x01:
            room = self._read_m(mem.MAP_ROOM)
            dungeon_id = self._read_m(mem.MAP_DUNGEON_ID)
            room_key = (dungeon_id, room)
            if room_key not in self.visited_dungeon_rooms:
                self.visited_dungeon_rooms.add(room_key)
                r += config.EXPLORE_NEW_ROOM
                self.cumulative_rewards["explore_room"] += config.EXPLORE_NEW_ROOM

        # Bônus intrínseco 1/sqrt(n) (Bellemare et al.)
        pos_key = self._get_position_key()
        visit_count = self.visited_positions.get(pos_key, 0)
        if visit_count > 0:
            count_bonus = config.EXPLORE_COUNT_BONUS / math.sqrt(visit_count)
            r += count_bonus
            self.cumulative_rewards["explore_count"] += count_bonus

        # ---- 6. INSTRUMENTOS (milestone principal) ----
        cur_instruments = self._count_instruments()
        delta_inst = cur_instruments - self.prev_instruments
        if delta_inst > 0:
            r += delta_inst * config.INSTRUMENT_REWARD
            self.cumulative_rewards["instruments"] += delta_inst * config.INSTRUMENT_REWARD
        self.prev_instruments = cur_instruments

        # ---- 7. NOVOS ITENS ----
        cur_inventory = self._get_inventory_set()
        new_items = cur_inventory - self.prev_inventory
        if new_items:
            item_reward = len(new_items) * config.NEW_ITEM_REWARD
            r += item_reward
            self.cumulative_rewards["new_items"] += item_reward
        self.prev_inventory = cur_inventory

        # ---- 8. UPGRADES DE EQUIPAMENTO ----
        cur_sword = self._read_m(mem.SWORD_LEVEL)
        if cur_sword > self.prev_sword_level:
            r += config.EQUIPMENT_UPGRADE
            self.cumulative_rewards["equipment"] += config.EQUIPMENT_UPGRADE
        self.prev_sword_level = cur_sword

        cur_shield = self._read_m(mem.SHIELD_LEVEL)
        if cur_shield > self.prev_shield_level:
            r += config.EQUIPMENT_UPGRADE
            self.cumulative_rewards["equipment"] += config.EQUIPMENT_UPGRADE
        self.prev_shield_level = cur_shield

        cur_bracelet = self._read_m(mem.BRACELET_LEVEL)
        if cur_bracelet > self.prev_bracelet_level:
            r += config.EQUIPMENT_UPGRADE
            self.cumulative_rewards["equipment"] += config.EQUIPMENT_UPGRADE
        self.prev_bracelet_level = cur_bracelet

        # Heart containers
        cur_max_hp = self._read_m(mem.MAX_HEALTH)
        if cur_max_hp > self.prev_max_health:
            hearts_reward = (cur_max_hp - self.prev_max_health) * config.HEART_CONTAINER
            r += hearts_reward
            self.cumulative_rewards["hearts"] += hearts_reward
        self.prev_max_health = cur_max_hp

        # ---- 9. DUNGEON KEYS ----
        cur_dk = self._count_dungeon_entrance_keys()
        delta_dk = cur_dk - self.prev_dungeon_keys_total
        if delta_dk > 0:
            r += delta_dk * config.DUNGEON_KEY_REWARD
            self.cumulative_rewards["dungeon_keys"] += delta_dk * config.DUNGEON_KEY_REWARD
        self.prev_dungeon_keys_total = cur_dk

        cur_sk = self._read_m(mem.DUNGEON_SMALL_KEYS)
        delta_sk = cur_sk - self.prev_small_keys
        if delta_sk > 0:
            r += delta_sk * config.SMALL_KEY_REWARD
            self.cumulative_rewards["small_keys"] += delta_sk * config.SMALL_KEY_REWARD
        self.prev_small_keys = cur_sk

        # Dungeon items (map, compass, owl beak, nightmare key)
        cur_di = self._count_current_dungeon_items()
        delta_di = cur_di - self.prev_dungeon_items
        if delta_di > 0:
            r += delta_di * config.DUNGEON_ITEM_REWARD
            self.cumulative_rewards["dungeon_items"] += delta_di * config.DUNGEON_ITEM_REWARD
        self.prev_dungeon_items = cur_di

        # ---- 10. COLECIONÁVEIS ----
        cur_shells = self._read_m(mem.SECRET_SHELLS)
        delta_shells = cur_shells - self.prev_shells
        if delta_shells > 0:
            r += delta_shells * config.SHELL_REWARD
            self.cumulative_rewards["shells"] += delta_shells * config.SHELL_REWARD
        self.prev_shells = cur_shells

        cur_leaves = self._read_m(mem.GOLDEN_LEAVES)
        delta_leaves = cur_leaves - self.prev_leaves
        if delta_leaves > 0:
            r += delta_leaves * config.GOLDEN_LEAF_REWARD
            self.cumulative_rewards["leaves"] += delta_leaves * config.GOLDEN_LEAF_REWARD
        self.prev_leaves = cur_leaves

        cur_trading = self._read_m(mem.TRADING_ITEM)
        delta_trading = cur_trading - self.prev_trading
        if delta_trading > 0:
            r += delta_trading * config.TRADING_STEP_REWARD
            self.cumulative_rewards["trading"] += delta_trading * config.TRADING_STEP_REWARD
        self.prev_trading = cur_trading

        # ---- 11. HEALTH LOSS (tomar dano) ----
        if cur_health_frac < self.prev_health_frac and cur_health_raw > 0:
            hp_lost = self.prev_health_frac - cur_health_frac
            hp_cost = hp_lost * config.HEALTH_LOSS_PENALTY
            r += hp_cost
            self.cumulative_rewards["health_loss"] += hp_cost

        # ---- 12. TIME PENALTY + IDLE PENALTY ----
        moved = (cur_x != self.prev_x or cur_y != self.prev_y or cur_room != self.prev_room)
        if moved:
            r += config.TIME_PENALTY_BASE
            self.cumulative_rewards["time"] += config.TIME_PENALTY_BASE
        else:
            r += config.IDLE_PENALTY
            self.cumulative_rewards["idle"] += config.IDLE_PENALTY

        # ---- 13. STUCK PENALTY ----
        stuck_cost = self._get_stuck_penalty()
        r += stuck_cost
        self.cumulative_rewards["stuck"] += stuck_cost

        return r * config.REWARD_SCALE

    # ===========================
    # EMULADOR
    # ===========================

    def _run_action(self, action):
        self.pyboy.send_input(VALID_ACTIONS[action])
        render = not self.headless
        hold_frames = min(8, config.ACTION_FREQ - 1)
        for _ in range(hold_frames):
            self.pyboy.tick(1, render)
        self.pyboy.send_input(RELEASE_ACTIONS[action])
        remaining = config.ACTION_FREQ - hold_frames - 1
        for _ in range(remaining):
            self.pyboy.tick(1, render)
        self.pyboy.tick(1, True)

    # ===========================
    # LEITURA DE MEMÓRIA
    # ===========================

    def _read_m(self, addr):
        return self.pyboy.memory[addr]

    def _get_health_fraction(self):
        current = self._read_m(mem.CURRENT_HEALTH)
        maximum = self._read_m(mem.MAX_HEALTH) * 8
        return current / max(maximum, 1)

    def _read_rupees(self):
        high = self._read_m(mem.RUPEES_HIGH)
        low = self._read_m(mem.RUPEES_LOW)
        try:
            return int(f"{high:02x}{low:02x}")
        except ValueError:
            return 0

    def _get_inventory_set(self):
        items = set()
        for addr in range(mem.INVENTORY_START, mem.INVENTORY_END + 1):
            val = self._read_m(addr)
            if val in mem.ALL_ITEMS:
                items.add(val)
        for addr in [mem.HELD_ITEM_A, mem.HELD_ITEM_B]:
            val = self._read_m(addr)
            if val in mem.ALL_ITEMS:
                items.add(val)
        return items

    def _count_instruments(self):
        count = 0
        for i in range(mem.NUM_INSTRUMENTS):
            if self._read_m(mem.INSTRUMENTS_START + i) == mem.INSTRUMENT_COLLECTED:
                count += 1
        return count

    def _count_dungeon_entrance_keys(self):
        count = 0
        for i in range(mem.NUM_DUNGEON_KEYS):
            if self._read_m(mem.DUNGEON_KEY_START + i) > 0:
                count += 1
        return count

    def _count_current_dungeon_items(self):
        count = 0
        for addr in [mem.DUNGEON_MAP_FLAG, mem.DUNGEON_COMPASS,
                      mem.DUNGEON_OWL_BEAK, mem.DUNGEON_NIGHTMARE_KEY]:
            if self._read_m(addr) > 0:
                count += 1
        return count

    def _get_kill_counter(self):
        return (
            self._read_m(mem.PIECE_OF_POWER_KILLS)
            + self._read_m(mem.GUARDIAN_ACORN_KILLS)
        )

    def _fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs)).astype(np.float32)

    # ===========================
    # EXPLORAÇÃO / TRACKING
    # ===========================

    def _get_position_key(self):
        room = self._read_m(mem.MAP_ROOM)
        x = self._read_m(mem.LINK_X)
        y = self._read_m(mem.LINK_Y)
        cat = self._read_m(mem.MAP_CATEGORY)
        dng = self._read_m(mem.MAP_DUNGEON_ID) if cat != 0 else 0
        return f"{cat}:{dng}:{room}:{x >> 3}:{y >> 3}"

    def _update_position_tracking(self):
        pos_key = self._get_position_key()
        self.visited_positions[pos_key] = self.visited_positions.get(pos_key, 0) + 1

        cur_room = self._read_m(mem.MAP_ROOM)
        cat = self._read_m(mem.MAP_CATEGORY)
        screen_key = (cat, self._read_m(mem.MAP_DUNGEON_ID), cur_room)
        self.visited_screens.add(screen_key)

        if cur_room != self.last_room:
            self.steps_on_current_screen = 0
            self.last_room = cur_room
        else:
            self.steps_on_current_screen += 1

    def _get_stuck_penalty(self):
        if self.steps_on_current_screen > config.STUCK_SAME_SCREEN_STEPS:
            return config.STUCK_PENALTY

        pos_key = self._get_position_key()
        count = self.visited_positions.get(pos_key, 0)
        if count > config.STUCK_VISIT_THRESHOLD:
            return config.STUCK_PENALTY

        return 0.0

    # ===========================
    # INFO
    # ===========================

    def _build_info(self):
        return {
            "step": self.step_count,
            "screens_explored": self.prev_explored_count,
            "total_screens_visited": len(self.visited_screens),
            "dungeon_rooms": len(self.visited_dungeon_rooms),
            "screen_transitions": self.screen_transitions,
            "instruments": self._count_instruments(),
            "inventory_count": len(self._get_inventory_set()),
            "sword_level": self._read_m(mem.SWORD_LEVEL),
            "shield_level": self._read_m(mem.SHIELD_LEVEL),
            "max_hearts": self._read_m(mem.MAX_HEALTH),
            "hp_fraction": self._get_health_fraction(),
            "deaths": self.died_count,
            "kills": self.total_kills_detected,
            "shells": self._read_m(mem.SECRET_SHELLS),
            "leaves": self._read_m(mem.GOLDEN_LEAVES),
            "trading_item": self._read_m(mem.TRADING_ITEM),
            "rupees": self._read_rupees(),
            "dungeon_keys": self._count_dungeon_entrance_keys(),
            "in_dungeon": self._read_m(mem.MAP_CATEGORY) == 0x01,
            "current_room": self._read_m(mem.MAP_ROOM),
            "total_reward": self.total_reward,
            "unique_positions": len(self.visited_positions),
            **{f"rew_{k}": v for k, v in self.cumulative_rewards.items()},
        }

    # ===========================
    # RENDER / CLOSE
    # ===========================

    def render(self):
        return self._render_screen()

    def close(self):
        try:
            self.pyboy.stop()
        except Exception:
            pass
