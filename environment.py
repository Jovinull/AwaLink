"""
environment.py — Gymnasium Environment para Zelda: Link's Awakening via PyBoy.

V3 — Melhorias de nível senior:
  - Episódios curtos (~30k steps) para mais resets e aprendizado mais rápido
  - Anti-oscillation em screen transitions (cooldown de N rooms)
  - Auto-navegação do menu de Game Over (aperta A automaticamente)
  - Detecção de interação com o mundo via hash dos tile data (D700-D79B)
  - Observação de combate: contagem de entidades, power-up ativo, direção do Link
  - Observação de ammo: bombas/flechas/pó normalizados
  - Dungeon flags tracking (DB16-DB3D) para progresso em dungeons
  - Explore count cortado em 100 visitas (reduz ruído no sinal)
  - BCD parsing robusto para rupees/ammo
  - Reward normalization manual (sem VecNormalize reward)
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

A_BUTTON_IDX = 4


class ZeldaLinksAwakeningEnv(gym.Env):
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
                low=0, high=255, shape=self.output_shape, dtype=np.uint8,
            ),
            "health": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "position": spaces.Box(
                low=-1, high=1, shape=(self.enc_freqs * 3,), dtype=np.float32,
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
            "combat_info": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            "ammo": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            "recent_actions": spaces.MultiDiscrete(
                [len(VALID_ACTIONS)] * self.frame_stacks,
            ),
        })

        window = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            config.ROM_PATH, window=window,
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

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        # Exploração
        self.visited_positions = {}
        self.visited_screens = set()
        self.visited_dungeon_rooms = set()
        self.prev_explored_count = count_explored_screens(self._read_m)
        self.recent_rooms = []

        # Posição
        self.prev_x = self._read_m(mem.LINK_X)
        self.prev_y = self._read_m(mem.LINK_Y)
        self.prev_room = self._read_m(mem.MAP_ROOM)

        # Inventário / progresso
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

        # Saúde
        self.prev_health_raw = self._read_m(mem.CURRENT_HEALTH)
        self.prev_health_frac = self._get_health_fraction()
        self.is_dead = False
        self.dead_steps = 0

        # Interação com o mundo
        self.prev_map_hash = self._map_data_hash()
        self.prev_dungeon_flag_bits = self._count_dungeon_flag_bits()

        # Contadores
        self.died_count = 0
        self.total_kills_detected = 0
        self.step_count = 0
        self.steps_on_current_screen = 0
        self.last_room = self._read_m(mem.MAP_ROOM)
        self.screen_transitions = 0
        self.world_interactions = 0

        self.total_reward = 0.0
        self.cumulative_rewards = {
            "explore_screen": 0.0, "explore_room": 0.0, "explore_count": 0.0,
            "screen_transition": 0.0, "instruments": 0.0, "new_items": 0.0,
            "equipment": 0.0, "hearts": 0.0, "dungeon_keys": 0.0,
            "small_keys": 0.0, "dungeon_items": 0.0, "dungeon_flags": 0.0,
            "shells": 0.0, "leaves": 0.0, "trading": 0.0,
            "kills": 0.0, "rupees": 0.0, "hp_recovery": 0.0,
            "world_interact": 0.0,
            "time": 0.0, "idle": 0.0, "death": 0.0,
            "health_loss": 0.0, "stuck": 0.0,
        }

        self.reset_count += 1
        return self._get_obs(), {}

    # ===========================
    # STEP
    # ===========================

    def step(self, action):
        # Auto-navegar Game Over menu: após animação de morte, pressiona A
        if self.is_dead:
            self.dead_steps += 1
            if self.dead_steps > config.GAME_OVER_WAIT_STEPS:
                action = A_BUTTON_IDX

        self._run_action(action)
        self._update_recent_actions(action)
        self._update_position_tracking()

        step_reward = self._compute_reward()
        self.total_reward += step_reward

        self.prev_health_raw = self._read_m(mem.CURRENT_HEALTH)
        self.prev_health_frac = self._get_health_fraction()
        self.prev_x = self._read_m(mem.LINK_X)
        self.prev_y = self._read_m(mem.LINK_Y)
        self.prev_room = self._read_m(mem.MAP_ROOM)

        self.step_count += 1
        truncated = self.step_count >= config.MAX_STEPS_PER_EPISODE

        return self._get_obs(), step_reward, False, truncated, self._build_info()

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
            "combat_info": self._get_combat_info_obs(),
            "ammo": self._get_ammo_obs(),
            "recent_actions": self.recent_actions,
        }

    def _render_screen(self):
        return self.pyboy.screen.ndarray[::2, ::2, 0:1].astype(np.uint8)

    def _update_recent_screens(self, screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = screen[:, :, 0]

    def _update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def _get_health_obs(self):
        return np.array([
            self._get_health_fraction(),
            self._read_m(mem.MAX_HEALTH) / 14.0,
        ], dtype=np.float32)

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
        return np.array([
            self._read_m(mem.SWORD_LEVEL) / 2.0,
            self._read_m(mem.SHIELD_LEVEL) / 3.0,
            self._read_m(mem.BRACELET_LEVEL) / 2.0,
            float(self._read_m(mem.FLIPPERS) > 0),
            float(self._read_m(mem.POTION) > 0),
            1.0 if mem.ITEM_BOOTS in self._get_inventory_set() else 0.0,
        ], dtype=np.float32)

    def _get_instruments_binary(self):
        return np.array([
            1 if self._read_m(mem.INSTRUMENTS_START + i) == mem.INSTRUMENT_COLLECTED else 0
            for i in range(mem.NUM_INSTRUMENTS)
        ], dtype=np.int8)

    def _get_overworld_map_obs(self):
        return build_overworld_exploration_map(self._read_m)[:, :, np.newaxis]

    def _get_dungeon_state_obs(self):
        cat = self._read_m(mem.MAP_CATEGORY)
        return np.array([
            float(cat == 0x01),
            self._read_m(mem.MAP_DUNGEON_ID) / 9.0,
            float(self._read_m(mem.DUNGEON_MAP_FLAG) > 0),
            float(self._read_m(mem.DUNGEON_COMPASS) > 0),
            float(self._read_m(mem.DUNGEON_OWL_BEAK) > 0),
            float(self._read_m(mem.DUNGEON_NIGHTMARE_KEY) > 0),
            min(self._read_m(mem.DUNGEON_SMALL_KEYS), 9) / 9.0,
            self._read_m(mem.DUNGEON_GRID_POS) / 63.0,
        ], dtype=np.float32)

    def _get_held_items_obs(self):
        return np.array([
            self._read_m(mem.HELD_ITEM_A) / 13.0,
            self._read_m(mem.HELD_ITEM_B) / 13.0,
        ], dtype=np.float32)

    def _get_game_progress_obs(self):
        return np.array([
            min(self._read_m(mem.SECRET_SHELLS), 26) / 26.0,
            min(self._read_m(mem.GOLDEN_LEAVES), 6) / 6.0,
            self._read_m(mem.TRADING_ITEM) / 14.0,
            min(self._read_rupees(), 999) / 999.0,
            bin(self._read_m(mem.OCARINA_SONGS) & 0x07).count("1") / 3.0,
            min(self.total_kills_detected, 100) / 100.0,
        ], dtype=np.float32)

    def _get_combat_info_obs(self):
        """Entidades ativas, power-up, direção, estado do terreno."""
        return np.array([
            self._count_active_entities() / 15.0,
            float(self._read_m(mem.PIECE_OF_POWER_ACTIVE) == 0x01),
            self._read_m(mem.LINK_DIRECTION) / 3.0,
            min(self._read_m(mem.LINK_GROUND_STATE), 7) / 7.0,
        ], dtype=np.float32)

    def _get_ammo_obs(self):
        """Bombas, flechas e pó normalizados pela capacidade máxima."""
        max_b = max(self._read_bcd_byte(mem.MAX_BOMBS), 1)
        max_a = max(self._read_bcd_byte(mem.MAX_ARROWS), 1)
        max_p = max(self._read_bcd_byte(mem.MAX_POWDER), 1)
        return np.array([
            self._read_bcd_byte(mem.BOMB_COUNT) / max_b,
            self._read_bcd_byte(mem.ARROW_COUNT) / max_a,
            self._read_bcd_byte(mem.POWDER_COUNT) / max_p,
        ], dtype=np.float32)

    # ===========================
    # REWARD SYSTEM V3
    # ===========================

    def _compute_reward(self) -> float:
        r = 0.0
        cur_health_raw = self._read_m(mem.CURRENT_HEALTH)
        cur_health_frac = self._get_health_fraction()
        cur_room = self._read_m(mem.MAP_ROOM)
        cur_x = self._read_m(mem.LINK_X)
        cur_y = self._read_m(mem.LINK_Y)

        # ---- MORTE (HP → 0) ----
        if cur_health_raw == 0 and self.prev_health_raw > 0:
            self.is_dead = True
            self.dead_steps = 0
            self.died_count += 1
            r += config.DEATH_PENALTY
            self.cumulative_rewards["death"] += config.DEATH_PENALTY

        if self.is_dead and cur_health_raw > 0:
            self.is_dead = False
            self.dead_steps = 0

        if self.is_dead:
            r += config.TIME_PENALTY_BASE
            self.cumulative_rewards["time"] += config.TIME_PENALTY_BASE
            return r * config.REWARD_SCALE

        # ---- SCREEN TRANSITION (anti-oscillation) ----
        if cur_room != self.prev_room:
            cooldown = self.recent_rooms[-config.SCREEN_TRANSITION_COOLDOWN:]
            if cur_room not in cooldown:
                r += config.SCREEN_TRANSITION_REWARD
                self.cumulative_rewards["screen_transition"] += config.SCREEN_TRANSITION_REWARD
            self.screen_transitions += 1
            self.recent_rooms.append(cur_room)
            if len(self.recent_rooms) > 30:
                self.recent_rooms = self.recent_rooms[-30:]

        # ---- RUPEES ----
        cur_rupees = self._read_rupees()
        delta_rupees = cur_rupees - self.prev_rupees
        if delta_rupees > 0:
            val = delta_rupees * config.RUPEE_REWARD_SCALE
            r += val
            self.cumulative_rewards["rupees"] += val
        self.prev_rupees = cur_rupees

        # ---- HP RECOVERY ----
        if cur_health_frac > self.prev_health_frac and self.prev_health_frac > 0:
            val = (cur_health_frac - self.prev_health_frac) * config.HEALTH_RECOVERY_REWARD
            r += val
            self.cumulative_rewards["hp_recovery"] += val

        # ---- KILL COUNTERS ----
        cur_kills = self._get_kill_counter()
        delta_kills = cur_kills - self.prev_kill_counter
        if delta_kills > 0:
            val = delta_kills * config.KILL_BONUS
            r += val
            self.cumulative_rewards["kills"] += val
            self.total_kills_detected += delta_kills
        self.prev_kill_counter = cur_kills

        # ---- WORLD INTERACTION (map tile data changed) ----
        cur_hash = self._map_data_hash()
        if cur_hash != self.prev_map_hash:
            r += config.WORLD_INTERACTION_REWARD
            self.cumulative_rewards["world_interact"] += config.WORLD_INTERACTION_REWARD
            self.world_interactions += 1
        self.prev_map_hash = cur_hash

        # ---- OVERWORLD EXPLORATION ----
        cur_explored = count_explored_screens(self._read_m)
        delta_screens = cur_explored - self.prev_explored_count
        if delta_screens > 0:
            val = delta_screens * config.EXPLORE_NEW_SCREEN
            r += val
            self.cumulative_rewards["explore_screen"] += val
        self.prev_explored_count = cur_explored

        # ---- DUNGEON ROOMS ----
        if self._read_m(mem.MAP_CATEGORY) == 0x01:
            room_key = (self._read_m(mem.MAP_DUNGEON_ID), self._read_m(mem.MAP_ROOM))
            if room_key not in self.visited_dungeon_rooms:
                self.visited_dungeon_rooms.add(room_key)
                r += config.EXPLORE_NEW_ROOM
                self.cumulative_rewards["explore_room"] += config.EXPLORE_NEW_ROOM

        # ---- EXPLORE COUNT BONUS (capped at 100 visits) ----
        pos_key = self._get_position_key()
        visit_count = self.visited_positions.get(pos_key, 0)
        if 0 < visit_count < config.EXPLORE_COUNT_MAX_VISITS:
            val = config.EXPLORE_COUNT_BONUS / math.sqrt(visit_count)
            r += val
            self.cumulative_rewards["explore_count"] += val

        # ---- DUNGEON FLAGS (DB16-DB3D) ----
        cur_dfb = self._count_dungeon_flag_bits()
        delta_dfb = cur_dfb - self.prev_dungeon_flag_bits
        if delta_dfb > 0:
            val = delta_dfb * config.DUNGEON_FLAGS_REWARD
            r += val
            self.cumulative_rewards["dungeon_flags"] += val
        self.prev_dungeon_flag_bits = cur_dfb

        # ---- INSTRUMENTS ----
        cur_instruments = self._count_instruments()
        delta_inst = cur_instruments - self.prev_instruments
        if delta_inst > 0:
            val = delta_inst * config.INSTRUMENT_REWARD
            r += val
            self.cumulative_rewards["instruments"] += val
        self.prev_instruments = cur_instruments

        # ---- NEW ITEMS ----
        cur_inv = self._get_inventory_set()
        new_items = cur_inv - self.prev_inventory
        if new_items:
            val = len(new_items) * config.NEW_ITEM_REWARD
            r += val
            self.cumulative_rewards["new_items"] += val
        self.prev_inventory = cur_inv

        # ---- EQUIPMENT UPGRADES ----
        for attr, addr in [
            ("prev_sword_level", mem.SWORD_LEVEL),
            ("prev_shield_level", mem.SHIELD_LEVEL),
            ("prev_bracelet_level", mem.BRACELET_LEVEL),
        ]:
            cur = self._read_m(addr)
            if cur > getattr(self, attr):
                r += config.EQUIPMENT_UPGRADE
                self.cumulative_rewards["equipment"] += config.EQUIPMENT_UPGRADE
            setattr(self, attr, cur)

        # ---- HEART CONTAINERS ----
        cur_max_hp = self._read_m(mem.MAX_HEALTH)
        if cur_max_hp > self.prev_max_health:
            val = (cur_max_hp - self.prev_max_health) * config.HEART_CONTAINER
            r += val
            self.cumulative_rewards["hearts"] += val
        self.prev_max_health = cur_max_hp

        # ---- DUNGEON KEYS ----
        cur_dk = self._count_dungeon_entrance_keys()
        if cur_dk > self.prev_dungeon_keys_total:
            val = (cur_dk - self.prev_dungeon_keys_total) * config.DUNGEON_KEY_REWARD
            r += val
            self.cumulative_rewards["dungeon_keys"] += val
        self.prev_dungeon_keys_total = cur_dk

        cur_sk = self._read_m(mem.DUNGEON_SMALL_KEYS)
        if cur_sk > self.prev_small_keys:
            val = (cur_sk - self.prev_small_keys) * config.SMALL_KEY_REWARD
            r += val
            self.cumulative_rewards["small_keys"] += val
        self.prev_small_keys = cur_sk

        cur_di = self._count_current_dungeon_items()
        if cur_di > self.prev_dungeon_items:
            val = (cur_di - self.prev_dungeon_items) * config.DUNGEON_ITEM_REWARD
            r += val
            self.cumulative_rewards["dungeon_items"] += val
        self.prev_dungeon_items = cur_di

        # ---- COLLECTIBLES ----
        for attr, addr, reward_val in [
            ("prev_shells", mem.SECRET_SHELLS, config.SHELL_REWARD),
            ("prev_leaves", mem.GOLDEN_LEAVES, config.GOLDEN_LEAF_REWARD),
            ("prev_trading", mem.TRADING_ITEM, config.TRADING_STEP_REWARD),
        ]:
            cur = self._read_m(addr)
            if cur > getattr(self, attr):
                val = (cur - getattr(self, attr)) * reward_val
                r += val
                self.cumulative_rewards[attr.replace("prev_", "")] += val
            setattr(self, attr, cur)

        # ---- HEALTH LOSS ----
        if cur_health_frac < self.prev_health_frac and cur_health_raw > 0:
            val = (self.prev_health_frac - cur_health_frac) * config.HEALTH_LOSS_PENALTY
            r += val
            self.cumulative_rewards["health_loss"] += val

        # ---- TIME / IDLE PENALTY ----
        moved = (cur_x != self.prev_x or cur_y != self.prev_y or cur_room != self.prev_room)
        if moved:
            r += config.TIME_PENALTY_BASE
            self.cumulative_rewards["time"] += config.TIME_PENALTY_BASE
        else:
            r += config.IDLE_PENALTY
            self.cumulative_rewards["idle"] += config.IDLE_PENALTY

        # ---- STUCK ----
        r += self._get_stuck_penalty()

        return r * config.REWARD_SCALE

    # ===========================
    # EMULADOR
    # ===========================

    def _run_action(self, action):
        self.pyboy.send_input(VALID_ACTIONS[action])
        render = not self.headless
        hold = min(8, config.ACTION_FREQ - 1)
        for _ in range(hold):
            self.pyboy.tick(1, render)
        self.pyboy.send_input(RELEASE_ACTIONS[action])
        for _ in range(config.ACTION_FREQ - hold - 1):
            self.pyboy.tick(1, render)
        self.pyboy.tick(1, True)

    # ===========================
    # LEITURA DE MEMÓRIA
    # ===========================

    def _read_m(self, addr):
        return self.pyboy.memory[addr]

    def _read_bcd_byte(self, addr):
        """Decodifica um byte BCD. Clamp de nibbles inválidos."""
        val = self._read_m(addr)
        high = min((val >> 4) & 0x0F, 9)
        low = min(val & 0x0F, 9)
        return high * 10 + low

    def _get_health_fraction(self):
        current = self._read_m(mem.CURRENT_HEALTH)
        maximum = self._read_m(mem.MAX_HEALTH) * 8
        return current / max(maximum, 1)

    def _read_rupees(self):
        """BCD robusto para 2 bytes de rupees."""
        high = self._read_m(mem.RUPEES_HIGH)
        low = self._read_m(mem.RUPEES_LOW)
        d3 = min((high >> 4) & 0x0F, 9)
        d2 = min(high & 0x0F, 9)
        d1 = min((low >> 4) & 0x0F, 9)
        d0 = min(low & 0x0F, 9)
        return d3 * 1000 + d2 * 100 + d1 * 10 + d0

    def _get_inventory_set(self):
        items = set()
        all_set = set(mem.ALL_ITEMS)
        for addr in range(mem.INVENTORY_START, mem.INVENTORY_END + 1):
            val = self._read_m(addr)
            if val in all_set:
                items.add(val)
        for addr in (mem.HELD_ITEM_A, mem.HELD_ITEM_B):
            val = self._read_m(addr)
            if val in all_set:
                items.add(val)
        return items

    def _count_instruments(self):
        return sum(
            1 for i in range(mem.NUM_INSTRUMENTS)
            if self._read_m(mem.INSTRUMENTS_START + i) == mem.INSTRUMENT_COLLECTED
        )

    def _count_dungeon_entrance_keys(self):
        return sum(
            1 for i in range(mem.NUM_DUNGEON_KEYS)
            if self._read_m(mem.DUNGEON_KEY_START + i) > 0
        )

    def _count_current_dungeon_items(self):
        return sum(
            1 for addr in (
                mem.DUNGEON_MAP_FLAG, mem.DUNGEON_COMPASS,
                mem.DUNGEON_OWL_BEAK, mem.DUNGEON_NIGHTMARE_KEY,
            ) if self._read_m(addr) > 0
        )

    def _get_kill_counter(self):
        return self._read_m(mem.PIECE_OF_POWER_KILLS) + self._read_m(mem.GUARDIAN_ACORN_KILLS)

    def _count_active_entities(self):
        """Entidades ativas na tela (slots 1-15, slot 0 é Link)."""
        return sum(
            1 for i in range(1, 16)
            if self._read_m(mem.ENTITY_STATE_TABLE + i) != 0
        )

    def _map_data_hash(self):
        """Hash rápido dos tile data carregados (D700-D79B)."""
        h = 0
        for addr in range(mem.MAP_DATA_START, mem.MAP_DATA_END + 1):
            h = ((h << 5) - h + self._read_m(addr)) & 0xFFFFFFFF
        return h

    def _count_dungeon_flag_bits(self):
        """Total de bits setados em DB16-DB3D (progresso em dungeons)."""
        return sum(
            bin(self._read_m(addr)).count("1")
            for addr in range(mem.DUNGEON_FLAGS_START, mem.DUNGEON_FLAGS_END + 1)
        )

    def _fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs)).astype(np.float32)

    # ===========================
    # EXPLORAÇÃO / TRACKING
    # ===========================

    def _get_position_key(self):
        cat = self._read_m(mem.MAP_CATEGORY)
        dng = self._read_m(mem.MAP_DUNGEON_ID) if cat != 0 else 0
        room = self._read_m(mem.MAP_ROOM)
        x = self._read_m(mem.LINK_X) >> 3
        y = self._read_m(mem.LINK_Y) >> 3
        return (cat, dng, room, x, y)

    def _update_position_tracking(self):
        pos_key = self._get_position_key()
        self.visited_positions[pos_key] = self.visited_positions.get(pos_key, 0) + 1

        cur_room = self._read_m(mem.MAP_ROOM)
        cat = self._read_m(mem.MAP_CATEGORY)
        self.visited_screens.add((cat, self._read_m(mem.MAP_DUNGEON_ID), cur_room))

        if cur_room != self.last_room:
            self.steps_on_current_screen = 0
            self.last_room = cur_room
        else:
            self.steps_on_current_screen += 1

    def _get_stuck_penalty(self):
        if self.steps_on_current_screen > config.STUCK_SAME_SCREEN_STEPS:
            penalty = config.STUCK_PENALTY
            self.cumulative_rewards["stuck"] += penalty
            return penalty

        pos_key = self._get_position_key()
        count = self.visited_positions.get(pos_key, 0)
        if count > config.STUCK_VISIT_THRESHOLD:
            penalty = config.STUCK_PENALTY
            self.cumulative_rewards["stuck"] += penalty
            return penalty

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
            "world_interactions": self.world_interactions,
            "instruments": self._count_instruments(),
            "inventory_count": len(self._get_inventory_set()),
            "sword_level": self._read_m(mem.SWORD_LEVEL),
            "shield_level": self._read_m(mem.SHIELD_LEVEL),
            "max_hearts": self._read_m(mem.MAX_HEALTH),
            "hp_fraction": self._get_health_fraction(),
            "deaths": self.died_count,
            "kills": self.total_kills_detected,
            "entities": self._count_active_entities(),
            "shells": self._read_m(mem.SECRET_SHELLS),
            "leaves": self._read_m(mem.GOLDEN_LEAVES),
            "trading_item": self._read_m(mem.TRADING_ITEM),
            "rupees": self._read_rupees(),
            "dungeon_keys": self._count_dungeon_entrance_keys(),
            "dungeon_flag_bits": self.prev_dungeon_flag_bits,
            "in_dungeon": self._read_m(mem.MAP_CATEGORY) == 0x01,
            "current_room": self._read_m(mem.MAP_ROOM),
            "total_reward": self.total_reward,
            "unique_positions": len(self.visited_positions),
            **{f"rew_{k}": v for k, v in self.cumulative_rewards.items()},
        }

    def render(self):
        return self._render_screen()

    def close(self):
        try:
            self.pyboy.stop()
        except Exception:
            pass
