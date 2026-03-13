"""
global_map.py — Sistema de mapa global para Zelda: Link's Awakening.

O overworld de Link's Awakening é uma grade 16x16 = 256 telas.
Cada dungeon tem uma grade 8x8 = 64 salas.
Este módulo gerencia coordenadas globais unificadas para o agente de RL.
"""
import numpy as np
import memory_map as mem

OVERWORLD_SHAPE = (mem.OVERWORLD_HEIGHT, mem.OVERWORLD_WIDTH)  # (16, 16)
DUNGEON_SHAPE = (8, 8)
NUM_DUNGEONS = 9  # 8 + Color Dungeon

# Mapa global: overworld (16x16) + espaço para dungeon atual (8x8)
# Layout: overworld ocupa (0:16, 0:16), dungeon ocupa (16:24, 0:8)
GLOBAL_MAP_SHAPE = (24, 16)


def get_overworld_pos(room_number: int) -> tuple[int, int]:
    """Room number do overworld → (row, col) na grade 16x16."""
    col = room_number % mem.OVERWORLD_WIDTH
    row = room_number // mem.OVERWORLD_WIDTH
    return (
        np.clip(row, 0, mem.OVERWORLD_HEIGHT - 1),
        np.clip(col, 0, mem.OVERWORLD_WIDTH - 1),
    )


def get_dungeon_pos(grid_pos: int) -> tuple[int, int]:
    """Posição no grid 8x8 do dungeon → (row, col)."""
    col = grid_pos % DUNGEON_SHAPE[1]
    row = grid_pos // DUNGEON_SHAPE[1]
    return (
        np.clip(row, 0, DUNGEON_SHAPE[0] - 1),
        np.clip(col, 0, DUNGEON_SHAPE[1] - 1),
    )


def build_overworld_exploration_map(memory_reader) -> np.ndarray:
    """
    Constrói mapa 16x16 de exploração do overworld a partir da RAM.
    Cada byte em D800-D8FF representa o status de uma tela.
    Retorna array (16, 16) com valores 0-255 representando exploração.
    """
    explore_map = np.zeros(OVERWORLD_SHAPE, dtype=np.uint8)
    for i in range(mem.WORLD_MAP_SIZE):
        status = memory_reader(mem.WORLD_MAP_START + i)
        row, col = get_overworld_pos(i)
        if status & 0x80:  # visited
            explore_map[row, col] = 255
        elif status & 0x10:  # event occurred
            explore_map[row, col] = 192
        elif status & 0x20:  # owl talked
            explore_map[row, col] = 128
        elif status & 0x04:  # bombed
            explore_map[row, col] = 64
    return explore_map


def count_explored_screens(memory_reader) -> int:
    """Conta quantas telas do overworld foram visitadas (bit 0x80)."""
    count = 0
    for i in range(mem.WORLD_MAP_SIZE):
        if memory_reader(mem.WORLD_MAP_START + i) & 0x80:
            count += 1
    return count


def count_map_events(memory_reader) -> int:
    """Conta quantas telas tiveram eventos (bit 0x10)."""
    count = 0
    for i in range(mem.WORLD_MAP_SIZE):
        if memory_reader(mem.WORLD_MAP_START + i) & 0x10:
            count += 1
    return count
