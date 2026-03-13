"""
memory_map.py — Endereços de Memória RAM do The Legend of Zelda: Link's Awakening (Game Boy)

Fontes:
  - https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)/RAM_map
  - https://wiki.zeldahacking.net/oracle/LADX_RAM_addresses
  - https://github.com/zladx/LADX-Disassembly
"""

# ===========================
# POSIÇÃO DO JOGADOR (HRAM)
# ===========================
LINK_X = 0xFF98
LINK_Y = 0xFF99
LINK_DIRECTION = 0xFF9E       # 0=east, 1=west, 2=north, 3=south
GLOBAL_FRAME_COUNTER = 0xFFE7

# Sub-pixel positions (Bank 0)
LINK_SUB_X = 0xC11A
LINK_SUB_Y = 0xC11B

# ===========================
# ESTADO DO LINK (Bank 0)
# ===========================
LINK_SWIMMING = 0xC11C
LINK_GROUND_STATE = 0xC11F    # 0x00=dry, 0x01=steps, 0x03=wet/grass, 0x07=pit
LINK_SPIN_CHARGE = 0xC122     # 0x28 = fully charged
LINK_AERIAL = 0xC146
LINK_BOOTS_CHARGE = 0xC14B    # 0x1F = fully charged

# ===========================
# MAPA / SALA ATUAL (Bank 1)
# ===========================
MAP_CATEGORY = 0xD401         # 0x00=overworld, 0x01=dungeon, 0x02=side-scroll
MAP_DUNGEON_ID = 0xD402       # Sub-map / dungeon number (0x00-0x1F, 0xFF=Color Dungeon)
MAP_ROOM = 0xD403             # Room number within current sub-map
WARP_X = 0xD404
WARP_Y = 0xD405

DUNGEON_GRID_POS = 0xDBBE     # Position on 8x8 dungeon grid (when in dungeon)

# Currently loaded map tile data
MAP_DATA_START = 0xD700
MAP_DATA_END = 0xD79B

# ===========================
# MAPA MUNDIAL - STATUS DE EXPLORAÇÃO
# 256 bytes, um por tela do overworld (16x16 grid)
# Bitmask: 0x80=visited, 0x20=owl talked, 0x10=event occurred, 0x04=bombed
# ===========================
WORLD_MAP_START = 0xD800
WORLD_MAP_END = 0xD8FF
WORLD_MAP_SIZE = 256          # 16 x 16 screens

# ===========================
# INVENTÁRIO
# ===========================
HELD_ITEM_B = 0xDB00          # Item no botão B
HELD_ITEM_A = 0xDB01          # Item no botão A
INVENTORY_START = 0xDB02      # 10 slots de inventário (DB02-DB0B)
INVENTORY_END = 0xDB0B
INVENTORY_SIZE = 10

# Item IDs
ITEM_NONE = 0x00
ITEM_SWORD = 0x01
ITEM_BOMBS = 0x02
ITEM_BRACELET = 0x03
ITEM_SHIELD = 0x04
ITEM_BOW = 0x05
ITEM_HOOKSHOT = 0x06
ITEM_FIRE_ROD = 0x07
ITEM_BOOTS = 0x08
ITEM_OCARINA = 0x09
ITEM_FEATHER = 0x0A
ITEM_SHOVEL = 0x0B
ITEM_POWDER = 0x0C
ITEM_BOOMERANG = 0x0D

ALL_ITEMS = [
    ITEM_SWORD, ITEM_BOMBS, ITEM_BRACELET, ITEM_SHIELD,
    ITEM_BOW, ITEM_HOOKSHOT, ITEM_FIRE_ROD, ITEM_BOOTS,
    ITEM_OCARINA, ITEM_FEATHER, ITEM_SHOVEL, ITEM_POWDER,
    ITEM_BOOMERANG,
]
NUM_ITEMS = len(ALL_ITEMS)

# ===========================
# KEY ITEMS / FLAGS
# ===========================
FLIPPERS = 0xDB0C            # 01 = have
POTION = 0xDB0D              # 01 = have
TRADING_ITEM = 0xDB0E        # 0x00=none, 0x01=Yoshi doll ... 0x0E=magnifying glass
SECRET_SHELLS = 0xDB0F       # Quantidade
GOLDEN_LEAVES = 0xDB15       # Quantidade (vira Slime Key com >= 6)

TRADING_SEQUENCE_MAX = 0x0E  # Magnifying glass

# Dungeon entrance keys (5 keys: Tail, Angler, Face, Bird, Slime)
DUNGEON_KEY_START = 0xDB10
DUNGEON_KEY_END = 0xDB14
NUM_DUNGEON_KEYS = 5

# ===========================
# NÍVEIS DE EQUIPAMENTO
# ===========================
BRACELET_LEVEL = 0xDB43       # 1 or 2
SHIELD_LEVEL = 0xDB44         # 1, 2, or 3
ARROW_COUNT = 0xDB45
OCARINA_SONGS = 0xDB49        # 3-bit mask (bit0=Frog, bit1=Mambo, bit2=Ballad)
OCARINA_SELECTED = 0xDB4A
MUSHROOM_FLAG = 0xDB4B        # If != 0, powder becomes mushroom
POWDER_COUNT = 0xDB4C
BOMB_COUNT = 0xDB4D
SWORD_LEVEL = 0xDB4E          # 1 or 2 (2 = shoots beam at full health)

# ===========================
# CAPACIDADE MÁXIMA
# ===========================
MAX_POWDER = 0xDB76
MAX_BOMBS = 0xDB77
MAX_ARROWS = 0xDB78

# ===========================
# SAÚDE
# ===========================
CURRENT_HEALTH = 0xDB5A       # 0x08 per full heart, 0x04 per half heart
MAX_HEALTH = 0xDB5B           # Number of hearts (max 0x0E = 14)

# ===========================
# MORTES
# ===========================
DEATH_COUNT_SLOT1 = 0xDB56
DEATH_COUNT_SLOT2 = 0xDB57
DEATH_COUNT_SLOT3 = 0xDB58

# ===========================
# RUPEES (BCD format)
# ===========================
RUPEES_HIGH = 0xDB5D
RUPEES_LOW = 0xDB5E

# ===========================
# INSTRUMENTOS DOS DUNGEONS (Progresso Principal)
# 8 instrumentos, um por dungeon
# 0x00 = não tem, 0x03 = coletado
# ===========================
INSTRUMENTS_START = 0xDB65
INSTRUMENTS_END = 0xDB6C
NUM_INSTRUMENTS = 8
INSTRUMENT_COLLECTED = 0x03

# ===========================
# ITENS DO DUNGEON ATUAL
# ===========================
DUNGEON_MAP_FLAG = 0xDBCC
DUNGEON_COMPASS = 0xDBCD
DUNGEON_OWL_BEAK = 0xDBCE
DUNGEON_NIGHTMARE_KEY = 0xDBCF
DUNGEON_SMALL_KEYS = 0xDBD0

# Dungeon item flags — 5 bytes per dungeon (8 dungeons = 40 bytes)
DUNGEON_FLAGS_START = 0xDB16
DUNGEON_FLAGS_END = 0xDB3D
DUNGEON_FLAGS_BYTES_PER = 5

# ===========================
# COMBAT / POWER-UPS
# ===========================
PIECE_OF_POWER_KILLS = 0xD415
GUARDIAN_ACORN_KILLS = 0xD471
PIECE_OF_POWER_ACTIVE = 0xD47C   # 0x01 = active

# ===========================
# ENTIDADES / OBJETOS (Bank 0, 16 slots)
# Slot 0 geralmente é reservado/link-related
# ===========================
ENTITY_X_TABLE = 0xC200       # 16 bytes
ENTITY_Y_TABLE = 0xC210       # 16 bytes
ENTITY_STATE_TABLE = 0xC290   # 16 bytes
ENTITY_DIR_TABLE = 0xC380     # 16 bytes

# ===========================
# TOGGLE BLOCKS
# ===========================
TOGGLE_BLOCK_STATE = 0xD6FB

# ===========================
# MISC
# ===========================
PHOTO_FLAGS = 0xDC0C          # Bitset of photos obtained (2 bytes, DC0C-DC0D)
TUNIC_COLOR = 0xDC0F

# ===========================
# OVERWORLD MAP HELPERS
# ===========================
OVERWORLD_WIDTH = 16
OVERWORLD_HEIGHT = 16

def room_to_overworld_coords(room_number: int) -> tuple[int, int]:
    """Converte room number (0x00-0xFF) em coordenadas (col, row) do overworld."""
    return room_number % OVERWORLD_WIDTH, room_number // OVERWORLD_WIDTH

def overworld_coords_to_room(col: int, row: int) -> int:
    """Converte coordenadas do overworld em room number."""
    return row * OVERWORLD_WIDTH + col
