# Zelda: Link's Awakening — Reinforcement Learning Agent

Agente de RL treinado com **PPO (Proximal Policy Optimization)** para jogar **The Legend of Zelda: Link's Awakening** no Game Boy, usando o emulador **PyBoy**.

## Arquitetura

| Componente | Tecnologia |
|---|---|
| Algoritmo RL | PPO (Stable-Baselines3) |
| Emulador | PyBoy (Game Boy) |
| Observação | MultiInputPolicy (Dict) — tela + estado de jogo |
| Normalização | VecNormalize (apenas observações escalares) |
| Monitoramento | TensorBoard |

## Observação do Agente (V3)

O agente recebe uma observação multimodal rica:

- **screens** (72×80×4) — 4 frames empilhados em escala de cinza
- **health** — fração de HP + max hearts normalizado
- **position** — Fourier encoding de (x, y, room)
- **inventory** — vetor binário de 13 itens
- **equipment** — níveis de sword/shield/bracelet + flags
- **instruments** — 8 instrumentos dos dungeons
- **overworld_map** — mapa 16×16 de exploração do overworld
- **dungeon_state** — estado do dungeon atual (8 features)
- **held_items** — itens equipados nos botões A e B
- **game_progress** — shells, leaves, trading, rupees, songs, kills
- **combat_info** *(V3)* — entidades ativas na tela, power-up, direção e terreno do Link
- **ammo** *(V3)* — bombas/flechas/pó normalizados pela capacidade máxima
- **recent_actions** — últimas 4 ações

## Sistema de Recompensas (V3)

Reward shaping hierárquico com delta-reward e sinais densos:

| Prioridade | Componente | Reward |
|---|---|---|
| 1 | Instrumento de dungeon | +50.0 |
| 2 | Novo item no inventário | +10.0 |
| 3 | Heart container | +8.0 |
| 4 | Upgrade de equipamento | +5.0 |
| 5 | Dungeon entrance key | +3.0 |
| 6 | Trading sequence step | +3.0 |
| 7 | Nova tela do overworld | +3.0 |
| 8 | Nova sala de dungeon | +2.0 |
| 9 | Dungeon item (mapa/bússola) | +2.0 |
| 10 | Golden leaf | +2.0 |
| 11 | Small key | +1.5 |
| 12 | Secret shell | +1.0 |
| 13 | Dungeon flag bit (progresso) | +0.5 |
| 14 | Interação com o mundo (tiles) | +0.3 |
| 15 | Kill counter | +0.3 |
| 16 | Transição de tela (anti-osc.) | +0.15 |
| 17 | Recuperar HP | +0.5 × fração |
| 18 | Rupees | +0.05 × quantidade |
| - | Time penalty base | -0.0003 |
| - | Inatividade | -0.002 |
| - | Morte | -3.0 |
| - | Perder HP | -0.5 × fração |
| - | Stuck (mesmo lugar) | -0.15 |

Bônus intrínseco: `0.02 / sqrt(visitas)`, cortado após 100 visitas (Bellemare et al., 2016).

### Filosofia da V3

**Episódios curtos** (~30k steps ≈ 8 min de jogo) para que o agente veja muito mais resets ao longo do treinamento. Com 20M steps e 8 envs, cada env faz ~80 episódios em vez de ~25.

**Anti-oscillation**: transições de tela só são recompensadas se a room destino não está nas últimas 3 rooms visitadas. Impede o agente de exploitar A→B→A→B infinitamente.

**Auto Game Over**: quando o agente morre (HP=0), após a animação de morte o environment automaticamente pressiona A para navegar o menu de Game Over e selecionar "Continue". Economiza ~200 steps por morte.

**World Interaction**: monitora mudanças nos tile data carregados (D700-D79B). Quando tiles mudam (abrir baú, cortar grama, matar inimigo na tela), o agente recebe +0.3.

**Dungeon Flags**: monitora os bits em DB16-DB3D que representam progresso nos dungeons (baús, portas, switches). Cada novo bit setado = +0.5.

**Sem VecNormalize reward**: a normalização de reward com clipping destruía a hierarquia (instrumento +50 era clipado ao mesmo valor que 5 rupees). Agora os rewards são manuais e preservam a escala relativa.

**Combat awareness**: o agente agora vê quantas entidades estão ativas na tela, se Piece of Power está ativo, e seu estado de terreno, permitindo decisões de combate informadas.

## Técnicas Avançadas

- **VecNormalize seletivo**: normaliza apenas observações escalares (health, position, etc.); screens e overworld_map mantêm valores brutos (CombinedExtractor divide por 255 automaticamente)
- **Count-based exploration** com cutoff (Bellemare et al.): bônus intrínseco para estados pouco visitados, cortado após 100 visitas para reduzir ruído
- **Anti-oscillation cooldown**: evita exploiting de screen transitions repetidas
- **Auto Game Over navigation**: reduz tempo perdido em menus pós-morte
- **World interaction detection**: hash dos tile data como sinal de interação com o ambiente
- **Dungeon flag tracking**: progresso granular em dungeons via bitflags
- **BCD parsing robusto**: decodificação de valores BCD (rupees, ammo) com clamp de dígitos inválidos
- **N_EPOCHS=2**: reduz overfitting no rollout com reward ruidoso
- **Delta-reward**: recompensa = diferença entre estado atual e anterior
- **GAE (λ=0.95)**: Generalized Advantage Estimation para menor variância
- **High gamma (0.998)**: desconto alto para objetivos de longo prazo
- **4 frame stacks**: captura dinâmica de combate em tempo real

## Setup

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. ROM

Coloque a ROM `zelda.gb` (The Legend of Zelda: Link's Awakening, Game Boy) na raiz do projeto.

### 3. Criar save state inicial

```bash
python play.py --create-save
```

Jogue até um ponto inicial bom (recomendado: após pegar a espada na praia). Feche a janela para salvar.

### 4. Treinar

```bash
python train.py
```

Opções:
```bash
python train.py --fresh              # Treino do zero (OBRIGATÓRIO após mudanças V3)
python train.py --timesteps 5000000  # Definir timesteps
python train.py --num-envs 4         # Número de processos
```

> **IMPORTANTE**: Como V3 muda observação e rewards, use `--fresh` na primeira execução após atualizar!

### 5. Monitorar com TensorBoard

O TensorBoard deve ser usado **durante o treino**. Ele lê os logs gerados pelo `train.py` em tempo real. Deixe o treinamento rodando em um terminal e o TensorBoard em outro.

```bash
tensorboard --logdir logs/
```

Acesse `http://localhost:6006` no navegador.

Métricas disponíveis:
- `zelda/instruments` — instrumentos coletados (progresso principal)
- `zelda/screens_explored` — telas do overworld visitadas
- `zelda/screen_transitions` — trocas de tela
- `zelda/world_interactions` — interações com tiles do mapa
- `zelda/dungeon_flag_bits` — progresso em dungeons
- `zelda/inventory_count` — itens no inventário
- `zelda/entities` — entidades ativas na tela
- `zelda/deaths`, `zelda/kills` — mortes e kills
- `zelda/total_reward` — recompensa total
- `reward/*` — componentes individuais de reward

### 6. Assistir o agente jogar

```bash
python play.py                         # Melhor modelo
python play.py --model models/xyz.zip  # Modelo específico
```

## Estrutura do Projeto

```
├── config.py          # Hiperparâmetros centralizados
├── memory_map.py      # Endereços RAM do Zelda: Link's Awakening
├── environment.py     # Gymnasium Environment (ZeldaLinksAwakeningEnv)
├── global_map.py      # Sistema de mapa global (overworld 16x16 + dungeon)
├── train.py           # Treinamento paralelo com PPO
├── play.py            # Assistir agente / criar save state
├── requirements.txt   # Dependências Python
└── README.md          # Este arquivo
```

## Referências

- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Bellemare et al. (2016) — *Unifying Count-Based Exploration and Intrinsic Motivation*
- Burda et al. (2019) — *Exploration by Random Network Distillation*
- [PyBoy](https://github.com/Baekalfen/PyBoy) — Emulador Game Boy em Python
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — RL Framework
- [LADX Disassembly](https://github.com/zladx/LADX-Disassembly) — RAM map do Link's Awakening
