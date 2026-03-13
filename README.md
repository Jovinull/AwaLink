# Zelda: Link's Awakening — Reinforcement Learning Agent

Agente de RL treinado com **PPO (Proximal Policy Optimization)** para jogar **The Legend of Zelda: Link's Awakening** no Game Boy, usando o emulador **PyBoy**.

## Arquitetura

| Componente | Tecnologia |
|---|---|
| Algoritmo RL | PPO (Stable-Baselines3) |
| Emulador | PyBoy (Game Boy) |
| Observação | MultiInputPolicy (Dict) — tela + estado de jogo |
| Normalização | VecNormalize (obs + reward) |
| Monitoramento | TensorBoard |

## Observação do Agente

O agente recebe uma observação multimodal rica:

- **screens** (72×80×4) — 4 frames empilhados em escala de cinza
- **health** — fração de HP + max hearts normalizado
- **position** — Fourier encoding de (x, y, room)
- **inventory** — vetor binário de 13 itens
- **equipment** — níveis de sword/shield/bracelet + flags
- **instruments** — 8 instrumentos dos dungeons
- **overworld_map** — mapa 16×16 de exploração do overworld
- **dungeon_state** — estado do dungeon atual
- **held_items** — itens equipados nos botões A e B
- **game_progress** — shells, leaves, trading, rupees, songs, kills
- **recent_actions** — últimas 4 ações

## Sistema de Recompensas

Reward shaping hierárquico com delta-reward e sinais densos de aprendizado:

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
| 13 | Transição de tela | +0.15 |
| 14 | Recuperar HP | +0.5 por fração de HP |
| 15 | Rupees | +0.05 por rupee |
| 16 | Kill counter | +0.3 |
| - | Time penalty base | -0.0003 |
| - | Inatividade | -0.002 |
| - | Morte | -3.0 |
| - | Perder HP | -0.5 por fração de HP |
| - | Stuck (mesmo lugar) | -0.15 |

Bônus intrínseco de exploração baseado em contagem: `0.02 / sqrt(visitas)` (Bellemare et al., 2016).

### Filosofia da V2

- O agente precisa receber sinais úteis com frequência, não apenas quando encontra itens raros.
- Rewards densos como `rupees`, `hp_recovery` e `screen_transition` ensinam combate, coleta e movimentação.
- Milestones como `instrumentos`, `novos itens` e `heart containers` continuam com peso alto para direcionar o objetivo final.
- Mortes agora são detectadas em tempo real via `HP == 0`, sem depender do contador persistente do save file.

## Técnicas Avançadas

- **VecNormalize**: normalização de observações e rewards para estabilidade
- **Count-based exploration**: bônus intrínseco para estados pouco visitados
- **Time penalty adaptativo**: penalidade baixa ao se mover e maior ao ficar parado
- **Delta-reward**: recompensa = diferença entre estado atual e anterior
- **GAE (λ=0.95)**: Generalized Advantage Estimation para menor variância
- **Linear LR decay**: learning rate decai linearmente até zero
- **High gamma (0.998)**: desconto alto para objetivos de longo prazo
- **4 frame stacks**: captura dinâmica de combate em tempo real
- **Reward denso**: rupees, recuperação de HP e transições de tela ajudam o agente a aprender mais cedo

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
python train.py --fresh              # Treino do zero
python train.py --timesteps 5000000  # Definir timesteps
python train.py --num-envs 4         # Número de processos
```

Se você alterar significativamente o sistema de rewards, use:

```bash
python train.py --fresh
```

Isso evita continuar um modelo treinado com uma função de recompensa antiga.

### 5. Monitorar com TensorBoard

O TensorBoard deve ser usado principalmente **durante o treino**. Ele lê os logs gerados pelo `train.py` em tempo real, então você pode deixar o treinamento rodando em um terminal e o TensorBoard em outro.

```bash
tensorboard --logdir logs/
```

Métricas disponíveis:
- `zelda/instruments` — instrumentos coletados (progresso principal)
- `zelda/screens_explored` — telas do overworld visitadas
- `zelda/screen_transitions` — trocas de tela, útil para validar movimento real
- `zelda/inventory_count` — itens no inventário
- `zelda/sword_level`, `zelda/shield_level` — níveis de equipamento
- `zelda/deaths` — mortes
- `zelda/kills` — kills detectadas
- `zelda/total_reward` — recompensa total
- `reward/rupees` — coleta de rupees
- `reward/hp_recovery` — recuperação de vida
- `reward/screen_transition` — reward por movimentação entre telas
- `reward/idle` — penalidade por inatividade
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
