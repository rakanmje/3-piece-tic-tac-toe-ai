# 3-Piece Tic-Tac-Toe AI

An AI for a dynamic variant of Tic-Tac-Toe in which **each player can have only three pieces on the board**. Placing a fourth piece automatically removes that player's oldest piece, so the board keeps changing and games cannot end in a full-board draw.

Two search agents are implemented and compared:

- **Alpha-Beta Pruning**: depth-limited Minimax with alpha-beta cut-offs and a heuristic evaluation function.
- **Monte Carlo Tree Search (MCTS)**: UCB1-guided search with random rollouts and repeated-state detection.

The game can be played in the terminal or in a Tkinter GUI, with any combination of human and AI players, and includes built-in experiments for win rates and move timing.

> University course project for **Artificial Intelligence** at Imam Mohammad Ibn Saud Islamic University.

---

## Game Rules

- Standard 3 × 3 board, cells numbered 0–8. X always moves first.
- A player wins by getting three of their pieces in a row, column, or diagonal.
- Each player has at most **three pieces** on the board. When a player places a fourth, their oldest piece is removed first (FIFO).
- In the GUI, the piece that will be removed next is shown in **gray**.

```
  0 | 1 | 2
  ---------
  3 | 4 | 5
  ---------
  6 | 7 | 8
```

---

## How It Works

### State representation

The board is a list of 9 cells (`" "`, `"X"` or `"O"`). A separate *trail* dictionary stores the order in which each player placed their pieces, which is what makes the FIFO removal possible. Legal moves are simply the empty cells, and a terminal state is any board where one of the 8 winning lines belongs to a single player.

### Alpha-Beta Pruning

- Depth-limited Minimax with alpha-beta pruning, tested at depths **k = 2, 5 and 10**. The depth limit is required because the 3-piece rule means a game has no fixed length.
- Wins and losses score **+100 / −100**.
- Non-terminal positions are scored over all 8 lines: a line containing only the AI's pieces adds 10ⁿ (n = number of pieces), and a line containing only the opponent's pieces subtracts 10ⁿ. Lines held by both players count as zero.

### Monte Carlo Tree Search

Each iteration runs the four standard MCTS phases:

1. **Selection**: descend the tree using UCB1 with exploration constant c = 1.41.
2. **Expansion**: add a child node for one untried move.
3. **Simulation**: play random moves until someone wins, a state repeats, or 100 moves pass.
4. **Backpropagation**: update visit and win counts up to the root.

The move with the most visits is played. Tested with **200, 500 and 1000** iterations.

### Interfaces

- **Terminal mode**: choose an agent for each side and watch or play the game move by move.
- **GUI mode**: Tkinter board where the AI runs in a background thread, so the window stays responsive while it thinks.
- **Experiment mode**: automated AI-vs-AI match-ups and computation-time measurements.

---

## Results

Win counts over **100 games per match-up**:

| Match-up | X wins | O wins | Draws |
|---|---:|---:|---:|
| AB k=2 (X) vs AB k=5 (O) | 0 | 100 | 0 |
| AB k=2 (X) vs AB k=10 (O) | 0 | 100 | 0 |
| AB k=10 (X) vs AB k=5 (O) | 100 | 0 | 0 |
| AB k=2 (X) vs MCTS 200 (O) | 100 | 0 | 0 |
| MCTS 1000 (X) vs AB k=2 (O) | 3 | 97 | 0 |
| AB k=10 (X) vs MCTS 1000 (O) | 100 | 0 | 0 |

Alpha-Beta is deterministic, so Alpha-Beta vs Alpha-Beta match-ups produce the same game every time. Variation only appears when MCTS is involved.

Time to choose the first move:

| Agent | Configuration | First move time |
|---|---|---:|
| Alpha-Beta | k = 2 | ≈ 0.0000 s |
| Alpha-Beta | k = 5 | 0.1825 s |
| Alpha-Beta | k = 10 | 6.3246 s |
| MCTS | 1000 iterations | 0.1454 s |

Timings depend on the machine; the relative cost is what matters. Each extra level of Alpha-Beta depth multiplies the work, while MCTS cost grows linearly with the number of iterations.

### Takeaways

Ranked from strongest to weakest: **AB k=10 > AB k=5 > AB k=2 > MCTS**. With a well-designed heuristic, Alpha-Beta clearly outperforms MCTS on a board this small, and depth 5 gives the best balance between move quality and speed. MCTS relies on random rollouts, which carry little information in this game; a smarter rollout policy would be the natural way to improve it.

---

## Requirements

- **Python 3.8+**
- **Tkinter**, only for the GUI mode (terminal mode and experiments work without it)

Tkinter ships with Python on Windows and macOS. On Linux it may need a separate package:

```bash
sudo pacman -S tk              # Arch / Manjaro
sudo apt install python3-tk    # Debian / Ubuntu
```

No third-party packages are needed.

---

## Usage

```bash
python tictactoe_ai.py
```

```
  Main Menu:
    1. Play a game in Terminal
    2. Play a game in GUI
    3. Run experiment: AI vs AI
    4. Run experiment: Computation Time
    0. Exit
```

Available agents: Human, Alpha-Beta (k = 2, 5, 10), and MCTS (200, 500, 1000 iterations).

The number of games per match-up in the AI vs AI experiment is set by `GAMES_PER_MATCHUP` at the top of the file. It defaults to 10 for a quick run; set it to 100 to reproduce the results above. Match-ups that involve Alpha-Beta k=10 take noticeably longer.

---

## Team

- Tami Alotaibi
- Abdullah Alwohibi
- Rakan Majed Alharbi
- Nawaf Sami Alzomaia
