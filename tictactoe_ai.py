"""
3-Piece Tic-Tac-Toe AI.

A Tic-Tac-Toe variant where each player keeps at most three pieces on the
board: placing a fourth piece removes that player's oldest one (FIFO).
Two AI agents are implemented, Alpha-Beta pruning with a heuristic and
Monte Carlo Tree Search (UCB1), and can be played against each other or
against a human in the terminal or a Tkinter GUI.

Run:  python tictactoe_ai.py
"""

import math
import random
import threading
import time

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:  # GUI is optional; terminal mode and experiments still work
    tk = None

GAMES_PER_MATCHUP = 10  # games played for each match-up in the AI vs AI experiments

WINNING_COMBOS = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6],
]

def fresh_grid():
    return [" "] * 9

def open_squares(grid):
    return [i for i in range(9) if grid[i] == " "]

def has_won(grid, mark):
    return any(all(grid[i] == mark for i in combo) for combo in WINNING_COMBOS)

def render_grid(grid):
    print()
    print(f"  {grid[0]} | {grid[1]} | {grid[2]}     0 | 1 | 2")
    print("  ---------     ---------")
    print(f"  {grid[3]} | {grid[4]} | {grid[5]}     3 | 4 | 5")
    print("  ---------     ---------")
    print(f"  {grid[6]} | {grid[7]} | {grid[8]}     6 | 7 | 8")
    print()

def fresh_trail():
    return {"X": [], "O": []}

def place(grid, trail, mark, square):
    if len(trail[mark]) == 3:
        grid[trail[mark].pop(0)] = " "
    grid[square] = mark
    trail[mark].append(square)
    return has_won(grid, mark)

def score_grid(grid, me):
    rival = "O" if me == "X" else "X"
    if has_won(grid, me):   return  100
    if has_won(grid, rival): return -100
    total = 0
    for combo in WINNING_COMBOS:
        mine = sum(grid[i] == me    for i in combo)
        theirs = sum(grid[i] == rival for i in combo)
        if theirs == 0: total += 10 ** mine
        if mine   == 0: total -= 10 ** theirs
    return total

def minimax(grid, trail, depth, alpha, beta, maximising, me):
    rival = "O" if me == "X" else "X"
    if has_won(grid, me):    return  100
    if has_won(grid, rival): return -100
    if depth == 0:           return score_grid(grid, me)

    mover = me if maximising else rival
    best  = -math.inf if maximising else math.inf

    for sq in open_squares(grid):
        g2 = grid[:]
        t2 = {"X": trail["X"][:], "O": trail["O"][:]}
        place(g2, t2, mover, sq)
        val = minimax(g2, t2, depth-1, alpha, beta, not maximising, me)
        if maximising:
            best  = max(best, val);  alpha = max(alpha, best)
        else:
            best  = min(best, val);  beta  = min(beta,  best)
        if alpha >= beta: break

    return best

def best_ab_move(grid, trail, mark, depth):
    top_score, top_sq = -math.inf, None
    for sq in open_squares(grid):
        g2 = grid[:]
        t2 = {"X": trail["X"][:], "O": trail["O"][:]}
        place(g2, t2, mark, sq)
        s = minimax(g2, t2, depth-1, -math.inf, math.inf, False, mark)
        if s > top_score:
            top_score, top_sq = s, sq
    return top_sq


class TreeNode:
    def __init__(self, grid, trail, turn, parent=None, origin=None):
        self.grid, self.trail, self.turn = grid, trail, turn
        self.parent, self.origin = parent, origin
        self.wins = self.visits = 0
        self.untried = open_squares(grid)
        self.kids = []

def ucb(node, parent_visits, c=1.41):
    if node.visits == 0: return float("inf")
    return node.wins/node.visits + c*math.sqrt(math.log(parent_visits)/node.visits)

def best_mcts_move(grid, trail, mark, iterations=500):
    root = TreeNode(grid[:], {"X": trail["X"][:], "O": trail["O"][:]}, mark)

    for _ in range(iterations):
        node = root
        while not node.untried and node.kids:
            node = max(node.kids, key=lambda k: ucb(k, node.visits))

        terminal = None
        if node.untried:
            sq = random.choice(node.untried)
            node.untried.remove(sq)
            g2 = node.grid[:]
            t2 = {"X": node.trail["X"][:], "O": node.trail["O"][:]}
            mover = node.turn
            won = place(g2, t2, mover, sq)
            nxt = "O" if mover == "X" else "X"
            child = TreeNode(g2, t2, nxt, parent=node, origin=sq)
            node.kids.append(child)
            node = child
            if won:
                terminal = 1 if mover == mark else -1

        if terminal is None:
            sg, st, sp = node.grid[:], {"X": node.trail["X"][:], "O": node.trail["O"][:]}, node.turn
            seen = {}
            for _ in range(100):
                key = (tuple(sg), tuple(st["X"]), tuple(st["O"]))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 1: break
                avail = open_squares(sg)
                if not avail: break
                won = place(sg, st, sp, random.choice(avail))
                if won: break
                sp = "O" if sp == "X" else "X"
            rival = "O" if mark == "X" else "X"
            terminal = 1 if has_won(sg, mark) else (-1 if has_won(sg, rival) else 0)

        n = node
        while n:
            n.visits += 1
            n.wins += terminal if n.turn != mark else -terminal
            n = n.parent

    return max(root.kids, key=lambda k: k.visits).origin


def request_move(grid, trail, mark, agent):
    if agent == "human":
        while True:
            try:
                sq = int(input(f"  Your move ({mark}) – enter cell 0-8: "))
                if sq in open_squares(grid): return sq
                print("  Taken or invalid.")
            except ValueError:
                print("  Enter a number 0-8.")
    elif agent.startswith("ab"):
        return best_ab_move(grid, trail, mark, int(agent[2:]))
    elif agent.startswith("mcts"):
        return best_mcts_move(grid, trail, mark, int(agent[4:]))
    else:
        raise ValueError(f"Unknown agent: {agent}")


def conduct_match(ax, ao, show=True):
    grid, trail = fresh_grid(), fresh_trail()
    roster = {"X": ax, "O": ao}
    if show:
        print(f"\n{'='*40}\n  X: {ax.upper()}   vs   O: {ao.upper()}\n{'='*40}")
        render_grid(grid)

    deadline = time.time() + 1200
    turn_idx = 0

    while True:
        if time.time() > deadline:
            if show: print("  Time limit! Draw.")
            return "Draw"
        mark  = "XO"[turn_idx]
        agent = roster[mark]
        t0    = time.time()
        sq    = request_move(grid, trail, mark, agent)
        won   = place(grid, trail, mark, sq)
        if show:
            print(f"  {agent.upper()} ({mark}) → cell {sq}  [{time.time()-t0:.3f}s]")
            render_grid(grid)
        if won:
            if show: print(f"  *** {mark} wins! ***\n")
            return mark
        turn_idx ^= 1


def batch_matches(ax, ao, count):
    tally = {"X": 0, "O": 0, "Draw": 0}
    print(f"\n  Running {count} games: {ax.upper()} (X) vs {ao.upper()} (O)")
    for g in range(count):
        r = conduct_match(ax, ao, show=False)
        tally[r] += 1
        print(f"    Game {g+1}: {r}", end="\r")
    print(f"    Results: X wins={tally['X']}  O wins={tally['O']}  Draws={tally['Draw']}    ")
    return tally


def play_experiments():
    N = GAMES_PER_MATCHUP
    print("\n" + "#"*50 + "\n  PLAY EXPERIMENT 1: Alpha-Beta vs Alpha-Beta\n" + "#"*50)
    for a, b in [("ab2","ab5"),("ab2","ab10"),("ab10","ab5")]:
        batch_matches(a, b, N)

    print("\n" + "#"*50 + "\n  PLAY EXPERIMENT 2: Alpha-Beta vs MCTS\n" + "#"*50)
    for a, b in [("ab2","mcts200"),("mcts1000","ab2"),("ab10","mcts1000"),("mcts500","ab5")]:
        batch_matches(a, b, N)

    print("\n" + "#"*50 + "\n  PLAY EXPERIMENT 3: MCTS vs MCTS\n" + "#"*50)
    for a, b in [("mcts500","mcts200"),("mcts1000","mcts200"),("mcts500","mcts1000")]:
        batch_matches(a, b, N)
    print("\n" + "#"*50)


def timing_experiments():
    print("\n" + "#"*50 + "\n  TIME EXPERIMENT 1: First Move Timing\n" + "#"*50)
    print("\n  Alpha-Beta timing per depth:")
    for d in [2, 5, 10]:
        g, t = fresh_grid(), fresh_trail()
        t0 = time.time()
        best_ab_move(g, t, "X", d)
        print(f"    AB depth={d}:  {time.time()-t0:.4f}s")

    print("\n  MCTS timing per simulation count:")
    for s in [200, 500, 1000]:
        g, t = fresh_grid(), fresh_trail()
        t0 = time.time()
        best_mcts_move(g, t, "X", s)
        print(f"    MCTS sims={s}:  {time.time()-t0:.4f}s")

    print("\n" + "#"*50 + "\n  TIME EXPERIMENT 2: Average Move Timing\n" + "#"*50)
    print("\n  Alpha-Beta average time per move:")
    for d in [2, 5, 10]:
        g, t, p, times = fresh_grid(), fresh_trail(), "X", []
        for _ in range(200):
            if not open_squares(g): break
            t0 = time.time()
            sq = best_ab_move(g, t, p, d)
            times.append(time.time()-t0)
            if place(g, t, p, sq): break
            p = "O" if p == "X" else "X"
        print(f"    AB depth={d}:  {sum(times)/len(times):.4f}s avg  ({len(times)} moves)")

    print("\n  MCTS average time per move:")
    for s in [200, 500, 1000]:
        g, t, p, times = fresh_grid(), fresh_trail(), "X", []
        for _ in range(200):
            if not open_squares(g): break
            t0 = time.time()
            sq = best_mcts_move(g, t, p, s)
            times.append(time.time()-t0)
            if place(g, t, p, sq): break
            p = "O" if p == "X" else "X"
        print(f"    MCTS sims={s}:  {sum(times)/len(times):.4f}s avg  ({len(times)} moves)")


class GameWindow:
    AGENT_MAP = {
        "Human":"human","AB k=2":"ab2","AB k=5":"ab5","AB k=10":"ab10",
        "MCTS 200":"mcts200","MCTS 500":"mcts500","MCTS 1000":"mcts1000",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("3-Piece Tic-Tac-Toe")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)
        self.grid_state = fresh_grid()
        self.move_trail = fresh_trail()
        self.active_mark = "X"
        self.agent_map = {"X": None, "O": None}
        self._active = False
        self._human_waiting = False
        self._human_sq = None
        self._click_evt = threading.Event()
        self._game_id = 0  # bumped on every Start so stale game threads stop
        self._build()

    def _build(self):
        tk.Label(self.root, text="3-Piece Tic-Tac-Toe", font=("Arial",16,"bold"),
                 bg="#f0f0f0").grid(row=0, column=0, columnspan=3, pady=(15,5))

        bar = tk.Frame(self.root, bg="#f0f0f0")
        bar.grid(row=1, column=0, columnspan=3, pady=5)
        opts = list(self.AGENT_MAP.keys())

        tk.Label(bar, text="X:", font=("Arial",11,"bold"), bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self._xv = tk.StringVar(value="Human")
        ttk.Combobox(bar, textvariable=self._xv, values=opts, state="readonly",
                     width=12).grid(row=0, column=1, padx=5)

        tk.Label(bar, text="O:", font=("Arial",11,"bold"), bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self._ov = tk.StringVar(value="AB k=5")
        ttk.Combobox(bar, textvariable=self._ov, values=opts, state="readonly",
                     width=12).grid(row=0, column=3, padx=5)

        tk.Button(bar, text="Start", font=("Arial",11,"bold"), bg="#4CAF50", fg="white",
                  relief="flat", padx=10, command=self._launch).grid(row=0, column=4, padx=10)

        self.cells = []
        for i in range(9):
            b = tk.Button(self.root, text="", font=("Arial",22,"bold"), height=2, width=5,
                          bg="white", relief="groove", command=lambda i=i: self._clicked(i))
            b.grid(row=2+i//3, column=i%3, padx=4, pady=4)
            self.cells.append(b)

        self._msg = tk.StringVar(value="Pick agents and press Start.")
        tk.Label(self.root, textvariable=self._msg, font=("Arial",11),
                 bg="#f0f0f0").grid(row=5, column=0, columnspan=3, pady=6)
        tk.Button(self.root, text="Reset", font=("Arial",10),
                  command=self._reset).grid(row=6, column=0, columnspan=3, pady=(0,12))

    def _launch(self):
        self._reset()
        self.agent_map["X"] = self.AGENT_MAP[self._xv.get()]
        self.agent_map["O"] = self.AGENT_MAP[self._ov.get()]
        self._game_id += 1
        self._active = True
        self._msg.set(f"Player X's turn  ({self._xv.get()})")
        threading.Thread(target=self._loop, args=(self._game_id,), daemon=True).start()

    def _loop(self, game_id):
        while self._active and game_id == self._game_id:
            m = self.active_mark
            a = self.agent_map[m]
            if a == "human":
                self._human_waiting = True
                self._click_evt.clear()
                self.root.after(0, lambda m=m: self._msg.set(f"Player {m}'s turn — click a cell."))
                self._click_evt.wait()
                self._human_waiting = False
                if not self._active or game_id != self._game_id: break
                sq = self._human_sq
            else:
                lbl = self._xv.get() if m == "X" else self._ov.get()
                self.root.after(0, lambda m=m, l=lbl: self._msg.set(f"Player {m}'s turn — {l} thinking…"))
                sq = request_move(self.grid_state, self.move_trail, m, a)
            if not self._active or game_id != self._game_id: break
            won = place(self.grid_state, self.move_trail, m, sq)
            self.root.after(0, self._redraw)
            if won:
                self.root.after(0, lambda m=m: self._finish(f"Player {m} wins!"))
                return
            self.active_mark = "O" if m == "X" else "X"

    def _finish(self, msg):
        self._active = False
        self._msg.set(msg)
        messagebox.showinfo("Game Over", msg)

    def _clicked(self, idx):
        if not self._human_waiting: return
        if self.grid_state[idx] != " ":
            self._msg.set("That cell is taken, pick another.")
            return
        self._human_sq = idx
        self._click_evt.set()

    def _redraw(self):
        for i in range(9):
            v = self.grid_state[i]
            if v == " ":
                self.cells[i].config(text="", bg="white", fg="black")
            else:
                h = self.move_trail[v]
                oldest = len(h) == 3 and h[0] == i
                fg = "gray" if oldest else ("blue" if v == "X" else "red")
                self.cells[i].config(text=v, fg=fg, bg="white")

    def _reset(self):
        self._active = False
        self._human_waiting = False
        self._click_evt.set()
        self.grid_state = fresh_grid()
        self.move_trail = fresh_trail()
        self.active_mark = "X"
        self._msg.set("Pick agents and press Start.")
        for b in self.cells:
            b.config(text="", bg="white", fg="black")


ROSTER = {
    "1": ("Human",           "human"),
    "2": ("Alpha-Beta k=2",  "ab2"),
    "3": ("Alpha-Beta k=5",  "ab5"),
    "4": ("Alpha-Beta k=10", "ab10"),
    "5": ("MCTS 200 sims",   "mcts200"),
    "6": ("MCTS 500 sims",   "mcts500"),
    "7": ("MCTS 1000 sims",  "mcts1000"),
}

def choose_agent(role):
    print(f"\n  Pick agent for {role}:")
    for k, (name, _) in ROSTER.items():
        print(f"    {k}. {name}")
    while True:
        c = input("  Enter number: ").strip()
        if c in ROSTER: return ROSTER[c][1]
        print("  Invalid, try again.")

def main():
    print("\n" + "="*50 + "\n   3-Piece Tic-Tac-Toe  \n" + "="*50)
    while True:
        print("\n  Main Menu:\n    1. Play a game in Terminal\n    2. Play a game in GUI")
        print("    3. Run experiment: AI vs AI\n    4. Run experiment: Computation Time\n    0. Exit")
        c = input("\n  Select: ").strip()
        if   c == "0": print("  Goodbye!"); break
        elif c == "1": conduct_match(choose_agent("X (goes first)"), choose_agent("O"))
        elif c == "2":
            if tk is None:
                print("  Tkinter is not installed, so the GUI is unavailable.")
            else:
                root = tk.Tk(); GameWindow(root); root.mainloop()
        elif c == "3": play_experiments()
        elif c == "4": timing_experiments()
        else:          print("  Invalid choice.")

if __name__ == "__main__":
    main()
