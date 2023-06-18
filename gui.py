import tkinter as tk
from collections import deque

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chess Game")
        self.listbox = tk.Listbox(self.root)
        self.listbox.pack()
        self.textbox = tk.Text(self.root)
        self.textbox.pack()
        self.games = []
        self.last_moves = deque(maxlen=10)
        self.last_moves_start_index = None

    def add_game(self, game):
        self.games.append(game)
        self.listbox.insert(tk.END, f'Game {len(self.games)}')
        self.listbox.bind('<<ListboxSelect>>', self.show_moves)
        self.root.update_idletasks()  # Force an update of the GUI


    def show_moves(self, event):
        selection = event.widget.curselection()
        if not selection:
            return
        game = self.games[selection[0]]
        self.textbox.delete(1.0, tk.END)
        for i, stats in enumerate(game['stats'], 1):
            move = stats['move']
            self.textbox.insert(tk.END, f'Move {i}: {move}\n')
            self.textbox.tag_bind(f'move{i}', '<Button-1>', lambda e, stats=stats: self.show_move_info(stats, game['from_mcts']))
            self.textbox.tag_add(f'move{i}', f'{i}.0', f'{i}.end')
        self.show_last_moves_info()

    def show_move_info(self, stats, from_mcts):
        move = stats['move']
        if from_mcts:
            info = f"\nMove: {stats['move']}, Visits: {stats['n_visits']}, Prior: {stats['P']}, Average eval: {stats['ave_eval']}\n"
        else:
            info = f"\nMove: {move}, Value: {stats['value']}\n"
        self.last_moves.append(info)
        # Delete last moves info if it exists
        if self.last_moves_start_index is not None:
            self.textbox.delete(self.last_moves_start_index, tk.END)
        # Save the current end index as the start index for the last moves info
        self.last_moves_start_index = self.textbox.index(tk.END)
        self.show_last_moves_info()

    def show_last_moves_info(self):
        # self.textbox.insert(tk.END, '\nLast 10 moves info:\n')
        for move_info in list(self.last_moves):
            self.textbox.insert(tk.END, f'{move_info}\n\n')

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = GUI()
    gui.add_game({
        'stats': [{'move': 'e4', 'value': 0.8},
                  {'move': 'e5', 'value': 0.15},
                  {'move': 'Nf3', 'value': 0.05}],
        'from_mcts': False
    })
    gui.add_game({
        'stats': [{'move': 'e4', 'n_visits': 100, 'P': 0.8, 'ave_eval': 0.85},
                  {'move': 'e5', 'n_visits': 90, 'P': 0.15, 'ave_eval': 0.7},
                  {'move': 'Nf3', 'n_visits': 110, 'P': 0.05, 'ave_eval': 0.9}],
        'from_mcts': True
    })
    gui.run()
