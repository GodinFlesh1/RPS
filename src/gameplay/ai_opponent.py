import numpy as np

class AIOpponent:
    def __init__(self):
        self.player_history = []

    def update_history(self, player_move):
        # Only learn real gestures (0,1,2)
        if player_move in [0, 1, 2]:
            self.player_history.append(player_move)

    def predict_player_move(self):
        if len(self.player_history) < 3:
            return np.random.choice([0, 1, 2])

        last_moves = self.player_history[-5:]
        counts = np.bincount(last_moves, minlength=3)
        return np.argmax(counts)

    def choose_move(self):
        predicted = self.predict_player_move()
        counter_moves = {0: 1, 1: 2, 2: 0}
        return counter_moves[predicted]
