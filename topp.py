from neuralnet import NeuralNet


class Tournament:
    def __init__(self, nr_topp_games, checkpoint_path, M, episodes_per_game, game,
                 lrate=0.01, optimizer='SGD', loss='categorical_crossentropy', in_shape=(2,),
                 nn_dims=(1024, 512, 32, 1), hidden_act_func='relu'
                 ):
        self.NR_TOPP_GAMES = nr_topp_games
        self.CHECKPOINT_PATH = checkpoint_path
        self.M = M
        self.EPISODES_PER_GAME = episodes_per_game
        self.models = []
        self.game = game

    def load_models(self):
        for i in range(self.M + 1):
            number = i * self.EPISODES_PER_GAME
            model = NeuralNet(checkpoint_path=self.CHECKPOINT_PATH,
                              episodes_per_game=self.EPISODES_PER_GAME,
                              label=str(number))
            model.load_weights(number)
            self.models.append(model)

    def play_tournament(self):
        combinations = [(a, b) for idx, a in enumerate(self.models) for b in self.models[idx + 1:]]
        scores = {}

        for pair in combinations:
            model1 = pair[0]
            model2 = pair[1]
            score1 = 0
            score2 = 0

            for game_i in range(self.NR_TOPP_GAMES):
                game = self.game.create_copy()

