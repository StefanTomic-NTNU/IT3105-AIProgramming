import copy
import math
import random
import numpy as np
import time


class TreeNode:
    def __init__(self, state: dict):
        self.is_illegal = False
        self.parent = None
        self.score_a = []
        self.N_a = []
        self.Q_a = []
        self.edges = []
        self.children = []
        self.state = copy.copy(state)
        self.score = 0
        self.N = 1
        self.Q = 0

    def is_at_end(self):
        if not self.children:
            return False
        no_legal_moves = True
        for child in self.children:
            if not child.is_illegal:
                no_legal_moves = False
        return no_legal_moves


def add_child(parent, child, edge):
    if child not in parent.children:
        parent.children.append(child)
        parent.edges.append(edge)
        if not child.is_illegal:
            parent.N_a.append(1)
        else:
            parent.N_a.append(0)
        parent.score_a.append(0)
        parent.Q_a.append(0)
        child.parent = parent


def normalize(arr: np.array): return arr/np.sum(arr)


def generate_children(tree_node: TreeNode, game):
    if len(tree_node.children) == 0 and tree_node.state:
        edges, states, illegal_edges, illegal_states = game.generate_children_(tree_node.state)
        children = [TreeNode(child) for child in states]
        illegal_children = [TreeNode(illegal_child) for illegal_child in illegal_states]
        for i in range(len(children)):
            add_child(tree_node, children[i], edges[i])
        for j in range(len(illegal_children)):
            illegal_children[j].is_illegal = True
            illegal_children[j].N = 0
            add_child(tree_node, illegal_children[j], illegal_edges[j])


class MCTS:
    def __init__(self, number_actual_games, number_search_games, game, nr_actions, actor, search_time_limit=2):
        self.number_actual_games = number_actual_games
        self.number_search_games = number_search_games
        self.search_time_limit = search_time_limit
        self.game = game
        self.nr_actions = nr_actions
        self.actor = actor
        self.replay_buffer = []
        self.prob_disc_dict = {}
        self.rollout_time = 0

    def run(self):
        runtime_start = time.time()
        for g_a in range(self.number_actual_games):
            board_a = self.game.create_copy()
            root = TreeNode(board_a.get_state())
            episode_start_time = time.time()

            episode_rollout_time = 0
            tree_policy_loop_time = 0
            gen_children_time = 0
            while not board_a.is_game_over():
                board_mc = self.game.create_copy()
                board_mc.set_state(root.state)

                g_s_start_time = time.time()
                for g_s in range(self.number_search_games):
                    self.rollout_time = 0
                    board_mc = self.game.create_copy()
                    board_mc.set_state(root.state)

                    tree_policy_loop_time_start = time.time()

                    # TREE POLICY
                    node = root
                    node = self.pick_tree_node(node, board_mc)

                    tree_policy_loop_time_end = time.time()
                    tree_policy_loop_time += tree_policy_loop_time_end - tree_policy_loop_time_start

                    gen_children_time_start = time.time()
                    generate_children(node, self.game)    # Blue nodes
                    gen_children_time_end = time.time()
                    gen_children_time += gen_children_time_end - gen_children_time_start

                    # ROLLOUT
                    grey_node = node
                    single_rollout_start = time.time()
                    node = self.rollout(node, board_mc)
                    single_rollout_end = time.time()
                    episode_rollout_time += single_rollout_end - single_rollout_start

                    # BACKPROPAGATION
                    self.backpropagate(node, board_mc)

                    # Cleanup children:
                    if not grey_node.is_at_end():
                        for child in grey_node.children:
                            child.score_a = []
                            child.N_a = []
                            child.Q_a = []
                            child.edges = []
                            child.children = []

                    g_s_start_end = time.time()
                    if g_s_start_end - g_s_start_time > self.search_time_limit: break

                root_state = np.concatenate((np.ravel(root.state['board_state']), np.array([root.state['pid']],
                                                                                           dtype='float')))
                root_state = root_state.reshape(1, -1)
                D = copy.copy(normalize(np.array([root.N_a])))
                case = (root_state, D)
                self.replay_buffer.append(case)
                action, action_index = self.actor.pick_action(case[0], root)
                board_a.make_move(action)
                root = root.children[action_index]
                root.parent = None

            # TRAIN ACTOR
            training_time_start = time.time()
            if g_a % 1 == 0:
                batch_size = len(self.replay_buffer)
                number_from_batch = random.randrange(math.floor(batch_size/5), batch_size)
                if number_from_batch == 0:
                    number_from_batch = 1
                subbatch = random.sample(self.replay_buffer, number_from_batch)

                ex_batch_x = subbatch[0][0][0]
                ex_batch_y = subbatch[0][1][0]

                batch_x = np.zeros((number_from_batch, len(ex_batch_x)))
                for i in range(number_from_batch):
                    batch_x[i, :] = subbatch[i][0]
                batch_y = np.zeros((number_from_batch, len(ex_batch_y)))
                for i in range(number_from_batch):
                    batch_y[i, :] = subbatch[i][1]

                self.actor.fit(x=batch_x, y=batch_y)

            training_time_end = time.time()

            self.actor.decay_exploration_rate()
            episode_end_time = time.time()

            print(f'Episode {g_a+1}/{self.number_actual_games} '
                  f'time: {(episode_end_time - episode_start_time):.4f}s, '
                  f'\trollout-time: {episode_rollout_time:.4f}s, '
                  f'\ttraining-time: {(training_time_end - training_time_start):.4f}s, '
                  f'\tgen-children: {gen_children_time:.4f}s, '
                  f'\ttree-policy-time: {tree_policy_loop_time:.4f}s, ')

        # "OPTIMAL" GAME
        self.play_optimal_game()
        runtime_end = time.time()
        runtime = runtime_end - runtime_start
        minutes = np.floor_divide(runtime, 60)
        seconds = runtime % 60
        print(f'\nRuntime: {minutes:.0f}m, {seconds:.2f}s')

    def tree_policy(self, node: TreeNode):
        u = [1*np.sqrt(np.log(node.N)/(1 + N_sa)) for N_sa in node.N_a]

        if node.state['pid'] == 1:
            combined = np.add(node.Q_a, u)
            policy = np.argmax(combined)
        else:
            combined = np.subtract(node.Q_a, u)
            policy = np.argmin(combined)

        while node.children[policy].state is None:
            if node.state['pid'] == 1:
                combined[policy] = -100000
                policy = np.argmax(combined)
                if np.sum(combined) == -100000 * self.nr_actions:
                    return None
            else:
                combined[policy] = 100000
                policy = np.argmin(combined)
                if np.sum(combined) == 100000 * self.nr_actions:
                    return None
        return policy

    def pick_tree_node(self, node: TreeNode, board_mc):
        new_node = node
        while new_node.children and not new_node.is_at_end():
            chosen_node = self.tree_policy(new_node)
            if chosen_node is None: break
            if new_node.children[chosen_node].state is None: break
            board_mc.make_move(new_node.edges[chosen_node])
            new_node = new_node.children[chosen_node]
        return new_node

    def rollout(self, node: TreeNode, board_mc):
        new_node = node
        roll_gen_cld_sum = 0
        pick_action_sum = 0
        while new_node.state and not board_mc.is_game_over():
            roll_gen_cld_start = time.time_ns()
            generate_children(new_node, board_mc)
            roll_gen_cld_end = time.time_ns()

            pick_action_time_start = time.time_ns()
            nn_input = np.concatenate(
                (np.ravel(new_node.state['board_state']), np.array([new_node.state['pid']], dtype='float')))
            # nn_input = np.array([node.state['board_state'], node.state['pid']])
            nn_input = nn_input.reshape(1, -1)
            action, action_index = self.actor.pick_action(nn_input, new_node)
            pick_action_time_end = time.time_ns()
            roll_gen_cld_sum += roll_gen_cld_end - roll_gen_cld_start
            pick_action_sum += pick_action_time_end - pick_action_time_start
            if not board_mc.is_game_over():
                board_mc.make_move(action)
                new_node = new_node.children[action_index]
        self.rollout_time += roll_gen_cld_sum
        print(f'Roll gen cld: {roll_gen_cld_sum / 1_000_000_000}s \t Roll pick action {pick_action_sum / 1_000_000_000}s')
        return new_node

    def backpropagate(self, node: TreeNode, board_mc):
        new_node = node
        evaluation = -1 if board_mc.state['pid'] == 1 else 1
        parent = new_node.parent
        while parent:
            new_node.score += evaluation
            new_node.N += 1
            new_node.Q = new_node.score / new_node.N
            edge_index = new_node.parent.children.index(new_node)
            new_node.parent.score_a[edge_index] += evaluation
            new_node.parent.N_a[edge_index] += 1
            new_node.parent.Q_a[edge_index] = new_node.parent.score_a[edge_index] / new_node.parent.N_a[edge_index]
            new_node = new_node.parent
            parent = new_node.parent

    def play_optimal_game(self):
        self.actor.exploration_rate = 0
        for init_player in (1, 2):
            final_game = self.game.create_copy()
            final_game.state['pid'] = init_player
            print(init_player)
            while not final_game.is_game_over():
                final_game.render()
                node = TreeNode(final_game.state)
                generate_children(node, final_game)
                state = np.concatenate(
                    (np.ravel(final_game.state['board_state']), np.array([final_game.state['pid']], dtype='float')))
                state = state.reshape(1, -1)
                action, action_index = self.actor.pick_action(state, node)
                final_game.make_move(action)
            final_game.render()
            winner = 3 - final_game.state['pid']
            print(f'Winner is player {winner}\n\n')
