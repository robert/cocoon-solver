from copy import deepcopy
from typing import Callable

import networkx as nx
from attrs import define

BALL_HELD = "BALL_HELD"


@define(frozen=True)
class Position:
    inside_ball: str
    node: int


@define()
class State:
    ball_positions: dict[str, Position | str]
    ball_last_exit_positions: dict[str, Position]
    player_position: Position
    fairy_created: bool
    # TODO: where is dead fairy - needed in order to solve 51

    def held_ball(self) -> str | None:
        return next(
            (
                col
                for col in self.ball_positions.keys()
                if self.ball_positions[col] == BALL_HELD
            ),
            None,
        )

    def ball_position(self, color: str) -> Position:
        pos = self.ball_positions[color]
        if pos == BALL_HELD:
            return self.player_position
        else:
            return pos

    def ball_dropped_at_position(self, position: Position) -> str | None:
        return next(
            (
                col
                for col in self.ball_positions.keys()
                if self.ball_position(col) == position and self.held_ball() != col
            ),
            None,
        )


@define(frozen=True)
class World:
    graphs: dict[str, nx.Graph]

    def graph(self, color: str) -> nx.Graph:
        return self.graphs[color]


@define(frozen=True)
class Transition:
    def next_state(self, orig_state: State, world: World) -> State:
        pass

    def is_legal(self, state: State) -> bool:
        pass


@define(frozen=True)
class MovePlayer(Transition):
    target_position: Position

    def next_state(self, orig_state: State, world: World) -> State:
        new_state = deepcopy(orig_state)
        new_state.player_position = self.target_position
        if node_type(self.target_position, world) == "fairy_catcher":
            new_state.fairy_created = False
        return new_state

    def is_legal(self, orig_state: State, world: World) -> bool:
        graph = world.graph(self.target_position.inside_ball)
        if (
            node_is_vine(self.target_position, world)
            and not orig_state.ball_positions["green"] == BALL_HELD
        ):
            return False

        if (
            node_is_invisible_path(self.target_position, world)
            and not orig_state.ball_positions["red"] == BALL_HELD
        ):
            return False

        if (
            node_type(self.target_position, world) == "ball_blocker"
            and orig_state.held_ball() is not None
        ):
            return False

        cur_node = orig_state.player_position.node
        target_node = self.target_position.node
        return graph.has_edge(cur_node, target_node)


class EnterPortal(Transition):
    def next_state(self, orig_state: State, world: World) -> State:
        new_state = deepcopy(orig_state)

        new_player_position = node_portal_other_side(new_state.player_position, world)
        new_state.player_position = deepcopy(new_player_position)
        return new_state

    def is_legal(self, orig_state: State, world: World) -> bool:
        nt = node_type(orig_state.player_position, world)
        return nt == "portal"


class PlaceBall(Transition):
    def next_state(self, orig_state: State, world: World) -> State:
        new_state = deepcopy(orig_state)

        cur_held_ball = new_state.held_ball()
        cur_dropped_ball = new_state.ball_dropped_at_position(new_state.player_position)

        new_state.ball_positions[cur_held_ball] = new_state.player_position
        if cur_dropped_ball is not None:
            new_state.ball_positions[cur_dropped_ball] = BALL_HELD
        return new_state

    def is_legal(self, orig_state: State, world: World) -> bool:
        nt = node_type(orig_state.player_position, world)
        return nt in ["ball_stand", "pool"] and orig_state.held_ball() is not None


class PickupBall(Transition):
    def next_state(self, orig_state: State, world: World) -> State:
        new_state = deepcopy(orig_state)

        cur_held_ball = new_state.held_ball()
        if cur_held_ball is not None:
            new_state.ball_positions[cur_held_ball] = orig_state.player_position

        cur_dropped_ball = new_state.ball_dropped_at_position(new_state.player_position)
        new_state.ball_positions[cur_dropped_ball] = BALL_HELD
        return new_state

    def is_legal(self, orig_state: State, world: World) -> bool:
        nt = node_type(orig_state.player_position, world)
        dropped_ball = orig_state.ball_dropped_at_position(orig_state.player_position)
        return nt in ["ball_stand", "pool"] and dropped_ball is not None


class EnterBall(Transition):
    def next_state(self, orig_state: State, world: World) -> State:
        new_state = deepcopy(orig_state)

        dropped_ball = new_state.ball_dropped_at_position(new_state.player_position)
        next_position = new_state.ball_last_exit_positions[dropped_ball]
        new_state.player_position = deepcopy(next_position)

        return new_state

    def is_legal(self, orig_state: State, world: World) -> bool:
        nt = node_type(orig_state.player_position, world)
        dropped_ball = orig_state.ball_dropped_at_position(orig_state.player_position)
        return nt in ["pool"] and dropped_ball is not None


class ExitBall(Transition):
    def next_state(self, orig_state: State, world: World) -> State:
        new_state = deepcopy(orig_state)

        current_ball = new_state.player_position.inside_ball
        new_state.ball_last_exit_positions[current_ball] = deepcopy(
            new_state.player_position
        )
        next_position = new_state.ball_positions[current_ball]
        new_state.player_position = deepcopy(next_position)
        return new_state

    def is_legal(self, orig_state: State, world: World) -> bool:
        nt = node_type(orig_state.player_position, world)
        return nt in ["exit"]


class CreateFairy(Transition):
    def next_state(self, orig_state: State, world: World) -> State:
        new_state = deepcopy(orig_state)
        new_state.fairy_created = True
        return new_state

    def is_legal(self, orig_state: State, world: World) -> bool:
        nt = node_type(orig_state.player_position, world)
        return nt in ["fairy_creator"] and not orig_state.fairy_created


def node_type(position: Position, world: World) -> str:
    graph = world.graph(position.inside_ball)
    node = graph.nodes[position.node]
    return node["node_type"]


def node_neighbors(position: Position, world: World) -> list[int]:
    graph = world.graph(position.inside_ball)
    return graph.neighbors(position.node)


def node_is_vine(position: Position, world: World) -> bool:
    graph = world.graph(position.inside_ball)
    node = graph.nodes[position.node]
    return node.get("is_vine", False)


def node_is_invisible_path(position: Position, world: World) -> bool:
    graph = world.graph(position.inside_ball)
    node = graph.nodes[position.node]
    return node.get("is_invisible_path", False)


def node_portal_other_side(position: Position, world: World) -> Position:
    graph = world.graph(position.inside_ball)
    node = graph.nodes[position.node]
    return node["other_side"]


def legal_transitions(orig_state: State, world: World) -> list[Transition]:
    neighbours = node_neighbors(orig_state.player_position, world)
    all_move_players = [
        MovePlayer(
            target_position=Position(
                inside_ball=orig_state.player_position.inside_ball,
                node=node_idx,
            ),
        )
        for node_idx in neighbours
    ]
    transitions = [t for t in all_move_players if t.is_legal(orig_state, world)]
    other_transitions = [
        PlaceBall,
        PickupBall,
        EnterBall,
        ExitBall,
        CreateFairy,
        EnterPortal,
    ]
    for tc in other_transitions:
        t = tc()
        if t.is_legal(orig_state, world):
            transitions.append(t)

    return transitions


def puzzle_51() -> tuple[World, State]:
    pg = nx.Graph()
    pg.add_node(0, node_type="ball_stand")
    pg.add_node(1, node_type="pool")
    pg.add_node(2, node_type="fairy_catcher")
    pg.add_node(3, node_type="fairy_creator")
    pg.add_node(4, node_type="ball_stand")
    pg.add_node(5, node_type="pool")
    pg.add_edges_from(
        [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    )

    gg = nx.Graph()
    gg.add_node(0, node_type="ball_stand")
    gg.add_node(1, node_type="exit")
    gg.add_node(2, node_type="fairy_catcher")
    gg.add_node(3, node_type="fairy_creator")
    gg.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    rg = nx.Graph()
    rg.add_node(0, node_type="ball_stand")
    rg.add_node(1, node_type="exit")
    rg.add_edges_from([(0, 1)])

    world = World({"green": gg, "red": rg, "purple": pg})
    state = State(
        ball_positions={
            "red": Position(inside_ball="purple", node=4),
            "green": Position(inside_ball="purple", node=5),
        },
        ball_last_exit_positions={
            "red": Position(inside_ball="red", node=1),
            "green": Position(inside_ball="green", node=1),
        },
        player_position=Position(inside_ball="purple", node=4),
        fairy_created=False,
    )
    return (world, state)


def puzzle_50() -> tuple[World, State]:
    pg = nx.Graph()
    pg.add_node(0, node_type="ball_stand")
    pg.add_node(1, node_type="ball_stand")
    pg.add_node(2, node_type="exit")
    pg.add_node(3, node_type="vine", is_vine=True)
    pg.add_node(4, node_type="exit")
    pg.add_node(5, node_type="is_invisible_path", is_invisible_path=True)
    pg.add_node(6, node_type="pool")
    pg.add_node(7, node_type="ball_stand")
    pg.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (5, 7),
            (6, 7),
        ]
    )

    gg = nx.Graph()
    gg.add_node(0, node_type="ball_stand")
    gg.add_node(1, node_type="exit")
    gg.add_edges_from([(0, 1)])

    rg = nx.Graph()
    rg.add_node(0, node_type="ball_stand")
    rg.add_node(1, node_type="exit")
    rg.add_edges_from([(0, 1)])

    ow = nx.Graph()
    ow.add_node(0, node_type="ball_stand")
    ow.add_node(1, node_type="ball_stand")
    ow.add_node(2, node_type="pool")
    ow.add_edges_from([(0, 1), (0, 2), (1, 2)])

    world = World({"green": gg, "red": rg, "purple": pg, "overworld": ow})
    state = State(
        ball_positions={
            "red": Position(inside_ball="purple", node=7),
            "green": Position(inside_ball="overworld", node=0),
            "purple": Position(inside_ball="overworld", node=2),
        },
        ball_last_exit_positions={
            "red": Position(inside_ball="red", node=1),
            "green": Position(inside_ball="green", node=1),
            "purple": Position(inside_ball="purple", node=2),
        },
        player_position=Position(inside_ball="purple", node=7),
        fairy_created=False,
    )
    return (world, state)


def puzzle_49() -> tuple[World, State]:
    pg = nx.Graph()
    pg.add_node(0, node_type="ball_stand")
    pg.add_node(1, node_type="ball_stand")
    pg.add_node(2, node_type="exit")
    pg.add_node(3, node_type="vine", is_vine=True)
    pg.add_node(4, node_type="exit")
    pg.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)])

    gg = nx.Graph()
    gg.add_node(0, node_type="ball_stand")
    gg.add_node(1, node_type="exit")
    gg.add_edges_from([(0, 1)])

    rg = nx.Graph()
    rg.add_node(0, node_type="ball_stand")
    rg.add_node(1, node_type="exit")
    rg.add_edges_from([(0, 1)])

    ow = nx.Graph()
    ow.add_node(0, node_type="ball_stand")
    ow.add_node(1, node_type="ball_stand")
    ow.add_node(2, node_type="pool")
    ow.add_edges_from([(0, 1), (0, 2), (1, 2)])

    world = World({"green": gg, "red": rg, "purple": pg, "overworld": ow})
    state = State(
        ball_positions={
            "red": Position(inside_ball="purple", node=0),
            "green": Position(inside_ball="purple", node=1),
            "purple": Position(inside_ball="overworld", node=2),
        },
        ball_last_exit_positions={
            "red": Position(inside_ball="red", node=1),
            "green": Position(inside_ball="green", node=1),
            "purple": Position(inside_ball="purple", node=2),
        },
        player_position=Position(inside_ball="purple", node=0),
        fairy_created=False,
    )
    return (world, state)


def puzzle_92() -> tuple[World, State, Callable[[State], bool]]:
    pg = nx.Graph()
    pg.add_node(0, node_type="ball_stand")
    pg.add_node(1, node_type="pool")
    pg.add_node(2, node_type="portal", other_side=Position(inside_ball="green", node=0))

    pg.add_edges_from([(0, 1), (0, 2), (1, 2)])

    gg = nx.Graph()
    gg.add_node(
        0, node_type="portal", other_side=Position(inside_ball="purple", node=2)
    )
    gg.add_node(1, node_type="is_invisible_path", is_invisible_path=True)
    gg.add_node(2, node_type="ball_stand")
    gg.add_node(3, node_type="ball_stand")
    gg.add_node(4, node_type="ball_blocker")
    gg.add_node(5, node_type="exit")

    gg.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 5)])

    rg = nx.Graph()
    rg.add_node(0, node_type="ball_stand")
    rg.add_node(1, node_type="exit")
    rg.add_edges_from([(0, 1)])

    world = World({"green": gg, "red": rg, "purple": pg})
    state = State(
        ball_positions={
            "red": Position(inside_ball="purple", node=0),
            "green": Position(inside_ball="purple", node=1),
        },
        ball_last_exit_positions={
            "red": Position(inside_ball="red", node=1),
            "green": Position(inside_ball="green", node=3),
        },
        player_position=Position(inside_ball="purple", node=0),
        fairy_created=False,
    )

    def is_success(state: State) -> bool:
        success_positions = [
            Position(inside_ball="green", node=1),
            Position(inside_ball="green", node=2),
        ]
        return (
            state.player_position in success_positions
            and state.ball_position("red") in success_positions
            and state.ball_position("green") in success_positions
        )

    return (world, state, is_success)


def is_success_51(state: State) -> bool:
    return (
        state.player_position
        in [
            Position(inside_ball="purple", node=0),
            Position(inside_ball="purple", node=1),
        ]
        and state.fairy_created
    )


def is_success_50(state: State) -> bool:
    goal_positions = [
        Position(inside_ball="purple", node=6),
        Position(inside_ball="purple", node=7),
    ]
    return (
        state.player_position in goal_positions
        and state.ball_position("green") in goal_positions
    )


def is_success_49(state: State) -> bool:
    return state.player_position == Position(
        inside_ball="purple", node=4
    ) and state.ball_position("red") == Position(inside_ball="purple", node=4)


from collections import deque


def bfs_search(
    initial_state: State, world: World, is_success_func
) -> list[Transition] | None:
    """
    Performs a breadth-first search to find a solution path.
    Returns the list of transitions that lead to success, or None if no solution exists.
    """
    visited_states = set()

    def get_state_hash(state: State) -> tuple:
        """Creates a hashable representation of a state for visited checking"""
        # Handle ball positions
        ball_pos_tuple = tuple(
            sorted(
                (k, v if isinstance(v, str) else (v.inside_ball, v.node))
                for k, v in state.ball_positions.items()
            )
        )

        # Handle last exit positions
        last_exit_tuple = tuple(
            sorted(
                (k, (v.inside_ball, v.node))
                for k, v in state.ball_last_exit_positions.items()
            )
        )

        # Handle player position
        player_pos_tuple = (
            state.player_position.inside_ball,
            state.player_position.node,
        )

        return (ball_pos_tuple, last_exit_tuple, player_pos_tuple, state.fairy_created)

    # Queue entries are tuples of (state, path_to_state)
    queue = deque([(initial_state, [])])
    initial_hash = get_state_hash(initial_state)
    visited_states.add(initial_hash)

    while queue:
        current_state, path = queue.popleft()

        if is_success_func(current_state):
            return path

        # Get all possible next states
        transitions = legal_transitions(current_state, world)

        for transition in transitions:
            next_state = transition.next_state(current_state, world)
            next_hash = get_state_hash(next_state)

            if next_hash not in visited_states:
                visited_states.add(next_hash)
                queue.append((next_state, path + [transition]))

    return None


def print_state(state: State):
    """Prints the current state in a clear, formatted way"""
    print("\n=== Current State ===")
    print(
        f"🚶 Player Location: {state.player_position.inside_ball}:{state.player_position.node}"
    )

    # Print held ball info
    held_ball = state.held_ball()
    if held_ball:
        print(f"✋ Holding: {held_ball} ball")
    else:
        print("✋ Hands empty")

    # Print all ball positions
    print("\n📍 Ball Positions:")
    for color, pos in state.ball_positions.items():
        if pos != BALL_HELD:
            print(f"  • {color}: {pos.inside_ball}:{pos.node}")

    print("=" * 20)


def print_state_transitions(
    initial_state: State, world: World, transitions: list[Transition]
) -> None:
    """
    Prints a series of state transitions in a nicely formatted way with emojis.

    Args:
        initial_state (State): The starting state
        world (World): The world object containing the game graphs
        transitions (list[Transition]): List of transitions to apply
    """
    print("\n🎮 Starting State Visualization:")
    print("=" * 50)

    # Print initial state
    print("\n📝 Initial State:")
    print_state(initial_state)

    # Apply and print each transition
    current_state = initial_state
    for i, transition in enumerate(transitions, 1):
        print("\n" + "=" * 50)
        print(f"\n🔄 Step {i}: {transition.__class__.__name__}")

        # Apply transition
        try:
            current_state = transition.next_state(current_state, world)

            # Print the new state
            print("\n📍 Resulting State:")
            print_state(current_state)

            # Print additional transition-specific info
            if isinstance(transition, MovePlayer):
                print(
                    f"🚶 Moved to: {transition.target_position.inside_ball}:{transition.target_position.node}"
                )
            elif isinstance(transition, PlaceBall):
                print("🎯 Placed ball")
            elif isinstance(transition, PickupBall):
                print("✋ Picked up ball")
            elif isinstance(transition, EnterBall):
                print("⭐ Entered ball")
            elif isinstance(transition, ExitBall):
                print("🚪 Exited ball")
            elif isinstance(transition, CreateFairy):
                print("🧚 Created fairy")
            elif isinstance(transition, EnterPortal):
                print("🌀 Entered portal")

        except Exception as e:
            print(f"❌ Error applying transition: {str(e)}")
            break

    print("\n" + "=" * 50)
    print("✨ Visualization Complete")


def solve_puzzle(world, initial_state, is_success, puzzle_num):
    print(f"\n🧩 Solving Puzzle {puzzle_num}:")
    print("\n📝 Initial State:")
    print_state(initial_state)

    solution = bfs_search(initial_state, world, is_success)

    if solution is None:
        print("❌ No solution found!")
        return

    print(f"\n✨ Found solution with {len(solution)} steps!\n")

    # Apply and verify each step
    current_state = initial_state
    for i, transition in enumerate(solution, 1):
        print(f"\n🔄 Step {i}: {transition.__class__.__name__}")
        current_state = transition.next_state(current_state, world)
        print_state(current_state)

    # Print success confirmation
    success = is_success(current_state)
    print(f"\n{'🎉 Success!' if success else '❌ Failed!'}")


from typing import Callable

import networkx as nx
from attrs import define


@define(frozen=True)
class StateNode:
    state: State

    def __hash__(self) -> int:
        """Hash based on state contents"""
        ball_pos_tuple = tuple(
            sorted(
                (k, v if isinstance(v, str) else (v.inside_ball, v.node))
                for k, v in self.state.ball_positions.items()
            )
        )
        last_exit_tuple = tuple(
            sorted(
                (k, (v.inside_ball, v.node))
                for k, v in self.state.ball_last_exit_positions.items()
            )
        )
        player_pos_tuple = (
            self.state.player_position.inside_ball,
            self.state.player_position.node,
        )
        return hash(
            (
                ball_pos_tuple,
                last_exit_tuple,
                player_pos_tuple,
                self.state.fairy_created,
            )
        )

    def __eq__(self, other) -> bool:
        """Equality based on state contents"""
        if not isinstance(other, StateNode):
            return False
        return (
            self.state.ball_positions == other.state.ball_positions
            and self.state.ball_last_exit_positions
            == other.state.ball_last_exit_positions
            and self.state.player_position == other.state.player_position
            and self.state.fairy_created == other.state.fairy_created
        )

    def __str__(self) -> str:
        """Pretty print representation of the state"""
        lines = []
        lines.append("\n=== Current State ===")
        lines.append(
            f"🚶 Player Location: {self.state.player_position.inside_ball}:{self.state.player_position.node}"
        )

        held_ball = self.state.held_ball()
        if held_ball:
            lines.append(f"✋ Holding: {held_ball} ball")
        else:
            lines.append("✋ Hands empty")

        lines.append("\n📍 Ball Positions:")
        for color, pos in self.state.ball_positions.items():
            if pos != BALL_HELD:
                lines.append(f"  • {color}: {pos.inside_ball}:{pos.node}")

        lines.append("=" * 20)
        return "\n".join(lines)


@define
class StateEdge:
    transition: Transition

    def __str__(self) -> str:
        """Pretty print representation of the transition"""
        transition_type = self.transition.__class__.__name__
        if isinstance(self.transition, MovePlayer):
            return f"🚶 {transition_type}: Move to {self.transition.target_position.inside_ball}:{self.transition.target_position.node}"
        elif isinstance(self.transition, PlaceBall):
            return f"🎯 {transition_type}: Place ball"
        elif isinstance(self.transition, PickupBall):
            return f"✋ {transition_type}: Pick up ball"
        elif isinstance(self.transition, EnterBall):
            return f"⭐ {transition_type}: Enter ball"
        elif isinstance(self.transition, ExitBall):
            return f"🚪 {transition_type}: Exit ball"
        elif isinstance(self.transition, CreateFairy):
            return f"🧚 {transition_type}: Create fairy"
        elif isinstance(self.transition, EnterPortal):
            return f"🌀 {transition_type}: Enter portal"
        return f"🔄 {transition_type}"


def build_state_graph(initial_state: State, world: World) -> nx.DiGraph:
    """
    Builds a directed graph of all possible states reachable from the initial state.
    Each node is a StateNode containing a State object.
    Each edge is a StateEdge containing the Transition that leads to the next state.
    """
    G = nx.DiGraph()
    visited_states = set()

    def get_state_hash(state: State) -> tuple:
        """Creates a hashable representation of a state"""
        ball_pos_tuple = tuple(
            sorted(
                (k, v if isinstance(v, str) else (v.inside_ball, v.node))
                for k, v in state.ball_positions.items()
            )
        )
        last_exit_tuple = tuple(
            sorted(
                (k, (v.inside_ball, v.node))
                for k, v in state.ball_last_exit_positions.items()
            )
        )
        player_pos_tuple = (
            state.player_position.inside_ball,
            state.player_position.node,
        )
        return (ball_pos_tuple, last_exit_tuple, player_pos_tuple, state.fairy_created)

    # Create the initial node
    initial_node = StateNode(state=initial_state)
    G.add_node(initial_node)
    initial_hash = get_state_hash(initial_state)
    visited_states.add(initial_hash)

    # Queue entries are StateNode objects
    queue = deque([initial_node])

    while queue:
        current_node = queue.popleft()
        current_state = current_node.state

        # Get all possible transitions from current state
        transitions = legal_transitions(current_state, world)

        for transition in transitions:
            next_state = transition.next_state(current_state, world)
            next_hash = get_state_hash(next_state)

            if next_hash not in visited_states:
                next_node = StateNode(state=next_state)
                G.add_node(next_node)
                G.add_edge(
                    current_node, next_node, transition=StateEdge(transition=transition)
                )
                visited_states.add(next_hash)
                queue.append(next_node)
            else:
                # Find existing node with this state hash
                for node in G.nodes():
                    if get_state_hash(node.state) == next_hash:
                        G.add_edge(
                            current_node,
                            node,
                            transition=StateEdge(transition=transition),
                        )
                        break

    return G


def find_solution_path(
    G: nx.DiGraph, initial_node: StateNode, is_success_func: Callable[[State], bool]
) -> list[tuple[StateNode, StateEdge]] | None:
    """
    Finds a path from initial_node to a success state in the graph.
    Returns list of (node, edge) pairs representing the path, or None if no solution exists.
    """
    # Find all nodes that satisfy the success condition
    success_nodes = [node for node in G.nodes() if is_success_func(node.state)]
    if not success_nodes:
        return None

    # Find shortest path to any success node
    shortest_path = None
    shortest_length = float("inf")

    for target_node in success_nodes:
        try:
            path = nx.shortest_path(G, initial_node, target_node)
            if len(path) < shortest_length:
                shortest_path = path
                shortest_length = len(path)
        except nx.NetworkXNoPath:
            continue

    if shortest_path is None:
        return None

    # Convert path to list of (node, edge) pairs
    path_with_edges = []
    for i in range(len(shortest_path) - 1):
        current_node = shortest_path[i]
        next_node = shortest_path[i + 1]
        edge_data = G.get_edge_data(current_node, next_node)
        path_with_edges.append((current_node, edge_data["transition"]))

    # Add final node without edge
    path_with_edges.append((shortest_path[-1], None))

    return path_with_edges


def solve_puzzle_with_graph(world, initial_state, is_success, puzzle_num):
    """Solve puzzle using state graph approach"""
    print(f"\n🧩 Solving Puzzle {puzzle_num} using State Graph:")

    # Build state graph
    print("\n📊 Building state graph...")
    G = build_state_graph(initial_state, world)
    save_graph_image(G, initial_state, is_success, f"puzzle_{puzzle_num}_graph.png")
    print(
        f"Generated graph with {G.number_of_nodes()} states and {G.number_of_edges()} transitions"
    )

    # Find initial node
    initial_node = next(node for node in G.nodes() if node.state == initial_state)

    # Print initial state
    print("\n📝 Initial State:")
    print(initial_node)

    # Find solution path
    solution = find_solution_path(G, initial_node, is_success)

    if solution is None:
        print("❌ No solution found!")
        return

    print(f"\n✨ Found solution with {len(solution)-1} steps!\n")

    # Print each step
    for i, (node, edge) in enumerate(solution):
        if i > 0:  # Skip printing transition for initial state
            print(f"\n🔄 Step {i}: {edge}")
        print(node)

    # Print success confirmation
    success = is_success(solution[-1][0].state)
    print(f"\n{'🎉 Success!' if success else '❌ Failed!'}")


import matplotlib.pyplot as plt


def save_graph_image(G, initial_state, is_success, filename="graph.png"):
    plt.figure(figsize=(40, 40))  # Increased figure size
    pos = nx.kamada_kawai_layout(G)  # Better layout algorithm

    # Draw regular nodes with larger size
    nx.draw_networkx_nodes(
        G, pos, node_color="white", node_size=2000, edgecolors="black", linewidths=1
    )

    # Find and draw start node
    start_node = next(node for node in G.nodes() if node.state == initial_state)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[start_node],
        node_color="lightgreen",
        node_size=2500,
        edgecolors="darkgreen",
        linewidths=2,
    )

    # Find and draw goal nodes
    goal_nodes = [node for node in G.nodes() if is_success(node.state)]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=goal_nodes,
        node_color="lightcoral",
        node_size=2500,
        edgecolors="darkred",
        linewidths=2,
    )

    # Draw edges with better visibility
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, alpha=0.4, width=1, arrowsize=20
    )

    # Add node labels with increased size
    labels = {
        node: f"{node.state.player_position.inside_ball}:{node.state.player_position.node}"
        for node in G.nodes()
    }
    nx.draw_networkx_labels(
        G, pos, labels, font_size=12, font_weight="bold", font_color="black"
    )

    # Add edge labels for transitions with increased size and better positioning
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        transition = data["transition"].transition
        if isinstance(transition, MovePlayer):
            label = f"Move→{transition.target_position.node}"
        else:
            # Shorter names for common transitions
            label = transition.__class__.__name__.replace("Ball", "").replace(
                "Portal", ""
            )
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        font_size=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        horizontalalignment="center",
        verticalalignment="center",
        rotate=True,
    )

    plt.title("State Graph\nGreen: Start, Red: Goal", pad=20, fontsize=20)
    plt.axis("off")

    # Save with white background and higher DPI
    plt.savefig(
        filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()


import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class PuzzleInfo:
    """Information about a puzzle."""

    description: str
    generator: Callable
    success_checker: Optional[Callable] = None


def get_puzzle_registry() -> Dict[int, PuzzleInfo]:
    """Central registry of all available puzzles."""
    return {
        49: PuzzleInfo(
            description="Basic puzzle with balls and vines",
            generator=puzzle_49,
            success_checker=is_success_49,
        ),
        50: PuzzleInfo(
            description="Advanced puzzle with invisible paths",
            generator=puzzle_50,
            success_checker=is_success_50,
        ),
        51: PuzzleInfo(
            description="Fairy creation puzzle",
            generator=puzzle_51,
            success_checker=is_success_51,
        ),
        92: PuzzleInfo(
            description="Portal and ball blocker puzzle", generator=puzzle_92
        ),
    }


def get_puzzle_setup(puzzle_num: int) -> Tuple[World, State, Callable]:
    """Get the puzzle setup and success checker for a given puzzle number."""
    registry = get_puzzle_registry()
    if puzzle_num not in registry:
        raise ValueError(f"Invalid puzzle number: {puzzle_num}")

    puzzle_info = registry[puzzle_num]

    # Handle puzzles that return (world, state, success_func)
    result = puzzle_info.generator()
    if isinstance(result, tuple) and len(result) == 3:
        return result

    # Handle puzzles that return (world, state)
    world, state = result
    if puzzle_info.success_checker is None:
        raise ValueError(f"No success checker defined for puzzle {puzzle_num}")
    return world, state, puzzle_info.success_checker


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Puzzle solver and state graph generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Solve puzzle 92
    python solver.py solve 92
    
    # Generate state graph for puzzle 92
    python solver.py graph 92
    
    # Show available puzzles
    python solver.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a specific puzzle")
    solve_parser.add_argument("puzzle_num", type=int, help="Puzzle number to solve")

    # Graph command
    graph_parser = subparsers.add_parser(
        "graph", help="Generate state graph for a puzzle"
    )
    graph_parser.add_argument("puzzle_num", type=int, help="Puzzle number to analyze")

    # List command
    subparsers.add_parser("list", help="List available puzzles")

    return parser.parse_args(args)


def list_puzzles() -> None:
    """Display available puzzles and their descriptions."""
    print("\nAvailable Puzzles:")
    print("-" * 50)

    registry = get_puzzle_registry()
    for num, info in sorted(registry.items()):
        print(f"Puzzle {num}: {info.description}")


def solve_puzzle_wrapper(puzzle_num: int) -> None:
    """Wrapper to handle different puzzle return types."""
    world, state, success_func = get_puzzle_setup(puzzle_num)
    solve_puzzle(world, state, success_func, puzzle_num)


def solve_puzzle_with_graph_wrapper(puzzle_num: int) -> None:
    """Wrapper to handle different puzzle return types for graph generation."""
    world, state, success_func = get_puzzle_setup(puzzle_num)
    solve_puzzle_with_graph(world, state, success_func, puzzle_num)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)

    if parsed_args.command == "list":
        list_puzzles()
        return 0

    if parsed_args.command is None:
        print("Error: No command specified. Use --help for usage information.")
        return 1

    try:
        if parsed_args.command == "solve":
            solve_puzzle_wrapper(parsed_args.puzzle_num)
        elif parsed_args.command == "graph":
            solve_puzzle_with_graph_wrapper(parsed_args.puzzle_num)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return 1
    except Exception as e:
        print(f"Error executing puzzle: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
