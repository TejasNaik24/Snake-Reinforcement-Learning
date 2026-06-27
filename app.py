import os
# Force Pygame to run in headless dummy mode so it doesn't open a desktop window
os.environ["SDL_VIDEODRIVER"] = "dummy"

import streamlit as st
import time
import matplotlib.pyplot as plt

# Importing game/agent below also imports pygame; the SDL dummy-driver env var
# set above must stay before these imports so Pygame stays fully headless.
from agent import Agent
from game import SnakeGameAI, BLOCK_SIZE

# Set page config
st.set_page_config(
    page_title="Snake RL Trainer",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to render modern matplotlib progress chart
def draw_chart(scores, mean_scores):
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    ax.plot(scores, label='Score', color='#58a6ff', linewidth=2, alpha=0.9)
    ax.plot(mean_scores, label='Mean Score', color='#ff79c6', linewidth=2.5)
    
    ax.set_title('Deep Q-Network Training Progress', color='#c9d1d9', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Games', color='#8b949e', fontsize=10)
    ax.set_ylabel('Score', color='#8b949e', fontsize=10)
    
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['right'].set_color('#30363d')
    
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', loc='upper left')
    ax.grid(color='#30363d', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# Render the current game state as a lightweight SVG string. This is far
# cheaper than encoding a full 640x480 PNG every frame (which chokes on
# Streamlit Cloud's single shared CPU core and slow websocket) — the SVG is
# just a few KB of text that the browser draws natively, giving smooth motion.
def game_to_svg(game):
    parts = [
        f'<svg viewBox="0 0 {game.w} {game.h}" width="100%" '
        f'style="background:#000;border-radius:8px;display:block;aspect-ratio:{game.w}/{game.h};">'
    ]
    for pt in game.snake:
        x, y = int(pt.x), int(pt.y)
        parts.append(f'<rect x="{x}" y="{y}" width="20" height="20" fill="#0000ff"/>')
        parts.append(f'<rect x="{x + 4}" y="{y + 4}" width="12" height="12" fill="#0064ff"/>')
    fx, fy = int(game.food.x), int(game.food.y)
    parts.append(f'<rect x="{fx}" y="{fy}" width="20" height="20" fill="#c80000"/>')
    parts.append(
        f'<text x="6" y="26" fill="#fff" font-size="24" '
        f'font-family="monospace">Score: {game.score}</text>'
    )
    parts.append('</svg>')
    return "".join(parts)

class DummyClock:
    def tick(self, fps):
        pass

def reset_training():
    st.session_state.agent = Agent()
    st.session_state.game = SnakeGameAI()
    st.session_state.game.clock = DummyClock()
    st.session_state.plot_scores = []
    st.session_state.plot_mean_scores = []
    st.session_state.total_score = 0
    st.session_state.record = 0
    st.session_state.training_running = False
    st.session_state.game_won = False

@st.dialog("The Snake Beat the Game!")
def show_win_dialog():
    agent = st.session_state.agent
    st.write(
        "The agent filled the entire board with its body — there's nowhere left "
        "for food to spawn, which is the highest possible score in Snake."
    )
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Final Score", st.session_state.record)
    m_col2.metric("Games Played", agent.n_games)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Close", use_container_width=True):
            st.session_state.game_won = False
            st.rerun()
    with btn_col2:
        if st.button("Retrain", use_container_width=True, type="primary"):
            reset_training()
            st.rerun()

# Initialize persistent session states
if 'agent' not in st.session_state:
    st.session_state.agent = Agent()
if 'game' not in st.session_state:
    st.session_state.game = SnakeGameAI()
    # Disable internal pygame display clock tick to prevent loop blockages
    st.session_state.game.clock = DummyClock()
if 'plot_scores' not in st.session_state:
    st.session_state.plot_scores = []
if 'plot_mean_scores' not in st.session_state:
    st.session_state.plot_mean_scores = []
if 'total_score' not in st.session_state:
    st.session_state.total_score = 0
if 'record' not in st.session_state:
    st.session_state.record = 0
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_running' not in st.session_state:
    st.session_state.training_running = False
if 'game_won' not in st.session_state:
    st.session_state.game_won = False

# Render application UI
if not st.session_state.training_started:
    # ------------------ LANDING / HOME PAGE ------------------
    home_placeholder = st.empty()
    with home_placeholder.container():
        st.title("🐍 Snake Reinforcement Learning")
        st.caption("Train a Deep Q-Network (DQN) agent to play Snake in real-time in your browser")
        st.divider()

        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            with st.container(border=True):
                st.subheader("Project Overview")
                st.write(
                    "A **Deep Q-Learning** agent learning to play Snake completely from scratch — no hardcoded rules, "
                    "just a PyTorch Neural Network figuring out through trial and error how to chase food, dodge walls, "
                    "and avoid running into its own tail."
                )
                st.divider()
                st.subheader("Key Elements of the AI:")
                st.markdown("""
                *   **State Representation**: 11-dimensional binary vector representing danger (left, right, straight), moving direction, and food direction.
                *   **Action Vector**: Relative moves: `[1,0,0]` (straight), `[0,1,0]` (turn right), `[0,0,1]` (turn left).
                *   **Experience Replay**: Two levels of memory (Short-term after every step, Long-term random batch optimization of 1000 experiences on game-over).
                """)

            st.write("")
            st.write("")
            if st.button("Train Model", use_container_width=True, type="primary"):
                home_placeholder.empty()
                st.session_state.training_started = True
                st.session_state.training_running = True
                st.rerun()

else:
    # ------------------ TRAINING DASHBOARD ------------------
    st.title("Training Dashboard")
    
    # Metrics display bar
    stats_placeholder = st.empty()
    st.divider()

    # Top Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        if st.session_state.training_running:
            if st.button("Pause Training", use_container_width=True):
                st.session_state.training_running = False
                st.rerun()
        else:
            if st.button("Resume Training", use_container_width=True, type="primary"):
                st.session_state.training_running = True
                st.rerun()
    with ctrl_col2:
        if st.button("Reset Training", use_container_width=True):
            reset_training()
            st.rerun()
    with ctrl_col3:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.training_started = False
            st.session_state.training_running = False
            st.rerun()
            
    st.write("")
    
    # Main Dashboard Columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Game Screen")
        game_image_placeholder = st.empty()
        
    with col2:
        st.subheader("Training Metrics")
        chart_placeholder = st.empty()

    # Speed Slider at the bottom
    st.write("")
    speed = st.slider("Snake Speed (Frames per Second)", min_value=1, max_value=30, value=15)
    
    # Training Loop Controller
    agent = st.session_state.agent
    game = st.session_state.game
    
    # Helper to draw dynamic metrics
    def update_metrics_display(curr_score):
        with stats_placeholder.container():
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Games Played", agent.n_games)
            m_col2.metric("High Score", st.session_state.record)
            m_col3.metric("Current Score", curr_score)
            eps_val = max(0, agent.epsilon)
            m_col4.metric("Epsilon (Randomness %)", f"{eps_val}%")

    # First frame rendering if paused
    if not st.session_state.training_running:
        # Draw the current board as a lightweight SVG
        game_image_placeholder.html(game_to_svg(game))

        # Display latest plot
        if len(st.session_state.plot_scores) > 0:
            fig = draw_chart(st.session_state.plot_scores, st.session_state.plot_mean_scores)
            chart_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            chart_placeholder.info("Click 'Resume Training' to start playing and generating metrics!")
            
        update_metrics_display(0)

    # Maximum cells on the board; if the snake's body fills all but one of
    # them, the next food pickup would leave nowhere for food to spawn, so
    # we declare victory here instead of letting game.py recurse forever
    # looking for an empty cell to place food on.
    total_cells = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)

    # One training step is rendered per frame, so on screen the snake always
    # advances exactly one cell instead of teleporting forward, and each frame
    # is paced to a steady wall-clock budget set by the speed slider.
    target_frame_time = 1.0 / speed
    last_metrics_time = 0.0
    last_chart_time = 0.0

    # Active running loop
    while st.session_state.training_running:
        frame_start = time.time()

        if len(game.snake) >= total_cells - 1:
            st.session_state.game_won = True
            st.session_state.training_running = False
            break

        # 1. Get old state
        state_old = agent.get_state(game)

        # 2. Get action
        final_move = agent.get_action(state_old)

        # 3. Perform action
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # 6. Render the game frame every step as a lightweight SVG (cheap to
        # build and tiny to send, unlike PNG-encoding a full image each frame)
        game_image_placeholder.html(game_to_svg(game))

        # 7. Metric widgets are heavier to push than the SVG, so refresh them
        # a few times a second rather than on every single frame
        if frame_start - last_metrics_time >= 0.2:
            update_metrics_display(score)
            last_metrics_time = frame_start

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > st.session_state.record:
                st.session_state.record = score
                try:
                    agent.model.save()
                except Exception:
                    pass
            
            # Store scores
            st.session_state.plot_scores.append(score)
            st.session_state.total_score += score
            mean_score = st.session_state.total_score / agent.n_games
            st.session_state.plot_mean_scores.append(mean_score)
            
            # Re-draw plot, but throttle it: rendering a matplotlib figure
            # encodes a PNG, which is expensive on the cloud's shared core, so
            # we skip redraws that land within a second of the previous one.
            if frame_start - last_chart_time >= 1.0:
                fig = draw_chart(st.session_state.plot_scores, st.session_state.plot_mean_scores)
                chart_placeholder.pyplot(fig)
                plt.close(fig)
                last_chart_time = frame_start

        # Pace each frame to a steady cadence; subtract the time already spent
        # this iteration so a costly step (e.g. end-of-game training) doesn't
        # compound into the next frame's delay.
        elapsed = time.time() - frame_start
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)

    if st.session_state.game_won:
        show_win_dialog()
