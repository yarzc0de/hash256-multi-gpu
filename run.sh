#!/bin/bash
# HASH Miner Manager — tmux-based
# Usage: ./run.sh [start|stop|restart|status|logs]

SESSION="hashminer"
BINARY="./target/release/hash-miner-rs"

start() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "⚠️  Miner already running. Use: ./run.sh logs"
        return 1
    fi

    if [ ! -f "$BINARY" ]; then
        echo "🔨 Binary not found, building..."
        cargo build --release --features gpu || exit 1
    fi

    echo "🚀 Starting miner in tmux session '$SESSION'..."
    tmux new-session -d -s "$SESSION" "GPU=1 $BINARY; echo '--- MINER EXITED ---'; read"
    echo "✅ Miner started!"
    echo ""
    echo "   Logs:    ./run.sh logs"
    echo "   Stop:    ./run.sh stop"
    echo "   Status:  ./run.sh status"
}

stop() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "❌ Miner not running."
        return 1
    fi

    echo "🛑 Stopping miner..."
    tmux send-keys -t "$SESSION" C-c
    sleep 2

    # Force kill if still alive
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
    fi
    echo "✅ Miner stopped."
}

restart() {
    echo "🔄 Restarting miner..."
    stop 2>/dev/null
    sleep 1
    start
}

status() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "✅ Miner is RUNNING (tmux session: $SESSION)"
        echo ""
        echo "Last output:"
        tmux capture-pane -t "$SESSION" -p | tail -10
    else
        echo "❌ Miner is NOT running."
    fi
}

logs() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "❌ Miner not running."
        return 1
    fi

    echo "📺 Attaching to miner (Ctrl+B then D to detach)..."
    sleep 1
    tmux attach -t "$SESSION"
}

build() {
    echo "🔨 Building release binary..."
    cargo build --release --features gpu
    echo "✅ Build complete."
}

update() {
    echo "📥 Pulling latest code..."
    git pull || exit 1
    build
    echo ""
    echo "✅ Updated! Run './run.sh restart' to apply."
}

case "${1:-help}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    logs)    logs ;;
    build)   build ;;
    update)  update ;;
    *)
        echo "HASH Miner Manager"
        echo "=================="
        echo ""
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  start    Start miner in background (tmux)"
        echo "  stop     Stop miner (Ctrl+C + kill)"
        echo "  restart  Stop + Start"
        echo "  status   Check if running + last output"
        echo "  logs     Attach to miner output (Ctrl+B, D to detach)"
        echo "  build    Rebuild binary"
        echo "  update   Git pull + rebuild"
        echo ""
        echo "Config: edit .env file"
        ;;
esac
