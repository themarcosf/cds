# Image used does not allow shebang execution

# exit on error, unset vars, pipe failures
set -euo pipefail

# system architecture
ARCH=$(uname -m)
OS=$(uname -s)
echo "=== System Info ==="
echo -e "Architecture: $ARCH"
echo -e "Operating System: $OS\n"

# usage instructions for the script
show_help() {
    cat << HELP
Usage: bash run.sh --help
       bash run.sh --test live|offline

OPTIONS:
    --help            Shows this help message and exits.
    --test live       Runs test with live market data.
    --test offline    Runs test with fixed data.
HELP
}

# default action when no options are given
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

case $1 in
    --help)
        if [ $# -ne 1 ]; then
            show_help
            exit 1
        fi
        show_help
        exit 0
    ;;

    --test)
        if [ $# -ne 2 ] || [[ "$2" != "live" && "$2" != "offline" ]]; then
            echo "--test requires exactly one parameter: live or offline"
            show_help
            exit 1
        fi

        TEST_PATH="tests/risk-backtest.py"
        echo "Running tests..."
        echo "Filter: integration"
        echo "Path: $TEST_PATH"
        echo "Data: $2"
        TESTING=1 DATA_SOURCE="$2" python -B -m pytest -m integration "$TEST_PATH" --capture=no --log-cli-level=INFO
    ;;

    *)
        echo "Unknown option: $1"
        show_help
        exit 1
    ;;
esac
