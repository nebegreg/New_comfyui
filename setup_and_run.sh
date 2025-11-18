#!/bin/bash
#
# 🏔️ Mountain Studio ULTIMATE v3.0 - Setup & Launch
# ==================================================
#
# Script tout-en-un pour:
# 1. Vérifier installation ComfyUI
# 2. Auto-installer modèles & nodes si nécessaire
# 3. Lancer ComfyUI en arrière-plan
# 4. Lancer Mountain Studio
#

echo "🏔️  Mountain Studio ULTIMATE v3.0"
echo "===================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if ComfyUI is installed
echo "🔍 Checking for ComfyUI installation..."

if [ -d "../ComfyUI" ]; then
    COMFYUI_PATH="../ComfyUI"
    echo -e "${GREEN}✅ ComfyUI found: $COMFYUI_PATH${NC}"
elif [ -d "$HOME/ComfyUI" ]; then
    COMFYUI_PATH="$HOME/ComfyUI"
    echo -e "${GREEN}✅ ComfyUI found: $COMFYUI_PATH${NC}"
else
    echo -e "${YELLOW}⚠️  ComfyUI not found in standard locations${NC}"
    echo ""
    echo "Please install ComfyUI first:"
    echo "  git clone https://github.com/comfyanonymous/ComfyUI.git"
    echo ""
    echo "Or specify path:"
    read -p "Enter ComfyUI path (or press Enter to skip): " CUSTOM_PATH

    if [ -n "$CUSTOM_PATH" ]; then
        COMFYUI_PATH="$CUSTOM_PATH"
    else
        echo -e "${YELLOW}⚠️  Skipping ComfyUI setup. AI textures will not be available.${NC}"
        COMFYUI_PATH=""
    fi
fi

# Run auto-setup if ComfyUI is found
if [ -n "$COMFYUI_PATH" ]; then
    echo ""
    echo "🔧 Checking installation status..."
    python3 comfyui_auto_setup.py --comfyui-path "$COMFYUI_PATH" --check-only

    echo ""
    read -p "Run auto-setup to install missing components? (y/N): " RUN_SETUP

    if [[ "$RUN_SETUP" =~ ^[Yy]$ ]]; then
        echo ""
        echo "📦 Running auto-setup..."
        echo "⚠️  This will download ~7 GB of models. Continue? (y/N)"
        read CONFIRM_DOWNLOAD

        if [[ "$CONFIRM_DOWNLOAD" =~ ^[Yy]$ ]]; then
            python3 comfyui_auto_setup.py --comfyui-path "$COMFYUI_PATH"
        else
            echo "Skipping model download. You can run it later with:"
            echo "  python3 comfyui_auto_setup.py --comfyui-path $COMFYUI_PATH"
        fi
    fi

    # Check if ComfyUI is already running
    echo ""
    echo "🔍 Checking if ComfyUI is running..."

    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ComfyUI is already running!${NC}"
    else
        echo -e "${YELLOW}⚠️  ComfyUI not running${NC}"
        read -p "Start ComfyUI server in background? (Y/n): " START_COMFYUI

        if [[ ! "$START_COMFYUI" =~ ^[Nn]$ ]]; then
            echo "🚀 Starting ComfyUI..."
            cd "$COMFYUI_PATH"
            python main.py > /tmp/comfyui.log 2>&1 &
            COMFYUI_PID=$!
            echo "   PID: $COMFYUI_PID"
            echo "   Logs: /tmp/comfyui.log"

            # Wait for ComfyUI to start
            echo "   Waiting for server to start..."
            for i in {1..30}; do
                if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
                    echo -e "${GREEN}   ✅ ComfyUI started successfully!${NC}"
                    break
                fi
                sleep 1
                echo -n "."
            done
            echo ""

            cd - > /dev/null
        fi
    fi
fi

# Launch Mountain Studio
echo ""
echo "🏔️  Launching Mountain Studio ULTIMATE v3.0..."
echo ""

python3 mountain_studio_ultimate_v3.py

echo ""
echo "Application closed."

# Offer to stop ComfyUI if we started it
if [ -n "$COMFYUI_PID" ]; then
    echo ""
    read -p "Stop ComfyUI server? (y/N): " STOP_COMFYUI

    if [[ "$STOP_COMFYUI" =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping ComfyUI (PID: $COMFYUI_PID)..."
        kill $COMFYUI_PID 2>/dev/null
        echo "✅ ComfyUI stopped."
    else
        echo "ComfyUI still running. To stop later:"
        echo "  kill $COMFYUI_PID"
    fi
fi

echo ""
echo "👋 Thank you for using Mountain Studio!"
