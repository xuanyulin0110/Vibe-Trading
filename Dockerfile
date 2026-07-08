# ============================================================================
# Stage 1: Build frontend
# ============================================================================
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ============================================================================
# Stage 2: Python runtime
# ============================================================================
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Vibe-Trading" \
    org.opencontainers.image.description="Natural-language finance research AI agent with backtesting" \
    org.opencontainers.image.version="0.1.10" \
    org.opencontainers.image.source="https://github.com/HKUDS/Vibe-Trading" \
    org.opencontainers.image.licenses="MIT"

WORKDIR /app

# System deps
#   build-essential — compile any wheels without prebuilt manylinux artifacts.
#   The rest are weasyprint's runtime native libs (Pango/HarfBuzz/Fontconfig/
#   Cairo/gdk-pixbuf) per its official Debian install list; without them the
#   lazy `from weasyprint import HTML` in reporter.py fails and PDF rendering
#   silently downgrades to HTML-only. fonts-dejavu-core gives non-blank PDFs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    libfontconfig1 \
    libgdk-pixbuf-2.0-0 \
    libcairo2 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Python deps (install before copying code for layer caching)
COPY agent/requirements.txt agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Copy project
COPY pyproject.toml LICENSE README.md ./
COPY agent/ agent/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Install CLI entrypoint + the telegram channel extra -- python-telegram-bot
# is declared in pyproject.toml's optional "telegram"/"channels" groups, not
# the base dependency list, so a plain "pip install -e ." never pulls it in.
# Confirmed live 2026-07-08: the Telegram integration built and tested this
# session (notifier.py, chat_commands.py, ChannelRuntime's /deploy
# interception) was never actually reachable in this docker-compose
# deployment -- every boot logged "telegram channel not available:
# ModuleNotFoundError: No module named 'telegram'" and silently ran with
# "No channels enabled". Installing only the "telegram" extra (not the much
# heavier "channels" extra, which also pulls in Slack/Discord/WhatsApp/QQ/
# Matrix/DingTalk/Lark/WeCom clients nothing here configures) keeps the
# image lean while matching what agent/.env actually turns on.
RUN pip install --no-cache-dir -e ".[telegram]"

# Runtime should not run as root. Keep writable app data directories owned by
# the service user so named Docker volumes inherit usable permissions.
RUN useradd --create-home --shell /usr/sbin/nologin vibe \
    && mkdir -p agent/runs agent/sessions agent/uploads agent/.swarm/runs /home/vibe/.vibe-trading \
    && chown -R vibe:vibe /app /home/vibe/.vibe-trading
USER vibe

# Default port
EXPOSE 8899

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8899/health')" || exit 1

# Run API server (serves frontend/dist as static files)
CMD ["vibe-trading", "serve", "--host", "0.0.0.0", "--port", "8899"]
