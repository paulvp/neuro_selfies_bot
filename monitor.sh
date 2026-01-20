#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
MAX_LOG_SIZE_MB=10
MAX_RESTARTS_PER_HOUR=5
RESTART_DELAY=5
CLEANUP_INTERVAL=300  # 5 minutes

RESTART_COUNT=0
LAST_CLEANUP=0

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

cleanup_logs() {
    log "Cleaning up logs..."
    
    mkdir -p /app/logs
    
    find /app/logs -type f -name "*.log" -size +10M -exec sh -c '
        for file; do
            echo "Rotating large log: $file"
            mv "$file" "${file}.old" 2>/dev/null || true
            gzip "${file}.old" 2>/dev/null || true
        done
    ' sh {} + 2>/dev/null || true
    
    find /app/logs -type f -name "*.log.*.gz" 2>/dev/null | sort -r | tail -n +4 | xargs rm -f 2>/dev/null || true
    
    log "Log cleanup completed"
}

cleanup_temp() {
    log "Cleaning up temporary files..."
    
    mkdir -p /app/temp
    
    find /app/temp -type f -mmin +60 -delete 2>/dev/null || true
    
    find /app/temp -type d -empty -delete 2>/dev/null || true
    
    log "Temp cleanup completed"
}

check_restart_limit() {
    CURRENT_HOUR=$(date +%Y%m%d%H)
    
    if [ -f /tmp/restart_hour ]; then
        LAST_HOUR=$(cat /tmp/restart_hour)
        if [ "$CURRENT_HOUR" != "$LAST_HOUR" ]; then
            RESTART_COUNT=0
            echo "$CURRENT_HOUR" > /tmp/restart_hour
        fi
    else
        echo "$CURRENT_HOUR" > /tmp/restart_hour
    fi
    
    if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS_PER_HOUR" ]; then
        log_error "Too many restarts ($RESTART_COUNT) in the last hour. Waiting 10 minutes..."
        sleep 600
        RESTART_COUNT=0
    fi
}

periodic_cleanup() {
    CURRENT_TIME=$(date +%s)
    
    if [ $((CURRENT_TIME - LAST_CLEANUP)) -ge $CLEANUP_INTERVAL ]; then
        cleanup_logs
        cleanup_temp
        LAST_CLEANUP=$CURRENT_TIME
    fi
}

trap 'log "Received shutdown signal, cleaning up..."; cleanup_logs; cleanup_temp; exit 0' SIGTERM SIGINT

cleanup_logs
cleanup_temp

log "Starting bot monitor..."

while true; do
    check_restart_limit
    
    log "Starting bot (restart #$RESTART_COUNT)..."
    
    python /app/bot.py &
    BOT_PID=$!
    
    while kill -0 $BOT_PID 2>/dev/null; do
        sleep 30
        periodic_cleanup
    done
    
    wait $BOT_PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "Bot exited normally. Stopping monitor."
        cleanup_logs
        cleanup_temp
        break
    else
        log_error "Bot crashed with exit code $EXIT_CODE"
        RESTART_COUNT=$((RESTART_COUNT + 1))
        
        cleanup_temp
        
        log "Restarting in $RESTART_DELAY seconds..."
        sleep $RESTART_DELAY
    fi
done

log "Monitor script exiting"
