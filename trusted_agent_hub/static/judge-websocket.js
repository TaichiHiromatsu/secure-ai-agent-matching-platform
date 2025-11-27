/**
 * WebSocket Client for Collaborative Jury Judge Real-time Updates
 *
 * Handles real-time communication with the jury judge evaluation backend
 * and updates the UI with progress, phases, and discussion results.
 */

class JuryJudgeWebSocket {
    constructor(submissionId, options = {}) {
        this.submissionId = submissionId;
        this.options = {
            reconnectInterval: 3000,
            maxReconnectAttempts: 10,
            heartbeatInterval: 25000,
            ...options
        };

        this.ws = null;
        this.reconnectAttempts = 0;
        this.heartbeatTimer = null;
        this.isConnected = false;
        this.eventHandlers = {};

        // Status tracking
        this.currentPhase = null;
        this.currentRound = 0;
        this.jurorEvaluations = {};
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/submissions/${this.submissionId}/judge`;

        console.log(`[JuryJudgeWS] Connecting to ${wsUrl}`);

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => this.handleOpen();
            this.ws.onmessage = (event) => this.handleMessage(event);
            this.ws.onerror = (error) => this.handleError(error);
            this.ws.onclose = () => this.handleClose();

        } catch (error) {
            console.error('[JuryJudgeWS] Connection error:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket open event
     */
    handleOpen() {
        console.log('[JuryJudgeWS] Connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emit('connected');

        // Start heartbeat
        this.startHeartbeat();
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('[JuryJudgeWS] Message received:', data);

            // Handle heartbeat responses
            if (event.data === 'pong') {
                return;
            }

            // Route message by type
            const { type } = data;

            switch (type) {
                case 'phase_change':
                    this.handlePhaseChange(data);
                    break;
                case 'juror_evaluation':
                    this.handleJurorEvaluation(data);
                    break;
                case 'consensus_check':
                    this.handleConsensusCheck(data);
                    break;
                case 'discussion_start':
                    this.handleDiscussionStart(data);
                    break;
                case 'juror_statement':
                    this.handleJurorStatement(data);
                    break;
                case 'round_complete':
                    this.handleRoundComplete(data);
                    break;
                case 'final_judgment':
                    this.handleFinalJudgment(data);
                    break;
                case 'error':
                    this.handleBackendError(data);
                    break;
                default:
                    console.warn('[JuryJudgeWS] Unknown message type:', type);
                    this.emit('message', data);
            }

        } catch (error) {
            console.error('[JuryJudgeWS] Message parsing error:', error);
        }
    }

    /**
     * Handle phase change (Phase 1: Independent, Phase 2: Discussion, Phase 3: Final)
     */
    handlePhaseChange(data) {
        this.currentPhase = data.phase;
        this.emit('phase_change', {
            phase: data.phase,
            phaseNumber: data.phaseNumber,
            description: data.description
        });
    }

    /**
     * Handle individual juror evaluation
     */
    handleJurorEvaluation(data) {
        const { juror, phase, verdict, score, confidence, rationale } = data;

        if (!this.jurorEvaluations[juror]) {
            this.jurorEvaluations[juror] = {};
        }

        this.jurorEvaluations[juror][phase] = {
            verdict,
            score,
            confidence,
            rationale,
            timestamp: new Date()
        };

        this.emit('juror_evaluation', {
            juror,
            phase,
            verdict,
            score,
            confidence,
            rationale
        });
    }

    /**
     * Handle consensus check result
     */
    handleConsensusCheck(data) {
        this.emit('consensus_check', {
            consensusStatus: data.consensusStatus,
            consensusReached: data.consensusReached,
            consensusVerdict: data.consensusVerdict,
            round: data.round
        });
    }

    /**
     * Handle discussion start
     */
    handleDiscussionStart(data) {
        this.currentRound = data.round;
        this.emit('discussion_start', {
            round: data.round,
            speakerOrder: data.speakerOrder
        });
    }

    /**
     * Handle juror statement in discussion
     */
    handleJurorStatement(data) {
        this.emit('juror_statement', {
            round: data.round,
            juror: data.juror,
            statement: data.statement,
            positionChanged: data.positionChanged,
            newVerdict: data.newVerdict,
            newScore: data.newScore
        });
    }

    /**
     * Handle round completion
     */
    handleRoundComplete(data) {
        this.emit('round_complete', {
            round: data.round,
            consensusReached: data.consensusReached,
            stagnant: data.stagnant
        });
    }

    /**
     * Handle final judgment
     */
    handleFinalJudgment(data) {
        this.emit('final_judgment', {
            method: data.method,
            finalVerdict: data.finalVerdict,
            finalScore: data.finalScore,
            confidence: data.confidence,
            rationale: data.rationale,
            scoreBreakdown: data.scoreBreakdown
        });

        // Evaluation complete
        this.emit('evaluation_complete', data);
    }

    /**
     * Handle backend error
     */
    handleBackendError(data) {
        console.error('[JuryJudgeWS] Backend error:', data.message);
        this.emit('error', {
            message: data.message,
            details: data.details
        });
    }

    /**
     * Handle WebSocket error
     */
    handleError(error) {
        console.error('[JuryJudgeWS] WebSocket error:', error);
        this.emit('connection_error', error);
    }

    /**
     * Handle WebSocket close
     */
    handleClose() {
        console.log('[JuryJudgeWS] Connection closed');
        this.isConnected = false;
        this.stopHeartbeat();
        this.emit('disconnected');

        // Attempt reconnection
        this.scheduleReconnect();
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempts < this.options.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[JuryJudgeWS] Reconnecting in ${this.options.reconnectInterval}ms (attempt ${this.reconnectAttempts}/${this.options.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connect();
            }, this.options.reconnectInterval);
        } else {
            console.error('[JuryJudgeWS] Max reconnect attempts reached');
            this.emit('reconnect_failed');
        }
    }

    /**
     * Start heartbeat to keep connection alive
     */
    startHeartbeat() {
        this.stopHeartbeat();

        this.heartbeatTimer = setInterval(() => {
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, this.options.heartbeatInterval);
    }

    /**
     * Stop heartbeat timer
     */
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }

    /**
     * Register event handler
     */
    on(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
    }

    /**
     * Unregister event handler
     */
    off(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
        }
    }

    /**
     * Emit event to all registered handlers
     */
    emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`[JuryJudgeWS] Error in event handler for '${event}':`, error);
                }
            });
        }
    }

    /**
     * Disconnect from WebSocket
     */
    disconnect() {
        console.log('[JuryJudgeWS] Disconnecting');
        this.stopHeartbeat();

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.isConnected = false;
        this.reconnectAttempts = this.options.maxReconnectAttempts; // Prevent auto-reconnect
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            connected: this.isConnected,
            phase: this.currentPhase,
            round: this.currentRound,
            jurorEvaluations: this.jurorEvaluations
        };
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JuryJudgeWebSocket;
}
