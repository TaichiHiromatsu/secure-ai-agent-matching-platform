# モックエージェントの起動方法

## 前提条件

```bash
export GOOGLE_API_KEY="your-api-key"
```

## 各エージェントの起動

### ターミナル1: 仲介エージェント（ポート 8001）
```bash
cd /Users/taichihiromatsu/Development/secure-ai-agent-matching-platform
adk serve secure-mediation-agent/agent.py --port 8001
```

エージェントカード: http://localhost:8001/.well-known/agent.json

---

### ターミナル2: 航空会社エージェント（ポート 8002）
```bash
cd /Users/taichihiromatsu/Development/secure-ai-agent-matching-platform
adk serve external-agents/trusted-agents/airline-agent/agent.py --port 8002
```

エージェントカード: http://localhost:8002/.well-known/agent.json

---

### ターミナル3: ホテルエージェント（ポート 8003）
```bash
cd /Users/taichihiromatsu/Development/secure-ai-agent-matching-platform
adk serve external-agents/trusted-agents/hotel-agent/agent.py --port 8003
```

エージェントカード: http://localhost:8003/.well-known/agent.json

---

### ターミナル4: レンタカーエージェント（ポート 8004）
```bash
cd /Users/taichihiromatsu/Development/secure-ai-agent-matching-platform
adk serve external-agents/trusted-agents/car-rental-agent/agent.py --port 8004
```

エージェントカード: http://localhost:8004/.well-known/agent.json

---

## 確認方法

各エージェントが起動したら、ブラウザまたはcurlで確認：

```bash
# 仲介エージェント
curl http://localhost:8001/.well-known/agent.json

# 航空会社エージェント
curl http://localhost:8002/.well-known/agent.json

# ホテルエージェント
curl http://localhost:8003/.well-known/agent.json

# レンタカーエージェント
curl http://localhost:8004/.well-known/agent.json
```

## デモの実行

全てのエージェントが起動したら、沖縄旅行デモを実行：

```bash
python demo/okinawa_trip_demo.py
```
