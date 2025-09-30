# 🔒 HTTPS/WSS 보안 WebSocket 연결 해결 가이드

## 🚨 문제 상황
```
❌ 연결 실패: Failed to construct 'WebSocket': 
   An insecure WebSocket connection may not be initiated from a page loaded over HTTPS.
```

**원인**: HTTPS 페이지에서는 보안상 WSS(WebSocket Secure)만 사용 가능

## 💡 해결 방법 (우선순위별)

### 🥇 방법 1: HTTP 페이지로 접속 (즉시 해결)
```
❌ https://safety-server.bs-soft.co.kr:24015/webclient
✅ http://safety-server.bs-soft.co.kr:24015/webclient
```

**장점**: 즉시 사용 가능, 서버 설정 불필요
**단점**: 브라우저에서 "안전하지 않음" 경고

### 🥈 방법 2: 웹 클라이언트에서 프로토콜 강제 설정
1. **프로토콜 설정**: `WS (비보안)` 선택
2. **서버 주소**: `safety-server.bs-soft.co.kr` (HTTPS 제거)
3. **연결**: 🔗 연결 버튼 클릭

### 🥉 방법 3: Safety Server에 SSL 인증서 설정

#### A. Let's Encrypt 무료 SSL (권장)
```bash
# 1. Certbot 설치
sudo apt update
sudo apt install certbot python3-certbot-nginx

# 2. SSL 인증서 발급
sudo certbot certonly --standalone -d safety-server.bs-soft.co.kr

# 3. Safety Server를 HTTPS로 설정
# app.py 수정 (예시)
import ssl
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('/etc/letsencrypt/live/safety-server.bs-soft.co.kr/fullchain.pem',
                       '/etc/letsencrypt/live/safety-server.bs-soft.co.kr/privkey.pem')

# uvicorn 실행 시 SSL 적용
uvicorn app:app --host 0.0.0.0 --port 24015 --ssl-keyfile=/etc/letsencrypt/live/safety-server.bs-soft.co.kr/privkey.pem --ssl-certfile=/etc/letsencrypt/live/safety-server.bs-soft.co.kr/fullchain.pem
```

#### B. Nginx 리버스 프록시 (추천)
```nginx
# /etc/nginx/sites-available/safety-server
server {
    listen 443 ssl;
    server_name safety-server.bs-soft.co.kr;
    
    ssl_certificate /etc/letsencrypt/live/safety-server.bs-soft.co.kr/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/safety-server.bs-soft.co.kr/privkey.pem;
    
    # HTTP 페이지 프록시
    location / {
        proxy_pass http://127.0.0.1:24015;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket 프록시
    location /ws/ {
        proxy_pass http://127.0.0.1:24015;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# HTTP → HTTPS 리다이렉트
server {
    listen 80;
    server_name safety-server.bs-soft.co.kr;
    return 301 https://$server_name$request_uri;
}
```

## 🛠️ 웹 클라이언트 업데이트 사항

### ✅ 자동 프로토콜 감지
- **HTTPS 페이지**: 자동으로 WSS 사용
- **HTTP 페이지**: 자동으로 WS 사용

### ⚙️ 수동 프로토콜 선택
- **자동 감지**: 페이지 프로토콜에 따라 결정
- **WS (비보안)**: 강제로 ws:// 사용
- **WSS (보안)**: 강제로 wss:// 사용

### 🔧 설정 방법
1. 서버 설정 → 프로토콜 → `WS (비보안)` 선택
2. 서버 주소에서 `https://` 제거
3. 연결 시도

## 🎯 권장 운영 환경별 설정

### 🏠 로컬 테스트
```
페이지: http://localhost:24015/webclient
프로토콜: 자동 감지 (WS)
서버: localhost
```

### 🌐 인트라넷 환경
```
페이지: http://192.168.1.100:24015/webclient
프로토콜: WS (비보안)
서버: 192.168.1.100
```

### 🔒 인터넷 운영 환경
```
페이지: https://safety-server.bs-soft.co.kr/webclient
프로토콜: WSS (보안) - SSL 인증서 필요
서버: safety-server.bs-soft.co.kr
```

## 🚀 빠른 해결책 (지금 당장)

1. **브라우저 주소창에서**:
   ```
   https://safety-server.bs-soft.co.kr:24015/webclient
   ↓ 변경
   http://safety-server.bs-soft.co.kr:24015/webclient
   ```

2. **웹 클라이언트에서**:
   - 프로토콜: `WS (비보안)` 선택
   - 서버 주소: `safety-server.bs-soft.co.kr` (https 제거)
   - 🔗 연결 버튼 클릭

3. **브라우저 경고 무시**:
   - "안전하지 않음" 경고 → 계속 진행
   - 마이크 권한 → 허용

## 🔍 문제 진단 도구

### JavaScript 콘솔 확인
```javascript
// 현재 페이지 프로토콜 확인
console.log("페이지 프로토콜:", window.location.protocol);

// WebSocket 지원 확인
console.log("WebSocket 지원:", typeof WebSocket !== 'undefined');

// 보안 컨텍스트 확인
console.log("보안 컨텍스트:", window.isSecureContext);
```

### 네트워크 연결 테스트
```bash
# WebSocket 연결 테스트
wscat -c ws://safety-server.bs-soft.co.kr:24015/ws/room/테스트방/16000/int16/test

# 포트 접근 확인
telnet safety-server.bs-soft.co.kr 24015
```

---
**💡 요약**: HTTPS 페이지에서는 HTTP 페이지로 변경하거나 SSL 인증서 설정이 필요합니다.
