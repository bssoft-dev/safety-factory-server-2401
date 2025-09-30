# 🔒 Mixed Content 오류 해결 가이드

## 🚨 오류 상황
```
Mixed Content: The page at 'https://safety-server.bs-soft.co.kr/webclient' 
was loaded over HTTPS, but requested an insecure resource 
'http://safety-server.bs-soft.co.kr:24015/v1/create_room'. 
This request has been blocked; the content must be served over HTTPS.
```

## 🔍 원인 분석

### Mixed Content란?
- **HTTPS 페이지**에서 **HTTP 리소스**에 접근하려 할 때 발생
- 브라우저가 보안상 HTTP API 호출을 차단
- WebSocket(WS)과 API(HTTP) 모두 동일한 프로토콜 사용 필요

### 발생 시나리오
1. `https://safety-server.bs-soft.co.kr/webclient` 페이지 접속
2. 웹 클라이언트에서 `http://safety-server.bs-soft.co.kr:24015/v1/create_room` API 호출
3. 브라우저가 Mixed Content로 차단

## 🛠️ 해결 방법 (우선순위별)

### 🥇 방법 1: HTTP 페이지로 접속 (즉시 해결)
```
❌ https://safety-server.bs-soft.co.kr/webclient
✅ http://safety-server.bs-soft.co.kr:24015/webclient
```

**장점**: 즉시 사용 가능
**단점**: "안전하지 않음" 경고

### 🥈 방법 2: 웹 클라이언트 설정 변경
1. **HTTPS 페이지에서**:
   - API 프로토콜: `HTTPS` 선택
   - WS 프로토콜: `WSS (보안)` 선택

2. **HTTP 페이지에서**:
   - API 프로토콜: `HTTP` 선택 (또는 자동)
   - WS 프로토콜: `WS (비보안)` 선택 (또는 자동)

### 🥉 방법 3: Nginx 리버스 프록시 설정

#### Nginx 설정 예시
```nginx
# /etc/nginx/sites-available/safety-server-ssl
server {
    listen 443 ssl;
    server_name safety-server.bs-soft.co.kr;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # 웹 클라이언트 페이지
    location /webclient {
        proxy_pass http://127.0.0.1:24015;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API 엔드포인트
    location /v1/ {
        proxy_pass http://127.0.0.1:24015;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket 연결
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

#### 설정 적용
```bash
sudo ln -s /etc/nginx/sites-available/safety-server-ssl /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 🏆 방법 4: Safety Server에 SSL 직접 설정

#### uvicorn SSL 설정
```bash
# SSL 인증서 준비 (Let's Encrypt)
sudo certbot certonly --standalone -d safety-server.bs-soft.co.kr

# Safety Server SSL 실행
uvicorn app:app \
  --host 0.0.0.0 \
  --port 24015 \
  --ssl-keyfile=/etc/letsencrypt/live/safety-server.bs-soft.co.kr/privkey.pem \
  --ssl-certfile=/etc/letsencrypt/live/safety-server.bs-soft.co.kr/fullchain.pem
```

## 🎯 웹 클라이언트 업데이트 사항

### ✅ 새로운 기능
- **자동 프로토콜 감지**: 페이지 프로토콜에 따라 자동 설정
- **API 프로토콜 선택**: HTTP/HTTPS 수동 선택 가능
- **페이지 상태 표시**: 현재 환경과 권장 설정 안내
- **상세한 오류 메시지**: Mixed Content 오류 시 해결책 제시

### 🔧 설정 가이드

#### HTTPS 환경에서
```
페이지: https://safety-server.bs-soft.co.kr/webclient
API 프로토콜: HTTPS
WS 프로토콜: WSS (보안)
서버 주소: safety-server.bs-soft.co.kr
```

#### HTTP 환경에서
```
페이지: http://safety-server.bs-soft.co.kr:24015/webclient
API 프로토콜: 페이지와 동일 (또는 HTTP)
WS 프로토콜: 자동 감지 (또는 WS)
서버 주소: safety-server.bs-soft.co.kr
```

## 🚀 빠른 해결책 (지금 당장)

### 1단계: HTTP 페이지로 접속
```
브라우저 주소창:
https://safety-server.bs-soft.co.kr/webclient
↓ 변경
http://safety-server.bs-soft.co.kr:24015/webclient
```

### 2단계: 설정 확인
- **페이지 상태**: 🟢 HTTP 환경 확인
- **API 프로토콜**: "페이지와 동일" 선택
- **WS 프로토콜**: "자동 감지" 선택

### 3단계: 방 생성 및 연결
1. 🏠 방 생성 버튼 클릭 → ✅ 성공 확인
2. 🔗 연결 버튼 클릭 → ✅ WebSocket 연결 성공

## 🔍 문제 진단

### 브라우저 개발자 도구 확인
```javascript
// 콘솔에서 실행
console.log("페이지 프로토콜:", window.location.protocol);
console.log("보안 컨텍스트:", window.isSecureContext);
console.log("Mixed Content 정책:", 
  window.location.protocol === 'https:' ? 'HTTPS 필요' : 'HTTP 허용');
```

### 네트워크 탭 확인
- Mixed Content 차단: 🔴 빨간색 표시
- CORS 오류: 다른 도메인 접근 시
- SSL 인증서 오류: HTTPS 설정 문제

## 📊 프로토콜 조합표

| 페이지 프로토콜 | API 프로토콜 | WS 프로토콜 | 상태 |
|---|---|---|---|
| HTTPS | HTTPS | WSS | ✅ 완전 보안 |
| HTTPS | HTTP | WS | ❌ Mixed Content |
| HTTP | HTTP | WS | ✅ 정상 작동 |
| HTTP | HTTPS | WSS | ⚠️ 작동하지만 불필요 |

---
**💡 요약**: HTTPS 페이지에서는 모든 리소스가 HTTPS여야 하며, HTTP 페이지로 접속하면 즉시 해결됩니다.
