// ========== 클라이언트 측 WebSocket 연결 수정 가이드 ==========

// ❌ 잘못된 URL (현재)
const wrongUrl = "wss://safety-server.bs-soft.co.kr/ws/room/보온팀";

// ✅ 올바른 URL (수정 후)
const correctUrl = "wss://safety-server.bs-soft.co.kr/ws/room/보온팀/16000/float32/default";

// ========== React/TypeScript 클라이언트 수정 예시 ==========

class WebSocketAudioClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
    }

    // 올바른 WebSocket URL 생성 함수
    buildWebSocketUrl(roomName, sampleRate = 16000, dataType = 'float32', deviceId = 'default') {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = 'safety-server.bs-soft.co.kr';
        
        // URL 인코딩 처리
        const encodedRoomName = encodeURIComponent(roomName);
        const encodedDeviceId = encodeURIComponent(deviceId);
        
        return `${protocol}//${host}/ws/room/${encodedRoomName}/${sampleRate}/${dataType}/${encodedDeviceId}`;
    }

    async connect(audioSettings) {
        try {
            // 올바른 URL 구성
            const wsUrl = this.buildWebSocketUrl(
                audioSettings.selectedChannel.room_name,  // 방 이름
                audioSettings.sampleRate,                 // 샘플레이트
                'float32',                               // 데이터 타입
                audioSettings.deviceId || 'default'      // 디바이스 ID
            );

            console.log('WebSocket 연결 시도:', {
                audioSettings,
                wsUrl
            });

            // WebSocket 연결
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                this.isConnected = true;
                console.log('✅ WebSocket 연결 성공');
            };

            this.websocket.onmessage = (event) => {
                // 오디오 데이터 수신 처리
                this.handleAudioMessage(event.data);
            };

            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket 오류:', error);
            };

            this.websocket.onclose = () => {
                this.isConnected = false;
                console.log('🔌 WebSocket 연결 종료');
            };

        } catch (error) {
            console.error('WebSocket 연결 실패:', error);
        }
    }

    handleAudioMessage(audioData) {
        // 오디오 데이터 처리 로직
        console.log('오디오 데이터 수신:', audioData.byteLength, 'bytes');
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
}

// ========== 사용 예시 ==========

// 클라이언트에서 사용할 때
const client = new WebSocketAudioClient();

const audioSettings = {
    deviceId: "default",
    muted: false,
    sampleRate: 16000,
    volume: 50,
    selectedChannel: {
        persons: '강덕준',
        id: 802,
        room_name: '보온팀',
        num_person: 2,
        message: ''
    }
};

// 연결
client.connect(audioSettings);

// ========== 추가 디버깅 정보 ==========

// 서버 측에서 지원하는 경로 확인
const supportedPaths = [
    '/ws/room/{room_name}/{sr}/{dtype}/{device_id}',
    '예시: /ws/room/보온팀/16000/float32/default',
    '예시: /ws/room/테스트방/48000/int16/webclient-001'
];

console.log('지원되는 WebSocket 경로:', supportedPaths);

// ========== 오류 해결 체크리스트 ==========

/*
1. ✅ URL 파라미터 확인
   - room_name: 방 이름 (필수)
   - sr: 샘플레이트 (16000 또는 48000)
   - dtype: 데이터 타입 (int16 또는 float32)
   - device_id: 디바이스 ID (필수)

2. ✅ URL 인코딩 확인
   - 한글 방 이름은 encodeURIComponent() 사용
   - 특수문자 포함된 디바이스 ID도 인코딩

3. ✅ 프로토콜 확인
   - HTTPS 페이지에서는 WSS 사용
   - HTTP 페이지에서는 WS 사용

4. ✅ 서버 상태 확인
   - 서버가 실행 중인지 확인
   - 방이 존재하는지 확인

5. ✅ 방 생성 확인
   - WebSocket 연결 전에 방이 생성되어 있어야 함
   - POST /v1/create_room API 호출 필요
*/

// ========== 방 생성 API 호출 예시 ==========

async function createRoomIfNeeded(roomName) {
    try {
        const response = await fetch(`https://safety-server.bs-soft.co.kr/v1/create_room`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                room_name: roomName
            })
        });

        if (response.ok) {
            console.log('✅ 방 생성 성공:', roomName);
            return true;
        } else {
            console.log('⚠️ 방이 이미 존재하거나 생성 실패');
            return false;
        }
    } catch (error) {
        console.error('❌ 방 생성 오류:', error);
        return false;
    }
}

// 사용 예시
async function connectToRoom(roomName) {
    // 1. 방 생성 (필요시)
    await createRoomIfNeeded(roomName);
    
    // 2. WebSocket 연결
    const client = new WebSocketAudioClient();
    await client.connect({
        selectedChannel: { room_name: roomName },
        sampleRate: 16000,
        deviceId: 'webclient-001'
    });
} 