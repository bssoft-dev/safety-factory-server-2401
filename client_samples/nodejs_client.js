#!/usr/bin/env node
/**
 * Safety Server Node.js 클라이언트
 * 파일 재생, WebSocket 스트리밍
 */

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

class SafetyServerNodeClient {
    constructor(options = {}) {
        this.serverHost = options.host || 'localhost';
        this.serverPort = options.port || 24015;
        this.roomName = options.room || 'node테스트방';
        this.deviceId = options.device || 'nodejs-client';
        this.sampleRate = options.sampleRate || 16000;
        this.frameSize = 1024;
        
        this.wsUrl = `ws://${this.serverHost}:${this.serverPort}/ws/room/${this.roomName}/${this.sampleRate}/int16/${this.deviceId}`;
        
        this.websocket = null;
        this.isConnected = false;
        this.sentFrames = 0;
        this.receivedFrames = 0;
        this.startTime = null;
        
        console.log(`🎵 Safety Server Node.js Client`);
        console.log(`   WebSocket: ${this.wsUrl}`);
    }

    async createRoom() {
        const fetch = require('node-fetch');
        const createUrl = `http://${this.serverHost}:${this.serverPort}/v1/create_room`;
        
        try {
            const response = await fetch(createUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ room_name: this.roomName })
            });
            
            const result = await response.json();
            console.log(`✅ 방 생성: ${result.message}`);
        } catch (error) {
            console.log(`⚠️ 방 생성 오류: ${error.message}`);
        }
    }

    generateTestTone(durationMs = 64) {
        // 440Hz 사인파 생성
        const samples = Math.floor(this.sampleRate * durationMs / 1000);
        const buffer = Buffer.alloc(samples * 2); // 16-bit = 2 bytes
        
        for (let i = 0; i < samples; i++) {
            const sample = Math.sin(2 * Math.PI * 440 * i / this.sampleRate);
            const int16Value = Math.floor(sample * 16383);
            buffer.writeInt16LE(int16Value, i * 2);
        }
        
        return buffer;
    }

    async loadWavFile(filename) {
        try {
            if (!fs.existsSync(filename)) {
                console.log(`❌ 파일 없음: ${filename}`);
                return null;
            }
            
            const data = fs.readFileSync(filename);
            
            // 간단 WAV 헤더 파싱 (44바이트 건너뛰기)
            if (data.length < 44) {
                console.log(`❌ 유효하지 않은 WAV 파일`);
                return null;
            }
            
            const audioData = data.slice(44); // WAV 헤더 제거
            console.log(`📁 파일 로드: ${filename} (${audioData.length} 바이트)`);
            return audioData;
            
        } catch (error) {
            console.log(`❌ 파일 로드 실패: ${error.message}`);
            return null;
        }
    }

    connect() {
        return new Promise((resolve, reject) => {
            console.log(`🔗 연결 중...`);
            
            this.websocket = new WebSocket(this.wsUrl);
            
            this.websocket.on('open', () => {
                this.isConnected = true;
                this.startTime = Date.now();
                console.log('✅ WebSocket 연결 성공');
                
                // 통계 출력 타이머
                this.statsInterval = setInterval(() => {
                    this.printStats();
                }, 5000);
                
                resolve();
            });
            
            this.websocket.on('message', (data) => {
                this.receivedFrames++;
                // 수신 오디오 처리 (파일 저장 등)
            });
            
            this.websocket.on('error', (error) => {
                console.log(`❌ WebSocket 오류: ${error.message}`);
                reject(error);
            });
            
            this.websocket.on('close', () => {
                this.isConnected = false;
                console.log('🔌 연결 종료');
                
                if (this.statsInterval) {
                    clearInterval(this.statsInterval);
                }
            });
        });
    }

    async sendTestTone(durationSec = 10) {
        console.log(`🔊 테스트 톤 송신 (${durationSec}초)`);
        
        const intervalMs = 64; // 64ms 간격
        const totalFrames = Math.floor(durationSec * 1000 / intervalMs);
        
        for (let i = 0; i < totalFrames; i++) {
            if (!this.isConnected) break;
            
            const audioFrame = this.generateTestTone(intervalMs);
            this.websocket.send(audioFrame);
            this.sentFrames++;
            
            await this.sleep(intervalMs);
        }
    }

    async sendWavFile(filename) {
        const audioData = await this.loadWavFile(filename);
        if (!audioData) return;
        
        console.log(`🎵 파일 재생: ${filename}`);
        
        // 프레임 단위로 전송
        for (let i = 0; i < audioData.length; i += this.frameSize * 2) {
            if (!this.isConnected) break;
            
            const frameData = audioData.slice(i, i + this.frameSize * 2);
            if (frameData.length > 0) {
                this.websocket.send(frameData);
                this.sentFrames++;
            }
            
            await this.sleep(64); // 64ms 간격
        }
        
        console.log(`✅ 파일 전송 완료`);
    }

    printStats() {
        const elapsed = (Date.now() - this.startTime) / 1000;
        console.log(`📊 통계 (${elapsed.toFixed(1)}초): 송신=${this.sentFrames}, 수신=${this.receivedFrames}`);
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close();
        }
    }
}

// CLI 사용법
async function main() {
    const args = process.argv.slice(2);
    const options = {};
    
    // 간단한 인수 파싱
    for (let i = 0; i < args.length; i += 2) {
        const key = args[i]?.replace('--', '');
        const value = args[i + 1];
        if (key && value) {
            options[key] = value;
        }
    }
    
    const client = new SafetyServerNodeClient(options);
    
    try {
        // 방 생성
        await client.createRoom();
        
        // 연결
        await client.connect();
        
        // 모드에 따른 실행
        const mode = options.mode || 'test';
        
        if (mode === 'file' && options.file) {
            await client.sendWavFile(options.file);
        } else if (mode === 'test') {
            await client.sendTestTone(parseInt(options.duration) || 30);
        }
        
        // 잠시 대기 후 종료
        await client.sleep(2000);
        
    } catch (error) {
        console.log(`❌ 오류: ${error.message}`);
    } finally {
        client.disconnect();
        process.exit(0);
    }
}

// Ctrl+C 처리
process.on('SIGINT', () => {
    console.log('\n🛑 사용자 중단');
    process.exit(0);
});

if (require.main === module) {
    console.log('🚀 Safety Server Node.js Client');
    console.log('사용법:');
    console.log('  node nodejs_client.js --mode test --duration 10');
    console.log('  node nodejs_client.js --mode file --file audio.wav');
    console.log('  node nodejs_client.js --host 192.168.1.100 --room 테스트방');
    
    main().catch(console.error);
}

module.exports = SafetyServerNodeClient;
