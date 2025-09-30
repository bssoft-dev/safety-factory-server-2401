import aiohttp
import os
import wave
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from sqlmodel import Session, select
from database import engine
from models import Rooms, SttRecords
import glob

# BSM 연동을 위한 import 추가
from services.bsm_api_agent import bsm_agent


class SttProcessor:
    def __init__(self):
        self.infer_url = "https://stt.bs-soft.co.kr/v2401/stt/realtime/byte"
        self.byte_headers = {"Content-Type": "application/octet-stream"}
        self.text_remain_counter = 0
        self.recordings_dir = "recordings"
        
        # 오디오 버퍼링 설정
        self.sample_rate = 16000
        self.buffer_duration = 10  # 10초 버퍼
        self.min_audio_duration = 1  # 최소 1초 이상의 오디오만 저장
        self.silence_threshold = 500  # 무음 임계값 (int16 기준)
        
        # 클라이언트별 오디오 버퍼 {client_id: deque}
        self.audio_buffers = {}
        # 클라이언트별 마지막 활동 시간 {client_id: timestamp}
        self.last_activity = {}
        
        # STT 결과와 음원 파일 매칭을 위한 설정
        self.time_tolerance = 60  # 60초 이내의 음원 파일을 찾음
        
        # 녹음 파일 저장 디렉토리 생성
        os.makedirs(self.recordings_dir, exist_ok=True)

    def init_client_buffer(self, client_id: int):
        """클라이언트별 오디오 버퍼 초기화"""
        buffer_size = self.sample_rate * self.buffer_duration  # 10초 버퍼
        self.audio_buffers[client_id] = deque(maxlen=buffer_size)
        self.last_activity[client_id] = datetime.now()

    def add_audio_to_buffer(self, client_id: int, audio_data: np.ndarray):
        """오디오 데이터를 버퍼에 추가"""
        if client_id not in self.audio_buffers:
            self.init_client_buffer(client_id)
        
        # 오디오 데이터를 버퍼에 추가
        for sample in audio_data:
            self.audio_buffers[client_id].append(sample)
        
        # 음성 활동 감지 (무음이 아닌 경우)
        if np.max(np.abs(audio_data)) > self.silence_threshold:
            self.last_activity[client_id] = datetime.now()

    def get_relevant_audio(self, client_id: int, stt_timestamp: datetime) -> np.ndarray:
        """STT 결과 시점을 기준으로 관련 오디오 추출"""
        if client_id not in self.audio_buffers:
            return np.array([], dtype=np.int16)
        
        buffer = list(self.audio_buffers[client_id])
        if len(buffer) == 0:
            return np.array([], dtype=np.int16)
        
        # 현재 시간에서 역산하여 적절한 구간 추출
        # STT는 보통 발화 후 1-3초 지연되므로 최근 3-8초 구간을 추출
        end_samples = len(buffer)
        start_samples = max(0, end_samples - (self.sample_rate * 8))  # 최대 8초
        min_samples = max(0, end_samples - (self.sample_rate * 3))   # 최소 3초
        
        # 실제 음성이 있는 구간 찾기
        audio_segment = np.array(buffer[start_samples:end_samples], dtype=np.int16)
        
        # 음성 활동 구간 감지
        voice_segments = self.detect_voice_segments(audio_segment)
        
        if len(voice_segments) > 0:
            # 가장 최근의 음성 구간 선택
            last_segment = voice_segments[-1]
            start_idx = max(0, last_segment[0] - self.sample_rate // 2)  # 0.5초 여유
            end_idx = min(len(audio_segment), last_segment[1] + self.sample_rate // 2)
            return audio_segment[start_idx:end_idx]
        
        # 음성 구간을 찾지 못한 경우 최소 구간 반환
        return audio_segment[min_samples - start_samples:]

    def detect_voice_segments(self, audio: np.ndarray) -> list:
        """음성 구간 감지"""
        if len(audio) == 0:
            return []
        
        # 음성 활동 감지 (간단한 에너지 기반)
        frame_size = self.sample_rate // 10  # 100ms 프레임
        voice_segments = []
        current_segment = None
        
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            energy = np.mean(np.abs(frame))
            
            if energy > self.silence_threshold:
                if current_segment is None:
                    current_segment = [i, i + frame_size]
                else:
                    current_segment[1] = i + frame_size
            else:
                if current_segment is not None:
                    # 최소 길이 체크
                    duration = (current_segment[1] - current_segment[0]) / self.sample_rate
                    if duration >= self.min_audio_duration:
                        voice_segments.append(current_segment)
                    current_segment = None
        
        # 마지막 세그먼트 처리
        if current_segment is not None:
            duration = (current_segment[1] - current_segment[0]) / self.sample_rate
            if duration >= self.min_audio_duration:
                voice_segments.append(current_segment)
        
        return voice_segments

    def save_wav_file(self, audio_data: np.ndarray, room_name: str, worker: str, timestamp: datetime) -> str:
        """오디오 데이터를 WAV 파일로 저장"""
        try:
            if len(audio_data) == 0:
                return ""
            
            # 파일명 생성: room_worker_timestamp.wav
            filename = f"{room_name}_{worker}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.wav"
            filepath = os.path.join(self.recordings_dir, filename)
            
            # WAV 파일로 저장
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # 모노
                wav_file.setsampwidth(2)  # 16비트
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            duration = len(audio_data) / self.sample_rate
            print(f"WAV 파일 저장: {filename} ({duration:.2f}초)")
            return filepath
        except Exception as e:
            print(f"WAV 파일 저장 오류: {e}")
            return ""

    def find_matching_audio_files(self, room_name: str, worker: str, stt_timestamp: datetime) -> dict:
        """STT 결과 시간에 맞는 원본/처리 후 음원 파일 찾기"""
        try:
            # 날짜 폴더 경로
            date_str = stt_timestamp.strftime('%Y-%m-%d')
            room_dir = os.path.join(self.recordings_dir, date_str, room_name)
            
            if not os.path.exists(room_dir):
                print(f"방 디렉토리가 존재하지 않음: {room_dir}")
                return {"input": "", "output": ""}
            
            # 해당 발화자의 input/output 파일들 찾기
            input_pattern = os.path.join(room_dir, f"input_*_{worker}.wav")
            output_pattern = os.path.join(room_dir, f"output_*_{worker}.wav")
            
            input_files = glob.glob(input_pattern)
            output_files = glob.glob(output_pattern)
            
            if not input_files and not output_files:
                print(f"발화자 {worker}의 음원 파일을 찾을 수 없음")
                return {"input": "", "output": ""}
            
            # STT 시간과 가장 가까운 파일들 찾기
            best_input = ""
            best_output = ""
            min_time_diff = float('inf')
            
            # input 파일들 검사
            for input_file in input_files:
                filename = os.path.basename(input_file)
                time_part = filename.split('_')[1]  # 07h57m59s
                
                try:
                    # 시간 파싱
                    hour = int(time_part.split('h')[0])
                    minute = int(time_part.split('h')[1].split('m')[0])
                    second = int(time_part.split('m')[1].split('s')[0])
                    
                    # 파일 시간 생성
                    file_time = stt_timestamp.replace(hour=hour, minute=minute, second=second, microsecond=0)
                    
                    # 시간 차이 계산
                    time_diff = abs((stt_timestamp - file_time).total_seconds())
                    
                    # 허용 범위 내에서 가장 가까운 파일 선택
                    if time_diff <= self.time_tolerance and time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_input = input_file
                        
                        # 해당 시간의 output 파일도 찾기
                        output_filename = f"output_{time_part}_{worker}.wav"
                        output_file = os.path.join(room_dir, output_filename)
                        if os.path.exists(output_file):
                            best_output = output_file
                        else:
                            best_output = ""
                        
                except (ValueError, IndexError) as e:
                    print(f"파일명 시간 파싱 오류: {filename}, {e}")
                    continue
            
            # output 파일이 없으면 input만 반환
            if best_input:
                print(f"매칭된 음원 파일: input={best_input}, output={best_output} (시간차: {min_time_diff:.1f}초)")
                return {
                    "input": best_input,
                    "output": best_output
                }
            else:
                print(f"STT 시간 {stt_timestamp}에 맞는 음원 파일을 찾을 수 없음")
                return {"input": "", "output": ""}
                
        except Exception as e:
            print(f"음원 파일 검색 오류: {e}")
            return {"input": "", "output": ""}

    def find_recent_audio_files(self, room_name: str, worker: str, stt_timestamp: datetime, 
                               time_window: int = 120) -> list:
        """STT 시간 전후 일정 시간 내의 모든 음원 파일 찾기 (원본/처리 후 모두)"""
        try:
            date_str = stt_timestamp.strftime('%Y-%m-%d')
            room_dir = os.path.join(self.recordings_dir, date_str, room_name)
            
            if not os.path.exists(room_dir):
                return []
            
            # 해당 발화자의 input/output 파일들 찾기
            input_pattern = os.path.join(room_dir, f"input_*_{worker}.wav")
            output_pattern = os.path.join(room_dir, f"output_*_{worker}.wav")
            
            input_files = glob.glob(input_pattern)
            output_files = glob.glob(output_pattern)
            
            matching_files = []
            stt_time = stt_timestamp.replace(microsecond=0)
            
            # input 파일들 처리
            for input_file in input_files:
                filename = os.path.basename(input_file)
                time_part = filename.split('_')[1]
                
                try:
                    hour = int(time_part.split('h')[0])
                    minute = int(time_part.split('h')[1].split('m')[0])
                    second = int(time_part.split('m')[1].split('s')[0])
                    
                    file_time = stt_timestamp.replace(hour=hour, minute=minute, second=second, microsecond=0)
                    time_diff = abs((stt_time - file_time).total_seconds())
                    
                    if time_diff <= time_window:
                        # 해당하는 output 파일 찾기
                        output_filename = f"output_{time_part}_{worker}.wav"
                        output_file = os.path.join(room_dir, output_filename)
                        output_exists = os.path.exists(output_file)
                        
                        matching_files.append({
                            'input_file': input_file,
                            'output_file': output_file if output_exists else "",
                            'time': file_time,
                            'time_diff': time_diff,
                            'type': 'input'
                        })
                        
                except (ValueError, IndexError) as e:
                    continue
            
            # output 파일들 처리 (input에 없는 것들)
            for output_file in output_files:
                filename = os.path.basename(output_file)
                time_part = filename.split('_')[1]
                
                try:
                    hour = int(time_part.split('h')[0])
                    minute = int(time_part.split('h')[1].split('m')[0])
                    second = int(time_part.split('m')[1].split('s')[0])
                    
                    file_time = stt_timestamp.replace(hour=hour, minute=minute, second=second, microsecond=0)
                    time_diff = abs((stt_time - file_time).total_seconds())
                    
                    if time_diff <= time_window:
                        # 이미 input에서 처리된 것인지 확인
                        existing = any(f['time'] == file_time for f in matching_files)
                        if not existing:
                            matching_files.append({
                                'input_file': "",
                                'output_file': output_file,
                                'time': file_time,
                                'time_diff': time_diff,
                                'type': 'output_only'
                            })
                        
                except (ValueError, IndexError) as e:
                    continue
            
            # 시간순 정렬
            matching_files.sort(key=lambda x: x['time_diff'])
            return matching_files
            
        except Exception as e:
            print(f"최근 음원 파일 검색 오류: {e}")
            return []

    async def save_stt_record(self, room_name: str, worker: str, stt_text: str, 
                            input_wav_path: str, output_wav_path: str, timestamp: datetime):
        """STT 결과를 데이터베이스에 저장 (원본/처리 후 음원 경로 모두 저장)"""
        try:
            # 원본과 처리 후 음원 경로를 구분자로 결합
            wav_file_path = f"{input_wav_path}|{output_wav_path}" if input_wav_path and output_wav_path else input_wav_path or output_wav_path
            
            with Session(engine) as session:
                stt_record = SttRecords(
                    room_name=room_name,
                    worker=worker,
                    stt_text=stt_text,
                    wav_file_path=wav_file_path,
                    created_at=timestamp
                )
                session.add(stt_record)
                session.commit()
                print(f"STT 기록 저장 완료: {room_name} - {worker}: {stt_text}")
                print(f"  원본: {input_wav_path}")
                print(f"  처리후: {output_wav_path}")
        except Exception as e:
            print(f"STT 기록 저장 오류: {e}")

    async def send_audio(self, audio_data: bytes, room_name: str, worker: str = "unknown", client_id: int = None):
        """STT 처리 및 결과 저장"""
        timestamp = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(f"{self.infer_url}/{room_name}", data=audio_data, headers=self.byte_headers) as res:
                    if res.status == 200:
                        data = await res.json()
                        if data["result"] != "":
                            print(f"STT: {data}")
                            self.text_remain_counter = min(len(data["result"])//5, 15)
                            
                            # STT 결과에 맞는 원본/처리 후 음원 파일 찾기
                            audio_files = self.find_matching_audio_files(room_name, worker, timestamp)
                            
                            if audio_files["input"] or audio_files["output"]:
                                # 상대 경로로 저장
                                input_path = os.path.relpath(audio_files["input"], self.recordings_dir) if audio_files["input"] else ""
                                output_path = os.path.relpath(audio_files["output"], self.recordings_dir) if audio_files["output"] else ""
                                
                                # STT 결과 데이터베이스에 저장
                                await self.save_stt_record(room_name, worker, data["result"], input_path, output_path, timestamp)
                                
                                # BSM에 STT 결과 보고 추가
                                try:
                                    device_id = bsm_agent.extract_device_id_from_room(room_name)
                                    await bsm_agent.report_stt_result(
                                        device_id=device_id,
                                        room_name=room_name,
                                        stt_text=data["result"],
                                        worker=worker,
                                        input_wav_path=input_path,
                                        output_wav_path=output_path
                                    )
                                except Exception as e:
                                    print(f"BSM STT 결과 보고 오류: {e}")
                            else:
                                print(f"음원 파일을 찾을 수 없어 STT 결과만 저장: {data['result']}")
                                # 음원 파일 없이 텍스트만 저장
                                await self.save_stt_record(room_name, worker, data["result"], "", "", timestamp)
                                
                                # 음원 파일 없이도 BSM에 STT 결과 보고
                                try:
                                    device_id = bsm_agent.extract_device_id_from_room(room_name)
                                    await bsm_agent.report_stt_result(
                                        device_id=device_id,
                                        room_name=room_name,
                                        stt_text=data["result"],
                                        worker=worker
                                    )
                                except Exception as e:
                                    print(f"BSM STT 결과 보고 오류: {e}")
                            
                        else:
                            self.text_remain_counter = self.text_remain_counter - 1
                            
                        if (self.text_remain_counter <= 0) or (data["result"] != ""):
                            with Session(engine) as session:
                                try:
                                    room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
                                    if room:
                                        room.message = data["result"]
                                        session.add(room)
                                        session.commit()
                                except Exception as e:
                                    print(f"STT commit error: {e}")
                    else:
                        raise Exception(f"STT audio failed: {res.status}")
        except Exception as e:
            print(f"STT audio error: {e}")

    def cleanup_client(self, client_id: int):
        """클라이언트 연결 종료 시 버퍼 정리"""
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.last_activity:
            del self.last_activity[client_id]

    def get_audio_files_for_period(self, room_name: str, start_time: datetime, end_time: datetime) -> list:
        """특정 기간의 모든 음원 파일 조회 (원본/처리 후 모두)"""
        try:
            audio_files = []
            current_time = start_time
            
            while current_time <= end_time:
                date_str = current_time.strftime('%Y-%m-%d')
                room_dir = os.path.join(self.recordings_dir, date_str, room_name)
                
                if os.path.exists(room_dir):
                    input_pattern = os.path.join(room_dir, "input_*.wav")
                    output_pattern = os.path.join(room_dir, "output_*.wav")
                    
                    input_files = glob.glob(input_pattern)
                    output_files = glob.glob(output_pattern)
                    
                    # 파일별로 매칭 정보 생성
                    file_mapping = {}
                    
                    # input 파일들 처리
                    for input_file in input_files:
                        filename = os.path.basename(input_file)
                        time_part = filename.split('_')[1]
                        worker = filename.split('_')[2].replace('.wav', '')
                        
                        try:
                            hour = int(time_part.split('h')[0])
                            minute = int(time_part.split('h')[1].split('m')[0])
                            second = int(time_part.split('m')[1].split('s')[0])
                            
                            file_time = current_time.replace(hour=hour, minute=minute, second=second, microsecond=0)
                            
                            if start_time <= file_time <= end_time:
                                key = f"{time_part}_{worker}"
                                file_mapping[key] = {
                                    'input_file': input_file,
                                    'output_file': "",
                                    'time': file_time,
                                    'worker': worker
                                }
                                
                        except (ValueError, IndexError):
                            continue
                    
                    # output 파일들 처리
                    for output_file in output_files:
                        filename = os.path.basename(output_file)
                        time_part = filename.split('_')[1]
                        worker = filename.split('_')[2].replace('.wav', '')
                        
                        try:
                            hour = int(time_part.split('h')[0])
                            minute = int(time_part.split('h')[1].split('m')[0])
                            second = int(time_part.split('m')[1].split('s')[0])
                            
                            file_time = current_time.replace(hour=hour, minute=minute, second=second, microsecond=0)
                            
                            if start_time <= file_time <= end_time:
                                key = f"{time_part}_{worker}"
                                if key in file_mapping:
                                    file_mapping[key]['output_file'] = output_file
                                else:
                                    file_mapping[key] = {
                                        'input_file': "",
                                        'output_file': output_file,
                                        'time': file_time,
                                        'worker': worker
                                    }
                                
                        except (ValueError, IndexError):
                            continue
                    
                    # 결과에 추가
                    for key, file_info in file_mapping.items():
                        audio_files.append(file_info)
                
                current_time += timedelta(days=1)
            
            # 시간순 정렬
            audio_files.sort(key=lambda x: x['time'])
            return audio_files
            
        except Exception as e:
            print(f"기간별 음원 파일 조회 오류: {e}")
            return []
        