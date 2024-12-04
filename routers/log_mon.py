import os
import asyncio
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocket

log_mon = APIRouter(tags=["Logging"])

@log_mon.websocket("/logs/{filename}")
async def websocket_logs(filename: str, websocket: WebSocket):
    await websocket.accept()
    filepath = os.path.join("logs", filename)
    try:
        # 기존 로그 파일 내용 먼저 전송
        if os.path.exists(filepath):
            with open(filepath, 'r') as log_file:
                # 마지막 20줄만 읽어서 보냄 (메모리 고려)
                log_lines = log_file.readlines()[-20:]
                for line in log_lines:
                    await websocket.send_text(line.strip())

        # 실시간 로그 스트리밍
        with open(filepath, 'r') as log_file:
            # 파일의 끝으로 이동
            log_file.seek(0, os.SEEK_END)
            
            while True:
                line = log_file.readline()
                if not line:
                    # 새로운 내용이 없으면 잠시 대기
                    await asyncio.sleep(0.5)
                    continue
                await websocket.send_text(line.strip())
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
    
    
@log_mon.get("/log-viewer/{filename}")
async def get_log_viewer(filename: str):
    '''
    Filename: uvicorn.log, warning.log
    '''
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Uvicorn Log Viewer</title>
        <style>
            body {{ font-family: monospace; background-color: #f4f4f4; }}
            #logs {{ 
                height: 500px; 
                overflow-y: scroll; 
                background-color: white; 
                padding: 10px; 
                border: 1px solid #ddd; 
            }}
        </style>
    </head>
    <body>
        <h1>Real-time Logs</h1>
        <div id="logs"></div>
        <script>
            const logsDiv = document.getElementById('logs');
            const ws = new WebSocket('wss://' + window.location.host + '/logs/{filename}');
            
            ws.onmessage = function(event) {{
                const logEntry = document.createElement('div');
                logEntry.textContent = event.data;
                logsDiv.appendChild(logEntry);
                logsDiv.scrollTop = logsDiv.scrollHeight;
            }};
        </script>
    </body>
    </html>
    """)