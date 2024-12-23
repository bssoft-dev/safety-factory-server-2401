import pydub
import os

def extract_segment_from_file(src_file, dst_dir, duration=2):
    sound = pydub.AudioSegment.from_wav(src_file)
    for i in range(0, len(sound), duration * 1000):
        segment = sound[i:i+duration*1000]
        segment.export(f"{dst_dir}/{src_file.split('/')[-1].split('.')[0]}_seg_{i//(duration*1000)}.wav", format="wav")


def work1():
    SRC_DIR = "/home/bssoft/nfs/BS_DATA_WARM/위험상황_광주대_2418/소리 원본/screaming"
    DST_DIR = "/home/bssoft/nfs/BS_DATA_WARM/위험상황_광주대_2418/소리 선택본/일반"

    numbers = [1, 6, 12, 18, 20, 31, 35, 40, 41, 43, 47, 50, 58, 59, 61, 64, 67]

    for n in numbers:
        src_file = f"{SRC_DIR}/scream{n}.wav"
        extract_segment_from_file(src_file, DST_DIR)

def work2():
    SRC_DIR = "/home/bssoft/bssoft_12tb_mount/PROJECT 2024/_01. 지역SW서비스사업화사업/개발관련/작업현장 동영상/소리변환"
    DST_DIR = "/home/bssoft/nfs/BS_DATA_WARM/위험상황_광주대_2418/소리 선택본/일반"
    
    for file in os.listdir(SRC_DIR):
        if file.endswith(".wav"):
            extract_segment_from_file(f"{SRC_DIR}/{file}", DST_DIR)


work2()