import asyncio

async def log(message, file_path="async.log"):
    with open(file_path, mode="a") as log:
        log.write(str(message))
        log.write('\n')

def aprint(message, file_path="async.log"):
    asyncio.create_task(log(message, file_path))