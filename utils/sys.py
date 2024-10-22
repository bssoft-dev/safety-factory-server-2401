import asyncio

async def log(message):
    with open("async.log", mode="a") as log:
        log.write(str(message))
        log.write('\n')

async def aprint(message):
    asyncio.create_task(log(message))