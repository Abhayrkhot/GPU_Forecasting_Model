import asyncio
print("Script started")

async def test():
    print("Async function started")
    await asyncio.sleep(1)
    print("Async function completed")

print("About to run async")
asyncio.run(test())
print("Script completed")

