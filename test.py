import httpx
import asyncio

async def get_user_image_path(user_id: str) -> str | None:
    url = f"http://localhost:8010/users/{user_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        profile = response.json()
        return profile.get("image_path")

if __name__ == "__main__":
    path = asyncio.run(get_user_image_path("user456"))
    print(path)
