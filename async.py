import pandas as pd 
import aiohttp
import asyncio
import time 
from tqdm import tqdm 
import os


async def async_process_excel(
    input_path: str,
    output_path: str,
    api_key: str,
    source_column: int ,  
    target_column: int ,  
    batch_size: int = 10,
    max_concurrent: int = 5,
) -> None:
    """
    async_process_excel:
    1. extract contents of source_column (index = 6)
    2. deal with deepseek
    3. save results to target_column (index = 12)
    
    parameters:
        input_path 
        output_path
        api_key
        batch_size: 每批处理的行数
        max_concurrent: 最大并发请求数
    """
    # Validate the input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")
    if not input_path.endswith('.xlsx'):
        raise ValueError("The input file must be an Excel (.xlsx) file.")
    
    # Read the Excel file
    df = pd.read_excel(input_path, "Sheet1", header=None)
    print(f"Excel file read successfully.")
    required_columns = 13  # M列是第13列（0-based）
    while len(df.columns) < required_columns:
        df[len(df.columns)] = ""  # Add empty columns if needed
    
    # Async api request
    async def process_text(session: aiohttp.ClientSession, text: str) -> str:
        """deal single data"""
        if not text or not str(text).strip():
            return text
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": f"""
                以下文本由AI生成，请降低其中AI的成分，保持英文，看起来更口语，更自然，更通俗易懂，
                如果产生了多个结果，只保留一个意思最为接近的即可，结果不包含你分析的过程，
                不需要你在结果后面使用括号做解释，也不需要“Here’s a more natural and conversational version:” 这种说法与方式，
                只需要你选择好生成的回答后，只告诉我回答，我将直接保存你的返回值作为结果，所以不需要那么多的修饰，只输出转换后的结果，
                其他多余的一句话都不需要：{text}
                """}
            ],
            "temperature": 0.5,
        }

        try:
            async with session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"Error: {response.status} - {await response.text()}")
                    return text
        except Exception as e:
            print(f"Request failed: {e}")
            return text
    # 批量处理函数
    async def process_batch(session: aiohttp.ClientSession, texts: list) -> list:
        """处理一批文本的异步函数"""
        tasks = [process_text(session, text) for text in texts]
        return await asyncio.gather(*tasks)
    
    # 主处理函数
    print("开始异步处理文本...")
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent) # 限制并发请求数

    async with aiohttp.ClientSession() as session:
        results = []
        total_rows = len(df)
        # 使用进度条（可选）
        with tqdm(total=total_rows, desc="处理进度") as pbar:
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size, source_column].tolist()  # 第G列数据
                
                async with semaphore:
                    batch_results = await process_batch(session, batch)
                    results.extend(batch_results)
                    pbar.update(len(batch))  # 更新进度条
                
                await asyncio.sleep(0.5)  # 限流防止速率限制
    
    # 将结果写入M列（第13列）
    for i, result in enumerate(results):
        df.iloc[i, target_column] = result # 第M列数据
    
    # 保存结果到Excel文件
    df.to_excel(output_path, index=False)

    total_time = time.time() - start_time
    print(f"处理完成，耗时 {total_time:.2f} 秒。")
    print(f"结果已保存到 {output_path}。")
    print(f"平均速度: {total_rows / total_time:.2f} 行/秒。")


if __name__ == "__main__":
    asyncio.run(async_process_excel(
        # Required parameters
        input_path="input.xlsx",
        output_path="output.xlsx",
        api_key="", # TODO Add API Key here
        source_column=6,
        target_column=12,

        # Async parameters
        batch_size=10,
        max_concurrent=5
    ))