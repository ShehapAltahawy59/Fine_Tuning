
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
import json

async def generate_qa_from_chunk(chunk, agent):
    text_message = TextMessage(content=chunk, source="User")
    result = await agent.on_messages(
            [text_message], 
            cancellation_token=CancellationToken()
        )
    response = result.chat_message.content
    response = response.model_dump()
    return response

import asyncio


async def run(chunks, client):
    dataset = []
    for chunk in chunks:
        #try:
            qas = await generate_qa_from_chunk(chunk, client)
            qas = qas["output"]
            
            
            for qa in qas:
                if isinstance(qa, str):
                    qas_parsed = json.loads(qa)
                else:
                    qas_parsed = qa
                dataset.append({
                    "context": chunk,
                    "question": qas_parsed["question"],
                    "answer": qas_parsed["answer"]
                })
                await asyncio.sleep(10)
        # except Exception as e:
        #     print("Error processing chunk:", e)
    return dataset
