import asyncio
import logging
import traceback
import threading
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.patent_fetcher import patent_fetcher_node
from app.agents.innovation_extractor import innovation_extractor_node
from app.agents.synthesis import synthesis_node
from app.agents.ideation import ideation_node
from app.database import SessionLocal
from app.tools.db_tools import get_analysis
from app.tools.llm_provider import resolve_llm_config

logger = logging.getLogger(__name__)
ws_router = APIRouter()


@ws_router.websocket("/ws/analysis/{session_id}")
async def websocket_analysis(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Read API keys from first message (sent by frontend after connect)
    try:
        init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        openai_key = init_msg.get("openai_key")
        groq_key = init_msg.get("groq_key")
    except (asyncio.TimeoutError, Exception):
        openai_key = None
        groq_key = None

    # Resolve which LLM provider to use
    llm_config = resolve_llm_config(openai_key=openai_key, groq_key=groq_key)
    logger.info(f"Using LLM provider: {llm_config.provider} for session {session_id}")

    db = SessionLocal()
    try:
        analysis = get_analysis(db, UUID(session_id))
        if not analysis:
            await websocket.send_json({"type": "error", "message": "Analysis not found"})
            await websocket.close()
            return

        # Load previous analysis if linked
        previous_analysis = None
        if analysis.previous_session_id:
            prev = get_analysis(db, analysis.previous_session_id)
            if prev and prev.status == "completed":
                previous_analysis = {
                    "search_query": prev.search_query,
                    "synthesis": prev.synthesis,
                    "generated_ideas": prev.generated_ideas,
                }
        db.close()
        db = None

        # Queue for real-time status updates from pipeline thread
        update_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def on_status(update: dict):
            loop.call_soon_threadsafe(update_queue.put_nowait, update)

        pipeline_done = asyncio.Event()
        pipeline_error = None

        def run_in_thread():
            nonlocal pipeline_error
            try:
                state = {
                    "session_id": session_id,
                    "search_query": analysis.search_query,
                    "patents": [],
                    "innovations": [],
                    "synthesis": {},
                    "generated_ideas": [],
                    "status_updates": [],
                    "previous_analysis": previous_analysis,
                    "llm_provider": llm_config.provider,
                    "llm_api_key": llm_config.api_key,
                    "llm_models": llm_config.models,
                    "embedding_api_key": llm_config.embedding_api_key,
                }

                nodes = [
                    ("patent_fetcher", patent_fetcher_node),
                    ("innovation_extractor", innovation_extractor_node),
                    ("synthesis", synthesis_node),
                    ("ideation", ideation_node),
                ]

                last_sent = 0
                for name, node_fn in nodes:
                    result = node_fn(state)
                    state.update(result)
                    updates = result.get("status_updates", [])
                    for update in updates[last_sent:]:
                        on_status(update)
                    last_sent = len(updates)

            except Exception as e:
                pipeline_error = str(e)
                traceback.print_exc()
            finally:
                loop.call_soon_threadsafe(pipeline_done.set)

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        # Send updates as they arrive
        while not pipeline_done.is_set():
            try:
                update = await asyncio.wait_for(update_queue.get(), timeout=0.3)
                await websocket.send_json(update)
            except asyncio.TimeoutError:
                continue

        # Drain remaining
        while not update_queue.empty():
            update = update_queue.get_nowait()
            await websocket.send_json(update)

        if pipeline_error:
            await websocket.send_json({"type": "error", "message": pipeline_error})
        else:
            await websocket.send_json({"type": "complete", "session_id": session_id})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Pipeline error for session {session_id}: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if db:
            db.close()
        try:
            await websocket.close()
        except Exception:
            pass
