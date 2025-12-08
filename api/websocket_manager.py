import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        # Store active connections
        self.active_connections: List[WebSocket] = []

        # Store connections by group (e.g., analysis_id, user_id)
        self.connection_groups: Dict[str, Set[WebSocket]] = {}

        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

        # Track connection stats
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0
        }

    async def connect(self, websocket: WebSocket, group: Optional[str] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

        # Add to group if specified
        if group:
            if group not in self.connection_groups:
                self.connection_groups[group] = set()
            self.connection_groups[group].add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "connected_at": datetime.now(),
            "group": group,
            "messages_sent": 0,
            "messages_received": 0
        }

        # Update stats
        self.connection_stats["total_connections"] += 1
        self.connection_stats["active_connections"] = len(self.active_connections)

        logger.info(f"WebSocket connected. Group: {group}. Active connections: {len(self.active_connections)}")

        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Connected to TradeGraph Financial Advisor",
            "timestamp": datetime.now().isoformat(),
            "group": group
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Remove from groups
        metadata = self.connection_metadata.get(websocket, {})
        group = metadata.get("group")
        if group and group in self.connection_groups:
            self.connection_groups[group].discard(websocket)
            if not self.connection_groups[group]:  # Remove empty group
                del self.connection_groups[group]

        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        # Update stats
        self.connection_stats["active_connections"] = len(self.active_connections)

        logger.info(f"WebSocket disconnected. Group: {group}. Active connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Any, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            if isinstance(message, dict):
                message_str = json.dumps(message, default=str)
            else:
                message_str = str(message)

            await websocket.send_text(message_str)

            # Update stats
            self.connection_stats["messages_sent"] += 1
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_sent"] += 1

        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {str(e)}")
            self.disconnect(websocket)

    async def broadcast(self, message: Any):
        """Broadcast a message to all connected WebSocket clients."""
        if not self.active_connections:
            return

        logger.info(f"Broadcasting message to {len(self.active_connections)} connections")

        # Create list of tasks for concurrent sending
        tasks = []
        for connection in self.active_connections.copy():  # Copy to avoid modification during iteration
            tasks.append(self.send_personal_message(message, connection))

        # Send to all connections concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to_group(self, group: str, message: Any):
        """Send a message to all connections in a specific group."""
        if group not in self.connection_groups:
            logger.warning(f"Group {group} not found")
            return

        connections = list(self.connection_groups[group])  # Convert to list to avoid modification during iteration
        if not connections:
            return

        logger.info(f"Sending message to group {group} with {len(connections)} connections")

        # Create tasks for concurrent sending
        tasks = []
        for connection in connections:
            tasks.append(self.send_personal_message(message, connection))

        # Send to all group connections concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_analysis_update(self, analysis_id: str, update_data: Dict[str, Any]):
        """Send analysis update to relevant connections."""
        message = {
            "type": "analysis_update",
            "analysis_id": analysis_id,
            "data": update_data,
            "timestamp": datetime.now().isoformat()
        }

        # Send to analysis-specific group
        analysis_group = f"analysis_{analysis_id}"
        await self.send_to_group(analysis_group, message)

        # Also broadcast to general connections if it's a significant update
        if update_data.get("status") in ["completed", "failed"]:
            await self.broadcast({
                "type": "analysis_completed",
                "analysis_id": analysis_id,
                "status": update_data.get("status"),
                "timestamp": datetime.now().isoformat()
            })

    async def send_market_update(self, market_data: Dict[str, Any]):
        """Send market data update to all connections."""
        message = {
            "type": "market_update",
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)

    async def send_alert(self, alert_data: Dict[str, Any], target_groups: Optional[List[str]] = None):
        """Send alert to specific groups or broadcast to all."""
        message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": datetime.now().isoformat(),
            "urgency": alert_data.get("urgency", "medium")
        }

        if target_groups:
            # Send to specific groups
            for group in target_groups:
                await self.send_to_group(group, message)
        else:
            # Broadcast to all connections
            await self.broadcast(message)

    async def send_recommendation_update(self, recommendation_data: Dict[str, Any]):
        """Send trading recommendation update."""
        message = {
            "type": "recommendation_update",
            "data": recommendation_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)

    async def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        logger.info(f"Disconnecting all {len(self.active_connections)} WebSocket connections")

        # Send goodbye message to all connections
        goodbye_message = {
            "type": "server_shutdown",
            "message": "Server is shutting down",
            "timestamp": datetime.now().isoformat()
        }

        # Send goodbye messages concurrently
        tasks = []
        for connection in self.active_connections.copy():
            tasks.append(self.send_personal_message(goodbye_message, connection))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Close all connections
        for connection in self.active_connections.copy():
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {str(e)}")

        # Clear all data structures
        self.active_connections.clear()
        self.connection_groups.clear()
        self.connection_metadata.clear()

        logger.info("All WebSocket connections disconnected")

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            **self.connection_stats,
            "active_groups": len(self.connection_groups),
            "group_details": {
                group: len(connections)
                for group, connections in self.connection_groups.items()
            }
        }

    def get_connection_info(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection."""
        return self.connection_metadata.get(websocket)

    async def ping_all_connections(self):
        """Send ping to all connections to check if they're alive."""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.now().isoformat()
        }

        # Send ping to all connections
        dead_connections = []
        for connection in self.active_connections.copy():
            try:
                await self.send_personal_message(ping_message, connection)
            except Exception:
                dead_connections.append(connection)

        # Remove dead connections
        for connection in dead_connections:
            self.disconnect(connection)

        if dead_connections:
            logger.info(f"Removed {len(dead_connections)} dead connections")

    async def handle_client_message(self, websocket: WebSocket, message: str):
        """Handle incoming message from client."""
        try:
            # Update stats
            self.connection_stats["messages_received"] += 1
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_received"] += 1

            # Parse message
            try:
                data = json.loads(message)
                message_type = data.get("type", "unknown")
            except json.JSONDecodeError:
                # Treat as plain text message
                data = {"type": "text", "content": message}
                message_type = "text"

            logger.info(f"Received WebSocket message: {message_type}")

            # Handle different message types
            if message_type == "ping":
                await self.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, websocket)

            elif message_type == "subscribe":
                # Subscribe to a group
                group = data.get("group")
                if group:
                    if group not in self.connection_groups:
                        self.connection_groups[group] = set()
                    self.connection_groups[group].add(websocket)

                    # Update metadata
                    if websocket in self.connection_metadata:
                        self.connection_metadata[websocket]["group"] = group

                    await self.send_personal_message({
                        "type": "subscribed",
                        "group": group,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)

            elif message_type == "unsubscribe":
                # Unsubscribe from a group
                group = data.get("group")
                if group and group in self.connection_groups:
                    self.connection_groups[group].discard(websocket)
                    if not self.connection_groups[group]:
                        del self.connection_groups[group]

                    await self.send_personal_message({
                        "type": "unsubscribed",
                        "group": group,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)

            else:
                # Echo unknown messages back to client for debugging
                await self.send_personal_message({
                    "type": "echo",
                    "original_message": data,
                    "timestamp": datetime.now().isoformat()
                }, websocket)

        except Exception as e:
            logger.error(f"Error handling client message: {str(e)}")
            await self.send_personal_message({
                "type": "error",
                "message": "Error processing message",
                "timestamp": datetime.now().isoformat()
            }, websocket)
