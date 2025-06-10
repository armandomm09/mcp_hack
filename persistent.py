from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
import json
import os
import copy
from typing import Any, Optional, List, Dict

load_dotenv()

server_params = StdioServerParameters(
    command="uv",
    args=["run", "./server.py"],
)


CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

messages = {
    "messages": [
        {"role": "user", "content": "Can you show me points of interest near 48.8575° N, 2.3514° E?"},
    ]
}

class PersistentSegmentTreeNode:
    def __init__(self, left: int, right: int, value: Any = None, left_child=None, right_child=None):
        self.left = left
        self.right = right
        self.value = value
        self.left_child = left_child
        self.right_child = right_child

class PersistentSegmentTree:
    def __init__(self, size: int):
        self.size = size
        self.versions = []  # List of root nodes for each version
        self.versions.append(self._build(0, size - 1))

    def _build(self, l: int, r: int) -> PersistentSegmentTreeNode:
        if l == r:
            return PersistentSegmentTreeNode(l, r, None)
        m = (l + r) // 2
        left_child = self._build(l, m)
        right_child = self._build(m + 1, r)
        return PersistentSegmentTreeNode(l, r, None, left_child, right_child)

    def update(self, version: int, idx: int, value: Any) -> int:
        new_root = self._update(self.versions[version], idx, value)
        self.versions.append(new_root)
        return len(self.versions) - 1

    def _update(self, node: PersistentSegmentTreeNode, idx: int, value: Any) -> PersistentSegmentTreeNode:
        if node.left == node.right:
            return PersistentSegmentTreeNode(node.left, node.right, value)
        m = (node.left + node.right) // 2
        if idx <= m:
            left_child = self._update(node.left_child, idx, value)
            return PersistentSegmentTreeNode(node.left, node.right, node.value, left_child, node.right_child)
        else:
            right_child = self._update(node.right_child, idx, value)
            return PersistentSegmentTreeNode(node.left, node.right, node.value, node.left_child, right_child)

    def query(self, version: int, l: int, r: int) -> List[Any]:
        return self._query(self.versions[version], l, r)

    def _query(self, node: PersistentSegmentTreeNode, l: int, r: int) -> List[Any]:
        if node.right < l or node.left > r:
            return []
        if l <= node.left and node.right <= r:
            if node.left == node.right:
                return [node.value]
            else:
                return self._query(node.left_child, l, r) + self._query(node.right_child, l, r)
        return self._query(node.left_child, l, r) + self._query(node.right_child, l, r)

# search "branching" implementation
class FlightSearchBranchManager:
    def __init__(self, max_branches=100):
        self.tree = PersistentSegmentTree(max_branches)
        self.branch_params: List[Optional[Dict[str, Any]]] = [None] * max_branches
        self.current_version = 0
        self.next_idx = 0

    def add_branch(self, params: Dict[str, Any], result: Any, from_version: Optional[int] = None) -> int:
        idx = self.next_idx
        self.branch_params[idx] = {'params': params, 'result': result}
        if from_version is None:
            from_version = self.current_version
        new_version = self.tree.update(from_version, idx, result)
        self.next_idx += 1
        self.current_version = new_version
        return new_version

    def get_branch_result(self, version: int, idx: int) -> Any:
        return self.tree.query(version, idx, idx)[0]

    def get_all_results(self, version: int) -> List[Any]:
        return self.tree.query(version, 0, self.next_idx - 1)

flight_branch_manager = FlightSearchBranchManager()

async def main():

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await load_mcp_tools(session)

            # wrap flight tool so that calls are tracked in segtree
            def wrap_flight_tool(tool):
                async def wrapped(*args, **kwargs):
                    """Persistent segment tree tracking for branching search states."""
                    params = {}
                    if args:
                        param_names = ["origin", "destination", "departure_date", "return_date", "adults"]
                        for i, name in enumerate(param_names):
                            if i < len(args):
                                params[name] = args[i]
                    params.update(kwargs)
                    # original tool call
                    result = await tool(*args, **kwargs)
                    # save in segtree persistent
                    flight_branch_manager.add_branch(params, result)
                    return result
                return wrapped

            for i, tool in enumerate(tools):
                tool_name = getattr(tool, "__name__", None) or getattr(tool, "name", None) or ""
                if "flight" in tool_name or "tool_flight_offer" in tool_name:
                    tools[i] = wrap_flight_tool(tool)

            llm = ChatCerebras(
                model="llama-3.3-70b",
                temperature=0,
                max_tokens=10000,
                timeout=None,
                max_retries=1,
                api_key=CEREBRAS_API_KEY
            )
            agent = create_react_agent(
                llm,
                tools=tools,
                debug=True,
                prompt="""You are a helpful travel planner. When the user asks for travel options,
                            call exactly one tool (and no more). After calling that tool and getting its output,
                            stop reasoning and deliver the final answer. Do NOT think about calling any other tool. """,
            )
            agent_response = await agent.ainvoke(messages, stream_mode="debug")
            print(str(agent_response))


asyncio.run(main())
