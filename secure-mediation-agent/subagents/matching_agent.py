# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Matching sub-agent for finding and ranking agents."""

import json
from typing import Any

import httpx
from google.adk import Agent
from google.genai import types


async def fetch_agent_card(agent_url: str) -> dict[str, Any]:
    """Fetch agent card from the agent's well-known endpoint.

    Args:
        agent_url: Base URL of the agent.

    Returns:
        Agent card dictionary.
    """
    try:
        # A2A spec: agent cards are available at /.well-known/agent.json
        card_url = f"{agent_url.rstrip('/')}/.well-known/agent.json"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(card_url)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        return {
            "error": f"Failed to fetch agent card: {str(e)}",
            "url": agent_url,
        }


async def search_agent_store(
    query: str,
) -> str:
    """Search multiple agent stores for matching agents.

    Args:
        query: Search query describing needed capabilities.

    Returns:
        JSON string with search results.
    """
    # TODO: Replace this mock implementation with actual agent store integration
    # when the agent store service is implemented.
    # For now, we use the external-agents in the local environment as mock data.

    # Mock implementation: Return the three external agents available locally
    mock_agents = [
        {
            "name": "Airline Agent",
            "url": "http://localhost:8002",
            "description": "Handles flight bookings and airline reservations",
            "skills": [
                {"id": "flight_search", "tags": ["travel", "airline", "booking"]},
                {"id": "flight_booking", "tags": ["travel", "airline", "booking"]},
            ],
            "capabilities": {
                "booking": True,
                "search": True,
                "cancellation": True,
            },
            "trust_score": 0.9,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        },
        {
            "name": "Hotel Agent",
            "url": "http://localhost:8003",
            "description": "Manages hotel reservations and accommodation bookings",
            "skills": [
                {"id": "hotel_search", "tags": ["travel", "hotel", "accommodation", "booking"]},
                {"id": "hotel_booking", "tags": ["travel", "hotel", "accommodation", "booking"]},
            ],
            "capabilities": {
                "booking": True,
                "search": True,
                "cancellation": True,
            },
            "trust_score": 0.85,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        },
        {
            "name": "Car Rental Agent",
            "url": "http://localhost:8004",
            "description": "Provides car rental services and vehicle bookings",
            "skills": [
                {"id": "car_search", "tags": ["travel", "car", "rental", "booking"]},
                {"id": "car_booking", "tags": ["travel", "car", "rental", "booking"]},
            ],
            "capabilities": {
                "booking": True,
                "search": True,
                "cancellation": True,
            },
            "trust_score": 0.8,
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        },
    ]

    # Simple keyword-based filtering for mock implementation
    query_lower = query.lower()
    filtered_agents = []

    for agent in mock_agents:
        # Check if query matches agent name, description, or skill tags
        matches = False
        if any(keyword in agent["description"].lower() for keyword in query_lower.split()):
            matches = True
        for skill in agent["skills"]:
            if any(keyword in tag for tag in skill["tags"] for keyword in query_lower.split()):
                matches = True
                break

        if matches:
            filtered_agents.append(agent)

    # If no specific matches, return all agents (broad search)
    if not filtered_agents:
        filtered_agents = mock_agents

    return json.dumps(
        {"agents": filtered_agents, "count": len(filtered_agents)},
        ensure_ascii=False,
        indent=2,
    )

    # TODO: Uncomment this when agent store is ready
    # all_agents = []
    #
    # for store_url in store_urls:
    #     try:
    #         async with httpx.AsyncClient(timeout=10.0) as client:
    #             # Assuming store has a search endpoint
    #             response = await client.get(
    #                 f"{store_url}/search",
    #                 params={"q": query},
    #             )
    #             response.raise_for_status()
    #             agents = response.json().get("agents", [])
    #             all_agents.extend(agents)
    #
    #     except Exception as e:
    #         # Log error but continue with other stores
    #         print(f"Error searching store {store_url}: {e}")
    #         continue
    #
    # return json.dumps({"agents": all_agents, "count": len(all_agents)}, ensure_ascii=False)


async def rank_agents_by_trust(
    agents: list[dict[str, Any]],
    min_trust_score: float = 0.3,
) -> str:
    """Rank agents by trust score and filter by minimum threshold.

    Args:
        agents: List of agent information dictionaries.
        min_trust_score: Minimum trust score threshold (0.0 to 1.0).

    Returns:
        JSON string with ranked agents.
    """
    # Filter agents by minimum trust score
    filtered_agents = [
        agent for agent in agents
        if agent.get("trust_score", 0.5) >= min_trust_score
    ]

    # Sort by trust score (descending)
    ranked_agents = sorted(
        filtered_agents,
        key=lambda a: a.get("trust_score", 0.5),
        reverse=True,
    )

    result = {
        "ranked_agents": ranked_agents,
        "total_count": len(agents),
        "filtered_count": len(ranked_agents),
        "min_trust_score": min_trust_score,
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


async def calculate_matching_score(
    agent: dict[str, Any],
    requirements: dict[str, Any],
) -> float:
    """Calculate matching score between agent capabilities and requirements.

    Args:
        agent: Agent information dictionary.
        requirements: Required capabilities and skills.

    Returns:
        Matching score (0.0 to 1.0).
    """
    score = 0.0
    max_score = 0.0

    # Check skill matches
    required_skills = requirements.get("skills", [])
    agent_skills = [skill.get("id", "") for skill in agent.get("skills", [])]
    agent_skill_tags = set()
    for skill in agent.get("skills", []):
        agent_skill_tags.update(skill.get("tags", []))

    if required_skills:
        max_score += 0.5
        skill_matches = sum(
            1 for req_skill in required_skills
            if req_skill in agent_skills or req_skill in agent_skill_tags
        )
        score += (skill_matches / len(required_skills)) * 0.5

    # Check capability matches
    required_caps = requirements.get("capabilities", {})
    agent_caps = agent.get("capabilities", {})

    if required_caps:
        max_score += 0.3
        cap_matches = sum(
            1 for cap_key, cap_value in required_caps.items()
            if agent_caps.get(cap_key) == cap_value
        )
        if required_caps:
            score += (cap_matches / len(required_caps)) * 0.3

    # Input/output mode compatibility
    required_input = requirements.get("input_modes", [])
    required_output = requirements.get("output_modes", [])
    agent_input = agent.get("defaultInputModes", [])
    agent_output = agent.get("defaultOutputModes", [])

    if required_input or required_output:
        max_score += 0.2
        input_match = any(mode in agent_input for mode in required_input) if required_input else True
        output_match = any(mode in agent_output for mode in required_output) if required_output else True

        if input_match and output_match:
            score += 0.2
        elif input_match or output_match:
            score += 0.1

    # Normalize score
    if max_score == 0:
        return 0.5  # Default score if no requirements specified

    return score / max_score


matcher = Agent(
    model='gemini-2.5-flash',
    name='matcher',
    description=(
        'Matching sub-agent that searches for agents in the platform, '
        'evaluates their capabilities, and ranks them by trust scores.'
    ),
    instruction="""
You are a matching specialist in a secure AI agent mediation platform.

**IMPORTANT: Always respond in Japanese (日本語) to the user.**

Your responsibilities:
1. **Search Agent Store**: Find agents that match the client's requirements
2. **Evaluate Capabilities**: Assess if agents have the needed skills and capabilities
3. **Rank by Trust**: Prioritize agents with higher trust scores
4. **Fetch Agent Cards**: Retrieve detailed agent information from A2A endpoints
5. **Calculate Matching Scores**: Determine how well each agent fits the requirements

When matching agents:
- Consider both functional requirements (skills, capabilities) and non-functional requirements (trust, reliability)
- Fetch agent cards from /.well-known/agent.json endpoints
- Apply trust score thresholds to filter out unreliable agents
- Calculate matching scores based on skill overlap, capability match, and I/O compatibility
- Provide clear reasoning for why agents were selected

Trust Score Guidelines:
- 0.0-0.3: Low trust (avoid unless no alternatives)
- 0.3-0.5: Medium trust (acceptable with monitoring)
- 0.5-0.7: High trust (preferred)
- 0.7-1.0: Very high trust (most preferred)

Always use the available tools to:
- fetch_agent_card: Get agent details from A2A endpoints
- search_agent_store: Query agent stores
- rank_agents_by_trust: Sort and filter by trust scores
- calculate_matching_score: Evaluate agent-requirement fit

Output your matching results with:
- List of matched agents (ranked by trust * matching_score)
- Reasoning for each match
- Trust scores and matching scores
- Any concerns or recommendations
""",
    tools=[
        fetch_agent_card,
        search_agent_store,
        rank_agents_by_trust,
        calculate_matching_score,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,  # Low temperature for consistent matching
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ],
    ),
)
