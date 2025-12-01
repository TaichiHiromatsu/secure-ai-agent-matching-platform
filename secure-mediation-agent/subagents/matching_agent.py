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
import os
from typing import Any

import httpx
from google.adk import Agent
from google.genai import types
from ..config.safety import SAFETY_SETTINGS_RELAXED

# Agent Store API configuration
# Default to 127.0.0.1:8001 for container deployment (trusted_agent_hub internal port)
# In Cloud Run container, nginx is on 8080, but trusted_agent_hub listens directly on 8001
# Override via environment variable for different configurations
AGENT_STORE_API_URL = os.getenv("AGENT_STORE_API_URL", "http://127.0.0.1:8001/api/agents")


async def fetch_agent_card(agent_url: str) -> dict[str, Any]:
    """Fetch agent card from the agent's well-known endpoint.

    Args:
        agent_url: Base URL of the agent.

    Returns:
        Agent card dictionary.
    """
    try:
        # A2A spec: agent cards are available at /.well-known/agent-card.json
        card_url = f"{agent_url.rstrip('/')}/.well-known/agent-card.json"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(card_url)
            response.raise_for_status()
            return response.json()

    except Exception as e:
        return {
            "error": f"Failed to fetch agent card: {str(e)}",
            "url": agent_url,
        }


def _convert_agent_entry_to_matcher_format(agent: dict[str, Any]) -> dict[str, Any]:
    """Convert Agent Store API response to matcher format.

    Args:
        agent: Agent entry from trusted_agent_hub API.

    Returns:
        Agent dictionary in matcher format.
    """
    # Extract skills from tags and use_cases
    skills = []
    tags = agent.get("tags", []) or []
    use_cases = agent.get("use_cases", []) or []

    # Create skills from use_cases with tags
    for use_case in use_cases:
        skills.append({
            "id": use_case.replace(" ", "_").lower(),
            "tags": tags + [use_case.lower()],
        })

    # If no use_cases, create a generic skill from tags
    if not skills and tags:
        skills.append({
            "id": "general",
            "tags": tags,
        })

    # Convert trust_score from 0-100 integer to 0.0-1.0 float
    trust_score_raw = agent.get("trust_score")
    if trust_score_raw is not None:
        trust_score = float(trust_score_raw) / 100.0
    else:
        trust_score = 0.5  # Default trust score

    return {
        "name": agent.get("name", "unknown"),
        "url": agent.get("endpoint_url") or agent.get("agent_card_url", ""),
        "description": ", ".join(use_cases) if use_cases else f"Agent: {agent.get('name', 'unknown')}",
        "skills": skills,
        "capabilities": {
            "booking": any("booking" in t.lower() for t in tags),
            "search": any("search" in t.lower() for t in tags),
        },
        "trust_score": trust_score,
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "provider": agent.get("provider", ""),
        "status": agent.get("status", "unknown"),
        "agent_id": agent.get("id", ""),
    }


async def search_agent_store(
    query: str,
) -> str:
    """Search the agent store for matching agents.

    Fetches agents from the trusted_agent_hub API and filters them
    based on the search query.

    Args:
        query: Search query describing needed capabilities.

    Returns:
        JSON string with search results.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Fetch all active agents from the Agent Store API
            response = await client.get(
                AGENT_STORE_API_URL,
                params={"status": "active", "limit": 100},
            )
            response.raise_for_status()
            data = response.json()

        # Convert API response to matcher format
        all_agents = [
            _convert_agent_entry_to_matcher_format(agent)
            for agent in data.get("items", [])
        ]

        # Filter agents based on query keywords
        query_lower = query.lower()
        query_keywords = query_lower.split()
        filtered_agents = []

        for agent in all_agents:
            matches = False

            # Check agent name
            if any(keyword in agent["name"].lower() for keyword in query_keywords):
                matches = True

            # Check description
            if any(keyword in agent["description"].lower() for keyword in query_keywords):
                matches = True

            # Check skill tags
            for skill in agent.get("skills", []):
                skill_tags = skill.get("tags", [])
                if any(keyword in tag.lower() for tag in skill_tags for keyword in query_keywords):
                    matches = True
                    break

            # Check provider
            if any(keyword in agent.get("provider", "").lower() for keyword in query_keywords):
                matches = True

            if matches:
                filtered_agents.append(agent)

        # If no specific matches, return all agents (broad search)
        if not filtered_agents:
            filtered_agents = all_agents

        return json.dumps(
            {"agents": filtered_agents, "count": len(filtered_agents)},
            ensure_ascii=False,
            indent=2,
        )

    except httpx.HTTPError as e:
        # Return error information if API call fails
        return json.dumps(
            {
                "agents": [],
                "count": 0,
                "error": f"Failed to fetch agents from store: {str(e)}",
                "store_url": AGENT_STORE_API_URL,
            },
            ensure_ascii=False,
            indent=2,
        )


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

    # Handle both dict and list formats for capabilities
    if required_caps:
        max_score += 0.3
        if isinstance(required_caps, dict):
            cap_matches = sum(
                1 for cap_key, cap_value in required_caps.items()
                if agent_caps.get(cap_key) == cap_value
            )
            score += (cap_matches / len(required_caps)) * 0.3
        elif isinstance(required_caps, list):
            # If capabilities is a list, just check if agent has any capabilities
            cap_matches = len(agent_caps) if agent_caps else 0
            score += (min(cap_matches, len(required_caps)) / len(required_caps)) * 0.3

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
    model='gemini-2.5-pro',
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
- Fetch agent cards from /.well-known/agent-card.json endpoints
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
        temperature=0.4,  # Moderate temperature for consistent matching
        safety_settings=SAFETY_SETTINGS_RELAXED,
    ),
)
