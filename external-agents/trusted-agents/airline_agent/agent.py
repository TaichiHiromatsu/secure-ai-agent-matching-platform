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

"""Mock Airline Agent for demo purposes."""

import random
from datetime import datetime, timedelta

from google.adk import Agent
from google.genai import types


async def search_flights(
    departure: str,
    destination: str,
    departure_date: str,
    return_date: str = "",
    passengers: int = 1,
) -> str:
    """Search for available flights.

    Args:
        departure: Departure airport/city.
        destination: Destination airport/city.
        departure_date: Departure date (YYYY-MM-DD).
        return_date: Return date (YYYY-MM-DD), optional.
        passengers: Number of passengers.

    Returns:
        JSON string with available flights.
    """
    import json

    # Mock flight options
    flights = [
        {
            "flight_id": f"FL{random.randint(1000, 9999)}",
            "airline": "Sky Airlines",
            "departure": departure,
            "destination": destination,
            "departure_time": "08:30",
            "arrival_time": "11:45",
            "price_per_person": random.randint(15000, 35000),
            "currency": "JPY",
            "seats_available": random.randint(5, 50),
        },
        {
            "flight_id": f"FL{random.randint(1000, 9999)}",
            "airline": "Ocean Air",
            "departure": departure,
            "destination": destination,
            "departure_time": "14:20",
            "arrival_time": "17:35",
            "price_per_person": random.randint(15000, 35000),
            "currency": "JPY",
            "seats_available": random.randint(5, 50),
        },
    ]

    result = {
        "search_criteria": {
            "departure": departure,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date if return_date else "one-way",
            "passengers": passengers,
        },
        "flights": flights,
        "total_results": len(flights),
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


async def book_flight(
    flight_id: str,
    passenger_name: str,
    passenger_email: str,
    passengers: int = 1,
) -> str:
    """Book a flight.

    Args:
        flight_id: The flight ID to book.
        passenger_name: Name of the passenger.
        passenger_email: Email of the passenger.
        passengers: Number of passengers.

    Returns:
        JSON string with booking confirmation.
    """
    import json

    # Mock booking confirmation
    booking_id = f"BK{random.randint(100000, 999999)}"
    confirmation_code = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))}"

    booking = {
        "status": "confirmed",
        "booking_id": booking_id,
        "confirmation_code": confirmation_code,
        "flight_id": flight_id,
        "passenger_name": passenger_name,
        "passenger_email": passenger_email,
        "number_of_passengers": passengers,
        "total_price": random.randint(15000, 35000) * passengers,
        "currency": "JPY",
        "booking_date": datetime.now().isoformat(),
        "message": f"Flight successfully booked! Confirmation code: {confirmation_code}",
    }

    return json.dumps(booking, indent=2, ensure_ascii=False)


async def cancel_booking(booking_id: str) -> str:
    """Cancel a flight booking.

    Args:
        booking_id: The booking ID to cancel.

    Returns:
        JSON string with cancellation confirmation.
    """
    import json

    result = {
        "status": "cancelled",
        "booking_id": booking_id,
        "cancellation_date": datetime.now().isoformat(),
        "refund_amount": random.randint(15000, 35000),
        "currency": "JPY",
        "message": "Booking successfully cancelled. Refund will be processed within 7-10 business days.",
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='airline_booking_agent',
    description=(
        'Airline booking agent that can search for flights and make reservations. '
        'Provides flight search, booking, and cancellation services.'
    ),
    instruction="""
You are an airline booking assistant. You MUST use the available tools to complete tasks.

Your capabilities:
1. Search for available flights between cities - ALWAYS use the search_flights tool
2. Book flights for passengers - ALWAYS use the book_flight tool
3. Cancel existing bookings - ALWAYS use the cancel_booking tool

IMPORTANT INSTRUCTIONS:
- When a user requests flight information, IMMEDIATELY call the search_flights tool
- Do NOT ask for more information if you already have: departure, destination, and departure_date
- Use reasonable defaults: passengers=1, return_date="" (empty string for one-way) if not specified
- ALWAYS call the appropriate tool first, then present the results to the user
- Do NOT make up flight information - ALWAYS use the tool results

Example: If user says "Search flights from Tokyo to Osaka on 2024-12-20 for 2 passengers"
→ Immediately call: search_flights(departure="Tokyo", destination="Osaka", departure_date="2024-12-20", passengers=2)

When you receive structured data with parameters, immediately use them to call the tool.
""",
    tools=[
        search_flights,
        book_flight,
        cancel_booking,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ],
    ),
)
